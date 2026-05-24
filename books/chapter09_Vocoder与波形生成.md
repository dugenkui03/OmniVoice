# 第九章：Vocoder 与波形生成 —— 从声学表示还原到声音

第八章讲的是声学模型如何把文本、发音结构、说话人和风格条件变成声学表示。本章接着讲下一段：这些声学表示如何变成最终可以播放的 waveform（波形）。

在完整 TTS（Text-to-Speech，文本转语音）系统里，vocoder / decoder 不是声学模型的小助手，而是下游的波形还原模块。它不负责决定“这句话应该读成什么内容”，但会强烈影响最终声音的清晰度、真实感、细节、毛刺、爆音、采样率和推理延迟。

读这一章时，最重要的不是先背 vocoder 名字，而是先分清三层：主生成对象是什么，中间是否还有声学细节生成，最终由哪个模块还原 waveform。9.1 会用定位图展开这三层。

本章重点建立四种工程判断能力：

```text
第一，知道 vocoder / decoder 在 TTS 系统里的位置。
第二，知道它主要影响最终结果的哪些部分。
第三，能按输入输出理解不同波形还原方案。
第四，知道模型训练完成后，推理时哪些地方还能干预，哪些地方不能硬改。
```

## 本章目录

| 章节 | 主题 | 解决的问题 |
| --- | --- | --- |
| 9.1 | Vocoder / decoder 在完整 TTS 方案中的位置 | 它和声学模型、文本前端、后处理如何分工 |
| 9.2 | 这个模块影响最终结果的哪些部分 | 哪些听感问题更像波形还原问题 |
| 9.3 | 从声学表示到 waveform：还原为什么难 | mel、latent、token 到波形并非简单格式转换 |
| 9.4 | 按主生成对象拆解波形还原链路 | mel vocoder、codec decoder、AudioVAE decoder 和组合链路如何工作 |
| 9.5 | 主流系统里的选型对照 | OmniVoice、IndexTTS2、VoxCPM2、CosyVoice3 的输入输出怎么读 |
| 9.6 | 工程实践提示：选型、推理期干预与排错 | 如何选型、哪些能调、如何把内容问题和音质问题分开 |
| 9.7 | 本章小结 | 本章应该记住哪些判断 |

## 9.1 Vocoder / decoder 在完整 TTS 方案中的位置

```mermaid
flowchart LR
    subgraph inputStage["前端表示 / 条件输入"]
        direction TB
        A["文本 / 音素 / 拼音<br/>说什么、怎么发音"]
        B["参考音频 / 风格 / 情绪<br/>谁在说、怎么说"]
    end

    subgraph generatorStage["主生成对象"]
        direction TB
        C["semantic / speech token<br/>高层语音结构"]
        D["mel-spectrogram<br/>频谱声学表示"]
        E["codec token<br/>可被 codec decoder 还原"]
        F["continuous latent<br/>可被 latent decoder 还原"]
    end

    subgraph detailStage["声学细节生成 / 可选"]
        direction TB
        G["S2M / flow matching / acoustic generator<br/>semantic token -> mel / acoustic feature"]
        H["无额外转换<br/>直接进入还原模块"]
    end

    subgraph restoreStage["本章展开：波形还原"]
        direction TB
        I["neural vocoder<br/>HiFi-GAN / BigVGAN / HIFT 等"]
        J["codec decoder<br/>audio tokenizer decode"]
        K["AudioVAE / latent decoder"]
    end

    subgraph outputStage["最终输出"]
        direction TB
        L["waveform<br/>可播放音频"]
        M["后处理<br/>响度 / 拼接 / 格式"]
    end

    A --> C
    A --> D
    A --> E
    A --> F
    B --> C
    B --> D
    B --> E
    B --> F
    C --> G --> I
    D --> I
    E --> H --> J
    F --> H --> K
    I --> L
    J --> L
    K --> L
    L --> M

    classDef normal fill:#F0EEFF,stroke:#9B8CFF,color:#2D2A3A;
    classDef focus fill:#FFF2CC,stroke:#D88B00,stroke-width:3px,color:#3B2A00;
    classDef output fill:#EEF8F3,stroke:#5CA982,color:#173B2B;
    class A,B,C,D,E,F,G,H,M normal;
    class I,J,K focus;
    class L output;
    style inputStage fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:6 4,color:#5A5134;
    style generatorStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
    style detailStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#173B2B;
    style restoreStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#5A5134;
    style outputStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#5A5134;
```

声学模型输出的 mel、latent 或 codec token 不是最终音频文件。它们更像“声音中间表示”，还需要一个还原模块把它们变成 waveform。semantic / speech token 更靠近高层语音结构，通常还要先经过 S2M、flow matching 或 acoustic generator，才能进入真正的波形还原模块。

这里最容易混淆的是 decoder 这个词。TTS 里至少有两种 decoder：

| 叫法 | 所在位置 | 输入 | 输出 | 本章是否重点讨论 |
| --- | --- | --- | --- | --- |
| acoustic decoder / generator | 声学模型内部 | hidden representation、条件、对齐信息 | mel / latent / codec token | 第八章重点 |
| vocoder / codec decoder / AudioVAE decoder | 声学模型之后 | mel / latent / codec representation | waveform | 本章重点 |

因此，本章说的 decoder 主要指第二种：把声学表示还原成最终波形的音频解码器。它和第八章里的 acoustic decoder 不是同一个层级。

完整链路中，常见模块分工如下：

| 模块 | 主要负责 | 典型输入 | 典型输出 | 层级提醒 |
| --- | --- | --- | --- | --- |
| 文本前端 / G2P | 文字规范化、音素、拼音、声调、多音字 | 原始文本 | 发音结构或文本 token | 前端输入层 |
| 主生成模型 / speech generator | 生成主目标表示 | 文本、音素、参考音频、条件 | semantic token / mel / latent / codec token | 声学生成层 |
| 声学细节生成 / acoustic generator | 把高层 token 落到更细的声学表示 | semantic / speech token、prompt 条件 | mel、acoustic feature、continuous representation | 可选中间层 |
| vocoder / codec decoder / AudioVAE decoder | 把可还原表示变成波形 | mel / codec token / continuous latent | waveform | 本章核心层 |
| 后处理 | 让音频适合交付和播放 | waveform | wav / mp3 / stream | 工程交付层 |

工程上可以把第九章这段理解成：

```text
主生成模型决定“先生成哪一种声音中间对象”。
声学细节生成模块决定“高层 token 如何落到可还原的声学表示”。
vocoder / decoder 决定“这个表示还原成波形后听起来有多干净、真实、稳定”。
```

现实系统里二者会互相影响。声学模型输出已经错了，vocoder 很难救回读错、漏读和韵律错乱；vocoder 或 decoder 泛化差，也可能把正常的中间表示还原成有毛刺、金属感或高频缺失的声音。

## 9.2 这个模块影响最终结果的哪些部分

vocoder / decoder 对最终结果影响很大，但它影响的维度和声学模型不一样。

| 听感维度 | 更像谁负责 | 工程解释 |
| --- | --- | --- |
| 内容是否读对 | 文本前端、对齐、声学模型 | 波形还原模块通常不理解文本内容 |
| 漏读、重复、跳字 | 声学模型、alignment、duration | 中间表示本身的时序已经有问题 |
| 语速、停顿、节奏 | duration、韵律建模、推理切句 | vocoder 会还原节奏，但通常不主动决定文本单位时长 |
| 音色是否像 | prompt speech、speaker 表征、声学模型、decoder | 音色线索需要被生成模型使用，也要被 decoder 保住 |
| 清晰度和细节 | vocoder / decoder、声学表示质量 | 高频、瞬态、齿音、气声等细节很依赖还原模块 |
| 毛刺、爆音、金属感 | vocoder / decoder、采样率、音频预处理 | 常见于输入分布不匹配、采样率配置不一致或 decoder 泛化差 |
| 发闷、带宽窄 | vocoder / decoder、mel 参数、训练音频带宽 | 高频建模不足或目标采样率偏低时更明显 |
| 延迟和吞吐 | vocoder / decoder、采样步数、部署后端 | 波形还原可能成为服务延迟的关键部分 |

一个实用判断是：

```text
如果“说的内容、停顿、节奏”已经错了，优先看文本前端、对齐和声学模型。
如果“内容基本对，但声音糊、炸、刺、金属、发闷”，优先看 vocoder / decoder、采样率和音频参数。
```

这也是为什么不能说第九章只是“方案选型，影响不大”。它不一定决定语义内容，但会直接决定最终音频是否可听、耐听、稳定、适合上线。

## 9.3 从声学表示到 waveform：还原为什么难

waveform 是一串非常密集的采样点。以 24 kHz 音频为例，1 秒声音就有 24000 个采样点。声学模型通常不会直接生成这么长的采样点序列，而是先生成更短、更好建模的中间表示。

不同声学表示丢失或压缩的信息并不一样。mel-spectrogram（梅尔频谱）描述“每个时间片有哪些频率能量”，但它不是音频文件，缺少 phase（相位）、高频细节、微小瞬态和采样点级别的周期结构。codec token 是离散索引，需要回到 codec 自己学到的 codebook 和 decoder 空间。continuous latent 是连续压缩坐标，需要由配套的 latent decoder 上采样并还原细节。

```mermaid
flowchart LR
    subgraph melStage["mel 路线"]
        direction LR
        A["mel-spectrogram<br/>频率能量表示"] --> B["GAN vocoder<br/>补相位 / 高频 / 周期结构"]
        B --> C["waveform"]
    end

    subgraph codecStage["codec token 路线"]
        direction LR
        D["codec token<br/>离散 codebook 索引"] --> E["codec decoder<br/>按 codec 压缩空间重建"]
        E --> F["waveform"]
    end

    subgraph latentStage["continuous latent 路线"]
        direction LR
        G["continuous latent<br/>连续压缩表示"] --> H["AudioVAE / latent decoder<br/>上采样并还原细节"]
        H --> I["waveform"]
    end

    classDef focus fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef output fill:#EEF8F3,stroke:#5CA982,color:#173B2B;
    class B,E,H focus;
    class C,F,I output;
    style melStage fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:6 4,color:#5A5134;
    style codecStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
    style latentStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#173B2B;
```

传统算法也可以尝试从 spectrogram（频谱图）的幅度信息估计 waveform，但质量有限。这个现象揭示了一个核心事实：

```text
对 mel / spectrogram 路线来说，只有频谱幅度还不够，波形还原还需要恢复相位和细节。
```

现代 neural vocoder（神经声码器）和 neural audio decoder（神经音频解码器）的作用，就是从大量真实音频里学习“什么样的波形听起来像真实语音”。它们不是简单格式转换器，而是一个学习出来的生成模块。

## 9.4 按主生成对象拆解波形还原链路

本书关注的不是完整 vocoder 历史年表，而是读懂 OmniVoice、IndexTTS2、VoxCPM2、CosyVoice3 这类系统时最常遇到的波形还原方案。早期算法式重建、逐采样点自回归 vocoder 等历史路线有背景价值，但不是本节主线。

当前更值得关注的是下面四类链路。这里故意把“主生成对象”“声学细节生成”和“最终波形还原”分开，因为它们不是同一层级：

| 链路类型 | 主生成对象 | 声学细节生成 / 中间转换 | 最终波形还原 | 典型对应系统 |
| --- | --- | --- | --- | --- |
| mel -> GAN vocoder | mel-spectrogram | 通常不需要额外转换 | GAN vocoder | IndexTTS / IndexTTS2、许多混合路线 |
| codec token -> codec decoder | codec token / acoustic token | 通常直接进入 codec decoder | codec decoder | OmniVoice、部分 Speech LM 路线 |
| continuous latent -> AudioVAE decoder | continuous speech latent | latent 由上游生成或修正 | AudioVAE / latent decoder | VoxCPM2、连续表示路线 |
| speech token -> flow/S2M -> vocoder | semantic / speech token | flow matching、S2M 或 acoustic generator 生成 mel / acoustic feature | vocoder / decoder | CosyVoice / CosyVoice3 类级联系统 |

这里没有把 diffusion vocoder 单独列成本章主线。需要区分两个概念：diffusion vocoder 指在 waveform 空间逐步去噪生成最终波形；diffusion / flow matching acoustic generator 指在 mel、latent 或与 token 条件相关的声学表示空间里生成中间表示。前者不是本书关注模型族里的主要波形还原方案，后者会在第十章作为新一代 TTS 生成范式重点展开。

### 9.4.1 mel -> GAN vocoder：把梅尔频谱补成波形

mel -> GAN vocoder 是工程中很常见的成熟路线。声学模型先生成 mel-spectrogram，vocoder 再把 mel 还原成 waveform。

```mermaid
flowchart LR
    A["文本 / 参考音频 / 条件"] --> B["声学模型"]
    B --> C["mel-spectrogram<br/>梅尔频谱"]
    C --> D["GAN vocoder<br/>HiFi-GAN / BigVGAN 类"]
    D --> E["waveform"]
```

基本思路是：mel 已经描述了每个时间片的频率能量，但缺少相位、高频和采样点级细节。GAN vocoder 的 generator（生成器）学习从 mel 生成波形，discriminator（判别器）学习判断生成波形像不像真实录音。

```mermaid
flowchart LR
    A["mel-spectrogram<br/>梅尔频谱"] --> B["generator<br/>生成波形"]
    B --> C["generated waveform"]
    D["real waveform<br/>真实录音"] --> E["discriminator<br/>判断真假"]
    C --> E
    E --> F["loss<br/>推动生成器更像真实语音"]
```

常见训练目标包括：

| loss | 极简解释 |
| --- | --- |
| adversarial loss（对抗损失） | 让生成波形更像真实录音 |
| feature matching loss（特征匹配损失） | 让判别器中间特征也接近真实音频 |
| mel reconstruction loss（梅尔重建损失） | 让生成音频再提 mel 后接近目标 mel |

这条路线的工程优点是快、成熟、容易单独替换和调试。需要小心的是 mel 配置必须匹配，包括 `sample_rate`、`n_fft`、`hop_length`、`n_mels`、`fmin`、`fmax` 等。mel 参数不匹配时，vocoder 很容易发闷、爆音或出现金属感。

### 9.4.2 codec token -> codec decoder：把语音 token 还原成声音

codec token 路线不是先生成 mel，而是把语音压缩成离散 token。完整 codec 通常包含 encoder 和 decoder 两端：

```text
waveform -> codec encoder -> codec token -> codec decoder -> waveform
```

TTS 推理时，主模型通常只负责生成目标 codec token，codec decoder 再把这些 token 还原成 waveform。

```mermaid
flowchart LR
    A["文本 token"] --> C["Speech LM / 主生成模型"]
    B["prompt audio token<br/>参考音频 token"] --> C
    C --> D["target codec token"]
    D --> E["codec decoder"]
    E --> F["waveform"]
```

这条路线适合 speech LM（语音语言模型）和 zero-shot voice cloning（零样本声音克隆），因为文本 token、参考音频 token 和目标音频 token 可以更自然地组织在同一套序列建模框架里。

它的关键边界是：codec token 空间和 codec decoder 强绑定。不同 tokenizer、codebook 数量、token rate、采样率或 decoder 版本不匹配，通常不能随便互换。OmniVoice 就属于这类需要关注 audio tokenizer / codec decoder 的路线。

### 9.4.3 continuous latent -> AudioVAE decoder：把连续潜变量还原成高采样率音频

continuous latent 路线不把语音主要表示成离散 token，而是使用连续潜变量。主生成模型在连续空间里生成或修正语音表示，后面的 AudioVAE / latent decoder 再把连续表示还原成 waveform。这里关注的是 latent decoder 这一步，不是 waveform 空间的 diffusion vocoder。

```mermaid
flowchart LR
    A["文本 / prompt / 控制条件"] --> B["flow / diffusion / autoregressive generator"]
    B --> C["continuous speech latent"]
    C --> D["AudioVAE / latent decoder"]
    D --> E["waveform"]
```

这条路线的直觉是：连续表示可能保留更多细腻声学变化，例如情绪、气声、节奏和局部音色细节；同时又比直接生成原始 waveform 更短、更适合模型处理。

它的工程边界也很明确：latent decoder 必须和 latent space 成套使用。VoxCPM2 这类连续表示路线的重点，不是“把 token 喂给 codec decoder”，而是“生成连续语音表示，再通过 AudioVAE V2 这类 decoder 输出音频”。

### 9.4.4 speech token -> flow/S2M -> vocoder：分层生成再还原

还有一类当前很常见的强系统，不是只选一种表示，而是分层组合。它的关键是：speech / semantic token 通常不是最终 vocoder 的直接输入，flow matching、S2M 或 acoustic generator 还要把它转换成更细的声学表示。

```text
高层 speech / semantic token 负责内容、语义和粗粒度韵律。
flow matching / S2M / acoustic generator 负责生成更细的声学表示。
vocoder / decoder 负责把声学表示还原成 waveform。
```

```mermaid
flowchart LR
    subgraph highStage["主生成对象"]
        direction TB
        A["文本 / prompt speech / 指令"] --> B["LLM / T2S<br/>生成 semantic / speech token"]
    end

    subgraph acousticStage["声学细节生成"]
        direction TB
        C["flow matching / S2M / acoustic generator"]
        D["mel / acoustic feature / continuous representation"]
    end

    subgraph restoreStage["最终波形还原"]
        direction TB
        E["vocoder / decoder"]
        F["waveform"]
    end

    B --> C --> D --> E --> F

    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef output fill:#EEF8F3,stroke:#5CA982,color:#173B2B;
    class C,E core;
    class F output;
    style highStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
    style acousticStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#173B2B;
    style restoreStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
```

CosyVoice / CosyVoice3 这类系统适合用这种分层视角理解。LLM 或 T2S 模块负责高层 token 建模，flow matching 或声学生成器负责把 token 变成更细的声学表示，最后再由 vocoder / decoder 输出波形。

这类方案的优势是能力强、扩展性好，适合 zero-shot、多语言、情绪、流式等复杂场景。代价是模块更多，排错时需要区分：问题出在 LLM token 生成、flow 声学生成，还是最后的 vocoder / decoder。

## 9.5 主流系统里的选型对照

当前主流 TTS 系统通常不是单一路线，而是把 LLM、speech token、flow matching、vocoder、codec decoder、AudioVAE 和推理调度组合起来。读方案时不要只问“用了哪个模型”，而要问：

```text
主模型生成什么？
中间是否还有声学细节生成？
最终还原模块输入是什么？
还原模块输出什么？
这个选择带来什么工程优势和限制？
```

下面按本书关注的模型族做模块级对照。

| 系统 / 路线 | 主生成对象 | 声学细节生成 / 中间转换 | 最终波形还原模块 | 还原模块输入 | 还原模块输出 | 工程理解 |
| --- | --- | --- | --- | --- | --- | --- |
| OmniVoice | 多 codebook audio / acoustic token | 通常直接进入 audio tokenizer decode | audio tokenizer decode / codec decoder | codec token | 24 kHz waveform | 主模型在 token 空间生成声音表示，音频 tokenizer 的 decoder 负责把 token 变回波形 |
| IndexTTS2 | 自回归 semantic token | S2M 生成 mel-spectrogram，必要时结合 GPT latent 等稳定声学细节 | BigVGANv2 类 vocoder | mel / acoustic feature | waveform | 强调 duration control、情绪和音色解耦，最终由 vocoder 把 mel 类表示落到波形 |
| VoxCPM2 | continuous speech representation | Local DiT / CFM 等上游模块生成或修正 continuous latent | AudioVAE V2 / latent decoder | continuous latent | 48 kHz waveform | 不以离散 audio token 为主，重点是连续表示、AudioVAE 解码和高采样率输出 |
| CosyVoice3 / CosyVoice 系列 | LLM 生成 speech / semantic token | flow matching / acoustic generator 把 token 转成更细声学表示 | vocoder / decoder | mel、acoustic feature 或连续声学表示 | waveform | 体现“LLM token -> flow/acoustic generator -> vocoder”的级联式趋势 |

这张表不是为了给模型排名，而是帮助定位“第九章这段”在不同系统里的形态：

```mermaid
flowchart LR
    subgraph mainStage["主生成对象"]
        direction TB
        A["OmniVoice<br/>codec token"]
        B["IndexTTS2<br/>semantic token"]
        C["VoxCPM2<br/>continuous latent 路线"]
        D["CosyVoice3 / CosyVoice<br/>speech / semantic token"]
    end

    subgraph acousticStage["声学细节生成 / 中间转换"]
        direction TB
        E["直接进入 codec decoder"]
        F["S2M<br/>semantic token -> mel"]
        G["Local DiT / CFM<br/>生成或修正 latent"]
        H["flow matching / acoustic generator<br/>token -> acoustic feature"]
    end

    subgraph restoreStage["最终波形还原"]
        direction TB
        I["codec decoder"]
        J["BigVGANv2 类 vocoder"]
        K["AudioVAE V2 decoder"]
        L["vocoder / decoder"]
    end

    A --> E --> I
    B --> F --> J
    C --> G --> K
    D --> H --> L
    I --> M["waveform"]
    J --> M
    K --> M
    L --> M

    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef output fill:#EEF8F3,stroke:#5CA982,color:#173B2B;
    class I,J,K,L core;
    class M output;
    style mainStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
    style acousticStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#173B2B;
    style restoreStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
```

## 9.6 工程实践提示：选型、推理期干预与排错

前面几节回答“有哪些模块”和“不同系统怎么落地”。工程实践还要再问三件事：选型时看什么，推理时哪些能调，出问题时先查哪一层。

### 9.6.1 选型时真正要看的问题

读一个 TTS 方案时，波形还原部分可以按下面几个问题拆：

| 问题 | 为什么重要 |
| --- | --- |
| 主生成对象是什么 | 决定它是 mel、codec token、continuous latent，还是还要继续转换的 semantic token |
| 中间是否还有声学细节生成 | 避免把 flow matching / S2M 误当成最终 vocoder |
| 最终还原模块输入是什么 | 输入不兼容时，decoder 再强也不能正常工作 |
| 输出采样率是多少 | 直接影响带宽、清晰度、文件大小和部署成本 |
| 是否支持流式 | 影响首包延迟和实时服务体验 |
| 能否单独替换 | 影响工程调优和系统升级空间 |
| 坏音质如何排查 | 决定要看 mel 参数、codec token、latent 分布还是后处理 |

不同选型的输入输出差异，会直接影响工程能力：

| 选型 | 优势 | 需要小心 |
| --- | --- | --- |
| mel -> GAN vocoder | 成熟、快、好排查 | mel 参数和采样率必须匹配，mel 本身压缩了相位和部分高频细节 |
| codec token -> codec decoder | 适合 speech LM、prompt speech、离散 token 建模 | codec token 空间绑定 decoder，codec 质量限制最终上限 |
| continuous latent -> AudioVAE decoder | 保留连续细节，适合 flow / diffusion 类生成 | latent 空间和 decoder 强绑定，采样和部署复杂度更高 |
| speech token -> flow/S2M -> vocoder | 能把高层语义和声学细节分层建模 | 模块多，归因时要分清 token 生成、声学细节生成和最终波形还原 |

可以把选型判断压缩成一句话：

```text
先看主模型生成什么，再看中间是否还要转换，最后看波形还原模块接收什么输入；输入空间不兼容，所谓“换一个更强 decoder”通常不会生效。
```

### 9.6.2 推理期哪些环节还能干预

模型训练完成后，权重通常不会在普通推理中更新。对第九章来说，工程可控性要按模块层级看：上游输入会改变生成内容和声学表示，采样会改变生成分布，decoder 配置会影响波形还原，后处理只负责交付形态。

```mermaid
flowchart LR
    subgraph inputStage["输入构造"]
        A["文本 / 参考音频 / 控制条件"] --> B["文本前端 / G2P / 切句"]
    end

    subgraph generationStage["声学生成"]
        C["采样参数 / duration / speed"]
    end

    subgraph restoreStage["波形还原"]
        D["vocoder / decoder<br/>兼容前提下调整或替换"]
    end

    subgraph postStage["后处理"]
        E["响度 / 拼接 / 格式"]
    end

    B --> C --> D --> E

    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef output fill:#EEF8F3,stroke:#5CA982,color:#173B2B;
    class D core;
    class E output;
    style inputStage fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:6 4,color:#5A5134;
    style generationStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
    style restoreStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style postStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#173B2B;
```

常见可干预点如下：

| 环节 | 推理时能做什么 | 边界 |
| --- | --- | --- |
| 文本规范化 | 把数字、日期、缩写写成更稳定的读法 | 只能影响输入发音结构，不能让模型学会训练中没有的发音规律 |
| G2P / 拼音 / 音素 | 指定多音字读法、外语发音、中文声调 | 需要模型训练时见过或能理解对应符号 |
| 切句和停顿 | 控制长文本分块、标点、静音间隔 | 切得太碎会破坏上下文和韵律连续性 |
| prompt speech / speaker 条件 | 换参考音频、参考文本、音色描述 | 参考音频质量差会污染音色和风格 |
| duration / speed | 控制整体时长、语速或部分节奏 | 过度拉伸会导致不自然、吞字或拖音 |
| sampling steps / temperature / guidance | 权衡稳定性、多样性、条件遵从和速度 | 参数只能改变生成分布，不能补出模型完全没学到的能力 |
| vocoder / decoder | 在兼容前提下换更强的 vocoder，或调整解码配置 | 输入表示、采样率、mel 参数、codec token 空间必须兼容 |
| 后处理 | 响度归一化、去静音、淡入淡出、格式转换 | 后处理不能修复已经读错或已经严重失真的生成内容 |

工程上可以归纳为：

```text
文本到音素、拼音、多音字、切句、参考音频、duration、speed、采样参数、音频后处理，通常都是推理期比较灵活的干预点。
vocoder / decoder 也可能是推理期可替换或可调的部分，但必须满足表示空间和配置兼容。
已经训练好的主模型权重和 decoder 权重本身，普通推理时通常不更新；如果能力缺失，往往需要重新训练、微调或换模型方案。
```

对第九章来说，最重要的工程提示是：不要把“可干预”理解成“可以随便改”。TTS 的每个模块都有自己的输入分布，推理期干预要尽量沿着模型训练时见过的接口走。

### 9.6.3 工程排错：把内容问题和音质问题分开

排错时可以先按听感现象区分问题来源。

| 听感问题 | 更可能优先排查 |
| --- | --- |
| 字读错、漏读、重复 | 文本前端、G2P、alignment、声学模型 |
| 语速奇怪、停顿奇怪 | duration、韵律预测、切句策略 |
| 音色不像 | prompt speech、speaker 表征、声学模型、decoder 是否保住音色细节 |
| 毛刺、爆音、金属感 | vocoder / decoder、采样率、音频预处理 |
| 声音闷、细节少 | vocoder 能力、mel 质量、高频建模、目标采样率 |
| 音量忽大忽小 | loudness（响度）归一化、能量分布、后处理 |
| 流式首包慢 | 主模型生成速度、vocoder / decoder 速度、服务调度 |

vocoder 排查中最常见的硬条件，是确认训练和推理的音频参数一致。

```text
sample_rate
n_fft
hop_length
win_length
n_mels
fmin
fmax
codec codebook 配置
latent 维度和帧率
```

这些参数不一致，会导致输入分布偏移。输入分布偏了，即使模型本身质量很好，也可能出现发闷、金属感、爆音或细节缺失。

如果系统能导出中间表示，排查会更清晰：

```mermaid
flowchart LR
    A["文本前端输出"] --> B["声学表示<br/>mel / latent / token"]
    B --> C["波形还原输出<br/>waveform"]
    C --> D["后处理结果"]
    B -.-> E["声学表示已异常<br/>优先查文本前端 / 对齐 / 声学模型"]
    C -.-> F["声学表示正常但 waveform 异常<br/>优先查 vocoder / decoder / 采样率"]
    D -.-> G["waveform 正常但交付结果异常<br/>优先查后处理 / 拼接 / 格式"]
```

## 9.7 本章小结

本章核心结论：

```text
vocoder / decoder 是完整 TTS 链路里的波形还原模块，不是声学模型内部的小助手。
它主要影响音质、细节、真实感、毛刺、爆音、带宽、延迟和稳定性。
mel、codec token、continuous latent 是不同声学表示，对应不同还原模块。
semantic / speech token、flow matching、vocoder 不是同一层级：token 偏高层结构，flow / S2M 偏声学细节生成，vocoder / decoder 偏最终波形还原。
OmniVoice、IndexTTS2、VoxCPM2、CosyVoice3 的差异，可以从“主生成对象 -> 中间声学生成 -> 最终还原模块输入输出”来读。
训练完成后的推理干预主要发生在输入构造、发音控制、采样参数、解码配置和后处理上。
排错时要把“内容是否说对”和“波形是否还原好”分开看。
```

下一章进入 diffusion（扩散模型）与 flow matching（流匹配）：它们不仅可以用于 waveform 空间的 vocoder，也可以作为 acoustic model / speech generator 在 mel、latent 或 codec token 空间里的生成范式。
