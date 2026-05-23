# 第八章：声学模型 Acoustic Model —— 从发音条件到声学表示

声学模型（acoustic model）负责把文本、音素和条件信息转换成声学表示。它是 TTS（Text-to-Speech，文本转语音）里最核心的“中间生成器”之一。

可以先记住一句话：

```text
声学模型决定“谁在说、说什么、说话方式、发音结构”如何变成 mel / latent / codec token。
```

这里的输出还不是最终音频。最终 waveform（波形）通常还需要 vocoder（声码器）或 codec decoder（编码解码器的解码端）还原。

![声学模型结构图](./images/chapter08_声学模型_结构图.svg)

## 本章导图

```mermaid
flowchart LR
    A["speaker condition<br/>谁在说"] --> B["encoder（编码器）"]
    C["文本 / 音素<br/>说什么、发音结构"] --> B
    D["style / emotion condition<br/>说话方式"] --> B
    B --> E["alignment / duration<br/>对齐、时长"]
    E --> F["decoder（解码器）"]
    G["pitch / energy<br/>音高、能量"] --> F
    F --> H["mel / latent / codec token<br/>声学表示"]
    H --> I["vocoder / decoder<br/>还原波形"]
```

这张图是工程视角，不是所有模型都完全长这样。Tacotron 类模型更依赖 attention（注意力）隐式对齐，FastSpeech 类模型更依赖 duration（时长）显式展开，VITS / diffusion / flow 类模型还会引入 latent（潜变量）或 noise（噪声）。

## 8.1 声学模型到底输入什么、输出什么

声学模型的输入通常不是原始文本，而是文本前端处理后的结构化条件。

常见输入：

| 输入 | 极简解释 |
| --- | --- |
| character（字符） | 原始或规范化后的字符 |
| phoneme（音素） | 更接近发音的最小单位 |
| pinyin（拼音） | 中文 TTS 常用发音表示 |
| tone（声调） | 中文音高类别信息 |
| stress（重音） | 英文常见重音信息 |
| speaker embedding（说话人向量） | 控制谁在说 |
| style embedding（风格向量） | 控制说话方式 |
| emotion label（情绪标签） | 控制情绪类别 |
| prompt speech（提示语音） | 从参考音频中提取音色和风格 |

常见输出：

| 输出 | 极简解释 | 后续模块 |
| --- | --- | --- |
| mel-spectrogram（梅尔频谱） | TTS 最常见中间声学表示 | vocoder |
| latent（潜变量） | 压缩后的连续声学表示 | decoder / vocoder |
| codec token（语音编码 token） | neural codec 产生的离散或连续表示 | codec decoder |

工程上可以把声学模型看成一个“从结构化请求生成中间结果”的服务：

```text
入参：说话人条件 + 文本内容 / 发音结构 + 说话方式条件
出参：一段声学表示
```

如果入参里的发音、时长、说话人、风格本身就错了，后面的 vocoder 很难补救。

## 8.2 权重、向量和矩阵：模型到底在算什么

第六章讲了 speaker embedding（说话人向量）、style embedding（风格向量）、prosody embedding（韵律向量）这些概念。到了第八章，可以进一步问一个更工程的问题：

```text
这些向量进入模型后，到底发生了什么计算？
```

最简化的答案是：

```text
输入 = 向量
权重 = 很多矩阵 / 张量
模型计算 = 用这些矩阵和张量不断变换输入向量
输出 = 新的向量序列，最后再解释成 mel / latent / codec token
```

### 一层神经网络的最小直觉

一层最简单的神经网络可以写成：

```text
y = W x + b
```

| 符号 | 含义 | 工程直觉 |
| --- | --- | --- |
| `x` | 输入向量 | 文本、音素、说话人、风格等条件变成的数字表示 |
| `W` | 权重矩阵 | 模型训练出来的参数，决定如何变换输入 |
| `b` | 偏置向量 | 给输出增加可学习的平移量 |
| `y` | 输出向量 | 下一层要继续处理的 hidden representation |

```mermaid
flowchart LR
    A["输入向量 x"] --> B["权重矩阵 W"]
    B --> C["线性变换<br/>W x"]
    D["偏置向量 b"] --> E["相加"]
    C --> E
    E --> F["非线性函数<br/>GELU / ReLU / SiLU"]
    F --> G["输出向量 y"]
```

真实模型不是只做一次 `W x`，而是把这种向量变换堆很多层，并加入 attention（注意力）、normalization（归一化）、residual connection（残差连接）等结构。

### 模型权重不是一个矩阵，而是一组张量

说“权重是矩阵”适合作为第一层直觉，但真实大模型的权重更像一个很大的参数仓库：

| 权重类型 | 常见形态 | 作用 |
| --- | --- | --- |
| embedding matrix（嵌入矩阵） | 二维矩阵 | 把 token id 查成向量 |
| attention Q/K/V matrix | 二维矩阵 | 让每个位置读取上下文 |
| MLP projection matrix | 二维矩阵 | 对 hidden vector 做非线性变换 |
| LayerNorm 参数 | 向量 | 稳定每层数值分布 |
| output head matrix | 二维矩阵 | 把 hidden vector 投影成输出概率 |
| codec / audio head | 多个矩阵 | 把 hidden vector 解释成音频 token 概率 |

所以更准确地说：

```text
模型权重 = 很多层、很多个矩阵 / 向量 / 高维张量的集合。
```

这些权重是训练过程学出来的。训练时，模型不断比较预测结果和真实目标之间的差距，然后用反向传播调整这些矩阵里的数字。

### 输入条件如何变成模型能计算的向量

声学模型通常不会直接拿“文字字符串”或“参考音频文件”做矩阵乘法，而是先把它们变成向量：

```mermaid
flowchart LR
    A["文本 / 音素 id"] --> B["text embedding matrix"]
    B --> C["文本向量序列"]
    D["说话人 id / 参考音频"] --> E["speaker encoder<br/>或 speaker embedding table"]
    E --> F["说话人向量"]
    G["风格 / 情绪条件"] --> H["style embedding"]
    H --> I["风格向量"]
    C --> J["声学模型"]
    F --> J
    I --> J
    J --> K["mel / latent / codec token"]
```

这个过程可以类比成后端服务把不同字段整理成统一的数据结构：

```text
text field -> text vector
speaker field -> speaker vector
style field -> style vector
然后一起送进模型计算。
```

### speaker embedding 是输入条件，不是模型权重本身

这里最容易混淆的是 speaker embedding（说话人向量）和模型权重。

| 概念 | 是什么 | 会不会随每次请求变化 |
| --- | --- | --- |
| 模型权重 | 训练好的矩阵 / 张量参数 | 通常不变 |
| speaker embedding | 描述“谁在说”的条件向量 | 会随说话人或参考音频变化 |
| text embedding | 描述“说什么”的输入向量 | 会随文本变化 |
| style embedding | 描述“说话方式 / 韵律风格”的条件向量 | 会随风格、情绪或提示变化 |

可以这样理解：

```mermaid
flowchart LR
    A["模型权重<br/>会说话的计算规则"] --> D["TTS 推理"]
    B["文本向量<br/>说什么"] --> D
    C["speaker embedding<br/>谁在说"] --> D
    E["style / emotion vector<br/>说话方式"] --> D
    D --> F["目标语音表示"]
```

模型权重像一套已经学好的“发声计算系统”；speaker embedding 是本次请求告诉系统“要像谁说话”的条件。它不是单独决定一切，但会影响模型如何生成音色、音区、发声质感和一部分韵律习惯。

### 说话人向量丰富，就能模拟很多人吗？

大方向可以这样理解：如果训练数据覆盖了足够多说话人，并且模型结构允许说话人信息有效进入生成过程，那么模型会形成一个相对连续的 speaker space（说话人空间）。新的参考音频可以被编码成这个空间里的一个位置，然后模型按这个位置去生成相似音色。

```mermaid
flowchart LR
    A["大量说话人训练数据"] --> B["模型学到 speaker space"]
    C["新参考音频"] --> D["提取 speaker embedding"]
    B --> E["声学模型生成"]
    D --> E
    F["目标文本"] --> E
    E --> G["接近参考说话人的声学表示"]
```

但这不是一个“向量越长就越万能”的问题，而是下面几件事共同决定效果：

| 条件 | 为什么重要 |
| --- | --- |
| 说话人覆盖丰富 | 模型见过足够多音色变化，才能学到稳定的说话人空间 |
| embedding 表达力足够 | 向量要能区分音色、音区、发声方式等差异 |
| 模型真的使用这个条件 | speaker embedding 必须进入 encoder、decoder 或 attention 等关键路径 |
| prompt speech 质量好 | 参考音频太短、太吵或情绪极端，提取出来的向量会偏 |
| vocoder / decoder 足够强 | 最后还原波形时要能保住音色细节 |

所以更准确的说法是：

```text
speaker embedding 给模型提供“谁在说”的坐标；
模型权重决定如何使用这个坐标；
训练数据决定这个坐标空间是否真的有意义；
vocoder / decoder 决定最后声音细节能不能还原出来。
```

这也解释了为什么有些 TTS 模型“音色像了，但语气不像”：speaker embedding 更多控制“谁在说”，而情绪、停顿、语速、表演感还需要 style、prosody、prompt speech 或显式控制条件一起参与。

## 8.3 encoder（编码器）：把输入条件变成模型能用的表示

encoder（编码器）负责把离散输入变成 hidden representation（隐藏表示）。例如：

```text
phoneme id -> embedding -> encoder -> phoneme hidden states
```

你可以把 encoder（编码器）理解成后端服务里的“特征整理层”：它不直接输出音频，而是把输入条件整理成后续 decoder（解码器）更容易使用的表示。

常见 encoder 会处理：

```text
音素顺序
上下文关系
声调 / 重音
说话人条件
语言条件
风格条件
```

在现代模型中，encoder 可能是 RNN（循环神经网络）、CNN（卷积神经网络）、Transformer（自注意力网络）或 DiT（Diffusion Transformer，扩散 Transformer）的一部分。

## 8.4 alignment（对齐）：文本 token 如何对应语音帧

alignment（对齐）是 TTS 中非常关键的问题。

文本通常很短：

```text
我 / 想 / 学 / 习 / TTS
```

mel 帧却很多：

```text
frame 1, frame 2, frame 3, ..., frame 300
```

模型必须知道“哪个文本 token 对应哪些语音帧”。如果对齐错了，就容易出现：

```text
漏读
重复读
跳字
停顿奇怪
长文本崩溃
```

不同模型处理 alignment（对齐）的方式不同：

| 路线 | 对齐方式 | 代表直觉 |
| --- | --- | --- |
| Tacotron 类 | attention（注意力）隐式学习 | 边生成边看文本位置 |
| FastSpeech 类 | duration（时长）显式展开 | 每个音素占几帧先算出来 |
| VITS / Grad-TTS 类 | MAS 等单调对齐 | 学习文本和语音的单调对应 |
| CTC 类方法 | CTC alignment（连接时序分类对齐） | 从序列监督中估计对齐 |

语音对齐通常有一个强假设：发音顺序和文本顺序基本一致。这叫 monotonic alignment（单调对齐）。TTS 大多依赖这个性质。

## 8.5 duration（时长）：最直接的对齐形式

duration（时长）描述每个音素、拼音、字或音节持续多少帧。

例如：

```text
phoneme:  n i h ao
duration: 5 4 6  8
```

这表示第一个音素占 5 帧，第二个音素占 4 帧，以此类推。

FastSpeech 类模型里，duration predictor（时长预测器）和 length regulator（长度调节器）非常关键：

```mermaid
flowchart LR
    A["phoneme hidden<br/>音素隐藏表示"] --> B["duration predictor<br/>预测每个音多久"]
    B --> C["length regulator<br/>按时长复制展开"]
    C --> D["mel decoder<br/>并行生成每一帧"]
```

工程上，duration（时长）一旦出错，现象通常很明显：

| duration 问题 | 可能现象 |
| --- | --- |
| 某些音预测太短 | 吞字、发音不清 |
| 某些音预测太长 | 拖音、节奏慢 |
| 边界预测不稳 | 停顿奇怪 |
| 长文本 duration 累积误差 | 后半句节奏漂移 |

## 8.6 decoder（解码器）：生成声学表示

decoder（解码器）负责根据 encoder 输出和条件信息生成声学表示。

不同模型的 decoder 形态差异很大：

| decoder 类型 | 生成方式 | 典型问题 |
| --- | --- | --- |
| autoregressive decoder（自回归解码器） | 一帧一帧生成 | 慢，容易 exposure bias |
| non-autoregressive decoder（非自回归解码器） | 并行生成所有帧 | 依赖 duration 和对齐质量 |
| diffusion decoder（扩散解码器） | 从噪声逐步去噪 | 推理步数和速度 |
| flow matching decoder（流匹配解码器） | 学习噪声到数据的路径 | 采样器和条件设计 |

Tacotron 类 decoder 通常输出 mel-spectrogram（梅尔频谱）。现代模型也可能输出 latent（潜变量）或 codec token（语音编码 token）。

## 8.7 attention（注意力）：为什么 Tacotron 会漏读和重复

attention（注意力）用于在生成每一帧时选择当前应该关注哪个文本位置。

简化理解：

```text
当前要生成第 120 帧 mel
模型需要知道：这帧大概对应文本里的哪个音素？
attention 给出一个权重分布
```

理想情况下，attention 会从左到右稳定移动：

```text
第 1 个音素 -> 第 2 个音素 -> 第 3 个音素 -> ...
```

如果 attention 停在同一个位置太久，就可能重复读。如果 attention 跳过某些位置，就可能漏读。

所以 Tacotron 类模型虽然自然度高，但长文本和复杂文本场景下需要额外工程保护，例如：

```text
文本切句
attention 约束
重复检测
最大输出长度限制
fallback 机制
```

## 8.8 variance adaptor（变化信息适配器）：控制 pitch、energy、duration

FastSpeech 2 中常见 variance adaptor（变化信息适配器），它把一些影响韵律和表达的特征显式加入模型。

常见变量：

```text
duration（时长）
pitch / F0（音高 / 基频）
energy（能量）
```

这三个变量分别解决：

| 变量 | 控制什么 | 工程直觉 |
| --- | --- | --- |
| duration（时长） | 每个音持续多久 | 语速和节奏 |
| pitch（音高） | F0 走势 | 语调和情绪起伏 |
| energy（能量） | 每帧强弱 | 重音和表达力度 |

它们不是全部韵律信息，但足够建立一个可控入口。很多情绪、风格、语气变化，最终都会在这几个曲线上体现一部分。

## 8.9 Tacotron 类模型：自然但慢，依赖 attention

Tacotron / Tacotron 2 属于 autoregressive TTS（自回归文本转语音）。

结构直觉：

```mermaid
flowchart LR
    A["文本序列"] --> B["encoder<br/>编码器"]
    B --> C["attention<br/>注意力对齐"]
    C --> D["autoregressive decoder<br/>逐帧生成"]
    D --> E["mel-spectrogram<br/>梅尔频谱"]
    E --> F["vocoder<br/>声码器"]
    F --> G["waveform<br/>波形"]
```

核心概念：

| 术语 | 极简解释 |
| --- | --- |
| autoregressive（自回归） | 当前输出依赖之前输出 |
| teacher forcing（教师强制） | 训练时使用真实上一帧辅助学习 |
| exposure bias（暴露偏差） | 训练和推理时输入分布不一致 |
| stop token（停止标记） | 预测什么时候结束生成 |
| attention alignment（注意力对齐） | 让声学帧对应文本位置 |

工程判断：

```text
如果问题是长文本漏读、重复读，先怀疑 attention / 对齐。
如果问题是音质粗糙，可能更接近 vocoder 或声学表示质量问题。
```

## 8.10 FastSpeech 类模型：快、稳、可控

FastSpeech / FastSpeech 2 属于 non-autoregressive TTS（非自回归文本转语音）。它不再逐帧生成，而是通过 duration（时长）把文本隐藏表示展开后并行生成 mel。

核心结构：

```text
phoneme -> encoder -> duration/pitch/energy -> length regulator -> decoder -> mel
```

优势：

| 优势 | 解释 |
| --- | --- |
| 快 | 可以并行生成帧 |
| 稳 | 不依赖逐帧 attention 漂移 |
| 可控 | duration、pitch、energy 可以作为显式控制项 |
| 工程友好 | 更容易定位时长、音高、能量问题 |

代价：

```text
需要更可靠的对齐数据。
需要提取 pitch / energy 等监督特征。
预测结果可能比自回归模型更平均，需要更好的建模或后处理。
```

## 8.11 VITS 类模型：声学模型和 vocoder 边界变模糊

VITS 类模型把 acoustic model（声学模型）和 vocoder（声码器）的边界变得不那么清晰。它不一定先显式生成 mel，再用独立 vocoder 还原波形，而是通过 latent（潜变量）和 generator（生成器）端到端生成语音。

关键组件：

```text
posterior encoder（后验编码器）
prior encoder（先验编码器）
latent z（潜变量）
normalizing flow（标准化流）
stochastic duration predictor（随机时长预测器）
adversarial loss（对抗损失）
feature matching loss（特征匹配损失）
```

它重要的原因不是“必须学会 VITS 所有公式”，而是它展示了现代 TTS 的几个趋势：

```text
用潜变量表达一对多读法。
用 flow 改善分布建模。
用 GAN 提升波形真实感。
把多个模块放到端到端训练里。
```

## 8.12 声学模型常见排错入口

工程实践中，可以按下面方式初步定位问题：

| 现象 | 优先怀疑 |
| --- | --- |
| 字读错 | 文本前端、G2P、多音字消歧 |
| 漏读 / 重复 | alignment、attention、duration |
| 语速奇怪 | duration predictor、切句策略 |
| 语调平 | pitch / F0 建模、风格条件 |
| 力度不自然 | energy 建模、训练数据风格 |
| 音色不像 | speaker embedding、prompt speech、训练数据 |
| 音质毛刺 / 爆音 | vocoder、采样率、音频预处理 |

一个实用原则：

```text
先看声学模型输入是否正确，再听声学模型输出，再判断 vocoder。
```

如果模型有中间 mel 可视化，排查会更容易：mel 已经异常，问题多半在声学模型或输入条件；mel 看起来正常但音频有毛刺，问题更可能在 vocoder 或音频后处理。

## 8.13 本章小结

本章最重要的直觉：

```text
声学模型把文本、音素、说话人、风格等条件变成声学表示。
模型权重是一组矩阵 / 向量 / 张量；输入条件会先变成向量，再被这些权重层层计算。
speaker embedding 是“谁在说”的条件输入，不是模型权重本身；它是否好用取决于训练数据、表达力、模型结构和 decoder 能力。
alignment（对齐）连接文本 token 和语音帧，是 TTS 稳定性的关键。
duration（时长）是最直接、最工程化的对齐形式。
Tacotron 类依赖 attention，声音自然但推理慢、长文本不稳。
FastSpeech 类显式建模 duration / pitch / energy，更快、更稳、更可控。
VITS 类把 latent、flow、GAN 和端到端训练引入 TTS。
```

下一章进入 vocoder（声码器）：它负责把 mel、latent 或 codec representation（语音编码表示）变成最终 waveform（波形）。
