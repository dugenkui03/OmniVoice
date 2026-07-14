# 第十九章：扩展知识二 —— 神经音频 Codec、Codebook 与语音 Token

现代 TTS 和语音大模型经常把声音转换成 token。文本里的 token 通常来自分词器和词表，语音里的 token 则常常来自 neural audio codec（神经音频编解码器）。理解 codec、codebook、RVQ、semantic token 和 acoustic token，是读懂 OmniVoice、VALL-E、CosyVoice、SpeechTokenizer、EnCodec 这类路线的关键。

![神经音频 Codec、Codebook、RVQ、Token 形状与 OmniVoice 映射图](./images/chapter19_神经音频Codec_Codebook_RVQ与OmniVoice映射图.png)

本章补充两个常见问题：

```text
语音为什么可以像文本一样变成 token？
一帧声音为什么会有多层 codebook token？
```

## 19.1 声音变成 token 的完整链路

神经音频 codec 的目标是：把 waveform（波形）压缩成更短的中间表示，并且尽量保真地还原回 waveform。它通常包含 encoder、quantizer 和 decoder 三个部分。

```mermaid
flowchart LR
    subgraph rawStage["原始音频"]
        A["waveform<br/>连续采样点"]
    end

    subgraph encodeStage["编码压缩"]
        B["codec encoder<br/>波形 -> 连续 latent"]
        C["continuous latent<br/>按时间排列的连续向量"]
    end

    subgraph quantStage["离散化 / token 化"]
        D["RVQ quantizer<br/>多层 codebook 量化"]
        E["codec / acoustic tokens<br/>离散整数 ID"]
    end

    subgraph decodeStage["波形还原"]
        F["codec decoder<br/>token -> waveform"]
        G["reconstructed waveform<br/>重建音频"]
    end

    A --> B --> C --> D --> E --> F --> G
    G -.-> L["图例：紫=数据 / 表示｜橙=处理模块｜绿=输出"]

    classDef data fill:#F0EEFF,stroke:#9B8CFF,color:#2D2A3A;
    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef output fill:#EEF8F3,stroke:#5CA982,color:#173B2B;
    classDef note fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:3 3,color:#5A5134;
    class A,C,E data;
    class B,D,F core;
    class G output;
    class L note;
    style rawStage fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:6 4,color:#5A5134;
    style encodeStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style quantStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style decodeStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#173B2B;
```

这条链路里，**encoder** 把原始波形压成连续向量，**quantizer** 把连续向量变成离散编号，**decoder** 再根据这些编号还原声音。

文本 tokenizer 的工作大致是：

```text
文本 -> token -> token id
```

声学 tokenizer / neural codec 的工作更接近：

```text
waveform -> continuous latent -> codec token -> waveform
```

两者都产出整数 ID，但来源不同。文本 token 来自语言符号切分，声学 token 来自神经网络对声音的压缩和量化。

## 19.2 Codebook 是什么

codebook 可以理解成一组模型学出来的“声音向量字典”。每个 codebook 里有很多 codeword（码字），每个 codeword 是一个向量。

```text
codebook 0
  id 0   -> 向量 v0
  id 1   -> 向量 v1
  id 2   -> 向量 v2
  ...
  id 1023 -> 向量 v1023
```

codec encoder 输出的 latent 是连续向量。quantizer 会在 codebook 里找一个最接近的 codeword，然后用这个 codeword 的编号代替原始连续向量。

```mermaid
flowchart LR
    subgraph latentStage["连续表示"]
        A["encoder latent z<br/>一个连续向量"]
    end

    subgraph bookStage["codebook"]
        B["codeword 0"]
        C["codeword 1"]
        D["codeword 42<br/>最近"]
        E["codeword 1023"]
    end

    subgraph tokenStage["离散 token"]
        F["token id = 42"]
        G["量化向量 z_q<br/>= codebook[42]"]
    end

    A --> D
    D --> F
    F --> G
    G -.-> H["图例：紫=表示｜橙=查表 / 量化"]

    classDef data fill:#F0EEFF,stroke:#9B8CFF,color:#2D2A3A;
    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef note fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:3 3,color:#5A5134;
    class A,F,G data;
    class B,C,D,E core;
    class H note;
    style latentStage fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:6 4,color:#5A5134;
    style bookStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style tokenStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
```

这个过程叫 Vector Quantization（向量量化，VQ）。它的直觉是：连续世界里的声音细节很多，模型先用有限个“典型声音向量”近似它们，再用整数 ID 表示被选中的向量。

codebook 越大，可选码字越多，理论上量化越精细；但 codebook 太大也会增加训练难度，并可能出现很多码字长期不用的 codebook 利用率问题。

从数据结构看，一个 codebook 可以写成一张形状为 `(K, D)` 的向量表：

```text
K = codeword 数量，例如 1024
D = 每个 codeword 的向量维度
```

token ID 只是这张表的行号。例如 `token_id=42` 表示选择 `codebook[42]`。整数 `42` 本身没有“音高为 42”或“音色为 42”的物理含义，真正参与解码的是第 42 行保存的向量。

多层 RVQ 为每一层维护独立的 codebook。`codebook 0` 中的 ID 42 与 `codebook 1` 中的 ID 42 指向两张不同表中的向量，两者不要求具有相同含义。因此，读取声学 token 时必须同时知道 **codebook 层号** 和 **token ID**。

## 19.3 RVQ：一帧为什么有多个 codebook token

一层 VQ 只从一个 codebook 里选一个 codeword，表示能力有限。Residual Vector Quantization（RVQ，残差向量量化）会用多层 codebook 逐步补充细节。

RVQ 的核心过程是：

```text
第 1 层 codebook 先近似原始 latent
计算剩下没表示好的 residual（残差）
第 2 层 codebook 再近似这个 residual
继续计算新的 residual
后续 codebook 继续补充细节
```

```mermaid
flowchart LR
    subgraph inputStage["输入 latent"]
        A["连续 latent z"]
    end

    subgraph rvqStage["RVQ 多层量化"]
        B["codebook 1<br/>选 token c1"]
        C["residual 1<br/>剩余误差"]
        D["codebook 2<br/>选 token c2"]
        E["residual 2<br/>剩余误差"]
        F["..."]
        G["codebook 8<br/>选 token c8"]
    end

    subgraph outputStage["单个时间帧的 token"]
        H["[c1, c2, ..., c8]<br/>一帧的 8 层 token"]
    end

    A --> B --> C --> D --> E --> F --> G --> H

    classDef data fill:#F0EEFF,stroke:#9B8CFF,color:#2D2A3A;
    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    class A,C,E,H data;
    class B,D,F,G core;
    style inputStage fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:6 4,color:#5A5134;
    style rvqStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style outputStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
```

因此，**一个时间帧可以对应多个 token**。这不是因为这帧被切成了 8 段，而是因为同一个时间位置的 latent 用 8 层 codebook 共同描述。

一个简化的二维向量例子可以展示“残差”如何逐层缩小。假设某个时间帧的 encoder latent 是：

```text
原始 latent z = [3.2, 1.7]

第 1 层选中 q1 = [3.0, 1.0]
剩余残差 r1 = z - q1 = [0.2, 0.7]

第 2 层选中 q2 = [0.1, 0.6]
剩余残差 r2 = r1 - q2 = [0.1, 0.1]

重建向量 z_q = q1 + q2 = [3.1, 1.6]
```

实际系统中的 latent 维度更高，codebook 层也更多，但原则相同：第一层近似当前向量，下一层不再重复量化原向量，而是量化上一层留下的误差。最终送给 decoder 的量化向量，通常由各层选中的 codeword 相加得到。

可以把它想成一张表：

```text
          t0    t1    t2    t3    ...
cb0       12    88   301    44    ...
cb1      510    23    90   771    ...
cb2       18   302   119    65    ...
...
cb7      700   112     8   433    ...
```

这里横向的 `t0, t1, t2...` 是时间帧，纵向的 `cb0` 到 `cb7` 是 codebook 层。**每一列的 8 个 token 合在一起，描述同一个时间位置的声音。**

## 19.4 `(C=8, T)` 与 `(B, C, S)` 这种形状怎么读

在代码和论文里，声学 token 常写成类似 `[B, Nq, T]` 或 `(C, T)` 的形状。

| 维度 | 含义 | 直觉 |
| --- | --- | --- |
| `B` | batch size | 一次处理几条音频 |
| `Nq` / `C` | codebook 层数 | 每个时间帧用几层 token 描述 |
| `T` | 时间帧数量 | 这段音频被压缩成多少个时间位置 |

OmniVoice 里常看到 `(C=8, T)`，可以读成：

```text
C = 8 个 codebook 层
T = 时间帧数量
```

如果一段音频被编码成 `(8, 250)`，意思不是“8 段音频，每段 250 个点”，而是：

```text
250 个时间帧
每个时间帧有 8 个 codebook token
```

这一点和图像里的通道维度有一点相似：RGB 图像的同一个像素位置会有 R、G、B 三个通道值；多 codebook 声学 token 的同一个时间位置会有多层 token ID。不过语音 codebook 不是颜色通道，而是神经 codec 学出来的分层离散表示。

下面是一份完整的 `(8, 4)` 示例：

```text
             t0    t1    t2    t3
codebook 0  101   102   103   104
codebook 1  201   202   203   204
codebook 2  301   302   303   304
codebook 3  401   402   403   404
codebook 4  501   502   503   504
codebook 5  601   602   603   604
codebook 6  701   702   703   704
codebook 7  801   802   803   804
```

它包含 `8 × 4 = 32` 个 token ID，表示 4 个时间帧。按列读取时：

```text
t0 = [101, 201, 301, 401, 501, 601, 701, 801]
t1 = [102, 202, 302, 402, 502, 602, 702, 802]
t2 = [103, 203, 303, 403, 503, 603, 703, 803]
t3 = [104, 204, 304, 404, 504, 604, 704, 804]
```

每一列的 8 个 ID 共同描述同一个时间帧。每一行则表示同一个 codebook 在连续时间上的 token 序列。矩阵中的元素仍然都是整数 ID；“二维”只说明这些 ID 同时按 **codebook 层** 和 **时间帧** 两个轴组织。

### 三维形状可以从右向左理解

形状 `(A, B, C)` 表示一个三维 Tensor。理解它时可以从最右侧开始：

```text
C 个数字组成一个一维向量
B 个这样的向量组成一个二维矩阵，形状为 (B, C)
A 个这样的矩阵叠在一起，组成三维 Tensor，形状为 (A, B, C)
```

例如，下面的 Tensor 形状是 `(2, 3, 4)`：

```text
[
  [                          # 第 0 个矩阵，形状 (3, 4)
    [101, 102, 103, 104],
    [201, 202, 203, 204],
    [301, 302, 303, 304]
  ],
  [                          # 第 1 个矩阵，形状 (3, 4)
    [105, 106, 107, 108],
    [205, 206, 207, 208],
    [305, 306, 307, 308]
  ]
]
```

最里层每个向量有 4 个数字；每个矩阵有 3 行；最外层有 2 个矩阵，因此总元素数为 `2 × 3 × 4 = 24`。访问 `tensor[a, b, c]` 时，三个下标依次表示第 `a` 个矩阵、第 `b` 行和第 `c` 个位置。

### OmniVoice 中的 `(B, C, S)`

OmniVoice 主模型的混合输入 `input_ids` 通常使用 `(B, C, S)`：

| 维度 | 在 OmniVoice 中的含义 |
| --- | --- |
| `B` | batch size，一次共同处理的 TTS 样本数量 |
| `C` | codebook 层数，当前固定为 8 |
| `S` | 每条样本的完整混合序列长度 |

从右向左读取时：

```text
S 个序列位置组成一行
8 个 codebook 行组成一条样本的 (8, S) 矩阵
B 个 (8, S) 矩阵组成完整的 (B, 8, S) batch
```

例如 `(2, 8, 100)` 表示一次批量处理 2 条 TTS 样本；每条样本使用 8 个 codebook 层，并被整理成长度为 100 的混合序列。第一维 `B=2` 不是新的声音属性，而是把两条独立样本的矩阵堆叠起来，以便 GPU 并行计算。

```text
batch
├── 样本 0：(8, 100)
│   ├── codebook 0：100 个序列位置
│   ├── codebook 1：100 个序列位置
│   └── ...
└── 样本 1：(8, 100)
    ├── codebook 0：100 个序列位置
    ├── codebook 1：100 个序列位置
    └── ...
```

这里使用 `S` 而不是 `T`，是因为主模型的序列不只有音频时间帧。它会按顺序包含：

```text
[语言 / 风格信息] + [文本] + [可选的参考音频 token] + [目标音频 token]
```

文本位置本来只有一层 token ID，OmniVoice 会把它复制到 8 层，以便和音频 token 一起放进规则的 `(B,8,S)` Tensor。`audio_mask` 再标记哪些序列位置属于音频，模型据此选择文本 embedding 或音频 embedding。

对于不含文本和风格区域的纯音频 token，时间轴仍写作 `T`：

```text
(C, T)    = 一条音频的 8 层 codebook token
(B, C, T) = B 条音频组成的 token batch
```

批量样本的实际长度可能不同。工程实现通常会补齐到当前 batch 的最大长度，再使用 attention mask 区分有效位置和 padding。OmniVoice 的迭代推理还会为 Classifier-Free Guidance 构造条件与无条件两份输入，因此内部某次 `forward()` 的第一维可能暂时表现为 `2B`；它仍然来源于原始的 `B` 条 TTS 样本。

整数 token Tensor 进入 Transformer 前，还会经过 embedding lookup：

```text
input_ids: (B, C, S) 整数 token ID
        ↓ 文本 / 音频 embedding 查表
inputs_embeds: (B, S, H) 连续浮点向量
```

同一音频位置的 8 层 codebook ID 会分别查表并相加，形成该时间位置的一个 `H` 维复合表示。`H` 是 Transformer 的隐藏维度，不再是 codebook 层数。

## 19.5 hop_length 决定时间帧率，不等于简单切块大小

`hop_length` 表示相邻两个时间位置在原始采样点上相隔多少。它更准确叫 frame shift（帧移 / 步长），而不是严格意义上的“每帧大小”。

以 24 kHz 音频和 `hop_length = 960` 为例：

```text
sampling_rate = 24000
hop_length = 960
24000 / 960 = 25
```

所以大致可以得到：

```text
1 秒音频 -> 约 25 个 token 时间帧
每个时间位置间隔约 40ms
```

如果 codec 有 8 个 codebook，那么 1 秒音频大约会形成：

```text
8 层 codebook x 25 个时间帧 = 200 个 token ID
```

这里要避免一个误解：**一帧 token 不等于机械地只看 960 个采样点**。神经 codec 的 encoder 往往包含卷积、下采样和上下文建模，一个时间位置的 latent 可能受到前后邻近音频的影响。工程上说 `hop_length=960`，主要是在说明时间轴上每隔 960 个采样点产出一个 token 帧。

## 19.6 Codebook 越多，信息一定越多吗

每帧使用更多 codebook，理论上可以保留更多声学信息。因为每个时间位置不再只靠一个离散 ID 描述，而是由多层 token 共同描述。

```text
更多 codebook
-> 单帧表达容量更大
-> 重建音质、音色细节和瞬态信息可能更好
```

但它也会带来成本：

```text
更多 token 需要预测
模型训练和推理更复杂
采样空间更大
错误传播和不一致风险更高
```

因此，codebook 数量是一种工程取舍。较少的 codebook 更短、更容易建模，但细节可能不足；较多的 codebook 还原能力更强，但主模型要生成的 token 数量也更多。

从残差量化过程看，多层 codebook 可以用“先近似、再修正”理解：

| 层级 | 量化职责 | 常见效果 |
| --- | --- | --- |
| 前几层 | 先近似 encoder latent | 通常承担较大的重建贡献 |
| 后几层 | 继续量化前面留下的残差 | 逐步降低重建误差并补充细节 |

“前层负责内容、后层负责音色”并不是 RVQ 结构天然保证的性质。EnCodec 的核心目标是高保真重建，RVQ 层首先是残差量化层；SpeechTokenizer 额外使用 HuBERT 语义教师约束第一层，才有意识地让第一层更偏内容、后续层补充音色和韵律。具体每层捕捉什么，取决于 tokenizer 的架构、训练目标和数据，不能只根据层号直接命名为音高层、音色层或情绪层。

## 19.7 Semantic token 与 acoustic token

语音 token 大致可以分成两类：semantic token（语义 token）和 acoustic token（声学 token）。

| 类型 | 更偏向保存什么 | 常见来源 | 优点 | 局限 |
| --- | --- | --- | --- | --- |
| semantic token | 说了什么、音素结构、较高层内容 | HuBERT、w2v-BERT 等 SSL 模型的聚类，或被语义蒸馏约束的 RVQ 层 | 和文本对齐更好，适合语言建模 | 音色、细节和自然度可能不足 |
| acoustic token | 音色、韵律、局部声学细节、重建信息 | EnCodec、SoundStream、DAC 等 neural codec | 重建质量高，保留声音细节 | 信息复杂，和文本内容不一定强对齐 |

SpeechTokenizer 这类工作试图把两者统一起来：让第一层 RVQ token 更接近 semantic token，承担内容和文本对齐；让后续 RVQ 层补充 timbre（音色）、prosody（韵律）和 acoustic details（声学细节）。

```mermaid
flowchart LR
    subgraph semStage["语义层"]
        A["RVQ-1 / semantic-like token<br/>内容、音素、文本对齐"]
    end

    subgraph acousticStage["声学层"]
        B["RVQ-2:8 / acoustic tokens<br/>音色、韵律、细节"]
    end

    subgraph modelStage["语音生成"]
        C["AR / NAR / diffusion / flow<br/>生成或补全 token"]
        D["codec decoder<br/>还原 waveform"]
    end

    A --> C
    B --> C
    C --> D

    classDef data fill:#F0EEFF,stroke:#9B8CFF,color:#2D2A3A;
    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    class A,B data;
    class C,D core;
    style semStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
    style acousticStage fill:#FFFDF4,stroke:#8B80D8,stroke-dasharray:6 4,color:#4A436E;
    style modelStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
```

这种分层思路解释了为什么很多系统会采用两阶段或分层生成：

```text
先生成更靠近内容的 token
再生成更靠近声学细节的 token
最后用 decoder 还原成 waveform
```

## 19.8 OmniVoice 中的对应关系

OmniVoice 属于 codec token / Speech LM 思路。主模型并不直接输出 waveform，也不以 mel-spectrogram 作为主要输出，而是在 token 空间里生成多 codebook audio token。

```mermaid
flowchart LR
    subgraph inputStage["条件输入"]
        A["目标文本"]
        B["参考音频<br/>voice prompt"]
    end

    subgraph tokenizerStage["前置 tokenizer"]
        C["text tokenizer<br/>文本 -> token id"]
        D["audio tokenizer.encode<br/>参考音频 -> audio tokens"]
    end

    subgraph mainStage["主生成模型"]
        E["OmniVoice backbone<br/>生成 / 补全目标 audio tokens"]
        F["target audio tokens<br/>(C=8, T)"]
    end

    subgraph restoreStage["波形还原"]
        G["audio tokenizer.decode<br/>codec decoder"]
        H["24 kHz waveform"]
    end

    A --> C --> E
    B --> D --> E
    E --> F --> G --> H
    H -.-> I["图例：紫=数据｜橙=模块｜绿=输出"]

    classDef data fill:#F0EEFF,stroke:#9B8CFF,color:#2D2A3A;
    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef output fill:#EEF8F3,stroke:#5CA982,color:#173B2B;
    classDef note fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:3 3,color:#5A5134;
    class A,B,F data;
    class C,D,E,G core;
    class H output;
    class I note;
    style inputStage fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:6 4,color:#5A5134;
    style tokenizerStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style mainStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style restoreStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#173B2B;
```

工程上可以这样读 OmniVoice 里的相关概念：

| 概念 | 在 OmniVoice 中的直觉 |
| --- | --- |
| `audio_tokenizer` | 负责 waveform 与 audio tokens 互转的神经音频 tokenizer / codec |
| `num_audio_codebook=8` | 每个时间帧用 8 层 codebook token 表示 |
| 每层 `codebook_size=1024` | 每个 codebook 有 1024 个可选 codeword，正常 token ID 范围是 0 到 1023 |
| `audio_tokens` | 主模型要处理和生成的离散声学 token |
| `audio_tokenizer.encode` | 把参考音频编码成 prompt audio tokens |
| `audio_tokenizer.decode` | 把生成出的 audio tokens 还原成 waveform |
| `hop_length` | 决定 token 时间帧的步长，例如 24 kHz 下 960 约等于 40ms |
| `audio_vocab_size=1025` | 主模型每层输出 1025 类：1024 个 codec ID 加 1 个 MASK ID |
| `audio_mask_id=1024` | OmniVoice 迭代生成时使用的待填充标记，不是正常的 codec 码字 |

声学 tokenizer 在这里不是普通工具函数，而是模型链路中的关键组件。它定义了主模型所在的声学 token 空间，也定义了哪些 token 能被正确还原成声音。

OmniVoice 的目标 token 帧率为 25 帧/秒。每秒音频需要生成：

```text
25 个时间帧 × 8 个 codebook ID = 200 个音频 token ID
```

这 200 个 ID 不是 200 个按顺序播放的独立声音片段，而是一个形状为 `(8, 25)` 的 token 矩阵。codec decoder 会先把各层 ID 查表还原为向量，再综合连续时间上的表示生成波形。

## 19.9 RVQ 的量化顺序与 OmniVoice 的生成顺序

RVQ 的编码过程和 OmniVoice 的 token 生成过程都会涉及“先后”，但两者不是同一件事。

**RVQ 的量化顺序**发生在 audio tokenizer 内部。对一个已知音频帧，第一层 codebook 先量化 latent，第二层量化第一层留下的残差，后续层继续量化新的残差。这个顺序定义了 `(C,T)` 中各层 token 如何从原音频得到。

**OmniVoice 的生成顺序**发生在主生成模型内部。推理时还没有目标音频可供编码，因此模型先建立一个全部为 MASK 的 `(8,T)` 目标矩阵，再通过离散掩码扩散逐步预测其中的 codec token ID。

```mermaid
flowchart LR
    subgraph codecStage["Audio Tokenizer：已知音频的 RVQ 编码"]
        A["waveform<br/>已知音频数据"]
        B["encoder latent<br/>连续表示"]
        C["RVQ：逐层量化残差<br/>codebook 1 -> ... -> 8"]
        D["audio tokens<br/>(8,T)"]
        A --> B --> C --> D
    end

    subgraph generationStage["OmniVoice：未知目标音频的 token 生成"]
        E["文本 / 参考音频<br/>条件数据"]
        F["全 MASK 矩阵<br/>(8,T)"]
        G["多轮 mask-fill<br/>预测并选取高置信度位置"]
        H["完整 audio tokens<br/>(8,T)"]
        E --> G
        F --> G --> H
    end

    subgraph restoreStage["波形还原"]
        I["codec decoder<br/>处理模块"]
        J["waveform<br/>输出数据"]
        I --> J
    end

    D --> I
    H --> I
    J -.-> L["图例：紫=数据 / 表示｜橙=模块｜绿=输出"]

    classDef data fill:#F0EEFF,stroke:#9B8CFF,color:#2D2A3A;
    classDef core fill:#FFF2CC,stroke:#D88B00,stroke-width:2px,color:#3B2A00;
    classDef output fill:#EEF8F3,stroke:#5CA982,color:#173B2B;
    classDef note fill:#FFFDF4,stroke:#B7A870,stroke-dasharray:3 3,color:#5A5134;
    class A,B,D,E,F,H data;
    class C,G,I core;
    class J output;
    class L note;
    style codecStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style generationStage fill:#FFFDF4,stroke:#D88B00,stroke-dasharray:6 4,color:#3B2A00;
    style restoreStage fill:#FFFDF4,stroke:#5CA982,stroke-dasharray:6 4,color:#173B2B;
```

每一轮 mask-fill 都会同时计算所有未填位置的候选分布，然后按置信度选出一批位置写入 token ID。已经写入的位置会成为下一轮的上下文，并且当前实现不会再次修改。选择范围覆盖全部 `8 × T` 个位置，因此时间轴上没有严格的从左到右顺序。

OmniVoice 还会对靠后的 codebook 施加 `layer_penalty_factor`，让前面的 codebook 更容易较早被选中，形成总体上的“前层优先、后层补充”倾向。这是生成时的位置选择策略，不等于先完整生成 codebook 0，再完整生成 codebook 1。置信度和采样随机性仍可能让不同层、不同时间帧交错完成。

```text
初始：8 × T 个位置全部为 MASK
第 1 轮：预测全部空位，填入一批高置信度 token
第 2 轮：利用已填 token 再预测剩余空位
...
最后一轮：填完所有剩余 MASK
```

这种过程属于 **discrete masked diffusion / MaskGIT-style decoding**。它处理的是离散 token ID，不是在波形上逐步去除高斯噪声，也不是在每轮中只生成某一个完整时间帧。

## 19.10 读相关论文时先抓什么

阅读 EnCodec、SoundStream、DAC、SpeechTokenizer 这类论文时，可以先抓以下几个问题。

| 问题 | 关注点 |
| --- | --- |
| codec 的输入输出是什么 | waveform、latent、token、reconstructed waveform |
| token 的形状是什么 | `[B, Nq, T]`、codebook 数、token frame rate |
| 使用几层 codebook | Nq 越多，单帧容量越大，但生成成本也更高 |
| codebook 大小是多少 | 每层有多少可选 codeword |
| token 帧率是多少 | 每秒多少个时间帧，决定序列长度 |
| 训练目标是什么 | waveform 重建、频谱损失、GAN loss、commitment loss、语义蒸馏 |
| token 更偏语义还是声学 | 决定适合先做内容建模，还是直接做高质量重建 |

本书参考目录中的两篇资料可以按下面顺序阅读：

| 文件 | 适合解决的问题 |
| --- | --- |
| [重要：codec,token.pdf](Reference/重要：codec,token.pdf) | 理解 EnCodec、neural codec、RVQ、codebook、`[B, Nq, T]` |
| [重要语义token-声学token.pdf](Reference/重要语义token-声学token.pdf) | 理解 semantic token、acoustic token、SpeechTokenizer 和 RVQ 分层语义 |

## 19.11 常见误区

| 误区 | 更准确的理解 |
| --- | --- |
| `C=8` 表示 8 段时间 | `C=8` 表示 8 层 codebook，同一个时间帧有 8 个 token |
| `hop_length=960` 表示每帧只包含 960 个采样点 | 它主要表示相邻 token 时间位置间隔 960 个采样点；encoder 可能看到上下文 |
| codebook 就是文本词表 | codebook 是神经 codec 学出来的向量字典，不是自然语言词表 |
| acoustic token 就等于语义 token | acoustic token 更重建导向，semantic token 更内容 / 文本对齐导向 |
| 每个 codebook 固定表示一种物理属性 | codebook 层量化的是学习到的向量与残差，通常不能直接命名为音高层或音色层 |
| 相同 token ID 在不同 codebook 含义相同 | 每层有独立的向量表，必须结合 codebook 层号解释 token ID |
| codebook 越多一定越好 | 更多 codebook 提高表达容量，也增加生成难度和计算成本 |
| OmniVoice 按时间从左到右填 token | 它在全部 codebook 和时间位置中选择高置信度空位进行 mask-fill |
| RVQ 量化顺序就是扩散迭代顺序 | 前者定义已知音频如何编码，后者定义未知目标 token 如何逐步生成 |
| audio tokenizer 可以随便替换 | 主模型、codebook、token 空间和 decoder 强绑定，通常不能随意混用 |

## 19.12 本章小结

神经音频 codec 把 waveform 压缩成连续 latent，再通过 VQ / RVQ 变成离散 codec token，最后由 decoder 还原成 waveform。

codebook 是模型学出来的向量字典。每个时间帧会从每层 codebook 中选出一个 token；如果有 8 层 codebook，一个时间帧就有 8 个 token。

`(C=8, T)` 表示 8 层 codebook 和 T 个时间帧。横向是时间，纵向是 codebook 层。每一列的 8 个 token 合在一起，描述同一个时间位置的声音。

`hop_length` 决定 token 时间帧率。24 kHz 音频下 `hop_length=960` 时，1 秒大约产生 25 个时间帧。

semantic token 更偏“说了什么”，acoustic token 更偏“声音如何被重建”。SpeechTokenizer 这类方法尝试让第一层 RVQ token 承担更多语义信息，让后续层补充声学细节。

OmniVoice 的主模型在 codec token 空间里工作。参考音频先由 audio tokenizer 编成 prompt audio tokens，主模型生成目标 audio tokens，最后由 audio tokenizer 的 decoder 还原成 24 kHz waveform。

OmniVoice 的生成顺序不同于 RVQ 的量化顺序：RVQ 在编码已知音频时逐层量化残差；OmniVoice 则从全 MASK 的 `(8,T)` 矩阵出发，多轮填入高置信度 token，并通过层惩罚形成前层 codebook 优先的倾向。
