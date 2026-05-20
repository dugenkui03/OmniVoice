# 第零章：书籍介绍

这本书的目标不是只介绍某一个 OmniVoice demo，而是围绕 TTS（Text-to-Speech）建立一套可持续扩展的知识框架：先理解声音和语音的基本规律，再理解神经 TTS 模型如何把文本、音素、韵律、音色、情绪和风格组织成可生成的声学表示，最后落到训练、推理、评测、部署和 OmniVoice 工程实践。

## 写作目标

本书重点回答两类问题：

1. TTS 相关的基本原理：表征、音素、因素、音色、韵律、情绪、F0、energy、duration、mel、codec token 等术语到底是什么意思，它们在语音信号和模型特征中通常如何分工。
2. TTS 相关的工程技术：声学模型、vocoder、扩散模型、flow matching、模型训练、推理采样、控制、评测和部署如何串成完整工程链路。

这里把“因素”理解为两类：

```text
语音学术语：phoneme 音素、syllable 音节、tone 声调、stress 重音
模型条件因素：speaker、timbre、emotion、style、speed、prosody、accent
```

## 本书结构

本书分成三编：

```text
第一编：声音与语音基础
声音 → 人声产生 → 文本前端 → 音素 → 声学表征 → 韵律/音色/情绪分解

第二编：TTS 模型技术
模型演化 → 声学模型 → vocoder → diffusion / flow matching

第三编：工程实践
数据 → 特征 → 对齐 → 损失函数 → 训练 → 推理 → 评测 → 部署 → OmniVoice 案例
```

核心知识地图：

```text
文本内容：
character（字符） / phoneme（音素） / pinyin（拼音） / BPE（子词切分）

发音结构：
phoneme（音素）, tone（声调）, stress（重音）, syllable（音节）

韵律：
duration（时长）, pitch/F0（音高/基频）, energy（能量）, pause（停顿）, rhythm（节奏）

说话人 / 音色：
speaker embedding（说话人向量）, d-vector（声纹向量）, prompt speech（提示语音）

情绪 / 风格：
emotion embedding（情绪向量）, style embedding（风格向量）, reference encoder（参考编码器）, prosody embedding（韵律向量）

声学表征：
mel-spectrogram（梅尔频谱）, linear spectrogram（线性频谱）, codec latent（编码潜变量）, acoustic token（声学 token）

生成模型：
autoregressive（自回归）, non-autoregressive（非自回归）, VAE（变分自编码器）, flow（流模型）, GAN（生成对抗网络）, diffusion（扩散模型）, flow matching（流匹配）

工程链路：
data cleaning（数据清洗） → feature extraction（特征提取） → alignment（对齐） → training（训练） → sampling（采样） → evaluation（评测） → deployment（部署）
```

一句话总结：

> 研究扩散 TTS 的关键，不只是学 diffusion，而是先搞清楚 TTS 把“说什么、谁在说、怎么说、情绪如何”分别放进了哪些表示和条件里；然后再理解 diffusion / flow matching 是在 mel、waveform、latent 还是 codec token 空间里做生成。

## 总目录

### 第一编：声音与语音基础

1. [TTS 到底在解决什么问题](chapter01_TTS到底在解决什么问题.md)
2. [声音是什么：波形、采样率与频率](chapter02_声音是什么.md)
3. [人声的产生机制：声带、基频与共振峰](chapter03_人声的产生机制.md)
4. [文本前端与音素：文字不是发音](chapter04_文本前端与音素.md)
5. [语音信号的时频表示：mel、F0、energy 与 codec token](chapter05_语音信号的时频表示.md)
6. [语音信息分解：表征、音色、韵律与情绪](chapter06_语音信息分解.md)

### 第二编：TTS 模型技术

7. [TTS 模型的历史演化](chapter07_TTS模型的历史演化.md)
8. [声学模型 Acoustic Model](chapter08_声学模型.md)
9. [Vocoder 与波形生成](chapter09_Vocoder与波形生成.md)
10. [扩散模型与新一代 TTS](chapter10_扩散模型与新一代TTS.md)

### 第三编：工程实践

11. [训练工程：数据、特征、对齐与损失函数](chapter11_训练工程.md)
12. [推理、控制、评测与部署](chapter12_推理控制评测与部署.md)
13. [OmniVoice 生态：开源模型、商业 SaaS 与同名系统](chapter13_OmniVoice生态.md)
14. [OmniVoice 本地实战：零样本声音克隆](chapter14_OmniVoice本地实战.md)
15. [OmniVoice 可控生成：音色克隆与口音修改](chapter15_OmniVoice可控生成.md)

## 文件目录

本节记录 `books/` 下的文件与章节映射。后续新增、重命名或拆分章节时，需要同步更新这里和上面的总目录。

```text
books/
├── TODO_基本原理补充.md
├── chapter00_书籍介绍中.md
├── chapter01_TTS到底在解决什么问题.md
├── chapter02_声音是什么.md
├── chapter03_人声的产生机制.md
├── chapter04_文本前端与音素.md
├── chapter05_语音信号的时频表示.md
├── chapter06_语音信息分解.md
├── chapter07_TTS模型的历史演化.md
├── chapter08_声学模型.md
├── chapter09_Vocoder与波形生成.md
├── chapter10_扩散模型与新一代TTS.md
├── chapter11_训练工程.md
├── chapter12_推理控制评测与部署.md
├── chapter13_OmniVoice生态.md
├── chapter14_OmniVoice本地实战.md
└── chapter15_OmniVoice可控生成.md
```

## 章节更新参考

后续更新每一章时，优先参考本节，保证章节之间的职责边界清晰，避免把扩散模型、vocoder、声学基础和 OmniVoice 案例混在同一章里。

| 章节 | 当前定位 | 后续更新重点 |
| --- | --- | --- |
| 第 0 章：书籍介绍 | 全书目标、目录、阅读路径、章节职责说明 | 只放正文之外的内容，包括写作目标、学习路线、目录变更说明 |
| 第 1 章：TTS 到底在解决什么问题 | 建立 TTS 全局任务视角 | 补充 TTS 任务定义、输入输出、评价目标、系统总流程图 |
| 第 2 章：声音是什么 | 声音物理基础 | 补充 waveform、sample rate、amplitude、frequency、phase、harmonic、noise |
| 第 3 章：人声的产生机制 | 人声声学基础 | 补充 source-filter model、F0、formant、vocal tract、清浊音、元音辅音 |
| 第 4 章：文本前端与音素 | 文本到发音结构 | 补充 text normalization、G2P、拼音、音素、声调、多音字、韵律边界 |
| 第 5 章：语音信号的时频表示 | 模型常用声学表示 | 补充 STFT、spectrogram、mel、F0、energy、duration、codec token 的区别 |
| 第 6 章：语音信息分解 | 内容、音色、情绪、风格的特征分工 | 补充 speaker embedding、style embedding、prompt speech、特征解耦和混合表示 |
| 第 7 章：TTS 模型的历史演化 | 模型路线图 | 补充 Tacotron、FastSpeech、VITS、Diffusion、Flow Matching 的演化逻辑 |
| 第 8 章：声学模型 | 文本/音素到声学表示 | 补充 encoder、decoder、attention、duration、variance adaptor、latent variable |
| 第 9 章：Vocoder 与波形生成 | 声学表示到 waveform | 补充 HiFi-GAN、DiffWave、BigVGAN、GAN loss、diffusion vocoder |
| 第 10 章：扩散模型与新一代 TTS | diffusion / flow matching 生成建模 | 补充 DDPM、score、condition、cross-attention、Grad-TTS、latent diffusion、F5-TTS |
| 第 11 章：训练工程 | 数据到训练 batch | 补充数据清洗、特征提取、forced alignment、MAS、loss、mask、bucket、EMA |
| 第 12 章：推理、控制、评测与部署 | 从模型到服务 | 补充采样步数、guidance、RTF、MOS、speaker similarity、部署优化和监控 |
| 第 13 章：OmniVoice 生态 | OmniVoice 名称和生态边界 | 保留生态澄清，避免和前面基础章节混写 |
| 第 14 章：OmniVoice 本地实战 | zero-shot voice cloning 工程案例 | 补充环境、依赖、设备 fallback、类型兼容、完整脚本 |
| 第 15 章：OmniVoice 可控生成 | 音色克隆与口音控制案例 | 补充 `ref_audio`、`instruct`、口音白名单、可控生成和特征分解的对应关系 |

## 阅读路径

如果目标是快速建立 TTS 研究框架，建议优先读：

```text
第 1 章：TTS 任务
第 5 章：mel / F0 / energy / duration
第 6 章：内容、音色、情绪、风格分解
第 8-10 章：声学模型、vocoder、diffusion / flow matching
第 11-12 章：训练、推理、评测、部署
```

如果目标是先跑通 demo，可以先读第 13-15 章，但建议回头补第 1-6 章，否则很容易只会改参数，不理解参数背后的声学含义。

## 参考阅读顺序

后续扩写正文时，建议围绕以下论文和系统逐步展开：

```text
Tacotron 2
FastSpeech 2
HiFi-GAN
DiffWave
Grad-TTS
VITS
DiffSinger / ProDiff / FastDiff
NaturalSpeech 系列
codec-based TTS
E2-TTS / F5-TTS / flow matching TTS
```
