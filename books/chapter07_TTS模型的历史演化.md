# 第七章：TTS 模型的历史演化

本章用于建立模型路线图：今天的扩散式 TTS、flow matching TTS 和 codec-based TTS 不是凭空出现的，而是从规则系统、拼接系统、统计参数系统和神经 TTS 一路演化而来。

## 7.1 演化顺序

```text
规则合成
 → 拼接式合成
 → HMM 统计参数合成
 → Tacotron / Tacotron 2
 → FastSpeech / FastSpeech 2
 → VITS
 → diffusion TTS
 → codec / flow matching / speech language model
```

## 7.2 每一代模型解决的问题

| 阶段 | 核心问题 |
| --- | --- |
| 规则合成 | 人工规则太硬，声音不自然 |
| 拼接式合成 | 数据库依赖强，泛化差 |
| HMM TTS | 可控但音质偏差 |
| Tacotron 类 | 端到端从文本生成 mel |
| FastSpeech 类 | 解决慢、错读、漏读、不可控 |
| VITS 类 | 端到端、潜变量、对抗训练 |
| Diffusion / Flow 类 | 建模复杂分布，提升自然度和多样性 |

## 7.3 阅读顺序建议

后续学习扩散 TTS 时，不建议跳过经典神经 TTS。推荐先理解：

```text
Tacotron 2：text → mel → vocoder
FastSpeech 2：duration / pitch / energy
HiFi-GAN：vocoder 工程
VITS：latent + flow + GAN + end-to-end
Grad-TTS：mel diffusion acoustic model
F5-TTS：flow matching + DiT + prompt-based TTS
```
