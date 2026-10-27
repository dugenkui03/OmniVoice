# 评测

使用标准 TTS 指标评测 OmniVoice 模型：WER（可懂度）、SIM-o（说话人相似度）和 UTMOS（自然度）。

## 支持的测试集

| 测试集 | 语言 | WER 模块 | 指标 |
|---|---|---|---|
| **LibriSpeech-PC** | 英语 | HuBERT WER | WER + 说话人相似度 + MOS |
| **Seed-TTS (en)** | 英语 | Whisper WER | WER + MOS |
| **Seed-TTS (zh)** | 中文 | Paraformer WER | WER + MOS |
| **FLEURS** | 102 种语言 | Omnilingual-ASR WER | WER（单语言 + 宏平均） |
| **MiniMax Multilingual** | 24 种语言 | Whisper + Paraformer | WER + MOS |

## 前置条件

```bash
pip install omnivoice[eval]
# 或者
uv sync --extra eval
```

## 快速开始

```bash
cd examples
bash run_eval.sh
# run_eval.sh 会：
# (1) 下载所有需要的测试集和评测模型；
# (2) 对每个测试集执行推理和评测。
```

## 指标解释

### WER（词错误率）

用 ASR 模型转写生成语音，再和参考文本比较，用来衡量生成语音是否容易听懂。数值越低越好。注意有些语言实际使用的是 CER（字符错误率）。

- **LibriSpeech-PC**：基于 HuBERT 的 ASR
- **Seed-TTS**：Whisper（英语）或 Paraformer（中文）
- **MiniMax**：非中文使用 Whisper，中文使用 Paraformer
- **FLEURS**：Omnilingual-ASR 多语言模型

### 说话人相似度

比较参考音频和生成音频的 speaker embedding（说话人向量，ECAPA-TDNN + WavLM）之间的余弦相似度。数值越高越好。

### UTMOS（预测 MOS）

用神经网络从音频预测 MOS（Mean Opinion Score，平均意见得分）。数值越高越好。
