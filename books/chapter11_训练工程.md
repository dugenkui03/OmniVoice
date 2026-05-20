# 第十一章：训练工程 —— 数据、特征、对齐与损失函数

TTS 的工程成败很大程度取决于数据和训练流程。本章关注从原始数据到可训练 batch 的完整链路。

## 11.1 数据清洗

常见处理：

```text
采样率统一
响度归一化
静音裁剪
文本清洗
音频文本对齐
坏样本过滤
说话人标签
情绪标签
语言标签
音频质量检查
```

常见问题：

```text
文本和音频不匹配
噪声太大
口音不一致
采样率混乱
音量不稳定
长音频切分错误
标点影响韵律
多音字错误
```

## 11.2 特征提取与对齐

需要理解：

```text
mel extraction
F0 extraction
energy extraction
phoneme duration extraction
forced alignment
MFA
CTC alignment
MAS
text-audio pair preprocessing
```

## 11.3 损失函数

| 模型类型 | 典型 loss |
| --- | --- |
| Tacotron 类 | mel loss, stop token loss |
| FastSpeech 类 | mel, duration, pitch, energy loss |
| VITS 类 | reconstruction, KL, duration, adversarial, feature matching |
| GAN vocoder | adversarial, feature matching, mel loss |
| Diffusion TTS | noise prediction / score matching |
| Flow matching TTS | vector field / flow matching loss |

## 11.4 长度、batch 与 mask

语音模型的输入输出长度变化很大，因此训练时必须处理：

```text
padding
mask
length bucket
dynamic batching
gradient accumulation
mixed precision
distributed training
checkpoint
EMA
learning rate schedule
warmup
```
