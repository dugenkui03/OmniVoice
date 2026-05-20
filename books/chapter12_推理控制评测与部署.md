# 第十二章：推理、控制、评测与部署

本章从“模型能训练”走向“模型能稳定服务”。

## 12.1 TTS 推理流程

两阶段 TTS 推理：

```text
输入文本
 → 文本前端
 → 音素 / 拼音
 → 声学模型生成 mel
 → vocoder 生成 waveform
 → 后处理
```

扩散 TTS 推理：

```text
输入文本和条件
 → 初始化随机噪声
 → 多步去噪
 → 得到 mel / latent / waveform
 → vocoder 或 decoder
 → 输出音频
```

## 12.2 可控 TTS

常见控制维度：

```text
speaker
emotion
style
speed
pitch
energy
pause
accent
language
voice age
voice gender
```

控制方式：

```text
显式特征控制：duration / pitch / energy
embedding 控制：speaker / emotion / style embedding
prompt 控制：参考音频
文本 prompt 控制：自然语言风格描述
guidance 控制：classifier-free guidance
编辑控制：局部重生成 / inpainting
```

## 12.3 评测

| 研究目标 | 主要评测 |
| --- | --- |
| 音质 | MOS, CMOS |
| 发音准确 | WER, 人工听测 |
| 音色克隆 | speaker similarity |
| 情绪控制 | emotion similarity, MOS |
| 推理效率 | RTF |
| 韵律自然 | MOS, F0/duration 分析 |

## 12.4 部署

工程落地需要关注：

```text
模型导出
ONNX
TensorRT
TorchScript
量化
流式生成
批量推理
GPU/CPU 性能
缓存文本前端
音频后处理
服务接口
失败样本监控
```
