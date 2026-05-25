# 生成参数

参数可以作为关键字参数传给 `model.generate(...)`，也可以通过 `OmniVoiceGenerationConfig` 数据类传入。下面列出完整参数，并说明每个参数所属类别。

```python
# 1) 直接使用关键字参数
audio = model.generate(text="Hello world", num_step=32, guidance_scale=2.0)

# 2) 通过 OmniVoiceGenerationConfig 数据类
from omnivoice import OmniVoiceGenerationConfig

config = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)
audio = model.generate(text="Hello world", generation_config=config)
```

## 解码

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `num_step` | int | 32 | 迭代式 unmask（解除掩码）的步数。值越大质量通常越好，但生成更慢。需要更快推理时可以用 16。 |
| `denoise` | bool | True | 在输入前追加 `<|denoise|>` token，用来提示模型生成更干净的语音。 |
| `guidance_scale` | float | 2.0 | classifier-free guidance（无分类器引导）的强度。 |
| `t_shift` | float | 0.1 | 噪声调度中的时间步偏移。较小的值会更强调解码早期步骤。 |

## 采样

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `position_temperature` | float | 5.0 | 选择 mask 位置时使用的 temperature（温度）。0 表示 greedy（贪心、确定性）。值越大随机性越强。 |
| `class_temperature` | float | 0.0 | 每一步采样 token 时使用的 temperature（温度）。0 表示 greedy（贪心、确定性）。值越大随机性越强。 |
| `layer_penalty_factor` | float | 5.0 | 施加到更深 codebook 层的惩罚，鼓励更早的低层 codebook 先解除 mask。 |

## 时长和语速

这些参数可以传单个值并应用到所有样本，也可以传逐样本列表（批量模式下很有用）：

```python
# 固定输出为 10 秒
audio = model.generate(text="Hello, this is a test of duration control", duration=10.0)

# 更快语速（比估计值快 1.2 倍）
audio = model.generate(text="Hello, this is a test of duration control", speed=1.2)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `duration` | float 或 list[float \| None] | None | 固定输出时长，单位为秒。设置后会覆盖 `speed`。 |
| `speed` | float 或 list[float \| None] | None | 语速因子。值 > 1.0 会生成更短音频（更快）；值 < 1.0 会生成更长音频（更慢）。设置 `duration` 时会被忽略。两者都为 None 时默认使用 1.0。 |

优先级：`duration` > `speed`。

> **注意：** 使用 `duration` 时，默认后处理可能会裁剪尾部静音，导致实际输出略短于请求时长。如果你需要输出时长和指定值**完全一致**，请设置 `postprocess_output=False` 以关闭静音移除。

## 前处理 / 后处理

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `preprocess_prompt` | bool | True | 是否对声音克隆的 prompt audio（提示音频）执行前处理，包括移除参考音频中的长静音、在参考文本末尾补标点。 |
| `postprocess_output` | bool | True | 是否对生成音频执行后处理，包括移除长静音。 |

## 长文本生成

为了在低显存占用下稳定支持长文本语音生成，当估计的生成语音时长超过 `audio_chunk_duration` 时，文本会被自动切成更小片段，每个片段大约生成 `audio_chunk_duration` 秒音频。这种方式让模型可以接收任意长度文本，并以近似恒定的显存占用生成任意长度语音。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `audio_chunk_duration` | float | 15.0 | 长文本切分时的目标 chunk（片段）时长，单位为秒。 |
| `audio_chunk_threshold` | float | 30.0 | 估计音频时长超过该阈值后启用 chunk（切分），单位为秒。 |
