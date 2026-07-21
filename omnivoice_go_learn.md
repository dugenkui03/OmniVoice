# OmniVoice 源码逐类学习：`omnivoice/models/omnivoice.py`

本文按 `omnivoice/models/omnivoice.py` 中定义的 7 个类，逐节介绍每个类的作用、关键字段和在推理链路中的位置。阅读时可以配合 `README_ZH_LEARN.md`（推理全链路）和 `omnivoice/models/forward.md`（`forward()` 细节）。

| # | 类 | 行 | 作用 |
| --- | --- | --- | --- |
| 1 | `VoiceClonePrompt` | 102 | 声音克隆 prompt 数据结构 |
| 2 | `OmniVoiceGenerationConfig` | 114 | 生成参数配置 |
| 3 | `GenerationTask` | 139 | 一批生成任务 |
| 4 | `OmniVoiceModelOutput` | 209 | forward 输出（logits/loss） |
| 5 | `OmniVoiceConfig` | 219 | 模型配置 |
| 6 | `EmbeddingAccessMixin` | 268 | embedding 访问接口 Mixin |
| 7 | `OmniVoice` | 290 | 主模型 |

## 1. `VoiceClonePrompt`（声音克隆 prompt 数据结构）

`VoiceClonePrompt` 是一个 `@dataclass`，用来缓存**参考音频的编码结果**，让同一个参考音色可以在多次 `generate()` 调用中复用，避免每次都重新跑一遍 `audio_tokenizer.encode`。

它由 `OmniVoice.create_voice_clone_prompt()` 构造一次，内部保存三样东西：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `ref_audio_tokens` | `torch.Tensor` | 参考音频编码后的离散 token，形状 `(C=8, T)`，是 8 层 codebook 的整数 ID，不是喂给模型权重的向量 |
| `ref_text` | `str` | 参考音频对应的文本，可由 Whisper 自动转写 |
| `ref_rms` | `float` | 参考音频响度（RMS 均方根 = `sqrt(mean(wav**2))`），用于给输出做音量归一 |

要点：

- **它是缓存，不是模型模块**：只存数据，最耗时的 `encode` 只做一次。
- **`ref_audio_tokens` 是 token 不是向量**：它是 tokenizer 的输出、embedding 的输入；后续会被拼进 `input_ids (B,C,S)`，再经 `audio_embeddings` 查表才变成 `(B,S,H)` 向量。
- **`ref_rms` 服务音量一致性**：编码前把过轻的参考音频放大，输出后按它归一，让克隆结果响度贴近参考。

## 2. `OmniVoiceGenerationConfig`（生成参数配置）

> 待补充：生成参数配置（温度、CFG、时间步等）。

## 3. `GenerationTask`（一批生成任务）

> 待补充：一批生成任务的数据组织与切分。

## 4. `OmniVoiceModelOutput`（forward 输出）

> 待补充：`forward()` 的返回结构（logits / loss）。

## 5. `OmniVoiceConfig`（模型配置）

> 待补充：模型配置项（`audio_vocab_size`、`audio_mask_id`、`num_audio_codebook` 等）。

## 6. `EmbeddingAccessMixin`（embedding 访问接口 Mixin）

> 待补充：转发文本 embedding 访问到内部 LLM 的 Mixin。

## 7. `OmniVoice`（主模型）

> 待补充：主模型，承载 embedding、forward、generate、迭代解码等核心逻辑。
