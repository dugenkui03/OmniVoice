# 训练

## 训练配置

所有训练都由一个 JSON training config（训练配置）文件和一个 JSON data config（数据配置）文件控制。

可参考 [examples/config/](../examples/config/) 中可直接使用的配置。

Emilia 上的训练配置文件是：[examples/config/train_config_emilia.json](../examples/config/train_config_emilia.json)

Emilia 的数据配置文件是：[examples/config/data_config_emilia.json](../examples/config/data_config_emilia.json)

训练配置文件中的关键字段：

| 字段 | 说明 | 默认值 |
|---|---|---|
| `llm_name_or_path` | 本地 LLM 路径或 Hugging Face ID | Qwen/Qwen3-0.6B |
| `steps` | 总训练步数 | 300,000 |
| `learning_rate` | 峰值学习率 | 1e-4 |
| `batch_tokens` | 每张 GPU 每个 batch 的 token 数 | 8192 |
| `attn_implementation` | attention 后端：`"flex_attention"` 或 `"sdpa"` | `"flex_attention"` |

`output_dir` 和 `data_config` 通过命令行传入（见下文）。

## Attention 实现

默认情况下，训练使用 `flex_attention`，它需要 PyTorch >= 2.5 和兼容 GPU（例如 NVIDIA Ampere 或更新架构）。如果你的环境不支持 `flex_attention`，请在训练配置中把 `attn_implementation` 设置为 `"sdpa"`。可参考 [examples/config/train_config_finetune_sdpa.json](../examples/config/train_config_finetune_sdpa.json) 中可直接使用的 SDPA 配置：

```json
{
    "attn_implementation": "sdpa",
    "max_sample_tokens": 2000,
    "min_sample_tokens": 50,
    "max_batch_size": 64
}
```

`"sdpa"` 使用 PyTorch 内置的 scaled dot-product attention（缩放点积注意力），适用于更广泛的硬件。

以下字段只在 `attn_implementation != "flex_attention"` 时生效：

| 字段 | 说明 | 默认值 |
|---|---|---|
| `max_sample_tokens` | 每个样本的最大 token 长度；更长样本会被丢弃 | 2000 |
| `min_sample_tokens` | 每个样本的最小 token 长度；更短样本会被丢弃 | 50 |
| `max_batch_size` | 每个 batch 的样本数上限 | 64 |

`batch_tokens` 仍然是控制显存使用的主要参数，它设置每个 batch 的总 token 预算。`max_batch_size` 是安全保护，用来避免大量短样本组成的 batch 产生异常大的 batch 维度。

### Batching 策略

两个后端使用**不同的 batching 策略**，会自动选择：

| 后端 | Batching 策略 | Batch shape | 说明 |
|---|---|---|---|
| `flex_attention` | Sequence packing（序列打包） | `[1, C, batch_tokens]` | 多个样本拼接成一个长序列；通过 `document_ids` 跟踪文档边界 |
| `sdpa` | Length-grouped padding（按长度分组后 padding） | `[B, C, max_len]` | token 长度相近的样本分到同一个 batch，并 padding 到该 batch 的局部最大长度 |

**为什么策略不同？**

- 使用 `flex_attention` 时，sequence packing 更省显存，因为紧凑的 `BlockMask`（不是稠密矩阵）描述了不同文档边界之间哪些 token 可以互相 attend。
- 使用 `sdpa` 时，改用 length-grouped padding：token 长度相近的样本一起组成 batch，并 padding 到局部最大长度，因此只需要轻量的 `[B, 1, max_len, max_len]` 布尔 attention mask，额外开销低且 padding 浪费少。

## 启动训练

```bash
accelerate launch \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --num_processes 8 \
    -m omnivoice.cli.train \
    --train_config config/train_config_emilia.json \
    --data_config config/data_config_emilia.json \
    --output_dir exp/omnivoice_emilia
```

## 恢复训练

在训练配置中设置 `resume_from_checkpoint`，即可从已有 checkpoint 恢复训练：

```json
{
    "resume_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## 从预训练模型初始化

如果要从预训练 OmniVoice checkpoint 开始训练（用于 fine-tuning，微调）：

```json
{
    "init_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## 监控

训练日志会写入 TensorBoard：

```bash
tensorboard --logdir exp/omnivoice_emilia/tensorboard
```
