# `omnivoice.data` 数据处理模块

该目录负责为 OmniVoice 的训练和评估准备数据。它不执行模型的 forward、backward 或权重更新，而是把磁盘中的原始样本逐步转换成模型可以直接接收的批量 Tensor。

整体数据链路：

```text
数据配置与原始文件
→ dataset.py：读取、解码单条原始样本
→ processor.py：将单条样本转换成训练字段
→ batching.py：按长度分组或打包多条样本
→ collator.py：整理成批量 Tensor
→ DataLoader
→ trainer.py：使用 batch 训练或评估模型
```

## `dataset.py`：读取原始数据

该文件负责确定数据从哪里来，以及如何把数据读取、解码成单条 Python 样本。

主要内容：

- `prepare_data_manifests_from_json()`：读取数据配置 JSON，生成训练集和验证集的 manifest 列表。
- `SampleDecoder`：解码样本中的音频、音频 Token 和标签等字段。
- `WebDatasetReader`：从 WebDataset 的 tar 分片中流式读取音频、文本等数据。
- `JsonlDatasetReader`：从 JSONL 文件中流式读取样本，主要供数据处理脚本使用。
- `MuxWebDatasetReader`：混合多个 WebDataset 数据源，可用于多语言或多数据集训练。
- `WrappedIterableDataset`：为其他流式 Dataset 包装器提供基础能力。

它输出的仍然是**单条原始样本**，还不是最终送入模型的 batch。

## `processor.py`：处理单条样本

该文件负责把一条原始音频/文本样本转换成模型训练需要的字段。

`OmniVoiceSampleProcessor` 会完成：

- 使用文本 tokenizer 把文本转换成 Token ID。
- 组合文本 Token、音频 Token、语言和 instruct 等信息。
- 划分 prompt 与目标生成区域。
- 按训练配置随机生成 mask 或丢弃部分条件。
- 构造模型训练标签。

主要输出：

```text
input_ids  ：模型输入 Token，形状为 [C, L]
labels     ：训练目标，形状为 [C, L]
audio_mask ：标记音频区域，形状为 [L]
length     ：当前样本的 Token 总长度
```

其中 `C` 是音频 codebook 层数，`L` 是当前样本的序列长度。

`OmniVoiceSimpleSampleProcessor` 是简化版本，当前正式训练链路使用的是 `OmniVoiceSampleProcessor`。

## `batching.py`：决定哪些样本组成一个 batch

语音样本长度差异较大。该文件根据样本长度组织 batch，减少无效 padding，并控制每个 batch 的 Token 数量。

提供两种策略：

### `PackingIterableDataset`

将多条处理后的样本装入一个固定 Token 预算的序列中。

```text
sample1 + sample2 + sample3 → 一条长序列
```

该策略与 `flex_attention` 配合使用。

### `StreamLengthGroupDataset`

把长度接近的样本放入同一个 batch：

```text
短样本与短样本一组
长样本与长样本一组
```

这样补齐长度时产生的 padding 更少。该策略用于 SDPA、eager 等非 `flex_attention` 训练路径。

这个文件主要决定**样本如何分组**，最终的批量 Tensor 由 `collator.py` 创建。

## `collator.py`：生成批量 Tensor

该文件接收已经分组的样本，将它们整理成形状统一、可以送入模型的 Tensor 字典。

### `PackingDataCollator`

把多条样本拼接成一条长序列，主要输出形状为：

```text
[1, C, L]
```

它还会生成 document IDs、position IDs 和 attention mask，避免不同样本之间发生错误的信息交互。

### `PaddingDataCollator`

把同一 batch 内的样本补齐到该 batch 的最大长度，再沿 batch 维堆叠：

```text
[B, C, max_len]
```

其中：

- `B`：当前 batch 的样本数。
- `C`：音频 codebook 层数。
- `max_len`：当前 batch 中最长样本的序列长度。

该方式用于 SDPA、eager 等常规 attention 实现。

## `__init__.py`：Python 包标记

该文件用于把 `omnivoice/data` 标记为 Python 包，当前没有额外业务逻辑。

## 与训练模块的关系

`omnivoice/training/builder.py` 负责把这些模块连接起来：

```text
WebDatasetReader
→ OmniVoiceSampleProcessor
→ PackingIterableDataset 或 StreamLengthGroupDataset
→ PackingDataCollator 或 PaddingDataCollator
→ DataLoader
```

`DataLoader` 产出的 batch 最终由 `omnivoice/training/trainer.py` 取出，并传入模型执行 forward、loss 计算和权重更新。
