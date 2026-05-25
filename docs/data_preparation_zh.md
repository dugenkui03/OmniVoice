# 数据准备

OmniVoice 使用自定义 WebDataset 格式训练：音频数据被打包进 **tar shards**（tar 分片），并配套 **JSONL metadata**（JSONL 元数据）文件。每个 tar shard 包含数百到数千个样本（以 `.npy` 音频 token 数组形式保存），可以在训练时大幅减少磁盘 I/O。分离的 jsonl 文件则便于修改元数据。本文详细说明数据格式，并给出完整准备流程。

## 1. 输入格式

准备一个 JSONL 文件，每一行都是一个 JSON 对象：

```jsonl
{"id": "sample_001", "audio_path": "/data/audio/001.wav", "text": "Hello world", "language_id": "en"}
{"id": "sample_002", "audio_path": "/data/audio/002.wav", "text": "你好世界", "language_id": "zh"}
```

字段：

- `id` — 唯一样本标识符（用于在 shard 和标签文件之间匹配样本）
- `audio_path` — 音频文件的绝对路径（wav/flac/mp3，会被重采样到 24 kHz）
- `text` — 转写文本
- `language_id` — 可选语言代码，用于多语言训练，可以省略

## 2. 处理流程

tokenization（token 化）脚本 `extract_audio_tokens.py` 会把音频转换成 8 层离散 token，并打包成 WebDataset shards。

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,4"  # 用于 token 提取的 GPU
python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl data.jsonl \
    --tar_output_pattern output/audios/shard-%06d.tar \
    --jsonl_output_pattern output/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3 \
    --shuffle True
```

它会做：

1. 读取你的 JSONL manifest（清单）
2. 使用 audio tokenizer（音频分词器）把每个音频文件编码成离散 token
3. 将 token 打包进 WebDataset tar shards，并生成配套 jsonl 元数据文件
4. 生成 `data.lst` manifest 文件

<details>
<summary><strong>替代方案：</strong>WebDataset 输入（如果你已经有原始音频 tar shards）</summary>

传入 `data.lst` manifest，而不是 `--input_jsonl`：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,4"  # 用于 token 提取的 GPU
python -m omnivoice.scripts.extract_audio_tokens \
    --input_manifest existing_data/data.lst \
    --tar_output_pattern output/audios/shard-%06d.tar \
    --jsonl_output_pattern output/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3 \
    --shuffle True
```

`existing_data/data.lst` 可通过下面命令生成：

```bash
python -m omnivoice.scripts.jsonl_to_webdataset \
    --input data.jsonl \
    --output data/shards \
    --sr 24000 \
    --shard-size 1000
```

该命令会把音频重采样到目标采样率，并将 FLAC 文件打包进 tar shards，同时生成配套 jsonl 元数据文件。

</details>

### 脚本参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_manifest` | None | 输入数据集 manifest（`data.lst`）路径，和 `--input_jsonl` 互斥 |
| `--input_jsonl` | None | 原始 JSONL 文件路径，和 `--input_manifest` 互斥 |
| `--tar_output_pattern` | 必填 | Tar shard 输出 pattern，例如 `output/audios/shard-%06d.tar` |
| `--jsonl_output_pattern` | 必填 | JSONL shard 输出 pattern，例如 `output/txts/shard-%06d.jsonl` |
| `--tokenizer_path` | `eustlb/higgs-audio-v2-tokenizer` | Hugging Face tokenizer 路径或本地路径 |
| `--nj_per_gpu` | 3 | 每张 GPU 的 worker 进程数 |
| `--loader_workers` | 24 | 用于流式 `IterableDataset` 的 DataLoader worker 数 |
| `--shuffle` | True | 分片前是否打乱样本 |
| `--shuffle-seed` | 42 | 打乱样本的随机种子 |
| `--samples_per_shard` | 1000 | 每个 tar shard 的最大样本数 |
| `--min_num_shards` | 32 | 最小输出 shard 数量（确保 shard 数 >= num_gpu × num_workers） |
| `--min_length` | 0.0 | 跳过短于该时长的音频，单位为秒 |
| `--max_length` | inf | 跳过长于该时长的音频，单位为秒 |
| `--skip_errors` | False | 处理失败时继续执行，而不是中止 |
| `--num_machines` | 1 | 分布式运行的机器总数 |
| `--machine_index` | 0 | 分布式预处理时从 0 开始的机器编号 |

### 输出结构

使用下面输出 pattern：

```bash
--tar_output_pattern output/audios/shard-%06d.tar \
--jsonl_output_pattern output/txts/shard-%06d.jsonl
```

会得到：

```text
output/
├── audios/                    # WebDataset tar shards（音频 token）
│   ├── shard-000000.tar       # 每个 tar 大约打包 1000 个样本
│   ├── shard-000001.tar
│   └── ...
├── txts/                      # 每个 shard 对应的 JSONL 标签文件
│   ├── shard-000000.jsonl     # 对应 tar 中每个样本一行 JSON
│   ├── shard-000001.jsonl
│   └── ...
├── data.lst                   # 连接 tar ↔ jsonl shard 的 manifest
└── errors.jsonl               # 处理失败的样本（如果有）
```

`data.lst` 和 `errors.jsonl` 会写到 `audios/` 和 `txts/` 的**父目录**。

### `data.lst` manifest

`data.lst` 中每一行描述一个 shard：

```text
/path/to/shard-000000.tar /path/to/shard-000000.jsonl 1000 3600.500
/path/to/shard-000001.tar /path/to/shard-000001.jsonl 800 2880.200
```

格式：`<tar_path> <jsonl_path> <num_samples> <total_duration_seconds>`

- 路径是**绝对路径**
- `.tar` 文件包含音频 token
- `.jsonl` 文件包含原始 JSONL 文件中的元数据，便于不解压 tar 文件就访问和修改元数据
- 训练数据配置引用的就是这个 manifest

### tar shard 内部

每个 `.tar` 文件会把**大量样本**（默认每个 shard 1000 个）打包进一个归档文件。这是 WebDataset 的关键优势：dataloader 不需要读取成千上万个小文件，而是顺序读取少量大 tar，从而显著降低磁盘 I/O 压力。

tar 中每个样本是一组 key 匹配的文件：

```text
shard-000000.tar:
  sample_001.npy    # 音频 token：numpy 数组，shape [8, T]，dtype int16
  sample_002.npy
  ...
  sample_1000.npy
```

## 3. 训练用数据配置

创建 WebDataset shards 后，写一个 data config JSON 来引用它们：

```json
{
    "train": [
        {
            "language_id": "en",
            "manifest_path": ["data/custom/tokens/train/data.lst"],
            "repeat": 1
        }
    ],
    "dev": [
        {
            "language_id": "en",
            "manifest_path": ["data/custom/tokens/dev/data.lst"],
            "repeat": 1
        }
    ]
}
```

- `manifest_path` — `data.lst` 文件列表（每个 shard 目录一个）
- `repeat` — 每个 epoch 中重复该数据集的次数（用于平衡多语言数据）
- `language_id` 当前不会被使用，只是为了更好地组织数据

可参考 [examples/config/](../examples/config/) 中可直接使用的数据配置文件。

> 去噪和噪声增强请参考 [docs/data_preparation_advanced_zh.md](../docs/data_preparation_advanced_zh.md)。
