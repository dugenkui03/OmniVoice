# OmniVoice 示例

本目录包含用于训练、微调和评估 OmniVoice 的脚本与配置文件。

| 使用场景 | 脚本 | 说明 |
|---|---|---|
| 从头训练 | [run_emilia.sh](run_emilia.sh) | 在 Emilia 数据集上运行完整流程：数据检查、音频 token 化、训练 |
| 微调 | [run_finetune.sh](run_finetune.sh) | 使用自己的 JSONL 数据，从预训练 checkpoint 开始微调 |
| 评估 | [run_eval.sh](run_eval.sh) | 在标准测试集上评估 WER、说话人相似度和 UTMOS |

---

## 从头训练（Emilia）

[run_emilia.sh](run_emilia.sh) 会分 3 个阶段运行完整流程：

| 阶段 | 做什么 |
|---|---|
| 0 | 检查 Emilia 数据集和 JSONL manifest 是否已经准备好 |
| 1 | 将音频 token 化，并打包成 WebDataset shards |
| 2 | 使用 `accelerate` 启动多 GPU 训练 |

**前置条件：**

1. 从 [OpenXLab](https://openxlab.org.cn/datasets/Amphion/Emilia) 下载 Emilia 数据集，并放到 `download/` 目录下：
   ```text
   download/Amphion___Emilia
   └── raw
       ├── EN
       └── ZH
   ```
2. 准备 JSONL manifests，并放到 `data/emilia/manifests/`：
   - `emilia_en_train.jsonl`、`emilia_en_dev.jsonl`
   - `emilia_zh_train.jsonl`、`emilia_zh_dev.jsonl`

   这些 manifest 可以从原始数据生成，也可以从 [HuggingFace](https://huggingface.co/datasets/zhu-han/Emilia-Manifests) 下载已经预处理好的版本。

**运行完整流程：**

```bash
bash examples/run_emilia.sh
```

也可以通过修改脚本顶部的 `stage` 和 `stop_stage` 单独运行某个阶段。例如设置 `stage=1`、`stop_stage=1` 时，只运行音频 token 化阶段。

> 配置细节、checkpoint 恢复训练和 TensorBoard 监控请参考 [docs/training.md](../docs/training.md)。

---

## 微调

[run_finetune.sh](run_finetune.sh) 会使用你自己的数据，从预训练 checkpoint 开始微调。

### 第 1 步：准备数据

创建一个 JSONL manifest，每一行描述一个音频样本：

```jsonl
{"id": "sample_001", "audio_path": "/data/audio/001.wav", "text": "Hello world", "language_id": "en"}
{"id": "sample_002", "audio_path": "/data/audio/002.wav", "text": "你好世界", "language_id": "zh"}
```

`id`、`audio_path` 和 `text` 是必填字段。`language_id` 是可选字段。

> 完整数据格式说明请参考 [docs/data_preparation.md](../docs/data_preparation.md)。

### 第 2 步：配置脚本

编辑 `run_finetune.sh` 顶部的变量：

```bash
TRAIN_JSONL="data/my_data_train.jsonl"   # 训练集 JSONL 路径
DEV_JSONL="data/my_data_dev.jsonl"       # 验证集 JSONL 路径
GPU_IDS="0,1"                            # 使用哪些 GPU
NUM_GPUS=2
OUTPUT_DIR="exp/omnivoice_finetune"      # 输出目录
```

### 第 3 步：运行

```bash
bash examples/run_finetune.sh
```

这个脚本会执行两件事：

1. 将你的音频 token 化，并打包成 WebDataset shards
2. 使用 `accelerate` 启动微调

微调配置 [config/train_config_finetune.json](config/train_config_finetune.json) 和 Emilia 训练配置 [config/train_config_emilia.json](config/train_config_emilia.json) 的主要区别如下：

| 参数 | Emilia（从头训练） | 微调 | 原因 |
|---|---|---|---|
| `init_from_checkpoint` | `null` | `"k2-fsa/OmniVoice"` | 加载预训练权重 |
| `steps` | 300,000 | 5,000 | 微调步数更少，可根据数据和任务调整 |
| `learning_rate` | 1e-4 | 5e-5 | 微调时使用更低学习率，可根据数据和任务调整 |

如果要使用其他预训练 checkpoint，可以修改配置文件中的 `init_from_checkpoint`。

如果你的 GPU 在使用 `flex_attention` 时遇到问题，可以改用 [config/train_config_finetune_sdpa.json](config/train_config_finetune_sdpa.json)。这个配置使用 SDPA attention，硬件兼容性更好。详情请参考 [docs/training.md](../docs/training.md#attention-implementation)。

---

## 评估

先安装评估依赖：

```bash
pip install omnivoice[eval]
# 或者
uv sync --extra eval
```

支持的测试集包括：`librispeech_pc`、`seedtts_en`、`seedtts_zh`、`fleurs`、`minimax`。

```bash
bash examples/run_eval.sh
```

> 指标细节、测试集准备和单独运行某个指标的方法，请参考 [docs/evaluation.md](../docs/evaluation.md)。
