#!/bin/bash

# This script demonstrates how to fine-tune OmniVoice from a JSONL manifest.
#
# 中文说明:
#   这个脚本演示如何用自己的 JSONL 数据微调 OmniVoice 主模型。
#   微调流程分两步:
#     1. 先用 audio tokenizer 把音频文件转换成训练用的 audio token,
#        并打包成 WebDataset shards。
#     2. 再调用 omnivoice.cli.train, 从预训练 checkpoint 初始化并继续训练。
#   注意: 这里微调的是 OmniVoice 主模型; audio tokenizer 只用于预处理音频,
#   不会在这个脚本里一起被微调。

set -euo pipefail

# Stage control.
# 中文说明:
#   stage / stop_stage 用来控制执行范围, 方便断点续跑。
#   stage=0, stop_stage=0: 只做音频 token 化。
#   stage=1, stop_stage=1: 跳过 token 化, 只启动微调。
#   stage=0, stop_stage=1: 从头跑完整微调流程。
stage=0
stop_stage=1

# ====== Modify as needed ======
# GPUs to use.
# 中文说明:
#   GPU_IDS 指定可见 GPU 编号; NUM_GPUS 要和 GPU_IDS 里的数量一致。
#   例如只用一张卡可以写 GPU_IDS="0", NUM_GPUS=1。
GPU_IDS="0,1"
NUM_GPUS=2

# Path to your input JSONL file.
# (each line: {"id": ..., "audio_path": ..., "text": ..., "language_id": ...})
# 中文说明:
#   训练集 JSONL 路径。每一行是一条音频样本, 至少包含:
#     id: 样本 ID
#     audio_path: 音频文件路径
#     text: 音频对应文本
#   language_id 可选, 如 "zh" / "en"。
TRAIN_JSONL="data/my_data_train.jsonl"

# Path to your dev JSONL file. Set to empty string to skip dev set.
# 中文说明:
#   验证集 JSONL 路径。没有验证集时可以设为 DEV_JSONL=""。
DEV_JSONL="data/my_data_dev.jsonl"

# Directory to write tokenized WebDataset shards.
# 中文说明:
#   audio token 和对应文本索引会写到这里。
#   后续 data_config_finetune.json 会引用这里生成的 data.lst。
TOKEN_DIR="data/finetune/tokens"

# Audio tokenizer model (HuggingFace repo or local path).
# 中文说明:
#   用来把 waveform 编码成 8 层 audio codebook token。
#   可以是 Hugging Face repo, 也可以替换成本地 audio_tokenizer 目录。
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

# Training config file.
# If you encounter issues with flex_attention on your GPU, use the SDPA config instead:
# TRAIN_CONFIG="config/train_config_finetune_sdpa.json"
# 中文说明:
#   默认配置使用 flex_attention, 更省显存但对环境要求更高。
#   如果 GPU / PyTorch 不支持 flex_attention, 改用 train_config_finetune_sdpa.json。
TRAIN_CONFIG="config/train_config_finetune.json"

# Data config file.
# 中文说明:
#   这里记录训练集 / 验证集 token 数据 manifest 的位置。
#   默认文件指向 TOKEN_DIR 下的 train/dev data.lst。
data_config="config/data_config_finetune.json"

# Output directory for fine-tuned checkpoints.
# 中文说明:
#   微调产生的 checkpoint 会保存到这里, 例如 checkpoint-500。
OUTPUT_DIR="exp/omnivoice_finetune"
# =================================

# Add project root to PYTHONPATH so `python -m omnivoice...` works
# when this script is launched from the examples directory.
# 中文说明:
#   把项目根目录加入 PYTHONPATH, 确保脚本能找到本地 omnivoice 包。
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"


# Stage 0: Tokenize audio into WebDataset shards
# 中文说明:
#   第 0 阶段: 读取 JSONL 中的音频路径, 加载音频, 通过 audio tokenizer
#   提取 audio token, 再写成 WebDataset 训练数据。
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Tokenizing audio"

    for split_jsonl_path in ${TRAIN_JSONL} ${DEV_JSONL}; do
        if [ -z "${split_jsonl_path}" ]; then
            continue
        fi

        if [ "${split_jsonl_path}" = "${TRAIN_JSONL}" ]; then
            split="train"
        else
            split="dev"
        fi

        echo "  Tokenizing ${split} from ${split_jsonl_path}"

        # extract_audio_tokens writes:
        #   - tar shards under ${TOKEN_DIR}/${split}/audios/
        #   - text JSONL shards under ${TOKEN_DIR}/${split}/txts/
        #   - a data.lst manifest consumed by the training data config.
        # 中文说明:
        #   这个命令会把原始音频转换成 audio token, 并生成训练读取所需的 data.lst。
        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            python -m omnivoice.scripts.extract_audio_tokens \
            --input_jsonl "${split_jsonl_path}" \
            --tar_output_pattern "${TOKEN_DIR}/${split}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 3 \
            --shuffle True

        echo "  Done. Manifest written to ${TOKEN_DIR}/${split}/data.lst"
    done
fi


# Stage 1: Fine-tune
# 中文说明:
#   第 1 阶段: 启动 Accelerate 分布式训练。
#   具体是否从预训练模型初始化, 由 TRAIN_CONFIG 里的 init_from_checkpoint 控制。
#   官方微调配置中 init_from_checkpoint="k2-fsa/OmniVoice"。
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Fine-tuning"

    accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        -m omnivoice.cli.train \
        --train_config ${TRAIN_CONFIG} \
        --data_config ${data_config} \
        --output_dir ${OUTPUT_DIR}
fi
