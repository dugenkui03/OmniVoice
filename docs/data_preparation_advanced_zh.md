# 高级数据准备

高级流程在基础 tokenization（token 化）流程之上增加了 **denoising（去噪）** 和 **prompt noise augmentation（提示音频噪声增强）**。每个阶段都是可选的。

## 前置条件

- **Denoising（去噪）**：来自 <https://huggingface.co/sarulab-speech/sidon-v0.1/tree/main> 的 Sidon 模型 checkpoints（`feature_extractor_cuda.pt`、`decoder_cuda.pt`）。
- **Noise augmentation（噪声增强）**：带 `data.lst` manifest 的 noise + RIR tar shards。

## 流程概览

```text
Step 1（可选）：去噪
  原始音频 -> Sidon denoiser -> 干净音频

Step 2：Tokenize（可选噪声增强）
  干净音频 + 前缀噪声增强 -> audio tokenizer -> tokens
```

## 去噪

使用 [Sidon](https://github.com/sarulab-speech/Sidon) speech enhancement（语音增强）模型移除原始音频中的背景噪声。

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python -m omnivoice.scripts.denoise_audio \
    --input_jsonl data.jsonl \
    --tar_output_pattern data/denoised/audios/shard-%06d.tar \
    --jsonl_output_pattern data/denoised/txts/shard-%06d.jsonl \
    --feature_extractor_path /path/to/sidon_feature_extractor_cuda.pt \
    --decoder_path /path/to/sidon_decoder_cuda.pt \
    --target_sample_rate 24000 \
    --batch_duration 200.0
```

它会做：

1. 读取你的 JSONL manifest
2. 对每个音频文件运行 Sidon denoiser（去噪器）
3. 以自定义 WebDataset tar/jsonl shards 输出去噪后的音频
4. 在 `data/denoised/` 下生成 `data.lst` manifest

> 如果你已经有自定义 WebDataset 格式数据集，也可以传入 `--input_manifest /path/to/data.lst`。
> 下一步是把生成的 `data.lst` 文件通过 `--input_manifest` 传给 `omnivoice.scripts.extract_audio_tokens`，用于提取 tokens。

### 带噪声增强的 tokenization

在 tokenization 期间向 **prompt audio（提示音频）** 添加环境噪声和房间混响，使模型在推理时面对有噪参考音频也更稳健。注意，在我们的模型中，只对一小部分数据添加噪声增强，以确保模型也能在干净参考音频下生成高质量音频。

你需要两个额外的 WebDataset 格式数据集：

- **Noise recordings（噪声录音）**：带 `data.lst` manifest 的环境噪声 tar shards
- **Room impulse responses (RIR，房间脉冲响应)**：带 `data.lst` manifest 的 RIR tar shards

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,4"
python -m omnivoice.scripts.extract_audio_tokens_add_noise \
    --input_jsonl data.jsonl \
    --tar_output_pattern data/tokens/shard-%06d.tar \
    --jsonl_output_pattern data/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --noise_manifest data/noise_shards/data.lst \
    --rir_manifest data/rir_shards/data.lst \
    --nj_per_gpu 3
```

> 如果你已经有自定义 WebDataset 格式数据集，也可以传入 `--input_manifest /path/to/data.lst`。
