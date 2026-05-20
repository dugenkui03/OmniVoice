# OmniVoice 🌍

<p align="center">
  <img width="200" height="200" alt="OmniVoice" src="https://zhu-han.github.io/omnivoice/pics/omnivoice.jpg" />
</p>

<p align="center">
  <a href="https://huggingface.co/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.00688"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
  &nbsp;
  <a href="https://zhu-han.github.io/omnivoice"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
  &nbsp;
  <a href="https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

OmniVoice 是一个业界领先的大规模多语言零样本（zero-shot）文本到语音（TTS）模型，支持 **600+ 种语言**。它采用了一种新颖的扩散语言模型（diffusion language model）架构，在生成高质量语音的同时具备极快的推理速度，并支持**音色克隆**与**音色设计**两种工作方式。

**目录**：[核心特性](#核心特性) | [安装](#安装) | [快速开始](#快速开始) | [Python API](#python-api) | [命令行工具](#命令行工具) | [训练与评测](#训练与评测) | [交流讨论](#交流讨论) | [引用](#引用)

## 核心特性

- **支持 600+ 种语言**：在零样本 TTS 模型中拥有最广的语言覆盖（[完整语言列表](docs/languages.md)）。
- **音色克隆（Voice Cloning）**：业界领先的克隆音质。
- **音色设计（Voice Design）**：通过指定的说话人属性（性别、年龄、音高、方言/口音、耳语等）控制声音风格。
- **细粒度控制**：支持非言语标签（如 `[laughter]`）以及通过拼音或音素进行的发音纠正。
- **快速推理**：RTF 可低至 0.025（约为实时的 40 倍速）。
- **扩散语言模型式架构**：简洁、流畅、可扩展的设计，兼顾质量与速度。

---

## 安装

下面 **pip** 与 **uv** 两种方式 **任选其一** 即可。

### pip

> 建议在一个干净的虚拟环境（如 `conda`、`venv` 等）里安装，避免依赖冲突。

**第 1 步**：安装 PyTorch

<details>
<summary>NVIDIA GPU</summary>

```bash
# 按你的 CUDA 版本安装对应 PyTorch，例如：
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```
> 其他版本请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

</details>

<details>
<summary>Apple Silicon</summary>

```bash
pip install torch==2.8.0 torchaudio==2.8.0
```

</details>

**第 2 步**：安装 OmniVoice（任选一种）

```bash
# 从 PyPI 安装（稳定版）
pip install omnivoice

# 直接从 GitHub 最新源码安装（无需手动 clone）
pip install git+https://github.com/k2-fsa/OmniVoice.git

# 用于开发（先 clone，再以可编辑模式安装）
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
pip install -e .
```

### uv

clone 仓库后直接同步依赖即可：

```bash
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
uv sync
```

> **提示**：如果在国内网络下载较慢，可以使用镜像：`uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"`

---

## 快速开始

不写一行代码也可以体验 OmniVoice：

- 在本地启动 Web UI：`omnivoice-demo --ip 0.0.0.0 --port 8001`

- 或者直接在 [HuggingFace Space](https://huggingface.co/spaces/k2-fsa/OmniVoice) 上试用

- 也可以在 Google Colab 里运行：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb)

> 如果下载预训练模型时连接 HuggingFace 有问题，可以先 `export HF_ENDPOINT="https://hf-mirror.com"` 再运行。

完整用法请参考下面的 [Python API](#python-api) 和 [命令行工具](#命令行工具) 两节。

---

## Python API

OmniVoice 支持三种生成模式，本节的所有功能也都可以通过 [命令行工具](#命令行工具) 使用。

### 音色克隆（Voice Cloning）

从一段较短的参考音频中克隆音色，提供 `ref_audio` 和 `ref_text` 即可：

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)
# Apple Silicon 用户请改用 device_map="mps"

audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    ref_text="Transcription of the reference audio.",
)  # audio 是 list[np.ndarray]，每条形状为 (T,)，采样率 24 kHz

# 如果不想手动填写 ref_text，可以直接省略它，
# 模型会自动调用 Whisper ASR 来转写参考音频。

sf.write("out.wav", audio[0], 24000)
```

> **小贴士**
>
> - 推荐使用 3–10 秒的参考音频。过长的音频会拖慢推理速度，并可能降低克隆质量。
> - 想要标准发音，请使用与目标语种**相同语言**的参考音频。在跨语种克隆（参考音频与目标文本语种不同）的场景下，生成的语音会带有参考音频对应语种的口音。
> - 对于阿拉伯数字，建议先做文本规范化（例如把 `"123"` 转成 `"one hundred twenty-three"`），可以使用 [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) 等工具。
>
> 更多技巧请见 [docs/tips.md](docs/tips.md)。

### 音色设计（Voice Design）

直接用文字描述目标声音的说话人属性，无需任何参考音频。支持的属性包括：**性别**（male/female）、**年龄**（child 到 elderly）、**音高**（very low 到 very high）、**风格**（whisper 等）、**英文口音**（American、British 等）、**中文方言**（四川话、陕西话等）。多个属性用英文逗号分隔，可自由组合。

```python
audio = model.generate(
    text="Hello, this is a test of zero-shot voice design.",
    instruct="female, low pitch, british accent",
)
```

> **说明**：Voice Design 仅在中文和英文数据上训练。它可以泛化到其他语种，但在部分低资源语种上效果可能不稳定。

完整的属性参考、中文对应词与使用技巧请见 [docs/voice-design.md](docs/voice-design.md)。

### 自动音色（Auto Voice）

不指定任何提示，让模型自动选一个音色：

```python
audio = model.generate(text="This is a sentence without any voice prompt.")
```

### 生成参数

上述三种模式共用同一个 `model.generate()` API。你可以通过关键字参数进一步控制生成行为：

```python
audio = model.generate(
    text="...",
    num_step=32,   # 扩散步数（追求更快推理可用 16）
    speed=1.0,     # 语速因子（>1.0 更快，<1.0 更慢）
    duration=10.0, # 固定输出时长（秒），优先级高于 speed
    # ... 更多参数
)
```

更详细的控制说明请见 [docs/generation-parameters.md](docs/generation-parameters.md)。

### 非言语符号与发音控制

OmniVoice 支持在输入文本中直接内联 **非言语符号** 和 **发音纠正**。

**非言语符号**：直接在文本中插入 `[laughter]` 这类标签即可加入富表现力的非言语声音。

```python
audio = model.generate(text="[laughter] You really got me. I didn't see that coming at all.")
```

支持的标签：`[laughter]`、`[sigh]`、`[confirmation-en]`、`[question-en]`、`[question-ah]`、`[question-oh]`、`[question-ei]`、`[question-yi]`、`[surprise-ah]`、`[surprise-oh]`、`[surprise-wa]`、`[surprise-yo]`、`[dissatisfaction-hnn]`。

**中文发音纠正**：使用带声调数字的拼音来纠正特定字的发音。

```python
audio = model.generate(text="这批货物打ZHE2出售后他严重SHE2本了，再也经不起ZHE1腾了。")
```

**英文发音纠正**：使用 [CMU 发音词典](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a)（大写，放在方括号中）覆盖默认的英文发音。

```python
audio = model.generate(text="He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.")
```

---

## 命令行工具

项目提供了 3 个命令行入口。这些 CLI 工具支持 Python API 中的所有功能（音色克隆、音色设计、自动音色、生成参数等），全部通过命令行参数进行控制。

| 命令 | 说明 | 源码 |
|---|---|---|
| `omnivoice-demo` | 交互式 Gradio Web Demo | [omnivoice/cli/demo.py](omnivoice/cli/demo.py) |
| `omnivoice-infer` | 单条推理 | [omnivoice/cli/infer.py](omnivoice/cli/infer.py) |
| `omnivoice-infer-batch` | 跨多 GPU 的批量推理 | [omnivoice/cli/infer_batch.py](omnivoice/cli/infer_batch.py) |

### Demo

```bash
omnivoice-demo --ip 0.0.0.0 --port 8001
```

会提供一个用于音色克隆和音色设计的 Web UI。完整参数请执行 `omnivoice-demo --help` 查看。

### 单条推理

```bash
# 音色克隆
# ref_text 可省略（会由 Whisper 自动转写 ref_audio 得到）。
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --ref_audio ref.wav \
    --ref_text "Transcription of the reference audio." \
    --output hello.wav

# 音色设计
omnivoice-infer --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --instruct "male, British accent" \
    --output hello.wav

# 自动音色
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech."\
    --output hello.wav
```

### 批量推理

`omnivoice-infer-batch` 可以把批量推理分布到多张 GPU 上，专为大规模 TTS 任务设计。

```bash
omnivoice-infer-batch \
    --model k2-fsa/OmniVoice \
    --test_list test.jsonl \
    --res_dir results/
```

测试列表是一个 JSONL 文件，每一行是一个 JSON 对象：
```json
{"id": "sample_001", "text": "Hello world", "ref_audio": "/path/to/ref.wav", "ref_text": "Reference transcript", "instruct": "female, british accent", "language_id": "en", "duration": 10.0, "speed": 1.0}
```
只有 `id` 和 `text` 是必填字段。`ref_audio` 和 `ref_text` 用于音色克隆模式；`instruct` 用于音色设计模式。如果既不提供参考音频也不提供 instruct，模型会以一个随机音色生成。

`language_id`、`duration` 和 `speed` 都是可选项。`duration`（秒）用来固定输出时长；`speed` 控制语速。如果同时提供了 `duration` 和 `speed`，`speed` 会被忽略。

---

## 训练与评测

完整的流水线（从数据准备到训练、评测和微调）请见 [examples/](examples/) 目录。

---

## 交流讨论

可以直接在 [GitHub Issues](https://github.com/k2-fsa/OmniVoice/issues) 上发起讨论。

也可以扫描下方二维码加入我们的微信群或关注微信公众号。

| 微信群 | 微信公众号 |
| ------ | ---------- |
| ![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_group.jpg) | ![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_account.jpg) |

---

## 社区项目

OmniVoice 拥有一个不断壮大的社区生态。
更多内容请见 [Community Projects](docs/community-projects.md)。

---

## 引用

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```

---

## 免责声明

严禁将本模型用于未经授权的音色克隆、声音冒充、欺诈、诈骗以及任何其他违法或违反道德伦理的活动。所有用户须确保完全遵守适用的当地法律法规与伦理标准。开发者不对本模型的任何滥用承担责任，并倡导负责任地开发和使用 AI，鼓励社区在 AI 研究与应用中坚守安全与伦理原则。
