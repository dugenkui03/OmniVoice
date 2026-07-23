#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core OmniVoice model implementation.

Defines the ``OmniVoice`` model class, generation config, and inference pipeline.
This is the main entry point for both inference and training:

- **Inference**: ``OmniVoice.from_pretrained()`` loads the model, then
  ``model.generate()`` supports voice cloning, voice design, and auto voice.
- **Training**: ``model.forward()`` computes the training loss; the model is
  built and used by ``omnivoice.training.builder`` and ``omnivoice.training.trainer``.

中文翻译：

定义 ``OmniVoice`` 模型类、生成配置以及推理流水线。
本文件是推理和训练的主入口：

- **推理**：``OmniVoice.from_pretrained()`` 加载模型，随后调用
  ``model.generate()`` 支持音色克隆（voice cloning）、音色设计（voice design）
  以及自动音色（auto voice）。
- **训练**：``model.forward()`` 计算训练损失；模型由
  ``omnivoice.training.builder`` 构建，并由 ``omnivoice.training.trainer`` 使用。

"""

import difflib
import logging
import math
import os
import re
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

try:
    from torch.nn.attention.flex_attention import create_block_mask

    _flex_attention_available = True
except ImportError:
    _flex_attention_available = False
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HiggsAudioV2TokenizerModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

from omnivoice.utils.audio import (
    cross_fade_chunks,
    fade_and_pad_audio,
    load_audio,
    remove_silence,
    trim_long_audio,
)
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES
from omnivoice.utils.text import add_punctuation, chunk_text_punctuation
from omnivoice.utils.voice_design import (
    _INSTRUCT_ALL_VALID,
    _INSTRUCT_EN_TO_ZH,
    _INSTRUCT_MUTUALLY_EXCLUSIVE,
    _INSTRUCT_VALID_EN,
    _INSTRUCT_VALID_ZH,
    _INSTRUCT_ZH_TO_EN,
    _ZH_RE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VoiceClonePrompt:
    """Voice Cloning 模式的可复用参考提示。

    由 :meth:`OmniVoice.create_voice_clone_prompt` 构造一次后，
    可在多次 ``generate()`` 调用中重复使用，避免重复跑 audio_tokenizer.encode。
    """
    ref_audio_tokens: torch.Tensor  # 参考音频离散 token, shape = (C=8, T)
    ref_text: str                   # 参考音频对应的文本(可由 Whisper 自动转写)
    ref_rms: float                  # 参考音频响度(RMS 均方根 = sqrt(mean(wav**2))), 用来给输出做音量归一


@dataclass
class OmniVoiceGenerationConfig:
    """生成阶段的解码超参集合.

    这些参数控制 mask-then-fill 迭代解码的行为, 与模型权重无关.
    """
    num_step: int = 32              # 迭代解码总步数 (类似 diffusion 的 T)
    guidance_scale: float = 2.0     # Classifier-Free Guidance 强度, 0 表示关闭
    t_shift: float = 0.1            # 时间步采样的偏移, 越小越偏向 "先解低 SNR/噪声大" 的位置
    layer_penalty_factor: float = 5.0   # codebook 层惩罚: 让靠前(语义)的 codebook 先被解出
    position_temperature: float = 5.0   # 选位置时加在 score 上的 Gumbel 噪声温度
    class_temperature: float = 0.0      # 选 token id 时的采样温度, 0 = 贪心 argmax
    denoise: bool = True                # 是否在序列前插入 <|denoise|> 特殊 token
    preprocess_prompt: bool = True      # 是否对参考音频做去静音/裁剪等预处理
    postprocess_output: bool = True     # 是否对输出做去静音/淡入淡出
    audio_chunk_duration: float = 15.0  # 长文本分块时单段目标时长(秒)
    audio_chunk_threshold: float = 30.0 # 区分长短音频的阈值(秒)

    @classmethod
    def from_dict(cls, kwargs_dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class GenerationTask:
    """一次 TTS 推理的内部任务包。

    它不是用户直接调用 ``generate()`` 时传入的原始参数，而是
    ``_preprocess_all()`` 完成 batch 归一化、语言解析、参考音频编码和
    目标时长估计后得到的结构化结果。除 ``batch_size`` 外，其余列表字段
    都按 batch 对齐：同一索引 ``i`` 表示同一条 TTS 样本。
    """
    batch_size: int  # 当前任务包含的 TTS 样本数，也就是各平行列表的长度
    texts: List[str]  # 每条样本要合成的目标文本
    target_lens: List[int]  # 每条目标音频要生成的 token 帧数，直接影响输出时长
    langs: List[Optional[str]]  # 解析后的语言代码，例如 "zh"、"en"；未指定时为 None
    instructs: List[Optional[str]]  # Voice Design 风格指令；未使用时为 None
    ref_texts: List[Optional[str]]  # 参考音频对应的文本；非声音克隆模式为 None
    ref_audio_tokens: List[Optional[torch.Tensor]]  # 参考音频 token，单项形状 (C=8, T)
    ref_rms: List[Optional[float]]  # 参考音频响度 RMS，用于生成结果的音量归一
    speed: Optional[List[float]] = None  # 每条样本的时长缩放比例，长文本分块时使用

    def get_indices(
        self, config: OmniVoiceGenerationConfig, frame_rate: int
    ) -> Tuple[List[int], List[int]]:
        """按预计输出长度，把 batch 样本划分为短音频和长音频两组。
        区分长短音频的阈值在 OmniVoiceGenerationConfig.audio_chunk_threshold

        Args:
            config: OmniVoice 生成参数。其中 ``audio_chunk_threshold`` 表示
                启用分块生成的音频时长阈值，单位为秒。
            frame_rate: Audio Tokenizer 每秒产生的 token 时间帧数。
                当前模型通常为 25，即 1 秒音频约对应 25 个 token 帧。

        Returns:
            一个包含两个列表的元组 ``(short_idx, long_idx)``：

            - ``short_idx``：预计目标帧数小于或等于阈值的样本下标，
              这些样本走普通 batch 迭代生成。
            - ``long_idx``：预计目标帧数大于阈值的样本下标，
              这些样本走分块生成，以控制显存占用。

        ``audio_chunk_threshold`` 乘以 ``frame_rate`` 后，会从秒数阈值
        转换为可与 ``target_lens`` 比较的 token 帧数阈值。
        """
        # 例如阈值为 30 秒、帧率为 25，则 threshold=750 个目标 token 帧。
        threshold = int(config.audio_chunk_threshold * frame_rate)
        # enumerate 同时取得样本下标 i 和该样本的目标帧数 l。
        short_idx = [i for i, l in enumerate(self.target_lens) if l <= threshold]
        long_idx = [i for i, l in enumerate(self.target_lens) if l > threshold]
        return short_idx, long_idx

    def slice_task(self, indices: List[int]):
        """根据样本下标提取子任务集合，并保持所有 batch 字段一一对应。

        例如 ``indices=[0, 2]`` 会从当前任务中取出第 0、2 条样本，
        生成一个 ``batch_size=2`` 的新 ``GenerationTask``。该方法用于把
        ``get_indices()`` 得到的短样本和长样本分别交给不同生成路径。
        """
        # 空下标没有可生成的样本，返回 None，调用方会跳过该生成分支。
        if not indices:
            return None
        # 所有字段必须使用相同 indices 切片，保证文本、语言、参考音频等仍属于同一条样本。
        return GenerationTask(
            batch_size=len(indices),
            texts=[self.texts[i] for i in indices],
            target_lens=[self.target_lens[i] for i in indices],
            langs=[self.langs[i] for i in indices],
            instructs=[self.instructs[i] for i in indices],
            ref_texts=[self.ref_texts[i] for i in indices],
            ref_audio_tokens=[self.ref_audio_tokens[i] for i in indices],
            ref_rms=[self.ref_rms[i] for i in indices],
            speed=[self.speed[i] for i in indices] if self.speed else None,
        )


@dataclass
class OmniVoiceModelOutput(ModelOutput):
    """OmniVoice ``forward()`` 的标准返回结构。

    ``ModelOutput`` 是 Hugging Face 提供的模型输出基类，支持通过
    ``output.logits`` 等属性以及字典/元组形式访问返回字段。
    """

    # note 训练损失；传入 labels 时计算，推理时通常为 None。
    loss: Optional[torch.Tensor] = None
    # note 各 codebook、各序列位置对候选 audio token 的原始分数，形状为 [B, C, S, V]。
    logits: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Config & Model
# ---------------------------------------------------------------------------


class OmniVoiceConfig(PretrainedConfig):
    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        audio_codebook_weights: Optional[list[float]] = None,
        llm_config: Optional[Union[dict, PretrainedConfig]] = None,
        **kwargs,
    ):

        if isinstance(llm_config, dict):
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)

        self.llm_config = llm_config

        super().__init__(**kwargs)
        self.audio_vocab_size = audio_vocab_size
        self.audio_mask_id = audio_mask_id
        self.num_audio_codebook = num_audio_codebook
        if audio_codebook_weights is None:
            audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]
        self.audio_codebook_weights = audio_codebook_weights


def _resolve_model_path(name_or_path: str) -> str:
    """
    Resolve a local model directory or Hugging Face repo id into a local path.

    Args:
        name_or_path: Either an existing local directory or a Hugging Face model
            repository id such as ``k2-fsa/OmniVoice``.

    Returns:
        A filesystem path that can be passed to ``from_pretrained`` loaders.
    """
    if os.path.isdir(name_or_path):
        return name_or_path

    # Download to the Hugging Face cache only when the caller did not pass a
    # local directory. snapshot_download returns the cached snapshot path.
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)


class EmbeddingAccessMixin:
    """Embedding 访问适配 Mixin。

    Mixin 可以理解成“能力拼装类”：它通常不单独实例化，而是被其他类继承，
    给目标类补充一组小能力。这里补充的是 Hugging Face 模型常用的
    ``get_input_embeddings`` / ``set_input_embeddings`` 接口。

    OmniVoice 外层模型包了一层音频 token 逻辑，但文本 token 的 embedding
    仍然由内部 LLM backbone 管理。这个 Mixin 把外层模型的 embedding 访问
    转发给 ``self.llm``，方便新增特殊 token、resize embedding、加载
    checkpoint 和训练初始化等通用流程复用 Transformers 的标准接口。
    """

    def get_input_embeddings(self):
        """Return the inner LLM's text embedding table."""
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Replace the inner LLM's text embedding table."""
        self.llm.set_input_embeddings(value)


class OmniVoice(EmbeddingAccessMixin, PreTrainedModel):
    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(self, config: OmniVoiceConfig, llm: Optional[PreTrainedModel] = None):
        super().__init__(config)

        if llm is not None:
            # If an LLM instance is provided, use it directly
            # (skipping config-based init).
            self.llm = llm
        else:
            # Otherwise, initialize the LLM from the config.
            self.llm = AutoModel.from_config(self.config.llm_config)

        # 把 8 个 codebook 的 embedding 合到一张大表里:
        #   行号 = codebook_id * audio_vocab_size + token_id
        # 这样 8 层 token 共享同一个 nn.Embedding, 又能通过 codebook_layer_offsets
        # 给不同层加上独立偏移, 保证不冲突.
        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            self.config.llm_config.hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )

        # 一次性输出 8 层 codebook 的 logits, 后面 reshape 成 [B, C, T, V]
        self.audio_heads = nn.Linear(
            self.config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        # 归一化的 codebook 权重: 训练 loss 中越靠后的层权重越小 (语义信息少)
        self.normalized_audio_codebook_weights = [
            w / sum(config.audio_codebook_weights)
            for w in config.audio_codebook_weights
        ]

        self.post_init()

        # Inference-only attributes (set by from_pretrained when not in train mode)
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.duration_estimator = None
        self.sampling_rate = None
        self._asr_pipe = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """加载 OmniVoice 预训练模型 (主干 + 推理所需的全部周边组件).

        本方法在标准 HuggingFace ``PreTrainedModel.from_pretrained`` 之上做了三件事:
            1. 先把 ``pretrained_model_name_or_path`` 解析成本地路径 (必要时触发 HF Hub 下载),
               避免父类内部多次重复下载.
            2. 调父类 ``from_pretrained`` 加载主干权重与 LLM 配置.
            3. 推理模式 (``train=False``) 下额外装配下列组件, 凑齐端到端 TTS 推理流水线:
                - ``text_tokenizer``        : 文本分词器
                - ``audio_tokenizer``       : 音频 ↔ 离散 token 编解码器 (Higgs Audio V2)
                - ``feature_extractor``     : 音频特征提取器 (供 audio_tokenizer 使用)
                - ``sampling_rate``         : 采样率 (来自 feature_extractor)
                - ``duration_estimator``    : 基于字符权重的时长估计器, 推算目标音频帧数
                - ``_asr_pipe`` (可选)      : Whisper ASR 流水线, 用于自动转写参考音频

        Args:
            pretrained_model_name_or_path (str | os.PathLike):
                HuggingFace 模型仓库名 (如 ``"k2-fsa/OmniVoice"``) 或本地目录路径.
                如果是仓库名且本地缓存不存在, 会触发 HF Hub 下载.
            *args:
                透传给父类 ``PreTrainedModel.from_pretrained`` 的位置参数, 通常不用.
            **kwargs:
                包含本方法自定义关键字, 其余透传给父类. 自定义关键字:

                - train (bool, 默认 ``False``):
                    是否以训练模式加载. 训练模式下**不会**加载 text_tokenizer /
                    audio_tokenizer / duration_estimator 等推理辅助组件, 加载更快、显存更省.
                    推理 / Demo 场景请保持默认 ``False``.
                - load_asr (bool, 默认 ``False``):
                    是否同时加载 Whisper ASR 模型. 仅在用户没有手动提供 ``ref_text``
                    (参考音频对应的文本) 时需要, 用 ASR 自动转写参考音频得到 ref_text.
                    仅在 ``train=False`` 时生效.
                - asr_model_name (str, 默认 ``"openai/whisper-large-v3-turbo"``):
                    要加载的 Whisper 模型仓库名或本地路径. 仅在 ``load_asr=True`` 时生效.

                其余常见透传 kwargs (由父类处理) 例如:

                - torch_dtype: 权重精度, 如 ``torch.bfloat16`` / ``torch.float16``.
                - device_map: 设备映射, 如 ``"cuda"`` / ``"auto"`` / ``{"": 0}``.
                - low_cpu_mem_usage: 是否启用低内存加载.
                - revision / cache_dir / token: HF Hub 相关参数.

        Returns:
            OmniVoiceForConditionalGeneration:
                加载完毕的模型实例. 推理模式下 ``text_tokenizer`` / ``audio_tokenizer`` /
                ``feature_extractor`` / ``sampling_rate`` / ``duration_estimator`` 已就绪;
                若 ``load_asr=True``, ``_asr_pipe`` 也已就绪, 可直接调用 ``model.transcribe``.

        Notes:
            - audio_tokenizer 优先使用 checkpoint 自带的 ``audio_tokenizer/`` 子目录;
              缺失时回落到 HuggingFace 上的 ``eustlb/higgs-audio-v2-tokenizer``.
            - MPS (Apple Silicon) 后端不支持 audio_tokenizer 内部卷积的大通道数,
              此时会自动把 audio_tokenizer 放到 CPU, 主模型仍在 MPS 上.
            - 加载过程中会临时压制 transformers/huggingface_hub 的 INFO 日志,
              避免控制台噪音; 结束后自动恢复.

        Example:
            >>> import torch
            >>> model = OmniVoiceForConditionalGeneration.from_pretrained(
            ...     "k2-fsa/OmniVoice",
            ...     torch_dtype=torch.bfloat16,
            ...     device_map="cuda",
            ...     load_asr=True,
            ... )
        """
        train_mode = kwargs.pop("train", False)
        load_asr = kwargs.pop("load_asr", False)
        asr_model_name = kwargs.pop("asr_model_name", "openai/whisper-large-v3-turbo")

        # Suppress noisy INFO logs from transformers/huggingface_hub during loading
        _prev_disable = logging.root.manager.disable
        logging.disable(logging.INFO)

        try:
            # 先解析成本地路径; 如果还没下载到本地缓存才会触发 HF hub 下载
            resolved_path = _resolve_model_path(pretrained_model_name_or_path)

            # 先调 HF PreTrainedModel.from_pretrained 加载主干 (含权重 + LLM config)
            model = super().from_pretrained(resolved_path, *args, **kwargs)

            # 推理模式下还要顺带加载 tokenizer / audio tokenizer / 时长估计器
            if not train_mode:
                model.text_tokenizer = AutoTokenizer.from_pretrained(resolved_path)

                # 优先使用 checkpoint 自带的 audio_tokenizer; 否则回落到 HuggingFace
                # 上的 higgs-audio-v2-tokenizer (24kHz, 8 codebook)
                audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")

                if not os.path.isdir(audio_tokenizer_path):
                    audio_tokenizer_path = _resolve_model_path(
                        "eustlb/higgs-audio-v2-tokenizer"
                    )

                # higgs-audio-v2-tokenizer 的 conv 输出通道 > 65536, MPS 不支持,
                # 这里把它强制放到 CPU 避免 MPS 报错.
                tokenizer_device = (
                    "cpu" if str(model.device).startswith("mps") else model.device
                )
                # 音频 ↔ 离散 token 的编解码器 (推理流水线里负责"音频 token 化"那一环)
                model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                    audio_tokenizer_path, device_map=tokenizer_device
                )
                model.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    audio_tokenizer_path
                )

                model.sampling_rate = model.feature_extractor.sampling_rate

                # 基于字符权重的时长估计器, 用来推算目标音频应该有多少帧
                model.duration_estimator = RuleDurationEstimator()

                # ASR 是可选的, 仅在用户没提供 ref_text 时用它自动转写参考音频
                if load_asr:
                    model.load_asr_model(model_name=asr_model_name)
        finally:
            logging.disable(_prev_disable)

        return model

    # -------------------------------------------------------------------
    # ASR support (optional, for auto-transcription)
    # -------------------------------------------------------------------

    def load_asr_model(self, model_name: str = "openai/whisper-large-v3-turbo"):
        """Load a Whisper ASR model for reference audio transcription.
           加载一个 asr 模型用来识别音频字幕
        Args:
            model_name: HuggingFace model name or local path for the Whisper model.
        """
        from transformers import pipeline as hf_pipeline

        logger.info("Loading ASR model %s ...", model_name)
        asr_dtype = (
            torch.float16 if str(self.device).startswith(("cuda", "xpu")) else torch.float32
        )

        model_name = _resolve_model_path(model_name)

        # 建好 ASR 流水线并挂到 self._asr_pipe 上, 供 transcribe() 复用:
        #   任务= 自动语音识别(语音→文本); model= 上面解析好的本地/HF 路径;
        #   dtype= GPU 用 fp16 省显存提速, CPU 用 fp32; device_map= 放到模型所在设备。
        self._asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            dtype=asr_dtype,
            device_map=self.device,
        )
        logger.info("ASR model loaded on %s.", self.device)

    # @torch.inference_mode(): 推理专用装饰器, 关闭梯度记录, 省显存/提速
    # (转写不需要反向传播)
    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, tuple],
    ) -> str:
        """Transcribe audio with the loaded Whisper ASR pipeline.

        使用已加载的 Whisper ASR 模型把参考音频转成文字。这个方法主要用于
        Voice Cloning 模式下用户没有传入 ``ref_text`` 的场景。

        Args:
            audio: File path or ``(waveform, sample_rate)`` tuple.
                Waveform can be a numpy array or torch.Tensor of shape
                ``(1, T)`` or ``(T,)``.

        Returns:
            Transcribed text.
        """
        # 前置检查: 必须先 load_asr_model() 才有 _asr_pipe 可用
        if self._asr_pipe is None:
            raise RuntimeError(
                "ASR model is not loaded. Call model.load_asr_model() first."
            )

        if isinstance(audio, str):
            # 传的是文件路径: pipeline 能直接读, 取出 text 并去掉首尾空白
            return self._asr_pipe(audio)["text"].strip()
        else:
            # 传的是 (波形, 采样率) 元组: 需先整理成 pipeline 要的输入格式
            waveform, sr = audio
            # 波形可能是 torch 张量或 numpy 数组; 统一成 numpy 供后续处理:
            # 若是 torch 张量, 先 .cpu() 从 GPU 搬回内存(numpy 只能在 CPU),
            # 再 .numpy() 转成 numpy; 本来就是 numpy 则跳过此 if。
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()  # torch → numpy
            # np.squeeze: 删掉数组中所有长度为1的维度(任意位置都删), 把波形压成一维 (T,)。
            # 形状记号说明: (x, y) 描述数组形状, 表示 x 行、每行 y 个数字;
            #   (T,)   一维: 一排 T 个数;
            #   (1, T) 二维: 1 行 T 列, 即单声道(声道维度长度为1)。
            # 这里的波形至多是 (1, T), 因此 squeeze 后统一变成 (T,);
            # 本来就是 (T,) 则无长度为1的维度可删, 原样返回。
            # (若为真正多声道如 (2, T), 长度2的维度不会被删——但上游已保证是单声道)
            waveform = np.squeeze(waveform)  # (1, T) or (T,) → (T,) 压成一维
            # pipeline 接受 {"array": 一维波形, "sampling_rate": 采样率} 形式的输入，这个是字典dict类型
            audio_input = {
                "array": waveform,
                "sampling_rate": sr,
            }
            # 调用 ASR pipeline 得到 {"text": ...}, 取文本并去首尾空白。
            # 这里传入的是 dict(字段类型参数): {"array": 波形, "sampling_rate": 采样率};
            # _asr_pipe() 支持两种输入: ① str 音频文件路径; ② 上面这种含波形+采样率的 dict。
            return self._asr_pipe(audio_input)["text"].strip()

    def _prepare_embed_inputs(
        self, input_ids: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        """把混合的 [文本 token / 8 层音频 token] 序列映射成统一的输入 embedding.

        - 文本位置: 取第 0 层 token id, 走 LLM 自带的 text embedding.
        - 音频位置: 8 层 codebook id 分别加上层偏移后, 在共享 audio_embeddings
          里查表再相加, 形成该帧的复合表示.
        - 最后用 audio_mask 在两种 embedding 之间逐位置选择.

        input_ids:  (B, C, S),  audio_mask: (B, S),  返回: (B, S, H).
        """
        # 文本 embedding: input_ids[:, 0, :] 表示每个 batch 样本都取第 0 个 codebook 的完整序列；8 层文本 token 相同，取一层即可
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])

        # 为不同 codebook 加上独立偏移, 让 8 层共享同一张大 embedding 表也不会撞 id.
        # input_ids * audio_mask.unsqueeze(1): mask 为 1/True 时保留原 ID，为 0/False 时将该位置置 0
        # (后面会被 torch.where 丢弃, 但归零更安全, 避免 id 越界).
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1)
        ) + self.codebook_layer_offsets.view(1, -1, 1)

        # 8 层 codebook 分别 lookup 后在 codebook 维度求和, 表示一个音频帧
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        # 文本位置用 text_embeds, 音频位置用 audio_embeds
        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor, # self()#batch_input_ids：Cond 与 Uncond 的 token
        audio_mask: torch.Tensor, # self()#batch_audio_mask：区分文本位置和音频位置
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # self()#batch_attention_mask 控制序列位置之间能否互相关注
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """执行一次 OmniVoice 主模型前向计算（并在训练时计算 audio token loss）.

        Args:
            input_ids: 混合序列 token, 形状通常为 ``(B, C, S)``。文本位置在
                各 codebook 层复制文本 token; 音频位置是多层 audio codebook token。
            audio_mask: 标记哪些位置是音频 token, 形状为 ``(B, S)``。
            labels: 训练目标 audio token, 形状通常为 ``(B, C, S)``。值为
                ``-100`` 的位置会被 cross entropy 忽略。
            attention_mask: 普通 attention mask, 常用于 SDPA 路径。
            document_ids: sequence packing 时的样本边界, 用于构造 flex attention
                block mask, 避免不同样本错误互相 attend。
            position_ids: 可选位置 id, 透传给底层 LLM。

        Returns:
            ``OmniVoiceModelOutput``:
                - ``logits``: 每层 audio codebook 的分类 logits, 形状为
                  ``(B, C, S, audio_vocab_size)``。
                - ``loss``: 如果传入 labels, 则为 8 层 codebook 加权后的
                  cross entropy loss; 否则为 ``None``。
        """

        # 1. 把输入序列变成 Transformer 可处理的 embedding。
        #    文本位置走 LLM 原生 text embedding; 音频位置走 audio_embeddings。
        # 两个参数形状分别是 (B, C, S) 和 (B, S)
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)

        # 2. flex_attention 的训练路径会使用 document_ids 表示 packed sequence
        #    里的样本边界。这里把边界转换成 block mask, 防止多个样本拼接后
        #    在 attention 中互相看见。
        if attention_mask is None and document_ids is not None:
            if not _flex_attention_available:
                raise RuntimeError(
                    "flex_attention is not available in the current environment. "
                    "If you do not need flex_attention, set "
                    '"attn_implementation": "sdpa" in your training config.'
                )
            # create_block_mask 是 PyTorch flex_attention 的辅助函数:
            # 它把一个“哪些 query token 可以看哪些 key token”的规则函数,
            # 编译成更高效的 BlockMask, 后面传给 LLM attention 使用。
            #
            # _get_packed_mask(document_ids[0]) 会生成规则函数:
            #   只有 document_id 相同的 token 才能互相 attend。
            # 这样 sequence packing 把多个样本拼成一个长序列后,
            # 不同原始样本之间仍然不会在 self-attention 中互相泄漏信息。
            attention_mask = create_block_mask(
                # document_ids shape 通常是 [1, S], 取第 0 行得到长度为 S 的样本 id 序列。
                # to(inputs_embeds.device) 确保 mask 规则和模型输入在同一设备上。
                _get_packed_mask(
                    document_ids[0].to(inputs_embeds.device),
                ),
                # B/H 设为 None 表示这个 mask 规则对所有 batch 和 attention head 共享。
                B=None,
                H=None,
                # Q_LEN/KV_LEN 是当前 packed 序列的 query/key 长度; 自注意力下二者相同。
                Q_LEN=input_ids.size(-1),
                KV_LEN=input_ids.size(-1),
                # _compile=True 让 flex_attention 编译这个 BlockMask, 提升训练时 attention 效率。
                _compile=True,
                device=inputs_embeds.device,
            )

        # 3. 底层 Transformer / LLM 主干做上下文建模。
        #    这里输入已经是 inputs_embeds, 所以不会再走普通文本 token embedding。
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=position_ids,
        )
        # hidden_states 是 Transformer 主干输出的上下文向量序列, 不是最终 token。
        # 形状通常是 [B, S, H]: batch 内每个序列位置都有一个 H 维向量。
        # 每个向量已经融合了该位置能 attend 到的文本、参考音频和目标 token 上下文,
        # 后续 audio_heads 会把这些向量再投影成 audio token 的分类 logits。
        hidden_states = llm_outputs[0]

        # 4. 没有 labels 时只返回 logits, 常见于只想拿预测分布的场景。
        #    训练 / eval 时 dataloader 会提供 labels, 下面才会计算 loss。
        loss = None

        # 5. audio_heads 把每个时间位置的 hidden state 投影成 8 层 codebook
        #    的 token 分类分布。Shape: [B, S, C * Vocab]
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)
        # Shape: [B, S, C, Vocab] -> [B, C, S, Vocab]
        audio_logits = logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)

        if labels is not None:

            # 6. 逐 token 计算交叉熵。每个位置都是一次 audio token 分类:
            #    模型给出 vocab 概率分布, labels 给出正确 token id。
            # audio_logits.permute(0, 3, 1, 2):
            # [Batch, Layer, Seq, Vocab] -> [Batch, Vocab, Layer, Seq]
            # per_token_loss shape: [Batch, Layer, Seq]，ignore -100
            per_token_loss = torch.nn.functional.cross_entropy(
                audio_logits.permute(0, 3, 1, 2),
                labels,
                reduction="none",
                ignore_index=-100,
            )
            # 7. valid_mask 只保留真正需要监督的位置; labels=-100 的位置不计入 loss。
            # valid_mask shape: [Batch, Layer, Seq]
            valid_mask = (labels != -100).float()

            # 8. 先分别统计每一层 codebook 的平均 loss。
            # layer_means shape: [num_layers]
            layer_means = (per_token_loss * valid_mask).sum(
                dim=(0, 2)
            ) / valid_mask.sum(dim=(0, 2)).clamp(min=1.0)

            # 9. 不同 codebook 层的信息量不同, 用配置里的 audio_codebook_weights
            #    归一化后加权求和, 得到最终训练 loss。
            weights = torch.tensor(
                self.normalized_audio_codebook_weights, device=audio_logits.device
            )
            loss = (layer_means * weights).sum()

        return OmniVoiceModelOutput(
            loss=loss,
            logits=audio_logits,
        )

    def supported_language_ids(self) -> set[str]:
        """Return a list of supported language IDs."""
        return LANG_IDS

    def supported_language_names(self) -> set[str]:
        """Return a list of supported language names."""
        return LANG_NAMES

    # -------------------------------------------------------------------
    # Inference API
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str], None] = None,
        ref_text: Union[str, list[str], None] = None,
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,
        voice_clone_prompt: Union[
            VoiceClonePrompt, list[VoiceClonePrompt], None
        ] = None,
        instruct: Union[str, list[str], None] = None,
        duration: Union[float, list[Optional[float]], None] = None,
        speed: Union[float, list[Optional[float]], None] = None,
        generation_config: Optional[OmniVoiceGenerationConfig] = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Generate speech audio given text in various modes.

        Supports three modes:

        1. **Voice clone** — clone the voice style from the reference audio.
            Should provide ``voice_clone_prompt`` (from
           :meth:`create_voice_clone_prompt`) or ``ref_text`` + ``ref_audio``.
        2. **Voice design** — provide ``instruct`` text describing
           the desired voice style; no reference audio needed.
        3. **Auto** — provide neither; the model picks a voice itself.

        Args:
            text: Target text (single string or list for batch).
            language: Language name (e.g. ``"English"``) or code
                (e.g. ``"en"``). ``None`` for language-agnostic mode.
                Performance is slightly better if you specify the language.
            ref_text: Optional reference text for voice cloning mode.
            ref_audio: Optional reference audio for voice cloning mode.
                Can be a file path or a (waveform, sample_rate) tuple.
            voice_clone_prompt: Reusable prompt from :meth:`create_voice_clone_prompt`.
                If provided, it overrides ``ref_text`` and ``ref_audio``.
            instruct: Style instruction for voice design mode.
            duration: Fixed output duration in seconds. If a single float,
                applies to all items; if a list, one value per item.
                ``None`` (default) lets the model estimate duration from text.
                Overrides ``speed`` when both are provided.
            speed: Speaking speed factor. ``> 1.0`` for faster, ``< 1.0`` for
                slower. If a list, one value per item. ``None`` (default) uses
                the model's default estimation.
            generation_config: Explicit config object. If provided, takes
                precedence over ``**kwargs``.
            **kwargs: Generation config or its fields:
                denoise: Whether to prepend the ``<|denoise|>`` token.
                num_step: Number of iterative decoding steps.
                guidance_scale: Classifier-free guidance scale.
                t_shift: Time-step shift (smaller → emphasise low-SNR).
                postprocess_output: Post-process output (remove silence, fade-in/out, pad edges).
                layer_penalty_factor: Penalty encouraging earlier codebook
                    layers to unmask first.
                position_temperature: Temperature for position selection.
                class_temperature: Temperature for token sampling (0 = greedy).
                audio_chunk_duration: If > 0, split long text into chunks of
                    this duration (seconds) and generate chunk by chunk.
                audio_chunk_threshold: Only apply chunking if estimated audio
                    duration exceeds this threshold (seconds).
        Returns:
            ``audios`` a list of 1-D ``np.ndarray`` with shape ``(T,)`` and
            sampling rate consistent with the model's audio tokenizer
            (usually 24 000 Hz).  Can be saved directly with
            ``soundfile.write("out.wav", audios[0], model.sampling_rate)``.
        """

        if self.audio_tokenizer is None or self.text_tokenizer is None:
            raise RuntimeError(
                "Model is not loaded with audio/text tokenizers. Make sure you "
                "loaded the model with OmniVoice.from_pretrained()."
            )
        gen_config = (
            generation_config
            if generation_config is not None
            else OmniVoiceGenerationConfig.from_dict(kwargs)
        )

        self.eval()

        # Step 1: 输入清洗 + 估时长。把五花八门的用户输入(文本/语言/参考音频/instruct/
        #   speed/duration)统一成按 batch 对齐的 GenerationTask, 期间会做文本归一化、
        #   语言/风格解析、参考音频 encode + 必要时 ASR 转写, 【并估计每条的目标tts的音频时长/token 帧数】。
        # full_task 类型如下：
        # GenerationTask(
        #     batch_size=batch_size,
        #     texts=text_list,                       # 待合成文本
        #     target_lens=num_target_tokens_list,    # 每条要生成多少帧 token (决定输出时长)
        #     langs=language_list,                   # 解析后的语言码
        #     instructs=instruct_list,               # 规范化后的风格指令
        #     ref_texts=ref_text_list,               # 参考文本 (无则 None)
        #     ref_audio_tokens=ref_audio_tokens_list,  # 参考音频 token (无则 None)
        #     ref_rms=ref_rms_list,                  # 参考音频 RMS, 用于输出音量归一
        #     speed=speed_list,                      # 每条的缩放比例 (供分块按比例缩放)
        # )
        full_task = self._preprocess_all(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_audio=ref_audio,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
            preprocess_prompt=gen_config.preprocess_prompt,
            speed=speed,
            duration=duration,
        )

        # Step 2: 按估计长度切分: 短句直接 batch 解码, 长句改走分段解码以控制显存
        # 区分长短音频的阈值在 OmniVoiceGenerationConfig.audio_chunk_threshold
        short_idx, long_idx = full_task.get_indices(
            gen_config, self.audio_tokenizer.config.frame_rate
        )
        # 结果数组长度
        results = [None] * full_task.batch_size

        if short_idx:
            # 获取短音频子任务集合
            short_task = full_task.slice_task(short_idx)
            # _generate_iterative 是推理生成的核心函数，负责按帧迭代解码音频 token，
            short_results = self._generate_iterative(short_task, gen_config)
            # zip()：把多个序列"按位置配对"一起遍历
            for idx, res in zip(short_idx, short_results):
                results[idx] = res # 把短音频子任务集合的生成结果保存到 results 数组中

        if long_idx:
            # 获取长音频子任务集合
            long_task = full_task.slice_task(long_idx)
            long_results = self._generate_chunked(long_task, gen_config)
            for idx, res in zip(long_idx, long_results):
                results[idx] = res

        # Step 3: 解码 audio token → 波形, 并按 ref_rms 做音量归一/淡入淡出
        generated_audios = []
        for i in range(full_task.batch_size):
            assert results[i] is not None, f"Result {i} was not generated"
            generated_audios.append(
                self._decode_and_post_process(
                    results[i], full_task.ref_rms[i], gen_config  # type: ignore[arg-type]
                )
            )

        return generated_audios

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple[torch.Tensor, int]],
        ref_text: Optional[str] = None,
        preprocess_prompt: bool = True,
    ) -> VoiceClonePrompt:
        """Create a reusable voice clone prompt from reference audio.

        Args:
            ref_audio: File path (str) or ``(waveform, sample_rate)`` tuple.
                waveform should be a 1-D or 2-D torch.Tensor (channels x samples).
            ref_text: Transcript of the reference audio. If ``None``, the
                ASR model will be used to auto-transcribe (must call
                :meth:`load_asr_model` first).
            preprocess_prompt: If ``True`` (default), apply silence removal and
                trimming to the reference audio, add punctuation in the end
                of reference text (if not already)

        Returns:
            A :class:`VoiceClonePrompt` that can be passed to :meth:`generate`.

        中文说明:
            从参考音频构造一个可复用的声音克隆提示 (VoiceClonePrompt)。
            核心是把参考音频清洗后编码成离散 token, 之后多次 generate() 可复用,
            避免重复编码。

            参数:
                ref_audio: 文件路径(str) 或 ``(波形, 采样率)`` 元组;
                    波形为 1 维或 2 维 torch.Tensor (通道 x 采样点)。
                ref_text: 参考音频的文本。为 ``None`` 时用 ASR 模型自动转写
                    (需先调用 load_asr_model)。
                preprocess_prompt: 默认 ``True`` 时对参考音频做去静音/裁剪,
                    并在参考文本末尾补标点 (若没有)。

            返回:
                一个可传给 generate() 的 VoiceClonePrompt。
        """
        # 前置检查: 必须已加载 audio_tokenizer (否则无法把波形编码成 token)
        if self.audio_tokenizer is None:
            raise RuntimeError(
                "Audio tokenizer is not loaded. Make sure you loaded the model "
                "with OmniVoice.from_pretrained()."
            )

        # 把 ref_audio 统一加载/归一化成模型采样率下的单声道波形
        if isinstance(ref_audio, str):
            # 传的是文件路径: 直接按模型采样率读入。ref_wav 是 (1, T) 的 numpy 数组
            ref_wav = load_audio(ref_audio, self.sampling_rate)
        else:
            # 传的是 (波形, 采样率) 元组 tuple[torch.Tensor, int]: 手动做格式归一
            waveform, sr = ref_audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()  # tensor → numpy
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]  # 1 维 → (1, T), 补出通道维
            if waveform.shape[0] > 1:
                # 多声道 → 沿第 0 轴(通道轴 axis)取平均混为单声道;
                # np.mean 表示求平均，keepdims=True 保留该轴长度为 1, 使形状从 (C, T) 变成 (1, T)
                waveform = np.mean(waveform, axis=0, keepdims=True)
            if sr != self.sampling_rate:
                # 采样率不一致 → 重采样到模型采样率
                waveform = torchaudio.functional.resample(
                    torch.from_numpy(waveform),   # numpy 波形 → torch.Tensor(resample 只吃张量)
                    orig_freq=sr,                 # 原采样率
                    new_freq=self.sampling_rate,  # 目标采样率(模型要求)
                ).numpy()                          # 返回 torch.Tensor, 再转回 numpy 接回流水线
            # 归一完成: ref_wav 是 CPU 上的 numpy 波形, 形状 (1, T), 单声道, 采样率已对齐到 self.sampling_rate
            ref_wav = waveform

        # 计算 RMS(均方根, 衡量响度); 过于安静(0<rms<0.1)时放大到 0.1, 避免克隆质量下降。
        # 该 ref_rms 会随 prompt 返回, 用于给最终输出做音量归一。
        ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
        if 0 < ref_rms < 0.1:
            ref_wav = ref_wav * 0.1 / ref_rms

        if preprocess_prompt:
            # Trim long reference audio (>20s) by splitting at the largest silence gap.
            # Skip trimming when ref_text is user-provided, otherwise the
            # trimmed audio will no longer match the full transcript.
            # 裁剪过长(>trim_threshold 秒)的参考音频: 在 max_duration(默认15s)之前、
            # 尽量靠后的静音处切一刀, 只保留前面一段, 使切口落在静音里、不从词中间截断。
            # 仅当 ref_text 为 None(需自动转写)时才裁剪; 若 ref_text 是用户提供的,
            # 裁剪后音频会和完整文本对不上, 所以跳过。
            if ref_text is None:
                ref_wav = trim_long_audio(
                    ref_wav,
                    self.sampling_rate,
                    trim_threshold=20.0,  # 时长超过该秒数(20s)才触发裁剪, 否则原样返回
                )
            # 去静音: 收敛中间长静音(>200ms), 并裁掉首尾静音(各保留 100/200ms)
            ref_wav = remove_silence(
                ref_wav,
                self.sampling_rate,
                mid_sil=200,
                lead_sil=100,
                trail_sil=200,
            )
            # 去静音后若整段为空, 说明输入几乎全是静音, 报错提示关闭预处理
            if ref_wav.shape[-1] == 0:
                raise ValueError(
                    "Reference audio is empty after silence removal. "
                    "Try setting preprocess_prompt=False."
                )

        # 参考音频过长(>20s)会拖慢生成、增大显存且降低克隆质量, 给出告警(建议 3-10s)
        ref_duration = ref_wav.shape[-1] / self.sampling_rate
        if ref_duration > 20.0:
            logger.warning(
                "Reference audio is %.1fs long (>20s). This may cause slower "
                "generation, higher memory usage, and degraded voice cloning "
                "quality. We recommend trimming it to 3-10s.",
                ref_duration,
            )

        # Auto-transcribe if ref_text not provided
        # 未提供 ref_text 时, 用 ASR 模型自动转写参考音频得到文本
        if ref_text is None:
            if self._asr_pipe is None:
                # ASR 尚未加载则现场加载
                logger.info("ASR model not loaded yet, loading on-the-fly ...")
                self.load_asr_model()
            ref_text = self.transcribe((ref_wav, self.sampling_rate))
            logger.debug("Auto-transcribed ref_text: %s", ref_text)

        # 对齐 hop_length: 把尾部不足一帧的采样点裁掉, 保证 audio tokens 的时间帧整齐
        # (否则 encode 出来的最后一帧不完整)。
        # 采样点 → 时间帧 → codebook ID (24kHz, hop_length=960):
        #   960 个采样点 → 1 个时间帧 → 8 个 codebook ID
        #   1 秒 = 24000 采样点 ≈ 25 帧 ≈ 25 × 8 = 200 个 ID
        chunk_size = self.audio_tokenizer.config.hop_length
        clip_size = int(ref_wav.shape[-1] % chunk_size)  # clip_size 是被裁掉的尾部采样点数 / 取余: 尾部凑不满一帧的采样点数(要裁掉)
        ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav # 最终保留的采样点数 = 总采样点数 - 被裁掉的采样点数
        # numpy → torch at tokenizer boundary
        # 在进 tokenizer 前把 numpy 波形转成 torch 张量, 并搬到 tokenizer 所在设备
        ref_wav_tensor = torch.from_numpy(ref_wav).to(self.audio_tokenizer.device)
        # 关键: 参考音频通过 HiggsAudioV2 tokenizer 编码为 (C=8, T) 的离散 token,
        # 后续整个推理过程都在 token 空间里进行, 不再接触波形.
        # 注意不是“1 个采样点 → 8 个 ID”: encoder 先把约 hop_length 个采样点下采样成
        # 1 个 latent 向量, 再由 RVQ 8 层把这一个向量量化成 8 个 ID。
        #   - hop_length 定义在 tokenizer 的 config: self.audio_tokenizer.config.hop_length
        #   - “采样点 → codebook”的转换就发生在这里的 encode(), 返回 audio_codes (C, T)

        # ① 补 batch 维: (1, T_samples) -> (1, 1, T_samples), encode 期望带 batch 的输入
        batched_wav = ref_wav_tensor.unsqueeze(0)
        # ② 编码: 波形 -> tokenizer 输出对象(含 audio_codes 等字段)；然后取出离散 token: audio_codes 形状为 (1, C, T)(带 batch)
        # 重点：这一步就是 采样 -> codebook token 的过程
        encoded = self.audio_tokenizer.encode(batched_wav)
        batched_codes = encoded.audio_codes
        # 3️⃣ 去掉 batch 维: (1, C, T) -> (C, T), 得到这条参考音频的 8 层 codebook token
        ref_audio_tokens = batched_codes.squeeze(0)  # (C, T) squeeze(n)：只删长度为 1 的维度，否则静默不删、不报错

        # 预处理开启时, 给参考文本末尾补标点 (有助于模型理解句子边界)
        if preprocess_prompt:
            ref_text = add_punctuation(ref_text) # 给参考文本末尾补标点 (有助于模型理解句子边界)

        # 打包成可复用的提示: token + 文本 + 响度
        return VoiceClonePrompt(
            ref_audio_tokens=ref_audio_tokens,  # 参考音频离散 token (C=8, T)
            ref_text=ref_text,                  # 参考文本(可能为自动转写/补标点后)
            ref_rms=ref_rms,                    # 参考响度, 用于输出音量归一
        )

    def _decode_and_post_process(
        self,
        tokens: Union[torch.Tensor, List[torch.Tensor]],
        rms: Union[float, None],
        gen_config: OmniVoiceGenerationConfig,
    ) -> np.ndarray:
        """
        Args:
            tokens: Audio tokens — either a single tensor of shape
                (num_codebooks, seq_len) or a list of chunk tensors.
            rms: RMS of the reference audio for volume adjustment.
            gen_config: Generation config for post-processing options.
        Returns:
            Decoded and post-processed audio array of shape (T,).
        """
        tokenizer_device = self.audio_tokenizer.device
        if isinstance(tokens, list):
            # 长文本分段生成: 每个 chunk 单独解码后用 cross-fade 重叠拼接,
            # 避免在拼接处出现 click/爆音
            chunk_audios = [
                self.audio_tokenizer.decode(t.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .cpu()
                .numpy()
                for t in tokens
            ]
            audio_waveform = cross_fade_chunks(chunk_audios, self.sampling_rate)
        else:
            # 短句: 一次性 decode (C, T) → 波形
            audio_waveform = (
                self.audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .cpu()
                .numpy()
            )

        audio_waveform = self._post_process_audio(
            audio_waveform,
            postprocess_output=gen_config.postprocess_output,
            ref_rms=rms,
        )
        return audio_waveform.squeeze(0)

    def _post_process_audio(
        self,
        generated_audio: np.ndarray,
        postprocess_output: bool,
        ref_rms: Union[float, None],
    ) -> np.ndarray:
        """Optionally remove long silences, adjust volume, and add edge padding.

        Args:
            generated_audio: Numpy array of shape (1, T).
            postprocess_output: If True, remove long silences and apply fade/pad.
            ref_rms: RMS of the reference audio for volume normalisation.
        Returns:
            Processed numpy array of shape (1, T).
        """
        if postprocess_output:
            generated_audio = remove_silence(
                generated_audio,
                self.sampling_rate,
                mid_sil=500,
                lead_sil=100,
                trail_sil=100,
            )

        if ref_rms is not None and ref_rms < 0.1:
            generated_audio = generated_audio * ref_rms / 0.1
        elif ref_rms is None:
            peak = np.abs(generated_audio).max()
            if peak > 1e-6:
                generated_audio = generated_audio / peak * 0.5

        generated_audio = fade_and_pad_audio(
            generated_audio,
            sample_rate=self.sampling_rate,
        )
        return generated_audio

    def _generate_chunked(
        self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig
    ) -> List[List[torch.Tensor]]:
        """Generate long audio by splitting text into chunks and batching.

        Each item in the returned list corresponds to one input and contains
        a list of audio token tensors — one per text chunk.

        Args:
            task: A :class:`GenerationTask` with one or more items whose
                estimated audio exceeds ``audio_chunk_threshold``.
            gen_config: Generation config (``audio_chunk_duration`` controls
                chunk size).
        Returns:
            Per-item list of chunk token-tensor lists.
        """
        # Chunk each item's text
        all_chunks = []
        for i in range(task.batch_size):
            avg_tokens_per_char = task.target_lens[i] / len(task.texts[i])
            text_chunk_len = int(
                gen_config.audio_chunk_duration
                * self.audio_tokenizer.config.frame_rate
                / avg_tokens_per_char
            )
            chunks = chunk_text_punctuation(
                text=task.texts[i],
                chunk_len=text_chunk_len,
                min_chunk_len=3,
            )
            logger.debug(f"Item {i} chunked into {len(chunks)} pieces: {chunks}")
            all_chunks.append(chunks)

        has_ref = [t is not None for t in task.ref_audio_tokens]
        assert all(has_ref) or not any(has_ref), (
            "Chunked inference requires all items to either have or not have "
            "ref_audio. Mixed ref/non-ref is not supported."
        )

        max_num_chunks = max(len(c) for c in all_chunks)

        # chunk_results[item_idx] = list of generated token tensors per chunk
        chunk_results = [[] for _ in range(task.batch_size)]

        def _run_batch(indices, texts, ref_audios, ref_texts):
            speed_list = task.speed
            target_lens = [
                self._estimate_target_tokens(
                    texts[j],
                    ref_texts[j],
                    ref_audios[j].size(-1) if ref_audios[j] is not None else None,
                    speed=speed_list[i] if speed_list else 1.0,
                )
                for j, i in enumerate(indices)
            ]
            sub_task = GenerationTask(
                batch_size=len(indices),
                texts=texts,
                target_lens=target_lens,
                langs=[task.langs[i] for i in indices],
                instructs=[task.instructs[i] for i in indices],
                ref_texts=ref_texts,
                ref_audio_tokens=ref_audios,
                ref_rms=[task.ref_rms[i] for i in indices],
                speed=[task.speed[i] for i in indices] if task.speed else None,
            )
            gen_tokens = self._generate_iterative(sub_task, gen_config)
            for j, idx in enumerate(indices):
                chunk_results[idx].append(gen_tokens[j])

        if all(has_ref):
            # 所有样本都有参考音频: 同一个 chunk 索引下的不同样本可以一起 batch,
            # 单个样本内部仍按 chunk 顺序串行生成, 这样既能批量加速又不至于显存爆炸.
            for ci in range(max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[task.ref_audio_tokens[i] for i in indices],
                    ref_texts=[task.ref_texts[i] for i in indices],
                )
        else:
            # 没有参考音频: 先把每个样本的 chunk 0 生成出来,
            # 然后用 chunk 0 作为后续所有 chunk 的参考, 保证整段音色一致.
            indices_0 = [i for i in range(task.batch_size) if len(all_chunks[i]) > 0]
            _run_batch(
                indices_0,
                texts=[all_chunks[i][0] for i in indices_0],
                ref_audios=[None] * len(indices_0),
                ref_texts=[None] * len(indices_0),
            )
            first_chunk_map = {idx: chunk_results[idx][0] for idx in indices_0}

            # Batch all remaining chunks, using chunk 0 as fixed reference
            for ci in range(1, max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[first_chunk_map[i] for i in indices],
                    ref_texts=[all_chunks[i][0] for i in indices],
                )

        return chunk_results

    def _preprocess_all(
        self,
        text: Union[str, list[str]],  # 待合成文本; 单条 str 或一批 list[str], 决定 batch_size
        language: Union[str, list[str], None] = None,  # 语言 "English"/"en" 等; None=语言无关模式
        ref_text: Union[str, list[str], None] = None,  # 参考音频对应文本(voice clone); None 时自动 ASR 转写
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,  # 参考音频(voice clone): 文件路径 str 或 (波形, 采样率) 元组; None=无参考
        voice_clone_prompt: Union[
            VoiceClonePrompt, list[VoiceClonePrompt], None
        ] = None,  # 预构造好的可复用参考提示; 若给出则优先, ref_text/ref_audio 被忽略
        instruct: Union[str, list[str], None] = None,  # voice design 风格指令 "male, british accent" 等; None=不用
        preprocess_prompt: bool = True,  # 是否对参考音频做裁剪/去静音并给 ref_text 补标点
        speed: Union[float, list[Optional[float]], None] = None,  # 语速缩放; >1 更快 <1 更慢; None=模型默认
        duration: Union[float, list[Optional[float]], None] = None,  # 固定输出时长(秒); 覆盖 speed; None=按文本估计
    ) -> GenerationTask:
        """把五花八门的用户输入整理成统一的 GenerationTask, 供后续解码直接消费.

        这是 generate() 的第 1 步, 相当于"输入清洗 + 估时长"的总装配, 依次做 5 件事:
          1) 文本归一化成 batch (单条字符串也统一成列表), 定下 batch_size.
          2) 解析语言 (language) 与风格指令 (instruct) 为模型内部可用形式.
          3) 处理参考音频 (voice clone): 必要时把 ref_audio encode 成 token、
             用 ASR 自动转写 ref_text, 拆成三条平行列表.
          4) 归一化 speed / duration 为"每条样本一个值"的列表.
          5) 估计每条样本的目标 token 数 target_lens (掩码扩散必须先知道铺多少帧).

        返回:
            GenerationTask: 一个按 batch 对齐的"任务包", 每个字段都是长度 = batch_size
            的平行列表, 第 i 项对应第 i 条输入; 其中最关键的是 target_lens (每条要生成
            多少帧 token, 直接决定输出时长).
        """

        # ===== ① 文本归一化成 batch: 单条字符串也统一成列表, 并定下 batch_size =====
        if isinstance(text, str):
            text_list = [text]
        else:
            assert isinstance(
                text, list
            ), "text should be a string or a list of strings"
            text_list = text
        batch_size = len(text_list)

        # ===== ② 解析语言与风格指令 =====
        # 确认输入的语言是否为列表，如果是列表，则确保列表长度与 batch_size 一致；否则广播成列表
        language_list = self._ensure_list(language, batch_size)
        # 把用户传入的 language 规范化成模型内部使用的语言代码 (如 "zh")
        language_list = [_resolve_language(lang) for lang in language_list]
        # instruct: voice design 风格指令广播成 batch (下面逐条校验/规范化)
        instruct_list = self._ensure_list(instruct, batch_size)
        for i, s in enumerate(instruct_list):
            if s is None:
                continue
            use_zh = bool(text_list[i] and _ZH_RE.search(text_list[i])) # 判断文本是否【包含】中文
            instruct_list[i] = _resolve_instruct(s, use_zh=use_zh) # 规范化风格指令

        # ===== ③ 处理参考音频 (voice clone) =====
        # voice_clone_prompt 与 ref_text/ref_audio 二者都给时, 以前者为准
        if voice_clone_prompt is not None and (
            ref_text is not None or ref_audio is not None
        ):
            logger.warning(
                "Both voice_clone_prompt and ref_text/ref_audio are provided. "
                "ref_text/ref_audio will be ignored."
            )
        if voice_clone_prompt is None and ref_audio is not None:
            # 如果 voice_clone_prompt 为空 并且 ref_audio 不为空，
            #   则从 ref_audio 构造 voice_clone_prompt
            ref_text_list = self._ensure_list(ref_text, batch_size, auto_repeat=False)
            ref_audio_list = self._ensure_list(ref_audio, batch_size, auto_repeat=False)

            voice_clone_prompt = []
            for i in range(len(ref_text_list)):
                voice_clone_prompt.append( # 重点
                    # 创建 VoiceClonePrompt，这个对象是输入缓存、包括音频 tokenize 之后的 codebook token
                    self.create_voice_clone_prompt(
                        ref_audio=ref_audio_list[i],
                        ref_text=ref_text_list[i],
                        preprocess_prompt=preprocess_prompt,
                    )
                )

        # 把 voice clone 提示拆成三条平行列表; 没有参考音频时全部填 None (auto voice / voice design)。
        # 拆成 token / text / rms 三条独立列表, 是为了让下游按下标 i 直接取用, 无需再感知 VoiceClonePrompt。
        voice_clone_prompt_list = self._ensure_list(voice_clone_prompt, batch_size)
        if voice_clone_prompt_list[0] is not None:
            # 将 voice_clone_prompt_list 中的元素拆解成3个list
            ref_text_list = [vc.ref_text for vc in voice_clone_prompt_list]
            ref_audio_tokens_list = [
                vc.ref_audio_tokens for vc in voice_clone_prompt_list
            ]
            ref_rms_list = [vc.ref_rms for vc in voice_clone_prompt_list]
        else:
            ref_text_list = [None] * batch_size
            ref_audio_tokens_list = [None] * batch_size
            ref_rms_list = [None] * batch_size

        # ===== ④ 归一化 speed / duration 为"每条样本一个值"的列表 (可能含 None) =====
        if speed is not None:
            if isinstance(speed, (int, float)):
                 # [float(speed)] 先构造单元素列表 [x]，再 * batch_size 复制 batch_size 份 → [x,x,x ...batch_size个]（把一个语速套到全体）
                user_speed = [float(speed)] * batch_size
            else:
                # 如果本来就是list，则直接转换成list
                user_speed = list(speed)
        else:
            user_speed = None

        if duration is not None:
            if isinstance(duration, (int, float)):
                # [float(duration)] 先构造单元素列表 [x]，再 * batch_size 复制 batch_size 份 → [x,x,x ...batch_size个]（把一个时长套到全体）
                durations = [float(duration)] * batch_size
            else:
                durations = list(duration)
        else:
            durations = None

        # ===== ⑤ 估计每条样本的目标 token 数 (掩码扩散必须先知道铺多少帧) =====
        num_target_tokens_list = [] # 重要，保存每个 tts 文本预估的时长
        for i in range(batch_size):
            has_dur = durations is not None and durations[i] is not None
            # 如果指定了 duration[i]，则设置 speed 为 1.0；这里只是估计阶段先用 speed=1.0 拿到一个"原始估值"，真正的帧数稍后由 duration 直接覆盖（target_tokens = duration × frame_rate）
            # 否则使用 user_speed[i] 作为缩放比例
            item_speed = 1.0 if has_dur else (user_speed[i] if user_speed else 1.0)
            # 预估 tts - text 的时长、参考音频讲的快也会影响预估的时长，并且按照 speed 对结果时长进行缩放。另外，即使参考音频讲的快，speed也是在原参考音频基础之上进行缩放
            est = self._estimate_target_tokens(
                text_list[i],
                ref_text_list[i],
                ref_audio_tokens_list[i].size(-1) # ref_audio_tokens 形状为 (C=8, T), .size(-1) 取最后一维 = 【参考音频的时间帧数 T】
                if ref_audio_tokens_list[i] is not None
                else None,
                speed=item_speed,
            )
            num_target_tokens_list.append(est)

        # duration 覆盖: 若用户指定了固定时长, 直接把帧数设为 duration × frame_rate
        # frame_rate 是指每秒多少帧，这个在音频 tokenizer 的 config 中定义
        speed_list: Optional[List[float]] = None
        if durations is not None:
            # 每秒钟有多少音频帧
            frame_rate = self.audio_tokenizer.config.frame_rate
            speed_list = []
            for i in range(batch_size):
                if durations[i] is not None:
                    target_tokens = max(1, int(durations[i] * frame_rate))
                    est = num_target_tokens_list[i]
                    # 反算缩放比例 est/target: >1 表示要"压缩"(说快些), <1 表示"拉长",
                    speed_list.append(est / target_tokens if target_tokens > 0 else 1.0)
                    num_target_tokens_list[i] = target_tokens
                else:
                    s = user_speed[i] if user_speed else None
                    speed_list.append(s if s is not None else 1.0)
        elif user_speed is not None:
            # 遍历 user_speed 列表，如果 s 不为 None，则将 s 添加到 speed_list 中，否则添加 1.0
            speed_list = [s if s is not None else 1.0 for s in user_speed]

        # ===== 返回: 按 batch 对齐的任务包 (每个字段都是长度 = batch_size 的平行列表) =====
        return GenerationTask(
            batch_size=batch_size,
            texts=text_list,                       # 待合成文本
            target_lens=num_target_tokens_list,    # 每条要生成多少帧 token (决定输出时长)
            langs=language_list,                   # 解析后的语言码
            instructs=instruct_list,               # 规范化后的风格指令
            ref_texts=ref_text_list,               # 参考文本 (无则 None)
            ref_audio_tokens=ref_audio_tokens_list,  # 参考音频 token (无则 None)
            ref_rms=ref_rms_list,                  # 参考音频 RMS, 用于输出音量归一
            speed=speed_list,                      # 每条的缩放比例 (供分块按比例缩放)
        )

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        """Estimate number of target audio tokens."""
        if num_ref_audio_tokens is None or ref_text is None or len(ref_text) == 0:
            # Fall back to a simple heuristic
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = 25

        # 根据 ref_text, num_ref_audio_tokens 预估 text 的时长，时长单位同 num_ref_audio_tokens，基本思路见代码注释
        est = self.duration_estimator.estimate_duration(
            text, ref_text, num_ref_audio_tokens
        )
        if speed > 0 and speed != 1.0:
            est = est / speed # 速度越快目标时长越短 - 这个是完全按照时间倍数来的啊！
        return max(1, int(est)) # 至少1帧？这个好像也不太行，1帧才40ms，也太短了

    def _ensure_list(
        self, x: Union[Any, List[Any]], batch_size: int, auto_repeat: bool = True
    ) -> List[Any]:
        """把"单个值或列表"归一化成长度对齐 batch 的列表, 供批量处理统一消费.

        Args:
            x: 单个值 (如一个字符串) 或列表 (批量).
            batch_size: 目标批大小, 即待合成文本的条数.
            auto_repeat: 为 True 且 x 只有 1 个元素时, 复制成 batch_size 份
                ("一个值应用到所有样本"); 为 False 时保留原样, 由调用方自行处理
                (如 ref_text/ref_audio, 避免把同一份参考错误复制成多份).

        Returns:
            长度为 1 或 batch_size 的列表.
        """
        # 不是列表的单个值 → 裹成单元素列表
        x_list = x if isinstance(x, list) else [x]
        # 校验长度: 只允许 1 或 batch_size, 否则说明用户输入对不齐 (如 3 条文本配 2 个语言)
        if len(x_list) not in (
            1,
            batch_size,
        ):
            raise ValueError(
                f"should be either the number of the text or 1, but got {len(x_list)}"
            )
        # 按需广播: 只有 1 个元素时复制成 batch_size 份, 让该值应用到每条样本
        if auto_repeat and len(x_list) == 1 and batch_size is not None:
            x_list = x_list * batch_size
        return x_list

    def _prepare_inference_inputs(
        self,
        text: str,
        num_target_tokens: int,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        lang: Optional[str] = None,
        instruct: Optional[str] = None,
        denoise: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Prepare input_ids and audio masks for inference.
        Args:
            text: Target text to generate.
            num_target_tokens: Number of audio tokens to generate.
            ref_text: Optional reference text for voice cloning.
            ref_audio_tokens: Optional reference audio tokens for voice cloning.
                Shape is ``(C, T)``: ``C`` is the number of codebook layers
                (fixed to 8 in OmniVoice), and ``T`` is the number of encoded
                time frames. At 24 kHz with ``hop_length=960``, the tokenizer
                produces about 25 frames per second, so longer reference audio
                normally produces a larger ``T`` (``T ≈ duration_seconds × 25``).
                即参考音频的多层 codebook token 表示：C=8，T 为时间帧数；
                音频越长，T 通常越大。
            lang: Optional language ID.
            instruct: Optional style instruction for voice design.
            denoise: Whether to include the <|denoise|> token.

        Returns:
            dict[str, torch.Tensor]: 单条样本的模型输入, 含两个键:
              - ``input_ids``: (1, C, S) 混合 token 序列,
                内容顺序为 [style][text][可选 ref audio][target MASK]。
              - ``audio_mask``: (1, S) 布尔标记, False=style/text 位置(走文本 embedding),
                True=ref audio/target 位置(走 8 层 codebook 音频 embedding)。
                注意它是"位置类型标记", 不是 attention_mask, 也不是值为 1024 的 target MASK token。
        """

        # 1) Style 段: 可选 <|denoise|> + 语言标签 + Voice Design 指令标签
        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        lang_str = lang if lang else "None"
        instruct_str = instruct if instruct else "None"
        style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

        # 文本 token 只有 1 层, 这里 repeat 到 C=8 层方便和音频 token 在 codebook 维度对齐;
        # 后面 _prepare_embed_inputs 会根据 audio_mask 决定走文本路径还是音频路径.
        style_tokens = (
            self.text_tokenizer(style_text, return_tensors="pt")
            .input_ids.repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(
            self.device
        )  # [1, C, N1]

        # 2) Text 段: ref_text + text, 整段用 <|text_start|>...<|text_end|> 包起来,
        #    并解析 [laughter] 等非言语标签为特殊 token id.
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_tokens = (
            _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer)
            .repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(
            self.device
        )  # [1, C, N2]

        # 3) Target 段: 全部初始化为 MASK, 由后续迭代解码逐步填充。
        # torch.full(size, fill_value, ...) 会创建指定形状的 Tensor，
        # 并把其中每个位置都初始化成同一个值。这里各参数表示:
        #   size=(1, C, T): B=1 条样本、C=8 层 codebook、T=num_target_tokens 个目标时间帧;
        #   fill_value=audio_mask_id: 所有目标位置先填 MASK ID（当前为 1024）;
        #   dtype=torch.long: 使用整数类型保存 token ID;
        #   device=self.device: 直接在主模型所在的 CPU / GPU / MPS 设备上创建。
        # 得到的形状为 (1, 8, T)，之后模型会逐轮把 MASK 替换成预测出的音频 token ID。
        target_audio_tokens = torch.full(
            (1, self.config.num_audio_codebook, num_target_tokens),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        # 拼接顺序: [style] + [text] + (可选 [ref_audio_tokens]) + [target MASKs]
        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
        parts.append(target_audio_tokens)
        # torch.cat(tensors, dim) 把多个 Tensor 沿某个“已有维度”首尾拼接:
        #   tensors=parts: 待拼接的 Tensor 列表; 每项形状均为 (B=1, C=8, 各自序列长度)。
        #   dim=2: 沿第 2 维（从 0 开始计数的第三维，即序列维 S）拼接。
        # 它会保持相同的 batch 和 codebook 对应关系，分别拼接每个 [b, c, :] 序列;
        # token ID 不做算术加法，只有最后一维的长度相加。例如:
        #   (1, 8, N1) + (1, 8, N2) + (1, 8, T_ref) + (1, 8, T_target)
        #   -> (1, 8, N1 + N2 + T_ref + T_target)
        # 最终内容顺序为 [style][text][可选 ref audio][target MASK]。
        cond_input_ids = torch.cat(parts, dim=2)

        # audio_mask 标识 "哪些位置应当用音频 embedding": ref_audio + target 区域
        # Tensor.shape 是形状属性，返回类似元组的 torch.Size。
        # 例如 cond_input_ids.shape == [1, 8, 100] 时:
        #   shape[0] == 1   -> 第 0 维 B（batch 数量）
        #   shape[1] == 8   -> 第 1 维 C（codebook 层数）
        #   shape[2] == 100 -> 第 2 维 S（拼接后的序列总长度）
        # 因此这里取 shape[2]，保存 style/text/ref audio/target 拼接后的总长度。
        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)

        # torch.zeros 创建形状为 (B=1, S=cond_total_length) 的全 0 Tensor:
        #   dtype=torch.bool 会把 0 表示成 False;
        #   device=self.device 让 mask 和主模型位于同一 CPU / GPU / MPS 设备。
        # 这个 Tensor 不是音频数据，而是标记每个混合序列位置属于文本还是音频的布尔地图。
        cond_audio_mask = torch.zeros(
            1, cond_total_length, dtype=torch.bool, device=self.device
        )
        # [0, cond_audio_start_idx:] 表示第 0 条样本中，从音频起始下标直到序列末尾的所有位置。
        # 将这些位置设为 True 后，mask 结构类似:
        #   [False, ..., False, True, ..., True]
        #    style/text 区域       ref audio/target audio 区域
        # 后续 _prepare_embed_inputs 会据此让 False 位置走文本 embedding，
        # True 位置走 8 层 codebook 的音频 embedding。
        cond_audio_mask[0, cond_audio_start_idx:] = True

        # 返回主模型后续前向计算所需的两项输入:
        #   input_ids: 形状 (1, 8, S) 的完整混合 token 序列，
        #       内容为 [style][text][可选 ref audio][target MASK]。
        #   audio_mask: 形状 (1, S) 的布尔类型标记，
        #       False 表示 style/text 位置，True 表示 ref audio/target audio 位置。
        # audio_mask 用于选择文本或音频 embedding，不是 attention_mask。
        # 同时要和值为 1024 的 target MASK token 区分: 前者是位置类型标记，
        # 后者是“尚未生成的目标音频 token”所使用的整数 ID。
        return {
            "input_ids": cond_input_ids,
            "audio_mask": cond_audio_mask,
        }

    def _generate_iterative(
        self,
        task: GenerationTask,
        gen_config: OmniVoiceGenerationConfig
    ) -> List[torch.Tensor]:
        """N-step iterative unmasked decoding.
        Args:
            task: A :class:`GenerationTask` containing batch texts, target
                lengths, languages, instructions, and optional reference data.
            gen_config: A :class:`OmniVoiceGenerationConfig` controlling
                decoding steps, guidance, temperatures, etc.
        Returns:
            List of generated audio token tensors of shape (C, T) (one per input text).

        中文说明:
            掩码扩散(MaskGIT 式)迭代解码: 从全 MASK 的 (C, T) 目标出发, 分 num_step
            轮逐步把 MASK 填成真实 codec token, 每轮只填一批"最有把握"的位置。
            基本逻辑分 6 步:
              step 1: 为每条样本构造推理输入(style + text + 可选 ref + target 全 MASK)。
              step 2: 拼成 batch_size=2B 的 batch —— 前 B 条为"条件"(cond, 含 style/text/ref),
                      后 B 条为"无条件"(uncond, 只保留 target 区域); 一次 forward 同时得
                      两路 logits, 供 Classifier-Free Guidance(CFG) 融合。
              step 3: 初始化 tokens(当前已解出的 target 状态)为全 MASK。
              step 4: 计算时间步 timesteps 与每步要 unmask 的 token 数 schedules。
              step 5: 迭代主循环(num_step 轮), 每轮: 前向 → CFG 融合打分 → 层惩罚 + Gumbel
                      噪声选位置 → 屏蔽已填位置 → top-k 填入预测 token → 回写上下文。
              step 6: 按各自 target_lens 裁掉 padding, 返回每条 (C, T)。
        """

        B = task.batch_size

        for i in range(B):
            logger.debug(
                "Item %d — text: %s | ref_text: %s | instruct: %s | lang: %s | target_tokens: %d",
                i,
                task.texts[i],
                task.ref_texts[i],
                task.instructs[i],
                task.langs[i],
                task.target_lens[i],
            )

        # ===== step 1: 为每条样本构造推理输入 =====
        # 【重要】每条得到 {input_ids: (1,C,S), audio_mask: (1,S)}:
        # 【style】 + 【text】 + 【可选 ref 音频】 + 【target 全 MASK 段】
        # note 详细结构说明: books/code_notes/_prepare_inference_inputs_input_ids.md
        # 各条长度可能不同(文本/参考不同), 后面统一 pad 到 max_c_len。
        inputs_list = [
            self._prepare_inference_inputs(
                task.texts[i],  # target text；ref_text 由下一参数传入并在函数内部拼接
                task.target_lens[i],
                task.ref_texts[i],
                task.ref_audio_tokens[i],
                task.langs[i],
                task.instructs[i],
                gen_config.denoise,
            )
            for i in range(B) # NOTE 这里
        ]

        # ===== step 2: 拼成 batch_size=2B 的 batch (cond + uncond, 供 CFG) =====
        # for inp in inputs_list 中 inp 是 {input_ids: (1,C,S), audio_mask: (1,S)}
        #  (1,C,S).size(2) 是 S，即序列长度
        # 最后结果 c_lens 就是一个列表，每个元素对应每个input的序列长度 【重要】
        c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
        # max_c_len 是 所有 TTS 输入的 input_ids 张量的序列长度（S）中的最大值
        max_c_len = max(c_lens)

        # pad 填充；mask 掩盖。使用 audio_mask_id 填充
        pad_id = self.config.audio_mask_id
        batch_input_ids = torch.full(
            (2 * B, self.config.num_audio_codebook, max_c_len), # 这个参数是shape， (2B, C, max_c_len)
            pad_id,  # fill_value: 用 MASK id 填满
            dtype=torch.long, # 整数 token ID
            device=self.device,  # 放到模型所在设备
        )
        batch_audio_mask = torch.zeros(
            (2 * B, max_c_len),  # shape: (2B, max_c_len)
            dtype=torch.bool,  # 布尔值
            device=self.device  # 放到模型所在设备
        )
        # note 控制自注意力中各序列位置是否可以相互关注，True 表示允许 attention。
        # note 详细说明（第 3 节）: books/code_notes/_batch_input_ids_and_attention_mask.md
        batch_attention_mask = torch.zeros(
            # shape (2B, 1, max_c_len, max_c_len)
            # max_c_len 所有 TTS 输入的 input_ids 张量的序列长度（S）中的最大值
            (2 * B, 1, max_c_len, max_c_len),
            dtype=torch.bool,
            device=self.device
        )

        # inp 是 {input_ids: (1,C,S), audio_mask: (1,S)}
        for i, inp in enumerate(inputs_list):
            # c_len 是当前条件输入 input_ids 的完整序列长度，u_len 是待生成目标音频的 token 帧数。
            c_len, u_len = c_lens[i], task.target_lens[i]

            # Cond (前 B 条): 完整序列, attention 在前 c_len 范围内全连通
            # note 切片赋值含义与示例（第 1 节）: books/code_notes/_batch_input_ids_and_attention_mask.md
            batch_input_ids[i, :, :c_len] = inp["input_ids"] 
            batch_audio_mask[i, :c_len] = inp["audio_mask"] 
            batch_attention_mask[i, :, :c_len, :c_len] = True

            # note 切片赋值含义与示例（第 2 节）: books/code_notes/_batch_input_ids_and_attention_mask.md
            batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]
            batch_audio_mask[B + i, :u_len] = inp["audio_mask"][..., -u_len:]
            batch_attention_mask[B + i, :, :u_len, :u_len] = True
            # 如果最大的 input_ids 序列最大长度大于待生成音频的 token 数量
            if max_c_len > u_len:
                # torch.arange(2,5)生成左闭右开的 tensor([2, 3, 4])
                pad_diag = torch.arange(
                    u_len,  # start：起始索引，待生成音频的 token 数量
                    max_c_len,  # end：结束索引， input_ids 序列最大长度
                    device=self.device  # 在模型所在设备上创建张量
                )
                batch_attention_mask[B + i, :, pad_diag, pad_diag] = True

        # ===== step 3: 初始化 target 状态为全 MASK =====
        # tokens 是 "当前已解出的 target token 状态" (B, C, T), 一开始全是 MASK,
        # 迭代过程中被逐步填成真实 codec token; 它是本函数最终要产出的东西。
        tokens = torch.full(
            (B, self.config.num_audio_codebook, max(task.target_lens)),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        # ===== step 4: 计算时间步与每步 unmask 数量 (schedules) =====
        # 生成 num_step 个时间步, t_shift < 1 时偏向先解早期(噪声大)的步
        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=gen_config.num_step,
            t_shift=gen_config.t_shift,
        ).tolist()
        # schedules[i][step] = 第 i 条样本在第 step 步要 "unmask" 多少个 token.
        # 整体满足 sum = t_len * C, 即所有 codebook × 帧数 的 mask 总量被均匀分摊.
        schedules = []
        for t_len in task.target_lens:
            total_mask = t_len * self.config.num_audio_codebook
            rem = total_mask
            sched = []
            for step in range(gen_config.num_step):
                num = (
                    rem
                    if step == gen_config.num_step - 1
                    else min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        rem,
                    )
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)

        # 层惩罚向量 (0..C-1), 用于在选位置时让靠前 codebook 优先被解 (粗→细)
        layer_ids = torch.arange(
            self.config.num_audio_codebook, device=self.device
        ).view(1, -1, 1)

        # ===== step 5: 迭代式 mask-fill 主循环 (共 num_step 轮) =====
        for step in range(gen_config.num_step):
            # 【重要】
            # (5.1) 调用到了 OmniVoice 的 forward() 方法，调用路径是：self() -> nn.Module.__call__() -> forward()
            #       一次 forward 同时跑 cond + uncond, 形状: [2B, C, S, V]，内部走 Transformer + audio_heads。
            # note 以 self(...) 调用入口为切入点的 forward 说明: books/code_notes/forward.md
            batch_logits = self(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)

            # 逐条样本处理: k = 本轮该样本要新填多少个 token (来自 schedules)
            for i in range(B):
                k = schedules[i][step]
                if k <= 0:
                    continue  # 本轮不给这条填 (调度分配为 0), 跳过

                c_len, t_len = c_lens[i], task.target_lens[i]

                # (5.2) 取出 "target 区域" 的 logits:
                #   cond:    序列尾部 t_len 个位置 (target 段被拼在 style/text/ref 之后)
                #   uncond:  序列头部 t_len 个位置 (uncond 输入没拼 style/text, target 在最前)
                c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :]
                u_logits = batch_logits[B + i : B + i + 1, :, :t_len, :]

                # (5.3) CFG 融合两路 logits, 得到每个位置的预测 token id 和置信度 score
                #       (score 越高表示模型对该位置的预测越有把握)
                pred_tokens, scores = self._predict_tokens_with_scoring(
                    c_logits, u_logits, gen_config
                )

                # (5.4) 层惩罚: codebook 越靠后分数越低, 让靠前(粗)层优先被选, 形成粗→细
                scores = scores - (layer_ids * gen_config.layer_penalty_factor)

                # (5.5) 位置温度: 给 score 加 Gumbel 噪声, 引入位置选择的随机性(非纯贪心)
                if gen_config.position_temperature > 0.0:
                    scores = _gumbel_sample(scores, gen_config.position_temperature)

                # (5.6) 已经填过的位置不能再被选, 把它们的分数设为 -inf 排除
                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(
                    sample_tokens != self.config.audio_mask_id, -float("inf")
                )

                # (5.7) 在 (C × t_len) 个位置里按分数取 top-k, 只把这 k 个位置填上预测 token
                #       (其余位置保持 MASK, 留给后续轮次)
                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                # (5.8) 把新填的 token 回写: tokens(最终结果) + cond/uncond 两份输入,
                #       让已确定的 token 成为下一轮预测的上下文。
                tokens[i : i + 1, :, :t_len] = sample_tokens
                batch_input_ids[i : i + 1, :, c_len - t_len : c_len] = sample_tokens
                batch_input_ids[B + i : B + i + 1, :, :t_len] = sample_tokens

        # ===== step 6: 裁掉 padding, 返回每条 (C, T) =====
        # tokens 是按 max(target_lens) 铺的, 每条按自己的 target_lens[i] 截取有效帧。
        return [tokens[i, :, : task.target_lens[i]] for i in range(B)]

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
        """对 cond/uncond logits 做 CFG 融合, 返回预测 token id 与 confidence.

        CFG 公式 (log 空间):  log p = log p_c + s * (log p_c - log p_u)
        s = guidance_scale, 越大越偏向条件分布.
        """
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)

        # 永远不能预测出 MASK token, 直接屏蔽
        log_probs[..., self.config.audio_mask_id] = -float("inf")

        if gen_config.class_temperature > 0.0:
            # 类别温度 > 0: 先 top-10% 过滤, 再用 Gumbel-softmax 采样
            filtered_probs = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(
                filtered_probs, gen_config.class_temperature
            ).argmax(dim=-1)
        else:
            # 贪心: 直接取最大概率 token id
            pred_tokens = log_probs.argmax(dim=-1)

        # confidence = 该位置预测分布的最大 log 概率, 作为后面选 top-k 位置的依据
        confidence_scores = log_probs.max(dim=-1)[0]

        return pred_tokens, confidence_scores


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _get_packed_mask(document_ids):
    return partial(_mask_mod_packed, document_ids)


def _mask_mod_packed(document_ids, b, h, q_idx, kv_idx):
    # 1. Sequence Packing Logic: Tokens must belong to the same document.
    # Note: The doc_id for padding tokens is -1, which will automatically not match
    # (if handled correctly) or be ignored.
    same_doc = document_ids[q_idx] == document_ids[kv_idx]
    return same_doc


def _resolve_language(language: Optional[str]) -> Union[str, None]:
    # 把用户传入的 language 规范化成模型内部使用的语言代码 (如 "zh")。
    # 入参可以是: None / "none" (语言无关)、语言代码 (如 "zh")、或语言全名 (如 "Chinese")。
    # LANG_IDS: 合法语言代码集合; LANG_NAME_TO_ID: 小写语言名 -> 语言代码 的映射。
    from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

    # 情况一: 未指定语言, 返回 None 走语言无关模式。
    if language is None or language.lower() == "none":
        return None
    # 情况二: 已经是合法的语言代码 (如 "zh"), 直接返回。
    if language in LANG_IDS:
        return language
    # 情况三: 传的是语言全名, 转小写后查映射表得到语言代码 (如 "Chinese" -> "zh")。
    key = language.lower()
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]
    # 情况四: 无法识别, 给出警告并退回 None (语言无关模式), 不中断推理。
    logger.warning(
        f"Language '{language}' is not recognized. "
        f"Please use a valid language ID (e.g., 'en', 'zh', 'ja', 'de') "
        f"or a full language name (e.g., 'English', 'Chinese', 'Japanese'). "
        f"See supported_language_ids() or supported_language_names() for details. "
        f"Falling back to None (language-agnostic mode)."
    )
    return None


def _resolve_instruct(
    instruct: Optional[str], use_zh: bool = False
) -> Union[str, None]:
    """Validate and normalise a voice-design instruct string.

    Supported instruct items (case-insensitive for English):

    English (comma + space separated):
        gender: male, female
        age: child, teenager, young adult, middle-aged, elderly
        pitch: very low pitch, low pitch, moderate pitch,
               high pitch, very high pitch
        style: whisper
        accent: american accent, british accent, australian accent, ...

    Chinese (full-width comma separated):
        gender: 男, 女
        age: 儿童, 少年, 青年, 中年, 老年
        pitch: 极低音调, 低音调, 中音调, 高音调, 极高音调
        style: 耳语
        dialect: 河南话, 陕西话, 四川话, 贵州话, 云南话,
                 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话,
                 青岛话, 东北话

    Minor issues (auto-fixed):
      - Wrong separator (half-width comma in Chinese instruct or
        full-width comma in English instruct)
      - Leading / trailing commas

    Major issues (raise ``ValueError``):
      - Unsupported or misspelled instruct items
      - Suggestions are offered for close matches

    Args:
        instruct: Raw instruct string, or ``None``.
        use_zh: If True, normalise all items to Chinese (used when the
            synthesis text contains Chinese and no accent is specified).

    Returns:
        Normalised instruct string, or ``None``.

    Raises:
        ValueError: if any instruct item is unsupported or misspelled.

    例子:
        # 规范化分隔符/大小写/首尾逗号 (use_zh=False, 保持英文)
        _resolve_instruct("Male， young adult,") -> "male, young adult"
        # 文本含中文时统一成中文形式 (use_zh=True, 中文用全角逗号分隔)
        _resolve_instruct("female, whisper", use_zh=True) -> "女，耳语"
        # 拼错会报 ValueError, 并提示最接近的候选
        _resolve_instruct("mael")            # ValueError: 建议 "male"
        # 空或 None 直接返回 None
        _resolve_instruct(None) -> None
    """
    if instruct is None:
        return None

    instruct_str = instruct.strip()
    if not instruct_str:
        return None

    # Split on both half-width and full-width commas
    raw_items = re.split(r"\s*[,，]\s*", instruct_str)
    raw_items = [x for x in raw_items if x]

    # Validate each item
    unknown = []
    normalised = []
    for raw in raw_items:
        n = raw.strip().lower()
        if n in _INSTRUCT_ALL_VALID:
            normalised.append(n)
        else:
            sug = difflib.get_close_matches(n, _INSTRUCT_ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, n, sug[0] if sug else None))

    if unknown:
        lines = []
        for raw, n, sug in unknown:
            if sug:
                lines.append(f"  '{raw}' -> '{n}' (unsupported; did you mean '{sug}'?)")
            else:
                lines.append(f"  '{raw}' -> '{n}' (unsupported)")
        err = (
            f"Unsupported instruct items found in {instruct_str}:\n"
            + "\n".join(lines)
            + "\n\nValid English items: "
            + ", ".join(sorted(_INSTRUCT_VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(_INSTRUCT_VALID_ZH))
            + "\n\nTip: Use only English or only Chinese instructs. "
            "English instructs should use comma + space (e.g. "
            "'male, indian accent'),\nChinese instructs should use full-width "
            "comma (e.g. '男，河南话')."
        )
        raise ValueError(err)

    # --- Language consistency: dialect forces Chinese, accent forces English ---
    has_dialect = any(n.endswith("话") for n in normalised)
    has_accent = any(" accent" in n for n in normalised)

    if has_dialect and has_accent:
        raise ValueError(
            "Cannot mix Chinese dialect and English accent in a single instruct. "
            "Dialects are for Chinese speech, accents for English speech."
        )

    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False

    # --- Unify to single language ---
    if use_zh:
        normalised = [_INSTRUCT_EN_TO_ZH.get(n, n) for n in normalised]
    else:
        normalised = [_INSTRUCT_ZH_TO_EN.get(n, n) for n in normalised]

    # --- Category conflict check ---
    conflicts = []
    for cat in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [n for n in normalised if n in cat]
        if len(hits) > 1:
            conflicts.append(hits)
    if conflicts:
        parts = []
        for group in conflicts:
            parts.append(" vs ".join(f"'{x}'" for x in group))
        raise ValueError(
            "Conflicting instruct items within the same category: "
            + "; ".join(parts)
            + ". Each category (gender, age, pitch, style, accent, dialect) "
            "allows at most one item."
        )

    # Determine separator based on language
    has_zh = any(any("\u4e00" <= c <= "\u9fff" for c in n) for n in normalised)
    separator = "，" if has_zh else ", "

    return separator.join(normalised)


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    ``non-verbal tags``（非语言标签）表示不是普通文字、但需要模型生成的
    笑声、叹气、语气或情绪声音，例如 ``[laughter]``（笑声）和
    ``[sigh]``（叹气）。例如输入 ``"你好[laughter]很高兴见到你"`` 时，
    会分别处理 ``"你好"``、``"[laughter]"`` 和 ``"很高兴见到你"``，
    避免标签的 token ID 受到前后中文或英文文本影响。

    一句话理解：将标签单独分割出来、单独 tokenize（标签的 ``[]`` 原样保留参与
    tokenize，不剥离），再与普通文字段拼接，从而保证标签 token id 的稳定性。

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs tensor of shape (1, seq_len).
    """
    parts = []  # 收集每一段(普通文字段 / 标签段)的 token id 列表，最后按序拼接
    last_end = 0  # 上一个已处理片段的结束位置（游标），用于切出"标签之间的普通文字"

    # 用正则扫描出所有非语言标签 [laughter]、[sigh] 等，逐个处理
    # finditer 产出每个匹配的 Match 对象，含 start()/end()/group()
    for m in _NONVERBAL_PATTERN.finditer(text):
        # m.start() > last_end 说明当前标签前面还有"普通文字段"未处理（即上一个标签结尾到本标签开头之间有内容）
        # 例如 "你好[laughter]很高兴" 中，处理到 [laughter] 时，last_end=0、m.start()=2，
        # 此时需要先切出 "你好" 这段普通文字单独 tokenize
        if m.start() > last_end:
            segment = text[last_end : m.start()]  # 切出标签前面的普通文字段
            # add_special_tokens=False: 只切普通子词，不自动加 [CLS]/[SEP] 等特殊 token。
            # 因为这里要把多段分别 tokenize 再拼接，若每段都自动加特殊 token，
            # 拼接后会出现重复的 [CLS]/[SEP]，破坏整个序列结构（特殊 token 应由上层统一加一次）。
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:  # 该段可能为空或 token 化后为空，非空才收集
                parts.append(ids)

        # 标签本身（如 "[laughter]"）单独 tokenize，保证其 token id 不受前后文语言影响
        # 同样用 add_special_tokens=False，避免给标签段单独塞特殊 token
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)

        last_end = m.end()  # 移动游标到当前标签结尾，为下一轮切分做准备

    # 循环结束后，若游标还没到文本末尾，说明末尾还有普通文字段（最后一个标签之后的内容）需要处理
    # 例如 "你好[laughter]很高兴见到你" 处理完 [laughter] 后，last_end=12，末尾 "很高兴见到你" 在此切出
    if last_end < len(text):
        segment = text[last_end:]  # 切出最后一个标签之后的普通文字段
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)

    # 兜底：如果整个文本没有任何标签（parts 为空），直接整体 tokenize
    # 注意这里用了 return_tensors="pt" 直接返回 tensor，与下面的拼接分支保持一致的输出形状
    if not parts:
        result = tokenizer(text, return_tensors="pt").input_ids
    else:
        # 将各段 token id 列表按出现顺序展平拼接成一维序列，再包成 (1, seq_len) 的 tensor
        combined = []
        for p in parts:
            combined.extend(p)
        result = torch.tensor([combined], dtype=torch.long)
    return result


def _combine_text(text, ref_text: Optional[str] = None) -> str:
    """合并参考文本与目标文本，并清理成 tokenizer 使用的单行字符串。

    Args:
        text: 准备合成语音的目标文本。
        ref_text: 参考音频对应的转写文本；没有参考文本时传 ``None``。

    Returns:
        清理后的完整文本。存在 ``ref_text`` 时，顺序为“参考文本 + 目标文本”；
        否则只返回清理后的目标文本。

    Example:
        输入::

            ref_text = " 你好，我是小明。\\n"
            text = "  今天（天气）  很好。\\t "

        输出::

            "你好，我是小明。今天(天气)很好。"
    """

    # 有参考文本时，分别删除两段文本首尾的空白，再按
    # “参考文本 + 一个空格 + 目标文本”的顺序拼接。
    # ref_text 是参考音频说了什么，text 是希望模型接着生成什么。
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        # 没有参考文本（如 Auto / Voice Design 模式）时，只清理目标文本首尾空白。
        full_text = text.strip()

    # 删除换行符 \n 和回车符 \r，把多行输入整理成单行。
    full_text = re.sub(r"[\r\n]+", "", full_text)

    # 将中文全角括号替换为英文半角括号，统一标点形式。
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")

    # 把连续的普通空格或制表符压缩成一个空格。
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # 删除汉字前后的空白。例如“你好 世界”会变成“你好世界”。
    # chinese_range 匹配一个常用汉字；两个分支分别匹配“汉字后的空白”
    # 和“汉字前的空白”。
    chinese_range = r"[\u4e00-\u9fff]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    # 返回 tokenizer 最终接收的单行文本。
    return full_text


# ---------------------------------------------------------------------------
# Register with HuggingFace Auto classes
# ---------------------------------------------------------------------------

AutoConfig.register("omnivoice", OmniVoiceConfig)
AutoModel.register(OmniVoiceConfig, OmniVoice)
