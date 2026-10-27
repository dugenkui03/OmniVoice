"""学习分支合并回归：使用小型 CPU 模型和替身，不下载权重或启动训练任务。"""

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import Qwen3Config

import omnivoice.models.omnivoice as model_module
from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceConfig,
    OmniVoiceGenerationConfig,
    VoiceClonePrompt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_model():
    """随机初始化小型 Qwen3 主干，避免依赖真实 checkpoint 与 GPU。"""
    llm_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attn_implementation="sdpa",
    )
    config = OmniVoiceConfig(
        llm_config=llm_config,
        audio_vocab_size=9,
        audio_mask_id=8,
        num_audio_codebook=2,
        audio_codebook_weights=[1, 1],
    )
    return OmniVoice(config).eval()


def test_embedding_access_and_forward_backward(tiny_model):
    """Mixin 与拆行后的嵌入计算必须保留文本/音频选择和可反向传播的损失。"""
    replacement = torch.nn.Embedding(32, 16)
    tiny_model.set_input_embeddings(replacement)
    assert tiny_model.get_input_embeddings() is replacement
    ids = torch.tensor([[[1, 2, 3], [1, 4, 5]]])
    mask = torch.tensor([[False, True, True]])
    shifted = ids * mask.unsqueeze(1) + tiny_model.codebook_layer_offsets.view(1, -1, 1)
    expected = torch.where(
        mask.unsqueeze(-1),
        tiny_model.audio_embeddings(shifted).sum(dim=1),
        replacement(ids[:, 0, :]),
    )
    torch.testing.assert_close(tiny_model._prepare_embed_inputs(ids, mask), expected)
    labels = ids.clone()
    labels[:, :, 0] = -100
    result = tiny_model(
        input_ids=ids,
        audio_mask=mask,
        labels=labels,
        attention_mask=torch.ones(1, 1, 3, 3, dtype=torch.bool),
    )
    assert result.logits.shape == (1, 2, 3, 9)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert replacement.weight.grad is not None
    assert torch.isfinite(tiny_model.audio_heads.weight.grad).all()


@pytest.mark.parametrize("guidance_scale", [0.0, 2.0])
def test_prediction_ids_and_confidence(tiny_model, guidance_scale):
    """CFG、MASK 屏蔽及 max()[0] 拆行后仍返回正确的 ID 和最大分数。"""
    c_logits = torch.arange(9, dtype=torch.float32).expand(1, 2, 3, 9)
    u_logits = torch.zeros_like(c_logits)
    config = OmniVoiceGenerationConfig(guidance_scale=guidance_scale)
    pred, scores = tiny_model._predict_tokens_with_scoring(c_logits, u_logits, config)
    c_log = c_logits.log_softmax(-1)
    expected = (
        (c_log + guidance_scale * (c_log - u_logits.log_softmax(-1))).log_softmax(-1)
        if guidance_scale
        else c_log
    )
    expected[..., 8] = -float("inf")
    assert pred.shape == scores.shape == (1, 2, 3)
    assert (pred == 7).all()
    torch.testing.assert_close(scores, expected.max(-1).values)


def test_preprocessing_preserves_duration_ratio_and_reference(tiny_model, monkeypatch):
    """指定时长前保留估算值；文本规范化只作用于目标，不改变参考转写。"""
    tiny_model.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=25))
    estimate = Mock(return_value=50)
    monkeypatch.setattr(tiny_model, "_estimate_target_tokens", estimate)
    normalize = Mock(side_effect=lambda text, lang: "normalized " + text)
    monkeypatch.setattr(model_module, "_normalize_text", normalize)
    prompt = VoiceClonePrompt(torch.ones(2, 3, dtype=torch.long), "reference 7", 0.1)
    task = tiny_model._preprocess_all(
        text=["target 1", "target 2"],
        language="en",
        voice_clone_prompt=prompt,
        duration=[0.08, None],
        speed=[3.0, 2.0],
        normalize_text=True,
    )
    assert task.texts == ["normalized target 1", "normalized target 2"]
    assert task.ref_texts == ["reference 7", "reference 7"]
    assert task.target_lens == [2, 50]
    assert task.speed == [25.0, 2.0]
    assert [call.kwargs["speed"] for call in estimate.call_args_list] == [1.0, 2.0]
    assert normalize.call_count == 2


@pytest.mark.parametrize(
    "device,dtype", [("cpu", torch.float32), ("cuda:2", torch.float16)]
)
def test_asr_saved_options_reach_pipeline(tiny_model, monkeypatch, device, dtype):
    """按需加载 ASR 时保留指定模型与设备；替身禁止实际下载或占用 GPU。"""
    tiny_model._asr_model_name = "local-whisper"
    tiny_model._asr_device = device
    monkeypatch.setattr(model_module, "_resolve_model_path", lambda name: name)
    pipeline = Mock()
    monkeypatch.setattr("transformers.pipeline", pipeline)
    tiny_model.load_asr_model()
    assert pipeline.call_args.kwargs == {
        "model": "local-whisper",
        "dtype": dtype,
        "device": device,
    }
    assert tiny_model._asr_pipe is pipeline.return_value


def test_from_pretrained_preserves_lazy_asr_options(tiny_model, monkeypatch):
    """不立即加载 ASR 时也要保存模型名和设备，避免后续转写回退到默认值。"""
    monkeypatch.setattr(model_module, "_resolve_model_path", lambda name: name)
    monkeypatch.setattr(
        model_module.PreTrainedModel, "from_pretrained", Mock(return_value=tiny_model)
    )
    monkeypatch.setattr(model_module.AutoTokenizer, "from_pretrained", Mock())
    monkeypatch.setattr(
        model_module.HiggsAudioV2TokenizerModel, "from_pretrained", Mock()
    )
    monkeypatch.setattr(
        model_module.AutoFeatureExtractor,
        "from_pretrained",
        Mock(return_value=SimpleNamespace(sampling_rate=24000)),
    )
    load_asr = Mock()
    monkeypatch.setattr(tiny_model, "load_asr_model", load_asr)
    loaded = OmniVoice.from_pretrained(
        "local-model", asr_model_name="custom-whisper", asr_device="cpu", load_asr=False
    )
    assert loaded is tiny_model
    assert loaded._asr_model_name == "custom-whisper"
    assert loaded._asr_device == "cpu"
    load_asr.assert_not_called()


def test_postprocessing_uses_generation_config(tiny_model):
    """上游新增的 pad/fade 参数必须传到波形后处理，而不是固定为默认时长。"""
    tiny_model.sampling_rate = 1000
    config = OmniVoiceGenerationConfig.from_dict(
        {"postprocess_output": False, "pad_duration": 0.003, "fade_duration": 0.002}
    )
    output = tiny_model._post_process_audio(
        np.ones((1, 20)), ref_rms=0.1, gen_config=config
    )
    assert output.shape == (1, 26)
    np.testing.assert_array_equal(output[:, :3], 0)
    np.testing.assert_array_equal(output[:, -3:], 0)
    # 淡入/淡出各两点，不能仅添加静音而忽略 fade_duration。
    np.testing.assert_array_equal(output[:, 3:5], [[0, 1]])
    np.testing.assert_array_equal(output[:, -5:-3], [[1, 0]])


def test_optimizer_excludes_frozen_parameters(tiny_model):
    """冻结主干后，优化器只绑定可训练权重，保留上游 LoRA 参数筛选行为。"""
    from omnivoice.training.config import TrainingConfig
    from omnivoice.training.trainer import OmniTrainer

    tiny_model.llm.requires_grad_(False)
    trainer = object.__new__(OmniTrainer)
    trainer.model = tiny_model
    trainer.config = TrainingConfig(steps=10, warmup_type="steps", warmup_steps=0)
    optimizer, _ = trainer.create_optimizer_and_scheduler()
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for p in tiny_model.parameters() if p.requires_grad}
    assert actual == expected


def test_emilia_training_command_keeps_all_arguments():
    """执行训练阶段的 shell 语法，用替身捕获参数，不启动 Accelerate。"""
    script = (ROOT / "examples/run_emilia.sh").read_text()
    stage = script.split("# Stage 2: Train", 1)[1]
    setup = """
set -eu
stage=2; stop_stage=2; GPU_IDS=0; NUM_GPUS=1
TRAIN_CONFIG=train.json; data_config=data.json; OUTPUT_DIR=exp
accelerate() { printf '<%s>\\n' "$@"; }
"""
    result = subprocess.run(
        ["bash", "-c", setup + stage], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    for argument in [
        "--train_config",
        "train.json",
        "--data_config",
        "data.json",
        "--output_dir",
        "exp",
    ]:
        assert f"<{argument}>" in result.stdout


@pytest.mark.parametrize("override", [None, "/tmp/omnivoice custom models"])
def test_download_directory_is_portable(tmp_path, override):
    """检查脚本默认目录和显式覆盖；替换 mkdir/uv，确保不创建模型目录或下载。"""
    env = os.environ.copy()
    env.pop("MODEL_DIR", None)
    if override is not None:
        env["MODEL_DIR"] = override
    harness = """
mkdir() { return 0; }
uv() { printf 'ARG:%s\\n' "$@"; return 77; }
export -f mkdir uv
exec bash "$1"
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "test",
            str(ROOT / "scripts/download_omnivoice_weight_hf_mirror.sh"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 77, result.stderr
    expected = override or str(ROOT / ".models/OmniVoice")
    assert f"ARG:{expected}\n" in result.stdout
