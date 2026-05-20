#!/usr/bin/env python3
"""Generate a Zhen Huan style learning podcast with OmniVoice TTS."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from omnivoice.models.omnivoice import OmniVoice


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / ".models" / "OmniVoice"
DEFAULT_AI_DUBBING_RUN = Path(
    "/Users/dugenkui/workspace/ai_dubbing/test_data/测试输出/"
    "batch_english_learning_from_E06_queue_20260510_213936/runs/"
    "甄嬛传_E11_下集_教科书级向上管理_曹贵人巧言化解华妃雷霆之怒"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "podcast"
SAMPLE_RATE = 24000


@dataclass(frozen=True)
class VoiceReferenceSpec:
    """Describe which original episode lines should become one clone reference."""

    key: str
    speaker: str
    style: str
    line_ids: tuple[int, ...]


@dataclass(frozen=True)
class VoiceReference:
    """Store the generated clone reference audio path and its transcript."""

    key: str
    speaker: str
    style: str
    line_ids: tuple[int, ...]
    audio_path: str
    ref_text: str
    duration_sec: float


@dataclass(frozen=True)
class PodcastLine:
    """Store one podcast utterance and the TTS controls for that utterance."""

    line_id: str
    part: str
    speaker: str
    reference_key: str
    emotion: str
    text: str
    duration_sec: float
    pause_after_sec: float


VOICE_REFERENCE_SPECS = (
    VoiceReferenceSpec(
        key="taijian_notice",
        speaker="太监",
        style="通报、主持、略带恭敬",
        line_ids=(0, 1),
    ),
    VoiceReferenceSpec(
        key="huafei_angry",
        speaker="华妃",
        style="强势、愤怒、压迫感",
        line_ids=(7, 8, 9, 10),
    ),
    VoiceReferenceSpec(
        key="huafei_power",
        speaker="华妃",
        style="责问、失权后的怒气",
        line_ids=(17, 18, 19, 21, 22),
    ),
    VoiceReferenceSpec(
        key="huafei_skeptical",
        speaker="华妃",
        style="怀疑、冷笑、短促",
        line_ids=(24, 25, 26, 35, 42),
    ),
    VoiceReferenceSpec(
        key="huafei_soft",
        speaker="华妃",
        style="收束、缓和、仍带威严",
        line_ids=(36, 37, 39, 45, 46),
    ),
    VoiceReferenceSpec(
        key="cao_soft",
        speaker="曹贵人",
        style="请罪、低姿态、温和",
        line_ids=(2, 3, 4, 5, 6),
    ),
    VoiceReferenceSpec(
        key="cao_strategy",
        speaker="曹贵人",
        style="分析、劝解、谋略感",
        line_ids=(27, 29, 30, 31, 33, 34),
    ),
    VoiceReferenceSpec(
        key="cao_service",
        speaker="曹贵人",
        style="效劳、转圜、稳住局面",
        line_ids=(40, 41, 43, 44),
    ),
)


PODCAST_LINES = (
    PodcastLine(
        "001",
        "开场",
        "周宁海",
        "taijian_notice",
        "随手开麦",
        "[question-ah] 哎，麦开了吗？...开了哈。娘娘，曹贵人，今天不升堂，咱们就随便聊两句。",
        6.5,
        0.22,
    ),
    PodcastLine(
        "002",
        "开场",
        "华妃",
        "huafei_angry",
        "半开玩笑",
        "[dissatisfaction-hnn] 轻松？本宫刚演完那场，火气还没散呢。[sigh] 你们先别笑。",
        6.0,
        0.2,
    ),
    PodcastLine(
        "003",
        "开场",
        "曹贵人",
        "cao_soft",
        "接话",
        "[laughter] 哈哈哈...我先声明啊，戏里请罪，戏外聊天，不许再让我跪。",
        5.8,
        0.22,
    ),
    PodcastLine(
        "004",
        "开场",
        "周宁海",
        "taijian_notice",
        "主持",
        "[confirmation-en] 行。那我先把茶放这儿...咱们从刚才那场开始，聊着聊着，再捡几个英文词。",
        6.9,
        0.28,
    ),
    PodcastLine(
        "005",
        "剧情",
        "华妃",
        "huafei_skeptical",
        "抛问题",
        "曹贵人，我真想问你：[question-ah] 你一进来那句夏日炎炎，到底是灭火，还是试探我？",
        6.5,
        0.22,
    ),
    PodcastLine(
        "006",
        "剧情",
        "曹贵人",
        "cao_soft",
        "回应",
        "嗯...都有一点。[sigh] 您那会儿像一锅刚开的水，我总不能一上来就加盐吧。",
        6.1,
        0.22,
    ),
    PodcastLine(
        "007",
        "剧情",
        "华妃",
        "huafei_skeptical",
        "被逗笑",
        "加盐？[surprise-oh] [laughter] 你是真会说话。怪不得戏里跪着，也像在开会。",
        5.8,
        0.22,
    ),
    PodcastLine(
        "008",
        "剧情",
        "周宁海",
        "taijian_notice",
        "旁观接梗",
        "我小声说啊...[surprise-wa] 我在旁边听着都愣了。表面是请罪，实际像带着方案来汇报。",
        6.2,
        0.22,
    ),
    PodcastLine(
        "009",
        "剧情",
        "曹贵人",
        "cao_strategy",
        "解释",
        "没办法呀...[sigh] 娘娘失了协理六宫之权，这不是小事。",
        5.2,
        0.2,
    ),
    PodcastLine(
        "010",
        "剧情",
        "华妃",
        "huafei_power",
        "吐槽",
        "[dissatisfaction-hnn] 是啊...协理六宫没了，差不多就是管理员权限被撤了。",
        5.4,
        0.2,
    ),
    PodcastLine(
        "011",
        "剧情",
        "周宁海",
        "taijian_notice",
        "笑场",
        "[laughter] 哈哈哈，后宫群聊里，娘娘突然不能踢人了。",
        4.2,
        0.2,
    ),
    PodcastLine(
        "012",
        "剧情",
        "周宁海",
        "taijian_notice",
        "追问",
        "我压低声音问一句啊...[question-oh] 曹贵人，你说恭喜娘娘的时候，真不怕娘娘当场翻脸？",
        6.6,
        0.22,
    ),
    PodcastLine(
        "013",
        "剧情",
        "华妃",
        "huafei_skeptical",
        "补刀",
        "我当时心里真想说：[question-ah] 你是不是串台了？...但我忍住了。",
        5.0,
        0.2,
    ),
    PodcastLine(
        "014",
        "剧情",
        "曹贵人",
        "cao_strategy",
        "接招",
        "[sigh] 怕呀。但不这么说，娘娘就只会继续骂我。我得先让您问一句，喜从何来。",
        6.9,
        0.22,
    ),
    PodcastLine(
        "015",
        "剧情",
        "周宁海",
        "taijian_notice",
        "点题",
        "[surprise-oh] 哦，这就是钩子。不是解释，是先把对方的话接出来。",
        5.2,
        0.22,
    ),
    PodcastLine(
        "016",
        "剧情",
        "华妃",
        "huafei_skeptical",
        "追问",
        "[question-ah] 那你接下来怎么圆？本宫都失权了，你还说是喜。",
        5.2,
        0.22,
    ),
    PodcastLine(
        "017",
        "剧情",
        "曹贵人",
        "cao_strategy",
        "解释证据",
        "[confirmation-en] 我圆的不是权，是证据。丽嫔说得再多，也只是风言风语。",
        6.0,
        0.22,
    ),
    PodcastLine(
        "018",
        "剧情",
        "周宁海",
        "taijian_notice",
        "回应",
        "[question-oh] 也就是说，不是没危险，是暂时没有实锤。",
        4.3,
        0.2,
    ),
    PodcastLine(
        "019",
        "剧情",
        "曹贵人",
        "cao_strategy",
        "点头",
        "[confirmation-en] 对。没有实锤，就还有转身的余地。",
        3.9,
        0.22,
    ),
    PodcastLine(
        "020",
        "剧情",
        "华妃",
        "huafei_power",
        "接话",
        "[sigh] 可本宫气的不是这个。本宫气的是，连见皇上一面都难。",
        5.7,
        0.22,
    ),
    PodcastLine(
        "021",
        "剧情",
        "曹贵人",
        "cao_service",
        "回应",
        "[confirmation-en] 所以我才提温宜。皇上念旧，也疼孩子。",
        4.7,
        0.2,
    ),
    PodcastLine(
        "022",
        "剧情",
        "华妃",
        "huafei_skeptical",
        "反应",
        "[surprise-oh] 你这话一说，本宫就知道，你不是空手来的。",
        4.7,
        0.2,
    ),
    PodcastLine(
        "023",
        "剧情",
        "周宁海",
        "taijian_notice",
        "接梗",
        "[laughter] 哈哈哈，曹贵人随身带方案，入宫自带 PPT。",
        4.5,
        0.2,
    ),
    PodcastLine(
        "024",
        "剧情",
        "曹贵人",
        "cao_soft",
        "笑",
        "[laughter] 别这么说嘛，我那叫提前准备，不叫 PPT。",
        4.5,
        0.2,
    ),
    PodcastLine(
        "025",
        "剧情",
        "华妃",
        "huafei_soft",
        "收束",
        "[sigh] 所以赐座不是原谅，是本宫愿意听下一页。",
        4.7,
        0.28,
    ),
    PodcastLine(
        "026",
        "生词",
        "周宁海",
        "taijian_notice",
        "转场",
        "[confirmation-en] 好，剧情先到这。咱们顺手捡词，别紧张，都是刚才聊出来的。",
        6.0,
        0.25,
    ),
    PodcastLine(
        "027",
        "生词",
        "华妃",
        "huafei_skeptical",
        "抛词",
        "[question-ah] 第一个是 hardly，对吧？我那会儿 hardly calm。",
        4.7,
        0.2,
    ),
    PodcastLine(
        "028",
        "生词",
        "曹贵人",
        "cao_strategy",
        "回应 hardly",
        "[confirmation-en] 对，hardly 是几乎不。hardly calm，就是几乎不冷静。",
        5.4,
        0.2,
    ),
    PodcastLine(
        "029",
        "生词",
        "周宁海",
        "taijian_notice",
        "接梗",
        "[laughter] 这个例句太贴了，娘娘刚才确实不太 calm。",
        4.2,
        0.2,
    ),
    PodcastLine(
        "030",
        "生词",
        "华妃",
        "huafei_skeptical",
        "笑骂",
        "[laughter] 周宁海，你最近胆子见长啊。",
        3.5,
        0.2,
    ),
    PodcastLine(
        "031",
        "生词",
        "曹贵人",
        "cao_strategy",
        "抛词 build",
        "[confirmation-en] 第二个词是 build，建立。娘娘说提拔我，就是 helped me build my position。",
        6.8,
        0.22,
    ),
    PodcastLine(
        "032",
        "生词",
        "周宁海",
        "taijian_notice",
        "回应 build",
        "[surprise-wa] 这个好记。宫里的位置不是坐上去，是 build 出来的。",
        4.8,
        0.2,
    ),
    PodcastLine(
        "033",
        "生词",
        "华妃",
        "huafei_power",
        "抛词 loss",
        "[question-ah] 那 loss 呢？本宫这一集最懂 loss。",
        4.0,
        0.2,
    ),
    PodcastLine(
        "034",
        "生词",
        "曹贵人",
        "cao_service",
        "回应 loss",
        "[sigh] loss 是损失。loss of power，就是失去权力。",
        4.6,
        0.2,
    ),
    PodcastLine(
        "035",
        "生词",
        "周宁海",
        "taijian_notice",
        "接梗",
        "[dissatisfaction-hnn] 说轻了是 loss，说重了是娘娘的心口一堵。",
        4.5,
        0.22,
    ),
    PodcastLine(
        "036",
        "生词",
        "华妃",
        "huafei_skeptical",
        "笑",
        "[laughter] 哈哈哈，这句可以记。",
        2.6,
        0.2,
    ),
    PodcastLine(
        "037",
        "生词",
        "曹贵人",
        "cao_strategy",
        "抛词 secure",
        "[confirmation-en] secure，是保全。secure your own safety，就是明哲保身。",
        5.2,
        0.2,
    ),
    PodcastLine(
        "038",
        "生词",
        "周宁海",
        "taijian_notice",
        "回应 secure",
        "[surprise-oh] 这词也像曹贵人的风格。先保命，再谈翻盘。",
        4.7,
        0.2,
    ),
    PodcastLine(
        "039",
        "生词",
        "曹贵人",
        "cao_soft",
        "调侃",
        "[laughter] 话别说这么直嘛，哈哈哈。",
        3.2,
        0.2,
    ),
    PodcastLine(
        "040",
        "生词",
        "华妃",
        "huafei_soft",
        "抛词 past",
        "[sigh] past 是过去。皇上念旧，就是 values his past。",
        4.9,
        0.2,
    ),
    PodcastLine(
        "041",
        "生词",
        "周宁海",
        "taijian_notice",
        "接话",
        "[confirmation-en] 所以温宜一出来，旧情这扇门就没完全关上。",
        4.4,
        0.2,
    ),
    PodcastLine(
        "042",
        "生词",
        "曹贵人",
        "cao_service",
        "抛词 favourite",
        "[confirmation-en] 最后 favourite，特别受喜欢的人。温宜就是皇上的 favourite。",
        5.4,
        0.22,
    ),
    PodcastLine(
        "043",
        "结尾",
        "华妃",
        "huafei_skeptical",
        "总结",
        "[sigh] 这一场说白了，就是本宫火很大，曹贵人嘴很稳。",
        5.2,
        0.2,
    ),
    PodcastLine(
        "044",
        "结尾",
        "曹贵人",
        "cao_soft",
        "回应",
        "[laughter] 娘娘火大，我不稳一点，咱俩就没有今天这期播客了。",
        5.4,
        0.2,
    ),
    PodcastLine(
        "045",
        "结尾",
        "周宁海",
        "taijian_notice",
        "笑",
        "[laughter] 对，那我也没有主持费了，哈哈哈。",
        3.6,
        0.2,
    ),
    PodcastLine(
        "046",
        "结尾",
        "华妃",
        "huafei_soft",
        "收束",
        "[confirmation-en] 行了，今天就到这儿。曹贵人，这次先赐座。",
        4.8,
        0.2,
    ),
    PodcastLine(
        "047",
        "结尾",
        "曹贵人",
        "cao_service",
        "告别",
        "[sigh] 多谢娘娘。各位听众，记得把 hardly、build、loss 都带走。",
        5.6,
        0.2,
    ),
    PodcastLine(
        "048",
        "结尾",
        "周宁海",
        "taijian_notice",
        "告别",
        "[confirmation-en] 咱们下回继续边聊剧情边背词。收麦，收麦。",
        4.6,
        0.0,
    ),
)


def get_best_device() -> str:
    """Choose the fastest available local device for OmniVoice inference."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the podcast generator."""

    parser = argparse.ArgumentParser(
        description="Generate a Zhen Huan E11 learning podcast with OmniVoice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--ai-dubbing-run", type=Path, default=DEFAULT_AI_DUBBING_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--t-shift", type=float, default=0.1)
    parser.add_argument("--preprocess-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--export-mp3", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_logging() -> None:
    """Configure console logging for long-running local generation."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        force=True,
    )


def load_episode_analysis(run_dir: Path) -> dict[str, Any]:
    """Load the E11 analysis JSON produced by the ai_dubbing project."""

    analysis_path = run_dir / "episode_analyses" / "ep11_analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing episode analysis: {analysis_path}")
    return json.loads(analysis_path.read_text(encoding="utf-8"))


def get_dialogue_text(dialogue: list[dict[str, Any]], line_id: int) -> str:
    """Return the Chinese transcript for one original episode dialogue line."""

    return str(dialogue[line_id]["text_zh"]).strip()


def get_segment_path(run_dir: Path, line_id: int) -> Path:
    """Return the original Chinese line audio path for one dialogue index."""

    path = run_dir / "episode_cache" / "ep11" / "tts_segments" / f"line_{line_id:04d}.wav"
    if not path.exists():
        raise FileNotFoundError(f"Missing source line audio: {path}")
    return path


def read_mono_audio(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read an audio file as mono float32 samples at the target sample rate."""

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr == target_sr:
        return mono.astype(np.float32, copy=False)

    wav = torch.from_numpy(mono).unsqueeze(0)
    resampled = torchaudio.functional.resample(wav, sr, target_sr)
    return resampled.squeeze(0).numpy().astype(np.float32, copy=False)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono float audio to a WAV file, creating parent directories first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def build_reference_audio(
    spec: VoiceReferenceSpec,
    dialogue: list[dict[str, Any]],
    run_dir: Path,
    ref_dir: Path,
) -> VoiceReference:
    """Create one reusable voice-clone reference WAV from selected episode lines."""

    chunks: list[np.ndarray] = []
    texts: list[str] = []
    silence = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)

    for line_id in spec.line_ids:
        chunks.append(read_mono_audio(get_segment_path(run_dir, line_id)))
        chunks.append(silence)
        texts.append(get_dialogue_text(dialogue, line_id))

    merged = np.concatenate(chunks[:-1]) if chunks else np.zeros(0, dtype=np.float32)
    ref_path = ref_dir / f"{spec.key}.wav"
    write_wav(ref_path, merged)

    return VoiceReference(
        key=spec.key,
        speaker=spec.speaker,
        style=spec.style,
        line_ids=spec.line_ids,
        audio_path=str(ref_path),
        ref_text="".join(texts),
        duration_sec=round(float(merged.shape[-1] / SAMPLE_RATE), 3),
    )


def build_voice_references(
    dialogue: list[dict[str, Any]],
    run_dir: Path,
    ref_dir: Path,
) -> dict[str, VoiceReference]:
    """Build all configured character voice references for podcast generation."""

    refs = {}
    for spec in VOICE_REFERENCE_SPECS:
        refs[spec.key] = build_reference_audio(spec, dialogue, run_dir, ref_dir)
    return refs


def select_podcast_lines(max_lines: int | None) -> list[PodcastLine]:
    """Return the requested prefix of the built-in E11 podcast script."""

    lines = list(PODCAST_LINES)
    if max_lines is None:
        return lines
    return lines[:max_lines]


def write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON with stable formatting for inspection and reruns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_script_markdown(
    path: Path,
    lines: list[PodcastLine],
    refs: dict[str, VoiceReference],
) -> None:
    """Write a human-readable podcast script with voice reference metadata."""

    total_duration = sum(line.duration_sec + line.pause_after_sec for line in lines)
    chunks = [
        "# 甄嬛版 E11 下中文学习播客脚本",
        "",
        f"- 预计时长：{total_duration / 60:.1f} 分钟",
        f"- 台词数：{len(lines)}",
        "",
        "## 声音参考",
        "",
    ]
    for ref in refs.values():
        line_ids = ", ".join(f"line_{line_id:04d}" for line_id in ref.line_ids)
        chunks.append(
            f"- `{ref.key}`：{ref.speaker}，{ref.style}，"
            f"{ref.duration_sec:.2f}s，素材 {line_ids}"
        )

    chunks.extend(["", "## 正文", ""])
    for line in lines:
        chunks.append(
            f"**{line.line_id}. {line.part}｜{line.speaker}｜{line.emotion}｜"
            f"`{line.reference_key}`**"
        )
        chunks.append("")
        chunks.append(line.text)
        chunks.append("")

    path.write_text("\n".join(chunks), encoding="utf-8")


def load_model(model_dir: Path, device: str) -> OmniVoice:
    """Load OmniVoice from a local checkpoint using the same API as the CLI."""

    dtype = torch.float32 if device == "cpu" else torch.float16
    logging.info("Loading OmniVoice model from %s on %s", model_dir, device)
    return OmniVoice.from_pretrained(str(model_dir), device_map=device, dtype=dtype)


def create_voice_prompt_cache(
    model: OmniVoice,
    refs: dict[str, VoiceReference],
    preprocess_prompt: bool,
) -> dict[str, Any]:
    """Convert reference WAVs into reusable OmniVoice voice-clone prompts."""

    cache = {}
    for key, ref in refs.items():
        logging.info("Creating voice prompt: %s (%s)", key, ref.style)
        cache[key] = model.create_voice_clone_prompt(
            ref_audio=ref.audio_path,
            ref_text=ref.ref_text,
            preprocess_prompt=preprocess_prompt,
        )
    return cache


def get_segment_output_path(segment_dir: Path, line: PodcastLine) -> Path:
    """Return the deterministic WAV output path for one generated podcast line."""

    safe_speaker = line.speaker.replace("/", "_")
    return segment_dir / f"{line.line_id}_{safe_speaker}_{line.reference_key}.wav"


def generate_one_line(
    model: OmniVoice,
    line: PodcastLine,
    prompt_cache: dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate one podcast line and return timing metadata for reports."""

    start = time.time()
    audios = model.generate(
        text=line.text,
        language="zh",
        voice_clone_prompt=prompt_cache[line.reference_key],
        duration=line.duration_sec,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        postprocess_output=args.postprocess_output,
    )
    elapsed = time.time() - start
    write_wav(output_path, audios[0], model.sampling_rate)
    actual_duration = float(len(audios[0]) / model.sampling_rate)
    return {
        "line_id": line.line_id,
        "part": line.part,
        "speaker": line.speaker,
        "reference_key": line.reference_key,
        "target_duration_sec": line.duration_sec,
        "audio_duration_sec": round(actual_duration, 3),
        "generation_wall_time_sec": round(elapsed, 3),
        "rtf": round(elapsed / actual_duration, 3) if actual_duration else None,
        "output_file": str(output_path),
        "text": line.text,
    }


def generate_segments(
    model: OmniVoice,
    lines: list[PodcastLine],
    refs: dict[str, VoiceReference],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Generate or reuse all podcast segment WAVs."""

    segment_dir = output_dir / "segments"
    prompt_cache = create_voice_prompt_cache(model, refs, args.preprocess_prompt)
    timing_rows = []

    for index, line in enumerate(lines, start=1):
        output_path = get_segment_output_path(segment_dir, line)
        if output_path.exists() and not args.overwrite:
            audio_info = sf.info(output_path)
            logging.info("Reusing existing segment %s/%s: %s", index, len(lines), output_path)
            timing_rows.append(
                {
                    "line_id": line.line_id,
                    "part": line.part,
                    "speaker": line.speaker,
                    "reference_key": line.reference_key,
                    "target_duration_sec": line.duration_sec,
                    "audio_duration_sec": round(float(audio_info.duration), 3),
                    "generation_wall_time_sec": 0.0,
                    "rtf": 0.0,
                    "output_file": str(output_path),
                    "text": line.text,
                }
            )
            continue

        logging.info("Generating segment %s/%s: %s %s", index, len(lines), line.line_id, line.speaker)
        timing_rows.append(generate_one_line(model, line, prompt_cache, output_path, args))

    return timing_rows


def write_timing_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-line generation timing to a TSV file."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def concatenate_segments(
    lines: list[PodcastLine],
    segment_dir: Path,
    output_path: Path,
) -> float:
    """Concatenate generated segment WAVs with scripted pauses."""

    chunks: list[np.ndarray] = []
    for line in lines:
        path = get_segment_output_path(segment_dir, line)
        audio = read_mono_audio(path)
        chunks.append(audio)
        if line.pause_after_sec > 0:
            chunks.append(np.zeros(int(SAMPLE_RATE * line.pause_after_sec), dtype=np.float32))

    combined = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0.98:
        combined = combined / peak * 0.98
    write_wav(output_path, combined)
    return float(combined.shape[-1] / SAMPLE_RATE)


def export_mp3_if_available(wav_path: Path, mp3_path: Path) -> bool:
    """Export an MP3 copy with ffmpeg when ffmpeg exists on this machine."""

    if shutil.which("ffmpeg") is None:
        logging.warning("ffmpeg not found; skipping MP3 export.")
        return False
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_path),
        ],
        check=True,
    )
    return True


def prepare_output_dir(output_root: Path, run_name: str | None) -> Path:
    """Create and return the run-specific output directory."""

    if run_name is None:
        run_name = "zhenhuan_e11_down_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    """Run the complete podcast generation pipeline."""

    configure_logging()
    args = parse_args()
    output_dir = prepare_output_dir(args.output_root, args.run_name)
    ref_dir = output_dir / "refs"

    analysis = load_episode_analysis(args.ai_dubbing_run)
    dialogue = analysis["dialogue"]
    refs = build_voice_references(dialogue, args.ai_dubbing_run, ref_dir)
    lines = select_podcast_lines(args.max_lines)

    write_json(output_dir / "voice_references.json", {key: asdict(ref) for key, ref in refs.items()})
    write_json(output_dir / "podcast_lines.json", [asdict(line) for line in lines])
    write_script_markdown(output_dir / "podcast_script.md", lines, refs)

    if args.skip_tts:
        logging.info("Skip TTS enabled. Wrote script and references to %s", output_dir)
        return

    device = args.device or get_best_device()
    model = load_model(args.model, device)
    timing_rows = generate_segments(model, lines, refs, output_dir, args)
    write_timing_tsv(output_dir / "timing.tsv", timing_rows)

    wav_path = output_dir / "zhenhuan_e11_down_podcast.wav"
    duration = concatenate_segments(lines, output_dir / "segments", wav_path)
    logging.info("Wrote podcast WAV: %s (%.2fs)", wav_path, duration)

    if args.export_mp3:
        mp3_path = output_dir / "zhenhuan_e11_down_podcast.mp3"
        if export_mp3_if_available(wav_path, mp3_path):
            logging.info("Wrote podcast MP3: %s", mp3_path)


if __name__ == "__main__":
    main()
