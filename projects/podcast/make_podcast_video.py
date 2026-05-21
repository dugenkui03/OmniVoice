#!/usr/bin/env python3
"""Compose the Zhen Huan podcast audio into a vertical short video.

The script treats the generated podcast audio as the master timeline, then
uses the original E11 clip plus episode analysis metadata as visual material.
It produces a machine-readable visual plan, ASS overlays, intermediate video
segments, and a final 1080x1920 MP4.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageSequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PODCAST_DIR = REPO_ROOT / "output" / "podcast" / "zhenhuan_e11_down_emotion_v3_20260521_0113"
DEFAULT_SOURCE_VIDEO = Path(
    "/Users/dugenkui/workspace/ai_dubbing/test_data/测试输入/电视剧-四级单词/"
    "甄嬛传_E11_下集_教科书级向上管理_曹贵人巧言化解华妃雷霆之怒.mp4"
)
DEFAULT_ANALYSIS_JSON = Path(
    "/Users/dugenkui/workspace/ai_dubbing/test_data/测试输出/"
    "batch_english_learning_from_E06_queue_20260510_213936/runs/"
    "甄嬛传_E11_下集_教科书级向上管理_曹贵人巧言化解华妃雷霆之怒/"
    "episode_analyses/ep11_analysis.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "podcast_video"
DEFAULT_EFFECTS_PLAN = Path(__file__).with_name("effects_plan.json")
DEFAULT_EMOJI_ASSET_DIR = Path(__file__).with_name("assets") / "animated_emojis"
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_FPS = 25
MIN_VISUAL_SEGMENT_SEC = 6.0
MAX_VISUAL_SEGMENT_SEC = 12.0
VIDEO_TOP_MARGIN = 96
LAYOUT_VERTICAL_SHIFT = VIDEO_TOP_MARGIN
CJK_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
)


@dataclass(frozen=True)
class PodcastLineTiming:
    """Represent one podcast utterance on the final audio timeline."""

    line_id: str
    part: str
    speaker: str
    emotion: str
    reference_key: str
    text: str
    clean_text: str
    start_sec: float
    end_sec: float
    pause_after_sec: float

    @property
    def duration_sec(self) -> float:
        """Return the spoken duration without the scripted pause."""

        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class VisualSegment:
    """Describe one visual block used to render a piece of the base video."""

    segment_id: str
    visual_type: str
    line_ids: list[str]
    podcast_start_sec: float
    podcast_end_sec: float
    source_start_sec: float
    source_end_sec: float
    anchor_dialogue_index: int | None
    vocab_word: str | None
    gag_text: str
    overlay_hint: str

    @property
    def duration_sec(self) -> float:
        """Return how long this visual block lasts in the podcast timeline."""

        return max(0.0, self.podcast_end_sec - self.podcast_start_sec)


@dataclass(frozen=True)
class EffectEvent:
    """Store one resolved sticker/effect event on the podcast timeline."""

    effect_id: str
    effect_type: str
    text: str
    body: str
    emojis: list[str]
    icon: str
    style: str
    position: str
    start_sec: float
    end_sec: float
    priority: int


@dataclass(frozen=True)
class VocabDisplayWindow:
    """Store the time range where one vocabulary card should stay visible."""

    segment_id: str
    vocab_word: str
    start_sec: float
    end_sec: float


def parse_args() -> argparse.Namespace:
    """Parse CLI options for preview and full video generation."""

    parser = argparse.ArgumentParser(
        description="Make a vertical short video from the Zhen Huan podcast output."
    )
    parser.add_argument("--podcast-dir", type=Path, default=DEFAULT_PODCAST_DIR)
    parser.add_argument("--source-video", type=Path, default=DEFAULT_SOURCE_VIDEO)
    parser.add_argument("--analysis-json", type=Path, default=DEFAULT_ANALYSIS_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None, help="Optional output folder name.")
    parser.add_argument("--max-duration", type=float, default=None, help="Render only a prefix, in seconds.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--crf", type=int, default=21, help="Final H.264 quality; lower is larger/better.")
    parser.add_argument(
        "--effects-plan",
        type=Path,
        default=DEFAULT_EFFECTS_PLAN,
        help="JSON file that describes sticker/callout effects for the overlay layer.",
    )
    parser.add_argument("--keep-segments", action="store_true", help="Keep rendered segment MP4 files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in an existing run folder.")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise command-line logging for long FFmpeg runs."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file and return the parsed value."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON with stable formatting for hand inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_tool(name: str) -> str:
    """Return an executable path for an external tool or raise a helpful error."""

    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"Cannot find required tool on PATH: {name}")
    return resolved


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess and include the tail of stderr when it fails."""

    logging.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("Command failed: %s", " ".join(cmd))
        logging.error("stderr tail:\n%s", result.stderr[-3000:])
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)


def ffprobe_duration_sec(path: Path, ffprobe: str) -> float:
    """Return media duration in seconds using ffprobe."""

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return max(0.0, float(result.stdout.strip()))


def find_audio_path(podcast_dir: Path) -> Path:
    """Find the generated podcast audio, preferring WAV over MP3."""

    preferred_names = [
        "zhenhuan_e11_down_podcast.wav",
        "zhenhuan_e11_down_podcast.mp3",
    ]
    for name in preferred_names:
        candidate = podcast_dir / name
        if candidate.exists():
            return candidate

    candidates = sorted(
        p for p in podcast_dir.glob("*podcast.*") if p.suffix.lower() in {".wav", ".mp3", ".m4a"}
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No podcast audio found in {podcast_dir}")


def strip_voice_tags(text: str) -> str:
    """Remove OmniVoice emotion tags so captions read naturally on screen."""

    text = re.sub(r"\[[^\]]+\]\s*", "", text)
    text = text.replace("...", "……")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_timing_rows(timing_path: Path) -> dict[str, dict[str, str]]:
    """Read timing.tsv into a line-id keyed dictionary."""

    with timing_path.open("r", encoding="utf-8", newline="") as f:
        return {row["line_id"]: row for row in csv.DictReader(f, delimiter="\t")}


def build_podcast_timeline(podcast_dir: Path, max_duration: float | None) -> list[PodcastLineTiming]:
    """Combine podcast_lines.json and timing.tsv into absolute line timings."""

    podcast_lines = read_json(podcast_dir / "podcast_lines.json")
    timing_rows = read_timing_rows(podcast_dir / "timing.tsv")
    timeline: list[PodcastLineTiming] = []
    cursor = 0.0

    for line in podcast_lines:
        line_id = str(line["line_id"])
        timing = timing_rows.get(line_id, {})
        spoken_duration = float(timing.get("audio_duration_sec") or line.get("duration_sec") or 0.0)
        pause_after = float(line.get("pause_after_sec") or 0.0)
        start = cursor
        end = start + spoken_duration
        cursor = end + pause_after

        if max_duration is not None and start >= max_duration:
            break

        clipped_end = min(end, max_duration) if max_duration is not None else end
        if clipped_end <= start:
            continue

        timeline.append(
            PodcastLineTiming(
                line_id=line_id,
                part=str(line.get("part", "")),
                speaker=str(line.get("speaker", "")),
                emotion=str(line.get("emotion", "")),
                reference_key=str(line.get("reference_key", "")),
                text=str(line.get("text", "")),
                clean_text=strip_voice_tags(str(line.get("text", ""))),
                start_sec=round(start, 3),
                end_sec=round(clipped_end, 3),
                pause_after_sec=pause_after,
            )
        )

        if max_duration is not None and end >= max_duration:
            break

    if not timeline:
        raise ValueError(f"No podcast lines selected from {podcast_dir}")
    return timeline


def group_lines_for_visuals(timeline: list[PodcastLineTiming]) -> list[list[PodcastLineTiming]]:
    """Group line timings into 6-12 second visual beats."""

    groups: list[list[PodcastLineTiming]] = []
    current: list[PodcastLineTiming] = []

    for line in timeline:
        if not current:
            current = [line]
            continue

        current_duration = current[-1].end_sec - current[0].start_sec
        candidate_duration = line.end_sec - current[0].start_sec
        part_changed = line.part != current[-1].part
        if part_changed:
            groups.append(current)
            current = [line]
            continue

        should_close = (
            current_duration >= MIN_VISUAL_SEGMENT_SEC
            and candidate_duration > MAX_VISUAL_SEGMENT_SEC
        )

        if should_close:
            groups.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        groups.append(current)
    return groups


def pick_visual_type(lines: list[PodcastLineTiming]) -> str:
    """Choose the visual mode for a grouped podcast beat."""

    parts = {line.part for line in lines}
    if "开场" in parts:
        return "opening_scene"
    if "生词" in parts:
        return "vocab_card"
    if "结尾" in parts:
        return "closing_scene"
    return "story_scene"


def pick_vocab_card(
    lines: list[PodcastLineTiming],
    vocab_cards: list[dict[str, Any]],
    used_vocab_words: set[str],
) -> dict[str, Any] | None:
    """Find the vocabulary card most relevant to the current visual group."""

    combined = " ".join(line.clean_text.lower() for line in lines)
    for card in vocab_cards:
        word = str(card.get("word", "")).lower()
        if word and word not in used_vocab_words and re.search(rf"\b{re.escape(word)}\b", combined):
            return card
    return None


def pick_gag_text(lines: list[PodcastLineTiming], vocab_card: dict[str, Any] | None) -> str:
    """Generate a compact on-screen punchline from the dialogue content."""

    combined = " ".join(line.clean_text for line in lines)
    if "管理员权限" in combined or "群聊" in combined:
        return "后宫权限管理事故"
    if "请罪" in combined and ("方案" in combined or "汇报" in combined):
        return "表面请罪，实际汇报"
    if "恭喜" in combined or "喜从何来" in combined:
        return "高压锅里递喜报"
    if "实锤" in combined or "证据" in combined:
        return "曹贵人：先看证据链"
    if "温宜" in combined:
        return "温宜一出场，局势就软了"
    if vocab_card:
        return f"单词小灶：{vocab_card.get('word', '')}"
    if any(line.part == "结尾" for line in lines):
        return "本期后宫复盘收工"
    return "甄嬛播客局：边聊边补刀"


def dialogue_index_for_podcast_line(line_id: str, dialogue_count: int) -> int:
    """Map a podcast line id onto the closest original episode dialogue index."""

    if dialogue_count <= 1:
        return 0
    try:
        numeric_id = int(line_id)
    except ValueError:
        numeric_id = 1
    return min(dialogue_count - 1, max(0, numeric_id - 1))


def clamp_source_start(anchor_sec: float, duration_sec: float, video_duration_sec: float) -> float:
    """Pick a source-video start time that can cover the requested duration."""

    latest_start = max(0.0, video_duration_sec - duration_sec - 0.2)
    return min(max(0.0, anchor_sec - 1.0), latest_start)


def build_visual_plan(
    timeline: list[PodcastLineTiming],
    analysis: dict[str, Any],
    video_duration_sec: float,
) -> tuple[list[VisualSegment], list[dict[str, Any]]]:
    """Create visual segments and serializable caption events from project metadata."""

    dialogue = list(analysis.get("dialogue") or [])
    vocab_cards = list(analysis.get("vocab_cards") or [])
    groups = group_lines_for_visuals(timeline)
    visual_segments: list[VisualSegment] = []
    used_vocab_words: set[str] = set()

    for index, group in enumerate(groups):
        visual_type = pick_visual_type(group)
        vocab_card = pick_vocab_card(group, vocab_cards, used_vocab_words) if visual_type == "vocab_card" else None
        if vocab_card:
            used_vocab_words.add(str(vocab_card.get("word", "")).lower())
        elif visual_type == "vocab_card":
            visual_type = "vocab_intro"

        if vocab_card and isinstance(vocab_card.get("dialogue_index"), int):
            anchor_index = min(max(0, int(vocab_card["dialogue_index"])), max(0, len(dialogue) - 1))
        else:
            anchor_index = dialogue_index_for_podcast_line(group[0].line_id, len(dialogue))

        anchor_dialogue = dialogue[anchor_index] if dialogue else {}
        anchor_sec = float(anchor_dialogue.get("start_ms", 0)) / 1000.0
        duration_sec = max(0.2, group[-1].end_sec - group[0].start_sec)
        source_start = clamp_source_start(anchor_sec, duration_sec, video_duration_sec)
        gag_text = pick_gag_text(group, vocab_card)

        visual_segments.append(
            VisualSegment(
                segment_id=f"v{index + 1:04d}",
                visual_type=visual_type,
                line_ids=[line.line_id for line in group],
                podcast_start_sec=round(group[0].start_sec, 3),
                podcast_end_sec=round(group[-1].end_sec, 3),
                source_start_sec=round(source_start, 3),
                source_end_sec=round(source_start + duration_sec, 3),
                anchor_dialogue_index=anchor_index if dialogue else None,
                vocab_word=str(vocab_card.get("word")) if vocab_card else None,
                gag_text=gag_text,
                overlay_hint=build_overlay_hint(group, vocab_card, gag_text),
            )
        )

    caption_events = [asdict(line) for line in timeline]
    return visual_segments, caption_events


def build_vocab_display_windows(visual_segments: list[VisualSegment]) -> list[VocabDisplayWindow]:
    """Extend each word card until the next word card or the end of the vocab block."""

    windows: list[VocabDisplayWindow] = []
    vocab_types = {"vocab_card", "vocab_intro"}

    for index, segment in enumerate(visual_segments):
        if segment.visual_type != "vocab_card" or not segment.vocab_word:
            continue

        end_sec = segment.podcast_end_sec
        for next_segment in visual_segments[index + 1 :]:
            if next_segment.visual_type == "vocab_card":
                end_sec = next_segment.podcast_start_sec
                break
            if next_segment.visual_type not in vocab_types:
                end_sec = next_segment.podcast_start_sec
                break
            end_sec = next_segment.podcast_end_sec

        windows.append(
            VocabDisplayWindow(
                segment_id=segment.segment_id,
                vocab_word=segment.vocab_word,
                start_sec=segment.podcast_start_sec,
                end_sec=round(max(segment.podcast_start_sec + 0.4, end_sec), 3),
            )
        )

    return windows


def load_effects_plan(path: Path | None) -> dict[str, Any]:
    """Load the optional sticker/effect plan used by the overlay renderer."""

    if path is None or not path.exists():
        return {"version": 1, "effects": []}
    plan = read_json(path)
    if not isinstance(plan, dict):
        raise ValueError(f"Effects plan must be a JSON object: {path}")
    effects = plan.get("effects", [])
    if not isinstance(effects, list):
        raise ValueError(f"Effects plan field 'effects' must be a list: {path}")
    return plan


def resolve_effect_events(
    effects_plan: dict[str, Any],
    timeline: list[PodcastLineTiming],
    visual_segments: list[VisualSegment],
) -> list[EffectEvent]:
    """Resolve declarative effect rules into concrete timeline events."""

    line_by_id = {line.line_id: line for line in timeline}
    events: list[EffectEvent] = []

    for raw_effect in effects_plan.get("effects", []):
        if not isinstance(raw_effect, dict):
            continue
        base = normalize_effect_rule(raw_effect)
        line_ids = [str(line_id) for line_id in raw_effect.get("line_ids", [])]
        visual_types = [str(visual_type) for visual_type in raw_effect.get("visual_types", [])]

        if line_ids:
            matched_lines = [line_by_id[line_id] for line_id in line_ids if line_id in line_by_id]
            if matched_lines:
                events.append(event_from_lines(base, matched_lines))

        for segment in visual_segments:
            if visual_types and segment.visual_type in visual_types:
                events.append(event_from_segment(base, segment))

    return sorted(events, key=lambda event: (event.start_sec, event.priority, event.effect_id))


def normalize_effect_rule(raw_effect: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults for one effect rule before timeline resolution."""

    return {
        "effect_id": str(raw_effect.get("id", "effect")),
        "effect_type": str(raw_effect.get("effect_type", "sticker_label")),
        "text": str(raw_effect.get("text", "")),
        "body": str(raw_effect.get("body", "")),
        "emojis": [str(item) for item in raw_effect.get("emojis", [])],
        "icon": str(raw_effect.get("icon", "spark")),
        "style": str(raw_effect.get("style", "gold")),
        "position": str(raw_effect.get("position", "video_top_right")),
        "duration_sec": float(raw_effect.get("duration_sec", 3.8)),
        "offset_sec": float(raw_effect.get("offset_sec", 0.0)),
        "priority": int(raw_effect.get("priority", 50)),
    }


def event_from_lines(base: dict[str, Any], lines: list[PodcastLineTiming]) -> EffectEvent:
    """Create an effect event from one or more podcast line timings."""

    start_sec = max(0.0, min(line.start_sec for line in lines) + base["offset_sec"])
    natural_end = max(line.end_sec for line in lines)
    end_sec = min(natural_end, start_sec + base["duration_sec"])
    return EffectEvent(
        effect_id=base["effect_id"],
        effect_type=base["effect_type"],
        text=base["text"],
        body=base["body"],
        emojis=base["emojis"],
        icon=base["icon"],
        style=base["style"],
        position=base["position"],
        start_sec=round(start_sec, 3),
        end_sec=round(max(start_sec + 0.4, end_sec), 3),
        priority=base["priority"],
    )


def event_from_segment(base: dict[str, Any], segment: VisualSegment) -> EffectEvent:
    """Create an effect event that follows a visual segment."""

    start_sec = max(0.0, segment.podcast_start_sec + base["offset_sec"])
    end_sec = min(segment.podcast_end_sec, start_sec + base["duration_sec"])
    return EffectEvent(
        effect_id=f"{base['effect_id']}_{segment.segment_id}",
        effect_type=base["effect_type"],
        text=base["text"],
        body=base["body"],
        emojis=base["emojis"],
        icon=base["icon"],
        style=base["style"],
        position=base["position"],
        start_sec=round(start_sec, 3),
        end_sec=round(max(start_sec + 0.4, end_sec), 3),
        priority=base["priority"],
    )


def build_overlay_hint(lines: list[PodcastLineTiming], vocab_card: dict[str, Any] | None, gag_text: str) -> str:
    """Summarize why a visual block exists for humans reading visual_plan.json."""

    speakers = "、".join(dict.fromkeys(line.speaker for line in lines))
    if vocab_card:
        return f"{speakers} 讲到 {vocab_card.get('word')}，画面切词卡：{gag_text}"
    return f"{speakers} 对话推进剧情，屏幕梗点：{gag_text}"


def prepare_output_dir(output_root: Path, podcast_dir: Path, run_name: str | None, overwrite: bool) -> Path:
    """Create the output directory for this render run."""

    if run_name is None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{podcast_dir.name}_video_{suffix}"
    output_dir = output_root / run_name
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already has files; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ass_time(sec: float) -> str:
    """Convert seconds to ASS h:mm:ss.cc time format."""

    sec = max(0.0, sec)
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    centiseconds = int(round((sec - math.floor(sec)) * 100))
    if centiseconds >= 100:
        seconds += 1
        centiseconds -= 100
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def ass_escape(text: str) -> str:
    """Escape user-facing text for ASS dialogue fields."""

    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", " ")
    text = text.replace("\n", r"\N")
    return text


def wrap_ass_text(text: str, max_chars: int) -> str:
    """Insert ASS line breaks so Chinese captions fit the vertical layout."""

    text = text.strip()
    if len(text) <= max_chars:
        return text
    chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    return r"\N".join(chunks)


def style_for_speaker(speaker: str) -> str:
    """Map a speaker name to a dedicated caption style."""

    if speaker == "华妃":
        return "CaptionHua"
    if speaker == "曹贵人":
        return "CaptionCao"
    return "CaptionZhou"


def write_ass_file(
    ass_path: Path,
    timeline: list[PodcastLineTiming],
    visual_segments: list[VisualSegment],
    vocab_display_windows: list[VocabDisplayWindow],
    analysis: dict[str, Any],
    duration_sec: float,
) -> None:
    """Write ASS overlays for title, speaker captions, gags, and vocab cards."""

    vocab_by_word = {str(card.get("word")): card for card in analysis.get("vocab_cards", [])}
    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        "Style: Title,Hiragino Sans GB,46,&H00FFFFFF,&H000000FF,&H00000000,&HA8000000,1,0,0,0,100,100,0,0,3,2,0,8,70,70,58,1",
        "Style: Gag,Hiragino Sans GB,42,&H001FE6FF,&H000000FF,&H00000000,&H99000000,1,0,0,0,100,100,0,0,3,2,0,8,80,80,178,1",
        "Style: CaptionZhou,Hiragino Sans GB,42,&H00FFFFFF,&H000000FF,&H00202020,&HD0000000,0,0,0,0,100,100,0,0,3,2,0,2,82,82,138,1",
        "Style: CaptionHua,Hiragino Sans GB,42,&H00D4C2FF,&H000000FF,&H00202020,&HD0000000,0,0,0,0,100,100,0,0,3,2,0,2,82,82,138,1",
        "Style: CaptionCao,Hiragino Sans GB,42,&H00DDF5D2,&H000000FF,&H00202020,&HD0000000,0,0,0,0,100,100,0,0,3,2,0,2,82,82,138,1",
        "Style: Vocab,Hiragino Sans GB,38,&H00FFFFFF,&H000000FF,&H001A1A1A,&HD8222218,0,0,0,0,100,100,0,0,3,2,0,5,86,86,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    lines.append(
        f"Dialogue: 3,{ass_time(0)},{ass_time(duration_sec)},Title,,0,0,0,,"
        f"{ass_escape('甄嬛播客局  EP11 下  ·  剧情复盘 + 单词小灶')}"
    )

    for segment in visual_segments:
        gag_start = segment.podcast_start_sec + 0.25
        gag_end = min(segment.podcast_end_sec, gag_start + 4.0)
        lines.append(
            f"Dialogue: 3,{ass_time(gag_start)},{ass_time(gag_end)},Gag,,0,0,0,,"
            f"{ass_escape(segment.gag_text)}"
        )

    for window in vocab_display_windows:
        card = vocab_by_word.get(window.vocab_word)
        if card:
            vocab_text = build_vocab_overlay_text(card)
            lines.append(
                f"Dialogue: 4,{ass_time(window.start_sec + 0.6)},"
                f"{ass_time(window.end_sec - 0.4)},Vocab,,0,0,0,,"
                f"{{\\pos(540,1110)}}{ass_escape(vocab_text)}"
            )

    for line in timeline:
        speaker_label = f"{line.speaker}："
        text = wrap_ass_text(speaker_label + line.clean_text, 22)
        lines.append(
            f"Dialogue: 5,{ass_time(line.start_sec)},{ass_time(line.end_sec)},"
            f"{style_for_speaker(line.speaker)},,0,0,0,,{ass_escape(text)}"
        )

    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_vocab_overlay_text(card: dict[str, Any]) -> str:
    """Create a compact vocabulary card for on-screen display."""

    word = str(card.get("word", "")).strip()
    meaning = str(card.get("meaning_zh") or card.get("cet_meaning_zh") or "").strip()
    example = str(card.get("simple_sentence") or card.get("original_sentence") or "").strip()
    explanation = str(card.get("explanation_zh") or "").strip()
    explanation = explanation[:34] + "…" if len(explanation) > 34 else explanation
    return wrap_ass_text(f"单词小灶  {word}\n{meaning}\n例句：{example}\n{explanation}", 26)


def build_segment_filter(width: int, height: int, fps: int) -> str:
    """Return the FFmpeg filter that creates the vertical podcast layout."""

    fg_x, fg_y, fg_width, fg_height = video_frame_layout(width)
    return (
        f"[0:v]fps={fps},split=2[raw_bg][raw_fg];"
        f"[raw_bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=26,eq=brightness=-0.08:saturation=0.95[bg];"
        f"[raw_fg]scale={fg_width}:{fg_height}:force_original_aspect_ratio=decrease,"
        f"pad={fg_width}:{fg_height}:(ow-iw)/2:(oh-ih)/2:color=black[fg];"
        f"[bg]drawbox=x=0:y=0:w={width}:h={height}:color=black@0.18:t=fill[dim];"
        f"[dim][fg]overlay={fg_x}:{fg_y},"
        f"drawbox=x={fg_x}:y={fg_y}:w={fg_width}:h={fg_height}:color=white@0.18:t=2,"
        f"format=yuv420p"
    )


def video_frame_layout(width: int) -> tuple[int, int, int, int]:
    """Return the safe-positioned source video rectangle for vertical output."""

    frame_width = int(round((width * 0.87) / 2) * 2)
    frame_height = video_frame_height_for_width(frame_width)
    frame_x = (width - frame_width) // 2
    frame_y = VIDEO_TOP_MARGIN + LAYOUT_VERTICAL_SHIFT
    return frame_x, frame_y, frame_width, frame_height


def video_frame_height_for_width(frame_width: int) -> int:
    """Return an even 16:9 height that is never smaller than scaled input."""

    return int(math.ceil((frame_width * 9 / 16) / 2) * 2)


def render_visual_segment(
    segment: VisualSegment,
    source_video: Path,
    output_path: Path,
    ffmpeg: str,
    width: int,
    height: int,
    fps: int,
    crf: int,
) -> None:
    """Render one vertical base-video segment from the original drama clip."""

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{segment.source_start_sec:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{segment.duration_sec:.3f}",
        "-vf",
        build_segment_filter(width, height, fps),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-r",
        str(fps),
        str(output_path),
    ]
    run_command(cmd)


def write_concat_file(path: Path, segment_paths: list[Path]) -> None:
    """Write an FFmpeg concat-demuxer file with absolute segment paths."""

    rows = [f"file '{str(segment_path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for segment_path in segment_paths]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def concat_segments(concat_file: Path, output_path: Path, ffmpeg: str) -> None:
    """Concatenate uniformly encoded segment MP4 files without re-encoding."""

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]
    run_command(cmd)


def compose_final_video(
    base_video: Path,
    overlay_video: Path,
    audio_path: Path,
    output_path: Path,
    duration_sec: float,
    ffmpeg: str,
    crf: int,
) -> None:
    """Overlay the transparent text layer and attach the podcast audio."""

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(base_video),
        "-i",
        str(overlay_video),
        "-i",
        str(audio_path),
        "-t",
        f"{duration_sec:.3f}",
        "-filter_complex",
        "[0:v][1:v]overlay=0:0:format=auto[v]",
        "-map",
        "[v]",
        "-map",
        "2:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    run_command(cmd)


def find_cjk_font_path() -> Path:
    """Return a Chinese-capable system font for Pillow overlay rendering."""

    for path in CJK_FONT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Cannot find a Chinese-capable system font for overlay rendering.")


def load_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType/OpenType font at the requested size."""

    return ImageFont.truetype(str(font_path), size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Measure rendered text width and height with Pillow."""

    if not text:
        return (0, 0)
    box = draw.textbbox((0, 0), text, font=font)
    return (box[2] - box[0], box[3] - box[1])


def wrap_text_by_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    """Wrap mixed Chinese/English text while keeping English words intact."""

    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    lines: list[str] = []
    current = ""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'._-]*|\s+|.", text)
    for token in tokens:
        if not current and token.isspace():
            continue
        joiner = "" if token.isspace() or not current or current.endswith(" ") else ""
        candidate = current + joiner + token
        if current and text_size(draw, candidate, font)[0] > max_width:
            lines.append(current)
            current = token.lstrip()
            if len(lines) >= max_lines:
                break
        else:
            current = candidate

    if current and len(lines) < max_lines:
        lines.append(current)

    if len(lines) == max_lines and len("".join(lines)) < len(text):
        lines[-1] = lines[-1].rstrip("，。,. ") + "…"
    return lines


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
) -> None:
    """Draw one line centered inside a rectangular box."""

    width, height = text_size(draw, text, font)
    x1, y1, x2, y2 = box
    draw.text((x1 + (x2 - x1 - width) / 2, y1 + (y2 - y1 - height) / 2 - 4), text, font=font, fill=fill)


def active_line_at(t: float, timeline: list[PodcastLineTiming]) -> PodcastLineTiming | None:
    """Return the spoken podcast line active at timestamp t."""

    for line in timeline:
        if line.start_sec <= t < line.end_sec:
            return line
    return None


def active_segment_at(t: float, visual_segments: list[VisualSegment]) -> VisualSegment | None:
    """Return the visual segment active at timestamp t."""

    for segment in visual_segments:
        if segment.podcast_start_sec <= t < segment.podcast_end_sec:
            return segment
    return visual_segments[-1] if visual_segments else None


def active_vocab_word_at(t: float, vocab_display_windows: list[VocabDisplayWindow]) -> str | None:
    """Return the vocabulary word whose card should be visible at timestamp t."""

    for window in vocab_display_windows:
        if window.start_sec <= t < window.end_sec:
            return window.vocab_word
    return None


def active_effect_events(t: float, effect_events: list[EffectEvent]) -> list[EffectEvent]:
    """Return all sticker/effect events active at timestamp t."""

    return [event for event in effect_events if event.start_sec <= t < event.end_sec]


def speaker_color(speaker: str) -> tuple[int, int, int, int]:
    """Return the accent color used for the active speaker."""

    if speaker == "华妃":
        return (255, 128, 188, 255)
    if speaker == "曹贵人":
        return (255, 225, 124, 255)
    return (225, 226, 238, 255)


def speaker_fill_color(speaker: str) -> tuple[int, int, int, int]:
    """Return a soft active fill color for a speaker badge."""

    if speaker == "华妃":
        return (255, 240, 248, 238)
    if speaker == "曹贵人":
        return (255, 248, 221, 230)
    return (246, 246, 250, 230)


def effect_palette(style: str) -> dict[str, tuple[int, int, int, int]]:
    """Return color tokens for a sticker/effect style."""

    palettes = {
        "gold": {
            "fill": (24, 19, 10, 222),
            "outline": (255, 218, 92, 230),
            "text": (255, 232, 126, 255),
            "icon": (255, 232, 126, 255),
        },
        "pink": {
            "fill": (30, 8, 22, 222),
            "outline": (255, 122, 190, 230),
            "text": (255, 176, 218, 255),
            "icon": (255, 126, 195, 255),
        },
        "slate": {
            "fill": (12, 14, 20, 218),
            "outline": (215, 218, 232, 170),
            "text": (244, 244, 248, 255),
            "icon": (225, 226, 238, 255),
        },
    }
    return palettes.get(style, palettes["gold"])


def with_alpha(color: tuple[int, int, int, int], alpha_scale: float) -> tuple[int, int, int, int]:
    """Return an RGBA color with its alpha multiplied for frame animation."""

    r, g, b, a = color
    return (r, g, b, max(0, min(255, int(a * alpha_scale))))


def load_animated_emoji_cache(asset_dir: Path = DEFAULT_EMOJI_ASSET_DIR) -> dict[str, list[Image.Image]]:
    """Load local animated emoji GIF frames for use in overlay rendering."""

    cache: dict[str, list[Image.Image]] = {}
    if not asset_dir.exists():
        return cache
    for path in sorted(asset_dir.glob("*.gif")):
        try:
            source = Image.open(path)
            frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(source)]
        except Exception as exc:  # noqa: BLE001 - broken downloaded assets should not stop rendering.
            logging.warning("Skipping animated emoji %s: %s", path, exc)
            continue
        if frames:
            cache[path.stem] = frames
    return cache


def animated_emoji_frame(
    emoji_frames: dict[str, list[Image.Image]],
    emoji_name: str,
    t: float,
    fps: int = 12,
) -> Image.Image | None:
    """Return the frame for an animated emoji at timestamp t."""

    frames = emoji_frames.get(emoji_name)
    if not frames:
        return None
    frame_index = int(max(0.0, t) * fps) % len(frames)
    return frames[frame_index]


def paste_animated_emoji(
    image: Image.Image,
    emoji_frames: dict[str, list[Image.Image]],
    emoji_name: str,
    t: float,
    center: tuple[int, int],
    size: int,
    alpha_scale: float,
) -> bool:
    """Paste one animated emoji frame on the transparent overlay image."""

    frame = animated_emoji_frame(emoji_frames, emoji_name, t)
    if frame is None:
        return False
    resized = frame.resize((size, size), Image.Resampling.LANCZOS)
    if alpha_scale < 0.99:
        alpha = resized.getchannel("A").point(lambda value: int(value * alpha_scale))
        resized.putalpha(alpha)
    x = int(center[0] - size / 2)
    y = int(center[1] - size / 2)
    image.alpha_composite(resized, (x, y))
    return True


def draw_title(draw: ImageDraw.ImageDraw, fonts: dict[str, ImageFont.FreeTypeFont], width: int) -> None:
    """Draw the persistent show title and small format label."""

    draw.rounded_rectangle((54, 38, width - 54, 122), radius=18, fill=(0, 0, 0, 158))
    draw_centered_text(
        draw,
        (70, 46, width - 70, 116),
        "甄嬛播客局  EP11 下  ·  剧情复盘 + 单词小灶",
        fonts["title"],
        (255, 255, 255, 255),
    )


def draw_speaker_badges(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    active_speaker: str | None,
    top_y: int,
) -> None:
    """Draw three podcast-role badges and highlight the active speaker."""

    badges = [("周宁海", 90), ("华妃", 420), ("曹贵人", 750)]
    for speaker, x in badges:
        active = speaker == active_speaker
        fill = speaker_fill_color(speaker) if active else (8, 7, 10, 148)
        outline = speaker_color(speaker) if active else (255, 204, 232, 120)
        text_fill = (40, 40, 52, 255) if active and speaker == "周宁海" else speaker_color(speaker) if active else (245, 245, 248, 220)
        draw.rounded_rectangle((x, top_y, x + 240, top_y + 62), radius=27, fill=fill, outline=outline, width=3)
        draw_centered_text(draw, (x, top_y, x + 240, top_y + 60), speaker, fonts["badge"], text_fill)


def draw_video_frame_overlay(image: Image.Image, draw: ImageDraw.ImageDraw, width: int) -> None:
    """Mask the rectangular source video into a rounded-card visual shape."""

    x, y, frame_width, frame_height = video_frame_layout(width)
    radius = 34
    mask = Image.new("L", (frame_width, frame_height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, frame_width, frame_height), radius=radius, fill=255)

    background_patch = image.crop((x, y, x + frame_width, y + frame_height))
    transparent = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
    corner_cover = Image.composite(transparent, background_patch, mask)
    image.paste(corner_cover, (x, y), corner_cover)

    draw.rounded_rectangle(
        (x, y, x + frame_width, y + frame_height),
        radius=radius,
        outline=(255, 210, 158, 145),
        width=4,
    )


def draw_gag(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    segment: VisualSegment,
    t: float,
    width: int,
    top_y: int,
) -> None:
    """Draw a short-lived punchline sticker for the active visual segment."""

    start = segment.podcast_start_sec + 0.25
    end = min(segment.podcast_end_sec, start + 4.0)
    if not (start <= t < end):
        return
    text = segment.gag_text
    tw, th = text_size(draw, text, fonts["gag"])
    pad_x = 34
    box_w = min(width - 120, tw + pad_x * 2)
    x1 = (width - box_w) // 2
    y1 = top_y
    draw.rounded_rectangle((x1, y1, x1 + box_w, y1 + 72), radius=26, fill=(32, 23, 0, 210))
    draw.text((x1 + pad_x, y1 + (72 - th) / 2 - 4), text, font=fonts["gag"], fill=(255, 231, 72, 255))


def draw_vocab_card(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    vocab_word: str | None,
    vocab_by_word: dict[str, dict[str, Any]],
    width: int,
    top_y: int,
) -> None:
    """Draw a centered vocabulary card when the visual segment is a word explainer."""

    if not vocab_word:
        return
    card = vocab_by_word.get(vocab_word)
    if not card:
        return

    x1, y1, x2, y2 = 70, top_y, width - 70, top_y + 600
    draw.rounded_rectangle((x1 + 8, y1 + 8, x2 + 8, y2 + 8), radius=34, fill=(0, 0, 0, 90))
    draw.rounded_rectangle((x1, y1, x2, y2), radius=34, fill=(13, 12, 11, 232), outline=(255, 218, 92, 235), width=4)
    draw.text((x1 + 42, y1 + 30), "单词小灶", font=fonts["badge"], fill=(255, 222, 104, 255))
    draw.text((x1 + 42, y1 + 98), str(card.get("word", "")), font=fonts["word"], fill=(255, 255, 255, 255))

    meaning = str(card.get("meaning_zh") or card.get("cet_meaning_zh") or "").strip()
    example = str(card.get("simple_sentence") or card.get("original_sentence") or "").strip()
    explanation = str(card.get("explanation_zh") or "").strip()
    original = str(card.get("original_sentence") or "").strip()
    explanation = explanation[:38] + "…" if len(explanation) > 38 else explanation

    draw.text((x1 + 42, y1 + 192), meaning, font=fonts["meaning"], fill=(255, 230, 118, 255))
    row_y = y1 + 260
    rows = [
        ("例句", example),
        ("解释", explanation),
        ("原句", original),
    ]
    for label, raw in rows:
        if not raw:
            continue
        draw_dashed_line(draw, x1 + 42, row_y - 18, x2 - 42, fill=(255, 222, 104, 80))
        draw.text((x1 + 42, row_y), label, font=fonts["label"], fill=(255, 222, 104, 255))
        text_x = x1 + 138
        text_y = row_y - 2
        for wrapped in wrap_text_by_width(draw, raw, fonts["body"], x2 - text_x - 42, 2):
            draw.text((text_x, text_y), wrapped, font=fonts["body"], fill=(245, 245, 245, 245))
            text_y += 42
        row_y += 102


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: int,
    y: int,
    x2: int,
    fill: tuple[int, int, int, int],
) -> None:
    """Draw a subtle dashed separator line inside a card."""

    dash = 12
    gap = 10
    x = x1
    while x < x2:
        draw.line((x, y, min(x + dash, x2), y), fill=fill, width=1)
        x += dash + gap


def draw_icon(
    draw: ImageDraw.ImageDraw,
    icon: str,
    center: tuple[int, int],
    size: int,
    fill: tuple[int, int, int, int],
) -> None:
    """Draw a small vector-like icon without relying on external assets."""

    cx, cy = center
    half = size // 2
    if icon == "book":
        draw.rounded_rectangle((cx - half, cy - half, cx, cy + half), radius=4, outline=fill, width=3)
        draw.rounded_rectangle((cx, cy - half, cx + half, cy + half), radius=4, outline=fill, width=3)
        draw.line((cx, cy - half + 4, cx, cy + half - 4), fill=fill, width=2)
    elif icon == "mic":
        draw.rounded_rectangle((cx - 9, cy - half, cx + 9, cy + 6), radius=9, outline=fill, width=3)
        draw.arc((cx - 20, cy - 4, cx + 20, cy + 28), 0, 180, fill=fill, width=3)
        draw.line((cx, cy + 18, cx, cy + half), fill=fill, width=3)
        draw.line((cx - 14, cy + half, cx + 14, cy + half), fill=fill, width=3)
    elif icon == "shield":
        points = [(cx, cy - half), (cx + half, cy - half + 12), (cx + half - 6, cy + half - 4), (cx, cy + half), (cx - half + 6, cy + half - 4), (cx - half, cy - half + 12)]
        draw.line(points + [points[0]], fill=fill, width=3)
        draw.line((cx - 10, cy, cx - 2, cy + 8, cx + 13, cy - 10), fill=fill, width=3)
    elif icon == "doc":
        draw.rounded_rectangle((cx - half + 4, cy - half, cx + half - 4, cy + half), radius=5, outline=fill, width=3)
        draw.line((cx - 10, cy - 7, cx + 12, cy - 7), fill=fill, width=2)
        draw.line((cx - 10, cy + 5, cx + 12, cy + 5), fill=fill, width=2)
    elif icon == "seat":
        draw.arc((cx - half, cy - half, cx + half, cy + half), 200, 340, fill=fill, width=3)
        draw.line((cx - 16, cy + 8, cx + 16, cy + 8), fill=fill, width=3)
        draw.line((cx - 11, cy + 8, cx - 16, cy + half), fill=fill, width=3)
        draw.line((cx + 11, cy + 8, cx + 16, cy + half), fill=fill, width=3)
    else:
        draw.line((cx, cy - half, cx, cy + half), fill=fill, width=3)
        draw.line((cx - half, cy, cx + half, cy), fill=fill, width=3)
        draw.line((cx - 13, cy - 13, cx + 13, cy + 13), fill=fill, width=2)
        draw.line((cx + 13, cy - 13, cx - 13, cy + 13), fill=fill, width=2)


def effect_position_box(
    event: EffectEvent,
    text_width: int,
    width: int,
    caption_y: int,
    vocab_y: int,
) -> tuple[int, int, int, int]:
    """Return the rounded label box for an effect based on its named position."""

    frame_x, frame_y, frame_width, frame_height = video_frame_layout(width)
    box_width = min(width - 120, max(170, text_width + 92))
    box_height = 58
    if event.position == "video_top_left":
        x1, y1 = frame_x + 22, frame_y + 18
    elif event.position == "video_top_right":
        x1, y1 = frame_x + frame_width - box_width - 22, frame_y + 18
    elif event.position == "video_bottom_right":
        x1, y1 = frame_x + frame_width - box_width - 22, frame_y + frame_height - box_height - 20
    elif event.position == "caption_top_right":
        x1, y1 = width - box_width - 78, caption_y - 68
    elif event.position == "vocab_top_right":
        x1, y1 = width - box_width - 92, vocab_y + 26
    else:
        x1, y1 = width - box_width - 78, caption_y + 188
    return (x1, y1, x1 + box_width, y1 + box_height)


def draw_effect_event(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    emoji_frames: dict[str, list[Image.Image]],
    event: EffectEvent,
    t: float,
    width: int,
    caption_y: int,
    vocab_y: int,
    vocab_word: str | None,
) -> None:
    """Draw either a small sticker label or a large callout card effect."""

    if event.effect_type == "callout_card":
        if vocab_word is None:
            draw_effect_callout_card(image, draw, fonts, emoji_frames, event, t, width, vocab_y)
        return

    palette = effect_palette(event.style)
    text_width, _ = text_size(draw, event.text, fonts["effect"])
    x1, y1, x2, y2 = effect_position_box(event, text_width, width, caption_y, vocab_y)
    draw.rounded_rectangle((x1 + 5, y1 + 5, x2 + 5, y2 + 5), radius=24, fill=(0, 0, 0, 80))
    draw.rounded_rectangle((x1, y1, x2, y2), radius=24, fill=palette["fill"], outline=palette["outline"], width=2)
    draw_icon(draw, event.icon, (x1 + 32, y1 + 29), 30, palette["icon"])
    draw.text((x1 + 62, y1 + 13), event.text, font=fonts["effect"], fill=palette["text"])


def draw_effect_callout_card(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    emoji_frames: dict[str, list[Image.Image]],
    event: EffectEvent,
    t: float,
    width: int,
    top_y: int,
) -> None:
    """Draw a prominent animated card in the vocabulary-card area before words appear."""

    palette = effect_palette(event.style)
    progress = max(0.0, min(1.0, (t - event.start_sec) / 0.45))
    fade_out = max(0.0, min(1.0, (event.end_sec - t) / 0.35))
    alpha_scale = min(progress, fade_out)
    slide_y = int((1.0 - progress) * 38)
    pulse = 1.0 + 0.012 * math.sin(max(0.0, t - event.start_sec) * math.tau * 2.2)

    x1, y1, x2, y2 = 70, top_y + slide_y, width - 70, top_y + 600 + slide_y
    center_x = (x1 + x2) // 2
    inflated = int(8 * pulse)
    draw.rounded_rectangle(
        (x1 + 10, y1 + 12, x2 + 10, y2 + 12),
        radius=34 + inflated,
        fill=(0, 0, 0, int(96 * alpha_scale)),
    )
    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=34 + inflated,
        fill=with_alpha(palette["fill"], alpha_scale),
        outline=with_alpha(palette["outline"], alpha_scale),
        width=5,
    )

    icon_y = y1 + 112
    emoji_names = event.emojis or []
    main_emoji_drawn = False
    if emoji_names:
        size = 154 + int(10 * math.sin(max(0.0, t - event.start_sec) * math.tau * 1.6))
        main_emoji_drawn = paste_animated_emoji(
            image,
            emoji_frames,
            emoji_names[0],
            t - event.start_sec,
            (center_x, icon_y),
            size,
            alpha_scale,
        )
    if len(emoji_names) > 1:
        paste_animated_emoji(
            image,
            emoji_frames,
            emoji_names[1],
            t - event.start_sec + 0.23,
            (center_x + 170, icon_y + 10),
            96,
            alpha_scale * 0.92,
        )
    if not main_emoji_drawn:
        draw_icon(draw, event.icon, (center_x, icon_y), 82, with_alpha(palette["icon"], alpha_scale))
    title_width, _ = text_size(draw, event.text, fonts["effect_title"])
    draw.text(
        (center_x - title_width / 2, y1 + 180),
        event.text,
        font=fonts["effect_title"],
        fill=with_alpha(palette["text"], alpha_scale),
    )

    body = event.body or "这一段是剧情里的重点梗，马上接上台词精听。"
    wrapped = wrap_text_by_width(draw, body, fonts["effect_body"], x2 - x1 - 140, 3)
    body_y = y1 + 292
    for line in wrapped:
        line_width, _ = text_size(draw, line, fonts["effect_body"])
        draw.text(
            (center_x - line_width / 2, body_y),
            line,
            font=fonts["effect_body"],
            fill=(255, 255, 255, int(238 * alpha_scale)),
        )
        body_y += 54

    draw.rounded_rectangle(
        (center_x - 180, y2 - 98, center_x + 180, y2 - 38),
        radius=26,
        fill=(255, 255, 255, int(28 * alpha_scale)),
        outline=with_alpha(palette["outline"], alpha_scale),
        width=2,
    )
    draw_centered_text(
        draw,
        (center_x - 180, y2 - 98, center_x + 180, y2 - 38),
        "马上进入单词小灶",
        fonts["effect"],
        with_alpha(palette["text"], alpha_scale),
    )


def draw_caption(
    draw: ImageDraw.ImageDraw,
    fonts: dict[str, ImageFont.FreeTypeFont],
    line: PodcastLineTiming | None,
    width: int,
    top_y: int,
) -> None:
    """Draw the active podcast subtitle with speaker label."""

    if line is None:
        return
    x1, y1, x2, y2 = 70, top_y, width - 70, top_y + 245
    accent = speaker_color(line.speaker)
    draw.rounded_rectangle((x1 + 8, y1 + 8, x2 + 8, y2 + 8), radius=30, fill=(0, 0, 0, 80))
    draw.rounded_rectangle((x1, y1, x2, y2), radius=30, fill=(0, 0, 0, 210), outline=accent, width=3)
    draw.rounded_rectangle((x1 + 26, y1 + 28, x1 + 34, y2 - 28), radius=4, fill=accent)
    draw.text((x1 + 58, y1 + 26), line.speaker, font=fonts["speaker"], fill=accent)
    pill_x = x1 + 190
    emotion_w = min(170, max(96, text_size(draw, line.emotion, fonts["small"])[0] + 42))
    draw.rounded_rectangle((pill_x, y1 + 34, pill_x + emotion_w, y1 + 76), radius=20, fill=(255, 128, 188, 40), outline=(255, 128, 188, 80), width=1)
    draw_centered_text(draw, (pill_x, y1 + 34, pill_x + emotion_w, y1 + 76), line.emotion, fonts["small"], (230, 210, 220, 230))

    wrapped = wrap_text_by_width(draw, line.clean_text, fonts["caption"], x2 - x1 - 120, 3)
    y = y1 + 96
    for text in wrapped:
        draw.text((x1 + 58, y), text, font=fonts["caption"], fill=(255, 255, 255, 252))
        y += 56


def render_overlay_frame(
    t: float,
    timeline: list[PodcastLineTiming],
    visual_segments: list[VisualSegment],
    vocab_display_windows: list[VocabDisplayWindow],
    effect_events: list[EffectEvent],
    vocab_by_word: dict[str, dict[str, Any]],
    emoji_frames: dict[str, list[Image.Image]],
    fonts: dict[str, ImageFont.FreeTypeFont],
    width: int,
    height: int,
) -> Image.Image:
    """Render one transparent overlay frame at timestamp t."""

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    line = active_line_at(t, timeline)
    segment = active_segment_at(t, visual_segments)
    vocab_word = active_vocab_word_at(t, vocab_display_windows)
    _, video_y, _, video_height = video_frame_layout(width)
    speaker_y = video_y + video_height + 26
    caption_y = speaker_y + 86
    vocab_y = caption_y + 270
    gag_y = video_y + 20

    draw_video_frame_overlay(image, draw, width)
    draw_speaker_badges(draw, fonts, line.speaker if line else None, speaker_y)
    if segment:
        draw_gag(draw, fonts, segment, t, width, gag_y)
    draw_vocab_card(draw, fonts, vocab_word, vocab_by_word, width, vocab_y)
    draw_caption(draw, fonts, line, width, caption_y)
    for event in active_effect_events(t, effect_events):
        draw_effect_event(image, draw, fonts, emoji_frames, event, t, width, caption_y, vocab_y, vocab_word)
    return image


def render_overlay_video(
    output_path: Path,
    timeline: list[PodcastLineTiming],
    visual_segments: list[VisualSegment],
    vocab_display_windows: list[VocabDisplayWindow],
    effect_events: list[EffectEvent],
    analysis: dict[str, Any],
    width: int,
    height: int,
    fps: int,
    duration_sec: float,
    ffmpeg: str,
) -> None:
    """Render a transparent MOV overlay using Pillow and FFmpeg rawvideo input."""

    font_path = find_cjk_font_path()
    fonts = {
        "title": load_font(font_path, 42),
        "badge": load_font(font_path, 34),
        "gag": load_font(font_path, 40),
        "speaker": load_font(font_path, 46),
        "small": load_font(font_path, 28),
        "caption": load_font(font_path, 45),
        "body": load_font(font_path, 32),
        "effect": load_font(font_path, 30),
        "effect_title": load_font(font_path, 58),
        "effect_body": load_font(font_path, 37),
        "label": load_font(font_path, 31),
        "meaning": load_font(font_path, 35),
        "word": load_font(font_path, 78),
    }
    vocab_by_word = {str(card.get("word")): card for card in analysis.get("vocab_cards", [])}
    emoji_frames = load_animated_emoji_cache()
    logging.info("Loaded %s animated emoji assets", len(emoji_frames))

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "qtrle",
        str(output_path),
    ]
    frame_count = int(math.ceil(duration_sec * fps))
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index in range(frame_count):
            frame_time = frame_index / fps
            frame = render_overlay_frame(
                frame_time,
                timeline,
                visual_segments,
                vocab_display_windows,
                effect_events,
                vocab_by_word,
                emoji_frames,
                fonts,
                width,
                height,
            )
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()

    assert process.stderr is not None
    stderr = process.stderr.read()
    process.wait()
    if process.returncode != 0:
        logging.error("Overlay render stderr:\n%s", stderr.decode("utf-8", errors="ignore")[-3000:])
        raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr)


def render_video(
    visual_segments: list[VisualSegment],
    timeline: list[PodcastLineTiming],
    vocab_display_windows: list[VocabDisplayWindow],
    effect_events: list[EffectEvent],
    analysis: dict[str, Any],
    source_video: Path,
    audio_path: Path,
    ass_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Render segment clips, concatenate them, and compose the final MP4."""

    ffmpeg = find_tool("ffmpeg")
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_paths: list[Path] = []

    for index, segment in enumerate(visual_segments, start=1):
        segment_path = segment_dir / f"{segment.segment_id}_{segment.visual_type}.mp4"
        logging.info("Rendering visual segment %s/%s: %s", index, len(visual_segments), segment.segment_id)
        render_visual_segment(
            segment,
            source_video,
            segment_path,
            ffmpeg,
            args.width,
            args.height,
            args.fps,
            args.crf,
        )
        segment_paths.append(segment_path)

    concat_file = output_dir / "concat_segments.txt"
    base_video = output_dir / "base_visuals.mp4"
    write_concat_file(concat_file, segment_paths)
    concat_segments(concat_file, base_video, ffmpeg)

    overlay_video = output_dir / "podcast_overlay.mov"
    logging.info("Rendering transparent text overlay: %s", overlay_video)
    duration_sec = visual_segments[-1].podcast_end_sec
    render_overlay_video(
        overlay_video,
        timeline,
        visual_segments,
        vocab_display_windows,
        effect_events,
        analysis,
        args.width,
        args.height,
        args.fps,
        duration_sec,
        ffmpeg,
    )

    final_path = output_dir / "zhenhuan_e11_down_podcast_video.mp4"
    compose_final_video(base_video, overlay_video, audio_path, final_path, duration_sec, ffmpeg, args.crf)

    if not args.keep_segments:
        shutil.rmtree(segment_dir, ignore_errors=True)
    return final_path


def write_readme(path: Path, final_path: Path, visual_segments: list[VisualSegment], duration_sec: float) -> None:
    """Write a tiny run report so generated folders explain themselves."""

    content = f"""# 甄嬛播客视频化输出

- 成片：`{final_path.name}`
- 时长：{duration_sec:.2f}s
- 视觉段落：{len(visual_segments)}
- 视觉方案：`visual_plan.json`
- 效果事件：`visual_plan.json` 中的 `effect_events`
- 单词展示窗口：`visual_plan.json` 中的 `vocab_display_windows`
- 字幕参考：`podcast_overlays.ass`
- 透明字幕视频：`podcast_overlay.mov`

```mermaid
flowchart LR
  A["播客音频"] --> B["视觉段落"]
  C["原剧视频"] --> B
  D["Pillow UI/词卡/贴纸"] --> E["最终 MP4"]
  F["effects_plan.json"] --> D
  B --> E
```
"""
    path.write_text(content, encoding="utf-8")


def validate_inputs(args: argparse.Namespace) -> None:
    """Fail early when required project artifacts are missing."""

    required_paths = [
        args.podcast_dir / "podcast_lines.json",
        args.podcast_dir / "timing.tsv",
        args.source_video,
        args.analysis_json,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")


def main() -> None:
    """Run the complete podcast-video rendering workflow."""

    configure_logging()
    args = parse_args()
    validate_inputs(args)

    ffprobe = find_tool("ffprobe")
    audio_path = find_audio_path(args.podcast_dir)
    video_duration_sec = ffprobe_duration_sec(args.source_video, ffprobe)
    output_dir = prepare_output_dir(args.output_root, args.podcast_dir, args.run_name, args.overwrite)

    timeline = build_podcast_timeline(args.podcast_dir, args.max_duration)
    analysis = read_json(args.analysis_json)
    visual_segments, caption_events = build_visual_plan(timeline, analysis, video_duration_sec)
    vocab_display_windows = build_vocab_display_windows(visual_segments)
    effects_plan = load_effects_plan(args.effects_plan)
    effect_events = resolve_effect_events(effects_plan, timeline, visual_segments)
    final_duration_sec = visual_segments[-1].podcast_end_sec

    visual_plan = {
        "podcast_dir": str(args.podcast_dir),
        "audio_path": str(audio_path),
        "source_video": str(args.source_video),
        "analysis_json": str(args.analysis_json),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "duration_sec": round(final_duration_sec, 3),
        "effects_plan": str(args.effects_plan) if args.effects_plan else None,
        "visual_segments": [asdict(segment) for segment in visual_segments],
        "vocab_display_windows": [asdict(window) for window in vocab_display_windows],
        "caption_events": caption_events,
        "effect_events": [asdict(event) for event in effect_events],
    }
    write_json(output_dir / "visual_plan.json", visual_plan)

    ass_path = output_dir / "podcast_overlays.ass"
    write_ass_file(ass_path, timeline, visual_segments, vocab_display_windows, analysis, final_duration_sec)

    final_path = render_video(
        visual_segments,
        timeline,
        vocab_display_windows,
        effect_events,
        analysis,
        args.source_video,
        audio_path,
        ass_path,
        output_dir,
        args,
    )
    write_readme(output_dir / "README.md", final_path, visual_segments, final_duration_sec)
    logging.info("Wrote podcast video: %s", final_path)


if __name__ == "__main__":
    main()
