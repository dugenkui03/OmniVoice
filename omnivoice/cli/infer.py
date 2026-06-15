"""Single-item inference CLI for OmniVoice.

Generates audio from a single text input using voice cloning,
voice design, or auto voice.

Usage:
    # Voice cloning
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." \
        --ref_audio ref.wav --ref_text "Reference transcript." --output out.wav

    # Voice design
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." \
        --instruct "male, British accent" --output out.wav

    # Auto voice
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." --output out.wav
"""

import argparse
import logging

import torch

import soundfile as sf

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device, str2bool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice single-item inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output WAV file path.",
    )
    # Voice cloning
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Reference audio file path for voice cloning.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Reference text describing the reference audio.",
    )
    # Voice design
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for voice design mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name (e.g. 'English') or code (e.g. 'en').",
    )
    # Generation parameters
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Fixed output duration in seconds. If set, overrides the "
        "model's duration estimation. The speed factor is automatically "
        "adjusted to match while preserving language-aware pacing.",
    )
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument(
        "--postprocess_output",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Auto-detected if not specified.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    # 没指定 --device 时按 CUDA > MPS > CPU 自动挑
    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    # 加载 OmniVoice 主模型, 同时会把 text_tokenizer / audio_tokenizer /
    # duration_estimator 一起准备好 (见 OmniVoice.from_pretrained).
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    logging.info(f"Generating audio for: {args.text[:80]}...")
    # 单条生成: 三种模式由参数自动判断
    #   - 给了 ref_audio  → Voice Cloning
    #   - 给了 instruct   → Voice Design
    #   - 都没给          → Auto Voice
    audios = model.generate(
        # 要合成的目标文本; 示例: "Hello, welcome to OmniVoice."
        text=args.text,
        # 文本语言, 有助于发音更稳; 示例: "English" / "en" / "Chinese".
        language=args.language,
        # 参考音频路径, 用于声音克隆; 示例: "ref.wav".
        ref_audio=args.ref_audio,
        # 参考音频对应文本; 示例: "This is the reference voice."
        ref_text=args.ref_text,
        # 音色/风格描述, 用于 voice design; 示例: "male, British accent".
        instruct=args.instruct,
        # 固定输出时长(秒), 会覆盖自动时长估计; 示例: 5.0.
        duration=args.duration,
        # 迭代生成步数, 越大通常越慢但更细; 示例: 32.
        num_step=args.num_step,
        # 条件引导强度, 越大越贴近条件; 示例: 2.0.
        guidance_scale=args.guidance_scale,
        # 语速倍率, >1 更快, <1 更慢; 示例: 1.2.
        speed=args.speed,
        # 时间步偏移, 调整生成过程的时间分布; 示例: 0.1.
        t_shift=args.t_shift,
        # 是否加入 denoise token 降噪; 示例: True.
        denoise=args.denoise,
        # 是否做静音裁剪、淡入淡出等后处理; 示例: True.
        postprocess_output=args.postprocess_output,
        # codebook 层解码惩罚系数; 示例: 5.0.
        layer_penalty_factor=args.layer_penalty_factor,
        # 位置选择温度, 越高随机性越强; 示例: 5.0.
        position_temperature=args.position_temperature,
        # token 采样温度, 0 表示贪心选择; 示例: 0.0.
        class_temperature=args.class_temperature,
    )

    # 输出已经是 24kHz 的 1-D ndarray, 直接落盘即可
    sf.write(args.output, audios[0], model.sampling_rate)
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
