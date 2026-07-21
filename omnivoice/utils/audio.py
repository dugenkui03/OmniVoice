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

"""Audio I/O and processing utilities.

Provides functions for loading, resampling, silence removal,
chunking, cross-fading, and format conversion.

All public functions in this module operate on **numpy float32 arrays**
with shape ``(C, T)`` (channels-first).
"""

import io
import logging

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_nonsilent, split_on_silence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_waveform(audio_path: str):
    """Load audio from a file path, returning (data, sample_rate).

    Tries two backends in order:
    1. soundfile — covers WAV/FLAC/OGG etc., no ffmpeg needed.
    2. librosa — covers MP3/M4A etc. via audioread + ffmpeg.

    Returns:
        (data, sample_rate) where data is a numpy float32 array of
        shape (C, T).

    中文说明:
        从文件路径读取音频, 返回 (波形数据, 采样率)。
        按顺序尝试两个后端:
          1. soundfile — 覆盖 WAV/FLAC/OGG 等, 无需 ffmpeg。
          2. librosa — 经 audioread + ffmpeg 覆盖 MP3/M4A 等。
        返回的 data 是形状 (C, T) 的 numpy float32 数组。

        用到的第三方包:
          - soundfile (sf): 基于 libsndfile 的音频读写库, 不依赖 ffmpeg,
            读无损/未压缩格式快且依赖少, 故作首选。
          - librosa: 音频分析库, 底层经 audioread+ffmpeg 可解码 MP3/M4A 等
            压缩格式; 功能全但更重, 故仅在 soundfile 失败时兜底。
    """
    try:
        # 首选 soundfile: always_2d=True 保证即使单声道也返回二维 (T, C)
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        return data.T, sr  # (T, C) → (C, T) 转置成项目约定的"通道在前"
    except Exception:
        # soundfile cannot handle MP3/M4A etc., fall back to librosa.
        # soundfile 读不了 MP3/M4A 等压缩格式时, 回落到 librosa
        import librosa  # 延迟导入: 仅在确实需要时才加载这个较重的库

        # sr=None 表示保留文件原采样率(不重采样); mono=False 表示保留多声道
        data, sr = librosa.load(audio_path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]  # 单声道 (T,) → (1, T), 补出通道维
        return data, sr


def load_audio(audio_path: str, sampling_rate: int) -> np.ndarray:
    """Load a waveform from file and resample to the target rate.

    Parameters:
        audio_path: path of the audio.
        sampling_rate: target sampling rate.

    Returns:
        Numpy float32 array of shape (1, T).

    中文说明:
        从文件读取音频并规整: 转单声道 + 重采样到目标采样率。
        参数: audio_path 音频路径; sampling_rate 目标采样率。
        返回: 形状 (1, T) 的 numpy float32 数组 (单声道, T 为采样点数)。
    """
    # 先读出原始波形 (C, T) 与原采样率 sr (内部按 soundfile→librosa 兜底)
    data, sr = load_waveform(audio_path)

    # 多声道(C>1) → 沿通道轴(axis=0)求均值混成单声道; keepdims=True 保持二维 (1, T)
    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    # 原采样率与目标不一致 → 重采样 (torchaudio 只吃 torch 张量, 故 numpy→torch 算完再转回)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate
        ).numpy()

    return data


def load_audio_bytes(raw: bytes, sampling_rate: int) -> np.ndarray:
    """Load audio from in-memory bytes and resample.

    Parameters:
        raw: raw audio file bytes (e.g. from WebDataset).
        sampling_rate: target sampling rate.

    Returns:
        Numpy float32 array of shape (1, T).
    """
    buf = io.BytesIO(raw)

    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        data = data.T  # (T, C) → (C, T)
    except Exception:
        import librosa

        buf.seek(0)
        data, sr = librosa.load(buf, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]

    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate
        ).numpy()

    return data


# ---------------------------------------------------------------------------
# Audio processing (all numpy in / numpy out)
# ---------------------------------------------------------------------------


def numpy_to_audiosegment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert a numpy float32 array of shape (C, T) to a pydub AudioSegment."""
    audio_int = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()  # interleave channels
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def audiosegment_to_numpy(aseg: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to a numpy float32 array of shape (C, T)."""
    data = np.array(aseg.get_array_of_samples()).astype(np.float32) / 32768.0
    if aseg.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, aseg.channels).T


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> np.ndarray:
    """Remove middle silences longer than *mid_sil* ms and trim edge silences.

    Parameters:
        audio: numpy array with shape (C, T).
        sampling_rate: sampling rate of the audio.
        mid_sil: middle-silence threshold in ms (0 to skip).
        lead_sil: kept leading silence in ms.
        trail_sil: kept trailing silence in ms.

    Returns:
        Numpy array with shape (C, T').
    """
    wave = numpy_to_audiosegment(audio, sampling_rate)

    if mid_sil > 0:
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    wave = remove_silence_edges(wave, lead_sil, trail_sil, -50)

    return audiosegment_to_numpy(wave)


def remove_silence_edges(
    audio: AudioSegment,
    lead_sil: int = 100,
    trail_sil: int = 300,
    silence_threshold: float = -50,
) -> AudioSegment:
    """Remove edge silences, keeping *lead_sil* / *trail_sil* ms."""
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    audio = audio[start_idx:]

    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def fade_and_pad_audio(
    audio: np.ndarray,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Apply fade-in/out and pad with silence to prevent clicks.

    Args:
        audio: numpy array of shape (C, T).
        pad_duration: silence padding duration per side (seconds).
        fade_duration: fade curve duration (seconds).
        sample_rate: audio sampling rate.

    Returns:
        Processed numpy array of shape (C, T_new).
    """
    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)

    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            fade_in = np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., :k] *= fade_in

            fade_out = np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= fade_out

    if pad_samples > 0:
        silence = np.zeros(
            (processed.shape[0], pad_samples),
            dtype=processed.dtype,
        )
        processed = np.concatenate([silence, processed, silence], axis=-1)

    return processed


def trim_long_audio(
    audio: np.ndarray,
    sampling_rate: int,
    max_duration: float = 15.0,
    min_duration: float = 3.0,
    trim_threshold: float = 20.0,
) -> np.ndarray:
    """Trim audio to <= *max_duration* by splitting at the largest silence gap.

    Only trims when the audio exceeds *trim_threshold* seconds.

    Args:
        audio: numpy array of shape (C, T).
        sampling_rate: audio sampling rate.
        max_duration: maximum duration in seconds.
        min_duration: minimum duration in seconds.
        trim_threshold: only trim if audio is longer than this (seconds).

    Returns:
        Trimmed numpy array.

    中文说明:
        把过长音频在静音间隙处切断, 使时长不超过 max_duration。
        仅当音频超过 trim_threshold 秒时才裁剪, 尽量切在静音处以免截断词语。

        参数:
            audio: 形状 (C, T) 的 numpy 数组。
            sampling_rate: 采样率。
            max_duration: 裁剪后允许的最大时长(秒)。
            min_duration: 裁剪后至少保留的时长(秒)。
            trim_threshold: 只有超过该秒数才裁剪。
        返回:
            裁剪后的 numpy 数组。
    """
    # 总时长(秒) = 采样点数 / 采样率; 未超过阈值则原样返回, 不裁剪
    duration = audio.shape[-1] / sampling_rate
    if duration <= trim_threshold:
        return audio

    # numpy 波形 → pydub AudioSegment, 以便用静音检测工具
    seg = numpy_to_audiosegment(audio, sampling_rate)
    # 检测所有“非静音”区间 [(start_ms, end_ms), ...]
    #   min_silence_len=100: 至少 100ms 静音才算一段间隔
    #   silence_thresh=-40: 低于 -40dB 视为静音
    #   seek_step=10: 每 10ms 扫描一步
    nonsilent = detect_nonsilent(
        seg, min_silence_len=100, silence_thresh=-40, seek_step=10
    )
    # 整段几乎全是静音(检测不到非静音) → 无从切分, 原样返回
    if not nonsilent:
        return audio

    # 秒 → 毫秒, 后续 pydub 的下标单位是毫秒
    max_ms = int(max_duration * 1000)
    min_ms = int(min_duration * 1000)

    # 在 max_ms 之前, 找一个尽量靠后的“非静音段起点”作为切点,
    # 这样切口落在静音间隙里, 不会把一个词从中间截断
    best_split = 0
    for start, end in nonsilent:
        if start > best_split and start <= max_ms:
            best_split = start
        # 已经越过 max_ms, 后面的段无需再看
        if end > max_ms:
            break

    # 找到的切点太靠前(短于 min_ms) → 直接切到 max_ms(或音频末尾), 保证够长
    if best_split < min_ms:
        best_split = min(max_ms, len(seg))

    # 按切点截取前半段, 再转回 numpy 返回
    trimmed = seg[:best_split]
    return audiosegment_to_numpy(trimmed)


def cross_fade_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    silence_duration: float = 0.3,
) -> np.ndarray:
    """Concatenate audio chunks with silence gaps and cross-fade at boundaries.

    Args:
        chunks: list of numpy arrays, each (C, T).
        sample_rate: audio sample rate.
        silence_duration: total silence gap duration in seconds.

    Returns:
        Merged numpy array (C, T_total).
    """
    if len(chunks) == 1:
        return chunks[0]

    total_n = int(silence_duration * sample_rate)
    fade_n = total_n // 3
    silence_n = fade_n
    merged = chunks[0].copy()

    for chunk in chunks[1:]:
        parts = [merged]

        fout_n = min(fade_n, merged.shape[-1])
        if fout_n > 0:
            w_out = np.linspace(1, 0, fout_n, dtype=np.float32)[np.newaxis, :]
            parts[-1][..., -fout_n:] *= w_out

        parts.append(np.zeros((chunks[0].shape[0], silence_n), dtype=np.float32))

        fade_in = chunk.copy()
        fin_n = min(fade_n, fade_in.shape[-1])
        if fin_n > 0:
            w_in = np.linspace(0, 1, fin_n, dtype=np.float32)[np.newaxis, :]
            fade_in[..., :fin_n] *= w_in

        parts.append(fade_in)
        merged = np.concatenate(parts, axis=-1)

    return merged
