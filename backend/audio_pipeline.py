from __future__ import annotations

import importlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal


TARGET_SAMPLE_RATE = 16000


@dataclass
class AudioProcessingResult:
    path: str
    sample_rate: int
    duration_seconds: float
    speech_seconds: float
    rnnoise_applied: bool
    vad_applied: bool


def load_wav_float32(wav_path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(str(wav_path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate and len(audio):
        new_len = max(1, int(len(audio) * target_sample_rate / sample_rate))
        audio = signal.resample_poly(audio, target_sample_rate, sample_rate).astype(np.float32)
        if len(audio) > new_len:
            audio = audio[:new_len]
        sample_rate = target_sample_rate

    return sample_rate, np.clip(audio, -1.0, 1.0).astype(np.float32)


def write_wav_float32(wav_path: str | Path, sample_rate: int, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    wavfile.write(str(wav_path), sample_rate, (pcm * 32767.0).astype(np.int16))


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if not len(audio):
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    target_peak = float(os.environ.get("AUDIO_TARGET_PEAK", "0.92"))
    if peak > target_peak or peak < 0.25:
        audio = audio * min(8.0, target_peak / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _fallback_noise_gate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """A conservative fallback when RNNoise is not installed."""
    if len(audio) < sample_rate // 5:
        return audio
    frame = max(1, int(0.02 * sample_rate))
    hop = frame
    rms_values = []
    for start in range(0, len(audio) - frame + 1, hop):
        chunk = audio[start : start + frame]
        rms_values.append(float(np.sqrt(np.mean(chunk * chunk))))
    if not rms_values:
        return audio
    floor = np.percentile(rms_values, 20)
    threshold = max(0.006, floor * 2.2)
    cleaned = audio.copy()
    for start in range(0, len(cleaned), hop):
        chunk = cleaned[start : start + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
        if rms < threshold:
            cleaned[start : start + frame] *= 0.25
    return cleaned.astype(np.float32)


def _try_rnnoise(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, bool]:
    if os.environ.get("RNNOISE_ENABLED", "1").lower() in {"0", "false", "no"}:
        return audio, False

    try:
        from pyrnnoise import RNNoise

        rnnoise_rate = 48000
        audio48 = audio
        if sample_rate != rnnoise_rate:
            audio48 = signal.resample_poly(audio, rnnoise_rate, sample_rate).astype(np.float32)
        pcm = (np.clip(audio48, -1.0, 1.0) * 32767.0).astype(np.int16)[None, :]
        denoiser = RNNoise(sample_rate=rnnoise_rate)
        frames = []
        for _, frame in denoiser.denoise_chunk(pcm, partial=True):
            arr = np.asarray(frame)
            if arr.ndim > 1:
                arr = arr[0]
            frames.append(arr.astype(np.float32))
        if frames:
            denoised48 = np.concatenate(frames)
            if np.max(np.abs(denoised48)) > 2:
                denoised48 = denoised48 / 32768.0
            if sample_rate != rnnoise_rate:
                denoised = signal.resample_poly(denoised48, sample_rate, rnnoise_rate).astype(np.float32)
            else:
                denoised = denoised48.astype(np.float32)
            if len(denoised) > len(audio):
                denoised = denoised[: len(audio)]
            elif len(denoised) < len(audio):
                denoised = np.pad(denoised, (0, len(audio) - len(denoised)))
            return np.clip(denoised, -1.0, 1.0), True
    except Exception:
        pass

    try:
        rnnoise_mod = importlib.import_module("rnnoise")
    except Exception:
        return _fallback_noise_gate(audio, sample_rate), False

    # Different Python RNNoise bindings expose slightly different APIs. Support
    # the common class/function shapes and fall back cleanly if none match.
    for attr in ("RNNoise", "Denoiser"):
        cls = getattr(rnnoise_mod, attr, None)
        if cls is None:
            continue
        try:
            denoiser = cls()
            for method_name in ("process", "denoise", "filter"):
                method = getattr(denoiser, method_name, None)
                if callable(method):
                    out = method(audio, sample_rate) if method_name != "process" else method(audio)
                    out = np.asarray(out, dtype=np.float32)
                    if out.shape == audio.shape:
                        return np.clip(out, -1.0, 1.0), True
        except Exception:
            continue

    for fn_name in ("denoise", "process"):
        fn = getattr(rnnoise_mod, fn_name, None)
        if callable(fn):
            try:
                out = np.asarray(fn(audio, sample_rate), dtype=np.float32)
                if out.shape == audio.shape:
                    return np.clip(out, -1.0, 1.0), True
            except Exception:
                continue

    return _fallback_noise_gate(audio, sample_rate), False


def _try_silero_vad(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, bool]:
    if os.environ.get("SILERO_VAD_ENABLED", "1").lower() in {"0", "false", "no"}:
        return audio, False
    if len(audio) < int(sample_rate * 0.25):
        return audio, False

    try:
        import torch
        from silero_vad import collect_chunks, get_speech_timestamps, load_silero_vad
    except Exception:
        return audio, False

    try:
        model = load_silero_vad()
        tensor = torch.from_numpy(audio)
        speech = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=sample_rate,
            threshold=float(os.environ.get("SILERO_VAD_THRESHOLD", "0.5")),
            min_speech_duration_ms=int(os.environ.get("SILERO_MIN_SPEECH_MS", "250")),
            min_silence_duration_ms=int(os.environ.get("SILERO_MIN_SILENCE_MS", "120")),
            speech_pad_ms=int(os.environ.get("SILERO_SPEECH_PAD_MS", "180")),
        )
        if not speech:
            return audio, False
        collected = collect_chunks(speech, tensor).numpy().astype(np.float32)
        # Avoid returning tiny fragments from accidental clicks.
        min_seconds = float(os.environ.get("SILERO_MIN_COLLECTED_SECONDS", "0.35"))
        if len(collected) < int(sample_rate * min_seconds):
            return audio, False
        return np.clip(collected, -1.0, 1.0), True
    except Exception:
        return audio, False


def prepare_audio_for_stt(wav_path: str | Path) -> AudioProcessingResult:
    sample_rate, audio = load_wav_float32(wav_path, TARGET_SAMPLE_RATE)
    original_seconds = len(audio) / float(sample_rate or TARGET_SAMPLE_RATE)

    audio, rnnoise_applied = _try_rnnoise(audio, sample_rate)
    audio = _normalize_audio(audio)
    speech_audio, vad_applied = _try_silero_vad(audio, sample_rate)
    speech_audio = _normalize_audio(speech_audio)

    fd, out_path = tempfile.mkstemp(suffix=".processed.wav")
    os.close(fd)
    write_wav_float32(out_path, sample_rate, speech_audio)
    return AudioProcessingResult(
        path=out_path,
        sample_rate=sample_rate,
        duration_seconds=original_seconds,
        speech_seconds=len(speech_audio) / float(sample_rate or TARGET_SAMPLE_RATE),
        rnnoise_applied=rnnoise_applied,
        vad_applied=vad_applied,
    )
