"""Audio augmentation: synthetic RIR, SNR mixing, gain jitter."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def generate_synthetic_rir(
    sr: int = 16000,
    duration_sec: float = 0.35,
    decay: float = 8.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simple exponentially decaying impulse + early reflections."""
    rng = np.random.default_rng(seed)
    length = max(32, int(duration_sec * sr))
    t = np.arange(length, dtype=np.float32) / sr
    rir = np.exp(-decay * t)

    for _ in range(rng.integers(2, 6)):
        delay = int(rng.integers(8, min(length // 2, 800)))
        gain = float(rng.uniform(0.08, 0.35))
        if delay < length:
            rir[delay] += gain

    rir += rng.normal(0.0, 0.002, size=length).astype(np.float32)
    rir /= np.max(np.abs(rir)) + 1e-8
    return rir.astype(np.float32)


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if len(audio) < 8:
        return audio
    out = fftconvolve(audio, rir, mode="same")
    return out.astype(np.float32)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into signal at the requested SNR (dB)."""
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)[: len(signal)]
    else:
        start = np.random.randint(0, max(1, len(noise) - len(signal)))
        noise = noise[start : start + len(signal)]

    sig_power = np.mean(signal**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    return (signal + scaled_noise).astype(np.float32)


def random_gain(audio: np.ndarray, min_gain: float = 0.35, max_gain: float = 1.0) -> np.ndarray:
    gain = float(np.random.uniform(min_gain, max_gain))
    return (audio * gain).astype(np.float32)


def augment_clip(
    audio: np.ndarray,
    sr: int,
    background: np.ndarray | None = None,
    *,
    use_rir: bool = True,
    use_noise: bool = True,
    use_gain: bool = True,
) -> np.ndarray:
    out = audio.astype(np.float32)
    if use_rir:
        rir = generate_synthetic_rir(sr=sr, seed=np.random.randint(0, 10_000))
        out = apply_rir(out, rir)
    if use_gain:
        out = random_gain(out)
    if use_noise and background is not None and len(background) > sr // 4:
        snr = float(np.random.uniform(4.0, 18.0))
        out = mix_at_snr(out, background.astype(np.float32), snr)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 1e-6:
        out = (out / peak * 0.95).astype(np.float32)
    return out
