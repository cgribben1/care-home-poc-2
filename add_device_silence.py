"""Add ESP32-style silence clips to normal class and retrain classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE

ROOT = Path(__file__).resolve().parent
NORMAL_DIR = ROOT / "data" / "train" / "normal"
CACHE_PATH = ROOT / "data" / "embedding_cache.npz"


def generate_device_silence(count: int = 500) -> None:
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)

    for i in range(count):
        n = int(rng.uniform(0.8, 2.0) * SAMPLE_RATE)
        kind = i % 4
        if kind == 0:
            # All-zero frames (like broken INMP441 stream)
            audio = np.zeros(n, dtype=np.float32)
        elif kind == 1:
            # int16 quantisation noise
            audio = rng.integers(-120, 120, n).astype(np.float32) / 32768.0
        elif kind == 2:
            # Very low hum + noise (quiet room)
            t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
            audio = (
                0.0015 * np.sin(2 * np.pi * 60 * t)
                + rng.normal(0, 0.0012, n).astype(np.float32)
            )
        else:
            # Occasional tiny spike then silence (serial glitch pattern)
            audio = rng.normal(0, 0.0006, n).astype(np.float32)
            spike_pos = int(rng.integers(0, max(1, n - 200)))
            audio[spike_pos : spike_pos + 20] += rng.normal(0, 0.008, 20)

        sf.write(NORMAL_DIR / f"device_silence_{i:04d}.wav", audio, SAMPLE_RATE)

    print(f"Wrote {count} device-silence clips to {NORMAL_DIR}")


def main() -> int:
    generate_device_silence()
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print(f"Removed stale cache {CACHE_PATH}")
    print("Run: python train_classifier.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
