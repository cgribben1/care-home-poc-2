"""Record room ambient noise for training (normal class) and RMS calibration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import serial

from config import SAMPLE_RATE
from record_serial import save_wav
from serial_io import collect_pcm, wait_for_stream

ROOT = Path(__file__).resolve().parent
CAL_DIR = ROOT / "data" / "calibration"
TRAIN_NORMAL_DIR = ROOT / "data" / "train" / "normal"
CALIBRATION_PATH = CAL_DIR / "noise_floor.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record quiet room ambient audio for calibration and training"
    )
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--seconds",
        type=float,
        default=600.0,
        help="Recording length (default: 600s — leave room at normal background level)",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=5.0,
        help="Split into training chunks of this length",
    )
    parser.add_argument(
        "--no-training-chunks",
        action="store_true",
        help="Skip writing chunks to data/train/normal/",
    )
    return parser.parse_args()


def window_rms(waveform: np.ndarray, window_samples: int) -> np.ndarray:
    if len(waveform) < window_samples:
        return np.array([float(np.sqrt(np.mean(waveform**2)))], dtype=np.float32)
    trims = len(waveform) - (len(waveform) % window_samples)
    frames = waveform[:trims].reshape(-1, window_samples)
    return np.sqrt(np.mean(frames**2, axis=1))


def main() -> int:
    args = parse_args()
    target_samples = int(args.seconds * SAMPLE_RATE)

    print("=== Room ambient recording ===")
    print("Leave the room at NORMAL background level (TV/HVAC on if usually on).")
    print("Do not talk, clap, or walk near the mic during recording.")
    print(f"Duration: {args.seconds:.0f}s\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as exc:
        print(f"Could not open serial port: {exc}")
        return 1

    if not wait_for_stream(ser):
        ser.close()
        return 1

    print("Recording ambient...")
    pcm = collect_pcm(ser, target_samples, timeout_sec=args.seconds + 15)
    ser.close()

    if pcm is None or len(pcm) < target_samples // 2:
        print("Recording failed — not enough audio received.")
        return 1

    pcm = pcm[:target_samples]
    waveform = pcm.astype(np.float32) / 32768.0
    dc = float(np.mean(waveform))
    if abs(dc) > 0.01:
        print(f"DC offset detected ({dc:.4f}) — removing before calibration")
        waveform = waveform - dc

    window_samples = int(args.chunk_sec * SAMPLE_RATE)
    rms_frames = window_rms(waveform, window_samples)
    rms_median = float(np.median(rms_frames))
    rms_p95 = float(np.percentile(rms_frames, 95))
    rms_max = float(np.max(rms_frames))
    rms_mean = float(np.mean(rms_frames))

    # Alert threshold: well above typical ambient, below intentional events
    suggested_min_alert = max(rms_p95 * 2.5, rms_median * 4.0, 0.008)
    suggested_min_preprocess = max(rms_p95 * 2.0, rms_median * 3.0, 0.006)

    CAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    full_path = CAL_DIR / f"room_ambient_{stamp}.wav"
    latest_path = CAL_DIR / "room_ambient_latest.wav"

    save_wav(full_path, pcm, SAMPLE_RATE)
    save_wav(latest_path, pcm, SAMPLE_RATE)

    chunk_paths: list[str] = []
    if not args.no_training_chunks:
        TRAIN_NORMAL_DIR.mkdir(parents=True, exist_ok=True)
        n_chunks = len(waveform) // window_samples
        for i in range(n_chunks):
            start = i * window_samples
            chunk = pcm[start : start + window_samples]
            chunk_path = TRAIN_NORMAL_DIR / f"room_ambient_{stamp}_{i:03d}.wav"
            save_wav(chunk_path, chunk, SAMPLE_RATE)
            chunk_paths.append(str(chunk_path.relative_to(ROOT)))

    calibration = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": len(pcm) / SAMPLE_RATE,
        "source_file": str(full_path.relative_to(ROOT)),
        "window_sec": args.chunk_sec,
        "rms_mean": round(rms_mean, 6),
        "rms_median": round(rms_median, 6),
        "rms_p95": round(rms_p95, 6),
        "rms_max": round(rms_max, 6),
        "suggested_min_alert_rms": round(suggested_min_alert, 6),
        "suggested_min_preprocess_rms": round(suggested_min_preprocess, 6),
        "training_chunks": chunk_paths,
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    print(f"\nSaved ambient recording:")
    print(f"  {full_path}")
    print(f"  {latest_path}")
    print("\nRoom noise stats (per {:.1f}s window):".format(args.chunk_sec))
    print(f"  median rms = {rms_median:.4f}")
    print(f"  p95 rms    = {rms_p95:.4f}")
    print(f"  max rms    = {rms_max:.4f}")
    print("\nSuggested calibration:")
    print(f"  MIN_ALERT_RMS      = {suggested_min_alert:.4f}")
    print(f"  MIN_PREPROCESS_RMS = {suggested_min_preprocess:.4f}")
    print(f"\nWrote {CALIBRATION_PATH}")

    if chunk_paths:
        print(f"Added {len(chunk_paths)} training clips -> data/train/normal/")
        print("\nTo retrain with your room noise included:")
        print("  del data\\embedding_cache.npz")
        print("  python train_classifier.py")

    print("\nLive detection will use this calibration automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
