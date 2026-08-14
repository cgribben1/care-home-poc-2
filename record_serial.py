"""Record PCM from XIAO serial stream and save a WAV for playback."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import serial

from config import SAMPLE_RATE
from serial_io import collect_pcm, open_device_serial, wait_for_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record built-in (or INMP441) mic via serial to WAV"
    )
    parser.add_argument("--port", default="COM5", help="Serial port (default: COM5)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Recording length in seconds (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="recording.wav",
        help="Output WAV path (default: recording.wav)",
    )
    return parser.parse_args()


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())


def main() -> int:
    args = parse_args()
    target_samples = int(args.seconds * SAMPLE_RATE)
    out_path = Path(args.output).resolve()

    print(f"Opening {args.port}...")
    print(f"Will record {args.seconds}s ({target_samples} samples) -> {out_path}")
    print("Speak or clap after recording starts.\n")

    try:
        print("(Resetting board and waiting for USB serial...)")
        ser = open_device_serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as exc:
        print(f"Could not open serial port: {exc}")
        return 1

    if not wait_for_stream(ser):
        ser.close()
        return 1

    print("Recording...")
    pcm = collect_pcm(ser, target_samples, timeout_sec=args.seconds + 10)
    ser.close()

    if pcm is None or len(pcm) < target_samples // 2:
        print("Recording failed — not enough audio received.")
        return 1

    pcm = pcm[:target_samples]
    waveform = pcm.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(waveform**2)))
    peak = float(np.max(np.abs(waveform)))

    save_wav(out_path, pcm, SAMPLE_RATE)

    print(f"\nSaved {len(pcm) / SAMPLE_RATE:.1f}s to:")
    print(f"  {out_path}")
    print(f"  rms={rms:.4f}  peak={peak:.4f}")
    print("\nOpen the WAV in any player (e.g. Windows Media Player, VLC).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
