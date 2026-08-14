"""Read framed PCM audio from XIAO ESP32S3 and run YAMNet detection."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import serial

from config import CLASSIFIER_ALERT_STREAK, MIN_ALERT_RMS, SAMPLE_RATE
from analyze import apply_alert_streak, get_analyzer
from calibration import get_min_alert_rms, load_calibration
from serial_io import collect_pcm, open_device_serial, wait_for_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live acoustic detection from XIAO ESP32S3 serial stream"
    )
    parser.add_argument("--port", default="COM5", help="Serial port (default: COM5)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.5,
        help="Seconds of audio per analysis window",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=1.0,
        help="Minimum seconds between analyses",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=["auto", "yamnet", "classifier", "hybrid"],
        default="auto",
        help="Detection mode: yamnet (default), classifier, or hybrid (both must agree)",
    )
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-highpass", action="store_true")
    return parser.parse_args()


def format_result(result) -> str:
    lines = []
    if result.alert:
        lines.append(f"ALERT: {result.alert.upper()} ({result.method})")
    else:
        lines.append("No alert")

    for cat in result.categories:
        top = cat.top_labels[0] if cat.top_labels else None
        match = f"{top.label} ({top.score:.2f})" if top else "—"
        flag = " ***" if cat.triggered else ""
        lines.append(f"  {cat.category:10} {cat.score:.2f}{flag}  [{match}]")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    window_samples = int(args.window_sec * SAMPLE_RATE)

    try:
        mode_name, analyze = get_analyzer(args.mode)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(f"Detection mode: {mode_name}")
    cal = load_calibration()
    if cal:
        print(
            f"Room calibration: median rms={cal['rms_median']:.4f}, "
            f"alert floor={cal['suggested_min_alert_rms']:.4f}"
        )
    else:
        print(f"No room calibration yet — run .\\record_ambient.bat")
    print(f"Opening {args.port} @ {args.baud}...")
    print("Waiting for framed PCM from firmware...")
    print(f"Analyzing every ~{args.interval_sec}s using {args.window_sec}s windows\n")

    try:
        print("(Resetting board and waiting for USB serial...)")
        ser = open_device_serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as exc:
        print(f"Could not open serial port: {exc}")
        return 1

    if not wait_for_stream(ser):
        ser.close()
        return 1

    last_run = 0.0
    alert_streak = 0
    last_alert: str | None = None

    try:
        while True:
            audio_i16 = collect_pcm(ser, window_samples)
            if audio_i16 is None:
                print(
                    "Timed out waiting for audio frames — re-syncing "
                    "(check firmware upload & wiring if this repeats)."
                )
                continue

            now = time.time()
            if now - last_run < args.interval_sec:
                continue
            last_run = now

            waveform = audio_i16.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(waveform**2)))
            analyze_kwargs = {
                "denoise": not args.no_denoise,
                "highpass": not args.no_highpass,
                "threshold": args.threshold,
            }
            if mode_name in ("classifier", "hybrid"):
                analyze_kwargs["rms"] = rms
            result = analyze(waveform, SAMPLE_RATE, **analyze_kwargs)

            if mode_name in ("classifier", "hybrid"):
                if result.alert:
                    if result.alert == last_alert:
                        alert_streak += 1
                    else:
                        alert_streak = 1
                        last_alert = result.alert
                else:
                    alert_streak = 0
                    last_alert = None
                result = apply_alert_streak(
                    result, alert_streak, CLASSIFIER_ALERT_STREAK
                )

            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] rms={rms:.4f}")
            print(format_result(result))
            print("-" * 40)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
