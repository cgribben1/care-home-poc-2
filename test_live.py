"""Quick live test: 3 analysis windows from COM5 serial stream."""

from __future__ import annotations

import time

import numpy as np
import serial

from analyze import get_analyzer
from calibration import load_calibration
from config import SAMPLE_RATE
from serial_io import collect_pcm, wait_for_stream


def main() -> int:
    cal = load_calibration()
    print("=== Live serial test (3 windows) ===")
    if cal:
        print(f"Calibration: alert floor={cal['suggested_min_alert_rms']:.4f}\n")

    mode_name, analyze = get_analyzer("auto")
    print(f"Mode: {mode_name}\n")

    ser = serial.Serial("COM5", 115200, timeout=0.2)
    if not wait_for_stream(ser):
        ser.close()
        return 1

    window = int(1.5 * SAMPLE_RATE)
    for i in range(3):
        pcm = collect_pcm(ser, window, timeout_sec=10)
        if pcm is None:
            print(f"Window {i+1}: timeout")
            continue
        wf = pcm.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(wf**2)))
        result = analyze(wf, SAMPLE_RATE, denoise=True, highpass=True, threshold=None, rms=rms)
        alert = result.alert or "none"
        print(f"Window {i+1}: rms={rms:.4f} alert={alert} ({result.method})")
        time.sleep(0.5)

    ser.close()
    print("\nDone. Try coughing once, then run again to compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
