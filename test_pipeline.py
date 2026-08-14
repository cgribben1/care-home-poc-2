"""End-to-end test: calibration, classifier, and YAMNet on sample clips."""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

from analyze import get_analyzer
from calibration import load_calibration

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000


def test_clip(path: Path, mode: str, expect_alert: bool | None) -> bool:
    waveform, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    rms = float(np.sqrt(np.mean(waveform.astype(np.float32) ** 2)))
    _, analyze = get_analyzer(mode)
    result = analyze(waveform, sr, denoise=True, highpass=True, threshold=None, rms=rms)
    ok = True
    if expect_alert is True and not result.alert:
        ok = False
    if expect_alert is False and result.alert:
        ok = False
    status = "OK" if ok else "FAIL"
    alert = result.alert or "none"
    print(f"  [{status}] {path.name:40} rms={rms:.4f} alert={alert}")
    return ok


def main() -> int:
    print("=== Pipeline test ===\n")

    cal = load_calibration()
    if cal:
        print("Calibration loaded:")
        print(f"  median rms     = {cal['rms_median']:.4f}")
        print(f"  alert floor    = {cal['suggested_min_alert_rms']:.4f}")
        print(f"  training chunks = {len(cal.get('training_chunks', []))}\n")
    else:
        print("WARNING: no calibration file\n")

    modes = ["yamnet", "hybrid"]
    if not (ROOT / "models" / "classifier.joblib").exists():
        modes = ["yamnet"]
        print("Classifier model missing — testing YAMNet only\n")

    all_ok = True
    ambient = ROOT / "data" / "calibration" / "room_ambient_latest.wav"
    tests: list[tuple[Path, bool | None]] = []

    if ambient.exists():
        tests.append((ambient, False))

    for label, expect in [("fall", True), ("cough", True), ("distress", True), ("normal", False)]:
        class_dir = ROOT / "data" / "train" / label
        if not class_dir.exists():
            continue
        wavs = sorted(class_dir.glob("*.wav"))
        if not wavs:
            continue
        # Pick a clean (non-augmented) clip if possible
        clean = [w for w in wavs if "_aug" not in w.name and "device_silence" not in w.name]
        pick = clean[0] if clean else wavs[0]
        tests.append((pick, expect))

    for mode in modes:
        print(f"--- mode: {mode} ---")
        for path, expect in tests:
            if not path.exists():
                continue
            if not test_clip(path, mode, expect):
                all_ok = False
        print()

    if all_ok:
        print("All tests passed.")
        return 0
    print("Some tests failed — thresholds may need tuning.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
