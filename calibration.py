"""Load room noise-floor calibration from ambient recordings."""

from __future__ import annotations

import json
from pathlib import Path

from config import MIN_ALERT_RMS, MIN_PREPROCESS_RMS

ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "data" / "calibration" / "noise_floor.json"


def load_calibration() -> dict | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_min_alert_rms(default: float = MIN_ALERT_RMS) -> float:
    cal = load_calibration()
    if not cal:
        return default
    return float(cal.get("suggested_min_alert_rms", default))


def get_min_preprocess_rms(default: float = MIN_PREPROCESS_RMS) -> float:
    cal = load_calibration()
    if not cal:
        return default
    return float(cal.get("suggested_min_preprocess_rms", default))
