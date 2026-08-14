"""Unified detection entry — YAMNet baseline or trained embedding classifier."""

from __future__ import annotations

import numpy as np

from classifier_detector import embedding_classifier
from config import CLASSIFIER_ONLY_CLASSES, CLASSIFIER_PATH, MIN_ALERT_RMS
from calibration import get_min_alert_rms, load_calibration
from detector import DetectionResult, detector
from quiet_result import quiet_result


def get_analyzer(mode: str = "auto"):
    if mode == "yamnet":
        return "yamnet", _analyze_yamnet
    if mode == "classifier":
        if not embedding_classifier.load():
            raise FileNotFoundError(
                f"Classifier not found at {CLASSIFIER_PATH}. Run train_classifier.py first."
            )
        detector.load()
        return "classifier", _analyze_classifier
    if mode == "hybrid":
        if not embedding_classifier.load():
            raise FileNotFoundError(
                f"Classifier not found at {CLASSIFIER_PATH}. Run train_classifier.py first."
            )
        detector.load()
        return "hybrid", _analyze_hybrid

    if mode == "auto":
        if embedding_classifier.load() and load_calibration():
            detector.load()
            return "hybrid", _analyze_hybrid
        if embedding_classifier.load():
            detector.load()
            return "classifier", _analyze_classifier
        return "yamnet", _analyze_yamnet

    raise ValueError(f"Unknown mode: {mode}")


def _analyze_yamnet(
    waveform: np.ndarray,
    sr: int,
    *,
    denoise: bool,
    highpass: bool,
    threshold: float | None,
    rms: float | None = None,
) -> DetectionResult:
    return detector.analyze_array(
        waveform, sr, denoise=denoise, highpass=highpass, threshold=threshold
    )


def _analyze_classifier(
    waveform: np.ndarray,
    sr: int,
    *,
    denoise: bool,
    highpass: bool,
    threshold: float | None,
    rms: float | None = None,
) -> DetectionResult:
    return embedding_classifier.analyze_array(
        waveform,
        sr,
        denoise=denoise,
        highpass=highpass,
        threshold=threshold,
        rms=rms,
        model=detector._model,
    )


def _analyze_hybrid(
    waveform: np.ndarray,
    sr: int,
    *,
    denoise: bool,
    highpass: bool,
    threshold: float | None,
    rms: float | None = None,
) -> DetectionResult:
    """Alert only when YAMNet and classifier agree (best for live ESP32 stream)."""
    if rms is None:
        rms = float(np.sqrt(np.mean(waveform.astype(np.float32) ** 2)))
    if rms < get_min_alert_rms(MIN_ALERT_RMS):
        return quiet_result(len(waveform) / sr, method="hybrid")

    yam = _analyze_yamnet(
        waveform, sr, denoise=denoise, highpass=highpass, threshold=threshold, rms=rms
    )
    clf = _analyze_classifier(
        waveform, sr, denoise=denoise, highpass=highpass, threshold=threshold, rms=rms
    )

    alert: str | None = None
    if clf.alert in CLASSIFIER_ONLY_CLASSES:
        # Classifier-only classes (e.g. fall) don't require YAMNet agreement
        alert = clf.alert
    elif yam.alert and clf.alert and yam.alert == clf.alert:
        # Hybrid agreement required for all other alert classes
        alert = yam.alert

    merged = DetectionResult(
        duration_sec=clf.duration_sec,
        categories=clf.categories,
        top_overall=[
            *clf.top_overall[:2],
            *yam.top_overall[:2],
        ],
        alert=alert,
        method="hybrid",
    )
    for cat in merged.categories:
        cat.triggered = cat.category == alert
    return merged


def apply_alert_streak(result: DetectionResult, streak: int, required: int) -> DetectionResult:
    """Suppress alert until the same class triggers N times in a row."""
    if result.alert is None:
        return result
    if streak < required:
        result.alert = None
        for cat in result.categories:
            cat.triggered = False
    return result
