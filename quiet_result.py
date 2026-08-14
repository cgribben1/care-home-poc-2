"""Return a no-alert result without running the classifier on silence."""

from __future__ import annotations

from detector import CategoryResult, DetectionResult, LabelScore


def quiet_result(duration_sec: float, method: str = "classifier") -> DetectionResult:
    categories = [
        CategoryResult("fall", 0.0, False, []),
        CategoryResult("distress", 0.0, False, []),
        CategoryResult("cough", 0.0, False, []),
    ]
    return DetectionResult(
        duration_sec=duration_sec,
        categories=categories,
        top_overall=[LabelScore("silence (skipped)", 1.0)],
        alert=None,
        method=method,
    )
