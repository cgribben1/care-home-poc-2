from __future__ import annotations

import csv
from dataclasses import dataclass, field
from io import StringIO

import librosa
import numpy as np
import tensorflow_hub as hub

from config import (
    CATEGORY_LABELS,
    CATEGORY_SCORING,
    CATEGORY_THRESHOLDS,
    DEFAULT_THRESHOLD,
    FALSE_POSITIVE_LABELS,
    FALSE_POSITIVE_PENALTY,
    FALSE_POSITIVE_TRIGGER,
    SAMPLE_RATE,
    SUSTAINED_MIN_FRAMES,
)
from preprocess import preprocess_audio


@dataclass
class LabelScore:
    label: str
    score: float


@dataclass
class CategoryResult:
    category: str
    score: float
    triggered: bool
    top_labels: list[LabelScore] = field(default_factory=list)


@dataclass
class DetectionResult:
    duration_sec: float
    categories: list[CategoryResult]
    top_overall: list[LabelScore]
    alert: str | None
    method: str = "yamnet"


class YAMNetDetector:
    def __init__(self) -> None:
        self._model = None
        self._class_names: list[str] = []
        self._category_indices: dict[str, list[int]] = {}
        self._false_positive_indices: list[int] = []

    def load(self) -> None:
        if self._model is not None:
            return

        print("Loading YAMNet from TensorFlow Hub (first run may download ~17 MB)...")
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, encoding="utf-8") as f:
            class_map_csv = f.read()
        self._class_names = self._parse_class_map(class_map_csv)
        self._category_indices = self._build_category_indices()
        self._false_positive_indices = self._build_false_positive_indices()
        print(f"YAMNet ready ({len(self._class_names)} AudioSet classes).")

    @staticmethod
    def _parse_class_map(csv_text: str) -> list[str]:
        names: list[str] = []
        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            names.append(row["display_name"])
        return names

    def _build_category_indices(self) -> dict[str, list[int]]:
        lookup = {name.lower(): idx for idx, name in enumerate(self._class_names)}
        indices: dict[str, list[int]] = {}

        for category, labels in CATEGORY_LABELS.items():
            matched: list[int] = []
            for label in labels:
                idx = lookup.get(label.lower())
                if idx is not None:
                    matched.append(idx)
            indices[category] = matched

        return indices

    def _build_false_positive_indices(self) -> list[int]:
        lookup = {name.lower(): idx for idx, name in enumerate(self._class_names)}
        return [
            lookup[name.lower()]
            for name in FALSE_POSITIVE_LABELS
            if name.lower() in lookup
        ]

    def _load_waveform(self, audio_path: str) -> tuple[np.ndarray, int]:
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return waveform.astype(np.float32), sr

    def analyze_file(
        self,
        audio_path: str,
        *,
        denoise: bool = True,
        highpass: bool = True,
        threshold: float | None = None,
    ) -> DetectionResult:
        self.load()
        waveform, sr = self._load_waveform(audio_path)
        waveform = preprocess_audio(waveform, sr, denoise=denoise, highpass=highpass)
        return self._analyze_waveform(waveform, sr, threshold)

    def analyze_array(
        self,
        waveform: np.ndarray,
        sr: int,
        *,
        denoise: bool = True,
        highpass: bool = True,
        threshold: float | None = None,
    ) -> DetectionResult:
        self.load()
        if sr != SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        waveform = preprocess_audio(waveform.astype(np.float32), sr, denoise=denoise, highpass=highpass)
        return self._analyze_waveform(waveform, sr, threshold)

    @staticmethod
    def _aggregate_category_score(
        cat_frame_scores: np.ndarray,
        mode: str,
        threshold: float,
    ) -> float:
        if cat_frame_scores.size == 0:
            return 0.0

        per_frame = cat_frame_scores.max(axis=1)
        if mode == "max_frame":
            return float(per_frame.max())

        top_k = np.sort(per_frame)[::-1][:3]
        score = float(top_k.mean())
        sustained = int(np.sum(per_frame >= threshold * 0.5))
        if sustained < SUSTAINED_MIN_FRAMES:
            score *= 0.65
        return score

    def _false_positive_penalty(self, frame_scores: np.ndarray) -> float:
        if not self._false_positive_indices:
            return 0.0
        fp_scores = frame_scores[:, self._false_positive_indices].max(axis=1)
        peak_fp = float(fp_scores.max())
        if peak_fp < FALSE_POSITIVE_TRIGGER:
            return 0.0
        return FALSE_POSITIVE_PENALTY * min(1.0, peak_fp / 0.5)

    def _analyze_waveform(
        self,
        waveform: np.ndarray,
        sr: int,
        threshold: float | None,
    ) -> DetectionResult:
        scores, _, _ = self._model(waveform)
        frame_scores = scores.numpy()
        clip_scores = frame_scores.max(axis=0)
        fp_penalty = self._false_positive_penalty(frame_scores)

        top_overall = self._top_labels(clip_scores, limit=8)
        categories: list[CategoryResult] = []

        for category, idxs in self._category_indices.items():
            if not idxs:
                categories.append(CategoryResult(category, 0.0, False, []))
                continue

            cat_threshold = threshold if threshold is not None else CATEGORY_THRESHOLDS.get(
                category, DEFAULT_THRESHOLD
            )
            mode = CATEGORY_SCORING.get(category, "max_frame")
            cat_frame_scores = frame_scores[:, idxs]
            category_score = self._aggregate_category_score(
                cat_frame_scores, mode, cat_threshold
            )
            category_score = max(0.0, category_score - fp_penalty)

            label_scores = [
                LabelScore(self._class_names[idx], float(clip_scores[idx]))
                for idx in idxs
            ]
            label_scores.sort(key=lambda item: item.score, reverse=True)
            categories.append(
                CategoryResult(
                    category=category,
                    score=category_score,
                    triggered=category_score >= cat_threshold,
                    top_labels=label_scores[:5],
                )
            )

        categories.sort(key=lambda item: item.score, reverse=True)
        triggered = [c for c in categories if c.triggered]
        alert = triggered[0].category if triggered else None

        return DetectionResult(
            duration_sec=len(waveform) / sr,
            categories=categories,
            top_overall=top_overall,
            alert=alert,
            method="yamnet",
        )

    def _top_labels(self, clip_scores: np.ndarray, limit: int = 8) -> list[LabelScore]:
        top_idx = np.argsort(clip_scores)[::-1][:limit]
        return [
            LabelScore(self._class_names[idx], float(clip_scores[idx]))
            for idx in top_idx
        ]


detector = YAMNetDetector()
