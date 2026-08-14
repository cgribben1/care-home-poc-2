"""Embedding classifier trained on top of frozen YAMNet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import librosa
import numpy as np

from config import (
    CLASSIFIER_CLASSES,
    CLASSIFIER_MAX_NORMAL,
    CLASSIFIER_META_PATH,
    CLASSIFIER_NORMAL_MARGIN,
    CLASSIFIER_PATH,
    CLASSIFIER_THRESHOLDS,
    MIN_ALERT_RMS,
    MIN_PREPROCESS_RMS,
    NON_ALERT_CLASSES,
    SAMPLE_RATE,
)
from calibration import get_min_alert_rms, get_min_preprocess_rms
from detector import CategoryResult, DetectionResult, LabelScore
from embeddings import extract_frame_embeddings, pool_clip_embedding
from preprocess import preprocess_audio
from quiet_result import quiet_result


@dataclass
class ClassifierBundle:
    pipeline: object
    classes: list[str]
    meta: dict


@dataclass
class EmbeddingClassifier:
    bundle: ClassifierBundle | None = None

    def load(self, model_path: str | Path = CLASSIFIER_PATH) -> bool:
        path = Path(model_path)
        if not path.exists():
            return False
        pipeline = joblib.load(path)
        meta: dict = {}
        meta_path = Path(CLASSIFIER_META_PATH)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = meta.get("classes", CLASSIFIER_CLASSES)
        self.bundle = ClassifierBundle(pipeline=pipeline, classes=classes, meta=meta)
        return True

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def analyze_array(
        self,
        waveform: np.ndarray,
        sr: int,
        *,
        denoise: bool = True,
        highpass: bool = True,
        threshold: float | None = None,
        rms: float | None = None,
        model=None,
    ) -> DetectionResult:
        if not self.ready:
            raise RuntimeError("Classifier not loaded")

        assert self.bundle is not None
        if sr != SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        raw = waveform.astype(np.float32)
        raw_rms = rms if rms is not None else float(np.sqrt(np.mean(raw**2)))
        duration_sec = len(raw) / sr

        min_alert_rms = get_min_alert_rms(MIN_ALERT_RMS)
        min_preprocess_rms = get_min_preprocess_rms(MIN_PREPROCESS_RMS)

        if raw_rms < min_alert_rms:
            return quiet_result(duration_sec)

        if raw_rms < min_preprocess_rms:
            processed = raw
        else:
            processed = preprocess_audio(
                raw,
                sr,
                denoise=denoise,
                highpass=highpass,
                normalize=True,
            )

        if model is None:
            from detector import detector

            detector.load()
            model = detector._model

        frame_emb = extract_frame_embeddings(model, processed)
        clip_emb = pool_clip_embedding(frame_emb)
        probs = self.bundle.pipeline.predict_proba(clip_emb.reshape(1, -1))[0]

        class_to_prob = {
            name: float(probs[i]) for i, name in enumerate(self.bundle.classes)
        }
        # Combined non-alert score — model should prefer normal/other over alert classes
        normal_score = sum(class_to_prob.get(c, 0.0) for c in NON_ALERT_CLASSES)

        categories: list[CategoryResult] = []
        alert: str | None = None
        best_alert_score = 0.0

        for class_name in self.bundle.classes:
            if class_name in NON_ALERT_CLASSES:
                continue

            score = class_to_prob[class_name]
            cat_threshold = threshold if threshold is not None else CLASSIFIER_THRESHOLDS.get(
                class_name, 0.88
            )
            beats_normal = score >= normal_score + CLASSIFIER_NORMAL_MARGIN
            normal_low_enough = normal_score <= CLASSIFIER_MAX_NORMAL
            triggered = (
                score >= cat_threshold and beats_normal and normal_low_enough
            )

            if triggered and score > best_alert_score:
                best_alert_score = score
                alert = class_name

            categories.append(
                CategoryResult(
                    category=class_name,
                    score=score,
                    triggered=triggered,
                    top_labels=[LabelScore(f"classifier:{class_name}", score)],
                )
            )

        categories.sort(key=lambda item: item.score, reverse=True)
        top_overall = [
            LabelScore("classifier:normal", normal_score),
            *[
                LabelScore(f"classifier:{name}", class_to_prob[name])
                for name in sorted(class_to_prob.keys())
                if name != "normal"
            ][:3],
        ]

        return DetectionResult(
            duration_sec=duration_sec,
            categories=categories,
            top_overall=top_overall,
            alert=alert,
            method="classifier",
        )


embedding_classifier = EmbeddingClassifier()
