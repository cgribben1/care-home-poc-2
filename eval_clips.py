"""Evaluate YAMNet baseline vs embedding classifier on data/train holdout."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classifier_detector import embedding_classifier
from config import SAMPLE_RATE
from detector import detector

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"


def load_labeled_clips(data_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    waveforms: list[np.ndarray] = []
    labels: list[str] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for wav_path in sorted(class_dir.glob("*.wav")):
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            waveforms.append(audio.astype(np.float32))
            labels.append(class_dir.name)
    return waveforms, labels


def predict_yamnet(waveform: np.ndarray) -> str:
    result = detector.analyze_array(waveform, SAMPLE_RATE)
    if result.alert:
        return result.alert
    return "normal"


def predict_classifier(waveform: np.ndarray) -> str:
    result = embedding_classifier.analyze_array(waveform, SAMPLE_RATE, model=detector._model)
    if result.alert:
        return result.alert
    return "normal"


def evaluate(name: str, predictor, waveforms: list[np.ndarray], y_true: list[str]) -> None:
    y_pred = [predictor(w) for w in waveforms]
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate detectors on prepared clips")
    parser.add_argument("--data-dir", default=str(TRAIN_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print("No data/train — run prepare_data.py first")
        return 1

    waveforms, labels = load_labeled_clips(data_dir)
    _, X_test, _, y_test = train_test_split(
        waveforms, labels, test_size=0.2, random_state=42, stratify=labels
    )

    detector.load()
    evaluate("YAMNet label mapping (improved)", predict_yamnet, X_test, y_test)

    if embedding_classifier.load():
        evaluate("Embedding classifier", predict_classifier, X_test, y_test)
    else:
        print("\nClassifier model not found — run train_classifier.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
