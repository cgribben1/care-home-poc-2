"""Train a YAMNet-embedding classifier on prepared data/train clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CLASSIFIER_CLASSES, CLASSIFIER_META_PATH, CLASSIFIER_PATH, SAMPLE_RATE
from detector import detector
from embeddings import extract_frame_embeddings, pool_clip_embedding

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
CACHE_PATH = ROOT / "data" / "embedding_cache.npz"


def extract_features_streaming(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract YAMNet embeddings one clip at a time — never holds waveforms in RAM."""
    detector.load()

    wav_paths: list[Path] = []
    labels: list[str] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for wav_path in sorted(class_dir.glob("*.wav")):
            wav_paths.append(wav_path)
            labels.append(class_dir.name)

    total = len(wav_paths)
    print(f"Extracting YAMNet embeddings for {total} clips (streaming)...")

    features: list[np.ndarray] = []
    valid_labels: list[str] = []
    skipped = 0

    for idx, (wav_path, label) in enumerate(zip(wav_paths, labels)):
        if idx % 500 == 0 and idx:
            print(f"  {idx}/{total}  ({len(features)} ok, {skipped} skipped)")
        try:
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            if len(audio) < SAMPLE_RATE // 4:
                skipped += 1
                continue
            frame_emb = extract_frame_embeddings(detector._model, audio.astype(np.float32))
            features.append(pool_clip_embedding(frame_emb))
            valid_labels.append(label)
        except Exception as exc:
            skipped += 1
            continue

    print(f"Done: {len(features)} embeddings extracted, {skipped} skipped")
    return np.vstack(features), np.array(valid_labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train embedding classifier")
    parser.add_argument("--data-dir", default=str(TRAIN_DIR))
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing embedding cache")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print("Training data missing. Run: python prepare_data.py")
        return 1

    # Cache check FIRST — avoid loading any audio if embeddings are already computed
    if CACHE_PATH.exists() and not args.no_cache:
        print(f"Loading cached embeddings from {CACHE_PATH} ...")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        X = cache["X"]
        y = cache["y"]
        print(f"Loaded {len(X)} embeddings from cache")
    else:
        X, y = extract_features_streaming(data_dir)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE_PATH, X=X, y=y)
        print(f"Cached embeddings -> {CACHE_PATH}")

    if len(X) < 40:
        print(f"Too few clips ({len(X)}). Run prepare_data.py first.")
        return 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Training classifier...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nValidation report:\n", report)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, ROOT / CLASSIFIER_PATH)

    meta = {
        "classes": list(pipeline.named_steps["clf"].classes_),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X.shape[1]),
        "sources": ["ESC-50", "synthetic_falls", "SAFE", "COUGHVID"],
    }
    (ROOT / CLASSIFIER_META_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved model -> {ROOT / CLASSIFIER_PATH}")
    print("Use: python serial_bridge.py --mode classifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
