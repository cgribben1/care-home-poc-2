"""Train a small CNN on log-mel spectrograms for on-device TFLite inference."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

# Mel spectrogram parameters — must match firmware exactly
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64
DURATION = 1.0
# center=False: N_FRAMES = (AUDIO_SAMPLES - N_FFT) // HOP_LENGTH + 1
N_FRAMES = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1  # 97

CLASSES = ["fall", "cough", "normal", "other"]
TRAIN_DIR = Path("data/train")
MODEL_DIR = Path("models")

MAX_PER_CLASS = 5000  # cap to balance classes and keep epoch time manageable


def load_mel(wav_path: Path) -> np.ndarray | None:
    try:
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        target = int(SAMPLE_RATE * DURATION)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        y = y[:target]
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=20, fmax=8000, center=False,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        log_mel = log_mel.T  # (n_frames, n_mels)
        if log_mel.shape[0] > N_FRAMES:
            log_mel = log_mel[:N_FRAMES]
        elif log_mel.shape[0] < N_FRAMES:
            log_mel = np.pad(log_mel, ((0, N_FRAMES - log_mel.shape[0]), (0, 0)))
        return log_mel
    except Exception:
        return None


def build_dataset():
    X, y = [], []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found — skipping")
            continue
        wavs = list(class_dir.rglob("*.wav"))
        if len(wavs) > MAX_PER_CLASS:
            rng = np.random.default_rng(42)
            wavs = list(rng.choice(wavs, MAX_PER_CLASS, replace=False))
        print(f"  {class_name}: {len(wavs)} clips")
        for wav_path in wavs:
            mel = load_mel(wav_path)
            if mel is not None:
                X.append(mel)
                y.append(class_idx)
    return np.array(X, dtype=np.float32)[..., np.newaxis], np.array(y, dtype=np.int32)


def build_model(n_frames: int, n_mels: int, n_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(n_frames, n_mels, 1))
    x = inp
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    X, y = build_dataset()
    print(f"Total: {len(X)} samples  shape={X.shape}")

    counts = np.bincount(y, minlength=len(CLASSES))
    class_weights = {
        i: len(y) / (len(CLASSES) * c) for i, c in enumerate(counts) if c > 0
    }
    print("Class weights:", {CLASSES[i]: round(w, 2) for i, w in class_weights.items()})

    idx = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[idx[:split]], y[idx[:split]]
    X_val, y_val = X[idx[split:]], y[idx[split:]]

    mean = float(X_tr.mean())
    std = float(X_tr.std()) + 1e-8
    X_tr = (X_tr - mean) / std
    X_val = (X_val - mean) / std

    (MODEL_DIR / "cnn_norm_stats.json").write_text(
        json.dumps({"mean": mean, "std": std}, indent=2)
    )

    model = build_model(N_FRAMES, N_MELS, len(CLASSES))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                str(MODEL_DIR / "cnn_model_best.h5"),
                save_best_only=True,
                monitor="val_accuracy",
                verbose=1,
            ),
        ],
    )

    model.save(str(MODEL_DIR / "cnn_model.h5"))

    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\nPer-class validation accuracy:")
    for i, cls in enumerate(CLASSES):
        mask = y_val == i
        if not mask.any():
            continue
        acc = (y_pred[mask] == i).mean()
        print(f"  {cls:10s}: {acc:.1%}  ({mask.sum()} samples)")

    print(f"\nSaved: {MODEL_DIR}/cnn_model.h5")
    print("Next: python convert_tflite.py")


if __name__ == "__main__":
    main()
