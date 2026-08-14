#!/usr/bin/env python3
"""
Micro-AST training script — care-home acoustic event classification.

Architecture per spec:
  32x32 log-mel (80-7800 Hz, hop=512, 1.024s)
  → Conv2D patch embed 8x8 → 16 tokens × d=32
  → Learnable positional embeddings
  → 1-layer Transformer (2 heads, key_dim=16, FFN 32→64→32 GELU)
  → Global Average Pool → Dense(num_classes) → Softmax
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

# ── Audio / spectrogram params (§2) ──────────────────────────────────────────
SAMPLE_RATE = 16000
N_FFT       = 512
HOP_LENGTH  = 512       # non-overlapping windows
N_MELS      = 32
FMIN        = 80
FMAX        = 7800
DURATION    = 1.024     # seconds → 16384 samples
N_SAMPLES   = int(SAMPLE_RATE * DURATION)                         # 16384
N_FRAMES    = (N_SAMPLES - N_FFT) // HOP_LENGTH + 1               # 32

# ── AST architecture (§3) ────────────────────────────────────────────────────
PATCH_H   = 8
PATCH_W   = 8
N_PATCHES = (N_FRAMES // PATCH_H) * (N_MELS // PATCH_W)           # 16 tokens
D_MODEL   = 32
N_HEADS   = 2           # key_dim = D_MODEL // N_HEADS = 16
MLP_DIM   = 64

CLASSES   = ["fall", "cough", "normal", "other"]

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_DIR    = Path("data/train")
VAL_DIR      = Path("data/val")
MODEL_DIR    = Path("models")
MAX_PER_CLASS = 7000
BATCH_SIZE   = 32
EPOCHS       = 80
LR           = 1e-3


# ── Data helpers ──────────────────────────────────────────────────────────────

def _log_mel(path: Path, mean: float, std: float,
             add_noise: bool = False) -> np.ndarray | None:
    try:
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                            duration=DURATION)
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
        y = y[:N_SAMPLES]
        if add_noise:
            y = y + np.random.normal(0, 0.002 * (np.std(y) + 1e-8), y.shape)
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, center=False,
        )
        lm = librosa.power_to_db(mel, ref=np.max).astype(np.float32).T
        if lm.shape[0] > N_FRAMES:
            lm = lm[:N_FRAMES]
        elif lm.shape[0] < N_FRAMES:
            lm = np.pad(lm, ((0, N_FRAMES - lm.shape[0]), (0, 0)))
        return ((lm - mean) / std)[..., np.newaxis].astype(np.float32)
    except Exception:
        return None


def compute_norm_stats(wav_paths: list[Path]) -> tuple[float, float]:
    vals = []
    for p in wav_paths[:400]:
        try:
            y, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True,
                                duration=DURATION)
            if len(y) < N_SAMPLES:
                y = np.pad(y, (0, N_SAMPLES - len(y)))
            y = y[:N_SAMPLES]
            mel = librosa.feature.melspectrogram(
                y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX, center=False,
            )
            vals.append(librosa.power_to_db(mel, ref=np.max).flatten())
        except Exception:
            pass
    all_vals = np.concatenate(vals)
    return float(np.mean(all_vals)), float(np.std(all_vals))


def load_split(split_dir: Path, mean: float, std: float) -> tuple:
    X, y = [], []
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        wavs = sorted(cls_dir.rglob("*.wav"))
        random.shuffle(wavs)
        wavs = wavs[:MAX_PER_CLASS]
        loaded = 0
        for wav in wavs:
            mel = _log_mel(wav, mean, std)
            if mel is not None:
                X.append(mel)
                y.append(cls_idx)
                loaded += 1
        print(f"  {cls}: {loaded}")
    if not X:
        return None, None
    return (np.array(X, dtype=np.float32),
            tf.keras.utils.to_categorical(y, len(CLASSES)))


def augment(X: np.ndarray, y: np.ndarray, factor: int = 3) -> tuple:
    """SpecAugment (time + frequency masks) + Gaussian noise on mel values."""
    xs, ys = [X], [y]
    for _ in range(factor - 1):
        Xa = X.copy()
        for i in range(len(Xa)):
            # Time mask
            t = random.randint(0, max(1, N_FRAMES // 4))
            t0 = random.randint(0, max(0, N_FRAMES - t - 1))
            Xa[i, t0:t0 + t, :, :] = 0.0
            # Frequency mask
            f = random.randint(0, max(1, N_MELS // 4))
            f0 = random.randint(0, max(0, N_MELS - f - 1))
            Xa[i, :, f0:f0 + f, :] = 0.0
            # Gaussian noise on mel values
            Xa[i] += np.random.normal(0, 0.1, Xa[i].shape).astype(np.float32)
        xs.append(Xa)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


# ── Model (§3.1) ──────────────────────────────────────────────────────────────

class AddPositionalEmbedding(tf.keras.layers.Layer):
    """Learnable 1D positional embeddings — converts to TFLite Add op."""
    def build(self, input_shape):
        self.pos = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            name="pos",
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos

    def get_config(self):
        return super().get_config()


class StaticMultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention using only Keras layers with static shapes.
    Keras's built-in MHA uses tf.shape() internally, generating SHAPE/PACK/REDUCE_PROD
    ops that are not available in tflm_esp32. This implementation uses Reshape/Permute
    layers which produce static-shape RESHAPE/TRANSPOSE ops instead.
    """
    def __init__(self, num_heads, key_dim, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim   = key_dim
        self.seq_len   = seq_len
        self.d_model   = num_heads * key_dim
        self.scale     = key_dim ** -0.5

    def build(self, input_shape):
        H, D, S = self.num_heads, self.key_dim, self.seq_len
        self.q_dense = tf.keras.layers.Dense(H * D, use_bias=False, name=self.name + "_q")
        self.k_dense = tf.keras.layers.Dense(H * D, use_bias=False, name=self.name + "_k")
        self.v_dense = tf.keras.layers.Dense(H * D, use_bias=False, name=self.name + "_v")
        self.o_dense = tf.keras.layers.Dense(H * D, name=self.name + "_o")
        # Static Reshape: (B, S, H*D) → (B, S, H, D)  [batch not included in spec]
        self.rq = tf.keras.layers.Reshape((S, H, D))
        self.rk = tf.keras.layers.Reshape((S, H, D))
        self.rv = tf.keras.layers.Reshape((S, H, D))
        # Permute: (B, S, H, D) → (B, H, S, D)   [1-indexed, batch is implicit]
        self.pq = tf.keras.layers.Permute((2, 1, 3))
        self.pk = tf.keras.layers.Permute((2, 1, 3))
        self.pv = tf.keras.layers.Permute((2, 1, 3))
        # Permute back: (B, H, S, D) → (B, S, H, D)
        self.p_back = tf.keras.layers.Permute((2, 1, 3))
        # Final reshape: (B, S, H, D) → (B, S, H*D)
        self.r_out  = tf.keras.layers.Reshape((S, H * D))
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        super().build(input_shape)

    def call(self, x):
        Q = self.pq(self.rq(self.q_dense(x)))   # (B, H, S, D)
        K = self.pk(self.rk(self.k_dense(x)))
        V = self.pv(self.rv(self.v_dense(x)))

        scores = tf.matmul(Q, K, transpose_b=True) * self.scale  # (B, H, S, S)
        attn   = self.softmax(scores)
        ctx    = tf.matmul(attn, V)                               # (B, H, S, D)

        ctx = self.r_out(self.p_back(ctx))   # (B, S, H*D)
        return self.o_dense(ctx)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads,
                    "key_dim":   self.key_dim,
                    "seq_len":   self.seq_len})
        return cfg


def build_micro_ast() -> tf.keras.Model:
    # Input: (batch, 32 time frames, 32 mel bins, 1 channel)
    inp = tf.keras.Input(shape=(N_FRAMES, N_MELS, 1), name="mel_input")

    # Patch embedding: (32,32,1) → (4,4,32) → (16,32)
    x = tf.keras.layers.Conv2D(
        D_MODEL,
        kernel_size=(PATCH_H, PATCH_W),
        strides=(PATCH_H, PATCH_W),
        padding="valid",
        use_bias=True,
        name="patch_embed",
    )(inp)
    x = tf.keras.layers.Reshape((N_PATCHES, D_MODEL), name="patch_reshape")(x)

    # Learnable positional embeddings
    x = AddPositionalEmbedding(name="pos_embed")(x)

    # Transformer encoder — 1 layer, pre-norm, 2 heads, key_dim=16
    residual = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln1")(x)
    x = StaticMultiHeadAttention(
        num_heads=N_HEADS,
        key_dim=D_MODEL // N_HEADS,
        seq_len=N_PATCHES,
        name="mha",
    )(x)
    x = tf.keras.layers.Add(name="add1")([residual, x])

    # FFN: 32 → 64 → 32, ReLU6
    residual = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln2")(x)
    x = tf.keras.layers.Dense(MLP_DIM, activation="relu6", name="mlp1")(x)
    x = tf.keras.layers.Dense(D_MODEL, name="mlp2")(x)
    x = tf.keras.layers.Add(name="add2")([residual, x])

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_out")(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax",
                                name="output")(x)

    return tf.keras.Model(inp, out, name="micro_ast")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    MODEL_DIR.mkdir(exist_ok=True)

    all_wavs = sorted(TRAIN_DIR.rglob("*.wav"))
    if not all_wavs:
        raise FileNotFoundError(f"No WAV files in {TRAIN_DIR}")
    print(f"Found {len(all_wavs)} training files (capped at {MAX_PER_CLASS}/class)")

    print("Computing normalisation stats (balanced sample)...")
    sampled = []
    for cls in CLASSES:
        cls_wavs = sorted((TRAIN_DIR / cls).rglob("*.wav")) \
                   if (TRAIN_DIR / cls).exists() else []
        random.shuffle(cls_wavs)
        sampled.extend(cls_wavs[:100])
    mean, std = compute_norm_stats(sampled)
    print(f"  mean={mean:.3f}  std={std:.3f}")
    (MODEL_DIR / "ast_norm_stats.json").write_text(json.dumps({
        "mean": mean, "std": std,
        "n_fft": N_FFT, "hop": HOP_LENGTH,
        "n_mels": N_MELS, "n_frames": N_FRAMES,
        "fmin": FMIN, "fmax": FMAX, "duration": DURATION,
    }))

    print("Loading training data...")
    X_train, y_train = load_split(TRAIN_DIR, mean, std)
    if X_train is None:
        raise RuntimeError("No data loaded")
    print(f"  Total: {len(X_train)} samples  shape={X_train.shape}")

    print("Augmenting (×3, SpecAugment + Gaussian noise)...")
    X_train, y_train = augment(X_train, y_train, factor=3)
    print(f"  After augmentation: {len(X_train)} samples")

    X_val, y_val = None, None
    if VAL_DIR.exists():
        X_val, y_val = load_split(VAL_DIR, mean, std)
        if X_val is not None:
            print(f"  Validation: {len(X_val)} samples")

    model = build_micro_ast()
    model.summary()
    print(f"\nN_FRAMES={N_FRAMES}  N_MELS={N_MELS}  N_PATCHES={N_PATCHES}"
          f"  N_HEADS={N_HEADS}  FMIN={FMIN}  FMAX={FMAX}\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    val_monitor = "val_accuracy" if X_val is not None else "accuracy"
    val_loss_key = "val_loss" if X_val is not None else "loss"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "ast_model_best.h5"),
            monitor=val_monitor, save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=val_loss_key, factor=0.5, patience=8,
            min_lr=1e-5, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=val_monitor, patience=20, restore_best_weights=True,
        ),
    ]

    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val) if X_val is not None else None,
        callbacks=callbacks,
        shuffle=True,
    )

    model.save(str(MODEL_DIR / "ast_model.h5"))
    print("\nDone. Run: python convert_ast.py")


if __name__ == "__main__":
    main()
