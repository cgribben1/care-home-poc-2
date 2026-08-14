"""YAMNet embedding extraction helpers."""

from __future__ import annotations

import numpy as np


def extract_frame_embeddings(model, waveform: np.ndarray) -> np.ndarray:
    """Return YAMNet embeddings with shape (num_frames, 1024)."""
    _, embeddings, _ = model(waveform.astype(np.float32))
    return embeddings.numpy()


def pool_clip_embedding(frame_embeddings: np.ndarray, method: str = "mean_max") -> np.ndarray:
    """Collapse frame embeddings to one vector per clip."""
    if frame_embeddings.size == 0:
        return np.zeros(1024, dtype=np.float32)

    if method == "mean":
        return frame_embeddings.mean(axis=0).astype(np.float32)
    if method == "max":
        return frame_embeddings.max(axis=0).astype(np.float32)

    mean = frame_embeddings.mean(axis=0)
    peak = frame_embeddings.max(axis=0)
    return (0.65 * mean + 0.35 * peak).astype(np.float32)
