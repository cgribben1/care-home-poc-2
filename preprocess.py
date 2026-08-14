import numpy as np
from scipy.signal import butter, sosfilt

from config import MIN_PEAK_NORMALIZE


def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """Remove very low-frequency rumble (HVAC, mic handling)."""
    if len(audio) < 32:
        return audio
    sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def reduce_background_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Light stationary noise reduction — keep mild to preserve impacts."""
    import noisereduce as nr

    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=0.35,
    )
    return reduced.astype(np.float32)


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if max_abs < 1e-8:
        return audio
    return (audio / max_abs * peak).astype(np.float32)


def preprocess_audio(
    audio: np.ndarray,
    sr: int,
    *,
    denoise: bool = True,
    highpass: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    processed = audio.astype(np.float32)
    if highpass:
        processed = highpass_filter(processed, sr)
    max_abs = float(np.max(np.abs(processed))) if len(processed) else 0.0
    # Denoise + peak-normalize only when there is real signal — otherwise
    # silence gets blown up to full scale and confuses the classifier.
    if denoise and max_abs >= MIN_PEAK_NORMALIZE:
        processed = reduce_background_noise(processed, sr)
        max_abs = float(np.max(np.abs(processed))) if len(processed) else 0.0
    if normalize and max_abs >= MIN_PEAK_NORMALIZE:
        processed = normalize_peak(processed)
    return processed
