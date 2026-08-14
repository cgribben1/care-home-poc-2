"""Tier 1: lightweight always-on event gate (no ML — runs cheaply on every chunk)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration import get_min_alert_rms
from config import MIN_ALERT_RMS, SAMPLE_RATE


@dataclass
class Tier1Config:
    """Tune how sensitive the always-on listener is."""

    spike_multiplier: float = 3.5
    jump_multiplier: float = 4.0
    ema_alpha: float = 0.04
    cooldown_sec: float = 2.0
    min_absolute_rms: float | None = None  # uses calibration if None


@dataclass
class Tier1Result:
    triggered: bool
    rms: float
    peak: float
    baseline_rms: float
    reason: str


class Tier1Gate:
    """
    Always-on listener. Uses energy + adaptive baseline only (no TensorFlow).

    In production this logic can move onto the ESP32; for the POC it runs on the PC
    while the board streams raw PCM.
    """

    def __init__(self, config: Tier1Config | None = None) -> None:
        self.config = config or Tier1Config()
        floor = get_min_alert_rms(MIN_ALERT_RMS)
        self._min_rms = self.config.min_absolute_rms or floor
        self._baseline = max(floor, 0.001)
        self._prev_rms = self._baseline
        self._cooldown_until = 0.0
        self._sample_clock = 0

    @property
    def baseline_rms(self) -> float:
        return self._baseline

    def _update_baseline(self, rms: float, triggered: bool) -> None:
        if triggered:
            return
        alpha = self.config.ema_alpha
        self._baseline = (1 - alpha) * self._baseline + alpha * rms

    def check(self, waveform: np.ndarray, now_sec: float) -> Tier1Result:
        wf = waveform.astype(np.float32)
        wf = wf - np.mean(wf)  # remove DC bias
        rms = float(np.sqrt(np.mean(wf**2))) if len(wf) else 0.0
        peak = float(np.max(np.abs(wf))) if len(wf) else 0.0

        if now_sec < self._cooldown_until:
            self._prev_rms = rms
            return Tier1Result(
                triggered=False,
                rms=rms,
                peak=peak,
                baseline_rms=self._baseline,
                reason="cooldown",
            )

        spike_threshold = max(self._min_rms, self._baseline * self.config.spike_multiplier)
        jump = abs(rms - self._prev_rms)
        jump_threshold = max(self._min_rms * 0.5, self._baseline * self.config.jump_multiplier)

        triggered = False
        reason = "quiet"

        if rms >= self._min_rms and rms >= spike_threshold:
            triggered = True
            reason = f"energy spike ({rms:.4f} > {spike_threshold:.4f})"
        elif rms >= self._min_rms and jump >= jump_threshold and rms > self._baseline * 1.5:
            triggered = True
            reason = f"sudden jump (+{jump:.4f})"
        elif peak >= 0.08 and rms >= self._min_rms:
            triggered = True
            reason = f"sharp peak ({peak:.3f})"

        if triggered:
            self._cooldown_until = now_sec + self.config.cooldown_sec
        else:
            self._update_baseline(rms, triggered=False)

        self._prev_rms = rms
        return Tier1Result(
            triggered=triggered,
            rms=rms,
            peak=peak,
            baseline_rms=self._baseline,
            reason=reason,
        )
