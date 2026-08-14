"""Two-tier monitor: Tier 1 gate + Tier 2 hybrid model on buffered audio."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import serial

from analyze import get_analyzer
from config import SAMPLE_RATE
from detector import DetectionResult
from serial_io import collect_pcm, wait_for_stream
from tier1_gate import Tier1Gate, Tier1Result


@dataclass
class EventRecord:
    timestamp: str
    tier1_reason: str
    tier1_rms: float
    alert: str | None
    scores: dict[str, float]
    method: str


@dataclass
class MonitorState:
    running: bool = False
    connected: bool = False
    status: str = "Idle"
    tier1_rms: float = 0.0
    tier1_peak: float = 0.0
    tier1_baseline: float = 0.0
    tier1_triggered: bool = False
    tier1_reason: str = ""
    tier2_status: str = "Waiting for event…"
    last_alert: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    event_log: list[EventRecord] = field(default_factory=list)
    error: str | None = None


class TwoTierMonitor:
    """
    Tier 1 runs on every small chunk (cheap).
    Tier 2 (YAMNet + classifier hybrid) runs only when Tier 1 fires.
    """

    def __init__(
        self,
        port: str = "COM5",
        chunk_sec: float = 0.4,
        analysis_sec: float = 2.0,
    ) -> None:
        self.port = port
        self.chunk_samples = int(chunk_sec * SAMPLE_RATE)
        self._analysis_samples = int(analysis_sec * SAMPLE_RATE)
        self._gate = Tier1Gate()
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        self._max_buffer = int(4.0 * SAMPLE_RATE)

        self._state = MonitorState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._analyze = None
        self._mode_name = "hybrid"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._state.running = False
            self._state.status = "Stopped"

    def get_state(self) -> MonitorState:
        with self._lock:
            return MonitorState(
                running=self._state.running,
                connected=self._state.connected,
                status=self._state.status,
                tier1_rms=self._state.tier1_rms,
                tier1_peak=self._state.tier1_peak,
                tier1_baseline=self._state.tier1_baseline,
                tier1_triggered=self._state.tier1_triggered,
                tier1_reason=self._state.tier1_reason,
                tier2_status=self._state.tier2_status,
                last_alert=self._state.last_alert,
                scores=dict(self._state.scores),
                event_log=list(self._state.event_log[-8:]),
                error=self._state.error,
            )

    def _push_buffer(self, chunk: np.ndarray) -> None:
        self._buffer.append(chunk)
        self._buffer_samples += len(chunk)
        while self._buffer_samples > self._max_buffer and self._buffer:
            old = self._buffer.popleft()
            self._buffer_samples -= len(old)

    def _analysis_window(self) -> np.ndarray:
        if not self._buffer:
            return np.array([], dtype=np.float32)
        parts = list(self._buffer)
        audio = np.concatenate(parts)
        if len(audio) > self._analysis_samples:
            audio = audio[-self._analysis_samples :]
        return audio.astype(np.float32)

    def _set_state(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)

    def _run_tier2(self, audio: np.ndarray, tier1: Tier1Result) -> None:
        self._set_state(status="Analysing…", tier2_status="Running full model…")
        rms = float(np.sqrt(np.mean(audio**2)))
        assert self._analyze is not None
        result: DetectionResult = self._analyze(
            audio,
            SAMPLE_RATE,
            denoise=True,
            highpass=True,
            threshold=None,
            rms=rms,
        )
        scores = {c.category: c.score for c in result.categories}
        alert = result.alert
        ts = time.strftime("%H:%M:%S")
        record = EventRecord(
            timestamp=ts,
            tier1_reason=tier1.reason,
            tier1_rms=tier1.rms,
            alert=alert,
            scores=scores,
            method=result.method,
        )
        with self._lock:
            self._state.scores = scores
            self._state.last_alert = alert
            self._state.tier2_status = (
                f"Alert: {alert.upper()}" if alert else "No alert (normal / uncertain)"
            )
            self._state.status = "Event processed"
            self._state.event_log.append(record)
            if len(self._state.event_log) > 20:
                self._state.event_log = self._state.event_log[-20:]

    def _run(self) -> None:
        try:
            mode_name, analyze = get_analyzer("hybrid")
            self._analyze = analyze
            self._mode_name = mode_name
        except FileNotFoundError as exc:
            self._set_state(error=str(exc), status="Error")
            return

        self._set_state(running=True, status="Connecting…", error=None)

        try:
            ser = serial.Serial(self.port, 115200, timeout=0.2)
        except serial.SerialException as exc:
            self._set_state(running=False, error=f"Serial: {exc}", status="Error")
            return

        if not wait_for_stream(ser):
            ser.close()
            self._set_state(running=False, error="No audio stream from device", status="Error")
            return

        self._set_state(connected=True, status="Listening (Tier 1)")

        try:
            while not self._stop.is_set():
                chunk_i16 = collect_pcm(ser, self.chunk_samples, timeout_sec=3.0)
                if chunk_i16 is None:
                    continue

                chunk = chunk_i16.astype(np.float32) / 32768.0
                self._push_buffer(chunk)
                now = time.time()
                tier1 = self._gate.check(chunk, now)

                self._set_state(
                    tier1_rms=tier1.rms,
                    tier1_peak=tier1.peak,
                    tier1_baseline=tier1.baseline_rms,
                    tier1_triggered=tier1.triggered,
                    tier1_reason=tier1.reason,
                    status="Listening (Tier 1)",
                )

                if tier1.triggered:
                    self._set_state(
                        status="Event detected — Tier 1",
                        tier2_status="Preparing clip for full model…",
                    )
                    window = self._analysis_window()
                    if len(window) >= SAMPLE_RATE // 2:
                        self._run_tier2(window, tier1)

        except Exception as exc:
            self._set_state(error=str(exc), status="Error")
        finally:
            ser.close()
            self._set_state(running=False, connected=False, status="Stopped")
