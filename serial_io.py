"""Serial frame parser for ESP32 PCM stream."""

from __future__ import annotations

import time

import numpy as np
import serial

FRAME_MAGIC = bytes([0xA5, 0x5A])


def open_device_serial(
    port: str, baud: int = 115200, timeout: float = 0.2
) -> serial.Serial:
    """Open ESP32-S3 USB CDC serial after reset/re-enumeration (Windows)."""
    # First open toggles DTR/RTS and resets the board; USB may drop briefly.
    probe = serial.Serial(port, baud, timeout=timeout)
    probe.close()
    time.sleep(2.5)
    return serial.Serial(port, baud, timeout=timeout)


def wait_for_stream(ser: serial.Serial, timeout_sec: float = 25.0) -> bool:
    """Wait for STREAM_BEGIN text or valid binary frames."""
    print("Listening for device (read immediately — don't miss boot text)...")
    print("  (INMP441 firmware runs ~10s of boot diagnostics before streaming)")
    deadline = time.time() + timeout_sec
    binary_hits = 0
    total_bytes = 0

    while time.time() < deadline:
        if ser.in_waiting:
            # Peek/consume: try line first if mostly printable pending
            chunk = ser.read(ser.in_waiting)
            total_bytes += len(chunk)

            # Check for banner text
            if b"STREAM_BEGIN" in chunk:
                print("  device: STREAM_BEGIN")
                print("Audio stream started.")
                return True
            if b"DIAG" in chunk or b"WARN" in chunk:
                try:
                    text = chunk.decode("utf-8", errors="ignore")
                    for line in text.splitlines():
                        line = line.strip()
                        if line.startswith("DIAG") or line.startswith("WARN"):
                            print(f"  device: {line}")
                except Exception:
                    pass
            if b"I2S init failed" in chunk or b"ERROR" in chunk:
                print("Firmware reported I2S/mic error — check INMP441 wiring.")
                return False

            # Count valid frame magics in binary data
            binary_hits += chunk.count(FRAME_MAGIC)
            if binary_hits >= 3:
                # Put chunk back isn't possible — parser will resync on stream
                print("  device: binary audio frames detected")
                print("Audio stream started.")
                return True

            # Show any readable banner fragments once
            try:
                text = chunk.decode("utf-8", errors="ignore")
                for line in text.splitlines():
                    line = line.strip()
                    if line and line.isprintable() and len(line) < 120:
                        print(f"  device: {line}")
            except Exception:
                pass
        else:
            time.sleep(0.05)

    print(f"Timed out waiting for audio stream ({total_bytes} bytes received).")
    if total_bytes == 0:
        print("  No data at all — check USB cable, COM port, and that board powers on.")
        print("  Re-upload stream firmware: .\\upload_inmp441.bat")
    elif total_bytes < 200:
        print("  Very little data — board may be stuck in ERROR loop; try .\\run_boot_log.bat")
    return False


def read_frame_body(ser: serial.Serial) -> np.ndarray | None:
    """Read count + PCM payload after FRAME_MAGIC has already been consumed."""
    count_bytes = _read_exact(ser, 2, timeout_sec=2.0)
    if count_bytes is None:
        return None

    count = int.from_bytes(count_bytes, "little")
    if count == 0 or count > 4096:
        return None

    payload = _read_exact(ser, count * 2, timeout_sec=3.0)
    if payload is None:
        return None

    return np.frombuffer(payload, dtype="<i2").copy()


def read_frame(ser: serial.Serial) -> np.ndarray | None:
    """Resync to the next frame magic, then read one complete PCM frame."""
    if not resync(ser):
        return None
    return read_frame_body(ser)


def _read_exact(ser: serial.Serial, n: int, timeout_sec: float = 5.0) -> bytes | None:
    buf = bytearray()
    deadline = time.time() + timeout_sec
    while len(buf) < n and time.time() < deadline:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
        else:
            time.sleep(0.002)
    if len(buf) != n:
        return None
    return bytes(buf)


def resync(ser: serial.Serial, limit: int = 16384) -> bool:
    """Scan forward for the next frame magic."""
    pending = bytearray()
    scanned = 0
    while scanned < limit:
        b = ser.read(1)
        if not b:
            return False
        scanned += 1
        pending.append(b[0])
        if len(pending) > 2:
            pending.pop(0)
        if pending == bytearray(FRAME_MAGIC):
            return True
    return False


def collect_pcm(
    ser: serial.Serial, target_samples: int, timeout_sec: float = 15.0
) -> np.ndarray | None:
    chunks: list[np.ndarray] = []
    total = 0
    deadline = time.time() + timeout_sec
    synced = False

    while total < target_samples and time.time() < deadline:
        if not synced:
            if not resync(ser):
                time.sleep(0.01)
                continue
            synced = True
            frame = read_frame_body(ser)
        else:
            # Frames are back-to-back: magic, count, payload, magic, ...
            header = _read_exact(ser, 2, timeout_sec=2.0)
            if header is None:
                synced = False
                continue
            if header != FRAME_MAGIC:
                synced = False
                continue
            frame = read_frame_body(ser)

        if frame is None:
            synced = False
            continue

        chunks.append(frame)
        total += len(frame)

    if total < target_samples:
        return None

    return np.concatenate(chunks)[:target_samples]
