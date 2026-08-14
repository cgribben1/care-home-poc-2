"""Read firmware boot/diag text only (no binary frame parsing).

Use after upload or reset — clap during the first few seconds.
"""

from __future__ import annotations

import sys
import time

import serial
import serial.tools.list_ports

from serial_io import open_device_serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"
WAIT_SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

print("Available ports:")
for p in serial.tools.list_ports.comports():
    print(f"  {p.device} — {p.description}")

print(f"\nOpening {PORT} for {WAIT_SEC:.0f}s (board resets)...")
print(">>> CLAP / TALK NOW during boot diagnostics <<<\n")

ser = open_device_serial(PORT, 115200, timeout=0.2)
time.sleep(1.0)

deadline = time.time() + WAIT_SEC
buffer = ""
total_bytes = 0

while time.time() < deadline:
    if ser.in_waiting:
        chunk_bytes = ser.read(ser.in_waiting)
        total_bytes += len(chunk_bytes)
        chunk = chunk_bytes.decode("utf-8", errors="ignore")
        buffer += chunk
        # Print complete lines as they arrive (ignore binary garbage).
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.isprintable() or line.startswith("DIAG"):
                print(line)
            elif any(c in line for c in ("BOOT", "CARE_", "WARN", "ERROR", "STREAM")):
                print(repr(line))
            else:
                print(f"[raw] {line!r}")
    else:
        time.sleep(0.05)

ser.close()
print(f"\n--- Total bytes received: {total_bytes} ---")
if total_bytes == 0:
    print("NOTHING from board — try .\\run_serial_sniff.bat")
    print("If still zero: unplug ALL INMP441 wires, re-upload, retry (checks for 3V3/GND short).")
elif total_bytes > 0:
    print("If you saw WARN / maxL=0 maxR=0 → hardware or wiring.")
    print("If maxL or maxR is large (>10000) but live UI still zero → tell us the DIAG line.")
print("Next: .\\upload_mic_debug.bat  then  .\\run_mic_debug.bat")
