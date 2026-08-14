"""Dump raw serial bytes — use when boot_log shows nothing."""

from __future__ import annotations

import sys
import time

import serial
import serial.tools.list_ports

from serial_io import open_device_serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"
WAIT_SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

print("Available ports:")
for p in serial.tools.list_ports.comports():
    print(f"  {p.device} — {p.description}")

print(f"\nOpening {PORT} for {WAIT_SEC:.0f}s...")
print("Resetting board and waiting for USB serial...\n")

ser = open_device_serial(PORT, 115200, timeout=0.2)
time.sleep(1.0)

deadline = time.time() + WAIT_SEC
total = 0
raw = bytearray()

while time.time() < deadline:
    if ser.in_waiting:
        chunk = ser.read(ser.in_waiting)
        total += len(chunk)
        raw.extend(chunk)
        # Show printable fragments immediately.
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                print(f"TEXT: {line}")
    else:
        time.sleep(0.05)

ser.close()

print(f"\n--- Total bytes received: {total} ---")
if total == 0:
    print("NOTHING from the board. Likely causes:")
    print("  1. INMP441 wiring shorting 3V3 to GND — unplug ALL mic wires and retry")
    print("  2. Bad USB cable / try another USB port")
    print("  3. Board not booting — check if COM5 disappears in Device Manager")
    print("  4. Re-upload: .\\upload_inmp441.bat  then run this again")
elif total < 50:
    print(f"First bytes (hex): {raw[:64].hex(' ')}")
else:
    preview = raw[:200].decode("utf-8", errors="replace")
    print("Start of stream:")
    print(preview)
