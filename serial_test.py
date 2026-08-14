"""Quick check: does the board print anything on COM5?"""

import sys
import time

import numpy as np
import serial
import serial.tools.list_ports

from serial_io import read_frame

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"

print("Available ports:")
for p in serial.tools.list_ports.comports():
    print(f"  {p.device} — {p.description}")

print(f"\nOpening {PORT} (board resets when port opens)...")
print(">>> CLAP or talk near the mic during the first 5 seconds <<<\n")
ser = serial.Serial(PORT, 115200, timeout=0.2)

deadline = time.time() + 25.0
total_bytes = 0
binary_bytes = 0
diag_lines: list[str] = []
firmware_line = ""

while time.time() < deadline:
    waiting = ser.in_waiting
    if waiting:
        chunk = ser.read(waiting)
        total_bytes += len(chunk)

        # Pull printable boot/diag lines out of mixed text+binary chunks.
        text = chunk.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.isprintable():
                continue
            if any(
                line.startswith(p)
                for p in ("BOOT", "CARE_", "DIAG", "WARN", "ERROR", "STREAM_")
            ):
                print(f"  {line}")
                if line.startswith("CARE_"):
                    firmware_line = line
                if line.startswith("DIAG") or line.startswith("WARN"):
                    diag_lines.append(line)
            elif len(line) < 80:
                print(f"  {line}")

        if b"\xa5Z" in chunk:
            binary_bytes += chunk.count(b"\xa5Z") * 644  # rough frame size
    else:
        time.sleep(0.05)

frame_count = 0
max_rms = 0.0
max_peak = 0
all_zero_frames = 0
frame_deadline = time.time() + 8.0
print("\nReading audio frames (clap/talk now)...")
while time.time() < frame_deadline and frame_count < 20:
    frame = read_frame(ser)
    if frame is None:
        time.sleep(0.05)
        continue
    frame_count += 1
    peak = int(np.max(np.abs(frame)))
    max_peak = max(max_peak, peak)
    if peak == 0:
        all_zero_frames += 1
    wf = frame.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(wf**2)))
    max_rms = max(max_rms, rms)
    print(f"FRAME {frame_count}: {len(frame)} samples, peak={peak}, rms={rms:.4f}")

ser.close()
print(f"\nDone. Received {total_bytes} bytes (~{binary_bytes} binary est.).")
if firmware_line:
    print(f"Firmware: {firmware_line}")
if frame_count:
    print(
        f"Decoded {frame_count} audio frame(s), max peak={max_peak}, max rms={max_rms:.4f}, "
        f"{all_zero_frames}/{frame_count} frames all-zero"
    )

if diag_lines:
    print("\nMic diagnostics from firmware:")
    for line in diag_lines:
        print(f"  {line}")

if "builtin" in firmware_line and "inmp441" not in firmware_line:
    print(
        "\nNOTE: Built-in mic firmware is running — it ignores the external INMP441."
    )
    print("  Upload external mic firmware:")
    print("  .\\upload_inmp441.bat")

if frame_count and max_rms >= 0.01:
    print("\nMic looks good — run .\\run_live_ui.bat or .\\run_serial_bridge.bat")
elif frame_count and max_rms < 0.001:
    print("\nMic is streaming but still silent (all-zero or near-zero PCM).")
    print("Most likely causes after soldering:")
    print("  1. Wrong firmware — need inmp441 env for external mic (see above)")
    print("  2. INMP441 pin order — verify VDD/GND/SD/SCK/WS/L-R on your board")
    print("  3. L/R pin — try 3V3 instead of GND (or vice versa)")
    print("  4. SCK & WS swapped — try swapping D1 and D2 wires")
    print("  5. Cold solder on SD (data) or VDD — reflow with multimeter check")
    print("\nNext step — full wiring sweep (text only, no Python stream):")
    print("  .\\upload_mic_debug.bat")
    print("  .\\run_mic_debug.bat")
elif total_bytes == 0:
    print("\nNo data at all. Try re-upload firmware or a different USB cable.")
