# Care Audio POC

Acoustic monitoring proof-of-concept for care-home settings: detect **falls**, **distress**, and **coughing** from live audio streamed by an ESP32.

## Hardware

| Component | Notes |
|-----------|--------|
| **Seeed XIAO ESP32S3 Plus** | Streams 16 kHz mono PCM over USB serial |
| **Adafruit SPH0645LM4H** | I2S MEMS microphone breakout |

### Wiring (SPH0645 → XIAO Plus)

| Mic pin | XIAO pin |
|---------|----------|
| 3V | 3V3 |
| GND | GND |
| SEL | GND (or 3V3 for right channel) |
| BCLK | D4 |
| LRCL | D5 |
| DOUT | D6 |

Alternative wiring (D0/D1/D2) is supported — see `upload_sph0645.bat`.

## Quick start

### 1. Python environment

```powershell
cd care-audio-poc
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Flash firmware

Close anything using the COM port, plug in USB, then:

```powershell
.\upload_sph0645_d456.bat    # D4/D5/D6 wiring
# or
.\upload_sph0645.bat         # D0/D1/D2 wiring
```

Update `upload_port` in `firmware/platformio.ini` if your COM port is not COM3.

### 3. Setup and test

```powershell
.\setup_and_test_d456.bat
```

Clap during the first 3 seconds. Check boot DIAG lines and `recording.wav`.

### 4. Live monitoring

```powershell
.\run_live_ui.bat            # Gradio two-tier UI
.\run_serial_bridge.bat      # CLI hybrid detection
```

## Architecture

```
ESP32 (mic) ──USB serial──► PC Python stack
                              │
                              ├─ Tier 1: energy gate (always on, no ML)
                              └─ Tier 2: YAMNet + trained classifier (on trigger)
```

- **Tier 1** — cheap listener (`tier1_gate.py`): RMS spikes vs calibrated room baseline  
- **Tier 2** — hybrid model (`analyze.py`): YAMNet embeddings + scikit-learn classifier  

In production, Tier 1 can move to ESP32 firmware; Tier 2 can run as TFLite on-device or on a gateway.

## Training (optional)

Requires downloaded datasets (ESC-50, etc.):

```powershell
python prepare_data.py
python train_classifier.py
.\record_ambient.bat         # room noise calibration
```

Artifacts go to `data/` and `models/` (gitignored — regenerate locally).

## Project layout

| Path | Purpose |
|------|---------|
| `firmware/` | PlatformIO — ESP32 stream firmware |
| `serial_bridge.py` | Live CLI detection from serial |
| `two_tier_monitor.py` | Tier 1 + Tier 2 pipeline |
| `live_ui.py` | Gradio test UI |
| `tier1_gate.py` | Always-on energy gate |
| `train_classifier.py` | Train embedding classifier |

## Debug tools

```powershell
.\run_boot_log.bat           # firmware boot / DIAG text
.\run_serial_sniff.bat       # raw byte dump
.\run_mic_test.bat           # quick RMS / frame check
```

## Requirements

- Python 3.10+
- [PlatformIO](https://platformio.org/) (for firmware upload)
- Windows tested; Python stack is cross-platform

## License

MIT
