@echo off
REM Hybrid = YAMNet + classifier must agree (safest for live ESP32 stream)
cd /d "%~dp0"
call venv\Scripts\activate.bat
python serial_bridge.py --port COM5 --mode hybrid %*
