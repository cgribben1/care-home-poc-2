@echo off
REM Live detection from XIAO ESP32S3 on COM5
cd /d "%~dp0"
call venv\Scripts\activate.bat
python serial_bridge.py --port COM5 %*
