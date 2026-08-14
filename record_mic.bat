@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python record_serial.py --port COM5 --seconds 5 %*
