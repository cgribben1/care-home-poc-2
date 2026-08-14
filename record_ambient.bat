@echo off
REM Record 60s of room ambient noise for calibration + training
cd /d "%~dp0"
call venv\Scripts\activate.bat
python record_ambient.py --port COM5 --seconds 60 %*
