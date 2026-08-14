@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo Mic stream test — clap/talk when prompted.
echo Close live_ui / serial_bridge first (COM port must be free).
echo.
python serial_test.py %*
