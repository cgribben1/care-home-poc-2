@echo off
cd /d "%~dp0"
echo.
echo INMP441 wiring debug (text-only firmware).
echo.
echo Step 1 — upload mic_debug (once):
echo   .\upload_mic_debug.bat
echo.
echo Step 2 — this opens serial monitor. Clap when prompted.
echo   Close live_ui / serial_bridge first.
echo.
pause
set PIO=%USERPROFILE%\.platformio\penv\Scripts\platformio.exe
"%PIO%" device monitor -d firmware -e mic_debug
