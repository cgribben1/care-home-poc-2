@echo off
cd /d "%~dp0"
set PIO=%USERPROFILE%\.platformio\penv\Scripts\platformio.exe
if not exist "%PIO%" (
  echo PlatformIO not found at %PIO%
  echo Install PlatformIO or open the firmware folder in VS Code/Cursor with PlatformIO extension.
  exit /b 1
)
echo Uploading INMP441 stream firmware...
"%PIO%" run -d firmware -e inmp441 -t upload
exit /b %ERRORLEVEL%
