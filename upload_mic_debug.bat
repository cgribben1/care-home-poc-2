@echo off
cd /d "%~dp0"
set PIO=%USERPROFILE%\.platformio\penv\Scripts\platformio.exe
if not exist "%PIO%" (
  echo PlatformIO not found at %PIO%
  exit /b 1
)
echo Uploading mic_debug wiring-test firmware...
"%PIO%" run -d firmware -e mic_debug -t upload
exit /b %ERRORLEVEL%
