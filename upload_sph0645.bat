@echo off
cd /d "%~dp0"
set PIO=%USERPROFILE%\.platformio\penv\Scripts\platformio.exe
if not exist "%PIO%" (
  echo PlatformIO not found at %PIO%
  exit /b 1
)
echo Uploading SPH0645 stream firmware (XIAO ESP32S3 Plus)...
echo Wiring: 3V/GND, SEL-GND or 3V3, DOUT-D0, BCLK-D1, LRCL-D2
"%PIO%" run -d firmware -e sph0645 -t upload
exit /b %ERRORLEVEL%
