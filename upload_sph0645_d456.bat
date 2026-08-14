@echo off
cd /d "%~dp0"
where pio >nul 2>&1
if errorlevel 1 (
  echo PlatformIO 'pio' not found in PATH. Run: pip install platformio
  exit /b 1
)
echo Uploading SPH0645 firmware — wiring D4=BCLK, D5=LRCL, D6=DOUT
pio run -d firmware -e sph0645_d456 -t upload
exit /b %ERRORLEVEL%
