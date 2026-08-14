@echo off
cd /d "%~dp0"
set PIO=%USERPROFILE%\.platformio\penv\Scripts\platformio.exe
if not exist "%PIO%" (
  echo PlatformIO not found at %PIO%
  exit /b 1
)
echo Uploading built-in mic stream firmware...
"%PIO%" run -d firmware -e builtin_stream -t upload
exit /b %ERRORLEVEL%
