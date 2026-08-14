@echo off
cd /d "%~dp0"
echo === SPH0645 setup + test (D4=BCLK, D5=LRCL, D6=DOUT) ===
echo.
call upload_sph0645_d456.bat
if errorlevel 1 exit /b 1
echo.
call venv\Scripts\activate.bat
echo Running boot log — CLAP during first 3 seconds...
python boot_log.py COM5 35
echo.
echo Recording 5s test WAV...
python record_serial.py --port COM5 --seconds 5 -o recording.wav
exit /b %ERRORLEVEL%
