@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Close live_ui / serial_bridge first.
python boot_log.py %*
