@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Raw serial dump — shows ANY bytes from the board.
echo If this stays empty with mic connected, try unplugging INMP441 wires and re-run.
echo.
python serial_sniff.py %*
