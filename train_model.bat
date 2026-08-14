@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo === Installing deps ===
pip install -q scikit-learn joblib
echo === Preparing training data (ESC-50 + synthetic falls) ===
python prepare_data.py
echo === Training embedding classifier ===
python train_classifier.py
echo === Evaluation ===
python eval_clips.py
echo.
echo Done. Run detection with: .\run_serial_bridge.bat
