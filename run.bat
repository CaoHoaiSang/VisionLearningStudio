@echo off
cd /d "%~dp0"
python modern_app.py
if errorlevel 1 (
  echo Cai thu vien bang lenh: python -m pip install -r requirements.txt
  pause
)
