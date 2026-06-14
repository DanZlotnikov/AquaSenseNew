@echo off
:: Run AquaSense inference using the installed virtual environment.
:: Usage:  run.bat --input "path\to\recording.csv"

if not exist venv\Scripts\python.exe (
    echo ERROR: Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

venv\Scripts\python run.py %*
