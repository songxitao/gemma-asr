@echo off
chcp 936 >nul
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [¥ÌŒÛ] Œ¥’“µΩ–Èƒ‚ª∑æ≥: %~dp0.venv\Scripts\python.exe
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\evaluate.py" %*
