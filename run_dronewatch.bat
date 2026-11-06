@echo off
title DroneWatch Edge AI Launcher
cd /d "%~dp0"
echo ============================================================
echo       DRONEWATCH - OFFLINE EDGE AI DRONE DETECTION
echo ============================================================
echo.
echo Mode: 100%% Local Edge AI (PyTorch CUDA / YOLOv8 - Zero Cloud API)
echo.

set PYTHON_CMD=python
if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON_CMD="%~dp0.venv\Scripts\python.exe"
)

echo [1/2] Starting DroneWatch Local Vision Backend on http://localhost:8000 ...
start "DroneWatch Backend" cmd /k "cd /d "%~dp0backend" && %PYTHON_CMD% -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting DroneWatch Vite Frontend on http://localhost:5173 ...
start "DroneWatch Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ============================================================
echo Services Launched Successfully!
echo Frontend UI:        http://localhost:5173
echo Backend API Docs:   http://localhost:8000/docs
echo System Health:      http://localhost:8000/status
echo ============================================================
echo.
pause
