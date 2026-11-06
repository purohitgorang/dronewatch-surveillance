@echo off
title DroneWatch Launcher
cd /d "%~dp0"
echo ============================================================
echo           DRONEWATCH FULL-STACK LAUNCHER
echo ============================================================
echo.
echo [1/2] Starting DroneWatch FastAPI Backend on http://localhost:8000 ...
start "DroneWatch Backend" cmd /k "cd /d "%~dp0backend" && uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting DroneWatch Vite Frontend on http://localhost:5173 ...
start "DroneWatch Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both services are now running!
echo Frontend: http://localhost:5173
echo Backend API Docs: http://localhost:8000/docs
echo.
pause
