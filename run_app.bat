@echo off
title Drone Hawk X Launcher
cd /d "%~dp0"

set PYTHON_CMD=python
if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON_CMD="%~dp0.venv\Scripts\python.exe"
)

echo Starting Local Vision Backend...
start "Drone Hawk X Backend" cmd /k "cd backend && %PYTHON_CMD% -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo Starting Frontend...
start "Drone Hawk X Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Application started!
echo Frontend: http://localhost:5173
echo Backend: http://localhost:8000/docs
echo.
pause
