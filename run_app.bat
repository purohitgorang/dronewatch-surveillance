@echo off
echo Starting Backend...
start "DroneWatch Backend" cmd /k "cd backend && uvicorn main:app --reload"

echo Starting Frontend...
start "DroneWatch Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Application started!
echo Please open http://localhost:5173 in your browser.
echo.
pause
