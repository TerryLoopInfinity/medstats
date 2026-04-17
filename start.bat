@echo off
echo Starting MedStats...

start "Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --reload --port 8000"
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
