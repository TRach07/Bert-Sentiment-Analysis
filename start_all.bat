@echo off
echo Starting Backend and Frontend...
echo.

start "Backend API" cmd /k "cd backend && python -m uvicorn app:app --reload --port 8000"

timeout /t 3 /nobreak >nul

start "" "frontend\index.html"

echo.
echo Backend started on http://localhost:8000
echo Frontend opened in browser
echo.
echo Press any key to close this window (backend will continue running)...
pause >nul

