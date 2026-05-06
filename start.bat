@echo off
title Clinic AI - Launcher
color 0A

echo.
echo  ==========================================
echo    CLINIC AI - Starting up...
echo  ==========================================
echo.

:: Kill anything on ports 8000, 5173, 5174, 5175
echo [1/3] Clearing ports...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " 2^>nul') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5173 " 2^>nul') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5174 " 2^>nul') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5175 " 2^>nul') do taskkill /F /PID %%a >nul 2>&1
echo     Done.

:: Start backend in a new window
echo [2/3] Starting Backend (FastAPI)...
start "Clinic AI - Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

:: Wait for backend to be ready
echo     Waiting for backend...
timeout /t 4 /nobreak >nul

:: Start frontend in a new window
echo [3/3] Starting Frontend (React)...
start "Clinic AI - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

:: Wait for frontend to start
timeout /t 4 /nobreak >nul

:: Open the browser
echo.
echo  ==========================================
echo    Opening http://localhost:5173 ...
echo  ==========================================
echo.
start http://localhost:5173

echo  Both servers are running in separate windows.
echo  Close those windows to stop the app.
echo.
pause
