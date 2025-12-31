@echo off
echo =======================================
echo     Research Agent Frontend Server
echo =======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Run the frontend server
echo Starting frontend server on http://localhost:8080
echo Press Ctrl+C to stop the server
echo.

python run_frontend.py

pause
