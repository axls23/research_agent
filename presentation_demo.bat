@echo off
setlocal EnableExtensions EnableDelayedExpansion

title Research Agent Presentation Demo
cd /d "%~dp0"

echo =======================================
echo   Research Agent Presentation Demo
echo =======================================
echo.

call :check_python || goto :fail

echo Preparing clean demo ports...
call :free_port 8000
call :free_port 3000
call :free_port 8080

echo [1/4] Ensuring Ollama server is live...
call :ensure_ollama || goto :fail

echo [2/4] Starting backend API on http://localhost:8000 ...
start "Research Backend API" cmd /k "cd /d ""%~dp0"" && python api.py"
call :wait_http "http://localhost:8000/health" 45 || goto :fail_backend

echo [3/4] Starting frontend ...
start "Research Frontend" cmd /k "cd /d ""%~dp0"" && run_frontend.bat"

set "FRONTEND_URL=http://localhost:3000"
call :wait_http "%FRONTEND_URL%" 90
if errorlevel 1 (
  set "FRONTEND_URL=http://localhost:8080/index.html"
  call :wait_http "%FRONTEND_URL%" 30 || goto :fail_frontend
)

echo [4/4] Opening presentation pages...
if /I "%FRONTEND_URL%"=="http://localhost:3000" (
  start "" "http://localhost:3000/backend"
  start "" "http://localhost:3000/chat"
  start "" "http://localhost:3000/workflow"
) else (
  start "" "%FRONTEND_URL%"
)

echo.
echo Demo is running.
echo - Backend monitor: http://localhost:3000/backend
echo - Research chat:   http://localhost:3000/chat
echo - Workflow page:   http://localhost:3000/workflow
echo.
echo Keep launched terminal windows open during the demo.
echo.
goto :end

:check_python
python --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python is not installed or not in PATH.
  exit /b 1
)
exit /b 0

:ensure_ollama
call :wait_http "http://localhost:11434/api/tags" 1
if not errorlevel 1 (
  echo Ollama is already running.
  exit /b 0
)

where ollama >nul 2>&1
if errorlevel 1 (
  echo ERROR: Ollama CLI was not found in PATH.
  echo Install Ollama or start it manually, then re-run this demo launcher.
  exit /b 1
)

echo Starting Ollama server...
start "Ollama Server" cmd /k "ollama serve"
call :wait_http "http://localhost:11434/api/tags" 40
if errorlevel 1 (
  echo ERROR: Ollama did not become ready on http://localhost:11434.
  exit /b 1
)
echo Ollama is live.
exit /b 0

:wait_http
set "URL=%~1"
set "MAX_TRIES=%~2"
if "%MAX_TRIES%"=="" set "MAX_TRIES=30"

set /a TRY=0
:wait_loop
set /a TRY+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%URL%' -Method GET -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
  echo Ready: %URL%
  exit /b 0
)
if %TRY% GEQ %MAX_TRIES% (
  echo Timeout waiting for %URL%
  exit /b 1
)
timeout /t 1 >nul
goto :wait_loop

:free_port
set "PORT=%~1"
if "%PORT%"=="" exit /b 0
powershell -NoProfile -Command "$conns = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; foreach ($c in $conns) { try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop } catch {} }" >nul 2>&1
exit /b 0

:fail_backend
echo.
echo ERROR: Backend did not start correctly.
goto :fail

:fail_frontend
echo.
echo ERROR: Frontend did not start correctly.
goto :fail

:fail
echo.
echo Presentation demo launcher failed.
echo Check the spawned terminal windows for details.
exit /b 1

:end
exit /b 0
