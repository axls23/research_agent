@echo off
setlocal EnableExtensions EnableDelayedExpansion

title Research Agent Launcher
cd /d "%~dp0"

echo =======================================
echo   Research Agent - Launch All Servers
echo =======================================
echo.

call :check_python || goto :fail

echo [1/3] Checking local LLM server (llama.cpp, http://127.0.0.1:8001)...
call :ensure_llamacpp

echo.
echo [2/3] Starting backend API on http://localhost:8000 ...
call :free_port 8000
start "Research Backend API" cmd /k "cd /d "%~dp0" && python "%~dp0api.py""
call :wait_http "http://localhost:8000/health" 45 || goto :fail_backend

echo.
echo [3/3] Starting frontend on http://localhost:3000 ...
call :free_port 3000
start "Research Frontend" cmd /k "cd /d "%~dp0" && "%~dp0run_frontend.bat""
call :wait_http "http://localhost:3000" 90 || goto :fail_frontend

echo.
echo =======================================
echo   All servers launched
echo =======================================
echo   Backend:  http://localhost:8000  (health: /health)
echo   Frontend: http://localhost:3000  (redirects to /workspace)
echo.
echo   Keep the spawned terminal windows open.
echo   NEO4J_PASSWORD / GROQ_API_KEY must be set in your environment
echo   if the run needs Neo4j or Groq - see SYSTEM_STATUS.md.
echo =======================================
goto :end

:check_python
python --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python is not installed or not in PATH.
  exit /b 1
)
exit /b 0

:ensure_llamacpp
set "LLAMA_EXE=C:\dev\alfred\src\local_storage_steward\intelligence\inference\llama.cpp-bin\llama-server.exe"
set "LLAMA_MODEL=C:\models\gemma-4-E2B-it-Q4_K_M.gguf"

call :wait_http "http://127.0.0.1:8001/v1/models" 1
if not errorlevel 1 (
  echo llama.cpp server is already running on :8001.
  exit /b 0
)

if not exist "%LLAMA_EXE%" (
  echo WARNING: llama-server.exe not found at %LLAMA_EXE%.
  echo Start your LLM backend manually ^(see config/config.yaml "llm.provider"^).
  exit /b 0
)
if not exist "%LLAMA_MODEL%" (
  echo WARNING: model not found at %LLAMA_MODEL%.
  echo Start your LLM backend manually ^(see config/config.yaml "llm.provider"^).
  exit /b 0
)

echo Starting llama.cpp server natively ^(Vulkan GPU, no WSL^) on :8001...
start "llama.cpp Server" cmd /k ""%LLAMA_EXE%" -m "%LLAMA_MODEL%" --port 8001 --n-gpu-layers 99"
call :wait_http "http://127.0.0.1:8001/v1/models" 60
if errorlevel 1 (
  echo WARNING: llama.cpp server did not become ready on :8001 within the timeout.
  echo Check the spawned window for errors, or start your LLM backend manually.
)
exit /b 0

:wait_http
set "URL=%~1"
set "MAX_TRIES=%~2"
if "%MAX_TRIES%"=="" set "MAX_TRIES=30"

set /a TRY=0
:wait_loop
set /a TRY+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%URL%' -Method GET -UseBasicParsing -TimeoutSec 6; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
  echo Ready: %URL%
  exit /b 0
)
if %TRY% GEQ %MAX_TRIES% (
  echo Timeout waiting for %URL%
  exit /b 1
)
"%SystemRoot%\System32\timeout.exe" /t 1 >nul
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
echo Launch failed. Check the spawned terminal windows for details.
exit /b 1

:end
exit /b 0
