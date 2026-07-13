#!/usr/bin/env bash
# Bash port of presentation_demo.bat for Fedora (this machine's local Ollama/venv setup).
# See CLAUDE.md for why paths point at /run/media/heytanix/Shared instead of system defaults.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_BIN="/run/media/heytanix/Shared/Ollama/bin/ollama"
OLLAMA_MODELS_DIR="/run/media/heytanix/Shared/Ollama/models"
OLLAMA_LOG_DIR="/run/media/heytanix/Shared/Ollama/logs"
OLLAMA_HOST_ADDR="127.0.0.1:11434"

echo "======================================="
echo "  Research Agent Presentation Demo"
echo "======================================="
echo

fail() {
  echo
  echo "ERROR: $1"
  echo "Presentation demo launcher failed."
  exit 1
}

wait_http() {
  local url="$1"
  local max_tries="${2:-30}"
  local try=0
  while (( try < max_tries )); do
    if curl -sf -o /dev/null --max-time 2 "$url"; then
      echo "Ready: $url"
      return 0
    fi
    ((try++))
    sleep 1
  done
  echo "Timeout waiting for $url"
  return 1
}

free_port() {
  local port="$1"
  [ -z "$port" ] && return 0
  local pids=""
  if command -v fuser >/dev/null 2>&1; then
    pids=$(fuser "${port}/tcp" 2>/dev/null)
  elif command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -t -i ":${port}" 2>/dev/null)
  fi
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  return 0
}

ensure_ollama() {
  if wait_http "http://localhost:11434/api/tags" 1; then
    echo "Ollama is already running."
    return 0
  fi

  if [ ! -x "$OLLAMA_BIN" ]; then
    echo "ERROR: Ollama binary not found at $OLLAMA_BIN"
    return 1
  fi

  echo "Starting Ollama server..."
  mkdir -p "$OLLAMA_LOG_DIR"
  OLLAMA_MODELS="$OLLAMA_MODELS_DIR" OLLAMA_HOST="$OLLAMA_HOST_ADDR" \
    nohup "$OLLAMA_BIN" serve > "$OLLAMA_LOG_DIR/serve.log" 2>&1 &
  disown

  wait_http "http://localhost:11434/api/tags" 40 || {
    echo "ERROR: Ollama did not become ready on http://localhost:11434."
    return 1
  }
  echo "Ollama is live."
}

command -v python3 >/dev/null 2>&1 || fail "Python 3 is not installed or not in PATH."
[ -f ".venv/bin/activate" ] || fail ".venv not found — run: ~/.local/bin/python3.10 -m venv .venv"

echo "Preparing clean demo ports..."
free_port 8000
free_port 3000

echo "[1/4] Ensuring Ollama server is live..."
ensure_ollama || fail "Ollama did not start."

echo "[2/4] Starting backend API on http://localhost:8000 ..."
# shellcheck disable=SC1091
source .venv/bin/activate
nohup python api.py > "$OLLAMA_LOG_DIR/api.log" 2>&1 &
disown
wait_http "http://localhost:8000/health" 45 || fail "Backend did not start correctly. Check $OLLAMA_LOG_DIR/api.log"

echo "[3/4] Starting frontend ..."
(cd frontend-next && nohup npm run dev > "$OLLAMA_LOG_DIR/frontend.log" 2>&1 & disown)

FRONTEND_URL="http://localhost:3000"
wait_http "$FRONTEND_URL" 90 || fail "Frontend did not start correctly. Check $OLLAMA_LOG_DIR/frontend.log"

echo "[4/4] Opening presentation pages..."
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:3000/backend" >/dev/null 2>&1 &
  xdg-open "http://localhost:3000/chat" >/dev/null 2>&1 &
  xdg-open "http://localhost:3000/workflow" >/dev/null 2>&1 &
else
  echo "xdg-open not found — open these manually:"
fi

echo
echo "Demo is running."
echo "- Backend monitor: http://localhost:3000/backend"
echo "- Research chat:   http://localhost:3000/chat"
echo "- Workflow page:   http://localhost:3000/workflow"
echo
echo "Logs: $OLLAMA_LOG_DIR/{serve,api,frontend}.log"
echo "Stop servers with: pkill -f 'ollama serve'; pkill -f 'python api.py'; pkill -f 'npm run dev'"
echo
