"""Frontend runner supporting Next.js (`frontend-next`) and legacy static frontend."""

import http.server
import os
import shutil
import socketserver
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
NEXT_DIR = ROOT / "frontend-next"
LEGACY_DIR = ROOT / "frontend"


def _open_browser_delayed(url: str, delay: float = 2.0) -> None:
    time.sleep(delay)
    print(f"Opening browser at: {url}")
    webbrowser.open(url)


def _run_next_frontend() -> int:
    npm = shutil.which("npm")
    if not npm:
        print("Error: npm was not found in PATH.")
        print("Install Node.js (includes npm) to run frontend-next.")
        return 1

    if not (NEXT_DIR / "package.json").exists():
        print("Error: frontend-next/package.json not found.")
        return 1

    port = os.getenv("FRONTEND_PORT", "3000")
    url = f"http://localhost:{port}"

    if not (NEXT_DIR / "node_modules").exists():
        print("Installing frontend-next dependencies (first run)...")
        install_rc = subprocess.call([npm, "install"], cwd=str(NEXT_DIR))
        if install_rc != 0:
            print("Error: npm install failed.")
            return install_rc

    print(f"Starting Next.js frontend on {url}")
    print("Press Ctrl+C to stop the server")

    browser_thread = threading.Thread(target=_open_browser_delayed, args=(url,))
    browser_thread.daemon = True
    browser_thread.start()

    return subprocess.call([npm, "run", "dev", "--", "--port", port], cwd=str(NEXT_DIR))


class _LegacyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(LEGACY_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()


def _run_legacy_frontend() -> int:
    if not LEGACY_DIR.exists():
        print("Error: neither frontend-next nor legacy frontend directory was found.")
        return 1

    port = int(os.getenv("LEGACY_FRONTEND_PORT", "8080"))
    url = f"http://localhost:{port}/index.html"
    print(f"Starting legacy static frontend on {url}")
    print("Press Ctrl+C to stop the server")

    browser_thread = threading.Thread(target=_open_browser_delayed, args=(url,))
    browser_thread.daemon = True
    browser_thread.start()

    try:
        with socketserver.TCPServer(("", port), _LegacyHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    print("=======================================")
    print("    Research Agent Frontend Server")
    print("=======================================")

    try:
        if NEXT_DIR.exists():
            raise SystemExit(_run_next_frontend())
        raise SystemExit(_run_legacy_frontend())
    except KeyboardInterrupt:
        print("\nServer stopped.")
