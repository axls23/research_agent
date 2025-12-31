"""
Frontend Runner
Quick script to run the frontend with a simple HTTP server
"""

import http.server
import socketserver
import os
import webbrowser
import threading
import time
from pathlib import Path

# Configuration
PORT = 8080
DIRECTORY = "frontend"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add CORS headers for API calls
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def run_server():
    """Run the HTTP server"""
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}/")
        print("Serving files from: " + os.path.abspath(DIRECTORY))
        httpd.serve_forever()

def open_browser():
    """Open the browser after a short delay"""
    time.sleep(2)  # Wait for server to start
    url = f"http://localhost:{PORT}/index.html"
    print(f"Opening browser at: {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════╗
    ║     Research Agent Frontend Server    ║
    ╚═══════════════════════════════════════╝
    """)
    
    # Check if frontend directory exists
    if not os.path.exists(DIRECTORY):
        print(f"Error: Frontend directory '{DIRECTORY}' not found!")
        exit(1)
    
    # Check if index.html exists
    index_path = os.path.join(DIRECTORY, "index.html")
    if not os.path.exists(index_path):
        print(f"Error: index.html not found in '{DIRECTORY}'!")
        exit(1)
    
    print(f"\nStarting frontend server on port {PORT}...")
    print("Press Ctrl+C to stop the server\n")
    
    # Start browser opener in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Run the server
        run_server()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"Error: {e}")
