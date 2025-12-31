# Frontend Quick Start Guide

## Running the Frontend

There are several ways to run the Research Agent frontend:

### Option 1: Using Python Script (Recommended)
```bash
python run_frontend.py
```
This will:
- Start a local web server on port 8080
- Automatically open your browser to the frontend
- Serve the frontend files with proper CORS headers

### Option 2: Using Batch File (Windows)
Double-click `run_frontend.bat` or run in Command Prompt:
```cmd
run_frontend.bat
```

### Option 3: Using PowerShell (Windows)
```powershell
.\run_frontend.ps1
```

### Option 4: Using Python's Built-in Server
```bash
cd frontend
python -m http.server 8080
```
Then open http://localhost:8080/index.html in your browser.

### Option 5: Direct File Access
Simply open `frontend/index.html` in your browser. However, this may have limitations with API calls due to CORS restrictions.

## Frontend Features

Once the frontend is running, you can:

1. **Create Research Projects**
   - Enter project name and description
   - Select task type (Literature Review, Paper Construction, etc.)
   - Click "Start Research" to begin

2. **Monitor Progress**
   - Real-time progress updates
   - Token usage visualization
   - Status messages

3. **Construct Papers**
   - Enter paper title and abstract
   - Select citation style (APA, MLA, Chicago)
   - Choose output format (PDF, Word, LaTeX)
   - Click "Construct Paper" to generate

## Connecting to Backend

The frontend expects the Research Agent API server to be running on http://localhost:8000.

To run the complete system with backend:
```bash
python run_research_agent.py
```

This will start both the API server and open the frontend.

## Troubleshooting

### Port Already in Use
If port 8080 is already in use, edit `run_frontend.py` and change the PORT variable:
```python
PORT = 8081  # or any other available port
```

### CORS Issues
If you're having issues with API calls, make sure:
1. The API server is running on http://localhost:8000
2. You're using one of the server-based methods (not direct file access)

### Browser Not Opening
If the browser doesn't open automatically, manually navigate to:
http://localhost:8080/index.html

## Development

To modify the frontend:
1. Edit `frontend/index.html` for UI changes
2. Edit `frontend/api-bridge.js` for API integration changes
3. Refresh your browser to see changes (no restart needed)

The frontend uses:
- TailwindCSS for styling
- Font Awesome for icons
- Vanilla JavaScript for functionality
- WebSocket for real-time updates
