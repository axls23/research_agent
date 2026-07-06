# Frontend Quick Start Guide

The frontend is a Next.js app living in `frontend-next/`. It runs on http://localhost:3000 and talks to the backend API on http://localhost:8000.

## Running the Frontend

### Option 1: Using Python Script (Recommended)
```bash
python run_frontend.py
```
This will:
- Install `frontend-next` dependencies on first run (`npm install`)
- Start the Next.js dev server on port 3000
- Automatically open your browser to the frontend

### Option 2: Using Batch File (Windows)
Double-click `run_frontend.bat` or run in Command Prompt:
```cmd
run_frontend.bat
```

### Option 3: Using PowerShell (Windows)
```powershell
.\run_frontend.ps1
```

### Option 4: Using npm Directly
```bash
cd frontend-next
npm install   # first run only
npm run dev
```
Then open http://localhost:3000 in your browser.

## Frontend Pages

- **/** — landing page
- **/chat** — research chat, streams responses from the backend (`POST /api/chat/stream`)
- **/backend** — live backend monitor (`GET /api/backend/monitor`)
- **/workflow** — workflow overview
- **/upload** — document upload UI
- **/about** — project info

## Connecting to Backend

The frontend expects the Research Agent API server to be running on http://localhost:8000. Start it with:
```bash
python api.py
```

To point the frontend at a different backend, set `NEXT_PUBLIC_API_BASE_URL` before starting the dev server.

For a full demo (Ollama + backend + frontend), use:
```cmd
presentation_demo.bat
```

## Troubleshooting

### Port Already in Use
If port 3000 is already in use, set a different port before running:
```bash
set FRONTEND_PORT=3001
python run_frontend.py
```

### "Failed to fetch" / CORS Issues
If API calls fail, make sure:
1. The API server is running on http://localhost:8000 (check http://localhost:8000/health)
2. You're accessing the frontend via http://localhost:3000 (the only origin allowed by the backend's CORS config in `api.py`)

### Browser Not Opening
If the browser doesn't open automatically, manually navigate to:
http://localhost:3000

## Development

To modify the frontend:
1. Pages live in `frontend-next/app/<route>/page.tsx`
2. Shared UI components are in `frontend-next/components/`
3. The dev server hot-reloads on save (no restart needed)

The frontend uses:
- Next.js (App Router) with TypeScript
- Tailwind CSS for styling
- Server-Sent Events for streaming chat responses
