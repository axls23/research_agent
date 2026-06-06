"""
core/job_queue.py
=================
SQLite-based task queue for asynchronous subagent execution.
"""

import sqlite3
import json
import time
import os
from typing import Any, Dict, Optional, Tuple

DB_PATH = "outputs/job_queue.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                result TEXT
            )
        ''')

def enqueue_job(agent_name: str, payload: Dict[str, Any]) -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        now = time.time()
        cursor.execute(
            "INSERT INTO jobs (agent_name, payload, status, created_at, updated_at) VALUES (?, ?, 'PENDING', ?, ?)",
            (agent_name, json.dumps(payload), now, now)
        )
        return cursor.lastrowid

def get_next_job(agent_name: str) -> Optional[Tuple[int, Dict[str, Any]]]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Find oldest pending job for this agent
        cursor.execute(
            "SELECT id, payload FROM jobs WHERE agent_name = ? AND status = 'PENDING' ORDER BY created_at ASC LIMIT 1",
            (agent_name,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        job_id, payload_str = row
        # Mark as in progress
        now = time.time()
        cursor.execute("UPDATE jobs SET status = 'IN_PROGRESS', updated_at = ? WHERE id = ?", (now, job_id))
        return job_id, json.loads(payload_str)

def complete_job(job_id: int, result: Dict[str, Any]):
    with sqlite3.connect(DB_PATH) as conn:
        now = time.time()
        conn.execute(
            "UPDATE jobs SET status = 'COMPLETED', result = ?, updated_at = ? WHERE id = ?",
            (json.dumps(result), now, job_id)
        )

def fail_job(job_id: int, error: str):
    with sqlite3.connect(DB_PATH) as conn:
        now = time.time()
        conn.execute(
            "UPDATE jobs SET status = 'FAILED', result = ?, updated_at = ? WHERE id = ?",
            (json.dumps({"error": error}), now, job_id)
        )

def get_job_status(job_id: int) -> Optional[str]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        return None
