"""Chat history in a SQL database (SQLite file by default, plain SQL, no ORM)."""
import json
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("CHAT_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "chat_history.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    mode        TEXT,
    sources     TEXT,               -- JSON list of retrieved chunks
    latency_ms  INTEGER,
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
"""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def create_chat(title):
    with connect() as conn:
        return conn.execute("INSERT INTO chats (title) VALUES (?)", (title[:80],)).lastrowid


def list_chats():
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT id, title, created_at FROM chats ORDER BY id DESC")]


def chat_exists(chat_id):
    with connect() as conn:
        return conn.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone() is not None


def delete_chat(chat_id):
    with connect() as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def add_message(chat_id, role, content, mode=None, sources=None, latency_ms=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, mode, sources, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, mode, json.dumps(sources) if sources is not None else None, latency_ms))


def get_messages(chat_id):
    with connect() as conn:
        rows = conn.execute("SELECT role, content, mode, sources, latency_ms, created_at FROM messages "
                            "WHERE chat_id = ? ORDER BY id", (chat_id,))
        return [dict(r, sources=json.loads(r["sources"]) if r["sources"] else []) for r in rows]
