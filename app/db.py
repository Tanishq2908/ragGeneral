"""Chat history in a SQL database, plain SQL, no ORM.

Local default: SQLite file data/chat_history.db.
Deployed: set DATABASE_URL=postgresql://... (e.g. Render Postgres) and the same queries run on Postgres.
"""
import json
import os
import sqlite3
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "")
IS_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))
DB_PATH = Path(os.getenv("CHAT_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "chat_history.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          {pk},
    title       TEXT    NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id          {pk},
    chat_id     INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    mode        TEXT,
    sources     TEXT,               -- JSON list of retrieved chunks
    latency_ms  INTEGER,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
""".format(pk="SERIAL PRIMARY KEY" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT")


class _Conn:
    """Tiny wrapper so the same '?'-style SQL works on SQLite and Postgres."""

    def __init__(self):
        if IS_POSTGRES:
            import psycopg
            from psycopg.rows import dict_row
            self.conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        else:
            self.conn = sqlite3.connect(DB_PATH)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")

    def execute(self, sql, params=()):
        if IS_POSTGRES:
            sql = sql.replace("?", "%s")
        return self.conn.execute(sql, params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()


def connect():
    return _Conn()


def _row(r):
    d = dict(r)
    if d.get("created_at") is not None:
        d["created_at"] = str(d["created_at"])
    return d


def init_db():
    if not IS_POSTGRES:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        for stmt in filter(str.strip, SCHEMA.split(";")):
            c.execute(stmt)


def create_chat(title):
    with connect() as c:
        return c.execute("INSERT INTO chats (title) VALUES (?) RETURNING id", (title[:80],)).fetchone()["id"]


def list_chats():
    with connect() as c:
        return [_row(r) for r in c.execute("SELECT id, title, created_at FROM chats ORDER BY id DESC").fetchall()]


def chat_exists(chat_id):
    with connect() as c:
        return c.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone() is not None


def delete_chat(chat_id):
    with connect() as c:
        c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def add_message(chat_id, role, content, mode=None, sources=None, latency_ms=None):
    with connect() as c:
        c.execute(
            "INSERT INTO messages (chat_id, role, content, mode, sources, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, mode, json.dumps(sources) if sources is not None else None, latency_ms))


def get_messages(chat_id):
    with connect() as c:
        rows = c.execute("SELECT role, content, mode, sources, latency_ms, created_at FROM messages "
                         "WHERE chat_id = ? ORDER BY id", (chat_id,)).fetchall()
        return [dict(_row(r), sources=json.loads(r["sources"]) if r["sources"] else []) for r in rows]
