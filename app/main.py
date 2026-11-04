"""FastAPI backend.  Run from the repo root:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import db

engine = None


@asynccontextmanager
async def lifespan(app):
    global engine
    db.init_db()
    from .rag_engine import RagEngine   # heavy import: loads models and indexes the PDFs once
    engine = RagEngine()
    yield


app = FastAPI(title="RAG Chat", lifespan=lifespan)
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"   # same page Vercel serves


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_id: int | None = None          # omit to start a new chat
    mode: str = "none"                  # none | rewrite | multi_query | hyde


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {"ok": engine is not None, "chunks": len(engine.chunks) if engine else 0}


@app.get("/api/chats")
def chats():
    return db.list_chats()


@app.get("/api/chats/{chat_id}/messages")
def messages(chat_id: int):
    if not db.chat_exists(chat_id):
        raise HTTPException(404, "chat not found")
    return db.get_messages(chat_id)


@app.delete("/api/chats/{chat_id}")
def delete(chat_id: int):
    db.delete_chat(chat_id)
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest):
    from .rag_engine import MODES
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    chat_id = req.chat_id
    if chat_id is None:
        chat_id = db.create_chat(req.question)
    elif not db.chat_exists(chat_id):
        raise HTTPException(404, "chat not found")

    history = db.get_messages(chat_id)
    db.add_message(chat_id, "user", req.question, mode=req.mode)
    try:
        result = engine.answer(req.question, req.mode, history)
    except Exception as e:
        raise HTTPException(502, f"RAG pipeline failed: {e}")
    latency = result["retrieval_ms"] + result["generation_ms"]
    db.add_message(chat_id, "assistant", result["answer"], mode=req.mode, sources=result["sources"], latency_ms=latency)
    return {"chat_id": chat_id, **result, "latency_ms": latency}
