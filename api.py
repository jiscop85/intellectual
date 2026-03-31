from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import VectorStore
from generator import answer_question


INDEX_DIR = Path("storage/index")

app = FastAPI(title="SmartDoc AI Assistant API", version="1.0.0")
store = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


@app.on_event("startup")
def startup_event():
    global store
    index_file = INDEX_DIR / "index.faiss"
    chunks_file = INDEX_DIR / "chunks.json"
    if index_file.exists() and chunks_file.exists():
        store = VectorStore.load(INDEX_DIR)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_ready": store is not None,
    }


@app.post("/ask")
def ask(req: AskRequest):
    if store is None:
        raise HTTPException(status_code=400, detail="Index is not ready.")

    hits = store.search(req.question, top_k=req.top_k)
    answer = answer_question(req.question, hits)

    return {
        "question": req.question,
        "answer": answer,
        "sources": hits,
    }