"""HTTP service.

    uvicorn app.api:app --reload
    open http://127.0.0.1:8000

Endpoints
    POST /chat          one turn; session_id carries the conversation
    GET  /health        capability + corpus report, for readiness checks
    GET  /escalations   the human-handoff queue
    POST /ingest        re-index the knowledge base (freshness path)
    GET  /sessions/{id} full transcript with the decision trail
    GET  /              the demo chat UI

The API returns the diagnostics alongside the reply rather than hiding them.
An answer you cannot inspect is an answer you cannot trust, and the UI uses
the same payload a reviewer sees in curl.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.ingest import run_ingest
from app.pipeline import SupportAssistant

WEB_DIR = Path(__file__).parent / "web"

_assistant: SupportAssistant | None = None


def get_assistant() -> SupportAssistant:
    global _assistant
    if _assistant is None:
        _assistant = SupportAssistant()
    return _assistant


@asynccontextmanager
async def lifespan(app: FastAPI):
    assistant = get_assistant()
    # Self-heal an empty database rather than serving confident nonsense from
    # a corpus that was never indexed.
    if assistant.store.stats()["documents"] == 0:
        run_ingest(store=assistant.store, verbose=False)
    yield
    assistant.close()


app = FastAPI(
    title="LearnForge Support Assistant",
    version="0.1.0",
    description="Retrieval-augmented support assistant with verification and human escalation.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    include_diagnostics: bool = True


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    action: str
    confidence: float
    intent: str
    citations: list[str]
    sources: list[dict[str, Any]]
    reason_codes: list[str]
    caveats: list[str]
    escalation_id: str | None = None
    queue: str | None = None
    priority: str | None = None
    standalone_query: str
    latency_ms: int
    diagnostics: dict[str, Any] | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = get_assistant().ask(request.message, session_id=request.session_id)
    payload = result.as_dict()
    if not request.include_diagnostics:
        payload.pop("diagnostics", None)
    payload.pop("question", None)
    return ChatResponse(**payload)


@app.get("/health")
def health() -> dict[str, Any]:
    assistant = get_assistant()
    stats = assistant.store.stats()
    return {
        "status": "ok" if stats["documents"] else "empty_index",
        "capabilities": {
            "generation": assistant.llm.model if assistant.llm.is_live else "offline-extractive",
            "generation_live": assistant.llm.is_live,
            "embeddings": assistant.embedder.model if assistant.embedder.is_live else None,
            "embeddings_live": assistant.embedder.is_live,
            "retrieval_mode": "hybrid" if assistant.embedder.is_live else "lexical-only",
            "verifier": settings.enable_verifier and assistant.llm.is_live,
            "reranker": settings.enable_llm_rerank and assistant.llm.is_live,
        },
        "corpus": stats,
        "thresholds": {
            "min_dense_score": settings.min_dense_score,
            "min_lexical_score": settings.min_lexical_score,
            "min_score_margin": settings.min_score_margin,
            "min_groundedness": settings.min_groundedness,
            "stale_after_days": settings.stale_after_days,
        },
    }


@app.get("/escalations")
def escalations(limit: int = 50) -> dict[str, Any]:
    rows = get_assistant().store.list_escalations(limit)
    for row in rows:
        row["reason_codes"] = json.loads(row["reason_codes"] or "[]")
        row["context_snapshot"] = json.loads(row["context_snapshot"] or "{}")
    return {"count": len(rows), "escalations": rows}


@app.post("/ingest")
def ingest(force: bool = False) -> dict[str, Any]:
    """Re-index the corpus.

    The freshness story in miniature: ingest is idempotent (content-hashed),
    so this is safe to call on a schedule or from a webhook when a help-centre
    article changes. Only changed documents are re-chunked and re-embedded.
    """
    result = run_ingest(store=get_assistant().store, force=force, verbose=False)
    return result


@app.get("/sessions/{session_id}")
def session(session_id: str) -> dict[str, Any]:
    assistant = get_assistant()
    rows = assistant.store.conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY message_id", (session_id,)
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="unknown session")
    conversation = assistant.store.conn.execute(
        "SELECT * FROM conversations WHERE session_id = ?", (session_id,)
    ).fetchone()
    return {
        "session": dict(conversation) if conversation else None,
        "messages": [dict(r) for r in rows],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")
