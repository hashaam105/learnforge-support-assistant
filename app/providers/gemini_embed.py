"""Google Gemini embeddings adapter.

Groq does not expose an embedding endpoint, so dense retrieval borrows
Gemini's free-tier embedding model. The two providers are independent: losing
either one degrades the system rather than breaking it (no Gemini key =>
BM25-only retrieval; no Groq key => extractive answers).

Gemini supports asymmetric embedding via task_type: indexing a passage and
embedding a question are different jobs, and telling the model which one it is
measurably improves recall on short support queries.
"""

from __future__ import annotations

import math
import time

import httpx

from app.config import Settings, settings

from .base import Embedder, LLMError

BASE = "https://generativelanguage.googleapis.com/v1beta"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4
_BATCH = 50  # batchEmbedContents accepts up to 100; 50 keeps payloads small


class GeminiEmbedder(Embedder):
    name = "gemini"

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self.model = self.cfg.embedding_model
        self.dim = self.cfg.embedding_dim
        if not self.cfg.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self._client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))

    # ------------------------------------------------------------------
    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            out.extend(self._embed_batch(texts[i : i + _BATCH], task))
        return out

    # ------------------------------------------------------------------
    def _embed_batch(self, batch: list[str], task: str) -> list[list[float]]:
        url = f"{BASE}/models/{self.model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task,
                    # gemini-embedding-001 defaults to 3072 dimensions. 768 is
                    # ample for a corpus this size and keeps a vector at 3 KB
                    # instead of 12 KB. Truncated vectors are no longer unit
                    # length, which is why _normalise below is not optional.
                    "outputDimensionality": self.dim,
                }
                for t in batch
            ]
        }
        params = {"key": self.cfg.gemini_api_key}

        last_error = "unknown"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = self._client.post(url, json=payload, params=params)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                if attempt == _MAX_ATTEMPTS:
                    break
                time.sleep(min(2**attempt, 8))
                continue

            if resp.status_code == 200:
                data = resp.json()
                vectors = [e["values"] for e in data.get("embeddings", [])]
                if len(vectors) != len(batch):
                    raise LLMError(
                        f"Gemini returned {len(vectors)} embeddings for {len(batch)} inputs"
                    )
                return [_normalise(v) for v in vectors]

            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 8))
                continue
            break

        raise LLMError(f"Gemini embedding failed after {_MAX_ATTEMPTS} attempts — {last_error}")

    def close(self) -> None:
        self._client.close()


def _normalise(vec: list[float]) -> list[float]:
    """L2-normalise so cosine similarity reduces to a dot product at query time."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]
