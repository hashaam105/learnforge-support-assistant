"""Central configuration, loaded from environment / .env.

Every tunable the pipeline reads lives here so that a reviewer can find the
knobs in one place, and so the eval harness can override them deterministically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

try:  # optional dependency — the package still imports without it
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- generation ---
    groq_api_key: str = field(default_factory=lambda: _str("GROQ_API_KEY"))
    llm_provider: str = field(default_factory=lambda: _str("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: _str("LLM_MODEL", "openai/gpt-oss-120b"))
    llm_temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.1))
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 1200))

    # --- embeddings ---
    gemini_api_key: str = field(default_factory=lambda: _str("GEMINI_API_KEY"))
    embedding_provider: str = field(default_factory=lambda: _str("EMBEDDING_PROVIDER", "gemini"))
    embedding_model: str = field(default_factory=lambda: _str("EMBEDDING_MODEL", "gemini-embedding-001"))
    embedding_dim: int = field(default_factory=lambda: _int("EMBEDDING_DIM", 768))

    # --- storage ---
    db_path: str = field(default_factory=lambda: _str("DB_PATH", "learnforge.db"))
    kb_path: str = field(default_factory=lambda: _str("KB_PATH", "data/learnforge-knowledge-base"))

    # --- retrieval ---
    retrieval_candidates: int = field(default_factory=lambda: _int("RETRIEVAL_CANDIDATES", 24))
    retrieval_top_k: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K", 5))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    enable_llm_rerank: bool = field(default_factory=lambda: _bool("ENABLE_LLM_RERANK", True))

    # --- confidence gates ---
    # Two thresholds, because the two retrieval modes produce scores on
    # different scales. Cosine similarity from an embedding model does not sit
    # near zero for unrelated text (typically 0.3-0.5), so the dense gate is
    # high; the BM25 proxy is length-normalised and sits lower.
    min_dense_score: float = field(default_factory=lambda: _float("MIN_DENSE_SCORE", 0.62))
    min_lexical_score: float = field(default_factory=lambda: _float("MIN_LEXICAL_SCORE", 0.30))
    min_score_margin: float = field(default_factory=lambda: _float("MIN_SCORE_MARGIN", 0.04))
    min_groundedness: float = field(default_factory=lambda: _float("MIN_GROUNDEDNESS", 0.70))
    enable_verifier: bool = field(default_factory=lambda: _bool("ENABLE_VERIFIER", True))

    # --- freshness ---
    stale_after_days: int = field(default_factory=lambda: _int("STALE_AFTER_DAYS", 365))
    today_override: str = field(default_factory=lambda: _str("TODAY_OVERRIDE"))

    # ------------------------------------------------------------------
    @property
    def db_file(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else ROOT / p

    @property
    def kb_dir(self) -> Path:
        p = Path(self.kb_path)
        return p if p.is_absolute() else ROOT / p

    @property
    def has_llm(self) -> bool:
        return self.llm_provider == "groq" and bool(self.groq_api_key)

    @property
    def has_embeddings(self) -> bool:
        return self.embedding_provider == "gemini" and bool(self.gemini_api_key)

    def min_score_for(self, mode: str) -> float:
        """The relevance floor for the retrieval mode actually used."""
        return self.min_dense_score if mode == "hybrid" else self.min_lexical_score

    def today(self) -> date:
        if self.today_override:
            try:
                return datetime.fromisoformat(self.today_override).date()
            except ValueError:
                pass
        return datetime.now(timezone.utc).date()

    def describe(self) -> str:
        """One-line capability banner, printed by the CLI and /health."""
        llm = f"groq:{self.llm_model}" if self.has_llm else "offline-extractive (no GROQ_API_KEY)"
        emb = (
            f"gemini:{self.embedding_model}"
            if self.has_embeddings
            else "disabled — BM25-only retrieval (no GEMINI_API_KEY)"
        )
        return f"generation={llm} | embeddings={emb}"


settings = Settings()
