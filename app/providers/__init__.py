"""Model providers.

Two narrow interfaces (`LLM`, `Embedder`) plus concrete adapters. Swapping
Groq for Gemini/OpenRouter/vLLM is a new file in this package and one env var,
not a change anywhere in the pipeline.
"""

from __future__ import annotations

from app.config import Settings, settings

from .base import LLM, Embedder, LLMError
from .gemini_embed import GeminiEmbedder
from .groq_llm import GroqLLM
from .offline import NullEmbedder, OfflineLLM

__all__ = [
    "LLM",
    "Embedder",
    "LLMError",
    "GroqLLM",
    "GeminiEmbedder",
    "OfflineLLM",
    "NullEmbedder",
    "get_llm",
    "get_embedder",
]


def get_llm(cfg: Settings | None = None, *, role: str = "primary") -> LLM:
    """Return the model for a role.

    `primary` composes the answer shown to the learner. `utility` handles
    query rewriting, reranking, entailment checking and summarising — easier
    tasks that make up most of the calls in a turn and do not need the larger
    model. See Settings.utility_model for why the split is worth having.
    """
    cfg = cfg or settings
    if not cfg.has_llm:
        return OfflineLLM(cfg)
    if role == "utility":
        return GroqLLM(
            cfg,
            model=cfg.utility_model or cfg.llm_model,
            reasoning_effort=cfg.utility_reasoning_effort,
        )
    return GroqLLM(cfg)


def get_embedder(cfg: Settings | None = None) -> Embedder:
    cfg = cfg or settings
    if cfg.has_embeddings:
        return GeminiEmbedder(cfg)
    return NullEmbedder(cfg)
