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


def get_llm(cfg: Settings | None = None) -> LLM:
    cfg = cfg or settings
    if cfg.has_llm:
        return GroqLLM(cfg)
    return OfflineLLM(cfg)


def get_embedder(cfg: Settings | None = None) -> Embedder:
    cfg = cfg or settings
    if cfg.has_embeddings:
        return GeminiEmbedder(cfg)
    return NullEmbedder(cfg)
