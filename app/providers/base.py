"""Provider interfaces."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMError(RuntimeError):
    """Raised when a provider call fails in a way the caller must handle.

    The pipeline treats this as a *degradation* signal, never a crash: a failed
    LLM call becomes an escalation, not a 500.
    """


class LLM(ABC):
    """Chat-completion interface."""

    name: str = "llm"
    model: str = "unknown"

    @property
    def is_live(self) -> bool:
        """False for the offline stub, so the pipeline can label its output."""
        return True

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        ...

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Complete and parse JSON, tolerating fenced or prose-wrapped output."""
        raw = self.complete(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return extract_json(raw)


class Embedder(ABC):
    """Text-embedding interface."""

    name: str = "embedder"
    model: str = "none"
    dim: int = 0

    @property
    def is_live(self) -> bool:
        return True

    @abstractmethod
    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Return one L2-normalised vector per input text."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON recovery from a model response.

    Models occasionally wrap JSON in prose or code fences even when asked not
    to. Failing the whole turn over a stray backtick would turn a good answer
    into an escalation, so we try three increasingly forgiving strategies
    before giving up.
    """
    raw = (raw or "").strip()
    if not raw:
        raise LLMError("empty response from model")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _FENCE.search(raw)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start, depth, in_str, esc = raw.find("{"), 0, False, False
    if start != -1:
        for i in range(start, len(raw)):
            ch = raw[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break

    raise LLMError(f"could not parse JSON from model response: {raw[:200]!r}")
