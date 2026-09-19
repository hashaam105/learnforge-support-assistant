"""Groq chat-completions adapter (OpenAI-compatible endpoint).

Chosen for generation because the free tier is genuinely free, needs no card,
and serves 70B-class open models fast enough that a two-pass pipeline
(generate then verify) still feels interactive.
"""

from __future__ import annotations

import time

import httpx

from app.config import Settings, settings

from .base import LLM, LLMError

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Free tiers rate-limit aggressively. A bounded retry with backoff is the
# difference between "the prototype works" and "the prototype works unless the
# reviewer runs the eval suite".
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4


class GroqLLM(LLM):
    name = "groq"

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self.model = self.cfg.llm_model
        if not self.cfg.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self._client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.llm_temperature if temperature is None else temperature,
            "max_tokens": self.cfg.llm_max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.cfg.groq_api_key}",
            "Content-Type": "application/json",
        }

        last_error = "unknown"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = self._client.post(ENDPOINT, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                if attempt == _MAX_ATTEMPTS:
                    break
                time.sleep(min(2**attempt, 8))
                continue

            if resp.status_code == 200:
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError) as exc:
                    raise LLMError(f"unexpected Groq response shape: {exc}") from exc

            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS:
                # Honour Retry-After when the provider sends one.
                wait = resp.headers.get("retry-after")
                delay = float(wait) if wait and wait.replace(".", "", 1).isdigit() else min(2**attempt, 8)
                time.sleep(delay)
                continue
            break

        raise LLMError(f"Groq call failed after {_MAX_ATTEMPTS} attempts — {last_error}")

    def close(self) -> None:
        self._client.close()
