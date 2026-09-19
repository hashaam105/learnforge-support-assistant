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
    """Groq adapter.

    One provider-specific detail is load-bearing. The gpt-oss models are
    *reasoning* models, and `max_tokens` bounds reasoning tokens plus output
    tokens together. A budget that looks generous for a one-line JSON answer
    can be consumed entirely by reasoning, leaving empty content — which the
    API reports as HTTP 400 `json_validate_failed`, not as a truncation.

    That failure mode is quiet and expensive: it silently disabled multi-turn
    query rewriting here, because the rewriter caught the error, fell back to
    the raw follow-up, and "It's the biology one" went to the retriever
    unresolved. Two defences:

      * send `reasoning_effort` (low for the short structured tasks), so the
        model spends its budget on the answer rather than the deliberation;
      * treat `json_validate_failed` as retryable with a doubled budget, for
        models or future versions that ignore the hint.
    """

    name = "groq"

    def __init__(
        self,
        cfg: Settings | None = None,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.model = model or self.cfg.llm_model
        self.reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else self.cfg.llm_reasoning_effort
        )
        if not self.cfg.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self._client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._supports_reasoning_effort = True

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        budget = self.cfg.llm_max_tokens if max_tokens is None else max_tokens
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.llm_temperature if temperature is None else temperature,
            "max_tokens": budget,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and self._supports_reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

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
            body = resp.text

            # The model reasoned past its budget and returned no content.
            # Give it more room rather than losing the turn.
            if resp.status_code == 400 and "json_validate_failed" in body and attempt < _MAX_ATTEMPTS:
                budget = min(budget * 2, 4000)
                payload["max_tokens"] = budget
                payload.setdefault("reasoning_effort", "low")
                continue

            # Some models do not accept reasoning_effort; drop it and retry once.
            if (
                resp.status_code == 400
                and "reasoning_effort" in body
                and self._supports_reasoning_effort
            ):
                self._supports_reasoning_effort = False
                payload.pop("reasoning_effort", None)
                continue

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
