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

# The longest we will sit inside a retry backoff. Per-minute throttling
# resolves well inside this; a daily-quota rejection does not, and waiting it
# out would strand the turn.
_MAX_RETRY_WAIT = 10.0


def _retry_after(header: str | None, attempt: int) -> float:
    """Seconds to wait, from the provider's hint or exponential backoff.

    Groq sends plain seconds ("332.64"); the HTTP spec also allows an integer
    or a date, so anything unparseable falls back to backoff.
    """
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return float(min(2**attempt, 8))


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
        max_attempts: int | None = None,
    ) -> str:
        attempts = _MAX_ATTEMPTS if max_attempts is None else max(1, max_attempts)
        budget = self.cfg.llm_max_tokens if max_tokens is None else max_tokens
        grew_budget = False
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
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post(ENDPOINT, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                if attempt == attempts:
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
            # Give it more room once. Only once: if double the budget and
            # reduced reasoning still produce nothing, more of both will not
            # help, and repeatedly retrying a slow call is worse than failing.
            # (Retrying this four times turned a fast reranker fallback into a
            # four-minute stall.)
            if (
                resp.status_code == 400
                and "json_validate_failed" in body
                and not grew_budget
                and attempt < attempts
            ):
                grew_budget = True
                budget = min(budget * 2, 4000)
                payload["max_tokens"] = budget
                payload["reasoning_effort"] = "low"
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

            if resp.status_code in _RETRY_STATUS and attempt < attempts:
                # Honour Retry-After, but only up to a point. A provider that
                # has exhausted a *daily* quota answers "try again in 5m32s",
                # and sleeping that out inside a support turn is not waiting,
                # it is hanging: the learner sees nothing for five minutes and
                # the pipeline cannot degrade because it never regains control.
                # Past the cap we give up and let the caller fall back — the
                # escalation path exists for exactly this.
                delay = _retry_after(resp.headers.get("retry-after"), attempt)
                if delay > _MAX_RETRY_WAIT:
                    last_error = f"HTTP {resp.status_code}: provider asked for {delay:.0f}s backoff"
                    break
                time.sleep(delay)
                continue
            break

        raise LLMError(f"Groq call failed after {attempts} attempt(s) — {last_error}")

    def close(self) -> None:
        self._client.close()
