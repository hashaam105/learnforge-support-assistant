"""Turn a follow-up message into a standalone, retrievable query.

TICKET-07 in the sample corpus is the whole argument for this module:

    USER:  Cancel my LearnForge.
    AGENT: Subscription, enrollment, or account?
    USER:  The payment.
    AGENT: Stop future payments, or refund one already made?
    USER:  The payment from last week.
    ...
    USER:  It's the biology one.

Embedding "It's the biology one" retrieves nothing useful. Every turn after
the first has to be rewritten against the conversation before it touches the
index.

Two mechanisms, deliberately layered:

* a **deterministic slot filler** (regex) that accumulates the concrete
  identifiers a support conversation produces — course name, order number,
  amount, platform, the fact that a screenshot was offered. These persist in
  `conversations.slots` and survive summarisation, because losing an order
  number to a context window is a real failure mode.
* an **LLM rewriter** that resolves pronouns and ellipsis using recent turns.

The slot filler runs even when no LLM key is configured, so the keyless mode
still carries entities forward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, settings
from app.providers.base import LLM, LLMError

# --- deterministic slot patterns -------------------------------------------

_COURSE_QUOTED = re.compile(r"[\"“]([A-Z][\w][^\"”]{2,60})[\"”]")
_COURSE_NAMED = re.compile(
    r"\b(?:the\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s+(?:course|masterclass|class)\b"
)
_COURSE_TOPIC = re.compile(r"\bthe\s+(biology|physics|chemistry|python|ux|design|data)\s+one\b", re.I)
_ORDER_ID = re.compile(r"\border\s*(?:number|no\.?|#|id)?\s*[:#]?\s*([A-Z0-9][A-Z0-9-]{4,})\b", re.I)
_AMOUNT = re.compile(r"([$£€]\s?\d{1,5}(?:[.,]\d{2})?)")
_PLATFORM = {
    "ios": ("iphone", "ipad", "app store", "apple"),
    "android": ("android", "google play", "play store"),
    "mobile_app": ("mobile app", "the app", "mobile application"),
    "web": ("browser", "laptop", "desktop", "website", "chrome", "firefox", "safari", "edge"),
}

# Sensitive values a support assistant must refuse to accept, per POLICY-07
# and POLICY-10 ("Support agents must not request complete card numbers, CVV
# codes, PINs, banking passwords"). Detected on the way IN so the value is
# never embedded, never logged in full, and never echoed back.
_CARD_LIKE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_CVV_LIKE = re.compile(
    r"\b(?:cvv|cvc|security code|pin)\b\s*(?:is|are|=|:|#)?\s*\d{3,6}\b", re.I
)
_EMAIL = re.compile(r"\b[\w.+-]+@([\w-]+\.[\w.-]+)\b")

# Phrases that mean "stop talking to me, get a person".
_HUMAN_REQUEST = re.compile(
    r"\b(speak|talk|connect|transfer|escalate)\w*\s+(to|with|me\s+to)?\s*(a\s+)?"
    r"(human|person|agent|representative|someone|manager|supervisor)\b|"
    r"\b(real|live)\s+(person|human|agent)\b|\bhuman\s+support\b",
    re.I,
)

_FOLLOWUP_HINT = re.compile(
    r"^\s*(it|that|this|they|them|those|these|he|she|the\s+(one|payment|course|charge|order))\b"
    r"|^\s*(yes|yeah|yep|no|nope|ok|okay|sure|correct|right)\b\s*[.,!]?\s*$"
    r"|^\s*(what|how|why|when|and|but|so)\s+about\b",
    re.I,
)


@dataclass
class TurnContext:
    """Everything downstream stages need to know about this turn."""

    raw_query: str
    standalone_query: str
    slots: dict[str, Any] = field(default_factory=dict)
    is_followup: bool = False
    rewritten_by_llm: bool = False
    redactions: list[str] = field(default_factory=list)
    wants_human: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "standalone_query": self.standalone_query,
            "slots": self.slots,
            "is_followup": self.is_followup,
            "rewritten_by_llm": self.rewritten_by_llm,
            "redactions": self.redactions,
            "wants_human": self.wants_human,
        }


class Contextualizer:
    def __init__(self, llm: LLM, cfg: Settings | None = None) -> None:
        self.llm = llm
        self.cfg = cfg or settings

    # ------------------------------------------------------------------
    def build(
        self,
        query: str,
        history: list[dict[str, Any]],
        prior_slots: dict[str, Any] | None = None,
    ) -> TurnContext:
        safe_query, redactions = redact(query)
        slots = dict(prior_slots or {})
        slots.update(extract_slots(safe_query))

        ctx = TurnContext(
            raw_query=safe_query,
            standalone_query=safe_query,
            slots=slots,
            redactions=redactions,
            wants_human=bool(_HUMAN_REQUEST.search(safe_query)),
        )

        if not history:
            # First turn: nothing to dereference. Skipping the LLM call here is
            # a meaningful latency win, since most sessions are one turn.
            ctx.standalone_query = _enrich_with_slots(safe_query, slots)
            return ctx

        ctx.is_followup = bool(_FOLLOWUP_HINT.search(safe_query)) or len(safe_query.split()) <= 6

        if self.llm.is_live:
            rewritten = self._llm_rewrite(safe_query, history, slots)
            if rewritten:
                ctx.standalone_query = rewritten
                ctx.rewritten_by_llm = True
                return ctx

        # No LLM (or the rewrite failed): fall back to appending known entities.
        # Crude, but it recovers the common case where the user drops a noun
        # they established three turns ago.
        ctx.standalone_query = _enrich_with_slots(safe_query, slots)
        return ctx

    # ------------------------------------------------------------------
    def _llm_rewrite(
        self, query: str, history: list[dict[str, Any]], slots: dict[str, Any]
    ) -> str | None:
        transcript = "\n".join(
            f"{h['role'].upper()}: {_clip(h['content'], 300)}" for h in history[-6:]
        )
        known = ", ".join(f"{k}={v}" for k, v in slots.items() if v) or "none"
        system = (
            "TASK: contextualize\n"
            "You rewrite the latest message in a customer-support conversation into a single "
            "self-contained search query for a knowledge base.\n"
            "Rules:\n"
            "- Resolve every pronoun and ellipsis using the conversation "
            '("the biology one" -> "Biology Essentials course").\n'
            "- Keep the user's intent exactly; never answer, never add facts, never guess "
            "details that were not stated.\n"
            "- If the latest message already stands alone, return it unchanged.\n"
            "- Write it as a question or request, not as a keyword list.\n"
            'Reply with JSON only: {"standalone_query": "...", "is_followup": true|false}'
        )
        user = (
            f"KNOWN DETAILS: {known}\n\nCONVERSATION SO FAR:\n{transcript}\n\n"
            f"LATEST MESSAGE: {query}"
        )
        try:
            data = self.llm.complete_json(
                system, user, temperature=0.0, max_tokens=700, max_attempts=3
            )
        except (LLMError, TypeError):
            return None
        rewritten = str(data.get("standalone_query", "")).strip()
        if not rewritten or len(rewritten) > 500:
            return None
        return rewritten


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def redact(text: str) -> tuple[str, list[str]]:
    """Strip payment secrets before the text is stored, embedded or sent onward.

    The corpus is explicit that agents must never hold this data, and an LLM
    pipeline leaks it in more places than a human agent does: the prompt, the
    embedding request, the message log, the escalation payload. Removing it at
    the boundary is the only place that covers all four.

    Email addresses are reduced to their domain: enough to reason about
    "university email" (FAQ-09) without persisting the identifier.
    """
    redactions: list[str] = []

    def _card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) < 13:
            return match.group(0)
        redactions.append("card_number")
        return f"[REDACTED-CARD ending {digits[-4:]}]"

    out = _CARD_LIKE.sub(_card, text)
    if _CVV_LIKE.search(out):
        out = _CVV_LIKE.sub("[REDACTED-SECURITY-CODE]", out)
        redactions.append("security_code")
    if _EMAIL.search(out):
        out = _EMAIL.sub(lambda m: f"[email@{m.group(1)}]", out)
        redactions.append("email")
    return out, redactions


def extract_slots(text: str) -> dict[str, Any]:
    """Pull durable conversation entities out of one message."""
    slots: dict[str, Any] = {}

    course = _COURSE_QUOTED.search(text) or _COURSE_NAMED.search(text)
    if course:
        slots["course_name"] = course.group(1).strip()
    else:
        topic = _COURSE_TOPIC.search(text)
        if topic:
            slots["course_topic"] = topic.group(1).lower()

    order = _ORDER_ID.search(text)
    if order:
        slots["order_id"] = order.group(1).strip()

    amount = _AMOUNT.search(text)
    if amount:
        slots["amount"] = amount.group(1).replace(" ", "")

    lowered = text.lower()
    for platform, needles in _PLATFORM.items():
        if any(n in lowered for n in needles):
            slots["platform"] = platform
            break

    if "screenshot" in lowered:
        slots["has_screenshot"] = True
    return slots


def _enrich_with_slots(query: str, slots: dict[str, Any]) -> str:
    """Append entities the query does not already name.

    Used on the first turn and whenever the LLM rewriter is unavailable. It
    cannot resolve coreference, but it does stop a bare "the payment" from
    being searched without the course name established two turns earlier.
    """
    lowered = query.lower()
    extras = [
        str(value)
        for key, value in slots.items()
        if key in {"course_name", "course_topic", "platform"}
        and value
        and str(value).lower() not in lowered
    ]
    return f"{query} ({'; '.join(extras)})" if extras else query


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
