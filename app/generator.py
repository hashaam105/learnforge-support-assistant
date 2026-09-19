"""Grounded answer generation.

The prompt does most of the anti-hallucination work, and it is built around
four properties of this specific corpus:

1. **Citations are mandatory and machine-checkable.** Every context block is
   labelled `[FAQ-02]`, and the model must return the ids it used in a
   separate `citations` field. `app/verifier.py` then checks those ids against
   what was actually retrieved, which catches the most common failure mode —
   a fluent answer attributed to a document that was never in context.

2. **Authority is stated explicitly, per block.** Each block declares whether
   it is policy, FAQ or a past ticket. Without that, the model happily quotes
   an agent's off-hand "our standard refund period is 14 days" from TICKET-03
   as though it were the policy.

3. **Retired statements are named in the prompt.** The claims extracted into
   `deprecated_claims` are listed as things the model may describe as former
   policy but must never assert as current. This is what stops "you have 7
   days to request a refund" — a sentence that appears verbatim in POLICY-02,
   inside a paragraph explaining that it is no longer true.

4. **Abstention is a first-class output.** `answerable: false` is a normal,
   expected result, not a failure. A model that cannot say "I don't know" will
   invent something instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, settings
from app.providers.base import LLM, LLMError
from app.retrieval import Candidate, RetrievalResult

SOURCE_LABEL = {
    "policy": "POLICY — authoritative, this governs",
    "faq": "FAQ — official help-centre explanation",
    "ticket": "PAST TICKET — one agent's handling of one case; NOT policy",
}

SYSTEM_PROMPT = """TASK: answer
You are the LearnForge support assistant. LearnForge is an online course platform.
You answer learner questions using ONLY the knowledge-base passages supplied below.

GROUNDING RULES
- Use only the supplied passages. If they do not contain the answer, set
  "answerable": false and say what is missing. Never fill a gap from general
  knowledge about other e-learning platforms.
- Every factual sentence must trace to a passage. Put the source id inline in
  square brackets, e.g. "refunds are generally available within 14 days [POLICY-02]".
- List every id you used in "citations". Never cite an id that is not in the
  passages above.
- Do not invent order numbers, dates, amounts, URLs, timelines or contact
  addresses. Do not promise a specific outcome ("you will be refunded") — the
  passages describe eligibility and review, not guarantees.

SOURCE AUTHORITY (highest first)
1. POLICY passages state the governing rule.
2. FAQ passages explain it to learners.
3. PAST TICKET passages are anecdotes. They show how a case was handled once.
   Never present a ticket as the rule. You may say "in a similar past case...".
When sources disagree, follow the highest-authority and most recently dated
passage, say plainly that the sources differ, and set "conflict_detected": true.

RETIRED INFORMATION
Some passages quote statements that LearnForge has since withdrawn. Any such
statement is listed under RETIRED STATEMENTS. You may mention one as former
wording, clearly marked as no longer current. You must never state it as the
current rule, and never as the direct answer to the question.

SAFETY
- Never ask for, repeat or confirm a full card number, CVV/security code, PIN,
  password or authentication code. If the learner has supplied one, tell them
  not to share it and continue without it.
- For suspected fraud or an unrecognised charge, describe LearnForge's process
  and point to the bank/provider route as well; do not adjudicate the claim.

STYLE
- Speak to the learner directly, warm and plain. 2-5 short paragraphs or a
  short numbered list. No preamble, no "based on the provided context".
- If the answer depends on something you do not know (which product, which
  store processed the payment), give the answer for the likely case and state
  the condition, or ask ONE specific clarifying question.

Reply with JSON only, no prose around it:
{
  "answer": "the reply shown to the learner, with inline [ID] citations",
  "citations": ["POLICY-02"],
  "answerable": true,
  "conflict_detected": false,
  "conflict_explanation": "",
  "missing_information": "",
  "clarifying_question": "",
  "self_confidence": 0.0
}"""


@dataclass
class Generation:
    answer: str
    citations: list[str] = field(default_factory=list)
    answerable: bool = True
    conflict_detected: bool = False
    conflict_explanation: str = ""
    missing_information: str = ""
    clarifying_question: str = ""
    self_confidence: float = 0.0
    model: str = ""
    degraded: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "answerable": self.answerable,
            "conflict_detected": self.conflict_detected,
            "conflict_explanation": self.conflict_explanation,
            "missing_information": self.missing_information,
            "clarifying_question": self.clarifying_question,
            "self_confidence": self.self_confidence,
            "model": self.model,
            "degraded": self.degraded,
            "error": self.error,
        }


class Generator:
    def __init__(self, llm: LLM, cfg: Settings | None = None) -> None:
        self.llm = llm
        self.cfg = cfg or settings

    def generate(
        self,
        question: str,
        retrieval: RetrievalResult,
        *,
        history: list[dict[str, Any]] | None = None,
        slots: dict[str, Any] | None = None,
    ) -> Generation:
        if not retrieval.candidates:
            return Generation(
                answer="",
                answerable=False,
                missing_information="No knowledge-base passage matched the question.",
                model=self.llm.model,
                degraded=not self.llm.is_live,
            )

        user_prompt = build_user_prompt(question, retrieval, history=history, slots=slots)
        try:
            data = self.llm.complete_json(SYSTEM_PROMPT, user_prompt)
        except LLMError as exc:
            # A provider failure must not become a fabricated answer. We return
            # an unanswerable Generation and let the escalation gate route it.
            return Generation(
                answer="",
                answerable=False,
                missing_information="The answering model was unavailable.",
                model=self.llm.model,
                degraded=True,
                error=str(exc),
            )

        citations = [str(c).strip().upper() for c in _as_list(data.get("citations"))]
        return Generation(
            answer=str(data.get("answer", "")).strip(),
            citations=citations,
            answerable=bool(data.get("answerable", True)),
            conflict_detected=bool(data.get("conflict_detected", False))
            or bool(retrieval.conflicts),
            conflict_explanation=str(data.get("conflict_explanation", "")).strip(),
            missing_information=str(data.get("missing_information", "")).strip(),
            clarifying_question=str(data.get("clarifying_question", "")).strip(),
            self_confidence=_as_float(data.get("self_confidence"), default=0.5),
            model=self.llm.model,
            degraded=not self.llm.is_live,
        )


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def build_user_prompt(
    question: str,
    retrieval: RetrievalResult,
    *,
    history: list[dict[str, Any]] | None = None,
    slots: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []

    if history:
        transcript = "\n".join(
            f"{h['role'].upper()}: {_clip(h['content'], 250)}" for h in history[-6:]
        )
        parts.append(f"CONVERSATION SO FAR:\n{transcript}")

    known = {k: v for k, v in (slots or {}).items() if v}
    if known:
        parts.append(
            "DETAILS THE LEARNER HAS ALREADY GIVEN:\n"
            + "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in known.items())
        )

    parts.append(f"QUESTION: {question}")
    parts.append("KNOWLEDGE-BASE PASSAGES:\n" + format_context(retrieval.candidates))

    if retrieval.deprecated_claims:
        parts.append(
            "RETIRED STATEMENTS — present in the passages above but NO LONGER CURRENT.\n"
            "Never give any of these as the answer:\n"
            + "\n".join(
                f"- [{c['doc_id']}] {_clip(c['claim_text'], 260)}"
                for c in retrieval.deprecated_claims
            )
        )

    if retrieval.conflicts:
        lines = []
        for conflict in retrieval.conflicts:
            detail = "; ".join(
                f"{value} {conflict['unit']}(s) in {', '.join(docs)}"
                for value, docs in conflict["sources"].items()
            )
            lines.append(f"- competing values: {detail}")
        parts.append(
            "DETECTED DISAGREEMENT between the passages. Resolve it by authority and date, "
            "tell the learner the sources differ, and set conflict_detected:\n" + "\n".join(lines)
        )

    return "\n\n".join(parts)


def format_context(candidates: list[Candidate]) -> str:
    """Render passages as labelled, citable blocks.

    The `[DOC-ID]` prefix is load-bearing: it is what the model cites, what the
    verifier validates against, and what the offline stub parses. Keep the
    shape stable.
    """
    blocks: list[str] = []
    for c in candidates:
        meta = [SOURCE_LABEL.get(c.source_type, c.source_type)]
        if c.date_label:
            meta.append(c.date_label)
        else:
            meta.append("no review date recorded")
        if c.ticket_status:
            meta.append(f"ticket outcome: {c.ticket_status}")
        if c.is_stale:
            meta.append("NOTE: not reviewed recently")
        if c.has_deprecation_notice:
            meta.append("NOTE: contains a withdrawn statement")
        header = f"[{c.doc_id}] ({' | '.join(meta)})"
        body = c.text.strip()
        if c.heading:
            body = f"{c.heading}\n{body}"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _as_float(value: Any, *, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
