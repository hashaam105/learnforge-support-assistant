"""The turn orchestrator: one user message in, one grounded decision out.

    contextualize -> retrieve -> generate -> verify -> decide -> persist

Each stage is a separate object with a narrow interface, which is what makes
the eval harness possible: `eval/run.py` can exercise retrieval alone,
generation alone, or the whole loop, and a regression in one stage does not
hide inside another.

Everything about a turn is written to `messages` — the rewritten query, the
chunks retrieved, the citations, every score, the action and its reason codes.
A support lead asking "why did the bot tell that learner 7 days?" can answer
it from the database without re-running anything.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, settings
from app.contextualizer import Contextualizer, TurnContext
from app.escalation import (
    Decision,
    abstain_message,
    build_escalation_payload,
    decide,
    handoff_message,
)
from app.generator import Generation, Generator
from app.providers import get_embedder, get_llm
from app.providers.base import LLM, Embedder, LLMError
from app.retrieval import RetrievalResult, Retriever
from app.store import Store
from app.text import content_terms
from app.verifier import Verification, Verifier

SUMMARISE_EVERY = 6  # turns


@dataclass
class TurnResult:
    session_id: str
    question: str
    reply: str
    action: str
    confidence: float
    citations: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    escalation_id: str | None = None
    queue: str | None = None
    priority: str | None = None
    intent: str = "general_support"
    standalone_query: str = ""
    latency_ms: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "reply": self.reply,
            "action": self.action,
            "confidence": self.confidence,
            "citations": self.citations,
            "sources": self.sources,
            "reason_codes": self.reason_codes,
            "caveats": self.caveats,
            "escalation_id": self.escalation_id,
            "queue": self.queue,
            "priority": self.priority,
            "intent": self.intent,
            "standalone_query": self.standalone_query,
            "latency_ms": self.latency_ms,
            "diagnostics": self.diagnostics,
        }


class SupportAssistant:
    def __init__(
        self,
        cfg: Settings | None = None,
        store: Store | None = None,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.store = store or Store(self.cfg)
        self.llm = llm if llm is not None else get_llm(self.cfg)
        # The generator gets the primary model; everything else runs on the
        # smaller utility model. Tests and eval can override both by passing a
        # single `llm`, which is then used for every stage.
        self.utility_llm = llm if llm is not None else get_llm(self.cfg, role="utility")
        self.embedder = embedder if embedder is not None else get_embedder(self.cfg)

        self.retriever = Retriever(
            self.store, self.cfg, embedder=self.embedder, llm=self.utility_llm
        )
        self.contextualizer = Contextualizer(self.utility_llm, self.cfg)
        self.generator = Generator(self.llm, self.cfg)
        self.verifier = Verifier(self.utility_llm, self.cfg)

    # ------------------------------------------------------------------
    def ask(self, question: str, session_id: str | None = None) -> TurnResult:
        started = time.perf_counter()
        session_id = session_id or f"S-{uuid.uuid4().hex[:12]}"

        conversation = self.store.ensure_conversation(session_id)
        history = self.store.history(session_id)
        prior_slots = json.loads(conversation.get("slots") or "{}")

        # 1. contextualize -------------------------------------------------
        ctx = self.contextualizer.build(question, history, prior_slots)

        # 2. retrieve ------------------------------------------------------
        retrieval = self.retriever.retrieve(ctx.standalone_query)

        # 3. generate ------------------------------------------------------
        generation = self.generator.generate(
            ctx.standalone_query, retrieval, history=history, slots=ctx.slots
        )

        # 4. verify --------------------------------------------------------
        verification = self.verifier.verify(generation, retrieval)

        # 5. decide --------------------------------------------------------
        decision = decide(
            question=f"{ctx.raw_query} {ctx.standalone_query}",
            retrieval=retrieval,
            generation=generation,
            verification=verification,
            wants_human=ctx.wants_human,
            prior_failures=_prior_failures(self.store, session_id),
            cfg=self.cfg,
        )

        # 6. compose + persist ---------------------------------------------
        reply = self._compose_reply(ctx, generation, decision)
        latency_ms = int((time.perf_counter() - started) * 1000)

        turn_index = self.store.next_turn_index(session_id)
        self.store.add_message(
            session_id=session_id,
            turn_index=turn_index,
            role="user",
            content=ctx.raw_query,
            standalone_query=ctx.standalone_query,
        )
        message_id = self.store.add_message(
            session_id=session_id,
            turn_index=turn_index,
            role="assistant",
            content=reply,
            standalone_query=ctx.standalone_query,
            retrieved_chunk_ids=json.dumps([c.chunk_id for c in retrieval.candidates]),
            citations=json.dumps(generation.citations),
            retrieval_score=retrieval.top_score,
            score_margin=retrieval.margin,
            groundedness=verification.groundedness,
            confidence=decision.confidence,
            action=decision.action,
            reason_codes=json.dumps(decision.reason_codes),
            latency_ms=latency_ms,
            model=self.llm.model,
        )

        escalation_id = None
        if decision.action == "escalated":
            escalation_id = self.store.add_escalation(
                build_escalation_payload(
                    session_id=session_id,
                    message_id=message_id,
                    question=ctx.standalone_query,
                    decision=decision,
                    retrieval=retrieval,
                    generation=generation,
                    verification=verification,
                    slots=ctx.slots,
                    history=history,
                )
            )

        self.store.update_conversation(
            session_id,
            slots=ctx.slots,
            summary=self._maybe_summarise(conversation, history, ctx),
            escalated=decision.action == "escalated",
        )

        return TurnResult(
            session_id=session_id,
            question=question,
            reply=reply,
            action=decision.action,
            confidence=decision.confidence,
            citations=[c for c in generation.citations if c in set(retrieval.doc_ids)],
            sources=_source_cards(retrieval, generation),
            reason_codes=decision.reason_codes,
            caveats=decision.caveats,
            escalation_id=escalation_id,
            queue=decision.queue if escalation_id else None,
            priority=decision.priority if escalation_id else None,
            intent=decision.intent,
            standalone_query=ctx.standalone_query,
            latency_ms=latency_ms,
            diagnostics={
                "context": ctx.as_dict(),
                "retrieval": retrieval.diagnostics(),
                "generation": generation.as_dict(),
                "verification": verification.as_dict(),
                "decision": decision.as_dict(),
                "capabilities": {
                    "llm_live": self.llm.is_live,
                    "embeddings_live": self.embedder.is_live,
                    "model": self.llm.model,
                },
            },
        )

    # ------------------------------------------------------------------
    def _compose_reply(
        self, ctx: TurnContext, generation: Generation, decision: Decision
    ) -> str:
        if decision.action == "escalated":
            parts = [handoff_message(decision)]
            # Show the draft when it is safe to: partial information plus an
            # explicit "a person will confirm" beats a bare "please wait".
            if (
                generation.answer
                and generation.answerable
                and "retired_claim_leak" not in decision.reason_codes
                and "invalid_citation" not in decision.reason_codes
                and "unsafe_information_request" not in decision.reason_codes
            ):
                parts.append(
                    "In the meantime, here is what our help centre says — a colleague will "
                    f"confirm how it applies to your account:\n\n{generation.answer}"
                )
            if ctx.redactions:
                parts.append(_redaction_notice(ctx.redactions))
            return "\n\n".join(parts)

        if decision.action == "abstained":
            return abstain_message()

        parts = [generation.answer.strip()]
        if generation.clarifying_question and generation.clarifying_question not in generation.answer:
            parts.append(generation.clarifying_question)
        if decision.caveats:
            parts.append("_" + " ".join(decision.caveats) + "_")
        if ctx.redactions:
            parts.append(_redaction_notice(ctx.redactions))
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    def _maybe_summarise(
        self, conversation: dict[str, Any], history: list[dict[str, Any]], ctx: TurnContext
    ) -> str:
        """Compress old turns so long conversations stay inside the context budget.

        Slots are *not* summarised — an order number must survive verbatim,
        and asking a model to preserve it through a paraphrase is the kind of
        lossy step that makes a learner repeat themselves.
        """
        existing = conversation.get("summary") or ""
        turn_count = int(conversation.get("turn_count") or 0)
        if turn_count == 0 or turn_count % SUMMARISE_EVERY != 0 or not self.utility_llm.is_live:
            return existing

        transcript = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in history)
        try:
            data = self.utility_llm.complete_json(
                "TASK: summarize\nYou compress a support conversation for an agent picking it up "
                "mid-thread. Keep the learner's goal, what has been established, what has been "
                "tried, and anything still unresolved. Drop pleasantries. Never invent details.\n"
                'Reply with JSON only: {"summary": "..."}',
                f"PREVIOUS SUMMARY: {existing or 'none'}\n\nTRANSCRIPT:\n{transcript}",
                temperature=0.0,
                max_tokens=800,
            )
            return str(data.get("summary", existing)).strip() or existing
        except (LLMError, TypeError):
            return existing

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.store.close()


# ---------------------------------------------------------------------------


def _source_cards(retrieval: RetrievalResult, generation: Generation) -> list[dict[str, Any]]:
    """Per-document provenance for the UI, cited documents first.

    When several chunks of one document are in context, the card shows the one
    that actually supports the answer rather than simply the highest-scoring
    one. Those differ more often than you would expect: POLICY-02 states the
    14-day refund rule in its first chunk, but its second chunk ranks higher
    for a general refund question, so the card used to display text that said
    nothing about 14 days — directly under an answer claiming exactly that.
    A citation a reader cannot verify by looking is worse than no citation,
    because it looks checked.
    """
    cited = set(generation.citations)
    answer_terms = set(content_terms(generation.answer or ""))

    by_doc: dict[str, list] = {}
    for c in retrieval.candidates:
        by_doc.setdefault(c.doc_id, []).append(c)

    seen: set[str] = set()
    cards: list[dict[str, Any]] = []
    for candidate in retrieval.candidates:
        if candidate.doc_id in seen:
            continue
        seen.add(candidate.doc_id)
        c = _best_supporting_chunk(by_doc[candidate.doc_id], answer_terms)
        cards.append(
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "source_type": c.source_type,
                "authority_tier": c.authority_tier,
                "date_label": c.date_label,
                "cited": c.doc_id in cited,
                "score": round(c.final_score, 3),
                "is_stale": c.is_stale,
                "has_deprecation_notice": c.has_deprecation_notice,
                "source_uri": c.source_uri,
                "excerpt": _evidence_excerpt(c.text, answer_terms),
            }
        )
    cards.sort(key=lambda card: (not card["cited"], -card["score"]))
    return cards


_EXCERPT_CHARS = 260
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _evidence_excerpt(text: str, answer_terms: set[str], limit: int = _EXCERPT_CHARS) -> str:
    """A window centred on the sentence that supports the answer.

    Taking the first N characters is the obvious thing and the wrong one. The
    rule a reader wants to check is rarely the opening line of its chunk:
    POLICY-02 opens on cancellation and states the 14-day refund window
    several sentences later, so a leading excerpt showed none of what the
    answer claimed. The citation then looks checkable and isn't, which is
    worse than showing nothing.
    """
    text = text.strip()
    if len(text) <= limit:
        return text

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not answer_terms or not sentences:
        return text[:limit].rstrip() + "…"

    best_at, best_score = 0, -1
    cursor = 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        cursor = (start if start != -1 else cursor) + len(sentence)
        score = len(answer_terms & set(content_terms(sentence)))
        if score > best_score:
            best_at, best_score = (start if start != -1 else 0), score

    # Centre the window on that sentence, then clamp to the text.
    start = max(0, best_at - limit // 4)
    end = min(len(text), start + limit)
    start = max(0, end - limit)
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _best_supporting_chunk(chunks: list, answer_terms: set[str]):
    """Of one document's retrieved chunks, the one the answer actually drew on.

    Overlap with the answer's own vocabulary is a crude signal, but it is the
    right one here: we are picking what to *show* as evidence, and the useful
    excerpt is the passage a reader can match against the sentence above it.
    With no answer to compare against (an abstention, say) the ranking order
    stands.
    """
    if len(chunks) == 1 or not answer_terms:
        return chunks[0]
    return max(
        chunks,
        key=lambda c: (len(answer_terms & set(content_terms(c.text))), c.final_score),
    )


def _prior_failures(store: Store, session_id: str) -> int:
    """How many recent assistant turns already failed this learner."""
    rows = store.conn.execute(
        """
        SELECT action FROM messages
        WHERE session_id = ? AND role = 'assistant' AND action IS NOT NULL
        ORDER BY message_id DESC LIMIT 3
        """,
        (session_id,),
    ).fetchall()
    return sum(1 for r in rows if r["action"] in {"escalated", "abstained"})


def _redaction_notice(redactions: list[str]) -> str:
    kinds = {
        "card_number": "your full card number",
        "security_code": "a security code",
        "email": "your email address",
    }
    named = [kinds.get(r, r) for r in redactions if r != "email"]
    if not named:
        return ""
    return (
        f"_One more thing: please don't send {' or '.join(named)} — we never need it, and I've "
        "removed it from this conversation._"
    )
