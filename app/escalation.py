"""The escalation gate: when to stop answering and fetch a human.

Design position
---------------
The model does not decide this. A language model asked "are you confident?"
answers from the same distribution that produced the answer, so its confidence
correlates with fluency rather than correctness. Every rule below reads a
signal computed *outside* the generation call — retrieval scores, citation
validity, the verifier's judgement, deterministic intent patterns — and the
gate is plain Python that a support lead can read, argue with and change
without touching a prompt.

Where the rules come from
-------------------------
Mostly from the corpus. The sample tickets that ended in `STATUS: Escalated`
cluster tightly:

* TICKET-03 — user has a screenshot of a 30-day guarantee, policy says 14
* TICKET-08 — FAQ, cancellation page and subscription terms disagree
* TICKET-11 — receipt versus subscription enrolment, ownership unclear

Those are not retrieval failures. Retrieval worked; the *corpus itself* is
contradictory or the case turns on account-specific facts the knowledge base
cannot contain. A RAG system that only escalates on low similarity would
answer all three confidently and wrongly. So the gate has three independent
families of trigger: I could not find it, I do not trust what I produced, and
this class of request should never be settled by a bot.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, settings
from app.generator import Generation
from app.retrieval import RetrievalResult
from app.verifier import Verification

# --- intent patterns -------------------------------------------------------
# Ordered: the first match wins, so the most consequential intents come first.
INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "fraud_or_unrecognised_charge",
        re.compile(
            r"\b(don'?t recognis|do not recognis|didn'?t (?:buy|purchase|authoris|authoriz)|"
            r"never (?:bought|purchased|signed up|authoris|authoriz|agreed)|"
            r"not authoris|not authoriz|unauthoris|unauthoriz|fraud|stolen|"
            r"someone else used|hacked|chargeback|dispute the charge)\w*",
            re.I,
        ),
    ),
    (
        "duplicate_charge",
        re.compile(
            r"\b(charged twice|two charges|double charge|duplicate (?:charge|payment|transaction)|"
            r"billed twice|charged me again)\b",
            re.I,
        ),
    ),
    (
        "account_ownership",
        re.compile(
            r"\b(wrong (?:email|account)|different email|transfer (?:the )?(?:course|enrol)|"
            r"move the course|two accounts|second account|merge (?:my )?account|"
            r"my (?:daughter|son|child|wife|husband|brother|sister) (?:used|completed|did)|"
            r"certificate has my name|change the name on)\w*|"
            # A purchase made under one identity and an account under another.
            # Deliberately requires BOTH a purchase verb and a mismatching
            # identity, so "can I change my account email?" (FAQ-09, a normal
            # answerable question) does not get dragged into escalation.
            r"\b(?:bought|purchased|paid for|receipt|order)\b[^.?!]{0,80}\b"
            r"(?:work|university|school|different|another|other|old|personal|second)\s+"
            r"(?:e-?mail|account)\b|"
            r"\b(?:work|university|school|different|another|other|old|personal|second)\s+"
            r"(?:e-?mail|account)\b[^.?!]{0,80}\b(?:bought|purchased|paid for|receipt|order)\b",
            re.I,
        ),
    ),
    (
        "refund_dispute",
        re.compile(
            r"\b(?:refund|money back|money-back)\b[^.?!]{0,80}\b(?:but|however|says?|said|"
            r"guarantee|promised|screenshot|30[- ]day|outside)\b|"
            r"\b(?:your|the) (?:website|page|faq|policy|article) (?:said|says)\b|"
            r"\bi (?:was )?(?:told|promised)\b|"
            # A refund claim resting on something LearnForge published. TICKET-15
            # is exactly this: "Your Offline Learning Guide said I could
            # download to my laptop... that's the only reason I bought this."
            # The earlier pattern only matched the literal words "article" or
            # "website", so a named guide slipped through and got answered.
            r"\byour\b[^.?!]{0,60}\b(?:guide|article|help ?(?:page|article|cent(?:re|er))|"
            r"documentation|docs|faq)\b[^.?!]{0,40}\b(?:said|says|stated|claimed|told)\b|"
            r"\b(?:only reason|that'?s why|the reason) i (?:bought|purchased|subscribed)\b",
            re.I,
        ),
    ),
    (
        "accessibility_barrier",
        re.compile(
            r"\b(no captions|missing captions|without captions|screen reader|"
            r"can'?t access.{0,30}(?:video|lesson|content)|not accessible|"
            r"accessibility (?:issue|problem|barrier))\b",
            re.I,
        ),
    ),
    (
        "content_error",
        re.compile(
            r"\b(?:article|help page|documentation|guide|faq) (?:is )?(?:wrong|incorrect|"
            r"outdated|out of date|misleading)\b|\byour (?:article|docs?) says\b",
            re.I,
        ),
    ),
    (
        "legal_or_complaint",
        re.compile(r"\b(sue|lawyer|legal action|ombudsman|consumer rights|gdpr|complaint to)\b", re.I),
    ),
    ("refund_request", re.compile(r"\b(refund|money back|cancel (?:my )?(?:order|purchase))\b", re.I)),
    ("billing_question", re.compile(r"\b(charge|billing|invoice|payment|subscription|renew)\w*", re.I)),
    ("access_issue", re.compile(r"\b(can'?t (?:log ?in|access|see)|password|locked out|missing course)\b", re.I)),
    ("technical_issue", re.compile(r"\b(video|playback|loading|browser|crash|error|not working)\b", re.I)),
]

# Intents a bot must never settle on its own, even with perfect retrieval,
# because the decision needs account data, money movement or a judgement call.
ALWAYS_ESCALATE = {
    "fraud_or_unrecognised_charge",
    "duplicate_charge",
    "account_ownership",
    "refund_dispute",
    "legal_or_complaint",
}

QUEUE_FOR_INTENT = {
    "fraud_or_unrecognised_charge": ("trust_and_safety", "urgent"),
    "duplicate_charge": ("billing", "high"),
    "account_ownership": ("trust_and_safety", "high"),
    "refund_dispute": ("billing", "high"),
    "refund_request": ("billing", "normal"),
    "billing_question": ("billing", "normal"),
    "accessibility_barrier": ("accessibility", "high"),
    "content_error": ("content", "normal"),
    "legal_or_complaint": ("trust_and_safety", "urgent"),
    "access_issue": ("general", "normal"),
    "technical_issue": ("general", "low"),
    "general_support": ("general", "low"),
}

_FRUSTRATION = re.compile(
    r"\b(ridiculous|unacceptable|appalling|terrible|awful|useless|worst|furious|angry|"
    r"annoying|annoyed|fed up|sick of|still (?:not|hasn'?t|haven'?t)|third time|"
    r"asked (?:you )?(?:three|3|four|4|several) times|again and again|no one (?:has )?(?:replied|helped))\b",
    re.I,
)
_SHOUTING = re.compile(r"\b[A-Z]{4,}\b")

# In-domain signal: used to tell "I can't help with sourdough" (abstain, no
# ticket) from "I can't find anything about my missing refund" (escalate).
_IN_DOMAIN = re.compile(
    r"\b(course|courses|learnforge|refund|charge|charged|payment|subscription|certificate|"
    r"lesson|quiz|instructor|enrol|enroll|account|password|login|log ?in|invoice|billing|"
    r"captions?|transcript|download|offline|progress|my learning|order)\b",
    re.I,
)


@dataclass
class Decision:
    action: str                       # answered | answered_with_caveat | abstained | escalated
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    intent: str = "general_support"
    sensitive: bool = False
    queue: str = "general"
    priority: str = "normal"
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "reason_codes": self.reason_codes,
            "intent": self.intent,
            "sensitive": self.sensitive,
            "queue": self.queue,
            "priority": self.priority,
            "caveats": self.caveats,
        }


def classify_intent(text: str) -> str:
    for intent, pattern in INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return "general_support"


def composite_confidence(
    retrieval: RetrievalResult, verification: Verification, generation: Generation
) -> float:
    """Blend the independent signals into one 0..1 number.

    Weighted towards retrieval and verification because those are measured;
    the model's own `self_confidence` gets the smallest share precisely
    because it is self-reported. A hard verification failure zeroes the result
    rather than averaging away.
    """
    if verification.hard_failure or not generation.answerable:
        return 0.0
    margin_component = min(1.0, retrieval.margin / 0.10) if retrieval.margin > 0 else 0.0
    return round(
        0.40 * retrieval.top_score
        + 0.35 * verification.groundedness
        + 0.15 * margin_component
        + 0.10 * generation.self_confidence,
        4,
    )


def decide(
    *,
    question: str,
    retrieval: RetrievalResult,
    generation: Generation,
    verification: Verification,
    wants_human: bool = False,
    prior_failures: int = 0,
    cfg: Settings | None = None,
) -> Decision:
    """The gate. Every branch is explicit and independently testable."""
    cfg = cfg or settings
    reasons: list[str] = []
    caveats: list[str] = []

    intent = classify_intent(question)
    sensitive = intent in ALWAYS_ESCALATE
    queue, priority = QUEUE_FOR_INTENT.get(intent, ("general", "normal"))
    confidence = composite_confidence(retrieval, verification, generation)

    # --- family 1: I could not find it -------------------------------
    floor = cfg.min_score_for(retrieval.mode)
    if not retrieval.candidates:
        reasons.append("no_relevant_context")
    elif retrieval.top_score < floor:
        reasons.append("low_retrieval_confidence")
    elif retrieval.margin < cfg.min_score_margin and not _top_source_was_used(
        retrieval, generation
    ):
        # A small margin means the retriever has no strong opinion about which
        # document governs. That only matters if the answer did not then settle
        # on the top-ranked one: when POLICY-02 and FAQ-02 tie *and* the answer
        # cites POLICY-02, the tie is corroboration, not ambiguity, and warning
        # the learner about it just undermines a correct answer.
        reasons.append("ambiguous_sources")
        caveats.append("Several knowledge-base articles matched equally well.")

    # --- family 2: I do not trust what I produced --------------------
    if generation.error:
        reasons.append("model_unavailable")
    if not generation.answerable:
        reasons.append("answer_not_supported_by_sources")
    reasons.extend(verification.reason_codes())
    if verification.judge_ran and verification.groundedness < cfg.min_groundedness:
        reasons.append("low_groundedness")
    if generation.degraded:
        caveats.append(
            "Running without a generation model — this is a verbatim excerpt, not a composed answer."
        )

    # --- family 3: this should not be settled by a bot ---------------
    if sensitive:
        reasons.append("sensitive_intent")
    if wants_human:
        reasons.append("user_requested_human")
    if _is_frustrated(question):
        reasons.append("user_frustration")
        priority = _raise(priority)
    if prior_failures >= 2:
        reasons.append("repeated_failure")
        priority = _raise(priority)

    # Conflict: only escalate when the model did not resolve it. A detected
    # disagreement that the answer explains ("policy says 14 days; the 7-day
    # figure is from a withdrawn article") is a *good* answer, not a failure.
    if retrieval.conflicts or generation.conflict_detected:
        if generation.conflict_explanation and generation.answerable:
            caveats.append("Sources disagreed; answered from the current, highest-authority policy.")
        else:
            reasons.append("unresolved_conflict")

    # Freshness: flag rather than block. Stale is a reason to caveat and to
    # fix the corpus, not a reason to refuse a learner an answer.
    stale = [c.doc_id for c in retrieval.candidates if c.is_stale]
    if stale:
        caveats.append(
            f"Based on {'an article' if len(stale) == 1 else 'articles'} not reviewed in over "
            f"{cfg.stale_after_days // 30} months ({', '.join(sorted(set(stale)))})."
        )
        reasons.append("stale_source")

    # --- resolve ------------------------------------------------------
    blocking = {
        "no_relevant_context",
        "low_retrieval_confidence",
        "answer_not_supported_by_sources",
        "invalid_citation",
        "retired_claim_leak",
        "unsafe_information_request",
        "low_groundedness",
        "unresolved_conflict",
        "sensitive_intent",
        "user_requested_human",
        "model_unavailable",
        "repeated_failure",
    }
    hit = [r for r in reasons if r in blocking]

    if hit:
        # Out-of-domain questions get a polite abstention, not a support
        # ticket. Opening a billing case because someone asked about sourdough
        # would be worse than saying "not something I can help with".
        #
        # Once retrieval has failed, everything downstream is a *consequence*
        # of that failure rather than independent evidence: the generator
        # refuses, and the verifier then correctly reports that the refusal is
        # not grounded in LearnForge policy. Letting those derived signals vote
        # turned every out-of-scope question into an escalation — the live eval
        # caught it on "what's a good sourdough starter recipe". Verifying a
        # non-answer is meaningless, so they are excluded here.
        retrieval_failed = {"no_relevant_context", "low_retrieval_confidence"} & set(hit)
        consequential = {
            "answer_not_supported_by_sources",
            "low_groundedness",
            "unsupported_claims",
            "uncited_answer",
        }
        independent = [r for r in hit if r not in consequential and r not in retrieval_failed]

        if (
            retrieval_failed
            and not independent
            and not sensitive
            and not _IN_DOMAIN.search(question)
        ):
            action = "abstained"
        else:
            action = "escalated"
    elif caveats:
        action = "answered_with_caveat"
    else:
        action = "answered"

    return Decision(
        action=action,
        confidence=confidence,
        reason_codes=_dedupe(reasons),
        intent=intent,
        sensitive=sensitive,
        queue=queue,
        priority=priority,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# handoff artefact
# ---------------------------------------------------------------------------

REASON_TEXT = {
    "no_relevant_context": "nothing in the knowledge base matched the question",
    "low_retrieval_confidence": "the best matching article was only weakly relevant",
    "ambiguous_sources": "several articles matched equally well",
    "answer_not_supported_by_sources": "the sources did not support a complete answer",
    "invalid_citation": "the draft answer cited a document that was not retrieved",
    "retired_claim_leak": "the draft answer restated a withdrawn policy as current",
    "unsafe_information_request": "the draft answer asked for information we must never request",
    "low_groundedness": "the factuality check could not support parts of the draft answer",
    "unresolved_conflict": "the knowledge base contradicts itself on this point",
    "stale_source": "the governing article has not been reviewed recently",
    "sensitive_intent": "this request type is always handled by a person",
    "user_requested_human": "the learner asked for a human",
    "user_frustration": "the learner is frustrated",
    "repeated_failure": "earlier turns in this conversation already failed",
    "model_unavailable": "the answering model was unavailable",
    "uncited_answer": "the draft answer made claims without citing a source",
    "unsupported_claims": "the factuality check rejected specific claims",
}


def build_escalation_payload(
    *,
    session_id: str,
    message_id: int | None,
    question: str,
    decision: Decision,
    retrieval: RetrievalResult,
    generation: Generation,
    verification: Verification,
    slots: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the work item a human agent picks up.

    Written for the agent, not for the logs: what the learner wants, what has
    already been established, why the bot stopped, and what it would have said
    so the agent can correct rather than start over.
    """
    reasons = [REASON_TEXT.get(r, r) for r in decision.reason_codes]
    established = (
        "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in slots.items() if v) or "nothing yet"
    )
    summary_lines = [
        f"Learner asked: {question}",
        f"Detected intent: {decision.intent.replace('_', ' ')}",
        f"Already established: {established}",
        f"Handing over because: {'; '.join(reasons) if reasons else 'confidence below threshold'}",
        f"Knowledge-base articles consulted: {', '.join(retrieval.doc_ids) or 'none'}",
    ]
    if retrieval.conflicts:
        for conflict in retrieval.conflicts:
            detail = "; ".join(
                f"{v} {conflict['unit']}(s) per {', '.join(docs)}"
                for v, docs in conflict["sources"].items()
            )
            summary_lines.append(f"Source disagreement to resolve: {detail}")
    if verification.unsupported_claims:
        summary_lines.append(
            "Claims the checker rejected: " + "; ".join(verification.unsupported_claims[:3])
        )

    return {
        "session_id": session_id,
        "message_id": message_id,
        "queue": decision.queue,
        "priority": decision.priority,
        "reason_codes": json.dumps(decision.reason_codes),
        "user_intent": decision.intent,
        "summary": "\n".join(summary_lines),
        "attempted_answer": generation.answer or "",
        "context_snapshot": json.dumps(
            {
                "slots": slots,
                "confidence": decision.confidence,
                "retrieval": retrieval.diagnostics(),
                "verification": verification.as_dict(),
                "turns": len(history),
            },
            ensure_ascii=False,
        ),
    }


def handoff_message(decision: Decision) -> str:
    """What the learner sees when we escalate. Never blames them, never
    pretends the bot succeeded, and says what happens next."""
    if "user_requested_human" in decision.reason_codes:
        opening = "Of course — I'm passing you to a human colleague now."
    elif decision.sensitive:
        opening = (
            "This one needs a human on it. Requests involving payments, account ownership or "
            "a disputed charge are always handled by a person, not by me."
        )
    elif "unresolved_conflict" in decision.reason_codes:
        opening = (
            "I've found conflicting information in our help centre on this, and I don't want to "
            "give you an answer that turns out to be wrong."
        )
    elif {"no_relevant_context", "low_retrieval_confidence"} & set(decision.reason_codes):
        opening = "I couldn't find anything in our help centre that reliably answers this."
    else:
        opening = "I'm not confident enough in what I found to answer this myself."

    return (
        f"{opening} I've opened a case for our {decision.queue.replace('_', ' ')} team with "
        "everything you've told me so far, so you won't need to repeat yourself."
    )


def abstain_message() -> str:
    return (
        "I can only help with LearnForge courses, accounts, billing and technical issues, and I "
        "couldn't find anything relevant to this in our help centre. If it is about your "
        "LearnForge account, tell me a bit more and I'll take another look."
    )


# ---------------------------------------------------------------------------


def _top_source_was_used(retrieval: RetrievalResult, generation: Generation) -> bool:
    """Did the answer actually rest on the highest-ranked document?"""
    docs = retrieval.doc_ids
    return bool(docs) and docs[0] in set(generation.citations)


def _is_frustrated(text: str) -> bool:
    if _FRUSTRATION.search(text):
        return True
    if text.count("!") >= 2:
        return True
    shouted = _SHOUTING.findall(text)
    return len(shouted) >= 2


_PRIORITY_LADDER = ["low", "normal", "high", "urgent"]


def _raise(priority: str) -> str:
    try:
        return _PRIORITY_LADDER[min(_PRIORITY_LADDER.index(priority) + 1, 3)]
    except ValueError:
        return priority


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
