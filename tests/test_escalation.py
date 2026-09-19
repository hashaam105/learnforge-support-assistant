"""The escalation gate: intent routing and the abstain/answer/escalate decision."""

from __future__ import annotations

import pytest

from app.escalation import (
    ALWAYS_ESCALATE,
    classify_intent,
    composite_confidence,
    decide,
)
from app.generator import Generation
from app.retrieval import Candidate, RetrievalResult
from app.verifier import Verification


def _retrieval(*, top_score=0.8, margin=0.1, docs=("POLICY-02",), conflicts=()) -> RetrievalResult:
    candidates = [
        Candidate(
            chunk_id=f"{d}#c0", doc_id=d, text="text", heading="", source_type="policy",
            authority_tier=1, effective_date="2026-01-01", date_label="January 2026",
            ticket_status=None, topics=[], has_deprecation_notice=False,
            source_uri="", title=d, final_score=0.5,
        )
        for d in docs
    ]
    return RetrievalResult(
        query="q", candidates=candidates, mode="lexical-only",
        top_score=top_score, margin=margin, conflicts=list(conflicts),
    )


def _good_generation(**kw) -> Generation:
    defaults = dict(answer="You generally have 14 days [POLICY-02].", citations=["POLICY-02"],
                    answerable=True, self_confidence=0.8)
    defaults.update(kw)
    return Generation(**defaults)


def _clean_verification(**kw) -> Verification:
    v = Verification(groundedness=0.95, judge_ran=True)
    for key, value in kw.items():
        setattr(v, key, value)
    return v


# --- intent classification -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("There's a $79 charge I don't recognise", "fraud_or_unrecognised_charge"),
        ("I was charged twice for the same course", "duplicate_charge"),
        ("I bought it with my work email but my account is Gmail", "account_ownership"),
        ("My daughter completed it but the certificate has my name", "account_ownership"),
        ("Your website said 30 days when I bought it", "refund_dispute"),
        ("The chemistry video has no captions", "accessibility_barrier"),
        ("I'll take legal action over this", "legal_or_complaint"),
        ("How do I reset my password?", "access_issue"),
    ],
)
def test_intent_classification(text, expected):
    assert classify_intent(text) == expected


def test_money_and_identity_intents_are_always_escalated():
    for intent in ("fraud_or_unrecognised_charge", "duplicate_charge",
                   "account_ownership", "refund_dispute"):
        assert intent in ALWAYS_ESCALATE


# --- the gate --------------------------------------------------------------


def test_clean_answer_is_answered(cfg):
    decision = decide(
        question="How long do I have to request a refund on a course?",
        retrieval=_retrieval(), generation=_good_generation(),
        verification=_clean_verification(), cfg=cfg,
    )
    assert decision.action == "answered"
    assert decision.confidence > 0.7


def test_empty_retrieval_out_of_domain_abstains(cfg):
    decision = decide(
        question="What's a good sourdough starter recipe?",
        retrieval=_retrieval(top_score=0.0, margin=0.0, docs=()),
        generation=Generation(answer="", answerable=False),
        verification=Verification(groundedness=0.0), cfg=cfg,
    )
    assert decision.action == "abstained"


def test_empty_retrieval_in_domain_escalates(cfg):
    """No answer about a LearnForge charge is a support case, not a shrug."""
    decision = decide(
        question="Why was my LearnForge course refund never processed?",
        retrieval=_retrieval(top_score=0.0, margin=0.0, docs=()),
        generation=Generation(answer="", answerable=False),
        verification=Verification(groundedness=0.0), cfg=cfg,
    )
    assert decision.action == "escalated"


def test_sensitive_intent_escalates_despite_perfect_confidence(cfg):
    decision = decide(
        question="There's a $79 charge on my card that I never authorised",
        retrieval=_retrieval(top_score=0.99, margin=0.5),
        generation=_good_generation(self_confidence=1.0),
        verification=_clean_verification(groundedness=1.0), cfg=cfg,
    )
    assert decision.action == "escalated"
    assert decision.queue == "trust_and_safety"
    assert decision.priority == "urgent"


def test_explicit_human_request_is_honoured(cfg):
    decision = decide(
        question="How do I reset my password?",
        retrieval=_retrieval(), generation=_good_generation(),
        verification=_clean_verification(), wants_human=True, cfg=cfg,
    )
    assert decision.action == "escalated"
    assert "user_requested_human" in decision.reason_codes


def test_invalid_citation_blocks_the_answer(cfg):
    decision = decide(
        question="How long do I have to request a refund?",
        retrieval=_retrieval(),
        generation=_good_generation(citations=["POLICY-99"]),
        verification=_clean_verification(invalid_citations=["POLICY-99"]), cfg=cfg,
    )
    assert decision.action == "escalated"
    assert decision.confidence == 0.0


def test_retired_claim_leak_blocks_the_answer(cfg):
    decision = decide(
        question="How long do I have to request a refund?",
        retrieval=_retrieval(),
        generation=_good_generation(answer="You have 7 days to request a refund."),
        verification=_clean_verification(retired_claim_leaks=["POLICY-02: '7 day' asserted as current"]),
        cfg=cfg,
    )
    assert decision.action == "escalated"


def test_low_groundedness_blocks_the_answer(cfg):
    decision = decide(
        question="How do I reset my password?",
        retrieval=_retrieval(), generation=_good_generation(),
        verification=_clean_verification(groundedness=0.3, unsupported_claims=["made up"]),
        cfg=cfg,
    )
    assert decision.action == "escalated"
    assert "low_groundedness" in decision.reason_codes


def test_explained_conflict_is_answered_with_a_caveat(cfg):
    """A conflict the model resolved is good behaviour, not a failure."""
    decision = decide(
        question="How long do I have to request a refund?",
        retrieval=_retrieval(conflicts=[{"unit": "day", "values": [7, 14], "sources": {}}]),
        generation=_good_generation(
            conflict_detected=True,
            conflict_explanation="The 7-day figure comes from a withdrawn article.",
        ),
        verification=_clean_verification(), cfg=cfg,
    )
    assert decision.action == "answered_with_caveat"


def test_unexplained_conflict_escalates(cfg):
    decision = decide(
        question="How long do I have to request a refund?",
        retrieval=_retrieval(conflicts=[{"unit": "day", "values": [14, 30], "sources": {}}]),
        generation=_good_generation(conflict_detected=True, conflict_explanation=""),
        verification=_clean_verification(), cfg=cfg,
    )
    assert decision.action == "escalated"
    assert "unresolved_conflict" in decision.reason_codes


def test_frustration_raises_priority(cfg):
    calm = decide(
        question="I want a refund for my course",
        retrieval=_retrieval(), generation=_good_generation(),
        verification=_clean_verification(), cfg=cfg,
    )
    angry = decide(
        question="This is absolutely ridiculous, I want a refund for my course!!",
        retrieval=_retrieval(), generation=_good_generation(),
        verification=_clean_verification(), cfg=cfg,
    )
    assert "user_frustration" in angry.reason_codes
    assert angry.priority != calm.priority


def test_repeated_failure_escalates(cfg):
    decision = decide(
        question="How do I reset my password?",
        retrieval=_retrieval(), generation=_good_generation(),
        verification=_clean_verification(), prior_failures=2, cfg=cfg,
    )
    assert decision.action == "escalated"
    assert "repeated_failure" in decision.reason_codes


def test_stale_source_caveats_but_does_not_block(cfg):
    retrieval = _retrieval()
    retrieval.candidates[0].is_stale = True
    decision = decide(
        question="How do I reset my password?",
        retrieval=retrieval, generation=_good_generation(),
        verification=_clean_verification(), cfg=cfg,
    )
    assert decision.action == "answered_with_caveat"
    assert any("not reviewed" in c for c in decision.caveats)


# --- confidence ------------------------------------------------------------


def test_hard_failure_zeroes_confidence():
    score = composite_confidence(
        _retrieval(top_score=0.99, margin=0.9),
        Verification(groundedness=1.0, invalid_citations=["X-1"]),
        _good_generation(self_confidence=1.0),
    )
    assert score == 0.0


def test_confidence_tracks_its_inputs():
    high = composite_confidence(_retrieval(top_score=0.9, margin=0.2),
                                Verification(groundedness=1.0, judge_ran=True),
                                _good_generation(self_confidence=0.9))
    low = composite_confidence(_retrieval(top_score=0.35, margin=0.01),
                               Verification(groundedness=0.4, judge_ran=True),
                               _good_generation(self_confidence=0.3))
    assert high > low
    assert 0.0 <= low <= high <= 1.0
