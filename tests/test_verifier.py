"""Verification: the deterministic checks that do not need a model."""

from __future__ import annotations

from app.generator import Generation
from app.providers.offline import OfflineLLM
from app.retrieval import Candidate, RetrievalResult
from app.verifier import Verifier


def _retrieval(docs=("POLICY-02",), claims=None) -> RetrievalResult:
    candidates = [
        Candidate(
            chunk_id=f"{d}#c0", doc_id=d,
            text="For individual course purchases, LearnForge's standard refund period is "
                 "generally 14 days. Older help-center documentation referenced a 7-day "
                 "refund period for all digital products.",
            heading="", source_type="policy", authority_tier=1,
            effective_date="2026-01-01", date_label="January 2026", ticket_status=None,
            topics=[], has_deprecation_notice=True, source_uri="", title=d,
        )
        for d in docs
    ]
    return RetrievalResult(
        query="refund window", candidates=candidates, mode="lexical-only",
        top_score=0.8, margin=0.1,
        deprecated_claims=claims
        if claims is not None
        else [{
            "doc_id": "POLICY-02",
            "claim_text": "Older help-center documentation referenced a 7-day refund period "
                          "for all digital products.",
        }],
    )


def _verifier(cfg) -> Verifier:
    return Verifier(OfflineLLM(cfg), cfg)


def test_citation_to_a_document_that_was_never_retrieved_is_caught(cfg):
    result = _verifier(cfg).verify(
        Generation(answer="You get 14 days [POLICY-11].", citations=["POLICY-11"]),
        _retrieval(),
    )
    assert result.invalid_citations == ["POLICY-11"]
    assert result.hard_failure


def test_valid_citation_passes(cfg):
    result = _verifier(cfg).verify(
        Generation(answer="You generally get 14 days [POLICY-02].", citations=["POLICY-02"]),
        _retrieval(),
    )
    assert result.invalid_citations == []
    assert not result.hard_failure


def test_inline_citations_are_validated_even_if_omitted_from_the_list(cfg):
    result = _verifier(cfg).verify(
        Generation(answer="See [POLICY-77] for details on the refund window and eligibility.",
                   citations=[]),
        _retrieval(),
    )
    assert "POLICY-77" in result.invalid_citations


def test_retired_claim_asserted_as_current_is_caught(cfg):
    """The hard case: '7 days' is real retrieved text, so entailment alone passes it."""
    result = _verifier(cfg).verify(
        Generation(answer="You have 7 days from purchase to request a refund [POLICY-02].",
                   citations=["POLICY-02"]),
        _retrieval(),
    )
    assert result.retired_claim_leaks
    assert result.hard_failure


def test_retired_claim_described_as_historical_is_allowed(cfg):
    result = _verifier(cfg).verify(
        Generation(
            answer="The current window is 14 days [POLICY-02]. An older help-centre article "
                   "mentioned 7 days, but that guidance is no longer current.",
            citations=["POLICY-02"],
        ),
        _retrieval(),
    )
    assert result.retired_claim_leaks == []


def test_hyphenated_form_of_a_retired_figure_is_caught(cfg):
    result = _verifier(cfg).verify(
        Generation(answer="LearnForge offers a 7-day refund window on all digital products.",
                   citations=["POLICY-02"]),
        _retrieval(),
    )
    assert result.retired_claim_leaks


def test_asking_for_a_security_code_is_caught(cfg):
    result = _verifier(cfg).verify(
        Generation(answer="Please provide your full card number and CVV so I can refund you.",
                   citations=["POLICY-02"]),
        _retrieval(),
    )
    assert result.unsafe_request
    assert result.hard_failure


def test_clarifying_question_is_not_flagged_as_uncited(cfg):
    result = _verifier(cfg).verify(
        Generation(answer="Which course was the charge for?", citations=[]),
        _retrieval(),
    )
    assert not result.uncited


def test_long_uncited_factual_answer_is_flagged(cfg):
    result = _verifier(cfg).verify(
        Generation(
            answer="Refunds are always processed within one hour and we never ask any "
                   "questions about your purchase history or your account at all.",
            citations=[],
        ),
        _retrieval(),
    )
    assert result.uncited
    assert "uncited_answer" in result.reason_codes()


def test_empty_answer_scores_zero(cfg):
    result = _verifier(cfg).verify(Generation(answer="   ", citations=[]), _retrieval())
    assert result.groundedness == 0.0


def test_hard_failure_caps_groundedness(cfg):
    result = _verifier(cfg).verify(
        Generation(answer="You have 7 days to request a refund [POLICY-02].",
                   citations=["POLICY-02"]),
        _retrieval(),
    )
    assert result.groundedness <= 0.25


# --- regressions found by the live evaluation ------------------------------


def test_current_guidance_sharing_a_noun_with_a_retired_claim_is_not_a_leak(cfg):
    """POLICY-04 retires "recommended downloading over cellular data", but its
    current recommendation mentions cellular data too. Treating the bare noun
    as diagnostic failed a correct answer."""
    retrieval = _retrieval(claims=[{
        "doc_id": "POLICY-04",
        "claim_text": "The previous mobile help article recommended downloading lessons over "
                      "cellular data. This recommendation has been removed.",
    }])
    result = _verifier(cfg).verify(
        Generation(
            answer="Use Wi-Fi for large offline downloads rather than cellular data, because "
                   "big downloads can use a lot of your data allowance [POLICY-04].",
            citations=["POLICY-02"],
        ),
        retrieval,
    )
    assert result.retired_claim_leaks == []


# --- named entities are derived, not enumerated ----------------------------


def test_named_entities_are_derived_from_any_claim(cfg):
    """A withdrawn policy this corpus has never seen must still be detectable."""
    from app.verifier import _named_entities

    claim = 'An older article listed Netscape Navigator as a supported browser.'
    assert "netscape navigator" in _named_entities(claim)


def test_replacement_wording_is_not_treated_as_retired(cfg):
    """POLICY-06 quotes BOTH the retired term and the one that replaced it."""
    from app.verifier import _named_entities

    claim = ('Archived documentation previously claimed that progress was saved "instantly." '
             'That wording has been replaced with "automatically synchronized".')
    assert "automatically synchronized" not in _named_entities(claim)


def test_lowercase_common_phrases_are_not_named_entities(cfg):
    """'cellular data' is vocabulary the current guidance uses too."""
    from app.verifier import _named_entities

    claim = ("The previous mobile help article recommended downloading lessons over "
             "cellular data. This recommendation has been removed.")
    assert not any("cellular" in t for t in _named_entities(claim))


def test_unseen_retired_product_leaks_are_caught(cfg):
    """End-to-end: a claim with a product name nothing hardcoded knows about."""
    retrieval = _retrieval(claims=[{
        "doc_id": "POLICY-99",
        "claim_text": "An older article listed Netscape Navigator as a supported browser. "
                      "That information is obsolete.",
    }])
    result = _verifier(cfg).verify(
        Generation(answer="You can use Netscape Navigator to view the course.",
                   citations=["POLICY-02"]),
        retrieval,
    )
    assert result.retired_claim_leaks
