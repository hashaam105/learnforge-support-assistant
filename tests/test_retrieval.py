"""Retrieval: fusion, metadata re-scoring, conflict detection, safety of FTS input."""

from __future__ import annotations

import pytest

from app.retrieval import Candidate, Retriever, detect_conflicts
from app.text import content_terms


@pytest.fixture(scope="module")
def retriever(store, cfg) -> Retriever:
    return Retriever(store, cfg)


# --- query handling --------------------------------------------------------


def test_stopwords_are_stripped():
    assert content_terms("what is the capital of France") == ["capital", "france"]


def test_apostrophes_do_not_break_fts(store):
    """A bare apostrophe is an FTS5 delimiter; it used to silently return zero rows."""
    hits = store.lexical_search("I don't recognise this charge and can't log in", 10)
    assert hits


@pytest.mark.parametrize(
    "query",
    ['refund" OR 1=1 --', "course*", "^refund", "(((", "NEAR(a b)", "-'"],
)
def test_fts_syntax_cannot_be_injected(store, query):
    """Malformed or adversarial input must degrade to no results, never raise."""
    assert isinstance(store.lexical_search(query, 5), list)


# --- ranking ---------------------------------------------------------------


def test_policy_outranks_faq_for_the_governing_rule(retriever):
    result = retriever.retrieve("How long do I have to request a refund on a course?")
    assert result.doc_ids[0] == "POLICY-02"


def test_out_of_scope_query_retrieves_nothing(retriever):
    result = retriever.retrieve("What is the capital of France?")
    assert result.mode == "empty"
    assert result.top_score == 0.0


def test_in_scope_query_clears_the_confidence_floor(retriever, cfg):
    result = retriever.retrieve("I forgot my password, how do I reset it?")
    assert result.top_score >= cfg.min_score_for(result.mode)
    assert "FAQ-05" in result.doc_ids


def test_no_single_document_monopolises_the_context(retriever):
    result = retriever.retrieve("refund policy for a course purchase")
    counts: dict[str, int] = {}
    for c in result.candidates:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
    assert max(counts.values()) <= 2


def test_best_lexical_hit_always_survives_rescoring(retriever, store):
    """The recall guard: metadata weights must not evict the top raw match."""
    query = "I've been charged twice for the same course"
    best_chunk = store.lexical_search(query, 1)[0][0]
    result = retriever.retrieve(query)
    assert best_chunk in {c.chunk_id for c in result.candidates}


def test_chunk_correcting_a_retired_claim_is_not_demoted(retriever):
    """POLICY-05's correction chunk IS the answer to the five-quizzes question."""
    result = retriever.retrieve("Does every course need at least five quizzes?")
    assert "POLICY-05" in result.doc_ids


# --- conflict detection ----------------------------------------------------


def _candidate(doc_id: str, text: str) -> Candidate:
    return Candidate(
        chunk_id=f"{doc_id}#c0", doc_id=doc_id, text=text, heading="",
        source_type="policy", authority_tier=1, effective_date=None, date_label=None,
        ticket_status=None, topics=[], has_deprecation_notice=False,
        source_uri="", title=doc_id,
    )


def test_conflicting_refund_windows_are_detected():
    conflicts = detect_conflicts([
        _candidate("POLICY-02", "The standard refund period is generally 14 days after purchase."),
        _candidate("OTHER-01", "The standard refund period is generally 30 days after purchase."),
    ])
    assert conflicts
    assert conflicts[0]["unit"] == "day"
    assert conflicts[0]["values"] == [14, 30]


def test_unrelated_quantities_are_not_a_conflict():
    """'14 days' for refunds and '30 days' for something else is not a contradiction."""
    conflicts = detect_conflicts([
        _candidate("POLICY-02", "The standard refund period is generally 14 days after purchase."),
        _candidate("POLICY-09", "Browser support is reviewed every 30 days by the platform team."),
    ])
    assert conflicts == []


def test_retired_value_is_not_treated_as_a_live_conflict():
    """POLICY-02 states both 14 days and the withdrawn 7-day figure."""
    candidates = [
        _candidate("POLICY-02", "The standard refund period is generally 14 days. Older "
                                "help-center documentation referenced a 7-day refund period."),
    ]
    claims = [{
        "doc_id": "POLICY-02",
        "claim_text": "Older help-center documentation referenced a 7-day refund period "
                      "for all digital products.",
    }]
    assert detect_conflicts(candidates, claims) == []
    # Without the quarantine it WOULD look like a contradiction:
    assert detect_conflicts(candidates, []) != []


def test_user_asserted_figure_conflicts_with_policy():
    """TICKET-03: the learner holds us to a 30-day guarantee we cannot support."""
    candidates = [
        _candidate("POLICY-02", "The standard refund period is generally 14 days for course purchases.")
    ]
    conflicts = detect_conflicts(
        candidates, [], query="your site promised a 30 day refund period on this course"
    )
    assert conflicts
    assert 30 in conflicts[0]["values"]
