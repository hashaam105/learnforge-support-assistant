"""End-to-end behaviour and the audit trail, all in keyless mode."""

from __future__ import annotations

import json

import pytest

from app.pipeline import SupportAssistant


@pytest.fixture(scope="module")
def assistant(store, cfg) -> SupportAssistant:
    return SupportAssistant(cfg=cfg, store=store)


def test_a_turn_produces_a_decision_and_a_reply(assistant):
    result = assistant.ask("How do I reset my password?")
    assert result.reply.strip()
    assert result.action in {"answered", "answered_with_caveat", "abstained", "escalated"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms >= 0


def test_out_of_scope_abstains_without_opening_a_case(assistant):
    before = len(assistant.store.list_escalations(100))
    result = assistant.ask("What's a good sourdough starter recipe?")
    assert result.action == "abstained"
    assert result.escalation_id is None
    assert len(assistant.store.list_escalations(100)) == before


def test_sensitive_request_opens_a_routed_case(assistant):
    result = assistant.ask("There's a $79 LearnForge charge I never authorised.")
    assert result.action == "escalated"
    assert result.escalation_id
    assert result.queue == "trust_and_safety"
    assert result.priority == "urgent"

    row = next(
        r for r in assistant.store.list_escalations(50)
        if r["escalation_id"] == result.escalation_id
    )
    # The handoff has to be actionable without re-reading the transcript.
    assert "Learner asked:" in row["summary"]
    assert "Handing over because:" in row["summary"]
    assert json.loads(row["reason_codes"])
    assert json.loads(row["context_snapshot"])["retrieval"]


def test_every_turn_is_auditable(assistant):
    result = assistant.ask("How long do I have to request a refund on a course?")
    row = assistant.store.conn.execute(
        "SELECT * FROM messages WHERE session_id = ? AND role = 'assistant'",
        (result.session_id,),
    ).fetchone()
    assert row["standalone_query"]
    assert json.loads(row["retrieved_chunk_ids"])
    assert row["action"] == result.action
    assert json.loads(row["reason_codes"]) == result.reason_codes
    assert row["retrieval_score"] is not None


def test_conversation_state_survives_across_turns(assistant):
    first = assistant.ask("I want a refund for the Biology Essentials course")
    assistant.ask("It's been 21 days", session_id=first.session_id)

    row = assistant.store.conn.execute(
        "SELECT slots, turn_count FROM conversations WHERE session_id = ?",
        (first.session_id,),
    ).fetchone()
    slots = json.loads(row["slots"])
    assert slots.get("course_name") == "Biology Essentials"
    assert row["turn_count"] >= 2


def test_citations_are_always_a_subset_of_what_was_retrieved(assistant):
    for question in [
        "How do I reset my password?",
        "Can I download courses to watch offline?",
        "What happens when my subscription expires?",
    ]:
        result = assistant.ask(question)
        assert set(result.citations) <= {s["doc_id"] for s in result.sources}


def test_card_details_never_reach_the_reply_or_the_log(assistant):
    result = assistant.ask("My card 4111 1111 1111 1111 CVV 123 was charged, refund me")
    assert "4111 1111 1111 1111" not in result.reply
    rows = assistant.store.conn.execute(
        "SELECT content, standalone_query FROM messages WHERE session_id = ?",
        (result.session_id,),
    ).fetchall()
    for row in rows:
        assert "4111 1111 1111 1111" not in (row["content"] or "")
        assert "4111 1111 1111 1111" not in (row["standalone_query"] or "")


def test_sources_are_returned_with_provenance(assistant):
    result = assistant.ask("Which browsers are supported?")
    assert result.sources
    for source in result.sources:
        assert source["doc_id"]
        assert source["source_type"] in {"policy", "faq", "ticket"}
        assert 1 <= source["authority_tier"] <= 3
        assert "excerpt" in source


def test_degraded_mode_is_declared_not_hidden(assistant):
    """Without a generation key the reply must say so rather than look authoritative."""
    result = assistant.ask("How do I reset my password?")
    capabilities = result.diagnostics["capabilities"]
    assert capabilities["llm_live"] is False
    assert result.diagnostics["generation"]["degraded"] is True


def test_diagnostics_expose_the_whole_decision_trail(assistant):
    result = assistant.ask("Can my brother use my account?")
    for section in ("context", "retrieval", "generation", "verification", "decision"):
        assert section in result.diagnostics
    assert "candidates" in result.diagnostics["retrieval"]
    assert "reason_codes" in result.diagnostics["decision"]


# --- provenance cards must show evidence a reader can check ----------------


def test_card_shows_the_chunk_that_supports_the_answer():
    """POLICY-02 states the 14-day rule in its first chunk, but its second
    chunk often outranks it on a general refund question. Showing the
    higher-ranked chunk put unrelated text under a 14-day claim."""
    from app.pipeline import _best_supporting_chunk
    from app.text import content_terms

    class _C:
        def __init__(self, text, score):
            self.text, self.final_score = text, score

    supporting = _C("For individual course purchases the standard refund period "
                    "is generally 14 days.", 0.40)
    higher_ranked = _C("Certain promotional bundles and third-party purchases may "
                       "have separate terms.", 0.90)
    answer_terms = set(content_terms("You generally have 14 days to request a refund "
                                     "for an individual course."))
    assert _best_supporting_chunk([higher_ranked, supporting], answer_terms) is supporting


def test_excerpt_is_centred_on_the_supporting_sentence():
    from app.pipeline import _evidence_excerpt
    from app.text import content_terms

    text = ("You can cancel an active subscription at any time through Account Settings. "
            "Cancellation normally prevents the next renewal. " + ("Filler sentence. " * 12) +
            "For individual course purchases the standard refund period is generally 14 days.")
    excerpt = _evidence_excerpt(text, set(content_terms("refund period is 14 days")))
    assert "14 days" in excerpt
    assert len(excerpt) < len(text)


def test_short_chunks_are_shown_whole():
    from app.pipeline import _evidence_excerpt

    text = "Refunds are generally available within 14 days."
    assert _evidence_excerpt(text, {"refund"}) == text


def test_excerpt_survives_an_abstention_with_no_answer_terms():
    from app.pipeline import _evidence_excerpt

    text = "A" * 400
    excerpt = _evidence_excerpt(text, set())
    assert excerpt.endswith("…") and len(excerpt) <= 261
