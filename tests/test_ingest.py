"""Ingestion: parsing, metadata extraction, chunking, deprecation quarantine."""

from __future__ import annotations

import json

from app.ingest import (
    extract_deprecated_claims,
    chunk_document,
    infer_topics,
    parse_file,
    run_ingest,
)


def test_corpus_parses_completely(store):
    stats = store.stats()
    assert stats["documents"] == 40
    assert stats["documents_by_type"] == {"faq": 15, "policy": 10, "ticket": 15}
    assert stats["chunks"] > stats["documents"]


def test_authority_tiers_follow_source_type(store):
    rows = dict(
        store.conn.execute("SELECT source_type, MIN(authority_tier) FROM documents GROUP BY 1")
    )
    assert rows == {"policy": 1, "faq": 2, "ticket": 3}


def test_dates_are_normalised(store):
    row = store.conn.execute(
        "SELECT effective_date, date_label FROM documents WHERE doc_id = 'POLICY-09'"
    ).fetchone()
    assert row["effective_date"] == "2026-06-01"
    assert "June 2026" in row["date_label"]


def test_undated_policies_are_flagged(store):
    # POLICY-05 and POLICY-08 carry no review date in the source corpus.
    undated = {
        r["doc_id"]
        for r in store.conn.execute(
            "SELECT doc_id FROM documents WHERE source_type='policy' AND has_explicit_date = 0"
        )
    }
    assert "POLICY-05" in undated and "POLICY-08" in undated


def test_ticket_status_is_classified(store):
    statuses = dict(
        store.conn.execute("SELECT doc_id, ticket_status FROM documents WHERE source_type='ticket'")
    )
    assert statuses["TICKET-03"] == "escalated"
    assert statuses["TICKET-08"] == "escalated"
    assert statuses["TICKET-01"] == "resolved"
    assert statuses["TICKET-06"] == "awaiting_info"


def test_deprecated_claims_cover_every_retired_policy(store):
    flagged = {r["doc_id"] for r in store.conn.execute("SELECT doc_id FROM deprecated_claims")}
    # Each of these documents explicitly retires one of its own statements.
    for doc_id in ("POLICY-01", "POLICY-02", "POLICY-04", "POLICY-05",
                   "POLICY-06", "POLICY-08", "POLICY-09", "POLICY-10"):
        assert doc_id in flagged, f"{doc_id} retires a statement that was not quarantined"


def test_deprecated_claim_captures_the_retired_rule_not_just_the_retraction(store):
    """'That information is obsolete' is useless without the sentence before it."""
    claim = store.conn.execute(
        "SELECT claim_text FROM deprecated_claims WHERE doc_id = 'POLICY-09'"
    ).fetchone()["claim_text"]
    assert "Internet Explorer" in claim
    assert "obsolete" in claim.lower()


def test_no_false_positive_deprecations(store):
    """Ordinary prose containing 'no longer' must not be quarantined.

    FAQ-05 says "If you no longer have access to the email address..." — a user
    situation, not a withdrawn policy. An earlier marker list flagged it.
    """
    flagged = {r["doc_id"] for r in store.conn.execute("SELECT doc_id FROM deprecated_claims")}
    assert "FAQ-05" not in flagged
    assert "FAQ-13" not in flagged


def test_topics_are_specific(cfg):
    docs = {d.doc_id: d for d in parse_file(cfg.kb_dir / "faqs.md", cfg.kb_dir)}
    assert "refunds" in docs["FAQ-02"].topics
    assert "accessibility" in docs["FAQ-14"].topics
    # A refund FAQ should not be tagged as a technical document.
    assert "technical" not in docs["FAQ-02"].topics


def test_infer_topics_prefers_dominant_signal():
    text = "refund refund refund refund money back. The browser is mentioned once."
    assert infer_topics(text)[0] == "refunds"


def test_faq_chunks_carry_the_question_as_heading(cfg):
    docs = {d.doc_id: d for d in parse_file(cfg.kb_dir / "faqs.md", cfg.kb_dir)}
    chunks = chunk_document(docs["FAQ-02"])
    assert chunks
    assert all("FAQ-02" in c["heading"] for c in chunks)
    assert any("refund" in c["heading"].lower() for c in chunks)


def test_ticket_chunks_keep_user_and_agent_together(cfg):
    docs = {d.doc_id: d for d in parse_file(cfg.kb_dir / "tickets.md", cfg.kb_dir)}
    chunks = chunk_document(docs["TICKET-07"])
    joined = " ".join(c["text"] for c in chunks)
    assert "USER:" in joined and "AGENT:" in joined


def test_chunk_ids_are_stable_and_ordered(cfg):
    docs = {d.doc_id: d for d in parse_file(cfg.kb_dir / "policies.md", cfg.kb_dir)}
    chunks = chunk_document(docs["POLICY-02"])
    assert [c["chunk_id"] for c in chunks] == [f"POLICY-02#c{i}" for i in range(len(chunks))]


def test_reingest_is_idempotent(cfg, store):
    """Unchanged content must not be re-chunked or re-embedded.

    This is the freshness mechanism: ingest is safe to run on a schedule
    because it is content-hashed.
    """
    before = store.stats()
    result = run_ingest(cfg=cfg, store=store, verbose=False)
    assert result["docs_seen"] == 40
    assert result["docs_changed"] == 0
    assert result["chunks_written"] == 0
    assert store.stats()["chunks"] == before["chunks"]


def test_force_reingest_rewrites_everything(cfg, store):
    result = run_ingest(cfg=cfg, store=store, force=True, verbose=False)
    assert result["docs_changed"] == 40
    assert result["chunks_written"] == store.stats()["chunks"]


def test_deprecation_flag_is_on_the_right_chunk(cfg):
    docs = {d.doc_id: d for d in parse_file(cfg.kb_dir / "policies.md", cfg.kb_dir)}
    chunks = chunk_document(docs["POLICY-09"])
    flagged = [c for c in chunks if c["has_deprecation_notice"]]
    assert flagged
    assert all("Internet Explorer" in c["text"] for c in flagged)
    claims = extract_deprecated_claims(docs["POLICY-09"], chunks)
    assert claims and claims[0]["chunk_id"] in {c["chunk_id"] for c in flagged}


def test_topics_stored_as_json(store):
    row = store.conn.execute("SELECT topics FROM chunks LIMIT 1").fetchone()
    assert isinstance(json.loads(row["topics"]), list)
