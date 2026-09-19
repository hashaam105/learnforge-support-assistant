-- ===========================================================================
-- LearnForge Support Assistant — canonical data schema
--
-- Engine: SQLite 3 (FTS5 required). The schema is written in portable SQL so
-- it maps 1:1 onto Postgres + pgvector at scale; docs/DATA_SCHEMA.md carries
-- the migration notes and the pgvector/Qdrant equivalent of each table.
--
-- Design rules encoded here:
--   1. documents = one row per authored source entry (FAQ-01, POLICY-02, ...)
--   2. chunks    = the retrieval unit; hot filter columns are denormalised
--                  onto it so retrieval never needs a join
--   3. embeddings live in their own table keyed by (chunk_id, model) so a
--      model swap is additive and reversible, never a destructive rewrite
--   4. deprecated_claims quarantines self-annotated stale statements so they
--      can be surfaced as "this used to be true" but never served as current
--   5. conversations/messages/escalations make multi-turn state and every
--      answer decision auditable after the fact
-- ===========================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- 1. DOCUMENTS — one row per source entry
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT    PRIMARY KEY,           -- 'FAQ-02', 'POLICY-10', 'TICKET-07'
    source_type     TEXT    NOT NULL               -- drives authority + prompt framing
                            CHECK (source_type IN ('policy', 'faq', 'ticket')),
    title           TEXT    NOT NULL,              -- 'Can I get a refund for a course?'
    source_path     TEXT    NOT NULL,              -- data/learnforge-knowledge-base/faqs.md
    source_uri      TEXT,                          -- human-resolvable link, shown in citations

    -- Authority: the single most important field for conflict resolution.
    -- 1 = policy (normative)  2 = faq (explanatory)  3 = ticket (anecdotal)
    -- A ticket transcript is evidence of what ONE agent said once. It is never
    -- policy, and the generator is forbidden from citing tier 3 as authority.
    authority_tier  INTEGER NOT NULL CHECK (authority_tier BETWEEN 1 AND 3),

    -- Freshness. effective_date is normalised ISO-8601; date_label keeps the
    -- raw string ('Last reviewed: February 2026') for display and debugging.
    effective_date  TEXT,                          -- '2026-02-01'
    date_label      TEXT,
    has_explicit_date INTEGER NOT NULL DEFAULT 0,  -- undated docs are a staleness risk

    ticket_status   TEXT                           -- tickets only
                            CHECK (ticket_status IN
                              ('resolved','escalated','pending','awaiting_info', NULL)),

    topics          TEXT    NOT NULL DEFAULT '[]', -- JSON array, controlled vocabulary
    raw_text        TEXT    NOT NULL,

    -- Idempotent re-ingest: unchanged content_hash => skip re-chunk and re-embed.
    content_hash    TEXT    NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,    -- bumped when content_hash changes
    ingested_at     TEXT    NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1     -- soft delete: retracted docs stop
                                                   -- being retrieved without losing the
                                                   -- audit trail of past citations
);

CREATE INDEX IF NOT EXISTS idx_documents_type      ON documents(source_type, is_active);
CREATE INDEX IF NOT EXISTS idx_documents_authority ON documents(authority_tier, effective_date DESC);


-- ---------------------------------------------------------------------------
-- 2. CHUNKS — the retrieval unit
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT    PRIMARY KEY,           -- 'POLICY-02#c1'
    doc_id          TEXT    NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,              -- position within the document
    text            TEXT    NOT NULL,              -- the embedded and cited span
    heading         TEXT,                          -- parent heading, prepended at embed
                                                   -- time so orphan chunks keep context
    token_estimate  INTEGER NOT NULL,

    -- Denormalised from documents: retrieval filters and re-scores on these
    -- on every query and must not pay for a join.
    source_type     TEXT    NOT NULL,
    authority_tier  INTEGER NOT NULL,
    effective_date  TEXT,
    topics          TEXT    NOT NULL DEFAULT '[]',

    -- Set when the chunk contains a self-annotated stale statement
    -- ('That wording is outdated', 'should not be treated as current').
    -- Such chunks are down-ranked and force a caveat in the answer.
    has_deprecation_notice INTEGER NOT NULL DEFAULT 0,

    content_hash    TEXT    NOT NULL,
    ingested_at     TEXT    NOT NULL,
    UNIQUE (doc_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc    ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_filter ON chunks(source_type, authority_tier);


-- ---------------------------------------------------------------------------
-- 3. EMBEDDINGS — keyed by (chunk_id, model)
--    A separate table means re-embedding with a new model is an INSERT, not a
--    migration. Two models can coexist while you A/B them, and rollback is a
--    DELETE of one model's rows.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id        TEXT    NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model           TEXT    NOT NULL,              -- 'text-embedding-004'
    dim             INTEGER NOT NULL,              -- 768
    vector          BLOB    NOT NULL,              -- float32 LE, L2-normalised
                                                   -- (normalised => cosine == dot product)
    created_at      TEXT    NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);


-- ---------------------------------------------------------------------------
-- 4. DEPRECATED_CLAIMS — quarantined stale statements
--    The corpus self-documents its own rot ('An older 2024 article listed
--    Internet Explorer as a supported browser. That information is obsolete').
--    Extracting these at ingest time is what stops the assistant from
--    confidently serving a retired policy as current.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS deprecated_claims (
    claim_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT    NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_id        TEXT    REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    claim_text      TEXT    NOT NULL,              -- the full sentence(s)
    marker          TEXT    NOT NULL,              -- phrase that triggered detection
    superseded_by   TEXT,                          -- doc_id carrying the current rule
    detected_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deprecated_doc ON deprecated_claims(doc_id);


-- ---------------------------------------------------------------------------
-- 5. CHUNKS_FTS — lexical (BM25) half of hybrid retrieval
--    Kept in sync by ingest. Gives exact-term recall that dense vectors lose:
--    order numbers, '14 days', 'CVV', 'FAQ-02', 'Atomic Orbitals'.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    heading,
    topics,
    tokenize = 'porter unicode61'
);


-- ---------------------------------------------------------------------------
-- 6. CONVERSATIONS / MESSAGES — multi-turn state and decision audit trail
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    session_id      TEXT    PRIMARY KEY,
    created_at      TEXT    NOT NULL,
    last_active_at  TEXT    NOT NULL,
    summary         TEXT    NOT NULL DEFAULT '',   -- rolling summary, keeps long chats
                                                   -- inside the context budget
    slots           TEXT    NOT NULL DEFAULT '{}', -- JSON: {"course_name":"Biology
                                                   -- Essentials","order_id":null,
                                                   -- "platform":"ios"} — this is what
                                                   -- resolves "the biology one" on turn 4
    turn_count      INTEGER NOT NULL DEFAULT 0,
    escalated       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    turn_index      INTEGER NOT NULL,
    role            TEXT    NOT NULL CHECK (role IN ('user','assistant')),
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,

    -- Assistant-turn diagnostics. Every field here exists so that a bad answer
    -- can be explained after the fact without re-running the pipeline.
    standalone_query    TEXT,                      -- the de-referenced query actually searched
    retrieved_chunk_ids TEXT,                      -- JSON array, in final ranked order
    citations           TEXT,                      -- JSON array of doc_ids the answer used
    retrieval_score     REAL,                      -- fused top-1 score
    score_margin        REAL,                      -- top1 - top2
    groundedness        REAL,                      -- verifier output, 0..1
    confidence          REAL,                      -- composite, 0..1
    action              TEXT CHECK (action IN
                          ('answered','answered_with_caveat','abstained','escalated', NULL)),
    reason_codes        TEXT,                      -- JSON array, why that action was taken
    latency_ms          INTEGER,
    model               TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_messages_action  ON messages(action, created_at);


-- ---------------------------------------------------------------------------
-- 7. ESCALATIONS — the handoff artefact
--    An escalation is not "the bot gave up". It is a structured work item a
--    human agent can action without re-reading the whole transcript.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id   TEXT    PRIMARY KEY,
    session_id      TEXT    NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    message_id      INTEGER REFERENCES messages(message_id) ON DELETE SET NULL,
    created_at      TEXT    NOT NULL,
    queue           TEXT    NOT NULL CHECK (queue IN
                      ('billing','trust_and_safety','content','accessibility','general')),
    priority        TEXT    NOT NULL CHECK (priority IN ('low','normal','high','urgent')),
    reason_codes    TEXT    NOT NULL,              -- JSON array, e.g. ["policy_conflict"]
    user_intent     TEXT    NOT NULL,
    summary         TEXT    NOT NULL,              -- agent-facing brief
    attempted_answer TEXT,                         -- what the bot would have said, kept so a
                                                   -- human can see what the learner nearly got
    context_snapshot TEXT   NOT NULL,              -- JSON: slots + cited chunks + scores
    resolved        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_escalations_queue ON escalations(queue, resolved, created_at);


-- ---------------------------------------------------------------------------
-- 8. INGEST_RUNS — freshness and reproducibility bookkeeping
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id          TEXT    PRIMARY KEY,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    kb_path         TEXT    NOT NULL,
    embedding_model TEXT,
    docs_seen       INTEGER NOT NULL DEFAULT 0,
    docs_changed    INTEGER NOT NULL DEFAULT 0,
    chunks_written  INTEGER NOT NULL DEFAULT 0,
    vectors_written INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'running'
);
