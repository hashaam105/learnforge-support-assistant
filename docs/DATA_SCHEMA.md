# Data Schema

The executable definition is [`schema.sql`](../schema.sql). This document
explains *why* each table and column exists, and how the design moves to
Postgres + pgvector when the corpus outgrows one file.

**Engine:** SQLite 3 with FTS5. See the README trade-offs for why a 40-document
corpus does not get a vector database.

---

## Overview

| Table | Grain | Purpose |
|---|---|---|
| `documents` | one authored entry (`FAQ-02`) | source of truth, authority, freshness |
| `chunks` | one retrieval unit (`POLICY-02#c1`) | what gets embedded, ranked and cited |
| `embeddings` | one `(chunk, model)` pair | dense vectors, model-versioned |
| `chunks_fts` | one chunk | BM25 lexical index |
| `deprecated_claims` | one withdrawn statement | quarantine for stale rules |
| `conversations` | one session | multi-turn state and slot memory |
| `messages` | one turn | full decision audit trail |
| `escalations` | one human handoff | the work item an agent picks up |
| `ingest_runs` | one indexing run | freshness and reproducibility |

---

## 1. `documents`

One row per authored entry in the knowledge base. The sample corpus yields
exactly 40: 15 FAQs, 10 policies, 15 ticket transcripts.

| Column | Type | Notes |
|---|---|---|
| `doc_id` | TEXT PK | `FAQ-02`, `POLICY-10`, `TICKET-07`. Stable, human-readable, and **the thing the model cites** — which is what makes citations verifiable. |
| `source_type` | TEXT | `policy` \| `faq` \| `ticket` |
| `title` | TEXT | from the source heading |
| `source_path`, `source_uri` | TEXT | provenance shown to the learner |
| `authority_tier` | INTEGER | `1` policy, `2` faq, `3` ticket |
| `effective_date` | TEXT | ISO-8601, parsed from prose |
| `date_label` | TEXT | raw string, e.g. `Last reviewed: February 2026` |
| `has_explicit_date` | INTEGER | undated documents are a distinct staleness risk |
| `ticket_status` | TEXT | `resolved` \| `escalated` \| `pending` \| `awaiting_info` |
| `topics` | TEXT | JSON array from a controlled vocabulary |
| `raw_text` | TEXT | unchunked body, for re-chunking without re-reading files |
| `content_hash` | TEXT | SHA-256 of the body |
| `version` | INTEGER | incremented when `content_hash` changes |
| `is_active` | INTEGER | soft delete |

### Why `authority_tier` is the most important column

The corpus contradicts itself on purpose, and the contradictions are not
random — they are *stratified*. TICKET-03 has an agent saying "our standard
course refund period is 14 days" in conversation. POLICY-02 states the rule.
FAQ-02 explains it. All three are lexically excellent matches for "how long do
I have to get a refund?".

A ticket transcript is evidence of **how one case was handled once**. It is not
policy, and an assistant that quotes it as policy is wrong even when it happens
to be accurate — because the next ticket it quotes will be TICKET-08, where the
agent escalated precisely because the policy was unclear.

`authority_tier` is therefore used three times in the pipeline:

1. **Ranking** — a gentle multiplicative boost in `app/retrieval.py`.
2. **Prompting** — each context block is labelled `PAST TICKET — one agent's
   handling of one case; NOT policy`, so the model can see the difference.
3. **Conflict resolution** — when sources disagree, the highest tier and most
   recent date wins, and the disagreement is stated to the learner.

### Why `effective_date` is parsed rather than stored as prose

Policies in the sample carry dates as free text — `Last reviewed: February
2026`, `Effective: December 2025`, `Updated: March 2026` — and two
(POLICY-05, POLICY-08) carry none at all. Normalising to ISO-8601 makes
`STALE_AFTER_DAYS` a real comparison instead of a string match, and
`has_explicit_date = 0` gives undated documents their own, smaller penalty.
An undated policy is not necessarily old; it is *unverifiable*, which is a
different problem and deserves a different weight.

---

## 2. `chunks`

The retrieval unit. 40 documents produce 66 chunks — most entries are a single
semantic unit and survive intact.

| Column | Type | Notes |
|---|---|---|
| `chunk_id` | TEXT PK | `POLICY-02#c1` — document id plus ordinal, so a chunk's provenance is readable without a join |
| `doc_id` | TEXT FK | cascade delete |
| `ordinal` | INTEGER | position within the document |
| `text` | TEXT | the span that is embedded and quoted |
| `heading` | TEXT | parent heading, plus the FAQ question |
| `token_estimate` | INTEGER | context budgeting |
| `source_type`, `authority_tier`, `effective_date`, `topics` | | **denormalised from `documents`** |
| `has_deprecation_notice` | INTEGER | chunk contains a withdrawn statement |
| `content_hash` | TEXT | per-chunk change detection |

### Why metadata is denormalised onto the chunk

Retrieval re-scores every candidate on `authority_tier` and `effective_date`
on every query. At this scale a join costs nothing; at a scale where it does,
the read path is the last place you want one. Denormalisation is safe here
because chunks are never updated in place — `replace_chunks` deletes and
reinserts, so the copies cannot drift.

### Why `heading` is stored separately from `text`

The heading carries the FAQ's *question*, and the question is the strongest
retrieval signal in the record: learners describe their problem the way the
question is phrased, not the way the answer resolves it. So the heading is
prepended **at embed time** to give the vector that signal — but it is not part
of `text`, so a citation quotes only the substantive body. Getting this
backwards either weakens retrieval or pollutes every quote with a heading.

### Why `has_deprecation_notice` does *not* affect ranking

An earlier version down-ranked chunks containing withdrawn statements. That
turned out to be exactly wrong. POLICY-05's second chunk reads:

> An older instructor guide stated that courses must contain at least five
> quizzes. **This is no longer a universal requirement.**

That chunk is the *best* answer to "do courses need five quizzes?" — it
contains both the retired rule and its correction. Penalising it promoted
passages that mention quizzes *without* the correction, which is the opposite
of the goal. The flag still travels with the chunk: it annotates the prompt and
arms the verifier. It just must not touch the score. The eval suite caught
this; `tests/test_retrieval.py::test_chunk_correcting_a_retired_claim_is_not_demoted`
keeps it caught.

---

## 3. `embeddings`

```sql
PRIMARY KEY (chunk_id, model)
vector BLOB  -- float32 little-endian, L2-normalised
```

Separate from `chunks`, keyed by model. Three consequences worth having:

- **Re-embedding is additive.** A new model is an `INSERT`, not a migration.
  Both models coexist while you A/B them; rollback is `DELETE WHERE model = ?`.
- **Vectors are L2-normalised at write time**, so cosine similarity reduces to
  a dot product at query time and the whole search is one matrix multiply.
  `gemini-embedding-001` truncated to 768 dimensions is *not* unit length, so
  the normalisation in `app/providers/gemini_embed.py` is load-bearing.
- **Storage is predictable.** 768 × 4 bytes = 3 KB per chunk; the whole sample
  corpus is under 200 KB of vectors.

---

## 4. `deprecated_claims` — the quarantine

The single most corpus-specific piece of this schema, and the one that stops
the most likely hallucination.

The knowledge base documents its own rot:

> Older help-center documentation referenced a **7-day refund period** for all
> digital products. That article remains accessible in some archived search
> results but **should not be treated as the current standard policy**.
> — POLICY-02

That sentence lives inside the highest-authority, most-retrieved document for
every refund question. A naive RAG pipeline retrieves POLICY-02, sees "7-day
refund period" in its context, and answers "you have 7 days" — and **every
standard groundedness check passes**, because the claim genuinely is supported
by the retrieved text.

So at ingest time each withdrawn statement is lifted into its own row:

| Column | Notes |
|---|---|
| `claim_text` | the retired rule *and* its retraction, so the pair stays legible |
| `marker` | the phrase that triggered detection, for auditing the detector |
| `superseded_by` | the document carrying the current rule |

These rows are then used twice:

1. The prompt gets a **RETIRED STATEMENTS** block listing them as things the
   model may describe as former wording but must never assert as current.
2. `app/verifier.py` scans the finished answer for each claim's salient tokens
   (`7 days`, `five quizzes`, `Internet Explorer`) and fails the turn if one
   appears without a retirement marker nearby.

Twelve claims are extracted from the sample corpus, covering all eight
policies that retire a statement, with no false positives. Precision matters
here: an earlier marker list included a bare `"no longer"` and flagged FAQ-05's
"if you **no longer** have access to the email address" — a user situation, not
a withdrawn policy.

---

## 5. `chunks_fts` — lexical index

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED, text, heading, topics,
    tokenize = 'porter unicode61'
);
```

Kept in sync by `replace_chunks`. FTS5 has no foreign keys, so its rows are
deleted explicitly rather than by cascade — an easy leak to introduce.

Queries are built from **content terms only** (`app/text.py`). This is both a
precision and a safety measure: user text goes straight into a `MATCH`
expression where `"`, `*`, `^` and `-` are operators, so tokenising to word
characters makes FTS-syntax injection impossible. A live bug came from exactly
this — an unescaped apostrophe in "I don't recognise this charge" raised a
syntax error that was swallowed into "no results found", which looked identical
to "we have nothing about unrecognised charges".

---

## 6. `conversations` and `messages`

`conversations.slots` is the part that makes multi-turn work:

```json
{"course_name": "Biology Essentials", "order_id": null,
 "platform": "ios", "amount": "$79", "has_screenshot": true}
```

Slots are extracted deterministically by regex and **never summarised**. When a
long conversation is compressed into `conversations.summary`, the order number
must survive verbatim; asking a model to preserve an identifier through a
paraphrase is exactly the lossy step that makes a learner repeat themselves.

`messages` carries the full decision trail per assistant turn:

| Column | Answers the question |
|---|---|
| `standalone_query` | what did we actually search, after rewriting? |
| `retrieved_chunk_ids` | what was in front of the model, in what order? |
| `citations` | what did it claim to use? |
| `retrieval_score`, `score_margin` | how relevant, and how clearly? |
| `groundedness` | did verification support it? |
| `confidence`, `action`, `reason_codes` | what did we do, and why? |

A support lead asking "why did the bot tell that learner 7 days?" can answer it
from the database without re-running anything. An answer you cannot reconstruct
is an answer you cannot fix.

---

## 7. `escalations`

An escalation is a **work item**, not a shrug. `summary` is written for the
agent who picks it up:

```
Learner asked: There's a $79 LearnForge charge I never authorised.
Detected intent: fraud or unrecognised charge
Already established: amount: $79
Handing over because: this request type is always handled by a person
Knowledge-base articles consulted: FAQ-15, POLICY-07, POLICY-10
```

`attempted_answer` keeps the draft the learner *nearly* received, so an agent
can correct rather than start over. `context_snapshot` holds the slots, scores
and full retrieval diagnostics as JSON.

`queue` routes to `billing`, `trust_and_safety`, `content`, `accessibility` or
`general` — a taxonomy taken from what the sample tickets actually needed, not
invented.

---

## 8. `ingest_runs`

Records each indexing pass: corpus path, embedding model, documents seen and
changed, chunks and vectors written, status. Makes "when was this last
indexed, and with which model?" a query rather than an archaeology exercise.

---

## Scaling to Postgres + pgvector

The schema is written so the move is mechanical. Nothing about the application
logic changes — `app/store.py` is the only file that touches SQL.

| SQLite (now) | Postgres + pgvector (at scale) |
|---|---|
| `documents`, `chunks` | unchanged; `topics TEXT` → `TEXT[]` or `JSONB` |
| `embeddings.vector BLOB` | `vector(768)` + HNSW index |
| brute-force NumPy cosine | `ORDER BY embedding <=> $1 LIMIT k` |
| `chunks_fts` (FTS5) | `tsvector` + GIN, or OpenSearch |
| in-process RRF | unchanged — it is a pure function of two rank lists |
| one file | connection pool; `conversations`/`messages` may split to their own service |

```sql
-- the only structural change that matters
CREATE EXTENSION vector;
ALTER TABLE embeddings ALTER COLUMN vector TYPE vector(768);
CREATE INDEX ON embeddings USING hnsw (vector vector_cosine_ops);
```

The crossover is roughly 10⁴–10⁵ chunks, where a full scan stops fitting
comfortably in memory and per-query latency starts to be dominated by it. At
66 chunks a brute-force scan is microseconds and **exact** — an HNSW index
would be slower to build, approximate, and an operational dependency bought
for no benefit.

### What would need real thought at scale

- **Multi-tenancy.** `documents` would need an `org_id` and every query a
  tenant filter — cheap to add now, expensive to retrofit.
- **Chunk-level ACLs.** Internal agent-only runbooks in the same index as
  learner-facing FAQs need filtering *before* ranking, not after.
- **Embedding backfill.** Re-embedding 10⁶ chunks is a job queue, not a
  request handler. The `(chunk_id, model)` key already supports running the
  old and new model side by side while the backfill drains.
