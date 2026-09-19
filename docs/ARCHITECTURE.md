# System Design

All diagrams are Mermaid and render natively on GitHub.

---

## 1. High-level architecture

```mermaid
flowchart TB
    subgraph clients["Clients"]
        CLI["CLI<br/>python -m app.cli"]
        WEB["Web chat UI<br/>GET /"]
        API["HTTP clients<br/>POST /chat"]
    end

    subgraph service["Assistant service (app/pipeline.py)"]
        direction TB
        CTX["1 · Contextualizer<br/>redact · slot-fill · rewrite follow-ups"]
        RET["2 · Retriever<br/>BM25 + dense, RRF, metadata re-score"]
        GEN["3 · Generator<br/>grounded answer, mandatory citations"]
        VER["4 · Verifier<br/>citations · retired claims · safety · entailment"]
        GATE["5 · Escalation gate<br/>deterministic rules"]
        CTX --> RET --> GEN --> VER --> GATE
    end

    subgraph providers["Model providers (swappable)"]
        GROQ["Groq<br/>llama-3.3-70b<br/>generation"]
        GEM["Gemini<br/>text-embedding-004<br/>embeddings"]
        OFF["Offline stubs<br/>keyless fallback"]
    end

    subgraph storage["SQLite (schema.sql)"]
        DOCS[("documents<br/>chunks<br/>embeddings")]
        FTS[("chunks_fts<br/>BM25 index")]
        DEP[("deprecated_claims<br/>quarantined")]
        CONV[("conversations<br/>messages")]
        ESC[("escalations")]
    end

    subgraph ingestion["Ingestion (app/ingest.py)"]
        KB["Knowledge base<br/>faqs · policies · tickets"]
        PARSE["Parse + enrich<br/>authority tier, dates,<br/>topics, retired claims"]
        CHUNK["Structure-aware chunking"]
        EMBED["Embed changed chunks only<br/>content-hash gated"]
        KB --> PARSE --> CHUNK --> EMBED
    end

    clients --> service
    EMBED --> DOCS
    CHUNK --> FTS
    PARSE --> DEP

    RET <--> FTS
    RET <--> DOCS
    RET <--> DEP
    CTX <--> CONV
    GATE --> ESC
    GATE --> CONV

    CTX -.-> GROQ
    RET -.-> GEM
    RET -.-> GROQ
    GEN -.-> GROQ
    VER -.-> GROQ
    providers -.fallback.-> OFF

    ESC ==> HUMAN["Human agent queues<br/>billing · trust &amp; safety ·<br/>content · accessibility"]
```

**The shape of the argument.** Retrieval and generation are the easy half. The
half that decides whether this is safe to put in front of learners is
everything after the arrow out of `Generator`: an independent verification
pass, and a gate made of plain Python rules rather than model self-assessment.

---

## 2. How a query flows end to end

```mermaid
flowchart TD
    Q["Learner message"] --> RED{"Card number,<br/>CVV or PIN present?"}
    RED -->|yes| STRIP["Redact before anything<br/>is stored, embedded or sent"]
    RED -->|no| SLOT
    STRIP --> SLOT["Extract slots<br/>course, order id, amount, platform"]

    SLOT --> FU{"Follow-up turn?"}
    FU -->|yes| RW["Rewrite against history<br/>'the biology one'<br/>-> 'Biology Essentials course'"]
    FU -->|no| SEARCH
    RW --> SEARCH["Standalone query"]

    SEARCH --> BM["BM25 over FTS5<br/>exact terms: '14 days', 'CVV'"]
    SEARCH --> DN["Dense cosine<br/>paraphrase: 'my course vanished'"]
    BM --> RRF["Reciprocal Rank Fusion"]
    DN --> RRF

    RRF --> META["Metadata re-score<br/>policy &gt; faq &gt; ticket<br/>recency · staleness"]
    META --> RR["LLM rerank<br/>top-12 -> top-5"]
    RR --> GUARD["Recall guard:<br/>each branch's best raw hit<br/>is kept regardless"]
    GUARD --> DIV["Diversify<br/>max 2 chunks per document"]

    DIV --> SCORE{"top_score &ge; floor?"}
    SCORE -->|no| INDOM{"In-domain<br/>vocabulary?"}
    INDOM -->|no| ABSTAIN["ABSTAIN<br/>no ticket opened"]
    INDOM -->|yes| ESC

    SCORE -->|yes| PROMPT["Build grounded prompt<br/>+ labelled authority per block<br/>+ RETIRED STATEMENTS block<br/>+ detected disagreements"]
    PROMPT --> LLM["Generate<br/>JSON: answer, citations,<br/>answerable, conflict"]

    LLM --> V1{"Cited docs<br/>actually retrieved?"}
    V1 -->|no| ESC
    V1 -->|yes| V2{"Withdrawn claim<br/>stated as current?"}
    V2 -->|yes| ESC
    V2 -->|no| V3{"Asks for card<br/>number or CVV?"}
    V3 -->|yes| ESC
    V3 -->|no| V4["Entailment judge<br/>claim-by-claim groundedness"]

    V4 --> GATE{"Escalation gate"}
    GATE -->|"sensitive intent<br/>money · identity · legal"| ESC
    GATE -->|"human requested"| ESC
    GATE -->|"groundedness &lt; 0.70"| ESC
    GATE -->|"unresolved conflict"| ESC
    GATE -->|"2 prior failed turns"| ESC
    GATE -->|"stale source"| CAVEAT["ANSWER + caveat"]
    GATE -->|"all clear"| ANSWER["ANSWER<br/>with inline citations"]

    ESC["ESCALATE<br/>routed, prioritised,<br/>with a written handover"] --> PERSIST
    ABSTAIN --> PERSIST
    CAVEAT --> PERSIST
    ANSWER --> PERSIST["Persist the full decision trail:<br/>rewritten query, chunks, scores,<br/>action, reason codes"]
```

### The three failure families the gate encodes

| Family | Trigger | Action |
|---|---|---|
| **I could not find it** | empty retrieval, `top_score` under the floor, no rank separation | abstain if out-of-domain, escalate if the learner is asking about their account |
| **I do not trust what I produced** | invalid citation, retired-claim leak, unsafe request, groundedness under 0.70, model unavailable | escalate — never show the answer |
| **This is not a bot's decision** | fraud, duplicate charge, account ownership, refund dispute, legal, explicit request for a human, repeated failure | escalate regardless of confidence |

Staleness is deliberately *not* in this table. A source that has not been
reviewed recently earns a caveat and a note to the content team, not a refusal.

---

## 3. Data model

```mermaid
erDiagram
    documents ||--o{ chunks : "split into"
    documents ||--o{ deprecated_claims : "quarantines"
    chunks ||--o{ embeddings : "embedded per model"
    chunks ||--o| chunks_fts : "indexed for BM25"
    chunks ||--o{ deprecated_claims : "located in"
    conversations ||--o{ messages : "contains"
    conversations ||--o{ escalations : "may raise"
    messages ||--o| escalations : "triggers"

    documents {
        text doc_id PK "FAQ-02, POLICY-10"
        text source_type "policy|faq|ticket"
        int authority_tier "1 policy, 2 faq, 3 ticket"
        text effective_date "ISO, parsed from prose"
        int has_explicit_date "undated = staleness risk"
        text ticket_status "escalated|resolved|..."
        text topics "JSON array"
        text content_hash "drives idempotent re-ingest"
        int version
        int is_active "soft delete"
    }
    chunks {
        text chunk_id PK "POLICY-02#c1"
        text doc_id FK
        text text "the cited span"
        text heading "FAQ question, carried in"
        int authority_tier "denormalised: no join on read"
        int has_deprecation_notice
    }
    embeddings {
        text chunk_id PK "also FK to chunks"
        text model PK "swap is additive, not a migration"
        blob vector "float32, L2-normalised"
    }
    deprecated_claims {
        int claim_id PK
        text claim_text "the withdrawn rule"
        text marker "phrase that flagged it"
    }
    messages {
        int message_id PK
        text standalone_query "what was actually searched"
        text retrieved_chunk_ids "JSON, ranked"
        real retrieval_score
        real groundedness
        text action "answered|abstained|escalated"
        text reason_codes "JSON, why"
    }
    escalations {
        text escalation_id PK
        text queue "billing|trust_and_safety|..."
        text priority
        text summary "agent-facing brief"
        text attempted_answer "what we nearly said"
    }
```

Full column-by-column commentary is in [DATA_SCHEMA.md](DATA_SCHEMA.md); the
executable definition is [`schema.sql`](../schema.sql).

---

## 4. Ingestion and freshness

```mermaid
sequenceDiagram
    autonumber
    participant SRC as Help-centre source
    participant ING as app/ingest.py
    participant DB as SQLite
    participant EMB as Gemini embeddings

    SRC->>ING: markdown entries<br/>(# FAQ-01 — title)
    ING->>ING: split on authored boundaries<br/>(never mid-rule)
    ING->>ING: parse "Last reviewed: February 2026"<br/>-> effective_date
    ING->>ING: assign authority tier from source type
    ING->>ING: detect withdrawn statements<br/>("That wording is outdated")
    ING->>DB: SHA-256 of body vs stored content_hash

    alt unchanged
        DB-->>ING: same hash
        ING->>ING: skip — no re-chunk, no re-embed
    else changed or new
        ING->>DB: upsert document, bump version
        ING->>DB: replace chunks + FTS rows
        ING->>DB: write deprecated_claims
        ING->>EMB: embed only the new chunks
        EMB-->>DB: vectors keyed by (chunk_id, model)
    end

    ING->>DB: record ingest_run (counts, model, status)
```

Because ingest is content-hashed and idempotent, `POST /ingest` is safe to run
on a cron or fire from a CMS webhook. Re-embedding the whole corpus on every
publish is the thing that makes teams stop re-indexing, and a corpus that
stops being re-indexed is how stale answers happen.

---

## 5. What this design is not

Honest limits, stated once here and expanded in the README's trade-offs:

- **Not horizontally scalable as written.** One SQLite file, one process,
  brute-force vector scan. Correct at 66 chunks; wrong at 10⁵.
- **No user or order data.** The assistant reasons about *policy*, never about
  a specific account. That is why every account-specific question escalates
  rather than guessing — there is no system of record behind it.
- **Thresholds are tuned on 37 golden cases.** They are a starting point
  calibrated on a small set, not values earned from production traffic.
- **The entailment judge shares a model family with the generator**, so their
  blind spots correlate. The two deterministic checks exist precisely because
  the judge cannot be the only line of defence.
