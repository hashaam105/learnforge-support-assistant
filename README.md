# LearnForge AI Support Assistant

A retrieval-augmented support assistant over an ed-tech knowledge base:
hybrid retrieval, grounded generation with verifiable citations, multi-turn
conversation, and a human handoff governed by rules rather than by the model's
opinion of itself.

    contextualize -> retrieve -> generate -> verify -> decide -> persist

Most of the work is the ordinary path — chunking that respects authored
structure, BM25 fused with dense retrieval, a grounded prompt, and a
verification pass. On top of that sits a thin layer for the thing this
particular corpus does that most do not: it contradicts itself on purpose, and
it documents its own stale policy inline. That layer is about 5% of the code
and it is the most interesting 5%, so it gets discussed at length below — but
13 of the 37 evaluation cases are ordinary questions with no trap in them, and
those are the ones a support assistant answers all day.

Corpus: 40 documents — 15 course FAQs, 10 help-centre policies, 15 past support
ticket transcripts.

```bash
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY and GEMINI_API_KEY
python -m app.ingest          # index the knowledge base
python -m app.cli --demo      # scripted walkthrough of every behaviour
```

It also runs with **no API keys at all** — retrieval degrades to BM25 and
answers become verbatim extracts, clearly labelled as such. `pytest` and
`python -m eval.run --retrieval-only` both pass with zero configuration.

| | |
|---|---|
| **System design diagrams** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| **Data schema** | [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) · [schema.sql](schema.sql) |
| **Evaluation set** | [eval/golden_set.yaml](eval/golden_set.yaml) |
| **Results + what the failures taught** | [jump to Results](#results) |

Live run, 37 golden cases: **recall@5 0.968 · MRR 0.935 · groundedness 0.981 ·
zero invalid citations · escalation recall 0.90**. Five failures, four of them
real defects the eval caught and unit tests could not — written up in full
under [Results](#results), because that is the more useful half.

---

## What the sample data is actually testing

The corpus README says the contradictions are deliberate, and each one forces a
specific mechanism. This table is about the hard cases, not the common ones —
the everyday path is the pipeline above, and it carries the majority of the
traffic and of the evaluation set.

| Trap in the data | Mechanism it forces |
|---|---|
| POLICY-02 states the current **14-day** refund window *and* quotes a withdrawn **7-day** article, in the same document | Withdrawn claims are **extracted into quarantine at ingest time**. Without this, a model answers "7 days" and every groundedness check passes, because the text really is in the retrieved context. |
| TICKET-03: a learner has a screenshot of a **30-day guarantee**. TICKET-08: FAQ, cancellation page and subscription terms disagree. Both escalated. | **Conflict detection that includes the learner's own claim**, plus escalation on unresolved contradiction. Corpus-vs-corpus checking alone never sees these. |
| An agent in TICKET-03 says "our standard refund period is 14 days" — a perfect lexical match for any refund question | **Authority tiers.** A ticket is evidence of how one case was handled once. Policy governs; FAQ explains; tickets are anecdote. |
| Policies dated Dec 2025 → Jun 2026; POLICY-05 and POLICY-08 carry no date at all | **Parsed `effective_date`**, recency weighting, staleness caveats, and a separate smaller penalty for *undated* documents. |
| TICKET-07: `"Cancel my LearnForge"` → `"The payment"` → `"The payment from last week"` → `"It's the biology one"` | **Query rewriting plus durable slot memory.** Turn 4 is unretrievable on its own. |
| POLICY-07/10: never request full card numbers, CVV, PIN — and POLICY-10 records that an internal instruction to ask for card digits was *retired* | **Redaction at the input boundary** and a **safety check on the output**. |
| Tickets ending `STATUS: Escalated` cluster on money, identity and ambiguous promises | The escalation policy is **derived from the corpus**, not invented. |

---

## How a turn works

```
contextualize -> retrieve -> generate -> verify -> decide -> persist
```

**1 · Contextualize.** Redact card numbers and security codes *before* anything
is stored, embedded or sent to a provider. Extract durable slots (course, order
id, amount, platform). If this is a follow-up, rewrite it into a standalone
query against the conversation.

**2 · Retrieve.** BM25 over FTS5 and dense cosine, fused by Reciprocal Rank
Fusion, re-scored on metadata (authority, recency, staleness), reranked by the
LLM, then diversified to at most two chunks per document.

**3 · Generate.** The primary model composes the answer. Every context block
is labelled with its authority and date.
Retired statements are listed explicitly as things that must never be asserted
as current. Output is JSON with mandatory citations and an `answerable` flag,
because a model that cannot say "I don't know" will invent something instead.

**4 · Verify.** Four independent checks — citation validity, retired-claim
leakage, unsafe requests, and claim-by-claim entailment. The first three are
deterministic and cannot be talked out of a failure by a fluent answer.

A note on which model runs where: rewriting, reranking, entailment checking and
summarising are narrower tasks than composing a grounded answer, and they are
about three quarters of the calls a turn makes. They run on a smaller
`UTILITY_MODEL`; only the answer the learner reads uses the primary model. That
cut measured end-to-end latency from ~5.2s to ~4.0s with no change in golden-set
results. Set `UTILITY_MODEL` equal to `LLM_MODEL` to turn the split off.

**5 · Decide.** A plain-Python gate reads signals computed *outside* the
generation call and chooses: answer, answer with caveat, abstain, or escalate.

**6 · Persist.** The rewritten query, retrieved chunks, scores, action and
reason codes all land in `messages`, so any answer can be explained after the
fact without re-running it.

---

## Failure handling

### Low-confidence answers

Confidence is a **composite of measured signals**, not the model's opinion of
itself:

```
0.40 × retrieval top-score   (cosine, or a coverage-blended BM25 proxy)
0.35 × verifier groundedness (claim-by-claim entailment)
0.15 × score margin          (separation between #1 and #2)
0.10 × model self-confidence (smallest weight, deliberately)
```

A model asked "are you confident?" answers from the same forward pass that
produced the error, so its self-report correlates with fluency, not
correctness. It gets the smallest share. Any hard verification failure sets the
composite to **0.0** outright rather than averaging away.

Three independent families of trigger send a turn to a human:

| Family | Examples | Result |
|---|---|---|
| **I could not find it** | empty retrieval, top-score below the floor, no rank separation | abstain if out-of-domain, escalate if it is about the learner's account |
| **I do not trust what I produced** | invalid citation, retired-claim leak, unsafe request, groundedness < 0.70, provider outage | escalate — the draft is never shown as an answer |
| **This is not a bot's decision** | fraud, duplicate charge, account ownership, refund dispute, legal threat, explicit request for a human, two prior failed turns | escalate regardless of confidence |

That third family matters most. TICKET-03, TICKET-08 and TICKET-11 were not
retrieval failures — retrieval worked. They turn on account facts the knowledge
base cannot contain, or on a genuine contradiction. **A system that only
escalates on low similarity answers all three confidently and wrongly.**

Abstention and escalation are kept distinct. "What's a good sourdough recipe?"
gets a polite redirect; opening a billing case for it would be its own failure.

### Stale data

Four mechanisms, applied at different stages:

1. **Parsed dates.** `Last reviewed: February 2026` becomes
   `effective_date = 2026-02-01`, so `STALE_AFTER_DAYS` is a real comparison.
2. **Undated documents get their own penalty.** POLICY-05 and POLICY-08 carry
   no date. That is not the same as being old — it is *unverifiable*, so it
   earns a smaller penalty than a genuinely stale document.
3. **Withdrawn claims are quarantined at ingest**, listed in the prompt as
   never-assert-as-current, and checked for in the finished answer.
4. **Staleness caveats, not refusals.** A learner with a question about an
   article reviewed 14 months ago should get the answer *and* be told. Refusing
   would be worse service and would not make the article fresher.

Freshness of the index itself is handled by content-hashed, idempotent ingest:
unchanged documents are not re-chunked or re-embedded, so `POST /ingest` is
safe on a cron or a CMS webhook. Re-embedding everything on every publish is
what makes teams stop re-indexing, and a corpus nobody re-indexes is how stale
answers happen.

### Bad retrieval

- **Hybrid search.** BM25 catches exact tokens embeddings blur (`14 days`,
  `CVV`, `Atomic Orbitals`); dense catches paraphrase ("my course vanished").
- **Recall guard.** Whatever the metadata weights and the LLM reranker decide,
  each branch's single best raw hit stays in the context. The eval suite caught
  both heuristics evicting the only relevant document — TICKET-02 is the sole
  document about being charged twice, and the tier-3 penalty alone pushed it
  out of the top five.
- **Source diversity.** At most two chunks per document, so two adjacent
  paragraphs of one FAQ cannot crowd out the policy that governs the answer.
- **Injection-safe lexical queries.** User text is tokenised to word characters
  before it reaches a `MATCH` expression.
- **Empty is a real answer.** Out-of-domain questions return nothing and
  abstain, rather than surfacing the least-bad match.

### Provider failures

Every provider call is retried with backoff and degrades rather than crashing.
No Gemini key or a failing embedding call → BM25-only retrieval. No Groq key or
a failed generation → the turn becomes an escalation, never a fabricated
answer. Degraded mode is **declared** in the response, the `/health` payload
and the UI — a degraded answer is never allowed to look like a confident one.

---

## Evaluation plan

`eval/golden_set.yaml` holds 37 cases, all derived from the sample corpus so
each has a defensible ground truth, in seven buckets:

| Bucket | n | What it protects |
|---|---|---|
| `answerable` | 13 | the everyday path |
| `stale` | 7 | withdrawn policy is never served as current |
| `conflict` | 2 | contradictions escalate instead of resolving silently |
| `escalation` | 6 | money, identity and legal never get bot-adjudicated |
| `out_of_scope` | 4 | abstain, and do *not* open a ticket |
| `safety` | 2 | card data never reaches the reply or the log |
| `multi_turn` | 3 | coreference resolution, scored on the final turn |

Buckets exist because one accuracy number hides the failures that matter. A
system that answers every FAQ perfectly and also confidently answers "what is
the capital of France" is not a good system.

```bash
python -m eval.run                   # full suite
python -m eval.run --judge           # + LLM-as-judge answer quality
python -m eval.run --retrieval-only  # deterministic, no keys, no model calls
python -m eval.run --bucket stale
```

### Metrics, and why each one

**Retrieval is measured separately from answering** — recall@k and MRR against
expected `doc_id`s. They fail differently and are fixed differently: a recall
drop is a chunking or ranking problem, a faithfulness drop with healthy recall
is a prompt or model problem. Blending them hides both.

**Hallucination is measured three ways**, because no single proxy is honest:

| Metric | How | Trustworthiness |
|---|---|---|
| Invalid-citation rate | cited a doc that was never retrieved | exact, zero false positives |
| Retired-claim leak rate | restated a withdrawn policy as current | corpus-specific; **generic RAG metrics miss this entirely**, because the bad text *is* in the context |
| Unsupported-claim rate | verifier entailment judgement | broadest coverage, least trustworthy — an LLM grading an LLM |

**Abstention and escalation get precision and recall of their own.** They are
the safety valves and they fail in opposite, equally bad directions: never
escalating is dangerous, always escalating is useless. One number cannot see
both.

**Answer quality** is LLM-as-judge against per-case `reference_points`, run
only under `--judge`. It is the softest metric here and is treated as such.

Every run writes a JSON report to `eval_reports/` so two runs can be diffed.

### Results

Full suite, live, `openai/gpt-oss-120b` generation + `gemini-embedding-001`
hybrid retrieval, judge enabled. Report: `eval_reports/`.

| Metric | Value |
|---|---|
| Cases passed | **32 / 37** (0.865) |
| Retrieval recall@5 | **0.968** |
| Retrieval MRR | **0.935** |
| Invalid-citation rate | **0.0** |
| Retired-claim leak rate | 0.027 (1 case, a false positive — see below) |
| Unsupported-claim rate | 0.095 |
| Mean groundedness | **0.981** |
| Answer quality (LLM judge) | **0.92** |
| Escalation precision / recall | 0.60 / **0.90** |
| Abstention precision / recall | 1.00 / 0.33 |
| Latency p50 / p95 | 21s / 26s (rate-limited free tier; ~4s unthrottled) |

Retrieval-only mode isolates the retriever from the model, and comparing the
two configurations is the clearest single argument for hybrid search:

| `--retrieval-only` | cases | recall@5 | MRR |
|---|---|---|---|
| keyless (BM25 over FTS5 alone) | 37 / 37 | 1.0 | 0.781 |
| hybrid (BM25 + dense, RRF) | 37 / 37 | 1.0 | 0.858 |

Both configurations *find* the right document for every case — this corpus is
small and BM25 is strong on it. The difference is rank: dense retrieval puts
the governing document first materially more often, which is what determines
whether it survives into the top-5 context window on a larger corpus.

#### What the five failures were, and what they were worth

This is the part of an eval that earns its keep. Four of the five were real
defects in the system, not noise, and none of them were visible from unit tests
or from trying the assistant by hand.

1. **Every out-of-scope question escalated instead of abstaining.** A logic
   bug. When retrieval fails, the generator refuses, and the verifier then
   correctly reports that a refusal is not grounded in LearnForge policy —
   and those derived signals were allowed to vote on the outcome. So "what's a
   good sourdough starter recipe?" opened a support case. Verifying a
   non-answer is meaningless; consequential signals no longer count when
   retrieval has already failed. *Fixed, re-validated, regression test added.*

2. **A correct answer was failed as a retired-claim leak.** POLICY-04 retires
   "recommended downloading lessons over cellular data", and the leak detector
   treated the bare phrase "cellular data" as diagnostic. But the *current*
   recommendation discusses cellular data too, so the right answer ("use Wi-Fi
   rather than cellular data for large downloads") was blocked. A named-entity
   marker is only valid if it appears solely in the withdrawn statement.
   *Fixed, re-validated, regression test added.*

3. **A refund demand resting on a named help article was answered, not
   escalated.** TICKET-15 exactly: "Your Offline Learning Guide said I could
   download to my laptop — that's the only reason I bought this course." The
   intent pattern matched the literal words "article" and "website", so a
   *named* guide slipped through. *Fixed, re-validated, regression test added.*

4. **`oos-write-code` escalated on `model_unavailable`** — the provider's daily
   token quota was exhausted by this point in the run. Escalating when the
   model is unavailable is correct behaviour: with no answer to verify, the
   safe action is a human. Worth noting because it exposed a separate real bug
   (below) rather than a scoring one.

5. **`multi-ticket07-biology` missed its expected documents.** The one case
   where the *label* was wrong rather than the system. The four turns resolve
   to "the status of the payment for the Biology course from last week", which
   is a payment question; the original label expected the refund policy,
   because that is where the real ticket ended up — but only after the agent
   asked two disambiguating questions that this sequence omits. The assistant
   retrieves TICKET-07 itself as precedent and escalates. The label was
   widened *and* the genuine gap it exposed is written up above: there is no
   `clarify` action, which is what the human agent actually used.

Two further bugs surfaced while investigating, both in the provider adapter and
both invisible to the test suite because they only appear against a live API:

- **Multi-turn rewriting was silently disabled.** `gpt-oss` models count
  reasoning tokens against `max_tokens`, so a 250-token budget was consumed
  before any JSON was emitted. The API reports that as HTTP 400
  `json_validate_failed`, the rewriter caught it and fell back to the raw
  follow-up, and "It's the biology one" reached the retriever unresolved. No
  crash, no log line — just a feature quietly doing nothing.
- **A daily-quota rejection hung the turn for five minutes.** The adapter
  honoured `Retry-After` literally, and an exhausted daily quota answers "try
  again in 5m32s". Backoff waits are now capped; past the cap the call fails
  and the pipeline degrades, which is what the escalation path is for.

**Status of these numbers.** The table is a complete, unmodified run taken
*before* the fixes. Fixes 1–3 were each re-validated by re-running the affected
cases (all now pass) and are covered by regression tests, but the provider's
daily token quota was exhausted before a full post-fix suite could run, so no
clean 37/37 run is claimed here. Re-run `python -m eval.run --judge` once quota
resets to reproduce.

### What this eval plan does *not* cover

Stated plainly, because a plan that claims completeness is not credible:

- **37 cases written by the same person who wrote the system.** They encode my
  assumptions about what matters. Real coverage needs cases sampled from
  production traffic and labelled by support agents.
- **No inter-annotator agreement.** Single author, no second opinion on the
  ground truth.
- **The judge shares a model family with the generator**, so their blind spots
  correlate. This is exactly why the two deterministic checks exist — the judge
  cannot be the only line of defence.
- **Thresholds are calibrated on these 37 cases.** They are a defensible
  starting point, not values earned from traffic.
- **No adversarial suite.** Prompt injection via knowledge-base content is a
  real risk for any system that ingests user-editable help articles, and it is
  not tested here.

### What I would measure in production

Offline suites go stale. The signals that would actually run the system:

- **Deflection rate** — resolved without a human — against **escalation
  precision**. Optimising deflection alone is how support bots become hated.
- **Reopen rate** after a bot answer: the honest proxy for a wrong answer.
- **Agent correction rate** on escalations — an agent flagging `attempted_answer`
  as wrong is a free, high-quality hallucination label.
- **Citation click-through**, as a weak signal that citations are real and useful.
- **Time-to-first-response** for escalated versus answered turns.
- **Corpus health**: which documents are retrieved but never cited (poor
  chunking), which are cited while stale (content debt), which questions
  repeatedly abstain (knowledge gaps — the most valuable output of all).

---

## Trade-offs

### SQLite instead of a vector database

At 66 chunks a brute-force NumPy scan is microseconds and **exact**. An HNSW
index would be slower to build, approximate, and an operational dependency
bought for no benefit. Keeping documents, chunks, vectors, the lexical index,
conversation state and the escalation queue in one file also means the whole
system is one `git clone` and one `pip install` — which matters when the brief
says the prototype has to actually run.

`app/store.py` is the only file that touches SQL, and the schema is written in
portable DDL. The crossover is roughly 10⁴–10⁵ chunks:

```sql
CREATE EXTENSION vector;
ALTER TABLE embeddings ALTER COLUMN vector TYPE vector(768);
CREATE INDEX ON embeddings USING hnsw (vector vector_cosine_ops);
```

**With more budget:** Postgres + pgvector, not a dedicated vector DB. Keeping
documents, chunks and vectors transactionally consistent in one database is
worth more than the last few percent of ANN throughput, and one fewer system to
operate.

### Reciprocal Rank Fusion instead of weighted score blending

BM25 scores and cosine similarities live on incomparable scales, and BM25
additionally shifts with query length. Any fixed `α·bm25 + (1−α)·cosine` is a
number that needs recalibrating whenever either side changes. RRF reads only
*ranks*, so it needs no calibration and cannot be destabilised by one branch's
outliers.

**The cost:** RRF discards magnitude, so its output cannot serve as an absolute
confidence signal — a query with no good answer still produces a high RRF score
for its least-bad hit. That is why confidence comes from cosine (or a
coverage-blended BM25 proxy) rather than the fused score. **With a labelled
click dataset** I would train a small cross-encoder reranker and use its
calibrated score directly, which removes the proxy entirely.

### Authority tiers instead of letting relevance decide

Pure relevance ranks TICKET-03 near the top for refund questions, because an
agent said the right words there. The fix is metadata: `policy > faq > ticket`.

**The cost:** ticket transcripts hold real operational knowledge that policies
do not — TICKET-02 is the only document in the corpus about being charged
twice. Blanket down-ranking loses that. The recall guard is the compromise:
tickets are demoted but the best raw hit is never evicted. **With more time**
I would index tickets separately as *precedent* — retrieved explicitly for "has
this happened before?" rather than competing with policy for the same slot.

### Keyword topic tagging instead of an LLM tagger

Deterministic, inspectable, free, and reruns identically. An LLM tagger would
handle nuance better but makes ingestion non-reproducible and costs a call per
document. Topics are a soft signal, never a hard filter, so a wrong tag costs a
little ranking rather than making a document unreachable.

**With more time:** LLM tagging at ingest with the keyword tagger kept as a
regression check on its output.

### Two-pass verification instead of trusting the generator

The verifier roughly doubles latency and token cost per answered turn. It is
worth it: the deterministic half catches the two failure modes that a
single-pass system cannot see at all — citing a document that was never
retrieved, and restating a withdrawn policy that is genuinely present in the
context.

**The cost:** on the free tier, an answered turn runs 3–4 model calls (rerank,
generate, verify, and the rewrite on follow-ups) and takes **5–15 seconds**.
That is too slow for a live chat widget.

**With more budget**, in order of value:
1. **Drop the LLM reranker** for a small cross-encoder — removes one call and
   is usually better at ranking anyway.
2. **Stream the answer** while the verifier runs concurrently; retract or
   caveat on failure. Perceived latency drops to first-token time.
3. **Already done: the verifier and reranker run on a smaller model.** Next
   would be a purpose-trained NLI model rather than a general instruct model —
   entailment is a classification task and does not need a chat model at all.
4. **Cache aggressively.** Support traffic is Zipfian — a small set of
   questions is most of the volume. Semantic caching of `(query → verified
   answer)` keyed on corpus version would cut both cost and latency sharply.

### Deterministic escalation rules instead of an LLM router

An LLM router would generalise to intents I did not anticipate. But this gate
decides when to stop trusting a language model, and building it *out of* a
language model makes it fail in correlated ways — the cases where the model is
most confidently wrong are exactly where the router would wave it through.

Plain rules are auditable, testable in isolation (`tests/test_escalation.py`),
and changeable by a support lead without touching a prompt. **With more time**
I would run an LLM classifier *alongside* as a second opinion that can only
ever escalate, never de-escalate — strictly additive safety.

### Structure-aware chunking instead of fixed windows

Every entry is already a clean semantic unit delimited by `# FAQ-01 — Title`.
Fixed-size windowing would cut a refund rule away from its qualifier
("...provided that the course has not been substantially consumed"), which is
precisely the kind of split that produces confidently wrong answers.

**The cost:** this depends on the corpus being well-structured markdown. A real
help centre is messier — HTML, PDFs, inconsistent headings. **With more time**
the ingestion layer needs per-source-type parsers and a layout-aware chunker,
and that is where a meaningful share of the engineering would go.

### The missing fourth action: clarify

The gate can answer, caveat, abstain or escalate. It cannot ask a question and
wait — and the sample corpus shows that is what a good agent does. TICKET-07 is
four turns of a human agent narrowing "Cancel my LearnForge" down to a specific
refund request, one question at a time.

The generator already returns a `clarifying_question`, and it gets appended to
answers, but there is no `clarify` *action* that suspends the turn, records
what is still unknown, and resolves on the next message. Without it, an
ambiguous in-domain request escalates when one question would have settled it —
which is a worse learner experience and a more expensive one. Adding it means a
small amount of state (what was asked, what would unblock the answer) and a cap
on consecutive clarifications so the assistant cannot interrogate someone
indefinitely. This is the first thing I would build after the account lookup.

### A note on fitting the dataset

Two detectors were overfit to the sample *file* rather than to the domain, and
are worth naming because the eval could not catch them — the eval uses the same
corpus the mistakes were copied from.

The elliptical-reference matcher enumerated `(biology|physics|chemistry|python|
ux|design|data)`: the exact course subjects in the sample tickets. "The
astronomy one" silently failed. It now matches on shape — `the <subject> one` —
with a filler-word guard so "the last one" is not read as a subject.

The retired-claim detector hardcoded two named entities, `Internet Explorer`
and `five-user family plan`, lifted straight out of this corpus. That is a
lookup table for one dataset, not a detector, and it would do nothing on any
other knowledge base. Names are now derived from each quarantined claim:
multi-word capitalised phrases and multi-word quoted phrases. The derivation
also has to skip the *replacement* wording, because a retirement sentence
quotes both sides of the change — POLICY-06 retires "instantly" and introduces
"automatically synchronized", and flagging the second would fail every correct
answer about progress syncing.

Domain-specific is not the same as overfit. The intent patterns (fraud,
duplicate charge, account ownership) and the topic vocabulary are tuned to
ed-tech support on purpose; a generic classifier would route worse. The
distinction is whether swapping in a different ed-tech knowledge base would
break it.

### What I would build next, in priority order

1. **An order/account lookup tool.** The single biggest quality win available.
   Most escalations here exist because the assistant has no system of record —
   it cannot see whether a charge is a renewal (TICKET-05) or whether a course
   was bought individually (TICKET-11). A read-only tool call converts a large
   class of escalations into answers, and is *also* a new hallucination surface
   that would need its own verification.
2. **The `clarify` action described above.**
3. **Prompt-injection defence on ingested content.** Help articles are
   user-editable in most organisations. Today a crafted article could instruct
   the model. Content sanitisation plus instruction-hierarchy prompting.
4. **A corpus-health dashboard** driven by `messages`: documents retrieved but
   never cited, stale documents still being cited, questions that repeatedly
   abstain. Feeding real gaps back to the content team beats almost any
   model-side improvement.
5. **Answer caching keyed on corpus version**, invalidated by `ingest_runs`.
6. **Per-tenant isolation and chunk-level ACLs** before this serves more than
   one organisation.

---

## Repository layout

```
app/
  config.py          all tunables, one place
  text.py            stopwords + content-term tokenisation
  store.py           SQLite data access; the only file with SQL
  ingest.py          parse -> enrich -> chunk -> index -> embed
  retrieval.py       hybrid search, RRF, metadata re-scoring, conflicts
  contextualizer.py  redaction, slot memory, follow-up rewriting
  generator.py       grounded prompt + structured answer
  verifier.py        citation / retired-claim / safety / entailment checks
  escalation.py      the gate, intent routing, handoff payload
  pipeline.py        orchestration and persistence
  providers/         Groq, Gemini, and keyless fallbacks
  api.py  cli.py  web/index.html
eval/
  golden_set.yaml    37 labelled cases
  run.py             metrics + JSON reports
tests/               95 tests, no keys or network required
docs/                architecture diagrams, data schema
schema.sql           the executable data model
```

## Interfaces

```bash
python -m app.cli                  # chat; /why /sources /escalations /stats
python -m app.cli --ask "..." --json
python -m app.cli --demo           # every behaviour, scripted

uvicorn app.api:app --reload       # http://127.0.0.1:8000
#   POST /chat   GET /health   GET /escalations
#   POST /ingest GET /sessions/{id}
```

The `/chat` response carries the full decision trail — rewritten query,
candidate scores, verification result, reason codes — and the web UI renders
it. An answer you cannot inspect is an answer you cannot trust.

## Configuration

Everything is environment-driven; see [.env.example](.env.example). The
thresholds worth knowing:

| Variable | Default | Meaning |
|---|---|---|
| `MIN_DENSE_SCORE` | `0.62` | cosine floor below which we treat retrieval as failed |
| `MIN_LEXICAL_SCORE` | `0.30` | the same floor for BM25-only mode |
| `MIN_GROUNDEDNESS` | `0.70` | verifier score below which an answer is not shown |
| `STALE_AFTER_DAYS` | `365` | older sources earn a freshness caveat |
| `LLM_MODEL` | `openai/gpt-oss-120b` | composes the answer |
| `UTILITY_MODEL` | `openai/gpt-oss-20b` | rewrite, rerank, verify, summarise |
| `ENABLE_LLM_RERANK` | `true` | set `false` to trade some ranking quality for ~1s |
| `ENABLE_VERIFIER` | `true` | disabling removes the entailment check, not the deterministic ones |
