"""Ingestion: parse -> enrich -> chunk -> index -> embed.

Three things here matter more than the mechanics:

1. **Structure-aware chunking.** Every entry in this corpus is already a clean
   semantic unit delimited by `# FAQ-01 — Title`. Fixed-size windowing would
   cut a refund policy in half and strand the qualifier ("...provided that the
   course has not been substantially consumed") in a different chunk from the
   rule. We split on the authored boundaries and only sub-split when an entry
   exceeds the budget.

2. **Metadata is the product.** authority_tier, effective_date and
   has_deprecation_notice are what let retrieval resolve the contradictions
   the corpus deliberately contains. Without them you have a search box.

3. **Deprecation extraction.** This corpus annotates its own rot: "That
   wording is outdated", "should not be treated as the current standard
   policy", "That information is obsolete". We lift those sentences into
   `deprecated_claims` at ingest time so a retired rule can be shown as
   history but never served as current policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import Settings, settings
from app.providers import get_embedder
from app.providers.base import LLMError
from app.store import Store

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

AUTHORITY_TIER = {"policy": 1, "faq": 2, "ticket": 3}

FILE_SOURCE_TYPE = {"policies.md": "policy", "faqs.md": "faq", "tickets.md": "ticket"}

ENTRY_RE = re.compile(
    r"^#\s+(?P<id>(?:FAQ|POLICY|TICKET)-\d+)\s*[—\-–]\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)

# Stray section banners left in the sample files ("SECTION 2 — POLICY / ...").
NOISE_RE = re.compile(r"^SECTION\s+\d+\s*[—\-–].*$", re.MULTILINE)

DATE_RE = re.compile(
    r"(?P<label>Last reviewed|Last updated|Effective date|Effective|Updated|Reviewed)"
    r"\s*:\s*(?P<month>January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(?P<year>\d{4})",
    re.IGNORECASE,
)

MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}

STATUS_RE = re.compile(r"^STATUS:\s*(?P<status>.+?)\s*$", re.MULTILINE)

# Phrases the corpus uses to retire its own statements.
#
# These are deliberately specific. An earlier version of this list included
# bare "no longer" / "outdated" and produced false positives on ordinary prose
# ("if you no longer have access to your old university email", "an outdated
# browser"). Flagging those would have down-ranked healthy chunks, so every
# marker below names a *document* being retired, not a user situation.
DEPRECATION_MARKERS = [
    "should not be treated as the current standard policy",
    "should not be used when diagnosing current issues",
    "that instruction has been retired",
    "this recommendation has been removed",
    "that wording is outdated",
    "that wording has been replaced",
    "that information is obsolete",
    "the instructions are outdated",
    "appears to contain older instructions",
    "is no longer a universal requirement",
    "is no longer offered universally",
    "may no longer apply",
    "an older version of this article",
    "a previous version of",
    "the previous mobile help article",
    "older help-center documentation",
    "older documentation referred to",
    "archived documentation previously claimed",
    "an older instructor guide stated",
    "an older internal billing document",
    "an older 2024 article",
    "some older help articles",
]

# Backreference openers: when the marker sentence starts with one of these it
# is commenting on the *previous* sentence, which carries the actual claim.
BACKREF_OPENERS = ("that ", "this ", "it ", "those ", "these ")

# Controlled topic vocabulary. Keyword matching is deliberate: it is
# inspectable, deterministic, and costs nothing at ingest time. An LLM tagger
# is the obvious upgrade and is listed in the README trade-offs.
#
# Topics are a *soft* signal (they feed the lexical index and the agent-facing
# escalation routing), never a hard retrieval filter — a wrong tag should cost
# a little ranking, not make a document unreachable.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "refunds": ("refund", "money-back", "money back", "reimburse", "price adjustment", "price-adjustment", "refundable"),
    "billing": ("billing", "invoice", "charged", "charge on", "payment method", "authorization hold", "declined", "order number", "transaction"),
    "subscriptions": ("subscription", "auto-renewal", "auto-renew", "automatic renewal", "renewal payment", "annual plan", "billing period", "expires"),
    "disputes": ("chargeback", "dispute", "unauthorized", "fraud", "don't recognize", "do not recognize", "duplicate charge", "two charges"),
    "account_access": ("password", "sign-in page", "forgot password", "reset link", "social login", "account email", "change the email", "email address on the account"),
    "course_access": ("my learning", "access the course", "course does not appear", "enrollment", "course disappeared", "restore access", "orders & billing"),
    "progress": ("progress", "synchroniz", "completion event", "lesson progress", "modules", "resume the lesson"),
    "certificates": ("certificate", "completion page", "minimum assessment score", "final quiz", "final assessment"),
    "offline_mobile": ("offline", "download", "mobile application", "app store", "google play", "cellular", "iphone", "tablet"),
    "accessibility": ("accessib", "caption", "transcript", "screen reader", "assistive", "alternative text", "colour contrast", "color contrast"),
    "security": ("credentials", "compromis", "cvv", "security code", "verify ownership", "terminate unfamiliar sessions", "fraudulent activity"),
    "technical": ("browser", "playback", "video stays", "loading screen", "extension", "firewall", "incognito", "bandwidth", "operating system"),
    "organization_family": ("family plan", "organization", "classroom", "seat", "administrator", "learner profile", "share my", "shared account", "multi-user"),
    "instructor": ("instructor", "course creator", "published course", "copyright", "course description"),
}

# A topic must score at least MIN_TOPIC_SCORE *and* reach TOPIC_SCORE_RATIO of
# the document's strongest topic. The ratio is what stops a single incidental
# mention of "browser" from making a refund policy a technical document.
MIN_TOPIC_SCORE = 2
TOPIC_SCORE_RATIO = 0.34
MAX_TOPICS = 4

TARGET_CHARS = 900
MAX_CHARS = 1500
OVERLAP_CHARS = 180


# ---------------------------------------------------------------------------
# data holders
# ---------------------------------------------------------------------------


@dataclass
class ParsedDoc:
    doc_id: str
    source_type: str
    title: str
    body: str
    source_path: str
    effective_date: str | None = None
    date_label: str | None = None
    ticket_status: str | None = None
    topics: list[str] = field(default_factory=list)

    @property
    def authority_tier(self) -> int:
        return AUTHORITY_TIER[self.source_type]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()

    @property
    def source_uri(self) -> str:
        return f"{self.source_path}#{self.doc_id.lower()}"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def parse_file(path: Path, kb_root: Path) -> list[ParsedDoc]:
    text = NOISE_RE.sub("", path.read_text(encoding="utf-8"))
    source_type = FILE_SOURCE_TYPE.get(path.name)
    if source_type is None:
        return []

    rel = path.relative_to(kb_root.parent.parent).as_posix() if kb_root.parent.parent in path.parents else path.name
    matches = list(ENTRY_RE.finditer(text))
    docs: list[ParsedDoc] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        # Drop the horizontal-rule separators between entries.
        body = re.sub(r"^\s*-{3,}\s*$", "", body, flags=re.MULTILINE).strip()
        if not body:
            continue

        doc = ParsedDoc(
            doc_id=match.group("id"),
            source_type=source_type,
            title=match.group("title").strip(),
            body=body,
            source_path=rel,
        )
        _attach_date(doc)
        _attach_ticket_status(doc)
        doc.topics = infer_topics(f"{doc.title}\n{doc.body}")
        docs.append(doc)
    return docs


def _attach_date(doc: ParsedDoc) -> None:
    match = DATE_RE.search(doc.body)
    if not match:
        return
    month = MONTHS[match.group("month").lower()]
    doc.effective_date = f"{int(match.group('year')):04d}-{month:02d}-01"
    doc.date_label = match.group(0).strip()


def _attach_ticket_status(doc: ParsedDoc) -> None:
    if doc.source_type != "ticket":
        return
    match = STATUS_RE.search(doc.body)
    if not match:
        return
    raw = match.group("status").lower()
    # Order matters: an escalated ticket that also says "resolved" is escalated.
    if "escalat" in raw:
        doc.ticket_status = "escalated"
    elif "awaiting" in raw:
        doc.ticket_status = "awaiting_info"
    elif any(w in raw for w in ("pending", "requested", "opened", "reported", "monitoring")):
        doc.ticket_status = "pending"
    elif "resolved" in raw or "completed" in raw or "provided" in raw:
        doc.ticket_status = "resolved"
    else:
        doc.ticket_status = "pending"


def infer_topics(text: str) -> list[str]:
    """Rank topics by keyword *occurrences* and keep only the dominant ones.

    Counting occurrences rather than distinct keywords matters: a topic with
    one very precise term ("refund", said eight times) is a stronger signal
    than a topic with three incidental ones, and distinct-keyword counting
    got that backwards.
    """
    lowered = text.lower()
    scored = [
        (topic, sum(lowered.count(w) for w in words))
        for topic, words in TOPIC_KEYWORDS.items()
    ]
    scored = [(t, n) for t, n in scored if n > 0]
    if not scored:
        return ["general"]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))

    top = scored[0][1]
    cutoff = max(MIN_TOPIC_SCORE, top * TOPIC_SCORE_RATIO)
    confident = [t for t, n in scored if n >= cutoff][:MAX_TOPICS]
    # Nothing cleared the bar: keep the single best weak signal so the document
    # is still reachable by topic rather than dumped into "general".
    return confident or [scored[0][0]]


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_document(doc: ParsedDoc) -> list[dict[str, Any]]:
    """Split an entry into retrieval units, preserving authored structure.

    FAQ entries carry their QUESTION into every chunk's `heading` because the
    question is the strongest retrieval signal in the whole record — users
    phrase their problem the way the FAQ phrases the question, not the way the
    answer phrases the resolution.
    """
    heading = _heading_for(doc)
    segments = _segments_for(doc)

    chunks: list[str] = []
    buffer = ""
    for seg in segments:
        if not seg.strip():
            continue
        if len(seg) > MAX_CHARS:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_split_long(seg))
            continue
        candidate = f"{buffer}\n\n{seg}".strip() if buffer else seg.strip()
        if len(candidate) <= TARGET_CHARS or not buffer:
            buffer = candidate
        else:
            chunks.append(buffer.strip())
            buffer = _overlap_tail(buffer) + seg.strip()
    if buffer.strip():
        chunks.append(buffer.strip())

    out: list[dict[str, Any]] = []
    for ordinal, body in enumerate(chunks):
        out.append(
            {
                "chunk_id": f"{doc.doc_id}#c{ordinal}",
                "doc_id": doc.doc_id,
                "ordinal": ordinal,
                "text": body,
                "heading": heading,
                "token_estimate": max(1, len(body) // 4),
                "source_type": doc.source_type,
                "authority_tier": doc.authority_tier,
                "effective_date": doc.effective_date,
                "topics": json.dumps(doc.topics),
                "has_deprecation_notice": int(bool(_find_markers(body))),
                "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            }
        )
    return out


def _heading_for(doc: ParsedDoc) -> str:
    base = f"{doc.doc_id} — {doc.title}"
    question = re.search(r"^QUESTION:\s*(.+?)(?:\n\s*\n|\nANSWER:)", doc.body, re.DOTALL | re.MULTILINE)
    if question:
        return f"{base} | {' '.join(question.group(1).split())}"
    return base


def _segments_for(doc: ParsedDoc) -> list[str]:
    if doc.source_type == "ticket":
        # Keep USER/AGENT exchanges intact: a reply is meaningless without the
        # turn it answers, so we pair them before considering a split.
        turns = re.split(r"\n(?=(?:USER|AGENT|STATUS):)", doc.body)
        paired: list[str] = []
        for turn in turns:
            if paired and paired[-1].lstrip().startswith("USER:") and turn.lstrip().startswith("AGENT:"):
                paired[-1] = f"{paired[-1].strip()}\n{turn.strip()}"
            else:
                paired.append(turn.strip())
        return paired
    return [p.strip() for p in re.split(r"\n\s*\n", doc.body)]


def _split_long(segment: str) -> list[str]:
    sentences = _SENT_SPLIT.split(segment)
    out, buffer = [], ""
    for sentence in sentences:
        if len(buffer) + len(sentence) + 1 > TARGET_CHARS and buffer:
            out.append(buffer.strip())
            buffer = _overlap_tail(buffer) + sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer.strip():
        out.append(buffer.strip())
    return out


def _overlap_tail(text: str) -> str:
    """Carry the last sentence forward so a rule and its qualifier stay together."""
    tail = text[-OVERLAP_CHARS:]
    parts = _SENT_SPLIT.split(tail)
    carried = parts[-1] if len(parts) > 1 else tail
    return f"{carried.strip()} "


# ---------------------------------------------------------------------------
# deprecation extraction
# ---------------------------------------------------------------------------


def _find_markers(text: str) -> list[str]:
    lowered = text.lower()
    return [m for m in DEPRECATION_MARKERS if m in lowered]


def extract_deprecated_claims(doc: ParsedDoc, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        sentences = [s.strip() for s in _SENT_SPLIT.split(chunk["text"].replace("\n", " ")) if s.strip()]
        for i, sentence in enumerate(sentences):
            lowered = sentence.lower()
            hit = next((m for m in DEPRECATION_MARKERS if m in lowered), None)
            if not hit:
                continue
            claim = sentence
            # "That information is obsolete." carries no claim of its own; the
            # retired statement is the sentence before it.
            if i > 0 and lowered.startswith(BACKREF_OPENERS):
                claim = f"{sentences[i - 1]} {sentence}"
            key = claim[:120]
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                {
                    "doc_id": doc.doc_id,
                    "chunk_id": chunk["chunk_id"],
                    "claim_text": claim,
                    "marker": hit,
                    "superseded_by": doc.doc_id,  # the same doc states the current rule
                }
            )
    return _dedupe_claims(claims)


def _dedupe_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the longest form of each overlapping claim.

    Two markers often fire on the same retirement — one on the sentence that
    states the old rule, one on the sentence that revokes it — and the
    backreference merge makes the second a superset of the first. Storing both
    would double-count the same stale statement in the prompt.
    """
    ordered = sorted(claims, key=lambda c: len(c["claim_text"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for claim in ordered:
        if any(claim["claim_text"] in k["claim_text"] for k in kept):
            continue
        kept.append(claim)
    return sorted(kept, key=lambda c: (c["chunk_id"] or "", c["claim_text"]))


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_ingest(
    cfg: Settings | None = None,
    store: Store | None = None,
    *,
    force: bool = False,
    embed: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg = cfg or settings
    owns_store = store is None
    store = store or Store(cfg)
    store.init_schema()

    embedder = get_embedder(cfg) if embed else None
    model_name = embedder.model if embedder and embedder.is_live else None
    run_id = store.start_ingest_run(str(cfg.kb_dir), model_name)

    counters = {"docs_seen": 0, "docs_changed": 0, "chunks_written": 0, "vectors_written": 0}
    status = "ok"

    try:
        files = sorted(p for p in cfg.kb_dir.glob("*.md") if p.name in FILE_SOURCE_TYPE)
        if not files:
            raise FileNotFoundError(f"no knowledge-base markdown found in {cfg.kb_dir}")

        with store.tx():
            for path in files:
                for doc in parse_file(path, cfg.kb_dir):
                    counters["docs_seen"] += 1
                    previous_hash = store.get_document_hash(doc.doc_id)
                    if previous_hash == doc.content_hash and not force:
                        continue  # unchanged: skip re-chunk and re-embed
                    counters["docs_changed"] += 1

                    store.upsert_document(
                        {
                            "doc_id": doc.doc_id,
                            "source_type": doc.source_type,
                            "title": doc.title,
                            "source_path": doc.source_path,
                            "source_uri": doc.source_uri,
                            "authority_tier": doc.authority_tier,
                            "effective_date": doc.effective_date,
                            "date_label": doc.date_label,
                            "has_explicit_date": int(bool(doc.effective_date)),
                            "ticket_status": doc.ticket_status,
                            "topics": json.dumps(doc.topics),
                            "raw_text": doc.body,
                            "content_hash": doc.content_hash,
                        }
                    )
                    chunks = chunk_document(doc)
                    store.replace_chunks(doc.doc_id, chunks)
                    store.add_deprecated_claims(extract_deprecated_claims(doc, chunks))
                    counters["chunks_written"] += len(chunks)
                    if verbose:
                        print(f"  {doc.doc_id:<12} {len(chunks)} chunk(s)  topics={','.join(doc.topics)}")

        if embedder is not None and embedder.is_live:
            counters["vectors_written"] = _embed_missing(store, embedder, verbose=verbose)
        elif verbose:
            print("  embeddings skipped — no GEMINI_API_KEY; retrieval will use BM25 only")

    except Exception as exc:  # noqa: BLE001 — recorded, then re-raised
        status = f"failed: {exc}"[:200]
        store.finish_ingest_run(run_id, **counters, status=status)
        if owns_store:
            store.close()
        raise

    store.finish_ingest_run(run_id, **counters, status=status)
    result = {"run_id": run_id, **counters, "stats": store.stats()}
    if owns_store:
        store.close()
    return result


def _embed_missing(store: Store, embedder: Any, *, verbose: bool) -> int:
    pending = store.chunks_missing_embeddings(embedder.model)
    if not pending:
        if verbose:
            print("  embeddings up to date")
        return 0
    if verbose:
        print(f"  embedding {len(pending)} chunk(s) with {embedder.model} ...")

    # Heading is prepended at embed time (not stored in `text`) so the vector
    # carries the FAQ question while citations still quote only the body.
    payloads = [f"{r['heading']}\n\n{r['text']}" if r["heading"] else r["text"] for r in pending]
    try:
        vectors = embedder.embed(payloads, is_query=False)
    except LLMError as exc:
        print(f"  embedding failed ({exc}); continuing with BM25-only retrieval")
        return 0

    written = store.write_embeddings(
        embedder.model, embedder.dim, zip([r["chunk_id"] for r in pending], vectors)
    )
    store.conn.commit()
    return written


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest the LearnForge knowledge base")
    parser.add_argument("--force", action="store_true", help="re-chunk every document")
    parser.add_argument("--no-embed", action="store_true", help="skip the embedding pass")
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"Ingesting from {settings.kb_dir}")
    print(f"  {settings.describe()}")
    result = run_ingest(force=args.force, embed=not args.no_embed)
    stats = result["stats"]
    print(
        f"\nDone. {result['docs_seen']} docs seen, {result['docs_changed']} changed, "
        f"{result['chunks_written']} chunks written, {result['vectors_written']} vectors written."
    )
    print(
        f"Corpus: {stats['documents']} documents {stats['documents_by_type']}, "
        f"{stats['chunks']} chunks, {stats['embeddings']} vectors, "
        f"{stats['deprecated_claims']} deprecated claims quarantined."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
