"""Hybrid retrieval: BM25 + dense, fused, re-scored on metadata, then reranked.

Why hybrid rather than dense-only
---------------------------------
Support queries are full of exact tokens that embeddings blur: "14 days",
"CVV", "Atomic Orbitals", "FAQ-02", an order number. BM25 nails those and
misses paraphrase ("my course vanished" vs "course does not appear"); dense
retrieval is the mirror image. Running both and fusing by rank costs one extra
SQL query and removes an entire class of silent misses.

Why Reciprocal Rank Fusion rather than score blending
-----------------------------------------------------
BM25 scores and cosine similarities live on incomparable scales, and the BM25
scale additionally shifts with query length. Any fixed alpha for
`alpha*bm25 + (1-alpha)*cosine` is a number that needs recalibrating every
time either side changes. RRF only reads *ranks*, so it needs no calibration
and cannot be destabilised by one branch's outlier scores.

Why a metadata pass after fusion
--------------------------------
Pure relevance is the wrong objective on this corpus. A ticket transcript
where one agent said "our standard refund period is 14 days" is lexically a
perfect match for "how long do I have to refund?", but it is tier-3 anecdote.
POLICY-02 is the answer. The re-scoring pass encodes that: authority first,
then recency, then a penalty for chunks carrying retired statements.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np

from app.config import Settings, settings
from app.providers import get_embedder, get_llm
from app.providers.base import LLM, Embedder, LLMError
from app.store import Store
from app.text import content_terms

# --- metadata re-scoring weights ------------------------------------------
# Multiplicative, centred on 1.0, and deliberately gentle. RRF compresses every
# score into a narrow band (~0.40-0.50 here), so even a modest multiplier moves
# a candidate several ranks. These were tightened after the eval suite showed
# an 18% authority bonus evicting the single best lexical match.
AUTHORITY_WEIGHT = {1: 1.15, 2: 1.06, 3: 0.93}  # policy / faq / ticket
NO_DATE_PENALTY = 0.97
STALE_PENALTY = 0.90
FRESH_BONUS = 1.04
FRESH_WINDOW_DAYS = 200

# NOTE: there is deliberately no penalty for `has_deprecation_notice`.
# An earlier version down-ranked chunks containing withdrawn statements, which
# turned out to be exactly wrong. POLICY-05's second chunk reads "An older
# instructor guide stated that courses must contain at least five quizzes.
# This is no longer a universal requirement." — that chunk is the *best*
# answer to "do courses need five quizzes?", and penalising it promoted
# passages that mention quizzes without the correction. The flag still travels
# with the chunk: it annotates the prompt (RETIRED STATEMENTS) and arms the
# verifier's leak check. It just must not affect ranking.

# At most this many chunks from any one document may occupy the final context.
# Without a cap, two adjacent chunks of the same FAQ routinely take three of
# five slots and crowd out the policy that actually governs the answer.
MAX_CHUNKS_PER_DOC = 2
MAX_DEPRECATED_CLAIMS = 6

# Conflict detection: how much text around a quantity counts as its subject,
# and how many subject words two quantities must share before a difference in
# value is treated as a contradiction rather than a coincidence.
ANCHOR_WINDOW = 90
MIN_SHARED_ANCHORS = 2

# Claim patterns used for cheap, explainable conflict detection.
_QUANTITY_RE = re.compile(
    r"\b(?:(\d{1,3})|(one|two|three|four|five|six|seven|eight|nine|ten))"
    r"[-\s]?(day|days|hour|hours|month|months|week|weeks|user|users|quiz|quizzes|seat|seats)\b",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_UNIT_CANON = {
    "day": "day", "days": "day", "hour": "hour", "hours": "hour",
    "month": "month", "months": "month", "week": "week", "weeks": "week",
    "user": "user", "users": "user", "quiz": "quiz", "quizzes": "quiz",
    "seat": "seat", "seats": "seat",
}


@dataclass
class Candidate:
    chunk_id: str
    doc_id: str
    text: str
    heading: str
    source_type: str
    authority_tier: int
    effective_date: str | None
    date_label: str | None
    ticket_status: str | None
    topics: list[str]
    has_deprecation_notice: bool
    source_uri: str
    title: str
    lexical_rank: int | None = None
    dense_rank: int | None = None
    lexical_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    final_score: float = 0.0
    boosts: dict[str, float] = field(default_factory=dict)
    is_stale: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "source_type": self.source_type,
            "authority_tier": self.authority_tier,
            "date_label": self.date_label,
            "ticket_status": self.ticket_status,
            "topics": self.topics,
            "has_deprecation_notice": self.has_deprecation_notice,
            "is_stale": self.is_stale,
            "lexical_score": round(self.lexical_score, 4),
            "dense_score": round(self.dense_score, 4),
            "fused_score": round(self.fused_score, 4),
            "final_score": round(self.final_score, 4),
            "boosts": {k: round(v, 3) for k, v in self.boosts.items()},
            "text": self.text,
            "source_uri": self.source_uri,
        }


@dataclass
class RetrievalResult:
    query: str
    candidates: list[Candidate]
    mode: str                         # 'hybrid' | 'lexical-only' | 'empty'
    top_score: float                  # absolute relevance of the best hit, 0..1
    margin: float                     # separation between #1 and #2
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    deprecated_claims: list[dict[str, Any]] = field(default_factory=list)
    reranked: bool = False

    @property
    def doc_ids(self) -> list[str]:
        seen, out = set(), []
        for c in self.candidates:
            if c.doc_id not in seen:
                seen.add(c.doc_id)
                out.append(c.doc_id)
        return out

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "top_score": round(self.top_score, 4),
            "margin": round(self.margin, 4),
            "reranked": self.reranked,
            "candidates": [c.as_dict() for c in self.candidates],
            "conflicts": self.conflicts,
            "deprecated_claims": self.deprecated_claims,
        }


class Retriever:
    def __init__(
        self,
        store: Store,
        cfg: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.store = store
        self.embedder = embedder if embedder is not None else get_embedder(self.cfg)
        self.llm = llm if llm is not None else get_llm(self.cfg)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.cfg.retrieval_top_k
        n = self.cfg.retrieval_candidates

        lexical = self.store.lexical_search(query, n)
        dense = self._dense_search(query, n)
        mode = "hybrid" if dense else ("lexical-only" if lexical else "empty")

        if not lexical and not dense:
            return RetrievalResult(query=query, candidates=[], mode="empty", top_score=0.0, margin=0.0)

        candidates = self._fuse(lexical, dense)
        self._apply_metadata_scoring(candidates)
        candidates.sort(key=lambda c: c.final_score, reverse=True)

        # Rerank a wider slice than we will keep: the reranker's job is to fix
        # fusion's mistakes, which it cannot do if fusion already truncated.
        shortlist = candidates[: max(top_k * 2, top_k + 4)]
        reranked = False
        if self.cfg.enable_llm_rerank and self.llm.is_live and len(shortlist) > top_k:
            shortlist, reranked = self._llm_rerank(query, shortlist, top_k)

        final = _diversify(shortlist, top_k)
        final = _guarantee_top_raw_hits(final, candidates, lexical, dense, top_k)
        top_score = self._absolute_top_score(query, final, dense_available=bool(dense))
        margin = (
            final[0].final_score - final[1].final_score if len(final) > 1 else final[0].final_score
        ) if final else 0.0

        result = RetrievalResult(
            query=query,
            candidates=final,
            mode=mode,
            top_score=top_score,
            margin=round(margin, 4),
            reranked=reranked,
        )
        result.deprecated_claims = self._scoped_deprecated_claims(result.doc_ids)
        result.conflicts = detect_conflicts(final, result.deprecated_claims, query=query)
        return result

    def _scoped_deprecated_claims(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        """Only the retired statements that belong to retrieved documents, in
        retrieval order and capped.

        These go into the generation prompt as things the model must not assert
        as current. Sending every claim in the corpus would be both wasteful
        and counterproductive — a long list of irrelevant warnings dilutes the
        two that actually apply to this question.
        """
        rank = {doc_id: i for i, doc_id in enumerate(doc_ids[:3])}
        claims = [c for c in self.store.deprecated_claims_for(list(rank)) if c["doc_id"] in rank]
        claims.sort(key=lambda c: rank[c["doc_id"]])
        return claims[:MAX_DEPRECATED_CLAIMS]

    # ------------------------------------------------------------------
    def _dense_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if not self.embedder.is_live:
            return []
        ids, matrix = self.store.vector_matrix(self.embedder.model)
        if not ids:
            return []
        try:
            vectors = self.embedder.embed([query], is_query=True)
        except LLMError:
            # Embedding outage must degrade to BM25, not fail the request.
            return []
        if not vectors:
            return []
        q = np.asarray(vectors[0], dtype="float32")
        # Both sides are L2-normalised, so the dot product IS cosine similarity.
        scores = matrix @ q
        order = np.argsort(-scores)[:limit]
        return [(ids[i], float(scores[i])) for i in order]

    # ------------------------------------------------------------------
    def _fuse(
        self, lexical: list[tuple[str, float]], dense: list[tuple[str, float]]
    ) -> list[Candidate]:
        k = self.cfg.rrf_k
        contributions: dict[str, float] = {}
        lex_rank: dict[str, int] = {}
        den_rank: dict[str, int] = {}
        lex_score: dict[str, float] = {}
        den_score: dict[str, float] = {}

        for rank, (chunk_id, score) in enumerate(lexical, start=1):
            contributions[chunk_id] = contributions.get(chunk_id, 0.0) + 1.0 / (k + rank)
            lex_rank[chunk_id] = rank
            lex_score[chunk_id] = score
        for rank, (chunk_id, score) in enumerate(dense, start=1):
            contributions[chunk_id] = contributions.get(chunk_id, 0.0) + 1.0 / (k + rank)
            den_rank[chunk_id] = rank
            den_score[chunk_id] = score

        rows = self.store.get_chunks(list(contributions))
        # Normalise so a chunk ranked #1 by both branches scores 1.0. This is
        # a readability convenience for logs; it is monotonic in the raw RRF
        # sum, so it changes no ordering.
        max_possible = 2.0 / (k + 1)

        out: list[Candidate] = []
        for chunk_id, raw in contributions.items():
            row = rows.get(chunk_id)
            if row is None:
                continue
            out.append(
                Candidate(
                    chunk_id=chunk_id,
                    doc_id=row["doc_id"],
                    text=row["text"],
                    heading=row["heading"] or "",
                    source_type=row["source_type"],
                    authority_tier=row["authority_tier"],
                    effective_date=row["effective_date"],
                    date_label=row["date_label"],
                    ticket_status=row["ticket_status"],
                    topics=json.loads(row["topics"] or "[]"),
                    has_deprecation_notice=bool(row["has_deprecation_notice"]),
                    source_uri=row["source_uri"] or "",
                    title=row["title"],
                    lexical_rank=lex_rank.get(chunk_id),
                    dense_rank=den_rank.get(chunk_id),
                    lexical_score=lex_score.get(chunk_id, 0.0),
                    dense_score=den_score.get(chunk_id, 0.0),
                    fused_score=raw / max_possible,
                )
            )
        return out

    # ------------------------------------------------------------------
    def _apply_metadata_scoring(self, candidates: list[Candidate]) -> None:
        today = self.cfg.today()
        for c in candidates:
            boosts: dict[str, float] = {}

            boosts["authority"] = AUTHORITY_WEIGHT.get(c.authority_tier, 1.0)

            age_days = _age_in_days(c.effective_date, today)
            if age_days is None:
                boosts["undated"] = NO_DATE_PENALTY
            elif age_days > self.cfg.stale_after_days:
                boosts["stale"] = STALE_PENALTY
                c.is_stale = True
            elif age_days <= FRESH_WINDOW_DAYS:
                boosts["fresh"] = FRESH_BONUS

            multiplier = 1.0
            for value in boosts.values():
                multiplier *= value
            c.boosts = boosts
            c.final_score = c.fused_score * multiplier

    # ------------------------------------------------------------------
    def _llm_rerank(
        self, query: str, shortlist: list[Candidate], top_k: int
    ) -> tuple[list[Candidate], bool]:
        """Ask the model to order candidates by usefulness to *this* question.

        Fusion is lexical/semantic; it cannot tell that a passage about
        *subscription* refunds does not answer a question about a *course*
        refund. That distinction is exactly what TICKET-08 is about, and it is
        the kind of judgement a reranker is for.

        A reranker failure is non-fatal: we keep the fused order and continue.
        """
        blocks = "\n\n".join(
            f"[{c.doc_id}] ({c.source_type}, chunk {c.chunk_id})\n{_truncate(c.text, 600)}"
            for c in shortlist
        )
        system = (
            "TASK: rerank\n"
            "You rank knowledge-base passages by how directly they answer a customer-support "
            "question for the LearnForge e-learning platform.\n"
            "Rules:\n"
            "- A passage that states the governing rule outranks one that merely mentions the topic.\n"
            "- Policy documents outrank FAQs; FAQs outrank past ticket transcripts, because a "
            "ticket records what one agent said once and is not authoritative.\n"
            "- A passage about a different product line (subscription vs individual course) is "
            "NOT a match, however similar the wording.\n"
            'Reply with JSON only: {"ranking": ["DOC-ID", ...]} — most useful first, '
            "using the exact ids shown in brackets, omitting passages that are irrelevant."
        )
        user = f"QUESTION: {query}\n\nPASSAGES:\n{blocks}"

        try:
            data = self.llm.complete_json(system, user, temperature=0.0, max_tokens=900)
            ranking = [str(x).strip().upper() for x in data.get("ranking", [])]
        except (LLMError, AttributeError, TypeError):
            return shortlist, False
        if not ranking:
            return shortlist, False

        # Rank by document, then keep each document's chunks in fused order.
        priority = {doc_id: i for i, doc_id in enumerate(ranking)}
        kept = [c for c in shortlist if c.doc_id in priority]
        dropped = [c for c in shortlist if c.doc_id not in priority]
        if not kept:
            return shortlist, False
        kept.sort(key=lambda c: (priority[c.doc_id], -c.final_score))
        # Dropped candidates stay available as a tail: the reranker is advisory,
        # and we would rather over-supply context than lose the only good hit.
        return kept + dropped, True

    # ------------------------------------------------------------------
    def _absolute_top_score(
        self, query: str, final: list[Candidate], *, dense_available: bool
    ) -> float:
        """A confidence signal that is comparable *across* queries.

        RRF scores are not: they describe agreement between two rankings of
        whatever came back, so a question with no good answer in the corpus
        still yields a high RRF score for its least-bad hit. Cosine similarity
        *is* comparable, so we use it whenever dense retrieval is live.

        With BM25 only we fall back to a length-normalised BM25 proxy. BM25
        grows with the number of matching query terms, so dividing by the term
        count keeps a one-word and a ten-word question on roughly the same
        scale. It is coarser than cosine and the README flags it as the main
        calibration limitation of the keyless mode.
        """
        if not final:
            return 0.0
        if dense_available:
            return round(max(c.dense_score for c in final), 4)

        terms = content_terms(query)
        if not terms:
            return 0.0

        # Raw BM25 alone is a poor abstention signal on a corpus this small:
        # "refund" appears in a third of the documents, so its IDF is low and a
        # perfectly answerable refund question scores lower than a narrow
        # question about captions. Coverage — how much of what the user asked
        # about actually appears in the best passage — is the stable half of
        # the signal, so we average the two.
        normalised_bm25 = min(1.0, max(c.lexical_score for c in final) / (2.2 * len(terms)))
        coverage = max(_term_coverage(terms, c) for c in final)
        return round(0.5 * normalised_bm25 + 0.5 * coverage, 4)


# ---------------------------------------------------------------------------
# conflict detection
# ---------------------------------------------------------------------------


USER_CLAIM = "USER-CLAIM"


def detect_conflicts(
    candidates: list[Candidate],
    deprecated_claims: list[dict[str, Any]] | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Flag retrieved passages that assert different quantities for the same unit.

    This is the cheap, deterministic half of conflict handling: "14 days" in
    POLICY-02, the "7-day refund period" in the archived note, and the "30-day
    money-back guarantee" a learner quotes in TICKET-03 all reduce to competing
    values for the unit `day`. The generator does the semantic half and can
    overrule a false positive; this pass guarantees we never miss the blatant
    numeric case, which is the one learners escalate over.

    Values that appear *only* inside a quarantined claim are marked retired and
    do not count towards a live conflict. This matters: POLICY-02 states both
    "14 days" and "7-day", but the corpus already resolves that itself, and
    treating it as an open contradiction would escalate the single most common
    refund question in the whole knowledge base.
    """
    retired = {(q.unit, q.value) for q in _quantity_mentions(
        " ".join(c.get("claim_text", "") for c in (deprecated_claims or [])), "__retired__"
    )}

    mentions: list[_Quantity] = []
    for c in candidates:
        mentions.extend(_quantity_mentions(c.text, c.doc_id))

    # The learner's own message is a source too. TICKET-03 ("Your website said
    # 30 days when I bought it") and TICKET-08 ("the cancellation page says
    # cancel within 14 days for a full refund") are both escalations caused by
    # a figure the *user* is holding us to that the knowledge base does not
    # support. Corpus-versus-corpus checking alone never sees those.
    if query:
        mentions.extend(_quantity_mentions(query, USER_CLAIM))

    # Group by unit, then only pair up quantities that are talking about the
    # same subject. Without the subject test, "14 days" (refund window) and
    # "30 minutes"/"90 days" from an unrelated article register as a policy
    # contradiction, and the gate escalates half the traffic.
    grouped: dict[str, list[_Quantity]] = {}
    for q in mentions:
        grouped.setdefault(q.unit, []).append(q)

    conflicts: list[dict[str, Any]] = []
    for unit, group in grouped.items():
        live = [q for q in group if (q.unit, q.value) not in retired]
        clashing: dict[int, set[str]] = {}
        shared_subject: set[str] = set()
        for i, a in enumerate(live):
            for b in live[i + 1 :]:
                if a.value == b.value:
                    continue
                overlap = a.anchors & b.anchors
                if len(overlap) < MIN_SHARED_ANCHORS:
                    continue
                clashing.setdefault(a.value, set()).add(a.doc_id)
                clashing.setdefault(b.value, set()).add(b.doc_id)
                shared_subject |= overlap
        if len(clashing) < 2:
            continue
        conflicts.append(
            {
                "unit": unit,
                "subject": sorted(shared_subject)[:6],
                "values": sorted(clashing),
                "sources": {str(v): sorted(d) for v, d in sorted(clashing.items())},
                "retired_values": sorted({q.value for q in group if (q.unit, q.value) in retired}),
            }
        )
    return conflicts


@dataclass
class _Quantity:
    unit: str
    value: int
    doc_id: str
    anchors: frozenset[str]


def _quantity_mentions(text: str, doc_id: str) -> list[_Quantity]:
    """Every '<n> <unit>' in a passage, with the words around it.

    The surrounding words are the subject test: "14 days" next to {refund,
    period, course} and "7 days" next to {refund, period, digital} share
    enough context to be about the same rule, while "5 users" next to {family,
    plan} shares nothing with "5 quizzes" next to {assessment, category}.
    """
    out: list[_Quantity] = []
    for match in _QUANTITY_RE.finditer(text):
        digits, word, unit_raw = match.group(1), match.group(2), match.group(3)
        value = int(digits) if digits else _WORD_NUMBERS[word.lower()]
        unit = _UNIT_CANON[unit_raw.lower()]
        start = max(0, match.start() - ANCHOR_WINDOW)
        end = min(len(text), match.end() + ANCHOR_WINDOW)
        anchors = set(content_terms(text[start:end])) - {unit, f"{unit}s", str(value)}
        out.append(_Quantity(unit, value, doc_id, frozenset(anchors)))
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _guarantee_top_raw_hits(
    final: list[Candidate],
    all_candidates: list[Candidate],
    lexical: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    top_k: int,
) -> list[Candidate]:
    """Make sure each branch's single best raw hit survives re-scoring.

    Metadata weights and the LLM reranker are heuristics applied *after* the
    two retrievers have spoken, and either can be wrong. The eval suite caught
    both doing it: TICKET-02 is the only document in the corpus about being
    charged twice, and the tier-3 authority penalty alone pushed it out of the
    top five.

    A cheap floor removes the whole class of problem: whatever else happens,
    the top BM25 chunk and the top dense chunk are in the context. Each one
    displaces the current lowest-scoring occupant rather than growing the
    context, so the token budget is unchanged.
    """
    if not final:
        return final
    by_id = {c.chunk_id: c for c in all_candidates}
    must_have = [b[0][0] for b in (lexical, dense) if b]
    out = list(final)
    for chunk_id in must_have:
        if chunk_id in {c.chunk_id for c in out} or chunk_id not in by_id:
            continue
        if len(out) >= top_k:
            out.pop()  # `out` stays score-ordered, so this is the weakest slot
        out.append(by_id[chunk_id])
        out.sort(key=lambda c: c.final_score, reverse=True)
    return out[:top_k]


def _term_coverage(terms: list[str], candidate: Candidate) -> float:
    """Fraction of the question's content terms that appear in one passage.

    Prefix matching mirrors the porter stemmer used by the FTS index, so
    "lessons" in the question matches "lesson" in the passage.
    """
    haystack = f"{candidate.heading} {candidate.text}".lower()
    hits = sum(1 for t in terms if t in haystack or t[:-1] in haystack)
    return hits / len(terms)


def _diversify(candidates: list[Candidate], top_k: int) -> list[Candidate]:
    """Take the best `top_k` while capping how many chunks one document may own.

    Cheap source diversification rather than full MMR: the corpus is small and
    each document is already a tight semantic unit, so redundancy shows up as
    "same doc, adjacent chunk" far more often than as "two docs saying the
    same thing". If the cap starves the result we top up from the remainder so
    we never return fewer than `top_k` when candidates exist.
    """
    per_doc: dict[str, int] = {}
    picked: list[Candidate] = []
    overflow: list[Candidate] = []
    for c in candidates:
        if per_doc.get(c.doc_id, 0) < MAX_CHUNKS_PER_DOC:
            per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
            picked.append(c)
        else:
            overflow.append(c)
        if len(picked) == top_k:
            return picked
    return (picked + overflow)[:top_k]


def _age_in_days(effective_date: str | None, today: date) -> int | None:
    if not effective_date:
        return None
    try:
        return (today - datetime.fromisoformat(effective_date).date()).days
    except ValueError:
        return None


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
