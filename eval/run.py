"""Evaluation harness.

    python -m eval.run                 # full suite
    python -m eval.run --bucket stale  # one bucket
    python -m eval.run --judge         # + LLM-as-judge answer quality
    python -m eval.run --retrieval-only  # no generation; fast, deterministic

What is measured and why
------------------------
*Retrieval* (recall@k, MRR) is separated from *answering* because they fail
differently and are fixed differently. A drop in recall is a chunking or
ranking problem; a drop in faithfulness with healthy recall is a prompt or
model problem. One blended score would hide both.

*Hallucination* is measured three ways, because no single proxy is honest:
  - invalid-citation rate: cited a document that was never retrieved. Exact,
    deterministic, zero false positives.
  - retired-claim leak rate: restated a withdrawn policy as current. This is
    the corpus-specific failure that generic RAG metrics miss entirely,
    because the bad text IS in the retrieved context.
  - unsupported-claim rate: the verifier's entailment judgement. Broadest
    coverage, and the least trustworthy, since it is an LLM grading an LLM.

*Abstention and escalation* get precision and recall of their own. They are
the safety valves, and they fail in opposite, equally bad directions: a system
that never escalates is dangerous, one that always escalates is useless.

Every run writes a JSON report to eval_reports/ so two runs can be diffed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.table import Table

from app.config import ROOT, settings
from app.terminal import make_console
from app.pipeline import SupportAssistant, TurnResult
from app.providers.base import LLMError
from app.store import Store

console = make_console()
GOLDEN = Path(__file__).parent / "golden_set.yaml"
REPORTS = ROOT / "eval_reports"

ANSWER_ACTIONS = {"answered", "answered_with_caveat"}


@dataclass
class CaseResult:
    case_id: str
    bucket: str
    question: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    action: str = ""
    expected_actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    groundedness: float = 0.0
    retrieved: list[str] = field(default_factory=list)
    expected_docs: list[str] = field(default_factory=list)
    hit: bool | None = None
    reciprocal_rank: float | None = None
    invalid_citations: list[str] = field(default_factory=list)
    retired_leaks: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    judge_score: float | None = None
    judge_notes: str = ""
    latency_ms: int = 0
    reply: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def load_cases(bucket: str | None = None, only: str | None = None) -> list[dict[str, Any]]:
    data = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    cases = data["cases"]
    if bucket:
        cases = [c for c in cases if c["bucket"] == bucket]
    if only:
        cases = [c for c in cases if c["id"] == only]
    return cases


def run_case(
    assistant: SupportAssistant, case: dict[str, Any], *, retrieval_only: bool
) -> CaseResult:
    turns: list[str] = case.get("turns") or [case["question"]]
    expected_docs = case.get("expect_docs") or []
    expected_actions = case.get("expect_action") or []
    if isinstance(expected_actions, str):
        expected_actions = [expected_actions]

    result = CaseResult(
        case_id=case["id"],
        bucket=case["bucket"],
        question=turns[-1],
        passed=True,
        expected_actions=expected_actions,
        expected_docs=expected_docs,
    )

    if retrieval_only:
        # Multi-turn cases are meaningless without the rewriter, so in
        # retrieval-only mode we concatenate the turns rather than pretending
        # the last one stands alone.
        query = " ".join(turns) if len(turns) > 1 else turns[0]
        started = time.perf_counter()
        retrieval = assistant.retriever.retrieve(query)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        result.retrieved = retrieval.doc_ids
        _score_retrieval(result)
        result.passed = result.hit is not False
        if result.hit is False:
            result.failures.append(f"none of {expected_docs} retrieved")
        return result

    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    turn: TurnResult | None = None
    for message in turns:
        turn = assistant.ask(message, session_id=session_id)
    assert turn is not None

    diagnostics = turn.diagnostics
    verification = diagnostics.get("verification", {})

    result.action = turn.action
    result.confidence = turn.confidence
    result.groundedness = float(verification.get("groundedness", 0.0))
    result.retrieved = [s["doc_id"] for s in turn.sources]
    result.invalid_citations = verification.get("invalid_citations", [])
    result.retired_leaks = verification.get("retired_claim_leaks", [])
    result.unsupported_claims = verification.get("unsupported_claims", [])
    result.latency_ms = turn.latency_ms
    result.reply = turn.reply

    _score_retrieval(result)

    # --- assertions ---------------------------------------------------
    if expected_actions and turn.action not in expected_actions:
        result.failures.append(f"action {turn.action!r}, expected one of {expected_actions}")
    if expected_docs and result.hit is False:
        result.failures.append(f"retrieval missed all of {expected_docs}")
    if case.get("expect_queue") and turn.queue and turn.queue != case["expect_queue"]:
        result.failures.append(f"queue {turn.queue!r}, expected {case['expect_queue']!r}")
    if result.invalid_citations:
        result.failures.append(f"cited unavailable docs {result.invalid_citations}")
    if case.get("forbid_retired", True) and result.retired_leaks:
        result.failures.append(f"restated withdrawn claim: {result.retired_leaks}")
    for pattern in case.get("forbid_patterns") or []:
        if re.search(pattern, turn.reply):
            result.failures.append(f"reply matched forbidden pattern {pattern!r}")

    result.passed = not result.failures
    return result


def _score_retrieval(result: CaseResult) -> None:
    if not result.expected_docs:
        return
    expected = set(result.expected_docs)
    result.hit = any(d in expected for d in result.retrieved)
    for rank, doc_id in enumerate(result.retrieved, start=1):
        if doc_id in expected:
            result.reciprocal_rank = 1.0 / rank
            return
    result.reciprocal_rank = 0.0


# ---------------------------------------------------------------------------
# LLM-as-judge (optional)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """TASK: judge
You grade a customer-support answer against a list of reference points that a
correct answer should convey. You are strict but fair.

Score 0.0-1.0:
  1.0  conveys every reference point, adds nothing false
  0.7  conveys the main points, minor omission
  0.4  partially correct, or hedged past usefulness
  0.0  wrong, contradicts a reference point, or answers a different question

A refusal or escalation scores 0.0 on coverage but is NOT penalised as false.
Note that separately in "notes".
Reply with JSON only: {"score": 0.0, "notes": "one sentence"}"""


def judge_case(assistant: SupportAssistant, case: dict[str, Any], result: CaseResult) -> None:
    points = case.get("reference_points") or []
    if not points or not result.reply or not assistant.llm.is_live:
        return
    user = (
        "REFERENCE POINTS:\n"
        + "\n".join(f"- {p}" for p in points)
        + f"\n\nQUESTION: {result.question}\n\nANSWER:\n{result.reply}"
    )
    try:
        data = assistant.llm.complete_json(JUDGE_SYSTEM, user, temperature=0.0, max_tokens=300)
    except (LLMError, TypeError):
        return
    try:
        result.judge_score = max(0.0, min(1.0, float(data.get("score", 0))))
    except (TypeError, ValueError):
        result.judge_score = None
    result.judge_notes = str(data.get("notes", ""))[:300]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    scored = [r for r in results if r.hit is not None]
    answered = [r for r in results if r.action in ANSWER_ACTIONS]
    judged = [r for r in results if r.judge_score is not None]

    # Abstention / escalation confusion, computed against the golden labels.
    def _expected(r: CaseResult, name: str) -> bool:
        return bool(r.expected_actions) and r.expected_actions == [name]

    def _prf(name: str) -> dict[str, float]:
        predicted = [r for r in results if r.action == name]
        should = [r for r in results if _expected(r, name)]
        tp = sum(1 for r in predicted if _expected(r, name))
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(should) if should else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "predicted": len(predicted),
            "expected": len(should),
        }

    latencies = sorted(r.latency_ms for r in results if r.latency_ms)

    by_bucket: dict[str, dict[str, Any]] = {}
    for r in results:
        entry = by_bucket.setdefault(r.bucket, {"total": 0, "passed": 0, "failures": []})
        entry["total"] += 1
        entry["passed"] += int(r.passed)
        if not r.passed:
            entry["failures"].append({"id": r.case_id, "why": r.failures})

    return {
        "cases": total,
        "passed": sum(1 for r in results if r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / total, 3) if total else 0.0,
        "retrieval": {
            "scored_cases": len(scored),
            "recall_at_k": round(sum(1 for r in scored if r.hit) / len(scored), 3) if scored else None,
            "mrr": round(
                statistics.mean([r.reciprocal_rank or 0.0 for r in scored]), 3
            ) if scored else None,
        },
        "hallucination": {
            "answered_cases": len(answered),
            "invalid_citation_rate": _rate(answered, lambda r: bool(r.invalid_citations)),
            "retired_claim_leak_rate": _rate(results, lambda r: bool(r.retired_leaks)),
            "unsupported_claim_rate": _rate(answered, lambda r: bool(r.unsupported_claims)),
            "mean_groundedness": round(
                statistics.mean([r.groundedness for r in answered]), 3
            ) if answered else None,
        },
        "abstention": _prf("abstained"),
        "escalation": _prf("escalated"),
        "answer_quality": {
            "judged_cases": len(judged),
            "mean_score": round(statistics.mean([r.judge_score or 0 for r in judged]), 3)
            if judged
            else None,
        },
        "latency_ms": {
            "p50": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[int(len(latencies) * 0.95) - 1] if len(latencies) >= 2 else None,
            "max": latencies[-1] if latencies else None,
        },
        "by_bucket": by_bucket,
    }


def _rate(rows: list[CaseResult], predicate) -> float | None:
    if not rows:
        return None
    return round(sum(1 for r in rows if predicate(r)) / len(rows), 3)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def report(results: list[CaseResult], summary: dict[str, Any], mode: str) -> None:
    table = Table(title="Cases", header_style="bold", box=None)
    for col in ("", "case", "bucket", "action", "docs", "conf", "grnd", "ms"):
        table.add_column(col)
    for r in results:
        table.add_row(
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            r.case_id,
            r.bucket,
            r.action or "-",
            ("yes" if r.hit else "NO") if r.hit is not None else "-",
            f"{r.confidence:.2f}",
            f"{r.groundedness:.2f}",
            str(r.latency_ms),
        )
    console.print(table)

    failures = [r for r in results if not r.passed]
    if failures:
        console.print("\n[bold red]Failures[/bold red]")
        for r in failures:
            console.print(f"  [red]{r.case_id}[/red]: " + "; ".join(r.failures))

    console.print(f"\n[bold]Summary[/bold]  (mode: {mode})")
    console.print_json(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the support assistant")
    parser.add_argument("--bucket", help="run only one bucket")
    parser.add_argument("--case", help="run only one case id")
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge answer quality")
    parser.add_argument("--retrieval-only", action="store_true", help="skip generation")
    parser.add_argument("--report", help="write the JSON report here")
    args = parser.parse_args(argv)

    cases = load_cases(args.bucket, args.case)
    if not cases:
        console.print("[red]no cases matched[/red]")
        return 1

    # Evaluate against a throwaway database so a run never pollutes the
    # conversation history a reviewer is looking at, and so runs are
    # reproducible regardless of what was asked in the CLI beforehand.
    eval_db = REPORTS / "eval.db"
    REPORTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(eval_db) + suffix)
        if candidate.exists():
            candidate.unlink()

    store = Store(settings, db_path=eval_db)
    store.init_schema()
    from app.ingest import run_ingest

    run_ingest(store=store, verbose=False)

    assistant = SupportAssistant(store=store)
    mode = (
        f"generation={'live:' + assistant.llm.model if assistant.llm.is_live else 'offline'}, "
        f"retrieval={'hybrid' if assistant.embedder.is_live else 'lexical-only'}"
    )
    if not assistant.llm.is_live and not args.retrieval_only:
        console.print(
            "[yellow]No GROQ_API_KEY: answer-quality, groundedness and escalation metrics "
            "reflect the offline stub, not a real model. Retrieval metrics are still "
            "meaningful. Use --retrieval-only for a clean keyless run.[/yellow]\n"
        )

    results: list[CaseResult] = []
    with console.status("evaluating…", spinner="dots"):
        for case in cases:
            result = run_case(assistant, case, retrieval_only=args.retrieval_only)
            if args.judge and not args.retrieval_only:
                judge_case(assistant, case, result)
            results.append(result)

    summary = aggregate(results)
    report(results, summary, mode)

    out = Path(args.report) if args.report else REPORTS / f"report-{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "mode": mode,
                "settings": {
                    "llm_model": settings.llm_model,
                    "embedding_model": settings.embedding_model,
                    "retrieval_top_k": settings.retrieval_top_k,
                    "min_dense_score": settings.min_dense_score,
                    "min_lexical_score": settings.min_lexical_score,
                    "min_groundedness": settings.min_groundedness,
                },
                "summary": summary,
                "cases": [r.as_dict() for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(f"\nreport written to {out}")

    assistant.close()
    return 0 if summary["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
