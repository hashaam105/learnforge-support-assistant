"""Interactive terminal client.

    python -m app.cli                      # chat
    python -m app.cli --ask "question"     # one shot
    python -m app.cli --demo               # scripted multi-turn walkthrough

In-chat commands: /why (full diagnostics for the last turn), /sources,
/escalations, /stats, /new, /quit.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from app.config import settings
from app.terminal import make_console
from app.pipeline import SupportAssistant, TurnResult

console = make_console()

ACTION_STYLE = {
    "answered": ("green", "answered"),
    "answered_with_caveat": ("yellow", "answered with caveat"),
    "abstained": ("cyan", "abstained"),
    "escalated": ("magenta", "escalated to a human"),
}

# A scripted conversation that exercises every branch of the gate. Used by
# --demo and mirrored in the README so a reviewer can reproduce it exactly.
DEMO: list[tuple[str, str]] = [
    ("Straightforward, well-grounded answer", "How do I reset my LearnForge password?"),
    ("Stale/withdrawn policy must not be repeated", "Is my annual subscription billed monthly?"),
    ("Authority ordering: policy beats a ticket transcript", "How long do I have to ask for a refund on a course?"),
    ("Out of scope: abstain, do not open a ticket", "What's a good sourdough starter recipe?"),
    ("Sensitive intent: always a human", "There's a $79 charge on my card that I don't recognise."),
    ("Multi-turn coreference", "Cancel my LearnForge"),
    ("  ... follow-up 1", "The payment from last week"),
    ("  ... follow-up 2", "It's the biology one"),
]


def render(result: TurnResult, *, verbose: bool = False) -> None:
    colour, label = ACTION_STYLE.get(result.action, ("white", result.action))

    console.print()
    console.print(Panel(Markdown(result.reply), border_style=colour, title="LearnForge Support",
                        title_align="left"))

    bits = [f"[{colour}]{label}[/{colour}]", f"confidence {result.confidence:.2f}",
            f"{result.latency_ms} ms"]
    if result.citations:
        bits.append("cited " + ", ".join(result.citations))
    if result.escalation_id:
        bits.append(f"case {result.escalation_id} -> {result.queue} ({result.priority})")
    console.print("  " + "  |  ".join(bits), style="dim")

    if result.reason_codes:
        console.print("  why: " + ", ".join(result.reason_codes), style="dim italic")

    if verbose and result.sources:
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        for col in ("doc", "type", "score", "cited", "flags"):
            table.add_column(col)
        for s in result.sources:
            flags = []
            if s["is_stale"]:
                flags.append("stale")
            if s["has_deprecation_notice"]:
                flags.append("withdrawn-text")
            table.add_row(
                s["doc_id"], s["source_type"], f"{s['score']:.3f}",
                "yes" if s["cited"] else "", ",".join(flags),
            )
        console.print(table)


def banner() -> None:
    console.print(Panel.fit(
        "[bold]LearnForge AI Support Assistant[/bold]\n"
        f"{settings.describe()}\n"
        "[dim]/why  /sources  /escalations  /stats  /new  /quit[/dim]",
        border_style="blue",
    ))
    if not settings.has_llm:
        console.print(
            "  [yellow]No GROQ_API_KEY - answers are verbatim excerpts, not composed replies. "
            "Copy .env.example to .env and add a key for the real pipeline.[/yellow]"
        )
    if not settings.has_embeddings:
        console.print("  [yellow]No GEMINI_API_KEY - retrieval is BM25-only (no dense search).[/yellow]")


def _print_diagnostics(result: TurnResult) -> None:
    console.print_json(json.dumps(result.diagnostics, ensure_ascii=False, default=str))


def _print_escalations(assistant: SupportAssistant) -> None:
    rows = assistant.store.list_escalations(10)
    if not rows:
        console.print("  no escalations yet", style="dim")
        return
    for row in rows:
        console.print(
            Panel(
                row["summary"],
                title=f"{row['escalation_id']} | {row['queue']} | {row['priority']}",
                title_align="left",
                border_style="magenta",
            )
        )


def _print_stats(assistant: SupportAssistant) -> None:
    stats: dict[str, Any] = assistant.store.stats()
    table = Table(box=None, show_header=False)
    for key, value in stats.items():
        table.add_row(str(key), json.dumps(value, default=str) if not isinstance(value, int) else str(value))
    console.print(table)


def chat(assistant: SupportAssistant) -> int:
    banner()
    session_id: str | None = None
    last: TurnResult | None = None

    while True:
        try:
            message = console.input("\n[bold cyan]you >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return 0
        if not message:
            continue

        lowered = message.lower()
        if lowered in {"/quit", "/exit", "/q"}:
            return 0
        if lowered == "/new":
            session_id = None
            console.print("  started a new conversation", style="dim")
            continue
        if lowered == "/why":
            _print_diagnostics(last) if last else console.print("  nothing yet", style="dim")
            continue
        if lowered == "/sources":
            if last:
                render(last, verbose=True)
            continue
        if lowered == "/escalations":
            _print_escalations(assistant)
            continue
        if lowered == "/stats":
            _print_stats(assistant)
            continue

        with console.status("thinking...", spinner="dots"):
            last = assistant.ask(message, session_id=session_id)
        session_id = last.session_id
        render(last)


def demo(assistant: SupportAssistant) -> int:
    banner()
    session_id: str | None = None
    for i, (label, message) in enumerate(DEMO):
        # The final three lines are one continuing conversation; everything
        # before that starts fresh so the scenarios do not contaminate.
        if not label.startswith("  "):
            session_id = None
        console.rule(f"[bold]{label}")
        console.print(f"[bold cyan]you >[/bold cyan] {message}")
        result = assistant.ask(message, session_id=session_id)
        session_id = result.session_id
        render(result, verbose=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LearnForge support assistant")
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")
    parser.add_argument("--demo", action="store_true", help="run the scripted walkthrough")
    parser.add_argument("--json", action="store_true", help="emit raw JSON (with --ask)")
    parser.add_argument("--session", help="continue an existing session id")
    args = parser.parse_args(argv)

    assistant = SupportAssistant()
    if assistant.store.stats()["documents"] == 0:
        console.print("[red]The knowledge base is empty. Run:  python -m app.ingest[/red]")
        return 1

    try:
        if args.ask:
            result = assistant.ask(args.ask, session_id=args.session)
            if args.json:
                print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str))
            else:
                render(result, verbose=True)
            return 0
        if args.demo:
            return demo(assistant)
        return chat(assistant)
    finally:
        assistant.close()


if __name__ == "__main__":
    sys.exit(main())
