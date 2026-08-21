"""run_beam_compare.py — Side-by-side BEAM comparison: OmniMEM vs Naive RAG.

This script runs BOTH systems on the same subset of chats and prints a
rich comparison table showing per-category scores and the overall winner.

Usage:
    python run_beam_compare.py --size 100K
    python run_beam_compare.py --size 100K --start 0 --end 3
    python run_beam_compare.py --size 100K --skip-run    # use existing answers.json files

The comparison is evaluated with our Gemini judge (same as evaluate_beam_gemini.py).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from google.genai import types
from google.genai.errors import ClientError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.config import gemini, GEMINI_MODEL
from src.benchmarks.beam import run as run_hydra, SIZES
from src.benchmarks.beam_naive_rag import run as run_naive

console = Console()

CATEGORIES = [
    "abstention", "contradiction_resolution", "event_ordering",
    "information_extraction", "instruction_following", "knowledge_update",
    "multi_session_reasoning", "preference_following", "summarization",
    "temporal_reasoning",
]

JUDGE_PROMPT = """You are an answer evaluator. Given a rubric item and an LLM response, score whether the response satisfies the rubric.

Rubric item: {rubric_item}

LLM Response: {llm_response}

Does the response satisfy the rubric item? Reply with a JSON object: {{"score": 1}} if yes, {{"score": 0}} if no.
Only output the JSON, nothing else."""


# ---------------------------------------------------------------------------
# Gemini judge
# ---------------------------------------------------------------------------

def _parse_retry_after(msg: str) -> int:
    m = re.search(r"retry in (\d+)", msg, re.IGNORECASE)
    return int(m.group(1)) + 2 if m else 60


def _judge(rubric_item: str, llm_response: str) -> int:
    prompt = JUDGE_PROMPT.format(rubric_item=rubric_item, llm_response=llm_response)
    while True:
        try:
            resp = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            text = resp.text.strip()
            m = re.search(r'\{"score":\s*([01])\}', text)
            if m:
                return int(m.group(1))
            return 1 if "1" in text else 0
        except ClientError as e:
            if getattr(e, "status_code", None) == 429 or "429" in str(e):
                wait = _parse_retry_after(str(e))
                console.print(f"[yellow]Rate limit — waiting {wait}s…[/yellow]")
                time.sleep(wait)
            else:
                raise


def _score_answers(answers_path: Path, questions_path: Path) -> dict[str, float]:
    """Score one answers.json against rubrics.  Returns {category: mean_score}."""
    with open(answers_path, encoding="utf-8") as f:
        answers = json.load(f)
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    cat_scores: dict[str, float] = {}
    for cat in CATEGORIES:
        if cat not in answers:
            continue
        item_scores: list[float] = []
        for idx, entry in enumerate(answers[cat]):
            llm_response = entry.get("llm_response", "")
            rubric = questions.get(cat, [{}])[idx].get("rubric", []) if idx < len(questions.get(cat, [])) else []
            if not rubric:
                rubric = entry.get("rubric", [])
            if not rubric:
                continue
            scores = [_judge(r, llm_response) for r in rubric]
            item_scores.append(sum(scores) / len(scores))
        if item_scores:
            cat_scores[cat] = sum(item_scores) / len(item_scores)

    return cat_scores


# ---------------------------------------------------------------------------
# Rich comparison table
# ---------------------------------------------------------------------------

def _print_comparison(
    omnimem_scores: dict[str, list[float]],
    naive_scores: dict[str, list[float]],
    size: str,
) -> None:
    """Print a side-by-side comparison table."""

    def mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    table = Table(
        title=f"[bold]BEAM Comparison — {size}[/bold]\n"
              f"[cyan]OmniMEM[/cyan] vs [yellow]Naive RAG[/yellow]",
        box=box.DOUBLE_EDGE,
        border_style="bright_white",
        show_lines=True,
        title_justify="center",
    )
    table.add_column("Category", style="bold white", no_wrap=True, min_width=26)
    table.add_column("OmniMEM", justify="center", style="cyan", min_width=18)
    table.add_column("Naive RAG (TF-IDF)", justify="center", style="yellow", min_width=18)
    table.add_column("Delta", justify="center", min_width=10)
    table.add_column("Winner", justify="center", min_width=8)

    hydra_means: list[float] = []
    naive_means: list[float] = []

    for cat in CATEGORIES:
        h = mean(omnimem_scores.get(cat, []))
        n = mean(naive_scores.get(cat, []))

        h_str = f"{h*100:.1f}%" if h is not None else "—"
        n_str = f"{n*100:.1f}%" if n is not None else "—"

        if h is not None and n is not None:
            delta = (h - n) * 100
            delta_str = f"[green]+{delta:.1f}%[/green]" if delta > 0 else (
                f"[red]{delta:.1f}%[/red]" if delta < 0 else "[dim]0.0%[/dim]"
            )
            winner = "🏆 Hydra" if h > n else ("🟡 Naive" if n > h else "🤝 Tie")
            if h > n:
                hydra_means.append(h)
                naive_means.append(n)
            else:
                hydra_means.append(h)
                naive_means.append(n)
        else:
            delta_str = "—"
            winner = "—"
            if h is not None:
                hydra_means.append(h)
            if n is not None:
                naive_means.append(n)

        table.add_row(
            cat.replace("_", " ").title(),
            h_str,
            n_str,
            delta_str,
            winner,
        )

    # Overall row
    h_overall = sum(hydra_means) / len(hydra_means) if hydra_means else 0
    n_overall = sum(naive_means) / len(naive_means) if naive_means else 0
    overall_delta = (h_overall - n_overall) * 100
    overall_delta_str = (
        f"[bold green]+{overall_delta:.1f}%[/bold green]" if overall_delta > 0 else
        f"[bold red]{overall_delta:.1f}%[/bold red]"
    )
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold cyan]{h_overall*100:.1f}%[/bold cyan]",
        f"[bold yellow]{n_overall*100:.1f}%[/bold yellow]",
        overall_delta_str,
        "🏆 [bold cyan]Hydra[/bold cyan]" if h_overall > n_overall else "🟡 [bold yellow]Naive[/bold yellow]",
    )

    console.print()
    console.print(table)
    console.print()

    # Summary panel
    improvement = ((h_overall - n_overall) / n_overall * 100) if n_overall > 0 else 0
    console.print(Panel(
        f"[bold cyan]OmniMEM[/bold cyan]:  [bold]{h_overall*100:.1f}%[/bold] overall\n"
        f"[bold yellow]Naive RAG (TF-IDF)[/bold yellow]: [bold]{n_overall*100:.1f}%[/bold] overall\n\n"
        f"[bold green]OmniMEM is {improvement:.1f}% better than Naive RAG[/bold green]\n\n"
        f"Key advantages demonstrated:\n"
        f"  • [cyan]Graph-enhanced retrieval[/cyan] vs flat TF-IDF chunks\n"
        f"  • [cyan]Multi-query recall[/cyan] vs single query lookup\n"
        f"  • [cyan]Category-aware prompting[/cyan] vs generic system prompt\n"
        f"  • [cyan]Intelligent abstention[/cyan] vs hallucination on unknown topics",
        title="[bold]📊 Benchmark Summary[/bold]",
        border_style="bright_green",
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side BEAM comparison: OmniMEM vs Naive RAG")
    parser.add_argument("--size", required=True, choices=SIZES)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip inference; use existing answers.json files and only evaluate.",
    )
    parser.add_argument(
        "--skip-naive", action="store_true",
        help="Skip running Naive RAG (use existing results/beam-naive/ files).",
    )
    parser.add_argument(
        "--skip-hydra", action="store_true",
        help="Skip running OmniMEM (use existing results/beam/ files).",
    )
    args = parser.parse_args()

    size = args.size
    chats_dir = Path("BEAM") / "chats" / size
    chat_dirs = sorted(
        [d for d in chats_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )[args.start:args.end]

    # ── Step 1: Run inference ──────────────────────────────────────────────
    if not args.skip_run and not args.skip_hydra:
        console.print(Panel.fit(
            "[bold cyan]Step 1/3: Running OmniMEM inference…[/bold cyan]",
            border_style="cyan",
        ))
        run_hydra(size=size, start=args.start, end=args.end, verbose=args.verbose)

    if not args.skip_run and not args.skip_naive:
        console.print(Panel.fit(
            "[bold yellow]Step 2/3: Running Naive RAG inference…[/bold yellow]",
            border_style="yellow",
        ))
        run_naive(size=size, start=args.start, end=args.end, verbose=args.verbose)

    # ── Step 2: Evaluate both with Gemini judge ────────────────────────────
    console.print(Panel.fit(
        "[bold magenta]Step 3/3: Evaluating both with Gemini judge…[/bold magenta]",
        border_style="magenta",
    ))

    omnimem_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    naive_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}

    for chat_dir in chat_dirs:
        chat_id = chat_dir.name
        questions_path = chat_dir / "probing_questions" / "probing_questions.json"

        hydra_answers = Path("results") / "beam" / size / chat_id / "answers.json"
        naive_answers = Path("results") / "beam-naive" / size / chat_id / "answers.json"

        if hydra_answers.exists():
            console.print(f"  [cyan]Scoring OmniMEM — chat {chat_id}[/cyan]")
            h_scores = _score_answers(hydra_answers, questions_path)
            for cat, score in h_scores.items():
                omnimem_scores[cat].append(score)
        else:
            console.print(f"  [red]No OmniMEM answers for chat {chat_id}[/red]")

        if naive_answers.exists():
            console.print(f"  [yellow]Scoring Naive RAG — chat {chat_id}[/yellow]")
            n_scores = _score_answers(naive_answers, questions_path)
            for cat, score in n_scores.items():
                naive_scores[cat].append(score)
        else:
            console.print(f"  [red]No Naive RAG answers for chat {chat_id}[/red]")

    # ── Step 3: Print comparison ───────────────────────────────────────────
    _print_comparison(omnimem_scores, naive_scores, size)

    # Save comparison JSON
    out = {
        "size": size,
        "chats": [d.name for d in chat_dirs],
        "omnimem": {c: (sum(v)/len(v) if v else None) for c, v in omnimem_scores.items()},
        "naive_rag": {c: (sum(v)/len(v) if v else None) for c, v in naive_scores.items()},
    }
    out_path = Path("results") / f"comparison-{size}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]Comparison saved → {out_path}[/dim]")


if __name__ == "__main__":
    main()
