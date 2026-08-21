"""evaluate_beam_gemini.py — Score BEAM answers using Gemini as the LLM judge.

Reads:
  results/beam/<size>/<chat_id>/answers.json  (produced by run_beam.py)
  BEAM/chats/<size>/<chat_id>/probing_questions/probing_questions.json

Writes:
  results/beam/<size>/<chat_id>/evaluation-results-gemini.json

Usage:
  python evaluate_beam_gemini.py --size 100K
  python evaluate_beam_gemini.py --size 100K --start 0 --end 5
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
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.config import gemini, GEMINI_MODEL

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


def _parse_retry_after(msg: str) -> int:
    m = re.search(r"retry in (\d+)", msg, re.IGNORECASE)
    return int(m.group(1)) + 2 if m else 60


def judge_rubric_item(rubric_item: str, llm_response: str) -> int:
    prompt = JUDGE_PROMPT.format(rubric_item=rubric_item, llm_response=llm_response)
    while True:
        try:
            resp = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            text = resp.text.strip()
            # parse JSON
            m = re.search(r'\{"score":\s*([01])\}', text)
            if m:
                return int(m.group(1))
            # fallback
            return 1 if "1" in text else 0
        except ClientError as e:
            if getattr(e, "status_code", None) == 429 or "429" in str(e):
                wait = _parse_retry_after(str(e))
                console.print(f"[yellow]Rate limit — waiting {wait}s…[/yellow]")
                time.sleep(wait)
            else:
                raise


def evaluate_chat(answers_path: Path, questions_path: Path) -> dict:
    with open(answers_path, encoding="utf-8") as f:
        answers = json.load(f)
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    results = {}
    for cat in CATEGORIES:
        if cat not in answers:
            continue
        cat_results = []
        for idx, entry in enumerate(answers[cat]):
            llm_response = entry.get("llm_response", "")
            rubric = questions.get(cat, [{}])[idx].get("rubric", [])
            if not rubric:
                rubric = entry.get("rubric", [])

            scores = [judge_rubric_item(r, llm_response) for r in rubric]
            judge_score = sum(scores) / len(scores) if scores else 0.0
            cat_results.append({
                "question": entry.get("question", ""),
                "llm_response": llm_response,
                "rubric": rubric,
                "rubric_scores": scores,
                "llm_judge_score": judge_score,
                "tau_norm": judge_score,  # used by report_results.py for event_ordering
            })
        results[cat] = cat_results
    return results


def print_scores(all_results: dict[str, dict], size: str) -> None:
    # Aggregate across chats
    cat_scores: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    for chat_id, results in all_results.items():
        for cat, items in results.items():
            for item in items:
                cat_scores[cat].append(item["llm_judge_score"])

    table = Table(title=f"BEAM Results — {size}", box=box.ROUNDED)
    table.add_column("Category", style="cyan", min_width=28)
    table.add_column("Score", justify="right", min_width=8)
    table.add_column("N", justify="right", style="dim")

    cat_means = []
    for cat in CATEGORIES:
        vals = cat_scores[cat]
        if vals:
            m = sum(vals) / len(vals)
            cat_means.append(m)
            table.add_row(cat.replace("_", " ").title(), f"{m*100:.1f}%", str(len(vals)))
        else:
            table.add_row(cat.replace("_", " ").title(), "—", "0")

    if cat_means:
        overall = sum(cat_means) / len(cat_means)
        table.add_row("[bold]Overall[/bold]", f"[bold green]{overall*100:.1f}%[/bold green]", "")

    console.print()
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BEAM answers.json files using Gemini as judge."
    )
    parser.add_argument("--size", default="100K",
                        help="Chat corpus size (100K / 500K / 1M / 10M).")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--results-dir", default=None,
        help="Override the results directory (default: results/beam/<size>). "
             "Use e.g. results/beam-naive/100K to evaluate the Naive RAG baseline.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path("results") / "beam" / args.size
    label = str(results_dir)
    console.print(Panel.fit(f"[bold magenta]BEAM Evaluation — {args.size} (Gemini judge)[/bold magenta]\n[dim]{label}[/dim]"))

    chats_dir = Path("BEAM") / "chats" / args.size

    chat_dirs = sorted(
        [d for d in chats_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )[args.start:args.end]

    all_results: dict[str, dict] = {}

    for chat_dir in chat_dirs:
        chat_id = chat_dir.name
        answers_path = results_dir / chat_id / "answers.json"
        questions_path = chat_dir / "probing_questions" / "probing_questions.json"

        if not answers_path.exists():
            console.print(f"[yellow]Skipping chat {chat_id} — no answers.json[/yellow]")
            continue

        console.print(f"\n[cyan]Evaluating chat {chat_id}…[/cyan]")
        results = evaluate_chat(answers_path, questions_path)
        all_results[chat_id] = results

        # Save per-chat evaluation
        out_path = results_dir / chat_id / "evaluation-results-gemini.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        console.print(f"  [green]✓ Saved → {out_path}[/green]")

    if all_results:
        print_scores(all_results, args.size)


if __name__ == "__main__":
    main()

