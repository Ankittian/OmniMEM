"""score_report.py — Unified score dashboard for BEAM.

Usage (after running BEAM benchmark and evaluation):
    python score_report.py

Reads:
  - results/beam/100K/<chat_id>/evaluation-results-*.json  (BEAM, LLM-judged)
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

BEAM_CATEGORIES = [
    "abstention", "contradiction_resolution", "event_ordering",
    "information_extraction", "instruction_following", "knowledge_update",
    "multi_session_reasoning", "preference_following", "summarization",
    "temporal_reasoning",
]


def load_beam_scores(results_dir="results/beam"):
    pattern = os.path.join(results_dir, "**", "evaluation-results*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    category_scores = {c: [] for c in BEAM_CATEGORIES}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for cat, items in data.items():
            if cat not in category_scores:
                continue
            for item in items:
                if cat == "event_ordering":
                    score = float(item.get("tau_norm", item.get("llm_judge_score", 0)))
                else:
                    score = float(item.get("llm_judge_score", 0))
                category_scores[cat].append(score)
    return category_scores


def mean(lst):
    return float(np.mean(lst)) if lst else float("nan")

def pct(v):
    return f"{v * 100:.1f}%" if not np.isnan(v) else "—"


def print_beam_table(scores):
    if scores is None:
        console.print("[yellow]BEAM evaluation results not found.[/yellow]")
        console.print("  Run: python BEAM/src/evaluation/run_evaluation.py ...")
        return None
    table = Table(title="BEAM Scores (100K)", box=box.ROUNDED, show_footer=True)
    table.add_column("Category", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("N", justify="right", style="dim")
    cat_means = []
    for cat in BEAM_CATEGORIES:
        vals = scores[cat]
        m = mean(vals)
        cat_means.append(m)
        table.add_row(cat.replace("_", " ").title(), pct(m), str(len(vals)))
    overall = float(np.nanmean(cat_means))
    table.add_row("[bold]Overall[/bold]", f"[bold]{pct(overall)}[/bold]", "")
    console.print(table)
    return overall


def main():
    console.print(Panel.fit("[bold magenta]HydraGraph Memory — Benchmark Score Dashboard[/bold magenta]"))
    beam_scores = load_beam_scores()
    console.print()
    beam_overall = print_beam_table(beam_scores)
    console.print()
    summary = Table(title="Summary", box=box.DOUBLE_EDGE, show_header=False)
    summary.add_column("Benchmark", style="bold")
    summary.add_column("Score", justify="right", style="bold green")
    summary.add_row("BEAM (100K)", pct(beam_overall) if beam_overall is not None else "not evaluated")
    console.print(summary)

if __name__ == "__main__":
    main()
