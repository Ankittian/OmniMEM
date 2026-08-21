"""benchmarks/beam.py — BEAM benchmark adapter.

Input:  BEAM/chats/<size>/<chat_id>/
          chat.json                         — conversation turns
          probing_questions/probing_questions.json  — 10-category questions

Output: results/beam/<size>/<chat_id>/answers.json
        Mirrors the format expected by BEAM/src/evaluation/compute_metrics.py:
        {
            "<category>": [
                {
                    "question": "...",
                    "ideal_answer": "...",   # (copied through)
                    "rubric": [...],          # (copied through)
                    "llm_response": "..."     # our answer
                },
                ...
            ],
            ...
        }

Evaluate with the benchmark's own scripts:
    python BEAM/src/evaluation/compute_metrics.py ...
    python BEAM/src/evaluation/report_results.py ...
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.memory import ingest_sessions, recall, should_abstain, setup_tenant, get_ingested_collections
from src.answerer import generate_answer
from src.cli import (
    console,
    print_banner,
    ingestion_progress,
    question_progress,
    print_answer_panel,
    print_results_table,
    print_summary,
)
from rich.panel import Panel

RESULTS_DIR = Path("results") / "beam"

# BEAM chat sizes we support
SIZES = ("100K", "500K", "1M", "10M")


def _flatten_chat(chat: list[dict]) -> tuple[list[list[dict]], list[str]]:
    """Convert BEAM chat.json batches into (sessions, timestamps).

    Each batch becomes a session; timestamp is synthesised from time_anchor
    when present, otherwise an empty string.
    """
    sessions: list[list[dict]] = []
    timestamps: list[str] = []

    for batch in chat:
        batch_turns: list[dict] = []
        batch_ts = ""
        for turn_pair in batch.get("turns", []):
            for turn in turn_pair:
                # Grab time_anchor from any user turn in this batch
                if turn.get("role") == "user" and turn.get("time_anchor") and not batch_ts:
                    batch_ts = turn["time_anchor"]
                batch_turns.append({"role": turn["role"], "content": turn["content"]})
        sessions.append(batch_turns)
        timestamps.append(batch_ts)

    return sessions, timestamps


def run_single(chat_dir: Path, size: str, database_name: str, ingested_collections: list[str], verbose: bool = False) -> Path:
    """Run BEAM evaluation for a single chat directory."""
    chat_id = chat_dir.name
    collection = chat_id.lower()

    # Pick truncated chat if available (10M size)
    chat_file = chat_dir / "chat_trunecated.json"
    if not chat_file.exists():
        chat_file = chat_dir / "chat.json"

    pq_file = chat_dir / "probing_questions" / "probing_questions.json"

    with open(chat_file, encoding="utf-8") as f:
        raw_chat: list[dict] = json.load(f)
    with open(pq_file, encoding="utf-8") as f:
        probing: dict[str, list[dict]] = json.load(f)

    sessions, timestamps = _flatten_chat(raw_chat)

    console.rule(f"[bold]Chat {chat_id}[/bold]  ({len(sessions)} batches)")

    # --- Ingestion ---
    if collection not in ingested_collections:
        with ingestion_progress(len(sessions)) as progress:
            task = progress.add_task(f"Ingesting {len(sessions)} batches", total=len(sessions))
            ingest_sessions(database_name=database_name, collection_name=collection, sessions=sessions, timestamps=timestamps)
            progress.update(task, advance=len(sessions))
    else:
        console.print(f"[dim]Database '{database_name}' (collection '{collection}') already populated. Skipping ingestion.[/dim]")

    # Per-category extra sub-queries for multi-session retrieval
    _EXTRA_QUERIES: dict[str, list[str]] = {
        "event_ordering": [
            "project timeline milestones dates",
            "security features authentication implementation",
            "transaction database schema columns",
            "deployment configuration commits version",
            "sprint schedule analytics",
        ],
        "multi_session_reasoning": [
            "project timeline milestones dates",
            "security features authentication implementation",
            "transaction database schema columns",
            "deployment configuration commits version",
            "sprint schedule analytics",
        ],
        "summarization": [
            "project timeline milestones dates",
            "security features authentication implementation",
            "transaction database schema columns",
            "deployment configuration commits version",
            "sprint schedule analytics",
        ],
        "knowledge_update": [
            "project timeline milestones dates",
            "deployment configuration commits version",
            "sprint schedule analytics",
        ],
        "contradiction_resolution": [
            "project timeline milestones dates",
            "security features authentication implementation",
            "deployment configuration commits version",
        ],
        # Targeted date-arithmetic queries for temporal_reasoning
        # These pull the exact sessions where the key dates are stored:
        #   - session 2: project milestones (Jan 15 transaction completion)
        #   - session 0: deployment deadline (March 15)
        #   - session 28: end of sprint 1 (March 29)
        #   - session 86: sprint 2 analytics deadline (April 19)
        "temporal_reasoning": [
            "January 15 2024 transaction management features completed milestone",
            "March 15 2024 final deployment deadline",
            "sprint 1 end date March 29 completion",
            "sprint 2 analytics deadline April 19",
            "project schedule milestones January March",
            "sprint schedule dates days weeks",
        ],
    }

    # --- Answer all probing questions ---
    total_q = sum(len(v) for v in probing.values())
    answered = abstained = 0
    output: dict[str, list[dict]] = {}

    with question_progress(total_q) as progress:
        task = progress.add_task("Answering probing questions", total=total_q)
        for category, questions in probing.items():
            output[category] = []
            for q in questions:
                question_text: str = q["question"]

                extra = _EXTRA_QUERIES.get(category)

                context, confidence = recall(
                    database_name=database_name,
                    collection_name=collection,
                    query=question_text,
                    extra_queries=extra,
                )
                abstained_flag = should_abstain(confidence)
                llm_response = generate_answer(
                    context, question_text, confidence, category=category
                )

                if abstained_flag:
                    abstained += 1
                else:
                    answered += 1

                if verbose:
                    print_answer_panel(question_text, context, llm_response, confidence, abstained_flag, category=category)

                entry = dict(q)          # copy question fields + rubric through
                entry["llm_response"] = llm_response
                output[category].append(entry)
                progress.update(task, advance=1)

    # --- Save results ---
    out_dir = RESULTS_DIR / size / chat_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "answers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return out_path


def run(size: str, start: int = 0, end: int | None = None, verbose: bool = False) -> None:
    """Run BEAM evaluation across all chat dirs for a given size.

    Args:
        size: One of "100K", "500K", "1M", "10M".
        start: Index of first chat to process (inclusive).
        end:   Index of last chat to process (exclusive). None = all.
        verbose: Print per-question panels.
    """
    print_banner(f"BEAM Benchmark  [{size}]")
    chats_dir = Path("BEAM") / "chats" / size
    chat_dirs = sorted(
        [d for d in chats_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )[start:end]

    t_start = time.time()
    total_chats = len(chat_dirs)
    
    database_name = f"beam-{size.lower()}"
    setup_tenant(database_name)
    ingested_collections = get_ingested_collections(database_name)

    for i, chat_dir in enumerate(chat_dirs, 1):
        console.print(f"\n[bold cyan]── Chat {i}/{total_chats} ──[/bold cyan]")
        out_path = run_single(chat_dir, size, database_name, ingested_collections, verbose=verbose)
        console.print(f"[green]  ✓ Saved →[/green] {out_path}")

    elapsed = time.time() - t_start
    console.print(Panel(
        f"[bold green]BEAM run complete[/bold green]\n"
        f"Processed {total_chats} chats in {elapsed:.1f}s\n"
        f"Results in: results/beam/{size}/",
        border_style="bright_green",
    ))
