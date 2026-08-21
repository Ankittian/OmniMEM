"""benchmarks/beam_naive_rag.py — BEAM benchmark adapter for Naive RAG baseline.

Mirrors beam.py but uses the naive TF-IDF retriever instead of HydraDB.

Input:  BEAM/chats/<size>/<chat_id>/
          chat.json
          probing_questions/probing_questions.json

Output: results/beam-naive/<size>/<chat_id>/answers.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.naive_rag import build_index, recall, generate_answer
from src.cli import (
    console,
    print_banner,
    question_progress,
    print_answer_panel,
)
from rich.panel import Panel

RESULTS_DIR = Path("results") / "beam-naive"

SIZES = ("100K", "500K", "1M", "10M")


def _flatten_chat(chat: list[dict]) -> tuple[list[list[dict]], list[str]]:
    """Convert BEAM chat.json batches into (sessions, timestamps)."""
    sessions: list[list[dict]] = []
    timestamps: list[str] = []

    for batch in chat:
        batch_turns: list[dict] = []
        batch_ts = ""
        for turn_pair in batch.get("turns", []):
            for turn in turn_pair:
                if turn.get("role") == "user" and turn.get("time_anchor") and not batch_ts:
                    batch_ts = turn["time_anchor"]
                batch_turns.append({"role": turn["role"], "content": turn["content"]})
        sessions.append(batch_turns)
        timestamps.append(batch_ts)

    return sessions, timestamps


def run_single(chat_dir: Path, size: str, verbose: bool = False) -> Path:
    """Run Naive RAG BEAM evaluation for a single chat directory."""
    chat_id = chat_dir.name

    chat_file = chat_dir / "chat_trunecated.json"
    if not chat_file.exists():
        chat_file = chat_dir / "chat.json"

    pq_file = chat_dir / "probing_questions" / "probing_questions.json"

    with open(chat_file, encoding="utf-8") as f:
        raw_chat: list[dict] = json.load(f)
    with open(pq_file, encoding="utf-8") as f:
        probing: dict[str, list[dict]] = json.load(f)

    sessions, timestamps = _flatten_chat(raw_chat)

    console.rule(f"[bold]Chat {chat_id}[/bold]  ({len(sessions)} batches) [dim]— Naive RAG[/dim]")

    # --- Build in-memory index (no persistent storage) ---
    console.print(f"[dim]Building TF-IDF index for {len(sessions)} sessions…[/dim]")
    retriever = build_index(sessions, timestamps)
    console.print(f"[dim]  ✓ Index built ({retriever._chunks.__len__()} chunks)[/dim]")

    # --- Answer all probing questions ---
    total_q = sum(len(v) for v in probing.values())
    output: dict[str, list[dict]] = {}

    with question_progress(total_q) as progress:
        task = progress.add_task("[Naive RAG] Answering probing questions", total=total_q)
        for category, questions in probing.items():
            output[category] = []
            for q in questions:
                question_text: str = q["question"]

                context, confidence = recall(retriever, question_text)
                llm_response = generate_answer(context, question_text)

                if verbose:
                    print_answer_panel(question_text, context, llm_response, confidence, False, category=category)

                entry = dict(q)
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
    """Run Naive RAG BEAM evaluation across all chat dirs for a given size."""
    print_banner(f"Naive RAG Baseline  [{size}]")
    chats_dir = Path("BEAM") / "chats" / size
    chat_dirs = sorted(
        [d for d in chats_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )[start:end]

    t_start = time.time()
    total_chats = len(chat_dirs)

    for i, chat_dir in enumerate(chat_dirs, 1):
        console.print(f"\n[bold yellow]── Chat {i}/{total_chats} ──[/bold yellow]")
        out_path = run_single(chat_dir, size, verbose=verbose)
        console.print(f"[green]  ✓ Saved →[/green] {out_path}")

    elapsed = time.time() - t_start
    console.print(Panel(
        f"[bold green]Naive RAG run complete[/bold green]\n"
        f"Processed {total_chats} chats in {elapsed:.1f}s\n"
        f"Results in: results/beam-naive/{size}/",
        border_style="bright_yellow",
    ))
