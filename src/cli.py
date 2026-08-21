"""cli.py — Rich-powered CLI helpers: banner, spinners, tables, panels."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import box

console = Console()

BANNER = r"""
  _   _           _            ____                 _
 | | | |_   _  __| |_ __ __ _/ ___|_ __ __ _ _ __ | |__
 | |_| | | | |/ _` | '__/ _` | |  _| '__/ _` | '_ \| '_ \
 |  _  | |_| | (_| | | | (_| | |_| | | | (_| | |_) | | | |
 |_| |_|\__, |\__,_|_|  \__,_|\____|_|  \__,_| .__/|_| |_|
        |___/                                 |_|
"""


def print_banner(subtitle: str = "") -> None:
    console.print(Panel(
        f"[bold cyan]{BANNER}[/bold cyan]\n[dim]{subtitle}[/dim]",
        border_style="bright_blue",
        expand=False,
    ))


@contextmanager
def spinner(message: str) -> Generator[None, None, None]:
    """Context manager that shows a spinner while work is done."""
    with console.status(f"[bold yellow]{message}[/bold yellow]", spinner="dots"):
        yield


def ingestion_progress(total: int) -> Progress:
    """Return a configured Rich Progress bar for session ingestion."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def question_progress(total: int) -> Progress:
    """Return a configured Rich Progress bar for question answering."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_answer_panel(
    question: str,
    context_snippet: str,
    answer: str,
    confidence: float,
    abstained: bool,
) -> None:
    """Print a styled panel showing a single Q&A result."""
    status = "🚫 [red]ABSTAINED[/red]" if abstained else (
        "⚠️  [yellow]LOW CONFIDENCE[/yellow]" if confidence < 0.5
        else "✅ [green]ANSWERED[/green]"
    )
    body = (
        f"[bold bright_yellow]❓ {question}[/bold bright_yellow]\n\n"
        f"[bold white]💬 {answer}[/bold white]\n\n"
        f"[dim]Confidence: {confidence:.2f}  |  Status: {status}[/dim]"
    )
    console.print(Panel(body, border_style="cyan", expand=False))


def print_results_table(scores: dict[str, float], title: str = "Results") -> None:
    """Print a color-coded score table for benchmark categories."""
    table = Table(title=title, box=box.ROUNDED, border_style="bright_blue", show_lines=True)
    table.add_column("Category", style="bold white", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("", justify="center")

    for category, score in sorted(scores.items(), key=lambda x: -x[1]):
        pct = score * 100 if score <= 1.0 else score
        if pct >= 70:
            color, icon = "green", "🟢"
        elif pct >= 40:
            color, icon = "yellow", "🟡"
        else:
            color, icon = "red", "🔴"
        table.add_row(category, f"[{color}]{pct:.1f}%[/{color}]", icon)

    console.print(table)


def print_summary(total: int, answered: int, abstained: int, elapsed: float) -> None:
    """Print the final run summary panel."""
    pct = answered / total * 100 if total else 0
    body = (
        f"✅  Answered:  [green]{answered}[/green] / {total}  ({pct:.1f}%)\n"
        f"🚫  Abstained: [red]{abstained}[/red] / {total}\n"
        f"⏱   Elapsed:   [cyan]{elapsed:.1f}s[/cyan]"
    )
    console.print(Panel(body, title="[bold]Run Summary[/bold]", border_style="bright_green"))
