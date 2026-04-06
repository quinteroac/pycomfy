"""``parallax jobs`` subcommand group — list, monitor, and cancel inference jobs."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from typing import Any

import typer

app = typer.Typer(
    name="jobs",
    help="List, monitor, and cancel inference jobs.",
    no_args_is_help=True,
)

# ── Terminal states ───────────────────────────────────────────────────────────

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


def _display_status(raw: str) -> str:
    """Map internal status names to user-facing labels."""
    return "queued" if raw == "pending" else raw


# ── Thin queue wrappers (isolated for testing) ────────────────────────────────


def _call_list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Return the *limit* most recent jobs from the queue."""

    async def _inner() -> list[dict[str, Any]]:
        from server.job_queue import close_queue, get_queue

        queue = await get_queue()
        try:
            return await queue.list_jobs(limit=limit)
        finally:
            await close_queue()

    return asyncio.run(_inner())


def _call_get_job(job_id: str) -> dict[str, Any] | None:
    """Return the job record for *job_id*, or ``None`` if not found."""

    async def _inner() -> dict[str, Any] | None:
        from server.job_queue import close_queue, get_queue

        queue = await get_queue()
        try:
            return await queue.get(job_id)
        finally:
            await close_queue()

    return asyncio.run(_inner())


def _call_cancel_job(job_id: str) -> bool:
    """Cancel *job_id*.  Returns ``True`` on success, ``False`` otherwise."""

    async def _inner() -> bool:
        from server.job_queue import close_queue, get_queue

        queue = await get_queue()
        try:
            return await queue.cancel(job_id)
        finally:
            await close_queue()

    return asyncio.run(_inner())


# ── Commands ──────────────────────────────────────────────────────────────────


@app.command("list")
def list_jobs() -> None:
    """Print a table of the 20 most recent jobs (ID, STATUS, MODEL, CREATED)."""
    from rich.console import Console
    from rich.table import Table

    rows = _call_list_jobs(limit=20)

    table = Table(title="Recent Jobs", show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("STATUS")
    table.add_column("MODEL")
    table.add_column("CREATED")

    for row in rows:
        data_dict: dict[str, Any] = json.loads(row.get("data") or "{}")
        model = data_dict.get("model", "")
        table.add_row(
            row["id"],
            _display_status(row["status"]),
            model,
            row["created_at"],
        )

    Console().print(table)


@app.command("status")
def status(job_id: str = typer.Argument(..., help="Job ID to inspect")) -> None:
    """Print the full job record as formatted JSON."""
    row = _call_get_job(job_id)
    if row is None:
        typer.echo(f"Error: job {job_id} not found.", err=True)
        raise typer.Exit(1)

    record: dict[str, Any] = dict(row)
    record["status"] = _display_status(record["status"])

    # Deserialise embedded JSON fields for readability
    for field in ("result", "progress", "data"):
        if record.get(field):
            try:
                record[field] = json.loads(record[field])
            except (json.JSONDecodeError, TypeError):
                pass

    typer.echo(json.dumps(record, indent=2))


@app.command("watch")
def watch(job_id: str = typer.Argument(..., help="Job ID to watch")) -> None:
    """Render a live progress bar until the job reaches a terminal state."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    status_str = "pending"
    row: dict[str, Any] | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Waiting\u2026", total=100)

        while True:
            row = _call_get_job(job_id)
            if row is None:
                typer.echo(f"Error: job {job_id} not found.", err=True)
                raise typer.Exit(1)

            status_str = row["status"]
            pct = 0.0
            if row.get("progress"):
                try:
                    prog = json.loads(row["progress"])
                    pct = float(prog.get("pct", 0.0))
                except (json.JSONDecodeError, TypeError):
                    pass

            progress.update(
                task_id,
                completed=int(pct * 100),
                description=f"[{_display_status(status_str)}]",
            )

            if status_str in _TERMINAL_STATES:
                break

            time.sleep(1.0)

    if status_str == "completed":
        result: dict[str, Any] = {}
        if row and row.get("result"):
            try:
                result = json.loads(row["result"])
            except (json.JSONDecodeError, TypeError):
                pass
        output_path = result.get("output_path", "")
        typer.echo(output_path)
    elif status_str == "failed":
        result = {}
        if row and row.get("result"):
            try:
                result = json.loads(row["result"])
            except (json.JSONDecodeError, TypeError):
                pass
        error = result.get("error", "Unknown error")
        typer.echo(f"Failed: {error}", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"Job {job_id} was cancelled.")


@app.command("cancel")
def cancel(job_id: str = typer.Argument(..., help="Job ID to cancel")) -> None:
    """Cancel a queued job and print ``Cancelled <job_id>`` on success."""
    row = _call_get_job(job_id)
    if row is None:
        typer.echo(f"Error: job {job_id} not found.", err=True)
        raise typer.Exit(1)

    success = _call_cancel_job(job_id)
    if success:
        typer.echo(f"Cancelled {job_id}")
    else:
        current_status = _display_status(row["status"])
        typer.echo(f"Error: job {job_id} is already {current_status}.", err=True)
        raise typer.Exit(1)


@app.command("open")
def open_job(job_id: str = typer.Argument(..., help="Job ID whose output to open")) -> None:
    """Open the job output file with the OS default application."""
    row = _call_get_job(job_id)
    if row is None:
        typer.echo(f"Error: job {job_id} not found.", err=True)
        raise typer.Exit(1)

    if not row.get("result"):
        typer.echo(f"Error: job {job_id} has no output yet.", err=True)
        raise typer.Exit(1)

    try:
        result: dict[str, Any] = json.loads(row["result"])
    except (json.JSONDecodeError, TypeError):
        result = {}

    output_path = result.get("output_path")
    if not output_path:
        typer.echo(f"Error: job {job_id} has no output path.", err=True)
        raise typer.Exit(1)

    if sys.platform == "darwin":
        opener = "open"
    else:
        opener = "xdg-open"

    subprocess.run([opener, output_path])
