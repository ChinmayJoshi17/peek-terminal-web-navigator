"""Evaluation harness — WebArena-style mini benchmarks for nano-browser-agent.

Executes each task defined in `benchmark_tasks.json` sequentially using AgentLoop
and reports structured metrics: success, steps, duration, latency, and failure reason.

Run:
    python eval/runner.py
    python eval/runner.py --tasks eval/benchmark_tasks.json --output eval/results.json

Output:
    - Console summary table (Rich)
    - JSON results file (eval/results_<timestamp>.json)
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve project root relative to this package
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from core.config import Settings
from agent.loop import AgentLoop
from agent.providers.groq import GroqClient
from browser.controller import BrowserController

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TASKS_FILE = _PROJECT_ROOT / "eval" / "benchmark_tasks.json"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "eval"
RESULTS_FILENAME_PREFIX = "results"


# ---------------------------------------------------------------------------
# Task loader
# ---------------------------------------------------------------------------

def load_tasks(path: Path) -> List[Dict[str, Any]]:
    """Load benchmark tasks from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks: List[Dict[str, Any]] = []
    for item in raw:
        tasks.append({
            "task_id": str(item["task_id"]),
            "name": str(item["name"]),
            "start_url": str(item["start_url"]),
            "goal": str(item["goal"]),
            "expected_pattern": str(item.get("expected_pattern", "")),
            "max_steps": int(item.get("max_steps", 10)),
        })
    return tasks


# ---------------------------------------------------------------------------
# Task evaluator
# ---------------------------------------------------------------------------

def evaluate_answer(final_answer: str, expected_pattern: str) -> bool:
    """Check if final_answer matches the expected regex pattern (case-insensitive)."""
    if not expected_pattern:
        return False
    try:
        return bool(re.search(expected_pattern, final_answer, re.IGNORECASE))
    except re.error:
        # Fall back to substring match if pattern is invalid
        return expected_pattern.lower() in final_answer.lower()


def _get_latest_jsonl() -> Optional[Path]:
    """Return the most recent agent telemetry JSONL file."""
    logs_dir = _PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        return None
    files = sorted(logs_dir.glob("agent_run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return parsed records."""
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return records


async def run_task(
    task: Dict[str, Any],
    settings: Settings,
    console: Console,
) -> Dict[str, Any]:
    """Execute a single benchmark task and return structured results."""
    task_id = task["task_id"]
    name = task["name"]
    goal = task["goal"]
    start_url = task["start_url"]
    max_steps = task["max_steps"]
    expected = task.get("expected_pattern", "")

    console.print(f"\n[dim]▶ Starting task[/dim] [bold]{task_id}[/bold]: {name}")

    step_start = time.perf_counter()

    # Set up browser controller
    driver = BrowserController(
        headless=settings.browser.headless,
        slow_mo=settings.browser.slow_mo,
        viewport_width=settings.browser.viewport_width,
        viewport_height=settings.browser.viewport_height,
        load_timeout=settings.browser.load_timeout,
    )

    # Set up LLM client
    llm_client = GroqClient(
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        timeout=settings.llm.timeout,
        max_retries=max(settings.llm.max_retries, 5),
        temperature=settings.llm.temperature,
    )

    # Create agent (no rich UI callback during eval)
    agent = AgentLoop(
        driver=driver,
        llm_client=llm_client,
        settings=settings,
        ui_callback=None,
        max_steps=max_steps,
    )

    result_message: str = "(no result)"
    success: bool = False
    failure_reason: Optional[str] = None
    steps_taken: int = 0
    total_tokens: int = 0
    action_latencies: List[float] = []

    try:
        await driver.start()
        result_message = await agent.run(goal=goal, start_url=start_url)
        steps_taken = agent.state.current_step
        total_tokens = agent.state.total_tokens_used

        # Extract action latencies from the most recent telemetry JSONL
        latest_log = _get_latest_jsonl()
        if latest_log:
            records = _read_jsonl(latest_log)
            action_latencies = [
                r.get("latency_ms", 0.0)
                for r in records
                if r.get("success") and r.get("latency_ms", 0) > 0
            ]

        # Evaluate answer against expected pattern
        matched = evaluate_answer(result_message, expected)

        if agent.state.status.value == "completed":
            if matched:
                success = True
            else:
                success = False
                failure_reason = (
                    f"Answer did not match expected pattern '{expected}'. "
                    f"Got: {result_message[:200]}"
                )
        elif agent.state.status.value == "failed":
            success = False
            failure_reason = f"Agent reported failure: {result_message}"
        else:
            success = False
            failure_reason = f"Unexpected status: {agent.state.status.value}"

    except Exception as e:
        success = False
        failure_reason = f"Exception: {e}"
        steps_taken = 0
        total_tokens = 0

    finally:
        await driver.close()
        await llm_client.close()

    elapsed_seconds = time.perf_counter() - step_start
    avg_latency_ms = (
        sum(action_latencies) / len(action_latencies) if action_latencies else 0.0
    )

    task_result: Dict[str, Any] = {
        "task_id": task_id,
        "name": name,
        "start_url": start_url,
        "goal": goal,
        "expected_pattern": expected,
        "max_steps": max_steps,
        "success": success,
        "final_answer": result_message,
        "steps_taken": steps_taken,
        "duration_seconds": round(elapsed_seconds, 2),
        "avg_action_latency_ms": round(avg_latency_ms, 2),
        "total_tokens_used": total_tokens,
        "failure_reason": failure_reason,
        "evaluated_at": datetime.now().isoformat(),
    }

    # Status indicator
    if success:
        console.print(
            f"  [green]✓ PASS[/green] — "
            f"{steps_taken} steps, {elapsed_seconds:.1f}s, "
            f"{avg_latency_ms:.0f}ms avg latency"
        )
    else:
        reason_preview = (failure_reason or "Unknown")[:80]
        console.print(f"  [red]✗ FAIL[/red] — {reason_preview}")

    return task_result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    tasks_file: Path,
    output_dir: Path,
    headless: bool = False,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """Run the full benchmark suite and return results + summary."""
    _console = console or Console()

    _console.print(Rule("[bold]Nano Browser Agent — Evaluation Harness[/bold]"))
    _console.print(f"Tasks file : {tasks_file}")
    _console.print(f"Output dir : {output_dir}")
    _console.print()

    tasks = load_tasks(tasks_file)
    _console.print(f"Loaded [bold]{len(tasks)}[/bold] task(s)\n")

    # Load settings and override headless
    settings = Settings.load()
    settings.browser.headless = headless

    results: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        _console.print(f"[dim][{i}/{len(tasks)}][/dim]", end=" ")
        result = await run_task(task, settings, _console)
        results.append(result)
        if i < len(tasks):
            await asyncio.sleep(2.5)

    # ---------------------------------------------------------------------------
    # Compute summary statistics
    # ---------------------------------------------------------------------------

    successful = [r for r in results if r["success"]]
    success_rate = (len(successful) / len(results) * 100) if results else 0.0

    if successful:
        mean_steps = sum(r["steps_taken"] for r in successful) / len(successful)
        mean_latency = sum(r["avg_action_latency_ms"] for r in successful) / len(successful)
        mean_duration = sum(r["duration_seconds"] for r in successful) / len(successful)
    else:
        mean_steps = 0.0
        mean_latency = 0.0
        mean_duration = 0.0

    summary = {
        "total_tasks": len(results),
        "success_count": len(successful),
        "failure_count": len(results) - len(successful),
        "success_rate_percent": round(success_rate, 1),
        "mean_steps_to_success": round(mean_steps, 2),
        "mean_latency_ms": round(mean_latency, 2),
        "mean_duration_seconds": round(mean_duration, 2),
    }

    # ---------------------------------------------------------------------------
    # Save results to JSON
    # ---------------------------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"{RESULTS_FILENAME_PREFIX}_{timestamp}.json"

    output: Dict[str, Any] = {
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "tasks_file": str(tasks_file),
            "total_tasks": len(tasks),
        },
        "summary": summary,
        "results": results,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------

    _console.print()
    _console.print(Rule("[bold]Results[/bold]"))

    table = Table(
        title="Task Results",
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )
    table.add_column("Task ID", style="dim", width=20)
    table.add_column("Name", width=28)
    table.add_column("Status", width=10)
    table.add_column("Steps", justify="right", width=6)
    table.add_column("Duration (s)", justify="right", width=12)
    table.add_column("Avg Latency (ms)", justify="right", width=16)
    table.add_column("Failure Reason", width=30)

    for r in results:
        status = "[green]PASS[/green]" if r["success"] else "[red]FAIL[/red]"
        failure = r.get("failure_reason") or ""
        failure = failure[:60] + ("…" if len(failure) > 60 else "")
        table.add_row(
            r["task_id"],
            r["name"][:27],
            status,
            str(r["steps_taken"]),
            f'{r["duration_seconds"]:.1f}',
            f'{r["avg_action_latency_ms"]:.1f}',
            failure,
        )

    _console.print(table)

    # ---------------------------------------------------------------------------
    # Print summary panel
    # ---------------------------------------------------------------------------

    summary_text = (
        f"[bold]Success Rate:[/bold] {summary['success_rate_percent']:.1f}%\n"
        f"[bold]Mean Steps (success):[/bold] {summary['mean_steps_to_success']:.2f}\n"
        f"[bold]Mean Latency (success):[/bold] {summary['mean_latency_ms']:.2f} ms\n"
        f"[bold]Mean Duration (success):[/bold] {summary['mean_duration_seconds']:.2f} s\n"
        f"[bold]Results saved to:[/bold] {results_file.name}"
    )

    _console.print()
    _console.print(Panel(summary_text, title="[bold]Benchmark Summary[/bold]", expand=False))

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run nano-browser-agent evaluation benchmarks.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=str(DEFAULT_TASKS_FILE),
        help="Path to benchmark tasks JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode",
    )
    args = parser.parse_args()

    tasks_file = Path(args.tasks)
    output_dir = Path(args.output)

    if not tasks_file.exists():
        print(f"Error: tasks file not found: {tasks_file}")
        sys.exit(1)

    console = Console()
    asyncio.run(run_benchmark(tasks_file, output_dir, headless=args.headless, console=console))


if __name__ == "__main__":
    main()
