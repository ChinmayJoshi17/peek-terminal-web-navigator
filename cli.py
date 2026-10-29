"""PEEK · Terminal Web Navigator CLI.

Modern terminal interface for autonomous web browsing with live step indicators,
interactive shell mode, and clean Unicode result cards.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
import time
import warnings
from typing import Any, Dict, Optional

# Suppress async pipe warnings from Playwright on Windows
warnings.filterwarnings("ignore", category=ResourceWarning)

# Ensure proper Unicode encoding in standard Windows terminals
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from agent.loop import AgentLoop
from agent.providers.groq import GroqClient
from browser.controller import BrowserController
from core.config import Settings
from core.types import LoopStatus

# Suppress verbose Loguru console stdout traces while preserving JSONL logging
from loguru import logger
logger.remove()

BANNER = """╭──────────────────────────────────────────────────────────────╮
│  🔍 PEEK · Terminal Web Navigator  ·  v1.0                    │
╰──────────────────────────────────────────────────────────────╯"""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="peek",
        description="PEEK · Terminal Web Navigator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-g", "--goal",
        type=str,
        default=None,
        help="Target web query or action",
    )
    parser.add_argument(
        "-u", "--start-url",
        type=str,
        default="https://www.google.com",
        help="Starting webpage",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        default=False,
        help="Run with browser window visible",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum step limit for navigation",
    )
    return parser.parse_args()


class MinimalStepReporter:
    """Provides clean, minimal step indicators without raw debug dumps."""

    def __init__(self, max_steps: int, start_url: str) -> None:
        self.max_steps = max_steps
        self.start_url = start_url
        self._last_step = 0
        self._navigated = False

    def __call__(self, event: str, data: Dict[str, Any]) -> None:
        step = data.get("step", self._last_step)
        self._last_step = step

        if event == "observe_start":
            if not self._navigated:
                url_display = (self.start_url[:48] + "...") if len(self.start_url) > 48 else self.start_url
                print(f"  ● [{step}/{self.max_steps}] Navigating to {url_display}...")
                self._navigated = True
        elif event == "observe_complete":
            print(f"  ● [{step}/{self.max_steps}] Locating target elements & reasoning next action...")
        elif event == "act_start":
            print(f"  ● [{step}/{self.max_steps}] Executing action...")


def print_result_card(
    answer: str,
    duration: float,
    steps: int,
    success: bool,
) -> None:
    """Render the final answer inside a clean 64-char Unicode result card."""
    width = 64
    inner_width = width - 4  # 60 chars usable content
    status_str = "PASS" if success else "FAIL"

    top_border = "┌── Result " + "─" * (width - 12) + "┐"
    bottom_border = "└" + "─" * (width - 2) + "┘"

    print("\n" + top_border)
    clean_answer = answer.strip() if answer else "(No answer provided)"
    wrapped = textwrap.wrap(clean_answer, width=inner_width - 2)
    if not wrapped:
        wrapped = ["(No answer provided)"]

    for line in wrapped:
        print(f"│  {line:<{inner_width - 2}}  │")

    print(bottom_border)
    print(f"⏱  {duration:.1f}s  ·  {steps} steps  ·  Status: {status_str}\n")


async def execute_task(
    goal: str,
    start_url: str,
    headless: bool,
    max_steps: int,
) -> None:
    """Execute a single agent browsing session."""
    # Ensure loguru console output remains disabled
    logger.remove()

    settings = Settings.load()
    settings.browser.headless = headless
    settings.loop.max_steps = max_steps

    if not settings.llm.api_key:
        print("\n[Error] GROQ_API_KEY not configured. Please set GROQ_API_KEY in your .env file.\n")
        return

    driver = BrowserController(
        headless=settings.browser.headless,
        slow_mo=300 if not settings.browser.headless else 0,
        viewport_width=settings.browser.viewport_width,
        viewport_height=settings.browser.viewport_height,
        load_timeout=settings.browser.load_timeout,
    )

    llm_client = GroqClient(
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
        temperature=settings.llm.temperature,
    )

    reporter = MinimalStepReporter(max_steps=max_steps, start_url=start_url)
    agent = AgentLoop(
        driver=driver,
        llm_client=llm_client,
        settings=settings,
        ui_callback=reporter,
        max_steps=max_steps,
    )

    start_time = time.perf_counter()
    outcome = ""
    is_success = False
    steps_taken = 0

    try:
        await driver.start()
        outcome = await agent.run(goal=goal, start_url=start_url)
        is_success = agent.state.status == LoopStatus.COMPLETED
        steps_taken = agent.state.current_step
    except asyncio.CancelledError:
        outcome = "Execution cancelled."
        is_success = False
    except Exception as e:
        outcome = f"Agent error: {e}"
        is_success = False
    finally:
        await driver.close()
        await llm_client.close()

    duration = time.perf_counter() - start_time
    print_result_card(
        answer=outcome,
        duration=duration,
        steps=steps_taken,
        success=is_success,
    )


def run_interactive_shell(
    start_url: str,
    headless: bool,
    max_steps: int,
) -> None:
    """Interactive query loop allowing consecutive searches until exit/quit."""
    print("Type your query or 'exit' / 'quit' to terminate.\n")
    while True:
        try:
            query = input("peek > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting PEEK. Goodbye!\n")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("\nExiting PEEK. Goodbye!\n")
            break

        try:
            asyncio.run(
                execute_task(
                    goal=query,
                    start_url=start_url,
                    headless=headless,
                    max_steps=max_steps,
                )
            )
        except KeyboardInterrupt:
            print("\n[Interrupted] Current search aborted.\n")


def main() -> None:
    """CLI entry point with global exception handling."""
    try:
        args = parse_args()
        print(BANNER + "\n")

        headless = not args.headed
        if args.goal and args.goal.strip():
            # One-liner execution mode
            asyncio.run(
                execute_task(
                    goal=args.goal.strip(),
                    start_url=args.start_url,
                    headless=headless,
                    max_steps=args.max_steps,
                )
            )
        else:
            # Interactive shell mode
            run_interactive_shell(
                start_url=args.start_url,
                headless=headless,
                max_steps=args.max_steps,
            )
    except KeyboardInterrupt:
        print("\n\nExiting PEEK. Goodbye!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
