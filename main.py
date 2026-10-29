"""Nano Browser Agent CLI — thin entry point wiring modular components.

This file is intentionally thin: it parses args, builds components,
wires UI callbacks, and delegates all logic to domain modules.

Importers/callers: Direct invocation via `python main.py` or package entry point.
External consumers: None (CLI-only).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Optional

# Suppress Windows Proactor pipe cleanup ResourceWarnings from Playwright
warnings.filterwarnings("ignore", category=ResourceWarning)

from dotenv import load_dotenv
load_dotenv()

from core.config import Settings, get_settings, reset_settings
from core.types import LoopStatus
from browser.controller import BrowserController
from agent.loop import AgentLoop
from agent.providers.groq import GroqClient
from ui.console import ConsoleUI


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with full configuration options."""
    parser = argparse.ArgumentParser(
        description="Nano Browser Agent — autonomous web browsing with LLM guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -p "Search Wikipedia for AI" -u https://wikipedia.org
  python main.py --prompt "Find the latest news" --headless
  python main.py --max-steps 20 --model openai/gpt-oss-120b
        """,
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="The objective/task for the browser agent",
    )
    parser.add_argument(
        "-u", "--url",
        type=str,
        default="https://www.wikipedia.org",
        help="Initial URL to open (default: https://www.wikipedia.org)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum loop iterations before stopping (default: from config or 15)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model identifier (default: from config or openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich terminal UI output",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override GROQ_API_KEY from environment",
    )
    return parser.parse_args()


class UIEventHandler:
    """Handles UI callback events from the agent loop.

    Provides live terminal feedback during execution without coupling
    business logic to presentation.
    """

    def __init__(self, console: ConsoleUI, max_steps: int) -> None:
        self.console = console
        self.max_steps = max_steps
        self._start_time: Optional[float] = None

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        """Handle UI events from AgentLoop."""
        if event == "run_start":
            self._start_time = datetime.now().timestamp()
            self.console.header("Nano Browser Agent", "ReAct Loop Active")
            self.console.task_info(data["goal"], data["start_url"])

        elif event == "observe_start":
            self.console.step_panel(
                data["step"],
                self.max_steps,
                "Scanning page for interactive elements...",
            )

        elif event == "observe_complete":
            url = data.get("url", "")
            count = data.get("element_count", 0)
            if url:
                self.console.step_panel(
                    data["step"],
                    self.max_steps,
                    f"Found {count} elements",
                    url=url,
                    element_count=count,
                )

        elif event == "reason_start":
            self.console.debug_panel("LLM", "Thinking...")

        elif event == "reason_complete":
            action = data.get("action", "")
            duration = data.get("duration_ms", 0)
            tokens = data.get("tokens", 0)
            self.console.action_panel(
                action,
                thought=f"Reasoning complete in {duration:.0f}ms, {tokens} tokens",
            )

        elif event == "act_start":
            action = data.get("action", "")
            element_id = data.get("element_id", "")
            self.console.debug_panel("ACT", f"{action} on {element_id}")

        elif event == "step_complete":
            step = data["step"]
            action = data.get("action", "")
            success = data.get("success", False)
            status_icon = "✓" if success else "✗"
            self.console.debug_panel(
                f"Step {step}",
                f"{status_icon} {action}",
            )

        elif event == "run_complete":
            if data["status"] == "success":
                self.console.result_panel(
                    data.get("final_answer", "Task complete."),
                    success=True,
                )
            elif data["status"] == "failed":
                self.console.result_panel(
                    data.get("error", "Unknown error"),
                    success=False,
                )
            elif data["status"] == "max_steps":
                self.console.warning_panel(
                    f"Max steps reached. URL: {data.get('url', 'unknown')}",
                )
            elif data["status"] == "error":
                self.console.error_panel(
                    data.get("error", "Unknown error"),
                    suggestion="Check logs for details",
                )

        elif event == "observe_error":
            self.console.error_panel(
                f"Observation failed at step {data['step']}: {data['error']}",
            )

        elif event == "reason_error":
            self.console.error_panel(
                f"LLM reasoning failed: {data['error']}",
            )


async def run_agent(
    goal: str,
    start_url: str,
    settings: Settings,
    ui_handler: Optional[UIEventHandler],
    max_steps: Optional[int],
) -> tuple[str, float]:
    """Execute the agent loop with full lifecycle management.

    Returns:
        Tuple of (result_message, duration_seconds)
    """
    start_time = datetime.now()

    driver = BrowserController(
        headless=settings.browser.headless,
        slow_mo=settings.browser.slow_mo,
        viewport_width=settings.browser.viewport_width,
        viewport_height=settings.browser.viewport_height,
        load_timeout=settings.browser.load_timeout,
    )
    await driver.start()

    api_key = settings.llm.api_key
    llm_client = GroqClient(
        api_key=api_key,
        model=settings.llm.model,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
        temperature=settings.llm.temperature,
    )

    effective_max_steps = max_steps or settings.loop.max_steps
    agent = AgentLoop(
        driver=driver,
        llm_client=llm_client,
        settings=settings,
        ui_callback=ui_handler,
        max_steps=effective_max_steps,
    )

    try:
        result = await agent.run(goal=goal, start_url=start_url)
        duration = (datetime.now() - start_time).total_seconds()
        return result, duration
    finally:
        await driver.close()


async def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    settings = Settings.load()

    if args.model:
        settings.llm.model = args.model
    if args.api_key:
        settings.llm.api_key = args.api_key
    if args.headless:
        settings.browser.headless = True
    if args.max_steps:
        settings.loop.max_steps = args.max_steps

    goal = args.prompt
    if not goal:
        console = ConsoleUI()
        console.header("Nano Browser Agent", "Welcome")
        console.console.print(
            "\n[dim]Enter your browsing task (or press Enter for default example):[/dim]"
        )
        goal = input("> ").strip()
        if not goal:
            goal = "Go to wikipedia.org, search for Savitribai Phule, and extract the name of her spouse."

    if not settings.llm.api_key:
        console = ConsoleUI()
        console.error_panel(
            "GROQ_API_KEY not found",
            suggestion="Set GROQ_API_KEY in .env file or pass --api-key",
        )
        sys.exit(1)

    effective_max_steps = args.max_steps or settings.loop.max_steps
    use_rich = not args.no_rich

    ui_handler: Optional[UIEventHandler] = None
    if use_rich:
        console = ConsoleUI()
        ui_handler = UIEventHandler(console, effective_max_steps)

    result, duration = await run_agent(
        goal=goal,
        start_url=args.url,
        settings=settings,
        ui_handler=ui_handler,
        max_steps=args.max_steps,
    )

    if not use_rich:
        print(f"\n{'='*55}")
        print(f"RESULT: {result}")
        print(f"Duration: {duration:.2f}s")
        print(f"{'='*55}")


if __name__ == "__main__":
    asyncio.run(main())