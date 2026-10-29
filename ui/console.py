"""Rich terminal presentation layer — completely decoupled from business logic.

Per ECC specs: ui/console.py provides status spinners, step panels, execution
summary tables, and logging output. No business logic here — just presentation.

Importers/callers: main.py imports from ui.console for rendering loops.
Affected API: None (pure presentation layer).
Data schemas: None (consumes types from core.types but does not define schemas).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text
from rich.style import Style


# Global console instance
_console: Optional[Console] = None


def get_console() -> Console:
    """Get or create the global Rich console instance."""
    global _console
    if _console is None:
        _console = Console(
            width=100,
            color_system="auto",
            legacy_windows=False,
        )
    return _console


class ConsoleUI:
    """Rich terminal presentation layer for nano-browser-agent.

    Features:
    - Status spinners for async operations
    - Step panels showing progress
    - Execution summary tables
    - Error/warning displays

    Usage:
        ui = ConsoleUI()
        ui.header("Task: Search Wikipedia")
        ui.step_panel(1, 15, "Scanning page for elements...", url="https://example.com")
        ui.progress_bar(5, 10, tokens=1500, duration_ms=120.5)
        ui.summary_table(results)
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or get_console()
        self._start_time: float = 0.0
        self._steps: List[Dict[str, Any]] = []

    def header(self, title: str, subtitle: str = "") -> None:
        """Display a header panel with the task title."""
        content = Text()
        content.append(title.strip(), style=Style(bold=True, color="cyan"))
        if subtitle:
            content.append(f"\n{subtitle}", style=Style(color="gray"))

        self.console.print(Panel(content, expand=False, border_style="cyan"))

    def task_info(self, goal: str, start_url: str) -> None:
        """Display task information panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(Style="cyan", width=10)
        table.add_column(radio=None)

        table.add_row("[bold]Goal:[/bold]", goal)
        table.add_row("[bold]Start URL:[/bold]", start_url)
        table.add_row("[bold]Started:[/bold]", datetime.now().strftime("%H:%M:%S"))

        self.console.print(table)

    def step_panel(
        self,
        step: int,
        max_steps: int,
        status: str,
        url: str = "",
        element_count: int = 0,
    ) -> None:
        """Display a step progress panel.

        Args:
            step: Current step number (1-indexed)
            max_steps: Maximum steps allowed
            status: Current status message
            url: Current URL (optional)
            element_count: Number of elements found (optional)
        """
        progress = f"Step {step}/{max_steps}"

        lines = [
            f"[bold]{progress}[/bold]",
            f"[dim]{status}[/dim]",
        ]

        if url:
            lines.append(f"[dim]URL:[/dim] {url}")
        if element_count > 0:
            lines.append(f"[dim]Elements:[/dim] {element_count}")

        panel = Panel(
            "\n".join(lines),
            title=f"Step {step}",
            border_style="blue",
            expand=False,
        )
        self.console.print(panel)

    def action_panel(
        self,
        action: str,
        element_id: str = "",
        text: str = "",
        thought: str = "",
    ) -> None:
        """Display an action being taken.

        Args:
            action: The action type (click, type, navigate, done)
            element_id: Target element ID (optional)
            text: Text being typed (optional)
            thought: LLM reasoning (optional)
        """
        color = {
            "click": "green",
            "type": "yellow",
            "navigate": "cyan",
            "done": "magenta",
            "scroll": "blue",
            "wait": "blue",
        }.get(action, "white")

        lines = [f"[bold {color}]Action: {action.upper()}[/{color}]"]

        if element_id:
            lines.append(f"  Element: {element_id}")
        if text:
            lines.append(f"  Text: {repr(text)}")
        if thought:
            lines.append(f"  Reasoning: {thought[:80]}{'...' if len(thought) > 80 else ''}")

        self.console.print(Panel("\n".join(lines), border_style=color, expand=False))

    def spinner_panel(
        self,
        task: str,
        description: str = "",
        spinner: str = "dots",
    ) -> Progress:
        """Create a progress spinner for async operations.

        Returns the Progress object for later completion:
            progress = ui.spinner_panel("Loading", "Waiting for browser...")
            # ... async work ...
            progress.stop()
        """
        progress = Progress(
            SpinnerColumn(spinner_name=spinner),
            TextColumn("[description]"),
            console=self.console,
            transient=True,
        )
        progress.start()
        if description:
            self.console.print(f"[dim]{description}[/dim]")
        return progress

    def progress_bar(
        self,
        current: int,
        total: int,
        label: str = "Progress",
        tokens: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Display a progress bar for step completion.

        Args:
            current: Current step number
            total: Total steps
            label: Progress label
            tokens: Tokens used (optional)
            duration_ms: Duration in milliseconds (optional)
        """
        percent = (current / total * 100) if total > 0 else 0

        table = Table(grid=True, padding=(0, 2))
        table.add_column(expand=True)
        table.add_column(width=10)

        bar_fill = int(percent / 2)  # 50 char bar
        bar = "[" + "█" * bar_fill + "░" * (50 - bar_fill) + "]"

        self.console.print(f"[cyan]{label}[/cyan] {bar} {percent:.0f}%")

        if tokens is not None or duration_ms is not None:
            details = []
            if tokens is not None:
                details.append(f"tokens={tokens}")
            if duration_ms is not None:
                details.append(f"duration={duration_ms:.1f}ms")
            self.console.print(f"  Step {current}/{total} | {' | '.join(details)}")

    def element_list_panel(
        self,
        elements: List[Dict[str, Any]],
        max_display: int = 10,
    ) -> None:
        """Display a list of interactive elements found on the page.

        Args:
            elements: List of DOM elements from browser scan
            max_display: Maximum number to display
        """
        if not elements:
            self.console.print("[dim]No interactive elements found on this page.[/dim]")
            return

        table = Table(title=f"Interactive Elements (showing {min(len(elements), max_display)}/{len(elements)})", show_header=True)
        table.add_column("ID", style="cyan", max_width=5)
        table.add_column("Tag", style="green", max_width=10)
        table.add_column("Text", max_width=30)

        for el in elements[:max_display]:
            text = el.get("text", "")[:30]
            tag = el.get("tag", "unknown")
            el_id = el.get("id", "?")
            table.add_row(el_id, tag, text)

        if len(elements) > max_display:
            self.console.print(table)
            self.console.print(f"[dim]... and {len(elements) - max_display} more elements[/dim]")
        else:
            self.console.print(table)

    def result_panel(
        self,
        result: str,
        success: bool = True,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Display the final result panel.

        Args:
            result: The final result text
            success: Whether the task succeeded
            duration_ms: Total duration in milliseconds
        """
        border_style = "green" if success else "red"
        title = "[bold green]✓ SUCCESS[/bold green]" if success else "[bold red]✗ FAILED[/bold red]"

        table = Table.grid(padding=(1, 2))
        table.add_column()

        table.add_row(f"[bold]Result:[/bold]\n{result}")

        if duration_ms is not None:
            table.add_row(f"[dim]Duration: {duration_ms:.2f}s[/dim]")

        self.console.print(Panel(table, title=title, border_style=border_style))

    def error_panel(
        self,
        error: str,
        suggestion: str = "",
    ) -> None:
        """Display an error panel.

        Args:
            error: Error message
            suggestion: Suggested recovery action
        """
        lines = [f"[bold red]Error:[/bold red] {error}"]
        if suggestion:
            lines.append(f"[dim]Suggestion: {suggestion}[/dim]")

        self.console.print(Panel("\n".join(lines), border_style="red"))

    def warning_panel(self, message: str) -> None:
        """Display a warning panel.

        Args:
            message: Warning message
        """
        self.console.print(Panel(message, border_style="yellow", title="[bold yellow]Warning[/bold yellow]"))

    def summary_table(
        self,
        steps: List[Dict[str, Any]],
        total_duration_ms: float,
        success: bool,
    ) -> None:
        """Display an execution summary table.

        Args:
            steps: List of step result records
            total_duration_ms: Total execution time in milliseconds
            success: Whether the task succeeded
        """
        table = Table(title="Execution Summary", show_header=True)
        table.add_column("#", style="cyan", max_width=4)
        table.add_column("Action", style="green", max_width=10)
        table.add_column("Element/URL", max_width=25)
        table.add_column("Status", max_width=8)
        table.add_column("Time (ms)", justify="right", max_width=10)

        for step_data in steps:
            step_num = step_data.get("step_number", "?")
            action = step_data.get("action", "unknown")
            element = step_data.get("element_id", "")
            if not element and step_data.get("url"):
                element = step_data["url"][:20]
            status = "[green]✓[/green]" if step_data.get("success") else "[red]✗[/red]"
            duration = f"{step_data.get('duration_ms', 0):.1f}"

            table.add_row(str(step_num), action, element, status, duration)

        # Footer row
        success_icon = "✓" if success else "✗"
        footer_style = "green" if success else "red"

        self.console.print(table)
        self.console.print(f"\n[bold {footer_style}]{success_icon} Completed in {total_duration_ms/1000:.2f}s[/bold {footer_style}]")

    def debug_panel(
        self,
        key: str,
        value: str,
        style: str = "dim",
    ) -> None:
        """Display a debug/value panel.

        Args:
            key: Key name
            value: Value to display
            style: Rich style for the value
        """
        self.console.print(f"[{style}]{key:20s}[/{style}] {value}")

    def json_output(
        self,
        data: Dict[str, Any],
        title: str = "Output",
    ) -> None:
        """Display JSON data in a formatted output panel.

        Args:
            data: Dictionary to display as JSON
            title: Panel title
        """
        import json
        json_str = json.dumps(data, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        self.console.print(Panel(syntax, title=title, border_style="cyan"))

    def live_text(self, text: str) -> None:
        """Print live updating text (for streaming output).

        Args:
            text: Text to print
        """
        self.console.print(text)


# Convenience functions for quick usage
_header_printed = False


def header(title: str, subtitle: str = "") -> None:
    """Quick header function."""
    global _header_printed
    console = get_console()
    if not _header_printed:
        content = Text()
        content.append(title.strip(), style=Style(bold=True, color="cyan"))
        if subtitle:
            content.append(f"\n{subtitle}", style=Style(color="gray"))
        console.print(Panel(content, expand=False, border_style="cyan"))
        _header_printed = True


def step(step_num: int, max_steps: int, status: str, url: str = "") -> None:
    """Quick step indicator function."""
    ui = ConsoleUI()
    ui.step_panel(step_num, max_steps, status, url)