"""Browser package — isolated Playwright controller and DOM observation utilities.

Importers/callers: main.py, agent/loop.py, eval/runner.py
Affected API: BrowserController, DOMObserver, format_elements, prune_elements
Data schemas: Uses core.types.Observation
"""

from browser.controller import BrowserController
from browser.observer import DOMObserver, format_elements, prune_elements

__all__ = [
    "BrowserController",
    "DOMObserver",
    "format_elements",
    "prune_elements",
]
