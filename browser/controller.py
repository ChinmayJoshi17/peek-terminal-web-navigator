"""Isolated Playwright browser controller with explicit lifecycle handling.

Importers/callers: main.py instantiates BrowserController; agent/loop.py calls scan/click/type/navigate
Affected API: BrowserController (lifecycle, DOM extraction, actions)
Data schemas: Returns Dict with elements, url, title (compatible with core.types.Observation)
"""

from __future__ import annotations

import asyncio
import gc
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Locator, Page, Playwright

from core.errors import BrowserError, DOMExtractionError


class BrowserController:
    """Isolated Playwright controller with explicit lifecycle.

    Features:
    - Explicit start()/close() lifecycle
    - Resilient locator resolution with user-facing attribute fallbacks
    - Element interaction with focus/click event listener triggering
    - Keyboard dispatch fallback for dynamic inputs
    - DOM pruning via injected JavaScript
    - Error boundaries: missing DOM elements do not crash the agent
    - Defensive badge cleanup
    - Auto-retry on transient errors
    """

    def __init__(
        self,
        headless: bool = False,
        slow_mo: int = 300,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        load_timeout: int = 30000,
    ) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.load_timeout = load_timeout

        # Internal state
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._element_cache: Dict[str, Dict[str, Any]] = {}

        # Resolve dom_pruner.js path once
        self._dom_pruner_path = Path(__file__).parent / "dom_pruner.js"

    @property
    def page(self) -> Optional[Page]:
        """Return current page (for status checks)."""
        return self._page

    @property
    def is_started(self) -> bool:
        """Check if browser is started."""
        return self._browser is not None and self._page is not None

    async def start(self) -> str:
        """Launch Playwright Chromium with configuration and create a page with viewport."""
        if self.is_started:
            return "Browser already started"

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
            )
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
            await self._page.set_viewport_size({
                "width": self.viewport_width,
                "height": self.viewport_height,
            })
            # Block heavy, non-essential network assets irrelevant to DOM parsing & LLM reasoning
            await self._page.route(
                "**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf,otf,eot,mp4,webm,mp3}",
                lambda route: route.abort(),
            )
            return "Browser started successfully"
        except Exception as e:
            await self._cleanup_on_error()
            raise BrowserError(
                f"Error starting browser: {str(e)}",
                original_exception=e,
            ) from e

    async def navigate(self, url: str) -> str:
        """Navigate to the specified URL and wait for DOM content to load."""
        if not self.is_started:
            raise BrowserError("Browser not started. Call start() first.")

        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=self.load_timeout)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"

    async def scan(self) -> Dict[str, Any]:
        """Read dom_pruner.js, execute it in the page, and return interactive elements."""
        if not self.is_started or self._page is None:
            return {
                "error": "Browser not started",
                "elements": [],
                "url": "",
                "title": "",
            }

        try:
            dom_pruner_code = self._dom_pruner_path.read_text(encoding="utf-8")
            await self._page.evaluate(dom_pruner_code)

            try:
                elements = await self._page.evaluate("() => window.getInteractiveElements()")
            except Exception as e:
                raise DOMExtractionError(
                    f"Failed to extract interactive elements: {str(e)}",
                    url=self._page.url,
                ) from e

            if elements is None or not isinstance(elements, list):
                elements = []

            # Cache element metadata for fallback selector resolution
            self._element_cache = {
                el["id"]: el for el in elements if isinstance(el, dict) and el.get("id")
            }

            current_url = self._page.url
            try:
                title = await self._page.title()
            except Exception:
                title = ""

            return {
                "elements": elements,
                "url": current_url,
                "title": title,
            }

        except Exception as e:
            return {
                "error": f"Error scanning page: {str(e)}",
                "elements": [],
                "url": self._page.url if self._page else "",
                "title": "",
            }

    async def _resolve_locator(
        self,
        element_id: str,
        is_input: bool = False,
    ) -> Optional[Locator]:
        """Resolve Playwright locator by agent id with resilient fallback resolution.

        Attempts:
        1. Direct [data-agent-id="..."] selector.
        2. Cached attribute resolution:
           - get_by_role("combobox", name=...) or get_by_role("searchbox")
           - get_by_placeholder(...)
           - get_by_label(...)
        3. Common user-facing input / searchbox fallbacks for dynamic pages.
        """
        if not self._page:
            return None

        # 1. Primary resolution: data-agent-id with short actionability timeout
        if element_id:
            try:
                loc = self._page.locator(f'[data-agent-id="{element_id}"]')
                if await loc.count() > 0 and await loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

        cached = self._element_cache.get(element_id, {})
        tag = str(cached.get("tag", "")).lower()
        role = str(cached.get("role", "")).lower()
        aria_label = str(cached.get("aria_label", "")).strip()
        placeholder = str(cached.get("placeholder", "")).strip()
        text = str(cached.get("text", "")).strip()
        name = str(cached.get("name", "")).strip()

        # 2. Cached user-facing attribute fallbacks
        # Try placeholder
        if placeholder:
            try:
                loc = self._page.get_by_placeholder(placeholder, exact=False)
                if await loc.count() > 0 and await loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

        # Try aria-label / label
        if aria_label:
            try:
                loc = self._page.get_by_label(aria_label, exact=False)
                if await loc.count() > 0 and await loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

        # Try role + accessible name
        if role:
            for query_name in [text, aria_label, placeholder]:
                if not query_name:
                    continue
                try:
                    loc = self._page.get_by_role(role, name=query_name, exact=False)
                    if await loc.count() > 0 and await loc.first.is_visible():
                        return loc.first
                except Exception:
                    pass

        # 3. Dynamic input & searchbox fallbacks
        if is_input or tag in ("input", "textarea") or role in ("combobox", "searchbox"):
            # Try combobox with user-facing name
            for query_name in [aria_label, placeholder, text, name]:
                if not query_name:
                    continue
                try:
                    loc = self._page.get_by_role("combobox", name=query_name, exact=False)
                    if await loc.count() > 0 and await loc.first.is_visible():
                        return loc.first
                except Exception:
                    pass

            # Try combobox without name
            try:
                loc = self._page.get_by_role("combobox")
                if await loc.count() > 0 and await loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

            # Try searchbox role
            try:
                loc = self._page.get_by_role("searchbox")
                if await loc.count() > 0 and await loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

            # Search by common input names and placeholders
            search_selectors = [
                'input[type="search"]',
                'input[name="search"]',
                'input[name="q"]',
                'input[title*="Search" i]',
                'input[placeholder*="Search" i]',
                'textarea[placeholder*="Search" i]',
            ]
            for sel in search_selectors:
                try:
                    loc = self._page.locator(sel)
                    if await loc.count() > 0 and await loc.first.is_visible():
                        return loc.first
                except Exception:
                    continue

        # 4. Actionable button/link text fallbacks
        if text:
            for r in ["button", "link"]:
                try:
                    loc = self._page.get_by_role(r, name=text, exact=False)
                    if await loc.count() > 0 and await loc.first.is_visible():
                        return loc.first
                except Exception:
                    pass
            try:
                loc = self._page.get_by_text(text, exact=False)
                if await loc.count() > 0 and await loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

        # 5. Last-ditch: wait for [data-agent-id] with a brief actionability timeout
        if element_id:
            try:
                loc = self._page.locator(f'[data-agent-id="{element_id}"]')
                await loc.first.wait_for(state="visible", timeout=2000)
                return loc.first
            except Exception:
                pass

        return None

    async def click(self, element_id: str) -> str:
        """Locate element by data-agent-id with fallback resolution and click it."""
        if not self.is_started or self._page is None:
            return "Error: Browser not started"
        if not element_id:
            return "Error: element_id is required for click"

        try:
            target = await self._resolve_locator(element_id, is_input=False)
            if target is None:
                return f"Error: element {element_id} not found or not visible"

            # Check visibility and invoke explicit focus to trigger attached JS listeners
            try:
                await target.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass

            try:
                await target.focus(timeout=2000)
            except Exception:
                pass

            await target.click(timeout=5000)
            await self._clean_badges()

            try:
                await self._page.wait_for_load_state("networkidle", timeout=min(5000, self.load_timeout))
            except Exception:
                pass

            return f"Clicked element {element_id}"

        except Exception as e:
            return f"Error clicking element {element_id}: {str(e)}"

    async def type(
        self,
        element_id: str,
        text: str,
        press_enter: bool = False,
    ) -> str:
        """Locate element by data-agent-id with fallback, fill text, optionally press Enter.

        Ensures explicit click()/focus() to activate dynamic inputs and event listeners,
        uses locator.fill(text) with a 5000ms timeout, and falls back to page.keyboard.type.
        """
        if not self.is_started or self._page is None:
            return "Error: Browser not started"
        if not element_id:
            return "Error: element_id is required for type"

        try:
            target = await self._resolve_locator(element_id, is_input=True)
            if target is None:
                return f"Error: element {element_id} not found or not visible"

            # Check visibility and invoke explicit click() or focus() to trigger JS listeners
            try:
                await target.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass

            try:
                await target.click(timeout=3000)
            except Exception:
                try:
                    await target.focus(timeout=2000)
                except Exception:
                    pass

            # Direct locator fill with short 5000ms actionability timeout
            typed = False
            try:
                await target.fill(text, timeout=5000)
                typed = True
            except Exception:
                # Fallback to keyboard dispatch
                try:
                    await target.focus(timeout=2000)
                except Exception:
                    pass
                try:
                    await self._page.keyboard.type(text)
                    typed = True
                except Exception as e_kb:
                    return f"Error typing into element {element_id}: {str(e_kb)}"

            if press_enter:
                # Dispatch Enter, falling back to keyboard press if locator press times out
                try:
                    await target.press("Enter", timeout=3000)
                except Exception:
                    try:
                        await self._page.keyboard.press("Enter")
                    except Exception:
                        pass

            await self._clean_badges()

            try:
                await self._page.wait_for_load_state("domcontentloaded", timeout=min(5000, self.load_timeout))
            except Exception:
                pass

            return f"Typed into element {element_id}"

        except Exception as e:
            return f"Error typing into element {element_id}: {str(e)}"

    async def scroll(self, direction: str = "down", amount: int = 500) -> str:
        """Scroll the page in a given direction."""
        if not self.is_started or self._page is None:
            return "Error: Browser not started"

        try:
            if direction == "down":
                await self._page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                await self._page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "left":
                await self._page.evaluate(f"window.scrollBy(-{amount}, 0)")
            elif direction == "right":
                await self._page.evaluate(f"window.scrollBy({amount}, 0)")
            else:
                return f"Error: invalid scroll direction '{direction}'"

            await asyncio.sleep(0.3)
            return f"Scrolled {direction} by {amount}px"
        except Exception as e:
            return f"Error scrolling: {str(e)}"

    async def _clean_badges(self) -> None:
        """Defensively clean up injected DOM badges."""
        if not self._page:
            return
        try:
            await self._page.evaluate(
                "() => { if (typeof window.cleanBadges === 'function') window.cleanBadges(); }"
            )
        except Exception:
            pass

    async def _cleanup_on_error(self) -> None:
        """Clean up resources on error during start."""
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

        self._context = None
        self._browser = None
        self._page = None
        self._playwright = None
        self._element_cache.clear()

    async def close(self) -> str:
        """Close browser context and Playwright cleanly."""
        try:
            if self._page:
                try:
                    await self._page.close()
                except Exception:
                    pass
                self._page = None

            if self._context:
                try:
                    await self._context.close()
                except Exception:
                    pass
                self._context = None

            if self._browser:
                try:
                    await self._browser.close()
                except Exception:
                    pass
                self._browser = None

            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

            self._element_cache.clear()
            gc.collect()
            await asyncio.sleep(0.1)

            return "Browser closed successfully"
        except Exception as e:
            return f"Error closing browser: {str(e)}"

    async def __aenter__(self) -> "BrowserController":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
