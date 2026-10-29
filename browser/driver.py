import asyncio
import gc
import os
from playwright.async_api import async_playwright
from typing import List, Dict, Any, Optional


class BrowserDriver:
    def __init__(self, headless: bool = False, slow_mo: int = 300):
        self.headless = headless
        self.slow_mo = slow_mo
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None

    async def start(self) -> str:
        """Launch Playwright Chromium with configuration and create a page with viewport."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo
            )

            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()

            # Set viewport to 1280x800
            await self.page.set_viewport_size({
                "width": 1280,
                "height": 800
            })

            return "Browser started successfully"
        except Exception as e:
            return f"Error starting browser: {str(e)}"

    async def navigate(self, url: str) -> str:
        """Navigate to the specified URL and wait for DOM content to load."""
        try:
            await self.page.goto(url, wait_until="domcontentloaded")
            return f"Navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"

    async def scan(self) -> Dict[str, Any]:
        """Read dom_pruner.js, execute it in the page, and return interactive elements."""
        try:
            # Resolve dom_pruner.js path relative to this file's directory
            dom_pruner_path = os.path.join(os.path.dirname(__file__), 'dom_pruner.js')
            with open(dom_pruner_path, 'r') as f:
                dom_pruner_code = f.read()

            # Execute the dom_pruner.js code in the page
            await self.page.evaluate(dom_pruner_code)

            # Call window.getInteractiveElements()
            elements = await self.page.evaluate("() => window.getInteractiveElements()")

            # Get current page info
            current_url = self.page.url
            title = await self.page.title()

            return {
                "elements": elements,
                "url": current_url,
                "title": title
            }
        except Exception as e:
            return {
                "error": f"Error scanning page: {str(e)}",
                "elements": [],
                "url": self.page.url if self.page else "",
                "title": await self.page.title() if self.page else ""
            }

    async def click(self, element_id: str) -> str:
        """Locate element by data-agent-id and click it, then clean badges."""
        try:
            # Wait for the element to be available
            await self.page.wait_for_selector(f'[data-agent-id="{element_id}"]', state="visible")

            # Click the element
            await self.page.click(f'[data-agent-id="{element_id}"]')

            # Call window.cleanBadges() defensively — no-op if not yet defined
            try:
                await self.page.evaluate("() => { if (typeof window.cleanBadges === 'function') window.cleanBadges(); }")
            except Exception:
                pass

            # Wait for network idle or domcontentloaded
            await self.page.wait_for_load_state("networkidle")

            return f"Clicked element {element_id}"
        except Exception as e:
            return f"Error clicking element {element_id}: {str(e)}"

    async def type(self, element_id: str, text: str, press_enter: bool = False) -> str:
        """Locate element by data-agent-id, fill text, optionally press Enter, and clean badges."""
        try:
            # Wait for the element to be available
            await self.page.wait_for_selector(f'[data-agent-id="{element_id}"]', state="visible")

            # Fill the text
            await self.page.fill(f'[data-agent-id="{element_id}"]', text)

            # Optionally press Enter
            if press_enter:
                await self.page.press(f'[data-agent-id="{element_id}"]', "Enter")

            # Call window.cleanBadges() defensively — no-op if not yet defined
            try:
                await self.page.evaluate("() => { if (typeof window.cleanBadges === 'function') window.cleanBadges(); }")
            except Exception:
                pass

            # Wait for load state
            await self.page.wait_for_load_state("domcontentloaded")

            return f"Typed into element {element_id}"
        except Exception as e:
            return f"Error typing into element {element_id}: {str(e)}"

    async def close(self) -> str:
        """Close browser context and Playwright cleanly."""
        try:
            if self.page:
                await self.page.close()
                self.page = None

            if self.context:
                await self.context.close()
                self.context = None

            if self.browser:
                await self.browser.close()
                self.browser = None

            if self.playwright:
                await self.playwright.stop()
                self.playwright = None

            # Force GC to collect transport/subprocess objects before interpreter shutdown
            gc.collect()
            await asyncio.sleep(0.1)

            return "Browser closed successfully"
        except Exception as e:
            return f"Error closing browser: {str(e)}"

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
