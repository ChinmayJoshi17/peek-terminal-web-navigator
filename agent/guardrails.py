"""Configurable loop guardrails to prevent infinite repetition.

Importers/callers: agent/loop.py uses Guardrail for action validation
Affected API: Guardrail.check() returns Optional[str] warning
Data schemas: Uses core.types.Action TypedDict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.types import Action


@dataclass
class GuardrailConfig:
    """Configuration for guardrail behavior."""

    max_history: int = 3
    max_same_element_retries: int = 2
    url_change_required: bool = True
    block_on_stale_elements: bool = True


class Guardrail:
    """Configurable guardrail preventing infinite action loops.

    Monitors action history and blocks repeated actions that fail to
    change page state. Called by AgentLoop before executing each action.

    Usage:
        guardrail = Guardrail()
        warning = guardrail.check(action, element_id, url, history)
        if warning:
            # Re-prompt LLM with warning
    """

    def __init__(self, config: Optional[GuardrailConfig] = None) -> None:
        self.config = config or GuardrailConfig()
        self._action_counts: dict[str, int] = {}

    def check(
        self,
        action: Action,
        element_id: str,
        current_url: str,
        history: List[Tuple[str, str, str]],
    ) -> Optional[str]:
        """Check if an action should be blocked.

        Args:
            action: The proposed Action
            element_id: Target element ID
            current_url: Current page URL
            history: List of (action, element_id, url) tuples

        Returns:
            Warning string if blocked, None if action is allowed
        """
        action_type = action.get("action", "")
        key = f"{action_type}:{element_id}"

        # Count same action+element repetitions
        self._action_counts[key] = self._action_counts.get(key, 0) + 1
        if self._action_counts[key] > self.config.max_same_element_retries:
            return (
                f"ACTION BLOCKED: Same action '{action_type}' on element '{element_id}' "
                f"exceeded retry limit ({self.config.max_same_element_retries}). "
                f"Choose a DIFFERENT approach."
            )

        # Check history for loops
        check_entries = history[-self.config.max_history:] if history else []
        for prev_action, prev_element, prev_url in check_entries:
            if (
                prev_action == action_type
                and prev_element == element_id
                and prev_url == current_url
                and self.config.url_change_required
            ):
                return (
                    f"GUARDRAIL ALERT: Repeating action '{action_type}' on element "
                    f"'{element_id}' at same URL without page state change. "
                    f"Do NOT repeat this action. Choose a DIFFERENT approach or scroll."
                )

        return None

    def check_element_valid(
        self,
        element_id: str,
        available_elements: List[dict],
    ) -> Optional[str]:
        """Verify element_id is in the current element list.

        Args:
            element_id: Element ID to validate
            available_elements: List from browser scan

        Returns:
            Warning string if element not found, None if valid
        """
        if not element_id:
            return None

        if not available_elements:
            return "WARNING: No interactive elements found on page. Try scrolling first."

        element_ids = {el.get("id", "") for el in available_elements}
        if element_id not in element_ids:
            return (
                f"WARNING: Element '{element_id}' not found in current page elements. "
                f"Available: {sorted(element_ids)[:10]}. "
                f"Re-scan the page and try again."
            )

        return None

    def check_navigate_valid(self, url: str) -> Optional[str]:
        """Validate a navigation URL.

        Args:
            url: Target URL

        Returns:
            Warning string if URL is invalid, None if valid
        """
        if not url or not url.strip():
            return "WARNING: Empty URL provided for navigation."

        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            # Try to fix common issues
            if url.startswith("www."):
                url = "https://" + url
            else:
                return (
                    f"WARNING: URL '{url}' doesn't look valid. "
                    "URLs should start with http:// or https://."
                )

        return None

    def reset(self) -> None:
        """Reset guardrail state for a new session."""
        self._action_counts.clear()

    def reset_element(self, element_id: str) -> None:
        """Reset count for a specific element."""
        keys_to_remove = [k for k in self._action_counts if k.endswith(f":{element_id}")]
        for k in keys_to_remove:
            del self._action_counts[k]