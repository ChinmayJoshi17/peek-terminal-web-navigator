"""DOM Observer & Token Pruning for LLM observation formatting.

Responsible for pruning raw DOM element trees into strictly actionable items,
stripping non-interactive wrappers, and serializing element lists into compact,
token-efficient representations kept well below character caps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

# Maximum characters allowed for serialized DOM elements in the LLM prompt.
MAX_ELEMENTS_TEXT_CHARS: int = 8000

# Strict target size kept well below 8000 to prevent provider token-per-minute (TPM) overflow
TARGET_MAX_CHARS: int = 3500

# Strictly actionable HTML tags
ACTIONABLE_TAGS: Set[str] = {
    "input",
    "textarea",
    "button",
    "select",
    "option",
}

# Roles that represent actionable interactive elements
ACTIONABLE_ROLES: Set[str] = {
    "button",
    "link",
    "searchbox",
    "combobox",
    "menuitem",
    "tab",
    "checkbox",
    "radio",
    "textbox",
}

# Non-interactive wrapper tags to strip unless explicitly interactive
NON_ACTIONABLE_WRAPPERS: Set[str] = {
    "div",
    "span",
    "section",
    "p",
    "header",
    "footer",
    "nav",
    "main",
    "article",
    "aside",
    "ul",
    "ol",
    "li",
}

# Values representing empty/dummy anchor targets
EMPTY_HREF_VALUES: Set[str] = {
    "",
    "#",
    "/#",
    "javascript:void(0)",
    "javascript:void(0);",
    "javascript:;",
    "javascript:",
}

# Patterns for non-actionable clutter (e.g. Wikipedia citation numbers [1], [edit], jump links)
CLUTTER_TEXT_PATTERNS = [
    re.compile(r"^\[\d+\]$"),
    re.compile(r"^\[edit\]$", re.IGNORECASE),
    re.compile(r"^jump to", re.IGNORECASE),
]


class DOMObserver:
    """Observer utility that prunes DOM elements and serializes them for LLM consumption."""

    MAX_ELEMENTS_TEXT_CHARS: int = MAX_ELEMENTS_TEXT_CHARS
    TARGET_MAX_CHARS: int = TARGET_MAX_CHARS

    @staticmethod
    def is_actionable(element: Dict[str, Any]) -> bool:
        """Determine whether an extracted DOM element is strictly actionable."""
        if not isinstance(element, dict):
            return False

        tag = str(element.get("tag", "")).strip().lower()
        role = str(element.get("role", "")).strip().lower()
        cursor = str(element.get("cursor", "")).strip().lower()
        href = str(element.get("href", "")).strip()
        text = str(element.get("text", "")).strip()
        placeholder = str(element.get("placeholder", "")).strip()
        aria_label = str(element.get("aria_label", "")).strip()
        input_type = str(element.get("type", "")).strip().lower()

        # Reject hidden inputs
        if input_type == "hidden":
            return False

        # Strictly actionable form controls and buttons
        if tag in ACTIONABLE_TAGS:
            return True

        # Elements with actionable ARIA roles or cursor pointer
        if role in ACTIONABLE_ROLES or role == "button":
            return True

        # Strip clutter / citations like [1], [edit]
        if any(p.match(text) for p in CLUTTER_TEXT_PATTERNS):
            return False

        if cursor == "pointer":
            if text or aria_label or placeholder:
                return True

        # Links (a[href])
        if tag == "a":
            # Strip empty/dummy anchor targets unless explicitly marked as a button
            if not href or href in EMPTY_HREF_VALUES or href.startswith("javascript:") or href.startswith("#"):
                return bool(role == "button" and (text or aria_label))
            # Has a valid href target; retain if it has meaningful label/text
            return bool(text or aria_label or placeholder)

        # Non-interactive wrappers (div, span, section, footer, nav) without explicit event listeners
        if tag in NON_ACTIONABLE_WRAPPERS:
            if role in ACTIONABLE_ROLES or cursor == "pointer":
                return bool(text or aria_label or placeholder)
            return False

        # Discard anything that has no text, no placeholder, no aria-label, and is not an input
        if not (text or placeholder or aria_label):
            return False

        return False

    @classmethod
    def prune_elements(cls, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of extracted DOM elements to retain strictly actionable elements."""
        if not elements:
            return []
        return [el for el in elements if cls.is_actionable(el)]

    @classmethod
    def serialize(
        cls,
        elements: List[Dict[str, Any]],
        max_chars: Optional[int] = None,
    ) -> str:
        """Serialize interactive elements into a concise prompt representation.

        Strips redundant wrappers, caps individual field sizes, and ensures the
        total formatted string remains well below the maximum character threshold.
        """
        pruned = cls.prune_elements(elements)
        if not pruned:
            return "No interactive elements found on this page."

        # Keep well below 8000 (cap at min of passed max_chars and TARGET_MAX_CHARS)
        effective_limit = min(max_chars or cls.TARGET_MAX_CHARS, cls.TARGET_MAX_CHARS)
        lines: List[str] = []
        total_chars = 0
        truncated = False

        for el in pruned:
            el_id = str(el.get("id", "")).strip()
            tag = str(el.get("tag", "")).strip().lower()
            type_val = str(el.get("type", "")).strip()
            role_val = str(el.get("role", "")).strip()
            text = str(el.get("text", "")).strip()
            placeholder = str(el.get("placeholder", "")).strip()
            aria_label = str(el.get("aria_label", "")).strip()

            # Compact whitespaces within text
            text = re.sub(r"\s+", " ", text)
            placeholder = re.sub(r"\s+", " ", placeholder)
            aria_label = re.sub(r"\s+", " ", aria_label)

            # Compact per-element field caps
            safe_text = (text[:80] + "…") if len(text) > 80 else text
            safe_placeholder = (
                (placeholder[:50] + "…") if len(placeholder) > 50 else placeholder
            )
            safe_aria = (aria_label[:50] + "…") if len(aria_label) > 50 else aria_label

            parts = [f"{el_id}: ({tag}"]
            if type_val and type_val != "text":
                parts.append(f", type={type_val}")
            if role_val and role_val != tag:
                parts.append(f", role={role_val}")
            parts.append(f') "{safe_text}"')

            if safe_placeholder:
                parts.append(f' placeholder="{safe_placeholder}"')
            if safe_aria and safe_aria != safe_text:
                parts.append(f' aria-label="{safe_aria}"')

            line = "".join(parts)

            # Check if adding this line would exceed the character limit
            if total_chars + len(line) + 1 > effective_limit:
                truncated = True
                break

            lines.append(line)
            total_chars += len(line) + 1

        output = "\n".join(lines)
        if truncated:
            output += (
                f"\n\n[... Truncated: showing {len(lines)} of {len(pruned)} "
                f"actionable elements to prevent token overflow ...]"
            )

        return output


def prune_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience helper to prune non-actionable elements."""
    return DOMObserver.prune_elements(elements)


def format_elements(
    elements: List[Dict[str, Any]],
    max_chars: int = TARGET_MAX_CHARS,
) -> str:
    """Convenience helper to prune and serialize elements for prompt inclusion."""
    return DOMObserver.serialize(elements, max_chars=max_chars)
