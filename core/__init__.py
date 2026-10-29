"""Core package — configuration, types, and foundational abstractions."""

from core.config import Settings
from core.types import Action, Observation, StepResult, DOMElement
from core.errors import (
    AgentError,
    BrowserError,
    LLMError,
    ConfigurationError,
    GuardrailError,
)

__all__ = [
    "Settings",
    "Action",
    "Observation",
    "StepResult",
    "DOMElement",
    "AgentError",
    "BrowserError",
    "LLMError",
    "ConfigurationError",
    "GuardrailError",
]
