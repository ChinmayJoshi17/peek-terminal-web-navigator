"""Typed exception hierarchy for nano-browser-agent.

Importers/callers: main.py, agent/loop.py, browser/controller.py
"""

from __future__ import annotations

from typing import Any, Optional


class AgentError(Exception):
    """Base exception for all agent errors.

    Raised when any agent operation fails. Imported from core.errors into
    main.py, agent/loop.py, and browser/controller.py for structured error
    handling.
    """

    def __init__(
        self,
        message: str,
        *,
        step_number: Optional[int] = None,
        action: Optional[str] = None,
        element_id: Optional[str] = None,
        url: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.step_number = step_number
        self.action = action
        self.element_id = element_id
        self.url = url
        self.original_exception = original_exception

    def __str__(self) -> str:
        base = f"[{self.__class__.__name__}] {self.message}"
        if self.action:
            base += f" | action={self.action}"
        if self.element_id:
            base += f" | element={self.element_id}"
        if self.step_number is not None:
            base += f" | step={self.step_number}"
        return base


class ConfigurationError(AgentError):
    """Raised when configuration is invalid or missing.

    Affected callers: config.py (Settings.load()), main.py (main())
    """

    def __init__(self, message: str, *, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.field = field


class BrowserError(AgentError):
    """Raised when a browser operation fails.

    Affected callers: browser/controller.py, browser/driver.py
    """

    def __init__(
        self,
        message: str,
        *,
        url: Optional[str] = None,
        element_id: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message,
            url=url,
            element_id=element_id,
            original_exception=original_exception,
        )


class LLMError(AgentError):
    """Raised when an LLM call fails.

    Affected callers: agent/llm/base.py, agent/llm/groq.py
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, original_exception=original_exception)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.response_body = response_body


class GuardrailError(AgentError):
    """Raised when a guardrail blocks an action.

    Affected callers: agent/guardrails.py, agent/loop.py
    """

    def __init__(
        self,
        message: str,
        *,
        action: Optional[str] = None,
        element_id: Optional[str] = None,
        history_length: Optional[int] = None,
    ) -> None:
        super().__init__(message, action=action, element_id=element_id)
        self.history_length = history_length


class CheckpointError(AgentError):
    """Raised when checkpoint save/restore fails.

    Affected callers: loop/state.py, loop/operator.py
    """

    def __init__(self, message: str, *, checkpoint_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.checkpoint_id = checkpoint_id


class DOMExtractionError(AgentError):
    """Raised when DOM element extraction fails or returns no elements.

    Affected callers: browser/driver.py, core/types.py (DOMElement)
    """

    def __init__(
        self,
        message: str,
        *,
        element_id: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        super().__init__(message, element_id=element_id, url=url)


class ValidationError(AgentError):
    """Raised when LLM response fails schema validation.

    Affected callers: core/errors.py, agent/loop.py (_call_llm error returns)
    """

    def __init__(
        self,
        message: str,
        *,
        raw_response: Optional[str] = None,
        errors: Optional[list] = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.errors = errors or []

    def __str__(self) -> str:
        base = super().__str__()
        if self.errors:
            base += f" | validation errors: {self.errors}"
        return base