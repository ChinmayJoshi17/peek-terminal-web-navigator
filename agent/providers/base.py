"""Abstract LLM client base class — provider-agnostic interface.

Importers/callers: agent/loop.py imports from agent.providers.base
Affected API: Chat completion interface with error handling
Data schemas: Uses core.types.Action for structured output
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.types import Action


@dataclass
class LLMResponse:
    """Structured LLM response.

    Fields: action (core.types.Action), raw_text (str), success (bool),
    error (Optional[str]), provider (str), model (str), tokens_used (int),
    duration_ms (float).
    """

    action: Action
    raw_text: str
    success: bool = True
    error: Optional[str] = None
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    duration_ms: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if the response contains a valid action."""
        return self.success and self.action is not None


class LLMClient(ABC):
    """Abstract base class for LLM providers.

    Provides a provider-agnostic interface for the agent loop.
    Concrete implementations include GroqClient, OpenAIClient, OllamaClient.

    Usage:
        client = GroqClient(api_key="...", model="openai/gpt-oss-120b")
        response = await client.chat(messages)
    """

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_format: Optional response format override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            LLMResponse with parsed action and metadata

        Raises:
            LLMError: On API failure, network error, or timeout
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name."""
        ...

    @abstractmethod
    def get_model(self) -> str:
        """Return the current model identifier."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up provider resources."""
        ...

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[tuple]] = None,
    ) -> List[Dict[str, str]]:
        """Build a messages list for the chat API.

        Args:
            system_prompt: System/role prompt
            user_prompt: User message
            history: Optional conversation history (action, element_id, url)

        Returns:
            List of message dicts
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if history:
            for action, element_id, url in history:
                messages.append({
                    "role": "assistant",
                    "content": f'{{"action": "{action}", "element_id": "{element_id}"}}',
                })
                messages.append({
                    "role": "user",
                    "content": f"Previous action at {url}. Current state...",
                })

        messages.append({"role": "user", "content": user_prompt})
        return messages