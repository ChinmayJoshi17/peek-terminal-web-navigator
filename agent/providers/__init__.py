"""LLM provider abstractions — provider-agnostic interface with Groq implementation.

Per ECC specs: agent/providers/ provides provider-agnostic LLM interface
(base class with Groq implementation, ready for local Ollama/OpenAI extension).

Importers/callers: agent/loop.py imports from agent.providers
Affected API: Chat completion interface with error handling
Data schemas: Uses core.types.Action for structured output
"""

from agent.providers.base import LLMClient, LLMResponse
from agent.providers.groq import GroqClient

__all__ = ["LLMClient", "LLMResponse", "GroqClient"]