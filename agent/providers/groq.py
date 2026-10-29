"""Groq LLM provider implementation.

Importers/callers: agent/loop.py creates GroqClient instances
Affected API: Groq API chat completions endpoint
Data schemas: Uses core.types.Action for structured output
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
import requests

from agent.providers.base import LLMClient, LLMResponse
from core.errors import LLMError
from core.types import Action


class GroqClient(LLMClient):
    """Groq LLM client implementation.

    Features:
    - Automatic retry with exponential backoff
    - Structured JSON output enforcement
    - Error handling with detailed diagnostics
    - Performance metrics collection
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        *,
        timeout: int = 60,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

        self._metrics: Dict[str, Any] = {
            "requests": 0,
            "successes": 0,
            "errors": 0,
            "total_tokens": 0,
            "total_time_ms": 0.0,
        }

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat completion request to Groq API.

        Implements retry logic and error handling per ECC guidelines.
        """
        start_time = time.time()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build request payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = (
                {"type": "json_object"}
                if response_format == "json"
                else response_format
            )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._metrics["requests"] += 1
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                # Handle HTTP errors
                if response.status_code >= 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get(
                            "message", str(error_data)
                        )
                    except Exception:
                        error_msg = response.text[:200]
                    raise LLMError(
                        f"Groq API HTTP {response.status_code}: {error_msg}",
                        provider="groq",
                        model=self.model,
                        status_code=response.status_code,
                        response_body=response.text[:500],
                    )

                data = response.json()

                # Extract content with defensive parsing
                choices = data.get("choices") or [{}]
                message = choices[0].get("message", {}) if choices else {}
                raw_content = message.get("content", "{}")

                # Parse JSON content
                try:
                    parsed = json.loads(raw_content)
                except json.JSONDecodeError as e:
                    raise LLMError(
                        f"Non-JSON response from LLM: {raw_content[:200]}",
                        provider="groq",
                        model=self.model,
                    ) from e

                # Validate and convert to Action TypedDict
                action_dict: Action = {
                    "thought": parsed.get("thought", ""),
                    "action": parsed.get("action", "done"),
                    "element_id": parsed.get("element_id", ""),
                    "text": parsed.get("text", ""),
                    "final_answer": parsed.get("final_answer", ""),
                    "scroll_direction": parsed.get(
                        "scroll_direction", "down"
                    ),
                    "wait_ms": int(parsed.get("wait_ms", 0)),
                }

                duration_ms = (time.time() - start_time) * 1000

                # Update metrics
                self._metrics["successes"] += 1
                self._metrics["total_tokens"] += data.get("usage", {}).get(
                    "total_tokens", 0
                )
                self._metrics["total_time_ms"] += duration_ms

                return LLMResponse(
                    action=action_dict,
                    raw_text=raw_content,
                    success=True,
                    provider="groq",
                    model=self.model,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                    duration_ms=duration_ms,
                )

            except Exception as e:
                last_error = e
                self._metrics["errors"] += 1
                if attempt < self.max_retries:
                    wait_time = float(2 ** attempt)
                    err_str = str(e)
                    if "429" in err_str or "Rate limit" in err_str:
                        m = re.search(r"try again in ([\d\.]+)s", err_str)
                        if m:
                            try:
                                wait_time = float(m.group(1)) + 1.2
                            except Exception:
                                wait_time = max(wait_time, 3.5)
                        else:
                            wait_time = max(wait_time, 3.5)
                    await asyncio.sleep(wait_time)
                else:
                    break

        # All retries exhausted
        duration_ms = (time.time() - start_time) * 1000
        error_msg = (
            str(last_error)
            if last_error
            else f"Groq API request failed after {self.max_retries} retries"
        )

        raise LLMError(
            f"LLM request failed: {error_msg}",
            provider="groq",
            model=self.model,
            original_exception=last_error,
        ) from last_error

    def get_name(self) -> str:
        """Return the provider name."""
        return "groq"

    def get_model(self) -> str:
        """Return the current model identifier."""
        return self.model

    async def close(self) -> None:
        """Clean up provider resources."""
        # Groq uses requests which doesn't need explicit cleanup
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics."""
        return self._metrics.copy()