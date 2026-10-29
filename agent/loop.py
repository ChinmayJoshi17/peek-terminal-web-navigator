"""ReAct loop orchestrator — state machine for Observe → Reason → Act → Verify.

Importers/callers: main.py imports AgentLoop
Affected API: AgentLoop.run() executes the autonomous loop
Data schemas: Uses core.types (Action, Observation, LoopState, StepResult)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import Settings, get_settings
from core.errors import AgentError, LLMError, ConfigurationError
from core.types import (
    Action,
    LoopState,
    LoopStatus,
    Observation,
    StepResult,
)
from core.telemetry import log_step_event, make_telemetry, measure_latency
from agent.guardrails import Guardrail, GuardrailConfig
from agent.providers.base import LLMClient
from agent.providers.groq import GroqClient
from browser.controller import BrowserController
from browser.observer import DOMObserver

# System prompt for the agent
SYSTEM_PROMPT = """
You are a browser automation agent that operates by selecting elements and performing actions.

Your job is to achieve the user's goal by repeatedly scanning the current page for interactive elements, choosing the next action, and executing it.

After each action, you'll see the result. Learn from whether the page state changed (new URL, new elements) or stayed the same.

Return your decision as STRICT JSON matching this schema:

{
  "thought": "concise reasoning",
  "action": "click" | "type" | "navigate" | "done",
  "element_id": "iX (or empty)",
  "text": "input text if typing",
  "final_answer": "final answer string if done"
}

Rules:
- Use "navigate" with element_id empty and text set to the URL to navigate somewhere.
- Use "click" with element_id set to the element you want to click.
- Use "type" with element_id set to the input field and text set to the text to type.
- Use "done" with final_answer set to your conclusion when the goal is achieved.
- If you cannot find a suitable element to achieve the goal, use "done" and explain.
- You will see elements in the format: i1: (button) "Click me" placeholder=""
- Do NOT repeat the exact same action on the same element without a page state change.
"""


class AgentLoop:
    MAX_ELEMENTS_TEXT_CHARS: int = 8000
    MAX_PROMPT_CHARS: int = 12000

    """ReAct decision loop agent.

    State machine: Observe → Reason → Act → Verify

    Features:
    - Clean separation between observe/reason/act
    - Configurable guardrails preventing infinite loops
    - Rich UI callback hooks for progress tracking
    - Checkpoint state for recovery
    - Error boundaries for graceful failure
    """

    def __init__(
        self,
        driver: BrowserController,
        llm_client: Optional[LLMClient] = None,
        *,
        settings: Optional[Settings] = None,
        ui_callback: Optional[Callable[[str, Any], None]] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """Initialize the agent loop.

        Args:
            driver: BrowserController for browser automation
            llm_client: Optional LLM client (creates GroqClient if not provided)
            settings: Optional Settings for configuration
            ui_callback: Optional callback for UI updates
            max_steps: Optional max steps override
        """
        self.driver = driver
        self.settings = settings or get_settings()
        self.ui_callback = ui_callback

        # LLM client
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            if not self.settings.llm.api_key:
                raise ConfigurationError(
                    "API key required: set GROQ_API_KEY in .env or pass llm_client"
                )
            self.llm_client = GroqClient(
                api_key=self.settings.llm.api_key,
                model=self.settings.llm.model,
                timeout=self.settings.llm.timeout,
                max_retries=self.settings.llm.max_retries,
                temperature=self.settings.llm.temperature,
            )

        # Loop configuration
        self.max_steps = max_steps or self.settings.loop.max_steps

        # Guardrails
        guardrail_config = GuardrailConfig(
            max_history=self.settings.loop.guardrail_max_history,
            max_same_element_retries=self.settings.loop.guardrail_max_retries,
        )
        self.guardrail = Guardrail(guardrail_config)

        # State
        self.state = LoopState(
            goal="",
            start_url="",
            max_steps=self.max_steps,
        )

    def _format_elements(self, elements: List[Dict[str, Any]]) -> str:
        """Format scanned elements into a concise prompt for the LLM using DOMObserver.

        Strips redundant non-interactive wrappers, retains strictly actionable items,
        and truncates the total text to MAX_ELEMENTS_TEXT_CHARS to prevent
        Groq HTTP 413 (token-per-minute / token limit) overflow.
        """
        return DOMObserver.serialize(elements, max_chars=self.MAX_ELEMENTS_TEXT_CHARS)

    def _build_user_prompt(
        self,
        goal: str,
        observation: Observation,
        step: int,
    ) -> str:
        """Build the user prompt for the LLM.

        Truncates the final prompt to MAX_PROMPT_CHARS to prevent
        HTTP 413 (token-per-minute / request token limit) overflow.
        """
        formatted_elements = self._format_elements(observation.get('elements', []))

        prompt = f"""
Goal: {goal}
Current page title: {observation.get('title', '')}
Current URL: {observation.get('url', '')}
Step {step} of {self.max_steps}

Available interactive elements on this page:
{formatted_elements}

What is your next action? Output STRICT JSON only.
"""

        # Hard cap the total prompt size to prevent HTTP 413 from Groq.
        if len(prompt) > self.MAX_PROMPT_CHARS:
            prompt = prompt[: self.MAX_PROMPT_CHARS] + (
                "\n[... Content truncated to prevent token limit overflow ...]"
            )
        return prompt

    def _emit(self, event: str, data: Any) -> None:
        """Emit UI callback event if registered, and log as telemetry.

        All UI events are also emitted as structured telemetry to the
        JSONL log file so that every agent action is auditable.
        """
        step = self.state.current_step
        url = self.state.start_url
        if event == "observe_complete":
            url = data.get("url", self.state.start_url)
        elif event == "reason_complete":
            url = self.state.start_url
        elif event == "step_complete":
            url = self.state.start_url

        # Log UI event as telemetry (structured, non-blocking)
        try:
            log_step_event(make_telemetry(
                step=step,
                url=url,
                action_type=event,
                action_target=data.get("element_id") or data.get("action", ""),
                thought=data.get("thought") or data.get("final_answer"),
                latency_ms=data.get("duration_ms", 0.0) if event == "reason_complete" else 0.0,
                success=data.get("success", True) if event == "step_complete" else True,
                error=data.get("error") if event in ("observe_error", "reason_error", "act_error", "run_complete") and data.get("status") == "error" else None,
            ))
        except Exception:
            pass  # Telemetry never crashes the loop

        if self.ui_callback:
            try:
                self.ui_callback(event, data)
            except Exception:
                pass  # Don't let UI errors crash the loop

    async def _observe(self) -> Observation:
        """OBSERVE: Scan the browser page for elements."""
        self._emit("observe_start", {"step": self.state.current_step})

        step = self.state.current_step
        url = ""  # Will be filled in after scan
        start = time.perf_counter()

        try:
            scan_result = await self.driver.scan()
            if "error" in scan_result:
                raise AgentError(
                    f"Error scanning page: {scan_result['error']}",
                    step_number=step,
                )

            observation: Observation = {
                "elements": scan_result.get("elements", []),
                "url": scan_result.get("url", ""),
                "title": scan_result.get("title", ""),
                "screenshot_b64": None,
                "viewport_height": self.settings.browser.viewport_height,
                "viewport_width": self.settings.browser.viewport_width,
                "scroll_y": 0,
                "page_load_complete": True,
                "network_idle": True,
            }

            url = observation.get("url", "")
            latency_ms = measure_latency(start)
            self._emit("observe_complete", {
                "step": step,
                "element_count": len(observation["elements"]),
                "url": url,
            })

            return observation

        except Exception as e:
            latency_ms = measure_latency(start)
            # Log observation failure as telemetry
            log_step_event(make_telemetry(
                step=step,
                url=url,
                action_type="observe",
                action_target=None,
                thought=None,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            ))
            self._emit("observe_error", {"step": step, "error": str(e)})
            raise

    async def _reason(
        self,
        goal: str,
        observation: Observation,
        guardrail_warning: Optional[str] = None,
    ) -> Tuple[Action, Optional[str]]:
        """REASON: Call LLM to decide the next action."""
        self._emit("reason_start", {"step": self.state.current_step})

        step = self.state.current_step
        url = observation.get("url", "")
        start = time.perf_counter()

        try:
            user_prompt = self._build_user_prompt(goal, observation, self.state.current_step)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            if guardrail_warning:
                messages.append({
                    "role": "user",
                    "content": f"⚠️ GUARDRAIL WARNING: {guardrail_warning}\nChoose a DIFFERENT action."
                })

            # Call LLM
            response = await self.llm_client.chat(
                messages,
                response_format="json",
            )

            if not response.success:
                raise LLMError(
                    f"LLM call failed: {response.error}",
                    provider=response.provider,
                    model=response.model,
                )

            action = response.action
            latency_ms = measure_latency(start)

            # Log successful reason step with LLM latency
            log_step_event(make_telemetry(
                step=step,
                url=url,
                action_type="reason",
                action_target=None,
                thought=action.get("thought", ""),
                latency_ms=latency_ms,
                success=True,
            ))

            self._emit("reason_complete", {
                "step": step,
                "action": action.get("action", ""),
                "duration_ms": latency_ms,
                "tokens": response.tokens_used,
            })

            return action, guardrail_warning

        except Exception as e:
            latency_ms = measure_latency(start)
            # Structured error telemetry — never crash the process
            log_step_event(make_telemetry(
                step=step,
                url=url,
                action_type="reason",
                action_target=None,
                thought=None,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            ))
            self._emit("reason_error", {"step": step, "error": str(e)})
            raise

    async def _act(
        self,
        action: Action,
        observation: Observation,
    ) -> Tuple[bool, str]:
        """ACT: Execute the chosen action on the browser."""
        self._emit("act_start", {
            "step": self.state.current_step,
            "action": action.get("action", ""),
            "element_id": action.get("element_id", ""),
        })

        action_type = action.get("action", "")
        element_id = action.get("element_id", "")
        text = action.get("text", "")

        step = self.state.current_step
        url = observation.get("url", "")
        thought = action.get("thought", "")
        start = time.perf_counter()

        try:
            result_msg: str

            if action_type == "done":
                result_msg = action.get("final_answer", "Task complete.")
                success = True
            elif action_type == "click" and element_id:
                result_msg = await self.driver.click(element_id)
                if "Error" in result_msg:
                    raise AgentError(
                        f"Click failed: {result_msg}",
                        step_number=step,
                        action="click",
                        element_id=element_id,
                    )
                success = True
            elif action_type == "type" and element_id:
                press_enter = True  # Default to pressing enter after typing
                result_msg = await self.driver.type(element_id, text, press_enter=press_enter)
                if "Error" in result_msg:
                    raise AgentError(
                        f"Type failed: {result_msg}",
                        step_number=step,
                        action="type",
                        element_id=element_id,
                    )
                success = True
            elif action_type == "navigate" and text:
                result_msg = await self.driver.navigate(text)
                if "Error" in result_msg:
                    raise AgentError(
                        f"Navigate failed: {result_msg}",
                        step_number=step,
                        action="navigate",
                    )
                success = True
            else:
                raise AgentError(
                    f"Unknown or invalid action: {action_type}",
                    step_number=step,
                    action=action_type,
                )

            latency_ms = measure_latency(start)
            log_step_event(make_telemetry(
                step=step,
                url=url,
                action_type=action_type,
                action_target=element_id,
                thought=thought,
                latency_ms=latency_ms,
                success=True,
            ))
            return success, result_msg

        except Exception as e:
            latency_ms = measure_latency(start)
            error_msg = str(e)
            # Structured error telemetry — never crash the process
            log_step_event(make_telemetry(
                step=step,
                url=url,
                action_type=action_type,
                action_target=element_id,
                thought=thought,
                latency_ms=latency_ms,
                success=False,
                error=error_msg,
            ))
            self._emit("act_error", {"step": step, "error": error_msg})
            return False, error_msg

    async def _verify(
        self,
        action: Action,
        success: bool,
        observation: Observation,
    ) -> StepResult:
        """VERIFY: Record the step result and update state."""
        step = self.state.current_step

        step_result = StepResult(
            step_number=step,
            action=action,
            observation=observation,
            success=success,
        )

        self.state.steps.append(step_result)
        self._emit("step_complete", {
            "step": step,
            "action": action.get("action", ""),
            "success": success,
        })

        return step_result

    async def run(self, goal: str, start_url: str) -> str:
        """Run the ReAct decision loop.

        State machine: Observe → Reason → Act → Verify

        Args:
            goal: The user's goal to achieve
            start_url: The URL to start from

        Returns:
            The final answer or error message
        """
        run_start_time = time.perf_counter()
        self.state = LoopState(
            goal=goal,
            start_url=start_url,
            max_steps=self.max_steps,
        )
        self.state.status = LoopStatus.RUNNING
        self.state.started_at = datetime.now()

        # Emit telemetry for run start
        log_step_event(make_telemetry(
            step=0,
            url=start_url,
            action_type="run_start",
            action_target=None,
            thought=goal,
            latency_ms=0.0,
            success=True,
        ))

        self._emit("run_start", {"goal": goal, "start_url": start_url})

        try:
            # Navigate to start URL
            nav_result = await self.driver.navigate(start_url)
            if "Error" in nav_result:
                self.state.status = LoopStatus.FAILED
                # Log the navigation failure
                log_step_event(make_telemetry(
                    step=0,
                    url=start_url,
                    action_type="navigate",
                    action_target=start_url,
                    thought="Initial navigation",
                    latency_ms=measure_latency(run_start_time),
                    success=False,
                    error=nav_result,
                ))
                return f"Failed to start: {nav_result}"

            # Main loop
            for step in range(1, self.max_steps + 1):
                self.state.current_step = step

                # OBSERVE: Scan the page
                observation = await self._observe()

                # Guardrail: Validate element exists
                element_id = ""
                action: Action = {}
                guardrail_warning: Optional[str] = None

                # REASON: Get action from LLM
                action, guardrail_warning = await self._reason(goal, observation, guardrail_warning)

                element_id = action.get("element_id", "")
                action_type = action.get("action", "")

                # Guardrail: Check element validity
                if action_type in ("click", "type"):
                    el_warning = self.guardrail.check_element_valid(
                        element_id, observation.get("elements", [])
                    )
                    if el_warning:
                        # Re-reason with warning
                        action, guardrail_warning = await self._reason(
                            goal, observation, el_warning
                        )
                        element_id = action.get("element_id", "")
                        action_type = action.get("action", "")

                # Guardrail: Check for loop patterns
                loop_warning = self.guardrail.check(
                    action, element_id, observation.get("url", ""), self.state.history
                )
                if loop_warning:
                    # Re-reason with warning
                    action, guardrail_warning = await self._reason(
                        goal, observation, loop_warning
                    )
                    element_id = action.get("element_id", "")
                    action_type = action.get("action", "")

                # Guardrail: Validate navigation URL
                if action_type == "navigate":
                    nav_warning = self.guardrail.check_navigate_valid(action.get("text", ""))
                    if nav_warning:
                        action, guardrail_warning = await self._reason(
                            goal, observation, nav_warning
                        )
                        action_type = action.get("action", "")

                # Guardrail: Clear warning if action changed
                if action_type != "done":
                    self.guardrail.reset_element(element_id)

                # Record in history
                if action_type != "done":
                    self.state.history.append((
                        action_type,
                        element_id,
                        observation.get("url", ""),
                    ))
                    # Trim history
                    if len(self.state.history) > self.settings.loop.guardrail_max_history:
                        self.state.history = self.state.history[-self.settings.loop.guardrail_max_history:]

                # Check for done
                if action_type == "done":
                    final_answer = action.get("final_answer", "Task complete.")
                    self.state.status = LoopStatus.COMPLETED
                    self.state.completed_at = datetime.now()

                    # Log successful completion
                    log_step_event(make_telemetry(
                        step=step,
                        url=observation.get("url", ""),
                        action_type="done",
                        action_target=element_id,
                        thought=action.get("thought", ""),
                        latency_ms=measure_latency(run_start_time),
                        success=True,
                    ))

                    self._emit("run_complete", {
                        "status": "success",
                        "final_answer": final_answer,
                    })
                    return final_answer

                # ACT: Execute the action
                success, result = await self._act(action, observation)

                # VERIFY: Record step
                await self._verify(action, success, observation)

                if not success:
                    self.state.status = LoopStatus.FAILED
                    # Log action failure
                    log_step_event(make_telemetry(
                        step=step,
                        url=observation.get("url", ""),
                        action_type=action_type,
                        action_target=element_id,
                        thought=action.get("thought", ""),
                        latency_ms=measure_latency(run_start_time),
                        success=False,
                        error=result,
                    ))
                    self._emit("run_complete", {
                        "status": "failed",
                        "error": result,
                    })
                    return f"Action failed: {result}"

            # Max steps reached
            self.state.status = LoopStatus.COMPLETED
            self.state.completed_at = datetime.now()
            self._emit("run_complete", {
                "status": "max_steps",
                "url": observation.get("url", ""),
            })
            return (
                f"Max steps ({self.max_steps}) reached. Current URL: {observation.get('url', '')}. "
                f"Goal: {goal}. Partial completion."
            )

        except Exception as e:
            self.state.status = LoopStatus.FAILED
            self.state.completed_at = datetime.now()
            # Structured error telemetry for uncaught exceptions
            log_step_event(make_telemetry(
                step=self.state.current_step,
                url=start_url,
                action_type="run_error",
                action_target=None,
                thought=None,
                latency_ms=measure_latency(run_start_time),
                success=False,
                error=str(e),
            ))
            self._emit("run_complete", {"status": "error", "error": str(e)})
            return f"Agent error: {str(e)}"

        finally:
            # Log overall run latency in telemetry
            total_latency_ms = measure_latency(run_start_time)
            log_step_event(make_telemetry(
                step=self.state.current_step,
                url=self.state.start_url,
                action_type="run_complete",
                action_target=None,
                thought=f"Run finished: {self.state.status.value}",
                latency_ms=total_latency_ms,
                success=self.state.status == LoopStatus.COMPLETED,
            ))
            await self.llm_client.close()

    async def __aenter__(self) -> "AgentLoop":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.llm_client.close()