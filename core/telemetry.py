"""Structured telemetry and logging for nano-browser-agent.

Uses Loguru for colorized console output and append-only JSONL persistence
for structured step telemetry.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Loguru setup (console only; JSONL is written directly for full control)
# ---------------------------------------------------------------------------

from loguru import logger

# Resolve project root relative to this package
_PROJECT_ROOT = Path(__file__).parent.parent
_LOGS_DIR = _PROJECT_ROOT / "logs"

# Ensure logs directory exists
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- JSONL file ---
# One log file per agent run, named agent_run_<timestamp>.jsonl
_timestamp_for_filename = datetime.now().strftime("%Y-%m-%dT%H%M%S")
_JSONL_FILE = _LOGS_DIR / f"agent_run_{_timestamp_for_filename}.jsonl"

# --- configure logger -------------------------------------------------------

# Remove default handler
logger.remove()

# Colorized console format
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Console handler (colorized)
logger.add(
    sink=lambda m: print(m),
    format=CONSOLE_FORMAT,
    colorize=True,
    level="INFO",
)


# ---------------------------------------------------------------------------
# Pydantic model for step telemetry
# ---------------------------------------------------------------------------

class StepTelemetry(BaseModel):
    """Telemetry record for a single agent loop step.

    Attributes:
        timestamp: ISO 8601 UTC timestamp of the step
        step: Sequential step number in the current run (0 for run-level events)
        url: Current page URL
        action_type: Type of action taken (e.g., click, type, navigate, done,
            observe, reason, run_start, run_complete)
        action_target: Optional target element ID or URL
        thought: Optional reasoning/thought from the LLM
        latency_ms: Execution latency in milliseconds
        success: Whether the action succeeded
        error: Optional error message if the step failed
    """

    model_config = {"use_enum_values": True, "coerce_numbers_to_str": False}

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of the step",
    )
    step: int = Field(
        ge=0,
        description="Sequential step number in the current run (0 for run-level)",
    )
    url: str = Field(
        min_length=1,
        description="Current page URL",
    )
    action_type: str = Field(
        min_length=1,
        description="Type of action taken",
    )
    action_target: Optional[str] = Field(
        default=None,
        description="Target element ID or URL",
    )
    thought: Optional[str] = Field(
        default=None,
        description="LLM reasoning/thought for this step",
    )
    latency_ms: float = Field(
        ge=0,
        description="Execution latency in milliseconds",
    )
    success: bool = Field(
        description="Whether the action succeeded",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if step failed",
    )


# ---------------------------------------------------------------------------
# Telemetry logging utility
# ---------------------------------------------------------------------------

_step_counter: int = 0


def log_step_event(event: StepTelemetry) -> None:
    """Log a structured step telemetry event.

    Writes a single-line JSON record to the active JSONL log file and emits
    a colorized Loguru message to the console.

    Args:
        event: A StepTelemetry instance containing all step metadata.
    """
    global _step_counter
    _step_counter += 1

    # Serialize to compact JSON (one line)
    line = event.model_dump_json(
        include={"timestamp", "step", "url", "action_type", "action_target", "thought", "latency_ms", "success", "error"}
    )

    # Write to JSONL file directly (defensive — telemetry must never crash agent)
    try:
        with open(_JSONL_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass  # Best-effort: never crash the agent

    # Emit to Loguru (visible, structured)
    if event.error:
        logger.warning(
            f"STEP {event.step} | {event.action_type} | "
            f"success={event.success} | latency={event.latency_ms:.1f}ms | error={event.error}"
        )
    else:
        logger.info(
            f"STEP {event.step} | {event.action_type} | "
            f"success={event.success} | latency={event.latency_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Convenience wrappers for the agent loop
# ---------------------------------------------------------------------------

def measure_latency(start: float) -> float:
    """Return elapsed time in milliseconds since *start* (perf_counter epoch)."""
    return (time.perf_counter() - start) * 1000


def make_telemetry(
    *,
    step: int,
    url: str,
    action_type: str,
    action_target: Optional[str] = None,
    thought: Optional[str] = None,
    latency_ms: float,
    success: bool,
    error: Optional[str] = None,
) -> StepTelemetry:
    """Build a StepTelemetry instance from individual components.

    Convenience factory so callers don't manually construct the Pydantic model
    each time.
    """
    return StepTelemetry(
        step=step,
        url=url,
        action_type=action_type,
        action_target=action_target,
        thought=thought,
        latency_ms=latency_ms,
        success=success,
        error=error,
    )


# ---------------------------------------------------------------------------
# Module-level sanity check (import-time only)
# ---------------------------------------------------------------------------

# Ensure the JSONL file path is initialized
try:
    _JSONL_FILE.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
