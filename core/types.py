"""Strict TypedDict/Pydantic schemas for Action, Observation, StepResult, and DOMElement."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from pydantic import BaseModel, Field, ConfigDict


class ActionType(str, Enum):
    """Supported action types for the agent."""

    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    DONE = "done"
    SCROLL = "scroll"
    WAIT = "wait"


class DOMElement(TypedDict, total=False):
    """TypedDict for DOM element metadata extracted by dom_pruner.js."""

    id: str
    tag: str
    text: str
    type: str
    placeholder: str
    href: str
    role: str
    aria_label: str
    is_visible: bool
    bounding_box: Dict[str, float]  # x, y, width, height


class Action(TypedDict, total=False):
    """TypedDict for LLM action response."""

    thought: str
    action: Literal["click", "type", "navigate", "done", "scroll", "wait"]
    element_id: str
    text: str
    final_answer: str
    scroll_direction: Literal["up", "down", "left", "right"]
    wait_ms: int


class Observation(TypedDict, total=False):
    """TypedDict for page observation returned by browser scan."""

    elements: List[DOMElement]
    url: str
    title: str
    screenshot_b64: Optional[str]
    viewport_height: int
    viewport_width: int
    scroll_y: int
    page_load_complete: bool
    network_idle: bool


@dataclass
class StepResult:
    """Result of a single agent loop step."""

    step_number: int
    action: Action
    observation: Observation
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    guardrail_warning: Optional[str] = None
    score: Optional[float] = None


class LoopStatus(str, Enum):
    """Current state of the agent loop."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    CHECKPOINT_SAVED = "checkpoint_saved"


@dataclass
class LoopState:
    """Complete state of the agent loop for checkpointing."""

    goal: str
    start_url: str
    status: LoopStatus = LoopStatus.IDLE
    current_step: int = 0
    max_steps: int = 15
    steps: List[StepResult] = field(default_factory=list)
    history: List[tuple] = field(default_factory=list)  # (action, element_id, url)
    checkpoint_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_tokens_used: int = 0

    @property
    def total_duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    @property
    def avg_step_duration_ms(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.duration_ms for s in self.steps) / len(self.steps)

    @property
    def success_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.success) / len(self.steps)


# Pydantic models for API boundaries and serialization
class ActionModel(BaseModel):
    """Pydantic model for Action validation and serialization."""

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(..., min_length=1, description="Reasoning for the action")
    action: ActionType = Field(..., description="Action to perform")
    element_id: str = Field(default="", description="Target element identifier")
    text: str = Field(default="", description="Text input for type/navigate")
    final_answer: str = Field(default="", description="Final answer when done")
    scroll_direction: Literal["up", "down", "left", "right"] = "down"
    wait_ms: int = Field(default=0, ge=0, le=60000)

    def to_action(self) -> Action:
        """Convert to Action TypedDict."""
        return Action(
            thought=self.thought,
            action=self.action.value,
            element_id=self.element_id,
            text=self.text,
            final_answer=self.final_answer,
            scroll_direction=self.scroll_direction,
            wait_ms=self.wait_ms,
        )


class DOMElementModel(BaseModel):
    """Pydantic model for DOMElement validation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    tag: str
    text: str = ""
    type: str = ""
    placeholder: str = ""
    href: str = ""
    role: str = ""
    aria_label: str = ""
    is_visible: bool = True
    bounding_box: Dict[str, float] = Field(default_factory=dict)


class ObservationModel(BaseModel):
    """Pydantic model for Observation validation."""

    model_config = ConfigDict(extra="ignore")

    elements: List[DOMElementModel] = Field(default_factory=list)
    url: str = ""
    title: str = ""
    screenshot_b64: Optional[str] = None
    viewport_height: int = 800
    viewport_width: int = 1280
    scroll_y: int = 0
    page_load_complete: bool = False
    network_idle: bool = False

    def to_observation(self) -> Observation:
        """Convert to Observation TypedDict."""
        return Observation(
            elements=[e.model_dump() for e in self.elements],
            url=self.url,
            title=self.title,
            screenshot_b64=self.screenshot_b64,
            viewport_height=self.viewport_height,
            viewport_width=self.viewport_width,
            scroll_y=self.scroll_y,
            page_load_complete=self.page_load_complete,
            network_idle=self.network_idle,
        )


class StepResultModel(BaseModel):
    """Pydantic model for StepResult serialization."""

    model_config = ConfigDict(extra="ignore")

    step_number: int
    action: ActionModel
    observation: ObservationModel
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    guardrail_warning: Optional[str] = None
    score: Optional[float] = None


class LoopStateModel(BaseModel):
    """Pydantic model for LoopState serialization."""

    model_config = ConfigDict(extra="ignore")

    goal: str
    start_url: str
    status: LoopStatus = LoopStatus.IDLE
    current_step: int = 0
    max_steps: int = 15
    steps: List[StepResultModel] = Field(default_factory=list)
    history: List[Dict[str, str]] = Field(default_factory=list)
    checkpoint_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_tokens_used: int = 0

    def to_loop_state(self) -> LoopState:
        """Convert to LoopState dataclass."""
        steps = [StepResult(
            step_number=s.step_number,
            action=s.action.to_action(),
            observation=s.observation.to_observation(),
            success=s.success,
            error=s.error,
            duration_ms=s.duration_ms,
            timestamp=s.timestamp,
            guardrail_warning=s.guardrail_warning,
            score=s.score,
        ) for s in self.steps]

        history = [(h.get("action", ""), h.get("element_id", ""), h.get("url", "")) for h in self.history]

        return LoopState(
            goal=self.goal,
            start_url=self.start_url,
            status=self.status,
            current_step=self.current_step,
            max_steps=self.max_steps,
            steps=steps,
            history=history,
            checkpoint_id=self.checkpoint_id,
            started_at=self.started_at,
            completed_at=self.completed_at,
            total_tokens_used=self.total_tokens_used,
        )