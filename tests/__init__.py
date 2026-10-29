"""Smoke tests for nano-browser-agent modular architecture.

Importers/callers: Test runner (pytest or direct invocation)
External consumers: None (test-only)
Data schemas: None (test fixtures only)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loading():
    """Test that Pydantic settings load correctly."""
    from core.config import Settings, LLMSettings, BrowserSettings, LoopSettings

    settings = Settings.load()
    assert settings.llm is not None
    assert settings.browser is not None
    assert settings.loop is not None
    assert settings.llm.model == "openai/gpt-oss-120b"
    assert settings.loop.max_steps == 15
    print("[OK] Configuration loads correctly")


def test_type_schemas():
    """Test that all type schemas can be constructed."""
    from core.types import (
        Action,
        Observation,
        StepResult,
        ActionModel,
        DOMElementModel,
        ObservationModel,
    )
    from datetime import datetime

    # Test Action TypedDict
    action: Action = {
        "thought": "Click search button",
        "action": "click",
        "element_id": "i5",
    }
    assert action["action"] == "click"

    # Test Pydantic models
    am = ActionModel(thought="test", action="click")
    assert am.thought == "test"
    assert am.action.value == "click"

    el = DOMElementModel(id="i1", tag="button", text="Submit")
    assert el.id == "i1"

    obs = ObservationModel(elements=[el], url="https://example.com", title="Example")
    assert len(obs.elements) == 1

    sr = StepResult(
        step_number=1,
        action=action,
        observation=obs.to_observation(),
        success=True,
    )
    assert sr.step_number == 1
    print("[OK] Type schemas validate correctly")


def test_error_hierarchy():
    """Test that error hierarchy is properly defined."""
    from core.errors import (
        AgentError,
        BrowserError,
        LLMError,
        ConfigurationError,
        GuardrailError,
        DOMExtractionError,
        ValidationError,
    )

    # Test base AgentError
    err = AgentError("test", step_number=5, action="click")
    assert "test" in str(err)
    assert "step=5" in str(err)

    # Test LLMError
    llm_err = LLMError("API failed", provider="groq", model="test-model", status_code=500)
    assert llm_err.provider == "groq"
    assert llm_err.status_code == 500

    # Test BrowserError
    browser_err = BrowserError("Element not found", element_id="i5")
    assert browser_err.element_id == "i5"

    # Test ValidationError
    val_err = ValidationError("Invalid", errors=["missing field"])
    assert "validation errors" in str(val_err)

    print("[OK] Error hierarchy works correctly")


def test_guardrail_logic():
    """Test guardrail prevents repeated actions."""
    from agent.guardrails import Guardrail, GuardrailConfig
    from core.types import Action

    guardrail = Guardrail(GuardrailConfig(max_same_element_retries=2, max_history=3))

    # Test first action is allowed
    action1: Action = {"action": "click", "element_id": "i1"}
    warning = guardrail.check(action1, "i1", "https://example.com", [])
    assert warning is None, "First action should be allowed"

    # Test repetition
    warning = guardrail.check(action1, "i1", "https://example.com", [])
    assert warning is None, "Second action should still be allowed"

    # Third attempt should be blocked
    warning = guardrail.check(action1, "i1", "https://example.com", [])
    assert warning is not None, "Third action should be blocked"
    assert "BLOCKED" in warning or "ALERT" in warning, f"Expected guardrail warning, got: {warning}"

    # Test element validation
    warning = guardrail.check_element_valid("i99", [{"id": "i1"}, {"id": "i2"}])
    assert warning is not None
    assert "not found" in warning

    # Test URL validation
    warning = guardrail.check_navigate_valid("not-a-url")
    assert warning is not None

    warning = guardrail.check_navigate_valid("https://example.com")
    assert warning is None

    print("[OK] Guardrail logic works correctly")


def test_module_imports():
    """Test that all modules can be imported without errors."""
    import core
    from core import config, types, errors

    import agent
    from agent import loop as agent_loop, guardrails
    from agent.providers import base, groq

    import browser
    from browser import controller

    import ui
    from ui import console

    print("[OK] All modules import successfully")


def test_rich_console():
    """Test Rich console UI is properly initialized."""
    from ui.console import ConsoleUI, get_console
    from rich.console import Console

    console = get_console()
    assert isinstance(console, Console)

    ui = ConsoleUI(console)
    assert ui.console is not None
    print("[OK] Rich console UI works")


def test_action_typed_dict_to_observation():
    """Test ActionModel serialization round-trip."""
    from core.types import ActionModel, ObservationModel, DOMElementModel

    am = ActionModel(thought="Reasoning", action="click", element_id="i3", text="")
    action = am.to_action()
    assert action["action"] == "click"
    assert action["element_id"] == "i3"
    assert action["thought"] == "Reasoning"

    el = DOMElementModel(id="i3", tag="button", text="Search")
    obs = ObservationModel(elements=[el], url="https://google.com", title="Google")
    obs_dict = obs.to_observation()
    assert obs_dict["url"] == "https://google.com"
    assert len(obs_dict["elements"]) == 1
    assert obs_dict["elements"][0]["id"] == "i3"
    print("[OK] Type serialization round-trips work")


def test_loop_state():
    """Test LoopState properties."""
    from core.types import LoopState, LoopStatus
    from datetime import datetime, timedelta

    state = LoopState(goal="Find AI news", start_url="https://example.com")
    assert state.status == LoopStatus.IDLE
    assert state.max_steps == 15
    assert state.success_rate == 0.0
    assert state.avg_step_duration_ms == 0.0

    state.started_at = datetime.now()
    state.completed_at = datetime.now() + timedelta(seconds=2)
    assert abs(state.total_duration_ms - 2000) < 50  # ~2s

    print("[OK] LoopState works correctly")


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running nano-browser-agent smoke tests")
    print("=" * 60)
    print()

    tests = [
        test_module_imports,
        test_config_loading,
        test_type_schemas,
        test_error_hierarchy,
        test_guardrail_logic,
        test_rich_console,
        test_action_typed_dict_to_observation,
        test_loop_state,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)