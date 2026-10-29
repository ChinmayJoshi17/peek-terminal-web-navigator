"""Streamlit UI Cockpit for nano-browser-agent.

Live dashboard for controlling agent runs and monitoring telemetry, plus an
Evaluations view for browsing benchmark harness results.

Reads structured JSONL logs from `logs/agent_run_*.jsonl` for live runs and
reads `eval/results_*.json` for benchmark history.

Run with:
    streamlit run ui/dashboard.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import threading
import pandas as pd
import streamlit as st
from streamlit.runtime import get_instance

# Resolve project root
_PROJECT_ROOT = Path(__file__).parent.parent
_LOGS_DIR = _PROJECT_ROOT / "logs"
_EVAL_DIR = _PROJECT_ROOT / "eval"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Nano Browser Agent — Cockpit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Background Browser Session Watchdog (Auto-shutdown on tab close)
# ---------------------------------------------------------------------------

@st.cache_resource
def start_browser_watchdog() -> None:
    """Launch a background daemon thread monitoring active browser client sessions.

    Gracefully terminates the Streamlit process when all browser tabs are closed.
    """
    def _monitor() -> None:
        # 12-second grace period at initial launch for browser to connect
        time.sleep(12.0)

        # Wait until at least one session connects
        has_connected = False
        while not has_connected:
            runtime = get_instance()
            if runtime and hasattr(runtime, "_session_mgr") and runtime._session_mgr:
                session_mgr = runtime._session_mgr
                count = (
                    session_mgr.num_active_sessions()
                    if hasattr(session_mgr, "num_active_sessions")
                    else len(session_mgr.list_active_sessions())
                    if hasattr(session_mgr, "list_active_sessions")
                    else len(session_mgr.list_sessions())
                    if hasattr(session_mgr, "list_sessions")
                    else 0
                )
                if count > 0:
                    has_connected = True
                    break
            time.sleep(1.5)

        # Continuously poll active sessions every 1.5 seconds
        while True:
            time.sleep(1.5)
            runtime = get_instance()
            if not runtime:
                continue

            session_mgr = getattr(runtime, "_session_mgr", None)
            if not session_mgr:
                continue

            count = (
                session_mgr.num_active_sessions()
                if hasattr(session_mgr, "num_active_sessions")
                else len(session_mgr.list_active_sessions())
                if hasattr(session_mgr, "list_active_sessions")
                else len(session_mgr.list_sessions())
                if hasattr(session_mgr, "list_sessions")
                else 0
            )

            # If all browser sessions are closed, shut down cleanly
            if count == 0:
                print("\n[Streamlit] Browser tab closed. Shutting down server...\n", flush=True)
                os._exit(0)

    thread = threading.Thread(target=_monitor, name="BrowserSessionWatchdog", daemon=True)
    thread.start()


# Ensure watchdog is initiated once per server session
start_browser_watchdog()



# ---------------------------------------------------------------------------
# Helpers — Live telemetry
# ---------------------------------------------------------------------------

def _latest_log_file() -> Optional[Path]:
    """Return the most recent `agent_run_*.jsonl` log file, or None."""
    if not _LOGS_DIR.exists():
        return None
    files = sorted(_LOGS_DIR.glob("agent_run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of parsed records (best-effort)."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return records


def _step_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics for the most recent records."""
    if not records:
        return {
            "step": 0,
            "last_action": "—",
            "last_target": "—",
            "last_thought": "",
            "last_latency_ms": 0.0,
            "last_success": None,
            "last_error": None,
            "last_url": "",
            "total_steps": 0,
            "total_errors": 0,
            "avg_latency_ms": 0.0,
        }

    # Filter to actionable steps (those with non-zero step numbers and not run_complete)
    real_steps = [
        r for r in records
        if r.get("step", 0) > 0
        and r.get("action_type") not in ("run_complete", "run_start", "run_error")
    ]

    if real_steps:
        last = real_steps[-1]
    else:
        last = records[-1]

    # Latencies (only on success records for cleanliness)
    success_latencies = [
        r.get("latency_ms", 0.0)
        for r in records
        if r.get("success") and r.get("latency_ms", 0) > 0
    ]
    avg_latency = (sum(success_latencies) / len(success_latencies)) if success_latencies else 0.0

    return {
        "step": last.get("step", 0),
        "last_action": last.get("action_type", "—"),
        "last_target": last.get("action_target") or "—",
        "last_thought": last.get("thought") or "",
        "last_latency_ms": last.get("latency_ms", 0.0),
        "last_success": last.get("success"),
        "last_error": last.get("error"),
        "last_url": last.get("url", ""),
        "total_steps": len(real_steps),
        "total_errors": sum(1 for r in records if r.get("error")),
        "avg_latency_ms": avg_latency,
    }


def _errors(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return records that contain errors or guardrail warnings."""
    out: List[Dict[str, Any]] = []
    for r in records:
        if r.get("error") or r.get("action_type") in (
            "act_error", "observe_error", "reason_error", "run_error"
        ):
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Helpers — Eval / benchmark results
# ---------------------------------------------------------------------------

def _list_eval_runs() -> List[Path]:
    """Return a sorted (newest first) list of `eval/results_*.json` files."""
    if not _EVAL_DIR.exists():
        return []
    return sorted(
        _EVAL_DIR.glob("results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_eval_run(path: Path) -> Optional[Dict[str, Any]]:
    """Read and parse one eval results JSON file. Returns None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Sidebar — View selector & configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🧭 Navigation")
    view = st.radio(
        "Select View",
        options=["Live Agent Cockpit", "Evaluation Harness Results"],
        index=0,
        help="Switch between the interactive live cockpit and benchmark evaluations.",
    )

    st.divider()

    if view == "Live Agent Cockpit":
        st.subheader("⚙️ Live Settings")
        refresh_interval = st.slider("Auto-refresh rate (s)", min_value=1, max_value=10, value=2, step=1)
        auto_refresh = st.checkbox("Enable auto-refresh", value=False)
        st.caption("Telemetry stream reads from `logs/agent_run_*.jsonl`.")
    else:
        st.subheader("📂 Benchmark Archive")
        eval_runs = _list_eval_runs()
        if not eval_runs:
            st.info("No benchmark runs found in `eval/`.")
        else:
            selected_run = st.selectbox(
                "Select Benchmark Run",
                options=eval_runs,
                index=0,
                format_func=lambda p: f"{p.name} ({p.stat().st_size} bytes)",
                help="Historical benchmark runs, sorted newest first.",
            )
            st.session_state["selected_eval_run"] = selected_run

        st.caption("Re-run benchmarks with:\n`python eval/runner.py --headless`")


# ---------------------------------------------------------------------------
# Main panel — Live Agent Cockpit
# ---------------------------------------------------------------------------

if view == "Live Agent Cockpit":
    st.title("🤖 Nano Browser Agent — Live Cockpit")
    st.caption("Autonomous browser agent execution console, step telemetry, and live response card.")

    # -----------------------------------------------------------------------
    # Section 1: Goal Input & Execution Controls
    # -----------------------------------------------------------------------
    with st.container():
        st.markdown("### 🎯 Goal & Execution Controls")

        goal_input = st.text_area(
            "Agent Goal",
            value="Go to wikipedia.org, search for Alan Turing, and report what field he is considered the father of.",
            height=85,
            placeholder="Describe what the browser agent should accomplish...",
            help="High-level goal for the agent ReAct loop.",
        )

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([3, 1.5, 1.5, 2])
        with ctrl_col1:
            start_url_input = st.text_input(
                "Start URL",
                value="https://en.wikipedia.org/wiki/Main_Page",
                help="Initial page URL for agent navigation.",
            )
        with ctrl_col2:
            headless_input = st.toggle(
                "Headless Mode",
                value=True,
                help="Run Chromium in headless mode without opening a visible window.",
            )
        with ctrl_col3:
            max_steps_input = st.number_input(
                "Max Steps",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Iteration budget for this run.",
            )
        with ctrl_col4:
            model_input = st.text_input(
                "LLM Model",
                value="openai/gpt-oss-120b",
                help="Inference model identifier.",
            )

        run_btn = st.button("🚀 Run Agent", type="primary", use_container_width=True)

    # Execute agent subprocess on run click
    if run_btn:
        if not goal_input.strip():
            st.error("Please enter a valid goal before starting the agent.")
        else:
            cmd = [
                sys.executable,
                "main.py",
                "--prompt", goal_input,
                "--url", start_url_input,
                "--max-steps", str(int(max_steps_input)),
                "--model", model_input,
                "--no-rich",
            ]
            if headless_input:
                cmd.append("--headless")

            with st.spinner("Agent is operating in the browser..."):
                start_time = time.perf_counter()
                try:
                    proc_result = subprocess.run(
                        cmd,
                        cwd=str(_PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    elapsed = time.perf_counter() - start_time
                    st.session_state["last_run_result"] = {
                        "returncode": proc_result.returncode,
                        "stdout": proc_result.stdout,
                        "stderr": proc_result.stderr,
                        "elapsed": elapsed,
                        "goal": goal_input,
                    }
                except subprocess.TimeoutExpired:
                    st.session_state["last_run_result"] = {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": "Execution timed out after 300 seconds.",
                        "elapsed": 300.0,
                        "goal": goal_input,
                    }
                except Exception as ex:
                    st.session_state["last_run_result"] = {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": str(ex),
                        "elapsed": 0.0,
                        "goal": goal_input,
                    }

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2: Final Response Card
    # -----------------------------------------------------------------------
    st.markdown("### 🏁 Final Response Card")
    last_run = st.session_state.get("last_run_result")

    if last_run:
        ret_code = last_run["returncode"]
        stdout_txt = last_run.get("stdout", "")
        stderr_txt = last_run.get("stderr", "")
        elapsed_sec = last_run.get("elapsed", 0.0)

        # Extract final answer line if present in stdout
        answer_match = re.search(r"Final Answer:\s*(.+)", stdout_txt, re.IGNORECASE)
        final_answer = answer_match.group(1).strip() if answer_match else ""

        if not final_answer and stdout_txt:
            # Fallback to last meaningful line
            lines = [l.strip() for l in stdout_txt.splitlines() if l.strip()]
            if lines:
                final_answer = lines[-1]

        if ret_code == 0:
            st.success(f"**Execution Completed in {elapsed_sec:.2f}s**")
            st.markdown(f"**Outcome / Answer:**\n\n> {final_answer or 'Task completed successfully.'}")
        else:
            st.error(f"**Execution Finished with Error Code {ret_code}** ({elapsed_sec:.2f}s)")
            if stderr_txt:
                st.code(stderr_txt, language="text")

        with st.expander("Full Console Output", expanded=False):
            if stdout_txt:
                st.text(stdout_txt)
            if stderr_txt:
                st.caption("Stderr:")
                st.text(stderr_txt)
    else:
        st.info("No active agent run output yet. Click **'Run Agent'** above to launch an autonomous session.")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3: Live Telemetry & Step Metrics
    # -----------------------------------------------------------------------
    st.markdown("### 📊 Live Telemetry & Step Metrics")

    log_file = _latest_log_file()
    records = _read_jsonl(log_file) if log_file else []
    metrics = _step_metrics(records)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current Step", metrics["step"])
        st.metric("Total Events", metrics["total_steps"])
    with m2:
        st.metric("Last Action", metrics["last_action"])
        st.metric("Action Target", str(metrics["last_target"]))
    with m3:
        st.metric("Action Latency", f"{metrics['last_latency_ms']:.1f} ms")
        st.metric("Avg Latency", f"{metrics['avg_latency_ms']:.1f} ms")
    with m4:
        status_label = "✅ Pass" if metrics["last_success"] else ("❌ Fail" if metrics["last_success"] is False else "⏸️ Idle")
        st.metric("Status", status_label)
        st.metric("Recorded Errors", metrics["total_errors"])

    if metrics.get("last_thought"):
        st.markdown(f"**Latest Agent Reasoning:**\n> *{metrics['last_thought']}*")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 4: Live Logs & Error Feed
    # -----------------------------------------------------------------------
    st.markdown("### 📜 Telemetry Logs & Stream")

    if log_file:
        st.caption(f"Active Log: `{log_file.name}`  •  Size: {log_file.stat().st_size} bytes")
    else:
        st.caption("No log files recorded yet.")

    err_list = _errors(records)
    if err_list:
        with st.expander(f"⚠️ Errors & Warnings ({len(err_list)})", expanded=True):
            for err_rec in err_list[-5:][::-1]:
                st.warning(
                    f"Step {err_rec.get('step', '?')} [{err_rec.get('action_type', '?')}]: "
                    f"{err_rec.get('error', 'Unknown error')}"
                )

    with st.expander("Recent Telemetry Events (Last 20)", expanded=False):
        if records:
            df_records = pd.DataFrame(records[-20:])
            st.dataframe(df_records, use_container_width=True, height=280)
        else:
            st.caption("No telemetry events recorded.")

    # Auto-refresh rerun trigger
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ---------------------------------------------------------------------------
# Main panel — Evaluation Harness Results
# ---------------------------------------------------------------------------

else:
    st.title("📊 Evaluation Harness Results")
    st.caption("Structured benchmark metrics across WebArena-style mini evaluation suites.")

    selected_run: Optional[Path] = st.session_state.get("selected_eval_run")
    if selected_run is None or not selected_run.exists():
        st.info("No benchmark run selected. Select a run from the sidebar archive.")
        st.stop()

    data = _load_eval_run(selected_run)
    if data is None:
        st.error(f"Failed to parse {selected_run.name}.")
        st.stop()

    metadata = data.get("metadata", {}) or {}
    summary = data.get("summary", {}) or {}
    results = data.get("results", []) or []

    st.caption(
        f"Run File: `{selected_run.name}`  •  "
        f"Timestamp: `{metadata.get('run_at', '—')}`"
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Structured Benchmark Metric Grid (Task 4 mandate)
    # -----------------------------------------------------------------------
    st.markdown("### 🧮 Benchmark Metric Summary")

    total_tasks = int(summary.get("total_tasks", len(results)))
    pass_rate = float(summary.get("success_rate_percent", 0.0))
    avg_latency = float(summary.get("mean_latency_ms", 0.0))
    total_steps = sum(int(r.get("steps_taken", 0)) for r in results)

    grid_col1, grid_col2, grid_col3, grid_col4 = st.columns(4)
    with grid_col1:
        st.metric("Total Tasks", total_tasks)
    with grid_col2:
        st.metric("Pass Rate %", f"{pass_rate:.1f}%")
    with grid_col3:
        st.metric("Average Latency ms", f"{avg_latency:.1f} ms")
    with grid_col4:
        st.metric("Total Steps", total_steps)

    st.divider()

    # -----------------------------------------------------------------------
    # Task Results Table
    # -----------------------------------------------------------------------
    st.markdown("### 📋 Task Results Grid")

    if not results:
        st.info("This run contains no individual task results.")
        st.stop()

    table_rows: List[Dict[str, Any]] = []
    for r in results:
        status_str = "✅ PASS" if r.get("success") else "❌ FAIL"
        table_rows.append({
            "Task ID": r.get("task_id", "—"),
            "Task Name": r.get("name", "—"),
            "Status": status_str,
            "Steps": r.get("steps_taken", 0),
            "Duration (s)": round(float(r.get("duration_seconds", 0.0)), 2),
            "Avg Latency (ms)": round(float(r.get("avg_action_latency_ms", 0.0)), 1),
            "Failure Reason": r.get("failure_reason") or "—",
        })

    df = pd.DataFrame(table_rows)
    st.dataframe(
        df,
        use_container_width=True,
        height=min(450, 60 + 38 * len(df)),
        column_config={
            "Duration (s)": st.column_config.NumberColumn(format="%.2f"),
            "Avg Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Per-Task Drilldown
    # -----------------------------------------------------------------------
    st.markdown("### 🔍 Per-Task Drilldown")

    for i, r in enumerate(results):
        t_id = r.get("task_id", f"task_{i}")
        name = r.get("name", "—")
        success = bool(r.get("success"))
        header = f"{'✅' if success else '❌'} [{t_id}] {name}"

        with st.expander(header, expanded=not success):
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**Goal:**")
                st.write(r.get("goal", "—"))
                st.markdown("**Start URL:**")
                st.code(r.get("start_url", "—"), language="text")
            with c_right:
                st.markdown("**Agent Final Answer:**")
                st.write(r.get("final_answer") or "—")
                st.caption(
                    f"Steps: **{r.get('steps_taken', 0)}**  •  "
                    f"Duration: **{r.get('duration_seconds', 0.0):.2f}s**  •  "
                    f"Latency: **{r.get('avg_action_latency_ms', 0.0):.1f}ms**"
                )

            if not success and r.get("failure_reason"):
                st.error(f"Failure Reason: {r.get('failure_reason')}")
