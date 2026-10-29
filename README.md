# 🔍 PEEK · Terminal Web Navigator

> A lightweight, resilient web navigation runtime built from scratch with Playwright, Groq-accelerated LLM reasoning, structured telemetry, and a deterministic benchmark evaluation harness.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Playwright](https://img.shields.io/badge/Playwright-Chromium-45ba4b.svg)](https://playwright.dev/python/)
[![Groq](https://img.shields.io/badge/Groq-Inference_LPU-f55036.svg)](https://groq.com/)
[![Benchmark](https://img.shields.io/badge/Evaluation-6%2F6_PASS_(100%25)-brightgreen.svg)](eval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏛️ Architectural Highlights

- **Zero-Framework Core**: Modular ReAct decision loop (`Observe` → `Reason` → `Act` → `Verify`) built from first principles with zero heavyweight agent framework dependencies.
- **Resilient Selector Fallbacks**: Multi-tier cascade (`combobox`/`searchbox` ARIA roles, label/placeholder matching, and explicit click/focus triggers) preventing actionability timeouts on dynamic SPAs and modern web applications.
- **DOM Token Pruning (`browser/observer.py`)**: Strips non-interactive wrappers (`div`, `span`, `section`, empty links, and citation clutter) to keep serialized observation strings compact (~3,500 chars), preventing token overflows and provider rate limits.
- **Deterministic Telemetry**: Append-only structured JSONL event logging (`logs/agent_run_*.jsonl`) capturing step-level latencies, element actions, and reasoning state.
- **Session-Aware Streamlit Cockpit**: Integrated live dashboard with an automatic background daemon watchdog that monitors client sessions and cleanly terminates the server process upon browser tab closure.

---

## 📊 Benchmark Performance

PEEK is validated against a deterministic WebArena-style mini evaluation harness testing real-world multi-step tasks across Wikipedia, E-commerce, and domain verification.

| Metric | Benchmark Result |
| :--- | :--- |
| **Evaluation Tasks** | **6 / 6 (100% PASS)** |
| **Mean Step Latency** | **~2.39s** |
| **Mean Steps per Task** | **1.33** |
| **Core Benchmark Suite** | Wikipedia Factual Lookup, Category Navigation, Dynamic Search, Multi-step Extraction |

### Task Breakdown

| Task ID | Task Description | Status | Steps | Avg Latency |
| :--- | :--- | :---: | :---: | :---: |
| `wikipedia_main_page` | Wikipedia: read main page & extract title | **PASS** | 1 | ~1.55s |
| `wikipedia_python_lang` | Wikipedia: identify creator of Python | **PASS** | 1 | ~1.69s |
| `books_toscrape_home` | Books to Scrape: count categories | **PASS** | 1 | ~2.44s |
| `example_domain_check` | Example.com: extract domain purpose | **PASS** | 1 | ~2.32s |
| `books_toscrape_category_navigation` | Books to Scrape: navigate Travel category | **PASS** | 2 | ~2.63s |
| `wikipedia_search_multi_step` | Wikipedia: dynamic search for Alan Turing | **PASS** | 2 | ~3.72s |

---

## ⚡ Quickstart Guide

### Prerequisites
- **Python 3.10+**
- **Groq API Key** (Free tier available at [console.groq.com](https://console.groq.com/keys))

### Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/peek-web-navigator.git
cd peek-web-navigator

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Windows CMD:
.\.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Playwright browser engine
playwright install chromium
```

### Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

*(See `.env.example` for reference)*

---

## 🚀 How to Run

### 1. One-Click Windows Launcher (`run.bat`)
Launch the interactive batch menu:
```cmd
.\run.bat
```
Provides quick access to the interactive terminal CLI, visual web cockpit, and headless evaluation suite.

### 2. Interactive Terminal CLI (`cli.py`)
Launch the PEEK terminal shell:
```bash
python cli.py
```
```text
╭──────────────────────────────────────────────────────────────╮
│  🔍 PEEK · Terminal Web Navigator  ·  v1.0                    │
╰──────────────────────────────────────────────────────────────╯

Type your query or 'exit' / 'quit' to terminate.

peek > Who created the Python programming language?
  ● [1/10] Navigating to https://www.google.com...
  ● [1/10] Locating target elements & reasoning next action...

┌── Result ────────────────────────────────────────────────────┐
│  Python was created by Guido van Rossum.                     │
└──────────────────────────────────────────────────────────────┘
⏱  4.2s  ·  1 steps  ·  Status: PASS
```

### 3. One-Shot Query Command
```bash
# Headless mode (fast)
python cli.py --goal "Who founded Wikipedia and when?" -u https://en.wikipedia.org

# Headed mode (visible browser window)
python cli.py --goal "Search for Alan Turing" -u https://en.wikipedia.org --headed
```

### 4. Visual Web Cockpit (Streamlit Dashboard)
```bash
python -m streamlit run ui/dashboard.py
```
- **Live Cockpit**: Run real-time tasks with live step metrics, thoughts, and an automatic response card.
- **Evaluation Harness Browser**: Inspect historical benchmark runs with structured metric grids.
- **Auto-Shutdown**: Background watchdog cleanly terminates the server process when you close the browser tab.

### 5. Benchmark Test Harness
```bash
python eval/runner.py --headless
```
Executes all 6 tasks in headless mode, reporting accuracy, steps taken, and action latencies. Results are saved to `eval/results_<timestamp>.json`.

---

## 📁 Project Structure

```text
nano-browser-agent/
├── agent/
│   ├── loop.py                 # Core ReAct decision engine (Observe-Reason-Act)
│   ├── guardrails.py           # Loop protection & element validation
│   └── providers/
│       ├── base.py             # Abstract LLM provider interface
│       └── groq.py             # Groq OpenAI-compatible client with backoff
├── browser/
│   ├── controller.py           # Playwright controller with resilient fallbacks
│   ├── observer.py             # DOM tree token pruner & serializer
│   └── dom_pruner.js           # Injected client-side element extractor
├── core/
│   ├── config.py               # Pydantic v2 application configuration
│   ├── telemetry.py            # Append-only JSONL telemetry recorder
│   ├── errors.py               # Structured domain exception hierarchy
│   └── types.py                # TypedDict & Pydantic state schemas
├── eval/
│   ├── benchmark_tasks.json    # Standardized evaluation task definitions
│   └── runner.py               # WebArena-style headless benchmark harness
├── ui/
│   ├── dashboard.py            # Streamlit cockpit with auto-shutdown watchdog
│   └── console.py              # Rich terminal presentation components
├── cli.py                      # PEEK Terminal CLI with clean step indicators
├── run.bat                     # One-click Windows menu launcher
├── COMMANDS.txt                # Plaintext quick reference guide
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git exclusions (credentials, caches, logs)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
