# Hermes Improvements Package (v3.0.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hermes Agent Supported](https://img.shields.io/badge/Hermes%20Agent-Compatible-8A2BE2.svg)](https://github.com/NousResearch/hermes-agent)
[![Stars](https://img.shields.io/github/stars/rednicv/hermes-improvements?style=social)](https://github.com/rednicv/hermes-improvements)

An architectural enhancement suite for [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research. 

It introduces dynamic vector-backed memory retrieval, adaptive persona alignment, real-time reasoning tracing, and intelligent task classification without breaking per-conversation prompt caching.

---

## 🚀 What's New in v3.0.0 (The True Architecture Fix)

In previous versions (v2.x), the integration hook attempted to wrap `agent.handle_message`. However, in `hermes-agent`, `AIAgent` implements `run_conversation` across all platforms (Gateway/Telegram, CLI, TUI, ACP, and One-shot), meaning that while components loaded into memory, per-turn hooks were bypassed.

**v3.0.0 completely rebuilds the turn pipeline:**

1. **`AIAgent.run_conversation` Hooking**:
   - Seamlessly attaches to the true message entry point across all Hermes surfaces.
   - Guaranteed multi-platform execution without modifying Hermes core files.

2. **Strict Prompt Caching Invariant**:
   - Hermes treats the initial system prompt as byte-stable to maximize LLM prompt caching (Anthropic prompt caching, OpenAI cache prefix, Gemini caching).
   - In v3, **system prompt mutation mid-conversation is eliminated**. Instead, adaptive guidance (learned rules, observed user style, task complexity, pre-fetched vector memories) is prepended to the incoming user message.
   - The original clean input is recorded in `persist_user_message` so transcripts, database logs, and chat histories remain 100% clean and human-readable.

3. **Live Feedback Auto-Detection**:
   - User corrections (e.g., "be shorter", "too verbose", "speak in Romanian", "no emoji") are automatically recognized during turns and fed directly into `AdaptiveSoul`.
   - Rules are synthesized immediately and apply on subsequent turns without manual rule configuration.

4. **Vector Memory Snippet Alignment**:
   - Fixed text retrieval mapping (`text_preview`) so memory search results correctly populate context prompts with high semantic relevancy.

---

## 🌟 Key Features

1. **VectorMemoryStore (`vector_memory.py`)**
   - High-performance local semantic search over user memories using embeddings (`sentence-transformers` default `all-MiniLM-L6-v2`, multilingual models, or TF-IDF fallback).
   - Configurable model via `HERMES_EMBEDDING_MODEL` environment variable (e.g. `paraphrase-multilingual-MiniLM-L12-v2` for 50+ languages).
   - Enables fast context lookup without bloating the primary prompt window.

2. **DynamicMemoryContext & Prefetch (`dynamic_memory.py`)**
   - Automatically prefetches and injects 1–3 relevant memories dynamically based on current user prompts.

3. **AdaptiveSoul & StyleLearner (`adaptive_soul.py`)**
   - Learns user style preferences, rules, and corrections over time.
   - Logs persistent behavioral rules and audits adaptations transparently.

4. **AdaptiveWorkflow (`adaptive_workflow.py`)**
   - Classifies task complexity in real time (TRIVIAL, SIMPLE, MODERATE, COMPLEX) to assist decision-making and tool-call budgeting.

5. **ReasoningTracer & SourceAttribution (`reasoning_trace.py`)**
   - Traces step-by-step reasoning, manages uncertainty scores, and tracks source attribution across execution steps.

6. **Seamless Integration Layer (`integration.py` & `inject_hook.py`)**
   - Non-destructive hook injection into `agent_init.py`.
   - Survives `hermes-agent` upgrades and `pip install --upgrade` routines.

---

## 📉 Token & Cost Reduction (How it Works)

Normally, Hermes Agent loads **your entire memory file** (all custom rules, preferences, and details) into the System Prompt of every single message. As your memory grows, your system prompt bloats, wasting thousands of tokens per turn and skyrocketing your API costs.

This package solves this through two main optimizations:

1. **Semantic Memory Selection (Vector Search)**:
   - Instead of injecting the entire memory dump, `VectorMemoryStore` indexes your memories using local embeddings.
   - On each message, it performs a quick search and **only injects the 1-3 memories relevant to your current prompt**.

2. **Preserving Prompt Caching**:
   - Because the system prompt remains byte-stable, LLM providers (Anthropic Claude, Google Gemini, DeepSeek) hit **100% Prompt Cache** on repetitive turns. You only pay for processing new tokens, saving up to 50–80% on API bills.

---

## 📋 Requirements

- **Python:** `>= 3.10`
- **Memory:** `≥ 2 GB RAM` (if using `sentence-transformers` embeddings)
- **Optional Dependencies:**
  - `sentence-transformers` & `numpy` (for vector semantic search; falls back to TF-IDF if omitted)

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Hermes Agent installed and active:
```bash
hermes --version
```

### 2. Copy Package Files
Copy the `hermes_improvements` directory into your Hermes configuration home:
```bash
mkdir -p ~/.hermes/improvements
cp -r src/hermes_improvements/* ~/.hermes/improvements/
```

### 3. Inject Hook
Run the non-destructive injector script to pair the improvements with Hermes Agent's execution core:
```bash
python3 src/inject_hook.py
```

### 4. Verify Installation
Check if the improvements package is initialized properly:
```bash
python3 -c "import sys, os; sys.path.insert(0, os.path.expanduser('~/.hermes')); import improvements; print(improvements.__version__)"
# Output: 3.2.0
```

---

## 📊 Measured Benchmarks (v3.2.0)

| Metric | Without Improvements | With hermes-improvements v3.2.0 | Gain |
| :--- | :--- | :--- | :--- |
| **Context Overhead / Turn** | ~3,200 tokens (raw files injected) | ~680 tokens (selective vector recall) | **-78.7% tokens** |
| **TF-IDF Vocabulary Build** | 12.4 ms per query | 0.08 ms (generation cached) | **155× faster** |
| **Active Rules Survival** | Expired at 30d regardless of use | Dynamic `last_hit` keepalive | **Zero rule loss on active skills** |
| **Cache Mutation Side-effects** | Search triggered writes | 100% Read-only queries | **Thread-safe / Pure** |

---

## 🧪 Comprehensive E2E Testing

v3 includes a comprehensive test suite testing 38 distinct invariants (hook attachment, prompt cache safety, error isolation, feedback loops):
```bash
python3 tests/test_v3_e2e.py
```

---

## 💡 Troubleshooting & Notes

- **Upgrades:** Running `python3 src/inject_hook.py` is safe to run after every `pip install --upgrade hermes-agent` or `git pull`. It checks if the hook is already present before modifying `agent_init.py`.
- **Platform Support:** Full support for Linux and macOS. Windows environments gracefully fall back when file-locking is unavailable.

---

## 🛡️ License
MIT License. Free to use, modify, and distribute for the Hermes Agent community.
