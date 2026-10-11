# Hermes Improvements Package (v3.4.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hermes Agent Supported](https://img.shields.io/badge/Hermes%20Agent-Compatible-8A2BE2.svg)](https://github.com/NousResearch/hermes-agent)
[![Stars](https://img.shields.io/github/stars/rednicv/hermes-improvements?style=social)](https://github.com/rednicv/hermes-improvements)

An architectural enhancement suite for [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research. 

It introduces dynamic vector-backed memory retrieval, adaptive persona alignment, real-time reasoning tracing, intelligent task classification, and live token savings telemetry without breaking per-conversation prompt caching.

---

## 🚀 What's New in v3.4.0

1. **Full Memory Text Preservation**:
   - The entire memory text is retained losslessly in `entry["text"]`; trimming to preview limit only occurs on prompt injection or display.
2. **High-Performance Bulk Ingestion (`bulk_add`)**:
   - Added batch insertion with single-generation advancement (`bulk_add`), preventing massive re-embedding cascades during bulk imports.
3. **Dynamic Real-World Telemetry Baseline**:
   - Telemetry baseline measures the exact size of actual memory sources (`MEMORY.md`, `USER.md`, soul rules) dynamically rather than relying on a hardcoded token estimate.
4. **Transient Failure Recovery for SentenceTransformer**:
   - `_get_model()` implements a TTL retry mechanism (5 minutes) so temporary load failures do not permanently lock the system in fallback mode.
5. **Configurable TF-IDF Capacity**:
   - Vocabulary size is configurable via `HERMES_TFIDF_MAX_VOCAB` (default 512, floor 128) and tokenization natively retains technical alphanumeric identifiers.

---

## 🚀 What's New in v3.3.1

1. **TF-IDF Vocabulary Stability & Discriminative Power**:
   - Retained hapax legomena (words with single document occurrences) in vocabulary construction; highly discriminative personal memory tokens (names, specific technical identifiers) are no longer discarded.
   - Guaranteed deterministic vocabulary indexing via sorted alphabetical ordering of features across calls.
   - Added generation-tracking to vector entries: out-of-sync embeddings are refreshed on-the-fly during search.
2. **Hardened Regression Testing**:
   - Upgraded `test_tfidf_dimension_drift_regression` to assert strict rank ordering and positive similarity score (`score > 0.05`), ensuring genuine semantic retrieval over insertion-order artifacts.

---

## 🚀 What's New in v3.3.0

1. **Auto-Healing TF-IDF Dimension Drift**:
   - In environments without `sentence-transformers`, vocabulary growth no longer silently causes dimension clashes or dropped memories.
   - Out-of-sync embeddings are automatically re-computed on-the-fly during search.
2. **Token Savings Telemetry**:
   - `get_agent_stats()` now includes a real-time `telemetry` metric tracking injected blocks, baseline tokens vs actual tokens, and percentage saved per turn (~93–95% savings).
3. **Configurable Memory Preview**:
   - Preview size is configurable via `HERMES_MEMORY_PREVIEW_CHARS` (default 200, safe bounds 50+).
4. **Hermetic Test Suite**:
   - Full test isolation in E2E tests preventing host directory pollution; 100% green test passes across unit and E2E suites.

---

## 🚀 Architecture Highlights (v3.0.0+)

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
# Output: 3.2.2
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
