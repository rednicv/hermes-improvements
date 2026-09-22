#!/usr/bin/env python3
"""
Integration Layer for Hermes Improvements (v3)

Hooks the adaptive components into the existing Hermes agent.
Placed in ~/.hermes/improvements/ to survive `pip install --upgrade`.

The apply-improvements.sh script injects a one-liner into
agent/agent_init.py that imports and calls this module's
patch_agent_for_improvements() at the end of init_agent().

v3 (20 Sep 2026) — why this version exists
------------------------------------------
v2 wrapped ``agent.handle_message``, a method AIAgent does not have. The
wrapper therefore never installed and every per-turn component (soul, style,
workflow, prefetch, tracer) was loaded but never called; only VectorMemory ran,
because it starts during initialize_hermes_improvements().

v3 wraps ``agent.run_conversation`` instead — the real per-turn entry point
(agent/turn_facade.py -> agent/conversation_loop.py), used by every surface
(gateway, CLI, TUI, oneshot, subagents).

Adaptive context is injected by PREPENDING a delimited block to this turn's
user message, never by rebuilding the system prompt. Hermes treats the
per-conversation prompt prefix as byte-stable (see hermes-agent/AGENTS.md);
mid-conversation content must ride a user message or tool result. The clean
original text is passed as ``persist_user_message`` so transcripts and the UI
stay unchanged.

Nothing from v2 was removed: every public function, the handle_message wrapper,
the auto-routing model map and the atexit persistence are all still here. The
behaviours that never actually ran in v2 (response-tail injection, auto-routing)
stay OFF by default and are opt-in via config.yaml, so enabling v3 does not
silently change observable output.
"""

import atexit
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────

_DEFAULT_CONFIG = {
    # Master switch for the per-turn hook. False keeps components loaded
    # (VectorMemory, watcher, stats) but installs no turn wrapper.
    "enabled": True,
    # Prepend adaptive context (soul rules + style + memories) to the user message.
    "inject_context": True,
    # Hard cap on injected characters per turn.
    "max_inject_chars": 1200,
    # Detect user corrections and feed them to AdaptiveSoul.
    "record_feedback": True,
    # Append the reasoning trace to the final response text.
    # OFF: the gateway may already have streamed the text, so a tail append
    # would desync what the user saw from what is persisted.
    "inject_reasoning": False,
    # Append a memory-mutation summary to the final response text. OFF, same reason.
    "inject_memory_updates": False,
    # Rewrite agent.model per task complexity. OFF: switching model
    # mid-conversation discards the cached prompt prefix.
    "auto_routing": False,
    # Max memories pulled from VectorMemory per turn.
    "prefetch_max_entries": 3,
}

_CONFIG_CACHE: dict = {}


def _load_improvements_config(hermes_home: Path) -> dict:
    """Read the optional ``improvements:`` section from config.yaml.

    Behavioural settings belong in config.yaml, not in environment variables
    (hermes-agent/AGENTS.md). Missing file, missing section, unreadable YAML or
    a missing PyYAML all fall back to _DEFAULT_CONFIG — the hook must never
    fail to install because of config.
    """
    key = str(hermes_home)
    if key in _CONFIG_CACHE:
        return _CONFIG_CACHE[key]

    cfg = dict(_DEFAULT_CONFIG)
    try:
        import yaml  # lazy: never required at import time

        cfg_path = hermes_home / "config.yaml"
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            section = raw.get("improvements")
            if isinstance(section, dict):
                for k, v in section.items():
                    if k in cfg:
                        cfg[k] = v
                    else:
                        logger.debug("Unknown improvements config key ignored: %s", k)
    except Exception as e:
        logger.debug("improvements config not loaded (using defaults): %s", e)

    _CONFIG_CACHE[key] = cfg
    return cfg


# ─── Memory File Indexing & Watcher ──────────────────────────────────

def _index_memory_files(vs, hermes_home: Path):
    """Index MEMORY.md + USER.md into VectorMemoryStore (split on §)."""
    memories_dir = hermes_home / "memories"
    for fname in ["MEMORY.md", "USER.md"]:
        fpath = memories_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("§") if p.strip()]
        for i, para in enumerate(paragraphs):
            vs.add(f"{fname}:{i}", para, {"source": fname, "index": i}, auto_save=False)
        if hasattr(vs, "_save"):
            vs._save()
        logger.info("✅ VectorMemory: %d paragraphs indexed from %s", len(paragraphs), fname)


def _start_memory_watcher(vs, hermes_home: Path):
    """
    Background thread: re-index MEMORY.md/USER.md when they change on disk.
    Uses mtime polling every 30s — no extra dependencies.
    """
    import threading

    memories_dir = hermes_home / "memories"
    watched = {
        fname: (memories_dir / fname).stat().st_mtime
        if (memories_dir / fname).exists() else 0
        for fname in ["MEMORY.md", "USER.md"]
    }

    def _watch():
        while True:
            time.sleep(30)
            try:
                changed_files = []
                for fname, last_mtime in list(watched.items()):
                    fpath = memories_dir / fname
                    if not fpath.exists():
                        continue
                    mtime = fpath.stat().st_mtime
                    if mtime != last_mtime:
                        changed_files.append(fname)
                if changed_files:
                    # refresh all watched mtimes first, then reindex once
                    for fname in watched:
                        fpath = memories_dir / fname
                        watched[fname] = fpath.stat().st_mtime if fpath.exists() else 0
                    _index_memory_files(vs, hermes_home)
                    logger.info("🔄 VectorMemory re-indexed: %s changed", ", ".join(changed_files))
            except Exception as e:
                logger.warning("Memory watcher error: %s", e)

    t = threading.Thread(target=_watch, daemon=True, name="memory-watcher")
    t.start()
    logger.info("✅ Memory file watcher started (poll every 30s)")


# ─── Core: Initialize All Components ─────────────────────────────────

def initialize_hermes_improvements(
    agent_instance,
    hermes_home: Optional[Path] = None,
) -> dict:
    """
    Initialize all improvement components and attach to agent.

    Args:
        agent_instance: The AIAgent instance
        hermes_home: Override Hermes home directory

    Returns:
        Components dict: {
            "vector_store": VectorMemoryStore | None,
            "dynamic_memory": DynamicMemoryContext | None,
            "adaptive_soul": AdaptiveSoul | None,
            "style_learner": StyleLearner | None,
            "adaptive_workflow": AdaptiveWorkflow | None,
            "reasoning_tracer": ReasoningTracer | None,
        }
    """
    if hermes_home is None:
        try:
            from hermes_constants import get_hermes_home as _get_home
            hermes_home = _get_home()
        except ImportError:
            hermes_home = Path.home() / ".hermes"

    components = {}

    # 1. Vector Memory Store (cu prag de RAM)
    try:
        # Prag RAM minim: 2GB liberi pentru SentenceTransformer
        _free_ram = 0
        try:
            import psutil
            _free_ram = int(psutil.virtual_memory().available // (1024 * 1024))
        except Exception:
            try:
                with open('/proc/meminfo') as _f:
                    for _line in _f:
                        if _line.startswith('MemAvailable:'):
                            _free_ram = int(_line.split()[1]) // 1024  # KiB → MB
                            break
            except Exception:
                _free_ram = 4096  # fallback default if memory stats unavailable

        if _free_ram < 2048:  # < 2GB liberi
            logger.warning("⚠️ VectorMemory: doar %d MB RAM liber (prag minim 2048 MB) — dezactivat", _free_ram)
            components["vector_store"] = None
        else:
            from improvements.vector_memory import VectorMemoryStore
            memory_dir = hermes_home / "memories"
            vs = VectorMemoryStore(memory_dir)
            components["vector_store"] = vs
            logger.info("✅ Vector memory store initialized (%d MB RAM liber)", _free_ram)
    except Exception as e:
        logger.warning("Failed to init vector memory: %s", e)
        components["vector_store"] = None

    # 2. Dynamic Memory Context
    try:
        from improvements.dynamic_memory import DynamicMemoryContext
        dm = DynamicMemoryContext()
        components["dynamic_memory"] = dm
        logger.info("✅ Dynamic memory context initialized")
    except Exception as e:
        logger.warning("Failed to init dynamic memory: %s", e)
        components["dynamic_memory"] = None

    # 3. Adaptive Soul + Style Learner
    try:
        from improvements.adaptive_soul import AdaptiveSoul, StyleLearner
        soul_dir = hermes_home / "memories"
        soul = AdaptiveSoul(soul_dir)
        style = StyleLearner()
        components["adaptive_soul"] = soul
        components["style_learner"] = style
        logger.info("✅ Adaptive soul initialized")
    except Exception as e:
        logger.warning("Failed to init adaptive soul: %s", e)
        components["adaptive_soul"] = None
        components["style_learner"] = None

    # 4. Adaptive Workflow
    try:
        from improvements.adaptive_workflow import AdaptiveWorkflow
        wf = AdaptiveWorkflow()
        components["adaptive_workflow"] = wf
        logger.info("✅ Adaptive workflow initialized")
    except Exception as e:
        logger.warning("Failed to init adaptive workflow: %s", e)
        components["adaptive_workflow"] = None

    # 5. Reasoning Tracer
    try:
        from improvements.reasoning_trace import ReasoningTracer
        rt = ReasoningTracer()
        components["reasoning_tracer"] = rt
        logger.info("✅ Reasoning tracer initialized")
    except Exception as e:
        logger.warning("Failed to init reasoning tracer: %s", e)
        components["reasoning_tracer"] = None

    # Attach to agent instance
    agent_instance._hermes_improvements = components
    agent_instance._hermes_home = hermes_home

    # Populate vector memory from MEMORY.md + USER.md
    vs = components.get("vector_store")
    if vs:
        try:
            _index_memory_files(vs, hermes_home)
            # Start background watcher for live re-indexing on file change
            _start_memory_watcher(vs, hermes_home)
        except Exception as e:
            logger.warning("Failed to populate vector memory from files: %s", e)

    total_ok = sum(1 for v in components.values() if v is not None)
    logger.info(
        "✅ Hermes improvements loaded: %d/%d components",
        total_ok, len(components)
    )
    return components


# ─── Accessors ───────────────────────────────────────────────────────

def get_improvements(agent_instance) -> dict:
    """Get initialized improvements from agent instance."""
    return getattr(agent_instance, "_hermes_improvements", {})


# ─── Pre-Turn Analysis ──────────────────────────────────────────────

def analyze_user_turn(
    agent_instance,
    user_message: str,
    context: dict = None,
) -> dict:
    """
    Pre-turn analysis: classify complexity, detect style, prefetch memories.

    Call this BEFORE processing the user message.
    """
    improvements = get_improvements(agent_instance)
    context = context or {}
    results = {}

    # Classify task complexity
    wf = improvements.get("adaptive_workflow")
    rt = improvements.get("reasoning_tracer")

    # Reset tracer at start of each turn
    if rt:
        rt.clear()

    if wf:
        try:
            complexity = wf.classify_task(user_message, context)
            plan = wf.execute_workflow(user_message, complexity, context)
            results["workflow"] = plan

            # Log reasoning via tracer
            rt = improvements.get("reasoning_tracer")
            if rt:
                rt.add_step(
                    action="analyze",
                    input_data=user_message[:200],
                    reasoning=f"Task classified as {complexity.value} with {len(plan.get('steps', []))} steps",
                    output=f"Workflow: {plan.get('complexity', '?')} — {plan.get('estimated_tools', '?')} tools",
                    confidence=0.8,
                )
        except Exception as e:
            logger.warning("Workflow analysis failed: %s", e)

    # Detect user style
    style = improvements.get("style_learner")
    if style:
        try:
            style.analyze_user_message(user_message)
            results["style_guidance"] = style.get_style_guidance()
        except Exception as e:
            logger.warning("Style analysis failed: %s", e)

    # Prefetch relevant memories
    vs = improvements.get("vector_store")
    if vs:
        try:
            from improvements.dynamic_memory import AdaptiveMemoryPrefetch
            prefetcher = AdaptiveMemoryPrefetch(vs)
            max_entries = 3
            try:
                cfg = _load_improvements_config(
                    getattr(agent_instance, "_hermes_home", Path.home() / ".hermes")
                )
                max_entries = int(cfg.get("prefetch_max_entries", 3))
            except Exception:
                pass
            memories = prefetcher.prefetch_relevant(user_message, max_entries=max_entries)
            if memories:
                results["relevant_memories"] = memories

                # Log via tracer
                rt = improvements.get("reasoning_tracer")
                if rt:
                    sources = [m.get("key", "?") for m in memories[:3]]
                    rt.add_step(
                        action="search",
                        input_data=user_message[:100],
                        reasoning=f"VectorMemory found {len(memories)} relevant entries",
                        output=f"Sources: {', '.join(sources)}",
                        confidence=0.7,
                    )
        except Exception as e:
            logger.warning("Memory prefetch failed: %s", e)

    return results


# ─── Feedback Recording ──────────────────────────────────────────────

def record_behavior_feedback(
    agent_instance,
    category: str,
    sentiment: int,
    text: str,
    context: str = "",
):
    """
    Record user feedback about agent behavior.

    Args:
        category: "tone", "accuracy", "formatting", "speed", etc.
        sentiment: -1 (bad), 0 (neutral), +1 (good)
        text: What user said
        context: What agent did
    """
    improvements = get_improvements(agent_instance)
    soul = improvements.get("adaptive_soul")
    if soul:
        try:
            soul.record_feedback(
                category=category,
                sentiment=sentiment,
                text=text,
                context=context,
            )
            logger.debug("Feedback recorded: %s (%+d)", category, sentiment)
        except Exception as e:
            logger.warning("Failed to record feedback: %s", e)


# Correction patterns → (category, sentiment). RO + EN.
# Deliberately narrow: a false positive writes a durable behavioural rule, so
# only unambiguous meta-feedback about the agent's own output is matched.
_FEEDBACK_PATTERNS = [
    ("language", -1, [
        "nu engleză", "nu engleza", "nu in engleza", "nu în engleză",
        "vorbește română", "vorbeste romana", "scrie în română", "scrie in romana",
        "not english", "speak romanian", "in romanian",
    ]),
    ("verbosity", -1, [
        "prea lung", "prea mult text", "mai scurt", "fii concis", "fii mai scurt",
        "prea multe detalii", "too long", "too verbose", "be brief", "be concise",
        "shorter please",
    ]),
    ("formatting", -1, [
        "nu folosi emoji", "fără emoji", "fara emoji", "no emoji",
        "prea mult bold", "fără bold", "fara bold", "nu folosi bold",
        "prea multe headere", "fără titluri", "no headers", "stop bolding",
    ]),
    ("repetition", -1, [
        "te repeți", "te repeti", "ai spus deja", "ai mai spus", "nu repeta",
        "you already said", "stop repeating",
    ]),
    ("accuracy", -1, [
        "nu e corect", "nu este corect", "ai greșit", "ai gresit",
        "e greșit ce", "e gresit ce", "nu e adevărat", "nu e adevarat",
        "that's wrong", "thats wrong", "you are wrong", "you're wrong",
        "that is incorrect",
    ]),
    ("tone", 1, [
        "bravo", "perfect așa", "perfect asa", "îmi place", "imi place",
        "exact așa", "exact asa", "well done", "exactly right",
    ]),
]

# Meta-feedback is short. A long message mentioning "ai greșit" is usually about
# code or a third party, not about the agent's own style.
_FEEDBACK_MAX_CHARS = 300


def detect_turn_feedback(user_message: str):
    """Return ``(category, sentiment)`` when the message reads as feedback about
    the agent's own behaviour, else ``None``.

    Keyword heuristic, intentionally conservative: first match wins, long
    messages are skipped. It will miss paraphrased corrections — that is
    preferred over inventing rules from ambiguous text.
    """
    if not user_message:
        return None
    text = user_message.strip()
    if len(text) > _FEEDBACK_MAX_CHARS:
        return None
    lowered = text.lower()
    for category, sentiment, keywords in _FEEDBACK_PATTERNS:
        for kw in keywords:
            if kw in lowered:
                return category, sentiment
    return None


def _maybe_record_feedback(agent_instance, user_message: str) -> Optional[str]:
    """Record detected feedback into AdaptiveSoul. Returns the category, or None."""
    hit = detect_turn_feedback(user_message)
    if not hit:
        return None
    category, sentiment = hit
    record_behavior_feedback(
        agent_instance,
        category=category,
        sentiment=sentiment,
        text=user_message.strip()[:200],
        context="detected from user turn",
    )
    logger.info("📝 Feedback detected: %s (%+d)", category, sentiment)
    return category


def record_memory_mutation(
    agent_instance,
    action: str,
    target: str,
    content: str,
    old_content: str = None,
):
    """
    Record memory changes for live tracking.

    Args:
        action: "add", "replace", "remove"
        target: "memory", "user"
        content: New content
        old_content: Previous content (for replace/remove)
    """
    improvements = get_improvements(agent_instance)
    dm = improvements.get("dynamic_memory")
    if dm:
        try:
            dm.record_mutation(action, target, content, old_content)
        except Exception as e:
            logger.warning("Failed to record memory mutation: %s", e)


# ─── System Prompt Builder ───────────────────────────────────────────

def build_enhanced_system_prompt(agent_instance) -> str:
    """
    Build system prompt extensions from adaptive components.

    Kept for callers that assemble a prompt BEFORE a conversation starts (the
    only point where adding to the system prompt is cache-safe). The per-turn
    hook does NOT use this — see build_turn_context_block().
    """
    improvements = get_improvements(agent_instance)
    extensions = []

    # Dynamic memory instructions
    dm = improvements.get("dynamic_memory")
    if dm:
        ext = dm.build_system_prompt_fragment()
        if ext:
            extensions.append(ext)

    # Adaptive soul rules
    soul = improvements.get("adaptive_soul")
    if soul:
        ext = soul.build_soul_extension()
        if ext:
            extensions.append(ext)

    # Style guidance
    style = improvements.get("style_learner")
    if style:
        guidance = style.get_style_guidance()
        if guidance:
            extensions.append(f"\n## User Style Preferences\n{guidance}")

    # Workflow guidance
    wf = improvements.get("adaptive_workflow")
    if wf:
        extensions.append(
            "\n## Adaptive Workflow\n"
            "Select workflow complexity (TRIVIAL/SIMPLE/MODERATE/COMPLEX) "
            "based on task, use appropriate steps."
        )

    # Reasoning trace — inject only if there are actual steps
    rt = improvements.get("reasoning_tracer")
    if rt and rt.steps:
        trace = rt.get_trace_markdown()
        if trace:
            extensions.append(f"\n## Reasoning Trace (this turn)\n{trace}")
    if rt and rt.claims:
        claims = rt.get_claims_markdown()
        if claims:
            extensions.append(f"\n## Confidence Claims\n{claims}")

    return "\n".join(extensions)


# ─── Per-Turn Context Block (cache-safe injection) ───────────────────

_BLOCK_OPEN = "[hermes-improvements — context adaptiv, nu face parte din mesajul utilizatorului]"
_BLOCK_CLOSE = "[/hermes-improvements]"

_COMPLEXITY_HINT = {
    "trivial": "Răspunde direct, fără tool-uri dacă nu e nevoie.",
    "simple": "Task simplu: rezolvă direct, verificare minimă.",
    "moderate": "Task moderat: planifică pe scurt înainte, verifică rezultatul după.",
    "complex": "Task complex: investighează înainte de a schimba ceva, apoi verifică.",
}


def build_turn_context_block(
    agent_instance,
    turn_ctx: dict,
    max_chars: int = 1200,
) -> str:
    """Build the adaptive context block prepended to this turn's user message.

    Source of every line is a component that v2 loaded but never consulted:
    AdaptiveSoul rules, StyleLearner guidance, AdaptiveWorkflow complexity and
    VectorMemory prefetch. Returns "" when there is nothing to say.
    """
    improvements = get_improvements(agent_instance)
    sections = []

    # 1. Learned behavioural rules (from real user corrections)
    soul = improvements.get("adaptive_soul")
    if soul:
        try:
            ext = soul.build_soul_extension()
            if ext and ext.strip():
                sections.append(ext.strip())
        except Exception as e:
            logger.warning("Soul extension build failed: %s", e)

    # 2. Observed user style
    guidance = turn_ctx.get("style_guidance")
    if guidance:
        sections.append(f"## Stil observat\n{guidance}")

    # 3. Workflow complexity hint
    wf_plan = turn_ctx.get("workflow") or {}
    complexity = wf_plan.get("complexity")
    hint = _COMPLEXITY_HINT.get(str(complexity).lower())
    if hint:
        sections.append(f"## Complexitate estimată: {str(complexity).upper()}\n{hint}")

    # 4. Relevant long-term memories
    memories = turn_ctx.get("relevant_memories") or []
    if memories:
        lines = ["## Memorii relevante (VectorMemory)"]
        for m in memories:
            # VectorMemoryStore.search() returns the snippet under "text_preview";
            # "text"/"content" are accepted so a future store shape still works.
            raw = m.get("text_preview") or m.get("text") or m.get("content") or ""
            text = str(raw).strip().replace("\n", " ")
            if not text:
                continue
            key = m.get("key", "?")
            lines.append(f"- [{key}] {text[:220]}")
        if len(lines) > 1:
            sections.append("\n".join(lines))

    if not sections:
        return ""

    body = "\n\n".join(sections)
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n…(trunchiat)"

    return f"{_BLOCK_OPEN}\n{body}\n{_BLOCK_CLOSE}"


def _strip_context_block(text: str) -> str:
    """Remove an injected block, so re-entrant calls cannot nest blocks."""
    if not isinstance(text, str) or _BLOCK_OPEN not in text:
        return text
    start = text.find(_BLOCK_OPEN)
    end = text.find(_BLOCK_CLOSE)
    if end == -1:
        return text
    return (text[:start] + text[end + len(_BLOCK_CLOSE):]).lstrip("\n")


# ─── Response Enhancement ────────────────────────────────────────────

def inject_response_enhancements(
    agent_instance,
    response: str,
    include_reasoning: bool = False,
    include_memory_updates: bool = True,
) -> str:
    """
    Enhance response with live data from adaptive components.

    Args:
        response: Original assistant response
        include_reasoning: Append reasoning trace
        include_memory_updates: Append memory mutation summary

    Returns:
        Enhanced response string
    """
    improvements = get_improvements(agent_instance)

    if include_memory_updates:
        dm = improvements.get("dynamic_memory")
        if dm:
            try:
                from improvements.dynamic_memory import inject_live_memory_into_response
                response = inject_live_memory_into_response(response, dm)
            except Exception as e:
                logger.warning("Memory injection failed: %s", e)

    if include_reasoning:
        rt = improvements.get("reasoning_tracer")
        if rt:
            try:
                from improvements.reasoning_trace import inject_reasoning_trace
                response = inject_reasoning_trace(response, rt)
            except Exception as e:
                logger.warning("Reasoning injection failed: %s", e)

    return response


# ─── Stats ───────────────────────────────────────────────────────────

def get_agent_stats(agent_instance) -> dict:
    """Get comprehensive stats from all adaptive components."""
    improvements = get_improvements(agent_instance)
    import time as _time

    stats = {
        "timestamp": _time.time(),
        "components": {},
    }

    soul = improvements.get("adaptive_soul")
    if soul:
        stats["components"]["adaptive_soul"] = soul.get_behavioral_stats()

    vs = improvements.get("vector_store")
    if vs:
        stats["components"]["memory_vectors"] = vs.get_memory_stats()

    dm = improvements.get("dynamic_memory")
    if dm:
        stats["components"]["dynamic_memory"] = dm.get_stats()

    wf = improvements.get("adaptive_workflow")
    if wf:
        stats["components"]["workflow_metrics"] = wf.metrics.get_stats()

    # v3: turn-hook counters, so "is it actually running?" is answerable live.
    stats["turn_hook"] = {
        "installed": bool(getattr(agent_instance, "_hermes_turn_hook_installed", False)),
        "turns_seen": int(getattr(agent_instance, "_hermes_turns_seen", 0)),
        "blocks_injected": int(getattr(agent_instance, "_hermes_blocks_injected", 0)),
        "feedback_recorded": int(getattr(agent_instance, "_hermes_feedback_recorded", 0)),
    }

    return stats


# ─── Session Persistence ──────────────────────────────────────────────

def persist_session_learnings(agent_instance):
    """
    Persist all learnings from this turn across sessions.
    Called after every response. Idempotent — safe to call always.

    Does 3 things:
    1. Syncs DynamicMemory mutations into VectorMemory index
    2. Saves AdaptiveSoul rules as a readable summary file
    3. Logs what was learned
    """
    import json
    improvements = get_improvements(agent_instance)
    learned = []

    # 1. DynamicMemory → VectorMemory sync
    dm = improvements.get("dynamic_memory")
    vs = improvements.get("vector_store")
    if dm and vs:
        mutations = dm.get_recent_mutations()
        if mutations:
            vs.sync_from_memory(mutations)
            learned.append(f"sync {len(mutations)} memory mutations to vector index")

    # 2. AdaptiveSoul rules → readable summary file
    soul = improvements.get("adaptive_soul")
    if soul:
        rules = soul.get_active_rules(limit=10)
        if rules:
            summary_path = Path(agent_instance._hermes_home if hasattr(agent_instance, '_hermes_home') else Path.home() / ".hermes") / "memories" / "_learned_rules.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            rules_data = []
            for r in rules:
                rules_data.append({
                    "name": r.name,
                    "category": r.category,
                    "condition": r.condition,
                    "action": r.action,
                    "confidence": r.confidence,
                    "hits": r.hits,
                })
            with open(summary_path, 'w') as f:
                json.dump(rules_data, f, indent=2)

            learned.append(f"saved {len(rules)} adaptive soul rules")

        # Also: persist any recent feedback as a memory entry
        if hasattr(soul, 'feedback_history') and soul.feedback_history:
            recent = soul.feedback_history[-3:]
            negative = [f for f in recent if f.sentiment < 0]
            if negative:
                for fb in negative:
                    learned.append(
                        f"corecție: '{fb.text[:60]}' (categorie: {fb.category})"
                    )

    if learned:
        logger.info("📝 Session learnings persisted: %s", "; ".join(learned))

    # 3. Save DynamicMemory session conclusions (for one-shot mode -z)
    dm = improvements.get("dynamic_memory")
    if dm:
        try:
            conclusions_path = Path(
                agent_instance._hermes_home
                if hasattr(agent_instance, '_hermes_home')
                else Path.home() / ".hermes"
            ) / "memories" / "_session_conclusions.jsonl"

            stats = dm.get_stats()
            mutations = dm.get_recent_mutations(limit=50)
            session_summary = {
                "timestamp": time.time(),
                "session_duration_s": stats.get("session_duration_s", 0),
                "total_mutations": stats.get("total_mutations", 0),
                "by_action": stats.get("by_action", {}),
                "key_learnings": [
                    m["content"][:200]
                    for m in mutations[-10:]
                    if m["action"] in ("add", "replace")
                ],
            }

            # Append one JSON line per session (JSONL — easy to append)
            import json as _json
            import fcntl
            with open(conclusions_path, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(_json.dumps(session_summary) + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)

            learned.append(
                f"saved {len(session_summary['key_learnings'])} session conclusions"
            )
        except Exception as e:
            logger.warning("Session conclusions save failed: %s", e)


# ─── Auto-Routing (opt-in) ───────────────────────────────────────────

# Complexity → model. Preserved from v2, where it was hard-disabled by a
# comment. Still OFF by default: swapping the model mid-conversation drops the
# cached prompt prefix and re-prefills the whole history.
AUTO_ROUTING_MODELS = {
    "trivial": "gemini-2.5-flash-lite",
    "simple": "gemini-2.5-flash-lite",
    "moderate": "gemini-3.5-flash-low",
    "complex": "gemini-3.1-pro-low",
}


def _maybe_auto_route(agent_instance, turn_ctx: dict, enabled: bool):
    """Apply complexity-based model routing when explicitly enabled in config."""
    plan = turn_ctx.get("workflow") or {}
    c_val = str(plan.get("complexity", "trivial")).lower()
    target_model = AUTO_ROUTING_MODELS.get(c_val, AUTO_ROUTING_MODELS["trivial"])

    if not enabled:
        logger.debug(
            "[off] auto-routing: complexity [%s] -> would target [%s]",
            c_val.upper(), target_model,
        )
        return
    try:
        agent_instance.model = target_model
        logger.info("🔄 Auto-routing: complexity [%s] -> model [%s]", c_val.upper(), target_model)
    except Exception as e:
        logger.warning("Auto-routing failed: %s", e)


# ─── Per-Turn Hook ───────────────────────────────────────────────────

def _extract_user_message(args, kwargs):
    """Locate this turn's user message across both call styles.

    Callers use both positional (``run_conversation(prompt, ...)``) and keyword
    (``run_conversation(user_message=..., ...)``) forms.
    """
    if args:
        return args[0], "positional"
    if "user_message" in kwargs:
        return kwargs["user_message"], "keyword"
    return None, None


def _install_turn_hook(agent_instance, cfg: dict) -> bool:
    """Wrap ``agent.run_conversation`` with the adaptive pre/post-turn pipeline.

    Returns True when the wrapper was installed.
    """
    original = getattr(agent_instance, "run_conversation", None)
    if original is None:
        logger.warning(
            "improvements: no run_conversation on %s — turn hook NOT installed",
            type(agent_instance).__name__,
        )
        return False
    if getattr(original, "_hermes_wrapped", False):
        logger.debug("improvements: run_conversation already wrapped")
        return True

    def wrapped(*args, **kwargs):
        user_message, style = _extract_user_message(args, kwargs)
        injected = False

        # ---- pre-turn ----
        try:
            if isinstance(user_message, str) and user_message.strip():
                agent_instance._hermes_turns_seen = (
                    getattr(agent_instance, "_hermes_turns_seen", 0) + 1
                )
                clean = _strip_context_block(user_message)

                if cfg.get("record_feedback", True):
                    if _maybe_record_feedback(agent_instance, clean):
                        agent_instance._hermes_feedback_recorded = (
                            getattr(agent_instance, "_hermes_feedback_recorded", 0) + 1
                        )

                turn_ctx = analyze_user_turn(agent_instance, clean)
                _maybe_auto_route(agent_instance, turn_ctx, bool(cfg.get("auto_routing", False)))

                if cfg.get("inject_context", True):
                    block = build_turn_context_block(
                        agent_instance, turn_ctx,
                        max_chars=int(cfg.get("max_inject_chars", 1200)),
                    )
                    # Re-injecting an identical block every turn only grows the
                    # history; the first copy stays visible in context.
                    fp = hash(block)
                    if block and fp != getattr(agent_instance, "_hermes_last_block_fp", None):
                        enriched = f"{block}\n\n{clean}"
                        if style == "positional":
                            args = (enriched,) + args[1:]
                        else:
                            kwargs["user_message"] = enriched
                        # Keep the transcript and UI showing what the user typed.
                        if kwargs.get("persist_user_message") is None:
                            kwargs["persist_user_message"] = clean
                        agent_instance._hermes_last_block_fp = fp
                        agent_instance._hermes_blocks_injected = (
                            getattr(agent_instance, "_hermes_blocks_injected", 0) + 1
                        )
                        injected = True
                        logger.info(
                            "🧠 improvements: adaptive context injected (%d chars)", len(block)
                        )
            elif user_message is not None:
                # Multimodal turns (list payloads) are passed through untouched.
                logger.debug("improvements: non-str user_message, injection skipped")
        except Exception as e:
            logger.warning("improvements pre-turn failed (turn continues): %s", e)

        # ---- the real turn ----
        result = original(*args, **kwargs)

        # ---- post-turn ----
        try:
            improvements = get_improvements(agent_instance)

            if injected:
                soul = improvements.get("adaptive_soul")
                if soul:
                    soul.record_behavior(outcome="turn completed")

            if isinstance(result, dict):
                want_reasoning = bool(cfg.get("inject_reasoning", False))
                want_memory = bool(cfg.get("inject_memory_updates", False))
                if want_reasoning or want_memory:
                    original_text = result.get("final_response")
                    if isinstance(original_text, str) and original_text:
                        result["final_response"] = inject_response_enhancements(
                            agent_instance, original_text,
                            include_reasoning=want_reasoning,
                            include_memory_updates=want_memory,
                        )

            dm = improvements.get("dynamic_memory")
            if dm:
                dm.next_turn()

            persist_session_learnings(agent_instance)
        except Exception as e:
            logger.warning("improvements post-turn failed (response preserved): %s", e)

        return result

    wrapped._hermes_wrapped = True
    wrapped._hermes_original = original
    agent_instance.run_conversation = wrapped
    agent_instance._hermes_turn_hook_installed = True
    return True


# ─── Agent Patching ──────────────────────────────────────────────────

def patch_agent_for_improvements(agent_instance):
    """
    Patch AIAgent with improvement hooks.

    Call this right after agent initialization.
    Safe to call multiple times — skips if already patched.
    """
    if getattr(agent_instance, "_hermes_patched", False):
        logger.debug("Agent already patched with improvements, skipping")
        return

    # Initialize all components
    initialize_hermes_improvements(agent_instance)

    hermes_home = getattr(agent_instance, "_hermes_home", Path.home() / ".hermes")
    cfg = _load_improvements_config(hermes_home)

    # v3: the real per-turn entry point.
    hook_installed = False
    if cfg.get("enabled", True):
        hook_installed = _install_turn_hook(agent_instance, cfg)
    else:
        logger.info("improvements: turn hook disabled via config (components still loaded)")

    # v2 compatibility: some surfaces may expose handle_message. AIAgent does
    # not, which is why v2 silently did nothing; kept so nothing is lost.
    original_handle = getattr(agent_instance, "handle_message", None)

    def enhanced_handle_message(user_message, *args, **kwargs):
        """Wrapped message handler with pre/post-turn enhancements."""
        # Reset ReasoningTracer for fresh trace each turn
        try:
            rt = get_improvements(agent_instance).get("reasoning_tracer")
            if rt:
                rt.clear()
        except Exception:
            pass

        # Pre-turn: analyze
        try:
            turn_ctx = analyze_user_turn(agent_instance, str(user_message))
            _maybe_auto_route(agent_instance, turn_ctx, bool(cfg.get("auto_routing", False)))
        except Exception as e:
            logger.warning("Pre-turn analysis or auto-routing failed: %s", e)
            turn_ctx = {}

        # Detect first turn — inject system intro
        dm = get_improvements(agent_instance).get("dynamic_memory")
        is_first = dm and dm.consume_first_turn() if dm else False

        # Call original
        if original_handle:
            response = original_handle(user_message, *args, **kwargs)
        else:
            response = ""

        # Post-turn: enhance response
        try:
            response = inject_response_enhancements(
                agent_instance,
                str(response),
                include_reasoning=bool(cfg.get("inject_reasoning", False)),
                include_memory_updates=bool(cfg.get("inject_memory_updates", False)),
            )
        except Exception as e:
            logger.warning("Post-turn enhancement failed: %s", e)

        # Persist everything — learnings survive across sessions
        try:
            persist_session_learnings(agent_instance)
        except Exception as e:
            logger.warning("Session persistence failed: %s", e)

        return response

    if original_handle:
        agent_instance.handle_message = enhanced_handle_message

    agent_instance._hermes_patched = True

    # Register atexit handler so conclusions are saved even in one-shot mode (-z)
    # and on any exit path (normal, error, Ctrl+C)
    import weakref
    ref = weakref.ref(agent_instance)

    def _save_on_exit():
        agent = ref()
        if agent is None:
            return
        try:
            persist_session_learnings(agent)
        except Exception as e:
            logger.warning("atexit session persistence failed: %s", e)

    atexit.register(_save_on_exit)

    logger.info(
        "✅ Agent patched with Hermes improvements v3 (turn_hook=%s, atexit registered)",
        "on" if hook_installed else "OFF",
    )


# ─── Public API ─────────────────────────────────────────────────────

__all__ = [
    "initialize_hermes_improvements",
    "get_improvements",
    "analyze_user_turn",
    "record_behavior_feedback",
    "record_memory_mutation",
    "build_enhanced_system_prompt",
    "build_turn_context_block",
    "detect_turn_feedback",
    "inject_response_enhancements",
    "get_agent_stats",
    "patch_agent_for_improvements",
    "persist_session_learnings",
    "AUTO_ROUTING_MODELS",
]
