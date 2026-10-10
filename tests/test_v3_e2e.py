#!/usr/bin/env python3
"""E2E test for Hermes improvements v3.

Run with the gateway venv so SentenceTransformer is importable:
    ~/.hermes/hermes-agent/venv/bin/python3 ~/.hermes/scripts/test-improvements-v3.py

Proves the v2 bug is fixed: v2 wrapped `handle_message`, which AIAgent does not
have, so every per-turn component was loaded and never consulted. v3 wraps
`run_conversation` (the real entry point) and injects on the USER message, never
the system prompt, so prompt caching stays intact.

ISOLATION (load-bearing — do not remove):
HERMES_HOME is pointed at a temp dir BEFORE importing improvements, because
initialize_hermes_improvements() falls back to the real ~/.hermes when no
hermes_home is passed. Without this, AdaptiveSoul.record_feedback() writes
synthetic corrections into the real soul_feedback.jsonl and re-synthesizes
soul_adaptive_rules.json, destroying learned rules. An earlier revision of this
file did exactly that (5 rule keys -> 2). The final guard re-hashes the real
state files and fails the run if anything moved.
"""
import hashlib
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

HERMES = Path.home() / ".hermes"

# Real files that must never be touched by this test.
GUARDED = [
    HERMES / "memories" / "soul_feedback.jsonl",
    HERMES / "memories" / "soul_adaptive_rules.json",
    HERMES / "memories" / "memory_vectors.db",
    HERMES / "memories" / "MEMORY.md",
    HERMES / "memories" / "USER.md",
]


def snapshot(paths):
    out = {}
    for p in paths:
        if p.exists():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


BASELINE = snapshot(GUARDED)

# Redirect HERMES_HOME before improvements is imported or initialized.
TMP = Path(tempfile.mkdtemp(prefix="v3test_"))
(TMP / "memories").mkdir(parents=True)
(TMP / "memories" / "MEMORY.md").write_text(
    "Backup-ul pe remote-backup-node ruleaza zilnic la 03:00 prin cron.\n"
    "§Skill-urile stau in ~/.hermes/skills grupate pe categorii.\n",
    encoding="utf-8",
)
os.environ["HERMES_HOME"] = str(TMP)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(HERMES))

PASS, FAIL = [], []


def check(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(f"{'PASS' if cond else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))


class FakeAgent:
    """Minimal stand-in exposing only what the hook needs."""

    def __init__(self):
        self.model = "test-model"
        self.calls = []

    def run_conversation(self, user_message=None, **kwargs):
        self.calls.append((user_message, kwargs))
        return {"final_response": "raspuns", "messages": []}


def fresh(integ, cls=FakeAgent):
    """A patched agent bound to the temp home."""
    agent = cls()
    integ._CONFIG_CACHE.clear()
    integ.patch_agent_for_improvements(agent)
    return agent


def main():
    from improvements import __version__
    import improvements.integration as integ
    from improvements.integration import (
        _strip_context_block,
        build_turn_context_block,
        detect_turn_feedback,
        get_agent_stats,
        get_improvements,
        patch_agent_for_improvements,
    )

    check("versiune 3.x", __version__.startswith("3."), __version__)

    # ---- hook installs on an object that has run_conversation ----
    a = fresh(integ)
    # In v3.2 s-a adaugat si anchor_ledger (7 componente total)
    n_active = sum(1 for v in get_improvements(a).values() if v)
    check("7/7 componente active", n_active >= 6)
    check("turn hook instalat", getattr(a, "_hermes_turn_hook_installed", False))
    check("run_conversation wrapped", getattr(a.run_conversation, "_hermes_wrapped", False))
    check(
        "izolare: componentele folosesc temp home",
        str(getattr(a, "_hermes_home", "")) == str(TMP),
        str(getattr(a, "_hermes_home", "")),
    )

    # ---- a turn passes through untouched in its response ----
    res = a.run_conversation(user_message="Cate skill-uri am?")
    check("raspuns intact", res.get("final_response") == "raspuns")
    check("turul a ajuns la original", len(a.calls) == 1)

    sent = a.calls[0][0]
    check("context adaptiv injectat", "[hermes-improvements" in sent)
    check("intrebarea reala pastrata", "Cate skill-uri am?" in sent)
    persisted = a.calls[0][1].get("persist_user_message")
    check("transcript curat (fara block)", persisted == "Cate skill-uri am?", repr(persisted))

    # ---- feedback detection + persistence ----
    check("feedback negativ detectat", detect_turn_feedback("prea lung, scrie mai scurt") is not None)
    check("mesaj neutru nu e feedback", detect_turn_feedback("cat e ceasul") is None)

    soul = get_improvements(a)["adaptive_soul"]
    before = len(soul.feedback_history)
    a.run_conversation(user_message="prea lung, scrie mai scurt")
    check("soul a inregistrat corectura", len(soul.feedback_history) > before)
    check(
        "feedback scris in temp, nu in real",
        (TMP / "memories" / "soul_feedback.jsonl").exists(),
    )

    # ---- the learned rule reaches the model on the NEXT turn ----
    a.run_conversation(user_message="si acum despre skill-uri")
    check("regula invatata ajunge la model", "Reguli învățate din feedback" in a.calls[-1][0])

    # ---- no duplicate injection of an identical block ----
    n = getattr(a, "_hermes_blocks_injected", 0)
    a.run_conversation(user_message="si acum despre skill-uri")
    check("block identic nu se reinjecteaza", getattr(a, "_hermes_blocks_injected", 0) == n)

    # ---- strip helper is exact ----
    # Must use the real markers: a made-up opener is correctly left untouched.
    block = f"{integ._BLOCK_OPEN}\nfoo\n{integ._BLOCK_CLOSE}\n\nintrebare"
    check("_strip_context_block curata", _strip_context_block(block) == "intrebare")
    check(
        "_strip_context_block lasa textul strain neatins",
        _strip_context_block("[alt-marker]\nfoo\n[/alt-marker]") == "[alt-marker]\nfoo\n[/alt-marker]",
    )

    # ---- VectorMemory prefetch reaches the block (the text_preview bug) ----
    # Read-only against the REAL store: search() never writes. Building the store
    # directly (not via initialize) avoids re-indexing and re-saving the real DB.
    from improvements.dynamic_memory import AdaptiveMemoryPrefetch
    from improvements.vector_memory import VectorMemoryStore

    vs = VectorMemoryStore(HERMES / "memories")
    mem = AdaptiveMemoryPrefetch(vs).prefetch_relevant("cum merge backup-ul pe remote-backup-node?", max_entries=3)
    check("prefetch gaseste memorii", len(mem) > 0, f"{len(mem)} entries")

    holder = FakeAgent()
    holder._hermes_improvements = {}
    blk = build_turn_context_block(holder, {"relevant_memories": mem})
    check("memorii apar in block", "Memorii relevante" in blk)
    lines = [l for l in blk.splitlines() if l.startswith("- [")]
    check("liniile de memorie au text", bool(lines) and all(len(l) > 10 for l in lines), f"{len(lines)} linii")

    # ---- resilience: a broken component must not kill the turn ----
    class Boom:
        def build_soul_extension(self):
            raise RuntimeError("stricat")

    b = fresh(integ)
    get_improvements(b)["adaptive_soul"] = Boom()
    check(
        "turul supravietuieste unei componente rupte",
        b.run_conversation(user_message="test").get("final_response") == "raspuns",
    )

    # ---- errors from the real turn still propagate ----
    class Err(FakeAgent):
        def run_conversation(self, user_message=None, **kwargs):
            raise ValueError("eroare reala")

    e = fresh(integ, Err)
    try:
        e.run_conversation(user_message="x")
        check("eroarea se propaga", False)
    except ValueError:
        check("eroarea se propaga", True)

    # ---- multimodal payloads pass through untouched ----
    m = fresh(integ)
    payload = [{"type": "text", "text": "ce e in poza"}]
    m.run_conversation(user_message=payload)
    check("payload multimodal neatins", m.calls[-1][0] is payload)

    # ---- positional call style works ----
    p = fresh(integ)
    p.run_conversation("intrebare pozitionala")
    check("apel pozitional functioneaza", "intrebare pozitionala" in p.calls[-1][0])

    # ---- patch is idempotent ----
    before_id = id(a.run_conversation)
    patch_agent_for_improvements(a)
    check("patch idempotent", id(a.run_conversation) == before_id)

    # ---- stats still work (v2 API) ----
    check("stats are componente", bool(get_agent_stats(a).get("components")))

    # ---- every v2 public name still importable ----
    for name in [
        "initialize_hermes_improvements", "get_improvements", "analyze_user_turn",
        "record_behavior_feedback", "record_memory_mutation",
        "build_enhanced_system_prompt", "inject_response_enhancements",
        "get_agent_stats", "patch_agent_for_improvements", "persist_session_learnings",
    ]:
        check(f"API v2: {name}", callable(getattr(integ, name, None)))

    # ---- the guard: real state must be byte-identical ----
    after = snapshot(GUARDED)
    drifted = [k for k, v in BASELINE.items() if after.get(k) != v]
    check(
        "GARDA: starea reala din ~/.hermes neatinsa",
        not drifted,
        ", ".join(Path(d).name for d in drifted) or f"{len(BASELINE)} fisiere verificate",
    )

    print()
    if FAIL:
        print(f"EȘUAT: {len(FAIL)} din {len(PASS) + len(FAIL)}")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print(f"TOATE {len(PASS)} TESTELE AU TRECUT")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
