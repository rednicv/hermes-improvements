#!/usr/bin/env python3
"""
Helper script for apply-improvements.sh
Injects the Hermes improvements hook into agent_init.py

Fixes applied 23 Jun 2026 (Kiro audit):
  - P0: Anchor changed from __all__ to end of init_agent() function
  - P1a: Backup uses PID to avoid same-second collision, reuses already-read content
"""

import sys
import os
from pathlib import Path

MARKER = "# HERMES_IMPROVEMENTS_HOOK - auto-applied"

HOOK_CODE = """    # HERMES_IMPROVEMENTS_HOOK - auto-applied from ~/.hermes/scripts/apply-improvements.sh
    # Do not remove this block — it loads adaptive improvements from ~/.hermes/improvements/
    try:
        import sys as _sys, os as _os
        _home = _os.environ.get("HERMES_HOME", _os.path.expanduser("~/.hermes"))
        _parent = _os.path.dirname(_os.path.join(_home, "improvements"))
        if _parent not in _sys.path:
            _sys.path.insert(0, _parent)
        from improvements.integration import patch_agent_for_improvements
        patch_agent_for_improvements(agent)
    except Exception as _e:
        import logging as _logging
        _logging.getLogger("run_agent").warning(
            "Hermes improvements not loaded: %s", _e
        )
"""


def find_agent_init() -> str | None:
    """Find agent_init.py in site-packages or pipx venv."""
    import glob

    home_dir = Path.home()

    # Priority 1: pipx venv (Hermes installed via pipx)
    pipx_venvs = home_dir / ".local/share/pipx/venvs"
    if pipx_venvs.exists():
        for pyver in ["3.12", "3.11", "3.10", "3.13"]:
            candidate = pipx_venvs / "hermes-agent" / "lib" / f"python{pyver}" / "site-packages" / "agent" / "agent_init.py"
            if candidate.exists():
                return str(candidate)
        # Fallback: glob orice python version
        for f in glob.glob(str(pipx_venvs / "hermes-agent/lib/python*/site-packages/agent/agent_init.py")):
            return f

    # Priority 2: via Python's site module
    try:
        import site
        for sp in site.getsitepackages():
            candidate = Path(sp) / "agent" / "agent_init.py"
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass

    # Priority 3: source checkout (~/.hermes/hermes-agent/)
    source_checkout = home_dir / ".hermes" / "hermes-agent" / "agent" / "agent_init.py"
    if source_checkout.exists():
        return str(source_checkout)

    # Fallback: common paths
    candidates = [
        home_dir / ".local/lib/python3.12/site-packages/agent/agent_init.py",
        Path("/usr/local/lib/python3.12/site-packages/agent/agent_init.py"),
        Path("/usr/lib/python3.12/site-packages/agent/agent_init.py"),
        Path("/usr/lib/python3/dist-packages/agent/agent_init.py"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    return None


def _find_function_end(lines: list[str], func_def_prefix: str) -> int | None:
    """
    Find the last line index of a function definition.

    First finds the start of the function, scans past the parameter list
    until the colon closing the header, and then finds the last line of the body.
    """
    # Find the beginning of the function
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(func_def_prefix):
            start_idx = i
            break

    if start_idx is None:
        return None

    # Step 1: Find where the function signature header actually ends (the colon after args)
    body_start_idx = None
    for i in range(start_idx, len(lines)):
        stripped = lines[i].strip()
        if stripped.endswith(':'):
            body_start_idx = i + 1
            break

    if body_start_idx is None:
        return None

    # Step 2: Find the first unindented non-empty line AFTER the body has started
    for i in range(body_start_idx, len(lines)):
        stripped = lines[i].strip()
        if stripped and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
            return i - 1

    return len(lines) - 1


def inject_hook(filepath: str) -> bool:
    """Inject the hook into agent_init.py at the end of init_agent()."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Already applied?
    if MARKER in content:
        return False  # No changes needed

    lines = content.split('\n')

    # Find end of init_agent() function (P0 fix: use function end, not __all__)
    func_end = _find_function_end(lines, 'def init_agent(')
    if func_end is None:
        print("[improvements] ⚠️ Could not find init_agent() — falling back to append at end")
        new_content = content.rstrip('\n') + '\n\n' + HOOK_CODE + '\n'
    else:
        # Insert after the last line of init_agent()
        # Use the same indentation as the function body (4 spaces from HOOK_CODE)
        before = '\n'.join(lines[:func_end + 1])
        after = '\n'.join(lines[func_end + 1:])
        new_content = before.rstrip('\n') + '\n' + HOOK_CODE + '\n' + after.lstrip('\n')

    # Create backup with PID to avoid same-second collision (P1a fix)
    import time
    backup = f"{filepath}.bak.{time.strftime('%Y%m%d_%H%M%S')}.{os.getpid()}"
    # Reuse already-read 'content' instead of re-reading from disk (P1a fix)
    with open(backup, 'w') as f:
        f.write(content)

    # Write modified content
    with open(filepath, 'w') as f:
        f.write(new_content)

    # Verify
    with open(filepath, 'r') as f:
        if MARKER in f.read():
            print(f"[improvements] ✅ Hook injected at {filepath}")
            print(f"[improvements] Backup saved to {backup}")
            return True
        else:
            # Restore from backup
            with open(backup, 'r') as fb:
                with open(filepath, 'w') as fw:
                    fw.write(fb.read())
            print(f"[improvements] ❌ Injection failed, restored from backup")
            return False


def main():
    filepath = find_agent_init()
    if not filepath:
        print("[improvements] ERROR: Could not find agent_init.py")
        sys.exit(1)

    print(f"[improvements] Found agent_init.py at: {filepath}")

    changed = inject_hook(filepath)
    if changed:
        print("[improvements] ✅ Hook applied successfully")
    else:
        print("[improvements] Hook already present — nothing to do")

    # Verify improvements package can be imported
    hermes_dir = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser().resolve()
    sys.path.insert(0, str(hermes_dir))
    try:
        import improvements
        print(f"[improvements] ✅ Package found: {improvements.__file__}")
        print(f"[improvements] Version: {improvements.__version__}")
    except ImportError as e:
        print(f"[improvements] ❌ Import failed: {e}")
        sys.exit(1)

    print("[improvements] ✅ All checks passed. Improvements active.")


if __name__ == "__main__":
    main()
