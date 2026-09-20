"""``safe_call`` must be given a LITERAL verb. Every safety tool in this repo depends on it.

Nothing in MAST parses Python to find its Nanonis calls. The abort-policy checker,
the security audit and the API-coverage census all do the same cheap thing: grep for
``safe_call("…")`` and read the string. So a call written as

    for key, verb, args in TABLE:
        rec = context.safe_call(verb, *args)     # ← invisible

still reaches the instrument, and **not one of the tools that guard this system can
see it**. The call is outside the safety net while looking exactly like it is inside.

This has now happened twice.

  * 2026-07-12 — ``function_generator.py`` resolved its stop verb through a variable.
    The abort-policy property test then reported ``FunGen1Ch_Stop`` as a *phantom*
    allow-list entry: a verb on the post-abort allow-list that (as far as the tool
    could tell) nothing called. The allow-list entry was right; the tool was blind.
  * 2026-07-13 — ``readback.py`` and six other new files used table-driven loops. The
    coverage census duly reported 40 settings MAST "could not read back" — every one
    of which it could. The number that was supposed to measure the fix was measuring
    the shape of the code instead.

Both times the *analysis* was wrong in a way that looked like a *finding*, which is
the expensive kind of wrong. A test is cheaper.

WHAT TO DO INSTEAD
==================
Two literal branches, or a thunk that holds the literal::

    poll = lambda: context.safe_call("Current_Get")            # grep sees it
    ...
    for key, thunk in (("z", lambda: context.safe_call("ZCtrl_ZPosGet")), …):
        rec = thunk()

The verb stays inside the ``safe_call(...)`` call expression, which is all the tools
need. It costs a few lines and it keeps every guard in this system honest.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4] / "MASTv2" / "mast"

# safe_call's own IMPLEMENTATIONS necessarily take the verb as a parameter — they are
# the plumbing, not a call site. Same for the thin pool wrappers that forward to it.
_IMPLEMENTATIONS = {
    ("core", "execution_context.py"),
    ("core", "executor.py"),
    # ConnectionPool itself: safe_call is defined here and urgent_call forwards
    # to it. (This entry used to name "connection_pool.py", a file that has
    # never existed in v2 — so the exemption covered nothing.)
    ("core", "connection.py"),
    ("vision", "scan_monitor.py"),   # _call() wrapper: forwards to the pool
}

#: Both entry points onto the Nanonis link. ``urgent_call`` is the emergency
#: path added 2026-07-28 (bounded role-lock wait + force-unstick); its verbs
#: must stay just as greppable as ``safe_call``'s, or the abort-policy checker
#: and the safety audits would stop seeing the three verbs the E-STOP issues.
_CALL_NAMES = {"safe_call", "urgent_call"}


def _is_implementation(path: Path) -> bool:
    parts = path.parts
    return any(pkg in parts and path.name == fname for pkg, fname in _IMPLEMENTATIONS)


def _offenders() -> list[str]:
    """Every safe_call whose first positional argument is not a string literal."""
    bad: list[str] = []
    for py in sorted(_ROOT.rglob("*.py")):
        if _is_implementation(py):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else \
                (fn.id if isinstance(fn, ast.Name) else None)
            if name not in _CALL_NAMES or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                continue   # a literal — good
            rel = py.relative_to(_ROOT.parent)
            shown = ast.unparse(first) if hasattr(ast, "unparse") else "<expr>"
            bad.append(f"{rel}:{node.lineno}  {name}({shown}, …)")
    return bad


def test_the_scan_is_not_vacuous():
    """A property test that finds nothing because it looked nowhere is worse than none."""
    total = 0
    for py in _ROOT.rglob("*.py"):
        if _is_implementation(py):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else \
                    (fn.id if isinstance(fn, ast.Name) else None)
                if nm in _CALL_NAMES:
                    total += 1
    assert total > 400, f"只找到 {total} 处 safe_call —— AST 扫描坏了"


def test_every_safe_call_verb_is_a_string_literal():
    bad = _offenders()
    assert not bad, (
        "这些 safe_call 的动词藏在变量里，本仓库的**每一个安全工具**都看不见它们"
        "（中止策略检查 / 安全审计 / API 覆盖率普查全靠 grep `safe_call(\"…\")`）：\n"
        + "\n".join(f"  {b}" for b in bad)
        + "\n\n改成两个字面量分支，或一个持有字面量的 thunk：\n"
          '  poll = lambda: context.safe_call("Current_Get")\n'
          '  for key, thunk in (("z", lambda: context.safe_call("ZCtrl_ZPosGet")), …):\n'
          '      rec = thunk()\n'
    )


@pytest.mark.parametrize("tool", ["abort-policy", "coverage-census"])
def test_the_tools_that_depend_on_this_still_see_everything(tool):
    """Pin the consequence, not just the rule: the grep the other tools use must find
    a verb we know is called from a thunk-style loop."""
    import re
    pat = re.compile(r'safe_call\(\s*"([A-Za-z0-9_]+)"')
    seen: set[str] = set()
    for py in _ROOT.rglob("*.py"):
        seen |= set(pat.findall(py.read_text(encoding="utf-8", errors="replace")))

    # Verbs that ONLY exist inside a thunk table. If the thunk pattern ever regresses
    # to `safe_call(verb)`, these vanish from the grep and the tools go blind again.
    for verb in ("ZCtrl_OnOffGet",          # readback.GetZControllerState
                 "HSSwp_SwpChSigListGet",   # optional_sweepers (table)
                 "Current_Get",             # capture_signal_buffer (thunk branch)
                 "SpectrumAnlzr_BandRMSGet"):
        assert verb in seen, (
            f"{tool} 用的 grep 看不见 {verb} —— 它被写回了变量形式，"
            "安全工具又瞎了。"
        )
