"""Contract: every private attribute the API layer reaches for on the live app
must actually exist on ``CoreRuntime``.

WHY THIS TEST EXISTS
------------------------------------------------------
``api/routes/agents_control.py`` resolved HITL interrupts by calling
``getattr(app, "_build_decision", None)`` and degrading when absent. That method
lived on ``MASTApp`` in ``mast/gui/app.py``; commit ``7aa1996`` deleted the
Gradio layer **without migrating it**. ``CoreRuntime`` never had it, so the
lookup returned ``None`` on every call and every approval answered::

    {"status": "degraded", "detail": "live decision builder unavailable"}

For ~5 weeks **no human approval worked at all** — DANGEROUS skill gates,
composite workflow human nodes, and ``buffer_hitl`` for CRITICAL hardware events
(tip_quality_drop / E_STOP / retract_needed) alike. An operator's click reached
the backend and got a well-formed reply; it simply never woke the blocked
worker. The run stayed stuck, pinning a stream worker, which is what made the
whole group chat appear "blocked" .

The unit tests did not catch it because the fake live app **defined its own**
``_build_decision`` — asserting a contract no real object satisfied. Green, and
worthless.

This test does not care about HITL specifically: it asserts the *shape* of the
relay contract, so the next attribute lost in a refactor fails here instead of
silently degrading in the field.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def _find_mastv2_root() -> Path:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return p / "MASTv2"
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != str(_MASTV2_ROOT):
    while str(_MASTV2_ROOT) in sys.path:
        sys.path.remove(str(_MASTV2_ROOT))
    sys.path.insert(0, str(_MASTV2_ROOT))
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402

# Variable names that hold the live core inside API handlers.
_LIVE_VARS = {"app", "live_app", "rt", "runtime", "core"}

# Attributes that are legitimately optional / not owned by CoreRuntime.
# Keep this list SHORT and justified — every entry is a hole in the contract.
_ALLOWED_ABSENT = {
    # Set by the GUI-era app only; API treats absence as "no orchestrator yet"
    # and degrades on purpose (documented house rule 2).
    "_orchestrator",
}


def _iter_api_files() -> list[Path]:
    return sorted((_MASTV2_ROOT / "mast" / "api").rglob("*.py"))


def _core_runtime_attrs() -> set[str]:
    """Every attribute a live ``CoreRuntime`` can carry.

    ``hasattr(CoreRuntime, x)`` alone is NOT enough: most of what the API layer
    legitimately reaches for (``_storage``, ``_state``, ``_pool`` …) is an
    INSTANCE attribute assigned in ``__init__``, invisible on the class. So we
    union the class namespace with every ``self._x = …`` target found in the
    class body — that is the set of names a running instance may expose.
    """
    names = {n for n in dir(CoreRuntime)}
    src = (_MASTV2_ROOT / "mast" / "core" / "runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "CoreRuntime"):
            continue
        for sub in ast.walk(node):
            targets: list[ast.expr] = []
            if isinstance(sub, ast.Assign):
                targets = list(sub.targets)
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                targets = [sub.target]
            # Flatten tuple/list unpacking: ``self._a, self._b = f()`` gives an
            # ast.Tuple target, so a scan that only looks at ast.Attribute misses
            # BOTH names. That is not hypothetical — ``self._v2_repos,
            # self._v2_eid = open_live_v2()`` is exactly this shape, and an
            # honest getattr for _v2_repos was reported as dangling (2026-07-27).
            flat: list[ast.expr] = []
            for t in targets:
                if isinstance(t, (ast.Tuple, ast.List)):
                    flat.extend(t.elts)
                else:
                    flat.append(t)
            for t in flat:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    names.add(t.attr)
            # ``setattr(self, "_x", …)`` is used for a few lazily wired handles.
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id == "setattr" and len(sub.args) >= 2
                    and isinstance(sub.args[0], ast.Name)
                    and sub.args[0].id == "self"
                    and isinstance(sub.args[1], ast.Constant)
                    and isinstance(sub.args[1].value, str)):
                names.add(sub.args[1].value)
    return names


def _dangling_getattrs() -> list[tuple[str, int, str]]:
    """Return (file, lineno, attr) for every getattr(<live>, "_attr") whose
    attr is missing from CoreRuntime."""
    known = _core_runtime_attrs()
    hits: list[tuple[str, int, str]] = []
    for f in _iter_api_files():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except (SyntaxError, OSError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2):
                continue
            target, attr = node.args[0], node.args[1]
            if not (isinstance(target, ast.Name) and target.id in _LIVE_VARS):
                continue
            if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
                continue
            name = attr.value
            if not name.startswith("_"):
                continue
            if name in _ALLOWED_ABSENT:
                continue
            if name not in known:
                rel = f.relative_to(_MASTV2_ROOT).as_posix()
                hits.append((rel, node.lineno, name))
    return hits


def test_api_layer_never_reaches_for_a_nonexistent_core_attribute():
    """The regression that broke every HITL approval for 5 weeks.

    A relay that asks the live core for something it does not have degrades
    silently and forever — the worst failure shape, because the UI still
    renders, the request still succeeds, and only the *effect* is missing.
    """
    dangling = _dangling_getattrs()
    assert not dangling, (
        "API layer reaches for attributes CoreRuntime does not define — these "
        "degrade silently at runtime:\n"
        + "\n".join(f"  {f}:{ln}  getattr(app, {a!r})" for f, ln, a in dangling)
        + "\n\nEither implement it on CoreRuntime, or stop gating behaviour on it."
    )


@pytest.mark.parametrize("attr", ["agents_interrupts", "emergency_stop"])
def test_core_runtime_still_exposes_the_read_side(attr):
    """Guard the counterpart: the read path that DID survive the migration."""
    assert hasattr(CoreRuntime, attr), f"CoreRuntime lost {attr}"


def test_hitl_translation_lives_in_core_not_behind_a_live_lookup():
    """Verdict translation is pure — it must not be reachable only via a live
    object, which is precisely how it went missing."""
    from mast.core import hitl_decision

    for fn in ("build_decision", "enforce_allowed", "build_workflow_route",
               "coerce_arg_value", "canonical_verdict"):
        assert callable(getattr(hitl_decision, fn, None)), f"missing {fn}"
