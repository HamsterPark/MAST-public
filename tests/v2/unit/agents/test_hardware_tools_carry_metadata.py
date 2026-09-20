"""Anything that can reach the instrument must be gateable.

``SafetyGateMiddleware._check`` (safety_mw.py) fails OPEN by design::

    meta_obj = (tool.metadata or {}).get("skill_metadata")
    if meta_obj is None or not isinstance(meta_obj, SkillMetadata):
        return None            # not a skill tool — pass through

That is the right call for hand-off / buffer / analysis tools, and it is the
ONLY reason those work. But it means the whole safety layer — bounds checks,
the coarse-approach block, the DANGEROUS gate, the abort policy — is skipped for
any tool that lacks ``skill_metadata``. A plain ``@tool`` that imported
``mast.instruments`` or drove the ConnectionPool directly would sail past every
one of them while looking exactly like a governed tool.

Nothing enforced that. The safety_invariant 钩子（不随仓） hook checks
that BaseSkill subclasses declare ``safety_level``; there was no check in the
other direction — "if you touch hardware, you must be a skill" (dispatch audit
2026-07-28 严重级「SafetyGate fail-open 分流」). The convention held, but only
by discipline.

This test is that check. It is deliberately a STATIC scan: it must catch the
mistake at review time, and it must not need a live instrument.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_hardware_tools_carry_metadata.py -q
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_MASTV2 = Path(__file__).resolve().parents[4] / "MASTv2"
_ROOT = _MASTV2 / "mast"
if sys.path and sys.path[0] != str(_MASTV2):
    while str(_MASTV2) in sys.path:
        sys.path.remove(str(_MASTV2))
    sys.path.insert(0, str(_MASTV2))


#: Modules whose import means "this code can drive the physical instrument".
_HARDWARE_MODULES = (
    "mast.instruments",
    "mast.core.connection",
    "mast.core.execution_context",
    "mast.core.executor",
)

#: Names that, called on any object, put a command on the wire.
_HARDWARE_CALLS = {"safe_call", "urgent_call", "connect_all", "break_role"}

#: The ONE adapter that attaches ``skill_metadata``. A tool defined here is a
#: wrapped BaseSkill and is therefore fully gated.
_ADAPTER = ("agents", "_shared", "skill_adapter.py")

#: Files that legitimately touch the hardware modules and define ``@tool``s that
#: do NOT — each pairing checked by eye and recorded here rather than left to a
#: blanket exemption. Adding a name here is a deliberate act.
_REVIEWED: dict[str, str] = {
    # Builds/holds the runtime; its @tools are meta/admin, not instrument verbs.
    "core/runtime.py": "runtime owns the pool; its @tools are orchestration glue",
    # The registry's @tool is the skill-listing helper, not a skill.
    "core/registry.py": "registry listing helper",
    # tool_skills bridges REGISTERED skills into menus — the skills themselves
    # carry metadata; this module only re-exports them.
    "skills/composite/tool_skills.py": "bridges wrapped skills, adds no verbs",
}


def _iter_py():
    for p in sorted(_ROOT.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


def _rel(p: Path) -> str:
    return p.relative_to(_ROOT).as_posix()


def _is_adapter(p: Path) -> bool:
    return all(part in p.parts for part in _ADAPTER[:-1]) and p.name == _ADAPTER[-1]


def _parse(p: Path):
    try:
        return ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover
        return None


def _imports_hardware(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if any(a.name.startswith(m) for m in _HARDWARE_MODULES):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod.startswith(m) for m in _HARDWARE_MODULES):
                return True
    return False


def _calls_hardware(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else None)
            if name in _HARDWARE_CALLS:
                return True
    return False


def _tool_defs(tree) -> list[str]:
    """Functions decorated with @tool / @tool(...)."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            nm = (target.attr if isinstance(target, ast.Attribute)
                  else target.id if isinstance(target, ast.Name) else None)
            if nm == "tool":
                out.append(node.name)
                break
    return out


def test_the_scan_is_not_vacuous():
    """A guard that looked nowhere is worse than no guard."""
    files = list(_iter_py())
    assert len(files) > 200, len(files)
    with_tools = [p for p in files
                  if (t := _parse(p)) is not None and _tool_defs(t)]
    assert len(with_tools) > 10, [_rel(p) for p in with_tools]
    hw = [p for p in files if (t := _parse(p)) is not None and _imports_hardware(t)]
    assert len(hw) >= 15, len(hw)


def test_no_plain_tool_lives_in_a_module_that_drives_the_instrument():
    """The reverse of the hook: if you can reach hardware, be a skill.

    A ``@tool`` in a module that imports the instrument stack (or calls
    ``safe_call``/``urgent_call``) has no ``skill_metadata``, so
    SafetyGateMiddleware, the abort policy and the DANGEROUS gate all wave it
    through. Wrap it with ``wrap_skill`` instead, or — if it genuinely does not
    touch hardware — record why in ``_REVIEWED``.
    """
    offenders: list[str] = []
    for p in _iter_py():
        if _is_adapter(p):
            continue          # THE adapter — it attaches the metadata
        tree = _parse(p)
        if tree is None:
            continue
        tools = _tool_defs(tree)
        if not tools:
            continue
        if not (_imports_hardware(tree) or _calls_hardware(tree)):
            continue
        rel = _rel(p)
        if rel in _REVIEWED:
            continue
        offenders.append(f"{rel}: @tool {', '.join(sorted(tools))}")

    assert not offenders, (
        "这些模块能驱动仪器，却在里面直接定义了裸 @tool —— 裸 @tool 没有 "
        "skill_metadata，SafetyGateMiddleware / 中止策略 / DANGEROUS 门控 "
        "会**全部放行**（safety_mw.py:482-485 是 fail-open 的）：\n"
        + "\n".join(f"  {o}" for o in offenders)
        + "\n\n改法：用 BaseSkill + wrap_skill() 包装（这样才带 skill_metadata），"
          "或者确认它确实不碰硬件后，把它记进本测试的 _REVIEWED 并写清理由。"
    )


def test_every_reviewed_exemption_still_exists():
    """An exemption for a file that no longer exists is rot, and rot is how the
    next real offender slips in unnoticed."""
    missing = [rel for rel in _REVIEWED if not (_ROOT / rel).exists()]
    assert not missing, f"_REVIEWED lists files that are gone: {missing}"


def test_the_skill_adapter_really_attaches_metadata():
    """The exemption above is only safe because of this."""
    src = (_ROOT / "agents" / "_shared" / "skill_adapter.py").read_text(
        encoding="utf-8", errors="replace")
    assert '"skill_metadata": meta' in src, (
        "the adapter no longer stamps SkillMetadata onto the tool — the whole "
        "gate keys on that field")
    assert "tool.metadata = {" in src


def test_safety_gate_still_reads_skill_metadata():
    """If the gate ever stops keying on ``skill_metadata``, this whole invariant
    is about the wrong field."""
    src = (_ROOT / "agents" / "_shared" / "safety_mw.py").read_text(
        encoding="utf-8", errors="replace")
    assert 'get("skill_metadata")' in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
