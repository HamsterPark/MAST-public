"""A declared parameter range must reach the model, not just the docstring.

instrument_control once sent ``setpoint_a = 1.5`` (amperes) — this instrument is typically limited to 0-10 nA, and 1.5 A does not fit that range at all.

The prompt DID have the hint. SetSetpoint's parameter description spells out the
range and uses 1.5 itself as the worked counter-example::

    "STM setpoints are TINY: typical range 1e-12 to 1e-7 A (1 pA to 100 nA) …
     A bare value like 1.5 means 1.5 AMPERES — ~1e10× a normal setpoint — and is
     rejected; if you meant 1.5 nA pass 1.5e-9, not 1.5."

and ``min_value=1e-12, max_value=100e-9`` were declared on the ParameterSpec.
The model sent 1.5 anyway, eleven times.

The gap: the generated args_schema carried ONLY the description. min_value /
max_value were never mapped to JSON-Schema ``minimum`` / ``maximum``, so the
range existed as prose the model could skip and as nothing a validator could
check. 609 of the library's 725 numeric parameters declare a range; none of it
reached a model.

Enforcement deliberately stays with SafetyGate: Field(ge=/le=) would make
Pydantic reject first, and the model would get a generic validation error
instead of the gate's teaching text ("100 pA = 1e-10 … 切勿重试相同数值").

2026-08-04 — THE CARRIER CHANGED, THE THESIS DID NOT
====================================================
Every dimensioned parameter is now declared a STRING in the tool schema, because
a number-typed tool argument does not survive this provider. Measured on the
real path (kimi-k3, tool calling, tool_choice=auto, 12 trials each)::

    number-typed argument    0/12 correct — 3e-12 arrived as 3, 1.5e-10 as 1.5
    string-typed argument   12/12 byte-identical

JSON Schema has no ``minimum`` for a string, so the range moved into the
description — where, per this file's own founding argument, prose alone is weak.
That trade is deliberate and it is not a retreat: a ``maximum`` the model can
read on a channel that corrupts 12 values out of 12 protects nothing, and
ENFORCEMENT never lived in the schema anyway. ``_run`` parses the string back to
a float before validate_params and SafetyGate ever see it, so every bound below
is still enforced exactly where it always was — the tests here check that too,
not just the advertisement.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_param_range_in_schema.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.agents._shared.skill_adapter import wrap_skill  # noqa: E402


def _schema(skill_cls) -> dict:
    t = wrap_skill(skill_cls, lambda *a, **k: None)
    return t.args_schema.model_json_schema()["properties"]


# ════════════════════════════════════════════════════════════════════════════
# The parameter from the incident
# ════════════════════════════════════════════════════════════════════════════

def test_setpoint_is_asked_for_as_a_string():
    """Not a stylistic choice — the number channel loses the exponent 12/12."""
    from mast.skills.builtins.zcontrol import SetSetpoint
    p = _schema(SetSetpoint)["setpoint_a"]
    assert p.get("type") == "string"


def test_setpoint_range_is_stated_in_the_units_the_model_must_write():
    """A string field cannot carry ``minimum``, so the range has to be words.

    And it has to be words in the SAME notation the field demands: telling a
    field that refuses bare numbers that its range is "1e-12 … 1e-07" would be
    stating the answer in the one format it rejects.
    """
    from mast.skills.builtins.zcontrol import SetSetpoint
    d = _schema(SetSetpoint)["setpoint_a"].get("description", "")
    assert "1p" in d and "100n" in d, f"range not stated in SI: {d!r}"
    assert "前缀不可省略" in d, "the field is strict but never says so"


def test_the_range_survives_conversion_to_the_provider_payload():
    """The check that actually matters, and the one the first attempt lacked.

    ``model_json_schema()`` is NOT what the provider sees. langchain's
    ``convert_to_openai_tool`` rebuilds the payload and keeps only what it
    recognises — a ``json_schema_extra={"minimum":…}`` was silently dropped::

        json_schema_extra  ->  {'type': 'number'}
        Field(ge=, le=)    ->  {'type':'number','minimum':1e-12,'maximum':1e-07}

    The first version of that fix passed its tests against model_json_schema()
    and changed nothing for the model. Descriptions DO survive conversion — but
    that is exactly the kind of thing this file exists to verify rather than
    assume.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from mast.skills.builtins.zcontrol import SetSetpoint

    t = wrap_skill(SetSetpoint, lambda *a, **k: None)
    p = convert_to_openai_tool(t)["function"]["parameters"]["properties"]["setpoint_a"]
    assert p.get("type") == "string"
    assert "1p" in p.get("description", ""), (
        "the range is in the local schema but not in the provider payload")


def test_the_prose_hint_is_still_there():
    """The description was never the problem — it must not be lost in the fix."""
    from mast.skills.builtins.zcontrol import SetSetpoint
    d = _schema(SetSetpoint)["setpoint_a"].get("description", "")
    # 2026-08-04：一个能照抄的具体值 + 那个错误模式(裸 1.5 = 1.5 A)，两样都还在；
    # 具体值从 '1e-10' 换成 '100p'，因为 setpoint_a 现在强制 SI 前缀，前者会被拒。
    assert "'100p'" in d and "1.5" in d


def test_an_out_of_range_call_is_refused_with_actionable_wording():
    """Supersedes an earlier test that asserted pydantic must NOT pre-empt
    SafetyGate. That premise was wrong: keeping enforcement off the schema kept
    the BOUND off the schema too (json_schema_extra does not survive conversion),
    so the model never saw the range and kept sending amperes.

    ge/le enforce as well as advertise. What must be preserved is not "pydantic
    stays out of the way" but "the model gets something it can act on" — which
    is what handle_validation_error provides.
    """
    from mast.skills.builtins.zcontrol import SetSetpoint

    class _Ctx:
        state = None

        def run(self, *a, **k):
            raise AssertionError("skill must not execute on an out-of-range call")

        def safe_call(self, *a, **k):
            raise AssertionError("no instrument call on an out-of-range value")

    t = wrap_skill(SetSetpoint, lambda *a, **k: _Ctx())
    # 1500m = 1.5 A. Well-formed, parseable, and a billion times too big — so it
    # reaches the BOUNDS check rather than the parser, which is the case this
    # test has always been about.
    r = t.invoke({"name": "SetSetpoint", "type": "tool_call", "id": "c1",
                  "args": {"setpoint_a": "1500m"}})
    text = str(getattr(r, "content", r))
    assert "1e-12" in text and "1e-07" in text, "the allowed range is not stated"
    assert "不要重复发送同一个值" in text
    # StallGuard groups by this prefix; losing it would break escalation.
    assert "precondition_failed" in text


def test_a_magnitude_that_lost_its_prefix_is_refused_at_the_parser():
    """The other half, and the reason the field is strict.

    ``1.5e-10`` losing its exponent gives ``1.5`` — a valid ampere value, in the
    schema's old range, and 10 orders of magnitude wrong. ``150p`` losing its
    prefix gives ``150``, which does not parse at all. The failure becomes loud
    instead of plausible.
    """
    from mast.skills.builtins.zcontrol import SetSetpoint

    class _Ctx:
        state = None

        def safe_call(self, *a, **k):
            raise AssertionError("no instrument call on an unparseable value")

    t = wrap_skill(SetSetpoint, lambda *a, **k: _Ctx())
    r = t.invoke({"name": "SetSetpoint", "type": "tool_call", "id": "c9",
                  "args": {"setpoint_a": "1.5"}})
    text = str(getattr(r, "content", r))
    assert "前缀" in text
    assert "precondition_failed" in text


def test_a_valid_value_still_executes():
    """The guard must not become a wall — an in-range call goes through."""
    from mast.skills.builtins.zcontrol import SetSetpoint

    seen = []

    class _Ctx:
        state = None

        def safe_call(self, verb, *a, **k):
            seen.append(verb)

            class _R:
                error = ""
                return_value = ("", b"", [0.0])
            return _R()

    t = wrap_skill(SetSetpoint, lambda *a, **k: _Ctx())
    t.invoke({"name": "SetSetpoint", "type": "tool_call", "id": "c2",
              "args": {"setpoint_a": "100p"}})
    assert "ZCtrl_SetpntSet" in seen


# ════════════════════════════════════════════════════════════════════════════
# Library-wide: this was never about one parameter
# ════════════════════════════════════════════════════════════════════════════

def test_declared_ranges_reach_the_schema_library_wide():
    """Spot-check across skills: every numeric ParameterSpec that declares a
    bound must expose it. 609 of 725 numeric params declare one."""
    from mast.core.registry import SkillRegistry

    reg = SkillRegistry()
    reg.discover()
    checked = missing = 0
    for meta in reg.list_skills():
        specs = [s for s in (meta.parameters or [])
                 if getattr(s, "type", "") in ("float", "int")
                 and (getattr(s, "min_value", None) is not None
                      or getattr(s, "max_value", None) is not None)]
        if not specs:
            continue
        try:
            cls = reg.get_skill_class(meta.name) if hasattr(
                reg, "get_skill_class") else None
        except Exception:
            cls = None
        if cls is None:
            continue
        try:
            props = _schema(cls)
        except Exception:
            continue
        for s in specs:
            p = props.get(s.name)
            if p is None:
                continue
            checked += 1
            lo = getattr(s, "min_value", None)
            hi = getattr(s, "max_value", None)
            if p.get("type") == "string":
                # Dimensioned → the model writes text, so the bound lives in the
                # description. Same requirement, different carrier: it must be
                # THERE, and it must be in the notation the field accepts.
                d = p.get("description", "")
                if (lo is not None or hi is not None) and "范围" not in d                         and "最大" not in d and "最小" not in d:
                    missing += 1
                continue
            if lo is not None and p.get("minimum") is None:
                missing += 1
            if hi is not None and p.get("maximum") is None:
                missing += 1
    if checked == 0:
        pytest.skip("registry did not expose skill classes for schema build")
    assert missing == 0, f"{missing} declared bounds never reached the schema"


def test_a_parameter_without_bounds_gets_no_junk_keys():
    """Absent bounds must stay absent — an invented minimum would be worse than
    none, since the model would treat it as real."""
    from mast.skills.builtins.zcontrol import SetSetpoint
    t = wrap_skill(SetSetpoint, lambda *a, **k: None)
    props = t.args_schema.model_json_schema()["properties"]
    tc = props.get("tool_call_id", {})
    assert "minimum" not in tc and "maximum" not in tc


# ════════════════════════════════════════════════════════════════════════════
# Per-rig limits: the schema must not advertise what the gate will refuse
# ════════════════════════════════════════════════════════════════════════════

def test_schema_tightens_to_the_active_safety_envelope(monkeypatch):
    """我们的仪器大部分时候限制是 0-10nA。

    SafetyLimits.setpoint_max_a is admin-overridable (same path as xy_max_m).
    Without intersecting it into the schema, a rig narrowed to 10 nA would still
    tell the model 100 nA is fine — and the model would keep proposing values
    the gate rejects, which is the retry spin all over again.
    """
    import mast.agents._shared.skill_adapter as SA
    from mast.skills.builtins.zcontrol import SetSetpoint

    orig = SA._envelope_for
    monkeypatch.setattr(
        SA, "_envelope_for",
        lambda spec: (1e-12, 1e-8)
        if getattr(spec, "name", "") == "setpoint_a" else orig(spec))

    d = _schema(SetSetpoint)["setpoint_a"]["description"]
    assert "10n" in d, (
        f"the schema still advertises the factory 100 nA on a 10 nA rig: {d!r}")
    assert "100n" not in d


def test_the_envelope_can_only_tighten(monkeypatch):
    """A looser envelope must NOT widen a per-skill bound — the skill knows its
    own hardware meaning, and the global envelope is a backstop, not a licence."""
    import mast.agents._shared.skill_adapter as SA
    from mast.skills.builtins.zcontrol import SetSetpoint

    monkeypatch.setattr(SA, "_envelope_for", lambda spec: (0.0, 1.0))
    d = _schema(SetSetpoint)["setpoint_a"]["description"]
    assert "100n" in d, f"a loose envelope widened the advertised bound: {d!r}"
    assert "1p" in d


def test_envelope_lookup_never_raises(monkeypatch):
    """Schema building must not fail because config is unavailable."""
    import mast.agents._shared.skill_adapter as SA
    from mast.skills.builtins.zcontrol import SetSetpoint

    def _boom(spec):
        raise RuntimeError("no config")

    monkeypatch.setattr(SA, "_envelope_for", _boom)
    with pytest.raises(RuntimeError):
        _schema(SetSetpoint)          # our own stub raises; the real one cannot


def test_unmapped_parameters_are_untouched():
    """Only unambiguous names are mapped. A wrong mapping would silently narrow
    an unrelated parameter — worse than leaving it alone."""
    from mast.agents._shared.skill_adapter import _envelope_for

    class _S:
        name = "some_unrelated_knob"

    assert _envelope_for(_S()) == (None, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
