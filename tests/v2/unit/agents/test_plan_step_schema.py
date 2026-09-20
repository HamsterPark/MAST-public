"""`show_plan_on_map` must TELL the model the scale, not only judge it after.

. The signature was ``steps: list`` — the provider
payload carried ``{"steps": {"type": "array"}}`` and nothing else: no field
names, no units, no bounds. The same agent, in the same turn, wrote
``FullScan(center_x_m=1.4e-06, height_m=5e-08)`` correctly (typed floats WITH
bounds) and ``{"x_m": 1.4, "w_m": 5.0}`` here. It then decided the exponent was
being eaten in transit, retried twice in plain decimal, got it wrong again, and
no route ever reached the map.

Two properties, and they are in tension — which is why both are pinned:

1. **The bounds are IN THE PAYLOAD.** Description text alone was not enough for
   the sibling case (the 1.5 A setpoint); a machine-readable
   ``minimum``/``maximum`` was. Note these are ``json_schema_extra``, not
   ``ge``/``le``: at the top level of a tool signature langchain drops
   ``json_schema_extra``, but inside a nested model it survives.
2. **Pydantic does NOT own the rejection.** ``_check_plan_step_magnitudes``
   phrases it — step index AND label, how far out, the µm/nm readings as
   arithmetic, and an explicit disclaimer about intent (). ``ge``/``le``
   would preempt all of that with "Input should be less than or equal to 1e-05".

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_plan_step_schema.py -q
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

import json  # noqa: E402

import pytest  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from mast.agents._shared.meta_tools import make_meta_tools  # noqa: E402

#: The exact payload the field agent sent three times.
FIELD_STEP = {"kind": "scan", "label": "+1V 50nm",
              "w_m": 5.0, "x_m": 1.4, "y_m": 1.4}


@pytest.fixture()
def tool():
    return next(t for t in make_meta_tools(lambda: {})
                if t.name == "show_plan_on_map")


@pytest.fixture()
def item_schema(tool):
    params = convert_to_openai_tool(tool)["function"]["parameters"]
    steps = params["properties"]["steps"]
    assert steps.get("type") == "array"
    items = steps.get("items")
    assert items, "steps 仍是无 item schema 的裸数组 —— 模型什么都看不到"
    return items


# ════════════════════════════════════════════════════════════════════════════
# 1. What the model is actually shown
# ════════════════════════════════════════════════════════════════════════════

def test_every_step_field_is_named_in_the_payload(item_schema):
    props = item_schema.get("properties") or {}
    for field in ("kind", "x_m", "y_m", "w_m", "h_m", "label"):
        assert field in props, f"{field} 不在 payload 里 —— 模型只能猜"


@pytest.mark.parametrize("field,ceiling", [
    ("x_m", 1e-3), ("y_m", 1e-3), ("w_m", 1e-5), ("h_m", 1e-5),
])
def test_numeric_bounds_reach_the_provider(item_schema, field, ceiling):
    """A constraint only exists if it is in the payload. This is the property
    that fixed the sibling `1.5 A` setpoint; description text alone was not."""
    spec = item_schema["properties"][field]
    assert spec.get("maximum") == pytest.approx(ceiling), f"{field} 缺 maximum"
    assert spec.get("minimum") == pytest.approx(-ceiling), f"{field} 缺 minimum"


def test_the_scale_is_stated_in_words_too(item_schema):
    """Bounds say what is illegal; the description says what is TYPICAL."""
    desc = item_schema["properties"]["x_m"].get("description", "")
    assert "米" in desc
    assert "1.4e-06" in desc, "没给出一个可照抄的正确量级"


def test_the_item_description_is_for_the_model_not_the_maintainer(item_schema):
    """The class docstring becomes the item schema's description and is sent on
    every call — it must not carry the forensics note that explains the bug."""
    desc = item_schema.get("description", "")
    assert len(desc) < 200, f"item description 过长({len(desc)} 字符),白烧 token"
    assert "2026-07-28" not in desc and "feedback" not in desc.lower()


# ════════════════════════════════════════════════════════════════════════════
# 2. Who owns the rejection
# ════════════════════════════════════════════════════════════════════════════

def test_the_field_payload_is_still_rejected_by_the_checker(tool):
    """Not by pydantic — by the checker, with the wording established."""
    out = json.loads(tool.invoke({"steps": [FIELD_STEP]}))
    assert out["success"] is False
    blob = " ".join(out["problems"])
    assert "+1V 50nm" in blob, "步骤标签丢了 —— pydantic 抢走了拒绝权"
    assert "单位没错" in blob and "量级" in blob
    assert "无法判定你的本意" in blob


def test_a_correct_call_still_publishes(tool):
    from mast.io.plan_overlay import get_plan_overlay
    get_plan_overlay().clear()
    out = json.loads(tool.invoke({"steps": [
        {"kind": "scan", "label": "+1V 50nm",
         "x_m": 1.4e-6, "y_m": 1.4e-6, "w_m": 5e-8, "h_m": 5e-8}]}))
    assert out["success"] is True and out["steps"] == 1
    assert len(get_plan_overlay().snapshot()) == 1


def test_plain_dicts_still_work(tool):
    """The JSON route and the tests hand dicts, not PlanStep instances."""
    from mast.io.plan_overlay import get_plan_overlay
    get_plan_overlay().clear()
    out = json.loads(tool.invoke({"steps": [{"x_m": 1e-7, "y_m": 1e-7}]}))
    assert out["success"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
