"""v2 unit tests for mast.skills.builtins.navigation.

Skills covered: MoveToXY (CONFIRM) — 1 skill total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_navigation.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from dataclasses import dataclass, field
from typing import Any

import pytest

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.types import NanonisCallRecord
from mast.skills.builtins.navigation import MoveToXY


# ── FakeCtx ──────────────────────────────────────────────────────────────────

@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, tuple]] = field(default_factory=list)

    def safe_call(self, method: str, *args, role: str = "main") -> NanonisCallRecord:
        self.calls.append((method, args))
        if method in self.canned:
            entry = self.canned[method]
            return NanonisCallRecord(
                method=method, args=args,
                return_value=entry.get("return_value"),
                error=entry.get("error", ""),
            )
        return NanonisCallRecord(method=method, args=args, error=f"unmocked: {method}")


def make_provider(canned: dict[str, Any] | None = None):
    canned = canned or {}
    return lambda: FakeCtx(canned=canned)


def _invoke(tool, **kwargs) -> Any:
    return tool.func(tool_call_id="test-call-1", state={}, **kwargs)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_move_to_xy_shape():
    tool = wrap_skill(MoveToXY, make_provider())
    assert tool.name == "MoveToXY"
    assert tool.metadata["danger_level"] == "CONFIRM"
    fields = tool.args_schema.model_fields
    assert "x_m" in fields
    assert "y_m" in fields
    assert fields["x_m"].is_required()
    assert fields["y_m"].is_required()
    assert "wait" in fields
    assert not fields["wait"].is_required()
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter）
    assert fields["x_m"].annotation is str
    # 有量纲 → 模型侧是 SI 字符串（skill_adapter）
    assert fields["y_m"].annotation is str


def test_skill_source_points_to_navigation_module():
    tool = wrap_skill(MoveToXY, make_provider())
    assert tool.metadata["skill_source"].endswith(".navigation")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_move_to_xy_executes():
    canned = {
        "FolMe_XYPosSet": {"return_value": ("", b"", [])},
        "FolMe_XYPosGet": {"return_value": ("", b"", [1e-8, 2e-8])},
    }
    ctx = FakeCtx(canned=canned)
    tool = wrap_skill(MoveToXY, lambda: ctx)
    result = _invoke(tool, x_m=1e-8, y_m=2e-8)
    update = result.update
    assert update["executed_skills"] == ["MoveToXY"]
    assert update["messages"][0].status == "success"
    assert not update.get("error_log")
    assert ("FolMe_XYPosGet", (1,)) in ctx.calls


def test_move_to_xy_calls_correct_method():
    """移动指令以 wait=0 下发，避免对端等待运动完成时占用通信连接。

    到位状态由本技能轮询 FolMe_XYPosGet 确认；本测试核验下发参数，
    等待和超时边界由 test_move_to_xy_does_not_block_the_socket.py 覆盖。
    """
    canned = {"FolMe_XYPosSet": {"return_value": ("", b"", [])},
              "FolMe_XYPosGet": {"return_value": ("", b"", [1e-8, 2e-8])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(MoveToXY, capturing_provider)
    _invoke(tool, x_m=1e-8, y_m=2e-8, wait=True)
    last_ctx = instances[-1]
    pos_calls = [c for c in last_ctx.calls if c[0] == "FolMe_XYPosSet"]
    assert len(pos_calls) == 1
    # wait 参数恒为 0:等待由我们自己轮询,不交给对端阻塞。
    assert pos_calls[0][1] == (1e-8, 2e-8, 0)


def test_move_to_xy_missing_required():
    tool = wrap_skill(MoveToXY, make_provider())
    result = _invoke(tool)  # missing x_m + y_m
    msg_content = result.update["messages"][0].content
    assert (
        "precondition_failed" in msg_content
        or "Missing required" in msg_content
        or "x_m" in msg_content
    )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
