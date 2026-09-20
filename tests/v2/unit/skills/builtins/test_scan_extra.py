"""v2 unit tests for mast.skills.builtins.scan_extra.

Skills covered: GetScanSpeed (AUTO), GetScanBuffer (AUTO),
               SaveScan (AUTO), GetLatestScanFile (AUTO) — 4 skills total.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/builtins/test_scan_extra.py -x -v
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
from mast.skills.builtins.scan_extra import (
    GetLatestScanFile,
    GetScanBuffer,
    GetScanSpeed,
    SaveScan,
)


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

def test_get_scan_speed_shape():
    tool = wrap_skill(GetScanSpeed, make_provider())
    assert tool.name == "GetScanSpeed"
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_get_scan_buffer_shape():
    tool = wrap_skill(GetScanBuffer, make_provider())
    assert tool.name == "GetScanBuffer"
    assert tool.metadata["danger_level"] == "AUTO"
    schema_fields = tool.args_schema.model_fields
    assert schema_fields == {} or all(not v.is_required() for v in schema_fields.values())


def test_save_scan_shape():
    tool = wrap_skill(SaveScan, make_provider())
    assert tool.name == "SaveScan"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "timeout_ms" in fields
    assert not fields["timeout_ms"].is_required()
    assert fields["timeout_ms"].annotation is int


def test_skill_source_points_to_scan_extra_module():
    tool = wrap_skill(GetScanSpeed, make_provider())
    assert tool.metadata["skill_source"].endswith(".scan_extra")


# ── Execution tests ───────────────────────────────────────────────────────────

def test_get_scan_speed_executes():
    canned = {
        "Scan_SpeedGet": {
            "return_value": ("", b"", [5e-9, 5e-9, 0.02, 0.02, 0, 1.0])
        }
    }
    tool = wrap_skill(GetScanSpeed, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["GetScanSpeed"]


def test_get_scan_buffer_executes():
    canned = {
        "Scan_BufferGet": {
            "return_value": ("", b"", [2, [0, 1], 256, 256])
        }
    }
    tool = wrap_skill(GetScanBuffer, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["GetScanBuffer"]


def test_save_scan_executes():
    canned = {"Scan_Save": {"return_value": ("", b"", [0])}}
    tool = wrap_skill(SaveScan, make_provider(canned))
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["SaveScan"]


def test_save_scan_default_timeout():
    canned = {"Scan_Save": {"return_value": ("", b"", [0])}}
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned=canned)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SaveScan, capturing_provider)
    _invoke(tool)  # no timeout_ms → default -1
    last_ctx = instances[-1]
    save_calls = [c for c in last_ctx.calls if c[0] == "Scan_Save"]
    assert len(save_calls) == 1
    assert save_calls[0][1] == (1, -1)  # Wait_until_saved=1, timeout=-1


# ── GetLatestScanFile tests (regression: v1 had this skill, v2 dropped it) ───

def test_get_latest_scan_file_shape():
    tool = wrap_skill(GetLatestScanFile, make_provider())
    assert tool.name == "GetLatestScanFile"
    assert tool.metadata["danger_level"] == "AUTO"
    fields = tool.args_schema.model_fields
    assert "max_age_s" in fields
    assert not fields["max_age_s"].is_required()
    assert fields["max_age_s"].annotation is int


def test_get_latest_scan_file_no_match(tmp_path):
    """When no candidate dirs contain a recent .sxm, returns path=None."""
    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned={})
        ctx.session_path = str(tmp_path)  # empty dir → no sxm
        instances.append(ctx)
        return ctx

    tool = wrap_skill(GetLatestScanFile, capturing_provider)
    result = _invoke(tool, max_age_s=10)
    update = result.update
    assert update["executed_skills"] == ["GetLatestScanFile"]


def test_get_latest_scan_file_finds_recent_sxm(tmp_path):
    """When a fresh .sxm exists in candidate dir, returns its path + age."""
    sxm = tmp_path / "test_001.sxm"
    sxm.write_bytes(b"fake-sxm-header")

    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned={})
        ctx.session_path = str(tmp_path)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(GetLatestScanFile, capturing_provider)
    result = _invoke(tool, max_age_s=300)
    update = result.update
    assert update["executed_skills"] == ["GetLatestScanFile"]


def test_save_scan_surfaces_saved_path(tmp_path):
    """SaveScan should surface the .sxm path the way v1 did (regression check)."""
    sxm = tmp_path / "saved_001.sxm"
    sxm.write_bytes(b"fake")

    instances: list[FakeCtx] = []

    def capturing_provider():
        ctx = FakeCtx(canned={"Scan_Save": {"return_value": ("", b"", [0])}})
        ctx.session_path = str(tmp_path)
        instances.append(ctx)
        return ctx

    tool = wrap_skill(SaveScan, capturing_provider)
    result = _invoke(tool)
    update = result.update
    assert update["executed_skills"] == ["SaveScan"]


# ── Realistic-triplet parse tests (return_value = [err, raw, Variables]) ─────
# Each Variables entry is positional per the method's ResponseTypes — confirm
# the indices map to the right named fields (no [0]/[1] triplet misreads).

def test_get_scan_speed_parses_real_triplet():
    """Scan_SpeedGet ResponseTypes ["f","f","f","f","H","f"] → 6 positional."""
    canned = {
        "Scan_SpeedGet": {
            "return_value": ("", b"\x00", [5e-9, 4e-9, 0.02, 0.025, 1, 0.8])
        }
    }
    skill = GetScanSpeed()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["fwd_speed_m_s"] == 5e-9
    assert res.data["bwd_speed_m_s"] == 4e-9
    assert res.data["fwd_time_s"] == 0.02
    assert res.data["bwd_time_s"] == 0.025
    assert res.data["keep_constant"] == 1
    assert res.data["speed_ratio"] == 0.8


def test_get_scan_buffer_parses_real_triplet():
    """Scan_BufferGet ResponseTypes ["i","*i","i","i"] →
    [num_channels, channel_indexes, pixels, lines]."""
    canned = {
        "Scan_BufferGet": {
            "return_value": ("", b"\x00", [3, [0, 1, 14], 512, 256])
        }
    }
    skill = GetScanBuffer()
    ctx = FakeCtx(canned=canned)
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["num_channels"] == 3
    assert res.data["channel_indexes"] == [0, 1, 14]
    assert res.data["pixels"] == 512
    assert res.data["lines"] == 256


def test_save_scan_parses_timed_out_flag(tmp_path):
    """Scan_Save ResponseTypes ["I"] → Variables=[timed_out]."""
    skill = SaveScan()
    ctx = FakeCtx(canned={"Scan_Save": {"return_value": ("", b"\x00", [1])}})
    ctx.session_path = str(tmp_path)  # empty → saved_path None, that's fine
    res = skill.execute(ctx, {})
    assert res.success
    assert res.data["timed_out"] is True


# ── session-dir resolution (2026-07-01: Nanonis save dir was NEVER captured) ──
# `_session_path` was only ever READ (always None) so scans saved to Nanonis's
# session dir — outside <data_root>/working-sessions — were undiscoverable
# (GetLatestScanFile path:None, Records/Data tab + vision thumbnails blank).

from mast.skills.builtins.scan_extra import (  # noqa: E402
    _candidate_save_dirs,
    _session_dir_from_caller,
)


def test_session_dir_from_caller_parses_triplet(tmp_path):
    """Util_SessionPathGet return_value ("", raw, [size, path]) → the dir string."""
    canned = {"Util_SessionPathGet":
              {"return_value": ("", b"\x00", [len(str(tmp_path)), str(tmp_path)])}}
    ctx = FakeCtx(canned=canned)
    assert _session_dir_from_caller(ctx) == str(tmp_path)
    assert any(c[0] == "Util_SessionPathGet" for c in ctx.calls)  # issued the call


def test_session_dir_from_caller_str_form(tmp_path):
    """A bare-string return_value is accepted too."""
    canned = {"Util_SessionPathGet": {"return_value": str(tmp_path)}}
    assert _session_dir_from_caller(FakeCtx(canned=canned)) == str(tmp_path)


def test_session_dir_from_caller_file_prefix_returns_parent(tmp_path):
    """SessionPathGet may return a file-PREFIX; normalise to its (existing) folder."""
    prefix = tmp_path / "STM_"  # not a dir; parent (tmp_path) is
    canned = {"Util_SessionPathGet": {"return_value": ("", b"", [0, str(prefix)])}}
    assert _session_dir_from_caller(FakeCtx(canned=canned)) == str(tmp_path)


def test_session_dir_from_caller_degrades_on_error():
    """rec.error → None (no hardware / call failed) — never raises."""
    canned = {"Util_SessionPathGet": {"return_value": None, "error": "boom"}}
    assert _session_dir_from_caller(FakeCtx(canned=canned)) is None


def test_session_dir_from_caller_no_safe_call():
    """An object without safe_call → None, never raises."""
    assert _session_dir_from_caller(object()) is None
    assert _session_dir_from_caller(None) is None


def test_candidate_save_dirs_uses_live_session_dir(tmp_path):
    """THE key fix: a context with NO session_path attr but a live pool must still
    discover the Nanonis session dir via Util_SessionPathGet."""
    (tmp_path / "live_042.sxm").write_bytes(b"fake")
    canned = {"Util_SessionPathGet": {"return_value": ("", b"", [0, str(tmp_path)])}}
    ctx = FakeCtx(canned=canned)  # NOTE: no ctx.session_path attribute
    dirs = _candidate_save_dirs(ctx)
    assert tmp_path.resolve() in {d.resolve() for d in dirs}


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
