"""工具 schema 通告范围必须包含于安全闸门接受范围。

schema 与闸门应读取同一份配置覆写；否则模型可能遵守通告仍被拒绝。
允许 schema 更严格，不允许它通告必被闸门拒绝的值。"""
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

from mast.agents._shared.safety_mw import SafetyGateMiddleware  # noqa: E402
from mast.agents._shared.skill_adapter import _ENVELOPE_FIELDS, _envelope_for  # noqa: E402
from mast.config import SafetyLimits  # noqa: E402
from mast.core.types import ParameterSpec  # noqa: E402


class _Reg:
    """一个把 xy / setpoint / z 都改过的覆写源 —— 出厂值与它必须不同，
    否则这条测试在「两边都用出厂值」时也会通过，变成一条恒真的空断言。"""

    OVERRIDES = {
        "xy_min_m": -2.0e-6, "xy_max_m": 2.0e-6,
        "z_min_m": -4.0e-7, "z_max_m": 4.0e-7,
        "setpoint_max_a": 1e-8,
        "scan_size_max_m": 4.0e-6,
    }

    @classmethod
    def get(cls):
        return cls()

    def get_safety_limits(self):
        return dict(self.OVERRIDES)

    def get_safety_checks(self):
        return None


@pytest.fixture()
def _overridden(monkeypatch):
    """让**两边**都看到同一份覆写（各自按自己的路径读）。"""
    import mast.admin.override_store as store
    monkeypatch.setattr(store, "ConfigOverrideRegistry", _Reg)
    yield _Reg


def _gate_bounds(mw, name, unit):
    """闸门对某个参数认的 (min, max)；None 表示它不管这个参数。

    ⚠️ 只取前四项：``safety_mw.SafetyGate`` 的 ``_resolved_checks`` 是六元组
    （尾部两项是 ``min_attr`` / ``max_attr``，用来在拒绝语里点名该调哪个 knob），
    而 ``core.safety.SafetyGuard`` 的是四元组。**两份实现的形状不一样** —— 按四元组
    解包会当场 ValueError（本文件初稿就是）。
    """
    for row in mw.gate._resolved_checks:
        pat, unit_pat, lo, hi = row[0], row[1], row[2], row[3]
        if pat in name.lower() and unit_pat == (unit or "").lower():
            return lo, hi
    return None


# ════════════════════════════════════════════════════════════════════════════
def test_the_override_actually_differs_from_the_factory_default(_overridden) -> None:
    """先证明这组覆写确实改变了什么 —— 否则下面的断言两边都是出厂值，恒真。"""
    base = SafetyLimits()
    for k, v in _Reg.OVERRIDES.items():
        assert getattr(base, k) != pytest.approx(v), (
            f"{k} 的覆写值和出厂值一样，这条测试对它没有鉴别力")


def test_schema_range_is_inside_the_gate_range(_overridden) -> None:
    """**核心不变式。** schema 通告的每个范围都要落在闸门接受的范围内。

    方向不对称：schema 比闸门严可以（少用余量），比闸门松不行（通告一个必被拒的
    范围，而拒绝语给的上界和 schema 说的对不上 —— 模型拿不到任何可行值）。
    """
    mw = SafetyGateMiddleware(limits=SafetyLimits(), registry=_Reg.get())
    offenders = []
    for name in _ENVELOPE_FIELDS:
        unit = "a" if name.endswith("_a") else "m"
        spec = ParameterSpec(name=name, type="float", unit=unit)
        s_lo, s_hi = _envelope_for(spec)
        g = _gate_bounds(mw, name, unit)
        if g is None or s_lo is None or s_hi is None:
            continue
        g_lo, g_hi = g
        if s_hi > g_hi or s_lo < g_lo:
            offenders.append(
                f"{name}: schema [{s_lo:.4g}, {s_hi:.4g}] 超出闸门 [{g_lo:.4g}, {g_hi:.4g}]")
    assert not offenders, (
        "工具 schema 通告了闸门会拒绝的范围 —— 模型照着通告写，被拒，而拒绝语给的"
        "上界和它读到的对不上：\n  " + "\n  ".join(offenders))


def test_the_gate_sees_the_same_overrides_the_schema_does(_overridden) -> None:
    """更直接的一条：两边读的是同一份覆写。

    §2.16 的根因就是它们读的不是同一份 —— schema 读覆写，闸门拿的是原始 config。
    """
    mw = SafetyGateMiddleware(limits=SafetyLimits(), registry=_Reg.get())
    assert mw.gate._limits.xy_max_m == pytest.approx(2.0e-6)
    assert mw.gate._limits.setpoint_max_a == pytest.approx(1e-8)

    spec = ParameterSpec(name="center_x_m", type="float", unit="m")
    assert _envelope_for(spec)[1] == pytest.approx(2.0e-6)


def test_a_gate_without_a_registry_still_gets_the_instrument_clamp() -> None:
    """没注入 registry 时**不合并覆写**（既有契约），但**仍然收紧**。

    收紧的依据是用户登记的仪器事实，与 registry 无关；agent 路径原来完全没有
    这一层，于是它对「前放只有 ±10 nA」这件事的认知和手动路径不一致。
    """
    from mast.core import instrument_profile as iprof

    saved = iprof.get_profile()
    iprof.set_persist_sink(None)
    try:
        iprof.set_profile({"preamp_full_scale_a": 1e-8})
        mw = SafetyGateMiddleware(limits=SafetyLimits())     # registry=None
        assert mw.gate._limits.setpoint_max_a == pytest.approx(1e-8), (
            "agent 侧闸门没有做仪器事实收紧")
    finally:
        iprof.set_profile(saved)


def test_clamping_never_widens_on_the_agent_path() -> None:
    """红线在 agent 这一侧同样成立：前放比配置宽时什么都不做。"""
    from mast.core import instrument_profile as iprof

    saved = iprof.get_profile()
    iprof.set_persist_sink(None)
    try:
        iprof.set_profile({"preamp_full_scale_a": 200e-9})
        mw = SafetyGateMiddleware(limits=SafetyLimits())
        assert mw.gate._limits.setpoint_max_a == pytest.approx(
            SafetyLimits().setpoint_max_a)
    finally:
        iprof.set_profile(saved)
