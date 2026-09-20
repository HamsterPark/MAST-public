"""发表级出图 —— 载重的是**开头那道闸**。

要求：

> 花 **1h** 扫一张完美图像，前提是针尖已经达到那个水平。

所以这个技能最重要的行为不是「会扫图」，是「**该拒绝的时候拒绝**」：
它要花一小时，一次误放行的代价就是一小时 + 一根可能已经在退化的针尖。

这里同时钉住三件会静默出错的事：

* 「验不了」不许当成「够好」；
* 静置必须排在改视野 / 改速度 / 改像素**之后**（那三样都会动针尖）；
* `ConfigureScan` 会把扫描速度改回档位表 —— 必须 `set_scan_speed=False`。
"""
from __future__ import annotations

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

import pytest  # noqa: E402

from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite import publication_frame as PF  # noqa: E402

FRAME = {"center_x_m": 1e-7, "center_y_m": 2e-7, "width_m": 5e-9,
         "height_m": 5e-9, "angle_deg": 0.0}
SCAN_FILE = {"path": "D:/x/pub_0001.sxm", "saved_path": "D:/x/pub_0001.sxm"}


def assess(conc, passed=True):
    return {"passed": passed, "angular_concentration": conc,
            "period_fast_axis_nm": 0.249, "reasons": [] if passed else ["not_a_lattice"]}


class FakeCtx:
    def __init__(self, script=None):
        self.script = dict(script or {})
        self.calls: list[tuple[str, dict]] = []
        self.slept: list[float] = []
        self.run_id = "test-pub"

    def run(self, name, params):
        self.calls.append((name, dict(params or {})))
        v = self.script.get(name, {})
        if v is PF and False:  # pragma: no cover
            pass
        if isinstance(v, dict) or v is None:
            return SkillResult(skill_name=name, success=True, data=dict(v or {}))
        if v == "FAIL":
            return SkillResult(skill_name=name, success=False, error="scripted fail")
        return SkillResult(skill_name=name, success=True, data=dict(v))

    def check_abort(self):
        return False

    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)

    def params_of(self, name):
        return [p for n, p in self.calls if n == name]

    def index_of(self, name):
        for i, (n, _) in enumerate(self.calls):
            if n == name:
                return i
        return -1


def _script(**over):
    s = {
        "GetScanFrame": FRAME,
        "ScanAt": {}, "SaveScan": SCAN_FILE, "GetLatestScanFile": SCAN_FILE,
        "AssessAtomicResolution": assess(900.0),
        "SetScanSpeed": {}, "GetScanSpeed": {"fwd_time_s": 3.0},
        "SetScanBuffer": {}, "ConfigureScan": {},
        "AutoTilt": {}, "MoveToXY": {},
        "StartScan": {}, "WaitScanComplete": {},
    }
    s.update(over)
    return s


def _run(ctx, **params):
    skill = PF.ScanPublicationFrame()
    # abortable_sleep 在这里不该真的睡一小时 —— 记下来就行
    skill.abortable_sleep = lambda c, s: ctx.slept.append(float(s))  # type: ignore
    from mast.skills.composite.graph_executor import GraphExecutor  # noqa: F401
    return skill.execute(ctx, params)


# ── 入口闸：该拒绝的时候必须拒绝 ──────────────────────────────────
def test_a_mediocre_tip_is_refused_before_the_hour_is_spent():
    ctx = FakeCtx(_script(AssessAtomicResolution=assess(65.0)))
    res = _run(ctx)
    assert res.success, res.error
    assert res.data["outcome"] == "tip_not_good_enough", res.data["outcome"]
    assert ctx.count("StartScan") == 0, "针尖没到水平就已经开扫了 —— 那一小时白花"
    text = res.data["summary_cn"]
    assert "集中度 65.0" in text, "拒绝时没把实测值报出来，人没法决定要不要覆盖"
    assert "不进入" in text


def test_a_failed_verdict_is_refused_too_even_with_a_high_number():
    """判据说 not_a_lattice 时，那个数字再高也不算数。"""
    ctx = FakeCtx(_script(AssessAtomicResolution=assess(5000.0, passed=False)))
    res = _run(ctx)
    assert res.data["outcome"] == "tip_not_good_enough"
    assert ctx.count("StartScan") == 0


def test_cannot_verify_is_not_the_same_as_good_enough():
    """拿不到验针尖那一帧 ⇒ 拒绝。**「验不了」不是「够好」。**"""
    ctx = FakeCtx(_script(SaveScan={}, GetLatestScanFile={}))
    res = _run(ctx)
    assert res.data["outcome"] == "tip_check_unavailable"
    assert ctx.count("StartScan") == 0
    assert "验不了" in res.data["summary_cn"]


def test_a_good_tip_proceeds():
    ctx = FakeCtx(_script())
    res = _run(ctx)
    assert res.data["outcome"] == "publication_frame_ready", res.data
    assert res.data["frame_path"] == SCAN_FILE["path"]
    assert ctx.count("StartScan") == 1


def test_the_gate_can_be_skipped_only_explicitly():
    ctx = FakeCtx(_script(AssessAtomicResolution=assess(1.0, passed=False)))
    res = _run(ctx, skip_tip_check=True)
    assert res.data["outcome"] == "publication_frame_ready"
    assert ctx.count("ScanAt") == 0, "跳过闸门时还扫了验证帧"


# ── 顺序：静置必须在改设置之后 ────────────────────────────────────
def test_the_settle_happens_after_frame_speed_and_pixels_are_set():
    """改视野 / 改速度 / 改像素每一样都会动针尖 —— 先静置再改设置等于白等。"""
    ctx = FakeCtx(_script())
    _run(ctx, settle_s=90.0)
    i_move = ctx.index_of("MoveToXY")
    for earlier in ("SetScanSpeed", "SetScanBuffer", "ConfigureScan", "AutoTilt"):
        j = ctx.index_of(earlier)
        assert j >= 0 and j < i_move, f"{earlier} 排在了移到起点之后"
    assert ctx.index_of("StartScan") > i_move, "还没静置就起扫了"
    assert ctx.slept == [90.0], ctx.slept


def test_the_tip_is_parked_at_the_scan_start_corner_not_the_centre():
    """StartScan 会把针尖拉回框的**起点**；静置就该在那个点上。"""
    ctx = FakeCtx(_script())
    _run(ctx)
    p = ctx.params_of("MoveToXY")[0]
    assert p["x_m"] == pytest.approx(FRAME["center_x_m"] - FRAME["width_m"] / 2)
    assert p["y_m"] == pytest.approx(FRAME["center_y_m"] - FRAME["width_m"] / 2)


def test_zero_settle_does_not_sleep():
    ctx = FakeCtx(_script())
    _run(ctx, settle_s=0.0)
    assert ctx.slept == []


# ── ConfigureScan 不许动速度 ──────────────────────────────────────
def test_configure_scan_is_told_not_to_touch_the_speed():
    """ConfigureScan 不得覆盖已经显式设置的线时间。"""
    ctx = FakeCtx(_script())
    _run(ctx)
    p = ctx.params_of("ConfigureScan")[0]
    assert p.get("set_scan_speed") is False, p


def test_the_line_time_is_read_back_after_configure():
    """设了速度还要回读 —— 「设过了」不等于「现在是这个值」。"""
    ctx = FakeCtx(_script())
    res = _run(ctx)
    setup = [p for p in res.data["phases"] if p.get("phase") == "setup"]
    assert setup and setup[0]["line_time_s_after_configure"] == 3.0, setup


def test_slow_line_time_and_pixels_reach_the_instrument():
    ctx = FakeCtx(_script())
    _run(ctx, line_time_s=4.0, pixels=1024)
    sp = ctx.params_of("SetScanSpeed")[0]
    assert sp["fwd_line_time"] == 4.0 and sp["bwd_line_time"] == 4.0
    assert ctx.params_of("SetScanBuffer")[0]["pixels"] == 1024


def test_it_refuses_when_the_current_frame_cannot_be_read():
    """读不到当前扫描框就**不去猜一个视野**开扫。"""
    ctx = FakeCtx(_script(GetScanFrame={}))
    res = _run(ctx)
    assert res.data["outcome"] == "scan_failed"
    assert ctx.count("StartScan") == 0
    assert "不去猜" in res.data["summary_cn"]
