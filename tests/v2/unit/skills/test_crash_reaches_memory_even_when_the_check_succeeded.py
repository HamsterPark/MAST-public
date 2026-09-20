"""撞针必须进得了记忆 —— 哪怕那个检查**自己是成功的**、哪怕它说不出在哪。

## 这一组守的是什么

一次成功检测到撞针的检查,**是一次成功的运行**:它被问「撞了没有」,它答了。
所以 `CheckScanForCrash`(`skills/builtins/scan_frame.py`)和
`CheckTipCrashByAmplitude`(`skills/builtins/qplus_amplitude.py`)撞针时都返回
`success=True`,判决放在 `data["crash_indicator"]` 里 —— **这是对的,不要去改它。**

错的是消费方:`runtime.crash_point` 从前第一行就是
`payload.get("success") is not False → return None`,拿「这个检查没跑成」当成了
「这个检查发现撞了」的代理。于是上面两个技能报出来的**每一次**撞针都进不了闸,
而 `record_crash` 的唯一调用点在手写的 `full_scan.py` 里、也不覆盖它们 ——
**两份撞针记忆谁都收不到**,症状是「一切正常」。

(`crash_point` 的 docstring 一直写着「Keyed on the RESULT carrying
`crash_indicator`」—— 文档说的和代码做的不是一回事,现在代码追上了文档。)

## 三条不变量

1. **`crash_indicator is True` 就是撞针**,与 `success` 无关;
2. **`crash_indicator is None` 是「判不了」,永远不算撞针** —— qPlus 在激励关着
   或没有基线时报的就是 `None`,把它当成撞针会让每一次没驱动音叉的读数都变成一次
   假撞针;
3. **定不了位 ≠ 没撞**:坐标读不到时记成 unlocated,既不丢掉、也不编一个。
   编一个坐标会在好表面上画一个 150 nm 禁区,同时让真正的坑继续可选 —— 两头都错。

## 判据不读源码

`inspect.getsource` 出过假绿。这一组用运行时对象:真实 `SkillRegistry.discover()`、
注册进去的类本身、技能的 `metadata()`,以及直接驱动记录器本人。
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
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

from mast.core.registry import SkillRegistry  # noqa: E402
from mast.core.runtime import (  # noqa: E402
    CoreRuntime,
    crash_point,
    is_crash_report,
)
from mast.core.tip_crash_tracker import get_tip_crash_tracker  # noqa: E402

WHAT_BROKE = (
    "\n"
    "════ 撞针又收不到了 ════\n"
    "不变量:`crash_indicator is True` ⇒ 必须进得了记忆,与 `success` 无关;\n"
    "        坐标读不到 ⇒ 记成 unlocated,不丢弃、不编造。\n"
    "⚠️ **不要**靠把检查技能改成 `success=False` 来修 —— 那会把「这个检查没跑成」\n"
    "   和「这个检查发现撞了」折进同一个值,是同一个错换个方向再犯一次。\n"
    "要修的是消费方(`runtime.is_crash_report` / `crash_point` / `_record_map_marker`)。\n"
    "════════════════════\n"
)

# 两个技能撞针时真实的 payload 形状:success=True,判决在 data 里,**没有坐标**。
CHECK_SCAN_CRASH = {
    "skill": "CheckScanForCrash", "success": True,
    "params": {"channels": "0,14"},
    "data": {"crash_indicator": True, "status": "crash", "crash_channel": "ch0"},
}
# ⚠️ 形状要和 ``qplus_amplitude.py:266`` 真正返回的一致 —— ``amplitude`` +
# ``baseline`` 正是 ``crash_scope`` 用来认出「点式判据」的那对正面证据,少写一个
# 字段这条 payload 就会被当成帧式的,测的就不是线上那条路了。
QPLUS_CRASH = {
    "skill": "CheckTipCrashByAmplitude", "success": True, "params": {},
    "data": {"crash_indicator": True, "status": "crash",
             "amplitude": 4.0e-13, "baseline": 5.0e-11,
             "fraction_of_baseline": 0.008, "threshold_fraction": 0.1},
}
QPLUS_UNDECIDABLE = {
    "skill": "CheckTipCrashByAmplitude", "success": True, "params": {},
    "data": {"crash_indicator": None, "status": "unavailable"},
}
FULL_SCAN_CRASH = {
    "skill": "FullScan", "success": False, "error": "CRASH_DETECTED: ...",
    "params": {"center_x_m": 1e-7, "center_y_m": 2e-7},
    "data": {"crash_indicator": True},
}


# ── 记录器的替身 ────────────────────────────────────────────────────────

class FakeStorage:
    def __init__(self):
        self.markers: list[dict] = []

    def log_marker(self, **kw):
        self.markers.append(kw)


class FakeState:
    """``snapshot()`` 返回自己 —— 记录器只读属性。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def snapshot(self):
        return self


_REAL_REGISTRY = SkillRegistry()
_REAL_REGISTRY.discover()


class Recorder:
    """驱动真正的 ``CoreRuntime._record_map_marker``,不构造整个 runtime。

    只喂它实际会读的四个属性。**断言看的是「标记真的落下来了」**,不是「没抛异常」
    —— 那个方法整段包在 ``except`` 里,一个坏掉的替身会安静地什么都不做。

    ⚠️ ``_registry`` 必须是**真的**。记录器用它查技能的 category,而 category 决定
    `classify_skill` 认不认这个技能:``CheckScanForCrash`` 是 ``ANALYSIS`` ⇒ 零足迹,
    但 ``category=None`` 时同一个函数按名字里的 "scan" 判成 ``"scan"`` ⇒ 会多出一个
    足迹标记。拿 ``None`` 当替身,测的就不是线上那条路了 —— 这个仓自己的
    ``test_tip_conditioning_builtins`` 里写着同一条教训(``Ctx(registry=None)``)。
    """

    def __init__(self, *, state=None):
        self._registry = _REAL_REGISTRY
        self._storage = FakeStorage()
        self._state = state
        self._experiment_log = None

    def record(self, payload) -> list[dict]:
        CoreRuntime._record_map_marker(self, payload)
        return self._storage.markers

    def crash_rows(self) -> list[dict]:
        return [m for m in self._storage.markers if m.get("kind") == "crash"]


# ── 1. 认得出来:success=True 的撞针也是撞针 ─────────────────────────────

@pytest.mark.parametrize("payload,name", [
    (CHECK_SCAN_CRASH, "CheckScanForCrash"),
    (QPLUS_CRASH, "CheckTipCrashByAmplitude"),
    (FULL_SCAN_CRASH, "FullScan"),
])
def test_a_crash_report_is_recognised_whatever_its_success_flag(payload, name):
    assert is_crash_report(payload) is True, (
        f"{name} 报的撞针没被认出来 —— 它 success="
        f"{payload['success']}。" + WHAT_BROKE)


def test_undecidable_is_never_a_crash():
    """`crash_indicator is None` = 判不了。三态不许塌成两态。

    qPlus 在**激励关着**时报的就是 `None`(未驱动解调器的噪声底拿去比基线,
    永远比出「塌了」)。把 `None` 当成撞针,等于每一次 STM 模式下的振幅读数
    都变成一次假撞针,并且会在好表面上撒满 150 nm 禁区。
    """
    assert is_crash_report(QPLUS_UNDECIDABLE) is False, WHAT_BROKE
    assert crash_point(QPLUS_UNDECIDABLE) is None
    assert Recorder().record(QPLUS_UNDECIDABLE) == [], (
        "「判不了」在地图上留下了记录" + WHAT_BROKE)
    assert get_tip_crash_tracker().snapshot()["tracked_regions"] == 0


def test_a_failure_that_is_not_a_crash_still_writes_no_crash_row():
    """扫描可以因为很多原因失败 —— 只有撞针才标记表面。"""
    payload = {"skill": "FullScan", "success": False,
               "params": {"center_x_m": 1e-7, "center_y_m": 2e-7},
               "data": {"timed_out": True}}
    assert is_crash_report(payload) is False
    assert Recorder().record(payload) == [] or not Recorder().crash_rows()


# ── 2. 定位:payload → 快照帧中心 → unlocated ────────────────────────────

def test_the_scan_frame_centre_places_a_crash_the_payload_could_not():
    """帧式判据 + 小帧:payload 说不出在哪 ⇒ 用帧中心,并如实标注来源。

    100 nm 帧的半对角 ≈ 71 nm < 150 nm 避让半径 ⇒ 真实撞点必在圈内,这个圈站得住。
    """
    rec = Recorder(state=FakeState(scan_center_x_m=3e-7, scan_center_y_m=-4e-7,
                                   scan_width_m=1e-7, scan_height_m=1e-7))
    rec.record(CHECK_SCAN_CRASH)

    rows = rec.crash_rows()
    assert len(rows) == 1, f"撞针没落到地图上:{rec._storage.markers}" + WHAT_BROKE
    assert (rows[0]["x_m"], rows[0]["y_m"]) == (3e-7, -4e-7)
    assert rows[0]["meta"]["pos_src"] == "state_scan_frame", (
        "位点来源没有如实标注 —— 读者分不清这是技能报的还是推出来的")


# ── 2b. 点式判据撞在针尖那一点,不是帧中心 ──────────────────────────────

def test_a_point_wise_verdict_is_placed_at_the_tip_not_the_frame_centre():
    """qPlus 振幅是**点式**判据:针尖撞在它当时所在的地方。

    帧中心对它是错的 —— 一次 500 nm 的扫描框里,针尖可能离帧中心 354 nm。
    """
    rec = Recorder(state=FakeState(
        x_pos_m=1.1e-7, y_pos_m=-2.2e-7,
        scan_center_x_m=9e-7, scan_center_y_m=9e-7,
        scan_width_m=1e-7, scan_height_m=1e-7))
    rec.record(QPLUS_CRASH)

    rows = rec.crash_rows()
    assert len(rows) == 1
    assert (rows[0]["x_m"], rows[0]["y_m"]) == (1.1e-7, -2.2e-7), (
        "点式判据被记在了帧中心 —— 那不是针尖撞的地方" + WHAT_BROKE)
    assert rows[0]["meta"]["pos_src"] == "tip_position"


def test_a_frame_wise_verdict_never_uses_the_tip_position():
    """反过来:帧式判据**不许**退到针尖位置。

    post-scan 的判据是对整帧负责的,而扫描结束时针尖停在帧的边角 —— 拿它当撞点
    是另一种假精度。帧式只有「帧中心(过尺寸闸)」和「unlocated」两条路。
    """
    rec = Recorder(state=FakeState(x_pos_m=1.1e-7, y_pos_m=-2.2e-7))
    rec.record(CHECK_SCAN_CRASH)   # 没有 scan_center ⇒ 只能 unlocated

    rows = rec.crash_rows()
    assert len(rows) == 1
    assert rows[0]["x_m"] is None, (
        "帧式判据退到了针尖位置" + WHAT_BROKE)
    assert rows[0]["meta"]["unlocated_reason"] == "no_position"


def test_a_skill_may_declare_its_own_scope():
    """``data["crash_scope"]`` 优先于形状启发式 —— 未来的判据有干净的表达方式,
    不必去改那条启发式(改它就是又一个「同一个动作 N 份实现」)。"""
    from mast.core.runtime import crash_scope

    assert crash_scope(QPLUS_CRASH) == "point"          # 形状认出来的
    assert crash_scope(CHECK_SCAN_CRASH) == "frame"
    declared = {"data": {"crash_indicator": True, "amplitude": 1.0,
                         "baseline": 2.0, "crash_scope": "frame"}}
    assert crash_scope(declared) == "frame", "声明没有压过启发式"
    assert crash_scope({"data": {"crash_indicator": True}}) == "frame", (
        "认不出的判据没有落到保守的那一档")


# ── 2c. 尺寸闸:大帧的帧中心是错的,不是不精确的 ─────────────────────────

def test_a_frame_too_large_to_vouch_for_its_centre_falls_back_to_unlocated():
    """500 nm 帧:半对角 ≈ 354 nm > 150 nm 半径 ⇒ **不画那个圈**。

    那个圈两头都错:没圈住真正撞过的点,却圈掉了一块没撞过的好表面。
    宁可说「知道大概在哪但不敢画圈」。
    """
    rec = Recorder(state=FakeState(scan_center_x_m=3e-7, scan_center_y_m=-4e-7,
                                   scan_width_m=5e-7, scan_height_m=5e-7))
    rec.record(CHECK_SCAN_CRASH)

    rows = rec.crash_rows()
    assert len(rows) == 1
    assert rows[0]["x_m"] is None, (
        f"用一个 500 nm 帧的中心画了 150 nm 的圈 —— 真实撞点可能在圈外 354 nm 处"
        + WHAT_BROKE)
    assert rows[0]["meta"]["unlocated_reason"] == "frame_too_large"
    located, unlocated = get_tip_crash_tracker().crash_points()
    assert unlocated == 1, "大帧退化后没进哨兵格 —— 这次撞针彻底没人记"


def test_an_unknown_frame_size_cannot_vouch_for_the_circle_either():
    """读不到帧多大 ⇒ 证不出这个圈盖得住撞点 ⇒ 不画。

    「不知道」不能当成「小到没问题」—— 那正是本仓最常吃亏的那一步。
    """
    rec = Recorder(state=FakeState(scan_center_x_m=3e-7, scan_center_y_m=-4e-7))
    rec.record(CHECK_SCAN_CRASH)

    rows = rec.crash_rows()
    assert rows[0]["x_m"] is None
    assert rows[0]["meta"]["unlocated_reason"] == "frame_size_unknown"


@pytest.mark.parametrize("radius_m,half_diag_m,placed", [
    (150e-9, 70.7e-9, True),    # 100 nm 帧 —— 圈站得住
    (150e-9, 353.6e-9, False),  # 500 nm 帧 —— 圈站不住
])
def test_the_gate_uses_the_configured_radius_not_a_second_hardcoded_150(
        radius_m, half_diag_m, placed):
    """尺寸闸的半径必须是**注入**的那个,不是这条路上写死的 150。

    钉法:直接调纯函数,把半径调大一倍 —— 原本被拦下的 500 nm 帧必须放行。
    如果有人在这里写了第二个 150,放大半径不会改变结果,这条就红。
    """
    from mast.core.runtime import crash_position

    side = half_diag_m * 2.0 / (2 ** 0.5)
    st = FakeState(scan_center_x_m=1e-7, scan_center_y_m=1e-7,
                   scan_width_m=side, scan_height_m=side)

    xy, src, why = crash_position(CHECK_SCAN_CRASH, st, crash_r_m=radius_m)
    assert (xy is not None) is placed, (xy, src, why)

    # 把半径放大到**一定盖得住**半对角 ⇒ 两种帧都该放行。
    # (注意不能只放大一倍:150→300 nm 仍然拦得住 354 nm 的半对角,那样测的是
    #  算术而不是「半径有没有被用上」。)
    xy2, _src2, _why2 = crash_position(CHECK_SCAN_CRASH, st,
                                       crash_r_m=half_diag_m * 2.0)
    assert xy2 is not None, (
        "把避让半径放大到远超半对角之后依然被拦 —— 这条路上有一个写死的半径"
        + WHAT_BROKE)

    # 反向:把半径缩到远小于半对角 ⇒ 两种帧都该被拦。
    xy3, _src3, why3 = crash_position(CHECK_SCAN_CRASH, st,
                                      crash_r_m=half_diag_m / 10.0)
    assert xy3 is None and why3 == "frame_too_large", (
        "把避让半径缩到远小于半对角之后依然放行 —— 尺寸闸没有在用这个半径"
        + WHAT_BROKE)


def test_a_crash_nobody_can_place_becomes_unlocated_not_silence_and_not_zero():
    """连帧中心都没有 ⇒ 记成「撞了,位置未知」。

    三件事同时要成立:
      * 地图上有一条**无坐标**的审计行(画不出圈是事实,不是借口);
      * 进程内记忆的**哨兵格**加一,好让 `FindCleanSpot` 的
        `crash_memory_unlocated` 把这件事说给调用方听;
      * **绝不**变成 (0, 0) —— 那会在压电正中心画一个 150 nm 禁区。
    """
    rec = Recorder(state=None)
    rec.record(CHECK_SCAN_CRASH)

    rows = rec.crash_rows()
    assert len(rows) == 1, "撞针被静默丢掉了" + WHAT_BROKE
    assert rows[0]["x_m"] is None and rows[0]["y_m"] is None, (
        f"给一次定不了位的撞针编了坐标:{rows[0]['x_m'], rows[0]['y_m']}"
        + WHAT_BROKE)
    assert rows[0]["meta"]["pos_src"] == "unlocated"

    located, unlocated = get_tip_crash_tracker().crash_points()
    assert unlocated == 1, "哨兵格没加上 —— FindCleanSpot 不会说出这件事" + WHAT_BROKE
    assert located == [], "定不了位的撞针变成了一个具体的点" + WHAT_BROKE


def test_the_unlocated_crash_reaches_the_spot_picker_as_a_warning():
    """端到端:定不了位的撞针 → `FindCleanSpot` 明说这个点避不开。"""
    from mast.skills.builtins.clean_spot import FindCleanSpot

    Recorder(state=None).record(QPLUS_CRASH)

    class Ctx:
        state = None
        _registry = None

        def safe_call(self, method, *a, role="main"):
            class R:
                error = ""
                return_value = ("", b"", [0.0, 0.0])
            return R()

        def check_abort(self):
            return False

    res = FindCleanSpot().execute(Ctx(), {})
    assert res.data["crash_memory_unlocated"] == 1
    assert "读不到坐标" in res.data["reason"]
    assert "避让圈画不出来" in res.data["reason"]


# ── 3. 不许把一次撞针数成两次 ───────────────────────────────────────────

def test_a_located_crash_is_not_also_counted_in_the_tracker():
    """`full_scan.py` 已经 `record_crash` 过了,记录器不许再记一次。

    阈值是 2 ⇒ 数两遍就等于「一次撞针 = 同点连撞」,会逼出一次不该发生的换区
    (退针 + 粗动),而那是分钟量级的动作。
    """
    rec = Recorder(state=FakeState(scan_center_x_m=1e-7, scan_center_y_m=2e-7))
    rec.record(FULL_SCAN_CRASH)

    assert len(rec.crash_rows()) == 1
    located, unlocated = get_tip_crash_tracker().crash_points()
    assert (located, unlocated) == ([], 0), (
        "定得了位的撞针也写进了 tracker —— 与 full_scan.py 重复计数" + WHAT_BROKE)


# ── 4. 保持不动的那条:孪生盖过手写 FullScan ────────────────────────────

def test_the_declarative_full_scan_twin_never_registers_over_the_handwritten_one():
    """`record_crash` 的唯一调用点在手写那份里 —— 被孪生盖掉就整条消失。

    今天 `loader._is_builtin_composite_twin`(`loader.py:157`)挡着它。
    这条是那道闸的到期报警。
    """
    from mast.skills.composite._base import CompositeSkillGraph
    from mast.skills.composite.interpreter import SpecComposite

    cls = _REAL_REGISTRY.get("FullScan")

    assert issubclass(cls, CompositeSkillGraph)
    assert not issubclass(cls, SpecComposite), (
        f"声明式孪生盖过了手写的 FullScan({cls.__module__}.{cls.__qualname__})"
        f" —— `record_crash` 的唯一调用点从活路径上消失了。")
    assert cls.__module__ == "mast.skills.composite.full_scan"
