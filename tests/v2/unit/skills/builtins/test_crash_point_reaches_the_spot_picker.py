"""撞针点必须对选点器可见 —— 两个来源,一套避让约定。

## 这一组钉的是什么

`FindCleanSpot` 只读**落库的地图标记**。撞针的另一份记忆在进程内的
`tip_crash_tracker` 里,而两者的失效方式正好互补:

* 地图那条链会断 —— 没有活动实验时 `runtime._record_map_marker` 直接 return,
  落库整段又包在 `except` 里(`_log_swallowed`)。断的时候地图上一个字都没有。
* 断的**同时** tracker 往往是唯一还记得刚才撞过的人(`full_scan.py` 撞针那一刻
  两边都写:`record_crash` 立刻生效,落库是 fire-and-forget)。

于是存在一整类情形:针刚撞出一个坑,选点器问地图,地图如实回答「我这儿没有记录」,
选点器就把针尖送回那个坑 —— **零报错**。这一组用例就是不让它再发生。

## 不新增第二套避让约定

避让圈是既有的:`DAMAGE_KINDS["crash"] → AnalysisConfig.crash_r_m`
(`avoid_radius_crash_nm`,出厂 150 nm),落库的 crash 标记一直在用它。
tracker 来的点转成**同一个** `kind="crash"`,因此走同一个半径。
:func:`test_the_radius_is_the_existing_one_not_a_private_number` 钉死这一条 ——
一旦有人在这条路上塞一个私有半径,它就红。
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

from mast.core.map_scope import analysis_config, crash_memory_markers  # noqa: E402
from mast.core.tip_crash_tracker import get_tip_crash_tracker  # noqa: E402
from mast.io.map_analysis import DAMAGE_KINDS, build_avoid_circles  # noqa: E402
from mast.skills.builtins.clean_spot import FindCleanSpot  # noqa: E402


class Ctx:
    """针尖停在 ``tip``;其余调用一律成功。与 test_tip_conditioning_builtins 同款。

    这套 fixture 里没有活动实验 ⇒ ``load_markers()`` 返回
    ``([], 0, False)`` ⇒ ``map_known=False``。**这正是本组要覆盖的那个状态**:
    地图什么都不知道,而 tracker 知道。
    """

    def __init__(self, tip=(0.0, 0.0)):
        self.tip = tip
        self.state = None
        self._registry = None

    def safe_call(self, method, *args, role="main"):
        outer = self

        class R:
            error = ""
            return_value = ("", b"", list(outer.tip)
                            if method == "FolMe_XYPosGet" else [0.0])

        return R()

    def check_abort(self):
        return False


def _crash_radius_m() -> float:
    return float(analysis_config().crash_r_m)


# ── 头号用例:撞一个点 → 选点器不再选它 ──────────────────────────────────

@pytest.mark.parametrize("purpose", ["pulse", "tip_shape"])
def test_the_picker_will_not_hand_back_the_crater_it_just_made(purpose):
    """针在 (0,0) 撞了 → 从 (0,0) 出发选点,返回的点必须在撞针避让圈之外。

    修复前:tracker 里有这次撞针,地图里没有,选点器只问地图 ⇒ 它会把 (0,0)
    自己作为距离 0 的候选原样交回来(``nearest_clean_from`` 的文档明确写了
    「针尖当前位置在还干净时本身就是候选,距离 0 排第一」)。
    """
    get_tip_crash_tracker().record_crash(0.0, 0.0)

    res = FindCleanSpot().execute(Ctx(tip=(0.0, 0.0)), {"purpose": purpose})
    r_crash = _crash_radius_m()

    # ⚠️ 2026-08-17 起有**两条合格的答案**,原因是中心区开了(请求:「2 直接开,
    # 就是要换区的效果」)。有 XY 粗动时可用区收到 ±250 nm,而脉冲网格步长
    # 1000 nm ⇒ 区里只有 (0,0) 一个脉冲落点;它正好就是刚炸出来的那个坑 ⇒
    # **一个候选都没有**才是对的,而那会走 success=False 那条路。
    #
    # 这条测试要防的始终是同一件事:**别把针尖送回自己刚炸出来的坑**。
    # 「没地方可去」同样满足它 —— 而且它是设计意图(接下来该 RelocateCoarseXY)。
    # 所以下面按两条路分别验,而不是把 success 当成前提。
    if not res.success:
        assert "spent" in (res.error or "").lower()             or "没有" in (res.error or "") or "no undamaged" in (res.error or "").lower(), (
                f"没找到落点,但理由不像「这片表面用完了」:{res.error}")
        # 关键:失败这条路上,**避让来源照样要说得出来** —— 否则读起来像
        # 「地图什么都不知道」,而真相是「撞针记忆挡住了唯一那个点」。
        assert res.data.get("crash_memory_points") == 1
        return

    # 返回的首选点
    d0 = (res.data["x_m"] ** 2 + res.data["y_m"] ** 2) ** 0.5
    assert d0 > r_crash, (
        f"选点器把针尖送回了自己刚炸出来的坑:返回点距坑心 {d0 * 1e9:.1f} nm,"
        f"撞针避让半径是 {r_crash * 1e9:.0f} nm")
    # 以及**每一个**候选 —— 调用方常常拿的是 candidates 而不是首选点
    for c in res.data["candidates"]:
        d = (c["x_m"] ** 2 + c["y_m"] ** 2) ** 0.5
        assert d > r_crash, f"候选 {c} 落在撞针避让圈内"


def test_the_crash_is_reported_as_an_answered_source():
    """避让确实发生过,要能从结果里看出来 —— 不是「碰巧没选中」。"""
    get_tip_crash_tracker().record_crash(0.0, 0.0)
    # ``tip_shape`` 而不是默认的 ``pulse``:30 nm 的避让圈在 ±250 nm 的中心区里
    # 有的是地方,所以这条走得到**成功**那一路 —— 而 ``reason`` 只在成功时才有
    # (失败那一路对应的字段是 ``error``)。
    # 用 pulse 的话中心区里唯一那个落点就是坑本身,拿不到 reason,
    # 这条测试就会因为一个和「来源要说得出来」无关的理由变红。
    res = FindCleanSpot().execute(Ctx(tip=(0.0, 0.0)), {"purpose": "tip_shape"})
    assert res.success, res.error

    assert res.data["crash_memory_points"] == 1
    assert "crash_memory" in res.data["avoidance_sources"]
    # 这套 fixture 没有活动实验,地图那一路本来就没答上
    assert res.data["map_known"] is False
    assert "map" not in res.data["avoidance_sources"]
    assert "撞针点已经避开" in res.data["reason"]


def test_a_crash_never_leaves_the_run_with_nowhere_to_go():
    """撞了一个点之后**照样找得到地方** —— 而且那个地方避开了坑。

    ── 这条翻过面(2026-08-17)

    它原来断言原点撞一次就 ``success is False`` —— 那是中心区 ±250 nm 硬生效
    的时代:区里只有区心一个脉冲落点,撞了就没了。

    真机证明那是个**死锁**:粗动换区必然重新进针,进针又在工作点留一个盘,
    新区在第一发之前就废了,于是无限换区。现在「区装不下一个落点」会退回
    压电范围,所以撞一次**不该**让整跑停摆。
    """
    get_tip_crash_tracker().record_crash(0.0, 0.0)
    res = FindCleanSpot().execute(Ctx(tip=(0.0, 0.0)), {"purpose": "pulse"})

    assert res.success, f"撞一个点就没地方去了 —— 那是死锁,不是保护:{res.error}"
    assert res.data["crash_memory_points"] == 1
    assert "crash_memory" in res.data["avoidance_sources"]
    r_crash = _crash_radius_m()
    d = (res.data["x_m"] ** 2 + res.data["y_m"] ** 2) ** 0.5
    assert d > r_crash, "给回来的点就在刚炸出来的坑里"


def test_two_crashes_in_one_cell_are_one_point_two_crashes_apart_are_two():
    """8 nm 量化格:同格合并、异格分开。计数如实带在 meta 里。"""
    tracker = get_tip_crash_tracker()
    tracker.record_crash(0.0, 0.0)
    tracker.record_crash(1e-9, 1e-9)        # 同一个 8 nm 格
    tracker.record_crash(500e-9, 500e-9)    # 另一个格

    markers, unlocated = crash_memory_markers()
    assert unlocated == 0
    assert len(markers) == 2
    assert sorted(m.meta["crash_count"] for m in markers) == [1, 2]


# ── 「读不到」不许当成「干净」 ────────────────────────────────────────────

def test_no_source_answered_is_said_out_loud_not_silently_clean():
    """地图读不到 + 本进程没撞过 = **不知道**,不是「干净」。

    这是本仓最贵的那条形状(`unknown_is_not_an_answer`):两个来源都没答上时,
    结果和「两个来源都说干净」在几何上完全一样 —— 差别只能靠说出来。
    """
    res = FindCleanSpot().execute(Ctx(tip=(0.0, 0.0)), {})

    assert res.success
    assert res.data["avoidance_sources"] == [], (
        "没有任何来源答上,却报告了一个来源")
    assert res.data["map_known"] is False
    assert res.data["crash_memory_points"] == 0
    assert "无法确认此处是否干净" in res.data["reason"]


def test_a_crash_with_no_coordinates_is_surfaced_as_unavoidable():
    """撞针记到了、坐标读不到 ⇒ 画不出圈 ⇒ 必须明说这个点避不开。

    ``record_crash(None, None)`` 是真实路径:`full_scan.py` 读不到扫描中心时
    就是这样记的。把它折叠成「没有撞针」是 `read_failure_folded_into_a_value`;
    把它变成一个猜出来的坐标画个圈,则是伪造事实。两个都不做 —— 单独报出来。
    """
    get_tip_crash_tracker().record_crash(None, None)

    res = FindCleanSpot().execute(Ctx(tip=(0.0, 0.0)), {})

    assert res.data["crash_memory_unlocated"] == 1
    assert res.data["crash_memory_points"] == 0, "无坐标的撞针不许变成一个圈"
    assert "读不到坐标" in res.data["reason"]
    assert "避让圈画不出来" in res.data["reason"]


# ── 走既有约定,不发明第二套 ──────────────────────────────────────────────

def test_tracker_points_use_the_existing_crash_kind():
    """tracker 来的点必须是 ``kind="crash"`` —— 与落库那条显式分支同一个词。

    绕过既有的 kind 就等于绕过 ``DAMAGE_KINDS`` 的查表:标记造出来了,却不产生
    任何避让圈,而且零报错。
    """
    get_tip_crash_tracker().record_crash(0.0, 0.0)
    markers, _ = crash_memory_markers()

    assert [m.kind for m in markers] == ["crash"]
    assert "crash" in DAMAGE_KINDS, "既有避让约定里没有 crash,这组用例的前提没了"

    circles = build_avoid_circles(markers, analysis_config())
    assert len(circles) == 1
    assert circles[0].kind == "crash"


def test_the_radius_is_the_existing_one_not_a_private_number():
    """避让半径必须**取自** ``AnalysisConfig.crash_r_m``,不是这条路上写死的数。

    钉法:改 config 的半径,tracker 这一路的避让圈必须跟着改。如果有人在
    map_scope / clean_spot / tip_crash_tracker 里塞了一个自己的常数,这里就红。
    """
    from dataclasses import replace

    get_tip_crash_tracker().record_crash(0.0, 0.0)
    markers, _ = crash_memory_markers()

    cfg = analysis_config()
    doubled = replace(cfg, crash_r_m=cfg.crash_r_m * 2.0)

    r1 = build_avoid_circles(markers, cfg)[0].radius_m
    r2 = build_avoid_circles(markers, doubled)[0].radius_m
    assert r1 == pytest.approx(cfg.crash_r_m)
    assert r2 == pytest.approx(cfg.crash_r_m * 2.0), (
        "半径没有跟着 AnalysisConfig 走 —— 这条路上有一个私有的避让半径")


def test_markers_seen_still_counts_only_persisted_rows():
    """合并进来的 tracker 点不许混进 ``markers_seen``。

    那个数回答的是「地图上有多少条记录」,被内存里的东西撑大之后,
    「地图是空的」这件事就再也读不出来了。
    """
    get_tip_crash_tracker().record_crash(0.0, 0.0)
    res = FindCleanSpot().execute(Ctx(tip=(0.0, 0.0)), {})

    assert res.data["markers_seen"] == 0
    assert res.data["crash_memory_points"] == 1


def test_the_tracker_is_not_turned_into_a_second_writer():
    """这条路只读:问一次 tracker 不许改变它的内容。

    tracker 是撞针阻断的判据源(``is_blocked`` → 强制换区)。选点器顺手往里写
    一笔,就会把「选过这个点」算成一次撞针。
    """
    tracker = get_tip_crash_tracker()
    tracker.record_crash(0.0, 0.0)
    before = tracker.snapshot()

    FindCleanSpot().execute(Ctx(tip=(0.0, 0.0)), {})
    FindCleanSpot().execute(Ctx(tip=(0.0, 0.0)), {})

    assert tracker.snapshot() == before
    assert tracker.crash_count(0.0, 0.0) == 1
