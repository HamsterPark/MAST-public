'ANALYSIS 技能不得生成空间足迹或推进计划路线。\n\n这些测试遍历真实技能注册表，验证类别过滤；对已知会被名称误分类的技能，还经过记录器读取数据库并检查路线未被推进。FindSpectralPeaks 只读取谱文件，因此与其他纯分析技能一同参与这些零足迹验证。\n\n无类别过滤的对照必须严格等于完整已知集合，避免注册表为空造成假通过。真实扫描与显式撞针记录同时保留反向行为测试。'
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.core.registry import SkillRegistry  # noqa: E402
from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.types import SkillCategory  # noqa: E402
from mast.io.exp_map import MapMarker, classify_skill  # noqa: E402
from mast.io.plan_overlay import get_plan_overlay  # noqa: E402
from mast.logging.storage import ExperimentStorage  # noqa: E402

EXP, SAMP = "exp-zero-footprint", "samp-zero-footprint"

#: 纯分析技能可能因名称被误判为空间动作；完整集合同时驱动行为回归。
KNOWN_FAKE_FOOTPRINTS = ["AutoProcessScanBatch", "CheckScanForCrash",
                         "LoadScanFrameFromFile", "FindSpectralPeaks"]


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    r = SkillRegistry()
    r.discover()
    return r


@pytest.fixture()
def rt(tmp_path, registry):
    """真实记录路径，接了真注册表；`_state=None` ⇒ 位置只能来自显式参数。"""
    r = CoreRuntime.__new__(CoreRuntime)  # 跳过 setup()，只接记录需要的东西
    r._storage = ExperimentStorage(str(tmp_path / "exp.db"))
    r._experiment_log = SimpleNamespace(current_experiment_id=EXP,
                                        current_sample_id=SAMP)
    r._state = None
    r._registry = registry
    return r


@pytest.fixture(autouse=True)
def _clean_plan_overlay():
    """`get_plan_overlay()` 是进程级单例 —— 不清会串味。"""
    get_plan_overlay().clear()
    yield
    get_plan_overlay().clear()


def _markers(rt) -> list[dict]:
    return rt._storage.get_markers(EXP, SAMP)


# ── 闸门：遍历注册表，ANALYSIS ⇒ 不落标记 ────────────────────────────────────

def test_no_analysis_skill_anywhere_produces_a_marker(registry):
    """§5.4 的那一条：**遍历**注册表，不用硬编码名单。

    这条测试在修之前是红的，红在 §1.4a 的三个上。它是唯一能把「没人知道」变成
    「有人知道」的东西 —— 将来有人给一个分析技能起名叫 `...Scan...`/`...STS...`
    时，它当场红，而不是等到实验记录里多出一个没发生过的足迹。
    """
    offenders = [
        (m.name, classify_skill(m.name, m.category))
        for m in sorted(registry.list_skills(), key=lambda m: m.name)
        if m.category is SkillCategory.ANALYSIS
        and classify_skill(m.name, m.category)
    ]
    assert offenders == [], (
        "这些 ANALYSIS 技能会往实验地图里写一个没发生过的空间事件："
        f"{offenders}。ANALYSIS 的定义是「Data processing, no hardware "
        "interaction」，它落标记永远是 bug —— 假足迹进 coverage_pct 的预算账，"
        "还会推进计划路线。修技能或（附逐条理由后）修分类规则，不要删这条断言。")


def test_the_sweep_catches_the_complete_known_name_only_misclassification_set(registry):
    """关闭类别过滤的对照必须命中完整集合，防止空注册表造成假通过。"""
    caught = [m.name for m in registry.list_skills()
              if m.category is SkillCategory.ANALYSIS and classify_skill(m.name)]
    assert sorted(caught) == sorted(KNOWN_FAKE_FOOTPRINTS), (
        f"名称分类对照必须命中完整已知集合，实际 {sorted(caught)}")


# ── 纯分析技能的零足迹：行为验证，零 marker ─────────────────────────────────────────

@pytest.mark.parametrize("skill", KNOWN_FAKE_FOOTPRINTS)
def test_known_fake_footprint_records_nothing(rt, skill, registry):
    """走真实记录路径，断言 storage 里**一行都没有**。

    payload 里刻意给了显式 xy：修之前这正是它落标记的位置来源，所以这条测试如果
    因为「压根没位置可落」而绿，那是白绿。
    """
    assert registry._get_metadata(registry.get(skill)).category is SkillCategory.ANALYSIS

    rt._record_map_marker({
        "skill": skill,
        "params": {"x_m": 1.2e-8, "y_m": -3.4e-8,
                   "width_m": 5e-8, "height_m": 5e-8},
        "success": True,
        "data": {"status": "ok"},
    })

    assert _markers(rt) == [], (
        f"{skill} 是 ANALYSIS（只读文件/缓冲区，不碰仪器），却在地图上留下了标记")


@pytest.mark.parametrize("skill", KNOWN_FAKE_FOOTPRINTS)
def test_known_fake_footprint_does_not_advance_the_route(rt, skill):
    """§1.4a 的后半句：假标记还会**推进计划路线**。

    「读一个昨天的 .sxm 会消耗今天勘测路线上的一步」—— 路线摆在标记要落的位置
    正上方，所以只要它落了标记就一定会消耗一步。
    """
    plan = [MapMarker(kind="plan", x_m=1.2e-8, y_m=-3.4e-8),
            MapMarker(kind="plan", x_m=9.0e-8, y_m=9.0e-8)]
    get_plan_overlay().set_plan(plan, title="勘测路线")

    rt._record_map_marker({
        "skill": skill,
        "params": {"x_m": 1.2e-8, "y_m": -3.4e-8},
        "success": True,
    })

    assert len(get_plan_overlay().snapshot()) == 2, (
        f"{skill} 只是读了一个文件，却消耗掉了勘测路线上的一步")


def test_check_scan_for_crash_still_records_a_real_crash(rt):
    """收得太紧同样是缺陷：撞针点走的是**显式分支**，不经 classify_skill。

    `CheckScanForCrash` 被类别排除掉的只有那个假扫描足迹；真撞上了的那一行必须
    还在 —— 它是「这个位置弄坏过针尖」的永久记录，避让模型靠它。
    """
    rt._record_map_marker({
        "skill": "CheckScanForCrash",
        "params": {"x_m": 2e-8, "y_m": 2e-8},
        "success": False,
        "data": {"crash_indicator": True, "center_x_m": 2e-8, "center_y_m": 2e-8},
    })
    rows = _markers(rt)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["crash"], (
        f"撞针点应当且只应当留一行 crash，实际 {kinds} —— "
        "假扫描足迹没删干净，或者把真撞针记录一起删了")


# ── 反方向：真事件行为不变 ──────────────────────────────────────────────────

def test_a_real_scan_still_records_and_advances(rt):
    """`FullScan` 是 WRITE 类，类别排除碰不到它 —— 标记照落，路线照推进。"""
    get_plan_overlay().set_plan([MapMarker(kind="plan", x_m=1e-8, y_m=2e-8)])

    rt._record_map_marker({
        "skill": "FullScan",
        "params": {"x_m": 1e-8, "y_m": 2e-8, "width_m": 5e-8, "height_m": 5e-8},
        "success": True,
    })

    rows = _markers(rt)
    assert [r["kind"] for r in rows] == ["scan"]
    assert rows[0]["status"] == "done"
    assert get_plan_overlay().is_empty(), "真扫描没有推进计划路线"


def test_a_failed_scan_records_but_does_not_advance(rt):
    """单标记路径此前**不传** advance（默认 True）⇒ 一次失败的扫描也把路线推进
    一步，等于把「说做了」记成「做了」。判据与多标记路径的 `bool(r_ok)` 对齐。
    """
    get_plan_overlay().set_plan([MapMarker(kind="plan", x_m=1e-8, y_m=2e-8)])

    rt._record_map_marker({
        "skill": "FullScan",
        "params": {"x_m": 1e-8, "y_m": 2e-8, "width_m": 5e-8, "height_m": 5e-8},
        "success": False,
    })

    rows = _markers(rt)
    assert [r["status"] for r in rows] == ["failed"], "失败的扫描必须**记**下来"
    assert len(get_plan_overlay().snapshot()) == 1, (
        "扫描失败了，路线却前进了一步 —— 没走到的一步被记成走到了")


def test_marker_from_skill_honours_the_category_itself():
    """`marker_from_skill` 是导出的公开函数（`__all__`），它自己也要认类别。

    在 `runtime` 里这一层是**冗余**的：上游那道 `classify_skill(skill, category)
    is None` 闸门先返回了，所以砍掉透传不会让任何走 runtime 的测试变红（变异验证
    实测如此）。冗余不等于可以不测 —— 它是这个函数的公开契约，直接调它的人（离线
    重建地图、别的记录路径）拿不到那道闸门。所以在**这一层**钉住。
    """
    from mast.io.exp_map import marker_from_skill

    params = {"x_m": 1e-8, "y_m": 2e-8, "width_m": 5e-8, "height_m": 5e-8}
    assert marker_from_skill("AutoProcessScanBatch", params, None) is not None, (
        "不传类别时应当**仍然**按名字落标记 —— 否则下面那条断言是白绿的")
    assert marker_from_skill("AutoProcessScanBatch", params, None,
                             category=SkillCategory.ANALYSIS) is None
    assert marker_from_skill("AutoProcessScanBatch", params, None,
                             category="analysis") is None
    assert marker_from_skill("FullScan", params, None,
                             category=SkillCategory.WRITE) is not None


def test_category_only_subtracts_never_adds(registry):
    """类别参数只做减法：非 ANALYSIS 时结果与不传类别**逐个技能**完全一致。

    这条挡的是「加了个参数顺手改了名字规则」——它必须是纯增量。
    """
    diffs = [(m.name, classify_skill(m.name), classify_skill(m.name, m.category))
             for m in registry.list_skills()
             if m.category is not SkillCategory.ANALYSIS
             and classify_skill(m.name) != classify_skill(m.name, m.category)]
    assert diffs == [], f"类别参数改变了非分析类技能的分类：{diffs}"


def test_missing_registry_falls_back_to_name_rules(tmp_path):
    """查不到类别 = 只按名字判，**不是**当成分析类。

    代价不对称（`_never_positions` 的注释）：误记看得见，漏记是一次真实的扎针从
    实验记录里消失，而修针流程正是靠地图避开扎过的位置。所以注册表缺席
    （`__new__` 出来的替身、手动活动侦测、还没进注册表的 spec 技能）必须回退到
    既有行为。
    """
    r = CoreRuntime.__new__(CoreRuntime)
    r._storage = ExperimentStorage(str(tmp_path / "exp.db"))
    r._experiment_log = SimpleNamespace(current_experiment_id=EXP,
                                        current_sample_id=SAMP)
    r._state = None
    # 刻意**不设** _registry —— 现役测试替身就是这个形状。

    r._record_map_marker({
        "skill": "FullScan",
        "params": {"x_m": 1e-8, "y_m": 2e-8, "width_m": 5e-8, "height_m": 5e-8},
        "success": True,
    })

    rows = r._storage.get_markers(EXP, SAMP)
    assert [x["kind"] for x in rows] == ["scan"], (
        "没有注册表时真实扫描从实验记录里消失了 —— 漏记比误记危险")


def test_unregistered_skill_falls_back_to_name_rules(rt):
    """「查不到」的**第二条**路径：注册表在，但这个名字不在注册表里。

    与上一条是两段不同的代码（`_registry is None` 提前返回 vs `registry.get()`
    抛 KeyError 落进 except），变异验证抓到过：只测了前者时，把 except 改成返回
    "analysis" 这条测试照绿。「读不到」被当成一个具体答案，正是这个仓库反复吃亏
    的形状。
    """
    assert not rt._registry.has("TotallyMadeUpScanSkill")

    rt._record_map_marker({
        "skill": "TotallyMadeUpScanSkill",
        "params": {"x_m": 1e-8, "y_m": 2e-8, "width_m": 5e-8, "height_m": 5e-8},
        "success": True,
    })

    assert [x["kind"] for x in _markers(rt)] == ["scan"], (
        "技能不在注册表里 ⇒ 类别「读不到」，而它被当成了「分析类」，"
        "一次真实扫描从实验记录里消失")


# ── D17：assess 动词前缀 ────────────────────────────────────────────────────

def test_assess_prefix_never_positions():
    """D17：`AssessSpectrum` 含 "spectr"，不加前缀会被 sts 规则认领成一个真实的
    谱学测量点 —— `MakeSpectroscopyTip` 那个教训的第三次复发。

    这里用**还没注册**的名字，正是因为类别排除救不了它：类别要查注册表，而伤害在
    技能落地的第一天就发生了。两条防线各管一段。
    """
    for name in ("AssessSpectrum", "AssessSpectrumQuality", "AssessSpectrumBatch",
                 "AssessSTSCurve", "AssessScanQuality", "AssessDidvSymmetry"):
        assert classify_skill(name) is None, f"{name} 会落一个假的谱学/扫描点"


def test_every_registered_assess_skill_is_unchanged_by_the_prefix(registry):
    """加前缀前全仓 `Assess*` 实测全部 None —— 加完仍然全部 None。

    **行为完全不变，只是把运气换成保证**。这条同时证明前缀没有误伤：如果哪个
    `Assess*` 其实是个真实的表面操作，它在这里就会现形（而不是等到实验记录里少
    了一次扎针）。
    """
    assessors = [m.name for m in registry.list_skills()
                 if m.name.lower().startswith("assess")]
    assert assessors, "一个 Assess* 都没找到 —— 这条测试在空转"
    still_marking = [n for n in assessors if classify_skill(n)]
    assert still_marking == [], f"{still_marking} 仍会落标记"


def test_classify_skill_never_returns_empty_string():
    """docstring 曾经说调用方可以靠 `""` 区分「skip」与「unknown」—— 那是假的：
    `return kind or None` 把 `""` 折叠成了 None，这个函数**产生不出** `""`。

    O7 要求核实「案② 不会误伤靠 `""` 承重的调用方」，结论是没有这种调用方**能**
    存在。这条把结论钉住：哪天有人把折叠去掉，`runtime` 里那两处 `is None` 闸门会
    静默失效（`"" is None` 为假 ⇒ 假标记全部放行）。
    """
    for name in ("PreScanCheck", "AssessImageQuality", "AssessScan", "TrackDrift",
                 "StopAutoApproach", "GetAutoApproachStatus", "SetBias", ""):
        got = classify_skill(name)
        assert got is None, f"classify_skill({name!r}) 返回了 {got!r}，不是 None"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
