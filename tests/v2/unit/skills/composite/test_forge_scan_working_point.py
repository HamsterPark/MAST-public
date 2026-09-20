"""验证流程工作点经统一参数组装到达 ScanAt 并被解析器使用；同时检查配置消费者与调用入口。"""
from __future__ import annotations

import ast
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

from mast.core import scan_policy  # noqa: E402
from mast.core.noble_tip_workflow import (  # noqa: E402
    NOBLE_METAL_BASELINE,
    NobleTipWorkflow,
    forge_line_time_s,
)
from mast.skills.composite._tip_phases import scan_at_params  # noqa: E402
from mast.skills.composite.forge_au_tip import ForgeAuTip, _forge_wf  # noqa: E402
from mast.skills.composite.scan_at import ScanAt  # noqa: E402

# 隔离自动使用的测试 fixture，避免注册状态泄漏。
from tests.v2.unit.skills.composite.test_forge_au_tip import (  # noqa: E402
    _base_script,
    _fast_and_offline,  # noqa: F401 —— autouse fixture,导入即生效,别删
    _run,
)

_MAST_SRC = Path(_MASTV2_ROOT) / "mast"
_WF_SRC = _MAST_SRC / "core" / "noble_tip_workflow.py"

#: 会读流程表字段的模块。加一个新的消费方就加在这里。
_CONSUMER_SRCS = (
    _MAST_SRC / "skills" / "composite" / "_tip_phases.py",
    _MAST_SRC / "skills" / "composite" / "forge_au_tip.py",
    _MAST_SRC / "skills" / "composite" / "prepare_noble_tip.py",
    _MAST_SRC / "skills" / "composite" / "make_special_tip.py",
    _MAST_SRC / "skills" / "builtins" / "tip_conditioning_selfcheck.py",
    _MAST_SRC / "core" / "special_tip_workflow.py",
)

#: 会 yield ScanAt 步骤的模块(闸门 B 扫这些)。
_SCAN_CALLER_SRCS = (
    _MAST_SRC / "skills" / "composite" / "_tip_phases.py",
    _MAST_SRC / "skills" / "composite" / "forge_au_tip.py",
    _MAST_SRC / "skills" / "composite" / "prepare_noble_tip.py",
)

#: 闸门 A 的**显式**豁免名单。每一条都要写清楚它是什么、以及什么能把它移出去。
#:
#: `approach_bias_v`(出厂 4.0 V,注释「坏针尖导电性差……几伏先把隧穿建立起来」):
#: `prepare_noble_tip._wf_from` 把它列进读取键 ⇒ 值会进流程表,但**全仓没有任何
#: 一处读 `wf.approach_bias_v`**,而且 PrepareNobleTip 也没有为它声明
#: ParameterSpec ⇒ 模型传了还会被 pydantic 静默丢弃。这是一条既有的死配置,
#: 正是本闸门要防的形状,只是修它意味着给流程新增一次真实的硬件动作
#: (进针前设 4 V,接在 RelocateCoarseXY(reapproach) 还是 ApproachTip 之前?)
#: —— 那是用户的科学判断,不是这次改动的范围。
#: **把它移出豁免名单需要回答**:进针偏压该由哪一步下发?
_ORPHAN_EXEMPT = {"approach_bias_v"}


# ── AST 小工具:注释与字符串天然不参与 ──────────────────────────────────────
#
# 刻意不用正则:一条「解释为什么不这么写」的注释会被正则当成「这么写了」,
# 而这个文件要防的恰恰是「读起来像在生效」。

def _wf_field_names() -> list[str]:
    tree = ast.parse(_WF_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "NobleTipWorkflow":
            return [n.target.id for n in node.body
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
    raise AssertionError("在 noble_tip_workflow.py 里找不到 NobleTipWorkflow 类")


def _attr_reads(path: Path) -> set[str]:
    """该模块里所有 ``X.attr`` 的 attr 名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


def _scan_at_steps(path: Path) -> list[ast.Call]:
    """该模块里每一个 ``CompositeStep(..., skill_name="ScanAt", ...)`` 调用。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "CompositeStep"):
            continue
        for kw in node.keywords:
            if (kw.arg == "skill_name" and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "ScanAt"):
                out.append(node)
    return out


# ══════════════════════════════════════════════════════════════════════
# 主验收:值真的到达 ScanAt,而且 ScanAt 真的用了它
# ══════════════════════════════════════════════════════════════════════

def _scan_calls_by_size(ctx) -> dict[int, list[dict]]:
    """按视野(nm,取整)把 ScanAt 的实参分组。"""
    out: dict[int, list[dict]] = {}
    for p in ctx.params_for("ScanAt"):
        nm = round(float(p["size_m"]) * 1e9)
        out.setdefault(nm, []).append(p)
    return out


def test_every_forge_scan_carries_the_measured_working_point():
    """端到端:跑完一遍外环,**每一次** ScanAt 都带着工作点。

    这条是本次修复的主验收。改前:三处 ScanAt 一个都不带 pixels / line_time_s。
    """
    res, ctx = _run(_base_script(), max_sites=1)
    assert res.data["outcome"] == "ready", res.data.get("summary_cn")

    calls = ctx.params_for("ScanAt")
    assert len(calls) >= 3, f"外环应该至少扫台阶/簇/验收三张图,实到 {len(calls)}"
    wf = _forge_wf({})
    for p in calls:
        assert "pixels" in p and "line_time_s" in p, (
            f"这一次 ScanAt 没有带工作点,会落回按尺寸查表的档位:{p}")
    # 三张图各自拿到自己那一档(而不是共用一个数)。
    by_size = _scan_calls_by_size(ctx)
    # 2026-08-14 起 ScanAt 路径上可能有第三种视野:100 nm 上没找到台阶时的
    # **200 nm 回退图**(要求:「换 200 nm 接着找,不要把小起伏硬算作台阶」)。
    # 它是否出现取决于台阶判据,所以这里用**子集**断言 —— 但集合里绝不许出现
    # 计划外的第四种视野(那意味着有人偷偷加了一张没人管工作点的图)。
    allowed = {round(wf.step_scan_nm), round(wf.cluster_scan_nm),
               round(wf.step_fallback_nm)}
    assert set(by_size) <= allowed, f"出现了计划外的扫描视野:{by_size}"
    assert {round(wf.step_scan_nm), round(wf.cluster_scan_nm)} <= set(by_size), by_size
    # ⚠️ 2026-08-15起,期望值要**过一遍派生**再比。
    #
    # 这条测试问的一直是「流程表那个数有没有到达 ScanAt」,而从这天起「流程表那
    # 个数」不再是字段里的字面量,而是字面量**经速度上限收口之后**的值(台阶/验收
    # 图 0.15 s ⇒ 667 nm/s 超上限 ⇒ 派生成 0.586 s)。直接比字面量会让这条测试
    # 变成「派生没生效才绿」—— 断言为真、被测系统是坏的,本文件顶上那条教训。
    def want(size_nm: float, raw_lt: float) -> float:
        lt, _why = forge_line_time_s(wf, size_nm, raw_lt)
        return lt

    # ⚠️ **视野已经不是身份了。** 2026-08-15 把找台面图从 100 → 200 nm 之后,
    # 它和「找不到台阶时的回退图」(step_fallback_nm = 200)同尺寸;在那之前它和
    # 台阶/验收图(100 nm)同尺寸。两次都是同一个坑:按 nm 分桶,桶里混进了
    # 两种图,而断言拿错了那一张。
    #
    # 分辨靠**行数**(找台面图 128 px,其余 256 px)—— 它恰好唯一,而且这条
    # 唯一性由 `test_the_site_scan_line_time_keeps_the_tip_at_the_pinned_speed`
    # 旁边那组测试盯着。真要彻底,该让每一步在参数里自报身份(scan_at_params
    # 的 `origin` 已经是这个方向),那是另一次改动。
    def _is_site_frame(p) -> bool:
        return int(p["pixels"]) == int(wf.poke_site_pixels) != int(wf.step_pixels)

    def _check_site_frame(p) -> None:
        assert p["line_time_s"] == pytest.approx(
            want(wf.poke_site_scan_nm, wf.poke_site_line_time_s))

    for p in by_size[round(wf.step_scan_nm)]:
        if _is_site_frame(p):
            _check_site_frame(p)
            continue
        assert p["pixels"] == wf.step_pixels
        assert p["line_time_s"] == pytest.approx(
            want(wf.step_scan_nm, wf.step_line_time_s))
    for p in by_size[round(wf.cluster_scan_nm)]:
        assert p["pixels"] == wf.cluster_pixels
        assert p["line_time_s"] == pytest.approx(
            want(wf.cluster_scan_nm, wf.cluster_line_time_s))
    # 回退图的线时**按视野比例放大** —— 保住针尖速度,而不是保住帧时。
    # (放大之后仍要过速度上限:基准本身超速时,忠实的缩放只会把超速一并带过去。)
    for p in by_size.get(round(wf.step_fallback_nm), []):
        if _is_site_frame(p):          # 200 nm 上也有找台面图(见上面那段)
            _check_site_frame(p)
            continue
        assert p["pixels"] == wf.step_pixels
        assert p["line_time_s"] == pytest.approx(want(
            wf.step_fallback_nm,
            wf.step_line_time_s * wf.step_fallback_nm / wf.step_scan_nm)), (
            "回退图沿用了 100 nm 那张的线时 ⇒ 针尖速度随视野翻倍")


def test_scan_at_actually_honours_what_forge_hands_it():
    """把外环真正传出去的那个 params dict 喂给**真的** ScanAt 解析一遍。

    「参数被传出去了」证明不了下游不会丢掉它。这一条走的是 ScanAt 自己的
    `_resolve`(→ scan_resolver 的 explicit 优先级链),断言解析结果里
    每线时间与像素就是流程表那两个数,帧时因此从档位表的 512-614 s 掉到 ~77 s。
    """
    _res, ctx = _run(_base_script(), max_sites=1)
    wf = _forge_wf({})
    seen = 0
    for p in ctx.params_for("ScanAt"):
        resolved = ScanAt()._resolve(p)
        nm = round(float(p["size_m"]) * 1e9)
        is_fallback = nm == round(wf.step_fallback_nm)
        # 2026-08-14:D 相「找台面」的图也是 100 nm,但**参数与台阶图不同**
        # (128 px / 0.586 s ⇒ 2.5 min,只用来找台面,不用 0.391 nm/px 那么细)。
        # 按视野分不开它们,要按像素数认。
        is_poke_site = (nm == round(wf.poke_site_scan_nm)
                        and int(p.get("pixels") or 0) == int(wf.poke_site_pixels)
                        and int(wf.poke_site_pixels) != int(wf.step_pixels))
        if is_poke_site:
            want_px, want_lt = wf.poke_site_pixels, wf.poke_site_line_time_s
        elif is_fallback:
            # 回退图:像素同 step,线时**按视野放大**以保住针尖速度。
            want_px = wf.step_pixels
            want_lt = wf.step_line_time_s * wf.step_fallback_nm / wf.step_scan_nm
        elif nm == round(wf.step_scan_nm):
            want_px, want_lt = wf.step_pixels, wf.step_line_time_s
        else:
            want_px, want_lt = wf.cluster_pixels, wf.cluster_line_time_s
        # 期望值过一遍速度上限的派生—— 理由同上一条测试里那段。
        want_lt, _why = forge_line_time_s(wf, float(nm), want_lt)
        assert resolved.configure_scan["line_time_s"] == pytest.approx(want_lt), (
            f"{nm} nm 的图:ScanAt 解析出来的每线时间是 "
            f"{resolved.configure_scan['line_time_s']},不是流程表的 {want_lt}")
        assert resolved.set_scan_buffer["pixels"] == want_px
        # 预计帧时由线数、每线时间和两个扫描方向推导。
        assert resolved.estimated_scan_s == pytest.approx(
            scan_policy.estimate_scan_seconds(want_px, want_lt))
        # 帧预算随几何比例变化。
        assert resolved.estimated_scan_s <= wf.scan_timeout_s / 2.0, (
            f"{nm} nm 的图要 {resolved.estimated_scan_s:.0f} s,超过等待预算 "
            f"{wf.scan_timeout_s:.0f} s 的一半 —— 要么工作点没生效(落回了档位表"
            "的 512-614 s),要么改了几何没回来重算 forge_scan_timeout_s。")
        # 所有扫描帧都必须遵守配置的速度上限。
        speed = nm / want_lt
        assert speed <= wf.forge_v_tip_nm_s * (1.0 + 1e-9), (
            f"{nm} nm 扫描速度 {speed:.0f} nm/s 超出工作流声明上限")
        # 来源必须记成「显式」,不是档位表 —— trace 是用户事后审计的唯一依据。
        assert resolved.trace["line_time_s"]["source"] == "explicit"
        assert resolved.trace["pixels"]["source"] == "explicit"
        seen += 1
    assert seen >= 3, f"只检了 {seen} 次 ScanAt,自检失败"


def test_the_wait_budget_can_hold_the_frame_that_will_actually_run():
    """等待上限要对着**真正会跑的那一帧**,不是对着假设的 77 s。

    改前的同类断言写的是 `forge_scan_timeout_s >= 2 * 77.0` —— 断言为真、被测
    系统是坏的:真正会跑的帧是 512-614 s,预算 180 s(最多延长到 270 s)必然
    判死。**证据回答的不是被问的那个问题。**
    """
    _res, ctx = _run(_base_script(), max_sites=1)
    for p in ctx.params_for("ScanAt"):
        est = ScanAt()._resolve(p).estimated_scan_s
        budget = ScanAt._wait_timeout_s(p, est)
        assert budget >= 2.0 * est, (
            f"{round(p['size_m'] * 1e9)} nm:帧要 {est:.0f} s,预算只有 "
            f"{budget:.0f} s")


def test_the_verify_prescan_gets_a_budget_that_can_hold_it():
    """B 阶段:设分辨率的人要为继承它的那一步的预算负责。

    PreScanCheck **从不设线数**(ConfigureScan 走 `Scan_BufferSet(ch, 0, 0)`),
    所以它扫几行由上一次扫描决定 —— 而这条流程自己刚把它设成 256。
    256 行 × 0.1 s × 2 = 51.2 s,而 PreScanCheck 的出厂预算是 15 s。
    """
    from mast.skills.composite.prescan_check import (
        DEFAULT_LINE_TIME_S,
        DEFAULT_WAIT_TIMEOUT_S,
    )

    _res, ctx = _run(_base_script(), max_sites=1)
    calls = ctx.params_for("PreScanCheck")
    assert calls, "外环没有做过验证扫描?"
    wf = _forge_wf({})
    inherited_s = wf.step_pixels * DEFAULT_LINE_TIME_S * 2.0
    # 2026-08-10:出厂盲估已经改成按 512 行算(163 s),所以它本身就装得下这里的
    # 256 行了 —— 但 forge 仍然要传自己的预算:它知道自己设了多少行,
    # 比盲估精确,而**设分辨率的人为继承它的那一步负责**这条原则没变。
    assert DEFAULT_WAIT_TIMEOUT_S >= inherited_s, (
        "出厂盲估又装不下 256 行了 —— 那正是真机两次 timeout 的形状")
    for p in calls:
        assert p.get("wait_timeout_s") == pytest.approx(wf.scan_timeout_s)
        assert p["wait_timeout_s"] > inherited_s, (
            f"验证扫描要 {inherited_s:.0f} s(继承 {wf.step_pixels} 行),"
            f"预算只有 {p['wait_timeout_s']:.0f} s")


def test_the_prescan_budget_follows_the_line_count():
    """预扫描等待预算由几何与行数推导。"""
    from mast.core.types import NanonisCallRecord
    from mast.skills.composite.prescan_check import (
        DEFAULT_LINE_TIME_S,
        DEFAULT_WAIT_TIMEOUT_S,
        PreScanCheck,
    )

    class _Ctx:
        def __init__(self, lines):
            self.lines = lines

        def safe_call(self, method, *args, role="main"):
            if method == "Scan_BufferGet":
                # 按协议的单元素元组列表形状构造通道回包。
                return NanonisCallRecord(
                    method=method, args=args, error="",
                    return_value=("", b"", [1, [(0,)], self.lines, self.lines]))
            return NanonisCallRecord(method=method, args=args, error="",
                                     return_value=("", b"", [0.0]))

    skill = PreScanCheck()
    skill._all_calls = []
    budget = skill._derived_timeout_s(_Ctx(256), DEFAULT_LINE_TIME_S)
    frame_s = 256 * DEFAULT_LINE_TIME_S * 2
    assert budget >= frame_s, (
        f"缓冲区里是 256 行(={frame_s:.0f} s),而预算只有 {budget:.0f} s")

    # **跟着行数走**才是这条的重点:盲估的 DEFAULT 现在已经够大(512 行),
    # 所以「比 DEFAULT 大」不再能证明派生生效 —— 只有「不同行数给出不同预算」能。
    skill._all_calls = []
    b1024 = skill._derived_timeout_s(_Ctx(1024), DEFAULT_LINE_TIME_S)
    assert b1024 > budget, (
        f"1024 行({b1024:.0f} s)与 256 行({budget:.0f} s)预算一样 —— 没在派生")
    assert b1024 >= 1024 * DEFAULT_LINE_TIME_S * 2

    # 读不回行数就退回出厂常数 —— 猜一个行数比用旧常数更坏。
    class _Blind(_Ctx):
        def safe_call(self, method, *args, role="main"):
            return NanonisCallRecord(method=method, args=args,
                                     error="no buffer", return_value=None)

    skill._all_calls = []
    assert skill._derived_timeout_s(_Blind(0), DEFAULT_LINE_TIME_S) == \
        DEFAULT_WAIT_TIMEOUT_S


def test_a_blind_prescan_budgets_for_the_worst_realistic_buffer():
    """预扫描必须容纳配置内的几何和等待预算。"""
    from mast.core.scan_policy import estimate_scan_seconds, wait_budget_s
    from mast.skills.composite.prescan_check import (
        DEFAULT_LINE_TIME_S,
        DEFAULT_WAIT_TIMEOUT_S,
        PreScanCheck,
        _BLIND_LINES,
        _V1_FLOOR_S,
    )

    assert DEFAULT_LINE_TIME_S == 0.1
    assert _V1_FLOOR_S == 15.0, "v1 那个常数仍然是下限，只是不再是预算本身"
    # 等待预算必须覆盖由行数、线时和方向计算出的帧时。
    assert DEFAULT_WAIT_TIMEOUT_S >= 256 * DEFAULT_LINE_TIME_S * 2, (
        f"盲估预算 {DEFAULT_WAIT_TIMEOUT_S:.0f} s 仍然装不下 256 行的预扫描")
    # 而且它就是共用公式算出来的，不是又一个手写常数。
    assert DEFAULT_WAIT_TIMEOUT_S == pytest.approx(wait_budget_s(
        estimate_scan_seconds(_BLIND_LINES, DEFAULT_LINE_TIME_S),
        floor_s=_V1_FLOOR_S))

    # 裸调用一张 50 nm 的预扫描：线时来自档位表，预算按**同一个线时**盲估。
    from mast.core.scan_policy import get_tier_for_size
    from mast.skills.composite.prescan_check import resolve_line_time_s

    lt = resolve_line_time_s(5e-8)
    assert lt == pytest.approx(float(get_tier_for_size(5e-8)["line_time_s"])), \
        "线时不再来自档位表 —— 那正是「预扫描比它要预测的扫描更快」的复发"
    steps = {s.step_id: s for s in PreScanCheck().plan(
        {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 5e-8})}
    assert steps["set_speed"].params["fwd_line_time"] == pytest.approx(lt)
    # 针尖横向速度 = 宽 / 线时。这是**刮不刮针**的那个数，所以直接钉它。
    assert steps["set_speed"].params["fwd_speed"] == pytest.approx(5e-8 / lt)
    assert steps["set_speed"].params["fwd_speed"] == pytest.approx(5e-8), \
        "50 nm/s；旧值 5e-7（500 nm/s）是把针尖刮坏的那个速度"
    # 预算跟着线时走：模块常数按 0.1 s 算死，用它会低估 10 倍。
    assert steps["wait_scan"].params["timeout_ms"] == int(1000 * wait_budget_s(
        estimate_scan_seconds(_BLIND_LINES, lt), floor_s=_V1_FLOOR_S))
    assert steps["wait_scan"].params["timeout_ms"] > int(
        DEFAULT_WAIT_TIMEOUT_S * 1000), (
        "盲估预算仍等于那个按 0.1 s/线 算死的模块常数 —— "
        "线时慢了 10 倍而预算没跟上，每一次预扫描都会超时")


def test_the_wait_budget_formula_has_one_source():
    """`ScanAt` 与 `PreScanCheck` 必须用**同一条**公式，只有下限可以不同。

    抄第二份是这条教训的第三次（档位表 / recursion_limit / 这里），
    所以这次是抽出来共用。结构闸门钉住两处都调它、且都不再自己写 `* 1.3`。
    """
    import ast

    from mast.core.scan_policy import estimate_scan_seconds, wait_budget_s

    est = estimate_scan_seconds(512, 0.1)
    assert wait_budget_s(est, floor_s=15.0) == pytest.approx(est * 1.3 + 30.0)
    assert wait_budget_s(1.0, floor_s=300.0) == 300.0      # 下限生效

    root = Path(_MASTV2_ROOT) / "mast" / "skills" / "composite"
    for name in ("scan_at.py", "prescan_check.py"):
        src = (root / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "wait_budget_s" in called, f"{name} 没有走共用的预算公式"


def test_prepare_noble_tip_still_lets_the_policy_table_decide():
    """非 forge 的那条路径(原地修一次)行为不变:两个键一个都不下发。

    通用字段出厂是 None = 空即 no-op。若哪天给它们填了出厂值,用户的档位表
    就在 PrepareNobleTip 上被悄悄旁路了。
    """
    assert NOBLE_METAL_BASELINE.step_pixels is None
    assert NOBLE_METAL_BASELINE.step_line_time_s is None
    assert NOBLE_METAL_BASELINE.cluster_pixels is None
    assert NOBLE_METAL_BASELINE.cluster_line_time_s is None
    p = scan_at_params(NOBLE_METAL_BASELINE, 1e-8, 2e-8,
                       size_nm=NOBLE_METAL_BASELINE.step_scan_nm,
                       pixels=NOBLE_METAL_BASELINE.step_pixels,
                       line_time_s=NOBLE_METAL_BASELINE.step_line_time_s,
                       origin="clean_spot")
    assert "pixels" not in p and "line_time_s" not in p, (
        f"None 应该是「不下发」而不是「下发 None」:{p}")
    assert p["size_m"] == pytest.approx(NOBLE_METAL_BASELINE.step_scan_nm * 1e-9)
    assert p["wait_timeout_s"] == NOBLE_METAL_BASELINE.scan_timeout_s


def test_the_forge_scan_working_point_keeps_its_configuration_contract():
    """配置基线与实际速度限制分开验证，不将默认配置视作实验标定。"""
    wf = NOBLE_METAL_BASELINE
    assert wf.forge_step_pixels == 256
    assert wf.forge_step_line_time_s == pytest.approx(0.15)
    assert scan_policy.estimate_scan_seconds(
        wf.forge_step_pixels, wf.forge_step_line_time_s) == pytest.approx(76.8)
    step_speed = wf.forge_step_scan_nm / wf.forge_step_line_time_s
    cluster_speed = wf.forge_cluster_scan_nm / wf.forge_cluster_line_time_s
    assert cluster_speed < step_speed
    assert step_speed > wf.forge_v_tip_nm_s, "实际线时还需经过针尖速度上限约束"


# ══════════════════════════════════════════════════════════════════════
# 闸门 A:没有死配置
# ══════════════════════════════════════════════════════════════════════

def test_every_workflow_field_has_a_consumer():
    """每个声明字段都应被相应消费方使用。"""
    fields = _wf_field_names()
    reads: set[str] = set()
    for src in _CONSUMER_SRCS:
        assert src.exists(), f"消费方模块不见了:{src}"
        reads |= _attr_reads(src)
    # 流程表自己也算(descend_pulse_v 由 descend_sequence 读)。
    reads |= _attr_reads(_WF_SRC)

    # ── 自检:检测器既不恒真也不恒假 ──────────────────────────────────
    assert len(fields) >= 36, f"只解析出 {len(fields)} 个字段,AST 走偏了"
    assert len(_CONSUMER_SRCS) >= 4
    assert "pulse_v" in reads, "已知被消费的字段都检不出来,检测器坏了"
    assert "definitely_not_a_workflow_field" not in reads, "检测器恒真"

    orphans = sorted(set(fields) - reads - _ORPHAN_EXEMPT)
    assert not orphans, (
        f"这些字段没有任何消费方(读起来像在生效的死配置):{orphans}。"
        "要么接上消费方,要么加进 _ORPHAN_EXEMPT 并写清楚理由与翻盘条件。")


def test_the_exemption_list_only_holds_fields_that_really_are_orphans():
    """豁免名单不许养一条已经修好的:修好了还挂着,下一条真的死配置就藏在里面。"""
    fields = set(_wf_field_names())
    reads: set[str] = set()
    for src in _CONSUMER_SRCS:
        reads |= _attr_reads(src)
    reads |= _attr_reads(_WF_SRC)
    for name in _ORPHAN_EXEMPT:
        assert name in fields, f"豁免名单里的 {name!r} 已经不是流程表字段了"
        assert name not in reads, (
            f"{name!r} 已经有消费方了 —— 请把它从 _ORPHAN_EXEMPT 里删掉")


# ══════════════════════════════════════════════════════════════════════
# 闸门 B:没有 ScanAt 调用点绕过唯一那个组装点
# ══════════════════════════════════════════════════════════════════════

def test_no_scan_at_call_site_bypasses_the_assembler():
    """扫描参数通过统一组装器生成。"""
    found = 0
    for src in _SCAN_CALLER_SRCS:
        for call in _scan_at_steps(src):
            found += 1
            params_kw = next((k for k in call.keywords if k.arg == "params"), None)
            assert params_kw is not None, f"{src.name}:ScanAt 步骤没有 params"
            val = params_kw.value
            ok = (isinstance(val, ast.Call) and isinstance(val.func, ast.Name)
                  and val.func.id == "scan_at_params")
            assert ok, (
                f"{src.name}:{ast.unparse(val)[:60]}… 没有走 scan_at_params —— "
                "这个调用点不会带上评估图的工作点")
    assert found >= 4, f"只找到 {found} 处 ScanAt 步骤,AST 匹配写歪了"


def test_the_assembler_emits_both_keys_when_the_workflow_has_them():
    """组装点本身:有值就写进去,None 就不写。

    ⚠️ 线时取 **1.0 s**(100 nm ⇒ 100 nm/s)而不是从前的 0.25 s:0.25 在 100 nm
    上是 400 nm/s,会被 修复项 的速度上限拽慢,于是这条测试比的就不再是它名字里
    那件事了。这里要证明的是「有值 → 写进去」与「None → 不写」这一对契约,
    所以刻意选一个**上限管不着**的值,把两件事分开测(限速本身由
    ``test_forge_line_time_never_exceeds_tip_speed_ceiling`` 那组管)。
    """
    wf = NobleTipWorkflow(step_pixels=128, step_line_time_s=1.0)
    p = scan_at_params(wf, 0.0, 0.0, size_nm=100.0,
                       pixels=wf.step_pixels, line_time_s=wf.step_line_time_s,
                       origin="clean_spot")
    assert p["pixels"] == 128 and p["line_time_s"] == pytest.approx(1.0)
    bare = scan_at_params(wf, 0.0, 0.0, size_nm=100.0, pixels=None,
                          line_time_s=None, origin="clean_spot")
    assert "pixels" not in bare and "line_time_s" not in bare


# ══════════════════════════════════════════════════════════════════════
# 闸门 C:工作点不许做成模型可填的参数
# ══════════════════════════════════════════════════════════════════════

#: 只有这一对是用户会逐字说出口的形状(「扫图**一律** 256 px / 0.15 s」);
#: 按阶段分的那四个是测量值,不是让模型逐阶段填的偏好。
_ALLOWED_WORKING_POINT_PARAMS = {"forge_pixels", "forge_line_time_s"}


def test_only_the_operator_shaped_override_is_a_tool_parameter():
    """全局覆盖和分阶段参数均须按声明优先级传递。"""
    from mast.skills.composite.forge_au_tip import _WF_PARAM_KEYS

    declared = {p.name for p in ForgeAuTip().metadata().parameters}
    exposed = {n for n in (set(_WF_PARAM_KEYS) | declared)
               if n.endswith(("_line_time_s", "_pixels"))}
    assert exposed == _ALLOWED_WORKING_POINT_PARAMS, (
        f"工作点参数面变了:{sorted(exposed)}。按阶段的那四个不该开给模型"
        "(它们是测量值);用户形状的那一对不该关掉(关掉 agent 会绕过整个技能)。")
    # ⑮ 的那条仍然成立:读了就必须声明,否则模型传了会被静默丢弃。
    assert _ALLOWED_WORKING_POINT_PARAMS <= declared
    assert _ALLOWED_WORKING_POINT_PARAMS <= set(_WF_PARAM_KEYS)
    # 自检:名单不是空的,否则上面几条什么都没检。
    assert len(declared) >= 12 and len(_WF_PARAM_KEYS) >= 8


def test_the_operator_override_reaches_scan_at():
    """用户逐字说的那个数,要一路到达 ScanAt —— 而且盖住按阶段的表值。

    这是 ⑮ 的完整形状:声明了、读了、还要**真的落到硬件参数上**。缺任何一段,
    调用方以为自己传了、技能以为没人传,而两端都看不出问题在哪。
    """
    _res, ctx = _run(_base_script(), max_sites=1,
                     forge_pixels=128, forge_line_time_s=0.4)
    calls = ctx.params_for("ScanAt")
    assert len(calls) >= 3
    for p in calls:
        assert p["pixels"] == 128, p
        assert p["line_time_s"] == pytest.approx(0.4), p
        resolved = ScanAt()._resolve(p)
        assert resolved.set_scan_buffer["pixels"] == 128
        assert resolved.configure_scan["line_time_s"] == pytest.approx(0.4)
    # 留空仍然走表值 —— 覆写是覆写,不是新默认。
    assert _forge_wf({}).step_line_time_s == pytest.approx(
        NOBLE_METAL_BASELINE.forge_step_line_time_s)


def test_an_envelope_substitution_reaches_the_report(monkeypatch):
    """出厂默认被针尖包络改过,**用户必须看得见**。

    只写 log 等于静默替换,而静默替换与静默夹紧一样坏 —— 后果很具体:
    报告说「打满 20 发仍未跳变」,而打的是通用档的 3 V 不是表上的 10 V,
    用户会去调一个根本没在生效的数。

    (第一版把 notes 丢进了 `_` 变量 —— 就在写完「每一次替换都出声」那句注释之后。)
    """
    import mast.core.tip_conditioning_policy as pol
    import mast.core.tip_state as tip_state

    monkeypatch.setattr(tip_state, "current_tip_facts", lambda: None,
                        raising=False)
    # 载体必须是**合成的紧包络**,不能再靠「通用档恰好比出厂默认严」——
    # 2026-08-12 起所有档的包络都拉满了(10 V / 10 nm),真跑时一个值都不会被换,
    # 于是这条测试会因为「没有替换发生」而红,而它要考的其实是
    # **替换一旦发生,那句话到不到得了报告**。把两件事解耦。
    monkeypatch.setattr(pol, "resolve_policy",
                        lambda *a, **kw: {"max_abs_pulse_v": 3.0,
                                          "max_poke_depth_m": 5.0e-10},
                        raising=False)
    res, _ctx = _run(_base_script(), max_sites=1)
    notes = res.data.get("envelope_notes") or []
    assert notes, (
        "未登记针尖上出厂默认被降到了通用档,而报告里一个字都没提;"
        f"data 的键:{sorted(res.data)}")
    assert any("pulse_v" in n for n in notes), notes
    assert "安全包络" in res.data["summary_cn"], res.data["summary_cn"]


def test_a_slow_override_does_not_walk_back_into_the_timeout_bug():
    """覆盖工作点覆写后的等待预算：分辨率与线时间增加时，预算必须随帧时推导。"""
    _res, ctx = _run(_base_script(), max_sites=1,
                     forge_pixels=512, forge_line_time_s=1.0)
    calls = ctx.params_for("ScanAt")
    assert calls
    for p in calls:
        est = ScanAt()._resolve(p).estimated_scan_s
        budget = ScanAt._wait_timeout_s(p, est)
        assert est > 900, f"探针失效:这组参数只要 {est:.0f} s,测不到预算问题"
        assert budget >= 1.3 * est, (
            f"帧要 {est:.0f} s 而等待预算只有 {budget:.0f} s —— "
            "慢档覆写把 2026-08-10 修掉的那个超时缺陷带回来了")


def test_the_budget_still_honours_the_workflow_floor():
    """派生只**抬高**预算,不把表里那个数压下去。

    默认工作点派生出来的预算(2026-08-15 起:256 px × 0.586 s × 2 = 300 s/帧
    ⇒ 300×1.3+30 = 420 s)低于流程表写的 ``forge_scan_timeout_s`` —— 取大的那个,
    否则这次改动会悄悄把既有的余量削掉。
    (改前的数是 76.8 s/帧 ⇒ 130 s vs 表里 180 s,同一个方向。)
    """
    from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as base
    from mast.skills.composite._tip_phases import scan_at_params

    p = scan_at_params(_forge_wf({}), 0.0, 0.0, size_nm=100.0,
                       pixels=base.forge_step_pixels,
                       line_time_s=base.forge_step_line_time_s,
                       origin="clean_spot")
    assert p["wait_timeout_s"] == pytest.approx(base.forge_scan_timeout_s)


def test_the_override_is_visible_in_the_tool_schema():
    """探针有效性:中间隔着一层 args_schema,当初⑮ 断的正是那一层。

    两个参数走的是**两条**链,所以分开断:
      * `forge_line_time_s` 是带单位的 float ⇒ 进 schema 是**字符串**,由
        `_coerce_si_params` 转回 float(防指数丢失那套机制);
      * `forge_pixels` 是 int ⇒ 原样进 schema,`_si_params` 明确把它排除
        (「像素数没有指数可丢」)。拿字符串去断言它会被转,断的是别人的链。
    """
    from mast.agents._shared.skill_adapter import (
        _coerce_si_params,
        _si_params,
        wrap_skill,
    )

    meta = ForgeAuTip().metadata()
    tool = wrap_skill(ForgeAuTip, lambda: None)
    fields = tool.args_schema.model_fields
    for name in sorted(_ALLOWED_WORKING_POINT_PARAMS):
        assert name in fields, sorted(fields)

    si = _si_params(meta)
    assert "forge_line_time_s" in si and "forge_pixels" not in si, si
    kw, errs = _coerce_si_params(
        meta, {"forge_pixels": 128, "forge_line_time_s": "0.4"})
    assert not errs
    assert kw["forge_pixels"] == 128
    assert kw["forge_line_time_s"] == pytest.approx(0.4)
    # 值真的落到 ScanAt 的实参上(而不是停在 schema 层)—— 与 ⑮ 同一条链。
    wf = _forge_wf(kw)
    assert wf.step_pixels == 128 and wf.cluster_pixels == 128
    assert wf.step_line_time_s == pytest.approx(0.4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
