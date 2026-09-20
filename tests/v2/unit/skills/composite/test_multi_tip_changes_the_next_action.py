"""验证多针尖判决确实改变下一步动作，并把历史落点传到下一批选点器。"""
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

import inspect  # noqa: E402
import re  # noqa: E402

import pytest  # noqa: E402

from mast.core.noble_tip_workflow import NobleTipWorkflow  # noqa: E402
from mast.skills.composite import _tip_phases as TP  # noqa: E402
from mast.skills.composite.prepare_noble_tip import PokeConditionTip  # noqa: E402

from .test_prepare_noble_tip import (  # noqa: E402
    FakeCtx,
    SCAN_FILE,
    cluster,
    flat_sites,
    poke,
    run,
    spot,
)


def _ctx(verdicts=("cluster",) * 8, *, sites_per_scan=6, **extra):
    script = {"FindCleanSpot": spot(20.0),
              "TipShapeWithReadback": poke(list(verdicts)),
              "SaveScan": SCAN_FILE, "GetLatestScanFile": SCAN_FILE,
              "FindFlatRegion": flat_sites(n=sites_per_scan),
              "AssessClusterRoundness": cluster()}
    script.update(extra)
    return FakeCtx(script)


def _verdict(monkeypatch, verdict, score=0.19):
    """这里只替换判据返回值，验证调用方如何消费判决；判据准确性由独立测试覆盖。"""
    monkeypatch.setattr(
        TP, "_step_split_look",
        lambda *a, **k: {"verdict": verdict, "score": score,
                         "score_threshold": 0.16, "levels_pm": [0.0, 235.5]},
        raising=True)


# ── Q1:多针尖必须改变下一步 ────────────────────────────────────────────────

def test_a_split_tip_stops_the_poking_even_when_a_terrace_was_found(monkeypatch):
    """**找到台面了,也照样不许扎。** 这就是那 7 次被丢掉的判决。

    「找不到台面」是多针尖的一个**后果**(多针尖会把台面分裂,导致找不到台面),
    不是它的**判据**。拿后果当判据,就漏掉了后果
    还没显现的那些 —— 而那是 9 次里的 7 次。
    """
    _verdict(monkeypatch, "split")
    ctx = _ctx()
    res = run(PokeConditionTip(), ctx, poke_budget=8)
    out = res.data["phases"][0]

    assert out.get("split_tip") is True, "判决没传到精修相"
    assert out.get("needs_pulse") is True, (
        "多针尖没有把流程推去打脉冲 —— 这正是要求的『没有针对性的处置』")
    assert ctx.count("TipShapeWithReadback") == 0, (
        "台阶被劈开还在扎针:扎针改的是形状,长不回一个顶点")
    assert "脉冲" in str(out.get("reason") or ""), (
        f"理由没说清下一步是打脉冲:{out.get('reason')!r}")


def test_a_single_tip_verdict_still_hands_the_sites_over(monkeypatch):
    """反例:判成单针尖时,一切照旧。

    只断言「split 会拦」是不够的 —— 一个恒拦的闸门也能让上一条通过。
    """
    _verdict(monkeypatch, "single", score=0.11)
    ctx = _ctx()
    res = run(PokeConditionTip(), ctx, poke_budget=4)
    out = res.data["phases"][0]

    assert not out.get("split_tip"), "单针尖被当成了多针尖"
    assert ctx.count("TipShapeWithReadback") > 0, "单针尖时反而不扎了"


def test_undecidable_never_vetoes(monkeypatch):
    """三态:「判不了」**不否决**。

    未标定的阈值没资格否决人;而这条阈值标定过,所以只有它真的说 ``split``
    时才有否决权。``undecidable`` 与 ``single`` 在这里的行为必须一样。
    """
    _verdict(monkeypatch, "undecidable", score=0.0)
    ctx = _ctx()
    res = run(PokeConditionTip(), ctx, poke_budget=4)
    out = res.data["phases"][0]

    assert not out.get("split_tip")
    assert not out.get("needs_pulse"), "「判不了」把流程推去打脉冲了"
    assert ctx.count("TipShapeWithReadback") > 0


def test_the_veto_is_driven_by_an_equality_not_by_truthiness():
    """否决必须由 ``verdict == "split"`` 派生,**不是** ``bool(verdict)``。

    ``step_splitting`` 的三态里 ``"single"`` 和 ``"undecidable"`` 都是真值字符串
    —— 写成 ``if sp.get("verdict"):`` 会让**每一张图**都否决,而两条正例断言
    仍然全绿。同一条钉子 ``verify_phase`` 那侧已经有一根(见
    ``test_leveling_actually_runs``),这里钉的是 D 相这一侧。
    """
    src = inspect.getsource(TP.flat_poke_sites)
    hits = re.findall(r'if\s+str\(sp\.get\("verdict"\)[^\n]*==\s*"split"', src)
    assert hits, (
        "flat_poke_sites 里找不到一处对 verdict 的等值比较 —— "
        "要么闸门没了,要么它退化成了真值判断")


# ── Q3:别再扎回自己刚扎的坑 ────────────────────────────────────────────────

def test_the_landing_points_reach_the_site_picker(monkeypatch):
    """历史落点必须传给选点器，批内间距限制不能代替跨批避让。"""
    _verdict(monkeypatch, "single", score=0.11)
    # 一次只给一个落点 ⇒ 每扎一针就得重扫一张图再要一批,排除表才有机会说话。
    ctx = _ctx(sites_per_scan=1)
    run(PokeConditionTip(), ctx, poke_budget=4, critical_repeat_n=99)

    calls = ctx.params_for("FindFlatRegion")
    assert len(calls) >= 3, f"没触发重扫,这条测不到东西(只有 {len(calls)} 次)"
    assert all("exclude_used_spots" in p for p in calls), (
        "FindFlatRegion 的排除表这个键根本没传 —— 它一直就在,只是没人用")

    first = ctx.params_for("TipShapeWithReadback")
    assert first, "一针都没扎"
    # 第一次问的时候还没扎过,排除表该是空的;之后每一次都必须带上已扎的点。
    assert calls[0]["exclude_used_spots"] == "", "还没扎就有排除项"
    later = [p["exclude_used_spots"] for p in calls[1:]]
    assert all(s for s in later), f"扎过之后排除表还是空的:{later!r}"
    # 累计各轮已使用的落点。
    counts = [s.count(";") + 1 for s in later]
    assert counts == sorted(counts) and counts[-1] > counts[0], (
        f"排除表没有累积:{counts}")


def test_the_exclusion_string_is_readable_by_the_skill_that_consumes_it():
    """往返:``_spot_list`` 写出来的串,``FindFlatRegion`` 必须**全部**认得。

    ⚠️ 这条不是形式主义。``_parse_excluded`` 的自述写着:解析不了的块从前是
    静默 ``continue`` —— 于是「排除这几个已用过的点」会悄悄变成「一个都不排除」,
    然后技能高高兴兴地把针尖送回刚才那个坏点。所以格式错了必须当场看见。
    """
    from mast.skills.builtins.flat_region import FindFlatRegion

    pts = [(-4.79e-7, -3.87e-7), (1e-9, 0.0), (0.0, 0.0), (-1.2e-6, 3.5e-9)]
    s = TP._spot_list(pts)
    got, bad = FindFlatRegion._parse_excluded(s)

    assert bad == [], f"技能看不懂自己被喂的这些块:{bad!r}"
    assert len(got) == len(pts)
    for (gx, gy), (px, py) in zip(got, pts):
        assert gx == pytest.approx(px, rel=1e-6, abs=1e-15)
        assert gy == pytest.approx(py, rel=1e-6, abs=1e-15)


def test_the_exclusion_list_dedupes_and_survives_junk():
    """同一个点只写一次;坏数据丢掉而不是让整串报废。"""
    s = TP._spot_list([(1e-9, 2e-9), (1e-9, 2e-9)], [(1e-9, 2e-9), (3e-9, 4e-9)])
    assert s.count(";") == 1, f"没去重:{s!r}"

    from mast.skills.builtins.flat_region import FindFlatRegion
    got, bad = FindFlatRegion._parse_excluded(
        TP._spot_list([(1e-9, 2e-9), (float("nan"), 0.0), ("x", "y")]))
    assert bad == [] and got == [pytest.approx((1e-9, 2e-9))]


def test_the_exclusion_radius_is_wider_than_the_cluster_frame():
    """排除半径必须**大于簇图视野** —— 否则旧坑照样进画面。

    这是这一条修复的全部算术:排除半径 30 nm > 簇图 10 nm ⇒ 上一针的坑
    不可能再落进下一针的簇图里,判据也就没有机会把它数成第二个顶点。
    「移除诱因」而不是「给诱因造一个检测器」。
    """
    wf = NobleTipWorkflow()
    assert wf.poke_site_separation_nm > wf.cluster_scan_nm, (
        f"排除半径 {wf.poke_site_separation_nm} nm 没有超过簇图视野 "
        f"{wf.cluster_scan_nm} nm —— 旧坑还会进画面")


# ── Q2:「连续扎针圆 N 次」那个 N ────────────────────────────────────────────

def test_three_rounds_in_a_row_is_enough_to_finish():
    """达到配置要求的连续达标次数后立即结束，不继续干预。"""
    assert NobleTipWorkflow().critical_repeat_n == 3

    ctx = _ctx(verdicts=["cluster"] * 6)
    res = run(PokeConditionTip(), ctx, poke_budget=6)
    out = res.data["phases"][0]

    assert out["refined"] is True, out.get("reason")
    assert out["repeats"] == 3, f"收工时的连续次数不是 3:{out['repeats']}"
    # 一次深扎进入临界期后，严格执行配置要求的临界期次数。
    assert ctx.count("TipShapeWithReadback") == 1 + 3, (
        "临界期扎的针数不等于要求的连续次数 —— 多扎一针就是在好针尖上多赌一次")


def test_the_gate_still_needs_them_consecutive():
    """调小的是**次数**,不是「连续」这个词。中间断一次要从头数。

    「连续 N 次没被否掉」和「连续 N 次确实圆」是两回事 —— 前者把「判不了」
    也算成合格,而判不了恰恰是最该继续扎的那一档。
    """
    def look(params, n):
        return cluster(round_=False)(params, n) if n == 1 else cluster()(params, n)

    ctx = _ctx(verdicts=["cluster"] * 8, AssessClusterRoundness=look)
    res = run(PokeConditionTip(), ctx, poke_budget=8)
    out = res.data["phases"][0]

    assert out["refined"] is True
    assert ctx.count("TipShapeWithReadback") == 5, (
        "第 2 针不圆没有把计数打回零(1 针 + 断 1 针 + 再 3 针 = 5)")


def test_the_special_tip_line_still_asks_for_no_more_than_the_baseline():
    """取 3 而不是 2 的第三条理由:保住这条已经钉着的不变式。

    特异化针尖那条线**刻意**要得更少(``SPECTROSCOPY_TIP.critical_repeat_n = 2``)。
    把基线也压到 2 会把这个区别抹平 —— 而抹平之后没有任何东西会报警。
    """
    from mast.core.special_tip_workflow import SPECTROSCOPY_TIP

    assert SPECTROSCOPY_TIP.critical_repeat_n <= NobleTipWorkflow().critical_repeat_n
    assert SPECTROSCOPY_TIP.critical_repeat_n < NobleTipWorkflow().critical_repeat_n, (
        "两条线要的次数变成一样了 —— special 那条线要得更少是有意的")
