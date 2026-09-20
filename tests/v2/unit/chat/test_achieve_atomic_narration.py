# -*- coding: utf-8 -*-
"""原子分辨流程每次判断、升级、停止和结局都应产生旁白。

测试从实际发出方经过 narrate 与 ConversationStore，再读取落库记录，
以核验真实载荷层次与模板路径一致，而不只测试模板对手写平铺字典的渲染。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

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

import pytest  # noqa: E402

from mast.chat import narration as N  # noqa: E402
from mast.chat import narration_templates as T  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core.turn_context import turn_scope  # noqa: E402
from mast.skills.composite import achieve_atomic as A  # noqa: E402
from mast.skills.composite.achieve_atomic import (  # noqa: E402
    AchieveAtomicResolution)

CID = "conv-achieve-atomic-narration"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """真的 ``ConversationStore``,建在 tmp 上。

    **必须显式 set_store**:本仓「测试污染真实用户数据」已经发生五次,而旁白写的
    正是用户真实会话所在的那个库。``MAST2_PROJECT_ROOT`` 同理 —— 三联图产物
    绝不许落进用户真实的 artifacts。
    """
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    N.reset_for_tests()
    st = ConversationStore(tmp_path / "chat" / "mast_conversations.db")
    N.set_store(st)
    yield st
    N.flush(2.0)
    N.reset_for_tests()


# ── 驱动真的 plan_dynamic / aggregate ───────────────────────────────────────


class _Res:
    def __init__(self, data):
        self.data = dict(data or {})


def _drive(store, params=None, scripted=None, max_steps=40):
    """跑**真的** ``plan_dynamic`` + ``aggregate``,把落库的旁白读回来。

    形状照 ``tests/v2/unit/skills/composite/test_achieve_atomic.py::_drive``
    (那边测编排,这边测它说了什么)—— 两边共用同一个假执行器,
    于是「编排改了而旁白没跟上」在两处之一会当场红。
    """
    from mast.skills.composite.graph_executor import CompositeProgress

    scripted = dict(scripted or {})
    scripted.setdefault("p0:zctrl", {"controller_on": True,
                                     "setpoint": 5e-10, "z_m": -1.2e-8})

    class _Ex:
        def __init__(self):
            self.progress = CompositeProgress("AchieveAtomicResolution")
            self.sub_results = {}

        def set_partial(self, k, v):
            self.progress.partial_data[k] = v

        def set_total_steps(self, n):
            pass

    ex = _Ex()
    seen: list[str] = []
    with turn_scope(conversation_id=CID, run_id="run-1"):
        for st in AchieveAtomicResolution().plan_dynamic(dict(params or {}), ex):
            seen.append(st.step_id)
            ex.sub_results[st.step_id] = _Res(scripted.get(st.step_id, {}))
            if len(seen) > max_steps:  # pragma: no cover — 防跑飞
                pytest.fail("步骤停不下来:%s" % seen)
        out = AchieveAtomicResolution().aggregate({}, ex.progress)
    assert N.flush(10.0), "旁白写线程没排空"
    rows = [r for r in store.messages_since(CID, 0)
            if r["kind"] == N.NARRATION_KIND]
    return rows, out, seen


def _texts(rows) -> str:
    return "\n".join(r["text"] for r in rows)


def _kinds(rows) -> list:
    return [json.loads(r["meta"])["nk"] for r in rows]


def _absent(n=3, concs=(95.5, 126.3, 61.0)):
    return {"found": False, "attempts": n, "n_undetermined": 0,
            "history": [{"attempt": i + 1, "verdict": "absent",
                         "concentration": concs[i % len(concs)],
                         "path": "f%d.sxm" % (i + 1)} for i in range(n)]}


def _undetermined(n=3):
    return {"found": False, "attempts": n, "n_undetermined": n,
            "history": [{"attempt": i + 1, "verdict": "undetermined"}
                        for i in range(n)]}


def _found(path="X.sxm"):
    return {"found": True, "found_at_attempt": 1, "found_path": path,
            "history": [{"attempt": 1, "verdict": "atomic",
                         "concentration": 293.4}]}


# ── ① 这个文件存在的理由:一条兜底都不许有 ────────────────────────────────


def test_not_one_line_is_the_fallback(store):
    """**发出方发的形状必须渲染得出话。** 一条兜底都不许有。

    这是 ``poke_indent`` 30/30、``auto_tilt_result`` 26/26 那个坑的守卫:
    两边各自都对、放在一起对不上,不报错、不崩,只是永远说一句语法完全正确的
    空话。模板自测证明不了发出方发对了形状,只有从发出方进才证得到。
    """
    rows, _out, _seen = _drive(store, scripted={
        "r1": _absent(), "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    assert rows, "一条旁白都没有"
    fallbacks = {t.kind: t.fallback for t in T.TEMPLATES.values()}
    bad = [r["text"] for r in rows
           if r["text"] == fallbacks.get(json.loads(r["meta"])["nk"])]
    assert not bad, "这些走了兜底:%s" % bad
    degraded = [r["text"] for r in rows
                if json.loads(r["meta"]).get("degraded") is True]
    assert not degraded, "这些标着 degraded(必需字段没读到):%s" % degraded


def test_every_rung_speaks_and_carries_its_numbers(store):
    """**每一档都要出旁白,而且带读数。**

    用户看这条流程时问的是「它在想什么、为什么升级、为什么停」——
    一句只报告状态、不带数字的话,那三个问题一个都答不了。
    """
    rows, _out, _seen = _drive(store, scripted={
        "r1": _absent(), "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(4, (8.1, 14.2, 11.7, 9.9)),
        "r3b:recheck": _absent()})
    text = _texts(rows)

    # Phase 0 现状:反馈 / setpoint / Z / 预算
    assert "开始追原子分辨" in text, text
    assert "500 pA" in text, "setpoint 没进句子:%s" % text
    assert "反馈**开着**" in text or "反馈开着" in text, text

    # R1:扫了几帧、每帧的角向集中度
    assert "重扫档" in text
    assert "95.5" in text and "126.3" in text, "R1 每帧的读数没进句子:%s" % text

    # 每一档的升级理由
    assert "偏压抖动" in text, "R2 没出声:%s" % text
    assert "脉冲档" in text and "浅扎档" in text, "R3 两档没出声:%s" % text
    assert "8.1" in text and "14.2" in text, "脉冲后复评的读数没进句子:%s" % text
    assert "锻造" in text, "R4「没开」那一支没出声:%s" % text


def test_the_facts_carry_the_numbers_back(store):
    """meta.facts 必须保存能与旁白数值对应的输入字段；空 facts 应暴露载荷路径不匹配。"""
    rows, _out, _seen = _drive(store, scripted={"r1": _absent()})
    rung_rows = [r for r in rows if json.loads(r["meta"])["nk"] == "atomic_rung"]
    assert rung_rows
    for r in rung_rows:
        facts = json.loads(r["meta"]).get("facts") or {}
        assert "outcome" in facts and "rung" in facts, (
            "这条旁白的数对不回去:%s / %s" % (r["text"], facts))
        assert "at_min" in facts, "没记下这是起跑后第几分钟:%s" % facts


# ── ①b 「扫了几帧」必须是**真采到的帧数** ────────────────────────────────


def _absent_with_ledger(n_judged=1, n_failed=2, n_repeat=0, attempts=3,
                        concs=(95.5,)):
    """一轮 ``ScanUntilAtomicResolution`` 的回包 —— 带 2026-08-24 加的那本账。

    形状照生产方 ``scan_until_atomic.aggregate`` 的输出（下面
    ``test_the_ledger_keys_are_the_ones_the_producer_writes`` 钉住这件事）。
    """
    return {
        "found": False, "attempts": attempts, "n_undetermined": 0,
        "frames_judged": n_judged, "scans_failed": n_failed,
        "repeat_frames": n_repeat,
        "history": [{"attempt": i + 1, "verdict": "absent",
                     "concentration": concs[i % len(concs)],
                     "path": "f%d.sxm" % (i + 1)} for i in range(n_judged)],
    }


def test_the_rung_says_how_many_frames_were_really_acquired(store):
    """旁白应报告实际判断过的不同帧数，而不是文件副本数或尝试次数。
    尝试数与独立采集数不一致时，说明差额，避免重复数据冒充新证据。"""
    rows, _out, _seen = _drive(store, scripted={
        "r1": _absent_with_ledger(), "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent_with_ledger(), "r3b:recheck": _absent_with_ledger()})
    text = _texts(rows)
    assert "真的采到并判过" in text, "还在按 attempt 数报帧数:%s" % text
    assert "发起过" in text and "次 attempt" in text, (
        "attempt 数与真帧数不一致却没说出来:%s" % text)
    assert "根本没采到" in text, "没说清那两次 attempt 什么都没采到:%s" % text
    assert "不存盘、不判定" in text, (
        "没说清「没采到就不存盘」—— 那正是修掉的那个 bug:存的是上一帧的缓冲:%s"
        % text)


def test_a_clean_round_does_not_cry_wolf(store):
    """反证:三次 attempt 三次真采集时,**不许**冒出那两句告警。

    没有这一条,上面那条只证明了「这几个词出现在模板里」。
    """
    rows, _out, _seen = _drive(store, scripted={
        "r1": _absent_with_ledger(n_judged=3, n_failed=0, attempts=3,
                                  concs=(95.5, 126.3, 61.0)),
        "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent_with_ledger(n_judged=3, n_failed=0, attempts=3),
        "r3b:recheck": _absent_with_ledger(n_judged=3, n_failed=0, attempts=3)})
    text = _texts(rows)
    assert "真的采到并判过" in text
    assert "根本没采到" not in text, "一切正常却报了警:%s" % text
    assert "没给出新样本" not in text, text
    assert "发起过" not in text, "两个数相等时不该再摆一遍 attempt 数:%s" % text


def test_a_repeat_frame_is_named_in_the_narration(store):
    """「交回来的还是判过的那张 / 读数与上一帧逐位相同」也要出声。"""
    rows, _out, _seen = _drive(store, scripted={
        "r1": _absent_with_ledger(n_judged=1, n_failed=0, n_repeat=2)})
    text = _texts(rows)
    assert "没给出新样本" in text, text
    assert "逐位相同" in text, "没说清「同一块数据被存了两份」这件事:%s" % text


def test_a_frame_that_could_not_be_judged_is_not_silent(store):
    """采到了帧、判据却没跑成 ⇒ 也要出声,而且要说清它**不是**「没有原子分辨」。

    不说的话屏幕上只剩一个「真的采到并判过 0 帧」,而**为什么是 0** 无从查起 ——
    生产方明明算了这个数(``scan_until_atomic`` 的 ``assess_failed``),
    只是没人念它。「生产方接好了、消费方缺席」在本仓是常客。
    """
    d = _absent_with_ledger(n_judged=0, n_failed=0, attempts=2)
    d["assess_failed"] = 2
    rows, _out, _seen = _drive(store, scripted={"r1": d})
    text = _texts(rows)
    assert "判据却没跑成" in text, text
    assert "不是「没有原子分辨」" in text, text
    facts = [json.loads(r["meta"]).get("facts") or {} for r in rows
             if json.loads(r["meta"])["nk"] == "atomic_rung"]
    assert any(f.get("assess_failed") == 2 for f in facts), facts


def test_the_ledger_numbers_can_be_traced_back(store):
    """``meta.facts`` 要能把这三个新数对回原始读数。

    模板的 ``records`` 漏一个,句子照样念得出来,只是 facts 里查不到它 ——
    **一个查不到出处的数,和一个编出来的数在报告里长得一模一样。**
    """
    rows, _out, _seen = _drive(store, scripted={"r1": _absent_with_ledger()})
    rung = [r for r in rows
            if json.loads(r["meta"])["nk"] == "atomic_rung"
            and "真的采到并判过" in r["text"]]
    assert rung, "带帧数的那条旁白没发出来"
    facts = json.loads(rung[0]["meta"]).get("facts") or {}
    for k in ("frames_judged", "scans_failed", "repeat_frames"):
        assert k in facts, "%s 对不回原始读数:%s" % (k, facts)


def test_the_ledger_keys_are_the_ones_the_producer_writes():
    """**发出方读的键,生产方真的写。**

    本仓栽过一次同形的:发出方从 ``verify`` 读六个键,而验证方只回三个 ——
    三个数**永远是 None**,句子若无其事地少说三样,而两边的单测各自都绿。
    所以这里不比对源码字符串,直接跑**真的** ``aggregate``,看键在不在。
    """
    from mast.skills.composite.graph_executor import CompositeProgress
    from mast.skills.composite.scan_until_atomic import ScanUntilAtomicResolution

    prog = CompositeProgress("ScanUntilAtomicResolution")
    prog.partial_data.update({
        "attempts_done": 3, "scans_ok": 1, "scans_failed": 2,
        "history": [{"attempt": 1, "verdict": "absent", "concentration": 95.5}]})
    produced = ScanUntilAtomicResolution().aggregate({}, prog)
    wanted = set(A._scan_facts(produced))
    missing = sorted(k for k in wanted
                     if k != "concentrations" and k not in produced)
    assert not missing, (
        "旁白从这几个键读数,而生产方根本不产出它们 —— 它们永远是 None:%s"
        % missing)
    # 而且读回来的确实是生产方算的那几个数,不是 None
    assert A._scan_facts(produced)["frames_judged"] == 1
    assert A._scan_facts(produced)["scans_failed"] == 2


# ── ② 分诊:两条最要紧的话必须说出口 ─────────────────────────────────────


def test_the_undecidable_triage_says_why_it_will_not_touch_the_tip(store):
    """分诊一「判不了 ≠ 没有」:走这一支时要**说清为什么不动针尖**。

    这是本技能唯一的安全属性在屏幕上的样子。拿「判不了」当「没有」去修针,
    就是在自己造出来的空白上判读 —— 而用户看不见这条推理时,
    只会看到「它扫了三帧然后什么都没做」。
    """
    rows, out, _seen = _drive(store, scripted={"r1": _undetermined(3)})
    text = _texts(rows)
    assert "判不了" in text
    assert "不等于" in text, "必须明说它不等于「没有原子分辨」:%s" % text
    assert "不动针尖" in text or "针尖都不动" in text, (
        "没说清为什么不动针尖:%s" % text)
    # 结局那一条要把 aggregate 写的那段处置建议念出来
    assert "remedy" in text or "成像条件" in text, text
    assert out.get("advice")


def test_the_surface_triage_names_the_verdict_and_the_decision(store,
                                                               monkeypatch):
    """分诊二「表面 vs 针尖」:要念出 ``FindFlatRegion`` 的 verdict,
    以及**换不换区**、地图给的 axis/direction/steps 与理由。"""
    monkeypatch.setattr(A, "_coarse_suggestion",
                        lambda: ({"axis": "x", "direction": "+", "steps": 300,
                                  "reason": "往 +x 还有没去过的地方"},
                                 "往 +x 还有没去过的地方"))
    rows, _out, seen = _drive(store, scripted={
        "r1": _absent(), "surface": {"verdict": "no_usable_region"},
        "r1#2": _absent(), "surface#2": {"verdict": "ok"},
        "r2#2": {"outcome": "no_tip"},
        "r3a#2:recheck": _absent(), "r3b#2:recheck": _absent()})
    text = _texts(rows)
    assert "relocate" in seen, seen
    assert "no_usable_region" in text, "表面判据的 verdict 没念出来:%s" % text
    assert "表面**已无可用台面**" in text or "已无可用台面" in text, text
    assert "300" in text and "x+" in text, (
        "地图给的 axis/direction/steps 没念出来:%s" % text)
    assert "往 +x 还有没去过的地方" in text, "地图给的理由没念出来:%s" % text
    assert "坐标代次" in text, "换区之后旧坐标作废这件事没说:%s" % text
    # 第二个站点要认得出自己是第二站
    assert "第 2 站" in text, "换区之后的旁白分不出站点:%s" % text


def test_a_usable_surface_says_it_is_the_tips_problem(store, monkeypatch):
    """反证:表面还有台面 ⇒ 旁白要说这是**针尖**的问题,而不是沉默着去修针。"""
    monkeypatch.setattr(A, "_coarse_suggestion", lambda: (None, "x"))
    rows, _out, _seen = _drive(store, scripted={
        "r1": _absent(), "surface": {"verdict": "ok"},
        "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    text = _texts(rows)
    assert "这是**针尖**的问题" in text or "针尖**的问题" in text, text


def test_feedback_off_explains_itself_instead_of_going_quiet(store):
    """反馈没开那一支:要说清「这不是针尖的问题」并且**不替用户进针**。"""
    rows, _out, seen = _drive(store,
                              scripted={"p0:zctrl": {"controller_on": False}})
    assert seen == ["p0:zctrl"], seen
    text = _texts(rows)
    assert "反馈**没开**" in text or "反馈没开" in text, text
    assert "不替你进针" in text, "没说清为什么不自己进针:%s" % text
    assert "撞针" in text


def test_an_unreadable_controller_state_is_not_read_as_off(store):
    """``controller_on`` 读不到时**不许念成「没开」**。

    「读不到」被折叠成一个具体的值,是本仓一天里出现过五次的形状。
    这里三态各说各的:开着 / 没开 / 读不到。
    """
    rows, _out, _seen = _drive(store, scripted={
        "p0:zctrl": {}, "r1": _absent(), "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    begin = [r["text"] for r in rows
             if json.loads(r["meta"])["nk"] == "atomic_begin"]
    assert begin, "开跑那条旁白没发出来"
    assert "读不到" in begin[0], begin[0]
    assert "没开" not in begin[0], "把「读不到」念成了「没开」:%s" % begin[0]


# ── ③ 结局:成或不成都要有一段带分析图的旁白 ──────────────────────────────


def _lattice_sxm(path: Path, n: int = 64) -> Path:
    """一张**带真晶格**的最小 .sxm(正反扫两块,合成二维正弦)。

    形状照 ``test_atomic_result_narration.py::_lattice_sxm`` —— 三联图要在
    谱上找得到峰才画得出东西。
    """
    header = (
        ":SCAN_PIXELS:\n" f"{n} {n}\n"
        ":SCAN_OFFSET:\n" "0.0 0.0\n"
        ":SCAN_RANGE:\n" "3E-9 3E-9\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.0\t0.0\n"
        "\n:SCANIT_END:\n"
    )
    y, x = np.mgrid[0:n, 0:n]
    img = (np.sin(x * 0.9) + np.sin(y * 0.9)) * 3e-11
    path.write_bytes(header.encode() + b"\x1a\x04"
                     + img.astype(">f4").tobytes()
                     + img[:, ::-1].astype(">f4").tobytes())
    return path


def test_a_run_that_failed_still_narrates_its_result_with_the_plot(store,
                                                                  tmp_path):
    """**没拿到的时候也要说,而且照样挂三联分析图。**

    要求的那一句:「无论成或不成」。失败时那张图恰恰是「为什么不算」的
    证据 —— 功率谱上没有离散峰、剖面没有周期,一眼看得出来;而一句
    「走完了允许的档位仍没拿到」看不出任何东西。
    """
    sxm = _lattice_sxm(tmp_path / "last_frame.sxm")
    hist = _absent()
    hist["history"][-1]["path"] = str(sxm)
    rows, out, _seen = _drive(store, scripted={
        "r1": hist, "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(), "r3b:recheck": _absent()})

    final = [r for r in rows if json.loads(r["meta"])["nk"] == "atomic_run_result"]
    assert len(final) == 1, "整跑结局旁白应当正好一条,实际 %d" % len(final)
    text = final[0]["text"]
    assert "没有拿到" in text, text
    assert "走过" in text and "档" in text, "没说代价花在哪:%s" % text
    # aggregate 写的那段处置建议要原样念出来
    assert "ForgeAuTip" in text, "advice 没念出来:%s" % text
    assert unicode_head(out["advice"]) in text, (
        "念出来的和 aggregate 写的不是同一段:%s" % text)

    out_dir = tmp_path / "artifacts" / "atomic_reports"
    pngs = list(out_dir.glob("*.png")) if out_dir.is_dir() else []
    assert pngs, "失败的那一跑没有挂分析图 —— %s 里一个 png 都没有" % out_dir
    img = json.loads(final[0]["meta"]).get("image") or {}
    assert img.get("src") == str(pngs[0]), "挂的不是刚画的那张:%s" % img
    assert img.get("origin") == "milestone_png", (
        "origin 必须是 milestone_png —— ``sxm`` 那一档没有生产方,"
        "取图端点连门都不开(挂着图却永远 404 比没有图更坏)")


def test_the_failed_run_describes_the_analysis_not_just_the_verdict(store,
                                                                   tmp_path):
    """要求的是「**详细介绍其分析结果**,包括分析图,无论成或不成」。

    在此之前没拿到那一支只说一句「没有拿到」+ 一个角向集中度。而三联图渲染时
    **本来就量过**反扫、周期、帧内两半 —— 那三样正是「为什么不算」的证据
    (功率谱上没有离散峰、只在一个方向上有周期、两半差得远),它们只是从没往
    这条路上交过。

    ⚠️ 这一条同时守着「键名要对着生产方读」:这四个数的生产方是
    ``atomic_report_plot._render`` 里那个 ``stats.update({...})``。上一版在
    ``verify`` 上栽过 —— 六个键里三个生产方根本不产出,句子若无其事地少说三样。
    """
    sxm = _lattice_sxm(tmp_path / "last_frame.sxm")
    hist = _absent()
    hist["history"][-1]["path"] = str(sxm)
    rows, _out, _seen = _drive(store, scripted={
        "r1": hist, "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    final = [r for r in rows
             if json.loads(r["meta"])["nk"] == "atomic_run_result"][0]
    text, facts = final["text"], json.loads(final["meta"]).get("facts") or {}

    assert "反扫" in text, "没说反扫 —— 「只在一个方向上有周期」查不出来:%s" % text
    assert "针尖产物" in text, text
    assert "周期" in text, "没说量到的周期:%s" % text
    for k in ("concentration", "concentration_bwd", "period_nm"):
        assert facts.get(k) is not None, (
            "%s 没到手 —— 发出方读的键生产方不产出?facts=%s" % (k, facts))


def unicode_head(s: str, n: int = 24) -> str:
    """一段话的开头(断言用):念出来的必须是 aggregate 写的**那一段**。"""
    return str(s or "")[:n]


def test_a_verified_success_does_not_say_it_twice(store):
    """拿到了并且验过了 ⇒ 结论只说一遍。

    ``_verify`` 那一支已经发过 ``atomic_resolution_achieved``(带三联图),
    ``aggregate`` 不许再发一条 ``atomic_run_result`` —— 同一条结论说两遍
    会把真正有信息量的那几条淹掉。
    """
    rows, out, _seen = _drive(store, scripted={
        "r1": _found(), "verify": {"verdict": "atomic_resolved",
                                   "angular_concentration": 293.4}})
    kinds = _kinds(rows)
    assert "atomic_resolution_achieved" in kinds, kinds
    assert "atomic_run_result" not in kinds, (
        "同一条结论说了两遍:%s" % kinds)
    assert out["achieved"] is True


def test_a_hit_that_fails_verification_is_not_narrated_as_a_win(store):
    """初判过了但验证没过 ⇒ 旁白不许说成拿到了。"""
    rows, _out, _seen = _drive(store, scripted={
        "r1": _found(), "verify": {"verdict": "undecidable",
                                   "angular_concentration": 12.0}})
    text = _texts(rows)
    assert "判不了" in text, text
    assert "拿到原子分辨" not in text, text


def test_a_stop_for_no_next_site_narrates_the_maps_reason(store, monkeypatch):
    """地图说没有下一站 ⇒ 要把**地图给的理由**念出来,并说清为什么不接着修针。"""
    monkeypatch.setattr(A, "_coarse_suggestion",
                        lambda: (None, "这一带已经用完,建议换样品"))
    rows, _out, _seen = _drive(store, scripted={
        "r1": _absent(), "surface": {"verdict": "no_usable_region"}})
    text = _texts(rows)
    assert "这一带已经用完,建议换样品" in text, "地图给的理由没念出来:%s" % text
    assert "换样品" in text
    assert "不接着修针" in text or "没有接着去修针" in text, text


def test_the_tally_separates_ladder_rungs_from_triage_and_relocation(store,
                                                                    monkeypatch):
    """「代价花在哪」这句话里的两个数必须分开报。

    ``rungs_used`` 是一份记账清单:除了阶梯上的档,还有分诊和粗动换区
    (换区一次动作还记两条:moving + moved)。统统念成「档」时屏幕上会出现
    「走过 12 档」—— 而这部阶梯**只有四档**。用户看到那句话的第一个问题
    必然是「哪来的 12 档」,而它本来是要回答「代价花在哪」的。
    """
    monkeypatch.setattr(A, "_coarse_suggestion",
                        lambda: ({"axis": "x", "direction": "+", "steps": 300,
                                  "reason": "往 +x 还有没去过的地方"}, "r"))
    rows, out, _seen = _drive(store, scripted={
        "r1": _absent(), "surface": {"verdict": "no_usable_region"},
        "r1#2": _absent(), "surface#2": {"verdict": "ok"},
        "r2#2": {"outcome": "no_tip"},
        "r3a#2:recheck": _absent(), "r3b#2:recheck": _absent()})
    final = [r["text"] for r in rows
             if json.loads(r["meta"])["nk"] == "atomic_run_result"][0]
    assert "阶梯上开过" in final, final
    assert "一共走过" in final, final
    # 阶梯上真的开过的是 r1 / r2 / r3a / r3b —— surface / relocate 不是档
    assert "开过 4 档" in final, final
    assert "surface" in final, "清单本身还是要摆出来(它是审计用的)"


def test_a_rung_that_was_never_opened_is_not_billed(store):
    """**没开过的档不许进「代价花在哪」。**

    ``allow_forge`` 默认关 ⇒ ``_note(T("r4"), "not_allowed")`` 记了一条
    ``rung="r4"``,而那一档**一次都没跑**。以前 ``rungs_used`` 只滤掉了
    「预算用完」那一种,于是屏幕上出现「开过 5 档」而其中一档从未打开 ——
    一句把没花的代价算进账里的话。
    """
    _rows, out, _seen = _drive(store, scripted={
        "r1": _absent(), "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    assert any(r.get("outcome") == "not_allowed" for r in out["rungs"]), (
        "这一跑没走到 forge 那一支,这条测试测不到它要测的东西")
    assert "r4" not in out["rungs_used"], out["rungs_used"]
    assert set(out["rungs_used"]) >= {"r1", "r2", "r3a", "r3b"}, out["rungs_used"]


# ── ④ 结构闸门:加一档而忘了措辞,当场红 ─────────────────────────────────


def _emitter_notes() -> list:
    """源码里每一处 ``_note(rung, outcome, …)`` 的 ``(rung 词根, outcome)``。

    走 AST 不走字符串:第一个实参多半是 ``T("r1")``(带站点后缀的那个包装),
    按字符串找会漏掉全部带后缀的档位。
    """
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    out: list = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "_note"
                and len(node.args) >= 2):
            continue
        rung_node = node.args[0]
        if isinstance(rung_node, ast.Constant):
            rung = str(rung_node.value)
        elif (isinstance(rung_node, ast.Call)
              and getattr(rung_node.func, "id", "") == "T"
              and rung_node.args and isinstance(rung_node.args[0], ast.Constant)):
            rung = str(rung_node.args[0].value)
        else:  # pragma: no cover — 写法变了,下面那条自检会先红
            continue
        if not isinstance(node.args[1], ast.Constant):  # pragma: no cover
            continue
        out.append((rung.split(":")[0], str(node.args[1].value)))
    return out


def test_the_gate_can_actually_see_the_emitters_notes():
    """闸门自检 —— 一个扫不到任何东西的闸门永远是绿的。

    本仓刚为这句话付过学费(``_check_spec_preimport.py`` 引用了早就删掉的模块,
    烂了很久没人发现,正因为没有任何东西跑它)。
    """
    notes = _emitter_notes()
    assert len(notes) >= 10, "只扫到 %d 处 _note —— 写法变了,这道闸门失效了" % len(notes)
    assert ("r1", "no_lattice") in notes, notes


def test_every_outcome_the_emitter_can_note_has_a_phrase():
    """``_note`` 能写出的每一个结局码,措辞表里都要有一条。

    漏一条的后果不是报错,是那一档的旁白退回 ``结局「xxx」`` —— 一句语法完全
    正确、看起来只是有点生硬的话。这正是「加了一档忘了旁白」的样子,
    而它只能靠结构挡。
    """
    missing = sorted({o for _r, o in _emitter_notes()
                      if o not in T._ATOMIC_OUTCOME_ZH})
    assert not missing, (
        "这些结局码没有措辞:%s —— 加一档时两处一起改" % missing)


def test_every_phrase_is_reachable_from_the_emitter():
    """反证:措辞表里不许有**发不出来**的死条目。

    一条永远不会被用到的翻译,等于一句没有证据支持的「已经支持了」——
    与 ``FORGE_OUTCOME_ZH`` 里那条「``verified`` 不在这里」是同一条纪律。
    """
    emitted = {o for _r, o in _emitter_notes()}
    dead = sorted(set(T._ATOMIC_OUTCOME_ZH) - emitted)
    assert not dead, "措辞表里这几条发不出来:%s" % dead


def test_every_rung_name_the_emitter_uses_has_a_chinese_name():
    """档位名同理:漏一个就会在屏幕上显示成一个英文 slug。"""
    missing = sorted({r for r, _o in _emitter_notes()
                      if r.split("#")[0] not in T._ATOMIC_RUNG_ZH})
    assert not missing, "这些档位名没有中文:%s" % missing


def test_the_ladder_narration_is_not_wired_into_result_kind_for_skill():
    """三条新旁白**不进** ``RESULT_KIND_FOR_SKILL`` / ``BEGIN_KIND_FOR_SKILL``。

    那两张表驱动的是 ``GraphExecutor`` 的通用发点,它发的是 ``result=data``
    (即 ``result.*`` 路径),而这三条模板是**根级**路径 —— 接上去等于 100%
    走兜底,正是 ``poke_indent`` 那个坑。而且本技能已经在自己的编排里发过了,
    接上去就是同一件事说两遍。
    """
    for kind in ("atomic_begin", "atomic_rung", "atomic_run_result"):
        assert kind not in T.RESULT_KIND_FOR_SKILL.values(), kind
        assert kind not in T.BEGIN_KIND_FOR_SKILL.values(), kind
    assert "AchieveAtomicResolution" not in T.RESULT_KIND_FOR_SKILL
    assert "AchieveAtomicResolution" not in T.BEGIN_KIND_FOR_SKILL


# ── ⑤ 旁白绝不许弄坏实验 ────────────────────────────────────────────────


def test_a_broken_narration_channel_never_breaks_the_run(store, monkeypatch):
    """旁白链路整条炸掉时,编排必须照跑、结论必须照出。

    这条和 ``narrate()`` 自己的「永不抛」是**两件事**:那边保的是 narrate 内部,
    这边保的是发出方那几行 —— 画图、取路径、拼 payload 都在实验线程上。
    """
    def _boom(*_a, **_k):
        raise RuntimeError("旁白炸了")

    monkeypatch.setattr("mast.chat.narration.narrate", _boom)
    monkeypatch.setattr(A, "_render_report", _boom)
    _rows, out, seen = _drive(store, scripted={
        "r1": _absent(), "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    assert "r3b:recheck" in seen, "旁白炸了把编排也带走了:%s" % seen
    assert out.get("advice"), "旁白炸了把结论也带走了"


def test_a_missing_frame_still_narrates_the_result(store):
    """图画不出来时**话照说** —— 配图绝不许把旁白带走。"""
    rows, _out, _seen = _drive(store, scripted={
        "r1": _absent(), "r2": {"outcome": "no_tip"},
        "r3a:recheck": _absent(), "r3b:recheck": _absent()})
    final = [r for r in rows if json.loads(r["meta"])["nk"] == "atomic_run_result"]
    assert final and "没有拿到" in final[0]["text"]
    assert not (json.loads(final[0]["meta"]).get("image") or {}), "凭空挂了一张图"
