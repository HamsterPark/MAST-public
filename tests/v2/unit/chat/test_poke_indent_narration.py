# -*- coding: utf-8 -*-
"""整形检测应进入旁白，且明确区分检测不足与未接触。

insufficient_data 需要改善采集条件，不能被解释为应加深下压；
测试覆盖每种检测状态与其对应说明。"""
from __future__ import annotations

import pytest

from mast.chat import narration_templates as NT


def _render(**kw):
    t = NT.TEMPLATES["poke_indent"]
    return t.render(dict(kw))


def test_the_template_is_registered():
    assert "poke_indent" in NT.TEMPLATES
    t = NT.TEMPLATES["poke_indent"]
    # verdict 是字符串，进得了 requires；缺了要走 fallback 并标 degraded
    assert "verdict" in t.requires


def test_cluster_says_it_landed_and_gives_the_number():
    s = _render(verdict="cluster", dz_pm=240.0, tol_pm=20.0,
                feedback_segment_source="current", feedback_segment_s=1.20)
    assert "已接触" in s
    assert "+240" in s
    assert "1.20" in s, "要说清楚读的是反馈恢复之后那一段"


def test_no_change_is_reported_plainly():
    """术语是「未接触」，不是「没扎上」 —— 闸门 COLLOQUIAL 表里就有这一条。"""
    s = _render(verdict="no_change", dz_pm=0.2, tol_pm=20.0,
                feedback_segment_source="current", feedback_segment_s=1.40)
    assert "未接触" in s
    assert "判不了" not in s


def test_insufficient_data_never_advises_a_deeper_poke():
    """**这是这条模板存在的核心理由。**

    要推翻它：拿出一种情形，其中「第四段没录到」的正确下一步确实是加深。
    """
    s = _render(verdict="insufficient_data", dz_pm=0.0, tol_pm=20.0,
                feedback_segment_source="no_return", feedback_segment_s=None)
    assert "判不了" in s
    assert "不是" in s and "未接触" in s, "必须明说它不是「未接触」"
    assert "post_roll_s" in s, "要指出真正该改的那个参数"
    assert "别据此增大下压深度" in s
    assert not any(c.isdigit() for c in s.split("post_roll_s")[-1]), (
        "劝告里不许写死秒数 —— 真源是 _POKE_POST_ROLL_S")


def test_insufficient_data_says_which_kind_of_failure():
    """``no_return`` 与 ``too_short`` 是两种不同的没录到 —— 话要不一样。"""
    a = _render(verdict="insufficient_data", feedback_segment_source="no_return")
    b = _render(verdict="insufficient_data", feedback_segment_source="too_short",
                feedback_segment_s=0.1)
    assert "再没回到 setpoint" in a
    assert a != b


def test_no_press_is_carried_into_the_sentence():
    """电流没升上去 ⇒ 可能根本没压到表面。这**不是**判不了，但要说出来。"""
    s = _render(verdict="no_change", dz_pm=1.0, tol_pm=20.0,
                feedback_segment_source="no_press")
    assert "未接触" in s
    assert "没压到表面" in s


def test_a_pit_or_tip_change_is_distinguished_from_no_change():
    s = _render(verdict="tip_changed_or_pit", dz_pm=-150.0, tol_pm=20.0,
                feedback_segment_source="current", feedback_segment_s=1.0)
    assert "针尖" in s or "坑" in s
    assert "未接触" not in s


def test_an_unknown_verdict_does_not_invent_one():
    s = _render(verdict="something_new", feedback_segment_source="current")
    assert "something_new" in s, "不认识的判定要如实带出来，不许编一个"


def test_tone_flags_the_two_ends():
    t = NT.TEMPLATES["poke_indent"]
    assert t.tone_of({"verdict": "cluster"}) == "good"
    assert t.tone_of({"verdict": "insufficient_data"}) == "warn"
    assert t.tone_of({"verdict": "no_change"}) == "info"


# ── 配图 ──────────────────────────────────────────────────────────────────

def test_the_panel_renders_a_four_segment_trace(tmp_path):
    """Z 曲线图画得出来，而且四段各归各位。"""
    from mast.io import z_trace
    from mast.vision.poke_trace_panel import render_poke_trace_panel

    z, t, c, ct = [], [], [], []
    now = 0.0
    for zz, ii, dur in ((0.0, 200e-12, 0.3), (-300e-12, 10e-9, 0.6),
                        (0.0, 10e-9, 0.3), (240e-12, 200e-12, 1.0)):
        for _ in range(int(dur / 0.005)):
            z.append(zz); t.append(now); c.append(ii); ct.append(now)
            now += 0.005
    ind = z_trace.step_verdict(z, t, 0.3, post_roll_s=0.2, tol_k=4.0,
                               tol_abs_m=0.02e-9, current_s=c, current_t=ct)
    png = render_poke_trace_panel(z, t, c, ct, event_t=0.3, indent=ind,
                                  meta={"tip_lift_m": -300e-12},
                                  label="unit", out_dir=tmp_path)
    assert png, "四段齐全的曲线应该画得出来"
    assert (tmp_path / png.split("\\")[-1].split("/")[-1]).exists()


def test_the_panel_never_raises_on_garbage(tmp_path):
    """一张配图坏掉绝不许弄坏实验 —— 任何输入都只返回 None。"""
    from mast.vision.poke_trace_panel import render_poke_trace_panel

    assert render_poke_trace_panel([], [], [], [], event_t=0.0, indent={},
                                   out_dir=tmp_path) is None
    assert render_poke_trace_panel([1.0, 2.0], [0.0], None, None, event_t=0.0,
                                   indent={}, out_dir=tmp_path) is None


def test_the_panel_is_deterministic_for_the_same_trace(tmp_path):
    """同一条曲线重画不该多落一个文件（内容指纹用 hashlib，不用内置 hash）。"""
    from mast.vision.poke_trace_panel import render_poke_trace_panel

    z = [0.0] * 40 + [-3e-10] * 40 + [0.0] * 20 + [2.7e-10] * 60
    t = [i * 0.01 for i in range(len(z))]
    c = [2e-10] * 40 + [1e-8] * 60 + [2e-10] * 60
    ct = list(t)
    ind = {"verdict": "cluster", "delta_m": 2.7e-10,
           "feedback_segment_source": "current", "feedback_restored_t": 1.0}
    a = render_poke_trace_panel(z, t, c, ct, event_t=0.4, indent=ind,
                                meta={"tip_lift_m": -3e-10}, label="x",
                                out_dir=tmp_path)
    b = render_poke_trace_panel(z, t, c, ct, event_t=0.4, indent=ind,
                                meta={"tip_lift_m": -3e-10}, label="x",
                                out_dir=tmp_path)
    assert a == b
    assert len(list(tmp_path.glob("*.png"))) == 1


# 从发出方验证旁白载荷层次。
# 模板可能读取根级键，也可能读取 result.*；两种约定必须与实际发出的载荷一致。
# 仅调用 Template.render 的替身无法证明这一接线。

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
from unittest.mock import MagicMock as _MagicMock  # noqa: E402

_sys.modules.setdefault("nanonis_spm", _MagicMock())

from mast.chat import narration as _narration  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core.turn_context import turn_scope  # noqa: E402

_CID = "conv-poke-indent-wiring"


@pytest.fixture()
def narration_store(tmp_path):
    """真的 ``ConversationStore``,建在 tmp 上。

    **不用替身**:替身会说真组件不会说的话,而这一整条 bug 的形状正是
    「两边各自都对,放在一起对不上」。同理由见 ``test_narration_noop.py``。
    ``set_store`` 必须传 —— 本仓「测试污染真实用户数据」已经发生五次,
    而旁白写的正是用户真实会话所在的那个库。
    """
    _narration.reset_for_tests()
    st = ConversationStore(tmp_path / "chat" / "mast_conversations.db")
    _narration.set_store(st)
    yield st
    _narration.flush(2.0)
    _narration.reset_for_tests()


def _emit_and_read(narration_store, ind: dict, depth_m: float = -5e-10) -> dict:
    """跑真的发出方,把落库的那一行读回来。"""
    from mast.skills.composite import _tip_phases as P

    with turn_scope(conversation_id=_CID, run_id="run-1"):
        P._narrate_indent({"channels": {}}, ind, depth_m)
    assert _narration.flush(3.0), "写线程没排空"
    rows = [r for r in narration_store.messages_since(_CID, 0)
            if r["kind"] == _narration.NARRATION_KIND]
    assert len(rows) == 1, f"应当正好落一行,实际 {len(rows)} 行"
    return rows[0]


def test_the_emitted_payload_actually_renders_the_verdict(narration_store):
    """**这一条就是那个 bug。** 发出方发出来的那份数据必须渲染得出判定。"""
    row = _emit_and_read(narration_store, {
        "verdict": "cluster", "delta_m": 240e-12, "tol_m": 20e-12,
        "feedback_segment_source": "current", "feedback_segment_s": 1.20})
    text = row["text"]
    assert text != NT.TEMPLATES["poke_indent"].fallback, (
        f"发出方发的形状渲染不出判定 —— 说的还是 fallback 那句:{text}")
    assert "已接触" in text, text
    assert "+240" in text, f"Δz 没进句子:{text}"
    assert "1.20" in text, f"读的是哪一段没进句子:{text}"


def test_the_row_records_the_numbers_for_reconciliation(narration_store):
    """meta.facts 必须记录旁白使用的输入字段，防止载荷路径不匹配时只留下空事实集。"""
    import json

    row = _emit_and_read(narration_store, {
        "verdict": "no_change", "delta_m": 2e-13, "tol_m": 20e-12,
        "feedback_segment_source": "current", "feedback_segment_s": 1.40})
    meta = json.loads(row["meta"])
    assert meta.get("degraded") is not True, "标着 degraded —— 必需字段没读到"
    facts = meta.get("facts") or {}
    assert facts.get("verdict") == "no_change", facts
    assert "dz_pm" in facts and "depth_pm" in facts, facts


@pytest.mark.parametrize("verdict", ["cluster", "no_change",
                                     "tip_changed_or_pit", "insufficient_data"])
def test_every_verdict_survives_the_wire(narration_store, verdict):
    """四态**逐个**走一遍真发出方 —— 一态漏掉就是一种处境永远看不见。"""
    row = _emit_and_read(narration_store, {
        "verdict": verdict, "delta_m": 1e-12, "tol_m": 20e-12,
        "feedback_segment_source": "no_return"})
    assert row["text"] != NT.TEMPLATES["poke_indent"].fallback, row["text"]


def test_the_template_paths_and_the_emitter_agree_on_nesting():
    """模板声明的路径与发出方发的键**必须同层**。

    这是上面那几条的静态孪生:它不需要跑起来,所以换一种发法(比如有人又给
    包一层)时它也会红。射程是**这一条模板**,不是全仓 —— 全仓那道闸门是
    另一件事,见报告。
    """
    import ast

    from mast.skills.composite import _tip_phases as P

    # 走 AST,**不走字符串切分**:源码里解释这个 bug 的注释本身就写着
    # ``narrate("poke_indent", result={...})``,按字符串找会先撞上那段注释
    # ——「报错内容就是数据本身」的又一次(第一版就是这么写的,当场红了)。
    tree = ast.parse(_Path(P.__file__).read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "narrate"
        and n.args and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "poke_indent"
    ]
    assert len(calls) == 1, f"poke_indent 的发出方有 {len(calls)} 处,应当只有一处"
    sent = {kw.arg for kw in calls[0].keywords if kw.arg}
    tpl = NT.TEMPLATES["poke_indent"]
    for path in tpl.requires + tpl.records:
        assert "." not in path, (
            f"模板路径 {path} 带了层级,而发出方是平铺发的 —— 两边对不上")
        assert path in sent, (
            f"模板要 {path},而发出方发的键是 {sorted(sent)}")
    assert "result" not in sent, (
        "发出方又把载荷包进 result= 了 —— 模板读的是根级路径,"
        "这正是 30/30 走 fallback 的那个形状")
