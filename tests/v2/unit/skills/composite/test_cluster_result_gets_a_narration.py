"""扎完针,团簇的判读要有一条**带图**的旁白 —— 而且三态如实。

## 出处

要求:旁白要更具体,扎针时团簇的分析图与分析结果也要能返回一条旁白。

## 为什么到今天才做

``RESULT_KIND_FOR_SKILL`` 从 2026-08-11 起**故意是空的**,理由逐字:

    结论类旁白(圆不圆 / 一致不一致 / 扎出团簇没有)要读子技能的 result.data,
    而那些字段的真源文件当时正在被另一条线改。照着一份会变的词汇表写模板,
    写出来的是**一句永远走 fallback、看起来却完全正常**的话。
    ……一个空 dict 是诚实的,一条读错字段的模板不是。

08-15/16 那两天做的正是把 ``AssessClusterRoundness`` 的返回字段钉死
(单测 + 产物声明)。**前提到位了,阶段 4 才开工。**

## 这个文件盯三件事

1. 旁白真的发出来了,而且带的是**判读用的那张图**;
2. **三态如实** —— 「判不了」不许被写成「不圆」;
3. **没钉死词汇表的技能仍然不发**(那不是遗漏,是同一条纪律)。
"""
from __future__ import annotations

import pytest

#: 判据看过的那一帧,与画出来挂上去的那一张。两者**不是同一个文件**:
#: 送进判据的是 .sxm,挂到旁白上的是当场画好落盘的 PNG。
JUDGED_SXM = r"D:\data\cluster_0298.sxm"
PANEL_PNG = r"D:\artifacts\cluster_panels\cluster_0298_dead.png"


@pytest.fixture
def captured(monkeypatch):
    """截住旁白出口,拿到 (kind, image, data)。"""
    import mast.chat.narration as N

    out: list = []
    monkeypatch.setattr(
        N, "_emit",
        lambda cid, kind, image, data, event_t: out.append((kind, image, data)),
        raising=True)
    return out


def _fire(data: dict, *, skill: str = "AssessClusterRoundness",
          scan_path: str = r"D:\x\_0298.sxm"):
    from mast.core.types import SkillResult
    from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor

    ex = object.__new__(GraphExecutor)
    ex._composite_name = "ForgeAuTip"
    step = CompositeStep(step_id="D:p1:roundness", skill_name=skill,
                         params={"scan_path": scan_path, "select": "center"})
    GraphExecutor._narrate_step_result(
        ex, step, SkillResult(skill_name=skill, success=True, data=data))


def _render(entry) -> str:
    from mast.chat.narration_templates import TEMPLATES

    kind, _img, data = entry
    return TEMPLATES[kind].render(data)


def test_a_round_cluster_says_so_with_the_number(captured):
    _fire({"is_round": True, "equivalent_axis_ratio": 0.856,
           "area_px": 1445, "n_components": 1, "multi_tip": False})
    from mast.chat.narration_templates import (CLUSTER_MULTI_TIP_UNDECIDABLE,
                                               CLUSTER_MULTI_TIP_YES,
                                               CLUSTER_ROUND_HEAD,
                                               CLUSTER_UNDECIDABLE_HEAD)

    assert len(captured) == 1
    s = _render(captured[0])
    assert CLUSTER_ROUND_HEAD in s and "0.856" in s
    assert CLUSTER_UNDECIDABLE_HEAD not in s
    # multi_tip=False 是**仪器背书了「没有」**,不是「没查」⇒ 一个字都不用说。
    # 三态里只有 True 和 None 需要出声。
    assert CLUSTER_MULTI_TIP_YES not in s
    assert CLUSTER_MULTI_TIP_UNDECIDABLE not in s


def test_the_image_is_rendered_from_the_frame_that_was_judged(captured,
                                                              monkeypatch):
    """配图必须由**送进 skill.execute 的那一份** scan_path 画出来。

    借了别的图不是「没有图」,是**说着 A 配着 B** —— #76/#78 那个事故
    (历史里所有缩略图静默变成最新那张整图)就是这个形状。
    """
    seen: list = []

    def fake_render(src, result, **kw):
        seen.append((src, result))
        return PANEL_PNG

    monkeypatch.setattr("mast.vision.cluster_panel.render_cluster_panel",
                        fake_render, raising=True)
    _fire({"is_round": True, "equivalent_axis_ratio": 0.9}, scan_path=JUDGED_SXM)

    assert seen and seen[0][0] == JUDGED_SXM
    # 判据的返回 data 也一并送进去 —— 图上标的数必须**就是报文里那几个数**。
    # 重算会长出第二个真源,两处一旦不一致,看图的人无从知道该信哪个。
    assert seen[0][1]["equivalent_axis_ratio"] == 0.9

    _kind, image, _data = captured[0]
    assert image == {"src": PANEL_PNG, "origin": "milestone_png"}


def test_the_origin_is_one_the_endpoint_will_actually_serve(captured, monkeypatch):
    """``origin`` 必须落在端点的**白名单**里,否则那张图永远取不到。

    ── 这条测试是一次事故换来的(2026-08-16,同日)

    第一版挂的是 ``{"src": scan_path, "origin": "sxm"}``。它:

      · 通过了 ``narration._normalise_image``(那里两档都合法);
      · 存进了 DB,前端也拿到了 ``has_image=True``;
      · 然后 ``GET /api/chat/narration-image/{seq}`` 一律 **404** ——
        ``_ALLOWED_ORIGINS`` 只有 ``("milestone_png",)``。

    也就是**一条挂着图、图却永远取不到的旁白**。这比没有图更坏:发出方一切正常,
    存储一切正常,前端只是「加载不出来」,没有任何一处会说出真正的原因。

    ⇒ 这里**不写死** ``"milestone_png"`` 这个字面量,而是去问端点那张白名单。
    哪天端点改了名单,红的是这条测试,而不是三个月后某个人的截图。
    """
    monkeypatch.setattr("mast.vision.cluster_panel.render_cluster_panel",
                        lambda src, result, **kw: PANEL_PNG, raising=True)
    _fire({"is_round": True, "equivalent_axis_ratio": 0.9})

    from mast.api.routes.chat_narration import _ALLOWED_ORIGINS

    _kind, image, _data = captured[0]
    assert image is not None, "配图丢了"
    assert image["origin"] in _ALLOWED_ORIGINS, (
        f"origin={image['origin']!r} 不在端点白名单 {_ALLOWED_ORIGINS} 里 —— "
        "这张图发得出去、取不回来")


def test_a_broken_panel_costs_the_image_not_the_narration(captured, monkeypatch):
    """画图失败 ⇒ **旁白照发,只是没有图**。一张配图绝不许弄坏正在跑的实验。"""
    monkeypatch.setattr("mast.vision.cluster_panel.render_cluster_panel",
                        lambda src, result, **kw: None, raising=True)
    _fire({"is_round": True, "equivalent_axis_ratio": 0.9})

    assert len(captured) == 1, "旁白没发出来 —— 画图失败不该有这个代价"
    _kind, image, _data = captured[0]
    assert image is None
    assert "0.900" in _render(captured[0])


def test_an_undecidable_cluster_is_never_rendered_as_not_round(captured):
    """**三态如实。** 「判不了」写成「不圆」= 用户读到一个系统没下过的判决。

    团簇小于 20 px 时像素化本身就能让一个完美的圆读出 0.64–0.77,
    那个尺度上给判决比不给更糟 —— 旁白是用户唯一会读的那一层。
    """
    _fire({"is_round": None, "roundness_undecidable": "只有 4 像素,低于 20 像素下限",
           "area_px": 4, "n_components": 16, "multi_tip": None})
    from mast.chat.narration_templates import (CLUSTER_NOT_ROUND_HEAD,
                                               CLUSTER_UNDECIDABLE_HEAD)

    s = _render(captured[0])
    # 断言的是**关系**(第三态在、第二态不在),不是任何一个形容词。
    # 文案已经被改过一次(措辞不理想),还会再改;纪律不会。
    assert CLUSTER_UNDECIDABLE_HEAD in s
    assert CLUSTER_NOT_ROUND_HEAD not in s, f"把第三态写成了第二态:{s}"
    assert "4 像素" in s or "20" in s, "没说出为什么给不出读数"


def test_multi_tip_undecidable_is_not_reported_as_absent(captured):
    """``multi_tip`` 也是三态:None 要说「判不了」,不能默不作声。

    默不作声等于让用户以为「查过了,没有多针尖」。
    """
    from mast.chat.narration_templates import (CLUSTER_MULTI_TIP_UNDECIDABLE,
                                               CLUSTER_MULTI_TIP_YES)

    _fire({"is_round": False, "equivalent_axis_ratio": 0.5, "multi_tip": None})
    s = _render(captured[0])
    assert CLUSTER_MULTI_TIP_UNDECIDABLE in s
    assert CLUSTER_MULTI_TIP_YES not in s, f"量不出被说成了量到了:{s}"

    captured.clear()
    _fire({"is_round": False, "equivalent_axis_ratio": 0.5, "multi_tip": True})
    assert CLUSTER_MULTI_TIP_YES in _render(captured[0])

    # 第三档 False = 仪器背书了「没有」⇒ 不出声。三态各有各的说法。
    captured.clear()
    _fire({"is_round": False, "equivalent_axis_ratio": 0.5, "multi_tip": False})
    s3 = _render(captured[0])
    assert CLUSTER_MULTI_TIP_YES not in s3
    assert CLUSTER_MULTI_TIP_UNDECIDABLE not in s3


def test_a_skill_without_a_pinned_vocabulary_stays_silent(captured):
    """没接线的技能**一条都不发** —— 那不是遗漏,是同一条纪律。

    宁可不说,也不说一句读错字段的话:后者永远走 fallback,而且看起来完全正常。
    """
    _fire({"anything": 1}, skill="ScanAt")
    _fire({"anything": 1}, skill="AssessTipSharpness")
    assert captured == []


def test_every_wired_skill_has_its_fields_pinned_somewhere():
    """接线表里的每一条,**它读的字段都得有东西钉着**。

    原来这条写的是 ``== {"AssessClusterRoundness": ...}`` —— 一条「只许有一条」
    的断言。08-17 加第二、三条时它当场红了,而红的不是它要防的那件事:
    它要防的是**接一个字段还在变的技能**,不是「表里超过一条」。

    现在断言的是关系:表里每一个技能,在下面这份**登记**里都说得出「谁钉着它」。
    加第四条的人必须同时补一行登记 —— 补不出来,就说明那个技能的返回字段
    确实还没被钉住,而那正是这条测试存在的理由。
    """
    from mast.chat.narration_templates import RESULT_KIND_FOR_SKILL

    #: 技能 → 钉住它返回字段的那个测试。**不是文档,是可执行的对账。**
    PINNED_BY = {
        "AssessClusterRoundness":
            "tests/v2/unit/skills/builtins/test_cluster_roundness_criterion.py",
        "FindCleanSpot":
            "tests/v2/unit/chat/test_the_narration_says_where_the_tip_went.py"
            "::test_every_path_the_template_reads_exists_in_what_the_skill_returns",
        "BiasPulseWithReadback":
            "tests/v2/unit/chat/test_the_narration_says_where_the_tip_went.py"
            "::test_the_pulse_template_reads_the_key_the_skill_actually_writes",
        # AutoTilt 的返回字段被 _tip_phases.level_phase 逐个读进 out["tilt"]
        # (action / skipped / reason / z_span_before_m / z_span_after_m)——
        # 那就是它们被钉住的地方:改了那边,这条旁白立刻走 fallback,
        # 而 test_the_levelling_step_says_whether_it_happened 会红。
        "AutoTilt":
            "tests/v2/unit/chat/test_the_narration_says_where_the_tip_went.py"
            "::test_the_levelling_step_says_whether_it_happened",
    }
    unpinned = sorted(set(RESULT_KIND_FOR_SKILL) - set(PINNED_BY))
    assert not unpinned, (
        f"这些技能接了结论旁白,却没有人钉它们的返回字段:{unpinned} —— "
        "先写那条钉子,再接线(一条读错字段的模板会永远走 fallback 且看起来完全正常)")

    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[5]
    for skill, where in PINNED_BY.items():
        if skill not in RESULT_KIND_FOR_SKILL:
            continue          # 接线撤了,登记留着不算错
        f = root / where.split("::")[0]
        assert f.exists(), f"{skill} 的钉子指向一个不存在的文件:{where}"


def test_a_missing_verdict_falls_back_instead_of_inventing_one(captured):
    """``is_round`` 都没有 ⇒ 走 fallback,**不编一个判决**。"""
    from mast.chat.narration_templates import TEMPLATES

    tpl = TEMPLATES["cluster_roundness"]
    from mast.chat.narration_templates import (CLUSTER_NOT_ROUND_HEAD,
                                               CLUSTER_ROUND_HEAD)

    # ⚠️ ``is_round`` **不能**放进 requires(2026-08-17)。
    # ``_satisfied`` 为了防「``True`` 被念成打一发 1 V」而排除所有 bool,
    # 而 is_round 是真正的布尔判读 —— 放进去等于让这条模板永远走 fallback。
    # 真机上它就是这么坏了一整天:每一次都说「分析结果没保存」。
    # 「读没读到」现在由句子自己按**键在不在**判。
    assert "result.is_round" not in tpl.requires
    assert "result.roundness_undecidable" in tpl.records
    # fallback 里**一个判决都不准有** —— 连 is_round 都没读到的时候,
    # 说「形状规整」或「还不够规整」都是凭空发明一个结论。
    assert CLUSTER_ROUND_HEAD not in tpl.fallback
    assert CLUSTER_NOT_ROUND_HEAD not in tpl.fallback


def test_narration_never_breaks_the_run(captured, monkeypatch):
    """旁白链路出任何事,只能丢一条旁白 —— 绝不许弄坏正在跑的实验。"""
    import mast.chat.narration as N

    def boom(*a, **k):
        raise RuntimeError("narration backend down")

    monkeypatch.setattr(N, "_emit", boom, raising=True)
    _fire({"is_round": True, "equivalent_axis_ratio": 0.9})   # 不该抛
