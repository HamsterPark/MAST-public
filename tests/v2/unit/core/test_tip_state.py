"""tip_state —— 词表归一、skill 层事实、注入块渲染。

两条硬规则在这里钉住:

  * **认不出就返回 None,绝不猜**。把「镝钪合金」猜成 W,修针方案表就会按钨针
    给参数,而那是一根完全不同的针。
  * 注入块里每个人读单位的数字**同行必须带 SI**(见
    tests/v2/unit/agents/test_injection_si_units.py 的论证:模型拿到的每个技能
    参数都是 SI,漏一个指数就把针扎进样品)。
"""

from __future__ import annotations

import re

from mast.core import tip_state

_SI_FORM = re.compile(r"\d(?:\.\d+)?e[+-]?\d+", re.I)
_HUMAN_UNIT = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(nm(?:/s)?|pA|nA|µV|uV|mV|µm|um|mm)\b")


def _si_offenders(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines()
            if _HUMAN_UNIT.search(ln) and not _SI_FORM.search(ln)]


# ── 词表归一 ────────────────────────────────────────────────────────────────

def test_material_normalisation_accepts_chinese_and_alloy_spellings() -> None:
    assert tip_state.normalize_material("钨") == "W"
    assert tip_state.normalize_material("tungsten") == "W"
    assert tip_state.normalize_material("W") == "W"
    assert tip_state.normalize_material("铂铱") == "PtIr"
    assert tip_state.normalize_material("Pt80Ir20") == "PtIr"
    assert tip_state.normalize_material("Pt-Ir") == "PtIr"
    assert tip_state.normalize_material("platinum-iridium") == "PtIr"
    assert tip_state.normalize_material("铁") == "Fe"
    assert tip_state.normalize_material("  fe  ") == "Fe"


def test_unknown_material_returns_none_rather_than_guessing() -> None:
    assert tip_state.normalize_material("镝钪合金") is None
    assert tip_state.normalize_material("") is None
    assert tip_state.normalize_material(None) is None
    assert tip_state.normalize_material(123) is None


def test_fabrication_normalisation_covers_the_four_the_operator_named() -> None:
    assert tip_state.normalize_fabrication("电化学腐蚀") == "etched"
    assert tip_state.normalize_fabrication("electrochemical etching") == "etched"
    assert tip_state.normalize_fabrication("钳子剪") == "cut"
    assert tip_state.normalize_fabrication("clipped") == "cut"
    assert tip_state.normalize_fabrication("打磨") == "ground"
    assert tip_state.normalize_fabrication("FIB") == "fib"
    assert tip_state.normalize_fabrication("聚焦离子束") == "fib"
    assert tip_state.normalize_fabrication("凭空捏造") is None


def test_form_normalisation() -> None:
    assert tip_state.normalize_form("qPlus") == "qplus"
    assert tip_state.normalize_form("q-plus") == "qplus"
    assert tip_state.normalize_form("音叉") == "qplus"
    assert tip_state.normalize_form("tuning fork") == "qplus"
    assert tip_state.normalize_form("普通") == "stm_wire"
    assert tip_state.normalize_form("stm_wire") == "stm_wire"


def test_candidate_lists_are_offered_for_a_miss() -> None:
    """词表 miss 时模型要拿得到可选项,才能问用户而不是重试同一个词。"""
    mats = tip_state.material_candidates()
    assert any(c.startswith("W（") for c in mats)
    assert any("PtIr" in c for c in mats)
    assert len(tip_state.fabrication_candidates()) == len(tip_state.TIP_FABRICATIONS)
    assert len(tip_state.form_candidates()) == len(tip_state.TIP_FORMS)


def test_auto_name_is_human_recognisable() -> None:
    assert tip_state.auto_name("W", "etched", 3) == "W-etched #3"
    assert tip_state.auto_name("PtIr", "unknown", 1) == "PtIr #1"
    assert tip_state.auto_name("W", "etched", None) == "W-etched"


# ── holder / facts ──────────────────────────────────────────────────────────

def test_facts_are_none_when_no_tip_is_registered() -> None:
    tip_state.set_current_tip(None)
    assert tip_state.current_tip_facts() is None
    assert tip_state.get_current_tip() is None
    assert tip_state.is_qplus() is False


def test_facts_expose_what_the_skill_layer_needs() -> None:
    tip_state.set_current_tip({
        "id": "t1", "name": "W-etched #1", "material": "W",
        "fabrication": "etched", "form": "stm_wire", "wire_diameter_mm": 0.25,
        "installed_at": "2026-07-28T10:00:00", "qplus_q": None,
    })
    f = tip_state.current_tip_facts()
    assert f["material"] == "W" and f["fabrication"] == "etched"
    assert f["form"] == "stm_wire" and f["wire_diameter_mm"] == 0.25
    assert tip_state.is_qplus() is False
    tip_state.set_current_tip(None)


def test_holder_returns_a_copy_not_the_live_dict() -> None:
    tip_state.set_current_tip({"id": "t1", "material": "W"})
    got = tip_state.get_current_tip()
    got["material"] = "PtIr"
    assert tip_state.get_current_tip()["material"] == "W"
    tip_state.set_current_tip(None)


# ── 注入块 ──────────────────────────────────────────────────────────────────

def test_block_is_never_empty_even_with_nothing_registered() -> None:
    """「针尖未登记」「偏压极性未声明」本身就是模型需要知道的事实。"""
    block = tip_state.format_tip_block(None, None)
    assert block.strip()
    assert "未登记" in block
    assert "未声明" in block


def test_block_states_bias_polarity_meaning_for_sample_bias() -> None:
    block = tip_state.format_tip_block(None, {"bias_applied_to": "sample"})
    assert "样品" in block and "空态" in block and "占据态" in block


def test_tip_bias_polarity_says_the_sign_is_reversed() -> None:
    """加在针尖上时符号整体反号 —— 不说这句,模型会照搬文献的样品偏压约定。"""
    block = tip_state.format_tip_block(None, {"bias_applied_to": "tip"})
    assert "反号" in block


def test_unknown_bias_polarity_tells_the_model_not_to_assert() -> None:
    block = tip_state.format_tip_block(None, {})
    assert "未声明" in block and "不要断言" in block


def test_block_renders_the_current_tip(tmp_path=None) -> None:
    tip = {
        "name": "W-etched #2", "material": "W", "fabrication": "etched",
        "form": "stm_wire", "wire_diameter_mm": 0.25,
        "installed_at": "2026-07-28T10:00:00",
    }
    block = tip_state.format_tip_block(tip, {"bias_applied_to": "sample"})
    assert "W-etched #2" in block
    assert "钨" in block and "电化学腐蚀" in block
    assert "2026-07-28" in block


def test_wire_diameter_carries_si_alongside_mm() -> None:
    tip = {"name": "t", "material": "W", "fabrication": "etched",
           "form": "stm_wire", "wire_diameter_mm": 0.25}
    block = tip_state.format_tip_block(tip, {})
    assert "0.25 mm" in block
    assert "2.500e-04" in block or "2.5e-04" in block


def test_every_human_unit_in_the_block_has_an_si_partner() -> None:
    tip = {"name": "q1", "material": "PtIr", "fabrication": "cut",
           "form": "qplus", "wire_diameter_mm": 0.1,
           "qplus_sensor_model": "TF-32k", "qplus_f0_hz": 32768.0,
           "qplus_q": 30000.0, "qplus_k_n_per_m": 1800.0}
    profile = {"bias_applied_to": "sample", "preamp_model": "FEMTO DLPCA-200",
               "preamp_gain_v_per_a": 1e9,
               "qplus_f0_measured_hz": 32701.0, "qplus_q_measured": 21000.0}
    block = tip_state.format_tip_block(tip, profile)
    offenders = _si_offenders(block)
    assert not offenders, (
        "注入块给了人读单位却没在同一行给 SI 值:\n  " + "\n  ".join(offenders))


#: qPlus 那一段里必须出现的字样(用来判「这一段在不在」)。
_QPLUS_NOTE_MARK = "扎针前偏压会自动缓降到 20 mV"


def test_the_qplus_note_says_poking_is_routine_not_forbidden() -> None:
    """这一段**每一轮都注入进模型的上下文**,所以它必须是真的。

    ── 这条测试翻过面(2026-08-17)

    原来它叫 ``test_qplus_tip_gets_the_poke_warning``,钉的是
    「戳表面类处理会毁音叉,而且不可逆」+ ``allow_on_qplus`` 必须出现。

    **那个前提是错的**:nm 尺度下 qPlus tip 完全可以扎,这个误解从源头污染了
    整套系统的判据;而且默认值早在 08-16 就翻了 —— 旧描述连现状都不对。

    真实伤害不是「多一句废话」:模型每轮读到「有毁掉传感器的风险 + 默认拒绝」,
    于是该动手时回来问「要不要扎针」——一个过度谨慎、畏首畏尾的判据。

    ⇒ 现在钉两件事:**说了真正要注意的**(降偏压 + 深度包络),
      而且**不许再出现那些让模型犹豫的措辞**。
    """
    tip = {"name": "q1", "material": "PtIr", "fabrication": "cut", "form": "qplus"}
    block = tip_state.format_tip_block(tip, {})
    assert "qPlus" in block
    assert _QPLUS_NOTE_MARK in block, "没说「扎针前会自动降偏压」—— 那才是真正护针的那一条"
    assert "包络" in block, "没说深度包络 —— 那是「2 nm 以内」真正的执行者"

    # 这些措辞会让模型把一个该自己做的决定拿回来问。
    # **只查 qPlus 那一段** —— 「显式」之类的词在别的段落里另有正当用法
    # (「你显式给的数会被采纳」),整块搜会误伤,而一条误伤的断言下一步就是被删掉。
    para = next(ln for ln in block.splitlines() if _QPLUS_NOTE_MARK in ln)
    for banned in ("默认拒绝", "不可逆", "allow_on_qplus", "有毁掉音叉", "风险"):
        assert banned not in para, (
            f"qPlus 那一段里又出现了「{banned}」—— 这正是让 agent 瞻前顾后的那类话")


def test_stm_wire_tip_has_no_qplus_note() -> None:
    """qPlus 那一段只对 qPlus 出现 —— 别拿别的针尖的注意事项去污染判断。"""
    tip = {"name": "w1", "material": "W", "fabrication": "etched", "form": "stm_wire"}
    block = tip_state.format_tip_block(tip, {})
    assert _QPLUS_NOTE_MARK not in block


def test_block_tells_the_model_not_to_invent_conditioning_parameters() -> None:
    """既定的那条:LLM 不懂 STM,别让它自己想参数。"""
    block = tip_state.format_tip_block(None, {})
    assert "不要自己发明" in block or "留空" in block


def test_preamp_line_states_the_gain_relation() -> None:
    block = tip_state.format_tip_block(
        None, {"preamp_model": "FEMTO DLPCA-200", "preamp_gain_v_per_a": 1e9})
    assert "DLPCA-200" in block
    assert "1e+09 V/A" in block or "1e9 V/A" in block or "1e+09" in block
    assert "读数电压" in block


def test_missing_preamp_is_stated_as_missing() -> None:
    block = tip_state.format_tip_block(None, {})
    assert "前置放大器：未登记" in block


def test_service_days_are_counted_from_installed_at() -> None:
    import time as _time
    tip = {"name": "t", "material": "W", "fabrication": "etched",
           "form": "stm_wire", "installed_at": "2026-07-28T00:00:00"}
    now = _time.mktime((2026, 7, 31, 0, 0, 0, 0, 0, -1))
    block = tip_state.format_tip_block(tip, {}, now=now)
    assert "已服役 3 天" in block


def test_render_survives_garbage_fields() -> None:
    """渲染绝不能因为一个脏字段就抛 —— 它在每次 model call 的路径上。"""
    tip = {"name": None, "material": 42, "fabrication": [], "form": "qplus",
           "wire_diameter_mm": "粗", "qplus_f0_hz": float("nan"),
           "qplus_q": "多", "installed_at": "不是日期"}
    block = tip_state.format_tip_block(tip, {"preamp_gain_v_per_a": "很大"})
    assert block.strip()
