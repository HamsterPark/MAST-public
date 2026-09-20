"""修针方案表与解析链。

2026-07-31 定案：「要有完整参数方案表。（当前水平的）LLM 可不懂 STM 实验，
让他自己想参数就麻烦了。」这里钉住那句话变成代码后的三条性质:

  1. **没给参数就查表**,而且查出来的值随针尖变(钨腐蚀针 ≠ 铂铱剪切针 ≠ qPlus);
  2. **每个数字的来源可查**(explicit / 覆写 / 方案表 / 通用默认);
  3. **超安全包络是拒绝,不是夹紧** —— 与 scan_resolver 的 clamp 分属两层:
     那里是参数卫生,这里是不可逆硬件损伤。
"""

from __future__ import annotations

from mast.core.tip_conditioning_policy import resolve_policy
from mast.core.tip_conditioning_resolver import (
    SOURCE_EXPLICIT,
    SOURCE_OVERRIDE,
    SOURCE_POLICY,
    resolve_conditioning,
)

_W_ETCHED = {"name": "W-etched #1", "material": "W",
             "fabrication": "etched", "form": "stm_wire"}
_PTIR_CUT = {"name": "PtIr-cut #2", "material": "PtIr",
             "fabrication": "cut", "form": "stm_wire"}
_QPLUS = {"name": "qPlus #1", "material": "PtIr",
          "fabrication": "cut", "form": "qplus"}
_FE = {"name": "Fe #1", "material": "Fe", "fabrication": "etched",
       "form": "stm_wire"}


def _resolve(fields, explicit=None, facts=None):
    return resolve_conditioning(fields, explicit, facts=facts, overrides={},
                                skip_facts_lookup=True)


# ── 方案表:值确实随针尖变 ──────────────────────────────────────────────────

def test_different_tips_get_different_pulse_voltages() -> None:
    """如果所有针尖拿到同一个数,这整张表就没有存在意义。"""
    w = _resolve(("pulse_v",), facts=_W_ETCHED).params["pulse_v"]
    ptir = _resolve(("pulse_v",), facts=_PTIR_CUT).params["pulse_v"]
    qplus = _resolve(("pulse_v",), facts=_QPLUS).params["pulse_v"]
    assert w > ptir > qplus, (
        f"钨(硬)应比铂铱(软)强、铂铱应比 qPlus(易损)强，实际 {w}/{ptir}/{qplus}")


def test_magnetic_tips_get_a_gentler_policy_than_tungsten() -> None:
    """磁性针尖的处理会改变磁构型 —— 不只是几何形状。"""
    fe = _resolve(("pulse_v",), facts=_FE).params["pulse_v"]
    w = _resolve(("pulse_v",), facts=_W_ETCHED).params["pulse_v"]
    assert fe < w


def test_unregistered_tip_falls_back_to_the_conservative_default() -> None:
    r = _resolve(("pulse_v", "max_abs_pulse_v"), facts=None)
    assert r.params["pulse_v"] == 3.0
    assert r.tip is None
    assert any("未登记" in n for n in r.notes)


def test_lookup_falls_back_from_exact_to_coarser_keys() -> None:
    """表里只写差异 —— 没收录的组合要落到更粗的那一档,而不是掉到通用默认。"""
    # (Ir, fib, stm_wire) 没有精确条目，应落到 (Ir, *, stm_wire)
    ir = _resolve(("pulse_v",), facts={"material": "Ir", "fabrication": "fib",
                                       "form": "stm_wire"})
    generic = _resolve(("pulse_v",), facts={"material": "Zz", "fabrication": "fib",
                                            "form": "stm_wire"})
    assert ir.params["pulse_v"] == 4.0
    assert generic.params["pulse_v"] == 4.0   # (*, *, stm_wire) 那一档


def test_qplus_envelope_is_the_tightest() -> None:
    """qPlus 仍然是最严的一档 —— **但 2026-08-10 起不再是每个维度都更严**。

    qPlus 的 ``max_poke_depth_m`` 从 0.5 nm 抬到 5 nm(真机判断:
    5 nm 以内的下压对 W-qPlus 音叉无损),而金属丝档的通用值正好也是 5 nm。
    于是下压深度这一维**变成了相等**。

    **后续更新**:脉冲电压那一维也抬齐了(3 V → 10 V)。于是 qPlus 的**两个包络**现在都与金属丝档相等,「每一维都更严」这个
    不变式整个不成立。

    这条测试因此改成断言**两件不同的事**,而不是把 ``<`` 松成 ``<=`` 了事:
      * **包络**(墙):qPlus ``<=`` 金属丝,且两个数都钉住标定给的值 ——
        免得下一个人看到相等以为是表写漏了、顺手改回 0.5 nm / 3 V;
      * **方案表推荐值**(平时走多少):qPlus 仍然严格更轻。
        墙被抬高不等于平时就该贴着墙走,这两层不能混。

    **要推翻(即恢复「每一维都更严」)需要回答**:这两个数都是在一台
    W-qPlus 上标定的;换一根 qPlus 针、换一台机器,它们还成立吗?
    """
    q = resolve_policy(_QPLUS)
    w = resolve_policy(_W_ETCHED)
    # 两个维度都被抬齐到与金属丝档相同:下压深度 0.5 nm → 5 nm;脉冲电压 3 V → 10 V。
    # 所以「qPlus 的**包络**每一维都更严」这个不变式**整个不成立了**,不要再往回改。
    # 后续更新:包络那一半彻底没了 —— 所有档的两个上限统一拉满(10 V / 10 nm)。
    # ⇒ 「qPlus 的包络更严」现在连 `<=` 的意义都没有了,它就是**相等**。
    # 钉住相等而不是删掉这两行:免得下一个人看到 qPlus 和金属丝一样,
    # 以为是表写漏了、顺手把 qPlus 改回 3 V / 0.5 nm。
    assert q["max_abs_pulse_v"] == w["max_abs_pulse_v"] == 10.0
    assert q["max_poke_depth_m"] == w["max_poke_depth_m"] == 1.0e-8
    # 仍然更严的是**方案表推荐值**(不是包络):qPlus 上一发脉冲、一次尝试更少、
    # 扎得更浅。包络是「不许越过的墙」,方案表是「平时该用多少」——
    # 墙被抬高不等于平时就该贴着墙走。
    assert q["pulse_v"] < w["pulse_v"]
    assert q["pulse_count"] <= w.get("pulse_count", q["pulse_count"])
    assert abs(q["poke_deep_depth_m"]) < abs(w["poke_deep_depth_m"])


# ── 优先级链 + 来源痕迹 ─────────────────────────────────────────────────────

def test_explicit_value_wins_and_is_traced() -> None:
    r = _resolve(("pulse_v",), {"pulse_v": 4.5}, facts=_W_ETCHED)
    assert r.params["pulse_v"] == 4.5
    assert r.trace["pulse_v"] == SOURCE_EXPLICIT


def test_policy_value_is_traced_as_such() -> None:
    r = _resolve(("pulse_v",), facts=_W_ETCHED)
    assert r.trace["pulse_v"] == SOURCE_POLICY


def test_operator_override_beats_the_factory_table() -> None:
    """出厂值是文献起点,标着「待真机标定」;实机标出来的真值必须能盖掉它。"""
    r = resolve_conditioning(("pulse_v",), {}, facts=_W_ETCHED,
                             overrides={"pulse_v": 6.5},
                             skip_facts_lookup=True)
    assert r.params["pulse_v"] == 6.5
    assert r.trace["pulse_v"] == SOURCE_OVERRIDE


def test_explicit_still_beats_an_operator_override() -> None:
    r = resolve_conditioning(("pulse_v",), {"pulse_v": 2.0}, facts=_W_ETCHED,
                             overrides={"pulse_v": 6.5},
                             skip_facts_lookup=True)
    assert r.params["pulse_v"] == 2.0
    assert r.trace["pulse_v"] == SOURCE_EXPLICIT


def test_fields_are_resolved_independently() -> None:
    """逐字段独立走链,不是整组切换 —— 给了 pulse_v 不该让 count 也退回默认。"""
    r = _resolve(("pulse_v", "pulse_count"), {"pulse_v": 4.5}, facts=_QPLUS)
    assert r.trace["pulse_v"] == SOURCE_EXPLICIT
    assert r.trace["pulse_count"] == SOURCE_POLICY


def test_human_trace_names_every_source() -> None:
    """三种来源要在同一行痕迹里分得清:调用方给的、方案表给的、没人管落到默认的。

    ``pulse_count`` 在钨那一档里没被覆盖 —— 痕迹如实标「通用默认」而不是假装
    它来自方案表,这正是要钉住的诚实。"""
    r = _resolve(("pulse_v", "shaper_bias_v", "pulse_count"),
                 {"pulse_v": 4.5}, facts=_W_ETCHED)
    text = r.human_trace()
    assert "pulse_v" in text and "调用方指定" in text
    assert "shaper_bias_v" in text and "针尖方案表" in text
    assert "pulse_count" in text and "通用默认" in text


# ── 拒绝而不是夹紧 ──────────────────────────────────────────────────────────

def test_over_envelope_pulse_is_refused_not_clamped() -> None:
    """夹紧会让调用方以为自己用的是原来那个值 —— 而这里的错代价是硬件。"""
    # 11 V:qPlus 的脉冲包络 2026-08-10 被用户定到 10 V,所以载体从 9 V 改成
    # 一个仍然越界的值。断言一个字没改 —— 变的是墙的位置,不是「越墙要被拒」。
    r = _resolve(("pulse_v",), {"pulse_v": 11.0}, facts=_QPLUS)
    assert not r.ok
    assert r.params["pulse_v"] == 11.0, "拒绝时不改值,原样回报让人看清自己给了什么"
    assert any("超出" in x for x in r.refusals)


def test_qplus_refusal_explains_the_irreversibility() -> None:
    r = _resolve(("pulse_v",), {"pulse_v": 11.0}, facts=_QPLUS)
    joined = " ".join(r.refusals)
    assert "不可逆" in joined
    assert "拆机" in joined or "重装" in joined


def test_within_envelope_is_allowed() -> None:
    r = _resolve(("pulse_v",), {"pulse_v": 2.5}, facts=_QPLUS)
    assert r.ok


def test_poke_depth_over_envelope_is_refused() -> None:
    """探针值必须**跟着包络走**,否则这条测试会变成一条永远绿的空话。

    搬过两次家:2026-08-10 包络 0.5 nm → 5 nm,探针 -5e-9 → -6e-9;
    2026-08-12 包络 5 nm → **10 nm**(全档拉满),探针 -6e-9 → **-1.1e-8**。
    每一次都是同一个道理:**一个不再越界的「越界值」什么都没测**。
    """
    r = _resolve(("poke_deep_depth_m",), {"poke_deep_depth_m": -1.1e-8},
                 facts=_QPLUS)
    assert not r.ok
    assert any("下压深度" in x for x in r.refusals)


def test_poke_depth_exactly_at_the_envelope_is_allowed() -> None:
    """边界是闭的 —— 上限本身就该能扎(2026-08-12 起上限是 10 nm)。"""
    r = _resolve(("poke_deep_depth_m",), {"poke_deep_depth_m": -1.0e-8},
                 facts=_QPLUS)
    assert r.ok, r.refusals


def test_pulse_count_over_envelope_is_refused() -> None:
    r = _resolve(("pulse_count",), {"pulse_count": 10}, facts=_QPLUS)
    assert not r.ok


def test_unregistered_tip_does_not_refuse_a_plausible_value() -> None:
    """未登记时系统不知道装的是什么针,没资格替用户否决(fail-open)。"""
    r = _resolve(("pulse_v",), {"pulse_v": 5.0}, facts=None)
    assert r.ok


def test_a_wildly_large_value_is_still_refused_when_unregistered() -> None:
    """fail-open 不等于没有底线 —— 通用档也有包络。"""
    r = _resolve(("pulse_v",), {"pulse_v": 50.0}, facts=None)
    assert not r.ok


# ── 健壮性 ──────────────────────────────────────────────────────────────────

def test_junk_values_do_not_crash_the_envelope_check() -> None:
    r = _resolve(("pulse_v", "poke_deep_depth_m"),
                 {"pulse_v": "很大", "poke_deep_depth_m": None}, facts=_QPLUS)
    assert isinstance(r.refusals, list)


def test_notes_carry_the_tip_specific_advice() -> None:
    r = _resolve(("pulse_v",), facts=_PTIR_CUT)
    joined = " ".join(r.notes)
    assert "软" in joined or "轻" in joined


def test_resolve_policy_marks_factory_values_as_provisional() -> None:
    """出厂值是文献起点,不是真值 —— 表里每一档都要说出这件事。"""
    import mast.core.tip_conditioning_policy as pol

    marked = sum(1 for v in pol._POLICY.values()
                 if "待真机标定" in str(v.get("_note", "")))
    assert marked >= 8, "方案表里绝大多数档都应标注「待真机标定」"
