"""有量纲的 skill 参数：**广告出去的防护 = 真正执行的防护**，全仓逐个钉。

WHY THIS FILE EXISTS
====================
2026-08-10。``ConfigureScan.center_x_m`` 的描述里逐字写着：

    **写成带 SI 前缀的字符串** …… 前缀不可省略 —— 裸数字会被拒绝,
    因为一个丢了量级的裸数字仍然是合法数字,错一万亿倍也无人察觉。

而把 ``"2.2"`` 送进去，它返回 ``2.2`` —— **2.2 米**，一声不吭。

根因不是漏了防护，是**同一个判断被算了两遍**：

* ``_schema_from_metadata`` 用「ParameterSpec 自己的范围 ∩ 活动安全包络」算 strict；
* ``_si_params``（解析侧）只用「ParameterSpec 自己的范围」算 strict。

对于 ``center_x_m`` 这种**范围完全来自 ``SafetyLimits.xy_*_m``、自己不写 min/max**
的参数，两个答案相反：广告 strict，执行 lenient。实测受影响 16 个参数 / 8 个技能，
**全是米制中心坐标**，而它们不写 min/max 恰恰是因为写了就成了第二真源 ——
**越是按规范写，防护掉得越干净。**

修法是让两边共用 ``effective_bounds()``。本文件是那条修法的承重钉。

WHAT EACH TEST PINS
===================
``test_advertised_equals_enforced``   ← 今天这个 bug 的回归钉（判定层）
``test_bare_mantissa_is_refused``     ← **落地那个字节**：真的送一串 "2.2" 进解析器
``test_every_metre_param_is_covered`` ← 新加一个 unit="m" 的数值参数而不接防护 = 红
``test_the_exemptions_are_still_true``← 豁免名单里的理由现在还成不成立

「判定层等价、只有落地不同」的性质，判定层的钉子原理上抓不到 —— 所以
``test_bare_mantissa_is_refused`` 走的是 ``_coerce_si_params`` 的真实返回值，
不是重算一遍 ``needs_strict_prefix``。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_dimensioned_param_coverage.py -x -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
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

import pytest

from mast.agents._shared.skill_adapter import (
    _coerce_si_params,
    _schema_from_metadata,
    _si_params,
    effective_bounds,
)
from mast.core.registry import SkillRegistry
from mast.core.si_quantity import needs_strict_prefix

_DISCOVER = ("mast.skills.builtins", "mast.skills.composite", "mast.skills.paper")


@pytest.fixture(scope="module")
def metas():
    reg = SkillRegistry()
    reg.discover(*_DISCOVER)
    return reg.list_skills()


# ─────────────────────────────────────────────────────────────────────────────
# 豁免名单 —— **显式**，且每条都要写清为什么。
#
# 名单只许放「量纲是米、但裸数字确实合法」的参数。它是空的，而空名单也要留着：
# 下一个想加豁免的人会看见这里要求一个理由，而不是顺手把判据改松。
# （闸门只编码「哪些必须严」而不编码「哪些允许松」的话，下一轮会把该严的一起放过。）
# ─────────────────────────────────────────────────────────────────────────────
METRE_PARAM_EXEMPTIONS: dict[tuple[str, str], str] = {
    # ("SkillName", "param_name"): "为什么裸数字在这里是合法的"
}


def _numeric_dimensioned(meta):
    """本技能里「有量纲的数值参数」→ [(spec, 是否在 string 通道)]。"""
    si = _si_params(meta)
    out = []
    for spec in getattr(meta, "parameters", None) or []:
        if getattr(spec, "allowed_values", None):
            continue
        if getattr(spec, "type", "") not in ("float", "number"):
            continue
        if not getattr(spec, "unit", ""):
            continue
        out.append((spec, spec.name in si))
    return out


def _metre_specs(metas):
    for meta in metas:
        for spec, in_channel in _numeric_dimensioned(meta):
            if (getattr(spec, "unit", "") or "").strip() == "m":
                yield meta, spec, in_channel


# ── 自检：一条匹配不到任何东西的闸门，和「确实没问题」输出一模一样 ──────────────
def test_selfcheck_the_gate_actually_matches_things(metas):
    assert len(metas) >= 400, f"registry 只 discover 到 {len(metas)} 个 skill"
    dimensioned = [s for m in metas for s in _numeric_dimensioned(m)]
    assert len(dimensioned) >= 300, f"有量纲参数只找到 {len(dimensioned)} 个"
    metres = list(_metre_specs(metas))
    assert len(metres) >= 40, f"米制参数只找到 {len(metres)} 个"
    # 三个已知的锚点必须在里面（名字从真源取，不是我编的）
    names = {(m.name, s.name) for m, s, _ in metres}
    for anchor in (("ConfigureScan", "center_x_m"),      # 今天的 bug
                   ("FindCleanSpot", "from_x_m"),        # 真机报上来的那个
                   ("ScanAt", "center_x_m")):            # 范式来源
        assert anchor in names, f"锚点 {anchor} 不在米制参数里 —— 判据坏了"


# ── 主钉 1：广告 = 执行 ──────────────────────────────────────────────────────
#: 严格档的描述里必然出现的那句承诺（真源：``_schema_from_metadata`` 的 hint）。
_PROMISE = "前缀不可省略"


def test_advertised_equals_enforced(metas):
    """描述里承诺的严格度，必须就是解析器执行的严格度。

    **读的是真正生成给模型看的那段描述文本**，不是把 ``needs_strict_prefix``
    再算一遍 —— 重算只会证明「我这次用了同一个表达式」，而 2026-08-10 的缺陷
    恰恰是两个表达式各自都「算得对」。要钉的是模型读到的那个字节。
    """
    mismatched = []
    for meta in metas:
        enforced = _si_params(meta)
        try:
            js = _schema_from_metadata(meta).model_json_schema()
        except Exception as exc:  # pragma: no cover — schema 必须建得起来
            pytest.fail(f"{meta.name} 的 args schema 建不起来: {exc}")
        for spec, in_channel in _numeric_dimensioned(meta):
            if not in_channel:
                continue
            desc = (js.get("properties", {}).get(spec.name, {})
                    .get("description", "") or "")
            advertised = _PROMISE in desc
            if advertised != enforced[spec.name]:
                mismatched.append(
                    f"{meta.name}.{spec.name}: 描述里{'有' if advertised else '没有'}"
                    f"「{_PROMISE}」，而解析器 strict={enforced[spec.name]}")
    assert not mismatched, (
        "描述承诺的和解析器执行的不一致 —— 模型会读到「前缀不可省略」，"
        "然后发现裸数字被接受了：\n  " + "\n  ".join(mismatched))


def test_selfcheck_the_promise_string_still_exists(metas):
    """``_PROMISE`` 是从描述里按子串找的。它一旦被改写，上面那条会**全绿**
    （所有 advertised 都变成 False，而如果同时没有 strict 参数就无人察觉）。
    所以单独钉一条：仓里必须真的有参数带着这句话。"""
    seen = 0
    for meta in metas:
        js = _schema_from_metadata(meta).model_json_schema()
        for spec, in_channel in _numeric_dimensioned(meta):
            if in_channel and _PROMISE in (
                    js.get("properties", {}).get(spec.name, {})
                    .get("description", "") or ""):
                seen += 1
    assert seen >= 80, (
        f"只有 {seen} 个参数的描述里出现「{_PROMISE}」—— 措辞被改过？"
        f"改了就把 _PROMISE 一起改，否则 test_advertised_equals_enforced 会假绿。")


# ── 主钉 2：落地那个字节 ─────────────────────────────────────────────────────
def test_bare_mantissa_is_refused(metas):
    """真的把 ``"2.2"`` 送进每一个米制参数，必须拿到解析错误。

    **不是**再算一遍 ``needs_strict_prefix`` —— 那样钉的是判定层，而这次的缺陷
    判定层两边都「对」，只有落地不同。所以这里读的是 ``_coerce_si_params`` 的
    真实返回值。
    """
    leaked = []
    for meta, spec, in_channel in _metre_specs(metas):
        if (meta.name, spec.name) in METRE_PARAM_EXEMPTIONS:
            continue
        out, errors = _coerce_si_params(meta, {spec.name: "2.2"})
        if not errors:
            leaked.append(f"{meta.name}.{spec.name} → {out.get(spec.name)!r}"
                          f"（当成 {out.get(spec.name)} 米收下了）")
    assert not leaked, (
        "这些米制参数把裸数字 '2.2' 当成 2.2 米收下了。STM 压电量程 ±1.5 µm，"
        "2.2 米差六个数量级，而它不报错：\n  " + "\n  ".join(leaked))


def test_a_prefixed_value_still_parses(metas):
    """反向钉：把闸门改严不能顺手把正常值也拒了。"""
    for meta, spec, _ in _metre_specs(metas):
        out, errors = _coerce_si_params(meta, {spec.name: "2.2n"})
        assert not errors, f"{meta.name}.{spec.name} 拒了合法的 '2.2n': {errors}"
        assert out[spec.name] == pytest.approx(2.2e-9)


def test_internal_float_callers_pass_through(metas):
    """composite / executor / 测试持有的是真 float，strict 不该拦它们。"""
    for meta, spec, _ in _metre_specs(metas):
        out, errors = _coerce_si_params(meta, {spec.name: 2.2e-9})
        assert not errors, f"{meta.name}.{spec.name} 拒了内部 float: {errors}"
        assert out[spec.name] == pytest.approx(2.2e-9)


# ── 主钉 3：新参数不接防护 = 红 ───────────────────────────────────────────────
def test_every_metre_param_is_covered(metas):
    """每个 ``unit="m"`` 的数值参数都必须 (a) 走 string 通道 (b) 强制前缀。

    **刻意不要求它一定有 min/max。** 米这个量纲本身就足以判定「裸尾数不可能」
    （见 ``_STRICT_BY_DIMENSION``）—— 若这里改成要求声明范围，就等于把
    「谁记得写 min/max」重新变成防护的前提，而那正是这次缺陷的成因。
    """
    problems = []
    for meta, spec, in_channel in _metre_specs(metas):
        key = (meta.name, spec.name)
        if key in METRE_PARAM_EXEMPTIONS:
            continue
        if not in_channel:
            problems.append(f"{meta.name}.{spec.name}: 不在 SI string 通道里")
            continue
        if not _si_params(meta).get(spec.name):
            problems.append(
                f"{meta.name}.{spec.name}: 走了 string 通道但前缀不是强制的 "
                f"(生效范围 {effective_bounds(spec)})")
    assert not problems, (
        "米制参数缺防护。首选修法：确认它的 unit 确实是 \"m\"（``_STRICT_BY_DIMENSION`` "
        "会自动让它强制前缀）。若这个参数上裸数字真的合法，进 METRE_PARAM_EXEMPTIONS "
        "并写清理由：\n  " + "\n  ".join(problems))


def test_the_dimension_rule_covers_the_ones_bounds_would_miss(metas):
    """量纲规则**在承重**：有多少米制参数是「只靠量纲」才强制的。

    ⚠️ 这条**不是**变异测试，删掉 ``_STRICT_BY_DIMENSION`` 不会让它红
    （实测过：变异②只红了另外两条）—— 它数的是 ``needs_strict_prefix`` 那一侧，
    而那一侧不受量纲规则影响。写下这句是因为第一版的 docstring 声称它是变异防护，
    而那是假话；承重验证由 ``test_bare_mantissa_is_refused`` 做。

    它存在的理由是 既有教训：下一个人会重新
    想到「靠 min/max 判就够了」（因为那看起来显然对）。这条把「有 N 个参数
    光靠 min/max 是保不住的」变成一个数字，让那个念头当场被数据反驳。
    """
    only_dimension = []
    for meta, spec, in_channel in _metre_specs(metas):
        if not in_channel:
            continue
        if not needs_strict_prefix(*effective_bounds(spec)):
            only_dimension.append(f"{meta.name}.{spec.name}")
    assert len(only_dimension) >= 15, (
        f"只有 {len(only_dimension)} 个米制参数是靠量纲规则才强制的 —— "
        f"要么有人补了 min/max（好事，但请更新这个下限），要么规则没生效。"
        f"\n  {only_dimension}")


# 这些米制参数声明了范围，测试必须核对越界拒绝，避免 SI 前缀丢失后接受错误尺度。
BOUNDED_METRE_PARAMS: tuple[tuple[str, str], ...] = (
    ("FindCleanSpot", "from_x_m"),
    ("FindCleanSpot", "from_y_m"),
    ("FindCleanSpot", "max_distance_m"),
)


@pytest.mark.parametrize("skill_name,param", BOUNDED_METRE_PARAMS)
def test_declared_metre_bounds_are_enforced(metas, skill_name, param):
    """这些参数的 min/max 必须**存在且真的拒绝越界值**。

    与前缀防护是两件不同的事，所以分开钉：
      * 前缀强制  → 拦「丢了量级的裸数字」（``"2.2"``）
      * min/max  → 拦「量级写对了但超出这个技能自己范围」的值（``"5m"`` = 5 mm）
    只钉前一件的话，把 min/max 删掉闸门照样全绿 —— 实测过（变异③）。
    """
    reg = SkillRegistry()
    reg.discover(*_DISCOVER)
    skill = reg.get(skill_name)()
    spec = next((s for s in skill.metadata().parameters if s.name == param), None)
    assert spec is not None, f"{skill_name} 上没有 {param} 了"
    assert spec.max_value is not None, (
        f"{skill_name}.{param} 的 max_value 被删了 —— 一个前缀写对的越界值"
        f"（如 '5m' = 5 毫米）会被原样收下")
    over = abs(spec.max_value) * 10
    errors = skill.validate_params({param: over})
    assert any(param in e for e in errors), (
        f"{skill_name}.{param} = {over} 超出上限 {spec.max_value}，"
        f"validate_params 却没拒：{errors}")


def test_a_metre_named_param_cannot_leave_the_gate(metas):
    """名字以 ``_m`` 结尾的数值参数，必须**还在**米制集合里。

    上面所有检查都从 ``unit == "m"`` 起步 —— 于是把 ``unit="m"`` 删掉（或写成
    ``unit=""``）的参数会**从闸门的视野里消失**，闸门照样全绿。这正是
    「闸门看不见它要拦的东西」那一类假绿：拆掉防护和防护完好，输出一模一样。

    所以这一条换一个**独立的**判据 —— 参数名 —— 来数同一批东西。两个判据要
    同时被绕过，才可能悄悄退出。
    """
    import re

    # `_nm` / `_pm` / `_um` / `_mm` 是自带前缀的长度，值在 1 附近，不归这条管。
    # `_per_m` 结尾的量纲**不是**米，是「每米」：`k_n_per_m` 是牛顿每米的弹性常数，
    # 给它 unit="m" 会让 SI 前缀通道把它当长度解析。名字里的 per 是判据的一部分。
    metre_named = re.compile(r"(?<![npum])(?<!per)_m$")
    live = {(m.name, s.name) for m, s, _ in _metre_specs(metas)}
    escaped = []
    for meta in metas:
        for spec in getattr(meta, "parameters", None) or []:
            if getattr(spec, "type", "") not in ("float", "number"):
                continue
            if getattr(spec, "allowed_values", None):
                continue
            if not metre_named.search(spec.name):
                continue
            key = (meta.name, spec.name)
            if key in METRE_PARAM_EXEMPTIONS or key in live:
                continue
            escaped.append(f"{meta.name}.{spec.name}: 名字说它是米，"
                           f"但 unit={getattr(spec, 'unit', '')!r} → 不在 SI 通道里")
    assert not escaped, (
        "这些参数名字带 _m 却没有 unit=\"m\"，于是绕开了本文件所有其它检查：\n  "
        + "\n  ".join(escaped))


def test_the_exemptions_are_still_true(metas):
    """豁免名单里的参数必须还存在。删掉的参数留在名单里 = 名单在骗人。"""
    live = {(m.name, s.name) for m, s, _ in _metre_specs(metas)}
    stale = sorted(k for k in METRE_PARAM_EXEMPTIONS if k not in live)
    assert not stale, f"豁免名单里这些参数已经不存在了，请删除：{stale}"
    for key, reason in METRE_PARAM_EXEMPTIONS.items():
        assert reason and len(reason) > 10, f"{key} 的豁免理由太短，说不清为什么"
