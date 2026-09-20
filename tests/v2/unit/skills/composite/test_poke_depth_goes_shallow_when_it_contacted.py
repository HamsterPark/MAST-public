"""D 相的深度必须能**往回走** —— 2026-08-15 一百针标定纠正的方向错误。

## 被否掉的那个设计(这个文件就是钉它的)

    「同一深度换了 N 个地方都不圆 ⇒ 深度不够 ⇒ 加深」

能走到那一行,**就说明已经接触上了**(没接触走的是 no_change 那条分支)。
接触上了还不圆,不是碰得不够深,是**搬多了**。一百针的实测:

    深度 → 轴比   Spearman ρ = −0.46, p = 0.005  (n=35,只算真接触上的)
    对照组(没接触的那 54 针)      ρ = −0.14, p = 0.19   ⇒ 不是地形伪迹
    机制  深度 → 面积 +0.53,面积 → 轴比 −0.60
    各档接触时轴比中位  0.740 / 0.641 / 0.441 / 0.527 / 0.353

到 2 nm 判据几乎不可能过。旧逻辑是朝「永远达不到」走,而且加深不可逆。
用户范本序列 500 → 500 → 500 → **200** pm,最后一步也是变浅。

## 为什么要一个专门的测试

方向错了但代码**跑得通、日志也好看**:`depth * frac` 这行认不出自己在往哪走,
`entry["note"]` 里那句「加深到 xxx pm」照样打印得理直气壮。这种错只能靠
「把方向本身钉成断言」抓住 —— 见 feedback/pin_rejected_designs_as_tests。
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _dont_touch_the_tip_registry(monkeypatch):
    """``_forge_wf`` 会去读真的针尖登记表 —— 测试不许碰用户数据。

    (同一个坑在 test_forge_time_budget_and_merge 里踩过一次。)
    """
    import mast.skills.composite.forge_au_tip as m

    if hasattr(m, "_tip_policy_now"):
        monkeypatch.setattr(m, "_tip_policy_now", lambda: None, raising=True)


def _wf(**over):
    from mast.skills.composite.forge_au_tip import _forge_wf

    return _forge_wf(dict(over))


def _not_round_branch() -> str:
    """截出「扎上了但不圆」那一支的源码。

    第一版用的是 ``src[head:head + 2600]`` —— 固定字符窗口。同一天在这个分支
    上多加了几行注释,窗口就滑出了要断言的代码,两条测试**变绿之外的方式**红了。
    行序/字符偏移不该承重(见 feedback/per_page_wiring_needs_a_sweep),
    所以改成锚到分支的**结束标志**上。
    """
    import inspect

    from mast.skills.composite import _tip_phases

    src = inspect.getsource(_tip_phases)
    head = src.index('if cl.get("double_tip") is True or not cl.get("is_round"):')
    tail = src.index('# 单峰且圆', head)          # 这一支的下一支
    body = src[head:tail]
    assert len(body) > 400, "截出来的分支太短,锚点可能挪位了"
    return body


# ── 参数本身 ────────────────────────────────────────────────────────────────
def test_shallow_factor_is_the_operator_sequence_not_an_invented_number():
    """0.4 必须正好是范本 500 → 200 pm 的那一跳。"""
    assert _wf().poke_shallow_frac == pytest.approx(200.0 / 500.0)


def test_shallow_factor_cannot_be_set_to_deepen():
    """填 ≥1 就把这条分支悄悄变回加深 —— 闸门必须拦住。

    这是「判据认不出方向」的补偿:代码看不出 1.3 是加深,`_BOUNDS` 看得出。

    这里的语义是仓库统一的那条:**越界拒绝、不夹紧、不抛** —— 越界值被丢弃,
    字段回落到默认。夹紧才是危险的那个:调用方以为自己设的是 1.2,
    实际拿到 0.99,而且没人告诉他(见 special_tip_workflow._BOUNDS 同一条论证)。
    """
    from mast.core.noble_tip_workflow import _BOUNDS, _EXCLUSIVE_UPPER, resolve

    lo, hi = _BOUNDS["poke_shallow_frac"]
    assert hi == 1.0
    assert "poke_shallow_frac" in _EXCLUSIVE_UPPER, "1.0 必须是开区间:等于 1 就是不变浅"

    default = resolve({}).poke_shallow_frac
    for bad in (1.0, 1.2, 2.0, -0.5):
        got = resolve({"poke_shallow_frac": bad}).poke_shallow_frac
        assert got == default, f"{bad} 被接受了 —— 这条分支会变回加深"
        assert got < 1.0

    for ok in (0.6, 0.99):                              # 界内的仍要能设
        assert resolve({"poke_shallow_frac": ok}).poke_shallow_frac == pytest.approx(ok), \
            "闸门把合法值也拦了"


def test_it_is_not_an_operator_knob_yet_and_that_is_deliberate():
    """`poke_shallow_frac` **故意不在** ``_WF_PARAM_KEYS`` 里。

    第一版把边界写进 `_BOUNDS` 就以为这是可调项 —— 于是这个测试的
    「合法值 0.6 也被拦」当场红了。真相是白名单没收这个键,`_forge_wf`
    连读都不读它,所有值(合法的、非法的)一律回落默认。

    **这不是缺陷,是选择**:0.4 只有一根针、一次会话的证据,还不该变成旋钮。
    而那张单子的注释写得很清楚 —— 加进去而不同时在 ``metadata()`` 里声明,
    就成了「死读」:读起来像可调项,调了没有任何效果,两边都看不出问题。

    **要把它变成旋钮,必须同时做两件事**:加进 ``_WF_PARAM_KEYS`` +
    在 ``metadata()`` 的参数表里声明。这个断言就是那道提醒。
    """
    from mast.skills.composite.forge_au_tip import _WF_PARAM_KEYS

    assert "poke_shallow_frac" not in _WF_PARAM_KEYS
    assert "poke_depth_min_nm" not in _WF_PARAM_KEYS


def test_depth_floor_is_below_the_initial_and_maximum_depth():
    """验证配置的深度下限与起始深度、上限的关系；公开默认值不构成目标仪器标定。"""
    wf = _wf()
    assert wf.poke_depth_min_nm == pytest.approx(0.2)
    assert wf.poke_depth_min_nm < wf.poke_depth_nm, "地板不该高过起扎深度"
    assert wf.poke_depth_min_nm < wf.poke_depth_max_nm


# ── 方向 ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("start_nm", [0.5, 1.0, 2.0])
def test_contacted_but_not_round_goes_shallower_never_deeper(start_nm):
    """核心断言:这条分支算出来的下一档,**必须 ≤ 当前档**。"""
    wf = _wf()
    nxt = max(start_nm * wf.poke_shallow_frac, wf.poke_depth_min_nm)
    assert nxt <= start_nm, f"{start_nm} nm → {nxt} nm 是加深,方向反了"


def test_it_does_not_dig_below_the_floor():
    wf = _wf()
    deep = wf.poke_depth_min_nm
    for _ in range(20):                      # 反复触发这条分支
        deep = max(deep * wf.poke_shallow_frac, wf.poke_depth_min_nm)
    assert deep == pytest.approx(wf.poke_depth_min_nm)


def test_no_contact_still_deepens():
    """**没碰到**那条分支不许被这次改动带歪 —— 它该继续加深。

    两种失败长得像(都是「这一针不成」),处置正相反。把两条都断言上,
    以后谁想「统一一下」就会看见这个测试。
    """
    wf = _wf()
    assert wf.poke_deepen_frac > 0.0
    assert 1.0 * (1.0 + wf.poke_deepen_frac) > 1.0


# ── 代码里真的这么写了吗 ────────────────────────────────────────────────────
def test_the_not_round_branch_calls_shallow_not_deepen():
    """读源码确认:`not is_round` 那一段里出现的是 `poke_shallow_frac`。

    上面几条都在算术上成立,但算术成立不等于**这条分支用了它** ——
    见 feedback/producer_wired_consumer_absent:「已经有了」是生产方的话。
    """
    import inspect

    from mast.skills.composite import _tip_phases

    body = _not_round_branch()
    assert "poke_shallow_frac" in body, "不圆的分支没用上变浅系数"
    assert "poke_deepen_frac" not in body, "不圆的分支里还留着加深"


def test_it_says_so_when_it_cannot_go_shallower():
    """贴地板时不许打印「变浅到 xxx」—— 那是一句假话。

    「读不到/做不到 被折叠成一个看着正常的值」是这个仓库最常复发的缺陷族;
    日志也算值。
    """
    import inspect

    from mast.skills.composite import _tip_phases

    body = _not_round_branch()
    assert "已在最浅档" in body, "触底时没有单独的说法"
    assert body.index("已在最浅档") < body.index("变浅到"), "触底那一支必须先判"
