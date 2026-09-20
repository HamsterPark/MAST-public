"""扎针落点应位于干净台面中央，找不到则按预算重试，不回退到未经验证的几何位置。"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _dont_touch_the_tip_registry(monkeypatch):
    import mast.skills.composite.forge_au_tip as m

    if hasattr(m, "_tip_policy_now"):
        monkeypatch.setattr(m, "_tip_policy_now", lambda: None, raising=True)


def _wf(**over):
    from mast.skills.composite.forge_au_tip import _forge_wf

    return _forge_wf(dict(over))


def test_the_window_is_much_wider_than_the_cluster_frame():
    """窗要比簇图**宽得多**,不是刚好装下。

    「刚好装下」只保证不跨台阶;要保证**视野里没有别的东西**,得留出余量。
    #36 那三块相距 10.75 nm —— 比 10 nm 的簇图还宽,20 nm 的窗拦不住它。
    """
    wf = _wf()
    # 4× → 3×(2026-08-16):窗从 50 收到 35 nm 是**真机测出来的拐点**
    # (6 张真 200 nm 图上有解率 3/6 → 5/6,落点总数 3 → 14)。
    # 「宽得多」这个理由不变,倍数跟着样品的台面宽度走。
    assert wf.poke_flat_window_nm >= 3 * wf.cluster_scan_nm, (
        f"平区窗 {wf.poke_flat_window_nm} nm 相对簇图 {wf.cluster_scan_nm} nm 不够宽")


def test_it_keeps_looking_instead_of_giving_up_after_three():
    """「找不到就**反复找**」。3 次就认输是把这句话执行反了。"""
    wf = _wf()
    assert wf.poke_flat_dry_refills >= 10
    # 一次重扫 = 一张找台面图,**来回两遍**(第一版这里只算了单向,
    # 于是 60 分钟的上限碰巧过了 —— 断言过不是因为对,是因为公式错)。
    frame_min = 2 * wf.poke_site_pixels * wf.poke_site_line_time_s / 60.0
    total = wf.poke_flat_dry_refills * frame_min
    assert 4.5 <= frame_min <= 5.5, f"找台面帧时算出来 {frame_min:.1f} min,与文档不符"
    # 搜索预算需要覆盖允许的重扫次数；耗尽后应换区。
    assert total <= 120.0, (
        f"反复找总共 {total:.0f} min —— 再多就该换区了,不是接着扫")


def test_the_site_scan_line_time_keeps_the_tip_at_the_pinned_speed():
    """视野翻倍而线时不变 = 针尖速度也翻倍。这个数必须是**派生的**。

    488 nm/s 刮坏过针尖;`forge_v_tip_nm_s` = 170.65 是他钉过的那一帧的速度。
    ``forge_line_time_s()`` 会兜底限速,但**存着的值就该是对的** ——
    靠兜底意味着每次跑都吐一条「已放慢」的说明,而存值与实跑值不一致
    正是下一个人对不上账的起点。
    """
    wf = _wf()
    speed = wf.poke_site_scan_nm / wf.poke_site_line_time_s
    assert speed == pytest.approx(wf.forge_v_tip_nm_s, rel=0.01), (
        f"找台面图针尖速度 {speed:.0f} nm/s ≠ 上限 {wf.forge_v_tip_nm_s:.0f}")

    # 而且兜底那条路在这个工作点上**不该动手**(动了就说明值设错了)
    from mast.core.noble_tip_workflow import forge_line_time_s

    lt, note = forge_line_time_s(wf, wf.poke_site_scan_nm, wf.poke_site_line_time_s)
    assert lt == pytest.approx(wf.poke_site_line_time_s)
    assert not note, f"出厂值触发了限速说明:{note}"


def test_a_bigger_frame_actually_buys_search_area():
    """放大视野的**理由**要成立:可放窗区域是 (图 − 窗)²,不是图²。

    100 nm 图 / 50 nm 窗 ⇒ 50×50;200 nm 图 ⇒ 150×150 = **9 倍**。
    这条钉住的是「为什么值得多花一倍帧时」,不是某个具体数字。
    """
    wf = _wf()
    room = wf.poke_site_scan_nm - wf.poke_flat_window_nm
    assert room > 0, "窗装不进图"
    assert room >= 2 * wf.poke_flat_window_nm, (
        f"可放窗边长只有 {room:.0f} nm,相对 {wf.poke_flat_window_nm:.0f} nm 的窗"
        "太局促 —— 多数图会零个点,「反复找」会变成空转")


def test_the_window_must_fit_inside_the_site_scan():
    """窗 ≥ 图 ⇒ 永远找不到,而流程会报「这一片台阶太密」——**一句假话**。

    密的不是台阶,是我们要了一个装不下的窗。配置错误被折叠成一个看着合理的
    结论,是这个仓库最常复发的缺陷族;所以当场抛,不要让它跑完再骗人。
    """
    from mast.core.noble_tip_workflow import resolve
    from mast.skills.composite._tip_phases import flat_poke_sites

    wf = _wf()
    assert wf.poke_flat_window_nm < wf.poke_site_scan_nm, "出厂值自己就装不下"

    # ⚠️ 用 ``resolve`` 而不是 ``_forge_wf``:后者只读 ``_WF_PARAM_KEYS`` 白名单,
    # 而这个键不在里面(**故意的** —— 窗口尺寸由流程配置管理,不该是随手可调的
    # 旋钮;加进白名单而不同时在 metadata() 声明就是"死读")。
    bad = resolve({"poke_flat_window_nm": 200.0})   # > poke_site_scan_nm=100
    assert bad.poke_flat_window_nm == pytest.approx(200.0)
    gen = flat_poke_sites(object(), bad, step_prefix="D", used=[], want=3)
    with pytest.raises(ValueError, match="永远找不到"):
        next(gen)


def test_the_site_is_the_centre_of_the_flat_window():
    """「在其中间扎针尖」—— 读的必须是窗的**中心**,不是窗的任一角。"""
    import inspect

    from mast.skills.composite import _tip_phases

    src = inspect.getsource(_tip_phases.flat_poke_sites)
    assert 'center_x_m' in src and 'center_y_m' in src
    assert 'min_window_m' in src, "窗大小要传下去"
    assert 'same_terrace' in src, "必须在**单台面内**,两半各自平坦不算"


def test_the_new_knobs_are_bounded_and_the_int_one_is_int():
    from mast.core.noble_tip_workflow import _BOUNDS, _INT_FIELDS, resolve

    assert "poke_flat_window_nm" in _BOUNDS, "无界的旋钮 = 没有闸门"
    assert "poke_flat_dry_refills" in _BOUNDS
    assert "poke_flat_dry_refills" in _INT_FIELDS
    assert resolve({"poke_flat_dry_refills": 7.4}).poke_flat_dry_refills == 7
    # 越界拒绝、不夹紧、不抛 —— 与本模块其余字段同一条语义
    d = resolve({}).poke_flat_dry_refills
    assert resolve({"poke_flat_dry_refills": 0}).poke_flat_dry_refills == d
    assert resolve({"poke_flat_dry_refills": 999}).poke_flat_dry_refills == d


def test_the_default_window_is_wider_than_the_cluster_frame():
    """验证默认窗口与簇图尺寸的关系，不附带样品标定数据。"""
    wf = _wf()
    assert wf.poke_flat_window_nm == pytest.approx(35.0)
    assert wf.poke_flat_window_nm >= 3 * wf.cluster_scan_nm, (
        "窗仍要明显宽于簇图,否则视野里会混进别的东西")


def test_no_usable_region_is_an_answer_not_an_undecidable():
    """明确没有可用平区时应继续搜索；不得解释为判据未知而回退到几何选点。"""
    import inspect

    from mast.skills.builtins.flat_region import VERDICT_NO_USABLE_REGION
    from mast.skills.composite import _tip_phases

    # ⚠️ 断言**常量**,不是源码字面量。第一版断言的是
    # ``'"verdict": "no_usable_region"' in getsource(flat_region)``,
    # 同一天把它提成共享常量之后当场红了 —— 而红的不是它要防的那件事。
    # (同一个毛病今天犯了第二次:上一次是 reach 那条公式。)
    assert VERDICT_NO_USABLE_REGION == "no_usable_region"

    # 生产方:那个键真的会出现在返回里(拿一张没有平区的真帧跑)
    from mast.skills.builtins import flat_region
    src_prod = inspect.getsource(flat_region)
    assert "VERDICT_NO_USABLE_REGION," in src_prod, "生产方没把它放进 data"

    # 消费方:2026-08-17 重写之后这件事由**类型**保证,比读常量更硬 ——
    # ``flat_poke_sites`` 只有两态(``spent`` / ``ok``),``undecidable`` 那一态
    # 连同它背后的「退回按几何选点继续扎」一起删掉了。
    # 那条兜底等同于「闭着眼睛乱扎」——扎针并非在调平后的台面上进行,而是
    # 直接问地图,导致一次次扎在台阶上。
    #
    # 所以这里不再查「有没有读那个常量」(读了也可能导向错的分支),
    # 改查**那条错的分支根本不存在**。
    src = inspect.getsource(_tip_phases.flat_poke_sites)
    # 查**返回语句**,不查那个词 —— docstring 里正解释着为什么删掉它
    # (「grep 命中 ≠ 源码里有」,本仓为这个形状付过学费)。
    assert 'return "undecidable"' not in src, (
        "flat_poke_sites 又长出了 undecidable 那一态 —— "
        "它的终点是「找不到台面就随便扎」")
    assert 'return "ok", []' in src, "「这一片没有」没有导向「换一块再扫」"
    assert 'return "spent", None' in src, "「换不动了」这条出口没了"


def test_the_search_effort_is_worth_the_time_it_costs():
    """「加大搜索力度」要在时间预算里说得清。"""
    wf = _wf()
    assert wf.poke_flat_dry_refills >= 20
    frame_min = 2 * wf.poke_site_pixels * wf.poke_site_line_time_s / 60.0
    total = wf.poke_flat_dry_refills * frame_min
    assert total <= 120.0, f"反复找总共 {total:.0f} min —— 再多就该换区了"
    # 重扫次数下限确保流程确实允许重复搜索。
    assert wf.poke_flat_dry_refills >= 10
