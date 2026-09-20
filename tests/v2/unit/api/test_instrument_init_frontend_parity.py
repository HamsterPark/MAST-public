"""新仪器初始化：后端目录 ↔ 前端页面的接线检查。

这一页刻意做成**后端驱动**：分组、项目、区间、枚举全部由
``GET /api/instrument-init`` 现给，TSX 里一个都不抄。所以需要钉住的不是「两张表
一样」，而是剩下那几处**只能靠接线**的地方：

1. 严重级别的三个名字 —— TSX 里那张 ``SEVERITY_META`` 用它们查表，查不到会**静默**
   退回「推荐」：一个红色的「必填」变成琥珀色的「推荐」，没有任何报错。
2. 页面**接上去了** —— 路由里有 ``/setup``、AppLayout 里挂了横幅、设置页里有手动
   入口。本仓的教训是「功能缺失往往不是没实现，是没接线」（三处已存在未接）。
3. 徽章色调用的是 ``ui.tsx`` 真的认识的键名（同样是静默退回 default）。

读 TSX 源码文本，与 ``test_instrument_profile_frontend_parity`` 同款：这里没有
能求值它的构建步骤，而**比对两份源**正是重点。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mast.core import instrument_init as ii


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "frontend" / "src").is_dir():
            return p
        p = p.parent
    pytest.skip("frontend/ not present in this checkout")


def _src(*parts: str) -> str:
    f = _repo_root().joinpath("frontend", "src", *parts)
    if not f.is_file():
        pytest.skip(f"{'/'.join(parts)} not present")
    return f.read_text(encoding="utf-8")


def test_severity_names_match_the_backend():
    """TSX 查不到的级别名会静默退回「推荐」—— 必填的红色徽章就没了。"""
    body = _src("pages", "SetupPage.tsx").split("const SEVERITY_META", 1)[1] \
        .split("};", 1)[0]
    declared = set(re.findall(r"^\s{2}(\w+):\s*\{", body, re.M))
    assert declared == {ii.REQUIRED, ii.RECOMMENDED, ii.OPTIONAL}, (
        f"SetupPage 的 SEVERITY_META 覆盖的级别是 {sorted(declared)}，"
        f"后端用的是 {sorted({ii.REQUIRED, ii.RECOMMENDED, ii.OPTIONAL})}")


def test_badge_tones_are_names_ui_actually_knows():
    """``Badge`` 对不认识的 tone **静默**退回 default。"""
    known = set(re.findall(r"^\s{2}(\w+):\s*\"",
                           _src("components", "ui.tsx").split("const BADGE_TONE", 1)[1]
                           .split("};", 1)[0], re.M))
    used = set(re.findall(r'tone="([A-Za-z]+)"', _src("pages", "SetupPage.tsx")))
    used |= set(re.findall(r'tone:\s*"([A-Za-z]+)"', _src("pages", "SetupPage.tsx")))
    bad = sorted(used - known)
    assert not bad, f"SetupPage 用了 ui.tsx 不认识的徽章色调（会静默变中性色）：{bad}"


def test_the_page_is_actually_wired_up():
    """四处接线，缺任何一处这一页就到不了用户面前。

    2026-08-06（「大标签太多」）：顶栏 17 项合并成 10 项，初始化页
    从顶栏的一项降成「设置」下的一段 —— 地址 ``/setup`` → ``/settings/setup``
    （旧地址仍然重定向）。**这一页必须可达这条要求一个字都没变**，变的只是它
    表达在哪里：router.tsx 现在从 ``lib/nav.ts`` 派生路由路径，所以「路由里有没有
    它」这件事要分两处查 —— 导航表里有那一段（否则没人找得到），router 里有那个
    组件映射（否则点进去一片空白）。

    2026-08-04 那次的教训是「功能缺失往往不是没实现，是没接线」。这次的变体是
    「接线的地方换了，而检查还盯着旧地方」—— 检查会红，这没问题；真正要避免的是
    随手把它改成一句更松的断言。所以这里改成查**新的真源**，条数只增不减。
    """
    nav = _src("lib", "nav.ts")
    assert '"setup"' in nav or "seg: \"setup\"" in nav, (
        "lib/nav.ts 的导航表里没有初始化页那一段 —— 没有它，这一页又会变成"
        "「只能进去一次」（2026-08-04 那次的形状）")

    router = _src("router.tsx")
    assert "SetupPage" in router, "router 里没有 SetupPage"
    assert '"/settings/setup"' in router, (
        "router 里没有把 /settings/setup 映射到组件 —— 导航表里有这一段但点进去"
        "会是空白")

    layout = _src("layout", "AppLayout.tsx")
    assert "SetupBanner" in layout, "AppLayout 没有挂自动弹出的横幅"

    settings = _src("pages", "SettingsPage.tsx")
    assert "SetupEntryRow" in settings, "设置页没有手动重新打开的入口"


def test_the_old_setup_url_still_works():
    """``/setup`` 这个地址在用户的书签里存了两天，合并之后不能变 404。

    2026-08-04 之后有人**被要求**去用这一页改退针方向，那个地址很可能就在他的
    浏览器历史里。一条打不开的书签看起来像「这个功能被删了」。
    """
    nav = _src("lib", "nav.ts")
    assert '"/setup": "/settings/setup"' in nav, (
        "LEGACY_REDIRECTS 里没有 /setup —— 旧书签会 404")


def test_the_banner_hides_itself_when_the_checklist_cannot_be_read():
    """读不出来时**不显示** —— 一条读不出来就常亮的红条，一周之内会被无视，
    而那正好毁掉它在真的缺数时的作用。"""
    banner = _src("components", "shell", "SetupBanner.tsx")
    assert "d.degraded" in banner
    assert "should_prompt" in banner


def test_the_page_reads_bounds_from_the_payload_not_from_a_local_copy():
    """一旦有人在 TSX 里抄一份区间，这一页就开始与后端漂移。

    判据用的是「有没有出现 instrument_profile 里那几个特征常量」——它们是真源里
    的边界值，出现在 TSX 里就说明有人抄了。
    """
    page = _src("pages", "SetupPage.tsx")
    for literal in ("1e13", "1.0e13", "400.0", "1e-12"):
        assert literal not in page, (
            f"SetupPage 里出现了字面量 {literal} —— 区间应当来自 "
            "GET /api/instrument-init 的 items[].input，不要在前端抄一份")
