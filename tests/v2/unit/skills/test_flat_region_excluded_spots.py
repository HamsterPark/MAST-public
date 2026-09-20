"""`exclude_used_spots` 解析不了时必须**拒绝**，不能静默丢弃（2026-08-04）。

这个参数的意思是「别再选这几个点」——它们是刚才已经用过、或者已经弄坏了的位置。
原来的实现是::

    try:
        x_s, y_s = chunk.split(",")
        out.append((float(x_s), float(y_s)))
    except ValueError:
        continue          # ← 看不懂就当没说过

于是「排除这几个点」会悄悄变成「一个都不排除」，技能**高高兴兴地把针尖送回刚才
那个坏点**，而调用方以为回避生效了。不报错、不降级、没有任何痕迹。

两件事一起改：

1. **接受 SI 前缀。** 有量纲参数 2026-08-04 起整体走字符串通道，模型到处被教
   「写 '100n' 不要写 1e-7」—— 它多半会把这个习惯带进这个自由格式串。而这里用的
   是裸 ``float()``，``'1n'`` 直接 ValueError → 走进上面那个静默分支。
   **新规矩让这个旧洞从「偶尔踩到」变成「大概率踩到」。**
2. **解析失败要说出来。** 这是 既有教训 的又一例：一个
   `except: continue` 把「防止重选坏点」这条保护变成了空的。

从仓库根跑::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/test_flat_region_excluded_spots.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

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
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.skills.builtins.flat_region import FindFlatRegion  # noqa: E402

_parse = FindFlatRegion._parse_excluded


# ════════════════════════════════════════════════════════════════════════════
# 前缀形式必须能用 —— 模型现在到处被教这么写
# ════════════════════════════════════════════════════════════════════════════

def test_si_prefixed_coordinates_parse() -> None:
    pts, bad = _parse("100n,-50n;1.2u,0")
    assert bad == []
    assert pts[0][0] == pytest.approx(1e-7)
    assert pts[0][1] == pytest.approx(-5e-8)
    assert pts[1][0] == pytest.approx(1.2e-6)
    assert pts[1][1] == pytest.approx(0.0)


def test_plain_and_exponent_forms_still_parse() -> None:
    """非 strict —— 旧的写法不能因为这次改动失效。"""
    pts, bad = _parse("1e-9,2e-9;0.0000000030,-1e-9")
    assert bad == []
    assert pts[0] == pytest.approx((1e-9, 2e-9))
    assert pts[1][0] == pytest.approx(3e-9)


def test_mixed_forms_in_one_string() -> None:
    pts, bad = _parse("100n,2e-9;0,-0.5u")
    assert bad == []
    assert pts[0] == pytest.approx((1e-7, 2e-9))
    assert pts[1] == pytest.approx((0.0, -5e-7))


# ════════════════════════════════════════════════════════════════════════════
# 看不懂的必须报出来 —— 本文件的核心
# ════════════════════════════════════════════════════════════════════════════

def test_an_unparseable_chunk_is_reported_not_dropped() -> None:
    pts, bad = _parse("100n,-50n;上次那个点;3u,1u")
    assert bad == ["上次那个点"], "看不懂的块被静默吞掉了"
    assert len(pts) == 2, "能解析的仍要解析出来"


def test_a_chunk_with_the_wrong_arity_is_reported() -> None:
    """`1n,2n,3n` 是三个数 —— 不知道哪两个是坐标，不能猜。"""
    _, bad = _parse("1n,2n,3n")
    assert bad == ["1n,2n,3n"]


def test_empty_input_is_not_an_error() -> None:
    """没填 = 没有要排除的点，这是合法的常态。"""
    for empty in ("", None, "   ", ";;"):
        pts, bad = _parse(empty)
        assert pts == [] and bad == []


# ════════════════════════════════════════════════════════════════════════════
# 技能层：有看不懂的就整体失败
# ════════════════════════════════════════════════════════════════════════════

def test_the_skill_refuses_rather_than_picking_a_spot_it_cannot_avoid() -> None:
    """**这条是这次改动的全部意义。**

    与其在「不知道该躲开哪里」的状态下挑一个点回去，不如让这一步失败。挑出来的
    点看起来和正常结果一模一样 —— 调用方拿不到任何线索说明回避根本没发生。
    """
    skill = FindFlatRegion()
    res = skill.execute(
        MagicMock(),
        {"scan_path": "nonexistent.sxm", "exclude_used_spots": "看不懂的东西"},
    )
    assert not res.success
    assert "exclude_used_spots" in (res.error or ""), res.error
    assert "看不懂的东西" in (res.error or "")


def test_the_refusal_shows_the_accepted_form() -> None:
    """拒绝要能自解 —— 不指路的拒绝会招来同一个错误的重试。"""
    res = FindFlatRegion().execute(
        MagicMock(), {"scan_path": "x.sxm", "exclude_used_spots": "???"})
    assert "100n" in (res.error or ""), res.error
