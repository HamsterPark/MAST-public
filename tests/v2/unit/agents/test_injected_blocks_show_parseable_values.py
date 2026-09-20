"""注入给模型的数值必须是**能直接抄进技能参数、且真的能解析**的形式（2026-08-04）。

有量纲参数自 2026-08-04 起走字符串通道，其中「整个量程远小于 1」的那些
（扫描尺寸、坐标、设定点、增益…）**强制要求 SI 前缀** —— 对它们 `"5e-08"` 不是
不推荐，是 ``parse_si`` **直接拒绝**。

而两个注入块当时仍在用 ``%g`` / ``%e`` 印数：

* ``experiment_prefs.format_prefs_block`` —— 印 ``(= 5e-08 → size_m)``，
  而同一个块的抬头写着「**调用技能时用括号里的 SI 值**」。它在**直接指示**模型
  写一个必被拒的值。
* ``live_state_mw`` —— 印 ``Scan frame size: 5.000e-08 x ...``，并在
  MAGNITUDE CHECK 里写「typically 1e-9 to 1e-7 m」。

这两个块存在的唯一理由就是**让模型有个正确的数可以照抄**（不给数，它就得自己换算，
而那正是 2026-07-27 坐标事故的成因）。所以「印出来的数抄过去能不能用」不是风格
问题，是这两个块**有没有在起作用**的问题。

判据不靠正则看「像不像对的」—— 把块里印的数**逐个喂给解析器**。判断格式对不对的
唯一权威是解析器本身。

从仓库根跑::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_injected_blocks_show_parseable_values.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import re
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

from mast.core.si_quantity import (  # noqa: E402
    SIParseError,
    parse_quantity,
)

#: 指数写法 —— strict 参数会拒绝它
_EXPONENT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?[eE]-\d+(?![\w])")
#: 被引号括起来的值：块里给模型照抄的就是这些
_QUOTED = re.compile(r"'([^']{1,24})'")


def _quoted_values_parse_strictly(text: str) -> list[str]:
    """返回块里**引号中、看起来像数值、却过不了 strict 解析**的那些。"""
    bad = []
    for cand in _QUOTED.findall(text):
        s = cand.strip()
        if not re.match(r"^-?\d", s):        # 不是数值，跳过
            continue
        try:
            parse_quantity(s, strict=True, what="x")
        except SIParseError:
            bad.append(s)
    return bad


# ════════════════════════════════════════════════════════════════════════════
# 用户默认偏好块
# ════════════════════════════════════════════════════════════════════════════

def _prefs_block():
    from mast.agents._shared.experiment_prefs import format_prefs_block

    return format_prefs_block({
        "scan_size_nm": 50.0,
        "setpoint_pa": 100.0,
        "bias_v": -2.0,
    })


def test_the_prefs_block_is_not_empty_for_this_input() -> None:
    """先证明构造有效 —— 一个空块会让下面每条断言都白过。"""
    block = _prefs_block()
    assert block, "块是空的，测试数据没对上 _FIELD_SPEC 的键"
    assert "→" in block, "没渲染出 SI 换算,后面的断言无从谈起"


def test_the_prefs_block_shows_no_exponent_form() -> None:
    """它明说「用括号里的 SI 值」—— 那个值必须是能用的那个。"""
    block = _prefs_block()
    hits = _EXPONENT.findall(block)
    assert not hits, f"偏好块里仍在给模型看指数写法: {hits}"


def test_every_quoted_value_in_the_prefs_block_parses_strictly() -> None:
    bad = _quoted_values_parse_strictly(_prefs_block())
    assert not bad, f"这些值抄进 strict 参数会被拒: {bad}"


# ════════════════════════════════════════════════════════════════════════════
# 实时仪器状态块
# ════════════════════════════════════════════════════════════════════════════

def _live_block():
    """⚠️ 不能用 MagicMock —— 渲染函数对每个字段做 ``f"{v:.4e}"``，而 MagicMock
    的 ``__format__`` 直接抛 TypeError。用一个只有真值和 None 的简单对象。
    """
    from mast.agents._shared.live_state_mw import format_live_state_block

    class _State:
        def __getattr__(self, _name):     # 未显式设的字段一律 None
            return None

    state = _State()
    state.scan_width_m = 5e-8
    state.scan_height_m = 5e-8
    state.scan_center_x_m = 1.4e-6
    state.scan_center_y_m = -3e-7
    state.scan_angle_deg = 0.0
    return format_live_state_block(state)


def test_the_live_state_block_is_not_empty() -> None:
    assert "Scan frame size" in _live_block()


def test_the_live_state_block_shows_no_exponent_form() -> None:
    """模型照抄的就是这里印的数。印 5.000e-08，它就会写 5.000e-08。"""
    hits = _EXPONENT.findall(_live_block())
    assert not hits, f"实时状态块里仍在给模型看指数写法: {hits}"


def test_every_quoted_value_in_the_live_state_block_parses_strictly() -> None:
    bad = _quoted_values_parse_strictly(_live_block())
    assert not bad, f"这些值抄进 strict 参数会被拒: {bad}"


def test_the_magnitude_check_still_warns_about_the_bare_one() -> None:
    """改格式不能顺手弄丢那条警告 —— `1` = 1 米是这个块最初的存在理由。"""
    block = _live_block()
    assert "1 metre" in block or "1 米" in block


# ════════════════════════════════════════════════════════════════════════════
# 判据自检
# ════════════════════════════════════════════════════════════════════════════

def test_the_detectors_are_not_vacuous() -> None:
    """反向自检：判据必须能抓到它要抓的东西，也不能误伤正确写法。"""
    assert _EXPONENT.findall("size 5.000e-08 m")
    assert not _EXPONENT.findall("size '50n' m")
    assert _quoted_values_parse_strictly("用 '5e-08' 这个值") == ["5e-08"]
    assert _quoted_values_parse_strictly("用 '50n' 这个值") == []
    # 非数值的引号内容不该被当成候选
    assert _quoted_values_parse_strictly("模式 'approach'") == []
