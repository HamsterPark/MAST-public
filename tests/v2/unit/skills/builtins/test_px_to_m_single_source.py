"""像素↔米的换算**只许有一份**。第四份出现时当场红。

## 这条闸门**盖不住什么** —— 先说这个

引发它的那次失败,**它不会开火**。

2026-08-10:这套换算一度有三份实现。第三份不在仓里 —— 它在 **scratchpad 的一次性
分析脚本**里,用来算团簇偏移的径向分解,当时未曾察觉这是第三次重新实现。
而 scratchpad 不在这条闸门的扫描范围内(也不该在:那里就是给一次性
东西用的)。

**所以不要因为这条闸门存在,就以为「内联坐标变换」这类事被挡住了。** 它挡的是
**进了仓的**第四份;挡不住任何人在分析脚本里再写一次。

真正对付那种情况的是**另一半**:把 `mast.io.mosaic.px_to_m` 做得一行能导入、
带例子、文档里搜得到 —— 追问「当时为什么内联」,答案是「**import 那份比自己写
四行麻烦**」。**移除诱因**那一半在 `mosaic.px_to_m` 的 docstring 里;
这一条只是拦截那一半。

## 频率数据:**同一个人、同一天、三次**

不是「理论上可能」。2026-08-10 一天之内,**同一个人在一次性分析里重新实现了三次
已经被钉好的东西**:

1. **内联坐标变换** —— 造出第三份 px→m,而且是唯一没被测过的那份;
2. **逐行去趋势** —— 用了一个已知会吃掉团簇的预处理(`_detrend` 那一族);
3. **y 符号** —— 把 `np.mgrid` 的行号当物理 y,而 `scan_dir='down'` 时行 0 是
   **最大** y。这个约定**就写在共享实现的 docstring 里**,他没用它。
   后果:反推的 `-M⁻¹` 与 `TiltCalibrate` 的 G 在 G22 上差 **199.9%**;
   把 y 翻正之后差 **0.1%**(0.9602 vs 0.9610),非对角符号也随之对上。

三次的形状完全相同:**「临时分析里重新实现了一个已经被钉好的东西」。**

⇒ **所以移除诱因(一行能导入 + 可抄的例子)比这条闸门重要。**
闸门只覆盖第 1 类,而且只在它进了仓的时候;
第 2、3 类**闸门永远看不见** —— 它们是"用错了已有的东西",不是"造了新的一份"。
唯一对三类都起作用的,是让正确的那份**比自己写更省事**。

（把闸门的射程写在闸门里,是因为「有闸门了」本身就会变成一种假安全 ——
本仓 KNOWN_ISSUES 里那一族的又一个形状。）

## 它盖得住什么

仓内任何**新的**「拿 scan_offset / scan_range 和像素索引做算术」的地方。
白名单只有一条(共享实现本身);两个调用方都已改为调用它。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

_PKG = Path(_MASTV2_ROOT) / "mast"

#: 允许自己做这套算术的地方 → 为什么。
#: **只有共享实现**。要加第二条,先问「为什么不能调 `mosaic.px_to_m`」。
_ALLOWED: dict[str, str] = {
    "io/mosaic.py": "共享实现本身(px_to_m + parse_xy_meta)",
}

#: 「把帧几何和像素索引混在一起算」的指纹:同一个函数体里既出现帧尺寸/中心,
#: 又出现「除以像素数」或「乘 0.5 的半宽」。宁可宽一点误报,
#: 也不要漏掉一份新的内联 —— 误报的代价是加一条白名单并写清理由。
_GEOM = re.compile(r"\b(w_m|h_m|scan_range|scan_offset|cx_m|cy_m)\b")
_PIXEL = re.compile(r"/\s*(nx|ny|n_x|n_y|nx_full|ny_full|cols|rows)\b")
_HALF = re.compile(r"\b(w_m|h_m)\s*\*\s*0\.5|\b0\.5\s*\*\s*(w_m|h_m)")


def _offenders() -> list[str]:
    out = []
    for path in sorted(_PKG.rglob("*.py")):
        rel = str(path.relative_to(_PKG)).replace("\\", "/")
        if rel in _ALLOWED:
            continue
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = "\n".join(lines[node.lineno - 1:node.end_lineno])
            if not _GEOM.search(body):
                continue
            if _PIXEL.search(body) and _HALF.search(body):
                out.append(f"{rel}::{node.name} (line {node.lineno})")
    return out


def test_the_sweep_itself_finds_the_shared_implementation():
    """自检:一条什么都梳不到的闸门,和「确实没问题」输出一模一样。

    把白名单清空,共享实现自己必须被逮到 —— 否则这把梳子的指纹是错的。
    """
    global _ALLOWED
    saved = _ALLOWED
    try:
        _ALLOWED = {}
        found = _offenders()
    finally:
        _ALLOWED = saved
    assert any("io/mosaic.py" in f for f in found), (
        f"梳子逮不到共享实现本身,说明指纹写错了(逮到的是 {found})")


def test_there_is_only_one_px_to_m_implementation():
    offenders = _offenders()
    assert not offenders, (
        "这些地方在自己做像素↔米的换算,而全仓只该有一份:\n  "
        + "\n  ".join(offenders)
        + "\n\n改法:`from mast.io.mosaic import parse_xy_meta, px_to_m`,"
        "一行就能用(那个 docstring 里有可直接抄的例子)。\n"
        "如果确实不能调它,把文件加进本文件的 _ALLOWED 并写清**为什么** ——"
        "「顺手写的」不是理由,这条闸门就是为那个理由存在的。")


def test_both_known_callers_go_through_the_shared_one():
    """两个调用方必须**真的**在调它 —— 光是自己不算术,不代表接上了。

    没有这一条,把两处的换算整个删掉(于是坐标全错)也能让上面那条全绿。
    """
    for rel in ("skills/builtins/flat_region.py",
                "skills/builtins/cluster_extract.py"):
        src = (_PKG / rel).read_text(encoding="utf-8")
        assert "from mast.io.mosaic import px_to_m" in src, (
            f"{rel} 没有在用共享的 px_to_m")


def test_the_shared_one_is_importable_in_one_line():
    """「移除诱因」那一半的可执行部分:它必须一行就能拿到,并且带着例子。

    当时内联的理由是「import 那份比自己写四行麻烦」。如果这个函数变得难导入,
    诱因就回来了 —— 而那不会有任何测试红,除非这一条在。
    """
    from mast.io.mosaic import parse_xy_meta, px_to_m
    assert callable(px_to_m) and callable(parse_xy_meta)
    doc = px_to_m.__doc__ or ""
    assert "from mast.io.mosaic import" in doc, "docstring 里没有可直接抄的导入例子"
    # 关键约定必须写在它自己身上,而不是散在调用方
    for word in ("左下", "y 翻转", "scan_angle"):
        assert word in doc, f"共享实现的 docstring 没写清约定:缺「{word}」"
