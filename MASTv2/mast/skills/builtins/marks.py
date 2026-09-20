"""Scan markers — the agent can finally point at where it did something.

Nanonis draws points and lines onto the scan frame (``Marks_*``, 10 API methods).
MAST had a skill for none of them, which meant the agent could take an STS curve
at (x, y) and then had **no way to mark on the image where it took it**. The
operator got a spectrum and a scan and had to correlate them by hand.

These are shaped as TASKS, not as a 1:1 wrapper of the API. Nanonis exposes
``PointDraw``/``PointsDraw``/``PointsErase``/``PointsGet``/``PointsVisibleSet`` and
the same five for lines — ten calls for what is really three actions: **draw**,
**erase**, **list**. A skill list is a menu the model reads on every turn; ten
near-identical entries there cost routing accuracy and buy nothing.

Coordinates are in METRES, in the scanner frame — the same coordinates
``MoveToXY`` / ``BiasSpectr`` use, so an STS point can be marked with the very
coordinates it was taken at.
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

# Nanonis marker colour is a packed integer. These are the useful ones by name so
# the model does not have to guess a magic number.
_COLORS = {
    "red": 0xFF0000, "green": 0x00FF00, "blue": 0x0000FF,
    "yellow": 0xFFFF00, "cyan": 0x00FFFF, "magenta": 0xFF00FF,
    "white": 0xFFFFFF, "black": 0x000000, "orange": 0xFF8000,
}


def _values(record) -> list:
    rv = getattr(record, "return_value", None)
    if isinstance(rv, (list, tuple)) and len(rv) > 2 and isinstance(rv[2], (list, tuple)):
        return list(rv[2])
    return []


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _color(name: str) -> int:
    key = str(name or "red").strip().lower()
    if key in _COLORS:
        return _COLORS[key]
    try:                                   # allow a raw 0xRRGGBB too
        return int(str(name), 0)
    except (TypeError, ValueError):
        return _COLORS["red"]


class DrawScanMarker(BaseSkill):
    """Mark a point or a line on the scan image."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="DrawScanMarker",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在 Nanonis 扫描帧上画一个**点**或一条**线**标记。用它记录你**在哪里**做过什么 —— 标出每一个 STS 位置、"
                "你选中的平坦区域、你发现的一个缺陷、你取剖面所沿的那条线。坐标是扫描器坐标系下的米（和 MoveToXY 及各谱学技能用的是同一套）"
                "，所以你可以用测量时的精确坐标去标一条谱。\n"
                "\n"
                "画标记只碰显示 —— 它什么都不移动。"
            ),
            parameters=[
                ParameterSpec(
                    name="kind", type="str",
                    description="'point' 或 'line'",
                    required=True, allowed_values=["point", "line"],
                ),
                ParameterSpec(
                    name="x_m", type="float",
                    description="点的 X，或线的**起点** X（m，扫描器坐标系）",
                    unit="m", required=True,
                ),
                ParameterSpec(
                    name="y_m", type="float",
                    description="点的 Y，或线的**起点** Y（m，扫描器坐标系）",
                    unit="m", required=True,
                ),
                ParameterSpec(
                    name="x2_m", type="float",
                    description="线的**终点** X（m）。kind='line' 时必填。",
                    unit="m", required=False, default=None,
                ),
                ParameterSpec(
                    name="y2_m", type="float",
                    description="线的**终点** Y（m）。kind='line' 时必填。",
                    unit="m", required=False, default=None,
                ),
                ParameterSpec(
                    name="text", type="str",
                    description="显示在标记旁边的标签（仅点标记有）",
                    required=False, default="",
                ),
                ParameterSpec(
                    name="color", type="str",
                    description=(
                        "red/green/blue/yellow/cyan/magenta/white/black/orange，"
                        "或 0xRRGGBB"
                    ),
                    required=False, default="red",
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["marks", "annotation", "scan"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        kind = str(params["kind"]).strip().lower()
        x, y = float(params["x_m"]), float(params["y_m"])
        col = _color(params.get("color", "red"))

        if kind == "line":
            x2, y2 = params.get("x2_m"), params.get("y2_m")
            if x2 is None or y2 is None:
                return _fail("DrawScanMarker",
                             "kind='line' 需要 x2_m 与 y2_m（线的终点）", [])
            rec = context.safe_call("Marks_LineDraw", x, y,
                                    float(x2), float(y2), col)
            if rec.error:
                return _fail("DrawScanMarker", rec.error, [rec])
            return SkillResult(
                skill_name="DrawScanMarker", success=True,
                data={"kind": "line", "start": [x, y], "end": [float(x2), float(y2)]},
                nanonis_calls=[rec],
            )

        if kind != "point":
            return _fail("DrawScanMarker", f"kind 必须是 'point' 或 'line'，收到 {kind!r}", [])
        rec = context.safe_call("Marks_PointDraw", x, y,
                                str(params.get("text", "") or ""), col)
        if rec.error:
            return _fail("DrawScanMarker", rec.error, [rec])
        return SkillResult(
            skill_name="DrawScanMarker", success=True,
            data={"kind": "point", "x_m": x, "y_m": y,
                  "text": params.get("text", "")},
            nanonis_calls=[rec],
        )


class ListScanMarkers(BaseSkill):
    """List the markers currently on the scan frame."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ListScanMarkers",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "列出 Nanonis 扫描帧上的点标记与线标记及其坐标。挑下一个位置之前，用它回想一下你已经在哪儿测过了。"
            ),
            parameters=[],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["marks", "annotation", "scan", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        pts = context.safe_call("Marks_PointsGet")
        lns = context.safe_call("Marks_LinesGet")
        calls = [pts, lns]
        if pts.error and lns.error:
            return _fail("ListScanMarkers", pts.error or lns.error, calls)
        return SkillResult(
            skill_name="ListScanMarkers", success=True,
            data={"points": _values(pts) if not pts.error else [],
                  "lines": _values(lns) if not lns.error else []},
            nanonis_calls=calls,
        )


class EraseScanMarkers(BaseSkill):
    """Erase, or hide, markers on the scan frame."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="EraseScanMarkers",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "擦掉一个标记（或者只隐藏、不删除）。index=-1 会擦掉该类型的**全部**标记。擦标记只碰显示。"
            ),
            parameters=[
                ParameterSpec(
                    name="kind", type="str",
                    description="'point' 或 'line'",
                    required=True, allowed_values=["point", "line"],
                ),
                ParameterSpec(
                    name="index", type="int",
                    description="标记序号；-1 = 该类型的全部",
                    required=True, min_value=-1, max_value=10000,
                ),
                ParameterSpec(
                    name="hide_only", type="bool",
                    description="True = 隐藏但保留；False = 擦掉它",
                    required=False, default=False,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["marks", "annotation", "scan"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        kind = str(params["kind"]).strip().lower()
        idx = int(params["index"])
        hide = bool(params.get("hide_only", False))
        if kind not in ("point", "line"):
            return _fail("EraseScanMarkers",
                         f"kind 必须是 'point' 或 'line'，收到 {kind!r}", [])
        # Literal verbs — every safety tool MAST has (the post-abort allow-list
        # guard, the API coverage census) reads them back by grepping
        # `safe_call("…")`. A verb behind a variable is invisible to all of them.
        if hide and kind == "point":
            rec = context.safe_call("Marks_PointsVisibleSet", idx, 0)   # 0 = hide
        elif hide:
            rec = context.safe_call("Marks_LinesVisibleSet", idx, 0)
        elif kind == "point":
            rec = context.safe_call("Marks_PointsErase", idx)
        else:
            rec = context.safe_call("Marks_LinesErase", idx)
        if rec.error:
            return _fail("EraseScanMarkers", rec.error, [rec])
        return SkillResult(
            skill_name="EraseScanMarkers", success=True,
            data={"kind": kind, "index": idx,
                  "action": "hidden" if hide else "erased"},
            nanonis_calls=[rec],
        )


__all__ = ["DrawScanMarker", "ListScanMarkers", "EraseScanMarkers"]
