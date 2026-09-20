"""The scan map's colour table exists twice — keep the two copies honest.

``mast/io/exp_map.py`` KIND_STYLE and ``ScanMapCanvas.tsx`` KIND_COLOR/KIND_LABEL
are two hand-maintained copies of one table: Konva paints onto a canvas and
cannot read the backend's palette. The failure mode when they drift is not a
crash — an unknown kind falls back to grey with an empty label — so a newly added
marker kind shows up as an anonymous grey dot with no legend entry, and nobody
notices until someone asks what the grey dots are.

That already happened once with ``coarse_move``/``approach``/``crash``. This test
is the guard, and it reads the TSX as text on purpose: there is no build step here
that could evaluate it, and the point is precisely to compare the two SOURCES.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mast.io.exp_map import KIND_STYLE

#: Kinds that exist only in the analysis overlay, never as a stored marker kind,
#: so the frontend legitimately carries them while the backend table does not.
FRONTEND_ONLY_KINDS = {"manual_avoid"}


def _canvas_source() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        cand = p / "frontend" / "src" / "components" / "vision" / "ScanMapCanvas.tsx"
        if cand.is_file():
            return cand.read_text(encoding="utf-8")
        p = p.parent
    pytest.skip("frontend/ not present in this checkout")


def _object_literal(src: str, name: str) -> dict[str, str]:
    body = src.split(f"const {name} = {{", 1)[1].split("};", 1)[0]
    return dict(re.findall(r"(\w+):\s*\"([^\"]+)\"", body))


def test_every_backend_kind_has_a_frontend_colour():
    colours = _object_literal(_canvas_source(), "KIND_COLOR")
    missing = sorted(set(KIND_STYLE) - set(colours))
    assert not missing, (
        f"kinds present in exp_map.KIND_STYLE but not in ScanMapCanvas.KIND_COLOR: "
        f"{missing} — they would render as anonymous grey dots"
    )


def test_colours_match_exactly():
    colours = _object_literal(_canvas_source(), "KIND_COLOR")
    drift = {
        kind: (style[0], colours[kind])
        for kind, style in KIND_STYLE.items()
        if kind in colours and colours[kind].lower() != style[0].lower()
    }
    assert not drift, f"colour drift (backend, frontend): {drift}"


def test_every_backend_kind_has_a_frontend_label():
    labels = _object_literal(_canvas_source(), "KIND_LABEL")
    missing = sorted(set(KIND_STYLE) - set(labels))
    assert not missing, f"kinds with no Chinese label in the frontend: {missing}"


def test_frontend_has_no_kinds_the_backend_never_writes():
    """A colour for a kind nothing produces is dead code — or a typo in one that
    is produced, which is worse: the real kind silently falls back to grey."""
    colours = _object_literal(_canvas_source(), "KIND_COLOR")
    extra = sorted(set(colours) - set(KIND_STYLE) - FRONTEND_ONLY_KINDS)
    assert not extra, (
        f"frontend colours for kinds absent from exp_map.KIND_STYLE: {extra} — "
        f"either dead entries or a misspelling of a real kind"
    )


def test_the_damage_kinds_the_analysis_relies_on_are_all_styled():
    """The avoidance model turns these into keep-out zones, and the operator has
    to be able to tell them apart on the map to sanity-check the analysis."""
    from mast.io.map_analysis import DAMAGE_KINDS

    colours = _object_literal(_canvas_source(), "KIND_COLOR")
    for kind in DAMAGE_KINDS:
        assert kind in KIND_STYLE, f"damage kind {kind!r} has no backend style"
        assert kind in colours, f"damage kind {kind!r} has no frontend colour"


def test_legend_covers_the_new_decision_relevant_kinds():
    """A kind with a colour but no legend row is a colour nobody can read."""
    src = _canvas_source()
    legend = src.split("export function ScanMapLegend()", 1)[1]
    for label in ("进针", "撞针", "粗动换区"):
        assert label in legend, f"legend missing an entry for {label}"
