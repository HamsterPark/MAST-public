"""Single source of truth for which scan/spectrum INPUT formats the system loads,
plus a validator that checks a skill's advertised formats against it.

This had two halves. The first (parser convergence) is handled by making
:mod:`mast.io.nanonis_files` the single .sxm/.dat/.3ds byte parser (the duplicate
``mast.data.formats`` parser was removed 2026-07-20). The second is *truthful
metadata*: a skill's parameter description that says it accepts ``.sxm`` must not
be a lie — the canonical loader must actually read ``.sxm``. Before, a skill
could advertise a format the loader had never wired, and the mismatch only
surfaced at run time on the operator's real file.

:data:`CANONICAL_INPUT_EXT` is the union of :data:`SUPPORTED_IMAGE_EXT` and
:data:`SUPPORTED_SPECTRUM_EXT` (the extensions ``load_image_2d`` / ``load_spectrum``
handle). :func:`unsupported_advertised_formats` extracts the input formats a
skill advertises in its path-parameter descriptions and returns any that the
loader does NOT support. A registration-time test (tests/v2) asserts this is
empty for every skill, turning a lying advertisement into a build failure.
"""
from __future__ import annotations

import re

from mast.data.loaders import SUPPORTED_IMAGE_EXT, SUPPORTED_SPECTRUM_EXT

# Every extension the canonical loaders actually read.
CANONICAL_INPUT_EXT: tuple[str, ...] = tuple(
    sorted(set(SUPPORTED_IMAGE_EXT) | set(SUPPORTED_SPECTRUM_EXT))
)

# Extension tokens we recognise as a scan/spectrum INPUT format when they appear
# in a path-parameter description. This vocabulary deliberately INCLUDES formats
# the loader does not support (.ibw, .mtrx, .nid, …) so that advertising one is
# caught as a lie rather than silently ignored. Pure output/render formats
# (.png, .jpg, .pdf, .json, .svg) are NOT here — a skill may write those.
_SCAN_INPUT_VOCAB: frozenset[str] = frozenset({
    ".sxm", ".npy", ".npz", ".sm4", ".dat", ".txt", ".csv", ".asc", ".tsv",
    ".xyz", ".3ds", ".ibw", ".mtrx", ".nid", ".gwy", ".spm", ".mat", ".h5",
    ".hdf5", ".sm3", ".sm2",
})

# Only INPUT path params advertise loadable formats; save/output params advertise
# where a PRODUCT is written and must not be validated against the input loader.
_INPUT_PARAM_HINTS = ("path", "file", "image", "scan", "spectr")
_OUTPUT_PARAM_HINTS = ("save", "output", "dest", "dst", "out_")

_EXT_RE = re.compile(r"\.[A-Za-z0-9]{2,5}")


def _is_input_param(name: str) -> bool:
    n = (name or "").lower()
    if any(h in n for h in _OUTPUT_PARAM_HINTS):
        return False
    return any(h in n for h in _INPUT_PARAM_HINTS)


def advertised_input_extensions(meta) -> set[str]:
    """The scan/spectrum input formats a skill's metadata advertises.

    Scans the description of every INPUT path parameter for extension tokens that
    belong to :data:`_SCAN_INPUT_VOCAB`. Returns a set of lowercase extensions
    (e.g. ``{".sxm", ".npy"}``). Empty when the skill loads no files.
    """
    exts: set[str] = set()
    for spec in getattr(meta, "parameters", []) or []:
        if not _is_input_param(getattr(spec, "name", "")):
            continue
        desc = (getattr(spec, "description", "") or "").lower()
        for tok in _EXT_RE.findall(desc):
            if tok in _SCAN_INPUT_VOCAB:
                exts.add(tok)
    return exts


def unsupported_advertised_formats(meta) -> set[str]:
    """Input formats a skill advertises that the canonical loader does NOT read.

    Non-empty ⇒ the skill's metadata lies about what it can load. A
    registration-time test fails the build on any non-empty result.
    """
    return advertised_input_extensions(meta) - set(CANONICAL_INPUT_EXT)


__all__ = [
    "CANONICAL_INPUT_EXT",
    "advertised_input_extensions",
    "unsupported_advertised_formats",
]
