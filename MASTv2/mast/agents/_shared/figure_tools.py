"""list_figures — the read side of the ``figures`` artifact.

data_processing renders figures (plot_scan / plot_spectrum / mosaic_scans /
compose_montage) into ONE directory; paper_writing embeds them into the draft and
paper_review checks the draft's claims against them. Both of those need to know
WHAT IS THERE, and until now neither could: paper_writing's only file entry point
was ``embed_figure(scan_path=…)``, whose first act is ``Path(scan_path).is_file()``
— it demands an absolute path that paper_writing had no way to obtain. Its only
source was data_processing spelling the path out in a handoff message, so a
compaction, a fresh session, or an operator asking for a report on yesterday's run
broke the chain silently, and the model's recovery was to invent a filename.

plot_scan has in fact been telling the model "paper_writing 可用 list_figures()
找到它" since it was written (2026-07-27) — a tool name inside another tool's
output is an instruction to the model, and that one did not resolve. This module
is that promise being kept.

Lives in _shared/ because paper_writing and paper_review both need it and
``agents/A ↛ agents/B`` is a hard invariant — ``agents._shared.*`` is the only
legal sharing point.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
_MAX_LIMIT = 200


def _fmt_age(seconds: float) -> str:
    """Human-readable age, so this run's output is distinguishable from an
    earlier one at a glance."""
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


@tool("list_figures")
def list_figures(pattern: str = "", limit: int = 30) -> str:
    """List the figures data_processing has rendered, NEWEST FIRST.

    This is how you find out which figures exist — you are not expected to know
    or guess a filename. Returns each figure's absolute path (exactly what
    embed_figure wants), its size and how long ago it was written, so you can
    tell this run's output from an earlier one.

    `pattern`: optional case-insensitive substring filter (a sample name, a
    scan_id, a label given to plot_scan). Empty = everything.
    `limit`: how many to return (newest first).

    An empty result means no figure has been rendered yet — ask data_processing
    to render one; do NOT invent a filename.
    """
    from mast.agents._shared.data_paths import figures_dir

    try:
        d = figures_dir(create=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_figures: figures dir unavailable: %s", exc)
        return f"list_figures failed: figures directory unavailable ({exc})"

    if not d.is_dir():
        return (f"没有任何图（目录尚未创建：{d}）。请让 data_processing 先用 "
                "plot_scan / plot_spectrum 出图；不要自己编造文件名。")

    pat = (pattern or "").strip().lower()
    try:
        entries = [p for p in d.iterdir()
                   if p.is_file() and p.suffix.lower() in _IMAGE_EXT
                   and (not pat or pat in p.name.lower())]
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_figures: scan failed: %s", exc)
        return f"list_figures failed: could not read {d} ({exc})"

    if not entries:
        where = f"（目录 {d}）"
        if pat:
            return (f"没有匹配 {pattern!r} 的图{where}。用 list_figures() 不带 "
                    "pattern 看全部；若确实还没出图，请让 data_processing 先出图，"
                    "不要自己编造文件名。")
        return (f"还没有任何图{where}。请让 data_processing 先用 plot_scan / "
                "plot_spectrum 出图；不要自己编造文件名。")

    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    n_total = len(entries)
    try:
        n = max(1, min(int(limit), _MAX_LIMIT))
    except (TypeError, ValueError):
        n = 30
    shown, now = entries[:n], time.time()

    lines = [f"{n_total} 张图 in {d}"
             + (f"（匹配 {pattern!r}）" if pat else "")
             + (f"，以下为最新 {len(shown)} 张：" if len(shown) < n_total else "：")]
    for p in shown:
        st = p.stat()
        lines.append(f"  {p.name}  [{st.st_size / 1024:.0f} KB, "
                     f"{_fmt_age(now - st.st_mtime)}]\n    {p}")
    lines.append("把上面的绝对路径原样传给 embed_figure(scan_path=…)。")
    return "\n".join(lines)


FIGURE_TOOLS: list = [list_figures]

__all__ = ["list_figures", "FIGURE_TOOLS"]
