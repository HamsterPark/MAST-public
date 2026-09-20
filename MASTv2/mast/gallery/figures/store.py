# -*- coding: utf-8 -*-
"""出图产物：列出、取文件、480 px 预览（设计文档 D16 / D21，T32）。

* :func:`list_figures` **纯读**：``figures/`` 不存在就回空，不建目录。条目来自各类别目录里的
  ``<base>.figure.json``；文件取 figure.json 里记的 ``files``（第一个是主图），再附上 figure.json 本身。
* :func:`figure_file` / :func:`preview_file`：路径 resolve 后必须落在 ``figures/`` 之内。
* :func:`preview_file` 是唯一会写的地方：源文件存在时才在 ``figures/.previews/`` 里生成 480 px 宽的
  JPEG（源比缓存新就重做）。**源文件不存在时绝不 mkdir、绝不写** —— ``test_boot_smoke`` 用
  ``path=nope`` 请求它，不能因此在冷机器上留下一个目录。
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from mast.gallery import paths as _paths
from mast.gallery.figures import common as C

PREVIEW_W = 480
_IMAGE_EXT = (".png", ".jpg", ".jpeg")
FILE_URL = "/api/gallery/figures/file/"
PREVIEW_URL = "/api/gallery/figures/preview/"


def _file_entry(cat: str, p: Path) -> dict | None:
    try:
        st = p.stat()
    except OSError:
        return None
    rel = quote(f"{cat}/{p.name}", safe="/")
    v = int(st.st_mtime)
    ext = p.suffix.lower().lstrip(".")
    if p.name.endswith(C.FIGURE_JSON):
        ext = "json"
    return {
        "name": p.name,
        "url": f"{FILE_URL}{rel}?v={v}",
        "preview_url": f"{PREVIEW_URL}{rel}?v={v}" if p.suffix.lower() in _IMAGE_EXT else None,
        "ext": ext,
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _main_first(names: list[str], base: str) -> list[str]:
    """主图排第一：``<base>.png``，否则第一张 png，再否则第一张 jpg；其余保持 figure.json 里的顺序。"""
    main = base + ".png"
    if main not in names:
        main = (next((n for n in names if n.lower().endswith(".png")), None)
                or next((n for n in names if n.lower().endswith((".jpg", ".jpeg"))), None))
    return names if main is None else [main] + [n for n in names if n != main]


def list_figures(lay: _paths.Layout | None = None) -> dict:
    """键正好是 ``GalleryFiguresList``。只读。"""
    lay = lay or _paths.layout()
    root = C.figures_dir(lay)
    categories = []
    for cat, title in C.CATEGORIES:
        d = root / cat
        figs = []
        if d.is_dir():
            try:
                names = sorted(os.listdir(d))
            except OSError:
                names = []
            present = {n for n in names if _paths.TMP_MARK not in n}
            for mj in (n for n in names if n.endswith(C.FIGURE_JSON)):
                meta = _paths.read_json(d / mj, None)
                if not isinstance(meta, dict):
                    continue
                base = str(meta.get("base") or mj[: -len(C.FIGURE_JSON)])
                wanted = [str(n) for n in (meta.get("files") or []) if str(n) in present]
                if not meta.get("files"):
                    wanted = [n for n in sorted(present) if n.startswith(base) and not n.endswith(C.FIGURE_JSON)]
                wanted = _main_first(wanted, base)
                files = [e for e in (_file_entry(cat, d / n) for n in wanted + [mj]) if e]
                figs.append({
                    "key": f"{cat}/{base}",
                    "category": cat,
                    "kind": meta.get("kind") or {"frames": "frame_sheet", "grids": "grid_sheets",
                                                 "sts_lines": "sts_lines", "sts_stitch": "sts_stitch",
                                                 "series": "series_stack"}[cat],
                    "title": str(meta.get("title") or base),
                    "base": base,
                    "files": files,
                    "created": str(meta.get("created") or ""),
                    "ids": [str(i) for i in (meta.get("ids") or [])],
                    "series": [str(s) for s in (meta.get("series") or [])],
                    "options": meta.get("options") if isinstance(meta.get("options"), dict) else {},
                    "summary": meta.get("summary") if isinstance(meta.get("summary"), dict) else {},
                })
        figs.sort(key=lambda f: f["base"])
        categories.append({"key": cat, "title": title, "figures": figs})
    return {"categories": categories, "degraded": False, "detail": None}


def _inside(root: Path, path: str) -> Path | None:
    try:
        base = root.resolve()
        target = (base / str(path)).resolve()
    except (OSError, ValueError):
        return None
    if not target.is_relative_to(base) or target == base:
        return None
    return target


def figure_file(path: str, lay: _paths.Layout | None = None) -> Path | None:
    """``figures/`` 里的一个产物文件；不在里面、不是文件、是临时文件或预览缓存 ⇒ None。只读。"""
    lay = lay or _paths.layout()
    root = C.figures_dir(lay)
    target = _inside(root, path)
    if target is None or not target.is_file() or _paths.TMP_MARK in target.name:
        return None
    rel = target.relative_to(root.resolve())
    if rel.parts and rel.parts[0] == C.PREVIEW_DIR:
        return None
    return target


def preview_file(path: str, lay: _paths.Layout | None = None) -> Path | None:
    """一张图片产物的 480 px JPEG 预览。源不存在 ⇒ None，且什么都不写。"""
    lay = lay or _paths.layout()
    src = figure_file(path, lay)
    if src is None or src.suffix.lower() not in _IMAGE_EXT:
        return None
    root = C.figures_dir(lay).resolve()
    rel = src.relative_to(root)
    prev = root / C.PREVIEW_DIR / rel.parent / (rel.name + ".jpg")
    try:
        if prev.is_file() and prev.stat().st_mtime_ns >= src.stat().st_mtime_ns:
            return prev
    except OSError:
        pass
    from PIL import Image

    with Image.open(src) as im:
        img = im.convert("RGB")
        if img.width > PREVIEW_W:
            img = img.resize((PREVIEW_W, max(1, round(img.height * PREVIEW_W / img.width))),
                             Image.Resampling.LANCZOS)
    _paths.atomic_write_bytes(prev, C.jpeg_bytes(img, quality=85))
    return prev
