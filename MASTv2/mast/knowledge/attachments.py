"""Supplementary information (SI) files attached to a paper.

A paper's methods increasingly live in its SI rather than its body — the exact
bias, the setpoint, the anneal schedule are in "Supplementary Note 3" while the
main text says "see Supplementary Information". So reading only the main PDF and
reporting "the paper does not state the setpoint" is often wrong about the paper.

Layout (next to the ingested paper, never mixed in with it)::

    <papers>/<slug>/
        source.pdf          ← the paper (written by ingest_pdf)
        fulltext.txt        ← its extracted text
        attachments/
            manifest.json   ← [{file, label, sha256, uploaded_at, n_chars, ocr_used}]
            si_1.pdf        ← the supplement as uploaded
            si_1.txt        ← its text, extracted ONCE at upload time

**An SI is not a paper.** It is deliberately not ingested, not embedded, not
promoted into the big library and never given a work_id: it has no abstract of
its own, would pollute semantic search with half-sentences, and would show up in
citation lists as a phantom publication. It belongs to exactly one paper and is
read only when that paper is read (``literature/deep_read``).

Text extraction — including the slow OCR fallback — happens **at upload**, not at
read time. A deep-read pass runs several papers in parallel under a wall-clock
budget; discovering there that a supplement is a scanned image and needs a
minute of OCR would blow that budget for every paper in the batch.

Everything here returns a dict and never raises: an attachment that fails to
attach must not take down the ingest or the request that carried it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "ATTACHMENTS_DIRNAME", "attachments_dir", "attach_si",
    "list_attachments", "attachment_text", "detach_si",
]

ATTACHMENTS_DIRNAME = "attachments"
_MANIFEST = "manifest.json"

#: Below this many non-whitespace characters we treat a PDF as having no usable
#: text layer and try OCR. Same threshold ``ingest_pdf`` uses for papers.
_MIN_TEXT_CHARS = 200

#: Guard against a pathological upload eating the disk. SI PDFs are routinely
#: bigger than papers (raw spectra, video stills), hence larger than ingest's.
_MAX_BYTES = 120 * 1024 * 1024


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(name: str) -> str:
    """A single, boring filename — never a path.

    Everything outside ``[A-Za-z0-9._-]`` becomes ``_`` and any directory part is
    dropped, so an upload named ``../../etc/passwd`` or ``C:\\evil.pdf`` can only
    ever land inside this paper's attachments directory.
    """
    base = Path(str(name or "")).name
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._-")
    if not base:
        base = "attachment"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base[:120]


def _resolve_slug(work_id_or_slug: str) -> str:
    """Accept a work_id (bare / URL / ``local:``) or an already-computed slug."""
    raw = (work_id_or_slug or "").strip()
    if not raw:
        return ""
    try:
        from mast.knowledge.ingest import slug_for_work_id
        return slug_for_work_id(raw)
    except Exception:  # pragma: no cover — ingest deps missing
        return re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("._-")[:120]


def _paper_dir(work_id_or_slug: str) -> Path | None:
    slug = _resolve_slug(work_id_or_slug)
    if not slug:
        return None
    try:
        from mast.knowledge.paths import papers_dir
        return papers_dir() / slug
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("papers_dir() unavailable: %s", exc)
        return None


def attachments_dir(work_id_or_slug: str, *, create: bool = False) -> Path | None:
    d = _paper_dir(work_id_or_slug)
    if d is None:
        return None
    a = d / ATTACHMENTS_DIRNAME
    if create:
        a.mkdir(parents=True, exist_ok=True)
    return a


def _read_manifest(adir: Path) -> list[dict]:
    f = adir / _MANIFEST
    try:
        if f.is_file():
            raw = json.loads(f.read_text(encoding="utf-8"))
            items = raw.get("attachments") if isinstance(raw, dict) else raw
            return [dict(x) for x in (items or []) if isinstance(x, dict)]
    except Exception as exc:  # corrupt manifest → behave like an empty one
        logger.warning("attachment manifest unreadable (%s): %s", adir, exc)
    return []


def _write_manifest(adir: Path, items: list[dict]) -> None:
    try:
        adir.mkdir(parents=True, exist_ok=True)
        tmp = adir / (_MANIFEST + ".tmp")
        tmp.write_text(json.dumps({"attachments": items}, ensure_ascii=False,
                                  indent=2), encoding="utf-8")
        os.replace(tmp, adir / _MANIFEST)
    except Exception as exc:  # pragma: no cover — best-effort
        logger.warning("attachment manifest write failed (%s): %s", adir, exc)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _extract_text(pdf_path: Path, *, ocr: bool,
                  extractor: Callable[[Any], Any] | None,
                  ocr_fn: Callable[[Any], str] | None) -> tuple[str, bool]:
    """``(text, ocr_used)`` for one supplement. Never raises."""
    text = ""
    try:
        if extractor is not None:
            got = extractor(pdf_path)
            text = getattr(got, "text", None) or (got if isinstance(got, str) else "")
        else:
            from mast.knowledge.ingest import extract_pdf
            text = extract_pdf(pdf_path).text or ""
    except Exception as exc:
        logger.info("SI text extraction failed (%s): %s", pdf_path.name, exc)
        text = ""

    if len("".join(text.split())) >= _MIN_TEXT_CHARS:
        return text, False

    if not ocr:
        return text, False
    try:
        fn = ocr_fn
        if fn is None:
            from mast.knowledge.ocr import ocr_available, ocr_pdf
            if not ocr_available():
                return text, False
            fn = ocr_pdf
        ocr_text = fn(pdf_path) or ""
    except Exception as exc:
        logger.info("SI OCR failed (%s): %s", pdf_path.name, exc)
        return text, False
    if len("".join(ocr_text.split())) > len("".join(text.split())):
        return ocr_text, True
    return text, False


def attach_si(work_id_or_slug: str, src_path: str | os.PathLike[str], *,
              label: str = "", ocr: bool = True,
              extractor: Callable[[Any], Any] | None = None,
              ocr_fn: Callable[[Any], str] | None = None,
              filename: str = "") -> dict[str, Any]:
    """Attach one SI PDF to an already-ingested paper.

    Args:
        work_id_or_slug: the paper. It must already be ingested — an attachment
            with no paper has nothing to be supplementary *to*, and creating the
            directory anyway would leave orphan SI nobody ever reads.
        src_path:   the PDF on disk; copied in (the source is left alone).
        label:      operator-facing description ("Supplementary Note 3").
        ocr:        allow the OCR fallback when the PDF has no text layer.
        extractor / ocr_fn: injection points for tests.
        filename:   preferred stored name; defaults to the source's.

    Returns ``{ok, slug, file, label, n_chars, ocr_used, duplicate, error}``.
    """
    out: dict[str, Any] = {"ok": False, "slug": "", "file": "", "label": label,
                           "n_chars": 0, "ocr_used": False, "duplicate": False,
                           "error": ""}
    src = Path(str(src_path or ""))
    if not src.is_file():
        out["error"] = f"找不到文件：{src}"
        return out
    try:
        if src.stat().st_size > _MAX_BYTES:
            out["error"] = f"文件过大（>{_MAX_BYTES // (1024 * 1024)} MB）"
            return out
    except OSError as exc:  # pragma: no cover — defensive
        out["error"] = f"无法读取文件：{exc}"
        return out

    pdir = _paper_dir(work_id_or_slug)
    if pdir is None:
        out["error"] = "无效的 work_id"
        return out
    slug = pdir.name
    out["slug"] = slug
    if not pdir.is_dir():
        out["error"] = ("这篇论文的正文还没有入库 —— 请先上传/摄取论文正文，"
                        "再挂它的补充材料。")
        return out

    adir = pdir / ATTACHMENTS_DIRNAME
    try:
        adir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        out["error"] = f"无法创建附件目录：{exc}"
        return out

    try:
        digest = _sha256_file(src)
    except Exception as exc:  # pragma: no cover — defensive
        out["error"] = f"无法校验文件：{exc}"
        return out

    items = _read_manifest(adir)
    for it in items:
        if it.get("sha256") == digest:
            # Same bytes already here: re-uploading is a no-op, not a second copy.
            out.update(ok=True, duplicate=True, file=str(it.get("file", "")),
                       label=str(it.get("label", "") or label),
                       n_chars=int(it.get("n_chars", 0) or 0),
                       ocr_used=bool(it.get("ocr_used", False)))
            return out

    name = _safe_name(filename or src.name)
    taken = {str(it.get("file", "")) for it in items}
    if name in taken or (adir / name).exists():
        stem, suffix = name[:-4], name[-4:]
        n = 2
        while f"{stem}-{n}{suffix}" in taken or (adir / f"{stem}-{n}{suffix}").exists():
            n += 1
        name = f"{stem}-{n}{suffix}"

    dest = adir / name
    try:
        tmp = adir / (name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    except Exception as exc:
        out["error"] = f"无法保存附件：{exc}"
        return out

    text, ocr_used = _extract_text(dest, ocr=ocr, extractor=extractor, ocr_fn=ocr_fn)
    try:
        (adir / (name[:-4] + ".txt")).write_text(text, encoding="utf-8")
    except Exception as exc:  # pragma: no cover — the PDF is what matters
        logger.info("SI text cache write failed (%s): %s", name, exc)

    items.append({"file": name, "label": (label or "").strip(), "sha256": digest,
                  "uploaded_at": _now_iso(), "n_chars": len(text),
                  "ocr_used": bool(ocr_used)})
    _write_manifest(adir, items)

    out.update(ok=True, file=name, n_chars=len(text), ocr_used=bool(ocr_used))
    return out


def list_attachments(work_id_or_slug: str) -> list[dict[str, Any]]:
    """Every supplement for a paper, manifest rows first.

    PDFs sitting in the directory without a manifest row are listed too, labelled
    ``(未登记)``: an operator who drops a file in by hand has still told us this
    is supplementary material, and silently ignoring it would make the copy the
    system reads differ from the copy they can see.
    """
    adir = attachments_dir(work_id_or_slug)
    if adir is None or not adir.is_dir():
        return []
    items = _read_manifest(adir)
    known = {str(it.get("file", "")) for it in items}
    out: list[dict[str, Any]] = []
    for it in items:
        f = str(it.get("file", ""))
        if not f or not (adir / f).is_file():
            continue  # manifest row whose file went away
        out.append({"file": f, "label": str(it.get("label", "") or ""),
                    "n_chars": int(it.get("n_chars", 0) or 0),
                    "ocr_used": bool(it.get("ocr_used", False)),
                    "uploaded_at": str(it.get("uploaded_at", "") or ""),
                    "registered": True})
    for p in sorted(adir.glob("*.pdf")):
        if p.name not in known:
            out.append({"file": p.name, "label": "(未登记)", "n_chars": 0,
                        "ocr_used": False, "uploaded_at": "", "registered": False})
    return out


def attachment_text(work_id_or_slug: str, file: str) -> str:
    """The cached text of one supplement, or ''.

    Falls back to a live parse for hand-dropped files that never went through
    :func:`attach_si` (so they have no ``.txt`` beside them). Never OCRs here —
    that cost belongs at upload time.
    """
    adir = attachments_dir(work_id_or_slug)
    if adir is None or not adir.is_dir():
        return ""
    name = _safe_name(file)
    cached = adir / (name[:-4] + ".txt")
    try:
        if cached.is_file():
            return cached.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover — defensive
        logger.info("SI text cache unreadable (%s): %s", name, exc)

    pdf = adir / name
    if not pdf.is_file():
        return ""
    text, _ = _extract_text(pdf, ocr=False, extractor=None, ocr_fn=None)
    return text


def detach_si(work_id_or_slug: str, file: str) -> dict[str, Any]:
    """Remove one supplement (PDF + cached text + manifest row)."""
    out: dict[str, Any] = {"ok": False, "error": ""}
    adir = attachments_dir(work_id_or_slug)
    if adir is None or not adir.is_dir():
        out["error"] = "没有附件目录"
        return out
    name = _safe_name(file)
    try:
        (adir / name).unlink(missing_ok=True)
        (adir / (name[:-4] + ".txt")).unlink(missing_ok=True)
    except Exception as exc:
        out["error"] = f"删除失败：{exc}"
        return out
    items = [it for it in _read_manifest(adir) if str(it.get("file", "")) != name]
    _write_manifest(adir, items)
    out["ok"] = True
    return out
