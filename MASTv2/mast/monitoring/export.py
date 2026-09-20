"""Export monitored current segments as a labelled training corpus.

Offline batch work, deliberately a CLI rather than an endpoint: it copies
gigabytes and must not run on a request thread.

    python -m mast.monitoring.export --pinned-only --weak-labels

Output lands in ``artifacts/current_corpus/<timestamp>/`` (gitignored) as one
``.npy`` plus one ``.json`` sidecar per segment, and a flat ``dataset.parquet``
/ ``dataset.jsonl`` for training.

**The weak-label alignment trap.** The vision buffer's ``tip_status_journal`` is
the obvious source of free labels: it already holds a per-frame good/degraded/bad
verdict. But its timestamps are ``time.monotonic_ns()`` — a counter with an
arbitrary origin that resets every process start. Offline, there is no way to
convert one into wall-clock, and doing it naively (treating it as epoch
nanoseconds, or offsetting by "now") produces labels that look plausible and are
attached to the wrong segments, which is worse than no labels at all.

So alignment goes through anchors that carry real wall-clock:

1. a ``frame_path`` in an event payload — those filenames embed an epoch
   millisecond by the repo's frame-writing convention, and the file's mtime is a
   fallback;
2. failing that, nothing. The segment gets no weak label.

Every weak label records which anchor produced it and the error bound implied,
and lands in its own column: a human verdict is never overwritten by a guess.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Epoch-millisecond stamp embedded in frame filenames by the repo's convention.
_STAMP_RE = re.compile(r"(?<!\d)(\d{13})(?!\d)")

#: How far from an anchor a journal row may sit and still be trusted. This
#: bounds clock drift and the risk that the process restarted in between (which
#: would reset the monotonic origin and invalidate the anchor entirely).
_ANCHOR_MAX_GAP_S = 300.0

#: Uncertainty contributed by the anchor itself. A frame filename carries an
#: epoch millisecond, but it is stamped when the file is written rather than
#: when the event fired, so a second is the honest bound. Note this does NOT
#: grow with distance from the anchor: monotonic and wall clocks advance at the
#: same rate, so a row 200 s away is placed just as precisely as one 2 s away —
#: distance only affects whether the anchor is still trustworthy at all, which
#: is what _ANCHOR_MAX_GAP_S is for.
_ANCHOR_PRECISION_S = 1.0


def _default_out_root() -> Path:
    from mast._runtime_paths import project_root
    return Path(project_root()) / "artifacts" / "current_corpus"


def _vision_wal_path() -> Path:
    from mast._runtime_paths import project_root
    return Path(project_root()) / "experiments" / "vision_buffer.wal.sqlite"


# ── weak labels ─────────────────────────────────────────────────────────────


def _monotonic_anchors(conn: sqlite3.Connection) -> list[tuple[int, float]]:
    """(t_mono_ns, wall_clock_s) pairs recovered from event payloads.

    Returns them sorted. An empty list means the journal cannot be placed on the
    wall clock at all, and the caller must skip weak labelling rather than guess.
    """
    anchors: list[tuple[int, float]] = []
    try:
        rows = conn.execute(
            "SELECT t_mono_ns, payload_json FROM event_journal "
            "WHERE payload_json IS NOT NULL"
        ).fetchall()
    except sqlite3.Error:
        return []
    for t_mono_ns, payload_json in rows:
        try:
            payload = json.loads(payload_json or "{}")
        except (TypeError, ValueError):
            continue
        for key in ("frame_path", "file_path", "path", "sxm_path"):
            raw = payload.get(key)
            if not raw:
                continue
            wall = _wall_clock_from_path(str(raw))
            if wall is not None:
                anchors.append((int(t_mono_ns), wall))
                break
    anchors.sort()
    return anchors


def _wall_clock_from_path(path: str) -> Optional[float]:
    """Epoch seconds from a frame filename stamp, else its mtime."""
    m = _STAMP_RE.search(Path(path).name)
    if m:
        ms = int(m.group(1))
        # Sanity: reject stamps outside a plausible range so a random 13-digit
        # run of characters cannot masquerade as a timestamp.
        if 1_000_000_000 < ms / 1000 < 4_000_000_000:
            return ms / 1000.0
    try:
        p = Path(path)
        if p.is_file():
            return p.stat().st_mtime
    except OSError:
        pass
    return None


def _mono_to_wall(t_mono_ns: int, anchors: list[tuple[int, float]]) -> tuple[Optional[float], str, float, float]:
    """Map a monotonic stamp onto the wall clock.

    Returns ``(wall, method, err_s, anchor_distance_s)``. ``err_s`` is the
    anchor's own precision, not the distance — see :data:`_ANCHOR_PRECISION_S`.
    """
    if not anchors:
        return None, "none", float("inf"), float("inf")
    best = min(anchors, key=lambda a: abs(a[0] - t_mono_ns))
    delta_s = (t_mono_ns - best[0]) / 1e9
    if abs(delta_s) > _ANCHOR_MAX_GAP_S:
        return None, "out_of_range", float("inf"), abs(delta_s)
    return best[1] + delta_s, "frame_anchor", _ANCHOR_PRECISION_S, abs(delta_s)


def collect_weak_labels(wal_path: Path | None = None) -> list[dict]:
    """Vision tip verdicts placed on the wall clock, where that is possible."""
    path = Path(wal_path) if wal_path else _vision_wal_path()
    if not path.is_file():
        logger.info("no vision buffer WAL at %s — skipping weak labels", path)
        return []
    out: list[dict] = []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        logger.warning("could not open the vision WAL read-only", exc_info=True)
        return []
    try:
        anchors = _monotonic_anchors(conn)
        if not anchors:
            logger.warning(
                "vision journal has no wall-clock anchor (its timestamps are "
                "monotonic and cannot be converted offline) — no weak labels")
            return []
        rows = conn.execute(
            "SELECT t_mono_ns, quality, confidence, scan_id FROM tip_status_journal"
        ).fetchall()
        for t_mono_ns, quality, confidence, scan_id in rows:
            wall, method, err, dist = _mono_to_wall(int(t_mono_ns), anchors)
            if wall is None:
                continue
            out.append({"ts": wall, "quality": quality,
                        "confidence": float(confidence or 0.0),
                        "scan_id": scan_id or "",
                        "align_method": method, "align_error_bound_s": err,
                        "anchor_distance_s": dist})
    except sqlite3.Error:
        logger.warning("could not read the vision journal", exc_info=True)
    finally:
        conn.close()
    out.sort(key=lambda r: r["ts"])
    return out


def _weak_for(segment: dict, weak: list[dict]) -> Optional[dict]:
    """The vision verdict overlapping a segment, if one is close enough.

    The window is the segment's own span plus the anchor precision. It is
    deliberately NOT widened by how far the verdict sat from its anchor: doing
    that would let a verdict minutes away get vacuumed onto a segment it has
    nothing to do with.
    """
    if not weak:
        return None
    t0 = float(segment.get("t_start") or 0.0)
    t1 = float(segment.get("t_end") or 0.0)
    mid = 0.5 * (t0 + t1)
    best = min(weak, key=lambda r: abs(r["ts"] - mid))
    span = max(1.0, t1 - t0)
    if abs(best["ts"] - mid) > 0.5 * span + best["align_error_bound_s"]:
        return None
    return best


# ── export ──────────────────────────────────────────────────────────────────


def export_corpus(*, out_dir: Path, since: float | None = None,
                  until: float | None = None, pinned_only: bool = True,
                  weak_labels: bool = False, fmt: str = "parquet",
                  store=None, wal_path: Path | None = None) -> dict:
    """Write the corpus. Returns a manifest dict (also saved as manifest.json)."""
    from mast.monitoring.store import get_store

    store = store or get_store()
    out_dir = Path(out_dir)
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    weak = collect_weak_labels(wal_path) if weak_labels else []
    listing = store.segments_query(since=since, until=until,
                                   pinned=True if pinned_only else None,
                                   limit=1_000_000)
    segments = listing.get("segments") or []

    rows: list[dict] = []
    copied = 0
    skipped_no_file = 0
    weak_hits = 0

    for seg in segments:
        seg_id = int(seg.get("id") or seg.get("seg_id") or 0)
        feats = store.feature_row(seg_id) or {}
        record: dict[str, Any] = {
            "seg_id": seg_id,
            "t_start": seg.get("t_start"),
            "t_end": seg.get("t_end"),
            "fs_hz": seg.get("fs_hz"),
            "n_samples": seg.get("n_samples"),
            "gap_s": seg.get("gap_s"),
            "discontinuity": bool(seg.get("discontinuity")),
            "channel_name": seg.get("channel_name") or "",
            "source": seg.get("source") or "",
            "pinned": bool(seg.get("pinned")),
            "pin_reason": seg.get("pin_reason") or "",
            "label_human": seg.get("label"),
            "label_note": seg.get("label_note") or "",
            "verdict": seg.get("alert_level") or (feats or {}).get("alert_level") or "",
            "npy": "",
        }
        for key, value in (feats or {}).items():
            if key in ("segment_id", "extra_json"):
                continue
            record.setdefault(key, value)

        if weak:
            hit = _weak_for(seg, weak)
            if hit:
                weak_hits += 1
                record.update({
                    "label_weak": hit["quality"],
                    "label_weak_confidence": hit["confidence"],
                    "label_weak_align_method": hit["align_method"],
                    "label_weak_align_error_s": hit["align_error_bound_s"],
                    "label_weak_anchor_distance_s": hit.get("anchor_distance_s"),
                })
        record.setdefault("label_weak", None)

        src = seg.get("npy_path")
        if src and Path(src).is_file():
            dst = seg_dir / f"{seg_id}.npy"
            try:
                shutil.copy2(src, dst)
                record["npy"] = f"segments/{seg_id}.npy"
                copied += 1
            except OSError:
                logger.warning("could not copy %s", src, exc_info=True)
        else:
            skipped_no_file += 1

        (seg_dir / f"{seg_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=_jsonable),
            encoding="utf-8")
        rows.append(record)

    written = _write_dataset(out_dir, rows, fmt)
    manifest = {
        "created_at": time.time(),
        "segments": len(rows),
        "waveforms_copied": copied,
        "waveforms_missing": skipped_no_file,
        "weak_labels_attached": weak_hits,
        "weak_label_rows_available": len(weak),
        "human_labels": sum(1 for r in rows if r.get("label_human")),
        "since": since, "until": until,
        "pinned_only": pinned_only, "weak_labels": weak_labels,
        "files": written,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _jsonable(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _write_dataset(out_dir: Path, rows: list[dict], fmt: str) -> list[str]:
    """Flat dataset file(s). JSONL always; parquet when asked and available."""
    written: list[str] = []
    jsonl = out_dir / "dataset.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=_jsonable) + "\n")
    written.append(jsonl.name)

    if fmt == "parquet" and rows:
        try:
            import pandas as pd
            path = out_dir / "dataset.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            written.append(path.name)
        except Exception:  # noqa: BLE001 — jsonl is already on disk
            logger.warning("parquet export unavailable; jsonl written instead",
                           exc_info=True)
    return written


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m mast.monitoring.export",
        description="导出隧道电流监控语料(段波形 + 特征 + 标签)")
    ap.add_argument("--out", help="输出目录(默认 artifacts/current_corpus/<时间戳>)")
    ap.add_argument("--since", type=float, help="起始 unix 时间戳")
    ap.add_argument("--until", type=float, help="结束 unix 时间戳")
    ap.add_argument("--all", action="store_true",
                    help="导出全部段,而不仅是钉住的")
    ap.add_argument("--weak-labels", action="store_true",
                    help="尝试用视觉针尖判定做弱标签(需要墙钟锚点,详见模块文档)")
    ap.add_argument("--format", choices=("parquet", "jsonl"), default="parquet")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    out = Path(args.out) if args.out else (
        _default_out_root() / time.strftime("%Y%m%d_%H%M%S"))
    manifest = export_corpus(out_dir=out, since=args.since, until=args.until,
                             pinned_only=not args.all,
                             weak_labels=args.weak_labels, fmt=args.format)
    print(f"导出完成:{out}")
    print(f"  段数 {manifest['segments']}  波形 {manifest['waveforms_copied']}"
          f"(缺失 {manifest['waveforms_missing']})")
    print(f"  人工标签 {manifest['human_labels']}"
          f"  弱标签 {manifest['weak_labels_attached']}")
    if args.weak_labels and manifest["weak_label_rows_available"] == 0:
        print("  注意:视觉判定无法定位到墙钟(其时间戳是单调钟),未附弱标签。")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["export_corpus", "collect_weak_labels", "main"]
