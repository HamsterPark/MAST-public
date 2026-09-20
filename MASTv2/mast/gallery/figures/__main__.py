# -*- coding: utf-8 -*-
"""``python -m mast.gallery.figures``：同步出图（运维与对账用；尊重 ``MAST_GALLERY_DIR``）。

    python -m mast.gallery.figures run marked_frames
    python -m mast.gallery.figures run sts_lines --series S1 S2 S3 --options '{"station_marks": {"5": "V"}}'
    python -m mast.gallery.figures plan --series S1 S2 S3
    python -m mast.gallery.figures list
"""
from __future__ import annotations

import argparse
import json
import sys


def _print(line: str) -> None:
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    from mast.gallery.figures import service

    ap = argparse.ArgumentParser(prog="python -m mast.gallery.figures", description="数据图库 → 出图")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="出图（同步）")
    r.add_argument("kind", choices=service.KINDS)
    r.add_argument("--ids", nargs="*", default=[])
    r.add_argument("--series", nargs="*", default=[])
    r.add_argument("--options", default="{}", help="JSON 对象")
    p = sub.add_parser("plan", help="拉线谱站位推断预演")
    p.add_argument("--series", nargs="+", required=True)
    p.add_argument("--options", default="{}")
    sub.add_parser("list", help="列出已有产物")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        res = service.run_job(a.kind, a.ids, a.series, json.loads(a.options),
                              progress=service.JobProgress(echo=_print))
        _print("阶段 %s · 出了 %d 项 · 失败 %d 项" % (res["phase"], len(res["made"]), len(res["errors"])))
        if res.get("detail"):
            _print("  " + str(res["detail"]))
        return 0 if res["phase"] in ("done", "cancelled") else 1
    if a.cmd == "plan":
        from mast.gallery.figures import lines

        plan = lines.plan_from_store(a.series, json.loads(a.options))
        _print(json.dumps(plan, ensure_ascii=False, indent=1))
        return 0 if plan.get("ok") else 1
    from mast.gallery.figures import store

    for cat in store.list_figures()["categories"]:
        _print("%s（%d）" % (cat["title"], len(cat["figures"])))
        for f in cat["figures"]:
            _print("  %s  %s  %s" % (f["created"], f["base"], f["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
