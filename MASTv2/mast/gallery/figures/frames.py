# -*- coding: utf-8 -*-
"""标记帧对比页。

每帧一张 PNG：左「原版（只减平面）」、右「逐行调平」（每行减中位数再减平面），各自 1–99% afmhot，
面板左上角写色标跨度 pm；像素按整数倍放大到面板宽 ≥ 384 px（上限 512）；底部一行
「开始时刻 · 偏压 · 电流 · 尺寸」，右边「目录/编号 · 评级 · 扫了 n/N 行 · 转角 · 反扫 · 重复保存」。

只取扫到的行（第一个到最后一个整行有限的行）；方向走 ``sxm_oriented_frames``（T4）。
文件名 = ``采集开始时刻_目录末段_编号.png``，按文件名排序就是时间顺序。
"""
from __future__ import annotations

import math

from mast.gallery import paths as _paths
from mast.gallery.figures import common as C

VER = 1
CAT = "frames"


def _is_frame(k: str, by: dict) -> bool:
    it = by.get(k)
    return (it.get("k") == "f") if it else k.lower().endswith(".sxm")


def marked_frame_ids(marks: dict, by: dict, max_series_frames: int = 100) -> list[str]:
    """单条标记的帧 + 帧数 ≤ ``max_series_frames`` 的系列成员，按开始时刻排。"""
    ids = {k for k in (marks.get("items") or {}) if _is_frame(k, by)}
    for sv in (marks.get("series") or {}).values():
        frs = [i for i in (sv.get("ids") or []) if _is_frame(i, by)]
        if frs and len(frs) <= int(max_series_frames):
            ids.update(frs)
    return sorted(ids, key=lambda i: ((by.get(i) or {}).get("t") or 0, i))


def make_frame_sheet(lay: _paths.Layout, it: dict, mark: dict | None, options: dict | None = None) -> str:
    from PIL import Image, ImageDraw

    z, bwd_only = C.load_z(it["p"])
    i0, i1, n_ok = C.valid_rows(z)
    rall = int(z.shape[0])
    zz = z[i0:i1] * 1e12
    raw = C.plane(zz)
    lev = C.plane(C.row_level(zz))

    nx = int(it.get("nx") or z.shape[1])
    ny = int(it.get("ny") or z.shape[0])
    w = float(it.get("w") or 0.0) or 1.0
    h = float(it.get("hn") or 0.0) or w
    wpan = min(nx * max(1, math.ceil(384 / nx)), 512)
    table = C.lut("afmhot")
    panels = []
    for arr, lab in ((raw, "原版（只减平面）"), (lev, "逐行调平")):
        img, lo, hi = C.to_img(arr, table)
        hp = max(1, round(img.height * wpan / img.width * (h / w) / (ny / nx)))
        rs = Image.Resampling.NEAREST if wpan >= img.width else Image.Resampling.LANCZOS
        panels.append((img.resize((wpan, hp), rs), lab, hi - lo))

    gap = 4
    W = wpan * 2 + gap
    Hp = panels[0][0].height
    canvas = Image.new("RGB", (W, Hp + 52), C.BG)
    d = ImageDraw.Draw(canvas)
    for k, (img, lab, span) in enumerate(panels):
        x = k * (wpan + gap)
        canvas.paste(img, (x, 0))
        C.tag(d, (x + 5, 5), "%s  %.0f pm" % (lab, span))

    t = it.get("t")
    left = "%s    %s    %s    %s" % (C.local(t, "%Y-%m-%d %H:%M:%S"), C.fmt_bias(it.get("b")),
                                     C.fmt_cur(it.get("sp")), C.fmt_size(it.get("w"), it.get("hn")))
    extra = ["目录%s/%s" % (C.dir_short(it["d"]), C.num4(it["fn"]))]
    rating = (mark or {}).get("r")
    if rating in C.RWORD:
        extra.append(C.RWORD[rating])
    if n_ok < rall:
        extra.append("扫了 %d/%d 行" % (n_ok, rall))
    if it.get("ang"):
        extra.append("转角 %g°" % float(it["ang"]))
    if bwd_only:
        extra.append("反扫")
    dup = it.get("dup")
    if dup:
        extra.append("重复保存=%s" % C.num4(str(dup).rsplit("/", 1)[-1]))
    used = C.caption(d, W, Hp, left, " · ".join(extra))
    canvas = canvas.crop((0, 0, W, Hp + used))

    base = "%s_%s_%s" % (C.local(t, "%Y%m%d-%H%M%S") if t else "unknown",
                         it["d"].rsplit("/", 1)[-1], C.num4(it["fn"]))
    name = base + ".png"
    C.write_bytes(lay, CAT, name, C.png_bytes(canvas))

    span_raw, span_lev = float(panels[0][2]), float(panels[1][2])
    summary = {"色标 原版 pm": round(span_raw, 1), "色标 调平 pm": round(span_lev, 1),
               "有效行": f"{n_ok}/{rall}"}
    if rating in C.RWORD:
        summary["评级"] = C.RWORD[rating]
    if bwd_only:
        summary["反扫"] = "是"
    if dup:
        summary["重复保存"] = C.num4(str(dup).rsplit("/", 1)[-1])
    detail = {"span_raw": span_raw, "span_lev": span_lev, "rows_ok": n_ok, "rall": rall,
              "i0": i0, "i1": i1, "bwd": bwd_only, "rating": rating, "dup": dup, "name": name}
    title = "%s %s · 原版 | 逐行调平" % (C.dir_title(it["d"]), C.num4(it["fn"]))
    return C.write_figure_json(lay, CAT, base, kind="frame_sheet", title=title, files=[name],
                               ids=[it["id"]], options=options, summary=summary, detail=detail,
                               maker_version=VER)
