"""把脚本产出的东西收回来 —— 图进图库，数值原样回传。

两条硬约定
==========

1. **图必须拷进 ``figures_dir()``。** ``list_figures()``
   （``_shared/figure_tools.py:64-80``）**只读那一个目录** —— 留在会话 ``out/``
   里的图，对下游写报告的 agent 完全不存在。这不是「最好也拷一份」，是「不拷就
   等于没画」。
2. **``result.json`` 逐字节回传。** 这是让数值不经过模型 token 流的那条路：
   模型是**读**到 ``3.2e-12``，不是凭记忆重建它。本仓记着一次
   ``3e-12`` 被念成「3 米」的事故，而重建出来的数看上去总是合理的。

上限全是**说清楚**而不是静默截断
==============================
超了就点名是哪些文件、并告诉下一步怎么做（「完整输出在 logs/stepNN.out，用
py_run 读它的后 200 行」）。静默截断会让人以为那就是全部。
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_NEW_BYTES = 500 * 1024 ** 2       # 单次运行新增产物总量
MAX_FIGURE_BYTES = 30 * 1024 ** 2     # 单张图
RESULT_JSON_CHARS = 4000

FIGURE_EXT = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}


@dataclass
class HarvestResult:
    new_files: list[str] = field(default_factory=list)     # 相对 out/ 的路径
    figures: list[str] = field(default_factory=list)       # 拷进图库后的绝对路径
    result_json: str = ""                                  # 原样文本（可能截断）
    result_truncated: bool = False
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_products(self) -> bool:
        return bool(self.new_files or self.result_json)


def snapshot(session) -> dict[str, tuple[int, int]]:
    """跑之前给 ``out/`` 拍个照。``{相对路径: (size, mtime_ns)}``。

    用 (size, mtime_ns) 而不只是路径：脚本**覆盖**一张已有的图也算新产物 ——
    只看「有没有新文件名」会漏掉「重画了同一张」这个最常见的迭代动作。
    """
    out = session.out_dir
    snap: dict[str, tuple[int, int]] = {}
    if not out.is_dir():
        return snap
    for p in out.rglob("*"):
        try:
            if p.is_file():
                st = p.stat()
                snap[str(p.relative_to(out)).replace("\\", "/")] = (
                    st.st_size, st.st_mtime_ns)
        except OSError:
            pass
    return snap


def _figures_dir() -> Path | None:
    try:
        from mast.agents._shared.data_paths import figures_dir
        return figures_dir()
    except Exception as exc:  # noqa: BLE001
        logger.warning("figures_dir 拿不到（图不会进图库）：%s", exc)
        return None


def _unique(dst_dir: Path, stem: str, suffix: str) -> Path:
    cand = dst_dir / f"{stem}{suffix}"
    n = 1
    while cand.exists():
        cand = dst_dir / f"{stem}_{n}{suffix}"
        n += 1
    return cand


def harvest(session, before: dict[str, tuple[int, int]]) -> HarvestResult:
    """收走本次运行新增/改动的产物。"""
    res = HarvestResult()
    out = session.out_dir
    if not out.is_dir():
        return res

    after = snapshot(session)
    changed = [rel for rel, sig in after.items() if before.get(rel) != sig]
    changed.sort()

    total_new = 0
    for rel in changed:
        size = after[rel][0]
        total_new += size
    if total_new > MAX_NEW_BYTES:
        biggest = sorted(changed, key=lambda r: after[r][0], reverse=True)[:5]
        res.notes.append(
            f"本次产出 {total_new / 1024**2:.0f} MiB，超过 "
            f"{MAX_NEW_BYTES / 1024**2:.0f} MiB 上限，未收取。最大的几个："
            + "、".join(f"{r}({after[r][0] / 1024**2:.0f} MiB)" for r in biggest)
            + "。请只把需要留存的结果写进 out/。")
        res.skipped = changed
        return res

    figdir = _figures_dir()
    for rel in changed:
        res.new_files.append(rel)
        p = out / rel
        if p.suffix.lower() not in FIGURE_EXT:
            continue
        if after[rel][0] > MAX_FIGURE_BYTES:
            res.skipped.append(rel)
            res.notes.append(
                f"{rel} 有 {after[rel][0] / 1024**2:.0f} MiB，超过单图 "
                f"{MAX_FIGURE_BYTES / 1024**2:.0f} MiB 上限，没有进图库"
                "（降低 dpi 或改存 .npz 数据）。")
            continue
        if figdir is None:
            continue
        try:
            dst = _unique(figdir, f"{session.sid}_{Path(rel).stem}", p.suffix)
            shutil.copy2(p, dst)
            res.figures.append(str(dst))
        except OSError as exc:
            res.notes.append(f"{rel} 拷进图库失败：{exc}")

    rp = out / "result.json"
    if "result.json" in changed and rp.is_file():
        try:
            raw = rp.read_text(encoding="utf-8")
            json.loads(raw)                      # 只验形状，回传的是原文
            if len(raw) > RESULT_JSON_CHARS:
                res.result_json = raw[:RESULT_JSON_CHARS]
                res.result_truncated = True
                res.notes.append(
                    f"result.json 有 {len(raw)} 字符，只回传了前 "
                    f"{RESULT_JSON_CHARS}。完整文件在 out/result.json，"
                    "可以用 record_analysis(metrics_path=...) 直接读它。")
            else:
                res.result_json = raw
        except json.JSONDecodeError as exc:
            res.notes.append(f"out/result.json 不是合法 JSON（{exc}）—— "
                             "用 mastdata.save_result(...) 写它最稳妥。")
        except OSError as exc:
            res.notes.append(f"读不到 out/result.json：{exc}")

    return res


def images_for_toolmessage(res: HarvestResult) -> list[str]:
    """挂到 ToolMessage ``additional_kwargs`` 上的图（最多 N 张）。

    上限用 ``vision_mw.MAX_IMAGES_PER_REQUEST`` —— **import 常量，不重打字面量**。
    生产方在记、消费方读不到，是这个仓踩过的静默失败形状。
    """
    try:
        from mast.agents._shared.vision_mw import MAX_IMAGES_PER_REQUEST
        cap = int(MAX_IMAGES_PER_REQUEST)
    except Exception:  # noqa: BLE001
        cap = 2
    return res.figures[:cap]


__all__ = ["HarvestResult", "MAX_FIGURE_BYTES", "MAX_NEW_BYTES",
           "harvest", "images_for_toolmessage", "snapshot"]
