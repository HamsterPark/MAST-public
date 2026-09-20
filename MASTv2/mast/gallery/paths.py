# -*- coding: utf-8 -*-
"""图库状态目录的位置，与原子写入。

状态目录：``MAST_GALLERY_DIR`` > ``<experiment_root()>/_gallery``（设计文档 D3）。

两条不能忘：

* **每次调用都重新解析，不缓存路径。** 测试靠 ``MAST_GALLERY_DIR`` 把状态目录指到 tmp；
  一个缓存住的路径会在 env 复位之后继续指着上一个值 —— 或者更糟，回落到真实数据根。
  需要在一次长任务里保持不变的调用方（后台构建）在开头取一次 :func:`layout`，
  之后一路显式传下去，不在半路重新读 env。
* **临时文件后缀是 ``.tmp-<pid>-<tid>``，绝不含 ``.part-``**：runtime 启动时对实验根跑
  ``logging.v2.filestore.sweep_partials``，删掉所有 ``*.part-*``（陷阱 T2）。状态目录默认
  就在实验根下，用 ``.part-`` 的临时文件会在一次重启里被当成崩溃残留删掉。

读路径（:func:`state_dir` / :func:`layout` / :func:`read_json`）不创建任何目录。
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_STATE_DIR = "MAST_GALLERY_DIR"
STATE_DIRNAME = "_gallery"

#: 原子写入的临时文件标记。**不许是 ``.part-``**，见模块说明。
TMP_MARK = ".tmp-"


def state_dir() -> Path:
    """状态目录。只解析，不创建。"""
    env = os.environ.get(ENV_STATE_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    from mast.core.experiment_paths import experiment_root

    return experiment_root() / STATE_DIRNAME


@dataclass(frozen=True)
class Layout:
    """一个状态目录里各文件的位置。纯路径计算，不碰磁盘。"""

    state: Path

    @property
    def config(self) -> Path:
        return self.state / "config.json"

    @property
    def cache(self) -> Path:
        return self.state / "cache"

    @property
    def thumbs(self) -> Path:
        return self.state / "thumbs"

    @property
    def index(self) -> Path:
        return self.state / "index.json"

    @property
    def index_gz(self) -> Path:
        return self.state / "index.json.gz"

    @property
    def marks(self) -> Path:
        return self.state / "marks.json"

    @property
    def marks_md(self) -> Path:
        return self.state / "marks.md"

    @property
    def marks_csv(self) -> Path:
        return self.state / "marks.csv"

    @property
    def marks_series_csv(self) -> Path:
        return self.state / "marks_series.csv"

    @property
    def marks_backup(self) -> Path:
        return self.state / "marks_backup"

    def cache_file(self, name: str) -> Path:
        return self.cache / name


def layout(state: Path | str | None = None) -> Layout:
    return Layout(Path(state) if state is not None else state_dir())


def ensure_state_dir(lay: Layout | None = None) -> Path:
    """写入方在第一次写之前调。读路径绝不调它。"""
    lay = lay or layout()
    lay.state.mkdir(parents=True, exist_ok=True)
    return lay.state


def cache_dir() -> Path:
    return layout().cache


def thumbs_dir() -> Path:
    return layout().thumbs


def index_path() -> Path:
    return layout().index


def marks_path() -> Path:
    return layout().marks


# ── 原子写入 ───────────────────────────────────────────────────────────


def tmp_path_for(path: Path | str) -> Path:
    p = Path(path)
    return p.with_name(f"{p.name}{TMP_MARK}{os.getpid()}-{threading.get_ident():x}")


#: ``os.replace`` 撞上「目标正被别的句柄读着」时的重试间隔（秒）。合计约 1.5 s。
_REPLACE_RETRY_S = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45)


def atomic_write_bytes(path: Path | str, data: bytes, *, tolerate_locked: bool = False) -> bool:
    """先写临时文件再 ``os.replace``。返回是否真的替换成功。

    **Windows 上「正在被读」也会让替换失败。** Python 的 ``open()`` 与 Starlette 的
    ``FileResponse`` 打开文件时都不带 ``FILE_SHARE_DELETE``，于是目标正被别的线程读着的
    那一瞬（``GET /marks`` 读 marks.json、``GET /index`` 读 index.json、缩略图正在发送），
    ``os.replace`` 抛 ``PermissionError``。这种占用只有几毫秒，所以先按
    :data:`_REPLACE_RETRY_S` 短暂重试；一直占着（Excel 打开的 csv）才算真的锁住。

    ``tolerate_locked``：重试完仍被占着时，置真则删掉临时文件、返回 False，下一次写再试
    （陷阱 T24）。**只给派生文件用**，``marks.json`` 本身绝不能静默写不进 —— 它会抛，
    由调用方回 degraded，前端把改动留在待存队列里 5 s 后补发。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path_for(p)
    with open(tmp, "wb") as fh:
        fh.write(data)
    for delay in (*_REPLACE_RETRY_S, None):
        try:
            os.replace(tmp, p)
            return True
        except PermissionError:
            if delay is None:
                break
            time.sleep(delay)
    try:
        tmp.unlink()
    except OSError:
        pass
    if tolerate_locked:
        return False
    raise PermissionError(f"{p} 一直被占用，替换失败（已重试 {len(_REPLACE_RETRY_S)} 次）")


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8",
                      tolerate_locked: bool = False) -> bool:
    return atomic_write_bytes(path, text.encode(encoding), tolerate_locked=tolerate_locked)


def atomic_write_json(path: Path | str, obj: Any, *, indent: int | None = None) -> None:
    if indent is None:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(obj, ensure_ascii=False, indent=indent)
    atomic_write_text(path, text)


def read_json(path: Path | str, default: Any) -> Any:
    """缓存类文件的读取：不存在或损坏都回 ``default``（缓存坏了重算即可）。

    **标记文件不走这里** —— ``marks.json`` 损坏时回默认值，下一次存盘就会用一个
    空文档把操作员的全部标记覆盖掉。见 :func:`mast.gallery.marks.load_marks`。
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default
