"""实验文件夹的路径真源 —— 一个实验 = 一个自包含文件夹。

设计文档：``docs/v2/design/experiment_folder_persistence.md``

这个模块只做**纯路径计算**：给定实验/样品的 id 与名字，算出它们在磁盘上的位置。
它不碰 DB、不碰 Nanonis、不做 I/O（除非调用方显式传 ``create=True``）。

两条必须守住的规则
------------------

1. **每个函数懒调用** :func:`mast._runtime_paths.project_root` / :func:`experiment_root`，
   绝不在 import 时把路径冻结成模块常量。冻结的启动器和测试都会在 import 之后
   才改 env / 注入设置（``data_paths.py`` 的同一条注释记的是 ，
   那次就是因为 import 时冻结导致打包版所有数据路径指向不存在的目录）。

2. **目录名创建时冻结，改名只改 DB**。理由是硬的：``logging/v2/schema.py`` 把
   ``scan_files`` 放进 ``_APPEND_ONLY_FACT_TABLES``，BEFORE UPDATE 触发器直接
   ABORT —— ``current_path`` 永远无法 UPDATE。目录一改名，所有历史行就指向
   不存在的路径。目录名里的 ``__<id8>`` 后缀保证它永远能被 id 找回，名字过时
   不影响可发现性。

experiment_root 的解析顺序
--------------------------

``MAST_EXPERIMENT_ROOT`` env  >  :func:`set_experiment_root` 注入的设置  >  默认值

默认值是**仓库/安装盘的根目录下的 ``MAST-Data/experiments``**（Windows 上即
``D:\\MAST-Data\\experiments``），刻意放在代码仓库和安装目录之外：实验数据的
生命周期比软件长得多，OTA 更新、重装、换版本都不该碰到它。

设置的注入方向是单向的 settings → holder → 消费者（与
``agents/_shared/experiment_prefs.py`` 同一形状）。这样 ``core/`` 不必 import
``webui/``，层依赖保持干净。
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from pathlib import Path

from mast._runtime_paths import project_root

__all__ = [
    "experiment_root",
    "set_experiment_root",
    "get_experiment_root_setting",
    "index_dir",
    "quarantine_dir",
    "slug",
    "experiment_dir_name",
    "sample_dir_name",
    "experiment_dir",
    "sample_dir",
    "raw_dir",
    "fits_path_budget",
    "is_within",
    "RAW_KIND_DIRS",
    "MANAGED_SUBDIRS",
]

# ── 命名 ──────────────────────────────────────────────────────────────

# 与 agents/_shared/data_paths.py:100 的 _SLUG_RE 同源。Python 的 ``\w`` 本身是
# Unicode-aware，中文原样保留；``< > : " / \ | ? *`` 和控制字符都不是 ``\w``，
# 会被一并替换成 ``_`` —— 这正好覆盖 Windows 的非法文件名字符。
_SLUG_RE = re.compile(r"[^\w一-鿿-]+")

# Windows 保留设备名。这些名字**作为目录名会直接创建失败**，而且带扩展名的
# ``CON.txt`` 同样非法。大小写不敏感。
_WIN_RESERVED = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})

# 实验/样品名 slug 的字符上限。原位模式下 raw/nanonis/ 会再吃掉 12 字符，
# 所以那时用更紧的预算（见 fits_path_budget）。
SLUG_MAX_EXPERIMENT = 40
SLUG_MAX_SAMPLE = 32
SLUG_MAX_EXPERIMENT_INPLACE = 32
SLUG_MAX_SAMPLE_INPLACE = 24

# Windows MAX_PATH。留 58 字符给文件名（Nanonis 的 basename 最长见过 ~40）。
_MAX_PATH = 260
_FILENAME_BUDGET = 58

#: ``raw/`` 下按扩展名分的子目录。测量文件落在这里 —— 这是 MAST 自管区，
#: 只含完整、已哈希、已登记的文件。
RAW_KIND_DIRS: dict[str, str] = {
    ".sxm": "sxm",
    ".dat": "dat",
    ".3ds": "3ds",
}
#: 认不出扩展名的测量文件的落点。
RAW_OTHER_DIR = "other"

#: 原位模式下 Nanonis 自己写入的子目录。**不属于** MAST 自管区：
#: 允许出现 .ini / 半成品 / 垃圾，MAST 只读+登记，不碰不删。
RAW_NANONIS_DIR = "nanonis"

#: MAST 自管区（相对样品目录 / 实验目录）。watcher 对这些路径下的文件
#: **无条件忽略** —— 这是断掉 copy→detect→copy 递归的那一刀。
#: 详见 logging/v2/filestore.py 的 classify()。
#:
#: 加新顶层子目录时**必须同时改 ``_scaffold_experiment``**（两个独立列表，本项目
#: 反复踩的「一处定义、多处白名单」）；且**先进这里、再上写入者** —— 反过来的话
#: 写入的瞬间 watcher 就会把它当外来文件去认领。
MANAGED_SUBDIRS: tuple[str, ...] = (
    "derived", "map", "chats", "env", "plans", "reports", "exports", "logs",
    "library", ".mast",
)


def slug(text: str, fallback: str = "untitled", *, max_chars: int = 40) -> str:
    """把任意标题变成一个跨平台安全的目录名片段。

    保留中文；把 Windows 非法字符压成 ``_``；躲开保留设备名；去掉 Windows 会
    静默吞掉的尾部点和空格；按**字符数**截断（并保证 UTF-8 字节数不失控）。
    """
    s = unicodedata.normalize("NFC", str(text or "")).strip()
    s = _SLUG_RE.sub("_", s)
    s = s.strip("_. ")
    if not s:
        s = fallback
    if len(s) > max_chars:
        s = s[:max_chars].strip("_. ") or fallback
    # 某些 CJK 字符 NFC 后仍可能让字节数偏大；目录名的字节预算按 3 倍字符算足够，
    # 但极端情况下（组合字符）再收一刀，避免个别文件系统的 255 字节上限。
    while len(s.encode("utf-8", "ignore")) > 200 and len(s) > 1:
        s = s[:-1].strip("_. ") or fallback
    if s.split(".")[0].upper() in _WIN_RESERVED:
        s = "_" + s
    return s


def experiment_dir_name(experiment_id: str, title: str, created_date: str,
                        *, max_chars: int = SLUG_MAX_EXPERIMENT) -> str:
    """``YYYY-MM-DD__<slug>__<id8>``。

    ``created_date`` 取 ISO 时间戳的前 10 个字符即可（``2026-07-28T14:03:11`` →
    ``2026-07-28``）。

    同一天可以有多个同名实验，日期与标题不足以唯一标识目录，
    所以保留 ``__<id8>`` 后缀。双下划线做分隔，让 slug 内仍可使用单下划线。
    """
    date = (created_date or "")[:10] or "0000-00-00"
    return f"{date}__{slug(title, 'experiment', max_chars=max_chars)}__{_id8(experiment_id)}"


def sample_dir_name(index: int, name: str, sample_id: str,
                    *, max_chars: int = SLUG_MAX_SAMPLE) -> str:
    """``S<NN>__<slug>__<sid6>``。

    ``index`` 是该实验内的 1-based 创建序号，**单调递增且不回收**（样品改名或
    不再使用都不回收），这样 S02 永远只指一个东西，资源管理器的字母序也就等于
    真实的实验流程顺序。
    """
    n = max(1, int(index or 1))
    return f"S{n:02d}__{slug(name, 'sample', max_chars=max_chars)}__{_id8(sample_id, 6)}"


def _id8(value: str, n: int = 8) -> str:
    """id 的前 n 个十六进制字符（去掉 UUID 的连字符）。"""
    raw = re.sub(r"[^0-9a-zA-Z]", "", str(value or ""))
    return (raw[:n] or "0" * n).lower()


# ── 根目录 ────────────────────────────────────────────────────────────

_lock = threading.RLock()
_root_setting: str = ""


def set_experiment_root(value: str | Path | None) -> None:
    """注入设置里的 experiment_root（settings → holder 单向流）。

    传空值 = 回到默认。由 runtime 启动 hydration 和 ``POST /api/settings``
    调用；本模块不 import ``webui.settings_store``，避免 core → webui 的反向依赖。
    """
    global _root_setting
    with _lock:
        _root_setting = str(value or "").strip()


def get_experiment_root_setting() -> str:
    """当前注入的设置值（空字符串 = 未设置，走默认）。"""
    with _lock:
        return _root_setting


def _default_root() -> Path:
    """默认实验根：安装盘根目录下的 ``MAST-Data/experiments``。

    Windows 上 ``project_root()`` 是 ``D:\\...\\MAST`` → anchor ``D:\\`` →
    ``D:\\MAST-Data\\experiments``。刻意在仓库/安装目录之外：实验数据比软件活得久。

    非 Windows（CI / 测试）上写文件系统根是不可行的，退到 ``project_root()``
    的兄弟目录。
    """
    root = project_root()
    if os.name == "nt":
        anchor = root.anchor
        if anchor:
            return Path(anchor) / "MAST-Data" / "experiments"
    return root.parent / "MAST-Data" / "experiments"


def experiment_root(*, create: bool = False) -> Path:
    """所有实验文件夹的父目录。

    ``MAST_EXPERIMENT_ROOT`` env > :func:`set_experiment_root` 的设置 > 默认值。
    """
    env = os.environ.get("MAST_EXPERIMENT_ROOT", "").strip()
    if env:
        d = Path(env).expanduser()
    else:
        setting = get_experiment_root_setting()
        d = Path(setting).expanduser() if setting else _default_root()
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def index_dir(*, create: bool = False) -> Path:
    """``<root>/_index/`` —— 跨实验索引，不属于任何实验。"""
    d = experiment_root() / "_index"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def quarantine_dir(*, create: bool = False) -> Path:
    """``<root>/_quarantine/`` —— 无归属数据的隔离区。

    没有活跃实验/样品时收到的文件落这里。**绝不丢字节**：宁可让用户事后
    认领，也不能因为"当时没选样品"就把一次真实测量扔掉。
    """
    d = experiment_root() / "_quarantine"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# ── 实验 / 样品目录 ───────────────────────────────────────────────────

def experiment_dir(dir_name: str, *, create: bool = False) -> Path:
    """给定目录名，返回实验目录。

    ``create=True`` 时**懒创建**并铺好骨架。空实验不预建目录 —— 这既是"十年后
    重启"的正确行为（那时才建），也让迁移时几十条空壳测试实验不污染顶层。
    """
    d = experiment_root() / dir_name
    if create:
        _scaffold_experiment(d)
    return d


def sample_dir(exp_dir: Path, dir_name: str, *, create: bool = False) -> Path:
    """给定实验目录与样品目录名，返回样品目录。"""
    d = Path(exp_dir) / "samples" / dir_name
    if create:
        _scaffold_sample(d)
    return d


def raw_dir(sample_path: Path, suffix: str, *, create: bool = False) -> Path:
    """样品的 ``raw/<kind>/`` 目录。``suffix`` 是带点的扩展名（``.sxm``）。"""
    kind = RAW_KIND_DIRS.get(str(suffix or "").lower(), RAW_OTHER_DIR)
    d = Path(sample_path) / "raw" / kind
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def nanonis_inplace_dir(sample_path: Path, *, create: bool = False) -> Path:
    """样品的 ``raw/nanonis/`` —— 原位模式下 Nanonis 自己写入的地盘。

    刻意**不是** ``raw/sxm/``：Nanonis 会在 session 目录里写自己的 .ini、
    settings 备份和 ``unnamed####`` 半成品。给它一块专属区域，``raw/{sxm,dat,3ds}/``
    才能保持"只含完整、已哈希、已登记文件"的语义，MANAGED 区的定义也才干净。
    """
    d = Path(sample_path) / "raw" / RAW_NANONIS_DIR
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _scaffold_experiment(d: Path) -> None:
    """铺实验目录骨架。幂等。

    这是 ``MANAGED_SUBDIRS`` 之外的**第二份**清单（``samples`` 只在这里，因为它不是
    自管区）。加子目录时两处都要改。
    """
    for sub in ("samples", "chats", "env", "plans", "reports", "exports", "logs",
                "library", ".mast"):
        (d / sub).mkdir(parents=True, exist_ok=True)


def _scaffold_sample(d: Path) -> None:
    """铺样品目录骨架。幂等。``raw/nanonis/`` 只在开原位模式时才建。"""
    for sub in ("raw/sxm", "raw/dat", "raw/3ds",
                "derived/figures", "derived/analysis", "derived/vision",
                "map", "chats", "env"):
        (d / sub).mkdir(parents=True, exist_ok=True)


# ── 路径预算 ──────────────────────────────────────────────────────────

def fits_path_budget(root: Path, exp_name: str, sample_name: str,
                     *, inplace: bool = False) -> bool:
    """这组目录名加上最深的一层文件后，是否仍在 Windows MAX_PATH 之内。

    最深路径是 ``<root>/<exp>/samples/<sample>/raw/nanonis/<filename>``
    （原位模式）或 ``.../raw/sxm/<filename>``。
    """
    deepest = "raw/nanonis/" if inplace else "raw/sxm/"
    n = (len(str(root)) + 1 + len(exp_name) + len("/samples/")
         + len(sample_name) + 1 + len(deepest) + _FILENAME_BUDGET)
    return n <= _MAX_PATH


def shrink_to_budget(root: Path, title: str, sample_name: str,
                     *, inplace: bool = False) -> tuple[int, int]:
    """为这对名字挑一组能放得下的 slug 长度上限。

    返回 ``(experiment_max_chars, sample_max_chars)``。逐级收紧到纯 id 后缀；
    **绝不因为名字太长而让创建失败** —— 名字是可以短的，数据不能丢。
    """
    exp_cap = SLUG_MAX_EXPERIMENT_INPLACE if inplace else SLUG_MAX_EXPERIMENT
    smp_cap = SLUG_MAX_SAMPLE_INPLACE if inplace else SLUG_MAX_SAMPLE
    for e, s in ((exp_cap, smp_cap), (24, 20), (12, 10), (0, 0)):
        exp_name = experiment_dir_name("0" * 8, title, "2026-01-01", max_chars=max(e, 1))
        smp_name = sample_dir_name(1, sample_name, "0" * 6, max_chars=max(s, 1))
        if e == 0 or fits_path_budget(root, exp_name, smp_name, inplace=inplace):
            return max(e, 1), max(s, 1)
    return 1, 1


# ── 归属判定 ──────────────────────────────────────────────────────────

def is_within(path: str | Path, parent: str | Path) -> bool:
    """``path`` 是否在 ``parent`` 之内（含相等）。永不抛。

    用 ``Path.parts`` 前缀比较而不是字符串 ``startswith``：后者会把
    ``.../S01_foo_extra`` 误判成在 ``.../S01_foo`` 里。不解析 symlink（Windows
    上 resolve 对不存在的路径行为不一致，且我们比的是逻辑归属）。
    """
    try:
        p = Path(path).absolute().parts
        q = Path(parent).absolute().parts
    except (OSError, ValueError):
        return False
    if len(q) > len(p):
        return False
    if os.name == "nt":
        return [x.lower() for x in p[:len(q)]] == [x.lower() for x in q]
    return p[:len(q)] == q
