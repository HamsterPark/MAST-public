"""文献子系统的**单一路径解析器** —— 大库 / 全文 / 库注册表。

设计文档：``docs/v2/design/document_and_library_management.md`` §4.3（陷阱 ⑧）

为什么存在
----------

在这之前 ``ingest.py`` / ``libraries.py`` / ``literature_index.py`` / ``fetch.py`` /
``fetch_board.py`` **各带一份私有的 ``_find_repo_root()``**，五份实现四种写法（有的
判 frozen，有的不判；有的 ``parents[3]``，有的走目录探测循环）。同类分家已经咬过一次：
里 paper_writing / paper_review 各自的 repo-walk 在打包版落到不存在的
安装目录，结果「草稿目录找不到」「实验 DB 缺失」。收敛到这里，加 env 覆盖。

两条纪律
--------

1. **每个函数懒解析。** 绝不在 import 时把路径冻结成模块常量 —— 冻结正是 #137 的
   成因（也是 ``documents/paths.py`` 与 ``core/experiment_paths.py`` 的同一条规矩）。
   测试或启动器晚一步改 env 仍然生效。
2. **默认解析出的实际路径保持兼容。** 移动已存在的索引或注册表路径可能让用户
   无法找到自己的库。各目录的默认布局如下：

   ===================  ==========================================
   ``big_index_dir()``  ``<base>/MASTv2/artifacts/literature_index``
   ``libs_dir()``       ``<base>/MASTv2/artifacts/literature_libs``
   ``papers_dir()``     ``<base>/MASTv2/data/papers``
   ===================  ==========================================

``base`` 为什么不是 ``project_root()``
--------------------------------------

``project_root()`` 是**用户数据根**：启动器可以把它指到 ``D:\\MAST-data``（env
``MAST2_PROJECT_ROOT`` / ``data_dir.txt``）。但大库和 ``MASTv2/artifacts/`` 是
**随安装包发的二进制邻接资产**，不在用户数据根里 —— 这一点 ``literature_index.py``
的 ``_index_base()`` 已经明确写过（"NOT project_root(), which may be a custom
data_dir.txt user-data location that has no index"）。

所以 base 的解析是：**frozen → exe 所在目录；dev → ``project_root()``**。
dev 下两者相等（``project_root()`` 就是仓库根），所以默认路径不变；frozen 下忽略
数据根重定向，跟收敛前的行为也一致。要把某个目录单独挪走，用下面三个 env
覆盖 —— 那是唯一被支持的搬家方式。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mast._runtime_paths import project_root

__all__ = [
    "base_dir",
    "big_index_dir",
    "papers_dir",
    "libs_dir",
    "registry_path",
    "fetch_board_path",
    "ENV_BIG_INDEX_DIR",
    "ENV_PAPERS_DIR",
    "ENV_LIBS_DIR",
]

#: env 覆盖名。三个目录各自独立可搬 —— 大库是只读资产，全文和注册表是可写用户数据，
#: 现实部署里它们完全可能落在不同的盘上。
ENV_BIG_INDEX_DIR = "MAST_LITERATURE_INDEX_DIR"
ENV_PAPERS_DIR = "MAST_PAPERS_DIR"
ENV_LIBS_DIR = "MAST_LITERATURE_LIBS_DIR"

#: 三个目录共同的中间层。历史遗留（v1 时代 ``mast/`` 与 ``MASTv2/`` 是兄弟目录），
#: 但**不能改** —— 改了就是把用户既有的库和大库判成不存在。
_MASTV2 = "MASTv2"


def _env_dir(var: str) -> Path | None:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser() if raw else None


def base_dir() -> Path:
    """三个文献目录的公共 base（见模块 docstring 末节）。

    frozen 时刻意**不用** ``project_root()``：打包版里 ``project_root()`` 可能被
    启动器指向自定义用户数据目录，而 ``MASTv2/artifacts/`` 是随 exe 发的资产。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return project_root()


def big_index_dir(*, create: bool = False) -> Path:
    """大库 —— 50k 摘要 + ``vectors.npy``（205 MB）+ parquet 元数据。

    ``MAST_LITERATURE_INDEX_DIR`` env > ``<base>/MASTv2/artifacts/literature_index``。
    默认 ``create=False``：这个目录**存在与否本身就是信息**（未 provision 的机器上
    检索要如实报「大库缺失」，而不是被一个我们刚建出来的空目录骗过去）。
    """
    d = _env_dir(ENV_BIG_INDEX_DIR) or base_dir() / _MASTV2 / "artifacts" / "literature_index"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def papers_dir(*, create: bool = False) -> Path:
    """全文 —— ``<papers>/<workid_slug>/{source.pdf,fulltext.txt,chunks.parquet,meta.json}``。

    ``MAST_PAPERS_DIR`` env > ``<base>/MASTv2/data/papers``。首次 ingest 时懒建。
    """
    d = _env_dir(ENV_PAPERS_DIR) or base_dir() / _MASTV2 / "data" / "papers"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def libs_dir(*, create: bool = False) -> Path:
    """库注册表 + 取文请求板所在目录。

    ``MAST_LITERATURE_LIBS_DIR`` env > ``<base>/MASTv2/artifacts/literature_libs``。
    """
    d = _env_dir(ENV_LIBS_DIR) or base_dir() / _MASTV2 / "artifacts" / "literature_libs"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def registry_path() -> Path:
    """``<libs>/registry.json`` —— 全局库索引。

    注意角色：实验专属库的**成员权威在实验文件夹**的 ``library/members.jsonl``，
    这个文件只是全局索引 + 缓存（见 ``knowledge/experiment_library.py``）。
    """
    return libs_dir() / "registry.json"


def fetch_board_path() -> Path:
    """``<libs>/fetch_requests.json`` —— 取文请求板。"""
    return libs_dir() / "fetch_requests.json"
