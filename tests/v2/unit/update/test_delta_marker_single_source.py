"""增量更新 marker 的读写双方应共享同一字段定义。

缺失文件名可能把目录误当作更新包，产生误导性错误。
测试核验 write_delta_marker 与启动器使用相同键名。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_MASTV2 = str(_ROOT / "MASTv2")
if sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.update.client import (  # noqa: E402
    DELTA_MARKER,
    DELTA_MARKER_FIELDS,
    write_delta_marker,
)

_LAUNCHER = _ROOT / "mast2_launcher.py"


def _launcher_marker_keys() -> set[str]:
    """启动器**实际**从 marker 里取的键 —— 从源码 AST 取,不靠读。

    形状:``info.get("filename", "")`` / ``info.get("filename")``,其中 info 是
    ``_json.loads(marker.read_text(...))`` 的结果。这里保守地收集所有
    ``.get("…")`` 的字面量首参,再与真源求交 —— 只要有一个真源字段**没被读到**
    或有一个被读的字段**不在真源里**,就说明两侧漂开了。
    """
    tree = ast.parse(_LAUNCHER.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    return keys


def test_the_field_the_launcher_reads_exists_in_the_single_source():
    """`filename` —— 写错它不会报「找不到」,会报「权限不足」。"""
    assert "filename" in DELTA_MARKER_FIELDS
    assert "filename" in _launcher_marker_keys(), (
        "启动器不再读 'filename' 了?那 DELTA_MARKER_FIELDS 也要跟着改 —— "
        "两侧漂开的代价见本文件顶部")
    # 曾经写错的那个名字**不许**成为真源的一部分
    assert "file" not in DELTA_MARKER_FIELDS


def test_every_source_field_is_something_the_launcher_or_applier_reads():
    """真源里不该有没人读的字段 —— 那是「读起来像在生效」的死配置。"""
    read_somewhere = _launcher_marker_keys()
    applier = (Path(_MASTV2) / "mast" / "update" / "client.py").read_text(encoding="utf-8")
    unread = [f for f in DELTA_MARKER_FIELDS
              if f not in read_somewhere and f'"{f}"' not in applier]
    assert not unread, f"这些 marker 字段没人读:{unread}"


def test_write_delta_marker_emits_exactly_those_fields_no_bom(tmp_path):
    p = write_delta_marker(tmp_path, "delta_1.0.0_to_1.0.1.zip",
                           "1.0.0", "1.0.1", "ab" * 32)
    assert p.name == DELTA_MARKER
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), (
        "marker 带了 BOM —— 读侧 json.loads 会炸(OTA 上线时踩过)")
    data = json.loads(raw.decode("utf-8"))
    assert set(data) == set(DELTA_MARKER_FIELDS)
    assert data["filename"] == "delta_1.0.0_to_1.0.1.zip"


def test_a_marker_missing_filename_resolves_to_the_directory_itself():
    """钉住那个**失败模式本身**,因为它伪装成权限问题。

    这条不是在测我们的代码,是在记录「为什么少一个键会变成 PermissionError」——
    下一个看到那句 Errno 13 的人,应该能从这里找到答案,而不是去查 ACL(我查了)。
    """
    info: dict = {"file": "delta.zip"}          # 手写时写错的那份
    pdir = Path(r"D:\MAST-data\pending_update")
    assert pdir / str(info.get("filename", "")) == pdir, (
        "filename 缺席时,启动器算出来的不是一个坏路径,而是 pending_update **目录本身** "
        "—— 它 exists()，于是下一步拿目录当 zip 打开")
