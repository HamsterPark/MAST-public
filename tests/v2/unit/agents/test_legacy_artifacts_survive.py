"""旧文档不能因为「新建了一份同类文档」就从产物列表里消失。

2026-08-02 打包前审计（`artifacts.py:553/:588`）。原来的 legacy 枚举写成：

    if art_id in doc_classes:      # 该类文档库非空 → 整个 legacy 走查跳过
        continue

两件事叠起来把它从「去重」变成「静默数据丢失」：

* `documents/migrate.py` 在生产里**零调用方**，所以真机上那些旧文件从来没被导入过
  —— legacy 分支是它们**唯一**的枚举者；
* 这个开关是按**类**、不是按**文件**的。用户随便新建一份 draft，`doc_classes` 就
  含了 `"draft"`，**同一瞬间全部旧 draft 从列表里消失**。字节还在盘上，没有任何
  东西列得出来，也没有任何东西说过一句。

去重本来就是**按文件**的问题，所以按文件问：只跳过迁移真的导入过的那些
（`legacy_stem`）。迁移是**复制**（「复制导入，原文件不动」），所以这一步不能省 ——
省了就会一份文件出现两次。
"""

from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.agents._shared import artifacts as A  # noqa: E402


class _Meta:
    def __init__(self, kind, title, legacy_stem=None, root_kind="experiment"):
        self.kind = kind
        self.title = title
        self.legacy_stem = legacy_stem
        self.root_kind = root_kind


class _Entry:
    """The bits of DocEntry that list_existing touches."""

    def __init__(self, doc_id, kind, title, path, legacy_stem=None):
        self.doc_id = doc_id
        self.meta = _Meta(kind, title, legacy_stem)
        self.dir = path.parent
        self.latest_version = 1
        self._path = path

    def version_path(self):
        return self._path


class _Store:
    def __init__(self, entries):
        self._entries = entries

    def list(self):
        return list(self._entries)


@pytest.fixture
def legacy_tree(tmp_path, monkeypatch):
    """A rig that predates the document store: files in the old global dirs."""
    drafts = tmp_path / "data" / "drafts"
    reviews = tmp_path / "data" / "reviews"
    plans = tmp_path / "experiments" / "plans"
    for d in (drafts, reviews, plans):
        d.mkdir(parents=True)
    (drafts / "Au111_step_v001.md").write_text("old draft v1", encoding="utf-8")
    (drafts / "Au111_step_v002.md").write_text("old draft v2", encoding="utf-8")
    (reviews / "Au111_step_review_v001.md").write_text("old review", encoding="utf-8")
    (plans / "plan_abc123.md").write_text("old plan", encoding="utf-8")

    # list_existing imports these from data_paths INSIDE the function, so the
    # patch has to land on the source module, not on `artifacts`.
    import mast.agents._shared.data_paths as dp

    monkeypatch.setattr(dp, "drafts_dir", lambda: drafts)
    monkeypatch.setattr(dp, "reviews_dir", lambda: reviews)
    monkeypatch.setattr(dp, "figures_dir", lambda create=False: tmp_path / "nope")
    monkeypatch.setattr(A, "plans_dir", lambda: plans)
    return tmp_path, drafts, reviews, plans


def _install_store(monkeypatch, entries):
    import mast.documents as _documents

    monkeypatch.setattr(_documents, "store", lambda: _Store(entries), raising=False)


def _names(rows, art_id):
    return sorted(Path(r["path"]).name for r in rows if r["artifact_id"] == art_id)


def test_legacy_files_are_listed_when_the_store_is_empty(legacy_tree, monkeypatch):
    """基线：这一条以前就是对的，留着当对照。"""
    _install_store(monkeypatch, [])
    rows = A.list_existing()
    assert _names(rows, "draft") == ["Au111_step_v001.md", "Au111_step_v002.md"]
    assert _names(rows, "review") == ["Au111_step_review_v001.md"]
    assert _names(rows, "experiment_plan") == ["plan_abc123.md"]


def test_one_new_document_does_not_erase_the_old_ones(legacy_tree, monkeypatch):
    """**这条是那个 bug。** 新建一份 draft，旧 draft 全部消失。"""
    tmp_path, *_ = legacy_tree
    new_doc = tmp_path / "exp" / "reports" / "d1" / "v001.md"
    new_doc.parent.mkdir(parents=True)
    new_doc.write_text("brand new", encoding="utf-8")
    _install_store(monkeypatch, [
        _Entry("doc-new", "experiment_report", "刚写的报告", new_doc),
    ])

    rows = A.list_existing()

    assert _names(rows, "draft") == [
        "Au111_step_v001.md", "Au111_step_v002.md", "v001.md",
    ], "新文档必须与旧文件并存 —— 旧的还没被迁移，没有别的东西列得出它们"


def test_one_new_plan_document_does_not_erase_legacy_plans(legacy_tree, monkeypatch):
    """同一个形状的第二处（`:588` 的 experiment_plan 分支）。"""
    tmp_path, *_ = legacy_tree
    new_doc = tmp_path / "exp" / "plans" / "p1" / "v001.md"
    new_doc.parent.mkdir(parents=True)
    new_doc.write_text("new plan", encoding="utf-8")
    _install_store(monkeypatch, [
        _Entry("doc-plan", "experiment_plan", "新计划", new_doc),
    ])

    rows = A.list_existing()
    assert "plan_abc123.md" in _names(rows, "experiment_plan")


def test_a_migrated_file_is_not_listed_twice(legacy_tree, monkeypatch):
    """去重仍然要成立 —— 迁移是复制，原文件留在盘上。

    `migrate_legacy_markdown` 按**版本族基名**归档，所以 legacy_stem 记的是
    `Au111_step`，而磁盘上是 `Au111_step_v001.md` / `_v002.md`：两个都要被认出来。
    """
    tmp_path, *_ = legacy_tree
    migrated = tmp_path / "exp" / "reports" / "d1" / "v002.md"
    migrated.parent.mkdir(parents=True)
    migrated.write_text("migrated content", encoding="utf-8")
    _install_store(monkeypatch, [
        _Entry("doc-mig", "experiment_report", "Au111 step",
               migrated, legacy_stem="Au111_step"),
    ])

    rows = A.list_existing()

    assert _names(rows, "draft") == ["v002.md"], (
        "已迁移的版本族不该再从 legacy 目录里列一遍"
    )
    # …而没被迁移的 review 仍然在
    assert _names(rows, "review") == ["Au111_step_review_v001.md"]


def test_a_migrated_plan_is_not_listed_twice(legacy_tree, monkeypatch):
    """计划的 legacy_stem 是完整 stem（`plan_<id>`），不带版本后缀。"""
    tmp_path, *_ = legacy_tree
    migrated = tmp_path / "exp" / "plans" / "p1" / "v001.md"
    migrated.parent.mkdir(parents=True)
    migrated.write_text("migrated plan", encoding="utf-8")
    _install_store(monkeypatch, [
        _Entry("doc-plan", "experiment_plan", "计划 abc123",
               migrated, legacy_stem="plan_abc123"),
    ])

    assert _names(A.list_existing(), "experiment_plan") == ["v001.md"]


def test_an_unavailable_document_store_still_lists_legacy(legacy_tree, monkeypatch):
    """文档库读不出来时，legacy 走查是唯一的枚举者 —— 绝不能因此空手而归。"""
    import mast.documents as _documents

    def _boom():
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(_documents, "store", _boom, raising=False)

    rows = A.list_existing()
    assert _names(rows, "draft") == ["Au111_step_v001.md", "Au111_step_v002.md"]
    assert _names(rows, "experiment_plan") == ["plan_abc123.md"]
