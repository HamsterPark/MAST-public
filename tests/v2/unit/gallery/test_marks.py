# -*- coding: utf-8 -*-
"""标记（设计文档 D11 / T23 / T24）。语义照旧版兼容格式的存盘服务。"""
from __future__ import annotations

import csv
import io
import json
import os
import time

import pytest

from mast.gallery import index, marks, paths
from mast.gallery import config as gcfg


# ── 合并语义 ───────────────────────────────────────────────────────────


def test_a_late_edit_is_dropped(gallery_dir):
    marks.patch_marks({"items": {"R/a.sxm": {"r": 2, "ts": 200}}})
    marks.patch_marks({"items": {"R/a.sxm": {"r": 1, "ts": 100}}})
    assert marks.load_marks()["items"]["R/a.sxm"]["r"] == 2


def test_a_tombstone_blocks_a_late_resurrection(gallery_dir):
    marks.patch_marks({"items": {"R/a.sxm": {"r": 1, "ts": 100}}})
    marks.patch_marks({"items": {"R/a.sxm": {"del": True, "ts": 200}}})
    marks.patch_marks({"items": {"R/a.sxm": {"r": 2, "ts": 150}}})
    doc = marks.load_marks()
    assert "R/a.sxm" not in doc["items"] and doc["tomb"]["R/a.sxm"] == 200
    marks.patch_marks({"items": {"R/a.sxm": {"r": 2, "ts": 300}}})
    doc = marks.load_marks()
    assert doc["items"]["R/a.sxm"]["r"] == 2 and "R/a.sxm" not in doc["tomb"]


def test_an_empty_mark_is_a_delete_but_an_anchor_alone_is_a_mark(gallery_dir):
    marks.patch_marks({"items": {"R/a.sxm": {"r": 1, "ts": 1}}})
    marks.patch_marks({"items": {"R/a.sxm": {"r": 0, "tags": [], "note": "  ", "ts": 2}}})
    assert "R/a.sxm" not in marks.load_marks()["items"]
    marks.patch_marks({"items": {"R/s.dat": {"anchor": {"id": "R/a.sxm"}, "ts": 3}}})
    assert "R/s.dat" in marks.load_marks()["items"]


def test_a_series_is_removed_by_del_or_by_empty_ids(gallery_dir):
    marks.patch_marks({"series": {"S1": {"name": "x", "ids": ["R/a.sxm"], "ts": 1}}})
    assert "S1" in marks.load_marks()["series"]
    marks.patch_marks({"series": {"S1": {"name": "x", "ids": [], "ts": 2}}})
    doc = marks.load_marks()
    assert "S1" not in doc["series"] and doc["stomb"]["S1"] == 2
    marks.patch_marks({"series": {"S2": {"name": "y", "ids": ["R/a.sxm"], "ts": 3}}})
    marks.patch_marks({"series": {"S2": {"del": True, "ts": 4}}})
    assert "S2" not in marks.load_marks()["series"]


def test_directory_marks_and_the_tag_table(gallery_dir):
    marks.patch_marks({"days": {"R/20010910": {"done": True, "note": "针尖好", "ts": 5}},
                       "tags": ["a", " ", "b"]})
    marks.patch_marks({"days": {"R/20010910": {"done": False, "note": "旧的", "ts": 4}}})
    doc = marks.load_marks()
    assert doc["days"]["R/20010910"]["note"] == "针尖好"
    assert doc["tags"] == ["a", "b"]


def test_marks_round_trip_exactly_through_a_fresh_read(gallery_dir):
    mark = {"r": 2, "tags": ["超结构"], "note": "n", "t": "2001-09-13 10:00:00", "ts": 5,
            "anchor": {"id": "R/f.sxm", "fn": "f.sxm", "rel": "next", "dt": 4, "desc": "谱结束后 4 秒开始扫描",
                       "u": 0.3, "v": 0.2, "inside": True}}
    marks.patch_marks({"items": {"R/a.dat": mark}})
    raw = json.loads(paths.layout().marks.read_text(encoding="utf-8"))
    assert raw["items"]["R/a.dat"] == mark


# ── 写出 ───────────────────────────────────────────────────────────────


def test_every_patch_bumps_rev_and_writes_the_derived_files(gallery_dir):
    rev1, _u = marks.patch_marks({"items": {"R/a.sxm": {"r": 1, "ts": 1}}})
    rev2, _u = marks.patch_marks({"items": {"R/b.sxm": {"r": 2, "ts": 2}}})
    assert (rev1, rev2) == (1, 2)
    lay = paths.layout()
    for p in (lay.marks, lay.marks_md, lay.marks_csv, lay.marks_series_csv):
        assert p.is_file(), p
    assert lay.marks_csv.read_bytes().startswith(b"\xef\xbb\xbf")        # Excel 认得中文


def test_md_and_csv_carry_full_paths(gallery_dir, tmp_path):
    root = tmp_path / "SPM"
    root.mkdir()
    roots, _e = gcfg.normalise_roots([{"name": "SPM", "path": str(root)}])
    gcfg.save_config(gcfg.GalleryConfig(roots=roots))
    lay = paths.layout()
    item = {"id": "SPM/d/in_index.sxm", "k": "f", "d": "SPM/d", "fn": "in_index.sxm",
            "p": r"X:\data\in_index.sxm", "t": 1.0, "w": 5.0, "b": 1.0, "sp": 300.0}
    index.write_index(index.make_doc([item], [], "2001-09-13 10:00"), lay)
    marks.patch_marks({
        "items": {"SPM/d/in_index.sxm": {"r": 2, "ts": 1, "note": "好"},
                  "SPM/d/on_disk_only.sxm": {"r": 1, "ts": 2},
                  "GONE/x.sxm": {"r": -1, "ts": 3}},
        "series": {"S1": {"name": "一组", "k": "f", "ts": 4,
                          "ids": ["SPM/d/in_index.sxm", "SPM/d/on_disk_only.sxm"]}},
    })
    md = lay.marks_md.read_text(encoding="utf-8")
    assert r"X:\data\in_index.sxm" in md                      # 索引里的绝对路径
    assert str(root / "d" / "on_disk_only.sxm") in md          # 不在索引里：根路径 + 相对路径
    assert "`GONE/x.sxm`（图库里没有）" in md                 # 连根都不认识
    assert "### · 一组" in md and "▤ 一组" in md
    rows = list(csv.reader(io.StringIO(lay.marks_csv.read_text(encoding="utf-8-sig"))))
    assert rows[0] == ["目录", "文件", "评级", "标签", "备注", "系列", "位置系于", "参数", "标记时间", "路径"]
    by_file = {r[1]: r for r in rows[1:]}
    assert by_file["in_index.sxm"][2] == "★ 重点" and by_file["in_index.sxm"][9] == r"X:\data\in_index.sxm"
    srows = list(csv.reader(io.StringIO(lay.marks_series_csv.read_text(encoding="utf-8-sig"))))
    assert srows[1][0] == "一组" and srows[1][5] == "2"


def test_snapshots_are_throttled_to_one_per_half_hour(gallery_dir):
    lay = paths.layout()

    def snaps():
        return sorted(lay.marks_backup.iterdir())

    marks.patch_marks({"items": {"R/a.sxm": {"r": 1, "ts": 1}}})
    assert len(snaps()) == 1
    marks.patch_marks({"items": {"R/b.sxm": {"r": 1, "ts": 2}}})
    assert len(snaps()) == 1
    old = snaps()[-1].rename(lay.marks_backup / "marks_20000101_000000.json")
    t = time.time() - marks.BACKUP_EVERY_S - 60
    os.utime(old, (t, t))
    marks.patch_marks({"items": {"R/c.sxm": {"r": 1, "ts": 3}}})
    assert len(snaps()) == 2


def test_a_locked_csv_does_not_stop_the_marks_from_saving(gallery_dir, monkeypatch):
    """T24：Excel 开着 marks.csv 时，marks.json 照样写成；csv 等下一次。"""
    marks.patch_marks({"items": {"R/a.sxm": {"r": 1, "ts": 1}}})
    lay = paths.layout()
    before = lay.marks_csv.read_bytes()
    real = os.replace

    def locked(src, dst):
        if str(dst).endswith(".csv"):
            raise PermissionError("Excel")
        return real(src, dst)

    monkeypatch.setattr(paths.os, "replace", locked)
    rev, _u = marks.patch_marks({"items": {"R/b.sxm": {"r": 2, "ts": 2}}})
    assert rev == 2 and "R/b.sxm" in marks.load_marks()["items"]
    assert lay.marks_csv.read_bytes() == before
    assert not [p for p in lay.state.iterdir() if paths.TMP_MARK in p.name]


def test_a_corrupt_marks_file_is_never_replaced_by_an_empty_one(gallery_dir):
    lay = paths.layout()
    lay.state.mkdir(parents=True)
    lay.marks.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        marks.load_marks()
    with pytest.raises(ValueError):
        marks.patch_marks({"items": {"R/a.sxm": {"r": 1, "ts": 1}}})
    assert lay.marks.read_text(encoding="utf-8") == "{broken"


# ── 导入 / 导出 ────────────────────────────────────────────────────────


def _standalone_doc() -> dict:
    """旧版兼容格式 marks.json 的形状（键相对它的 RAW 根，目录键是日期）。"""
    day = "2001/200109/20010907"
    return {
        "version": 1, "rev": 113, "tags": ["原子分辨佳", "新标签"],
        "items": {
            f"{day}/a.sxm": {"r": 2, "tags": [], "note": "", "t": "2001-09-13 18:59:41",
                             "ts": 1000382400000, "k": "f", "meta": "m"},
            f"{day}/rep.dat": {"r": 2, "t": "2001-09-13 20:20:15", "ts": 1000386000000, "k": "s",
                               "anchor": {"id": f"{day}/a.sxm", "fn": "a.sxm", "rel": "next", "dt": 4,
                                          "desc": "谱结束后 4 秒开始扫描", "u": 0.3, "v": 0.16, "inside": True}},
            f"{day}/missing.sxm": {"r": 1, "t": "2001-09-13 19:00:00", "ts": 1},
        },
        "series": {"S1": {"name": "一组", "k": "f", "ids": [f"{day}/a.sxm", f"{day}/missing.sxm"],
                          "t": "2001-09-13 19:00:00", "ts": 2}},
        "days": {"20010907": {"done": True, "note": "针尖好"}, "20010908": {"done": False, "note": ""}},
        "tomb": {}, "stomb": {},
    }


def test_importing_a_standalone_marks_file(gallery_dir):
    lay = paths.layout()
    d = "SPM/2001/200109/20010907"
    items = [{"id": f"{d}/a.sxm", "k": "f", "d": d, "fn": "a.sxm"},
             {"id": f"{d}/rep.dat", "k": "s", "d": d, "fn": "rep.dat"}]
    index.write_index(index.make_doc(items, [], "x"), lay)
    res = marks.import_marks(_standalone_doc(), key_prefix="SPM")
    assert (res["items"], res["series"], res["days"], res["tags_added"]) == (3, 1, 1, 1)
    assert res["unmatched"] == 1                           # missing.sxm：照样导入，只计数
    doc = marks.load_marks()
    assert set(doc["items"]) == {f"{d}/a.sxm", f"{d}/rep.dat", f"{d}/missing.sxm"}
    assert doc["items"][f"{d}/rep.dat"]["anchor"]["id"] == f"{d}/a.sxm"
    assert doc["series"]["S1"]["ids"] == [f"{d}/a.sxm", f"{d}/missing.sxm"]
    assert set(doc["days"]) == {d} and doc["days"][d]["note"] == "针尖好"   # 日期目录键映射到索引里的目录
    again = marks.import_marks(_standalone_doc(), key_prefix="SPM/")
    assert (again["items"], again["series"], again["days"], again["tags_added"]) == (0, 0, 0, 0)
    assert again["rev"] == res["rev"], "什么都没变就不该写"


def test_import_takes_only_the_newer_mark(gallery_dir):
    marks.patch_marks({"items": {"P/a.sxm": {"r": 1, "t": "2001-09-14 00:00:00", "ts": 1}}})
    res = marks.import_marks({"items": {"a.sxm": {"r": 2, "t": "2001-09-13 00:00:00"}}}, key_prefix="P")
    assert res["items"] == 0 and marks.load_marks()["items"]["P/a.sxm"]["r"] == 1
    res = marks.import_marks({"items": {"a.sxm": {"r": 2, "t": "2001-09-15 00:00:00"}}}, key_prefix="P")
    assert res["items"] == 1 and marks.load_marks()["items"]["P/a.sxm"]["r"] == 2


def test_export_is_generated_on_the_fly_without_writing(gallery_dir):
    data, media, name = marks.export_file("md")
    assert name == "marks.md" and media.startswith("text/markdown")
    assert data.decode("utf-8").startswith("# 数据标记清单")
    with pytest.raises(ValueError):
        marks.export_file("xlsx")
    assert not gallery_dir.exists()
