# -*- coding: utf-8 -*-
"""状态目录、原子写入、配置（设计文档 T2 / T3 / D2 / D4）。"""
from __future__ import annotations

import fnmatch
import os

import pytest

from mast.gallery import config as gcfg
from mast.gallery import paths as gpaths


# ── 状态目录 ───────────────────────────────────────────────────────────


def test_state_dir_is_re_resolved_on_every_call(tmp_path, monkeypatch):
    """不缓存路径：env 一变就跟着变（缓存住的路径会在复位后指向别处）。"""
    monkeypatch.setenv("MAST_GALLERY_DIR", str(tmp_path / "a"))
    assert gpaths.state_dir() == tmp_path / "a"
    monkeypatch.setenv("MAST_GALLERY_DIR", str(tmp_path / "b"))
    assert gpaths.state_dir() == tmp_path / "b"


def test_default_state_dir_is_under_the_experiment_root(tmp_path, monkeypatch):
    monkeypatch.delenv("MAST_GALLERY_DIR", raising=False)
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(tmp_path / "exps"))
    assert gpaths.state_dir() == tmp_path / "exps" / "_gallery"


def test_the_suite_never_points_at_the_real_state_dir(gallery_dir):
    """tests/v2/conftest.py 的 autouse 必须已经把状态目录指到了 tmp（T3，阻断型）。"""
    assert gpaths.state_dir() == gallery_dir
    assert "MAST-Data" not in str(gpaths.state_dir())


# ── 原子写入 ───────────────────────────────────────────────────────────


def test_temp_files_are_not_swept_as_crash_leftovers(tmp_path):
    """T2：runtime 启动时 sweep_partials 删掉实验根下所有 *.part-*。"""
    from mast.logging.v2.filestore import sweep_partials

    tmp = gpaths.tmp_path_for(tmp_path / "marks.json")
    assert not fnmatch.fnmatch(tmp.name, "*.part-*")
    tmp.write_text("{}", encoding="utf-8")
    sweep_partials(tmp_path)
    assert tmp.exists(), "写到一半的临时文件被启动清扫删掉了"


def test_atomic_write_replaces_and_leaves_no_temp_behind(tmp_path):
    p = tmp_path / "x" / "a.json"
    gpaths.atomic_write_json(p, {"a": 1})
    gpaths.atomic_write_json(p, {"a": 2})
    assert p.read_text(encoding="utf-8") == '{"a":2}'
    assert [f.name for f in p.parent.iterdir()] == ["a.json"]


def test_a_locked_target_is_tolerated_only_when_asked(tmp_path, monkeypatch):
    """T24：Excel 开着 csv 时 os.replace 抛 PermissionError。"""
    real_replace = os.replace

    def locked(src, dst):
        if str(dst).endswith("marks.csv"):
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(gpaths.os, "replace", locked)
    p = tmp_path / "marks.csv"
    assert gpaths.atomic_write_bytes(p, b"x", tolerate_locked=True) is False
    with pytest.raises(PermissionError):
        gpaths.atomic_write_bytes(p, b"x")
    assert not [f for f in tmp_path.iterdir() if gpaths.TMP_MARK in f.name], "临时文件没有清掉"
    assert not p.exists()


def test_read_paths_never_create_the_state_dir(gallery_dir):
    """T3：GET 背后的每一个读函数都不 mkdir、不写文件。"""
    from mast.gallery import index, marks, service

    assert gcfg.load_config().roots == []
    assert index.read_index_bytes(prefer_gzip=True) is None
    doc = marks.load_marks()
    assert doc["items"] == {} and doc["tags"] == marks.DEFAULT_TAGS
    for fmt in ("json", "md", "csv", "series_csv"):
        marks.export_file(fmt)
    gcfg.suggest_roots()
    assert service.get_status()["phase"] == "idle"
    assert not gallery_dir.exists()


# ── 配置 ───────────────────────────────────────────────────────────────


def test_config_round_trips_through_a_fresh_load(gallery_dir, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    roots, errors = gcfg.normalise_roots([{"path": str(data)}])
    assert errors == []
    gcfg.save_config(gcfg.GalleryConfig(roots=roots, workers=3))
    again = gcfg.load_config()
    assert again.roots == roots and again.workers == 3
    assert again.roots[0].name == "data" and os.path.isabs(again.roots[0].path)


def test_normalise_roots_refuses_what_would_break_ids(tmp_path):
    a = tmp_path / "a"
    b = a / "b"
    b.mkdir(parents=True)
    c = tmp_path / "c"
    c.mkdir()
    _r, errors = gcfg.normalise_roots([{"path": str(a)}, {"path": str(b)}])
    assert any("嵌套" in e for e in errors), errors
    _r, errors = gcfg.normalise_roots([{"path": str(a)}, {"path": str(a)}])
    assert any("重复" in e for e in errors), errors
    _r, errors = gcfg.normalise_roots([{"path": str(tmp_path / "missing")}])
    assert any("不是已存在的目录" in e for e in errors), errors
    for bad in ("_x", ".x", "a/b", "a:b", "CON", "x."):
        _r, errors = gcfg.normalise_roots([{"name": bad, "path": str(c)}])
        assert errors, bad
    _r, errors = gcfg.normalise_roots([{"name": "SPM", "path": str(a)},
                                       {"name": "spm", "path": str(c)}])
    assert any("根名重复" in e for e in errors), errors


def test_derived_names_are_unique_and_do_not_steal_explicit_ones(tmp_path):
    x1, x2, x3 = tmp_path / "one" / "SPM", tmp_path / "two" / "SPM", tmp_path / "three"
    for d in (x1, x2, x3):
        d.mkdir(parents=True)
    roots, errors = gcfg.normalise_roots([{"path": str(x1)}, {"path": str(x2)},
                                          {"name": "SPM_2", "path": str(x3)}])
    assert errors == []
    names = [r.name for r in roots]
    assert names[0] == "SPM" and names[2] == "SPM_2"
    assert len({n.casefold() for n in names}) == 3


def test_a_root_inside_the_state_dir_is_refused(gallery_dir):
    inner = gallery_dir / "thumbs"
    inner.mkdir(parents=True)
    _r, errors = gcfg.normalise_roots([{"path": str(inner)}])
    assert any("状态目录" in e for e in errors), errors


def test_suggestions_come_from_recorded_session_dirs_without_touching_the_instrument(
        tmp_path, gallery_dir):
    from mast.core import scan_registry

    session = tmp_path / "SPM" / "2001" / "200109" / "20010910"
    session.mkdir(parents=True)
    scan_registry.clear()
    try:
        scan_registry.record_session_dir(str(session))
        got = dict(gcfg.suggest_roots())
    finally:
        scan_registry.clear()
    assert str(tmp_path / "SPM") in got           # 上溯三级 = 整棵数据树
    assert str(session) in got
    assert not gallery_dir.exists()


def test_configured_roots_are_not_suggested_again(tmp_path, gallery_dir):
    from mast.core import scan_registry

    session = tmp_path / "SPM" / "2001" / "200109" / "20010910"
    session.mkdir(parents=True)
    roots, _e = gcfg.normalise_roots([{"name": "SPM", "path": str(tmp_path / "SPM")}])
    gcfg.save_config(gcfg.GalleryConfig(roots=roots))
    scan_registry.clear()
    try:
        scan_registry.record_session_dir(str(session))
        got = dict(gcfg.suggest_roots())
    finally:
        scan_registry.clear()
    assert str(tmp_path / "SPM") not in got
