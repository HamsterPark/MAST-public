# -*- coding: utf-8 -*-
"""构建与索引（设计文档 D8 / D9 / D10 / §4.1 / T1）。"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
from urllib.parse import unquote

import numpy as np
import pytest

from mast.gallery import build, index, paths, render
from mast.gallery import config as gcfg

DAY = "2001/200109/20010910"


def _configure(root, name="SPM", workers=2):
    roots, errors = gcfg.normalise_roots([{"name": name, "path": str(root)}])
    assert errors == []
    gcfg.save_config(gcfg.GalleryConfig(roots=roots, workers=workers))


def _populate(root, synth, t0_ns):
    d = root / "2001" / "200109" / "20010910"
    synth.sxm(d / "topo_0001.sxm", np.random.default_rng(0).normal(0, 1e-11, (64, 64)),
              rec_time="15:00:00", mtime_ns=t0_ns)
    V = np.linspace(-2.0, 2.0, 101)
    I = np.tanh(V) * 1e-10
    synth.dat(d / "sts_0001.dat", ["Bias calc (V)", "Current (A)", "Current [bwd] (A)"],
              np.column_stack([V, I, I]), mtime_ns=t0_ns + 10**9)
    f = np.logspace(0, 3, 40)
    synth.dat(d / "psd_0001.dat", ["Frequency (Hz)", "Z PSD"], np.column_stack([f, 1.0 / f]),
              header={"Experiment": "Spectrum"}, mtime_ns=t0_ns + 2 * 10**9)
    nx = ny = 8
    npts = 7
    Vg = np.linspace(2.0, -2.0, npts)
    ch = {"Current [AVG] (A)": np.tanh(Vg)[None, None, :] * 1e-10 * np.ones((ny, nx, 1)),
          "Bias [AVG] (V)": np.broadcast_to(Vg, (ny, nx, npts)).copy()}
    synth.grid(d / "Grid Spectroscopy001.3ds", nx=nx, ny=ny, npts=npts, channels=ch)
    return d


def _items() -> dict:
    return {it["fn"]: it for it in index.read_index()["items"]}


def test_build_writes_the_documented_index(tmp_path, synth, gallery_dir, t0_ns):
    root = tmp_path / "SPM"
    d = _populate(root, synth, t0_ns)
    _configure(root)
    res = build.run_build(state=gallery_dir)
    assert res["phase"] == "done" and res["n_failed"] == 0, res["errors"]
    assert set(res) >= set(build.STATUS_KEYS)
    doc = index.read_index()
    assert doc["built"] and doc["version"] == index.INDEX_VERSION
    assert (doc["n_frames"], doc["n_spectra"], doc["n_grids"]) == (1, 2, 1)
    assert doc["roots"] == [{"name": "SPM", "path": str(root), "enabled": True, "exists": True}]
    items = _items()
    f = items["topo_0001.sxm"]
    assert f["id"] == f"SPM/{DAY}/topo_0001.sxm" and f["d"] == f"SPM/{DAY}" and f["k"] == "f"
    assert f["pf"] == "topo" and f["p"] == str(d / "topo_0001.sxm")
    assert f["mt"] == pytest.approx(t0_ns / 1e9)
    for key in index.FRAME_ALWAYS + ("w", "hn", "b", "sp", "nx", "ny", "ang", "cx", "cy", "sd",
                                     "acq", "t", "ad"):
        assert key in f, key
    assert f["th"].startswith(f"/api/gallery/thumb/SPM/{DAY}/topo_0001.sxm.jpg?v=")
    g = items["Grid Spectroscopy001.3ds"]
    assert "Grid%20Spectroscopy001.3ds.png?v=" in g["th"]
    rel = unquote(g["th"].split("/api/gallery/thumb/", 1)[1].split("?", 1)[0])
    assert (gallery_dir / "thumbs").joinpath(*rel.split("/")).is_file()
    assert (g["gx"], g["gy"], g["have"]) == (8, 8, 64)
    s = items["sts_0001.dat"]
    assert s["dn"] == 1 and s["lic"] == 0 and "ex" not in s
    assert items["psd_0001.dat"]["ex"] == "Spectrum"
    for it in items.values():
        assert None not in it.values(), it


def test_a_second_build_does_no_work(tmp_path, synth, gallery_dir, t0_ns):
    root = tmp_path / "SPM"
    _populate(root, synth, t0_ns)
    _configure(root)
    build.run_build(state=gallery_dir)
    res = build.run_build(state=gallery_dir)
    assert (res["total"], res["n_render"], res["n_analysis"], res["n_new"], res["n_changed"]) == (0, 0, 0, 0, 0)
    assert res["n_files"] == 4


def test_same_size_new_content_is_rerendered_and_reindexed(tmp_path, synth, gallery_dir, t0_ns):
    """T1 在构建层面：同名同尺寸的新测量必须重读、重出图、重进索引。"""
    root = tmp_path / "SPM"
    d = _populate(root, synth, t0_ns)
    _configure(root)
    build.run_build(state=gallery_dir)
    p = d / "topo_0001.sxm"
    size = p.stat().st_size
    synth.sxm(p, np.random.default_rng(9).normal(0, 1e-11, (64, 64)), rec_time="16:00:00",
              mtime_ns=t0_ns + 3600 * 10**9)
    assert p.stat().st_size == size
    res = build.run_build(state=gallery_dir)
    assert res["n_changed"] == 1 and res["n_render"] == 1
    it = _items()["topo_0001.sxm"]
    assert it["mt"] == pytest.approx((t0_ns + 3600 * 10**9) / 1e9)
    assert it["t"] == pytest.approx(time.mktime(time.strptime("10.09.2001 16:00:00", "%d.%m.%Y %H:%M:%S")))


def test_a_partial_build_keeps_the_rest_of_the_index(tmp_path, synth, gallery_dir, t0_ns):
    root = tmp_path / "SPM"
    _populate(root, synth, t0_ns)
    other = root / "2001" / "200109" / "20010911"
    synth.sxm(other / "later_0001.sxm", np.zeros((16, 16)) + 1e-11, mtime_ns=t0_ns + 9 * 10**9)
    _configure(root)
    build.run_build(state=gallery_dir)
    before = len(index.read_index()["items"])
    synth.sxm(other / "later_0002.sxm", np.ones((16, 16)) * 1e-11, mtime_ns=t0_ns + 10 * 10**9)
    res = build.run_build(state=gallery_dir, only="SPM/2001/200109/20010911")
    assert res["total"] == 1
    assert len(index.read_index()["items"]) == before + 1


def test_a_deleted_thumbnail_is_rendered_again(tmp_path, synth, gallery_dir, t0_ns):
    root = tmp_path / "SPM"
    _populate(root, synth, t0_ns)
    _configure(root)
    build.run_build(state=gallery_dir)
    (gallery_dir / "thumbs").joinpath("SPM", *DAY.split("/"), "topo_0001.sxm.jpg").unlink()
    res = build.run_build(state=gallery_dir)
    assert res["n_render"] == 1 and res["n_analysis"] == 0
    assert "topo_0001.sxm" in _items()


def test_a_vanished_file_leaves_the_index_and_its_thumbnail(tmp_path, synth, gallery_dir, t0_ns):
    root = tmp_path / "SPM"
    d = _populate(root, synth, t0_ns)
    _configure(root)
    build.run_build(state=gallery_dir)
    thumb = (gallery_dir / "thumbs").joinpath("SPM", *DAY.split("/"), "sts_0001.dat.png")
    assert thumb.is_file()
    (d / "sts_0001.dat").unlink()
    build.run_build(state=gallery_dir)
    assert "sts_0001.dat" not in _items() and not thumb.exists()


def test_a_cancelled_build_still_writes_a_consistent_index(tmp_path, synth, gallery_dir, t0_ns):
    root = tmp_path / "SPM"
    _populate(root, synth, t0_ns)
    _configure(root)
    cancel = threading.Event()
    cancel.set()
    res = build.run_build(state=gallery_dir, cancel=cancel)
    assert res["phase"] == "cancelled" and res["n_render"] == 0
    assert index.read_index()["items"] == []
    res = build.run_build(state=gallery_dir)
    assert res["phase"] == "done" and len(index.read_index()["items"]) == 4


def test_a_render_exception_is_retried_by_the_next_build(tmp_path, synth, gallery_dir, t0_ns, monkeypatch):
    """异常不是数据本身的结论：不能被缓存成永久的「出不了图」。"""
    root = tmp_path / "SPM"
    _populate(root, synth, t0_ns)
    _configure(root)
    real = render.render_spectrum

    def boom(*a, **k):
        raise RuntimeError("transient")

    monkeypatch.setattr(render, "render_spectrum", boom)
    res = build.run_build(state=gallery_dir)
    assert res["n_failed"] == 2 and "sts_0001.dat" not in _items()
    monkeypatch.setattr(render, "render_spectrum", real)
    res = build.run_build(state=gallery_dir)
    assert res["n_render"] == 2 and res["n_failed"] == 0 and "sts_0001.dat" in _items()


def test_an_empty_frame_is_not_retried(tmp_path, synth, gallery_dir, t0_ns):
    root = tmp_path / "SPM"
    z = np.full((32, 32), np.nan)
    z[:3] = 1e-11
    synth.sxm(root / "d" / "stub.sxm", z, mtime_ns=t0_ns)
    _configure(root)
    res = build.run_build(state=gallery_dir)
    assert res["n_render"] == 1 and res["n_failed"] == 0
    assert build.run_build(state=gallery_dir)["total"] == 0
    assert index.read_index()["items"] == []


def test_a_disabled_root_leaves_the_index(tmp_path, synth, gallery_dir, t0_ns):
    root = tmp_path / "SPM"
    _populate(root, synth, t0_ns)
    _configure(root)
    build.run_build(state=gallery_dir)
    cfg = gcfg.load_config()
    cfg.roots = [gcfg.RootSpec(r.name, r.path, enabled=False) for r in cfg.roots]
    gcfg.save_config(cfg)
    res = build.run_build(state=gallery_dir)
    assert res["phase"] == "done" and index.read_index()["items"] == []


def test_gzip_is_used_only_when_it_is_not_older_than_the_json(gallery_dir):
    lay = paths.layout()
    index.write_index(index.make_doc([], [], "2001-09-13 10:00"), lay)
    data, gz = index.read_index_bytes(prefer_gzip=True)
    assert gz and json.loads(gzip.decompress(data))["built"] is True
    _d, gz2 = index.read_index_bytes(prefer_gzip=False)
    assert gz2 is False
    st = os.stat(lay.index)
    os.utime(lay.index_gz, ns=(st.st_mtime_ns - 10**9, st.st_mtime_ns - 10**9))
    _d, gz3 = index.read_index_bytes(prefer_gzip=True)
    assert gz3 is False


def test_thumb_url_quotes_each_segment_and_carries_a_version(tmp_path):
    thumbs = tmp_path / "t"
    rel = "SPM/目录 A/Grid Spectroscopy001.3ds.png"
    p = thumbs.joinpath(*rel.split("/"))
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    url = index.thumb_url(thumbs, rel)
    assert url.startswith("/api/gallery/thumb/SPM/%E7%9B%AE%E5%BD%95%20A/Grid%20Spectroscopy001.3ds.png?v=")
    assert index.thumb_url(thumbs, "SPM/missing.png") == ""
