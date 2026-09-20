# -*- coding: utf-8 -*-
"""清点（设计文档 T1 / T14 / D4 / D15）。"""
from __future__ import annotations

import time

import numpy as np
import pytest

from mast.gallery import config as gcfg
from mast.gallery import inventory as inv


def _cfg(**roots) -> gcfg.GalleryConfig:
    return gcfg.GalleryConfig(roots=[gcfg.RootSpec(name=n, path=str(p)) for n, p in roots.items()])


def _frame(n: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1e-11, (n, n))


def _scan(cfg, only=None, state=None):
    files, scopes = inv.walk_roots(cfg, only=only, state=state)
    return inv.collapse(files), scopes


# ── 遍历 ───────────────────────────────────────────────────────────────


def test_walk_skips_sidecar_dirs_state_dirs_and_other_files(tmp_path, synth, gallery_dir):
    root = tmp_path / "SPM"
    synth.sxm(root / "20010910" / "a.sxm", _frame())
    synth.dat(root / "env" / "b.dat", ["Bias calc (V)", "Current (A)"], [[0, 0], [1, 1]])
    synth.sxm(root / "_gallery" / "c.sxm", _frame())
    (root / "notes.txt").write_text("x")
    files, scopes = inv.walk_roots(_cfg(SPM=root), state=gallery_dir)
    assert [f.id for f in files] == ["SPM/20010910/a.sxm"]
    assert scopes == ["SPM"]


def test_a_state_dir_placed_inside_a_root_is_not_walked(tmp_path, synth):
    root = tmp_path / "SPM"
    state = root / "somewhere"
    synth.sxm(state / "thumbs" / "x.sxm", _frame())
    synth.sxm(root / "d" / "a.sxm", _frame())
    files, _s = inv.walk_roots(_cfg(SPM=root), state=state)
    assert [f.id for f in files] == ["SPM/d/a.sxm"]


def test_only_walks_that_subtree_and_matches_by_path_segment(tmp_path, synth):
    root = tmp_path / "SPM"
    synth.sxm(root / "20010910" / "a.sxm", _frame())
    synth.sxm(root / "200109101" / "b.sxm", _frame())
    files, scopes = inv.walk_roots(_cfg(SPM=root), only="SPM/20010910")
    assert [f.id for f in files] == ["SPM/20010910/a.sxm"]
    assert scopes == ["SPM/20010910"]
    assert inv.in_scope("SPM/20010910/a.sxm", "SPM/20010910")
    assert not inv.in_scope("SPM/200109101/b.sxm", "SPM/20010910")


def test_disabled_and_missing_roots_are_not_walked(tmp_path, synth):
    synth.sxm(tmp_path / "A" / "a.sxm", _frame())
    cfg = gcfg.GalleryConfig(roots=[gcfg.RootSpec("A", str(tmp_path / "A"), enabled=False),
                                    gcfg.RootSpec("USB", str(tmp_path / "unplugged"))])
    files, scopes = inv.walk_roots(cfg)
    assert files == [] and scopes == []


# ── 副本折叠（T14）────────────────────────────────────────────────────


def test_byte_identical_copies_fold_and_same_name_different_mtime_does_not(
        tmp_path, synth, monkeypatch, t0_ns):
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(tmp_path / "exps"))
    z = _frame()
    a = synth.sxm(tmp_path / "SPM" / "d1" / "scan.sxm", z, mtime_ns=t0_ns)
    b = synth.sxm(tmp_path / "exps" / "E1" / "raw" / "scan.sxm", z, mtime_ns=t0_ns)
    # 同名、不同测量。差 1 s 而不是 1 ns：NTFS 的时间戳精度是 100 ns，差 1 ns 会被舍成同一个值。
    synth.sxm(tmp_path / "SPM" / "d2" / "scan.sxm", z, mtime_ns=t0_ns + 10**9)
    groups, _s = _scan(_cfg(SPM=tmp_path / "SPM", EXP=tmp_path / "exps"))
    assert len(groups) == 2
    folded = [(rep, members) for rep, members in groups if len(members) == 2]
    assert len(folded) == 1
    rep, members = folded[0]
    assert rep.id == "SPM/d1/scan.sxm", "代表应当是实验根之外的原件"
    assert {m.path for m in members} == {str(a), str(b)}


# ── 缓存身份（T1）─────────────────────────────────────────────────────


def test_same_size_but_new_mtime_is_a_new_measurement(tmp_path, synth, t0_ns):
    """Nanonis 重用编号：同名同尺寸的新测量不能被旧记录顶替。"""
    root = tmp_path / "SPM"
    p = synth.sxm(root / "d" / "unnamed0001.sxm", _frame(seed=1), bias=1.0, mtime_ns=t0_ns)
    cfg = _cfg(SPM=root)
    cache: dict = {}
    groups, _s = _scan(cfg)
    new, changed, failed = inv.update_inventory(groups, cache, "2001-09-13 10:00")
    iid = "SPM/d/unnamed0001.sxm"
    assert new == [iid] and changed == [] and failed == []
    assert cache[iid]["bias_V"] == pytest.approx(1.0)

    groups, _s = _scan(cfg)
    assert inv.update_inventory(groups, cache, "2001-09-13 10:30") == ([], [], [])

    size = p.stat().st_size
    synth.sxm(p, _frame(seed=2), bias=2.0, mtime_ns=t0_ns + 3_600 * 10**9)
    assert p.stat().st_size == size, "这条测试要的正是「尺寸不变」"
    groups, _s = _scan(cfg)
    new, changed, failed = inv.update_inventory(groups, cache, "2001-09-13 11:00")
    assert changed == [iid]
    assert cache[iid]["bias_V"] == pytest.approx(2.0)
    assert cache[iid]["added"] == "2001-09-13 11:00", "内容换了就是新测量，应当进最新一批"


def test_a_parser_version_bump_keeps_the_batch(tmp_path, synth, t0_ns, monkeypatch):
    root = tmp_path / "SPM"
    synth.sxm(root / "d" / "a.sxm", _frame(), mtime_ns=t0_ns)
    cfg = _cfg(SPM=root)
    cache: dict = {}
    groups, _s = _scan(cfg)
    inv.update_inventory(groups, cache, "2001-09-13 10:00")
    monkeypatch.setattr(inv, "VER_INVENTORY", inv.VER_INVENTORY + 1)
    groups, _s = _scan(cfg)
    _new, changed, _f = inv.update_inventory(groups, cache, "2001-09-13 11:00")
    assert changed == ["SPM/d/a.sxm"]
    assert cache["SPM/d/a.sxm"]["added"] == "2001-09-13 10:00"


def test_purge_only_happens_inside_walked_scopes(tmp_path, synth):
    root = tmp_path / "SPM"
    synth.sxm(root / "d1" / "a.sxm", _frame())
    synth.sxm(root / "d2" / "b.sxm", _frame())
    cfg = _cfg(SPM=root)
    cache: dict = {}
    groups, _s = _scan(cfg)
    inv.update_inventory(groups, cache, "s")
    (root / "d1" / "a.sxm").unlink()
    groups, scopes = _scan(cfg, only="SPM/d2")          # 只走了 d2：d1 的消失不作数
    assert inv.purge(cache, scopes, {r.id for r, _m in groups}) == []
    groups, scopes = _scan(cfg)
    assert inv.purge(cache, scopes, {r.id for r, _m in groups}) == ["SPM/d1/a.sxm"]


def test_an_unplugged_root_keeps_its_entries(tmp_path):
    cache = {"USB/d/a.sxm": {"size": 1, "mtime_ns": 1}}
    groups, scopes = _scan(_cfg(USB=tmp_path / "unplugged"))
    assert inv.purge(cache, scopes, set()) == []
    assert "USB/d/a.sxm" in cache


def test_an_unreadable_file_is_recorded_not_fatal(tmp_path):
    root = tmp_path / "SPM"
    (root / "d").mkdir(parents=True)
    (root / "d" / "broken.sxm").write_bytes(b"not a scan")
    cache: dict = {}
    groups, _s = _scan(_cfg(SPM=root))
    new, _c, failed = inv.update_inventory(groups, cache, "s")
    assert new == ["SPM/d/broken.sxm"] and failed and failed[0][0] == "SPM/d/broken.sxm"
    assert cache["SPM/d/broken.sxm"]["err"]


# ── 名字与头信息 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("fn, prefix", [
    ("topo_STS_002_0010.sxm", "topo_STS_002"),
    ("unnamed0033.sxm", "unnamed"),
    ("Grid Spectroscopy001.3ds", "Grid Spectroscopy"),
    ("Spectrum001.dat", "Spectrum"),
    ("sample_2001-08-09_0018.sxm", "sample_2001-08-09"),
    ("0001.sxm", "0001"),
    ("topo.sxm", "topo"),
])
def test_file_prefix(fn, prefix):
    assert inv.file_prefix(fn) == prefix


def test_dir_labels():
    assert inv.dir_key("SPM/2001/200109/20010910/a.sxm") == "SPM/2001/200109/20010910"
    assert inv.dir_key("SPM/a.sxm") == "SPM"
    assert inv.dir_label("SPM/2001/200109/20010910") == "2001-09-10"
    assert inv.dir_short("SPM/2001/200109/20010910") == "0910"
    assert inv.dir_label("W/session-A") == "session-A"


def test_sxm_summary_reads_the_header(tmp_path, synth):
    p = synth.sxm(tmp_path / "a.sxm", _frame(64), scan_dir="up", rec_date="13.09.2001",
                  rec_time="15:56:00", acq_s=328.3, range_nm=(3.0, 3.0),
                  offset_nm=(-70.91086, 197.9593), angle=22.5, bias=1.0, setpoint_a=3e-10)
    s = inv.sxm_summary(str(p))
    assert s["t0"] == time.mktime(time.strptime("13.09.2001 15:56:00", "%d.%m.%Y %H:%M:%S"))
    assert s["w_nm"] == pytest.approx(3.0) and s["h_nm"] == pytest.approx(3.0)
    assert s["cx_nm"] == pytest.approx(-70.91086, abs=1e-5)
    assert s["cy_nm"] == pytest.approx(197.9593, abs=1e-5)
    assert s["angle"] == pytest.approx(22.5)
    assert (s["nx"], s["ny"], s["scan_dir"]) == (64, 64, "up")
    assert s["bias_V"] == pytest.approx(1.0)
    assert s["setpoint_pA"] == pytest.approx(300.0)
    assert s["acq_s"] == pytest.approx(328.3)
    assert s["channels"] == ["Z"]


def test_dat_summary_reads_the_header_and_columns(tmp_path, synth):
    V = np.linspace(2.0, -2.5, 11)
    I = V * 1e-11
    p = synth.dat(tmp_path / "s.dat",
                  ["Bias calc (V)", "Current [00001] (A)", "Current [00002] (A)", "LI Demod 1 X [00001] (A)"],
                  np.column_stack([V, I, I, I]))
    s = inv.dat_summary(str(p))
    assert s["n"] == 11 and s["sweeps"] == 2
    assert s["Vmin"] == pytest.approx(-2.5) and s["Vmax"] == pytest.approx(2.0)
    assert s["has_LI"] and s["has_current"]
    assert s["x_nm"] == pytest.approx(-66.9234) and s["zoff_pm"] == pytest.approx(150.0)
    assert s["t0"] == time.mktime(time.strptime("10.09.2001 19:04:22", "%d.%m.%Y %H:%M:%S"))
    # 合成 .dat 头缺少工作点：偏压缺失，设定点记 0。
    assert s["bias_V"] is None and s["setpoint_pA"] == 0.0


def test_grid_summary_counts_completed_points(tmp_path, synth):
    nx = ny = 6
    npts = 5
    ch = {"Current [AVG] (A)": np.zeros((ny, nx, npts)), "Bias [AVG] (V)": np.zeros((ny, nx, npts))}
    p = synth.grid(tmp_path / "g.3ds", nx=nx, ny=ny, npts=npts, channels=ch, have=13)
    s = inv.grid_summary(str(p))
    assert (s["nx"], s["ny"], s["npts"], s["have"]) == (6, 6, 5, 13)
    assert s["w_nm"] == pytest.approx(3.0) and s["h_nm"] == pytest.approx(3.0)
    assert s["sweeps"] == 2 and s["v0"] == pytest.approx(2.0) and s["v1"] == pytest.approx(-2.5)
    assert s["t1"] == time.mktime(time.strptime("11.09.2001 19:09:34", "%d.%m.%Y %H:%M:%S"))
