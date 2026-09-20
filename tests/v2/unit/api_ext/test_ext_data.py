"""原始数据出口：路径约束（穿越 / 根外 / 网络路径 / 扩展名）与帧的朝向约定。

朝向约定在这里**独立**写一遍（不调实现来证明实现）：
* ``scan_dir=down``：文件第一行就是图的顶边；``scan_dir=up``：第一行是底边 ⇒ 翻转；
* backward 块沿 x 镜像存储 ⇒ 去镜像后与 forward 同一几何朝向。
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from _ext_world import H, write_sxm


def _data_dir(world):
    d = world.tmp / "proj" / "working-sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _npz(resp):
    assert resp.status_code == 200, resp.text
    z = np.load(io.BytesIO(resp.content), allow_pickle=False)
    return z, json.loads(str(z["meta_json"]))


@pytest.mark.parametrize("scan_dir", ["down", "up"])
def test_frames_come_back_in_one_geometric_orientation(world, scan_dir):
    raw_fwd = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw_bwd = 100 + np.arange(12, dtype=np.float32).reshape(3, 4)
    p = write_sxm(_data_dir(world) / f"f_{scan_dir}.sxm", raw_fwd, raw_bwd, scan_dir=scan_dir)
    z, meta = _npz(world.client.get("/data/frame", params={"path": str(p)}, headers=H()))
    exp_f = raw_fwd if scan_dir == "down" else raw_fwd[::-1]
    exp_b = raw_bwd[:, ::-1] if scan_dir == "down" else raw_bwd[:, ::-1][::-1]
    np.testing.assert_array_equal(z["forward"], exp_f)
    np.testing.assert_array_equal(z["backward"], exp_b)
    assert meta["scan_dir"] == scan_dir and meta["served_block"] == "forward"
    assert meta["width_nm"] == pytest.approx(5.0) and meta["nm_per_px"] == pytest.approx(1.25)


def test_raw_file_download_is_byte_identical(world):
    p = write_sxm(_data_dir(world) / "raw.sxm", np.zeros((2, 2)), np.ones((2, 2)))
    r = world.client.get("/data/file", params={"path": str(p)}, headers=H())
    assert r.status_code == 200 and r.content == p.read_bytes()


def test_a_missing_channel_lists_the_ones_that_exist(world):
    p = write_sxm(_data_dir(world) / "c.sxm", np.zeros((2, 2)), np.ones((2, 2)))
    r = world.client.get("/data/frame", params={"path": str(p), "channel": "Current"},
                         headers=H())
    assert r.status_code == 404 and r.json()["channels"] == ["Z"]


def test_paths_outside_the_allowed_roots_are_refused(world, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "secret.sxm"
    write_sxm(outside, np.zeros((2, 2)), np.ones((2, 2)))
    r = world.client.get("/data/file", params={"path": str(outside)}, headers=H())
    assert r.status_code == 403 and r.json()["error"] == "path_not_allowed"


def test_traversal_is_resolved_before_the_check(world, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere2") / "x.sxm"
    write_sxm(outside, np.zeros((2, 2)), np.ones((2, 2)))
    base = _data_dir(world)
    rel = "\\".join([".."] * len(base.parts)) + str(outside)[2:]   # base\..\..\<outside>
    sneaky = str(base) + "\\" + rel
    r = world.client.get("/data/file", params={"path": sneaky}, headers=H())
    assert r.status_code in (403, 404), r.text
    assert r.status_code != 200


def test_non_data_extensions_are_refused_even_inside_the_roots(world):
    db = _data_dir(world) / "records.db"
    db.write_bytes(b"SQLite format 3\x00")
    r = world.client.get("/data/file", params={"path": str(db)}, headers=H())
    assert r.status_code == 403 and r.json()["error"] == "extension_not_allowed"


def test_network_paths_are_refused(world):
    r = world.client.get("/data/file", params={"path": r"\\server\share\x.sxm"}, headers=H())
    assert r.status_code == 403


def test_the_listing_never_takes_a_directory_from_the_client(world, tmp_path_factory):
    """``records_export`` 的目录参数是**排他覆盖** —— 转发它就等于任意目录读。"""
    write_sxm(_data_dir(world) / "listed.sxm", np.zeros((2, 2)), np.ones((2, 2)))
    decoy = tmp_path_factory.mktemp("decoy")
    write_sxm(decoy / "decoy.sxm", np.zeros((2, 2)), np.ones((2, 2)))
    r = world.client.get("/data/files", params={"n": 50, "dir": str(decoy)}, headers=H())
    assert r.status_code == 200
    paths = [f["path"] for f in r.json()["files"]]
    assert any(p.endswith("listed.sxm") for p in paths), paths
    assert not any(p.endswith("decoy.sxm") for p in paths), paths
