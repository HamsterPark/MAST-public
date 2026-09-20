"""Incremental (delta) update: build + apply + tamper detection.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/update/test_delta.py -x -v
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.update.delta import (
    DeltaError, apply_delta, build_delta, diff_trees, hash_tree,
)


def _write(root: Path, files: dict):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_diff_trees(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, {"a.py": "1", "b.py": "2", "sub/c.py": "3"})
    _write(new, {"a.py": "1", "b.py": "CHANGED", "sub/d.py": "4"})  # b changed, c removed, d added
    d = diff_trees(old, new)
    assert d["added"] == ["sub/d.py"]
    assert d["changed"] == ["b.py"]
    assert d["removed"] == ["sub/c.py"]


def test_build_and_apply(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, {"a.py": "v1", "b.py": "keep", "gone.py": "x", "_internal/m.py": "old"})
    _write(new, {"a.py": "v2", "b.py": "keep", "added.py": "new", "_internal/m.py": "new"})
    delta = tmp_path / "delta.zip"
    build_delta(old, new, delta, from_version="1.0.0", to_version="1.0.1")

    # apply to a copy of old → should become identical to new
    target = tmp_path / "target"
    import shutil
    shutil.copytree(old, target)
    res = apply_delta(delta, target)
    assert res["to_version"] == "1.0.1"
    assert hash_tree(target) == hash_tree(new)        # byte-identical to new
    assert not (target / "gone.py").exists()          # removed
    assert (target / "added.py").read_text() == "new"
    assert (target / "a.py").read_text() == "v2"


def test_delta_is_smaller_than_full(tmp_path):
    # a 1-file change in a big tree → tiny delta
    old, new = tmp_path / "old", tmp_path / "new"
    big = {f"_internal/f{i}.py": "x" * 1000 for i in range(50)}
    _write(old, big)
    changed = dict(big); changed["_internal/f0.py"] = "y" * 1000
    _write(new, changed)
    delta = tmp_path / "d.zip"
    build_delta(old, new, delta, from_version="1", to_version="2")
    m = __import__("mast.update.delta", fromlist=["read_delta_manifest"]).read_delta_manifest(delta)
    assert m["changed"] == ["_internal/f0.py"] and not m["added"] and not m["removed"]


def test_tamper_detection_aborts_before_writing(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, {"a.py": "v1"})
    _write(new, {"a.py": "v2"})
    delta = tmp_path / "d.zip"
    build_delta(old, new, delta, from_version="1", to_version="2")

    # corrupt the file blob inside the delta (manifest sha stays old → mismatch)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(delta) as zin, zipfile.ZipFile(bad, "w") as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item == "files/a.py":
                data = b"TAMPERED"
            zout.writestr(item, data)

    target = tmp_path / "target"
    import shutil
    shutil.copytree(old, target)
    with pytest.raises(DeltaError):
        apply_delta(bad, target)
    # target untouched (still v1) — no half-patch
    assert (target / "a.py").read_text() == "v1"


def test_missing_target(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, {"a.py": "1"})
    _write(new, {"a.py": "2"})
    delta = tmp_path / "d.zip"
    build_delta(old, new, delta, from_version="1", to_version="2")
    with pytest.raises(DeltaError):
        apply_delta(delta, tmp_path / "does_not_exist")


def test_manifest_deltas_roundtrip():
    from mast.update.manifest import Manifest, find_delta
    m = Manifest(version="2.1.14", filename="setup.exe", sha256="abc",
                 size_bytes=100, published_at="now",
                 deltas=[{"from_version": "2.1.13", "filename": "d.zip",
                          "sha256": "def", "size_bytes": 10}])
    m2 = Manifest.from_dict(m.to_dict())
    assert m2.deltas == m.deltas
    assert find_delta(m2, "2.1.13")["filename"] == "d.zip"
    assert find_delta(m2, "2.0.0") is None


def test_apply_pending_delta_end_to_end(tmp_path):
    import json
    from mast.update.client import DELTA_MARKER, apply_pending_delta, pending_dir

    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, {"mast/a.py": "v1", "mast/gone.py": "x"})
    _write(new, {"mast/a.py": "v2", "mast/added.py": "n"})

    data_root = tmp_path / "data"
    pdir = pending_dir(data_root)
    delta = pdir / "delta.zip"
    build_delta(old, new, delta, from_version="2.1.13", to_version="2.1.14")
    (pdir / DELTA_MARKER).write_text(json.dumps({
        "filename": "delta.zip", "from_version": "2.1.13", "to_version": "2.1.14",
    }), encoding="utf-8")

    # the "installed _internal" starts as old
    install = tmp_path / "install_internal"
    import shutil
    shutil.copytree(old, install)

    status, detail = apply_pending_delta(data_root, install)
    assert status == "applied", detail
    assert hash_tree(install) == hash_tree(new)         # now identical to new
    assert not (pdir / DELTA_MARKER).exists()           # marker cleared


def test_apply_pending_delta_none(tmp_path):
    from mast.update.client import apply_pending_delta
    status, _ = apply_pending_delta(tmp_path / "data", tmp_path / "x")
    assert status == "none"


# ──────────────────────────────────────────────────────────────────────
# Literature library participates in the delta (incremental lib updates),
# while the rest of MASTv2/ (user data / runtime) stays excluded.
# ──────────────────────────────────────────────────────────────────────

def test_default_exclude_keeps_user_data_includes_lit_index(tmp_path):
    from mast.update.delta import DEFAULT_EXCLUDE_TOP, DEFAULT_INCLUDE_PREFIXES
    old, new = tmp_path / "old", tmp_path / "new"
    common = {
        "MAST2.exe": "code-v1",
        # literature library — BINARY-shipped, must be delta-tracked
        "MASTv2/artifacts/literature_index/vectors.npy": "vecA",
        "MASTv2/artifacts/literature_index/metadata.parquet": "metaA",
        # user / runtime under MASTv2/ — must NEVER enter a delta
        "MASTv2/artifacts/literature_libs/registry.json": "USER-LIBS",
        "MASTv2/artifacts/tts_cache/x.wav": "cache",
        "MASTv2/artifacts/buffer.wal.sqlite": "wal",
        # top-level user data
        "api key/kimi.env": "SECRET",
        "experiments/exp.db": "USER-DB",
    }
    _write(old, common)
    nw = dict(common)
    nw["MAST2.exe"] = "code-v2"                                   # code changed
    nw["MASTv2/artifacts/literature_index/vectors.npy"] = "vecB"  # lib refreshed
    nw["MASTv2/artifacts/literature_index/abstracts.parquet"] = "absNEW"  # lib added
    _write(new, nw)

    delta = tmp_path / "d.zip"
    man = build_delta(old, new, delta, from_version="1", to_version="2",
                      exclude_top=DEFAULT_EXCLUDE_TOP,
                      include_prefixes=DEFAULT_INCLUDE_PREFIXES)
    tracked = set(man["added"]) | set(man["changed"]) | set(man["removed"])
    # code + the lit-index changes ARE in the delta
    assert "MAST2.exe" in man["changed"]
    assert "MASTv2/artifacts/literature_index/vectors.npy" in man["changed"]
    assert "MASTv2/artifacts/literature_index/abstracts.parquet" in man["added"]
    # user data / runtime under MASTv2/ and top-level NEVER appear
    for forbidden in ("MASTv2/artifacts/literature_libs/registry.json",
                      "MASTv2/artifacts/tts_cache/x.wav",
                      "MASTv2/artifacts/buffer.wal.sqlite",
                      "api key/kimi.env", "experiments/exp.db"):
        assert forbidden not in tracked, f"{forbidden} must be excluded from delta"

    # applying preserves user data + updates the lib
    install = tmp_path / "install"
    shutil.copytree(old, install)
    apply_delta(delta, install)
    assert (install / "MAST2.exe").read_text() == "code-v2"
    assert (install / "MASTv2/artifacts/literature_index/vectors.npy").read_text() == "vecB"
    assert (install / "MASTv2/artifacts/literature_index/abstracts.parquet").read_text() == "absNEW"
    # the user's library pointers + secrets are untouched by the delta
    assert (install / "MASTv2/artifacts/literature_libs/registry.json").read_text() == "USER-LIBS"
    assert (install / "api key/kimi.env").read_text() == "SECRET"


# ──────────────────────────────────────────────────────────────────────
# OTA P0 wiring: install-root apply target + data-only hot-apply gate 
# ──────────────────────────────────────────────────────────────────────

def test_hot_apply_prefixes_mirror_delta_include_prefixes():
    """``client.HOT_APPLY_PREFIXES`` 的注释声称它 mirrors
    ``delta.DEFAULT_INCLUDE_PREFIXES`` —— 在此之前没有任何东西守着这句话。

    2026-08-02 实测两边**已经漂开了**：VIGIL v2.5 上线、``mast_vision_m12.pt``
    退役之后，两个常量都还指着那个不存在的 checkpoint。后果各不相同且都静默：
    delta 侧漂了 → 新权重永远不会被推下去（客户端报着新版本号跑旧权重）；
    client 侧漂了 → 权重增量包被判成「不是纯数据」，白走一次离线应用器重启。

    ``HOT_APPLY_PREFIXES`` 允许比 include 集**更宽**（SPA 与 docs 是随包资产，
    不走 delta 的 include 名单），但反过来不行：**每一个被 delta 追踪的出厂资产
    都必须是可热应用的**，否则那条增量路径拿不到它承诺的「不重启」。
    """
    from mast.update.client import HOT_APPLY_PREFIXES
    from mast.update.delta import DEFAULT_INCLUDE_PREFIXES

    def _covered(inc: str) -> bool:
        return any(inc == p.rstrip("/") or inc.startswith(p.rstrip("/") + "/")
                   or p.rstrip("/").startswith(inc + "/") or inc == p
                   for p in HOT_APPLY_PREFIXES)

    missing = sorted(p for p in DEFAULT_INCLUDE_PREFIXES if not _covered(p))
    assert not missing, (
        "这些出厂资产会进 delta，却不在 HOT_APPLY_PREFIXES 里，"
        f"增量更新会因此多要一次重启：{missing}")


def test_delta_is_data_only():
    from mast.update.client import delta_is_data_only
    assert delta_is_data_only(["MASTv2/artifacts/literature_index/vectors.npy",
                               "MASTv2/artifacts/literature_index/metadata.parquet"])
    assert delta_is_data_only(["MASTv2/artifacts/mast_vision_v25.pt"])
    assert delta_is_data_only(["MASTv2/artifacts/stm_quality_v1_dino.joblib"])
    # code / exe / dll → must go via the full installer
    assert not delta_is_data_only(["MAST2.exe"])
    assert not delta_is_data_only(["_internal/mast/foo.py"])
    assert not delta_is_data_only(["_internal/python313.dll"])
    # mixed (one unsafe path) → not data-only
    assert not delta_is_data_only(
        ["MASTv2/artifacts/literature_index/x.npy", "MAST2.exe"])
    assert not delta_is_data_only([])


def test_apply_pending_delta_lands_at_install_root(tmp_path):
    """Regression: the delta's install-ROOT-relative paths (MASTv2/artifacts/...,
    MAST2.exe) must land at the install ROOT, NOT under _internal."""
    import json
    import shutil
    from mast.update.client import DELTA_MARKER, apply_pending_delta, pending_dir

    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, {"MAST2.exe": "v1",
                 "MASTv2/artifacts/literature_index/vectors.npy": "A"})
    _write(new, {"MAST2.exe": "v1",  # unchanged
                 "MASTv2/artifacts/literature_index/vectors.npy": "B"})  # refreshed
    data_root = tmp_path / "data"
    pdir = pending_dir(data_root)
    delta = pdir / "d.zip"
    # diff raw (no exclude) so the lit-index path is tracked with its full relpath
    build_delta(old, new, delta, from_version="1", to_version="2",
                exclude_top=frozenset(), include_prefixes=frozenset())
    (pdir / DELTA_MARKER).write_text(json.dumps({"filename": "d.zip"}), encoding="utf-8")

    install = tmp_path / "install"
    shutil.copytree(old, install)
    status, detail = apply_pending_delta(data_root, install)
    assert status == "applied", detail
    # the lit-index file updated AT THE INSTALL ROOT (install/MASTv2/...), not
    # install/_internal/MASTv2/...
    assert (install / "MASTv2/artifacts/literature_index/vectors.npy").read_text() == "B"
    assert not (install / "_internal").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
