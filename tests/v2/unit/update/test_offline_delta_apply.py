"""Feature B — true incremental code updates (offline applier).

Pins the Python-side invariants (the PowerShell self-updater is Windows-only and
E2E-tested separately):
  * content-normalized diff drops a re-timestamped base_library.zip
  * a real content change to a zip IS shipped
  * stage_delta verifies sha256 + rejects traversal/tamper, extracts off-target
  * delta_is_data_only gates hot-apply (SPA/docs live, exe/base_library offline)
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest


def _root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 not found")


_R = _root()
if _R not in sys.path:
    sys.path.insert(0, _R)


def _zip(p: Path, members: dict, date):
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w") as z:
        for name, content in members.items():
            z.writestr(zipfile.ZipInfo(name, date_time=date), content)


def _w(p: Path, data: bytes):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_content_norm_drops_retimestamped_zip(tmp_path):
    from mast.update.delta import build_delta
    old, new = tmp_path / "old", tmp_path / "new"
    # identical members, different timestamps → must NOT be flagged changed
    _zip(old / "_internal" / "base_library.zip", {"m.py": b"x=1"}, (2020, 1, 1, 0, 0, 0))
    _zip(new / "_internal" / "base_library.zip", {"m.py": b"x=1"}, (2022, 6, 6, 6, 6, 6))
    (old / "MAST.exe").write_bytes(b"v1")
    (new / "MAST.exe").write_bytes(b"v2")
    m = build_delta(old, new, tmp_path / "d.zip", from_version="1", to_version="2",
                    exclude_top=frozenset())
    assert "_internal/base_library.zip" not in m["changed"]
    assert "MAST.exe" in m["changed"]


def test_content_norm_keeps_real_zip_change(tmp_path):
    from mast.update.delta import build_delta
    old, new = tmp_path / "old", tmp_path / "new"
    _zip(old / "a.zip", {"m.py": b"x=1"}, (2020, 1, 1, 0, 0, 0))
    _zip(new / "a.zip", {"m.py": b"x=2"}, (2020, 1, 1, 0, 0, 0))  # content differs
    m = build_delta(old, new, tmp_path / "d.zip", from_version="1", to_version="2",
                    exclude_top=frozenset())
    assert "a.zip" in m["changed"]


def test_stage_delta_verifies_and_extracts_off_target(tmp_path):
    from mast.update.delta import build_delta, stage_delta
    old, new = tmp_path / "old", tmp_path / "new"
    _w(old / "MAST.exe", b"old")
    _w(new / "MAST.exe", b"NEW")
    _w(new / "_internal" / "x.pyc", b"pyc")
    build_delta(old, new, tmp_path / "d.zip", from_version="1", to_version="2",
                exclude_top=frozenset())
    install = tmp_path / "install"
    _w(install / "MAST.exe", b"old")
    stage = tmp_path / "stage"
    man = stage_delta(tmp_path / "d.zip", stage, install)
    # extracted to STAGE, install untouched
    assert (stage / "MAST.exe").read_bytes() == b"NEW"
    assert (install / "MAST.exe").read_bytes() == b"old"
    assert man["to_version"] == "2"


def test_stage_delta_rejects_tampered_blob(tmp_path):
    from mast.update.delta import DeltaError, build_delta, stage_delta
    old, new = tmp_path / "old", tmp_path / "new"
    _w(old / "f", b"a")
    _w(new / "f", b"b")
    dz = tmp_path / "d.zip"
    build_delta(old, new, dz, from_version="1", to_version="2", exclude_top=frozenset())
    # corrupt the blob inside the delta zip
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(dz) as zin, zipfile.ZipFile(buf, "w") as zout:
        for i in zin.infolist():
            data = zin.read(i.filename)
            if i.filename == "files/f":
                data = b"TAMPERED"
            zout.writestr(i, data)
    dz.write_bytes(buf.getvalue())
    with pytest.raises(DeltaError):
        stage_delta(dz, tmp_path / "stage", tmp_path / "install")


def test_delta_is_data_only_gates():
    from mast.update.client import delta_is_data_only
    assert delta_is_data_only(["_internal/frontend/dist/assets/x.js", "docs/n.txt"]) is True
    assert delta_is_data_only(["MASTv2/artifacts/literature_index/vectors.npy"]) is True
    assert delta_is_data_only(["MAST.exe"]) is False
    assert delta_is_data_only(["_internal/base_library.zip"]) is False
    assert delta_is_data_only(["_internal/frontend/dist/x.js", "MAST.exe"]) is False  # mixed
    assert delta_is_data_only([]) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
