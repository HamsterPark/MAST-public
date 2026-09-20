"""Regression: timm must load the bundled backbone WITHOUT going through
huggingface_hub's cache resolution.

(the rig, v5.5.1): the bundled 86 MB weight file was
present and readable, yet every start logged::

    LocalEntryNotFoundError ... cannot find the requested files in the local cache
    VisionModule using MockBackend (VIGIL M12 load failed)

Root cause: ``huggingface_hub`` freezes its cache path into a MODULE-LEVEL
CONSTANT at import time. ``configure_backbone_cache`` set ``HF_HOME`` at *load*
time, far too late to move it, so timm looked in the (empty, nonexistent)
user-level cache. v5.5.0 and v5.5.1 both "fixed" the directory probing — which
was never broken; v5.5.1's own log line shows it found the directory and set
HF_HOME, and the load still failed.

The fix is to resolve the weight file ourselves and pass timm an explicit
``pretrained_cfg_overlay={"file": ...}``. These tests pin that the resolution
depends ONLY on the file existing — not on env vars, not on hub cache state.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.vision._vigil import backbone_cache as bc  # noqa: E402

TIMM_ID = "vit_small_patch16_dinov3.lvd1689m"
REV = "3bf4720a82ec2066db88137180ff1f83a675cef0"

_KEYS = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "HF_HUB_CACHE"]


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in _KEYS}
    for k in _KEYS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _make_cache(root: Path, *, timm_id: str = TIMM_ID, size: int = 2_000_000,
                name: str = "model.safetensors") -> Path:
    """Build a minimal HF-hub cache layout and return the weight file."""
    repo = root / "hub" / f"models--timm--{timm_id}"
    snap = repo / "snapshots" / REV
    snap.mkdir(parents=True)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(REV, encoding="utf-8")
    w = snap / name
    w.write_bytes(b"\0" * size)
    return w


# ── id normalisation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (TIMM_ID, TIMM_ID),
    (f"timm/{TIMM_ID}", TIMM_ID),
    (f"hf-hub:timm/{TIMM_ID}", TIMM_ID),
    ("  " + TIMM_ID + "  ", TIMM_ID),
    ("", ""),
])
def test_normalize_timm_id(raw, expected):
    assert bc.normalize_timm_id(raw) == expected


# ── weight-file resolution ──────────────────────────────────────────────────

def test_finds_weight_in_hub_layout(tmp_path):
    w = _make_cache(tmp_path)
    assert bc.local_weight_file(TIMM_ID, tmp_path) == w


def test_accepts_decorated_model_ids(tmp_path):
    w = _make_cache(tmp_path)
    assert bc.local_weight_file(f"hf-hub:timm/{TIMM_ID}", tmp_path) == w


def test_returns_none_when_repo_absent(tmp_path):
    (tmp_path / "hub").mkdir()
    assert bc.local_weight_file(TIMM_ID, tmp_path) is None


def test_returns_none_for_empty_weight_file(tmp_path):
    """A 0-byte placeholder must not be mistaken for real weights."""
    _make_cache(tmp_path, size=0)
    assert bc.local_weight_file(TIMM_ID, tmp_path) is None


def test_falls_back_to_blobs_when_snapshot_missing(tmp_path):
    """Copying an HF cache on Windows can leave snapshots/ dangling while the
    real bytes remain in blobs/. The scan fallback must still find them."""
    repo = tmp_path / "hub" / f"models--timm--{TIMM_ID}"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (repo / "snapshots" / REV).mkdir(parents=True)      # empty snapshot dir
    blob = blobs / "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0"
    blob.write_bytes(b"\0" * 2_000_000)
    assert bc.local_weight_file(TIMM_ID, tmp_path) == blob


# ── the actual regression ───────────────────────────────────────────────────

def test_resolution_ignores_hf_env_completely(tmp_path, monkeypatch):
    """THE root-cause pin: a poisoned/empty HF cache must not affect us.

    This is precisely the rig situation — user-level cache absent — under
    which v5.5.1 fell back to MockBackend.
    """
    w = _make_cache(tmp_path)
    os.environ["HF_HUB_CACHE"] = str(tmp_path / "__does_not_exist__" / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: "/no/such/root")

    assert bc.local_weight_file(TIMM_ID, tmp_path) == w
    kwargs = bc.timm_pretrained_kwargs(TIMM_ID, tmp_path)
    assert kwargs == {"pretrained_cfg_overlay": {"file": str(w)}}


def test_timm_kwargs_empty_when_nothing_bundled(tmp_path, monkeypatch):
    """No bundle → no overlay; caller then fails FAST offline (→ Mock),
    which is the documented degrade path, not a hang."""
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: str(tmp_path))
    assert bc.timm_pretrained_kwargs(TIMM_ID, None) == {}


def test_configure_returns_root_and_forces_offline(tmp_path):
    """configure() must hand back the root — that return value is what makes
    the load work; HF_HOME alone provably cannot (see module docstring)."""
    _make_cache(tmp_path)
    root = bc.configure(tmp_path)
    assert root == tmp_path
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HOME"] == str(tmp_path)


def test_hf_home_lets_later_callers_find_same_bundle(tmp_path, monkeypatch):
    """dinov_loader / quality_model resolve without being handed the path:
    configure() pinned HF_HOME, and resolve_cache_root honours it."""
    w = _make_cache(tmp_path)
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: "/no/such/root")
    bc.configure(tmp_path)
    assert bc.local_weight_file(TIMM_ID, None) == w


# ── call-site wiring ────────────────────────────────────────────────────────

def test_dinov_loader_passes_file_overlay_to_timm(tmp_path, monkeypatch):
    """The wiring that actually fixes the product: load_backbone must hand
    timm.create_model the explicit file overlay."""
    _make_cache(tmp_path)
    bc.configure(tmp_path)

    seen: dict = {}

    class _FakeTimm:
        @staticmethod
        def create_model(model_id, **kwargs):
            seen["model_id"] = model_id
            seen["kwargs"] = kwargs
            raise RuntimeError("stop-after-capture")

    monkeypatch.setitem(sys.modules, "timm", _FakeTimm)
    from mast.vision._vigil.backbone import dinov_loader

    with pytest.raises(RuntimeError, match="stop-after-capture"):
        dinov_loader.load_backbone("dinov3-vits16", in_channels=3, freeze=True)

    assert seen["model_id"] == TIMM_ID
    overlay = seen["kwargs"].get("pretrained_cfg_overlay")
    assert overlay is not None, "timm was called WITHOUT the bundled-file overlay"
    assert Path(overlay["file"]).is_file()
    assert seen["kwargs"]["pretrained"] is True
