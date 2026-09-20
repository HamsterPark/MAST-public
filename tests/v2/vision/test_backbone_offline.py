"""Regression: the DINOv3 backbone cache config must force HF/timm OFFLINE even
when the bundled cache can't be located.

on a restricted network, configure_backbone_cache
early-returned when the cache dir wasn't found → timm.create_model(pretrained=True)
retry-hung on huggingface.co for minutes → the launcher's vision-load watcher
timed out → user saw "启动不起来" (the service was actually up). Offline must be
forced UNCONDITIONALLY so a missing cache fails FAST (→ Mock) instead of hanging.
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

from mast.vision._vigil import m12, v25  # noqa: E402

_KEYS = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME"]


@pytest.fixture(autouse=True)
def _restore_env():
    saved = {k: os.environ.get(k) for k in _KEYS}
    for k in _KEYS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.parametrize("configure", [v25.configure_backbone_cache, m12.configure_backbone_cache])
def test_offline_forced_with_no_cache_arg(configure, monkeypatch):
    # No cache reachable anywhere → offline must STILL be forced (fail-fast, not hang).
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: str("/no/such/root"))
    configure(None)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_v25_offline_forced_when_nothing_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: str(tmp_path))
    v25.configure_backbone_cache(tmp_path / "does_not_exist")
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert "HF_HOME" not in os.environ           # no hub/ anywhere → not pinned


def test_m12_offline_forced_when_cache_dir_missing(tmp_path, monkeypatch):
    # v5.5.2: m12 delegates to the SAME robust probing as v25 (shared
    # backbone_cache module), so project_root must be pointed at an empty tree —
    # otherwise the repo's own bundled cache is legitimately found and pinned.
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: str(tmp_path))
    m12.configure_backbone_cache(tmp_path / "does_not_exist")
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert "HF_HOME" not in os.environ


def test_v25_hf_home_pinned_from_caller_path(tmp_path):
    (tmp_path / "hub").mkdir()
    v25.configure_backbone_cache(tmp_path)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("HF_HOME") == str(tmp_path)


def test_v25_probes_project_root_when_caller_path_bad(tmp_path, monkeypatch):
    """The 2026-07-25 fix: caller's path is unresolvable (OTA install with a
    separate data-dir project_root), but the bundled cache IS reachable — probe
    project_root and pin HF_HOME so timm loads offline instead of failing to Mock."""
    cache = tmp_path / "MASTv2" / "artifacts" / "vision_backbone"
    (cache / "hub").mkdir(parents=True)
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: str(tmp_path))
    v25.configure_backbone_cache(None)           # caller resolved nothing
    assert os.environ.get("HF_HOME") == str(cache)
