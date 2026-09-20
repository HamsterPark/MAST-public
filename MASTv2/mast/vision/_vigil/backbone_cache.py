"""Bundled DINOv3 backbone cache — path resolution + offline weight handoff.

WHY THIS MODULE EXISTS (the rig)
-----------------------------------------------------------
``huggingface_hub`` freezes its cache location into MODULE-LEVEL CONSTANTS at
import time (``huggingface_hub.constants.HF_HUB_CACHE``). Setting ``HF_HOME``
afterwards — which is exactly what :func:`configure_backbone_cache` relied on —
**cannot move it**. Measured directly::

    import huggingface_hub                      # constant frozen here
    os.environ["HF_HOME"] = <bundled cache>     # too late
    huggingface_hub.constants.HF_HUB_CACHE      # -> still %USERPROFILE%\\.cache\\...

On the rig the user-level cache did not exist at all, so ``timm`` looked in
``C:\\Users\\<user>\\.cache\\huggingface\\hub``, found nothing, and — being
forced offline — raised ``LocalEntryNotFoundError``. The vision backend then
silently degraded to ``MockBackend`` while the bundled 86 MB weight file sat
right there, intact and readable.

v5.5.0 and v5.5.1 both tried to fix the *probing* of the cache directory. The
probing was never the problem: v5.5.1's log line proves it found the directory
and set ``HF_HOME`` — and the load still failed.

THE FIX: bypass huggingface_hub's cache resolution entirely. Locate the weight
file ourselves and hand timm an explicit ``pretrained_cfg_overlay={"file": …}``.
Verified to load 21,586,944 params with ``HF_HUB_CACHE`` pointed at a
non-existent directory AND ``HF_HUB_OFFLINE=1`` — i.e. it depends on neither the
environment variables nor the hub cache layout, only on the file existing.

The env-var pinning is KEPT as a belt-and-braces fallback: it still helps any
code path that reaches the hub through huggingface_hub before we get a chance
to overlay a file (and it is what makes the offline failure fast rather than a
minutes-long network retry-hang).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the bundled cache lives relative to the exe dir / project root.
_REL = Path("MASTv2/artifacts/vision_backbone")

# Weight file names timm/HF may use, in preference order.
_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin")


def force_offline() -> None:
    """Forbid any network fetch by timm/transformers.

    Applied unconditionally and FIRST: a blocked huggingface.co (restricted or
    air-gapped network) otherwise makes ``timm.create_model(pretrained=True)``
    retry-hang for minutes, the background vision warm never signals
    DONE/FAILED, and the launcher's vision watcher times out — the service is
    actually up and serving but the user sees "启动不起来".
    Offline fails FAST, which the callers fail-safe to Mock.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_cache_root(cache_dir: str | Path | None = None) -> Path | None:
    """Locate the directory HOLDING the ``hub/models--timm--…`` tree.

    The caller may hand us a path that doesn't resolve on this install (an OTA-patched install whose project_root is a SEPARATE data
    dir → the caller's resolve missed the exe-dir copy). So probe the caller's
    path first, THEN the standard bundled locations.
    """
    candidates: list[Path] = []
    if cache_dir:
        candidates.append(Path(cache_dir))
    # An earlier configure() pinned HF_HOME to the resolved root — honour it so
    # later call sites (dinov_loader, quality_model) find the SAME bundle
    # without every caller having to thread the path through.
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        candidates.append(Path(hf_home))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / _REL)
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.append(Path(meipass) / _REL)
    try:
        from mast._runtime_paths import project_root
        candidates.append(Path(project_root()) / _REL)
    except Exception:  # noqa: BLE001 — path helper is best-effort
        pass

    for cand in candidates:
        try:
            if (cand / "hub").is_dir():
                return cand
        except OSError:
            continue
    logger.warning(
        "backbone cache not located (tried %s)", [str(c) for c in candidates]
    )
    return None


def normalize_timm_id(model_id: str) -> str:
    """Strip ``hf-hub:`` / ``timm/`` decorations off a timm model identifier."""
    mid = (model_id or "").strip()
    if mid.startswith("hf-hub:"):
        mid = mid[len("hf-hub:"):]
    if mid.startswith("timm/"):
        mid = mid[len("timm/"):]
    return mid


def local_weight_file(
    model_id: str, cache_root: str | Path | None = None
) -> Path | None:
    """Return the on-disk weight file for *model_id*, or None if not bundled.

    Reads the HF-hub cache layout directly:
    ``<root>/hub/models--timm--<id>/snapshots/<revision>/model.safetensors``.
    ``refs/main`` is honoured when present, otherwise any snapshot is accepted.
    Falls back to a recursive scan of the repo dir, which also covers a cache
    whose ``snapshots`` entries are broken symlinks (a real risk on Windows,
    where copying an HF cache without dereferencing produces dangling links).
    """
    root = Path(cache_root) if cache_root is not None else resolve_cache_root()
    if root is None:
        return None
    timm_id = normalize_timm_id(model_id)
    if not timm_id:
        return None

    repo = root / "hub" / f"models--timm--{timm_id}"
    if not repo.is_dir():
        logger.debug("no bundled repo dir for %s at %s", timm_id, repo)
        return None

    snapshots = repo / "snapshots"
    revisions: list[Path] = []
    ref_main = repo / "refs" / "main"
    try:
        if ref_main.is_file():
            rev = ref_main.read_text(encoding="utf-8").strip()
            if rev:
                revisions.append(snapshots / rev)
    except OSError:
        pass
    try:
        if snapshots.is_dir():
            revisions.extend(d for d in snapshots.iterdir() if d.is_dir())
    except OSError:
        pass

    for rev_dir in revisions:
        for name in _WEIGHT_NAMES:
            f = rev_dir / name
            try:
                # is_file() is False for a DANGLING symlink — exactly what we
                # want to reject here, so the scan fallback can find the blob.
                if f.is_file() and f.stat().st_size > 0:
                    return f
            except OSError:
                continue

    # Fallback: any real weight-looking file anywhere under the repo dir
    # (covers dangling snapshot symlinks — blobs/ still holds the real bytes).
    try:
        best: Path | None = None
        for f in repo.rglob("*"):
            try:
                if not f.is_file():
                    continue
                if f.suffix in (".safetensors", ".bin") or f.parent.name == "blobs":
                    if f.stat().st_size > 1_000_000 and (
                        best is None or f.stat().st_size > best.stat().st_size
                    ):
                        best = f
            except OSError:
                continue
        if best is not None:
            logger.info("bundled weight for %s found via scan: %s", timm_id, best)
            return best
    except OSError:
        pass
    return None


def timm_pretrained_kwargs(
    model_id: str, cache_root: str | Path | None = None
) -> dict:
    """kwargs to add to ``timm.create_model`` so it loads from the bundle.

    Returns ``{"pretrained_cfg_overlay": {"file": "<abs path>"}}`` when the
    weight file is present, else ``{}`` (caller then relies on the HF cache /
    env-var path, and fails fast offline if that is empty too).
    """
    f = local_weight_file(model_id, cache_root)
    if f is None:
        logger.warning(
            "no bundled weight file for %s — falling back to huggingface_hub "
            "cache resolution (offline; will fail fast if the cache is empty)",
            model_id,
        )
        return {}
    logger.info("timm weights for %s ← %s (direct file, no hub lookup)", model_id, f)
    return {"pretrained_cfg_overlay": {"file": str(f)}}


def configure(cache_dir: str | Path | None) -> Path | None:
    """Force offline, pin HF_HOME when possible, and return the cache root.

    Order matters: offline FIRST (see :func:`force_offline`), then best-effort
    ``HF_HOME``. The return value is what callers should thread into
    :func:`timm_pretrained_kwargs` — that is the part that actually works.
    """
    force_offline()
    root = resolve_cache_root(cache_dir)
    if root is not None:
        # Kept as a fallback only — this CANNOT retarget an already-imported
        # huggingface_hub (see module docstring); the file overlay is the fix.
        os.environ["HF_HOME"] = str(root)
        logger.info("DINOv3 backbone cache → %s (offline)", root)
    return root


__all__ = [
    "configure",
    "force_offline",
    "local_weight_file",
    "normalize_timm_id",
    "resolve_cache_root",
    "timm_pretrained_kwargs",
]
