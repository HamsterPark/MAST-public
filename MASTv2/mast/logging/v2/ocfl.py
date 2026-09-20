"""OCFL 1.1 long-term archival for a campaign or experiment.

Implements the minimal subset of the Oxford Common File Layout 1.1 needed
to produce a valid v1 ``Object`` directory:

    <output_root>/<object_id>/
      0=ocfl_object_1.1
      inventory.json
      inventory.json.sha512
      v1/
        content/
          <file>...
        inventory.json
        inventory.json.sha512

Spec: https://ocfl.io/1.1/spec/

The OCFL ``Object`` produced is **immutable from this point forward**.
Adding more versions (v2/, v3/, ...) is not implemented here — the use
case in MAST is to snapshot a published experiment at one moment.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from mast.logging.v2.repos import V2Repos
from mast.logging.v2.rocrate import build_metadata

logger = logging.getLogger(__name__)

OCFL_OBJECT_DECL = "ocfl_object_1.1"
DIGEST_ALG = "sha512"  # OCFL 1.1 default


def _sha512_file(path: Path) -> str:
    h = hashlib.sha512()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha512_bytes(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def export(
    repos: V2Repos,
    experiment_id: str,
    output_root: str | Path,
    *,
    object_id: str | None = None,
) -> Path:
    """Archive an experiment as an OCFL 1.1 Object directory.

    Returns the object dir path.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    exp = repos.experiments.get(experiment_id)
    if not exp:
        raise ValueError(f"experiment {experiment_id} not found")

    obj_id = object_id or f"mast-exp-{experiment_id}"
    obj_dir = output_root / obj_id
    if obj_dir.exists():
        raise FileExistsError(f"OCFL Object already exists at {obj_dir}")
    obj_dir.mkdir()
    (obj_dir / f"0={OCFL_OBJECT_DECL}").write_bytes(b"")

    # ── v1 directory ──────────────────────────────────────────────────
    v1_dir = obj_dir / "v1"
    content_dir = v1_dir / "content"
    content_dir.mkdir(parents=True)

    # 1. Write a JSON-LD metadata file (we reuse RO-Crate output).
    metadata = build_metadata(repos, experiment_id)
    metadata_path = content_dir / "ro-crate-metadata.json"
    metadata_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)

    # 2. Copy scan files into content/files/<sha256> (CAS-aligned naming).
    files_dir = content_dir / "files"
    files_dir.mkdir(exist_ok=True)
    actions = repos.actions.for_experiment(experiment_id, limit=100_000)
    sf_seen: set[str] = set()
    for a in actions:
        for sf in repos.scan_files.for_action(a["id"]):
            sha = sf["sha256"]
            if sha in sf_seen:
                continue
            sf_seen.add(sha)
            src = Path(sf["current_path"])
            if not src.is_file():
                logger.warning("OCFL: file missing for sha %s: %s", sha, src)
                continue
            shutil.copy2(src, files_dir / sha)

    # ── Inventory generation ──────────────────────────────────────────
    inventory = _build_inventory(obj_id, obj_dir, version_name="v1")
    inventory_bytes = json.dumps(inventory, indent=2, ensure_ascii=False).encode("utf-8")
    sidecar = f"{_sha512_bytes(inventory_bytes)} inventory.json\n"

    # Write at v1/ and root level (OCFL requires identical files in both).
    for d in (v1_dir, obj_dir):
        (d / "inventory.json").write_bytes(inventory_bytes)
        (d / f"inventory.json.{DIGEST_ALG}").write_text(sidecar, encoding="utf-8")

    return obj_dir


def _build_inventory(object_id: str, obj_dir: Path, *, version_name: str = "v1") -> dict:
    """Walk <obj_dir>/<version>/content/ and assemble OCFL inventory.json.

    Manifest paths are relative to the OBJECT root (per OCFL §3.5.2).
    Logical state paths are relative to the version's content/ root.
    """
    version_dir = obj_dir / version_name
    content_root = version_dir / "content"
    manifest: dict[str, list[str]] = {}
    state: dict[str, list[str]] = {}
    for p in content_root.rglob("*"):
        if not p.is_file():
            continue
        digest = _sha512_file(p)
        rel = p.relative_to(obj_dir).as_posix()
        manifest.setdefault(digest, []).append(rel)
        logical = p.relative_to(content_root).as_posix()
        state.setdefault(digest, []).append(logical)

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    # Convert "+0800" → "+08:00" per OCFL ISO requirements.
    if len(created) >= 5 and (created[-5] == "+" or created[-5] == "-"):
        created = created[:-2] + ":" + created[-2:]

    return {
        "id": object_id,
        "type": "https://ocfl.io/1.1/spec/#inventory",
        "digestAlgorithm": DIGEST_ALG,
        "head": version_name,
        "contentDirectory": "content",
        "manifest": manifest,
        "versions": {
            version_name: {
                "created": created,
                "message": "Initial MAST experiment archive",
                "user": {"name": "MAST", "address": "mailto:noreply@local"},
                "state": state,
            }
        },
    }


def verify(object_dir: str | Path) -> tuple[bool, list[str]]:
    """Lightweight inventory consistency check. Returns (ok, errors)."""
    obj_dir = Path(object_dir)
    errors: list[str] = []
    if not (obj_dir / f"0={OCFL_OBJECT_DECL}").exists():
        errors.append("missing 0=ocfl_object_1.1 declaration")
    inv_path = obj_dir / "inventory.json"
    if not inv_path.exists():
        errors.append("missing inventory.json")
        return False, errors
    try:
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"bad inventory.json: {exc}"]
    head = inv.get("head")
    if not head or not (obj_dir / head).is_dir():
        errors.append(f"head version directory missing: {head}")
        return False, errors
    for digest, paths in inv.get("manifest", {}).items():
        for rel in paths:
            f = obj_dir / rel
            if not f.is_file():
                errors.append(f"missing manifest file: {rel}")
                continue
            got = _sha512_file(f)
            if got != digest:
                errors.append(f"digest mismatch for {rel}: {got} != {digest}")
    return (not errors), errors
