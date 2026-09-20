"""Manifest schema for the MAST intranet push-update system.

vendored from v1 mast/update/manifest.py — byte-for-byte identical so
admin tooling, clients, and on-disk manifest formats can be shared between
v1 and v2 (token/server addresses are configurable per machine).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    """One published MAST version's metadata.

    ``deltas`` lists incremental update packages keyed by the version they
    upgrade FROM, so a client already on a recent version downloads a small
    delta instead of the full ``filename`` installer (user request: 同时有安装包
    和增量更新版). Each entry: {from_version, filename, sha256, size_bytes}.
    A client that can't find/apply a matching delta falls back to ``filename``.
    Backward-compatible: from_dict filters unknown keys, so v1 clients ignore it.
    """

    version: str
    filename: str
    sha256: str
    size_bytes: int
    published_at: str
    release_notes_url: str = ""
    min_force_install: str = "0.0.0"
    schema_version: int = 1
    deltas: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def compute_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def load_manifest(path: Path) -> Manifest | None:
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return Manifest.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def write_manifest(path: Path, m: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(m.to_json(), encoding="utf-8")
    tmp.replace(path)


def make_manifest(
    setup_path: Path, version: str, *, release_notes_url: str = "",
    min_force_install: str = "0.0.0",
) -> Manifest:
    return Manifest(
        version=version,
        filename=setup_path.name,
        sha256=compute_sha256(setup_path),
        size_bytes=setup_path.stat().st_size,
        published_at=datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds"),
        release_notes_url=release_notes_url,
        min_force_install=min_force_install,
    )


def make_delta_descriptor(delta_path: Path, from_version: str) -> dict:
    """Manifest entry for an incremental update package."""
    delta_path = Path(delta_path)
    return {
        "from_version": from_version,
        "filename": delta_path.name,
        "sha256": compute_sha256(delta_path),
        "size_bytes": delta_path.stat().st_size,
    }


def find_delta(m: Manifest, from_version: str) -> dict | None:
    """Return the delta entry that upgrades FROM *from_version*, or None."""
    for d in (m.deltas or []):
        if isinstance(d, dict) and d.get("from_version") == from_version:
            return d
    return None


def version_tuple(s: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in s.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(remote: str, local: str) -> bool:
    return version_tuple(remote) > version_tuple(local)
