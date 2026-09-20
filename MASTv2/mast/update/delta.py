"""File-level incremental (delta) updates for the MAST onedir bundle.

The full installer is ~154 MB; most releases change only a handful of Python
files under ``_internal/``. A delta package ships just the *changed* + *added*
files plus a list of *removed* paths, so a client on version N can move to N+1
by downloading kilobytes instead of the whole installer (user request: 在发布
新版时使用增量式更新).

Format — a single zip containing:
  * ``delta_manifest.json`` — {from_version, to_version, added[], changed[],
    removed[], sha256{relpath: hex}} where sha256 covers every added+changed
    file (so the client verifies each one before writing it).
  * ``files/<relpath>`` — the bytes of each added/changed file.

:func:`apply_delta` writes each file atomically (temp + os.replace), verifies
its sha256, and deletes removed paths. If any hash mismatches it aborts BEFORE
touching the target, so a tampered/corrupt delta can't half-patch the install
(the client then falls back to the full installer).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

logger = logging.getLogger(__name__)

DELTA_MANIFEST = "delta_manifest.json"
_FILES_PREFIX = "files/"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _content_sha256(p: Path) -> str:
    """A hash STABLE across build non-determinism, used only to DECIDE what a
    delta must ship (not to verify downloads).

    PyInstaller re-zips ``base_library.zip`` with fresh member timestamps every
    build, so its raw sha256 differs even when the CONTENT is identical — which
    made EVERY code delta drag in base_library.zip and force a restart (feature B,
    审查). For a ``.zip`` we hash the sorted (member-name,
    member-bytes) instead of the container bytes, so a timestamp-only rebuild
    compares EQUAL and is left out of the delta. Non-zip files use the raw sha256.
    """
    if p.suffix.lower() == ".zip":
        try:
            h = hashlib.sha256()
            with zipfile.ZipFile(p) as z:
                for name in sorted(z.namelist()):
                    h.update(name.encode("utf-8", "surrogatepass"))
                    h.update(b"\0")
                    h.update(z.read(name))
            return "zipnorm:" + h.hexdigest()
        except Exception:  # not a valid/readable zip → fall back to raw bytes
            return _sha256_file(p)
    return _sha256_file(p)


# Top-level names that are USER DATA / install bookkeeping, never program files.
# Excluded from the delta diff so an incremental update can NEVER delete or
# touch the user's keys / experiments / models, or the Inno uninstaller.
DEFAULT_EXCLUDE_TOP = frozenset({
    "api key", "config", "experiments", "models", "working-sessions",
    "logs", "data", "data_dir.txt", "unins000.dat", "unins000.exe",
    "MASTv2",  # whole subtree excluded — EXCEPT the include-prefixes below
    # ── Runtime state that the SMOKE TEST itself creates (2026-08-05) ──────
    #
    # The release procedure says to smoke-test by running `dist\MAST\MAST.exe`
    # in place. That RUNS the app inside the very directory the delta is built
    # from, so it leaves runtime state behind — and the next delta ships it.
    #
    # Caught on 6.1.1→6.1.2: the manifest listed `.mast2_launcher.pid`,
    # `vision_selftest_result.txt` and `artifacts/diagnostics/refusals.jsonl`
    # as ADDED files. A stale PID file pushed onto an operator's machine is the
    # worst of the three — the launcher reads it to decide whether an instance
    # is already running.
    #
    # `experiments/` was already excluded and did catch its five DB files, which
    # is why nobody noticed the gap: the exclusion list covered the *biggest*
    # runtime directory and looked complete.
    #
    # NOTE the asymmetry: the SHIPPED vision weights and literature index live
    # under `MASTv2/artifacts/…` (excluded top + re-included prefix). The
    # top-level `artifacts/` here is a different path — runtime only.
    "artifacts",              # diagnostics / vision_frames / refusal ledgers
    ".mast2_launcher.pid",    # launcher instance lock
    "vision_selftest_result.txt",   # written by MAST.exe --selftest-vision
    "pyruntime_selftest_result.txt",  # written by MAST.exe --selftest-pyruntime
    #
    # NOTE on MASTv2/pyruntime (the DP analysis runtime, ~335 MB): it needs no
    # entry here — the whole `MASTv2` subtree above already excludes it, and it
    # is deliberately NOT in DEFAULT_INCLUDE_PREFIXES. Same reasoning as the
    # DINOv3 backbone: a pinned interpreter + pinned wheels is STATIC, so it
    # ships only in the full installer and is never re-pushed through a delta.
    # The accepted cost is that updating it requires a full install.
})

# Paths UNDER an excluded top that should nevertheless be delta-tracked. The
# literature library (vectors/metadata/classified/abstracts) is a BINARY-shipped
# asset, not user data, so when it's refreshed it should ship via the OTA delta
# (incrementally, file-level) instead of forcing a full ~380MB reinstall. The
# VIGIL vision checkpoint (~29 MB) and the classical-quality joblib are likewise
# binary assets that may be refreshed by a new training run — small enough to
# delta. The rest of MASTv2/ stays excluded — it is user/runtime data a delta
# must NEVER touch (artifacts/literature_libs, artifacts/tts_cache,
# artifacts/buffer.wal*), AND the ~165 MB DINOv3 backbone cache
# (artifacts/vision_backbone) which is STATIC: it never changes, so it ships only
# in the full installer and is never re-pushed through a delta (the client falls
# back to the full installer when no delta path exists).
#
# 2026-08-02: was still naming ``mast_vision_m12.pt``, retired when VIGIL v2.5
# (DINOv3-vits16 + 6 heads) went in. A stale name here fails silently — the new
# checkpoint is simply never delta-pushed, so an OTA-upgraded client keeps
# running the old weights while reporting the new version.
DEFAULT_INCLUDE_PREFIXES = frozenset({
    "MASTv2/artifacts/literature_index",
    "MASTv2/artifacts/mast_vision_v25.pt",
    "MASTv2/artifacts/stm_quality_v1_dino.joblib",
})


def _is_excluded(rel: str, exclude_top, include_prefixes) -> bool:
    """A file is excluded iff its top component is in *exclude_top* AND it is not
    under an explicitly-included prefix (include wins over the blanket exclude)."""
    if rel.split("/", 1)[0] not in exclude_top:
        return False
    for inc in include_prefixes:
        if rel == inc or rel.startswith(inc + "/"):
            return False  # binary-shipped asset under an excluded top → track it
    return True


def hash_tree(root: Path, exclude_top: frozenset[str] | set[str] | None = None,
              include_prefixes: frozenset[str] | set[str] | None = None,
              *, hasher=_sha256_file) -> dict[str, str]:
    """Map every file under *root* to a hash, keyed by POSIX relpath.

    ``exclude_top`` (first-path-component match) skips user-data / bookkeeping
    trees so they never enter a delta; ``include_prefixes`` carves binary-shipped
    assets (the literature index) back IN even when they live under an excluded
    top (so the big library can ship via delta). ``hasher`` selects the hash:
    :func:`_sha256_file` (raw, for verification) or :func:`_content_sha256`
    (build-noise-stable, for the diff comparison)."""
    root = Path(root)
    exclude_top = exclude_top or frozenset()
    include_prefixes = include_prefixes or frozenset()
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if _is_excluded(rel, exclude_top, include_prefixes):
                continue
            out[rel] = hasher(p)
    return out


def diff_trees(old_root: Path, new_root: Path,
               exclude_top: frozenset[str] | set[str] | None = None,
               include_prefixes: frozenset[str] | set[str] | None = None) -> dict:
    """Return {added, changed, removed, new_hashes} comparing two trees.

    The added/changed/removed decision uses the CONTENT-normalized hash
    (:func:`_content_sha256`) so a rebuild that only re-timestamps a zip
    (base_library.zip) is NOT flagged changed. ``new_hashes`` (which becomes the
    manifest's verification sha256) is the RAW file sha256 — the client verifies
    the raw blob it downloads, so it must match the file byte-for-byte."""
    old = hash_tree(old_root, exclude_top, include_prefixes, hasher=_content_sha256)
    new = hash_tree(new_root, exclude_top, include_prefixes, hasher=_content_sha256)
    added = sorted(p for p in new if p not in old)
    removed = sorted(p for p in old if p not in new)
    changed = sorted(p for p in new if p in old and new[p] != old[p])
    new_root = Path(new_root)
    new_raw = {rel: _sha256_file(new_root / rel) for rel in (added + changed)}
    return {"added": added, "changed": changed, "removed": removed,
            "new_hashes": new_raw}


def build_delta(old_root: Path, new_root: Path, out_zip: Path, *,
                from_version: str, to_version: str,
                exclude_top: frozenset[str] | set[str] | None = DEFAULT_EXCLUDE_TOP,
                include_prefixes: frozenset[str] | set[str] | None = DEFAULT_INCLUDE_PREFIXES) -> dict:
    """Build a delta zip taking *old_root* → *new_root*. Returns the manifest.

    ``exclude_top`` defaults to :data:`DEFAULT_EXCLUDE_TOP` so diffing an
    installed root (which mixes program files with user data) against a clean
    build never touches the user's data. ``include_prefixes`` (default
    :data:`DEFAULT_INCLUDE_PREFIXES`) carves the binary-shipped literature index
    back in so a refreshed big library ships incrementally via the delta. Pass
    ``frozenset()`` to either to diff raw."""
    old_root, new_root, out_zip = Path(old_root), Path(new_root), Path(out_zip)
    d = diff_trees(old_root, new_root, exclude_top, include_prefixes)
    payload = d["added"] + d["changed"]
    manifest = {
        "from_version": from_version,
        "to_version": to_version,
        "added": d["added"],
        "changed": d["changed"],
        "removed": d["removed"],
        "sha256": {rel: d["new_hashes"][rel] for rel in payload},
    }
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(DELTA_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        for rel in payload:
            z.write(new_root / rel, _FILES_PREFIX + rel)
    logger.info("Built delta %s→%s: +%d ~%d -%d (%s)", from_version, to_version,
                len(d["added"]), len(d["changed"]), len(d["removed"]), out_zip.name)
    return manifest


class DeltaError(RuntimeError):
    pass


def _safe_dest(target_root: Path, rel: str) -> Path:
    """Resolve ``target_root / rel`` and ensure it stays INSIDE *target_root*.

    ``rel`` comes from the delta manifest payload, which is attacker-influenceable
    (the OTA channel is sha256-only with no signature / MITM defence — see
    update/client.py). Phase 1 only verifies each blob's sha256; nothing stops a
    crafted manifest from listing ``rel='../../../evil.py'`` (or an absolute path,
    or a drive-letter / UNC path on Windows) to escape the install root and write
    or delete files anywhere the process can reach (path traversal → arbitrary
    write / delete). Reject any rel that does not normalise to a path strictly
    under *target_root*.
    """
    if not rel or not isinstance(rel, str):
        raise DeltaError(f"empty/invalid relpath in delta: {rel!r}")
    if "\x00" in rel:
        raise DeltaError(f"NUL in delta relpath: {rel!r}")
    # Reject absolute paths and Windows drive / UNC anchors outright (these would
    # ignore target_root entirely when joined).
    pr = PurePosixPath(rel)
    pw = PureWindowsPath(rel)
    if pr.is_absolute() or pw.is_absolute() or pw.drive or pw.anchor:
        raise DeltaError(f"absolute/anchored relpath not allowed in delta: {rel!r}")
    root = target_root.resolve()
    dest = (root / rel).resolve()
    # dest must be root itself? no — it must be strictly under root.
    if dest == root or root not in dest.parents:
        raise DeltaError(f"delta path escapes target root: {rel!r}")
    return dest


def read_delta_manifest(delta_zip: Path) -> dict:
    with zipfile.ZipFile(delta_zip) as z:
        return json.loads(z.read(DELTA_MANIFEST).decode("utf-8"))


def apply_delta(delta_zip: Path, target_root: Path) -> dict:
    """Apply *delta_zip* to *target_root* in place.

    Verifies every added/changed file's sha256 from the manifest BEFORE writing
    anything; on any mismatch raises :class:`DeltaError` without modifying the
    target (so the caller can fall back to a full install). Each accepted file
    is then written atomically (temp + os.replace); removed paths are deleted.
    """
    delta_zip, target_root = Path(delta_zip), Path(target_root)
    if not target_root.exists():
        raise DeltaError(f"target {target_root} does not exist")
    with zipfile.ZipFile(delta_zip) as z:
        names = set(z.namelist())
        if DELTA_MANIFEST not in names:
            raise DeltaError("delta missing manifest")
        manifest = json.loads(z.read(DELTA_MANIFEST).decode("utf-8"))
        payload = list(manifest.get("added", [])) + list(manifest.get("changed", []))
        expect = manifest.get("sha256", {})

        # Phase 1 — verify EVERYTHING before touching the target: both the
        # blob sha256 AND that each relpath stays inside target_root (path
        # traversal defence — rel is attacker-influenceable). Abort on the first
        # problem so a tampered delta can't half-patch / escape the install.
        staged: dict[Path, bytes] = {}
        for rel in payload:
            dest = _safe_dest(target_root, rel)  # raises DeltaError if it escapes
            arc = _FILES_PREFIX + rel
            if arc not in names:
                raise DeltaError(f"delta missing file blob for {rel}")
            data = z.read(arc)
            want = expect.get(rel, "")
            if not want or _sha256_bytes(data) != want.lower():
                raise DeltaError(f"sha256 mismatch for {rel}")
            staged[dest] = data

        # Validate removed paths too (a crafted manifest could delete arbitrary
        # files via ``removed: ['../../something']``). Resolve them now, before
        # any write, so the whole delta is rejected up front on traversal.
        removed_dests: list[Path] = [
            _safe_dest(target_root, rel) for rel in manifest.get("removed", [])
        ]

    # Phase 2 — write atomically + delete removed (paths already validated).
    written, removed = 0, 0
    for dest, data in staged.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=".dl_", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, str(dest))
            written += 1
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    for dest in removed_dests:
        try:
            if dest.is_file():
                dest.unlink()
                removed += 1
        except OSError as exc:  # pragma: no cover
            logger.warning("delta: could not remove %s: %s", dest, exc)
    result = {"written": written, "removed": removed,
              "from_version": manifest.get("from_version"),
              "to_version": manifest.get("to_version")}
    logger.info("Applied delta → %s: wrote %d, removed %d",
                result["to_version"], written, removed)
    return result


def stage_delta(delta_zip: Path, stage_dir: Path, install_root: Path) -> dict:
    """Verify a delta and extract its added/changed blobs into *stage_dir* WITHOUT
    touching *install_root* — for an OFFLINE apply (a delta touching the running
    exe / base_library.zip can't be written in place; the launcher's self-updater
    copies stage_dir → install_root AFTER MAST exits; feature B, 2026-07-03).

    Verification mirrors :func:`apply_delta` phase 1: every blob's sha256 is
    checked, and every added/changed/removed relpath is resolved against
    *install_root* (the REAL target) so a crafted manifest can't traverse out.
    Raises :class:`DeltaError` on any bad hash / path so the caller can fall back
    to the full installer. Returns the delta manifest (added/changed/removed)."""
    delta_zip, stage_dir, install_root = Path(delta_zip), Path(stage_dir), Path(install_root)
    with zipfile.ZipFile(delta_zip) as z:
        names = set(z.namelist())
        if DELTA_MANIFEST not in names:
            raise DeltaError("delta missing manifest")
        manifest = json.loads(z.read(DELTA_MANIFEST).decode("utf-8"))
        payload = list(manifest.get("added", [])) + list(manifest.get("changed", []))
        expect = manifest.get("sha256", {})
        staged: dict[str, bytes] = {}
        for rel in payload:
            _safe_dest(install_root, rel)  # traversal check vs the REAL target root
            arc = _FILES_PREFIX + rel
            if arc not in names:
                raise DeltaError(f"delta missing file blob for {rel}")
            data = z.read(arc)
            want = expect.get(rel, "")
            if not want or _sha256_bytes(data) != want.lower():
                raise DeltaError(f"sha256 mismatch for {rel}")
            staged[rel] = data
        for rel in manifest.get("removed", []):
            _safe_dest(install_root, rel)  # removed paths validated too

    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    for rel, data in staged.items():
        dest = stage_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    logger.info("Staged delta %s→%s: %d files in %s",
                manifest.get("from_version"), manifest.get("to_version"),
                len(staged), stage_dir)
    return manifest


__all__ = [
    "hash_tree", "diff_trees", "build_delta", "apply_delta", "stage_delta",
    "read_delta_manifest", "DeltaError", "DELTA_MANIFEST",
]
