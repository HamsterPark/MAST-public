"""Push-update client — daemon thread inside MAST's main service.

vendored from v1 mast/update/client.py — same wire protocol, same disk
layout, can target the same push server as v1 (a single admin can publish
both the legacy and current MAST releases from the same machine if they keep version numbers
distinct).
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PENDING_DIR_NAME = "pending_update"
TOKEN_FILE = "update_client_token.env"
SERVER_URL_FILE = "update_server_url.env"


def _is_safe_filename(name: str) -> bool:
    """True only for a plain filename (no path separators, no traversal).

    The server-supplied manifest is attacker-influenceable (default protocol is
    plaintext HTTP and the threat model explicitly includes MITM — see the
    sha256 comment below). A malicious manifest with
    ``filename='..\\..\\evil.exe'`` would otherwise let the client write the
    downloaded payload (and the .tmp) anywhere relative to the pending dir.
    Reject anything that is not a bare basename.
    """
    if not name or not isinstance(name, str):
        return False
    # No path separators (either platform), no NUL, no traversal, no
    # leading dot (which could hide files or denote relative segments).
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    if name.startswith("."):
        return False
    # os.path.basename strips any directory component; if the result differs,
    # the name carried a path. (Belt-and-suspenders with the checks above.)
    if os.path.basename(name) != name:
        return False
    # ntpath also treats a drive letter / UNC prefix as a path; guard against
    # 'C:evil.exe' or '\\\\host\\share' style names on Windows clients.
    import ntpath
    if ntpath.basename(name) != name or ntpath.splitdrive(name)[0]:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# SUPPLY-CHAIN SECURITY GAP — manifest/payload authenticity is NOT verified.
#
# TODO(security, needs key infrastructure): the OTA channel currently trusts a
# Bearer token for AUTHORIZATION and an end-to-end sha256 for INTEGRITY, but has
# NO cryptographic AUTHENTICITY check. The manifest (version, filename,
# size_bytes, sha256, deltas[]) is consumed as-is from the server, and every
# downloaded blob is verified ONLY against the sha256 *in that same manifest*.
# So an attacker who can rewrite the response (MITM, a compromised/poisoned push
# server, or DNS hijack) simply ships malicious bytes WITH a matching sha256 and
# the client installs them — the sha256 proves the bytes match the manifest, not
# that the manifest came from us. This must be closed with a real signing chain:
#
#   1. Generate an offline Ed25519 (or RSA-PSS) release keypair; the PRIVATE key
#      stays offline with the release signer, NEVER on the push server.
#   2. At publish time, sign the canonicalised manifest bytes; ship the signature
#      alongside (e.g. manifest.json + manifest.sig).
#   3. EMBED the PUBLIC key in this client binary (so a MITM cannot swap it) and,
#      before trusting ANY field below, verify the signature over the manifest.
#      Only then do the existing sha256/size checks become meaningful.
#
# Deliberately NOT stubbing a fake/"self-signed-by-the-server" signature here —
# that would give false confidence. Designing + provisioning the keypair and the
# publish-side signing is a human task (see also delta.apply_delta, which also
# trusts manifest-supplied relpaths and now defends path traversal but still
# cannot attest authenticity without this signing chain). Until then, the two
# defences below shrink the window: (a) reject plaintext http:// server URLs so
# transport encryption (TLS) raises the bar for a passive/active MITM, and
# (b) keep the strict path/filename traversal guards.
# ──────────────────────────────────────────────────────────────────────


_INSECURE_UPDATE_ENV = "MAST2_ALLOW_INSECURE_UPDATE"


def _is_private_host(host: str) -> bool:
    """True for loopback / RFC1918 / link-local / unique-local hosts (a network
    where a passive internet MITM cannot sit). Non-IP hostnames → False (treated
    as public; they must use HTTPS)."""
    h = (host or "").strip().strip("[]").lower()
    if h in ("localhost",):
        return True
    try:
        import ipaddress
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


def _require_https(server_url: str) -> str:
    """Return ``""`` if *server_url* may be used; else a non-empty error string
    explaining why it was rejected.

    Because there is no manifest signature yet (see the TODO above), TLS is our
    only line of defence against a network MITM tampering with the manifest /
    payload in flight, so:

      * ``https://``                                     → always allowed.
      * ``http://`` to a PUBLIC host                     → always REJECTED
        (a plaintext public channel is wide open to MITM).
      * ``http://`` to loopback                          → allowed (local test).
      * ``http://`` to a PRIVATE/LAN host (RFC1918 etc.) → allowed ONLY when the
        operator explicitly opts in via ``MAST2_ALLOW_INSECURE_UPDATE=1``.

    The LAN exception is deliberate: MAST ships a default plaintext private-IP
    push server for a controlled campus LAN, and hard HTTPS-only would silently
    brick OTA there. Requiring an explicit env opt-in keeps that an informed
    decision instead of a silent default, while public http is never allowed.
    """
    u = (server_url or "").strip()
    low = u.lower()
    if low.startswith("https://"):
        return ""
    if low.startswith("http://"):
        host = u[len("http://"):].split("/", 1)[0].split(":", 1)[0].strip().strip("[]").lower()
        if host in ("localhost", "127.0.0.1", "::1"):
            return ""  # loopback — local dev/test
        if _is_private_host(host):
            if os.environ.get(_INSECURE_UPDATE_ENV, "").strip() in ("1", "true", "yes"):
                logger.warning(
                    "OTA over plaintext http to private host %s — allowed by %s; "
                    "no manifest signature, integrity rests on the LAN being trusted.",
                    host, _INSECURE_UPDATE_ENV,
                )
                return ""
            return (f"更新服务器为明文 http 私有地址（{host}）。明文升级包易被中间人篡改，"
                    f"且当前无 manifest 签名。如确在可信内网，请显式设置环境变量 "
                    f"{_INSECURE_UPDATE_ENV}=1 以允许；否则请改用 https://。")
        return ("更新服务器 URL 必须使用 HTTPS（明文 http:// 公网地址易被中间人篡改"
                "升级包）。请改用 https:// 地址。")
    return f"无法识别的更新服务器 URL 协议（需 https://）：{server_url!r}"


def _read_one_line(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        pass
    return ""


SERVER_CA_FILE = "update_server_ca.pem"


def _verify_arg(data_root: Path | None):
    """httpx ``verify=`` value. Trust order for the self-signed LAN update server:
      1. a per-client pin at ``<data>/api key/update_server_ca.pem`` (admin override)
      2. the BUILD-BUNDLED server cert shipped inside the installer (so the operator
         doesn't have to copy it to every client) — frozen: ``_MEIPASS/…``
      3. the system CA store (real public certs)
    So a self-signed LAN server is trusted, while a MITM presenting any OTHER cert
    is rejected. NEVER returns False / disables verification."""
    import sys

    try:
        root = data_root
        if root is None:
            from mast._runtime_paths import project_root
            root = project_root()
        ca = Path(root) / "api key" / SERVER_CA_FILE
        if ca.exists() and ca.stat().st_size > 0:
            return str(ca)
    except Exception:  # noqa: BLE001 — fall through to bundled / system CA
        pass
    # Build-bundled server cert (shipped in the app; no per-client copy needed).
    try:
        cands = []
        mei = getattr(sys, "_MEIPASS", "")
        if mei:
            cands.append(Path(mei) / SERVER_CA_FILE)
        cands.append(Path(sys.executable).parent / "_internal" / SERVER_CA_FILE)
        for c in cands:
            if c.exists() and c.stat().st_size > 0:
                return str(c)
    except Exception:  # noqa: BLE001 — fall back to system CA, never to insecure
        pass
    return True


def read_server_url(data_root: Path) -> str:
    file_url = _read_one_line(data_root / "api key" / SERVER_URL_FILE)
    if file_url:
        return file_url
    from mast.update.defaults import get_default_server_url
    return get_default_server_url()


def write_server_url(data_root: Path, url: str) -> Path:
    p = data_root / "api key" / SERVER_URL_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# MAST 推送服务器 URL，必须使用 HTTPS（明文 http 已被拒绝），"
        "例如 https://updates.example.com:8766\n"
        f"{url.strip()}\n",
        encoding="utf-8",
    )
    return p


def _read_token(data_root: Path) -> str:
    p = data_root / "api key" / TOKEN_FILE
    if p.exists():
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
        except OSError:
            pass
    from mast.update.defaults import get_default_token
    return get_default_token()


def write_token(data_root: Path, token: str) -> Path:
    p = data_root / "api key" / TOKEN_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# MAST 推送客户端 token。需要与服务器 update_server_token.env 一致。\n"
        f"{token.strip()}\n",
        encoding="utf-8",
    )
    return p


def pending_dir(data_root: Path) -> Path:
    p = data_root / PENDING_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def installed_dir(data_root: Path) -> Path:
    p = pending_dir(data_root) / "installed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_pending(data_root: Path):
    from mast.update.manifest import compute_sha256, load_manifest

    pdir = pending_dir(data_root)
    m = load_manifest(pdir / "manifest.json")
    if m is None:
        return None, None
    if not _is_safe_filename(m.filename):
        logger.warning("Pending manifest has unsafe filename: %r — discarding", m.filename)
        return None, None
    setup = pdir / m.filename
    if not setup.exists():
        return None, None
    if setup.stat().st_size != m.size_bytes:
        logger.warning("Pending update size mismatch — discarding")
        return None, None
    actual = compute_sha256(setup)
    if actual.lower() != m.sha256.lower():
        logger.warning("Pending update hash mismatch — discarding")
        return None, None
    return m, setup


def archive_installed(data_root: Path) -> None:
    pdir = pending_dir(data_root)
    target = installed_dir(data_root)
    for name in ("manifest.json",):
        src = pdir / name
        if src.exists():
            try:
                src.replace(target / name)
            except OSError:
                pass
    for f in pdir.iterdir():
        if f.is_file() and f.suffix.lower() == ".exe":
            try:
                f.replace(target / f.name)
            except OSError:
                pass


def check_and_download(server_url: str, token: str, data_root: Path,
                       *, current_version: str, progress_cb=None
                       ) -> tuple[str | None, str | None]:
    """Check the push server and download a newer build if available.

    ``progress_cb(downloaded_bytes, total_bytes)`` is called periodically during
    the download so a launcher UI can draw a progress bar (total may be 0 if the
    server omits Content-Length; callers should handle that as indeterminate).
    """
    import httpx

    from mast.update.manifest import (
        compute_sha256, is_newer, load_manifest, write_manifest, Manifest,
    )

    # Enforce TLS — no manifest signature yet, so an http:// channel is wide open
    # to a MITM swapping the manifest+payload (see _require_https / module TODO).
    tls_err = _require_https(server_url)
    if tls_err:
        return "error", tls_err

    headers = {"Authorization": f"Bearer {token}"}
    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), verify=_verify_arg(data_root)) as c:
            r = c.get(server_url.rstrip("/") + "/manifest.json", headers=headers)
            if r.status_code != 200:
                return "error", f"manifest GET {r.status_code}: {r.text[:200]}"
            raw_manifest = r.json()
            sig = ""  # signature (hex); may 404 on an unsigned/transition publisher
            try:
                rs = c.get(server_url.rstrip("/") + "/manifest.sig", headers=headers)
                if rs.status_code == 200:
                    sig = str((rs.json() or {}).get("signature", "") or "")
            except Exception:  # noqa: BLE001 — absence handled below
                sig = ""
            remote = Manifest.from_dict(raw_manifest)
    except Exception as exc:
        return "no-server", f"{type(exc).__name__}: {exc}"

    # ── AUTHENTICITY (closes the supply-chain gap): verify the Ed25519 signature
    #    over the manifest BEFORE trusting ANY field. With a release public key
    #    embedded in this build, a missing/invalid signature is FATAL — a MITM /
    #    compromised server can no longer ship malicious bytes with a matching
    #    hash. With NO embedded key (transition builds) we skip + warn (integrity
    #    then rests on TLS + the Bearer token only).
    from mast.update.signing import get_release_public_key, verify_manifest
    _pubkey = get_release_public_key()
    if _pubkey:
        if not sig:
            return "error", ("更新被拒：本客户端要求签名验证，但服务器未提供 manifest.sig"
                             "（发布端未签名，或中间人剥离了签名）。")
        if not verify_manifest(raw_manifest, sig, _pubkey):
            return "error", "更新被拒：manifest 签名验证失败（内容被篡改或密钥不匹配）。"
        logger.info("manifest signature verified (remote v%s)", remote.version)
    else:
        logger.warning("manifest signature NOT verified — no release public key "
                       "embedded in this build (transition mode; rely on TLS).")

    # Reject a manifest whose filename is anything other than a plain basename:
    # the server is not trusted to stay inside the pending dir (path traversal /
    # MITM defence). Do this BEFORE the filename touches any path or URL.
    if not _is_safe_filename(remote.filename):
        logger.warning("Rejecting manifest with unsafe filename: %r", remote.filename)
        return "error", f"bad filename in manifest: {remote.filename!r}"

    if not is_newer(remote.version, current_version):
        return "no-update", f"remote={remote.version} local={current_version}"

    pdir = pending_dir(data_root)
    existing = load_manifest(pdir / "manifest.json")
    if existing is not None and existing.version == remote.version and (pdir / remote.filename).exists():
        return "no-update", f"already pending {remote.version}"

    # Prefer a small incremental delta from the running version, if published.
    # On any failure we fall through to the full installer below.
    from mast.update.manifest import find_delta
    delta = find_delta(remote, current_version)
    if delta:
        ok, msg = _download_delta(server_url, token, pdir, remote, delta,
                                  progress_cb=progress_cb)
        if ok:
            return "downloaded-delta", msg
        logger.warning("delta %s→%s failed (%s) — falling back to full installer",
                       current_version, remote.version, msg)

    target = pdir / remote.filename
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with httpx.Client(timeout=httpx.Timeout(None, connect=10.0), verify=_verify_arg(data_root)) as c:
            with c.stream("GET", server_url.rstrip("/") + "/download/" + remote.filename,
                          headers=headers) as resp:
                if resp.status_code != 200:
                    return "error", f"download GET {resp.status_code}"
                total = int(resp.headers.get("content-length", 0)) or remote.size_bytes
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if progress_cb is not None:
                            try:
                                progress_cb(done, total)
                            except Exception:
                                pass
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return "error", f"{type(exc).__name__}: {exc}"

    if tmp.stat().st_size != remote.size_bytes:
        tmp.unlink(missing_ok=True)
        return "error", "size mismatch after download"
    if compute_sha256(tmp).lower() != remote.sha256.lower():
        tmp.unlink(missing_ok=True)
        return "error", "sha256 mismatch — possible corruption / mitm"

    # Defence-in-depth: the resolved target must still sit directly inside the
    # pending dir. Catches any traversal that slipped past _is_safe_filename.
    if target.resolve().parent != pdir.resolve():
        tmp.unlink(missing_ok=True)
        return "error", f"download target escaped pending dir: {target}"

    try:
        tmp.replace(target)
    except OSError as exc:
        return "error", f"rename {exc}"
    write_manifest(pdir / "manifest.json", remote)
    logger.info(
        "Downloaded MAST %s to %s (next launch will prompt force update)",
        remote.version, target,
    )
    return "downloaded", f"{remote.version} → {target.name}"


def post_feedback(server_url: str, token: str, *, text: str,
                  category: str = "feature", client_version: str = "",
                  ) -> tuple[str, str]:
    """POST a user wish/feedback to the server's ``/feedback`` endpoint.

    Returns ``(status, detail)``: status ∈ {"ok","no-config","no-server","error"};
    on "ok", detail is the server-assigned feedback id. Mirrors
    check_and_download's error handling. Bounded timeout so the GUI never hangs.
    """
    import httpx

    if not server_url or not token:
        return "no-config", "未配置更新服务器 URL / token（先在启动器配置）。"
    tls_err = _require_https(server_url)
    if tls_err:
        return "error", tls_err
    text = (text or "").strip()
    if not text:
        return "error", "反馈内容为空"
    try:
        with httpx.Client(timeout=httpx.Timeout(20.0, connect=10.0), verify=_verify_arg(None)) as c:
            r = c.post(
                server_url.rstrip("/") + "/feedback",
                headers={"Authorization": f"Bearer {token}"},
                json={"text": text[:8000], "category": category or "feature",
                      "client_version": client_version},
            )
        if r.status_code == 200:
            try:
                fid = (r.json() or {}).get("id", "")
            except Exception:
                fid = ""
            return "ok", str(fid or "")
        return "error", f"feedback POST {r.status_code}: {r.text[:200]}"
    except Exception as exc:  # network down / DNS / timeout
        return "no-server", f"{type(exc).__name__}: {exc}"


DELTA_MARKER = "delta.json"

#: delta.json 的字段名必须与启动器保持一致。启动器读取 filename；
#: 缺失该键会把 pending_update 目录本身误当更新文件，产生误导性错误。
#: 使用统一写入口，避免各推送路径手写不兼容的键名。
DELTA_MARKER_FIELDS = ("filename", "from_version", "to_version", "sha256")


def write_delta_marker(pdir: "Path", filename: str, from_version: "str | None",
                       to_version: "str | None", sha256: "str | None") -> "Path":
    """把 ``delta.json`` 写进 *pdir*,字段名取自 :data:`DELTA_MARKER_FIELDS`。

    **不要手写这个文件。** 侧门推送(scp 一个增量包上去再自己摆 marker)与正常
    下载路径用的是同一个读侧,所以必须共用同一个写侧。

    编码固定 UTF-8 **无 BOM**:PowerShell 5.1 的 ``Out-File -Encoding utf8``
    会加 BOM,而读侧 ``json.loads(marker.read_text(encoding="utf-8"))`` 会在
    BOM 上炸 —— 这个坑 OTA 上线时踩过一次,见 既有教训。
    """
    import json as _j
    marker = pdir / DELTA_MARKER
    payload = dict(zip(DELTA_MARKER_FIELDS,
                       (filename, from_version, to_version, sha256)))
    marker.write_text(_j.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return marker

# Data assets safe to hot-apply while MAST.exe is running (the running exe /
# locked DLLs cannot be replaced in place). Mirrors delta.DEFAULT_INCLUDE_PREFIXES.
# Install-ROOT-relative prefixes whose files are safe to overwrite while MAST is
# running (never a loaded DLL / the running exe / base_library.zip). Data assets
# (literature index, vision ckpt) + the TS SPA (served per-request via
# FileResponse, briefly opened then closed — os.replace swaps atomically) + docs
# (read-only reference). A delta touching ONLY these hot-applies live, no restart
# (feature B, 审查). Anything else → offline applier.
# 2026-08-02：这里曾与 delta.DEFAULT_INCLUDE_PREFIXES 一起停留在退役的
# mast_vision_m12.pt 上。两处都得改 —— 上面那句注释声称「Mirrors」，一旦漂开就是
# 一条没人守的不变式：视觉权重的增量包会被判成「不是纯数据」，白白走一次离线
# 应用器（重启），而这正是热应用当初想省掉的那次重启。
HOT_APPLY_PREFIXES = (
    "MASTv2/artifacts/literature_index/",
    "MASTv2/artifacts/mast_vision_v25.pt",
    "MASTv2/artifacts/stm_quality_v1_dino.joblib",
    "_internal/frontend/dist/",
    "docs/",
)


def delta_is_data_only(paths) -> bool:
    """True iff EVERY relpath is a data asset safe to overwrite live (literature
    index / vision checkpoint). A delta that touches code / exe / dll must go via
    the full installer (which replaces a not-running target), so this returns
    False for it. Empty → False (nothing to hot-apply)."""
    paths = list(paths or [])
    if not paths:
        return False
    for rel in paths:
        low = str(rel).replace("\\", "/")
        if not any(low == p.rstrip("/") or low.startswith(p) for p in HOT_APPLY_PREFIXES):
            return False
    return True


def _download_delta(server_url, token, pdir, remote, delta, *, progress_cb=None) -> tuple[bool, str]:
    """Download + verify an incremental delta. Writes a pending delta marker
    on success so the launcher applies it instead of running the full installer."""
    import httpx

    from mast.update.manifest import compute_sha256, write_manifest

    fname = delta.get("filename")
    if not fname:
        return False, "delta has no filename"
    # Same path-traversal guard as the full-installer path: the delta filename
    # comes from the (untrusted) server manifest.
    if not _is_safe_filename(fname):
        logger.warning("Rejecting delta with unsafe filename: %r", fname)
        return False, f"bad delta filename: {fname!r}"
    headers = {"Authorization": f"Bearer {token}"}
    target = pdir / fname
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with httpx.Client(timeout=httpx.Timeout(None, connect=10.0), verify=_verify_arg(pdir.parent)) as c:
            with c.stream("GET", server_url.rstrip("/") + "/download/" + fname,
                          headers=headers) as resp:
                if resp.status_code != 200:
                    return False, f"delta GET {resp.status_code}"
                total = int(resp.headers.get("content-length", 0)) or int(delta.get("size_bytes", 0))
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if progress_cb is not None:
                            try:
                                progress_cb(done, total)
                            except Exception:
                                pass
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"

    if delta.get("size_bytes") and tmp.stat().st_size != delta["size_bytes"]:
        tmp.unlink(missing_ok=True)
        return False, "delta size mismatch"
    if compute_sha256(tmp).lower() != str(delta.get("sha256", "")).lower():
        tmp.unlink(missing_ok=True)
        return False, "delta sha256 mismatch — corruption / mitm"
    try:
        tmp.replace(target)
    except OSError as exc:
        return False, f"rename {exc}"

    write_delta_marker(pdir, fname, delta.get("from_version"),
                       remote.version, delta.get("sha256"))
    write_manifest(pdir / "manifest.json", remote)
    logger.info("Downloaded incremental delta %s→%s (%s)",
                delta.get("from_version"), remote.version, fname)
    return True, f"delta {delta.get('from_version')}→{remote.version} ({fname})"


def apply_pending_delta(data_root: Path, install_root: Path) -> tuple[str, str]:
    """If a pending delta is staged, apply it to *install_root* — the install
    directory that CONTAINS ``MAST2.exe`` / ``_internal/`` / ``MASTv2/`` (NOT the
    ``_internal`` subdir). The delta's relpaths are install-ROOT-relative
    (e.g. ``_internal/mast/x.py``, ``MAST2.exe``,
    ``MASTv2/artifacts/literature_index/vectors.npy``), so the target MUST be the
    root or the files land in the wrong place. Returns (status, detail), status ∈
    {applied, none, error}. Called by the launcher at update-apply time; on any
    failure the caller falls back to the full installer (still in pending).
    """
    import json

    from mast.update.delta import DeltaError, apply_delta

    pdir = pending_dir(data_root)
    marker = pdir / DELTA_MARKER
    if not marker.exists():
        return "none", "no pending delta"
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        dfname = info.get("filename", "")
        if not _is_safe_filename(dfname):
            return "error", f"bad delta filename in marker: {dfname!r}"
        dz = pdir / dfname
        if not dz.exists():
            return "error", "delta file missing"
        res = apply_delta(dz, Path(install_root))
        marker.unlink(missing_ok=True)
        try:
            dz.unlink(missing_ok=True)
        except OSError:
            pass
        return "applied", (f"{info.get('from_version')}→{info.get('to_version')}: "
                           f"wrote {res['written']}, removed {res['removed']}")
    except DeltaError as exc:
        return "error", f"delta apply failed: {exc}"
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


def stage_offline_delta(data_root: Path, install_root: Path) -> tuple[str, "dict | str"]:
    """Verify + stage a pending delta for an OFFLINE apply.

    A delta touching the running exe / base_library.zip / a loaded DLL cannot be
    written in place (Windows locks them), so it can't hot-apply. Instead we
    VERIFY it (sha256 + traversal) and extract its files to a staging tree; the
    launcher then spawns a self-updater that, once MAST has exited, copies the
    staged files over *install_root* and relaunches (feature B, 2026-07-03).

    Returns (status, plan|detail); status ∈ {staged, none, error}. On 'staged'
    the plan dict feeds the self-updater: {stage_dir, install_root, copies[rel],
    removed[rel], from_version, to_version}. On error the caller falls back to the
    full installer (which is still pending / re-downloadable)."""
    import json as _json

    from mast.update.delta import DeltaError, stage_delta

    pdir = pending_dir(data_root)
    marker = pdir / DELTA_MARKER
    if not marker.exists():
        return "none", "no pending delta"
    try:
        info = _json.loads(marker.read_text(encoding="utf-8"))
        dfname = info.get("filename", "")
        if not _is_safe_filename(dfname):
            return "error", f"bad delta filename in marker: {dfname!r}"
        dz = pdir / dfname
        if not dz.exists():
            return "error", "delta file missing"
        stage_dir = pdir / "offline_stage"
        manifest = stage_delta(dz, stage_dir, Path(install_root))
        plan = {
            "stage_dir": str(stage_dir),
            "install_root": str(install_root),
            "copies": list(manifest.get("added", [])) + list(manifest.get("changed", [])),
            "removed": list(manifest.get("removed", [])),
            "from_version": manifest.get("from_version"),
            "to_version": manifest.get("to_version"),
        }
        return "staged", plan
    except DeltaError as exc:
        return "error", f"delta stage failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {exc}"


class UpdateChecker:
    """Daemon thread wrapper that does check_and_download on a schedule."""

    def __init__(self, data_root: Path, current_version: str,
                 *, interval_min: int = 30):
        self._data_root = data_root
        self._version = current_version
        self._interval = max(60, int(interval_min * 60))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_check: datetime | None = None
        self._last_status: str = "(not started)"
        self._last_detail: str = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mast2-update")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def trigger_now(self) -> tuple[str, str]:
        return self._do_check()

    def status_summary(self) -> dict[str, Any]:
        return {
            "last_check": self._last_check.isoformat(timespec="seconds") if self._last_check else None,
            "last_status": self._last_status,
            "last_detail": self._last_detail,
        }

    def _do_check(self) -> tuple[str, str]:
        self._last_check = datetime.now()
        server_url = read_server_url(self._data_root)
        token = _read_token(self._data_root)
        if not server_url:
            self._last_status = "no-server-url"
            self._last_detail = "set api key/update_server_url.env"
            return ("no-server-url", "")
        if not token:
            self._last_status = "no-token"
            self._last_detail = "set api key/update_client_token.env"
            return ("no-token", "")
        status, detail = check_and_download(
            server_url, token, self._data_root,
            current_version=self._version,
        )
        self._last_status = status
        self._last_detail = detail or ""
        return status, detail or ""

    def _loop(self) -> None:
        self._stop.wait(timeout=60)
        while not self._stop.is_set():
            try:
                self._do_check()
            except Exception as exc:
                logger.warning("update check raised: %s", exc)
                self._last_status = "exception"
                self._last_detail = str(exc)
            self._stop.wait(timeout=self._interval)
