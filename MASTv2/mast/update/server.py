"""MAST push-update server (FastAPI) — runs on the admin's machine.

vendored from v1 mast/update/server.py. Default port 8766 (one above
v1's 8765) so a single admin machine can host both.
"""
from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

# ⚠️ 这个名字必须在**模块** namespace 里。本文件顶部有
# ``from __future__ import annotations``（PEP 563），函数签名里的注解因此是字符串，
# 而 FastAPI 靠 ``get_type_hints`` 在**模块**全局里解析它们 —— 一个只在
# ``build_app()`` 函数体内 import 的 ``Request``，FastAPI 一辈子也找不到。
# 症状不是报错，是那个参数被当成**查询参数**，请求全部 422
# （"Field required: query.request"）。本仓在 ``NotRequired`` 上踩过同一个坑。
#
# 用 starlette 而不是 fastapi：它是更底层那个包，且 fastapi 的 Request 就是它。
try:
    from starlette.requests import Request
except ImportError:                     # 没装 fastapi/starlette 时
    # 本模块的 token / 目录那部分不依赖 HTTP 栈，仍要能 import。
    Request = object                    # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

PUSH_DIR_NAME = "push_distribution"
TOKEN_FILE = "update_server_token.env"
#: 发布凭据。**与上面那个是两件东西**，理由见 :func:`_read_publish_token`。
PUBLISH_TOKEN_FILE = "update_publish_token.env"
ACCESS_LOG = "access.log"


def _read_token(data_root: Path) -> str:
    """客户端凭据 —— **每一台装了 MAST 的机器都有它**。

    取值链：``api key/update_client_token.env`` > ``_defaults.DEFAULT_TOKEN``。
    后者由打包脚本写进 ``mast/update/_defaults.py``，随字节码进 ``MAST.exe``
    的 PYZ，所以**它等价于公开值** —— 谁拿到安装包谁就有它。

    正因为如此，它只够用来**读**，以及做客户端自己那几件写（投稿技能、分享
    订阅、提反馈）。真正把内容推给所有机器的那一个动作要另一把钥匙，见
    :func:`_read_publish_token`。
    """
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


def _read_publish_token(data_root: Path) -> str:
    """发布凭据 —— **只在发布机上，永不进任何客户端包**。

    ⚠️ **故意没有默认值，也故意不回落到客户端 token。**

    回落是这道门唯一会失效的方式：客户端 token 是公开的（见
    :func:`_read_token`），一旦「没配发布 token 就用客户端 token」，那么每一台
    装了 MAST 的机器又都能发布了 —— 门还在，判据没了。本仓在别处反复吃过这个
    形状（守卫看着在、其实没有）。

    没配置的正确表现是**发布端点 503 并说清楚怎么配**，不是悄悄放行。
    """
    p = data_root / "api key" / PUBLISH_TOKEN_FILE
    if p.exists():
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
        except OSError:
            pass
    return ""


def write_token(data_root: Path, token: str) -> Path:
    p = data_root / "api key" / TOKEN_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# MAST 推送服务器**客户端** token。客户端 update_client_token.env 填同一个值。\n"
        "# 它随安装包分发，等价于公开值 —— 只够读，以及客户端自己的投稿/反馈。\n"
        "# 发布技能包用的是另一把：update_publish_token.env。\n"
        f"{token.strip()}\n",
        encoding="utf-8",
    )
    return p


def write_publish_token(data_root: Path, token: str) -> Path:
    """写发布凭据。**这个文件不该出现在任何客户端机器上。**"""
    p = data_root / "api key" / PUBLISH_TOKEN_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# MAST **发布** token —— 只有它能调 POST /skills/pack/publish。\n"
        "# 只放在发布机上；不要写进 push_defaults*.json，也不要随包分发。\n"
        "# 客户端拉更新用的是另一把：update_client_token.env。\n"
        f"{token.strip()}\n",
        encoding="utf-8",
    )
    return p


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def push_dir(data_root: Path) -> Path:
    p = data_root / PUSH_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log_access(data_root: Path, **fields: Any) -> None:
    log_path = push_dir(data_root) / ACCESS_LOG
    fields["ts"] = datetime.now().isoformat(timespec="seconds")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _log_feedback(data_root: Path, **fields: Any) -> None:
    """Append a user wish/feedback record to push_dir/feedback.jsonl (admin reads it)."""
    log_path = push_dir(data_root) / "feedback.jsonl"
    fields["ts_server"] = datetime.now().isoformat(timespec="seconds")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")
    except OSError:
        pass


def build_app(data_root: Path):
    import shutil

    from fastapi import Body, FastAPI, HTTPException, Header
    from fastapi.responses import FileResponse, JSONResponse

    from mast.update.manifest import load_manifest

    app = FastAPI(title="MAST Push Update Server", version="1.0")

    def _check_auth(authorization: str | None, *, publish: bool = False) -> None:
        """两级凭据。

        ``publish=False``（默认）—— 客户端能做的事：读清单/下载，以及客户端
        自己那几件写（投稿技能、分享订阅、提反馈）。**客户端 token 或发布
        token 都算数**：发布机也要能读。

        ``publish=True`` —— 把内容推给所有机器的那一个动作
        （``POST /skills/pack/publish``）。**只认发布 token**。

        ⚠️ 没配发布 token 时这里 **503，不回落到客户端 token**。客户端 token
        随安装包分发、等价于公开值，一旦回落，每台机器就又都能发布了 ——
        那时门还在，判据没了。
        """
        presented = ""
        if authorization and authorization.startswith("Bearer "):
            presented = authorization[len("Bearer "):].strip()

        if publish:
            pub = _read_publish_token(data_root)
            if not pub:
                raise HTTPException(
                    status_code=503,
                    detail="publish token not configured — 发布凭据未配置。"
                           "在发布机上跑 `python -m mast.update publish-token` "
                           f"生成，它会写进 <data>/api key/{PUBLISH_TOKEN_FILE}。"
                           "**不要**用客户端 token 代替：那个值随安装包分发。")
            if not presented:
                raise HTTPException(status_code=401, detail="missing bearer token")
            # Constant-time comparison to avoid leaking the token via timing.
            if not secrets.compare_digest(presented, pub):
                raise HTTPException(
                    status_code=403,
                    detail="this endpoint needs the PUBLISH token, not the "
                           "client one — 发布端点要发布凭据，客户端 token 不够")
            return

        token = _read_token(data_root)
        if not token:
            raise HTTPException(status_code=503, detail="server token not configured")
        if not presented:
            raise HTTPException(status_code=401, detail="missing bearer token")
        pub = _read_publish_token(data_root)
        # 发布机手上只有发布 token 时也要读得动。
        if not (secrets.compare_digest(presented, token)
                or (pub and secrets.compare_digest(presented, pub))):
            raise HTTPException(status_code=401, detail="bad token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        m = load_manifest(push_dir(data_root) / "manifest.json")
        return {
            "status": "ok",
            "version": m.version if m else None,
            "published_at": m.published_at if m else None,
        }

    @app.get("/manifest.json")
    def manifest(authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        m_path = push_dir(data_root) / "manifest.json"
        m = load_manifest(m_path)
        if m is None:
            raise HTTPException(status_code=404, detail="no published version")
        _log_access(data_root, kind="manifest", version=m.version)
        return JSONResponse(m.to_dict())

    @app.get("/manifest.sig")
    def manifest_sig(authorization: str | None = Header(default=None)):
        """Ed25519 signature (hex) over the canonical manifest bytes. 404 when the
        publisher did not sign (transition builds). The client verifies this
        against its embedded public key before trusting the manifest."""
        _check_auth(authorization)
        p = push_dir(data_root) / "manifest.sig"
        if not p.exists():
            raise HTTPException(status_code=404, detail="manifest not signed")
        sig = ""
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sig = line
                break
        return JSONResponse({"signature": sig})

    @app.get("/download/{filename}")
    def download(filename: str, authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="bad filename")
        m = load_manifest(push_dir(data_root) / "manifest.json")
        if m is None:
            raise HTTPException(status_code=404, detail="no published version")
        # Serve the full installer OR any published incremental delta (each delta
        # has its own filename in manifest.deltas[]). Without this the client's
        # delta GET 404s and OTA incremental updates can never download.
        delta_names = {d.get("filename") for d in (m.deltas or [])
                       if isinstance(d, dict)}
        if filename != m.filename and filename not in delta_names:
            raise HTTPException(status_code=404, detail="not the current published file")
        target = push_dir(data_root) / filename
        if not target.exists():
            raise HTTPException(status_code=404, detail="file missing")
        _log_access(data_root, kind="download",
                    filename=filename, version=m.version, size=m.size_bytes)
        return FileResponse(
            path=target, filename=filename, media_type="application/octet-stream",
        )

    @app.post("/skills/upload")
    def skills_upload(payload: dict = Body(default={}),  # noqa: B008
                      authorization: str | None = Header(default=None)):
        """实验室技能库上行（P5）：只收 CompositeSpec manifest JSON——
        **永不接收 .py / 代码文件**（OTA 签名链建成前，代码分发=RCE 渠道，
        这是红线而非实现细节）。落 skill_inbox/ 待管理员人工审核，审核后经
        既有 OTA data-delta 白名单回发各机。"""
        _check_auth(authorization)
        manifest = (payload or {}).get("manifest")
        if not isinstance(manifest, dict) or not manifest.get("name"):
            raise HTTPException(status_code=400,
                                detail="manifest (CompositeSpec JSON) required")
        if not isinstance(manifest.get("nodes"), list):
            raise HTTPException(status_code=400,
                                detail="manifest.nodes must be a list "
                                       "(code files are NOT accepted)")
        raw = json.dumps(manifest, ensure_ascii=False)
        if len(raw) > 512_000:
            raise HTTPException(status_code=413, detail="manifest too large")
        import re as _re
        name = str(manifest["name"])
        if not _re.match(r"^[A-Za-z0-9_一-鿿][A-Za-z0-9_\-一-鿿]{0,80}$", name):
            raise HTTPException(status_code=400, detail="bad skill name")
        sid = "sk-" + secrets.token_hex(6)
        inbox = push_dir(data_root) / "skill_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / f"{sid}.json").write_text(raw, encoding="utf-8")
        entry = {
            "id": sid, "name": name,
            "version": manifest.get("version"),
            "description": str(manifest.get("description", ""))[:200],
            "author": str(manifest.get("_author", ""))[:64],
            "machine": str(manifest.get("_machine", ""))[:64],
            "content_sha256": str(manifest.get("_content_sha256", ""))[:64],
            "client_version": str((payload or {}).get("client_version", ""))[:64],
            "status": "pending_review",
            "ts_server": datetime.now().isoformat(timespec="seconds"),
        }
        with open(inbox / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("skill upload: %s (%s v%s from %s)", sid, name,
                    entry["version"], entry["machine"])
        return {"ok": True, "id": sid, "status": "pending_review"}

    @app.get("/skills/index")
    def skills_index(authorization: str | None = Header(default=None)):
        """中心技能索引（P5）：inbox 摘要（名/版本/哈希/作者/状态）——
        防碎片化用索引而非全量内容托管。"""
        _check_auth(authorization)
        idx = push_dir(data_root) / "skill_inbox" / "index.jsonl"
        entries: list = []
        if idx.exists():
            for line in idx.read_text(encoding="utf-8").splitlines():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {"skills": entries[-500:]}

    # ── 订阅列表索引（2026-08-26）────────────────────────────────────
    #
    # 分享的是一份**订阅列表**（哪些技能构成一个人的日常工作面），不是技能本身。
    # 它对新人的价值在于「一个做 qPlus 的人手上是哪 40 个技能」—— 那份筛选是经验，
    # 而经验在这个仓库里一直没有被表达的地方。
    #
    # 红线与 /skills/upload 同一条，而且**收得更紧**：条目的键是**白名单**
    # （name/source/version/spec），不是黑名单。理由是黑名单要求我预见到所有能藏
    # 代码的键名，而白名单只要求我说清楚哪几个键是数据 —— 后者我答得上来。
    # 内嵌的 spec 仍然必须是 CompositeSpec 形状（nodes 是 list），与老门同判。

    _SUB_KIND = "mast-skill-subscription"
    _SUB_ENTRY_KEYS = {"name", "source", "version", "spec"}

    @app.post("/subscriptions/upload")
    def subscriptions_upload(payload: dict = Body(default={}),  # noqa: B008
                             authorization: str | None = Header(default=None)):
        """上传一份订阅列表 manifest（纯数据；**永不接收代码**）。"""
        _check_auth(authorization)
        man = (payload or {}).get("manifest")
        if not isinstance(man, dict):
            raise HTTPException(status_code=400, detail="manifest required")
        if man.get("kind") != _SUB_KIND:
            raise HTTPException(status_code=400,
                                detail=f"manifest.kind must be {_SUB_KIND!r}")
        entries = man.get("entries")
        if not isinstance(entries, list):
            raise HTTPException(status_code=400, detail="manifest.entries must be a list")
        if len(entries) > 2000:
            raise HTTPException(status_code=413, detail="too many entries")
        for e in entries:
            if not isinstance(e, dict) or not isinstance(e.get("name"), str):
                raise HTTPException(status_code=400, detail="each entry needs a name")
            extra = set(e) - _SUB_ENTRY_KEYS
            if extra:
                # 白名单，不是黑名单 —— 见上面那段注释。
                raise HTTPException(
                    status_code=400,
                    detail=f"entry {e.get('name')!r} has unexpected keys "
                           f"{sorted(extra)} (code files are NOT accepted)")
            spec = e.get("spec")
            if spec is not None:
                if not isinstance(spec, dict) or not isinstance(spec.get("nodes"), list):
                    raise HTTPException(
                        status_code=400,
                        detail="entry.spec must be a CompositeSpec with a nodes list "
                               "(code files are NOT accepted)")
        raw = json.dumps(man, ensure_ascii=False)
        if len(raw) > 2_000_000:
            raise HTTPException(status_code=413, detail="manifest too large")

        sid = "sub-" + secrets.token_hex(6)
        inbox = push_dir(data_root) / "subscription_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / f"{sid}.json").write_text(raw, encoding="utf-8")
        entry = {
            "id": sid,
            "label": str((payload or {}).get("label", ""))[:80],
            "note": str((payload or {}).get("note", ""))[:400],
            "machine": str(man.get("machine", ""))[:64],
            "exported_at": str(man.get("exported_at", ""))[:32],
            "app_version": str(man.get("app_version", ""))[:32],
            "skill_count": len(entries),
            "embedded_specs": sum(1 for e in entries if isinstance(e.get("spec"), dict)),
            "client_version": str((payload or {}).get("client_version", ""))[:64],
            "status": "pending_review",
            "ts_server": datetime.now().isoformat(timespec="seconds"),
        }
        with open(inbox / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("subscription upload: %s (%d skills from %s)",
                    sid, entry["skill_count"], entry["machine"])
        return {"ok": True, "id": sid, "status": "pending_review"}

    @app.get("/subscriptions/index")
    def subscriptions_index(authorization: str | None = Header(default=None)):
        """中心订阅索引：摘要（谁的、多少个技能、什么时候）—— 不含条目本身。"""
        _check_auth(authorization)
        idx = push_dir(data_root) / "subscription_inbox" / "index.jsonl"
        entries: list = []
        if idx.exists():
            for line in idx.read_text(encoding="utf-8").splitlines():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {"subscriptions": entries[-500:]}

    @app.get("/subscriptions/download/{sub_id}")
    def subscriptions_download(sub_id: str,
                               authorization: str | None = Header(default=None)):
        """取回一份订阅 manifest 的正文（索引只给摘要）。"""
        _check_auth(authorization)
        import re as _re
        if not _re.match(r"^sub-[0-9a-f]{12}$", sub_id):
            raise HTTPException(status_code=400, detail="bad id")
        p = push_dir(data_root) / "subscription_inbox" / f"{sub_id}.json"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return json.loads(p.read_text(encoding="utf-8"))

    # ── 签名技能包（P5.5，2026-08-20）─────────────────────────────────
    #
    # 与上面 /skills/upload 那条红线**不冲突**：那扇门拒的是**未签名**的代码载荷，
    # 对它仍然成立（test_skills_upload_still_rejects_code 钉住它没被顺手拆掉）。
    # 这三个端点收的是 Ed25519 签名包，而且——
    #
    # **服务器只存不签。** 签名永远在管理员机器上离线产生，服务器手上没有私钥，
    # 被攻破也签不出东西来。服务器自己先验一遍，纯粹是防管理员手滑上传了一个
    # 验不过的包——那种包推给每台机器都会被拒，而拒绝发生在很远的地方。

    def _packs_dir() -> Path:
        d = push_dir(data_root) / "skillpacks"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @app.post("/skills/pack/publish")
    async def skills_pack_publish(request: Request,
                                 authorization: str | None = Header(default=None)):
        """上传一个**已签名**的技能包（raw zip body）。服务器验一遍，验不过不存。"""
        import tempfile

        from mast.update import skillpack as SP

        _check_auth(authorization, publish=True)
        n = 0
        tmp = Path(tempfile.mkdtemp(prefix="skpack_")) / "pack.zip"
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                n += len(chunk)
                if n > SP.MAX_PACK_BYTES:
                    f.close()
                    shutil.rmtree(tmp.parent, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"pack exceeds {SP.MAX_PACK_BYTES} bytes")
                f.write(chunk)
        try:
            # ⚠️ app_version 留空：服务器不知道也不该知道**目标机器**的版本，
            # 那道门属于客户端。在这里用服务器自己的版本判，会让一个专门给新版
            # 打的包在发布这一步就被挡下来。
            res = SP.verify_pack(tmp, require_signature=True)
            if not res.ok or res.manifest is None:
                raise HTTPException(status_code=400, detail={
                    "error": "pack rejected", "reasons": res.reasons})
            man = res.manifest
            target = _packs_dir() / SP.pack_filename(man.pack_id, man.version)
            shutil.copy2(tmp, target)
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

        entry = {
            "pack_id": man.pack_id, "version": man.version,
            "filename": target.name, "size": target.stat().st_size,
            "author": man.author[:64], "description": man.description[:200],
            "min_app_version": man.min_app_version,
            "max_app_version": man.max_app_version,
            "n_files": len(man.files),
            "sha256": SP.sha256_bytes(target.read_bytes()),
            "ts_server": datetime.now().isoformat(timespec="seconds"),
        }
        with open(_packs_dir() / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("skillpack published: %s v%s (%d files, sig %s)",
                    man.pack_id, man.version, len(man.files), res.pubkey8)
        return {"ok": True, "pack_id": man.pack_id, "version": man.version,
                "filename": target.name, "signature": f"verified:{res.pubkey8}"}

    @app.get("/skills/pack/index")
    def skills_pack_index(authorization: str | None = Header(default=None)):
        """发布过哪些包。每个 pack_id 只列**最新**那一版。"""
        _check_auth(authorization)
        idx = _packs_dir() / "index.jsonl"
        latest: dict[str, dict] = {}
        if idx.exists():
            from mast.update.manifest import version_tuple
            for line in idx.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = str(e.get("pack_id") or "")
                if not pid:
                    continue
                cur = latest.get(pid)
                if cur is None or version_tuple(str(e.get("version") or "")) >=                         version_tuple(str(cur.get("version") or "")):
                    latest[pid] = e
        return {"packs": sorted(latest.values(), key=lambda d: d["pack_id"])}

    @app.get("/skills/pack/download/{pack_id}")
    def skills_pack_download(pack_id: str,
                             authorization: str | None = Header(default=None)):
        """下最新版的那个包。"""
        from mast.update import skillpack as SP

        _check_auth(authorization)
        if not SP.is_valid_pack_id(pack_id):
            raise HTTPException(status_code=400, detail="bad pack_id")
        idx = skills_pack_index(authorization)
        hit = next((e for e in idx["packs"] if e["pack_id"] == pack_id), None)
        if hit is None:
            raise HTTPException(status_code=404, detail="no such pack")
        target = _packs_dir() / str(hit["filename"])
        # 纵深防御：解析出来的路径必须还在包目录里（同 client.py 那条 escaped 检查）
        if not target.is_file() or target.resolve().parent != _packs_dir().resolve():
            raise HTTPException(status_code=404, detail="pack file missing")
        _log_access(data_root, kind="pack_download", filename=target.name,
                    version=hit.get("version"), size=target.stat().st_size)
        return FileResponse(path=target, filename=target.name,
                            media_type="application/zip")

    @app.post("/feedback")
    def feedback(payload: dict = Body(default={}),  # noqa: B008
                 authorization: str | None = Header(default=None)):
        """Receive a user wish/feedback from a client and append it to
        push_dir/feedback.jsonl (the admin reviews these). Auth-gated, bounded."""
        _check_auth(authorization)
        text = str((payload or {}).get("text", "") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty feedback text")
        text = text[:8000]
        category = str((payload or {}).get("category", "feature") or "feature").strip()[:64]
        client_version = str((payload or {}).get("client_version", "") or "")[:64]
        fid = "fb-" + secrets.token_hex(6)
        _log_feedback(data_root, id=fid, kind="feedback", text=text,
                      category=category, client_version=client_version)
        logger.info("feedback received: %s (%s, v%s)", fid, category, client_version)
        return {"ok": True, "id": fid}

    return app


def run_server(data_root: Path, host: str = "0.0.0.0", port: int = 8766,
               ssl_certfile: str | None = None, ssl_keyfile: str | None = None) -> int:
    import uvicorn

    on_disk = ""
    file_path = data_root / "api key" / TOKEN_FILE
    if file_path.exists():
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                on_disk = line
                break

    if not on_disk:
        from mast.update.defaults import get_default_token
        baked = get_default_token().strip()
        if baked:
            write_token(data_root, baked)
            logger.info(
                "No on-disk token found; persisted the bundle's baked "
                "DEFAULT_TOKEN to %s so distributed clients keep working.",
                file_path,
            )
        else:
            new_token = generate_token()
            write_token(data_root, new_token)
            logger.warning(
                "WARNING: no token file AND no baked DEFAULT_TOKEN — "
                "generated a fresh random one (%s).", new_token,
            )
            print(
                f"\n=== MAST 推送服务器 token (NEW — copy to clients!) ===\n"
                f"{new_token}\n已写入: {file_path}\n"
            )

    push_dir(data_root)
    # TLS: when a cert+key are supplied the server speaks HTTPS, so the Bearer
    # token + manifest + installer bytes are encrypted in flight (the client
    # REJECTS plaintext http to non-loopback hosts unless the operator opts into
    # MAST2_ALLOW_INSECURE_UPDATE). Falls back to http only for loopback/dev.
    ssl_kw: dict = {}
    scheme = "http"
    if ssl_certfile and ssl_keyfile and Path(ssl_certfile).exists() and Path(ssl_keyfile).exists():
        ssl_kw = {"ssl_certfile": str(ssl_certfile), "ssl_keyfile": str(ssl_keyfile)}
        scheme = "https"
    logger.info("MAST push server starting on %s://%s:%d  (data: %s, TLS=%s)",
                scheme, host, port, data_root, bool(ssl_kw))
    uvicorn.run(build_app(data_root), host=host, port=port, log_level="info", **ssl_kw)
    return 0
