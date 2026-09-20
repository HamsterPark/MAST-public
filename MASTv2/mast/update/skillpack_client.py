"""装签名技能包 —— 本地文件 / 推送服务器两条来路，同一套门。

来路只影响**怎么拿到那个 zip**，不影响**凭什么信它**：两条路都走
:func:`mast.update.skillpack.verify_pack`，一个字节都不例外。侧门推送
（``push_skillpack_to_rig.py``）之所以安全，不是因为「是我们自己推的」，
是因为它推的东西一样要过验签。

全成或全不成
==========
先解到 ``.staging-<id>/``，全部验完才 :func:`os.replace` 到位。中途任何一步失败，
整个临时目录删掉，``_packs/<id>/`` 保持原样 —— **不留半个文件**。
半应用状态在这里格外坏：覆盖层会把 ``_packs/`` 下的东西标成「已签名」，
半个包意味着一部分技能是新的、一部分是旧的，而界面上看起来一切正常。

自动启用
========
装完自动登记进 ``overlay.json`` 并启用（``Manifest.auto_enable_packs``，默认开），
每台机器可以单独关。这和「丢文件进目录不会自动生效」那条纪律不矛盾 ——
门不一样：松散文件没有门，签名包有验签 + 逐文件 sha256 + 加载时再复核三道。

**但启用不等于生效**：技能还要等一次重载。所以这里返回 ``needs_reload=True``，
由调用方去推那次重载（任务运行中会排队）。把「装好了」说成「生效了」，
正是这一整套功能的头号事故形态。
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from mast.update import skillpack as SP

logger = logging.getLogger(__name__)


@dataclass
class InstallResult:
    ok: bool = False
    pack_id: str = ""
    version: str = ""
    reasons: list[str] = field(default_factory=list)
    installed: list[str] = field(default_factory=list)   # 覆盖层相对路径
    enabled: list[str] = field(default_factory=list)     # 自动启用了哪些
    shadowed: list[str] = field(default_factory=list)    # 被本机松散文件遮盖的
    needs_reload: bool = False
    replaced_version: str = ""

    def describe(self) -> str:
        if not self.ok:
            return "技能包没有装上：" + "；".join(self.reasons or ("原因不明",))
        bits = [f"{self.pack_id} v{self.version} 已装入 {len(self.installed)} 个文件"]
        if self.replaced_version:
            bits.append(f"（替换 v{self.replaced_version}）")
        if self.enabled:
            bits.append(f"，自动启用 {len(self.enabled)} 个")
        else:
            bits.append("，未自动启用（本机关掉了 auto_enable_packs）")
        if self.shadowed:
            bits.append(f"；{len(self.shadowed)} 个被本机同名文件遮盖")
        if self.needs_reload:
            bits.append(" —— 尚未生效，需要重载技能")
        return "".join(bits)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "pack_id": self.pack_id, "version": self.version,
                "reasons": list(self.reasons), "installed": list(self.installed),
                "enabled": list(self.enabled), "shadowed": list(self.shadowed),
                "needs_reload": self.needs_reload,
                "replaced_version": self.replaced_version,
                "summary": self.describe()}


def _shadowing_loose_files(rels: list[str]) -> list[str]:
    """这些相对路径上，本机有没有**松散**的同名覆盖文件。

    同名冲突时松散文件胜（用户自己的改动 > 推过来的），但**必须说出来**。
    不说这一句就会有那个经典事故：「我推了修复过去，它没生效」——
    而两边看起来都正常。
    """
    from mast.skills.overlay import paths as P

    return [r for r in rels if P.entry_path(r).is_file()]


def install_pack_file(zip_path: Path | str, *, public_key_hex: str = "",
                      app_version: str = "", auto_enable: bool | None = None
                      ) -> InstallResult:
    """把一个本地 ``.zip`` 装进 ``config/skill_overlay/_packs/<id>/``。**永不抛。**

    ``auto_enable=None`` 表示按清单里的 ``auto_enable_packs`` 走（默认开）。
    """
    from mast.api.version import get_version
    from mast.skills.overlay import manifest as M, paths as P

    res = InstallResult()
    zp = Path(zip_path)

    v = SP.verify_pack(zp, public_key_hex=public_key_hex,
                       app_version=app_version or get_version())
    if not v.ok or v.manifest is None:
        res.reasons = list(v.reasons) or ["验证失败，原因不明"]
        logger.warning("技能包被拒（%s）：%s", zp.name, "；".join(res.reasons))
        return res

    man = v.manifest
    res.pack_id, res.version = man.pack_id, man.version

    packs = P.packs_dir(create=True)
    dest = packs / man.pack_id
    staging = packs / f".staging-{man.pack_id}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    # 已经装过的版本 —— 报告里要说清楚这是替换而不是新增
    old = SP.verify_installed(dest, public_key_hex=public_key_hex) \
        if dest.is_dir() else None
    if old is not None and old.manifest is not None:
        res.replaced_version = old.manifest.version

    try:
        staging.mkdir(parents=True)
        with zipfile.ZipFile(zp) as zf:
            for f in man.files:
                # 路径已在 verify_pack 里查过；这里再确认落点没跑出去 ——
                # 纵深防御，同 client.py 那条 "download target escaped" 检查。
                target = (staging / f.path).resolve()
                if not str(target).startswith(str(staging.resolve())):
                    raise ValueError(f"落点跑出了 staging 目录：{f.path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(SP.FILES_PREFIX + f.path))
        SP.write_installed_manifest(staging, man, v.signature)

        # 落位前再验一遍**磁盘上**这份 —— 写盘可能截断，zip 里对不代表盘上对
        chk = SP.verify_installed(staging, public_key_hex=public_key_hex)
        if not chk.ok:
            raise ValueError("解出来之后自检不过：" + "；".join(chk.reasons))

        if dest.exists():
            shutil.rmtree(dest)
        os.replace(staging, dest)
    except Exception as exc:                                # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        res.reasons.append(f"{type(exc).__name__}: {exc}")
        logger.error("技能包 %s 落盘失败，已回退，未留下半个文件：%s",
                     man.pack_id, exc)
        return res

    rels = [f"{P.PACKS_DIR}/{man.pack_id}/{f.path}" for f in man.files]
    res.installed = rels
    res.ok = True
    res.shadowed = _shadowing_loose_files([f.path for f in man.files])

    # ── 自动启用 ────────────────────────────────────────────────────
    mani = M.load()
    if mani.unreadable:
        # 「读不到」不能当成「空的」：当成空的写回去，就把用户已启用的条目
        # 全抹掉了，而界面上什么都不会说。
        res.reasons.append(
            f"包已装入，但覆盖层清单读不出来（{mani.unreadable}），"
            "所以没有自动启用 —— 修好 overlay.json 再手工启用。")
        res.needs_reload = False
        logger.warning("技能包 %s 已装入但清单读不出来：%s", man.pack_id,
                       mani.unreadable)
        return res

    want = mani.auto_enable_packs if auto_enable is None else bool(auto_enable)
    if want:
        for rel in rels:
            mani.upsert(M.Entry(path=rel, enabled=True,
                                note=f"签名包 {man.pack_id} v{man.version}"))
        M.save(mani)
        res.enabled = list(rels)
        res.needs_reload = True

    # 大声 —— 一个包自己装上并启用了，这件事用户必须在日志里看得见。
    logger.warning(
        "签名技能包已装入并启用：%s v%s（签名 %s，%d 个文件）%s%s"
        "。**尚未生效**，等一次技能重载。要关掉自动启用："
        "在 overlay.json 里写 \"auto_enable_packs\": false。",
        man.pack_id, man.version, v.pubkey8, len(rels),
        f"，替换 v{res.replaced_version}" if res.replaced_version else "",
        f"，其中 {len(res.shadowed)} 个被本机同名文件遮盖："
        + "、".join(res.shadowed) if res.shadowed else "")
    return res


# ---------------------------------------------------------------------------
# 从推送服务器拉
# ---------------------------------------------------------------------------

def fetch_and_install(server_url: str, token: str, pack_id: str, *,
                      data_root: Path | None = None,
                      public_key_hex: str = "") -> InstallResult:
    """从推送服务器下一个包并装上。**永不抛。**

    网络这一层刻意很薄：它只负责把字节弄到本地，凭什么信它完全由
    :func:`install_pack_file` 决定。HTTPS 强制与证书参数复用
    ``update.client`` 的既有实现，不另写一套。
    """
    from mast.update.client import _require_https, _verify_arg

    res = InstallResult(pack_id=pack_id)
    tls = _require_https(server_url)
    if tls:
        res.reasons.append(tls)
        return res
    if not SP.is_valid_pack_id(pack_id):
        res.reasons.append(f"pack_id {pack_id!r} 不合法")
        return res

    try:
        import httpx
    except ImportError as exc:
        res.reasons.append(f"httpx 不可用：{exc}")
        return res

    from mast._runtime_paths import project_root
    root = data_root or project_root()
    tmpdir = root / "config" / "skill_overlay" / ".download"
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp = tmpdir / f"{pack_id}.zip.part"
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0),
                          verify=_verify_arg(root)) as c:
            with c.stream("GET",
                          server_url.rstrip("/") + "/skills/pack/download/" + pack_id,
                          headers={"Authorization": f"Bearer {token}"}) as r:
                if r.status_code != 200:
                    res.reasons.append(f"下载 HTTP {r.status_code}")
                    return res
                n = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 18):
                        n += len(chunk)
                        if n > SP.MAX_PACK_BYTES:
                            res.reasons.append(
                                f"服务器发来的东西超过 {SP.MAX_PACK_BYTES / 1e6:.0f} MB "
                                "上限，已中断下载")
                            return res
                        f.write(chunk)
    except Exception as exc:                                # noqa: BLE001
        res.reasons.append(f"{type(exc).__name__}: {exc}")
        return res
    finally:
        if res.reasons:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    out = install_pack_file(tmp, public_key_hex=public_key_hex)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return out


def installed_packs(*, public_key_hex: str = "") -> list[dict]:
    """本机装了哪些包，以及**它们现在还验得过吗**。

    每次都真的复核（读文件算 sha），不缓存 —— 这个函数存在的全部理由就是回答
    「落盘之后有没有被改过」，而缓存正好把那件事挡在外面。包很小，代价可忽略。
    """
    from mast.skills.overlay import paths as P

    out: list[dict] = []
    d = P.packs_dir()
    if not d.is_dir():
        return out
    for sub in sorted(x for x in d.iterdir() if x.is_dir()):
        if sub.name.startswith("."):
            continue                            # .staging-* / .download
        v = SP.verify_installed(sub, public_key_hex=public_key_hex)
        out.append({
            "pack_id": sub.name,
            "version": v.manifest.version if v.manifest else "",
            "author": v.manifest.author if v.manifest else "",
            "description": v.manifest.description if v.manifest else "",
            "n_files": len(v.manifest.files) if v.manifest else 0,
            "valid": v.ok,
            "reasons": list(v.reasons),
            "signature": f"verified:{v.pubkey8}" if v.ok else "invalid",
        })
    return out


def remove_pack(pack_id: str) -> tuple[bool, str]:
    """删掉一个包，并把它在清单里的条目一起摘掉。

    只删目录不摘清单，下次重载就会有一串「文件不在」的条目 —— 那些条目会被
    当成故障来查，而它们只是这次删除没做完。
    """
    from mast.skills.overlay import manifest as M, paths as P

    if not SP.is_valid_pack_id(pack_id):
        return False, f"pack_id {pack_id!r} 不合法"
    d = P.packs_dir() / pack_id
    if not d.is_dir():
        return False, f"没有装过 {pack_id}"
    try:
        shutil.rmtree(d)
    except OSError as exc:
        return False, f"删不掉：{exc}"
    mani = M.load()
    if not mani.unreadable:
        prefix = f"{P.PACKS_DIR}/{pack_id}/"
        keep = [e for e in mani.entries if not e.path.startswith(prefix)]
        if len(keep) != len(mani.entries):
            mani.entries = keep
            M.save(mani)
    logger.warning("签名技能包已移除：%s —— 尚未生效，等一次技能重载。", pack_id)
    return True, f"{pack_id} 已移除 —— 尚未生效，需要重载技能"


__all__ = ["InstallResult", "fetch_and_install", "install_pack_file",
           "installed_packs", "remove_pack"]
