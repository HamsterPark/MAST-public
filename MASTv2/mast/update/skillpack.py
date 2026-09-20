"""签名技能包 —— 改好的 skill 推给其它实验室机器，不发版本、不重启。

格式
====
``skillpack-<pack_id>-<version>.zip``::

    manifest.json     被签的那份：pack 元数据 + 每个文件的 sha256
    manifest.sig      {"signature": "<hex>"}
    files/…           覆盖层条目，路径相对覆盖层目录（builtins/bias.py）

**签 manifest 就传递性覆盖全部文件** —— manifest 里有每个文件的 sha256，
签名保护 manifest，manifest 保护文件。与 OTA 那条链完全同形，所以
:mod:`mast.update.signing` 一行都不用改。

字段名只有一份
============
包格式被三方读写：打包脚本、服务端、客户端、加载器。字段名各写各的，就是
``update/client.py:473-484`` 记着的那次事故（写侧 ``{"file":}``、读侧
``{"filename":}``，两边各自都「对」）。所以名字全在 :data:`K` 里，
谁也不许打字面量。

为什么另开一条通道，不动 ``/skills/upload``
=========================================
那扇门的红线是「**永不接收 .py / 代码文件**」（``update/server.py:157-164``），
理由是「OTA 签名链建成前，代码分发 = RCE 渠道」。链现在建成了，但那条红线的
语境是**未签名**的代码载荷 —— 对那扇门仍然成立。新通道有三道门：验签、
逐文件 sha256、落盘后加载时再复核一遍。老门原样不动，
``test_skills_upload_still_rejects_code`` 钉住它没被顺手拆掉。

fail-closed
===========
没装 ``cryptography`` 时 ``signing.HAVE_CRYPTO`` 为 False，
``verify_manifest`` 返回 False ⇒ **一个包都装不上**。这是对的：一台验不了签的
机器，不该因为验不了就接受任何东西。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PACK_SCHEMA = 1
MANIFEST_NAME = "manifest.json"
SIG_NAME = "manifest.sig"
FILES_PREFIX = "files/"

#: 包大小上限。技能是文本，一个包几十 KB 就够；20 MB 已经很宽松了。
#: 上限本身不是安全边界（签名才是），它挡的是「解压炸弹把磁盘写满」。
MAX_PACK_BYTES = 20 * 1024 * 1024
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_FILES = 500

_PACK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class K:
    """manifest 的字段名 —— **唯一真源**，读写两侧共用。

    见模块 docstring 里那条：名字分开写，两边各自都「对」，而链子是断的。
    """

    SCHEMA = "schema"
    PACK_ID = "pack_id"
    VERSION = "version"
    CREATED = "created"
    AUTHOR = "author"
    DESCRIPTION = "description"
    MIN_APP = "min_app_version"
    MAX_APP = "max_app_version"
    FILES = "files"
    #: files[] 里每项
    PATH = "path"
    SHA256 = "sha256"
    SIZE = "size"
    #: 签名不进被签的那份（``canonical_manifest_bytes`` 会剥掉它）
    SIGNATURE = "signature"


@dataclass
class PackFile:
    path: str          # 相对覆盖层目录，如 builtins/bias.py
    sha256: str
    size: int

    def to_dict(self) -> dict:
        return {K.PATH: self.path, K.SHA256: self.sha256, K.SIZE: self.size}

    @classmethod
    def from_dict(cls, d: dict) -> "PackFile":
        return cls(path=str(d.get(K.PATH) or ""),
                   sha256=str(d.get(K.SHA256) or "").lower(),
                   size=int(d.get(K.SIZE) or 0))


@dataclass
class PackManifest:
    pack_id: str = ""
    version: str = ""
    created: str = ""
    author: str = ""
    description: str = ""
    min_app_version: str = ""
    max_app_version: str = ""
    files: list[PackFile] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            K.SCHEMA: PACK_SCHEMA,
            K.PACK_ID: self.pack_id,
            K.VERSION: self.version,
            K.CREATED: self.created,
            K.AUTHOR: self.author,
            K.DESCRIPTION: self.description,
            K.MIN_APP: self.min_app_version,
            K.MAX_APP: self.max_app_version,
            K.FILES: [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PackManifest":
        return cls(
            pack_id=str(d.get(K.PACK_ID) or ""),
            version=str(d.get(K.VERSION) or ""),
            created=str(d.get(K.CREATED) or ""),
            author=str(d.get(K.AUTHOR) or ""),
            description=str(d.get(K.DESCRIPTION) or ""),
            min_app_version=str(d.get(K.MIN_APP) or ""),
            max_app_version=str(d.get(K.MAX_APP) or ""),
            files=[PackFile.from_dict(x) for x in (d.get(K.FILES) or [])
                   if isinstance(x, dict)],
        )


@dataclass
class VerifyResult:
    """``ok`` 之外还要说**为什么** —— 一句「验证失败」帮不了任何人。

    ``reasons`` 是全部问题，不是第一个。签名坏了同时还有两个文件对不上，
    用户应该一次看到三条，而不是修一条再撞一条。
    """

    ok: bool = False
    reasons: list[str] = field(default_factory=list)
    manifest: PackManifest | None = None
    signature: str = ""
    pubkey8: str = ""

    def describe(self) -> str:
        if self.ok and self.manifest is not None:
            return (f"{self.manifest.pack_id} v{self.manifest.version}"
                    f"（{len(self.manifest.files)} 个文件，签名 {self.pubkey8}）")
        return "包被拒绝：" + "；".join(self.reasons or ("原因不明",))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def is_valid_pack_id(pack_id: str) -> bool:
    return bool(_PACK_ID_RE.match(str(pack_id or "")))


def pack_filename(pack_id: str, version: str) -> str:
    return f"skillpack-{pack_id}-{version}.zip"


def _version_tuple(s: str) -> tuple[int, ...]:
    from mast.update.manifest import version_tuple
    return version_tuple(s)


def _check_entry_path(rel: str) -> str:
    """包里的条目路径合法吗。返回空串 = 合法，否则是拒绝理由。

    复用覆盖层自己的路径规则（``overlay.paths.is_valid_rel``）而不是另写一套 ——
    包里的东西最终就是覆盖层条目，两套规则会漂。额外禁掉 ``_packs/``：
    包里再套一层包没有意义，而且会让落点路径变成 ``_packs/<id>/_packs/…``。
    """
    from mast.skills.overlay import paths as P

    ok, why = P.is_valid_rel(rel)
    if not ok:
        return f"{rel}：{why}"
    if P.normalise_rel(rel).split("/")[0] == P.PACKS_DIR:
        return f"{rel}：包里不能再套 {P.PACKS_DIR}/"
    return ""


# ---------------------------------------------------------------------------
# 验证 —— 服务端、客户端、侧门脚本共用这一个函数
# ---------------------------------------------------------------------------

def verify_pack(zip_path: Path | str, *, public_key_hex: str = "",
                app_version: str = "", require_signature: bool = True
                ) -> VerifyResult:
    """把一个 ``.zip`` 从头验到尾。**永不抛。**

    顺序刻意是「先验签名，再碰文件」：签名不过就不该去解释 manifest 里的任何
    字段，更不该按它去读文件 —— 那些字段全是攻击者可控的。

    ``require_signature=False`` 只给一个场景：本机管理员用
    ``make_skillpack.py`` 打完包自查。**客户端永远不许传 False**，
    ``test_unsigned_pack_is_rejected`` 钉住这一点。
    """
    from mast.update import signing

    res = VerifyResult()
    p = Path(zip_path)
    if not p.is_file():
        res.reasons.append(f"文件不存在：{p}")
        return res
    if p.stat().st_size > MAX_PACK_BYTES:
        res.reasons.append(
            f"包 {p.stat().st_size / 1e6:.1f} MB，超过上限 "
            f"{MAX_PACK_BYTES / 1e6:.0f} MB")
        return res

    try:
        zf = zipfile.ZipFile(p)
    except (zipfile.BadZipFile, OSError) as exc:
        res.reasons.append(f"打不开：{type(exc).__name__}: {exc}")
        return res

    with zf:
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            res.reasons.append(f"包里没有 {MANIFEST_NAME}")
            return res
        try:
            raw = zf.read(MANIFEST_NAME)
            doc = json.loads(raw)
        except (KeyError, ValueError, OSError) as exc:
            res.reasons.append(f"{MANIFEST_NAME} 读不出来：{exc}")
            return res
        if not isinstance(doc, dict):
            res.reasons.append(f"{MANIFEST_NAME} 不是一个 JSON 对象")
            return res

        sig = ""
        if SIG_NAME in names:
            try:
                sig = str((json.loads(zf.read(SIG_NAME)) or {}).get(
                    K.SIGNATURE, "") or "")
            except (ValueError, OSError):
                sig = ""
        res.signature = sig

        # ── ① 签名。**在解释 manifest 的任何字段之前。** ────────────────
        if require_signature:
            pub = public_key_hex or signing.get_release_public_key()
            if not pub:
                res.reasons.append(
                    "这个构建里没有烘焙发布公钥 —— 验不了签就一个包都不装。"
                    "「验不了」不能当成「通过」。")
                return res
            if not sig:
                res.reasons.append(f"包里没有 {SIG_NAME}（未签名的包不装）")
                return res
            if not signing.HAVE_CRYPTO:
                res.reasons.append(
                    "本机没有 cryptography，验不了 Ed25519 签名 —— fail-closed，"
                    "一个包都不装。")
                return res
            if not signing.verify_manifest(doc, sig, pub):
                res.reasons.append(
                    "签名验证不通过 —— manifest 被改过，或者签名者不是发布方。")
                return res
            res.pubkey8 = pub[:8]

        # ── ② 到这里才敢读字段 ────────────────────────────────────────
        man = PackManifest.from_dict(doc)
        res.manifest = man
        if int(doc.get(K.SCHEMA) or 0) != PACK_SCHEMA:
            res.reasons.append(
                f"schema 是 {doc.get(K.SCHEMA)!r}，本版认得的是 {PACK_SCHEMA}")
        if not is_valid_pack_id(man.pack_id):
            res.reasons.append(
                f"pack_id {man.pack_id!r} 不合法（小写字母数字，2–64 位，"
                "可含 - 和 _）—— 它会成为磁盘上的目录名")
        if not man.version:
            res.reasons.append("manifest 没有 version")
        if not man.files:
            res.reasons.append("包里一个文件都没有")
        if len(man.files) > MAX_FILES:
            res.reasons.append(f"{len(man.files)} 个文件，超过上限 {MAX_FILES}")

        # ── ③ 版本范围 ──────────────────────────────────────────────
        if app_version:
            cur = _version_tuple(app_version)
            if man.min_app_version and cur < _version_tuple(man.min_app_version):
                res.reasons.append(
                    f"这个包要求 MAST ≥ {man.min_app_version}，本机是 {app_version}"
                    "（多半是它用到了本版还没有的 core 函数）")
            if man.max_app_version and cur > _version_tuple(man.max_app_version):
                res.reasons.append(
                    f"这个包声明只支持到 MAST {man.max_app_version}，"
                    f"本机是 {app_version}")

        # ── ④ 逐文件 ────────────────────────────────────────────────
        listed = {f.path for f in man.files}
        for f in man.files:
            why = _check_entry_path(f.path)
            if why:
                res.reasons.append("路径不合法：" + why)
                continue
            arc = FILES_PREFIX + f.path
            if arc not in names:
                res.reasons.append(f"manifest 里有 {f.path}，包里没有")
                continue
            try:
                data = zf.read(arc)
            except (KeyError, OSError) as exc:
                res.reasons.append(f"{f.path} 读不出来：{exc}")
                continue
            if len(data) > MAX_FILE_BYTES:
                res.reasons.append(
                    f"{f.path} {len(data)} 字节，超过单文件上限 {MAX_FILE_BYTES}")
                continue
            if f.size and len(data) != f.size:
                res.reasons.append(
                    f"{f.path} 大小对不上（manifest {f.size}，实际 {len(data)}）")
            got = sha256_bytes(data)
            if got != f.sha256:
                res.reasons.append(
                    f"{f.path} sha256 对不上 —— 文件被改过"
                    f"（manifest {f.sha256[:12]}，实际 {got[:12]}）")

        # ⑤ 包里有、manifest 里没有的文件 —— 夹带。签名覆盖不到它们。
        extra = sorted(n[len(FILES_PREFIX):] for n in names
                       if n.startswith(FILES_PREFIX) and not n.endswith("/")
                       and n[len(FILES_PREFIX):] not in listed)
        if extra:
            res.reasons.append(
                "包里夹带了 manifest 没登记的文件：" + "、".join(extra[:6])
                + "。签名保护的是 manifest，登记外的文件不受任何保护 —— 整包拒绝。")

    res.ok = not res.reasons
    return res


# ---------------------------------------------------------------------------
# 落盘后的复核 —— 「手放文件冒充签名版」这条路要关掉
# ---------------------------------------------------------------------------

INSTALLED_MANIFEST = ".pack.json"


def write_installed_manifest(pack_dir: Path, man: PackManifest,
                             signature: str) -> Path:
    """把 manifest 连同签名一起留在解出来的目录里。

    留着是为了**每次加载都能再验一遍**。只在下载时验一次的话，落盘之后谁改了
    文件都没人知道 —— 而覆盖层的 ``_packs/`` 会把里面的东西标成
    ``verified:<id>``。那个标记必须有凭据，否则它就是一句谎话。
    """
    doc = man.to_dict()
    doc[K.SIGNATURE] = signature
    p = pack_dir / INSTALLED_MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    import os
    os.replace(tmp, p)
    return p


def verify_installed(pack_dir: Path | str, *, public_key_hex: str = "",
                     require_signature: bool = True) -> VerifyResult:
    """复核一个**已经解出来**的 pack 目录。**永不抛。**

    加载时调用。它回答的是「这个目录现在还是当初验过的那份吗」——
    和下载时那次验证是两个不同的问题，中间隔着一段任何人都能写的时间。
    """
    from mast.update import signing

    res = VerifyResult()
    d = Path(pack_dir)
    mp = d / INSTALLED_MANIFEST
    if not mp.is_file():
        res.reasons.append(
            f"{d.name} 里没有 {INSTALLED_MANIFEST} —— 它不是通过签名通道装进来的"
            "（有人手工放的？）。这个目录下的东西会被标成「已签名」，所以整包拒绝。")
        return res
    try:
        doc = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        res.reasons.append(f"{INSTALLED_MANIFEST} 读不出来：{exc}")
        return res
    if not isinstance(doc, dict):
        res.reasons.append(f"{INSTALLED_MANIFEST} 不是一个 JSON 对象")
        return res

    sig = str(doc.get(K.SIGNATURE) or "")
    res.signature = sig
    if require_signature:
        pub = public_key_hex or signing.get_release_public_key()
        if not pub:
            res.reasons.append("这个构建里没有烘焙发布公钥 —— 验不了签就不加载。")
            return res
        if not signing.HAVE_CRYPTO:
            res.reasons.append("本机没有 cryptography，验不了签 —— fail-closed。")
            return res
        if not signing.verify_manifest(doc, sig, pub):
            res.reasons.append(
                f"{INSTALLED_MANIFEST} 的签名不过 —— 它被本地改动过。")
            return res
        res.pubkey8 = pub[:8]

    man = PackManifest.from_dict(doc)
    res.manifest = man
    for f in man.files:
        fp = d / f.path
        if not fp.is_file():
            res.reasons.append(f"{f.path} 不在了")
            continue
        try:
            got = sha256_bytes(fp.read_bytes())
        except OSError as exc:
            res.reasons.append(f"{f.path} 读不出来：{exc}")
            continue
        if got != f.sha256:
            res.reasons.append(
                f"{f.path} 被本地改动过（sha256 对不上）")
    res.ok = not res.reasons
    return res


__all__ = ["FILES_PREFIX", "INSTALLED_MANIFEST", "K", "MANIFEST_NAME",
           "MAX_FILES", "MAX_FILE_BYTES", "MAX_PACK_BYTES", "PACK_SCHEMA",
           "PackFile", "PackManifest", "SIG_NAME", "VerifyResult",
           "is_valid_pack_id", "pack_filename", "sha256_bytes",
           "verify_installed", "verify_pack", "write_installed_manifest"]
