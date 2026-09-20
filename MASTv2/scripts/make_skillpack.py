"""把覆盖层里的 skill 打成**签名**技能包，推给其它实验室机器。

在**管理员机器**上跑。私钥离线放在 ``<data>/api key/release_signing_key.pem``，
和 OTA 发布用的是同一把 —— 客户端里烘焙的公钥只有一个，另起一把就等于要求
所有机器重新出厂。

    # 把覆盖层里两个改好的技能打成一个包
    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/make_skillpack.py \\
        --pack-id atomic-res --version 1.0.0 \\
        builtins/bias.py builtins/scan_prep.py

    # 不给文件名 = 打包覆盖层清单里**已启用**的全部条目
    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/make_skillpack.py \\
        --pack-id atomic-res --version 1.0.0 --all-enabled

打完就地自查
==========
打完立刻用**客户端那一个** :func:`~mast.update.skillpack.verify_pack` 验一遍，
不另写一套检查。管理员这边「打包时觉得对」而客户端「验的时候不对」，是这类
工具最典型的失败方式，而它只会在推出去之后才被发现。

不签就不出包
==========
没有私钥时直接失败，**不产出一个未签名的 zip**。产出了，它就会被人拷来拷去，
最后落到某台机器上，而那台机器只会说「未签名的包不装」—— 一次没必要的困惑。
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "MASTv2"))

from mast.update import signing, skillpack as SP     # noqa: E402
from mast.update.skillpack import K, PackFile, PackManifest   # noqa: E402


def _private_key_path() -> Path:
    from mast._runtime_paths import project_root
    return project_root() / "api key" / "release_signing_key.pem"


def collect(rels: list[str], *, all_enabled: bool) -> list[tuple[str, bytes]]:
    """``[(覆盖层相对路径, 字节)]``。读不到就抛，不静默跳过。"""
    from mast.skills.overlay import manifest as M, paths as P

    if all_enabled:
        rels = [e.path for e in M.load().entries if e.enabled]
        if not rels:
            raise SystemExit(
                "覆盖层清单里没有启用的条目 —— 没有东西可打包。"
                "（--all-enabled 打的是**已启用**的那些，不是目录里所有文件。）")
    out: list[tuple[str, bytes]] = []
    for rel in rels:
        rel = P.normalise_rel(rel)
        why = SP._check_entry_path(rel)
        if why:
            raise SystemExit("路径不合法：" + why)
        p = P.entry_path(rel)
        if not p.is_file():
            raise SystemExit(f"{p} 不存在 —— 覆盖层里没有这个条目。")
        out.append((rel, p.read_bytes()))
    return out


def build(pack_id: str, version: str, items: list[tuple[str, bytes]], *,
          author: str = "", description: str = "",
          min_app: str = "", max_app: str = "",
          out_dir: Path | None = None) -> Path:
    if not SP.is_valid_pack_id(pack_id):
        raise SystemExit(
            f"pack_id {pack_id!r} 不合法：小写字母数字 2–64 位，可含 - 和 _。"
            "（它会成为操作员机器上的目录名。）")

    man = PackManifest(
        pack_id=pack_id, version=version,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        author=author, description=description,
        min_app_version=min_app, max_app_version=max_app,
        files=[PackFile(path=rel, sha256=SP.sha256_bytes(data), size=len(data))
               for rel, data in items],
    )

    key_path = _private_key_path()
    if not key_path.is_file():
        raise SystemExit(
            f"找不到私钥 {key_path} —— 不签就不出包。\n"
            "  产出一个未签名的 zip，它会被拷来拷去，最后在某台机器上被拒，"
            "而那时已经没人记得它是怎么来的。")
    if not signing.HAVE_CRYPTO:
        raise SystemExit("本机没有 cryptography，签不了名。pip install cryptography")
    sig = signing.sign_manifest(man.to_dict(), key_path.read_text(encoding="utf-8"))

    out_dir = out_dir or (REPO_ROOT / "dist" / "skillpacks")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / SP.pack_filename(pack_id, version)
    tmp = target.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(SP.MANIFEST_NAME,
                   json.dumps(man.to_dict(), ensure_ascii=False, indent=1))
        z.writestr(SP.SIG_NAME,
                   json.dumps({K.SIGNATURE: sig}, ensure_ascii=False))
        for rel, data in items:
            z.writestr(SP.FILES_PREFIX + rel, data)
    import os
    os.replace(tmp, target)

    # ── 就地自查：**两问**，用的都是客户端那一个 verify_pack ────────────
    #
    # ① 用这把私钥对应的公钥验 —— 「签名这一步本身对不对」。不过就是打包器有 bug。
    # ② 用**构建里烘焙的**发布公钥验 —— 「客户端会不会接受它」。
    #
    # 只做 ①，会打出一个每台机器都拒装的包，而拒绝发生在离错误很远的地方
    # （推过去之后）。只做 ②，一次用错私钥看起来会和打包器坏掉一模一样。
    from mast.api.version import get_version

    # ⚠️ 自查**不传 app_version**。版本门回答的是「**目标**机器能不能装」，而
    # 管理员机器的版本和实验室机器本来就可能不同 —— 一个特意声明
    # min_app_version=9.0.0（给还没发的版本准备的）的包，在打包时被自己拦下来
    # 是荒谬的。那道门属于客户端。
    own_pub = signing.public_key_from_private_pem(
        key_path.read_text(encoding="utf-8"))
    res = SP.verify_pack(target, public_key_hex=own_pub)
    if not res.ok:
        target.unlink(missing_ok=True)
        raise SystemExit(
            "打出来的包用自己的密钥都验不过，已删除：\n  - "
            + "\n  - ".join(res.reasons)
            + "\n（用的就是客户端那个 verify_pack —— 这说明打包这一步有问题。）")

    baked = signing.get_release_public_key()
    if not baked:
        print("! 这个构建里没有烘焙发布公钥，验不了「客户端会不会接受」。"
              "推之前请在一台真装过的机器上确认。")
    elif baked != own_pub:
        raise SystemExit(
            f"签名用的私钥（公钥 {own_pub[:8]}）和这个构建烘焙的发布公钥"
            f"（{baked[:8]}）不是一对 —— 打出来的包在每一台机器上都会被拒。\n"
            "  发布签名只有一把钥匙；另起一把等于要求所有机器重新出厂。\n"
            f"  私钥：{key_path}")
    print("已打包并自查通过：" + str(target))
    print("  " + res.describe())
    if man.min_app_version or man.max_app_version:
        # 声明了版本范围就把它念出来 —— 它决定哪些机器会**拒装**这个包，
        # 而那件事发生在很远的地方（推过去之后），到时候没人会想起这里填过什么。
        rng = (man.min_app_version or "不限") + " – " + (man.max_app_version or "不限")
        print(f"  适用 MAST 版本：{rng}（本机 {get_version()}）")
        cur = SP._version_tuple(get_version())
        if ((man.min_app_version and cur < SP._version_tuple(man.min_app_version))
                or (man.max_app_version
                    and cur > SP._version_tuple(man.max_app_version))):
            print("  ! 本机版本不在这个范围内 —— 这个包装不到这台机器上"
                  "（如果是特意给别的版本打的，那就没问题）")
    for f in man.files:
        print(f"    {f.path}  {f.sha256[:12]}  {f.size} 字节")
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rels", nargs="*", help="覆盖层相对路径，如 builtins/bias.py")
    ap.add_argument("--pack-id", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--all-enabled", action="store_true",
                    help="打包覆盖层清单里全部**已启用**的条目")
    ap.add_argument("--author", default="")
    ap.add_argument("--description", default="")
    ap.add_argument("--min-app-version", default="",
                    help="低于这个版本的 MAST 会拒装（技能用到了新的 core 函数时填）")
    ap.add_argument("--max-app-version", default="")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    if not a.rels and not a.all_enabled:
        ap.error("要么给出文件路径，要么用 --all-enabled")
    items = collect(list(a.rels), all_enabled=bool(a.all_enabled))
    build(a.pack_id, a.version, items, author=a.author,
          description=a.description, min_app=a.min_app_version,
          max_app=a.max_app_version, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
