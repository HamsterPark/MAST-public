"""CLI entry: ``python -m mast.update publish <setup.exe> --version X.Y.Z``."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _data_root_from_argv(default_arg: str | None) -> Path:
    if default_arg:
        return Path(default_arg).resolve()
    import os
    env = os.environ.get("MAST2_USER_ROOT", os.environ.get("MAST_USER_ROOT", "")).strip()
    if env:
        return Path(env).resolve()
    return Path.cwd()


def cmd_publish(args: argparse.Namespace) -> int:
    from mast.update.manifest import make_manifest, write_manifest
    from mast.update.server import push_dir

    setup = Path(args.setup_exe).resolve()
    if not setup.exists():
        print(f"ERROR: not found: {setup}", file=sys.stderr)
        return 2

    data_root = _data_root_from_argv(args.data_root)
    print(f"Data root: {data_root}")
    pdir = push_dir(data_root)

    target = pdir / setup.name
    if setup.resolve() != target.resolve():
        print(f"Copying {setup.name} → {target}")
        shutil.copy2(setup, target)

    print(f"Computing SHA256 of {target.name} ...")
    m = make_manifest(
        target, version=args.version,
        release_notes_url=args.release_notes_url or "",
        min_force_install=args.min_force_install or "0.0.0",
    )

    # Optional incremental (delta) package(s): --delta a.zip:1.2.3 [--delta b.zip:1.2.2]
    from mast.update.manifest import make_delta_descriptor
    for spec in (args.delta or []):
        if ":" not in spec:
            print(f"ERROR: --delta must be PATH:FROM_VERSION, got {spec!r}", file=sys.stderr)
            return 2
        dpath_s, from_v = spec.rsplit(":", 1)
        dpath = Path(dpath_s).resolve()
        if not dpath.exists():
            print(f"ERROR: delta not found: {dpath}", file=sys.stderr)
            return 2
        dtarget = pdir / dpath.name
        if dpath.resolve() != dtarget.resolve():
            print(f"Copying delta {dpath.name} → {dtarget}")
            shutil.copy2(dpath, dtarget)
        desc = make_delta_descriptor(dtarget, from_v)
        m.deltas.append(desc)
        print(f"  + delta from {from_v}: {desc['filename']} "
              f"({desc['size_bytes']} B, sha {desc['sha256'][:12]}…)")

    write_manifest(pdir / "manifest.json", m)
    signed = _sign_published_manifest(m, args, data_root, pdir)
    print("\nPublished:")
    print(f"  version            {m.version}")
    print(f"  filename           {m.filename}")
    print(f"  sha256             {m.sha256}")
    print(f"  size_bytes         {m.size_bytes}")
    print(f"  deltas             {len(m.deltas)}")
    print(f"  signed             {signed}")
    print(f"  published_at       {m.published_at}")
    print(f"  min_force_install  {m.min_force_install}")
    print(f"\nManifest written:  {pdir / 'manifest.json'}")
    print(f"Setup hosted at:   {target}")
    return 0


def _sign_published_manifest(m, args: argparse.Namespace, data_root: Path, pdir: Path) -> bool:
    """Sign the just-written manifest with the OFFLINE Ed25519 private key and
    write ``manifest.sig`` next to it. Key source: ``--sign-key`` or
    ``<data>/api key/release_signing_key.pem``. Skips (loud warning) when no key
    / no cryptography so a transition still publishes — but a client with an
    embedded public key will then REJECT the unsigned version."""
    from mast.update.signing import HAVE_CRYPTO, sign_manifest

    key_path = None
    if getattr(args, "sign_key", None):
        key_path = Path(args.sign_key).resolve()
    else:
        default_key = data_root / "api key" / "release_signing_key.pem"
        if default_key.exists():
            key_path = default_key
    sig_file = pdir / "manifest.sig"
    if key_path is None:
        sig_file.unlink(missing_ok=True)  # never ship a stale sig for a new manifest
        print("\n⚠️  未签名发布：无签名私钥（--sign-key 或 <data>/api key/"
              "release_signing_key.pem 缺失）。已内嵌公钥的客户端会拒绝此版本。"
              "先运行 `python -m mast.update keygen`。", file=sys.stderr)
        return False
    if not HAVE_CRYPTO:
        sig_file.unlink(missing_ok=True)
        print("\n⚠️  未签名：cryptography 不可用（pip install cryptography>=43）。",
              file=sys.stderr)
        return False
    try:
        sig = sign_manifest(m.to_dict(), key_path.read_text(encoding="utf-8"))
        sig_file.write_text(sig + "\n", encoding="utf-8")
        print(f"  signature          {sig[:16]}… (manifest.sig)")
        return True
    except Exception as exc:  # noqa: BLE001
        sig_file.unlink(missing_ok=True)
        print(f"\n⚠️  签名失败（未写 manifest.sig）：{exc}", file=sys.stderr)
        return False


def cmd_keygen(args: argparse.Namespace) -> int:
    """One-time: generate the Ed25519 release keypair. Private key → offline file;
    public key hex → paste into installer/push_defaults_mast2.json (baked into
    clients so they verify signatures)."""
    from mast.update.signing import HAVE_CRYPTO, generate_keypair

    if not HAVE_CRYPTO:
        print("ERROR: 需要 cryptography（pip install cryptography>=43）", file=sys.stderr)
        return 2
    data_root = _data_root_from_argv(args.data_root)
    key_path = (Path(args.out).resolve() if args.out
                else data_root / "api key" / "release_signing_key.pem")
    if key_path.exists() and not args.force:
        print(f"ERROR: 私钥已存在：{key_path}（--force 覆盖）", file=sys.stderr)
        return 2
    pem, pub_hex = generate_keypair()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(pem, encoding="utf-8")
    try:
        import os
        os.chmod(key_path, 0o600)  # best-effort (POSIX); harmless on Windows
    except Exception:  # noqa: BLE001
        pass
    print("✅ 已生成 Ed25519 发布签名密钥对")
    print(f"  私钥（离线保存，切勿提交/放到推送服务器）：{key_path}")
    print("\n  公钥 HEX —— 填进 installer/push_defaults_mast2.json 的 "
          "\"signing_pubkey\"，随构建烧进客户端：")
    print(f"  {pub_hex}\n")
    return 0


def cmd_publish_token(args: argparse.Namespace) -> int:
    """生成**发布**凭据 —— 只有它能调 ``POST /skills/pack/publish``。

    与客户端 token 分开的理由：客户端那把随安装包分发（打包时写进
    ``mast/update/_defaults.py``，编进 PYZ），等价于公开值。用它守发布端点，
    等于每台装了 MAST 的机器都能往推送服务器发布技能包。

    这把只留在发布机上：不写进 ``push_defaults*.json``，不随包分发。
    """
    from mast.update.server import (PUBLISH_TOKEN_FILE, _read_publish_token,
                                    generate_token, write_publish_token)

    data_root = _data_root_from_argv(args.data_root)
    existing = _read_publish_token(data_root)
    if existing and not args.force:
        print(f"ERROR: 发布凭据已存在：{data_root / 'api key' / PUBLISH_TOKEN_FILE}"
              "（--force 覆盖；覆盖后所有发布方都要换成新值）", file=sys.stderr)
        return 2
    tok = generate_token()
    p = write_publish_token(data_root, tok)
    print("✅ 已生成发布凭据")
    print(f"  文件（只留在发布机上）：{p}")
    print(f"  值：{tok}")
    print("\n  发布时带上它：")
    print(f'    curl -H "Authorization: Bearer {tok}" \\')
    print('         --data-binary @pack.zip <server>/skills/pack/publish')
    print("\n  ⚠️ 不要把它写进 push_defaults*.json 或任何随包分发的文件 ——")
    print("     那样它就和客户端 token 一样公开了，这次分离也就白做了。\n")
    return 0


def cmd_delta(args: argparse.Namespace) -> int:
    """Build a delta zip: old_root → new_root (the onedir ROOT that contains
    MAST.exe + _internal/, NOT the _internal subdir — the client applies the delta
    to the install root, so paths must be root-relative incl. MAST.exe)."""
    from mast.update.delta import build_delta
    out = Path(args.out).resolve()
    man = build_delta(Path(args.old).resolve(), Path(args.new).resolve(), out,
                      from_version=args.from_version, to_version=args.to_version)
    print(f"Delta written: {out}")
    print(f"  +{len(man['added'])} added  ~{len(man['changed'])} changed  "
          f"-{len(man['removed'])} removed  ({out.stat().st_size} B)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from mast.update.manifest import load_manifest
    from mast.update.server import push_dir

    data_root = _data_root_from_argv(args.data_root)
    m = load_manifest(push_dir(data_root) / "manifest.json")
    if m is None:
        print("(no manifest published yet)")
        return 0
    print(m.to_json())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mast.update",
        description="MAST 内网推送升级 — 服务器端发布工具",
    )
    parser.add_argument("--data-root", help="MAST 数据目录（默认从环境/当前目录推断）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pub = sub.add_parser("publish", help="发布一个新版本到推送服务器目录")
    p_pub.add_argument("setup_exe", help="MAST 安装包 .exe 路径")
    p_pub.add_argument("--version", required=True, help="版本号 (e.g. 2.0.0)")
    p_pub.add_argument("--release-notes-url", default="", help="release 链接 (可选)")
    p_pub.add_argument("--min-force-install", default="0.0.0", help="强制升级下界")
    p_pub.add_argument("--delta", action="append", metavar="PATH:FROM_VERSION",
                       help="增量包及其升级起始版本,可多次 (e.g. delta.zip:2.1.13)")
    p_pub.add_argument("--sign-key", help="Ed25519 私钥 PEM 路径 "
                       "(默认 <data>/api key/release_signing_key.pem)")
    p_pub.set_defaults(func=cmd_publish)

    p_kg = sub.add_parser("keygen", help="生成 Ed25519 发布签名密钥对(一次性)")
    p_kg.add_argument("--out", help="私钥输出路径(默认 <data>/api key/release_signing_key.pem)")
    p_kg.add_argument("--force", action="store_true", help="覆盖已存在的私钥")
    p_kg.set_defaults(func=cmd_keygen)

    p_pt = sub.add_parser("publish-token",
                          help="生成**发布**凭据（只有它能发布技能包；不随包分发）")
    p_pt.add_argument("--data-root", default="")
    p_pt.add_argument("--force", action="store_true", help="已存在时覆盖")
    p_pt.set_defaults(func=cmd_publish_token)

    p_st = sub.add_parser("status", help="显示当前发布的 manifest")
    p_st.set_defaults(func=cmd_status)

    p_d = sub.add_parser("delta", help="构建增量包 old_root → new_root（onedir 根目录）")
    p_d.add_argument("old", help="旧版 onedir 根目录（含 MAST.exe + _internal/）")
    p_d.add_argument("new", help="新版 onedir 根目录（含 MAST.exe + _internal/）")
    p_d.add_argument("out", help="输出 delta zip 路径")
    p_d.add_argument("--from-version", required=True, dest="from_version")
    p_d.add_argument("--to-version", required=True, dest="to_version")
    p_d.set_defaults(func=cmd_delta)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
