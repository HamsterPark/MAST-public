"""把 ``mast/**.py`` 作为 data 收进冻结包 —— 并且**不带凭据**。

为什么要带源码
==============
PyInstaller 把全部 ``.py`` 编译进 ``MAST.exe`` 内嵌的 PYZ 归档，冻结树里
``_internal/mast/`` 只剩十来个 data 文件。可是有两件事需要的是**文件**，不是模块：

* ``mast.pyexec`` 要把 ``sitecustomize.py`` / ``mastdata.py`` /
  ``io/nanonis_files.py`` **拷进会话目录** —— 分析子进程里没有 MAST，只能拿文件；
* skill 覆盖层的「导出到覆盖层」要拿到某个 skill 的源文件原文。

两个需求共用同一份副本，落点 ``mast/_src/``（对应
``mast.pyexec._srcfiles.source_root()``）。

⚠️ 凭据不能跟着源码走
=====================
``mast/update/_defaults.py`` 里有 **``DEFAULT_TOKEN``（推送服务器明文 token）**，
它被 ``.gitignore`` 忽略 —— gitignore 恰好标记了「本地生成、不该分发」的东西。
把它随源码带进包，再给 DP 的分析子进程一个只读视图，就等于把推送凭据交给一个
会写代码的 agent。

``mast2_build.ps1`` Step 4 本来就有「删掉打包进去的 ``api key\\*.env``」这道纪律，
但**源码通道会绕过它**。所以这里三道，缺一不可：

1. **判据而不是名单** —— 跳过任何被 git 忽略的文件。以后新增的本地生成物自动被
   挡住，不用有人记得去改名单。
2. **凭据扫描** —— 对最终收进来的文件扫敏感赋值，命中就**中止构建**。
3. 产物侧断言在 ``verify_frozen_artifact.py``（跑在真产物上；dev 树和发布产物里
   ``_defaults.py`` 的内容不同）。

第 1 道就足以挡住 ``_defaults.py``；第 2 道是「有人 ``git add -f`` 了它」时的保险。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

#: 收进包的源码落点（相对 ``_internal/``）。与
#: ``mast.pyexec._srcfiles.source_root()`` 的 ``<package_root>/_src`` 对应。
DEST_ROOT = os.path.join("mast", "_src")

#: 少于这个数就是收漏了 —— 走目录收进来的东西缺失时，一条丢弃记录都不会有。
MIN_FILES = 700

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

#: 敏感常量：**大写常量名** = **足够长的字符串字面量**。
#: 只认字面量赋值 —— ``token = get_token()`` 不是凭据，是代码。
#:
#: 两个跨行子句，作用**相反**，各管一头（一次变异实测把这件事逼出来了）：
#:
#: * ``[^=]*?`` —— 赋值号到引号之间**允许换行**。少了它，
#:   ``TOKEN = (\n    "real")`` 这种括号续行的真凭据会被漏掉。
#: * ``[^\"'\n]{12,}`` —— 引号里的内容**不许跨行**。少了它，
#:   ``DEFAULT_TOKEN = ""`` 会拿第二个引号当开头，一路吃到几行之后另一个字符串
#:   的引号，报出一个根本不存在的「凭据」（``mast/update/defaults.py`` 实测）。
#:
#: 一度把两处都写成不许跨行 —— 误报确实没了，但括号续行的真凭据也一起漏了，
#: 而那**没有任何测试会红**。承重的是第二个子句，第一个反而降低了召回。
#:
#: ⚠️ **前缀是 ``[A-Z0-9_]*``（可为空），不是 ``[A-Z][A-Z0-9_]*``。**
#: 后者要求关键词**前面至少有一个字符**，于是裸的 ``TOKEN = "…"`` /
#: ``API_KEY = "…"`` / ``SECRET_KEY = "…"`` 全都漏掉 —— 只有 ``DEFAULT_TOKEN``
#: 这种带前缀的才命中。2026-08-21 按验证清单 修复项 ③ 故意在 ``mast/`` 下放一个含
#: ``TOKEN = "<50 字符>"`` 的文件，**构建照常走完，守卫一声没吭**；逐个试下来
#: 13 个现实命名里漏 10 个。一道只在「名字恰好有前缀」时才生效的凭据守卫，
#: 等于没有 —— 而它是**唯一**挡住「把推送 token 交给一个会写代码的 agent」的东西。
_CRED_ASSIGN = re.compile(
    r"^\s*([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY"
    r"|PRIVATE_KEY|CREDENTIAL)[A-Z0-9_]*)\s*[:=][^=]*?[\"']([^\"'\n]{12,})[\"']",
    re.MULTILINE,
)

#: 显式放行。每一条都要有理由 —— 形状照抄 ``mast2.spec`` 的 ``_OPTIONAL_DATA``。
CREDENTIAL_ALLOWLIST = {
    # Ed25519 **公钥**。公开是它的用途：客户端拿它验发布签名。
    "DEFAULT_SIGNING_PUBKEY",
}

#: 明显的占位/空值，不是凭据。用**子串**判据而不是完整匹配 ——
#: ``CHANGEME_BEFORE_USE`` 这种带后缀的写法很常见，要求完整匹配会把它报成凭据。
#:
#: 权衡说清楚：这条放宽会让一个恰好含 "example" 的真 token 溜过去。接受，因为
#: 误报的代价更高 —— 一道会误报的守卫，人会学着忽略它，那时它挡不住任何东西。
#: 真 token 还有第二道（gitignore 判据）和第三道（产物侧断言）。
_PLACEHOLDER = re.compile(
    r"<[^>]*>|CHANGE_?ME|REPLACE_?ME|PLACEHOLDER|YOUR[_-]|DUMMY|EXAMPLE|"
    r"SAMPLE|TODO|FIXME|^x{4,}$|^\.{3,}$",
    re.IGNORECASE)


#: 值看起来是**一个位置**（文件名或路径），不是一个秘密。
#:
#: 这条是把名字前缀改成可选（见 :data:`_CRED_ASSIGN`）之后立刻需要的：
#: 加宽召回之后 ``TOKEN_FILE = "update_client_token.env"`` 这种**指向凭据的
#: 文件名**开始中招。全仓扫下来同形的正好两处（`update/client.py`、
#: `update/server.py`），值都以 ``.env`` 结尾。
#:
#: 判据放在**值**上而不是名字上，是因为要分的正是「秘密 vs 位置」这件事：
#: ``TOKEN_FILE`` 这个名字本身不保证安全（``TOKEN_FILE = "<真 token>"`` 仍该拦），
#: 而一个以 ``.env`` 结尾、或含路径分隔符的值，不可能是 base64/hex 的凭据本体。
#:
#: 权衡说清楚：一个恰好以这些后缀结尾的真 token 会溜过去。接受 —— 与
#: :data:`_PLACEHOLDER` 同一条理由，而且真 token 还有 gitignore 与产物侧两道。
_LOCATOR_VALUE = re.compile(
    r"[\\/]|\.(?:env|json|pem|txt|key|cfg|ini|ya?ml|toml|log|db|sqlite3?)$",
    re.IGNORECASE)


def iter_source_files(mast_pkg: Path):
    """``mast/`` 下所有 ``.py``（跳过缓存目录）。"""
    for root, dirs, files in os.walk(mast_pkg):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in sorted(files):
            if f.endswith(".py"):
                yield Path(root) / f


def gitignored(repo_root: Path, paths: list[Path]) -> tuple[set[Path], str]:
    """哪些文件被 git 忽略。返回 ``(被忽略的集合, 降级说明)``。

    用 ``git check-ignore -z --stdin`` 一次问完（每个文件问一次会很慢）。
    git 不可用时返回空集 **和一句说明** —— 调用方必须把它当成降级报出来，
    而不是当成「没有被忽略的文件」。
    """
    if not paths:
        return set(), ""
    try:
        # ⚠️ 必须 -z。默认输出会把含特殊字符的路径**加引号并 C-style 转义**
        # （core.quotePath）—— Windows 路径里的每一个反斜杠都触发它，于是回来的是
        # 带引号、反斜杠双写的字符串，集合比对一个都对不上。表现为「gitignore
        # 这道守卫静默失效」：_defaults.py 照样被收进包，而没有任何一行输出提到它。
        # -z 用 NUL 分隔，不加引号、不转义。
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "-z", "--stdin"],
            input="\0".join(str(p) for p in paths),
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return set(), f"git check-ignore 跑不起来（{exc}）"
    # 退出码：0 = 有命中，1 = 无命中，其它 = 出错
    if proc.returncode not in (0, 1):
        return set(), f"git check-ignore 出错（exit {proc.returncode}）：{proc.stderr.strip()[:200]}"
    out = {Path(s) for s in proc.stdout.split("\0") if s.strip()}
    return out, ""


def scan_credentials(path: Path) -> list[tuple[str, str]]:
    """找出这个文件里像凭据的赋值。返回 ``[(常量名, 值预览), …]``。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    hits: list[tuple[str, str]] = []
    for name, value in _CRED_ASSIGN.findall(text):
        if name in CREDENTIAL_ALLOWLIST:
            continue
        v = value.strip()
        if _PLACEHOLDER.search(v):
            continue
        if _LOCATOR_VALUE.search(v):
            continue
        hits.append((name, value[:8] + "…"))
    return hits


class BundleReport:
    """收集结果 —— 包括**没**收的那些和为什么。"""

    def __init__(self):
        self.files: list[tuple[str, str]] = []      # (src, dest_dir)
        self.skipped_gitignored: list[Path] = []
        self.violations: list[tuple[Path, str, str]] = []
        self.degraded: str = ""

    @property
    def n(self) -> int:
        return len(self.files)


def collect(mastv2_root: str | Path, repo_root: str | Path) -> BundleReport:
    """收集 ``mast/**.py`` 的 (src, dest) 列表 + 跳过的 + 违规的。

    **不**在这里抛异常：调用方（spec / 测试）各自决定怎么处理，但两边看到的是
    同一份结果 —— spec 里写一份收集逻辑、测试里再抄一份，是本仓「同一个问题两个
    地方回答」的形状。
    """
    mastv2_root = Path(mastv2_root)
    repo_root = Path(repo_root)
    mast_pkg = mastv2_root / "mast"
    rep = BundleReport()

    all_files = list(iter_source_files(mast_pkg))
    ignored, degraded = gitignored(repo_root, all_files)
    rep.degraded = degraded

    for src in all_files:
        if src in ignored:
            rep.skipped_gitignored.append(src)
            continue
        for name, preview in scan_credentials(src):
            rep.violations.append((src, name, preview))
        rel_dir = src.parent.relative_to(mast_pkg)
        dest = os.path.join(DEST_ROOT, str(rel_dir)) if str(rel_dir) != "." else DEST_ROOT
        rep.files.append((str(src), dest))
    return rep


def main(argv: list[str] | None = None) -> int:
    """`python bundle_sources.py [repo_root]` —— 本地干跑一次，看会收什么。"""
    argv = list(argv if argv is not None else sys.argv[1:])
    repo = Path(argv[0]) if argv else Path(__file__).resolve().parents[2]
    rep = collect(repo / "MASTv2", repo)
    print(f"收集 {rep.n} 个 .py -> {DEST_ROOT}")
    if rep.degraded:
        print(f"!! 降级：{rep.degraded}")
    if rep.skipped_gitignored:
        print(f"跳过 {len(rep.skipped_gitignored)} 个 git 忽略的文件：")
        for p in rep.skipped_gitignored:
            print(f"    {p}")
    if rep.violations:
        print("!! 疑似凭据：")
        for p, name, preview in rep.violations:
            print(f"    {p}: {name} = {preview}")
        return 1
    if rep.n < MIN_FILES:
        print(f"!! 只收到 {rep.n} 个（期望 >= {MIN_FILES}）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
