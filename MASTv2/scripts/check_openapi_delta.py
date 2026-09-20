"""并发写 `openapi.json` 的守门人：**你把别人的端点吞掉了吗？**

为什么 `gen:api:check` 不够
--------------------------

`npm run gen:api:check` 回答的是「你重新生成了吗」（重生成 + ``git diff --exit-code``）。
它**不回答**「你把别人未提交的端点一起卷走、然后又把它删掉了吗」——
而后者才是多个 agent 同时改 API 时的真实失败模式，并且它是**静默的**：

  吞掉之后 typecheck 照样过，因为 ``schema.d.ts`` 是从同一份被吞过的
  ``openapi.json`` 生成的 —— **两边自洽地一起错**。

也不能靠看 ``git diff`` 的行数。这份文件三万行、键序一变整份重排：同一次改动，
本仓 2026-08-04 一天之内先后量到 **26,227 行**和 **402 行**两个数字。
**行数不是信号，结构才是。**

用法
----

    # 默认：把工作树的 openapi.json 和 HEAD 里的那份比
    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/check_openapi_delta.py

    # 指定两边（ref 形如 HEAD / origin/main / <sha>，也可以直接给文件路径）
    ... check_openapi_delta.py --base origin/main
    ... check_openapi_delta.py --base a.json --head b.json

退出码
------

* ``0`` —— 没有任何**删除**。新增多少都正常（那就是你干的活）。
* ``1`` —— 有路径 / schema / 字段被删掉了。**这就是「吞掉了别人的东西」的信号。**
  正确的处置是回去看那几条是谁的，而不是把它们重新生成一遍就算了。
* ``2`` —— 取数失败（ref 不存在、JSON 坏了）。

已知边界（如实记）
----------------

只比「有没有」，不比「内容对不对」：一个路径的响应模型被换成了别的，
这里只会报 ``paths changed``，不会失败。真要那一层就得逐字段递归比，
而那会把这个工具从「三十秒看一眼」变成「另一个要维护的东西」。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REL = "frontend/openapi.json"


def _load(spec: str) -> dict:
    """从 git ref 或文件路径读一份 openapi。

    先当文件试 —— 一个叫 ``HEAD`` 的文件不存在，而一个叫 ``a.json`` 的 ref 也不存在，
    所以「文件优先、失败再当 ref」两边都不会误判。
    """
    p = Path(spec)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    ref = spec if ":" in spec else f"{spec}:{_DEFAULT_REL}"
    out = subprocess.run(["git", "show", ref], capture_output=True, cwd=_REPO_ROOT)
    if out.returncode != 0:
        raise SystemExit(
            f"[取数失败] 既不是文件也读不出这个 git ref：{spec}\n"
            f"        git show {ref} → {out.stderr.decode('utf-8', 'replace').strip()}")
    return json.loads(out.stdout.decode("utf-8"))


def _props(schema: Any) -> set[str]:
    return set(schema.get("properties", {})) if isinstance(schema, dict) else set()


def compare(base: dict, head: dict) -> dict:
    """两份 openapi 的结构差异。纯函数，好测。"""
    bp, hp = set(base.get("paths", {})), set(head.get("paths", {}))
    bs = set(base.get("components", {}).get("schemas", {}))
    hs = set(head.get("components", {}).get("schemas", {}))
    b_sch = base.get("components", {}).get("schemas", {})
    h_sch = head.get("components", {}).get("schemas", {})

    changed: dict[str, dict] = {}
    for name in sorted(bs & hs):
        if b_sch[name] == h_sch[name]:
            continue
        a, b = _props(b_sch[name]), _props(h_sch[name])
        changed[name] = {"added": sorted(b - a), "removed": sorted(a - b)}

    return {
        "paths_added": sorted(hp - bp),
        "paths_removed": sorted(bp - hp),
        "paths_changed": sorted(k for k in (bp & hp) if base["paths"][k] != head["paths"][k]),
        "schemas_added": sorted(hs - bs),
        "schemas_removed": sorted(bs - hs),
        "schemas_changed": changed,
        "n_paths": (len(bp), len(hp)),
        "n_schemas": (len(bs), len(hs)),
    }


def removals(d: dict) -> list[str]:
    """所有「东西不见了」的条目。非空 = 退出码 1。"""
    out = [f"路径被删除：{p}" for p in d["paths_removed"]]
    out += [f"schema 被删除：{s}" for s in d["schemas_removed"]]
    for name, ch in d["schemas_changed"].items():
        out += [f"{name}.{f} 字段被删除" for f in ch["removed"]]
    return out


def render(d: dict, base: str, head: str) -> str:
    L = [f"═══ openapi 结构比对：{base} → {head} ═══",
         f"  路径 {d['n_paths'][0]} → {d['n_paths'][1]}"
         f"   schema {d['n_schemas'][0]} → {d['n_schemas'][1]}", ""]

    def block(title: str, items: list[str]) -> None:
        L.append(f"  {title:16s}: {items if items else '（无）'}")

    block("paths added", d["paths_added"])
    block("paths removed", d["paths_removed"])
    block("paths changed", d["paths_changed"])
    block("schemas added", d["schemas_added"])
    block("schemas removed", d["schemas_removed"])
    if d["schemas_changed"]:
        L.append("  schemas changed :")
        for name, ch in d["schemas_changed"].items():
            L.append(f"      {name}: +{ch['added']} -{ch['removed']}")
    else:
        L.append("  schemas changed : （无）")

    bad = removals(d)
    L.append("")
    if bad:
        L.append("  ✗ 有东西不见了：")
        L += [f"      {b}" for b in bad]
        L.append("")
        L.append("  这是「重新生成时把别人未提交的端点卷走又删掉」的典型形状。")
        L.append("  别直接重跑 dump_openapi.py 了事 —— 先看这几条是谁的：")
        L.append("      git log -S '<被删的名字>' -- frontend/openapi.json")
        L.append("  确认无主再放行；有主的话，提交只属于你的那部分。")
    else:
        L.append("  ✓ 零删除。新增的都是你自己的活，可以提交。")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="check_openapi_delta.py",
        description="openapi.json 的结构比对：只关心「有没有东西不见了」")
    ap.add_argument("--base", default="HEAD",
                    help="比对基准：git ref 或文件路径（默认 HEAD）")
    ap.add_argument("--head", default=str(_REPO_ROOT / _DEFAULT_REL),
                    help="被检查的一份：git ref 或文件路径（默认工作树）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非报告")
    args = ap.parse_args(argv)

    try:
        base, head = _load(args.base), _load(args.head)
    except SystemExit:
        raise
    except json.JSONDecodeError as exc:
        print(f"[取数失败] JSON 解析不了：{exc}", file=sys.stderr)
        return 2

    d = compare(base, head)
    if args.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
    else:
        # 默认 head 是绝对路径，Windows 上还是反斜杠 —— 用 Path 比，别比字符串。
        is_worktree = Path(args.head) == (_REPO_ROOT / _DEFAULT_REL)
        print(render(d, args.base, "工作树" if is_worktree else args.head))
    return 1 if removals(d) else 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["compare", "removals", "render", "main"]
