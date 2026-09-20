"""MAST 技能合规检查 —— contrib 投稿、自定义 .py 技能、组合 spec 都用它。

判据本体在 ``MASTv2/mast/skills/compliance.py``；这里只是命令行外壳。

用法（仓库根）::

    python scripts/skill_check.py contrib/skills/EstimateScanDuration   # 一个投稿目录
    python scripts/skill_check.py path/to/MySkill.py                    # 一个技能文件（文件名 = 技能名）
    python scripts/skill_check.py path/to/spec.json                     # 一份组合 spec
    python scripts/skill_check.py --all-contrib                         # contrib/skills/ 下全部
    python scripts/skill_check.py --all-contrib --json                  # 机器可读

退出码：0 = 全部通过；1 = 至少一条 FAIL，或给的路径不存在 / contrib 下一个技能都没有。

``SKIPPED`` 行表示那一项**没有检查**（比如没装 nanonis_spm，命令名存在性就没核对），
它不算通过 —— 装齐依赖再跑一次。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_MASTV2 = str(REPO / "MASTv2")
if _MASTV2 not in sys.path:
    sys.path.insert(0, _MASTV2)

try:
    import nanonis_spm  # noqa: F401 — 装了就用真的；只 import，不连接
except ImportError:
    # 没有仪器、没装厂商库的机器：放一个替身，免得 import mast.core 就失败。
    # 命令名存在性（V02）读的是磁盘上的厂商库源码，不经过这个替身；没装库时它会被标成 SKIPPED。
    from unittest.mock import MagicMock

    _mock = MagicMock()
    sys.modules["nanonis_spm"] = _mock
    sys.modules["nanonis_spm.Nanonis"] = _mock.Nanonis


def _contrib_dirs() -> list[Path]:
    root = REPO / "contrib" / "skills"
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "manifest.json").is_file())


def _check(path: Path, *, smoke: bool):
    from mast.skills.compliance import check_contrib_dir, check_python_source, check_spec

    if path.is_dir():
        if not (path / "manifest.json").is_file():
            return None, f"{path} 不是 contrib 技能目录（没有 manifest.json）"
        return check_contrib_dir(path, run_smoke=smoke), ""
    if path.suffix == ".py":
        return check_python_source(path.read_text(encoding="utf-8"), filename=path.name, run_smoke=smoke), ""
    if path.suffix == ".json":
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, f"{path} 不是合法 JSON：{exc}"
        return check_spec(spec), ""
    return None, f"{path}：只认 contrib 技能目录、.py 技能文件、.json 组合 spec"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="contrib 技能目录 / .py 技能文件 / .json 组合 spec")
    ap.add_argument("--all-contrib", action="store_true", help="检查 contrib/skills/ 下的全部投稿")
    ap.add_argument("--json", action="store_true", help="输出 JSON（每个目标一份报告）")
    ap.add_argument("--no-smoke", action="store_true", help="不做冒烟执行（X01）；缺省对投稿目录与 .py 文件都做")
    ap.add_argument("-v", "--verbose", action="store_true", help="连 INFO 一起打印")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.ERROR, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")

    targets = [Path(p) for p in a.paths]
    if a.all_contrib:
        found = _contrib_dirs()
        if not found:
            print("contrib/skills/ 下一个技能目录都没有 —— --all-contrib 什么也没查", file=sys.stderr)
            return 1
        targets += found
    if not targets:
        ap.print_usage(sys.stderr)
        print("给一个路径，或者 --all-contrib", file=sys.stderr)
        return 1

    reports, failed = [], False
    for t in targets:
        if not t.exists():
            print(f"[FAIL] {t}：路径不存在", file=sys.stderr)
            failed = True
            continue
        rep, err = _check(t, smoke=not a.no_smoke)
        if rep is None:
            print(f"[FAIL] {err}", file=sys.stderr)
            failed = True
            continue
        try:
            rep.target = t.resolve().relative_to(REPO).as_posix()
        except ValueError:
            rep.target = t.as_posix()
        reports.append(rep)
        failed = failed or not rep.ok

    if a.json:
        print(json.dumps([r.to_dict() for r in reports], ensure_ascii=False, indent=1))
    else:
        for r in reports:
            print(r.render(verbose=a.verbose))
        skipped = [s for r in reports for s in r.skipped]
        if skipped:
            print(f"\n{len(skipped)} 项没有检查（SKIPPED 不算通过）：", file=sys.stderr)
            for s in skipped:
                print(f"  SKIPPED  {s}", file=sys.stderr)
        n_ok = sum(r.ok for r in reports)
        print(f"\n{n_ok}/{len(reports)} 通过" + ("" if not failed else "；有失败项"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
