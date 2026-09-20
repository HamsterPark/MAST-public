"""把 FastAPI 的 OpenAPI 文档导出到 ``frontend/openapi.json``。

前端的类型是从这个文件生成的（``frontend/package.json`` 的 ``gen:api`` →
``openapi-typescript ./openapi.json -o ./src/api/schema.d.ts``），但**导出这一步
一直没有脚本**，每次都靠手工拼一段 python。改完 ``api/schemas_*.py`` 忘了重生，
前端拿到的就是旧类型：`tsc` 不报错，运行时字段却是 undefined —— 这类漂移不会
在任何一层被抓住。

用法（改完任何 API schema 之后都要跑）::

    export PYTHONPATH=<repo>/MASTv2
    .venv-v2-py313/Scripts/python.exe MASTv2/scripts/dump_openapi.py
    cd frontend && npm run gen:api && npm run typecheck

不连硬件：``create_app()`` 用一个空的 ``AppContext``，所有 handler 都是未接线
（degraded）状态 —— 而路由与响应模型的**形状**与接线无关，那正是我们要导的东西。

``--check`` 只比对不写盘，退出码非 0 表示 openapi.json 已过期（可挂 CI）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_out() -> Path:
    # <repo>/MASTv2/scripts/dump_openapi.py → <repo>/frontend/openapi.json
    return Path(__file__).resolve().parents[2] / "frontend" / "openapi.json"


def build_spec() -> dict:
    from mast.api.app import create_app
    app = create_app(dev_cors=False)
    return app.openapi()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="输出路径（默认 frontend/openapi.json）")
    ap.add_argument("--check", action="store_true",
                    help="只比对不写盘；不一致时退出码 1")
    args = ap.parse_args(argv)

    out = args.out or _default_out()
    spec = build_spec()
    # 与 FastAPI 的 /openapi.json 一致的紧凑写法 + 尾随换行，让 git diff 只在
    # 内容真的变了时才出现。
    text = json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    n_paths = len(spec.get("paths") or {})
    n_schemas = len((spec.get("components") or {}).get("schemas") or {})

    if args.check:
        old = out.read_text(encoding="utf-8") if out.is_file() else ""
        if old == text:
            print(f"openapi.json 已是最新（{n_paths} 条路径 / {n_schemas} 个 schema）")
            return 0
        print(f"openapi.json 已过期 —— 请跑 python {Path(__file__).name} 并重新 "
              f"npm run gen:api", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"已写出 {out}（{n_paths} 条路径 / {n_schemas} 个 schema）")
    print("下一步：cd frontend && npm run gen:api && npm run typecheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
