"""挂载与契约：私有树里外部面确实挂上了；主 openapi 零改动；v1 契约只增不减。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

_BASELINE = Path(__file__).with_name("contract_v1.json")
_APP = Path(__file__).resolve().parents[4] / "MASTv2" / "mast" / "api" / "app.py"


def _ext_openapi():
    from types import SimpleNamespace

    from mast.api.ext.app import create_ext_app
    from mast.api.ext.jobs import JobManager

    return create_ext_app(SimpleNamespace(), job_manager=JobManager("unused")).openapi()


def test_the_private_tree_mounts_the_gateway():
    """受保护的 import 不许变成「功能静默消失」—— 私有树里它必须真的挂上。"""
    from starlette.routing import Mount

    from mast.api.app import create_app

    app = create_app(dev_cors=False)
    mounts = [r.path for r in app.routes if isinstance(r, Mount)]
    assert "/api/ext/v1" in mounts, mounts
    # 且在 SPA 兜底路由之前（之后注册的永远匹配不到）
    paths = [getattr(r, "path", "") for r in app.routes]
    if "/{full_path:path}" in paths:
        assert paths.index("/api/ext/v1") < paths.index("/{full_path:path}")


def test_every_ext_get_degrades_on_a_cold_unwired_app():
    """``tests/v2/unit/api/test_boot_smoke.py`` 只枚举主 app 自己的 ``APIRoute``，挂载的
    子应用不在它的网里。这里照它的做法走一遍外部面：真 ``create_app``、没有仪器、没有实验，
    每个 GET 都只许降级（JSON、非 5xx），不许 500。"""
    from fastapi.routing import APIRoute
    from fastapi.testclient import TestClient
    from starlette.routing import Mount

    from mast.api.app import create_app

    app = create_app(dev_cors=False)
    mount = next(r for r in app.routes if isinstance(r, Mount) and r.path == "/api/ext/v1")
    samples = {"job_id": "j_000000000000", "name": "GetBias", "request_id": "r-nope"}
    query = {"/skills/search": {"q": "bias"}, "/data/file": {"path": "C:/nope/x.sxm"},
             "/data/frame": {"path": "C:/nope/x.sxm"}}
    client = TestClient(app)
    walked, failures = [], []
    for route in mount.routes:
        if not isinstance(route, APIRoute) or "GET" not in route.methods:
            continue
        path = route.path
        for key, value in samples.items():
            path = path.replace("{" + key + "}", value)
        assert "{" not in path, f"{route.path} 的路径参数没有样例 —— 在 samples 里补一个"
        try:
            r = client.get("/api/ext/v1" + path, params=query.get(route.path))
        except Exception as exc:  # noqa: BLE001 — 抛出来本身就是缺陷
            failures.append(f"{route.path} raised {type(exc).__name__}: {exc}")
            continue
        walked.append(route.path)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        # 契约里的「没接线」答复：503 not_wired 且点名缺了什么 —— 是降级，不是崩溃
        declared = (r.status_code == 503 and body.get("error") == "not_wired"
                    and bool(body.get("missing")))
        if r.status_code >= 500 and not declared:
            failures.append(f"{route.path} → {r.status_code}: {r.text[:200]}")
        elif not r.headers.get("content-type", "").startswith("application/json"):
            failures.append(f"{route.path} → 不是 JSON（{r.headers.get('content-type')}）")
    assert len(walked) >= 12, f"只走到 {len(walked)} 个 GET —— 挂载的子应用路由没枚举出来？"
    assert not failures, "冷启动时外部面的 GET 出了服务端错误：\n  " + "\n  ".join(failures)


def test_the_main_openapi_is_untouched():
    from mast.api.app import create_app

    main = create_app(dev_cors=False).openapi()
    assert not [p for p in main["paths"] if p.startswith("/api/ext")]


def test_the_guarded_import_only_swallows_its_own_absence():
    """``except ModuleNotFoundError`` 必须判 ``exc.name`` 属于 ``mast.api.ext`` 再吞；
    别的缺失（网关自己依赖的某个包没了）要重抛 —— 否则一个真 bug 会被说成「这个构建没带网关」。"""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    handlers = [h for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler)
                and isinstance(h.type, ast.Name) and h.type.id == "ModuleNotFoundError"]
    assert handlers, "app.py 里没找到挂载网关的受保护 import"
    src = ast.unparse(handlers[0])
    assert "mast.api.ext" in src and "raise" in src


def test_the_v1_contract_only_grows():
    """``contract_v1.json`` 是 v1 的基线：里面的 (方法, 路径) 一个都不能少。
    新增端点请同时追加进基线 —— 基线文件本身只许增长。"""
    spec = _ext_openapi()
    have = {(m.upper(), p) for p, ops in spec["paths"].items() for m in ops}
    base = {tuple(x) for x in json.loads(_BASELINE.read_text(encoding="utf-8"))["endpoints"]}
    assert len(base) >= 20, "基线空转"
    missing = sorted(base - have)
    assert not missing, f"v1 契约里的端点不见了：{missing}"
    new = sorted(have - base)
    assert not new, f"新端点还没写进基线 contract_v1.json：{new}"


def _delta_checker():
    import importlib.util

    path = Path(__file__).resolve().parents[4] / "MASTv2" / "scripts" / "check_openapi_delta.py"
    spec = importlib.util.spec_from_file_location("check_openapi_delta", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_openapi(endpoints, schemas) -> dict:
    paths: dict = {}
    for method, path in endpoints:
        paths.setdefault(path, {})[method.lower()] = {}
    return {"paths": paths, "components": {"schemas": {
        n: {"properties": {f: {} for f in fields}} for n, fields in schemas.items()}}}


def test_no_request_body_field_disappears_within_v1():
    """请求体字段同样只增不减（``JobSubmit.request_id`` 没了，重发就不再幂等）。判据复用
    ``MASTv2/scripts/check_openapi_delta.py`` 的 ``compare`` / ``removals``，不另写一份。"""
    C = _delta_checker()
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert sum(len(f) for f in baseline["schemas"].values()) >= 25, "基线里的字段空转"
    live = _ext_openapi()
    head_schemas = {n: sorted((s.get("properties") or {}).keys())
                    for n, s in live["components"]["schemas"].items()}
    head_endpoints = [(m.upper(), p) for p, ops in live["paths"].items() for m in ops]
    gone = C.removals(C.compare(_as_openapi(baseline["endpoints"], baseline["schemas"]),
                                _as_openapi(head_endpoints, head_schemas)))
    assert not gone, "v1 契约里的东西不见了：\n  " + "\n  ".join(gone)
    new = sorted(f"{n}.{f}" for n, fields in head_schemas.items() if n in baseline["schemas"]
                 for f in fields if f not in baseline["schemas"][n])
    assert not new, f"新字段还没写进基线 contract_v1.json：{new}"


def test_the_contract_check_sees_a_removed_field():
    """守卫的守卫：拿基线自己删掉一个字段，判据必须报出来。"""
    C = _delta_checker()
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    cut = {n: [f for f in fields if not (n == "JobSubmit" and f == "request_id")]
           for n, fields in baseline["schemas"].items()}
    gone = C.removals(C.compare(_as_openapi(baseline["endpoints"], baseline["schemas"]),
                                _as_openapi(baseline["endpoints"], cut)))
    assert gone == ["JobSubmit.request_id 字段被删除"], gone


def test_the_bootstrap_wires_the_cognition_context():
    src = (Path(__file__).resolve().parents[4] / "MASTv2" / "mast" / "api" / "bootstrap.py"
           ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    assigned = {t.attr for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name) and t.value.id == "ctx"}
    assert "cognition" in assigned, "外部面读 ctx.cognition，bootstrap 必须挂上它"
