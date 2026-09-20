"""Boot the REAL app and touch every route — nothing may 500, nothing may hide.

The per-route test files each build a tiny FastAPI with one router mounted. That
is the right unit for behaviour, and it is blind to the two failures that only
exist in the ASSEMBLED app:

  1. **A route that 500s on a cold, hardware-less machine.** Every read endpoint
     is contractually degrade-safe (return ``degraded: true``, never a traceback)
     — but "every" is only true if something actually walks the whole list. The
     operator's first five minutes with MAST are exactly this state: no
     instrument, no experiment, empty directories.

  2. **A route SHADOWED by an earlier parameterised one.** FastAPI matches in
     registration order, so ``GET /agents/{agent_id}/x`` registered before
     ``GET /agents/group-transcript`` swallows the literal path — and the handler
     that *is* reached happily returns an empty, 200-OK, non-degraded body. That
     is a bug with no error anywhere: the 群聊记录 page just showed nothing, and
     looked healthy doing it (2026-06-25). A unit test of the group-transcript
     router alone passes, because in that app there is nothing to shadow it.

So: assemble the real app, enumerate its real routes, and check both.
"""
from __future__ import annotations

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from mast.api.app import create_app


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(scope="module")
def client(app) -> TestClient:
    return TestClient(app)


def _api_routes(app) -> list[APIRoute]:
    return [r for r in app.routes if isinstance(r, APIRoute)]


def _has_params(path: str) -> bool:
    return "{" in path


# Plausible values for the path params that exist, so a parameterised route is
# exercised with something shaped like real input rather than a bare "1". Every
# one of these is expected to MISS — the point is that a miss degrades, never 500s.
_PARAM_SAMPLES = {
    "agent_id": "instrument_control",
    # /api/prompts/manifest/{agent} 与 /api/prompts/capture/latest/{agent}
    # （2026-08-24）。给一个**真实存在**的 agent：这条测试要的是「后端不在时
    # 优雅降级」，用一个不存在的名字会走 404 分支，等于没测到降级那条路。
    "agent": "instrument_control",
    "doc_id": "draft:nope_v001",
    "skill_name": "GetBias",
    "skill": "GetBias",
    "name": "GetBias",
    "experiment_id": "1",
    "sample_id": "1",
    "conversation_id": "nope",
    "event_id": "nope",
    "run_id": "nope",
    "task_id": "nope",
    "action_id": "1",
    "artifact_id": "draft",
    "key": "nope",
    "path": "nope",
    "lib_id": "nope",
    "library_id": "nope",
    "work_id": "W123456789",
    "session_id": "nope",
    # 一份从来不存在的 conduct(引擎多半还没开)。2026-08-20 补:改名提交
    # c9b6dc16 把路径参数从 ``campaign_id`` 改成了 ``conduct_id``,而这张表没跟 ——
    # 于是这条测试自己红了,**顺带把 conduct 面板端点整个漏出了检查范围**。
    # 一个漏检的端点和一个没有端点长得一样。旧键留着是给可能存在的第三方路径用的,
    # 今天 ``mast/api`` 里已经没有 ``{campaign_id}``。
    "conduct_id": "nope",
    "campaign_id": "nope",
    # /api/skill-market/lab-fetch/{sub_id}（2026-08-26）—— 中心索引里的一份订阅
    # 列表。给一个**形状合法但不存在**的 id：这条测试要的是「实验室服务器没配
    # /连不上时优雅降级」，而不是 id 格式校验那条早退路径。
    "sub_id": "sub-000000000000",
    "idea_id": "1",
    "trajectory_id": "1",
    "alert_id": "999999",              # current-monitor alert that never existed
    "seg_id": "999999",                # ...and a segment id past the end
    "spectrum_id": "999999",           # ...and an archived spectrum that is not there
    "baseline_id": "999999",           # ...and a noise baseline that was never measured
    "point_id": "999999",              # ...and a working point inside it
    "seqno": "999999",                 # ...and a buffer event past the end (#93 放大)
    "seq": "999999",                   # ...and a 旁白 row past the end of a transcript
    "ns": "nope",
    "prompt_id": "nope.not.a.prompt",
    "index": "999",                    # capture ring index past the end
    # /api/gallery/marks/export/{fmt}（数据图库，2026-09-13）。给一个**真实存在**的
    # 格式，理由同上面的 ``agent``：假值会被 Literal 校验在 422 早退，测不到
    # 「一条标记都还没存过时，现场生成导出、且不写盘」的那条降级路径。
    "fmt": "md",
    # enum-ish selectors: a bogus value must degrade (or 4xx), never 500
    "kind": "nope",
    "ktype": "nope",
    "view": "nope",
    "section": "nope",
    "category": "nope",
}

# The SPA catch-all serves index.html for any unmatched path — it is not an API
# route and matching it here would only test starlette's StaticFiles.
_NOT_AN_API_ROUTE = {"/{full_path:path}"}


def _fill(path: str) -> "str | None":
    out = path
    while "{" in out:
        start = out.index("{")
        end = out.index("}", start)
        raw = out[start + 1:end]
        name = raw.split(":")[0]           # "{doc_id:path}" → doc_id
        if name not in _PARAM_SAMPLES:
            return None                    # unknown param — reported, not guessed
        out = out[:start] + _PARAM_SAMPLES[name] + out[end + 1:]
    return out


# ════════════════════════════════════════════════════════════════════════
# The app assembles at all
# ════════════════════════════════════════════════════════════════════════

def test_the_app_boots_with_no_hardware_and_no_experiment(app):
    routes = _api_routes(app)
    assert len(routes) > 60, f"only {len(routes)} routes — did a router fail to mount?"


def test_openapi_schema_generates(app):
    """A response_model that cannot be serialised blows up here, not in the
    frontend's `npm run gen:api`."""
    schema = app.openapi()
    assert schema["paths"], "the OpenAPI spec is empty"
    assert schema["info"]["title"] == "MAST API"


# ════════════════════════════════════════════════════════════════════════
# 1. Nothing 500s on a cold machine
# ════════════════════════════════════════════════════════════════════════

def _gets(app) -> list[str]:
    return sorted({r.path for r in _api_routes(app)
                   if "GET" in r.methods and r.path not in _NOT_AN_API_ROUTE})


def test_every_parameterless_get_returns_without_a_server_error(client, app):
    """The cold-start walk: no instrument, no DB, empty dirs."""
    failures = []
    for path in _gets(app):
        if _has_params(path):
            continue
        try:
            r = client.get(path)
        except Exception as exc:  # noqa: BLE001 — an unhandled raise IS the bug
            failures.append((path, f"raised {type(exc).__name__}: {exc}"))
            continue
        if r.status_code >= 500:
            failures.append((path, f"{r.status_code}: {r.text[:160]}"))
    assert not failures, "GET routes that 500 with no hardware attached:\n" + "\n".join(
        f"  {p} → {why}" for p, why in failures)


def test_every_parameterised_get_degrades_on_a_miss(client, app):
    """A not-found id must come back as a degraded/empty body, not a traceback —
    the operator types a stale link and the page must still render."""
    failures, unknown = [], []
    for path in _gets(app):
        if not _has_params(path):
            continue
        filled = _fill(path)
        if filled is None:
            unknown.append(path)
            continue
        try:
            r = client.get(filled)
        except Exception as exc:  # noqa: BLE001
            failures.append((filled, f"raised {type(exc).__name__}: {exc}"))
            continue
        if r.status_code >= 500:
            failures.append((filled, f"{r.status_code}: {r.text[:160]}"))
    assert not unknown, (
        "these routes have path params this test has no sample for — add one to "
        f"_PARAM_SAMPLES so they are actually exercised: {unknown}")
    assert not failures, "GET routes that 500 on a missing id:\n" + "\n".join(
        f"  {p} → {why}" for p, why in failures)


# ════════════════════════════════════════════════════════════════════════
# 2. No route is shadowed — the silent-empty-page bug
# ════════════════════════════════════════════════════════════════════════

def test_no_literal_route_is_shadowed_by_an_earlier_parameterised_one(app):
    """``GET /agents/group-transcript`` under ``GET /agents/{agent_id}/…`` reaches
    the WRONG handler, which returns 200 + an empty list. Nothing errors; the page
    is simply, permanently blank. Registration order is the whole contract, so
    check it directly: for every literal path, no earlier route may match it.
    """
    routes = _api_routes(app)
    shadowed = []
    for i, route in enumerate(routes):
        if _has_params(route.path):
            continue
        for earlier in routes[:i]:
            if not _has_params(earlier.path):
                continue
            if not (route.methods & earlier.methods):
                continue
            if earlier.path_regex.match(route.path):
                shadowed.append((sorted(route.methods)[0], route.path, earlier.path))
    assert not shadowed, (
        "these literal routes are swallowed by an earlier parameterised route — "
        "they will silently return the WRONG handler's (usually empty) body:\n"
        + "\n".join(f"  {m} {p}  ←shadowed by→  {by}" for m, p, by in shadowed))


def test_the_group_chat_routes_resolve_to_their_own_handlers(client, app):
    """The specific regression, pinned by NAME as well as by the general rule
    above: these two were shadowed by /agents/{agent_id}/… and the 群聊记录 page
    showed an empty list while looking perfectly healthy."""
    for path in ("/api/agents/group-conversations", "/api/agents/group-transcript"):
        matches = [r for r in _api_routes(app) if r.path == path]
        assert matches, f"{path} is not registered at all"
        r = client.get(path)
        assert r.status_code < 500, f"{path} → {r.status_code}"
        # the tell-tale of a shadowed hit: the *other* handler's shape comes back
        body = r.json()
        assert isinstance(body, dict), f"{path} returned {type(body).__name__}"


def test_no_two_routes_share_a_path_and_method(app):
    """A duplicate registration means the second one is dead code — and whichever
    of the two you were reading when you wrote the test is a coin flip."""
    seen: dict[tuple[str, str], int] = {}
    for r in _api_routes(app):
        for m in r.methods:
            seen[(m, r.path)] = seen.get((m, r.path), 0) + 1
    dupes = [k for k, n in seen.items() if n > 1]
    assert not dupes, f"duplicate route registrations (the later one is dead): {dupes}"
