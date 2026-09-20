"""直调技能端点无需经过模型，但仍须经过 ExecutionContext.run 的安全闸门、仪器仲裁、
中止事件、状态回写和地图记录。ok 表示端点调用完成，success 表示技能执行成功。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.api.routes.skill_exec import router  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_module_state():
    """把被替换掉的模块级函数放回去。

    第一版忘了还原,于是 ``wired=False`` 那条测试拿到的是**上一条测试装的**
    假 ExecutionContext —— 一条测试污染另一条。今天已经因为同一个形状
    (scan_policy 的进程级全局)排查过一次 flake,不该在自己写的测试里再来一遍。
    """
    import mast.api.routes.skill_exec as mod

    orig = mod._execution_context
    yield
    mod._execution_context = orig


def _client(*, wired: bool = True, result=None, raises: Exception | None = None):
    api = FastAPI()
    # AppContext 的 skill_registry 是只读 property —— 用一个鸭子类型的替身,
    # 不去硬塞真类(硬塞会让测试依赖 AppContext 的内部实现)。
    ctx = SimpleNamespace(connection_pool=None, state=None, skill_registry=None,
                          live_app=None, app=None)
    calls: list = []
    if wired:
        ctx.connection_pool = SimpleNamespace(name="pool")
        ctx.state = SimpleNamespace(name="state")
        ctx.skill_registry = SimpleNamespace(name="registry")

        class _EC:
            owner = "技能直调 API"

            def run(self, name, params, version=None):
                calls.append((name, params))
                if raises is not None:
                    raise raises
                return result

        import mast.api.routes.skill_exec as mod

        mod._execution_context = lambda _c: (_EC(), [])   # type: ignore[assignment]
    api.state.ctx = ctx
    api.include_router(router, prefix="/api")
    return TestClient(api), calls


def _ok_result():
    return SimpleNamespace(success=True, data={"current_a": 9.6e-11},
                           error="", summary="读到 96 pA",
                           nanonis_calls=[1])


def test_it_runs_the_skill_through_execution_context():
    """必须经 ``ExecutionContext.run(name, params)`` —— 不许另起一条执行路径。"""
    c, calls = _client(result=_ok_result())
    r = c.post("/api/skills/GetCurrent/execute", json={"params": {}})
    assert r.status_code == 200
    assert calls == [("GetCurrent", {})], (
        f"没有走 ExecutionContext.run:{calls}")


def test_params_reach_the_skill():
    c, calls = _client(result=_ok_result())
    c.post("/api/skills/MoveToXY/execute",
           json={"params": {"x_m": 1e-7, "y_m": -2e-7}})
    assert calls[0][1] == {"x_m": 1e-7, "y_m": -2e-7}


def test_a_successful_skill_is_reported_as_such():
    c, _ = _client(result=_ok_result())
    body = c.post("/api/skills/GetCurrent/execute", json={"params": {}}).json()
    assert body["ok"] is True and body["success"] is True
    assert body["data"]["current_a"] == pytest.approx(9.6e-11)
    assert body["nanonis_calls"] == 1


def test_a_failed_skill_is_not_an_endpoint_error():
    """技能失败 ⇒ ``ok=True, success=False``。两件事不许合并。"""
    bad = SimpleNamespace(success=False, data={}, error="前置不满足: z_controller_on",
                          summary="", nanonis_calls=[])
    c, _ = _client(result=bad)
    body = c.post("/api/skills/MoveToXY/execute", json={"params": {}}).json()
    assert body["ok"] is True, "技能失败被报成了端点失败"
    assert body["success"] is False
    assert "z_controller_on" in body["error"]


def test_a_skill_that_raises_does_not_500():
    c, _ = _client(raises=RuntimeError("boom"))
    r = c.post("/api/skills/Whatever/execute", json={"params": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["success"] is False
    assert "RuntimeError" in body["error"]


def test_a_half_wired_process_says_what_is_missing():
    """缺少运行状态与仪器未连接是不同故障，诊断应指向实际缺少的依赖。"""
    c, _ = _client(wired=False)
    body = c.post("/api/skills/GetCurrent/execute", json={"params": {}}).json()
    assert body["ok"] is False and body["degraded"] is True
    assert set(body["missing"]) >= {"connection_pool", "state", "skill_registry"}


def test_an_empty_body_is_allowed():
    """无参技能不该被逼着传一个空 params。"""
    c, calls = _client(result=_ok_result())
    r = c.post("/api/skills/GetCurrent/execute")
    assert r.status_code == 200
    assert calls == [("GetCurrent", {})]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ── 「永不 500」必须对**所有**返回体成立 ────────────────────────────────

def test_bytes_in_the_result_do_not_500():
    """技能返回值含原始字节等非 JSON 类型时，端点应给出可读诊断而非无说明的 HTTP 500。
    异常处理必须同时覆盖技能抛错和结果序列化失败。"""
    res = SimpleNamespace(success=True, data={"trace": ("", b"\x00" * 4096, [1.0])},
                          error="", summary="", nanonis_calls=[1])
    c, _ = _client(result=res)
    r = c.post("/api/skills/GetDualScopeData/execute", json={"params": {}})
    assert r.status_code == 200, f"又 500 了:{r.status_code}"
    body = r.json()
    assert body["ok"] is True and body["success"] is True
    assert "bytes len=4096" in json.dumps(body["data"], ensure_ascii=False)


def test_non_finite_numbers_survive_serialisation():
    """NaN/inf 不是合法 JSON —— 转成一句说明,而不是让整个响应炸掉。"""
    res = SimpleNamespace(success=True, data={"v": float("nan")},
                          error="", summary="", nanonis_calls=[])
    c, _ = _client(result=res)
    r = c.post("/api/skills/X/execute", json={"params": {}})
    assert r.status_code == 200
    assert "非有限值" in json.dumps(r.json()["data"], ensure_ascii=False)


def test_a_huge_array_is_truncated_not_streamed_whole():
    """一条谱有上万点。整条塞进 HTTP 响应对调用方毫无用处,还会撑爆日志。"""
    res = SimpleNamespace(success=True, data={"spectrum": list(range(50000))},
                          error="", summary="", nanonis_calls=[])
    c, _ = _client(result=res)
    body = c.post("/api/skills/X/execute", json={"params": {}}).json()
    sp = body["data"]["spectrum"]
    assert sp["_truncated"] is True and sp["len"] == 50000
    assert len(sp["head"]) == 8 and len(sp["tail"]) == 8


def test_an_unserialisable_object_becomes_its_repr():
    """连 numpy 都不是的怪东西 ⇒ repr,截断。就是不许 500。"""

    class _Weird:
        def __repr__(self):
            return "<weird object>"

    res = SimpleNamespace(success=True, data={"o": _Weird()},
                          error="", summary="", nanonis_calls=[])
    c, _ = _client(result=res)
    r = c.post("/api/skills/X/execute", json={"params": {}})
    assert r.status_code == 200
    assert "weird" in json.dumps(r.json()["data"], ensure_ascii=False)


# ── 模式闸不会从这扇门漏掉（2026-08-27） ──────────────────────────────

def test_the_endpoint_does_not_claim_human_authorisation():
    """这条路建出来的 ``ExecutionContext`` 的 ``approval_source`` 不许是 human。

    SAFE/SEMI 的模式闸（``ExecutionContext.run`` 里那道）与硬闸都写着
    ``!= "human"`` —— 手动 GUI 路径本身就是人工授权，所以豁免。而这个端点
    是**程序化**调用：谁 POST 的、有没有人在看，它并不知道。一旦有人为了
    「让直调更顺手」把它标成 human，SAFE 下的脉冲就会从这里静默漏出去，
    而且没有任何测试会红。

    这里刻意用**真的** ``ExecutionContext``（不是本文件其余用例里的替身）：
    在替身上断言模式闸等于什么都没测。
    """
    import mast.api.routes.skill_exec as mod
    from mast.core.registry import SkillRegistry

    ctx = SimpleNamespace(connection_pool=SimpleNamespace(name="pool"),
                          state=SimpleNamespace(name="state"),
                          skill_registry=SkillRegistry(),
                          live_app=None, app=None)
    ec, missing = mod._execution_context(ctx)
    assert missing == [] and ec is not None
    assert ec._approval_source != "human", (
        "直调端点自称人工授权 —— SAFE 的模式闸和五道硬闸都会对它放行")


def test_the_real_execution_context_carries_the_mode_gate():
    """并且它走的那个 ``run`` 确实带着模式闸。

    上一条只说「没自称 human」；这一条说「不自称 human 是有意义的」——
    两条合起来才是「这扇门被 SAFE 管着」。
    """
    import inspect

    from mast.core.execution_context import ExecutionContext

    src = inspect.getsource(ExecutionContext.run)
    assert "mode_refusal" in src, (
        "ExecutionContext.run 里没有模式闸了 —— 直调 API、conduct、"
        "composite 子步三条路同时失去 SAFE 保护")
