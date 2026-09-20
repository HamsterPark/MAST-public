"""``POST /api/skill-overlay/packs/fetch`` —— 「从服务器装一个签名包」的 HTTP 入口。

这条路的后端（``skillpack_client.fetch_and_install``）2026-08-20 就写好了，**却一直
零调用方**：界面上只看得到已装的包，装新包得有人到机器前面跑脚本。这是本仓
[[producer_wired_consumer_absent]] 的一个标本 —— 「已经能做了」是生产方的话。

这份测试盯三件事：

1. **信任链一行都不在这里重实现。** 端点只负责取配置、调 ``fetch_and_install``、
   把结果翻译成响应。验签 / 逐文件 sha256 / fail-closed 全在下面那层。这里用一条
   结构断言钉住它没在路由里长出第二份。
2. **装 ≠ 生效。** 默认不重载，``needs_reload`` 说清楚还差一步；带 ``reload_now``
   才重载，那时响应带三态生效字段。
3. **被遮盖的条目必须列出来。** 本机松散文件胜过推来的包是**有意的**，但不说的话
   它长得像「装了没生效」。

替身是手写记录器，不是 MagicMock（负例要能红）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from mast.api.routes import skill_overlay as SO  # noqa: E402
from mast.update.skillpack_client import InstallResult  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    from mast.api.app import create_app

    return TestClient(create_app())


class _Fetcher:
    """记录调用参数并回一个真的 InstallResult。"""

    def __init__(self, result=None, boom=None):
        self.calls: list[tuple] = []
        self._result = result
        self._boom = boom

    def fetch_and_install(self, url, token, pack_id, **kw):
        self.calls.append((url, token, pack_id, kw))
        if self._boom:
            raise self._boom
        return self._result or InstallResult(
            ok=True, pack_id=pack_id, version="1.2.0",
            installed=["builtins/bias.py"], enabled=["builtins/bias.py"],
            needs_reload=True)


class _Runtime:
    def __init__(self, queued=False, boom=False):
        self.reloads = 0
        self._queued = queued
        self._boom = boom
        self._registry = None

    def reload_overlay_skills(self, *, reason=""):
        self.reloads += 1
        if self._boom:
            raise RuntimeError("wedged")
        return {"summary": "已重载 1 个覆盖模块",
                "refresh": {"queued": self._queued, "agent_path_pending": self._queued}}


def _wire(client, monkeypatch, fetcher, *, url="https://push.example", token="tok",
          runtime=None):
    monkeypatch.setattr("mast.update.client.read_server_url", lambda root: url)
    monkeypatch.setattr("mast.update.client._read_token", lambda root: token)
    monkeypatch.setattr("mast.update.skillpack_client.fetch_and_install",
                        fetcher.fetch_and_install)
    if runtime is not None:
        class _Ctx:
            live_app = runtime
            skill_registry = None
        client.app.state.ctx = _Ctx()
    return fetcher


# ─────────────────────────────────────────────────────────────────────────────
# 接线：这条路终于有入口了
# ─────────────────────────────────────────────────────────────────────────────

def test_the_endpoint_is_registered(client):
    paths = {r.path for r in client.app.routes if hasattr(r, "path")}
    assert "/api/skill-overlay/packs/fetch" in paths, (
        "fetch_and_install 又一次没有调用方了")


def test_it_actually_calls_fetch_and_install(client, monkeypatch):
    f = _wire(client, monkeypatch, _Fetcher())
    body = client.post("/api/skill-overlay/packs/fetch",
                       json={"pack_id": "tip-tools"}).json()
    assert f.calls, "端点存在，但它根本没调那条后端 —— 这正是要修的病"
    url, token, pack_id, _kw = f.calls[0]
    assert (url, token, pack_id) == ("https://push.example", "tok", "tip-tools")
    assert body["ok"] is True and body["version"] == "1.2.0"


def _code_only(fn) -> str:
    """函数的**代码**，剥掉 docstring 与 ``#`` 注释。

    第一版没剥，闸门当场判红了自己的 docstring —— 那句话正是在**解释为什么不该
    在这里验签**。同一形状本仓已经栽过两次（``mutation_must_prove_it_landed`` 第 2
    条），修法从来不是把解释删掉。
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    fn_node = tree.body[0]
    body = fn_node.body[1:] if ast.get_docstring(fn_node) else fn_node.body
    return "\n".join(ast.unparse(n) for n in body)


def test_the_route_does_not_reimplement_the_trust_chain(client):
    """信任链只该有一份实现。路由**代码**里出现验签/哈希字样 = 有人开始抄第二份。"""
    src = _code_only(SO.overlay_pack_fetch)
    for smell in ("verify_pack", "sha256", "Ed25519", "signature",
                  "get_release_public_key"):
        assert smell not in src, (
            f"路由里出现了 {smell!r} —— 验签与哈希是 skillpack 那一层的事，"
            "两处各写一遍迟早只有一处是对的")
    # 自检：剥完注释之后扫描器仍然看得见真正的调用，否则上面那圈断言是空的
    assert "fetch_and_install" in src, "剥注释剥过头了 —— 代码本身也没了"


# ─────────────────────────────────────────────────────────────────────────────
# 装 ≠ 生效
# ─────────────────────────────────────────────────────────────────────────────

def test_install_alone_does_not_reload(client, monkeypatch):
    rt = _Runtime()
    _wire(client, monkeypatch, _Fetcher(), runtime=rt)
    body = client.post("/api/skill-overlay/packs/fetch",
                       json={"pack_id": "tip-tools"}).json()
    assert rt.reloads == 0, "默认就重载了 —— 装是一次编辑，生效是一次决定"
    assert body["needs_reload"] is True
    assert body["reloaded"] is False
    assert "尚未生效" in body["summary"] or body["needs_reload"]


def test_reload_now_reloads_and_reports_three_state(client, monkeypatch):
    rt = _Runtime()
    _wire(client, monkeypatch, _Fetcher(), runtime=rt)
    body = client.post("/api/skill-overlay/packs/fetch",
                       json={"pack_id": "tip-tools", "reload_now": True}).json()
    assert rt.reloads == 1
    assert body["reloaded"] is True and body["needs_reload"] is False
    assert body["agent_path_pending"] is False
    assert "fingerprint_matches" in body


def test_queued_reload_is_reported_as_pending(client, monkeypatch):
    rt = _Runtime(queued=True)
    _wire(client, monkeypatch, _Fetcher(), runtime=rt)
    body = client.post("/api/skill-overlay/packs/fetch",
                       json={"pack_id": "tip-tools", "reload_now": True}).json()
    assert body["agent_path_pending"] is True, "排队了却报没在等 —— 替未来打包票"


def test_reload_failure_does_not_lose_the_install(client, monkeypatch):
    rt = _Runtime(boom=True)
    _wire(client, monkeypatch, _Fetcher(), runtime=rt)
    body = client.post("/api/skill-overlay/packs/fetch",
                       json={"pack_id": "tip-tools", "reload_now": True}).json()
    assert body["ok"] is True, "重载失败把「包已经装上了」这件事也否掉了"
    assert body["agent_path_pending"] is True
    assert "重载失败" in body["summary"]


def test_headless_says_restart(client, monkeypatch):
    _wire(client, monkeypatch, _Fetcher())      # 没有 live_app
    body = client.post("/api/skill-overlay/packs/fetch",
                       json={"pack_id": "tip-tools", "reload_now": True}).json()
    assert body["ok"] is True
    assert "重启" in body["summary"]
    assert body["agent_path_pending"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 诚实与拒绝
# ─────────────────────────────────────────────────────────────────────────────

def test_shadowed_entries_are_listed(client, monkeypatch):
    """本机文件胜过推来的包是有意的 —— 但不说的话它长得像「装了没生效」。"""
    _wire(client, monkeypatch, _Fetcher(InstallResult(
        ok=True, pack_id="p", version="1", installed=["builtins/bias.py"],
        shadowed=["builtins/bias.py"], needs_reload=True)))
    body = client.post("/api/skill-overlay/packs/fetch", json={"pack_id": "p"}).json()
    assert body["shadowed"] == ["builtins/bias.py"]
    assert "遮盖" in body["summary"]


def test_a_refused_pack_carries_its_reasons(client, monkeypatch):
    """fail-closed 的理由要带出来（没有公钥 / 验签不过 / 夹带文件）。"""
    _wire(client, monkeypatch, _Fetcher(InstallResult(
        ok=False, pack_id="p", reasons=["没有烘焙公钥，拒绝安装任何包"])))
    body = client.post("/api/skill-overlay/packs/fetch", json={"pack_id": "p"}).json()
    assert body["ok"] is False
    assert "公钥" in body["reason"]
    assert body["needs_reload"] is False


def test_an_empty_pack_id_is_refused_before_any_network(client, monkeypatch):
    f = _wire(client, monkeypatch, _Fetcher())
    body = client.post("/api/skill-overlay/packs/fetch", json={"pack_id": "  "}).json()
    assert body["ok"] is False and f.calls == []


def test_no_server_configured_says_what_to_do(client, monkeypatch):
    f = _wire(client, monkeypatch, _Fetcher(), url="", token="")
    body = client.post("/api/skill-overlay/packs/fetch", json={"pack_id": "p"}).json()
    assert body["ok"] is False
    assert "配置" in body["reason"]
    assert f.calls == [], "没有配置却还是发了一次网络请求"


def test_a_throwing_backend_is_not_a_500(client, monkeypatch):
    _wire(client, monkeypatch, _Fetcher(boom=RuntimeError("boom")))
    r = client.post("/api/skill-overlay/packs/fetch", json={"pack_id": "p"})
    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_extra_fields_are_refused(client):
    r = client.post("/api/skill-overlay/packs/fetch",
                    json={"pack_id": "p", "public_key_hex": "deadbeef"})
    assert r.status_code == 422, (
        "请求收下了 public_key_hex —— 让调用方指定信任根，等于让它自己发一个包给自己")


def test_the_pin_gate_applies(client, monkeypatch, tmp_path):
    from mast.api.admin_pin import set_pin

    ok, why = set_pin("1234")
    assert ok, why
    f = _wire(client, monkeypatch, _Fetcher())
    body = client.post("/api/skill-overlay/packs/fetch",
                       json={"pack_id": "p", "pin": "wrong"}).json()
    assert body["ok"] is False and f.calls == []

    body = client.post("/api/skill-overlay/packs/fetch",
                       json={"pack_id": "p", "pin": "1234"}).json()
    assert body["ok"] is True and len(f.calls) == 1
