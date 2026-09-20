"""分享上行 / 中心索引 / 拉包 —— 三条二期通道的客户端侧。

这三条端点都有**同一个**失败形态：网络那一头不在。所以它们的第一职责不是成功路径，
而是**在配置缺席、服务器拒绝、连接失败时说得出人话**——三种情况下都不能 500、也不能
只回一句「失败」。服务器的拒绝理由要**原样带回来**：`/subscriptions/upload` 那条
「代码不随单走」的红线的报文就在 detail 里，压成一句「发布失败」会让人下次还这么发。

替身用的是**手写的记录器**，不是 MagicMock：`test_double_makes_assertions_vacuous`
—— MagicMock 上任何属性访问都成功，`len()` 是 0、`startswith()` 恒真，负例会恒绿。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from mast.agents.instrument_control.tools import discover_instrument_skills  # noqa: E402
from mast.api.routes import skill_market as SM  # noqa: E402
from mast.webui import builder_api  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return discover_instrument_skills()


@pytest.fixture
def client(subscription_store, registry, monkeypatch):
    from mast.api.app import create_app

    builder_api.set_live_registry(registry)
    builder_api.invalidate_catalog()
    app = create_app()

    class _Ctx:
        skill_registry = registry
        live_app = None

    app.state.ctx = _Ctx()
    yield TestClient(app)
    builder_api.set_live_registry(None)
    builder_api.invalidate_catalog()


class _Resp:
    """一个真的响应对象（不是 MagicMock）—— 属性访问不会凭空成功。"""

    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or ""
        self.headers = {"content-type": "application/json"}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Http:
    """记录器：把每一次请求原样存下来，供断言。"""

    def __init__(self, post=None, get=None):
        self.posts: list[tuple] = []
        self.gets: list[tuple] = []
        self._post = post or _Resp(200, {"ok": True, "id": "sub-abc", "status": "pending_review"})
        self._get = get or _Resp(200, {"subscriptions": []})

    def post(self, url, **kw):
        self.posts.append((url, kw))
        if isinstance(self._post, Exception):
            raise self._post
        return self._post

    def get(self, url, **kw):
        self.gets.append((url, kw))
        if isinstance(self._get, Exception):
            raise self._get
        return self._get


@pytest.fixture
def lab(monkeypatch):
    """假装配好了一台实验室服务器，并把 httpx 换成记录器。"""
    http = _Http()
    monkeypatch.setattr(SM, "_lab_server", lambda: ("https://lab.example", "tok", ""))
    import sys
    monkeypatch.setitem(sys.modules, "httpx", http)
    return http


# ─────────────────────────────────────────────────────────────────────────────
# 没配服务器：说人话，不 500
# ─────────────────────────────────────────────────────────────────────────────

def test_publish_without_a_server_says_what_to_do(client, monkeypatch):
    monkeypatch.setattr(SM, "_lab_server", lambda: ("", "", "未配置实验室服务器（server_url / token）"))
    body = client.post("/api/skill-market/share/publish", json={}).json()
    assert body["ok"] is False
    assert "配置" in body["reason"], "只说「失败」，不说下一步该干什么"


def test_lab_index_without_a_server_is_degraded_not_500(client, monkeypatch):
    monkeypatch.setattr(SM, "_lab_server", lambda: ("", "", "未配置实验室服务器"))
    r = client.get("/api/skill-market/lab-index")
    assert r.status_code == 200
    assert r.json()["degraded"] is True and r.json()["reason"]


def test_real_config_path_does_not_explode(client):
    """不打桩地走一遍真实配置读取 —— 这台机器上大概率没配，那也要 200。"""
    for r in (client.post("/api/skill-market/share/publish", json={}),
              client.get("/api/skill-market/lab-index")):
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 发布
# ─────────────────────────────────────────────────────────────────────────────

def test_publish_sends_the_same_manifest_as_export(client, lab, registry):
    """发布的和导出的必须是**同一份** —— 两处各拼一份迟早不是一个东西。"""
    exported = client.get("/api/skill-market/export").json()
    body = client.post("/api/skill-market/share/publish",
                       json={"label": "qPlus 日常", "note": "给新人"}).json()
    assert body["ok"] is True and body["id"] == "sub-abc"
    assert body["skill_count"] == len(exported["entries"])

    assert len(lab.posts) == 1
    url, kw = lab.posts[0]
    assert url == "https://lab.example/subscriptions/upload"
    assert kw["headers"]["Authorization"] == "Bearer tok"
    sent = kw["json"]["manifest"]
    assert sent["kind"] == exported["kind"]
    assert [e["name"] for e in sent["entries"]] == [e["name"] for e in exported["entries"]]
    assert kw["json"]["label"] == "qPlus 日常"


def test_publish_reflects_the_subscription_not_the_market(client, lab):
    client.post("/api/skill-market/subscription", json={"unsubscribe": ["SetBias"]})
    client.post("/api/skill-market/share/publish", json={})
    sent = lab.posts[0][1]["json"]["manifest"]
    assert "SetBias" not in {e["name"] for e in sent["entries"]}


def test_a_server_refusal_is_quoted_verbatim(client, monkeypatch):
    """红线的报文必须原样带回来 —— 压成「发布失败」会让人下次还这么发。"""
    http = _Http(post=_Resp(400, {"detail": "entry 'X' has unexpected keys ['code'] "
                                            "(code files are NOT accepted)"}))
    monkeypatch.setattr(SM, "_lab_server", lambda: ("https://lab.example", "tok", ""))
    import sys
    monkeypatch.setitem(sys.modules, "httpx", http)

    body = client.post("/api/skill-market/share/publish", json={}).json()
    assert body["ok"] is False
    assert "code files are NOT accepted" in body["reason"]
    assert "400" in body["reason"]


def test_a_network_failure_is_a_reason_not_a_500(client, monkeypatch):
    http = _Http(post=RuntimeError("connection refused"))
    monkeypatch.setattr(SM, "_lab_server", lambda: ("https://lab.example", "tok", ""))
    import sys
    monkeypatch.setitem(sys.modules, "httpx", http)

    r = client.post("/api/skill-market/share/publish", json={})
    assert r.status_code == 200
    assert "connection refused" in r.json()["reason"]


def test_publish_request_forbids_extra_fields(client):
    r = client.post("/api/skill-market/share/publish",
                    json={"label": "x", "manifest": {"kind": "whatever"}})
    assert r.status_code == 422, (
        "发布请求收下了 manifest 字段 —— 那就等于让调用方自己拼一份发出去，"
        "而发布的必须是本机导出的那一份")


# ─────────────────────────────────────────────────────────────────────────────
# 索引与取回
# ─────────────────────────────────────────────────────────────────────────────

def test_lab_index_relays_the_summaries(client, monkeypatch):
    http = _Http(get=_Resp(200, {"subscriptions": [
        {"id": "sub-1", "label": "qPlus", "skill_count": 40, "machine": "rig-2",
         "status": "pending_review", "unknown_field": "ignored"},
    ]}))
    monkeypatch.setattr(SM, "_lab_server", lambda: ("https://lab.example", "tok", ""))
    import sys
    monkeypatch.setitem(sys.modules, "httpx", http)

    rows = client.get("/api/skill-market/lab-index").json()["subscriptions"]
    assert [r["id"] for r in rows] == ["sub-1"]
    assert rows[0]["skill_count"] == 40
    assert "unknown_field" not in rows[0], "服务端多给的字段被原样塞进了响应模型"


def test_lab_fetch_returns_a_manifest_but_does_not_import_it(client, monkeypatch):
    """取回 ≠ 导入。换掉工作面要他自己按一次。"""
    man = {"kind": "mast-skill-subscription", "schema_version": 1,
           "entries": [{"name": "SetBias", "source": "builtin"}]}
    http = _Http(get=_Resp(200, man))
    monkeypatch.setattr(SM, "_lab_server", lambda: ("https://lab.example", "tok", ""))
    import sys
    monkeypatch.setitem(sys.modules, "httpx", http)

    from mast.skills import subscription as sub
    before = sub.is_customised()
    got = client.get("/api/skill-market/lab-fetch/sub-1").json()
    assert got["ok"] is True
    assert [e["name"] for e in got["manifest"]["entries"]] == ["SetBias"]
    assert sub.is_customised() is before, "取回顺手把订阅改了 —— 那导入那一步是摆设"
    assert lab_calls_get_only(http)


@pytest.mark.parametrize("resp,why", [
    (_Resp(404, {"detail": "not found"}), "404"),
    (_Resp(200, {"kind": "conduct", "entries": []}), "订阅列表"),
    (_Resp(200, ["not", "a", "dict"]), "订阅列表"),
])
def test_a_failed_fetch_has_its_own_field_not_a_data_slot(client, monkeypatch, resp, why):
    """失败不折叠进 manifest 的某个数据位。

    第一版把错误塞进了 ``exported_at``，前端靠 ``kind`` 空不空来判断成没成 ——
    那正是 [[read_failure_folded_into_a_value]]：故障被答成了一个值，而且合理得
    没人会去核。
    """
    http = _Http(get=resp)
    monkeypatch.setattr(SM, "_lab_server", lambda: ("https://lab.example", "tok", ""))
    import sys
    monkeypatch.setitem(sys.modules, "httpx", http)

    body = client.get("/api/skill-market/lab-fetch/sub-1").json()
    assert body["ok"] is False
    assert why in body["reason"]
    assert body["manifest"] is None, "失败了却还给出一份 manifest —— 那是一个假答案"


def test_fetch_without_a_server_is_not_a_500(client, monkeypatch):
    monkeypatch.setattr(SM, "_lab_server", lambda: ("", "", "未配置实验室服务器"))
    r = client.get("/api/skill-market/lab-fetch/sub-1")
    assert r.status_code == 200
    assert r.json()["ok"] is False and r.json()["reason"]


def lab_calls_get_only(http) -> bool:
    return http.posts == [] and len(http.gets) == 1
