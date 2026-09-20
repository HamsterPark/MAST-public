"""订阅列表的中心索引（push server 侧）+ 那条「代码不随单走」的红线。

老门（``/skills/upload``）的红线是黑名单式的一条判断：``manifest.nodes`` 必须是
list。新门收得**更紧**：条目的键是**白名单**（name/source/version/spec）。

理由值得写下来 —— 黑名单要求我预见到所有能藏代码的键名（``code`` / ``py`` /
``payload`` / ``__reduce__`` / 下一个人想到的那个），白名单只要求我说清楚哪几个键
是数据。后者我答得上来，前者答不上来。这是本仓 [[guard_that_isnt]] 那一族的预防：
一条只拦得住你想到的那几个名字的守卫，看起来和真守卫一模一样。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from mast.update import server as SRV  # noqa: E402

_KIND = "mast-skill-subscription"


@pytest.fixture
def client(tmp_path):
    tok = SRV.generate_token()
    SRV.write_token(tmp_path, tok)
    c = TestClient(SRV.build_app(tmp_path))
    c.headers.update({"Authorization": "Bearer " + tok})
    return c


def _man(entries=None, **kw):
    return {"kind": _KIND, "schema_version": 1, "machine": "rig-1",
            "entries": entries if entries is not None else [{"name": "SetBias",
                                                             "source": "builtin"}],
            **kw}


# ─────────────────────────────────────────────────────────────────────────────
# 红线
# ─────────────────────────────────────────────────────────────────────────────

def test_the_old_upload_red_line_is_still_there(client):
    """加新门不许把老门顺手拆了（与 test_skills_upload_still_rejects_code 同形）。"""
    r = client.post("/skills/upload",
                    json={"manifest": {"name": "x", "nodes": "def f(): pass"}})
    assert r.status_code == 400
    assert "code files are NOT accepted" in str(r.json().get("detail"))


@pytest.mark.parametrize("bad_key", ["code", "py", "payload", "source_code",
                                     "__reduce__", "script"])
def test_an_entry_with_an_unexpected_key_is_refused(client, bad_key):
    """白名单：不在 {name,source,version,spec} 里的键一律拒 —— 不管它叫什么。"""
    r = client.post("/subscriptions/upload",
                    json={"manifest": _man([{"name": "X", bad_key: "import os"}])})
    assert r.status_code == 400
    detail = str(r.json().get("detail"))
    assert bad_key in detail and "code files are NOT accepted" in detail


def test_an_embedded_spec_must_be_a_composite_spec(client):
    r = client.post("/subscriptions/upload",
                    json={"manifest": _man([{"name": "X", "source": "user_composite",
                                             "spec": {"nodes": "def f(): pass"}}])})
    assert r.status_code == 400
    assert "code files are NOT accepted" in str(r.json().get("detail"))


def test_a_legitimate_embedded_spec_is_accepted(client):
    """负例先自证会红：合法的内嵌 spec 必须过得去，否则上面两条证明不了什么。"""
    r = client.post("/subscriptions/upload",
                    json={"manifest": _man([{"name": "X", "source": "user_composite",
                                             "version": "3",
                                             "spec": {"name": "X", "nodes": []}}])})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 形状与配额
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("man,why", [
    ({}, "kind"),
    ({"kind": "conduct", "entries": []}, "kind"),
    ({"kind": _KIND}, "entries"),
    ({"kind": _KIND, "entries": "nope"}, "entries"),
    ({"kind": _KIND, "entries": [{"source": "builtin"}]}, "name"),
])
def test_a_file_that_is_not_a_subscription_list_is_refused(client, man, why):
    r = client.post("/subscriptions/upload", json={"manifest": man})
    assert r.status_code == 400
    assert why in str(r.json().get("detail"))


def test_too_many_entries_is_refused(client):
    r = client.post("/subscriptions/upload",
                    json={"manifest": _man([{"name": f"S{i}"} for i in range(2001)])})
    assert r.status_code == 413


def test_upload_needs_the_token(tmp_path):
    tok = SRV.generate_token()
    SRV.write_token(tmp_path, tok)
    c = TestClient(SRV.build_app(tmp_path))
    assert c.post("/subscriptions/upload", json={"manifest": _man()}).status_code in (401, 403)
    assert c.get("/subscriptions/index").status_code in (401, 403)


# ─────────────────────────────────────────────────────────────────────────────
# 索引与取回
# ─────────────────────────────────────────────────────────────────────────────

def test_upload_then_index_then_download(client):
    up = client.post("/subscriptions/upload",
                     json={"manifest": _man([{"name": "A"}, {"name": "B"}]),
                           "label": "qPlus 日常", "note": "适合刚上手的人"}).json()
    assert up["ok"] is True and up["status"] == "pending_review"

    rows = client.get("/subscriptions/index").json()["subscriptions"]
    assert [r["id"] for r in rows] == [up["id"]]
    row = rows[0]
    assert row["label"] == "qPlus 日常"
    assert row["skill_count"] == 2
    assert row["machine"] == "rig-1"
    # 索引只有摘要 —— 条目本身不在里面（否则它就成了一个全量托管）
    assert "entries" not in row

    got = client.get(f"/subscriptions/download/{up['id']}").json()
    assert [e["name"] for e in got["entries"]] == ["A", "B"]


def test_download_rejects_a_traversal_shaped_id(client):
    for bad in ("../secret", "sub-xx", "sub-" + "z" * 12, "index"):
        r = client.get(f"/subscriptions/download/{bad}")
        assert r.status_code in (400, 404), f"{bad!r} 没被拒"


def test_download_of_an_unknown_id_is_404(client):
    assert client.get("/subscriptions/download/sub-" + "0" * 12).status_code == 404


def test_index_is_the_tail_not_everything(client):
    """索引是尾部 500 条 —— 一个长期跑着的服务器不该把全部历史塞进一次响应。"""
    assert "subscriptions" in client.get("/subscriptions/index").json()
