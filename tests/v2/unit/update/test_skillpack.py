"""签名技能包 —— 把改好的 skill 推给别的机器，不发版本、不重启。

这份测试盯的是**四个各自独立的信任边界**，每一个失守都足以让整套东西变成
「看起来在验签」：

* **传输**：改一个字节、改 manifest、拿掉签名、夹带一个没登记的文件 —— 全拒。
* **落盘**：任何一步失败都不留半个文件。半个包意味着一部分技能是新的、一部分
  是旧的，而界面上一切正常。
* **加载**：下载时验过一次**不够** —— 那次和现在之间隔着一段任何人都能写的时间。
  覆盖层会把 ``_packs/`` 下的东西标成 ``verified:``，那个标记必须有凭据。
* **老门**：``/skills/upload`` 那条「永不接收代码」的红线不能被顺手拆掉。

所有测试都自己造一把密钥并把 ``get_release_public_key`` 指过去 —— 不依赖构建里
烘焙的那把（dev 树里它可能是任何东西，而一条「本机恰好没配公钥所以全跳过」的
测试比没有测试更坏）。
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mast.update import signing
from mast.update import skillpack as SP
from mast.update import skillpack_client as SC
from mast.update.skillpack import K, PackFile, PackManifest

pytestmark = pytest.mark.skipif(not signing.HAVE_CRYPTO,
                                reason="没装 cryptography")

SKILL_SRC = (
    "from mast.skills.base import BaseSkill\n"
    "class PackProbeSkill(BaseSkill):\n"
    "    pass\n"
)


@pytest.fixture
def keys(monkeypatch):
    """一把只在本测试里存在的密钥，并让它成为「出厂烘焙」的那把。"""
    priv, pub = signing.generate_keypair()
    monkeypatch.setattr(signing, "get_release_public_key", lambda: pub)
    return priv, pub


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """覆盖层目录指到 tmp —— 测试写进用户真实数据目录，本仓被咬过五次。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    yield


def make_pack(tmp_path: Path, priv: str, *, pack_id="probe", version="1.0.0",
              files: dict[str, bytes] | None = None,
              min_app="", max_app="", sign=True) -> Path:
    """造一个包。**不走 make_skillpack.py** —— 这里要能造出畸形的包。"""
    files = files or {"builtins/pack_probe.py": SKILL_SRC.encode()}
    man = PackManifest(
        pack_id=pack_id, version=version,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        author="test",
        min_app_version=min_app, max_app_version=max_app,
        files=[PackFile(path=r, sha256=SP.sha256_bytes(b), size=len(b))
               for r, b in files.items()])
    out = tmp_path / SP.pack_filename(pack_id, version)
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(SP.MANIFEST_NAME, json.dumps(man.to_dict(), ensure_ascii=False))
        if sign:
            z.writestr(SP.SIG_NAME, json.dumps(
                {K.SIGNATURE: signing.sign_manifest(man.to_dict(), priv)}))
        for r, b in files.items():
            z.writestr(SP.FILES_PREFIX + r, b)
    return out


def mutate(src: Path, dst: Path, fn) -> Path:
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
        fn(zi, zo)
    return dst


# ---------------------------------------------------------------------------
# 传输边界
# ---------------------------------------------------------------------------

def test_a_well_formed_pack_verifies(tmp_path, keys):
    priv, pub = keys
    res = SP.verify_pack(make_pack(tmp_path, priv), public_key_hex=pub)
    assert res.ok, res.reasons
    assert res.pubkey8 == pub[:8]


def test_unsigned_pack_is_rejected(tmp_path, keys):
    """未签名的包不装。**没有「本机没配公钥所以放行」这条路。**"""
    priv, pub = keys
    res = SP.verify_pack(make_pack(tmp_path, priv, sign=False), public_key_hex=pub)
    assert not res.ok
    assert any("未签名" in r or "manifest.sig" in r for r in res.reasons)


def test_verification_fails_closed_without_a_public_key(tmp_path, keys, monkeypatch):
    """构建里没有烘焙公钥时，**一个包都不装**。

    「验不了」不能当成「通过」—— 那正好在这道门最需要工作的时候把它变成摆设。
    """
    priv, _pub = keys
    monkeypatch.setattr(signing, "get_release_public_key", lambda: "")
    res = SP.verify_pack(make_pack(tmp_path, priv))
    assert not res.ok
    assert any("公钥" in r for r in res.reasons)


def test_tampered_file_is_rejected(tmp_path, keys):
    """改一个字节 → 整包拒。签名保护 manifest，manifest 保护文件。"""
    priv, pub = keys
    src = make_pack(tmp_path, priv)
    bad = mutate(src, tmp_path / "bad.zip", lambda zi, zo: [
        zo.writestr(i, zi.read(i.filename)
                    + (b"\n# x\n" if i.filename.startswith(SP.FILES_PREFIX) else b""))
        for i in zi.infolist()])
    res = SP.verify_pack(bad, public_key_hex=pub)
    assert not res.ok
    assert any("对不上" in r for r in res.reasons)


def test_tampered_manifest_is_rejected(tmp_path, keys):
    priv, pub = keys
    src = make_pack(tmp_path, priv)

    def _m(zi, zo):
        for i in zi.infolist():
            d = zi.read(i.filename)
            if i.filename == SP.MANIFEST_NAME:
                j = json.loads(d)
                j[K.PACK_ID] = "evil"
                d = json.dumps(j).encode()
            zo.writestr(i, d)

    res = SP.verify_pack(mutate(src, tmp_path / "bad.zip", _m), public_key_hex=pub)
    assert not res.ok
    assert any("签名验证不通过" in r for r in res.reasons)


def test_a_stowaway_file_is_rejected(tmp_path, keys):
    """包里夹带 manifest 没登记的文件 → 整包拒。

    签名保护的是 manifest；登记之外的文件不受任何保护。放过它们，「已签名」这个
    标记就只对包里的一部分成立 —— 而 UI 上是整包一个徽章。
    """
    priv, pub = keys
    src = make_pack(tmp_path, priv)

    def _m(zi, zo):
        for i in zi.infolist():
            zo.writestr(i, zi.read(i.filename))
        zo.writestr(SP.FILES_PREFIX + "builtins/stowaway.py", b"import os\n")

    res = SP.verify_pack(mutate(src, tmp_path / "bad.zip", _m), public_key_hex=pub)
    assert not res.ok
    assert any("夹带" in r for r in res.reasons)


def test_path_traversal_in_the_manifest_is_rejected(tmp_path, keys):
    """``../`` 会让文件落在覆盖层目录之外。签名是**签名者**说了算，不是路径。"""
    priv, pub = keys
    src = make_pack(tmp_path, priv,
                    files={"../../evil.py": b"import os\n"})
    res = SP.verify_pack(src, public_key_hex=pub)
    assert not res.ok
    assert any("路径不合法" in r for r in res.reasons)


def test_min_app_version_blocks_an_older_client(tmp_path, keys):
    priv, pub = keys
    src = make_pack(tmp_path, priv, min_app="9.0.0")
    assert not SP.verify_pack(src, public_key_hex=pub, app_version="6.2.45").ok
    assert SP.verify_pack(src, public_key_hex=pub, app_version="9.1.0").ok


def test_version_gate_is_not_applied_when_no_version_is_given(tmp_path, keys):
    """不给 ``app_version`` 就不判版本 —— 打包工具靠这个。

    管理员机器的版本和目标机器本来就可能不同；一个特意给未发布版本打的包，
    在打包时被自己拦下来是荒谬的。那道门属于客户端。
    """
    priv, pub = keys
    assert SP.verify_pack(make_pack(tmp_path, priv, min_app="9.0.0"),
                          public_key_hex=pub).ok


def test_reasons_lists_every_problem_not_just_the_first(tmp_path, keys):
    """三个文件全改坏 → 三条理由。修一条撞一条是最消耗人的失败方式。"""
    priv, pub = keys
    src = make_pack(tmp_path, priv, files={
        "builtins/a.py": b"a\n", "builtins/b.py": b"b\n", "builtins/c.py": b"c\n"})
    bad = mutate(src, tmp_path / "bad.zip", lambda zi, zo: [
        zo.writestr(i, zi.read(i.filename)
                    + (b"x" if i.filename.startswith(SP.FILES_PREFIX) else b""))
        for i in zi.infolist()])
    res = SP.verify_pack(bad, public_key_hex=pub)
    assert len([r for r in res.reasons if "对不上" in r]) >= 3


# ---------------------------------------------------------------------------
# 落盘边界
# ---------------------------------------------------------------------------

def test_install_puts_files_under_packs_and_enables_them(tmp_path, keys):
    priv, pub = keys
    from mast.skills.overlay import manifest as M, paths as P

    res = SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub)
    assert res.ok, res.reasons
    assert (P.packs_dir() / "probe" / "builtins" / "pack_probe.py").is_file()
    assert res.enabled == ["_packs/probe/builtins/pack_probe.py"]
    assert [e.path for e in M.load().entries if e.enabled] == res.enabled
    assert res.needs_reload, "「装好了」不等于「生效了」——中间还差一次重载"


def test_no_partial_pack_on_failure(tmp_path, keys, monkeypatch):
    """落盘中途炸掉 → ``_packs/<id>/`` 不存在，不留半个文件。"""
    priv, pub = keys
    from mast.skills.overlay import paths as P

    real = SP.write_installed_manifest

    def boom(*a, **k):
        raise OSError("模拟磁盘满")

    monkeypatch.setattr(SP, "write_installed_manifest", boom)
    res = SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub)
    assert not res.ok
    assert not (P.packs_dir() / "probe").exists()
    assert not list(P.packs_dir().glob(".staging-*")), "临时目录也要清掉"
    monkeypatch.setattr(SP, "write_installed_manifest", real)


def test_a_rejected_pack_leaves_the_previous_version_alone(tmp_path, keys):
    """新包验不过时，已经装好的那一版必须原样还在。"""
    priv, pub = keys
    from mast.skills.overlay import paths as P

    assert SC.install_pack_file(make_pack(tmp_path, priv, version="1.0.0"),
                                public_key_hex=pub).ok
    bad = make_pack(tmp_path, priv, version="2.0.0", sign=False)
    assert not SC.install_pack_file(bad, public_key_hex=pub).ok
    v = SP.verify_installed(P.packs_dir() / "probe", public_key_hex=pub)
    assert v.ok and v.manifest is not None and v.manifest.version == "1.0.0"


def test_install_reports_a_shadowing_loose_file(tmp_path, keys):
    """本机同名松散文件会遮盖 pack 里的那份 —— **必须说出来**。

    不说这一句就会有那个经典事故：「我推了修复过去，它没生效」，而两边看起来
    都正常。
    """
    priv, pub = keys
    from mast.skills.overlay import paths as P

    loose = P.entry_path("builtins/pack_probe.py")
    loose.parent.mkdir(parents=True, exist_ok=True)
    loose.write_text("# 本机自己改的\n", encoding="utf-8")
    res = SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub)
    assert res.ok
    assert res.shadowed == ["builtins/pack_probe.py"]
    assert "遮盖" in res.describe()


def test_auto_enable_can_be_turned_off_per_machine(tmp_path, keys):
    """有些机器正在跑长实验，用户要自己决定什么时候换。"""
    priv, pub = keys
    from mast.skills.overlay import manifest as M

    man = M.load()
    man.auto_enable_packs = False
    M.save(man)
    res = SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub)
    assert res.ok and res.enabled == []
    assert not res.needs_reload
    assert [e for e in M.load().entries if e.enabled] == []
    assert M.load().auto_enable_packs is False, "改清单不能把它悄悄打开"


def test_unreadable_manifest_blocks_auto_enable_instead_of_wiping_it(tmp_path, keys):
    """清单读不出来 ⇒ **不写**。当成空的写回去会抹掉用户已启用的全部条目。"""
    priv, pub = keys
    from mast.skills.overlay import paths as P

    P.overlay_dir(create=True)
    P.manifest_path().write_text("{ 这不是 JSON", encoding="utf-8")
    res = SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub)
    assert res.ok and res.enabled == []
    assert any("读不出来" in r for r in res.reasons)
    assert "这不是 JSON" in P.manifest_path().read_text(encoding="utf-8"), (
        "坏清单被覆盖了 —— 用户原来启用的条目会一起没")


def test_remove_pack_also_drops_its_manifest_entries(tmp_path, keys):
    """只删目录不摘清单，下次重载会有一串「文件不在」，看起来像故障。"""
    priv, pub = keys
    from mast.skills.overlay import manifest as M

    assert SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub).ok
    ok, msg = SC.remove_pack("probe")
    assert ok, msg
    assert [e.path for e in M.load().entries if e.path.startswith("_packs/")] == []


# ---------------------------------------------------------------------------
# 加载边界 —— 「手放文件冒充签名版」这条路必须是关的
# ---------------------------------------------------------------------------

def test_pack_files_are_reverified_at_load_time(tmp_path, keys):
    """落盘之后手改一个字节 → 复核必须发现。

    下载时验过一次不够：那次和现在之间隔着一段任何人都能写的时间，而覆盖层会把
    这个目录下的东西标成 ``verified:``。
    """
    priv, pub = keys
    from mast.skills.overlay import paths as P

    assert SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub).ok
    d = P.packs_dir() / "probe"
    assert SP.verify_installed(d, public_key_hex=pub).ok

    f = d / "builtins" / "pack_probe.py"
    f.write_bytes(f.read_bytes() + "\n# 本地改的\n".encode("utf-8"))
    v = SP.verify_installed(d, public_key_hex=pub)
    assert not v.ok
    assert any("被本地改动过" in r for r in v.reasons)


def test_a_hand_placed_directory_cannot_pose_as_a_signed_pack(tmp_path, keys):
    """手工造一个 ``_packs/<id>/`` 目录 → 复核拒绝。

    这是这一整套东西的核心：``verified:`` 那个标记必须有凭据，否则它就是一句
    谎话，而它出现在 UI 上、出现在执行记录里。
    """
    _priv, pub = keys
    from mast.skills.overlay import paths as P

    fake = P.packs_dir(create=True) / "fakepack" / "builtins"
    fake.mkdir(parents=True)
    (fake / "x.py").write_text(SKILL_SRC, encoding="utf-8")
    v = SP.verify_installed(P.packs_dir() / "fakepack", public_key_hex=pub)
    assert not v.ok
    assert any(".pack.json" in r for r in v.reasons)


def test_a_tampered_installed_manifest_is_rejected(tmp_path, keys):
    """改 ``.pack.json`` 里的 sha 想让改过的文件「对上」→ 签名不过。"""
    priv, pub = keys
    from mast.skills.overlay import paths as P

    assert SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub).ok
    d = P.packs_dir() / "probe"
    f = d / "builtins" / "pack_probe.py"
    f.write_bytes("# 换掉的内容\n".encode("utf-8"))
    mp = d / SP.INSTALLED_MANIFEST
    doc = json.loads(mp.read_text(encoding="utf-8"))
    doc[K.FILES][0][K.SHA256] = SP.sha256_bytes(f.read_bytes())
    doc[K.FILES][0][K.SIZE] = f.stat().st_size
    mp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    v = SP.verify_installed(d, public_key_hex=pub)
    assert not v.ok
    assert any("签名" in r for r in v.reasons)


def test_installed_packs_reports_invalid_ones_rather_than_hiding_them(tmp_path, keys):
    priv, pub = keys
    from mast.skills.overlay import paths as P

    assert SC.install_pack_file(make_pack(tmp_path, priv), public_key_hex=pub).ok
    f = P.packs_dir() / "probe" / "builtins" / "pack_probe.py"
    f.write_bytes(b"# changed\n")
    rows = SC.installed_packs(public_key_hex=pub)
    assert len(rows) == 1
    assert rows[0]["valid"] is False
    assert rows[0]["signature"] == "invalid"
    assert rows[0]["reasons"], "拒绝了却不说为什么，等于没说"


# ---------------------------------------------------------------------------
# 老门那条红线
# ---------------------------------------------------------------------------

def test_skills_upload_still_rejects_code(tmp_path):
    """``/skills/upload`` 那条「永不接收 .py / 代码文件」的红线还在。

    新通道另开一扇门，是因为它有验签；老门的语境是**未签名**的代码载荷，
    对它仍然成立。这条测试钉住它没在加新功能时被顺手拆掉。
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mast.update import server as SRV

    tok = SRV.generate_token()
    SRV.write_token(tmp_path, tok)
    c = TestClient(SRV.build_app(tmp_path))
    r = c.post("/skills/upload",
               json={"manifest": {"name": "x", "nodes": "def f(): pass"}},
               headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 400
    assert "code files are NOT accepted" in str(r.json().get("detail"))


def test_server_refuses_to_publish_an_unsigned_pack(tmp_path, keys):
    """服务器只存不签，但它自己先验一遍 —— 防管理员手滑传上一个验不过的包。"""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mast.update import server as SRV

    priv, _pub = keys
    SRV.write_token(tmp_path, SRV.generate_token())
    # 发布端点要**发布**凭据（2026-09-10 起与客户端 token 分开）。
    pub_tok = SRV.generate_token()
    SRV.write_publish_token(tmp_path, pub_tok)
    c = TestClient(SRV.build_app(tmp_path))
    H = {"Authorization": "Bearer " + pub_tok}

    good = make_pack(tmp_path, priv)
    assert c.post("/skills/pack/publish", content=good.read_bytes(),
                  headers=H).status_code == 200
    bad = make_pack(tmp_path, priv, pack_id="nosig", sign=False)
    r = c.post("/skills/pack/publish", content=bad.read_bytes(), headers=H)
    assert r.status_code == 400
    assert "pack rejected" in str(r.json().get("detail"))


def test_pack_endpoints_need_a_token(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mast.update import server as SRV

    SRV.write_token(tmp_path, SRV.generate_token())
    # 两把都配上：这条测的是「裸请求要被拒」，不是「服务器没配好」。
    SRV.write_publish_token(tmp_path, SRV.generate_token())
    c = TestClient(SRV.build_app(tmp_path))
    assert c.post("/skills/pack/publish", content=b"x").status_code == 401
    assert c.get("/skills/pack/index").status_code == 401
    assert c.get("/skills/pack/download/probe").status_code == 401


def test_download_refuses_a_bad_pack_id(tmp_path):
    """``pack_id`` 会成为磁盘上的路径。"""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mast.update import server as SRV

    tok = SRV.generate_token()
    SRV.write_token(tmp_path, tok)
    c = TestClient(SRV.build_app(tmp_path))
    H = {"Authorization": "Bearer " + tok}
    for bad in ("..", "../x", "A_B", "x" * 200):
        r = c.get("/skills/pack/download/" + bad, headers=H)
        assert r.status_code in (400, 404), (bad, r.status_code)


# ---------------------------------------------------------------------------
# 字段名只有一份
# ---------------------------------------------------------------------------

def test_manifest_round_trips_through_the_shared_field_names():
    """写侧和读侧共用 :class:`K`。

    ``update/client.py:473-484`` 记着那次事故：写侧 ``{"file":}``、读侧
    ``{"filename":}``，两边各自都「对」，而链子是断的。往返一遍就能钉住。
    """
    man = PackManifest(pack_id="p", version="1.2.3", author="a",
                       description="d", min_app_version="6.0.0",
                       max_app_version="7.0.0",
                       files=[PackFile("builtins/x.py", "ab" * 32, 7)])
    back = PackManifest.from_dict(man.to_dict())
    assert back == man


def test_signature_is_not_part_of_the_signed_bytes():
    """签名字段本身不能进被签的那份，否则永远验不过。"""
    doc = {"a": 1, K.SIGNATURE: "deadbeef"}
    assert signing.canonical_manifest_bytes(doc) == \
        signing.canonical_manifest_bytes({"a": 1})


def test_pack_id_rules_are_enforced_before_it_becomes_a_directory_name():
    for good in ("probe", "atomic-res", "a1_b2"):
        assert SP.is_valid_pack_id(good)
    for bad in ("", "A", "x", "../x", "a/b", "a" * 200, "-x", "空格 x"):
        assert not SP.is_valid_pack_id(bad), bad


def test_verify_pack_never_raises_on_garbage(tmp_path):
    """喂垃圾进去只能得到 ``ok=False``，不能抛。"""
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip at all")
    assert SP.verify_pack(junk).ok is False
    assert SP.verify_pack(tmp_path / "missing.zip").ok is False
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w"):
        pass
    assert SP.verify_pack(empty).ok is False


def test_verify_installed_never_raises_on_garbage(tmp_path):
    assert SP.verify_installed(tmp_path / "nope").ok is False
    d = tmp_path / "d"
    d.mkdir()
    (d / SP.INSTALLED_MANIFEST).write_text("{ 坏 json", encoding="utf-8")
    assert SP.verify_installed(d).ok is False


def test_oversized_pack_is_rejected_without_reading_it_all(tmp_path, keys):
    _priv, pub = keys
    big = tmp_path / "big.zip"
    big.write_bytes(b"\0" * (SP.MAX_PACK_BYTES + 1))
    res = SP.verify_pack(big, public_key_hex=pub)
    assert not res.ok
    assert any("上限" in r for r in res.reasons)


def test_the_zip_body_is_not_read_before_the_signature_is_checked(tmp_path, keys):
    """签名不过时**不该**去解释 manifest 里的字段。

    那些字段全是攻击者可控的。顺序错了，就是「先按不可信的数据行事，再问它可不可信」。
    这里用一个签名坏、同时 manifest 里写着荒唐 pack_id 的包：拒绝理由里应该只有
    签名那条，不该出现 pack_id 那条 —— 说明代码在签名这一步就返回了。
    """
    priv, pub = keys
    src = make_pack(tmp_path, priv)

    def _m(zi, zo):
        for i in zi.infolist():
            d = zi.read(i.filename)
            if i.filename == SP.MANIFEST_NAME:
                j = json.loads(d)
                j[K.PACK_ID] = "../../ESCAPE"
                d = json.dumps(j).encode()
            zo.writestr(i, d)

    res = SP.verify_pack(mutate(src, tmp_path / "b.zip", _m), public_key_hex=pub)
    assert not res.ok
    assert len(res.reasons) == 1 and "签名" in res.reasons[0], res.reasons


def test_install_via_a_stream_is_capped(tmp_path, keys):
    """服务端上传要边读边卡上限 —— 先整个读进内存再判大小，那道上限就没用了。"""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mast.update import server as SRV

    SRV.write_token(tmp_path, SRV.generate_token())
    tok = SRV.generate_token()
    SRV.write_publish_token(tmp_path, tok)   # 发布端点要发布凭据
    c = TestClient(SRV.build_app(tmp_path))
    r = c.post("/skills/pack/publish",
               content=io.BytesIO(b"\0" * (SP.MAX_PACK_BYTES + 1024)).read(),
               headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 413
