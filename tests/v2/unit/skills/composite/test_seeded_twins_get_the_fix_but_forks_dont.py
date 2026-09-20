"""出厂孪生体的**重播守卫**:修好的东西要真的到机器上,而人写的东西不许被冲掉。

## 这个文件存在的原因(2026-08-14,任务 #11)

`seed_builtin_composites` 原来是 `if not store.exists(name): save()`。
于是 修复项 修好孪生体之后发现:**代码修好了,在役机器上那份还是旧的** ——
库里已经有这个名字,就永远不再写。而那份旧的带着两个真身早已修掉的危险值:

    height_m = max(width_m*0.05, 1e-9)   细条:256 行挤在 2.5 nm 里
    fwd_line_time = 0.1                  50 nm 上 = 488 nm/s = 刮针的那个速度

builder 里 fork 出去的每一份都继承它们。「产物发出去了」和「机器上那份变了」
是两件事 —— 这个文件钉的就是第二件。

## 两个方向都要钉,而且它们互相制约

* **修得到**:没被改过的旧副本必须被替换成新版(否则修复到不了机器);
* **冲不掉**:用户改过的必须原样不动(否则一次升级毁掉人家的编辑)。

只钉一边都会诱使人把另一边做错:只钉「修得到」→ 无条件覆盖;
只钉「冲不掉」→ 退回 if-not-exists。

## 判据是**内容指纹**,而它有一条静默失效路径

守卫靠「库里这份的指纹 == 某一代出厂内容的指纹」认亲。这里有两处只要一分家
就会**永远不再重播、而且不报错**:

1. `factory_content_hash` 与 `version_store.save` 里那份配方必须逐字相同;
2. 指纹必须扛得住 `to_dict → from_dict → to_dict` 的往返。

两条都单独钉在下面 —— 它们不是「多余的白盒测试」,它们是这个机制唯一会
**无声死掉**的两种死法。

Run:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_seeded_twins_get_the_fix_but_forks_dont.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── path bootstrap ──
_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

from mast.skills.composite.builtin_composites import (  # noqa: E402
    KEY_FACTORY_MARKER,
    _LEGACY_FACTORY_HASHES,
    factory_content_hash,
    reconstructed_composites,
    seed_builtin_composites,
)
from mast.skills.composite.spec import CompositeSpec  # noqa: E402
from mast.skills.composite.version_store import CompositeVersionStore  # noqa: E402

_TWIN = "PreScanCheck"


@pytest.fixture()
def store(tmp_path):
    return CompositeVersionStore(root=tmp_path / "composites")


def _factory(name: str = _TWIN) -> CompositeSpec:
    return next(s for s in reconstructed_composites() if s.name == name)


def _step(spec: CompositeSpec, step_id: str) -> dict:
    """按 id 取步骤 —— **不按下标**。

    第一版写的是 `nodes[0]`(那时 configure 恰好排第一),而任务 #11 在最前面
    插了一道 `need_line_time` 闸,三条测试当场 KeyError。位置是会变的,id 不会
    ——「行序不该承重」的同一条教训。
    """
    return next(n for n in spec.nodes if n.get("id") == step_id)


def _an_old_shipped_copy() -> CompositeSpec:
    """一份**真的发出去过**的旧出厂内容。

    直接照 `_LEGACY_FACTORY_HASHES` 里那一代重建太脆(要么读 git、要么抄一份
    spec 进测试)。这里改成:拿当前出厂 spec,把两个危险值**改回旧样子**,
    然后把它的指纹**临时登记**进历史表 —— 等价于「这台机器上装的是上一代」。
    这样测的是**守卫的逻辑**,而不是某一个写死的哈希还对不对
    (后者由 `test_the_frozen_legacy_table_still_matches_git_history` 单管)。
    """
    old = CompositeSpec.from_dict(_factory().to_dict())
    _step(old, "configure")["params"]["height_m"] = {
        "$expr": "max(width_m * 0.05, 1e-9)"}
    speed = _step(old, "speed")
    speed["params"]["fwd_line_time"] = 0.1
    speed["params"]["bwd_line_time"] = 0.1
    return old


@pytest.fixture()
def legacy_registered(monkeypatch):
    """把「上一代」的指纹登记进历史表,并返回那份 spec。"""
    old = _an_old_shipped_copy()
    patched = dict(_LEGACY_FACTORY_HASHES)
    patched[_TWIN] = frozenset(
        set(_LEGACY_FACTORY_HASHES.get(_TWIN, frozenset()))
        | {factory_content_hash(old)})
    monkeypatch.setattr(
        "mast.skills.composite.builtin_composites._LEGACY_FACTORY_HASHES",
        patched)
    return old


# ══════════════════════════════════════════════════════════════════════
# 一、修得到:没被改过的旧副本会被重播
# ══════════════════════════════════════════════════════════════════════

def test_an_untouched_factory_copy_is_replayed(store, legacy_registered):
    """在役机器上那份没人动过的旧副本 ⇒ 换成新版。"""
    store.save(legacy_registered)
    assert _step(store.load(_TWIN), "configure")["params"]["height_m"] == {
        "$expr": "max(width_m * 0.05, 1e-9)"}, "前提没搭对:库里不是旧的那份"

    written = seed_builtin_composites(store)

    assert _TWIN in written
    now = store.load(_TWIN)
    assert factory_content_hash(now) == factory_content_hash(_factory())
    # 两个危险值都没了 —— 这才是这次重播要送到机器上的东西。
    assert _step(now, "configure")["params"]["height_m"] == {
        "$expr": "width_m"}, "细条还在"
    speed = _step(now, "speed")
    for key in ("fwd_line_time", "bwd_line_time"):
        assert speed["params"][key] == {"$expr": "line_time_s"}, (
            f"{key} 还是写死的刮针速度")


def test_the_replaced_version_is_archived_not_destroyed(store, legacy_registered):
    """重播**不是破坏性的**:旧版进 history,用户随时 restore。

    这条撑着上一条的正当性 —— 「替换」若真会丢东西,那守卫再准也不该动手。
    """
    store.save(legacy_registered)
    seed_builtin_composites(store)

    versions = [v["version"] for v in store.list_versions(_TWIN)]
    assert len(versions) >= 2, f"旧版没归档:{versions}"
    restored = store.restore(_TWIN, min(versions))
    assert _step(restored, "configure")["params"]["height_m"] == {
        "$expr": "max(width_m * 0.05, 1e-9)"}, "归档的不是被替换掉的那一版"


def test_a_missing_entry_is_seeded_and_stamped(store):
    """首次 seed 要**盖上出厂标记** —— 下一代靠它自证,不必再查历史表。"""
    written = seed_builtin_composites(store)
    assert _TWIN in written
    meta = store.load_meta(_TWIN)
    assert meta.get(KEY_FACTORY_MARKER) == factory_content_hash(_factory()), (
        "没盖标记 —— 那么下一次改孪生体时,又得往历史表里手工加哈希")


def test_a_stamped_untouched_copy_replays_without_the_legacy_table(store,
                                                                   monkeypatch):
    """标记那一路必须**独立于**历史表工作(否则迁移垫片永远删不掉)。

    做法:先按老版本 seed 并盖标记,再把历史表清空,然后改出厂内容重播。
    """
    old = _an_old_shipped_copy()
    store.save(old, extra_meta={KEY_FACTORY_MARKER: factory_content_hash(old)})
    monkeypatch.setattr(
        "mast.skills.composite.builtin_composites._LEGACY_FACTORY_HASHES", {})

    written = seed_builtin_composites(store)

    assert _TWIN in written, "带标记的未改副本没被认出来 —— 标记那一路没在工作"
    assert factory_content_hash(store.load(_TWIN)) == factory_content_hash(
        _factory())


# ══════════════════════════════════════════════════════════════════════
# 二、冲不掉:用户改过的一律不碰
# ══════════════════════════════════════════════════════════════════════

def test_an_operator_edited_copy_is_never_clobbered(store, legacy_registered):
    """人改过的东西不许被一次升级冲掉。"""
    edited = legacy_registered
    edited.description = "用户自己写的说明"
    store.save(edited)

    written = seed_builtin_composites(store)

    assert _TWIN not in written
    assert store.load(_TWIN).description == "用户自己写的说明"


def test_an_unrecognised_copy_is_left_alone_and_says_so(store, caplog):
    """身世不明 ⇒ **不碰**,而且日志要说出来(静默跳过 = 又一个查不出的洞)。"""
    import logging

    mine = CompositeSpec.from_dict(_factory().to_dict())
    _step(mine, "configure")["params"]["width_m"] = {"$expr": "width_m * 2"}
    store.save(mine)

    with caplog.at_level(logging.INFO,
                         logger="mast.skills.composite.builtin_composites"):
        written = seed_builtin_composites(store)

    assert _TWIN not in written
    said = [r.getMessage() for r in caplog.records if _TWIN in r.getMessage()]
    assert said, "跳过了却一声不吭 —— 用户无从知道自己那份不会再更新"
    assert any("不动它" in m for m in said), said


def test_seeding_is_idempotent(store):
    """跑第二遍什么都不该写 —— 否则每次开面板都白涨一个版本号。"""
    seed_builtin_composites(store)
    v1 = {s["name"]: s["version"] for s in store.list_specs()}
    assert seed_builtin_composites(store) == []
    assert {s["name"]: s["version"] for s in store.list_specs()} == v1


# ══════════════════════════════════════════════════════════════════════
# 三、这个机制**唯一会无声死掉**的两种死法
# ══════════════════════════════════════════════════════════════════════

def test_the_hash_recipe_matches_the_one_the_store_writes(store):
    """两份配方分家 ⇒ 每一条都被判成「被改过」⇒ **永不重播,而且不报错**。

    `version_store.save` 自己也算一份 `_content_sha256`(用于漂移检测)。
    守卫必须与它同口径,否则两边各算各的,而症状是「什么都没发生」。
    """
    spec = _factory()
    store.save(spec)
    assert store.load_meta(_TWIN).get("_content_sha256") == (
        factory_content_hash(spec)), (
        "守卫的指纹配方与 version_store.save 里那份不一样了 —— "
        "改动其中任何一处都必须同时改另一处")


@pytest.mark.parametrize("spec", reconstructed_composites(),
                         ids=lambda s: s.name)
def test_the_hash_survives_a_store_round_trip(spec):
    """指纹必须扛得住 `to_dict → from_dict → to_dict`。

    守卫比的是「库里读回来的那份」与「内存里的出厂那份」。中间隔着一次
    JSON 往返,只要有一个字段在往返中被规范化掉,两个指纹就永远对不上,
    于是**每一台机器都被判成改过**。这条对全部 10 个孪生体都跑。
    """
    assert factory_content_hash(spec) == factory_content_hash(
        CompositeSpec.from_dict(spec.to_dict())), f"{spec.name} 往返后指纹变了"


def test_the_frozen_legacy_table_still_matches_git_history():
    """历史表里必须真有**上一代已发布**的那个指纹,而且不含当前这一代。

    * 含不到上一代 ⇒ 在役机器全部拿不到修复(这次修的正是这个);
    * 误含当前这一代 ⇒ 「已经是最新」与「该重播」两条分支会打架。

    上一代 = git HEAD 里那份(修复项/#11 的改动都还没提交)。这个值 2026-08-14
    由遍历该文件全部 10 次提交算出,去重 6 代,全部登记在案。
    """
    head_hash = "6eaebcf237a1a44900700bf762803250109a955757e7efd1a21bedf2defd3b8b"
    table = _LEGACY_FACTORY_HASHES.get(_TWIN, frozenset())
    assert head_hash in table, (
        "HEAD 那一代出厂内容不在历史表里 —— 在役机器上的副本会被判成「改过」,"
        "这次修复一台机器都到不了。重新生成见 builtin_composites 里的注释。")
    assert factory_content_hash(_factory()) not in table, (
        "当前出厂内容被登记成了「历史代」—— 那一条判决会与「已经是最新」冲突")


# ══════════════════════════════════════════════════════════════════════
# 四、变异验证:先证明动了手,再看它红不红
# ══════════════════════════════════════════════════════════════════════

def test_the_mutation_would_be_caught(store, legacy_registered, monkeypatch):
    """把守卫退回 if-not-exists ⇒ 重播必须停止发生。

    不改文件,用 monkeypatch 模拟「有人把老逻辑写回来」——改文件的变异跑一次
    就把仓库弄脏,而这里要的只是「这条测试抓不抓得住」。
    """
    import mast.skills.composite.builtin_composites as bc

    store.save(legacy_registered)
    old_hash = factory_content_hash(store.load(_TWIN))

    # ① 变异:身世永远查不出来 ⇒ 守卫认不了亲 ⇒ 退化成只写缺失的那些。
    monkeypatch.setattr(bc, "_stored_content_hash", lambda *a, **k: None)
    written = seed_builtin_composites(store)

    # 先证明**变异真的生效了**:库里那份没被换掉。
    assert _TWIN not in written, "变异没生效 —— 那么下面的「抓住了」不算数"
    assert factory_content_hash(store.load(_TWIN)) == old_hash

    # ② 复原之后必须重新能重播(证明上面的「没换」是变异造成的,不是环境)。
    monkeypatch.undo()
    monkeypatch.setattr(
        "mast.skills.composite.builtin_composites._LEGACY_FACTORY_HASHES",
        {_TWIN: frozenset({old_hash})})
    assert _TWIN in seed_builtin_composites(store)
    assert factory_content_hash(store.load(_TWIN)) == factory_content_hash(
        _factory())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
