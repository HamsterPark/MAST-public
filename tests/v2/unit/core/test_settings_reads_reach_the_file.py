"""七条设置读路径,读的到底是不是用户写进文件的那个值。

背景(2026-08-10 实测):``core/`` 里有七处写着 ``SettingsStore().get(key)`` ——
无参。``SettingsStore.__init__`` 的 ``config_dir`` 是**必填位置参数**,于是每一次
调用都抛 TypeError、被外面那圈 ``except Exception`` 吞掉、返回兜底值。六条被逐一
实测:没有一条读到过用户写的值。

**这个缺陷之所以在任何界面上都看不出来**,是因为兜底值恰好等于默认值:归档照常
跑、CSV 照常写、对话照常导出。用户把开关关掉,什么反馈都不会有。

所以这份测试钉的**不是函数签名**,是端到端的那一问:
**盘上那个字节写着 X,这条读路径返回的是不是 X。**
形状照抄当时那个探针(``docs/v2/audit/scripts/probe_noarg_settingsstore.py``):
文件里写的每个值都**与默认相反** —— 只有这样,「读到了」和「没读到、返回了默认」
才是两个不同的观测。写成默认值的测试对这个缺陷是全绿的。

跑法::

    $env:PYTHONPATH='<repo>\\MASTv2'
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/core/test_settings_reads_reach_the_file.py -q
"""
from __future__ import annotations

import json

import pytest

from mast.core.runtime import CoreRuntime
from mast.webui.settings_store import (
    KNOWN_KEYS,
    SettingsStore,
    reset_process_store,
    set_process_store,
)

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of

#: 每个值都**与默认相反**。默认值见 KNOWN_KEYS 的注释与各读函数的兜底分支:
#: ingest_* / env_csv / conv_export 默认 True,copy_mode 默认 "copy"。
_OPPOSITE_OF_DEFAULTS: dict = {
    "ingest_enabled": False,
    "ingest_watcher_enabled": False,
    "ingest_copy_mode": "hardlink",
    "env_csv_enabled": False,
    "conv_export_enabled": False,
    "tip_conditioning_overrides": {"PROBE": {"pulse_v": 0.123}},
}


def _unwired_runtime() -> CoreRuntime:
    """一个**没跑过 __init__** 的 CoreRuntime:没有 ``_settings``,方法齐全。

    ``__new__`` 而不是随便一个哑对象:这五个读方法一个 ``self`` 属性都没用,但它们
    要调 ``self._setting(...)`` —— 哑对象上那一句是 AttributeError,会被方法自己的
    ``except`` 吞掉,于是**无论修没修好都返回兜底值**。审计那份探针用的正是哑对象,
    所以它测不出修复(它当时要测的是缺陷,方向相反,没有踩到)。
    """
    return CoreRuntime.__new__(CoreRuntime)


@pytest.fixture()
def settings_on_disk(tmp_path, monkeypatch):
    """把 ``_OPPOSITE_OF_DEFAULTS`` 写进 ``<tmp>/config/ui_settings.json``。

    ``MAST2_PROJECT_ROOT`` 指向 tmp,是为了让**没有 ``self._settings`` 的那条
    回退路径**也解析到这里 —— 否则它会去读用户真实的 config,测试就不再是
    自足的(而且结论会随那台机器的设置变)。
    """
    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    (cfg / "ui_settings.json").write_text(
        json.dumps(_OPPOSITE_OF_DEFAULTS, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    reset_process_store()
    yield tmp_path
    reset_process_store()


def test_probe_values_are_all_whitelisted():
    """探针自检:写进文件的键必须在白名单里,否则 ``_load`` 会把它们丢掉,
    而「读不到」的原因就变成了探针自己写错 —— 一个长得一模一样的假阳性。"""
    unknown = sorted(set(_OPPOSITE_OF_DEFAULTS) - set(KNOWN_KEYS))
    assert not unknown, f"这些键不在 KNOWN_KEYS 里,测试本身写错了:{unknown}"


def test_probe_values_really_differ_from_the_fallbacks(tmp_path, monkeypatch):
    """探针自检之二:文件里的值必须**与兜底值不同**,而兜底值确实是那些值。

    若某天默认值改成了和探针一样,下面每一条断言都会在缺陷仍然存在的情况下变绿
    —— 那正是「测出来是零」与「没测出来」混成一句话的形状。
    """
    hardcoded_fallbacks = {
        "ingest_enabled": True, "ingest_watcher_enabled": True,
        "ingest_copy_mode": "copy", "env_csv_enabled": True,
        "conv_export_enabled": True,
    }
    for k, v in hardcoded_fallbacks.items():
        assert _OPPOSITE_OF_DEFAULTS[k] != v, (
            f"{k} 的探针值等于兜底值 {v!r} —— 这条断言无法区分「读到了」与"
            f"「读失败返回了默认」,换一个与默认不同的值")

    # 而且兜底值确实是这些值:把设置目录指向一个空目录,五条读路径必须给出
    # hardcoded_fallbacks。少了这一步,上面那圈比对的只是作者写的两份常量。
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "empty"))
    reset_process_store()
    try:
        rt = _unwired_runtime()
        assert {
            "ingest_enabled": CoreRuntime._ingest_enabled(rt),
            "ingest_watcher_enabled": CoreRuntime._ingest_watcher_enabled(rt),
            "ingest_copy_mode": CoreRuntime._ingest_copy_mode(rt),
            "env_csv_enabled": CoreRuntime._env_csv_enabled(rt),
            "conv_export_enabled": CoreRuntime._chat_export_enabled(rt),
        } == hardcoded_fallbacks
    finally:
        reset_process_store()


# ── ① 没有 self._settings 的调用方(回退路径) ─────────────────────────────────
def test_runtime_reads_reach_the_file_without_a_wired_store(settings_on_disk):
    """五条 runtime 读路径,在**没有**接线 store 时也必须读到盘上的值。

    用 ``__new__`` 造的 CoreRuntime(没跑 __init__,所以没有 ``_settings``),不构造
    整个运行时。缺陷版本在这里会返回 True/True/'copy'/True/True。
    """
    rt = _unwired_runtime()
    got = {
        "ingest_enabled": CoreRuntime._ingest_enabled(rt),
        "ingest_watcher_enabled": CoreRuntime._ingest_watcher_enabled(rt),
        "ingest_copy_mode": CoreRuntime._ingest_copy_mode(rt),
        "env_csv_enabled": CoreRuntime._env_csv_enabled(rt),
        "conv_export_enabled": CoreRuntime._chat_export_enabled(rt),
    }
    expect = {k: _OPPOSITE_OF_DEFAULTS[k] for k in got}
    assert got == expect, (
        f"这些读路径没有读到 {settings_on_disk / 'config' / 'ui_settings.json'} "
        f"里的值:实得 {got},盘上是 {expect}")


def test_tip_conditioning_overrides_reach_the_file(settings_on_disk):
    """第六条:修针方案的用户覆写。缺陷版本恒返回 ``{}``,而 ``{}`` 正好等于
    「没设覆写」—— 所以整条解析链看上去完全正常。"""
    from mast.core.tip_conditioning_resolver import _read_overrides

    assert _read_overrides() == _OPPOSITE_OF_DEFAULTS["tip_conditioning_overrides"]


def test_experiment_root_reaches_the_holder(tmp_path, monkeypatch):
    """第七条:``experiment_root``(内联在 ``_init_experiment_folders`` 里)。

    这一条走的是完整的那个方法:它先把设置注入 holder,再看 ``ingest_enabled``
    决定要不要继续。文件里 ingest 关着,所以它注入完就返回 —— 于是这条测试
    验的是**注入**,不需要起后台线程。

    断言用 ``get_experiment_root_setting()`` 而不是 ``experiment_root()``:后者
    优先看 ``MAST_EXPERIMENT_ROOT`` 环境变量,那会把「设置有没有被读到」这一问
    答成另一个问题。
    """
    from mast.core import experiment_paths as ep

    cfg = tmp_path / "config"
    cfg.mkdir(parents=True)
    wanted = str(tmp_path / "PROBE_EXPERIMENTS")
    (cfg / "ui_settings.json").write_text(
        json.dumps({"experiment_root": wanted, "ingest_enabled": False}),
        encoding="utf-8")
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    reset_process_store()

    before = ep.get_experiment_root_setting()
    try:
        CoreRuntime._init_experiment_folders(_unwired_runtime())
        assert ep.get_experiment_root_setting() == wanted, (
            "experiment_root 设置没有到达 holder —— 「实验数据比软件活得久,OTA "
            "和重装都不该碰它」这句承诺现在做不到")
    finally:
        ep.set_experiment_root(before)   # 进程级 holder,必须还原
        reset_process_store()


# ── ② 有 self._settings 的调用方(接线路径) ───────────────────────────────────
def test_wired_store_wins_over_the_file_on_disk(tmp_path, monkeypatch):
    """接了 ``self._settings`` 时读的是**那个对象**,不是 project_root 下的文件。

    这不是风格问题:``POST /api/settings`` 写的就是那个对象(bootstrap 把
    ``ctx.settings_store`` 接成 ``rt._settings``)。读另一个实例 = 读它构造时的
    快照,用户刚改的值要等重启才看得见。
    """
    disk = tmp_path / "disk"
    (disk / "config").mkdir(parents=True)
    (disk / "config" / "ui_settings.json").write_text(
        json.dumps({"ingest_copy_mode": "copy"}), encoding="utf-8")
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(disk))
    reset_process_store()

    wired_dir = tmp_path / "wired"
    wired = SettingsStore(wired_dir)
    wired.update(ingest_copy_mode="hardlink")

    rt = _unwired_runtime()
    rt._settings = wired
    try:
        assert CoreRuntime._ingest_copy_mode(rt) == "hardlink"
    finally:
        reset_process_store()


def test_a_second_store_on_the_same_dir_is_blind_to_the_first_ones_write(tmp_path):
    """为什么「同一个对象」是一条真要求,而不是洁癖。

    ``SettingsStore`` 在 ``__init__`` 里读一次文件,之后只答自己的 ``_data``。
    所以第二个长命实例即使目录完全正确,也看不见第一个实例后来写的东西。
    这条测试钉的是**那个机制** —— 如果哪天 ``get()`` 改成每次都看盘(mtime 之类),
    这条会红,而那时「必须是同一个对象」这条要求就可以放松了。没有它,下面那两条
    身份断言看起来就只是风格偏好,迟早被当成冗余删掉。
    """
    cfg = tmp_path / "config"
    first = SettingsStore(cfg)
    second = SettingsStore(cfg)          # 同一个目录,另一个实例
    first.update(ingest_enabled=False)   # 盘上已经是 False 了
    assert second.get("ingest_enabled") is None, (
        "第二个实例看见了第一个实例的写入 —— SettingsStore 现在会重读文件了?"
        "那就去放松 settings_store_for_runtime 的身份要求,并把这条测试改掉")


def test_a_write_is_visible_to_the_next_read_without_a_restart(tmp_path, monkeypatch):
    """写进去 → 立刻读得到,读的是**发布出去的那个 store**。

    ``MAST2_PROJECT_ROOT`` 故意指向另一个空目录:一个「不看发布、每次照
    project_root 新建一个」的实现,路径就会落到那个空目录上 —— 于是这条测试
    区分得开「读对了对象」和「碰巧读对了文件」。
    """
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "elsewhere"))
    store = SettingsStore(tmp_path / "config")
    set_process_store(store)
    try:
        rt = _unwired_runtime()
        assert CoreRuntime._ingest_enabled(rt) is True            # 缺省 = 开
        store.update(ingest_enabled=False)                        # 用户关掉
        assert CoreRuntime._ingest_enabled(rt) is False, (
            "写进 store 的值没有被下一次读看到 —— 读的多半是第二个实例的旧快照")
    finally:
        reset_process_store()


def test_runtime_publishes_its_store_so_module_level_readers_agree(tmp_path):
    """``CoreRuntime.setup()`` 必须把 ``self._settings`` 发布出去。

    没有它,``_read_overrides()`` 这类**没有 self 可用**的读者会自己建第二个
    实例:路径对、值也对,但用户刚写的那次改动看不见。这条测试只跑发布这一步
    (setup 全程要起一堆子系统),断言的是发布之后模块级读者拿到的是**同一个对象**。
    """
    from mast.webui.settings_store import settings_store_for_runtime

    store = SettingsStore(tmp_path / "config")
    set_process_store(store)
    try:
        assert settings_store_for_runtime() is store
    finally:
        reset_process_store()


def test_setup_wires_the_publication(monkeypatch):
    """真的是 ``setup()`` 在发布 —— 不是这份测试自己手动调的。

    读源码级的断言在这里是恰当的:``setup()`` 会起 ConnectionPool / 视觉 /
    orchestrator,整段跑不动;而要钉的性质只有一句「那一行在不在」。
    """
    import inspect

    src = source_of(CoreRuntime.setup)
    # 先剥注释:这个文件里正好有几行注释在**解释**这件事,不剥就会把解释当成实现
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "set_process_store(self._settings)" in code, (
        "CoreRuntime.setup 没有发布它的 SettingsStore —— 模块级读者会各建各的")
