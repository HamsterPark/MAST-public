"""结构闸门:``KNOWN_KEYS`` 里的每个键,要么写得进去,要么在这里写下为什么不。

## 为什么需要这一条

一个设置项要真的可用,**三张手维护的名单必须同时点头**:

* ``SettingsStore.KNOWN_KEYS`` —— 不在里面,``update()`` 静默丢弃;
* ``SettingsWriteRequest``     —— 不在上面,pydantic ``extra='ignore'`` 静默丢弃;
* ``SettingsResponse``         —— 不在上面,``GET /api/settings`` 永远返回 None。

三张表之间**没有任何东西在对账**,而三种漏法的症状完全一样:界面一切正常,值没
生效。2026-08-10 逮到的两件事都是这个形状 —— Nanonis 主机与四个端口在第一、
第三张表里而不在第二张,于是「保存」是个 no-op 而 toast 绿字「已保存」;
``tip_conditioning_overrides`` 同样只缺第二张。

既有教训 记着这个形状已经第五次。所以这里不再补
第四次名单,而是让「只改了一边」在结构上红。

## 判据的两个方向

只写「新键必须有写入口」是不够的:那样的闸门只会把每一个新键推进豁免名单里。
所以断言是**集合相等** —— 补上了写入口却忘了从豁免名单删掉,同样红。

## 这张闸门看不见什么(2026-08-10 补记)

写下这张闸门的当天,它**全绿**,而 ``POST {"tool_refine_min_chars": 1234}`` 在
6.2.6 上得到 ``ok:true, rejected:{}`` 而值根本没写进去 —— 因为那 18 个键当时都
躺在下面的豁免名单里,各带一句「TODO: 待补」。闸门答的是「有没有写下为什么没有
写入口」,答案是「写下了」;它**原理上**答不了「一次真实的写有没有落盘」和
「写一个名单外的键会不会报成功」—— 那两件是运行时行为,不是集合关系。

那一半在 ``test_settings_write_lands.py``:它发 HTTP,逐键 POST→GET→重建 store,
并且断言**每一个豁免键都必须被明确拒绝**。两张一起才完整;删掉任何一张,另一张
都会继续绿着。

跑法::

    $env:PYTHONPATH='<repo>\\MASTv2'
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/api/test_settings_write_reachability.py -q
"""
from __future__ import annotations

import pytest

from mast.api.schemas import SettingsResponse
from mast.api.schemas_settings_admin_write import SettingsWriteRequest
from mast.webui.settings_store import KNOWN_KEYS

# 豁免理由的三种标签。写进豁免名单时必须选一个 —— 「只该由程序写」和
# 「本来该有、还没做」是完全不同的两件事,混在一起这张表就只是一堆待办。
DEDICATED = "专用路由"   # 有自己的端点写它,不该走统一设置写
NOT_BUILT = "未实现"     # KNOWN_KEYS 的说明承诺了一个不存在的功能
TODO = "待补"            # 程序在读、用户该管,而写入口确实缺着
_TAGS = (DEDICATED, NOT_BUILT, TODO)

#: 键 → "标签: 为什么"。**这张表只能因为一个键真的变了状态而改。**
#: 2026-08-10 逐个复核过一遍(不是照抄审计报告 —— 每一条都自己查过读写位置)。
WRITE_ENTRY_EXEMPT: dict[str, str] = {
    # ── 有自己的端点 ────────────────────────────────────────────────────────
    "instrument_init": (
        f"{DEDICATED}: 由 POST /api/instrument-init/... 写"
        "(routes/instrument_init.py:267 `store.update(**{ii.SETTINGS_KEY: record})`)。"
        "它是一条**完成记录**不是一组参数,数值都住在各自的真源里,"
        "让它走统一写会给出第二张与真源不一致的表。"
    ),

    # ── 真的缺写入口(程序在读,用户该管) ───────────────────────────────────
    #
    # 2026-08-10 第二轮:这一组原本有 17 条,现在只剩 1 条。其余 13 条补进了
    # SettingsWriteRequest + SettingsResponse,3 条(ingest_max_file_mb /
    # ingest_max_gb / nanonis_follow_session_path)因为**全仓零引用**从
    # KNOWN_KEYS 里删掉了(墓碑在 webui/settings_store.py,连同「要加回来需要
    # 回答什么」)。
    "experiment_root": (
        f"{TODO}: 归档根目录。**补字段之前要先想清楚运行中改它意味着什么**。"
        "已核对(2026-08-10,不是照抄):`logging/v2/schema.py:196` 的 "
        "`scan_files.current_path` 存的是路径且该表在 `_APPEND_ONLY_FACT_TABLES` "
        "里(BEFORE UPDATE 触发器 ABORT),所以历史行**永远改不回来** —— 改根之后"
        "实验树分裂成两半、老行指着老根,而没有任何一条 UPDATE 能修它。"
        "另外 `core/experiment_paths.py:180` 的 docstring 现在写着「由 runtime "
        "启动 hydration 和 POST /api/settings 调用」,后半句是假的(路由从没调过"
        "`set_experiment_root`)—— 补写入口时那句话才会变成真的,别只加字段。"
        "**要推翻这条豁免需要回答**:改根之后老行怎么办 —— 迁移、双根共存、"
        "还是只允许在没有任何实验行时改?三个答案的实现完全不同。"
    ),
}

#: 写 schema 上**故意不进 KNOWN_KEYS** 的字段。
WRITE_ONLY_FIELDS: dict[str, str] = {
    "admin_pin": (
        "PIN 只上一次线,与 config/admin_pin.txt 的 SHA-256 比对后当场丢弃 —— "
        "它进了 KNOWN_KEYS 就会被明文写进 ui_settings.json。"
    ),
}

#: 写得进但不在读 schema 上的键(有专用读端点的)。
READ_ENTRY_EXEMPT: dict[str, str] = {
    "zctrl_presets": (
        f"{DEDICATED}: 由 GET /api/settings/zctrl-presets 读"
        "(routes/settings.py:75),前端 ZCtrlPresetsSection 走的就是那个端点。"
    ),
}


def _write_fields() -> set[str]:
    return set(SettingsWriteRequest.model_fields)


def _read_fields() -> set[str]:
    return set(SettingsResponse.model_fields)


# ── 闸门自检 ─────────────────────────────────────────────────────────────────
def test_the_gate_is_looking_at_something():
    """三张表都得是非空的真表。

    一条匹配不到任何东西的闸门会一直绿,和「确实没问题」输出一模一样。
    """
    assert len(KNOWN_KEYS) > 40, f"KNOWN_KEYS 只有 {len(KNOWN_KEYS)} 个,真源多半取错了"
    assert len(_write_fields()) > 20
    assert len(_read_fields()) > 20


def test_every_exemption_carries_a_tagged_reason():
    """豁免必须写清楚是哪一种、为什么。

    没有标签的话「只该由程序写」和「本来该有、还没做」会混成一堆,
    而这两件事的下一步动作完全相反。
    """
    for key, reason in {**WRITE_ENTRY_EXEMPT, **READ_ENTRY_EXEMPT}.items():
        assert any(reason.startswith(t + ":") for t in _TAGS), (
            f"{key} 的豁免理由没有以标签开头(应为 {_TAGS} 之一):{reason[:40]!r}")
        body = reason.split(":", 1)[1]
        assert len(body.strip()) >= 20, (
            f"{key} 的豁免理由太短,说不清为什么:{reason!r}")


def test_a_todo_exemption_says_what_would_overturn_it():
    """``TODO`` 豁免必须写下「要推翻它需要回答什么」。

    ``DEDICATED`` / ``NOT_BUILT`` 是**决定**(有专用路由 / 这个功能不存在),
    而 ``TODO`` 是一句**承诺**:「本来该有,还没做」。承诺不会自己到期,也不会
    让任何人难受 —— 2026-08-10 就是这么攒到 17 条的,而那 17 条恰好等于当时
    「写不进却报成功」的全部键。

    要求写下推翻条件,是把承诺变回决定:后来人面对的不是「要不要推翻前人」
    (只能靠信任),而是一个可查的问题。
    """
    for key, reason in WRITE_ENTRY_EXEMPT.items():
        if not reason.startswith(TODO + ":"):
            continue
        assert "要推翻" in reason, (
            f"{key} 是 TODO 豁免却没写「要推翻这条豁免需要回答什么」——"
            f"没有推翻条件的待办会一直待下去,而它挡着的是一个用户改不了的设置。")


# ── 主闸门 ───────────────────────────────────────────────────────────────────
def test_every_known_key_has_a_write_entry_or_a_written_down_reason():
    """``KNOWN_KEYS - 写 schema`` 必须**正好等于**豁免名单。

    两个方向都要:
      * 多出来的键 = 新加了一个存得下却写不进的设置(静默 no-op);
      * 少掉的键   = 补上了写入口而豁免名单没删,这张表就开始骗人。
    """
    gap = set(KNOWN_KEYS) - _write_fields()
    exempt = set(WRITE_ENTRY_EXEMPT)
    missing = sorted(gap - exempt)
    stale = sorted(exempt - gap)
    assert not missing, (
        f"这些键存得进 SettingsStore 却没有任何 HTTP 写入口 —— 一次 POST 会被 "
        f"pydantic 静默丢掉而响应仍然 ok=True:{missing}\n"
        f"要么给 SettingsWriteRequest 补上字段(别忘了 SettingsResponse 那一侧,"
        f"否则表单存得进读不回),要么在 WRITE_ENTRY_EXEMPT 里写下它为什么不该有。")
    assert not stale, (
        f"这些键已经有写入口了,豁免名单该删掉它们:{stale}")


def test_no_write_field_is_silently_dropped_by_the_store():
    """写 schema 上的字段必须在 KNOWN_KEYS 里,否则 ``update()`` 静默丢弃。

    这是反方向的同一个洞:字段加在了 pydantic 上、前端也在送,而 store 不认,
    响应照样 ok=True。
    """
    orphan = sorted(_write_fields() - set(KNOWN_KEYS) - set(WRITE_ONLY_FIELDS))
    assert not orphan, (
        f"这些字段收得下却存不进(SettingsStore.update 白名单外):{orphan}")
    # 反方向:WRITE_ONLY_FIELDS 里的东西必须真的不在 KNOWN_KEYS 里
    leaked = sorted(set(WRITE_ONLY_FIELDS) & set(KNOWN_KEYS))
    assert not leaked, (
        f"这些字段被标成「故意不持久化」,却进了 KNOWN_KEYS —— 会被写进 "
        f"ui_settings.json:{leaked}")


def test_what_you_can_write_you_can_read_back():
    """写得进就得读得回来,否则表单存完只能靠信仰。

    这条是 D-2 的镜像:那次是「读得到写不进」,反过来同样坏 —— 界面拿 GET 回来的
    值重填表单,读不到就永远显示空的,用户会以为没保存成功而再存一次。
    """
    gap = _write_fields() - _read_fields() - set(WRITE_ONLY_FIELDS)
    exempt = set(READ_ENTRY_EXEMPT)
    missing = sorted(gap - exempt)
    stale = sorted(exempt - gap)
    assert not missing, (
        f"这些键写得进但 GET /api/settings 不返回:{missing}\n"
        f"要么给 SettingsResponse 补字段,要么在 READ_ENTRY_EXEMPT 里写下"
        f"它的专用读端点是哪个。")
    assert not stale, f"这些键已经在读 schema 里了,READ_ENTRY_EXEMPT 该删:{stale}"


def test_read_schema_stays_inside_the_whitelist():
    """读 schema 上的字段必须在 KNOWN_KEYS 里,否则它永远返回 None。"""
    orphan = sorted(_read_fields() - set(KNOWN_KEYS))
    assert not orphan, f"这些读字段的值永远不会有(不在白名单里):{orphan}"


# ── 已经修好的两件,钉住别退回去 ──────────────────────────────────────────────
@pytest.mark.parametrize("key", [
    "nanonis_host", "nanonis_port_main", "nanonis_port_monitor",
    "nanonis_port_data", "nanonis_port_emergency",
])
def test_nanonis_connection_keys_are_writable(key):
    """2026-08-10 修好的那五个。上面的集合相等已经覆盖了它们,这里再点一次名 ——
    集合断言红的时候只说「豁免名单对不上」,而这五条会直接说是哪个功能没了。"""
    assert key in _write_fields(), f"{key} 又从写 schema 上消失了 —— 保存按钮变回 no-op"
    assert key in KNOWN_KEYS and key in _read_fields()
