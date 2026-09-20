"""实验文件夹的自解读元数据 —— ``experiment.json`` / ``sample.json`` / ``README.md``。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §5

这些边车文件是"自包含"的实质内容：整个实验文件夹拷到另一台没装 MAST 的电脑，
靠 ``experiment.json`` + ``README.md`` 就能读懂结构，靠 ``raw/_manifest.jsonl``
就能知道每个原始文件的来历。DB 是可以从这些文件重建的索引，不是唯一真相。

INCREMENTAL-ONLY（硬不变式）
----------------------------

实验**没有终态**。判据：不做归档 —— 有的实验可能过了十年重启；
给一个实验写总结、写报告，也不必以归档为前提。

因此：

* ``experiment.json`` 里**没有** ``ended_at``，**没有** ``status: completed|archived``。
  有一条测试专门断言这两个键不存在，防止后来人加回来。
* **禁止**编写 ``on_experiment_end`` / ``on_archive`` / ``finalize_*`` 形式的
  批量收尾函数。所有写入都是增量的：任意时刻拔电源，磁盘上已有的内容必须自洽。
* 所有 ``*.json`` 用 tmp + ``os.replace`` 原子替换（与
  ``webui/settings_store.py::_save`` 同一手法），永远不会读到半个 JSON。

``last_active_at`` 是**描述不是状态**：它记录"上次动过这个实验"，供切换器按
它倒序排 —— 这正是「做了一个月这个又回去做那个」需要的排序。它有 30 秒节流，
否则 2 秒一次的环境采样会把这个文件写成热点。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import itertools

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

#: ``last_active_at`` 的最小重写间隔。2 s 采样 × 多传感器会让这个文件变成
#: 写热点，而它的用途（排序、显示"上次活动"）对 30 秒精度完全无所谓。
_TOUCH_THROTTLE_S = 30.0

_lock = threading.RLock()
_last_touch: dict[str, float] = {}

#: 临时文件名的去重计数器 —— 同一进程同一线程连续两次写同一个文件时，
#: 光靠 pid+tid 还是会同名。
_tmp_counter = itertools.count()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: 原子写的重试次数与退避。`experiment.json` 可能成为并发写入热点，
#: Windows 上 `os.replace` 偶尔撞上杀毒/索引器瞬间持有目标文件而抛
#: `PermissionError(13)`。几十毫秒后就好了，所以短退避重试即可。
_ATOMIC_RETRIES = 4
_ATOMIC_BACKOFF_S = 0.02


def write_json_atomic(path: Path, data: dict) -> bool:
    """用唯一临时文件和 os.replace 完成原子写入；返回成功与否，不抛异常。
    
    临时名包含进程、线程和计数器信息，避免多个写入者互相覆盖同一个临时文件。
    目标被外部程序短暂占用时采用有界短退避重试。失败后清理临时文件，
    不能把失败的状态更新伪装成写入成功。"""
    tmp: Path | None = None
    last_exc: OSError | None = None
    for attempt in range(_ATOMIC_RETRIES):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 唯一临时名 —— 见 docstring 的原因一。
            tmp = path.with_name(
                f"{path.name}.{os.getpid()}.{threading.get_ident()}."
                f"{next(_tmp_counter)}.tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8")
            # 只把**替换**这一步串起来（写 tmp 是各写各的，不必抢锁）。
            # 唯一临时名解决了「两个写入者共用一个 tmp」，但两个线程同时
            # os.replace 同一个**目标**在 Windows 上仍会 PermissionError ——
            # 实测 20 线程 × 20 次并发写，加锁前残留 1 次失败，加锁后为 0。
            # 临界区只有一次 rename，几十微秒；模块里这把锁本来就在
            # （touch_last_active 用它，且先释放再调本函数，不会嵌套）。
            with _lock:
                os.replace(tmp, path)
            return True
        except OSError as exc:
            last_exc = exc
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            if attempt < _ATOMIC_RETRIES - 1:
                time.sleep(_ATOMIC_BACKOFF_S * (2 ** attempt))
    logger.warning("atomic write failed after %d tries (%s): %r",
                   _ATOMIC_RETRIES, path, last_exc)
    return False


def read_json(path: Path) -> dict:
    """读 JSON。损坏/缺失都返回 ``{}`` —— 元数据坏了不该让实验打不开。"""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# ── experiment.json ───────────────────────────────────────────────────

def write_experiment_manifest(
    exp_dir: Path, *,
    experiment_id: str,
    title: str,
    goal: str = "",
    created_at: str = "",
    dir_name: str = "",
    samples: list[dict] | None = None,
    provenance: dict | None = None,
    extra: dict | None = None,
) -> bool:
    """写/更新 ``experiment.json``。保留已有的 ``title_history`` 与 ``activity``。

    注意这个函数**没有** ``ended_at`` / ``status`` 参数 —— 那是刻意的，见模块
    docstring。
    """
    path = Path(exp_dir) / "experiment.json"
    prev = read_json(path)

    history = list(prev.get("title_history") or [])
    prev_title = prev.get("title")
    if prev_title and prev_title != title:
        history.append({"title": prev_title, "changed_at": _now_iso()})

    data = {
        "schema_version": SCHEMA_VERSION,
        "id": experiment_id,
        "title": title,
        "title_history": history,
        "goal": goal or prev.get("goal") or "",
        # dir_name 是创建时冻结的目录名。改名只改 title，目录名不动 ——
        # scan_files 是 append-only，current_path 永远无法 UPDATE，目录一改名
        # 所有历史行就指向不存在的路径。
        "dir_name": dir_name or prev.get("dir_name") or Path(exp_dir).name,
        "created_at": prev.get("created_at") or created_at or _now_iso(),
        "last_active_at": _now_iso(),
        "activity": prev.get("activity") or {
            "first_data_at": None, "last_data_at": None,
            "raw_file_count": 0, "raw_bytes": 0,
        },
        "nanonis": prev.get("nanonis") or {
            "session_path_at_last_check": None,
            "previous_session_path": None,
            "inplace_mode": False,
        },
        "samples": samples if samples is not None else (prev.get("samples") or []),
        "provenance": {**(prev.get("provenance") or {}), **(provenance or {})},
    }
    if prev.get("dir_name_truncated"):
        data["dir_name_truncated"] = True
    if extra:
        data.update(extra)
    return write_json_atomic(path, data)


def touch_last_active(exp_dir: Path, *, force: bool = False) -> None:
    """更新 ``last_active_at``，带 30 秒节流。永不抛。"""
    key = str(exp_dir)
    now = time.monotonic()
    with _lock:
        if not force and now - _last_touch.get(key, 0.0) < _TOUCH_THROTTLE_S:
            return
        _last_touch[key] = now
    path = Path(exp_dir) / "experiment.json"
    data = read_json(path)
    if not data:
        return
    data["last_active_at"] = _now_iso()
    write_json_atomic(path, data)


def bump_activity(exp_dir: Path, *, added_files: int = 0, added_bytes: int = 0) -> None:
    """累加 ``activity`` 观测量（不是状态量：它描述发生过什么，不描述实验处于什么阶段）。"""
    path = Path(exp_dir) / "experiment.json"
    data = read_json(path)
    if not data:
        return
    act = dict(data.get("activity") or {})
    now = _now_iso()
    if added_files:
        act["raw_file_count"] = int(act.get("raw_file_count") or 0) + added_files
        act["last_data_at"] = now
        if not act.get("first_data_at"):
            act["first_data_at"] = now
    if added_bytes:
        act["raw_bytes"] = int(act.get("raw_bytes") or 0) + added_bytes
    data["activity"] = act
    data["last_active_at"] = now
    write_json_atomic(path, data)


def set_nanonis_state(exp_dir: Path, **kw) -> None:
    """记录 Nanonis session path 状态（原位模式的可逆性保证）。

    ``previous_session_path`` 是用户点「把 Nanonis 保存目录指向当前样品」之前
    的原路径 —— UI 靠它提供常驻的「恢复原保存目录」按钮。
    """
    path = Path(exp_dir) / "experiment.json"
    data = read_json(path)
    if not data:
        return
    data["nanonis"] = {**(data.get("nanonis") or {}), **kw}
    write_json_atomic(path, data)


# ── sample.json ───────────────────────────────────────────────────────

def write_sample_manifest(
    sample_path: Path, *,
    sample_id: str,
    name: str,
    experiment_id: str = "",
    description: str = "",
    sample_type: str = "",
    sample_subtype: str = "",
    created_at: str = "",
    dir_name: str = "",
    index: int = 0,
    extra: dict | None = None,
) -> bool:
    """写/更新 ``sample.json``。同样没有终态字段。"""
    path = Path(sample_path) / "sample.json"
    prev = read_json(path)
    data = {
        "schema_version": SCHEMA_VERSION,
        "id": sample_id,
        "experiment_id": experiment_id or prev.get("experiment_id") or "",
        "name": name,
        "index": index or prev.get("index") or 0,
        "dir_name": dir_name or prev.get("dir_name") or Path(sample_path).name,
        "description": description or prev.get("description") or "",
        "sample_type": sample_type or prev.get("sample_type") or "",
        "sample_subtype": sample_subtype or prev.get("sample_subtype") or "",
        "created_at": prev.get("created_at") or created_at or _now_iso(),
        "last_active_at": _now_iso(),
        # env_window 让 UI/导出能按时间窗从实验级 CSV 现切这段温度。
        # ended_at 为 None 表示"还在用" —— 不是状态，是"还没记录过结束时刻"。
        "env_window": prev.get("env_window") or {"started_at": _now_iso(), "ended_at": None},
    }
    if extra:
        data.update(extra)
    return write_json_atomic(path, data)


# ── README.md ─────────────────────────────────────────────────────────

def write_readme(exp_dir: Path) -> bool:
    """从 ``experiment.json`` 生成人读的一页纸。

    第一行就是**当前**名字 —— 目录名是创建时冻结的，可能已经过时，所以人打开
    文件夹时要能立刻看到实验现在叫什么。
    """
    exp_dir = Path(exp_dir)
    d = read_json(exp_dir / "experiment.json")
    if not d:
        return False
    samples = d.get("samples") or []
    act = d.get("activity") or {}
    lines = [
        f"# {d.get('title') or '(未命名实验)'}",
        "",
        f"- 实验 ID：`{d.get('id', '')}`",
        f"- 创建时间：{d.get('created_at', '')}",
        f"- 上次活动：{d.get('last_active_at', '')}",
    ]
    if d.get("goal"):
        lines.append(f"- 目标：{d['goal']}")
    if d.get("dir_name") and d["dir_name"] != d.get("title"):
        lines.append(f"- 文件夹名（创建时冻结，不随改名变化）：`{d['dir_name']}`")
    hist = d.get("title_history") or []
    if hist:
        old = "、".join(str(h.get("title", "")) for h in hist if h.get("title"))
        lines.append(f"- 曾用名：{old}")
    lines += [
        "",
        f"原始数据文件 {act.get('raw_file_count', 0)} 个"
        f"（{_human_bytes(int(act.get('raw_bytes') or 0))}）。",
        "",
        "## 样品",
        "",
    ]
    if samples:
        lines += ["| # | 名称 | 类型 | 文件夹 |", "|---|---|---|---|"]
        for s in samples:
            lines.append(
                f"| {s.get('index', '')} | {s.get('name', '')} | "
                f"{s.get('sample_type', '') or '—'} | `samples/{s.get('dir_name', '')}` |"
            )
    else:
        lines.append("（还没有样品）")
    lines += [
        "",
        "## 文件夹结构",
        "",
        "```",
        "experiment.json            本实验的元数据（机器可读，本文件由它生成）",
        "chats/                     还没选样品时开的对话（规划讨论）",
        "samples/S01__.../",
        "    sample.json            样品元数据",
        "    raw/                   ★ Nanonis 原始文件副本",
        "        _manifest.jsonl      每行一个文件：sha256 / 来源 / 时间 / 关联动作",
        "        sxm/ dat/ 3ds/       按类型分（只含完整、已校验的文件）",
        "        nanonis/             仅当把 Nanonis 保存目录指向了这里时存在",
        "    derived/               分析产物（全部可再生）",
        "    map/                   扫描地图标记",
        "    chats/                 该样品下的对话",
        "    env/                   该样品时段的环境记录",
        "env/                       实验级环境记录（跨样品连续，按天分文件）",
        "reports/                   文献报告 / 实验报告 / 论文草稿 / 评审报告",
        "    _assets/                 本实验的图池（报告里的 ../_assets/x.png）",
        "    <类型>__<日期>__<名字>__<id8>/   一个文档一个目录",
        "        v001.md v002.md      ★ 版本，永不覆盖",
        "        doc.json             文档元数据（标题、归属、关联实验）",
        "        versions.jsonl       每行一个版本：sha256 / 作者 / 时间 / 来自哪次对话",
        "plans/                     实验计划（同上结构）",
        "    plan__.../progress.md    执行进度（视图，会被重写）",
        "    plan__.../progress.jsonl 进度事件（只追加）",
        "library/                   本实验的文献库",
        "    members.jsonl            ★ 书目（只追加：加入/移除/全文就位）",
        "exports/                   按需导出（HTML 报告、记录快照，多份共存）",
        "```",
        "",
        "> 这个实验没有「结束」或「归档」状态：它永远可以继续做。",
        "> 写报告不需要先关闭实验。",
        "",
        "> 报告和计划的历史版本都在各自目录里（`v001.md`、`v002.md` …），",
        "> 永远不会被覆盖。文献库里存的是文献指针，原文在本机的大库中。",
        "",
    ]
    try:
        (exp_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
        return True
    except OSError as exc:
        logger.warning("README write failed: %r", exc)
        return False


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"
