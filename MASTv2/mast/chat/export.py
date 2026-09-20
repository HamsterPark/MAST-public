"""把对话增量导出进实验文件夹 —— 一个样品含多组对话，它们得在文件夹里。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §9

为什么是增量导出而不是实时追加
------------------------------

一次 turn 有 30+ 条消息。实时追加等于把 fsync 塞进聊天热路径，触碰「UI 绝不冻结」
这条铁律。而 DB 和 checkpointer 本来就是持久的 —— 文件夹里这份是**投影**，不是
WAL，晚 60 秒完全没有代价。

★ 这不只是复制，它是唯一能保住被裁掉的历史的地方
------------------------------------------------

``ConversationStore.append_message`` 超过 ``_MAX_TRANSCRIPT_ROWS``(8000) 会**裁掉
最旧的行**。导出按 ``seq`` 增量且只追加，所以只要导出跑过一次，那段转录就永久留
在实验文件夹里 —— 哪怕 DB 里已经被裁掉了。一个跑了三个月的群聊，它的开头只在
这里。

崩溃一致性
----------

**先写文件，再更新 state**。顺序反过来会在两者之间崩溃时永久丢掉那一段；
按现在的顺序最坏是重复导出几行，而重复由 ``seq`` 去重消掉。

归属
----

按 ``conversations`` 表的 ``sample_id`` / ``experiment_id`` 分流一次：

* 有 sample_id → ``samples/<S..>/chats/``
* 只有 experiment_id → ``<exp>/chats/``（没选样品时开的规划讨论）
* 两者都没有 → **跳过**（DB 里仍在、UI 照常可见，不丢任何东西）

归属在对话创建时冻结，**不随后续换样品搬家** —— 与 ``chat/store.py`` 的既有契约
一致：``experiment_id``/``sample_id`` 是 "WHERE-IT-WAS-STARTED tags, not a lease"。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_NAME = "chat_state.json"
_MAX_PER_PASS = 5000

#: 文件名里的会话标题片段长度。
_TITLE_SLUG_MAX = 28


def _slugify(text: str, fallback: str = "对话") -> str:
    from mast.core.experiment_paths import slug
    return slug(text or "", fallback, max_chars=_TITLE_SLUG_MAX)


def _state_path(exp_dir: Path) -> Path:
    return Path(exp_dir) / ".mast" / _STATE_NAME


def _load_state(exp_dir: Path) -> dict:
    try:
        return json.loads(_state_path(exp_dir).read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(exp_dir: Path, state: dict) -> None:
    p = _state_path(exp_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        import os
        os.replace(tmp, p)
    except OSError as exc:
        logger.debug("chat_state save failed: %r", exc)


def _dest_dir(exp_dir: Path, sample_dir_name: str | None) -> Path:
    return (Path(exp_dir) / "samples" / sample_dir_name / "chats"
            if sample_dir_name else Path(exp_dir) / "chats")


def _basename(conv: dict) -> str:
    cid = str(conv.get("conversation_id") or "conv")
    title = _slugify(str(conv.get("title") or ""), "对话")
    return f"{cid[:8]}__{title}"


def export_conversation(
    exp_dir: Path,
    store: Any,
    conv: dict,
    *,
    sample_dir_name: str | None = None,
    state: dict | None = None,
) -> int:
    """导出一条对话的新增部分。返回新写入的条数。**绝不抛。**"""
    cid = str(conv.get("conversation_id") or "")
    if not cid:
        return 0
    dest = _dest_dir(exp_dir, sample_dir_name)
    base = _basename(conv)
    own_state = state is None
    st = _load_state(exp_dir) if own_state else state

    # state key 带目标路径：一条对话的归属若被用户更正过，
    # 新位置要从头导一遍，而不是继承旧位置的进度。
    key = f"{cid}|{sample_dir_name or ''}"
    last = int(st.get(key) or 0)

    try:
        rows = store.messages_since(cid, last, limit=_MAX_PER_PASS)
    except Exception as exc:  # noqa: BLE001
        logger.debug("messages_since failed for %s: %r", cid, exc)
        return 0
    if not rows:
        return 0

    try:
        dest.mkdir(parents=True, exist_ok=True)
        # ① 先写文件（追加）。
        with open(dest / f"{base}.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(_jsonl_row(r), ensure_ascii=False) + "\n")
            f.flush()
        _rewrite_markdown(dest / f"{base}.md", conv, dest / f"{base}.jsonl")
    except OSError as exc:
        logger.warning("chat export write failed (%s): %r", dest, exc)
        return 0

    # ② 文件写成功之后才推进 state。反过来会在两者之间崩溃时永久丢掉这一段。
    st[key] = int(rows[-1].get("seq") or last)
    if own_state:
        _save_state(exp_dir, st)
    return len(rows)


def _jsonl_row(r: dict) -> dict:
    """转录行 → 导出行。列与 conversation_messages 一致，可往返。"""
    meta = r.get("meta")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {"raw": meta}
    return {
        "seq": r.get("seq"),
        "t": r.get("t"),
        "role": r.get("role"),
        "agent_id": r.get("agent_id"),
        "kind": r.get("kind"),
        "text": r.get("text"),
        "meta": meta or {},
    }


_ROLE_LABEL = {
    "user": "用户", "operator": "用户",
    "assistant": "助手", "system": "系统", "tool": "工具",
}


def _rewrite_markdown(md_path: Path, conv: dict, jsonl_path: Path) -> None:
    """从 jsonl 全量重渲染 .md（人读的那一半）。

    重渲染而不是追加：markdown 有标题和结构，追加会把它写成一堆碎片。jsonl 才是
    增量的载体，md 是它的视图。
    """
    lines = [
        f"# {conv.get('title') or '(未命名对话)'}",
        "",
        f"- 会话 ID：`{conv.get('conversation_id', '')}`",
        f"- 类型：{'群聊' if conv.get('kind') == 'group' else '私聊'}"
        f"（{conv.get('agent_id', '')}）",
        f"- 创建：{conv.get('created_at', '')}",
        f"- 最后更新：{conv.get('updated_at', '')}",
        "",
        "---",
        "",
    ]
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # 崩溃留下的半行 —— 丢弃，不是错误
                lines.extend(_md_entry(r))
    except OSError:
        return
    try:
        md_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        logger.debug("markdown render failed: %r", exc)


def _md_entry(r: dict) -> list[str]:
    kind = str(r.get("kind") or "message")
    text = str(r.get("text") or "").strip()
    who = _ROLE_LABEL.get(str(r.get("role") or ""), str(r.get("role") or ""))
    agent = str(r.get("agent_id") or "")
    ts = _fmt_t(r.get("t"))

    if kind == "status":
        return [f"> _{text}_", ""] if text else []
    if kind == "interrupt":
        meta = r.get("meta") or {}
        return [
            f"> ⚠ **人工审批** · {meta.get('skill') or meta.get('interrupt_kind') or ''}",
            f"> {text}" if text else "",
            "",
        ]
    if kind == "compaction":
        return [f"> ✂ _上下文压缩：{text}_", ""]

    head = f"**{who}**" + (f" · {agent}" if agent and who == "助手" else "")
    out = [f"### {head}{f'  <sub>{ts}</sub>' if ts else ''}", ""]
    out.append(text if text else "_(空)_")
    out.append("")
    meta = r.get("meta") or {}
    tools = meta.get("tool_calls") or meta.get("tools")
    if tools:
        out += ["<details><summary>工具调用</summary>", "",
                "```json",
                json.dumps(tools, ensure_ascii=False, indent=2)[:4000],
                "```", "", "</details>", ""]
    return out


def _fmt_t(t: Any) -> str:
    try:
        return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return ""


def export_conversations(
    exp_dir: Path,
    store: Any,
    *,
    experiment_id: str,
    sample_dir_resolver: Any = None,
    limit: int = 200,
) -> dict:
    """导出该实验下所有对话的新增部分。

    ``sample_dir_resolver(sample_id) -> str | None`` 把 sample_id 翻成样品目录名；
    返回 None 时该对话落到实验级 ``chats/``。

    返回 ``{"conversations": n, "entries": m, "skipped": k}``。**绝不抛。**
    """
    out = {"conversations": 0, "entries": 0, "skipped": 0}
    if not experiment_id:
        return out
    try:
        convs = store.list(experiment_id=experiment_id, limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.debug("conversation list failed: %r", exc)
        return out

    state = _load_state(exp_dir)
    for conv in convs or []:
        sid = conv.get("sample_id")
        sample_dir = None
        if sid and sample_dir_resolver is not None:
            try:
                sample_dir = sample_dir_resolver(sid)
            except Exception:  # noqa: BLE001
                sample_dir = None
        try:
            n = export_conversation(exp_dir, store, conv,
                                    sample_dir_name=sample_dir, state=state)
        except Exception as exc:  # noqa: BLE001 — 一条对话失败不能带走其它的
            logger.debug("export failed for %s: %r", conv.get("conversation_id"), exc)
            out["skipped"] += 1
            continue
        if n:
            out["conversations"] += 1
            out["entries"] += n
    _save_state(exp_dir, state)
    return out
