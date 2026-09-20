"""Auto-generated utterances for the IC-reachable skills not hand-authored in
skill_utterances.py (the deep Nanonis config getters/setters — PLL / lock-in /
pattern / atom-track / spectroscopy-config / piezo / signals).

Templated from each skill's Chinese description (mast/webui/skill_zh.json) for 中文
and its English metadata description for English, so EVERY instrument_control
builtins+composite skill gets a zh+en utterance — without 130 hand-typed mechanical
lines. Hand-authored entries in HAND always take precedence (the harness merges with
setdefault).
"""
from __future__ import annotations

import json
from pathlib import Path

_ZH_PATH = (
    Path(__file__).resolve().parents[3] / "MASTv2" / "mast" / "webui" / "skill_zh.json"
)


def _load_zh() -> dict:
    try:
        return json.loads(_ZH_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _en_request(desc: str, name: str) -> str:
    d = (desc or "").replace("\n", " ").split(". ")[0].strip().rstrip(".")
    if not d:
        return f"Please run {name}."
    return f"Please {d[0].lower() + d[1:]}."


def _zh_request(desc_zh: str, name: str) -> str:
    d = (desc_zh or "").replace("\n", "").split("。")[0].strip()
    return f"请帮我{d}。" if d else f"请执行 {name}。"


def auto_utterances(registry) -> dict[str, dict[str, str]]:
    """{skill_name: {"zh","en"}} for IC-reachable (builtins+composite) skills."""
    out: dict[str, dict[str, str]] = {}
    if registry is None:
        return out
    try:
        from skill_utterances import HAND  # type: ignore
    except Exception:  # noqa: BLE001
        HAND = {}

    zh_map = _load_zh()
    try:
        skills = registry.list_skills()
    except Exception:  # noqa: BLE001
        return out

    for entry in skills:
        name = getattr(entry, "name", entry if isinstance(entry, str) else None)
        if not name or name in HAND:
            continue
        try:
            cls = registry.get(name)
            mod = getattr(cls, "__module__", "") or ""
        except Exception:  # noqa: BLE001
            continue
        if ".skills.builtins." not in mod and ".skills.composite." not in mod:
            continue  # only IC-reachable (paper BaseSkills aren't agent-callable)
        try:
            md = cls().metadata()
            en_desc = md.description or ""
        except Exception:  # noqa: BLE001
            en_desc = ""
        zh_desc = zh_map.get(name, {}).get("description_zh", "")
        out[name] = {
            "zh": _zh_request(zh_desc, name),
            "en": _en_request(en_desc, name),
        }
    return out


__all__ = ["auto_utterances"]
