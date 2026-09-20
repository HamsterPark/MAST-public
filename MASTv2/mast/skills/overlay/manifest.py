"""``overlay.json`` —— **显式启用**清单。

丢一个 ``.py`` 进覆盖层目录**不会**被执行。启用是用户的一次显式动作，这条纪律
沿用 ``skills/custom_loader.py:6-8`` 已经确立的做法，理由也一样：一个忘了删的实验
文件、一个从别处拷来的半成品，都不该因为「在目录里」就跑起来。

格式::

    {
      "schema": 1,
      "entries": [
        {"path": "builtins/bias.py", "enabled": true},
        {"path": "builtins/scan.py", "enabled": true,
         "allow_removals": ["ScanAreaLegacy"]},
        {"path": "_new/my_thing.py", "enabled": false}
      ]
    }

写盘走**原子替换**（``.part`` + ``os.replace``）—— 任意时刻拔电源，盘上的内容
必须自洽（``docs/v2/architecture/v2.md:147-162`` 第 6 条）。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from mast.skills.overlay import paths as P

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    path: str
    enabled: bool = False
    allow_removals: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        d: dict = {"path": self.path, "enabled": bool(self.enabled)}
        if self.allow_removals:
            d["allow_removals"] = list(self.allow_removals)
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class Manifest:
    entries: list[Entry] = field(default_factory=list)
    unreadable: str = ""       # 读失败的原因（**不**当成空清单，见 load）
    #: 签名包装进来之后自动启用。默认开 —— 推一个修复过去却要人跑到每台机器上
    #: 再点一次「启用」，那条通道就没有意义了。
    #:
    #: 它和「丢文件进目录不会自动生效」那条纪律**不矛盾**，因为门不一样：
    #: 松散文件的门是用户的一次显式动作，签名包的门是密码学事实（验签 +
    #: 逐文件 sha256 + 加载时再复核）。前者没有门，后者有一道更硬的。
    #:
    #: 每台机器可以单独关掉（写 false）—— 有些机器正在跑长实验，用户要自己
    #: 决定什么时候换。关掉之后包照样装进 _packs/，只是不自动登记进清单。
    auto_enable_packs: bool = True

    def enabled(self) -> list[Entry]:
        return [e for e in self.entries if e.enabled]

    def get(self, rel: str) -> Entry | None:
        rel = P.normalise_rel(rel)
        for e in self.entries:
            if e.path == rel:
                return e
        return None

    def upsert(self, entry: Entry) -> None:
        entry.path = P.normalise_rel(entry.path)
        for i, e in enumerate(self.entries):
            if e.path == entry.path:
                self.entries[i] = entry
                return
        self.entries.append(entry)


def load() -> Manifest:
    """读清单。文件不存在 = 空清单（正常）；**读不出来 = 记下原因**。

    这两件事必须分开：不存在意味着「还没人用覆盖层」，读不出来意味着
    「有人配了但我们没看懂」—— 后者当成空清单处理，就会把用户显式启用的东西
    静默地全部停掉，而界面上什么都不会说。
    """
    p = P.manifest_path()
    if not p.is_file():
        return Manifest()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.error("覆盖层清单读不出来（%s）：%s —— 本次不加载任何覆盖，"
                     "已启用的条目**没有**被停用，修好文件再重载即可", p, exc)
        return Manifest(unreadable=f"{type(exc).__name__}: {exc}")

    items = raw.get("entries") if isinstance(raw, dict) else raw
    out: list[Entry] = []
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        rel = P.normalise_rel(str(it.get("path") or ""))
        if not rel:
            continue
        out.append(Entry(
            path=rel,
            enabled=bool(it.get("enabled")),
            allow_removals=[str(x) for x in (it.get("allow_removals") or [])],
            note=str(it.get("note") or ""),
        ))
    # 缺省 True。**只有显式写 false 才关** —— 把「字段不存在」读成 False，
    # 会让每一台还没写过这个字段的机器（也就是全部）都悄悄关掉自动启用。
    auto = raw.get("auto_enable_packs", True)
    return Manifest(entries=out, auto_enable_packs=bool(auto) if auto is not None else True)


def atomic_write_json(p: Path, doc) -> Path:
    """写 JSON，落盘是原子的。

    ``.part`` + :func:`os.replace` —— 半个文件是本仓反复被咬的形状，而这里的读者
    （UI 与加载器）随时可能在写到一半时来读。清单和 eject 的 sidecar 共用这一份，
    不各写各的。
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


def save(man: Manifest) -> Path:
    doc = {"schema": 1, "entries": [e.to_dict() for e in man.entries]}
    # 非默认值才写出来，免得每台机器的 overlay.json 都多一行噪声。
    # 但**一定要写**：漏了它，用户关掉自动启用之后第一次改清单就被悄悄打开。
    if not man.auto_enable_packs:
        doc["auto_enable_packs"] = False
    return atomic_write_json(P.manifest_path(), doc)


def discover_files() -> list[str]:
    """覆盖层目录里**所有** ``.py`` 的相对路径（不管启没启用）。

    给 UI 用：让用户看得见「目录里有什么」和「哪些启用了」的差别 ——
    一个拷进来却没启用的文件，不说的话会被当成「怎么没生效」。
    """
    root = P.overlay_dir()
    if not root.is_dir():
        return []
    out: list[str] = []
    for f in sorted(root.rglob("*.py")):
        rel = P.normalise_rel(str(f.relative_to(root)))
        ok, _why = P.is_valid_rel(rel)
        if ok:
            out.append(rel)
    return out


__all__ = ["Entry", "Manifest", "discover_files", "load", "save"]
