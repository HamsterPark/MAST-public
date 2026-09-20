"""原子写入应为并发写者使用独立临时文件。
固定临时名会使 write 与 replace 竞争同一路径；测试用并发合成写入核验更新完整性。"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

_MASTV2 = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)

from mast.logging.v2.manifest import read_json, write_json_atomic


def test_a_single_write_round_trips_including_non_ascii(tmp_path):
    p = tmp_path / "sub" / "experiment.json"
    assert write_json_atomic(p, {"a": 1, "标题": "中文实验"}) is True
    assert read_json(p) == {"a": 1, "标题": "中文实验"}


def test_no_temp_files_are_left_behind(tmp_path):
    """失败一次留一个 .tmp，实验目录会被残渣糊满。"""
    p = tmp_path / "experiment.json"
    for i in range(5):
        write_json_atomic(p, {"i": i})
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_writers_never_lose_an_update(tmp_path):
    """核心回归：并发写同一个文件，**一次都不许失败**。

    旧实现在这里 400 次失败 202 次（共用临时名 + 无重试 + 无锁）。
    """
    p = tmp_path / "experiment.json"
    failures: list[str] = []

    def w(i: int) -> None:
        for k in range(20):
            try:
                if not write_json_atomic(p, {"writer": i, "k": k}):
                    failures.append(f"{i}/{k} returned False")
            except Exception as exc:  # noqa: BLE001 — 它承诺永不抛
                failures.append(f"{i}/{k} raised {exc!r}")

    ts = [threading.Thread(target=w, args=(i,)) for i in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert not failures, (
        f"{len(failures)}/400 次并发写失败 —— 每一次都是一条被丢掉的状态更新："
        f"{failures[:5]}")
    assert list(tmp_path.glob("*.tmp")) == [], "并发路径留下了临时文件残渣"
    assert read_json(p), "并发写之后文件读不出来了"


def test_never_raises_when_the_path_is_unusable(tmp_path):
    """承诺是「永不抛」。写不进去要返回 False，不是把异常甩给调用方 ——
    调用方全是 fire-and-forget 的记录路径。"""
    victim = tmp_path / "not_a_dir"
    victim.write_text("x", encoding="utf-8")
    assert write_json_atomic(victim / "experiment.json", {"a": 1}) is False


def test_temp_names_are_unique_per_write(tmp_path, monkeypatch):
    """把 os.replace 停在半途，检查两次写用的是**不同**的临时名。

    这是根因本身：同名就会互相踩。
    """
    seen: list[str] = []
    real = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    p = tmp_path / "experiment.json"
    write_json_atomic(p, {"n": 1})
    write_json_atomic(p, {"n": 2})
    assert len(seen) == 2 and seen[0] != seen[1], f"两次写用了同一个临时名: {seen}"
