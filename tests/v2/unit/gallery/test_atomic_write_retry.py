# -*- coding: utf-8 -*-
"""原子写入在 Windows「目标正在被读」时的行为（``mast.gallery.paths.atomic_write_bytes``）。

Python 的 ``open()`` 与 Starlette 的 ``FileResponse`` 打开文件都不带 ``FILE_SHARE_DELETE``，
所以 ``GET /marks`` 恰好在读 marks.json 的那几毫秒里，``POST /marks/patch`` 的
``os.replace`` 会抛 ``PermissionError``。短暂占用要靠重试扛过去；一直占着（Excel 打开的
csv）才按 ``tolerate_locked`` 处理；``marks.json`` 这种不许容忍的，必须抛出来。
"""
from __future__ import annotations

import os

import pytest

from mast.gallery import paths


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(paths.time, "sleep", lambda _s: None)


def _flaky_replace(monkeypatch, fail_times: int):
    real = os.replace
    calls = {"n": 0}

    def fake(src, dst):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise PermissionError(13, "the file is being used by another process")
        return real(src, dst)

    monkeypatch.setattr(paths.os, "replace", fake)
    return calls


def _leftover_tmp(directory) -> list[str]:
    return [f for f in os.listdir(directory) if paths.TMP_MARK in f]


def test_brief_lock_is_ridden_out(tmp_path, monkeypatch):
    target = tmp_path / "marks.json"
    target.write_text("old", encoding="utf-8")
    calls = _flaky_replace(monkeypatch, fail_times=3)

    assert paths.atomic_write_bytes(target, b"new") is True
    assert target.read_bytes() == b"new"
    assert calls["n"] == 4
    assert _leftover_tmp(tmp_path) == []


def test_persistent_lock_raises_for_the_source_of_truth(tmp_path, monkeypatch):
    target = tmp_path / "marks.json"
    target.write_text("old", encoding="utf-8")
    calls = _flaky_replace(monkeypatch, fail_times=10_000)

    with pytest.raises(PermissionError):
        paths.atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old", "a failed replace must leave the old document intact"
    assert calls["n"] == len(paths._REPLACE_RETRY_S) + 1
    assert _leftover_tmp(tmp_path) == []


def test_persistent_lock_is_tolerated_for_derived_files(tmp_path, monkeypatch):
    target = tmp_path / "marks.csv"
    target.write_bytes(b"old")
    _flaky_replace(monkeypatch, fail_times=10_000)

    assert paths.atomic_write_bytes(target, b"new", tolerate_locked=True) is False
    assert target.read_bytes() == b"old"
    assert _leftover_tmp(tmp_path) == []


@pytest.mark.skipif(os.name != "nt", reason="只有 Windows 上一个读句柄会挡住 os.replace")
def test_a_real_reader_handle_blocks_the_replace_until_it_closes(tmp_path, monkeypatch):
    """上面几条给 ``os.replace`` 打了桩，证明不了锁是真的。这条用一个真实的读句柄占住目标：
    第一次替换必须真的失败（于是进入等待），读者在等待期间读完关闭，下一次替换成功。"""
    target = tmp_path / "marks.json"
    target.write_bytes(b"old")
    reader = open(target, "rb")
    waits: list[float] = []

    def reader_finishes_while_we_wait(seconds):
        waits.append(seconds)
        reader.close()

    monkeypatch.setattr(paths.time, "sleep", reader_finishes_while_we_wait)
    try:
        assert paths.atomic_write_bytes(target, b"new") is True
    finally:
        reader.close()
    assert len(waits) == 1, "the open reader must have made the first replace fail for real"
    assert target.read_bytes() == b"new"
    assert _leftover_tmp(tmp_path) == []
