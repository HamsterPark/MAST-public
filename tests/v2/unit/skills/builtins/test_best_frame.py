"""最佳帧跨会话保留：更优候选替换记录，低质量候选不得覆盖，并核对图像路径与质量数据一致。"""
from __future__ import annotations

import json

import pytest

from mast.skills.builtins.best_frame import TrackBestFrame


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """别写进真实实验目录 —— 测试污染真实数据这条本仓已经付过五次学费。"""
    import mast.skills.builtins.best_frame as BF

    monkeypatch.setattr(BF, "_store_path",
                        lambda tag: tmp_path / f"{tag}.json")
    return tmp_path


def run(**params):
    return TrackBestFrame().execute(None, params)


def test_the_first_frame_is_always_the_best():
    r = run(tag="t", frame_path="a.sxm", quality=100.0)
    assert r.success and r.data["is_best"] is True
    assert r.data["dry_rounds"] == 0
    assert r.data["best_path"] == "a.sxm"


def test_a_clearly_better_frame_replaces_the_incumbent_and_resets_dry():
    run(tag="t", frame_path="a.sxm", quality=100.0)
    run(tag="t", frame_path="b.sxm", quality=90.0)
    r = run(tag="t", frame_path="c.sxm", quality=400.0)
    assert r.data["is_best"] is True
    assert r.data["best_path"] == "c.sxm"
    assert r.data["dry_rounds"] == 0


def test_noise_sized_gains_do_not_count_as_progress():
    """卡在 1.0 会把判据自己的抖动记成进步 —— 于是永远不收手。"""
    run(tag="t", frame_path="a.sxm", quality=100.0)
    r = run(tag="t", frame_path="b.sxm", quality=103.0)     # +3%，低于 1.05
    assert r.data["is_best"] is False, "3% 的抖动被当成了进步"
    assert r.data["dry_rounds"] == 1
    assert r.data["best_path"] == "a.sxm"


def test_dry_rounds_accumulate_until_the_limit_says_stop():
    run(tag="t", frame_path="a.sxm", quality=500.0)
    for i, q in enumerate((100.0, 200.0, 300.0), start=1):
        r = run(tag="t", frame_path=f"x{i}.sxm", quality=q, dry_limit=3)
        assert r.data["dry_rounds"] == i
    assert r.data["good_enough_to_stop"] is True
    assert "收手" in r.summary


def test_unmeasurable_quality_is_not_a_failed_round():
    """**「判不了」既不算更好也不算更差。** 把它记成「不够好」，
    流程就会在一个读不到数的故障上收手。"""
    run(tag="t", frame_path="a.sxm", quality=500.0)
    run(tag="t", frame_path="b.sxm", quality=100.0)          # dry = 1
    r = run(tag="t", frame_path="c.sxm")                     # quality 缺失
    assert r.success
    assert r.data["is_best"] is None
    assert r.data["recorded"] is False
    assert r.data["dry_rounds"] == 1, "「判不了」推进了 dry 计数"
    assert "判不了" in r.data["note"]


def test_peek_does_not_change_anything():
    run(tag="t", frame_path="a.sxm", quality=500.0)
    run(tag="t", frame_path="b.sxm", quality=10.0)
    before = run(tag="t", peek=True).data
    after = run(tag="t", peek=True).data
    assert before["dry_rounds"] == after["dry_rounds"] == 1
    assert before["n_seen"] == after["n_seen"] == 2
    assert after["recorded"] is False


def test_two_hunts_do_not_pollute_each_other():
    run(tag="hunt-a", frame_path="a.sxm", quality=900.0)
    r = run(tag="hunt-b", frame_path="b.sxm", quality=10.0)
    assert r.data["is_best"] is True, "另一轮追猎的 incumbent 串进来了"


def test_a_missing_tag_is_refused():
    r = run(frame_path="a.sxm", quality=1.0)
    assert r.success is False
    assert "tag" in r.error


def test_a_corrupt_record_is_refused_not_overwritten(_isolated_store):
    """读不懂的历史**不许覆盖** —— 那比没有历史更该让人来看一眼。"""
    (_isolated_store / "t.json").write_text("{ not json", encoding="utf-8")
    r = run(tag="t", frame_path="a.sxm", quality=1.0)
    assert r.success is False
    assert "不覆盖" in r.error
    assert (_isolated_store / "t.json").read_text(encoding="utf-8") == "{ not json"


def test_the_record_is_readable_by_a_human(_isolated_store):
    """几天之后回头，「为什么最后是这一张」必须答得出来。"""
    run(tag="t", frame_path="a.sxm", quality=100.0)
    run(tag="t", frame_path="b.sxm", quality=800.0)
    rec = json.loads((_isolated_store / "t.json").read_text(encoding="utf-8"))
    assert [e["frame_path"] for e in rec["entries"]] == ["a.sxm", "b.sxm"]
    assert rec["best"]["frame_path"] == "b.sxm"
    assert all("quality" in e and "at" in e for e in rec["entries"]), (
        "没被采纳的那些也要留痕，否则说不清最后为什么是这一张")
