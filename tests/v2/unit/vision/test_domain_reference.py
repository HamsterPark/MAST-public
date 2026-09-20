"""畴参照系缺失时应明确返回无参照，而不抛异常或虚构默认参照。
测试将 MAST2_PROJECT_ROOT 重定向到临时目录，核验实际读取的存储根。"""

from __future__ import annotations

import json

import pytest

from mast.vision import domain_phase as D
from mast.vision import domain_reference as R

GOOD = {
    "schema": 1,
    "version": "v001",
    "sample": "synthetic",
    "created": "2000-08-14",
    "provenance": "2000-08-14 合成语料;本文件由测试生成,不代表任何真实样品。",
    "confirmed_by": "test",
    "symmetry_deg": 60.0,
    "labels": ["A", "B"],
    "prototypes": {
        "A": {"peaks": [[17.0, 0.2494, 1.0], [77.0, 0.2494, 0.98]],
              "source_frames": ["/tmp/a.sxm"]},
        "B": {"peaks": [[42.0, 0.2494, 1.0], [102.0, 0.2494, 0.97]]},
    },
    "match": {"w_angle": 1.0, "w_period": 1.0, "match_tol": 0.06,
              "ambiguity_margin": 0.03, "mixed_coverage_min": 0.8},
    "measured_separation": {"intra_max": 0.04, "inter_min": 0.17, "ratio": 4.2,
                            "n_frames_per_cluster": 5},
}


@pytest.fixture
def refs_dir(tmp_path, monkeypatch):
    """把参照系目录整个重定向到 ``tmp_path``,并清掉 mtime 缓存。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    R.clear_cache()
    d = tmp_path / "config" / R.DIRNAME
    d.mkdir(parents=True)
    yield d
    R.clear_cache()


def write(d, body, name="synthetic-2000-08-14-v001.json"):
    (d / name).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    R.clear_cache()
    return d / name


# ── 1. 没有参照系 ───────────────────────────────────────────────────────────

def test_no_directory_means_no_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "nowhere"))
    R.clear_cache()
    assert R.load_reference() is None
    assert R.list_references() == ()


def test_empty_directory_means_no_reference(refs_dir):
    assert R.load_reference() is None
    assert R.list_references() == ()


def test_no_reference_yields_no_label(refs_dir):
    """R6 的通过判据:没有参照系时判定必须是 ``undetermined(no_reference)``。"""
    fp = D.DomainFingerprint(
        peaks=(D.DomainPeak(17.0, 17.0, 0.2494, 1.0),), n_peaks=1,
        scan_angle_deg=0.0, nm_per_px=0.0195, scale="full")
    assert fp.comparable
    v = D.classify(fp, R.load_reference())
    assert (v.verdict, v.reason, v.label) == ("undetermined", "no_reference", None)


# ── 2. 好文件 ───────────────────────────────────────────────────────────────

def test_loads_a_valid_reference(refs_dir):
    write(refs_dir, GOOD)
    ref = R.load_reference()
    assert ref is not None
    assert ref.version == "v001" and ref.sample == "synthetic"
    assert ref.labels == ("A", "B")
    assert ref.symmetry_deg == 60.0
    assert ref.calibrated
    assert len(ref.peaks_of("A")) == 2
    assert ref.peaks_of("A")[0] == (17.0, 0.2494, 1.0)
    assert ref.peaks_of("nope") == ()
    assert ref.measured_separation["ratio"] == 4.2
    assert ref.source_path.endswith(".json")


def test_angles_are_folded_mod_180(refs_dir):
    body = json.loads(json.dumps(GOOD))
    body["prototypes"]["A"]["peaks"] = [[197.0, 0.2494, 1.0]]
    write(refs_dir, body)
    assert R.load_reference().peaks_of("A")[0][0] == pytest.approx(17.0)


# ── 3. 未标定 ≠ 零容差 ══════════════════════════════════════════════════════

@pytest.mark.parametrize("key", ["match_tol", "ambiguity_margin",
                                 "mixed_coverage_min"])
def test_zero_match_parameter_means_uncalibrated(refs_dir, key):
    """``match.*`` 留 0 表示**未标定**,不许当「零容差」用。"""
    body = json.loads(json.dumps(GOOD))
    body["match"][key] = 0.0
    write(refs_dir, body)
    ref = R.load_reference()
    assert ref is not None, "未标定的参照系仍然要读得出来(好报清楚缺什么)"
    assert not ref.calibrated
    fp = D.DomainFingerprint(
        peaks=(D.DomainPeak(17.0, 17.0, 0.2494, 1.0),), n_peaks=1,
        scan_angle_deg=0.0, nm_per_px=0.0195, scale="full")
    assert D.classify(fp, ref).reason == "no_reference"


def test_negative_match_parameter_is_rejected(refs_dir):
    body = json.loads(json.dumps(GOOD))
    body["match"]["match_tol"] = -1.0
    write(refs_dir, body)
    assert R.load_reference() is None


# ── 4. 坏文件 → 「没有参照系」,不抛异常 ═══════════════════════════════════

@pytest.mark.parametrize("mutate,why", [
    (lambda b: b.__setitem__("schema", 2), "schema 不认识"),
    (lambda b: b.__setitem__("labels", []), "labels 空"),
    (lambda b: b.__setitem__("labels", ["A", "A"]), "labels 重名"),
    (lambda b: b.__setitem__("symmetry_deg", 0.0), "symmetry 为 0"),
    (lambda b: b.__setitem__("symmetry_deg", -60.0), "symmetry 为负"),
    (lambda b: b.__setitem__("provenance", ""), "provenance 空"),
    (lambda b: b.__setitem__("prototypes", {}), "没有原型"),
    (lambda b: b["prototypes"]["A"].__setitem__("peaks", []), "原型没有峰"),
    (lambda b: b["prototypes"]["A"].__setitem__("peaks", [[17.0, 0.0, 1.0]]),
     "周期为 0"),
    (lambda b: b["prototypes"]["A"].__setitem__("peaks", [[17.0, 0.25]]),
     "峰少一列"),
    (lambda b: b["prototypes"]["A"].__setitem__("peaks", "nope"), "峰不是数组"),
    (lambda b: b.__setitem__("match", "nope"), "match 不是对象"),
])
def test_broken_reference_is_no_reference_not_an_exception(refs_dir, mutate, why):
    body = json.loads(json.dumps(GOOD))
    mutate(body)
    write(refs_dir, body)
    assert R.load_reference() is None, why
    assert R.list_references() == ()


def test_unparseable_json_is_no_reference(refs_dir):
    (refs_dir / "broken-v001.json").write_text("{not json", encoding="utf-8")
    R.clear_cache()
    assert R.load_reference() is None


def test_one_broken_file_does_not_hide_a_good_one(refs_dir):
    (refs_dir / "broken-v009.json").write_text("{not json", encoding="utf-8")
    write(refs_dir, GOOD)
    ref = R.load_reference()
    assert ref is not None and ref.version == "v001"


# ── 5. 版本与样品 ═══════════════════════════════════════════════════════════

def test_latest_version_wins(refs_dir):
    write(refs_dir, GOOD)
    v3 = json.loads(json.dumps(GOOD))
    v3["version"] = "v003"
    v3["match"]["match_tol"] = 0.09
    write(refs_dir, v3, name="synthetic-2000-08-20-v003.json")
    v2 = json.loads(json.dumps(GOOD))
    v2["version"] = "v002"
    write(refs_dir, v2, name="synthetic-2000-08-16-v002.json")
    assert [r.version for r in R.list_references()] == ["v003", "v002", "v001"]
    assert R.load_reference().version == "v003"
    assert R.load_reference().match_tol == 0.09


def test_explicit_version_never_falls_back(refs_dir):
    """指定版本找不到 ⇒ ``None``,**不退到最新**。

    判定结果里记着 ``reference_version``,悄悄换一版会让事后对账对不上。
    """
    write(refs_dir, GOOD)
    assert R.load_reference("v001") is not None
    assert R.load_reference("v999") is None


def test_multiple_samples_without_an_explicit_choice_is_no_reference(refs_dir):
    """两个样品的参照系并存时**不猜** —— 挑错样品会让判定看起来一切正常。"""
    write(refs_dir, GOOD)
    other = json.loads(json.dumps(GOOD))
    other["sample"] = "other-sample"
    other["version"] = "v007"
    write(refs_dir, other, name="other-sample-2000-08-14-v007.json")
    assert R.load_reference() is None
    assert R.load_reference(sample="synthetic").version == "v001"
    assert R.load_reference(sample="other-sample").version == "v007"
    assert R.load_reference(sample="nobody") is None


# ── 6. 缓存与只读 ═══════════════════════════════════════════════════════════

def test_file_change_is_picked_up(refs_dir):
    write(refs_dir, GOOD)
    assert R.load_reference().match_tol == 0.06
    changed = json.loads(json.dumps(GOOD))
    changed["match"]["match_tol"] = 0.123456789
    (refs_dir / "synthetic-2000-08-14-v001.json").write_text(
        json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    R.clear_cache()          # 同一秒内改文件时 mtime 可能不变
    assert R.load_reference().match_tol == 0.123456789


def test_cache_returns_the_same_object(refs_dir):
    write(refs_dir, GOOD)
    assert R.list_references() is R.list_references()


def test_loader_never_writes_anything(refs_dir):
    """只读:加载器不许在目录里留下任何东西(连空文件都不许)。"""
    write(refs_dir, GOOD)
    before = sorted(p.name for p in refs_dir.iterdir())
    R.load_reference()
    R.load_reference("v001")
    R.load_reference(sample="nobody")
    R.list_references()
    assert sorted(p.name for p in refs_dir.iterdir()) == before


def test_reference_from_mapping_is_pure(refs_dir):
    """纯函数入口:校验只写一遍,写入方(R7 固化那一步)将来复用同一个。"""
    assert R.reference_from_mapping(GOOD) is not None
    assert R.reference_from_mapping({"schema": 1}) is None
    assert R.reference_from_mapping(None) is None
    assert R.reference_from_mapping("nope") is None
    assert R.reference_from_mapping(GOOD, source="x").source_path == "x"
