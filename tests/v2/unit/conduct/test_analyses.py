"""``analysis`` 步能调的纯函数 —— 以及它们**拒绝**的时候。

设计:``campaign_director_design.md`` §4.3/§6-4;不发明坐标那条见
``io/map_analysis.py`` 开头(「never by asking a language model to look at a
picture and guess」)。

这一层最要紧的不是它们算得对,而是**它们不算它们不该算的东西**:点位没给就
拒绝,不替谁挑几个;规划器拒绝就把拒绝原样带出来,不降级成一个空计划。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.conduct import analyses as A


# ── 注册表 ───────────────────────────────────────────────────────────────

def test_the_registry_answers_what_it_has(tmp_path):
    assert {"plan_bias_series", "plan_sts_points"} <= A.known_names()
    assert A.get("plan_bias_series") is A.plan_bias_series


def test_an_unknown_name_raises_and_lists_what_exists():
    with pytest.raises(KeyError) as e:
        A.get("plan_something")
    assert "plan_bias_series" in str(e.value)


def test_registering_a_different_implementation_under_a_taken_name_is_refused():
    """静默覆盖会让两个不同的东西共用一个名字,而调用方拿到哪一个取决于
    import 顺序。"""
    with pytest.raises(ValueError):
        A.register("plan_bias_series", lambda p: {})


# ── 偏压序列 ─────────────────────────────────────────────────────────────

def test_a_bias_series_plan_has_one_frame_per_bias():
    out = A.plan_bias_series({"biases_v": "-1.0,-0.5,0.5,1.0", "size_m": 5e-9})
    plan = json.loads(out["plan_json"])
    assert out["n_frames"] == 4
    assert [f["bias_v"] for f in plan["frames"]] == [-1.0, -0.5, 0.5, 1.0]
    assert all(f["size_m"] == 5e-9 for f in plan["frames"])


def test_the_plan_carries_a_coordinate_epoch_stamp_when_one_is_readable(monkeypatch):
    """修复项 的生产侧要求:conduct L0 步下发坐标时一律带章。

    消费侧 ``ExecuteScanPlan`` 认这个章,粗动之后整批拒绝 —— 没有章,一份粗动
    之前排的计划会照样执行,每一帧都扫在错的地方而且扫得很成功。
    """
    import mast.core.coord_epoch as ce
    monkeypatch.setattr(ce, "read_current_epoch", lambda: 4)
    out = A.plan_bias_series({"biases_v": "0.5", "size_m": 5e-9})
    assert json.loads(out["plan_json"])["coord_epoch"] == 4


def test_an_unreadable_epoch_leaves_the_plan_unstamped_not_zero(monkeypatch):
    """0 是「还没粗动过」这个真实答案,不能拿它冒充「不知道」。"""
    import mast.core.coord_epoch as ce
    monkeypatch.setattr(ce, "read_current_epoch", lambda: None)
    out = A.plan_bias_series({"biases_v": "0.5", "size_m": 5e-9})
    assert "coord_epoch" not in json.loads(out["plan_json"])
    assert out["coord_epoch"] is None


def test_an_empty_bias_list_is_refused_not_invented():
    """规划器不发明序列 —— 不给就拒绝。"""
    with pytest.raises(A.AnalysisError) as e:
        A.plan_bias_series({"biases_v": " , ", "size_m": 5e-9})
    assert "不由这里发明" in str(e.value)


def test_a_missing_parameter_says_which_one_and_where_to_put_it():
    with pytest.raises(A.AnalysisError) as e:
        A.plan_bias_series({"size_m": 5e-9})
    assert "biases_v" in str(e.value) and "不猜数" in str(e.value)


def test_a_non_numeric_bias_is_refused():
    with pytest.raises(A.AnalysisError):
        A.plan_bias_series({"biases_v": "-1.0,abc", "size_m": 5e-9})


def test_a_planner_rejection_comes_out_as_a_rejection_not_an_empty_plan():
    """规划器要么给完整计划,要么给带 code 的拒绝。把拒绝降级成空计划,
    下游就会拿着 0 帧「执行成功」。"""
    with pytest.raises(A.AnalysisError) as e:
        A.plan_bias_series({"biases_v": "0.5", "size_m": 0.0})
    assert "规划器拒绝" in str(e.value)


# ── 取谱点位 ─────────────────────────────────────────────────────────────

def test_points_are_parsed_from_what_the_caller_gave():
    out = A.plan_sts_points({"positions_m": "1e-8,2e-8; -3e-8,0", "n_points": 5})
    assert out["n_points"] == 2
    assert json.loads(out["positions_json"])[0] == {"x_m": 1e-8, "y_m": 2e-8}
    assert out["truncated"] is False


def test_the_cap_truncates_and_says_so():
    """``n_points`` 是**上限**不是目标数:多了截断并说出来,少了不补。"""
    out = A.plan_sts_points({"positions_m": "1e-8,0; 2e-8,0; 3e-8,0",
                             "n_points": 2})
    assert out["n_points"] == 2 and out["truncated"] is True


def test_no_positions_means_refuse_not_invent():
    """让占位实现去挑几个点,是「发明坐标」那条老账的入口。"""
    with pytest.raises(A.AnalysisError) as e:
        A.plan_sts_points({"n_points": 3})
    assert "不猜数" in str(e.value)
    with pytest.raises(A.AnalysisError) as e2:
        A.plan_sts_points({"positions_m": "  ", "n_points": 3})
    assert "不由这里发明" in str(e2.value)


def test_a_point_outside_the_piezo_range_is_refused_not_clamped():
    """超出压电行程不是「远一点」,是根本到不了 —— 拒绝,不夹紧。"""
    with pytest.raises(A.AnalysisError) as e:
        A.plan_sts_points({"positions_m": "5,0", "n_points": 3})
    assert "不夹紧" in str(e.value)
    assert "单位是**米**" in str(e.value)


def test_a_malformed_point_is_refused():
    with pytest.raises(A.AnalysisError):
        A.plan_sts_points({"positions_m": "1e-8,2e-8,3e-8", "n_points": 3})


def test_a_zero_cap_is_refused():
    with pytest.raises(A.AnalysisError):
        A.plan_sts_points({"positions_m": "1e-8,0", "n_points": 0})


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
