"""外部 agent 网关测试的共用零件（探针技能、runtime 替身、假池、假状态、小工具）。

夹具本身在同目录的 conftest.py；两边都 import 这个模块，保证是同一批对象。

一个「世界」= 真的 ``ExperimentStorage`` + 真的 ``ExperimentLog``（注册成活动日志）+
真的 v2 记录库 + 一个装着探针技能的真注册表 + 假连接池 + 假仪器状态 + 一个由
``CoreRuntime`` 真方法拼成的 runtime 替身，全部落在 tmp 目录。

隔离（``tests/v2/conftest.py`` 里有「真实数据只读」的 autouse 守卫，漏一个就红）：
心愿单（``MAST2_PROJECT_ROOT`` + ``reset_default_board``）、实验根目录与文档库
（``documents_root``）、自定义技能目录（``skill_author._CUSTOM_SKILLS_DIR``）、组合技能
版本库（``composite_panel._store``）、作业 journal（显式目录）、运行模式（测试后解绑）。
语义记忆索引关掉（它会去探测 DashScope）—— 召回退回子串检索。
"""
from __future__ import annotations

import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.core.registry import SkillRegistry  # noqa: E402
from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.types import (  # noqa: E402
    HardwareState,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# 探针技能
# ─────────────────────────────────────────────────────────────────────

RAN: list[tuple[str, dict]] = []
GATES: dict[str, threading.Event] = {}


def _p(name, type_="float", unit="", **kw):
    return ParameterSpec(name=name, type=type_, unit=unit, required=False,
                         description=kw.pop("description", name), **kw)


class ExtRead(BaseSkill):
    def metadata(self):
        return SkillMetadata(name="ExtRead", version="1.0.0", category=SkillCategory.READ,
                             safety_level=SafetyLevel.AUTO, description="读一个数",
                             parameters=[_p("x_m", unit="m", min_value=-1e-6, max_value=1e-6)],
                             tags=["read"], estimated_duration_s=0.1)

    def execute(self, context, params):
        RAN.append(("ExtRead", dict(params)))
        return SkillResult(skill_name="ExtRead", success=True, data={"value": 42},
                           summary="ok")


class ExtSlow(BaseSkill):
    """慢技能：等 ``GATES['slow']`` 或被中止。只读 → 不取仪器令牌。"""

    def metadata(self):
        return SkillMetadata(name="ExtSlow", version="1.0.0", category=SkillCategory.READ,
                             safety_level=SafetyLevel.AUTO, description="慢", tags=["read"])

    def execute(self, context, params):
        RAN.append(("ExtSlow", dict(params)))
        gate = GATES.setdefault("slow", threading.Event())
        for _ in range(400):
            if gate.wait(0.05):
                return SkillResult(skill_name="ExtSlow", success=True, summary="released")
            if context.check_abort():
                return SkillResult(skill_name="ExtSlow", success=False,
                                   error="aborted: " + (context.abort_reason() or ""))
        return SkillResult(skill_name="ExtSlow", success=False, error="timeout")


class ExtWrite(BaseSkill):
    """写技能（取仪器令牌）。"""

    def metadata(self):
        return SkillMetadata(name="ExtWrite", version="1.0.0", category=SkillCategory.WRITE,
                             safety_level=SafetyLevel.AUTO, description="写一个设置",
                             parameters=[], tags=["setting"])

    def execute(self, context, params):
        RAN.append(("ExtWrite", dict(params)))
        return SkillResult(skill_name="ExtWrite", success=True)


class ExtPulse(BaseSkill):
    """电脉冲：SAFE 模式下必须在执行前被拒。"""

    def metadata(self):
        return SkillMetadata(name="ExtPulse", version="1.0.0", category=SkillCategory.WRITE,
                             safety_level=SafetyLevel.CONFIRM, description="打一发脉冲",
                             parameters=[_p("pulse_v", unit="V", min_value=-10.0, max_value=10.0)],
                             tags=["pulse"], capabilities=frozenset({"bias_pulse"}))

    def execute(self, context, params):
        RAN.append(("ExtPulse", dict(params)))
        return SkillResult(skill_name="ExtPulse", success=True)


class ExtBoom(BaseSkill):
    def metadata(self):
        return SkillMetadata(name="ExtBoom", version="1.0.0", category=SkillCategory.READ,
                             safety_level=SafetyLevel.AUTO, description="会抛", tags=["read"])

    def execute(self, context, params):
        raise RuntimeError("probe exploded")


class ExtStopScan(BaseSkill):
    """名字就是 ``StopScan``：停止类补救技能（仪器锁对它放行，作业上限也要对它放行）。"""

    def metadata(self):
        return SkillMetadata(name="StopScan", version="1.0.0", category=SkillCategory.WRITE,
                             safety_level=SafetyLevel.AUTO, description="停止扫描", tags=["stop"])

    def execute(self, context, params):
        RAN.append(("StopScan", dict(params)))
        return SkillResult(skill_name="StopScan", success=True)


PROBES = (ExtRead, ExtSlow, ExtWrite, ExtPulse, ExtBoom, ExtStopScan)


# ─────────────────────────────────────────────────────────────────────
# runtime 替身：借 CoreRuntime 的真方法
# ─────────────────────────────────────────────────────────────────────

class RT:
    _record_direct_skill_call = CoreRuntime._record_direct_skill_call
    _record_v1_action = CoreRuntime._record_v1_action
    _record_v2_action = CoreRuntime._record_v2_action
    _note_skill_activity = CoreRuntime._note_skill_activity
    emergency_latch_state = CoreRuntime.emergency_latch_state
    add_shutdown_hook = CoreRuntime.add_shutdown_hook

    def __init__(self, storage, log, v2, cognition):
        self._storage = storage
        self._experiment_log = log
        self._v2_repos, self._v2_eid = v2
        self._active_thread_id = "agents-some-chat"
        self._map_last_skill_ts = 0.0
        self._orch_abort = threading.Event()
        self._orch_abort_emergency = False
        self._orch_abort_why = ""
        self._cognition = cognition
        self._env_alarm_log = []
        self._shutdown_hooks = []
        self.markers: list[dict] = []
        self.estops: list[str] = []
        self.session_dir_asked = 0

    def _record_map_marker(self, payload):
        self.markers.append(dict(payload))

    def _resolve_session_dir(self):
        """真 runtime 上这是一条 ``Util_SessionPathGet`` TCP 命令 —— 简报不许调它。"""
        self.session_dir_asked += 1
        return None

    def emergency_stop(self, why: str = ""):
        from mast.core.execution_context import mark_abort

        self.estops.append(why)
        mark_abort(self._orch_abort, why or "用户按下了急停(E-STOP)")
        self._orch_abort_emergency = True
        self._orch_abort_why = why
        return {"aborted": True, "stopped_motion": True, "retracted": True, "errors": []}


class Pool:
    """假连接池：``get`` 说 main 连着；任何 TCP 命令都算失败（简报不许发命令）。"""

    def __init__(self):
        self.calls: list[str] = []

    def get(self, role):
        if role != "main":
            raise ConnectionError(f"{role} not connected")
        return object()

    def safe_call(self, method, *args, **kw):  # pragma: no cover — 网关自己不该发命令
        self.calls.append(method)
        raise AssertionError(f"外部面发了 Nanonis 命令 {method}")


class State:
    def __init__(self):
        self.refreshed = 0

    def snapshot(self):
        return HardwareState(bias_v=-0.2, current_a=5e-10, setpoint_a=5e-10,
                             z_controller_on=True, scan_running=False)

    def refresh(self):
        self.refreshed += 1
        raise AssertionError("外部面调了 state.refresh()（真硬件 I/O）")

    def history(self, channel):
        return []


def H(actor="tester", session="s1"):
    return {"X-MAST-Actor": actor, "X-MAST-Session": session}


def wait_terminal(client, job_id, *, actor="tester", total_s=20.0):
    import time

    deadline = time.monotonic() + total_s
    while True:
        v = client.get(f"/jobs/{job_id}", params={"wait_s": 2}, headers=H(actor)).json()
        if v.get("terminal") or time.monotonic() > deadline:
            return v


def write_sxm(path: Path, fwd: np.ndarray, bwd: np.ndarray, *, scan_dir: str = "down",
              range_m: float = 5e-9) -> Path:
    """最小但真实的 ``.sxm``：一个 Z 通道，正反两块（文件里按**采集顺序**存）。"""
    ny, nx = fwd.shape
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
        ":REC_DATE:\n 14.08.2026\n"
        ":REC_TIME:\n12:00:00\n"
        ":BIAS:\n\t-1.200000E+0\n"
        f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
        f":SCAN_OFFSET:\n{0.0:>19.6E}{0.0:>19.6E}\n"
        f":SCAN_RANGE:\n{range_m:>19.6E}{range_m:>19.6E}\n"
        ":SCAN_ANGLE:\n\t0.000E+0\n"
        f":SCAN_DIR:\n{scan_dir}\n"
        ":Z-CONTROLLER>SETPOINT:\n50.0000E-12\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t9.000E-9\t0.000E+0\n"
        ":SCANIT_END:\n\n"
    )
    arr = np.concatenate([fwd.astype(np.float32).ravel(), bwd.astype(np.float32).ravel()])
    blob = struct.pack(">%df" % arr.size, *arr.tolist())
    path.write_bytes(header.encode("utf-8") + b"\x1a\x04" + blob)
    return path
