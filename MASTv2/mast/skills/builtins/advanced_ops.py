"""Advanced operations: quit Nanonis, multi-pass config files, blocking scan wait.

Three of the four powers behind the 高级 gate (the fourth, script file I/O, is in
nanonis_script_files.py because it needs the allow-list). Each was deliberately left
unwritten on 2026-07-13 and is now available, OFF by default, from 高级 → 高级能力.

Also here, and NOT gated: ``SetMultiPass``. Multi-pass scanning — scanning the same
line several times with different bias/Z on each pass — is an ordinary scan feature.
It is the CONFIG FILE I/O that needed the gate, not the mode.
"""

from __future__ import annotations

import logging
import time as _time

from mast.core.diagnostics import record
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

# Scan_Action(action, direction): 1 = STOP
_SCAN_STOP = 1


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _rv(record):
    """统一解包并返回回包 body，见 io.nanonis_files.decode_reply。

    不把 (error, raw_bytes, body) 信封直接当成读数交给调用方。
    """
    from mast.io.nanonis_files import decode_reply

    rv = getattr(record, "return_value", None)
    return None if rv is None else decode_reply(rv)


class QuitNanonis(BaseSkill):
    """Quit the Nanonis software — after making the tip safe."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="QuitNanonis",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "退出 Nanonis 软件。\n"
                "\n"
                "**它总是先停掉扫描、并把针尖退回来。** 在针尖仍处于工作状态时退出，会把它留在表面里、"
                "且没有任何软件盯着它 —— Z 反馈会随着进程一起死掉。如果退针无法被确认，这个技能会拒绝退出。"
                "\n"
                "\n"
                "它成功之后，MAST 与仪器的连接就没了，它关于仪器所相信的一切也不再为真。在 Nanonis 重启之前，"
                "别的什么都不会工作。\n"
                "\n"
                "这走的是 Nanonis 自己的优雅关闭流程，也是停止它的**正确**方式 —— 在事务进行到一半时强杀进程，"
                "会永久损坏 TCP 端口，反正也得重启 Nanonis 才能恢复。"
            ),
            parameters=[
                ParameterSpec(name="save_settings", type="bool",
                              description="退出途中保存当前的设置／布局",
                              required=False, default=True),
                ParameterSpec(name="settings_name", type="str",
                              description="保存设置时用的名字（留空 = 当前那个）",
                              required=False, default=""),
                ParameterSpec(name="layout_name", type="str",
                              description="保存布局时用的名字（留空 = 当前那个）",
                              required=False, default=""),
            ],
            estimated_duration_s=15.0,
            composition_level=0,
            tags=["system", "quit", "advanced", "dangerous"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "QuitNanonis"
        calls: list = []

        # 1. Stop the scan. A scan running into a quit is a half-written file.
        rec = context.safe_call("Scan_Action", _SCAN_STOP, 0)
        calls.append(rec)
        if rec.error:
            logger.warning("QuitNanonis: 停止扫描失败（%s）——继续退针", rec.error)

        # 2. Retract. This is the one that is not optional.
        rec = context.safe_call("ZCtrl_Withdraw", 1, -1)
        calls.append(rec)
        if rec.error:
            return _fail(name,
                         f"**拒绝退出**：退针失败（{rec.error}）。"
                         "带着进针状态退出 Nanonis，等于让针尖留在表面而没有任何软件看管它——"
                         "Z 反馈会随进程一起死掉。先把针退干净。", calls)

        # 3. Confirm the retract against the REAL-TIME controller, not the module's
        #    opinion and not our own memory of having asked. (See mast.skills.verify:
        #    Nanonis' manual says the two disagree during the communication delay.)
        from mast.skills.verify import verify_z_controller
        v = verify_z_controller(context, expect=False)
        calls.append(v["record"])
        if not (v["verified"] and v["on"] is False):
            why = ("实时控制器回报 Z 反馈仍然闭合" if v["verified"]
                   else f"无法确认 Z 反馈状态（{v['error']}）")
            return _fail(name,
                         f"**拒绝退出**：{why}。退针未确认就退出软件，"
                         "针尖会在无人看管的情况下留在表面。", calls)

        record("note", "QuitNanonis",
               "agent 退出了 Nanonis 软件（已先停扫描、已确认退针）",
               save_settings=bool(params.get("save_settings", True)))

        # 4. Quit. The connection dies with it — a missing/garbled response here is
        #    EXPECTED, not a failure. Util_Quit(Use_Stored_Values, Settings_Name,
        #    Layout_Name, Save_Signals)
        save = 1 if bool(params.get("save_settings", True)) else 0
        rec = context.safe_call(
            "Util_Quit", save,
            str(params.get("settings_name", "") or ""),
            str(params.get("layout_name", "") or ""),
            save,
        )
        calls.append(rec)
        # Do NOT treat an error here as a failure to quit: Nanonis tears the socket
        # down as it exits, so the response often never arrives. Report it honestly
        # instead of claiming either outcome.
        if rec.error:
            return SkillResult(
                skill_name=name, success=True, nanonis_calls=calls,
                data={"quit_sent": True, "response": None, "tip_retracted": True},
                summary=("已发出退出指令，针尖已确认退回。未收到响应——这是正常的："
                         "Nanonis 退出时会直接断开 socket。请确认 Nanonis 已关闭；"
                         "在它重启之前 MAST 无法再与仪器通信。"),
            )
        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"quit_sent": True, "response": _rv(rec), "tip_retracted": True},
            summary="Nanonis 已退出（已先停扫描、已确认退针）。重启 Nanonis 前 MAST 无法与仪器通信。",
        )


class SetMultiPass(BaseSkill):
    """Multi-pass scanning on/off. An ordinary scan feature — NOT behind the gate."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetMultiPass",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "打开或关闭多程扫描（multi-pass）。\n"
                "\n"
                "多程扫描会用不同的设置把同一条线扫上好几遍 —— 这是把形貌与静电／磁性信号分开的标准做法（第 1 程在反馈开启下记录形貌；"
                "第 2 程在某个抬升高度上、反馈关闭地重走一遍）。\n"
                "\n"
                "每一程各自的参数（偏压、Z 偏移、反馈开／关）在 Nanonis 界面里设定。把它打**开**，"
                "会让此后每一次扫描都要花 N 倍的时间。"
            ),
            parameters=[
                ParameterSpec(name="on", type="bool",
                              description="启用多程扫描",
                              required=True),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["scan", "multipass", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        on = bool(params["on"])
        rec = context.safe_call("MPass_Activate", 1 if on else 0)
        if rec.error:
            return _fail("SetMultiPass", f"MPass_Activate failed: {rec.error}", [rec])
        return SkillResult(skill_name="SetMultiPass", success=True, nanonis_calls=[rec],
                           data={"multipass_on": on},
                           summary=f"多程扫描已{'启用' if on else '关闭'}")


class LoadMultiPassConfig(BaseSkill):
    """Load the multi-pass configuration from a file."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="LoadMultiPassConfig",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "从 Nanonis 机器上的一个文件载入一份多程扫描配置。\n"
                "\n"
                "这个文件决定了**每一程**做什么 —— 它的偏压、它的 Z 偏移、反馈开不开。MAST 读不了这个文件，"
                "也没法告诉你它将要做什么。一份带着大幅负 Z 偏移、且反馈关闭的配置，会在每一次扫描的每一行上把针尖开进表面。"
                "\n"
                "\n"
                "请载入你自己写的配置。然后在扫描之前用 GetScanPatternConfig／Nanonis 界面把它读回来核对。"
            ),
            parameters=[
                ParameterSpec(name="file_path", type="str",
                              description="**NANONIS 机器上**多程配置文件的路径",
                              required=True),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["scan", "multipass", "file", "advanced", "dangerous"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        path = str(params["file_path"])
        rec = context.safe_call("MPass_Load", path)
        if rec.error:
            return _fail("LoadMultiPassConfig", f"MPass_Load failed: {rec.error}", [rec])
        record("note", "LoadMultiPassConfig",
               "多程扫描配置已从文件载入（MAST 看不见文件内容——每一程的偏压/Z 偏移由它决定）",
               file_path=path)
        return SkillResult(
            skill_name="LoadMultiPassConfig", success=True, nanonis_calls=[rec],
            data={"file_path": path},
            summary=(f"多程配置已载入：{path}。"
                     "**扫描前请先确认每一程的偏压/Z 偏移**——MAST 看不见这个文件的内容。"),
        )


class SaveMultiPassConfig(BaseSkill):
    """Save the current multi-pass configuration to a file."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SaveMultiPassConfig",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把当前的多程扫描配置保存到 Nanonis 机器上的一个文件。它不改变仪器上的任何东西；"
                "它会覆盖给定路径上已有的文件。"
            ),
            parameters=[
                ParameterSpec(name="file_path", type="str",
                              description="**NANONIS 机器上**的目标路径",
                              required=True),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["scan", "multipass", "file", "advanced"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        path = str(params["file_path"])
        rec = context.safe_call("MPass_Save", path)
        if rec.error:
            return _fail("SaveMultiPassConfig", f"MPass_Save failed: {rec.error}", [rec])
        record("note", "SaveMultiPassConfig", "多程扫描配置已保存到文件", file_path=path)
        return SkillResult(
            skill_name="SaveMultiPassConfig", success=True, nanonis_calls=[rec],
            data={"file_path": path}, summary=f"多程配置已保存到 {path}",
        )


class WaitForScanEndBlocking(BaseSkill):
    """Nanonis' own blocking wait, **sliced** so the operator's stop can land.

    ## 为什么是切片的（2026-08-11）

    这里原来是一条 ``Scan_WaitEndOfScan(timeout_s * 1000)`` —— 一次最长 **1800 秒**
    的阻塞调用，中间**没有任何检查点**。它的 description 当时写着「中止 still works
    (the emergency port is a separate socket)」：急停口确实是另一条 socket，但**用户
    的软停今天不往那条 socket 上发任何东西**（``runtime.abort_run`` 只 ``ev.set()``，
    只有 ``emergency_stop()`` 会发停止动词）。于是那句话描述的是一个没人走的通道 ——
    正是 ``core/abortability`` 那个模块要消灭的句式。

    切片修好了它，而且**不损失这个技能存在的理由**：``Scan_WaitEndOfScan`` 在扫描
    结束的那一刻就返回，不管你给的超时是 1800 秒还是 1 秒。所以把一次长等待换成
    「一串短等待」之后：

      * 扫描真的结束时，仍然是**那一刻**返回（精度没变 —— 这正是它相对
        ``WaitScanComplete`` 的唯一卖点）；
      * 每个切片之间多了一次 ``check_abort()``，于是停止延迟 ≤ 一个切片。

    换句话说这一条从 ``BLOCKING_HELD`` 变成了 ``POLLED``，代价是每片一次 TCP 往返。
    """

    #: 一个切片的长度。停止延迟的上界；也是空转往返的频率。
    _SLICE_S = 1.0

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="WaitForScanEndBlocking",
            version="1.1.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "用 Nanonis 自己的等待，阻塞直到当前扫描结束。\n"
                "\n"
                "**你几乎肯定想要的是 WaitScanComplete** —— 轮询版的那个。它不占住连接，"
                "而且会报告进度。这一个是留给「你需要扫描结束的那个精确时刻、容忍不了一个轮询间隔的误差」"
                "的场合。\n"
                "\n"
                "它的代价：整段等待期间它**占住主 TCP 连接**，所以在它返回之前，那条连接上其他技能一个都跑不了。"
                "环境监控仍然工作（它走自己的端口）。\n"
                "\n"
                "**中止**：这段等待被切成 ~1 s 一段的 Nanonis 等待，两段之间夹一次 abort 检查，"
                "所以按下停止大约一秒内就会结束 —— 而且**不会**丢掉扫描结束的那个精确时刻（不论给它多长的 timeout，"
                "Nanonis 都会在扫描结束的瞬间返回）。它**不会**停掉扫描本身；它停的是「等它」"
                "这件事。\n"
                "\n"
                "那个 timeout 是一条真的界限，不是走过场：挑一个你等得起的。"
            ),
            parameters=[
                ParameterSpec(name="timeout_s", type="float",
                              description="最长阻塞多久（在这段时间里主连接是不可用的）",
                              unit="s", required=True,
                              min_value=1.0, max_value=1800.0),
            ],
            estimated_duration_s=60.0,
            composition_level=0,
            tags=["scan", "wait", "blocking", "advanced"],
        )

    @staticmethod
    def _timed_out(record_) -> bool:
        """Nanonis 的返回是 ``(timeout_status, path_size, path)``；第 0 项 1=超时。

        读不出来就当作**没有超时**（= 扫描结束了，退出等待）。反过来猜的话，一个
        读不懂的返回会让这个技能一直等到总超时 —— 把一次读取故障变成一次长等待。
        """
        rv = _rv(record_)
        try:
            first = rv[0] if isinstance(rv, (list, tuple)) else rv
            # 数值数组字段在这个库里是一串 1-元组（见 nanonis_array_fields_are_tuples）
            if isinstance(first, (list, tuple)):
                first = first[0]
            return int(first) == 1
        except (TypeError, ValueError, IndexError):
            return False

    def execute(self, context, params: dict) -> SkillResult:
        timeout_s = float(params["timeout_s"])
        deadline = _time.monotonic() + timeout_s
        calls: list = []
        # Scan_WaitEndOfScan takes MILLISECONDS. Passing seconds would wait 1000× too
        # short and report "scan finished" the instant it started.
        while True:
            if context.check_abort():
                return SkillResult(
                    skill_name="WaitForScanEndBlocking", success=False,
                    nanonis_calls=calls,
                    data={"timeout_s": timeout_s, "aborted": True,
                          "waited_s": round(timeout_s - max(0.0, deadline - _time.monotonic()), 2)},
                    error=("aborted by operator — 已停止等待扫描结束。"
                           "**扫描本身没有被停**（这个技能只负责等，不负责停）；"
                           "需要停扫描请用 StopScan / 紧急停止。"),
                )
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                return SkillResult(
                    skill_name="WaitForScanEndBlocking", success=True,
                    nanonis_calls=calls,
                    data={"timeout_s": timeout_s, "timed_out": True,
                          "result": _rv(calls[-1]) if calls else None},
                    summary=f"等到上限仍未结束（上限 {timeout_s:g} s）",
                )
            slice_ms = int(min(self._SLICE_S, remaining) * 1000)
            rec = context.safe_call("Scan_WaitEndOfScan", max(1, slice_ms))
            calls.append(rec)
            if rec.error:
                return _fail("WaitForScanEndBlocking",
                             f"Scan_WaitEndOfScan failed: {rec.error}", calls)
            if not self._timed_out(rec):
                return SkillResult(
                    skill_name="WaitForScanEndBlocking", success=True,
                    nanonis_calls=calls,
                    data={"timeout_s": timeout_s, "timed_out": False,
                          "result": _rv(rec)},
                    summary=f"扫描已结束（上限 {timeout_s:g} s）",
                )
