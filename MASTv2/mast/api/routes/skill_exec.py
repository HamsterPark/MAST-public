"""POST /api/skills/{name}/execute —— 不经 LLM 直接跑一个技能。

## 为什么有它

真机测试要「测每个单独环节」,而在此之前唯一的路径是**通过 agent 私聊** ——
每验一个环节都要烧一整轮模型 token,而且 agent 会自己决定要不要调、调几次、
中途还可能被它自己的 stall-guard 拦掉(2026-08-13 真的发生了:守卫按旧 socket
的失败史在回合开始就锁死工具,连试都不试)。

想测的是**技能**,不该每次都先说服一个模型。

## 它不绕过什么

刻意复用 ``ExecutionContext`` —— 与 agent 工具边界、群聊、信号采集 API 走的是
同一套:

  * 安全闸门(``SafetyGuard`` / ``safety_level`` / DANGEROUS 的 HITL);
  * 仪器仲裁(``instrument_lock``:一台物理仪器,取令牌的机制清单在它的 docstring 里);
  * 中止事件(急停/环境告警一样能停住它);
  * 状态缓存回写 + 扫描地图标记(composite 子步骤那条路也在这里)。

**唯一被绕过的是「模型决定要不要做」。** 硬件那一侧一个闸门都没少。

2026-09-18 起构造与执行都委托 ``mast.api.direct_exec``(与外部 agent 网关共用):
动作进实验记录(v1/v2 ``actions``、``scan_files``、实验文件夹归档)、每次一个新的
``run_id``、门口判一次样品门控、SI 字符串还原。请求头 ``X-MAST-Actor`` 可选 ——
给了就作为调用方身份记进 ``actions.context``。

## 谁能用

与其余 API 同一道认证。这不是给外部脚本开的后门 —— 它是给**用户和测试**用的
直连入口,和界面上的按钮同级。**长任务请改用外部网关的作业接口**
(``POST /api/ext/v1/jobs``):这个端点是同步的,客户端超时放弃之后技能仍在跑、
仍占着仪器。
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from mast.api import direct_exec
from mast.api.direct_exec import jsonable as _jsonable  # noqa: F401 — 保留旧名(测试与调用方)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills"])

#: 这个入口在仪器令牌拒绝消息里的名字 —— 被拒的一方读到的是「谁在开车」。
OWNER = "技能直调 API"


class SkillExecuteRequest(BaseModel):
    params: dict = Field(default_factory=dict,
                         description="技能参数,与 /api/skills/{name} 里声明的一致")


class SkillExecuteResponse(BaseModel):
    ok: bool = False              # 端点本身是否正常工作
    success: bool | None = None   # 技能自己的成败(None = 没跑起来)
    skill: str = ""
    data: dict = Field(default_factory=dict)
    error: str = ""
    summary: str = ""
    elapsed_s: float = 0.0
    nanonis_calls: int = 0
    degraded: bool = False        # 没有活的连接池/状态/注册表
    missing: list[str] = Field(default_factory=list)


def _execution_context(ctx: Any):
    """与 ``signals.py`` 同一份判据 —— 缺什么就说缺什么,不报成「没连仪器」。"""
    return direct_exec.build_context(ctx, owner=OWNER)


def _actor(request: Request) -> str:
    """可选的调用方身份(清洗成一个短标识)。空 = 没声明。"""
    return direct_exec.actor_slug(request.headers.get("x-mast-actor", ""))


@router.post("/skills/{name}/execute", response_model=SkillExecuteResponse)
def execute_skill(name: str, request: Request,
                  body: SkillExecuteRequest | None = None) -> SkillExecuteResponse:
    """跑一个技能,把它的原始结果回给调用方。永不 500。

    返回体里 ``ok`` 和 ``success`` 是**两件事**:``ok`` 说端点工作正常,
    ``success`` 说技能自己成没成。一个失败的技能不是这个端点的错误 ——
    把两者合并会让「读不到」和「调不动」再一次分不开。
    """
    params = (body.params if body is not None else {}) or {}
    ctx = request.app.state.ctx
    ec, missing = _execution_context(ctx)
    if ec is None:
        return SkillExecuteResponse(ok=False, degraded=True, skill=name,
                                    missing=missing,
                                    error=f"ExecutionContext 不可用,缺少: {missing}")
    actor = _actor(request)
    # ``ExecutionContext.run`` 收的是**技能名**,注册表查找、安全闸门、仪器仲裁
    # 都在它里面 —— direct_exec 不自己查、不自己拼,就是为了不出现第二条执行路径。
    run = direct_exec.run_and_record(
        ec, name, params,
        runtime=direct_exec.live_runtime(ctx),
        agent_id=(f"ext:{actor}" if actor else "skill_exec_api"),
        thread_id=None,
        context=(f"ext:{actor}" if actor else OWNER),
        approval_source=(direct_exec.APPROVAL_LLM if actor else direct_exec.APPROVAL_AUTO),
    )
    if run.error and not run.success:
        logger.info("skill-exec %s failed: %s", name, run.error[:200])
    return SkillExecuteResponse(
        ok=True,
        success=run.success,
        skill=name,
        data=_jsonable(run.data),
        error=run.error,
        summary=run.summary,
        elapsed_s=run.elapsed_s,
        nanonis_calls=run.nanonis_calls,
    )
