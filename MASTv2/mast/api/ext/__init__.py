"""外部 agent 网关 —— ``/api/ext/v1``：给 Claude Code 等外部 agent 用的稳定外部面。

## 它是什么

一个挂在主 API 下的 FastAPI **子应用**（``mast.api.app`` 里 ``app.mount``），把内部
agent 白拿的那些东西翻译成外部拿得到的形状：

* **作业**：提交 → 轮询 → 取消 → 幂等重发；客户端超时不再等于「失败」。
* **入账**：外部动作进同一套实验记录（v1/v2 ``actions``、``scan_files``、实验文件夹
  归档），并署名 ``ext:<名字>``。
* **简报**：内部 agent 每一轮看到的针尖 / 仪器档案 / live / 偏好 / 续工块，加上
  「在我之前谁做了什么」，一次给全。
* **技能检索**：按动作 / Nanonis 命令找技能；技能卡带这台机器上的实测耗时。
* **原始数据**、**笔记**（写进 MAST 记忆库，内部 agent 会自动召回）、**向操作员
  发问**、**交接报告**、**组合技能 / Python 技能提议**。

设计稿（陷阱清单、契约、决策理由）：``docs/v2/design/external_agent_gateway.md``。

## 契约规则

* ``v1`` 之内**只增不减**：字段与端点可以加，不能删、不能改语义（契约基线测试钉着）。
* 这是薄门面：执行只走 ``ExecutionContext.run``（经 ``mast.api.direct_exec``），
  这里不发 ``safe_call``、不取仪器令牌、不自己拼 ``ExecutionContext``。
* 身份头 ``X-MAST-Actor`` 只用于**归属**，不是认证；认证沿用主 API。

## 刻意不进外部面的东西

HITL / 审批的 resolve（外部 agent 不替人拍板）、admin、settings 写、技能覆盖层、
运行模式切换 —— 运行模式是操作员的选择；SAFE 下需要针尖处理就经 ``/requests`` 问人。
"""

from __future__ import annotations

from mast.api.ext.common import API_VERSION, EXT_PREFIX

__all__ = ["API_VERSION", "EXT_PREFIX", "create_ext_app"]


def create_ext_app(ctx, **kw):
    """建外部面子应用（惰性 import，免得只想要常量的调用方把整个网关拉进来）。"""
    from mast.api.ext.app import create_ext_app as _create

    return _create(ctx, **kw)
