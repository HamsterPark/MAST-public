"""外部面的请求体模型。响应体是普通 JSON（字段在各端点的 docstring 与设计稿第 3 节）。"""

from __future__ import annotations

import json
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


#: 一次作业参数序列化后的上限（字节）。技能参数是几个数与短字符串；更大的东西不该塞进参数
#: （它会原样进 journal、作业视图与实验记录）。
PARAMS_MAX_BYTES = 64_000


class JobSubmit(BaseModel):
    skill: str = Field(..., min_length=1, max_length=120, description="技能名（注册表里的名字）")
    params: dict[str, Any] = Field(default_factory=dict,
                                   description="技能参数；带量纲的可写 '5n' 这类带 SI 前缀的字符串")
    request_id: str | None = Field(None, max_length=128,
                                   description="幂等键（同一调用方内）：同 id 同内容 ⇒ 回原作业")
    note: str = Field("", max_length=500, description="给人看的备注，进作业视图")

    @field_validator("params")
    @classmethod
    def _params_are_small(cls, v: dict[str, Any]) -> dict[str, Any]:
        size = len(json.dumps(v, ensure_ascii=False, default=str).encode("utf-8"))
        if size > PARAMS_MAX_BYTES:
            raise ValueError(f"params 序列化后 {size} 字节，超过上限 {PARAMS_MAX_BYTES}")
        return v


class CancelBody(BaseModel):
    reason: str = Field("", max_length=300)


class EstopBody(BaseModel):
    reason: str = Field("", max_length=300)


class ScopeRef(BaseModel):
    id: str | None = Field(None, max_length=120, description="已有实验/样品的 id（切换）")
    name: str | None = Field(None, max_length=200, description="名字（按名字新建或复用）")
    goal: str = Field("", max_length=2000)
    description: str = Field("", max_length=2000)
    sample_type: str = Field("", max_length=120)


class ScopeBody(BaseModel):
    experiment: ScopeRef | None = None
    sample: ScopeRef | None = None
    force: bool = Field(False, description="仪器正被占用时仍然切换（在跑的动作会记到新作用域下）")
    reason: str = Field("", max_length=300)


class NoteBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=20000)
    kind: str = Field("note", max_length=40,
                      description="note | insight | summary | hypothesis | protocol")
    tags: list[str] = Field(default_factory=list)
    scope: Literal["experiment", "global"] = "experiment"


class RequestBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    kind: str = Field("question", max_length=40, description="question | action | info")


class HandoverBody(BaseModel):
    summary: str = Field(..., min_length=1, max_length=20000)
    next_steps: Union[str, list[str]] = Field(default_factory=list)
    title: str = Field("", max_length=200)
    since: str | None = Field(None, max_length=40,
                              description="ISO 时间；只汇总此后的作业与动作（缺省 = 本进程里的全部）")


class ProposalBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    code: str = Field(..., min_length=1, max_length=200_000)
    rationale: str = Field(..., min_length=1, max_length=4000)


class CompositeBody(BaseModel):
    spec: Union[dict[str, Any], str] = Field(..., description="CompositeSpec JSON 对象；'?' 返回格式说明")
    base_version: int = Field(-1, description="覆盖已存在的同名技能时必须给（乐观锁）")


__all__ = ["CancelBody", "CompositeBody", "EstopBody", "HandoverBody", "JobSubmit",
           "NoteBody", "ProposalBody", "RequestBody", "ScopeBody", "ScopeRef"]
