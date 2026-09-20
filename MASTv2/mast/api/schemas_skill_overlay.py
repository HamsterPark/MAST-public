"""Skill 覆盖层 API 的请求/响应模型。

一条贯穿全文的纪律：**「排队了」和「生效了」是两件事**。响应里不能只有一个
``ok`` —— 那会让 UI 把「已安排」显示成「已完成」，而这个功能的头号事故形态就是
「以为生效其实没有」。所以每个响应都带 ``agent_path_pending`` 和
``fingerprint_matches``：前者说图重建有没有跟上，后者说 agent 手上那张表到底是不是
注册表现在的样子。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OverlayEntryInfo(BaseModel):
    path: str = Field(description="相对覆盖层目录的 posix 路径，如 builtins/bias.py")
    enabled: bool = False
    exists: bool = Field(default=False, description="盘上有没有这个文件")
    overlay_of: str = Field(default="", description="它覆盖哪个内置模块（纯新增为空）")
    applied: bool = Field(default=False, description="这一轮**真的生效**了吗")
    skills: list[str] = Field(default_factory=list, description="它提供的技能名")
    sha256: str = ""
    error: str = Field(default="", description="被拒绝的原因（空 = 没被拒）")
    valid_path: bool = True
    path_error: str = ""


class OverlaySkillInfo(BaseModel):
    """一个技能此刻的来历 —— 给技能列表页的徽章。"""

    name: str
    origin: str = "other"
    module: str = ""
    source_path: str = ""
    short_sha: str = ""
    displaced_module: str = ""
    signature: str = "n/a"
    described: str = ""


class OverlayStatusResponse(BaseModel):
    degraded: bool = False
    reason: str = Field(default="", description="degraded 时说明为什么")

    overlay_dir: str = ""
    entries: list[OverlayEntryInfo] = Field(default_factory=list)
    #: 目录里有、但清单里没登记的文件。**必须单独列出来** —— 一个拷进来却没启用的
    #: 文件，不说的话会被当成「怎么没生效」。
    untracked: list[str] = Field(default_factory=list)
    overlaid_skills: list[OverlaySkillInfo] = Field(default_factory=list)

    #: 生效探针。``fingerprint_matches=False`` ⇒ agent 工具表还是旧的。
    fingerprint_matches: bool | None = Field(
        default=None,
        description="None = 还没建过工具表（进程刚起），判断不了 —— **不是** False")
    registry_fingerprint: str = ""
    wrapped_fingerprint: str = ""
    wrapped_at: float = 0.0
    agent_overlaid: list[str] = Field(
        default_factory=list, description="agent 手上那张表里的覆盖版技能")
    registry_overlaid: list[str] = Field(
        default_factory=list, description="注册表里的覆盖版技能")

    last_reload: str = Field(default="", description="上一次重载的可读汇总")
    baseline_drift: list[str] = Field(default_factory=list)
    pending_rebuild: bool = False


class OverlayReloadRequest(BaseModel):
    pin: str = Field(default="", description="管理员 PIN（未设置 PIN 时可空）")
    reason: str = ""


class OverlayReloadResponse(BaseModel):
    ok: bool = False
    degraded: bool = False
    status: str = Field(default="", description="ok / nochange / busy / queued")
    reason: str = ""
    summary: str = Field(default="", description="给用户看的多行汇总")

    applied: list[str] = Field(default_factory=list)
    failed: list[dict] = Field(default_factory=list)
    restored: list[str] = Field(default_factory=list)
    baseline_drift: list[str] = Field(default_factory=list)

    #: **诚实**：排队了 ≠ 已生效。UI 必须按这个显示，而不是按 ``ok``。
    agent_path_pending: bool | None = None
    refresh: str = Field(default="", description="三条刷新链各自发生了什么")
    fingerprint_matches: bool | None = None


class OverlayEntryRequest(BaseModel):
    path: str
    enabled: bool
    allow_removals: list[str] = Field(
        default_factory=list,
        description="允许这个模块不再提供的技能名 —— 会被**真的移除**")
    note: str = ""
    pin: str = ""


class OverlayEntryResponse(BaseModel):
    ok: bool = False
    degraded: bool = False
    reason: str = ""
    entry: OverlayEntryInfo | None = None


class OverlayRestoreResponse(BaseModel):
    ok: bool = False
    degraded: bool = False
    reason: str = ""
    restored: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# 导出到覆盖层（eject）
# ---------------------------------------------------------------------------
# 打包版里源码全在 MAST.exe 内嵌的 PYZ 里，磁盘上一个 .py 都没有 —— 用户想改
# 一个技能，手上根本没有那份文件。eject 把它取出来，同时回答一个**比取文件更
# 重要**的问题：覆盖它之后，哪些入口还会继续跑内置版。

class EjectImportSite(BaseModel):
    """一处直接 import —— 覆盖层管不着的入口。"""

    file: str = Field(description="相对 mast 包根的 posix 路径")
    line: int = 0
    kind: str = Field(default="", description="from_module | from_package | import_module")
    names: list[str] = Field(default_factory=list)
    text: str = ""
    binds_skill: bool = Field(
        default=False,
        description="绑的是技能类（要紧），还是基类/常量/辅助函数（多半不要紧）")


class EjectableModule(BaseModel):
    dotted: str
    rel: str = Field(description="导出后在覆盖层里的相对路径")
    n_bytes: int = 0
    already: bool = Field(default=False, description="覆盖层里已经有这一份了")
    skill_classes: list[str] = Field(default_factory=list)
    undecidable: list[str] = Field(
        default_factory=list,
        description="判断不了是不是技能类的类名 —— **不是**「不是技能类」")


class EjectableListResponse(BaseModel):
    degraded: bool = False
    reason: str = ""
    source_root: str = ""
    modules: list[EjectableModule] = Field(default_factory=list)


class EjectRequest(BaseModel):
    dotted: str = Field(description="要导出的模块，如 mast.skills.builtins.bias")
    overwrite: bool = False
    pin: str = ""


class EjectResponse(BaseModel):
    ok: bool = False
    degraded: bool = False
    reason: str = ""
    dotted: str = ""
    rel: str = ""
    path: str = ""
    sha256: str = ""
    n_bytes: int = 0
    skill_classes: list[str] = Field(default_factory=list)
    undecidable: list[str] = Field(default_factory=list)
    n_skill_binds: int = 0
    #: 一句话总结，UI 直接显示。三种形态对应三种**决定**，不是三种措辞。
    warning: str = ""
    importers: list[EjectImportSite] = Field(default_factory=list)


class OverlayDriftInfo(BaseModel):
    """内置版在这份覆盖导出之后变过没有。

    这是「以为生效」的另一个变体，而且更隐蔽：覆盖**确实生效了**，但它基于三个
    版本前的代码，上游后来修的 bug 被它原样盖了回去。
    """

    rel: str = ""
    known: bool = Field(default=False, description="有没有 sidecar 能回答这个问题")
    drifted: bool | None = Field(
        default=None, description="None = 判断不了 —— **不是** False")
    reason: str = ""
    orig_sha: str = ""
    builtin_sha: str = ""
    ejected_app_version: str = ""


# ---------------------------------------------------------------------------
# 签名技能包
# ---------------------------------------------------------------------------

class SkillPackInfo(BaseModel):
    """本机装了哪个包，以及**它现在还验得过吗**。

    ``valid`` 每次都真算（读文件比 sha），不缓存 —— 这个字段存在的全部理由就是
    回答「落盘之后有没有被改过」，而缓存正好把那件事挡在外面。
    """

    pack_id: str
    version: str = ""
    author: str = ""
    description: str = ""
    n_files: int = 0
    valid: bool = False
    reasons: list[str] = Field(default_factory=list)
    signature: str = Field(
        default="", description="verified:<签名者公钥前8位>，或 invalid")


class SkillPackListResponse(BaseModel):
    degraded: bool = False
    reason: str = ""
    packs: list[SkillPackInfo] = Field(default_factory=list)
    #: 本机会不会自动启用签名包。关掉之后包照装，只是不自动登记进清单。
    auto_enable: bool = True


class SkillPackRemoveRequest(BaseModel):
    pack_id: str
    pin: str = ""


class SkillPackAutoEnableRequest(BaseModel):
    enabled: bool
    pin: str = ""


class SkillPackActionResponse(BaseModel):
    ok: bool = False
    degraded: bool = False
    reason: str = ""
    summary: str = ""
    #: 装/删都**不会**自己生效 —— 中间还差一次重载。
    needs_reload: bool = False


class SkillPackFetchRequest(BaseModel):
    """从推送服务器拉一个签名包装上。

    ``pack_id`` 从 ``GET /skills/pack/index`` 来（服务器只列每个 id 的最新版）。
    """

    pack_id: str = ""
    pin: str = Field(default="", description="管理员 PIN（未设置 PIN 时可空）")
    #: 拉下来之后要不要顺手重载。默认 **False** —— 装是一次编辑，生效是一次决定
    #: （与 ``/skill-overlay/entry`` 刻意不自动重载同一条纪律）。
    reload_now: bool = False

    model_config = {"extra": "forbid"}


class SkillPackFetchResponse(SkillPackActionResponse):
    pack_id: str = ""
    version: str = ""
    installed: list[str] = Field(default_factory=list)
    enabled: list[str] = Field(default_factory=list)
    #: 被本机同名松散文件遮盖的条目。**必须列出来** —— 本机文件胜过推来的包是
    #: 有意的，但不说的话它长得像「装了没生效」。
    shadowed: list[str] = Field(default_factory=list)
    replaced_version: str = ""
    #: 只有 ``reload_now`` 时才有值。三态诚实，与本模块其余响应同一条纪律。
    reloaded: bool = False
    agent_path_pending: bool | None = None
    fingerprint_matches: bool | None = None
