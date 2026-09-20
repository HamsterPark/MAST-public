# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for MAST2 desktop bundle.

Build:
    .venv-v2-py313/Scripts/python.exe -m PyInstaller mast2.spec --noconfirm --clean

Output:
    dist/MAST2/MAST2.exe   (entry: mast2_launcher.py)
    dist/MAST2/_internal/  (Python runtime + bundled deps)

HISTORICAL GUARD — this repo once had BOTH ``mast/`` (v1) and ``MASTv2/mast/``
(v2) at the top level. v1 was archived out on 2026-06-01, so the root
``mast/`` is gone; the guard below is kept because a stale worktree or cache
can still resurrect it, and the failure mode is silent — a bundle built from
the wrong tree. Original strategy:
  1. Remove the repo root from sys.path so v1 mast/ is invisible
  2. Insert MASTv2/ first so ``import mast`` resolves to v2
  3. Pass only MASTV2_ROOT to ``pathex``; do NOT include REPO_ROOT
  4. Use Tree(...) to add the MASTv2/mast tree directly as datas, in
     case any data files (yaml/json) aren't picked up via collect_*
"""

import os
import pathlib
import sys

from PyInstaller.utils.hooks import collect_submodules, collect_all


REPO_ROOT = os.path.abspath(os.path.dirname(SPEC))  # noqa: F821 — SPEC injected by PyInstaller
MASTV2_ROOT = os.path.join(REPO_ROOT, "MASTv2")


# Force MASTv2 to be the first place Python looks for `mast.*` packages.
# Strip REPO_ROOT so v1 `mast/` isn't accidentally pulled in by PyInstaller's
# Analysis (which adds the spec dir to sys.path).
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(REPO_ROOT)]
if MASTV2_ROOT not in sys.path:
    sys.path.insert(0, MASTV2_ROOT)

# Wipe any already-cached `mast.*` modules (e.g. if pyinstaller-hooks pre-imported v1)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

# Pre-import the v2 mast package so it's the cached one in sys.modules.
# Without this, PyInstaller's Analysis subprocess does `import mast` itself,
# and even with pathex putting MASTv2 first, the spec file's own directory
# (REPO_ROOT) is also on Analysis's sys.path → v1 wins.
import importlib
importlib.invalidate_caches()
import mast  # noqa: E402,F401
assert "MASTv2" in (getattr(mast, "__file__", "") or "").replace("\\", "/"), (
    f"Pre-import resolved to v1: {mast.__file__}. PyInstaller would bundle "
    "the wrong tree. Check sys.path manipulation above."
)
# Force-import the most important submodules so they cache as v2.
for _modname in (
    "mast.webui", "mast.api", "mast.api.app", "mast.core.runtime",
    "mast.skills", "mast.skills.builtins", "mast.skills.composite",
    "mast.agents", "mast.agents._shared",
    "mast.agents.orchestrator", "mast.agents.literature",
    "mast.agents.experiment_design", "mast.agents.instrument_control",
    "mast.agents.data_processing", "mast.agents.paper_writing",
    "mast.agents.paper_review", "mast.agentruntime", "mast.vision", "mast.knowledge",
    "mast.config", "mast.core", "mast.buffer", "mast.voice",
    "mast.monitoring", "mast.documents", "mast.envhistory",
    "mast.logging", "mast.update", "mast.admin", "mast.pipeline",
):
    try:
        importlib.import_module(_modname)
    except Exception as _e:
        # Some submodules may have heavy side-effects at import time
        # (e.g. mast.webui.app launches Gradio) — best effort only.
        pass


# ---- mast subpackages — only v2 should match ---------------------------------
mast_hidden: list[str] = []
for sub in (
    "mast",
    "mast.core",
    "mast.config",
    "mast.webui",
    "mast.api",
    "mast.api.routes",
    "mast.core.runtime",
    "mast.skills",
    "mast.skills.builtins",
    "mast.skills.composite",
    "mast.skills.custom",
    "mast.knowledge",
    "mast.logging",
    "mast.voice",
    "mast.buffer",
    "mast.vision",
    # Tunnelling-current monitor. collect_submodules("mast") above already
    # reaches it, but that call is wrapped in a bare try/except — one subpackage
    # with an import side-effect takes the whole recursive collection down with
    # it, and the failure mode here is silent: the build succeeds and the
    # feature is simply absent on the instrument.
    "mast.monitoring",
    # 实验文档子系统（报告 / 计划 / 论文草稿的版本化落盘）。与 mast.monitoring
    # 同理显式列出：漏了它打包版不会报错，只是 save_draft 一调用就 ImportError，
    # 而那要等到仪器上真的写报告时才会发现。
    "mast.documents",
    # 环境参数历史记录器。同一条理由，而且它的失败最难察觉：所有入口都吞异常
    # （记录器坏掉的正确表现就是"没有历史"），漏打包在界面上看起来只是
    # "这台机器还没攒出数据"，可能几周都没人发现。
    "mast.envhistory",
    # 计费账本 + 成本闸门探针（mast.billing.run_meter）。**同一条理由，而且这个
    # 的静默程度更高**：编排器的 USD 闸门是靠 runtime 里一句 try/except 导入
    # RunMeter 接上的，导入失败就把 _run_meter 置 None —— 也就是**闸门回到
    # 「谁也没 seed 过」那个状态**，而那正是它此前一直没有生效、单次 run 唯一
    # 上界只剩 recursion_limit（≈$50）的原因。日志里只有一行 debug，界面上
    # 什么都看不出来，直到某次跑飞才发现上限从来没起作用过。
    "mast.billing",
    # 多天 campaign 指挥层(引擎 + 实验模板)。同一条理由,而且模板子包尤其危险:
    # 模板是**数据形态的代码**,漏进去不会有 ImportError,只会让 campaign 面板的
    # 模板下拉变成空的 —— 看起来像「还没建过模板」。
    "mast.campaign",
    "mast.campaign.templates",
    # agent 会话运行时（退出 LangGraph 的新家，2026-08-26/27）。**必须显式列出**，
    # 而且理由比上面几个更硬：生产代码对它的引用**全部是函数内惰性 import**
    # （`runtime.py` 的 `_ensure_message_store` / `_import_chat_history_once` /
    # `_build_instrument_loop_v2`、`agent_node._run_via_loop`、
    # `routes/orchestrator._drive_v2`、`chat/engine_v2`），PyInstaller 的静态分析
    # 看不见那些。
    #
    # 漏了它的失败形态是这次迁移**最不该有**的那一种：四个开关默认关，所以打包版
    # 一切正常；直到操作员翻开某个开关 —— 那一刻才 ImportError，而每条切换面的
    # 兜底都会**静默退回旧路径**。也就是说：开关翻了、界面显示已生效、实际什么都
    # 没变，且日志里只有一行 warning。
    "mast.agentruntime",
    # 以下两个是 2026-08-27 由 test_packaging_declares_the_runtime 那道新闸门查出来的
    # **既有**缺口 —— 与 agentruntime 同形状（只被函数内 import），只是先于它存在。
    #
    # mast.conduct —— 多天 day 层执行体（常驻线程 + SQLite 状态机 + 闸门级恢复）。
    #   全部引用在 `agents/_shared/conduct_tools.py` 的函数体里。漏了它不会有
    #   ImportError 弹出来：那些工具的 except 会把失败读成「conduct 服务没起来」，
    #   而 `cd_enabled` 默认 0 —— 于是「没装进去」和「按设计关着」在界面上完全一样。
    "mast.conduct",
    # mast.pyexec —— 沙箱 Python 执行（数据分析 agent 的 run_python）。
    #   引用在 `agents/data_processing/tools.py` 的函数体里。漏了它的表现是
    #   「这个工具一调用就失败」，而失败信息指向一个操作员不认得的模块名。
    "mast.pyexec",
    # mast.goals —— 目标终止判据（闭集谓词 + 证据收集器；求值核复用
    #   mast.conduct.rules）。全部引用都在函数体里：orchestrator 的
    #   `_goal_verdict`、routes/orchestrator 的 422 校验、runtime 的唤醒准入。
    #   漏了它**不会**报错，只会静默退回「没有目标判据」——也就是这条改动之前的
    #   行为：模型自己判何时结束。冻结包与源码跑出两种停机时机，而两边都说自己
    #   正常。
    "mast.goals",
    "mast.agents",
    "mast.agents._shared",
    "mast.agents.orchestrator",
    "mast.agents.literature",
    "mast.agents.experiment_design",
    "mast.agents.instrument_control",
    "mast.agents.data_processing",
    "mast.agents.paper_writing",
    "mast.agents.paper_review",
    # 2026-08-27：以下三个由 test_packaging_declares_the_runtime 的新闸门查出来 ——
    # 磁盘上有、而这张**显式**名单里没有。它们此前只靠上面 collect_submodules("mast")
    # 那一句递归兜底，而那句裹在裸 try/except 里：任何一个子包 import 有副作用，
    # 整棵树静默丢光。这张名单存在的全部理由就是不信那句兜底。
    #
    # research_director —— **第七个 agent**（Campaign 层「为什么做」，2026-08-21 加入）。
    #   前六个都在名单里，它加进来时没人回来补这一行 —— 正是「每处各自记得」的形状。
    "mast.agents.research_director",
    # brainstorm —— facilitator + 约 6 个视角 agent 的讨论编排（GUI 可调用）。
    #   2026-08-27 已改写成零 langgraph 的显式循环。
    "mast.agents.brainstorm",
    # buffer_summarizer —— 把 Vision 的结构化结果转成 IC agent 读得懂的叙述。
    #   漏了它的表现是「IC 看不见视觉结论」，而那在界面上像「视觉没跑出东西」。
    "mast.agents.buffer_summarizer",
    "mast.admin",
    "mast.update",
    "mast.pipeline",
    # v1 supporting modules (vendored for MAST2 v1-style GUI):
    "mast.llm",
    "mast.environment",
    "mast.planning",
    "mast.citations",
    # 2026-08-02：以下几个此前只靠上面 collect_submodules("mast") 那一句递归兜底，
    # 而那句裹在裸 try/except 里 —— 任何一个子包 import 有副作用，整棵树就静默丢
    # 光，而这些恰好是本轮 13 个子系统的落点：mast.io 里是 map_analysis（扫描地图
    # 智能化）/ coarse_map（粗动）/ z_trace（针尖修整），mast.chat 里是私聊导出。
    # 显式列出后各自独立 try/except，一个坏了不牵连其余。
    "mast.io",
    "mast.chat",
    "mast.instruments",
    "mast.memory",
    "mast.net",
    "mast.prompts",
    "mast.wishlist",
    "mast.data",
    # campaign 指挥层。上面的裸名单只能带进 __init__,而 M1-c 的四个模块
    # (adapters / service / journal / settings)都是**运行时才 import 的**
    # —— runtime 挂点与路由里那种函数内 import。递归收集是唯一稳妥的写法:
    # 漏了不会有 ImportError,只会让「启用 campaign」这个开关在打包版上
    # 永远不起作用,而日志里只有一行 debug。
    "mast.campaign",
    # 数据图库（docs/v2/design/data_gallery.md）。它的每一处生产引用都是 api/routes/gallery.py
    # 函数体里的惰性 import（路由在包坏掉时要能 degraded 而不是让整个 app 起不来），
    # PyInstaller 的静态分析看不见 —— 漏了不会报错，只会让打包版的「数据图库」永远 degraded。
    "mast.gallery",
):
    try:
        mast_hidden += collect_submodules(sub)
    except Exception:
        pass

mast_hidden += [
    "mast._buildinfo",
    "mast._runtime_paths",
    "mast.core.platform_patch",
    "mast.core.nanonis_patch",
]


# ---- third-party deps PyInstaller often misses ------------------------------
extra_datas: list[tuple[str, str]] = []
extra_binaries: list[tuple[str, str]] = []
extra_hidden: list[str] = []

for pkg in (
    # Web service stack (Gradio REMOVED in the TS rewrite — FastAPI serves the
    # typed API + the bundled TS SPA; the old gradio/gradio_client/safehttpx/
    # groovy/ffmpy/pydub/markdown_it deps are no longer bundled).
    "starlette",
    "fastapi",
    "uvicorn",
    "websockets",
    "wsproto",
    "h11",
    "httpx",
    "httpcore",
    "anyio",
    "sniffio",
    "multipart",
    "python_multipart",
    "aiofiles",
    "uvloop",
    "pydantic",
    "pydantic_core",
    "anthropic",
    "openai",
    "langchain",
    "langchain_core",
    "langchain_anthropic",
    "langchain_openai",
    "langgraph",
    "langgraph_checkpoint",
    "langgraph_checkpoint_sqlite",
    "tiktoken",
    "yaml",
    "skimage",
    "scipy",
    "matplotlib",
    "PIL",
    "Pillow",
    "pystray",
    "RestrictedPython",
    "pymupdf",
    "fitz",
    "aiosqlite",
    "nanonis_spm",
    # --- Phase 9 M12 vision stack (DINOv3-ViT-L/16 + LoRA + 3 heads) ---
    # torch/timm load the backbone; peft loads the LoRA; transformers +
    # accelerate are hard transitive deps of peft; safetensors +
    # huggingface_hub read the bundled offline backbone cache. timm's model
    # registry + transformers' lazy submodules are dynamic, so collect_all is
    # required (PyInstaller static analysis misses them).
    "torch",
    "torchvision",
    "timm",
    "peft",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "tokenizers",
    "regex",
    # torch import-time transitive deps (filelock/fsspec/jinja2/networkx/sympy
    # per `torch` Requires). sympy + networkx do dynamic submodule imports that
    # PyInstaller's static analysis misses, so collect_all them explicitly;
    # mpmath is sympy's runtime dep.
    "sympy",
    "mpmath",
    "networkx",
    "jinja2",
    "filelock",
    "fsspec",
    # Environment sensors (DL-7 vacuum / Lakeshore temperature over serial).
    # collect_all picks up serial.tools.list_ports_windows (the conditional
    # platform backend PyInstaller's static analysis can miss), so COM-port
    # auto-detection works in the frozen binary, not just configured ports.
    "serial",
    # LAN TLS (self-signed cert auto-gen) + OTA manifest signing (Ed25519). Both
    # import cryptography LAZILY inside functions, which PyInstaller's static
    # analysis misses → without this the frozen app silently loses HTTPS + update
    # signature verification. cryptography ships C/OpenSSL bindings → collect_all.
    "cryptography",
    # pandas 3.x parquet + string columns route through pyarrow's C++ DLLs, and
    # mast/__init__.py eager-imports it to fix the Windows DLL load order (late
    # pyarrow init after the langchain/torch stack access-violates). Bundle the
    # whole package so the frozen app keeps that guarantee.
    "pyarrow",
    # Word 导出（report_docx）。**必须 collect_all,不能只靠 hiddenimports**:
    # python-docx 自带 ~440 KB 的模板数据(docx/templates/default.docx +
    # word/styles.xml),`Document()` 一上来就读它 —— 数据文件没跟过去的话打包版
    # 直接抛异常,而那是「打包成功、功能在仪器上不存在」那一类静默失效
    # (与 mast.monitoring 那条注释同一个教训)。lxml 是它的 C 扩展依赖。
    "docx",
    "lxml",
):
    try:
        d, b, h = collect_all(pkg)
        extra_datas += d
        extra_binaries += b
        extra_hidden += h
    except Exception:
        pass


# ---- knowledge / logo data files --------------------------------------------
data_files = [
    (os.path.join(MASTV2_ROOT, "mast", "knowledge", "stm_glossary.yaml"),
     os.path.join("mast", "knowledge")),
    (os.path.join(MASTV2_ROOT, "mast", "knowledge", "literature_priors.json"),
     os.path.join("mast", "knowledge")),
    (os.path.join(MASTV2_ROOT, "mast", "knowledge", "material_coverage.json"),
     os.path.join("mast", "knowledge")),
    # Skill Chinese display names. builder_api._packaged_skill_zh() loads it via
    # Path(__file__).parent / "skill_zh.json" and swallows the failure at DEBUG
    # level, so a missing file degrades SILENTLY to English skill names in the
    # frozen build. Never registered here after the gui→webui rename.
    (os.path.join(MASTV2_ROOT, "mast", "webui", "skill_zh.json"),
     os.path.join("mast", "webui")),
    (os.path.join(REPO_ROOT, "logo", "MAST2_logo.png"), "logo"),
    (os.path.join(REPO_ROOT, "logo", "MAST2_logo.ico"), "logo"),
    # OTA update server's self-signed LAN cert (PUBLIC certificate, NO private
    # key) — bundled so every client auto-trusts the HTTPS update server without a
    # per-client copy. The update client's _verify_arg falls back to
    # _MEIPASS/update_server_ca.pem. Dropped by the exists() filter if absent
    # (then clients rely on a manually-placed <data>/api key/update_server_ca.pem).
    (os.path.join(REPO_ROOT, "installer", "update_server_ca.pem"), "."),
]
# Bundle any prompts/ directories under mast.llm or mast.knowledge
for _root, _, _files in os.walk(os.path.join(MASTV2_ROOT, "mast", "llm", "prompts")):
    for _f in _files:
        _src = os.path.join(_root, _f)
        _rel = os.path.relpath(_src, MASTV2_ROOT)
        _dst = os.path.dirname(_rel)
        data_files.append((_src, _dst))

# Bundle the BUILT TS SPA (frontend/dist) — the new UI, served by mast.api at
# "/". collect_submodules only picks up .py, so these web assets need an
# explicit walk. dst mirrors "frontend/dist/..." so the frozen create_app finds
# it under sys._MEIPASS (see mast.api.app SPA-path resolution). Run
# `npm --prefix frontend run build` BEFORE packaging or this tree is empty.
for _root, _, _files in os.walk(os.path.join(REPO_ROOT, "frontend", "dist")):
    for _f in _files:
        _src = os.path.join(_root, _f)
        _rel = os.path.relpath(_src, REPO_ROOT)   # frontend/dist/...
        data_files.append((_src, os.path.dirname(_rel)))

# Bundle the release-notes markdown into docs/ so the launcher's
# "当前版本更新说明" button (mast2_launcher._open_release_notes → _resource_path
# ("docs")) can open them in the FROZEN build. Only the notes files, not all docs.
_docs_dir = os.path.join(REPO_ROOT, "docs")
if os.path.isdir(_docs_dir):
    for _f in os.listdir(_docs_dir):
        if "release_notes" in _f and _f.endswith((".md", ".txt")):
            data_files.append((os.path.join(_docs_dir, _f), "docs"))

# ---- mast/**.py 的源码副本（mast/_src/）-------------------------------------
#
# PyInstaller 把全部 .py 编译进 MAST.exe 内嵌的 PYZ，冻结树里 _internal/mast/
# 只剩十来个 data 文件。但有两件事要的是**文件**而不是模块：
#
#   * mast.pyexec 把 sitecustomize.py / mastdata.py / io/nanonis_files.py
#     **拷进分析会话目录** —— 那个子进程里没有 MAST，只能拿文件；
#   * skill 覆盖层的「导出到覆盖层」要拿到 skill 源文件的原文。
#
# 两个需求共用这一份，落点对应 mast.pyexec._srcfiles.source_root()。
# 收集与凭据扫描的逻辑在 MASTv2/scripts/bundle_sources.py —— spec 和测试
# **调同一份**，免得「spec 里一份、测试里另一份」各自漂移。
sys.path.insert(0, os.path.join(MASTV2_ROOT, "scripts"))
import bundle_sources as _bundle_sources  # noqa: E402

_src_report = _bundle_sources.collect(MASTV2_ROOT, REPO_ROOT)

if _src_report.degraded:
    # git 问不出来 = 第一道判据没生效。**说出来**，别让它悄悄退化成
    # 「只靠凭据扫描」。
    print("")
    print("!! 源码收集降级：%s" % _src_report.degraded)
    print("   （git 忽略的文件这次没被排除，只剩凭据扫描这一道）")
    print("")

if _src_report.violations:
    print("")
    print("!! 源码里发现疑似凭据 —— 这些**不能**随包发出：")
    for _p, _name, _preview in _src_report.violations:
        print("     %s: %s = %s" % (_p, _name, _preview))
    raise SystemExit(
        "打包中止：%d 处疑似凭据在待打包的源码里。分析子进程会拿到 mast/_src/ 的"
        "只读视图，而那是一个会写代码的 agent —— 明文 token 随源码发出去就等于"
        "把推送凭据交给它。把文件加进 .gitignore（首选，判据自动生效），或者"
        "确认它是公开值之后加进 bundle_sources.CREDENTIAL_ALLOWLIST 并写明理由。"
        % len(_src_report.violations))

if _src_report.skipped_gitignored:
    print("   源码副本跳过 %d 个 git 忽略的文件（本地生成物/凭据）："
          % len(_src_report.skipped_gitignored))
    for _p in _src_report.skipped_gitignored:
        print("     %s" % _p)

if _src_report.n < _bundle_sources.MIN_FILES:
    raise SystemExit(
        "打包中止：mast/**.py 只收到 %d 个（期望 >= %d）。走目录收进来的东西"
        "缺失时一条丢弃记录都不会有 —— 而少了它，打包版里 py_run 装不上审计钩子、"
        "技能覆盖层也导不出源码。"
        % (_src_report.n, _bundle_sources.MIN_FILES))

data_files += _src_report.files
print("   mast/**.py 源码副本 %d 个文件已收入 -> %s"
      % (_src_report.n, _bundle_sources.DEST_ROOT))


# ---- 静默丢弃的守卫 ---------------------------------------------------------
#
# 这一行下面原来只有一句 `if os.path.exists(src)` 的列表推导：**文件不在就悄悄
# 滤掉，不报警、不报错**。打包照样"成功"，缺失只在真机上以别的面目出现 ——
# skill_zh.json 没进去 = 技能名全变英文；logo 没进去 = 启动器没图标；
# CA 没进去 = 每台客户端都要手工放证书。
#
# 这个形状在本仓咬过不止一次（硬编码产物名 + 静默丢弃）。所以现在：
#   * 丢掉任何东西都**大声说**；
#   * 丢掉**必需**的东西直接让构建失败 —— 一个缺文件的安装包比构建失败糟得多，
#     因为它要到真机上才暴露，而且暴露成一个看起来无关的症状。
#
# `_OPTIONAL_DATA` 是**明确允许缺**的那些，每一条都要有理由（不是"懒得管"）。
_OPTIONAL_DATA = {
    # 公开证书（不含私钥）。缺了客户端就退回手工放 <data>/api key/update_server_ca.pem，
    # 是有出路的降级，不是断链。
    "update_server_ca.pem",
}

_before = list(data_files)
data_files = [(src, dst) for src, dst in data_files if os.path.exists(src)]
_dropped_data = [(src, dst) for src, dst in _before if not os.path.exists(src)]
if _dropped_data:
    print("")
    print("!! 打包资源缺失 —— 以下条目被丢弃：")
    for _s, _d in _dropped_data:
        print("     %s  ->  %s" % (_s, _d))
    _required = [(_s, _d) for _s, _d in _dropped_data
                 if os.path.basename(_s) not in _OPTIONAL_DATA]
    if _required:
        raise SystemExit(
            "打包中止：%d 个**必需**资源不存在（见上）。"
            "补齐文件，或者如果它确实可选，把文件名加进 mast2.spec 的 "
            "_OPTIONAL_DATA 并写明理由。" % len(_required))
    print("   （以上均在 _OPTIONAL_DATA 里，允许缺失，继续打包）")
    print("")

# 前端 SPA 是**走目录**收进来的，所以它缺失时连一个条目都不会出现在上面 ——
# 「丢弃了什么」看不见它。单独数一遍：dist 空 = 冻结版打开就是白屏。
_spa_n = sum(1 for _s, _d in data_files if _d.startswith(os.path.join("frontend", "dist")))
if _spa_n == 0:
    raise SystemExit(
        "打包中止：frontend/dist 里一个文件都没收到 —— 冻结版会白屏。"
        "先跑 `npm --prefix frontend run build`。")
print("   前端 SPA 资源 %d 个文件已收入" % _spa_n)


# ---- Analysis ---------------------------------------------------------------
# CRITICAL: PyInstaller's Analysis runs `import mast.*` with sys.path
# including the cwd (which is REPO_ROOT when invoked from there). v1's
# mast/ at REPO_ROOT then wins. The fix is to run `python -m PyInstaller`
# from `MASTv2/` so cwd's mast resolves to v2. The launch wrapper
# `installer/mast2_build.ps1` does this; if you invoke pyinstaller
# directly, cd to MASTv2/ first.

a = Analysis(
    [os.path.join(REPO_ROOT, "mast2_launcher.py")],
    pathex=[MASTV2_ROOT],
    binaries=extra_binaries,
    datas=extra_datas + data_files,
    hiddenimports=sorted(set(mast_hidden + extra_hidden)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # v1 modules that we intentionally don't ship even with the v1 GUI
        # vendor (heavy training-only deps).
        "mast.training",
        "mast.rl",
        "mast.execution",
        "mast.safety",
        # NOTE: mast.io is NOT excluded — v2's mast/io/ (mosaic / exp_map /
        # plan_overlay / signal_fft) backs the vision + scan-map API routes.
        # mast.data likewise kept (used by v2 data processing).
        # ML stack: Phase 9 ships the M12 DINOv3 vision model, so torch +
        # timm + peft (+ transformers/accelerate, hard transitive deps of peft)
        # are now BUNDLED (see collect_all loop below). Only the genuinely
        # unused pieces stay excluded to keep the bundle from ballooning.
        "torchaudio",
        "torchao",
        "triton",
        "gpytorch",
        # Notebooks / tests
        "tests",
        "pytest",
        "pytest_asyncio",
        "_pytest",
        "jupyter",
        "ipython",
        "IPython",
        "ipykernel",
        "notebook",
        "nbformat",
        "nbconvert",
        # Other GUI toolkits
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "wx",
        # Heavy science extras. NOTE: sympy is NOT excluded — torch._dynamo /
        # torch.fx.experimental.symbolic_shapes import it at import time (via
        # timm→torchvision→torch._dynamo), so excluding it breaks `import timm`
        # in the frozen build (caught by --selftest-vision).
        "pandas.tests",
        "scipy.io.tests",
    ],
    noarchive=False,
    optimize=0,
)

# Sanity check: after Analysis, verify that no v1 mast path snuck into pure modules.
# (PyInstaller's Analysis stores discovered modules in a.pure with module + path.)
_v1_root = os.path.abspath(os.path.join(REPO_ROOT, "mast"))
_filtered_pure = []
_dropped = []
for entry in a.pure:
    # entry is (modname, filepath, type)
    if len(entry) >= 2:
        modname, fpath = entry[0], entry[1]
        if modname == "mast" or modname.startswith("mast."):
            if fpath and os.path.abspath(fpath).startswith(_v1_root):
                _dropped.append((modname, fpath))
                continue
    _filtered_pure.append(entry)
if _dropped:
    print(f"[mast2.spec] dropped {len(_dropped)} v1 mast entries from Analysis:")
    for m, p in _dropped[:20]:
        print(f"  - {m}  ({p})")
    if len(_dropped) > 20:
        print(f"  ... and {len(_dropped) - 20} more")
    a.pure[:] = _filtered_pure


# ---- PYZ 凭据守卫（第二条通道）----------------------------------------------
#
# 上面 `bundle_sources` 那道凭据扫描盖住的是 **mast/_src/ 文本副本**这一条通道，
# 而且它**故意跳过 git 忽略的文件** —— 那条规则的本意是「本地生成物不该分发」。
#
# 可是 **gitignore 管不到 PyInstaller**。只要磁盘上有 `mast/update/_defaults.py`，
# `mast.update.defaults` 就在模块级 import 它（`defaults.py:17`），Analysis 把它
# 收进 `a.pure`，PyInstaller 再把它的**字节码**编译进 MAST.exe 内嵌的 PYZ ——
# 反编译 .pyc 即可读出明文。也就是说：**被 gitignore 挡在文本通道之外的那份
# 凭据，从字节码通道原样发了出去。**
#
# 两条通道，一道门只盖了一条。这里补上另一条，判据用同一个
# `scan_credentials()` —— 不另写一份，免得两边各自漂移。
# ⚠️ **分两级，因为这两种情况的正确反应不同。**
#
# `mast/update/_defaults.py` 是**已知会带凭据的部署配置载体**：客户端的
# `_read_token()` 链是 `api key/update_client_token.env` > `get_default_token()`，
# 也就是说那个 token 是**开箱即用的默认值**，移走它 OTA 就要每台机器手工配。
# 它进包是当前设计的一部分，在这里中止构建等于拦住合法发布。
#
# 但它**值得每次构建都被看见一次**：这个值随包分发，等于公开。它现在只是
# **客户端**凭据 —— 够拉更新、够客户端自己那几件写（投稿技能 / 分享订阅 /
# 提反馈）。
#
# （2026-09-10 之前它还是对称的：同一把也能 `POST /skills/pack/publish`，
# 于是拿到安装包就等于拿到发布权限。现在发布走
# `update_publish_token.env`，那把只在发布机上、没有出厂默认值、不进任何包，
# 见 `mast.update.server._read_publish_token`。）
#
# 其余任何 `mast.*` 模块里出现凭据，都是**意外**，直接中止。
_KNOWN_CRED_MODULES = {
    # 部署配置载体，见上。凭据进包是设计如此，但每次构建都要报出来。
    "mast.update._defaults",
}

_pyz_creds: list[tuple[str, str, str]] = []
_pyz_known: list[tuple[str, str, str]] = []
for entry in a.pure:
    if len(entry) < 2:
        continue
    _mod, _fp = entry[0], entry[1]
    if not (_mod == "mast" or _mod.startswith("mast.")):
        continue          # 第三方库不在本仓管辖内
    if not _fp or not os.path.isfile(_fp):
        continue
    for _name, _preview in _bundle_sources.scan_credentials(pathlib.Path(_fp)):
        _bucket = _pyz_known if _mod in _KNOWN_CRED_MODULES else _pyz_creds
        _bucket.append((_mod, _name, _preview))

if _pyz_known:
    print("")
    print("!! 以下凭据会随字节码进 MAST.exe（**设计如此，不中止**）：")
    for _m, _n, _p in _pyz_known:
        print("     %s: %s = %s" % (_m, _n, _p))
    print("   反编译 .pyc 即可读出。注意这条通道**不受 .gitignore 保护** ——")
    print("   文件不进 git 不代表它不进包，所以把它当成公开值。")
    print("   它是**客户端**凭据：够拉更新、够客户端自己那几件写（投稿技能 /")
    print("   分享订阅 / 提反馈）。发布已经分出去了 —— POST /skills/pack/publish")
    print("   要 update_publish_token.env，那把只在发布机上、不进任何包。")
    print("")

if _pyz_creds:
    print("")
    print("!! 编译进 PYZ 的模块里发现**意料之外**的凭据：")
    for _m, _n, _p in _pyz_creds:
        print("     %s: %s = %s" % (_m, _n, _p))
    raise SystemExit(
        "打包中止：%d 处疑似凭据会随字节码进 MAST.exe，而它们不在已知的部署配置"
        "模块里。**这条通道不受 .gitignore 保护** —— 文件不进 git 不代表它不进包。"
        "把值换成占位符，或者确认它是公开值之后加进 "
        "bundle_sources.CREDENTIAL_ALLOWLIST 并写明理由；如果它确实是又一个部署"
        "配置载体，加进本文件的 _KNOWN_CRED_MODULES 并说明为什么它必须进包。"
        % len(_pyz_creds))


pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MAST",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(REPO_ROOT, "logo", "MAST2_logo.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MAST",
)
