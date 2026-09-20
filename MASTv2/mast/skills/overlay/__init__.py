"""Skill 覆盖层 —— 改 skill 不用发新版本。

417 个技能里绝大多数是编译进 ``MAST.exe`` 的 Python ``BaseSkill`` 类。改一个判据
阈值要走完整的「构建 → 打 delta → 推送 → 安装 → 重启」，而实测 6.2.35→6.2.41 那个
增量包 117 MB 里 116 MB 是 ``MAST.exe`` 一个文件 —— 改一行技能和改整个系统在 delta
眼里是同一件事。

覆盖层让 ``<data_root>/config/skill_overlay/builtins/bias.py`` 覆盖
``mast.skills.builtins.bias``：改完点一次「重新加载技能」，**不重启进程**。

四条设计判断（各自的理由在对应模块里）：

* **独立命名空间** ``mast.skills._overlay.*``，不替换 ``sys.modules`` —— 见 paths.py
* **只能收紧不能放松**，比对 raw ↔ raw —— 见 checks.py
* **provenance 只能被观测、不能被声明** —— 见 provenance.py
* **任务运行中整体挂起**，不是只挂起图重建 —— 见 loader.py
"""
