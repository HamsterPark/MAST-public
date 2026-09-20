# -*- coding: utf-8 -*-
"""数据图库 —— 数据的程序化预处理、展示与人工筛选。


来源是一个独立运行过的图库程序（下称「旧版兼容格式」）：一个增量预处理脚本 + 一个本地存盘服务
+ 一个单页。这里把前两者收编进 MAST：

* :mod:`mast.gallery.inventory` —— 清点数据根里的 .sxm / .dat / .3ds，读头信息摘要；
* :mod:`mast.gallery.render`    —— 缩略图（帧 / lock-in 帧 / 谱 / 网格谱），版式照旧版兼容格式；
* :mod:`mast.gallery.analysis`  —— 自动判据（原子相、超结构）、重复保存、同一次采集的片段；
* :mod:`mast.gallery.index`     —— 前端一次拉取的 ``index.json``；
* :mod:`mast.gallery.build`     —— 把上面串起来的增量构建（后台线程 / CLI）；
* :mod:`mast.gallery.service`   —— 进程内唯一的后台构建；
* :mod:`mast.gallery.marks`     —— 标记（评级 / 标签 / 备注 / 系列 / 谱系于帧），``marks.json`` 唯一真源；
* :mod:`mast.gallery.config`    —— 要索引哪些数据根；
* :mod:`mast.gallery.paths`     —— 状态目录与原子写入。

本包不在 import 时做任何重活，也不 import 本包之外的重依赖（matplotlib / PIL / scipy
都在函数体里惰性导入）：API 路由在请求里才 import 它，它坏了只让图库 degraded。
"""
