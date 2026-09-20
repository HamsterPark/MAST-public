# -*- coding: utf-8 -*-
"""数据图库出图：从用户标记与系列生成图表。

拉线谱的区组、线和站位由输入系列与几何位置推断；
旋转叠加的晶格种子取自输入首帧的功率谱。没有预置数据或材料参数。

模块：

* :mod:`.common`   目录、figure.json、字体、色表、平面 / 逐行调平、数值导数、帧时间线
* :mod:`.frames`   标记帧对比页（原版 | 逐行调平）
* :mod:`.grids`    网格谱逐层页
* :mod:`.spectra`  谱的读取与 STS 图左栏的 STM 面板
* :mod:`.lines`    拉线谱：站位推断（纯函数）+ 比较 / 均值的热图与瀑布
* :mod:`.stitch`   单根谱宽范围拼接
* :mod:`.series`   旋转系列：16:9 拼图 + 刚性 / 晶格校正叠加
* :mod:`.store`    列出产物、取文件、480 px 预览
* :mod:`.service`  单槽后台任务（同一时刻一个出图任务）+ 同步的 ``run_job``（CLI 与测试用）

路由 ``api/routes/gallery_figures.py`` 只调 ``store`` / ``service`` / ``lines.plan_from_store``。
本包不在 import 时拖入 matplotlib / scipy / Pillow。
"""
