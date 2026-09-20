# -*- coding: utf-8 -*-
"""出图子包的结构约束：不碰 pyplot / 全局 rc（T7 / T27）、与本体共用绘图锁、契约字面量一致、两种导数窗口分开（T31）。"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import get_args

import numpy as np

import mast
from mast.api.schemas_gallery_figures import FigureCategory, FigureKind
from mast.gallery import render
from mast.gallery.figures import common as C
from mast.gallery.figures import service

PKG = Path(C.__file__).resolve().parent


def test_no_pyplot_and_no_global_rc_in_the_figures_package():
    bad = []
    for py in sorted(PKG.glob("*.py")):
        for node in ast.walk(ast.parse(py.read_text("utf-8"))):
            if isinstance(node, ast.Import):
                bad += [f"{py.name}:{node.lineno} import {a.name}" for a in node.names
                        if "pyplot" in a.name or a.name == "pylab"]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                bad += [f"{py.name}:{node.lineno} from {mod} import {a.name}" for a in node.names
                        if "pyplot" in mod or a.name in ("pyplot", "rcParams", "rc_context", "rc", "rcdefaults")]
            elif isinstance(node, ast.Attribute) and node.attr in ("rcParams", "rc_context", "rcdefaults"):
                bad.append(f"{py.name}:{node.lineno} .{node.attr}")
    assert bad == []


def test_rendering_a_figure_does_not_import_pyplot():
    code = ("import sys\n"
            "from mast.gallery.figures import common as C, frames, grids, lines, series, service, spectra, stitch, store\n"
            "fig = C.new_figure((3, 2)); ax = fig.add_subplot(); spectra.stm_panel(ax, None, [], [], [], 0)\n"
            "ax.set_yscale('symlog', linthresh=1.0); ax.set_title('中文 x₀ D₋', fontproperties=C.fp(9))\n"
            "C.savefig_bytes(fig, 40)\n"
            "print('PYPLOT', 'matplotlib.pyplot' in sys.modules)\n")
    env = dict(os.environ, PYTHONPATH=str(Path(mast.__file__).resolve().parents[1]), PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                         timeout=300, env=env)
    assert out.returncode == 0, out.stderr
    assert "PYPLOT False" in out.stdout


def test_the_figures_share_the_build_lock():
    assert C.MPL_LOCK is render._MPL_LOCK


def test_contract_literals_match_the_implementation():
    assert tuple(get_args(FigureKind)) == service.KINDS
    assert tuple(get_args(FigureCategory)) == tuple(k for k, _ in C.CATEGORIES)
    assert set(C.KIND_CATEGORY) == set(service.KINDS)


def test_the_two_derivative_window_rules_stay_separate():
    """T31：出图按电压定窗口（≈ 0.08 V），缩略图按点数定（≈ n/25）。合并必然改掉其中一个的输出。"""
    V = np.linspace(-2.0, 2.0, 401)
    I = np.tanh(3 * V) + np.random.default_rng(0).normal(0, 0.01, V.size)
    by_volts = C.deriv_by_volts(V, I)
    by_points, _w = render.num_deriv(V, I)
    truth = 3 / np.cosh(3 * V) ** 2
    assert not np.allclose(by_volts, by_points)
    assert np.abs(by_volts - truth).mean() < 0.2 and np.abs(by_points - truth).mean() < 0.2
