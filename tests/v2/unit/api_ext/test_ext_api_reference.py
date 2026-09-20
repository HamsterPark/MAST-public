"""``docs/external/{zh,en}/07-api-reference.md`` 与外部面的 OpenAPI 一致。

参考页由 ``scripts/gen_ext_api_reference.py`` 从 API 自己的 OpenAPI 生成；改了端点、
参数或请求体却没重新生成，这里变红。新端点没配中英说明，生成本身就失败。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

_GEN = _REPO / "scripts" / "gen_ext_api_reference.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_ext_api_reference", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def openapi(gen):
    return gen.ext_openapi()


def test_the_reference_pages_match_the_api(gen, openapi):
    docs = gen.build(openapi)
    for lang, text in docs.items():
        path = gen.OUT[lang]
        assert path.is_file(), f"{path} 不存在 —— 运行 python scripts/gen_ext_api_reference.py"
        assert path.read_bytes().decode("utf-8") == text, (
            f"{path.relative_to(_REPO)} 与 API 不一致 —— 运行 python scripts/gen_ext_api_reference.py")


def test_every_endpoint_is_documented_in_both_languages(gen, openapi):
    docs = gen.build(openapi)
    for (method, path) in gen.ENDPOINTS:
        heading = f"### `{method} {path}`"
        assert heading in docs["zh"] and heading in docs["en"], heading
    assert len(gen.ENDPOINTS) == sum(len(ops) for ops in openapi["paths"].values())


def test_an_undocumented_endpoint_fails_the_generation(gen, openapi):
    """守卫的守卫：多一个没有说明的端点，生成必须失败，而不是静默漏掉它。"""
    import copy

    spec = copy.deepcopy(openapi)
    spec["paths"]["/brand-new"] = {"get": {"responses": {}}}
    with pytest.raises(SystemExit, match="brand-new"):
        gen.build(spec)


def test_a_changed_parameter_changes_the_page(gen, openapi):
    import copy

    spec = copy.deepcopy(openapi)
    op = spec["paths"]["/jobs/{job_id}"]["get"]
    for p in op["parameters"]:
        if p["name"] == "wait_s":
            p["schema"]["maximum"] = 99.0
    assert gen.build(spec)["en"] != gen.build(openapi)["en"]
