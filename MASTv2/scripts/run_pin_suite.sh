#!/usr/bin/env bash
# 「针套件」：只跑那些把字符串**钉在 description 里**的测试文件。
#
# ## 为什么要它
#
# 2026-08-24/25 把 457 个技能描述中文化时发现的：`pytest tests/v2/unit/skills`
# 要 **26 分钟**，而且**盖不住全部的针** —— 全树 32 个含针的测试文件里，
# **有 15 个在 `tests/v2/unit/skills` 之外**（`unit/agents/`、`unit/core/`、
# `unit/knowledge/`、`agents/literature/`、`skills/composite/`、
# `unit/test_wrap_skill_minimal.py`）。
#
# `test_wrap_skill_minimal.py` 就是这么漏掉的：它红过，而那次 26 分钟的技能全量
# **看不见它**。改任何技能描述之后，跑这个套件比跑技能全量更该先做 —— 约 70 秒。
#
# ## 什么是「针」
#
# `assert "some english" in skill.metadata().description` 这类断言。
# 它们拿**措辞**当**规则**的代理，所以：改语言必红（规则没动），
# 而规则被删、那个词恰好还在别处出现时又不红。两个方向都不可靠。
#
# 碰到红的，正确修法是**把针换成承载同一主张的中文串**（断言强度不变），
# 而不是删断言。否定形的针（`not in`）尤其要注意：描述译成中文之后，
# 一条只挡英文的 `not in` 就**永远打不响了** —— 那句假话真要回来，
# 回来的会是中文版。这时要**补上中文那一侧，别删英文那一侧**。
#
# ## 用法
#
#     bash MASTv2/scripts/run_pin_suite.sh
#
# 文件清单**每次从 AST 现算**（`find_description_pins.py`），不写死 ——
# 写死的清单会在下一个人加测试时静默失效。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv-v2-py313/Scripts/python.exe"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

cd "$REPO"

# 1) AST 找出所有 `"x" in/not in ...description...` 形状的断言所在文件
"$PY" "$HERE/find_description_pins.py" \
  | grep -oE '^  [A-Za-z0-9_\\./-]+\.py' | sed 's/^  //' \
  | tr '\\' '/' | sed 's#^#tests/v2/#' | sort -u > "$LIST"

N=$(wc -l < "$LIST")
echo "含 description 针的测试文件：$N 个"
echo "其中不在 tests/v2/unit/skills 下（技能全量盖不住的）："
grep -v '^tests/v2/unit/skills/' "$LIST" | sed 's/^/    /' || true
echo

# 2) 只跑这些文件
PYTHONPATH="$REPO/MASTv2" "$PY" -m pytest $(tr '\n' ' ' < "$LIST") \
  -q -p no:cacheprovider --no-header
