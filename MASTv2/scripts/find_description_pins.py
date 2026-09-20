"""找出所有「把字符串钉在 description 里」的断言，含否定形 not in。

按 AST 找 Compare(In/NotIn)，左边是字符串常量、右边的表达式里出现 description
（允许 .lower()/.upper()/or ""/变量名叫 d/desc/spec.description 等）。
"""
import ast, sys
from pathlib import Path
#: 仓库自洽：从本文件位置推出 tests/v2，别写死绝对路径
ROOT = Path(__file__).resolve().parents[2] / "tests" / "v2"

def mentions_desc(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == "description":
            return True
        if isinstance(n, ast.Name) and n.id in ("d", "desc", "descr", "summary"):
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == "description":
            return True
    return False

rows = []
for p in sorted(ROOT.rglob("test_*.py")):
    try:
        src = p.read_text(encoding="utf-8"); tree = ast.parse(src)
    except Exception:
        continue
    if "description" not in src:
        continue
    for n in ast.walk(tree):
        if not isinstance(n, ast.Compare) or len(n.ops) != 1:
            continue
        op = n.ops[0]
        if not isinstance(op, (ast.In, ast.NotIn)):
            continue
        left, right = n.left, n.comparators[0]
        if not (isinstance(left, ast.Constant) and isinstance(left.value, str)):
            continue
        if not mentions_desc(right):
            continue
        rows.append((str(p.relative_to(ROOT)), n.lineno,
                     "not in" if isinstance(op, ast.NotIn) else "in",
                     left.value, ast.unparse(right)[:70]))
print(f"钉在 description 上的断言：{len(rows)} 条")
for f, ln, op, s, r in rows:
    print(f"  {f}:{ln}  {s!r} {op} {r}")
