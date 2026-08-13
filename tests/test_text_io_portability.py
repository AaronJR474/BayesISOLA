import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "BayesISOLA"


def _text_write_open_calls_without_encoding(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "open":
            continue

        mode = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        for keyword in node.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                mode = keyword.value.value

        if not isinstance(mode, str) or "b" in mode or not any(flag in mode for flag in "wa+"):
            continue
        if not any(keyword.arg == "encoding" for keyword in node.keywords):
            failures.append(node.lineno)
    return failures


def test_text_writes_use_explicit_encoding():
    failures = {}
    for path in PACKAGE.glob("*.py"):
        lines = _text_write_open_calls_without_encoding(path)
        if lines:
            failures[path.name] = lines

    assert failures == {}


def test_no_nonrecursive_os_mkdir_remains_in_package():
    offenders = []
    for path in PACKAGE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "os.mkdir(" in text:
            offenders.append(path.name)

    assert offenders == []
