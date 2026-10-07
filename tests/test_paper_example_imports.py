"""The source-archive paper drivers must use exports present in the stable wheel."""

import ast
import importlib
from pathlib import Path


def test_paper_drivers_use_available_ye3t_imports():
    root = Path(__file__).resolve().parents[1] / "examples" / "publication" / "cost_comparison"
    scripts = (root / "run.py", *sorted((root / "workflow").glob("*.py")))
    assert len(scripts) == 18
    for path in scripts:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith(("ye3t_methods", "ye3t.")):
                continue
            module = importlib.import_module(node.module)
            missing = [alias.name for alias in node.names if not hasattr(module, alias.name)]
            assert not missing, f"{path.name}: {node.module} lacks {missing}"
