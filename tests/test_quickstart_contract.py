"""Keep the recommended installed-package examples on the public config API."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "examples"
_EXAMPLES = sorted(ROOT.rglob("*.py"))
_SECTIONS = {"metadata", "basis", "representation", "runtime", "model",
             "targets", "validation"}


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda path: str(path.relative_to(ROOT)))
def test_recommended_quickstart_uses_visible_public_contract(path):
    name = str(path.relative_to(ROOT))
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    configs = [node.value for node in tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "config"
                       for target in node.targets)
               and isinstance(node.value, ast.Dict)]
    assert len(configs) == 1, name
    keys = {key.value for key in configs[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    assert _SECTIONS <= keys <= _SECTIONS | {"extensions"}, name
    metadata = next(value for key, value in zip(configs[0].keys, configs[0].values)
                    if isinstance(key, ast.Constant) and key.value == 'metadata')
    assert isinstance(metadata, ast.Dict), name
    assert {'schema', 'name', 'status'} <= {key.value for key in metadata.keys
                                          if isinstance(key, ast.Constant)}, name
    assert 'sys.path' not in text
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, name
            module = node.module or ''
            assert not module.startswith(('ye3t_ace', 'examples.', 'tools.')), name
            assert not any(part.startswith('_') for part in module.split('.')), name
            assert not any(item.name.startswith('_') for item in node.names), name
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not node.value.startswith(('../ye3t-workflows', '/mnt/', 'C:\\')), name
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {'max_features_per_rank', 'max_labels_per_rank'}:
                    assert isinstance(value, ast.Constant) and value.value is None, name


def test_example_front_doors_are_present():
    names = {path.name for path in _EXAMPLES}
    assert {'ase_descriptors.py', 'tagged_descriptors.py', 'ase_octupole_descriptors.py',
            'density_fit.py', 'tagged_fit.py', 'shell_density_fit.py',
            'combined_density_tagged_fit.py', 'sklearn_fit.py', 'coupled_factors.py',
            'explicit_phi_cluster.py', 'paper_ni_nve.py', 'saved_ase_export.py',
            'per_atom_vector_to_lammps.py', 'paper_ni_60.py',
            'refit_paper_ni.py', 'reproduce_ni_rmse.py'} <= names
