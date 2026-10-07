"""Keep the recommended installed-package examples on the public config API."""

import ast
from pathlib import Path

import pytest


_EXAMPLES = (
    "ase_descriptors.py",
    "ase_octupole_descriptors.py",
    "density_fit.py",
    "combined_density_tagged_fit.py",
    "per_atom_vector_to_lammps.py",
    "paper_ni_portable_ase.py",
    "paper_ni_nve.py",
    "tagged_fit.py",
)
_SECTIONS = {"metadata", "basis", "representation", "runtime", "model",
             "targets", "validation"}


@pytest.mark.parametrize("name", _EXAMPLES)
def test_recommended_quickstart_uses_visible_public_contract(name):
    path = Path(__file__).resolve().parents[1] / "examples" / "quickstart" / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    configs = [node.value for node in tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "config"
                       for target in node.targets)
               and isinstance(node.value, ast.Dict)]
    assert len(configs) == 1, name
    keys = {key.value for key in configs[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    assert keys == _SECTIONS, name
    assert not any(isinstance(node, ast.ImportFrom) and node.module and
                   node.module.startswith("ye3t_methods.atomistic")
                   for node in ast.walk(tree)), name
