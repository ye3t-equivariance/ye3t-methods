
"""Compatibility alias package for the canonical :mod:`ye3t.core.basis` API.

The implementation now lives under ``ye3t.core.basis``.  This package keeps
older ``ye3t_ace.ace_labeler`` imports working without maintaining one wrapper
file per submodule.
"""

import importlib
import sys

from ye3t.core import basis as _basis
from ye3t.core.basis import *

__all__ = tuple(_basis.__all__)

_SUBMODULE_ALIASES = {
    "benchmark": "ye3t.core.basis.benchmark",
    "builder": "ye3t.core.basis.builder",
    "characters": "ye3t.core.basis.characters",
    "exact_basis": "ye3t.core.basis.exact_basis",
    "formatters": "ye3t.core.basis.formatters",
    "homogeneous": "ye3t.core.basis.homogeneous",
    "labels": "ye3t.core.basis.labels",
    "lie_nullspace": "ye3t.core.basis.lie_nullspace",
    "metadata": "ye3t.core.basis.metadata",
    "naive_gramian": "ye3t.core.basis.naive_gramian",
    "theory": "ye3t.core.basis.theory",
    "tree": "ye3t.core.basis.tree",
    "validation": "ye3t.core.basis.validation",
    "young_exact": "ye3t.core.basis.young_exact",
}

for _public_name, _target in _SUBMODULE_ALIASES.items():
    _module = importlib.import_module(_target)
    sys.modules[f"{__name__}.{_public_name}"] = _module
    globals()[_public_name] = _module

del importlib, sys, _basis, _public_name, _target, _module
