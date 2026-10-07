"""Read-only module aliases for models saved under the historical name.

The implementation lives in ye3t_methods.atomistic. This package keeps old
Torch pickle module names importable without making new code depend on them.
"""

import importlib
import importlib.abc
import importlib.util
import sys

from ye3t_methods import atomistic as _implementation


class _SavedModelModuleAliases(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("ye3t_ace."):
            return None
        replacement = "ye3t_methods.atomistic." + fullname[len("ye3t_ace."):]
        try:
            original_spec = importlib.util.find_spec(replacement)
        except (ImportError, ModuleNotFoundError):
            return None
        if original_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, self,
            is_package=original_spec.submodule_search_locations is not None,
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        replacement = "ye3t_methods.atomistic." + module.__name__[len("ye3t_ace."):]
        source = importlib.import_module(replacement)
        module.__doc__ = source.__doc__
        module.__getattr__ = lambda name: getattr(source, name)
        module.__dir__ = lambda: sorted(set(module.__dict__) | set(dir(source)))


if not any(isinstance(finder, _SavedModelModuleAliases) for finder in sys.meta_path):
    sys.meta_path.insert(0, _SavedModelModuleAliases())

__version__ = _implementation.__version__


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
