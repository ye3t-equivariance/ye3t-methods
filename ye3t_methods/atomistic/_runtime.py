"""Compatibility shim for ``ye3t_methods.atomistic.utils.runtime``."""

try:
    from .utils.runtime import *
except ImportError:  # pragma: no cover - direct local-module fallback
    from ye3t_methods.atomistic.utils.runtime import *
