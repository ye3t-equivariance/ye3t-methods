"""Compatibility shim for ``ye3t_methods.atomistic.ace.yace``."""

try:
    from .ace.yace import *
except ImportError:  # pragma: no cover - direct local-module fallback
    from ye3t_methods.atomistic.ace.yace import *
