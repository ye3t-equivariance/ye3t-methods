"""Compatibility shim for ``ye3t_methods.atomistic.ace.labels``."""

try:
    from .ace.labels import *
except ImportError:  # pragma: no cover - direct local-module fallback
    from ye3t_methods.atomistic.ace.labels import *
