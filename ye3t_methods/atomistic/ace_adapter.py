"""Compatibility shim for ``ye3t_methods.atomistic.ace.adapter``."""

try:
    from .ace.adapter import *
except ImportError:  # pragma: no cover - direct local-module fallback
    from ye3t_methods.atomistic.ace.adapter import *
