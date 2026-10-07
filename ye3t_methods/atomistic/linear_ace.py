"""Compatibility shim for ``ye3t_methods.atomistic.ace.linear_ace``."""

try:
    from .ace.linear_ace import *
except ImportError:  # pragma: no cover - direct local-module fallback
    from ye3t_methods.atomistic.ace.linear_ace import *
