"""Compatibility shim for ``ye3t_ace.ace.adapter``."""

try:
    from .ace.adapter import *
except ImportError:  # pragma: no cover - direct local-module fallback
    from ye3t_ace.ace.adapter import *
