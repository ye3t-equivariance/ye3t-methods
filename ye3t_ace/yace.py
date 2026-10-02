"""Compatibility shim for ``ye3t_ace.ace.yace``."""

try:
    from .ace.yace import *
except ImportError:  # pragma: no cover - direct local-module fallback
    from ye3t_ace.ace.yace import *
