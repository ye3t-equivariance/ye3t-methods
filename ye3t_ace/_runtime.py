"""Compatibility shim for ``ye3t_ace.utils.runtime``."""

try:
    from .utils.runtime import *
except ImportError:  # pragma: no cover - direct local-module fallback
    from ye3t_ace.utils.runtime import *
