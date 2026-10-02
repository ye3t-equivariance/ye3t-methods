"""Atomic-base normalization and cache records.

This module keeps the ordinary ``A``-basis normalization contract explicit:
raw atomic-base values are stored separately from normalized values, and
normalization derivatives are available for force/Jacobian chain rules.
"""

import hashlib
import json
import math

import numpy as np

from ye3t_ace._record import recordclass, record_replace

try:  # pragma: no cover - torch is a required runtime dependency, but keep imports local-friendly.
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ye3t_ace import __version__ as _YE3T_ACE_VERSION
except Exception:  # pragma: no cover - source-tree fallback
    _YE3T_ACE_VERSION = "unknown"


def _json_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_fingerprint(value):
    if torch is not None and torch.is_tensor(value):
        detached = value.detach().contiguous().cpu()
        bytes_value = np.ascontiguousarray(detached.numpy()).view(np.uint8).tobytes()
        return {
            "kind": "torch.Tensor",
            "shape": tuple(int(x) for x in value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": hashlib.sha256(bytes_value).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "numpy.ndarray",
            "shape": tuple(int(x) for x in array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return None


def _fingerprint(value):
    tensor = _tensor_fingerprint(value)
    if tensor is not None:
        return tensor
    if isinstance(value, dict):
        return {str(k): _fingerprint(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return tuple(_fingerprint(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "repr": repr(value)}


def _value_memory_bytes(value):
    if value is None:
        return 0
    if torch is not None and torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_value_memory_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_memory_bytes(v) for v in value)
    return 0


def _constant_like(raw_values, value):
    if torch is not None and torch.is_tensor(raw_values):
        return torch.zeros_like(raw_values) + torch.as_tensor(value, dtype=raw_values.dtype, device=raw_values.device)
    if isinstance(raw_values, np.ndarray):
        return np.zeros_like(raw_values) + np.asarray(value, dtype=raw_values.dtype)
    return value


@recordclass(('name', 'applies_to', 'bounds', 'is_affine', 'scale', 'shift', 'raw_bounds', 'convention_hash'), frozen = True)
class NormalizationMap:
    """Elementwise normalization map for raw atomic-base values.

    ``forward`` computes normalized values. ``derivative`` returns
    ``d normalized / d raw`` with the same broadcast shape as the input.
    """
    applies_to = "A"
    bounds = (-math.inf, math.inf)
    is_affine = True
    scale = 1.0
    shift = 0.0
    raw_bounds = None
    convention_hash = None

    def __post_init__(self):
        applies_to = str(self.applies_to)
        if applies_to not in {"A", "A_s", "site_basis"}:
            raise ValueError("NormalizationMap.applies_to must be one of A, A_s, or site_basis")
        if not math.isfinite(float(self.scale)) or float(self.scale) == 0.0:
            raise ValueError("NormalizationMap.scale must be finite and nonzero")
        if not math.isfinite(float(self.shift)):
            raise ValueError("NormalizationMap.shift must be finite")
        if self.raw_bounds is not None:
            lo, hi = (float(self.raw_bounds[0]), float(self.raw_bounds[1]))
            if not lo < hi:
                raise ValueError("NormalizationMap.raw_bounds must be an increasing pair")
            object.__setattr__(self, "raw_bounds", (lo, hi))
        object.__setattr__(self, "applies_to", applies_to)
        object.__setattr__(self, "bounds", (float(self.bounds[0]), float(self.bounds[1])))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "shift", float(self.shift))
        if self.convention_hash is None:
            payload = {
                "name": str(self.name),
                "applies_to": applies_to,
                "bounds": self.bounds,
                "is_affine": bool(self.is_affine),
                "scale": float(self.scale),
                "shift": float(self.shift),
                "raw_bounds": self.raw_bounds,
            }
            object.__setattr__(self, "convention_hash", _json_hash(payload))

    @classmethod
    def identity(cls, *, applies_to = "A"):
        return cls(name="identity", applies_to=applies_to, bounds=(-math.inf, math.inf), scale=1.0, shift=0.0)

    @classmethod
    def affine(
        cls,
        *,
        scale,
        shift = 0.0,
        bounds = (-math.inf, math.inf),
        applies_to = "A",
        name = "affine",
    ):
        return cls(name=name, applies_to=applies_to, bounds=bounds, scale=scale, shift=shift)

    @classmethod
    def minus_one_one(
        cls,
        raw_bounds,
        *,
        applies_to = "A",
        name = "minus_one_one",
    ):
        lo, hi = (float(raw_bounds[0]), float(raw_bounds[1]))
        if not lo < hi:
            raise ValueError("raw_bounds must be an increasing pair")
        scale = 2.0 / (hi - lo)
        shift = -1.0 - scale * lo
        return cls(
            name=name,
            applies_to=applies_to,
            bounds=(-1.0, 1.0),
            scale=scale,
            shift=shift,
            raw_bounds=(lo, hi),
        )

    def forward(self, raw_values):
        if not self.is_affine:
            raise NotImplementedError("Non-affine NormalizationMap.forward needs an explicit backend implementation")
        return raw_values * self.scale + self.shift

    def derivative(self, raw_values):
        if not self.is_affine:
            raise NotImplementedError("Non-affine NormalizationMap.derivative needs an explicit backend implementation")
        return _constant_like(raw_values, self.scale)

    def metadata(self):
        return {
            "name": str(self.name),
            "applies_to": str(self.applies_to),
            "bounds": tuple(float(x) for x in self.bounds),
            "is_affine": bool(self.is_affine),
            "raw_bounds": None if self.raw_bounds is None else tuple(float(x) for x in self.raw_bounds),
            "convention_hash": str(self.convention_hash),
        }


@recordclass(('cache_key', 'normalization', 'neighbor_list', 'edge_vectors', 'edge_lengths', 'edge_directions', 'radial_values', 'radial_derivatives', 'radial_convention', 'angular_values', 'angular_derivatives', 'angular_convention', 'phi_values', 'phi_derivatives', 'A_raw', 'A_normalized', 'dAraw_dR', 'dAnorm_dAraw', 'dtype', 'device', 'convention_hash', 'hits', 'misses'), frozen = True)
class AtomicBaseCache:
    """Raw and normalized ordinary atomic-base cache entry."""
    neighbor_list = None
    edge_vectors = None
    edge_lengths = None
    edge_directions = None
    radial_values = None
    radial_derivatives = None
    radial_convention = None
    angular_values = None
    angular_derivatives = None
    angular_convention = None
    phi_values = None
    phi_derivatives = None
    A_raw = None
    A_normalized = None
    dAraw_dR = None
    dAnorm_dAraw = None
    dtype = None
    device = None
    convention_hash = None
    hits = 0
    misses = 0

    def __post_init__(self):
        if self.A_raw is not None and self.A_normalized is not None:
            if tuple(getattr(self.A_raw, "shape", ())) != tuple(getattr(self.A_normalized, "shape", ())):
                raise ValueError("A_raw and A_normalized must have matching shapes")
        if self.dAnorm_dAraw is not None and self.A_raw is not None:
            if tuple(getattr(self.dAnorm_dAraw, "shape", ())) != tuple(getattr(self.A_raw, "shape", ())):
                raise ValueError("dAnorm_dAraw must have the same shape as A_raw")
        if self.convention_hash is None:
            object.__setattr__(self, "convention_hash", self.normalization.convention_hash)

    @classmethod
    def key(
        cls,
        *,
        neighbor_list,
        site_basis,
        radial_basis,
        angular_convention,
        dtype,
        device,
        normalization_convention,
        package_version_hash = None,
    ):
        payload = {
            "neighbor_list": _fingerprint(neighbor_list),
            "site_basis": _fingerprint(site_basis),
            "radial_basis": _fingerprint(radial_basis),
            "angular_convention": _fingerprint(angular_convention),
            "dtype": str(dtype),
            "device": str(device),
            "normalization_convention": _fingerprint(normalization_convention),
            "package_version_hash": package_version_hash or _json_hash({"ye3t-ace": _YE3T_ACE_VERSION}),
        }
        return _json_hash(payload)

    @classmethod
    def from_raw(
        cls,
        *,
        raw_values,
        normalization,
        cache_key,
        **kwargs,
    ):
        return cls(
            cache_key=cache_key,
            normalization=normalization,
            A_raw=raw_values,
            A_normalized=normalization.forward(raw_values),
            dAnorm_dAraw=normalization.derivative(raw_values),
            **kwargs,
        )

    def with_stats(self, *, hits = None, misses = None):
        return record_replace(
            self,
            hits=self.hits if hits is None else int(hits),
            misses=self.misses if misses is None else int(misses),
        )

    def normalized_position_derivative(self):
        if self.dAraw_dR is None:
            raise ValueError("dAraw_dR is not stored in this AtomicBaseCache")
        if self.dAnorm_dAraw is None:
            raise ValueError("dAnorm_dAraw is not stored in this AtomicBaseCache")
        derivative = self.dAnorm_dAraw
        while len(getattr(derivative, "shape", ())) < len(getattr(self.dAraw_dR, "shape", ())):
            derivative = derivative[..., None]
        return derivative * self.dAraw_dR

    def memory_size_bytes(self):
        return sum(
            _value_memory_bytes(value)
            for value in (
                self.edge_vectors,
                self.edge_lengths,
                self.edge_directions,
                self.radial_values,
                self.radial_derivatives,
                self.angular_values,
                self.angular_derivatives,
                self.phi_values,
                self.phi_derivatives,
                self.A_raw,
                self.A_normalized,
                self.dAraw_dR,
                self.dAnorm_dAraw,
            )
        )

    def report(self):
        return {
            "cache_key": str(self.cache_key),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "memory_size_bytes": int(self.memory_size_bytes()),
            "dtype": self.dtype,
            "device": self.device,
            "convention_hash": str(self.convention_hash),
            "normalization": self.normalization.metadata(),
            "radial_convention": _fingerprint(self.radial_convention),
            "angular_convention": _fingerprint(self.angular_convention),
            "stores_raw_atomic_base": self.A_raw is not None,
            "stores_normalized_atomic_base": self.A_normalized is not None,
            "stores_normalization_derivative": self.dAnorm_dAraw is not None,
        }


class DescriptorRuntimeCache:
    """Shared in-memory descriptor/runtime cache with per-family hit reports."""

    def __init__(self, max_entries_per_family=128):
        self.max_entries_per_family = int(max_entries_per_family)
        self._stores = {}
        self._hits = {}
        self._misses = {}
        self._writes = {}

    def _store(self, family):
        family = str(family)
        if family not in self._stores:
            self._stores[family] = {}
            self._hits[family] = 0
            self._misses[family] = 0
            self._writes[family] = 0
        return self._stores[family]

    def get(self, family, key):
        store = self._store(family)
        if key in store:
            self._hits[str(family)] += 1
            return store[key]
        self._misses[str(family)] += 1
        return None

    def put(self, family, key, value):
        store = self._store(family)
        family = str(family)
        store[key] = value
        self._writes[family] += 1
        if self.max_entries_per_family > 0:
            while len(store) > self.max_entries_per_family:
                oldest = next(iter(store))
                del store[oldest]
        return value

    def get_or_build(self, family, key, builder):
        cached = self.get(family, key)
        if cached is not None:
            return cached, True
        value = builder()
        self.put(family, key, value)
        return value, False

    def family_report(self, family):
        family = str(family)
        store = self._store(family)
        return {
            "family": family,
            "entries": int(len(store)),
            "hits": int(self._hits.get(family, 0)),
            "misses": int(self._misses.get(family, 0)),
            "writes": int(self._writes.get(family, 0)),
        }

    def report(self):
        families = sorted(self._stores)
        return {
            "cache_kind": "DescriptorRuntimeCache",
            "max_entries_per_family": int(self.max_entries_per_family),
            "families": {family: self.family_report(family) for family in families},
            "total_entries": int(sum(len(store) for store in self._stores.values())),
            "total_hits": int(sum(self._hits.values())),
            "total_misses": int(sum(self._misses.values())),
            "total_writes": int(sum(self._writes.values())),
        }


class EvaluationContext:
    """Evaluation-scoped shared caches and runtime provenance."""

    def __init__(
        self,
        *,
        runtime_cache=None,
        geometry=None,
        cell=None,
        cutoff=None,
        basis_spec=None,
        dtype=None,
        device=None,
        backend=None,
        version=None,
    ):
        self.runtime_cache = runtime_cache if runtime_cache is not None else DescriptorRuntimeCache()
        self.geometry = geometry
        self.cell = cell
        self.cutoff = cutoff
        self.basis_spec = basis_spec
        self.dtype = dtype
        self.device = device
        self.backend = backend
        self.version = version or _YE3T_ACE_VERSION

    def key_payload(self):
        return {
            "geometry": _fingerprint(self.geometry),
            "cell": _fingerprint(self.cell),
            "cutoff": _fingerprint(self.cutoff),
            "basis_spec": _fingerprint(self.basis_spec),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "backend": str(self.backend),
            "version": str(self.version),
        }

    def cache_key(self, family, extra=None):
        payload = self.key_payload()
        payload["family"] = str(family)
        payload["extra"] = _fingerprint({} if extra is None else extra)
        return _json_hash(payload)

    def report(self):
        return {
            "context_kind": "EvaluationContext",
            "cache_key": self.cache_key("evaluation_context"),
            "key_payload": self.key_payload(),
            "runtime_cache": self.runtime_cache.report(),
        }
