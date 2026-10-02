
"""Reusable caches for descriptor-label and descriptor-artifact construction."""

from collections import OrderedDict
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
import pickle

from ye3t_ace.equivariant_calc.ace_eval_v2 import GeneralizedCouplingLibrary
from ye3t_ace.equivariant_calc.labeling import CompactLabel, normalize_compact_label
from ye3t_ace._record import recordclass


DESCRIPTOR_BUILD_CACHE_SCHEMA = "descriptor_build_cache_v1"
DESCRIPTOR_BUILD_CACHE_MANIFEST_FORMAT = "descriptor_build_cache_manifest_v1"


def _normalize_cache_dir(cache_dir):
    if cache_dir is None:
        return None
    text = str(cache_dir).strip()
    if not text:
        return None
    return Path(cache_dir)


def _stable_key_digest(value):
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _stable_manifest_hash(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "source"


def _disk_cache_path(cache_dir, family, key):
    root = _normalize_cache_dir(cache_dir)
    if root is None:
        return None
    return root / "descriptor_build_cache" / str(family) / (_stable_key_digest(key) + ".pkl")


def _read_disk_cache(cache_dir, family, key):
    path = _disk_cache_path(cache_dir, family, key)
    if path is None or not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("cache_schema") != DESCRIPTOR_BUILD_CACHE_SCHEMA:
        return None
    if payload.get("key_digest") != _stable_key_digest(key):
        return None
    return payload.get("value")


def _write_disk_cache(cache_dir, family, key, value):
    path = _disk_cache_path(cache_dir, family, key)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            {
                "cache_schema": DESCRIPTOR_BUILD_CACHE_SCHEMA,
                "family": str(family),
                "key_digest": _stable_key_digest(key),
                "value": value,
            },
            handle,
            protocol=4,
        )
    os.replace(tmp_path, path)
    write_descriptor_build_cache_manifest(cache_dir)
    return path


def default_descriptor_build_cache_manifest_path(cache_dir):
    """Return the default manifest path for a descriptor build cache."""
    root = _normalize_cache_dir(cache_dir)
    if root is None:
        raise ValueError("A descriptor cache directory is required for a cache manifest.")
    return root / "descriptor_build_cache_manifest.json"


def _descriptor_cache_entry_from_file(root, path):
    relative_path = path.relative_to(root).as_posix()
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "family": str(path.parent.name),
        "key_digest": str(path.stem),
        "cache_file": relative_path,
        "payload_hash": content_hash,
        "payload_bytes": int(path.stat().st_size),
    }


def _descriptor_cache_manifest_payload(cache_dir):
    root = _normalize_cache_dir(cache_dir)
    if root is None:
        raise ValueError("A descriptor cache directory is required for a cache manifest.")
    cache_root = root / "descriptor_build_cache"
    entries = []
    if cache_root.exists():
        entries = [
            _descriptor_cache_entry_from_file(root, path)
            for path in sorted(cache_root.glob("*/*.pkl"), key=lambda item: item.as_posix())
            if path.is_file()
        ]
    families = OrderedDict()
    for entry in entries:
        family = str(entry["family"])
        current = families.get(family, {"entry_count": 0, "payload_bytes": 0})
        current["entry_count"] = int(current["entry_count"]) + 1
        current["payload_bytes"] = int(current["payload_bytes"]) + int(entry["payload_bytes"])
        families[family] = current
    payload = {
        "format": DESCRIPTOR_BUILD_CACHE_MANIFEST_FORMAT,
        "manifest_scope": "package_cache",
        "cache_kind": "descriptor_build_cache",
        "cache_schema": DESCRIPTOR_BUILD_CACHE_SCHEMA,
        "package_versions": {
            "ye3t": _package_version("ye3t"),
            "ye3t-ace": _package_version("ye3t-ace"),
        },
        "source_identifiers": {
            "descriptor_cache": "ye3t_ace.cache.descriptors",
            "coupling_library": GeneralizedCouplingLibrary.__name__,
            "compact_label": CompactLabel.__name__,
        },
        "families": dict(families),
        "entry_count": int(len(entries)),
        "entries": entries,
    }
    payload["manifest_hash"] = _stable_manifest_hash(payload)
    return payload


def write_descriptor_build_cache_manifest(cache_dir):
    """Write a portable manifest for descriptor-label and coupling-artifact cache files."""
    path = default_descriptor_build_cache_manifest_path(cache_dir)
    payload = _descriptor_cache_manifest_payload(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)
    result = dict(payload)
    result["manifest_path"] = str(path)
    return result


def load_descriptor_build_cache_manifest(cache_dir):
    """Load a descriptor build-cache manifest."""
    path = default_descriptor_build_cache_manifest_path(cache_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def settings_cache_key(settings):
    """Return an immutable key for descriptor-generation settings."""
    return (
        tuple(int(x) for x in settings.ranks),
        str(settings.basis_type),
        tuple(str(x) for x in settings.elems),
        tuple(int(x) for x in settings.nmax),
        tuple(int(x) for x in settings.lmax),
        tuple(int(x) for x in settings.lmin),
        int(settings.L_R),
        tuple(int(x) for x in settings.M_R_values),
        int(settings.k_o_max),
        tuple(int(x) for x in settings.k_max),
        int(settings.aux_lmax),
        None if settings.max_labels_per_rank is None else int(settings.max_labels_per_rank),
        str(settings.tree_type),
        str(getattr(settings, "parity_filter", "natural")),
    )


def compact_label_cache_key(
    compact_labels,
):
    """Return an immutable key for a compact-label list."""
    normalized = [normalize_compact_label(label) for label in compact_labels]
    return tuple(
        (
            tuple(int(x) for x in label.n_tuple),
            tuple(int(x) for x in label.l_tuple),
            tuple(label.internal_Ls),
            str(label.tree_type),
            tuple(label.basis_key),
        )
        for label in normalized
    )


def optional_int_tuple(values):
    if values is None:
        return None
    return tuple(int(x) for x in values)


def optional_float(value):
    if value is None:
        return None
    return float(value)


@recordclass(('settings', 'basis_mode', 'exact_primitive_timeout_seconds'), frozen = True)
class DescriptorEnumerationCacheKey:
    """Key for compact-label enumeration."""


@recordclass(('settings', 'compact_labels', 'basis_mode', 'exact_primitive_timeout_seconds', 'center_mu_values', 'restrict_neighbor_mu', 'max_variants_per_label', 'scalar_coordinate_compiler'), frozen = True)
class DescriptorArtifactCacheKey:
    """Key for compiled descriptor artifacts."""


def descriptor_artifact_cache_key(
    settings,
    compact_labels,
    *,
    basis_mode,
    exact_primitive_timeout_seconds,
    center_mu_values,
    restrict_neighbor_mu,
    max_variants_per_label,
    scalar_coordinate_compiler=None,
):
    return DescriptorArtifactCacheKey(
        settings=settings_cache_key(settings),
        compact_labels=compact_label_cache_key(compact_labels),
        basis_mode=None if basis_mode is None else str(basis_mode),
        exact_primitive_timeout_seconds=optional_float(exact_primitive_timeout_seconds),
        center_mu_values=optional_int_tuple(center_mu_values),
        restrict_neighbor_mu=optional_int_tuple(restrict_neighbor_mu),
        max_variants_per_label=None if max_variants_per_label is None else int(max_variants_per_label),
        scalar_coordinate_compiler=(
            "scalar_coordinate_compiler_v1:missing_only:python"
            if scalar_coordinate_compiler is None
            else "scalar_coordinate_compiler_v1:"
            + json.dumps(
                scalar_coordinate_compiler,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    )


class DescriptorBuildCache:
    """Shared cache for compact-label enumeration and compiled descriptor artifacts."""

    def __init__(self, max_compact_label_entries = 64, max_artifact_entries = 16, cache_dir = None):
        self.max_compact_label_entries = int(max_compact_label_entries)
        self.max_artifact_entries = int(max_artifact_entries)
        self.cache_dir = _normalize_cache_dir(cache_dir)
        self._compact_labels = OrderedDict()
        self._artifacts = OrderedDict()
        self.compact_label_hits = 0
        self.compact_label_misses = 0
        self.artifact_hits = 0
        self.artifact_misses = 0
        self.compact_label_disk_hits = 0
        self.compact_label_disk_misses = 0
        self.compact_label_disk_writes = 0
        self.artifact_disk_hits = 0
        self.artifact_disk_misses = 0
        self.artifact_disk_writes = 0

    def clear(self):
        self._compact_labels.clear()
        self._artifacts.clear()
        self.compact_label_hits = 0
        self.compact_label_misses = 0
        self.artifact_hits = 0
        self.artifact_misses = 0
        self.compact_label_disk_hits = 0
        self.compact_label_disk_misses = 0
        self.compact_label_disk_writes = 0
        self.artifact_disk_hits = 0
        self.artifact_disk_misses = 0
        self.artifact_disk_writes = 0

    def _remember_compact_labels(self, key, value):
        if self.max_compact_label_entries <= 0:
            return
        self._compact_labels.pop(key, None)
        self._compact_labels[key] = value
        while len(self._compact_labels) > self.max_compact_label_entries:
            self._compact_labels.popitem(last=False)

    def _remember_artifacts(self, key, value):
        if self.max_artifact_entries <= 0:
            return
        self._artifacts.pop(key, None)
        self._artifacts[key] = value
        while len(self._artifacts) > self.max_artifact_entries:
            self._artifacts.popitem(last=False)

    def get_compact_labels(self, key):
        item = self._compact_labels.get(key)
        if item is None:
            self.compact_label_misses += 1
            disk_item = _read_disk_cache(self.cache_dir, "compact_labels", key)
            if disk_item is not None:
                self.compact_label_disk_hits += 1
                self._remember_compact_labels(key, disk_item)
                return disk_item
            if self.cache_dir is not None:
                self.compact_label_disk_misses += 1
            return None
        self.compact_label_hits += 1
        self._compact_labels.move_to_end(key)
        return item

    def put_compact_labels(self, key, value):
        self._remember_compact_labels(key, value)
        if self.cache_dir is not None:
            _write_disk_cache(self.cache_dir, "compact_labels", key, value)
            self.compact_label_disk_writes += 1

    def get_artifacts(self, key):
        item = self._artifacts.get(key)
        if item is None:
            self.artifact_misses += 1
            disk_item = _read_disk_cache(self.cache_dir, "artifacts", key)
            if disk_item is not None:
                self.artifact_disk_hits += 1
                self._remember_artifacts(key, disk_item)
                return disk_item
            if self.cache_dir is not None:
                self.artifact_disk_misses += 1
            return None
        self.artifact_hits += 1
        self._artifacts.move_to_end(key)
        return item

    def put_artifacts(
        self,
        key,
        value,
    ):
        self._remember_artifacts(key, value)
        if self.cache_dir is not None:
            _write_disk_cache(self.cache_dir, "artifacts", key, value)
            self.artifact_disk_writes += 1

    def get_stats(self):
        return {
            "compact_label_entries": len(self._compact_labels),
            "artifact_entries": len(self._artifacts),
            "disk_cache_enabled": bool(self.cache_dir is not None),
            "compact_label_hits": int(self.compact_label_hits),
            "compact_label_misses": int(self.compact_label_misses),
            "compact_label_disk_hits": int(self.compact_label_disk_hits),
            "compact_label_disk_misses": int(self.compact_label_disk_misses),
            "compact_label_disk_writes": int(self.compact_label_disk_writes),
            "artifact_hits": int(self.artifact_hits),
            "artifact_misses": int(self.artifact_misses),
            "artifact_disk_hits": int(self.artifact_disk_hits),
            "artifact_disk_misses": int(self.artifact_disk_misses),
            "artifact_disk_writes": int(self.artifact_disk_writes),
        }


def _ace_env(name, default=None):
    return os.getenv("YE3T_ACE_" + str(name), os.getenv("gne3_ace_" + str(name), default))


_SHARED_DESCRIPTOR_BUILD_CACHE = DescriptorBuildCache(
    max_compact_label_entries=int(_ace_env("COMPACT_LABEL_CACHE_MAX_ENTRIES", "64")),
    max_artifact_entries=int(_ace_env("DESCRIPTOR_ARTIFACT_CACHE_MAX_ENTRIES", "16")),
    cache_dir=_ace_env("DESCRIPTOR_BUILD_CACHE_DIR", None),
)


def get_shared_descriptor_build_cache():
    """Return the process-wide descriptor build cache."""
    return _SHARED_DESCRIPTOR_BUILD_CACHE


def resolve_descriptor_build_cache(
    cache,
    *,
    use_cache = True,
):
    """Resolve an explicit or shared descriptor cache."""
    if not use_cache or _ace_env("DISABLE_DESCRIPTOR_CACHES") == "1":
        return None
    if cache is not None:
        return cache
    return get_shared_descriptor_build_cache()
