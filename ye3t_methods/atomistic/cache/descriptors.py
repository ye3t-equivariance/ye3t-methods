
"""Reusable caches for descriptor-label and descriptor-artifact construction."""

from collections import OrderedDict
from collections.abc import Mapping
from io import BytesIO
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
import re
import sys
import uuid
import zipfile

import numpy as np

from ye3t.cache.artifacts import (
    ARTIFACT_STORE_SCHEMA,
    ArtifactCacheMiss,
    ArtifactCacheValidationError,
    YE3TArtifactStore,
    _ArtifactLock,
    artifact_hash,
)

from ye3t_methods.atomistic.equivariant_calc.ace_eval_v2 import GeneralizedCouplingLibrary
from ye3t_methods.atomistic.equivariant_calc.labeling import CompactLabel, normalize_compact_label
from ye3t_methods.atomistic._record import recordclass


DESCRIPTOR_BUILD_CACHE_SCHEMA = "descriptor_build_cache_v3"
DESCRIPTOR_BUILD_CACHE_MANIFEST_FORMAT = "descriptor_build_cache_manifest_v3"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARRAY_NAME = re.compile(r"^a[0-9]+_[mc]$")
_SETTINGS_FIELDS = (
    "ranks", "basis_type", "elems", "nmax", "lmax", "lmin", "L_R",
    "M_R_values", "k_o_max", "k_max", "aux_lmax", "max_labels_per_rank",
    "tree_type", "parity_filter",
)


def _normalize_cache_dir(cache_dir):
    if cache_dir is None:
        return None
    text = str(cache_dir).strip()
    if not text:
        return None
    return Path(cache_dir)


def _stable_manifest_hash(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "source-tree" if name == "ye3t" else "source"


def _cache_mode():
    mode = str(os.environ.get("YE3T_CACHE_MODE", "auto")).strip().lower()
    if mode == "refresh":
        return "rebuild"
    if mode not in {"auto", "read_only", "rebuild", "off"}:
        raise ValueError("Descriptor cache mode must be auto, read_only, refresh, rebuild, or off.")
    return mode


def _request_for_key(family, key):
    return {
        "family": str(family),
        "key": {
            name: getattr(key, name)
            for name in key.__record_fields__
        },
    }


def _store(cache_dir, mode):
    return YE3TArtifactStore(directory=cache_dir, mode=mode, verify="full")


def _safe_data(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise TypeError("Descriptor cache accepts only numeric arrays.")
        return _safe_data(value.tolist())
    if isinstance(value, np.generic):
        return _safe_data(value.item())
    if isinstance(value, complex):
        return {"__ye3t_complex__": [float(value.real), float(value.imag)]}
    if isinstance(value, tuple):
        return {"__ye3t_tuple__": [_safe_data(item) for item in value]}
    if isinstance(value, list):
        return [_safe_data(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) or name.startswith("__ye3t_")
               for name in value):
            raise TypeError("Descriptor cache metadata requires string keys.")
        return {name: _safe_data(item) for name, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported descriptor cache value {type(value).__name__}.")


def _restore_data(value, arrays=None, references=None):
    if isinstance(value, list):
        return [_restore_data(item, arrays, references) for item in value]
    if isinstance(value, dict):
        if set(value) in ({"__ye3t_npz__"}, {"__ye3t_npz__", "decode"}):
            key = value["__ye3t_npz__"]
            if arrays is None or key not in arrays:
                raise ValueError("Descriptor cache has an unbound numeric array.")
            references.add(key)
            if "decode" in value:
                if (value["decode"] != "complex_pairs" or
                        arrays[key].ndim != 2 or arrays[key].shape[1] != 2 or
                        arrays[key].dtype != np.dtype("float64")):
                    raise ValueError("Descriptor cache complex array shape differs.")
                return [complex(real, imag) for real, imag in arrays[key]]
            return arrays[key].tolist()
        if set(value) == {"__ye3t_complex__"}:
            pair = value["__ye3t_complex__"]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Malformed complex descriptor-cache value.")
            return complex(*pair)
        if set(value) == {"__ye3t_tuple__"}:
            items = value["__ye3t_tuple__"]
            if not isinstance(items, list):
                raise ValueError("Malformed tuple descriptor-cache value.")
            return tuple(_restore_data(item, arrays, references) for item in items)
        return {name: _restore_data(item, arrays, references)
                for name, item in value.items()}
    return value


def _numeric_sidecar(cache_dir, digest):
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("Descriptor cache numeric payload has an invalid hash.")
    return Path(cache_dir) / "arrays" / (digest + ".npz")


def _load_numeric_arrays(cache_dir, payload):
    record = payload.get("numeric_arrays")
    if not isinstance(record, dict) or set(record) != {"sha256", "bytes", "inventory"}:
        raise ValueError("Descriptor cache numeric inventory is missing.")
    digest = record["sha256"]
    path = _numeric_sidecar(cache_dir, digest)
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError("Descriptor cache numeric payload is missing.") from error
    if (len(encoded) != record["bytes"] or len(encoded) > 256 * 1024 * 1024 or
            hashlib.sha256(encoded).hexdigest() != digest):
        raise ValueError("Descriptor cache numeric payload hash or size differs.")
    inventory = record["inventory"]
    if (not isinstance(inventory, dict) or len(inventory) > 100000 or
            any(_ARRAY_NAME.fullmatch(key) is None for key in inventory)):
        raise ValueError("Descriptor cache numeric array inventory is invalid.")
    try:
        with zipfile.ZipFile(BytesIO(encoded)) as archive:
            infos = archive.infolist()
            if (len(infos) != len(inventory) or
                    {item.filename for item in infos} !=
                    {key + ".npy" for key in inventory} or
                    sum(item.file_size for item in infos) > 512 * 1024 * 1024):
                raise ValueError("Descriptor cache NPZ member inventory differs.")
        with np.load(BytesIO(encoded), allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in inventory}
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("Descriptor cache NPZ cannot be read.") from error
    for key, item in inventory.items():
        array = arrays[key]
        if (not isinstance(item, dict) or set(item) != {"dtype", "shape"} or
                str(array.dtype) != item["dtype"] or
                list(array.shape) != item["shape"] or
                array.dtype not in (np.dtype("int64"), np.dtype("float64")) or
                not np.isfinite(array).all()):
            raise ValueError("Descriptor cache numeric array metadata differs.")
    return arrays


def _write_numeric_arrays(cache_dir, encoded):
    if len(encoded) > 256 * 1024 * 1024:
        raise ValueError("Descriptor cache numeric payload exceeds 256 MiB.")
    digest = hashlib.sha256(encoded).hexdigest()
    path = _numeric_sidecar(cache_dir, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return digest
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _memory_bytes(value, seen=None):
    """Conservatively count the retained Python object graph for one entry."""
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return 0
    seen.add(marker)
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + int(value.nbytes)
    if isinstance(value, Mapping):
        return size + sum(
            _memory_bytes(key, seen) + _memory_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(_memory_bytes(item, seen) for item in value)
    if hasattr(value, "__record_fields__"):
        return size + sum(
            _memory_bytes(getattr(value, name), seen)
            for name in value.__record_fields__ if hasattr(value, name)
        )
    if isinstance(value, GeneralizedCouplingLibrary):
        return size + _memory_bytes(value.data, seen) + _memory_bytes(value.metadata, seen)
    return size


def _encode_cache_value(family, value):
    arrays = {}
    if family == "compact_labels":
        labels = value
        library_record = None
    else:
        labels, library, collection = value
        if tuple(collection.compact_labels) != tuple(labels):
            raise ValueError("Descriptor cache labels differ from the collection.")
        data = {}
        for M, M_block in library.data.items():
            ranks = {}
            for rank, rank_block in M_block.items():
                records = {}
                for name, source in rank_block.items():
                    record = _safe_data(source)
                    for field, suffix, dtype in (("ms_combs", "m", np.int64),
                                                 ("coeffs", "c", np.float64)):
                        raw = source.get(field)
                        if not isinstance(raw, list):
                            continue
                        decode = None
                        try:
                            array = np.asarray(raw, dtype=dtype)
                        except (TypeError, ValueError):
                            if field != "coeffs":
                                continue
                            try:
                                complex_array = np.asarray(raw, dtype=np.complex128)
                            except (TypeError, ValueError):
                                continue
                            array = np.stack((complex_array.real, complex_array.imag), axis=-1)
                            decode = "complex_pairs"
                        if not np.isfinite(array).all():
                            raise ValueError("Descriptor cache numeric array is nonfinite.")
                        key = "a" + str(len(arrays)) + "_" + suffix
                        arrays[key] = array
                        record[field] = {"__ye3t_npz__": key}
                        if decode is not None:
                            record[field]["decode"] = decode
                    records[str(name)] = record
                ranks[str(rank)] = records
            data[str(M)] = ranks
        library_record = {
            "L_R": int(library.L_R),
            "data": data,
            "metadata": _safe_data(library.metadata),
        }
    payload = {
        "cache_schema": DESCRIPTOR_BUILD_CACHE_SCHEMA,
        "family": str(family),
        "labels": [normalize_compact_label(label).to_dict() for label in labels],
    }
    if library_record is not None:
        payload["library"] = library_record
        stream = BytesIO()
        np.savez_compressed(stream, **arrays)
        encoded = stream.getvalue()
        payload["numeric_arrays"] = {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
            "inventory": {key: {"dtype": str(array.dtype),
                                "shape": list(array.shape)}
                          for key, array in arrays.items()},
        }
        return payload, encoded
    return payload, None


def _decode_cache_value(family, key, payload, cache_dir=None):
    if (not isinstance(payload, dict)
            or payload.get("cache_schema") != DESCRIPTOR_BUILD_CACHE_SCHEMA
            or payload.get("family") != family):
        raise ValueError("Descriptor cache schema or family differs.")
    labels = tuple(normalize_compact_label(item) for item in payload["labels"])
    if family == "compact_labels":
        return labels
    if compact_label_cache_key(labels) != key.compact_labels:
        raise ValueError("Descriptor cache label order differs from its key.")
    from ye3t_methods.atomistic.equivariant_calc.descriptor_sets import (
        DescriptorGenerationSettings,
        build_descriptor_specs_from_settings,
    )

    record = payload["library"]
    if cache_dir is None:
        raise ValueError("Descriptor cache numeric root is missing.")
    arrays = _load_numeric_arrays(cache_dir, payload)
    references = set()
    data = {
        int(M): {
            int(rank): {
                str(name): _restore_data(item, arrays, references)
                for name, item in rank_block.items()
            }
            for rank, rank_block in M_block.items()
        }
        for M, M_block in record["data"].items()
    }
    metadata = _restore_data(record["metadata"], arrays, references)
    if references != set(arrays):
        raise ValueError("Descriptor cache numeric arrays are unbound or unused.")
    library = GeneralizedCouplingLibrary(
        data,
        L_R=int(record["L_R"]),
        metadata=metadata,
    )
    settings = DescriptorGenerationSettings.from_dict(
        dict(zip(_SETTINGS_FIELDS, key.settings))
    )
    if int(library.L_R) != int(settings.L_R):
        raise ValueError("Descriptor cache output angular rank differs.")
    collection = build_descriptor_specs_from_settings(
        labels,
        settings,
        library,
        center_mu_values=key.center_mu_values,
        restrict_neighbor_mu=key.restrict_neighbor_mu,
        max_variants_per_label=key.max_variants_per_label,
        physical_content_channels=key.physical_content_channels,
    )
    return labels, library, collection


def _read_disk_cache(cache_dir, family, key):
    mode = _cache_mode()
    if mode in {"off", "rebuild"}:
        return None
    store = _store(cache_dir, "read_only")
    required_checks = ("safe_json", "descriptor_roundtrip")
    if family == "artifacts":
        required_checks += ("hash_bound_npz",)
    validated = []

    def validate(candidate):
        decoded = _decode_cache_value(family, key, candidate, cache_dir)
        validated.append(decoded)
        return decoded

    try:
        result = store.resolve(
            "ace_descriptor_" + str(family),
            DESCRIPTOR_BUILD_CACHE_SCHEMA,
            _request_for_key(family, key),
            builder=None,
            validator=validate,
            required_certificate_checks=required_checks,
            producer={"ye3t_methods_version": _package_version("ye3t-methods")},
        )
    except (ArtifactCacheMiss, ArtifactCacheValidationError):
        if mode == "read_only":
            raise
        return None
    if result["status"] == "hit" and validated:
        return validated[-1]
    return _decode_cache_value(family, key, result["payload"], cache_dir)


def _write_disk_cache(cache_dir, family, key, value):
    mode = _cache_mode()
    if mode == "off":
        return None
    payload, numeric_bytes = _encode_cache_value(family, value)
    if numeric_bytes is not None:
        _write_numeric_arrays(cache_dir, numeric_bytes)
    _decode_cache_value(family, key, payload, cache_dir)
    store = _store(cache_dir, mode)
    required_checks = ("safe_json", "descriptor_roundtrip")
    if family == "artifacts":
        required_checks += ("hash_bound_npz",)
    validated = []

    def validate(candidate):
        decoded = _decode_cache_value(family, key, candidate, cache_dir)
        validated.append(decoded)
        return decoded

    result = store.resolve(
        "ace_descriptor_" + str(family),
        DESCRIPTOR_BUILD_CACHE_SCHEMA,
        _request_for_key(family, key),
        builder=lambda: payload,
        validator=validate,
        certificate={
            "passed": True,
            "checks": {name: True for name in required_checks},
        },
        required_certificate_checks=required_checks,
        producer={"ye3t_methods_version": _package_version("ye3t-methods")},
    )
    write_descriptor_build_cache_manifest(cache_dir)
    if not validated:
        raise RuntimeError("Descriptor cache write did not validate the stored entry.")
    return result["status"], validated[-1]


def default_descriptor_build_cache_manifest_path(cache_dir):
    """Return the default manifest path for a descriptor build cache."""
    root = _normalize_cache_dir(cache_dir)
    if root is None:
        raise ValueError("A descriptor cache directory is required for a cache manifest.")
    return root / "descriptor_build_cache_manifest.json"


def _descriptor_cache_entry_from_file(root, path, package_versions):
    encoded = path.read_bytes()
    envelope = json.loads(encoded.decode("utf-8"))
    if not isinstance(envelope, dict):
        raise ValueError("Descriptor-cache envelope is not a mapping.")
    family = path.parent.name.removeprefix("ace_descriptor_")
    if envelope.get("artifact_schema") != DESCRIPTOR_BUILD_CACHE_SCHEMA:
        return None, "legacy_schema"
    if not isinstance(envelope.get("payload"), dict):
        raise ValueError("Descriptor-cache payload is not a mapping.")
    if family not in {"compact_labels", "artifacts"}:
        raise ValueError("Unknown descriptor-cache family.")
    if (envelope.get("store_schema") != ARTIFACT_STORE_SCHEMA or
            envelope.get("artifact_type") != "ace_descriptor_" + family or
            envelope.get("semantic_hash") != path.stem or
            envelope.get("request_hash") != artifact_hash(envelope["request"]) or
            envelope.get("payload_hash") != artifact_hash(envelope["payload"]) or
            envelope.get("envelope_hash") != artifact_hash({
                name: item for name, item in envelope.items() if name != "envelope_hash"}) or
            envelope.get("semantic_hash") != artifact_hash({
                name: envelope[name] for name in (
                    "store_schema", "artifact_type", "artifact_schema", "request",
                    "dependency_hashes", "producer")}) or
            envelope["payload"].get("cache_schema") != DESCRIPTOR_BUILD_CACHE_SCHEMA or
            envelope["payload"].get("family") != family):
        raise ValueError("Descriptor-cache envelope identity or hash differs.")
    producer = envelope.get("producer", {})
    if not isinstance(producer, dict) or producer.get("package") != "ye3t":
        raise ValueError("Descriptor-cache producer differs.")
    if (producer.get("package_version") != package_versions["ye3t"] or
            producer.get("ye3t_methods_version") != package_versions["ye3t-methods"]):
        return None, "stale_producer"
    certificate = envelope.get("certificate", {})
    if not isinstance(certificate, dict):
        raise ValueError("Descriptor-cache certificate is not a mapping.")
    checks = certificate.get("checks", {})
    if not isinstance(checks, dict):
        raise ValueError("Descriptor-cache checks are not a mapping.")
    required = {"safe_json", "descriptor_roundtrip"}
    if family == "artifacts":
        required.add("hash_bound_npz")
    if (certificate.get("passed") is not True or
            any(checks.get(name) is not True for name in required)):
        raise ValueError("Descriptor-cache certificate is incomplete.")
    numeric_hash = None
    if family == "artifacts":
        arrays = _load_numeric_arrays(root, envelope["payload"])
        references = set()
        _restore_data(envelope["payload"]["library"], arrays, references)
        if references != set(arrays):
            raise ValueError("Descriptor-cache numeric references differ.")
        numeric_hash = envelope["payload"]["numeric_arrays"]["sha256"]
    return {
        "family": family,
        "artifact_hash": path.stem,
        "cache_file": path.relative_to(root).as_posix(),
        "envelope_sha256": hashlib.sha256(encoded).hexdigest(),
        "envelope_bytes": len(encoded),
        "payload_hash": envelope["payload_hash"],
        "numeric_sha256": numeric_hash,
    }, None


def _descriptor_cache_manifest_payload(cache_dir):
    root = _normalize_cache_dir(cache_dir)
    if root is None:
        raise ValueError("A descriptor cache directory is required for a cache manifest.")
    cache_root = root / "artifacts"
    entries = []
    ignored_entries = []
    package_versions = {
        "ye3t": _package_version("ye3t"),
        "ye3t-methods": _package_version("ye3t-methods"),
    }
    if cache_root.exists():
        for path in sorted(cache_root.glob("ace_descriptor_*/*.json"),
                           key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            try:
                entry, reason = _descriptor_cache_entry_from_file(
                    root, path, package_versions)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                    KeyError, TypeError, ValueError) as error:
                entry, reason = None, "invalid: " + str(error)
            if entry is None:
                ignored_entries.append({
                    "cache_file": path.relative_to(root).as_posix(),
                    "reason": reason,
                })
            else:
                entries.append(entry)
    array_root = root / "arrays"
    numeric_payloads = []
    orphan_numeric_payloads = []
    referenced = {entry["numeric_sha256"] for entry in entries
                  if entry["numeric_sha256"] is not None}
    if array_root.exists():
        for path in sorted(array_root.glob("*.npz"), key=lambda item: item.name):
            if not path.is_file():
                continue
            item = {
                "cache_file": path.relative_to(root).as_posix(),
                "sha256": path.stem,
                "bytes": int(path.stat().st_size),
            }
            if path.stem in referenced:
                numeric_payloads.append(item)
            else:
                orphan_numeric_payloads.append(item)
    families = OrderedDict()
    for entry in entries:
        family = str(entry["family"])
        current = families.get(family, {"entry_count": 0, "envelope_bytes": 0})
        current["entry_count"] = int(current["entry_count"]) + 1
        current["envelope_bytes"] = int(current["envelope_bytes"]) + int(entry["envelope_bytes"])
        families[family] = current
    payload = {
        "format": DESCRIPTOR_BUILD_CACHE_MANIFEST_FORMAT,
        "manifest_scope": "package_cache",
        "cache_kind": "descriptor_build_cache",
        "cache_schema": DESCRIPTOR_BUILD_CACHE_SCHEMA,
        "package_versions": package_versions,
        "source_identifiers": {
            "descriptor_cache": "ye3t_methods.atomistic.cache.descriptors",
            "coupling_library": GeneralizedCouplingLibrary.__name__,
            "compact_label": CompactLabel.__name__,
        },
        "families": dict(families),
        "entry_count": int(len(entries)),
        "entries": entries,
        "ignored_entries": ignored_entries,
        "numeric_payload_count": int(len(numeric_payloads)),
        "numeric_payload_bytes": int(sum(item["bytes"] for item in numeric_payloads)),
        "numeric_payloads": numeric_payloads,
        "orphan_numeric_payloads": orphan_numeric_payloads,
    }
    payload["manifest_hash"] = _stable_manifest_hash(payload)
    return payload


def write_descriptor_build_cache_manifest(cache_dir):
    """Write a portable manifest for descriptor-label and coupling-artifact cache files."""
    path = default_descriptor_build_cache_manifest_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ArtifactLock(path.with_suffix(path.suffix + ".lock")):
        payload = _descriptor_cache_manifest_payload(cache_dir)
        tmp_path = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    result = dict(payload)
    result["manifest_path"] = str(path)
    return result


def load_descriptor_build_cache_manifest(cache_dir):
    """Load a descriptor build-cache manifest after checking its live inventory."""
    path = default_descriptor_build_cache_manifest_path(cache_dir)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactCacheValidationError("Descriptor-cache manifest cannot be read.") from error
    if (not isinstance(saved, dict) or
            saved.get("format") != DESCRIPTOR_BUILD_CACHE_MANIFEST_FORMAT or
            saved.get("cache_schema") != DESCRIPTOR_BUILD_CACHE_SCHEMA or
            saved.get("manifest_hash") != _stable_manifest_hash({
                name: item for name, item in saved.items() if name != "manifest_hash"}) or
            saved != _descriptor_cache_manifest_payload(cache_dir)):
        raise ArtifactCacheValidationError(
            "Descriptor-cache manifest differs from the live verified inventory."
        )
    return saved


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


@recordclass(('settings', 'compact_labels', 'basis_mode', 'exact_primitive_timeout_seconds', 'center_mu_values', 'restrict_neighbor_mu', 'max_variants_per_label', 'scalar_coordinate_compiler', 'physical_content_channels'), frozen = True)
class DescriptorArtifactCacheKey:
    """Key for compiled descriptor artifacts."""
    physical_content_channels = None


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
    physical_content_channels=None,
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
            else ("ace_coordinate_compiler_v1:" if int(settings.L_R) != 0
                  else "scalar_coordinate_compiler_v1:")
            + json.dumps(
                scalar_coordinate_compiler,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        physical_content_channels=(None if physical_content_channels is None else
                                   tuple(sorted(tuple(int(value) for value in row)
                                                for row in physical_content_channels))),
    )


class DescriptorBuildCache:
    """Shared cache for compact-label enumeration and compiled descriptor artifacts."""

    def __init__(self, max_compact_label_entries = 64, max_artifact_entries = 16,
                 cache_dir = None, max_compact_label_bytes = 64 * 1024 * 1024,
                 max_artifact_bytes = 256 * 1024 * 1024):
        self.max_compact_label_entries = int(max_compact_label_entries)
        self.max_artifact_entries = int(max_artifact_entries)
        self.max_compact_label_bytes = int(max_compact_label_bytes)
        self.max_artifact_bytes = int(max_artifact_bytes)
        self.cache_dir = _normalize_cache_dir(cache_dir)
        self._compact_labels = OrderedDict()
        self._artifacts = OrderedDict()
        self._compact_label_sizes = {}
        self._artifact_sizes = {}
        self.compact_label_bytes = 0
        self.artifact_bytes = 0
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
        self._compact_label_sizes.clear()
        self._artifact_sizes.clear()
        self.compact_label_bytes = 0
        self.artifact_bytes = 0
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

    def _remember(self, entries, sizes, byte_field, max_entries, max_bytes, key, value):
        entries.pop(key, None)
        total = int(getattr(self, byte_field)) - sizes.pop(key, 0)
        size = (_memory_bytes((key, value))
                + sys.getsizeof({key: value}) - sys.getsizeof({}))
        if max_entries > 0 and max_bytes > 0 and size <= max_bytes:
            entries[key] = value
            sizes[key] = size
            total += size
            while len(entries) > max_entries or total > max_bytes:
                oldest, _ = entries.popitem(last=False)
                total -= sizes.pop(oldest)
        setattr(self, byte_field, total)

    def _remember_compact_labels(self, key, value):
        self._remember(
            self._compact_labels, self._compact_label_sizes,
            "compact_label_bytes", self.max_compact_label_entries,
            self.max_compact_label_bytes, key, value,
        )

    def _remember_artifacts(self, key, value):
        self._remember(
            self._artifacts, self._artifact_sizes,
            "artifact_bytes", self.max_artifact_entries,
            self.max_artifact_bytes, key, value,
        )

    def get_compact_labels(self, key):
        item = (None if _cache_mode() == "rebuild"
                else self._compact_labels.get(key))
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
        if self.cache_dir is not None:
            result = _write_disk_cache(self.cache_dir, "compact_labels", key, value)
            if result is not None:
                status, value = result
            else:
                status = None
            if status == "miss":
                self.compact_label_disk_writes += 1
        self._remember_compact_labels(key, value)
        return value

    def get_artifacts(self, key):
        item = (None if _cache_mode() == "rebuild"
                else self._artifacts.get(key))
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
        if self.cache_dir is not None:
            result = _write_disk_cache(self.cache_dir, "artifacts", key, value)
            if result is not None:
                status, value = result
            else:
                status = None
            if status == "miss":
                self.artifact_disk_writes += 1
        self._remember_artifacts(key, value)
        return value

    def get_stats(self):
        return {
            "compact_label_entries": len(self._compact_labels),
            "artifact_entries": len(self._artifacts),
            "compact_label_bytes": int(self.compact_label_bytes),
            "artifact_bytes": int(self.artifact_bytes),
            "max_compact_label_bytes": int(self.max_compact_label_bytes),
            "max_artifact_bytes": int(self.max_artifact_bytes),
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
    max_compact_label_bytes=int(_ace_env("COMPACT_LABEL_CACHE_MAX_BYTES", str(64 * 1024 * 1024))),
    max_artifact_bytes=int(_ace_env("DESCRIPTOR_ARTIFACT_CACHE_MAX_BYTES", str(256 * 1024 * 1024))),
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
