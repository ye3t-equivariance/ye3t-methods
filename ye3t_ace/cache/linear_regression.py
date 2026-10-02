"""Persistent target-free row and target caches for linear YE3T fitting."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time
import uuid

import numpy as np

from ye3t.cache import canonical_json_bytes


LINEAR_ARRAY_CACHE_SCHEMA = "ye3t_linear_array_cache_v1"
LINEAR_STATISTICS_CACHE_SCHEMA = "ye3t_linear_statistics_cache_v1"
_CACHE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LinearCacheValidationError(RuntimeError):
    """Raised when a persistent regression cache fails identity validation."""


def default_linear_cache_directory():
    override = os.environ.get("YE3T_ACE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    shared = os.environ.get("YE3T_CACHE_DIR")
    if shared:
        return Path(shared).expanduser() / "applications"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "ye3t" / "applications"
    return Path.home() / ".cache" / "ye3t" / "applications"


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"ye3t_linear_array_v1\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _manifest_hash(body):
    return _hash_bytes(canonical_json_bytes(body))


def _fsync_directory(path):
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class _LinearCacheLock:
    def __init__(self, path, timeout=30.0):
        self.path = Path(path)
        self.timeout = float(timeout)
        self.acquired = False

    def __enter__(self):
        started = time.monotonic()
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                if self._remove_dead_owner():
                    continue
                if time.monotonic() - started >= self.timeout:
                    raise TimeoutError(
                        f"Timed out waiting for linear-cache lock {self.path}."
                    )
                time.sleep(0.05)
                continue
            try:
                owner = {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                }
                os.write(descriptor, canonical_json_bytes(owner) + b"\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.acquired = True
            return self

    def _remove_dead_owner(self):
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if str(owner.get("hostname", "")) != socket.gethostname():
            return False
        try:
            pid = int(owner["pid"])
        except (KeyError, TypeError, ValueError):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return True
        except PermissionError:
            return False
        return False

    def __exit__(self, exc_type, exc_value, traceback):
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        return False


def _write_array(path, value):
    array = np.ascontiguousarray(value)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "file": path.name,
        "dtype": str(array.dtype),
        "shape": tuple(int(value) for value in array.shape),
        "array_hash": _array_hash(array),
        "bytes": int(array.nbytes),
    }


def _read_array(directory, record, mmap_mode):
    filename = str(record["file"])
    if Path(filename).name != filename:
        raise LinearCacheValidationError("Array manifest contains an unsafe path.")
    path = Path(directory) / filename
    try:
        value = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise LinearCacheValidationError(
            f"Cannot load cached array {filename}: {error}."
        ) from error
    if str(value.dtype) != str(record["dtype"]):
        raise LinearCacheValidationError(f"Cached array {filename} dtype mismatch.")
    if tuple(value.shape) != tuple(int(item) for item in record["shape"]):
        raise LinearCacheValidationError(f"Cached array {filename} shape mismatch.")
    if _array_hash(value) != str(record["array_hash"]):
        raise LinearCacheValidationError(f"Cached array {filename} hash mismatch.")
    value.setflags(write=False)
    return value


def _cache_directory(kind, request_hash, directory):
    kind = str(kind)
    request_hash = str(request_hash)
    if _CACHE_COMPONENT.fullmatch(kind) is None:
        raise ValueError("Linear-cache kind is not a safe path component.")
    if _SHA256.fullmatch(request_hash) is None:
        raise ValueError("Linear-cache request identity must be a SHA-256 digest.")
    root = default_linear_cache_directory() if directory is None else Path(directory)
    return Path(root).expanduser() / kind / request_hash


def _write_manifest(directory, body):
    payload = {**body, "manifest_hash": _manifest_hash(body)}
    path = Path(directory) / "manifest.json"
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(directory)


def _read_manifest(directory, kind, request_hash):
    path = Path(directory) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LinearCacheValidationError(
            f"Cannot decode linear-cache manifest: {error}."
        ) from error
    expected = str(payload.get("manifest_hash", ""))
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if not expected or expected != _manifest_hash(body):
        raise LinearCacheValidationError("Linear-cache manifest hash mismatch.")
    if body.get("schema") != LINEAR_ARRAY_CACHE_SCHEMA:
        raise LinearCacheValidationError("Unsupported linear-cache manifest schema.")
    if str(body.get("kind")) != str(kind):
        raise LinearCacheValidationError("Linear-cache kind mismatch.")
    if str(body.get("request_hash")) != str(request_hash):
        raise LinearCacheValidationError("Linear-cache identity mismatch.")
    return body


def _persist_directory(final, build, validate_existing):
    final.parent.mkdir(parents=True, exist_ok=True)
    lock = final.parent / (final.name + ".lock")
    with _LinearCacheLock(lock):
        if final.exists():
            try:
                validate_existing()
            except LinearCacheValidationError:
                quarantine = final.with_name(
                    final.name + ".invalid." + uuid.uuid4().hex
                )
                os.replace(final, quarantine)
            else:
                return final
        temporary = final.parent / ("." + final.name + "." + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            build(temporary)
            os.replace(temporary, final)
            _fsync_directory(final.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return final


def persist_lifted_cauchy_geometry_row_cache(cache, directory=None):
    """Persist one target-free A1 cache as hash-bound ``.npy`` chunks."""

    if cache.get("schema") != "ye3t_lifted_cauchy_geometry_row_cache_v2":
        raise ValueError("Unsupported lifted-Cauchy geometry-row cache schema.")
    cache_hash = str(cache["cache_hash"])
    request_hash = str(cache["request_hash"])
    final = _cache_directory(
        "lifted_cauchy_geometry_rows", request_hash, directory
    )

    def build(temporary):
        arrays = {
            "runtime_from_fit_coordinates": _write_array(
                temporary / "runtime_from_fit_coordinates.npy",
                cache["runtime_from_fit_coordinates"],
            ),
            "orthogonal_coordinate_norm_squared": _write_array(
                temporary / "orthogonal_coordinate_norm_squared.npy",
                cache["orthogonal_coordinate_norm_squared"],
            ),
        }
        record_manifests = []
        for index, record in enumerate(cache["records"]):
            prefix = f"record_{index:06d}"
            record_manifests.append(
                {
                    "structure_index": int(record["structure_index"]),
                    "atom_count": int(record["atom_count"]),
                    "geometry_hash": str(record["geometry_hash"]),
                    "row_content_hash": str(record["row_content_hash"]),
                    "retained_bytes": int(record["retained_bytes"]),
                    "site_design": _write_array(
                        temporary / (prefix + "_site.npy"), record["site_design"]
                    ),
                    "force_design": _write_array(
                        temporary / (prefix + "_force.npy"), record["force_design"]
                    ),
                    "species_counts": _write_array(
                        temporary / (prefix + "_species.npy"),
                        record["species_counts"],
                    ),
                }
            )
        excluded = {
            "records",
            "runtime_from_fit_coordinates",
            "orthogonal_coordinate_norm_squared",
        }
        metadata = {key: value for key, value in cache.items() if key not in excluded}
        _write_manifest(
            temporary,
            {
                "schema": LINEAR_ARRAY_CACHE_SCHEMA,
                "kind": "lifted_cauchy_geometry_rows",
                "cache_hash": cache_hash,
                "request_hash": request_hash,
                "metadata": metadata,
                "arrays": arrays,
                "records": tuple(record_manifests),
            },
        )

    def validate_existing():
        loaded = load_lifted_cauchy_geometry_row_cache(
            request_hash, directory, mmap_mode="r"
        )
        if str(loaded.get("cache_hash")) != cache_hash:
            raise LinearCacheValidationError(
                "Geometry-row request resolved to different cached content."
            )

    _persist_directory(final, build, validate_existing)
    _read_manifest(final, "lifted_cauchy_geometry_rows", request_hash)
    return final


def load_lifted_cauchy_geometry_row_cache(
    request_hash,
    directory=None,
    mmap_mode="r",
):
    """Load and verify one target-free A1 cache, using mmap by default."""

    final = _cache_directory(
        "lifted_cauchy_geometry_rows", request_hash, directory
    )
    manifest = _read_manifest(
        final, "lifted_cauchy_geometry_rows", request_hash
    )
    arrays = manifest["arrays"]
    metadata = dict(manifest["metadata"])
    tuple_fields = (
        "central_species_order",
        "descriptor_coordinate_ids",
        "feature_coordinate_ids",
        "structure_hashes",
        "row_content_hashes",
    )
    for name in tuple_fields:
        if name in metadata:
            metadata[name] = tuple(metadata[name])
    records = []
    for record in manifest["records"]:
        records.append(
            {
                "structure_index": int(record["structure_index"]),
                "atom_count": int(record["atom_count"]),
                "geometry_hash": str(record["geometry_hash"]),
                "row_content_hash": str(record["row_content_hash"]),
                "retained_bytes": int(record["retained_bytes"]),
                "site_design": _read_array(final, record["site_design"], mmap_mode),
                "force_design": _read_array(
                    final, record["force_design"], mmap_mode
                ),
                "species_counts": _read_array(
                    final, record["species_counts"], mmap_mode
                ),
            }
        )
    return {
        **metadata,
        "runtime_from_fit_coordinates": _read_array(
            final, arrays["runtime_from_fit_coordinates"], mmap_mode
        ),
        "orthogonal_coordinate_norm_squared": _read_array(
            final, arrays["orthogonal_coordinate_norm_squared"], mmap_mode
        ),
        "records": tuple(records),
    }


def persist_lifted_cauchy_target_cache(cache, directory=None):
    """Persist one A2 target/reference result independently of A1 rows."""

    if cache.get("schema") not in {
        "ye3t_lifted_cauchy_target_cache_v2",
        "ye3t_lifted_cauchy_target_cache_v3",
    }:
        raise ValueError("Unsupported lifted-Cauchy target cache schema.")
    cache_hash = str(cache["cache_hash"])
    request_hash = str(cache["request_hash"])
    final = _cache_directory("lifted_cauchy_targets", request_hash, directory)

    def build(temporary):
        record_manifests = []
        for index, record in enumerate(cache["records"]):
            record_manifests.append(
                {
                    "structure_index": int(record["structure_index"]),
                    "geometry_hash": str(record["geometry_hash"]),
                    "energy_target": float(record["energy_target"]),
                    "target_content_hash": str(record["target_content_hash"]),
                    "retained_bytes": int(record["retained_bytes"]),
                    "force_target": _write_array(
                        temporary / f"record_{index:06d}_force_target.npy",
                        record["force_target"],
                    ),
                }
            )
        metadata = {key: value for key, value in cache.items() if key != "records"}
        _write_manifest(
            temporary,
            {
                "schema": LINEAR_ARRAY_CACHE_SCHEMA,
                "kind": "lifted_cauchy_targets",
                "cache_hash": cache_hash,
                "request_hash": request_hash,
                "metadata": metadata,
                "records": tuple(record_manifests),
            },
        )

    def validate_existing():
        loaded = load_lifted_cauchy_target_cache(
            request_hash, directory, mmap_mode="r"
        )
        if str(loaded.get("cache_hash")) != cache_hash:
            raise LinearCacheValidationError(
                "Target request resolved to different cached content."
            )

    _persist_directory(final, build, validate_existing)
    _read_manifest(final, "lifted_cauchy_targets", request_hash)
    return final


def load_lifted_cauchy_target_cache(request_hash, directory=None, mmap_mode="r"):
    """Load and verify one A2 target/reference cache."""

    final = _cache_directory("lifted_cauchy_targets", request_hash, directory)
    manifest = _read_manifest(final, "lifted_cauchy_targets", request_hash)
    metadata = dict(manifest["metadata"])
    if "target_content_hashes" in metadata:
        metadata["target_content_hashes"] = tuple(
            metadata["target_content_hashes"]
        )
    if "geometry_records" in metadata:
        metadata["geometry_records"] = tuple(metadata["geometry_records"])
    records = []
    for record in manifest["records"]:
        records.append(
            {
                "structure_index": int(record["structure_index"]),
                "geometry_hash": str(record["geometry_hash"]),
                "energy_target": float(record["energy_target"]),
                "target_content_hash": str(record["target_content_hash"]),
                "retained_bytes": int(record["retained_bytes"]),
                "force_target": _read_array(
                    final, record["force_target"], mmap_mode
                ),
            }
        )
    return {**metadata, "records": tuple(records)}


def persist_lifted_cauchy_normal_equations(problem, directory=None):
    """Persist one identity-bound A3 normal-equation problem."""

    request_hash = str(problem.get("normal_equation_request_hash", ""))
    problem_hash = str(problem.get("problem_hash", ""))
    if not request_hash or not problem_hash:
        raise ValueError(
            "A persistent normal-equation problem requires request and problem hashes."
        )
    final = _cache_directory(
        "lifted_cauchy_normal_equations", request_hash, directory
    )
    array_names = (
        "runtime_from_fit_coordinates",
        "feature_mean",
        "feature_scale",
        "fit_coordinate_metric",
        "XtX",
        "Xty",
    )

    def build(temporary):
        arrays = {
            name: _write_array(temporary / (name + ".npy"), problem[name])
            for name in array_names
        }
        metadata = {
            key: value for key, value in problem.items() if key not in array_names
        }
        _write_manifest(
            temporary,
            {
                "schema": LINEAR_ARRAY_CACHE_SCHEMA,
                "kind": "lifted_cauchy_normal_equations",
                "cache_hash": problem_hash,
                "request_hash": request_hash,
                "metadata": metadata,
                "arrays": arrays,
            },
        )

    def validate_existing():
        loaded = load_lifted_cauchy_normal_equations(
            request_hash, directory, mmap_mode="r"
        )
        if str(loaded.get("problem_hash")) != problem_hash:
            raise LinearCacheValidationError(
                "Normal-equation request resolved to different cached content."
            )

    _persist_directory(final, build, validate_existing)
    _read_manifest(final, "lifted_cauchy_normal_equations", request_hash)
    return final


def load_lifted_cauchy_normal_equations(
    request_hash,
    directory=None,
    mmap_mode="r",
):
    """Load and verify one A3 normal-equation problem."""

    final = _cache_directory(
        "lifted_cauchy_normal_equations", request_hash, directory
    )
    manifest = _read_manifest(
        final, "lifted_cauchy_normal_equations", request_hash
    )
    metadata = dict(manifest["metadata"])
    if str(metadata.get("normal_equation_request_hash", "")) != str(
        request_hash
    ):
        raise LinearCacheValidationError(
            "Cached normal-equation metadata is bound to another request."
        )
    if str(metadata.get("problem_hash", "")) != str(
        manifest.get("cache_hash", "")
    ):
        raise LinearCacheValidationError(
            "Cached normal-equation problem hash disagrees with its manifest."
        )
    for name in (
        "central_species_order",
        "selected_record_indices",
        "selected_feature_indices",
        "selected_feature_coordinate_ids",
        "selected_descriptor_coordinate_ids",
    ):
        if metadata.get(name) is not None:
            metadata[name] = tuple(metadata[name])
    arrays = manifest["arrays"]
    return {
        **metadata,
        **{
            name: _read_array(final, arrays[name], mmap_mode)
            for name in (
                "runtime_from_fit_coordinates",
                "feature_mean",
                "feature_scale",
                "fit_coordinate_metric",
                "XtX",
                "Xty",
            )
        },
    }


def persist_linear_sufficient_statistics(cache, directory=None):
    """Persist a generic identity-bound collection of Gram/statistics arrays.

    The caller owns the scientific request identity and JSON metadata.  This
    layer owns safe paths, atomic publication, per-array hashes, and corruption
    detection.  Geometry-row and target/reference identities should remain
    separate fields in the request metadata so changing a reference potential
    does not invalidate target-free descriptor rows.
    """

    if cache.get("schema") != LINEAR_STATISTICS_CACHE_SCHEMA:
        raise ValueError("Unsupported linear sufficient-statistics cache schema.")
    request_hash = str(cache.get("request_hash", ""))
    cache_hash = str(cache.get("cache_hash", ""))
    if _SHA256.fullmatch(request_hash) is None or _SHA256.fullmatch(cache_hash) is None:
        raise ValueError("Statistics request and cache identities must be SHA-256 digests.")
    arrays = dict(cache.get("arrays", {}))
    if not arrays:
        raise ValueError("A statistics cache requires at least one array.")
    if any(_CACHE_COMPONENT.fullmatch(str(name)) is None for name in arrays):
        raise ValueError("Statistics array names must be safe path components.")
    metadata = dict(cache.get("metadata", {}))
    final = _cache_directory("linear_sufficient_statistics", request_hash, directory)

    def build(temporary):
        array_records = {
            str(name): _write_array(temporary / (str(name) + ".npy"), value)
            for name, value in sorted(arrays.items())
        }
        _write_manifest(
            temporary,
            {
                "schema": LINEAR_ARRAY_CACHE_SCHEMA,
                "kind": "linear_sufficient_statistics",
                "cache_hash": cache_hash,
                "request_hash": request_hash,
                "metadata": metadata,
                "arrays": array_records,
            },
        )

    def validate_existing():
        loaded = load_linear_sufficient_statistics(
            request_hash, directory, mmap_mode="r"
        )
        if str(loaded.get("cache_hash")) != cache_hash:
            raise LinearCacheValidationError(
                "Statistics request resolved to different cached content."
            )

    _persist_directory(final, build, validate_existing)
    _read_manifest(final, "linear_sufficient_statistics", request_hash)
    return final


def load_linear_sufficient_statistics(
    request_hash,
    directory=None,
    mmap_mode="r",
):
    """Load and verify a generic sufficient-statistics collection."""

    final = _cache_directory("linear_sufficient_statistics", request_hash, directory)
    manifest = _read_manifest(final, "linear_sufficient_statistics", request_hash)
    arrays = {
        str(name): _read_array(final, record, mmap_mode)
        for name, record in manifest["arrays"].items()
    }
    return {
        "schema": LINEAR_STATISTICS_CACHE_SCHEMA,
        "request_hash": str(manifest["request_hash"]),
        "cache_hash": str(manifest["cache_hash"]),
        "metadata": dict(manifest["metadata"]),
        "arrays": arrays,
    }


__all__ = [
    "LINEAR_ARRAY_CACHE_SCHEMA",
    "LINEAR_STATISTICS_CACHE_SCHEMA",
    "LinearCacheValidationError",
    "default_linear_cache_directory",
    "load_linear_sufficient_statistics",
    "load_lifted_cauchy_geometry_row_cache",
    "load_lifted_cauchy_normal_equations",
    "load_lifted_cauchy_target_cache",
    "persist_lifted_cauchy_geometry_row_cache",
    "persist_lifted_cauchy_normal_equations",
    "persist_lifted_cauchy_target_cache",
    "persist_linear_sufficient_statistics",
]
