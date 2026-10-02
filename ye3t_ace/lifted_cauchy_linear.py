"""Linear lifted-Cauchy descriptors compiled by :mod:`ye3t.couplings`.

This module materializes the role-resolved atomistic source and executes an
already compiled descriptor artifact.  It does not enumerate Young labels,
coupling paths, or multiplicity coordinates.
"""

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import time

import numpy as np
import torch

from ye3t.couplings import CompiledLiftedCauchyScalar
from ye3t.couplings import compile as compile_coupling
from ye3t.core.spherical import (
    real_spherical_harmonics_l_from_cartesian_with_derivatives,
    real_spherical_harmonics_l_from_unit_cartesian,
)
from ye3t_ace.equivariant_calc.edge_geometry import (
    directed_edges_bruteforce,
    edge_displacements_from_indices,
    normalize_pbc,
)
from ye3t_ace.equivariant_calc.neighbors import neighbor_data_from_ase_atoms
from ye3t.couplings.orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    shifted_jacobi_normalization_squared as _shifted_jacobi_normalization_squared,
    shifted_jacobi_power_coefficients as _shifted_jacobi_power_coefficients,
)
from ye3t_ace.utils.fit_weights import structure_fit_weights


LIFTED_CAUCHY_SOURCE_SCHEMA = "ye3t_lifted_cauchy_polynomial_source_v1"
LIFTED_CAUCHY_SOURCE_FAMILY = "primitive_polynomial_envelope_v1"
LIFTED_CAUCHY_JOINT_SOURCE_SCHEMA = "ye3t_lifted_cauchy_joint_source_v2"
LIFTED_CAUCHY_JOINT_SOURCE_FAMILY = "orthogonal_shifted_jacobi_l1_v1"
LIFTED_CAUCHY_MIXED_L_SOURCE_SCHEMA = "ye3t_lifted_cauchy_joint_source_v3"
LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY = ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY
_COMPILED_ARTIFACT_CACHE = {}
_COMPILED_ARTIFACT_CACHE_LIMIT = 32
_ORTHOGONAL_OUTPUT_CACHE = {}


def _binary_complex(payload):
    real, imag = payload["binary64"]
    return complex(float(real), float(imag))


def _artifact_body_hash(payload):
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_compiled_artifact(compiled):
    payload = compiled.to_dict()
    expected = str(payload["self_hash"])
    if _artifact_body_hash(payload) != expected:
        raise ValueError("Lifted-Cauchy compiled artifact self-hash mismatch.")
    _COMPILED_ARTIFACT_CACHE[expected] = compiled
    while len(_COMPILED_ARTIFACT_CACHE) > _COMPILED_ARTIFACT_CACHE_LIMIT:
        _COMPILED_ARTIFACT_CACHE.pop(next(iter(_COMPILED_ARTIFACT_CACHE)))
    return compiled


def _load_compiled_artifact(value):
    if isinstance(value, CompiledLiftedCauchyScalar):
        return _cache_compiled_artifact(value)
    if isinstance(value, (str, Path)):
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        return _load_compiled_artifact(payload)
    if isinstance(value, dict):
        expected = str(value.get("self_hash", ""))
        if not expected or _artifact_body_hash(value) != expected:
            raise ValueError("Lifted-Cauchy compiled artifact self-hash mismatch.")
        cached = _COMPILED_ARTIFACT_CACHE.get(expected)
        if cached is not None:
            return _cache_compiled_artifact(cached)
        return _cache_compiled_artifact(
            CompiledLiftedCauchyScalar.from_dict(value)
        )
    raise TypeError(
        "lifted_cauchy_compiled must be a compiled artifact, mapping, or JSON path."
    )


def _load_precomputed_orthogonal_output_plan(compiled, value, expected_hash=None):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("orthogonal_plan must be a mapping or JSON path.")
    plan = dict(value)
    self_hash = str(plan.get("self_hash", ""))
    if not self_hash or _artifact_body_hash(plan) != self_hash:
        raise ValueError("Lifted-Cauchy orthogonal-plan self-hash mismatch.")
    if expected_hash is not None and self_hash != str(expected_hash):
        raise ValueError("Lifted-Cauchy orthogonal-plan certificate hash mismatch.")
    if plan.get("schema") != "ye3t_linear_lifted_cauchy_orthogonal_output_v1":
        raise ValueError("Unsupported lifted-Cauchy orthogonal-plan schema.")
    if str(plan.get("compiled_artifact_hash", "")) != str(compiled.self_hash):
        raise ValueError("Orthogonal plan is bound to a different compiler artifact.")
    validation = dict(plan.get("validation_report", {}))
    required = {
        "passed": True,
        "groups_partition_descriptors": True,
        "within_group_gram_diagonal_exact": True,
        "cross_group_gram_zero_exact": True,
        "runtime_transform_required": False,
    }
    for key, expected in required.items():
        if validation.get(key) != expected:
            raise ValueError(
                f"Orthogonal plan does not certify {key}={expected!r}."
            )
    return plan


def _orthogonal_matrix_from_plan(compiled, plan):
    descriptor_count = len(compiled.payload["descriptors"])
    matrix = np.zeros((descriptor_count, descriptor_count), dtype=np.complex128)
    norms = np.zeros(descriptor_count, dtype=float)
    seen = []
    for group in plan["groups"]:
        indices = tuple(int(value) for value in group["descriptor_indices"])
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("Orthogonal-plan groups must contain unique indices.")
        if any(index < 0 or index >= descriptor_count for index in indices):
            raise ValueError("Orthogonal-plan descriptor index is out of range.")
        local = np.asarray(
            [
                [_binary_complex(value) for value in row]
                for row in group["orthogonal_from_pivot"]
            ],
            dtype=np.complex128,
        )
        local_norms = np.asarray(
            [
                _binary_complex(value)
                for value in group["orthogonal_norm_squared"]
            ],
            dtype=np.complex128,
        )
        if local.shape != (len(indices), len(indices)) or local_norms.shape != (
            len(indices),
        ):
            raise ValueError("Orthogonal-plan group dimensions are inconsistent.")
        matrix[np.ix_(indices, indices)] = local
        if np.max(np.abs(local_norms.imag), initial=0.0) > 5.0e-13:
            raise ValueError("Orthogonal-output norms must be physically real.")
        norms[np.asarray(indices, dtype=int)] = local_norms.real
        seen.extend(indices)
    if tuple(sorted(seen)) != tuple(range(descriptor_count)):
        raise ValueError("Orthogonal-plan groups do not partition descriptors.")
    if np.max(np.abs(matrix.imag), initial=0.0) > 5.0e-13:
        raise ValueError("Orthogonal-output transform must be physically real.")
    return matrix.real, norms


def _orthogonal_output_plan_and_matrix(
    compiled, precomputed_plan=None, expected_hash=None
):
    compiled = _load_compiled_artifact(compiled)
    validated_precomputed = None
    if precomputed_plan is not None:
        validated_precomputed = _load_precomputed_orthogonal_output_plan(
            compiled, precomputed_plan, expected_hash=expected_hash
        )
    cached = _ORTHOGONAL_OUTPUT_CACHE.get(str(compiled.self_hash))
    if cached is not None:
        required_hash = (
            expected_hash
            if validated_precomputed is None
            else validated_precomputed["self_hash"]
        )
        if required_hash is not None and str(cached[0]["self_hash"]) != str(
            required_hash
        ):
            raise ValueError("Cached orthogonal plan disagrees with its certificate.")
        return cached
    if precomputed_plan is None:
        from ye3t.couplings import lifted_cauchy_orthogonal_output_plan

        plan = lifted_cauchy_orthogonal_output_plan(compiled)
        if expected_hash is not None and str(plan["self_hash"]) != str(expected_hash):
            raise ValueError("Computed orthogonal plan disagrees with its certificate.")
    else:
        plan = validated_precomputed
    matrix, norms = _orthogonal_matrix_from_plan(compiled, plan)
    result = (plan, matrix.real, norms)
    _ORTHOGONAL_OUTPUT_CACHE[str(compiled.self_hash)] = result
    return result


def _fit_coordinate_lowering(model, coordinate_policy):
    coordinate_policy = str(coordinate_policy).strip().lower()
    feature_count = model.evaluator.descriptor_count
    head_count = len(model.type_order)
    if coordinate_policy == "pivot":
        return None, np.eye(head_count * feature_count), np.ones(feature_count)
    if coordinate_policy != "orthogonal":
        raise ValueError("fit_coordinate_policy must be pivot or orthogonal.")
    if model.compiled is not None:
        plan, transform, norms = _orthogonal_output_plan_and_matrix(model.compiled)
    else:
        component_plans = []
        component_transforms = []
        component_norms = []
        plans = getattr(model, "component_orthogonal_output_plans", None)
        if plans is None:
            plans = (None,) * len(model.compiled_components)
        for compiled, supplied in zip(
            model.compiled_components, plans, strict=True
        ):
            component_plan, component_transform, norms = (
                _orthogonal_output_plan_and_matrix(
                    compiled,
                    precomputed_plan=supplied,
                    expected_hash=(
                        None if supplied is None else supplied["self_hash"]
                    ),
                )
            )
            component_plans.append(component_plan)
            component_transforms.append(component_transform)
            component_norms.append(norms)
        total = sum(transform.shape[0] for transform in component_transforms)
        transform = np.zeros((total, total), dtype=float)
        cursor = 0
        for component_transform in component_transforms:
            stop = cursor + component_transform.shape[0]
            transform[cursor:stop, cursor:stop] = component_transform
            cursor = stop
        norms = np.concatenate(tuple(component_norms))
        semantic = {
            "schema": "ye3t_lifted_cauchy_composite_orthogonal_output_v2",
            "composite_artifact_hash": str(model.artifact_hash),
            "descriptor_coordinate_ids": tuple(
                model.descriptor_coordinate_ids
            ),
            "component_plan_hashes": tuple(
                str(component_plan["self_hash"])
                for component_plan in component_plans
            ),
            "component_artifact_hashes": tuple(
                str(compiled.self_hash) for compiled in model.compiled_components
            ),
        }
        plan = {**semantic, "self_hash": _canonical_source_hash(semantic)}
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("Orthogonal-output coordinate norms must be positive.")
    normalized_transform = transform / np.sqrt(norms)[:, None]
    lowering = np.kron(np.eye(head_count), normalized_transform.T)
    return plan, lowering, norms


def prepare_lifted_cauchy_descriptor_payload(config, type_map):
    """Compile or load one descriptor artifact and bind its source identity."""

    config = dict(config or {})
    allowed = {
        "compiled",
        "compiled_artifact",
        "components",
        "request",
        "source",
    }
    extras = sorted(set(config) - allowed)
    if extras:
        raise ValueError(f"Unsupported lifted_cauchy descriptor keys: {extras}")
    compiled_value = config.get("compiled", config.get("compiled_artifact"))
    request = config.get("request")
    components = config.get("components")
    if sum(value is not None for value in (compiled_value, request, components)) != 1:
        raise ValueError(
            "lifted_cauchy requires exactly one of compiled/compiled_artifact, "
            "request, or components."
        )
    if components is not None:
        components = tuple(dict(component) for component in components)
        if not components:
            raise ValueError("lifted_cauchy components must not be empty.")
        allowed_component = {
            "compiled",
            "compiled_artifact",
            "component_id",
            "coordinate_ids",
            "opportunity_id",
            "orthogonal_output_plan_hash",
            "orthogonal_plan",
            "selection_hash",
            "strict_sector_ids",
        }
        compiled_components = []
        component_records = []
        component_orthogonal_output_plans = []
        component_ids = set()
        for position, component in enumerate(components):
            component_extras = sorted(set(component) - allowed_component)
            if component_extras:
                raise ValueError(
                    "Unsupported lifted_cauchy component keys: "
                    f"{component_extras}"
                )
            value = component.get(
                "compiled", component.get("compiled_artifact")
            )
            if value is None:
                raise ValueError(
                    "Every lifted_cauchy component requires a compiled artifact."
                )
            compiled_component = _load_compiled_artifact(value)
            component_id = str(
                component.get(
                    "component_id",
                    component.get("opportunity_id", compiled_component.self_hash),
                )
            )
            if not component_id or component_id in component_ids:
                raise ValueError(
                    "lifted_cauchy component IDs must be nonempty and unique."
                )
            component_ids.add(component_id)
            coordinate_ids = tuple(
                str(value) for value in component.get("coordinate_ids", ())
            )
            if len(coordinate_ids) != len(
                compiled_component.payload["descriptors"]
            ):
                raise ValueError(
                    "Component coordinate_ids must bind every compiled descriptor."
                )
            selection_hash = str(component.get("selection_hash", ""))
            if not selection_hash:
                raise ValueError(
                    "Every lifted_cauchy component requires selection_hash."
                )
            strict_sector_ids = tuple(
                str(value) for value in component.get("strict_sector_ids", ())
            )
            if not strict_sector_ids:
                raise ValueError(
                    "Every lifted_cauchy component requires strict_sector_ids."
                )
            orthogonal_value = component.get("orthogonal_plan")
            orthogonal_hash = component.get("orthogonal_output_plan_hash")
            if (orthogonal_value is None) != (orthogonal_hash is None):
                raise ValueError(
                    "Component orthogonal_plan and orthogonal_output_plan_hash "
                    "must be supplied together."
                )
            if orthogonal_value is None:
                orthogonal_plan = None
            else:
                orthogonal_plan, _transform, _norms = (
                    _orthogonal_output_plan_and_matrix(
                        compiled_component,
                        precomputed_plan=orthogonal_value,
                        expected_hash=orthogonal_hash,
                    )
                )
            compiled_components.append(compiled_component)
            component_orthogonal_output_plans.append(orthogonal_plan)
            component_records.append(
                {
                    "component_id": component_id,
                    "opportunity_id": str(
                        component.get("opportunity_id", component_id)
                    ),
                    "position": position,
                    "artifact_hash": str(compiled_component.self_hash),
                    "selection_hash": selection_hash,
                    "orthogonal_output_plan_hash": (
                        None
                        if orthogonal_plan is None
                        else str(orthogonal_plan["self_hash"])
                    ),
                    "coordinate_ids": coordinate_ids,
                    "strict_sector_ids": strict_sector_ids,
                    "descriptor_count": len(
                        compiled_component.payload["descriptors"]
                    ),
                }
            )
        channels, component_maps = _composite_channel_registry(
            compiled_components
        )
        source = LiftedCauchyPolynomialSource(
            compiled_components[0],
            source_config=config.get("source"),
            type_map=type_map,
            channel_registry=channels,
        )
        semantic, coordinate_ids = _composite_descriptor_identity(
            compiled_components,
            component_records,
            component_maps,
            source.source_plan_hash,
        )
        return {
            "lifted_cauchy_compiled_components": tuple(
                compiled.to_dict() for compiled in compiled_components
            ),
            "lifted_cauchy_component_records": tuple(component_records),
            "lifted_cauchy_component_orthogonal_output_plans": tuple(
                component_orthogonal_output_plans
            ),
            "lifted_cauchy_component_channel_maps": component_maps,
            "lifted_cauchy_source": dict(source.config),
            "lifted_cauchy_source_plan_hash": source.source_plan_hash,
            "lifted_cauchy_artifact_hash": _canonical_source_hash(semantic),
            "lifted_cauchy_coordinate_ids": coordinate_ids,
            "lifted_cauchy_descriptor_count": sum(
                record["descriptor_count"] for record in component_records
            ),
            "lifted_cauchy_channel_count": len(channels),
            "lifted_cauchy_role_dimension": 2,
            "lifted_cauchy_capabilities": {
                "composite": True,
                "component_count": len(component_records),
                "shared_source_materialization": True,
            },
        }
    compiled = (
        _load_compiled_artifact(compiled_value)
        if compiled_value is not None
        else compile_coupling(request)
    )
    source = LiftedCauchyPolynomialSource(
        compiled,
        source_config=config.get("source"),
        type_map=type_map,
    )
    return {
        "lifted_cauchy_compiled": compiled.to_dict(),
        "lifted_cauchy_source": dict(source.config),
        "lifted_cauchy_source_plan_hash": source.source_plan_hash,
        "lifted_cauchy_artifact_hash": str(compiled.self_hash),
        "lifted_cauchy_descriptor_count": len(compiled.payload["descriptors"]),
        "lifted_cauchy_channel_count": len(compiled.payload["channels"]),
        "lifted_cauchy_role_dimension": int(compiled.payload["role_dimension"]),
        "lifted_cauchy_capabilities": dict(compiled.payload["capabilities"]),
    }


def _canonical_source_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fraction_payload(value):
    value = Fraction(value)
    return {"numerator": int(value.numerator), "denominator": int(value.denominator)}


def _exact_integer_determinant(rows):
    """Return the exact determinant of a square integer matrix."""

    matrix = [list(map(int, row)) for row in rows]
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Expected a square integer matrix.")
    if size == 0:
        return 1
    sign = 1
    previous_pivot = 1
    for column in range(size - 1):
        pivot_row = next(
            (row for row in range(column, size) if matrix[row][column] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != column:
            matrix[column], matrix[pivot_row] = matrix[pivot_row], matrix[column]
            sign = -sign
        pivot = matrix[column][column]
        for row in range(column + 1, size):
            for inner_column in range(column + 1, size):
                numerator = (
                    matrix[row][inner_column] * pivot
                    - matrix[row][column] * matrix[column][inner_column]
                )
                if numerator % previous_pivot:
                    raise RuntimeError("Fraction-free determinant lost exact divisibility.")
                matrix[row][inner_column] = numerator // previous_pivot
            matrix[row][column] = 0
        previous_pivot = pivot
    return sign * matrix[-1][-1]


def _artifact_channels(compiled):
    channels = tuple(
        sorted(
            (dict(channel) for channel in compiled.payload["channels"]),
            key=lambda channel: int(channel["channel_index"]),
        )
    )
    indices = tuple(int(channel["channel_index"]) for channel in channels)
    if indices != tuple(range(len(channels))):
        raise ValueError("Lifted-Cauchy artifacts require dense channel indices.")
    keys = set()
    for channel in channels:
        key = (
            str(channel["neighbor_species"]),
            int(channel["radial_channel"]),
            int(channel["l"]),
            str(channel["source_family_id"]),
        )
        if key in keys:
            raise ValueError("Compiled lifted-Cauchy complete channels are duplicated.")
        keys.add(key)
    return channels


def _complete_channel_key(channel):
    return (
        str(channel["neighbor_species"]),
        str(channel["source_family_id"]),
        int(channel["l"]),
        int(channel["radial_channel"]),
    )


def _composite_channel_registry(compiled_components):
    compiled_components = tuple(compiled_components)
    if not compiled_components:
        raise ValueError("A lifted-Cauchy composite requires at least one component.")
    role_dimensions = {
        int(compiled.payload["role_dimension"])
        for compiled in compiled_components
    }
    if role_dimensions != {2}:
        raise ValueError(
            "Lifted-Cauchy composite components must all have role_dimension=2."
        )
    records = {}
    component_keys = []
    for compiled in compiled_components:
        keys = []
        for channel in _artifact_channels(compiled):
            key = _complete_channel_key(channel)
            previous = records.setdefault(key, dict(channel))
            if _complete_channel_key(previous) != key:
                raise RuntimeError("Composite channel registry lost channel identity.")
            keys.append(key)
        component_keys.append(tuple(keys))
    ordered_keys = tuple(sorted(records))
    key_position = {key: index for index, key in enumerate(ordered_keys)}
    channels = tuple(
        {
            **records[key],
            "channel_index": index,
        }
        for index, key in enumerate(ordered_keys)
    )
    component_maps = tuple(
        tuple(key_position[key] for key in keys) for keys in component_keys
    )
    return channels, component_maps


def _composite_descriptor_identity(
    compiled_components,
    component_records,
    component_channel_maps,
    source_plan_hash,
):
    if len(compiled_components) != len(component_records):
        raise ValueError("Composite component records are incomplete.")
    global_sectors = set()
    coordinate_ids = []
    normalized = []
    for position, (compiled, supplied) in enumerate(
        zip(compiled_components, component_records, strict=True)
    ):
        record = dict(supplied)
        if int(record.get("position", -1)) != position:
            raise ValueError("Composite component positions must be dense and ordered.")
        if str(record.get("artifact_hash", "")) != str(compiled.self_hash):
            raise ValueError("Composite component artifact identity changed.")
        descriptor_count = len(compiled.payload["descriptors"])
        if int(record.get("descriptor_count", -1)) != descriptor_count:
            raise ValueError("Composite component descriptor count changed.")
        component_coordinates = tuple(
            str(value) for value in record.get("coordinate_ids", ())
        )
        if len(component_coordinates) != descriptor_count:
            raise ValueError(
                "Every composite descriptor requires one compiler coordinate ID."
            )
        if len(component_coordinates) != len(set(component_coordinates)):
            raise ValueError("Composite component coordinate IDs are duplicated.")
        selection_hash = str(record.get("selection_hash", ""))
        if not selection_hash:
            raise ValueError("Every composite component requires a selection hash.")
        component_id = str(record.get("component_id", ""))
        opportunity_id = str(record.get("opportunity_id", component_id))
        if not component_id or not opportunity_id:
            raise ValueError(
                "Composite component and opportunity IDs must be nonempty."
            )
        strict_sector_ids = tuple(
            str(value) for value in record.get("strict_sector_ids", ())
        )
        if not strict_sector_ids or len(strict_sector_ids) != len(
            set(strict_sector_ids)
        ):
            raise ValueError(
                "Composite components require unique compiler strict-sector IDs."
            )
        overlap = global_sectors.intersection(strict_sector_ids)
        if overlap:
            raise ValueError(
                "Composite components overlap a compiler strict output sector."
            )
        global_sectors.update(strict_sector_ids)
        coordinate_ids.extend(component_coordinates)
        normalized.append(
            {
                "component_id": component_id,
                "opportunity_id": opportunity_id,
                "position": position,
                "artifact_hash": str(compiled.self_hash),
                "selection_hash": selection_hash,
                "coordinate_ids": component_coordinates,
                "strict_sector_ids": strict_sector_ids,
                "descriptor_count": descriptor_count,
            }
        )
    if len(coordinate_ids) != len(set(coordinate_ids)):
        raise ValueError("Composite descriptor coordinate IDs are duplicated globally.")
    semantic = {
        "schema": "ye3t_lifted_cauchy_composite_descriptor_v2",
        "components": tuple(normalized),
        "component_channel_maps": tuple(
            tuple(int(value) for value in mapping)
            for mapping in component_channel_maps
        ),
        "source_plan_hash": str(source_plan_hash),
    }
    return semantic, tuple(coordinate_ids)


def _legacy_source_channels(channels):
    radial_channels = {}
    for channel in channels:
        if int(channel["l"]) != 1:
            raise NotImplementedError(
                "The compatibility polynomial source currently supports l=1."
            )
        source_family_id = str(channel["source_family_id"])
        if source_family_id != LIFTED_CAUCHY_SOURCE_FAMILY:
            raise ValueError(
                "The compatibility polynomial source does not implement compiled "
                f"source_family_id={source_family_id!r}."
            )
        family_key = (str(channel["neighbor_species"]), source_family_id)
        radial_channels.setdefault(family_key, []).append(
            int(channel["radial_channel"])
        )
    for family_key, exponents in radial_channels.items():
        ordered = tuple(sorted(exponents))
        if ordered != tuple(range(len(ordered))):
            raise ValueError(
                "The compatibility polynomial source requires dense zero-based radial "
                f"polynomial exponents for {family_key}; got {ordered}."
            )


def _joint_source_groups(channels, source_family_id):
    grouped = {}
    for channel in channels:
        angular_l = int(channel["l"])
        channel_family = str(channel["source_family_id"])
        if channel_family != str(source_family_id):
            raise ValueError(
                "The joint source does not implement compiled source_family_id="
                f"{channel_family!r}."
            )
        if (
            source_family_id == LIFTED_CAUCHY_JOINT_SOURCE_FAMILY
            and angular_l != 1
        ):
            raise NotImplementedError(
                "The v2 joint source identity is certified only for l=1."
            )
        key = (str(channel["neighbor_species"]), channel_family, angular_l)
        grouped.setdefault(key, []).append(channel)
    result = []
    for key in sorted(grouped):
        records = tuple(
            sorted(grouped[key], key=lambda channel: int(channel["radial_channel"]))
        )
        radial_channels = tuple(int(channel["radial_channel"]) for channel in records)
        if radial_channels != tuple(range(len(records))):
            raise ValueError(
                "The joint source requires dense zero-based radial channel indices "
                f"within {key}; got {radial_channels}."
            )
        result.append(
            {
                "neighbor_species": key[0],
                "source_family_id": key[1],
                "l": int(key[2]),
                "channel_indices": tuple(
                    int(channel["channel_index"]) for channel in records
                ),
            }
        )
    return tuple(result)


def _joint_group_record(group):
    angular_l = int(group["l"])
    channel_count = len(group["channel_indices"])
    source_dimension = 2 * channel_count
    alpha = 4
    beta = 2 * angular_l + 2
    polynomials = []
    transform = []
    integer_transform = []
    for q in range(source_dimension):
        coefficients = _shifted_jacobi_power_coefficients(q, alpha, beta)
        norm = _shifted_jacobi_normalization_squared(q, angular_l)
        unnormalized_transform = [0] * source_dimension
        for power, coefficient in enumerate(coefficients):
            pair = power // 2
            if power % 2 == 0:
                unnormalized_transform[2 * pair] += coefficient
                unnormalized_transform[2 * pair + 1] += coefficient
            else:
                unnormalized_transform[2 * pair + 1] += coefficient
        scale = math.sqrt(float(norm))
        integer_transform.append(tuple(unnormalized_transform))
        transform.append(tuple(scale * value for value in unnormalized_transform))
        polynomials.append(
            {
                "q": q,
                "channel_radial_index": q // 2,
                "role_index": q % 2,
                "jacobi_degree": q,
                "total_radial_polynomial_degree": angular_l + q + 2,
                "shifted_jacobi_power_coefficients": coefficients,
                "normalization_squared": _fraction_payload(norm),
            }
        )
    integer_determinant = _exact_integer_determinant(integer_transform)
    if integer_determinant == 0:
        raise RuntimeError("The factorized joint-source transform is singular.")
    binary_transform = np.asarray(transform, dtype=float)
    singular_values = np.linalg.svd(binary_transform, compute_uv=False)
    if not np.all(np.isfinite(singular_values)) or singular_values[-1] <= 0.0:
        raise RuntimeError(
            "The binary64 factorized joint-source transform is singular."
        )
    return {
        **group,
        "source_dimension": source_dimension,
        "coordinate_order": "q=2*n+s",
        "polynomials": tuple(polynomials),
        "factorized_lowering": {
            "factor_order": "p_even_minus_p_odd,p_odd_by_n",
            "forward_equation": "A_Q=T*A_f",
            "reverse_equation": "bar_A_f=T^T*bar_A_Q",
            "unnormalized_integer_rows": tuple(integer_transform),
            "binary64_matrix": tuple(tuple(float(value) for value in row) for row in transform),
            "unnormalized_integer_determinant": str(integer_determinant),
            "determinant_nonzero_exact": integer_determinant != 0,
            "binary64_minimum_singular_value": float(singular_values[-1]),
        },
    }


def _joint_source_config(payload, compiled, channels, schema):
    if int(compiled.payload["role_dimension"]) != 2:
        raise ValueError("The joint source requires role_dimension=2.")
    cutoff = float(payload.get("cutoff_A", payload.get("cutoff", 5.2)))
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("lifted-Cauchy cutoff must be positive and finite.")
    if schema == LIFTED_CAUCHY_JOINT_SOURCE_SCHEMA:
        source_family_id = LIFTED_CAUCHY_JOINT_SOURCE_FAMILY
        angular_scope = "l=1_origin_regular_solid_harmonic"
    elif schema == LIFTED_CAUCHY_MIXED_L_SOURCE_SCHEMA:
        source_family_id = LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY
        angular_scope = "l>=0_origin_regular_real_tesseral"
    else:
        raise ValueError(f"Unsupported lifted-Cauchy joint source schema {schema!r}.")
    groups = tuple(
        _joint_group_record(group)
        for group in _joint_source_groups(channels, source_family_id)
    )
    periodic_image_mode = str(payload.get("periodic_image_mode", "all_images"))
    if periodic_image_mode not in {"all_images", "explicit", "nonperiodic"}:
        raise ValueError(
            "periodic_image_mode must be all_images, explicit, or nonperiodic."
        )
    neighbor_backend = str(payload.get("neighbor_backend", "auto"))
    if neighbor_backend not in {"auto", "ase", "matscipy"}:
        raise ValueError("neighbor_backend must be auto, ase, or matscipy.")
    semantic = {
        "schema": schema,
        "cutoff_A": cutoff,
        "source_family_id": source_family_id,
        "role_dimension": 2,
        "angular_scope": angular_scope,
        "radial_coordinate": "x=r/r_c",
        "radial_measure": "x^2_dx",
        "envelope": "(1-x)^2",
        "radial_channel_semantics": "q=2*n+s",
        "roles": (
            {"id": "even_shell_difference", "index": 0},
            {"id": "odd_shell", "index": 1},
        ),
        "groups": groups,
        "certificates": {
            "radial_source_gram": "identity_exact",
            "source_span_dimension": "2*C_per_species",
            "cutoff_value": "zero_exact",
            "cutoff_first_derivative": "zero_exact",
            "factorized_forward": "A_Q=T*A_f",
            "factorized_reverse": "bar_A_f=T^T*bar_A_Q",
            "runtime_gram_solve": False,
        },
        "density_normalization": "none",
    }
    source_plan_hash = _canonical_source_hash(semantic)
    normalized = {
        **semantic,
        "source_plan_hash": source_plan_hash,
        "periodic_image_mode": periodic_image_mode,
        "neighbor_backend": neighbor_backend,
    }
    supplied_hash = payload.get("source_plan_hash")
    if supplied_hash is not None and str(supplied_hash) != source_plan_hash:
        raise ValueError("Lifted-Cauchy joint source-plan hash mismatch.")
    generated_keys = {
        "source_family_id",
        "role_dimension",
        "angular_scope",
        "radial_coordinate",
        "radial_measure",
        "envelope",
        "radial_channel_semantics",
        "roles",
        "groups",
        "certificates",
        "density_normalization",
    }
    for key in generated_keys.intersection(payload):
        if json.loads(json.dumps(payload[key])) != json.loads(json.dumps(normalized[key])):
            raise ValueError(f"Lifted-Cauchy joint source field {key!r} is inconsistent.")
    return normalized


def _normalize_source_config(payload, compiled, channels=None):
    payload = dict(payload or {})
    channels = _artifact_channels(compiled) if channels is None else tuple(channels)
    schema = str(payload.get("schema", LIFTED_CAUCHY_SOURCE_SCHEMA))
    if schema in {
        LIFTED_CAUCHY_JOINT_SOURCE_SCHEMA,
        LIFTED_CAUCHY_MIXED_L_SOURCE_SCHEMA,
    }:
        allowed = {
            "schema",
            "cutoff_A",
            "cutoff",
            "source_family_id",
            "role_dimension",
            "angular_scope",
            "radial_coordinate",
            "radial_measure",
            "envelope",
            "radial_channel_semantics",
            "roles",
            "groups",
            "certificates",
            "density_normalization",
            "source_plan_hash",
            "periodic_image_mode",
            "neighbor_backend",
        }
        extras = sorted(set(payload) - allowed)
        if extras:
            raise ValueError(f"Unsupported lifted-Cauchy joint source keys: {extras}")
        return _joint_source_config(payload, compiled, channels, schema)
    allowed = {
        "schema",
        "cutoff_A",
        "cutoff",
        "envelope",
        "radial_coordinate",
        "primitive_radial",
        "source_family_id",
        "radial_channel_semantics",
        "roles",
        "density_normalization",
        "periodic_image_mode",
        "neighbor_backend",
        "source_plan_hash",
    }
    extras = sorted(set(payload) - allowed)
    if extras:
        raise ValueError(f"Unsupported lifted-Cauchy source keys: {extras}")
    if schema != LIFTED_CAUCHY_SOURCE_SCHEMA:
        raise ValueError(f"Unsupported lifted-Cauchy source schema {schema!r}.")
    cutoff = float(payload.get("cutoff_A", payload.get("cutoff", 5.2)))
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("lifted-Cauchy cutoff must be positive and finite.")
    if str(payload.get("envelope", "one_minus_x_squared")) not in {
        "one_minus_x_squared",
        "(1-x)^2",
    }:
        raise ValueError("The first lifted-Cauchy source requires envelope (1-x)^2.")
    if str(payload.get("radial_coordinate", "r_over_rc")) not in {
        "r_over_rc",
        "x=r/r_c",
    }:
        raise ValueError("The first lifted-Cauchy source requires x=r/r_c.")
    if str(payload.get("primitive_radial", "envelope_times_x_power")) not in {
        "envelope_times_x_power",
        "R_n=e(x)*x^n",
    }:
        raise ValueError(
            "The first lifted-Cauchy source requires R_n=(1-x)^2*x^n."
        )
    source_family_id = str(
        payload.get("source_family_id", LIFTED_CAUCHY_SOURCE_FAMILY)
    )
    if source_family_id != LIFTED_CAUCHY_SOURCE_FAMILY:
        raise ValueError(
            "The first lifted-Cauchy source requires source_family_id="
            f"{LIFTED_CAUCHY_SOURCE_FAMILY!r}."
        )
    radial_channel_semantics = str(
        payload.get("radial_channel_semantics", "zero_based_polynomial_exponent")
    )
    if radial_channel_semantics != "zero_based_polynomial_exponent":
        raise ValueError(
            "The first lifted-Cauchy source requires radial_channel to be the "
            "zero-based polynomial exponent."
        )
    roles = payload.get(
        "roles",
        (
            {"id": "inner", "kind": "one_minus_x"},
            {"id": "outer", "kind": "x"},
        ),
    )
    roles = tuple(dict(role) for role in roles)
    expected_roles = (("inner", "one_minus_x"), ("outer", "x"))
    actual_roles = tuple(
        (str(role.get("id")), str(role.get("kind", role.get("function"))))
        for role in roles
    )
    aliases = {
        "1-x": "one_minus_x",
        "one_minus_x": "one_minus_x",
        "x": "x",
    }
    actual_roles = tuple((name, aliases.get(kind, kind)) for name, kind in actual_roles)
    if actual_roles != expected_roles:
        raise ValueError(
            "The first lifted-Cauchy source requires ordered inner=(1-x), outer=x roles."
        )
    if int(compiled.payload["role_dimension"]) != 2:
        raise ValueError("The first lifted-Cauchy source requires role_dimension=2.")
    _legacy_source_channels(channels)
    density_normalization = str(payload.get("density_normalization", "none"))
    if density_normalization != "none":
        raise ValueError(
            "The first lifted-Cauchy source forbids density normalization and attention."
        )
    periodic_image_mode = str(payload.get("periodic_image_mode", "all_images"))
    if periodic_image_mode not in {"all_images", "explicit", "nonperiodic"}:
        raise ValueError(
            "periodic_image_mode must be all_images, explicit, or nonperiodic."
        )
    neighbor_backend = str(payload.get("neighbor_backend", "auto"))
    if neighbor_backend not in {"auto", "ase", "matscipy"}:
        raise ValueError("neighbor_backend must be auto, ase, or matscipy.")
    semantic = {
        "schema": schema,
        "cutoff_A": cutoff,
        "envelope": "one_minus_x_squared",
        "radial_coordinate": "r_over_rc",
        "primitive_radial": "envelope_times_x_power",
        "source_family_id": LIFTED_CAUCHY_SOURCE_FAMILY,
        "radial_channel_semantics": "zero_based_polynomial_exponent",
        "roles": (
            {"id": "inner", "kind": "one_minus_x"},
            {"id": "outer", "kind": "x"},
        ),
        "density_normalization": "none",
    }
    source_plan_hash = _canonical_source_hash(semantic)
    supplied_hash = payload.get("source_plan_hash")
    if supplied_hash is not None and str(supplied_hash) != source_plan_hash:
        raise ValueError("Lifted-Cauchy compatibility source-plan hash mismatch.")
    return {
        **semantic,
        "source_plan_hash": source_plan_hash,
        "periodic_image_mode": periodic_image_mode,
        "neighbor_backend": neighbor_backend,
    }


class LiftedCauchyPolynomialSource:
    """Hash-bound two-role source with direct and factored exact realizations."""

    def __init__(
        self,
        compiled,
        source_config=None,
        type_map=None,
        source_realization="auto",
        channel_registry=None,
    ):
        self.compiled = _load_compiled_artifact(compiled)
        if channel_registry is None:
            self.channels = _artifact_channels(self.compiled)
        else:
            supplied = tuple(dict(channel) for channel in channel_registry)
            indices = tuple(int(channel["channel_index"]) for channel in supplied)
            if indices != tuple(range(len(supplied))):
                raise ValueError(
                    "A composite source channel registry must have dense indices."
                )
            keys = tuple(_complete_channel_key(channel) for channel in supplied)
            if len(keys) != len(set(keys)):
                raise ValueError("A composite source channel registry is duplicated.")
            self.channels = supplied
        self.config = _normalize_source_config(
            source_config, self.compiled, self.channels
        )
        source_realization = str(source_realization).strip().lower()
        if source_realization not in {"auto", "direct", "factorized"}:
            raise ValueError(
                "source_realization must be auto, direct, or factorized."
            )
        if self.config["schema"] == LIFTED_CAUCHY_SOURCE_SCHEMA:
            if source_realization == "factorized":
                raise ValueError(
                    "The compatibility source has no separate factorized realization."
                )
            self.source_realization = "direct"
        else:
            self.source_realization = (
                "direct" if source_realization == "auto" else source_realization
            )
        self._source_tensor_cache = {}
        self.type_map = {
            str(key): int(value) for key, value in dict(type_map or {}).items()
        }
        missing = sorted(
            {
                str(channel["neighbor_species"])
                for channel in self.channels
            }.difference(self.type_map)
        )
        if missing:
            raise ValueError(
                "type_map is missing compiled neighbor species: " + ", ".join(missing)
            )
        values = tuple(self.type_map.values())
        if len(values) != len(set(values)):
            raise ValueError("type_map values must be unique.")

    @property
    def cutoff(self):
        return float(self.config["cutoff_A"])

    @property
    def source_plan_hash(self):
        return self.config.get("source_plan_hash")

    @property
    def real_component_count(self):
        return max(2 * int(channel["l"]) + 1 for channel in self.channels)

    def _source_tensor(self, values, reference, key):
        cache_key = (key, str(reference.dtype), str(reference.device))
        result = self._source_tensor_cache.get(cache_key)
        if result is None:
            result = torch.as_tensor(
                values, dtype=reference.dtype, device=reference.device
            )
            self._source_tensor_cache[cache_key] = result
        return result

    @staticmethod
    def _power_series_with_derivative(coefficients, x):
        value = torch.zeros_like(x) + float(coefficients[-1])
        derivative = torch.zeros_like(x)
        for coefficient in reversed(coefficients[:-1]):
            derivative = derivative * x + value
            value = value * x + float(coefficient)
        return value, derivative

    def edge_values_with_dx(
        self, displacements, neighbor_types, source_realization=None
    ):
        """Return edge source values and derivatives with respect to ``r_j-r_i``.

        Values have shape ``[edges, channels, roles, 3]`` in the compiler's
        physical real-tesseral order ``(c1,q0,s1)=(x,z,-y)``. Derivatives have
        one trailing Cartesian coordinate. For the joint source, ``direct``
        returns orthogonal Q coordinates while ``factorized`` returns the
        separated f coordinates consumed by the center-local transform.
        """

        realization = (
            self.source_realization
            if source_realization is None
            else str(source_realization).strip().lower()
        )
        if self.config["schema"] == LIFTED_CAUCHY_SOURCE_SCHEMA:
            if realization != "direct":
                raise ValueError("The compatibility source supports only direct execution.")
            return self._legacy_edge_values_with_dx(displacements, neighbor_types)
        if realization == "direct":
            return self._joint_direct_edge_values_with_dx(
                displacements, neighbor_types
            )
        if realization == "factorized":
            return self._joint_factorized_edge_values_with_dx(
                displacements, neighbor_types
            )
        raise ValueError("source_realization must be direct or factorized.")

    def _legacy_edge_values_with_dx(self, displacements, neighbor_types):

        disp = torch.as_tensor(displacements)
        if disp.ndim != 2 or int(disp.shape[1]) != 3:
            raise ValueError("displacements must have shape [edges,3].")
        neighbor_types = torch.as_tensor(
            neighbor_types, dtype=torch.long, device=disp.device
        )
        if tuple(neighbor_types.shape) != (int(disp.shape[0]),):
            raise ValueError("neighbor_types must have one value per edge.")
        edge_count = int(disp.shape[0])
        channel_count = len(self.channels)
        values = disp.new_zeros((edge_count, channel_count, 2, 3))
        derivatives = disp.new_zeros((edge_count, channel_count, 2, 3, 3))
        if not edge_count:
            return values, derivatives

        distance = torch.linalg.norm(disp, dim=1)
        cutoff = torch.as_tensor(self.cutoff, dtype=disp.dtype, device=disp.device)
        active = distance < cutoff
        safe_distance = torch.where(active, distance, torch.ones_like(distance))
        unit = disp / safe_distance.unsqueeze(1)
        physical = torch.stack((unit[:, 0], unit[:, 2], -unit[:, 1]), dim=1)
        physical_map = disp.new_tensor(
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
        )
        angular_dx = (
            physical_map.unsqueeze(0)
            - physical.unsqueeze(2) * unit.unsqueeze(1)
        ) / safe_distance.reshape(-1, 1, 1)
        x = distance / cutoff
        one_minus_x = 1.0 - x
        envelope = one_minus_x.square()
        envelope_dx = -2.0 * one_minus_x
        role_values = torch.stack((one_minus_x, x), dim=1)
        role_dx = disp.new_tensor((-1.0, 1.0)).reshape(1, 2).expand(edge_count, 2)

        for channel_index, channel in enumerate(self.channels):
            radial_power = int(channel["radial_channel"])
            if radial_power < 0:
                raise ValueError("radial_channel must be nonnegative.")
            x_power = x.pow(radial_power)
            if radial_power == 0:
                x_power_dx = torch.zeros_like(x)
            else:
                x_power_dx = radial_power * x.pow(radial_power - 1)
            radial = envelope * x_power
            radial_dx = envelope_dx * x_power + envelope * x_power_dx
            scalar = role_values * radial.unsqueeze(1)
            scalar_dr = (
                role_dx * radial.unsqueeze(1)
                + role_values * radial_dx.unsqueeze(1)
            ) / cutoff
            required_type = int(self.type_map[str(channel["neighbor_species"])])
            channel_active = active & (neighbor_types == required_type)
            mask = channel_active.to(dtype=disp.dtype).reshape(-1, 1, 1)
            values[:, channel_index] = (
                scalar.unsqueeze(2) * physical.unsqueeze(1) * mask
            )
            radial_part = (
                scalar_dr.unsqueeze(2).unsqueeze(3)
                * physical.unsqueeze(1).unsqueeze(3)
                * unit.unsqueeze(1).unsqueeze(1)
            )
            angular_part = scalar.unsqueeze(2).unsqueeze(3) * angular_dx.unsqueeze(1)
            derivatives[:, channel_index] = (radial_part + angular_part) * mask.unsqueeze(3)
        return values, derivatives

    def _joint_geometry(self, displacements, neighbor_types):
        disp = torch.as_tensor(displacements)
        if disp.ndim != 2 or int(disp.shape[1]) != 3:
            raise ValueError("displacements must have shape [edges,3].")
        neighbor_types = torch.as_tensor(
            neighbor_types, dtype=torch.long, device=disp.device
        )
        if tuple(neighbor_types.shape) != (int(disp.shape[0]),):
            raise ValueError("neighbor_types must have one value per edge.")
        distance = torch.linalg.norm(disp, dim=1)
        cutoff = torch.as_tensor(self.cutoff, dtype=disp.dtype, device=disp.device)
        # A zero-length supplied edge still has the regular-solid-harmonic
        # origin derivative. Self edges are excluded by the neighbor builder.
        active = distance < cutoff
        safe_distance = torch.where(
            distance > 0.0, distance, torch.ones_like(distance)
        )
        unit = disp / safe_distance.unsqueeze(1)
        x = distance / cutoff
        return disp, neighbor_types, x, unit, active, cutoff

    @staticmethod
    def _compiler_ordered_regular_solid(angular_l, geometry):
        disp, _neighbor_types, x, unit, _active, cutoff = geometry
        angular_l = int(angular_l)
        if angular_l == 0:
            return (
                torch.ones_like(x).unsqueeze(1),
                disp.new_zeros((int(disp.shape[0]), 1, 3)),
            )
        if angular_l == 1:
            physical_map = disp.new_tensor(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, -1.0, 0.0),
                )
            )
            return (
                disp @ physical_map.T / cutoff,
                physical_map.unsqueeze(0).expand(int(disp.shape[0]), -1, -1)
                / cutoff,
            )
        harmonics = real_spherical_harmonics_l_from_unit_cartesian(
            angular_l, unit
        )
        _detached_harmonics, angular_dx = (
            real_spherical_harmonics_l_from_cartesian_with_derivatives(
                angular_l, disp
            )
        )
        order = (
            tuple(
                angular_l + magnetic
                for magnetic in range(angular_l, 0, -1)
            )
            + (angular_l,)
            + tuple(
                angular_l - magnetic
                for magnetic in range(1, angular_l + 1)
            )
        )
        index = torch.as_tensor(order, dtype=torch.long, device=disp.device)
        angular = harmonics.index_select(0, index).transpose(0, 1)
        angular_dx = angular_dx.index_select(0, index).permute(1, 0, 2)
        scale = math.sqrt(4.0 * math.pi / (2 * angular_l + 1))
        angular = scale * angular.to(disp)
        angular_dx = scale * angular_dx.to(disp)
        radial_power = x.pow(angular_l)
        if angular_l == 0:
            radial_power_dr = torch.zeros_like(x)
        else:
            radial_power_dr = angular_l * x.pow(angular_l - 1) / cutoff
        solid = radial_power.unsqueeze(1) * angular
        solid_dx = (
            radial_power.unsqueeze(1).unsqueeze(2) * angular_dx
            + radial_power_dr.unsqueeze(1).unsqueeze(2)
            * angular.unsqueeze(2)
            * unit.unsqueeze(1)
        )
        return solid, solid_dx

    def _joint_fill(self, geometry, radial_records):
        disp, neighbor_types, _x, unit, active, cutoff = geometry
        values = disp.new_zeros(
            (
                int(disp.shape[0]),
                len(self.channels),
                2,
                self.real_component_count,
            )
        )
        derivatives = disp.new_zeros(
            (
                int(disp.shape[0]),
                len(self.channels),
                2,
                self.real_component_count,
                3,
            )
        )
        solid_cache = {}
        for group, group_records in zip(
            self.config["groups"], radial_records, strict=True
        ):
            angular_l = int(group["l"])
            if angular_l not in solid_cache:
                solid_cache[angular_l] = self._compiler_ordered_regular_solid(
                    angular_l, geometry
                )
            solid, solid_dx = solid_cache[angular_l]
            width = 2 * angular_l + 1
            required_type = int(self.type_map[str(group["neighbor_species"])])
            mask = (active & (neighbor_types == required_type)).to(disp.dtype)
            mask_value = mask.reshape(-1, 1)
            mask_derivative = mask.reshape(-1, 1, 1)
            for channel_position, role_records in enumerate(group_records):
                channel_index = int(group["channel_indices"][channel_position])
                for role_index, (radial, radial_dx) in enumerate(role_records):
                    value = radial.unsqueeze(1) * solid
                    derivative = (
                        radial.reshape(-1, 1, 1) * solid_dx
                        + radial_dx.reshape(-1, 1, 1)
                        * solid.unsqueeze(2)
                        * unit.unsqueeze(1)
                        / cutoff
                    )
                    values[:, channel_index, role_index, :width] = (
                        value * mask_value
                    )
                    derivatives[:, channel_index, role_index, :width] = (
                        derivative * mask_derivative
                    )
        return values, derivatives

    def _joint_direct_edge_values_with_dx(self, displacements, neighbor_types):
        geometry = self._joint_geometry(displacements, neighbor_types)
        x = geometry[2]
        one_minus_x = 1.0 - x
        envelope = one_minus_x.square()
        envelope_dx = -2.0 * one_minus_x
        records = []
        for group in self.config["groups"]:
            group_records = []
            polynomial_records = tuple(group["polynomials"])
            for channel_position in range(len(group["channel_indices"])):
                role_records = []
                for role_index in range(2):
                    polynomial = polynomial_records[2 * channel_position + role_index]
                    base, base_dx = self._power_series_with_derivative(
                        polynomial["shifted_jacobi_power_coefficients"], x
                    )
                    normalization = math.sqrt(
                        float(
                            Fraction(
                                int(polynomial["normalization_squared"]["numerator"]),
                                int(polynomial["normalization_squared"]["denominator"]),
                            )
                        )
                    )
                    radial = normalization * envelope * base
                    radial_dx = normalization * (
                        envelope_dx * base + envelope * base_dx
                    )
                    role_records.append((radial, radial_dx))
                group_records.append(tuple(role_records))
            records.append(tuple(group_records))
        return self._joint_fill(geometry, tuple(records))

    def _joint_factorized_edge_values_with_dx(
        self, displacements, neighbor_types
    ):
        geometry = self._joint_geometry(displacements, neighbor_types)
        x = geometry[2]
        one_minus_x = 1.0 - x
        envelope2 = one_minus_x.square()
        envelope3 = envelope2 * one_minus_x
        records = []
        for group in self.config["groups"]:
            group_records = []
            for channel_position in range(len(group["channel_indices"])):
                even_degree = 2 * channel_position
                even = x.pow(even_degree)
                even_dx = (
                    torch.zeros_like(x)
                    if even_degree == 0
                    else even_degree * x.pow(even_degree - 1)
                )
                odd_degree = even_degree + 1
                odd = x.pow(odd_degree)
                odd_dx = odd_degree * x.pow(odd_degree - 1)
                first = envelope3 * even
                first_dx = -3.0 * envelope2 * even + envelope3 * even_dx
                second = envelope2 * odd
                second_dx = -2.0 * one_minus_x * odd + envelope2 * odd_dx
                group_records.append(((first, first_dx), (second, second_dx)))
            records.append(tuple(group_records))
        return self._joint_fill(geometry, tuple(records))

    def _transform_density(self, density, transpose=False):
        transformed = torch.zeros_like(density)
        for group_position, group in enumerate(self.config["groups"]):
            indices = tuple(int(value) for value in group["channel_indices"])
            index = torch.as_tensor(indices, dtype=torch.long, device=density.device)
            selected = density.index_select(1, index)
            flat = selected.reshape(int(density.shape[0]), -1, int(density.shape[-1]))
            matrix = self._source_tensor(
                group["factorized_lowering"]["binary64_matrix"],
                density,
                ("factorized_transform", group_position),
            )
            if transpose:
                work = torch.einsum("qp,aqm->apm", matrix, flat)
            else:
                work = torch.einsum("qp,apm->aqm", matrix, flat)
            transformed.index_copy_(1, index, work.reshape_as(selected))
        return transformed

    def materialize(
        self,
        positions,
        atom_types,
        *,
        edge_index=None,
        cell=None,
        shifts=None,
        pbc=None,
        source_realization=None,
    ):
        positions = torch.as_tensor(positions)
        atom_types = torch.as_tensor(
            atom_types, dtype=torch.long, device=positions.device
        )
        if positions.ndim != 2 or int(positions.shape[1]) != 3:
            raise ValueError("positions must have shape [atoms,3].")
        if tuple(atom_types.shape) != (int(positions.shape[0]),):
            raise ValueError("atom_types must have one value per atom.")
        periodic_image_mode = str(self.config["periodic_image_mode"])
        pbc_t = normalize_pbc(pbc, positions.device)
        if periodic_image_mode == "nonperiodic":
            if bool(torch.any(pbc_t)):
                raise ValueError(
                    "periodic_image_mode='nonperiodic' rejects periodic geometry."
                )
            if shifts is not None and bool(
                torch.any(torch.as_tensor(shifts, device=positions.device) != 0)
            ):
                raise ValueError(
                    "periodic_image_mode='nonperiodic' rejects periodic image shifts."
                )
        if periodic_image_mode == "explicit" and edge_index is None:
            raise ValueError(
                "periodic_image_mode='explicit' requires an explicit edge list."
            )
        if edge_index is None:
            if bool(torch.any(pbc_t)):
                raise ValueError(
                    "Periodic lifted-Cauchy evaluation requires an explicit all-image edge list."
                )
            src, dst, disp, _ = directed_edges_bruteforce(
                positions, self.cutoff, cell=None, pbc=None
            )
            edge_index_t = torch.stack((src, dst), dim=0)
        else:
            edge_index_t = torch.as_tensor(
                edge_index, dtype=torch.long, device=positions.device
            )
            disp = edge_displacements_from_indices(
                positions,
                edge_index_t,
                cell=cell,
                shifts=shifts,
            )
            src = edge_index_t[0]
            dst = edge_index_t[1]
        realization = (
            self.source_realization
            if source_realization is None
            else str(source_realization).strip().lower()
        )
        edge_values, edge_dx = self.edge_values_with_dx(
            disp,
            atom_types.index_select(0, dst),
            source_realization=realization,
        )
        density = positions.new_zeros(
            (
                int(positions.shape[0]),
                len(self.channels),
                2,
                self.real_component_count,
            )
        )
        density.index_add_(0, src, edge_values)
        if (
            self.config["schema"]
            in {
                LIFTED_CAUCHY_JOINT_SOURCE_SCHEMA,
                LIFTED_CAUCHY_MIXED_L_SOURCE_SCHEMA,
            }
            and realization == "factorized"
        ):
            density = self._transform_density(density)
        context = {
            "edge_index": edge_index_t,
            "displacements": disp,
            "edge_values": edge_values,
            "edge_dx": edge_dx,
            "source_realization": realization,
        }
        return density, context

    def vjp(self, density_adjoint, context, atom_count=None, include_strain=True):
        """Apply the exact source transpose, including role and basis terms."""

        adjoint = torch.as_tensor(density_adjoint)
        edge_index = context["edge_index"]
        edge_dx = context["edge_dx"]
        displacements = context["displacements"]
        src = edge_index[0]
        dst = edge_index[1]
        if atom_count is None:
            atom_count = int(adjoint.shape[-4])
        expected = (
            int(atom_count),
            len(self.channels),
            2,
            self.real_component_count,
        )
        batched = adjoint.ndim == 5
        if tuple(adjoint.shape[-4:]) != expected or adjoint.ndim not in {4, 5}:
            raise ValueError("density_adjoint has the wrong lifted source shape.")
        work = adjoint if batched else adjoint.unsqueeze(0)
        if (
            self.config["schema"]
            in {
                LIFTED_CAUCHY_JOINT_SOURCE_SCHEMA,
                LIFTED_CAUCHY_MIXED_L_SOURCE_SCHEMA,
            }
            and context.get("source_realization") == "factorized"
        ):
            transformed = []
            for item in work:
                transformed.append(self._transform_density(item, transpose=True))
            work = torch.stack(tuple(transformed), dim=0)
        edge_adjoint = work.index_select(1, src)
        grad_disp = torch.einsum("becsq,ecsqd->bed", edge_adjoint, edge_dx)
        grad_pos = adjoint.new_zeros((int(work.shape[0]), int(atom_count), 3))
        grad_pos.index_add_(1, dst, grad_disp)
        grad_pos.index_add_(1, src, -grad_disp)
        result = {"position_gradient": grad_pos}
        if include_strain:
            result["strain_derivative"] = torch.einsum(
                "bea,ec->bac", grad_disp, displacements
            )
        if not batched:
            result = {key: value.squeeze(0) for key, value in result.items()}
        return result

    def neighbor_data(self, atoms):
        periodic_image_mode = str(self.config["periodic_image_mode"])
        if periodic_image_mode == "explicit":
            raise ValueError(
                "periodic_image_mode='explicit' requires caller-supplied edge data."
            )
        if periodic_image_mode == "nonperiodic" and bool(np.any(atoms.pbc)):
            raise ValueError(
                "periodic_image_mode='nonperiodic' rejects periodic ASE atoms."
            )
        return neighbor_data_from_ase_atoms(
            atoms,
            self.cutoff,
            self.type_map,
            backend=self.config["neighbor_backend"],
        )


class LiftedCauchyTorchEvaluator:
    """Differentiable canonical and factored execution of a compiled artifact."""

    def __init__(self, compiled):
        self.compiled = _load_compiled_artifact(compiled)
        self.channels = _artifact_channels(self.compiled)
        self._tensor_cache = {}
        self._block_rows = {}
        for template in self.compiled.payload["block_templates"]:
            rows = {}
            for row in template["analysis_rows"]:
                rows[
                    (
                        int(row["role_copy_index"]),
                        int(row["angular_copy_index"]),
                        int(row["M"]),
                    )
                ] = tuple(row["symmetric_power_terms"])
            self._block_rows[str(template["template_id"])] = rows
        self._outer_rows = {}
        for template in self.compiled.payload["outer_templates"]:
            self._outer_rows[str(template["template_id"])] = {
                int(row["outer_copy_index"]): tuple(row["terms"])
                for row in template["analysis_rows"]
            }

    @property
    def descriptor_count(self):
        return len(self.compiled.payload["descriptors"])

    def _constant(self, value, reference):
        key = (complex(value), str(reference.dtype), str(reference.device))
        cached = self._tensor_cache.get(key)
        if cached is None:
            dtype = (
                torch.complex128
                if reference.dtype in {torch.float64, torch.complex128}
                else torch.complex64
            )
            cached = torch.as_tensor(value, dtype=dtype, device=reference.device)
            self._tensor_cache[key] = cached
        return cached

    def _real_to_complex(self, density):
        if density.ndim != 4:
            raise ValueError("density must have shape [atoms,channels,roles,real_m].")
        expected = (
            len(self.channels),
            int(self.compiled.payload["role_dimension"]),
        )
        if int(density.shape[1]) != expected[0] or int(density.shape[2]) != expected[1]:
            raise ValueError("density channel or role width does not match the artifact.")
        forms = {
            str(record["real_form_id"]): record
            for record in self.compiled.payload["real_forms"]
        }
        bindings = {
            int(record["channel_index"]): str(record["real_form_id"])
            for record in self.compiled.payload["channel_real_form_ids"]
        }
        values = {}
        for channel_position, channel in enumerate(self.channels):
            channel_index = int(channel["channel_index"])
            record = forms[bindings[channel_index]]
            matrix = torch.stack(
                tuple(
                    torch.stack(
                        tuple(
                            self._constant(_binary_complex(value), density)
                            for value in row
                        )
                    )
                    for row in record["real_to_complex_matrix"]
                )
            )
            width = int(matrix.shape[1])
            if int(density.shape[-1]) < width:
                raise ValueError("density real-form width does not match the artifact.")
            physical = density[:, channel_position, :, :width]
            values[channel_index] = physical.to(dtype=matrix.dtype) @ matrix.T
        return values

    def _term_sum(self, records, values, remap_channel=None):
        reference = next(iter(values.values()))
        output = reference.new_zeros((int(reference.shape[0]),))
        for record in records:
            coefficient = self._constant(
                _binary_complex(record["coefficient"]), reference
            )
            term = coefficient.expand_as(output)
            for coordinate in record["coordinates"]:
                if remap_channel is None:
                    channel, role, magnetic = (int(value) for value in coordinate)
                else:
                    channel = int(remap_channel)
                    role, magnetic = (int(value) for value in coordinate)
                term = term * values[channel][:, role, magnetic]
            output = output + term
        return output

    def evaluate(self, density, realization="canonical"):
        values = self._real_to_complex(torch.as_tensor(density))
        realization = str(realization).strip().lower()
        if realization == "canonical":
            if not self.compiled.payload["capabilities"].get("canonical", False):
                raise ValueError("Artifact does not contain canonical execution.")
            outputs = tuple(
                self._term_sum(descriptor["canonical_terms"], values)
                for descriptor in self.compiled.payload["descriptors"]
            )
        elif realization == "factored":
            outputs = self._evaluate_factored(values)
        else:
            raise ValueError("realization must be canonical or factored.")
        if outputs:
            stacked = torch.stack(outputs, dim=1)
        else:
            reference = next(iter(values.values()))
            stacked = reference.real.new_zeros((int(reference.shape[0]), 0))
        if stacked.numel() and not torch.compiler.is_compiling():
            scale = max(1.0, float(stacked.detach().abs().max().cpu()))
            residual = float(stacked.detach().imag.abs().max().cpu())
            if residual > 5.0e-11 * scale:
                raise ValueError(
                    "Lifted-Cauchy physical descriptor has a material imaginary residual."
                )
        return stacked.real

    def _evaluate_factored(self, values):
        if not self.compiled.payload["capabilities"].get(
            "factored_symmetric_power_blocks", False
        ):
            raise ValueError("Artifact does not contain factored execution.")
        shared = {}
        outputs = []
        reference = next(iter(values.values()))
        for descriptor in self.compiled.payload["descriptors"]:
            schedule = descriptor["factored_schedule"]
            label = descriptor["label"]
            blocks = []
            for block_index, template_id in enumerate(schedule["block_template_ids"]):
                channel = int(schedule["block_channel_indices"][block_index])
                role_copy = int(schedule["role_copy_indices"][block_index])
                angular_copy = int(schedule["angular_copy_indices"][block_index])
                Lambda = int(label["block_Lambdas"][block_index])
                components = {}
                for M in range(-Lambda, Lambda + 1):
                    key = (
                        str(template_id), channel, role_copy, angular_copy, M
                    )
                    if key not in shared:
                        rows = self._block_rows[str(template_id)][
                            (role_copy, angular_copy, M)
                        ]
                        shared[key] = self._term_sum(
                            rows, values, remap_channel=channel
                        )
                    components[M] = shared[key]
                blocks.append(components)
            outer_rows = self._outer_rows[str(descriptor["outer_template_id"])][
                int(schedule["outer_copy_index"])
            ]
            parent_factor = self._constant(
                _binary_complex(schedule["parent_factor"]), reference
            )
            output = reference.new_zeros((int(reference.shape[0]),))
            for row in outer_rows:
                coefficient = parent_factor * self._constant(
                    _binary_complex(row["coefficient"]), reference
                )
                term = coefficient.expand_as(output)
                magnetic_tuple = tuple(
                    int(coordinate[0]) for coordinate in row["coordinates"]
                )
                for block_index, magnetic in enumerate(magnetic_tuple):
                    term = term * blocks[block_index][magnetic]
                output = output + term
            outputs.append(output)
        return tuple(outputs)

    def _empty_complex_adjoint(self, values, batch_count):
        return {
            channel: value.new_zeros(
                (int(batch_count),) + tuple(value.shape)
            )
            for channel, value in values.items()
        }

    def _physical_adjoint(self, complex_adjoint, density):
        forms = {
            str(record["real_form_id"]): record
            for record in self.compiled.payload["real_forms"]
        }
        bindings = {
            int(record["channel_index"]): str(record["real_form_id"])
            for record in self.compiled.payload["channel_real_form_ids"]
        }
        result = density.new_zeros(
            (
                int(next(iter(complex_adjoint.values())).shape[0]),
                int(density.shape[0]),
                len(self.channels),
                int(density.shape[2]),
                int(density.shape[3]),
            )
        )
        maximum_imaginary = 0.0
        maximum_scale = 1.0
        for channel_position, channel in enumerate(self.channels):
            channel_index = int(channel["channel_index"])
            matrix = torch.stack(
                tuple(
                    torch.stack(
                        tuple(
                            self._constant(_binary_complex(value), density)
                            for value in row
                        )
                    )
                    for row in forms[bindings[channel_index]][
                        "real_to_complex_matrix"
                    ]
                )
            )
            physical = torch.einsum(
                "barm,mq->barq", complex_adjoint[channel_index], matrix
            )
            if physical.numel():
                maximum_imaginary = max(
                    maximum_imaginary,
                    float(physical.imag.abs().max().cpu()),
                )
                maximum_scale = max(
                    maximum_scale,
                    float(physical.abs().max().cpu()),
                )
            result[:, :, channel_position, :, : int(physical.shape[-1])] = (
                physical.real
            )
        if maximum_imaginary > 5.0e-11 * maximum_scale:
            raise ValueError(
                "Lifted-Cauchy physical covector has a material imaginary residual."
            )
        return result

    def _canonical_explicit_adjoint(self, values, upstream):
        batch_count = int(upstream.shape[0])
        gradients = self._empty_complex_adjoint(values, batch_count)
        for descriptor_index, descriptor in enumerate(
            self.compiled.payload["descriptors"]
        ):
            descriptor_upstream = upstream[:, :, descriptor_index]
            for record in descriptor["canonical_terms"]:
                coefficient = self._constant(
                    _binary_complex(record["coefficient"]),
                    next(iter(values.values())),
                )
                coordinates = tuple(
                    tuple(int(value) for value in coordinate)
                    for coordinate in record["coordinates"]
                )
                factors = tuple(
                    values[channel][:, role, magnetic]
                    for channel, role, magnetic in coordinates
                )
                prefix = [torch.ones_like(factors[0])]
                for factor in factors:
                    prefix.append(prefix[-1] * factor)
                suffix = [None] * (len(factors) + 1)
                suffix[-1] = torch.ones_like(factors[0])
                for index in range(len(factors) - 1, -1, -1):
                    suffix[index] = suffix[index + 1] * factors[index]
                for active, (channel, role, magnetic) in enumerate(coordinates):
                    derivative = coefficient * prefix[active] * suffix[active + 1]
                    gradients[channel][:, :, role, magnetic] += (
                        descriptor_upstream * derivative.unsqueeze(0)
                    )
        return gradients

    def _factored_explicit_adjoint(self, values, upstream):
        batch_count = int(upstream.shape[0])
        reference = next(iter(values.values()))
        gradients = self._empty_complex_adjoint(values, batch_count)
        block_values = {}
        block_upstream = {}
        block_records = {}

        for descriptor_index, descriptor in enumerate(
            self.compiled.payload["descriptors"]
        ):
            schedule = descriptor["factored_schedule"]
            label = descriptor["label"]
            descriptor_blocks = []
            descriptor_keys = []
            for block_index, template_id in enumerate(
                schedule["block_template_ids"]
            ):
                channel = int(schedule["block_channel_indices"][block_index])
                role_copy = int(schedule["role_copy_indices"][block_index])
                angular_copy = int(schedule["angular_copy_indices"][block_index])
                Lambda = int(label["block_Lambdas"][block_index])
                components = {}
                keys = {}
                for M in range(-Lambda, Lambda + 1):
                    key = (
                        str(template_id),
                        channel,
                        role_copy,
                        angular_copy,
                        M,
                    )
                    if key not in block_values:
                        records = self._block_rows[str(template_id)][
                            (role_copy, angular_copy, M)
                        ]
                        block_values[key] = self._term_sum(
                            records, values, remap_channel=channel
                        )
                        block_records[key] = records
                        block_upstream[key] = reference.new_zeros(
                            (batch_count, int(reference.shape[0]))
                        )
                    components[M] = block_values[key]
                    keys[M] = key
                descriptor_blocks.append(components)
                descriptor_keys.append(keys)

            outer_rows = self._outer_rows[str(descriptor["outer_template_id"])][
                int(schedule["outer_copy_index"])
            ]
            parent_factor = self._constant(
                _binary_complex(schedule["parent_factor"]), reference
            )
            descriptor_upstream = upstream[:, :, descriptor_index]
            for row in outer_rows:
                coefficient = parent_factor * self._constant(
                    _binary_complex(row["coefficient"]), reference
                )
                magnetic_tuple = tuple(
                    int(coordinate[0]) for coordinate in row["coordinates"]
                )
                factors = tuple(
                    descriptor_blocks[index][magnetic]
                    for index, magnetic in enumerate(magnetic_tuple)
                )
                prefix = [torch.ones_like(factors[0])]
                for factor in factors:
                    prefix.append(prefix[-1] * factor)
                suffix = [None] * (len(factors) + 1)
                suffix[-1] = torch.ones_like(factors[0])
                for index in range(len(factors) - 1, -1, -1):
                    suffix[index] = suffix[index + 1] * factors[index]
                for active, magnetic in enumerate(magnetic_tuple):
                    scale = coefficient * prefix[active] * suffix[active + 1]
                    key = descriptor_keys[active][magnetic]
                    block_upstream[key] += descriptor_upstream * scale.unsqueeze(0)

        for key, records in block_records.items():
            channel = int(key[1])
            local_upstream = block_upstream[key]
            for record in records:
                coefficient = self._constant(
                    _binary_complex(record["coefficient"]), reference
                )
                coordinates = tuple(
                    (int(coordinate[0]), int(coordinate[1]))
                    for coordinate in record["coordinates"]
                )
                factors = tuple(
                    values[channel][:, role, magnetic]
                    for role, magnetic in coordinates
                )
                prefix = [torch.ones_like(factors[0])]
                for factor in factors:
                    prefix.append(prefix[-1] * factor)
                suffix = [None] * (len(factors) + 1)
                suffix[-1] = torch.ones_like(factors[0])
                for index in range(len(factors) - 1, -1, -1):
                    suffix[index] = suffix[index + 1] * factors[index]
                for active, (role, magnetic) in enumerate(coordinates):
                    derivative = coefficient * prefix[active] * suffix[active + 1]
                    gradients[channel][:, :, role, magnetic] += (
                        local_upstream * derivative.unsqueeze(0)
                    )
        return gradients

    def vjp(
        self,
        density,
        upstream,
        realization="canonical",
        create_graph=False,
        method="explicit",
    ):
        density = torch.as_tensor(density)
        method = str(method).strip().lower()
        realization = str(realization).strip().lower()
        if method not in {"explicit", "autograd"}:
            raise ValueError("Lifted-Cauchy VJP method must be explicit or autograd.")
        if method == "explicit" and create_graph:
            raise NotImplementedError(
                "The explicit lifted-Cauchy VJP does not construct higher derivatives."
            )
        work_density = (
            density.detach()
            if method == "explicit"
            else density
            if density.requires_grad
            else density.detach().requires_grad_(True)
        )
        features = self.evaluate(work_density, realization=realization)
        upstream = torch.as_tensor(
            upstream, dtype=features.dtype, device=features.device
        )
        batched = upstream.ndim == 3
        if tuple(upstream.shape[-2:]) != tuple(features.shape) or upstream.ndim not in {2, 3}:
            raise ValueError(
                "upstream must match [atoms,descriptors] or "
                "[batch,atoms,descriptors]."
            )
        if method == "autograd":
            gradient = torch.autograd.grad(
                (features * upstream).sum(),
                work_density,
                grad_outputs=None,
                create_graph=bool(create_graph),
            )[0] if not batched else torch.autograd.grad(
                features,
                work_density,
                grad_outputs=upstream,
                is_grads_batched=True,
                create_graph=bool(create_graph),
            )[0]
            return features, gradient

        work_upstream = upstream if batched else upstream.unsqueeze(0)
        values = self._real_to_complex(work_density)
        if realization == "canonical":
            complex_adjoint = self._canonical_explicit_adjoint(
                values, work_upstream
            )
        elif realization == "factored":
            complex_adjoint = self._factored_explicit_adjoint(
                values, work_upstream
            )
        else:
            raise ValueError("realization must be canonical or factored.")
        gradient = self._physical_adjoint(complex_adjoint, work_density)
        return features, gradient if batched else gradient.squeeze(0)


class _CompositeLiftedCauchyTorchEvaluator:
    """Concatenate compiler artifacts over one shared physical source."""

    def __init__(self, compiled_components, component_channel_maps):
        self.compiled_components = tuple(compiled_components)
        self.evaluators = tuple(
            LiftedCauchyTorchEvaluator(compiled)
            for compiled in self.compiled_components
        )
        self.component_channel_maps = tuple(
            tuple(int(value) for value in mapping)
            for mapping in component_channel_maps
        )
        if len(self.evaluators) != len(self.component_channel_maps):
            raise ValueError("Composite evaluator channel maps are incomplete.")
        offsets = [0]
        for evaluator in self.evaluators:
            offsets.append(offsets[-1] + evaluator.descriptor_count)
        self.component_feature_offsets = tuple(offsets)
        self._index_cache = {}

    @property
    def descriptor_count(self):
        return self.component_feature_offsets[-1]

    def _channel_index(self, component, reference):
        key = (int(component), str(reference.device))
        result = self._index_cache.get(key)
        if result is None:
            result = torch.as_tensor(
                self.component_channel_maps[component],
                dtype=torch.long,
                device=reference.device,
            )
            self._index_cache[key] = result
        return result

    def evaluate(self, density, realization="canonical"):
        density = torch.as_tensor(density)
        values = []
        for component, evaluator in enumerate(self.evaluators):
            local = density.index_select(
                1, self._channel_index(component, density)
            )
            values.append(evaluator.evaluate(local, realization=realization))
        return torch.cat(tuple(values), dim=1)

    def vjp(
        self,
        density,
        upstream,
        realization="canonical",
        method="explicit",
        create_graph=False,
    ):
        density = torch.as_tensor(density)
        upstream = torch.as_tensor(upstream, dtype=density.dtype, device=density.device)
        if upstream.ndim not in {2, 3}:
            raise ValueError(
                "Composite evaluator upstream must have shape [atoms,features] "
                "or [batch,atoms,features]."
            )
        if tuple(upstream.shape[-2:]) != (
            int(density.shape[0]),
            self.descriptor_count,
        ):
            raise ValueError("Composite evaluator upstream shape is inconsistent.")
        features = self.evaluate(density, realization=realization)
        batched = upstream.ndim == 3
        if batched:
            gradient = density.new_zeros((int(upstream.shape[0]),) + tuple(density.shape))
        else:
            gradient = torch.zeros_like(density)
        for component, evaluator in enumerate(self.evaluators):
            start = self.component_feature_offsets[component]
            stop = self.component_feature_offsets[component + 1]
            component_upstream = upstream[..., start:stop]
            if not bool(torch.any(component_upstream)):
                continue
            index = self._channel_index(component, density)
            local_density = density.index_select(1, index)
            _values, local_gradient = evaluator.vjp(
                local_density,
                component_upstream,
                realization=realization,
                method=method,
                create_graph=create_graph,
            )
            gradient.index_add_(2 if batched else 1, index, local_gradient)
        return features, gradient


class _LiftedCauchyLinearModel(torch.nn.Module):
    """Linear scalar readout over compiler-owned lifted-Cauchy descriptors."""

    def __init__(
        self,
        compiled,
        source_config,
        type_map,
        central_species_order=None,
        coefficients=None,
        offsets=None,
        realization="canonical",
        source_realization="auto",
        compiled_components=None,
        component_records=None,
        component_orthogonal_output_plans=None,
        expected_artifact_hash=None,
    ):
        super().__init__()
        if compiled_components is None:
            self.compiled = _load_compiled_artifact(compiled)
            self.compiled_components = (self.compiled,)
            self.source = LiftedCauchyPolynomialSource(
                self.compiled,
                source_config=source_config,
                type_map=type_map,
                source_realization=source_realization,
            )
            self.evaluator = LiftedCauchyTorchEvaluator(self.compiled)
            self.artifact_hash = str(self.compiled.self_hash)
            self.component_records = None
            self.component_orthogonal_output_plans = None
            self.descriptor_coordinate_ids = tuple(
                _canonical_source_hash(
                    {
                        "artifact_hash": self.artifact_hash,
                        "descriptor_index": index,
                    }
                )
                for index in range(self.evaluator.descriptor_count)
            )
        else:
            self.compiled_components = tuple(
                _load_compiled_artifact(value) for value in compiled_components
            )
            channels, channel_maps = _composite_channel_registry(
                self.compiled_components
            )
            self.compiled = None
            self.source = LiftedCauchyPolynomialSource(
                self.compiled_components[0],
                source_config=source_config,
                type_map=type_map,
                source_realization=source_realization,
                channel_registry=channels,
            )
            self.evaluator = _CompositeLiftedCauchyTorchEvaluator(
                self.compiled_components, channel_maps
            )
            self.component_records = tuple(component_records or ())
            if component_orthogonal_output_plans is None:
                self.component_orthogonal_output_plans = (
                    (None,) * len(self.compiled_components)
                )
            else:
                if len(component_orthogonal_output_plans) != len(
                    self.compiled_components
                ):
                    raise ValueError(
                        "Composite orthogonal-plan count differs from component count."
                    )
                normalized_plans = []
                for compiled_component, supplied in zip(
                    self.compiled_components,
                    component_orthogonal_output_plans,
                    strict=True,
                ):
                    if supplied is None:
                        normalized_plans.append(None)
                    else:
                        plan, _transform, _norms = (
                            _orthogonal_output_plan_and_matrix(
                                compiled_component,
                                precomputed_plan=supplied,
                                expected_hash=supplied["self_hash"],
                            )
                        )
                        normalized_plans.append(plan)
                self.component_orthogonal_output_plans = tuple(normalized_plans)
            semantic, self.descriptor_coordinate_ids = (
                _composite_descriptor_identity(
                    self.compiled_components,
                    self.component_records,
                    channel_maps,
                    self.source.source_plan_hash,
                )
            )
            self.artifact_hash = _canonical_source_hash(semantic)
            if expected_artifact_hash is not None and self.artifact_hash != str(
                expected_artifact_hash
            ):
                raise ValueError("Composite descriptor identity hash changed.")
        self.type_map = dict(self.source.type_map)
        if central_species_order is None:
            central_species_order = tuple(
                key for key, _value in sorted(
                    self.type_map.items(), key=lambda item: item[1]
                )
            )
        self.central_species_order = tuple(str(value) for value in central_species_order)
        if set(self.central_species_order) != set(self.type_map):
            raise ValueError(
                "central_species_order must name every type_map species exactly once."
            )
        if len(self.central_species_order) != len(set(self.central_species_order)):
            raise ValueError("central_species_order must not contain duplicates.")
        self.type_order = tuple(
            int(self.type_map[species]) for species in self.central_species_order
        )
        self.type_position = {
            int(atom_type): index for index, atom_type in enumerate(self.type_order)
        }
        feature_count = self.evaluator.descriptor_count
        if coefficients is None:
            coefficients = np.zeros((len(self.type_order), feature_count))
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (len(self.type_order), feature_count):
            raise ValueError("coefficients have the wrong central-type/feature shape.")
        if offsets is None:
            offsets = np.zeros(len(self.type_order))
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (len(self.type_order),):
            raise ValueError("offsets must have one value per central atom type.")
        self.coefficients = torch.nn.Parameter(
            torch.as_tensor(coefficients, dtype=torch.float64), requires_grad=False
        )
        self.offsets = torch.nn.Parameter(
            torch.as_tensor(offsets, dtype=torch.float64), requires_grad=False
        )
        self.realization = str(realization)
        self.fit_metadata = {}

    def atomic_features(
        self,
        positions,
        atom_types,
        *,
        edge_index=None,
        cell=None,
        shifts=None,
        pbc=None,
        realization=None,
    ):
        density, context = self.source.materialize(
            positions,
            atom_types,
            edge_index=edge_index,
            cell=cell,
            shifts=shifts,
            pbc=pbc,
        )
        features = self.evaluator.evaluate(
            density,
            realization=self.realization if realization is None else realization,
        )
        return features, density, context

    def atomic_energies(self, positions, atom_types, **geometry):
        features, _density, _context = self.atomic_features(
            positions, atom_types, **geometry
        )
        atom_types = torch.as_tensor(
            atom_types, dtype=torch.long, device=features.device
        )
        type_positions = torch.empty_like(atom_types)
        for atom_type, position in self.type_position.items():
            type_positions[atom_types == int(atom_type)] = int(position)
        unknown = ~torch.stack(
            tuple(atom_types == int(value) for value in self.type_order), dim=0
        ).any(dim=0)
        if bool(torch.any(unknown)):
            raise ValueError("atom_types contains a central type absent from type_map.")
        weights = self.coefficients.to(features).index_select(0, type_positions)
        offsets = self.offsets.to(features).index_select(0, type_positions)
        return (features * weights).sum(dim=1) + offsets

    def forward(self, positions, atom_types, **geometry):
        return self.atomic_energies(positions, atom_types, **geometry).sum()

    @property
    def parameter_count(self):
        return len(self.type_order) * (self.evaluator.descriptor_count + 1)

    def regression_rows(self, atoms, *, feature_chunk_size=32):
        """Return physical total-energy and force rows for one structure."""

        feature_chunk_size = int(feature_chunk_size)
        if feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be positive.")
        neighbor_data = self.source.neighbor_data(atoms)
        positions = torch.as_tensor(neighbor_data.positions, dtype=torch.float64)
        cell = torch.as_tensor(neighbor_data.cell, dtype=torch.float64)
        atom_types = torch.as_tensor(neighbor_data.atom_types, dtype=torch.long)
        edge_index = torch.as_tensor(neighbor_data.edge_index, dtype=torch.long)
        shifts = torch.as_tensor(neighbor_data.shifts, dtype=torch.float64)
        density, context = self.source.materialize(
            positions,
            atom_types,
            edge_index=edge_index,
            cell=cell,
            shifts=shifts,
            pbc=atoms.pbc,
        )
        density = density.detach()
        features = self.evaluator.evaluate(density, realization=self.realization)
        atom_count = int(features.shape[0])
        feature_count = int(features.shape[1])
        head_count = len(self.type_order)
        coefficient_count = head_count * feature_count
        energy_row = features.new_zeros((self.parameter_count,))
        site_design = features.new_zeros((atom_count, coefficient_count))
        head_masks = []
        for head, atom_type in enumerate(self.type_order):
            mask = atom_types == int(atom_type)
            head_masks.append(mask)
            start = head * feature_count
            site_design[mask, start : start + feature_count] = features[mask]
            energy_row[start : start + feature_count] = features[mask].sum(dim=0)
            energy_row[coefficient_count + head] = mask.sum()
        if sum(int(mask.sum()) for mask in head_masks) != atom_count:
            raise ValueError("Structure contains a central type absent from type_map.")

        force_rows = features.new_zeros((3 * atom_count, self.parameter_count))
        for start in range(0, coefficient_count, feature_chunk_size):
            stop = min(coefficient_count, start + feature_chunk_size)
            upstream = features.new_zeros((stop - start, atom_count, feature_count))
            for local, parameter in enumerate(range(start, stop)):
                head, feature = divmod(parameter, feature_count)
                upstream[local, head_masks[head], feature] = 1.0
            _values, density_adjoint = self.evaluator.vjp(
                density,
                upstream,
                realization=self.realization,
            )
            source_adjoint = self.source.vjp(
                density_adjoint,
                context,
                atom_count=atom_count,
                include_strain=False,
            )
            force_rows[:, start:stop] = -source_adjoint[
                "position_gradient"
            ].reshape(stop - start, -1).T
        return {
            "energy": energy_row.detach().cpu().numpy(),
            "forces": force_rows.detach().cpu().numpy(),
            "site_design": site_design.detach().cpu().numpy(),
            "atomic_features": features.detach().cpu().numpy(),
            "atom_count": atom_count,
        }

    def evaluate_atoms(self, atoms, *, forces=True, stress=False):
        neighbor_data = self.source.neighbor_data(atoms)
        positions = torch.as_tensor(
            neighbor_data.positions, dtype=torch.float64
        ).requires_grad_(bool(forces or stress))
        cell = torch.as_tensor(neighbor_data.cell, dtype=torch.float64)
        atom_types = torch.as_tensor(neighbor_data.atom_types, dtype=torch.long)
        edge_index = torch.as_tensor(neighbor_data.edge_index, dtype=torch.long)
        shifts = torch.as_tensor(neighbor_data.shifts, dtype=torch.float64)
        energy = self(
            positions,
            atom_types,
            edge_index=edge_index,
            cell=cell,
            shifts=shifts,
            pbc=atoms.pbc,
        )
        result = {"energy": energy}
        if forces:
            result["forces"] = -torch.autograd.grad(
                energy, positions, create_graph=False, retain_graph=bool(stress)
            )[0]
        if stress:
            strain = torch.zeros(
                (3, 3), dtype=positions.dtype, device=positions.device,
                requires_grad=True,
            )
            strained_positions = positions.detach() @ (torch.eye(3, dtype=positions.dtype) + strain).T
            strained_cell = cell @ (torch.eye(3, dtype=cell.dtype) + strain).T
            strained_energy = self(
                strained_positions,
                atom_types,
                edge_index=edge_index,
                cell=strained_cell,
                shifts=shifts,
                pbc=atoms.pbc,
            )
            result["strain_derivative"] = torch.autograd.grad(
                strained_energy, strain
            )[0]
        return result


class _OrdinaryLiftedCauchyLinearModel:
    """One scalar energy from an ordinary ACE backbone and lifted supplement."""

    def __init__(
        self,
        ordinary_bundle,
        ordinary_descriptor,
        lifted_model,
        fit_metadata=None,
    ):
        self.ordinary_bundle = ordinary_bundle
        self.ordinary_descriptor = ordinary_descriptor
        self.lifted_model = lifted_model
        self.fit_metadata = dict(fit_metadata or {})
        self._ye3t_linear_fit_metadata = dict(self.fit_metadata)
        self._ordinary_calculator = None

    @property
    def compiled(self):
        return self.lifted_model.compiled

    @property
    def source(self):
        return self.lifted_model.source

    @property
    def evaluator(self):
        return self.lifted_model.evaluator

    @property
    def coefficients(self):
        return self.lifted_model.coefficients

    @property
    def offsets(self):
        return self.lifted_model.offsets

    def _get_ordinary_calculator(self):
        if self._ordinary_calculator is None:
            from ye3t_ace.ace.linear_ace import LinearACEScalarCalculator

            self._ordinary_calculator = LinearACEScalarCalculator(
                self.ordinary_bundle,
                cutoff=self.ordinary_descriptor.cutoff,
                type_map=self.ordinary_descriptor.type_map,
                force_method="analytic_factorized",
                backend=self.ordinary_descriptor.backend,
                strict_backend=self.ordinary_descriptor.strict_backend,
                validate_backend=self.ordinary_descriptor.validate_backend,
                factorized_descriptor_runtime_policy=(
                    self.ordinary_descriptor.metadata.get(
                        "factorized_descriptor_runtime_policy", None
                    )
                ),
            )
        return self._ordinary_calculator

    def evaluate_atoms(self, atoms, *, forces=True, stress=False):
        probe = atoms.copy()
        probe.calc = self._get_ordinary_calculator()
        ordinary_energy = float(probe.get_potential_energy())
        ordinary_forces = probe.get_forces() if forces else None
        ordinary_strain = None
        if stress:
            ordinary_strain = (
                np.asarray(probe.get_stress(voigt=False), dtype=float)
                * float(probe.get_volume())
            )
        lifted = self.lifted_model.evaluate_atoms(
            atoms, forces=forces, stress=stress
        )
        result = {
            "energy": lifted["energy"]
            + torch.as_tensor(ordinary_energy, dtype=lifted["energy"].dtype)
        }
        if forces:
            result["forces"] = lifted["forces"] + torch.as_tensor(
                ordinary_forces, dtype=lifted["forces"].dtype
            )
        if stress:
            result["strain_derivative"] = lifted[
                "strain_derivative"
            ] + torch.as_tensor(
                ordinary_strain,
                dtype=lifted["strain_derivative"].dtype,
            )
        return result


def lifted_cauchy_model_from_descriptor(descriptor, model_config=None):
    """Build an unfitted or coefficient-initialized model from a descriptor."""

    metadata = dict(descriptor.metadata)
    if metadata.get("descriptor_family") != "linear_lifted_cauchy_scalar":
        raise ValueError("descriptor is not a linear lifted-Cauchy scalar descriptor.")
    config = dict(model_config or {})
    allowed = {"coefficients", "offsets", "realization", "source_realization"}
    extras = sorted(set(config) - allowed)
    if extras:
        raise ValueError(f"Unsupported lifted-Cauchy model construction keys: {extras}")
    compiled_components = metadata.get("lifted_cauchy_compiled_components")
    return _LiftedCauchyLinearModel(
        None if compiled_components is not None else metadata["lifted_cauchy_compiled"],
        metadata["lifted_cauchy_source"],
        descriptor.type_map,
        central_species_order=descriptor.elements,
        coefficients=config.get("coefficients"),
        offsets=config.get("offsets"),
        realization=config.get("realization", "canonical"),
        source_realization=config.get("source_realization", "auto"),
        compiled_components=compiled_components,
        component_records=metadata.get("lifted_cauchy_component_records"),
        component_orthogonal_output_plans=metadata.get(
            "lifted_cauchy_component_orthogonal_output_plans"
        ),
        expected_artifact_hash=metadata.get("lifted_cauchy_artifact_hash"),
    )


def _reference_energy(atoms, key):
    if key in atoms.info:
        return float(atoms.info[key])
    calculator = getattr(atoms, "calc", None)
    if calculator is not None and key in getattr(calculator, "results", {}):
        return float(calculator.results[key])
    raise KeyError(f"Structure is missing energy target {key!r}.")


def _reference_forces(atoms, key):
    if key in atoms.arrays:
        value = atoms.arrays[key]
    else:
        calculator = getattr(atoms, "calc", None)
        if calculator is None or key not in getattr(calculator, "results", {}):
            raise KeyError(f"Structure is missing force target {key!r}.")
        value = calculator.results[key]
    value = np.asarray(value, dtype=float)
    if value.shape != (len(atoms), 3):
        raise ValueError(f"Force target {key!r} must have shape [atoms,3].")
    return value


def build_lifted_cauchy_regression_problem(
    model,
    structures,
    *,
    energy_key="energy",
    force_key="forces",
    energy_weight=1.0,
    force_weight=1.0,
    feature_chunk_size=32,
    fit_coordinate_policy="orthogonal",
):
    """Build the frozen structure-balanced, training-scaled linear problem."""

    structures = list(structures)
    if not structures:
        raise ValueError("At least one training structure is required.")
    energy_weight = float(energy_weight)
    force_weight = float(force_weight)
    if energy_weight < 0.0 or force_weight < 0.0:
        raise ValueError("energy_weight and force_weight must be nonnegative.")
    if energy_weight == 0.0 and force_weight == 0.0:
        raise ValueError("At least one target weight must be positive.")

    rows = []
    energy_targets = []
    force_targets = []
    started = time.perf_counter()
    for atoms in structures:
        row = model.regression_rows(
            atoms, feature_chunk_size=feature_chunk_size
        )
        row["energy_target"] = _reference_energy(atoms, energy_key)
        row["force_target"] = _reference_forces(atoms, force_key)
        rows.append(row)
        energy_targets.append(row["energy_target"] / row["atom_count"])
        force_targets.append(row["force_target"].reshape(-1))
    orthogonal_plan, runtime_from_fit, coordinate_norms = (
        _fit_coordinate_lowering(model, fit_coordinate_policy)
    )
    fit_coordinate_normalization = (
        "pivot_identity"
        if str(fit_coordinate_policy).strip().lower() == "pivot"
        else "compiler_metric_unit_norm"
    )
    coefficient_count = int(model.coefficients.numel())
    for row in rows:
        row["energy"][:coefficient_count] = (
            row["energy"][:coefficient_count] @ runtime_from_fit
        )
        row["forces"][:, :coefficient_count] = (
            row["forces"][:, :coefficient_count] @ runtime_from_fit
        )
        row["site_design"] = row["site_design"] @ runtime_from_fit
    force_target_values = np.concatenate(force_targets)
    feature_minimum_scale = 1.0e-12
    target_minimum_scale = 1.0e-12
    all_site = np.concatenate(
        [np.asarray(row["site_design"], dtype=float) for row in rows], axis=0
    )
    feature_mean = np.mean(all_site, axis=0)
    feature_scale = np.maximum(
        np.std(all_site, axis=0), feature_minimum_scale
    )
    energy_scale = max(float(np.std(energy_targets)), target_minimum_scale)
    force_scale = max(
        float(np.sqrt(np.mean(np.square(force_target_values)))),
        target_minimum_scale,
    )

    design_blocks = []
    target_blocks = []
    row_kinds = []
    structure_count = len(rows)
    head_count = len(model.type_order)
    for structure_index, row in enumerate(rows):
        atom_count = int(row["atom_count"])
        if energy_weight > 0.0:
            multiplier = np.sqrt(energy_weight / structure_count) / energy_scale
            species_fractions = np.asarray(
                row["energy"][coefficient_count:] / atom_count,
                dtype=float,
            )
            mean_site = np.mean(row["site_design"], axis=0)
            design_blocks.append(
                multiplier
                * np.concatenate(
                    ((mean_site - feature_mean) / feature_scale, species_fractions)
                )[None, :]
            )
            target_blocks.append(
                np.asarray(
                    [row["energy_target"] * multiplier / atom_count],
                    dtype=float,
                )
            )
            row_kinds.append(("energy", structure_index, 1))
        if force_weight > 0.0:
            component_count = 3 * atom_count
            multiplier = (
                np.sqrt(force_weight / (structure_count * component_count))
                / force_scale
            )
            force_columns = row["forces"][:, :coefficient_count] / feature_scale
            design_blocks.append(
                multiplier
                * np.column_stack(
                    (force_columns, np.zeros((component_count, head_count)))
                )
            )
            target_blocks.append(row["force_target"].reshape(-1) * multiplier)
            row_kinds.append(("forces", structure_index, component_count))
    scaled_design = np.concatenate(design_blocks, axis=0)
    weighted_target = np.concatenate(target_blocks, axis=0)
    singular_values = np.linalg.svd(scaled_design, compute_uv=False)
    tolerance = (
        np.finfo(float).eps
        * max(scaled_design.shape)
        * (float(singular_values[0]) if singular_values.size else 1.0)
    )
    retained_singular_values = singular_values[singular_values > tolerance]
    full_rank = bool(
        singular_values.size
        and retained_singular_values.size == min(scaled_design.shape)
    )
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if full_rank and singular_values[-1] > 0.0
        else None
    )
    retained_condition_number = (
        float(retained_singular_values[0] / retained_singular_values[-1])
        if retained_singular_values.size
        else None
    )
    return {
        "problem_family": "linear_lifted_cauchy_scalar",
        "artifact_hash": str(model.artifact_hash),
        "source_plan_hash": model.source.source_plan_hash,
        "realization": str(model.realization),
        "source_realization": str(model.source.source_realization),
        "fit_coordinate_policy": str(fit_coordinate_policy).strip().lower(),
        "fit_coordinate_normalization": fit_coordinate_normalization,
        "runtime_from_fit_coordinates": runtime_from_fit,
        "orthogonal_output_plan_hash": (
            None if orthogonal_plan is None else str(orthogonal_plan["self_hash"])
        ),
        "runtime_from_fit_coordinates_hash": _lifted_cauchy_array_hash(
            runtime_from_fit
        ),
        "orthogonal_coordinate_norm_squared_hash": _lifted_cauchy_array_hash(
            coordinate_norms
        ),
        "orthogonal_coordinate_norm_squared": coordinate_norms,
        "type_map": dict(model.type_map),
        "central_species_order": tuple(model.central_species_order),
        "weighted_target": weighted_target,
        "scaled_design": scaled_design,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "penalized_parameter_count": coefficient_count,
        "metadata": {
            "backend": "lifted_cauchy_explicit_product_adjoint_rows",
            "source_plan_hash": model.source.source_plan_hash,
            "source_realization": str(model.source.source_realization),
            "fit_coordinate_policy": str(fit_coordinate_policy).strip().lower(),
            "fit_coordinate_normalization": fit_coordinate_normalization,
            "orthogonal_output_plan_hash": (
                None
                if orthogonal_plan is None
                else str(orthogonal_plan["self_hash"])
            ),
            "objective": "structure_balanced_train_scaled_E1_F1",
            "energy_weight": energy_weight,
            "force_weight": force_weight,
            "energy_target_policy": "total_energy_per_atom",
            "force_target_policy": "per_structure_component_mean",
            "energy_target_scale": energy_scale,
            "force_target_scale": force_scale,
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "feature_scale_policy": "training_site_standard_deviation",
            "feature_minimum_scale": feature_minimum_scale,
            "target_minimum_scale": target_minimum_scale,
            "unpenalized_parameters": "central_species_offsets",
            "central_species_order": model.central_species_order,
            "feature_count_per_species": model.evaluator.descriptor_count,
            "parameter_count": int(scaled_design.shape[1]),
            "row_count": int(scaled_design.shape[0]),
            "numerical_rank": int(np.count_nonzero(singular_values > tolerance)),
            "singular_values": singular_values.tolist(),
            "rank_tolerance": float(tolerance),
            "condition_number": condition_number,
            "condition_number_is_infinite": not full_rank,
            "retained_condition_number": retained_condition_number,
            "largest_singular_value": float(singular_values[0]) if singular_values.size else 0.0,
            "smallest_retained_singular_value": float(
                singular_values[singular_values > tolerance][-1]
            ) if np.any(singular_values > tolerance) else 0.0,
            "row_blocks": tuple(row_kinds),
            "build_seconds": float(time.perf_counter() - started),
        },
    }


def _validate_lifted_fit_runtime(device, evaluation_dtype, accumulation_dtype):
    resolved_device = "cpu" if device is None else str(device).strip().lower()
    if resolved_device not in {"cpu", "cpu:0"}:
        raise NotImplementedError(
            "Lifted-Cauchy force fitting currently supports explicit CPU execution only; "
            "a requested device is never silently replaced."
        )
    if str(evaluation_dtype).strip().lower() not in {
        "float64",
        "torch.float64",
        "numpy.float64",
    }:
        raise ValueError("Lifted-Cauchy fitting requires float64 evaluation.")
    if str(accumulation_dtype).strip().lower() not in {
        "float64",
        "numpy.float64",
    }:
        raise ValueError("Lifted-Cauchy normal equations require float64 accumulation.")
    return "cpu"


def lifted_cauchy_linear_fit_preflight(
    count_report,
    *,
    central_species_order,
    ordinary_feature_count=0,
    structure_atom_counts=None,
    structure_count=None,
    atom_count=None,
    include_force_rows=True,
    feature_chunk_size="auto",
):
    """Report exact fit dimensions without compiling coefficients or data rows."""

    payload = (
        count_report.to_dict()
        if hasattr(count_report, "to_dict")
        else dict(count_report)
    )
    if "report" in payload:
        payload = dict(payload["report"])
    if "descriptor_count" not in payload or "labels" not in payload:
        raise ValueError("Expected a lifted-Cauchy compiler count report.")
    descriptor_count = int(payload["descriptor_count"])
    labels = tuple(payload["labels"])
    if descriptor_count <= 0 or len(labels) != descriptor_count:
        raise ValueError("Lifted-Cauchy count report descriptor count is invalid.")
    central_species_order = tuple(str(value) for value in central_species_order)
    if not central_species_order or len(set(central_species_order)) != len(
        central_species_order
    ):
        raise ValueError("central_species_order must contain unique species.")
    ordinary_feature_count = int(ordinary_feature_count)
    if ordinary_feature_count < 0:
        raise ValueError("ordinary_feature_count must be nonnegative.")
    lifted_feature_count = descriptor_count * len(central_species_order)
    penalized_parameter_count = ordinary_feature_count + lifted_feature_count
    intercept_parameter_count = len(central_species_order)
    parameter_count = penalized_parameter_count + intercept_parameter_count
    feature_chunk = resolve_lifted_cauchy_feature_chunk_size(
        feature_chunk_size, penalized_parameter_count
    )

    if structure_atom_counts is not None:
        if structure_count is not None or atom_count is not None:
            raise ValueError(
                "Use structure_atom_counts or aggregate structure_count/atom_count, not both."
            )
        atom_counts = tuple(int(value) for value in structure_atom_counts)
        if not atom_counts or any(value <= 0 for value in atom_counts):
            raise ValueError("structure_atom_counts must contain positive integers.")
        resolved_structure_count = len(atom_counts)
        resolved_atom_count = sum(atom_counts)
    elif structure_count is not None or atom_count is not None:
        if structure_count is None or atom_count is None:
            raise ValueError("structure_count and atom_count must be supplied together.")
        resolved_structure_count = int(structure_count)
        resolved_atom_count = int(atom_count)
        if resolved_structure_count <= 0 or resolved_atom_count <= 0:
            raise ValueError("structure_count and atom_count must be positive.")
        if resolved_atom_count < resolved_structure_count:
            raise ValueError("atom_count cannot be smaller than structure_count.")
    else:
        resolved_structure_count = None
        resolved_atom_count = None

    fit_shape = None
    if resolved_structure_count is not None:
        energy_rows = resolved_structure_count
        force_rows = 3 * resolved_atom_count if include_force_rows else 0
        row_count = energy_rows + force_rows
        itemsize = np.dtype(np.float64).itemsize
        gram_bytes = parameter_count * parameter_count * itemsize
        fit_shape = {
            "structure_count": int(resolved_structure_count),
            "atom_count": int(resolved_atom_count),
            "energy_row_count": int(energy_rows),
            "force_row_count": int(force_rows),
            "regression_row_count": int(row_count),
            "regression_column_count": int(parameter_count),
            "gram_matrix_bytes": int(gram_bytes),
            "sufficient_statistics_bytes": int(
                gram_bytes + parameter_count * itemsize + itemsize
            ),
            "dense_fallback_design_and_target_bytes": int(
                row_count * (parameter_count + 1) * itemsize
            ),
            "streaming_dataset_passes": 2,
            "streaming_retained_structure_count": 1,
        }
    tensor_ranks = sorted({int(label["rank"]) for label in labels})
    request = dict(payload.get("request", {}))
    manual_labels = request.get("manual_labels", None)
    return {
        "schema": "ye3t_lifted_cauchy_linear_fit_preflight_v1",
        "catalogue_selection": (
            "manual_labels" if manual_labels is not None else "generated_families"
        ),
        "manual_label_count": (
            int(len(manual_labels)) if manual_labels is not None else 0
        ),
        "catalogue_exhaustiveness": (
            "not_certified_for_manual_selection"
            if manual_labels is not None
            else "defined_by_generator_request"
        ),
        "compiler_convention_hash": str(payload.get("convention_hash", "")),
        "descriptor_count_by_family": dict(payload.get("counts_by_family", {})),
        "compiler_descriptor_count": descriptor_count,
        "central_species_order": central_species_order,
        "central_species_count": len(central_species_order),
        "lifted_fit_feature_count": int(lifted_feature_count),
        "ordinary_fit_feature_count": int(ordinary_feature_count),
        "total_fit_feature_count": int(penalized_parameter_count),
        "intercept_parameter_count": int(intercept_parameter_count),
        "linear_parameter_count": int(parameter_count),
        "feature_chunk": feature_chunk,
        "tensor_ranks": tuple(tensor_ranks),
        "maximum_tensor_rank": max(tensor_ranks),
        "fit_shape": fit_shape,
        "coefficient_compilation_performed": False,
        "descriptor_evaluation_performed": False,
        "dataset_access_performed": False,
    }


def resolve_lifted_cauchy_feature_chunk_size(requested, penalized_feature_count):
    """Resolve the public feature-batch setting without geometry or data access."""

    feature_count = int(penalized_feature_count)
    if feature_count <= 0:
        raise ValueError("penalized_feature_count must be positive.")
    if isinstance(requested, str):
        if requested.strip().lower() != "auto":
            raise ValueError("feature_chunk_size must be a positive integer or 'auto'.")
        resolved = feature_count if feature_count <= 32 else 32
        reason = (
            "all_features_for_small_linear_problem"
            if feature_count <= 32
            else "conservative_cpu_vjp_batch"
        )
        normalized_request = "auto"
    else:
        if isinstance(requested, (bool, np.bool_)):
            raise ValueError("feature_chunk_size must be a positive integer or 'auto'.")
        try:
            resolved = int(requested)
            exact_integer = float(requested) == float(resolved)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "feature_chunk_size must be a positive integer or 'auto'."
            ) from error
        if resolved <= 0 or not exact_integer:
            raise ValueError("feature_chunk_size must be a positive integer or 'auto'.")
        reason = "explicit_user_override"
        normalized_request = resolved
    return {
        "policy": "lifted_cauchy_cpu_feature_chunk_v1",
        "requested": normalized_request,
        "resolved": int(resolved),
        "penalized_feature_count": feature_count,
        "reason": reason,
        "hardware_calibrated": False,
    }


def build_lifted_cauchy_normal_equations(
    model,
    structures,
    *,
    energy_key="energy",
    force_key="forces",
    energy_weight=1.0,
    force_weight=1.0,
    feature_chunk_size=32,
    fit_coordinate_policy="orthogonal",
    structure_weights=None,
    structure_weight_key=None,
    structure_group_key=None,
    structure_group_weights=None,
    structure_group_default_weight=None,
    structure_group_normalize_mean=True,
    boltzmann_temperature_K=None,
    boltzmann_energy_key=None,
    boltzmann_weight_nugget=0.0,
    boltzmann_weight_prefactor=1.0,
    boltzmann_normalize_mean=True,
    min_structure_weight=0.0,
    device="cpu",
    evaluation_dtype="float64",
    accumulation_dtype="float64",
    progress=None,
):
    """Accumulate a two-pass, one-structure-at-a-time lifted ridge problem."""

    resolved_device = _validate_lifted_fit_runtime(
        device, evaluation_dtype, accumulation_dtype
    )
    structures = list(structures)
    if not structures:
        raise ValueError("At least one training structure is required.")
    energy_weight = float(energy_weight)
    force_weight = float(force_weight)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (energy_weight, force_weight)
    ) or energy_weight + force_weight <= 0.0:
        raise ValueError("Target weights must be finite, nonnegative, and not both zero.")
    resolved_weights, weight_metadata = structure_fit_weights(
        structures,
        structure_weights=structure_weights,
        structure_weight_key=structure_weight_key,
        structure_group_key=structure_group_key,
        structure_group_weights=structure_group_weights,
        structure_group_default_weight=structure_group_default_weight,
        structure_group_normalize_mean=structure_group_normalize_mean,
        boltzmann_temperature_K=boltzmann_temperature_K,
        boltzmann_energy_key=boltzmann_energy_key,
        boltzmann_weight_nugget=boltzmann_weight_nugget,
        boltzmann_weight_prefactor=boltzmann_weight_prefactor,
        boltzmann_normalize_mean=boltzmann_normalize_mean,
        min_weight=min_structure_weight,
    )
    orthogonal_plan, runtime_from_fit, coordinate_norms = (
        _fit_coordinate_lowering(model, fit_coordinate_policy)
    )
    coefficient_count = int(model.coefficients.numel())
    head_count = len(model.type_order)
    parameter_count = coefficient_count + head_count
    feature_sum = np.zeros(coefficient_count, dtype=np.float64)
    feature_square_sum = np.zeros(coefficient_count, dtype=np.float64)
    energy_per_atom = []
    force_square_sum = 0.0
    force_component_count = 0
    atom_count = 0
    maximum_structure_row_bytes = 0
    scale_pass_started = time.perf_counter()
    for structure_index, atoms in enumerate(structures, start=1):
        if progress is not None:
            progress("scale", structure_index, len(structures))
        row = model.regression_rows(atoms, feature_chunk_size=feature_chunk_size)
        site = np.asarray(row["site_design"], dtype=np.float64) @ runtime_from_fit
        force = (
            np.asarray(row["forces"][:, :coefficient_count], dtype=np.float64)
            @ runtime_from_fit
        )
        energy_target = _reference_energy(atoms, energy_key)
        force_target = _reference_forces(atoms, force_key).reshape(-1)
        if not (
            np.all(np.isfinite(site))
            and np.all(np.isfinite(force))
            and np.all(np.isfinite(force_target))
            and math.isfinite(energy_target)
        ):
            raise ValueError("Lifted-Cauchy fit rows and targets must be finite.")
        feature_sum += np.sum(site, axis=0)
        feature_square_sum += np.sum(site * site, axis=0)
        atom_count += int(row["atom_count"])
        energy_per_atom.append(energy_target / int(row["atom_count"]))
        force_square_sum += float(np.dot(force_target, force_target))
        force_component_count += int(force_target.size)
        maximum_structure_row_bytes = max(
            maximum_structure_row_bytes,
            int(site.nbytes + force.nbytes + force_target.nbytes),
        )
    scale_pass_seconds = float(time.perf_counter() - scale_pass_started)
    feature_mean = feature_sum / atom_count
    feature_variance = (
        feature_square_sum / atom_count - feature_mean * feature_mean
    )
    roundoff_floor = 64.0 * np.finfo(np.float64).eps * np.maximum(
        feature_square_sum / atom_count, 1.0
    )
    if np.any(feature_variance < -roundoff_floor):
        raise FloatingPointError("Lifted-Cauchy feature variance became negative.")
    feature_scale = np.maximum(
        np.sqrt(np.maximum(feature_variance, 0.0)), 1.0e-12
    )
    energy_scale = max(float(np.std(energy_per_atom)), 1.0e-12)
    force_scale = max(
        float(np.sqrt(force_square_sum / force_component_count)), 1.0e-12
    )
    XtX = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    Xty = np.zeros(parameter_count, dtype=np.float64)
    yty = 0.0
    row_count = 0
    structure_count = len(structures)
    normal_equation_pass_started = time.perf_counter()
    for structure_index, atoms in enumerate(structures):
        if progress is not None:
            progress("normal_equations", structure_index + 1, structure_count)
        row = model.regression_rows(atoms, feature_chunk_size=feature_chunk_size)
        atom_count_i = int(row["atom_count"])
        site = np.asarray(row["site_design"], dtype=np.float64) @ runtime_from_fit
        force = (
            np.asarray(row["forces"][:, :coefficient_count], dtype=np.float64)
            @ runtime_from_fit
        )
        force_target = _reference_forces(atoms, force_key).reshape(-1)
        row_weight = float(resolved_weights[structure_index])
        if energy_weight > 0.0 and row_weight > 0.0:
            species_fractions = np.asarray(
                row["energy"][coefficient_count:] / atom_count_i,
                dtype=np.float64,
            )
            energy_row = np.concatenate(
                ((np.mean(site, axis=0) - feature_mean) / feature_scale,
                 species_fractions)
            )
            energy_target = _reference_energy(atoms, energy_key) / atom_count_i
            factor = row_weight * energy_weight / (
                structure_count * energy_scale * energy_scale
            )
            XtX += factor * np.outer(energy_row, energy_row)
            Xty += factor * energy_row * energy_target
            yty += factor * energy_target * energy_target
            row_count += 1
        if force_weight > 0.0 and row_weight > 0.0:
            scaled_force = force / feature_scale
            factor = row_weight * force_weight / (
                structure_count * force.shape[0] * force_scale * force_scale
            )
            XtX[:coefficient_count, :coefficient_count] += (
                factor * (scaled_force.T @ scaled_force)
            )
            Xty[:coefficient_count] += factor * (scaled_force.T @ force_target)
            yty += factor * float(np.dot(force_target, force_target))
            row_count += int(force.shape[0])
    normal_equation_pass_seconds = float(
        time.perf_counter() - normal_equation_pass_started
    )
    fit_coordinate_normalization = (
        "pivot_identity"
        if str(fit_coordinate_policy).strip().lower() == "pivot"
        else "compiler_metric_unit_norm"
    )
    dense_design_bytes = int(row_count * (parameter_count + 1) * 8)
    normal_equation_bytes = int(
        (parameter_count * parameter_count + parameter_count + 1) * 8
    )
    metadata = {
        "backend": "lifted_cauchy_streamed_explicit_product_adjoint_rows",
        "objective": "structure_balanced_train_scaled_E1_F1",
        "energy_weight": energy_weight,
        "force_weight": force_weight,
        "energy_target_policy": "total_energy_per_atom",
        "force_target_policy": "per_structure_component_mean",
        "energy_target_scale": energy_scale,
        "force_target_scale": force_scale,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "feature_scale_policy": "training_site_standard_deviation",
        "feature_scaling_role": "numerical_preconditioner_only",
        "feature_minimum_scale": 1.0e-12,
        "target_minimum_scale": 1.0e-12,
        "unpenalized_parameters": "central_species_offsets",
        "central_species_order": tuple(model.central_species_order),
        "feature_count_per_species": model.evaluator.descriptor_count,
        "parameter_count": parameter_count,
        "penalized_parameter_count": coefficient_count,
        "row_count": row_count,
        "structure_count": structure_count,
        "atom_count": atom_count,
        "force_component_count": force_component_count,
        "dataset_passes": 2,
        "retained_structure_count": 1,
        "retained_force_edge_contributions": False,
        "materialized_training_design": False,
        "dense_fallback_status": "not_used",
        "dense_design_bytes_avoided": dense_design_bytes,
        "normal_equation_bytes": normal_equation_bytes,
        "maximum_structure_row_bytes": maximum_structure_row_bytes,
        "structure_weights": weight_metadata,
        "device": resolved_device,
        "evaluation_dtype": "float64",
        "accumulation_dtype": "float64",
        "fit_coordinate_policy": str(fit_coordinate_policy).strip().lower(),
        "fit_coordinate_normalization": fit_coordinate_normalization,
        "orthogonal_output_plan_hash": (
            None if orthogonal_plan is None else str(orthogonal_plan["self_hash"])
        ),
        "source_plan_hash": model.source.source_plan_hash,
        "source_realization": str(model.source.source_realization),
    }
    return {
        "problem_family": "linear_lifted_cauchy_scalar_streamed_gram",
        "artifact_hash": str(model.artifact_hash),
        "source_plan_hash": model.source.source_plan_hash,
        "realization": str(model.realization),
        "source_realization": str(model.source.source_realization),
        "fit_coordinate_policy": str(fit_coordinate_policy).strip().lower(),
        "fit_coordinate_normalization": fit_coordinate_normalization,
        "runtime_from_fit_coordinates": runtime_from_fit,
        "orthogonal_output_plan_hash": metadata["orthogonal_output_plan_hash"],
        "orthogonal_coordinate_norm_squared": coordinate_norms,
        "type_map": dict(model.type_map),
        "central_species_order": tuple(model.central_species_order),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "fit_coordinate_metric": np.eye(coefficient_count, dtype=np.float64),
        "penalized_parameter_count": coefficient_count,
        "XtX": XtX,
        "Xty": Xty,
        "yty": float(yty),
        "metadata": metadata,
        "runtime_metrics": {
            "scale_pass_seconds": scale_pass_seconds,
            "normal_equation_pass_seconds": normal_equation_pass_seconds,
            "accumulation_seconds": (
                scale_pass_seconds + normal_equation_pass_seconds
            ),
        },
    }


def _ordinary_streaming_context(descriptor, device):
    from ye3t_ace.equivariant_calc.ace_eval_v2 import ACECovariantEvaluator
    config = descriptor.site_basis_config
    evaluator = ACECovariantEvaluator(
        config,
        backend=descriptor.backend,
        strict_backend=descriptor.strict_backend,
        validate_backend=descriptor.validate_backend,
        factorized_descriptor_runtime_policy=descriptor.metadata.get(
            "factorized_descriptor_runtime_policy", None
        ),
    )
    descriptors = tuple(descriptor.descriptor_specs)
    evaluator.precompile_descriptors(descriptors)
    return evaluator, descriptors, torch.device(device)


def _ordinary_streaming_rows(descriptor, context, atoms, feature_chunk_size):
    from ye3t_ace.equivariant_calc.gradients import (
        descriptor_sum_position_jacobian_analytic_product,
    )

    evaluator, descriptors, device = context
    neighbor_data = neighbor_data_from_ase_atoms(
        atoms, descriptor.cutoff, descriptor.type_map
    )
    positions = torch.as_tensor(
        np.asarray(atoms.positions, dtype=float),
        dtype=torch.float64,
        device=device,
    )
    cell = torch.as_tensor(
        np.asarray(atoms.cell.array, dtype=float),
        dtype=torch.float64,
        device=device,
    )
    edge_index = torch.as_tensor(
        neighbor_data.edge_index, dtype=torch.long, device=device
    )
    atom_types = torch.as_tensor(
        neighbor_data.atom_types, dtype=torch.long, device=device
    )
    shifts = torch.as_tensor(
        np.asarray(neighbor_data.shifts, dtype=float),
        dtype=torch.float64,
        device=device,
    )
    site, position_jacobian = descriptor_sum_position_jacobian_analytic_product(
        evaluator,
        positions,
        cell,
        edge_index,
        atom_types,
        descriptors,
        shifts=shifts,
        real_if_scalar=True,
        chunk_size=feature_chunk_size,
    )
    return (
        site.detach().cpu().numpy().astype(np.float64, copy=False),
        (-position_jacobian.T).detach().cpu().numpy().astype(
            np.float64, copy=False
        ),
    )


def build_ordinary_lifted_cauchy_normal_equations(
    lifted_model,
    ordinary_descriptor,
    structures,
    *,
    energy_key="energy",
    force_key="forces",
    energy_weight=1.0,
    force_weight=1.0,
    feature_chunk_size=32,
    fit_coordinate_policy="orthogonal",
    structure_weights=None,
    structure_weight_key=None,
    structure_group_key=None,
    structure_group_weights=None,
    structure_group_default_weight=None,
    structure_group_normalize_mean=True,
    boltzmann_temperature_K=None,
    boltzmann_energy_key=None,
    boltzmann_weight_nugget=0.0,
    boltzmann_weight_prefactor=1.0,
    boltzmann_normalize_mean=True,
    min_structure_weight=0.0,
    device="cpu",
    evaluation_dtype="float64",
    accumulation_dtype="float64",
    progress=None,
):
    """Accumulate joint ordinary/lifted Gram blocks without dataset-wide rows."""

    if len(lifted_model.type_order) != 1:
        raise NotImplementedError(
            "The first joint ordinary/lifted fit supports one central species."
        )
    if dict(ordinary_descriptor.type_map) != dict(lifted_model.type_map):
        raise ValueError("Ordinary and lifted descriptors must use the same type_map.")
    if not np.isclose(
        float(ordinary_descriptor.cutoff),
        float(lifted_model.source.cutoff),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError("Ordinary and lifted descriptors must use the same cutoff.")
    resolved_device = _validate_lifted_fit_runtime(
        device, evaluation_dtype, accumulation_dtype
    )
    structures = list(structures)
    if not structures:
        raise ValueError("At least one training structure is required.")
    energy_weight = float(energy_weight)
    force_weight = float(force_weight)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (energy_weight, force_weight)
    ) or energy_weight + force_weight <= 0.0:
        raise ValueError("Target weights must be finite, nonnegative, and not both zero.")
    resolved_weights, weight_metadata = structure_fit_weights(
        structures,
        structure_weights=structure_weights,
        structure_weight_key=structure_weight_key,
        structure_group_key=structure_group_key,
        structure_group_weights=structure_group_weights,
        structure_group_default_weight=structure_group_default_weight,
        structure_group_normalize_mean=structure_group_normalize_mean,
        boltzmann_temperature_K=boltzmann_temperature_K,
        boltzmann_energy_key=boltzmann_energy_key,
        boltzmann_weight_nugget=boltzmann_weight_nugget,
        boltzmann_weight_prefactor=boltzmann_weight_prefactor,
        boltzmann_normalize_mean=boltzmann_normalize_mean,
        min_weight=min_structure_weight,
    )
    ordinary_context = _ordinary_streaming_context(
        ordinary_descriptor, resolved_device
    )
    orthogonal_plan, lifted_lowering, coordinate_norms = (
        _fit_coordinate_lowering(lifted_model, fit_coordinate_policy)
    )
    ordinary_count = len(tuple(ordinary_descriptor.descriptor_specs))
    lifted_count = int(lifted_model.coefficients.numel())
    feature_count = ordinary_count + lifted_count
    parameter_count = feature_count + 1
    runtime_from_fit = np.zeros((feature_count, feature_count), dtype=np.float64)
    runtime_from_fit[:ordinary_count, :ordinary_count] = np.eye(ordinary_count)
    runtime_from_fit[ordinary_count:, ordinary_count:] = lifted_lowering
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    feature_square_sum = np.zeros(feature_count, dtype=np.float64)
    energy_per_atom = []
    force_square_sum = 0.0
    force_component_count = 0
    atom_count = 0
    maximum_structure_row_bytes = 0
    scale_pass_started = time.perf_counter()
    for structure_index, atoms in enumerate(structures, start=1):
        if progress is not None:
            progress("scale", structure_index, len(structures))
        ordinary_site, ordinary_force = _ordinary_streaming_rows(
            ordinary_descriptor, ordinary_context, atoms, feature_chunk_size
        )
        lifted = lifted_model.regression_rows(
            atoms, feature_chunk_size=feature_chunk_size
        )
        lifted_site = (
            np.asarray(lifted["site_design"], dtype=np.float64)
            @ lifted_lowering
        )
        lifted_force = (
            np.asarray(lifted["forces"][:, :lifted_count], dtype=np.float64)
            @ lifted_lowering
        )
        if ordinary_site.shape[0] != lifted_site.shape[0]:
            raise RuntimeError("Ordinary and lifted site rows have different atom counts.")
        site = np.column_stack((ordinary_site, lifted_site))
        force = np.column_stack((ordinary_force, lifted_force))
        energy_target = _reference_energy(atoms, energy_key)
        force_target = _reference_forces(atoms, force_key).reshape(-1)
        if not (
            np.all(np.isfinite(site))
            and np.all(np.isfinite(force))
            and np.all(np.isfinite(force_target))
            and math.isfinite(energy_target)
        ):
            raise ValueError("Joint ordinary/lifted rows and targets must be finite.")
        feature_sum += np.sum(site, axis=0)
        feature_square_sum += np.sum(site * site, axis=0)
        atom_count += len(atoms)
        energy_per_atom.append(energy_target / len(atoms))
        force_square_sum += float(np.dot(force_target, force_target))
        force_component_count += int(force_target.size)
        maximum_structure_row_bytes = max(
            maximum_structure_row_bytes,
            int(site.nbytes + force.nbytes + force_target.nbytes),
        )
    scale_pass_seconds = float(time.perf_counter() - scale_pass_started)
    feature_mean = feature_sum / atom_count
    variance = feature_square_sum / atom_count - feature_mean * feature_mean
    roundoff_floor = 64.0 * np.finfo(np.float64).eps * np.maximum(
        feature_square_sum / atom_count, 1.0
    )
    if np.any(variance < -roundoff_floor):
        raise FloatingPointError("Joint feature variance became negative.")
    feature_scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1.0e-12)
    energy_scale = max(float(np.std(energy_per_atom)), 1.0e-12)
    force_scale = max(
        float(np.sqrt(force_square_sum / force_component_count)), 1.0e-12
    )
    XtX = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    Xty = np.zeros(parameter_count, dtype=np.float64)
    yty = 0.0
    row_count = 0
    structure_count = len(structures)
    normal_equation_pass_started = time.perf_counter()
    for structure_index, atoms in enumerate(structures):
        if progress is not None:
            progress("normal_equations", structure_index + 1, structure_count)
        ordinary_site, ordinary_force = _ordinary_streaming_rows(
            ordinary_descriptor, ordinary_context, atoms, feature_chunk_size
        )
        lifted = lifted_model.regression_rows(
            atoms, feature_chunk_size=feature_chunk_size
        )
        site = np.column_stack(
            (
                ordinary_site,
                np.asarray(lifted["site_design"], dtype=np.float64)
                @ lifted_lowering,
            )
        )
        force = np.column_stack(
            (
                ordinary_force,
                np.asarray(lifted["forces"][:, :lifted_count], dtype=np.float64)
                @ lifted_lowering,
            )
        )
        force_target = _reference_forces(atoms, force_key).reshape(-1)
        row_weight = float(resolved_weights[structure_index])
        if energy_weight > 0.0 and row_weight > 0.0:
            energy_row = np.concatenate(
                ((np.mean(site, axis=0) - feature_mean) / feature_scale, (1.0,))
            )
            energy_target = _reference_energy(atoms, energy_key) / len(atoms)
            factor = row_weight * energy_weight / (
                structure_count * energy_scale * energy_scale
            )
            XtX += factor * np.outer(energy_row, energy_row)
            Xty += factor * energy_row * energy_target
            yty += factor * energy_target * energy_target
            row_count += 1
        if force_weight > 0.0 and row_weight > 0.0:
            scaled_force = force / feature_scale
            factor = row_weight * force_weight / (
                structure_count * force.shape[0] * force_scale * force_scale
            )
            XtX[:feature_count, :feature_count] += factor * (
                scaled_force.T @ scaled_force
            )
            Xty[:feature_count] += factor * (scaled_force.T @ force_target)
            yty += factor * float(np.dot(force_target, force_target))
            row_count += int(force.shape[0])
    normal_equation_pass_seconds = float(
        time.perf_counter() - normal_equation_pass_started
    )
    metadata = {
        "backend": "joint_ordinary_lifted_streamed_analytic_product_adjoint",
        "objective": "structure_balanced_train_scaled_E1_F1",
        "energy_weight": energy_weight,
        "force_weight": force_weight,
        "energy_target_scale": energy_scale,
        "force_target_scale": force_scale,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "feature_scale_policy": "training_site_standard_deviation",
        "feature_scaling_role": "numerical_preconditioner_only",
        "ordinary_feature_count": ordinary_count,
        "lifted_feature_count": lifted_count,
        "parameter_count": parameter_count,
        "penalized_parameter_count": feature_count,
        "row_count": row_count,
        "structure_count": structure_count,
        "atom_count": atom_count,
        "force_component_count": force_component_count,
        "dataset_passes": 2,
        "retained_structure_count": 1,
        "retained_force_edge_contributions": False,
        "materialized_training_design": False,
        "dense_fallback_status": "not_used",
        "dense_design_bytes_avoided": int(row_count * (parameter_count + 1) * 8),
        "normal_equation_bytes": int(
            (parameter_count * parameter_count + parameter_count + 1) * 8
        ),
        "maximum_structure_row_bytes": maximum_structure_row_bytes,
        "ordinary_lifted_cross_gram_retained": True,
        "structure_weights": weight_metadata,
        "device": resolved_device,
        "evaluation_dtype": "float64",
        "accumulation_dtype": "float64",
        "fit_coordinate_policy": str(fit_coordinate_policy).strip().lower(),
        "fit_coordinate_normalization": (
            "pivot_identity"
            if str(fit_coordinate_policy).strip().lower() == "pivot"
            else "compiler_metric_unit_norm"
        ),
        "orthogonal_output_plan_hash": (
            None if orthogonal_plan is None else str(orthogonal_plan["self_hash"])
        ),
        "source_plan_hash": lifted_model.source.source_plan_hash,
        "source_realization": str(lifted_model.source.source_realization),
    }
    return {
        "problem_family": "linear_ordinary_plus_lifted_cauchy_scalar_streamed_gram",
        "artifact_hash": str(lifted_model.artifact_hash),
        "source_plan_hash": lifted_model.source.source_plan_hash,
        "realization": str(lifted_model.realization),
        "source_realization": str(lifted_model.source.source_realization),
        "type_map": dict(lifted_model.type_map),
        "central_species_order": tuple(lifted_model.central_species_order),
        "runtime_from_fit_coordinates": runtime_from_fit,
        "orthogonal_coordinate_norm_squared": coordinate_norms,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "fit_coordinate_metric": np.eye(feature_count, dtype=np.float64),
        "penalized_parameter_count": feature_count,
        "ordinary_feature_count": ordinary_count,
        "lifted_feature_count": lifted_count,
        "XtX": XtX,
        "Xty": Xty,
        "yty": float(yty),
        "metadata": metadata,
        "runtime_metrics": {
            "scale_pass_seconds": scale_pass_seconds,
            "normal_equation_pass_seconds": normal_equation_pass_seconds,
            "accumulation_seconds": (
                scale_pass_seconds + normal_equation_pass_seconds
            ),
        },
    }


def _lifted_cauchy_geometry_hash(atoms):
    digest = hashlib.sha256()
    digest.update(b"ye3t_lifted_cauchy_geometry_v1\0")
    for value in (
        np.asarray(atoms.get_atomic_numbers(), dtype="<i8"),
        np.asarray(atoms.positions, dtype="<f8"),
        np.asarray(atoms.cell.array, dtype="<f8"),
        np.asarray(atoms.pbc, dtype=np.uint8),
    ):
        contiguous = np.ascontiguousarray(value)
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _lifted_cauchy_structure_row_hash(atoms, energy, forces):
    return _canonical_source_hash(
        {
            "schema": "ye3t_lifted_cauchy_structure_row_v2",
            "geometry_hash": _lifted_cauchy_geometry_hash(atoms),
            "target_hash": _lifted_cauchy_array_hash(
                np.asarray((energy,), dtype=np.float64),
                np.asarray(forces, dtype=np.float64),
            ),
        }
    )


def _lifted_cauchy_array_hash(*values):
    digest = hashlib.sha256()
    digest.update(b"ye3t_lifted_cauchy_array_record_v1\0")
    for value in values:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def lifted_cauchy_geometry_row_cache_request(
    model,
    structures,
    *,
    fit_coordinate_policy="orthogonal",
):
    """Return the target-free A1 request identity without evaluating rows."""

    structures = tuple(structures)
    if not structures:
        raise ValueError("At least one structure is required for a row cache.")
    orthogonal_plan, runtime_from_fit, coordinate_norms = (
        _fit_coordinate_lowering(model, fit_coordinate_policy)
    )
    request = {
        "schema": "ye3t_lifted_cauchy_geometry_row_request_v1",
        "artifact_hash": str(model.artifact_hash),
        "source_plan_hash": str(model.source.source_plan_hash),
        "realization": str(model.realization),
        "source_realization": str(model.source.source_realization),
        "fit_coordinate_policy": str(fit_coordinate_policy).strip().lower(),
        "orthogonal_output_plan_hash": (
            None if orthogonal_plan is None else str(orthogonal_plan["self_hash"])
        ),
        "runtime_from_fit_coordinates_hash": _lifted_cauchy_array_hash(
            runtime_from_fit
        ),
        "orthogonal_coordinate_norm_squared_hash": _lifted_cauchy_array_hash(
            coordinate_norms
        ),
        "structure_hashes": tuple(
            _lifted_cauchy_geometry_hash(atoms) for atoms in structures
        ),
        "coefficient_count": int(model.coefficients.numel()),
        "central_species_order": tuple(model.central_species_order),
        "descriptor_coordinate_ids": tuple(model.descriptor_coordinate_ids),
        "row_builder": "analytic_energy_and_cartesian_force_rows_float64_v1",
    }
    return {**request, "request_hash": _canonical_source_hash(request)}


def materialize_lifted_cauchy_geometry_row_cache(
    model,
    structures,
    *,
    feature_chunk_size=32,
    fit_coordinate_policy="orthogonal",
    progress=None,
):
    """Materialize immutable target-free rows in compiler fit coordinates."""

    structures = list(structures)
    if not structures:
        raise ValueError("At least one structure is required for a row cache.")
    orthogonal_plan, runtime_from_fit, coordinate_norms = (
        _fit_coordinate_lowering(model, fit_coordinate_policy)
    )
    request = lifted_cauchy_geometry_row_cache_request(
        model,
        structures,
        fit_coordinate_policy=fit_coordinate_policy,
    )
    coefficient_count = int(model.coefficients.numel())
    head_count = len(model.type_order)
    records = []
    structure_hashes = []
    retained_bytes = 0
    started = time.perf_counter()
    for structure_index, atoms in enumerate(structures, start=1):
        if progress is not None:
            progress("row_cache", structure_index, len(structures))
        row = model.regression_rows(
            atoms, feature_chunk_size=feature_chunk_size
        )
        site = (
            np.asarray(row["site_design"], dtype=np.float64)
            @ runtime_from_fit
        )
        force = (
            np.asarray(row["forces"][:, :coefficient_count], dtype=np.float64)
            @ runtime_from_fit
        )
        species_counts = np.asarray(
            row["energy"][coefficient_count:], dtype=np.float64
        )
        if species_counts.shape != (head_count,) or not np.isclose(
            float(np.sum(species_counts)), float(row["atom_count"])
        ):
            raise RuntimeError("Cached central-species counts are inconsistent.")
        if not (
            np.all(np.isfinite(site))
            and np.all(np.isfinite(force))
        ):
            raise ValueError("Lifted-Cauchy cached geometry rows must be finite.")
        record = {
            "structure_index": structure_index - 1,
            "atom_count": int(row["atom_count"]),
            "site_design": site,
            "force_design": force,
            "species_counts": species_counts,
            "geometry_hash": _lifted_cauchy_geometry_hash(atoms),
        }
        record["row_content_hash"] = _lifted_cauchy_array_hash(
            site,
            force,
            species_counts,
        )
        for value in record.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        record_bytes = sum(
            value.nbytes
            for value in record.values()
            if isinstance(value, np.ndarray)
        )
        record["retained_bytes"] = int(record_bytes)
        retained_bytes += int(record_bytes)
        records.append(record)
        structure_hashes.append(record["geometry_hash"])
    semantic = {
        "schema": "ye3t_lifted_cauchy_geometry_row_cache_v2",
        **{
            key: value
            for key, value in request.items()
            if key != "schema"
        },
        "row_content_hashes": tuple(
            record["row_content_hash"] for record in records
        ),
    }
    return {
        **semantic,
        "cache_hash": _canonical_source_hash(semantic),
        "type_map": dict(model.type_map),
        "head_count": head_count,
        "feature_count_per_species": model.evaluator.descriptor_count,
        "descriptor_coordinate_ids": tuple(model.descriptor_coordinate_ids),
        "feature_coordinate_ids": tuple(
            _canonical_source_hash(
                {
                    "central_species": species,
                    "descriptor_coordinate_id": coordinate_id,
                }
            )
            for species in model.central_species_order
            for coordinate_id in model.descriptor_coordinate_ids
        ),
        "runtime_from_fit_coordinates": runtime_from_fit,
        "orthogonal_coordinate_norm_squared": coordinate_norms,
        "records": tuple(records),
        "retained_bytes": int(retained_bytes),
        "build_seconds": float(time.perf_counter() - started),
        "dataset_passes": 1,
    }


def lifted_cauchy_target_cache_request(
    geometry_cache,
    *,
    energy_key="energy",
    force_key="forces",
    target_identity=None,
):
    """Return an A2 request identity without evaluating target values."""

    if geometry_cache.get("schema") != "ye3t_lifted_cauchy_geometry_row_cache_v2":
        raise ValueError("Unsupported lifted-Cauchy geometry-row cache schema.")
    geometry_records = []
    for record in geometry_cache["records"]:
        atom_count = int(record["atom_count"])
        force_component_count = int(record["force_design"].shape[0])
        if atom_count <= 0 or force_component_count != 3 * atom_count:
            raise ValueError("Cached target geometry has an invalid force-row count.")
        geometry_records.append(
            {
                "structure_index": int(record["structure_index"]),
                "geometry_hash": str(record["geometry_hash"]),
                "atom_count": atom_count,
                "force_component_count": force_component_count,
            }
        )
    identity = {} if target_identity is None else dict(target_identity)
    if not identity:
        raise ValueError(
            "A pre-value target request requires a nonempty target_identity."
        )
    request = {
        "schema": "ye3t_lifted_cauchy_target_request_v2",
        "geometry_records": tuple(geometry_records),
        "energy_key": str(energy_key),
        "force_key": str(force_key),
        "target_identity": identity,
    }
    return {**request, "request_hash": _canonical_source_hash(request)}


def materialize_lifted_cauchy_target_cache(
    geometry_cache,
    structures,
    *,
    energy_key="energy",
    force_key="forces",
    target_identity=None,
):
    """Bind raw or residual targets to an existing target-free geometry cache."""

    if geometry_cache.get("schema") != "ye3t_lifted_cauchy_geometry_row_cache_v2":
        raise ValueError("Unsupported lifted-Cauchy geometry-row cache schema.")
    structures = tuple(structures)
    if len(structures) != len(geometry_cache["records"]):
        raise ValueError("Target structures do not match the geometry-row count.")
    request = lifted_cauchy_target_cache_request(
        geometry_cache,
        energy_key=energy_key,
        force_key=force_key,
        target_identity=target_identity,
    )
    records = []
    retained_bytes = 0
    for geometry_record, atoms in zip(
        geometry_cache["records"], structures, strict=True
    ):
        geometry_hash = _lifted_cauchy_geometry_hash(atoms)
        if geometry_hash != str(geometry_record["geometry_hash"]):
            raise ValueError("Target structure geometry does not match its cached row.")
        energy = _reference_energy(atoms, energy_key)
        forces = _reference_forces(atoms, force_key).reshape(-1)
        expected_force_count = int(geometry_record["force_design"].shape[0])
        if forces.shape != (expected_force_count,):
            raise ValueError("Target force shape does not match its cached Jacobian.")
        if not math.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise ValueError("Lifted-Cauchy cached targets must be finite.")
        forces = np.asarray(forces, dtype=np.float64)
        forces.setflags(write=False)
        record = {
            "structure_index": int(geometry_record["structure_index"]),
            "geometry_hash": geometry_hash,
            "energy_target": float(energy),
            "force_target": forces,
            "target_content_hash": _lifted_cauchy_array_hash(
                np.asarray((energy,), dtype=np.float64), forces
            ),
        }
        record["retained_bytes"] = int(forces.nbytes)
        retained_bytes += int(forces.nbytes)
        records.append(record)
    semantic = {
        "schema": "ye3t_lifted_cauchy_target_cache_v3",
        "target_request_schema": str(request["schema"]),
        "request_hash": str(request["request_hash"]),
        "geometry_records": tuple(request["geometry_records"]),
        "energy_key": str(request["energy_key"]),
        "force_key": str(request["force_key"]),
        "target_identity": dict(request["target_identity"]),
        "target_content_hashes": tuple(
            record["target_content_hash"] for record in records
        ),
    }
    return {
        **semantic,
        "cache_hash": _canonical_source_hash(semantic),
        "records": tuple(records),
        "retained_bytes": int(retained_bytes),
    }


def combine_lifted_cauchy_geometry_and_target_caches(
    geometry_cache,
    target_cache,
):
    target_schema = str(target_cache.get("schema", ""))
    if target_schema == "ye3t_lifted_cauchy_target_cache_v2":
        if str(target_cache.get("geometry_cache_hash", "")) != str(
            geometry_cache.get("cache_hash", "")
        ):
            raise ValueError("Lifted-Cauchy geometry and target cache identities differ.")
    elif target_schema != "ye3t_lifted_cauchy_target_cache_v3":
        raise ValueError("Unsupported lifted-Cauchy target cache schema.")
    geometry_records = tuple(geometry_cache["records"])
    target_records = tuple(target_cache["records"])
    if len(geometry_records) != len(target_records):
        raise ValueError("Lifted-Cauchy geometry and target record counts differ.")
    records = []
    structure_hashes = []
    retained_bytes = 0
    for geometry, target in zip(geometry_records, target_records, strict=True):
        if (
            int(geometry["structure_index"]) != int(target["structure_index"])
            or str(geometry["geometry_hash"]) != str(target["geometry_hash"])
        ):
            raise ValueError("Lifted-Cauchy geometry and target records are misaligned.")
        record = {
            **geometry,
            "energy_target": float(target["energy_target"]),
            "force_target": target["force_target"],
        }
        record["row_content_hash"] = _lifted_cauchy_array_hash(
            record["site_design"],
            record["force_design"],
            record["species_counts"],
            np.asarray((record["energy_target"],), dtype=np.float64),
            record["force_target"],
        )
        record["retained_bytes"] = int(
            geometry["retained_bytes"] + target["retained_bytes"]
        )
        retained_bytes += record["retained_bytes"]
        records.append(record)
        structure_hashes.append(
            _canonical_source_hash(
                {
                    "schema": "ye3t_lifted_cauchy_structure_row_v2",
                    "geometry_hash": geometry["geometry_hash"],
                    "target_hash": target["target_content_hash"],
                }
            )
        )
    if target_schema == "ye3t_lifted_cauchy_target_cache_v3":
        expected_geometry = tuple(
            {
                "structure_index": int(record["structure_index"]),
                "geometry_hash": str(record["geometry_hash"]),
                "atom_count": int(record["atom_count"]),
                "force_component_count": int(record["force_design"].shape[0]),
            }
            for record in geometry_records
        )
        if tuple(target_cache.get("geometry_records", ())) != expected_geometry:
            raise ValueError("Lifted-Cauchy target request geometry is misaligned.")
    semantic = {
        "schema": "ye3t_lifted_cauchy_regression_row_cache_v1",
        "artifact_hash": str(geometry_cache["artifact_hash"]),
        "source_plan_hash": str(geometry_cache["source_plan_hash"]),
        "realization": str(geometry_cache["realization"]),
        "source_realization": str(geometry_cache["source_realization"]),
        "fit_coordinate_policy": str(geometry_cache["fit_coordinate_policy"]),
        "orthogonal_output_plan_hash": geometry_cache[
            "orthogonal_output_plan_hash"
        ],
        "runtime_from_fit_coordinates_hash": geometry_cache[
            "runtime_from_fit_coordinates_hash"
        ],
        "orthogonal_coordinate_norm_squared_hash": geometry_cache[
            "orthogonal_coordinate_norm_squared_hash"
        ],
        "structure_hashes": tuple(structure_hashes),
        "coefficient_count": int(geometry_cache["coefficient_count"]),
        "central_species_order": tuple(geometry_cache["central_species_order"]),
        "descriptor_coordinate_ids": tuple(
            geometry_cache["descriptor_coordinate_ids"]
        ),
        "row_content_hashes": tuple(
            record["row_content_hash"] for record in records
        ),
    }
    return {
        **semantic,
        "cache_hash": _canonical_source_hash(semantic),
        "geometry_cache_hash": str(geometry_cache["cache_hash"]),
        "target_cache_hash": str(target_cache["cache_hash"]),
        "type_map": dict(geometry_cache["type_map"]),
        "head_count": int(geometry_cache["head_count"]),
        "feature_count_per_species": int(
            geometry_cache["feature_count_per_species"]
        ),
        "feature_coordinate_ids": tuple(
            geometry_cache["feature_coordinate_ids"]
        ),
        "runtime_from_fit_coordinates": geometry_cache[
            "runtime_from_fit_coordinates"
        ],
        "orthogonal_coordinate_norm_squared": geometry_cache[
            "orthogonal_coordinate_norm_squared"
        ],
        "records": tuple(records),
        "retained_bytes": int(retained_bytes),
        "build_seconds": float(geometry_cache["build_seconds"]),
        "dataset_passes": int(geometry_cache["dataset_passes"]),
    }


def materialize_lifted_cauchy_regression_row_cache(
    model,
    structures,
    *,
    energy_key="energy",
    force_key="forces",
    feature_chunk_size=32,
    fit_coordinate_policy="orthogonal",
    progress=None,
):
    """Compatibility wrapper returning rows with attached targets."""

    structures = tuple(structures)
    geometry_cache = materialize_lifted_cauchy_geometry_row_cache(
        model,
        structures,
        feature_chunk_size=feature_chunk_size,
        fit_coordinate_policy=fit_coordinate_policy,
        progress=progress,
    )
    target_cache = materialize_lifted_cauchy_target_cache(
        geometry_cache,
        structures,
        energy_key=energy_key,
        force_key=force_key,
        target_identity={
            "kind": "attached_structure_targets",
            "energy_key": str(energy_key),
            "force_key": str(force_key),
        },
    )
    return combine_lifted_cauchy_geometry_and_target_caches(
        geometry_cache, target_cache
    )


def build_lifted_cauchy_normal_equations_from_geometry_and_targets(
    geometry_cache,
    target_cache,
    **settings,
):
    """Build the current objective from separately reusable rows and targets."""

    combined = combine_lifted_cauchy_geometry_and_target_caches(
        geometry_cache, target_cache
    )
    problem = build_lifted_cauchy_normal_equations_from_row_cache(
        combined, **settings
    )
    problem["geometry_cache_hash"] = str(geometry_cache["cache_hash"])
    problem["target_cache_hash"] = str(target_cache["cache_hash"])
    problem["metadata"]["geometry_cache_hash"] = str(
        geometry_cache["cache_hash"]
    )
    problem["metadata"]["target_cache_hash"] = str(target_cache["cache_hash"])
    return problem


def lifted_cauchy_normal_equation_cache_request(
    cache,
    *,
    record_indices=None,
    selected_feature_coordinate_ids=None,
    energy_weight=1.0,
    force_weight=1.0,
    structure_weights=None,
):
    """Return the A3 request identity without accumulating a Gram matrix."""

    if cache.get("schema") != "ye3t_lifted_cauchy_regression_row_cache_v1":
        raise ValueError("Unsupported lifted-Cauchy row-cache schema.")
    records = tuple(cache["records"])
    if record_indices is None:
        record_indices = tuple(range(len(records)))
    else:
        record_indices = tuple(int(value) for value in record_indices)
    if not record_indices or len(record_indices) != len(set(record_indices)):
        raise ValueError("record_indices must be nonempty and unique.")
    if any(index < 0 or index >= len(records) for index in record_indices):
        raise IndexError("record_indices contains an out-of-range value.")
    if selected_feature_coordinate_ids is None:
        selected_feature_coordinate_ids = tuple(cache["feature_coordinate_ids"])
    else:
        selected_feature_coordinate_ids = tuple(
            str(value) for value in selected_feature_coordinate_ids
        )
    available = set(cache["feature_coordinate_ids"])
    if (
        not selected_feature_coordinate_ids
        or len(selected_feature_coordinate_ids)
        != len(set(selected_feature_coordinate_ids))
        or not set(selected_feature_coordinate_ids).issubset(available)
    ):
        raise ValueError(
            "selected_feature_coordinate_ids must be unique cached coordinates."
        )
    energy_weight = float(energy_weight)
    force_weight = float(force_weight)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (energy_weight, force_weight)
    ) or energy_weight + force_weight <= 0.0:
        raise ValueError("Target weights must be finite, nonnegative, and not both zero.")
    if structure_weights is None:
        weights = np.ones(len(record_indices), dtype=np.float64)
    else:
        weights = np.asarray(structure_weights, dtype=np.float64)
        if weights.shape != (len(record_indices),):
            raise ValueError("structure_weights must have one value per selected record.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("structure_weights must be finite and nonnegative.")
    request = {
        "schema": "ye3t_lifted_cauchy_normal_equation_request_v1",
        "objective": "structure_balanced_train_scaled_E1_F1",
        "row_cache_hash": str(cache["cache_hash"]),
        "selected_record_indices": record_indices,
        "selected_feature_coordinate_ids": selected_feature_coordinate_ids,
        "energy_weight": energy_weight,
        "force_weight": force_weight,
        "structure_weight_hash": _lifted_cauchy_array_hash(weights),
    }
    return {**request, "request_hash": _canonical_source_hash(request)}


def build_lifted_cauchy_normal_equations_from_row_cache(
    cache,
    *,
    record_indices=None,
    feature_indices=None,
    feature_coordinate_ids=None,
    descriptor_coordinate_ids=None,
    energy_weight=1.0,
    force_weight=1.0,
    structure_weights=None,
):
    """Build one exact child Gram problem from immutable parent fit rows."""

    if cache.get("schema") != "ye3t_lifted_cauchy_regression_row_cache_v1":
        raise ValueError("Unsupported lifted-Cauchy row-cache schema.")
    records = tuple(cache["records"])
    if record_indices is None:
        record_indices = tuple(range(len(records)))
    else:
        record_indices = tuple(int(value) for value in record_indices)
    if not record_indices or len(record_indices) != len(set(record_indices)):
        raise ValueError("record_indices must be nonempty and unique.")
    if any(index < 0 or index >= len(records) for index in record_indices):
        raise IndexError("record_indices contains an out-of-range value.")
    coefficient_count = int(cache["coefficient_count"])
    selection_modes = sum(
        value is not None
        for value in (
            feature_indices,
            feature_coordinate_ids,
            descriptor_coordinate_ids,
        )
    )
    if selection_modes > 1:
        raise ValueError(
            "Select cached features by indices, feature coordinate IDs, or "
            "descriptor coordinate IDs, not more than one."
        )
    if descriptor_coordinate_ids is not None:
        descriptor_coordinate_ids = tuple(
            str(value) for value in descriptor_coordinate_ids
        )
        if not descriptor_coordinate_ids or len(descriptor_coordinate_ids) != len(
            set(descriptor_coordinate_ids)
        ):
            raise ValueError(
                "descriptor_coordinate_ids must be nonempty and unique."
            )
        descriptor_lookup = {
            value: index
            for index, value in enumerate(cache["descriptor_coordinate_ids"])
        }
        missing = sorted(set(descriptor_coordinate_ids) - set(descriptor_lookup))
        if missing:
            raise KeyError(
                "Unknown descriptor coordinate IDs: " + ", ".join(missing)
            )
        descriptor_count = len(cache["descriptor_coordinate_ids"])
        feature_indices = tuple(
            head * descriptor_count + descriptor_lookup[value]
            for head in range(int(cache["head_count"]))
            for value in descriptor_coordinate_ids
        )
    elif feature_coordinate_ids is not None:
        feature_coordinate_ids = tuple(str(value) for value in feature_coordinate_ids)
        if not feature_coordinate_ids or len(feature_coordinate_ids) != len(
            set(feature_coordinate_ids)
        ):
            raise ValueError(
                "feature_coordinate_ids must be nonempty and unique."
            )
        feature_lookup = {
            value: index
            for index, value in enumerate(cache["feature_coordinate_ids"])
        }
        missing = sorted(set(feature_coordinate_ids) - set(feature_lookup))
        if missing:
            raise KeyError("Unknown feature coordinate IDs: " + ", ".join(missing))
        feature_indices = tuple(feature_lookup[value] for value in feature_coordinate_ids)
    elif feature_indices is None:
        feature_indices = tuple(range(coefficient_count))
    else:
        feature_indices = tuple(int(value) for value in feature_indices)
    if not feature_indices or len(feature_indices) != len(set(feature_indices)):
        raise ValueError("feature_indices must be nonempty and unique.")
    if any(index < 0 or index >= coefficient_count for index in feature_indices):
        raise IndexError("feature_indices contains an out-of-range value.")
    selected_feature_coordinate_ids = tuple(
        cache["feature_coordinate_ids"][index] for index in feature_indices
    )
    selected = tuple(records[index] for index in record_indices)
    feature_index = np.asarray(feature_indices, dtype=int)
    energy_weight = float(energy_weight)
    force_weight = float(force_weight)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (energy_weight, force_weight)
    ) or energy_weight + force_weight <= 0.0:
        raise ValueError("Target weights must be finite, nonnegative, and not both zero.")
    if structure_weights is None:
        weights = np.ones(len(selected), dtype=np.float64)
    else:
        weights = np.asarray(structure_weights, dtype=np.float64)
        if weights.shape != (len(selected),):
            raise ValueError("structure_weights must have one value per selected record.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("structure_weights must be finite and nonnegative.")
    normal_equation_request = lifted_cauchy_normal_equation_cache_request(
        cache,
        record_indices=record_indices,
        selected_feature_coordinate_ids=selected_feature_coordinate_ids,
        energy_weight=energy_weight,
        force_weight=force_weight,
        structure_weights=weights,
    )
    feature_count = len(feature_indices)
    head_count = int(cache["head_count"])
    parameter_count = feature_count + head_count
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    feature_square_sum = np.zeros(feature_count, dtype=np.float64)
    energy_per_atom = []
    force_square_sum = 0.0
    force_component_count = 0
    atom_count = 0
    for record in selected:
        site = record["site_design"][:, feature_index]
        force_target = record["force_target"]
        feature_sum += np.sum(site, axis=0)
        feature_square_sum += np.sum(site * site, axis=0)
        atom_count += int(record["atom_count"])
        energy_per_atom.append(
            float(record["energy_target"]) / int(record["atom_count"])
        )
        force_square_sum += float(np.dot(force_target, force_target))
        force_component_count += int(force_target.size)
    feature_mean = feature_sum / atom_count
    second_moment = feature_square_sum / atom_count
    variance = second_moment - feature_mean * feature_mean
    floor = 64.0 * np.finfo(np.float64).eps * np.maximum(second_moment, 1.0)
    if np.any(variance < -floor):
        raise FloatingPointError("Cached feature variance became negative.")
    feature_scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1.0e-12)
    energy_scale = max(float(np.std(energy_per_atom)), 1.0e-12)
    force_scale = max(
        float(np.sqrt(force_square_sum / force_component_count)), 1.0e-12
    )
    XtX = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    Xty = np.zeros(parameter_count, dtype=np.float64)
    yty = 0.0
    row_count = 0
    structure_count = len(selected)
    for record, row_weight in zip(selected, weights, strict=True):
        if row_weight <= 0.0:
            continue
        atom_count_i = int(record["atom_count"])
        site = record["site_design"][:, feature_index]
        force = record["force_design"][:, feature_index]
        if energy_weight > 0.0:
            species_fractions = record["species_counts"] / atom_count_i
            row = np.concatenate(
                (
                    (np.mean(site, axis=0) - feature_mean) / feature_scale,
                    species_fractions,
                )
            )
            target = float(record["energy_target"]) / atom_count_i
            factor = row_weight * energy_weight / (
                structure_count * energy_scale * energy_scale
            )
            XtX += factor * np.outer(row, row)
            Xty += factor * row * target
            yty += factor * target * target
            row_count += 1
        if force_weight > 0.0:
            target = record["force_target"]
            scaled_force = force / feature_scale
            factor = row_weight * force_weight / (
                structure_count * force.shape[0] * force_scale * force_scale
            )
            XtX[:feature_count, :feature_count] += factor * (
                scaled_force.T @ scaled_force
            )
            Xty[:feature_count] += factor * (scaled_force.T @ target)
            yty += factor * float(np.dot(target, target))
            row_count += int(force.shape[0])
    problem = {
        "problem_family": "linear_lifted_cauchy_cached_parent_subset_gram",
        "artifact_hash": str(cache["artifact_hash"]),
        "source_plan_hash": str(cache["source_plan_hash"]),
        "row_cache_hash": str(cache["cache_hash"]),
        "normal_equation_request_hash": normal_equation_request["request_hash"],
        "realization": str(cache["realization"]),
        "source_realization": str(cache["source_realization"]),
        "type_map": dict(cache["type_map"]),
        "central_species_order": tuple(cache["central_species_order"]),
        "selected_record_indices": record_indices,
        "selected_feature_indices": feature_indices,
        "selected_feature_coordinate_ids": selected_feature_coordinate_ids,
        "selected_descriptor_coordinate_ids": (
            None
            if descriptor_coordinate_ids is None
            else descriptor_coordinate_ids
        ),
        "runtime_from_fit_coordinates": np.eye(feature_count, dtype=np.float64),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "fit_coordinate_metric": np.eye(feature_count, dtype=np.float64),
        "penalized_parameter_count": feature_count,
        "XtX": XtX,
        "Xty": Xty,
        "yty": float(yty),
        "metadata": {
            "backend": "lifted_cauchy_identity_bound_parent_row_cache",
            "objective": "structure_balanced_train_scaled_E1_F1",
            "energy_weight": energy_weight,
            "force_weight": force_weight,
            "energy_target_scale": energy_scale,
            "force_target_scale": force_scale,
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "parameter_count": parameter_count,
            "penalized_parameter_count": feature_count,
            "row_count": row_count,
            "structure_count": structure_count,
            "atom_count": atom_count,
            "force_component_count": force_component_count,
            "dataset_passes": 0,
            "parent_row_cache_hash": str(cache["cache_hash"]),
            "parent_retained_bytes": int(cache["retained_bytes"]),
            "selected_feature_count": feature_count,
            "selected_feature_indices": feature_indices,
            "selected_feature_coordinate_ids": selected_feature_coordinate_ids,
            "selected_record_indices": record_indices,
            "structure_weight_policy": (
                "uniform" if structure_weights is None else "explicit"
            ),
            "structure_weight_hash": _lifted_cauchy_array_hash(weights),
        },
    }
    identity = {
        "schema": "ye3t_lifted_cauchy_cached_problem_identity_v1",
        "normal_equation_request_hash": normal_equation_request["request_hash"],
        "row_cache_hash": str(cache["cache_hash"]),
        "selected_record_indices": record_indices,
        "selected_feature_coordinate_ids": selected_feature_coordinate_ids,
        "energy_weight": energy_weight,
        "force_weight": force_weight,
        "structure_weight_hash": _lifted_cauchy_array_hash(weights),
        "feature_mean_hash": _lifted_cauchy_array_hash(feature_mean),
        "feature_scale_hash": _lifted_cauchy_array_hash(feature_scale),
        "normal_equation_hash": _lifted_cauchy_array_hash(XtX, Xty),
        "fit_coordinate_metric_hash": _lifted_cauchy_array_hash(
            problem["fit_coordinate_metric"]
        ),
        "yty": float(yty),
    }
    problem["problem_hash"] = _canonical_source_hash(identity)
    problem["metadata"]["problem_hash"] = problem["problem_hash"]
    return problem


def subset_lifted_cauchy_normal_equations_from_parent(
    cache,
    parent_problem,
    *,
    descriptor_coordinate_ids,
):
    """Slice an identity-bound full-parent Gram problem by descriptor IDs."""

    if str(parent_problem.get("row_cache_hash", "")) != str(
        cache.get("cache_hash", "")
    ):
        raise ValueError("Parent problem and row cache identities differ.")
    coefficient_count = int(cache["coefficient_count"])
    if tuple(parent_problem.get("selected_feature_indices", ())) != tuple(
        range(coefficient_count)
    ):
        raise ValueError("Normal-equation slicing requires the complete parent problem.")
    descriptor_coordinate_ids = tuple(
        str(value) for value in descriptor_coordinate_ids
    )
    if not descriptor_coordinate_ids or len(descriptor_coordinate_ids) != len(
        set(descriptor_coordinate_ids)
    ):
        raise ValueError("descriptor_coordinate_ids must be nonempty and unique.")
    descriptor_lookup = {
        value: index
        for index, value in enumerate(cache["descriptor_coordinate_ids"])
    }
    missing = sorted(set(descriptor_coordinate_ids) - set(descriptor_lookup))
    if missing:
        raise KeyError("Unknown descriptor coordinate IDs: " + ", ".join(missing))
    descriptor_count = len(cache["descriptor_coordinate_ids"])
    head_count = int(cache["head_count"])
    selected_feature_indices = tuple(
        head * descriptor_count + descriptor_lookup[value]
        for head in range(head_count)
        for value in descriptor_coordinate_ids
    )
    selected_feature_coordinate_ids = tuple(
        cache["feature_coordinate_ids"][index]
        for index in selected_feature_indices
    )
    parameter_indices = selected_feature_indices + tuple(
        range(coefficient_count, coefficient_count + head_count)
    )
    index = np.asarray(parameter_indices, dtype=int)
    feature_index = np.asarray(selected_feature_indices, dtype=int)
    XtX = np.asarray(parent_problem["XtX"], dtype=np.float64)[np.ix_(index, index)]
    Xty = np.asarray(parent_problem["Xty"], dtype=np.float64)[index]
    feature_mean = np.asarray(
        parent_problem["feature_mean"], dtype=np.float64
    )[feature_index]
    feature_scale = np.asarray(
        parent_problem["feature_scale"], dtype=np.float64
    )[feature_index]
    parent_metric = np.asarray(
        parent_problem.get(
            "fit_coordinate_metric", np.eye(coefficient_count)
        ),
        dtype=np.float64,
    )
    fit_metric = parent_metric[np.ix_(feature_index, feature_index)]
    metadata = dict(parent_problem["metadata"])
    metadata.update(
        {
            "backend": "lifted_cauchy_identity_bound_parent_gram_slice",
            "parameter_count": len(selected_feature_indices) + head_count,
            "penalized_parameter_count": len(selected_feature_indices),
            "selected_feature_count": len(selected_feature_indices),
            "selected_feature_indices": selected_feature_indices,
            "selected_feature_coordinate_ids": selected_feature_coordinate_ids,
            "selected_descriptor_coordinate_ids": descriptor_coordinate_ids,
            "parent_problem_hash": str(parent_problem["problem_hash"]),
        }
    )
    problem = {
        **{
            key: parent_problem[key]
            for key in (
                "artifact_hash",
                "source_plan_hash",
                "row_cache_hash",
                "realization",
                "source_realization",
                "type_map",
                "central_species_order",
                "selected_record_indices",
                "yty",
            )
        },
        "problem_family": "linear_lifted_cauchy_cached_parent_gram_slice",
        "selected_feature_indices": selected_feature_indices,
        "selected_feature_coordinate_ids": selected_feature_coordinate_ids,
        "selected_descriptor_coordinate_ids": descriptor_coordinate_ids,
        "runtime_from_fit_coordinates": np.eye(
            len(selected_feature_indices), dtype=np.float64
        ),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "fit_coordinate_metric": fit_metric,
        "penalized_parameter_count": len(selected_feature_indices),
        "XtX": XtX,
        "Xty": Xty,
        "metadata": metadata,
    }
    identity = {
        "schema": "ye3t_lifted_cauchy_cached_problem_identity_v1",
        "row_cache_hash": str(cache["cache_hash"]),
        "selected_record_indices": tuple(parent_problem["selected_record_indices"]),
        "selected_feature_coordinate_ids": selected_feature_coordinate_ids,
        "energy_weight": float(metadata["energy_weight"]),
        "force_weight": float(metadata["force_weight"]),
        "structure_weight_hash": str(metadata["structure_weight_hash"]),
        "feature_mean_hash": _lifted_cauchy_array_hash(feature_mean),
        "feature_scale_hash": _lifted_cauchy_array_hash(feature_scale),
        "normal_equation_hash": _lifted_cauchy_array_hash(XtX, Xty),
        "fit_coordinate_metric_hash": _lifted_cauchy_array_hash(fit_metric),
        "yty": float(parent_problem["yty"]),
    }
    problem["problem_hash"] = _canonical_source_hash(identity)
    problem["metadata"]["problem_hash"] = problem["problem_hash"]
    return problem


def score_lifted_cauchy_cached_solution(
    cache,
    problem,
    solved,
    *,
    record_indices=None,
):
    """Score a cached child solution without reevaluating descriptors."""

    if str(problem.get("row_cache_hash", "")) != str(cache.get("cache_hash", "")):
        raise ValueError("Cached solution and row cache identities differ.")
    if str(solved.get("problem_hash", "")) != str(problem.get("problem_hash", "")):
        raise ValueError("Cached solution and regression problem identities differ.")
    indices = np.asarray(problem["selected_feature_indices"], dtype=int)
    if record_indices is None:
        record_indices = tuple(problem["selected_record_indices"])
    else:
        record_indices = tuple(int(value) for value in record_indices)
    if not record_indices or len(record_indices) != len(set(record_indices)):
        raise ValueError("Scoring record_indices must be nonempty and unique.")
    if any(index < 0 or index >= len(cache["records"]) for index in record_indices):
        raise IndexError("Scoring record_indices contains an out-of-range value.")
    coefficients = np.asarray(
        solved["fit_coordinate_coefficients"], dtype=np.float64
    )
    feature_count = len(indices)
    head_count = int(cache["head_count"])
    if coefficients.shape != (feature_count + head_count,):
        raise ValueError("Cached solution coefficient shape is inconsistent.")
    weights = coefficients[:feature_count]
    offsets = coefficients[feature_count:]
    energy_errors = []
    force_errors = []
    for record_index in record_indices:
        record = cache["records"][record_index]
        energy = float(
            np.sum(record["site_design"][:, indices] @ weights)
            + np.dot(record["species_counts"], offsets)
        )
        force = record["force_design"][:, indices] @ weights
        energy_errors.append(
            (energy - float(record["energy_target"])) / int(record["atom_count"])
        )
        force_errors.append(force - record["force_target"])
    energy_errors = np.asarray(energy_errors, dtype=np.float64)
    force_errors = np.concatenate(tuple(force_errors))
    energy_rmse = float(np.sqrt(np.mean(energy_errors * energy_errors)))
    force_rmse = float(np.sqrt(np.mean(force_errors * force_errors)))
    energy_scale = max(float(problem["metadata"]["energy_target_scale"]), 1.0e-12)
    force_scale = max(float(problem["metadata"]["force_target_scale"]), 1.0e-12)
    return {
        "energy_rmse_eV_per_atom": energy_rmse,
        "force_rmse_eV_per_A": force_rmse,
        "normalized_E1_F1_score": math.sqrt(
            (energy_rmse / energy_scale) ** 2
            + (force_rmse / force_scale) ** 2
        ),
        "structure_count": len(record_indices),
        "force_component_count": int(force_errors.size),
        "finite": bool(
            np.all(np.isfinite(energy_errors))
            and np.all(np.isfinite(force_errors))
        ),
    }


def _lifted_cauchy_cached_runtime_parameters(cache, problem, solved):
    """Lower one selected fit into full-parent runtime parameters."""

    if str(problem.get("row_cache_hash", "")) != str(cache.get("cache_hash", "")):
        raise ValueError("Cached problem and parent row cache identities differ.")
    if str(solved.get("problem_hash", "")) != str(problem.get("problem_hash", "")):
        raise ValueError("Cached solution and regression problem identities differ.")
    selected = np.asarray(problem["selected_feature_indices"], dtype=int)
    fit = np.asarray(solved["fit_coordinate_coefficients"], dtype=np.float64)
    coefficient_count = int(cache["coefficient_count"])
    head_count = int(cache["head_count"])
    if fit.shape != (len(selected) + head_count,):
        raise ValueError("Cached solution coefficient shape is inconsistent.")
    parent_fit = np.zeros(coefficient_count, dtype=np.float64)
    parent_fit[selected] = fit[: len(selected)]
    lowering = np.asarray(
        cache["runtime_from_fit_coordinates"], dtype=np.float64
    )
    if lowering.shape != (coefficient_count, coefficient_count):
        raise ValueError("Cached parent coordinate lowering has the wrong shape.")
    runtime = lowering @ parent_fit
    return (
        runtime.reshape(
            head_count, int(cache["feature_count_per_species"])
        ),
        fit[len(selected):],
    )


def lifted_cauchy_model_from_cached_solution(descriptor, cache, problem, solved):
    """Embed a cached child fit back into its exact parent runtime coordinates."""

    template = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "realization": cache["realization"],
            "source_realization": cache["source_realization"],
        },
    )
    if str(template.artifact_hash) != str(cache.get("artifact_hash", "")):
        raise ValueError("Descriptor and cached parent artifact identities differ.")
    if str(template.source.source_plan_hash) != str(
        cache.get("source_plan_hash", "")
    ):
        raise ValueError("Descriptor and cached source-plan identities differ.")
    selected = np.asarray(problem["selected_feature_indices"], dtype=int)
    runtime, offsets = _lifted_cauchy_cached_runtime_parameters(
        cache, problem, solved
    )
    fitted = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": runtime,
            "offsets": offsets,
            "realization": cache["realization"],
            "source_realization": cache["source_realization"],
        },
    )
    fitted.fit_metadata = {
        "model_family": "linear_lifted_cauchy_scalar_composite_parent_subset",
        "artifact_hash": str(template.artifact_hash),
        "source_plan_hash": str(template.source.source_plan_hash),
        "row_cache_hash": str(cache["cache_hash"]),
        "selected_feature_indices": tuple(int(value) for value in selected),
        "problem": dict(problem["metadata"]),
        "solve": dict(solved["metadata"]),
    }
    fitted._ye3t_linear_fit_metadata = dict(fitted.fit_metadata)
    return fitted


def solve_lifted_cauchy_normal_equations(
    problem,
    ridge_alpha=0.0,
    svd_rcond=1.0e-12,
    maximum_condition=1.0e24,
):
    """Solve one FP64 streamed Gram problem and lower to runtime coordinates."""

    ridge_alpha = float(ridge_alpha)
    svd_rcond = float(svd_rcond)
    maximum_condition = float(maximum_condition)
    if ridge_alpha < 0.0 or not math.isfinite(ridge_alpha):
        raise ValueError("ridge_alpha must be finite and nonnegative.")
    if not 0.0 < svd_rcond < 1.0:
        raise ValueError("svd_rcond must satisfy 0 < svd_rcond < 1.")
    if maximum_condition <= 1.0 or not math.isfinite(maximum_condition):
        raise ValueError("maximum_condition must be finite and greater than one.")
    XtX = np.asarray(problem["XtX"], dtype=np.float64)
    Xty = np.asarray(problem["Xty"], dtype=np.float64)
    penalized = int(problem["penalized_parameter_count"])
    if XtX.ndim != 2 or XtX.shape[0] != XtX.shape[1] or Xty.shape != (
        XtX.shape[0],
    ):
        raise ValueError("Streamed normal-equation shapes are inconsistent.")
    symmetry_defect = float(np.max(np.abs(XtX - XtX.T)))
    symmetry_limit = 256.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(XtX))), 1.0
    )
    if symmetry_defect > symmetry_limit:
        raise FloatingPointError("Streamed Gram matrix is not symmetric.")
    design_gram = 0.5 * (XtX + XtX.T)
    design_eigenvalues = np.linalg.eigvalsh(design_gram)
    negative_limit = 1024.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(design_eigenvalues))), 1.0
    )
    if float(design_eigenvalues[0]) < -negative_limit:
        raise FloatingPointError("Streamed Gram matrix is materially indefinite.")
    design_largest = max(float(design_eigenvalues[-1]), 0.0)
    design_cutoff = design_largest * max(
        svd_rcond * svd_rcond, np.finfo(np.float64).eps
    )
    design_keep = design_eigenvalues > design_cutoff
    design_numerical_rank = int(np.count_nonzero(design_keep))
    design_retained_condition = (
        float(design_eigenvalues[-1] / design_eigenvalues[design_keep][0])
        if design_numerical_rank
        else math.inf
    )
    singular_values = np.sqrt(np.maximum(design_eigenvalues, 0.0))[::-1]
    system = design_gram.copy()
    feature_scale = np.asarray(problem["feature_scale"], dtype=np.float64)
    if feature_scale.shape != (penalized,) or np.any(~np.isfinite(feature_scale)):
        raise ValueError("Streamed feature scales are invalid.")
    if np.any(feature_scale <= 0.0):
        raise ValueError("Streamed feature scales must be positive.")
    fit_metric = np.asarray(
        problem.get("fit_coordinate_metric", np.eye(penalized)),
        dtype=np.float64,
    )
    if fit_metric.shape != (penalized, penalized) or np.any(~np.isfinite(fit_metric)):
        raise ValueError("Fit-coordinate ridge metric is invalid.")
    metric_symmetry_defect = float(np.max(np.abs(fit_metric - fit_metric.T)))
    metric_scale = max(float(np.max(np.abs(fit_metric))), 1.0)
    metric_symmetry_limit = 256.0 * np.finfo(np.float64).eps * metric_scale
    if metric_symmetry_defect > metric_symmetry_limit:
        raise ValueError("Fit-coordinate ridge metric is not symmetric.")
    fit_metric = 0.5 * (fit_metric + fit_metric.T)
    metric_eigenvalues = np.linalg.eigvalsh(fit_metric)
    metric_positive_limit = 1024.0 * np.finfo(np.float64).eps * metric_scale
    if float(metric_eigenvalues[0]) <= metric_positive_limit:
        raise ValueError("Fit-coordinate ridge metric must be positive definite.")
    inverse_scale = 1.0 / feature_scale
    scaled_ridge_metric = (
        inverse_scale[:, None] * fit_metric * inverse_scale[None, :]
    )
    if ridge_alpha > 0.0 and penalized:
        system[:penalized, :penalized] += ridge_alpha * scaled_ridge_metric
    eigenvalues, eigenvectors = np.linalg.eigh(system)
    largest = max(float(eigenvalues[-1]), 0.0)
    cutoff = largest * max(svd_rcond * svd_rcond, np.finfo(np.float64).eps)
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise np.linalg.LinAlgError("Streamed Gram system has no retained direction.")
    retained_condition = float(eigenvalues[-1] / eigenvalues[keep][0])
    if retained_condition > maximum_condition:
        raise np.linalg.LinAlgError(
            "Streamed Gram system exceeds the configured FP64 condition limit."
        )
    scaled_coefficients = eigenvectors[:, keep] @ (
        (eigenvectors[:, keep].T @ Xty) / eigenvalues[keep]
    )
    residual_vector = system @ scaled_coefficients - Xty
    denominator = (
        float(np.linalg.norm(system, ord=2))
        * float(np.linalg.norm(scaled_coefficients))
        + float(np.linalg.norm(Xty))
    )
    backward_error = float(np.linalg.norm(residual_vector)) / max(
        denominator, np.finfo(np.float64).tiny
    )
    backward_limit = max(
        100.0 * XtX.shape[0] * np.finfo(np.float64).eps,
        10.0 * svd_rcond,
    )
    if backward_error > backward_limit:
        raise np.linalg.LinAlgError(
            "Streamed Gram solve failed its FP64 backward-error gate."
        )
    feature_mean = np.asarray(problem["feature_mean"], dtype=np.float64)
    fit_coefficients = np.empty_like(scaled_coefficients)
    fit_coefficients[:penalized] = (
        scaled_coefficients[:penalized] / feature_scale
    )
    centering_shift = float(
        np.dot(feature_mean, fit_coefficients[:penalized])
    )
    fit_coefficients[penalized:] = (
        scaled_coefficients[penalized:] - centering_shift
    )
    coefficients = fit_coefficients.copy()
    lowering = np.asarray(
        problem.get("runtime_from_fit_coordinates", np.eye(penalized)),
        dtype=np.float64,
    )
    if lowering.shape != (penalized, penalized):
        raise ValueError("Regression fit-coordinate lowering has the wrong shape.")
    coefficients[:penalized] = lowering @ fit_coefficients[:penalized]
    data_residual_squared = float(
        problem["yty"]
        - 2.0 * np.dot(scaled_coefficients, Xty)
        + scaled_coefficients @ design_gram @ scaled_coefficients
    )
    data_residual_squared = max(data_residual_squared, 0.0)
    ridge_penalty_squared = float(
        scaled_coefficients[:penalized]
        @ scaled_ridge_metric
        @ scaled_coefficients[:penalized]
    )
    ridge_penalty_squared = max(ridge_penalty_squared, 0.0)
    metric_payload = {
        "schema": "ye3t_fit_coordinate_ridge_metric_v1",
        "matrix": fit_metric.tolist(),
    }
    return {
        "coefficients": coefficients,
        "fit_coordinate_coefficients": fit_coefficients,
        "scaled_coefficients": scaled_coefficients,
        "problem_hash": problem.get("problem_hash"),
        "metadata": {
            "solver": "eigh_streamed_gram",
            "ridge_alpha": ridge_alpha,
            "svd_rcond": svd_rcond,
            "numerical_rank": design_numerical_rank,
            "design_numerical_rank": design_numerical_rank,
            "design_eigenvalue_cutoff": design_cutoff,
            "design_retained_condition_number": design_retained_condition,
            "singular_values": singular_values.tolist(),
            "regularized_system_rank": int(np.count_nonzero(keep)),
            "regularized_system_eigenvalue_cutoff": cutoff,
            "retained_system_condition_number": retained_condition,
            "maximum_condition": maximum_condition,
            "normal_system_backward_error": backward_error,
            "normal_system_backward_error_limit": backward_limit,
            "gram_symmetry_defect": symmetry_defect,
            "fit_coordinate_metric_symmetry_defect": metric_symmetry_defect,
            "fit_coordinate_metric_smallest_eigenvalue": float(
                metric_eigenvalues[0]
            ),
            "fit_coordinate_metric_sha256": _canonical_source_hash(
                metric_payload
            ),
            "ridge_penalty_coordinate": "unscaled_fit_coordinates",
            "ridge_penalty_norm": math.sqrt(ridge_penalty_squared),
            "weighted_residual_l2": math.sqrt(data_residual_squared),
            "scaled_coefficient_l2": float(np.linalg.norm(scaled_coefficients)),
            "scaled_penalized_coefficient_l2": float(
                np.linalg.norm(scaled_coefficients[:penalized])
            ),
            "physical_coefficient_l2": float(np.linalg.norm(coefficients)),
            "fit_coordinate_coefficient_l2": float(
                np.linalg.norm(fit_coefficients)
            ),
            "physical_penalized_coefficient_l2": float(
                np.linalg.norm(coefficients[:penalized])
            ),
        },
    }


def solve_ordinary_lifted_cauchy_normal_equations(
    descriptor,
    ordinary_descriptor,
    problem,
    *,
    ridge_alpha,
    svd_rcond,
    maximum_condition,
    realization,
):
    """Solve and lower one streamed joint ordinary/lifted Gram problem."""

    source_realization = str(problem.get("source_realization", "auto"))
    template = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "realization": realization,
            "source_realization": source_realization,
        },
    )
    if problem.get("problem_family") != (
        "linear_ordinary_plus_lifted_cauchy_scalar_streamed_gram"
    ):
        raise ValueError("Expected a streamed ordinary-plus-lifted Gram problem.")
    if str(problem.get("artifact_hash", "")) != str(template.artifact_hash):
        raise ValueError("Regression problem compiler artifact does not match.")
    if str(problem.get("source_plan_hash", "")) != str(
        template.source.source_plan_hash
    ):
        raise ValueError("Regression problem source plan does not match.")
    if str(problem.get("realization", "")) != str(realization):
        raise ValueError("Regression problem realization does not match.")
    if dict(problem.get("type_map", {})) != dict(template.type_map):
        raise ValueError("Regression problem type map does not match.")
    if tuple(problem.get("central_species_order", ())) != tuple(
        template.central_species_order
    ):
        raise ValueError("Regression problem central species order does not match.")
    solved = solve_lifted_cauchy_normal_equations(
        problem,
        ridge_alpha=ridge_alpha,
        svd_rcond=svd_rcond,
        maximum_condition=maximum_condition,
    )
    flat = np.asarray(solved["coefficients"], dtype=np.float64)
    fit_flat = np.asarray(
        solved["fit_coordinate_coefficients"], dtype=np.float64
    )
    ordinary_count = int(problem["ordinary_feature_count"])
    lifted_count = int(problem["lifted_feature_count"])
    if flat.shape != (ordinary_count + lifted_count + 1,):
        raise ValueError("Solved joint coefficient vector has the wrong shape.")
    ordinary_weight = flat[:ordinary_count]
    lifted_weight = flat[ordinary_count : ordinary_count + lifted_count]
    lifted_fit_weight = fit_flat[ordinary_count : ordinary_count + lifted_count]
    total_bias = float(flat[-1])

    from ye3t_ace.ace.linear_ace import LinearACEScalarModelBundle

    ordinary_bundle = LinearACEScalarModelBundle(
        settings=ordinary_descriptor.settings,
        site_basis_config=ordinary_descriptor.site_basis_config,
        descriptor_specs=ordinary_descriptor.descriptor_specs,
        weight=ordinary_weight,
        bias=total_bias,
        basis_mode=ordinary_descriptor.representation.basis_mode,
        fit_method="ridge_streaming_gram_joint_lifted_cauchy",
        fit_metadata={
            "fit_objective": {
                "kind": "structure_balanced_train_scaled_E1_F1",
            },
            "streamed_problem": dict(problem["metadata"]),
            "factorized_descriptor_runtime_policy": (
                ordinary_descriptor.metadata.get(
                    "factorized_descriptor_runtime_policy", None
                )
            ),
        },
    )
    lifted_model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": lifted_weight.reshape(1, lifted_count),
            "offsets": (0.0,),
            "realization": realization,
            "source_realization": source_realization,
        },
    )
    metadata = {
        "model_family": "linear_ordinary_plus_lifted_cauchy_scalar",
        "descriptor_first_flow": (
            "YE3TRepresentation -> YE3TDescriptors -> YE3TModel.linear"
        ),
        "fit_method": "ridge_streaming_gram",
        "ridge_alpha": float(ridge_alpha),
        "svd_rcond": float(svd_rcond),
        "ordinary_feature_count": ordinary_count,
        "lifted_feature_count": lifted_count,
        "total_feature_count": ordinary_count + lifted_count,
        "ordinary_catalogue": ordinary_descriptor.metadata.get(
            "ordinary_scalar_catalogue", None
        ),
        "problem": dict(problem["metadata"]),
        "solve": dict(solved["metadata"]),
        "artifact_hash": str(template.artifact_hash),
        "source_plan_hash": template.source.source_plan_hash,
        "realization": str(realization),
        "source_realization": str(lifted_model.source.source_realization),
        "ordinary_coefficient_l2": float(np.linalg.norm(ordinary_weight)),
        "lifted_coefficient_l2": float(np.linalg.norm(lifted_weight)),
        "lifted_fit_coordinate_coefficient_l2": float(
            np.linalg.norm(lifted_fit_weight)
        ),
        "physical_coefficient_l2": float(np.linalg.norm(flat)),
        "shared_physical_bias": total_bias,
        "force_sign": "F=-dE/dx",
        "strain_derivative": "dE/depsilon; LAMMPS virial is its negative",
    }
    fitted = _OrdinaryLiftedCauchyLinearModel(
        ordinary_bundle,
        ordinary_descriptor,
        lifted_model,
        fit_metadata=metadata,
    )
    fitted._ye3t_fit_runtime_metrics = dict(problem.get("runtime_metrics", {}))
    return fitted


def _solve_augmented_svd(design, target, penalized, ridge_alpha, svd_rcond):
    ridge_alpha = float(ridge_alpha)
    svd_rcond = float(svd_rcond)
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be nonnegative.")
    if not 0.0 <= svd_rcond < 1.0:
        raise ValueError("svd_rcond must satisfy 0 <= svd_rcond < 1.")
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    penalized = int(penalized)
    if ridge_alpha > 0.0 and penalized:
        penalty = np.zeros((penalized, design.shape[1]), dtype=float)
        penalty[:, :penalized] = np.sqrt(ridge_alpha) * np.eye(penalized)
        augmented_design = np.concatenate((design, penalty), axis=0)
        augmented_target = np.concatenate((target, np.zeros(penalized)))
    else:
        augmented_design = design
        augmented_target = target
    started = time.perf_counter()
    left, singular_values, right_t = np.linalg.svd(
        augmented_design, full_matrices=False
    )
    cutoff = (
        svd_rcond * float(singular_values[0]) if singular_values.size else 0.0
    )
    inverse = np.zeros_like(singular_values)
    keep = singular_values > cutoff
    inverse[keep] = 1.0 / singular_values[keep]
    scaled_coefficients = right_t.T @ (
        inverse * (left.T @ augmented_target)
    )
    residual = design @ scaled_coefficients - target
    retained_singular_values = singular_values[keep]
    full_rank = bool(
        singular_values.size
        and retained_singular_values.size == min(augmented_design.shape)
    )
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if full_rank and singular_values[-1] > 0.0
        else None
    )
    retained_condition_number = (
        float(retained_singular_values[0] / retained_singular_values[-1])
        if retained_singular_values.size
        else None
    )
    return {
        "scaled_coefficients": scaled_coefficients,
        "metadata": {
            "solver": "augmented_svd",
            "ridge_alpha": ridge_alpha,
            "svd_rcond": svd_rcond,
            "numerical_rank": int(np.count_nonzero(keep)),
            "augmented_singular_values": singular_values.tolist(),
            "condition_number": condition_number,
            "condition_number_is_infinite": not full_rank,
            "retained_condition_number": retained_condition_number,
            "largest_augmented_singular_value": float(singular_values[0]) if singular_values.size else 0.0,
            "smallest_retained_augmented_singular_value": float(
                singular_values[keep][-1]
            ) if np.any(keep) else 0.0,
            "weighted_residual_l2": float(np.linalg.norm(residual)),
            "solve_seconds": float(time.perf_counter() - started),
        },
    }


def solve_lifted_cauchy_ridge(problem, ridge_alpha=0.0, svd_rcond=1.0e-12):
    """Solve one augmented-SVD ridge problem with unpenalized offsets."""

    solved = _solve_augmented_svd(
        problem["scaled_design"],
        problem["weighted_target"],
        problem["penalized_parameter_count"],
        ridge_alpha,
        svd_rcond,
    )
    penalized = int(problem["penalized_parameter_count"])
    feature_scale = np.asarray(problem["feature_scale"], dtype=float)
    feature_mean = np.asarray(problem["feature_mean"], dtype=float)
    scaled_coefficients = np.asarray(solved["scaled_coefficients"], dtype=float)
    fit_coefficients = np.empty_like(scaled_coefficients)
    fit_coefficients[:penalized] = (
        scaled_coefficients[:penalized] / feature_scale
    )
    centering_shift = float(np.dot(feature_mean, fit_coefficients[:penalized]))
    fit_coefficients[penalized:] = (
        scaled_coefficients[penalized:] - centering_shift
    )
    coefficients = fit_coefficients.copy()
    lowering = np.asarray(
        problem.get("runtime_from_fit_coordinates", np.eye(penalized)),
        dtype=float,
    )
    if lowering.shape != (penalized, penalized):
        raise ValueError("Regression fit-coordinate lowering has the wrong shape.")
    coefficients[:penalized] = lowering @ fit_coefficients[:penalized]
    metadata = dict(solved["metadata"])
    metadata.update(
        {
            "scaled_coefficient_l2": float(np.linalg.norm(scaled_coefficients)),
            "scaled_penalized_coefficient_l2": float(
                np.linalg.norm(scaled_coefficients[:penalized])
            ),
            "physical_coefficient_l2": float(np.linalg.norm(coefficients)),
            "fit_coordinate_coefficient_l2": float(
                np.linalg.norm(fit_coefficients)
            ),
            "physical_penalized_coefficient_l2": float(
                np.linalg.norm(coefficients[:penalized])
            ),
        }
    )
    return {
        "coefficients": coefficients,
        "fit_coordinate_coefficients": fit_coefficients,
        "scaled_coefficients": scaled_coefficients,
        "metadata": metadata,
    }


def build_ordinary_lifted_cauchy_regression_problem(
    lifted_model,
    ordinary_descriptor,
    structures,
    *,
    energy_key,
    force_key,
    energy_weight,
    force_weight,
    feature_chunk_size,
    ordinary_descriptor_matrix_cache,
    fit_coordinate_policy="orthogonal",
):
    if len(lifted_model.type_order) != 1:
        raise NotImplementedError(
            "The first joint ordinary/lifted fit supports one central species; "
            "the standalone lifted model already supports multiple species."
        )
    if dict(ordinary_descriptor.type_map) != dict(lifted_model.type_map):
        raise ValueError("Ordinary and lifted descriptors must use the same type_map.")
    if not np.isclose(
        float(ordinary_descriptor.cutoff),
        float(lifted_model.source.cutoff),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError("Ordinary and lifted descriptors must use the same cutoff.")
    if not np.isclose(float(energy_weight), 1.0) or not np.isclose(
        float(force_weight), 1.0
    ):
        raise ValueError(
            "The frozen joint Task 56 objective requires energy_weight=force_weight=1."
        )
    from ye3t_ace.ace.linear_ace import build_linear_ace_regression_problem

    ordinary_design, ordinary_target, ordinary_metadata = (
        build_linear_ace_regression_problem(
            structures,
            settings=ordinary_descriptor.settings,
            descriptors=ordinary_descriptor.descriptor_specs,
            site_basis_config=ordinary_descriptor.site_basis_config,
            cutoff=ordinary_descriptor.cutoff,
            type_map=ordinary_descriptor.type_map,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=1.0,
            force_weight=1.0,
            backend=ordinary_descriptor.backend,
            strict_backend=ordinary_descriptor.strict_backend,
            validate_backend=ordinary_descriptor.validate_backend,
            descriptor_matrix_cache=ordinary_descriptor_matrix_cache,
            use_descriptor_matrix_cache=(
                ordinary_descriptor_matrix_cache is not None
            ),
            return_cache_metadata=True,
            device=ordinary_descriptor.metadata.get("device", None),
            factorized_descriptor_runtime_policy=(
                ordinary_descriptor.metadata.get(
                    "factorized_descriptor_runtime_policy", None
                )
            ),
            force_jacobian_mode="analytic_product_adjoint",
            force_jacobian_chunk_size=feature_chunk_size,
            fit_objective={
                "kind": "structure_balanced_train_scaled_E1_F1",
                "feature_minimum_scale": 1.0e-12,
                "target_minimum_scale": 1.0e-12,
            },
        )
    )
    lifted = build_lifted_cauchy_regression_problem(
        lifted_model,
        structures,
        energy_key=energy_key,
        force_key=force_key,
        energy_weight=1.0,
        force_weight=1.0,
        feature_chunk_size=feature_chunk_size,
        fit_coordinate_policy=fit_coordinate_policy,
    )
    lifted_design = np.asarray(lifted["scaled_design"], dtype=float)
    lifted_target = np.asarray(lifted["weighted_target"], dtype=float)
    ordinary_design = np.asarray(ordinary_design, dtype=float)
    ordinary_target = np.asarray(ordinary_target, dtype=float)
    if ordinary_design.shape[0] != lifted_design.shape[0]:
        raise RuntimeError("Ordinary and lifted regression row counts differ.")
    target_scale = max(
        1.0,
        float(np.max(np.abs(ordinary_target))),
        float(np.max(np.abs(lifted_target))),
    )
    if np.max(np.abs(ordinary_target - lifted_target)) > 5.0e-13 * target_scale:
        raise RuntimeError("Ordinary and lifted target scaling is inconsistent.")
    lifted_feature_count = int(lifted["penalized_parameter_count"])
    lifted_offset = lifted_design[:, lifted_feature_count]
    intercept_scale = max(1.0, float(np.max(np.abs(ordinary_design[:, 0]))))
    if np.max(np.abs(ordinary_design[:, 0] - lifted_offset)) > 5.0e-13 * intercept_scale:
        raise RuntimeError("Ordinary and lifted intercept rows are inconsistent.")
    ordinary_feature_count = int(ordinary_design.shape[1] - 1)
    combined_design = np.column_stack(
        (
            ordinary_design[:, 1:],
            lifted_design[:, :lifted_feature_count],
            ordinary_design[:, 0],
        )
    )
    return {
        "problem_family": "linear_ordinary_plus_lifted_cauchy_scalar",
        "artifact_hash": str(lifted_model.artifact_hash),
        "source_plan_hash": lifted_model.source.source_plan_hash,
        "realization": str(lifted_model.realization),
        "source_realization": str(lifted_model.source.source_realization),
        "type_map": dict(lifted_model.type_map),
        "central_species_order": tuple(lifted_model.central_species_order),
        "scaled_design": combined_design,
        "weighted_target": ordinary_target,
        "penalized_parameter_count": (
            ordinary_feature_count + lifted_feature_count
        ),
        "ordinary_feature_count": ordinary_feature_count,
        "lifted_feature_count": lifted_feature_count,
        "ordinary_metadata": dict(ordinary_metadata),
        "lifted_problem": lifted,
    }


def solve_ordinary_lifted_cauchy_regression_problem(
    descriptor,
    ordinary_descriptor,
    problem,
    *,
    ridge_alpha,
    svd_rcond,
    realization,
):
    """Solve a precomputed joint ordinary/lifted problem for one ridge value.

    The expensive descriptor and force rows are immutable across the ridge
    grid. This function validates their compiler/source identity and changes
    only the augmented linear solve and resulting readout.
    """

    source_realization = str(problem.get("source_realization", "auto"))
    template = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "realization": realization,
            "source_realization": source_realization,
        },
    )
    if problem.get("problem_family") != (
        "linear_ordinary_plus_lifted_cauchy_scalar"
    ):
        raise ValueError("Expected an ordinary-plus-lifted regression problem.")
    if str(problem.get("artifact_hash", "")) != str(template.artifact_hash):
        raise ValueError("Regression problem compiler artifact does not match.")
    if str(problem.get("source_plan_hash", "")) != str(
        template.source.source_plan_hash
    ):
        raise ValueError("Regression problem source plan does not match.")
    if str(problem.get("realization", "")) != str(realization):
        raise ValueError("Regression problem realization does not match.")
    if dict(problem.get("type_map", {})) != dict(template.type_map):
        raise ValueError("Regression problem type map does not match.")
    if tuple(problem.get("central_species_order", ())) != tuple(
        template.central_species_order
    ):
        raise ValueError("Regression problem central species order does not match.")
    solved = _solve_augmented_svd(
        problem["scaled_design"],
        problem["weighted_target"],
        problem["penalized_parameter_count"],
        ridge_alpha,
        svd_rcond,
    )
    scaled = np.asarray(solved["scaled_coefficients"], dtype=float)
    ordinary_count = int(problem["ordinary_feature_count"])
    lifted_count = int(problem["lifted_feature_count"])
    ordinary_metadata = problem["ordinary_metadata"]
    lifted_problem = problem["lifted_problem"]
    ordinary_scale = np.asarray(ordinary_metadata["feature_scale"], dtype=float)
    ordinary_mean = np.asarray(ordinary_metadata["feature_mean"], dtype=float)
    lifted_scale = np.asarray(lifted_problem["feature_scale"], dtype=float)
    lifted_mean = np.asarray(lifted_problem["feature_mean"], dtype=float)
    ordinary_weight = scaled[:ordinary_count] / ordinary_scale
    lifted_fit_weight = (
        scaled[ordinary_count : ordinary_count + lifted_count] / lifted_scale
    )
    lifted_lowering = np.asarray(
        lifted_problem.get(
            "runtime_from_fit_coordinates", np.eye(lifted_count)
        ),
        dtype=float,
    )
    if lifted_lowering.shape != (lifted_count, lifted_count):
        raise ValueError("Joint lifted fit-coordinate lowering has the wrong shape.")
    lifted_weight = lifted_lowering @ lifted_fit_weight
    scaled_intercept = float(scaled[-1])
    total_bias = float(
        scaled_intercept
        - np.dot(ordinary_mean, ordinary_weight)
        - np.dot(lifted_mean, lifted_fit_weight)
    )
    from ye3t_ace.ace.linear_ace import LinearACEScalarModelBundle

    ordinary_bundle = LinearACEScalarModelBundle(
        settings=ordinary_descriptor.settings,
        site_basis_config=ordinary_descriptor.site_basis_config,
        descriptor_specs=ordinary_descriptor.descriptor_specs,
        weight=ordinary_weight,
        bias=total_bias,
        basis_mode=ordinary_descriptor.representation.basis_mode,
        fit_method="ridge_augmented_svd_joint_lifted_cauchy",
        fit_metadata={
            "fit_objective": {
                "kind": "structure_balanced_train_scaled_E1_F1",
            },
            "descriptor_matrix_cache": dict(ordinary_metadata),
            "factorized_descriptor_runtime_policy": (
                ordinary_descriptor.metadata.get(
                    "factorized_descriptor_runtime_policy", None
                )
            ),
        },
    )
    lifted_model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": lifted_weight.reshape(1, lifted_count),
            "offsets": (0.0,),
            "realization": realization,
            "source_realization": source_realization,
        },
    )
    metadata = {
        "model_family": "linear_ordinary_plus_lifted_cauchy_scalar",
        "descriptor_first_flow": (
            "YE3TRepresentation -> YE3TDescriptors -> YE3TModel.linear"
        ),
        "ridge_alpha": float(ridge_alpha),
        "svd_rcond": float(svd_rcond),
        "ordinary_feature_count": ordinary_count,
        "lifted_feature_count": lifted_count,
        "total_feature_count": ordinary_count + lifted_count,
        "ordinary_catalogue": ordinary_descriptor.metadata.get(
            "ordinary_scalar_catalogue", None
        ),
        "ordinary_problem": dict(ordinary_metadata),
        "lifted_problem": dict(lifted_problem["metadata"]),
        "solve": dict(solved["metadata"]),
        "artifact_hash": str(template.artifact_hash),
        "source_plan_hash": template.source.source_plan_hash,
        "realization": str(realization),
        "source_realization": str(lifted_model.source.source_realization),
        "ordinary_coefficient_l2": float(np.linalg.norm(ordinary_weight)),
        "lifted_coefficient_l2": float(np.linalg.norm(lifted_weight)),
        "lifted_fit_coordinate_coefficient_l2": float(
            np.linalg.norm(lifted_fit_weight)
        ),
        "physical_coefficient_l2": float(
            np.linalg.norm(
                np.concatenate(
                    (ordinary_weight, lifted_weight, np.asarray((total_bias,)))
                )
            )
        ),
        "scaled_coefficient_l2": float(np.linalg.norm(scaled)),
        "scaled_penalized_coefficient_l2": float(
            np.linalg.norm(scaled[:-1])
        ),
        "shared_physical_bias": total_bias,
        "force_sign": "F=-dE/dx",
        "strain_derivative": "dE/depsilon; LAMMPS virial is its negative",
    }
    return _OrdinaryLiftedCauchyLinearModel(
        ordinary_bundle,
        ordinary_descriptor,
        lifted_model,
        fit_metadata=metadata,
    )


def fit_lifted_cauchy_linear_model(
    descriptor,
    structures,
    *,
    energy_key="energy",
    force_key="forces",
    energy_weight=1.0,
    force_weight=1.0,
    ridge_alpha=0.0,
    svd_rcond=1.0e-12,
    feature_chunk_size="auto",
    realization="factored",
    source_realization="auto",
    fit_coordinate_policy="orthogonal",
    fit_method="ridge_streaming_gram",
    normal_equation_maximum_condition=1.0e24,
    structure_weights=None,
    structure_weight_key=None,
    structure_group_key=None,
    structure_group_weights=None,
    structure_group_default_weight=None,
    structure_group_normalize_mean=True,
    boltzmann_temperature_K=None,
    boltzmann_energy_key=None,
    boltzmann_weight_nugget=0.0,
    boltzmann_weight_prefactor=1.0,
    boltzmann_normalize_mean=True,
    min_structure_weight=0.0,
    device="cpu",
    evaluation_dtype="float64",
    accumulation_dtype="float64",
    progress=None,
    ordinary_descriptor=None,
    ordinary_descriptor_matrix_cache=None,
):
    """Fit an exact compiler-owned lifted-Cauchy scalar linear model."""

    fit_method = str(fit_method).strip().lower()
    if fit_method not in {"ridge_streaming_gram", "ridge_augmented_svd"}:
        raise ValueError(
            "fit_method must be ridge_streaming_gram or ridge_augmented_svd."
        )
    template = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "realization": realization,
            "source_realization": source_realization,
        },
    )
    feature_count = int(template.coefficients.numel())
    if ordinary_descriptor is not None:
        feature_count += len(tuple(ordinary_descriptor.descriptor_specs))
    feature_chunk = resolve_lifted_cauchy_feature_chunk_size(
        feature_chunk_size, feature_count
    )
    feature_chunk_size = int(feature_chunk["resolved"])
    if ordinary_descriptor is not None:
        if fit_method == "ridge_streaming_gram":
            if ordinary_descriptor_matrix_cache is not None:
                raise ValueError(
                    "ordinary_descriptor_matrix_cache is a dense-reference option; "
                    "streamed fitting evaluates one structure at a time."
                )
            problem = build_ordinary_lifted_cauchy_normal_equations(
                template,
                ordinary_descriptor,
                structures,
                energy_key=energy_key,
                force_key=force_key,
                energy_weight=energy_weight,
                force_weight=force_weight,
                feature_chunk_size=feature_chunk_size,
                fit_coordinate_policy=fit_coordinate_policy,
                structure_weights=structure_weights,
                structure_weight_key=structure_weight_key,
                structure_group_key=structure_group_key,
                structure_group_weights=structure_group_weights,
                structure_group_default_weight=structure_group_default_weight,
                structure_group_normalize_mean=structure_group_normalize_mean,
                boltzmann_temperature_K=boltzmann_temperature_K,
                boltzmann_energy_key=boltzmann_energy_key,
                boltzmann_weight_nugget=boltzmann_weight_nugget,
                boltzmann_weight_prefactor=boltzmann_weight_prefactor,
                boltzmann_normalize_mean=boltzmann_normalize_mean,
                min_structure_weight=min_structure_weight,
                device=device,
                evaluation_dtype=evaluation_dtype,
                accumulation_dtype=accumulation_dtype,
                progress=progress,
            )
            problem["metadata"]["feature_chunk"] = feature_chunk
            return solve_ordinary_lifted_cauchy_normal_equations(
                descriptor,
                ordinary_descriptor,
                problem,
                ridge_alpha=ridge_alpha,
                svd_rcond=svd_rcond,
                maximum_condition=normal_equation_maximum_condition,
                realization=realization,
            )
        if any(
            value is not None
            for value in (
                structure_weights,
                structure_weight_key,
                structure_group_key,
                structure_group_weights,
                structure_group_default_weight,
                boltzmann_temperature_K,
                boltzmann_energy_key,
            )
        ) or (
            not structure_group_normalize_mean
            or float(boltzmann_weight_nugget) != 0.0
            or float(boltzmann_weight_prefactor) != 1.0
            or not boltzmann_normalize_mean
            or float(min_structure_weight) != 0.0
        ):
            raise ValueError(
                "Custom structure weights require fit_method='ridge_streaming_gram'."
            )
        _validate_lifted_fit_runtime(
            device, evaluation_dtype, accumulation_dtype
        )
        problem = build_ordinary_lifted_cauchy_regression_problem(
            template,
            ordinary_descriptor,
            structures,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=energy_weight,
            force_weight=force_weight,
            feature_chunk_size=feature_chunk_size,
            ordinary_descriptor_matrix_cache=ordinary_descriptor_matrix_cache,
            fit_coordinate_policy=fit_coordinate_policy,
        )
        problem["metadata"]["feature_chunk"] = feature_chunk
        return solve_ordinary_lifted_cauchy_regression_problem(
            descriptor,
            ordinary_descriptor,
            problem,
            ridge_alpha=ridge_alpha,
            svd_rcond=svd_rcond,
            realization=realization,
        )
    if fit_method == "ridge_streaming_gram":
        problem = build_lifted_cauchy_normal_equations(
            template,
            structures,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=energy_weight,
            force_weight=force_weight,
            feature_chunk_size=feature_chunk_size,
            fit_coordinate_policy=fit_coordinate_policy,
            structure_weights=structure_weights,
            structure_weight_key=structure_weight_key,
            structure_group_key=structure_group_key,
            structure_group_weights=structure_group_weights,
            structure_group_default_weight=structure_group_default_weight,
            structure_group_normalize_mean=structure_group_normalize_mean,
            boltzmann_temperature_K=boltzmann_temperature_K,
            boltzmann_energy_key=boltzmann_energy_key,
            boltzmann_weight_nugget=boltzmann_weight_nugget,
            boltzmann_weight_prefactor=boltzmann_weight_prefactor,
            boltzmann_normalize_mean=boltzmann_normalize_mean,
            min_structure_weight=min_structure_weight,
            device=device,
            evaluation_dtype=evaluation_dtype,
            accumulation_dtype=accumulation_dtype,
            progress=progress,
        )
        problem["metadata"]["feature_chunk"] = feature_chunk
        solved = solve_lifted_cauchy_normal_equations(
            problem,
            ridge_alpha=ridge_alpha,
            svd_rcond=svd_rcond,
            maximum_condition=normal_equation_maximum_condition,
        )
    else:
        if any(
            value is not None
            for value in (
                structure_weights,
                structure_weight_key,
                structure_group_key,
                structure_group_weights,
                structure_group_default_weight,
                boltzmann_temperature_K,
                boltzmann_energy_key,
            )
        ) or (
            not structure_group_normalize_mean
            or float(boltzmann_weight_nugget) != 0.0
            or float(boltzmann_weight_prefactor) != 1.0
            or not boltzmann_normalize_mean
            or float(min_structure_weight) != 0.0
        ):
            raise ValueError(
                "Custom structure weights require fit_method='ridge_streaming_gram'."
            )
        _validate_lifted_fit_runtime(
            device, evaluation_dtype, accumulation_dtype
        )
        problem = build_lifted_cauchy_regression_problem(
            template,
            structures,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=energy_weight,
            force_weight=force_weight,
            feature_chunk_size=feature_chunk_size,
            fit_coordinate_policy=fit_coordinate_policy,
        )
        problem["metadata"]["feature_chunk"] = feature_chunk
        solved = solve_lifted_cauchy_ridge(
            problem, ridge_alpha=ridge_alpha, svd_rcond=svd_rcond
        )
    feature_count = template.evaluator.descriptor_count
    head_count = len(template.type_order)
    flat = np.asarray(solved["coefficients"], dtype=float)
    coefficients = flat[: head_count * feature_count].reshape(
        head_count, feature_count
    )
    offsets = flat[head_count * feature_count :]
    fitted = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": coefficients,
            "offsets": offsets,
            "realization": realization,
            "source_realization": source_realization,
        },
    )
    fitted.fit_metadata = {
        "model_family": "linear_lifted_cauchy_scalar",
        "descriptor_first_flow": (
            "YE3TRepresentation.lifted_cauchy_scalar -> "
            "YE3TDescriptors.ye3t_basis -> YE3TModel.linear"
        ),
        "problem": dict(problem["metadata"]),
        "solve": dict(solved["metadata"]),
        "artifact_hash": str(template.artifact_hash),
        "source_plan_hash": template.source.source_plan_hash,
        "realization": str(realization),
        "source_realization": str(fitted.source.source_realization),
        "fit_method": fit_method,
        "force_sign": "F=-dE/dx",
        "strain_derivative": "dE/depsilon; LAMMPS virial is its negative",
    }
    fitted._ye3t_linear_fit_metadata = dict(fitted.fit_metadata)
    fitted._ye3t_fit_runtime_metrics = dict(problem.get("runtime_metrics", {}))
    return fitted


__all__ = [
    "LIFTED_CAUCHY_JOINT_SOURCE_FAMILY",
    "LIFTED_CAUCHY_JOINT_SOURCE_SCHEMA",
    "LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY",
    "LIFTED_CAUCHY_MIXED_L_SOURCE_SCHEMA",
    "LIFTED_CAUCHY_SOURCE_SCHEMA",
    "LIFTED_CAUCHY_SOURCE_FAMILY",
    "LiftedCauchyPolynomialSource",
    "LiftedCauchyTorchEvaluator",
    "build_lifted_cauchy_normal_equations",
    "build_lifted_cauchy_normal_equations_from_geometry_and_targets",
    "build_lifted_cauchy_normal_equations_from_row_cache",
    "build_lifted_cauchy_regression_problem",
    "build_ordinary_lifted_cauchy_normal_equations",
    "build_ordinary_lifted_cauchy_regression_problem",
    "combine_lifted_cauchy_geometry_and_target_caches",
    "fit_lifted_cauchy_linear_model",
    "lifted_cauchy_linear_fit_preflight",
    "lifted_cauchy_geometry_row_cache_request",
    "lifted_cauchy_target_cache_request",
    "lifted_cauchy_normal_equation_cache_request",
    "lifted_cauchy_model_from_cached_solution",
    "lifted_cauchy_model_from_descriptor",
    "materialize_lifted_cauchy_geometry_row_cache",
    "materialize_lifted_cauchy_regression_row_cache",
    "materialize_lifted_cauchy_target_cache",
    "prepare_lifted_cauchy_descriptor_payload",
    "resolve_lifted_cauchy_feature_chunk_size",
    "solve_lifted_cauchy_ridge",
    "solve_lifted_cauchy_normal_equations",
    "score_lifted_cauchy_cached_solution",
    "subset_lifted_cauchy_normal_equations_from_parent",
    "solve_ordinary_lifted_cauchy_normal_equations",
    "solve_ordinary_lifted_cauchy_regression_problem",
]
