"""Portable, hash-bound deployment bundles for linear lifted-Cauchy models."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


LIFTED_CAUCHY_BUNDLE_SCHEMA_V1 = "ye3t_lifted_cauchy_linear_bundle_v1"
LIFTED_CAUCHY_BUNDLE_SCHEMA_V2 = "ye3t_lifted_cauchy_linear_bundle_v2"
LIFTED_CAUCHY_BUNDLE_SCHEMA = "ye3t_lifted_cauchy_linear_bundle_v3"
LIFTED_CAUCHY_COMPOSITE_BUNDLE_SCHEMA = (
    "ye3t_lifted_cauchy_composite_linear_bundle_v1"
)
LIFTED_CAUCHY_MODEL_FILE = "model.ye3t.json"
LIFTED_CAUCHY_COMPILER_FILE = "compiled_lifted_cauchy.json"
LIFTED_CAUCHY_COMPILER_BINDING_FILE = "compiler_binding.ye3t.json"
LIFTED_CAUCHY_NATIVE_FILE = "native_runtime.ye3t.json"
LIFTED_CAUCHY_ORDINARY_FILE = "ordinary.yace"
LIFTED_CAUCHY_MANIFEST_FILE = "YE3T_LIFTED_BUNDLE_MANIFEST.sha256"
LIFTED_CAUCHY_NATIVE_SCHEMA_V1 = "ye3t_lifted_cauchy_native_runtime_v1"
LIFTED_CAUCHY_NATIVE_SCHEMA = "ye3t_lifted_cauchy_native_runtime_v2"
LIFTED_CAUCHY_COMPOSITE_NATIVE_SCHEMA = (
    "ye3t_lifted_cauchy_composite_native_runtime_v1"
)
LIFTED_CAUCHY_COMPILER_BINDING_SCHEMA = (
    "ye3t_lifted_cauchy_composite_compiler_binding_v1"
)

_MAX_COMPOSITE_PHYSICAL_TERMS = 5_000_000
_MAX_COMPOSITE_STATIC_BYTES = 128 * 1024 * 1024
_MAX_PORTABLE_JSON_BYTES = 96 * 1024 * 1024


_CONVENTIONS = {
    "precision": "binary64",
    "real_basis_id": "ye3t_physical_real_tesseral_v1",
    "l1_component_order": ["x", "z", "minus_y"],
    "real_pullback": "algebraic_transpose",
    "edge_displacement": "R_neighbor_minus_R_center_plus_periodic_image",
    "cutoff_support": "0_lt_r_lt_rc",
    "self_interaction": "excluded",
    "force_sign": "F_equals_minus_dE_dR",
    "strain_derivative": "dE_d_epsilon",
    "lammps_virial": "minus_strain_derivative",
    "lammps_virial_component_order": ["xx", "yy", "zz", "xy", "xz", "yz"],
}


def _canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _payload_hash(payload):
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _jsonable(value):
    """Convert fit metadata to deterministic finite JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Lifted-Cauchy bundle metadata must be finite.")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    raise TypeError(
        "Lifted-Cauchy bundle metadata contains an unsupported value of type "
        f"{type(value).__name__}."
    )


def _write_json(path, payload):
    Path(path).write_bytes(_pretty_json_bytes(payload))


def _pretty_json_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _finite_matrix(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}; expected {tuple(shape)}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite binary64 values.")
    return array


def _binary_complex(value):
    real, imaginary = value["binary64"]
    return complex(float(real), float(imaginary))


def _physical_real_descriptor_rows(compiled):
    payload = compiled.payload
    validation = dict(compiled.validation_report)
    reality = dict(payload["physical_scalar_reality_report"])
    equivalence = dict(payload["equivalence_certificate"])
    if (
        validation.get("passed") is not True
        or validation.get("real_form_maps_hash_bound") is not True
        or validation.get("physical_force_contract")
        != "bound_real_form_matrix_transpose"
        or reality.get("exactly_real") is not True
        or reality.get("proof") != "exact_canonical_polynomial_realification"
    ):
        raise ValueError("Compiler artifact lacks an exact physical-reality certificate.")
    reality_hash = str(equivalence.get("physical_scalar_reality_report_hash", ""))
    if len(reality_hash) != 64:
        raise ValueError("Compiler physical-reality report hash is missing.")
    channels = tuple(payload["channels"])
    role_dimension = int(payload["role_dimension"])
    channel_positions = {
        int(channel["channel_index"]): position
        for position, channel in enumerate(channels)
    }
    forms = {
        str(record["real_form_id"]): record
        for record in payload["real_forms"]
    }
    bindings = {
        int(record["channel_index"]): str(record["real_form_id"])
        for record in payload["channel_real_form_ids"]
    }
    channel_offsets = {}
    source_variable_count = 0
    for position, channel in enumerate(channels):
        channel_index = int(channel["channel_index"])
        component_count = len(
            forms[bindings[channel_index]]["real_coordinate_order"]
        )
        if component_count != 2 * int(channel["l"]) + 1:
            raise ValueError("Compiler real-form width disagrees with channel l.")
        channel_offsets[channel_index] = source_variable_count
        source_variable_count += role_dimension * component_count

    rows = []
    maximum_imaginary = 0.0
    maximum_scale = 1.0
    expanded_term_count = 0
    for descriptor in payload["descriptors"]:
        physical = {}
        for record in descriptor["canonical_terms"]:
            terms = {(): _binary_complex(record["coefficient"])}
            for coordinate in record["coordinates"]:
                channel, role, magnetic = (int(value) for value in coordinate)
                if channel not in channel_positions or not 0 <= role < role_dimension:
                    raise ValueError("Compiled lifted coordinate is out of range.")
                matrix_row = forms[bindings[channel]]["real_to_complex_matrix"][
                    magnetic
                ]
                factors = []
                for real_component, coefficient in enumerate(matrix_row):
                    value = _binary_complex(coefficient)
                    if value == 0.0:
                        continue
                    component_count = len(matrix_row)
                    source_index = (
                        channel_offsets[channel]
                        + role * component_count
                        + real_component
                    )
                    factors.append((source_index, value))
                next_terms = {}
                for coordinates, coefficient in terms.items():
                    for source_index, factor in factors:
                        key = tuple(sorted(coordinates + (source_index,)))
                        next_terms[key] = next_terms.get(key, 0.0j) + coefficient * factor
                terms = next_terms
            for coordinates, coefficient in terms.items():
                physical[coordinates] = physical.get(coordinates, 0.0j) + coefficient
        real_row = {}
        for coordinates, coefficient in physical.items():
            if coefficient == 0.0:
                continue
            maximum_imaginary = max(maximum_imaginary, abs(coefficient.imag))
            maximum_scale = max(maximum_scale, abs(coefficient))
            if coefficient.real != 0.0:
                real_row[coordinates] = float(coefficient.real)
        if maximum_imaginary > 5.0e-11 * maximum_scale:
            raise ValueError(
                "Native lifted-Cauchy physical-real lowering has a material imaginary residual."
            )
        expanded_term_count += len(real_row)
        if expanded_term_count > 5_000_000:
            raise MemoryError(
                "Native lifted-Cauchy direct lowering exceeds five million physical-real terms; "
                "use a compiled factored runtime plan instead."
            )
        rows.append(real_row)
    return tuple(rows), {
        "descriptor_count": len(rows),
        "physical_real_term_count_before_readout": int(expanded_term_count),
        "maximum_absolute_imaginary_residual": float(maximum_imaginary),
        "maximum_coefficient_scale": float(maximum_scale),
        "source_variable_count": int(source_variable_count),
        "channel_source_variable_offsets": [
            int(channel_offsets[int(channel["channel_index"])])
            for channel in channels
        ],
        "physical_scalar_reality_report_hash": reality_hash,
    }


def _physical_rows_hash(rows):
    row_hashes = []
    for row in rows:
        terms = tuple(
            {
                "source_indices": tuple(int(value) for value in coordinates),
                "coefficient": float(coefficient),
            }
            for coordinates, coefficient in sorted(row.items())
        )
        row_hashes.append(_payload_hash({"terms": terms}))
    return _payload_hash({"row_hashes": tuple(row_hashes)})


def _composite_physical_real_descriptor_rows(lifted_model):
    if lifted_model.compiled is not None:
        raise ValueError("Expected a composite lifted-Cauchy model.")
    component_maps = tuple(lifted_model.evaluator.component_channel_maps)
    if len(component_maps) != len(lifted_model.compiled_components):
        raise ValueError("Composite component channel maps are incomplete.")
    certified_expansion_count = sum(
        int(
            compiled.payload["physical_scalar_reality_report"][
                "transformed_coefficient_count"
            ]
        )
        for compiled in lifted_model.compiled_components
    )
    if certified_expansion_count > _MAX_COMPOSITE_PHYSICAL_TERMS:
        raise MemoryError(
            "Composite compiler certificates exceed five million physical-real "
            "coefficients; use a compiler-certified factored runtime plan instead."
        )
    global_offsets = {}
    source_variable_count = 0
    for channel in lifted_model.source.channels:
        channel_index = int(channel["channel_index"])
        global_offsets[channel_index] = source_variable_count
        source_variable_count += 2 * (2 * int(channel["l"]) + 1)

    rows = []
    component_certificates = []
    maximum_imaginary = 0.0
    maximum_scale = 1.0
    expanded_term_count = 0
    descriptor_offset = 0
    for position, (compiled, channel_map) in enumerate(
        zip(lifted_model.compiled_components, component_maps, strict=True)
    ):
        local_rows, certificate = _physical_real_descriptor_rows(compiled)
        local_to_global = []
        local_offset = 0
        local_channels = tuple(
            sorted(
                compiled.payload["channels"],
                key=lambda channel: int(channel["channel_index"]),
            )
        )
        if len(local_channels) != len(channel_map):
            raise ValueError("Composite component channel map has the wrong size.")
        for local_position, channel in enumerate(local_channels):
            if int(channel["channel_index"]) != local_position:
                raise ValueError("Composite compiler channels must be dense and ordered.")
            global_channel = int(channel_map[local_position])
            width = 2 * int(channel["l"]) + 1
            global_offset = global_offsets[global_channel]
            for role in range(2):
                for component in range(width):
                    local_to_global.append(global_offset + role * width + component)
                    local_offset += 1
        if local_offset != int(certificate["source_variable_count"]):
            raise RuntimeError("Composite source remap width changed.")
        remapped_rows = []
        for row in local_rows:
            remapped = {}
            for coordinates, coefficient in row.items():
                key = tuple(sorted(local_to_global[index] for index in coordinates))
                remapped[key] = remapped.get(key, 0.0) + float(coefficient)
            remapped_rows.append(remapped)
        rows.extend(remapped_rows)
        expanded_term_count += int(certificate["physical_real_term_count_before_readout"])
        if expanded_term_count > _MAX_COMPOSITE_PHYSICAL_TERMS:
            raise MemoryError(
                "Composite native lowering exceeds five million physical-real terms; "
                "use a compiler-certified factored runtime plan instead."
            )
        maximum_imaginary = max(
            maximum_imaginary,
            float(certificate["maximum_absolute_imaginary_residual"]),
        )
        maximum_scale = max(
            maximum_scale, float(certificate["maximum_coefficient_scale"])
        )
        component_certificates.append(
            {
                "position": position,
                "artifact_self_hash": str(compiled.self_hash),
                "descriptor_offset": descriptor_offset,
                "descriptor_count": len(remapped_rows),
                "component_channel_map": tuple(int(value) for value in channel_map),
                "physical_real_rows_sha256": _physical_rows_hash(remapped_rows),
                "physical_real_term_count": int(
                    certificate["physical_real_term_count_before_readout"]
                ),
                "maximum_absolute_imaginary_residual": float(
                    certificate["maximum_absolute_imaginary_residual"]
                ),
                "maximum_coefficient_scale": float(
                    certificate["maximum_coefficient_scale"]
                ),
                "physical_scalar_reality_report_hash": str(
                    certificate["physical_scalar_reality_report_hash"]
                ),
            }
        )
        descriptor_offset += len(remapped_rows)
    if descriptor_offset != int(lifted_model.evaluator.descriptor_count):
        raise RuntimeError("Composite descriptor concatenation changed.")
    return tuple(rows), {
        "descriptor_count": descriptor_offset,
        "physical_real_term_count_before_readout": expanded_term_count,
        "maximum_absolute_imaginary_residual": maximum_imaginary,
        "maximum_coefficient_scale": maximum_scale,
        "source_variable_count": source_variable_count,
        "channel_source_variable_offsets": tuple(
            int(global_offsets[int(channel["channel_index"])])
            for channel in lifted_model.source.channels
        ),
        "component_certificates": tuple(component_certificates),
    }


def _native_sparse_polynomial(rows, weights, offset):
    combined = {}
    constant = float(offset)
    for descriptor_index, row in enumerate(rows):
        weight = float(weights[descriptor_index])
        if weight == 0.0:
            continue
        for coordinates, coefficient in row.items():
            value = weight * float(coefficient)
            if not coordinates:
                constant += value
            else:
                combined[coordinates] = combined.get(coordinates, 0.0) + value

    factor_offsets = [0]
    factor_indices = []
    factor_exponents = []
    coefficients = []
    maximum_rank = 0
    maximum_factor_count = 0
    for coordinates in sorted(combined):
        coefficient = float(combined[coordinates])
        if coefficient == 0.0:
            continue
        factors = []
        for coordinate in coordinates:
            if factors and factors[-1][0] == coordinate:
                factors[-1][1] += 1
            else:
                factors.append([int(coordinate), 1])
        maximum_rank = max(maximum_rank, len(coordinates))
        maximum_factor_count = max(maximum_factor_count, len(factors))
        for coordinate, exponent in factors:
            factor_indices.append(coordinate)
            factor_exponents.append(exponent)
        factor_offsets.append(len(factor_indices))
        coefficients.append(coefficient)
    return {
        "offset": constant,
        "factor_offsets": factor_offsets,
        "factor_indices": factor_indices,
        "factor_exponents": factor_exponents,
        "monomial_coefficients": coefficients,
        "term_count": len(coefficients),
        "maximum_tensor_rank": int(maximum_rank),
        "maximum_factor_count": int(maximum_factor_count),
    }


def _sparse_polynomial_static_bytes(polynomial):
    return int(
        8 * len(polynomial["factor_offsets"])
        + 8 * len(polynomial["factor_indices"])
        + 8 * len(polynomial["factor_exponents"])
        + 8 * len(polynomial["monomial_coefficients"])
        + 8
    )


def _native_source_groups(source, central_species_order):
    species_positions = {
        str(species): index for index, species in enumerate(central_species_order)
    }
    channel_positions = {
        int(channel["channel_index"]): position
        for position, channel in enumerate(source.channels)
    }
    channel_offsets = {}
    source_variable_count = 0
    for channel in source.channels:
        channel_index = int(channel["channel_index"])
        channel_offsets[channel_index] = source_variable_count
        source_variable_count += 2 * (2 * int(channel["l"]) + 1)
    covered_channels = set()
    groups = []
    for group in source.config["groups"]:
        neighbor_species = str(group["neighbor_species"])
        if neighbor_species not in species_positions:
            raise ValueError(
                "Native lifted-Cauchy source species is absent from central_species_order."
            )
        channel_indices = tuple(int(value) for value in group["channel_indices"])
        covered_channels.update(channel_indices)
        channel_position_values = tuple(
            channel_positions[index] for index in channel_indices
        )
        q_source_offsets = []
        direct_polynomials = []
        factorized_radials = []
        for polynomial in group["polynomials"]:
            q = int(polynomial["q"])
            channel_position = channel_position_values[q // 2]
            channel_index = channel_indices[q // 2]
            role = q % 2
            component_count = 2 * int(group["l"]) + 1
            q_source_offsets.append(
                channel_offsets[channel_index] + role * component_count
            )
            norm = polynomial["normalization_squared"]
            scale = np.sqrt(float(int(norm["numerator"]) / int(norm["denominator"])))
            direct_polynomials.append(
                {
                    "q": q,
                    "coefficients": [
                        float(scale * int(value))
                        for value in polynomial[
                            "shifted_jacobi_power_coefficients"
                        ]
                    ],
                }
            )
            factorized_radials.append(
                {
                    "q": q,
                    "x_power": q,
                    "envelope_power": 3 if q % 2 == 0 else 2,
                }
            )
        groups.append(
            {
                "neighbor_species": neighbor_species,
                "neighbor_species_index": int(species_positions[neighbor_species]),
                "l": int(group["l"]),
                "real_component_count": 2 * int(group["l"]) + 1,
                "channel_indices": list(channel_indices),
                "channel_positions": list(channel_position_values),
                "source_dimension": int(group["source_dimension"]),
                "q_source_variable_offsets": q_source_offsets,
                "direct_q_polynomials": direct_polynomials,
                "factorized_radials": factorized_radials,
                "transform_q_from_f": [
                    [float(value) for value in row]
                    for row in group["factorized_lowering"]["binary64_matrix"]
                ],
            }
        )
    expected_channels = {int(channel["channel_index"]) for channel in source.channels}
    if covered_channels != expected_channels:
        raise ValueError("Native lifted-Cauchy source groups do not cover every channel.")
    return groups, channel_offsets, source_variable_count


def _native_runtime_payload(
    lifted_model, coefficients, offsets, model_family, ordinary_reference
):
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LIFTED_CAUCHY_JOINT_SOURCE_SCHEMA,
        LIFTED_CAUCHY_MIXED_L_SOURCE_SCHEMA,
    )

    if lifted_model.source.config["schema"] not in {
        LIFTED_CAUCHY_JOINT_SOURCE_SCHEMA,
        LIFTED_CAUCHY_MIXED_L_SOURCE_SCHEMA,
    }:
        return None
    rows, realification = _physical_real_descriptor_rows(lifted_model.compiled)
    source_groups, channel_offsets, source_variable_count = _native_source_groups(
        lifted_model.source, lifted_model.central_species_order
    )
    if int(realification["source_variable_count"]) != int(source_variable_count):
        raise RuntimeError("Native source packing disagrees with realification.")
    heads = []
    for head, species in enumerate(lifted_model.central_species_order):
        heads.append(
            {
                "central_species": str(species),
                "central_species_index": int(head),
                "polynomial": _native_sparse_polynomial(
                    rows, coefficients[head], offsets[head]
                ),
            }
        )
    deployment_identity = {
        "compiler_artifact_self_hash": str(lifted_model.compiled.self_hash),
        "source_plan_hash": str(lifted_model.source.source_plan_hash),
        "type_map": _jsonable(lifted_model.type_map),
        "central_species_order": list(lifted_model.central_species_order),
        "readout_coefficients": coefficients.tolist(),
        "readout_offsets": offsets.tolist(),
        "conventions": dict(_CONVENTIONS),
        "ordinary_reference": ordinary_reference,
    }
    payload = {
        "schema": LIFTED_CAUCHY_NATIVE_SCHEMA,
        "model_family": str(model_family),
        "deployment_identity_sha256": _payload_hash(deployment_identity),
        "compiler_artifact_self_hash": str(lifted_model.compiled.self_hash),
        "source_plan_hash": str(lifted_model.source.source_plan_hash),
        "type_map": _jsonable(lifted_model.type_map),
        "coordinate_convention": {
            "id": "ye3t_physical_real_tesseral_v1",
            "component_order": "cosine_l_to_1_then_m0_then_sine_1_to_l",
            "source_variable_order": "packed_channel_then_role_then_real_component",
            "channel_blocks": [
                {
                    "channel_index": int(channel["channel_index"]),
                    "channel_position": int(position),
                    "l": int(channel["l"]),
                    "real_component_count": 2 * int(channel["l"]) + 1,
                    "source_variable_offset": int(
                        channel_offsets[int(channel["channel_index"])]
                    ),
                }
                for position, channel in enumerate(lifted_model.source.channels)
            ],
        },
        "central_species_order": list(lifted_model.central_species_order),
        "role_dimension": 2,
        "real_component_count": int(lifted_model.source.real_component_count),
        "source_variable_count": int(source_variable_count),
        "cutoff_A": float(lifted_model.source.cutoff),
        "source_groups": source_groups,
        "heads": heads,
        "capabilities": {
            "direct_orthogonal_q_source": True,
            "factorized_source_then_center_transform": True,
            "factorized_reverse_uses_algebraic_transpose": True,
            "physical_real_sparse_polynomial": True,
            "runtime_gram_solve": False,
        },
        "certificates": {
            **realification,
            "source_gram": "identity_exact",
            "source_forward": "A_Q=T*A_f",
            "source_reverse": "bar_A_f=T^T*bar_A_Q",
            "readout_lowering": "fit_coordinates_to_pivot_then_physical_real_polynomial",
            "total_lowered_term_count": int(
                sum(head["polynomial"]["term_count"] for head in heads)
            ),
        },
    }
    payload["self_hash"] = _payload_hash(payload)
    return payload


def _composite_compiler_binding(
    lifted_model,
    coefficients,
    offsets,
    component_references,
    realification,
):
    records = tuple(dict(value) for value in lifted_model.component_records)
    if len(records) != len(component_references):
        raise ValueError("Composite compiler references are incomplete.")
    components = []
    for position, (compiled, record, reference, certificate) in enumerate(
        zip(
            lifted_model.compiled_components,
            records,
            component_references,
            realification["component_certificates"],
            strict=True,
        )
    ):
        if int(record["position"]) != position:
            raise ValueError("Composite component positions changed before export.")
        if str(record["artifact_hash"]) != str(compiled.self_hash):
            raise ValueError("Composite component artifact identity changed before export.")
        if str(certificate["artifact_self_hash"]) != str(compiled.self_hash):
            raise RuntimeError("Composite realification certificate changed artifact order.")
        components.append(
            {
                "position": position,
                "component_id": str(record["component_id"]),
                "opportunity_id": str(record["opportunity_id"]),
                "compiler_artifact": dict(reference),
                "selection_hash": str(record["selection_hash"]),
                "coordinate_ids": tuple(str(value) for value in record["coordinate_ids"]),
                "strict_sector_ids": tuple(
                    str(value) for value in record["strict_sector_ids"]
                ),
                "descriptor_offset": int(certificate["descriptor_offset"]),
                "descriptor_count": int(certificate["descriptor_count"]),
                "component_channel_map": tuple(
                    int(value) for value in certificate["component_channel_map"]
                ),
                "physical_real_rows_sha256": str(
                    certificate["physical_real_rows_sha256"]
                ),
                "orthogonal_output_plan_hash": record.get(
                    "orthogonal_output_plan_hash"
                ),
                "physical_scalar_reality_report_hash": str(
                    certificate["physical_scalar_reality_report_hash"]
                ),
                "physical_real_term_count": int(
                    certificate["physical_real_term_count"]
                ),
            }
        )
    source_registry = tuple(
        {
            "channel_index": int(channel["channel_index"]),
            "neighbor_species": str(channel["neighbor_species"]),
            "source_family_id": str(channel["source_family_id"]),
            "l": int(channel["l"]),
            "radial_channel": int(channel["radial_channel"]),
        }
        for channel in lifted_model.source.channels
    )
    body = {
        "schema": LIFTED_CAUCHY_COMPILER_BINDING_SCHEMA,
        "composite_artifact_hash": str(lifted_model.artifact_hash),
        "source_plan_hash": str(lifted_model.source.source_plan_hash),
        "descriptor_coordinate_ids": tuple(lifted_model.descriptor_coordinate_ids),
        "components": tuple(components),
        "source_registry": source_registry,
        "source_registry_sha256": _payload_hash({"channels": source_registry}),
        "readout": {
            "coefficient_shape": tuple(int(value) for value in coefficients.shape),
            "coefficients_sha256": _array_hash(coefficients),
            "coefficients_json_sha256": hashlib.sha256(
                _canonical_json_bytes(coefficients.tolist())
            ).hexdigest(),
            "offset_shape": tuple(int(value) for value in offsets.shape),
            "offsets_sha256": _array_hash(offsets),
            "offsets_json_sha256": hashlib.sha256(
                _canonical_json_bytes(offsets.tolist())
            ).hexdigest(),
        },
        "fit_coordinate_provenance": _jsonable(lifted_model.fit_metadata),
        "fit_coordinate_provenance_sha256": _payload_hash(
            {"fit_metadata": _jsonable(lifted_model.fit_metadata)}
        ),
        "physical_real_lowering": {
            "algorithm": "component_canonical_rows_then_shared_source_remap_v1",
            "real_basis_id": _CONVENTIONS["real_basis_id"],
            "descriptor_count": int(realification["descriptor_count"]),
            "source_variable_count": int(realification["source_variable_count"]),
            "physical_real_term_count_before_readout": int(
                realification["physical_real_term_count_before_readout"]
            ),
            "maximum_absolute_imaginary_residual": float(
                realification["maximum_absolute_imaginary_residual"]
            ),
            "maximum_coefficient_scale": float(
                realification["maximum_coefficient_scale"]
            ),
            "component_row_certificate_sha256": _payload_hash(
                {"components": realification["component_certificates"]}
            ),
        },
        "validation": {
            "passed": True,
            "component_artifact_hashes_verified": True,
            "component_order_verified": True,
            "strict_sector_disjoint_record_asserted": True,
            "strict_sector_identity_provenance": (
                "hash_bound_component_records_and_fit_certificate"
            ),
            "shared_source_registry_verified": True,
            "runtime_gram_solve": False,
        },
    }
    return {**body, "self_hash": _payload_hash(body)}


def _composite_native_runtime_payload(
    lifted_model,
    coefficients,
    offsets,
    compiler_binding,
    rows,
    realification,
):
    source_groups, channel_offsets, source_variable_count = _native_source_groups(
        lifted_model.source, lifted_model.central_species_order
    )
    if int(realification["source_variable_count"]) != int(source_variable_count):
        raise RuntimeError("Composite source packing disagrees with realification.")
    heads = []
    total_static_bytes = 0
    for head, species in enumerate(lifted_model.central_species_order):
        polynomial = _native_sparse_polynomial(rows, coefficients[head], offsets[head])
        total_static_bytes += _sparse_polynomial_static_bytes(polynomial)
        heads.append(
            {
                "central_species": str(species),
                "central_species_index": int(head),
                "polynomial": polynomial,
            }
        )
    if total_static_bytes > _MAX_COMPOSITE_STATIC_BYTES:
        raise MemoryError(
            "Composite native polynomial exceeds the 128 MiB static-array bound; "
            "use a compiler-certified factored runtime plan instead."
        )
    deployment_identity = {
        "compiler_binding_self_hash": str(compiler_binding["self_hash"]),
        "composite_artifact_hash": str(lifted_model.artifact_hash),
        "source_plan_hash": str(lifted_model.source.source_plan_hash),
        "type_map": _jsonable(lifted_model.type_map),
        "central_species_order": list(lifted_model.central_species_order),
        "readout_coefficients": coefficients.tolist(),
        "readout_offsets": offsets.tolist(),
        "conventions": dict(_CONVENTIONS),
        "ordinary_reference": None,
    }
    payload = {
        "schema": LIFTED_CAUCHY_COMPOSITE_NATIVE_SCHEMA,
        "model_family": "linear_lifted_cauchy_scalar",
        "deployment_identity_sha256": _payload_hash(deployment_identity),
        "compiler_binding_self_hash": str(compiler_binding["self_hash"]),
        "composite_artifact_hash": str(lifted_model.artifact_hash),
        "source_plan_hash": str(lifted_model.source.source_plan_hash),
        "type_map": _jsonable(lifted_model.type_map),
        "coordinate_convention": {
            "id": "ye3t_physical_real_tesseral_v1",
            "component_order": "cosine_l_to_1_then_m0_then_sine_1_to_l",
            "source_variable_order": "packed_channel_then_role_then_real_component",
            "channel_blocks": [
                {
                    "channel_index": int(channel["channel_index"]),
                    "channel_position": int(position),
                    "l": int(channel["l"]),
                    "real_component_count": 2 * int(channel["l"]) + 1,
                    "source_variable_offset": int(
                        channel_offsets[int(channel["channel_index"])]
                    ),
                }
                for position, channel in enumerate(lifted_model.source.channels)
            ],
        },
        "central_species_order": list(lifted_model.central_species_order),
        "role_dimension": 2,
        "real_component_count": int(lifted_model.source.real_component_count),
        "source_variable_count": int(source_variable_count),
        "cutoff_A": float(lifted_model.source.cutoff),
        "source_groups": source_groups,
        "heads": heads,
        "capabilities": {
            "direct_orthogonal_q_source": True,
            "factorized_source_then_center_transform": True,
            "factorized_reverse_uses_algebraic_transpose": True,
            "physical_real_sparse_polynomial": True,
            "composite_compiler_binding": True,
            "runtime_compiler_dom": False,
            "runtime_gram_solve": False,
        },
        "certificates": {
            "descriptor_count": int(realification["descriptor_count"]),
            "physical_real_term_count_before_readout": int(
                realification["physical_real_term_count_before_readout"]
            ),
            "maximum_absolute_imaginary_residual": float(
                realification["maximum_absolute_imaginary_residual"]
            ),
            "maximum_coefficient_scale": float(
                realification["maximum_coefficient_scale"]
            ),
            "source_variable_count": int(source_variable_count),
            "channel_source_variable_offsets": tuple(
                int(value)
                for value in realification["channel_source_variable_offsets"]
            ),
            "component_row_certificate_sha256": str(
                compiler_binding["physical_real_lowering"][
                    "component_row_certificate_sha256"
                ]
            ),
            "source_gram": "identity_exact",
            "source_forward": "A_Q=T*A_f",
            "source_reverse": "bar_A_f=T^T*bar_A_Q",
            "readout_lowering": (
                "fit_coordinates_to_component_pivots_then_shared_physical_real_polynomial"
            ),
            "total_lowered_term_count": int(
                sum(head["polynomial"]["term_count"] for head in heads)
            ),
            "estimated_static_array_bytes": total_static_bytes,
        },
    }
    payload["self_hash"] = _payload_hash(payload)
    if len(_pretty_json_bytes(payload)) > _MAX_PORTABLE_JSON_BYTES:
        raise MemoryError(
            "Composite native JSON exceeds the 96 MiB portable/GitHub-safe bound; "
            "use a compiler-certified factored runtime plan instead."
        )
    return payload


def _bundle_member(root, name):
    name = str(name)
    candidate = Path(name)
    if candidate.name != name or candidate.is_absolute() or name in {"", ".", ".."}:
        raise ValueError("Lifted-Cauchy bundle references must be simple relative filenames.")
    return Path(root) / name


def _read_manifest(path):
    entries = {}
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 2:
            raise ValueError(
                f"Malformed lifted bundle manifest line {line_number}."
            )
        digest, name = fields
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or name in entries
        ):
            raise ValueError(
                f"Malformed lifted bundle manifest line {line_number}."
            )
        _bundle_member(Path(path).parent, name)
        entries[name] = digest
    return entries


def _export_composite_lifted_cauchy_bundle(
    lifted_model,
    output,
    coefficients,
    offsets,
    fit_metadata,
):
    if lifted_model.compiled is not None or not lifted_model.compiled_components:
        raise ValueError("Expected a composite lifted-Cauchy model.")
    component_references = []
    manifest_entries = {}
    for position, compiled in enumerate(lifted_model.compiled_components):
        name = f"compiled_component_{position:03d}.json"
        path = output / name
        _write_json(path, compiled.to_dict())
        file_sha256 = _file_hash(path)
        component_references.append(
            {
                "file": name,
                "file_sha256": file_sha256,
                "artifact_self_hash": str(compiled.self_hash),
            }
        )
        manifest_entries[name] = file_sha256

    rows, realification = _composite_physical_real_descriptor_rows(lifted_model)
    binding = _composite_compiler_binding(
        lifted_model,
        coefficients,
        offsets,
        component_references,
        realification,
    )
    binding_path = output / LIFTED_CAUCHY_COMPILER_BINDING_FILE
    _write_json(binding_path, binding)
    binding_sha256 = _file_hash(binding_path)
    manifest_entries[LIFTED_CAUCHY_COMPILER_BINDING_FILE] = binding_sha256

    native = _composite_native_runtime_payload(
        lifted_model,
        coefficients,
        offsets,
        binding,
        rows,
        realification,
    )
    native_path = output / LIFTED_CAUCHY_NATIVE_FILE
    _write_json(native_path, native)
    native_sha256 = _file_hash(native_path)
    manifest_entries[LIFTED_CAUCHY_NATIVE_FILE] = native_sha256

    head_count, feature_count = coefficients.shape
    payload = {
        "schema": LIFTED_CAUCHY_COMPOSITE_BUNDLE_SCHEMA,
        "model_family": "linear_lifted_cauchy_scalar",
        "compiler_binding": {
            "file": LIFTED_CAUCHY_COMPILER_BINDING_FILE,
            "file_sha256": binding_sha256,
            "binding_self_hash": str(binding["self_hash"]),
            "composite_artifact_hash": str(lifted_model.artifact_hash),
        },
        "source": _jsonable(lifted_model.source.config),
        "source_plan_hash": str(lifted_model.source.source_plan_hash),
        "type_map": _jsonable(lifted_model.type_map),
        "central_species_order": list(lifted_model.central_species_order),
        "readout": {
            "dtype": "float64",
            "coefficient_shape": [head_count, feature_count],
            "coefficients": coefficients.tolist(),
            "offset_shape": [head_count],
            "offsets": offsets.tolist(),
        },
        "default_realization": str(lifted_model.realization),
        "default_source_realization": str(lifted_model.source.source_realization),
        "conventions": dict(_CONVENTIONS),
        "fit_metadata": _jsonable(fit_metadata),
        "ordinary_reference": None,
        "native_runtime_reference": {
            "file": LIFTED_CAUCHY_NATIVE_FILE,
            "format": LIFTED_CAUCHY_COMPOSITE_NATIVE_SCHEMA,
            "file_sha256": native_sha256,
            "plan_self_hash": str(native["self_hash"]),
        },
    }
    payload["self_hash"] = _payload_hash(payload)
    model_path = output / LIFTED_CAUCHY_MODEL_FILE
    _write_json(model_path, payload)
    manifest_entries[LIFTED_CAUCHY_MODEL_FILE] = _file_hash(model_path)
    (output / LIFTED_CAUCHY_MANIFEST_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(manifest_entries.items())
        ),
        encoding="utf-8",
    )
    return model_path


def export_lifted_cauchy_linear_bundle(model, directory):
    """
    Purpose:
        Export a portable inference bundle for a fitted lifted-Cauchy model.
    Mathematical contract:
        Preserve the compiler artifact, role source, real-basis convention,
        exact binary64 readout, and optional additive ordinary ACE component.
    Inputs:
        A fitted lifted-only or ordinary-plus-lifted model and an empty output
        directory path.
    Outputs:
        The path to a self-hashed model JSON with a SHA-256 file manifest.
    Does not:
        Compile symmetry labels, fit coefficients, or execute metadata in the
        timestep hot path.
    """

    from ye3t_methods.atomistic.ace.linear_ace import export_scalar_bundle_to_yace
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        _LiftedCauchyLinearModel,
        _OrdinaryLiftedCauchyLinearModel,
    )

    if isinstance(model, _OrdinaryLiftedCauchyLinearModel):
        lifted_model = model.lifted_model
        model_family = "linear_ordinary_plus_lifted_cauchy_scalar"
        fit_metadata = dict(model.fit_metadata)
        ordinary_bundle = model.ordinary_bundle
    elif isinstance(model, _LiftedCauchyLinearModel):
        lifted_model = model
        model_family = "linear_lifted_cauchy_scalar"
        fit_metadata = dict(model.fit_metadata)
        ordinary_bundle = None
    else:
        raise TypeError(
            "Expected a fitted lifted-Cauchy or ordinary-plus-lifted model."
        )

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            "Lifted-Cauchy bundle output directory must be empty."
        )

    coefficients = lifted_model.coefficients.detach().cpu().numpy()
    offsets = lifted_model.offsets.detach().cpu().numpy()
    feature_count = int(lifted_model.evaluator.descriptor_count)
    head_count = len(lifted_model.central_species_order)
    coefficients = _finite_matrix(
        coefficients, (head_count, feature_count), "lifted coefficients"
    )
    offsets = _finite_matrix(offsets, (head_count,), "lifted offsets")
    if lifted_model.compiled is None:
        if ordinary_bundle is not None:
            raise NotImplementedError(
                "Composite lifted export does not yet support an additive ordinary model."
            )
        return _export_composite_lifted_cauchy_bundle(
            lifted_model,
            output,
            coefficients,
            offsets,
            fit_metadata,
        )

    compiled_path = output / LIFTED_CAUCHY_COMPILER_FILE
    native_path = output / LIFTED_CAUCHY_NATIVE_FILE
    model_path = output / LIFTED_CAUCHY_MODEL_FILE
    manifest_path = output / LIFTED_CAUCHY_MANIFEST_FILE
    compiled_payload = lifted_model.compiled.to_dict()
    _write_json(compiled_path, compiled_payload)
    compiled_sha256 = _file_hash(compiled_path)

    ordinary_reference = None
    if ordinary_bundle is not None:
        ordinary_path = output / LIFTED_CAUCHY_ORDINARY_FILE
        export_scalar_bundle_to_yace(
            ordinary_bundle,
            ordinary_path,
            elements=list(lifted_model.central_species_order),
            compatibility="lammps_pace_linear_v1",
        )
        ordinary_reference = {
            "file": LIFTED_CAUCHY_ORDINARY_FILE,
            "format": "lammps_pace_linear_v1",
            "sha256": _file_hash(ordinary_path),
            "combination": "additive_energy_force_virial",
        }

    native_payload = _native_runtime_payload(
        lifted_model,
        coefficients,
        offsets,
        model_family,
        ordinary_reference,
    )
    native_reference = None
    if native_payload is not None:
        _write_json(native_path, native_payload)
        native_reference = {
            "file": LIFTED_CAUCHY_NATIVE_FILE,
            "format": LIFTED_CAUCHY_NATIVE_SCHEMA,
            "file_sha256": _file_hash(native_path),
            "plan_self_hash": str(native_payload["self_hash"]),
        }

    payload = {
        "schema": LIFTED_CAUCHY_BUNDLE_SCHEMA,
        "model_family": model_family,
        "compiler_artifact": {
            "file": LIFTED_CAUCHY_COMPILER_FILE,
            "file_sha256": compiled_sha256,
            "artifact_self_hash": str(lifted_model.compiled.self_hash),
        },
        "source": _jsonable(lifted_model.source.config),
        "source_plan_hash": str(lifted_model.source.source_plan_hash),
        "type_map": _jsonable(lifted_model.type_map),
        "central_species_order": list(lifted_model.central_species_order),
        "readout": {
            "dtype": "float64",
            "coefficient_shape": [head_count, feature_count],
            "coefficients": coefficients.tolist(),
            "offset_shape": [head_count],
            "offsets": offsets.tolist(),
        },
        "default_realization": str(lifted_model.realization),
        "default_source_realization": str(lifted_model.source.source_realization),
        "conventions": dict(_CONVENTIONS),
        "fit_metadata": _jsonable(fit_metadata),
        "ordinary_reference": ordinary_reference,
        "native_runtime_reference": native_reference,
    }
    payload["self_hash"] = _payload_hash(payload)
    _write_json(model_path, payload)

    manifest_entries = {
        LIFTED_CAUCHY_COMPILER_FILE: compiled_sha256,
        LIFTED_CAUCHY_MODEL_FILE: _file_hash(model_path),
    }
    if ordinary_reference is not None:
        manifest_entries[LIFTED_CAUCHY_ORDINARY_FILE] = ordinary_reference[
            "sha256"
        ]
    if native_reference is not None:
        manifest_entries[LIFTED_CAUCHY_NATIVE_FILE] = native_reference[
            "file_sha256"
        ]
    manifest_path.write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(manifest_entries.items())
        ),
        encoding="utf-8",
    )
    return model_path


def _load_composite_lifted_cauchy_bundle(
    root,
    payload,
    entries,
    realization,
    source_realization,
):
    from ye3t_methods.atomistic.lifted_cauchy_linear import _LiftedCauchyLinearModel

    expected_model_fields = {
        "schema",
        "model_family",
        "compiler_binding",
        "source",
        "source_plan_hash",
        "type_map",
        "central_species_order",
        "readout",
        "default_realization",
        "default_source_realization",
        "conventions",
        "fit_metadata",
        "ordinary_reference",
        "native_runtime_reference",
        "self_hash",
    }
    if set(payload) != expected_model_fields:
        raise ValueError("Composite lifted bundle model fields are incomplete.")
    if payload["model_family"] != "linear_lifted_cauchy_scalar":
        raise ValueError("Unsupported composite lifted-Cauchy model family.")
    if payload["ordinary_reference"] is not None:
        raise ValueError("Composite lifted bundle cannot contain an ordinary reference.")
    if payload["conventions"] != _CONVENTIONS:
        raise ValueError("Composite lifted bundle conventions are unsupported.")
    if str(payload.get("self_hash", "")) != _payload_hash(payload):
        raise ValueError("Composite lifted model self-hash mismatch.")

    binding_reference = dict(payload["compiler_binding"])
    if set(binding_reference) != {
        "file",
        "file_sha256",
        "binding_self_hash",
        "composite_artifact_hash",
    }:
        raise ValueError("Composite compiler-binding reference is incomplete.")
    binding_name = str(binding_reference["file"])
    binding_path = _bundle_member(root, binding_name)
    if entries.get(binding_name) != binding_reference["file_sha256"]:
        raise ValueError("Composite compiler-binding manifest identity mismatch.")
    if _file_hash(binding_path) != binding_reference["file_sha256"]:
        raise ValueError("Composite compiler-binding file SHA-256 mismatch.")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    expected_binding_fields = {
        "schema",
        "composite_artifact_hash",
        "source_plan_hash",
        "descriptor_coordinate_ids",
        "components",
        "source_registry",
        "source_registry_sha256",
        "readout",
        "fit_coordinate_provenance",
        "fit_coordinate_provenance_sha256",
        "physical_real_lowering",
        "validation",
        "self_hash",
    }
    if set(binding) != expected_binding_fields:
        raise ValueError("Composite compiler-binding fields are incomplete.")
    if binding["schema"] != LIFTED_CAUCHY_COMPILER_BINDING_SCHEMA:
        raise ValueError("Unsupported composite compiler-binding schema.")
    if str(binding.get("self_hash", "")) != _payload_hash(binding):
        raise ValueError("Composite compiler-binding self-hash mismatch.")
    if str(binding["self_hash"]) != str(binding_reference["binding_self_hash"]):
        raise ValueError("Composite compiler-binding self-reference mismatch.")
    if str(binding["composite_artifact_hash"]) != str(
        binding_reference["composite_artifact_hash"]
    ):
        raise ValueError("Composite artifact identity changed in its binding.")
    validation = dict(binding["validation"])
    if validation != {
        "passed": True,
        "component_artifact_hashes_verified": True,
        "component_order_verified": True,
        "strict_sector_disjoint_record_asserted": True,
        "strict_sector_identity_provenance": (
            "hash_bound_component_records_and_fit_certificate"
        ),
        "shared_source_registry_verified": True,
        "runtime_gram_solve": False,
    }:
        raise ValueError("Composite compiler-binding validation did not pass.")
    if str(binding["fit_coordinate_provenance_sha256"]) != _payload_hash(
        {"fit_metadata": binding["fit_coordinate_provenance"]}
    ) or binding["fit_coordinate_provenance"] != payload["fit_metadata"]:
        raise ValueError("Composite fit-coordinate provenance changed.")

    compiled_payloads = []
    component_records = []
    component_maps = []
    expected_files = {
        LIFTED_CAUCHY_MODEL_FILE,
        binding_name,
        LIFTED_CAUCHY_NATIVE_FILE,
    }
    descriptor_offset = 0
    for position, component in enumerate(binding["components"]):
        component = dict(component)
        expected_component_fields = {
            "position",
            "component_id",
            "opportunity_id",
            "compiler_artifact",
            "selection_hash",
            "coordinate_ids",
            "strict_sector_ids",
            "descriptor_offset",
            "descriptor_count",
            "component_channel_map",
            "physical_real_rows_sha256",
            "orthogonal_output_plan_hash",
            "physical_scalar_reality_report_hash",
            "physical_real_term_count",
        }
        if set(component) != expected_component_fields:
            raise ValueError("Composite compiler component fields are incomplete.")
        if int(component["position"]) != position:
            raise ValueError("Composite compiler component positions are not dense.")
        if int(component["descriptor_offset"]) != descriptor_offset:
            raise ValueError("Composite compiler descriptor offsets are inconsistent.")
        reference = dict(component["compiler_artifact"])
        if set(reference) != {"file", "file_sha256", "artifact_self_hash"}:
            raise ValueError("Composite component artifact reference is incomplete.")
        name = str(reference["file"])
        path = _bundle_member(root, name)
        expected_files.add(name)
        if entries.get(name) != reference["file_sha256"]:
            raise ValueError("Composite component manifest identity mismatch.")
        if _file_hash(path) != reference["file_sha256"]:
            raise ValueError("Composite component file SHA-256 mismatch.")
        compiled_payload = json.loads(path.read_text(encoding="utf-8"))
        if str(compiled_payload.get("self_hash", "")) != str(
            reference["artifact_self_hash"]
        ) or str(compiled_payload.get("self_hash", "")) != _payload_hash(
            compiled_payload
        ):
            raise ValueError("Composite component artifact self-hash mismatch.")
        descriptor_count = int(component["descriptor_count"])
        if descriptor_count != len(compiled_payload["payload"]["descriptors"]):
            raise ValueError("Composite component descriptor count changed.")
        coordinate_ids = tuple(str(value) for value in component["coordinate_ids"])
        if len(coordinate_ids) != descriptor_count:
            raise ValueError("Composite component coordinate count changed.")
        compiled_payloads.append(compiled_payload)
        component_maps.append(
            tuple(int(value) for value in component["component_channel_map"])
        )
        component_records.append(
            {
                "position": position,
                "component_id": str(component["component_id"]),
                "opportunity_id": str(component["opportunity_id"]),
                "artifact_hash": str(reference["artifact_self_hash"]),
                "selection_hash": str(component["selection_hash"]),
                "coordinate_ids": coordinate_ids,
                "strict_sector_ids": tuple(
                    str(value) for value in component["strict_sector_ids"]
                ),
                "descriptor_count": descriptor_count,
            }
        )
        descriptor_offset += descriptor_count
    if set(entries) != expected_files:
        raise ValueError("Composite lifted manifest contains missing or unexpected files.")
    if tuple(binding["descriptor_coordinate_ids"]) != tuple(
        coordinate
        for record in component_records
        for coordinate in record["coordinate_ids"]
    ):
        raise ValueError("Composite descriptor coordinate order changed.")

    readout = dict(payload["readout"])
    if set(readout) != {
        "dtype",
        "coefficient_shape",
        "coefficients",
        "offset_shape",
        "offsets",
    } or readout["dtype"] != "float64":
        raise ValueError("Composite lifted readout is incomplete.")
    coefficient_shape = tuple(int(value) for value in readout["coefficient_shape"])
    offset_shape = tuple(int(value) for value in readout["offset_shape"])
    coefficients = _finite_matrix(
        readout["coefficients"], coefficient_shape, "composite coefficients"
    )
    offsets = _finite_matrix(readout["offsets"], offset_shape, "composite offsets")
    binding_readout = dict(binding["readout"])
    if set(binding_readout) != {
        "coefficient_shape",
        "coefficients_sha256",
        "coefficients_json_sha256",
        "offset_shape",
        "offsets_sha256",
        "offsets_json_sha256",
    }:
        raise ValueError("Composite compiler-binding readout is incomplete.")
    if tuple(binding_readout["coefficient_shape"]) != coefficient_shape or str(
        binding_readout["coefficients_sha256"]
    ) != _array_hash(coefficients):
        raise ValueError("Composite coefficient identity changed.")
    if tuple(binding_readout["offset_shape"]) != offset_shape or str(
        binding_readout["offsets_sha256"]
    ) != _array_hash(offsets):
        raise ValueError("Composite offset identity changed.")
    if str(binding_readout["coefficients_json_sha256"]) != hashlib.sha256(
        _canonical_json_bytes(readout["coefficients"])
    ).hexdigest() or str(binding_readout["offsets_json_sha256"]) != hashlib.sha256(
        _canonical_json_bytes(readout["offsets"])
    ).hexdigest():
        raise ValueError("Composite JSON readout identity changed.")

    default_realization = str(payload["default_realization"])
    selected_realization = (
        default_realization if realization is None else str(realization).strip().lower()
    )
    if selected_realization not in {"canonical", "factored"}:
        raise ValueError("Composite lifted realization is unsupported.")
    selected_source_realization = (
        str(payload["default_source_realization"])
        if source_realization is None
        else str(source_realization).strip().lower()
    )
    if selected_source_realization not in {"auto", "direct", "factorized"}:
        raise ValueError("Composite lifted source realization is unsupported.")
    model = _LiftedCauchyLinearModel(
        None,
        payload["source"],
        payload["type_map"],
        central_species_order=payload["central_species_order"],
        coefficients=coefficients,
        offsets=offsets,
        realization=selected_realization,
        source_realization=selected_source_realization,
        compiled_components=tuple(compiled_payloads),
        component_records=tuple(component_records),
        expected_artifact_hash=binding["composite_artifact_hash"],
    )
    if tuple(model.evaluator.component_channel_maps) != tuple(component_maps):
        raise ValueError("Composite component channel registry changed.")
    source_registry = tuple(
        {
            "channel_index": int(channel["channel_index"]),
            "neighbor_species": str(channel["neighbor_species"]),
            "source_family_id": str(channel["source_family_id"]),
            "l": int(channel["l"]),
            "radial_channel": int(channel["radial_channel"]),
        }
        for channel in model.source.channels
    )
    if list(source_registry) != list(binding["source_registry"]) or str(
        binding["source_registry_sha256"]
    ) != _payload_hash({"channels": source_registry}):
        raise ValueError("Composite source registry changed.")
    if str(model.source.source_plan_hash) != str(payload["source_plan_hash"]) or str(
        binding["source_plan_hash"]
    ) != str(payload["source_plan_hash"]):
        raise ValueError("Composite source-plan identity changed.")

    native_reference = dict(payload["native_runtime_reference"])
    if set(native_reference) != {
        "file",
        "format",
        "file_sha256",
        "plan_self_hash",
    } or native_reference["format"] != LIFTED_CAUCHY_COMPOSITE_NATIVE_SCHEMA:
        raise ValueError("Composite native-runtime reference is incomplete.")
    native_path = _bundle_member(root, native_reference["file"])
    if entries.get(native_reference["file"]) != native_reference["file_sha256"]:
        raise ValueError("Composite native-runtime manifest identity mismatch.")
    if _file_hash(native_path) != native_reference["file_sha256"]:
        raise ValueError("Composite native-runtime file SHA-256 mismatch.")
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if str(native.get("self_hash", "")) != str(
        native_reference["plan_self_hash"]
    ) or str(native.get("self_hash", "")) != _payload_hash(native):
        raise ValueError("Composite native-runtime self-hash mismatch.")
    if str(native.get("compiler_binding_self_hash", "")) != str(
        binding["self_hash"]
    ) or str(native.get("composite_artifact_hash", "")) != str(
        binding["composite_artifact_hash"]
    ):
        raise ValueError("Composite native/compiler identity mismatch.")
    deployment_identity = {
        "compiler_binding_self_hash": str(binding["self_hash"]),
        "composite_artifact_hash": str(binding["composite_artifact_hash"]),
        "source_plan_hash": str(payload["source_plan_hash"]),
        "type_map": payload["type_map"],
        "central_species_order": list(payload["central_species_order"]),
        "readout_coefficients": readout["coefficients"],
        "readout_offsets": readout["offsets"],
        "conventions": payload["conventions"],
        "ordinary_reference": None,
    }
    if str(native.get("deployment_identity_sha256", "")) != _payload_hash(
        deployment_identity
    ):
        raise ValueError("Composite native deployment identity mismatch.")
    model.fit_metadata = dict(payload["fit_metadata"])
    model._ye3t_linear_fit_metadata = dict(model.fit_metadata)
    return {
        "lifted_model": model,
        "ordinary_yace_path": None,
        "native_runtime_path": native_path,
        "native_runtime_payload": native,
        "compiler_binding_payload": binding,
        "model_payload": payload,
        "manifest": entries,
        "bundle_root": root,
    }


def load_lifted_cauchy_linear_bundle(
    path, realization=None, source_realization=None
):
    """
    Purpose:
        Validate and load a portable lifted-Cauchy inference bundle.
    Mathematical contract:
        Reconstruct the exact source/readout against the hash-bound compiler
        artifact and preserve any ordinary `.yace` reference explicitly.
    Inputs:
        A bundle directory or its ``model.ye3t.json`` file and optional
        descriptor/source execution overrides.
    Outputs:
        A mapping containing ``lifted_model``, ``ordinary_yace_path``, and the
        validated manifest payload.
    Does not:
        Silently omit or evaluate an additive ordinary ACE component, compile
        new coupling paths, authenticate the author, or fit model parameters.
    """

    from ye3t_methods.atomistic.lifted_cauchy_linear import _LiftedCauchyLinearModel

    supplied = Path(path)
    root = supplied if supplied.is_dir() else supplied.parent
    model_path = (
        root / LIFTED_CAUCHY_MODEL_FILE
        if supplied.is_dir()
        else supplied
    )
    if model_path.name != LIFTED_CAUCHY_MODEL_FILE:
        raise ValueError(
            f"Lifted-Cauchy model filename must be {LIFTED_CAUCHY_MODEL_FILE!r}."
        )
    manifest_path = root / LIFTED_CAUCHY_MANIFEST_FILE
    entries = _read_manifest(manifest_path)
    if LIFTED_CAUCHY_MODEL_FILE not in entries:
        raise ValueError("Lifted bundle manifest does not contain the model JSON.")
    if _file_hash(model_path) != entries[LIFTED_CAUCHY_MODEL_FILE]:
        raise ValueError("Lifted bundle model JSON SHA-256 mismatch.")

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {
        LIFTED_CAUCHY_BUNDLE_SCHEMA_V1,
        LIFTED_CAUCHY_BUNDLE_SCHEMA_V2,
        LIFTED_CAUCHY_BUNDLE_SCHEMA,
        LIFTED_CAUCHY_COMPOSITE_BUNDLE_SCHEMA,
    }:
        raise ValueError("Unsupported lifted-Cauchy bundle schema.")
    if schema == LIFTED_CAUCHY_COMPOSITE_BUNDLE_SCHEMA:
        return _load_composite_lifted_cauchy_bundle(
            root,
            payload,
            entries,
            realization,
            source_realization,
        )
    allowed = {
        "schema",
        "model_family",
        "compiler_artifact",
        "source",
        "type_map",
        "central_species_order",
        "readout",
        "default_realization",
        "conventions",
        "fit_metadata",
        "ordinary_reference",
        "self_hash",
    }
    if schema in {LIFTED_CAUCHY_BUNDLE_SCHEMA_V2, LIFTED_CAUCHY_BUNDLE_SCHEMA}:
        allowed.update(
            {
                "source_plan_hash",
                "default_source_realization",
            }
        )
    if schema == LIFTED_CAUCHY_BUNDLE_SCHEMA:
        allowed.add("native_runtime_reference")
    extras = sorted(set(payload) - allowed)
    if extras:
        raise ValueError(f"Unsupported lifted bundle model fields: {extras}")
    if payload.get("model_family") not in {
        "linear_lifted_cauchy_scalar",
        "linear_ordinary_plus_lifted_cauchy_scalar",
    }:
        raise ValueError("Unsupported lifted-Cauchy model family.")
    if str(payload.get("self_hash", "")) != _payload_hash(payload):
        raise ValueError("Lifted-Cauchy model self-hash mismatch.")
    if payload.get("conventions") != _CONVENTIONS:
        raise ValueError("Lifted-Cauchy bundle conventions are unsupported.")

    compiled_reference = dict(payload.get("compiler_artifact", {}))
    if set(compiled_reference) != {
        "file",
        "file_sha256",
        "artifact_self_hash",
    }:
        raise ValueError("Lifted bundle compiler reference is incomplete.")
    compiled_name = str(compiled_reference["file"])
    compiled_path = _bundle_member(root, compiled_name)
    expected_files = {LIFTED_CAUCHY_MODEL_FILE, compiled_name}
    if entries.get(compiled_name) != compiled_reference["file_sha256"]:
        raise ValueError("Lifted bundle compiler manifest identity mismatch.")
    if _file_hash(compiled_path) != compiled_reference["file_sha256"]:
        raise ValueError("Lifted bundle compiler artifact SHA-256 mismatch.")
    compiled_payload = json.loads(compiled_path.read_text(encoding="utf-8"))
    if str(compiled_payload.get("self_hash", "")) != str(
        compiled_reference["artifact_self_hash"]
    ):
        raise ValueError("Lifted bundle compiler self-hash identity mismatch.")

    native_reference = payload.get("native_runtime_reference")
    native_path = None
    native_payload = None
    if native_reference is not None:
        if schema != LIFTED_CAUCHY_BUNDLE_SCHEMA:
            raise ValueError("Legacy lifted bundle cannot reference a native runtime plan.")
        native_reference = dict(native_reference)
        if set(native_reference) != {
            "file",
            "format",
            "file_sha256",
            "plan_self_hash",
        }:
            raise ValueError("Lifted bundle native-runtime reference is incomplete.")
        if native_reference["format"] not in {
            LIFTED_CAUCHY_NATIVE_SCHEMA_V1,
            LIFTED_CAUCHY_NATIVE_SCHEMA,
        }:
            raise ValueError("Lifted bundle native-runtime format is unsupported.")
        native_name = str(native_reference["file"])
        native_path = _bundle_member(root, native_name)
        expected_files.add(native_name)
        if entries.get(native_name) != native_reference["file_sha256"]:
            raise ValueError("Lifted bundle native-runtime manifest identity mismatch.")
        if _file_hash(native_path) != native_reference["file_sha256"]:
            raise ValueError("Lifted bundle native-runtime file SHA-256 mismatch.")
        native_payload = json.loads(native_path.read_text(encoding="utf-8"))
        if native_payload.get("schema") != native_reference["format"]:
            raise ValueError("Lifted bundle native-runtime schema is unsupported.")
        if str(native_payload.get("self_hash", "")) != str(
            native_reference["plan_self_hash"]
        ) or str(native_payload.get("self_hash", "")) != _payload_hash(
            native_payload
        ):
            raise ValueError("Lifted bundle native-runtime self-hash mismatch.")
        if str(native_payload.get("compiler_artifact_self_hash", "")) != str(
            compiled_reference["artifact_self_hash"]
        ):
            raise ValueError("Lifted bundle native/compiler identity mismatch.")
        if str(native_payload.get("source_plan_hash", "")) != str(
            payload.get("source_plan_hash", "")
        ):
            raise ValueError("Lifted bundle native/source identity mismatch.")
        if list(native_payload.get("central_species_order", ())) != list(
            payload.get("central_species_order", ())
        ):
            raise ValueError("Lifted bundle native species ordering mismatch.")
        deployment_identity = {
            "compiler_artifact_self_hash": str(
                compiled_reference["artifact_self_hash"]
            ),
            "source_plan_hash": str(payload.get("source_plan_hash", "")),
            "type_map": payload.get("type_map"),
            "central_species_order": list(payload.get("central_species_order", ())),
            "readout_coefficients": payload.get("readout", {}).get(
                "coefficients"
            ),
            "readout_offsets": payload.get("readout", {}).get("offsets"),
            "conventions": payload.get("conventions"),
            "ordinary_reference": payload.get("ordinary_reference"),
        }
        if str(native_payload.get("deployment_identity_sha256", "")) != str(
            _payload_hash(deployment_identity)
        ):
            raise ValueError("Lifted bundle native deployment identity mismatch.")

    ordinary_reference = payload.get("ordinary_reference")
    ordinary_path = None
    if ordinary_reference is not None:
        ordinary_reference = dict(ordinary_reference)
        if set(ordinary_reference) != {
            "file",
            "format",
            "sha256",
            "combination",
        }:
            raise ValueError("Lifted bundle ordinary reference is incomplete.")
        if ordinary_reference["format"] != "lammps_pace_linear_v1":
            raise ValueError("Lifted bundle ordinary reference format is unsupported.")
        if ordinary_reference["combination"] != "additive_energy_force_virial":
            raise ValueError("Lifted bundle ordinary combination rule is unsupported.")
        ordinary_name = str(ordinary_reference["file"])
        ordinary_path = _bundle_member(root, ordinary_name)
        expected_files.add(ordinary_name)
        if entries.get(ordinary_name) != ordinary_reference["sha256"]:
            raise ValueError("Lifted bundle ordinary manifest identity mismatch.")
        if _file_hash(ordinary_path) != ordinary_reference["sha256"]:
            raise ValueError("Lifted bundle ordinary `.yace` SHA-256 mismatch.")
    if set(entries) != expected_files:
        raise ValueError("Lifted bundle manifest contains missing or unexpected files.")
    if (
        ordinary_reference is None
        and payload["model_family"] != "linear_lifted_cauchy_scalar"
    ) or (
        ordinary_reference is not None
        and payload["model_family"]
        != "linear_ordinary_plus_lifted_cauchy_scalar"
    ):
        raise ValueError("Lifted bundle model family and ordinary reference disagree.")

    readout = dict(payload.get("readout", {}))
    if readout.get("dtype") != "float64":
        raise ValueError("Lifted-Cauchy bundle readout must use float64.")
    coefficient_shape = tuple(int(value) for value in readout.get("coefficient_shape", ()))
    offset_shape = tuple(int(value) for value in readout.get("offset_shape", ()))
    coefficients = _finite_matrix(
        readout.get("coefficients", ()), coefficient_shape, "lifted coefficients"
    )
    offsets = _finite_matrix(readout.get("offsets", ()), offset_shape, "lifted offsets")
    default_realization = str(payload.get("default_realization", ""))
    selected_realization = (
        default_realization if realization is None else str(realization).strip().lower()
    )
    if selected_realization not in {"canonical", "factored"}:
        raise ValueError("Lifted-Cauchy realization must be canonical or factored.")
    selected_source_realization = (
        str(payload.get("default_source_realization", "auto"))
        if source_realization is None
        else str(source_realization).strip().lower()
    )
    if selected_source_realization not in {"auto", "direct", "factorized"}:
        raise ValueError("Lifted-Cauchy source realization is unsupported.")
    model = _LiftedCauchyLinearModel(
        compiled_payload,
        payload["source"],
        payload["type_map"],
        central_species_order=payload["central_species_order"],
        coefficients=coefficients,
        offsets=offsets,
        realization=selected_realization,
        source_realization=selected_source_realization,
    )
    if schema in {
        LIFTED_CAUCHY_BUNDLE_SCHEMA_V2,
        LIFTED_CAUCHY_BUNDLE_SCHEMA,
    } and str(payload.get("source_plan_hash", "")) != str(
        model.source.source_plan_hash
    ):
        raise ValueError("Lifted bundle source-plan identity mismatch.")
    model.fit_metadata = dict(payload.get("fit_metadata", {}))
    model._ye3t_linear_fit_metadata = dict(model.fit_metadata)
    return {
        "lifted_model": model,
        "ordinary_yace_path": ordinary_path,
        "native_runtime_path": native_path,
        "native_runtime_payload": native_payload,
        "model_payload": payload,
        "manifest": entries,
        "bundle_root": root,
    }


__all__ = [
    "LIFTED_CAUCHY_BUNDLE_SCHEMA",
    "LIFTED_CAUCHY_BUNDLE_SCHEMA_V1",
    "LIFTED_CAUCHY_BUNDLE_SCHEMA_V2",
    "LIFTED_CAUCHY_COMPOSITE_BUNDLE_SCHEMA",
    "LIFTED_CAUCHY_COMPOSITE_NATIVE_SCHEMA",
    "LIFTED_CAUCHY_COMPILER_FILE",
    "LIFTED_CAUCHY_COMPILER_BINDING_FILE",
    "LIFTED_CAUCHY_COMPILER_BINDING_SCHEMA",
    "LIFTED_CAUCHY_MANIFEST_FILE",
    "LIFTED_CAUCHY_MODEL_FILE",
    "LIFTED_CAUCHY_NATIVE_FILE",
    "LIFTED_CAUCHY_NATIVE_SCHEMA",
    "LIFTED_CAUCHY_NATIVE_SCHEMA_V1",
    "LIFTED_CAUCHY_ORDINARY_FILE",
    "export_lifted_cauchy_linear_bundle",
    "load_lifted_cauchy_linear_bundle",
]
