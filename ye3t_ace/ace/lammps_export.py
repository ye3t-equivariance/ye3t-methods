"""Export fitted ordinary scalar ACE models for the YE3T LAMMPS runtime."""

import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

from ye3t.couplings import (
    compile_execution_plan,
    compile_scalar_ace_coordinate,
    count as count_couplings,
    execution_plan_from_repeated_angular_blocks,
    normalize_compact_label,
)
from ye3t.execution_plan import (
    YE3TExecutionPlan,
    YE3TExecutionPlanWiring,
    YE3TPackedCarrierSlice,
    YE3TSourceRealization,
)

from ye3t_ace.ace.linear_ace import (
    _bundle_to_yace_functions,
    _validate_bundle_ordinary_scalar_catalogue,
    export_scalar_bundle_to_yace,
)
from ye3t_ace.ace.yace import read_yace


_ENCODING = "ye3t_sorted_json_indent2_lf_v1"
_PLAN_SCHEMA = "ye3t_execution_plan_v2"
_MAP_SCHEMA = "ye3t_yace_function_map_v3"
_SIDECAR_SCHEMA = "ye3t_lammps_sidecar_v3"
_READOUT_SCHEMA = "ye3t_yace_candidate_readout_v1"
_READOUTS_SCHEMA = "ye3t_yace_candidate_readouts_v1"
_OUTPUT_BINDING_SCHEMA = "ye3t_yace_plan_output_binding_v1"
_OUTPUT_BINDINGS_SCHEMA = "ye3t_yace_plan_output_bindings_v1"
_CATALOGUE_SOURCE_SCHEMAS = {
    "ye3t_ordinary_scalar_catalogue_source_v1",
    "ye3t_task55_scalar_catalogue_source_v1",
}
_MANUAL_CATALOGUE_SOURCE_SCHEMA = "ye3t_ordinary_scalar_manual_labels_v1"
_EXPECTED_CERTIFICATE_FIELDS = (
    "certificate_sha256",
    "coordinate_identity_sha256",
    "collected_coefficient_sha256",
    "factorized_schedule_sha256",
    "blockwise_plan_convention_hash",
)


def _canonical_bytes(payload):
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _stable_hash(payload):
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_scalar_coordinate(compilation):
    table = compilation["coefficient_table"]
    magnetic_tuples, coefficients = table.component_terms(0)
    payload = {
        "schema": "ye3t_scalar_ace_coordinate_table_v1",
        "label": compilation["label"].to_dict(),
        "magnetic_tuples": magnetic_tuples.tolist(),
        "coefficients": [
            [float(complex(value).real), float(complex(value).imag)]
            for value in coefficients.tolist()
        ],
        "certificate": json.loads(
            json.dumps(compilation["certificate"], allow_nan=False)
        ),
    }
    payload["payload_sha256"] = _stable_hash(payload)
    return payload


def _selected_scalar_catalogue(source, profile_id):
    schema = source.get("schema")
    if schema == _MANUAL_CATALOGUE_SOURCE_SCHEMA:
        catalogue_id = str(source.get("catalogue_id", "manual_catalogue")).strip()
        if not catalogue_id:
            raise ValueError("A manual scalar catalogue requires catalogue_id.")
        if profile_id is not None and str(profile_id) != catalogue_id:
            raise ValueError("Manual catalogue_id differs from the requested identity.")
        raw_rows = tuple(source.get("manual_labels", ()))
        if not raw_rows:
            raise ValueError("A manual scalar catalogue requires manual_labels.")
        rows = []
        feature_ids = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                raise TypeError("manual_labels entries must be mappings.")
            if "compact_label" in raw_row:
                row = dict(raw_row)
                label = normalize_compact_label(row["compact_label"])
            else:
                row = {"compact_label": dict(raw_row)}
                label = normalize_compact_label(raw_row)
            feature_id = str(row.get("feature_id", "")).strip()
            if not feature_id:
                feature_id = (
                    f"ace_r{int(label.rank):02d}_"
                    f"{_stable_hash(label.to_dict())[:20]}"
                )
            row["feature_id"] = feature_id
            row["compact_label"] = label.to_dict()
            rows.append(row)
            feature_ids.append(feature_id)
        selection = {
            "application_schema": str(
                source.get("application_schema", "ye3t_ordinary_scalar_catalogue_v2")
            ),
            "compiler": dict(source.get("compiler", {})),
            "feature_ids": feature_ids,
        }
        return catalogue_id, selection, tuple(rows)
    if schema not in _CATALOGUE_SOURCE_SCHEMAS:
        raise ValueError("Unsupported scalar catalogue source schema.")
    profiles = source.get("profiles", {})
    if profile_id not in profiles:
        raise KeyError(f"Unknown catalogue profile: {profile_id}")
    selection = dict(profiles[profile_id])
    rows_source = tuple(source.get("rows", ()))
    row_by_id = {str(row["feature_id"]): dict(row) for row in rows_source}
    if len(row_by_id) != len(rows_source):
        raise ValueError("Catalogue source contains duplicate feature IDs.")
    feature_ids = [str(value) for value in selection["feature_ids"]]
    if any(feature_id not in row_by_id for feature_id in feature_ids):
        raise ValueError("Catalogue profile refers to an unknown feature ID.")
    return (
        str(profile_id),
        selection,
        tuple(row_by_id[feature_id] for feature_id in feature_ids),
    )


def compile_ordinary_scalar_catalogue(source, profile_id=None, progress=None):
    """Compile one ordered scalar catalogue with YE3T-owned coordinates.

    Purpose:
        Turn an editable manual-label catalogue or legacy source/profile into the validated
        application consumed by ordinary ACE fitting and native export.
    Mathematical contract:
        Every row is compiled by ``ye3t.couplings.compile_scalar_ace_coordinate``
        and retains its certificate and declared ordering.
    Inputs:
        A catalogue source mapping, optional legacy profile identifier, and progress
        callback accepting ``(index, count, feature_id)``.
    Outputs:
        A deterministic ordinary-scalar catalogue application mapping.
    Does not:
        Enumerate labels, choose a fit, or perform any LAMMPS runtime work.
    """
    source = dict(source)
    catalogue_id, selection, rows_source = _selected_scalar_catalogue(
        source,
        profile_id,
    )
    application_schema = str(
        selection.get("application_schema", "ye3t_ordinary_scalar_catalogue_v2")
    )
    if application_schema not in {
        "ye3t_ordinary_scalar_catalogue_v1",
        "ye3t_ordinary_scalar_catalogue_v2",
    }:
        raise ValueError("Unsupported scalar catalogue application schema.")
    feature_ids = [str(row["feature_id"]) for row in rows_source]
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("Catalogue source contains duplicate feature IDs.")
    compiler = dict(source.get("compiler", {}))
    compiler.update(selection.get("compiler", {}))
    rows = []
    multiplicity_reports = {}
    normalized_labels = []
    for index, source_row in enumerate(rows_source, start=1):
        feature_id = str(source_row["feature_id"])
        if progress is not None:
            progress(index, len(feature_ids), feature_id)
        row_compiler = dict(compiler)
        row_compiler.update(source_row.get("compiler", {}))
        if source_row.get("compiler") and application_schema.endswith("_v1"):
            raise ValueError(
                "Per-coordinate compiler options require catalogue application v2."
            )
        label = normalize_compact_label(source_row["compact_label"])
        if label in normalized_labels:
            raise ValueError("Catalogue source contains duplicate compact labels.")
        normalized_labels.append(label)
        multiplicity_report = None
        if int(label.rank) <= 8:
            membership_key = (
                tuple(label.n_tuple),
                tuple(label.l_tuple),
                int(label.L_R),
                str(label.tree_type),
            )
            multiplicity_report = multiplicity_reports.get(membership_key)
            if multiplicity_report is None:
                multiplicity_report = count_couplings(
                    content=tuple(label.n_tuple),
                    input_Ls=tuple(label.l_tuple),
                    target_L=int(label.L_R),
                    target_permutation="trivial",
                    carrier="ACE_density",
                    tree_schedule=str(label.tree_type),
                )
                multiplicity_reports[membership_key] = multiplicity_report
        compilation = compile_scalar_ace_coordinate(
            label,
            multiplicity_report=multiplicity_report,
            **row_compiler,
        )
        certificate = compilation["certificate"]
        if certificate.get("passed") is not True:
            raise RuntimeError(f"Compiler certificate failed for {feature_id}.")
        row = {
            "feature_id": feature_id,
            "compact_label": compilation["label"].to_dict(),
            "expected_compiler": {
                key: str(certificate[key]) for key in _EXPECTED_CERTIFICATE_FIELDS
            },
        }
        if application_schema.endswith("_v2"):
            row["compiler"] = row_compiler
            row["compiled_coordinate"] = _serialize_scalar_coordinate(compilation)
        rows.append(row)
    membership = {"profile_id": catalogue_id, "feature_ids": feature_ids}
    application = {
        "schema": application_schema,
        "profile_id": catalogue_id,
        "feature_ids": feature_ids,
        "rows": rows,
        "compiler": compiler,
    }
    return {
        **application,
        "membership_sha256": _stable_hash(membership),
        "application_sha256": _stable_hash(application),
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_field(payload, value):
    encoded = str(value).encode("utf-8")
    payload.extend(str(len(encoded)).encode("ascii"))
    payload.extend(b":")
    payload.extend(encoded)


def _identity_integer(payload, value):
    _identity_field(payload, int(value))


def _identity_double(payload, value):
    bits = int.from_bytes(struct.pack("<d", float(value)), "little")
    _identity_field(payload, bits)


def _identity_scale(payload, value):
    _identity_double(payload, value[0])
    _identity_double(payload, value[1])


def _binding_id(term):
    payload = bytearray()
    _identity_field(payload, "ye3t_candidate_binding_v1")
    _identity_field(payload, term["instruction_id"])
    _identity_integer(payload, term["channel_index"])
    _identity_integer(payload, term["tableau_index"])
    _identity_integer(payload, term["magnetic_index"])
    _identity_scale(payload, term["scale"])
    return hashlib.sha256(bytes(payload)).hexdigest()


def _readout_id(record):
    equivalence = record["equivalence"]
    payload = bytearray()
    _identity_field(payload, "ye3t_candidate_readout_identity_v1")
    _identity_field(payload, record["schema"])
    _identity_integer(payload, record["central_type"])
    _identity_integer(payload, record["function_index"])
    _identity_field(payload, record["feature_id"])
    _identity_field(payload, record["source_yace_sha256"])
    _identity_field(payload, record["variable_order_hash"])
    _identity_field(payload, equivalence["method"])
    _identity_field(payload, equivalence["derivative_rule"])
    _identity_integer(payload, equivalence["passed"])
    _identity_integer(payload, equivalence["support_equal"])
    _identity_integer(
        payload,
        equivalence["adjoint_certified_by_coefficient_identity"],
    )
    _identity_double(payload, equivalence["absolute_tolerance"])
    _identity_double(payload, equivalence["relative_tolerance"])
    _identity_double(
        payload,
        equivalence["maximum_absolute_coefficient_residual"],
    )
    _identity_double(payload, equivalence["relative_l2_residual"])
    _identity_field(payload, equivalence["tolerance_rule"])
    _identity_double(payload, equivalence["maximum_mixed_tolerance_ratio"])
    _identity_double(
        payload,
        equivalence["maximum_reference_coefficient_magnitude"],
    )
    _identity_field(payload, equivalence["plan_polynomial_sha256"])
    _identity_field(payload, equivalence["yace_polynomial_sha256"])
    _identity_integer(payload, len(record["terms"]))
    for term in record["terms"]:
        _identity_field(payload, term["binding_id"])
        _identity_field(payload, term["instruction_id"])
        _identity_integer(payload, term["channel_index"])
        _identity_integer(payload, term["tableau_index"])
        _identity_integer(payload, term["magnetic_index"])
        _identity_scale(payload, term["scale"])
    return hashlib.sha256(bytes(payload)).hexdigest()


def _candidate_id(candidate):
    payload = bytearray()
    _identity_field(payload, "ye3t_evaluator_candidate_identity_v1")
    _identity_field(payload, candidate["evaluator"])
    _identity_field(payload, candidate["availability"]["status"])
    _identity_field(payload, candidate["availability"]["reason"])
    capabilities = sorted(candidate["required_capabilities"])
    _identity_integer(payload, len(capabilities))
    for capability in capabilities:
        _identity_field(payload, capability)
    if candidate["evaluator"] == "explicit_ctilde":
        _identity_scale(payload, candidate["scale"])
        source = candidate["source_binding"]
        _identity_integer(payload, source["central_type"])
        _identity_integer(payload, source["function_index"])
        _identity_field(payload, source["feature_id"])
        _identity_field(payload, source["source_yace_sha256"])
    elif candidate["evaluator"] == "execution_plan_readout":
        _identity_field(payload, candidate["compiler_plan_hash"])
        _identity_field(payload, candidate["readout_id"])
    else:
        raise ValueError("Unsupported YE3T LAMMPS evaluator candidate.")
    return hashlib.sha256(bytes(payload)).hexdigest()


def _with_candidate_id(candidate):
    return {"alternative_id": _candidate_id(candidate), **candidate}


def _feature_payload(central_type, function_index, function):
    return {
        "central_type": int(central_type),
        "ctildes": [float(value) for value in function.ctildes],
        "function_index": int(function_index),
        "ls": [int(value) for value in function.ls],
        "ms_combs": [int(value) for value in function.ms_combs],
        "mu0": int(function.mu0),
        "mus": [int(value) for value in function.mus],
        "ndensity": int(function.ndensity),
        "ns": [int(value) for value in function.ns],
        "rank": int(function.rank),
    }


def _complex_value(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Complex table values must be real/imaginary pairs.")
        return complex(float(value[0]), float(value[1]))
    return complex(value)


def _add_polynomial(target, source, scale=1.0 + 0.0j):
    for exponents, coefficient in source.items():
        value = target.get(exponents, 0.0 + 0.0j) + scale * coefficient
        if value == 0.0:
            target.pop(exponents, None)
        else:
            target[exponents] = value


def _product_polynomial(left, right):
    result = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            exponents = tuple(left_exponents) + tuple(right_exponents)
            result[exponents] = (
                result.get(exponents, 0.0 + 0.0j)
                + left_coefficient * right_coefficient
            )
    return {key: value for key, value in result.items() if value != 0.0}


def _sparse_table_value(table, row, column):
    result = 0.0 + 0.0j
    for table_row, table_column, value in zip(
        table["row_indices"],
        table["column_indices"],
        table["values"],
    ):
        if int(table_row) == int(row) and int(table_column) == int(column):
            result += _complex_value(value)
    return result


def _block_component_polynomial(
    metadata,
    block_index,
    output_L,
    copy_index,
    component,
):
    records = tuple(
        record
        for record in metadata["block_power_plans"]
        if int(record["block_index"]) == int(block_index)
        and int(record["output_L"]) == int(output_L)
    )
    if len(records) != 1:
        raise RuntimeError("A block route does not identify one power plan.")
    entries = tuple(
        entry
        for entry in records[0]["plan"]["entries"]
        if int(entry["multiplicity_index"]) == int(copy_index)
        and int(entry["component_index"]) == int(component)
    )
    if len(entries) != 1:
        raise RuntimeError("A block route does not identify one output component.")
    result = {}
    for term in entries[0]["component_terms"]:
        exponents = tuple(int(value) for value in term["exponents"])
        result[exponents] = result.get(exponents, 0.0 + 0.0j) + _complex_value(
            term["coefficient"]
        )
    return {key: value for key, value in result.items() if value != 0.0}


def _plan_polynomial(plan, route_index=0):
    payload = plan.to_dict()
    if len(payload["instructions"]) != 1:
        raise RuntimeError("A fitted block candidate must contain one instruction.")
    instruction = payload["instructions"][0]
    metadata = instruction["metadata"]
    routes = tuple(
        route
        for route in metadata["routes"]
        if int(route["route_index"]) == int(route_index)
    )
    if len(routes) != 1:
        raise RuntimeError("The selected block route is not unique.")
    route = routes[0]
    output_Ls = tuple(int(value) for value in route["block_output_Ls"])
    copies = tuple(int(value) for value in route["block_multiplicity_indices"])
    tables = {table["table_id"]: table for table in payload["synthesis_tables"]}
    lr = _sparse_table_value(tables[metadata["lr_synthesis_table_id"]], 0, 0)
    widths = tuple(2 * value + 1 for value in output_Ls)
    angular_id = route.get("angular_synthesis_table_id")
    if angular_id is None:
        if len(widths) != 1 or widths[0] != 1:
            raise RuntimeError("An uncoupled block route is not scalar.")
        block = _block_component_polynomial(
            metadata,
            0,
            output_Ls[0],
            copies[0],
            0,
        )
        return {
            exponents: coefficient * lr.conjugate()
            for exponents, coefficient in block.items()
        }
    result = {}
    for row, column, value in zip(
        tables[angular_id]["row_indices"],
        tables[angular_id]["column_indices"],
        tables[angular_id]["values"],
    ):
        if int(column) != 0:
            continue
        row = int(row)
        components = [0] * len(widths)
        for block_index in range(len(widths) - 1, -1, -1):
            components[block_index] = row % widths[block_index]
            row //= widths[block_index]
        if row:
            raise RuntimeError("An outer angular row exceeds its product carrier.")
        polynomial = {(): 1.0 + 0.0j}
        for block_index, component in enumerate(components):
            block = _block_component_polynomial(
                metadata,
                block_index,
                output_Ls[block_index],
                copies[block_index],
                component,
            )
            polynomial = _product_polynomial(polynomial, block)
        _add_polynomial(
            result,
            polynomial,
            lr.conjugate() * _complex_value(value).conjugate(),
        )
    if not result:
        raise RuntimeError("Block analysis produced an empty scalar polynomial.")
    return result


def _yace_polynomial(function, blocks):
    offsets = []
    cursor = 0
    for block in blocks:
        stop = cursor + int(block["power"])
        offsets.append((cursor, stop, int(block["input_L"])))
        cursor = stop
    if cursor != int(function.rank):
        raise RuntimeError("Block powers do not cover the YACE function rank.")
    result = {}
    for row, coefficient in enumerate(function.ctildes):
        exponents = []
        magnetic_offset = row * int(function.rank)
        for begin, end, angular in offsets:
            local = [0] * (2 * angular + 1)
            for factor in range(begin, end):
                magnetic = int(function.ms_combs[magnetic_offset + factor])
                local[magnetic + angular] += 1
            exponents.extend(local)
        key = tuple(exponents)
        result[key] = result.get(key, 0.0 + 0.0j) + complex(float(coefficient))
    return {key: value for key, value in result.items() if value != 0.0}


def _polynomial_payload(polynomial):
    return [
        {
            "coefficient": [float(value.real), float(value.imag)],
            "exponents": list(exponents),
        }
        for exponents, value in sorted(polynomial.items())
    ]


def _certify_readout(function, blocks, polynomial, absolute_tolerance, relative_tolerance):
    expected = _yace_polynomial(function, blocks)
    if not expected:
        return None, None
    pivots = tuple(
        exponents
        for exponents in expected
        if polynomial.get(exponents, 0.0 + 0.0j) != 0.0
    )
    if not pivots:
        raise RuntimeError("The block candidate has no nonzero calibration pivot.")
    pivot = min(pivots, key=lambda item: (-abs(polynomial[item]), item))
    scale = expected[pivot] / polynomial[pivot]
    if not math.isfinite(scale.real) or not math.isfinite(scale.imag):
        raise RuntimeError("The block readout scale is non-finite.")
    actual = {key: scale * value for key, value in polynomial.items()}
    actual = {key: value for key, value in actual.items() if value != 0.0}
    support_equal = set(actual) == set(expected)
    maximum_residual = 0.0
    maximum_mixed_ratio = 0.0
    maximum_reference = 0.0
    squared_residual = 0.0
    squared_reference = 0.0
    for exponents in set(actual).union(expected):
        reference = expected.get(exponents, 0.0 + 0.0j)
        candidate = actual.get(exponents, 0.0 + 0.0j)
        residual = abs(candidate - reference)
        reference_magnitude = abs(reference)
        gate = absolute_tolerance + relative_tolerance * reference_magnitude
        maximum_residual = max(maximum_residual, residual)
        maximum_mixed_ratio = max(maximum_mixed_ratio, residual / gate)
        maximum_reference = max(maximum_reference, reference_magnitude)
        squared_residual += residual * residual
        squared_reference += reference_magnitude * reference_magnitude
    relative_residual = math.sqrt(squared_residual) / max(
        math.sqrt(squared_reference),
        absolute_tolerance,
    )
    if (
        not support_equal
        or maximum_reference <= 0.0
        or maximum_mixed_ratio > 1.0
    ):
        raise RuntimeError(
            "Block polynomial differs from the fitted YACE row: "
            f"support_equal={support_equal}, max_abs={maximum_residual!r}, "
            f"mixed_ratio={maximum_mixed_ratio!r}, rel_l2={relative_residual!r}."
        )
    equivalence = {
        "absolute_tolerance": float(absolute_tolerance),
        "adjoint_certified_by_coefficient_identity": True,
        "derivative_rule": "explicit_polynomial_product_rule",
        "maximum_absolute_coefficient_residual": float(maximum_residual),
        "maximum_mixed_tolerance_ratio": float(maximum_mixed_ratio),
        "maximum_reference_coefficient_magnitude": float(maximum_reference),
        "method": "coefficientwise_sparse_polynomial_mixed_v1",
        "passed": True,
        "plan_polynomial_sha256": _stable_hash(_polynomial_payload(actual)),
        "relative_l2_residual": float(relative_residual),
        "relative_tolerance": float(relative_tolerance),
        "support_equal": True,
        "tolerance_rule": "absolute_plus_relative_reference",
        "yace_polynomial_sha256": _stable_hash(_polynomial_payload(expected)),
    }
    return scale, equivalence


def _package_version(distribution):
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _compiler_coordinate(row, catalogue):
    compiler = dict(catalogue["compiler"])
    compiler.update(row.get("compiler", {}))
    compiled = compile_scalar_ace_coordinate(
        row["compact_label"],
        **compiler,
    )
    expected = dict(row["expected_compiler"])
    certificate = dict(compiled["certificate"])
    for name, value in expected.items():
        if str(certificate.get(name, "")) != str(value):
            raise RuntimeError(
                f"Compiler identity mismatch for {row['feature_id']!r}: {name}."
            )
    return compiled


def _compile_block_candidate(
    compiled,
    function,
    central_type,
    function_index,
    feature_id,
    maximum_plan_bytes,
):
    specs = tuple(dict(spec) for spec in compiled["factorized_schedule"].block_specs[0])
    blocks = []
    content = []
    cursor = 0
    for block_index, spec in enumerate(specs):
        power = int(spec.get("k_b", 1))
        radial = int(spec["n"])
        angular = int(spec["l"])
        stop = cursor + power
        if (
            tuple(int(value) for value in function.ns[cursor:stop])
            != (radial,) * power
            or tuple(int(value) for value in function.ls[cursor:stop])
            != (angular,) * power
        ):
            raise RuntimeError("Compiler block order differs from the fitted YACE row.")
        neighbor_types = tuple(int(value) for value in function.mus[cursor:stop])
        if len(set(neighbor_types)) != 1:
            return None, "repeated_formal_block_has_multiple_neighbor_types", None
        neighbor_type = neighbor_types[0]
        complete_key = (
            "central_type",
            int(central_type),
            "neighbor_type",
            neighbor_type,
            "radial_n",
            radial,
            "angular_l",
            angular,
            "pace_complex_magnetic_y00_1",
        )
        content.extend((complete_key,) * power)
        blocks.append(
            {
                "power": power,
                "input_L": angular,
                "slot_indices": tuple(range(cursor, stop)),
                "source_binding": {
                    "binding_id": (
                        f"{feature_id}:c{central_type}:b{block_index}:"
                        f"mu{neighbor_type}:n{radial}:l{angular}"
                    ),
                    "input_L": angular,
                    "yace": {
                        "angular_l": angular,
                        "central_type": int(central_type),
                        "factor_basis": "A",
                        "magnetic_order": "minus_l_to_plus_l",
                        "neighbor_type": neighbor_type,
                        "radial_n": radial,
                    },
                },
                "output_L": int(spec.get("Lambda", angular)),
                "multiplicity_index": int(spec.get("multiplicity_index", 0)),
            }
        )
        cursor = stop
    if cursor != int(function.rank):
        raise RuntimeError("Compiler blocks do not cover the fitted YACE row.")
    if not any(int(block["power"]) > 1 for block in blocks):
        return None, "no_repeated_complete_channel_block", None
    source = YE3TSourceRealization(
        kind="ordinary_density",
        rank=int(function.rank),
        content=tuple(content),
    )
    plan_blocks = tuple(
        {
            "power": block["power"],
            "input_L": block["input_L"],
            "slot_indices": block["slot_indices"],
            "source_binding": block["source_binding"],
        }
        for block in blocks
    )
    options = {}
    output_Ls = tuple(int(block["output_L"]) for block in blocks)
    multiplicities = tuple(int(block["multiplicity_index"]) for block in blocks)
    if len(blocks) <= 2:
        options["selected_routes"] = ((output_Ls, multiplicities),)
    else:
        options["compiler_labels"] = (compiled["label"],)
    plan_identity = f"{feature_id}_c{central_type}_f{function_index}"
    plan = execution_plan_from_repeated_angular_blocks(
        plan_blocks,
        id_prefix=plan_identity,
        parent_partition=(int(function.rank),),
        target_L=0,
        source_realization=source,
        spatial_symmetry="O3",
        expected_multiplicity=1,
        coefficient_materialization="exact",
        maximum_exact_symbolic_bytes=int(maximum_plan_bytes),
        **options,
    )
    routes = tuple(plan.instructions[0].metadata["routes"])
    if len(routes) != 1:
        raise RuntimeError("One fitted coordinate must lower to one block route.")
    return plan, None, blocks


def _output_binding(source_hash, central_type, function_index, feature_hash, instruction, scale):
    return {
        "central_type": int(central_type),
        "channel_index": 0,
        "feature_id": str(feature_hash),
        "function_index": int(function_index),
        "instruction_hash": _stable_hash(instruction.to_dict()),
        "instruction_id": str(instruction.instruction_id),
        "magnetic_index": 0,
        "scale": [float(scale.real), float(scale.imag)],
        "schema": _OUTPUT_BINDING_SCHEMA,
        "source_yace_sha256": str(source_hash),
        "tableau_index": 0,
    }


def _aggregate_plans(rows, source_hash, readouts, scope):
    layouts = []
    layout_payloads = set()
    assemblies = []
    tables = []
    instructions = []
    slices = []
    output_bindings = []
    cursor = 0
    for row in rows:
        plan = row["plan"]
        if len(plan.instructions) != 1 or len(plan.carrier_layouts) != 1:
            raise RuntimeError("Each fitted block plan must have one instruction/layout.")
        instruction = plan.instructions[0]
        layout = plan.carrier_layouts[0]
        layout_payload = _canonical_bytes(layout.to_dict())
        if layout_payload not in layout_payloads:
            layouts.append(layout)
            layout_payloads.add(layout_payload)
        assemblies.extend(plan.source_assemblies)
        tables.extend(plan.synthesis_tables)
        instructions.append(instruction)
        slice_id = (
            f"{row['catalogue_feature_id']}:c{row['central_type']}:"
            f"f{row['function_index']}:output"
        )
        slices.append(
            YE3TPackedCarrierSlice(
                slice_id=slice_id,
                buffer_id="yace_block_outputs",
                carrier_layout=layout,
                start=cursor,
                stop=cursor + layout.width,
                metadata={"feature_id": row["catalogue_feature_id"]},
            )
        )
        cursor += layout.width
        output_bindings.append((instruction.instruction_id, slice_id))
        output_bindings_record = _output_binding(
            source_hash,
            row["central_type"],
            row["function_index"],
            row["feature_hash"],
            instruction,
            row["scale"],
        )
        row["output_binding"] = output_bindings_record
    wiring = YE3TExecutionPlanWiring(
        wiring_id=scope + ":wiring",
        buffer_widths=(("yace_block_outputs", cursor),),
        packed_slices=tuple(slices),
        instruction_input_bindings=(),
        instruction_output_bindings=tuple(output_bindings),
        metadata={
            "compiler_owner": "ye3t",
            "output_write_semantics": "zero_then_accumulate",
            "source_inputs": "opaque_A_multiplet_bindings",
        },
    )
    schedule = tuple(instruction.instruction_id for instruction in instructions)
    certificate = {
        "all_paths_compiled_before_runtime": True,
        "passed": True,
        "scope": scope,
        "yace_candidate_readouts": {
            "records": readouts,
            "schema": _READOUTS_SCHEMA,
        },
        "yace_output_bindings": {
            "records": [row["output_binding"] for row in rows],
            "schema": _OUTPUT_BINDINGS_SCHEMA,
        },
    }
    return compile_execution_plan(
        carrier_layouts=tuple(layouts),
        source_assemblies=tuple(assemblies),
        synthesis_tables=tuple(tables),
        instructions=tuple(instructions),
        forward_schedule=schedule,
        reverse_schedule=tuple(reversed(schedule)),
        second_order_schedule=schedule,
        wiring=wiring,
        convention_id=rows[0]["plan"].convention_id,
        certificate=certificate,
        provenance={
            "api": "ye3t.couplings.execution_plan_from_repeated_angular_blocks",
            "aggregate": "YE3TExecutionPlanWiring",
            "compiler_owner": "ye3t",
            "runtime_path_discovery": False,
        },
    )


def _manifest(model, model_path, plan, plan_path, map_path):
    capabilities = sorted(
        {
            "block_symmetric_power_adjoint_v1",
            "block_symmetric_power_forward_v1",
            "energy_v1",
            "evaluator_candidate_vector_v1",
            "explicit_ctilde_forward_adjoint_v1",
            "force_v1",
            "linear_readout_v1",
            "ordinary_density_v1",
            "pace_cheb_exp_cos_uniform_cubic_hermite_v1",
            "pace_complex_magnetic_y00_1",
            "pace_linear_embedding_v1",
            "symmetric_power_adjoint_v1",
            "symmetric_power_forward_v1",
            "virial_v1",
            "yace_candidate_readout_v1",
            _PLAN_SCHEMA,
        }
    )
    return {
        "canonical_encoding": _ENCODING,
        "capabilities": {"required": capabilities},
        "compiler": {
            "api": "ye3t.couplings.execution_plan_from_repeated_angular_blocks",
            "coefficient_hash": plan.coefficient_hash,
            "convention_id": plan.convention_id,
            "execution_plan_schema": _PLAN_SCHEMA,
            "plan_hash": plan.plan_hash,
            "ye3t": {"version": _package_version("ye3t")},
            "ye3t_ace": {"version": _package_version("ye3t-ace")},
        },
        "conventions": {
            "analysis_orientation": "conjugate_transpose",
            "angular_source_basis": "pace_complex_magnetic_y00_1",
            "angular_transform": {"kind": "none", "payload": None},
            "atomic_base_normalization": "none",
            "byte_order": "little",
            "factor_normalization": "none",
            "linear_feature_transform": {"kind": "none", "payload": None},
            "radial_basis": "pace_cheb_exp_cos_uniform_cubic_hermite_v1",
            "scalar_type": "float64",
        },
        "dispatch": {
            "fallback_policy": "forbid_unlisted",
            "requested_route": "ye3t_evaluator_candidate_vector_v1",
        },
        "inputs": {"per_atom": []},
        "payloads": {
            "execution_plan": {
                "path": plan_path.name,
                "schema": _PLAN_SCHEMA,
                "sha256": _file_sha256(plan_path),
            },
            "yace_function_map": {
                "path": map_path.name,
                "schema": _MAP_SCHEMA,
                "sha256": _file_sha256(map_path),
            },
        },
        "schema": _SIDECAR_SCHEMA,
        "semantic_ledger": "task55_semantic_ledger_v1",
        "source_yace": {
            "compatibility_profile": "lammps_pace_linear_v1",
            "lammps_units": "metal",
            "ordered_elements": list(model["elements"]),
            "sha256": _file_sha256(model_path),
        },
    }


def _validate_emitted_bundle(root):
    root = Path(root)
    model_path = root / "model.yace"
    plan_path = root / "execution_plan.json"
    map_path = root / "yace_function_map.json"
    manifest_path = root / "manifest.json"
    model = read_yace(model_path, compatibility="lammps_pace_linear_v1")
    plan = YE3TExecutionPlan.from_json(plan_path.read_text(encoding="utf-8"))
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _canonical_bytes(plan.to_dict()) != plan_path.read_bytes():
        raise RuntimeError("The emitted execution plan is not canonical.")
    if _canonical_bytes(mapping) != map_path.read_bytes():
        raise RuntimeError("The emitted function map is not canonical.")
    if _canonical_bytes(manifest) != manifest_path.read_bytes():
        raise RuntimeError("The emitted sidecar manifest is not canonical.")
    source_hash = _file_sha256(model_path)
    if (
        manifest["schema"] != _SIDECAR_SCHEMA
        or manifest["compiler"]["plan_hash"] != plan.plan_hash
        or manifest["compiler"]["coefficient_hash"] != plan.coefficient_hash
        or manifest["payloads"]["execution_plan"]["sha256"]
        != _file_sha256(plan_path)
        or manifest["payloads"]["yace_function_map"]["sha256"]
        != _file_sha256(map_path)
        or mapping["source_yace_sha256"] != source_hash
        or mapping["plan_hash"] != plan.plan_hash
    ):
        raise RuntimeError("The emitted LAMMPS bundle hashes are inconsistent.")
    functions = [
        (central, index, function)
        for central, bucket in sorted(model["functions"].items())
        for index, function in enumerate(bucket)
    ]
    if len(mapping["entries"]) != len(functions):
        raise RuntimeError("The emitted function map is incomplete.")
    readouts = {
        record["readout_id"]: record
        for record in plan.certificate["yace_candidate_readouts"]["records"]
    }
    used = set()
    semantic_rows = []
    for entry, (central, function_index, function) in zip(
        mapping["entries"],
        functions,
    ):
        semantic = _feature_payload(central, function_index, function)
        semantic_rows.append(semantic)
        feature_hash = _stable_hash(semantic)
        if (
            entry["central_type"] != central
            or entry["function_index"] != function_index
            or entry["feature_id"] != feature_hash
            or entry["alternatives"][0]["evaluator"] != "explicit_ctilde"
        ):
            raise RuntimeError("A function-map entry is bound to another YACE row.")
        for candidate in entry["alternatives"]:
            body = {key: value for key, value in candidate.items() if key != "alternative_id"}
            if candidate["alternative_id"] != _candidate_id(body):
                raise RuntimeError("A candidate ID does not bind its semantics.")
            if candidate["evaluator"] == "execution_plan_readout":
                readout = readouts.get(candidate["readout_id"])
                if readout is None:
                    raise RuntimeError("A candidate references a missing readout.")
                body = {key: value for key, value in readout.items() if key != "readout_id"}
                if readout["readout_id"] != _readout_id(body):
                    raise RuntimeError("A readout ID does not bind its semantics.")
                used.add(readout["readout_id"])
    if used != set(readouts):
        raise RuntimeError("The emitted readout ledger has unused records.")
    expected_catalogue_hash = _stable_hash(
        {"functions": semantic_rows, "ordered_elements": list(model["elements"])}
    )
    if mapping["catalogue_hash"] != expected_catalogue_hash:
        raise RuntimeError("The emitted catalogue hash is inconsistent.")
    return {
        "candidate_count": sum(
            len(entry["alternatives"]) for entry in mapping["entries"]
        ),
        "function_count": len(functions),
        "optimized_candidate_count": len(readouts),
        "plan_hash": plan.plan_hash,
    }


def export_scalar_bundle_to_lammps(
    bundle,
    output_dir,
    *,
    elements,
    maximum_plan_bytes=128 * 1024 * 1024,
    absolute_tolerance=5.0e-11,
    relative_tolerance=1.0e-12,
    provenance=None,
):
    """Export a fitted ordinary scalar ACE bundle for PairYE3T.

    Purpose:
        Write a standalone PACE-compatible ``model.yace`` and a deterministic
        v3 candidate map/manifest over compiler-owned symmetric-power blocks.
    Mathematical contract:
        Every optimized row is coefficientwise identical to its fitted YACE
        polynomial and uses the explicit conjugate-transpose/product adjoint.
    Inputs:
        A fitted ``LinearACEScalarModelBundle``, output directory, and the
        ordered element names used by the fit.
    Outputs:
        ``model.yace``, ``execution_plan.json``, ``yace_function_map.json``,
        ``manifest.json``, and ``export_report.json``.
    Does not:
        Enumerate labels, change fitted coefficients, compile at LAMMPS load
        time, or add scalar-power/coupled-product-DAG candidates.
    """
    _validate_bundle_ordinary_scalar_catalogue(bundle)
    catalogue = dict(bundle.fit_metadata.get("ordinary_scalar_catalogue", {}) or {})
    if not catalogue:
        raise ValueError(
            "PairYE3T plan export requires fitted ordinary_scalar_catalogue metadata."
        )
    maximum_plan_bytes = int(maximum_plan_bytes)
    absolute_tolerance = float(absolute_tolerance)
    relative_tolerance = float(relative_tolerance)
    if maximum_plan_bytes <= 0:
        raise ValueError("maximum_plan_bytes must be positive.")
    if (
        not math.isfinite(absolute_tolerance)
        or absolute_tolerance <= 0.0
        or not math.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
    ):
        raise ValueError("Polynomial tolerances must be finite and nonnegative.")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("PairYE3T export directory must not already exist.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="." + destination.name + ".tmp-", dir=destination.parent)
    )
    try:
        model_path = temporary / "model.yace"
        export_scalar_bundle_to_yace(
            bundle,
            model_path,
            elements=tuple(elements),
            compatibility="lammps_pace_linear_v1",
        )
        model = read_yace(model_path, compatibility="lammps_pace_linear_v1")
        source_hash = _file_sha256(model_path)
        expected_by_central = _bundle_to_yace_functions(
            bundle,
            strict_pace=True,
        )
        row_by_feature = {
            str(row["feature_id"]): dict(row) for row in catalogue["rows"]
        }
        local_indices = {int(central): 0 for central in expected_by_central}
        plan_rows = []
        skipped = []
        descriptor_bindings = []
        for descriptor_index, (descriptor_row, function_spec) in enumerate(
            zip(catalogue["descriptor_rows"], bundle.descriptor_specs)
        ):
            catalogue_feature_id = str(descriptor_row["feature_id"])
            central_types = {int(channel.mu0) for channel in function_spec.channels}
            if len(central_types) != 1:
                raise RuntimeError("A fitted scalar descriptor has multiple central types.")
            central_type = central_types.pop()
            function_index = int(local_indices[central_type])
            local_indices[central_type] = function_index + 1
            function = model["functions"][central_type][function_index]
            expected_function = expected_by_central[central_type][function_index]
            if _feature_payload(central_type, function_index, function) != _feature_payload(
                central_type,
                function_index,
                expected_function,
            ):
                raise RuntimeError(
                    "Reparsed YACE function order differs from the fitted descriptor order."
                )
            semantic = _feature_payload(central_type, function_index, function)
            feature_hash = _stable_hash(semantic)
            binding = {
                "catalogue_feature_id": catalogue_feature_id,
                "central_type": central_type,
                "descriptor_index": descriptor_index,
                "feature_hash": feature_hash,
                "function_index": function_index,
            }
            descriptor_bindings.append(binding)
            row = row_by_feature[catalogue_feature_id]
            row_compiler = dict(catalogue["compiler"])
            row_compiler.update(row.get("compiler", {}))
            if row_compiler.get("coefficient_materialization") != "exact":
                skipped.append(
                    {
                        **binding,
                        "reason": "serialized_certified_numeric_coordinate_direct_only",
                    }
                )
                continue
            try:
                compiled = _compiler_coordinate(row, catalogue)
                plan, reason, blocks = _compile_block_candidate(
                    compiled,
                    function,
                    central_type,
                    function_index,
                    catalogue_feature_id,
                    maximum_plan_bytes,
                )
                if plan is None:
                    skipped.append({**binding, "reason": reason})
                    continue
                scale, equivalence = _certify_readout(
                    function,
                    blocks,
                    _plan_polynomial(plan),
                    absolute_tolerance,
                    relative_tolerance,
                )
                if scale is None:
                    skipped.append({**binding, "reason": "zero_fitted_row_direct_only"})
                    continue
            except MemoryError as error:
                skipped.append(
                    {
                        **binding,
                        "reason": "resource_guard_direct_only",
                        "detail": str(error),
                    }
                )
                continue
            except RuntimeError as error:
                if "Compiler identity mismatch" in str(error):
                    raise
                skipped.append(
                    {
                        **binding,
                        "reason": "uncertified_block_candidate_direct_only",
                        "detail": str(error),
                    }
                )
                continue
            plan_rows.append(
                {
                    **binding,
                    "blocks": blocks,
                    "equivalence": equivalence,
                    "plan": plan,
                    "scale": scale,
                }
            )
        if any(
            local_indices[int(central)] != len(functions)
            for central, functions in expected_by_central.items()
        ):
            raise RuntimeError("Fitted descriptor/YACE central-type mapping is incomplete.")
        if not plan_rows:
            raise RuntimeError(
                "No fitted row has a certified repeated-channel block candidate; "
                "use the standalone model.yace direct path."
            )
        readouts = []
        for row in plan_rows:
            term = {
                "channel_index": 0,
                "instruction_id": row["plan"].instructions[0].instruction_id,
                "magnetic_index": 0,
                "scale": [float(row["scale"].real), float(row["scale"].imag)],
                "tableau_index": 0,
            }
            term["binding_id"] = _binding_id(term)
            variable_order = []
            for block in row["blocks"]:
                source = block["source_binding"]["yace"]
                for magnetic in range(-int(block["input_L"]), int(block["input_L"]) + 1):
                    variable_order.append(
                        {
                            "angular_l": int(block["input_L"]),
                            "central_type": int(row["central_type"]),
                            "convention": "pace_complex_magnetic_y00_1",
                            "magnetic_m": magnetic,
                            "neighbor_type": int(source["neighbor_type"]),
                            "radial_n": int(source["radial_n"]),
                        }
                    )
            body = {
                "central_type": int(row["central_type"]),
                "equivalence": row["equivalence"],
                "feature_id": row["feature_hash"],
                "function_index": int(row["function_index"]),
                "schema": _READOUT_SCHEMA,
                "source_yace_sha256": source_hash,
                "terms": [term],
                "variable_order_hash": _stable_hash(variable_order),
            }
            readout = {"readout_id": _readout_id(body), **body}
            row["readout_id"] = readout["readout_id"]
            readouts.append(readout)
        scope = "fitted_ordinary_scalar_ace:" + str(catalogue["application_sha256"])
        aggregate = _aggregate_plans(plan_rows, source_hash, readouts, scope)
        plan_path = temporary / "execution_plan.json"
        plan_path.write_bytes(_canonical_bytes(aggregate.to_dict()))
        optimized_by_coordinate = {
            (row["central_type"], row["function_index"]): row for row in plan_rows
        }
        entries = []
        semantic_rows = []
        for central, bucket in sorted(model["functions"].items()):
            for function_index, function in enumerate(bucket):
                semantic = _feature_payload(central, function_index, function)
                semantic_rows.append(semantic)
                feature_hash = _stable_hash(semantic)
                direct = {
                    "availability": {"reason": "", "status": "available"},
                    "evaluator": "explicit_ctilde",
                    "required_capabilities": [
                        "explicit_ctilde_forward_adjoint_v1"
                    ],
                    "scale": [1.0, 0.0],
                    "source_binding": {
                        "central_type": int(central),
                        "feature_id": feature_hash,
                        "function_index": int(function_index),
                        "source_yace_sha256": source_hash,
                    },
                }
                alternatives = [_with_candidate_id(direct)]
                optimized = optimized_by_coordinate.get((int(central), int(function_index)))
                if optimized is not None:
                    block = {
                        "availability": {"reason": "", "status": "available"},
                        "compiler_plan_hash": aggregate.plan_hash,
                        "evaluator": "execution_plan_readout",
                        "readout_id": optimized["readout_id"],
                        "required_capabilities": [
                            "block_symmetric_power_adjoint_v1",
                            "block_symmetric_power_forward_v1",
                            "yace_candidate_readout_v1",
                        ],
                    }
                    alternatives.append(_with_candidate_id(block))
                entries.append(
                    {
                        "alternatives": alternatives,
                        "central_type": int(central),
                        "dispatch": "listed_candidates",
                        "feature_id": feature_hash,
                        "function_index": int(function_index),
                    }
                )
        mapping = {
            "catalogue_hash": _stable_hash(
                {
                    "functions": semantic_rows,
                    "ordered_elements": list(model["elements"]),
                }
            ),
            "coverage": "complete",
            "entries": entries,
            "plan_hash": aggregate.plan_hash,
            "schema": _MAP_SCHEMA,
            "selection_contract": {
                "fallback_policy": "forbid_unlisted",
                "mode": "load_time_portfolio_allowed",
                "schema": "ye3t_evaluator_candidates_v2",
            },
            "source_yace_sha256": source_hash,
        }
        map_path = temporary / "yace_function_map.json"
        map_path.write_bytes(_canonical_bytes(mapping))
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(
            _canonical_bytes(_manifest(model, model_path, aggregate, plan_path, map_path))
        )
        validation = _validate_emitted_bundle(temporary)
        report = {
            "catalogue_application_sha256": catalogue["application_sha256"],
            "catalogue_profile_id": catalogue["profile_id"],
            "descriptor_bindings": descriptor_bindings,
            "direct_candidate_count": len(entries),
            "function_count": len(entries),
            "maximum_plan_bytes": maximum_plan_bytes,
            "optimized_candidate_count": len(plan_rows),
            "plan_hash": aggregate.plan_hash,
            "polynomial_tolerances": {
                "absolute": absolute_tolerance,
                "relative": relative_tolerance,
            },
            "provenance": dict(provenance or {}),
            "schemas": {
                "execution_plan": _PLAN_SCHEMA,
                "function_map": _MAP_SCHEMA,
                "manifest": _SIDECAR_SCHEMA,
            },
            "skipped_candidates": skipped,
            "source_yace_sha256": source_hash,
            "validation": validation,
        }
        (temporary / "export_report.json").write_bytes(_canonical_bytes(report))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "execution_plan": destination / "execution_plan.json",
        "export_report": destination / "export_report.json",
        "manifest": destination / "manifest.json",
        "model": destination / "model.yace",
        "yace_function_map": destination / "yace_function_map.json",
    }


__all__ = ["compile_ordinary_scalar_catalogue", "export_scalar_bundle_to_lammps"]
