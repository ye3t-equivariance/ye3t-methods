"""Compiler-schedule runtime for the exact tagged-Cauchy physical image.

This module is the bounded Task 56T-0c bridge.  It does not enumerate a
tagged basis or recouple descriptors.  It consumes the exact complex moment
schedules and real-form maps supplied by
:mod:`ye3t.couplings.tagged_cauchy_image`, obtains the compiler-owned real
runtime program, binds it to a finite-cutoff shifted-Jacobi source, and exports
the resulting binary64 inference plan.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator

from ye3t.couplings import (
    CompiledTaggedCauchyImage,
    TAGGED_CAUCHY_REAL_SCHEDULE_SCHEMA,
    tagged_cauchy_real_schedule,
)
from ye3t.execution_plan import compile_tagged_moment_execution_portfolio

from ye3t_ace.lifted_cauchy_linear import LiftedCauchyPolynomialSource
from ye3t.couplings.orthogonal_shifted_jacobi import (
    shifted_jacobi_ladder_with_derivative,
)
from ye3t_ace.tagged_cauchy_linear import position_jacobian_from_edge_derivative
from ye3t_ace.equivariant_calc.edge_geometry import (
    directed_edges_all_images_bruteforce,
)


TAGGED_CAUCHY_IMAGE_MODEL_SCHEMA = "ye3t_tagged_cauchy_slice_v3"
TAGGED_CAUCHY_IMAGE_MODEL_SCHEMA_V4 = "ye3t_tagged_cauchy_slice_v4"
TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V1 = "ye3t_tagged_cauchy_direct_source_v1"
TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V2 = "ye3t_tagged_cauchy_direct_source_v2"
TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA = TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V2
TAGGED_CAUCHY_READOUT_SCHEMA = "ye3t_tagged_cauchy_readout_v1"

TAGGED_CAUCHY_V3_CONVENTIONS = {
    "precision": "binary64",
    "edge_displacement": "R_neighbor_minus_R_center_plus_periodic_image",
    "force_sign": "F_equals_minus_dE_dR",
    "lammps_virial": "minus_strain_derivative",
    "real_basis": "compiler_derived_orthonormal_real_tesseral",
    "exact_zero_distance_force_policy": (
        "reject_before_direction_evaluation_v1"
    ),
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
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _binding(payload):
    payload = _jsonable(payload)
    return {"payload": payload, "hash": _payload_hash(payload)}


def _deployment_identity(payload):
    return {
        "compiler_artifact_hash": payload["compiler_artifact_hash"],
        "source_plan_hash": payload["source_binding"]["hash"],
        "schedule_hash": payload["schedule_binding"]["hash"],
        "readout_hash": payload["readout_binding"]["hash"],
        "conventions": payload["conventions"],
    }


def _validate_binding(binding, name):
    if not isinstance(binding, dict) or set(binding) != {"payload", "hash"}:
        raise ValueError(f"Tagged-Cauchy {name} binding is malformed.")
    expected = str(binding["hash"])
    actual = _payload_hash(binding["payload"])
    if not expected or expected != actual:
        raise ValueError(f"Tagged-Cauchy {name} hash mismatch.")
    return binding["payload"]


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Tagged-Cauchy payloads require finite numbers.")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    raise TypeError(
        "Tagged-Cauchy payload contains unsupported type "
        f"{type(value).__name__}."
    )


def _load_compiled(compiled, compiler_validation="full"):
    if isinstance(compiled, CompiledTaggedCauchyImage):
        compiled = compiled.to_dict()
    return CompiledTaggedCauchyImage.from_dict(compiled, compiler_validation=compiler_validation)


def _generator_base(generator):
    return (
        str(generator["neighbor_species"]),
        str(generator["source_family_id"]),
        str(generator["support_id"]),
        int(generator["q"]),
        int(generator["l"]),
    )


def _inventory_base(record):
    return _generator_base(record["source_key"])


def _compiler_channel_records(compiled):
    forms = {
        int(record["angular_l"]): str(record["real_form_id"])
        for record in compiled.payload["real_forms"]
    }
    bases = sorted(
        {
            _generator_base(generator)
            for schedule in compiled.payload["moment_schedules"]
            for generator in schedule["source_generators"]
        }
    )
    return [
        {
            "channel_index": int(index),
            "neighbor_species": base[0],
            "source_family_id": base[1],
            "support_id": base[2],
            "q": int(base[3]),
            "l": int(base[4]),
            "real_form_id": forms[int(base[4])],
        }
        for index, base in enumerate(bases)
    ]


def _validate_forward_adjoint(program):
    feature_count = int(program["feature_count"])
    source_count = len(program["real_density_keys"])
    derived = {}
    for term in program["terms"]:
        feature = int(term["feature_index"])
        monomial = tuple(sorted(int(value) for value in term["density_factor_indices"]))
        coefficient = float(term["coefficient"])
        if not 0 <= feature < feature_count or not np.isfinite(coefficient):
            raise ValueError("Tagged-Cauchy forward schedule index or coefficient is invalid.")
        if any(not 0 <= value < source_count for value in monomial):
            raise ValueError("Tagged-Cauchy forward source index is invalid.")
        for source, multiplicity in Counter(monomial).items():
            remaining = list(monomial)
            remaining.remove(source)
            key = (feature, source, tuple(remaining))
            derived[key] = derived.get(key, 0.0) + multiplicity * coefficient
    supplied = {}
    for term in program["adjoint_terms"]:
        feature = int(term["feature_index"])
        source = int(term["source_index"])
        remaining = tuple(sorted(int(value) for value in term["remaining_source_indices"]))
        coefficient = float(term["coefficient"])
        if (
            not 0 <= feature < feature_count
            or not 0 <= source < source_count
            or any(not 0 <= value < source_count for value in remaining)
            or not np.isfinite(coefficient)
        ):
            raise ValueError("Tagged-Cauchy adjoint schedule index or coefficient is invalid.")
        key = (feature, source, remaining)
        supplied[key] = supplied.get(key, 0.0) + coefficient
    if set(derived) != set(supplied) or any(
        not math.isclose(
            derived[key], supplied[key], rel_tol=4.0e-14, abs_tol=4.0e-14
        )
        for key in derived
    ):
        raise ValueError("Tagged-Cauchy binary64 forward/adjoint schedules disagree.")


def realify_tagged_cauchy_image(compiled, compiler_validation="full"):
    """Return the exact real schedule materialized and owned by ``ye3t``."""

    return tagged_cauchy_real_schedule(compiled, compiler_validation=compiler_validation)


def tagged_cauchy_source_plan(
    compiled,
    cutoff,
    real_program=None,
    numerical_realization="shifted_jacobi_three_term_v1",
    pair_cutoffs=None,
    compiler_validation="full",
):
    """Bind the compiled normalized source algebra to one physical cutoff."""

    compiled = _load_compiled(compiled, compiler_validation=compiler_validation)
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("Tagged-Cauchy cutoff must be positive and finite.")
    program = (
        realify_tagged_cauchy_image(compiled, compiler_validation=compiler_validation)
        if real_program is None
        else dict(real_program)
    )
    request = compiled.plan.report.request
    algebra = compiled.payload.get("source_product_algebra", request.get("source_product_algebra"))
    inventory = {_inventory_base(record): record for record in algebra["source_inventory"]}
    forms = {
        int(record["angular_l"]): record for record in compiled.payload["real_forms"]
    }
    channels = []
    for channel in program["channels"]:
        base = _generator_base(channel)
        if base not in inventory:
            raise ValueError("Tagged-Cauchy runtime source is absent from the compiler inventory.")
        record = inventory[base]
        coefficients = [
            int(value["numerator"]) / int(value["denominator"])
            for value in record["shifted_jacobi_power_coefficients"]
        ]
        norm = record["normalization_squared"]
        normalization = math.sqrt(int(norm["numerator"]) / int(norm["denominator"]))
        channels.append(
            {
                **dict(channel),
                "shifted_jacobi_power_coefficients": list(
                    record["shifted_jacobi_power_coefficients"]
                ),
                "binary64_power_coefficients": coefficients,
                "normalization_squared": dict(norm),
                "binary64_normalization": float(normalization),
                "angular_racah_scale": float(
                    math.sqrt(4.0 * math.pi / (2 * int(channel["l"]) + 1))
                ),
            }
        )
    realizations = {
        "shifted_jacobi_three_term_v1": TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V2,
        "legacy_expanded_power_horner_v1": TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V1,
    }
    if numerical_realization not in realizations:
        raise ValueError(
            "Unsupported tagged-Cauchy numerical source realization: "
            f"{numerical_realization!r}."
        )
    body = {
        "schema": realizations[numerical_realization],
        "compiler_artifact_hash": str(compiled.self_hash),
        "source_product_algebra_hash": str(algebra["record_hash"]),
        "cutoff": cutoff,
        "species_order": list(request.get("species", sorted(
            {str(channel["neighbor_species"]) for channel in channels}
        ))),
        "source_family_id": str(algebra["source_family_id"]),
        "support_id": str(algebra["normalized_support"]["support_id"]),
        "exact_zero_distance_force_policy": str(
            algebra["normalized_support"]["exact_zero_distance_force_policy"]
        ),
        "radial_coordinate": "x=r/r_c",
        "radial_measure": "x^2_dx",
        "envelope": "(1-x)^2",
        "real_forms": list(forms.values()),
        "channels": channels,
        "certificate": {
            "passed": True,
            "cutoff_value_and_first_derivative_zero_exact": True,
            "compiler_source_inventory_consumed": True,
            "runtime_gram_solve": False,
        },
    }
    if body["schema"] == TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V2:
        body["numerical_evaluation"] = {
            "schema": "ye3t_shifted_jacobi_three_term_v1",
            "alpha": 4,
            "beta_rule": "2*l+2",
            "argument": "2*x-1",
            "derivative": "d/dx",
            "evaluation_order": "per_edge_per_l_ladder",
            "expanded_power_coefficients_use": "provenance_only",
        }
    if pair_cutoffs is not None:
        species = tuple(request.get("species", body["species_order"]))
        pairs = {str(key): float(value) for key, value in pair_cutoffs.items()}
        expected = {left+"-"+right for left in species for right in species}
        if set(pairs) != expected or any(not np.isfinite(value) or value <= 0 or value > cutoff for value in pairs.values()):
            raise ValueError("pair_cutoffs must cover every directed species pair within the host cutoff.")
        body["pair_cutoffs_A"] = pairs
        body["species_order"] = list(species)
    return {**body, "source_plan_hash": _payload_hash(body)}


def _validate_real_program(program, compiled):
    program = dict(program)
    supplied = str(program.pop("program_hash", ""))
    if not supplied or supplied != _payload_hash(program):
        raise ValueError("Tagged-Cauchy real schedule hash mismatch.")
    committed = str(compiled.payload.get("real_schedule_core_hash", ""))
    compiler_core = compiled.payload.get("real_schedule_core")
    if supplied != committed or _jsonable(program) != _jsonable(compiler_core):
        raise ValueError(
            "Tagged-Cauchy real schedule differs from its compiler commitment."
        )
    if program.get("schema") != TAGGED_CAUCHY_REAL_SCHEDULE_SCHEMA:
        raise ValueError("Unsupported tagged-Cauchy real schedule schema.")
    if (
        program.get("core_schema")
        != "ye3t_tagged_cauchy_real_schedule_core_v1"
        or program.get("lowering_convention")
        != "exact_complex_to_real_tesseral_v1"
        or program.get("coefficient_encoding")
        != "exact_algebraic_plus_binary64_v1"
    ):
        raise ValueError("Unsupported tagged-Cauchy compiler lowering convention.")
    if str(program["catalogue_hash"]) != str(compiled.payload["catalogue_hash"]):
        raise ValueError("Tagged-Cauchy real schedule catalogue binding changed.")
    if str(program["source_product_algebra_hash"]) != str(
        compiled.payload["source_product_algebra_hash"]
    ):
        raise ValueError("Tagged-Cauchy source-product schedule binding changed.")
    expected_hashes = [
        str(value["schedule_hash"])
        for value in compiled.payload["moment_schedules"]
    ]
    if list(program["compiler_schedule_hashes"]) != expected_hashes:
        raise ValueError("Tagged-Cauchy compiler schedule ordering changed.")
    expected_channels = _compiler_channel_records(compiled)
    if _jsonable(program["channels"]) != _jsonable(expected_channels):
        raise ValueError("Tagged-Cauchy compiler channel inventory changed.")
    expected_density_keys = [
        [int(channel["channel_index"]), component]
        for channel in expected_channels
        for component in range(2 * int(channel["l"]) + 1)
    ]
    if _jsonable(program["real_density_keys"]) != expected_density_keys:
        raise ValueError("Tagged-Cauchy real density inventory changed.")
    if int(program["feature_count"]) != len(compiled.payload["moment_schedules"]):
        raise ValueError("Tagged-Cauchy real schedule feature count changed.")
    if not all(
        bool(program["certificate"].get(key, False))
        for key in (
            "passed",
            "exact_realification",
            "compiler_adjoint_matches_real_forward_exact",
            "division_free_adjoint",
        )
    ):
        raise ValueError("Tagged-Cauchy real schedule is uncertified.")
    _validate_forward_adjoint(program)
    return {**program, "program_hash": supplied}


def _validate_source_plan(source_plan, compiled, program):
    source_plan = dict(source_plan)
    supplied = str(source_plan.pop("source_plan_hash", ""))
    if not supplied or supplied != _payload_hash(source_plan):
        raise ValueError("Tagged-Cauchy source-plan hash mismatch.")
    schema = source_plan.get("schema")
    if schema not in {
        TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V1,
        TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V2,
    }:
        raise ValueError("Unsupported tagged-Cauchy source-plan schema.")
    if schema == TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V2:
        expected_evaluation = {
            "schema": "ye3t_shifted_jacobi_three_term_v1",
            "alpha": 4,
            "beta_rule": "2*l+2",
            "argument": "2*x-1",
            "derivative": "d/dx",
            "evaluation_order": "per_edge_per_l_ladder",
            "expanded_power_coefficients_use": "provenance_only",
        }
        if source_plan.get("numerical_evaluation") != expected_evaluation:
            raise ValueError(
                "Tagged-Cauchy V2 source numerical realization is unsupported."
            )
    elif "numerical_evaluation" in source_plan:
        raise ValueError("Tagged-Cauchy V1 source plan contains V2 metadata.")
    if str(source_plan["compiler_artifact_hash"]) != str(compiled.self_hash):
        raise ValueError("Tagged-Cauchy source/compiler binding changed.")
    algebra = compiled.payload.get("source_product_algebra", compiled.plan.report.request.get("source_product_algebra"))
    if str(source_plan["source_product_algebra_hash"]) != str(
        algebra["record_hash"]
    ):
        raise ValueError("Tagged-Cauchy source-product algebra binding changed.")
    if str(source_plan["source_family_id"]) != str(algebra["source_family_id"]):
        raise ValueError("Tagged-Cauchy source family changed.")
    if str(source_plan["support_id"]) != str(
        algebra["normalized_support"]["support_id"]
    ):
        raise ValueError("Tagged-Cauchy source support changed.")
    if source_plan["exact_zero_distance_force_policy"] != (
        "reject_before_direction_evaluation_v1"
    ):
        raise ValueError("Tagged-Cauchy exact-overlap policy is unsupported.")
    expected_semantics = {
        "radial_coordinate": "x=r/r_c",
        "radial_measure": "x^2_dx",
        "envelope": "(1-x)^2",
    }
    if any(
        source_plan.get(key) != value
        for key, value in expected_semantics.items()
    ):
        raise ValueError("Tagged-Cauchy source semantics are unsupported.")
    certificate = source_plan.get("certificate")
    if not isinstance(certificate, dict) or any(
        certificate.get(key) is not value
        for key, value in {
            "passed": True,
            "cutoff_value_and_first_derivative_zero_exact": True,
            "compiler_source_inventory_consumed": True,
            "runtime_gram_solve": False,
        }.items()
    ):
        raise ValueError("Tagged-Cauchy source plan is uncertified.")
    if _jsonable(source_plan["real_forms"]) != _jsonable(
        compiled.payload["real_forms"]
    ):
        raise ValueError("Tagged-Cauchy source real forms changed.")
    if len(source_plan["channels"]) != len(program["channels"]):
        raise ValueError("Tagged-Cauchy source channel count changed.")
    inventory = {
        _inventory_base(record): record for record in algebra["source_inventory"]
    }
    logical_fields = (
        "channel_index",
        "neighbor_species",
        "source_family_id",
        "support_id",
        "q",
        "l",
        "real_form_id",
    )
    for source_channel, program_channel in zip(
        source_plan["channels"], program["channels"], strict=True
    ):
        if any(
            source_channel[field] != program_channel[field]
            for field in logical_fields
        ):
            raise ValueError("Tagged-Cauchy source/schedule channel binding changed.")
        base = _generator_base(source_channel)
        if base not in inventory:
            raise ValueError("Tagged-Cauchy source channel is absent from compiler inventory.")
        record = inventory[base]
        if _jsonable(source_channel["shifted_jacobi_power_coefficients"]) != _jsonable(
            record["shifted_jacobi_power_coefficients"]
        ):
            raise ValueError("Tagged-Cauchy exact radial coefficients changed.")
        exact_coefficients = [
            int(value["numerator"]) / int(value["denominator"])
            for value in record["shifted_jacobi_power_coefficients"]
        ]
        if not np.allclose(
            source_channel["binary64_power_coefficients"],
            exact_coefficients,
            rtol=0.0,
            atol=4.0e-15,
        ):
            raise ValueError("Tagged-Cauchy binary64 radial coefficients changed.")
        norm = record["normalization_squared"]
        if _jsonable(source_channel["normalization_squared"]) != _jsonable(norm):
            raise ValueError("Tagged-Cauchy exact radial normalization changed.")
        expected_normalization = math.sqrt(
            int(norm["numerator"]) / int(norm["denominator"])
        )
        expected_angular_scale = math.sqrt(
            4.0 * math.pi / (2 * int(source_channel["l"]) + 1)
        )
        if not math.isclose(
            float(source_channel["binary64_normalization"]),
            expected_normalization,
            rel_tol=4.0e-15,
            abs_tol=4.0e-15,
        ) or not math.isclose(
            float(source_channel["angular_racah_scale"]),
            expected_angular_scale,
            rel_tol=4.0e-15,
            abs_tol=4.0e-15,
        ):
            raise ValueError("Tagged-Cauchy binary64 source normalization changed.")
    expected_species = sorted(compiled.plan.report.request.get("species", {
        str(channel["neighbor_species"]) for channel in source_plan["channels"]}))
    if list(source_plan["species_order"]) != expected_species:
        raise ValueError("Tagged-Cauchy source species ordering changed.")
    if "pair_cutoffs_A" in source_plan:
        pairs = source_plan["pair_cutoffs_A"]
        if set(pairs) != {left+"-"+right for left in expected_species for right in expected_species} or any(
            not np.isfinite(value) or value <= 0 or value > source_plan["cutoff"] for value in pairs.values()):
            raise ValueError("Invalid directed-pair cutoff map.")
    return {**source_plan, "source_plan_hash": supplied}


class TaggedCauchyImageEvaluator:
    """Binary64 evaluator of a compiler-owned tagged physical image."""

    def __init__(self, compiled, source_plan, real_program, backend="reference", compiler_validation="full"):
        self.compiler_validation = compiler_validation
        self.compiled = _load_compiled(compiled, compiler_validation=compiler_validation)
        self.program = _validate_real_program(real_program, self.compiled)
        self.source_plan = _validate_source_plan(
            source_plan, self.compiled, self.program
        )
        self.channels = tuple(self.source_plan["channels"])
        self.species_order = tuple(self.source_plan["species_order"])
        self.type_map = {name: index for index, name in enumerate(self.species_order)}
        self.cutoff = float(self.source_plan["cutoff"])
        self._stable_jacobi = (
            self.source_plan["schema"] == TAGGED_CAUCHY_SOURCE_PLAN_SCHEMA_V2
        )
        self.feature_count = int(self.program["feature_count"])
        self._component_offsets = []
        offset = 0
        for channel in self.channels:
            self._component_offsets.append(offset)
            offset += 2 * int(channel["l"]) + 1
        self._component_count = offset
        self._density_flat = tuple(
            self._component_offsets[int(channel)] + int(component)
            for channel, component in self.program["real_density_keys"]
        )
        if backend not in {"auto", "native", "reference"}:
            raise ValueError("Tagged polynomial backend must be auto, native, or reference.")
        from ye3t.runtime.execution_plan import native_execution_plan_capabilities
        available = native_execution_plan_capabilities()["cpu"] if backend != "reference" else False
        if backend == "native" and not available:
            raise RuntimeError("The requested native tagged polynomial backend is not installed.")
        self.backend = "native" if available and backend != "reference" else "reference"
        self.execution_report = {"source_backend": "torch_analytic_jacobi_solid_harmonics",
            "polynomial_backend": self.backend, "derivatives": "compiler_explicit_product_rule",
            "compiler_validation": compiler_validation}
        if self.backend == "native":
            self._forward_table = self._polynomial_table(self.program["terms"], False)
            self._derivative_table = self._polynomial_table(self.program["adjoint_terms"], True)

    def _polynomial_table(self, terms, derivative):
        """Pack the compiler's literal monomials for the existing BSD kernel."""
        coordinates = (sorted({(int(term["feature_index"]), int(term["source_index"])) for term in terms})
                       if derivative else [(index,) for index in range(self.feature_count)])
        coordinate_index = {key: index for index, key in enumerate(coordinates)}
        grouped = [[] for _ in coordinates]
        for term in terms:
            key = ((int(term["feature_index"]), int(term["source_index"])) if derivative
                   else (int(term["feature_index"]),))
            grouped[coordinate_index[key]].append(term)
        width = len(self._density_flat)
        counts, outputs, coefficients, offsets = [], [], [], [0]
        for output, group in enumerate(grouped):
            for term in group:
                factors = term["remaining_source_indices" if derivative else "density_factor_indices"]
                row = [0]*width
                for index in factors:
                    row[int(index)] += 1
                counts.append(row)
                outputs.append(output)
                coefficients.append(float(term["coefficient"]))
            offsets.append(len(counts))
        return (torch.tensor(counts, dtype=torch.long).reshape(-1, width),
                torch.tensor(offsets, dtype=torch.long), torch.tensor(outputs, dtype=torch.long),
                torch.tensor(coefficients, dtype=torch.float64), tuple(coordinates))

    @staticmethod
    def _power_series_with_derivative(coefficients, coordinate):
        value = torch.zeros_like(coordinate) + float(coefficients[-1])
        derivative = torch.zeros_like(coordinate)
        for coefficient in reversed(coefficients[:-1]):
            derivative = derivative * coordinate + value
            value = value * coordinate + float(coefficient)
        return value, derivative

    def edge_sources_with_derivatives(self, displacements, neighbor_types, central_types=None):
        displacements = torch.as_tensor(displacements)
        if displacements.dtype != torch.float64:
            raise ValueError("Tagged-Cauchy source evaluation requires float64.")
        if displacements.ndim != 2 or int(displacements.shape[1]) != 3:
            raise ValueError("displacements must have shape [edges,3].")
        neighbor_types = torch.as_tensor(
            neighbor_types, dtype=torch.long, device=displacements.device
        )
        if tuple(neighbor_types.shape) != (int(displacements.shape[0]),):
            raise ValueError("neighbor_types must have one value per edge.")
        edge_count = int(displacements.shape[0])
        values = displacements.new_zeros((edge_count, self._component_count))
        derivatives = displacements.new_zeros(
            (edge_count, self._component_count, 3)
        )
        if edge_count == 0:
            return values, derivatives
        distance = torch.linalg.norm(displacements, dim=1)
        if bool(torch.any(distance == 0.0)):
            raise ValueError(
                "Tagged-Cauchy source rejects exact zero separation before "
                "direction evaluation."
            )
        cutoff = torch.as_tensor(
            self.cutoff, dtype=displacements.dtype, device=displacements.device
        ).expand_as(distance)
        pairs = self.source_plan.get("pair_cutoffs_A")
        if pairs is not None:
            if central_types is None:
                raise ValueError("Directed-pair sources require the central species of every edge.")
            central_types = torch.as_tensor(central_types, dtype=torch.long, device=displacements.device)
            if central_types.shape != neighbor_types.shape:
                raise ValueError("central_types must have one value per edge.")
            table = displacements.new_tensor([[pairs[left+"-"+right] for right in self.species_order]
                                             for left in self.species_order])
            cutoff = table[central_types, neighbor_types]
        coordinate = distance / cutoff
        unit = displacements / distance.unsqueeze(1)
        active = distance < cutoff
        fallback_coordinate = torch.full_like(coordinate, 0.5)
        evaluation_coordinate = torch.where(active, coordinate, fallback_coordinate)
        fallback_unit = torch.zeros_like(unit)
        fallback_unit[:, 0] = 1.0
        evaluation_unit = torch.where(active.unsqueeze(1), unit, fallback_unit)
        evaluation_displacements = (
            evaluation_coordinate.unsqueeze(1) * evaluation_unit
        )
        geometry = (
            evaluation_displacements,
            neighbor_types,
            evaluation_coordinate,
            evaluation_unit,
            active,
            1.0,
        )
        solid_cache = {}
        one_minus = 1.0 - evaluation_coordinate
        envelope = one_minus.square()
        envelope_derivative = -2.0 * one_minus
        jacobi_cache = {}
        if self._stable_jacobi:
            maximum_q_by_l = {}
            for channel in self.channels:
                angular_l = int(channel["l"])
                maximum_q_by_l[angular_l] = max(
                    maximum_q_by_l.get(angular_l, -1), int(channel["q"])
                )
            for angular_l, maximum_q in maximum_q_by_l.items():
                jacobi_cache[angular_l] = shifted_jacobi_ladder_with_derivative(
                    maximum_q,
                    4,
                    2 * angular_l + 2,
                    evaluation_coordinate,
                )
        for channel_index, channel in enumerate(self.channels):
            angular_l = int(channel["l"])
            if angular_l not in solid_cache:
                solid_cache[angular_l] = (
                    LiftedCauchyPolynomialSource._compiler_ordered_regular_solid(
                        angular_l, geometry
                    )
                )
            solid, normalized_solid_derivative = solid_cache[angular_l]
            solid_derivative = normalized_solid_derivative / cutoff.reshape(-1, 1, 1)
            if self._stable_jacobi:
                degree = int(channel["q"])
                polynomial = jacobi_cache[angular_l][0][degree]
                polynomial_derivative = jacobi_cache[angular_l][1][degree]
            else:
                polynomial, polynomial_derivative = self._power_series_with_derivative(
                    channel["binary64_power_coefficients"], evaluation_coordinate
                )
            normalization = float(channel["binary64_normalization"])
            radial = normalization * envelope * polynomial
            radial_derivative = normalization * (
                envelope_derivative * polynomial
                + envelope * polynomial_derivative
            )
            width = 2 * angular_l + 1
            begin = self._component_offsets[channel_index]
            end = begin + width
            species = self.type_map[str(channel["neighbor_species"])]
            mask = (active & (neighbor_types == species)).to(displacements.dtype)
            value = radial.unsqueeze(1) * solid
            derivative = (
                radial.reshape(-1, 1, 1) * solid_derivative
                + radial_derivative.reshape(-1, 1, 1)
                * solid.unsqueeze(2)
                * evaluation_unit.unsqueeze(1)
                / cutoff.reshape(-1, 1, 1)
            )
            values[:, begin:end] = value * mask.unsqueeze(1)
            derivatives[:, begin:end, :] = derivative * mask.reshape(-1, 1, 1)
        return values, derivatives

    def evaluate_edge_list(self, edge_index, displacements, neighbor_types, atom_count, atom_types=None):
        edge_index = torch.as_tensor(
            edge_index, dtype=torch.long, device=torch.as_tensor(displacements).device
        )
        if tuple(edge_index.shape) != (2, int(torch.as_tensor(displacements).shape[0])):
            raise ValueError("edge_index must have shape [2,edges].")
        atom_count = int(atom_count)
        source_values, source_derivatives = self.edge_sources_with_derivatives(
            displacements, neighbor_types,
            None if atom_types is None else torch.as_tensor(atom_types, device=edge_index.device).index_select(0, edge_index[0])
        )
        source_values = source_values[:, self._density_flat]
        source_derivatives = source_derivatives[:, self._density_flat, :]
        source_count = len(self._density_flat)
        centers = edge_index[0]
        moments = source_values.new_zeros((atom_count, source_count))
        if int(centers.numel()):
            moments.index_add_(0, centers, source_values)

        if self.backend == "native":
            from ye3t.runtime.execution_plan import symmetric_power_monomial_contraction
            forward = self._forward_table
            reverse = self._derivative_table
            features = symmetric_power_monomial_contraction(moments, *forward[:4], backend="native")
            derivatives = symmetric_power_monomial_contraction(moments, *reverse[:4], backend="native")
            edge_feature_derivative = source_values.new_zeros((len(centers), self.feature_count, 3))
            for feature in range(self.feature_count):
                entries = [(index, source) for index, (output, source) in enumerate(reverse[4]) if output == feature]
                if entries:
                    indices = torch.tensor([entry[0] for entry in entries], device=centers.device)
                    sources = torch.tensor([entry[1] for entry in entries], device=centers.device)
                    edge_feature_derivative[:, feature] = torch.einsum("es,esd->ed",
                        derivatives.index_select(0, centers).index_select(1, indices),
                        source_derivatives.index_select(1, sources))
            return features, edge_feature_derivative

        features = source_values.new_zeros((atom_count, self.feature_count))
        for term in self.program["terms"]:
            value = source_values.new_ones(atom_count)
            for source_index in term["density_factor_indices"]:
                value = value * moments[:, int(source_index)]
            features[:, int(term["feature_index"])] += (
                float(term["coefficient"]) * value
            )

        adjoint = source_values.new_zeros(
            (atom_count, self.feature_count, source_count)
        )
        for term in self.program["adjoint_terms"]:
            value = source_values.new_ones(atom_count)
            for source_index in term["remaining_source_indices"]:
                value = value * moments[:, int(source_index)]
            adjoint[:, int(term["feature_index"]), int(term["source_index"])] += (
                float(term["coefficient"]) * value
            )
        if int(centers.numel()):
            edge_feature_derivative = torch.einsum(
                "efs,esd->efd", adjoint.index_select(0, centers), source_derivatives
            )
        else:
            edge_feature_derivative = source_values.new_zeros(
                (0, self.feature_count, 3)
            )
        return features, edge_feature_derivative

    def materialize(self, positions, atom_types, cell=None, pbc=None):
        positions = torch.as_tensor(positions)
        if positions.dtype != torch.float64:
            raise ValueError("Tagged-Cauchy materialization requires float64 positions.")
        atom_types = torch.as_tensor(
            atom_types, dtype=torch.long, device=positions.device
        )
        cell_tensor = None
        if cell is not None:
            cell_tensor = torch.as_tensor(
                cell, dtype=positions.dtype, device=positions.device
            )
        src, dst, displacement, _distance = directed_edges_all_images_bruteforce(
            positions, self.cutoff, cell=cell_tensor, pbc=pbc
        )
        edge_index = torch.stack((src, dst))
        neighbor_types = atom_types.index_select(0, dst)
        values, edge_derivatives = self.evaluate_edge_list(
            edge_index, displacement, neighbor_types, int(positions.shape[0]), atom_types=atom_types
        )
        return edge_index, displacement, values, edge_derivatives

    def features_and_position_jacobian(self, positions, atom_types, cell=None, pbc=None):
        edge_index, _displacement, values, edge_derivatives = self.materialize(
            positions, atom_types, cell=cell, pbc=pbc
        )
        jacobian = position_jacobian_from_edge_derivative(
            edge_derivatives,
            edge_index,
            int(torch.as_tensor(positions).shape[0]),
            self.feature_count,
        )
        return values, jacobian


class TaggedCauchyImageLinearModel:
    """One compiler image, per-central-species readout, and atomic offsets."""

    def __init__(self, evaluator, beta, offsets):
        self.evaluator = evaluator
        self.species_order = tuple(evaluator.species_order)
        if isinstance(beta, dict):
            beta_by_species = {
                str(key): torch.as_tensor(value, dtype=torch.float64)
                for key, value in beta.items()
            }
        elif len(self.species_order) == 1:
            beta_by_species = {
                self.species_order[0]: torch.as_tensor(beta, dtype=torch.float64)
            }
        else:
            raise ValueError("Multi-species tagged models require per-species beta.")
        if set(beta_by_species) != set(self.species_order):
            raise ValueError("Tagged-Cauchy beta keys must equal species_order.")
        if any(
            tuple(value.shape) != (evaluator.feature_count,)
            or not bool(torch.all(torch.isfinite(value)))
            for value in beta_by_species.values()
        ):
            raise ValueError("Tagged-Cauchy beta vectors have invalid shape or values.")
        offsets = {str(key): float(value) for key, value in dict(offsets).items()}
        if set(offsets) != set(self.species_order) or not all(
            np.isfinite(value) for value in offsets.values()
        ):
            raise ValueError("Tagged-Cauchy offsets must cover species_order.")
        self.beta_by_species = beta_by_species
        self.offsets = offsets
        self.reference_terms = {}

    def _reference_values(self, atoms):
        """Restore the immutable reference terms bound into the model readout."""
        refs = self.reference_terms.get("atomic_energies", {})
        atomic = np.array([refs.get(symbol, 0.0) for symbol in atoms.get_chemical_symbols()])
        forces = np.zeros((len(atoms), 3))
        virial = np.zeros(6)
        zbl = self.reference_terms.get("zbl")
        if zbl is not None:
            from ye3t_ace.reference_potentials import evaluate_zbl_reference
            values = evaluate_zbl_reference([atoms], zbl)
            atomic += values["reference_atomic_energies"][0]
            forces += values["reference_forces"][0]
            virial += values["reference_virials"][0]
        return atomic, forces, virial

    def energy_forces_virial(self, positions, atom_types, cell=None, pbc=None):
        positions = torch.as_tensor(positions, dtype=torch.float64)
        atom_types = torch.as_tensor(
            atom_types, dtype=torch.long, device=positions.device
        )
        edge_index, displacement, features, edge_feature_derivative = (
            self.evaluator.materialize(positions, atom_types, cell=cell, pbc=pbc)
        )
        beta = torch.stack(
            tuple(
                self.beta_by_species[name].to(positions)
                for name in self.species_order
            )
        )
        offset = positions.new_tensor(
            tuple(self.offsets[name] for name in self.species_order)
        )
        atomic_energy = offset.index_select(0, atom_types) + torch.sum(
            features * beta.index_select(0, atom_types), dim=1
        )
        centers = edge_index[0]
        center_beta = beta.index_select(0, atom_types.index_select(0, centers))
        edge_gradient = torch.einsum(
            "ef,efd->ed", center_beta, edge_feature_derivative
        )
        forces = positions.new_zeros(tuple(positions.shape))
        if int(centers.numel()):
            forces.index_add_(0, centers, edge_gradient)
            forces.index_add_(0, edge_index[1], -edge_gradient)
        virial = positions.new_zeros(6)
        if int(centers.numel()):
            virial[0] = -torch.sum(displacement[:, 0] * edge_gradient[:, 0])
            virial[1] = -torch.sum(displacement[:, 1] * edge_gradient[:, 1])
            virial[2] = -torch.sum(displacement[:, 2] * edge_gradient[:, 2])
            virial[3] = -torch.sum(displacement[:, 0] * edge_gradient[:, 1])
            virial[4] = -torch.sum(displacement[:, 0] * edge_gradient[:, 2])
            virial[5] = -torch.sum(displacement[:, 1] * edge_gradient[:, 2])
        if self.reference_terms:
            from ase import Atoms
            atoms = Atoms(symbols=[self.species_order[int(value)] for value in atom_types],
                          positions=positions.detach().cpu().numpy(),
                          cell=None if cell is None else torch.as_tensor(cell).detach().cpu().numpy(),
                          pbc=False if pbc is None else pbc)
            reference_atomic, reference_forces, reference_virial = self._reference_values(atoms)
            atomic_energy = atomic_energy + positions.new_tensor(reference_atomic)
            forces = forces + positions.new_tensor(reference_forces)
            virial = virial + positions.new_tensor(reference_virial)
        return atomic_energy.sum(), forces, virial, atomic_energy

    def ase_calculator(self, backend="reference", *, native_library=None,
                       execution_policy="direct"):
        """Return an ASE energy/force/stress calculator for this linear model.

        Purpose:
            Evaluate a fitted tagged model on changing ASE structures.
        Mathematical contract:
            Forces and virial use the compiler's explicit product derivative.
        Inputs:
            ``backend`` is ``reference``, ``native_polynomial``, or
            ``native_cpu``. The latter uses the standalone LAMMPS CPU kernel.
        Outputs:
            An ASE calculator with energy, atomic energies, forces, and stress.
        Does not:
            Claim a full native source kernel for ``native_polynomial``.
        """

        if backend not in {"reference", "native_polynomial", "native_cpu"}:
            raise ValueError("Tagged ASE backend must be reference, native_polynomial, or native_cpu.")
        if backend == "native_cpu":
            from ye3t_ace.tagged_cauchy_native import _TaggedCauchyNativeRuntime

            runtime = _TaggedCauchyNativeRuntime(
                self, library_path=native_library, execution_policy=execution_policy,
            )
            return YE3TTaggedCauchyCalculator(self, native_runtime=runtime)
        requested = "reference" if backend == "reference" else "native"
        evaluator = TaggedCauchyImageEvaluator(
            self.evaluator.compiled, self.evaluator.source_plan,
            self.evaluator.program, backend=requested,
            compiler_validation=self.evaluator.compiler_validation,
        )
        model = TaggedCauchyImageLinearModel(evaluator, self.beta_by_species, self.offsets)
        model.reference_terms = dict(self.reference_terms)
        return YE3TTaggedCauchyCalculator(model)

    def export_lammps(self, path, *, format="ye3t"):
        """Export a hash-bound tagged bundle for ``pair_style ye3t``.

        The bundle carries the candidate schedules used by the CPU ``auto``
        policy. A ``yace`` request is accepted only after a certified lowering
        exists for every selected feature.
        """

        if format != "ye3t":
            raise ValueError("This tagged feature space has no certified YACE lowering.")
        return export_tagged_cauchy_image_model(path, self)


class YE3TTaggedCauchyCalculator(Calculator):
    """ASE calculator adapter for a fitted tagged-Cauchy linear model."""

    implemented_properties = ["energy", "energies", "forces", "stress"]

    def __init__(self, model, native_runtime=None):
        super().__init__()
        self.tagged_model = model
        self.native_runtime = native_runtime

    @classmethod
    def from_artifact(cls, path, *, native_library=None, execution_policy="direct"):
        """Load a fitted tagged or tagged-plus-ACE artifact into the C++ ASE calculator."""
        from ye3t_ace.tagged_cauchy_native import _TaggedCauchyNativeRuntime

        runtime = _TaggedCauchyNativeRuntime(
            path, library_path=native_library, execution_policy=execution_policy)
        return cls(None, native_runtime=runtime)

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes)
        current = self.atoms
        symbols = current.get_chemical_symbols()
        type_map = (self.tagged_model.evaluator.type_map if self.tagged_model is not None
                    else self.native_runtime.type_map)
        unknown = sorted(set(symbols) - set(type_map))
        if unknown:
            raise ValueError(f"Tagged model contains unknown species: {unknown}.")
        if self.native_runtime is None:
            atom_types = [type_map[symbol] for symbol in symbols]
            with torch.no_grad():
                energy, forces, virial, atomic = self.tagged_model.energy_forces_virial(
                    np.asarray(current.positions, dtype=np.float64), atom_types,
                    cell=np.asarray(current.cell.array, dtype=np.float64), pbc=current.pbc,
                )
            atomic = atomic.detach().cpu().numpy()
            forces = forces.detach().cpu().numpy()
            virial = virial.detach().cpu().numpy()
        else:
            energy, forces, virial, atomic = self.native_runtime.evaluate_atoms(current)
        self.results = {"energy": float(energy), "energies": atomic, "forces": forces}
        volume = float(current.get_volume()) if current.cell.rank == 3 else 0.0
        if volume > 0.0:
            values = virial
            # The model stores (xx, yy, zz, xy, xz, yz); ASE uses
            # (xx, yy, zz, yz, xz, xy), with stress = -virial / volume.
            self.results["stress"] = -values[[0, 1, 2, 5, 4, 3]] / volume
        elif "stress" in properties:
            raise ValueError("Stress requires a cell with positive volume.")


def export_tagged_cauchy_image_model(path, model, fit_metadata=None):
    """Write the hash-bound Python/native-CPU artifact, including fixed references."""

    evaluator = model.evaluator
    if fit_metadata is None:
        fit_metadata = getattr(model, "fit_metadata", {})
    source_binding = _binding(evaluator.source_plan)
    schedule_binding = _binding(evaluator.program)
    readout = {
        "schema": TAGGED_CAUCHY_READOUT_SCHEMA,
        "species_order": list(model.species_order),
        "feature_count": int(evaluator.feature_count),
        "beta": {
            name: model.beta_by_species[name].detach().cpu().tolist()
            for name in model.species_order
        },
        "offsets": dict(model.offsets),
    }
    general = evaluator.compiled.payload.get("coordinate_policy") == "exact_physical_pivots_v1"
    if general or model.reference_terms:
        readout["reference_terms"] = _jsonable(model.reference_terms)
        if readout["reference_terms"].get("zbl"):
            from ye3t_ace.reference_potentials import _portable_zbl_metadata
            readout["reference_terms"]["zbl"] = _portable_zbl_metadata(readout["reference_terms"]["zbl"])
    readout_binding = _binding(readout)
    compiler_payload = evaluator.compiled.to_dict()
    body = {
        "schema": TAGGED_CAUCHY_IMAGE_MODEL_SCHEMA_V4 if general or model.reference_terms else TAGGED_CAUCHY_IMAGE_MODEL_SCHEMA,
        "model_family": "linear_tagged_cauchy_image",
        "compiler_artifact": compiler_payload,
        "compiler_artifact_hash": str(evaluator.compiled.self_hash),
        "source_binding": source_binding,
        "schedule_binding": schedule_binding,
        "readout_binding": readout_binding,
        "conventions": dict(TAGGED_CAUCHY_V3_CONVENTIONS),
        "fit_metadata": _jsonable(fit_metadata),
        "tagged_execution_portfolio": compile_tagged_moment_execution_portfolio(
            evaluator.program, readout["beta"]
        ),
    }
    body["deployment_identity_hash"] = _payload_hash(_deployment_identity(body))
    payload = {**body, "self_hash": _payload_hash(body)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def load_tagged_cauchy_image_model(path, compiler_validation="full"):
    """Load a model, optionally trusting its stored V4 algebraic certificate.

    ``certificate`` skips symmetry/physical-image reconstruction, preserving
    integrity, coefficient, source, readout and executable-adjoint checks.
    ``full`` (default) additionally reconstructs the compiler proof.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(payload.pop("self_hash", ""))
    if not expected or expected != _payload_hash(payload):
        raise ValueError("Tagged-Cauchy V3 model self_hash mismatch.")
    if payload.get("schema") not in {TAGGED_CAUCHY_IMAGE_MODEL_SCHEMA, TAGGED_CAUCHY_IMAGE_MODEL_SCHEMA_V4}:
        raise ValueError("Unsupported tagged-Cauchy V3 model schema.")
    if str(payload.get("deployment_identity_hash", "")) != _payload_hash(
        _deployment_identity(payload)
    ):
        raise ValueError("Tagged-Cauchy V3 deployment identity mismatch.")
    if payload.get("conventions") != TAGGED_CAUCHY_V3_CONVENTIONS:
        raise ValueError("Unsupported tagged-Cauchy V3 conventions.")
    compiled = CompiledTaggedCauchyImage.from_dict(payload["compiler_artifact"], compiler_validation=compiler_validation)
    if str(compiled.self_hash) != str(payload["compiler_artifact_hash"]):
        raise ValueError("Tagged-Cauchy V3 compiler binding changed.")
    source_plan = _validate_binding(payload["source_binding"], "source")
    real_program = _validate_binding(payload["schedule_binding"], "schedule")
    readout = _validate_binding(payload["readout_binding"], "readout")
    evaluator = TaggedCauchyImageEvaluator(compiled, source_plan, real_program,
                                         compiler_validation=compiler_validation)
    if readout.get("schema") != TAGGED_CAUCHY_READOUT_SCHEMA:
        raise ValueError("Unsupported tagged-Cauchy readout schema.")
    if tuple(readout["species_order"]) != evaluator.species_order:
        raise ValueError("Tagged-Cauchy readout/source species ordering changed.")
    if int(readout["feature_count"]) != evaluator.feature_count:
        raise ValueError("Tagged-Cauchy readout feature width changed.")
    model = TaggedCauchyImageLinearModel(
        evaluator, readout["beta"], readout["offsets"]
    )
    model.reference_terms = dict(readout.get("reference_terms", {}))
    if model.reference_terms:
        references = model.reference_terms.get("atomic_energies", {})
        if set(references) != set(model.species_order) or not all(np.isfinite(float(value)) for value in references.values()):
            raise ValueError("Tagged-Cauchy atomic references must cover species_order with finite values.")
        if model.reference_terms.get("zbl") is not None:
            from ye3t_ace.reference_potentials import _portable_zbl_metadata
            zbl = model.reference_terms["zbl"]
            if _portable_zbl_metadata(zbl) != zbl or set(zbl["atomic_numbers"]) != set(model.species_order):
                raise ValueError("Tagged-Cauchy ZBL reference convention or species binding changed.")
    fit_metadata = payload.get("fit_metadata", {})
    if not isinstance(fit_metadata, dict):
        raise ValueError("Tagged-Cauchy V3 fit_metadata must be a mapping.")
    model.fit_metadata = dict(fit_metadata)
    model._ye3t_linear_fit_metadata = dict(fit_metadata)
    return model


__all__ = [
    "TAGGED_CAUCHY_IMAGE_MODEL_SCHEMA",
    "TaggedCauchyImageEvaluator",
    "TaggedCauchyImageLinearModel",
    "export_tagged_cauchy_image_model",
    "load_tagged_cauchy_image_model",
    "realify_tagged_cauchy_image",
    "tagged_cauchy_source_plan",
]
