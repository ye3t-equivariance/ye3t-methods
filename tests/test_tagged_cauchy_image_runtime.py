import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

import ye3t.couplings as coupling_api
from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import plan as plan_coupling
from ye3t.couplings import racah_harmonic_product_plan
from ye3t.couplings import tagged_cauchy_image_request
from ye3t.couplings.lifted_cauchy_scalar import (
    _exact_matrix_from_payload,
    _exact_scalar_from_payload,
)
from ye3t.couplings.orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    build_radial_species_product_record,
    shifted_jacobi_ladder_with_derivative,
    shifted_jacobi_power_coefficients,
)
from ye3t_methods.atomistic.tagged_cauchy_image import (
    TaggedCauchyImageEvaluator,
    TaggedCauchyImageLinearModel,
    _payload_hash,
    export_tagged_cauchy_image_model,
    load_tagged_cauchy_image_model,
    realify_tagged_cauchy_image,
    tagged_cauchy_source_plan,
)
from ye3t_methods.atomistic.ace.descriptors import YE3TDescriptors, YE3TModel
from ye3t_methods.atomistic.representations import YE3TRepresentation
from ye3t_methods.atomistic.tagged_cauchy_image_fit import (
    build_tagged_cauchy_image_normal_equations,
    score_tagged_cauchy_image_model,
    tagged_cauchy_reference_target_metadata,
)


@pytest.fixture(scope="module")
def compiled_tagged_image(tmp_path_factory):
    cache = tmp_path_factory.mktemp("tagged_image_cache")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("YE3T_CACHE_DIR", str(cache))
        angular = racah_harmonic_product_plan(
            (1,), maximum_collision_arity=2
        )
        source = {
            "neighbor_species": "Ta",
            "q": 0,
            "l": 1,
            "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
        }
        product = build_radial_species_product_record(
            (source,), angular, maximum_collision_arity=2
        )
        request = tagged_cauchy_image_request(
            product, angular, source_key=source
        )
        return compile_coupling(plan_coupling(request))


def _evaluator(compiled):
    program = realify_tagged_cauchy_image(compiled)
    source = tagged_cauchy_source_plan(compiled, 4.8, program)
    return TaggedCauchyImageEvaluator(compiled, source, program)


def _exact_power_value_derivative(coefficients, coordinate):
    coordinate = Fraction(coordinate)
    value = sum(
        Fraction(coefficient) * coordinate**power
        for power, coefficient in enumerate(coefficients)
    )
    derivative = sum(
        power * Fraction(coefficient) * coordinate ** (power - 1)
        for power, coefficient in enumerate(coefficients)
        if power
    )
    return float(value), float(derivative)


def test_shifted_jacobi_stable_recurrence_matches_exact_polynomials():
    coordinates = (
        Fraction(0),
        Fraction(1, 1000),
        Fraction(1, 7),
        Fraction(1, 2),
        Fraction(999, 1000),
        Fraction(1),
    )
    for angular_l in (0, 1, 2, 9):
        beta = 2 * angular_l + 2
        for coordinate in coordinates:
            values, derivatives = shifted_jacobi_ladder_with_derivative(
                18, 4, beta, float(coordinate)
            )
            for degree in range(19):
                exact = _exact_power_value_derivative(
                    shifted_jacobi_power_coefficients(degree, 4, beta),
                    coordinate,
                )
                np.testing.assert_allclose(
                    (values[degree], derivatives[degree]),
                    exact,
                    rtol=3.0e-12,
                    atol=3.0e-10,
                )
    values, derivatives = shifted_jacobi_ladder_with_derivative(2, 4, 2, 0.37)
    np.testing.assert_allclose(values[0], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(values[1], 8.0 * 0.37 - 3.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        values[2], 6.0 - 36.0 * 0.37 + 45.0 * 0.37**2, rtol=0.0, atol=2.0e-15
    )
    np.testing.assert_allclose(derivatives[2], -36.0 + 90.0 * 0.37, atol=2.0e-14)


def test_v2_source_plan_is_default_and_v1_replays(compiled_tagged_image):
    program = realify_tagged_cauchy_image(compiled_tagged_image)
    stable = tagged_cauchy_source_plan(compiled_tagged_image, 4.8, program)
    legacy = tagged_cauchy_source_plan(
        compiled_tagged_image,
        4.8,
        program,
        numerical_realization="legacy_expanded_power_horner_v1",
    )
    assert stable["schema"] == "ye3t_tagged_cauchy_direct_source_v2"
    assert stable["numerical_evaluation"] == {
        "schema": "ye3t_shifted_jacobi_three_term_v1",
        "alpha": 4,
        "beta_rule": "2*l+2",
        "argument": "2*x-1",
        "derivative": "d/dx",
        "evaluation_order": "per_edge_per_l_ladder",
        "expanded_power_coefficients_use": "provenance_only",
    }
    assert legacy["schema"] == "ye3t_tagged_cauchy_direct_source_v1"
    assert "numerical_evaluation" not in legacy
    assert stable["compiler_artifact_hash"] == legacy["compiler_artifact_hash"]
    assert stable["source_product_algebra_hash"] == legacy[
        "source_product_algebra_hash"
    ]
    assert stable["source_plan_hash"] != legacy["source_plan_hash"]

    stable_evaluator = TaggedCauchyImageEvaluator(
        compiled_tagged_image, stable, program
    )
    legacy_evaluator = TaggedCauchyImageEvaluator(
        compiled_tagged_image, legacy, program
    )
    displacements = torch.tensor(
        [[1.1, 0.2, -0.1], [-0.4, 1.3, 0.5], [0.3, -0.7, 1.5]],
        dtype=torch.float64,
    )
    types = torch.zeros(3, dtype=torch.long)
    stable_values = stable_evaluator.edge_sources_with_derivatives(
        displacements, types
    )
    legacy_values = legacy_evaluator.edge_sources_with_derivatives(
        displacements, types
    )
    for stable_value, legacy_value in zip(stable_values, legacy_values):
        torch.testing.assert_close(
            stable_value, legacy_value, rtol=2.0e-13, atol=2.0e-13
        )


def _complex_schedule_values(compiled, evaluator, displacements):
    edge_count = int(displacements.shape[0])
    values, _derivatives = evaluator.edge_sources_with_derivatives(
        displacements, torch.zeros(edge_count, dtype=torch.long)
    )
    channel_of = {
        (
            str(channel["neighbor_species"]),
            str(channel["source_family_id"]),
            str(channel["support_id"]),
            int(channel["q"]),
            int(channel["l"]),
        ): index
        for index, channel in enumerate(evaluator.channels)
    }
    forms = {
        int(record["angular_l"]): record
        for record in compiled.payload["real_forms"]
    }
    moments = values.sum(dim=0)
    outputs = []
    for schedule in compiled.payload["moment_schedules"]:
        generators = []
        for generator in schedule["source_generators"]:
            base = (
                str(generator["neighbor_species"]),
                str(generator["source_family_id"]),
                str(generator["support_id"]),
                int(generator["q"]),
                int(generator["l"]),
            )
            channel = channel_of[base]
            angular_l = int(generator["l"])
            row = int(generator["m"]) + angular_l
            offset = evaluator._component_offsets[channel]
            value = 0.0j
            for component, payload in enumerate(
                forms[angular_l]["real_to_complex_matrix"][row]
            ):
                coefficient = complex(_exact_scalar_from_payload(payload).evalf(17))
                value += coefficient * float(moments[offset + component])
            generators.append(value)
        output = 0.0j
        for term in schedule["forward_terms"]:
            coefficient = complex(
                _exact_scalar_from_payload(term["coefficient"]).evalf(17)
            )
            product = coefficient
            for index in term["source_indices"]:
                product *= generators[int(index)]
            output += product
        outputs.append(output)
    return np.asarray(outputs)


def _direct_tuple_two_tag_value(compiled, evaluator, displacements):
    edge_count = int(displacements.shape[0])
    values, _derivatives = evaluator.edge_sources_with_derivatives(
        displacements, torch.zeros(edge_count, dtype=torch.long)
    )
    companion = compiled.payload["physical_companions"][0]
    source_key = companion["source_key"]
    channel = next(
        index
        for index, record in enumerate(evaluator.channels)
        if str(record["neighbor_species"]) == str(source_key["neighbor_species"])
        and str(record["source_family_id"]) == str(source_key["source_family_id"])
        and str(record["support_id"]) == str(source_key["support_id"])
        and int(record["q"]) == int(source_key["q"])
        and int(record["l"]) == int(source_key["l"])
    )
    angular_l = int(source_key["l"])
    form = next(
        record
        for record in compiled.payload["real_forms"]
        if int(record["angular_l"]) == angular_l
    )
    offset = evaluator._component_offsets[channel]
    edge_sources = []
    for edge in range(edge_count):
        components = {}
        for magnetic in range(-angular_l, angular_l + 1):
            row = magnetic + angular_l
            value = 0.0j
            for component, payload in enumerate(
                form["real_to_complex_matrix"][row]
            ):
                coefficient = complex(
                    _exact_scalar_from_payload(payload).evalf(17)
                )
                value += coefficient * float(values[edge, offset + component])
            components[magnetic] = value
        edge_sources.append(components)
    if edge_count < 2:
        return 0.0j
    density = {
        magnetic: sum(source[magnetic] for source in edge_sources)
        for magnetic in range(-angular_l, angular_l + 1)
    }
    total = 0.0j
    for first in range(edge_count):
        for second in range(edge_count):
            if first == second:
                continue
            sources = {0: edge_sources[first], 1: edge_sources[second], 2: density}
            for term in companion["paired_source_terms"]:
                value = complex(
                    _exact_scalar_from_payload(term["coefficient"]).evalf(17)
                )
                for role, magnetic in zip(
                    term["role_word"], term["magnetic_tuple"], strict=True
                ):
                    value *= sources[int(role)][int(magnetic)]
                total += value
    return total


def test_real_schedule_matches_compiler_and_distinct_tag_neighbor_counts(
    compiled_tagged_image,
):
    evaluator = _evaluator(compiled_tagged_image)
    program = dict(evaluator.program)
    program_hash = program.pop("program_hash")
    assert program_hash == compiled_tagged_image.payload[
        "real_schedule_core_hash"
    ]
    assert program == compiled_tagged_image.payload["real_schedule_core"]
    clusters = (
        [[1.1, 0.2, -0.1]],
        [[1.1, 0.2, -0.1], [-0.4, 1.3, 0.5]],
        [[1.1, 0.2, -0.1], [-0.4, 1.3, 0.5], [0.3, -0.7, 1.5]],
    )
    for displacements_raw in clusters:
        displacements = torch.tensor(displacements_raw, dtype=torch.float64)
        edge_index = torch.stack(
            (
                torch.zeros(len(displacements_raw), dtype=torch.long),
                torch.arange(1, len(displacements_raw) + 1, dtype=torch.long),
            )
        )
        features, derivatives = evaluator.evaluate_edge_list(
            edge_index,
            displacements,
            torch.zeros(len(displacements_raw), dtype=torch.long),
            len(displacements_raw) + 1,
        )
        compiler = _complex_schedule_values(
            compiled_tagged_image, evaluator, displacements
        )
        np.testing.assert_allclose(
            features[0].detach().numpy(), compiler.real, rtol=2.0e-12, atol=2.0e-12
        )
        assert np.max(np.abs(compiler.imag)) < 2.0e-12
        raw_from_image = np.asarray(
            _exact_matrix_from_payload(
                compiled_tagged_image.payload["raw_from_image"]
            ).evalf(17),
            dtype=np.float64,
        )
        raw_two_tag = raw_from_image[2] @ features[0].detach().numpy()
        direct_two_tag = _direct_tuple_two_tag_value(
            compiled_tagged_image, evaluator, displacements
        )
        np.testing.assert_allclose(
            raw_two_tag, direct_two_tag.real, rtol=3.0e-12, atol=3.0e-12
        )
        assert abs(direct_two_tag.imag) < 3.0e-12
        step = 2.0e-6
        numerical = np.zeros_like(derivatives.detach().numpy())
        direct_numerical = np.zeros((len(displacements_raw), 3))
        for edge in range(len(displacements_raw)):
            for axis in range(3):
                plus = displacements.clone()
                minus = displacements.clone()
                plus[edge, axis] += step
                minus[edge, axis] -= step
                plus_value = evaluator.evaluate_edge_list(
                    edge_index,
                    plus,
                    torch.zeros(len(displacements_raw), dtype=torch.long),
                    len(displacements_raw) + 1,
                )[0]
                minus_value = evaluator.evaluate_edge_list(
                    edge_index,
                    minus,
                    torch.zeros(len(displacements_raw), dtype=torch.long),
                    len(displacements_raw) + 1,
                )[0]
                numerical[edge, :, axis] = (
                    (plus_value[0] - minus_value[0]).detach().numpy()
                    / (2.0 * step)
                )
                plus_direct = _direct_tuple_two_tag_value(
                    compiled_tagged_image, evaluator, plus
                )
                minus_direct = _direct_tuple_two_tag_value(
                    compiled_tagged_image, evaluator, minus
                )
                direct_numerical[edge, axis] = (
                    (plus_direct - minus_direct).real / (2.0 * step)
                )
        np.testing.assert_allclose(
            derivatives.detach().numpy(),
            numerical,
            rtol=3.0e-7,
            atol=3.0e-8,
        )
        raw_two_tag_derivative = np.einsum(
            "f,efd->ed",
            raw_from_image[2],
            derivatives.detach().numpy(),
        )
        np.testing.assert_allclose(
            raw_two_tag_derivative,
            direct_numerical,
            rtol=4.0e-7,
            atol=4.0e-8,
        )
        if len(displacements_raw) < 2:
            np.testing.assert_allclose(raw_two_tag, 0.0, atol=2.0e-12)
            np.testing.assert_allclose(
                raw_two_tag_derivative, 0.0, atol=2.0e-11
            )
        else:
            assert abs(float(raw_two_tag)) > 1.0e-10


def test_cartesian_vjp_reorder_and_exact_overlap_policy(compiled_tagged_image):
    evaluator = _evaluator(compiled_tagged_image)
    displacements = torch.tensor(
        [[1.1, 0.2, -0.1], [-0.4, 1.3, 0.5], [0.3, -0.7, 1.5]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    types = torch.zeros(3, dtype=torch.long)
    values, derivatives = evaluator.evaluate_edge_list(
        edge_index, displacements, types, 4
    )
    step = 2.0e-6
    numerical = np.zeros_like(derivatives.detach().numpy())
    for edge in range(3):
        for axis in range(3):
            plus = displacements.clone()
            minus = displacements.clone()
            plus[edge, axis] += step
            minus[edge, axis] -= step
            plus_value = evaluator.evaluate_edge_list(edge_index, plus, types, 4)[0]
            minus_value = evaluator.evaluate_edge_list(edge_index, minus, types, 4)[0]
            numerical[edge, :, axis] = (
                (plus_value[0] - minus_value[0]).detach().numpy() / (2.0 * step)
            )
    np.testing.assert_allclose(
        derivatives.detach().numpy(), numerical, rtol=2.0e-7, atol=2.0e-8
    )

    order = torch.tensor([2, 0, 1])
    reordered = evaluator.evaluate_edge_list(
        edge_index[:, order], displacements[order], types[order], 4
    )[0]
    torch.testing.assert_close(reordered, values, rtol=1.0e-13, atol=1.0e-13)

    with pytest.raises(ValueError, match="zero separation"):
        evaluator.evaluate_edge_list(
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.float64),
            torch.zeros(1, dtype=torch.long),
            2,
        )

    outside_values, outside_derivatives = evaluator.edge_sources_with_derivatives(
        torch.tensor([[1.0e100, -2.0e99, 3.0e99]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.long),
    )
    assert bool(torch.all(torch.isfinite(outside_values)))
    assert bool(torch.all(torch.isfinite(outside_derivatives)))
    torch.testing.assert_close(
        outside_values, torch.zeros_like(outside_values), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        outside_derivatives,
        torch.zeros_like(outside_derivatives),
        rtol=0.0,
        atol=0.0,
    )


def test_rotation_inversion_and_zero_factor_division_free_adjoint(
    compiled_tagged_image,
):
    evaluator = _evaluator(compiled_tagged_image)
    displacements = torch.tensor(
        [[1.1, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 1.5]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    types = torch.zeros(3, dtype=torch.long)
    source_values = evaluator.edge_sources_with_derivatives(displacements, types)[0]
    assert bool(torch.any(source_values == 0.0))
    values, derivatives = evaluator.evaluate_edge_list(
        edge_index, displacements, types, 4
    )
    assert bool(torch.all(torch.isfinite(derivatives)))

    angle = 0.731
    axis = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    axis = axis / torch.linalg.norm(axis)
    cross = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=torch.float64,
    )
    rotation = (
        torch.eye(3, dtype=torch.float64) * math.cos(angle)
        + (1.0 - math.cos(angle)) * torch.outer(axis, axis)
        + math.sin(angle) * cross
    )
    rotated = displacements @ rotation.T
    rotated_values, rotated_derivatives = evaluator.evaluate_edge_list(
        edge_index, rotated, types, 4
    )
    torch.testing.assert_close(rotated_values, values, rtol=2.0e-12, atol=2.0e-12)
    torch.testing.assert_close(
        rotated_derivatives,
        derivatives @ rotation.T,
        rtol=3.0e-11,
        atol=3.0e-11,
    )

    inverted_values, inverted_derivatives = evaluator.evaluate_edge_list(
        edge_index, -displacements, types, 4
    )
    torch.testing.assert_close(inverted_values, values, rtol=2.0e-12, atol=2.0e-12)
    torch.testing.assert_close(
        inverted_derivatives, -derivatives, rtol=2.0e-12, atol=2.0e-12
    )

    step = 2.0e-6
    numerical = np.zeros_like(derivatives.numpy())
    for edge in range(3):
        for direction in range(3):
            plus = displacements.clone()
            minus = displacements.clone()
            plus[edge, direction] += step
            minus[edge, direction] -= step
            plus_value = evaluator.evaluate_edge_list(
                edge_index, plus, types, 4
            )[0]
            minus_value = evaluator.evaluate_edge_list(
                edge_index, minus, types, 4
            )[0]
            numerical[edge, :, direction] = (
                (plus_value[0] - minus_value[0]).numpy() / (2.0 * step)
            )
    np.testing.assert_allclose(
        derivatives.numpy(), numerical, rtol=3.0e-7, atol=3.0e-8
    )


def test_v3_round_trip_energy_force_virial_and_tamper(
    compiled_tagged_image, tmp_path
):
    evaluator = _evaluator(compiled_tagged_image)
    model = TaggedCauchyImageLinearModel(
        evaluator,
        {"Ta": torch.tensor([0.7, -0.2], dtype=torch.float64)},
        {"Ta": -0.3},
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [-0.4, 1.3, 0.5], [0.3, -0.7, 1.5]],
        dtype=torch.float64,
    )
    types = torch.zeros(4, dtype=torch.long)
    expected = model.energy_forces_virial(
        positions, types, pbc=(False, False, False)
    )
    path = tmp_path / "tagged_v3.json"
    artifact = model.export_lammps(path)
    assert artifact["tagged_execution_portfolio"]["certificate"]["passed"]
    assert {entry["candidate_id"] for entry in artifact["tagged_execution_portfolio"]["candidates"]} == {
        "compiled_direct", "generic_dag", "symmetric_power", "block"}
    loaded = load_tagged_cauchy_image_model(path)
    actual = loaded.energy_forces_virial(
        positions, types, pbc=(False, False, False)
    )
    for expected_value, actual_value in zip(expected, actual):
        torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)

    step = 1.0e-6
    numerical_force = torch.zeros_like(positions)
    for atom in range(4):
        for axis in range(3):
            plus = positions.clone()
            minus = positions.clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            e_plus = loaded.energy_forces_virial(
                plus, types, pbc=(False, False, False)
            )[0]
            e_minus = loaded.energy_forces_virial(
                minus, types, pbc=(False, False, False)
            )[0]
            numerical_force[atom, axis] = -(e_plus - e_minus) / (2.0 * step)
    torch.testing.assert_close(actual[1], numerical_force, rtol=2.0e-7, atol=3.0e-8)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_binding"]["payload"]["cutoff"] += 0.1
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    payload["self_hash"] = _payload_hash(body)
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_tagged_cauchy_image_model(tampered)


    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["deployment_identity_hash"] = "0" * 64
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    payload["self_hash"] = _payload_hash(body)
    tampered = tmp_path / "tampered_deployment_identity.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="deployment identity mismatch"):
        load_tagged_cauchy_image_model(tampered)

    def rehash_binding(model_payload, name, self_hash_name):
        nested = model_payload[name]["payload"]
        nested_body = {
            key: value for key, value in nested.items() if key != self_hash_name
        }
        nested[self_hash_name] = _payload_hash(nested_body)
        model_payload[name]["hash"] = _payload_hash(nested)
        identity = {
            "compiler_artifact_hash": model_payload["compiler_artifact_hash"],
            "source_plan_hash": model_payload["source_binding"]["hash"],
            "schedule_hash": model_payload["schedule_binding"]["hash"],
            "readout_hash": model_payload["readout_binding"]["hash"],
            "conventions": model_payload["conventions"],
        }
        model_payload["deployment_identity_hash"] = _payload_hash(identity)
        model_body = {
            key: value for key, value in model_payload.items() if key != "self_hash"
        }
        model_payload["self_hash"] = _payload_hash(model_body)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schedule_binding"]["payload"]["channels"][0]["q"] += 1
    rehash_binding(payload, "schedule_binding", "program_hash")
    tampered = tmp_path / "tampered_schedule_channel.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its compiler commitment"):
        load_tagged_cauchy_image_model(tampered)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schedule_binding"]["payload"]["terms"][0]["coefficient"] += 0.125
    rehash_binding(payload, "schedule_binding", "program_hash")
    tampered = tmp_path / "tampered_forward_adjoint.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its compiler commitment"):
        load_tagged_cauchy_image_model(tampered)

    payload = json.loads(path.read_text(encoding="utf-8"))
    for term in payload["schedule_binding"]["payload"]["terms"]:
        term["coefficient"] *= 2.0
    for term in payload["schedule_binding"]["payload"]["adjoint_terms"]:
        term["coefficient"] *= 2.0
    rehash_binding(payload, "schedule_binding", "program_hash")
    tampered = tmp_path / "tampered_coherent_forward_adjoint.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its compiler commitment"):
        load_tagged_cauchy_image_model(tampered)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_binding"]["payload"]["channels"][0][
        "binary64_power_coefficients"
    ][0] += 0.125
    rehash_binding(payload, "source_binding", "source_plan_hash")
    tampered = tmp_path / "tampered_binary64_source.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="binary64 radial coefficients changed"):
        load_tagged_cauchy_image_model(tampered)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_binding"]["payload"]["radial_coordinate"] = "r"
    rehash_binding(payload, "source_binding", "source_plan_hash")
    tampered = tmp_path / "tampered_source_semantics.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source semantics are unsupported"):
        load_tagged_cauchy_image_model(tampered)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_binding"]["payload"]["certificate"][
        "runtime_gram_solve"
    ] = True
    rehash_binding(payload, "source_binding", "source_plan_hash")
    tampered = tmp_path / "tampered_source_certificate.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source plan is uncertified"):
        load_tagged_cauchy_image_model(tampered)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conventions"]["lammps_virial"] = "plus_strain_derivative"
    identity = {
        "compiler_artifact_hash": payload["compiler_artifact_hash"],
        "source_plan_hash": payload["source_binding"]["hash"],
        "schedule_hash": payload["schedule_binding"]["hash"],
        "readout_hash": payload["readout_binding"]["hash"],
        "conventions": payload["conventions"],
    }
    payload["deployment_identity_hash"] = _payload_hash(identity)
    model_body = {
        key: value for key, value in payload.items() if key != "self_hash"
    }
    payload["self_hash"] = _payload_hash(model_body)
    tampered = tmp_path / "tampered_conventions.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported tagged-Cauchy V3 conventions"):
        load_tagged_cauchy_image_model(tampered)


def test_native_ase_tagged_matches_reference(compiled_tagged_image):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    evaluator = _evaluator(compiled_tagged_image)
    model = TaggedCauchyImageLinearModel(
        evaluator, {"Ta": torch.tensor([0.7, -0.2], dtype=torch.float64)},
        {"Ta": -0.3},
    )
    atoms = Atoms("Ta4", positions=(
        (0.0, 0.0, 0.0), (1.1, 0.2, -0.1), (-0.4, 1.3, 0.5),
        (0.3, -0.7, 1.5),
    ), cell=(9.0, 9.0, 9.0), pbc=True)
    for policy in ("direct", "auto"):
        reference = atoms.copy()
        reference.calc = model.ase_calculator(backend="reference")
        native = atoms.copy()
        native.calc = model.ase_calculator(
            backend="native_cpu", native_library=library,
            execution_policy=policy,
        )
        np.testing.assert_allclose(native.get_potential_energy(),
                                   reference.get_potential_energy(), rtol=2.0e-10, atol=2.0e-10)
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                                   rtol=2.0e-9, atol=2.0e-9)
        np.testing.assert_allclose(native.get_stress(), reference.get_stress(),
                                   rtol=2.0e-9, atol=2.0e-9)
        runtime = native.calc.native_runtime
        expected_neighbors = ("matscipy_neighbor_list"
                              if importlib.util.find_spec("matscipy") is not None
                              else "ase_neighbor_list")
        assert runtime.last_neighbor_backend == expected_neighbors
        rebuilds = runtime.topology_rebuilds
        native.positions[0, 0] += 0.01
        reference.positions[0, 0] += 0.01
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                                   rtol=2.0e-9, atol=2.0e-9)
        assert runtime.topology_rebuilds == rebuilds
        native.positions[0, 0] += 0.2
        reference.positions[0, 0] += 0.2
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                                   rtol=2.0e-9, atol=2.0e-9)
        assert runtime.topology_rebuilds == rebuilds + 1
        from ase import units
        from ase.md.verlet import VelocityVerlet

        velocities = np.array(((0.01, 0.02, 0.0), (-0.01, 0.0, 0.01),
                               (0.0, -0.01, 0.02), (0.01, 0.0, -0.01)))
        native.set_velocities(velocities)
        moved_reference = reference.copy()
        moved_reference.calc = model.ase_calculator(backend="reference")
        moved_reference.set_velocities(velocities)
        VelocityVerlet(native, timestep=0.2 * units.fs).run(2)
        VelocityVerlet(moved_reference, timestep=0.2 * units.fs).run(2)
        np.testing.assert_allclose(native.positions, moved_reference.positions,
                                   rtol=2.0e-9, atol=2.0e-9)
        fast = atoms.copy()
        fast.set_cell((20.0, 20.0, 20.0), scale_atoms=False)
        fast.positions[1] = (19.6, 0.2, -0.1)
        fast_reference = fast.copy()
        fast.calc = model.ase_calculator(backend="native_cpu", native_library=library,
                                         execution_policy=policy)
        fast_reference.calc = model.ase_calculator(backend="reference")
        np.testing.assert_allclose(fast.get_potential_energy(), fast_reference.get_potential_energy(),
                                   rtol=2.0e-10, atol=2.0e-10)
        np.testing.assert_allclose(fast.get_forces(), fast_reference.get_forces(),
                                   rtol=2.0e-9, atol=2.0e-9)
        if importlib.util.find_spec("scipy") is not None:
            assert fast.calc.native_runtime.last_neighbor_backend == "scipy_ckdtree_periodic"
        fast.positions[1, 0] += 1.2  # Cross the periodic boundary and rebuild the cached list.
        fast_reference.positions[1, 0] += 1.2
        np.testing.assert_allclose(fast.get_forces(), fast_reference.get_forces(),
                                   rtol=2.0e-9, atol=2.0e-9)
        fast.set_pbc(False)
        fast_reference.set_pbc(False)
        np.testing.assert_allclose(fast.get_forces(), fast_reference.get_forces(),
                                   rtol=2.0e-9, atol=2.0e-9)
        if importlib.util.find_spec("scipy") is not None:
            assert fast.calc.native_runtime.last_neighbor_backend == "scipy_ckdtree_nonperiodic"


@pytest.mark.parametrize("neighbors", ("ase", "matscipy"))
def test_native_tagged_adversarial_periodic_geometry_matches_reference(
    compiled_tagged_image, neighbors,
):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    if neighbors == "matscipy":
        pytest.importorskip("matscipy", reason="requires optional matscipy neighbor dependency")
    evaluator = _evaluator(compiled_tagged_image)
    model = TaggedCauchyImageLinearModel(
        evaluator, {"Ta": torch.tensor([0.7, -0.2], dtype=torch.float64)},
        {"Ta": -0.3},
    )
    positions = ((0.1, 0.2, 0.3), (1.6, 0.6, 0.5), (0.7, 2.2, 1.1))
    cases = (
        Atoms("Ta3", positions=positions,
              cell=((3.2, 0, 0), (9.6, 3.2, 0), (0, 0, 7.0)), pbc=True),
        Atoms("Ta3", positions=positions,
              cell=((3.2, 0, 0), (0, 3.2, 0), (0, 0, 7.0)), pbc=True),
        Atoms("Ta3", positions=positions, cell=(3.8, 3.9, 4.0), pbc=True),
        Atoms("Ta3", positions=positions, cell=(3.8, 3.9, 8.0),
              pbc=(True, False, True)),
    )
    from ase.neighborlist import neighbor_list
    centers, neighbors_index, shifts = neighbor_list("ijS", cases[2], evaluator.cutoff)
    repeated = shifts[(centers == 0) & (neighbors_index == 1)]
    assert len({tuple(shift) for shift in repeated}) >= 2
    values = []
    for atoms in cases:
        reference = atoms.copy()
        reference.calc = model.ase_calculator(backend="reference")
        native = atoms.copy()
        native.calc = model.ase_calculator(
            backend="native_cpu", native_library=library, neighbors=neighbors,
        )
        np.testing.assert_allclose(native.get_potential_energy(),
                                   reference.get_potential_energy(), rtol=0, atol=2e-9)
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                                   rtol=0, atol=2e-8)
        np.testing.assert_allclose(native.get_stress(), reference.get_stress(),
                                   rtol=0, atol=2e-8)
        values.append((native.get_potential_energy(), native.get_forces(),
                       native.get_stress()))
    np.testing.assert_allclose(values[0][0], values[1][0], rtol=0, atol=2e-9)
    np.testing.assert_allclose(values[0][1], values[1][1], rtol=0, atol=2e-8)
    np.testing.assert_allclose(values[0][2], values[1][2], rtol=0, atol=2e-8)


def test_bruteforce_image_inventory_matches_ase_for_unwrapped_and_partial_cells():
    from ase.neighborlist import neighbor_list
    from ye3t_methods.atomistic.equivariant_calc.edge_geometry import directed_edges_all_images_bruteforce

    positions = ((0.1, 0.2, 0.3), (11.2, 0.6, 0.5), (0.7, 2.2, 1.1))
    cases = (
        Atoms("Ta3", positions=positions,
              cell=((3.2, 0, 0), (9.6, 3.2, 0), (0, 0, 7.0)), pbc=True),
        Atoms("Ta3", positions=positions,
              cell=((3.8, 0, 0), (0, 0, 0), (0, 0, 8.0)),
              pbc=(True, False, True)),
    )
    for atoms in cases:
        source, neighbor, displacement, _ = directed_edges_all_images_bruteforce(
            torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64),
            4.8, cell=torch.as_tensor(np.asarray(atoms.cell), dtype=torch.float64),
            pbc=atoms.pbc,
        )
        ase_source, ase_neighbor, shifts = neighbor_list(
            "ijS", atoms, 4.8, self_interaction=False)
        ase_displacement = (atoms.positions[ase_neighbor] - atoms.positions[ase_source]
                            + shifts @ atoms.cell.array)
        identity = lambda i, j, row: (int(i), int(j), *(round(float(value), 8)
                                                      for value in row))
        reference_edges = Counter(identity(i, j, row) for i, j, row in zip(
            source.tolist(), neighbor.tolist(), displacement.tolist()))
        ase_edges = Counter(identity(i, j, row) for i, j, row in zip(
            ase_source, ase_neighbor, ase_displacement))
        assert reference_edges == ase_edges


def test_bruteforce_periodic_images_fail_closed_on_ill_conditioned_or_huge_scan():
    from ye3t_methods.atomistic.equivariant_calc.edge_geometry import (
        directed_edges_all_images_bruteforce, periodic_shift_tuples,
    )

    positions = torch.tensor(((0, 0, 0), (0, 0, 0.0002)), dtype=torch.float64)
    cell = torch.diag(torch.tensor((1e12, 1e12, 8e-4), dtype=torch.float64))
    with pytest.raises(ValueError, match="ill-conditioned"):
        directed_edges_all_images_bruteforce(positions, 0.003, cell=cell, pbc=True)
    far = torch.tensor(((0, 0, 0), (600000, 0, 0)), dtype=torch.float64)
    with pytest.raises(MemoryError, match="one million shifts"):
        directed_edges_all_images_bruteforce(
            far, 1.0, cell=torch.eye(3, dtype=torch.float64),
            pbc=(True, False, False),
        )
    huge_origin = np.array(((1e20, 0, 0), (1e20 + 16384, 0, 0)))
    shifts = periodic_shift_tuples(
        np.diag((0.3, 1.0, 1.0)), (True, False, False), 0.2,
        huge_origin,
    )
    required = -int(round((huge_origin[1, 0] - huge_origin[0, 0]) / 0.3))
    assert abs(huge_origin[1, 0] - huge_origin[0, 0] + required * 0.3) < 0.2
    assert required in {int(shift[0]) for shift in shifts}


@pytest.mark.parametrize("neighbors,selected", (
    ("ase", "ase_neighbor_list"),
    ("matscipy", "matscipy_neighbor_list"),
))
def test_native_tagged_explicit_neighbor_policy_matches_reference(
    compiled_tagged_image, tmp_path, neighbors, selected,
):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    if neighbors == "matscipy":
        pytest.importorskip("matscipy", reason="requires optional matscipy neighbor dependency")
    evaluator = _evaluator(compiled_tagged_image)
    model = TaggedCauchyImageLinearModel(
        evaluator, {"Ta": torch.tensor([0.7, -0.2], dtype=torch.float64)},
        {"Ta": -0.3},
    )
    artifact = tmp_path / "tagged.ye3t.json"
    export_tagged_cauchy_image_model(artifact, model)
    from ye3t_methods.atomistic.tagged_cauchy_image import YE3TTaggedCauchyCalculator
    from ye3t_methods import LinearModel

    atoms = Atoms("Ta4", positions=(
        (0.0, 0.0, 0.0), (1.1, 0.2, -0.1), (-0.4, 1.3, 0.5),
        (0.3, -0.7, 1.5),
    ), cell=(9.0, 9.0, 9.0), pbc=True)
    reference = atoms.copy()
    reference.calc = model.ase_calculator(backend="reference")
    native = atoms.copy()
    native.calc = YE3TTaggedCauchyCalculator.from_artifact(
        artifact, native_library=library, neighbors=neighbors,
    )
    np.testing.assert_allclose(native.get_potential_energy(),
                               reference.get_potential_energy(), rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                               rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(native.get_stress(), reference.get_stress(),
                               rtol=2e-9, atol=2e-9)
    public = atoms.copy()
    public.calc = LinearModel.read(artifact).ase_calculator(
        evaluator="auto", neighbors=neighbors, native_library=library,
    )
    np.testing.assert_allclose(public.get_potential_energy(),
                               reference.get_potential_energy(), rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(public.get_forces(), reference.get_forces(),
                               rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(public.get_stress(), reference.get_stress(),
                               rtol=2e-9, atol=2e-9)
    assert public.calc.native_runtime.neighbors == neighbors
    runtime = native.calc.native_runtime
    assert runtime.neighbors == neighbors
    assert runtime.last_neighbor_backend == selected
    rebuilds = runtime.topology_rebuilds
    native.positions[0, 0] += 0.01
    reference.positions[0, 0] += 0.01
    np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                               rtol=2e-9, atol=2e-9)
    assert runtime.topology_rebuilds == rebuilds


def test_native_tagged_explicit_neighbors_error_without_dependency(monkeypatch):
    from ye3t_methods.atomistic.tagged_cauchy_native import _TaggedCauchyNativeRuntime

    monkeypatch.setitem(sys.modules, "matscipy.neighbours", None)
    with pytest.raises(ImportError, match="neighbors='matscipy'"):
        _TaggedCauchyNativeRuntime("unused.ye3t.json", neighbors="matscipy")
    with pytest.raises(ValueError, match="neighbors must be auto, ase, or matscipy"):
        _TaggedCauchyNativeRuntime("unused.ye3t.json", neighbors="unknown")


def test_native_per_atom_features_match_compiled_python():
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    descriptor = YE3TDescriptors.ye3t_basis({
        "elements": ["Ta"],
        "representation": YE3TRepresentation.tagged_cauchy_image(),
        "tagged_cauchy_image": {
            "tensor_order": 4, "selected_raw_tag_counts": [0, 1, 2],
            "source_family": "orthogonal_shifted_jacobi_origin_regular_v1",
            "radial_degrees": [0], "angular_degree": 1, "cutoff_A": 4.8,
            "coefficient_materialization": "compile",
        },
    })
    atoms = Atoms("Ta4", positions=((0.0, 0.0, 0.0), (1.1, 0.2, -0.1),
                                   (-0.4, 1.3, 0.5), (0.3, -0.7, 1.5)))
    expected = descriptor.create(atoms)
    for policy in ("direct", "auto"):
        actual = descriptor.create(atoms, backend="native_cpu", native_library=library,
                                   execution_policy=policy)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-9, atol=2.0e-9)
    assert len(descriptor.metadata["_tagged_native_descriptor_runtimes"]) == 2


def test_rank_four_two_block_angular_sectors_native_symmetry_and_derivatives():
    from ase.stress import voigt_6_to_full_3x3_stress
    from ye3t.couplings import count as count_coupling

    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    catalogue = {
        "nmax_per_rank": {4: 2}, "lmax_per_rank": {4: 1},
        "source_block_partitions_by_rank": {4: [[2, 2]]},
        "angular_patterns_by_rank": {4: [[1, 1, 1, 1]]},
        "tag_counts_by_rank": {4: [2]},
        "max_records_per_rank": 1, "max_features_per_rank": 2,
    }
    report = count_coupling(tagged_cauchy_image_request(
        species=["Ni"], catalogue=catalogue))
    routes = {}
    for row in report.labels:
        label = row["label"]
        if (label["block_sizes"] == (2, 2)
                and label["block_channel_indices"] == (0, 1)
                and label["block_kappas"] == ((2,), (2,))
                and label["role_copy_indices"] == (0, 3)
                and label["block_Lambdas"] in ((0, 0), (2, 2))):
            assert label["block_Lambdas"] not in routes
            routes[label["block_Lambdas"]] = row["coordinate_id"]
    assert set(routes) == {(0, 0), (2, 2)}
    selected_ids = (routes[(0, 0)], routes[(2, 2)])
    selected_catalogue = {**catalogue,
                          "selected_basis_coordinates_by_rank": {4: selected_ids}}
    compiled = compile_coupling(plan_coupling(tagged_cauchy_image_request(
        species=["Ni"], catalogue=selected_catalogue)))
    selected = compiled.payload["image_coordinate_provenance"]
    assert tuple(row["coordinate_id"] for row in selected) == selected_ids
    assert tuple(row["label"]["block_Lambdas"] for row in selected) == ((0, 0), (2, 2))
    assert compiled.validation_report["exact_image_dimension"] == 2

    evaluator = _evaluator(compiled)
    assert evaluator.feature_count == 2
    atoms = Atoms("Ni5", positions=((0.0, 0.0, 0.0), (1.4, 0.2, 0.3),
                                     (-0.5, 1.5, 0.6), (0.7, -0.6, 1.8),
                                     (1.2, 1.3, -0.8)),
                  cell=(8.5, 8.5, 8.5), pbc=True)
    def feature_rows(structure):
        values, _jacobian = evaluator.features_and_position_jacobian(
            torch.as_tensor(structure.positions, dtype=torch.float64),
            torch.zeros(len(structure), dtype=torch.long),
            cell=torch.as_tensor(structure.cell.array, dtype=torch.float64),
            pbc=structure.pbc)
        return values.detach().numpy()

    rows = feature_rows(atoms)
    assert np.linalg.matrix_rank(rows, tol=1e-10) == 2
    model = TaggedCauchyImageLinearModel(
        evaluator, {"Ni": torch.tensor((0.7, -0.4), dtype=torch.float64)},
        {"Ni": -0.2})
    reference = atoms.copy()
    reference.calc = model.ase_calculator(backend="reference")
    native_calc = model.ase_calculator(
        backend="native_cpu", native_library=library, neighbors="ase")
    native = atoms.copy()
    native.calc = native_calc
    try:
        energy = native.get_potential_energy()
        forces = native.get_forces()
        stress = native.get_stress()
        np.testing.assert_allclose(energy, reference.get_potential_energy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(forces, reference.get_forces(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(stress, reference.get_stress(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(native_calc.native_runtime.evaluate_atoms(
            atoms, return_features=True)[4], rows, rtol=0, atol=1e-8)

        axis = np.array((1.0, 2.0, 3.0))
        axis /= np.linalg.norm(axis)
        cross = np.array(((0.0, -axis[2], axis[1]),
                          (axis[2], 0.0, -axis[0]),
                          (-axis[1], axis[0], 0.0)))
        angle = np.deg2rad(37.0)
        rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
        rotated = atoms.copy()
        rotated.positions[:] = atoms.positions @ rotation.T
        rotated.set_cell(np.asarray(atoms.cell) @ rotation.T, scale_atoms=False)
        rotated.calc = native_calc
        np.testing.assert_allclose(feature_rows(rotated), rows, rtol=0, atol=1e-9)
        np.testing.assert_allclose(native_calc.native_runtime.evaluate_atoms(
            rotated, return_features=True)[4], rows, rtol=0, atol=1e-8)
        np.testing.assert_allclose(rotated.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(rotated.get_forces(), forces @ rotation.T, rtol=0, atol=1e-8)
        np.testing.assert_allclose(voigt_6_to_full_3x3_stress(rotated.get_stress()),
                                   rotation @ voigt_6_to_full_3x3_stress(stress) @ rotation.T,
                                   rtol=0, atol=1e-8)
        inverted = atoms.copy()
        inverted.positions *= -1
        inverted.calc = native_calc
        np.testing.assert_allclose(feature_rows(inverted), rows, rtol=0, atol=1e-9)
        np.testing.assert_allclose(native_calc.native_runtime.evaluate_atoms(
            inverted, return_features=True)[4], rows, rtol=0, atol=1e-8)
        np.testing.assert_allclose(inverted.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(inverted.get_forces(), -forces, rtol=0, atol=1e-8)
        np.testing.assert_allclose(inverted.get_stress(), stress, rtol=0, atol=1e-8)
        reordered = atoms[[2, 0, 4, 1, 3]]
        reordered.calc = native_calc
        np.testing.assert_allclose(feature_rows(reordered), rows[[2, 0, 4, 1, 3]],
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(native_calc.native_runtime.evaluate_atoms(
            reordered, return_features=True)[4], rows[[2, 0, 4, 1, 3]],
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(reordered.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(reordered.get_forces(), forces[[2, 0, 4, 1, 3]],
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(reordered.get_stress(), stress, rtol=0, atol=1e-8)

        step = 1e-5
        assert abs(forces[1, 0]) > 1e-5
        energies = []
        for direction in (-1, 1):
            displaced = atoms.copy()
            displaced.positions[1, 0] += direction * step
            displaced.calc = native_calc
            energies.append(displaced.get_potential_energy())
        np.testing.assert_allclose(forces[1, 0], -(energies[1] - energies[0]) / (2 * step),
                                   rtol=0, atol=5e-6)
        assert np.min(np.abs(stress)) > 1e-7
        for index, (first, second) in enumerate(((0, 0), (1, 1), (2, 2),
                                                  (1, 2), (0, 2), (0, 1))):
            strained_energies = []
            for direction in (-1, 1):
                strain = np.zeros((3, 3))
                strain[first, second] = direction * step
                if first != second:
                    strain[second, first] = direction * step
                deformed = atoms.copy()
                deformed.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + strain),
                                  scale_atoms=True)
                deformed.calc = native_calc
                strained_energies.append(deformed.get_potential_energy())
            multiplier = 2 if first != second else 1
            derivative = (strained_energies[1] - strained_energies[0]) / (
                2 * step * atoms.get_volume() * multiplier)
            np.testing.assert_allclose(stress[index], derivative, rtol=0, atol=5e-6)
    finally:
        native_calc.native_runtime.close()


def test_bounded_compiler_cache_reuses_exact_artifact(tmp_path, monkeypatch):
    config = {
        "elements": ["Ta"],
        "representation": YE3TRepresentation.tagged_cauchy_image(),
        "tagged_cauchy_image": {
            "tensor_order": 4, "selected_raw_tag_counts": [0, 2],
            "source_family": "orthogonal_shifted_jacobi_origin_regular_v1",
            "radial_degrees": [0], "angular_degree": 1, "cutoff_A": 4.8,
            "coefficient_materialization": "compile", "compiled_cache_dir": tmp_path,
        },
    }
    first = YE3TDescriptors.ye3t_basis(config)
    cached = list(tmp_path.glob("*.json"))
    assert len(cached) == 1

    def no_recompile(*args, **kwargs):
        raise AssertionError("compiled tagged cache was ignored")

    monkeypatch.setattr(coupling_api, "compile", no_recompile)
    second = YE3TDescriptors.ye3t_basis(config)
    assert second.metadata["tagged_cauchy_image_compiled"].self_hash == (
        first.metadata["tagged_cauchy_image_compiled"].self_hash)


def test_descriptor_first_v3_fit_reuses_target_free_rows(
    compiled_tagged_image,
    monkeypatch,
    tmp_path,
):
    config = {
        "elements": ["Ta"],
        "representation": YE3TRepresentation.tagged_cauchy_image(),
        "tagged_cauchy_image": {
            "tensor_order": 4,
            "selected_raw_tag_counts": [0, 1, 2],
            "source_family": "orthogonal_shifted_jacobi_origin_regular_v1",
            "radial_degrees": [0],
            "angular_degree": 1,
            "cutoff_A": 4.8,
            "coefficient_materialization": "defer",
        },
    }
    representation = config["representation"]
    representation_replay = YE3TRepresentation.from_spec(
        representation.to_spec()
    )
    assert representation_replay.construction_mode == "tagged_cauchy_exact"
    assert representation_replay.basis_mode == "tagged_cauchy_image"
    preflight = YE3TDescriptors.ye3t_basis(config)
    assert preflight.metadata["runtime_status"] == "preflight_only"
    assert preflight.metadata["materialization_status"] == "preflight_only"
    assert preflight.metadata["feature_count"] == 2
    assert len(preflight.metadata["raw_opportunity_labels"]) == 3
    assert len(preflight.metadata["planned_feature_slots"]) == 2
    assert preflight.feature_keys == ()
    routed_preflight = YE3TDescriptors.ye3t(config)
    assert routed_preflight.metadata["feature_count"] == 2
    assert routed_preflight.metadata["descriptor_family"] == (
        "linear_tagged_cauchy_image"
    )
    assert preflight.metadata["tagged_cauchy_image_preflight"].resource_report[
        "coefficient_materialization_performed"
    ] is False

    monkeypatch.setattr(
        coupling_api,
        "compile",
        lambda _plan: compiled_tagged_image,
    )
    compiled_config = copy.deepcopy(config)
    compiled_config["tagged_cauchy_image"][
        "coefficient_materialization"
    ] = "compile"
    descriptor = YE3TDescriptors.ye3t_basis(compiled_config)
    assert descriptor.metadata["runtime_status"] == "implemented_under_validation"
    assert descriptor.metadata["tagged_cauchy_image_evaluator"] is not None
    assert len(descriptor.feature_keys) == 2
    assert len(descriptor.metadata["feature_coordinate_provenance"]) == 2
    assert descriptor.metadata["runtime_capabilities"] == {
        "fit_and_model_evaluation": True,
        "generic_descriptor_create": True,
        "native_bundle_export": True,
    }
    assert descriptor.supports_runtime_evaluation

    oracle = TaggedCauchyImageLinearModel(
        descriptor.metadata["tagged_cauchy_image_evaluator"],
        {"Ta": torch.tensor([0.43, -0.17], dtype=torch.float64)},
        {"Ta": -0.28},
    )
    structures = []
    configurations = (
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [-0.2, 1.3, 0.3]],
        [[0.0, 0.0, 0.0], [1.4, -0.2, 0.1], [0.3, 1.0, 0.7]],
        [[0.0, 0.0, 0.0], [1.0, 0.4, -0.2], [-0.5, 1.1, 0.8]],
        [[0.0, 0.0, 0.0], [1.5, 0.2, 0.3], [-0.4, 0.9, 1.0]],
    )
    for positions in configurations:
        atoms = Atoms("Ta3", positions=positions, pbc=False)
        energy, forces, _virial, _atomic = oracle.energy_forces_virial(
            torch.as_tensor(positions, dtype=torch.float64),
            torch.zeros(3, dtype=torch.long),
            pbc=(False, False, False),
        )
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(energy),
            forces=forces.detach().cpu().numpy(),
        )
        structures.append(atoms)

    local = descriptor.create(structures[0])
    assert local.shape == (3, len(descriptor.feature_keys))
    assert len(descriptor.feature_labels) == local.shape[1]
    rows = descriptor.training_matrix(structures[:1], properties=("energy", "forces"))
    coefficients = np.r_[oracle.beta_by_species["Ta"].numpy(), oracle.offsets["Ta"]]
    np.testing.assert_allclose(rows["matrix"] @ coefficients,
                               np.r_[structures[0].get_potential_energy(),
                                     structures[0].get_forces().reshape(-1)],
                               rtol=2.0e-12, atol=2.0e-12)

    cache_dir = tmp_path / "geometry_rows"
    fitted = YE3TModel.linear(
        descriptor,
        structures=structures,
        ridge_alpha=1.0e-12,
        energy_weight=1.0,
        force_weight=1.0,
        geometry_cache_dir=cache_dir,
    )
    periodic = Atoms("Ta3", positions=configurations[0], cell=(9.0, 9.0, 9.0), pbc=True)
    periodic.calc = fitted.ase_calculator(backend="reference")
    energy = periodic.get_potential_energy()
    forces = periodic.get_forces()
    stress = periodic.get_stress()
    assert np.isfinite(energy) and np.all(np.isfinite(forces))
    assert stress.shape == (6,) and np.all(np.isfinite(stress))
    step = 1.0e-6
    displaced = periodic.copy()
    displaced.positions[1, 0] += step
    displaced.calc = fitted.ase_calculator()
    e_plus = displaced.get_potential_energy()
    displaced.positions[1, 0] -= 2.0 * step
    e_minus = displaced.get_potential_energy()
    np.testing.assert_allclose(forces[1, 0], -(e_plus - e_minus) / (2.0 * step),
                               rtol=2.0e-6, atol=2.0e-7)
    strained = periodic.copy()
    strained.cell[0, 0] *= 1.0 + step
    strained.positions[:, 0] *= 1.0 + step
    strained.calc = fitted.ase_calculator()
    e_plus = strained.get_potential_energy()
    strained.cell[0, 0] *= (1.0 - step) / (1.0 + step)
    strained.positions[:, 0] *= (1.0 - step) / (1.0 + step)
    e_minus = strained.get_potential_energy()
    np.testing.assert_allclose(stress[0], (e_plus - e_minus) / (2.0 * step * periodic.get_volume()),
                               rtol=2.0e-6, atol=2.0e-7)
    from ase.md.verlet import VelocityVerlet
    from ase import units
    periodic.set_momenta(np.zeros((len(periodic), 3)))
    VelocityVerlet(periodic, timestep=0.1 * units.fs).run(2)
    assert np.isfinite(periodic.get_potential_energy())
    periodic_training = []
    for positions in configurations:
        atoms = Atoms("Ta3", positions=positions, cell=(9.0, 9.0, 9.0), pbc=True)
        atoms.calc = oracle.ase_calculator()
        targets = (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
        atoms.calc = SinglePointCalculator(atoms, energy=targets[0], forces=targets[1], stress=targets[2])
        periodic_training.append(atoms)
    design = descriptor.training_matrix(periodic_training[:1])
    np.testing.assert_allclose(
        design["matrix"] @ coefficients,
        np.r_[periodic_training[0].get_potential_energy(),
              periodic_training[0].get_forces().reshape(-1),
              periodic_training[0].get_stress()],
        rtol=2.0e-11, atol=2.0e-11,
    )
    stress_fit = YE3TModel.linear(
        descriptor, structures=periodic_training, ridge_alpha=1.0e-12,
        energy_weight=1.0, force_weight=1.0, stress_weight=1.0,
    )
    assert stress_fit.fit_metadata["stress_rows"] == 6 * len(periodic_training)
    for atoms in periodic_training:
        trial = atoms.copy()
        trial.calc = stress_fit.ase_calculator()
        np.testing.assert_allclose(trial.get_stress(), atoms.get_stress(),
                                   rtol=2.0e-7, atol=2.0e-8)
    score = score_tagged_cauchy_image_model(fitted, structures)
    assert score["energy_rmse_eV_per_atom"] < 2.0e-10
    assert score["force_rmse_eV_per_A"] < 2.0e-9
    cached_score = score_tagged_cauchy_image_model(
        fitted,
        structures,
        geometry_cache_dir=cache_dir,
        descriptor=descriptor,
    )
    assert cached_score == pytest.approx(score, rel=2.0e-12, abs=2.0e-12)

    normal = build_tagged_cauchy_image_normal_equations(
        descriptor,
        structures,
        geometry_cache_dir=cache_dir,
    )
    assert normal["geometry_cache_hits"] == len(structures)
    replay = YE3TModel.linear(
        descriptor,
        normal_equations=normal,
        ridge_alpha=1.0e-12,
    )
    replay_score = score_tagged_cauchy_image_model(replay, structures)
    assert replay_score == score

    shifted_targets = np.asarray(
        [atoms.get_potential_energy() + 0.1 for atoms in structures]
    )
    changed = build_tagged_cauchy_image_normal_equations(
        descriptor,
        structures,
        target_energies=shifted_targets,
        geometry_cache_dir=cache_dir,
    )
    assert changed["geometry_cache_hits"] == len(structures)
    assert changed["target_identity"] != normal["target_identity"]
    assert changed["row_hashes"] == normal["row_hashes"]

    residual_energies = np.asarray(
        [atoms.get_potential_energy() for atoms in structures]
    )
    residual_forces = tuple(atoms.get_forces() for atoms in structures)
    reference_metadata = {"schema": "test_reference_v1", "name": "fixture"}
    target_metadata = tagged_cauchy_reference_target_metadata(
        reference_energies={"Ta": -0.25},
        reference_potential_metadata=reference_metadata,
    )
    assert target_metadata["elemental_energy_offsets"]["reference_energies"] == {
        "Ta": -0.25
    }
    assert target_metadata["external_reference_potential"]["metadata"] == (
        reference_metadata
    )
    referenced = YE3TModel.linear(
        descriptor,
        structures=structures,
        target_energies=residual_energies,
        target_forces=residual_forces,
        reference_potential_metadata=reference_metadata,
        geometry_cache_dir=cache_dir,
        ridge_alpha=1.0e-12,
    )
    bound = referenced.fit_metadata["reference_target_metadata"]
    assert bound["external_reference_potential"] == {
        "enabled": True,
        "metadata": reference_metadata,
        "target_convention": "caller_supplied_E_and_F_residuals",
    }
    exported = tmp_path / "fitted_with_provenance.ye3t.json"
    payload = export_tagged_cauchy_image_model(exported, referenced)
    assert payload["fit_metadata"]["normal_hash"] == referenced.fit_metadata[
        "normal_hash"
    ]
    loaded = load_tagged_cauchy_image_model(exported)
    assert loaded.fit_metadata == json.loads(
        json.dumps(referenced.fit_metadata, sort_keys=True)
    )

    with pytest.raises(ValueError, match="explicit residual"):
        YE3TModel.linear(
            descriptor,
            structures=structures,
            reference_potential_metadata=reference_metadata,
        )


def test_public_tagged_preflight_is_cwd_independent_and_fail_closed(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    example = (
        repository
        / "examples"
        / "publication"
        / "ta_tagged_cauchy_image_linear"
    )
    output = tmp_path / "preflight"
    command = [
        sys.executable,
        str(example / "train_export.py"),
        "--config",
        str(example / "config_quick.json"),
        "--output",
        str(output),
        "--cache-root",
        str(tmp_path / "cache"),
        "--preflight-only",
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert report["dataset_loaded"] is False
    assert report["image_descriptor_coefficients_materialized"] is False
    assert sorted(report["arms"]) == [
        "s02_joint",
        "s0_ordinary",
        "s1_duplicate_control",
        "s2_nontrivial",
    ]

    repeated = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert repeated.returncode != 0
    assert "Output path already exists" in repeated.stderr
