from functools import lru_cache
from fractions import Fraction
import math
import json
import socket

import numpy as np
import pytest
import torch


@lru_cache(maxsize=1)
def _compiled():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request

    request = first_lifted_cauchy_scalar_request(
        2,
        family_ids=("NT_NU4_K22_L0", "NT_NU2_MU2_SIGN_L1x1"),
    )
    return compile_coupling(request)


@lru_cache(maxsize=1)
def _compiled_all_nontrivial():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request

    request = first_lifted_cauchy_scalar_request(
        3,
        family_ids=(
            "NT_NU4_K22_L0",
            "NT_NU2_MU2_SIGN_L1x1",
            "NT_NU3_MU_K21x1_L1x1",
            "NT_NU2_MU_XI_SIGN_L1x1x1",
        ),
    )
    return compile_coupling(request)


@lru_cache(maxsize=2)
def _compiled_joint(channel_count=3):
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LIFTED_CAUCHY_JOINT_SOURCE_FAMILY,
    )

    request = first_lifted_cauchy_scalar_request(
        channel_count,
        source_family_id=LIFTED_CAUCHY_JOINT_SOURCE_FAMILY,
        family_ids=(
            "NT_NU4_K22_L0",
            "NT_NU2_MU2_SIGN_L1x1",
        ),
    )
    return compile_coupling(request)


@lru_cache(maxsize=1)
def _compiled_mixed_l():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import lifted_cauchy_fixed_content_scalar_request
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
    )

    channels = (
        {
            "channel_id": "n1_l0",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 0,
            "source_family_id": LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
        },
        {
            "channel_id": "n1_l1",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 1,
            "source_family_id": LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
        },
        {
            "channel_id": "n1_l2",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 2,
            "source_family_id": LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
        },
    )
    return compile_coupling(
        lifted_cauchy_fixed_content_scalar_request(
            channels,
            (1, 2, 2),
            role_dimension=2,
        )
    )


@lru_cache(maxsize=1)
def _compiled_mixed_l_scalar():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import lifted_cauchy_fixed_content_scalar_request
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
    )

    channels = (
        {
            "channel_id": "n1_l0",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 0,
            "source_family_id": LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
        },
    )
    return compile_coupling(
        lifted_cauchy_fixed_content_scalar_request(
            channels,
            (1,),
            role_dimension=2,
        )
    )


@lru_cache(maxsize=1)
def _compiled_mixed_l_l0_rank2():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import lifted_cauchy_fixed_content_scalar_request
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
    )

    channels = (
        {
            "channel_id": "n1_l0",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 0,
            "source_family_id": LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
        },
    )
    return compile_coupling(
        lifted_cauchy_fixed_content_scalar_request(
            channels,
            (2,),
            role_dimension=2,
        )
    )


def _source_config():
    return {
        "schema": "ye3t_lifted_cauchy_polynomial_source_v1",
        "cutoff_A": 5.2,
        "envelope": "one_minus_x_squared",
        "radial_coordinate": "r_over_rc",
        "primitive_radial": "envelope_times_x_power",
        "roles": (
            {"id": "inner", "kind": "one_minus_x"},
            {"id": "outer", "kind": "x"},
        ),
        "density_normalization": "none",
        "periodic_image_mode": "nonperiodic",
        "neighbor_backend": "ase",
    }


def _joint_source_config():
    return {
        "schema": "ye3t_lifted_cauchy_joint_source_v2",
        "cutoff_A": 5.2,
        "periodic_image_mode": "nonperiodic",
        "neighbor_backend": "ase",
    }


def _mixed_l_source_config():
    return {
        "schema": "ye3t_lifted_cauchy_joint_source_v3",
        "cutoff_A": 5.2,
        "periodic_image_mode": "nonperiodic",
        "neighbor_backend": "ase",
    }


def _composite_binding(compiled, name):
    descriptor_count = len(compiled.payload["descriptors"])
    return {
        "component_id": name,
        "opportunity_id": name,
        "compiled": compiled.to_dict(),
        "coordinate_ids": tuple(
            f"{name}:coordinate:{index}" for index in range(descriptor_count)
        ),
        "selection_hash": f"{name}:selection",
        "strict_sector_ids": (f"{name}:sector",),
    }


def _positions():
    return torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (1.8, 0.1, -0.2),
            (-0.4, 2.0, 0.3),
            (0.2, -0.7, 2.1),
        ),
        dtype=torch.float64,
    )


def _pack_native_density(density, native):
    blocks = native["coordinate_convention"]["channel_blocks"]
    packed = []
    next_offset = 0
    for block in blocks:
        assert int(block["source_variable_offset"]) == next_offset
        position = int(block["channel_position"])
        width = int(block["real_component_count"])
        packed.append(density[:, position, :, :width].reshape(len(density), -1))
        next_offset += 2 * width
    result = torch.cat(tuple(packed), dim=1)
    assert int(result.shape[1]) == int(native["source_variable_count"])
    return result


def _random_rotation(seed=17):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return torch.as_tensor(q, dtype=torch.float64)


def test_lifted_cauchy_representation_and_descriptor_public_flow():
    from ye3t_methods.atomistic import YE3TDescriptors, YE3TRepresentation

    representation = YE3TRepresentation.lifted_cauchy_scalar()
    restored = YE3TRepresentation.from_config(representation.as_dict())
    assert restored == representation
    assert restored.permutation_sector == "trivial"
    assert restored.construction_mode == "lifted_cauchy_exact"
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "representation": representation,
            "lifted_cauchy": {
                "compiled": _compiled().to_dict(),
                "source": _source_config(),
            },
        }
    )
    assert descriptor.metadata["descriptor_family"] == "linear_lifted_cauchy_scalar"
    assert descriptor.metadata["lifted_cauchy_artifact_hash"] == _compiled().self_hash
    assert descriptor.metadata["lifted_cauchy_descriptor_count"] == 3
    assert descriptor.metadata["ye3t_spec"]["target_rotation"]["L_R"] == 0
    assert descriptor.metadata["ye3t_spec"]["target_rotation"]["parity"] == "even"


def test_lifted_fit_preflight_reports_size_without_compilation_or_data_access():
    from ye3t_methods.atomistic import lifted_cauchy_linear_fit_preflight

    preflight = lifted_cauchy_linear_fit_preflight(
        _compiled().plan.report,
        central_species_order=("Ta", "W"),
        ordinary_feature_count=7,
        structure_atom_counts=(4, 6),
    )
    assert preflight["compiler_descriptor_count"] == 3
    assert preflight["catalogue_exhaustiveness"] == (
        "defined_by_generator_request"
    )
    assert preflight["lifted_fit_feature_count"] == 6
    assert preflight["ordinary_fit_feature_count"] == 7
    assert preflight["total_fit_feature_count"] == 13
    assert preflight["intercept_parameter_count"] == 2
    assert preflight["linear_parameter_count"] == 15
    assert preflight["tensor_ranks"] == (4,)
    assert preflight["fit_shape"] == {
        "structure_count": 2,
        "atom_count": 10,
        "energy_row_count": 2,
        "force_row_count": 30,
        "regression_row_count": 32,
        "regression_column_count": 15,
        "gram_matrix_bytes": 1800,
        "sufficient_statistics_bytes": 1928,
        "dense_fallback_design_and_target_bytes": 4096,
        "streaming_dataset_passes": 2,
        "streaming_retained_structure_count": 1,
    }
    assert preflight["coefficient_compilation_performed"] is False
    assert preflight["descriptor_evaluation_performed"] is False
    assert preflight["dataset_access_performed"] is False


def test_lifted_feature_chunk_auto_policy_is_dataset_free_and_bounded():
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        resolve_lifted_cauchy_feature_chunk_size,
    )

    small = resolve_lifted_cauchy_feature_chunk_size("auto", 11)
    assert small == {
        "policy": "lifted_cauchy_cpu_feature_chunk_v1",
        "requested": "auto",
        "resolved": 11,
        "penalized_feature_count": 11,
        "reason": "all_features_for_small_linear_problem",
        "hardware_calibrated": False,
    }
    large = resolve_lifted_cauchy_feature_chunk_size("auto", 500)
    assert large["resolved"] == 32
    assert large["reason"] == "conservative_cpu_vjp_batch"
    override = resolve_lifted_cauchy_feature_chunk_size(64, 11)
    assert override["resolved"] == 64
    assert override["reason"] == "explicit_user_override"
    for invalid in (0, -1, True, "default", None, 2.5):
        with pytest.raises(ValueError, match="positive integer or 'auto'"):
            resolve_lifted_cauchy_feature_chunk_size(invalid, 11)


def test_streamed_ridge_transports_identity_metric_through_feature_scaling():
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        solve_lifted_cauchy_normal_equations,
    )

    problem = {
        "XtX": np.eye(3, dtype=np.float64),
        "Xty": np.asarray((1.0, 2.0, 0.0), dtype=np.float64),
        "yty": 5.0,
        "penalized_parameter_count": 2,
        "feature_scale": np.asarray((2.0, 0.5), dtype=np.float64),
        "feature_mean": np.zeros(2, dtype=np.float64),
        "fit_coordinate_metric": np.eye(2, dtype=np.float64),
        "runtime_from_fit_coordinates": np.eye(2, dtype=np.float64),
    }
    solved = solve_lifted_cauchy_normal_equations(
        problem,
        ridge_alpha=1.0,
        svd_rcond=1.0e-12,
        maximum_condition=1.0e12,
    )

    np.testing.assert_allclose(
        solved["scaled_coefficients"], (0.8, 0.4, 0.0), atol=1.0e-14
    )
    np.testing.assert_allclose(
        solved["fit_coordinate_coefficients"], (0.4, 0.8, 0.0), atol=1.0e-14
    )
    assert solved["metadata"]["ridge_penalty_coordinate"] == (
        "unscaled_fit_coordinates"
    )
    assert solved["metadata"]["ridge_penalty_norm"] == pytest.approx(
        np.sqrt(0.8)
    )


def test_streamed_ridge_reports_unregularized_design_rank_separately():
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        solve_lifted_cauchy_normal_equations,
    )

    problem = {
        "XtX": np.diag((1.0, 0.0, 1.0)),
        "Xty": np.asarray((1.0, 0.0, 1.0), dtype=np.float64),
        "yty": 2.0,
        "penalized_parameter_count": 2,
        "feature_scale": np.ones(2, dtype=np.float64),
        "feature_mean": np.zeros(2, dtype=np.float64),
        "fit_coordinate_metric": np.eye(2, dtype=np.float64),
        "runtime_from_fit_coordinates": np.eye(2, dtype=np.float64),
    }
    solved = solve_lifted_cauchy_normal_equations(
        problem,
        ridge_alpha=1.0,
        svd_rcond=1.0e-12,
        maximum_condition=1.0e12,
    )

    assert solved["metadata"]["numerical_rank"] == 2
    assert solved["metadata"]["design_numerical_rank"] == 2
    assert solved["metadata"]["regularized_system_rank"] == 3
    assert solved["metadata"]["design_retained_condition_number"] == 1.0
    assert solved["metadata"]["retained_system_condition_number"] == 2.0


def test_lifted_cauchy_representation_rejects_reserved_semantic_override():
    from ye3t_methods.atomistic import YE3TRepresentation

    with pytest.raises(ValueError, match="cannot override target_L_R"):
        YE3TRepresentation.lifted_cauchy_scalar(metadata={"target_L_R": 2})


def test_polynomial_source_exposes_both_radial_and_role_derivatives():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    source = LiftedCauchyPolynomialSource(
        _compiled(), source_config=_source_config(), type_map={"Ta": 0}
    )
    disp = torch.tensor(((1.3, -0.4, 0.7),), dtype=torch.float64)
    types = torch.zeros(1, dtype=torch.long)
    values, derivative = source.edge_values_with_dx(disp, types)
    step = 2.0e-6
    numerical = torch.zeros_like(derivative)
    for axis in range(3):
        delta = torch.zeros_like(disp)
        delta[:, axis] = step
        plus = source.edge_values_with_dx(disp + delta, types)[0]
        minus = source.edge_values_with_dx(disp - delta, types)[0]
        numerical[..., axis] = (plus - minus) / (2.0 * step)
    assert torch.max(torch.abs(derivative - numerical)).item() < 2.0e-9
    assert not torch.allclose(derivative[:, :, 0], derivative[:, :, 1])
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("channel_count", (2, 3))
def test_joint_source_has_exact_identity_radial_gram(channel_count):
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    source = LiftedCauchyPolynomialSource(
        _compiled_joint(channel_count),
        source_config=_joint_source_config(),
        type_map={"Ta": 0},
    )
    group = source.config["groups"][0]
    polynomials = group["polynomials"]
    angular_l = int(group["l"])
    # Fractions do not expose square roots; all off-diagonal terms vanish
    # before normalization and each squared diagonal normalization is exact.
    for left_index, left in enumerate(polynomials):
        for right_index, right in enumerate(polynomials):
            unnormalized = Fraction(0)
            for left_power, left_coefficient in enumerate(
                left["shifted_jacobi_power_coefficients"]
            ):
                for right_power, right_coefficient in enumerate(
                    right["shifted_jacobi_power_coefficients"]
                ):
                    exponent = 2 + 2 * angular_l + left_power + right_power
                    unnormalized += Fraction(
                        left_coefficient
                        * right_coefficient
                        * math.factorial(exponent)
                        * math.factorial(4),
                        math.factorial(exponent + 5),
                    )
            if left_index != right_index:
                assert unnormalized == 0
            else:
                normalization_squared = Fraction(
                    left["normalization_squared"]["numerator"],
                    left["normalization_squared"]["denominator"],
                )
                assert unnormalized * normalization_squared == 1
    assert group["source_dimension"] == 2 * channel_count
    assert source.config["certificates"]["radial_source_gram"] == "identity_exact"
    lowering = group["factorized_lowering"]
    matrix = [
        [Fraction(value) for value in row]
        for row in lowering["unnormalized_integer_rows"]
    ]
    determinant = Fraction(1)
    for column in range(len(matrix)):
        pivot_row = next(
            row
            for row in range(column, len(matrix))
            if matrix[row][column] != 0
        )
        if pivot_row != column:
            matrix[column], matrix[pivot_row] = matrix[pivot_row], matrix[column]
            determinant = -determinant
        pivot = matrix[column][column]
        determinant *= pivot
        for row in range(column + 1, len(matrix)):
            scale = matrix[row][column] / pivot
            for inner_column in range(column, len(matrix)):
                matrix[row][inner_column] -= scale * matrix[column][inner_column]
    assert determinant.denominator == 1
    assert str(determinant.numerator) == lowering["unnormalized_integer_determinant"]
    assert lowering["determinant_nonzero_exact"] is True
    assert lowering["binary64_minimum_singular_value"] > 0.0


@pytest.mark.parametrize("channel_count", (2, 3))
def test_joint_source_direct_and_factorized_values_and_vjp_match(channel_count):
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    compiled = _compiled_joint(channel_count)
    direct = LiftedCauchyPolynomialSource(
        compiled,
        source_config=_joint_source_config(),
        type_map={"Ta": 0},
        source_realization="direct",
    )
    factorized = LiftedCauchyPolynomialSource(
        compiled,
        source_config=_joint_source_config(),
        type_map={"Ta": 0},
        source_realization="factorized",
    )
    positions = _positions()
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    direct_density, direct_context = direct.materialize(positions, atom_types)
    factorized_density, factorized_context = factorized.materialize(
        positions, atom_types
    )
    assert torch.max(torch.abs(direct_density - factorized_density)).item() < 5.0e-11

    generator = torch.Generator().manual_seed(181 + channel_count)
    adjoint = torch.randn(
        direct_density.shape, dtype=torch.float64, generator=generator
    )
    direct_vjp = direct.vjp(adjoint, direct_context)
    factorized_vjp = factorized.vjp(adjoint, factorized_context)
    assert torch.max(
        torch.abs(
            direct_vjp["position_gradient"]
            - factorized_vjp["position_gradient"]
        )
    ).item() < 5.0e-11
    assert torch.max(
        torch.abs(
            direct_vjp["strain_derivative"]
            - factorized_vjp["strain_derivative"]
        )
    ).item() < 5.0e-10


def test_joint_source_direct_edge_derivative_matches_finite_difference():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    source = LiftedCauchyPolynomialSource(
        _compiled_joint(2),
        source_config=_joint_source_config(),
        type_map={"Ta": 0},
        source_realization="direct",
    )
    displacement = torch.tensor(((1.3, -0.4, 0.7),), dtype=torch.float64)
    neighbor_types = torch.zeros(1, dtype=torch.long)
    _value, derivative = source.edge_values_with_dx(
        displacement, neighbor_types
    )
    step = 2.0e-6
    numerical = torch.zeros_like(derivative)
    for axis in range(3):
        delta = torch.zeros_like(displacement)
        delta[:, axis] = step
        plus = source.edge_values_with_dx(
            displacement + delta, neighbor_types
        )[0]
        minus = source.edge_values_with_dx(
            displacement - delta, neighbor_types
        )[0]
        numerical[..., axis] = (plus - minus) / (2.0 * step)
    assert torch.max(torch.abs(derivative - numerical)).item() < 2.0e-8


def test_mixed_l_source_and_evaluator_match_factorized_vjp_and_finite_difference():
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LiftedCauchyPolynomialSource,
        LiftedCauchyTorchEvaluator,
    )

    compiled = _compiled_mixed_l()
    direct = LiftedCauchyPolynomialSource(
        compiled,
        source_config=_mixed_l_source_config(),
        type_map={"Ta": 0},
        source_realization="direct",
    )
    factorized = LiftedCauchyPolynomialSource(
        compiled,
        source_config=_mixed_l_source_config(),
        type_map={"Ta": 0},
        source_realization="factorized",
    )
    positions = _positions()
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    direct_density, direct_context = direct.materialize(positions, atom_types)
    factorized_density, factorized_context = factorized.materialize(
        positions, atom_types
    )
    assert tuple(direct_density.shape[-3:]) == (3, 2, 5)
    assert torch.count_nonzero(direct_density[:, 0, :, 1:]).item() == 0
    assert torch.count_nonzero(direct_density[:, 1, :, 3:]).item() == 0
    assert torch.max(torch.abs(direct_density - factorized_density)).item() < 8.0e-11

    evaluator = LiftedCauchyTorchEvaluator(compiled)
    direct_features = evaluator.evaluate(direct_density, realization="canonical")
    factorized_features = evaluator.evaluate(
        factorized_density, realization="factored"
    )
    assert torch.max(torch.abs(direct_features - factorized_features)).item() < 2.0e-10
    generator = torch.Generator().manual_seed(5620)
    upstream = torch.randn(
        direct_features.shape, dtype=torch.float64, generator=generator
    )
    _features, density_adjoint = evaluator.vjp(
        direct_density,
        upstream,
        realization="canonical",
    )
    direct_vjp = direct.vjp(density_adjoint, direct_context)
    factorized_vjp = factorized.vjp(density_adjoint, factorized_context)
    assert torch.max(
        torch.abs(
            direct_vjp["position_gradient"]
            - factorized_vjp["position_gradient"]
        )
    ).item() < 2.0e-9

    displacement = torch.tensor(((1.3, -0.4, 0.7),), dtype=torch.float64)
    neighbor_types = torch.zeros(1, dtype=torch.long)
    _value, derivative = direct.edge_values_with_dx(displacement, neighbor_types)
    numerical = torch.zeros_like(derivative)
    step = 2.0e-6
    for axis in range(3):
        delta = torch.zeros_like(displacement)
        delta[:, axis] = step
        plus = direct.edge_values_with_dx(
            displacement + delta, neighbor_types
        )[0]
        minus = direct.edge_values_with_dx(
            displacement - delta, neighbor_types
        )[0]
        numerical[..., axis] = (plus - minus) / (2.0 * step)
    assert torch.max(torch.abs(derivative - numerical)).item() < 8.0e-8


def test_mixed_l_bundle_exports_packed_native_values_and_forces(tmp_path):
    from ye3t_methods.atomistic import (
        YE3TDescriptors,
        export_lifted_cauchy_linear_bundle,
        load_lifted_cauchy_linear_bundle,
    )
    from ye3t_methods.atomistic.lifted_cauchy_linear import lifted_cauchy_model_from_descriptor

    compiled = _compiled_mixed_l()
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "compiled": compiled.to_dict(),
                "source": _mixed_l_source_config(),
            },
        }
    )
    coefficient_count = len(compiled.payload["descriptors"])
    model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": (
                tuple(np.linspace(-0.6, 0.9, coefficient_count)),
            ),
            "offsets": (0.17,),
            "realization": "factored",
            "source_realization": "direct",
        },
    )
    bundle = tmp_path / "mixed_l_bundle"
    export_lifted_cauchy_linear_bundle(model, bundle)
    loaded = load_lifted_cauchy_linear_bundle(bundle)
    native = loaded["native_runtime_payload"]
    assert native["schema"] == "ye3t_lifted_cauchy_native_runtime_v2"
    blocks = native["coordinate_convention"]["channel_blocks"]
    assert [int(block["l"]) for block in blocks] == [0, 1, 2]
    assert [int(block["real_component_count"]) for block in blocks] == [1, 3, 5]
    assert [int(block["source_variable_offset"]) for block in blocks] == [0, 2, 8]
    assert int(native["source_variable_count"]) == 18

    positions = _positions().requires_grad_(True)
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    density, _context = model.source.materialize(positions, atom_types)
    native_variables = _pack_native_density(density, native)
    polynomial = native["heads"][0]["polynomial"]
    native_atomic = torch.full(
        (len(positions),), float(polynomial["offset"]), dtype=torch.float64
    )
    for term, coefficient in enumerate(polynomial["monomial_coefficients"]):
        value = torch.full_like(native_atomic, float(coefficient))
        begin = int(polynomial["factor_offsets"][term])
        end = int(polynomial["factor_offsets"][term + 1])
        for factor in range(begin, end):
            source_index = int(polynomial["factor_indices"][factor])
            exponent = int(polynomial["factor_exponents"][factor])
            value = value * native_variables[:, source_index].pow(exponent)
        native_atomic = native_atomic + value
    native_energy = native_atomic.sum()
    native_force = -torch.autograd.grad(native_energy, positions)[0]

    reference_positions = _positions().requires_grad_(True)
    reference_energy = model(reference_positions, atom_types)
    reference_force = -torch.autograd.grad(reference_energy, reference_positions)[0]
    assert torch.abs(native_energy - reference_energy).item() < 8.0e-11
    assert torch.max(torch.abs(native_force - reference_force)).item() < 2.0e-9


def test_composite_mixed_l_model_shares_source_and_matches_component_rows():
    from ase import Atoms

    from ye3t_methods.atomistic import YE3TDescriptors
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        _fit_coordinate_lowering,
        lifted_cauchy_model_from_descriptor,
    )

    compiled = (_compiled_mixed_l_scalar(), _compiled_mixed_l())
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "components": (
                    _composite_binding(compiled[0], "rank1_l0"),
                    _composite_binding(compiled[1], "rank5_mixed_l"),
                ),
                "source": _mixed_l_source_config(),
            },
        }
    )
    feature_counts = tuple(
        len(component.payload["descriptors"]) for component in compiled
    )
    total = sum(feature_counts)
    coefficients = np.linspace(-0.003, 0.004, total).reshape(1, total)
    composite = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": coefficients,
            "offsets": (0.17,),
            "realization": "canonical",
            "source_realization": "direct",
        },
    )
    positions = _positions().requires_grad_(True)
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    materialize_calls = 0
    materialize = composite.source.materialize

    def counted_materialize(*args, **kwargs):
        nonlocal materialize_calls
        materialize_calls += 1
        return materialize(*args, **kwargs)

    composite.source.materialize = counted_materialize
    composite_features, _density, _context = composite.atomic_features(
        positions, atom_types
    )
    assert materialize_calls == 1

    individual_features = []
    individual_models = []
    cursor = 0
    for position, component in enumerate(compiled):
        component_descriptor = YE3TDescriptors.ye3t_basis(
            {
                "elements": ("Ta",),
                "type_map": {"Ta": 0},
                "lifted_cauchy": {
                    "compiled": component.to_dict(),
                    "source": _mixed_l_source_config(),
                },
            }
        )
        stop = cursor + feature_counts[position]
        component_model = lifted_cauchy_model_from_descriptor(
            component_descriptor,
            {
                "coefficients": coefficients[:, cursor:stop],
                "offsets": (0.05 if position == 0 else 0.12,),
                "realization": "canonical",
                "source_realization": "direct",
            },
        )
        individual_models.append(component_model)
        individual_features.append(
            component_model.atomic_features(positions, atom_types)[0]
        )
        cursor = stop
    expected_features = torch.cat(tuple(individual_features), dim=1)
    assert torch.max(torch.abs(composite_features - expected_features)).item() < 2.0e-10

    composite_energy = composite(positions, atom_types)
    component_energy = sum(model(positions, atom_types) for model in individual_models)
    assert torch.abs(composite_energy - component_energy).item() < 2.0e-10
    composite_force = -torch.autograd.grad(
        composite_energy, positions, retain_graph=True
    )[0]
    component_force = -torch.autograd.grad(component_energy, positions)[0]
    assert torch.max(torch.abs(composite_force - component_force)).item() < 2.0e-9

    atoms = Atoms("Ta4", positions=_positions().numpy(), pbc=False)
    composite_row = composite.regression_rows(atoms, feature_chunk_size=8)
    component_rows = tuple(
        model.regression_rows(atoms, feature_chunk_size=8)
        for model in individual_models
    )
    expected_site = np.column_stack(
        tuple(row["site_design"] for row in component_rows)
    )
    expected_force = np.column_stack(
        tuple(row["forces"][:, :-1] for row in component_rows)
    )
    np.testing.assert_allclose(
        composite_row["site_design"], expected_site, atol=2.0e-10, rtol=2.0e-11
    )
    np.testing.assert_allclose(
        composite_row["forces"][:, :-1],
        expected_force,
        atol=2.0e-9,
        rtol=2.0e-11,
    )
    plan, lowering, norms = _fit_coordinate_lowering(composite, "orthogonal")
    assert plan["schema"] == "ye3t_lifted_cauchy_composite_orthogonal_output_v2"
    assert lowering.shape == (total, total)
    assert norms.shape == (total,)
    assert np.all(norms > 0.0)


def test_composite_bundle_flattens_components_and_round_trips_native_polynomial(
    tmp_path,
):
    from ye3t_methods.atomistic import (
        YE3TDescriptors,
        export_lifted_cauchy_linear_bundle,
        load_lifted_cauchy_linear_bundle,
    )
    from ye3t_methods.atomistic.lifted_cauchy_linear import lifted_cauchy_model_from_descriptor

    compiled = (_compiled_mixed_l_scalar(), _compiled_mixed_l_l0_rank2())
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "components": tuple(
                    _composite_binding(component, f"component_{index}")
                    for index, component in enumerate(compiled)
                ),
                "source": _mixed_l_source_config(),
            },
        }
    )
    feature_count = sum(len(component.payload["descriptors"]) for component in compiled)
    coefficients = np.linspace(-0.07, 0.11, feature_count).reshape(1, -1)
    model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": coefficients,
            "offsets": (0.23,),
            "realization": "canonical",
            "source_realization": "direct",
        },
    )
    model.fit_metadata = {"fixture": "composite_native_round_trip"}
    bundle = tmp_path / "composite_bundle"
    export_lifted_cauchy_linear_bundle(model, bundle)
    loaded = load_lifted_cauchy_linear_bundle(bundle)
    assert len(tuple(bundle.glob("compiled_component_*.json"))) == 2
    assert loaded["compiler_binding_payload"]["validation"]["passed"] is True
    assert (
        loaded["native_runtime_payload"]["capabilities"]["runtime_compiler_dom"]
        is False
    )

    positions = _positions().requires_grad_(True)
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    reference_energy = model(positions, atom_types)
    reference_force = -torch.autograd.grad(reference_energy, positions)[0]
    loaded_positions = _positions().requires_grad_(True)
    loaded_energy = loaded["lifted_model"](loaded_positions, atom_types)
    loaded_force = -torch.autograd.grad(loaded_energy, loaded_positions)[0]
    assert torch.abs(loaded_energy - reference_energy).item() < 2.0e-12
    assert torch.max(torch.abs(loaded_force - reference_force)).item() < 2.0e-11

    native_positions = _positions().requires_grad_(True)
    density, _context = model.source.materialize(native_positions, atom_types)
    native = loaded["native_runtime_payload"]
    variables = _pack_native_density(density, native)
    polynomial = native["heads"][0]["polynomial"]
    native_atomic = torch.full(
        (len(positions),), float(polynomial["offset"]), dtype=torch.float64
    )
    for term, coefficient in enumerate(polynomial["monomial_coefficients"]):
        value = torch.full_like(native_atomic, float(coefficient))
        begin = int(polynomial["factor_offsets"][term])
        end = int(polynomial["factor_offsets"][term + 1])
        for factor in range(begin, end):
            source_index = int(polynomial["factor_indices"][factor])
            exponent = int(polynomial["factor_exponents"][factor])
            value = value * variables[:, source_index].pow(exponent)
        native_atomic = native_atomic + value
    native_energy = native_atomic.sum()
    native_force = -torch.autograd.grad(native_energy, native_positions)[0]
    assert max(polynomial["factor_exponents"]) > 1
    assert torch.abs(native_energy - reference_energy.detach()).item() < 2.0e-11
    assert torch.max(torch.abs(native_force - reference_force.detach())).item() < 2.0e-10


def test_composite_uses_hash_bound_precomputed_orthogonal_plans(monkeypatch):
    import ye3t.couplings

    from ye3t.couplings import lifted_cauchy_orthogonal_output_plan
    from ye3t_methods.atomistic import YE3TDescriptors
    import ye3t_methods.atomistic.lifted_cauchy_linear as linear

    compiled = (_compiled_mixed_l_scalar(), _compiled_mixed_l_l0_rank2())
    plans = tuple(
        lifted_cauchy_orthogonal_output_plan(component)
        for component in compiled
    )
    bindings = []
    for index, (component, plan) in enumerate(zip(compiled, plans, strict=True)):
        binding = _composite_binding(component, f"component_{index}")
        binding["orthogonal_plan"] = plan
        binding["orthogonal_output_plan_hash"] = plan["self_hash"]
        bindings.append(binding)

    linear._ORTHOGONAL_OUTPUT_CACHE.clear()

    def fail_recompute(_compiled):
        raise AssertionError("certified orthogonal plan was recomputed")

    monkeypatch.setattr(
        ye3t.couplings,
        "lifted_cauchy_orthogonal_output_plan",
        fail_recompute,
    )
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "components": tuple(bindings),
                "source": _mixed_l_source_config(),
            },
        }
    )
    model = linear.lifted_cauchy_model_from_descriptor(descriptor)
    composite_plan, lowering, norms = linear._fit_coordinate_lowering(
        model, "orthogonal"
    )
    assert composite_plan["component_plan_hashes"] == tuple(
        plan["self_hash"] for plan in plans
    )
    assert lowering.shape[0] == lowering.shape[1] == sum(
        len(component.payload["descriptors"]) for component in compiled
    )
    assert np.all(norms > 0.0)

    tampered = dict(bindings[0])
    tampered["orthogonal_output_plan_hash"] = "0" * 64
    with pytest.raises(ValueError, match="certificate hash mismatch"):
        YE3TDescriptors.ye3t_basis(
            {
                "elements": ("Ta",),
                "type_map": {"Ta": 0},
                "lifted_cauchy": {
                    "components": (tampered,),
                    "source": _mixed_l_source_config(),
                },
            }
        )


def test_composite_parent_row_cache_matches_direct_gram_and_solution(tmp_path):
    from ase import Atoms

    from ye3t_methods.atomistic import YE3TDescriptors
    from ye3t_methods.atomistic.cache import (
        LinearCacheValidationError,
        load_lifted_cauchy_geometry_row_cache,
        load_lifted_cauchy_normal_equations,
        load_lifted_cauchy_target_cache,
        persist_lifted_cauchy_geometry_row_cache,
        persist_lifted_cauchy_normal_equations,
        persist_lifted_cauchy_target_cache,
    )
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        _lifted_cauchy_cached_runtime_parameters,
        build_lifted_cauchy_normal_equations,
        build_lifted_cauchy_normal_equations_from_geometry_and_targets,
        build_lifted_cauchy_normal_equations_from_row_cache,
        combine_lifted_cauchy_geometry_and_target_caches,
        lifted_cauchy_geometry_row_cache_request,
        lifted_cauchy_normal_equation_cache_request,
        lifted_cauchy_model_from_cached_solution,
        lifted_cauchy_model_from_descriptor,
        materialize_lifted_cauchy_geometry_row_cache,
        materialize_lifted_cauchy_regression_row_cache,
        materialize_lifted_cauchy_target_cache,
        score_lifted_cauchy_cached_solution,
        solve_lifted_cauchy_normal_equations,
        subset_lifted_cauchy_normal_equations_from_parent,
    )

    components = (_compiled_mixed_l_scalar(), _compiled_mixed_l_l0_rank2())
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "components": tuple(
                    _composite_binding(component, f"l0_rank{position + 1}")
                    for position, component in enumerate(components)
                ),
                "source": _mixed_l_source_config(),
            },
        }
    )
    feature_count = sum(
        len(component.payload["descriptors"]) for component in components
    )
    target = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": (
                tuple(np.linspace(-0.08, 0.11, feature_count)),
            ),
            "offsets": (-1.2,),
            "source_realization": "direct",
        },
    )
    structures = []
    for scale in (1.0, 1.07, 0.94):
        atoms = Atoms("Ta4", positions=_positions().numpy() * scale, pbc=False)
        prediction = target.evaluate_atoms(atoms, forces=True)
        atoms.info["energy"] = float(prediction["energy"].detach())
        atoms.arrays["forces"] = prediction["forces"].detach().numpy()
        structures.append(atoms)

    template = lifted_cauchy_model_from_descriptor(
        descriptor, {"source_realization": "direct"}
    )
    cache = materialize_lifted_cauchy_regression_row_cache(
        template,
        structures,
        feature_chunk_size=8,
        fit_coordinate_policy="orthogonal",
    )
    cached_problem = build_lifted_cauchy_normal_equations_from_row_cache(cache)
    geometry_cache = materialize_lifted_cauchy_geometry_row_cache(
        template,
        structures,
        feature_chunk_size=8,
        fit_coordinate_policy="orthogonal",
    )
    changed_source = _mixed_l_source_config()
    changed_source["cutoff_A"] = 4.8
    changed_descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "components": tuple(
                    _composite_binding(component, f"l0_rank{position + 1}")
                    for position, component in enumerate(components)
                ),
                "source": changed_source,
            },
        }
    )
    changed_template = lifted_cauchy_model_from_descriptor(
        changed_descriptor, {"source_realization": "direct"}
    )
    original_request = lifted_cauchy_geometry_row_cache_request(
        template, structures, fit_coordinate_policy="orthogonal"
    )
    changed_request = lifted_cauchy_geometry_row_cache_request(
        changed_template, structures, fit_coordinate_policy="orthogonal"
    )
    assert tuple(
        component.self_hash for component in changed_template.compiled_components
    ) == tuple(component.self_hash for component in template.compiled_components)
    assert changed_template.source.source_plan_hash != template.source.source_plan_hash
    assert changed_request["request_hash"] != original_request["request_hash"]
    target_cache = materialize_lifted_cauchy_target_cache(
        geometry_cache,
        structures,
        target_identity={"reference_stack": "none"},
    )
    split_problem = build_lifted_cauchy_normal_equations_from_geometry_and_targets(
        geometry_cache,
        target_cache,
    )
    unweighted_request = lifted_cauchy_normal_equation_cache_request(
        combine_lifted_cauchy_geometry_and_target_caches(
            geometry_cache, target_cache
        )
    )
    weighted_request = lifted_cauchy_normal_equation_cache_request(
        combine_lifted_cauchy_geometry_and_target_caches(
            geometry_cache, target_cache
        ),
        structure_weights=(1.0, 2.0, 1.0),
    )
    assert unweighted_request["request_hash"] != weighted_request["request_hash"]
    np.testing.assert_array_equal(split_problem["XtX"], cached_problem["XtX"])
    np.testing.assert_array_equal(split_problem["Xty"], cached_problem["Xty"])
    assert split_problem["yty"] == cached_problem["yty"]
    assert split_problem["geometry_cache_hash"] == geometry_cache["cache_hash"]
    changed_structures = [atoms.copy() for atoms in structures]
    for atoms, source in zip(changed_structures, structures, strict=True):
        atoms.info["energy"] = float(source.info["energy"]) + 0.25
        atoms.arrays["forces"] = np.asarray(source.arrays["forces"]) + 0.1
    changed_targets = materialize_lifted_cauchy_target_cache(
        geometry_cache,
        changed_structures,
        target_identity={"reference_stack": "changed"},
    )
    changed_problem = build_lifted_cauchy_normal_equations_from_geometry_and_targets(
        geometry_cache,
        changed_targets,
    )
    assert changed_targets["cache_hash"] != target_cache["cache_hash"]
    assert changed_problem["geometry_cache_hash"] == geometry_cache["cache_hash"]
    assert not np.array_equal(changed_problem["Xty"], split_problem["Xty"])
    persistent_root = tmp_path / "linear_cache"
    geometry_directory = persist_lifted_cauchy_geometry_row_cache(
        geometry_cache, persistent_root
    )
    persist_lifted_cauchy_target_cache(target_cache, persistent_root)
    mapped_geometry = load_lifted_cauchy_geometry_row_cache(
        geometry_cache["request_hash"], persistent_root
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_lifted_cauchy_geometry_row_cache("../outside", persistent_root)
    mapped_targets = load_lifted_cauchy_target_cache(
        target_cache["request_hash"], persistent_root
    )
    assert isinstance(mapped_geometry["records"][0]["site_design"], np.memmap)
    np.testing.assert_array_equal(
        mapped_geometry["records"][0]["site_design"],
        geometry_cache["records"][0]["site_design"],
    )
    np.testing.assert_array_equal(
        mapped_targets["records"][0]["force_target"],
        target_cache["records"][0]["force_target"],
    )
    mapped_problem = build_lifted_cauchy_normal_equations_from_geometry_and_targets(
        mapped_geometry,
        mapped_targets,
    )
    np.testing.assert_array_equal(mapped_problem["XtX"], split_problem["XtX"])
    persist_lifted_cauchy_normal_equations(split_problem, persistent_root)
    mapped_normal_equations = load_lifted_cauchy_normal_equations(
        split_problem["normal_equation_request_hash"], persistent_root
    )
    assert isinstance(mapped_normal_equations["XtX"], np.memmap)
    assert mapped_normal_equations["problem_hash"] == split_problem["problem_hash"]
    np.testing.assert_array_equal(
        mapped_normal_equations["XtX"], split_problem["XtX"]
    )
    np.testing.assert_array_equal(
        mapped_normal_equations["Xty"], split_problem["Xty"]
    )
    site_path = next(geometry_directory.glob("record_*_site.npy"))
    corrupted = bytearray(site_path.read_bytes())
    corrupted[-1] ^= 1
    site_path.write_bytes(corrupted)
    with pytest.raises(LinearCacheValidationError, match="hash mismatch"):
        load_lifted_cauchy_geometry_row_cache(
            geometry_cache["request_hash"], persistent_root
        )
    persist_lifted_cauchy_geometry_row_cache(geometry_cache, persistent_root)
    recovered_geometry = load_lifted_cauchy_geometry_row_cache(
        geometry_cache["request_hash"], persistent_root
    )
    np.testing.assert_array_equal(
        recovered_geometry["records"][0]["site_design"],
        geometry_cache["records"][0]["site_design"],
    )
    stale_lock = geometry_directory.parent / (geometry_directory.name + ".lock")
    stale_lock.write_text(
        json.dumps({"hostname": socket.gethostname(), "pid": 2147483647}),
        encoding="utf-8",
    )
    persist_lifted_cauchy_geometry_row_cache(geometry_cache, persistent_root)
    assert not stale_lock.exists()
    direct_problem = build_lifted_cauchy_normal_equations(
        template,
        structures,
        feature_chunk_size=8,
        fit_coordinate_policy="orthogonal",
    )
    np.testing.assert_allclose(
        cached_problem["XtX"], direct_problem["XtX"], atol=2.0e-11, rtol=2.0e-11
    )
    np.testing.assert_allclose(
        cached_problem["Xty"], direct_problem["Xty"], atol=2.0e-11, rtol=2.0e-11
    )
    np.testing.assert_allclose(
        cached_problem["feature_mean"],
        direct_problem["feature_mean"],
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        cached_problem["feature_scale"],
        direct_problem["feature_scale"],
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    assert abs(cached_problem["yty"] - direct_problem["yty"]) < 2.0e-11

    cached_solution = solve_lifted_cauchy_normal_equations(
        cached_problem, ridge_alpha=1.0e-8
    )
    direct_solution = solve_lifted_cauchy_normal_equations(
        direct_problem, ridge_alpha=1.0e-8
    )
    np.testing.assert_allclose(
        cached_solution["fit_coordinate_coefficients"],
        direct_solution["fit_coordinate_coefficients"],
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    fitted = lifted_cauchy_model_from_cached_solution(
        descriptor, cache, cached_problem, cached_solution
    )
    runtime, offsets = _lifted_cauchy_cached_runtime_parameters(
        cache, cached_problem, cached_solution
    )
    np.testing.assert_array_equal(runtime, fitted.coefficients.detach().numpy())
    np.testing.assert_array_equal(offsets, fitted.offsets.detach().numpy())
    score = score_lifted_cauchy_cached_solution(
        cache, cached_problem, cached_solution
    )
    assert score["finite"]
    assert not cache["records"][0]["site_design"].flags.writeable
    assert not cache["records"][0]["force_design"].flags.writeable
    selected_coordinate = cache["descriptor_coordinate_ids"][:1]
    subset_problem = build_lifted_cauchy_normal_equations_from_row_cache(
        cache,
        record_indices=(0, 1),
        descriptor_coordinate_ids=selected_coordinate,
    )
    assert subset_problem["selected_descriptor_coordinate_ids"] == (
        selected_coordinate
    )
    subset_parent = build_lifted_cauchy_normal_equations_from_row_cache(
        cache,
        record_indices=(0, 1),
    )
    sliced_problem = subset_lifted_cauchy_normal_equations_from_parent(
        cache,
        subset_parent,
        descriptor_coordinate_ids=selected_coordinate,
    )
    assert sliced_problem["selected_feature_indices"] == subset_problem[
        "selected_feature_indices"
    ]
    np.testing.assert_allclose(
        sliced_problem["XtX"], subset_problem["XtX"], atol=2.0e-12, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        sliced_problem["Xty"], subset_problem["Xty"], atol=2.0e-12, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        sliced_problem["feature_mean"],
        subset_problem["feature_mean"],
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        sliced_problem["feature_scale"],
        subset_problem["feature_scale"],
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    subset_solution = solve_lifted_cauchy_normal_equations(
        subset_problem, ridge_alpha=1.0e-8
    )
    held_out = score_lifted_cauchy_cached_solution(
        cache,
        subset_problem,
        subset_solution,
        record_indices=(2,),
    )
    assert held_out["finite"]
    assert held_out["structure_count"] == 1
    tampered_solution = dict(subset_solution)
    tampered_solution["problem_hash"] = "0" * 64
    with pytest.raises(ValueError, match="problem identities differ"):
        score_lifted_cauchy_cached_solution(
            cache,
            subset_problem,
            tampered_solution,
            record_indices=(2,),
        )
    for atoms in structures:
        prediction = fitted.evaluate_atoms(atoms, forces=True)
        assert torch.isfinite(prediction["energy"])
        assert torch.all(torch.isfinite(prediction["forces"]))


def test_composite_component_identity_rejects_overlapping_sectors():
    from ye3t_methods.atomistic import YE3TDescriptors

    compiled = _compiled_mixed_l_scalar()
    first = _composite_binding(compiled, "first")
    second = _composite_binding(compiled, "second")
    assert second["opportunity_id"] != first["opportunity_id"]
    second["strict_sector_ids"] = first["strict_sector_ids"]
    with pytest.raises(ValueError, match="overlap a compiler strict output sector"):
        YE3TDescriptors.ye3t_basis(
            {
                "elements": ("Ta",),
                "type_map": {"Ta": 0},
                "lifted_cauchy": {
                    "components": (first, second),
                    "source": _mixed_l_source_config(),
                },
            }
        )


def test_cached_descriptor_coordinate_selection_expands_all_central_heads():
    from ase import Atoms

    from ye3t_methods.atomistic import YE3TDescriptors
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        build_lifted_cauchy_normal_equations_from_row_cache,
        lifted_cauchy_model_from_descriptor,
        materialize_lifted_cauchy_regression_row_cache,
    )

    compiled = _compiled_mixed_l_scalar()
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta", "W"),
            "type_map": {"Ta": 0, "W": 1},
            "lifted_cauchy": {
                "components": (_composite_binding(compiled, "rank1_l0"),),
                "source": _mixed_l_source_config(),
            },
        }
    )
    model = lifted_cauchy_model_from_descriptor(descriptor)
    atoms = Atoms("Ta2", positions=((0.0, 0.0, 0.0), (2.4, 0.0, 0.0)))
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = np.zeros((2, 3))
    cache = materialize_lifted_cauchy_regression_row_cache(
        model, (atoms,), feature_chunk_size=8
    )
    selected = cache["descriptor_coordinate_ids"][:1]
    problem = build_lifted_cauchy_normal_equations_from_row_cache(
        cache, descriptor_coordinate_ids=selected
    )
    width = len(cache["descriptor_coordinate_ids"])
    assert problem["selected_feature_indices"] == (0, width)
    assert len(problem["selected_feature_coordinate_ids"]) == 2


def test_joint_source_has_the_regular_solid_harmonic_origin_derivative():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    compiled = _compiled_joint(2)
    direct = LiftedCauchyPolynomialSource(
        compiled,
        source_config=_joint_source_config(),
        type_map={"Ta": 0},
        source_realization="direct",
    )
    factorized = LiftedCauchyPolynomialSource(
        compiled,
        source_config=_joint_source_config(),
        type_map={"Ta": 0},
        source_realization="factorized",
    )
    displacement = torch.zeros((1, 3), dtype=torch.float64)
    neighbor_types = torch.zeros(1, dtype=torch.long)
    direct_value, direct_derivative = direct.edge_values_with_dx(
        displacement, neighbor_types
    )
    factorized_value, factorized_derivative = factorized.edge_values_with_dx(
        displacement, neighbor_types
    )
    assert torch.count_nonzero(direct_value).item() == 0
    assert torch.count_nonzero(factorized_value).item() == 0
    assert torch.isfinite(direct_derivative).all()
    transformed_derivative = torch.zeros_like(factorized_derivative)
    for group in factorized.config["groups"]:
        indices = torch.as_tensor(group["channel_indices"], dtype=torch.long)
        matrix = torch.as_tensor(
            group["factorized_lowering"]["binary64_matrix"],
            dtype=torch.float64,
        )
        selected = factorized_derivative.index_select(1, indices)
        work = torch.einsum(
            "qp,apij->aqij", matrix, selected.reshape(1, -1, 3, 3)
        )
        transformed_derivative.index_copy_(
            1, indices, work.reshape_as(selected)
        )
    assert torch.max(
        torch.abs(direct_derivative - transformed_derivative)
    ).item() < 5.0e-11
    assert torch.max(torch.abs(direct_derivative)).item() > 0.0

    step = 1.0e-8
    numerical = torch.zeros_like(direct_derivative)
    for axis in range(3):
        delta = torch.zeros_like(displacement)
        delta[:, axis] = step
        plus = direct.edge_values_with_dx(delta, neighbor_types)[0]
        minus = direct.edge_values_with_dx(-delta, neighbor_types)[0]
        numerical[..., axis] = (plus - minus) / (2.0 * step)
    assert torch.max(torch.abs(direct_derivative - numerical)).item() < 2.0e-5


def test_joint_source_plan_hash_rejects_semantic_tampering():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    compiled = _compiled_joint(2)
    source = LiftedCauchyPolynomialSource(
        compiled,
        source_config=_joint_source_config(),
        type_map={"Ta": 0},
    )
    tampered = json.loads(json.dumps(source.config))
    tampered["groups"][0]["polynomials"][1][
        "shifted_jacobi_power_coefficients"
    ][0] += 1
    with pytest.raises(ValueError, match="field 'groups' is inconsistent"):
        LiftedCauchyPolynomialSource(
            compiled, source_config=tampered, type_map={"Ta": 0}
        )


def test_polynomial_source_radial_and_tangential_directional_derivatives():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    source = LiftedCauchyPolynomialSource(
        _compiled(), source_config=_source_config(), type_map={"Ta": 0}
    )
    displacement = torch.tensor(((1.3, -0.4, 0.7),), dtype=torch.float64)
    atom_types = torch.zeros(1, dtype=torch.long)
    _values, derivative = source.edge_values_with_dx(displacement, atom_types)
    radial = displacement[0] / torch.linalg.norm(displacement[0])
    tangent = torch.linalg.cross(
        radial,
        torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64),
    )
    tangent = tangent / torch.linalg.norm(tangent)
    step = 2.0e-6
    for direction in (radial, tangent):
        plus = source.edge_values_with_dx(
            displacement + step * direction, atom_types
        )[0]
        minus = source.edge_values_with_dx(
            displacement - step * direction, atom_types
        )[0]
        numerical = (plus - minus) / (2.0 * step)
        analytical = torch.einsum("ecsqd,d->ecsq", derivative, direction)
        assert torch.max(torch.abs(analytical - numerical)).item() < 3.0e-9
        assert torch.linalg.norm(analytical).item() > 1.0e-6


def test_polynomial_role_sum_recovers_unweighted_source_and_derivative():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    source = LiftedCauchyPolynomialSource(
        _compiled(), source_config=_source_config(), type_map={"Ta": 0}
    )
    displacement = torch.tensor(
        ((1.3, -0.4, 0.7), (0.5, 1.1, -0.8)), dtype=torch.float64
    )
    atom_types = torch.zeros(2, dtype=torch.long)
    values, derivative = source.edge_values_with_dx(displacement, atom_types)
    distance = torch.linalg.norm(displacement, dim=1)
    x = distance / source.cutoff
    unit = displacement / distance[:, None]
    physical = torch.stack((unit[:, 0], unit[:, 2], -unit[:, 1]), dim=1)
    envelope = (1.0 - x).square()
    for channel_index, channel in enumerate(source.channels):
        exponent = int(channel["radial_channel"])
        expected = envelope[:, None] * x.pow(exponent)[:, None] * physical
        assert torch.max(
            torch.abs(values[:, channel_index].sum(dim=1) - expected)
        ).item() < 2.0e-15
        step = 2.0e-6
        numerical = torch.zeros_like(derivative[:, channel_index, 0])
        for axis in range(3):
            delta = torch.zeros_like(displacement)
            delta[:, axis] = step
            plus = source.edge_values_with_dx(
                displacement + delta, atom_types
            )[0][:, channel_index].sum(dim=1)
            minus = source.edge_values_with_dx(
                displacement - delta, atom_types
            )[0][:, channel_index].sum(dim=1)
            numerical[..., axis] = (plus - minus) / (2.0 * step)
        assert torch.max(
            torch.abs(derivative[:, channel_index].sum(dim=1) - numerical)
        ).item() < 3.0e-9


def test_polynomial_source_rejects_unimplemented_source_identity_and_radial_map():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    wrong_family = first_lifted_cauchy_scalar_request(
        1,
        family_ids=("NT_NU4_K22_L0",),
        source_family_id="another_radial_source",
    )
    with pytest.raises(ValueError, match="does not implement compiled source_family_id"):
        LiftedCauchyPolynomialSource(
            compile_coupling(wrong_family),
            source_config=_source_config(),
            type_map={"Ta": 0},
        )

    sparse_radial = first_lifted_cauchy_scalar_request(
        2, family_ids=("NT_NU4_K22_L0",)
    )
    channels = tuple(dict(channel) for channel in sparse_radial["channels"])
    channels[1]["radial_channel"] = 3
    sparse_radial["channels"] = channels
    with pytest.raises(ValueError, match="dense zero-based radial polynomial"):
        LiftedCauchyPolynomialSource(
            compile_coupling(sparse_radial),
            source_config=_source_config(),
            type_map={"Ta": 0},
        )


def test_polynomial_source_enforces_periodic_image_mode():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    atom_types = torch.zeros(4, dtype=torch.long)
    nonperiodic = LiftedCauchyPolynomialSource(
        _compiled(), source_config=_source_config(), type_map={"Ta": 0}
    )
    with pytest.raises(ValueError, match="rejects periodic geometry"):
        nonperiodic.materialize(_positions(), atom_types, pbc=True)

    explicit_config = {**_source_config(), "periodic_image_mode": "explicit"}
    explicit = LiftedCauchyPolynomialSource(
        _compiled(), source_config=explicit_config, type_map={"Ta": 0}
    )
    with pytest.raises(ValueError, match="requires an explicit edge list"):
        explicit.materialize(_positions(), atom_types)


def test_polynomial_source_explicit_vjp_matches_autograd_and_strain():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    source = LiftedCauchyPolynomialSource(
        _compiled(), source_config=_source_config(), type_map={"Ta": 0}
    )
    positions = _positions().requires_grad_(True)
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    density, context = source.materialize(positions, atom_types)
    generator = torch.Generator().manual_seed(31)
    adjoint = torch.randn(
        density.shape, dtype=density.dtype, generator=generator
    )
    automatic = torch.autograd.grad((density * adjoint).sum(), positions)[0]
    explicit = source.vjp(adjoint, context, atom_count=len(positions))
    assert torch.max(
        torch.abs(automatic - explicit["position_gradient"])
    ).item() < 2.0e-12

    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    transform = torch.eye(3, dtype=torch.float64) + strain
    strained = _positions() @ transform.T
    strained_density, _ = source.materialize(strained, atom_types)
    automatic_strain = torch.autograd.grad(
        (strained_density * adjoint).sum(), strain
    )[0]
    assert torch.max(
        torch.abs(automatic_strain - explicit["strain_derivative"])
    ).item() < 3.0e-12


def test_polynomial_source_shifted_periodic_edge_vjp_and_strain():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    config = {**_source_config(), "periodic_image_mode": "all_images"}
    source = LiftedCauchyPolynomialSource(
        _compiled(), source_config=config, type_map={"Ta": 0}
    )
    positions = torch.tensor(
        ((0.2, 0.4, 0.6), (5.3, 0.5, 0.7)),
        dtype=torch.float64,
        requires_grad=True,
    )
    cell = 6.0 * torch.eye(3, dtype=torch.float64)
    atom_types = torch.zeros(2, dtype=torch.long)
    edge_index = torch.tensor(((0, 1), (1, 0)), dtype=torch.long)
    shifts = torch.tensor(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    density, context = source.materialize(
        positions,
        atom_types,
        edge_index=edge_index,
        cell=cell,
        shifts=shifts,
        pbc=True,
    )
    generator = torch.Generator().manual_seed(37)
    adjoint = torch.randn(density.shape, dtype=density.dtype, generator=generator)
    automatic = torch.autograd.grad((density * adjoint).sum(), positions)[0]
    explicit = source.vjp(adjoint, context, atom_count=2)
    assert torch.max(
        torch.abs(automatic - explicit["position_gradient"])
    ).item() < 3.0e-12

    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    transform = torch.eye(3, dtype=torch.float64) + strain
    strained_density, _ = source.materialize(
        positions.detach() @ transform.T,
        atom_types,
        edge_index=edge_index,
        cell=cell @ transform.T,
        shifts=shifts,
        pbc=True,
    )
    automatic_strain = torch.autograd.grad(
        (strained_density * adjoint).sum(), strain
    )[0]
    assert torch.max(
        torch.abs(automatic_strain - explicit["strain_derivative"])
    ).item() < 4.0e-12


@pytest.mark.parametrize("realization", ("canonical", "factored"))
def test_torch_evaluator_matches_compiler_reference_value_and_vjp(realization):
    from ye3t.couplings import evaluate_lifted_cauchy_scalar
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyTorchEvaluator

    generator = torch.Generator().manual_seed(47)
    density = torch.randn(
        (2, 2, 2, 3), dtype=torch.float64, generator=generator
    ).requires_grad_(True)
    evaluator = LiftedCauchyTorchEvaluator(_compiled())
    output = evaluator.evaluate(density, realization=realization)
    upstream = torch.randn(output.shape, dtype=output.dtype, generator=generator)
    gradient = torch.autograd.grad((output * upstream).sum(), density)[0]
    explicit_output, explicit_gradient = evaluator.vjp(
        density.detach(),
        upstream,
        realization=realization,
        method="explicit",
    )
    batched_upstream = torch.stack((upstream, -0.3 * upstream), dim=0)
    _batched_output, batched_gradient = evaluator.vjp(
        density.detach(),
        batched_upstream,
        realization=realization,
        method="explicit",
    )
    assert torch.max(torch.abs(explicit_output - output.detach())).item() < 2.0e-13
    assert torch.max(torch.abs(explicit_gradient - gradient)).item() < 3.0e-11
    assert torch.max(torch.abs(batched_gradient[0] - gradient)).item() < 3.0e-11
    assert torch.max(
        torch.abs(batched_gradient[1] + 0.3 * gradient)
    ).item() < 3.0e-11

    reference_outputs = []
    reference_gradients = []
    for atom in range(2):
        values = {
            channel: density.detach().numpy()[atom, channel]
            for channel in range(2)
        }
        ref_output, ref_gradient = evaluate_lifted_cauchy_scalar(
            _compiled(),
            values,
            realization=realization,
            upstream=upstream.detach().numpy()[atom],
            input_basis="real_tesseral",
        )
        reference_outputs.append(ref_output)
        reference_gradients.append(
            np.stack(tuple(ref_gradient[channel] for channel in range(2)))
        )
    assert np.max(
        np.abs(output.detach().numpy() - np.stack(reference_outputs))
    ) < 2.0e-12
    assert np.max(
        np.abs(gradient.detach().numpy() - np.stack(reference_gradients))
    ) < 2.0e-11


def test_canonical_and_symmetric_power_block_paths_match_on_geometry():
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LiftedCauchyPolynomialSource,
        LiftedCauchyTorchEvaluator,
    )

    source = LiftedCauchyPolynomialSource(
        _compiled(), source_config=_source_config(), type_map={"Ta": 0}
    )
    evaluator = LiftedCauchyTorchEvaluator(_compiled())
    positions = _positions().requires_grad_(True)
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    density, _ = source.materialize(positions, atom_types)
    canonical = evaluator.evaluate(density, realization="canonical")
    factored = evaluator.evaluate(density, realization="factored")
    assert torch.max(torch.abs(canonical - factored)).item() < 2.0e-12
    generator = torch.Generator().manual_seed(59)
    upstream = torch.randn(canonical.shape, dtype=canonical.dtype, generator=generator)
    canonical_grad = torch.autograd.grad(
        (canonical * upstream).sum(), positions, retain_graph=True
    )[0]
    factored_grad = torch.autograd.grad((factored * upstream).sum(), positions)[0]
    assert torch.max(torch.abs(canonical_grad - factored_grad)).item() < 3.0e-11


@pytest.mark.parametrize("source_kind", ("compatibility", "orthogonal_joint"))
def test_lifted_cauchy_scalar_geometry_symmetries_and_neighbor_permutation(
    source_kind,
):
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LiftedCauchyPolynomialSource,
        LiftedCauchyTorchEvaluator,
    )

    if source_kind == "compatibility":
        compiled = _compiled_all_nontrivial()
        source_config = _source_config()
    else:
        compiled = _compiled_joint(3)
        source_config = _joint_source_config()
    source = LiftedCauchyPolynomialSource(
        compiled, source_config=source_config, type_map={"Ta": 0}
    )
    evaluator = LiftedCauchyTorchEvaluator(compiled)
    atom_types = torch.zeros(4, dtype=torch.long)

    def features(positions):
        density, _ = source.materialize(positions, atom_types)
        return evaluator.evaluate(density, realization="factored")

    positions = _positions()
    baseline = features(positions)
    rotation = _random_rotation()
    rotated = features(positions @ rotation.T)
    inverted = features(-positions)
    translated = features(positions + torch.tensor((1.2, -0.7, 0.9)))
    assert torch.max(torch.abs(baseline - rotated)).item() < 3.0e-11
    assert torch.max(torch.abs(baseline - inverted)).item() < 3.0e-12
    assert torch.max(torch.abs(baseline - translated)).item() < 3.0e-12

    permutation = torch.tensor((2, 0, 3, 1), dtype=torch.long)
    permuted = features(positions.index_select(0, permutation))
    inverse = torch.argsort(permutation)
    assert torch.max(
        torch.abs(baseline - permuted.index_select(0, inverse))
    ).item() < 3.0e-12

    generator = torch.Generator().manual_seed(71)
    weights = torch.randn(baseline.shape[1], generator=generator)

    def forces(position_values):
        position_values = position_values.detach().requires_grad_(True)
        energy = (features(position_values) * weights).sum()
        return -torch.autograd.grad(energy, position_values)[0]

    baseline_forces = forces(positions)
    rotated_forces = forces(positions @ rotation.T)
    inverted_forces = forces(-positions)
    assert torch.max(
        torch.abs(rotated_forces - baseline_forces @ rotation.T)
    ).item() < 5.0e-10
    assert torch.max(
        torch.abs(inverted_forces + baseline_forces)
    ).item() < 5.0e-11


def test_explicit_edge_storage_order_does_not_change_values_or_source_vjp():
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        LiftedCauchyPolynomialSource,
        LiftedCauchyTorchEvaluator,
    )

    source = LiftedCauchyPolynomialSource(
        _compiled(), source_config=_source_config(), type_map={"Ta": 0}
    )
    evaluator = LiftedCauchyTorchEvaluator(_compiled())
    positions = _positions()
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    density, context = source.materialize(positions, atom_types)
    generator = torch.Generator().manual_seed(171)
    order = torch.randperm(context["edge_index"].shape[1], generator=generator)
    shuffled_density, shuffled_context = source.materialize(
        positions,
        atom_types,
        edge_index=context["edge_index"].index_select(1, order),
        pbc=False,
    )
    assert torch.max(torch.abs(density - shuffled_density)).item() < 2.0e-15
    assert torch.max(
        torch.abs(
            evaluator.evaluate(density, realization="factored")
            - evaluator.evaluate(shuffled_density, realization="factored")
        )
    ).item() < 2.0e-14
    adjoint = torch.randn(density.shape, dtype=density.dtype, generator=generator)
    baseline = source.vjp(adjoint, context)
    shuffled = source.vjp(adjoint, shuffled_context)
    assert torch.max(
        torch.abs(
            baseline["position_gradient"] - shuffled["position_gradient"]
        )
    ).item() < 3.0e-15
    assert torch.max(
        torch.abs(
            baseline["strain_derivative"] - shuffled["strain_derivative"]
        )
    ).item() < 3.0e-14


def test_all_nontrivial_families_canonical_factored_and_role_collapse_tangent():
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyTorchEvaluator

    compiled = _compiled_all_nontrivial()
    evaluator = LiftedCauchyTorchEvaluator(compiled)
    assert evaluator.descriptor_count == 42
    generator = torch.Generator().manual_seed(73)
    density = torch.randn(
        (2, 3, 2, 3), dtype=torch.float64, generator=generator,
        requires_grad=True,
    )
    canonical = evaluator.evaluate(density, realization="canonical")
    factored = evaluator.evaluate(density, realization="factored")
    assert torch.max(torch.abs(canonical - factored)).item() < 5.0e-11

    collapsed_base = torch.randn(
        (2, 3, 1, 3), dtype=torch.float64, generator=generator
    )
    collapsed = collapsed_base.repeat(1, 1, 2, 1).requires_grad_(True)
    output = evaluator.evaluate(collapsed, realization="factored")
    assert torch.max(torch.abs(output)).item() < 5.0e-11
    upstream = torch.randn(output.shape, dtype=output.dtype, generator=generator)
    gradient = torch.autograd.grad((output * upstream).sum(), collapsed)[0]
    tangent_base = torch.randn(
        (2, 3, 1, 3), dtype=torch.float64, generator=generator
    )
    tangent = tangent_base.repeat(1, 1, 2, 1)
    assert torch.abs((gradient * tangent).sum()).item() < 8.0e-10

    role_vector = torch.tensor((0.35, -1.2), dtype=torch.float64)
    proportional_base = torch.randn(
        (2, 3, 1, 3), dtype=torch.float64, generator=generator
    )
    proportional = (
        proportional_base * role_vector.reshape(1, 1, 2, 1)
    ).requires_grad_(True)
    proportional_output = evaluator.evaluate(
        proportional, realization="factored"
    )
    assert torch.max(torch.abs(proportional_output)).item() < 8.0e-11
    proportional_upstream = torch.randn(
        proportional_output.shape,
        dtype=proportional_output.dtype,
        generator=generator,
    )
    proportional_gradient = torch.autograd.grad(
        (proportional_output * proportional_upstream).sum(), proportional
    )[0]
    base_tangent = torch.randn(
        (2, 3, 1, 3), dtype=torch.float64, generator=generator
    )
    role_tangent = torch.tensor((0.11, -0.07), dtype=torch.float64)
    tangent = (
        base_tangent * role_vector.reshape(1, 1, 2, 1)
        + proportional_base * role_tangent.reshape(1, 1, 2, 1)
    )
    assert torch.abs((proportional_gradient * tangent).sum()).item() < 2.0e-9


def test_source_keeps_neighbor_species_blocks_separate():
    from ye3t.couplings import compile as compile_coupling
    from ye3t.couplings import first_lifted_cauchy_scalar_request
    from ye3t_methods.atomistic.lifted_cauchy_linear import LiftedCauchyPolynomialSource

    request = first_lifted_cauchy_scalar_request(
        1, element="Ta", family_ids=("NT_NU4_K22_L0",)
    )
    request["channels"] = (
        request["channels"][0],
        {
            **request["channels"][0],
            "channel_id": 1,
            "neighbor_species": "W",
        },
    )
    compiled = compile_coupling(request)
    source = LiftedCauchyPolynomialSource(
        compiled,
        source_config=_source_config(),
        type_map={"Ta": 0, "W": 1},
    )
    disp = torch.tensor(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=torch.float64)
    values, _ = source.edge_values_with_dx(
        disp, torch.tensor((0, 1), dtype=torch.long)
    )
    assert torch.count_nonzero(values[0, 0]).item() > 0
    assert torch.count_nonzero(values[0, 1]).item() == 0
    assert torch.count_nonzero(values[1, 0]).item() == 0
    assert torch.count_nonzero(values[1, 1]).item() > 0


def test_central_species_heads_survive_numeric_type_remapping():
    from ye3t_methods.atomistic import YE3TDescriptors
    from ye3t_methods.atomistic.lifted_cauchy_linear import lifted_cauchy_model_from_descriptor

    def descriptor(type_map):
        return YE3TDescriptors.ye3t_basis(
            {
                "elements": ("Ta", "W"),
                "type_map": type_map,
                "lifted_cauchy": {
                    "compiled": _compiled().to_dict(),
                    "source": _source_config(),
                },
            }
        )

    model_a = lifted_cauchy_model_from_descriptor(
        descriptor({"Ta": 0, "W": 1}),
        {
            "coefficients": ((1.25, -0.5, 0.75), (-0.4, 0.8, 1.1)),
            "offsets": (0.3, -0.2),
            "realization": "factored",
        },
    )
    model_b = lifted_cauchy_model_from_descriptor(
        descriptor({"Ta": 1, "W": 0}),
        {
            "coefficients": ((1.25, -0.5, 0.75), (-0.4, 0.8, 1.1)),
            "offsets": (0.3, -0.2),
            "realization": "factored",
        },
    )
    positions = _positions()
    energy_a = model_a(positions, torch.tensor((0, 1, 0, 1)))
    energy_b = model_b(positions, torch.tensor((1, 0, 1, 0)))
    assert torch.abs(energy_a - energy_b).item() < 2.0e-12


def test_public_linear_fit_recovers_synthetic_energies_forces_and_force_sign(tmp_path):
    ase = pytest.importorskip("ase")
    from ye3t_methods.atomistic import YE3TDescriptors, YE3TModel
    from ye3t_methods.atomistic.lifted_cauchy_linear import lifted_cauchy_model_from_descriptor

    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "compiled": _compiled().to_dict(),
                "source": _source_config(),
            },
        }
    )
    truth = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": ((0.7, -1.1, 0.45),),
            "offsets": (-0.3,),
            "realization": "factored",
        },
    )
    structures = []
    rng = np.random.default_rng(83)
    for _index in range(6):
        positions = _positions().numpy() + rng.normal(scale=0.08, size=(4, 3))
        atoms = ase.Atoms("Ta4", positions=positions, pbc=False)
        target = truth.evaluate_atoms(atoms, forces=True)
        atoms.info["energy"] = float(target["energy"].detach())
        atoms.arrays["forces"] = target["forces"].detach().numpy()
        structures.append(atoms)

    fitted = YE3TModel.linear(
        descriptor,
        {
            "energy_weight": 1.0,
            "force_weight": 1.0,
            "ridge_alpha": 0.0,
            "svd_rcond": 1.0e-13,
            "feature_chunk_size": 2,
            "realization": "factored",
        },
        structures=structures,
    )
    assert fitted.fit_metadata["problem"]["objective"] == (
        "structure_balanced_train_scaled_E1_F1"
    )
    assert fitted.fit_metadata["problem"]["fit_coordinate_policy"] == "orthogonal"
    assert fitted.fit_metadata["problem"]["fit_coordinate_normalization"] == (
        "compiler_metric_unit_norm"
    )
    assert fitted.fit_metadata["problem"]["orthogonal_output_plan_hash"]
    assert fitted.fit_metadata["problem"]["dataset_passes"] == 2
    assert fitted.fit_metadata["problem"]["retained_structure_count"] == 1
    assert fitted.fit_metadata["problem"]["materialized_training_design"] is False
    assert fitted.fit_metadata["problem"]["dense_fallback_status"] == "not_used"
    max_energy_error = 0.0
    max_force_error = 0.0
    for atoms in structures:
        predicted = fitted.evaluate_atoms(atoms, forces=True)
        max_energy_error = max(
            max_energy_error,
            abs(float(predicted["energy"].detach()) - atoms.info["energy"]),
        )
        max_force_error = max(
            max_force_error,
            float(
                torch.max(
                    torch.abs(
                        predicted["forces"]
                        - torch.as_tensor(atoms.arrays["forces"])
                    )
                )
            ),
        )
    assert max_energy_error < 2.0e-9
    assert max_force_error < 2.0e-9

    ordinary_descriptor = YE3TDescriptors.ace(
        {
            "elements": ["Ta"],
            "type_map": {"Ta": 0},
            "cutoff": 5.2,
            "ranks": [1],
            "basis_type": "no_charge",
            "k_o_max": 0,
            "k_max": [0],
            "nmax": [1],
            "lmax": [0],
            "lmin": [0],
            "L_R": 0,
            "M_R_values": [0],
            "max_labels_per_rank": 1,
            "max_variants_per_label": 1,
            "site_basis_config": {
                "rc": [5.2],
                "lmbda": [0.25],
                "nradmax": 1,
                "lmax": 0,
                "kmax": 0,
                "possible_types": [0],
                "radial_basis": "PACE_ChebExpCos",
                "chemical_basis": "delta",
                "charge_mode": "none",
                "atomic_base_normalization": "none",
                "factor_normalization": "none",
                "spherical_backend": "complex",
                "spherical_normalization": "pace_y00_one",
                "source_backend": "torch",
                "dtype": "float64",
                "pace_cutoff_width": [0.8],
                "pace_spline_spacing": [0.1],
                "pace_inner_cutoff": [0.0],
                "pace_inner_cutoff_width": [0.0],
                "pace_crad_policy": "identity",
            },
            "backend": "pytorch",
        }
    )
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        build_ordinary_lifted_cauchy_normal_equations,
        build_ordinary_lifted_cauchy_regression_problem,
    )

    joint_template = lifted_cauchy_model_from_descriptor(
        descriptor,
        {"realization": "factored", "source_realization": "auto"},
    )
    dense_joint = build_ordinary_lifted_cauchy_regression_problem(
        joint_template,
        ordinary_descriptor,
        structures[:2],
        energy_key="energy",
        force_key="forces",
        energy_weight=1.0,
        force_weight=1.0,
        feature_chunk_size=2,
        ordinary_descriptor_matrix_cache=None,
    )
    streamed_joint = build_ordinary_lifted_cauchy_normal_equations(
        joint_template,
        ordinary_descriptor,
        structures[:2],
        feature_chunk_size=2,
    )
    joint_design = np.asarray(dense_joint["scaled_design"], dtype=np.float64)
    joint_target = np.asarray(dense_joint["weighted_target"], dtype=np.float64)
    dense_gram = joint_design.T @ joint_design
    dense_rhs = joint_design.T @ joint_target
    eps = np.finfo(np.float64).eps
    gram_roundoff = 64.0 * eps * np.maximum(
        np.abs(joint_design).T @ np.abs(joint_design), 1.0
    )
    rhs_roundoff = 64.0 * eps * np.maximum(
        np.abs(joint_design).T @ np.abs(joint_target), 1.0
    )
    np.testing.assert_array_less(
        np.abs(streamed_joint["XtX"] - dense_gram),
        gram_roundoff + 2.0e-10 * np.abs(dense_gram),
    )
    np.testing.assert_array_less(
        np.abs(streamed_joint["Xty"] - dense_rhs),
        rhs_roundoff + 2.0e-10 * np.abs(dense_rhs),
    )
    assert np.linalg.norm(streamed_joint["XtX"][0, 1:4]) > 1.0e-8
    joint = YE3TModel.linear(
        descriptor,
        {
            "ordinary_descriptor": ordinary_descriptor,
            "energy_weight": 1.0,
            "force_weight": 1.0,
            "ridge_alpha": 0.0,
            "svd_rcond": 1.0e-13,
            "feature_chunk_size": "auto",
            "realization": "factored",
        },
        structures=structures,
    )
    assert joint.fit_metadata["ordinary_feature_count"] == 1
    assert joint.fit_metadata["lifted_feature_count"] == 3
    assert joint.fit_metadata["fit_method"] == "ridge_streaming_gram"
    assert joint.fit_metadata["problem"]["dataset_passes"] == 2
    assert joint.fit_metadata["problem"]["ordinary_lifted_cross_gram_retained"] is True
    assert joint.fit_metadata["problem"]["materialized_training_design"] is False
    assert joint.fit_metadata["problem"]["dense_fallback_status"] == "not_used"
    assert joint.fit_metadata["problem"]["feature_chunk"]["resolved"] == 4
    for atoms in structures:
        predicted = joint.evaluate_atoms(atoms, forces=True)
        assert (
            abs(float(predicted["energy"].detach()) - atoms.info["energy"])
            < 3.0e-8
        )
        assert torch.max(
            torch.abs(
                predicted["forces"] - torch.as_tensor(atoms.arrays["forces"])
            )
        ).item() < 3.0e-8

    from ye3t_methods.atomistic import (
        export_lifted_cauchy_linear_bundle,
        load_lifted_cauchy_linear_bundle,
    )

    joint_bundle = tmp_path / "joint_bundle"
    export_lifted_cauchy_linear_bundle(joint, joint_bundle)
    loaded_joint = load_lifted_cauchy_linear_bundle(joint_bundle)
    assert loaded_joint["ordinary_yace_path"].name == "ordinary.yace"
    assert loaded_joint["ordinary_yace_path"].is_file()

    atoms = structures[0].copy()
    analytical = fitted.evaluate_atoms(atoms, forces=True)["forces"].detach().numpy()
    step = 2.0e-6
    displaced_plus = atoms.copy()
    displaced_minus = atoms.copy()
    displaced_plus.positions[1, 2] += step
    displaced_minus.positions[1, 2] -= step
    energy_plus = float(fitted.evaluate_atoms(displaced_plus, forces=False)["energy"])
    energy_minus = float(fitted.evaluate_atoms(displaced_minus, forces=False)["energy"])
    numerical_force = -(energy_plus - energy_minus) / (2.0 * step)
    assert abs(numerical_force - analytical[1, 2]) < 3.0e-8
    assert np.linalg.norm(np.sum(analytical, axis=0)) < 2.0e-10
    assert np.linalg.norm(np.sum(np.cross(atoms.positions, analytical), axis=0)) < 2.0e-9

    strain_derivative = fitted.evaluate_atoms(
        atoms, forces=False, stress=True
    )["strain_derivative"].detach().numpy()
    direction = np.zeros((3, 3))
    direction[0, 1] = 0.5
    direction[1, 0] = 0.5

    def strained_energy(scale):
        transform = np.eye(3) + scale * direction
        strained = atoms.copy()
        strained.positions = atoms.positions @ transform.T
        strained.set_cell(np.asarray(atoms.cell) @ transform.T, scale_atoms=False)
        return float(fitted.evaluate_atoms(strained, forces=False)["energy"])

    numerical_strain = (strained_energy(step) - strained_energy(-step)) / (
        2.0 * step
    )
    analytical_strain = float(np.sum(strain_derivative * direction))
    assert abs(numerical_strain - analytical_strain) < 4.0e-8


def test_streamed_lifted_gram_matches_dense_reference_and_structure_weights():
    ase = pytest.importorskip("ase")
    from ye3t_methods.atomistic import YE3TDescriptors
    from ye3t_methods.atomistic.lifted_cauchy_linear import (
        build_lifted_cauchy_normal_equations,
        build_lifted_cauchy_regression_problem,
        lifted_cauchy_model_from_descriptor,
    )

    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "compiled": _compiled().to_dict(),
                "source": _source_config(),
            },
        }
    )
    truth = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": ((0.7, -1.1, 0.45),),
            "offsets": (-0.3,),
            "realization": "factored",
        },
    )
    structures = []
    for structure_index, shift in enumerate((0.0, 0.17)):
        positions = _positions().numpy().copy()
        positions[1, 0] += shift
        atoms = ase.Atoms("Ta4", positions=positions, pbc=False)
        target = truth.evaluate_atoms(atoms, forces=True)
        atoms.info["energy"] = (
            float(target["energy"].detach()) + 0.2 * structure_index
        )
        atoms.arrays["forces"] = target["forces"].detach().numpy()
        structures.append(atoms)

    dense = build_lifted_cauchy_regression_problem(
        truth, structures, feature_chunk_size=2
    )
    streamed = build_lifted_cauchy_normal_equations(
        truth, structures, feature_chunk_size=2
    )
    design = np.asarray(dense["scaled_design"], dtype=np.float64)
    target = np.asarray(dense["weighted_target"], dtype=np.float64)
    np.testing.assert_allclose(streamed["XtX"], design.T @ design, atol=2.0e-12)
    np.testing.assert_allclose(streamed["Xty"], design.T @ target, atol=2.0e-12)
    np.testing.assert_allclose(streamed["yty"], target @ target, atol=2.0e-12)

    weights = np.asarray((0.25, 1.75), dtype=np.float64)
    weighted = build_lifted_cauchy_normal_equations(
        truth,
        structures,
        feature_chunk_size=2,
        structure_weights=weights,
    )
    row_scale = []
    for _kind, structure_index, block_size in dense["metadata"]["row_blocks"]:
        row_scale.extend((math.sqrt(weights[structure_index]),) * block_size)
    row_scale = np.asarray(row_scale, dtype=np.float64)
    weighted_design = design * row_scale[:, None]
    weighted_target = target * row_scale
    np.testing.assert_allclose(
        weighted["XtX"], weighted_design.T @ weighted_design, atol=2.0e-12
    )
    np.testing.assert_allclose(
        weighted["Xty"], weighted_design.T @ weighted_target, atol=2.0e-12
    )
    np.testing.assert_allclose(
        weighted["yty"], weighted_target @ weighted_target, atol=2.0e-12
    )
    assert weighted["metadata"]["structure_weights"]["sources"] == (
        "explicit_sequence",
    )
    with pytest.raises(NotImplementedError, match="requested device"):
        build_lifted_cauchy_normal_equations(
            truth, structures, device="cuda"
        )
    with pytest.raises(ValueError, match="float64 evaluation"):
        build_lifted_cauchy_normal_equations(
            truth, structures, evaluation_dtype="float32"
        )


def test_complete_periodic_shifted_image_energy_strain_derivative():
    ase = pytest.importorskip("ase")
    from ye3t_methods.atomistic import YE3TDescriptors
    from ye3t_methods.atomistic.lifted_cauchy_linear import lifted_cauchy_model_from_descriptor

    source_config = _source_config()
    source_config["periodic_image_mode"] = "all_images"
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "compiled": _compiled().to_dict(),
                "source": source_config,
            },
        }
    )
    model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": ((0.7, -1.1, 0.45),),
            "offsets": (-0.3,),
            "realization": "factored",
        },
    )
    atoms = ase.Atoms(
        "Ta2",
        positions=((0.3, 0.4, 0.5), (9.1, 0.6, 0.4)),
        cell=np.diag((10.0, 10.0, 10.0)),
        pbc=True,
    )
    analytical = model.evaluate_atoms(
        atoms, forces=True, stress=True
    )
    direction = np.zeros((3, 3))
    direction[0, 1] = 0.5
    direction[1, 0] = 0.5
    step = 2.0e-6

    def energy_at(scale):
        transform = np.eye(3) + scale * direction
        probe = atoms.copy()
        probe.positions = atoms.positions @ transform.T
        probe.set_cell(np.asarray(atoms.cell) @ transform.T, scale_atoms=False)
        return float(model.evaluate_atoms(probe, forces=False)["energy"])

    numerical = (energy_at(step) - energy_at(-step)) / (2.0 * step)
    expected = float(
        torch.sum(
            analytical["strain_derivative"]
            * torch.as_tensor(direction, dtype=torch.float64)
        )
    )
    assert abs(numerical - expected) < 8.0e-8
    forces = analytical["forces"].detach().numpy()
    assert np.linalg.norm(np.sum(forces, axis=0)) < 2.0e-10


def test_lifted_cauchy_bundle_round_trip_and_tamper_rejection(tmp_path):
    from ye3t_methods.atomistic import (
        YE3TDescriptors,
        export_lifted_cauchy_linear_bundle,
        load_lifted_cauchy_linear_bundle,
    )
    from ye3t_methods.atomistic.lifted_cauchy_linear import lifted_cauchy_model_from_descriptor

    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta", "W"),
            "type_map": {"Ta": 2, "W": 0},
            "lifted_cauchy": {
                "compiled": _compiled().to_dict(),
                "source": _source_config(),
            },
        }
    )
    model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": ((1.25, -0.5, 0.75), (-0.4, 0.8, 1.1)),
            "offsets": (0.3, -0.2),
            "realization": "factored",
        },
    )
    model.fit_metadata = {
        "model_family": "linear_lifted_cauchy_scalar",
        "finite_diagnostic": np.float64(1.25),
    }
    bundle_dir = tmp_path / "lifted_bundle"
    model_path = export_lifted_cauchy_linear_bundle(model, bundle_dir)
    loaded = load_lifted_cauchy_linear_bundle(bundle_dir)
    assert loaded["ordinary_yace_path"] is None
    assert loaded["model_payload"]["self_hash"]
    positions = _positions()
    atom_types = torch.tensor((2, 0, 2, 0), dtype=torch.long)
    reference = model(positions, atom_types)
    restored = loaded["lifted_model"](positions, atom_types)
    assert torch.abs(reference - restored).item() < 2.0e-13

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload["readout"]["coefficients"][0][0] += 0.25
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model JSON SHA-256 mismatch"):
        load_lifted_cauchy_linear_bundle(bundle_dir)


def test_joint_source_bundle_round_trip_can_switch_exact_source_backend(tmp_path):
    from ye3t_methods.atomistic import (
        YE3TDescriptors,
        export_lifted_cauchy_linear_bundle,
        load_lifted_cauchy_linear_bundle,
    )
    from ye3t_methods.atomistic.lifted_cauchy_linear import lifted_cauchy_model_from_descriptor

    compiled = _compiled_joint(2)
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "compiled": compiled.to_dict(),
                "source": _joint_source_config(),
            },
        }
    )
    factorized = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": (
                tuple(np.linspace(-0.7, 1.1, len(compiled.payload["descriptors"]))),
            ),
            "offsets": (0.2,),
            "realization": "factored",
            "source_realization": "factorized",
        },
    )
    bundle_dir = tmp_path / "joint_source_bundle"
    model_path = export_lifted_cauchy_linear_bundle(factorized, bundle_dir)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "ye3t_lifted_cauchy_linear_bundle_v3"
    assert payload["source_plan_hash"] == factorized.source.source_plan_hash
    assert payload["default_source_realization"] == "factorized"

    loaded = load_lifted_cauchy_linear_bundle(
        bundle_dir, source_realization="direct"
    )
    direct = loaded["lifted_model"]
    native = loaded["native_runtime_payload"]
    assert native["schema"] == "ye3t_lifted_cauchy_native_runtime_v2"
    assert native["coordinate_convention"]["source_variable_order"] == (
        "packed_channel_then_role_then_real_component"
    )
    assert native["source_plan_hash"] == factorized.source.source_plan_hash
    assert native["capabilities"]["runtime_gram_solve"] is False
    positions = _positions().requires_grad_(True)
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    factorized_energy = factorized(positions, atom_types)
    factorized_force = -torch.autograd.grad(
        factorized_energy, positions, retain_graph=True
    )[0]
    direct_energy = direct(positions, atom_types)
    direct_force = -torch.autograd.grad(direct_energy, positions)[0]
    assert torch.abs(factorized_energy - direct_energy).item() < 5.0e-11
    assert torch.max(torch.abs(factorized_force - direct_force)).item() < 5.0e-10

    native_positions = _positions().requires_grad_(True)
    native_density, _context = direct.source.materialize(
        native_positions, atom_types
    )
    native_variables = native_density.reshape(len(native_positions), -1)
    polynomial = native["heads"][0]["polynomial"]
    native_atomic = torch.full(
        (len(native_positions),),
        float(polynomial["offset"]),
        dtype=torch.float64,
    )
    for term, coefficient in enumerate(polynomial["monomial_coefficients"]):
        value = torch.full_like(native_atomic, float(coefficient))
        begin = int(polynomial["factor_offsets"][term])
        end = int(polynomial["factor_offsets"][term + 1])
        for factor in range(begin, end):
            source_index = int(polynomial["factor_indices"][factor])
            exponent = int(polynomial["factor_exponents"][factor])
            value = value * native_variables[:, source_index].pow(exponent)
        native_atomic = native_atomic + value
    native_energy = native_atomic.sum()
    native_force = -torch.autograd.grad(native_energy, native_positions)[0]
    reference_positions = _positions().requires_grad_(True)
    reference_energy = direct(reference_positions, atom_types)
    reference_force = -torch.autograd.grad(reference_energy, reference_positions)[0]
    assert torch.abs(native_energy - reference_energy).item() < 5.0e-11
    assert torch.max(torch.abs(native_force - reference_force)).item() < 5.0e-10

    native_path = loaded["native_runtime_path"]
    native_tampered = json.loads(native_path.read_text(encoding="utf-8"))
    native_tampered["heads"][0]["polynomial"]["offset"] += 0.125
    native_path.write_text(json.dumps(native_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="native-runtime file SHA-256 mismatch"):
        load_lifted_cauchy_linear_bundle(bundle_dir)
