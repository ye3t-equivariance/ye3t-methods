"""Low-rank semantic fixtures for the guarded LAMMPS-PACE YACE route."""

import math
import re
from pathlib import Path

import pytest
import torch
import yaml

from ye3t_ace.ace.yace import read_yace, write_yace


_FIXTURE = Path(__file__).parent / "fixtures" / "yace_minimal_linear_ta.yace"
_MULTI_ELEMENT_FIXTURE = (
    Path(__file__).parent / "fixtures" / "yace_minimal_linear_ta_w.yace"
)
_COMPATIBILITY = "lammps_pace_linear_v1"


def test_minimal_standard_yace_loads_sequence_bond_key_and_linear_records():
    payload = read_yace(_FIXTURE, compatibility=_COMPATIBILITY)

    assert payload["elements"] == ["Ta"]
    assert payload["E0"] == [0.25]
    assert payload["deltaSplineBins"] == pytest.approx(0.001)
    assert set(payload["bonds"]) == {(0, 0)}
    assert payload["bonds"][(0, 0)]["radbasename"] == "ChebExpCos"
    assert [function.rank for function in payload["functions"][0]] == [1, 2]
    assert payload["functions"][0][1].ms_combs == (-1, 1, 0, 0, 1, -1)


def test_linear_profile_accepts_both_pace_identity_embedding_names(tmp_path):
    text = _FIXTURE.read_text(encoding="utf-8")
    text = text.replace(
        "FinnisSinclair",
        "FinnisSinclairShiftedScaled",
        1,
    )
    path = tmp_path / "finnis_sinclair_identity.yace"
    path.write_text(text, encoding="utf-8")

    payload = read_yace(path, compatibility=_COMPATIBILITY)

    assert payload["embeddings"][0]["npoti"] == "FinnisSinclairShiftedScaled"


def test_multi_element_profile_preserves_ordered_types_and_directed_bonds():
    payload = read_yace(
        _MULTI_ELEMENT_FIXTURE,
        compatibility=_COMPATIBILITY,
    )

    assert payload["elements"] == ["Ta", "W"]
    assert payload["E0"] == [0.25, -0.1]
    assert set(payload["embeddings"]) == {0, 1}
    assert {entry["ndensity"] for entry in payload["embeddings"].values()} == {1}
    assert set(payload["bonds"]) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert payload["bonds"][(0, 1)]["nradmax"] == 2
    assert payload["bonds"][(1, 0)]["nradbasemax"] == 2
    assert payload["functions"][0][0].mus == (1,)
    assert payload["functions"][0][0].ns == (2,)
    assert payload["functions"][1][0].mu0 == 1


def test_multi_element_profile_requires_complete_directed_bonds(tmp_path):
    text = _MULTI_ELEMENT_FIXTURE.read_text(encoding="utf-8")
    start = text.index("  [1, 1]:\n")
    stop = text.index("functions:\n")
    path = tmp_path / "missing_directed_bond.yace"
    path.write_text(text[:start] + text[stop:], encoding="utf-8")

    with pytest.raises(ValueError, match=r"^bonds: expected one directed record"):
        read_yace(path, compatibility=_COMPATIBILITY)


def test_multi_element_profile_checks_pair_specific_radial_bounds(tmp_path):
    text = _MULTI_ELEMENT_FIXTURE.read_text(encoding="utf-8")
    marker = "      mus: [1]\n      ns: [2]\n"
    assert marker in text
    path = tmp_path / "bad_pair_radial_index.yace"
    path.write_text(
        text.replace(marker, "      mus: [1]\n      ns: [3]\n", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"^functions\.0\[0\]\.ns\[0\]"):
        read_yace(path, compatibility=_COMPATIBILITY)


def test_multi_element_profile_rejects_duplicate_element_names(tmp_path):
    text = _MULTI_ELEMENT_FIXTURE.read_text(encoding="utf-8")
    path = tmp_path / "duplicate_elements.yace"
    path.write_text(
        text.replace("elements: [Ta, W]", "elements: [Ta, Ta]", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"^elements: element names must be unique"):
        read_yace(path, compatibility=_COMPATIBILITY)


def test_multi_element_profile_keeps_one_density_per_element_limit(tmp_path):
    text = _MULTI_ELEMENT_FIXTURE.read_text(encoding="utf-8")
    marker = "  1:\n    ndensity: 1\n"
    assert marker in text
    path = tmp_path / "two_density_second_element.yace"
    path.write_text(
        text.replace(marker, "  1:\n    ndensity: 2\n", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"^embeddings\.1\.ndensity"):
        read_yace(path, compatibility=_COMPATIBILITY)


@pytest.mark.parametrize(
    ("old", "new", "error_path"),
    (
        (
            "nradmax: 1",
            "nradmax: 2",
            "bonds.[0,0].radcoefficients",
        ),
        (
            "radparameters: [0.5723]",
            "radparameters: [0.5723, 0.7]",
            "bonds.[0,0].radparameters",
        ),
        (
            "num_ms_combs: 3",
            "num_ms_combs: 2",
            "functions.0[1].ms_combs",
        ),
        (
            "prehc: 0.0",
            "prehc: 1.0",
            "bonds.[0,0].prehc",
        ),
        (
            "inner_cutoff_type: distance",
            "inner_cutoff_type: density",
            "bonds.[0,0].inner_cutoff_type",
        ),
        (
            "FinnisSinclair",
            "unsupported_embedding",
            "embeddings.0.npoti",
        ),
    ),
)
def test_linear_profile_rejects_unsupported_or_inconsistent_fields(
    tmp_path,
    old,
    new,
    error_path,
):
    text = _FIXTURE.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "invalid.yace"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"^{re.escape(error_path)}"):
        read_yace(path, compatibility=_COMPATIBILITY)


def test_linear_profile_rejects_unknown_top_level_field(tmp_path):
    text = _FIXTURE.read_text(encoding="utf-8") + "unsupported: true\n"
    path = tmp_path / "unknown_field.yace"
    path.write_text(text, encoding="utf-8")

    assert read_yace(path)["unsupported"] is True
    with pytest.raises(ValueError, match=r"^yace: unsupported field 'unsupported'"):
        read_yace(path, compatibility=_COMPATIBILITY)


def test_linear_profile_rejects_missing_required_field(tmp_path):
    text = _FIXTURE.read_text(encoding="utf-8")
    text = text.replace("deltaSplineBins: 0.001\n", "", 1)
    path = tmp_path / "missing_field.yace"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"^yace: missing required field 'deltaSplineBins'",
    ):
        read_yace(path, compatibility=_COMPATIBILITY)


def test_yace_loader_rejects_duplicate_mapping_key(tmp_path):
    text = _FIXTURE.read_text(encoding="utf-8") + "E0: [0.5]\n"
    path = tmp_path / "duplicate_key.yace"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'E0'"):
        read_yace(path)


def test_unknown_compatibility_profile_fails_closed():
    with pytest.raises(ValueError, match="unsupported profile"):
        read_yace(_FIXTURE, compatibility="unknown")


@pytest.mark.parametrize("fixture", [_FIXTURE, _MULTI_ELEMENT_FIXTURE])
def test_strict_writer_preserves_complete_pace_document(tmp_path, fixture):
    payload = read_yace(fixture, compatibility=_COMPATIBILITY)
    path = tmp_path / fixture.name

    write_yace(
        path,
        elements=payload["elements"],
        E0=payload["E0"],
        delta_spline_bins=payload["deltaSplineBins"],
        embeddings=payload["embeddings"],
        bonds=payload["bonds"],
        functions_by_mu0=payload["functions"],
        compatibility=_COMPATIBILITY,
    )

    actual = read_yace(path, compatibility=_COMPATIBILITY)
    assert actual == payload


def test_strict_writer_fails_before_writing_incomplete_document(tmp_path):
    payload = read_yace(_FIXTURE, compatibility=_COMPATIBILITY)
    path = tmp_path / "incomplete.yace"

    with pytest.raises(ValueError, match=r"^yace: missing required field"):
        write_yace(
            path,
            elements=payload["elements"],
            E0=payload["E0"],
            functions_by_mu0=payload["functions"],
            compatibility=_COMPATIBILITY,
        )

    assert not path.exists()


def test_writer_metadata_cannot_replace_standard_field(tmp_path):
    with pytest.raises(ValueError, match="must not replace YACE field 'elements'"):
        write_yace(
            tmp_path / "overridden.yace",
            elements=["Ta"],
            functions_by_mu0={},
            metadata={"elements": ["W"]},
        )


def _complex_cartesian_gradients(values, vectors):
    rows = []
    for component in range(values.shape[1]):
        real_gradient = torch.autograd.grad(
            values[:, component].real.sum(),
            vectors,
            retain_graph=True,
        )[0]
        imaginary_gradient = torch.autograd.grad(
            values[:, component].imag.sum(),
            vectors,
            retain_graph=True,
        )[0]
        rows.append(torch.complex(real_gradient, imaginary_gradient))
    return torch.stack(rows, dim=1)


def _rank2_l1_scalar_plan():
    from ye3t.couplings import symmetric_power_product_plan

    return symmetric_power_product_plan(
        (
            {
                "descriptor_index": 0,
                "channel_indices": (0, 1, 2),
                "power": 2,
                "input_L": 1,
                "output_L": 0,
                "multiplicity_index": 0,
                "component_index": 0,
            },
        ),
        descriptor_count=1,
        channel_count=3,
        factor_basis="A",
        normalization_convention="none",
        basis_convention="complex_magnetic",
    )


def _pace_g0_with_derivative(radii, cutoff):
    phase = math.pi * radii / cutoff
    values = 0.5 * (1.0 + torch.cos(phase))
    derivatives = -0.5 * math.pi / cutoff * torch.sin(phase)
    return values, derivatives


def _explicit_single_l_ctilde_terms(function, atomic_base):
    assert len(set(function.ls)) == 1
    magnetic_offset = function.ls[0]
    terms = []
    for row, coefficient in enumerate(function.ctildes):
        start = row * function.rank
        factors = atomic_base.new_ones(atomic_base.shape[0])
        for magnetic in function.ms_combs[start:start + function.rank]:
            factors = factors * atomic_base[:, magnetic + magnetic_offset]
        terms.append(coefficient * factors)
    return torch.stack(terms, dim=1)


def _explicit_single_l_ctilde_algebraic_root(function, atomic_base):
    assert len(set(function.ls)) == 1
    magnetic_offset = function.ls[0]
    root = torch.zeros_like(atomic_base)
    for row, coefficient in enumerate(function.ctildes):
        start = row * function.rank
        magnetic_row = function.ms_combs[start:start + function.rank]
        for factor, magnetic in enumerate(magnetic_row):
            product = atomic_base.new_full(
                (atomic_base.shape[0],),
                coefficient,
            )
            for other, other_magnetic in enumerate(magnetic_row):
                if other != factor:
                    product = product * atomic_base[
                        :, other_magnetic + magnetic_offset
                    ]
            root[:, magnetic + magnetic_offset] += product
    return root


def test_pace_scaled_complex_l1_and_real_transform_values_and_derivatives(
    monkeypatch,
):
    from ye3t.runtime.execution_plan import spherical_harmonics_with_derivative

    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    vectors = torch.tensor(
        ((1.0, 2.0, 3.0), (-2.0, 0.5, 1.25)),
        dtype=torch.float64,
    )
    complex_values, complex_derivatives = spherical_harmonics_with_derivative(
        vectors,
        1,
        real_output=False,
        backend="native",
    )
    pace_scale = math.sqrt(4.0 * math.pi)
    actual_values = pace_scale * complex_values
    actual_derivatives = pace_scale * complex_derivatives

    reference_vectors = vectors.detach().clone().requires_grad_(True)
    unit = reference_vectors / torch.linalg.norm(
        reference_vectors,
        dim=1,
        keepdim=True,
    )
    prefactor = math.sqrt(1.5)
    positive = -prefactor * torch.complex(unit[:, 0], unit[:, 1])
    reference_values = torch.stack(
        (
            -positive.conj(),
            torch.complex(
                math.sqrt(3.0) * unit[:, 2],
                torch.zeros_like(unit[:, 2]),
            ),
            positive,
        ),
        dim=1,
    )
    reference_derivatives = _complex_cartesian_gradients(
        reference_values,
        reference_vectors,
    )
    torch.testing.assert_close(
        actual_values,
        reference_values,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        actual_derivatives,
        reference_derivatives,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        actual_values[:, 0],
        -actual_values[:, 2].conj(),
        rtol=0.0,
        atol=2.0e-14,
    )

    real_values, real_derivatives = spherical_harmonics_with_derivative(
        vectors,
        1,
        real_output=True,
        backend="native",
    )
    transformed_values = torch.stack(
        (
            math.sqrt(2.0) * complex_values[:, 2].imag,
            complex_values[:, 1].real,
            -math.sqrt(2.0) * complex_values[:, 2].real,
        ),
        dim=1,
    )
    transformed_derivatives = torch.stack(
        (
            math.sqrt(2.0) * complex_derivatives[:, 2].imag,
            complex_derivatives[:, 1].real,
            -math.sqrt(2.0) * complex_derivatives[:, 2].real,
        ),
        dim=1,
    )
    torch.testing.assert_close(
        real_values,
        transformed_values,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        real_derivatives,
        transformed_derivatives,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    pace_real_negative = -math.sqrt(2.0) * actual_values[:, 2].imag
    torch.testing.assert_close(
        pace_real_negative,
        -pace_scale * real_values[:, 0],
        rtol=0.0,
        atol=2.0e-14,
    )


def test_rank2_ctilde_matches_compiler_native_symmetric_power_value_and_vjp(
    monkeypatch,
):
    import ye3t.runtime.execution_plan as execution_plan
    from ye3t.runtime.symmetric_power import (
        symmetric_power_product_plan_batched_adjoint,
        symmetric_power_product_plan_contraction,
    )

    payload = read_yace(_FIXTURE, compatibility=_COMPATIBILITY)
    function = payload["functions"][0][1]
    assert function.rank == 2
    assert function.ls == (1, 1)
    plan = _rank2_l1_scalar_plan()
    assert plan.validation_report["valid_labels_from"] == "ye3t.couplings.count"
    assert plan.validation_report["basis_convention"] == "complex_magnetic"
    assert plan.factor_basis == "A"
    assert plan.normalization_convention == "none"
    terms = {
        tuple(term["exponents"]): complex(term["coefficient"])
        for term in plan.entries[0].component_terms
    }
    assert terms == pytest.approx({(0, 2, 0): 1.0, (1, 0, 1): -2.0})

    atomic_base = torch.tensor(
        (
            (-0.3, 0.7, 0.3),
            (0.4, -0.2, -0.6),
            (0.0, 0.25, 0.0),
            (0.0, 0.0, 0.0),
        ),
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_input = atomic_base.detach().clone().requires_grad_(True)
    reference = reference_input.new_zeros((reference_input.shape[0], 1))
    for row, coefficient in enumerate(function.ctildes):
        first_m = function.ms_combs[2 * row]
        second_m = function.ms_combs[2 * row + 1]
        reference[:, 0] = reference[:, 0] + coefficient * (
            reference_input[:, first_m + 1]
            * reference_input[:, second_m + 1]
        )

    def forbid_reference_fallback(*args, **kwargs):
        raise AssertionError("strict native dispatch entered a reference fallback")

    monkeypatch.setattr(
        execution_plan,
        "symmetric_power_monomial_reference",
        forbid_reference_fallback,
    )
    monkeypatch.setattr(
        execution_plan,
        "symmetric_power_shared_monomial_batched_adjoint_reference",
        forbid_reference_fallback,
    )
    # The PACE rows give (2 A_-1 A_1 - A_0^2)/sqrt(3); the compiler plan
    # gives A_0^2 - 2 A_-1 A_1 in the same m=-1,0,1 channel ordering.
    scale = -1.0 / math.sqrt(3.0)
    actual = scale * symmetric_power_product_plan_contraction(
        atomic_base,
        plan,
        backend="native",
    )
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=1.0e-14)

    seeds = torch.tensor(
        (((1.0,), (-0.5,), (2.0,), (0.75,)),),
        dtype=torch.float64,
    )
    actual_roots = symmetric_power_product_plan_batched_adjoint(
        scale * seeds,
        atomic_base,
        plan,
        backend="native",
    )
    reference_roots = torch.stack(
        [
            torch.autograd.grad(
                reference,
                reference_input,
                seeds[seed],
                retain_graph=True,
            )[0]
            for seed in range(seeds.shape[0])
        ],
        dim=0,
    )
    torch.testing.assert_close(
        actual_roots,
        reference_roots,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert torch.count_nonzero(actual_roots[:, -1]) == 0


def test_fixed_octant_environment_A_descriptor_and_edge_adjoint(monkeypatch):
    import ye3t.runtime.execution_plan as execution_plan
    from ye3t.runtime.execution_plan import (
        density_accumulate,
        density_accumulate_adjoint,
        plain_site_basis_product_adjoint,
        plain_site_basis_product_with_derivative,
        spherical_harmonics_with_derivative,
    )
    from ye3t.runtime.symmetric_power import (
        symmetric_power_product_plan_batched_adjoint,
        symmetric_power_product_plan_contraction,
    )

    monkeypatch.setenv("YE3T_ENABLE_EXECUTION_PLAN_JIT", "1")
    payload = read_yace(_FIXTURE, compatibility=_COMPATIBILITY)
    rank_one, rank_two = payload["functions"][0]
    bond = payload["bonds"][(0, 0)]
    assert rank_one.ns == (1,)
    assert rank_two.ns == (1, 1)
    assert rank_two.ls == (1, 1)
    assert bond["radcoefficients"][0][1] == [1.0]

    vectors = torch.tensor(
        (
            (0.31, 0.47, 0.83),
            (-0.42, 0.58, 0.27),
            (0.73, -0.24, 0.51),
            (-0.67, -0.19, 0.37),
            (0.28, 0.63, -0.44),
            (-0.35, 0.22, -0.91),
            (0.59, -0.71, -0.18),
            (-0.76, -0.41, -0.29),
        ),
        dtype=torch.float64,
    )
    centers = torch.zeros(vectors.shape[0], dtype=torch.int64)
    radii = torch.linalg.norm(vectors, dim=1)
    unit = vectors / radii[:, None]
    cutoff = float(bond["rcut"])
    assert float(radii.max()) < cutoff - float(bond["dcut"])
    radial, radial_derivative = _pace_g0_with_derivative(radii, cutoff)

    angular, angular_derivative = spherical_harmonics_with_derivative(
        vectors,
        1,
        real_output=False,
        backend="native",
    )
    pace_scale = math.sqrt(4.0 * math.pi)
    angular = pace_scale * angular
    angular_derivative = pace_scale * angular_derivative
    complex_radial = torch.complex(radial, torch.zeros_like(radial))[None, :]
    complex_radial_derivative = torch.complex(
        radial_derivative,
        torch.zeros_like(radial_derivative),
    )[None, :]
    group_ones = torch.ones_like(complex_radial)
    group_zeros = torch.zeros_like(complex_radial)
    edge_values, edge_derivatives, charge_center, charge_neighbor = (
        plain_site_basis_product_with_derivative(
            complex_radial,
            complex_radial_derivative,
            angular.transpose(0, 1),
            angular_derivative.permute(1, 0, 2),
            group_ones,
            group_zeros,
            group_zeros,
            unit.to(torch.complex128),
            (0, 0, 0),
            (0, 1, 2),
            3,
            backend="native",
        )
    )
    expected_edges = radial[:, None] * angular
    expected_edge_derivatives = (
        radial_derivative[:, None, None]
        * unit[:, None, :]
        * angular[:, :, None]
        + radial[:, None, None] * angular_derivative
    )
    torch.testing.assert_close(
        edge_values,
        expected_edges,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        edge_derivatives,
        expected_edge_derivatives,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(charge_center, torch.zeros_like(charge_center))
    torch.testing.assert_close(charge_neighbor, torch.zeros_like(charge_neighbor))

    atomic_base = density_accumulate(
        edge_values,
        centers,
        1,
        backend="native",
    )
    torch.testing.assert_close(
        atomic_base,
        expected_edges.sum(dim=0, keepdim=True),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        atomic_base[:, 0],
        -atomic_base[:, 2].conj(),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert torch.all(atomic_base.abs() > 1.0e-8)
    reverse = torch.arange(vectors.shape[0] - 1, -1, -1)
    reordered_base = density_accumulate(
        edge_values[reverse],
        centers[reverse],
        1,
        backend="native",
    )
    torch.testing.assert_close(
        reordered_base,
        atomic_base,
        rtol=2.0e-15,
        atol=2.0e-15,
    )

    explicit_terms = _explicit_single_l_ctilde_terms(rank_two, atomic_base)[0]
    assert torch.all(explicit_terms.abs() > 1.0e-8)
    explicit_descriptor = explicit_terms.sum().reshape(1, 1)

    def forbid_reference_fallback(*args, **kwargs):
        raise AssertionError("strict native dispatch entered a reference fallback")

    monkeypatch.setattr(
        execution_plan,
        "symmetric_power_monomial_reference",
        forbid_reference_fallback,
    )
    monkeypatch.setattr(
        execution_plan,
        "symmetric_power_shared_monomial_batched_adjoint_reference",
        forbid_reference_fallback,
    )
    plan = _rank2_l1_scalar_plan()
    plan_scale = -1.0 / math.sqrt(3.0)
    native_descriptor = plan_scale * symmetric_power_product_plan_contraction(
        atomic_base,
        plan,
        backend="native",
    )
    torch.testing.assert_close(
        native_descriptor,
        explicit_descriptor,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        explicit_descriptor.imag,
        torch.zeros_like(explicit_descriptor.real),
        rtol=0.0,
        atol=2.0e-14,
    )

    native_hilbert_root = symmetric_power_product_plan_batched_adjoint(
        torch.full(
            (1, 1, 1),
            plan_scale,
            dtype=torch.complex128,
        ),
        atomic_base,
        plan,
        backend="native",
    )[0]
    explicit_root = _explicit_single_l_ctilde_algebraic_root(
        rank_two,
        atomic_base,
    )
    autograd_base = atomic_base.detach().clone().requires_grad_(True)
    autograd_descriptor = _explicit_single_l_ctilde_terms(
        rank_two,
        autograd_base,
    ).sum()
    autograd_root = torch.autograd.grad(
        autograd_descriptor.real,
        autograd_base,
    )[0]
    torch.testing.assert_close(
        native_hilbert_root,
        autograd_root,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    # The native VJP is a Hermitian/PyTorch adjoint. PACE's force weights are
    # algebraic roots in Re(sum_m root_m * dA_m), hence this explicit bridge.
    pace_algebraic_root = native_hilbert_root.conj()
    torch.testing.assert_close(
        pace_algebraic_root,
        explicit_root,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    embedding_scale = float(payload["embeddings"][0]["FS_parameters"][0])
    rank_one_density = rank_one.ctildes[0] * radial.sum()
    energy = (
        embedding_scale * (rank_one_density + explicit_descriptor.real[0, 0])
        + payload["E0"][0]
    )
    atomic_root = embedding_scale * pace_algebraic_root
    edge_root = density_accumulate_adjoint(
        atomic_root,
        centers,
        backend="native",
    )
    edge_weights = torch.ones(vectors.shape[0], dtype=torch.complex128)
    edge_weight_derivatives = torch.zeros(
        (vectors.shape[0], 3),
        dtype=torch.complex128,
    )
    rank_two_edge_gradient, charge_center, charge_neighbor = (
        plain_site_basis_product_adjoint(
            complex_radial,
            complex_radial_derivative,
            angular.transpose(0, 1),
            angular_derivative.permute(1, 0, 2),
            group_ones,
            group_zeros,
            group_zeros,
            unit.to(torch.complex128),
            edge_weights,
            edge_weight_derivatives,
            (0, 0, 0),
            (0, 1, 2),
            edge_root,
            backend="native",
        )
    )
    torch.testing.assert_close(
        rank_two_edge_gradient.imag,
        torch.zeros_like(rank_two_edge_gradient.real),
        rtol=0.0,
        atol=2.0e-13,
    )
    torch.testing.assert_close(charge_center, torch.zeros_like(charge_center))
    torch.testing.assert_close(charge_neighbor, torch.zeros_like(charge_neighbor))
    rank_one_edge_gradient = (
        embedding_scale
        * rank_one.ctildes[0]
        * radial_derivative[:, None]
        * unit
    )
    edge_gradient = rank_one_edge_gradient + rank_two_edge_gradient.real
    assert torch.all(edge_gradient.abs() > 1.0e-8)

    def explicit_energy(displacements):
        local_radii = torch.linalg.norm(displacements, dim=1)
        local_radial = _pace_g0_with_derivative(local_radii, cutoff)[0]
        local_angular = pace_scale * spherical_harmonics_with_derivative(
            displacements,
            1,
            real_output=False,
            backend="native",
        )[0]
        local_base = density_accumulate(
            local_radial[:, None] * local_angular,
            centers,
            1,
            backend="native",
        )
        local_descriptor = _explicit_single_l_ctilde_terms(
            rank_two,
            local_base,
        ).sum()
        return (
            embedding_scale
            * (
                rank_one.ctildes[0] * local_radial.sum()
                + local_descriptor.real
            )
            + payload["E0"][0]
        )

    torch.testing.assert_close(
        energy,
        explicit_energy(vectors),
        rtol=0.0,
        atol=2.0e-14,
    )
    finite_difference = torch.empty_like(vectors)
    step = 1.0e-6
    for edge in range(vectors.shape[0]):
        for axis in range(3):
            plus = vectors.clone()
            minus = vectors.clone()
            plus[edge, axis] += step
            minus[edge, axis] -= step
            finite_difference[edge, axis] = (
                explicit_energy(plus) - explicit_energy(minus)
            ) / (2.0 * step)
    torch.testing.assert_close(
        edge_gradient,
        finite_difference,
        rtol=3.0e-6,
        atol=3.0e-7,
    )
    center_force = edge_gradient.sum(dim=0)
    neighbor_forces = -edge_gradient
    torch.testing.assert_close(
        center_force + neighbor_forces.sum(dim=0),
        torch.zeros_like(center_force),
        rtol=0.0,
        atol=2.0e-13,
    )
