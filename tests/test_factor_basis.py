"""Source-neutral compiled Cauchy factors through the methods Basis interface."""

import numpy as np
import pytest
import torch

from ye3t import YE3TRepresentation, couplings
from ye3t_methods import Basis
from ye3t.representations.projectors import (
    adjacent_transposition_representation_matrix_numeric,
)


def _config(parent, L):
    channels = [
        {"factor_type": f"u_{index}", "l": 1,
         "source_family_id": "supplied_multiplets"}
        for index in range(3)
    ]
    return {
        "metadata": {"schema": "ye3t_config_v1", "name": "factor_basis_test"},
        "basis": {"tensor_product": {"kind": "ordered_role_cauchy",
                                      "role_kappa_policy": "all_valid"},
                  "channels": channels, "block_sizes": [1, 1, 1],
                  "role_dimension": 1},
        "representation": {
            "group": "O3", "ranks": [3],
            "parent": {"young_lambda": parent, "L": L, "parity": "odd"},
            "factorization": "cauchy", "subspace": "full",
            "uncoupled_factor_inputs": {
                "eta_count_per_rank": {3: 3}, "l_max_per_rank": {3: 1}},
            "intermediates": {"young_kappa": "all_valid",
                              "block_rotation": {"policy": "all_valid"}},
        },
        "runtime": {"device": "cpu", "dtype": "complex128",
                    "magnetic_basis": "complex_condon_shortley"},
        "model": {}, "targets": {},
        "validation": {"checks": ["count", "permutation", "rotation", "parity"]},
    }


def test_a_s_compatibility_report_counts_resolved_multiplicity_copies():
    from ye3t_methods.atomistic.ace.descriptors import (
        _a_s_matrix_unit_global_coupler_compatibility_report,
    )

    report = _a_s_matrix_unit_global_coupler_compatibility_report(
        {"slot_specht_partition": (2, 1), "power": 3,
         "target_L_R": 1, "l_in": 1},
        np.zeros((1, 3)),
    )
    candidate = report["slot_resolved_product_slot_candidate"]
    assert candidate["available"]
    assert candidate["alpha_label_count"] > candidate["global_label_group_count"]


def test_coefficient_descriptor_views_report_resolved_alpha_copies():
    from ye3t import CompileGlobalYE3TCouplers, YE3TRotationTarget, YE3TSpec
    from ye3t_methods.atomistic.ace.descriptors import YE3TDescriptorSet

    common = dict(
        content=(1, 2, 3), target_rotation=YE3TRotationTarget(L_R=0),
        carrier="external_tensor", coefficient_backend="global_coupler",
        runtime_status="planned_not_public", metadata={"input_Ls": (0, 0, 0)},
    )
    concrete = YE3TSpec(target_permutation="young:2,1", **common)
    coupler = CompileGlobalYE3TCouplers(concrete)
    assert len(coupler.alpha_labels()) > len(coupler.labels)
    values = torch.ones((1, coupler.sparse_coefficient_tables[0]["shape"][1]), dtype=torch.float64)

    descriptor = YE3TDescriptorSet(
        None, None, (), {}, 0.0, None, metadata={"ye3t_spec": concrete.to_dict()}
    )
    single = descriptor.evaluate_ye3t_coefficient_descriptor_view(values)
    record = single.sector_slices[0]
    assert record["alpha_label_count"] == len(coupler.alpha_labels())
    assert len(record["alpha_labels"]) == len(coupler.alpha_labels())
    assert record["global_label_group_count"] == len(coupler.labels)

    family_spec = YE3TSpec(target_permutation="full_irrep_decomposition", **common)
    descriptor.metadata = {"ye3t_spec": family_spec.to_dict()}
    family = descriptor.compile_global_coupler_family()
    sector_values = {
        partition: torch.ones((1, sector.sparse_coefficient_tables[0]["shape"][1]),
                              dtype=torch.float64)
        for partition, sector in zip(family.target_partitions, family.couplers, strict=True)
    }
    family_view = descriptor.evaluate_ye3t_coefficient_descriptor_view(sector_values)
    mixed = next(row for row in family_view.sector_slices
                 if row["target_partition"] == (2, 1))
    assert mixed["alpha_label_count"] == len(coupler.alpha_labels())
    assert len(mixed["alpha_labels"]) == len(coupler.alpha_labels())
    assert mixed["global_label_group_count"] == len(coupler.labels)


@pytest.mark.parametrize("parent,L", [("(3)", 1), ("(1,1,1)", 0), ("(2,1)", 1)])
def test_supplied_factor_basis_counts_actions_and_compiled_plan(parent, L):
    cfg = _config(parent, L)
    representation = YE3TRepresentation.from_config(cfg["representation"])
    basis = Basis.from_config(cfg["basis"], representation=representation,
                              runtime=cfg["runtime"])
    report = couplings.count(basis.coupling_request())
    assert len(basis.labels) == report["multiplet_count"] > 0
    assert basis.catalogue.counts()["multiplet_count"] == report["multiplet_count"]
    assert basis.resolution.capability_report["factor_evaluation_available"]
    assert not basis.resolution.capability_report["basis_create_available"]
    assert report["component_count"] == len(basis.labels) * report["tableau_count"] * (2 * L + 1)
    factors = torch.tensor([[[[0.2, 0.3, -0.5]], [[0.7, -0.1, 0.4]],
                             [[0.1, 0.9, -0.2]]]], dtype=torch.complex128)
    original = basis.create_factors(factors)
    assert original.shape == (1, len(basis.labels), report["tableau_count"], 2 * L + 1)
    assert torch.isfinite(original).all()
    action = torch.as_tensor(adjacent_transposition_representation_matrix_numeric(
        representation.parent_partition(3), 0), dtype=original.dtype)
    moved = basis.create_factors(factors[:, [1, 0, 2]], factor_types=[1, 0, 2])
    torch.testing.assert_close(moved, torch.einsum("tu,...aum->...atm", action, original),
                               atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(basis.create_factors(-factors), -original,
                               atol=1e-11, rtol=1e-11)
    if L == 1:
        angle = 0.31
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        spherical = np.array([[1 / np.sqrt(2), -1j / np.sqrt(2), 0],
                              [0, 0, 1],
                              [-1 / np.sqrt(2), -1j / np.sqrt(2), 0]])
        spin_one = torch.as_tensor(spherical @ rotation @ np.linalg.inv(spherical),
                                   dtype=original.dtype)
        rotated = basis.create_factors(factors @ spin_one.T)
        torch.testing.assert_close(rotated, original @ spin_one.T,
                                   atol=1e-11, rtol=1e-11)
    compiled = couplings.compile(couplings.plan(report))
    supplied = Basis.from_config(cfg["basis"], representation=representation,
                                 runtime=cfg["runtime"], compiled_plan=compiled)
    torch.testing.assert_close(supplied.create_factors(factors), original,
                               atol=1e-11, rtol=1e-11)
    execution = couplings.lower_ordered_role_cauchy_execution_plan(compiled)
    planned = Basis.from_config(cfg["basis"], representation=representation,
                                runtime=cfg["runtime"], compiled_plan=execution)
    torch.testing.assert_close(planned.create_factors(factors), original,
                               atol=1e-11, rtol=1e-11)


def test_supplied_factor_basis_rejects_wrong_compiler_source_identity():
    cfg = _config("(2,1)", 1)
    representation = YE3TRepresentation.from_config(cfg["representation"])
    basis = Basis.from_config(cfg["basis"], representation=representation,
                              runtime=cfg["runtime"])
    compiled = couplings.compile(couplings.plan(couplings.count(basis.coupling_request())))
    cfg["basis"]["channels"][0]["source_family_id"] = "different_source"
    with pytest.raises(ValueError, match="does not match"):
        Basis.from_config(cfg["basis"], representation=representation,
                          runtime=cfg["runtime"], compiled_plan=compiled)


def test_supplied_factor_basis_mixed_angular_channels_have_even_parity():
    cfg = _config("(2,1)", 1)
    cfg["representation"]["parent"]["parity"] = "even"
    cfg["basis"]["channels"][2]["l"] = 0
    representation = YE3TRepresentation.from_config(cfg["representation"])
    basis = Basis.from_config(cfg["basis"], representation=representation,
                              runtime=cfg["runtime"])
    report = couplings.count(basis.coupling_request())
    vectors = torch.tensor([[[0.2, 0.3, -0.5]]], dtype=torch.complex128)
    scalar = torch.tensor([[[0.7]]], dtype=torch.complex128)
    factors = (vectors, vectors * 0.8, scalar)
    values = basis.create_factors(factors)
    assert values.shape == (1, report["multiplet_count"],
                            report["tableau_count"], 3)
    assert report["component_count"] == values[0].numel()
    torch.testing.assert_close(basis.create_factors((-factors[0], -factors[1], factors[2])),
                               values, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(basis.create_factors((-factors[0], factors[1], factors[2])),
                               -values, atol=1e-11, rtol=1e-11)
    compiled = couplings.compile(couplings.plan(report))
    execution = couplings.lower_ordered_role_cauchy_execution_plan(compiled)
    assert execution.carrier_layouts[0].key.parity == 1
    cfg["representation"]["parent"]["parity"] = "odd"
    wrong = YE3TRepresentation.from_config(cfg["representation"])
    with pytest.raises(ValueError, match="parity"):
        Basis.from_config(cfg["basis"], representation=wrong,
                          runtime=cfg["runtime"])


def test_supplied_axial_factor_parity_is_part_of_compiled_identity():
    cfg = _config("(2,1)", 1)
    cfg["representation"]["parent"]["parity"] = "even"
    cfg["basis"]["channels"][0]["parity"] = 1
    representation = YE3TRepresentation.from_config(cfg["representation"])
    basis = Basis.from_config(cfg["basis"], representation=representation,
                              runtime=cfg["runtime"])
    report = couplings.count(basis.coupling_request())
    compiled = couplings.compile(couplings.plan(report))
    execution = couplings.lower_ordered_role_cauchy_execution_plan(compiled)
    assert tuple(key.parity for key in execution.instructions[0].input_carriers) == (1, -1, -1)
    assert execution.carrier_layouts[0].key.parity == 1
    factors = torch.tensor([[[[0.2, 0.3, -0.5]], [[0.7, -0.1, 0.4]],
                             [[0.1, 0.9, -0.2]]]], dtype=torch.complex128)
    values = basis.create_factors(factors)
    inverted = factors.clone()
    inverted[:, 1:] *= -1
    torch.testing.assert_close(basis.create_factors(inverted), values,
                               atol=1e-11, rtol=1e-11)
