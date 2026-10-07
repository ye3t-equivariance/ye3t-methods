"""Retained fixed Young descriptor evaluation, without the old network tests."""

import torch

from ye3t_methods.atomistic.linear_young_character import YE3TSavedDescriptorSetFeatureMap
from ye3t_methods.atomistic._record import record_replace
from ye3t_methods.atomistic.equivariant_calc.site_basis_v2 import SiteBasisV2


def test_saved_nontrivial_young_scalar_is_a_fixed_invariant_column():
    sign_source = {
        "L": 1,
        "copy_index": 0,
        "input_l": [1, 1],
        "input_n": [1, 1],
        "input_rank": 2,
        "permutation_irrep": "S_2:[1,1]",
        "permutation_partitions": [[1, 1]],
        "permutation_representation": "nontrivial",
        "density_n": [1, 2],
    }
    descriptor_set = {
        "name": "saved_copy_aware_sign_invariant",
        "descriptors": [{
            "descriptor_id": "rank2_nontrivial_invariant",
            "descriptor_rank": 4,
            "source": "same_group_nontrivial_young_invariant",
            "label": {"left": dict(sign_source), "right": dict(sign_source)},
        }],
    }
    feature_map = YE3TSavedDescriptorSetFeatureMap(
        descriptor_set, num_types=2, radial_count=2, cutoff=5.0,
        lmax=1, dtype=torch.float64,
    )
    positions = torch.tensor([
        [0.73, 0.21, 0.49], [-0.62, 0.37, 0.58],
        [0.51, -0.83, 0.31], [0.94, 0.46, -0.72],
        [-0.88, -0.24, 0.67], [-0.35, 0.79, -0.43],
        [0.27, -0.56, -0.91], [-0.76, -0.68, -0.29],
    ], dtype=torch.float64)
    types = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
    values = feature_map(types, positions)
    x_ij, edge_index, atom_types = feature_map._edge_inputs(types, positions)
    complex_basis = SiteBasisV2(record_replace(feature_map.site_basis_config, spherical_backend="complex"))
    _, complex_base = complex_basis.compute_atomic_base(
        x_ij=x_ij, edge_index=edge_index, atom_types=atom_types,
        channels=feature_map.channels,
    )
    density = feature_map._density(types, positions)
    for l_value, group in feature_map.channel_groups.items():
        width = 2 * int(l_value) + 1
        expected = complex_base[:, group["start"]:group["stop"]].reshape(
            len(positions), len(group["n_values"]), feature_map.num_types, feature_map.num_types, width,
        )
        for n_index, n_value in enumerate(group["n_values"]):
            torch.testing.assert_close(
                density[(int(n_value), int(l_value))], expected[:, n_index].sum(dim=(1, 2)),
                atol=1.0e-12, rtol=1.0e-12,
            )
    axis = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    axis = axis / torch.linalg.norm(axis)
    cross = torch.tensor([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=torch.float64)
    angle = torch.tensor(0.73, dtype=torch.float64)
    rotation = torch.eye(3, dtype=torch.float64) + torch.sin(angle) * cross + (1.0 - torch.cos(angle)) * cross @ cross
    rotated = feature_map(types, positions @ rotation.T)
    permutation = torch.tensor([6, 2, 7, 0, 5, 1, 4, 3])
    reordered = feature_map(types[permutation], positions[permutation])

    assert values.shape == (1, 1)
    assert torch.isfinite(values).all()
    assert values.abs().max().item() > 1.0e-10
    torch.testing.assert_close(rotated, values, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(reordered, values, atol=1.0e-10, rtol=1.0e-10)
    report = feature_map.report()
    assert report["nontrivial_descriptor_count"] == 1
    assert report["uses_runtime_gram_matrix"] is False
    assert report["density_angular_convention"] == "site_signed_m_reversed_to_ye3t_tesseral_v1"
    assert report["uses_scalar_proxy"] is False
