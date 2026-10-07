import numpy as np
import pytest
import torch
from ase import Atoms

from ye3t import YE3TRepresentation
from ye3t_methods import Basis


def _config():
    representation = {
        "group": "O3", "ranks": [8],
        "parent": {"young_lambda": "(4,4)", "L": 2, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {8: 2}, "l_max_per_rank": {8: 1},
        },
        "intermediates": {
            "young_kappa": {"policy": "explicit", "by_block_size": {4: ["(4)"]}},
            "block_rotation": {
                "policy": "explicit", "Lambda_values_by_block_size": {4: [0, 2, 4]},
            },
        },
    }
    basis = {
        "single_factors": {
            "species": ["Ni"],
            "radial": {
                "family": "pace_chebexp_cos", "cutoff_A": 4.638049165633364,
                "cutoff_width_A": 0.01, "lambda": 0.7928781217554153,
            },
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {
            "kind": "explicit_phi",
            "motif": {"kind": "rooted_star", "ordered_slots": True, "leaf_edges": []},
        },
        "catalogue": {
            "ranks": [8], "nmax_per_rank": {8: 2}, "lmax_per_rank": {8: 1},
            "source_block_partitions_by_rank": {8: [[4, 4]]},
            "fixed_content": [
                {"factor": {"species": "Ni", "radial_index": 0, "l": 1}, "copies": 4},
                {"factor": {"species": "Ni", "radial_index": 1, "l": 1}, "copies": 4},
            ],
            "selection": {
                "coupling_paths": [{"young_kappa": ["(4)", "(4)"], "Lambda": [0, 2]}],
            },
        },
    }
    runtime = {
        "evaluator": "torch", "neighbors": "ase", "cache": {"mode": "off"},
        "dtype": "float64", "device": "cpu",
    }
    return representation, basis, runtime


def test_ordered_phi_star_public_count_and_physical_occurrences():
    rep_config, basis_config, runtime = _config()
    representation = YE3TRepresentation.from_config(rep_config)
    basis = Basis.from_config(basis_config, representation=representation, runtime=runtime)
    counts = basis.catalogue.counts()["by_component"]["main"]
    assert counts["full_sector_multiplicity"] == 18
    assert counts["selected_block_path_count"] == 6
    assert counts["selected_path_count"] == 1
    assert counts["selected_full_alpha"] == 12
    assert len(basis.labels) == 1
    label = basis.labels[0].as_dict()
    assert label["selected_full_alpha"] == 12
    assert label["compiler_label"] == {
        "partition": [4, 4], "L_R": 2, "multiplicity_index": 12,
    }
    assert label["alpha_binding"]["block_Ls"] == (0, 2)
    assert "Phi N=8" in basis.describe(0)
    assert "\\Phi" in basis.describe(0, format="latex")
    assert basis.resolution.to_dict()["components"][0]["output_convention"] == (
        label["output_convention"]
    )
    assert label["output_convention"]["signed_M_order"] == [-2, -1, 0, 1, 2]
    assert len(label["output_convention"]["real_to_complex_sha256"]) == 64

    positions = np.array([
        [4.0, 4.0, 4.0],
        [5.2, 4.1, 4.0], [4.1, 5.1, 4.4], [4.4, 4.1, 5.3],
        [2.9, 3.8, 4.2], [4.1, 2.8, 3.7], [3.8, 4.2, 2.9],
        [5.1, 5.0, 4.8], [3.1, 5.0, 5.0],
    ])
    atoms = Atoms("Ni9", positions=positions, cell=[8.0, 8.0, 8.0], pbc=True)
    occurrences = [(index, (0, 0, 0)) for index in range(1, 9)]
    values = basis.create_cluster(atoms, 0, occurrences)
    assert values.shape == (1, 14, 5)
    assert np.isfinite(values).all()
    assert np.linalg.norm(values) > 0
    swapped = occurrences.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    np.testing.assert_allclose(
        basis.create_cluster(atoms, 0, swapped), values, atol=1e-10, rtol=1e-10,
    )
    angle = 0.41
    spin = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    rotated = atoms.copy()
    rotated.positions = (positions - positions[0]) @ spin.T + positions[0]
    rotated.cell = np.asarray(atoms.cell) @ spin.T
    rotated_values = basis.create_cluster(rotated, 0, occurrences)
    np.testing.assert_allclose(
        np.linalg.norm(rotated_values, axis=-1), np.linalg.norm(values, axis=-1),
        atol=1e-9, rtol=1e-9,
    )
    from ye3t.core.rotation import wigner_D_numeric
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    original_complex = real_tesseral_to_complex_multiplet(
        torch.as_tensor(values), 2,
    ).numpy()
    rotated_complex = real_tesseral_to_complex_multiplet(
        torch.as_tensor(rotated_values), 2,
    ).numpy()
    np.testing.assert_allclose(
        rotated_complex, original_complex @ wigner_D_numeric(2, spin).T,
        atol=1e-10, rtol=1e-8,
    )
    inverted = atoms.copy()
    inverted.positions = 2 * positions[0] - positions
    np.testing.assert_allclose(
        basis.create_cluster(inverted, 0, occurrences), values,
        atol=1e-10, rtol=1e-9,
    )
    image_atoms = atoms.copy()
    image_atoms.set_cell([4.0, 8.0, 8.0], scale_atoms=False)
    image_occurrences = occurrences.copy()
    image_occurrences[1] = (1, (-1, 0, 0))
    image_values = basis.create_cluster(image_atoms, 0, image_occurrences)
    assert image_values.shape == values.shape
    assert not np.allclose(image_values, values)
    with pytest.raises(ValueError, match="distinct"):
        basis.create_cluster(atoms, 0, occurrences[:7] + occurrences[:1])
    with pytest.raises(ValueError, match="cluster"):
        basis.create(atoms)
