"""Evaluate one selected rank-eight ordered Phi star on periodic fcc Ni."""

import numpy as np
from ase.build import bulk
from ase.neighborlist import neighbor_list

from ye3t import YE3TRepresentation
from ye3t_methods import Basis


config = {
    "metadata": {"schema": "ye3t_config_v1", "name": "ni_phi_rank8", "status": "experimental"},
    "representation": {
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
    },
    "basis": {
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
    },
    "runtime": {
        "evaluator": "torch", "neighbors": "ase", "cache": {"mode": "off"},
        "dtype": "float64", "device": "cpu",
    },
    "model": {"kind": "descriptor_only"},
    "targets": {},
    "validation": {
        "checks": ["compiler_certificate", "slot_intertwiner", "rotation_covariance"],
    },
}

atoms = bulk("Ni", "fcc", a=3.508, cubic=True).repeat((2, 2, 2))
center = 0
neighbors, other_atoms, shifts = neighbor_list(
    "ijS", atoms, config["basis"]["single_factors"]["radial"]["cutoff_A"],
)
ordered_occurrences = [
    (int(index), tuple(int(value) for value in shift))
    for root, index, shift in zip(neighbors, other_atoms, shifts)
    if int(root) == center
]
ordered_occurrences.sort(key=lambda row: (
    float(np.linalg.norm(atoms.positions[row[0]] + np.asarray(row[1]) @ atoms.cell
                         - atoms.positions[center])), row,
))
ordered_occurrences = ordered_occurrences[:8]
if len(ordered_occurrences) != 8 or len(set(ordered_occurrences)) != 8:
    raise RuntimeError("The Ni structure does not supply eight distinct neighbor images.")

representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
cluster = basis.create_cluster(atoms, center, ordered_occurrences)
counts = basis.catalogue.counts()["by_component"]["main"]
assert cluster.shape == (1, 14, 5)
assert np.isfinite(cluster).all()
assert counts["full_sector_multiplicity"] == 18
assert counts["selected_full_alpha"] == 12
print("ordered_occurrences", ordered_occurrences)
print("selected_full_alpha", counts["selected_full_alpha"])
print("label", basis.labels[0])
print("cluster_axes", cluster.shape)
print("cluster_norm", float(np.linalg.norm(cluster)))
