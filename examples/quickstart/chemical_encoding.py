"""Evaluate scalar ASE descriptors with a fixed two-channel species embedding.

Edit the species list, embedding matrix, radial source, and catalogue below.
Changing ``chemical.kind`` to ``explicit`` restores one-hot neighbor channels.
"""

import numpy as np
from ase import Atoms
from ye3t import YE3TRepresentation
from ye3t_methods import Basis


config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "chemical_encoding",
        "status": "stable",
        "system": {
            "symbols": "NiCuAl",
            "positions_A": [[0.0, 0.0, 0.0], [1.4, 0.1, 0.2], [0.3, 1.7, 0.4]],
        },
    },
    "representation": {
        "group": "O3", "ranks": [1, 2],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 1},
            "l_max_per_rank": {1: 0, 2: 0},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "single_factors": {
            "species": ["Ni", "Cu", "Al"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 4.5,
                       "cutoff_width_A": 0.01, "lambda": 0.79},
            "chemical": {
                "kind": "fixed_embedding",
                "species_order": ["Ni", "Cu", "Al"],
                "matrix": [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]],
            },
        },
        "tensor_product": {"kind": "density"},
        "catalogue": {
            "ranks": [1, 2],
            "nmax_per_rank": {1: 1, 2: 1},
            "lmax_per_rank": {1: 0, 2: 0},
            "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]]},
        },
    },
    "runtime": {"evaluator": "torch", "neighbors": "ase",
                "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"},
    "model": {}, "targets": {},
    "validation": {"checks": ["rotation", "atom_order"]},
}

system = config["metadata"]["system"]
atoms = Atoms(system["symbols"], positions=system["positions_A"])
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
descriptors = basis.create(atoms)

rotated = atoms.copy()
rotated.rotate(37.0, "z", center=(0, 0, 0))
np.testing.assert_allclose(basis.create(rotated), descriptors, rtol=1e-9, atol=1e-9)
order = [2, 0, 1]
np.testing.assert_allclose(basis.create(atoms[order]), descriptors[order],
                           rtol=1e-9, atol=1e-9)

print("embedding species order", config["basis"]["single_factors"]["chemical"]["species_order"])
print("embedding shape", np.shape(config["basis"]["single_factors"]["chemical"]["matrix"]))
print("descriptor shape", descriptors.shape)
print("compiler count per center", basis.catalogue.counts()["exact_total_per_center"])
print("first descriptor", basis.labels[0])
print("rotation and atom-order checks passed")
