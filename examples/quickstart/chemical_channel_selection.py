"""Keep one neighbor species in a scalar ASE density basis.

The fixed embedding column below maps Li neighbors to one and Na neighbors
to zero. Both species remain valid centers. Edit the matrix to select another
physical neighbor channel or to form a fixed mixture of species.
"""

import numpy as np
from ase import Atoms
from ye3t import YE3TRepresentation
from ye3t_methods import Basis


config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "chemical_channel_selection",
        "status": "stable",
        "system": {
            "symbols": "LiNaLi",
            "positions_A": [[0.0, 0.0, 0.0], [1.3, 0.2, 0.1],
                            [0.2, 1.5, 0.3]],
        },
    },
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
            "l_max_per_rank": {1: 0, 2: 1, 3: 1, 4: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "single_factors": {
            "species": ["Li", "Na"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 4.0,
                       "cutoff_width_A": 0.01, "lambda": 0.25},
            "chemical": {
                "kind": "fixed_embedding", "species_order": ["Li", "Na"],
                "matrix": [[1.0], [0.0]],
            },
        },
        "tensor_product": {"kind": "density"},
        "catalogue": {
            "ranks": [1, 2, 3, 4],
            "nmax_per_rank": {1: 2, 2: 2, 3: 2, 4: 2},
            "lmax_per_rank": {1: 0, 2: 1, 3: 1, 4: 1},
            "source_block_partitions_by_rank": {
                1: [[1]], 2: [[2]], 3: [[3]], 4: [[4]],
            },
        },
    },
    "runtime": {"evaluator": "torch", "neighbors": "ase",
                "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"},
    "model": {}, "targets": {},
    "validation": {"checks": ["excluded_neighbor", "rotation"]},
}

system = config["metadata"]["system"]
atoms = Atoms(system["symbols"], positions=system["positions_A"])
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
rows = basis.create(atoms)

na_moved = atoms.copy()
na_moved.positions[1] += [0.2, -0.1, 0.3]
np.testing.assert_allclose(basis.create(na_moved)[[0, 2]], rows[[0, 2]],
                           rtol=1e-9, atol=1e-9)
rotated = atoms.copy()
rotated.rotate(29.0, "y", center=(0, 0, 0))
np.testing.assert_allclose(basis.create(rotated), rows, rtol=1e-9, atol=1e-9)

print("selected neighbor species", "Li")
print("descriptor shape", rows.shape)
print("compiler count per center", basis.catalogue.counts()["exact_total_per_center"])
print("first descriptor", basis.labels[0])
print("excluded-neighbor and rotation checks passed")
