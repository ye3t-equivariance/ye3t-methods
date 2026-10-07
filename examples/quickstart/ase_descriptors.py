"""Create compiler-backed ordinary YE3T descriptor rows for ASE atoms.

Edit the species, radial source, cutoff, ranks, and angular/radial caps below
for another system. The source and ranks follow the Ni paper through rank
eight, with an enlarged rank-two radial cap. Selected source partitions and
angular caps give 96 independent scalar columns without loading a model.
Use paper_ni_portable_ase.py to inspect the exact saved Ni-127 basis.
"""

from ase.build import bulk
from ye3t import YE3TRepresentation
from ye3t_methods import Basis


config = {
    "metadata": {"schema": "ye3t_config_v1", "name": "ni_ase_descriptors",
                 "status": "stable", "element": "Ni", "crystal": "fcc",
                 "lattice_parameter_A": 3.52},
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4, 6, 8],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 1, 3: 1, 4: 1, 6: 1, 8: 1},
            "l_max_per_rank": {1: 0, 2: 2, 3: 2, 4: 2, 6: 1, 8: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "single_factors": {
            "species": ["Ni"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 4.638049165633364,
                       "cutoff_width_A": 0.01, "lambda": 0.7928781217554153},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "density"},
        "catalogue": {
            "ranks": [1, 2, 3, 4, 6, 8],
            "nmax_per_rank": {1: 4, 2: 4, 3: 3, 4: 3, 6: 2, 8: 2},
            "lmax_per_rank": {1: 0, 2: 2, 3: 2, 4: 2, 6: 1, 8: 1},
            "source_block_partitions_by_rank": {
                1: [[1]], 2: [[2], [1, 1]], 3: [[3], [2, 1]],
                4: [[4]], 6: [[6]], 8: [[8]],
            },
        },
    },
    "runtime": {"evaluator": "torch", "neighbors": "ase",
                "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"},
    "model": {}, "targets": {}, "validation": {},
}

atoms = bulk(
    config["metadata"]["element"], config["metadata"]["crystal"],
    a=config["metadata"]["lattice_parameter_A"], cubic=True,
)
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
descriptors = basis.create(atoms)
print("descriptor shape", descriptors.shape)
print("parent L", basis.labels[0].as_dict()["L"])
print("first descriptor", basis.labels[0])
print("compiler count", basis.catalogue.counts()["exact_total_per_center"])
