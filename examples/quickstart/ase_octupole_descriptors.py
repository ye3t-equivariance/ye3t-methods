"""Evaluate seven-component L=3 YE3T descriptors on an ASE structure.

Edit the target L, parity, angular cap, source, species, and cell for another
property. This CPU example demonstrates the general full-M interface and
does not represent a trained octupole model.
"""

import numpy as np
from ase.build import bulk
from ye3t import YE3TRepresentation
from ye3t_methods import Basis


config = {
    "metadata": {"schema": "ye3t_config_v1", "name": "ni_l3_descriptors",
                 "status": "experimental", "element": "Ni", "crystal": "fcc",
                 "lattice_parameter_A": 3.52},
    "representation": {
        "group": "O3", "ranks": [1],
        "parent": {"young_lambda": "(N)", "L": 3, "parity": "odd"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 2}, "l_max_per_rank": {1: 3},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
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
            "ranks": [1], "nmax_per_rank": {1: 2},
            "lmax_per_rank": {1: 3},
            "source_block_partitions_by_rank": {1: [[1]]},
        },
    },
    "runtime": {"evaluator": "torch", "neighbors": "ase",
                "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"},
    "model": {}, "targets": {}, "validation": {},
}

atoms = bulk(config["metadata"]["element"], config["metadata"]["crystal"],
             a=config["metadata"]["lattice_parameter_A"], cubic=True)
atoms.positions[1] += [0.12, -0.08, 0.05]
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation,
                          runtime=config["runtime"])
descriptors = basis.create(atoms)
assert descriptors.shape == (len(atoms), len(basis.labels), 7)
assert np.linalg.norm(descriptors) > 0
print("descriptor shape", descriptors.shape)
print("first descriptor", basis.labels[0])
print("compiler count", basis.catalogue.counts()["exact_total_per_center"])
