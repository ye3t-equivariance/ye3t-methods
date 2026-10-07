"""Fit a configured density model and inspect one compiler-selected column."""

from pathlib import Path

from ase.io import read
from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
feature_index = 0
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "inspect_features", "status": "stable",
        "training_structures": str(fixtures / "cu2_training.extxyz"),
        "evaluation_structure": str(fixtures / "cu2_structure.extxyz"),
    },
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
            "l_max_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "single_factors": {
            "species": ["Cu"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 3.5,
                       "cutoff_width_A": 0.01, "lambda": 0.79},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "density"},
        "catalogue": {
            "ranks": [1, 2, 3, 4],
            "nmax_per_rank": {1: 2, 2: 2, 3: 2, 4: 2},
            "lmax_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
            "source_block_partitions_by_rank": {
                1: [[1]], 2: [[2]], 3: [[3]], 4: [[4]],
            },
        },
    },
    "runtime": {"evaluator": "torch", "neighbors": "ase",
                "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"},
    "model": {
        "kind": "linear",
        "fit": {"solver": "ridge", "alpha": 1e-12,
                "weights": {"energy": 1.0, "forces": 1.0}},
        "reference_energy": {"per_species_E0_eV": {"Cu": 0.0}, "fit_E0": True},
    },
    "targets": {"energy": "energy", "forces": "forces", "stress": None},
    "validation": {"checks": ["round_trip"]},
}

representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation,
                          runtime=config["runtime"])
structures = read(config["metadata"]["training_structures"], index=":")
model = LinearModel(basis).fit(structures, config=config)
atoms = read(config["metadata"]["evaluation_structure"])
print(representation)
print(basis)
print("descriptor_shape", basis.create(atoms).shape)
print(basis.labels[feature_index])
print(basis.describe(feature_index))
print(basis.describe(feature_index, format="latex"))
print(model)
print(model.describe(feature_index))
