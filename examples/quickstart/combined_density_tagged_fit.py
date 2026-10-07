"""Fit one scalar model from ordinary density and tagged physical sources.

The energies below are manufactured workflow labels, not a Ni potential.
"""

from pathlib import Path

import numpy as np
from ase.build import bulk
from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel


output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "ni_combined_scalar",
        "status": "stable",
        "system": {
            "element": "Ni", "crystal": "fcc", "lattice_parameter_A": 3.52,
            "repeat": [2, 2, 2], "training_shifts_A": [0.0, 0.12, -0.08, 0.22],
        },
        "output_path": str(output_root / "ni_combined_scalar.ye3t"),
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
            "species": ["Ni"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 3.5,
                       "cutoff_width_A": 0.01, "lambda": 0.79},
            "chemical": {"kind": "explicit"},
        },
        "components": {
            "ordinary": {
                "tensor_product": {"kind": "density"},
                "catalogue": {
                    "ranks": [1, 2, 3, 4],
                    "nmax_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
                    "lmax_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
                    "source_block_partitions_by_rank": {
                        1: [[1]], 2: [[2]], 3: [[3]], 4: [[4]],
                    },
                },
            },
            "tagged": {
                "single_factors": {
                    "species": ["Ni"],
                    "radial": {"family": "shifted_jacobi", "cutoff_A": 3.1},
                    "chemical": {"kind": "explicit"},
                },
                "tensor_product": {"kind": "tagged",
                                   "tag_counts_per_rank": {4: [0, 2]}},
                "catalogue": {
                    "ranks": [4], "nmax_per_rank": {4: 1},
                    "lmax_per_rank": {4: 1},
                    "source_block_partitions_by_rank": {4: [[4]]},
                    "angular_patterns_by_rank": {4: [[1, 1, 1, 1]]},
                },
            },
        },
    },
    "runtime": {"evaluator": "auto", "neighbors": "auto",
                "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"},
    "model": {
        "kind": "linear",
        "fit": {"solver": "ridge", "alpha": 1e-8,
                "weights": {"energy": 1.0, "forces": 0.0}},
        "reference_energy": {"per_species_E0_eV": {"Ni": 0.0}, "fit_E0": True},
    },
    "targets": {"energy": "energy", "forces": None, "stress": None},
    "validation": {"checks": ["round_trip"]},
}

representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation,
                          runtime=config["runtime"])
system = config["metadata"]["system"]
seed = bulk(system["element"], system["crystal"],
            a=system["lattice_parameter_A"], cubic=True).repeat(system["repeat"])
oracle_weights = np.linspace(0.1, 0.2, len(basis.labels))
structures = []
for shift in system["training_shifts_A"]:
    atoms = seed.copy()
    atoms.positions[0, 0] += shift
    atoms.info["energy"] = float(basis.create(atoms).sum(axis=0) @ oracle_weights
                                 + 0.07 * len(atoms))
    structures.append(atoms)

model = LinearModel(basis).fit(structures, config=config)
output_path = Path(config["metadata"]["output_path"])
output_path.parent.mkdir(parents=True, exist_ok=True)
artifact = model.write(output_path)
restored = LinearModel.read(artifact)
probe = seed.copy()
probe.positions[0, 0] += 0.06
rows = restored.basis.create(probe)
probe.calc = restored.ase_calculator(evaluator="torch")
assert rows.shape == (len(probe), len(restored.labels))
assert np.isfinite(probe.get_potential_energy())
assert np.isfinite(probe.get_forces()).all()
print(representation)
print("source_families", [part["radial_source"]["family"]
                          for part in basis.resolved["components"]])
print("descriptor_shape", rows.shape)
print("saved_model", artifact)
print("energy_eV", probe.get_potential_energy())
print("maximum_force_eV_per_A", np.max(np.abs(probe.get_forces())))
