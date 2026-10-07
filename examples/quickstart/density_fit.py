"""Fit a paper-informed scalar Ni ACE model from real labeled ASE structures.

This short training subset demonstrates the interface. It uses the paper's
radial source and radial counts through rank four, with smaller angular caps;
it does not reproduce the selected 60/127/149-feature paper models or RMSE.
"""

import json
from pathlib import Path

import numpy as np
from ase.io import read
from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel


dataset = Path(__file__).resolve().parents[1] / "data" / "mlearn" / "Ni"
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "ni_density_fit", "status": "stable",
        "training_structures": str(dataset / "ni_all.xyz"),
        "evaluation_structure": str(dataset / "ni_all.xyz"),
        "output_path": "../ye3t-workflows/quickstart_linear/ni_density_fit.pt",
        "system": {"split_file": str(dataset / "moment_star_split.json"),
                   "training_count": 12},
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
            "species": ["Ni"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 4.638049165633364,
                       "cutoff_width_A": 0.01, "lambda": 0.7928781217554153},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "density"},
        "catalogue": {
            "ranks": [1, 2, 3, 4],
            "nmax_per_rank": {1: 4, 2: 3, 3: 3, 4: 3},
            "lmax_per_rank": {1: 0, 2: 1, 3: 1, 4: 1},
            "source_block_partitions_by_rank": {
                1: [[1]], 2: [[2]], 3: [[3]], 4: [[4]],
            },
        },
    },
    "runtime": {"evaluator": "torch", "neighbors": "ase",
                "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"},
    "model": {
        "kind": "linear",
        "fit": {"solver": "ridge", "alpha": 1e-8,
                "weights": {"energy": 1.0, "forces": 1.0}},
        "reference_energy": {"per_species_E0_eV": {"Ni": 0.0}, "fit_E0": True},
    },
    "targets": {"energy": "energy", "forces": "forces", "stress": None},
    "validation": {"checks": ["round_trip"]},
}
frames = read(config["metadata"]["training_structures"], index=":")
split = json.loads(Path(config["metadata"]["system"]["split_file"]).read_text(
    encoding="utf-8"))
train_indices = split["indices"]["train"]
selected = np.linspace(0, len(train_indices) - 1,
                       config["metadata"]["system"]["training_count"], dtype=int)
structures = [frames[train_indices[index]] for index in selected]
assert all(atoms.info["source_split"] == "training" for atoms in structures)
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
model = LinearModel(basis).fit(structures, config=config)
output_path = Path(config["metadata"]["output_path"])
output_path.parent.mkdir(parents=True, exist_ok=True)
artifact = model.write(output_path)
restored = LinearModel.read(artifact)
atoms = frames[split["indices"]["test"][0]].copy()
reference_energy = frames[split["indices"]["test"][0]].get_potential_energy()
atoms.calc = restored.ase_calculator(
    evaluator=config["runtime"]["evaluator"],
    neighbors=config["runtime"]["neighbors"],
    force_method="autograd",
)
print(representation)
print("features", len(basis.labels))
print("training_structures", len(structures))
print("saved_model", artifact)
print("heldout_energy_reference_eV", reference_energy)
print("heldout_energy_predicted_eV", atoms.get_potential_energy())
print("heldout_maximum_force_eV_per_A", abs(atoms.get_forces()).max())
