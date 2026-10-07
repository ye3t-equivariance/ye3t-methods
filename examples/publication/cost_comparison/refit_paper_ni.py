"""Refit the selected 127-column Ni paper model from labeled ASE frames.

This full example uses all 263 published training/validation structures and the
frozen selected basis and hyperparameters. It is a long CPU run. The saved
refit is a Torch ASE model; LAMMPS AUTO needs a newly validated native plan.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read

from ye3t_methods import LinearModel


example = Path(__file__).resolve().parent
dataset = example.parents[1] / "data" / "mlearn" / "Ni"
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "refit_paper_ni_127",
        "status": "long_publication_workflow",
        "dataset": str(dataset / "ni_all.xyz"),
        "dataset_sha256": "f713984c2dc1b1988ed71d7dabb5d8f77d7d611b4a96deb4da3d61443e92f93f",
        "split": str(dataset / "moment_star_split.json"),
        "output_model": "../ye3t-workflows/MLIP/ni_paper_refit/ni_127_refit.ye3t",
        "output_summary": "../ye3t-workflows/MLIP/ni_paper_refit/ni_127_refit.json",
    },
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "torch", "neighbors": "ase", "device": "cpu"},
    "model": {
        "kind": "linear", "feature_count": 127,
        "source_archive": str(example / "portable_models" / "Ni_ye3t_tagged_127.ye3t"),
        "source_archive_sha256":
        "a57647406108a71273794e7786252147c93830d8954d8c44d9d7e813cb6502a2",
        "fit": {
            "solver": "paper_scaled_ridge", "alpha": 1e-10,
            "weights": {"energy": 6.113905002213526, "forces": 1.0},
            "group_weights": {
                "AIMD-NVT": 1.226657592228842,
                "Elastic": 1.0298265385887098,
                "Surface": 0.7756267270732168,
                "Vacancy": 1.0206099630686312,
            },
            "tagged_penalty": 0.2708482089732946,
        },
    },
    "targets": {"energy": "energy", "forces": "forces",
                "group": "config_type", "stress": None},
    "validation": {"checks": ["force_fd", "round_trip"],
                   "expected_train_count": 263, "expected_test_count": 31},
}

source = Path(config["model"]["source_archive"])
assert hashlib.sha256(source.read_bytes()).hexdigest() == config["model"][
    "source_archive_sha256"]
data_path = Path(config["metadata"]["dataset"])
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == config["metadata"][
    "dataset_sha256"]
split = json.loads(Path(config["metadata"]["split"]).read_text(encoding="utf-8"))
frames = read(str(data_path), index=":")
training_indices = sorted(split["indices"]["train"] + split["indices"]["validation"])
test_indices = split["indices"]["test"]
assert len(training_indices) == config["validation"]["expected_train_count"]
assert len(test_indices) == config["validation"]["expected_test_count"]
assert not set(training_indices) & set(test_indices)
training = [frames[index] for index in training_indices]
assert all(atoms.info["source_split"] == "training" for atoms in training)

model = LinearModel.read(source)
basis = model.basis
assert len(basis.labels) == config["model"]["feature_count"]
model.fit(training, config=config)
saved = model.write(config["metadata"]["output_model"])
restored = LinearModel.read(saved)
assert [label.identity for label in restored.labels] == [
    label.identity for label in basis.labels]

energy_errors = []
force_squared_error = 0.0
force_components = 0
for index in test_indices:
    reference = frames[index]
    atoms = reference.copy()
    atoms.calc = restored.ase_calculator(evaluator="torch", neighbors="ase")
    energy_errors.append((atoms.get_potential_energy() -
                          reference.get_potential_energy()) / len(atoms))
    difference = atoms.get_forces() - reference.get_forces()
    force_squared_error += float(np.square(difference).sum())
    force_components += difference.size

summary = {
    "schema": "ye3t_methods_ni_paper_refit_summary_v1",
    "model": str(saved), "source_archive_sha256": config["model"]["source_archive_sha256"],
    "training_structures": len(training), "heldout_structures": len(test_indices),
    "selected_feature_count": len(restored.labels),
    "heldout_energy_rmse_eV_per_atom": float(np.sqrt(np.mean(np.square(energy_errors)))),
    "heldout_force_rmse_eV_per_A": float(np.sqrt(force_squared_error / force_components)),
}
summary_path = Path(config["metadata"]["output_summary"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
print("selected_features", len(restored.labels))
print("training_structures", len(training))
print("heldout_energy_rmse_eV_per_atom", summary["heldout_energy_rmse_eV_per_atom"])
print("heldout_force_rmse_eV_per_A", summary["heldout_force_rmse_eV_per_A"])
print("saved_model", saved)
print("saved_summary", summary_path)
