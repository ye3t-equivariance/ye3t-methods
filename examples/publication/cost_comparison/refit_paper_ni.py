"""Refit the selected 127-column Ni paper model from labeled ASE frames.

This full example uses all 263 published training/validation structures and the
frozen selected basis and hyperparameters. It is a long CPU run. The saved
refit supports Torch/native ASE and LAMMPS export with regenerated readout plans.
"""

import json
import os
from pathlib import Path


from ye3t_methods import LinearModel
from ye3t_methods.studies.ni_linear import load_refit_structures


example = Path(__file__).resolve().parent
dataset = example.parents[1] / "data" / "mlearn" / "Ni"
workflow = Path(os.environ.get("YE3T_WORKFLOW_ROOT", "~/ye3t-workflows")).expanduser()
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "refit_paper_ni_127",
        "status": "long_publication_workflow",
        "dataset": str(dataset / "ni_all.xyz"),
        "dataset_sha256": "f713984c2dc1b1988ed71d7dabb5d8f77d7d611b4a96deb4da3d61443e92f93f",
        "split": str(dataset / "moment_star_split.json"),
        "output_model": str(workflow / 'MLIP/ni_paper_refit/ni_127_refit.ye3t'),
        "output_summary": str(workflow / 'MLIP/ni_paper_refit/ni_127_refit.json'),
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

training, validation = load_refit_structures(config)
model = LinearModel.read(config["model"]["source_archive"])
print("selected basis", model.basis)
model.fit(training, config=config)
saved = model.write(config["metadata"]["output_model"])
restored = LinearModel.read(saved)
results = restored.score(validation, config=config)
results["training_structures"] = len(training)
results["selected_feature_count"] = len(restored.labels)
results["source_archive_sha256"] = config["model"]["source_archive_sha256"]
output = Path(config["metadata"]["output_summary"])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
print("validation results", results)
print("saved model", saved)
