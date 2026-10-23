"""Fit the paper's 60-feature tagged Ni basis on all 263 fitting frames.

The saved compiler catalogue fixes the paper's coordinate convention. Inspect
its labels and evaluate descriptors before fitting energy and force rows.
The fit streams one structure at a time. The 31 evaluation frames are separate.
Run paper_ni_60_tagged_02_run.py afterwards for ASE molecular dynamics.
"""
import json
import os
from pathlib import Path

from ye3t_methods import LinearModel
from ye3t_methods.studies.ni_linear import load_paper_basis, load_refit_structures

example = Path(__file__).resolve().parent
dataset = example.parents[1] / "data/mlearn/Ni"
workflow = Path(os.environ.get("YE3T_WORKFLOW_ROOT", "~/ye3t-workflows")).expanduser()
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "paper_ni_60_tagged_train",
        "status": "long_publication_workflow",
        "dataset": str(dataset / "ni_all.xyz"),
        "dataset_sha256": "f713984c2dc1b1988ed71d7dabb5d8f77d7d611b4a96deb4da3d61443e92f93f",
        "split": str(dataset / "moment_star_split.json"),
        "output_model": str(workflow / "MLIP/ni60_tagged/model.ye3t"),
        "output_lammps": str(workflow / "MLIP/ni60_tagged/lammps"),
        "output_summary": str(workflow / "MLIP/ni60_tagged/summary.json"),
    },
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "torch", "neighbors": "ase", "device": "cpu",
                "cache": {"mode": "auto", "directory": None}},
    "model": {
        "kind": "linear", "basis_family": "tagged", "feature_count": 60,
        "source_archive": str(example / "portable_models/Ni_ye3t_tagged_60.ye3t"),
        "source_archive_sha256": "14b8db7ec630208fb37818c6bc56fd626c08759d7583c8fe7fe427bd7f387443",
        "selection_manifest": str(example / "lammps/Ni/models/ye3t_tagged_60/model_manifest.json"),
        "fit": {
            "solver": "paper_scaled_ridge", "alpha": 1.8608330622048866e-09,
            "weights": {"energy": 4.210524750961562, "forces": 1.0},
            "group_weights": {'AIMD-NVT': 1.1837838620343406, 'Elastic': 1.5009281994764894, 'Surface': 0.48826048688110385, 'Vacancy': 1.1526994548908491},
            "tagged_penalty": 0.12933197989976236,
        },
    },
    "targets": {"energy": "energy", "forces": "forces", "group": "config_type", "stress": None},
    "validation": {"checks": ["force_fd", "round_trip"],
                   "expected_train_count": 263, "expected_test_count": 31},
}

training, validation = load_refit_structures(config)
# The archive supplies the representation, all compiler labels, and the exact
# paper selection. Change the source archive/selection together for another basis.
basis = load_paper_basis(config)
print(basis)
print("first/last labels", basis.labels[0], basis.labels[-1])
values, gradient = basis.create(training[0], derivatives="summed")
print("site descriptors", values.shape)
print("summed descriptor gradients [feature, atom, xyz]", gradient.shape)
# model.fit evaluates these rows for every frame and accumulates the regression
# statistics. The shared disk cache reuses geometry rows across fits, including
# changes to target labels and fit weights. Set runtime.cache.mode="off" to
# time fresh rows, or "read_only" to require previously computed rows.
# It avoids a full [center, feature, displaced_atom, xyz] allocation.
model = LinearModel(basis)
model.fit(training, config=config)
saved = model.write(config["metadata"]["output_model"])
restored = LinearModel.read(saved)
results = restored.score(validation, config=config)
exported = restored.export_lammps(config["metadata"]["output_lammps"])
output = Path(config["metadata"]["output_summary"])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print("evaluation RMSE", results)
print("saved ASE model", saved)
print("LAMMPS potential and AUTO files", exported)
