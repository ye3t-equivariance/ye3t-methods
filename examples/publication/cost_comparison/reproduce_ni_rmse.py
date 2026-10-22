"""Reproduce the six saved Ni energy/force RMSE rows on the full 31-frame split.

The source archive includes the fixed dataset and model bundle. Edit the paths
for separately downloaded copies; their hashes are checked against the study.
Requires the native methods installation. max_frames is an optional diagnostic
override; leave it None for the complete accuracy comparison.
"""
import os
from pathlib import Path
from ye3t_methods.studies.ni_linear import compare_saved_models

example = Path(__file__).resolve().parent
workflow = Path(os.environ.get("YE3T_WORKFLOW_ROOT", "~/ye3t-workflows")).expanduser()
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "ni_saved_model_rmse", "status": "publication",
        "dataset_root": str(example.parents[1] / "data" / "mlearn"),
        "models_root": str(example / "lammps" / "Ni"),
        "output_summary": str(workflow / 'MLIP/cost_comparison/ni_rmse_recomputed.json'),
    },
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "native_cpu", "native_library": None, "device": "cpu", "dtype": "float64"},
    "model": {"kind": "linear", "feature_counts": [60, 127, 149]},
    "targets": {"energy": "energy", "forces": "forces", "stress": None},
    "validation": {"max_frames": None},
}

results = compare_saved_models(config)
print("model                    energy / eV per atom    forces / eV per Angstrom")
for row in results["models"]:
    print(f"{row['model']:24s} {row['energy_rmse_eV_per_atom']:18.8f} {row['force_rmse_eV_per_A']:24.8f}")
print("saved summary", config["metadata"]["output_summary"])
