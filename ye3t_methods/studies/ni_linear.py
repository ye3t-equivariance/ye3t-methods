"""Recompute held-out Ni energy/force RMSE for the six saved linear models.

The input is the fixed mlearn snapshot with its published test split. Model
predictions include the exact ZBL overlay specified by each saved manifest.
The public examples supply dataset and model paths through their visible configs.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ase.calculators.mixing import SumCalculator
from ase.io import read

from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
from ye3t_methods.atomistic.yace_native import YE3TYACENativeCalculator
from ye3t_methods import LinearModel


EXPECTED_SPLIT_SHA256 = "0d8cdabef9c48100357911022afc0b366263c6eb34ce08e44d89a0daeafb87d0"
EXPECTED_ACCURACY_SHA256 = "63f8d99fd9e95e7e9a3353a8f3839af84e21a7b02b6386e54e73c64bcdca6fbb"
EXPECTED_MODEL_SHA256 = {
    "ace_60": "6c74654ef41de151be4f2af1f2d4846b3561d1a64378a6ca493dcb07d5277ade",
    "ace_127": "8312ac817500bad20288188c9e10d803a33363a871510d23c776b4cedbc4df87",
    "ace_149": "da1671ec655cd5e09bb80aef1a2a6a5a40410bb7929b34c603da5fd12fd1e034",
    "ye3t_tagged_60": "cebb60d833eb40a6c03acfd41c347cedbaff5648d7f9fe72a35b65bc6b6fd7d8",
    "ye3t_tagged_127": "d5c780926a6202693db6bc41903aee8cd4b4b85f90606e5c0202df8f4dd4bbac",
    "ye3t_tagged_149": "62ff7910a7a84972098f8cfba5044623089d64a9494b9a30ce2ca16726b14cce",
}
EXPECTED_MANIFEST_SHA256 = {
    "ace_60": "e79206950ed0fe4c0f006b2011378bffde1f3f0eab69e50f986b384f0b872386",
    "ace_127": "ebbe2c5eda14c8f566f7876e88204581f1caab2e76dfeecb7554b656de005cd4",
    "ace_149": "1f39da6c3daf5cf3e206e9d6b10d1ffe0381dbdbf2bf34440f646574a01e586c",
    "ye3t_tagged_60": "216b6873995182808f5fd269a1268c64c9acb4f5c002e12f9b5ab4dadd8e4f73",
    "ye3t_tagged_127": "eb19aeecfdbbf08af73ed695a2b431359d6ac5fd5014ad1edc1e1338d1a3a6af",
    "ye3t_tagged_149": "a22bf24b6b083920a0cde471df9e203d1975232ce4e61663cbdc670b94e6b4c1",
}
ENERGY_RMSE_TOLERANCE = 5e-7
FORCE_RMSE_TOLERANCE = 1e-8


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_test_frames(data_root, maximum):
    data_path = data_root / "Ni/ni_all.xyz"
    split_path = data_root / "Ni/moment_star_split.json"
    manifest_path = data_root / "MANIFEST.json"
    if not all(path.is_file() for path in (data_path, split_path, manifest_path)):
        raise FileNotFoundError(
            "Ni mlearn snapshot is missing. Set metadata.dataset_root to a directory containing "
            "MANIFEST.json and Ni/{ni_all.xyz,moment_star_split.json}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["Ni"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    data_sha = sha256(data_path)
    if (data_sha != manifest["sha256"] or data_sha != split["dataset"]["sha256"]
            or sha256(split_path) != EXPECTED_SPLIT_SHA256):
        raise ValueError("Dataset or fixed Ni split hash differs from the benchmark")
    indices = split["indices"]["test"]
    if (len(indices) != 31 or len(set(indices)) != 31
            or set(indices) & set(split["indices"]["train"])):
        raise ValueError("Published Ni test split is incomplete or overlaps training")
    if maximum is not None:
        indices = indices[:maximum]
    all_frames = list(read(str(data_path), index=":"))
    if len(all_frames) != manifest["frame_count"]:
        raise ValueError("Unexpected Ni dataset frame count")
    frames = [all_frames[index] for index in indices]
    if any(frame.info.get("source_split") != "test" for frame in frames):
        raise ValueError("Selected frames do not match the published test split")
    targets = [(float(frame.get_potential_energy()),
                np.asarray(frame.get_forces(), dtype=np.float64)) for frame in frames]
    return frames, targets, indices, data_sha, sha256(split_path)


def model_calculator(models_root, model_name, native_library):
    root = models_root / "models" / model_name
    manifest_path = root / "model_manifest.json"
    manifest_sha = sha256(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256[model_name]:
        raise ValueError("Saved model manifest differs from the Ni benchmark: " + model_name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise ValueError("Saved model artifact differs: " + str(path))
    artifact_path = root / ("potential.yace" if model_name.startswith("ace_")
                            else "model.ye3t.json")
    artifact_sha = sha256(artifact_path)
    if artifact_sha != EXPECTED_MODEL_SHA256[model_name]:
        raise ValueError("Model bytes differ from the Ni benchmark: " + model_name)
    if model_name.startswith("ace_"):
        linear = YE3TYACENativeCalculator.from_artifact(
            artifact_path, native_library=native_library, neighbor_skin=0.3)
        zbl = YE3TZBLCalculator.from_model_manifest(manifest_path)
        calculator = SumCalculator([linear, zbl])
        route = "native_yace_plus_manifest_zbl"
    else:
        model = LinearModel.read(artifact_path)
        calculator = model.ase_calculator(
            evaluator="native_cpu", native_library=native_library,
            execution_policy="direct")
        route = "public_linear_model_legacy_composite_native_cpu_plus_manifest_zbl"
    return calculator, artifact_sha, manifest_sha, route


def score(frames, targets, calculator):
    energy_sse = 0.0
    force_sse = 0.0
    force_components = 0
    for frame, (energy_reference, forces_reference) in zip(frames, targets):
        atoms = frame.copy()
        atoms.calc = calculator
        energy_error = (atoms.get_potential_energy() - energy_reference) / len(atoms)
        force_errors = np.asarray(atoms.get_forces()) - forces_reference
        if not math.isfinite(energy_error) or not np.isfinite(force_errors).all():
            raise ValueError("Nonfinite Ni model prediction")
        energy_sse += energy_error * energy_error
        force_sse += float(np.sum(force_errors * force_errors))
        force_components += force_errors.size
    return {
        "energy_rmse_eV_per_atom": math.sqrt(energy_sse / len(frames)),
        "force_rmse_eV_per_A": math.sqrt(force_sse / force_components),
        "test_frame_count": len(frames),
        "force_component_count": force_components,
    }


def compare_saved_models(config):
    """Reproduce the six Ni paper RMSE rows with pinned datasets and model bytes."""
    if config["runtime"]["evaluator"] != "native_cpu":
        raise ValueError("The saved ordinary/tagged comparison requires native_cpu.")
    if config["runtime"].get("device", "cpu") != "cpu" or config["runtime"].get("dtype", "float64") != "float64":
        raise ValueError("The saved comparison uses device='cpu' and dtype='float64'.")
    if config["targets"] != {"energy": "energy", "forces": "forces", "stress": None}:
        raise ValueError("The fixed Ni comparison uses energy/forces labels without stress targets.")
    data_root = Path(config["metadata"]["dataset_root"])
    models_root = Path(config["metadata"]["models_root"])
    output = Path(config["metadata"]["output_summary"])
    maximum = config["validation"].get("max_frames")
    counts = config["model"]["feature_counts"]
    if not counts or any(count not in (60,127,149) for count in counts):
        raise ValueError("feature_counts must select saved 60, 127 or 149 feature models.")
    native_library = config["runtime"].get("native_library")
    if maximum is not None and not 1 <= maximum <= 31:
        raise ValueError("validation.max_frames must be between 1 and 31")
    counts = sorted(set(counts))
    frames, targets, indices, data_sha, split_sha = load_test_frames(
        data_root, maximum)
    accuracy_path = models_root / "evidence/accuracy.csv"
    if sha256(accuracy_path) != EXPECTED_ACCURACY_SHA256:
        raise ValueError("Retained Ni accuracy table differs from the benchmark")
    with accuracy_path.open(newline="", encoding="utf-8") as handle:
        retained = {row["model"]: row for row in csv.DictReader(handle)}
    rows = []
    for count in counts:
        for model_name in ("ace_%d" % count, "ye3t_tagged_%d" % count):
            calculator, artifact_sha, manifest_sha, route = model_calculator(
                models_root, model_name, native_library)
            result = score(frames, targets, calculator)
            row = {"model": model_name, "artifact_sha256": artifact_sha,
                   "manifest_sha256": manifest_sha, "evaluator_route": route, **result}
            if maximum is None:
                reference = retained[model_name]
                for metric, retained_key in (
                    ("energy_rmse_eV_per_atom", "test_energy_rmse_eV_per_atom"),
                    ("force_rmse_eV_per_A", "test_force_rmse_eV_per_A"),
                ):
                    expected = float(reference[retained_key])
                    row["retained_" + metric] = expected
                    row["absolute_difference_" + metric] = abs(result[metric] - expected)
            rows.append(row)
    report = {
        "schema": "ye3t_ni_saved_model_test_rmse_reproduction_v1",
        "description": "Published Ni mlearn test split, total saved model plus manifest ZBL",
        "dataset_sha256": data_sha,
        "split_sha256": split_sha,
        "test_indices": indices,
        "retained_accuracy_sha256": sha256(accuracy_path),
        "diagnostic_subset": maximum is not None,
        "models": rows,
    }
    if maximum is None:
        report["baseline_tolerances"] = {
            "energy_rmse_eV_per_atom": ENERGY_RMSE_TOLERANCE,
            "force_rmse_eV_per_A": FORCE_RMSE_TOLERANCE,
        }
        report["baseline_reproduced"] = all(
            row["absolute_difference_energy_rmse_eV_per_atom"] <= ENERGY_RMSE_TOLERANCE
            and row["absolute_difference_force_rmse_eV_per_A"] <= FORCE_RMSE_TOLERANCE
            for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if maximum is None and not report["baseline_reproduced"]:
        raise RuntimeError("Saved Ni model RMSE differs from the retained baseline")

    return report


def load_paper_basis(config):
    """Load the paper's selected compiler coordinates for a fresh linear fit.

    The common archive contains the complete ordinary catalogue and tagged
    program. The density selection comes from the published model manifest.
    The returned Basis can be inspected, evaluated, and passed to LinearModel.
    """
    from ye3t_methods import Basis
    from ye3t_methods.portable_archive import read_portable_linear_archive, select_portable_density_basis

    settings = config["model"]
    source = Path(settings["source_archive"])
    if sha256(source) != settings["source_archive_sha256"]:
        raise ValueError("Paper basis archive differs from the configured source.")
    record = read_portable_linear_archive(source)
    if record is None:
        raise ValueError("Paper basis requires a portable Ni compiler archive.")
    family = settings["basis_family"]
    if family not in {"density", "tagged"}:
        raise ValueError("Paper basis_family must be density or tagged.")
    manifest_path = Path(settings["selection_manifest"])
    name = ("ace_" if family == "density" else "ye3t_tagged_") + str(settings["feature_count"])
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256.get(name):
        raise ValueError("Basis selection differs from the paper manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if family == "density":
        select_portable_density_basis(record, manifest["model"]["feature_indices"])
    elif record["weights"]["feature_indices"].tolist() != manifest["model"]["feature_indices"]:
        raise ValueError("Tagged archive coordinates differ from the paper selection.")
    if len(record["labels"]) != settings["feature_count"]:
        raise ValueError("Paper basis feature count differs from the selected source.")
    return Basis._from_portable_linear(record)


def load_refit_structures(config):
    """Read the fixed Ni fitting and evaluation split after checking provenance."""
    metadata = config["metadata"]
    data = Path(metadata["dataset"])
    split_path = Path(metadata["split"])
    source = Path(config["model"]["source_archive"])
    if (sha256(data) != metadata["dataset_sha256"] or
            sha256(source) != config["model"]["source_archive_sha256"] or
            sha256(split_path) != EXPECTED_SPLIT_SHA256):
        raise ValueError("Ni refit data, split or source model differs from its pinned hash.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    training_indices = sorted(split["indices"]["train"] + split["indices"]["validation"])
    test_indices = split["indices"]["test"]
    if (len(training_indices) != config["validation"]["expected_train_count"] or
            len(test_indices) != config["validation"]["expected_test_count"] or
            len(set(training_indices)) != len(training_indices) or
            len(set(test_indices)) != len(test_indices) or set(training_indices) & set(test_indices)):
        raise ValueError("Ni refit split counts or disjointness differ from the study.")
    frames = read(str(data), index=":")
    training = [frames[index] for index in training_indices]
    testing = [frames[index] for index in test_indices]
    if (any(atoms.info["source_split"] != "training" for atoms in training) or
            any(atoms.info["source_split"] != "test" for atoms in testing)):
        raise ValueError("Ni frame metadata differs from the selected split.")
    return training, testing
