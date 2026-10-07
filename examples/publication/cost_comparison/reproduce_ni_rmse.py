"""Recompute held-out Ni energy/force RMSE for the six saved linear models.

The input is the fixed mlearn snapshot with its published test split. Model
predictions include the exact ZBL overlay specified by each saved manifest.
Run this from an extracted ye3t-methods source archive after installing the
native methods wheel; use --dataset-root for a separately downloaded snapshot.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from ase.calculators.mixing import SumCalculator
from ase.io import read

from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
from ye3t_methods.atomistic.yace_native import YE3TYACENativeCalculator
from ye3t_methods import LinearModel


HERE = Path(__file__).resolve().parent
METHODS_ROOT = HERE.parents[2]
DEFAULT_DATA = METHODS_ROOT / "examples/data/mlearn"
DEFAULT_MODELS = HERE / "lammps/Ni"
DEFAULT_OUTPUT = (
    METHODS_ROOT.parent / "ye3t-workflows/MLIP/cost_comparison/ni_rmse_recomputed.json"
)
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
            "Ni mlearn snapshot is missing. Supply --dataset-root containing "
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--native-library", default=os.environ.get("YE3T_TAGGED_C_API_LIBRARY"))
    parser.add_argument("--counts", nargs="+", type=int, choices=(60, 127, 149),
                        default=(60, 127, 149))
    parser.add_argument("--max-frames", type=int,
                        help="Diagnostic subset only; omits the baseline comparison")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_frames is not None and not 1 <= args.max_frames <= 31:
        parser.error("--max-frames must be between 1 and 31")
    counts = sorted(set(args.counts))
    frames, targets, indices, data_sha, split_sha = load_test_frames(
        args.dataset_root, args.max_frames)
    accuracy_path = args.models_root / "evidence/accuracy.csv"
    if sha256(accuracy_path) != EXPECTED_ACCURACY_SHA256:
        raise ValueError("Retained Ni accuracy table differs from the benchmark")
    with accuracy_path.open(newline="", encoding="utf-8") as handle:
        retained = {row["model"]: row for row in csv.DictReader(handle)}
    rows = []
    print("model energy_RMSE_eV_per_atom force_RMSE_eV_per_A", flush=True)
    for count in counts:
        for model_name in ("ace_%d" % count, "ye3t_tagged_%d" % count):
            calculator, artifact_sha, manifest_sha, route = model_calculator(
                args.models_root, model_name, args.native_library)
            result = score(frames, targets, calculator)
            row = {"model": model_name, "artifact_sha256": artifact_sha,
                   "manifest_sha256": manifest_sha, "evaluator_route": route, **result}
            if args.max_frames is None:
                reference = retained[model_name]
                for metric, retained_key in (
                    ("energy_rmse_eV_per_atom", "test_energy_rmse_eV_per_atom"),
                    ("force_rmse_eV_per_A", "test_force_rmse_eV_per_A"),
                ):
                    expected = float(reference[retained_key])
                    row["retained_" + metric] = expected
                    row["absolute_difference_" + metric] = abs(result[metric] - expected)
            rows.append(row)
            print(model_name, result["energy_rmse_eV_per_atom"],
                  result["force_rmse_eV_per_A"], flush=True)
    report = {
        "schema": "ye3t_ni_saved_model_test_rmse_reproduction_v1",
        "description": "Published Ni mlearn test split, total saved model plus manifest ZBL",
        "dataset_sha256": data_sha,
        "split_sha256": split_sha,
        "test_indices": indices,
        "retained_accuracy_sha256": sha256(accuracy_path),
        "diagnostic_subset": args.max_frames is not None,
        "models": rows,
    }
    if args.max_frames is None:
        report["baseline_tolerances"] = {
            "energy_rmse_eV_per_atom": ENERGY_RMSE_TOLERANCE,
            "force_rmse_eV_per_A": FORCE_RMSE_TOLERANCE,
        }
        report["baseline_reproduced"] = all(
            row["absolute_difference_energy_rmse_eV_per_atom"] <= ENERGY_RMSE_TOLERANCE
            and row["absolute_difference_force_rmse_eV_per_A"] <= FORCE_RMSE_TOLERANCE
            for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.max_frames is None:
        print("baseline_reproduced", report["baseline_reproduced"], flush=True)
    else:
        print("diagnostic_subset", len(frames), flush=True)
    print("wrote", args.output)
    if args.max_frames is None and not report["baseline_reproduced"]:
        raise RuntimeError("Saved Ni model RMSE differs from the retained baseline")


if __name__ == "__main__":
    main()
