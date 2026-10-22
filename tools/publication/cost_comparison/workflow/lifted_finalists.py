#!/usr/bin/env python3
"""Fit and deploy approximately 127-coordinate lifted-density controls."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import finalize_models as final
import optimize_cached as study
from ye3t_methods.atomistic.ace.lammps_export import compile_ordinary_scalar_catalogue
from ye3t_methods.atomistic.ace.linear_ace import load_xyz_structures


ORDINARY_COUNT = 70
LIFTED_COUNT = 55
MODEL_ID = f"lifted_{ORDINARY_COUNT + LIFTED_COUNT}"
TRAINER = (
    study.PUBLIC.parent
    / "ta_lifted_linear_lammps"
    / "train_export.py"
)
SOURCE_LABELS = (
    TRAINER.parent / "catalogues" / "manual_labels_c2_lifted_full.json"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relabel(value, system):
    if isinstance(value, dict):
        return {key: relabel(item, system) for key, item in value.items()}
    if isinstance(value, list):
        return [relabel(item, system) for item in value]
    if value == "Ta":
        return system
    return value


def ordinary_radial_nmax(application):
    values = [
        int(n)
        for row in application["rows"]
        for n in row["compact_label"]["n_tuple"]
    ]
    if not values:
        raise ValueError("The ordinary catalogue has no radial labels.")
    return max(values)


def compile_backbone(system, output):
    source_path = (
        study.HERE
        / system
        / "catalogues"
        / "ordinary_catalogue_source.json"
    )
    source = study.read_json(source_path)
    profile_id = f"{system.lower()}_generated_linear_ace_{ORDINARY_COUNT}"
    if profile_id not in source["profiles"]:
        raise KeyError(f"Missing ordinary catalogue profile {profile_id}.")
    application_path = output / f"ordinary_catalogue_{ORDINARY_COUNT}.json"
    if application_path.is_file():
        application = study.read_json(application_path)
    else:
        application = compile_ordinary_scalar_catalogue(source, profile_id)
        study.write_json(application_path, application)
    if len(application["rows"]) != ORDINARY_COUNT:
        raise RuntimeError("The lifted control ordinary backbone changed size.")
    return application_path, application


def split_records(system, output):
    training = load_xyz_structures(
        study.data_root(system) / f"{system.lower()}_training.xyz"
    )
    test = load_xyz_structures(study.data_root(system) / f"{system.lower()}_test.xyz")
    combined_path = study.data_root(system) / f"{system.lower()}_all.xyz"
    assignments = study.fold_assignments(training, 5, 1701)
    validation = sorted(index for index, fold in assignments.items() if fold == 0)
    validation_set = set(validation)
    train = [index for index in range(len(training)) if index not in validation_set]
    test_indices = list(range(len(training), len(training) + len(test)))
    dataset = {
        "path": str(combined_path.resolve()),
        "frame_count": len(training) + len(test),
        "sha256": sha256(combined_path),
    }
    selection = {
        "schema": "ye3t_mlearn_lifted_selection_split_v1",
        "selection_blind_to_test_targets": True,
        "policy": "fold_zero_of_five_group_stratified_training_folds",
        "dataset": dataset,
        "indices": {
            "train": train,
            "validation": validation,
            "test": test_indices,
        },
    }
    final_split = {
        "schema": "ye3t_mlearn_lifted_final_split_v1",
        "selection_blind_to_test_targets": True,
        "selection_source": "selection_split.json and selection_fit/summary.json",
        "dataset": dataset,
        "indices": {
            "train": list(range(len(training))),
            "validation": test_indices,
        },
    }
    study.write_json(output / "selection_split.json", selection)
    study.write_json(output / "final_split.json", final_split)
    return training, test, selection, final_split


def source_catalogue(system, output):
    catalogue = relabel(study.read_json(SOURCE_LABELS), system)
    labels = catalogue["manual_labels"]
    if len(labels) != LIFTED_COUNT:
        raise RuntimeError(f"Expected {LIFTED_COUNT} lifted coordinates.")
    catalogue["catalogue_id"] = f"{system.lower()}_c2_lifted_full_rank4"
    catalogue["provenance"] = {
        "source": str(SOURCE_LABELS.resolve()),
        "source_sha256": sha256(SOURCE_LABELS),
        "mapping": f"exact neighbor-species relabel Ta to {system}",
    }
    path = output / "manual_labels.json"
    study.write_json(path, catalogue)
    return path, catalogue


def fit_weighting(system):
    fit = study.read_json(
        study.system_root(system)
        / "finalist"
        / "fits"
        / "ace_127.json"
    )["selected"]
    return {
        "energy_weight": float(fit["energy_weight"]),
        "force_weight": float(fit["force_weight"]),
        "group_weights": dict(fit["group_weights"]),
    }


def base_config(
    system,
    output,
    application_path,
    application,
    catalogue_path,
    training,
    selection,
    selected,
):
    system_config = study.read_system_config(system)
    weighting = fit_weighting(system)
    radial = selected["radial"]
    zbl = dict(selected["zbl"])
    zbl["executable"] = str(
        Path(zbl.get("executable", "lmp")).resolve()
        if Path(zbl.get("executable", "lmp")).is_file()
        else zbl.get("executable", "lmp")
    )
    directed = {
        f"{system}-{system}": {
            "cutoff_A": float(radial["cutoff_A"]),
            "cutoff_width_A": float(radial["cutoff_width_A"]),
            "inner_cutoff_A": 0.0,
            "inner_cutoff_width_A": 0.0,
            "lambda": float(radial["radial_lambda"]),
            "spline_spacing_A": 0.001,
        }
    }
    train_atoms = int(sum(len(training[index]) for index in selection["indices"]["train"]))
    return {
        "metadata": {
            "config_schema": "ye3t_example_config_v1",
            "name": f"{system.lower()}_ace70_plus_lifted55_cost_control",
            "status": "publication_control",
        },
        "basis": {
            "type": "ordinary_ace_plus_lifted_cauchy",
            "species": [system],
            "density_normalization": "none",
            "radial": {
                "type": "orthogonal_shifted_jacobi_l1_v1",
                "source_family_count_per_role": 2,
                "source_span_dimension": 4,
                "polynomial_degree_map": "q=2*n+s",
                "measure": "x^2 dx",
                "normalization": "identity_source_gram",
                "cutoff_A": float(radial["cutoff_A"]),
                "directed_bonds": directed,
            },
            "angular": {
                "l": 1,
                "kind": "physical_real_solid_harmonic",
                "normalization": "(dx,dz,-dy)/rc",
            },
            "descriptor_catalogue": {
                "mode": "manual_labels",
                "catalogue_id": f"{system.lower()}_c2_lifted_full_rank4",
                "manual_labels": str(catalogue_path.resolve()),
                "expected_descriptor_count": LIFTED_COUNT,
                "coordinate_policy": "parent_prefix_orthogonal",
                "coverage": "complete_c2_rank4_l1_parent_closure",
                "compiler": {
                    "source": "ye3t.couplings",
                    "emit_ordered_reference": False,
                    "emit_canonical": True,
                    "emit_factored": True,
                },
            },
            "ordinary_backbone": {
                "catalogue_id": str(application["profile_id"]),
                "compiled_application": str(application_path.resolve()),
                "compiled_application_sha256": sha256(application_path),
                "application_sha256": str(application["application_sha256"]),
                "membership_sha256": str(application["membership_sha256"]),
                "expected_feature_count": ORDINARY_COUNT,
                "fast_path": "disable",
                "angular": {
                    "kind": "complex",
                    "normalization": "pace_y00_one",
                },
                "radial": {
                    "type": "PACE_ChebExpCos",
                    "n_max": ordinary_radial_nmax(application),
                    "directed_bonds": directed,
                },
                "runtime": {
                    "backend": "pytorch",
                    "device": "cpu",
                    "dtype": "float64",
                    "strict_backend": True,
                    "validate_backend": True,
                },
            },
        },
        "representation": {
            "carrier": "A_s",
            "carrier_options": {
                "role_coordinate_policy": "role_resolved",
                "role_dimension": 2,
            },
            "coupling": {
                "backend": "exact",
                "fast_path": "symmetric_power_block_dag",
                "source": "ye3t.couplings",
            },
            "target": {"L": 0, "parity": "even", "permutation": "trivial"},
        },
        "runtime": {
            "backend": "pytorch",
            "cache": "auto",
            "derivatives": "explicit_product_adjoint",
            "device": "cpu",
            "dtype": "float64",
            "accumulation_dtype": "float64",
            "execution_mode": "fit_export",
            "profile": False,
            "profile_compiler_repeat": False,
            "output_directory": str((output / "selection_fit").resolve()),
            "strict_backend": True,
        },
        "model": {
            "type": "linear",
            "fit_method": "ridge_streaming_gram",
            "fit_coordinate_policy": "orthogonal",
            "realization": "factored",
            "source_realization": "factorized",
            "ridge_alphas": [
                1.0e-10,
                1.0e-8,
                1.0e-6,
                1.0e-4,
                1.0e-2,
                1.0e-1,
                1.0,
            ],
            "ridge_selection_metric": "validation_structure_balanced_normalized_E1_F1",
            "svd_rcond": 1.0e-12,
            "normal_equation_maximum_condition": 1.0e24,
            "feature_chunk_size": "auto",
        },
        "targets": {
            "dataset": {
                "path": selection["dataset"]["path"],
                "sha256": selection["dataset"]["sha256"],
            },
            "energy": "energy",
            "energy_weight": weighting["energy_weight"],
            "forces": "forces",
            "force_weight": weighting["force_weight"],
            "stress_fit": False,
            "reference_potential": zbl,
            "structure_weighting": {
                "group_key": "config_type",
                "group_weights": weighting["group_weights"],
                "default_weight": 1.0,
                "normalize_mean": True,
            },
        },
        "validation": {
            "split": {
                "mode": "manifest",
                "manifest": str((output / "selection_split.json").resolve()),
                "train_partition": "train",
                "validation_partition": "validation",
                "frame_limits": {"train": None, "validation": None},
                "preflight": {
                    "train_structure_count": len(selection["indices"]["train"]),
                    "train_atom_count": train_atoms,
                },
            },
            "checks": [
                "finite_predictions",
                "force_finite_difference",
                "bundle_round_trip",
                "byte_deterministic_export",
            ],
            "force_finite_difference_atol_eV_per_A": 1.0e-5,
            "mode": "dataset_and_derivatives",
        },
        "extensions": {
            "lifted_cauchy": {
                "basis_scope": "fixed_N_fixed_content_complete_C2_parent_coordinates",
                "ordinary_component": "generated 70-feature PACE-compatible backbone",
                "comparison_scope": (
                    "approximately count-matched lifted-density capacity control; "
                    "lifted shifted-Jacobi source is explicitly distinct from the "
                    "tagged PACE source"
                ),
            }
        },
    }


def prepare_system(system):
    output = study.system_root(system) / "finalist" / "lifted_control"
    output.mkdir(parents=True, exist_ok=True)
    selected = final.selected_inputs(system)
    application_path, application = compile_backbone(system, output)
    catalogue_path, _catalogue = source_catalogue(system, output)
    training, _test, selection, final_split = split_records(system, output)
    config = base_config(
        system,
        output,
        application_path,
        application,
        catalogue_path,
        training,
        selection,
        selected,
    )
    study.write_json(output / "selection_config.json", config)
    record = {
        "schema": "ye3t_mlearn_lifted_control_preparation_v1",
        "system": system,
        "model_id": MODEL_ID,
        "descriptor_count": ORDINARY_COUNT + LIFTED_COUNT,
        "ordinary_descriptor_count": ORDINARY_COUNT,
        "lifted_descriptor_count": LIFTED_COUNT,
        "selection_config": str(output / "selection_config.json"),
        "final_split": final_split,
        "radial_selection_sha256": selected["radial_sha256"],
        "zbl_selection_sha256": selected["zbl_sha256"],
    }
    study.write_json(output / "preparation.json", record)
    return output, config


def run_trainer(config, output, log, timeout):
    command = [
        sys.executable,
        str(TRAINER),
        "--config",
        str(config),
        "--output",
        str(output),
        "--execution-mode",
        "fit_export",
    ]
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    with Path(log).open("w", encoding="utf-8") as handle:
        handle.write("command: " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=TRAINER.parent,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Lifted fit failed with code {completed.returncode}; see {log}.")


def install_deployment(system, model_output):
    root = study.system_root(system) / "finalist"
    deploy = root / "deploy"
    target = deploy / "models" / MODEL_ID
    target.mkdir(parents=True, exist_ok=True)
    for source in model_output.iterdir():
        destination = target / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
    overlay = study.read_json(target / "lammps_hybrid_overlay.json")
    prefix = f"models/{MODEL_ID}/"
    pair_coefficients = []
    for line in overlay["pair_coeff"]:
        fields = line.split()
        for index, value in enumerate(fields):
            if value.startswith("model.ye3t/") or value.startswith("lifted_component.ye3t/"):
                fields[index] = prefix + value
        pair_coefficients.append("pair_coeff " + " ".join(fields))
    record = {
        "id": MODEL_ID,
        "label": f"lifted density ({ORDINARY_COUNT + LIFTED_COUNT})",
        "family": "lifted",
        "descriptor_count": ORDINARY_COUNT + LIFTED_COUNT,
        "ordinary_descriptor_count": ORDINARY_COUNT,
        "tagged_descriptor_count": 0,
        "pair_style": "pair_style " + overlay["pair_style"],
        "pair_coefficients": pair_coefficients,
        "model_directory": f"models/{MODEL_ID}",
    }
    matrix_path = deploy / "lammps_models.json"
    matrix = study.read_json(matrix_path)
    matrix["models"] = [row for row in matrix["models"] if row["id"] != MODEL_ID]
    matrix["models"].append(record)
    study.write_json(matrix_path, matrix)
    input_path = deploy / f"in.{system.lower()}_{MODEL_ID}"
    input_path.write_text(
        final.input_deck(
            study.read_system_config(system),
            record,
        ),
        encoding="utf-8",
    )
    return record


def fit_system(system, timeout):
    started = time.perf_counter()
    output, config = prepare_system(system)
    selection_output = output / "selection_fit"
    if not (selection_output / "summary.json").is_file():
        run_trainer(
            output / "selection_config.json",
            selection_output,
            output / "screen.selection_fit.log",
            timeout,
        )
    selection_summary = study.read_json(selection_output / "summary.json")
    selected_alpha = float(selection_summary["ridge_screen"]["selected_alpha"])
    final_config = copy.deepcopy(config)
    final_config["metadata"]["name"] = (
        f"{system.lower()}_ace70_plus_lifted55_published_test"
    )
    final_config["runtime"]["output_directory"] = str((output / "model").resolve())
    final_config["model"].pop("ridge_alphas")
    final_config["model"].pop("ridge_selection_metric")
    final_config["model"]["ridge_alpha"] = selected_alpha
    final_config["validation"]["split"].update(
        {
            "manifest": str((output / "final_split.json").resolve()),
            "train_partition": "train",
            "validation_partition": "validation",
            "preflight": {
                "train_structure_count": len(
                    study.read_json(output / "final_split.json")["indices"]["train"]
                ),
                "train_atom_count": int(
                    study.read_json(study.data_root(system) / "dataset_manifest.json")
                    ["splits"]["training"]["atoms"]
                ),
            },
        }
    )
    study.write_json(output / "final_config.json", final_config)
    model_output = output / "model"
    if not (model_output / "summary.json").is_file():
        run_trainer(
            output / "final_config.json",
            model_output,
            output / "screen.final_fit.log",
            timeout,
        )
    final_summary = study.read_json(model_output / "summary.json")
    record = install_deployment(system, model_output)
    validation = selection_summary["validation"]
    test = final_summary["validation"]
    summary = {
        "schema": "ye3t_mlearn_lifted_control_test_v1",
        "system": system,
        "model": MODEL_ID,
        "family": "lifted",
        "descriptor_count": ORDINARY_COUNT + LIFTED_COUNT,
        "ordinary_descriptor_count": ORDINARY_COUNT,
        "lifted_descriptor_count": LIFTED_COUNT,
        "cv_policy": "one predeclared group-stratified fold used for bounded control",
        "cv_energy_rmse_mean_eV_per_atom": float(
            validation["energy_rmse_eV_per_atom"]
        ),
        "cv_energy_rmse_std_eV_per_atom": 0.0,
        "cv_force_rmse_mean_eV_per_A": float(validation["force_rmse_eV_per_A"]),
        "cv_force_rmse_std_eV_per_A": 0.0,
        "test_energy_rmse_eV_per_atom": float(test["energy_rmse_eV_per_atom"]),
        "test_force_rmse_eV_per_A": float(test["force_rmse_eV_per_A"]),
        "ridge_alpha": selected_alpha,
        "energy_weight": float(final_config["targets"]["energy_weight"]),
        "force_weight": float(final_config["targets"]["force_weight"]),
        "group_weights": dict(
            final_config["targets"]["structure_weighting"]["group_weights"]
        ),
        "source_match_status": (
            "cutoff and ordinary PACE backbone matched; lifted shifted-Jacobi "
            "source intentionally distinct and orthogonal"
        ),
        "lammps_record": record,
        "elapsed_seconds": time.perf_counter() - started,
    }
    study.write_json(output / "test_summary.json", summary)
    study.append_progress("lifted_control_complete", **summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for system in args.systems:
        if args.prepare_only:
            prepare_system(system)
        else:
            fit_system(system, args.timeout_seconds)


if __name__ == "__main__":
    main()
