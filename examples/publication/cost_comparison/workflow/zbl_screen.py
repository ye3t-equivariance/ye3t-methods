#!/usr/bin/env python3
"""Select an exact LAMMPS-ZBL residual convention using training folds only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list

import optimize_cached as study
import radial_screen as radial
from ye3t_ace.ace.linear_ace import load_xyz_structures
from ye3t_ace.linear_statistics import structure_linear_statistics, sum_linear_statistics
from ye3t_ace.reference_potentials import lammps_zbl_reference_config


def minimum_training_distance(structures):
    minimum = np.inf
    for atoms in structures:
        distances = neighbor_list("d", atoms, 4.0)
        if len(distances):
            minimum = min(minimum, float(np.min(distances)))
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise RuntimeError("Could not determine a positive training-set pair distance.")
    return minimum


def reference_candidates(system, atomic_number, executable, minimum_distance):
    reference = study.read_json(study.CONFIG_PATH)["targets"][
        "reference_potential"
    ]
    if reference["type"] != "lammps_zbl_training_residual":
        raise ValueError("The ZBL screen requires lammps_zbl_training_residual.")
    ratios = tuple(
        tuple(float(value) for value in pair)
        for pair in reference["parameters"][
            "inner_outer_ratios_to_minimum_training_distance"
        ]
    )
    if not ratios or any(
        len(pair) != 2 or pair[0] <= 0.0 or pair[1] <= pair[0]
        for pair in ratios
    ):
        raise ValueError("Every ZBL inner/outer ratio pair must be positive and ordered.")
    rows = []
    for candidate, (inner_ratio, outer_ratio) in enumerate(ratios):
        config = lammps_zbl_reference_config(
            {
                "engine": "lammps_executable",
                "pair_style": "zbl",
                "units": "metal",
                "atom_style": "atomic",
                "inner_cutoff_A": inner_ratio * minimum_distance,
                "outer_cutoff_A": outer_ratio * minimum_distance,
                "atomic_numbers": {system: int(atomic_number)},
                "executable": str(executable),
                "timeout_seconds": 900,
            },
            (system,),
        )
        rows.append(
            {
                "candidate": candidate,
                "inner_ratio_to_training_minimum": inner_ratio,
                "outer_ratio_to_training_minimum": outer_ratio,
                "config": config,
            }
        )
    return tuple(rows)


def selected_radial_inputs(system):
    root = study.system_root(system) / "radial_screen"
    selection_path = root / "selection_frozen.json"
    selection = study.read_json(selection_path)
    if selection["test_split_opened"]:
        raise RuntimeError("The frozen radial selection claims test-set access.")
    selected = selection["selected"]
    candidate = int(selected["candidate"])
    subset = study.read_json(root / "subset.json")
    indices = tuple(int(value) for value in subset["indices"])
    if candidate:
        candidate_root = root / f"force_candidate_{candidate:02d}"
        for index in indices:
            for directory in ("row_cache_ordinary", "row_cache_tagged"):
                path = candidate_root / directory / f"frame_{index:04d}.npz"
                if not path.is_file():
                    raise FileNotFoundError(
                        "Selected radial force rows are incomplete: " + str(path)
                    )
    return selection_path, selection, selected, indices


def selected_radial_row(system, selected, global_index):
    candidate = int(selected["candidate"])
    if candidate == 0:
        ordinary = study.load_row(
            study.ordinary_row_root(system) / f"frame_{global_index:04d}.npz"
        )
        tagged = study.load_row(
            study.tagged_row_root(system) / f"frame_{global_index:04d}.npz"
        )
    else:
        root = (
            study.system_root(system)
            / "radial_screen"
            / f"force_candidate_{candidate:02d}"
        )
        ordinary = radial.load_value_row(
            root / "row_cache_ordinary" / f"frame_{global_index:04d}.npz"
        )
        tagged = radial.load_value_row(
            root / "row_cache_tagged" / f"frame_{global_index:04d}.npz"
        )
    if ordinary["atom_count"] != tagged["atom_count"]:
        raise RuntimeError("Selected ordinary/tagged radial rows disagree on atom count.")
    return {
        "feature_sums": np.concatenate(
            (ordinary["feature_sums"], tagged["feature_sums"])
        ),
        "feature_square_sums": np.concatenate(
            (ordinary["feature_square_sums"], tagged["feature_square_sums"])
        ),
        "force_design": np.concatenate(
            (ordinary["force_design"], tagged["force_design"]), axis=1
        ),
        "atom_count": int(ordinary["atom_count"]),
    }


def residual_shards(
    system, structures, global_indices, selected, energies, forces, fold_count, seed
):
    assignments = study.fold_assignments(structures, fold_count, seed)
    records = {}
    row_hasher = hashlib.sha256()
    for local_index, (global_index, atoms) in enumerate(
        zip(global_indices, structures, strict=True)
    ):
        row = selected_radial_row(system, selected, global_index)
        row_hasher.update(
            np.asarray([global_index, row["atom_count"]], dtype="<i8").tobytes()
        )
        for name in ("feature_sums", "feature_square_sums", "force_design"):
            value = np.ascontiguousarray(row[name], dtype="<f8")
            row_hasher.update(np.asarray(value.shape, dtype="<i8").tobytes())
            row_hasher.update(value.tobytes())
        record = structure_linear_statistics(
            row["feature_sums"],
            row["feature_square_sums"],
            row["force_design"],
            len(atoms),
            float(energies[local_index]) / len(atoms),
            np.asarray(forces[local_index], dtype=np.float64),
        )
        key = (
            str(atoms.info["config_type"]),
            int(assignments[local_index]),
        )
        records.setdefault(key, []).append(record)
    return {
        key: sum_linear_statistics(values) for key, values in records.items()
    }, row_hasher.hexdigest()


def persist_shards(path, shards, row_hash, reference):
    arrays, records = study.flatten_shards(shards)
    np.savez_compressed(path / "statistics.npz", **arrays)
    manifest = {
        "schema": "ye3t_mlearn_zbl_statistics_v1",
        "row_content_sha256": row_hash,
        "reference": reference,
        "shards": records,
    }
    study.write_json(path / "statistics.json", manifest)
    return manifest


def select_system(system, executable, fold_count, seed):
    started = time.perf_counter()
    root = study.system_root(system)
    output = root / "zbl_screen"
    output.mkdir(parents=True, exist_ok=True)
    system_config = study.read_system_config(system)
    all_structures = load_xyz_structures(
        study.data_root(system) / f"{system.lower()}_training.xyz"
    )
    radial_path, radial_selection, selected_radial, indices = selected_radial_inputs(
        system
    )
    structures = [all_structures[index] for index in indices]
    minimum_distance = minimum_training_distance(all_structures)
    candidates = reference_candidates(
        system,
        int(system_config["atomic_numbers"][0]),
        executable,
        minimum_distance,
    )
    index, _cache, _baseline_shards = study.load_statistics(system)
    models = {
        row["name"]: row
        for row in study.catalogue_models(index, study.component_inventory(system))
    }
    selected_models = (models["ace_127"], models["ye3t_tagged_127"])
    rows = []
    for candidate in candidates:
        candidate_output = output / f"candidate_{candidate['candidate']:02d}"
        candidate_output.mkdir(parents=True, exist_ok=True)
        config_path = candidate_output / "zbl_reference.json"
        study.write_json(config_path, candidate["config"])
        energies, forces, reference = study.reference_targets(
            structures, system, config_path, candidate_output
        )
        shards, row_hash = residual_shards(
            system,
            structures,
            indices,
            selected_radial,
            energies,
            forces,
            fold_count,
            seed,
        )
        persist_shards(candidate_output, shards, row_hash, reference)
        model_output = candidate_output / "models"
        model_output.mkdir(parents=True, exist_ok=True)
        results = []
        for position, model in enumerate(selected_models):
            results.append(
                study.optimize_model(
                    system,
                    model,
                    shards,
                    fold_count,
                    seed + 1009 * position + sum(map(ord, system)),
                    model_output,
                )
            )
        rows.append(
            {
                **candidate,
                "reference_evaluation": reference,
                "models": {
                    result["name"]: result["selected"] for result in results
                },
            }
        )
    model_names = ("ace_127", "ye3t_tagged_127")
    metric_names = (
        "energy_rmse_mean_eV_per_atom",
        "force_rmse_mean_eV_per_A",
    )
    minima = {
        (model, metric): min(
            float(row["models"][model][metric]) for row in rows
        )
        for model in model_names
        for metric in metric_names
    }
    for row in rows:
        ratios = {
            f"{model}.{metric}": float(row["models"][model][metric])
            / max(minima[(model, metric)], np.finfo(np.float64).tiny)
            for model in model_names
            for metric in metric_names
        }
        row["normalized_metric_ratios"] = ratios
        row["paired_normalized_score"] = float(np.mean(tuple(ratios.values())))
    best_score = min(row["paired_normalized_score"] for row in rows)
    near_best = [
        row for row in rows if row["paired_normalized_score"] <= 1.01 * best_score
    ]
    selected = max(
        near_best,
        key=lambda row: (
            float(row["config"]["outer_cutoff_A"]),
            -float(row["paired_normalized_score"]),
        ),
    )
    summary = {
        "schema": "ye3t_mlearn_zbl_selection_v1",
        "system": system,
        "selection_data": "published_training_split_only",
        "test_split_opened": False,
        "minimum_training_pair_distance_A": minimum_distance,
        "subset_structure_count": len(structures),
        "subset_indices": list(indices),
        "selected_radial": selected_radial,
        "radial_selection": str(radial_path),
        "radial_selection_sha256": hashlib.sha256(radial_path.read_bytes()).hexdigest(),
        "radial_fold_identity": radial_selection["fold_identity"],
        "selection_objective": "paired ACE/tagged energy-force inner-fold score",
        "near_best_relative_tolerance": 0.01,
        "tie_break": "largest ZBL outer cutoff among candidates within one percent",
        "selected_candidate": int(selected["candidate"]),
        "selected_config": selected["config"],
        "candidates": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    study.write_json(output / "selection_frozen.json", summary)
    study.append_progress(
        "zbl_selection_frozen",
        system=system,
        selected_candidate=summary["selected_candidate"],
        inner_cutoff_A=summary["selected_config"]["inner_cutoff_A"],
        outer_cutoff_A=summary["selected_config"]["outer_cutoff_A"],
        elapsed_seconds=summary["elapsed_seconds"],
    )
    print(
        json.dumps(
            {
                "system": system,
                "candidate": summary["selected_candidate"],
                "inner_cutoff_A": summary["selected_config"]["inner_cutoff_A"],
                "outer_cutoff_A": summary["selected_config"]["outer_cutoff_A"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.fold_count < 2:
        raise ValueError("--fold-count must be at least two.")
    executable = args.lammps.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    for system in args.systems:
        select_system(system, executable, args.fold_count, args.seed)


if __name__ == "__main__":
    main()
