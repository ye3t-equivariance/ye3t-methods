#!/usr/bin/env python3
"""Build reusable mlearn statistics and optimize fixed ACE/YE3T catalogues."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from ye3t.cache import canonical_json_bytes
from ye3t_ace.ace.linear_ace import load_xyz_structures
from ye3t_ace.cache import (
    LINEAR_STATISTICS_CACHE_SCHEMA,
    load_linear_sufficient_statistics,
    persist_linear_sufficient_statistics,
)
from ye3t_ace.linear_statistics import (
    assemble_prepared_normal_equations,
    feature_normalization,
    prepare_weighted_normal_equations,
    score_linear_statistics,
    select_linear_statistics,
    solve_ridge_statistics,
    structure_linear_statistics,
    sum_linear_statistics,
)
from ye3t_ace.reference_potentials import (
    evaluate_lammps_zbl_reference,
    lammps_zbl_reference_config,
)


CODE_ROOT = Path(__file__).resolve().parent
PUBLIC = Path(
    os.environ.get("YE3T_COST_PUBLIC_ROOT", str(CODE_ROOT.parent))
).resolve()
CONFIG_PATH = Path(
    os.environ.get("YE3T_COST_CONFIG", str(PUBLIC / "config.json"))
).resolve()
_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
_REQUESTED_WORKFLOW_ROOT = Path(
    os.environ.get(
        "YE3T_COST_WORKFLOW_ROOT",
        str(_CONFIG["runtime"]["workflow_root"]),
    )
)
if not _REQUESTED_WORKFLOW_ROOT.is_absolute():
    _REQUESTED_WORKFLOW_ROOT = (
        CONFIG_PATH.parent / _REQUESTED_WORKFLOW_ROOT
    ).resolve()
HERE = _REQUESTED_WORKFLOW_ROOT
SYSTEM_ROOT = Path(
    os.environ.get("YE3T_COST_SYSTEM_ROOT", str(PUBLIC / "systems"))
).resolve()
WORKSPACE = PUBLIC.parents[3] if len(PUBLIC.parents) > 3 else PUBLIC.parent
SYSTEMS = ("Li", "Mo", "Cu", "Ni", "Si", "Ge")
ARRAY_FIELDS = (
    "feature_sum",
    "feature_square_sum",
    "energy_gram",
    "energy_rhs",
    "force_fit_gram",
    "force_fit_rhs",
    "force_metric_gram",
    "force_metric_rhs",
)
SCALAR_FIELDS = (
    "energy_target_sum",
    "energy_target_square",
    "force_fit_target_square",
    "force_metric_target_square",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def system_config_path(system):
    path = SYSTEM_ROOT / f"{system}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_system_config(system):
    return read_json(system_config_path(system))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def semantic_hash(payload):
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def array_digest(arrays):
    digest = hashlib.sha256()
    digest.update(b"ye3t_linear_statistics_arrays_v1\0")
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def append_progress(event, **payload):
    path = HERE / "progress.jsonl"
    record = {
        "event": str(event),
        "timestamp_epoch": time.time(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def system_root(system):
    return HERE / system


def data_root(system):
    return system_root(system) / "data"


def ordinary_row_root(system):
    return system_root(system) / "generated_ordinary_controls" / "row_cache_ordinary"


def tagged_row_root(system):
    return system_root(system) / "screen" / "row_cache_tagged"


def tagged_catalogue_root(system):
    return system_root(system) / "tagged_catalogue"


def target(atoms):
    if "energy" in atoms.info:
        energy = float(atoms.info["energy"])
    else:
        energy = float(atoms.get_potential_energy())
    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=np.float64)
    else:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    return energy, forces


def load_row(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "feature_sums": np.asarray(data["feature_sums"], dtype=np.float64),
            "feature_square_sums": np.asarray(
                data["feature_square_sums"], dtype=np.float64
            ),
            "force_design": np.asarray(data["force_design"], dtype=np.float64),
            "atom_count": int(data["atom_count"]),
        }


def combined_row(system, index, row_hasher):
    ordinary = load_row(ordinary_row_root(system) / f"frame_{index:04d}.npz")
    tagged = load_row(tagged_row_root(system) / f"frame_{index:04d}.npz")
    if ordinary["atom_count"] != tagged["atom_count"]:
        raise RuntimeError("Ordinary and tagged row atom counts disagree.")
    row = {
        "feature_sums": np.concatenate(
            (ordinary["feature_sums"], tagged["feature_sums"])
        ),
        "feature_square_sums": np.concatenate(
            (ordinary["feature_square_sums"], tagged["feature_square_sums"])
        ),
        "force_design": np.concatenate(
            (ordinary["force_design"], tagged["force_design"]), axis=1
        ),
        "atom_count": ordinary["atom_count"],
        "ordinary_feature_count": ordinary["feature_sums"].size,
        "tagged_feature_count": tagged["feature_sums"].size,
    }
    row_hasher.update(np.asarray([index, row["atom_count"]], dtype="<i8").tobytes())
    for name in ("feature_sums", "feature_square_sums", "force_design"):
        value = np.ascontiguousarray(row[name], dtype="<f8")
        row_hasher.update(np.asarray(value.shape, dtype="<i8").tobytes())
        row_hasher.update(value.tobytes())
    return row


def fold_assignments(structures, fold_count, seed):
    by_group = {}
    for index, atoms in enumerate(structures):
        group = str(atoms.info["config_type"])
        by_group.setdefault(group, []).append(index)
    assignments = {}
    for group_index, group in enumerate(sorted(by_group)):
        rng = np.random.default_rng(int(seed) + 7919 * group_index)
        indices = np.asarray(by_group[group], dtype=np.int64)
        rng.shuffle(indices)
        for offset, index in enumerate(indices.tolist()):
            assignments[int(index)] = int(offset % fold_count)
    return assignments


def reference_targets(structures, system, reference_path, output):
    energies = []
    forces = []
    for atoms in structures:
        energy, force = target(atoms)
        energies.append(energy)
        forces.append(force)
    if reference_path is None:
        return (
            np.asarray(energies, dtype=np.float64),
            tuple(forces),
            {"operation": "raw_ab_initio", "semantic_sha256": "none"},
        )
    raw = read_json(reference_path)
    if "semantic_sha256" in raw:
        config = raw
    else:
        config = lammps_zbl_reference_config(raw, (system,))
    cache_path = output / ("reference_" + config["semantic_sha256"] + ".npz")
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            reference_energy = np.asarray(data["energies"], dtype=np.float64)
            splits = np.asarray(data["force_splits"], dtype=np.int64)
            flat_force = np.asarray(data["forces"], dtype=np.float64)
        reference_force = tuple(
            flat_force[splits[index] : splits[index + 1]]
            for index in range(len(structures))
        )
        metadata = read_json(metadata_path)
    else:
        evaluated = evaluate_lammps_zbl_reference(structures, config)
        reference_energy = evaluated["reference_energies"]
        reference_force = evaluated["reference_forces"]
        metadata = evaluated["metadata"]
        splits = np.cumsum([0] + [len(value) for value in reference_force])
        np.savez_compressed(
            cache_path,
            energies=reference_energy,
            forces=np.concatenate(reference_force),
            force_splits=splits,
        )
        write_json(metadata_path, metadata)
    residual_energy = np.asarray(energies) - reference_energy
    residual_force = tuple(
        value - reference
        for value, reference in zip(forces, reference_force, strict=True)
    )
    return residual_energy, residual_force, metadata


def flatten_shards(shards):
    arrays = {}
    records = []
    for position, ((group, fold), statistics) in enumerate(sorted(shards.items())):
        prefix = f"shard_{position:03d}"
        for name in ARRAY_FIELDS:
            arrays[prefix + "_" + name] = np.asarray(statistics[name])
        records.append(
            {
                "prefix": prefix,
                "group": group,
                "fold": int(fold),
                "feature_count": int(statistics["feature_count"]),
                "structure_count": int(statistics["structure_count"]),
                "atom_count": int(statistics["atom_count"]),
                "force_component_count": int(statistics["force_component_count"]),
                **{name: float(statistics[name]) for name in SCALAR_FIELDS},
            }
        )
    return arrays, records


def unflatten_shards(cache):
    arrays = cache["arrays"]
    shards = {}
    for record in cache["metadata"]["shards"]:
        prefix = record["prefix"]
        statistics = {
            "schema": "ye3t_linear_sufficient_statistics_v1",
            "feature_count": int(record["feature_count"]),
            "structure_count": int(record["structure_count"]),
            "atom_count": int(record["atom_count"]),
            "force_component_count": int(record["force_component_count"]),
        }
        for name in ARRAY_FIELDS:
            statistics[name] = arrays[prefix + "_" + name]
        for name in SCALAR_FIELDS:
            statistics[name] = float(record[name])
        shards[(str(record["group"]), int(record["fold"]))] = statistics
    return shards


def build_statistics(system, fold_count, seed, reference_path):
    started = time.perf_counter()
    root = system_root(system)
    output = root / "statistics"
    output.mkdir(parents=True, exist_ok=True)
    prefix = system.lower()
    structures = load_xyz_structures(data_root(system) / f"{prefix}_training.xyz")
    assignments = fold_assignments(structures, fold_count, seed)
    energies, forces, reference = reference_targets(
        structures, system, reference_path, output
    )
    records = {}
    row_hasher = hashlib.sha256()
    target_hasher = hashlib.sha256()
    ordinary_count = None
    tagged_count = None
    for index, atoms in enumerate(structures):
        row = combined_row(system, index, row_hasher)
        if ordinary_count is None:
            ordinary_count = int(row["ordinary_feature_count"])
            tagged_count = int(row["tagged_feature_count"])
        if row["ordinary_feature_count"] != ordinary_count or row["tagged_feature_count"] != tagged_count:
            raise RuntimeError("Cached row feature counts change between structures.")
        if row["atom_count"] != len(atoms):
            raise RuntimeError("Cached row and dataset atom counts disagree.")
        energy_per_atom = float(energies[index]) / len(atoms)
        force = np.asarray(forces[index], dtype=np.float64)
        target_hasher.update(np.asarray([energy_per_atom], dtype="<f8").tobytes())
        target_hasher.update(np.ascontiguousarray(force, dtype="<f8").tobytes())
        group = str(atoms.info["config_type"])
        fold = int(assignments[index])
        record = structure_linear_statistics(
            row["feature_sums"],
            row["feature_square_sums"],
            row["force_design"],
            len(atoms),
            energy_per_atom,
            force,
        )
        records.setdefault((group, fold), []).append(record)
    shards = {
        key: sum_linear_statistics(values) for key, values in records.items()
    }
    arrays, shard_records = flatten_shards(shards)
    manifest = read_json(data_root(system) / "dataset_manifest.json")
    request = {
        "schema": "ye3t_mlearn_parent_statistics_request_v1",
        "system": system,
        "training_dataset_sha256": manifest["splits"]["training"]["converted_sha256"],
        "row_content_sha256": row_hasher.hexdigest(),
        "target_content_sha256": target_hasher.hexdigest(),
        "reference_semantic_sha256": str(reference["semantic_sha256"]),
        "fold_count": int(fold_count),
        "fold_seed": int(seed),
        "assignments": [int(assignments[index]) for index in range(len(structures))],
        "ordinary_feature_count": ordinary_count,
        "tagged_feature_count": tagged_count,
    }
    request_hash = semantic_hash(request)
    cache_hash = semantic_hash(
        {
            "request_hash": request_hash,
            "array_content_sha256": array_digest(arrays),
            "shards": shard_records,
        }
    )
    cache = {
        "schema": LINEAR_STATISTICS_CACHE_SCHEMA,
        "request_hash": request_hash,
        "cache_hash": cache_hash,
        "metadata": {
            "request": request,
            "reference": reference,
            "shards": shard_records,
        },
        "arrays": arrays,
    }
    directory = persist_linear_sufficient_statistics(cache, output)
    index = {
        "schema": "ye3t_mlearn_parent_statistics_index_v1",
        "system": system,
        "request_hash": request_hash,
        "cache_hash": cache_hash,
        "cache_directory": str(directory),
        "groups": sorted({group for group, _fold in shards}),
        "fold_count": fold_count,
        "ordinary_feature_count": ordinary_count,
        "tagged_feature_count": tagged_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "index.json", index)
    append_progress("statistics_complete", **index)
    print(json.dumps(index, sort_keys=True), flush=True)
    return index


def load_statistics(system):
    index = read_json(system_root(system) / "statistics" / "index.json")
    cache = load_linear_sufficient_statistics(
        index["request_hash"], system_root(system) / "statistics"
    )
    if cache["cache_hash"] != index["cache_hash"]:
        raise RuntimeError("Statistics index/cache hash mismatch.")
    return index, cache, unflatten_shards(cache)


def component_inventory(system):
    manifest = read_json(tagged_catalogue_root(system) / "catalogue_manifest.json")
    rows = []
    cursor = 0
    for component in manifest["components"]:
        if component["status"] != "passed":
            continue
        count = int(component["independent_feature_count"])
        if count <= 0:
            continue
        rows.append(
            {
                "component_index": int(component["component_index"]),
                "tensor_order_N": int(component["tensor_order_N"]),
                "content_pattern": list(component["content_pattern"]),
                "n": list(component["n"]),
                "l": list(component["l"]),
                "count": count,
                "start": cursor,
                "stop": cursor + count,
                "request_hash": str(component["request_hash"]),
            }
        )
        cursor += count
    return rows


def configured_descriptor_counts():
    config = read_json(CONFIG_PATH)
    counts = tuple(
        int(value) for value in config["basis"]["target_descriptor_counts"]
    )
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("basis.target_descriptor_counts must contain positive values.")
    return tuple(sorted(set(counts)))


def bounded_component_selection(components, maximum):
    selected = []
    count = 0
    priority = sorted(
        components,
        key=lambda row: (
            -int(row["tensor_order_N"]),
            -int(row["count"]),
            int(row["component_index"]),
        ),
    )
    for row in priority:
        width = int(row["count"])
        if count + width <= maximum:
            selected.append(row)
            count += width
    return tuple(sorted(selected, key=lambda row: row["component_index"]))


def feature_activity_policy():
    policy = read_json(CONFIG_PATH)["model"]["feature_activity_screen"]
    required = {
        "policy",
        "absolute_standard_deviation_floor",
        "relative_to_maximum_standard_deviation",
    }
    if set(policy) != required:
        raise ValueError(
            "model.feature_activity_screen keys disagree with the public schema: "
            f"missing={sorted(required - set(policy))}, "
            f"extra={sorted(set(policy) - required)}"
        )
    absolute = float(policy["absolute_standard_deviation_floor"])
    relative = float(policy["relative_to_maximum_standard_deviation"])
    if absolute <= 0.0 or relative <= 0.0:
        raise ValueError("Feature activity thresholds must be positive.")
    return {**policy, "absolute_standard_deviation_floor": absolute,
            "relative_to_maximum_standard_deviation": relative}


def tagged_feature_activity(statistics, ordinary_count, tagged_count):
    policy = feature_activity_policy()
    _mean, scale = feature_normalization(
        statistics, minimum_scale=np.finfo(np.float64).tiny
    )
    ordinary_count = int(ordinary_count)
    tagged_count = int(tagged_count)
    tagged_scale = np.asarray(
        scale[ordinary_count : ordinary_count + tagged_count], dtype=np.float64
    )
    if tagged_scale.shape != (tagged_count,) or not np.all(np.isfinite(tagged_scale)):
        raise ValueError("Tagged feature scales are incomplete or non-finite.")
    maximum = float(np.max(tagged_scale))
    threshold = max(
        float(policy["absolute_standard_deviation_floor"]),
        float(policy["relative_to_maximum_standard_deviation"]) * maximum,
    )
    active = np.flatnonzero(tagged_scale > threshold)
    inactive = np.flatnonzero(tagged_scale <= threshold)
    if active.size == 0:
        raise RuntimeError("The training-only activity screen rejected every tagged feature.")
    return {
        "policy": policy["policy"],
        "selection_uses_training_only": True,
        "absolute_standard_deviation_floor": policy[
            "absolute_standard_deviation_floor"
        ],
        "relative_to_maximum_standard_deviation": policy[
            "relative_to_maximum_standard_deviation"
        ],
        "resolved_standard_deviation_threshold": threshold,
        "maximum_tagged_standard_deviation": maximum,
        "active_tagged_indices": [int(value) for value in active],
        "inactive_tagged_indices": [int(value) for value in inactive],
        "inactive_tagged_standard_deviations": [
            float(tagged_scale[value]) for value in inactive
        ],
    }


def catalogue_models(index, components, target_counts=None, tagged_activity=None):
    ordinary_count = int(index["ordinary_feature_count"])
    tagged_count = int(index["tagged_feature_count"])
    if sum(row["count"] for row in components) != tagged_count:
        raise RuntimeError("Tagged component inventory does not cover cached coordinates.")
    if tagged_activity is None:
        active_tagged = tuple(range(tagged_count))
        inactive_tagged = ()
    else:
        active_tagged = tuple(
            sorted(set(int(value) for value in tagged_activity["active_tagged_indices"]))
        )
        inactive_tagged = tuple(
            sorted(set(int(value) for value in tagged_activity["inactive_tagged_indices"]))
        )
        if set(active_tagged) & set(inactive_tagged):
            raise ValueError("Active and inactive tagged feature sets overlap.")
        if set(active_tagged) | set(inactive_tagged) != set(range(tagged_count)):
            raise ValueError("Tagged activity report does not partition the catalogue.")
    active_set = set(active_tagged)
    active_components = []
    for component in components:
        indices = tuple(
            value
            for value in range(int(component["start"]), int(component["stop"]))
            if value in active_set
        )
        if not indices:
            continue
        active_component = dict(component)
        active_component["indices"] = list(indices)
        active_component["count"] = len(indices)
        active_components.append(active_component)
    models = []
    if target_counts is None:
        target_counts = configured_descriptor_counts()
    for requested in target_counts:
        resolved = min(int(requested), ordinary_count)
        models.append(
            {
                "name": f"ace_{resolved}",
                "family": "ace",
                "requested_feature_count": requested,
                "feature_count": resolved,
                "feature_indices": list(range(resolved)),
                "ordinary_feature_count": resolved,
                "tagged_feature_count": 0,
                "tagged_components": [],
            }
        )
        desired_tagged = min(
            len(active_tagged), 45 if requested <= 70 else len(active_tagged)
        )
        selected_components = bounded_component_selection(
            active_components, desired_tagged
        )
        selected_tagged = []
        for row in selected_components:
            selected_tagged.extend(row["indices"])
        selected_tagged = tuple(selected_tagged)
        selected_ordinary = resolved - len(selected_tagged)
        if selected_ordinary < 0:
            raise RuntimeError("Complete tagged components exceed matched feature count.")
        models.append(
            {
                "name": f"ye3t_tagged_{resolved}",
                "family": "tagged_ye3t",
                "requested_feature_count": requested,
                "feature_count": resolved,
                "feature_indices": list(range(selected_ordinary))
                + [ordinary_count + value for value in selected_tagged],
                "ordinary_feature_count": selected_ordinary,
                "tagged_feature_count": len(selected_tagged),
                "tagged_components": [
                    int(row["component_index"]) for row in selected_components
                ],
                "inactive_tagged_indices": list(inactive_tagged),
            }
        )
    augmented_target = min(127, ordinary_count) + tagged_count
    augmented_ordinary = min(
        ordinary_count, max(0, augmented_target - len(active_tagged))
    )
    models.append(
        {
            "name": f"ye3t_augmented_{augmented_ordinary + len(active_tagged)}",
            "family": "tagged_ye3t_augmented",
            "requested_feature_count": augmented_target,
            "feature_count": augmented_ordinary + len(active_tagged),
            "feature_indices": list(range(augmented_ordinary))
            + [ordinary_count + value for value in active_tagged],
            "ordinary_feature_count": augmented_ordinary,
            "tagged_feature_count": len(active_tagged),
            "tagged_components": [
                int(row["component_index"]) for row in active_components
            ],
            "inactive_tagged_indices": list(inactive_tagged),
        }
    )
    unique = {}
    for model in models:
        unique[model["name"]] = model
    return tuple(unique.values())


def model_shards(parent_shards, feature_indices):
    return {
        key: select_linear_statistics(value, feature_indices)
        for key, value in parent_shards.items()
    }


def latin_hypercube(count, dimensions, seed):
    rng = np.random.default_rng(seed)
    values = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        values[:, dimension] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    return values


def hyperparameter_search_config():
    config = read_json(CONFIG_PATH)["model"]["hyperparameter_search"]
    required = {
        "lhs_candidates",
        "maximum_ga_generations",
        "ga_population",
        "ga_elite",
        "ga_stale_generations",
        "log10_ridge_alpha_bounds",
        "log10_energy_weight_bounds",
        "log10_tagged_relative_penalty_bounds",
        "log10_group_weight_half_width",
    }
    if set(config) != required:
        raise ValueError(
            "model.hyperparameter_search keys disagree with the public schema: "
            f"missing={sorted(required - set(config))}, "
            f"extra={sorted(set(config) - required)}"
        )
    return config


def candidate_from_unit(unit, groups, has_tagged, search):
    unit = np.asarray(unit, dtype=np.float64)
    alpha_bounds = search["log10_ridge_alpha_bounds"]
    energy_bounds = search["log10_energy_weight_bounds"]
    tagged_bounds = search["log10_tagged_relative_penalty_bounds"]
    log_alpha = alpha_bounds[0] + (alpha_bounds[1] - alpha_bounds[0]) * unit[0]
    log_energy_weight = energy_bounds[0] + (
        energy_bounds[1] - energy_bounds[0]
    ) * unit[1]
    log_tagged_penalty = tagged_bounds[0] + (
        tagged_bounds[1] - tagged_bounds[0]
    ) * unit[2]
    half_width = float(search["log10_group_weight_half_width"])
    group_logs = -half_width + 2.0 * half_width * unit[3 : 3 + len(groups)]
    group_logs = group_logs - np.mean(group_logs)
    return {
        "alpha": float(10.0**log_alpha),
        "energy_weight": float(10.0**log_energy_weight),
        "force_weight": 1.0,
        "tagged_penalty": float(10.0**log_tagged_penalty) if has_tagged else 1.0,
        "group_weights": {
            group: float(10.0**value)
            for group, value in zip(groups, group_logs, strict=True)
        },
    }


def cross_validation_problem(shards, fold_count):
    groups = tuple(sorted({group for group, _fold in shards}))
    folds = []
    for fold in range(fold_count):
        training_groups = {}
        for group in groups:
            values = [
                shard
                for (shard_group, shard_fold), shard in shards.items()
                if shard_group == group and shard_fold != fold
            ]
            training_groups[group] = sum_linear_statistics(values)
        validation = sum_linear_statistics(
            shard
            for (_group, shard_fold), shard in shards.items()
            if shard_fold == fold
        )
        folds.append(
            {
                "prepared": prepare_weighted_normal_equations(training_groups),
                "validation": validation,
            }
        )
    return groups, tuple(folds)


def evaluate_candidate(candidate, folds, feature_penalty):
    metrics = []
    try:
        for fold in folds:
            equations = assemble_prepared_normal_equations(
                fold["prepared"],
                group_weights=candidate["group_weights"],
                energy_weight=candidate["energy_weight"],
                force_weight=candidate["force_weight"],
            )
            solution = solve_ridge_statistics(
                equations, candidate["alpha"], feature_penalty
            )
            if not np.all(np.isfinite(solution["runtime_coefficients"])):
                raise FloatingPointError("Ridge solution is non-finite.")
            metrics.append(
                score_linear_statistics(
                    fold["validation"], solution["runtime_coefficients"]
                )
            )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        return {
            "status": "rejected_numerical",
            "stop_reason": str(error),
            "objective": 1.0e300,
            "energy_rmse_mean_eV_per_atom": 1.0e150,
            "energy_rmse_std_eV_per_atom": 0.0,
            "force_rmse_mean_eV_per_A": 1.0e150,
            "force_rmse_std_eV_per_A": 0.0,
            "fold_metrics": metrics,
        }
    energy = np.asarray([row["energy_rmse_eV_per_atom"] for row in metrics])
    force = np.asarray([row["force_rmse_eV_per_A"] for row in metrics])
    objective = float(np.mean((energy / 0.01) ** 2 + (force / 0.1) ** 2))
    return {
        "status": "passed",
        "stop_reason": None,
        "objective": objective,
        "energy_rmse_mean_eV_per_atom": float(np.mean(energy)),
        "energy_rmse_std_eV_per_atom": float(np.std(energy, ddof=1)),
        "force_rmse_mean_eV_per_A": float(np.mean(force)),
        "force_rmse_std_eV_per_A": float(np.std(force, ddof=1)),
        "fold_metrics": metrics,
    }


def optimize_model(system, model, shards, fold_count, seed, output):
    started = time.perf_counter()
    selected = model_shards(shards, model["feature_indices"])
    groups, folds = cross_validation_problem(selected, fold_count)
    has_tagged = int(model["tagged_feature_count"]) > 0
    penalty = np.ones(int(model["feature_count"]), dtype=np.float64)
    if has_tagged:
        penalty[int(model["ordinary_feature_count"]) :] = 1.0
    dimensions = 3 + len(groups)
    search = hyperparameter_search_config()
    initial = latin_hypercube(int(search["lhs_candidates"]), dimensions, seed)
    archive = []
    trial_path = output / (model["name"] + "_trials.jsonl")
    if trial_path.exists():
        trial_path.unlink()

    def evaluate(unit, stage, generation):
        candidate = candidate_from_unit(unit, groups, has_tagged, search)
        if has_tagged:
            penalty[int(model["ordinary_feature_count"]) :] = candidate[
                "tagged_penalty"
            ]
        metric = evaluate_candidate(candidate, folds, penalty)
        record = {
            "stage": stage,
            "generation": int(generation),
            "unit": [float(value) for value in unit],
            **candidate,
            **metric,
        }
        archive.append(record)
        with trial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    for unit in initial:
        evaluate(unit, "lhs", 0)
    rng = np.random.default_rng(seed + 104729)
    best_objective = min(record["objective"] for record in archive)
    stale = 0
    for generation in range(1, int(search["maximum_ga_generations"]) + 1):
        elite = sorted(archive, key=lambda row: row["objective"])[
            : int(search["ga_elite"])
        ]
        candidates = []
        for _index in range(int(search["ga_population"])):
            left = np.asarray(elite[int(rng.integers(len(elite)))]["unit"])
            right = np.asarray(elite[int(rng.integers(len(elite)))]["unit"])
            mix = rng.random()
            child = mix * left + (1.0 - mix) * right
            child += rng.normal(scale=0.16 / np.sqrt(generation), size=dimensions)
            candidates.append(np.clip(child, 0.0, np.nextafter(1.0, 0.0)))
        for unit in candidates:
            evaluate(unit, "ga", generation)
        current = min(record["objective"] for record in archive)
        if current < best_objective * (1.0 - 1.0e-3):
            best_objective = current
            stale = 0
        else:
            stale += 1
        if stale >= int(search["ga_stale_generations"]):
            break
    best = min(archive, key=lambda row: row["objective"])
    full_groups = {
        group: sum_linear_statistics(
            shard
            for (shard_group, _fold), shard in selected.items()
            if shard_group == group
        )
        for group in groups
    }
    prepared = prepare_weighted_normal_equations(full_groups)
    equations = assemble_prepared_normal_equations(
        prepared,
        group_weights=best["group_weights"],
        energy_weight=best["energy_weight"],
        force_weight=best["force_weight"],
    )
    penalty = np.ones(int(model["feature_count"]), dtype=np.float64)
    if has_tagged:
        penalty[int(model["ordinary_feature_count"]) :] = best["tagged_penalty"]
    solution = solve_ridge_statistics(equations, best["alpha"], penalty)
    training = score_linear_statistics(
        sum_linear_statistics(selected.values()), solution["runtime_coefficients"]
    )
    model_path = output / (model["name"] + ".npz")
    np.savez_compressed(
        model_path,
        runtime_coefficients=solution["runtime_coefficients"],
        fit_coefficients=solution["fit_coefficients"],
        feature_indices=np.asarray(model["feature_indices"], dtype=np.int64),
        feature_penalty=penalty,
        feature_mean=equations["feature_mean"],
        feature_scale=equations["feature_scale"],
    )
    result = {
        **model,
        "system": system,
        "selected": {
            key: value for key, value in best.items() if key not in {"unit", "fold_metrics"}
        },
        "fold_metrics": best["fold_metrics"],
        "training_metrics": training,
        "trial_count": len(archive),
        "ga_generations": max(int(record["generation"]) for record in archive),
        "hyperparameter_search": search,
        "model_npz": str(model_path),
        "model_npz_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / (model["name"] + ".json"), result)
    print(
        json.dumps(
            {
                "system": system,
                "model": model["name"],
                "features": model["feature_count"],
                "energy_cv": best["energy_rmse_mean_eV_per_atom"],
                "force_cv": best["force_rmse_mean_eV_per_A"],
                "trials": len(archive),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def optimize_system(system, seed):
    started = time.perf_counter()
    index, _cache, shards = load_statistics(system)
    components = component_inventory(system)
    aggregate = sum_linear_statistics(shards.values())
    activity = tagged_feature_activity(
        aggregate,
        index["ordinary_feature_count"],
        index["tagged_feature_count"],
    )
    models = catalogue_models(index, components, tagged_activity=activity)
    output = system_root(system) / "optimized_fixed_catalogues"
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "catalogues.json",
        {"models": models, "components": components, "tagged_activity": activity},
    )
    results = []
    for position, model in enumerate(models):
        results.append(
            optimize_model(
                system,
                model,
                shards,
                int(index["fold_count"]),
                int(seed) + 1009 * position + sum(map(ord, system)),
                output,
            )
        )
    rows = []
    for result in results:
        selected = result["selected"]
        rows.append(
            {
                "system": system,
                "model": result["name"],
                "family": result["family"],
                "feature_count": result["feature_count"],
                "ordinary_feature_count": result["ordinary_feature_count"],
                "tagged_feature_count": result["tagged_feature_count"],
                "alpha": selected["alpha"],
                "tagged_penalty": selected["tagged_penalty"],
                "energy_weight": selected["energy_weight"],
                "cv_energy_rmse_eV_per_atom": selected[
                    "energy_rmse_mean_eV_per_atom"
                ],
                "cv_force_rmse_eV_per_A": selected["force_rmse_mean_eV_per_A"],
                "training_energy_rmse_eV_per_atom": result["training_metrics"][
                    "energy_rmse_eV_per_atom"
                ],
                "training_force_rmse_eV_per_A": result["training_metrics"][
                    "force_rmse_eV_per_A"
                ],
            }
        )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "ye3t_mlearn_fixed_catalogue_optimization_v1",
        "system": system,
        "statistics_request_hash": index["request_hash"],
        "test_split_opened": False,
        "models": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    append_progress("fixed_catalogue_optimization_complete", **summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("statistics", "optimize", "all"), default="all")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--reference-config", type=Path)
    args = parser.parse_args()
    if args.fold_count < 2:
        raise ValueError("--fold-count must be at least two.")
    for system in args.systems:
        if args.stage in {"statistics", "all"}:
            build_statistics(system, args.fold_count, args.seed, args.reference_config)
        if args.stage in {"optimize", "all"}:
            optimize_system(system, args.seed)


if __name__ == "__main__":
    main()
