#!/usr/bin/env python3
"""Screen eight matched radial realizations on training-only mlearn subsets."""

import argparse
import ctypes
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.neighborlist import neighbor_list

import optimize_cached as study
from ye3t.couplings import normalize_compact_label
from ye3t.couplings.tagged_cauchy import _stable_free_moment_matrix
from ye3t_methods.atomistic import YE3TDescriptors, YE3TRepresentation
from ye3t_methods.atomistic.ace.linear_ace import load_xyz_structures
from ye3t_methods.atomistic.equivariant_calc import (
    ACECovariantEvaluator,
    neighbor_data_from_ase_atoms,
)
from ye3t_methods.atomistic.equivariant_calc.gradients import (
    descriptor_sum_position_jacobian_analytic_product,
    edge_vectors_from_positions,
)
from ye3t_methods.atomistic.tagged_cauchy_fit import (
    build_streamed_merged_arm_from_catalogue,
)


try:
    _MALLOC_TRIM = ctypes.CDLL(None).malloc_trim
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int
except AttributeError:
    _MALLOC_TRIM = None


def release_row_memory():
    """Release per-structure evaluator temporaries between streamed rows."""

    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


def radial_candidates(seed):
    radial = study.read_json(study.CONFIG_PATH)["basis"]["radial"]
    optimization = radial["optimization"]
    candidate_count = int(optimization["candidate_count"])
    if candidate_count < 2:
        raise ValueError("Radial optimization requires at least two candidates.")
    cutoff_bounds = tuple(float(value) for value in optimization["cutoff_A_bounds"])
    lambda_bounds = tuple(
        float(value) for value in optimization["pace_lambda_bounds"]
    )
    if len(cutoff_bounds) != 2 or not cutoff_bounds[0] < cutoff_bounds[1]:
        raise ValueError("cutoff_A_bounds must be an increasing pair.")
    if len(lambda_bounds) != 2 or not lambda_bounds[0] < lambda_bounds[1]:
        raise ValueError("pace_lambda_bounds must be an increasing pair.")
    values = study.latin_hypercube(candidate_count - 1, 2, seed)
    candidates = [
        {
            "candidate": 0,
            "cutoff_A": 5.0,
            "radial_lambda": 0.5,
            "cutoff_width_A": 0.01,
            "role": "validated_baseline",
        }
    ]
    for index, value in enumerate(values, start=1):
        candidates.append(
            {
                "candidate": index,
                "cutoff_A": float(
                    cutoff_bounds[0]
                    + (cutoff_bounds[1] - cutoff_bounds[0]) * value[0]
                ),
                "radial_lambda": float(
                    lambda_bounds[0]
                    + (lambda_bounds[1] - lambda_bounds[0]) * value[1]
                ),
                "cutoff_width_A": 0.01,
                "role": "lhs",
            }
        )
    return tuple(candidates)


def structure_summary(atoms, maximum_cutoff):
    energy, forces = study.target(atoms)
    force_norm = np.linalg.norm(np.asarray(forces), axis=1)
    distances = neighbor_list("d", atoms, maximum_cutoff)
    coordination = len(distances) / max(len(atoms), 1)
    if len(distances):
        minimum_distance = float(np.min(distances))
    else:
        minimum_distance = maximum_cutoff
    volume = float(abs(np.linalg.det(np.asarray(atoms.cell.array))))
    return np.asarray(
        [
            energy / len(atoms),
            np.sqrt(np.mean(force_norm * force_norm)),
            np.max(force_norm),
            volume / len(atoms),
            minimum_distance,
            coordination,
        ],
        dtype=np.float64,
    )


def farthest_subset(values, count):
    values = np.asarray(values, dtype=np.float64)
    if count >= len(values):
        return tuple(range(len(values)))
    scale = np.maximum(np.std(values, axis=0), 1.0e-12)
    normalized = (values - np.mean(values, axis=0)) / scale
    selected = [int(np.argmax(np.sum(normalized * normalized, axis=1)))]
    distance = np.sum((normalized - normalized[selected[0]]) ** 2, axis=1)
    while len(selected) < count:
        candidate = int(np.argmax(distance))
        selected.append(candidate)
        candidate_distance = np.sum(
            (normalized - normalized[candidate]) ** 2, axis=1
        )
        distance = np.minimum(distance, candidate_distance)
        distance[selected] = -1.0
    return tuple(sorted(selected))


def representative_subset(structures, fraction, maximum_cutoff):
    summaries = np.stack(
        [structure_summary(atoms, maximum_cutoff) for atoms in structures]
    )
    by_group = {}
    for index, atoms in enumerate(structures):
        by_group.setdefault(str(atoms.info["config_type"]), []).append(index)
    selected = []
    for group in sorted(by_group):
        indices = by_group[group]
        count = max(1, int(round(fraction * len(indices))))
        local = farthest_subset(summaries[indices], count)
        selected.extend(indices[position] for position in local)
    return tuple(sorted(selected)), summaries


def build_ordinary_descriptor(system, cutoff, radial_lambda, cutoff_width):
    catalogue_root = study.system_root(system) / "generated_ordinary_controls"
    application_path = catalogue_root / "catalogue_application.json"
    if not application_path.is_file():
        application_path = catalogue_root / "catalogue_150.json"
    application = study.read_json(application_path)
    labels = tuple(
        normalize_compact_label(row["compact_label"])
        for row in application["rows"]
    )
    ranks = tuple(sorted({int(label.rank) for label in labels}))
    nmax = tuple(
        max(
            max(int(value) for value in label.n_tuple)
            for label in labels
            if int(label.rank) == rank
        )
        for rank in ranks
    )
    lmax = tuple(
        max(
            max(int(value) for value in label.l_tuple)
            for label in labels
            if int(label.rank) == rank
        )
        for rank in ranks
    )
    representation = YE3TRepresentation.ace(
        basis_mode=None,
        fast_path_policy="disable",
        metadata={
            "global_young_sector": "(N)",
            "basis_convention": "pace_complex_magnetic_y00_1",
        },
    )
    descriptor = YE3TDescriptors.ace(
        {
            "elements": [system],
            "type_map": {system: 0},
            "cutoff": cutoff,
            "ranks": ranks,
            "basis_type": "no_charge",
            "k_o_max": 0,
            "k_max": [0] * len(ranks),
            "nmax": nmax,
            "lmax": lmax,
            "lmin": [0] * len(ranks),
            "L_R": 0,
            "M_R_values": [0],
            "ordinary_scalar_catalogue": application,
            "factorized_descriptor_runtime_policy": "disable",
            "site_basis_config": {
                "rc": [cutoff],
                "lmbda": [radial_lambda],
                "nradmax": max(nmax),
                "lmax": max(lmax),
                "kmax": 0,
                "possible_types": [0],
                "radial_basis": "PACE_ChebExpCos",
                "chemical_basis": "delta",
                "charge_mode": "none",
                "atomic_base_normalization": "none",
                "factor_normalization": "none",
                "spherical_backend": "complex",
                "spherical_normalization": "pace_y00_one",
                "source_backend": "torch",
                "dtype": "float64",
                "pace_cutoff_width": [cutoff_width],
                "pace_spline_spacing": [0.001],
                "pace_inner_cutoff": [0.0],
                "pace_inner_cutoff_width": [0.0],
                "pace_crad_policy": "identity",
            },
            "representation": representation,
            "backend": "pytorch",
            "strict_backend": True,
            "validate_backend": True,
            "device": "cpu",
        }
    )
    evaluator = ACECovariantEvaluator(
        descriptor.site_basis_config,
        backend="pytorch",
        strict_backend=True,
        validate_backend=True,
        factorized_descriptor_runtime_policy="disable",
    )
    evaluator.precompile_descriptors(descriptor.descriptor_specs)
    return descriptor, evaluator


def load_image_matrix(catalogue, request_hash, descriptor_count):
    binary = catalogue / "binary64_image_matrices" / f"{request_hash}.npz"
    if binary.is_file():
        with np.load(binary, allow_pickle=False) as data:
            return np.asarray(data["matrix"], dtype=np.float64)
    image = study.read_json(
        catalogue / "free_moment_images" / f"{request_hash}.json"
    )
    matrix = _stable_free_moment_matrix(
        image, descriptor_count=descriptor_count
    )
    if matrix.size and np.max(np.abs(matrix.imag)) > 1.0e-12 * max(
        1.0, float(np.max(np.abs(matrix.real)))
    ):
        raise RuntimeError("Stable tagged image has a material imaginary residual.")
    return np.asarray(matrix.real, dtype=np.float64)


def build_tagged_arm(system):
    catalogue = study.tagged_catalogue_root(system)
    manifest_path = catalogue / "catalogue_manifest.json"
    manifest = study.read_json(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runtime_path = catalogue / "streamed_runtime_catalogue.json"
    runtime_record = None
    if runtime_path.is_file():
        candidate = study.read_json(runtime_path)
        if candidate.get("source_manifest_sha256") == manifest_sha256:
            runtime_record = candidate
    if runtime_record is not None:
        return build_streamed_merged_arm_from_catalogue(
            runtime_record,
            cache_dir=catalogue / "artifacts",
            fit_coordinates="pooled",
            catalogue_path=runtime_path,
        )

    contents = []
    features = []
    channel_by_key = {}
    feature_index = 0
    for row in manifest["components"]:
        if row["status"] != "passed" or int(row["independent_feature_count"]) == 0:
            continue
        request_hash = str(row["request_hash"])
        artifact = study.read_json(
            catalogue / "artifacts" / f"{request_hash}.json"
        )
        if str(artifact["self_hash"]) != str(row["artifact_self_hash"]):
            raise RuntimeError("Tagged component artifact hash does not match its manifest.")
        descriptor_count = len(artifact["payload"]["descriptors"])
        matrix = load_image_matrix(
            catalogue, request_hash, descriptor_count
        )
        contents.append(
            {
                "content_index": int(row["component_index"]),
                "request_hash": request_hash,
                "artifact_self_hash": str(row["artifact_self_hash"]),
                "request_payload": artifact["plan"]["report"]["request"],
                "supported_count": int(matrix.shape[0]),
            }
        )
        for local_row in matrix:
            combination = [
                [int(column), [float(local_row[column]), 0.0]]
                for column in np.flatnonzero(local_row)
            ]
            if not combination:
                raise RuntimeError("Tagged physical-image coordinate is identically zero.")
            features.append(
                {
                    "feature_index": int(feature_index),
                    "content_index": int(row["component_index"]),
                    "combination": combination,
                    "stratum": [
                        int(row["tensor_order_N"]),
                        "selected_physical_image",
                        "selected_physical_image",
                    ],
                }
            )
            feature_index += 1
        for channel in artifact["payload"]["channels"]:
            key = (
                str(channel["neighbor_species"]),
                int(channel["radial_channel"]),
                int(channel["l"]),
            )
            channel_by_key.setdefault(key, dict(channel))
    channels = []
    for key in sorted(channel_by_key):
        channel = dict(channel_by_key[key])
        channel["channel_index"] = len(channels)
        channel["channel_id"] = len(channels)
        channels.append(channel)
    if feature_index != int(manifest["total_independent_feature_count"]):
        raise RuntimeError("Streamed tagged feature count disagrees with the manifest.")
    runtime_record = {
        "schema": "ye3t_mlearn_streamed_runtime_catalogue_v1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "catalogue_hash": manifest_sha256,
        "spec": {
            "role_bindings": manifest["role_bindings"],
            "cache_dir": str(catalogue / "artifacts"),
        },
        "channels": channels,
        "contents": contents,
        "features": features,
    }
    study.write_json(runtime_path, runtime_record)
    return build_streamed_merged_arm_from_catalogue(
        runtime_record,
        cache_dir=catalogue / "artifacts",
        fit_coordinates="pooled",
        catalogue_path=runtime_path,
    )


def ordinary_row(descriptor, evaluator, atoms):
    positions = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64)
    neighbor = neighbor_data_from_ase_atoms(
        atoms, descriptor.cutoff, descriptor.type_map
    )
    shifts = torch.as_tensor(np.asarray(neighbor.shifts), dtype=torch.float64)
    cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float64)
    edge_index = torch.as_tensor(neighbor.edge_index, dtype=torch.long)
    atom_types = torch.as_tensor(neighbor.atom_types, dtype=torch.long)
    with torch.no_grad():
        site, jacobian = descriptor_sum_position_jacobian_analytic_product(
            evaluator,
            positions,
            cell,
            edge_index,
            atom_types,
            descriptor.descriptor_specs,
            shifts=shifts,
            real_if_scalar=True,
        )
    site = np.asarray(site.detach().cpu(), dtype=np.float64)
    jacobian = np.asarray(jacobian.detach().cpu(), dtype=np.float64)
    return {
        "feature_sums": np.sum(site, axis=0),
        "feature_square_sums": np.sum(site * site, axis=0),
        "force_design": -jacobian.T,
        "atom_count": len(atoms),
    }


def ordinary_value_row(descriptor, evaluator, atoms):
    positions = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64)
    neighbor = neighbor_data_from_ase_atoms(
        atoms, descriptor.cutoff, descriptor.type_map
    )
    shifts = torch.as_tensor(np.asarray(neighbor.shifts), dtype=torch.float64)
    cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float64)
    edge_index = torch.as_tensor(neighbor.edge_index, dtype=torch.long)
    atom_types = torch.as_tensor(neighbor.atom_types, dtype=torch.long)
    x_ij = edge_vectors_from_positions(
        positions, cell, edge_index, shifts=shifts
    )
    with torch.no_grad():
        site = evaluator(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=descriptor.descriptor_specs,
            charges=None,
            aux_tensor_basis=None,
            real_if_scalar=True,
        )
    site = np.asarray(site.detach().cpu(), dtype=np.float64)
    return {
        "feature_sums": np.sum(site, axis=0),
        "feature_square_sums": np.sum(site * site, axis=0),
        "atom_count": len(atoms),
    }


def tagged_row(arm, atoms, cutoff, radial_config, cache_dir):
    del cache_dir
    positions = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64)
    atom_types = torch.zeros(len(atoms), dtype=torch.long)
    cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float64)
    pbc = tuple(bool(value) for value in atoms.pbc)
    with torch.no_grad():
        feature_sum, jacobian = arm.descriptors_and_jacobian(
            positions,
            atom_types,
            cell,
            pbc,
            cutoff,
            radial_config,
            term_chunk_size=4096,
        )
        site = arm.descriptors(
            positions,
            atom_types,
            cell,
            pbc,
            cutoff,
            radial_config,
        )
    site = np.asarray(site.detach().cpu(), dtype=np.float64)
    return {
        "feature_sums": np.asarray(feature_sum.detach().cpu(), dtype=np.float64),
        "feature_square_sums": np.sum(site * site, axis=0),
        "force_design": -np.asarray(jacobian.detach().cpu(), dtype=np.float64).reshape(
            arm.feature_count, -1
        ).T,
        "atom_count": len(atoms),
    }


def tagged_value_row(arm, atoms, cutoff, radial_config):
    positions = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64)
    atom_types = torch.zeros(len(atoms), dtype=torch.long)
    cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float64)
    with torch.no_grad():
        site = arm.descriptors(
            positions,
            atom_types,
            cell,
            tuple(bool(value) for value in atoms.pbc),
            cutoff,
            radial_config,
        )
    site = np.asarray(site.detach().cpu(), dtype=np.float64)
    return {
        "feature_sums": np.sum(site, axis=0),
        "feature_square_sums": np.sum(site * site, axis=0),
        "atom_count": len(atoms),
    }


def load_value_row(path):
    with np.load(path, allow_pickle=False) as data:
        row = {
            "feature_sums": np.asarray(data["feature_sums"], dtype=np.float64),
            "feature_square_sums": np.asarray(
                data["feature_square_sums"], dtype=np.float64
            ),
            "atom_count": int(data["atom_count"]),
        }
        if "force_design" in data:
            row["force_design"] = np.asarray(
                data["force_design"], dtype=np.float64
            )
        return row


def cached_candidate_rows(
    system, candidate, structures, indices, output, derivatives, shared_arm
):
    cutoff = float(candidate["cutoff_A"])
    radial_config = {
        "lmbda": float(candidate["radial_lambda"]),
        "cutoff_width": float(candidate["cutoff_width_A"]),
    }
    descriptor = None
    evaluator = None
    arm = shared_arm
    ordinary_dir = output / "row_cache_ordinary"
    tagged_dir = output / "row_cache_tagged"
    ordinary_dir.mkdir(parents=True, exist_ok=True)
    tagged_dir.mkdir(parents=True, exist_ok=True)
    for position, index in enumerate(indices):
        ordinary_path = ordinary_dir / f"frame_{index:04d}.npz"
        tagged_path = tagged_dir / f"frame_{index:04d}.npz"
        if int(candidate["candidate"]) == 0:
            ordinary = study.load_row(
                study.ordinary_row_root(system) / f"frame_{index:04d}.npz"
            )
            tagged = study.load_row(
                study.tagged_row_root(system) / f"frame_{index:04d}.npz"
            )
            if not derivatives:
                ordinary.pop("force_design")
                tagged.pop("force_design")
        elif ordinary_path.is_file():
            ordinary = load_value_row(ordinary_path)
        else:
            if descriptor is None:
                descriptor, evaluator = build_ordinary_descriptor(
                    system,
                    cutoff,
                    radial_config["lmbda"],
                    radial_config["cutoff_width"],
                )
            if derivatives:
                ordinary = ordinary_row(
                    descriptor, evaluator, structures[index]
                )
            else:
                ordinary = ordinary_value_row(
                    descriptor, evaluator, structures[index]
                )
            np.savez_compressed(ordinary_path, **ordinary)
        if int(candidate["candidate"]) == 0:
            pass
        elif tagged_path.is_file():
            tagged = load_value_row(tagged_path)
        else:
            if arm is None:
                raise RuntimeError(
                    "A prebuilt tagged arm is required for nonbaseline radial candidates."
                )
            if derivatives:
                tagged = tagged_row(
                    arm,
                    structures[index],
                    cutoff,
                    radial_config,
                    tagged_dir / "structure_rows",
                )
            else:
                tagged = tagged_value_row(
                    arm, structures[index], cutoff, radial_config
                )
            np.savez_compressed(tagged_path, **tagged)
        combined = {
            "feature_sums": np.concatenate(
                (ordinary["feature_sums"], tagged["feature_sums"])
            ),
            "feature_square_sums": np.concatenate(
                (ordinary["feature_square_sums"], tagged["feature_square_sums"])
            ),
            "atom_count": len(structures[index]),
        }
        if derivatives:
            combined["force_design"] = np.concatenate(
                (ordinary["force_design"], tagged["force_design"]), axis=1
            )
        if position == 0 or (position + 1) % 10 == 0 or position + 1 == len(indices):
            print(
                f"{system} radial {candidate['candidate']} rows {position + 1}/{len(indices)}",
                flush=True,
            )
        yield index, combined
        del combined, ordinary, tagged
        release_row_memory()


def candidate_statistics(
    structures, indices, rows, fold_count, seed, derivatives
):
    subset = [structures[index] for index in indices]
    local_folds = study.fold_assignments(subset, fold_count, seed)
    records = {}
    for local, (index, row) in enumerate(rows):
        if int(index) != int(indices[local]):
            raise RuntimeError("Streamed radial rows changed the subset order.")
        atoms = structures[index]
        energy, forces = study.target(atoms)
        if derivatives:
            force_design = row["force_design"]
            force_target = forces
        else:
            force_design = np.zeros(
                (3 * len(atoms), row["feature_sums"].size), dtype=np.float64
            )
            force_target = np.zeros((len(atoms), 3), dtype=np.float64)
        record = study.structure_linear_statistics(
            row["feature_sums"],
            row["feature_square_sums"],
            force_design,
            len(atoms),
            energy / len(atoms),
            force_target,
        )
        key = (str(atoms.info["config_type"]), int(local_folds[local]))
        if key in records:
            records[key] = study.sum_linear_statistics((records[key], record))
        else:
            records[key] = record
        del row, force_design, force_target, record
    release_row_memory()
    return records


def fixed_candidate_metric(system, model_name, model, shards, fold_count):
    selection = study.read_json(
        study.system_root(system)
        / "optimized_fixed_catalogues"
        / (model_name + ".json")
    )["selected"]
    selected = study.model_shards(shards, model["feature_indices"])
    _groups, folds = study.cross_validation_problem(selected, fold_count)
    penalty = np.ones(int(model["feature_count"]), dtype=np.float64)
    if int(model["tagged_feature_count"]):
        penalty[int(model["ordinary_feature_count"]) :] = float(
            selection["tagged_penalty"]
        )
    candidate = {
        "alpha": float(selection["alpha"]),
        "energy_weight": float(selection["energy_weight"]),
        "force_weight": float(selection["force_weight"]),
        "tagged_penalty": float(selection["tagged_penalty"]),
        "group_weights": dict(selection["group_weights"]),
    }
    return study.evaluate_candidate(candidate, folds, penalty)


def optimized_candidate_metric(
    system, model, shards, fold_count, seed, output
):
    """Optimize one fixed catalogue on one radial realization.

    Radial realizations must receive the same hyperparameter-search budget;
    carrying the baseline realization's optimum into every arm can otherwise
    confound radial quality with a mismatched ridge/weight choice.
    """
    output.mkdir(parents=True, exist_ok=True)
    result = study.optimize_model(
        system,
        model,
        shards,
        fold_count,
        seed,
        output,
    )
    selected = result["selected"]
    return {
        "status": selected["status"],
        "stop_reason": selected["stop_reason"],
        "objective": selected["objective"],
        "energy_rmse_mean_eV_per_atom": selected[
            "energy_rmse_mean_eV_per_atom"
        ],
        "energy_rmse_std_eV_per_atom": selected[
            "energy_rmse_std_eV_per_atom"
        ],
        "force_rmse_mean_eV_per_A": selected["force_rmse_mean_eV_per_A"],
        "force_rmse_std_eV_per_A": selected["force_rmse_std_eV_per_A"],
        "fold_metrics": result["fold_metrics"],
        "selected_hyperparameters": {
            key: value
            for key, value in selected.items()
            if key
            not in {
                "status",
                "stop_reason",
                "objective",
                "energy_rmse_mean_eV_per_atom",
                "energy_rmse_std_eV_per_atom",
                "force_rmse_mean_eV_per_A",
                "force_rmse_std_eV_per_A",
            }
        },
        "trial_count": int(result["trial_count"]),
        "ga_generations": int(result["ga_generations"]),
        "fit_json": str(output / (model["name"] + ".json")),
        "fit_npz": str(output / (model["name"] + ".npz")),
    }


def screen_system(
    system, candidate_ids, subset_fraction, fold_count, seed, mode
):
    started = time.perf_counter()
    root = study.system_root(system)
    output = root / "radial_screen"
    output.mkdir(parents=True, exist_ok=True)
    structures = load_xyz_structures(
        study.data_root(system) / f"{system.lower()}_training.xyz"
    )
    subset_path = output / "subset.json"
    if subset_path.is_file():
        subset_record = study.read_json(subset_path)
        indices = tuple(int(value) for value in subset_record["indices"])
    else:
        indices, summaries = representative_subset(
            structures, subset_fraction, 5.3
        )
        subset_record = {
            "schema": "ye3t_mlearn_radial_subset_v1",
            "system": system,
            "fraction": subset_fraction,
            "indices": list(indices),
            "structure_count": len(indices),
            "summary_columns": [
                "energy_eV_per_atom",
                "force_rms_eV_per_A",
                "maximum_force_eV_per_A",
                "volume_A3_per_atom",
                "minimum_distance_A",
                "coordination_at_5p3_A",
            ],
            "selected_summaries": summaries[np.asarray(indices)].tolist(),
            "test_split_used": False,
        }
        study.write_json(subset_path, subset_record)
    parent_index, _cache, _parent_shards = study.load_statistics(system)
    components = study.component_inventory(system)
    models = {
        row["name"]: row for row in study.catalogue_models(parent_index, components)
    }
    candidates = radial_candidates(seed)
    local_folds = study.fold_assignments(
        [structures[index] for index in indices], fold_count, seed
    )
    fold_vector = np.asarray(
        [local_folds[index] for index in range(len(indices))], dtype="<i8"
    )
    fold_identity = {
        "fold_count": int(fold_count),
        "fold_seed": int(seed),
        "assignment_sha256": hashlib.sha256(fold_vector.tobytes()).hexdigest(),
    }
    selected_candidates = [
        row for row in candidates if int(row["candidate"]) in candidate_ids
    ]
    requires_tagged_arm = any(
        int(row["candidate"]) != 0
        and any(
            not (
                output
                / f"{mode}_candidate_{int(row['candidate']):02d}"
                / "row_cache_tagged"
                / f"frame_{index:04d}.npz"
            ).is_file()
            for index in indices
        )
        for row in selected_candidates
    )
    shared_arm = None
    if requires_tagged_arm:
        arm_started = time.perf_counter()
        print(f"{system} building shared tagged arm", flush=True)
        shared_arm = build_tagged_arm(system)
        print(
            f"{system} shared tagged arm ready in "
            f"{time.perf_counter() - arm_started:.3f} s",
            flush=True,
        )
    results = []
    derivatives = mode == "force"
    for candidate in selected_candidates:
        candidate_output = output / f"{mode}_candidate_{int(candidate['candidate']):02d}"
        candidate_output.mkdir(parents=True, exist_ok=True)
        candidate_started = time.perf_counter()
        rows = cached_candidate_rows(
            system,
            candidate,
            structures,
            indices,
            candidate_output,
            derivatives,
            shared_arm,
        )
        shards = candidate_statistics(
            structures,
            indices,
            rows,
            fold_count,
            seed,
            derivatives,
        )
        metrics = {}
        for model_name in ("ace_127", "ye3t_tagged_127"):
            if mode == "force":
                metrics[model_name] = optimized_candidate_metric(
                    system,
                    models[model_name],
                    shards,
                    fold_count,
                    seed + 10007 * int(candidate["candidate"]),
                    candidate_output / "optimization",
                )
            else:
                # Energy-only screening is a cheap diagnostic and is never
                # eligible to freeze the promoted radial realization.
                metrics[model_name] = fixed_candidate_metric(
                    system, model_name, models[model_name], shards, fold_count
                )
        record = {
            **candidate,
            "system": system,
            "subset_structure_count": len(indices),
            "evaluation_mode": mode,
            "fold_identity": fold_identity,
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - candidate_started,
        }
        study.write_json(candidate_output / "result.json", record)
        results.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    available = []
    for candidate in candidates:
        path = (
            output
            / f"{mode}_candidate_{int(candidate['candidate']):02d}"
            / "result.json"
        )
        if path.is_file():
            available.append(study.read_json(path))
    summary = {
        "schema": "ye3t_mlearn_radial_screen_v1",
        "system": system,
        "subset": subset_record,
        "candidates": list(candidates),
        "results": available,
        "test_split_used": False,
        "evaluation_mode": mode,
        "fold_identity": fold_identity,
        "elapsed_seconds_this_call": time.perf_counter() - started,
    }
    study.write_json(output / f"{mode}_summary.json", summary)
    study.append_progress(
        "radial_screen_complete",
        system=system,
        evaluation_mode=mode,
        fold_identity=fold_identity,
        completed_candidates=[int(row["candidate"]) for row in available],
        elapsed_seconds=summary["elapsed_seconds_this_call"],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS)
    configured = study.read_json(study.CONFIG_PATH)["basis"]["radial"][
        "optimization"
    ]
    candidate_count = int(configured["candidate_count"])
    parser.add_argument(
        "--candidates", nargs="+", type=int, default=tuple(range(candidate_count))
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=float(configured["representative_training_fraction"]),
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=int(configured["candidate_seed"]))
    parser.add_argument("--mode", choices=("energy", "force"), default="energy")
    args = parser.parse_args()
    if not 0.0 < args.subset_fraction <= 1.0:
        raise ValueError("--subset-fraction must be in (0, 1].")
    if any(value < 0 or value >= candidate_count for value in args.candidates):
        raise ValueError(
            f"--candidates must contain values from zero through {candidate_count - 1}."
        )
    for system in args.systems:
        screen_system(
            system,
            set(args.candidates),
            args.subset_fraction,
            args.fold_count,
            args.seed,
            args.mode,
        )


if __name__ == "__main__":
    main()
