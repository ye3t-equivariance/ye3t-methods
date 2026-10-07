#!/usr/bin/env python3
"""Materialize reusable ordinary and stable-image tagged training rows."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ye3t.couplings import normalize_compact_label
from ye3t.couplings.lifted_cauchy_scalar import CompiledLiftedCauchyScalar
from ye3t.couplings.tagged_cauchy import _stable_free_moment_matrix
from ye3t_methods.atomistic import YE3TDescriptors, YE3TRepresentation
from ye3t_methods.atomistic.ace.linear_ace import load_xyz_structures
from ye3t_methods.atomistic.equivariant_calc import ACECovariantEvaluator, neighbor_data_from_ase_atoms
from ye3t_methods.atomistic.equivariant_calc.gradients import descriptor_sum_position_jacobian_analytic_product
from ye3t_methods.atomistic.lifted_cauchy_linear import _artifact_channels
from ye3t_methods.atomistic.tagged_cauchy_fit import TaggedArmEvaluator, structure_row


HERE = Path(__file__).resolve().parent
PUBLIC = Path(os.environ.get("YE3T_COST_PUBLIC_ROOT", str(HERE.parent))).resolve()
CONFIG_PATH = Path(
    os.environ.get("YE3T_COST_CONFIG", str(PUBLIC / "config.json"))
).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
WORKFLOW_ROOT = Path(
    os.environ.get(
        "YE3T_COST_WORKFLOW_ROOT",
        str(CONFIG["runtime"]["workflow_root"]),
    )
)
if not WORKFLOW_ROOT.is_absolute():
    WORKFLOW_ROOT = (CONFIG_PATH.parent / WORKFLOW_ROOT).resolve()
CATALOGUE_ROOT = WORKFLOW_ROOT / "Si" / "tagged_catalogue"
DATA = WORKFLOW_ROOT / "Si" / "data"
OUTPUT = WORKFLOW_ROOT / "Si" / "screen"
ORDINARY_APPLICATION = (
    WORKFLOW_ROOT
    / "Si"
    / "generated_ordinary_controls"
    / "catalogue_application.json"
)
CUTOFF = float(CONFIG["basis"]["radial"]["matched_control_cutoff_A"])
RADIAL_CONFIG = {
    "lmbda": float(CONFIG["basis"]["radial"]["pace_lambda"]),
    "cutoff_width": float(CONFIG["basis"]["radial"]["pace_cutoff_width_A"]),
}
SEED = int(CONFIG["validation"]["inner_seed"])
SYSTEM = "Si"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_matrix(image_path, matrix_path, descriptor_count):
    if matrix_path.is_file():
        with np.load(matrix_path, allow_pickle=False) as data:
            return data["matrix"], int(data["supported_descriptor_count"])
    image = read_json(image_path)
    matrix = _stable_free_moment_matrix(image, descriptor_count=descriptor_count)
    if matrix.size and np.max(np.abs(matrix.imag)) > 1.0e-12 * max(1.0, float(np.max(np.abs(matrix.real)))):
        raise RuntimeError("Stable tagged image has a material imaginary residual.")
    matrix = np.asarray(matrix.real, dtype=np.float64)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_path,
        matrix=matrix,
        supported_descriptor_count=np.asarray(image["supported_descriptor_count"]),
    )
    return matrix, int(image["supported_descriptor_count"])


def build_tagged_arm():
    manifest = read_json(CATALOGUE_ROOT / "catalogue_manifest.json")
    contents = []
    component_rows = []
    channel_by_key = {}
    for row in manifest["components"]:
        if row["status"] != "passed" or int(row["independent_feature_count"]) == 0:
            continue
        request_hash = row["request_hash"]
        compiled = CompiledLiftedCauchyScalar.from_dict(
            read_json(CATALOGUE_ROOT / "artifacts" / f"{request_hash}.json")
        )
        matrix, supported = load_matrix(
            CATALOGUE_ROOT / "free_moment_images" / f"{request_hash}.json",
            CATALOGUE_ROOT / "binary64_image_matrices" / f"{request_hash}.npz",
            len(compiled.payload["descriptors"]),
        )
        contents.append(
            {
                "content_id": f"component_{row['component_index']}",
                "compiled": compiled,
                "combination_matrix_override": matrix,
                "supported_count": supported,
                "catalogue_content_index": int(row["component_index"]),
                "tensor_order_N": int(row["tensor_order_N"]),
            }
        )
        component_rows.append(
            {
                "component_index": int(row["component_index"]),
                "tensor_order_N": int(row["tensor_order_N"]),
                "independent_feature_count": int(matrix.shape[0]),
            }
        )
        for channel in _artifact_channels(compiled):
            key = (str(channel["neighbor_species"]), int(channel["radial_channel"]), int(channel["l"]))
            channel_by_key.setdefault(key, dict(channel))
        print(
            f"loaded component {row['component_index']} N={row['tensor_order_N']} features={matrix.shape[0]}",
            flush=True,
        )
    channels = []
    for key in sorted(channel_by_key):
        channel = dict(channel_by_key[key])
        channel["channel_index"] = len(channels)
        channel["channel_id"] = len(channels)
        channels.append(channel)
    role_bindings = tuple(tuple(value) for value in manifest["role_bindings"])
    arm = TaggedArmEvaluator(
        role_bindings,
        contents,
        channels,
        use_pooled_basis=True,
        fit_coordinates="pooled",
        use_merged_real_program=True,
    )
    cursor = 0
    for record, entry in zip(component_rows, arm.contents, strict=True):
        record["feature_start"] = cursor
        cursor += int(entry["descriptor_count"])
        record["feature_stop"] = cursor
    expected = int(manifest["total_independent_feature_count"])
    if arm.feature_count != expected:
        raise RuntimeError(
            f"Tagged arm has {arm.feature_count} features; manifest declares {expected}."
        )
    return arm, component_rows


def ordinary_descriptor():
    catalogue = read_json(ORDINARY_APPLICATION)
    labels = [normalize_compact_label(row["compact_label"]) for row in catalogue["rows"]]
    ranks = sorted({int(label.rank) for label in labels})
    nmax = [max(max(int(v) for v in label.n_tuple) for label in labels if label.rank == rank) for rank in ranks]
    lmax = [max(max(int(v) for v in label.l_tuple) for label in labels if label.rank == rank) for rank in ranks]
    representation = YE3TRepresentation.ace(
        basis_mode=None,
        fast_path_policy="auto",
        metadata={"global_young_sector": "(N)", "basis_convention": "pace_complex_magnetic_y00_1"},
    )
    descriptor = YE3TDescriptors.ace(
        {
            "elements": [SYSTEM],
            "type_map": {SYSTEM: 0},
            "cutoff": CUTOFF,
            "ranks": ranks,
            "basis_type": "no_charge",
            "k_o_max": 0,
            "k_max": [0] * len(ranks),
            "nmax": nmax,
            "lmax": lmax,
            "lmin": [0] * len(ranks),
            "L_R": 0,
            "M_R_values": [0],
            "ordinary_scalar_catalogue": catalogue,
            "factorized_descriptor_runtime_policy": "auto",
            "site_basis_config": {
                "rc": [CUTOFF],
                "lmbda": [RADIAL_CONFIG["lmbda"]],
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
                "pace_cutoff_width": [RADIAL_CONFIG["cutoff_width"]],
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
    full_ids = [str(row["feature_id"]) for row in catalogue["rows"]]
    compact_indices = tuple(range(min(60, len(full_ids))))
    return descriptor, compact_indices, full_ids


def target(atoms):
    energy = float(atoms.info["energy"]) if "energy" in atoms.info else float(atoms.get_potential_energy())
    forces = np.asarray(atoms.arrays.get("forces", atoms.get_forces()), dtype=np.float64)
    return energy, forces


def ordinary_row(descriptor, evaluator, atoms):
    positions = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64)
    neighbor = neighbor_data_from_ase_atoms(atoms, descriptor.cutoff, descriptor.type_map)
    shifts = torch.as_tensor(np.asarray(neighbor.shifts), dtype=torch.float64)
    cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float64)
    edge_index = torch.as_tensor(neighbor.edge_index, dtype=torch.long)
    atom_types = torch.as_tensor(neighbor.atom_types, dtype=torch.long)
    with torch.no_grad():
        site, jacobian = descriptor_sum_position_jacobian_analytic_product(
            evaluator, positions, cell, edge_index, atom_types,
            descriptor.descriptor_specs, shifts=shifts, real_if_scalar=True,
        )
    site = np.asarray(site.detach().cpu(), dtype=np.float64)
    jacobian = np.asarray(jacobian.detach().cpu(), dtype=np.float64)
    return {
        "feature_sums": np.sum(site, axis=0),
        "feature_square_sums": np.sum(site * site, axis=0),
        "force_design": -jacobian.T,
        "atom_count": len(atoms),
    }


def cached_rows(kind, evaluator, structures, directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = {}
    for index, atoms in enumerate(structures):
        path = directory / f"frame_{index:04d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                row = {key: np.asarray(data[key]) for key in ("feature_sums", "feature_square_sums", "force_design")}
                row["atom_count"] = int(data["atom_count"])
        elif kind == "ordinary":
            row = ordinary_row(evaluator[0], evaluator[1], atoms)
            np.savez_compressed(path, **row)
        else:
            raw = structure_row(
                evaluator,
                atoms,
                cutoff=CUTOFF,
                radial_config=RADIAL_CONFIG,
                cache_dir=directory / "structure_rows",
                jacobian_mode="explicit",
            )
            positions = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64)
            atom_types = torch.zeros(len(atoms), dtype=torch.long)
            cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float64)
            with torch.no_grad():
                site = evaluator.descriptors(
                    positions, atom_types, cell, tuple(bool(v) for v in atoms.pbc), CUTOFF, RADIAL_CONFIG
                )
            site = np.asarray(site.detach().cpu(), dtype=np.float64)
            row = {
                "feature_sums": np.asarray(raw["feature_sum"], dtype=np.float64),
                "feature_square_sums": np.sum(site * site, axis=0),
                "force_design": -np.asarray(raw["jacobian"], dtype=np.float64).reshape(evaluator.feature_count, -1).T,
                "atom_count": len(atoms),
            }
            np.savez_compressed(path, **row)
        rows[index] = row
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(structures):
            print(f"{kind} rows {index + 1}/{len(structures)}", flush=True)
    return rows


def select(row, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return {
        "feature_sums": row["feature_sums"][indices],
        "feature_square_sums": row["feature_square_sums"][indices],
        "force_design": row["force_design"][:, indices],
        "atom_count": row["atom_count"],
    }


def statistics(rows, indices):
    atoms = sum(rows[index]["atom_count"] for index in indices)
    mean = np.sum([rows[index]["feature_sums"] for index in indices], axis=0) / atoms
    square = np.sum([rows[index]["feature_square_sums"] for index in indices], axis=0) / atoms
    scale = np.maximum(np.sqrt(np.maximum(square - mean * mean, 0.0)), 1.0e-12)
    return mean, scale


def build_design(blocks, indices, structures, energies, forces):
    stats = [statistics(rows, indices) for _name, rows in blocks]
    energy_values = np.asarray([energies[i] / len(structures[i]) for i in indices])
    force_values = np.concatenate([forces[i].reshape(-1) for i in indices])
    energy_scale = max(float(np.std(energy_values)), 1.0e-12)
    force_scale = max(float(np.sqrt(np.mean(force_values * force_values))), 1.0e-12)
    matrix = []
    values = []
    for index in indices:
        atom_count = len(structures[index])
        energy_columns = [np.asarray([1.0])]
        force_columns = [np.zeros((3 * atom_count, 1))]
        for (_name, rows), (mean, scale) in zip(blocks, stats, strict=True):
            energy_columns.append((rows[index]["feature_sums"] / atom_count - mean) / scale)
            force_columns.append(rows[index]["force_design"] / scale)
        ef = 1.0 / (np.sqrt(len(indices)) * energy_scale)
        ff = 1.0 / (np.sqrt(len(indices) * 3 * atom_count) * force_scale)
        matrix.append(ef * np.concatenate(energy_columns)[None, :])
        values.append(np.asarray([ef * energies[index] / atom_count]))
        matrix.append(ff * np.concatenate(force_columns, axis=1))
        values.append(ff * forces[index].reshape(-1))
    return np.concatenate(matrix), np.concatenate(values), stats


def runtime_coefficients(beta, stats):
    weights = []
    shift = 0.0
    cursor = 1
    for mean, scale in stats:
        width = mean.size
        weight = beta[cursor : cursor + width] / scale
        weights.append(weight)
        shift += float(mean @ weight)
        cursor += width
    return float(beta[0] - shift), weights


def score(bias, weights, blocks, indices, structures, energies, forces):
    energy_error = []
    force_error = []
    groups = {}
    for index in indices:
        ep = bias * len(structures[index])
        fp = np.zeros(3 * len(structures[index]))
        for (_name, rows), weight in zip(blocks, weights, strict=True):
            ep += float(rows[index]["feature_sums"] @ weight)
            fp += rows[index]["force_design"] @ weight
        ee = ep / len(structures[index]) - energies[index] / len(structures[index])
        fe = fp - forces[index].reshape(-1)
        energy_error.append(ee)
        force_error.append(fe)
        group = structures[index].info["config_type"]
        record = groups.setdefault(group, {"energy": [], "force": []})
        record["energy"].append(ee)
        record["force"].append(fe)
    force_error = np.concatenate(force_error)
    return {
        "energy_rmse_eV_per_atom": float(np.sqrt(np.mean(np.square(energy_error)))),
        "force_rmse_eV_per_A": float(np.sqrt(np.mean(force_error * force_error))),
        "groups": {
            group: {
                "structures": len(values["energy"]),
                "energy_rmse_eV_per_atom": float(np.sqrt(np.mean(np.square(values["energy"])))),
                "force_rmse_eV_per_A": float(
                    np.sqrt(np.mean(np.concatenate(values["force"]) ** 2))
                ),
            }
            for group, values in sorted(groups.items())
        },
    }


def fit_arm(name, blocks, fit_indices, score_indices, structures, energies, forces, alpha_grid, penalty_grid):
    matrix, values, stats = build_design(blocks, fit_indices, structures, energies, forces)
    widths = [blocks[i][1][fit_indices[0]]["feature_sums"].size for i in range(len(blocks))]
    best = None
    trials = []
    for alpha in alpha_grid:
        for tagged_penalty in penalty_grid:
            penalty = np.ones(matrix.shape[1] - 1)
            cursor = 0
            for (label, _rows), width in zip(blocks, widths, strict=True):
                if label.startswith("tagged"):
                    penalty[cursor : cursor + width] = tagged_penalty
                cursor += width
            if alpha:
                regularizer = np.zeros((penalty.size, matrix.shape[1]))
                regularizer[:, 1:] = np.diag(np.sqrt(penalty))
                x = np.concatenate((matrix, np.sqrt(alpha) * regularizer), axis=0)
                y = np.concatenate((values, np.zeros(penalty.size)))
            else:
                x, y = matrix, values
            beta = np.linalg.lstsq(x, y, rcond=1.0e-12)[0]
            bias, weights = runtime_coefficients(beta, stats)
            metric = score(bias, weights, blocks, score_indices, structures, energies, forces)
            objective = (metric["energy_rmse_eV_per_atom"] / 0.01) ** 2 + (metric["force_rmse_eV_per_A"] / 0.1) ** 2
            trial = {
                "arm": name,
                "alpha": float(alpha),
                "tagged_penalty": float(tagged_penalty),
                "feature_count": int(matrix.shape[1] - 1),
                "selection_metric": float(objective),
                "validation_energy_rmse_eV_per_atom": metric["energy_rmse_eV_per_atom"],
                "validation_force_rmse_eV_per_A": metric["force_rmse_eV_per_A"],
            }
            trials.append(trial)
            if best is None or objective < best[0]:
                best = (objective, trial, beta)
    selected = best[1]
    final_matrix, final_values, final_stats = build_design(blocks, fit_indices, structures, energies, forces)
    penalty = np.ones(final_matrix.shape[1] - 1)
    cursor = 0
    for (label, _rows), width in zip(blocks, widths, strict=True):
        if label.startswith("tagged"):
            penalty[cursor : cursor + width] = selected["tagged_penalty"]
        cursor += width
    alpha = selected["alpha"]
    if alpha:
        regularizer = np.zeros((penalty.size, final_matrix.shape[1]))
        regularizer[:, 1:] = np.diag(np.sqrt(penalty))
        final_matrix = np.concatenate((final_matrix, np.sqrt(alpha) * regularizer), axis=0)
        final_values = np.concatenate((final_values, np.zeros(penalty.size)))
    beta = np.linalg.lstsq(final_matrix, final_values, rcond=1.0e-12)[0]
    bias, weights = runtime_coefficients(beta, final_stats)
    return selected, bias, weights, trials


def stratified_validation(structures, fraction=0.2):
    rng = np.random.default_rng(SEED)
    by_group = {}
    for index, atoms in enumerate(structures):
        by_group.setdefault(atoms.info["config_type"], []).append(index)
    validation = []
    for indices in by_group.values():
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        validation.extend(shuffled[: max(1, int(round(fraction * len(shuffled))))].tolist())
    validation = tuple(sorted(validation))
    training = tuple(index for index in range(len(structures)) if index not in set(validation))
    return training, validation


def main():
    global CATALOGUE_ROOT, DATA, OUTPUT, ORDINARY_APPLICATION, CUTOFF, SEED, SYSTEM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE_ROOT)
    parser.add_argument(
        "--ordinary-application", type=Path, default=ORDINARY_APPLICATION
    )
    parser.add_argument("--system", default=SYSTEM)
    parser.add_argument("--cutoff", type=float, default=CUTOFF)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--stage", choices=("rows", "screen"), default="rows"
    )
    parser.add_argument(
        "--kind", choices=("tagged", "ordinary", "both"), default="tagged"
    )
    parser.add_argument(
        "--include-test-rows",
        action="store_true",
        help="Materialize target-free outer-test geometry rows as well as training rows.",
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(1.0e-6, 1.0e-4, 1.0e-2),
        help="Positive ridge penalties considered by the training-only selector.",
    )
    args = parser.parse_args()
    SYSTEM = str(args.system)
    CUTOFF = float(args.cutoff)
    SEED = int(args.seed)
    DATA = args.data.resolve()
    CATALOGUE_ROOT = args.catalogue.resolve()
    ORDINARY_APPLICATION = args.ordinary_application.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prefix = SYSTEM.lower()
    training_structures = load_xyz_structures(DATA / f"{prefix}_training.xyz")
    if args.stage == "screen" or args.include_test_rows:
        test_structures = load_xyz_structures(DATA / f"{prefix}_test.xyz")
    else:
        test_structures = []
    structures = list(training_structures) + list(test_structures)

    descriptor = None
    compact_indices = ()
    full_ids = ()
    ordinary_all = None
    if args.kind in {"ordinary", "both"}:
        descriptor, compact_indices, full_ids = ordinary_descriptor()
        ordinary_evaluator = ACECovariantEvaluator(
            descriptor.site_basis_config,
            backend="pytorch",
            strict_backend=True,
            validate_backend=True,
            factorized_descriptor_runtime_policy="auto",
        )
        ordinary_evaluator.precompile_descriptors(descriptor.descriptor_specs)
        ordinary_all = cached_rows(
            "ordinary",
            (descriptor, ordinary_evaluator),
            structures,
            output / "row_cache_ordinary",
        )

    tagged_arm = None
    component_rows = ()
    tagged_all = None
    if args.kind in {"tagged", "both"}:
        tagged_arm, component_rows = build_tagged_arm()
        tagged_all = cached_rows(
            "tagged", tagged_arm, structures, output / "row_cache_tagged"
        )

    if args.stage == "rows":
        payload = {
            "schema": "ye3t_mlearn_target_free_row_cache_v1",
            "system": SYSTEM,
            "kinds": [
                value
                for value, rows in (("ordinary", ordinary_all), ("tagged", tagged_all))
                if rows is not None
            ],
            "training_structures": len(training_structures),
            "test_geometry_structures": len(test_structures),
            "target_values_used": False,
            "ordinary_feature_count": 0 if ordinary_all is None else len(full_ids),
            "tagged_feature_count": 0 if tagged_arm is None else tagged_arm.feature_count,
            "catalogue": str(CATALOGUE_ROOT),
            "ordinary_application": str(ORDINARY_APPLICATION),
            "cutoff_A": CUTOFF,
            "radial_config": RADIAL_CONFIG,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(output / "row_cache_manifest.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return

    if args.kind != "both":
        raise ValueError("--stage screen requires --kind both.")
    energies = {}
    forces = {}
    for index, atoms in enumerate(structures):
        energies[index], forces[index] = target(atoms)
    train_indices, validation_indices = stratified_validation(training_structures)
    final_fit_indices = tuple(range(len(training_structures)))
    test_indices = tuple(range(len(training_structures), len(structures)))

    ordinary_compact = {index: select(row, compact_indices) for index, row in ordinary_all.items()}

    by_component = {row["component_index"]: row for row in component_rows}
    matched_indices = []
    for component in (4, 5, 7):
        row = by_component[component]
        matched_indices.extend(range(row["feature_start"], row["feature_stop"]))
    tagged_matched = {index: select(row, matched_indices) for index, row in tagged_all.items()}
    ordinary_matched_count = len(full_ids) - len(matched_indices)
    ordinary_matched = {
        index: select(row, tuple(range(ordinary_matched_count)))
        for index, row in ordinary_all.items()
    }
    arms = {
        "pace_fixed": (("pace", ordinary_compact),),
        "pace_expanded_fixed_radial": (("pace_expanded", ordinary_all),),
        "pace_count_matched_tagged": (("pace_reduced", ordinary_matched), ("tagged_matched", tagged_matched)),
        "pace_fixed_plus_tagged": (("pace", ordinary_compact), ("tagged_full", tagged_all)),
    }
    alpha_grid = tuple(float(value) for value in args.ridge_alphas)
    if not alpha_grid or any(value <= 0.0 for value in alpha_grid):
        raise ValueError("--ridge-alphas must contain positive values; unregularized publication fits are rejected.")
    metrics = []
    trials = []
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    for name, blocks in arms.items():
        penalty_grid = (0.1, 0.3, 1.0, 3.0) if any(label.startswith("tagged") for label, _rows in blocks) else (1.0,)
        selected, _selection_bias, _selection_weights, arm_trials = fit_arm(
            name, blocks, train_indices, validation_indices, structures, energies, forces, alpha_grid, penalty_grid
        )
        trials.extend(arm_trials)
        _ignored, bias, weights, _ = fit_arm(
            name, blocks, final_fit_indices, validation_indices, structures, energies, forces,
            (selected["alpha"],), (selected["tagged_penalty"],),
        )
        test_metric = score(bias, weights, blocks, test_indices, structures, energies, forces)
        row = {
            "arm": name,
            "feature_count": sum(weight.size for weight in weights),
            "selected_alpha": selected["alpha"],
            "selected_tagged_penalty": selected["tagged_penalty"],
            "validation_energy_rmse_eV_per_atom": selected["validation_energy_rmse_eV_per_atom"],
            "validation_force_rmse_eV_per_A": selected["validation_force_rmse_eV_per_A"],
            "test_energy_rmse_eV_per_atom": test_metric["energy_rmse_eV_per_atom"],
            "test_force_rmse_eV_per_A": test_metric["force_rmse_eV_per_A"],
        }
        metrics.append(row)
        arrays = {
            "bias": np.asarray(bias),
            "block_names": np.asarray([label for label, _rows in blocks]),
            "block_widths": np.asarray([weight.size for weight in weights]),
        }
        for block_index, weight in enumerate(weights):
            arrays[f"weight_{block_index}"] = weight
        np.savez_compressed(model_dir / f"{prefix}_{name}.npz", **arrays)
        write_json(output / f"test_groups_{name}.json", test_metric["groups"])
        print(json.dumps(row), flush=True)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "trials.csv", trials)
    write_json(
        output / "summary.json",
        {
            "schema": "ye3t_mlearn_element_tagged_ace_screen_v1",
            "system": SYSTEM,
            "status": "initial_fixed_published_test_screen",
            "cutoff_A": CUTOFF,
            "radial_config": RADIAL_CONFIG,
            "training_structures": len(training_structures),
            "selection_train_structures": len(train_indices),
            "selection_validation_structures": len(validation_indices),
            "test_structures": len(test_indices),
            "tagged_feature_count": tagged_arm.feature_count,
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
