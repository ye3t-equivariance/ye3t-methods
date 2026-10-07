"""Refit the exact selected Ni portable columns under the paper's linear loss.

The base archive supplies immutable physical source and compiler coordinates.
New coefficients live in a separate self-contained artifact. No native plan is
claimed for the new readout.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from ye3t_methods.atomistic.linear_statistics import (
    assemble_prepared_normal_equations,
    prepare_weighted_normal_equations,
    solve_ridge_statistics,
    score_linear_statistics,
    structure_linear_statistics,
    sum_linear_statistics,
)
from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator

from .portable_archive import (
    _json, portable_feature_design_row, read_portable_linear_archive,
)


_SCHEMA = "ye3t_methods_portable_ni_refit_v1"
_MEMBERS = {"manifest.json", "base.ye3t", "fit.json"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _finite(value, name, *, positive=False):
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive." if positive else "."))
    return number


def _fit_settings(config, record):
    if not isinstance(config, dict) or set(config) != {
            "metadata", "representation", "basis", "runtime",
            "model", "targets", "validation"}:
        raise ValueError("Portable Ni refit requires the complete seven-section config.")
    if any(not isinstance(config[name], dict) for name in config):
        raise ValueError("Every portable Ni refit config section must be an object.")
    if (config["metadata"].get("schema") != "ye3t_config_v1" or
            config["basis"] != {"from_saved_model": True} or
            config["representation"] != {"from_saved_model": True}):
        raise ValueError("Portable Ni refit must use the selected saved-model basis.")
    runtime = config["runtime"]
    if (runtime.get("evaluator") != "torch" or
            runtime.get("neighbors") not in {"auto", "ase"} or
            runtime.get("device", "cpu") != "cpu"):
        raise ValueError("Portable Ni refit requires the CPU Torch evaluator and ASE neighbors.")
    model = config["model"]
    if model.get("kind") != "linear" or not isinstance(model.get("fit"), dict):
        raise ValueError("Portable Ni refit requires a linear fit config.")
    if ("feature_count" in model and
            (type(model["feature_count"]) is not int or
             model["feature_count"] != len(record["labels"]))):
        raise ValueError("Portable Ni refit feature count differs from the selected basis.")
    if model.get("source_archive_sha256") != hashlib.sha256(
            record["archive_bytes"]).hexdigest():
        raise ValueError("Portable Ni refit source archive hash differs from the selected basis.")
    fit = model["fit"]
    if fit.get("solver") != "paper_scaled_ridge" or set(fit) != {
            "solver", "alpha", "weights", "group_weights", "tagged_penalty"}:
        raise ValueError("Portable Ni refit requires the paper_scaled_ridge objective.")
    weights = fit["weights"]
    if not isinstance(weights, dict) or set(weights) != {"energy", "forces"}:
        raise ValueError("Portable Ni refit needs explicit energy and force weights.")
    group_weights = fit["group_weights"]
    if not isinstance(group_weights, dict) or not group_weights:
        raise ValueError("Portable Ni refit needs explicit group weights.")
    parsed_groups = {str(name): _finite(value, f"group weight {name}", positive=True)
                     for name, value in group_weights.items()}
    targets = config["targets"]
    if (set(targets) != {"energy", "forces", "group", "stress"} or
            not all(isinstance(targets[name], str) and targets[name]
                    for name in ("energy", "forces", "group")) or
            targets["stress"] is not None):
        raise ValueError("Portable Ni refit needs energy, forces, and group targets without stress.")
    validation = config["validation"]
    checks = validation.get("checks", [])
    if (not isinstance(checks, (list, tuple)) or
            not all(isinstance(check, str) for check in checks) or
            len(set(checks)) != len(checks) or
            set(checks) - {"force_fd", "round_trip"}):
        raise ValueError("Portable Ni refit supports force_fd and round_trip checks.")
    return {
        "alpha": _finite(fit["alpha"], "alpha", positive=True),
        "energy_weight": _finite(weights["energy"], "energy weight", positive=True),
        "force_weight": _finite(weights["forces"], "force weight", positive=True),
        "group_weights": parsed_groups,
        "tagged_penalty": _finite(fit["tagged_penalty"], "tagged penalty", positive=True),
        "targets": targets, "checks": tuple(checks),
    }


def apply_portable_refit(base_archive, ordinary, tagged, e0, fit_metadata):
    """Rebuild every coefficient view from one trusted base and selected vector."""
    record = read_portable_linear_archive(base_archive)
    if record is None:
        raise ValueError("Portable refit needs a trusted Ni base archive.")
    ordinary = np.asarray(ordinary, dtype=np.float64)
    tagged = np.asarray(tagged, dtype=np.float64)
    e0 = _finite(e0, "Ni per-atom intercept")
    count = len(record["ordinary_bundle"].descriptor_specs)
    selected = tuple(record["tagged_indices"])
    if (ordinary.shape != (count,) or tagged.shape != (len(selected),) or
            not np.isfinite(ordinary).all() or not np.isfinite(tagged).all() or
            not isinstance(fit_metadata, dict)):
        raise ValueError("Portable refit coefficient shape or metadata is invalid.")
    full_tagged = np.zeros(69, dtype=np.float64)
    full_tagged[list(selected)] = tagged
    record["weights"]["ordinary"] = ordinary.copy()
    record["weights"]["tagged_selected"] = tagged.copy()
    record["weights"]["tagged_beta_69"] = full_tagged
    record["weights"]["per_species_E0_Ni"] = np.asarray([e0], dtype=np.float64)
    record["ordinary_bundle"].weight = ordinary.copy()
    record["ordinary_bundle"].bias = e0
    record["tagged_model"].beta = torch.as_tensor(full_tagged.copy())
    record["refit"] = {"schema": _SCHEMA, "fit_metadata": fit_metadata,
                       "native_plan_status": "unavailable_for_refit"}
    return record


def fit_portable_linear(record, structures, config):
    """Fit the frozen selected Ni basis with the paper's scaled E/F objective."""
    settings = _fit_settings(config, record)
    frames = tuple(structures)
    if not frames:
        raise ValueError("Portable Ni refit requires labeled ASE structures.")
    expected = config["validation"].get("expected_train_count")
    if expected is not None and (type(expected) is not int or
                                 expected != len(frames)):
        raise ValueError("Portable Ni refit training count differs from the config.")
    targets = settings["targets"]
    reference = YE3TZBLCalculator.from_model_manifest({
        "reference_potential": record["sources"]["reference_potential"],
    })
    by_group = {}
    training_hash = hashlib.sha256()
    design_hash = hashlib.sha256()
    residual_hash = hashlib.sha256()
    for index, atoms in enumerate(frames):
        if set(atoms.get_chemical_symbols()) != {"Ni"}:
            raise ValueError(f"Portable Ni refit structure {index} contains another species.")
        group = atoms.info.get(targets["group"])
        if group not in settings["group_weights"]:
            raise ValueError(f"Structure {index} has an unconfigured fit group {group!r}.")
        if targets["energy"] in atoms.info:
            target_energy = float(atoms.info[targets["energy"]])
        elif targets["energy"] == "energy" and atoms.calc is not None:
            target_energy = float(atoms.get_potential_energy())
        else:
            raise ValueError(f"Structure {index} lacks energy target {targets['energy']!r}.")
        if targets["forces"] in atoms.arrays:
            target_forces = np.asarray(atoms.arrays[targets["forces"]],
                                       dtype=np.float64)
        elif targets["forces"] == "forces" and atoms.calc is not None:
            target_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        else:
            raise ValueError(f"Structure {index} lacks force target {targets['forces']!r}.")
        if (not math.isfinite(target_energy) or
                target_forces.shape != (len(atoms), 3) or
                not np.isfinite(target_forces).all()):
            raise ValueError(f"Structure {index} has invalid fit targets.")
        row = portable_feature_design_row(record, atoms)
        zbl_atoms = atoms.copy()
        zbl_atoms.calc = reference
        residual_energy = target_energy - float(zbl_atoms.get_potential_energy())
        residual_forces = (target_forces -
                           np.asarray(zbl_atoms.get_forces(), dtype=np.float64))
        if (not math.isfinite(residual_energy) or
                residual_forces.shape != (len(atoms), 3) or
                not np.isfinite(residual_forces).all()):
            raise ValueError(f"Structure {index} has invalid residual targets.")
        for key in ("site_features", "energy", "forces"):
            values = np.ascontiguousarray(row[key], dtype="<f8")
            design_hash.update(_canonical([key, values.shape]))
            design_hash.update(values.tobytes())
        residual_hash.update(np.asarray([residual_energy], dtype="<f8").tobytes())
        residual_hash.update(np.ascontiguousarray(residual_forces, dtype="<f8").tobytes())
        features = row["site_features"]
        statistic = structure_linear_statistics(
            row["energy"], np.square(features).sum(axis=0),
            row["forces"], len(atoms), residual_energy / len(atoms),
            residual_forces,
        )
        by_group.setdefault(group, []).append(statistic)
        training_hash.update(_canonical({
            "symbols": atoms.get_chemical_symbols(),
            "positions_A": np.asarray(atoms.positions).tolist(),
            "cell_A": np.asarray(atoms.cell.array).tolist(),
            "pbc": np.asarray(atoms.pbc).tolist(),
            "group": group, "energy_eV": target_energy,
            "forces_eV_per_A": target_forces.tolist(),
        }))
    if set(by_group) != set(settings["group_weights"]):
        raise ValueError("Every declared fit group needs at least one training structure.")
    groups = {group: sum_linear_statistics(rows)
              for group, rows in by_group.items()}
    prepared = prepare_weighted_normal_equations(groups)
    normal = assemble_prepared_normal_equations(
        prepared, group_weights=settings["group_weights"],
        energy_weight=settings["energy_weight"],
        force_weight=settings["force_weight"],
    )
    ordinary_count = len(record["weights"]["ordinary"])
    penalty = np.ones(len(record["labels"]), dtype=np.float64)
    penalty[ordinary_count:] = settings["tagged_penalty"]
    solution = solve_ridge_statistics(normal, settings["alpha"], penalty)
    coefficients = np.asarray(solution["runtime_coefficients"], dtype=np.float64)
    if (coefficients.shape != (len(record["labels"]) + 1,) or
            not np.isfinite(coefficients).all()):
        raise FloatingPointError("Portable Ni refit returned invalid coefficients.")
    aggregate = sum_linear_statistics(groups.values())
    score = score_linear_statistics(aggregate, coefficients)
    metadata = {
        "schema": "ye3t_methods_paper_scaled_fit_v1",
        "base_archive_sha256": hashlib.sha256(record["archive_bytes"]).hexdigest(),
        "training_data_sha256": training_hash.hexdigest(),
        "design_rows_sha256": design_hash.hexdigest(),
        "residual_targets_sha256": residual_hash.hexdigest(),
        "normal_gram_sha256": hashlib.sha256(
            np.ascontiguousarray(normal["gram"], dtype="<f8").tobytes()).hexdigest(),
        "normal_rhs_sha256": hashlib.sha256(
            np.ascontiguousarray(normal["rhs"], dtype="<f8").tobytes()).hexdigest(),
        "training_score": score,
        "structure_count": len(frames),
        "group_structure_counts": {group: len(rows) for group, rows in by_group.items()},
        "objective": {"alpha": settings["alpha"],
                      "energy_weight": settings["energy_weight"],
                      "force_weight": settings["force_weight"],
                      "group_weights": settings["group_weights"],
                      "tagged_penalty": settings["tagged_penalty"],
                      "energy_row": "residual_energy_per_atom",
                      "force_row": "residual_cartesian_force_per_component",
                      "force_loss_aggregation": "mean_structures(mean_3N(squared_force_residual))",
                      "intercept": "residual_energy_per_atom",
                      "reference": record["sources"]["reference_potential"]},
        "normalization": {
            "feature_mean": prepared["feature_mean"].tolist(),
            "feature_scale": prepared["feature_scale"].tolist(),
            "energy_target_scale": prepared["energy_target_scale"],
            "force_target_scale": prepared["force_target_scale"],
        },
        "targets": targets, "validation_checks": list(settings["checks"]),
    }
    return apply_portable_refit(
        record["archive_bytes"], coefficients[1:1 + ordinary_count],
        coefficients[1 + ordinary_count:], coefficients[0], metadata,
    )


def write_portable_refit_archive(record, path):
    """Save a new Ni readout without changing the trusted selected basis."""
    if (record.get("refit", {}).get("schema") != _SCHEMA or
            record["refit"].get("native_plan_status") != "unavailable_for_refit"):
        raise ValueError("Only a completed portable Ni refit can be written.")
    base = record["archive_bytes"]
    base_hash = hashlib.sha256(base).hexdigest()
    fit = {
        "base_archive_sha256": base_hash,
        "feature_count": len(record["labels"]),
        "selected_coordinate_ids": [row["feature_id"] if row["branch"] == "ordinary"
                                    else row["compiler_request_hash"] + ":" +
                                    str(row["tagged_program_column"])
                                    for row in record["labels"]],
        "ordinary": np.asarray(record["weights"]["ordinary"]).tolist(),
        "tagged_selected": np.asarray(record["weights"]["tagged_selected"]).tolist(),
        "per_species_E0_Ni": float(record["weights"]["per_species_E0_Ni"][0]),
        "fit_metadata": record["refit"]["fit_metadata"],
    }
    fit_bytes = _canonical(fit)
    manifest = {
        "schema": _SCHEMA,
        "maturity": "bounded_ni_scalar_refit",
        "native_plan_status": "unavailable_for_refit",
        "base_sha256": base_hash,
        "fit_sha256": hashlib.sha256(fit_bytes).hexdigest(),
    }
    manifest["self_hash"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    target = Path(path)
    if target.suffix != ".ye3t":
        raise ValueError("Portable Ni refit bundles use a .ye3t path.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".",
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in (("manifest.json", _canonical(manifest)),
                               ("base.ye3t", base), ("fit.json", fit_bytes)):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_portable_refit_archive(path):
    """Read the bounded refit schema, returning None for another .ye3t schema."""
    target = Path(path)
    with zipfile.ZipFile(target) as archive:
        infos = archive.infolist()
        if "manifest.json" not in {info.filename for info in infos}:
            return None
        if archive.getinfo("manifest.json").file_size > 2 * 1024 * 1024:
            return None
        manifest_bytes = archive.read("manifest.json")
        manifest = _json(manifest_bytes)
        if not isinstance(manifest, dict):
            return None
        if manifest.get("schema") != _SCHEMA:
            return None
        if target.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Portable Ni refit exceeds the 8 MiB reader limit.")
        if len(manifest_bytes) > 4096:
            raise ValueError("Portable Ni refit manifest exceeds the reader limit.")
        if (len(infos) != 3 or {info.filename for info in infos} != _MEMBERS or
                any(info.file_size > 4 * 1024 * 1024 or
                    info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1 or
                    ((info.external_attr >> 16) & 0o170000) == 0o120000
                    for info in infos)):
            raise ValueError("Portable Ni refit has an invalid member inventory.")
        if (set(manifest) != {"schema", "maturity", "native_plan_status",
                             "base_sha256", "fit_sha256", "self_hash"} or
                manifest["maturity"] != "bounded_ni_scalar_refit" or
                manifest["native_plan_status"] != "unavailable_for_refit" or
                hashlib.sha256(_canonical({key: value for key, value in
                                           manifest.items() if key != "self_hash"})).hexdigest()
                != manifest["self_hash"]):
            raise ValueError("Portable Ni refit manifest identity is invalid.")
        base = archive.read("base.ye3t")
        fit_bytes = archive.read("fit.json")
    if (hashlib.sha256(base).hexdigest() != manifest["base_sha256"] or
            hashlib.sha256(fit_bytes).hexdigest() != manifest["fit_sha256"]):
        raise ValueError("Portable Ni refit member hash mismatch.")
    fit = _json(fit_bytes)
    if (not isinstance(fit, dict) or set(fit) != {
            "base_archive_sha256", "feature_count", "selected_coordinate_ids",
            "ordinary", "tagged_selected", "per_species_E0_Ni", "fit_metadata"} or
            fit["base_archive_sha256"] != manifest["base_sha256"]):
        raise ValueError("Portable Ni refit coefficient record is invalid.")
    record = apply_portable_refit(
        base, fit["ordinary"], fit["tagged_selected"],
        fit["per_species_E0_Ni"], fit["fit_metadata"],
    )
    expected_ids = [row["feature_id"] if row["branch"] == "ordinary"
                    else row["compiler_request_hash"] + ":" +
                    str(row["tagged_program_column"])
                    for row in record["labels"]]
    if (fit["feature_count"] != len(record["labels"]) or
            fit["selected_coordinate_ids"] != expected_ids or
            fit["fit_metadata"].get("base_archive_sha256") !=
            manifest["base_sha256"]):
        raise ValueError("Portable Ni refit selected coordinates changed.")
    return record
