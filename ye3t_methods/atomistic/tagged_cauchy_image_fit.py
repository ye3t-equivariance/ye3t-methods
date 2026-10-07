"""Streamed linear fitting for certified tagged-Cauchy V3 images."""

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from ase.neighborlist import neighbor_list

from ye3t_methods.atomistic.tagged_cauchy_image import TaggedCauchyImageLinearModel
from ye3t_methods.atomistic.utils.fit_weights import structure_fit_weights


def _canonical_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(tuple(int(v) for v in value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def tagged_cauchy_reference_target_metadata(
    reference_energies=None, reference_potential_metadata=None
):
    """Return the immutable target-transform record bound into a fit."""

    refs = {
        str(key): float(value)
        for key, value in dict(reference_energies or {}).items()
    }
    if any(not np.isfinite(value) for value in refs.values()):
        raise ValueError("Reference energies must be finite.")
    external = None
    if reference_potential_metadata is not None:
        if not isinstance(reference_potential_metadata, Mapping):
            raise TypeError("reference_potential_metadata must be a mapping.")
        external = json.loads(
            json.dumps(
                dict(reference_potential_metadata),
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    return {
        "schema": "ye3t_tagged_target_transform_v1",
        "elemental_energy_offsets": {
            "enabled": bool(refs),
            "reference_energies": refs,
            "target_convention": (
                "E_residual = E_input - sum_type(n_type * E_ref[type])"
            ),
            "force_target_convention": "unchanged_constant_energy_offset",
        },
        "external_reference_potential": {
            "enabled": external is not None,
            "metadata": external,
            "target_convention": "caller_supplied_E_and_F_residuals",
        },
    }


def _geometry_hash(atoms):
    digest = hashlib.sha256()
    digest.update(b"ye3t_tagged_cauchy_binary64_geometry_v1\0")
    digest.update(
        json.dumps(
            list(atoms.get_chemical_symbols()),
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    for value in (
        np.asarray(atoms.get_positions(), dtype="<f8"),
        np.asarray(atoms.cell.array, dtype="<f8"),
        np.asarray(atoms.pbc, dtype=np.uint8),
    ):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _energy_target(atoms, key):
    if str(key) in getattr(atoms, "info", {}):
        return float(atoms.info[str(key)])
    calculator = getattr(atoms, "calc", None)
    if calculator is not None and str(key) in getattr(calculator, "results", {}):
        return float(calculator.results[str(key)])
    if str(key) in {"energy", "E"}:
        return float(atoms.get_potential_energy())
    raise KeyError(f"Structure is missing energy key {str(key)!r}.")


def _force_target(atoms, key):
    if str(key) in getattr(atoms, "arrays", {}):
        value = atoms.arrays[str(key)]
    else:
        calculator = getattr(atoms, "calc", None)
        if calculator is not None and str(key) in getattr(calculator, "results", {}):
            value = calculator.results[str(key)]
        elif str(key) in {"force", "forces", "F"}:
            value = atoms.get_forces()
        else:
            raise KeyError(f"Structure is missing force key {str(key)!r}.")
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (len(atoms), 3) or not np.all(np.isfinite(value)):
        raise ValueError("Force targets must be finite with shape [atoms,3].")
    return value


def _stress_target(atoms, key):
    """Read one ASE-order stress target in eV/Angstrom cubed."""
    if str(key) in getattr(atoms, "info", {}):
        value = atoms.info[str(key)]
    else:
        calculator = getattr(atoms, "calc", None)
        if calculator is not None and str(key) in getattr(calculator, "results", {}):
            value = calculator.results[str(key)]
        elif str(key) == "stress":
            value = atoms.get_stress(voigt=True)
        else:
            raise KeyError(f"Structure is missing stress key {str(key)!r}.")
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("Stress targets must be finite ASE-order Voigt vectors of length 6.")
    return value


def _descriptor_runtime(descriptor):
    if descriptor.metadata.get("descriptor_family") != "linear_tagged_cauchy_image":
        raise ValueError("Expected a linear_tagged_cauchy_image descriptor.")
    evaluator = descriptor.metadata.get("tagged_cauchy_image_evaluator")
    if evaluator is None:
        raise ValueError(
            "The tagged-Cauchy descriptor is preflight-only; compile it before fitting."
        )
    if dict(descriptor.type_map) != dict(evaluator.type_map):
        raise ValueError("Tagged-Cauchy descriptor/evaluator type maps disagree.")
    return evaluator


def _row_identity(descriptor, atoms):
    evaluator = _descriptor_runtime(descriptor)
    body = {
        "schema": "ye3t_tagged_cauchy_geometry_row_v3",
        "compiler_artifact_hash": str(evaluator.compiled.self_hash),
        "source_plan_hash": str(evaluator.source_plan["source_plan_hash"]),
        "type_map": dict(sorted(descriptor.type_map.items())),
        "geometry_hash": _geometry_hash(atoms),
        "force_design_convention": "F_equals_minus_dE_dR_v1",
        "stress_design_convention": "ASE_xx_yy_zz_yz_xz_xy_positive_dE_dstrain_over_volume_v1",
        "energy_row_convention": "total_feature_sum_v1",
        "polynomial_backend": evaluator.backend,
    }
    return {**body, "row_hash": _canonical_hash(body)}


def tagged_cauchy_image_geometry_row(descriptor, atoms, cache_dir=None):
    """Return one target-free energy/force design block."""

    evaluator = _descriptor_runtime(descriptor)
    identity = _row_identity(descriptor, atoms)
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / (identity["row_hash"] + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as handle:
                if str(handle["row_hash"].item()) != identity["row_hash"]:
                    raise ValueError("Tagged-Cauchy geometry-row cache hash mismatch.")
                row = {
                    "identity": identity,
                    "feature_sums": handle["feature_sums"],
                    "feature_square_sums": handle["feature_square_sums"],
                    "force_design": handle["force_design"],
                    "stress_design": handle["stress_design"],
                    "species_counts": handle["species_counts"],
                    "atom_count": int(handle["atom_count"].item()),
                    "cache_hit": True,
                }
                supplied_hashes = {
                    name: str(handle[name + "_hash"].item())
                    for name in (
                        "feature_sums",
                        "feature_square_sums",
                        "force_design",
                    "stress_design",
                        "species_counts",
                    )
                }
            feature_count = int(evaluator.feature_count)
            species_count = len(evaluator.species_order)
            atom_count = len(atoms)
            expected_shapes = {
                "feature_sums": (species_count, feature_count),
                "feature_square_sums": (species_count, feature_count),
                "force_design": (
                    3 * atom_count,
                    species_count * feature_count,
                ),
                "stress_design": (6, species_count * feature_count),
                "species_counts": (species_count,),
            }
            if row["atom_count"] != atom_count or any(
                tuple(row[name].shape) != expected_shapes[name]
                or not np.all(np.isfinite(row[name]))
                or supplied_hashes[name] != _array_hash(row[name])
                for name in expected_shapes
            ):
                raise ValueError(
                    "Tagged-Cauchy geometry-row cache content is invalid."
                )
            if not np.isclose(
                np.sum(row["species_counts"]), atom_count, rtol=0.0, atol=0.0
            ):
                raise ValueError(
                    "Tagged-Cauchy cached species counts do not match atom_count."
                )
            return row

    symbols = tuple(str(value) for value in atoms.get_chemical_symbols())
    unknown = sorted(set(symbols) - set(evaluator.type_map))
    if unknown:
        raise ValueError(f"Tagged-Cauchy structure contains unknown species: {unknown}.")
    positions = torch.as_tensor(
        np.asarray(atoms.get_positions(), dtype=np.float64), dtype=torch.float64
    )
    atom_types = torch.as_tensor(
        [evaluator.type_map[symbol] for symbol in symbols], dtype=torch.long
    )
    src, dst, shifts = neighbor_list(
        "ijS", atoms, evaluator.cutoff, self_interaction=False
    )
    src = torch.as_tensor(src, dtype=torch.long)
    dst = torch.as_tensor(dst, dtype=torch.long)
    shifts = torch.as_tensor(shifts, dtype=torch.float64)
    cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float64)
    displacement = (
        positions.index_select(0, dst)
        - positions.index_select(0, src)
        + shifts @ cell
    )
    edge_index = torch.stack((src, dst))
    neighbor_types = atom_types.index_select(0, dst)
    with torch.no_grad():
        features, edge_derivatives = evaluator.evaluate_edge_list(
            edge_index,
            displacement,
            neighbor_types,
            len(atoms),
            atom_types=atom_types,
        )
    feature_count = int(evaluator.feature_count)
    species_count = len(evaluator.species_order)
    atom_count = len(atoms)
    features = features.detach().cpu().numpy()
    edge_index = edge_index.detach().cpu().numpy()
    edge_derivatives = edge_derivatives.detach().cpu().numpy()
    atom_type_array = atom_types.detach().cpu().numpy()
    feature_sums = np.zeros((species_count, feature_count), dtype=np.float64)
    feature_square_sums = np.zeros(
        (species_count, feature_count), dtype=np.float64
    )
    species_counts = np.zeros((species_count,), dtype=np.float64)
    for species_index in range(species_count):
        mask = atom_type_array == species_index
        feature_sums[species_index] = np.sum(features[mask], axis=0)
        feature_square_sums[species_index] = np.sum(
            features[mask] * features[mask], axis=0
        )
        species_counts[species_index] = float(np.count_nonzero(mask))
    force_design = np.zeros(
        (atom_count, 3, species_count, feature_count), dtype=np.float64
    )
    stress_design = np.zeros((6, species_count, feature_count), dtype=np.float64)
    volume = float(atoms.get_volume()) if atoms.cell.rank == 3 else 0.0
    displacement_array = displacement.detach().cpu().numpy()
    for edge in range(edge_index.shape[1]):
        center = int(edge_index[0, edge])
        neighbor = int(edge_index[1, edge])
        central_species = int(atom_type_array[center])
        derivative = edge_derivatives[edge].T
        force_design[center, :, central_species, :] += derivative
        force_design[neighbor, :, central_species, :] -= derivative
        if volume > 0.0:
            distance = displacement_array[edge]
            stress_design[:, central_species, :] += np.stack((
                distance[0] * derivative[0],
                distance[1] * derivative[1],
                distance[2] * derivative[2],
                distance[1] * derivative[2],
                distance[0] * derivative[2],
                distance[0] * derivative[1],
            )) / volume
    row = {
        "identity": identity,
        "feature_sums": feature_sums,
        "feature_square_sums": feature_square_sums,
        "force_design": force_design.reshape(
            3 * atom_count, species_count * feature_count
        ),
        "stress_design": stress_design.reshape(6, species_count * feature_count),
        "species_counts": species_counts,
        "atom_count": atom_count,
        "cache_hit": False,
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=path.stem + ".",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(
                temporary,
                row_hash=np.asarray(identity["row_hash"]),
                feature_sums=row["feature_sums"],
                feature_sums_hash=np.asarray(_array_hash(row["feature_sums"])),
                feature_square_sums=row["feature_square_sums"],
                feature_square_sums_hash=np.asarray(
                    _array_hash(row["feature_square_sums"])
                ),
                force_design=row["force_design"],
                force_design_hash=np.asarray(_array_hash(row["force_design"])),
                stress_design=row["stress_design"],
                stress_design_hash=np.asarray(_array_hash(row["stress_design"])),
                species_counts=row["species_counts"],
                species_counts_hash=np.asarray(
                    _array_hash(row["species_counts"])
                ),
                atom_count=np.asarray(atom_count, dtype=np.int64),
            )
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return row


def build_tagged_cauchy_image_normal_equations(
    descriptor,
    structures,
    *,
    energy_key="energy",
    force_key="forces",
    stress_key="stress",
    target_energies=None,
    target_forces=None,
    target_stresses=None,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=0.0,
    geometry_cache_dir=None,
    structure_weights=None,
    structure_weight_key=None,
    structure_group_key=None,
    structure_group_weights=None,
    structure_group_default_weight=None,
    structure_group_normalize_mean=True,
    boltzmann_temperature_K=None,
    boltzmann_energy_key=None,
    boltzmann_weight_nugget=0.0,
    boltzmann_weight_prefactor=1.0,
    boltzmann_normalize_mean=True,
    min_structure_weight=0.0,
    reference_target_metadata=None,
    fit_intercept=True,
):
    """Stream target-free cached rows into reusable sufficient statistics."""

    evaluator = _descriptor_runtime(descriptor)
    structures = list(structures)
    if not structures:
        raise ValueError("Tagged-Cauchy fitting requires at least one structure.")
    if min(float(energy_weight), float(force_weight), float(stress_weight)) < 0.0:
        raise ValueError("Energy, force, and stress weights must be nonnegative.")
    if max(float(energy_weight), float(force_weight), float(stress_weight)) == 0.0:
        raise ValueError("At least one target weight must be positive.")
    if target_energies is None:
        target_energies = [
            _energy_target(atoms, energy_key) for atoms in structures
        ]
    target_energies = np.asarray(target_energies, dtype=np.float64)
    if target_energies.shape != (len(structures),) or not np.all(
        np.isfinite(target_energies)
    ):
        raise ValueError("target_energies must have one finite value per structure.")
    if target_forces is None:
        target_forces = [
            _force_target(atoms, force_key) for atoms in structures
        ]
    target_forces = tuple(
        np.asarray(value, dtype=np.float64) for value in target_forces
    )
    if len(target_forces) != len(structures) or any(
        value.shape != (len(atoms), 3) or not np.all(np.isfinite(value))
        for atoms, value in zip(structures, target_forces, strict=True)
    ):
        raise ValueError("target_forces must match every structure with shape [atoms,3].")
    if float(stress_weight) > 0.0:
        if target_stresses is None:
            target_stresses = [_stress_target(atoms, stress_key) for atoms in structures]
        target_stresses = np.asarray(target_stresses, dtype=np.float64)
        if (target_stresses.shape != (len(structures), 6)
                or not np.all(np.isfinite(target_stresses))):
            raise ValueError("target_stresses must have finite ASE-order shape [structures,6].")
        if any(atoms.cell.rank != 3 or atoms.get_volume() <= 0.0 for atoms in structures):
            raise ValueError("Stress fitting requires a positive-volume three-dimensional cell.")
    weights, weight_metadata = structure_fit_weights(
        structures,
        structure_weights=structure_weights,
        structure_weight_key=structure_weight_key,
        structure_group_key=structure_group_key,
        structure_group_weights=structure_group_weights,
        structure_group_default_weight=structure_group_default_weight,
        structure_group_normalize_mean=structure_group_normalize_mean,
        boltzmann_temperature_K=boltzmann_temperature_K,
        boltzmann_energy_key=boltzmann_energy_key,
        boltzmann_weight_nugget=boltzmann_weight_nugget,
        boltzmann_weight_prefactor=boltzmann_weight_prefactor,
        boltzmann_normalize_mean=boltzmann_normalize_mean,
        min_weight=min_structure_weight,
    )
    reference_target_metadata = (
        None
        if reference_target_metadata is None
        else json.loads(
            json.dumps(
                reference_target_metadata,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    )
    target_identity_body = {
        "energy_hash": _array_hash(target_energies),
        "force_hashes": tuple(_array_hash(value) for value in target_forces),
        "reference_target_metadata": reference_target_metadata,
    }
    if float(stress_weight) > 0.0:
        target_identity_body["stress_hash"] = _array_hash(target_stresses)
    target_identity = _canonical_hash(target_identity_body)
    feature_count = int(evaluator.feature_count)
    species_count = len(evaluator.species_order)
    beta_width = species_count * feature_count
    fit_intercept = bool(fit_intercept)
    width = beta_width + (species_count if fit_intercept else 0)
    temporary_cache = None
    if geometry_cache_dir is None:
        temporary_cache = tempfile.TemporaryDirectory(
            prefix="ye3t_tagged_cauchy_rows_"
        )
        row_cache_dir = Path(temporary_cache.name)
    else:
        row_cache_dir = Path(geometry_cache_dir)
    try:
        feature_sum = np.zeros((beta_width,), dtype=np.float64)
        feature_square_sum = np.zeros((beta_width,), dtype=np.float64)
        energy_per_atom = []
        force_square_sum = 0.0
        force_component_count = 0
        stress_square_sum = 0.0
        total_atom_count = 0
        row_hashes = []
        cache_hits = 0
        for index, atoms in enumerate(structures):
            row = tagged_cauchy_image_geometry_row(
                descriptor, atoms, cache_dir=row_cache_dir
            )
            row_hashes.append(row["identity"]["row_hash"])
            cache_hits += int(row["cache_hit"])
            atom_count = int(row["atom_count"])
            total_atom_count += atom_count
            feature_sum += row["feature_sums"].reshape(-1)
            feature_square_sum += row["feature_square_sums"].reshape(-1)
            energy_per_atom.append(float(target_energies[index]) / atom_count)
            force_target = target_forces[index].reshape(-1)
            force_square_sum += float(force_target @ force_target)
            force_component_count += int(force_target.size)
            if float(stress_weight) > 0.0:
                stress_square_sum += float(target_stresses[index] @ target_stresses[index])
        feature_mean = feature_sum / total_atom_count
        second_moment = feature_square_sum / total_atom_count
        if not fit_intercept:
            # Centering without a fitted constant changes the model space.
            feature_mean = np.zeros_like(feature_mean)
        variance = second_moment - feature_mean * feature_mean
        negative_tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(
            second_moment, 1.0
        )
        if np.any(variance < -negative_tolerance):
            raise FloatingPointError("Tagged-Cauchy feature variance became negative.")
        feature_scale = np.maximum(
            np.sqrt(np.maximum(variance, 0.0)), 1.0e-12
        )
        energy_target_scale = max(
            float(np.std(energy_per_atom)), 1.0e-12
        )
        force_target_scale = max(
            float(
                np.sqrt(force_square_sum / max(force_component_count, 1))
            ),
            1.0e-12,
        )
        stress_target_scale = max(
            float(np.sqrt(stress_square_sum / max(6 * len(structures), 1))),
            1.0e-12,
        )
        gram = np.zeros((width, width), dtype=np.float64)
        rhs = np.zeros((width,), dtype=np.float64)
        target_norm = 0.0
        energy_rows = 0
        force_rows = 0
        stress_rows = 0
        structure_count = len(structures)
        for index, (atoms, structure_weight) in enumerate(
            zip(structures, weights, strict=True)
        ):
            if float(structure_weight) <= 0.0:
                continue
            row = tagged_cauchy_image_geometry_row(
                descriptor, atoms, cache_dir=row_cache_dir
            )
            atom_count = int(row["atom_count"])
            if float(energy_weight) > 0.0:
                energy_design = (
                    row["feature_sums"].reshape(-1) / atom_count - feature_mean
                ) / feature_scale
                if fit_intercept:
                    energy_design = np.concatenate(
                        (energy_design, row["species_counts"] / atom_count))
                energy_target = float(target_energies[index]) / atom_count
                factor = (
                    float(structure_weight)
                    * float(energy_weight)
                    / (
                        structure_count
                        * energy_target_scale
                        * energy_target_scale
                    )
                )
                gram += factor * np.outer(energy_design, energy_design)
                rhs += factor * energy_design * energy_target
                target_norm += factor * energy_target * energy_target
                energy_rows += 1
            if float(force_weight) > 0.0:
                force_design = row["force_design"] / feature_scale
                force_target = target_forces[index].reshape(-1)
                factor = (
                    float(structure_weight)
                    * float(force_weight)
                    / (
                        structure_count
                        * force_design.shape[0]
                        * force_target_scale
                        * force_target_scale
                    )
                )
                gram[:beta_width, :beta_width] += factor * (
                    force_design.T @ force_design
                )
                rhs[:beta_width] += factor * (force_design.T @ force_target)
                target_norm += factor * float(force_target @ force_target)
                force_rows += int(force_design.shape[0])
            if float(stress_weight) > 0.0:
                stress_design = row["stress_design"] / feature_scale
                stress_target = target_stresses[index]
                factor = (
                    float(structure_weight) * float(stress_weight)
                    / (structure_count * 6 * stress_target_scale * stress_target_scale)
                )
                gram[:beta_width, :beta_width] += factor * (stress_design.T @ stress_design)
                rhs[:beta_width] += factor * (stress_design.T @ stress_target)
                target_norm += factor * float(stress_target @ stress_target)
                stress_rows += 6
    finally:
        if temporary_cache is not None:
            temporary_cache.cleanup()
    gram = 0.5 * (gram + gram.T)
    body = {
        "schema": "ye3t_tagged_cauchy_normal_equations_v2",
        "objective": "structure_balanced_train_scaled_E1_F1",
        "compiler_artifact_hash": str(evaluator.compiled.self_hash),
        "source_plan_hash": str(evaluator.source_plan["source_plan_hash"]),
        "species_order": tuple(evaluator.species_order),
        "feature_count": feature_count,
        "fit_intercept": fit_intercept,
        "structure_count": len(structures),
        "energy_rows": energy_rows,
        "force_rows": force_rows,
        "energy_weight": float(energy_weight),
        "force_weight": float(force_weight),
        "structure_weights": tuple(float(value) for value in weights),
        "row_hashes": tuple(row_hashes),
        "target_identity": target_identity,
        "reference_target_metadata": reference_target_metadata,
        "feature_mean_hash": _array_hash(feature_mean),
        "feature_scale_hash": _array_hash(feature_scale),
        "energy_target_scale": energy_target_scale,
        "force_target_scale": force_target_scale,
        "gram_hash": _array_hash(gram),
        "rhs_hash": _array_hash(rhs),
        "target_norm": float(target_norm),
    }
    if float(stress_weight) > 0.0:
        body.update({
            "schema": "ye3t_tagged_cauchy_normal_equations_v3",
            "objective": "structure_balanced_train_scaled_E1_F1_S1",
            "stress_rows": stress_rows,
            "stress_weight": float(stress_weight),
            "stress_target_scale": stress_target_scale,
        })
    return {
        **body,
        "normal_hash": _canonical_hash(body),
        "gram": gram,
        "rhs": rhs,
        "target_norm": float(target_norm),
        "beta_width": beta_width,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "structure_weight_metadata": weight_metadata,
        "geometry_cache_hits": cache_hits,
        "geometry_cache_replay_hits": len(structures),
    }


def solve_tagged_cauchy_image_ridge(descriptor, normal_equations, ridge_alpha=0.0):
    """Solve one normalized ridge system without rebuilding descriptor rows."""

    evaluator = _descriptor_runtime(descriptor)
    normal = dict(normal_equations)
    if str(normal.get("compiler_artifact_hash", "")) != str(
        evaluator.compiled.self_hash
    ) or str(normal.get("source_plan_hash", "")) != str(
        evaluator.source_plan["source_plan_hash"]
    ):
        raise ValueError("Tagged-Cauchy normal equations belong to another descriptor.")
    ridge_alpha = float(ridge_alpha)
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be finite and nonnegative.")
    gram = np.asarray(normal["gram"], dtype=np.float64)
    rhs = np.asarray(normal["rhs"], dtype=np.float64)
    identity_keys = (
        "schema",
        "objective",
        "compiler_artifact_hash",
        "source_plan_hash",
        "species_order",
        "feature_count",
        "structure_count",
        "energy_rows",
        "force_rows",
        "energy_weight",
        "force_weight",
        "structure_weights",
        "row_hashes",
        "target_identity",
        "reference_target_metadata",
        "feature_mean_hash",
        "feature_scale_hash",
        "energy_target_scale",
        "force_target_scale",
        "gram_hash",
        "rhs_hash",
        "target_norm",
    )
    if "fit_intercept" in normal:
        identity_keys = (*identity_keys, "fit_intercept")
    if "stress_rows" in normal:
        identity_keys = (*identity_keys, "stress_rows", "stress_weight", "stress_target_scale")
    if (
        str(normal.get("normal_hash", ""))
        != _canonical_hash({key: normal[key] for key in identity_keys})
        or str(normal.get("gram_hash", "")) != _array_hash(gram)
        or str(normal.get("rhs_hash", "")) != _array_hash(rhs)
    ):
        raise ValueError("Tagged-Cauchy normal-equation hash mismatch.")
    beta_width = int(normal["beta_width"])
    width = int(gram.shape[0])
    if gram.shape != (width, width) or rhs.shape != (width,):
        raise ValueError("Tagged-Cauchy normal-equation shapes are invalid.")
    feature_mean = np.asarray(normal["feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(normal["feature_scale"], dtype=np.float64)
    if (
        feature_mean.shape != (beta_width,)
        or feature_scale.shape != (beta_width,)
        or np.any(~np.isfinite(feature_mean))
        or np.any(~np.isfinite(feature_scale))
        or np.any(feature_scale <= 0.0)
        or str(normal.get("feature_mean_hash", "")) != _array_hash(feature_mean)
        or str(normal.get("feature_scale_hash", "")) != _array_hash(feature_scale)
    ):
        raise ValueError("Tagged-Cauchy fit-coordinate statistics are invalid.")
    ridge_metric = np.zeros((width,), dtype=np.float64)
    ridge_metric[:beta_width] = 1.0 / (feature_scale * feature_scale)
    system = gram + np.diag(ridge_alpha * ridge_metric)
    if ridge_alpha > 0 and not normal.get("fit_intercept", True):
        # Every coordinate has a positive prescribed penalty. Solve the SPD
        # system directly: an empirical eigencut would drop physically valid
        # columns when other species are absent from a small training split.
        diagonal = np.sqrt(np.diag(system))
        equilibrated = system/diagonal[:, None]/diagonal[None, :]
        factor = np.linalg.cholesky(equilibrated)
        scaled_coefficients = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs/diagonal))/diagonal
        solve_method = "diagonally_equilibrated_cholesky"
        retained_condition = None
        numerical_rank = None  # No empirical descriptor-rank decision.
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(system)
        largest = max(float(eigenvalues[-1]), 0.0)
        cutoff = largest * max(1.0e-24, np.finfo(np.float64).eps)
        keep = eigenvalues > cutoff
        if not np.any(keep):
            raise np.linalg.LinAlgError(
                "Tagged-Cauchy ridge system has no retained direction."
            )
        scaled_coefficients = eigenvectors[:, keep] @ (
            (eigenvectors[:, keep].T @ rhs) / eigenvalues[keep]
        )
        solve_method = "symmetric_eigendecomposition"
        retained_condition = float(eigenvalues[-1] / eigenvalues[keep][0])
        numerical_rank = int(np.count_nonzero(keep))
    if not np.all(np.isfinite(scaled_coefficients)):
        raise FloatingPointError("Tagged-Cauchy ridge solve produced non-finite coefficients.")
    runtime_coefficients = scaled_coefficients.copy()
    runtime_coefficients[:beta_width] /= feature_scale
    centering_shift = float(
        feature_mean @ runtime_coefficients[:beta_width]
    )
    runtime_coefficients[beta_width:] -= centering_shift
    species_count = len(evaluator.species_order)
    feature_count = int(evaluator.feature_count)
    beta_matrix = runtime_coefficients[:beta_width].reshape(
        species_count, feature_count
    )
    offsets = runtime_coefficients[beta_width:]
    if not normal.get("fit_intercept", True):
        if width != beta_width or np.any(feature_mean != 0.0):
            raise ValueError("Fixed-intercept fitting must not add or center offset columns.")
        offsets = np.zeros(species_count, dtype=np.float64)
    model = TaggedCauchyImageLinearModel(
        evaluator,
        {
            species: beta_matrix[index]
            for index, species in enumerate(evaluator.species_order)
        },
        {
            species: float(offsets[index])
            for index, species in enumerate(evaluator.species_order)
        },
    )
    weighted_sse = float(
        normal["target_norm"]
        - 2.0 * scaled_coefficients @ rhs
        + scaled_coefficients @ gram @ scaled_coefficients
    )
    fit_metadata = {
        "schema": "ye3t_tagged_cauchy_fit_metadata_v2",
        "fit_method": "ridge_streaming_gram",
        "objective": str(normal["objective"]),
        "ridge_alpha": ridge_alpha,
        "ridge_metric": (
            "identity_in_compiler_pivot_coordinates"
            if evaluator.compiled.payload.get("coordinate_policy") == "exact_physical_pivots_v1"
            else "identity_in_compiler_orthogonal_coordinates"
        ),
        "fit_intercept": bool(normal.get("fit_intercept", True)),
        "runtime": dict(evaluator.execution_report),
        "feature_count": feature_count,
        "species_count": species_count,
        "coefficient_count": width,
        "energy_rows": int(normal["energy_rows"]),
        "force_rows": int(normal["force_rows"]),
        "weighted_sse": max(weighted_sse, 0.0),
        "condition_number": retained_condition,
        "numerical_rank": numerical_rank,
        "solve_method": solve_method,
        "normal_hash": str(normal["normal_hash"]),
        "geometry_cache_hits": int(normal["geometry_cache_hits"]),
        "structure_weight_metadata": normal["structure_weight_metadata"],
        "energy_target_scale": float(normal["energy_target_scale"]),
        "force_target_scale": float(normal["force_target_scale"]),
        "reference_target_metadata": normal["reference_target_metadata"],
        "descriptor_first_flow": (
            "YE3TRepresentation.tagged_cauchy_image -> "
            "YE3TDescriptors.ye3t_basis -> YE3TModel.linear"
        ),
    }
    if "stress_rows" in normal:
        fit_metadata["stress_rows"] = int(normal["stress_rows"])
        fit_metadata["stress_target_scale"] = float(normal["stress_target_scale"])
    model.fit_metadata = fit_metadata
    model._ye3t_linear_fit_metadata = dict(fit_metadata)
    return model


def fit_tagged_cauchy_image_linear_model(
    descriptor,
    structures,
    *,
    ridge_alpha=0.0,
    normal_equations=None,
    **kwargs,
):
    """Fit or reuse one V3 normal system and return a deployable model."""

    if normal_equations is None:
        normal_equations = build_tagged_cauchy_image_normal_equations(
            descriptor, structures, **kwargs
        )
    elif kwargs:
        raise ValueError(
            "Target/weight options cannot be supplied with prebuilt normal_equations."
        )
    return solve_tagged_cauchy_image_ridge(
        descriptor, normal_equations, ridge_alpha=ridge_alpha
    )


def score_tagged_cauchy_image_model(
    model,
    structures,
    *,
    energy_key="energy",
    force_key="forces",
    target_energies=None,
    target_forces=None,
    geometry_cache_dir=None,
    descriptor=None,
):
    """Return unweighted per-atom energy and force-component RMSE."""

    structures = list(structures)
    if target_energies is None:
        target_energies = [
            _energy_target(atoms, energy_key) for atoms in structures
        ]
    if target_forces is None:
        target_forces = [
            _force_target(atoms, force_key) for atoms in structures
        ]
    energy_residuals = []
    force_residuals = []
    beta = np.concatenate(
        [
            model.beta_by_species[species].detach().cpu().numpy()
            for species in model.evaluator.species_order
        ]
    )
    offsets = np.asarray(
        [model.offsets[species] for species in model.evaluator.species_order],
        dtype=np.float64,
    )
    for atoms, energy_target, force_target in zip(
        structures, target_energies, target_forces, strict=True
    ):
        if geometry_cache_dir is None:
            symbols = tuple(str(value) for value in atoms.get_chemical_symbols())
            positions = torch.as_tensor(
                np.asarray(atoms.get_positions()), dtype=torch.float64
            )
            atom_types = torch.as_tensor(
                [model.evaluator.type_map[symbol] for symbol in symbols],
                dtype=torch.long,
            )
            periodic = bool(np.any(np.asarray(atoms.pbc, dtype=bool)))
            cell = (
                torch.as_tensor(
                    np.asarray(atoms.cell.array), dtype=torch.float64
                )
                if periodic
                else None
            )
            pbc = tuple(bool(value) for value in atoms.pbc) if periodic else None
            energy, forces, _virial, _atomic = model.energy_forces_virial(
                positions, atom_types, cell=cell, pbc=pbc
            )
            predicted_energy = float(energy.detach().cpu())
            predicted_forces = forces.detach().cpu().numpy()
        else:
            if descriptor is None:
                raise ValueError(
                    "descriptor is required when scoring from geometry_cache_dir."
                )
            if _descriptor_runtime(descriptor) is not model.evaluator:
                raise ValueError("Scoring descriptor and model evaluators differ.")
            row = tagged_cauchy_image_geometry_row(
                descriptor,
                atoms,
                cache_dir=geometry_cache_dir,
            )
            predicted_energy = float(
                row["feature_sums"].reshape(-1) @ beta
                + row["species_counts"] @ offsets
            )
            predicted_forces = row["force_design"] @ beta
            predicted_forces = predicted_forces.reshape(len(atoms), 3)
            if model.reference_terms:
                reference_atomic, reference_forces, _virial = model._reference_values(atoms)
                predicted_energy += float(reference_atomic.sum())
                predicted_forces += reference_forces
        energy_residuals.append(
            (predicted_energy - float(energy_target)) / len(atoms)
        )
        force_residuals.extend(
            (
                predicted_forces - np.asarray(force_target, dtype=np.float64)
            ).reshape(-1).tolist()
        )
    return {
        "structure_count": len(structures),
        "energy_rmse_eV_per_atom": float(
            np.sqrt(np.mean(np.square(energy_residuals)))
        ),
        "force_rmse_eV_per_A": float(
            np.sqrt(np.mean(np.square(force_residuals)))
        ),
    }


__all__ = [
    "build_tagged_cauchy_image_normal_equations",
    "fit_tagged_cauchy_image_linear_model",
    "score_tagged_cauchy_image_model",
    "solve_tagged_cauchy_image_ridge",
    "tagged_cauchy_reference_target_metadata",
    "tagged_cauchy_image_geometry_row",
]
