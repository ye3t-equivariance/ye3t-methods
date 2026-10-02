"""Shared per-structure weighting utilities for linear potential fits."""

from collections.abc import Mapping, Sequence
import math

import numpy as np


KB_EV_PER_K = 8.617333262145e-5


def _energy_from_atoms(atoms, key):
    key = str(key)
    if key in getattr(atoms, "info", {}):
        return float(atoms.info[key])
    calc = getattr(atoms, "calc", None)
    if calc is not None and key in getattr(calc, "results", {}):
        return float(calc.results[key])
    if key in {"energy", "E"}:
        return float(atoms.get_potential_energy())
    raise KeyError(f"Structure is missing energy key {key!r}.")


def structure_fit_weights(
    structures,
    *,
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
    min_weight=0.0,
):
    """Return per-structure objective weights and metadata.

    The returned weights multiply both the energy row and all force rows for a
    structure.  Normal-equation builders should therefore use
    ``sqrt(base_weight * structure_weight)`` for the corresponding row block.
    """

    structures = list(structures)
    n_structures = len(structures)
    weights = np.ones((n_structures,), dtype=float)
    sources = []
    if structure_weights is not None:
        explicit = np.asarray(list(structure_weights), dtype=float)
        if explicit.shape != (n_structures,):
            raise ValueError(
                "structure_weights must have one value per structure; "
                f"got shape {tuple(explicit.shape)} for {n_structures} structures."
            )
        if np.any(~np.isfinite(explicit)) or np.any(explicit < 0.0):
            raise ValueError("structure_weights must be finite and nonnegative.")
        weights *= explicit
        sources.append("explicit_sequence")
    if structure_weight_key is not None:
        key_weights = []
        for index, atoms in enumerate(structures):
            if str(structure_weight_key) not in getattr(atoms, "info", {}):
                raise KeyError(f"Structure {index} is missing atoms.info[{str(structure_weight_key)!r}].")
            key_weights.append(float(atoms.info[str(structure_weight_key)]))
        key_weights = np.asarray(key_weights, dtype=float)
        if np.any(~np.isfinite(key_weights)) or np.any(key_weights < 0.0):
            raise ValueError("structure_weight_key values must be finite and nonnegative.")
        weights *= key_weights
        sources.append(f"info:{structure_weight_key}")
    group_metadata = None
    if structure_group_key is not None or structure_group_weights is not None:
        if structure_group_key is None or structure_group_weights is None:
            raise ValueError(
                "structure_group_key and structure_group_weights must be supplied together."
            )
        if not isinstance(structure_group_weights, Mapping):
            raise TypeError("structure_group_weights must be a mapping from group label to weight.")
        group_key = str(structure_group_key)
        configured = {str(key): float(value) for key, value in structure_group_weights.items()}
        if any(not math.isfinite(value) or value < 0.0 for value in configured.values()):
            raise ValueError("structure_group_weights must be finite and nonnegative.")
        default = (
            None
            if structure_group_default_weight is None
            else float(structure_group_default_weight)
        )
        if default is not None and (not math.isfinite(default) or default < 0.0):
            raise ValueError("structure_group_default_weight must be finite and nonnegative.")
        labels = []
        group_values = []
        counts = {}
        for index, atoms in enumerate(structures):
            if group_key not in getattr(atoms, "info", {}):
                raise KeyError(f"Structure {index} is missing atoms.info[{group_key!r}].")
            label = str(atoms.info[group_key])
            if label not in configured and default is None:
                raise KeyError(
                    f"Structure {index} has unconfigured {group_key} group {label!r}."
                )
            labels.append(label)
            group_values.append(configured.get(label, default))
            counts[label] = int(counts.get(label, 0)) + 1
        group_values = np.asarray(group_values, dtype=float)
        raw_mean = float(np.mean(group_values)) if group_values.size else 0.0
        if group_values.size and raw_mean <= 0.0:
            raise ValueError("Categorical structure weights cannot all be zero.")
        if bool(structure_group_normalize_mean) and group_values.size:
            group_values = group_values / raw_mean
        weights *= group_values
        sources.append(f"group:{group_key}")
        group_metadata = {
            "key": group_key,
            "configured_weights": dict(sorted(configured.items())),
            "default_weight": default,
            "normalize_mean": bool(structure_group_normalize_mean),
            "raw_mean": raw_mean,
            "labels": tuple(labels),
            "counts": dict(sorted(counts.items())),
            "realized_weight_sums": {
                label: float(np.sum(group_values[np.asarray(labels) == label]))
                for label in sorted(counts)
            },
        }
    boltzmann_metadata = None
    if boltzmann_temperature_K is not None:
        temperature = float(boltzmann_temperature_K)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("boltzmann_temperature_K must be positive when supplied.")
        nugget = float(boltzmann_weight_nugget)
        if not math.isfinite(nugget) or nugget < 0.0 or nugget >= 1.0:
            raise ValueError("boltzmann_weight_nugget must be finite and satisfy 0 <= nugget < 1.")
        prefactor = float(boltzmann_weight_prefactor)
        if not math.isfinite(prefactor) or prefactor < 0.0 or prefactor > 1.0:
            raise ValueError("boltzmann_weight_prefactor must be finite and satisfy 0 <= prefactor <= 1.")
        energy_key = "energy" if boltzmann_energy_key is None else str(boltzmann_energy_key)
        energies = np.asarray([_energy_from_atoms(atoms, energy_key) for atoms in structures], dtype=float)
        if np.any(~np.isfinite(energies)):
            raise ValueError("Boltzmann energies must be finite.")
        relative = energies - float(np.min(energies))
        beta = 1.0 / (KB_EV_PER_K * temperature)
        boltzmann = np.exp(-np.clip(relative * beta, 0.0, 745.0))
        if nugget > 0.0:
            boltzmann = nugget + (1.0 - nugget) * boltzmann
        boltzmann = (1.0 - prefactor) + prefactor * boltzmann
        if bool(boltzmann_normalize_mean) and boltzmann.size:
            mean = float(np.mean(boltzmann))
            if mean > 0.0:
                boltzmann = boltzmann / mean
        weights *= boltzmann
        sources.append("boltzmann")
        boltzmann_metadata = {
            "temperature_K": temperature,
            "energy_key": energy_key,
            "weight_nugget": nugget,
            "weight_prefactor": prefactor,
            "weight_formula": "(1 - prefactor) + prefactor * (nugget + (1 - nugget) * exp(-(E - min(E)) / (k_B T)))",
            "relative_energy_reference": "minimum_energy_in_fit_set",
            "normalize_mean": bool(boltzmann_normalize_mean),
            "energy_min_eV": float(np.min(energies)) if energies.size else 0.0,
            "energy_max_eV": float(np.max(energies)) if energies.size else 0.0,
            "relative_energy_max_eV": float(np.max(relative)) if relative.size else 0.0,
        }
    floor = float(min_weight)
    if floor < 0.0:
        raise ValueError("min_weight must be nonnegative.")
    if floor > 0.0:
        weights = np.maximum(weights, floor)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Combined structure fit weights must be finite and nonnegative.")
    if weights.size and not np.any(weights > 0.0):
        raise ValueError("Combined structure fit weights cannot all be zero.")
    metadata = {
        "enabled": bool(sources),
        "sources": tuple(sources),
        "structure_count": int(n_structures),
        "min": float(np.min(weights)) if weights.size else 0.0,
        "max": float(np.max(weights)) if weights.size else 0.0,
        "mean": float(np.mean(weights)) if weights.size else 0.0,
        "zero_count": int(np.count_nonzero(weights == 0.0)),
        "min_weight": floor,
        "boltzmann": boltzmann_metadata,
        "groups": group_metadata,
    }
    return weights, metadata
