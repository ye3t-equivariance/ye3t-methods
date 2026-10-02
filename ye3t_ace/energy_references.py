"""Per-element constant energy references for fixed linear models."""

import math

import numpy as np


def fit_element_reference_energies(
    structures, *, elements=None, initial_reference_energies=None,
    weighting="per_atom", ridge=0.0,
):
    """Fit composition-count offsets from precomputed ASE energy labels."""
    frames = list(structures)
    if not frames:
        raise ValueError("Need at least one structure to fit reference energies.")
    if weighting not in ("per_atom", "total"):
        raise ValueError("weighting must be 'per_atom' or 'total'")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")

    observed = sorted({str(symbol) for atoms in frames for symbol in atoms.get_chemical_symbols()})
    elements = observed if elements is None else [str(symbol) for symbol in elements]
    if len(elements) != len(set(elements)):
        raise ValueError("elements must not contain duplicates")
    missing = sorted(set(observed) - set(elements))
    if missing:
        raise ValueError("elements omits observed species " + repr(missing))
    initial = {
        str(key): float(value)
        for key, value in dict(initial_reference_energies or {}).items()
    }
    unknown_initial = sorted(set(initial) - set(elements))
    if unknown_initial:
        raise ValueError("initial_reference_energies contains unknown species " + repr(unknown_initial))
    initial_vector = np.asarray([initial.get(symbol, 0.0) for symbol in elements], dtype=float)
    counts = np.zeros((len(frames), len(elements)), dtype=float)
    energies = np.zeros(len(frames), dtype=float)
    atom_counts = np.zeros(len(frames), dtype=float)
    element_index = {symbol: index for index, symbol in enumerate(elements)}
    for row, atoms in enumerate(frames):
        symbols = [str(symbol) for symbol in atoms.get_chemical_symbols()]
        if not symbols:
            raise ValueError("reference-energy fitting does not accept empty structures")
        results = getattr(getattr(atoms, "calc", None), "results", {}) or {}
        energy = atoms.info.get("energy", results.get("energy"))
        if energy is None:
            raise ValueError("reference-energy fitting requires a precomputed energy label")
        atom_counts[row] = float(len(symbols))
        energies[row] = float(energy)
        if not np.isfinite(energies[row]):
            raise ValueError("reference-energy fitting requires finite energy labels")
        for symbol in symbols:
            counts[row, element_index[symbol]] += 1.0

    design = counts.copy()
    target = energies - counts @ initial_vector
    if weighting == "per_atom":
        design = design / atom_counts[:, None]
        target = target / atom_counts
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = (
        float(singular_values[0]) * float(max(design.shape)) * float(np.finfo(float).eps)
        if singular_values.size else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > tolerance else float("inf")
    )
    if ridge > 0.0:
        fit_design = np.concatenate((design, math.sqrt(ridge) * np.eye(len(elements))), axis=0)
        fit_target = np.concatenate((target, np.zeros(len(elements))), axis=0)
    else:
        fit_design, fit_target = design, target
    correction, _residuals, _fit_rank, _fit_singular_values = np.linalg.lstsq(
        fit_design, fit_target, rcond=None,
    )
    fitted_vector = initial_vector + correction
    residual_per_atom = (energies - counts @ fitted_vector) / atom_counts
    return {
        "reference_energies": {symbol: float(fitted_vector[index]) for index, symbol in enumerate(elements)},
        "corrections": {symbol: float(correction[index]) for index, symbol in enumerate(elements)},
        "elements": tuple(elements), "weighting": str(weighting), "ridge": float(ridge),
        "structure_count": int(len(frames)), "design_rank": int(rank),
        "design_column_count": int(len(elements)),
        "condition_number": float(condition_number),
        "singular_values": tuple(float(value) for value in singular_values),
        "train_residual_mean_eV_per_atom": float(np.mean(residual_per_atom)),
        "train_residual_std_eV_per_atom": float(np.std(residual_per_atom)),
        "train_residual_rmse_eV_per_atom": float(np.sqrt(np.mean(residual_per_atom**2))),
    }
