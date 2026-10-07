"""Sufficient statistics for streamed linear energy/force regression."""

import numpy as np


LINEAR_SUFFICIENT_STATISTICS_SCHEMA = "ye3t_linear_sufficient_statistics_v1"


def structure_linear_statistics(
    feature_sum,
    feature_square_sum,
    force_design,
    atom_count,
    energy_target,
    force_target,
):
    """Build one structure's raw-coordinate regression statistics.

    Purpose: retain the information needed for repeated weighted linear fits
    without retaining or rematerializing a dense energy/force design matrix.
    Math: energy uses one per-atom row ``[1, sum(B)/N]``.  Force fitting uses
    ``X_F.T X_F / (3N)`` so each structure has equal force-loss weight, while
    a second unnormalised force block is retained for componentwise RMSE.
    Input: target-free descriptor sums/Jacobian and residual energy/forces.
    Output: an additive sufficient-statistics dictionary.
    Does not: choose weights, regularisation, folds, or a reference potential.
    """

    feature_sum = np.asarray(feature_sum, dtype=np.float64)
    feature_square_sum = np.asarray(feature_square_sum, dtype=np.float64)
    force_design = np.asarray(force_design, dtype=np.float64)
    force_target = np.asarray(force_target, dtype=np.float64).reshape(-1)
    atom_count = int(atom_count)
    if atom_count <= 0:
        raise ValueError("atom_count must be positive.")
    if feature_sum.ndim != 1 or feature_square_sum.shape != feature_sum.shape:
        raise ValueError("Feature sums must be one-dimensional and shape matched.")
    if force_design.shape != (3 * atom_count, feature_sum.size):
        raise ValueError("force_design must have shape (3 * atom_count, features).")
    if force_target.shape != (3 * atom_count,):
        raise ValueError("force_target must contain three components per atom.")
    energy_target = float(energy_target)
    energy_row = np.concatenate((np.ones(1), feature_sum / atom_count))
    force_rows = np.concatenate(
        (np.zeros((3 * atom_count, 1)), force_design), axis=1
    )
    force_scale = float(3 * atom_count)
    return {
        "schema": LINEAR_SUFFICIENT_STATISTICS_SCHEMA,
        "feature_count": int(feature_sum.size),
        "structure_count": 1,
        "atom_count": atom_count,
        "force_component_count": 3 * atom_count,
        "feature_sum": feature_sum.copy(),
        "feature_square_sum": feature_square_sum.copy(),
        "energy_gram": np.outer(energy_row, energy_row),
        "energy_rhs": energy_row * energy_target,
        "energy_target_sum": energy_target,
        "energy_target_square": energy_target * energy_target,
        "force_fit_gram": force_rows.T @ force_rows / force_scale,
        "force_fit_rhs": force_rows.T @ force_target / force_scale,
        "force_fit_target_square": float(force_target @ force_target / force_scale),
        "force_metric_gram": force_rows.T @ force_rows,
        "force_metric_rhs": force_rows.T @ force_target,
        "force_metric_target_square": float(force_target @ force_target),
    }


def sum_linear_statistics(statistics):
    """Add compatible structure or shard statistics exactly.

    Purpose: form group, fold, or complete-training sufficient statistics.
    Math: Gram matrices, right-hand sides, target squares, and source moments
    are additive over structures.
    Input: a nonempty iterable produced by this module.
    Output: one independent aggregate dictionary.
    Does not: average, weight, centre, scale, or solve the regression problem.
    """

    statistics = tuple(statistics)
    if not statistics:
        raise ValueError("At least one statistics record is required.")
    feature_count = int(statistics[0]["feature_count"])
    array_names = (
        "feature_sum",
        "feature_square_sum",
        "energy_gram",
        "energy_rhs",
        "force_fit_gram",
        "force_fit_rhs",
        "force_metric_gram",
        "force_metric_rhs",
    )
    scalar_names = (
        "energy_target_sum",
        "energy_target_square",
        "force_fit_target_square",
        "force_metric_target_square",
    )
    result = {
        "schema": LINEAR_SUFFICIENT_STATISTICS_SCHEMA,
        "feature_count": feature_count,
        "structure_count": 0,
        "atom_count": 0,
        "force_component_count": 0,
    }
    for name in array_names:
        result[name] = np.zeros_like(
            np.asarray(statistics[0][name], dtype=np.float64)
        )
    for name in scalar_names:
        result[name] = 0.0
    for record in statistics:
        if record.get("schema") != LINEAR_SUFFICIENT_STATISTICS_SCHEMA:
            raise ValueError("Unsupported linear sufficient-statistics schema.")
        if int(record["feature_count"]) != feature_count:
            raise ValueError("Cannot combine statistics with different feature counts.")
        result["structure_count"] += int(record["structure_count"])
        result["atom_count"] += int(record["atom_count"])
        result["force_component_count"] += int(record["force_component_count"])
        for name in array_names:
            value = np.asarray(record[name], dtype=np.float64)
            if value.shape != result[name].shape:
                raise ValueError(f"Statistics array {name} has an incompatible shape.")
            result[name] += value
        for name in scalar_names:
            result[name] += float(record[name])
    return result


def select_linear_statistics(statistics, feature_indices):
    """Select a feature subspace without returning to descriptor rows.

    Purpose: derive matched-count ACE and YE3T problems from one parent cache.
    Math: take the principal Gram submatrix containing the intercept and the
    requested raw feature coordinates, together with matching RHS/moments.
    Input: one statistics shard and unique in-range feature indices.
    Output: an independent additive shard in the selected coordinate order.
    Does not: clip a representation-theory multiplicity; callers must select
    complete compiler-provided components when that is their catalogue unit.
    """

    if statistics.get("schema") != LINEAR_SUFFICIENT_STATISTICS_SCHEMA:
        raise ValueError("Unsupported linear sufficient-statistics schema.")
    indices = np.asarray(feature_indices, dtype=np.int64)
    if indices.ndim != 1 or len(set(indices.tolist())) != indices.size:
        raise ValueError("feature_indices must be a vector of unique indices.")
    feature_count = int(statistics["feature_count"])
    if np.any(indices < 0) or np.any(indices >= feature_count):
        raise IndexError("feature_indices contains an out-of-range coordinate.")
    coordinates = np.concatenate((np.zeros(1, dtype=np.int64), indices + 1))
    result = {
        "schema": LINEAR_SUFFICIENT_STATISTICS_SCHEMA,
        "feature_count": int(indices.size),
        "structure_count": int(statistics["structure_count"]),
        "atom_count": int(statistics["atom_count"]),
        "force_component_count": int(statistics["force_component_count"]),
        "feature_sum": np.asarray(statistics["feature_sum"])[indices].copy(),
        "feature_square_sum": np.asarray(statistics["feature_square_sum"])[indices].copy(),
    }
    for prefix in ("energy", "force_fit", "force_metric"):
        gram = np.asarray(statistics[prefix + "_gram"])
        result[prefix + "_gram"] = gram[np.ix_(coordinates, coordinates)].copy()
        result[prefix + "_rhs"] = np.asarray(statistics[prefix + "_rhs"])[
            coordinates
        ].copy()
    for name in (
        "energy_target_sum",
        "energy_target_square",
        "force_fit_target_square",
        "force_metric_target_square",
    ):
        result[name] = float(statistics[name])
    return result


def feature_normalization(statistics, minimum_scale=1.0e-12):
    """Return training-only per-site feature means and standard deviations.

    Purpose: condition a linear solve while keeping runtime coefficients in
    the original descriptor coordinates.
    Math: moments are accumulated over atomic sites, not structures.
    Input: aggregate statistics and a positive scale floor.
    Output: ``(mean, scale)`` arrays.
    Does not: whiten features or form a data-covariance Gram construction.
    """

    if statistics.get("schema") != LINEAR_SUFFICIENT_STATISTICS_SCHEMA:
        raise ValueError("Unsupported linear sufficient-statistics schema.")
    atom_count = int(statistics["atom_count"])
    if atom_count <= 0:
        raise ValueError("Statistics must contain at least one atom.")
    minimum_scale = float(minimum_scale)
    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive.")
    mean = np.asarray(statistics["feature_sum"], dtype=np.float64) / atom_count
    second = (
        np.asarray(statistics["feature_square_sum"], dtype=np.float64)
        / atom_count
    )
    scale = np.maximum(
        np.sqrt(np.maximum(second - mean * mean, 0.0)), minimum_scale
    )
    return mean, scale


def standardization_transform(feature_mean, feature_scale):
    """Map standardized fit coefficients into raw runtime coordinates.

    Purpose: apply centring/scaling to Gram matrices without materialising X.
    Math: ``X_standard = X_raw @ transform`` and therefore
    ``beta_raw = transform @ beta_standard``.
    Input: same-shaped one-dimensional mean and positive scale arrays.
    Output: a square transform including the unscaled intercept coordinate.
    Does not: modify descriptor rows or runtime evaluation.
    """

    feature_mean = np.asarray(feature_mean, dtype=np.float64)
    feature_scale = np.asarray(feature_scale, dtype=np.float64)
    if feature_mean.ndim != 1 or feature_scale.shape != feature_mean.shape:
        raise ValueError("Feature mean and scale must be same-shaped vectors.")
    if np.any(feature_scale <= 0.0):
        raise ValueError("Every feature scale must be positive.")
    transform = np.zeros(
        (feature_mean.size + 1, feature_mean.size + 1), dtype=np.float64
    )
    transform[0, 0] = 1.0
    transform[0, 1:] = -feature_mean / feature_scale
    transform[1:, 1:] = np.diag(1.0 / feature_scale)
    return transform


def prepare_weighted_normal_equations(
    group_statistics,
    minimum_scale=1.0e-12,
):
    """Standardize reusable group shards once before a weight search.

    Purpose: avoid repeated O(p^3) coordinate transforms during weight search.
    Math: target scales and feature standardization use the unweighted training
    shards; linearity permits each group/target Gram to be transformed once.
    Input: group-to-statistics mapping.
    Output: transformed per-group energy/force components and normalization.
    Does not: apply group/target weights, ridge, or holdout information.
    """

    if not group_statistics:
        raise ValueError("At least one group statistics shard is required.")
    groups = tuple(sorted(group_statistics))
    aggregate = sum_linear_statistics(group_statistics[group] for group in groups)
    mean, scale = feature_normalization(aggregate, minimum_scale=minimum_scale)
    transform = standardization_transform(mean, scale)
    energy_count = int(aggregate["structure_count"])
    energy_mean = float(aggregate["energy_target_sum"]) / energy_count
    energy_variance = max(
        float(aggregate["energy_target_square"]) / energy_count
        - energy_mean * energy_mean,
        0.0,
    )
    energy_scale = max(np.sqrt(energy_variance), float(minimum_scale))
    force_scale = max(
        np.sqrt(
            float(aggregate["force_metric_target_square"])
            / int(aggregate["force_component_count"])
        ),
        float(minimum_scale),
    )
    components = {}
    for group in groups:
        shard = group_statistics[group]
        components[group] = {
            "structure_count": int(shard["structure_count"]),
            "energy_gram": transform.T @ np.asarray(shard["energy_gram"]) @ transform,
            "energy_rhs": transform.T @ np.asarray(shard["energy_rhs"]),
            "force_gram": transform.T @ np.asarray(shard["force_fit_gram"]) @ transform,
            "force_rhs": transform.T @ np.asarray(shard["force_fit_rhs"]),
        }
    return {
        "components": components,
        "runtime_from_fit_coordinates": transform,
        "feature_mean": mean,
        "feature_scale": scale,
        "energy_target_scale": float(energy_scale),
        "force_target_scale": float(force_scale),
    }


def assemble_prepared_normal_equations(
    prepared,
    group_weights=None,
    energy_weight=1.0,
    force_weight=1.0,
):
    """Combine prestandardized group shards for one candidate objective."""

    components = dict(prepared["components"])
    if not components:
        raise ValueError("Prepared equations contain no group components.")
    group_weights = {} if group_weights is None else dict(group_weights)
    first = components[next(iter(sorted(components)))]
    width = np.asarray(first["energy_rhs"]).size
    gram = np.zeros((width, width), dtype=np.float64)
    rhs = np.zeros(width, dtype=np.float64)
    effective_count = 0.0
    for group, component in components.items():
        weight = float(group_weights.get(group, 1.0))
        if weight <= 0.0:
            raise ValueError("Every group weight must be positive.")
        effective_count += weight * int(component["structure_count"])
    energy_weight = float(energy_weight)
    force_weight = float(force_weight)
    if energy_weight <= 0.0 or force_weight <= 0.0:
        raise ValueError("Energy and force weights must be positive.")
    energy_scale = float(prepared["energy_target_scale"])
    force_scale = float(prepared["force_target_scale"])
    for group, component in components.items():
        weight = float(group_weights.get(group, 1.0))
        energy_factor = energy_weight * weight / (
            effective_count * energy_scale * energy_scale
        )
        force_factor = force_weight * weight / (
            effective_count * force_scale * force_scale
        )
        gram += energy_factor * np.asarray(component["energy_gram"])
        rhs += energy_factor * np.asarray(component["energy_rhs"])
        gram += force_factor * np.asarray(component["force_gram"])
        rhs += force_factor * np.asarray(component["force_rhs"])
    return {
        "gram": gram,
        "rhs": rhs,
        "runtime_from_fit_coordinates": np.asarray(
            prepared["runtime_from_fit_coordinates"]
        ),
        "feature_mean": np.asarray(prepared["feature_mean"]),
        "feature_scale": np.asarray(prepared["feature_scale"]),
        "energy_target_scale": energy_scale,
        "force_target_scale": force_scale,
        "effective_structure_count": float(effective_count),
    }


def assemble_weighted_normal_equations(
    group_statistics,
    group_weights=None,
    energy_weight=1.0,
    force_weight=1.0,
    minimum_scale=1.0e-12,
):
    """Assemble standardized normal equations from reusable group shards."""

    prepared = prepare_weighted_normal_equations(
        group_statistics, minimum_scale=minimum_scale
    )
    return assemble_prepared_normal_equations(
        prepared,
        group_weights=group_weights,
        energy_weight=energy_weight,
        force_weight=force_weight,
    )


def solve_ridge_statistics(normal_equations, alpha, feature_penalty=None):
    """Solve one ridge problem and return fit and raw runtime coefficients.

    Purpose: make alpha and block-relative penalty searches matrix-only.
    Math: solve ``(G + alpha*diag(0,p)) beta = h``; the intercept is never
    penalised.  A least-squares fallback handles singular zero-alpha probes.
    Input: output of ``assemble_weighted_normal_equations`` and penalties.
    Output: fit-coordinate and raw-coordinate coefficient vectors.
    Does not: select alpha or score a holdout.
    """

    gram = np.asarray(normal_equations["gram"], dtype=np.float64)
    rhs = np.asarray(normal_equations["rhs"], dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or rhs.shape != (gram.shape[0],):
        raise ValueError("Normal equations have incompatible shapes.")
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative.")
    if feature_penalty is None:
        feature_penalty = np.ones(gram.shape[0] - 1, dtype=np.float64)
    feature_penalty = np.asarray(feature_penalty, dtype=np.float64)
    if feature_penalty.shape != (gram.shape[0] - 1,):
        raise ValueError("feature_penalty must contain one value per feature.")
    if np.any(feature_penalty <= 0.0):
        raise ValueError("Every feature penalty must be positive.")
    system = gram.copy()
    system[1:, 1:] += alpha * np.diag(feature_penalty)
    try:
        fit = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        fit = np.linalg.lstsq(system, rhs, rcond=1.0e-12)[0]
    transform = np.asarray(
        normal_equations["runtime_from_fit_coordinates"], dtype=np.float64
    )
    return {
        "fit_coefficients": fit,
        "runtime_coefficients": transform @ fit,
    }


def score_linear_statistics(statistics, runtime_coefficients):
    """Compute energy-per-atom and force-component RMSE from statistics.

    Purpose: score train, validation, test, group, or fold shards without X.
    Math: ``SSE = beta.T G beta - 2 beta.T h + y.T y``.
    Input: one aggregate statistics shard and raw runtime coefficients.
    Output: RMSE values and nonnegative SSE values.
    Does not: combine groups or choose a selection objective.
    """

    beta = np.asarray(runtime_coefficients, dtype=np.float64)
    width = int(statistics["feature_count"]) + 1
    if beta.shape != (width,):
        raise ValueError("runtime_coefficients has an incompatible shape.")

    def squared_error(gram_name, rhs_name, target_name, accumulation_count):
        extended_beta = np.asarray(beta, dtype=np.longdouble)
        gram = np.asarray(statistics[gram_name], dtype=np.longdouble)
        rhs = np.asarray(statistics[rhs_name], dtype=np.longdouble)
        quadratic = extended_beta @ gram @ extended_beta
        linear = 2.0 * (extended_beta @ rhs)
        target_square = np.longdouble(statistics[target_name])
        value = quadratic - linear + target_square
        operation_count = (
            4 * width * width + int(accumulation_count) + 32
        )
        unit_roundoff = np.finfo(np.float64).eps
        gamma = operation_count * unit_roundoff
        gamma = gamma / max(1.0 - gamma, 0.5)
        roundoff_bound = gamma * (
            abs(quadratic) + abs(linear) + abs(target_square)
        )
        if value < -roundoff_bound:
            raise FloatingPointError("Sufficient-statistics SSE is materially negative.")
        return max(float(value), 0.0), float(roundoff_bound)

    energy_sse, energy_roundoff = squared_error(
        "energy_gram",
        "energy_rhs",
        "energy_target_square",
        statistics["structure_count"],
    )
    force_sse, force_roundoff = squared_error(
        "force_metric_gram",
        "force_metric_rhs",
        "force_metric_target_square",
        statistics["force_component_count"],
    )
    return {
        "energy_sse": float(energy_sse),
        "force_sse": float(force_sse),
        "energy_sse_roundoff_bound": energy_roundoff,
        "force_sse_roundoff_bound": force_roundoff,
        "energy_rmse_eV_per_atom": float(
            np.sqrt(energy_sse / int(statistics["structure_count"]))
        ),
        "force_rmse_eV_per_A": float(
            np.sqrt(force_sse / int(statistics["force_component_count"]))
        ),
    }


__all__ = [
    "LINEAR_SUFFICIENT_STATISTICS_SCHEMA",
    "assemble_prepared_normal_equations",
    "assemble_weighted_normal_equations",
    "feature_normalization",
    "prepare_weighted_normal_equations",
    "score_linear_statistics",
    "select_linear_statistics",
    "solve_ridge_statistics",
    "standardization_transform",
    "structure_linear_statistics",
    "sum_linear_statistics",
]
