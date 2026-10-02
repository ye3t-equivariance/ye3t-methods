import hashlib

import numpy as np
import pytest


def _records():
    from ye3t_ace.linear_statistics import structure_linear_statistics

    rng = np.random.default_rng(9127)
    records = []
    dense = []
    for atom_count in (2, 3, 4, 2, 5):
        site = rng.normal(size=(atom_count, 4))
        force_design = rng.normal(size=(3 * atom_count, 4))
        energy_target = float(rng.normal())
        force_target = rng.normal(size=3 * atom_count)
        records.append(
            structure_linear_statistics(
                np.sum(site, axis=0),
                np.sum(site * site, axis=0),
                force_design,
                atom_count,
                energy_target,
                force_target,
            )
        )
        dense.append((site, force_design, energy_target, force_target))
    return records, dense


def test_statistics_match_dense_weighted_ridge_and_score():
    from ye3t_ace.linear_statistics import (
        assemble_weighted_normal_equations,
        score_linear_statistics,
        solve_ridge_statistics,
        sum_linear_statistics,
    )

    records, dense = _records()
    groups = {
        "bulk": sum_linear_statistics(records[:3]),
        "surface": sum_linear_statistics(records[3:]),
    }
    equations = assemble_weighted_normal_equations(groups)
    aggregate = sum_linear_statistics(records)
    mean = equations["feature_mean"]
    scale = equations["feature_scale"]
    energy_values = np.asarray([record[2] for record in dense])
    force_values = np.concatenate([record[3] for record in dense])
    energy_scale = max(float(np.std(energy_values)), 1.0e-12)
    force_scale = max(float(np.sqrt(np.mean(force_values * force_values))), 1.0e-12)
    design = []
    targets = []
    structure_count = len(dense)
    for site, force_design, energy_target, force_target in dense:
        atom_count = site.shape[0]
        energy_row = np.concatenate(
            (np.ones(1), (np.mean(site, axis=0) - mean) / scale)
        )
        force_rows = np.concatenate(
            (np.zeros((3 * atom_count, 1)), force_design / scale), axis=1
        )
        design.append(energy_row[None, :] / (np.sqrt(structure_count) * energy_scale))
        targets.append(np.asarray([energy_target / (np.sqrt(structure_count) * energy_scale)]))
        design.append(
            force_rows
            / (np.sqrt(structure_count * 3 * atom_count) * force_scale)
        )
        targets.append(
            force_target
            / (np.sqrt(structure_count * 3 * atom_count) * force_scale)
        )
    design = np.concatenate(design)
    targets = np.concatenate(targets)
    np.testing.assert_allclose(equations["gram"], design.T @ design, atol=2.0e-14)
    np.testing.assert_allclose(equations["rhs"], design.T @ targets, atol=2.0e-14)

    penalty = np.asarray([1.0, 0.5, 2.0, 3.0])
    alpha = 3.0e-4
    expected_system = design.T @ design
    expected_system[1:, 1:] += alpha * np.diag(penalty)
    expected_fit = np.linalg.solve(expected_system, design.T @ targets)
    solution = solve_ridge_statistics(equations, alpha, penalty)
    np.testing.assert_allclose(solution["fit_coefficients"], expected_fit, atol=2.0e-12)

    runtime = solution["runtime_coefficients"]
    energy_error = []
    force_error = []
    for site, force_design, energy_target, force_target in dense:
        energy_prediction = runtime[0] + np.mean(site, axis=0) @ runtime[1:]
        force_prediction = force_design @ runtime[1:]
        energy_error.append(energy_prediction - energy_target)
        force_error.append(force_prediction - force_target)
    metric = score_linear_statistics(aggregate, runtime)
    np.testing.assert_allclose(
        metric["energy_rmse_eV_per_atom"],
        np.sqrt(np.mean(np.square(energy_error))),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        metric["force_rmse_eV_per_A"],
        np.sqrt(np.mean(np.concatenate(force_error) ** 2)),
        atol=2.0e-12,
    )


def test_group_weights_use_cached_shards_without_changing_normalization():
    from ye3t_ace.linear_statistics import (
        assemble_weighted_normal_equations,
        sum_linear_statistics,
    )

    records, _dense = _records()
    groups = {
        "bulk": sum_linear_statistics(records[:3]),
        "surface": sum_linear_statistics(records[3:]),
    }
    baseline = assemble_weighted_normal_equations(groups)
    weighted = assemble_weighted_normal_equations(
        groups,
        group_weights={"bulk": 1.0, "surface": 4.0},
        energy_weight=2.0,
        force_weight=0.5,
    )
    np.testing.assert_array_equal(weighted["feature_mean"], baseline["feature_mean"])
    np.testing.assert_array_equal(weighted["feature_scale"], baseline["feature_scale"])
    assert not np.allclose(weighted["gram"], baseline["gram"])


def test_generic_statistics_cache_round_trip_and_corruption_recovery(tmp_path):
    from ye3t_ace.cache import (
        LINEAR_STATISTICS_CACHE_SCHEMA,
        LinearCacheValidationError,
        load_linear_sufficient_statistics,
        persist_linear_sufficient_statistics,
    )

    request_hash = hashlib.sha256(b"request-v1").hexdigest()
    cache_hash = hashlib.sha256(b"content-v1").hexdigest()
    cache = {
        "schema": LINEAR_STATISTICS_CACHE_SCHEMA,
        "request_hash": request_hash,
        "cache_hash": cache_hash,
        "metadata": {"geometry_rows": "geometry-v1", "targets": "zbl-v1"},
        "arrays": {
            "bulk_energy_gram": np.arange(16, dtype=np.float64).reshape(4, 4),
            "bulk_energy_rhs": np.linspace(-1.0, 1.0, 4),
        },
    }
    directory = persist_linear_sufficient_statistics(cache, tmp_path)
    loaded = load_linear_sufficient_statistics(request_hash, tmp_path)
    assert isinstance(loaded["arrays"]["bulk_energy_gram"], np.memmap)
    assert loaded["metadata"]["geometry_rows"] == "geometry-v1"
    np.testing.assert_array_equal(
        loaded["arrays"]["bulk_energy_rhs"], cache["arrays"]["bulk_energy_rhs"]
    )
    path = directory / "bulk_energy_rhs.npy"
    corrupted = bytearray(path.read_bytes())
    corrupted[-1] ^= 1
    path.write_bytes(corrupted)
    with pytest.raises(LinearCacheValidationError, match="hash mismatch"):
        load_linear_sufficient_statistics(request_hash, tmp_path)
    persist_linear_sufficient_statistics(cache, tmp_path)
    recovered = load_linear_sufficient_statistics(request_hash, tmp_path)
    np.testing.assert_array_equal(
        recovered["arrays"]["bulk_energy_rhs"], cache["arrays"]["bulk_energy_rhs"]
    )
