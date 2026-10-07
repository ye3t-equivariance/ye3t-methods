from types import SimpleNamespace

import numpy as np
import pytest


def _records():
    return [
        {
            "site_design": np.asarray([[1.0, 2.0]], dtype=np.float64),
            "energy_target": -2.0,
            "force_design": np.asarray(
                [[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
                dtype=np.float64,
            ),
            "force_target": np.asarray([0.5, -1.0, 0.25], dtype=np.float64),
        },
        {
            "site_design": np.asarray(
                [[3.0, 4.0], [5.0, 8.0]],
                dtype=np.float64,
            ),
            "energy_target": 6.0,
            "force_design": np.asarray(
                [
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [1.0, 1.0],
                    [-1.0, 0.5],
                    [0.0, -2.0],
                    [3.0, 1.0],
                ],
                dtype=np.float64,
            ),
            "force_target": np.asarray(
                [-0.5, 0.75, 1.0, -0.25, 0.5, -1.5],
                dtype=np.float64,
            ),
        },
    ]


def _objective():
    return {
        "kind": "structure_balanced_train_scaled_E1_F1",
        "feature_minimum_scale": 1.0e-12,
        "target_minimum_scale": 1.0e-12,
    }


@pytest.mark.fast
def test_structure_balanced_scaled_design_matches_independent_formula():
    from ye3t_methods.atomistic.ace.linear_ace import _structure_balanced_scaled_design

    records = _records()
    X, y, metadata = _structure_balanced_scaled_design(records, _objective())

    all_site = np.concatenate([record["site_design"] for record in records])
    mean = np.mean(all_site, axis=0)
    scale = np.std(all_site, axis=0)
    energy_scale = np.std(np.asarray([-2.0, 3.0], dtype=np.float64))
    all_force = np.concatenate([record["force_target"] for record in records])
    force_scale = np.sqrt(np.mean(all_force ** 2))
    expected_rows = []
    expected_targets = []
    for record in records:
        atom_count = record["site_design"].shape[0]
        energy_factor = 1.0 / (np.sqrt(2.0) * energy_scale)
        expected_rows.append(
            energy_factor
            * np.concatenate(
                ([1.0], (np.mean(record["site_design"], axis=0) - mean) / scale)
            )[None, :]
        )
        expected_targets.append(
            np.asarray(
                [energy_factor * record["energy_target"] / atom_count],
                dtype=np.float64,
            )
        )
        force_columns = record["force_design"] / scale
        force_factor = 1.0 / (
            np.sqrt(2.0 * force_columns.shape[0]) * force_scale
        )
        expected_rows.append(
            force_factor
            * np.column_stack(
                (np.zeros(force_columns.shape[0], dtype=np.float64), force_columns)
            )
        )
        expected_targets.append(force_factor * record["force_target"])

    np.testing.assert_allclose(X, np.concatenate(expected_rows), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(y, np.concatenate(expected_targets), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(metadata["feature_mean"], mean, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(metadata["feature_scale"], scale, rtol=0.0, atol=0.0)
    assert metadata["energy_scale"] == energy_scale
    assert metadata["force_scale"] == force_scale
    assert metadata["structure_count"] == 2
    assert metadata["atom_count"] == 3


@pytest.mark.fast
@pytest.mark.parametrize("alpha", [0.0, 0.25])
def test_augmented_svd_solver_matches_independent_lstsq_and_foldback(alpha):
    from ye3t_methods.atomistic.ace.linear_ace import (
        _solve_structure_balanced_scaled_ridge,
        _structure_balanced_scaled_design,
    )

    X, y, metadata = _structure_balanced_scaled_design(_records(), _objective())
    weight, bias, report = _solve_structure_balanced_scaled_ridge(
        X,
        y,
        metadata,
        alpha,
        1.0e-12,
    )
    if alpha > 0.0:
        penalty = np.zeros((2, 3), dtype=np.float64)
        penalty[:, 1:] = np.eye(2, dtype=np.float64)
        expected_X = np.concatenate((X, np.sqrt(alpha) * penalty), axis=0)
        expected_y = np.concatenate((y, np.zeros(2, dtype=np.float64)))
    else:
        expected_X = X
        expected_y = y
    beta = np.linalg.lstsq(expected_X, expected_y, rcond=1.0e-12)[0]
    expected_weight = beta[1:] / np.asarray(metadata["feature_scale"])
    expected_bias = beta[0] - np.dot(metadata["feature_mean"], expected_weight)

    np.testing.assert_allclose(weight, expected_weight, rtol=0.0, atol=0.0)
    assert bias == expected_bias
    np.testing.assert_allclose(report["scaled_coefficients"], beta, rtol=0.0, atol=0.0)
    assert report["intercept_penalized"] is False
    assert report["solver"] == "numpy.linalg.lstsq_augmented_design"


@pytest.mark.fast
def test_scaled_objective_rejects_unknown_fields_and_invalid_shapes():
    from ye3t_methods.atomistic.ace.linear_ace import (
        _normalize_linear_fit_objective,
        _structure_balanced_scaled_design,
    )

    with pytest.raises(ValueError, match="Unsupported fit_objective fields"):
        _normalize_linear_fit_objective(
            {"kind": "structure_balanced_train_scaled_E1_F1", "mystery": 1}
        )
    records = _records()
    records[0] = dict(records[0], force_design=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="force-design shape"):
        _structure_balanced_scaled_design(records, _objective())


@pytest.mark.fast
@pytest.mark.parametrize("alpha", [0.0, 0.25])
def test_streamed_scaled_normal_matches_dense_objective(alpha):
    from ye3t_methods.atomistic.ace.linear_ace import (
        _accumulate_structure_balanced_scaled_normal,
        _solve_structure_balanced_scaled_ridge,
        _solve_structure_balanced_scaled_ridge_from_normal,
        _structure_balanced_scale_metadata,
        _structure_balanced_scaled_design,
    )

    records = _records()
    all_site = np.concatenate([record["site_design"] for record in records])
    force_targets = np.concatenate([record["force_target"] for record in records])
    metadata = _structure_balanced_scale_metadata(
        feature_sum=np.sum(all_site, axis=0),
        feature_square_sum=np.sum(all_site * all_site, axis=0),
        atom_count=all_site.shape[0],
        energy_per_atom=[
            record["energy_target"] / record["site_design"].shape[0]
            for record in records
        ],
        force_square_sum=np.dot(force_targets, force_targets),
        force_component_count=force_targets.size,
        objective=_objective(),
    )
    normal = {
        "XtX": np.zeros((3, 3), dtype=np.float64),
        "Xty": np.zeros((3,), dtype=np.float64),
        "yty": 0.0,
    }
    for record in records:
        normal["yty"] = _accumulate_structure_balanced_scaled_normal(
            normal["XtX"],
            normal["Xty"],
            normal["yty"],
            **record,
            scale_metadata=metadata,
            structure_weight=1.0,
            energy_weight=1.0,
            force_weight=1.0,
        )
    X, y, dense_metadata = _structure_balanced_scaled_design(records, _objective())
    np.testing.assert_allclose(normal["XtX"], X.T @ X, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(normal["Xty"], X.T @ y, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(normal["yty"], y @ y, rtol=1.0e-14, atol=1.0e-14)

    streamed_weight, streamed_bias, report = (
        _solve_structure_balanced_scaled_ridge_from_normal(
            normal,
            metadata,
            alpha,
            maximum_condition=1.0e12,
        )
    )
    dense_weight, dense_bias, _ = _solve_structure_balanced_scaled_ridge(
        X,
        y,
        dense_metadata,
        alpha,
        1.0e-12,
    )
    np.testing.assert_allclose(streamed_weight, dense_weight, rtol=1.0e-11, atol=1.0e-11)
    assert streamed_bias == pytest.approx(dense_bias, rel=1.0e-11, abs=1.0e-11)
    assert report["solver"] == "numpy.linalg.solve_streamed_gram"


@pytest.mark.fast
def test_streamed_scaled_normal_applies_structure_weights_to_complete_blocks():
    from ye3t_methods.atomistic.ace.linear_ace import (
        _accumulate_structure_balanced_scaled_normal,
        _structure_balanced_scale_metadata,
        _structure_balanced_scaled_design,
    )

    records = _records()
    all_site = np.concatenate([record["site_design"] for record in records])
    force_targets = np.concatenate([record["force_target"] for record in records])
    metadata = _structure_balanced_scale_metadata(
        feature_sum=np.sum(all_site, axis=0),
        feature_square_sum=np.sum(all_site * all_site, axis=0),
        atom_count=all_site.shape[0],
        energy_per_atom=[-2.0, 3.0],
        force_square_sum=np.dot(force_targets, force_targets),
        force_component_count=force_targets.size,
        objective=_objective(),
    )
    normal = {
        "XtX": np.zeros((3, 3), dtype=np.float64),
        "Xty": np.zeros((3,), dtype=np.float64),
        "yty": 0.0,
    }
    weights = (0.5, 1.5)
    for record, weight in zip(records, weights):
        normal["yty"] = _accumulate_structure_balanced_scaled_normal(
            normal["XtX"],
            normal["Xty"],
            normal["yty"],
            **record,
            scale_metadata=metadata,
            structure_weight=weight,
            energy_weight=1.0,
            force_weight=1.0,
        )
    X, y, _ = _structure_balanced_scaled_design(records, _objective())
    row_weights = []
    for record, weight in zip(records, weights):
        row_weights.extend([weight] * (1 + record["force_target"].size))
    weighted_X = np.sqrt(np.asarray(row_weights))[:, None] * X
    weighted_y = np.sqrt(np.asarray(row_weights)) * y
    np.testing.assert_allclose(normal["XtX"], weighted_X.T @ weighted_X)
    np.testing.assert_allclose(normal["Xty"], weighted_X.T @ weighted_y)
    np.testing.assert_allclose(normal["yty"], weighted_y @ weighted_y)


@pytest.mark.fast
def test_ard_fit_records_a_usable_posterior(monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    del sklearn
    from ye3t_methods.atomistic.ace import linear_ace

    rng = np.random.default_rng(914)
    X = rng.normal(size=(40, 3))
    y = X @ np.asarray([1.5, -0.25, 0.75]) + 0.01 * rng.normal(size=40)

    monkeypatch.setattr(
        linear_ace,
        "_descriptor_specs_from_settings",
        lambda **kwargs: (
            (SimpleNamespace(key="d0"), SimpleNamespace(key="d1")),
            {},
        ),
    )
    monkeypatch.setattr(
        linear_ace,
        "build_linear_ace_regression_problem",
        lambda *args, **kwargs: (X, y, {}),
    )

    bundle = linear_ace.fit_linear_ace(
        [SimpleNamespace(info={})],
        settings=SimpleNamespace(L_R=0),
        site_basis_config=SimpleNamespace(),
        type_map={"Ta": 0},
        cutoff=4.0,
        fit_method="ardregression",
        use_descriptor_cache=False,
    )
    posterior = bundle.fit_metadata["predictive_uncertainty"]
    active = posterior["active_column_indices"]
    covariance = np.asarray(posterior["coefficient_covariance_active"])

    assert posterior["status"] == "python_offline_only"
    assert posterior["design_column_order"] == (
        "descriptor_features_then_atom_count_bias"
    )
    assert covariance.shape == (len(active), len(active))
    assert posterior["noise_precision"] > 0.0
    assert len(posterior["coefficient_precision"]) == X.shape[1]
