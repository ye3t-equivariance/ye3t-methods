"""Scaled solver coordinates preserve physical means and ARD covariance."""
import numpy as np
import pytest

from ye3t_methods.linear import _fit_sklearn_design


@pytest.mark.parametrize('method', ['lasso', 'ardregression'])
def test_rms_solver_coordinates_match_independent_sklearn(method):
    sklearn = pytest.importorskip('sklearn.linear_model', reason='requires optional fitting extra')
    rng = np.random.default_rng(41)
    X = rng.normal(size=(180, 5)) * np.array([1e-6, .02, 1., 1e4, 1e7])
    X[:, -1] = 0
    y = X[:, 0]*2e6 - X[:, 3]*3e-4 + rng.normal(scale=.01, size=180)
    options = {'alpha': 1e-4, 'max_iter': 10000} if method == 'lasso' else {'max_iter': 500}
    beta, metadata = _fit_sklearn_design(
        [(X[:100], y[:100]), (X[100:], y[100:])], method,
        {**options, 'column_scaling': 'rms'}, 'test_columns')
    scales = np.sqrt(np.mean(X*X, axis=0))
    scales[scales == 0] = 1
    estimator = (sklearn.Lasso if method == 'lasso' else sklearn.ARDRegression)(
        fit_intercept=False, **options).fit(X/scales, y)
    np.testing.assert_allclose(X @ beta, estimator.predict(X/scales), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(metadata['design_scaling']['column_scales'], scales, rtol=1e-14)
    assert np.sqrt(np.mean((X @ beta-y)**2)) < .02
    if method == 'ardregression':
        posterior = metadata['predictive_uncertainty']
        active = posterior['active_column_indices']
        covariance = np.asarray(posterior['coefficient_covariance_active'])
        physical_variance = np.einsum('ij,jk,ik->i', X[:,active], covariance, X[:,active])
        _, std = estimator.predict(X/scales, return_std=True)
        np.testing.assert_allclose(physical_variance + 1/estimator.alpha_, std**2,
                                   atol=1e-12, rtol=1e-9)
        assert posterior['coefficient_precision_coordinates'] == 'rms_scaled_design'


def test_design_scaling_rejects_unknown_policy():
    with pytest.raises(ValueError, match='column_scaling'):
        _fit_sklearn_design([(np.eye(2), np.ones(2))], 'lasso',
                            {'column_scaling': 'center'}, 'test_columns')
