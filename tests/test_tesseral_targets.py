import numpy as np
import pytest
import torch

from ye3t.core.rotation import wigner_D_numeric
from ye3t.core.tesseral import real_tesseral_to_complex_multiplet
from ye3t_methods.tesseral_targets import (
    cartesian_to_real_tesseral, cartesian_tesseral_convention_hash,
    real_tesseral_to_cartesian,
)


def test_polar_vector_cartesian_tesseral_axes_and_generic_rotation(monkeypatch):
    axes = np.eye(3)
    np.testing.assert_array_equal(cartesian_to_real_tesseral(axes, 1, "odd"),
                                  [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    vector = np.array([[.4, -.7, .2], [1.1, .3, -.6]])
    tesseral = cartesian_to_real_tesseral(vector, 1, "odd")
    np.testing.assert_allclose(real_tesseral_to_cartesian(tesseral, 1, "odd"), vector)
    np.testing.assert_allclose(np.linalg.norm(tesseral, axis=-1),
                               np.linalg.norm(vector, axis=-1))
    cy, sy, cz, sz = np.cos(.41), np.sin(.41), np.cos(.29), np.sin(.29)
    rotation = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    turned = cartesian_to_real_tesseral(vector @ rotation.T, 1, "odd")
    np.testing.assert_allclose(
        real_tesseral_to_complex_multiplet(torch.as_tensor(turned), 1).numpy(),
        real_tesseral_to_complex_multiplet(torch.as_tensor(tesseral), 1).numpy()
        @ wigner_D_numeric(1, rotation).T, rtol=1e-10, atol=1e-11)
    baseline_hash = cartesian_tesseral_convention_hash(1, "odd")
    assert baseline_hash == cartesian_tesseral_convention_hash(1, "odd")
    monkeypatch.setattr("ye3t.core.tesseral.real_tesseral_to_complex_multiplet",
                        lambda values, L: -real_tesseral_to_complex_multiplet(values, L))
    assert cartesian_tesseral_convention_hash(1, "odd") != baseline_hash
    with pytest.raises(ValueError, match="polar L=1 odd"):
        cartesian_to_real_tesseral(vector, 1, "even")


def test_traceless_quadrupole_cartesian_tesseral_norm_rotation_and_rejection():
    raw = np.array([[[.7, .2, -.3], [.2, -.5, .4], [-.3, .4, -.2]],
                    [[-.2, .6, .1], [.6, .8, -.2], [.1, -.2, -.6]]])
    assert np.allclose(np.trace(raw, axis1=-2, axis2=-1), 0)
    tesseral = cartesian_to_real_tesseral(raw, 2, "even")
    np.testing.assert_allclose(real_tesseral_to_cartesian(tesseral, 2, "even"), raw,
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(tesseral, axis=-1),
                               np.linalg.norm(raw, axis=(-2, -1)), rtol=0, atol=1e-15)
    angle = .39
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    turned = cartesian_to_real_tesseral(rotation @ raw @ rotation.T, 2, "even")
    np.testing.assert_allclose(
        real_tesseral_to_complex_multiplet(torch.as_tensor(turned), 2).numpy(),
        real_tesseral_to_complex_multiplet(torch.as_tensor(tesseral), 2).numpy()
        @ wigner_D_numeric(2, rotation).T, rtol=1e-10, atol=1e-11)
    bad = raw.copy()
    bad[0, 0, 1] += .01
    with pytest.raises(ValueError, match="symmetric and traceless"):
        cartesian_to_real_tesseral(bad, 2, "even")
    bad = raw.copy()
    bad[0, 0, 0] += .01
    with pytest.raises(ValueError, match="symmetric and traceless"):
        cartesian_to_real_tesseral(bad, 2, "even")
    badly_scaled = np.stack((raw[0] * 1e9, raw[1]))
    badly_scaled[1, 0, 0] += .01
    with pytest.raises(ValueError, match="symmetric and traceless"):
        cartesian_to_real_tesseral(badly_scaled, 2, "even")
    with pytest.raises(ValueError, match="traceless L=2 even"):
        real_tesseral_to_cartesian(tesseral, 2, "odd")
