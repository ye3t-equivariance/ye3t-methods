"""Custom one-factor sources must enter values and Cartesian derivatives."""

import pytest
import torch

from ye3t_methods.atomistic.equivariant_calc.labeling import SingleChannelLabel
from ye3t_methods.atomistic.equivariant_calc.site_basis_v2 import (
    DefaultRadialBasisProvider,
    RadialBasisProvider,
    SiteBasisConfig,
    SiteBasisV2,
)


class ScaledRadialProvider(RadialBasisProvider):
    def __init__(self, factor=2.0):
        self.default = DefaultRadialBasisProvider()
        self.factor = float(factor)

    def evaluate(self, *, n, l, bond_idx, r, evaluator):
        return self.factor * self.default.evaluate(n=n, l=l, bond_idx=bond_idx,
                                         r=r, evaluator=evaluator)

    def derivative(self, *, n, l, bond_idx, r, evaluator):
        return self.factor * self.default.derivative(n=n, l=l, bond_idx=bond_idx,
                                           r=r, evaluator=evaluator)

    def max_abs(self, *, n, l, evaluator):
        return self.factor * self.default.max_abs(n=n, l=l, evaluator=evaluator)

    def convention_metadata(self, *, evaluator):
        return {"provider": "test_scaled_radial_v1", "factor": self.factor,
                "base": self.default.convention_metadata(evaluator=evaluator)}


def test_custom_radial_provider_scales_l0_through_l3_values_and_derivatives():
    config = SiteBasisConfig(
        rc=[3.5], lmbda=[0.35], nradmax=2, lmax=3,
        possible_types=(0,), charge_mode="none",
        atomic_base_normalization="none", factor_normalization="none",
        spherical_backend="complex", source_backend="torch",
        dtype=torch.float64, complex_dtype=torch.complex128,
    )
    channels = tuple(
        SingleChannelLabel(mu0=0, mu=0, kappa0=0, kappa=0,
                           n=1, l=degree, m=magnetic)
        for degree in range(4)
        for magnetic in range(-degree, degree + 1)
    )
    edges = torch.tensor([[0.31, 0.47, 0.83], [-0.42, 0.58, 0.27],
                          [0.73, -0.24, 0.51], [-0.67, -0.19, 0.37]],
                         dtype=torch.float64)
    edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 0, 2]], dtype=torch.long)
    atom_types = torch.zeros(3, dtype=torch.long)
    baseline = SiteBasisV2(config)
    scaled = SiteBasisV2(config, radial_provider=ScaledRadialProvider())

    original = baseline.compute_channel_edges_with_dx(
        edges, edge_index, atom_types, channels)
    modified = scaled.compute_channel_edges_with_dx(
        edges, edge_index, atom_types, channels)
    assert original[0] == modified[0]
    torch.testing.assert_close(modified[1], 2 * original[1], rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(modified[2], 2 * original[2], rtol=2e-12, atol=2e-12)
    assert torch.linalg.norm(original[1][:, -7:]) > 0
    step = 1e-6
    plus = edges.clone()
    minus = edges.clone()
    plus[0, 0] += step
    minus[0, 0] -= step
    value_plus = scaled.compute_channel_edges_with_dx(
        plus, edge_index, atom_types, channels)[1][0, -1]
    value_minus = scaled.compute_channel_edges_with_dx(
        minus, edge_index, atom_types, channels)[1][0, -1]
    torch.testing.assert_close((value_plus - value_minus) / (2 * step),
                               modified[2][0, -1, 0], rtol=2e-6, atol=2e-8)
    _, original_raw, original_final = baseline.compute_channels_raw_and_final(
        edges, edge_index, atom_types, channels)
    _, scaled_raw, scaled_final = scaled.compute_channels_raw_and_final(
        edges, edge_index, atom_types, channels)
    torch.testing.assert_close(scaled_raw, 2 * original_raw, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(scaled_final, 2 * original_final,
                               rtol=2e-13, atol=2e-13)
    assert scaled._radial_convention_metadata()["provider"] == "test_scaled_radial_v1"


def test_custom_radial_identity_is_stable_and_rejects_unbound_sources():
    config = SiteBasisConfig(
        rc=[3.5], lmbda=[0.35], nradmax=2, lmax=3,
        possible_types=(0,), charge_mode="none",
        atomic_base_normalization="none", factor_normalization="none",
        spherical_backend="complex", source_backend="torch",
        dtype=torch.float64, complex_dtype=torch.complex128,
    )
    first = SiteBasisV2(config, radial_provider=ScaledRadialProvider())
    second = SiteBasisV2(config, radial_provider=ScaledRadialProvider())
    changed = SiteBasisV2(config, radial_provider=ScaledRadialProvider(3.0))
    assert first._radial_runtime_cache_identity() == second._radial_runtime_cache_identity()
    assert first._radial_runtime_cache_identity() != changed._radial_runtime_cache_identity()

    class UnidentifiedRadial(RadialBasisProvider):
        pass

    with pytest.raises(NotImplementedError, match="stable JSON source metadata"):
        SiteBasisV2(config, radial_provider=UnidentifiedRadial())._radial_runtime_cache_identity()

    class InvalidMetadata(ScaledRadialProvider):
        def convention_metadata(self, *, evaluator):
            metadata = super().convention_metadata(evaluator=evaluator)
            metadata["factor"] = float("nan")
            return metadata

    with pytest.raises(ValueError, match="finite JSON data"):
        SiteBasisV2(config, radial_provider=InvalidMetadata())._radial_runtime_cache_identity()
