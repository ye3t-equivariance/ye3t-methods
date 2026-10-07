import pytest
import torch

from ye3t.core.labels import SingleChannelLabel
from ye3t_methods.atomistic.equivariant_calc import SiteBasisConfig, SiteBasisV2
from ye3t_methods.atomistic.equivariant_calc.site_basis_serialization import (
    deserialize_site_basis_config, serialize_site_basis_config,
)


def test_fixed_embedding_atomic_base_and_position_derivative_match_delta_reference():
    matrix = ((1.0, 0.0), (0.0, 1.0), (0.5, -0.25))
    source = {
        "rc": 3.5, "lmbda": 0.79, "nradmax": 1, "lmax": 0,
        "possible_types": (0, 1, 2), "radial_basis": "PACE_ChebExpCos",
        "source_backend": "torch", "factor_normalization": "none",
        "atomic_base_normalization": "none", "spherical_backend": "complex",
        "spherical_normalization": "pace_y00_one",
        "pace_cutoff_width": 0.01, "pace_spline_spacing": 0.001,
    }
    embedded_config = SiteBasisConfig(
        **source, chemical_basis="fixed_embedding", chemical_embedding=matrix)
    delta_config = SiteBasisConfig(**source, chemical_basis="delta")
    embedded = SiteBasisV2(embedded_config)
    delta = SiteBasisV2(delta_config)
    embedded_channels = tuple(
        SingleChannelLabel(mu0=center, mu=column, kappa0=0, kappa=0,
                           n=1, l=0, m=0)
        for center in range(3) for column in range(2))
    delta_channels = tuple(
        SingleChannelLabel(mu0=center, mu=neighbor, kappa0=0, kappa=0,
                           n=1, l=0, m=0)
        for center in range(3) for neighbor in range(3))
    edge_index = torch.tensor(((0, 0, 0, 1, 1), (1, 2, 3, 0, 2)))
    atom_types = torch.tensor((0, 1, 2, 1))
    displacements = torch.tensor(
        ((1.3, 0.1, 0.2), (0.2, 1.5, 0.3), (-1.2, 0.2, 0.4),
         (-1.3, -0.1, -0.2), (-1.1, 1.4, 0.1)),
        dtype=torch.float64, requires_grad=True)
    _, embedded_A = embedded.compute_site_basis(
        displacements, edge_index, atom_types, embedded_channels)
    _, delta_A = delta.compute_site_basis(
        displacements, edge_index, atom_types, delta_channels)
    expected = torch.einsum(
        "acn,nk->ack", delta_A.reshape(4, 3, 3),
        torch.tensor(matrix, dtype=torch.complex128)).reshape(4, 6)
    torch.testing.assert_close(embedded_A, expected, atol=1e-12, rtol=0)
    assert torch.linalg.norm(embedded_A).item() > 1e-8
    embedded_gradient = torch.autograd.grad(embedded_A[:, 1].real.sum(),
                                           displacements, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected[:, 1].real.sum(),
                                           displacements)[0]
    torch.testing.assert_close(embedded_gradient, expected_gradient,
                               atol=1e-11, rtol=0)
    _, edge_values, edge_derivative = embedded.compute_channel_edges_with_dx(
        displacements, edge_index, atom_types, embedded_channels)
    _, delta_edge_values, delta_edge_derivative = delta.compute_channel_edges_with_dx(
        displacements, edge_index, atom_types, delta_channels)
    projected_edge_values = torch.einsum(
        "ecn,nk->eck", delta_edge_values.reshape(5, 3, 3),
        torch.tensor(matrix, dtype=torch.complex128)).reshape(5, 6)
    projected_edge_derivative = torch.einsum(
        "ecnd,nk->eckd", delta_edge_derivative.reshape(5, 3, 3, 3),
        torch.tensor(matrix, dtype=torch.complex128)).reshape(5, 6, 3)
    torch.testing.assert_close(edge_values, projected_edge_values, atol=1e-12, rtol=0)
    torch.testing.assert_close(edge_derivative, projected_edge_derivative,
                               atol=1e-11, rtol=0)
    direct_gradient = torch.autograd.grad(edge_values[:, 1].real.sum(),
                                          displacements)[0]
    torch.testing.assert_close(edge_derivative[:, 1].real, direct_gradient,
                               atol=1e-11, rtol=0)
    restored = deserialize_site_basis_config(serialize_site_basis_config(embedded_config))
    assert restored.chemical_embedding == matrix
    assert SiteBasisV2(restored)._chemical_runtime_cache_identity() == (
        embedded._chemical_runtime_cache_identity())
    changed = SiteBasisConfig(
        **source, chemical_basis="fixed_embedding",
        chemical_embedding=((1.0, 0.0), (0.0, 1.0), (0.4, -0.25)))
    assert SiteBasisV2(changed)._chemical_runtime_cache_identity() != (
        embedded._chemical_runtime_cache_identity())


def test_fixed_embedding_rejects_invalid_matrix_and_unsupported_source_backend():
    legacy = SiteBasisConfig(3.0, 0.4, 1, 0, 0, (0,), "ChebExpCos",
                             "delta", "none")
    assert legacy.charge_mode == "none" and legacy.chemical_embedding is None
    del legacy.chemical_embedding
    assert "chemical_embedding" not in serialize_site_basis_config(legacy)
    assert SiteBasisV2(legacy)._chemical_runtime_cache_identity()[-1] is None
    with pytest.raises(ValueError, match="independent columns"):
        SiteBasisConfig(rc=3.0, lmbda=0.4, nradmax=1, lmax=0,
                        possible_types=(0, 1), chemical_basis="fixed_embedding",
                        chemical_embedding=((1.0, 2.0), (0.0, 0.0)),
                        source_backend="torch")
    with pytest.raises(NotImplementedError, match="source_backend='torch'"):
        SiteBasisConfig(rc=3.0, lmbda=0.4, nradmax=1, lmax=0,
                        possible_types=(0, 1), chemical_basis="fixed_embedding",
                        chemical_embedding=((1.0,), (0.5,)),
                        source_backend="auto")
