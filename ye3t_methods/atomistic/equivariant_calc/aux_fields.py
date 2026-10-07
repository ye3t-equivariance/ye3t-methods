
"""Auxiliary basis functions for non-spatial per-atom vector fields."""

import torch

from .trc_sph_harm import spherical_harmonics_l


def magnetic_orientation_basis(
    magnetic_vectors,
    edge_index,
    lmax,
    *,
    dtype = torch.float64,
    complex_dtype = torch.complex128,
):
    """Return per-edge spherical-harmonic basis values for vector moments.

    The basis is evaluated on the neighbor-site magnetic moment directions
    ``m_j / |m_j|``. For zero-magnitude vectors, ``l=0`` returns 1 and all higher
    ``l`` channels return 0.
    """
    if magnetic_vectors.ndim != 2 or magnetic_vectors.shape[1] != 3:
        raise ValueError(
            f"magnetic_vectors must have shape [n_atoms, 3] for non-collinear moments; got {tuple(magnetic_vectors.shape)}"
        )
    device = magnetic_vectors.device
    neighs = edge_index[1].to(device=device)
    vec = magnetic_vectors.to(dtype=dtype, device=device)[neighs]
    mag = torch.linalg.norm(vec, dim=-1)
    safe_mag = torch.clamp(mag, min=1.0e-12)
    unit = vec / safe_mag.unsqueeze(-1)
    theta = torch.arccos(torch.clamp(unit[:, 2], -1.0 + 1.0e-12, 1.0 - 1.0e-12))
    phi = torch.atan2(unit[:, 1], unit[:, 0])
    zero_mask = mag <= 1.0e-12

    out = {}
    for l in range(lmax + 1):
        harmonics = spherical_harmonics_l(l, theta, phi).to(complex_dtype)
        for m in range(-l, l + 1):
            values = harmonics[m + l]
            if torch.any(zero_mask):
                fill = torch.ones_like(values) if l == 0 and m == 0 else torch.zeros_like(values)
                values = torch.where(zero_mask, fill, values)
            out[(l, m)] = values
    return out
