"""Cartesian target adapters for the signed-M real-tesseral convention."""

import hashlib
import json
from math import sqrt

import numpy as np


def _require_cartesian_sector(L, parity):
    if (L, parity) not in ((1, "odd"), (2, "even")):
        raise ValueError("Cartesian targets require polar L=1 odd or traceless L=2 even.")


def cartesian_to_real_tesseral(values, L, parity):
    """Map polar vectors or symmetric traceless tensors to signed-M order.

    Algorithmic reference: DLMF 14.30 Condon–Shortley spherical harmonics;
    phases are fixed by ye3t.core.tesseral. Independent implementation.
    """
    _require_cartesian_sector(L, parity)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Cartesian targets must be finite.")
    if L == 1:
        if values.ndim < 1 or values.shape[-1] != 3:
            raise ValueError("Polar vector targets need a final Cartesian axis of length three.")
        return np.stack((values[..., 0], values[..., 2], -values[..., 1]), axis=-1)
    if values.ndim < 2 or values.shape[-2:] != (3, 3):
        raise ValueError("Quadrupole targets need final Cartesian axes (3, 3).")
    scale = np.maximum(1.0, np.max(np.abs(values), axis=(-2, -1), initial=0.0))
    asymmetry = np.max(np.abs(values - np.swapaxes(values, -2, -1)),
                       axis=(-2, -1), initial=0.0)
    trace = np.abs(np.trace(values, axis1=-2, axis2=-1))
    if np.any(asymmetry > 1e-10 * scale) or np.any(trace > 1e-10 * scale):
        raise ValueError("Quadrupole targets must be symmetric and traceless.")
    xx, yy, zz = values[..., 0, 0], values[..., 1, 1], values[..., 2, 2]
    return np.stack((
        (xx - yy) / sqrt(2), sqrt(2) * values[..., 0, 2],
        (2 * zz - xx - yy) / sqrt(6), -sqrt(2) * values[..., 1, 2],
        -sqrt(2) * values[..., 0, 1],
    ), axis=-1)


def real_tesseral_to_cartesian(values, L, parity):
    """Invert the declared orthonormal polar-vector or STF-tensor map."""
    _require_cartesian_sector(L, parity)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2 * L + 1 or not np.isfinite(values).all():
        raise ValueError("Real-tesseral values need finite complete signed-M components.")
    if L == 1:
        return np.stack((values[..., 0], -values[..., 2], values[..., 1]), axis=-1)
    zz = sqrt(2 / 3) * values[..., 2]
    xx = (-zz + sqrt(2) * values[..., 0]) / 2
    yy = (-zz - sqrt(2) * values[..., 0]) / 2
    xy, yz, xz = -values[..., 4] / sqrt(2), -values[..., 3] / sqrt(2), values[..., 1] / sqrt(2)
    return np.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), axis=-1).reshape(
        values.shape[:-1] + (3, 3))


def cartesian_tesseral_convention_hash(L, parity):
    """Bind a saved target to the exact Cartesian inverse axes and phases."""
    import torch
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    basis = real_tesseral_to_cartesian(np.eye(2 * L + 1), L, parity)
    complex_basis = real_tesseral_to_complex_multiplet(
        torch.eye(2 * L + 1, dtype=torch.float64), L).numpy()
    identity = json.dumps({"schema": "ye3t_cartesian_tesseral_target_v2",
                           "L": L, "parity": parity},
                          sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(
        identity + np.ascontiguousarray(basis, dtype="<f8").tobytes() +
        np.ascontiguousarray(complex_basis.view(np.float64), dtype="<f8").tobytes()
    ).hexdigest()
