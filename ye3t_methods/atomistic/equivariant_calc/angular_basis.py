"""Angular spherical-harmonic bases with explicit convention metadata."""
from ye3t_methods.atomistic._record import recordclass
import hashlib
import json
import math

import torch

from .trc_sph_harm import (
    real_spherical_harmonics_l,
    real_spherical_harmonics_l_from_cartesian_with_derivatives,
    real_spherical_harmonics_l_from_unit_cartesian,
    spherical_harmonic,
    spherical_harmonics_l,
)


def _json_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@recordclass(('kind', 'normalization', 'phase_convention', 'component_order', 'real_transform'), frozen = True)
class AngularConvention:
    component_order = "m=-l,...,l"
    real_transform = None

    def metadata(self, *, lmax = None):
        payload = {
            "kind": str(self.kind),
            "normalization": str(self.normalization),
            "phase_convention": str(self.phase_convention),
            "component_order": str(self.component_order),
            "real_transform": None if self.real_transform is None else str(self.real_transform),
        }
        if lmax is not None:
            payload["lmax"] = int(lmax)
        payload["convention_hash"] = _json_hash(payload)
        return payload

    @property
    def convention_hash(self):
        return self.metadata()["convention_hash"]


class AngularBasis:

    @property
    def convention(self):
        raise NotImplementedError

    def evaluate(self, l, rhat):
        raise NotImplementedError

    def cartesian_derivative(self, l, rvec):
        raise NotImplementedError

    def all_m(self, l, theta, phi):
        raise NotImplementedError

    def single_m(self, l, m, theta, phi):
        return self.all_m(l, theta, phi)[int(m) + int(l)]

    def convention_metadata(self, *, lmax = None):
        return self.convention.metadata(lmax=lmax)


@recordclass(('normalization', 'phase_convention'), frozen = True)
class ComplexSphericalHarmonicsBasis(AngularBasis):
    normalization = "orthonormal_4pi"
    phase_convention = "condon_shortley"

    @property
    def kind(self):
        return "complex_spherical"

    @property
    def convention(self):
        return AngularConvention(
            kind=self.kind,
            normalization=self.normalization,
            phase_convention=self.phase_convention,
        )

    def evaluate(self, l, rhat):
        rhat = torch.as_tensor(rhat)
        theta = torch.atan2(torch.linalg.norm(rhat[..., :2], dim=-1), rhat[..., 2])
        phi = torch.atan2(rhat[..., 1], rhat[..., 0])
        return spherical_harmonics_l(int(l), theta, phi)

    def cartesian_derivative(self, l, rvec):
        real_values, real_derivatives = (
            real_spherical_harmonics_l_from_cartesian_with_derivatives(
                int(l),
                rvec,
            )
        )
        transform = complex_to_real_tesseral_matrix(
            int(l),
            dtype=torch.complex128
            if real_values.dtype == torch.float64
            else torch.complex64,
            device=real_values.device,
        )
        inverse = transform.conj().transpose(0, 1)
        complex_values = torch.einsum(
            "ab,b...->a...",
            inverse,
            real_values.to(inverse.dtype),
        )
        complex_derivatives = torch.einsum(
            "ab,b...k->a...k",
            inverse,
            real_derivatives.to(inverse.dtype),
        )
        return complex_values, complex_derivatives

    def cartesian_values(self, l, rvec, *, eps = 1.0e-12):
        """Evaluate the nonsingular Cartesian form without detaching autograd."""
        rvec = torch.as_tensor(rvec)
        radius = torch.linalg.norm(rvec, dim=-1)
        safe_radius = torch.clamp(
            radius,
            min=torch.as_tensor(eps, dtype=rvec.dtype, device=rvec.device),
        )
        unit = rvec / safe_radius.unsqueeze(-1)
        real_values = real_spherical_harmonics_l_from_unit_cartesian(
            int(l),
            unit,
        )
        transform = complex_to_real_tesseral_matrix(
            int(l),
            dtype=torch.complex128
            if real_values.dtype == torch.float64
            else torch.complex64,
            device=real_values.device,
        )
        return torch.einsum(
            "ab,b...->a...",
            transform.conj().transpose(0, 1),
            real_values.to(transform.dtype),
        )

    def all_m(self, l, theta, phi):
        return spherical_harmonics_l(int(l), theta, phi)

    def single_m(self, l, m, theta, phi):
        return spherical_harmonic(int(m), int(l), theta, phi)


def complex_to_real_tesseral_matrix(l, *, dtype=torch.complex128, device=None):
    """Return ``U_l`` such that ``Y_real = U_l @ Y_complex``.

    Both bases use signed ``m`` row order ``[-l, ..., l]``.
    """

    l = int(l)
    if l < 0:
        raise ValueError("l must be non-negative")
    matrix = torch.zeros((2 * l + 1, 2 * l + 1), dtype=dtype, device=device)
    rt2_inv = 1.0 / math.sqrt(2.0)
    matrix[l, l] = 1.0 + 0.0j
    for p in range(1, l + 1):
        sign = -1.0 if (p % 2) else 1.0
        matrix[l + p, l - p] = rt2_inv
        matrix[l + p, l + p] = sign * rt2_inv
        matrix[l - p, l - p] = -1j * rt2_inv
        matrix[l - p, l + p] = 1j * sign * rt2_inv
    return matrix


def complex_to_real_tesseral_metadata(l):
    entries = []
    matrix = complex_to_real_tesseral_matrix(int(l), dtype=torch.complex128)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = complex(matrix[row, col].item())
            if abs(value) > 0.0:
                entries.append(
                    {
                        "real_m": int(row - int(l)),
                        "complex_m": int(col - int(l)),
                        "real": float(value.real),
                        "imag": float(value.imag),
                    }
                )
    payload = {
        "l": int(l),
        "source": "complex_spherical",
        "target": "real_tesseral",
        "row_order": "m=-l,...,l",
        "entries": tuple(entries),
    }
    payload["transform_hash"] = _json_hash(payload)
    return payload


@recordclass(('normalization', 'phase_convention', 'real_transform'), frozen = True)
class RealSphericalHarmonicsBasis(AngularBasis):
    normalization = "orthonormal_4pi"
    phase_convention = "condon_shortley"
    real_transform = "signed_real_tesseral"

    @property
    def kind(self):
        return "real_spherical"

    @property
    def convention(self):
        return AngularConvention(
            kind=self.kind,
            normalization=self.normalization,
            phase_convention=self.phase_convention,
            real_transform=self.real_transform,
        )

    def evaluate(self, l, rhat):
        rhat = torch.as_tensor(rhat)
        theta = torch.atan2(torch.linalg.norm(rhat[..., :2], dim=-1), rhat[..., 2])
        phi = torch.atan2(rhat[..., 1], rhat[..., 0])
        return real_spherical_harmonics_l(int(l), theta, phi)

    def cartesian_derivative(self, l, rvec):
        return real_spherical_harmonics_l_from_cartesian_with_derivatives(int(l), rvec)

    def all_m(self, l, theta, phi):
        return real_spherical_harmonics_l(int(l), theta, phi)

    def transform_metadata(self, l):
        return complex_to_real_tesseral_metadata(int(l))


def angular_basis_for_backend(backend):
    backend = str(backend).strip().lower()
    if backend == "complex":
        return ComplexSphericalHarmonicsBasis()
    if backend == "real":
        return RealSphericalHarmonicsBasis()
    raise ValueError("backend must be one of complex or real")
