"""Radial basis functions with explicit derivative and convention metadata."""
from ye3t_methods.atomistic._record import recordclass
import hashlib
import json
import math

import torch


_PACE_SPLINE_MAX_COEFFICIENT_BYTES = 128 * 1024 * 1024


def _json_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_tensor(value, *, like):
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _cutoff(r, rc):
    x = r / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
    pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
    inside = x <= 1.0
    values = 0.5 * (1.0 + torch.cos(pi * x))
    return torch.where(inside, values, torch.zeros_like(values))


def _cutoff_derivative(r, rc):
    x = r / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
    pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
    inside = x <= 1.0
    values = -0.5 * pi * torch.sin(pi * x) / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
    return torch.where(inside, values, torch.zeros_like(values))


def _chebyshev_first(x, n):
    n = int(n)
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x
    t_prev = torch.ones_like(x)
    t_curr = x
    for _ in range(2, n + 1):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
    return t_curr


def _chebyshev_first_derivative(x, n):
    n = int(n)
    if n == 0:
        return torch.zeros_like(x)
    if n == 1:
        return torch.ones_like(x)
    u_prev = torch.ones_like(x)
    if n == 2:
        return 4.0 * x
    u_curr = 2.0 * x
    for _ in range(2, n):
        u_prev, u_curr = u_curr, 2.0 * x * u_curr - u_prev
    return float(n) * u_curr


def _pace_cheb_exp_cos_table_with_derivative(
    r,
    *,
    rc,
    cutoff_width,
    lmbda,
    radial_count,
):
    """Evaluate PACE's one-based ChebExpCos base channels before splining."""

    radial_count = int(radial_count)
    if radial_count < 1:
        raise ValueError("PACE ChebExpCos requires at least one radial channel.")
    r = torch.as_tensor(r)
    rc = _as_tensor(rc, like=r)
    cutoff_width = _as_tensor(cutoff_width, like=r)
    lmbda = _as_tensor(lmbda, like=r)
    pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
    scaled = r / rc
    envelope = 0.5 * (1.0 + torch.cos(pi * scaled))
    envelope_derivative = -0.5 * pi * torch.sin(pi * scaled) / rc

    positive_width = cutoff_width > 0.0
    safe_width = torch.where(positive_width, cutoff_width, torch.ones_like(cutoff_width))
    switch_active = positive_width & (r > rc - cutoff_width)
    switch_phase = pi * (r - (rc - cutoff_width)) / safe_width
    outer = torch.where(
        switch_active,
        0.5 * (1.0 + torch.cos(switch_phase)),
        torch.ones_like(r),
    )
    outer_derivative = torch.where(
        switch_active,
        -0.5 * pi * torch.sin(switch_phase) / safe_width,
        torch.zeros_like(r),
    )
    combined = envelope * outer
    combined_derivative = envelope_derivative * outer + envelope * outer_derivative

    values = [combined]
    derivatives = [combined_derivative]
    if radial_count > 1:
        exponential = torch.exp(-lmbda * (scaled - 1.0))
        denominator = torch.expm1(lmbda)
        warped = 1.0 - 2.0 * (exponential - 1.0) / denominator
        warped_derivative = 2.0 * lmbda * exponential / (denominator * rc)
        previous = torch.ones_like(r)
        current = warped
        previous_derivative = torch.zeros_like(r)
        current_derivative = warped_derivative
        for radial_index in range(1, radial_count):
            if radial_index > 1:
                following = 2.0 * warped * current - previous
                following_derivative = 2.0 * (
                    warped_derivative * current + warped * current_derivative
                ) - previous_derivative
                previous, current = current, following
                previous_derivative, current_derivative = current_derivative, following_derivative
            base = 0.5 * (1.0 - current)
            base_derivative = -0.5 * current_derivative
            values.append(base * combined)
            derivatives.append(base_derivative * combined + base * combined_derivative)

    inside = r < rc
    value_table = torch.stack(values, dim=-1)
    derivative_table = torch.stack(derivatives, dim=-1)
    return (
        torch.where(inside.unsqueeze(-1), value_table, torch.zeros_like(value_table)),
        torch.where(inside.unsqueeze(-1), derivative_table, torch.zeros_like(derivative_table)),
    )


def _pace_uniform_cubic_spline_coefficients(
    *,
    rc,
    requested_spacing,
    cutoff_width,
    lmbda,
    radial_count,
    device,
    dtype,
):
    """Build the PACE uniform cubic-Hermite table for one directed bond."""

    cutoff = float(rc)
    spacing_request = float(requested_spacing)
    interval_count = int(cutoff / spacing_request)
    if interval_count < 2:
        raise ValueError("PACE spline requires at least two intervals.")
    coefficient_bytes = (
        (interval_count + 1)
        * int(radial_count)
        * 4
        * torch.empty((), dtype=dtype).element_size()
    )
    if coefficient_bytes > _PACE_SPLINE_MAX_COEFFICIENT_BYTES:
        raise MemoryError(
            "PACE spline coefficient table exceeds the 128 MiB allocation guard "
            f"({coefficient_bytes} bytes requested)."
        )
    cutoff_tensor = torch.as_tensor(cutoff, dtype=dtype, device=device)
    spacing = cutoff_tensor / float(interval_count)
    node_radii = spacing * torch.arange(
        1,
        interval_count + 1,
        dtype=dtype,
        device=device,
    )
    node_values, node_derivatives = _pace_cheb_exp_cos_table_with_derivative(
        node_radii,
        rc=cutoff_tensor,
        cutoff_width=torch.as_tensor(cutoff_width, dtype=dtype, device=device),
        lmbda=torch.as_tensor(lmbda, dtype=dtype, device=device),
        radial_count=radial_count,
    )
    zeros = torch.zeros((1, int(radial_count)), dtype=dtype, device=device)
    right_values = torch.cat((node_values[1:], zeros), dim=0)
    scaled_left_derivatives = node_derivatives * spacing
    scaled_right_derivatives = torch.cat((scaled_left_derivatives[1:], zeros), dim=0)
    coefficients = torch.stack(
        (
            node_values,
            scaled_left_derivatives,
            3.0 * (right_values - node_values)
            - scaled_right_derivatives
            - 2.0 * scaled_left_derivatives,
            -2.0 * (right_values - node_values)
            + scaled_right_derivatives
            + scaled_left_derivatives,
        ),
        dim=-1,
    )
    unused_interval_zero = torch.zeros(
        (1, int(radial_count), 4),
        dtype=dtype,
        device=device,
    )
    return torch.cat((unused_interval_zero, coefficients), dim=0), interval_count


def _pace_uniform_cubic_spline_evaluate_with_derivative(
    r,
    *,
    coefficients,
    interval_count,
    rc,
):
    """Evaluate one PACE spline table and its physical radial derivative."""

    r = torch.as_tensor(r)
    if not bool(torch.all(torch.isfinite(r))):
        raise ValueError("PACE spline radii must be finite.")
    cutoff = _as_tensor(rc, like=r)
    scale = float(interval_count) / cutoff
    inside = r < cutoff
    intervals = torch.floor(r * scale).to(torch.long)
    if bool(torch.any(inside & (intervals <= 0))):
        raise ValueError("PACE spline radius is below its first interval.")
    safe_intervals = intervals.clamp(min=1, max=int(interval_count))
    local = r * scale - safe_intervals.to(r.dtype)
    selected = coefficients.index_select(0, safe_intervals.reshape(-1)).reshape(
        tuple(r.shape) + tuple(coefficients.shape[1:])
    )
    local_column = local.unsqueeze(-1)
    values = (
        selected[..., 0]
        + selected[..., 1] * local_column
        + selected[..., 2] * local_column.square()
        + selected[..., 3] * local_column.pow(3)
    )
    derivatives = scale * (
        selected[..., 1]
        + 2.0 * selected[..., 2] * local_column
        + 3.0 * selected[..., 3] * local_column.square()
    )
    return (
        torch.where(inside.unsqueeze(-1), values, torch.zeros_like(values)),
        torch.where(inside.unsqueeze(-1), derivatives, torch.zeros_like(derivatives)),
    )


@recordclass(('kind', 'normalization', 'parameterization'), frozen = True)
class RadialConvention:
    normalization = "raw_with_cosine_cutoff"
    parameterization = None

    def metadata(self):
        payload = {
            "kind": str(self.kind),
            "normalization": str(self.normalization),
            "parameterization": {} if self.parameterization is None else dict(self.parameterization),
        }
        payload["convention_hash"] = _json_hash(payload)
        return payload

    @property
    def convention_hash(self):
        return self.metadata()["convention_hash"]


class RadialBasis:
    kind = "abstract"
    normalization = "raw_with_cosine_cutoff"

    @property
    def convention(self):
        return RadialConvention(kind=self.kind, normalization=self.normalization)

    def evaluate(self, r, species_pair=None):
        raise NotImplementedError

    def derivative(self, r, species_pair=None):
        raise NotImplementedError("RadialBasis.derivative requires a declared derivative backend.")

    def max_abs(self):
        return 1.0

    def convention_metadata(self):
        return self.convention.metadata()


@recordclass(('n', 'rc', 'lmbda', 'kind'), frozen = True)
class ChebExpCosRadialBasis(RadialBasis):
    kind = "ChebExpCos"

    @property
    def convention(self):
        return RadialConvention(
            kind=self.kind,
            parameterization={"n": int(self.n), "envelope": "compact_cosine", "warp": "exponential_chebyshev"},
        )

    def _parts(self, r):
        rc = _as_tensor(self.rc, like=r)
        lmbda = _as_tensor(self.lmbda, like=r)
        x = r / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
        inside = x <= 1.0
        numerator_exp = torch.exp(-lmbda * (x - 1.0))
        denominator = torch.exp(lmbda) - 1.0
        exp_scale = 1.0 - 2.0 * ((numerator_exp - 1.0) / denominator)
        return rc, lmbda, x, inside, numerator_exp, denominator, exp_scale

    def evaluate(self, r, species_pair=None):
        del species_pair
        r = torch.as_tensor(r)
        rc, _, x, inside, _, _, exp_scale = self._parts(r)
        pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
        n = int(self.n)
        if n == 0:
            values = torch.ones_like(r)
        elif n == 1:
            values = 0.5 * (1.0 + torch.cos(pi * x))
        else:
            cheb = _chebyshev_first(exp_scale, n)
            values = 0.25 * (1.0 - cheb) * (1.0 + torch.cos(pi * x))
        del rc
        return torch.where(inside, values, torch.zeros_like(values))

    def derivative(self, r, species_pair=None):
        del species_pair
        r = torch.as_tensor(r)
        rc, lmbda, x, inside, numerator_exp, denominator, exp_scale = self._parts(r)
        pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
        n = int(self.n)
        if n == 0:
            deriv = torch.zeros_like(r)
        elif n == 1:
            deriv = -0.5 * pi * torch.sin(pi * x) / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
        else:
            cheb = _chebyshev_first(exp_scale, n)
            dcheb_dexp = _chebyshev_first_derivative(exp_scale, n)
            dexp_dr = (2.0 * lmbda * numerator_exp / denominator) / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
            cutoff = 1.0 + torch.cos(pi * x)
            dcutoff_dr = -pi * torch.sin(pi * x) / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
            deriv = 0.25 * (-dcheb_dexp * dexp_dr * cutoff + (1.0 - cheb) * dcutoff_dr)
        return torch.where(inside, deriv, torch.zeros_like(deriv))


@recordclass(('n', 'nmax', 'rc', 'width_scale', 'kind'), frozen = True)
class GaussianRadialBasis(RadialBasis):
    width_scale = 0.25
    kind = "Gaussian"

    @property
    def convention(self):
        return RadialConvention(
            kind=self.kind,
            parameterization={"n": int(self.n), "nmax": int(self.nmax), "center_policy": "uniform_inside_cutoff"},
        )

    def _parts(self, r):
        rc = _as_tensor(self.rc, like=r)
        width_scale = torch.clamp(_as_tensor(self.width_scale, like=r).abs(), min=torch.finfo(r.dtype).eps)
        nmax = max(1, int(self.nmax))
        n = max(0, int(self.n))
        center = rc * ((n + 1.0) / (nmax + 1.0))
        sigma = torch.clamp(width_scale * rc, min=torch.finfo(r.dtype).eps)
        z = (r - center) / sigma
        gaussian = torch.exp(-0.5 * z * z)
        cutoff = _cutoff(r, rc)
        return rc, sigma, z, gaussian, cutoff

    def evaluate(self, r, species_pair=None):
        del species_pair
        r = torch.as_tensor(r)
        _, _, _, gaussian, cutoff = self._parts(r)
        return gaussian * cutoff

    def derivative(self, r, species_pair=None):
        del species_pair
        r = torch.as_tensor(r)
        rc, sigma, z, gaussian, cutoff = self._parts(r)
        dgaussian = gaussian * (-z / sigma)
        return dgaussian * cutoff + gaussian * _cutoff_derivative(r, rc)


@recordclass(('n', 'rc', 'kind'), frozen = True)
class BesselRadialBasis(RadialBasis):
    kind = "Bessel"

    @property
    def convention(self):
        return RadialConvention(
            kind=self.kind,
            parameterization={"n": int(self.n), "basis": "sinc_kpi_r_over_rc_with_cosine_cutoff"},
        )

    def _sinc_and_derivative(self, r):
        rc = _as_tensor(self.rc, like=r)
        k = max(1, int(self.n))
        x = k * math.pi * r / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
        tiny = torch.as_tensor(1.0e-7, dtype=r.dtype, device=r.device)
        sinc = torch.where(torch.abs(x) < tiny, torch.ones_like(x) - x * x / 6.0, torch.sin(x) / x)
        dsinc_dx = torch.where(
            torch.abs(x) < tiny,
            -x / 3.0,
            (x * torch.cos(x) - torch.sin(x)) / (x * x),
        )
        dx_dr = k * math.pi / torch.clamp(rc, min=torch.finfo(r.dtype).eps)
        return rc, sinc, dsinc_dx * dx_dr

    def evaluate(self, r, species_pair=None):
        del species_pair
        r = torch.as_tensor(r)
        _, sinc, _ = self._sinc_and_derivative(r)
        return sinc * _cutoff(r, _as_tensor(self.rc, like=r))

    def derivative(self, r, species_pair=None):
        del species_pair
        r = torch.as_tensor(r)
        rc, sinc, dsinc_dr = self._sinc_and_derivative(r)
        cutoff = _cutoff(r, rc)
        return dsinc_dr * cutoff + sinc * _cutoff_derivative(r, rc)


@recordclass(('grid', 'values', 'kind'), frozen = True)
class TabulatedRadialBasis(RadialBasis):
    kind = "TabulatedLinear"

    @property
    def convention(self):
        return RadialConvention(
            kind=self.kind,
            parameterization={"interpolation": "piecewise_linear", "grid_size": int(len(self.grid))},
        )

    def _prepared(self, r):
        grid = _as_tensor(self.grid, like=r).reshape(-1)
        values = _as_tensor(self.values, like=r).reshape(-1)
        if int(grid.numel()) < 2:
            raise ValueError("TabulatedRadialBasis requires at least two grid points.")
        if int(values.numel()) != int(grid.numel()):
            raise ValueError("TabulatedRadialBasis grid and values must have matching lengths.")
        return grid, values

    def evaluate(self, r, species_pair=None):
        del species_pair
        r = torch.as_tensor(r)
        grid, values = self._prepared(r)
        idx = torch.searchsorted(grid, r.contiguous(), right=True) - 1
        idx = torch.clamp(idx, 0, int(grid.numel()) - 2)
        left = grid[idx]
        right = grid[idx + 1]
        weight = (r - left) / torch.clamp(right - left, min=torch.finfo(r.dtype).eps)
        out = values[idx] * (1.0 - weight) + values[idx + 1] * weight
        in_range = (r >= grid[0]) & (r <= grid[-1])
        return torch.where(in_range, out, torch.zeros_like(out))

    def derivative(self, r, species_pair=None):
        del species_pair
        r = torch.as_tensor(r)
        grid, values = self._prepared(r)
        idx = torch.searchsorted(grid, r.contiguous(), right=True) - 1
        idx = torch.clamp(idx, 0, int(grid.numel()) - 2)
        slope = (values[idx + 1] - values[idx]) / torch.clamp(grid[idx + 1] - grid[idx], min=torch.finfo(r.dtype).eps)
        in_range = (r >= grid[0]) & (r <= grid[-1])
        return torch.where(in_range, slope, torch.zeros_like(slope))


def radial_basis_for_kind(kind, *, n, nmax, rc, lmbda):
    key = str(kind).strip().lower().replace("_", "").replace("-", "")
    if key in {"chebexpcos", "default"}:
        return ChebExpCosRadialBasis(n=int(n), rc=rc, lmbda=lmbda)
    if key in {"gaussian", "gaussians"}:
        return GaussianRadialBasis(n=int(n), nmax=int(nmax), rc=rc, width_scale=lmbda)
    if key in {"bessel", "sphericalbessel"}:
        return BesselRadialBasis(n=int(n), rc=rc)
    raise NotImplementedError(f"Unsupported radial basis: {kind}")

def smooth_cosine_cutoff(distance, cutoff):
    """C1 cutoff with zero value and first derivative at the boundary."""

    cutoff_value = torch.as_tensor(
        float(cutoff), dtype=distance.dtype, device=distance.device
    )
    return _cutoff(distance, cutoff_value)
