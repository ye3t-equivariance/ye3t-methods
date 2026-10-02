
"""Stress-basis helpers for examples with explicit per-particle stress dependence.

The scalar stress basis can include:
- the raw hydrostatic component ``T_00`` (L=0), and
- Chebyshev-exp-cos basis functions of the Frobenius norm of the stress.

Stress-norm hyperparameters are supplied per ordered bond type, analogous to the
radial hyperparameters. In the current per-site examples, the diagonal channel
``(mu0, mu0)`` is used for each center-site type.
"""

import itertools
import torch

from .trc_cheby import chebyshev_poly_first
from ye3t_ace._record import recordclass


@recordclass(('possible_types', 'sigma_max', 'sigma_lambda', 'n_scalar_max', 'l_aux_min', 'l_aux_max', 'dtype'), frozen = True)
class StressBasisSettings:
    """Settings for explicit stress basis functions.

    Parameters
    ----------
    possible_types
        Ordered chemical/site types.
    sigma_max, sigma_lambda
        Stress-norm basis hyperparameters supplied per ordered bond type in the
        order ``list(itertools.product(possible_types, possible_types))``.
        In the present per-site stress basis, the diagonal entry for the center
        site type is used.
    n_scalar_max
        Highest scalar stress-basis degree. If 0, only the raw hydrostatic
        channel is included when L=0 is requested.
    l_aux_min, l_aux_max
        Requested auxiliary angular range. Supported channels are L=0 and L=2.
    """

    possible_types = (0,)
    sigma_max = (1.0,)
    sigma_lambda = (0.5,)
    n_scalar_max = 0
    l_aux_min = 0
    l_aux_max = 2
    dtype = torch.float64

    def __post_init__(self):
        possible_types = tuple(self.possible_types)
        ntypes = len(possible_types)
        if ntypes < 1:
            raise ValueError('possible_types must contain at least one type')
        n_bond = ntypes * ntypes
        object.__setattr__(self, 'possible_types', possible_types)
        object.__setattr__(self, 'bond_inds', tuple(itertools.product(range(ntypes), range(ntypes))))
        object.__setattr__(self, 'sigma_max', tuple(self._expand(self.sigma_max, n_bond, 'sigma_max')))
        object.__setattr__(self, 'sigma_lambda', tuple(self._expand(self.sigma_lambda, n_bond, 'sigma_lambda')))

    @staticmethod
    def _expand(values, n_bond, name):
        if isinstance(values, (float, int)):
            return [float(values)] * n_bond
        vals = list(values)
        if len(vals) == 1:
            return [float(vals[0])] * n_bond
        if len(vals) != n_bond:
            raise ValueError(f'{name} must have length 1 or n_types**2={n_bond}; got {len(vals)}')
        return [float(v) for v in vals]


def frobenius_norm_from_tesseral(T_00, S_2M):
    """Return the Frobenius norm from scalar/tesseral stress components."""
    T_00 = T_00.reshape(-1, 1)
    return torch.sqrt(torch.clamp(T_00.pow(2).sum(dim=-1) + S_2M.pow(2).sum(dim=-1), min=0.0))


def _cheb_exp_cos(x01, degree, sigma_lambda):
    """ChebExpCos basis on x in [0,1], mirroring the radial-basis shape."""
    pi = torch.as_tensor(torch.pi, dtype=x01.dtype, device=x01.device)
    if degree == 0:
        return chebyshev_poly_first(x01, 0)
    if degree == 1:
        return 0.5 * (1.0 + torch.cos(pi * x01))
    numerator = torch.exp(-sigma_lambda * (x01 - 1.0)) - 1.0
    denominator = torch.exp(sigma_lambda) - 1.0
    exp_scale = 1.0 - 2.0 * (numerator / denominator)
    cheb = chebyshev_poly_first(exp_scale, degree)
    return 0.25 * (1.0 - cheb) * (1.0 + torch.cos(pi * x01))


def build_stress_basis(T_00, S_2M, atom_types, settings, return_target_report=False):
    """Construct scalar and rank-2 stress basis functions.

    Returns
    -------
    scalar_basis, rank2_basis
        ``scalar_basis`` has shape ``[N, n_scalar_features]`` and includes the
        hydrostatic channel when requested. ``rank2_basis`` has shape
        ``[N, n_rank2_features, 5]`` or ``None``.
    """
    dtype = settings.dtype
    device = T_00.device
    T_00 = T_00.reshape(-1, 1).to(dtype=dtype, device=device)
    S_2M = S_2M.to(dtype=dtype, device=device)
    atom_types = atom_types.to(device=device)

    scalar_parts = []
    if settings.l_aux_min <= 0 <= settings.l_aux_max:
        scalar_parts.append(T_00)
        if settings.n_scalar_max > 0:
            norm = frobenius_norm_from_tesseral(T_00, S_2M)
            ntypes = len(settings.possible_types)
            diag_idx = atom_types * ntypes + atom_types
            sigma_max = torch.as_tensor(settings.sigma_max, dtype=dtype, device=device)[diag_idx]
            sigma_lambda = torch.as_tensor(settings.sigma_lambda, dtype=dtype, device=device)[diag_idx]
            x01 = torch.clamp(norm / torch.clamp(sigma_max, min=1e-12), min=0.0, max=1.0)
            for degree in range(settings.n_scalar_max + 1):
                scalar_parts.append(_cheb_exp_cos(x01, degree, sigma_lambda).reshape(-1, 1))

    scalar_basis = torch.cat(scalar_parts, dim=1) if scalar_parts else torch.zeros((T_00.shape[0], 0), dtype=dtype, device=device)

    rank2_basis = None
    if settings.l_aux_min <= 2 <= settings.l_aux_max:
        rank2_basis = S_2M.unsqueeze(1)

    if return_target_report:
        from .property_targets import stress_target_spec, target_provenance_report

        spec = stress_target_spec(
            metadata={
                "basis": "explicit_per_particle_stress_basis",
                "scalar_feature_count": int(scalar_basis.shape[1]),
                "rank2_feature_count": 0 if rank2_basis is None else int(rank2_basis.shape[1]),
            }
        )
        report = target_provenance_report(
            spec,
            descriptor_plan={
                "row_family": "stress_basis_adapter",
                "uses_descriptor_cache": False,
                "uses_local_coupling_enumeration": False,
            },
            derivative_chain=("explicit_stress_input", "stress_basis_features"),
            row_source="build_stress_basis",
        )
        return scalar_basis, rank2_basis, report

    return scalar_basis, rank2_basis
