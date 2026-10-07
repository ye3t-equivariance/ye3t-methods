
"""Simple torch models built on top of evaluated ACE covariants."""

import torch


class LinearScalarACEModel(torch.nn.Module):
    """Linear scalar model ``E_i = B_i dot w + b``."""
    def __init__(self, n_desc):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(n_desc, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, B):
        return B @ self.weight + self.bias


class StressAwareScalarACEModel(torch.nn.Module):
    """Scalar energy-like model with explicit stress dependence.

    The model combines:

    - pure geometric scalar descriptors ``B0``
    - scalar stress basis functions ``S0_basis`` (e.g. hydrostatic stress and
      Chebyshev features of the stress Frobenius norm)
    - rank-2 geometric covariants ``B2_real``
    - rank-2 stress covariants ``S2_basis``

    The scalar output is::

        E_i = B0_i w_geo + S0_i w_stress
            + sum_{d,s} B0_{id} W0_{ds} S0_{is}
            + sum_{d,r,m} B2_{idm} W2_{dr} S2_{irm} + b

    where ``r`` indexes rank-2 stress basis channels.
    """

    def __init__(self, n_scalar_desc, n_scalar_stress, n_rank2_desc = 0, n_rank2_stress = 0):
        super().__init__()
        self.w_geo = torch.nn.Parameter(torch.zeros(n_scalar_desc, dtype=torch.float64))
        self.w_stress = torch.nn.Parameter(torch.zeros(n_scalar_stress, dtype=torch.float64))
        self.W0 = torch.nn.Parameter(torch.zeros(n_scalar_desc, n_scalar_stress, dtype=torch.float64))
        self.W2 = torch.nn.Parameter(torch.zeros(n_rank2_desc, n_rank2_stress, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, B0, S0_basis, B2_real = None, S2_basis = None):
        energy = B0 @ self.w_geo + S0_basis @ self.w_stress + self.bias
        if B0.numel() > 0 and S0_basis.numel() > 0:
            energy = energy + torch.einsum('nd,ds,ns->n', B0, self.W0, S0_basis)
        if B2_real is not None and S2_basis is not None and self.W2.numel() > 0:
            energy = energy + torch.einsum('ndm,dr,nrm->n', B2_real, self.W2, S2_basis)
        return energy
