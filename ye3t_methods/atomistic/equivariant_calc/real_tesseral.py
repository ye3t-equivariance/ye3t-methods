
"""Helpers for converting complex spherical multiplets to a real tesseral basis."""

import torch


def complex_multiplet_to_real_tesseral(values, L, M_values):
    if tuple(M_values) != tuple(range(-L, L + 1)):
        raise ValueError('M_values must be the full ordered range -L..L')
    idx = {m: i for i, m in enumerate(M_values)}
    parts = []
    rt2 = torch.sqrt(torch.tensor(2.0, dtype=values.real.dtype, device=values.device))
    for m in range(L, 0, -1):
        ym = values[..., idx[m]]
        yneg = values[..., idx[-m]]
        parts.append(((yneg + ((-1) ** m) * ym) / rt2).real)
    parts.append(values[..., idx[0]].real)
    for m in range(1, L + 1):
        ym = values[..., idx[m]]
        yneg = values[..., idx[-m]]
        parts.append(((yneg - ((-1) ** m) * ym) / (1j * rt2)).real)
    return torch.stack(parts, dim=-1)
