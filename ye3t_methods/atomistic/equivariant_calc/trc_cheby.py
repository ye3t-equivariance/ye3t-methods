

import torch


def _reshape_degree_for_broadcast(x, degree):
    while degree.ndim < x.ndim:
        degree = degree.unsqueeze(-1)
    return degree


def chebyshev_poly_first(x, k):
    """
    Evaluate Chebyshev polynomials of the first kind in a torch-native,
    autograd-friendly way.

    Parameters
    ----------
    x:
        Input tensor of arbitrary shape.
    k:
        Either a Python integer degree or an integer tensor that broadcasts
        against ``x``. This preserves the legacy row-wise use case while also
        supporting scalar-degree evaluation.

    Returns
    -------
    Tensor
        ``T_k(x)`` with the same broadcasted shape as ``x``.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if isinstance(k, int):
        if k < 0:
            raise ValueError('Chebyshev degree must be non-negative.')
        if k == 0:
            return torch.ones_like(x)
        if k == 1:
            return x
        t_prev = torch.ones_like(x)
        t_curr = x
        for _ in range(2, int(k) + 1):
            t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
        return t_curr

    if k.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('k must be an integer tensor or Python int.')

    degree = k.to(device=x.device)
    degree = _reshape_degree_for_broadcast(x, degree)
    _, degree = torch.broadcast_tensors(x, degree)
    max_degree = int(torch.max(degree).item()) if degree.numel() else 0
    if max_degree < 0:
        raise ValueError('Chebyshev degree must be non-negative.')

    out = torch.ones_like(x)
    if max_degree == 0:
        return out
    t_prev = torch.ones_like(x)
    t_curr = x
    out = torch.where(degree == 1, t_curr, out)
    for current_degree in range(2, max_degree + 1):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
        out = torch.where(degree == current_degree, t_curr, out)
    return out
