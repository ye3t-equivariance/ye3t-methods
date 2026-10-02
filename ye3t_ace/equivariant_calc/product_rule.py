"""Explicit ACE product-rule evaluation over compiled descriptor terms."""

from ye3t_ace._record import recordclass
import torch


def _complex_dtype_for(dtype):
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    return dtype


def _coefficient_dtype(atomic_base, compiled, *, imag_tol):
    base_dtype = atomic_base.dtype
    if torch.is_complex(atomic_base):
        return base_dtype
    max_imag = 0.0
    for coeffs in compiled.coeffs_cpu:
        if coeffs.numel() == 0:
            continue
        if torch.is_complex(coeffs):
            max_imag = max(max_imag, float(torch.max(torch.abs(coeffs.imag)).item()))
    if max_imag <= float(imag_tol):
        return base_dtype
    return _complex_dtype_for(base_dtype)


def _descriptor_count_from_compiled(compiled):
    return int(len(compiled.channel_rows_cpu))


def _validate_explicit_terms(compiled, descriptor_count):
    accelerated = []
    factorized = getattr(compiled, "factorized_plan", None)
    if factorized is not None:
        accelerated.extend(int(idx) for idx in getattr(factorized, "active_descriptor_indices", ()))
    direct_symmetric = getattr(compiled, "direct_symmetric_power_plan", None)
    if direct_symmetric is not None:
        accelerated.extend(int(idx) for idx in getattr(direct_symmetric, "active_descriptor_indices", ()))
    missing = [
        int(idx)
        for idx in accelerated
        if 0 <= int(idx) < int(descriptor_count)
        and int(compiled.channel_rows_cpu[int(idx)].shape[0]) == 0
        and int(compiled.coeffs_cpu[int(idx)].shape[0]) == 0
    ]
    if missing:
        raise ValueError(
            "Explicit product-rule evaluation requires descriptor term rows; "
            f"compiled descriptors {missing} are represented by an accelerated plan."
        )


def _descriptor_chunks(descriptor_count, chunk_size):
    if chunk_size is None:
        return (range(0, int(descriptor_count)),)
    size = max(1, int(chunk_size))
    return tuple(range(start, min(start + size, int(descriptor_count))) for start in range(0, int(descriptor_count), size))


@recordclass(('values', 'channel_derivatives', 'report'), frozen = True)
class ProductRuleResult:
    """Values and root-channel derivatives for compiled ACE product terms."""


class ExplicitProductRuleEvaluator:
    """Evaluate compiled descriptor products using direct prefix/suffix products."""

    def __init__(self, *, imag_tol=1.0e-12):
        self.imag_tol = float(imag_tol)

    def evaluate(self, atomic_base, compiled, *, descriptor_count=None, chunk_size=None):
        descriptor_count = _descriptor_count_from_compiled(compiled) if descriptor_count is None else int(descriptor_count)
        if atomic_base.ndim != 2:
            raise ValueError("atomic_base must have shape [n_sites, n_channels]")
        if descriptor_count < 0 or descriptor_count > _descriptor_count_from_compiled(compiled):
            raise ValueError("descriptor_count is out of range for this compiled descriptor batch")
        _validate_explicit_terms(compiled, descriptor_count)
        out_dtype = _coefficient_dtype(atomic_base, compiled, imag_tol=self.imag_tol)
        base = atomic_base.to(dtype=out_dtype)
        n_sites = int(base.shape[0])
        n_channels = int(base.shape[1])
        values = torch.zeros((n_sites, descriptor_count), dtype=out_dtype, device=base.device)
        derivatives = torch.zeros((n_sites, descriptor_count, n_channels), dtype=out_dtype, device=base.device)
        chunks = _descriptor_chunks(descriptor_count, chunk_size)
        term_count = 0
        max_rank = 0
        for chunk in chunks:
            for descriptor_index in chunk:
                rows = compiled.channel_rows_cpu[int(descriptor_index)].to(device=base.device)
                coeffs_source = compiled.coeffs_cpu[int(descriptor_index)]
                if not torch.is_complex(torch.empty((), dtype=out_dtype)) and torch.is_complex(coeffs_source):
                    coeffs = coeffs_source.real.to(device=base.device, dtype=out_dtype)
                else:
                    coeffs = coeffs_source.to(device=base.device, dtype=out_dtype)
                if int(rows.shape[0]) != int(coeffs.shape[0]):
                    raise ValueError("descriptor term rows and coefficients have inconsistent lengths")
                for row, coeff in zip(rows, coeffs):
                    row = row.to(dtype=torch.long, device=base.device)
                    rank = int(row.numel())
                    if rank == 0:
                        continue
                    max_rank = max(max_rank, rank)
                    if int(torch.min(row).item()) < 0 or int(torch.max(row).item()) >= n_channels:
                        raise ValueError("descriptor term references an atomic-base channel outside atomic_base")
                    factors = base.index_select(1, row)
                    prefix = [torch.ones((n_sites,), dtype=out_dtype, device=base.device)]
                    for slot in range(rank):
                        prefix.append(prefix[-1] * factors[:, slot])
                    suffix = [None for _ in range(rank + 1)]
                    suffix[rank] = torch.ones((n_sites,), dtype=out_dtype, device=base.device)
                    for slot in range(rank - 1, -1, -1):
                        suffix[slot] = factors[:, slot] * suffix[slot + 1]
                    product_value = prefix[rank]
                    values[:, int(descriptor_index)] += coeff * product_value
                    for slot in range(rank):
                        channel_index = int(row[slot].item())
                        derivatives[:, int(descriptor_index), channel_index] += coeff * prefix[slot] * suffix[slot + 1]
                    term_count += 1
        report = {
            "backend": "explicit_product_rule",
            "descriptor_count": int(descriptor_count),
            "term_count": int(term_count),
            "chunk_size": None if chunk_size is None else int(max(1, int(chunk_size))),
            "chunk_count": int(len(chunks)),
            "channel_count": int(n_channels),
            "max_rank": int(max_rank),
            "value_dtype": str(values.dtype),
            "uses_recursive_products": False,
        }
        return ProductRuleResult(values=values, channel_derivatives=derivatives, report=report)


def evaluate_explicit_product_rule(atomic_base, compiled, *, descriptor_count=None, chunk_size=None, imag_tol=1.0e-12):
    """Evaluate compiled ACE products and ``dB/dA_norm`` without a product graph."""

    return ExplicitProductRuleEvaluator(imag_tol=imag_tol).evaluate(
        atomic_base,
        compiled,
        descriptor_count=descriptor_count,
        chunk_size=chunk_size,
    )


def explicit_product_rule_linear_adjoint(atomic_base, compiled, descriptor_weight, *, imag_tol=1.0e-12):
    """Contract product-rule derivatives with descriptor weights."""

    result = evaluate_explicit_product_rule(
        atomic_base,
        compiled,
        descriptor_count=int(descriptor_weight.shape[-1]),
        imag_tol=imag_tol,
    )
    weights = descriptor_weight.to(device=result.channel_derivatives.device, dtype=result.channel_derivatives.dtype)
    if tuple(weights.shape) != tuple(result.values.shape):
        raise ValueError(f"descriptor_weight must have shape {tuple(result.values.shape)}; got {tuple(weights.shape)}")
    root_adjoint = torch.sum(weights.unsqueeze(-1) * result.channel_derivatives, dim=1)
    return result.values, root_adjoint
