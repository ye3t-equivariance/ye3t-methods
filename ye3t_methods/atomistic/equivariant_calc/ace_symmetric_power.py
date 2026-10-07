"""ACE-normalized real symmetric-power tables for repeated blocks."""

import itertools
from functools import lru_cache

import torch

from ye3t.core.basis.homogeneous import (
    homogeneous_basis_states_by_L,
    occupancy_expansion_to_m_vectors,
    thaw_magnetic_expansion,
)
from ye3t.core.tesseral import real_tesseral_to_complex_multiplet
from ye3t_methods.atomistic._record import recordclass


def is_native_real_l1_even_scalar_symmetric_power(power, input_L, output_L, multiplicity_index):
    """Return whether the block has a native real-tesseral scalar norm path.

    For the ACE independent homogeneous convention used by the existing
    symmetric-power tables, the tested ``Sym^{2p}(V_1) -> V_0`` multiplicity-0
    block evaluates to ``(x_0^2 + x_1^2 + x_2^2)^p`` in the real tesseral basis.
    The implementation is a runtime specialization of that existing table
    convention, not a full replacement for lower-output ``Sym^N(V_l)`` blocks.
    """

    return (
        int(input_L) == 1
        and int(output_L) == 0
        and int(multiplicity_index) == 0
        and int(power) >= 2
        and int(power) % 2 == 0
    )


def native_real_l1_even_scalar_symmetric_power(x, *, power):
    """Evaluate the native ``Sym^{2p}(V_1)->V_0`` real-tesseral scalar block."""

    power = int(power)
    if power < 2 or power % 2:
        raise ValueError("native real l=1 scalar symmetric power requires an even power >= 2.")
    norm2 = torch.sum(x.real * x.real, dim=-1, keepdim=True)
    value = norm2.pow(power // 2)
    return value.to(dtype=x.dtype)


@recordclass(
    (
        'power',
        'input_L',
        'output_L',
        'multiplicity_index',
        'monomial_indices',
        'output_m',
        'values',
        'basis_convention',
        'normalization',
        'backend',
        'fallback_reason',
    ),
    frozen=True,
)
class ACESymmetricPowerRealTable:
    pass


@lru_cache(maxsize=None)
def ace_complex_symmetric_power_entries(power, input_L, output_L, multiplicity_index):
    states = homogeneous_basis_states_by_L(int(power), int(input_L), basis_mode="independent").get(int(output_L), ())
    if int(multiplicity_index) < 0 or int(multiplicity_index) >= len(states):
        raise ValueError(
            f"Sym^{power}(V_{input_L}) -> V_{output_L} has no multiplicity index {multiplicity_index}."
        )
    state = states[int(multiplicity_index)]
    frozen = occupancy_expansion_to_m_vectors(
        int(input_L),
        int(power),
        state.occupancy_expansion_by_M,
    )
    expansion = thaw_magnetic_expansion(frozen)
    entries = []
    for M, block in sorted(expansion.items()):
        out_index = int(M) + int(output_L)
        for magnetic_tuple, coefficient in sorted(block.items()):
            entries.append(
                (
                    out_index,
                    tuple(int(m) + int(input_L) for m in magnetic_tuple),
                    complex(coefficient.evalf(30)),
                )
            )
    return tuple(entries)


@lru_cache(maxsize=None)
def _real_to_complex_matrix_cpu(L):
    L = int(L)
    eye = torch.eye(2 * L + 1, dtype=torch.float64)
    matrix = real_tesseral_to_complex_multiplet(eye, L).detach().cpu()
    return tuple(
        tuple(complex(matrix[real_index, complex_index].item()) for complex_index in range(2 * L + 1))
        for real_index in range(2 * L + 1)
    )


@lru_cache(maxsize=None)
def ace_real_symmetric_power_entries(power, input_L, output_L, multiplicity_index):
    entries = ace_complex_symmetric_power_entries(
        int(power),
        int(input_L),
        int(output_L),
        int(multiplicity_index),
    )
    input_transform = _real_to_complex_matrix_cpu(input_L)
    output_transform = _real_to_complex_matrix_cpu(output_L)
    accum = {}
    for output_complex, monomial_complex, coeff in entries:
        slot_expansions = []
        for complex_index in monomial_complex:
            options = []
            for real_index in range(2 * int(input_L) + 1):
                value = input_transform[real_index][int(complex_index)]
                if abs(value) > 1.0e-14:
                    options.append((int(real_index), value))
            slot_expansions.append(tuple(options))
        for output_real in range(2 * int(output_L) + 1):
            output_factor = output_transform[output_real][int(output_complex)].conjugate()
            if abs(output_factor) <= 1.0e-14:
                continue
            for expanded in itertools.product(*slot_expansions):
                real_row = tuple(int(item[0]) for item in expanded)
                value = complex(coeff) * output_factor
                for _, transform_value in expanded:
                    value *= complex(transform_value)
                key = (int(output_real), real_row)
                accum[key] = accum.get(key, 0.0 + 0.0j) + value
    rows = []
    for (output_real, real_row), value in sorted(accum.items()):
        if abs(value) <= 1.0e-12:
            continue
        if abs(value.imag) > 1.0e-10:
            raise RuntimeError("ACE real symmetric-power table has non-real coefficients.")
        rows.append((int(output_real), tuple(int(v) for v in real_row), float(value.real)))
    return tuple(rows)


def ace_real_symmetric_power_table(power, input_L, output_L, multiplicity_index, *, device=None, dtype=None):
    entries = ace_real_symmetric_power_entries(
        int(power),
        int(input_L),
        int(output_L),
        int(multiplicity_index),
    )
    if dtype is None:
        dtype = torch.float64
    if entries:
        monomial_indices = torch.tensor([entry[1] for entry in entries], dtype=torch.long, device=device)
        output_m = torch.tensor([entry[0] for entry in entries], dtype=torch.long, device=device)
        values = torch.tensor([entry[2] for entry in entries], dtype=dtype, device=device)
    else:
        monomial_indices = torch.empty((0, int(power)), dtype=torch.long, device=device)
        output_m = torch.empty((0,), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=dtype, device=device)
    return ACESymmetricPowerRealTable(
        power=int(power),
        input_L=int(input_L),
        output_L=int(output_L),
        multiplicity_index=int(multiplicity_index),
        monomial_indices=monomial_indices,
        output_m=output_m,
        values=values,
        basis_convention="real_tesseral",
        normalization="ace_independent_homogeneous",
        backend="ace_real_monomial_table",
        fallback_reason=None,
    )


def ace_real_symmetric_power_metadata(power, input_L, output_L, multiplicity_index):
    if is_native_real_l1_even_scalar_symmetric_power(
        int(power),
        int(input_L),
        int(output_L),
        int(multiplicity_index),
    ):
        return {
            "power": int(power),
            "input_L": int(input_L),
            "output_L": int(output_L),
            "multiplicity_index": int(multiplicity_index),
            "sym_power_backend": "native_real_l1_even_scalar_norm_power",
            "sym_power_normalization": "ace_independent_homogeneous",
            "sym_power_term_count": 1,
            "sym_power_fallback_reason": None,
        }
    try:
        entries = ace_real_symmetric_power_entries(
            int(power),
            int(input_L),
            int(output_L),
            int(multiplicity_index),
        )
        return {
            "power": int(power),
            "input_L": int(input_L),
            "output_L": int(output_L),
            "multiplicity_index": int(multiplicity_index),
            "sym_power_backend": "ace_real_monomial_table",
            "sym_power_normalization": "ace_independent_homogeneous",
            "sym_power_term_count": int(len(entries)),
            "sym_power_fallback_reason": None,
        }
    except Exception as exc:
        return {
            "power": int(power),
            "input_L": int(input_L),
            "output_L": int(output_L),
            "multiplicity_index": int(multiplicity_index),
            "sym_power_backend": "complex_fallback",
            "sym_power_normalization": "ace_independent_homogeneous",
            "sym_power_term_count": 0,
            "sym_power_fallback_reason": str(exc),
        }


def ace_real_symmetric_power_metadata_for_block_specs(block_specs):
    rows = []
    for specs in block_specs:
        for spec in specs:
            if str(spec["kind"]) != "sym":
                continue
            rows.append(
                ace_real_symmetric_power_metadata(
                    int(spec["k_b"]),
                    int(spec["l"]),
                    int(spec["Lambda"]),
                    int(spec["multiplicity_index"]),
                )
            )
    return tuple(rows)


__all__ = [
    "ACESymmetricPowerRealTable",
    "ace_complex_symmetric_power_entries",
    "ace_real_symmetric_power_entries",
    "ace_real_symmetric_power_metadata",
    "ace_real_symmetric_power_metadata_for_block_specs",
    "ace_real_symmetric_power_table",
    "is_native_real_l1_even_scalar_symmetric_power",
    "native_real_l1_even_scalar_symmetric_power",
]
