
"""Autograd utilities for ACE descriptor gradients."""

import itertools
import os
import time
import weakref
import torch

from ye3t.core.couplings import evaluate_factorized_schedule_torch, evaluate_real_factorized_schedule_torch
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None

from .ace_eval_v2 import (
    ACECovariantEvaluator,
    _complex_symmetric_power_entries,
    _complex_symmetric_power_product_plan,
    checked_real_scalar_projection,
)
from .ace_symmetric_power import ace_real_symmetric_power_table
from .site_basis_v2 import site_real_block_to_ye3t_tesseral
from .labeling import DescriptorSpec
from .product_dag import (
    ProductDAGValues,
    evaluate_product_dag,
    evaluate_product_dag_linear_form,
    product_dag_coefficients_are_real,
    product_dag_from_compiled_cached,
    reverse_product_dag_adjoint,
)
from .product_rule import explicit_product_rule_linear_adjoint
from ye3t.core.tesseral import real_tesseral_to_complex_multiplet
from ye3t.runtime.symmetric_power import symmetric_power_product_plan_batched_adjoint
from ye3t_methods.atomistic._record import recordclass


_SYMMETRIC_POWER_REVERSE_TENSOR_CACHE = {}
_SYMMETRIC_POWER_REVERSE_COMPILED_CACHE = {}
_REAL_TESSERAL_COMPLEX_MATRIX_CACHE = {}
_REAL_FACTORIZED_SCHEDULE_CACHE = weakref.WeakKeyDictionary()
_REAL_SYMMETRIC_POWER_REVERSE_CACHE = {}
_FACTORIZED_SCHEDULE_DEVICE_CACHE = weakref.WeakKeyDictionary()
_FACTORIZED_REVERSE_TENSOR_CACHE = weakref.WeakKeyDictionary()
_FACTORIZED_REVERSE_COMPILED_CACHE = {}
_NATIVE_REAL_FACTORIZED_TRITON_FAILURES = set()


def _linear_form_profile_enabled():
    raw = os.environ.get("YE3T_ACE_PROFILE_LINEAR_FORM", os.environ.get("YE3T_ACE_PROFILE_FULL_MODEL", "0"))
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _profile_start(tensor):
    if torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return time.perf_counter()


def _profile_stop(profile, key, start, tensor):
    if profile is None:
        return
    if torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    profile[key] = float(profile.get(key, 0.0)) + float(time.perf_counter() - start)


def _compiled_without_accelerated_plans(compiled):
    return type(compiled)(
        channels=compiled.channels,
        channel_rows_cpu=compiled.channel_rows_cpu,
        coeffs_cpu=compiled.coeffs_cpu,
        grouped_descriptor_indices_cpu=compiled.grouped_descriptor_indices_cpu,
        grouped_channel_rows_cpu=compiled.grouped_channel_rows_cpu,
        grouped_coeffs_cpu=compiled.grouped_coeffs_cpu,
        all_scalar=compiled.all_scalar,
        factorized_plan=None,
        direct_symmetric_power_plan=None,
    )


if triton is not None:  # pragma: no cover - exercised only on CUDA/Triton systems
    @triton.jit
    def _native_real_factorized_forward_reverse_kernel(
        values_ptr,
        output_adjoint_ptr,
        coeffs_ptr,
        component_index_ptr,
        label_index_ptr,
        block_real_indices_ptr,
        grouped_out_ptr,
        block_adjoint_ptr,
        n_atoms,
        term_count,
        component_count,
        basis_count,
        max_m_dim,
        value_stride_atom,
        value_stride_basis,
        value_stride_block,
        value_stride_m,
        out_stride_atom,
        out_stride_component,
        adj_stride_atom,
        adj_stride_component,
        block_adj_stride_atom,
        block_adj_stride_basis,
        block_adj_stride_block,
        block_adj_stride_m,
        block_count,
    ):
        atom_offsets = tl.program_id(0) * 16 + tl.arange(0, 16)
        term_offsets = tl.program_id(1) * 32 + tl.arange(0, 32)
        atom_mask = atom_offsets < n_atoms
        term_mask = term_offsets < term_count

        labels = tl.load(label_index_ptr + term_offsets, mask=term_mask, other=0)
        components = tl.load(component_index_ptr + term_offsets, mask=term_mask, other=0)
        coeffs = tl.load(coeffs_ptr + term_offsets, mask=term_mask, other=0.0)
        prod = tl.full((16, 32), 1.0, tl.float64)

        for block in tl.static_range(0, 8):
            block_active = block < block_count
            m_index = tl.load(
                block_real_indices_ptr + term_offsets * block_count + block,
                mask=term_mask & block_active,
                other=0,
            )
            values = tl.load(
                values_ptr
                + atom_offsets[:, None] * value_stride_atom
                + labels[None, :] * value_stride_basis
                + block * value_stride_block
                + m_index[None, :] * value_stride_m,
                mask=atom_mask[:, None] & term_mask[None, :] & block_active,
                other=1.0,
            )
            prod = prod * values

        term_values = prod * coeffs[None, :]
        tl.atomic_add(
            grouped_out_ptr
            + atom_offsets[:, None] * out_stride_atom
            + components[None, :] * out_stride_component,
            term_values,
            mask=atom_mask[:, None] & term_mask[None, :],
        )

        omega = tl.load(
            output_adjoint_ptr
            + atom_offsets[:, None] * adj_stride_atom
            + components[None, :] * adj_stride_component,
            mask=atom_mask[:, None] & term_mask[None, :],
            other=0.0,
        )
        term_adjoint = omega * coeffs[None, :]
        for target_block in tl.static_range(0, 8):
            target_active = target_block < block_count
            partial = tl.full((16, 32), 1.0, tl.float64)
            for block in tl.static_range(0, 8):
                if block != target_block:
                    block_active = block < block_count
                    m_index = tl.load(
                        block_real_indices_ptr + term_offsets * block_count + block,
                        mask=term_mask & block_active,
                        other=0,
                    )
                    values = tl.load(
                        values_ptr
                        + atom_offsets[:, None] * value_stride_atom
                        + labels[None, :] * value_stride_basis
                        + block * value_stride_block
                        + m_index[None, :] * value_stride_m,
                        mask=atom_mask[:, None] & term_mask[None, :] & block_active,
                        other=1.0,
                    )
                    partial = partial * values
            target_m = tl.load(
                block_real_indices_ptr + term_offsets * block_count + target_block,
                mask=term_mask & target_active,
                other=0,
            )
            tl.atomic_add(
                block_adjoint_ptr
                + atom_offsets[:, None] * block_adj_stride_atom
                + labels[None, :] * block_adj_stride_basis
                + target_block * block_adj_stride_block
                + target_m[None, :] * block_adj_stride_m,
                term_adjoint * partial,
                mask=atom_mask[:, None] & term_mask[None, :] & target_active,
            )


@recordclass(('per_atom_descriptors', 'pair_grad_rows', 'full_array'), frozen = True)
class LAMMPSPaceLikeOutput:
    pass


@recordclass(('x_ij', 'edge_index', 'atom_types'), frozen = True)
class EdgeGeometry:
    pass


def edge_vectors_from_positions(positions, cell, edge_index, shifts = None):
    centers = edge_index[0]
    neighs = edge_index[1]
    disp = positions[neighs] - positions[centers]
    if shifts is not None:
        if cell is None:
            raise ValueError('cell must be provided when shifts are used')
        shifts = shifts.to(dtype=positions.dtype, device=positions.device)
        cell = cell.to(dtype=positions.dtype, device=positions.device)
        if cell.ndim == 3:
            disp = disp + torch.einsum("ei,eij->ej", shifts, cell)
        else:
            disp = disp + shifts @ cell
    return disp


def _evaluate_descriptors_from_positions(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    shifts,
    charges,
    aux_tensor_basis,
    real_if_scalar,
):
    x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
    return evaluator(
        x_ij=x_ij,
        edge_index=edge_index,
        atom_types=atom_types,
        descriptors=descriptors,
        charges=charges,
        aux_tensor_basis=aux_tensor_basis,
        real_if_scalar=real_if_scalar,
    )


def _descriptor_gradients_autograd_loop(
    values,
    positions,
):
    N, K = values.shape
    grads = torch.zeros((N, K, N, 3), dtype=values.dtype, device=values.device)
    for i in range(N):
        for k in range(K):
            if torch.is_complex(values):
                grad_real = torch.autograd.grad(values[i, k].real, positions, retain_graph=True, allow_unused=False)[0]
                grad_imag = torch.autograd.grad(values[i, k].imag, positions, retain_graph=True, allow_unused=False)[0]
                grads[i, k] = torch.complex(grad_real, grad_imag)
            else:
                grad = torch.autograd.grad(values[i, k], positions, retain_graph=True, allow_unused=False)[0]
                grads[i, k] = grad
    return grads


def _descriptor_gradients_autograd_vectorized(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    shifts,
    charges,
    aux_tensor_basis,
    real_if_scalar,
):
    positions = positions.clone().requires_grad_(True)

    def values_fn(pos):
        return _evaluate_descriptors_from_positions(
            evaluator,
            pos,
            cell,
            edge_index,
            atom_types,
            descriptors,
            shifts,
            charges,
            aux_tensor_basis,
            real_if_scalar,
        )

    values = values_fn(positions)
    if torch.is_complex(values):
        def real_imag_fn(pos):
            return torch.view_as_real(values_fn(pos))

        jac = torch.autograd.functional.jacobian(real_imag_fn, positions, vectorize=True)
        grads = torch.complex(jac[:, :, 0], jac[:, :, 1])
    else:
        grads = torch.autograd.functional.jacobian(values_fn, positions, vectorize=True)
    return values, grads


def _descriptor_gradients_analytic_dag(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    shifts,
    charges,
    aux_tensor_basis,
    real_if_scalar,
):
    if len(descriptors) == 0:
        values = torch.zeros((atom_types.shape[0], 0), dtype=positions.dtype, device=positions.device)
        grads = torch.zeros((atom_types.shape[0], 0, atom_types.shape[0], 3), dtype=positions.dtype, device=positions.device)
        return values, grads

    x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
    compiled = evaluator._compile_descriptors(descriptors)
    if not (real_if_scalar and compiled.all_scalar):
        raise NotImplementedError("analytic_dag gradients currently support real scalar descriptor blocks only")
    _, atomic_base, atomic_base_jacobian = evaluator.site_basis.compute_channels_with_position_jacobian(
        x_ij=x_ij,
        edge_index=edge_index,
        atom_types=atom_types,
        channels=compiled.channels,
        charges=charges,
        aux_tensor_basis=aux_tensor_basis,
    )
    dag = product_dag_from_compiled_cached(compiled, descriptor_count=len(descriptors))
    dag_values = evaluate_product_dag(atomic_base, dag)
    values = checked_real_scalar_projection(
        dag_values.descriptor_values,
        imag_tol=1.0e-10,
        context="analytic_dag path",
    )
    n_atoms, n_descriptors = values.shape
    grads = torch.zeros((n_atoms, n_descriptors, n_atoms, 3), dtype=values.dtype, device=values.device)
    for descriptor_index in range(n_descriptors):
        output_adjoint = torch.zeros_like(dag_values.descriptor_values)
        output_adjoint[:, descriptor_index] = 1.0
        root_adjoint = reverse_product_dag_adjoint(output_adjoint, dag_values, dag)
        descriptor_grad = (root_adjoint[:, :, None, None] * atomic_base_jacobian).sum(dim=1)
        grads[:, descriptor_index, :, :] = descriptor_grad.real
    return values, grads


def _normalize_analytic_product_method(method):
    normalized = str(method).strip().lower().replace("-", "_")
    aliases = {
        "analytical": "analytic",
        "analytical_direct": "analytic",
        "direct": "analytic",
        "analytic_direct": "analytic",
        "analytical_dag": "analytic_dag",
        "streaming": "analytic_streaming",
        "analytical_streaming": "analytic_streaming",
        "analytic_direct_streaming": "analytic_streaming",
        "analytical_dag_streaming": "analytic_dag_streaming",
        "factorized": "analytic_factorized",
        "sym_power": "analytic_factorized",
        "symmetric_power": "analytic_factorized",
        "factorized_streaming": "analytic_factorized_streaming",
        "sym_power_streaming": "analytic_factorized_streaming",
        "symmetric_power_streaming": "analytic_factorized_streaming",
    }
    return aliases.get(normalized, normalized)


def _can_use_real_product_dag_path(
    evaluator,
    *,
    compiled,
    dag,
    atomic_base,
    real_if_scalar,
    imag_tol = 1.0e-10,
):
    if not (real_if_scalar and compiled.all_scalar):
        return False
    if evaluator.site_basis.cfg.spherical_backend != "real":
        return False
    if not torch.is_complex(atomic_base):
        return product_dag_coefficients_are_real(dag, tol=imag_tol)
    if not product_dag_coefficients_are_real(dag, tol=imag_tol):
        return False
    max_imag = torch.max(torch.abs(atomic_base.imag)) if atomic_base.numel() else torch.zeros((), device=atomic_base.device)
    return bool(max_imag <= torch.as_tensor(float(imag_tol), dtype=atomic_base.real.dtype, device=atomic_base.device))


def _direct_product_root_adjoint(
    atomic_base,
    compiled,
    *,
    descriptor_count,
    output_adjoint,
):
    """Evaluate descriptors and reverse products without building a ProductDAG.

    This is intentionally simple and independent from the shared ProductDAG
    implementation. It gives the analytic force path a slower but very useful
    cross-check backend for production debugging.
    """

    if tuple(output_adjoint.shape) != (atomic_base.shape[0], descriptor_count):
        raise ValueError(
            f"output_adjoint must have shape {(atomic_base.shape[0], descriptor_count)}; "
            f"got {tuple(output_adjoint.shape)}"
        )
    return explicit_product_rule_linear_adjoint(
        atomic_base,
        compiled,
        output_adjoint,
    )


def _descriptor_gradients_analytic_direct(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    shifts,
    charges,
    aux_tensor_basis,
    real_if_scalar,
):
    if len(descriptors) == 0:
        values = torch.zeros((atom_types.shape[0], 0), dtype=positions.dtype, device=positions.device)
        grads = torch.zeros((atom_types.shape[0], 0, atom_types.shape[0], 3), dtype=positions.dtype, device=positions.device)
        return values, grads

    x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
    compiled = evaluator._compile_descriptors(descriptors)
    if real_if_scalar and not compiled.all_scalar:
        raise NotImplementedError("analytic direct gradients can only project real scalar descriptor blocks")
    _, atomic_base, atomic_base_jacobian = evaluator.site_basis.compute_channels_with_position_jacobian(
        x_ij=x_ij,
        edge_index=edge_index,
        atom_types=atom_types,
        channels=compiled.channels,
        charges=charges,
        aux_tensor_basis=aux_tensor_basis,
    )
    n_atoms = int(atom_types.shape[0])
    n_descriptors = int(len(descriptors))
    values_out = None
    grads = torch.zeros((n_atoms, n_descriptors, n_atoms, 3), dtype=positions.dtype, device=positions.device)
    for descriptor_index in range(n_descriptors):
        output_adjoint = torch.zeros((n_atoms, n_descriptors), dtype=atomic_base.dtype, device=atomic_base.device)
        output_adjoint[:, descriptor_index] = 1.0
        descriptor_values, root_adjoint = _direct_product_root_adjoint(
            atomic_base,
            compiled,
            descriptor_count=n_descriptors,
            output_adjoint=output_adjoint,
        )
        if values_out is None:
            values_out = descriptor_values
        descriptor_grad = (root_adjoint[:, :, None, None] * atomic_base_jacobian).sum(dim=1)
        grads[:, descriptor_index, :, :] = descriptor_grad.real
    values = values_out if values_out is not None else torch.zeros((n_atoms, n_descriptors), dtype=atomic_base.dtype, device=atomic_base.device)
    if real_if_scalar and compiled.all_scalar:
        values = checked_real_scalar_projection(values, imag_tol=1.0e-10, context="analytic path")
    return values, grads


def _reverse_real_tesseral_to_complex_multiplet(output_adjoint, L, dtype):
    L = int(L)
    if L == 0:
        if output_adjoint.shape[-1] == 1:
            return output_adjoint.to(dtype=dtype)
        return output_adjoint.unsqueeze(-1).to(dtype=dtype)
    rt2 = torch.sqrt(torch.tensor(2.0, dtype=output_adjoint.real.dtype, device=output_adjoint.device))
    out = torch.zeros(output_adjoint.shape[:-1] + (2 * L + 1,), dtype=output_adjoint.dtype, device=output_adjoint.device)
    for m in range(1, L + 1):
        sign = (-1) ** m
        adj_pos = output_adjoint[..., m + L]
        adj_neg = output_adjoint[..., -m + L]
        out[..., L - m] = (sign * adj_pos + adj_neg) / rt2
        out[..., L + m] = (sign * (-1j) * adj_pos + 1j * adj_neg) / rt2
    out[..., L] = output_adjoint[..., L]
    return out.to(dtype=dtype)


def _real_to_complex_matrix_cpu(L):
    L = int(L)
    cached = _REAL_TESSERAL_COMPLEX_MATRIX_CACHE.get(L)
    if cached is not None:
        return cached
    eye = torch.eye(2 * L + 1, dtype=torch.float64)
    matrix = real_tesseral_to_complex_multiplet(eye, L).detach().cpu()
    cached = [[complex(matrix[real_index, complex_index].item()) for complex_index in range(2 * L + 1)] for real_index in range(2 * L + 1)]
    _REAL_TESSERAL_COMPLEX_MATRIX_CACHE[L] = cached
    return cached


def _real_factorized_schedule_tensors(schedule, *, device, dtype, block_count, max_m_dim):
    key = (str(device), str(dtype), int(block_count), int(max_m_dim))
    entries = _REAL_FACTORIZED_SCHEDULE_CACHE.get(schedule)
    cached = None if entries is None else entries.get(key)
    if cached is not None:
        return cached
    component_index = schedule.term_component_index.detach().cpu().tolist()
    label_index = schedule.term_label_index.detach().cpu().tolist()
    block_m_indices = schedule.block_m_indices.detach().cpu().tolist()
    block_L_by_label = schedule.block_L_by_label.detach().cpu().tolist()
    coeffs = schedule.coeffs.detach().cpu().tolist()
    accum = {}
    for term_index, complex_row in enumerate(block_m_indices):
        label = int(label_index[term_index])
        component = int(component_index[term_index])
        expansions_by_block = []
        for block_index, complex_m_index in enumerate(complex_row):
            block_L = int(block_L_by_label[label][block_index])
            transform = _real_to_complex_matrix_cpu(block_L)
            entries = []
            for real_index in range(2 * block_L + 1):
                value = transform[real_index][int(complex_m_index)]
                if abs(value) > 1.0e-14:
                    entries.append((int(real_index), value))
            expansions_by_block.append(tuple(entries))
        for expanded in itertools.product(*expansions_by_block):
            real_row = tuple(int(item[0]) for item in expanded)
            value = complex(coeffs[term_index])
            for _, transform_value in expanded:
                value *= complex(transform_value)
            key_row = (component, label, real_row)
            accum[key_row] = accum.get(key_row, 0.0 + 0.0j) + value
    rows = []
    for (component, label, real_row), value in sorted(accum.items()):
        if abs(value) <= 1.0e-12:
            continue
        if abs(value.imag) > 1.0e-10:
            return None
        rows.append((int(component), int(label), tuple(int(v) for v in real_row), float(value.real)))
    if rows:
        real_component_index = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
        real_label_index = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
        real_block_indices = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
        real_coeffs = torch.tensor([row[3] for row in rows], dtype=dtype, device=device)
    else:
        real_component_index = torch.empty((0,), dtype=torch.long, device=device)
        real_label_index = torch.empty((0,), dtype=torch.long, device=device)
        real_block_indices = torch.empty((0, int(block_count)), dtype=torch.long, device=device)
        real_coeffs = torch.empty((0,), dtype=dtype, device=device)
    real_local_columns = (
        real_label_index.view(-1, 1) * (int(block_count) * int(max_m_dim))
        + torch.arange(int(block_count), dtype=torch.long, device=device).view(1, int(block_count)) * int(max_m_dim)
        + real_block_indices
    ).contiguous()
    cached = (real_coeffs, real_component_index, real_label_index, real_block_indices, real_local_columns)
    if entries is None:
        entries = {}
        _REAL_FACTORIZED_SCHEDULE_CACHE[schedule] = entries
    entries[key] = cached
    return cached


def _real_factorized_schedule_forward_reverse(
    block_values,
    schedule,
    output_adjoint,
):
    values = block_values.real if torch.is_complex(block_values) else block_values
    values = values.contiguous()
    output_adjoint = output_adjoint.real if torch.is_complex(output_adjoint) else output_adjoint
    output_adjoint = output_adjoint.to(dtype=values.dtype)
    n_atoms = int(values.shape[0])
    basis_count = int(values.shape[1])
    block_count = int(values.shape[2])
    max_m_dim = int(values.shape[3])
    tensors = _real_factorized_schedule_tensors_auto(
        schedule,
        device=values.device,
        dtype=values.dtype,
        block_count=block_count,
        max_m_dim=max_m_dim,
    )
    if tensors is None:
        return None
    coeffs, component_index, label_index, block_indices, local_columns = tensors
    component_count = int(schedule.component_count)
    grouped_out = torch.zeros((n_atoms, component_count), dtype=values.dtype, device=values.device)
    block_adjoint = torch.zeros_like(values)
    term_count = int(coeffs.numel())
    if term_count == 0:
        return grouped_out, block_adjoint
    atom_index = torch.arange(n_atoms, dtype=torch.long, device=values.device).view(n_atoms, 1, 1)
    block_index = torch.arange(block_count, dtype=torch.long, device=values.device).view(1, 1, block_count)
    selected = values[
        atom_index,
        label_index.view(1, term_count, 1),
        block_index,
        block_indices.view(1, term_count, block_count),
    ]
    term_values = selected.prod(dim=2) * coeffs.view(1, term_count)
    grouped_out.index_add_(1, component_index, term_values)
    term_adjoint = output_adjoint.index_select(1, component_index) * coeffs.view(1, term_count)
    one = torch.ones((n_atoms, term_count, 1), dtype=values.dtype, device=values.device)
    if block_count == 1:
        partial_by_block = one
    else:
        left = torch.cat([one, torch.cumprod(selected[:, :, :-1], dim=2)], dim=2)
        right = torch.cat(
            [
                torch.cumprod(torch.flip(selected[:, :, 1:], dims=(2,)), dim=2).flip(2),
                one,
            ],
            dim=2,
        )
        partial_by_block = left * right
    contribution = term_adjoint.unsqueeze(-1) * partial_by_block
    flat = block_adjoint.reshape(n_atoms, basis_count * block_count * max_m_dim)
    flat.index_add_(1, local_columns.reshape(-1), contribution.reshape(n_atoms, term_count * block_count))
    return grouped_out, block_adjoint


def _real_factorized_schedule_tensors_auto(schedule, *, device, dtype, block_count, max_m_dim):
    tensors = _real_factorized_schedule_tensors(
        schedule,
        device=device,
        dtype=dtype,
        block_count=block_count,
        max_m_dim=max_m_dim,
    )
    if tensors is None:
        return None
    mode = _real_factorized_reverse_mode()
    if mode in {"force", "strict", "1", "true", "yes", "on"}:
        return tensors
    real_term_count = int(tensors[0].numel())
    complex_term_count = int(schedule.coeffs.numel())
    max_ratio = float(os.environ.get("YE3T_ACE_REAL_FACTORIZED_MAX_TERM_RATIO", "1.0"))
    if real_term_count < max_ratio * max(1, complex_term_count):
        return tensors
    return None


def _real_symmetric_power_block_value(x, *, power, input_L, output_L, multiplicity_index):
    tensors = _real_symmetric_power_tensors(
        power=int(power),
        input_L=int(input_L),
        output_L=int(output_L),
        multiplicity_index=int(multiplicity_index),
        device=x.device,
        dtype=x.dtype,
    )
    if tensors is None:
        return None
    monomial_indices, output_m, values = tensors
    out_dim = 2 * int(output_L) + 1
    out = torch.zeros((int(x.shape[0]), out_dim), dtype=x.dtype, device=x.device)
    if int(monomial_indices.numel()) == 0:
        return out
    monomials = torch.ones((int(x.shape[0]), int(monomial_indices.shape[0])), dtype=x.dtype, device=x.device)
    for slot in range(int(power)):
        monomials = monomials * x[:, monomial_indices[:, slot]]
    out.index_add_(1, output_m, monomials * values.view(1, -1))
    return out


def _real_symmetric_power_block_reverse(x, output_adjoint, *, power, input_L, output_L, multiplicity_index):
    tensors = _real_symmetric_power_tensors(
        power=int(power),
        input_L=int(input_L),
        output_L=int(output_L),
        multiplicity_index=int(multiplicity_index),
        device=x.device,
        dtype=x.dtype,
    )
    if tensors is None:
        return None
    indices, output_m, values = tensors
    x_adjoint = torch.zeros_like(x)
    if int(indices.numel()) == 0:
        return x_adjoint
    selected = x.index_select(1, indices.reshape(-1)).reshape(int(x.shape[0]), int(indices.shape[0]), int(power))
    term_adjoint = output_adjoint.index_select(1, output_m) * values.view(1, -1)
    for slot in range(int(power)):
        partial = torch.ones((int(x.shape[0]), int(indices.shape[0])), dtype=x.dtype, device=x.device)
        for other_slot in range(int(power)):
            if other_slot != slot:
                partial = partial * selected[:, :, other_slot]
        x_adjoint.index_add_(1, indices[:, slot], term_adjoint * partial)
    return x_adjoint


def _real_symmetric_power_tensors(*, power, input_L, output_L, multiplicity_index, device, dtype):
    key = (
        int(power),
        int(input_L),
        int(output_L),
        int(multiplicity_index),
        str(device),
        str(dtype),
    )
    cached = _REAL_SYMMETRIC_POWER_REVERSE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        table = ace_real_symmetric_power_table(
            int(power),
            int(input_L),
            int(output_L),
            int(multiplicity_index),
            device=device,
            dtype=dtype,
        )
    except Exception:
        return None
    cached = (table.monomial_indices, table.output_m, table.values)
    _REAL_SYMMETRIC_POWER_REVERSE_CACHE[key] = cached
    return cached


def _real_symmetric_power_tensors_from_complex(*, power, input_L, output_L, multiplicity_index, device, dtype):
    entries = _complex_symmetric_power_entries(
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
                key_row = (int(output_real), real_row)
                accum[key_row] = accum.get(key_row, 0.0 + 0.0j) + value
    rows = []
    for (output_real, real_row), value in sorted(accum.items()):
        if abs(value) <= 1.0e-12:
            continue
        if abs(value.imag) > 1.0e-10:
            return None
        rows.append((int(output_real), tuple(int(v) for v in real_row), float(value.real)))
    if rows:
        monomial_indices = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
        output_m = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
        values = torch.tensor([row[2] for row in rows], dtype=dtype, device=device)
    else:
        monomial_indices = torch.empty((0, int(power)), dtype=torch.long, device=device)
        output_m = torch.empty((0,), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=dtype, device=device)
    return monomial_indices, output_m, values


def _real_factorized_reverse_mode():
    return str(os.environ.get("YE3T_ACE_REAL_FACTORIZED_REVERSE", "auto")).strip().lower()


def _real_factorized_reverse_enabled():
    mode = _real_factorized_reverse_mode()
    return mode not in {"0", "false", "off", "no"}


def _native_real_factorized_cost_accepts(group, atomic_base):
    mode = _real_factorized_reverse_mode()
    if mode in {"force", "strict", "1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "off", "no"}:
        return False
    real_schedule = group.real_schedule
    if real_schedule is None:
        return False
    real_terms = int(real_schedule.term_count)
    complex_terms = max(1, int(real_schedule.complex_term_count))
    sym_count = 0
    max_power = 0
    for specs in group.block_specs:
        for spec in specs:
            if str(spec["kind"]) == "sym":
                sym_count += 1
                max_power = max(max_power, int(spec["k_b"]))
    if real_terms <= 0:
        return False
    if real_terms < float(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_DIRECT_TERM_RATIO", "0.75")) * complex_terms:
        return True
    min_auto_sym_power = int(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_MIN_AUTO_SYM_POWER", "2"))
    if sym_count > 0 and max_power >= min_auto_sym_power:
        max_ratio = float(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_HIGH_RANK_TERM_RATIO", "2.0"))
        return bool(real_terms <= max_ratio * complex_terms)
    benchmark_state = str(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_BENCHMARK_STATE", "")).strip().lower()
    if benchmark_state in {"pass", "passed", "native_real_passed"}:
        max_ratio = float(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_BENCHMARKED_TERM_RATIO", "1.25"))
        return bool(sym_count > 0 and real_terms <= max_ratio * complex_terms)
    if not atomic_base.is_cuda:
        return False
    if sym_count > 0 and max_power >= min_auto_sym_power and real_terms < complex_terms:
        return True
    return False


def _atomic_base_has_real_values(atomic_base, imag_tol = 1.0e-10):
    if not torch.is_complex(atomic_base):
        return True
    if not atomic_base.numel():
        return True
    max_imag = torch.max(torch.abs(atomic_base.imag))
    return bool(max_imag <= torch.as_tensor(float(imag_tol), dtype=atomic_base.real.dtype, device=atomic_base.device))


def _can_use_real_linear_product_path(evaluator, compiled, atomic_base, real_if_scalar):
    if not (real_if_scalar and compiled.all_scalar):
        return False
    if evaluator.site_basis.cfg.spherical_backend != "real":
        return False
    return _atomic_base_has_real_values(atomic_base)


def _add_linear_form_accumulator(site_linear, root_adjoint, add_linear, add_root):
    dtype = torch.promote_types(site_linear.dtype, add_linear.dtype)
    dtype = torch.promote_types(dtype, root_adjoint.dtype)
    dtype = torch.promote_types(dtype, add_root.dtype)
    return (
        site_linear.to(dtype=dtype) + add_linear.to(dtype=dtype),
        root_adjoint.to(dtype=dtype) + add_root.to(dtype=dtype),
    )


def _factorized_descriptor_group_linear_form_root_adjoint_real(evaluator, group, schedule, atomic_base, group_weights):
    if not _real_factorized_reverse_enabled():
        return None
    if evaluator.site_basis.cfg.spherical_backend != "real":
        return None
    if not _atomic_base_has_real_values(atomic_base):
        return None
    real_base = atomic_base.real if torch.is_complex(atomic_base) else atomic_base
    weights = group_weights.real if torch.is_complex(group_weights) else group_weights
    weights = weights.to(dtype=real_base.dtype, device=real_base.device)
    n_atoms = int(real_base.shape[0])
    if _real_factorized_schedule_tensors_auto(
        schedule,
        device=real_base.device,
        dtype=real_base.dtype,
        block_count=int(schedule.block_count),
        max_m_dim=int(schedule.max_block_m_dim),
    ) is None:
        return None
    block_values = torch.zeros(
        (
            n_atoms,
            int(schedule.basis_count),
            int(schedule.block_count),
            int(schedule.max_block_m_dim),
        ),
        dtype=real_base.dtype,
        device=real_base.device,
    )
    block_inputs = {}
    for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
        for block_index, (spec, indices) in enumerate(zip(block_specs, block_indices)):
            indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=real_base.device)
            selected = site_real_block_to_ye3t_tesseral(
                real_base.index_select(1, indices_tensor), int(spec["l"]),
            )
            if str(spec["kind"]) == "leaf":
                value = selected
            else:
                value = _real_symmetric_power_block_value(
                    selected,
                    power=int(spec["k_b"]),
                    input_L=int(spec["l"]),
                    output_L=int(spec["Lambda"]),
                    multiplicity_index=int(spec["multiplicity_index"]),
                )
                if value is None:
                    return None
            width = int(value.shape[-1])
            block_values[:, int(label_index), int(block_index), :width] = value
            block_inputs[(int(label_index), int(block_index))] = (spec, indices_tensor, selected, width)
    scheduled = _real_factorized_schedule_forward_reverse(
        block_values,
        schedule,
        weights,
    )
    if scheduled is None:
        return None
    grouped_out, block_adjoint = scheduled
    site_linear = (grouped_out * weights).sum(dim=1)
    root_adjoint = torch.zeros_like(real_base)
    for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
        for block_index, _ in enumerate(zip(block_specs, block_indices)):
            spec, indices_tensor, selected, width = block_inputs[(int(label_index), int(block_index))]
            block_out_adjoint = block_adjoint[:, int(label_index), int(block_index), :width]
            if str(spec["kind"]) == "leaf":
                x_adjoint = block_out_adjoint
            else:
                x_adjoint = _real_symmetric_power_block_reverse(
                    selected,
                    block_out_adjoint,
                    power=int(spec["k_b"]),
                    input_L=int(spec["l"]),
                    output_L=int(spec["Lambda"]),
                    multiplicity_index=int(spec["multiplicity_index"]),
                )
                if x_adjoint is None:
                    return None
            root_adjoint.index_add_(
                1, indices_tensor,
                site_real_block_to_ye3t_tesseral(x_adjoint, int(spec["l"])),
            )
    return site_linear, root_adjoint


def _native_real_schedule_forward_reverse(block_values, real_schedule, output_adjoint):
    schedule = real_schedule.to_torch(device=block_values.device, dtype=block_values.dtype)
    values = block_values.contiguous()
    output_adjoint = output_adjoint.to(dtype=values.dtype, device=values.device)
    n_atoms = int(values.shape[0])
    basis_count = int(values.shape[1])
    block_count = int(values.shape[2])
    max_m_dim = int(values.shape[3])
    term_count = int(schedule.term_count)
    component_count = int(schedule.component_count)
    mode = str(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_TRITON", "auto")).strip().lower()
    use_triton = bool(
        triton is not None
        and mode not in {"0", "false", "off", "no"}
        and values.is_cuda
        and values.dtype in (torch.float64,)
        and not torch.is_grad_enabled()
        and term_count > 0
        and block_count > 0
        and block_count <= int(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_TRITON_MAX_BLOCKS", "8"))
        and int(values.numel()) >= int(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_TRITON_MIN_ELEMENTS", "1024"))
    )
    if use_triton:
        key = (int(block_count), str(values.dtype), str(values.device))
        if key not in _NATIVE_REAL_FACTORIZED_TRITON_FAILURES:
            grouped_out = torch.zeros((n_atoms, component_count), dtype=values.dtype, device=values.device)
            block_adjoint = torch.zeros_like(values)
            block_atoms = int(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_TRITON_BLOCK_ATOMS", "16"))
            block_terms = int(os.environ.get("YE3T_ACE_NATIVE_REAL_FACTORIZED_TRITON_BLOCK_TERMS", "32"))
            if block_atoms != 16 or block_terms != 32:
                if mode in {"force", "strict"}:
                    raise RuntimeError(
                        "The current factorized Triton specialization requires "
                        "BLOCK_ATOMS=16 and BLOCK_TERMS=32."
                    )
                use_triton = False
        if use_triton and key not in _NATIVE_REAL_FACTORIZED_TRITON_FAILURES:
            try:
                adj_values = output_adjoint.contiguous()
                grid = (triton.cdiv(n_atoms, block_atoms), triton.cdiv(term_count, block_terms))
                _native_real_factorized_forward_reverse_kernel[grid](
                    values,
                    adj_values,
                    schedule.coeffs.contiguous(),
                    schedule.term_component_index.contiguous(),
                    schedule.term_label_index.contiguous(),
                    schedule.block_real_indices.contiguous(),
                    grouped_out,
                    block_adjoint,
                    n_atoms,
                    term_count,
                    component_count,
                    basis_count,
                    max_m_dim,
                    values.stride(0),
                    values.stride(1),
                    values.stride(2),
                    values.stride(3),
                    grouped_out.stride(0),
                    grouped_out.stride(1),
                    adj_values.stride(0),
                    adj_values.stride(1),
                    block_adjoint.stride(0),
                    block_adjoint.stride(1),
                    block_adjoint.stride(2),
                    block_adjoint.stride(3),
                    block_count,
                )
                return grouped_out, block_adjoint, "triton_real_factorized_forward_reverse"
            except Exception:
                _NATIVE_REAL_FACTORIZED_TRITON_FAILURES.add(key)
                if mode in {"force", "strict"}:
                    raise
    grouped_out = evaluate_real_factorized_schedule_torch(values, schedule, backend="torch")
    block_adjoint = torch.zeros_like(values)
    if term_count == 0:
        return grouped_out, block_adjoint, "torch_real_factorized_forward_reverse"
    atom_index = torch.arange(n_atoms, dtype=torch.long, device=values.device).view(n_atoms, 1, 1)
    block_index = torch.arange(block_count, dtype=torch.long, device=values.device).view(1, 1, block_count)
    selected = values[
        atom_index,
        schedule.term_label_index.view(1, term_count, 1),
        block_index,
        schedule.block_real_indices.view(1, term_count, block_count),
    ]
    term_adjoint = output_adjoint.index_select(1, schedule.term_component_index) * schedule.coeffs.view(1, term_count)
    one = torch.ones((n_atoms, term_count, 1), dtype=values.dtype, device=values.device)
    if block_count == 1:
        partial_by_block = one
    else:
        left = torch.cat([one, torch.cumprod(selected[:, :, :-1], dim=2)], dim=2)
        right = torch.cat(
            [
                torch.cumprod(torch.flip(selected[:, :, 1:], dims=(2,)), dim=2).flip(2),
                one,
            ],
            dim=2,
        )
        partial_by_block = left * right
    block_columns = (
        schedule.term_label_index.view(term_count, 1) * (block_count * max_m_dim)
        + torch.arange(block_count, dtype=torch.long, device=values.device).view(1, block_count) * max_m_dim
        + schedule.block_real_indices
    )
    contribution = term_adjoint.unsqueeze(-1) * partial_by_block
    flat = block_adjoint.reshape(n_atoms, basis_count * block_count * max_m_dim)
    flat.index_add_(1, block_columns.reshape(-1), contribution.reshape(n_atoms, term_count * block_count))
    return grouped_out, block_adjoint, "torch_real_factorized_forward_reverse"


def _factorized_descriptor_group_linear_form_root_adjoint_native_real(evaluator, group, atomic_base, group_weights):
    if group.real_schedule is None:
        return None
    if not _real_factorized_reverse_enabled():
        return None
    if evaluator.site_basis.cfg.spherical_backend != "real":
        return None
    if not _atomic_base_has_real_values(atomic_base):
        return None
    if not _native_real_factorized_cost_accepts(group, atomic_base):
        return None
    real_base = atomic_base.real if torch.is_complex(atomic_base) else atomic_base
    weights = group_weights.real if torch.is_complex(group_weights) else group_weights
    weights = weights.to(dtype=real_base.dtype, device=real_base.device)
    n_atoms = int(real_base.shape[0])
    real_schedule = group.real_schedule
    block_values = torch.zeros(
        (
            n_atoms,
            int(real_schedule.basis_count),
            int(real_schedule.block_count),
            int(2 * max([0] + [int(spec["Lambda"] if spec["kind"] == "sym" else spec["l"]) for specs in real_schedule.block_specs for spec in specs]) + 1),
        ),
        dtype=real_base.dtype,
        device=real_base.device,
    )
    block_inputs = {}
    block_value_cache = {}
    block_occurrences = {}
    block_value_calls = 0
    for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
        for block_index, (spec, indices) in enumerate(zip(block_specs, block_indices)):
            spec_key = tuple(sorted((str(k), str(v)) for k, v in dict(spec).items()))
            cache_key = (spec_key, tuple(int(v) for v in indices))
            cached = block_value_cache.get(cache_key)
            if cached is None:
                indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=real_base.device)
                selected = site_real_block_to_ye3t_tesseral(
                    real_base.index_select(1, indices_tensor), int(spec["l"]),
                )
                if str(spec["kind"]) == "leaf":
                    value = selected
                else:
                    value = _real_symmetric_power_block_value(
                        selected,
                        power=int(spec["k_b"]),
                        input_L=int(spec["l"]),
                        output_L=int(spec["Lambda"]),
                        multiplicity_index=int(spec["multiplicity_index"]),
                    )
                    if value is None:
                        return None
                block_value_calls += 1
                cached = (spec, indices_tensor, selected, value, int(value.shape[-1]))
                block_value_cache[cache_key] = cached
            spec, indices_tensor, selected, value, width = cached
            width = int(value.shape[-1])
            block_values[:, int(label_index), int(block_index), :width] = value
            block_inputs[(int(label_index), int(block_index))] = (cache_key, width)
            block_occurrences.setdefault(cache_key, 0)
            block_occurrences[cache_key] += 1
    grouped_out, block_adjoint, schedule_backend = _native_real_schedule_forward_reverse(
        block_values,
        real_schedule,
        weights,
    )
    evaluator._last_backend_counts["atomic_product_backend:" + str(schedule_backend)] = (
        evaluator._last_backend_counts.get("atomic_product_backend:" + str(schedule_backend), 0)
        + int(len(group.descriptor_indices))
    )
    site_linear = (grouped_out * weights).sum(dim=1)
    root_adjoint = torch.zeros_like(real_base)
    block_adjoint_cache = {}
    for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
        for block_index, _ in enumerate(zip(block_specs, block_indices)):
            cache_key, width = block_inputs[(int(label_index), int(block_index))]
            block_out_adjoint = block_adjoint[:, int(label_index), int(block_index), :width]
            cached_adjoint = block_adjoint_cache.get(cache_key)
            if cached_adjoint is None:
                block_adjoint_cache[cache_key] = block_out_adjoint.clone()
            else:
                cached_adjoint.add_(block_out_adjoint)
    for cache_key, block_out_adjoint in block_adjoint_cache.items():
        spec, indices_tensor, selected, value, width = block_value_cache[cache_key]
        block_out_adjoint = block_out_adjoint[:, :width]
        if str(spec["kind"]) == "leaf":
            x_adjoint = block_out_adjoint
        else:
            x_adjoint = _real_symmetric_power_block_reverse(
                selected,
                block_out_adjoint,
                power=int(spec["k_b"]),
                input_L=int(spec["l"]),
                output_L=int(spec["Lambda"]),
                multiplicity_index=int(spec["multiplicity_index"]),
            )
            if x_adjoint is None:
                return None
        root_adjoint.index_add_(
            1, indices_tensor,
            site_real_block_to_ye3t_tesseral(x_adjoint, int(spec["l"])),
        )
    evaluator._last_backend_counts["native_real_block_value_calls"] = (
        evaluator._last_backend_counts.get("native_real_block_value_calls", 0)
        + int(block_value_calls)
    )
    evaluator._last_backend_counts["native_real_block_occurrences"] = (
        evaluator._last_backend_counts.get("native_real_block_occurrences", 0)
        + int(sum(block_occurrences.values()))
    )
    evaluator._last_backend_counts["native_real_block_reverse_calls"] = (
        evaluator._last_backend_counts.get("native_real_block_reverse_calls", 0)
        + int(len(block_adjoint_cache))
    )
    return site_linear, root_adjoint


def _symmetric_power_compile_allowed(x):
    mode = str(os.environ.get("YE3T_ACE_COMPILE_SYMMETRIC_POWER_REVERSE", "1")).strip().lower()
    if mode in {"0", "false", "off", "no"}:
        return False
    if torch.is_complex(x) and mode not in {"force", "strict"}:
        return False
    return bool(
        hasattr(torch, "compile")
        and x.is_cuda
        and not torch.is_grad_enabled()
    )


def _symmetric_power_reverse_tensor(
    x,
    output_adjoint,
    monomial_indices,
    output_index,
    coeffs,
):
    x_adjoint = torch.zeros(
        output_adjoint.shape[:-1] + (int(x.shape[-1]),),
        dtype=output_adjoint.dtype,
        device=output_adjoint.device,
    )
    if int(monomial_indices.numel()) == 0:
        return x_adjoint
    selected = x.index_select(1, monomial_indices.reshape(-1)).reshape(
        int(x.shape[0]),
        int(monomial_indices.shape[0]),
        int(monomial_indices.shape[1]),
    )
    term_adjoint = output_adjoint.index_select(2, output_index) * coeffs.reshape(1, 1, -1)
    flat = x_adjoint.reshape(-1, int(x_adjoint.shape[-1]))
    batch_atoms = int(flat.shape[0])
    power_int = int(monomial_indices.shape[1])
    for slot in range(power_int):
        partial = torch.ones(
            (int(selected.shape[0]), int(selected.shape[1])),
            dtype=selected.dtype,
            device=selected.device,
        )
        for other_slot in range(power_int):
            if other_slot != slot:
                partial = partial * selected[:, :, other_slot]
        contribution = term_adjoint * partial.unsqueeze(0)
        flat.index_add_(
            1,
            monomial_indices[:, int(slot)],
            contribution.reshape(batch_atoms, -1),
    )
    return x_adjoint


def _compiled_symmetric_power_reverse_tensor():
    mode = str(os.environ.get("YE3T_ACE_COMPILE_SYMMETRIC_POWER_REVERSE", "1")).strip().lower()
    key = str(mode)
    cached = _SYMMETRIC_POWER_REVERSE_COMPILED_CACHE.get(key)
    if cached is not None:
        return cached
    if not hasattr(torch, "compile"):
        return _symmetric_power_reverse_tensor
    compile_kwargs = {"dynamic": False}
    if mode not in {"", "1", "true", "yes", "on", "auto"}:
        compile_kwargs["mode"] = mode
    try:
        cached = torch.compile(_symmetric_power_reverse_tensor, **compile_kwargs)
    except Exception:
        cached = _symmetric_power_reverse_tensor
    _SYMMETRIC_POWER_REVERSE_COMPILED_CACHE[key] = cached
    return cached


def _complex_symmetric_power_block_reverse(x, output_adjoint, *, power, input_L, output_L, multiplicity_index):
    output_adjoint = output_adjoint.to(dtype=x.dtype, device=x.device)
    plan = _complex_symmetric_power_product_plan(
        int(power),
        int(input_L),
        int(output_L),
        int(multiplicity_index),
    )
    # The analytic ACE chain uses the bilinear transpose J^T s, while the
    # shared runtime exposes the PyTorch/Hilbert adjoint J^H u.
    return symmetric_power_product_plan_batched_adjoint(
        output_adjoint.conj(),
        x,
        plan,
        backend="auto",
    ).conj()


def _factorized_reverse_compile_mode():
    return str(os.environ.get("YE3T_ACE_COMPILE_FACTORIZED_REVERSE", "1")).strip().lower()


def _factorized_reverse_compile_allowed(values):
    mode = _factorized_reverse_compile_mode()
    if mode in {"0", "false", "off", "no"}:
        return False
    if torch.is_complex(values) and mode not in {"force", "strict"}:
        return False
    return bool(
        hasattr(torch, "compile")
        and values.is_cuda
        and not torch.is_grad_enabled()
    )


def _factorized_schedule_to_device_cached(schedule, *, device, dtype):
    key = (str(device), str(dtype))
    entries = _FACTORIZED_SCHEDULE_DEVICE_CACHE.get(schedule)
    cached = None if entries is None else entries.get(key)
    if cached is None:
        cached = schedule.to(device=device, dtype=dtype)
        if entries is None:
            entries = {}
            _FACTORIZED_SCHEDULE_DEVICE_CACHE[schedule] = entries
        entries[key] = cached
    return cached


def _factorized_schedule_reverse_tensors(schedule, *, device, dtype, block_count, max_m_dim):
    key = (
        str(device),
        str(dtype),
        int(block_count),
        int(max_m_dim),
    )
    entries = _FACTORIZED_REVERSE_TENSOR_CACHE.get(schedule)
    cached = None if entries is None else entries.get(key)
    if cached is not None:
        return cached
    coeffs = schedule.coeffs.to(device=device)
    if not torch.is_complex(torch.empty((), dtype=dtype)) and torch.is_complex(coeffs):
        if int(coeffs.numel()) and torch.max(torch.abs(coeffs.imag)).item() <= 1e-12:
            coeffs = coeffs.real.to(dtype=dtype)
    coeffs = coeffs.to(dtype=dtype)
    component_index = schedule.term_component_index.to(device=device, dtype=torch.long).contiguous()
    label_index = schedule.term_label_index.to(device=device, dtype=torch.long).contiguous()
    block_m_indices = schedule.block_m_indices.to(device=device, dtype=torch.long).contiguous()
    term_count = int(block_m_indices.shape[0])
    block_index = torch.arange(int(block_count), device=device, dtype=torch.long).view(1, int(block_count))
    local_columns = (
        label_index.view(term_count, 1) * (int(block_count) * int(max_m_dim))
        + block_index * int(max_m_dim)
        + block_m_indices
    ).contiguous()
    cached = (coeffs, component_index, label_index, block_m_indices, local_columns)
    if entries is None:
        entries = {}
        _FACTORIZED_REVERSE_TENSOR_CACHE[schedule] = entries
    entries[key] = cached
    return cached


def _factorized_schedule_block_adjoint_tensor(
    values,
    output_adjoint,
    coeffs,
    component_index,
    label_index,
    block_m_indices,
    local_columns,
):
    output_adjoint = output_adjoint.to(dtype=values.dtype)
    coeffs = coeffs.to(dtype=values.dtype)
    block_adjoint = torch.zeros(
        (int(output_adjoint.shape[0]),) + tuple(values.shape),
        dtype=values.dtype,
        device=values.device,
    )
    term_count = int(block_m_indices.shape[0])
    block_count = int(values.shape[2])
    if term_count == 0 or block_count == 0:
        return block_adjoint
    term_adjoint = output_adjoint.index_select(2, component_index) * coeffs.view(1, 1, -1)
    n_atoms = int(values.shape[0])
    basis_count = int(values.shape[1])
    max_m_dim = int(values.shape[3])
    atom_index = torch.arange(n_atoms, device=values.device, dtype=torch.long).view(n_atoms, 1, 1)
    block_index = torch.arange(block_count, device=values.device, dtype=torch.long).view(1, 1, block_count)
    selected = values[
        atom_index,
        label_index.view(1, term_count, 1),
        block_index,
        block_m_indices.view(1, term_count, block_count),
    ]
    one = torch.ones((n_atoms, term_count, 1), dtype=selected.dtype, device=selected.device)
    if block_count == 1:
        partial_by_block = one
    else:
        left = torch.cat([one, torch.cumprod(selected[:, :, :-1], dim=2)], dim=2)
        right = torch.cat(
            [
                torch.cumprod(torch.flip(selected[:, :, 1:], dims=(2,)), dim=2).flip(2),
                one,
            ],
            dim=2,
        )
        partial_by_block = left * right
    contribution = term_adjoint.unsqueeze(-1) * partial_by_block.unsqueeze(0)
    flat = block_adjoint.reshape(int(output_adjoint.shape[0]) * n_atoms, basis_count * block_count * max_m_dim)
    flat.index_add_(
        1,
        local_columns.reshape(-1),
        contribution.reshape(int(output_adjoint.shape[0]) * n_atoms, term_count * block_count),
    )
    return block_adjoint


def _compiled_factorized_schedule_block_adjoint_tensor():
    mode = _factorized_reverse_compile_mode()
    key = str(mode)
    cached = _FACTORIZED_REVERSE_COMPILED_CACHE.get(key)
    if cached is not None:
        return cached
    if not hasattr(torch, "compile"):
        return _factorized_schedule_block_adjoint_tensor
    compile_kwargs = {"dynamic": False}
    if mode not in {"", "1", "true", "yes", "on", "auto"}:
        compile_kwargs["mode"] = mode
    try:
        cached = torch.compile(_factorized_schedule_block_adjoint_tensor, **compile_kwargs)
    except Exception:
        cached = _factorized_schedule_block_adjoint_tensor
    _FACTORIZED_REVERSE_COMPILED_CACHE[key] = cached
    return cached


def _factorized_schedule_block_adjoint(block_values, schedule, output_adjoint):
    if not torch.is_complex(block_values) and torch.is_complex(schedule.coeffs):
        coeffs0 = schedule.coeffs.to(device=block_values.device)
        if int(coeffs0.numel()) and torch.max(torch.abs(coeffs0.imag)).item() <= 1e-12:
            coeffs0 = coeffs0.real.to(dtype=block_values.dtype)
    else:
        coeffs0 = schedule.coeffs.to(device=block_values.device)
    value_dtype = torch.promote_types(block_values.dtype, coeffs0.dtype)
    value_dtype = torch.promote_types(value_dtype, output_adjoint.dtype)
    values = block_values.to(dtype=value_dtype)
    output_adjoint = output_adjoint.to(dtype=value_dtype)
    block_count = int(values.shape[2])
    max_m_dim = int(values.shape[3])
    coeffs, component_index, label_index, block_m_indices, local_columns = _factorized_schedule_reverse_tensors(
        schedule,
        device=values.device,
        dtype=value_dtype,
        block_count=block_count,
        max_m_dim=max_m_dim,
    )
    if _factorized_reverse_compile_allowed(values):
        fn = _compiled_factorized_schedule_block_adjoint_tensor()
        try:
            return fn(values, output_adjoint, coeffs, component_index, label_index, block_m_indices, local_columns)
        except Exception:
            if os.environ.get("YE3T_ACE_STRICT_COMPILE_FACTORIZED_REVERSE") == "1":
                raise
    return _factorized_schedule_block_adjoint_tensor(
        values,
        output_adjoint,
        coeffs,
        component_index,
        label_index,
        block_m_indices,
        local_columns,
    )


def _factorized_descriptor_plan_root_adjoint(evaluator, plan, atomic_base, output_adjoint):
    descriptor_count = int(plan.descriptor_count)
    batch = int(output_adjoint.shape[0])
    n_atoms = int(atomic_base.shape[0])
    work_dtype = evaluator.site_basis.cfg.complex_dtype if evaluator.site_basis.cfg.spherical_backend == "real" else atomic_base.dtype
    out = torch.zeros((n_atoms, descriptor_count), dtype=work_dtype, device=atomic_base.device)
    root_adjoint = torch.zeros((batch, n_atoms, int(atomic_base.shape[1])), dtype=work_dtype, device=atomic_base.device)
    for group in plan.groups:
        schedule = _factorized_schedule_to_device_cached(
            group.schedule,
            device=atomic_base.device,
            dtype=work_dtype,
        )
        block_values = torch.zeros(
            (
                n_atoms,
                int(schedule.basis_count),
                int(schedule.block_count),
                int(schedule.max_block_m_dim),
            ),
            dtype=work_dtype,
            device=atomic_base.device,
        )
        block_inputs = {}
        for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
            for block_index, (spec, indices) in enumerate(zip(block_specs, block_indices)):
                input_L = int(spec["l"])
                indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=atomic_base.device)
                selected = atomic_base.index_select(1, indices_tensor)
                if evaluator.site_basis.cfg.spherical_backend == "real":
                    x = real_tesseral_to_complex_multiplet(
                        site_real_block_to_ye3t_tesseral(selected.real, input_L), input_L,
                    )
                else:
                    x = selected
                if str(spec["kind"]) == "leaf":
                    value = x
                else:
                    value = evaluator._block_value_for_factorized_descriptor(
                        atomic_base,
                        spec=spec,
                        indices=indices,
                    )
                width = int(value.shape[-1])
                block_values[:, int(label_index), int(block_index), :width] = value
                block_inputs[(int(label_index), int(block_index))] = (spec, indices_tensor, x, width)
        grouped_out = evaluate_factorized_schedule_torch(
            block_values,
            schedule,
            backend="torch",
        )
        descriptor_indices = torch.tensor(group.descriptor_indices, dtype=torch.long, device=atomic_base.device)
        out.index_copy_(1, descriptor_indices, grouped_out)
        group_output_adjoint = output_adjoint.index_select(2, descriptor_indices)
        block_adjoint = _factorized_schedule_block_adjoint(block_values, schedule, group_output_adjoint)
        for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
            for block_index, _ in enumerate(zip(block_specs, block_indices)):
                spec, indices_tensor, x, width = block_inputs[(int(label_index), int(block_index))]
                block_out_adjoint = block_adjoint[:, :, int(label_index), int(block_index), :width]
                if str(spec["kind"]) == "leaf":
                    x_adjoint = block_out_adjoint
                else:
                    x_adjoint = _complex_symmetric_power_block_reverse(
                        x,
                        block_out_adjoint,
                        power=int(spec["k_b"]),
                        input_L=int(spec["l"]),
                        output_L=int(spec["Lambda"]),
                        multiplicity_index=int(spec["multiplicity_index"]),
                    )
                if evaluator.site_basis.cfg.spherical_backend == "real":
                    channel_adjoint = _reverse_real_tesseral_to_complex_multiplet(
                        x_adjoint,
                        int(spec["l"]),
                        work_dtype,
                    )
                    channel_adjoint = site_real_block_to_ye3t_tesseral(channel_adjoint, int(spec["l"]))
                else:
                    channel_adjoint = x_adjoint.to(dtype=atomic_base.dtype)
                root_adjoint.index_add_(2, indices_tensor, channel_adjoint)
    return out, root_adjoint


def _direct_symmetric_power_plan_root_adjoint(evaluator, plan, atomic_base, output_adjoint):
    descriptor_count = int(plan.descriptor_count)
    batch = int(output_adjoint.shape[0])
    n_atoms = int(atomic_base.shape[0])
    work_dtype = evaluator.site_basis.cfg.complex_dtype if evaluator.site_basis.cfg.spherical_backend == "real" else atomic_base.dtype
    out = torch.zeros((n_atoms, descriptor_count), dtype=work_dtype, device=atomic_base.device)
    root_adjoint = torch.zeros((batch, n_atoms, int(atomic_base.shape[1])), dtype=work_dtype, device=atomic_base.device)
    for entry in plan.entries:
        spec = dict(entry.block_spec)
        input_L = int(spec["l"])
        indices_tensor = torch.as_tensor(entry.channel_indices, dtype=torch.long, device=atomic_base.device)
        selected = atomic_base.index_select(1, indices_tensor)
        if evaluator.site_basis.cfg.spherical_backend == "real":
            x = real_tesseral_to_complex_multiplet(
                site_real_block_to_ye3t_tesseral(selected.real, input_L), input_L,
            )
        else:
            x = selected
        if str(spec["kind"]) == "leaf":
            value = x
        else:
            value = evaluator._block_value_for_factorized_descriptor(
                atomic_base,
                spec=spec,
                indices=entry.channel_indices,
            )
        component = int(entry.component_index)
        if component < 0 or component >= int(value.shape[-1]):
            raise RuntimeError("Direct symmetric-power component index is outside the evaluated multiplet width.")
        descriptor_index = int(entry.descriptor_index)
        out[:, descriptor_index] = value[:, component]
        block_out_adjoint = torch.zeros(
            (batch, n_atoms, int(value.shape[-1])),
            dtype=work_dtype,
            device=atomic_base.device,
        )
        block_out_adjoint[:, :, component] = output_adjoint[:, :, descriptor_index].to(dtype=work_dtype)
        if str(spec["kind"]) == "leaf":
            x_adjoint = block_out_adjoint
        else:
            x_adjoint = _complex_symmetric_power_block_reverse(
                x,
                block_out_adjoint,
                power=int(spec["k_b"]),
                input_L=input_L,
                output_L=int(spec["Lambda"]),
                multiplicity_index=int(spec["multiplicity_index"]),
            )
        if evaluator.site_basis.cfg.spherical_backend == "real":
            channel_adjoint = _reverse_real_tesseral_to_complex_multiplet(
                x_adjoint,
                input_L,
                work_dtype,
            )
            channel_adjoint = site_real_block_to_ye3t_tesseral(channel_adjoint, input_L)
        else:
            channel_adjoint = x_adjoint.to(dtype=atomic_base.dtype)
        root_adjoint.index_add_(2, indices_tensor, channel_adjoint)
    return out, root_adjoint


def _factorized_descriptor_plan_linear_form_root_adjoint(evaluator, plan, atomic_base, descriptor_weight):
    descriptor_count = int(plan.descriptor_count)
    n_atoms = int(atomic_base.shape[0])
    weights_dtype = descriptor_weight.dtype if torch.is_tensor(descriptor_weight) else atomic_base.dtype
    weights = torch.as_tensor(descriptor_weight, dtype=weights_dtype, device=atomic_base.device)
    if weights.ndim == 1:
        weights = weights.reshape(1, descriptor_count).expand(n_atoms, descriptor_count)
    if tuple(weights.shape) != (n_atoms, descriptor_count):
        raise ValueError(f"descriptor_weight must have shape {(descriptor_count,)} or {(n_atoms, descriptor_count)}")
    work_dtype = weights.dtype
    site_linear = torch.zeros((n_atoms,), dtype=work_dtype, device=atomic_base.device)
    root_adjoint = torch.zeros((n_atoms, int(atomic_base.shape[1])), dtype=work_dtype, device=atomic_base.device)
    for group in plan.groups:
        fallback_work_dtype = evaluator.site_basis.cfg.complex_dtype if evaluator.site_basis.cfg.spherical_backend == "real" else atomic_base.dtype
        schedule = _factorized_schedule_to_device_cached(
            group.schedule,
            device=atomic_base.device,
            dtype=fallback_work_dtype,
        )
        descriptor_indices = torch.tensor(group.descriptor_indices, dtype=torch.long, device=atomic_base.device)
        group_weights = weights.index_select(1, descriptor_indices)
        real_group = _factorized_descriptor_group_linear_form_root_adjoint_native_real(
            evaluator,
            group,
            atomic_base,
            group_weights,
        )
        if real_group is None:
            real_group = _factorized_descriptor_group_linear_form_root_adjoint_real(
                evaluator,
                group,
                schedule,
                atomic_base,
                group_weights,
            )
        if real_group is not None:
            site_linear, root_adjoint = _add_linear_form_accumulator(
                site_linear,
                root_adjoint,
                real_group[0],
                real_group[1],
            )
            continue
        block_values = torch.zeros(
            (
                n_atoms,
                int(schedule.basis_count),
                int(schedule.block_count),
                int(schedule.max_block_m_dim),
            ),
            dtype=fallback_work_dtype,
            device=atomic_base.device,
        )
        block_inputs = {}
        for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
            for block_index, (spec, indices) in enumerate(zip(block_specs, block_indices)):
                input_L = int(spec["l"])
                indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=atomic_base.device)
                selected = atomic_base.index_select(1, indices_tensor)
                if evaluator.site_basis.cfg.spherical_backend == "real":
                    x = real_tesseral_to_complex_multiplet(
                        site_real_block_to_ye3t_tesseral(selected.real, input_L), input_L,
                    )
                else:
                    x = selected
                if str(spec["kind"]) == "leaf":
                    value = x
                else:
                    value = evaluator._block_value_for_factorized_descriptor(
                        atomic_base,
                        spec=spec,
                        indices=indices,
                    )
                width = int(value.shape[-1])
                block_values[:, int(label_index), int(block_index), :width] = value
                block_inputs[(int(label_index), int(block_index))] = (spec, indices_tensor, x, width)
        grouped_out = evaluate_factorized_schedule_torch(
            block_values,
            schedule,
            backend="torch",
        )
        site_add = (grouped_out * group_weights.to(dtype=grouped_out.dtype)).sum(dim=1)
        group_output_adjoint = group_weights.reshape(1, n_atoms, int(group_weights.shape[1]))
        block_adjoint = _factorized_schedule_block_adjoint(block_values, schedule, group_output_adjoint)
        group_root = torch.zeros((n_atoms, int(atomic_base.shape[1])), dtype=block_adjoint.dtype, device=atomic_base.device)
        for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
            for block_index, _ in enumerate(zip(block_specs, block_indices)):
                spec, indices_tensor, x, width = block_inputs[(int(label_index), int(block_index))]
                block_out_adjoint = block_adjoint[0, :, int(label_index), int(block_index), :width]
                if str(spec["kind"]) == "leaf":
                    x_adjoint = block_out_adjoint
                else:
                    x_adjoint = _complex_symmetric_power_block_reverse(
                        x,
                        block_out_adjoint.unsqueeze(0),
                        power=int(spec["k_b"]),
                        input_L=int(spec["l"]),
                        output_L=int(spec["Lambda"]),
                        multiplicity_index=int(spec["multiplicity_index"]),
                    )[0]
                if evaluator.site_basis.cfg.spherical_backend == "real":
                    channel_adjoint = _reverse_real_tesseral_to_complex_multiplet(
                        x_adjoint,
                        int(spec["l"]),
                        block_adjoint.dtype,
                    )
                    channel_adjoint = site_real_block_to_ye3t_tesseral(channel_adjoint, int(spec["l"]))
                else:
                    channel_adjoint = x_adjoint.to(dtype=atomic_base.dtype)
                group_root.index_add_(1, indices_tensor, channel_adjoint)
        site_linear, root_adjoint = _add_linear_form_accumulator(
            site_linear,
            root_adjoint,
            site_add,
            group_root,
        )
    return site_linear, root_adjoint


def _direct_symmetric_power_plan_linear_form_root_adjoint(evaluator, plan, atomic_base, descriptor_weight):
    descriptor_count = int(plan.descriptor_count)
    n_atoms = int(atomic_base.shape[0])
    weights_dtype = descriptor_weight.dtype if torch.is_tensor(descriptor_weight) else atomic_base.dtype
    weights = torch.as_tensor(descriptor_weight, dtype=weights_dtype, device=atomic_base.device)
    if weights.ndim == 1:
        weights = weights.reshape(1, descriptor_count).expand(n_atoms, descriptor_count)
    if tuple(weights.shape) != (n_atoms, descriptor_count):
        raise ValueError(f"descriptor_weight must have shape {(descriptor_count,)} or {(n_atoms, descriptor_count)}")
    output_adjoint = weights.reshape(1, n_atoms, descriptor_count)
    values, root_adjoint = _direct_symmetric_power_plan_root_adjoint(
        evaluator,
        plan,
        atomic_base,
        output_adjoint,
    )
    site_linear = (values * weights.to(dtype=values.dtype)).sum(dim=1)
    return site_linear, root_adjoint[0]


def _accelerated_descriptor_plan_indices(compiled):
    indices = set()
    if compiled.factorized_plan is not None:
        indices.update(int(value) for value in compiled.factorized_plan.active_descriptor_indices)
    if compiled.direct_symmetric_power_plan is not None:
        indices.update(int(value) for value in compiled.direct_symmetric_power_plan.active_descriptor_indices)
    return tuple(sorted(indices))


def _accelerated_descriptor_plan_root_adjoint(evaluator, compiled, atomic_base, output_adjoint):
    descriptor_count = int(output_adjoint.shape[2])
    batch = int(output_adjoint.shape[0])
    n_atoms = int(atomic_base.shape[0])
    work_dtype = output_adjoint.dtype
    if (
        evaluator.site_basis.cfg.spherical_backend == "real"
        and (compiled.factorized_plan is not None or compiled.direct_symmetric_power_plan is not None)
    ):
        work_dtype = torch.promote_types(work_dtype, evaluator.site_basis.cfg.complex_dtype)
    values = torch.zeros((n_atoms, descriptor_count), dtype=work_dtype, device=atomic_base.device)
    root = torch.zeros((batch, n_atoms, int(atomic_base.shape[1])), dtype=work_dtype, device=atomic_base.device)
    if compiled.factorized_plan is not None:
        factorized_values, factorized_root = _factorized_descriptor_plan_root_adjoint(
            evaluator,
            compiled.factorized_plan,
            atomic_base,
            output_adjoint,
        )
        values = values + factorized_values
        root = root + factorized_root
    if compiled.direct_symmetric_power_plan is not None:
        direct_values, direct_root = _direct_symmetric_power_plan_root_adjoint(
            evaluator,
            compiled.direct_symmetric_power_plan,
            atomic_base,
            output_adjoint,
        )
        values = values + direct_values
        root = root + direct_root
    return values, root


def _accelerated_descriptor_plan_linear_form_root_adjoint(evaluator, compiled, atomic_base, weights):
    descriptor_count = int(weights.shape[1])
    n_atoms = int(atomic_base.shape[0])
    output_adjoint = weights.reshape(1, n_atoms, descriptor_count)
    values, root = _accelerated_descriptor_plan_root_adjoint(
        evaluator,
        compiled,
        atomic_base,
        output_adjoint,
    )
    site_linear = (values * weights.to(dtype=values.dtype)).sum(dim=1)
    return site_linear, root[0]


def _residual_descriptor_adjoint(compiled, descriptor_adjoint):
    active = _accelerated_descriptor_plan_indices(compiled)
    if not active:
        return descriptor_adjoint
    residual = descriptor_adjoint.clone()
    residual[:, :, list(active)] = 0.0
    return residual


def _residual_descriptor_weights(compiled, weights):
    active = _accelerated_descriptor_plan_indices(compiled)
    if not active:
        return weights
    residual = weights.clone()
    residual[:, list(active)] = 0.0
    return residual


def descriptor_sum_position_jacobian_analytic_product(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    shifts = None,
    charges = None,
    aux_tensor_basis = None,
    real_if_scalar = True,
    chunk_size = None,
):
    x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
    compiled = evaluator._compile_descriptors(descriptors)
    _, atomic_base, vjp_record = evaluator.site_basis.compute_channels_with_vjp_record(
        x_ij=x_ij,
        edge_index=edge_index,
        atom_types=atom_types,
        channels=compiled.channels,
        charges=charges,
        aux_tensor_basis=aux_tensor_basis,
    )
    descriptor_count = int(len(descriptors))
    zero_adjoint = torch.zeros((1, int(atom_types.shape[0]), descriptor_count), dtype=atomic_base.dtype, device=atomic_base.device)
    raw_values, _ = _accelerated_descriptor_plan_root_adjoint(
        evaluator,
        compiled,
        atomic_base,
        zero_adjoint,
    )
    dag = product_dag_from_compiled_cached(compiled, descriptor_count=descriptor_count)
    dag_values = evaluate_product_dag(atomic_base, dag)
    raw_values = raw_values + dag_values.descriptor_values
    values = raw_values
    if real_if_scalar and compiled.all_scalar:
        values = checked_real_scalar_projection(values, imag_tol=1.0e-10, context="analytic product-adjoint Jacobian path")

    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = descriptor_count
    chunk_size = max(1, int(chunk_size))
    jac_rows = []
    for start in range(0, descriptor_count, chunk_size):
        stop = min(descriptor_count, start + chunk_size)
        descriptor_adjoint = torch.zeros(
            (stop - start, int(atom_types.shape[0]), descriptor_count),
            dtype=raw_values.dtype,
            device=raw_values.device,
        )
        descriptor_adjoint[:, :, start:stop] = torch.eye(
            stop - start,
            dtype=raw_values.dtype,
            device=raw_values.device,
        ).view(stop - start, 1, stop - start)
        root_adjoint = torch.zeros(
            (stop - start, int(atom_types.shape[0]), int(atomic_base.shape[1])),
            dtype=atomic_base.dtype,
            device=atomic_base.device,
        )
        _, accelerated_root = _accelerated_descriptor_plan_root_adjoint(
            evaluator,
            compiled,
            atomic_base,
            descriptor_adjoint,
        )
        root_adjoint = root_adjoint + accelerated_root
        residual_adjoint = _residual_descriptor_adjoint(compiled, descriptor_adjoint)
        if dag.output_terms:
            residual_flat = residual_adjoint.reshape(-1, descriptor_count)
            min_elements = int(os.environ.get("YE3T_ACE_PRODUCT_DAG_BATCHED_REVERSE_MIN_ELEMENTS", "65536"))
            if int(residual_flat.numel()) >= min_elements:
                flat_values = ProductDAGValues(
                    all_node_values=dag_values.all_node_values.unsqueeze(0).expand(
                        int(residual_adjoint.shape[0]),
                        -1,
                        -1,
                    ).reshape(-1, int(dag_values.all_node_values.shape[1])),
                    descriptor_values=dag_values.descriptor_values.unsqueeze(0).expand(
                        int(residual_adjoint.shape[0]),
                        -1,
                        -1,
                    ).reshape(-1, descriptor_count),
                )
                residual_root = reverse_product_dag_adjoint(
                    residual_flat,
                    flat_values,
                    dag,
                ).reshape(
                    int(residual_adjoint.shape[0]),
                    int(atom_types.shape[0]),
                    int(atomic_base.shape[1]),
                )
            else:
                residual_roots = []
                for batch_index in range(int(residual_adjoint.shape[0])):
                    residual_roots.append(
                        reverse_product_dag_adjoint(
                            residual_adjoint[batch_index],
                            dag_values,
                            dag,
                        )
                    )
                residual_root = torch.stack(residual_roots, dim=0)
            root_adjoint = root_adjoint + residual_root
        position_grad = evaluator.site_basis.position_vjp_from_record_batched(vjp_record, root_adjoint)
        jac_rows.append(position_grad.reshape(stop - start, -1))
    jac = torch.cat(jac_rows, dim=0) if jac_rows else torch.zeros((0, int(positions.numel())), dtype=positions.dtype, device=positions.device)
    return values, jac


def descriptor_position_vjp(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    descriptor_adjoint,
    shifts = None,
    charges = None,
    aux_tensor_basis=None,
    real_if_scalar = True,
    method = "analytic_dag",
):
    """Return descriptor values and compact position VJP for ACE blocks.

    This is the production-oriented force primitive: it reverses descriptor
    products with either the shared ProductDAG (``method='analytic_dag'``) or
    an independent direct product-rule loop (``method='analytic'``), then
    contracts root-channel adjoints directly against compact edge derivatives
    and normalization adjoints.

    For complex covariant descriptor blocks, the VJP convention is the gradient
    of ``real(sum(descriptor_adjoint * descriptor_values))`` with respect to
    positions.
    """

    if len(descriptors) == 0:
        values = torch.zeros((atom_types.shape[0], 0), dtype=positions.dtype, device=positions.device)
        grad = torch.zeros((atom_types.shape[0], 3), dtype=positions.dtype, device=positions.device)
        return values, grad

    x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
    compiled = evaluator._compile_descriptors(descriptors)
    product_method = _normalize_analytic_product_method(method)
    if product_method not in {"analytic", "analytic_dag", "analytic_streaming", "analytic_dag_streaming"}:
        raise ValueError("method must be 'analytic', 'analytic_dag', 'analytic_streaming', or 'analytic_dag_streaming'")
    if product_method in {"analytic_streaming", "analytic_dag_streaming"}:
        _, raw_atomic_base, atomic_base = evaluator.site_basis.compute_channels_raw_and_final(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        vjp_record = None
    else:
        _, atomic_base, vjp_record = evaluator.site_basis.compute_channels_with_vjp_record(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        raw_atomic_base = vjp_record.raw_atomic_base
    descriptor_count = len(descriptors)
    zero_accelerated_adjoint = torch.zeros(
        (1, int(atom_types.shape[0]), descriptor_count),
        dtype=atomic_base.dtype,
        device=atomic_base.device,
    )
    accelerated_values, _ = _accelerated_descriptor_plan_root_adjoint(
        evaluator,
        compiled,
        atomic_base,
        zero_accelerated_adjoint,
    )
    real_dag_values = None
    real_dag_path = False
    if product_method in {"analytic_dag", "analytic_dag_streaming"}:
        dag = product_dag_from_compiled_cached(compiled, descriptor_count=descriptor_count)
        real_dag_path = _can_use_real_product_dag_path(
            evaluator,
            compiled=compiled,
            dag=dag,
            atomic_base=atomic_base,
            real_if_scalar=real_if_scalar,
        )
        if real_dag_path:
            real_atomic_base = atomic_base.real if torch.is_complex(atomic_base) else atomic_base
            real_dag_values = evaluate_product_dag(real_atomic_base.contiguous(), dag)
            dag_values = ProductDAGValues(
                all_node_values=torch.complex(
                    real_dag_values.all_node_values,
                    torch.zeros_like(real_dag_values.all_node_values),
                ),
                descriptor_values=torch.complex(
                    real_dag_values.descriptor_values,
                    torch.zeros_like(real_dag_values.descriptor_values),
                ),
            )
        else:
            dag_values = evaluate_product_dag(atomic_base, dag)
        raw_values = accelerated_values + dag_values.descriptor_values
    else:
        zero_adjoint = torch.zeros(
            (atom_types.shape[0], descriptor_count),
            dtype=atomic_base.dtype,
            device=atomic_base.device,
        )
        residual_values, _ = _direct_product_root_adjoint(
            atomic_base,
            _compiled_without_accelerated_plans(compiled),
            descriptor_count=descriptor_count,
            output_adjoint=zero_adjoint,
        )
        raw_values = accelerated_values + residual_values
    values = raw_values
    if real_if_scalar and compiled.all_scalar:
        values = checked_real_scalar_projection(values, imag_tol=1.0e-10, context="analytic VJP path")
    if tuple(descriptor_adjoint.shape) != tuple(values.shape):
        raise ValueError(f"descriptor_adjoint must have shape {tuple(values.shape)}; got {tuple(descriptor_adjoint.shape)}")
    output_adjoint = descriptor_adjoint.to(
        device=raw_values.device,
        dtype=raw_values.dtype,
    )
    if product_method in {"analytic_dag", "analytic_dag_streaming"}:
        residual_output_adjoint = _residual_descriptor_adjoint(
            compiled,
            output_adjoint.reshape(1, int(atom_types.shape[0]), descriptor_count),
        )[0]
        if real_dag_path and real_dag_values is not None:
            real_output_adjoint = residual_output_adjoint.real if torch.is_complex(residual_output_adjoint) else residual_output_adjoint
            real_root_adjoint = reverse_product_dag_adjoint(
                real_output_adjoint.contiguous(),
                real_dag_values,
                dag,
            )
            root_adjoint = real_root_adjoint
        else:
            root_adjoint = reverse_product_dag_adjoint(residual_output_adjoint, dag_values, dag)
    else:
        _, root_adjoint = _direct_product_root_adjoint(
            atomic_base,
            _compiled_without_accelerated_plans(compiled),
            descriptor_count=descriptor_count,
            output_adjoint=output_adjoint,
        )
    _, accelerated_root = _accelerated_descriptor_plan_root_adjoint(
        evaluator,
        compiled,
        atomic_base,
        output_adjoint.reshape(1, int(atom_types.shape[0]), descriptor_count),
    )
    root_adjoint = root_adjoint + accelerated_root[0]
    if product_method in {"analytic_streaming", "analytic_dag_streaming"}:
        position_grad = evaluator.site_basis.position_vjp_from_raw_channel_adjoint_streaming(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            raw_atomic_base=raw_atomic_base.real if real_dag_path and not torch.is_complex(root_adjoint) else raw_atomic_base,
            final_channel_adjoint=root_adjoint,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
    else:
        assert vjp_record is not None
        position_grad = evaluator.site_basis.position_vjp_from_record(vjp_record, root_adjoint)
    return values, position_grad


def descriptor_weighted_sum_position_vjp(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    descriptor_weight,
    shifts = None,
    charges = None,
    aux_tensor_basis=None,
    real_if_scalar = True,
    method = "analytic_dag",
):
    """Return per-atom descriptor values and the compact VJP for a linear form."""

    if len(descriptors) == 0:
        values = torch.zeros((atom_types.shape[0], 0), dtype=positions.dtype, device=positions.device)
        grad = torch.zeros((atom_types.shape[0], 3), dtype=positions.dtype, device=positions.device)
        return values, grad

    x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
    compiled = evaluator._compile_descriptors(descriptors)
    product_method = _normalize_analytic_product_method(method)
    if product_method not in {
        "analytic_dag",
        "analytic_dag_streaming",
        "analytic",
        "analytic_streaming",
        "analytic_factorized",
        "analytic_factorized_streaming",
    }:
        raise ValueError(
            "method must be 'analytic', 'analytic_dag', 'analytic_streaming', "
            "'analytic_dag_streaming', 'analytic_factorized', or 'analytic_factorized_streaming'"
        )
    if product_method in {"analytic_streaming", "analytic_dag_streaming", "analytic_factorized_streaming"}:
        _, raw_atomic_base, atomic_base = evaluator.site_basis.compute_channels_raw_and_final(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        vjp_record = None
    else:
        _, atomic_base, vjp_record = evaluator.site_basis.compute_channels_with_vjp_record(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        raw_atomic_base = vjp_record.raw_atomic_base

    descriptor_count = int(len(descriptors))
    weights = torch.as_tensor(descriptor_weight, dtype=atomic_base.dtype, device=atomic_base.device)
    if weights.ndim == 1:
        weights = weights.reshape(1, descriptor_count).expand(int(atom_types.shape[0]), descriptor_count)
    if tuple(weights.shape) != (int(atom_types.shape[0]), descriptor_count):
        raise ValueError(f"descriptor_weight must have shape {(descriptor_count,)} or {(int(atom_types.shape[0]), descriptor_count)}")

    batched_weight = weights.reshape(1, int(atom_types.shape[0]), descriptor_count)
    raw_values, root_adjoint = _accelerated_descriptor_plan_root_adjoint(
        evaluator,
        compiled,
        atomic_base,
        batched_weight,
    )
    residual_adjoint = _residual_descriptor_adjoint(compiled, batched_weight)

    dag = product_dag_from_compiled_cached(compiled, descriptor_count=descriptor_count)
    real_dag_values = None
    real_dag_path = False
    if dag.output_terms:
        if product_method in {"analytic_dag", "analytic_dag_streaming", "analytic_factorized", "analytic_factorized_streaming"}:
            real_dag_path = _can_use_real_product_dag_path(
                evaluator,
                compiled=compiled,
                dag=dag,
                atomic_base=atomic_base,
                real_if_scalar=real_if_scalar,
            )
        if real_dag_path:
            real_atomic_base = atomic_base.real if torch.is_complex(atomic_base) else atomic_base
            real_dag_values = evaluate_product_dag(real_atomic_base.contiguous(), dag)
            dag_values = ProductDAGValues(
                all_node_values=torch.complex(
                    real_dag_values.all_node_values,
                    torch.zeros_like(real_dag_values.all_node_values),
                ),
                descriptor_values=torch.complex(
                    real_dag_values.descriptor_values,
                    torch.zeros_like(real_dag_values.descriptor_values),
                ),
            )
        else:
            dag_values = evaluate_product_dag(atomic_base, dag)
        raw_values = raw_values + dag_values.descriptor_values
        residual_flat = residual_adjoint.reshape(-1, descriptor_count)
        if real_dag_path and real_dag_values is not None:
            residual_root = reverse_product_dag_adjoint(
                residual_flat.real if torch.is_complex(residual_flat) else residual_flat,
                real_dag_values,
                dag,
            )
        elif product_method in {"analytic_dag", "analytic_dag_streaming", "analytic_factorized", "analytic_factorized_streaming"}:
            residual_root = reverse_product_dag_adjoint(residual_flat, dag_values, dag)
        else:
            _, residual_root = _direct_product_root_adjoint(
                atomic_base,
                _compiled_without_accelerated_plans(compiled),
                descriptor_count=descriptor_count,
                output_adjoint=residual_flat,
            )
        root_adjoint = root_adjoint + residual_root.reshape(1, int(atom_types.shape[0]), int(atomic_base.shape[1]))

    values = raw_values
    if real_if_scalar and compiled.all_scalar:
        values = checked_real_scalar_projection(values, imag_tol=1.0e-10, context="analytic weighted VJP path")

    root = root_adjoint[0]
    if product_method in {"analytic_streaming", "analytic_dag_streaming", "analytic_factorized_streaming"}:
        position_grad = evaluator.site_basis.position_vjp_from_raw_channel_adjoint_streaming(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            raw_atomic_base=raw_atomic_base.real if real_dag_path and not torch.is_complex(root) else raw_atomic_base,
            final_channel_adjoint=root,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
    else:
        assert vjp_record is not None
        position_grad = evaluator.site_basis.position_vjp_from_record(vjp_record, root)
    return values, position_grad


def descriptor_linear_form_position_vjp(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    descriptor_weight,
    shifts = None,
    charges = None,
    aux_tensor_basis=None,
    real_if_scalar = True,
    method = "analytic_factorized",
):
    """Return the coefficient-weighted site energy and compact position VJP."""

    if len(descriptors) == 0:
        site_linear = torch.zeros((atom_types.shape[0],), dtype=positions.dtype, device=positions.device)
        grad = torch.zeros((atom_types.shape[0], 3), dtype=positions.dtype, device=positions.device)
        return site_linear, grad

    profile = {} if _linear_form_profile_enabled() else None
    total_start = _profile_start(positions) if profile is not None else None
    start = _profile_start(positions) if profile is not None else None
    x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
    _profile_stop(profile, "edge_vectors_seconds", start, x_ij)
    start = _profile_start(x_ij) if profile is not None else None
    compiled = evaluator._compile_descriptors(descriptors)
    _profile_stop(profile, "compile_lookup_seconds", start, x_ij)
    product_method = _normalize_analytic_product_method(method)
    if product_method not in {
        "analytic_dag",
        "analytic_dag_streaming",
        "analytic",
        "analytic_streaming",
        "analytic_factorized",
        "analytic_factorized_streaming",
    }:
        raise ValueError(
            "method must be 'analytic', 'analytic_dag', 'analytic_streaming', "
            "'analytic_dag_streaming', 'analytic_factorized', or 'analytic_factorized_streaming'"
        )
    start = _profile_start(x_ij) if profile is not None else None
    if product_method in {"analytic_streaming", "analytic_dag_streaming", "analytic_factorized_streaming"}:
        _, raw_atomic_base, atomic_base = evaluator.site_basis.compute_channels_raw_and_final(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        vjp_record = None
    else:
        _, atomic_base, vjp_record = evaluator.site_basis.compute_channels_with_vjp_record(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        raw_atomic_base = vjp_record.raw_atomic_base
    _profile_stop(profile, "site_basis_forward_seconds", start, atomic_base)

    descriptor_count = int(len(descriptors))
    n_atoms = int(atom_types.shape[0])
    real_linear_backend = _can_use_real_linear_product_path(
        evaluator,
        compiled,
        atomic_base,
        real_if_scalar,
    )
    weight_dtype = atomic_base.real.dtype if real_linear_backend and torch.is_complex(atomic_base) else atomic_base.dtype
    weights = torch.as_tensor(descriptor_weight, dtype=weight_dtype, device=atomic_base.device)
    if weights.ndim == 1:
        weights = weights.reshape(1, descriptor_count).expand(n_atoms, descriptor_count)
    if tuple(weights.shape) != (n_atoms, descriptor_count):
        raise ValueError(f"descriptor_weight must have shape {(descriptor_count,)} or {(n_atoms, descriptor_count)}")

    start = _profile_start(atomic_base) if profile is not None else None
    site_linear, root_adjoint = _accelerated_descriptor_plan_linear_form_root_adjoint(
        evaluator,
        compiled,
        atomic_base,
        weights,
    )
    _profile_stop(profile, "accelerated_product_vjp_seconds", start, root_adjoint)
    residual_weights = _residual_descriptor_weights(compiled, weights)

    dag = product_dag_from_compiled_cached(compiled, descriptor_count=descriptor_count)
    real_dag_path = False
    if dag.output_terms:
        start = _profile_start(atomic_base) if profile is not None else None
        if product_method in {"analytic_dag", "analytic_dag_streaming", "analytic_factorized", "analytic_factorized_streaming"}:
            real_dag_path = _can_use_real_product_dag_path(
                evaluator,
                compiled=compiled,
                dag=dag,
                atomic_base=atomic_base,
                real_if_scalar=real_if_scalar,
            )
        if real_dag_path:
            real_atomic_base = atomic_base.real if torch.is_complex(atomic_base) else atomic_base
            residual_linear, residual_root = evaluate_product_dag_linear_form(
                real_atomic_base.contiguous(),
                dag,
                residual_weights.real if torch.is_complex(residual_weights) else residual_weights,
            )
        elif product_method in {"analytic_dag", "analytic_dag_streaming", "analytic_factorized", "analytic_factorized_streaming"}:
            residual_linear, residual_root = evaluate_product_dag_linear_form(
                atomic_base,
                dag,
                residual_weights,
            )
        else:
            dag_values = evaluate_product_dag(atomic_base, dag)
            residual_flat = residual_weights.reshape(-1, descriptor_count)
            residual_root = reverse_product_dag_adjoint(residual_flat, dag_values, dag)
            residual_linear = (dag_values.descriptor_values * residual_weights).sum(dim=1)
        site_linear, root_adjoint = _add_linear_form_accumulator(
            site_linear,
            root_adjoint,
            residual_linear,
            residual_root.reshape(n_atoms, int(atomic_base.shape[1])),
        )
        _profile_stop(profile, "residual_product_vjp_seconds", start, root_adjoint)

    if real_if_scalar and compiled.all_scalar:
        site_linear = checked_real_scalar_projection(site_linear, imag_tol=1.0e-10, context="analytic linear-form VJP path")

    start = _profile_start(root_adjoint) if profile is not None else None
    if product_method in {"analytic_streaming", "analytic_dag_streaming", "analytic_factorized_streaming"}:
        position_grad = evaluator.site_basis.position_vjp_from_raw_channel_adjoint_streaming(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            raw_atomic_base=raw_atomic_base.real if not torch.is_complex(root_adjoint) and torch.is_complex(raw_atomic_base) else raw_atomic_base,
            final_channel_adjoint=root_adjoint,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
    else:
        assert vjp_record is not None
        position_grad = evaluator.site_basis.position_vjp_from_record(vjp_record, root_adjoint)
    _profile_stop(profile, "site_basis_vjp_seconds", start, position_grad)
    if profile is not None:
        _profile_stop(profile, "total_seconds", total_start, position_grad)
        profile["method"] = str(product_method)
        profile["descriptor_count"] = int(descriptor_count)
        profile["channel_count"] = int(len(compiled.channels))
        profile["factorized_descriptor_count"] = (
            0 if compiled.factorized_plan is None else int(compiled.factorized_plan.active_descriptor_count)
        )
        profile["residual_descriptor_count"] = (
            int(descriptor_count) if compiled.factorized_plan is None else int(compiled.factorized_plan.residual_descriptor_count)
        )
        profile["site_basis_scatter_backend"] = str(evaluator.site_basis.last_scatter_backend())
        profile["site_basis_vjp_backend"] = str(evaluator.site_basis.last_vjp_scatter_backend())
        profile["backend_counts"] = dict(getattr(evaluator, "_last_backend_counts", {}))
        evaluator._last_linear_form_profile = dict(profile)
    return site_linear, position_grad


def descriptor_charge_vjp(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    descriptor_adjoint,
    shifts = None,
    charges = None,
    aux_tensor_basis=None,
    real_if_scalar = True,
    method = "analytic_dag",
    return_target_report = False,
):
    """Return descriptor values and compact VJP with respect to per-atom charges.

    For complex covariant descriptor blocks, the VJP convention matches
    `descriptor_position_vjp`: gradient of
    ``real(sum(descriptor_adjoint * descriptor_values))``.
    """

    if len(descriptors) == 0:
        values = torch.zeros((atom_types.shape[0], 0), dtype=positions.dtype, device=positions.device)
        grad = torch.zeros((atom_types.shape[0],), dtype=positions.dtype, device=positions.device)
        if return_target_report:
            from .property_targets import charge_target_spec, target_provenance_report

            report = target_provenance_report(
                charge_target_spec(metadata={"descriptor_count": 0}),
                descriptor_plan={"descriptor_count": 0, "uses_local_coupling_enumeration": False},
                derivative_chain=("descriptor_adjoint", "charge_basis_vjp"),
                row_source="descriptor_charge_vjp",
            )
            return values, grad, report
        return values, grad

    x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
    compiled = evaluator._compile_descriptors(descriptors)
    _, atomic_base, vjp_record = evaluator.site_basis.compute_channels_with_vjp_record(
        x_ij=x_ij,
        edge_index=edge_index,
        atom_types=atom_types,
        channels=compiled.channels,
        charges=charges,
        aux_tensor_basis=aux_tensor_basis,
    )
    product_method = _normalize_analytic_product_method(method)
    if product_method not in {"analytic", "analytic_dag"}:
        raise ValueError("method must be 'analytic' or 'analytic_dag'")
    descriptor_count = len(descriptors)
    zero_accelerated_adjoint = torch.zeros(
        (1, int(atom_types.shape[0]), descriptor_count),
        dtype=atomic_base.dtype,
        device=atomic_base.device,
    )
    accelerated_values, _ = _accelerated_descriptor_plan_root_adjoint(
        evaluator,
        compiled,
        atomic_base,
        zero_accelerated_adjoint,
    )
    if product_method == "analytic_dag":
        dag = product_dag_from_compiled_cached(compiled, descriptor_count=descriptor_count)
        dag_values = evaluate_product_dag(atomic_base, dag)
        raw_values = accelerated_values + dag_values.descriptor_values
    else:
        zero_adjoint = torch.zeros(
            (atom_types.shape[0], descriptor_count),
            dtype=atomic_base.dtype,
            device=atomic_base.device,
        )
        residual_values, _ = _direct_product_root_adjoint(
            atomic_base,
            _compiled_without_accelerated_plans(compiled),
            descriptor_count=descriptor_count,
            output_adjoint=zero_adjoint,
        )
        raw_values = accelerated_values + residual_values
    values = raw_values
    if real_if_scalar and compiled.all_scalar:
        values = checked_real_scalar_projection(values, imag_tol=1.0e-10, context="analytic charge VJP path")
    if tuple(descriptor_adjoint.shape) != tuple(values.shape):
        raise ValueError(f"descriptor_adjoint must have shape {tuple(values.shape)}; got {tuple(descriptor_adjoint.shape)}")
    output_adjoint = descriptor_adjoint.to(
        device=raw_values.device,
        dtype=raw_values.dtype,
    )
    if product_method == "analytic_dag":
        residual_output_adjoint = _residual_descriptor_adjoint(
            compiled,
            output_adjoint.reshape(1, int(atom_types.shape[0]), descriptor_count),
        )[0]
        root_adjoint = reverse_product_dag_adjoint(residual_output_adjoint, dag_values, dag)
    else:
        _, root_adjoint = _direct_product_root_adjoint(
            atomic_base,
            _compiled_without_accelerated_plans(compiled),
            descriptor_count=descriptor_count,
            output_adjoint=output_adjoint,
        )
    _, accelerated_root = _accelerated_descriptor_plan_root_adjoint(
        evaluator,
        compiled,
        atomic_base,
        output_adjoint.reshape(1, int(atom_types.shape[0]), descriptor_count),
    )
    root_adjoint = root_adjoint + accelerated_root[0]
    charge_grad = evaluator.site_basis.charge_vjp_from_record(vjp_record, root_adjoint)
    if return_target_report:
        from .property_targets import charge_target_spec, target_provenance_report

        report = target_provenance_report(
            charge_target_spec(
                metadata={
                    "descriptor_count": int(descriptor_count),
                    "product_method": str(product_method),
                    "charge_mode": str(getattr(evaluator.site_basis.cfg, "charge_mode", "unknown")),
                }
            ),
            descriptor_plan={
                "descriptor_count": int(descriptor_count),
                "compiled_channel_count": int(len(compiled.channels)),
                "uses_local_coupling_enumeration": False,
            },
            derivative_chain=("descriptor_adjoint", "product_adjoint", "site_basis_charge_vjp"),
            row_source="descriptor_charge_vjp",
        )
        return values, charge_grad, report
    return values, charge_grad


def descriptor_gradients_wrt_positions(
    evaluator,
    positions,
    cell,
    edge_index,
    atom_types,
    descriptors,
    shifts = None,
    charges = None,
    aux_tensor_basis=None,
    real_if_scalar = True,
    method = "autograd_loop",
):
    method = _normalize_analytic_product_method(method)
    if method in {"autograd_vectorized", "vectorized"}:
        return _descriptor_gradients_autograd_vectorized(
            evaluator,
            positions,
            cell,
            edge_index,
            atom_types,
            descriptors,
            shifts,
            charges,
            aux_tensor_basis,
            real_if_scalar,
        )
    if method == "analytic_dag":
        return _descriptor_gradients_analytic_dag(
            evaluator,
            positions,
            cell,
            edge_index,
            atom_types,
            descriptors,
            shifts,
            charges,
            aux_tensor_basis,
            real_if_scalar,
        )
    if method == "analytic":
        return _descriptor_gradients_analytic_direct(
            evaluator,
            positions,
            cell,
            edge_index,
            atom_types,
            descriptors,
            shifts,
            charges,
            aux_tensor_basis,
            real_if_scalar,
        )
    if method not in {"autograd_loop", "loop"}:
        raise ValueError("method must be one of autograd_loop, autograd_vectorized, analytic, or analytic_dag")
    positions = positions.clone().requires_grad_(True)
    values = _evaluate_descriptors_from_positions(
        evaluator,
        positions,
        cell,
        edge_index,
        atom_types,
        descriptors,
        shifts,
        charges,
        aux_tensor_basis,
        real_if_scalar,
    )
    grads = _descriptor_gradients_autograd_loop(values, positions)
    return values, grads


def format_lammps_compute_pace_like(per_atom_descriptors, descriptor_gradients):
    """Pack descriptor values and gradients in a compute-pace-like array structure.

    The arrangement follows the documented row ordering of LAMMPS ``compute pace``
    with ``bikflag=1`` and ``dgradflag=1``: per-atom descriptor rows first,
    pair-gradient rows next, and one final summary row. The left-most three columns
    are reserved for ``(i, j, a)`` on the gradient rows.
    """
    if per_atom_descriptors.ndim != 2:
        raise ValueError('per_atom_descriptors must have shape [N, K]')
    if descriptor_gradients.ndim != 4:
        raise ValueError('descriptor_gradients must have shape [N, K, N, 3]')
    N, K = per_atom_descriptors.shape
    grad_rows = []
    for j in range(N):
        for i in range(N):
            for a in range(3):
                row = torch.zeros((K + 3,), dtype=per_atom_descriptors.dtype, device=per_atom_descriptors.device)
                row[0] = i
                row[1] = j
                row[2] = a
                row[3:] = descriptor_gradients[i, :, j, a]
                grad_rows.append(row)
    pair_grad_rows = torch.stack(grad_rows, dim=0)
    full_array = torch.zeros((N + 3 * N * N + 1, K + 3), dtype=per_atom_descriptors.dtype, device=per_atom_descriptors.device)
    full_array[:N, 3:] = per_atom_descriptors
    full_array[N:N + 3 * N * N] = pair_grad_rows
    return LAMMPSPaceLikeOutput(per_atom_descriptors=per_atom_descriptors, pair_grad_rows=pair_grad_rows, full_array=full_array)
