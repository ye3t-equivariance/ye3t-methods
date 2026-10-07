
from dataclasses import field
from functools import lru_cache
import hashlib
import json
import math
import numbers
import os
import time

import itertools
import torch
from ye3t_methods.atomistic._record import recordclass
from ye3t_methods.atomistic.equivariant_calc.atomic_base_cache import AtomicBaseCache, DescriptorRuntimeCache, EvaluationContext, NormalizationMap

try:
    from ye3t_methods.atomistic._runtime import configure_runtime_environment
except Exception:  # pragma: no cover - local-source fallback
    from _runtime import configure_runtime_environment

configure_runtime_environment()

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None

from .labeling import SingleChannelLabel
from .angular_basis import angular_basis_for_backend
from .radial_basis import (
    RadialBasis,
    _PACE_SPLINE_MAX_COEFFICIENT_BYTES,
    _pace_uniform_cubic_spline_coefficients,
    _pace_uniform_cubic_spline_evaluate_with_derivative,
    radial_basis_for_kind,
)
from .trc_sph_harm import (
    associated_legendre_l,
    real_spherical_harmonics_l_from_unit_cartesian,
)
from .trc_cheby import chebyshev_poly_first


_LOG_4PI = math.log(4.0 * math.pi)


def site_real_block_to_ye3t_tesseral(value, l_value):
    """Convert SiteBasisV2 signed-m real rows to YE3T tesseral order."""

    if int(l_value) == 0:
        return value
    return torch.flip(value, dims=(-1,))


@lru_cache(maxsize=None)
def _real_spherical_normalization_constants(l):
    vals = []
    for m in range(0, l + 1):
        log_norm = 0.5 * (
            math.log(2 * l + 1)
            - _LOG_4PI
            + math.lgamma(l - m + 1)
            - math.lgamma(l + m + 1)
        )
        vals.append(math.exp(log_norm))
    return tuple(vals)


def _real_spherical_harmonics_l_from_unit_cartesian(l, unit_xyz):
    """Return real tesseral harmonics from unit Cartesian directions."""

    return real_spherical_harmonics_l_from_unit_cartesian(l, unit_xyz)


def _ace_env(name, default=None):
    return os.environ.get("YE3T_ACE_" + str(name), os.environ.get("gne3_ace_" + str(name), default))


def _site_basis_profile_enabled():
    raw = _ace_env("PROFILE_FULL_MODEL", "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _profile_start_from_tensor(tensor):
    if torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return time.perf_counter()


def _profile_stop_from_tensor(profile, key, start, tensor):
    if profile is None:
        return
    if torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    profile[key] = float(profile.get(key, 0.0)) + float(time.perf_counter() - start)


def _tensor_runtime_cache_key(tensor):
    if not torch.is_tensor(tensor):
        return None
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(x) for x in tensor.shape),
        tuple(int(x) for x in tensor.stride()),
        int(tensor.data_ptr()),
        bool(tensor.requires_grad),
    )


def _runtime_cache_bucket(runtime_cache, name):
    if not isinstance(runtime_cache, dict):
        return None
    return runtime_cache.setdefault(str(name), {})


DEFAULT_ATOMIC_BASE_NORMALIZATION = "soft_neighbor"


def _metadata_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if triton is not None:  # pragma: no cover - exercised only when Triton is available
    @triton.jit
    def _scatter_sum_2d_kernel(
        src_ptr,
        index_ptr,
        out_ptr,
        num_rows,
        num_cols,
        stride_src_row,
        stride_src_col,
        stride_out_row,
        stride_out_col,
    ):
        pid_row = tl.program_id(axis=0)
        pid_col = tl.program_id(axis=1)
        row_offsets = pid_row * 64 + tl.arange(0, 64)
        col_offsets = pid_col * 64 + tl.arange(0, 64)
        row_mask = row_offsets < num_rows
        col_mask = col_offsets < num_cols
        dst_rows = tl.load(index_ptr + row_offsets, mask=row_mask, other=0)
        src_ptrs = src_ptr + row_offsets[:, None] * stride_src_row + col_offsets[None, :] * stride_src_col
        values = tl.load(src_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        out_ptrs = out_ptr + dst_rows[:, None] * stride_out_row + col_offsets[None, :] * stride_out_col
        tl.atomic_add(out_ptrs, values, mask=row_mask[:, None] & col_mask[None, :])

    @triton.jit
    def _real_plain_group_vjp_scatter_kernel(
        edge_adjoint_ptr,
        y_ptr,
        dy_ptr,
        rf_ptr,
        drf_dr_ptr,
        dr_dx_ptr,
        prefactor_ptr,
        channel_indices_ptr,
        m_positions_ptr,
        centers_ptr,
        neighs_ptr,
        out_ptr,
        n_edges,
        n_channels,
        n_atoms,
        group_size,
        y_stride_m,
        y_stride_e,
        dy_stride_m,
        dy_stride_e,
        dy_stride_c,
        BLOCK_EDGES,
        BLOCK_M,
    ):
        edges = tl.program_id(0) * 128 + tl.arange(0, 128)
        m_offsets = tl.arange(0, 64)
        edge_mask = edges < n_edges
        m_mask = m_offsets < group_size

        channel_indices = tl.load(channel_indices_ptr + m_offsets, mask=m_mask, other=0)
        m_positions = tl.load(m_positions_ptr + m_offsets, mask=m_mask, other=0)
        adj = tl.load(
            edge_adjoint_ptr + edges[:, None] * n_channels + channel_indices[None, :],
            mask=edge_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        y = tl.load(
            y_ptr + m_positions[None, :] * y_stride_m + edges[:, None] * y_stride_e,
            mask=edge_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        radial_adjoint = tl.sum(adj * y, axis=1)
        rf = tl.load(rf_ptr + edges, mask=edge_mask, other=0.0)
        drf_dr = tl.load(drf_dr_ptr + edges, mask=edge_mask, other=0.0)
        prefactor = tl.load(prefactor_ptr + edges, mask=edge_mask, other=0.0)
        centers = tl.load(centers_ptr + edges, mask=edge_mask, other=0)
        neighs = tl.load(neighs_ptr + edges, mask=edge_mask, other=0)

        for cart in tl.static_range(0, 3):
            dy = tl.load(
                dy_ptr
                + m_positions[None, :] * dy_stride_m
                + edges[:, None] * dy_stride_e
                + cart * dy_stride_c,
                mask=edge_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            angular_adjoint = tl.sum(adj * dy, axis=1)
            dr_dx = tl.load(dr_dx_ptr + edges * 3 + cart, mask=edge_mask, other=0.0)
            grad = prefactor * (radial_adjoint * drf_dr * dr_dx + rf * angular_adjoint)
            tl.atomic_add(out_ptr + neighs * 3 + cart, grad, mask=edge_mask)
            tl.atomic_add(out_ptr + centers * 3 + cart, -grad, mask=edge_mask)

    @triton.jit
    def _real_plain_group_soft_neighbor_vjp_scatter_kernel(
        edge_adjoint_ptr,
        y_ptr,
        dy_ptr,
        rf_ptr,
        drf_dr_ptr,
        dr_dx_ptr,
        prefactor_ptr,
        soft_weight_ptr,
        soft_weight_dx_ptr,
        channel_indices_ptr,
        m_positions_ptr,
        centers_ptr,
        neighs_ptr,
        out_ptr,
        n_edges,
        n_channels,
        n_atoms,
        group_size,
        y_stride_m,
        y_stride_e,
        dy_stride_m,
        dy_stride_e,
        dy_stride_c,
        BLOCK_EDGES,
        BLOCK_M,
    ):
        edges = tl.program_id(0) * 128 + tl.arange(0, 128)
        m_offsets = tl.arange(0, 64)
        edge_mask = edges < n_edges
        m_mask = m_offsets < group_size

        channel_indices = tl.load(channel_indices_ptr + m_offsets, mask=m_mask, other=0)
        m_positions = tl.load(m_positions_ptr + m_offsets, mask=m_mask, other=0)
        adj = tl.load(
            edge_adjoint_ptr + edges[:, None] * n_channels + channel_indices[None, :],
            mask=edge_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        y = tl.load(
            y_ptr + m_positions[None, :] * y_stride_m + edges[:, None] * y_stride_e,
            mask=edge_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        radial_adjoint = tl.sum(adj * y, axis=1)
        rf = tl.load(rf_ptr + edges, mask=edge_mask, other=0.0)
        drf_dr = tl.load(drf_dr_ptr + edges, mask=edge_mask, other=0.0)
        prefactor = tl.load(prefactor_ptr + edges, mask=edge_mask, other=0.0)
        soft_weight = tl.load(soft_weight_ptr + edges, mask=edge_mask, other=0.0)
        centers = tl.load(centers_ptr + edges, mask=edge_mask, other=0)
        neighs = tl.load(neighs_ptr + edges, mask=edge_mask, other=0)
        value_adjoint = prefactor * rf * radial_adjoint

        for cart in tl.static_range(0, 3):
            dy = tl.load(
                dy_ptr
                + m_positions[None, :] * dy_stride_m
                + edges[:, None] * dy_stride_e
                + cart * dy_stride_c,
                mask=edge_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            angular_adjoint = tl.sum(adj * dy, axis=1)
            dr_dx = tl.load(dr_dx_ptr + edges * 3 + cart, mask=edge_mask, other=0.0)
            soft_weight_dx = tl.load(soft_weight_dx_ptr + edges * 3 + cart, mask=edge_mask, other=0.0)
            numerator_grad = prefactor * (radial_adjoint * drf_dr * dr_dx + rf * angular_adjoint)
            grad = soft_weight * numerator_grad + value_adjoint * soft_weight_dx
            tl.atomic_add(out_ptr + neighs * 3 + cart, grad, mask=edge_mask)
            tl.atomic_add(out_ptr + centers * 3 + cart, -grad, mask=edge_mask)

    @triton.jit
    def _plain_real_site_basis_scatter_kernel(
        prefactor_ptr,
        y_ptr,
        channel_indices_ptr,
        m_positions_ptr,
        centers_ptr,
        out_ptr,
        n_edges,
        group_size,
        y_stride_m,
        y_stride_e,
        out_stride_atom,
        out_stride_channel,
        BLOCK_EDGES,
        BLOCK_M,
    ):
        edges = tl.program_id(0) * 64 + tl.arange(0, 64)
        m_offsets = tl.arange(0, 64)
        edge_mask = edges < n_edges
        m_mask = m_offsets < group_size
        channels = tl.load(channel_indices_ptr + m_offsets, mask=m_mask, other=0)
        m_pos = tl.load(m_positions_ptr + m_offsets, mask=m_mask, other=0)
        centers = tl.load(centers_ptr + edges, mask=edge_mask, other=0)
        prefactor = tl.load(prefactor_ptr + edges, mask=edge_mask, other=0.0)
        y = tl.load(
            y_ptr + m_pos[None, :] * y_stride_m + edges[:, None] * y_stride_e,
            mask=edge_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        values = prefactor[:, None] * y
        out_ptrs = out_ptr + centers[:, None] * out_stride_atom + channels[None, :] * out_stride_channel
        tl.atomic_add(out_ptrs, values, mask=edge_mask[:, None] & m_mask[None, :])

    @triton.jit
    def _plain_real_site_basis_scatter_backward_kernel(
        grad_out_ptr,
        prefactor_ptr,
        y_ptr,
        channel_indices_ptr,
        m_positions_ptr,
        centers_ptr,
        grad_prefactor_ptr,
        grad_y_ptr,
        n_edges,
        group_size,
        y_stride_m,
        y_stride_e,
        grad_out_stride_atom,
        grad_out_stride_channel,
        grad_y_stride_m,
        grad_y_stride_e,
        BLOCK_EDGES,
        BLOCK_M,
    ):
        edges = tl.program_id(0) * 64 + tl.arange(0, 64)
        m_offsets = tl.arange(0, 64)
        edge_mask = edges < n_edges
        m_mask = m_offsets < group_size
        channels = tl.load(channel_indices_ptr + m_offsets, mask=m_mask, other=0)
        m_pos = tl.load(m_positions_ptr + m_offsets, mask=m_mask, other=0)
        centers = tl.load(centers_ptr + edges, mask=edge_mask, other=0)
        grad = tl.load(
            grad_out_ptr + centers[:, None] * grad_out_stride_atom + channels[None, :] * grad_out_stride_channel,
            mask=edge_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        y = tl.load(
            y_ptr + m_pos[None, :] * y_stride_m + edges[:, None] * y_stride_e,
            mask=edge_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        prefactor = tl.load(prefactor_ptr + edges, mask=edge_mask, other=0.0)
        grad_prefactor = tl.sum(grad * y, axis=1)
        tl.store(grad_prefactor_ptr + edges, grad_prefactor, mask=edge_mask)
        grad_y = grad * prefactor[:, None]
        tl.atomic_add(
            grad_y_ptr + m_pos[None, :] * grad_y_stride_m + edges[:, None] * grad_y_stride_e,
            grad_y,
            mask=edge_mask[:, None] & m_mask[None, :],
        )

    @triton.jit
    def _packed_plain_real_site_basis_scatter_kernel(
        prefactors_ptr,
        term_y_ptr,
        term_group_ptr,
        term_channel_ptr,
        centers_ptr,
        out_ptr,
        n_edges,
        n_terms,
        pref_stride_group,
        pref_stride_edge,
        y_stride_term,
        y_stride_edge,
        out_stride_atom,
        out_stride_channel,
        BLOCK_EDGES,
        BLOCK_TERMS,
    ):
        edges = tl.program_id(0) * 64 + tl.arange(0, 64)
        terms = tl.program_id(1) * 16 + tl.arange(0, 16)
        edge_mask = edges < n_edges
        term_mask = terms < n_terms
        groups = tl.load(term_group_ptr + terms, mask=term_mask, other=0)
        channels = tl.load(term_channel_ptr + terms, mask=term_mask, other=0)
        centers = tl.load(centers_ptr + edges, mask=edge_mask, other=0)
        prefactor = tl.load(
            prefactors_ptr + groups[None, :] * pref_stride_group + edges[:, None] * pref_stride_edge,
            mask=edge_mask[:, None] & term_mask[None, :],
            other=0.0,
        )
        y = tl.load(
            term_y_ptr + terms[None, :] * y_stride_term + edges[:, None] * y_stride_edge,
            mask=edge_mask[:, None] & term_mask[None, :],
            other=0.0,
        )
        out_ptrs = out_ptr + centers[:, None] * out_stride_atom + channels[None, :] * out_stride_channel
        tl.atomic_add(out_ptrs, prefactor * y, mask=edge_mask[:, None] & term_mask[None, :])

    @triton.jit
    def _packed_plain_real_site_basis_scatter_backward_kernel(
        grad_out_ptr,
        prefactors_ptr,
        term_y_ptr,
        term_group_ptr,
        term_channel_ptr,
        centers_ptr,
        grad_prefactors_ptr,
        grad_term_y_ptr,
        n_edges,
        n_terms,
        pref_stride_group,
        pref_stride_edge,
        y_stride_term,
        y_stride_edge,
        grad_out_stride_atom,
        grad_out_stride_channel,
        grad_pref_stride_group,
        grad_pref_stride_edge,
        grad_y_stride_term,
        grad_y_stride_edge,
        BLOCK_EDGES,
        BLOCK_TERMS,
    ):
        edges = tl.program_id(0) * 64 + tl.arange(0, 64)
        terms = tl.program_id(1) * 16 + tl.arange(0, 16)
        edge_mask = edges < n_edges
        term_mask = terms < n_terms
        groups = tl.load(term_group_ptr + terms, mask=term_mask, other=0)
        channels = tl.load(term_channel_ptr + terms, mask=term_mask, other=0)
        centers = tl.load(centers_ptr + edges, mask=edge_mask, other=0)
        grad = tl.load(
            grad_out_ptr + centers[:, None] * grad_out_stride_atom + channels[None, :] * grad_out_stride_channel,
            mask=edge_mask[:, None] & term_mask[None, :],
            other=0.0,
        )
        prefactor = tl.load(
            prefactors_ptr + groups[None, :] * pref_stride_group + edges[:, None] * pref_stride_edge,
            mask=edge_mask[:, None] & term_mask[None, :],
            other=0.0,
        )
        y = tl.load(
            term_y_ptr + terms[None, :] * y_stride_term + edges[:, None] * y_stride_edge,
            mask=edge_mask[:, None] & term_mask[None, :],
            other=0.0,
        )
        tl.atomic_add(
            grad_prefactors_ptr + groups[None, :] * grad_pref_stride_group + edges[:, None] * grad_pref_stride_edge,
            grad * y,
            mask=edge_mask[:, None] & term_mask[None, :],
        )
        tl.store(
            grad_term_y_ptr + terms[None, :] * grad_y_stride_term + edges[:, None] * grad_y_stride_edge,
            grad * prefactor,
            mask=edge_mask[:, None] & term_mask[None, :],
        )

    @triton.jit
    def _packed_plain_real_site_basis_scatter_double_backward_kernel(
        grad_out_ptr,
        prefactors_ptr,
        term_y_ptr,
        grad_grad_prefactors_ptr,
        grad_grad_term_y_ptr,
        term_group_ptr,
        term_channel_ptr,
        centers_ptr,
        grad2_grad_out_ptr,
        grad2_prefactors_ptr,
        grad2_term_y_ptr,
        n_edges,
        n_terms,
        pref_stride_group,
        pref_stride_edge,
        y_stride_term,
        y_stride_edge,
        grad_out_stride_atom,
        grad_out_stride_channel,
        grad_grad_pref_stride_group,
        grad_grad_pref_stride_edge,
        grad_grad_y_stride_term,
        grad_grad_y_stride_edge,
        grad2_grad_out_stride_atom,
        grad2_grad_out_stride_channel,
        grad2_pref_stride_group,
        grad2_pref_stride_edge,
        grad2_y_stride_term,
        grad2_y_stride_edge,
        BLOCK_EDGES,
        BLOCK_TERMS,
    ):
        edges = tl.program_id(0) * 64 + tl.arange(0, 64)
        terms = tl.program_id(1) * 16 + tl.arange(0, 16)
        edge_mask = edges < n_edges
        term_mask = terms < n_terms
        mask = edge_mask[:, None] & term_mask[None, :]
        groups = tl.load(term_group_ptr + terms, mask=term_mask, other=0)
        channels = tl.load(term_channel_ptr + terms, mask=term_mask, other=0)
        centers = tl.load(centers_ptr + edges, mask=edge_mask, other=0)
        grad = tl.load(
            grad_out_ptr
            + centers[:, None] * grad_out_stride_atom
            + channels[None, :] * grad_out_stride_channel,
            mask=mask,
            other=0.0,
        )
        prefactor = tl.load(
            prefactors_ptr
            + groups[None, :] * pref_stride_group
            + edges[:, None] * pref_stride_edge,
            mask=mask,
            other=0.0,
        )
        y = tl.load(
            term_y_ptr
            + terms[None, :] * y_stride_term
            + edges[:, None] * y_stride_edge,
            mask=mask,
            other=0.0,
        )
        grad_grad_prefactor = tl.load(
            grad_grad_prefactors_ptr
            + groups[None, :] * grad_grad_pref_stride_group
            + edges[:, None] * grad_grad_pref_stride_edge,
            mask=mask,
            other=0.0,
        )
        grad_grad_y = tl.load(
            grad_grad_term_y_ptr
            + terms[None, :] * grad_grad_y_stride_term
            + edges[:, None] * grad_grad_y_stride_edge,
            mask=mask,
            other=0.0,
        )
        tl.atomic_add(
            grad2_grad_out_ptr
            + centers[:, None] * grad2_grad_out_stride_atom
            + channels[None, :] * grad2_grad_out_stride_channel,
            grad_grad_prefactor * y + grad_grad_y * prefactor,
            mask=mask,
        )
        tl.atomic_add(
            grad2_prefactors_ptr
            + groups[None, :] * grad2_pref_stride_group
            + edges[:, None] * grad2_pref_stride_edge,
            grad_grad_y * grad,
            mask=mask,
        )
        tl.store(
            grad2_term_y_ptr
            + terms[None, :] * grad2_y_stride_term
            + edges[:, None] * grad2_y_stride_edge,
            grad_grad_prefactor * grad,
            mask=mask,
        )


def _triton_available_for_tensor(src, index):
    return bool(
        triton is not None
        and _triton_runtime_supported()
        and not torch.is_grad_enabled()
        and src.is_cuda
        and index.is_cuda
        and src.is_contiguous()
        and index.is_contiguous()
        and src.dtype in (torch.float32, torch.float64)
        and index.dtype == torch.long
    )


def _triton_runtime_supported():
    if os.environ.get("GNE3_ACE_DISABLE_TRITON") == "1" or _ace_env("DISABLE_TRITON") == "1":
        return False
    return True


def _scatter_sum_index_add(src, index, dim_size):
    out = torch.zeros((int(dim_size),) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def _scatter_sum_triton_2d(src, index, dim_size):
    src2d = src.contiguous().view(src.shape[0], -1)
    out2d = torch.zeros((int(dim_size), src2d.shape[1]), dtype=src2d.dtype, device=src2d.device)
    if src2d.numel() == 0:
        return out2d.view((int(dim_size),) + tuple(src.shape[1:]))
    block_rows = 64
    block_cols = 64
    grid = (triton.cdiv(src2d.shape[0], block_rows), triton.cdiv(src2d.shape[1], block_cols))
    _scatter_sum_2d_kernel[grid](
        src2d,
        index.contiguous(),
        out2d,
        src2d.shape[0],
        src2d.shape[1],
        src2d.stride(0),
        src2d.stride(1),
        out2d.stride(0),
        out2d.stride(1),
    )
    return out2d.view((int(dim_size),) + tuple(src.shape[1:]))


def _scatter_sum_maybe_triton(src, index, dim_size):
    if src.is_complex():
        real_src = src.real.contiguous()
        imag_src = src.imag.contiguous()
        if _triton_available_for_tensor(real_src, index):
            try:
                real = _scatter_sum_triton_2d(real_src, index, dim_size)
                imag = _scatter_sum_triton_2d(imag_src, index, dim_size)
            except Exception:
                real = _scatter_sum_index_add(real_src, index, dim_size)
                imag = _scatter_sum_index_add(imag_src, index, dim_size)
        else:
            real = _scatter_sum_maybe_triton(real_src, index, dim_size)
            imag = _scatter_sum_maybe_triton(imag_src, index, dim_size)
        return torch.complex(real, imag)
    if _triton_available_for_tensor(src, index):
        try:
            return _scatter_sum_triton_2d(src, index, dim_size)
        except Exception:
            pass
    return _scatter_sum_index_add(src, index, dim_size)


def _edge_pair_scatter_maybe_triton(
    src_to_neigh,
    src_to_center,
    centers,
    neighs,
    dim_size,
):
    """Scatter edge-pair VJP contributions into atom-major gradients."""

    src_to_neigh = src_to_neigh.contiguous()
    src_to_center = src_to_center.contiguous()
    centers = centers.contiguous()
    neighs = neighs.contiguous()
    min_elements = int(_ace_env("TRITON_VJP_SCATTER_MIN_ELEMENTS", "8192"))
    can_triton = bool(
        (src_to_neigh.numel() + src_to_center.numel()) >= min_elements
        and
        _triton_available_for_tensor(src_to_neigh, neighs)
        and _triton_available_for_tensor(src_to_center, centers)
    )
    if can_triton:
        try:
            neigh_part = _scatter_sum_triton_2d(src_to_neigh, neighs, dim_size)
            center_part = _scatter_sum_triton_2d(src_to_center, centers, dim_size)
            return neigh_part + center_part, "triton"
        except Exception:
            pass
    out = torch.zeros((int(dim_size),) + tuple(src_to_neigh.shape[1:]), dtype=src_to_neigh.dtype, device=src_to_neigh.device)
    out.index_add_(0, neighs, src_to_neigh)
    out.index_add_(0, centers, src_to_center)
    return out, "index_add"


def _edge_pair_scatter_batched_maybe_triton(
    src_to_neigh,
    src_to_center,
    centers,
    neighs,
    dim_size,
):
    """Scatter batched edge-pair VJP contributions into atom-major gradients."""

    if src_to_neigh.ndim < 3 or src_to_center.ndim < 3:
        raise ValueError("batched edge-pair scatter expects tensors with shape [batch, edges, ...]")
    if tuple(src_to_neigh.shape) != tuple(src_to_center.shape):
        raise ValueError("src_to_neigh and src_to_center must have the same shape")
    batch = int(src_to_neigh.shape[0])
    edges = int(src_to_neigh.shape[1])
    tail = tuple(src_to_neigh.shape[2:])
    if batch == 0:
        return torch.zeros((0, int(dim_size)) + tail, dtype=src_to_neigh.dtype, device=src_to_neigh.device), "empty"
    centers = centers.to(device=src_to_neigh.device, dtype=torch.long).contiguous()
    neighs = neighs.to(device=src_to_neigh.device, dtype=torch.long).contiguous()
    offsets = (torch.arange(batch, device=src_to_neigh.device, dtype=torch.long) * int(dim_size)).view(batch, 1)
    flat_neighs = (neighs.view(1, edges) + offsets).reshape(-1)
    flat_centers = (centers.view(1, edges) + offsets).reshape(-1)
    flat_neigh_src = src_to_neigh.contiguous().reshape(batch * edges, *tail)
    flat_center_src = src_to_center.contiguous().reshape(batch * edges, *tail)
    flat_out, backend = _edge_pair_scatter_maybe_triton(
        flat_neigh_src,
        flat_center_src,
        flat_centers,
        flat_neighs,
        batch * int(dim_size),
    )
    return flat_out.reshape(batch, int(dim_size), *tail), "batched_" + str(backend)


def _batched_vjp_scatter_min_elements():
    return int(_ace_env("BATCHED_VJP_SCATTER_MIN_ELEMENTS", "65536"))


def _plain_real_site_basis_scatter_triton(
    out,
    *,
    prefactor,
    y_all,
    group_tensors,
    centers,
):
    if triton is None:
        raise RuntimeError("Triton is not available")
    prefactor = prefactor.contiguous()
    y_real = y_all.real.contiguous()
    centers = centers.contiguous()
    group_size = int(group_tensors.indices.numel())
    if group_size <= 0 or prefactor.numel() == 0:
        return out
    block_edges = 64
    block_m = int(min(max(triton.next_power_of_2(group_size), 1), 64))
    grid = (triton.cdiv(prefactor.shape[0], block_edges),)
    _plain_real_site_basis_scatter_kernel[grid](
        prefactor,
        y_real,
        group_tensors.indices.contiguous(),
        group_tensors.m_positions.contiguous(),
        centers,
        out,
        prefactor.shape[0],
        group_size,
        y_real.stride(0),
        y_real.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_EDGES=block_edges,
        BLOCK_M=block_m,
    )
    return out


class _PlainRealSiteBasisScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefactor, y_real, channel_indices, m_positions, centers, n_atoms, n_channels):
        if triton is None:
            raise RuntimeError("Triton is not available")
        prefactor = prefactor.contiguous()
        y_real = y_real.contiguous()
        channel_indices = channel_indices.contiguous()
        m_positions = m_positions.contiguous()
        centers = centers.contiguous()
        out = torch.zeros((int(n_atoms), int(n_channels)), dtype=prefactor.dtype, device=prefactor.device)
        group_tensors = _ChannelIndexTensors(indices=channel_indices, m_positions=m_positions)
        _plain_real_site_basis_scatter_triton(
            out,
            prefactor=prefactor,
            y_all=y_real,
            group_tensors=group_tensors,
            centers=centers,
        )
        ctx.save_for_backward(prefactor, y_real, channel_indices, m_positions, centers)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        prefactor, y_real, channel_indices, m_positions, centers = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_prefactor = torch.empty_like(prefactor)
        grad_y = torch.zeros_like(y_real)
        group_size = int(channel_indices.numel())
        if group_size > 0 and prefactor.numel() > 0:
            block_edges = 64
            block_m = int(min(max(triton.next_power_of_2(group_size), 1), 64))
            grid = (triton.cdiv(prefactor.shape[0], block_edges),)
            _plain_real_site_basis_scatter_backward_kernel[grid](
                grad_out,
                prefactor,
                y_real,
                channel_indices,
                m_positions,
                centers,
                grad_prefactor,
                grad_y,
                prefactor.shape[0],
                group_size,
                y_real.stride(0),
                y_real.stride(1),
                grad_out.stride(0),
                grad_out.stride(1),
                grad_y.stride(0),
                grad_y.stride(1),
                BLOCK_EDGES=block_edges,
                BLOCK_M=block_m,
            )
        else:
            grad_prefactor.zero_()
        return grad_prefactor, grad_y, None, None, None, None, None


def _plain_real_site_basis_scatter_autograd(
    *,
    prefactor,
    y_all,
    group_tensors,
    centers,
    n_atoms,
    n_channels,
):
    return _PlainRealSiteBasisScatterFunction.apply(
        prefactor.contiguous(),
        y_all.real.contiguous(),
        group_tensors.indices.contiguous(),
        group_tensors.m_positions.contiguous(),
        centers.contiguous(),
        int(n_atoms),
        int(n_channels),
    )


class _PackedPlainRealSiteBasisScatterBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_out,
        prefactors,
        term_y,
        term_group,
        term_channel,
        centers,
        n_atoms,
        n_channels,
    ):
        grad_out = grad_out.contiguous()
        prefactors = prefactors.contiguous()
        term_y = term_y.contiguous()
        term_group = term_group.contiguous()
        term_channel = term_channel.contiguous()
        centers = centers.contiguous()
        grad_prefactors = torch.zeros_like(prefactors)
        grad_term_y = torch.empty_like(term_y)
        n_terms = int(term_y.shape[0])
        if n_terms > 0 and prefactors.numel() > 0:
            block_edges = 64
            block_terms = 16
            grid = (
                triton.cdiv(prefactors.shape[1], block_edges),
                triton.cdiv(n_terms, block_terms),
            )
            _packed_plain_real_site_basis_scatter_backward_kernel[grid](
                grad_out,
                prefactors,
                term_y,
                term_group,
                term_channel,
                centers,
                grad_prefactors,
                grad_term_y,
                prefactors.shape[1],
                n_terms,
                prefactors.stride(0),
                prefactors.stride(1),
                term_y.stride(0),
                term_y.stride(1),
                grad_out.stride(0),
                grad_out.stride(1),
                grad_prefactors.stride(0),
                grad_prefactors.stride(1),
                grad_term_y.stride(0),
                grad_term_y.stride(1),
                BLOCK_EDGES=block_edges,
                BLOCK_TERMS=block_terms,
            )
        ctx.save_for_backward(
            grad_out,
            prefactors,
            term_y,
            term_group,
            term_channel,
            centers,
        )
        ctx.n_atoms = int(n_atoms)
        ctx.n_channels = int(n_channels)
        ctx.set_materialize_grads(False)
        return grad_prefactors, grad_term_y

    @staticmethod
    def backward(ctx, grad_grad_prefactors, grad_grad_term_y):
        (
            grad_out,
            prefactors,
            term_y,
            term_group,
            term_channel,
            centers,
        ) = ctx.saved_tensors
        if grad_grad_prefactors is None:
            grad_grad_prefactors = torch.zeros_like(prefactors)
        if grad_grad_term_y is None:
            grad_grad_term_y = torch.zeros_like(term_y)
        grad_grad_prefactors = grad_grad_prefactors.contiguous()
        grad_grad_term_y = grad_grad_term_y.contiguous()
        grad2_grad_out = torch.zeros_like(grad_out)
        grad2_prefactors = torch.zeros_like(prefactors)
        grad2_term_y = torch.empty_like(term_y)
        n_terms = int(term_y.shape[0])
        if n_terms > 0 and prefactors.numel() > 0:
            block_edges = 64
            block_terms = 16
            grid = (
                triton.cdiv(prefactors.shape[1], block_edges),
                triton.cdiv(n_terms, block_terms),
            )
            _packed_plain_real_site_basis_scatter_double_backward_kernel[grid](
                grad_out,
                prefactors,
                term_y,
                grad_grad_prefactors,
                grad_grad_term_y,
                term_group,
                term_channel,
                centers,
                grad2_grad_out,
                grad2_prefactors,
                grad2_term_y,
                prefactors.shape[1],
                n_terms,
                prefactors.stride(0),
                prefactors.stride(1),
                term_y.stride(0),
                term_y.stride(1),
                grad_out.stride(0),
                grad_out.stride(1),
                grad_grad_prefactors.stride(0),
                grad_grad_prefactors.stride(1),
                grad_grad_term_y.stride(0),
                grad_grad_term_y.stride(1),
                grad2_grad_out.stride(0),
                grad2_grad_out.stride(1),
                grad2_prefactors.stride(0),
                grad2_prefactors.stride(1),
                grad2_term_y.stride(0),
                grad2_term_y.stride(1),
                BLOCK_EDGES=block_edges,
                BLOCK_TERMS=block_terms,
            )
        else:
            grad2_term_y.zero_()
        return (
            grad2_grad_out,
            grad2_prefactors,
            grad2_term_y,
            None,
            None,
            None,
            None,
            None,
        )


class _PackedPlainRealSiteBasisScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefactors, term_y, term_group, term_channel, centers, n_atoms, n_channels):
        if triton is None:
            raise RuntimeError("Triton is not available")
        prefactors = prefactors.contiguous()
        term_y = term_y.contiguous()
        term_group = term_group.contiguous()
        term_channel = term_channel.contiguous()
        centers = centers.contiguous()
        out = torch.zeros((int(n_atoms), int(n_channels)), dtype=prefactors.dtype, device=prefactors.device)
        n_terms = int(term_y.shape[0])
        if n_terms > 0 and prefactors.numel() > 0:
            block_edges = 64
            block_terms = 16
            grid = (triton.cdiv(prefactors.shape[1], block_edges), triton.cdiv(n_terms, block_terms))
            _packed_plain_real_site_basis_scatter_kernel[grid](
                prefactors,
                term_y,
                term_group,
                term_channel,
                centers,
                out,
                prefactors.shape[1],
                n_terms,
                prefactors.stride(0),
                prefactors.stride(1),
                term_y.stride(0),
                term_y.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_EDGES=block_edges,
                BLOCK_TERMS=block_terms,
            )
        ctx.save_for_backward(prefactors, term_y, term_group, term_channel, centers)
        ctx.n_atoms = int(n_atoms)
        ctx.n_channels = int(n_channels)
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        prefactors, term_y, term_group, term_channel, centers = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad_prefactors, grad_term_y = (
                _PackedPlainRealSiteBasisScatterBackwardFunction.apply(
                    grad_out,
                    prefactors,
                    term_y,
                    term_group,
                    term_channel,
                    centers,
                    int(ctx.n_atoms),
                    int(ctx.n_channels),
                )
            )
            return grad_prefactors, grad_term_y, None, None, None, None, None
        grad_out = grad_out.contiguous()
        grad_prefactors = torch.zeros_like(prefactors)
        grad_term_y = torch.empty_like(term_y)
        n_terms = int(term_y.shape[0])
        if n_terms > 0 and prefactors.numel() > 0:
            block_edges = 64
            block_terms = 16
            grid = (triton.cdiv(prefactors.shape[1], block_edges), triton.cdiv(n_terms, block_terms))
            _packed_plain_real_site_basis_scatter_backward_kernel[grid](
                grad_out,
                prefactors,
                term_y,
                term_group,
                term_channel,
                centers,
                grad_prefactors,
                grad_term_y,
                prefactors.shape[1],
                n_terms,
                prefactors.stride(0),
                prefactors.stride(1),
                term_y.stride(0),
                term_y.stride(1),
                grad_out.stride(0),
                grad_out.stride(1),
                grad_prefactors.stride(0),
                grad_prefactors.stride(1),
                grad_term_y.stride(0),
                grad_term_y.stride(1),
                BLOCK_EDGES=block_edges,
                BLOCK_TERMS=block_terms,
            )
        return grad_prefactors, grad_term_y, None, None, None, None, None


def _packed_plain_real_site_basis_scatter_autograd(
    *,
    prefactors,
    term_y,
    term_group,
    term_channel,
    centers,
    n_atoms,
    n_channels,
):
    return _PackedPlainRealSiteBasisScatterFunction.apply(
        prefactors.contiguous(),
        term_y.contiguous(),
        term_group.contiguous(),
        term_channel.contiguous(),
        centers.contiguous(),
        int(n_atoms),
        int(n_channels),
    )


def site_basis_triton_available(*, device, dtype = torch.float64):
    resolved_device = torch.device(device)
    return bool(
        triton is not None
        and _triton_runtime_supported()
        and resolved_device.type == "cuda"
        and dtype in (torch.float32, torch.float64)
    )


@recordclass(('rc', 'lmbda', 'nradmax', 'lmax', 'kmax', 'possible_types', 'radial_basis', 'chemical_basis', 'charge_mode', 'charge_normalization_mode', 'charge_squash_scale', 'q_min', 'q_max', 'atomic_base_normalization', 'atomic_base_normalization_epsilon', 'factor_normalization', 'spherical_backend', 'source_backend', 'native_source_min_edges', 'dtype', 'complex_dtype', 'pace_cutoff_width', 'pace_spline_spacing', 'pace_inner_cutoff', 'pace_inner_cutoff_width', 'pace_crad_policy', 'spherical_normalization', 'chemical_embedding'))
class SiteBasisConfig:
    """Configuration for the single-channel site basis evaluator.

    Parameters
    ----------
    rc, lmbda
        Radial-basis hyperparameters supplied per ordered bond type
        ``(mu0, mu)``. For ``n_types`` chemical types, the expected order is
        ``list(itertools.product(possible_types, possible_types))``.
        A scalar/length-1 value is broadcast to all bond types.
    q_min, q_max
        Charge normalization bounds supplied per *element/site* type, not per
        bond type. These are used only when ``charge_mode='scalar'`` and map
        charges to the ``[-1, 1]`` interval before Chebyshev evaluation.
    possible_types
        Ordered integer chemical types present in the system. The order defines
        the bond-type flattening convention.
    pace_cutoff_width, pace_spline_spacing
        Ordered-bond outer-switch widths and requested PACE spline spacing.
        They are required only for ``radial_basis='PACE_ChebExpCos'``. This
        radial-only mode permits endpoint switch widths; strict standard-YACE
        export additionally requires ``0 < dcut < rc``.
    pace_inner_cutoff, pace_inner_cutoff_width, pace_crad_policy
        Explicit compatibility scope. The first implementation accepts only
        zero inner cutoffs and identity radial contractions.
    """
    kmax = 0
    possible_types = (0,)
    radial_basis = "ChebExpCos"
    chemical_basis = "delta"
    chemical_embedding = None
    charge_mode = "none"  # none | scalar
    charge_normalization_mode = "linear_clip"  # linear_clip | tanh
    charge_squash_scale = 1.0
    q_min = None
    q_max = None
    atomic_base_normalization = DEFAULT_ATOMIC_BASE_NORMALIZATION
    atomic_base_normalization_epsilon = 0.0
    factor_normalization = "bounded"  # none | bounded
    spherical_backend = "real"  # real | complex
    source_backend = "auto"  # auto | torch | native | native_cpu | native_cuda
    native_source_min_edges = 0
    dtype = torch.float64
    complex_dtype = torch.complex128
    pace_cutoff_width = None
    pace_spline_spacing = None
    pace_inner_cutoff = 0.0
    pace_inner_cutoff_width = 0.0
    pace_crad_policy = "identity"
    spherical_normalization = "orthonormal"  # orthonormal | pace_y00_one

    def __post_init__(self):
        possible_types = tuple(self.possible_types)
        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            for value in possible_types
        ):
            raise ValueError("possible_types must contain integer type identifiers")
        possible_types = tuple(int(value) for value in possible_types)
        ntypes = len(possible_types)
        if ntypes < 1:
            raise ValueError("possible_types must contain at least one type")
        if len(set(possible_types)) != ntypes:
            raise ValueError("possible_types must contain unique type identifiers")
        n_bond = ntypes * ntypes
        object.__setattr__(self, 'possible_types', possible_types)
        chemical_basis = str(self.chemical_basis)
        if chemical_basis == "delta":
            if self.chemical_embedding is not None:
                raise ValueError("Delta chemistry cannot declare a chemical_embedding matrix.")
        elif chemical_basis == "fixed_embedding":
            if self.chemical_embedding is None:
                raise ValueError("Fixed chemistry requires a chemical_embedding matrix.")
            matrix = tuple(tuple(float(value) for value in row)
                           for row in self.chemical_embedding)
            if (len(matrix) != ntypes or not matrix or not matrix[0] or
                    any(len(row) != len(matrix[0]) or
                        any(not math.isfinite(value) for value in row)
                        for row in matrix)):
                raise ValueError("chemical_embedding must have one finite, nonempty row per type.")
            singular_values = torch.linalg.svdvals(torch.tensor(matrix, dtype=torch.float64))
            tolerance = max(ntypes, len(matrix[0])) * torch.finfo(torch.float64).eps * float(singular_values[0])
            if (len(matrix[0]) > ntypes or
                    int(torch.count_nonzero(singular_values > tolerance)) != len(matrix[0])):
                raise ValueError("chemical_embedding must have independent columns.")
            object.__setattr__(self, "chemical_embedding", matrix)
        else:
            raise ValueError(f"Unsupported chemical basis: {chemical_basis}")
        object.__setattr__(self, "chemical_basis", chemical_basis)
        object.__setattr__(self, 'bond_inds', tuple(itertools.product(range(ntypes), range(ntypes))))
        object.__setattr__(self, 'bond_types', tuple((possible_types[i], possible_types[j]) for i, j in self.bond_inds))
        object.__setattr__(self, 'rc', tuple(self._expand_per_bond_type(self.rc, n_bond, 'rc')))
        object.__setattr__(self, 'lmbda', tuple(self._expand_per_bond_type(self.lmbda, n_bond, 'lmbda')))
        radial_kind = str(self.radial_basis).strip().lower().replace("_", "").replace("-", "")
        pace_radial = radial_kind in {"pacechebexpcos", "pacechebexpcosidentityspline"}
        if pace_radial:
            if self.pace_cutoff_width is None or self.pace_spline_spacing is None:
                raise ValueError(
                    "PACE_ChebExpCos requires explicit pace_cutoff_width and pace_spline_spacing."
                )
            pace_cutoff_width = tuple(
                self._expand_per_bond_type(self.pace_cutoff_width, n_bond, "pace_cutoff_width")
            )
            pace_spline_spacing = tuple(
                self._expand_per_bond_type(self.pace_spline_spacing, n_bond, "pace_spline_spacing")
            )
            pace_inner_cutoff = tuple(
                self._expand_per_bond_type(self.pace_inner_cutoff, n_bond, "pace_inner_cutoff")
            )
            pace_inner_cutoff_width = tuple(
                self._expand_per_bond_type(
                    self.pace_inner_cutoff_width,
                    n_bond,
                    "pace_inner_cutoff_width",
                )
            )
            pace_crad_policy = str(self.pace_crad_policy).strip().lower()
            if pace_crad_policy != "identity":
                raise NotImplementedError(
                    "PACE_ChebExpCos currently supports only pace_crad_policy='identity'."
                )
            coefficient_bytes = 0
            element_bytes = torch.empty((), dtype=self.dtype).element_size()
            for bond_index, (cutoff, lmbda, width, spacing, inner, inner_width) in enumerate(
                zip(
                    self.rc,
                    self.lmbda,
                    pace_cutoff_width,
                    pace_spline_spacing,
                    pace_inner_cutoff,
                    pace_inner_cutoff_width,
                )
            ):
                values = (cutoff, lmbda, width, spacing, inner, inner_width)
                if not all(math.isfinite(float(value)) for value in values):
                    raise ValueError(f"PACE radial bond {bond_index} contains a non-finite parameter.")
                if cutoff <= 0.0 or lmbda <= 0.0:
                    raise ValueError(f"PACE radial bond {bond_index} requires positive rc and lambda.")
                if width < 0.0 or width > cutoff:
                    raise ValueError(f"PACE radial bond {bond_index} requires 0 <= dcut <= rc.")
                if spacing <= 0.0 or int(cutoff / spacing) < 2:
                    raise ValueError(f"PACE radial bond {bond_index} requires at least two spline intervals.")
                if inner != 0.0 or inner_width != 0.0:
                    raise NotImplementedError(
                        "PACE_ChebExpCos does not yet support nonzero inner cutoffs."
                    )
                interval_count = int(cutoff / spacing)
                coefficient_bytes += (
                    (interval_count + 1)
                    * int(self.nradmax)
                    * 4
                    * element_bytes
                )
            if coefficient_bytes > _PACE_SPLINE_MAX_COEFFICIENT_BYTES:
                raise MemoryError(
                    "PACE spline coefficient tables exceed the 128 MiB allocation "
                    f"guard ({coefficient_bytes} bytes requested)."
                )
            object.__setattr__(self, "pace_cutoff_width", pace_cutoff_width)
            object.__setattr__(self, "pace_spline_spacing", pace_spline_spacing)
            object.__setattr__(self, "pace_inner_cutoff", pace_inner_cutoff)
            object.__setattr__(self, "pace_inner_cutoff_width", pace_inner_cutoff_width)
            object.__setattr__(self, "pace_crad_policy", pace_crad_policy)
        else:
            pace_inner_cutoff = self._expand_per_bond_type(
                self.pace_inner_cutoff,
                n_bond,
                "pace_inner_cutoff",
            )
            pace_inner_cutoff_width = self._expand_per_bond_type(
                self.pace_inner_cutoff_width,
                n_bond,
                "pace_inner_cutoff_width",
            )
            if (
                self.pace_cutoff_width is not None
                or self.pace_spline_spacing is not None
                or any(value != 0.0 for value in pace_inner_cutoff)
                or any(value != 0.0 for value in pace_inner_cutoff_width)
                or str(self.pace_crad_policy).strip().lower() != "identity"
            ):
                raise ValueError(
                    "PACE-specific radial knobs apply only to "
                    "radial_basis='PACE_ChebExpCos'."
                )
        q_min = self.q_min if self.q_min is not None else [-1.0] * ntypes
        q_max = self.q_max if self.q_max is not None else [1.0] * ntypes
        object.__setattr__(self, 'q_min', tuple(self._expand_per_type(q_min, ntypes, 'q_min')))
        object.__setattr__(self, 'q_max', tuple(self._expand_per_type(q_max, ntypes, 'q_max')))
        if any(qmax <= qmin for qmin, qmax in zip(self.q_min, self.q_max)):
            raise ValueError('Each q_max must be strictly greater than q_min')
        normalization = str(self.atomic_base_normalization).strip().lower()
        removed_normalizations = {"block_norm", "soft_neighbor_block_norm"}
        if normalization in removed_normalizations:
            raise ValueError(
                "atomic_base_normalization values 'block_norm' and "
                "'soft_neighbor_block_norm' were removed from the public "
                "normalization policy because they are empirical post-A "
                "rescalings. Use 'soft_neighbor' with factor_normalization="
                "'bounded' for the rigorous normalized path, or 'none' for "
                "extensive density moments."
            )
        valid_normalizations = {"none", "soft_neighbor"}
        if normalization not in valid_normalizations:
            raise ValueError(
                "atomic_base_normalization must be one of "
                f"{sorted(valid_normalizations)}; got {self.atomic_base_normalization!r}"
        )
        if float(self.atomic_base_normalization_epsilon) < 0.0:
            raise ValueError("atomic_base_normalization_epsilon must be non-negative")
        charge_normalization_mode = str(self.charge_normalization_mode).strip().lower()
        if charge_normalization_mode not in {"linear_clip", "tanh"}:
            raise ValueError("charge_normalization_mode must be one of linear_clip or tanh")
        if float(self.charge_squash_scale) <= 0.0:
            raise ValueError("charge_squash_scale must be positive")
        factor_normalization = str(self.factor_normalization).strip().lower()
        if factor_normalization not in {"none", "bounded"}:
            raise ValueError("factor_normalization must be one of none or bounded")
        if pace_radial and factor_normalization != "none":
            raise ValueError(
                "PACE_ChebExpCos requires factor_normalization='none'; its cubic "
                "spline has no certified unit bound."
            )
        spherical_backend = str(self.spherical_backend).strip().lower()
        if spherical_backend not in {"complex", "real"}:
            raise ValueError("spherical_backend must be one of complex or real")
        spherical_normalization = str(self.spherical_normalization).strip().lower()
        if spherical_normalization not in {"orthonormal", "pace_y00_one"}:
            raise ValueError(
                "spherical_normalization must be one of orthonormal or pace_y00_one"
            )
        source_backend = str(self.source_backend).strip().lower()
        if source_backend not in {
            "auto",
            "torch",
            "native",
            "native_cpu",
            "native_cuda",
        }:
            raise ValueError(
                "source_backend must be one of auto, torch, native, "
                "native_cpu, or native_cuda"
            )
        if int(self.native_source_min_edges) < 0:
            raise ValueError("native_source_min_edges must be non-negative")
        if pace_radial and source_backend in {"native", "native_cpu", "native_cuda"}:
            raise NotImplementedError(
                "PACE_ChebExpCos training currently supports source_backend='torch' or 'auto'; "
                "an explicit native request cannot fall back silently."
            )
        if chemical_basis == "fixed_embedding" and source_backend != "torch":
            raise NotImplementedError(
                "Fixed chemical embedding currently requires source_backend='torch'."
            )
        object.__setattr__(self, "charge_normalization_mode", charge_normalization_mode)
        object.__setattr__(self, "charge_squash_scale", float(self.charge_squash_scale))
        object.__setattr__(self, "atomic_base_normalization", normalization)
        object.__setattr__(self, "atomic_base_normalization_epsilon", float(self.atomic_base_normalization_epsilon))
        object.__setattr__(self, "factor_normalization", factor_normalization)
        object.__setattr__(self, "spherical_backend", spherical_backend)
        object.__setattr__(self, "spherical_normalization", spherical_normalization)
        object.__setattr__(self, "source_backend", source_backend)
        object.__setattr__(
            self,
            "native_source_min_edges",
            int(self.native_source_min_edges),
        )

    @staticmethod
    def _expand_per_bond_type(values, n_bond, name):
        if isinstance(values, (float, int)):
            return [float(values)] * n_bond
        vals = list(values)
        if len(vals) == 1:
            return [float(vals[0])] * n_bond
        if len(vals) != n_bond:
            raise ValueError(f"{name} must have length 1 or n_types**2={n_bond}; got length {len(vals)}")
        return [float(v) for v in vals]

    @staticmethod
    def _expand_per_type(values, ntypes, name):
        if isinstance(values, (float, int)):
            return [float(values)] * ntypes
        vals = list(values)
        if len(vals) == 1:
            return [float(vals[0])] * ntypes
        if len(vals) != ntypes:
            raise ValueError(f"{name} must have length 1 or n_types={ntypes}; got length {len(vals)}")
        return [float(v) for v in vals]

    @property
    def ntypes(self):
        return len(self.possible_types)

    @property
    def n_bond_channels(self):
        return self.ntypes * self.ntypes


@recordclass(('channels', 'raw_atomic_base', 'final_atomic_base', 'edge_dx', 'edge_dq_center', 'edge_dq_neighbor', 'centers', 'neighs', 'bond_idx', 'r', 'dr_dx', 'n_atoms'), frozen = True)
class SiteBasisVJPRecord:
    """Reusable forward record for compact analytic position VJPs."""


@recordclass(('indices', 'm_positions'), frozen = True)
class _ChannelIndexTensors:
    pass


@recordclass(
    (
        'group_indices',
        'channel_indices',
        'angular_indices',
        'radial_keys',
        'radial_group_indices',
        'chemical_keys',
        'chemical_group_indices',
        'charge_keys',
        'charge_group_indices',
        'angular_momenta',
    ),
    frozen = True,
)
class _PackedPlainEdgeTableTensors:
    pass


@recordclass(('mu0', 'mu', 'kappa0', 'kappa', 'n', 'l', 'channel_indices', 'm_positions', '_tensor_cache'), frozen = True)
class _PlainChannelGroup:
    _tensor_cache = field(default_factory=dict, init=False, repr=False, compare=False)

    def tensors(self, device):
        key = str(device)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = _ChannelIndexTensors(
                indices=torch.tensor(self.channel_indices, dtype=torch.long, device=device),
                m_positions=torch.tensor(self.m_positions, dtype=torch.long, device=device),
            )
            self._tensor_cache[key] = cached
        return cached


@recordclass(('channels', 'plain_groups', 'fallback_entries'), frozen = True)
class _BasisChannelSchedule:
    pass


def _source_kernel_backend(
    requested,
    tensor,
    native_source_min_edges,
    cuda_operation=None,
):
    requested = str(requested)
    if requested == "torch":
        return "reference"
    if requested == "deterministic":
        if str(cuda_operation) in {
            "density_accumulate",
            "edge_outer_accumulate",
            "softmax_gaussian_role_density",
        }:
            return "deterministic"
        return "reference"
    if requested == "native":
        return "native"
    if requested == "native_cpu":
        if tensor.device.type != "cpu":
            raise ValueError(
                "source_backend='native_cpu' requires CPU source tensors"
            )
        return "native"
    if requested == "native_cuda":
        if tensor.device.type != "cuda":
            raise ValueError(
                "source_backend='native_cuda' requires CUDA source tensors"
            )
        return "native"
    require_native = str(
        os.environ.get("YE3T_REQUIRE_NATIVE", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if require_native:
        return "native"
    if (
        tensor.device.type == "cpu"
        and int(tensor.shape[0]) >= int(native_source_min_edges)
    ):
        return "auto"
    if tensor.device.type == "cuda" and cuda_operation is not None:
        from ye3t.runtime import native_execution_plan_capabilities

        capabilities = native_execution_plan_capabilities()
        if str(cuda_operation) in capabilities["cuda_operations"]:
            return "auto"
    return "reference"


def _native_density_backend_label(runtime_backend, tensor, work_items):
    if runtime_backend == "native":
        return "native_cuda" if tensor.device.type == "cuda" else "native_cpu"
    if runtime_backend != "auto":
        return None
    from ye3t.runtime import native_execution_plan_capabilities

    capabilities = native_execution_plan_capabilities()
    if tensor.device.type == "cpu" and capabilities["cpu"]:
        return "native_cpu"
    threshold = capabilities["cuda_density_auto_min_work_items"]
    if (
        tensor.device.type == "cuda"
        and capabilities["cuda"]
        and threshold is not None
        and int(work_items) >= int(threshold)
    ):
        return "native_cuda"
    return None


def _native_edge_outer_backend_label(
    runtime_backend,
    tensor,
    work_items,
):
    if runtime_backend == "native":
        return (
            "native_edge_outer_cuda"
            if tensor.device.type == "cuda"
            else "native_edge_outer_cpu"
        )
    if runtime_backend != "auto":
        return None
    from ye3t.runtime import native_execution_plan_capabilities

    capabilities = native_execution_plan_capabilities()
    if tensor.device.type == "cuda":
        threshold = capabilities[
            "cuda_edge_outer_auto_min_work_items"
        ]
        native_label = "native_edge_outer_cuda"
    else:
        threshold = capabilities[
            "cpu_edge_outer_auto_min_work_items"
        ]
        native_label = "native_edge_outer_cpu"
    if (
        capabilities[tensor.device.type]
        and threshold is not None
        and int(work_items) >= int(threshold)
    ):
        return native_label
    return None


class SphericalHarmonicsProvider:
    def __init__(
        self,
        lmax,
        *,
        backend = "complex",
        source_backend = "auto",
        native_source_min_edges = 0,
    ):
        self.lmax = lmax
        self.backend = str(backend).strip().lower()
        self.source_backend = str(source_backend).strip().lower()
        self.native_source_min_edges = int(native_source_min_edges)
        self.last_runtime_backend = None
        self.basis = angular_basis_for_backend(self.backend)

    def convention_metadata(self):
        metadata = self.basis.convention_metadata(lmax=int(self.lmax))
        metadata["spherical_backend"] = str(self.backend)
        if self.backend == "real":
            metadata["complex_to_real_transforms"] = tuple(
                self.basis.transform_metadata(l) for l in range(int(self.lmax) + 1)
            )
        return metadata

    def all_m(self, l, theta, phi):
        return self.basis.all_m(l, theta, phi)

    def all_m_cartesian_with_derivatives(self, l, xyz):
        runtime_backend = _source_kernel_backend(
            self.source_backend,
            xyz,
            self.native_source_min_edges,
            cuda_operation="spherical_harmonics_with_derivative",
        )
        self.last_runtime_backend = (
            "native_cuda"
            if runtime_backend in {"auto", "native"}
            and xyz.device.type == "cuda"
            else runtime_backend
        )
        if runtime_backend != "reference":
            from ye3t.runtime import spherical_harmonics_with_derivative

            values, derivatives = spherical_harmonics_with_derivative(
                xyz,
                int(l),
                real_output=self.backend == "real",
                backend=runtime_backend,
            )
            return values.transpose(0, 1), derivatives.permute(1, 0, 2)
        return self.basis.cartesian_derivative(l, xyz)

    def __call__(self, l, m, theta, phi):
        return self.basis.single_m(l, m, theta, phi)


class ChebyshevProvider:
    def __call__(self, x, n):
        return chebyshev_poly_first(x, n)


class RadialBasisProvider:
    """Callable extension point for radial functions in ``phi_nlm``."""

    derivative_backend = None

    def __call__(self, *, n, l, bond_idx, r, evaluator):
        ...

    def evaluate(self, *, n, l, bond_idx, r, evaluator):
        return self(n=n, l=l, bond_idx=bond_idx, r=r, evaluator=evaluator)

    def derivative(self, *, n, l, bond_idx, r, evaluator):
        del n, l, bond_idx, r, evaluator
        raise NotImplementedError(
            "Custom or learned radial bases require a declared derivative backend; "
            "override RadialBasisProvider.derivative(...) for force paths."
        )

    def evaluate_with_derivative(self, *, n, l, bond_idx, r, evaluator):
        return (
            self.evaluate(
                n=n,
                l=l,
                bond_idx=bond_idx,
                r=r,
                evaluator=evaluator,
            ),
            self.derivative(
                n=n,
                l=l,
                bond_idx=bond_idx,
                r=r,
                evaluator=evaluator,
            ),
        )

    def max_abs(self, *, n, l, evaluator):
        raise NotImplementedError("Custom radial providers need an analytical max_abs hook for bounded normalization.")

    def convention_metadata(self, *, evaluator):
        del evaluator
        raise NotImplementedError(
            "Custom radial providers must declare stable JSON source metadata; "
            "object identity cannot define a reusable descriptor or saved model."
        )


class ChemicalBasisProvider:
    """Callable extension point for chemical embeddings in ``A_inlm``."""

    def __call__(self, *, mu0_edge, mu_edge, mu0, mu, evaluator):
        ...


class ChargeBasisProvider:
    """Callable extension point for scalar auxiliary/charge embeddings."""

    def __call__(
        self,
        *,
        q_i,
        q_j,
        mu0_edge,
        mu_edge,
        kappa0,
        kappa,
        evaluator,
    ):
        ...


class DefaultRadialBasisProvider:
    derivative_backend = "analytic"
    last_runtime_backend = None

    def __init__(self):
        self._pace_spline_cache = {}

    @staticmethod
    def _is_pace_radial(evaluator):
        key = str(evaluator.cfg.radial_basis).strip().lower()
        key = key.replace("_", "").replace("-", "")
        return key in {"pacechebexpcos", "pacechebexpcosidentityspline"}

    def _pace_spline_table(self, *, bond, r, evaluator):
        bond = int(bond)
        key = (
            bond,
            float(evaluator.cfg.rc[bond]),
            float(evaluator.cfg.lmbda[bond]),
            float(evaluator.cfg.pace_cutoff_width[bond]),
            float(evaluator.cfg.pace_spline_spacing[bond]),
            float(evaluator.cfg.pace_inner_cutoff[bond]),
            float(evaluator.cfg.pace_inner_cutoff_width[bond]),
            str(evaluator.cfg.pace_crad_policy),
            int(evaluator.cfg.nradmax),
            str(r.device),
            str(r.dtype),
        )
        cached = self._pace_spline_cache.get(key)
        if cached is None:
            cached = _pace_uniform_cubic_spline_coefficients(
                rc=evaluator.cfg.rc[bond],
                requested_spacing=evaluator.cfg.pace_spline_spacing[bond],
                cutoff_width=evaluator.cfg.pace_cutoff_width[bond],
                lmbda=evaluator.cfg.lmbda[bond],
                radial_count=int(evaluator.cfg.nradmax),
                device=r.device,
                dtype=r.dtype,
            )
            self._pace_spline_cache[key] = cached
        return cached

    def _pace_channel_with_derivative(self, *, n, bond_idx, r, evaluator):
        require_native = str(os.environ.get("YE3T_REQUIRE_NATIVE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if evaluator.cfg.source_backend == "auto" and require_native:
            raise NotImplementedError(
                "PACE_ChebExpCos has no native training-side radial kernel; "
                "YE3T_REQUIRE_NATIVE forbids the Torch spline implementation."
            )
        radial_label = int(n)
        if radial_label < 1 or radial_label > int(evaluator.cfg.nradmax):
            raise ValueError(
                "PACE_ChebExpCos radial labels must satisfy "
                f"1 <= n <= {int(evaluator.cfg.nradmax)}; got {radial_label}."
            )
        values = torch.zeros_like(r)
        derivatives = torch.zeros_like(r)
        for bond in range(evaluator.cfg.n_bond_channels):
            indices = torch.nonzero(bond_idx == int(bond), as_tuple=False).reshape(-1)
            if int(indices.numel()) == 0:
                continue
            coefficients, interval_count = self._pace_spline_table(
                bond=bond,
                r=r,
                evaluator=evaluator,
            )
            bond_values, bond_derivatives = _pace_uniform_cubic_spline_evaluate_with_derivative(
                r.index_select(0, indices),
                coefficients=coefficients,
                interval_count=interval_count,
                rc=evaluator.cfg.rc[int(bond)],
            )
            values = values.index_copy(0, indices, bond_values[:, radial_label - 1])
            derivatives = derivatives.index_copy(
                0,
                indices,
                bond_derivatives[:, radial_label - 1],
            )
        self.last_runtime_backend = "torch_pace_uniform_cubic_spline"
        return values, derivatives

    def _channel_basis(self, *, n, l, bond_idx, r, evaluator):
        del l
        rc, lmbda = evaluator._prepare_cutoffs_on_device(r.device)
        return radial_basis_for_kind(
            evaluator.cfg.radial_basis,
            n=int(n),
            nmax=int(evaluator.cfg.nradmax),
            rc=rc[bond_idx],
            lmbda=lmbda[bond_idx],
        )

    def __call__(self, *, n, l, bond_idx, r, evaluator):
        return self.evaluate(n=n, l=l, bond_idx=bond_idx, r=r, evaluator=evaluator)

    def evaluate(self, *, n, l, bond_idx, r, evaluator):
        if self._is_pace_radial(evaluator):
            return self._pace_channel_with_derivative(
                n=n,
                bond_idx=bond_idx,
                r=r,
                evaluator=evaluator,
            )[0]
        return self._channel_basis(n=n, l=l, bond_idx=bond_idx, r=r, evaluator=evaluator).evaluate(r)

    def derivative(self, *, n, l, bond_idx, r, evaluator):
        if self._is_pace_radial(evaluator):
            return self._pace_channel_with_derivative(
                n=n,
                bond_idx=bond_idx,
                r=r,
                evaluator=evaluator,
            )[1]
        return self._channel_basis(n=n, l=l, bond_idx=bond_idx, r=r, evaluator=evaluator).derivative(r)

    def evaluate_with_derivative(self, *, n, l, bond_idx, r, evaluator):
        if self._is_pace_radial(evaluator):
            return self._pace_channel_with_derivative(
                n=n,
                bond_idx=bond_idx,
                r=r,
                evaluator=evaluator,
            )
        key = str(evaluator.cfg.radial_basis).strip().lower()
        key = key.replace("_", "").replace("-", "")
        runtime_backend = _source_kernel_backend(
            evaluator.cfg.source_backend,
            r,
            evaluator.cfg.native_source_min_edges,
            cuda_operation="cheb_exp_cos_radial_with_derivative",
        )
        self.last_runtime_backend = (
            "native_cuda"
            if runtime_backend in {"auto", "native"}
            and r.device.type == "cuda"
            else runtime_backend
        )
        if key == "chebexpcos" and runtime_backend != "reference":
            from ye3t.runtime import cheb_exp_cos_radial_with_derivative

            rc, lmbda = evaluator._prepare_cutoffs_on_device(r.device)
            return cheb_exp_cos_radial_with_derivative(
                r,
                rc[bond_idx],
                lmbda[bond_idx],
                int(n),
                backend=runtime_backend,
            )
        return (
            self.evaluate(
                n=n,
                l=l,
                bond_idx=bond_idx,
                r=r,
                evaluator=evaluator,
            ),
            self.derivative(
                n=n,
                l=l,
                bond_idx=bond_idx,
                r=r,
                evaluator=evaluator,
            ),
        )

    def max_abs(self, *, n, l, evaluator):
        del n, l, evaluator
        return 1.0

    def convention_metadata(self, *, evaluator):
        metadata = {
            "provider": "DefaultRadialBasisProvider",
            "kind": str(evaluator.cfg.radial_basis),
            "normalization": "bounded" if evaluator._factor_normalization_enabled() else "raw",
            "nradmax": int(evaluator.cfg.nradmax),
            "rc": tuple(float(x) for x in evaluator.cfg.rc),
            "lmbda": tuple(float(x) for x in evaluator.cfg.lmbda),
            "derivative_backend": self.derivative_backend,
        }
        if self._is_pace_radial(evaluator):
            metadata.update(
                {
                    "dtype": str(evaluator.cfg.dtype),
                    "radial_label_indexing": "one_based_n_maps_to_zero_based_g",
                    "outer_cutoff_width": tuple(float(x) for x in evaluator.cfg.pace_cutoff_width),
                    "spline_requested_spacing": tuple(float(x) for x in evaluator.cfg.pace_spline_spacing),
                    "spline_interval_policy": "trunc_rc_over_requested_spacing",
                    "spline_interpolation": "uniform_cubic_hermite",
                    "inner_cutoff": tuple(float(x) for x in evaluator.cfg.pace_inner_cutoff),
                    "inner_cutoff_width": tuple(float(x) for x in evaluator.cfg.pace_inner_cutoff_width),
                    "crad_policy": str(evaluator.cfg.pace_crad_policy),
                }
            )
        return metadata


class DefaultChemicalBasisProvider:
    def __call__(self, *, mu0_edge, mu_edge, mu0, mu, evaluator):
        return evaluator._default_chemical_basis(mu0_edge=mu0_edge, mu_edge=mu_edge, mu0=mu0, mu=mu)


class DefaultChargeBasisProvider:
    def __call__(
        self,
        *,
        q_i,
        q_j,
        mu0_edge,
        mu_edge,
        kappa0,
        kappa,
        evaluator,
    ):
        return evaluator._default_charge_basis(
            q_i=q_i,
            q_j=q_j,
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            kappa0=kappa0,
            kappa=kappa,
        )


class SiteBasisV2(torch.nn.Module):
    """Vectorized single-channel ACE site-basis evaluator.

    The module evaluates only the channels actually required by the descriptor
    set and aggregates neighbor contributions with ``index_add_``.
    """

    def __init__(
        self,
        config,
        *,
        radial_provider = None,
        chemical_provider = None,
        charge_provider = None,
        spherical_provider = None,
    ):
        super().__init__()
        self.cfg = config
        self.sph = spherical_provider or SphericalHarmonicsProvider(
            config.lmax,
            backend=config.spherical_backend,
            source_backend=config.source_backend,
            native_source_min_edges=config.native_source_min_edges,
        )
        self.cheb = ChebyshevProvider()
        self.radial_provider = radial_provider or DefaultRadialBasisProvider()
        self.chemical_provider = chemical_provider or DefaultChemicalBasisProvider()
        self.charge_provider = charge_provider or DefaultChargeBasisProvider()
        self._type_to_local = {t: i for i, t in enumerate(self.cfg.possible_types)}
        self._cutoff_cache = {}
        self._charge_bound_cache = {}
        self._factor_scale_cache = {}
        self._basis_channel_schedule_cache = {}
        self._packed_plain_edge_table_cache = {}
        self._last_scatter_backend = "index_add"
        self._last_vjp_scatter_backend = "index_add"
        self._last_radial_table_backend = None
        self._last_angular_table_backend = None
        self._last_plain_product_backend = None
        self._last_plain_adjoint_backend = None
        self._last_profile = {}
        self._atomic_base_cache_seen_keys = set()
        self._atomic_base_cache_hits = 0
        self._atomic_base_cache_misses = 0
        self._last_atomic_base_cache = None
        self._evaluation_context = None

    def set_evaluation_context(self, context):
        if context is not None and not isinstance(context, EvaluationContext):
            raise TypeError("context must be an EvaluationContext or None")
        self._evaluation_context = context
        return self

    def evaluation_context(self):
        if self._evaluation_context is None:
            self._evaluation_context = EvaluationContext(runtime_cache=DescriptorRuntimeCache())
        return self._evaluation_context

    def last_profile(self):
        return dict(self._last_profile)

    def source_runtime_report(self):
        return {
            "requested_backend": str(self.cfg.source_backend),
            "native_source_min_edges": int(
                self.cfg.native_source_min_edges
            ),
            "radial_backend": getattr(
                self.radial_provider,
                "last_runtime_backend",
                None,
            ),
            "angular_backend": getattr(
                self.sph,
                "last_runtime_backend",
                None,
            ),
            "radial_table_backend": self._last_radial_table_backend,
            "angular_table_backend": self._last_angular_table_backend,
            "plain_product_backend": self._last_plain_product_backend,
            "plain_adjoint_backend": self._last_plain_adjoint_backend,
            "radial_family": str(self.cfg.radial_basis),
            "angular_convention": str(self.cfg.spherical_backend),
            "spherical_normalization": str(self.cfg.spherical_normalization),
            "density_accumulation_backend": str(
                self._last_scatter_backend
            ),
            "density_adjoint_backend": getattr(
                self,
                "_last_density_adjoint_backend",
                None,
            ),
        }

    def _native_source_derivative_tables(
        self,
        *,
        x_ij,
        bond_idx,
        r,
        channel_schedule,
        include_radial=True,
        include_angular=True,
    ):
        self._last_radial_table_backend = None
        self._last_angular_table_backend = None
        radial_keys = (
            {
                (int(group.n), int(group.l))
                for group in channel_schedule.plain_groups
            }
            if bool(include_radial)
            else set()
        )
        angular_momenta = (
            {
                int(group.l)
                for group in channel_schedule.plain_groups
            }
            if bool(include_angular)
            else set()
        )
        for _, channel in channel_schedule.fallback_entries:
            if bool(include_radial) and channel.eta is None:
                radial_keys.add((int(channel.n), int(channel.l)))
            if bool(include_angular) and channel.eta is None:
                angular_momenta.add(int(channel.l))

        radial_cache = {}
        radial_kind = str(self.cfg.radial_basis).strip().lower()
        radial_kind = radial_kind.replace("_", "").replace("-", "")
        radial_backend = _source_kernel_backend(
            self.cfg.source_backend,
            r,
            self.cfg.native_source_min_edges,
            cuda_operation="cheb_exp_cos_radial_table_with_derivative",
        )
        if (
            radial_keys
            and isinstance(
                self.radial_provider,
                DefaultRadialBasisProvider,
            )
            and radial_kind == "chebexpcos"
            and radial_backend != "reference"
        ):
            from ye3t.runtime import (
                cheb_exp_cos_radial_table_with_derivative,
            )

            rc, lmbda = self._prepare_cutoffs_on_device(r.device)
            maximum_n = max(key[0] for key in radial_keys)
            values, derivatives = (
                cheb_exp_cos_radial_table_with_derivative(
                    r,
                    rc[bond_idx],
                    lmbda[bond_idx],
                    maximum_n,
                    backend=radial_backend,
                )
            )
            for n, l in radial_keys:
                radial_cache[(n, l)] = (
                    self._normalize_radial_values_with_derivative(
                        n,
                        l,
                        values[:, n],
                        derivatives[:, n],
                    )
                )
            self.radial_provider.last_runtime_backend = (
                "native_cuda"
                if r.device.type == "cuda"
                else "native"
            )
            self._last_radial_table_backend = (
                self.radial_provider.last_runtime_backend
            )

        angular_cache = {}
        angular_backend = _source_kernel_backend(
            self.cfg.source_backend,
            x_ij,
            self.cfg.native_source_min_edges,
            cuda_operation="spherical_harmonics_table_with_derivative",
        )
        if angular_momenta and angular_backend != "reference":
            from ye3t.runtime import (
                spherical_harmonics_table_with_derivative,
            )

            maximum_l = max(angular_momenta)
            values, derivatives = (
                spherical_harmonics_table_with_derivative(
                    x_ij,
                    maximum_l,
                    real_output=self.cfg.spherical_backend == "real",
                    backend=angular_backend,
                )
            )
            for l in angular_momenta:
                start = l * l
                stop = (l + 1) * (l + 1)
                degree_values = values[:, start:stop].transpose(0, 1)
                degree_derivatives = derivatives[
                    :, start:stop, :
                ].permute(1, 0, 2)
                angular_cache[l] = (
                    self._normalize_spherical_values_with_derivatives(
                        l,
                        degree_values,
                        degree_derivatives,
                    )
                )
            self.sph.last_runtime_backend = (
                "native_cuda"
                if x_ij.device.type == "cuda"
                else "native"
            )
            self._last_angular_table_backend = (
                self.sph.last_runtime_backend
            )
        return radial_cache, angular_cache

    def _plain_source_product_tables(
        self,
        *,
        x_ij,
        mu0_edge,
        mu_edge,
        q_i,
        q_j,
        bond_idx,
        r,
        channel_schedule,
        radial_cache,
        angular_cache,
        chemical_cache,
        charge_cache,
        target_dtype,
    ):
        if (
            not channel_schedule.plain_groups
            or channel_schedule.fallback_entries
        ):
            return None

        packed = self._packed_plain_edge_table_tensors(
            channel_schedule,
            x_ij.device,
        )
        if int(packed.channel_indices.numel()) != len(
            channel_schedule.channels
        ):
            return None

        radial_values = []
        radial_derivatives = []
        angular_values = []
        angular_derivatives = []
        prefactors = []
        prefactor_derivatives_center = []
        prefactor_derivatives_neighbor = []
        for group in channel_schedule.plain_groups:
            chem_key = (group.mu0, group.mu)
            if chem_key not in chemical_cache:
                chemical_cache[chem_key] = self._chemical_basis(
                    mu0_edge,
                    mu_edge,
                    group.mu0,
                    group.mu,
                )
            chem = chemical_cache[chem_key].to(target_dtype)

            charge_key = (
                group.mu0,
                group.mu,
                group.kappa0,
                group.kappa,
            )
            if charge_key not in charge_cache:
                charge_cache[charge_key] = (
                    self._charge_basis_with_derivative(
                        q_i,
                        q_j,
                        mu0_edge,
                        mu_edge,
                        group.kappa0,
                        group.kappa,
                    )
                )
            qf, dqf_i, dqf_j = charge_cache[charge_key]

            radial_key = (group.n, group.l)
            if radial_key not in radial_cache:
                radial_cache[radial_key] = (
                    self._radial_basis_with_derivative(
                        group.n,
                        group.l,
                        bond_idx,
                        r,
                    )
                )
            rf, drf_dr = radial_cache[radial_key]

            if group.l not in angular_cache:
                y_values, y_derivatives = (
                    self.sph.all_m_cartesian_with_derivatives(
                        group.l,
                        x_ij,
                    )
                )
                angular_cache[group.l] = (
                    self._normalize_spherical_values_with_derivatives(
                        group.l,
                        y_values,
                        y_derivatives,
                    )
                )
            y_all, dy_all = angular_cache[group.l]
            group_tensors = group.tensors(x_ij.device)

            radial_values.append(rf.to(target_dtype))
            radial_derivatives.append(drf_dr.to(target_dtype))
            angular_values.append(
                y_all.index_select(
                    0,
                    group_tensors.m_positions,
                ).to(target_dtype)
            )
            angular_derivatives.append(
                dy_all.index_select(
                    0,
                    group_tensors.m_positions,
                ).to(target_dtype)
            )
            prefactors.append(chem * qf.to(target_dtype))
            prefactor_derivatives_center.append(
                chem * dqf_i.to(target_dtype)
            )
            prefactor_derivatives_neighbor.append(
                chem * dqf_j.to(target_dtype)
            )

        return (
            torch.stack(radial_values),
            torch.stack(radial_derivatives),
            torch.cat(angular_values, dim=0),
            torch.cat(angular_derivatives, dim=0),
            torch.stack(prefactors),
            torch.stack(prefactor_derivatives_center),
            torch.stack(prefactor_derivatives_neighbor),
            packed.group_indices,
            packed.channel_indices,
        )

    def _native_plain_source_product(
        self,
        *,
        x_ij,
        mu0_edge,
        mu_edge,
        q_i,
        q_j,
        bond_idx,
        r,
        dr_dx,
        channel_schedule,
        radial_cache,
        angular_cache,
        chemical_cache,
        charge_cache,
        target_dtype,
    ):
        self._last_plain_product_backend = None
        runtime_backend = _source_kernel_backend(
            self.cfg.source_backend,
            x_ij,
            self.cfg.native_source_min_edges,
            cuda_operation="plain_site_basis_product_with_derivative",
        )
        if runtime_backend == "reference":
            return None
        tables = self._plain_source_product_tables(
            x_ij=x_ij,
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            q_i=q_i,
            q_j=q_j,
            bond_idx=bond_idx,
            r=r,
            channel_schedule=channel_schedule,
            radial_cache=radial_cache,
            angular_cache=angular_cache,
            chemical_cache=chemical_cache,
            charge_cache=charge_cache,
            target_dtype=target_dtype,
        )
        if tables is None:
            return None

        from ye3t.runtime import (
            plain_site_basis_product_with_derivative,
        )

        output = plain_site_basis_product_with_derivative(
            *tables[:7],
            dr_dx.to(target_dtype),
            tables[7],
            tables[8],
            len(channel_schedule.channels),
            backend=runtime_backend,
        )
        self._last_plain_product_backend = (
            "native_cuda"
            if x_ij.device.type == "cuda"
            else "native_cpu"
        )
        return output

    def _native_plain_source_adjoint(
        self,
        *,
        x_ij,
        mu0_edge,
        mu_edge,
        q_i,
        q_j,
        bond_idx,
        r,
        dr_dx,
        channel_schedule,
        radial_cache,
        angular_cache,
        chemical_cache,
        charge_cache,
        edge_adjoint,
        edge_weights=None,
        edge_weight_derivatives=None,
    ):
        self._last_plain_adjoint_backend = None
        runtime_backend = _source_kernel_backend(
            self.cfg.source_backend,
            x_ij,
            self.cfg.native_source_min_edges,
            cuda_operation="plain_site_basis_product_adjoint",
        )
        if runtime_backend == "reference":
            return None
        target_dtype = edge_adjoint.dtype
        tables = self._plain_source_product_tables(
            x_ij=x_ij,
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            q_i=q_i,
            q_j=q_j,
            bond_idx=bond_idx,
            r=r,
            channel_schedule=channel_schedule,
            radial_cache=radial_cache,
            angular_cache=angular_cache,
            chemical_cache=chemical_cache,
            charge_cache=charge_cache,
            target_dtype=target_dtype,
        )
        if tables is None:
            return None
        if edge_weights is None:
            edge_weights = torch.ones(
                int(x_ij.shape[0]),
                dtype=target_dtype,
                device=x_ij.device,
            )
        else:
            edge_weights = edge_weights.to(target_dtype)
        if edge_weight_derivatives is None:
            edge_weight_derivatives = torch.zeros(
                (int(x_ij.shape[0]), 3),
                dtype=target_dtype,
                device=x_ij.device,
            )
        else:
            edge_weight_derivatives = edge_weight_derivatives.to(
                target_dtype
            )

        from ye3t.runtime import plain_site_basis_product_adjoint

        output = plain_site_basis_product_adjoint(
            *tables[:7],
            dr_dx.to(target_dtype),
            edge_weights,
            edge_weight_derivatives,
            tables[7],
            tables[8],
            edge_adjoint,
            backend=runtime_backend,
        )
        self._last_plain_adjoint_backend = (
            "native_cuda"
            if x_ij.device.type == "cuda"
            else "native_cpu"
        )
        return output

    def _density_accumulate(self, edge_values, centers, atom_count):
        runtime_backend = _source_kernel_backend(
            self.cfg.source_backend,
            edge_values,
            self.cfg.native_source_min_edges,
            cuda_operation="density_accumulate",
        )
        if runtime_backend != "reference":
            from ye3t.runtime import density_accumulate

            output = density_accumulate(
                edge_values,
                centers,
                int(atom_count),
                backend=runtime_backend,
            )
            native_label = _native_density_backend_label(
                runtime_backend,
                edge_values,
                edge_values.numel(),
            )
            self._last_scatter_backend = native_label or (
                "triton"
                if _triton_available_for_tensor(
                    edge_values.real.contiguous(),
                    centers,
                )
                else "index_add"
            )
            return output
        output = _scatter_sum_maybe_triton(
            edge_values,
            centers,
            int(atom_count),
        )
        self._last_scatter_backend = (
            "triton"
            if _triton_available_for_tensor(
                edge_values.real.contiguous(),
                centers,
            )
            else "index_add"
        )
        return output

    def _density_accumulate_adjoint(self, atomic_adjoint, centers):
        runtime_backend = _source_kernel_backend(
            self.cfg.source_backend,
            atomic_adjoint,
            self.cfg.native_source_min_edges,
            cuda_operation="density_accumulate_adjoint",
        )
        if runtime_backend != "reference":
            from ye3t.runtime import density_accumulate_adjoint

            output = density_accumulate_adjoint(
                atomic_adjoint,
                centers,
                backend=runtime_backend,
            )
            native_label = _native_density_backend_label(
                runtime_backend,
                atomic_adjoint,
                int(centers.numel()) * int(atomic_adjoint.shape[1]),
            )
            self._last_density_adjoint_backend = (
                native_label or "index_select"
            )
            return output
        self._last_density_adjoint_backend = "index_select"
        return atomic_adjoint.index_select(0, centers)

    def last_atomic_base_cache_report(self):
        if self._last_atomic_base_cache is None:
            return {
                "hits": int(self._atomic_base_cache_hits),
                "misses": int(self._atomic_base_cache_misses),
                "memory_size_bytes": 0,
                "stores_raw_atomic_base": False,
                "stores_normalized_atomic_base": False,
                "stores_normalization_derivative": False,
            }
        return self._last_atomic_base_cache.report()

    def _atomic_base_normalization_map(self):
        if self.cfg.atomic_base_normalization == "none":
            return NormalizationMap.identity(applies_to="A")
        return NormalizationMap(
            name=str(self.cfg.atomic_base_normalization),
            applies_to="A",
            bounds=(-math.inf, math.inf),
            is_affine=False,
            scale=1.0,
            shift=0.0,
        )

    def _atomic_base_cache_key(self, *, x_ij, edge_index, atom_types, channels, normalization, charges=None, aux_tensor_basis=None):
        return AtomicBaseCache.key(
            neighbor_list={
                "edge_index": edge_index,
                "atom_types": atom_types,
                "edge_vectors": x_ij,
                "charges": charges,
                "aux_tensor_basis": aux_tensor_basis,
            },
            site_basis={
                "channels": tuple(channels),
                "possible_types": tuple(int(x) for x in self.cfg.possible_types),
                "charge_mode": str(self.cfg.charge_mode),
                "factor_normalization": str(self.cfg.factor_normalization),
                "spherical_normalization": str(self.cfg.spherical_normalization),
            },
            radial_basis=self._radial_convention_metadata(),
            angular_convention=self._angular_convention_metadata(),
            dtype=self.cfg.dtype,
            device=x_ij.device,
            normalization_convention=normalization.metadata(),
        )

    def _record_atomic_base_cache(
        self,
        *,
        x_ij,
        edge_index,
        atom_types,
        channels,
        raw_atomic_base,
        final_atomic_base,
        dAraw_dR=None,
        dAnorm_dAraw=None,
        charges=None,
        aux_tensor_basis=None,
    ):
        normalization = self._atomic_base_normalization_map()
        cache_key = self._atomic_base_cache_key(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            normalization=normalization,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        if cache_key in self._atomic_base_cache_seen_keys:
            self._atomic_base_cache_hits += 1
        else:
            self._atomic_base_cache_seen_keys.add(cache_key)
            self._atomic_base_cache_misses += 1
        edge_lengths = torch.linalg.norm(x_ij.to(dtype=self.cfg.dtype), dim=-1)
        edge_directions = x_ij.to(dtype=self.cfg.dtype) / torch.clamp(
            edge_lengths,
            min=torch.as_tensor(1.0e-12, dtype=self.cfg.dtype, device=x_ij.device),
        ).unsqueeze(-1)
        self._last_atomic_base_cache = AtomicBaseCache(
            cache_key=cache_key,
            normalization=normalization,
            neighbor_list={"edge_index": edge_index, "atom_types": atom_types},
            edge_vectors=x_ij,
            edge_lengths=edge_lengths,
            edge_directions=edge_directions,
            A_raw=raw_atomic_base,
            A_normalized=final_atomic_base,
            dAraw_dR=dAraw_dR,
            dAnorm_dAraw=dAnorm_dAraw,
            radial_convention=self._radial_convention_metadata(),
            angular_convention=self._angular_convention_metadata(),
            dtype=str(self.cfg.dtype),
            device=str(x_ij.device),
            hits=self._atomic_base_cache_hits,
            misses=self._atomic_base_cache_misses,
        )
        return self._last_atomic_base_cache

    def _runtime_cache_get(self, family, key):
        if self._evaluation_context is None:
            return None
        return self._evaluation_context.runtime_cache.get(family, key)

    def _runtime_cache_put(self, family, key, value):
        if self._evaluation_context is None:
            return value
        return self._evaluation_context.runtime_cache.put(family, key, value)

    def _cache_key_for_raw_and_final(
        self,
        *,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges=None,
        aux_tensor_basis=None,
    ):
        normalization = self._atomic_base_normalization_map()
        return self._atomic_base_cache_key(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            normalization=normalization,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )

    def _radial_runtime_cache_identity(self):
        return str(self._radial_convention_metadata()["convention_hash"])

    def _radial_convention_metadata(self):
        if hasattr(self.radial_provider, "convention_metadata"):
            metadata = dict(self.radial_provider.convention_metadata(evaluator=self))
            supplied_hash = metadata.pop("convention_hash", None)
            if not metadata or not any(key in metadata for key in ("provider", "kind")):
                raise ValueError("Radial source metadata needs a provider or kind identity.")
            try:
                encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("Radial source metadata must be finite JSON data.") from exc
            computed_hash = hashlib.sha256(encoded).hexdigest()
            if supplied_hash is not None and supplied_hash != computed_hash:
                raise ValueError("Radial source metadata hash does not match its fields.")
            metadata["convention_hash"] = computed_hash
            return metadata
        if isinstance(self.radial_provider, DefaultRadialBasisProvider):
            metadata = {
                "provider": "DefaultRadialBasisProvider",
                "kind": str(self.cfg.radial_basis),
                "rc": tuple(float(x) for x in self.cfg.rc),
                "lmbda": tuple(float(x) for x in self.cfg.lmbda),
            }
            metadata["convention_hash"] = _metadata_hash(metadata)
            return metadata
        raise ValueError(
            "Custom radial providers must declare stable JSON source metadata."
        )

    def _angular_convention_metadata(self):
        metadata = dict(self.sph.convention_metadata())
        metadata.pop("convention_hash", None)
        metadata["site_spherical_normalization"] = str(
            self.cfg.spherical_normalization
        )
        metadata["site_factor_normalization"] = str(
            self.cfg.factor_normalization
        )
        metadata["site_scale_by_l"] = tuple(
            float(
                self._spherical_factor_scale(
                    ell,
                    device=torch.device("cpu"),
                    dtype=self.cfg.dtype,
                ).item()
            )
            for ell in range(int(self.cfg.lmax) + 1)
        )
        metadata["convention_hash"] = _metadata_hash(metadata)
        return metadata

    def _factor_normalization_enabled(self):
        return str(getattr(self.cfg, "factor_normalization", "none")).strip().lower() == "bounded"

    def _spherical_factor_scale(self, l, *, device, dtype):
        normalization = str(self.cfg.spherical_normalization)
        key = (
            "spherical",
            int(l),
            normalization,
            str(self.cfg.factor_normalization),
            str(device),
            str(dtype),
        )
        cached = self._factor_scale_cache.get(key)
        if cached is None:
            scale = math.sqrt(4.0 * math.pi) if normalization == "pace_y00_one" else 1.0
            if self._factor_normalization_enabled():
                scale *= math.sqrt(4.0 * math.pi / float(2 * int(l) + 1))
            cached = torch.as_tensor(scale, dtype=dtype, device=device)
            self._factor_scale_cache[key] = cached
        return cached

    def _normalize_spherical_values(self, l, values):
        if not self._factor_normalization_enabled() and self.cfg.spherical_normalization == "orthonormal":
            return values
        scale = self._spherical_factor_scale(int(l), device=values.device, dtype=values.real.dtype)
        return values * scale.to(values.dtype)

    def _normalize_spherical_values_with_derivatives(self, l, values, derivatives):
        if not self._factor_normalization_enabled() and self.cfg.spherical_normalization == "orthonormal":
            return values, derivatives
        scale = self._spherical_factor_scale(int(l), device=values.device, dtype=values.real.dtype)
        scale = scale.to(values.dtype)
        return values * scale, derivatives * scale

    def _spherical_values_for_forward(self, l, x_ij, theta, phi):
        radial_kind = str(self.cfg.radial_basis).strip().lower()
        radial_kind = radial_kind.replace("_", "").replace("-", "")
        if (
            self.cfg.spherical_backend == "complex"
            and radial_kind in {
                "pacechebexpcos",
                "pacechebexpcosidentityspline",
            }
        ):
            values = self.sph.basis.cartesian_values(l, x_ij)
            return self._normalize_spherical_values(l, values)
        return self._normalize_spherical_values(
            l,
            self.sph.all_m(l, theta, phi).to(self.cfg.complex_dtype),
        )

    def _radial_factor_scale(self, n, l, *, device, dtype):
        if not self._factor_normalization_enabled():
            return torch.ones((), dtype=dtype, device=device)
        key = ("radial", int(n), int(l), str(device), str(dtype), self._radial_runtime_cache_identity())
        cached = self._factor_scale_cache.get(key)
        if cached is None:
            max_abs = self.radial_provider.max_abs(n=int(n), l=int(l), evaluator=self)
            max_abs = float(max_abs)
            if not math.isfinite(max_abs) or max_abs <= 0.0:
                raise ValueError(f"radial_provider.max_abs must return a positive finite value; got {max_abs!r}")
            cached = torch.as_tensor(1.0 / max_abs, dtype=dtype, device=device)
            self._factor_scale_cache[key] = cached
        return cached

    def _normalize_radial_values(self, n, l, values):
        if not self._factor_normalization_enabled():
            return values
        return values * self._radial_factor_scale(int(n), int(l), device=values.device, dtype=values.dtype)

    def _normalize_radial_values_with_derivative(self, n, l, values, derivatives):
        if not self._factor_normalization_enabled():
            return values, derivatives
        scale = self._radial_factor_scale(int(n), int(l), device=values.device, dtype=values.dtype)
        return values * scale, derivatives * scale

    def _chemical_runtime_cache_identity(self):
        if isinstance(self.chemical_provider, DefaultChemicalBasisProvider):
            return ("default", str(self.cfg.chemical_basis),
                    tuple(int(x) for x in self.cfg.possible_types),
                    getattr(self.cfg, "chemical_embedding", None))
        return ("custom", id(self.chemical_provider))

    def _charge_runtime_cache_identity(self):
        if isinstance(self.charge_provider, DefaultChargeBasisProvider):
            return (
                "default",
                str(self.cfg.charge_mode),
                tuple(float(x) for x in self.cfg.q_min),
                tuple(float(x) for x in self.cfg.q_max),
                str(self.cfg.charge_normalization_mode),
                float(self.cfg.charge_squash_scale),
            )
        return ("custom", id(self.charge_provider))

    def _basis_channel_schedule(self, channels):
        channels_tuple = tuple(channels)
        cached = self._basis_channel_schedule_cache.get(channels_tuple)
        if cached is not None:
            return cached

        grouped = {}
        fallback_entries = []
        for idx, ch in enumerate(channels_tuple):
            if ch.l_aux is None and ch.m_aux is None:
                grouped.setdefault((ch.mu0, ch.mu, ch.kappa0, ch.kappa, ch.n, ch.l), []).append((idx, ch))
            else:
                fallback_entries.append((idx, ch))

        plain_groups = tuple(
            _PlainChannelGroup(
                mu0=mu0,
                mu=mu,
                kappa0=kappa0,
                kappa=kappa,
                n=n,
                l=l,
                channel_indices=tuple(idx for idx, _ in entries),
                m_positions=tuple(ch.m + l for _, ch in entries),
            )
            for (mu0, mu, kappa0, kappa, n, l), entries in grouped.items()
        )
        cached = _BasisChannelSchedule(
            channels=channels_tuple,
            plain_groups=plain_groups,
            fallback_entries=tuple(fallback_entries),
        )
        self._basis_channel_schedule_cache[channels_tuple] = cached
        return cached

    def _packed_plain_edge_table_tensors(self, channel_schedule, device):
        key = (channel_schedule.channels, str(device))
        cached = self._packed_plain_edge_table_cache.get(key)
        if cached is not None:
            return cached
        group_indices = []
        channel_indices = []
        angular_indices = []
        radial_keys = []
        chemical_keys = []
        charge_keys = []
        radial_key_indices = {}
        chemical_key_indices = {}
        charge_key_indices = {}
        radial_group_indices = []
        chemical_group_indices = []
        charge_group_indices = []
        angular_momenta = tuple(
            sorted(
                {
                    int(group.l)
                    for group in channel_schedule.plain_groups
                }
            )
        )
        angular_offsets = {}
        angular_offset = 0
        for angular_momentum in angular_momenta:
            angular_offsets[int(angular_momentum)] = int(
                angular_offset
            )
            angular_offset += 2 * int(angular_momentum) + 1
        for group_id, group in enumerate(channel_schedule.plain_groups):
            radial_key = (int(group.n), int(group.l))
            if radial_key not in radial_key_indices:
                radial_key_indices[radial_key] = len(radial_keys)
                radial_keys.append(radial_key)
            radial_group_indices.append(
                int(radial_key_indices[radial_key])
            )
            chemical_key = (int(group.mu0), int(group.mu))
            if chemical_key not in chemical_key_indices:
                chemical_key_indices[chemical_key] = len(
                    chemical_keys
                )
                chemical_keys.append(chemical_key)
            chemical_group_indices.append(
                int(chemical_key_indices[chemical_key])
            )
            charge_key = (
                int(group.mu0),
                int(group.mu),
                int(group.kappa0),
                int(group.kappa),
            )
            if charge_key not in charge_key_indices:
                charge_key_indices[charge_key] = len(charge_keys)
                charge_keys.append(charge_key)
            charge_group_indices.append(
                int(charge_key_indices[charge_key])
            )
            width = len(group.channel_indices)
            if width <= 0:
                continue
            group_indices.extend([int(group_id)] * width)
            channel_indices.extend(int(idx) for idx in group.channel_indices)
            angular_indices.extend(
                int(angular_offsets[int(group.l)]) +
                int(position)
                for position in group.m_positions
            )
        if group_indices:
            cached = _PackedPlainEdgeTableTensors(
                group_indices=torch.tensor(group_indices, dtype=torch.long, device=device),
                channel_indices=torch.tensor(channel_indices, dtype=torch.long, device=device),
                angular_indices=torch.tensor(angular_indices, dtype=torch.long, device=device),
                radial_keys=tuple(radial_keys),
                radial_group_indices=torch.tensor(radial_group_indices, dtype=torch.long, device=device),
                chemical_keys=tuple(chemical_keys),
                chemical_group_indices=torch.tensor(chemical_group_indices, dtype=torch.long, device=device),
                charge_keys=tuple(charge_keys),
                charge_group_indices=torch.tensor(charge_group_indices, dtype=torch.long, device=device),
                angular_momenta=angular_momenta,
            )
        else:
            cached = _PackedPlainEdgeTableTensors(
                group_indices=torch.empty((0,), dtype=torch.long, device=device),
                channel_indices=torch.empty((0,), dtype=torch.long, device=device),
                angular_indices=torch.empty((0,), dtype=torch.long, device=device),
                radial_keys=tuple(radial_keys),
                radial_group_indices=torch.tensor(radial_group_indices, dtype=torch.long, device=device),
                chemical_keys=tuple(chemical_keys),
                chemical_group_indices=torch.tensor(chemical_group_indices, dtype=torch.long, device=device),
                charge_keys=tuple(charge_keys),
                charge_group_indices=torch.tensor(charge_group_indices, dtype=torch.long, device=device),
                angular_momenta=angular_momenta,
            )
        self._packed_plain_edge_table_cache[key] = cached
        return cached

    def _map_types_to_local(self, types):
        out = torch.full_like(types, -1)
        for raw, local in self._type_to_local.items():
            out[types == raw] = local
        unknown = out < 0
        if bool(torch.any(unknown)):
            values = tuple(
                int(value)
                for value in torch.unique(types[unknown]).detach().cpu().tolist()
            )
            raise ValueError(
                "atom types are absent from possible_types: "
                f"{values}"
            )
        return out

    def _bond_index(self, mu0_edge, mu_edge):
        local_mu0 = self._map_types_to_local(mu0_edge)
        local_mu = self._map_types_to_local(mu_edge)
        return local_mu0 * self.cfg.ntypes + local_mu

    def _prepare_cutoffs(self):
        rc = torch.as_tensor(self.cfg.rc, dtype=self.cfg.dtype)
        lmbda = torch.as_tensor(self.cfg.lmbda, dtype=self.cfg.dtype)
        return rc, lmbda

    def _prepare_cutoffs_on_device(self, device):
        key = (str(device), str(self.cfg.dtype))
        cached = self._cutoff_cache.get(key)
        if cached is None:
            cached = tuple(t.to(device=device) for t in self._prepare_cutoffs())  # type: ignore[assignment]
            self._cutoff_cache[key] = cached
        return cached

    def _spherical_angles(self, x):
        r = torch.linalg.norm(x, dim=-1)
        eps = torch.as_tensor(1e-12, dtype=x.dtype, device=x.device)
        z = x[:, 2] / torch.clamp(r, min=eps)
        z = torch.clamp(z, -1.0 + 1e-12, 1.0 - 1e-12)
        theta = torch.arccos(z)
        phi = torch.atan2(x[:, 1], x[:, 0])
        return r, theta, phi

    def _edge_radius_and_unit(self, x):
        r = torch.linalg.norm(x, dim=-1)
        eps = torch.as_tensor(1e-12, dtype=x.dtype, device=x.device)
        unit = x / torch.clamp(r, min=eps).unsqueeze(-1)
        return r, unit

    def _angles_from_radius_and_unit(self, r, unit):
        del r
        z = torch.clamp(unit[:, 2], -1.0 + 1e-12, 1.0 - 1e-12)
        theta = torch.arccos(z)
        phi = torch.atan2(unit[:, 1], unit[:, 0])
        return theta, phi

    def _default_radial_basis(self, *, n, l, bond_idx, r):
        if str(self.cfg.radial_basis).strip().lower().replace("_", "").replace("-", "") != "chebexpcos":
            return self.radial_provider.evaluate(n=n, l=l, bond_idx=bond_idx, r=r, evaluator=self)
        rc, lmbda = self._prepare_cutoffs_on_device(r.device)
        rc_e = rc[bond_idx]
        lam_e = lmbda[bond_idx]
        r_scale = r / rc_e
        numerator = torch.exp(-lam_e * (r_scale - 1.0)) - 1.0
        denominator = torch.exp(lam_e) - 1.0
        exp_scale = 1.0 - 2.0 * (numerator / denominator)

        if self.cfg.radial_basis != "ChebExpCos":
            raise NotImplementedError(f"Unsupported radial basis: {self.cfg.radial_basis}")

        pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
        if n == 0:
            func = self.cheb(r_scale, n)
        elif n == 1:
            func = 0.5 * (1.0 + torch.cos(pi * r_scale))
        else:
            cheb = self.cheb(exp_scale, n)
            func = 0.25 * (1.0 - cheb) * (1.0 + torch.cos(pi * r_scale))

        return torch.where(r_scale <= 1.0, func, torch.zeros_like(func))

    def _default_radial_basis_many(self, *, n_values, l, bond_idx, r):
        """Evaluate several default radial channels while sharing edge intermediates."""

        n_values = tuple(sorted({int(n) for n in n_values}))
        if not n_values:
            return {}
        rc, lmbda = self._prepare_cutoffs_on_device(r.device)
        rc_e = rc[bond_idx]
        lam_e = lmbda[bond_idx]
        r_scale = r / rc_e

        if self.cfg.radial_basis != "ChebExpCos":
            raise NotImplementedError(f"Unsupported radial basis: {self.cfg.radial_basis}")

        pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
        inside = r_scale <= 1.0
        out = {}
        if 0 in n_values:
            out[0] = torch.where(inside, torch.ones_like(r), torch.zeros_like(r))
        if 1 in n_values:
            cutoff = 0.5 * (1.0 + torch.cos(pi * r_scale))
            out[1] = torch.where(inside, cutoff, torch.zeros_like(cutoff))

        high = tuple(n for n in n_values if n > 1)
        if high:
            numerator = torch.exp(-lam_e * (r_scale - 1.0)) - 1.0
            denominator = torch.exp(lam_e) - 1.0
            exp_scale = 1.0 - 2.0 * (numerator / denominator)
            cutoff_full = 1.0 + torch.cos(pi * r_scale)
            max_n = max(high)
            t_prev = torch.ones_like(exp_scale)
            t_curr = exp_scale
            for n in range(2, max_n + 1):
                t_next = 2.0 * exp_scale * t_curr - t_prev
                if n in high:
                    func = 0.25 * (1.0 - t_next) * cutoff_full
                    out[n] = torch.where(inside, func, torch.zeros_like(func))
                t_prev, t_curr = t_curr, t_next
        return out

    @staticmethod
    def _chebyshev_first_derivative(x, n):
        if int(n) == 0:
            return torch.zeros_like(x)
        if int(n) == 1:
            return torch.ones_like(x)
        u_prev = torch.ones_like(x)
        if int(n) == 2:
            return 2.0 * (2.0 * x)
        u_curr = 2.0 * x
        for _ in range(2, int(n)):
            u_prev, u_curr = u_curr, 2.0 * x * u_curr - u_prev
        return float(n) * u_curr

    def _default_radial_basis_with_derivative(self, *, n, l, bond_idx, r):
        if str(self.cfg.radial_basis).strip().lower().replace("_", "").replace("-", "") != "chebexpcos":
            values = self.radial_provider.evaluate(n=n, l=l, bond_idx=bond_idx, r=r, evaluator=self)
            derivatives = self.radial_provider.derivative(n=n, l=l, bond_idx=bond_idx, r=r, evaluator=self)
            return values, derivatives
        rc, lmbda = self._prepare_cutoffs_on_device(r.device)
        rc_e = rc[bond_idx]
        lam_e = lmbda[bond_idx]
        r_scale = r / rc_e

        if self.cfg.radial_basis != "ChebExpCos":
            raise NotImplementedError(f"Unsupported radial basis: {self.cfg.radial_basis}")

        pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
        if int(n) == 0:
            func = self.cheb(r_scale, int(n))
            deriv = torch.zeros_like(func)
        elif int(n) == 1:
            func = 0.5 * (1.0 + torch.cos(pi * r_scale))
            deriv = -0.5 * pi * torch.sin(pi * r_scale) / rc_e
        else:
            numerator_exp = torch.exp(-lam_e * (r_scale - 1.0))
            numerator = numerator_exp - 1.0
            denominator = torch.exp(lam_e) - 1.0
            exp_scale = 1.0 - 2.0 * (numerator / denominator)
            cheb = self.cheb(exp_scale, int(n))
            dcheb_dexp = self._chebyshev_first_derivative(exp_scale, int(n))
            dexp_dr = (2.0 * lam_e * numerator_exp / denominator) / rc_e
            cutoff = 1.0 + torch.cos(pi * r_scale)
            dcutoff_dr = -pi * torch.sin(pi * r_scale) / rc_e
            func = 0.25 * (1.0 - cheb) * cutoff
            deriv = 0.25 * (-dcheb_dexp * dexp_dr * cutoff + (1.0 - cheb) * dcutoff_dr)

        inside = r_scale <= 1.0
        return torch.where(inside, func, torch.zeros_like(func)), torch.where(inside, deriv, torch.zeros_like(deriv))

    def _radial_basis(self, n, l, bond_idx, r):
        values = self.radial_provider.evaluate(n=int(n), l=int(l), bond_idx=bond_idx, r=r, evaluator=self)
        return self._normalize_radial_values(int(n), int(l), values)

    def _radial_basis_with_derivative(self, n, l, bond_idx, r):
        values, derivatives = self.radial_provider.evaluate_with_derivative(
            n=int(n),
            l=int(l),
            bond_idx=bond_idx,
            r=r,
            evaluator=self,
        )
        return self._normalize_radial_values_with_derivative(int(n), int(l), values, derivatives)

    def _soft_neighbor_weights(self, *, bond_idx, r):
        rc, _ = self._prepare_cutoffs_on_device(r.device)
        rc_e = rc[bond_idx]
        r_scale = r / torch.clamp(rc_e, min=torch.finfo(r.dtype).eps)
        pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
        envelope = 0.5 * (1.0 + torch.cos(pi * torch.clamp(r_scale, 0.0, 1.0)))
        return torch.where(r_scale <= 1.0, envelope, torch.zeros_like(envelope))

    def _soft_neighbor_weights_with_dx(self, *, bond_idx, r, dr_dx):
        rc, _ = self._prepare_cutoffs_on_device(r.device)
        rc_e = rc[bond_idx]
        r_scale = r / torch.clamp(rc_e, min=torch.finfo(r.dtype).eps)
        pi = torch.as_tensor(torch.pi, dtype=r.dtype, device=r.device)
        inside = r_scale <= 1.0
        envelope = 0.5 * (1.0 + torch.cos(pi * torch.clamp(r_scale, 0.0, 1.0)))
        weights = torch.where(inside, envelope, torch.zeros_like(envelope))
        dw_dr = -0.5 * pi * torch.sin(pi * r_scale) / torch.clamp(rc_e, min=torch.finfo(r.dtype).eps)
        dw_dr = torch.where(inside, dw_dr, torch.zeros_like(dw_dr))
        return weights, dw_dr.unsqueeze(-1) * dr_dx

    def _apply_soft_neighbor_normalization(
        self,
        A,
        *,
        centers,
        bond_idx,
        r,
        n_atoms,
        soft_count = None,
    ):
        if soft_count is None:
            weights = self._soft_neighbor_weights(bond_idx=bond_idx, r=r).to(self.cfg.dtype)
            soft_count = self._density_accumulate(
                weights.unsqueeze(-1),
                centers,
                n_atoms,
            ).squeeze(-1)
        epsilon = torch.as_tensor(
            self.cfg.atomic_base_normalization_epsilon,
            dtype=self.cfg.dtype,
            device=A.device,
        )
        denom = soft_count + epsilon
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        return A / safe_denom.to(A.dtype).unsqueeze(-1)

    def compute_soft_neighbor_counts(
        self,
        x_ij,
        edge_index,
        atom_types,
        *,
        per_bond_type = False,
    ):
        """Return cutoff-weighted coordination features used by soft normalization."""

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")
        x_ij = x_ij.to(dtype=self.cfg.dtype)
        atom_types = atom_types.to(device=x_ij.device)
        centers = edge_index[0].to(device=x_ij.device)
        neighs = edge_index[1].to(device=x_ij.device)
        bond_idx = self._bond_index(atom_types[centers], atom_types[neighs])
        r, _, _ = self._spherical_angles(x_ij)
        weights = self._soft_neighbor_weights(bond_idx=bond_idx, r=r).to(self.cfg.dtype)
        n_atoms = int(atom_types.shape[0])
        if not per_bond_type:
            return self._density_accumulate(
                weights.unsqueeze(-1),
                centers,
                n_atoms,
            ).squeeze(-1)
        out = torch.zeros((n_atoms, self.cfg.n_bond_channels), dtype=self.cfg.dtype, device=x_ij.device)
        edge_values = torch.zeros((weights.shape[0], self.cfg.n_bond_channels), dtype=self.cfg.dtype, device=x_ij.device)
        edge_values[torch.arange(weights.shape[0], device=x_ij.device), bond_idx] = weights
        return self._density_accumulate(edge_values, centers, n_atoms)

    def compute_soft_neighbor_edge_weights(
        self,
        x_ij,
        edge_index,
        atom_types,
    ):
        """Return the per-edge weights used by ``compute_soft_neighbor_counts``."""

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")
        x_ij = x_ij.to(dtype=self.cfg.dtype)
        atom_types = atom_types.to(device=x_ij.device)
        centers = edge_index[0].to(device=x_ij.device)
        neighs = edge_index[1].to(device=x_ij.device)
        bond_idx = self._bond_index(atom_types[centers], atom_types[neighs])
        r, _, _ = self._spherical_angles(x_ij)
        return self._soft_neighbor_weights(bond_idx=bond_idx, r=r).to(self.cfg.dtype)

    def _apply_atomic_base_normalization(
        self,
        A,
        *,
        centers,
        bond_idx,
        r,
        n_atoms,
        channels,
        soft_count = None,
    ):
        mode = self.cfg.atomic_base_normalization
        if mode == "soft_neighbor":
            A = self._apply_soft_neighbor_normalization(
                A,
                centers=centers,
                bond_idx=bond_idx,
                r=r,
                n_atoms=n_atoms,
                soft_count=soft_count,
            )
        return A

    def _soft_neighbor_count_jacobian(
        self,
        *,
        centers,
        neighs,
        bond_idx,
        r,
        dr_dx,
        n_atoms,
    ):
        weights, weights_dx = self._soft_neighbor_weights_with_dx(bond_idx=bond_idx, r=r, dr_dx=dr_dx)
        soft_count = self._density_accumulate(
            weights.unsqueeze(-1),
            centers,
            n_atoms,
        ).squeeze(-1)
        count_jacobian = torch.zeros((n_atoms, n_atoms, 3), dtype=self.cfg.dtype, device=r.device)
        for edge in range(int(weights.shape[0])):
            center = int(centers[edge].item())
            neigh = int(neighs[edge].item())
            count_jacobian[center, neigh, :] += weights_dx[edge]
            count_jacobian[center, center, :] -= weights_dx[edge]
        return soft_count, count_jacobian

    def _apply_soft_neighbor_normalization_to_jacobian(
        self,
        A,
        jacobian,
        *,
        centers,
        neighs,
        bond_idx,
        r,
        dr_dx,
        n_atoms,
    ):
        soft_count, count_jacobian = self._soft_neighbor_count_jacobian(
            centers=centers,
            neighs=neighs,
            bond_idx=bond_idx,
            r=r,
            dr_dx=dr_dx,
            n_atoms=n_atoms,
        )
        epsilon = torch.as_tensor(self.cfg.atomic_base_normalization_epsilon, dtype=self.cfg.dtype, device=A.device)
        denom = soft_count + epsilon
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        normalized = A / safe_denom.to(A.dtype).unsqueeze(-1)
        normalized_jacobian = (
            jacobian / safe_denom.to(jacobian.dtype).reshape(n_atoms, 1, 1, 1)
            - A[:, :, None, None]
            * count_jacobian[:, None, :, :].to(jacobian.dtype)
            / safe_denom.to(jacobian.dtype).reshape(n_atoms, 1, 1, 1).pow(2)
        )
        return normalized, normalized_jacobian

    def _apply_atomic_base_normalization_to_jacobian(
        self,
        A,
        jacobian,
        *,
        centers,
        neighs,
        bond_idx,
        r,
        dr_dx,
        n_atoms,
        channels,
    ):
        mode = self.cfg.atomic_base_normalization
        if mode == "soft_neighbor":
            A, jacobian = self._apply_soft_neighbor_normalization_to_jacobian(
                A,
                jacobian,
                centers=centers,
                neighs=neighs,
                bond_idx=bond_idx,
                r=r,
                dr_dx=dr_dx,
                n_atoms=n_atoms,
            )
        return A, jacobian

    def _soft_neighbor_normalization_reverse(
        self,
        A,
        A_adjoint,
        *,
        centers,
        bond_idx,
        r,
        n_atoms,
    ):
        weights = self._soft_neighbor_weights(bond_idx=bond_idx, r=r).to(self.cfg.dtype)
        soft_count = self._density_accumulate(
            weights.unsqueeze(-1),
            centers,
            n_atoms,
        ).squeeze(-1)
        epsilon = torch.as_tensor(self.cfg.atomic_base_normalization_epsilon, dtype=self.cfg.dtype, device=A.device)
        denom = soft_count + epsilon
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        root_adjoint = A_adjoint / safe_denom.to(A_adjoint.dtype).unsqueeze(-1)
        count_adjoint = -(
            A_adjoint * A
        ).sum(dim=1).real / safe_denom.pow(2)
        return root_adjoint, count_adjoint

    def _normalization_reverse(
        self,
        raw_A,
        final_A_adjoint,
        *,
        centers,
        bond_idx,
        r,
        n_atoms,
        channels,
    ):
        mode = self.cfg.atomic_base_normalization
        if mode == "none":
            count_adjoint = torch.zeros((n_atoms,), dtype=self.cfg.dtype, device=raw_A.device)
            return final_A_adjoint, count_adjoint

        after_soft = raw_A
        if mode == "soft_neighbor":
            after_soft = self._apply_soft_neighbor_normalization(
                raw_A,
                centers=centers,
                bond_idx=bond_idx,
                r=r,
                n_atoms=n_atoms,
            )

        count_adjoint = torch.zeros((n_atoms,), dtype=self.cfg.dtype, device=raw_A.device)
        current_adjoint = final_A_adjoint
        if mode == "soft_neighbor":
            current_adjoint, count_adjoint = self._soft_neighbor_normalization_reverse(
                raw_A,
                current_adjoint,
                centers=centers,
                bond_idx=bond_idx,
                r=r,
                n_atoms=n_atoms,
            )
        return current_adjoint, count_adjoint

    def _normalize_charge(self, q, atom_type_local):
        key = (str(q.device), str(self.cfg.dtype))
        cached = self._charge_bound_cache.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(self.cfg.q_min, dtype=self.cfg.dtype, device=q.device),
                torch.as_tensor(self.cfg.q_max, dtype=self.cfg.dtype, device=q.device),
            )
            self._charge_bound_cache[key] = cached
        qmin = cached[0][atom_type_local]
        qmax = cached[1][atom_type_local]
        midpoint = 0.5 * (qmax + qmin)
        half_range = 0.5 * (qmax - qmin)
        scaled = (q - midpoint) / torch.clamp(half_range, min=1e-12)
        if self.cfg.charge_normalization_mode == "linear_clip":
            return torch.clamp(scaled, -1.0, 1.0)
        if self.cfg.charge_normalization_mode == "tanh":
            return torch.tanh(scaled / float(self.cfg.charge_squash_scale))
        raise NotImplementedError(f"Unsupported charge_normalization_mode: {self.cfg.charge_normalization_mode}")

    def _normalize_charge_with_derivative(self, q, atom_type_local):
        key = (str(q.device), str(self.cfg.dtype))
        cached = self._charge_bound_cache.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(self.cfg.q_min, dtype=self.cfg.dtype, device=q.device),
                torch.as_tensor(self.cfg.q_max, dtype=self.cfg.dtype, device=q.device),
            )
            self._charge_bound_cache[key] = cached
        qmin = cached[0][atom_type_local]
        qmax = cached[1][atom_type_local]
        midpoint = 0.5 * (qmax + qmin)
        half_range = torch.clamp(0.5 * (qmax - qmin), min=1e-12)
        scaled = (q - midpoint) / half_range
        if self.cfg.charge_normalization_mode == "linear_clip":
            inside = scaled.abs() <= 1.0
            return torch.clamp(scaled, -1.0, 1.0), torch.where(inside, 1.0 / half_range, torch.zeros_like(q))
        if self.cfg.charge_normalization_mode == "tanh":
            squash = float(self.cfg.charge_squash_scale)
            value = torch.tanh(scaled / squash)
            return value, (1.0 - value.pow(2)) / (squash * half_range)
        raise NotImplementedError(f"Unsupported charge_normalization_mode: {self.cfg.charge_normalization_mode}")

    def _default_charge_basis(self, *, q_i, q_j, mu0_edge, mu_edge, kappa0, kappa):
        if self.cfg.charge_mode == "none":
            return torch.ones_like(q_i)
        if self.cfg.charge_mode != "scalar":
            raise NotImplementedError(f"Unsupported charge mode: {self.cfg.charge_mode}")
        mu0_local = self._map_types_to_local(mu0_edge)
        mu_local = self._map_types_to_local(mu_edge)
        q_i_scaled = self._normalize_charge(q_i, mu0_local)
        q_j_scaled = self._normalize_charge(q_j, mu_local)
        return self.cheb(q_i_scaled, kappa0) * self.cheb(q_j_scaled, kappa)

    def _default_charge_basis_with_derivative(
        self,
        *,
        q_i,
        q_j,
        mu0_edge,
        mu_edge,
        kappa0,
        kappa,
    ):
        if self.cfg.charge_mode == "none":
            zeros = torch.zeros_like(q_i)
            return torch.ones_like(q_i), zeros, zeros
        if self.cfg.charge_mode != "scalar":
            raise NotImplementedError(f"Unsupported charge mode: {self.cfg.charge_mode}")
        mu0_local = self._map_types_to_local(mu0_edge)
        mu_local = self._map_types_to_local(mu_edge)
        q_i_scaled, dq_i_scaled = self._normalize_charge_with_derivative(q_i, mu0_local)
        q_j_scaled, dq_j_scaled = self._normalize_charge_with_derivative(q_j, mu_local)
        center = self.cheb(q_i_scaled, int(kappa0))
        neighbor = self.cheb(q_j_scaled, int(kappa))
        dcenter = self._chebyshev_first_derivative(q_i_scaled, int(kappa0)) * dq_i_scaled
        dneighbor = self._chebyshev_first_derivative(q_j_scaled, int(kappa)) * dq_j_scaled
        return center * neighbor, dcenter * neighbor, center * dneighbor

    def _charge_basis(self, q_i, q_j, mu0_edge, mu_edge, kappa0, kappa):
        return self.charge_provider(
            q_i=q_i,
            q_j=q_j,
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            kappa0=int(kappa0),
            kappa=int(kappa),
            evaluator=self,
        )

    def _charge_basis_with_derivative(
        self,
        q_i,
        q_j,
        mu0_edge,
        mu_edge,
        kappa0,
        kappa,
    ):
        if not isinstance(self.charge_provider, DefaultChargeBasisProvider):
            raise NotImplementedError("Analytic charge derivatives currently require the default charge provider")
        return self._default_charge_basis_with_derivative(
            q_i=q_i,
            q_j=q_j,
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            kappa0=int(kappa0),
            kappa=int(kappa),
        )

    def _default_chemical_basis(self, *, mu0_edge, mu_edge, mu0, mu):
        if self.cfg.chemical_basis == "delta":
            return ((mu0_edge == mu0) & (mu_edge == mu)).to(self.cfg.dtype)
        matrix = self.cfg.chemical_embedding
        if not 0 <= mu < len(matrix[0]):
            raise ValueError("Chemical embedding column is outside the declared matrix.")
        values = torch.zeros_like(mu_edge, dtype=self.cfg.dtype)
        for row, species_type in enumerate(self.cfg.possible_types):
            values = values + (mu_edge == species_type).to(self.cfg.dtype) * matrix[row][mu]
        return (mu0_edge == mu0).to(self.cfg.dtype) * values

    def _chemical_basis(self, mu0_edge, mu_edge, mu0, mu):
        return self.chemical_provider(
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            mu0=int(mu0),
            mu=int(mu),
            evaluator=self,
        )

    def compute_site_basis(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges = None,
        aux_tensor_basis = None,
        runtime_cache = None,
        real_output = False,
    ):
        """Compute the single-site atomic base ``A_i,nlm`` for channels.

        This is the public ACE-layer hook for replacing radial functions,
        chemical embeddings, auxiliary scalar embeddings, or spherical
        harmonics.  ``compute_channels`` is retained as a compatibility alias.
        """
        return self.compute_channels(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            runtime_cache=runtime_cache,
            real_output=real_output,
        )

    def compute_atomic_base(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges = None,
        aux_tensor_basis = None,
        runtime_cache = None,
        real_output = False,
    ):
        """Alias for ``compute_site_basis`` using the paper's ``A_i,nlm`` name."""
        return self.compute_site_basis(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            runtime_cache=runtime_cache,
            real_output=real_output,
        )

    def compute_channel_edges_with_dx(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges = None,
        aux_tensor_basis = None,
        real_output = False,
    ):
        """Return edge-channel values and analytic derivatives with respect to ``x_ij``.

        This is the compact local derivative used by analytic force paths.
        Normalization derivatives are handled after edge aggregation at the
        atomic-base level.
        """

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")

        x_ij = x_ij.to(dtype=self.cfg.dtype)
        atom_types = atom_types.to(device=x_ij.device)
        centers = edge_index[0].to(device=x_ij.device)
        neighs = edge_index[1].to(device=x_ij.device)
        mu0_edge = atom_types[centers]
        mu_edge = atom_types[neighs]
        bond_idx = self._bond_index(mu0_edge, mu_edge)
        r = torch.linalg.norm(x_ij, dim=-1)
        safe_r = torch.clamp(r, min=torch.as_tensor(1.0e-12, dtype=x_ij.dtype, device=x_ij.device))
        dr_dx = x_ij / safe_r.unsqueeze(-1)

        output_dtype = (
            self.cfg.dtype
            if bool(real_output)
            else self.cfg.complex_dtype
        )
        n_edges = int(x_ij.shape[0])
        n_channels = len(channels)
        edge_vals = torch.zeros((n_edges, n_channels), dtype=output_dtype, device=x_ij.device)
        edge_dx = torch.zeros((n_edges, n_channels, 3), dtype=output_dtype, device=x_ij.device)
        edge_dq_center = torch.zeros((n_edges, n_channels), dtype=output_dtype, device=x_ij.device)
        edge_dq_neighbor = torch.zeros((n_edges, n_channels), dtype=output_dtype, device=x_ij.device)

        if charges is None:
            charges = torch.zeros(int(atom_types.shape[0]), dtype=self.cfg.dtype, device=x_ij.device)
        else:
            charges = charges.to(dtype=self.cfg.dtype, device=x_ij.device)
        q_i = charges[centers]
        q_j = charges[neighs]

        Y_cache = {}
        R_cache = {}
        Q_cache = {}
        C_cache = {}


        channel_schedule = self._basis_channel_schedule(channels)
        if bool(real_output) and channel_schedule.fallback_entries:
            raise ValueError(
                "real_output currently requires plain SiteBasis channels"
            )
        native_radial_cache, native_angular_cache = (
            self._native_source_derivative_tables(
                x_ij=x_ij,
                bond_idx=bond_idx,
                r=r,
                channel_schedule=channel_schedule,
            )
        )
        R_cache.update(native_radial_cache)
        Y_cache.update(native_angular_cache)
        plain_product = self._native_plain_source_product(
            x_ij=x_ij,
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            q_i=q_i,
            q_j=q_j,
            bond_idx=bond_idx,
            r=r,
            dr_dx=dr_dx,
            channel_schedule=channel_schedule,
            radial_cache=R_cache,
            angular_cache=Y_cache,
            chemical_cache=C_cache,
            charge_cache=Q_cache,
            target_dtype=output_dtype,
        )
        if plain_product is not None:
            (
                edge_vals,
                edge_dx,
                edge_dq_center,
                edge_dq_neighbor,
            ) = plain_product
        else:
            for group in channel_schedule.plain_groups:
                mu0, mu, kappa0, kappa, n, l = group.mu0, group.mu, group.kappa0, group.kappa, group.n, group.l
                chem_key = (mu0, mu)
                if chem_key not in C_cache:
                    C_cache[chem_key] = self._chemical_basis(mu0_edge, mu_edge, mu0, mu)
                chem = C_cache[chem_key].to(output_dtype)

                q_key = (mu0, mu, kappa0, kappa)
                if q_key not in Q_cache:
                    Q_cache[q_key] = self._charge_basis_with_derivative(q_i, q_j, mu0_edge, mu_edge, kappa0, kappa)
                qf_raw, dqf_i_raw, dqf_j_raw = Q_cache[q_key]
                qf = qf_raw.to(output_dtype)
                dqf_i = dqf_i_raw.to(output_dtype)
                dqf_j = dqf_j_raw.to(output_dtype)

                r_key = (n, l)
                if r_key not in R_cache:
                    R_cache[r_key] = self._radial_basis_with_derivative(n, l, bond_idx, r)
                rf, drf_dr = R_cache[r_key]
                rf_c = rf.to(output_dtype)
                drf_dx = (
                    drf_dr.unsqueeze(-1).to(output_dtype)
                    * dr_dx.to(output_dtype)
                )

                if l not in Y_cache:
                    y_vals, y_derivs = self.sph.all_m_cartesian_with_derivatives(l, x_ij)
                    Y_cache[l] = self._normalize_spherical_values_with_derivatives(l, y_vals, y_derivs)
                y_all, dy_dx_all = Y_cache[l]
                group_tensors = group.tensors(x_ij.device)
                y = (
                    y_all.index_select(
                        0,
                        group_tensors.m_positions,
                    )
                    .transpose(0, 1)
                    .to(output_dtype)
                )
                dy = (
                    dy_dx_all.index_select(
                        0,
                        group_tensors.m_positions,
                    )
                    .permute(1, 0, 2)
                    .to(output_dtype)
                )

                prefactor = (chem * qf).unsqueeze(-1)
                values = prefactor * rf_c.unsqueeze(-1) * y
                deriv = prefactor.unsqueeze(-1) * (
                    drf_dx[:, None, :] * y[:, :, None]
                    + rf_c[:, None, None] * dy
                )
                charge_common = chem.unsqueeze(-1) * rf_c.unsqueeze(-1) * y

                edge_vals.index_copy_(1, group_tensors.indices, values)
                edge_dx.index_copy_(1, group_tensors.indices, deriv)
                edge_dq_center.index_copy_(1, group_tensors.indices, charge_common * dqf_i.unsqueeze(-1))
                edge_dq_neighbor.index_copy_(1, group_tensors.indices, charge_common * dqf_j.unsqueeze(-1))


        for idx, ch in channel_schedule.fallback_entries:
            chem_key = (ch.mu0, ch.mu)
            if chem_key not in C_cache:
                C_cache[chem_key] = self._chemical_basis(mu0_edge, mu_edge, ch.mu0, ch.mu)
            chem = C_cache[chem_key].to(self.cfg.complex_dtype)

            q_key = (ch.mu0, ch.mu, ch.kappa0, ch.kappa)
            if q_key not in Q_cache:
                Q_cache[q_key] = self._charge_basis_with_derivative(q_i, q_j, mu0_edge, mu_edge, ch.kappa0, ch.kappa)
            qf_raw, dqf_i_raw, dqf_j_raw = Q_cache[q_key]
            qf = qf_raw.to(self.cfg.complex_dtype)
            dqf_i = dqf_i_raw.to(self.cfg.complex_dtype)
            dqf_j = dqf_j_raw.to(self.cfg.complex_dtype)

            r_key = (ch.n, ch.l)
            if r_key not in R_cache:
                R_cache[r_key] = self._radial_basis_with_derivative(ch.n, ch.l, bond_idx, r)
            rf, drf_dr = R_cache[r_key]
            rf_c = rf.to(self.cfg.complex_dtype)
            drf_dx = drf_dr.unsqueeze(-1).to(self.cfg.complex_dtype) * dr_dx.to(self.cfg.complex_dtype)

            if ch.l not in Y_cache:
                y_vals, y_derivs = self.sph.all_m_cartesian_with_derivatives(ch.l, x_ij)
                Y_cache[ch.l] = self._normalize_spherical_values_with_derivatives(ch.l, y_vals, y_derivs)
            y_all, dy_dx_all = Y_cache[ch.l]
            ang = y_all[ch.m + ch.l].to(self.cfg.complex_dtype)
            dang_dx = dy_dx_all[ch.m + ch.l].to(self.cfg.complex_dtype)

            prefactor = chem * qf
            value = prefactor * rf_c * ang
            deriv = prefactor.unsqueeze(-1) * (drf_dx * ang.unsqueeze(-1) + rf_c.unsqueeze(-1) * dang_dx)
            charge_common = chem * rf_c * ang
            deriv_q_i = charge_common * dqf_i
            deriv_q_j = charge_common * dqf_j

            if ch.l_aux is not None or ch.m_aux is not None:
                if aux_tensor_basis is None:
                    raise ValueError("Channel requests auxiliary tensor/angular indices, but aux_tensor_basis is None")
                aux = aux_tensor_basis[(ch.l_aux, ch.m_aux)].to(device=x_ij.device, dtype=self.cfg.complex_dtype)
                value = value * aux
                deriv = deriv * aux.unsqueeze(-1)
                if "deriv_q_i" in locals():
                    deriv_q_i = deriv_q_i * aux
                    deriv_q_j = deriv_q_j * aux

            edge_vals[:, idx] = value
            edge_dx[:, idx, :] = deriv
            if "deriv_q_i" in locals():
                edge_dq_center[:, idx] = deriv_q_i
                edge_dq_neighbor[:, idx] = deriv_q_j
                del deriv_q_i, deriv_q_j

        if self.cfg.atomic_base_normalization == "soft_neighbor":
            weights, weights_dx = self._soft_neighbor_weights_with_dx(
                bond_idx=bond_idx,
                r=r,
                dr_dx=dr_dx,
            )
            weights_c = weights.to(edge_vals.dtype)
            weights_dx_c = weights_dx.to(edge_dx.dtype)
            unweighted_edge_vals = edge_vals
            edge_vals = unweighted_edge_vals * weights_c.unsqueeze(-1)
            edge_dx = (
                edge_dx * weights_c.reshape(-1, 1, 1)
                + unweighted_edge_vals.unsqueeze(-1) * weights_dx_c.unsqueeze(1)
            )
            edge_dq_center = edge_dq_center * weights_c.unsqueeze(-1)
            edge_dq_neighbor = edge_dq_neighbor * weights_c.unsqueeze(-1)

        self._last_edge_charge_derivatives = (edge_dq_center, edge_dq_neighbor)
        return list(channels), edge_vals, edge_dx

    def compute_channels_with_position_jacobian(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges = None,
        aux_tensor_basis = None,
    ):
        """Compute atomic-base values and analytic position Jacobians.

        Returns ``A`` with shape ``[N, C]`` and ``dA/dR`` with shape
        ``[N, C, N, 3]``. This dense form is primarily for validation and
        LAMMPS-style exports; production force paths should prefer compact edge
        derivatives plus VJP accumulation.
        """

        channels_list, edge_vals, edge_dx = self.compute_channel_edges_with_dx(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        edge_dq_center, edge_dq_neighbor = getattr(
            self,
            "_last_edge_charge_derivatives",
            (
                torch.zeros((edge_vals.shape[0], edge_vals.shape[1]), dtype=self.cfg.complex_dtype, device=edge_vals.device),
                torch.zeros((edge_vals.shape[0], edge_vals.shape[1]), dtype=self.cfg.complex_dtype, device=edge_vals.device),
            ),
        )
        atom_types = atom_types.to(device=edge_vals.device)
        centers = edge_index[0].to(device=edge_vals.device)
        neighs = edge_index[1].to(device=edge_vals.device)
        n_atoms = int(atom_types.shape[0])
        atomic_base = self._density_accumulate(
            edge_vals,
            centers,
            n_atoms,
        )
        jacobian = torch.zeros(
            (n_atoms, len(channels_list), n_atoms, 3),
            dtype=self.cfg.complex_dtype,
            device=edge_vals.device,
        )
        for edge in range(int(edge_vals.shape[0])):
            center = int(centers[edge].item())
            neigh = int(neighs[edge].item())
            jacobian[center, :, neigh, :] += edge_dx[edge]
            jacobian[center, :, center, :] -= edge_dx[edge]
        raw_jacobian = jacobian
        if self.cfg.atomic_base_normalization != "none":
            r = torch.linalg.norm(x_ij.to(dtype=self.cfg.dtype, device=edge_vals.device), dim=-1)
            safe_r = torch.clamp(r, min=torch.as_tensor(1.0e-12, dtype=self.cfg.dtype, device=edge_vals.device))
            dr_dx = x_ij.to(dtype=self.cfg.dtype, device=edge_vals.device) / safe_r.unsqueeze(-1)
            mu0_edge = atom_types[centers]
            mu_edge = atom_types[neighs]
            bond_idx = self._bond_index(mu0_edge, mu_edge)
            atomic_base, jacobian = self._apply_atomic_base_normalization_to_jacobian(
                atomic_base,
                jacobian,
                centers=centers,
                neighs=neighs,
                bond_idx=bond_idx,
                r=r,
                dr_dx=dr_dx,
                n_atoms=n_atoms,
                channels=channels_list,
            )
        self._record_atomic_base_cache(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels_list,
            raw_atomic_base=atomic_base if self.cfg.atomic_base_normalization == "none" else self._density_accumulate(edge_vals, centers, n_atoms),
            final_atomic_base=atomic_base,
            dAraw_dR=raw_jacobian,
            dAnorm_dAraw=torch.ones_like(atomic_base) if self.cfg.atomic_base_normalization == "none" else None,
        )
        return channels_list, atomic_base, jacobian

    def position_vjp_from_channel_adjoint(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        channel_adjoint,
        charges = None,
        aux_tensor_basis = None,
    ):
        """Compact VJP from atomic-base channel adjoints to position gradients.

        ``channel_adjoint`` is an adjoint with respect to the final, normalized
        atomic base ``A``. The returned position gradient has shape ``[N, 3]``
        and avoids materializing the dense ``[N, C, N, 3]`` Jacobian.
        """

        channels_list, final_A, record = self.compute_channels_with_vjp_record(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        position_grad = self.position_vjp_from_record(record, channel_adjoint)
        return channels_list, final_A, position_grad

    def compute_channels_with_vjp_record(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges = None,
        aux_tensor_basis = None,
    ):
        """Compute atomic-base values and reusable compact-VJP intermediates."""

        channels_tuple = tuple(channels)
        force_cache_key = self._cache_key_for_raw_and_final(
            x_ij=x_ij.to(dtype=self.cfg.dtype),
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels_tuple,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        cached_force = self._runtime_cache_get("force_vjp", force_cache_key)
        if cached_force is not None:
            return cached_force

        channels_list, edge_vals, edge_dx = self.compute_channel_edges_with_dx(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels_tuple,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        edge_dq_center, edge_dq_neighbor = getattr(
            self,
            "_last_edge_charge_derivatives",
            (
                torch.zeros((edge_vals.shape[0], edge_vals.shape[1]), dtype=self.cfg.complex_dtype, device=edge_vals.device),
                torch.zeros((edge_vals.shape[0], edge_vals.shape[1]), dtype=self.cfg.complex_dtype, device=edge_vals.device),
            ),
        )
        atom_types = atom_types.to(device=edge_vals.device)
        centers = edge_index[0].to(device=edge_vals.device)
        neighs = edge_index[1].to(device=edge_vals.device)
        n_atoms = int(atom_types.shape[0])
        r = torch.linalg.norm(x_ij.to(dtype=self.cfg.dtype, device=edge_vals.device), dim=-1)
        safe_r = torch.clamp(r, min=torch.as_tensor(1.0e-12, dtype=self.cfg.dtype, device=edge_vals.device))
        dr_dx = x_ij.to(dtype=self.cfg.dtype, device=edge_vals.device) / safe_r.unsqueeze(-1)
        mu0_edge = atom_types[centers]
        mu_edge = atom_types[neighs]
        bond_idx = self._bond_index(mu0_edge, mu_edge)
        # Use the same value-forward path as ordinary descriptor evaluation.
        # This matters for block normalization with tiny sigma: tiny differences
        # between angular value kernels can be strongly amplified for near-zero
        # partial m-blocks.
        _, raw_A, final_A = self.compute_channels_raw_and_final(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels_list,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )

        record = SiteBasisVJPRecord(
            channels=tuple(channels_list),
            raw_atomic_base=raw_A,
            final_atomic_base=final_A,
            edge_dx=edge_dx,
            edge_dq_center=edge_dq_center,
            edge_dq_neighbor=edge_dq_neighbor,
            centers=centers,
            neighs=neighs,
            bond_idx=bond_idx,
            r=r,
            dr_dx=dr_dx,
            n_atoms=n_atoms,
        )
        result = (channels_list, final_A, record)
        self._runtime_cache_put("force_vjp", force_cache_key, result)
        return result

    def compute_channels_raw_and_final(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges = None,
        aux_tensor_basis = None,
    ):
        """Compute raw and normalized atomic-base values without derivatives.

        This is the forward half of the streaming analytic-force path. It avoids
        constructing edge derivative tensors; those are streamed later after
        descriptor/product adjoints are known.
        """

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")

        x_ij = x_ij.to(dtype=self.cfg.dtype)
        atom_types = atom_types.to(device=x_ij.device)
        centers = edge_index[0].to(device=x_ij.device)
        neighs = edge_index[1].to(device=x_ij.device)
        mu0_edge = atom_types[centers]
        mu_edge = atom_types[neighs]
        bond_idx = self._bond_index(mu0_edge, mu_edge)

        r, theta, phi = self._spherical_angles(x_ij)
        n_atoms = int(atom_types.shape[0])
        n_channels = len(channels)

        if charges is None:
            charges = torch.zeros(n_atoms, dtype=self.cfg.dtype, device=x_ij.device)
        else:
            charges = charges.to(dtype=self.cfg.dtype, device=x_ij.device)
        q_i = charges[centers]
        q_j = charges[neighs]

        runtime_cache_key = self._cache_key_for_raw_and_final(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        cached_atomic_base = self._runtime_cache_get("atomic_base", runtime_cache_key)
        if cached_atomic_base is not None:
            self._atomic_base_cache_seen_keys.add(runtime_cache_key)
            self._atomic_base_cache_hits += 1
            self._last_atomic_base_cache = cached_atomic_base.with_stats(
                hits=self._atomic_base_cache_hits,
                misses=self._atomic_base_cache_misses,
            )
            return list(channels), self._last_atomic_base_cache.A_raw, self._last_atomic_base_cache.A_normalized

        Y_cache = {}
        R_cache = {}
        Q_cache = {}
        C_cache = {}


        channel_schedule = self._basis_channel_schedule(channels)
        soft_neighbor_edge_weights = None
        if self.cfg.atomic_base_normalization == "soft_neighbor":
            soft_neighbor_edge_weights = self._soft_neighbor_weights(
                bond_idx=bond_idx,
                r=r,
            )
        real_plain_forward = bool(
            self.cfg.spherical_backend == "real"
            and aux_tensor_basis is None
            and not channel_schedule.fallback_entries
        )
        if real_plain_forward:
            edge_vals_real = torch.zeros((x_ij.shape[0], n_channels), dtype=self.cfg.dtype, device=x_ij.device)
            for group in channel_schedule.plain_groups:
                mu0, mu, kappa0, kappa, n, l = group.mu0, group.mu, group.kappa0, group.kappa, group.n, group.l
                chem_key = (mu0, mu)
                if chem_key not in C_cache:
                    C_cache[chem_key] = self._chemical_basis(mu0_edge, mu_edge, mu0, mu)
                chem = C_cache[chem_key].to(self.cfg.dtype)

                q_key = (mu0, mu, kappa0, kappa)
                if q_key not in Q_cache:
                    Q_cache[q_key] = self._charge_basis(q_i, q_j, mu0_edge, mu_edge, kappa0, kappa)
                qf = Q_cache[q_key].to(self.cfg.dtype)

                r_key = (n, l)
                if r_key not in R_cache:
                    R_cache[r_key] = self._radial_basis(n, l, bond_idx, r)
                rf = R_cache[r_key].to(self.cfg.dtype)

                if l not in Y_cache:
                    Y_cache[l] = self._normalize_spherical_values(
                        l,
                        self.sph.all_m(l, theta, phi).real.to(self.cfg.dtype),
                    )
                group_tensors = group.tensors(x_ij.device)
                y = Y_cache[l].real.index_select(0, group_tensors.m_positions).transpose(0, 1)
                values = (chem * qf * rf).unsqueeze(-1) * y
                if soft_neighbor_edge_weights is not None:
                    values = values * soft_neighbor_edge_weights.to(self.cfg.dtype).unsqueeze(-1)
                edge_vals_real.index_copy_(1, group_tensors.indices, values)

            raw_A = self._density_accumulate(
                edge_vals_real,
                centers,
                n_atoms,
            )
            final_A = raw_A
            if self.cfg.atomic_base_normalization != "none":
                final_A = self._apply_atomic_base_normalization(
                    raw_A,
                    centers=centers,
                    bond_idx=bond_idx,
                    r=r,
                    n_atoms=n_atoms,
                    channels=channels,
                )
            self._record_atomic_base_cache(
                x_ij=x_ij,
                edge_index=edge_index,
                atom_types=atom_types,
                channels=channels,
                raw_atomic_base=raw_A,
                final_atomic_base=final_A,
                dAnorm_dAraw=torch.ones_like(raw_A) if self.cfg.atomic_base_normalization == "none" else None,
                charges=charges,
                aux_tensor_basis=aux_tensor_basis,
            )
            self._runtime_cache_put("atomic_base", runtime_cache_key, self._last_atomic_base_cache)
            return list(channels), raw_A, final_A

        edge_vals = torch.zeros((x_ij.shape[0], n_channels), dtype=self.cfg.complex_dtype, device=x_ij.device)
        for group in channel_schedule.plain_groups:
            mu0, mu, kappa0, kappa, n, l = group.mu0, group.mu, group.kappa0, group.kappa, group.n, group.l
            chem_key = (mu0, mu)
            if chem_key not in C_cache:
                C_cache[chem_key] = self._chemical_basis(mu0_edge, mu_edge, mu0, mu)
            chem = C_cache[chem_key].to(self.cfg.complex_dtype)

            q_key = (mu0, mu, kappa0, kappa)
            if q_key not in Q_cache:
                Q_cache[q_key] = self._charge_basis(q_i, q_j, mu0_edge, mu_edge, kappa0, kappa)
            qf = Q_cache[q_key].to(self.cfg.complex_dtype)

            r_key = (n, l)
            if r_key not in R_cache:
                R_cache[r_key] = self._radial_basis(n, l, bond_idx, r)
            rf = R_cache[r_key].to(self.cfg.complex_dtype)

            if l not in Y_cache:
                Y_cache[l] = self._spherical_values_for_forward(
                    l,
                    x_ij,
                    theta,
                    phi,
                )
            group_tensors = group.tensors(x_ij.device)
            y = Y_cache[l].index_select(0, group_tensors.m_positions).transpose(0, 1)
            values = (chem * qf * rf).unsqueeze(-1) * y
            if soft_neighbor_edge_weights is not None:
                values = values * soft_neighbor_edge_weights.to(self.cfg.complex_dtype).unsqueeze(-1)
            edge_vals.index_copy_(1, group_tensors.indices, values)


        for idx, ch in channel_schedule.fallback_entries:

            chem_key = (ch.mu0, ch.mu)
            if chem_key not in C_cache:
                C_cache[chem_key] = self._chemical_basis(mu0_edge, mu_edge, ch.mu0, ch.mu)
            chem = C_cache[chem_key]

            q_key = (ch.mu0, ch.mu, ch.kappa0, ch.kappa)
            if q_key not in Q_cache:
                Q_cache[q_key] = self._charge_basis(q_i, q_j, mu0_edge, mu_edge, ch.kappa0, ch.kappa)
            qf = Q_cache[q_key]

            r_key = (ch.n, ch.l)
            if r_key not in R_cache:
                R_cache[r_key] = self._radial_basis(ch.n, ch.l, bond_idx, r)
            rf = R_cache[r_key]

            if ch.l not in Y_cache:
                Y_cache[ch.l] = self._spherical_values_for_forward(
                    ch.l,
                    x_ij,
                    theta,
                    phi,
                )
            value = (
                chem.to(self.cfg.complex_dtype)
                * qf.to(self.cfg.complex_dtype)
                * rf.to(self.cfg.complex_dtype)
                * Y_cache[ch.l][ch.m + ch.l]
            )

            if ch.l_aux is not None or ch.m_aux is not None:
                if aux_tensor_basis is None:
                    raise ValueError("Channel requests auxiliary tensor/angular indices, but aux_tensor_basis is None")
                value = value * aux_tensor_basis[(ch.l_aux, ch.m_aux)].to(self.cfg.complex_dtype)

            if soft_neighbor_edge_weights is not None:
                value = value * soft_neighbor_edge_weights.to(self.cfg.complex_dtype)
            edge_vals[:, idx] = value

        raw_A = self._density_accumulate(edge_vals, centers, n_atoms)
        final_A = raw_A
        if self.cfg.atomic_base_normalization != "none":
            final_A = self._apply_atomic_base_normalization(
                raw_A,
                centers=centers,
                bond_idx=bond_idx,
                r=r,
                n_atoms=n_atoms,
                channels=channels,
            )
        self._record_atomic_base_cache(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            raw_atomic_base=raw_A,
            final_atomic_base=final_A,
            dAnorm_dAraw=torch.ones_like(raw_A) if self.cfg.atomic_base_normalization == "none" else None,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        self._runtime_cache_put("atomic_base", runtime_cache_key, self._last_atomic_base_cache)
        return list(channels), raw_A, final_A

    def _position_vjp_from_raw_channel_adjoint_streaming_real_plain(
        self,
        *,
        x_ij,
        edge_index,
        atom_types,
        channels,
        raw_atomic_base,
        final_channel_adjoint,
        charges,
        centers,
        neighs,
        mu0_edge,
        mu_edge,
        bond_idx,
        r,
        dr_dx,
        n_atoms,
        channel_schedule,
    ):
        """Real-only streaming position VJP for plain real-spherical channels."""

        final_channel_adjoint = final_channel_adjoint.to(device=x_ij.device, dtype=self.cfg.dtype)
        raw_atomic_base = raw_atomic_base.to(device=x_ij.device, dtype=self.cfg.dtype)
        if tuple(final_channel_adjoint.shape) != (n_atoms, len(channels)):
            raise ValueError(
                f"final_channel_adjoint must have shape {(n_atoms, len(channels))}; "
                f"got {tuple(final_channel_adjoint.shape)}"
            )
        raw_adjoint, count_adjoint = self._normalization_reverse(
            raw_atomic_base,
            final_channel_adjoint,
            centers=centers,
            bond_idx=bond_idx,
            r=r,
            n_atoms=n_atoms,
            channels=tuple(channels),
        )
        edge_adjoint = self._density_accumulate_adjoint(
            raw_adjoint,
            centers,
        ).contiguous()

        if charges is None:
            charges = torch.zeros(n_atoms, dtype=self.cfg.dtype, device=x_ij.device)
        else:
            charges = charges.to(dtype=self.cfg.dtype, device=x_ij.device)
        q_i = charges[centers]
        q_j = charges[neighs]

        R_cache, Y_cache = self._native_source_derivative_tables(
            x_ij=x_ij,
            bond_idx=bond_idx,
            r=r,
            channel_schedule=channel_schedule,
        )
        Q_cache = {}
        C_cache = {}
        position_grad = torch.zeros((n_atoms, 3), dtype=self.cfg.dtype, device=x_ij.device)
        used_group_triton = False
        used_torch_group = False
        edge_grad = torch.zeros((x_ij.shape[0], 3), dtype=self.cfg.dtype, device=x_ij.device)
        min_group_elements = int(_ace_env("TRITON_REAL_PLAIN_VJP_MIN_ELEMENTS", "4096"))
        soft_neighbor_edge_weights = None
        soft_neighbor_weights_dx = None
        if self.cfg.atomic_base_normalization == "soft_neighbor":
            soft_neighbor_edge_weights, soft_neighbor_weights_dx = self._soft_neighbor_weights_with_dx(
                bond_idx=bond_idx,
                r=r,
                dr_dx=dr_dx,
            )
            soft_neighbor_edge_weights = soft_neighbor_edge_weights.to(self.cfg.dtype)
            soft_neighbor_weights_dx = soft_neighbor_weights_dx.to(self.cfg.dtype)

        native_adjoint = self._native_plain_source_adjoint(
            x_ij=x_ij,
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            q_i=q_i,
            q_j=q_j,
            bond_idx=bond_idx,
            r=r,
            dr_dx=dr_dx,
            channel_schedule=channel_schedule,
            radial_cache=R_cache,
            angular_cache=Y_cache,
            chemical_cache=C_cache,
            charge_cache=Q_cache,
            edge_adjoint=edge_adjoint,
            edge_weights=soft_neighbor_edge_weights,
            edge_weight_derivatives=soft_neighbor_weights_dx,
        )
        if native_adjoint is not None:
            edge_grad = native_adjoint[0].real.to(self.cfg.dtype)
            if self.cfg.atomic_base_normalization == "soft_neighbor":
                edge_grad = edge_grad + count_adjoint.index_select(
                    0,
                    centers,
                ).unsqueeze(-1) * soft_neighbor_weights_dx
            position_grad, backend = _edge_pair_scatter_maybe_triton(
                edge_grad,
                -edge_grad,
                centers,
                neighs,
                n_atoms,
            )
            self._last_vjp_scatter_backend = (
                f"real_plain_streaming_native_{backend}"
            )
            return position_grad

        for group in channel_schedule.plain_groups:
            mu0, mu, kappa0, kappa, n, l = group.mu0, group.mu, group.kappa0, group.kappa, group.n, group.l
            chem_key = (mu0, mu)
            if chem_key not in C_cache:
                C_cache[chem_key] = self._chemical_basis(mu0_edge, mu_edge, mu0, mu).to(self.cfg.dtype)
            chem = C_cache[chem_key]

            q_key = (mu0, mu, kappa0, kappa)
            if q_key not in Q_cache:
                qf_raw, _, _ = self._charge_basis_with_derivative(q_i, q_j, mu0_edge, mu_edge, kappa0, kappa)
                Q_cache[q_key] = qf_raw.to(self.cfg.dtype)
            qf = Q_cache[q_key]

            r_key = (n, l)
            if r_key not in R_cache:
                R_cache[r_key] = self._radial_basis_with_derivative(n, l, bond_idx, r)
            rf, drf_dr = R_cache[r_key]

            if l not in Y_cache:
                y_all, dy_dx_all = self.sph.all_m_cartesian_with_derivatives(l, x_ij)
                y_all, dy_dx_all = self._normalize_spherical_values_with_derivatives(l, y_all, dy_dx_all)
                Y_cache[l] = (y_all.to(self.cfg.dtype), dy_dx_all.to(self.cfg.dtype))
            y_all, dy_dx_all = Y_cache[l]

            group_tensors = group.tensors(x_ij.device)
            prefactor = chem * qf
            can_group_triton = bool(
                triton is not None
                and _triton_runtime_supported()
                and not torch.is_grad_enabled()
                and x_ij.is_cuda
                and edge_adjoint.is_contiguous()
                and edge_adjoint.dtype in (torch.float32, torch.float64)
                and edge_adjoint.numel() >= min_group_elements
                and len(group.channel_indices) <= 32
            )
            if can_group_triton:
                try:
                    block_edges = 128
                    block_m = 1 << (max(1, len(group.channel_indices)) - 1).bit_length()
                    grid = (triton.cdiv(int(x_ij.shape[0]), block_edges),)
                    y_all_c = y_all.contiguous()
                    dy_dx_all_c = dy_dx_all.contiguous()
                    if soft_neighbor_edge_weights is None:
                        _real_plain_group_vjp_scatter_kernel[grid](
                            edge_adjoint.contiguous(),
                            y_all_c,
                            dy_dx_all_c,
                            rf.to(self.cfg.dtype).contiguous(),
                            drf_dr.to(self.cfg.dtype).contiguous(),
                            dr_dx.to(self.cfg.dtype).contiguous(),
                            prefactor.contiguous(),
                            group_tensors.indices.contiguous(),
                            group_tensors.m_positions.contiguous(),
                            centers.contiguous(),
                            neighs.contiguous(),
                            position_grad,
                            n_edges=int(x_ij.shape[0]),
                            n_channels=int(edge_adjoint.shape[1]),
                            n_atoms=int(n_atoms),
                            group_size=int(len(group.channel_indices)),
                            y_stride_m=int(y_all_c.stride(0)),
                            y_stride_e=int(y_all_c.stride(1)),
                            dy_stride_m=int(dy_dx_all_c.stride(0)),
                            dy_stride_e=int(dy_dx_all_c.stride(1)),
                            dy_stride_c=int(dy_dx_all_c.stride(2)),
                            BLOCK_EDGES=block_edges,
                            BLOCK_M=block_m,
                        )
                    else:
                        _real_plain_group_soft_neighbor_vjp_scatter_kernel[grid](
                            edge_adjoint.contiguous(),
                            y_all_c,
                            dy_dx_all_c,
                            rf.to(self.cfg.dtype).contiguous(),
                            drf_dr.to(self.cfg.dtype).contiguous(),
                            dr_dx.to(self.cfg.dtype).contiguous(),
                            prefactor.contiguous(),
                            soft_neighbor_edge_weights.contiguous(),
                            soft_neighbor_weights_dx.contiguous(),
                            group_tensors.indices.contiguous(),
                            group_tensors.m_positions.contiguous(),
                            centers.contiguous(),
                            neighs.contiguous(),
                            position_grad,
                            n_edges=int(x_ij.shape[0]),
                            n_channels=int(edge_adjoint.shape[1]),
                            n_atoms=int(n_atoms),
                            group_size=int(len(group.channel_indices)),
                            y_stride_m=int(y_all_c.stride(0)),
                            y_stride_e=int(y_all_c.stride(1)),
                            dy_stride_m=int(dy_dx_all_c.stride(0)),
                            dy_stride_e=int(dy_dx_all_c.stride(1)),
                            dy_stride_c=int(dy_dx_all_c.stride(2)),
                            BLOCK_EDGES=block_edges,
                            BLOCK_M=block_m,
                        )
                    used_group_triton = True
                    continue
                except Exception:
                    if os.environ.get("GNE3_DEBUG_TRITON") == "1":
                        raise

            y = y_all.index_select(0, group_tensors.m_positions).transpose(0, 1)
            dy = dy_dx_all.index_select(0, group_tensors.m_positions).permute(1, 0, 2)
            adjoint_group = edge_adjoint.index_select(1, group_tensors.indices)
            radial_adjoint = (adjoint_group * y).sum(dim=1)
            angular_adjoint = (adjoint_group.unsqueeze(-1) * dy).sum(dim=1)
            drf_dx = drf_dr.unsqueeze(-1).to(self.cfg.dtype) * dr_dx.to(self.cfg.dtype)
            unweighted_value = prefactor.unsqueeze(-1) * rf.to(self.cfg.dtype).unsqueeze(-1) * y
            group_edge_grad = prefactor.unsqueeze(-1) * (
                radial_adjoint.unsqueeze(-1) * drf_dx
                + rf.to(self.cfg.dtype).unsqueeze(-1) * angular_adjoint
            )
            if soft_neighbor_edge_weights is not None:
                value_adjoint = (adjoint_group * unweighted_value).sum(dim=1)
                group_edge_grad = (
                    group_edge_grad * soft_neighbor_edge_weights.unsqueeze(-1)
                    + value_adjoint.unsqueeze(-1) * soft_neighbor_weights_dx
                )
            edge_grad = edge_grad + group_edge_grad
            used_torch_group = True

        if self.cfg.atomic_base_normalization == "soft_neighbor":
            edge_grad = edge_grad + count_adjoint.index_select(0, centers).unsqueeze(-1) * soft_neighbor_weights_dx
            used_torch_group = True

        if used_torch_group:
            scattered_grad, backend = _edge_pair_scatter_maybe_triton(edge_grad, -edge_grad, centers, neighs, n_atoms)
            position_grad = position_grad + scattered_grad
        else:
            backend = "triton_group_scatter" if used_group_triton else "index_add"
        self._last_vjp_scatter_backend = f"real_plain_streaming_{backend}"
        return position_grad

    def position_vjp_from_raw_channel_adjoint_streaming(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        raw_atomic_base,
        final_channel_adjoint,
        charges = None,
        aux_tensor_basis = None,
        return_strain_derivative = False,
    ):
        """Stream basis derivatives into forces without storing ``edge_dx``.

        The input adjoint is with respect to the final normalized atomic base.
        This method reverses normalization, then recomputes one-bond basis
        derivatives channel-by-channel and immediately accumulates edge force
        contributions.
        """

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")

        x_ij = x_ij.to(dtype=self.cfg.dtype)
        atom_types = atom_types.to(device=x_ij.device)
        centers = edge_index[0].to(device=x_ij.device)
        neighs = edge_index[1].to(device=x_ij.device)
        mu0_edge = atom_types[centers]
        mu_edge = atom_types[neighs]
        bond_idx = self._bond_index(mu0_edge, mu_edge)
        r = torch.linalg.norm(x_ij, dim=-1)
        safe_r = torch.clamp(r, min=torch.as_tensor(1.0e-12, dtype=x_ij.dtype, device=x_ij.device))
        dr_dx = x_ij / safe_r.unsqueeze(-1)
        n_atoms = int(atom_types.shape[0])

        channel_schedule = self._basis_channel_schedule(channels)
        real_plain_streaming = bool(
            self.cfg.spherical_backend == "real"
            and not torch.is_complex(final_channel_adjoint)
            and not torch.is_complex(raw_atomic_base)
            and aux_tensor_basis is None
            and not channel_schedule.fallback_entries
            and not return_strain_derivative
        )
        if real_plain_streaming:
            return self._position_vjp_from_raw_channel_adjoint_streaming_real_plain(
                x_ij=x_ij,
                edge_index=edge_index,
                atom_types=atom_types,
                channels=channels,
                raw_atomic_base=raw_atomic_base,
                final_channel_adjoint=final_channel_adjoint,
                charges=charges,
                centers=centers,
                neighs=neighs,
                mu0_edge=mu0_edge,
                mu_edge=mu_edge,
                bond_idx=bond_idx,
                r=r,
                dr_dx=dr_dx,
                n_atoms=n_atoms,
                channel_schedule=channel_schedule,
            )

        final_channel_adjoint = final_channel_adjoint.to(device=x_ij.device, dtype=self.cfg.complex_dtype)
        if tuple(final_channel_adjoint.shape) != (n_atoms, len(channels)):
            raise ValueError(
                f"final_channel_adjoint must have shape {(n_atoms, len(channels))}; "
                f"got {tuple(final_channel_adjoint.shape)}"
            )
        raw_adjoint, count_adjoint = self._normalization_reverse(
            raw_atomic_base.to(device=x_ij.device, dtype=self.cfg.complex_dtype),
            final_channel_adjoint,
            centers=centers,
            bond_idx=bond_idx,
            r=r,
            n_atoms=n_atoms,
            channels=tuple(channels),
        )
        edge_adjoint = self._density_accumulate_adjoint(
            raw_adjoint,
            centers,
        )
        edge_grad = torch.zeros((x_ij.shape[0], 3), dtype=self.cfg.dtype, device=x_ij.device)

        if charges is None:
            charges = torch.zeros(n_atoms, dtype=self.cfg.dtype, device=x_ij.device)
        else:
            charges = charges.to(dtype=self.cfg.dtype, device=x_ij.device)
        q_i = charges[centers]
        q_j = charges[neighs]

        R_cache, Y_cache = self._native_source_derivative_tables(
            x_ij=x_ij,
            bond_idx=bond_idx,
            r=r,
            channel_schedule=channel_schedule,
        )
        Q_cache = {}
        C_cache = {}


        soft_neighbor_edge_weights = None
        soft_neighbor_weights_dx = None
        if self.cfg.atomic_base_normalization == "soft_neighbor":
            soft_neighbor_edge_weights, soft_neighbor_weights_dx = self._soft_neighbor_weights_with_dx(
                bond_idx=bond_idx,
                r=r,
                dr_dx=dr_dx,
            )
            soft_neighbor_edge_weights = soft_neighbor_edge_weights.to(self.cfg.complex_dtype)
            soft_neighbor_weights_dx = soft_neighbor_weights_dx.to(self.cfg.complex_dtype)

        native_adjoint = self._native_plain_source_adjoint(
            x_ij=x_ij,
            mu0_edge=mu0_edge,
            mu_edge=mu_edge,
            q_i=q_i,
            q_j=q_j,
            bond_idx=bond_idx,
            r=r,
            dr_dx=dr_dx,
            channel_schedule=channel_schedule,
            radial_cache=R_cache,
            angular_cache=Y_cache,
            chemical_cache=C_cache,
            charge_cache=Q_cache,
            edge_adjoint=edge_adjoint,
            edge_weights=soft_neighbor_edge_weights,
            edge_weight_derivatives=soft_neighbor_weights_dx,
        )
        if native_adjoint is not None:
            edge_grad = native_adjoint[0].real.to(self.cfg.dtype)
            if self.cfg.atomic_base_normalization == "soft_neighbor":
                edge_grad = edge_grad + count_adjoint.index_select(
                    0,
                    centers,
                ).unsqueeze(-1) * soft_neighbor_weights_dx.real.to(
                    self.cfg.dtype
                )
            position_grad, backend = _edge_pair_scatter_maybe_triton(
                edge_grad,
                -edge_grad,
                centers,
                neighs,
                n_atoms,
            )
            self._last_vjp_scatter_backend = (
                f"plain_streaming_native_{backend}"
            )
            if return_strain_derivative:
                strain_derivative = torch.einsum(
                    "ea,eb->ab",
                    edge_grad,
                    x_ij.to(edge_grad.dtype),
                )
                return position_grad, strain_derivative
            return position_grad

        for group in channel_schedule.plain_groups:
            mu0, mu, kappa0, kappa, n, l = group.mu0, group.mu, group.kappa0, group.kappa, group.n, group.l
            chem_key = (mu0, mu)
            if chem_key not in C_cache:
                C_cache[chem_key] = self._chemical_basis(mu0_edge, mu_edge, mu0, mu)
            chem = C_cache[chem_key].to(self.cfg.complex_dtype)

            q_key = (mu0, mu, kappa0, kappa)
            if q_key not in Q_cache:
                Q_cache[q_key] = self._charge_basis_with_derivative(q_i, q_j, mu0_edge, mu_edge, kappa0, kappa)
            qf_raw, _, _ = Q_cache[q_key]
            qf = qf_raw.to(self.cfg.complex_dtype)

            r_key = (n, l)
            if r_key not in R_cache:
                R_cache[r_key] = self._radial_basis_with_derivative(n, l, bond_idx, r)
            rf, drf_dr = R_cache[r_key]
            rf_c = rf.to(self.cfg.complex_dtype)
            drf_dx = drf_dr.unsqueeze(-1).to(self.cfg.complex_dtype) * dr_dx.to(self.cfg.complex_dtype)

            if l not in Y_cache:
                y_vals, y_derivs = self.sph.all_m_cartesian_with_derivatives(l, x_ij)
                Y_cache[l] = self._normalize_spherical_values_with_derivatives(l, y_vals, y_derivs)
            y_all, dy_dx_all = Y_cache[l]

            group_tensors = group.tensors(x_ij.device)
            y = y_all.index_select(0, group_tensors.m_positions).transpose(0, 1).to(self.cfg.complex_dtype)
            dy = dy_dx_all.index_select(0, group_tensors.m_positions).permute(1, 0, 2).to(self.cfg.complex_dtype)

            prefactor = (chem * qf).unsqueeze(-1)
            values = prefactor * rf_c.unsqueeze(-1) * y
            deriv = prefactor.unsqueeze(-1) * (
                drf_dx[:, None, :] * y[:, :, None]
                + rf_c[:, None, None] * dy
            )
            if soft_neighbor_edge_weights is not None:
                deriv = (
                    deriv * soft_neighbor_edge_weights.reshape(-1, 1, 1)
                    + values.unsqueeze(-1) * soft_neighbor_weights_dx.unsqueeze(1)
                )
            adjoint_group = edge_adjoint.index_select(1, group_tensors.indices)
            edge_grad = edge_grad + (adjoint_group.unsqueeze(-1) * deriv).sum(dim=1).real


        for idx, ch in channel_schedule.fallback_entries:
            chem_key = (ch.mu0, ch.mu)
            if chem_key not in C_cache:
                C_cache[chem_key] = self._chemical_basis(mu0_edge, mu_edge, ch.mu0, ch.mu)
            chem = C_cache[chem_key].to(self.cfg.complex_dtype)

            q_key = (ch.mu0, ch.mu, ch.kappa0, ch.kappa)
            if q_key not in Q_cache:
                Q_cache[q_key] = self._charge_basis_with_derivative(q_i, q_j, mu0_edge, mu_edge, ch.kappa0, ch.kappa)
            qf_raw, _, _ = Q_cache[q_key]
            qf = qf_raw.to(self.cfg.complex_dtype)

            r_key = (ch.n, ch.l)
            if r_key not in R_cache:
                R_cache[r_key] = self._radial_basis_with_derivative(ch.n, ch.l, bond_idx, r)
            rf, drf_dr = R_cache[r_key]
            rf_c = rf.to(self.cfg.complex_dtype)
            drf_dx = drf_dr.unsqueeze(-1).to(self.cfg.complex_dtype) * dr_dx.to(self.cfg.complex_dtype)

            if ch.l not in Y_cache:
                y_vals, y_derivs = self.sph.all_m_cartesian_with_derivatives(ch.l, x_ij)
                Y_cache[ch.l] = self._normalize_spherical_values_with_derivatives(ch.l, y_vals, y_derivs)
            y_all, dy_dx_all = Y_cache[ch.l]
            ang = y_all[ch.m + ch.l].to(self.cfg.complex_dtype)
            dang_dx = dy_dx_all[ch.m + ch.l].to(self.cfg.complex_dtype)

            prefactor = chem * qf
            value = prefactor * rf_c * ang
            deriv = prefactor.unsqueeze(-1) * (drf_dx * ang.unsqueeze(-1) + rf_c.unsqueeze(-1) * dang_dx)

            if ch.l_aux is not None or ch.m_aux is not None:
                if aux_tensor_basis is None:
                    raise ValueError("Channel requests auxiliary tensor/angular indices, but aux_tensor_basis is None")
                aux = aux_tensor_basis[(ch.l_aux, ch.m_aux)].to(device=x_ij.device, dtype=self.cfg.complex_dtype)
                value = value * aux
                deriv = deriv * aux.unsqueeze(-1)

            if soft_neighbor_edge_weights is not None:
                deriv = deriv * soft_neighbor_edge_weights.unsqueeze(-1) + value.unsqueeze(-1) * soft_neighbor_weights_dx

            edge_grad = edge_grad + (edge_adjoint[:, idx].unsqueeze(-1) * deriv).real

        if self.cfg.atomic_base_normalization == "soft_neighbor":
            _, weights_dx = self._soft_neighbor_weights_with_dx(
                bond_idx=bond_idx,
                r=r,
                dr_dx=dr_dx,
            )
            edge_grad = edge_grad + count_adjoint.index_select(0, centers).unsqueeze(-1) * weights_dx

        position_grad, backend = _edge_pair_scatter_maybe_triton(edge_grad, -edge_grad, centers, neighs, n_atoms)
        self._last_vjp_scatter_backend = backend
        if return_strain_derivative:
            strain_derivative = torch.einsum(
                "ea,eb->ab",
                edge_grad,
                x_ij.to(edge_grad.dtype),
            )
            return position_grad, strain_derivative
        return position_grad

    def position_vjp_from_record(self, record, channel_adjoint):
        """Apply a stored site-basis VJP record to final-channel adjoints."""

        channel_adjoint = channel_adjoint.to(
            device=record.final_atomic_base.device,
            dtype=self.cfg.complex_dtype,
        )
        if tuple(channel_adjoint.shape) != tuple(record.final_atomic_base.shape):
            raise ValueError(
                f"channel_adjoint must have shape {tuple(record.final_atomic_base.shape)}; "
                f"got {tuple(channel_adjoint.shape)}"
            )
        raw_adjoint, count_adjoint = self._normalization_reverse(
            record.raw_atomic_base,
            channel_adjoint,
            centers=record.centers,
            bond_idx=record.bond_idx,
            r=record.r,
            n_atoms=record.n_atoms,
            channels=record.channels,
        )
        edge_adjoint = self._density_accumulate_adjoint(
            raw_adjoint,
            record.centers,
        )
        edge_grad = (
            edge_adjoint.unsqueeze(-1) * record.edge_dx
        ).sum(dim=1).real
        if self.cfg.atomic_base_normalization == "soft_neighbor":
            _, weights_dx = self._soft_neighbor_weights_with_dx(
                bond_idx=record.bond_idx,
                r=record.r,
                dr_dx=record.dr_dx,
            )
            edge_grad = edge_grad + count_adjoint.index_select(0, record.centers).unsqueeze(-1) * weights_dx

        position_grad, backend = _edge_pair_scatter_maybe_triton(
            edge_grad,
            -edge_grad,
            record.centers,
            record.neighs,
            record.n_atoms,
        )
        self._last_vjp_scatter_backend = backend
        return position_grad

    def strain_vjp_from_record(self, record, channel_adjoint):
        """Apply a stored site-basis VJP record to homogeneous-strain rows."""

        channel_adjoint = channel_adjoint.to(
            device=record.final_atomic_base.device,
            dtype=self.cfg.complex_dtype,
        )
        if tuple(channel_adjoint.shape) != tuple(record.final_atomic_base.shape):
            raise ValueError(
                f"channel_adjoint must have shape {tuple(record.final_atomic_base.shape)}; "
                f"got {tuple(channel_adjoint.shape)}"
            )
        raw_adjoint, count_adjoint = self._normalization_reverse(
            record.raw_atomic_base,
            channel_adjoint,
            centers=record.centers,
            bond_idx=record.bond_idx,
            r=record.r,
            n_atoms=record.n_atoms,
            channels=record.channels,
        )
        edge_adjoint = self._density_accumulate_adjoint(
            raw_adjoint,
            record.centers,
        )
        edge_grad = (
            edge_adjoint.unsqueeze(-1) * record.edge_dx
        ).sum(dim=1).real
        if self.cfg.atomic_base_normalization == "soft_neighbor":
            _, weights_dx = self._soft_neighbor_weights_with_dx(
                bond_idx=record.bond_idx,
                r=record.r,
                dr_dx=record.dr_dx,
            )
            edge_grad = edge_grad + count_adjoint.index_select(0, record.centers).unsqueeze(-1) * weights_dx
        displacement = record.dr_dx.to(edge_grad.dtype) * record.r.to(edge_grad.dtype).unsqueeze(-1)
        return torch.einsum("ea,eb->ab", edge_grad, displacement)

    def _normalization_reverse_batched(
        self,
        raw_A,
        final_A_adjoint,
        *,
        centers,
        bond_idx,
        r,
        n_atoms,
        channels,
    ):
        mode = self.cfg.atomic_base_normalization
        batch = int(final_A_adjoint.shape[0])
        if mode == "none":
            count_adjoint = torch.zeros((batch, n_atoms), dtype=self.cfg.dtype, device=raw_A.device)
            return final_A_adjoint, count_adjoint

        after_soft = raw_A
        if mode == "soft_neighbor":
            after_soft = self._apply_soft_neighbor_normalization(
                raw_A,
                centers=centers,
                bond_idx=bond_idx,
                r=r,
                n_atoms=n_atoms,
            )

        count_adjoint = torch.zeros((batch, n_atoms), dtype=self.cfg.dtype, device=raw_A.device)
        current_adjoint = final_A_adjoint
        if mode == "soft_neighbor":
            current_adjoint, count_adjoint = self._soft_neighbor_normalization_reverse_batched(
                raw_A,
                current_adjoint,
                centers=centers,
                bond_idx=bond_idx,
                r=r,
                n_atoms=n_atoms,
            )
        return current_adjoint, count_adjoint

    def _soft_neighbor_normalization_reverse_batched(
        self,
        A,
        A_adjoint,
        *,
        centers,
        bond_idx,
        r,
        n_atoms,
    ):
        weights = self._soft_neighbor_weights(bond_idx=bond_idx, r=r).to(self.cfg.dtype)
        soft_count = self._density_accumulate(
            weights.unsqueeze(-1),
            centers,
            n_atoms,
        ).squeeze(-1)
        epsilon = torch.as_tensor(self.cfg.atomic_base_normalization_epsilon, dtype=self.cfg.dtype, device=A.device)
        denom = soft_count + epsilon
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        root_adjoint = A_adjoint / safe_denom.to(A_adjoint.dtype).view(1, -1, 1)
        count_adjoint = -(
            A_adjoint * A.unsqueeze(0)
        ).sum(dim=2).real / safe_denom.pow(2).view(1, -1)
        return root_adjoint, count_adjoint

    def position_vjp_from_record_batched(self, record, channel_adjoint):
        """Apply a stored site-basis VJP record to batched channel adjoints."""

        channel_adjoint = channel_adjoint.to(
            device=record.final_atomic_base.device,
            dtype=self.cfg.complex_dtype,
        )
        expected_tail = tuple(record.final_atomic_base.shape)
        if channel_adjoint.ndim != 3 or tuple(channel_adjoint.shape[1:]) != expected_tail:
            raise ValueError(
                f"channel_adjoint must have shape (batch, {expected_tail[0]}, {expected_tail[1]}); "
                f"got {tuple(channel_adjoint.shape)}"
            )
        raw_adjoint, count_adjoint = self._normalization_reverse_batched(
            record.raw_atomic_base,
            channel_adjoint,
            centers=record.centers,
            bond_idx=record.bond_idx,
            r=record.r,
            n_atoms=record.n_atoms,
            channels=record.channels,
        )
        edge_adjoint = raw_adjoint.index_select(1, record.centers)
        edge_grad = (edge_adjoint.unsqueeze(-1) * record.edge_dx.unsqueeze(0)).sum(dim=2).real
        if self.cfg.atomic_base_normalization == "soft_neighbor":
            _, weights_dx = self._soft_neighbor_weights_with_dx(
                bond_idx=record.bond_idx,
                r=record.r,
                dr_dx=record.dr_dx,
            )
            edge_grad = edge_grad + count_adjoint.index_select(1, record.centers).unsqueeze(-1) * weights_dx.unsqueeze(0)

        use_batched_scatter = bool(
            edge_grad.is_cuda
            and int(edge_grad.numel()) >= _batched_vjp_scatter_min_elements()
        )
        if use_batched_scatter:
            position_grad, backend = _edge_pair_scatter_batched_maybe_triton(
                edge_grad,
                -edge_grad,
                record.centers,
                record.neighs,
                record.n_atoms,
            )
        else:
            rows = []
            backend = "none"
            for batch_index in range(int(edge_grad.shape[0])):
                row, backend = _edge_pair_scatter_maybe_triton(
                    edge_grad[batch_index],
                    -edge_grad[batch_index],
                    record.centers,
                    record.neighs,
                    record.n_atoms,
                )
                rows.append(row)
            position_grad = torch.stack(rows, dim=0)
            backend = "loop_" + str(backend)
        self._last_vjp_scatter_backend = backend
        return position_grad

    def strain_vjp_from_record_batched(self, record, channel_adjoint):
        """Apply a stored site-basis VJP record to batched homogeneous-strain rows."""

        channel_adjoint = channel_adjoint.to(
            device=record.final_atomic_base.device,
            dtype=self.cfg.complex_dtype,
        )
        expected_tail = tuple(record.final_atomic_base.shape)
        if channel_adjoint.ndim != 3 or tuple(channel_adjoint.shape[1:]) != expected_tail:
            raise ValueError(
                f"channel_adjoint must have shape (batch, {expected_tail[0]}, {expected_tail[1]}); "
                f"got {tuple(channel_adjoint.shape)}"
            )
        raw_adjoint, count_adjoint = self._normalization_reverse_batched(
            record.raw_atomic_base,
            channel_adjoint,
            centers=record.centers,
            bond_idx=record.bond_idx,
            r=record.r,
            n_atoms=record.n_atoms,
            channels=record.channels,
        )
        edge_adjoint = raw_adjoint.index_select(1, record.centers)
        edge_grad = (edge_adjoint.unsqueeze(-1) * record.edge_dx.unsqueeze(0)).sum(dim=2).real
        if self.cfg.atomic_base_normalization == "soft_neighbor":
            _, weights_dx = self._soft_neighbor_weights_with_dx(
                bond_idx=record.bond_idx,
                r=record.r,
                dr_dx=record.dr_dx,
            )
            edge_grad = edge_grad + count_adjoint.index_select(1, record.centers).unsqueeze(-1) * weights_dx.unsqueeze(0)
        displacement = record.dr_dx.to(edge_grad.dtype) * record.r.to(edge_grad.dtype).unsqueeze(-1)
        return torch.einsum("bea,ec->bac", edge_grad, displacement)

    def charge_vjp_from_record(self, record, channel_adjoint):
        """Apply a stored site-basis VJP record to form per-atom charge gradients."""

        channel_adjoint = channel_adjoint.to(
            device=record.final_atomic_base.device,
            dtype=self.cfg.complex_dtype,
        )
        if tuple(channel_adjoint.shape) != tuple(record.final_atomic_base.shape):
            raise ValueError(
                f"channel_adjoint must have shape {tuple(record.final_atomic_base.shape)}; "
                f"got {tuple(channel_adjoint.shape)}"
            )
        raw_adjoint, _ = self._normalization_reverse(
            record.raw_atomic_base,
            channel_adjoint,
            centers=record.centers,
            bond_idx=record.bond_idx,
            r=record.r,
            n_atoms=record.n_atoms,
            channels=record.channels,
        )
        edge_adjoint = self._density_accumulate_adjoint(
            raw_adjoint,
            record.centers,
        )
        center_grad = (edge_adjoint * record.edge_dq_center).sum(dim=1).real
        neighbor_grad = (edge_adjoint * record.edge_dq_neighbor).sum(dim=1).real
        charge_grad, backend = _edge_pair_scatter_maybe_triton(
            neighbor_grad,
            center_grad,
            record.centers,
            record.neighs,
            record.n_atoms,
        )
        self._last_vjp_scatter_backend = backend
        return charge_grad

    def charge_vjp_from_channel_adjoint(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        channel_adjoint,
        charges = None,
        aux_tensor_basis = None,
    ):
        """Compact VJP from final atomic-base channel adjoints to charge gradients."""

        channels_list, final_A, record = self.compute_channels_with_vjp_record(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        return channels_list, final_A, self.charge_vjp_from_record(record, channel_adjoint)

    def compute_channels(
        self,
        x_ij,
        edge_index,
        atom_types,
        channels,
        charges = None,
        aux_tensor_basis = None,
        runtime_cache = None,
        real_output = False,
    ):
        """Compute and aggregate ``A[i, alpha]`` for the requested channels.

        Parameters
        ----------
        x_ij : [E,3] tensor
            Edge vectors from center ``i`` to neighbor ``j``.
        edge_index : [2,E] tensor
            Center and neighbor indices.
        atom_types : [N] tensor
            Integer site types matching ``possible_types``.
        channels : sequence of SingleChannelLabel
            Requested leaf-channel basis labels.
        charges : [N] tensor, optional
            Per-atom charges when ``charge_mode='scalar'``.
        aux_tensor_basis : mapping, optional
            Optional precomputed auxiliary tensor basis keyed by ``(l_aux,m_aux)``.

        Returns
        -------
        channels, A
            ``A`` has shape ``[N, n_channels]`` and complex dtype by default.
            When ``real_output`` is true and the requested real spherical
            channel schedule contains only plain channels, ``A`` is returned in
            the real tesseral basis with ``cfg.dtype``.
        """
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")

        profile = {} if _site_basis_profile_enabled() else None
        self._last_profile = {} if profile is None else profile
        self._last_plain_product_backend = None
        total_start = _profile_start_from_tensor(x_ij) if profile is not None else None

        setup_start = _profile_start_from_tensor(x_ij) if profile is not None else None
        x_ij = x_ij.to(dtype=self.cfg.dtype)
        atom_types = atom_types.to(device=x_ij.device)
        component_cache = _runtime_cache_bucket(runtime_cache, "site_basis_components")
        x_key = _tensor_runtime_cache_key(x_ij)
        edge_key = _tensor_runtime_cache_key(edge_index)
        atom_type_key = _tensor_runtime_cache_key(atom_types)
        geometry_key = ("edge_geometry", x_key, edge_key, atom_type_key, tuple(int(x) for x in self.cfg.possible_types))
        if component_cache is not None and geometry_key in component_cache:
            cached_geometry = component_cache[geometry_key]
            if len(cached_geometry) == 8:
                centers, neighs, mu0_edge, mu_edge, bond_idx, r, theta, phi = cached_geometry
                unit = x_ij / torch.clamp(r, min=torch.as_tensor(1e-12, dtype=x_ij.dtype, device=x_ij.device)).unsqueeze(-1)
            else:
                centers, neighs, mu0_edge, mu_edge, bond_idx, r, unit, theta, phi = cached_geometry
            if profile is not None:
                profile["site_basis_component_cache_hits"] = profile.get("site_basis_component_cache_hits", 0.0) + 1.0
                profile["site_basis_geometry_cache_hits"] = profile.get("site_basis_geometry_cache_hits", 0.0) + 1.0
        else:
            centers = edge_index[0].to(device=x_ij.device)
            neighs = edge_index[1].to(device=x_ij.device)
            mu0_edge = atom_types[centers]
            mu_edge = atom_types[neighs]
            bond_idx = self._bond_index(mu0_edge, mu_edge)
            r, unit = self._edge_radius_and_unit(x_ij)
            theta = None
            phi = None
            if component_cache is not None:
                component_cache[geometry_key] = (centers, neighs, mu0_edge, mu_edge, bond_idx, r, unit, theta, phi)
            if profile is not None:
                profile["site_basis_component_cache_misses"] = profile.get("site_basis_component_cache_misses", 0.0) + 1.0
                profile["site_basis_geometry_cache_misses"] = profile.get("site_basis_geometry_cache_misses", 0.0) + 1.0
        if profile is not None:
            _profile_stop_from_tensor(profile, "site_basis.edge_geometry", setup_start, r)
        n_atoms = int(atom_types.shape[0])
        n_channels = len(channels)
        channel_schedule = self._basis_channel_schedule(channels)
        real_plain_output = bool(
            real_output
            and self.cfg.spherical_backend == "real"
            and aux_tensor_basis is None
            and len(channel_schedule.fallback_entries) == 0
        )

        charge_start = _profile_start_from_tensor(x_ij) if profile is not None else None
        needs_charge_tensors = bool(
            self.cfg.charge_mode != "none"
        )
        if not needs_charge_tensors and charges is None:
            charges_key = ("implicit_no_charge_identity", str(x_ij.device), str(self.cfg.dtype), int(n_atoms))
            q_i = None
            q_j = None
        elif charges is None:
            charges_key = ("implicit_zero_charges", str(x_ij.device), str(self.cfg.dtype), int(n_atoms))
            if component_cache is not None and charges_key in component_cache:
                charges = component_cache[charges_key]
                if profile is not None:
                    profile["site_basis_component_cache_hits"] = profile.get("site_basis_component_cache_hits", 0.0) + 1.0
            else:
                charges = torch.zeros(n_atoms, dtype=self.cfg.dtype, device=x_ij.device)
                if component_cache is not None:
                    component_cache[charges_key] = charges
                if profile is not None:
                    profile["site_basis_component_cache_misses"] = profile.get("site_basis_component_cache_misses", 0.0) + 1.0
        else:
            charges = charges.to(dtype=self.cfg.dtype, device=x_ij.device)
            charges_key = _tensor_runtime_cache_key(charges)
        if needs_charge_tensors or charges is not None:
            q_i = charges[centers]
            q_j = charges[neighs]
        if profile is not None:
            _profile_stop_from_tensor(profile, "site_basis.charge_setup", charge_start, x_ij if q_i is None else q_i)

        Y_cache = {}
        R_cache = {}
        Q_cache = {}
        C_cache = {}

        def ensure_angles():
            nonlocal theta, phi
            if theta is None or phi is None:
                theta, phi = self._angles_from_radius_and_unit(r, unit)
                if component_cache is not None:
                    component_cache[geometry_key] = (centers, neighs, mu0_edge, mu_edge, bond_idx, r, unit, theta, phi)
            return theta, phi


        common_component_key = (
            str(self.cfg.dtype),
            str(self.cfg.complex_dtype),
            str(self.cfg.spherical_backend),
            str(self.cfg.spherical_normalization),
            str(getattr(self.cfg, "factor_normalization", "none")),
            x_key,
            edge_key,
            atom_type_key,
        )
        needed_radial_ns_by_l = {}
        for group in channel_schedule.plain_groups:
            needed_radial_ns_by_l.setdefault(int(group.l), set()).add(int(group.n))
        for _, ch in channel_schedule.fallback_entries:
            if ch.eta is None:
                needed_radial_ns_by_l.setdefault(int(ch.l), set()).add(int(ch.n))
        native_radial_cache, native_angular_cache = (
            self._native_source_derivative_tables(
                x_ij=x_ij,
                bond_idx=bond_idx,
                r=r,
                channel_schedule=channel_schedule,
            )
        )
        R_cache.update(
            {
                key: value_and_derivative[0]
                for key, value_and_derivative
                in native_radial_cache.items()
            }
        )
        Y_cache.update(
            {
                key: value_and_derivative[0]
                for key, value_and_derivative
                in native_angular_cache.items()
            }
        )

        def multiply_real_optional(base, *factors):
            out = base
            for factor in factors:
                if factor is not None:
                    out = out * factor.to(dtype=out.dtype)
            return out

        def multiply_complex_optional(base, *factors):
            out = base
            for factor in factors:
                if factor is not None:
                    out = out * factor.to(dtype=self.cfg.complex_dtype)
            return out

        def cached_aux_factor(channel):
            if channel.l_aux is None and channel.m_aux is None:
                return None
            if aux_tensor_basis is None:
                raise ValueError("Channel requests auxiliary tensor/angular indices, but aux_tensor_basis is None")
            key = (channel.l_aux, channel.m_aux)
            aux = aux_tensor_basis[key].to(self.cfg.complex_dtype)
            aux_cache = _runtime_cache_bucket(runtime_cache, "site_basis_aux_identity")
            aux_key = ("aux_identity", _tensor_runtime_cache_key(aux), key)
            identity = None if aux_cache is None else aux_cache.get(aux_key)
            if identity is None:
                identity = bool(torch.all(aux == torch.ones((), dtype=aux.dtype, device=aux.device)).detach().cpu().item())
                if aux_cache is not None:
                    aux_cache[aux_key] = identity
            if identity:
                if profile is not None:
                    profile["site_basis_aux_identity_skips"] = profile.get("site_basis_aux_identity_skips", 0.0) + 1.0
                return None
            return aux

        def cached_chemical(mu0, mu):
            chem_key = (int(mu0), int(mu))
            if chem_key in C_cache:
                return C_cache[chem_key]
            if (
                isinstance(self.chemical_provider, DefaultChemicalBasisProvider)
                and self.cfg.chemical_basis == "delta"
                and self.cfg.ntypes == 1
                and int(mu0) == int(self.cfg.possible_types[0])
                and int(mu) == int(self.cfg.possible_types[0])
            ):
                C_cache[chem_key] = None
                if profile is not None:
                    profile["site_basis_chemical_identity_skips"] = profile.get("site_basis_chemical_identity_skips", 0.0) + 1.0
                return None
            cache_key = (
                "chemical",
                self._chemical_runtime_cache_identity(),
                common_component_key,
                chem_key,
            )
            start = _profile_start_from_tensor(x_ij) if profile is not None else None
            if component_cache is not None and cache_key in component_cache:
                value = component_cache[cache_key]
                if profile is not None:
                    profile["site_basis_component_cache_hits"] = profile.get("site_basis_component_cache_hits", 0.0) + 1.0
                    profile["site_basis_chemical_cache_hits"] = profile.get("site_basis_chemical_cache_hits", 0.0) + 1.0
            else:
                value = self._chemical_basis(mu0_edge, mu_edge, mu0, mu)
                if component_cache is not None:
                    component_cache[cache_key] = value
                if profile is not None:
                    profile["site_basis_component_cache_misses"] = profile.get("site_basis_component_cache_misses", 0.0) + 1.0
                    profile["site_basis_chemical_cache_misses"] = profile.get("site_basis_chemical_cache_misses", 0.0) + 1.0
                    _profile_stop_from_tensor(profile, "site_basis.chemical_basis", start, value)
            C_cache[chem_key] = value
            return value

        def cached_charge(mu0, mu, kappa0, kappa):
            q_key = (int(mu0), int(mu), int(kappa0), int(kappa))
            if q_key in Q_cache:
                return Q_cache[q_key]
            if isinstance(self.charge_provider, DefaultChargeBasisProvider) and self.cfg.charge_mode == "none":
                Q_cache[q_key] = None
                if profile is not None:
                    profile["site_basis_charge_identity_skips"] = profile.get("site_basis_charge_identity_skips", 0.0) + 1.0
                return None
            cache_key = (
                "charge",
                self._charge_runtime_cache_identity(),
                common_component_key,
                charges_key,
                q_key,
            )
            start = _profile_start_from_tensor(x_ij) if profile is not None else None
            if component_cache is not None and cache_key in component_cache:
                value = component_cache[cache_key]
                if profile is not None:
                    profile["site_basis_component_cache_hits"] = profile.get("site_basis_component_cache_hits", 0.0) + 1.0
                    profile["site_basis_charge_cache_hits"] = profile.get("site_basis_charge_cache_hits", 0.0) + 1.0
            else:
                value = self._charge_basis(q_i, q_j, mu0_edge, mu_edge, kappa0, kappa)
                if component_cache is not None:
                    component_cache[cache_key] = value
                if profile is not None:
                    profile["site_basis_component_cache_misses"] = profile.get("site_basis_component_cache_misses", 0.0) + 1.0
                    profile["site_basis_charge_cache_misses"] = profile.get("site_basis_charge_cache_misses", 0.0) + 1.0
                    _profile_stop_from_tensor(profile, "site_basis.charge_basis", start, value)
            Q_cache[q_key] = value
            return value

        def cached_radial(n, l):
            r_key = (int(n), int(l))
            if r_key in R_cache:
                return R_cache[r_key]
            cache_key = (
                "radial",
                self._radial_runtime_cache_identity(),
                common_component_key,
                _tensor_runtime_cache_key(bond_idx),
                r_key,
            )
            start = _profile_start_from_tensor(x_ij) if profile is not None else None
            if component_cache is not None and cache_key in component_cache:
                value = component_cache[cache_key]
                if profile is not None:
                    profile["site_basis_component_cache_hits"] = profile.get("site_basis_component_cache_hits", 0.0) + 1.0
                    profile["site_basis_radial_cache_hits"] = profile.get("site_basis_radial_cache_hits", 0.0) + 1.0
            else:
                if (
                    isinstance(self.radial_provider, DefaultRadialBasisProvider)
                    and str(self.cfg.radial_basis).strip().lower().replace("_", "").replace("-", "") == "chebexpcos"
                ):
                    l_int = int(l)
                    missing = []
                    for n_int in sorted(needed_radial_ns_by_l.get(l_int, {int(n)})):
                        key_n = (
                            "radial",
                            self._radial_runtime_cache_identity(),
                            common_component_key,
                            _tensor_runtime_cache_key(bond_idx),
                            (int(n_int), l_int),
                        )
                        if (int(n_int), l_int) not in R_cache and not (component_cache is not None and key_n in component_cache):
                            missing.append(int(n_int))
                    values = self._default_radial_basis_many(n_values=missing or (int(n),), l=int(l), bond_idx=bond_idx, r=r)
                    for n_int, radial_value in values.items():
                        radial_value = self._normalize_radial_values(int(n_int), int(l), radial_value)
                        key_n = (
                            "radial",
                            self._radial_runtime_cache_identity(),
                            common_component_key,
                            _tensor_runtime_cache_key(bond_idx),
                            (int(n_int), l_int),
                        )
                        R_cache[(int(n_int), l_int)] = radial_value
                        if component_cache is not None:
                            component_cache[key_n] = radial_value
                    value = R_cache[r_key]
                    if profile is not None and len(values) > 1:
                        profile["site_basis_radial_fused_values"] = profile.get("site_basis_radial_fused_values", 0.0) + float(len(values))
                else:
                    value = self._radial_basis(n, l, bond_idx, r)
                    if component_cache is not None:
                        component_cache[cache_key] = value
                if profile is not None:
                    profile["site_basis_component_cache_misses"] = profile.get("site_basis_component_cache_misses", 0.0) + 1.0
                    profile["site_basis_radial_cache_misses"] = profile.get("site_basis_radial_cache_misses", 0.0) + 1.0
                    _profile_stop_from_tensor(profile, "site_basis.radial_basis", start, value)
            R_cache[r_key] = value
            return value

        def cached_spherical(l):
            l = int(l)
            if l in Y_cache:
                return Y_cache[l]
            cache_key = (
                "spherical",
                str(self.cfg.spherical_backend),
                str(self.cfg.complex_dtype),
                common_component_key,
                l,
            )
            start = _profile_start_from_tensor(x_ij) if profile is not None else None
            if component_cache is not None and cache_key in component_cache:
                value = component_cache[cache_key]
                if profile is not None:
                    profile["site_basis_component_cache_hits"] = profile.get("site_basis_component_cache_hits", 0.0) + 1.0
                    profile["site_basis_spherical_cache_hits"] = profile.get("site_basis_spherical_cache_hits", 0.0) + 1.0
            else:
                if self.cfg.spherical_backend == "real":
                    value = self._normalize_spherical_values(
                        l,
                        _real_spherical_harmonics_l_from_unit_cartesian(l, unit).to(self.cfg.dtype),
                    )
                    if profile is not None:
                        profile["site_basis_spherical_no_angle_evals"] = (
                            profile.get("site_basis_spherical_no_angle_evals", 0.0) + 1.0
                        )
                elif str(self.cfg.radial_basis).strip().lower().replace(
                    "_", ""
                ).replace("-", "") in {
                    "pacechebexpcos",
                    "pacechebexpcosidentityspline",
                }:
                    value = self._spherical_values_for_forward(
                        l,
                        x_ij,
                        None,
                        None,
                    )
                    if profile is not None:
                        profile["site_basis_spherical_no_angle_evals"] = (
                            profile.get("site_basis_spherical_no_angle_evals", 0.0)
                            + 1.0
                        )
                else:
                    theta_eval, phi_eval = ensure_angles()
                    value = self._normalize_spherical_values(
                        l,
                        self.sph.all_m(l, theta_eval, phi_eval).to(self.cfg.complex_dtype),
                    )
                if component_cache is not None:
                    component_cache[cache_key] = value
                if profile is not None:
                    profile["site_basis_component_cache_misses"] = profile.get("site_basis_component_cache_misses", 0.0) + 1.0
                    profile["site_basis_spherical_cache_misses"] = profile.get("site_basis_spherical_cache_misses", 0.0) + 1.0
                    _profile_stop_from_tensor(profile, "site_basis.spherical_harmonics", start, value)
            Y_cache[l] = value
            return value

        product_start = _profile_start_from_tensor(x_ij) if profile is not None else None
        enable_triton_site_basis = str(_ace_env("ENABLE_TRITON_SITE_BASIS", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        enable_triton_site_basis_grad = str(_ace_env("ENABLE_TRITON_SITE_BASIS_GRAD", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "force",
        }
        triton_plain_direct = bool(
            enable_triton_site_basis
            and (not torch.is_grad_enabled() or enable_triton_site_basis_grad)
            and triton is not None
            and _triton_runtime_supported()
            and x_ij.is_cuda
            and centers.is_cuda
            and centers.dtype == torch.long
            and self.cfg.dtype in (torch.float32, torch.float64)
            and self.cfg.spherical_backend == "real"
            and len(channel_schedule.fallback_entries) == 0
        )
        packed_edge_table_raw = str(_ace_env("ENABLE_PACKED_EDGE_TABLE", "1")).strip().lower()
        packed_edge_table_forced = packed_edge_table_raw == "force"
        enable_packed_edge_table = packed_edge_table_raw not in {"0", "false", "no", "off"}
        packed_plain_edge_table = bool(
            enable_packed_edge_table
            and not triton_plain_direct
            and self.cfg.spherical_backend == "real"
            and len(channel_schedule.plain_groups) > 1
            and (x_ij.is_cuda or packed_edge_table_forced)
        )
        alloc_start = _profile_start_from_tensor(x_ij) if profile is not None else None
        if triton_plain_direct:
            A_real = torch.zeros((n_atoms, n_channels), dtype=self.cfg.dtype, device=x_ij.device)
            edge_vals = None
            direct_prefactors = []
            direct_term_y = []
            direct_term_group = []
            direct_term_channel = []
            if profile is not None:
                profile["site_basis.triton_plain_direct_enabled"] = profile.get("site_basis.triton_plain_direct_enabled", 0.0) + 1.0
                _profile_stop_from_tensor(profile, "site_basis.atomic_base_allocation", alloc_start, A_real)
        else:
            edge_dtype = self.cfg.dtype if real_plain_output else self.cfg.complex_dtype
            edge_vals = torch.zeros((x_ij.shape[0], n_channels), dtype=edge_dtype, device=x_ij.device)
            A_real = None
            direct_prefactors = None
            direct_term_y = None
            direct_term_group = None
            direct_term_channel = None
            if profile is not None:
                _profile_stop_from_tensor(profile, "site_basis.edge_value_allocation", alloc_start, edge_vals)

        soft_neighbor_edge_weights = None
        if self.cfg.atomic_base_normalization == "soft_neighbor":
            soft_neighbor_edge_weights = self._soft_neighbor_weights(bond_idx=bond_idx, r=r).to(self.cfg.dtype)

        for group in channel_schedule.plain_groups:
            mu0, mu, kappa0, kappa, n, l = group.mu0, group.mu, group.kappa0, group.kappa, group.n, group.l
            chem = cached_chemical(mu0, mu)
            qf = cached_charge(mu0, mu, kappa0, kappa)
            rf = cached_radial(n, l)
            cached_spherical(l)
            group_tensors = group.tensors(x_ij.device)
            if triton_plain_direct:
                prefactor = multiply_real_optional(rf.to(self.cfg.dtype), chem, qf)
                if soft_neighbor_edge_weights is not None:
                    prefactor = prefactor * soft_neighbor_edge_weights
                if torch.is_grad_enabled():
                    group_id = len(direct_prefactors)
                    direct_prefactors.append(prefactor)
                    direct_term_y.append(Y_cache[l].real.index_select(0, group_tensors.m_positions))
                    direct_term_channel.append(group_tensors.indices)
                    direct_term_group.append(
                        torch.full(
                            group_tensors.indices.shape,
                            int(group_id),
                            dtype=torch.long,
                            device=x_ij.device,
                        )
                    )
                else:
                    _plain_real_site_basis_scatter_triton(
                        A_real,
                        prefactor=prefactor,
                        y_all=Y_cache[l],
                        group_tensors=group_tensors,
                        centers=centers,
                    )
            elif packed_plain_edge_table:
                continue
            elif real_plain_output:
                y = Y_cache[l].real.index_select(0, group_tensors.m_positions).transpose(0, 1)
                prefactor = multiply_real_optional(rf.to(self.cfg.dtype), chem, qf)
                if soft_neighbor_edge_weights is not None:
                    prefactor = prefactor * soft_neighbor_edge_weights
                edge_vals.index_copy_(1, group_tensors.indices, prefactor.unsqueeze(-1) * y)
            else:
                y = Y_cache[l].index_select(0, group_tensors.m_positions).transpose(0, 1)
                prefactor = multiply_complex_optional(rf.to(self.cfg.complex_dtype), chem, qf)
                if soft_neighbor_edge_weights is not None:
                    prefactor = prefactor * soft_neighbor_edge_weights.to(prefactor.dtype)
                edge_vals.index_copy_(1, group_tensors.indices, prefactor.unsqueeze(-1) * y)

        if (
            packed_plain_edge_table
            and channel_schedule.plain_groups
        ):
            packed_tensors = self._packed_plain_edge_table_tensors(channel_schedule, x_ij.device)
            radial_table = torch.stack(
                [
                    cached_radial(n, l).to(self.cfg.dtype)
                    for n, l in packed_tensors.radial_keys
                ],
                dim=0,
            )
            prefactors = radial_table.index_select(
                0,
                packed_tensors.radial_group_indices,
            )
            chemical_factors = [
                cached_chemical(mu0, mu)
                for mu0, mu in packed_tensors.chemical_keys
            ]
            if any(
                factor is not None
                for factor in chemical_factors
            ):
                chemical_table = torch.stack(
                    [
                        (
                            torch.ones_like(radial_table[0])
                            if factor is None
                            else factor.to(self.cfg.dtype)
                        )
                        for factor in chemical_factors
                    ],
                    dim=0,
                )
                prefactors = prefactors * chemical_table.index_select(
                    0,
                    packed_tensors.chemical_group_indices,
                )
            charge_factors = [
                cached_charge(mu0, mu, kappa0, kappa)
                for mu0, mu, kappa0, kappa
                in packed_tensors.charge_keys
            ]
            if any(factor is not None for factor in charge_factors):
                charge_table = torch.stack(
                    [
                        (
                            torch.ones_like(radial_table[0])
                            if factor is None
                            else factor.to(self.cfg.dtype)
                        )
                        for factor in charge_factors
                    ],
                    dim=0,
                )
                prefactors = prefactors * charge_table.index_select(
                    0,
                    packed_tensors.charge_group_indices,
                )
            if soft_neighbor_edge_weights is not None:
                prefactors = (
                    prefactors *
                    soft_neighbor_edge_weights.unsqueeze(0)
                )
            angular_table = torch.cat(
                [
                    cached_spherical(l).real
                    for l in packed_tensors.angular_momenta
                ],
                dim=0,
            )
            term_y = angular_table.index_select(
                0,
                packed_tensors.angular_indices,
            )
            packed_values = prefactors.index_select(0, packed_tensors.group_indices) * term_y
            self._last_plain_product_backend = (
                "torch_packed_batched_edge_table"
            )
            if not real_plain_output:
                packed_values = torch.complex(packed_values, torch.zeros_like(packed_values)).to(self.cfg.complex_dtype)
            edge_vals.index_copy_(1, packed_tensors.channel_indices, packed_values.transpose(0, 1))
            if profile is not None:
                profile["site_basis.packed_edge_table_enabled"] = (
                    profile.get("site_basis.packed_edge_table_enabled", 0.0) + 1.0
                )
                profile["site_basis_packed_edge_table_groups"] = (
                    profile.get("site_basis_packed_edge_table_groups", 0.0) + float(len(channel_schedule.plain_groups))
                )
                profile["site_basis_packed_edge_table_terms"] = (
                    profile.get("site_basis_packed_edge_table_terms", 0.0) + float(packed_tensors.channel_indices.numel())
                )


        for idx, ch in channel_schedule.fallback_entries:
            chem = cached_chemical(ch.mu0, ch.mu)
            qf = cached_charge(ch.mu0, ch.mu, ch.kappa0, ch.kappa)
            rf = cached_radial(ch.n, ch.l)
            cached_spherical(ch.l)
            ang = Y_cache[ch.l][ch.m + ch.l]

            value = multiply_complex_optional(rf.to(self.cfg.complex_dtype), chem, qf) * ang
            if soft_neighbor_edge_weights is not None:
                value = value * soft_neighbor_edge_weights.to(value.dtype)

            if ch.l_aux is not None or ch.m_aux is not None:
                aux = cached_aux_factor(ch)
                if aux is not None:
                    value = value * aux

            edge_vals[:, idx] = value

        if triton_plain_direct and torch.is_grad_enabled() and direct_prefactors:
            A_real = _packed_plain_real_site_basis_scatter_autograd(
                prefactors=torch.stack(direct_prefactors, dim=0),
                term_y=torch.cat(direct_term_y, dim=0),
                term_group=torch.cat(direct_term_group, dim=0),
                term_channel=torch.cat(direct_term_channel, dim=0),
                centers=centers,
                n_atoms=n_atoms,
                n_channels=n_channels,
            )

        if profile is not None:
            _profile_stop_from_tensor(
                profile,
                "site_basis.edge_channel_products",
                product_start,
                A_real if triton_plain_direct else edge_vals,
            )
        if triton_plain_direct:
            if real_plain_output:
                A = A_real
            else:
                A = torch.complex(A_real, torch.zeros_like(A_real)).to(self.cfg.complex_dtype)
            if profile is not None:
                profile["site_basis.scatter_fused_into_products"] = profile.get("site_basis.scatter_fused_into_products", 0.0) + 1.0
        else:
            scatter_start = _profile_start_from_tensor(edge_vals) if profile is not None else None
            A = self._density_accumulate(edge_vals, centers, n_atoms)
            if profile is not None:
                _profile_stop_from_tensor(profile, "site_basis.edge_to_atom_scatter", scatter_start, A)
        if self.cfg.atomic_base_normalization != "none":
            norm_start = _profile_start_from_tensor(A) if profile is not None else None
            soft_count = None
            if self.cfg.atomic_base_normalization == "soft_neighbor":
                soft_key = (
                    "soft_neighbor_count",
                    common_component_key,
                    _tensor_runtime_cache_key(bond_idx),
                    float(self.cfg.atomic_base_normalization_epsilon),
                )
                if component_cache is not None and soft_key in component_cache:
                    soft_count = component_cache[soft_key]
                    if profile is not None:
                        profile["site_basis_component_cache_hits"] = profile.get("site_basis_component_cache_hits", 0.0) + 1.0
                        profile["site_basis_soft_count_cache_hits"] = profile.get("site_basis_soft_count_cache_hits", 0.0) + 1.0
                else:
                    weights = self._soft_neighbor_weights(bond_idx=bond_idx, r=r).to(self.cfg.dtype)
                    soft_count = self._density_accumulate(
                        weights.unsqueeze(-1),
                        centers,
                        n_atoms,
                    ).squeeze(-1)
                    if component_cache is not None:
                        component_cache[soft_key] = soft_count
                    if profile is not None:
                        profile["site_basis_component_cache_misses"] = profile.get("site_basis_component_cache_misses", 0.0) + 1.0
                        profile["site_basis_soft_count_cache_misses"] = profile.get("site_basis_soft_count_cache_misses", 0.0) + 1.0
            A = self._apply_atomic_base_normalization(
                A,
                centers=centers,
                bond_idx=bond_idx,
                r=r,
                n_atoms=n_atoms,
                channels=channels,
                soft_count=soft_count,
            )
            if profile is not None:
                _profile_stop_from_tensor(profile, "site_basis.normalization", norm_start, A)
        if triton_plain_direct:
            self._last_scatter_backend = "triton_plain_direct"
        if profile is not None and total_start is not None:
            _profile_stop_from_tensor(profile, "site_basis.total", total_start, A)
        return list(channels), A

    def _charge_basis_table(
        self,
        *,
        q,
        mu_edge,
        max_kappa,
    ):
        if self.cfg.charge_mode == "none":
            return torch.ones((q.shape[0], max_kappa + 1), dtype=self.cfg.dtype, device=q.device)
        mu_local = self._map_types_to_local(mu_edge)
        q_scaled = self._normalize_charge(q, mu_local)
        return torch.stack(
            [self.cheb(q_scaled, int(kappa)).to(self.cfg.dtype) for kappa in range(int(max_kappa) + 1)],
            dim=1,
        )

    def _charge_basis_table_with_derivative(
        self,
        *,
        q,
        mu_edge,
        max_kappa,
    ):
        if self.cfg.charge_mode == "none":
            values = torch.ones((q.shape[0], max_kappa + 1), dtype=self.cfg.dtype, device=q.device)
            return values, torch.zeros_like(values)
        mu_local = self._map_types_to_local(mu_edge)
        q_scaled, dq_scaled = self._normalize_charge_with_derivative(q, mu_local)
        values = []
        derivatives = []
        for kappa in range(int(max_kappa) + 1):
            values.append(self.cheb(q_scaled, int(kappa)).to(self.cfg.dtype))
            derivatives.append((self._chebyshev_first_derivative(q_scaled, int(kappa)) * dq_scaled).to(self.cfg.dtype))
        return torch.stack(values, dim=1), torch.stack(derivatives, dim=1)








    def last_scatter_backend(self):
        return str(self._last_scatter_backend)

    def last_vjp_scatter_backend(self):
        return str(self._last_vjp_scatter_backend)
