"""Streamed reference evaluator for tagged Cauchy descriptors (WP3 slice).

This module materializes ordinary ACE edge primitives (radial times
spherical harmonic, any l), binds them to compiled lifted-Cauchy artifact
roles under an explicit tuple-tag convention, and pools ordered distinct
neighbor tuples per center into a physical descriptor.  It reuses
:class:`ye3t_ace.lifted_cauchy_linear.LiftedCauchyTorchEvaluator` for the
per-tuple algebra and :mod:`ye3t.couplings` for compilation; it does not
enumerate Young labels or coupling paths itself.

Tag count (how many ordered neighbor-tuple slots a descriptor consumes) and
basis mode (the physical channel content: species, radial index, angular
momentum) are independent request fields.  ``role_dimension`` is always
``tag_count + 1`` here (one context/density role appended after the edge
tag roles) and is set directly on the request independent of the channel
content, per :func:`compile_tagged_cauchy_artifact`.

Only ``beta`` and the per-species ``offset`` are fitted.  The compiled
artifact, the real-form tables, and the radial tables are fixed buffers.

Two execution strategies compute the identical pooled descriptor:
``TaggedTupleEvaluator`` (formula T1, direct ordered-tuple enumeration) and
``TaggedMomentEvaluator`` (formula T2, exact set-partition moment
reduction, WP3b).  Both are built from the same compiled artifact and
``role_bindings``; ``moment_equivalence_certificate`` checks they agree.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path

from ase import Atoms as _ASEAtoms
import numpy as np
import torch

from ye3t.couplings import CompiledLiftedCauchyScalar
from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import count as count_coupling
from ye3t.couplings import first_lifted_cauchy_scalar_request
from ye3t.couplings import plan as plan_coupling
from ye3t.execution_plan import compile_tagged_moment_execution_portfolio

# Lazy optional-sympy proxy (own package convention: `ye3t.couplings.lifted_cauchy_scalar`
# uses the same `ye3t._optional_sympy.sp` object for its exact-payload algebra).  Attribute
# access triggers the real `import sympy` on first use; WP3c's real-form lowering
# (`real_moment_program` below) is the only thing in this module that touches it.
from ye3t._optional_sympy import sp as _sympy

# WP1 deliverable, final for this stage: role_bindings normalization, the
# tuple-count/budget formula, and the tag-support oracle.  Imported directly
# (no fallback) per the lead's instruction; absence is a loud ImportError.
from ye3t.couplings.tagged_cauchy import (
    normalize_role_bindings,
    ordered_distinct_tuple_count,
    tag_support_report,
)

from ye3t_ace.lifted_cauchy_linear import (
    LiftedCauchyTorchEvaluator,
    _artifact_channels,
    _binary_complex,
    _load_compiled_artifact,
)
from ye3t_ace.equivariant_calc.angular_basis import ComplexSphericalHarmonicsBasis
from ye3t_ace.equivariant_calc.edge_geometry import directed_edges_all_images_bruteforce
from ye3t_ace.equivariant_calc.radial_basis import _pace_cheb_exp_cos_table_with_derivative


TAGGED_CAUCHY_SLICE_SCHEMA_V0 = "ye3t_tagged_cauchy_slice_v0"
TAGGED_CAUCHY_SLICE_SCHEMA_V1 = "ye3t_tagged_cauchy_slice_v1"
# v2 (WP3c) adds three blocks on top of v1: `real_moment_program` (the exact
# real-arithmetic / "real tesseral" lowering of the moment-reduction program,
# see `real_moment_program` below), `channel_real_forms` (per-channel
# real_form_id plus (l, radial_channel, neighbor_species) so a native loader
# does not need to re-derive channel bookkeeping), and `radial_definition`
# (the PACE ChebExpCos parameters and the explicit n=1,2 column mapping).
# Nothing in v1 is removed or renamed; `load_tagged_model` accepts both.
TAGGED_CAUCHY_SLICE_SCHEMA_V2 = "ye3t_tagged_cauchy_slice_v2"
TAGGED_CAUCHY_SLICE_SCHEMA = TAGGED_CAUCHY_SLICE_SCHEMA_V2
TAGGED_CAUCHY_COMPOSITE_SCHEMA = "ye3t_tagged_cauchy_composite_v1"

DEFAULT_TA_FAMILIES = ("NT_NU2_MU2_SIGN_L1x1", "TR_NU2_MU2_K2_L0", "NT_NU4_K22_L0")

TA_CUTOFF = 4.67637
TA_RADIAL_CONFIG = {"lmbda": 0.5723, "cutoff_width": 0.01}

_ANGULAR_BASIS = ComplexSphericalHarmonicsBasis()
_INVERSE_REAL_FORM_CACHE = {}


def _real_float64_tensor(value, name):
    tensor = torch.as_tensor(value, dtype=torch.complex128)
    max_imag = float(torch.max(torch.abs(tensor.imag)).item()) if tensor.numel() else 0.0
    if max_imag != 0.0:
        raise ValueError(f"{name} must be real-valued; max imaginary magnitude is {max_imag:.3e}.")
    return tensor.real


def compile_tagged_cauchy_artifact(
    role_dimension,
    *,
    element,
    family_ids=DEFAULT_TA_FAMILIES,
    radial_channel_count=2,
    angular_l=1,
    emit_factored=False,
    emit_ordered_reference=False,
):
    """Compile a lifted-Cauchy artifact with an explicit role_dimension.

    The public single-element request builder
    (:func:`ye3t.couplings.first_lifted_cauchy_scalar_request`) fixes
    role_dimension=2.  Tag count and basis mode are independent request
    fields, so role_dimension is patched onto its plain-dict output before
    ``count``/``plan``/``compile`` rather than re-deriving channel
    bookkeeping by hand.
    """

    request = dict(
        first_lifted_cauchy_scalar_request(
            int(radial_channel_count),
            element=element,
            angular_l=int(angular_l),
            family_ids=tuple(family_ids),
        )
    )
    request["role_dimension"] = int(role_dimension)
    request["emit_canonical"] = True
    request["emit_factored"] = bool(emit_factored)
    request["emit_ordered_reference"] = bool(emit_ordered_reference)
    report = count_coupling(request)
    compiler_plan = plan_coupling(report)
    return compile_coupling(compiler_plan)


def _species_order_from_channels(channels):
    return tuple(sorted({str(channel["neighbor_species"]) for channel in channels}))


def _single_species_atoms(positions, species):
    positions = np.asarray(positions, dtype=np.float64)
    return _ASEAtoms(symbols=[species] * positions.shape[0], positions=positions, pbc=False)


def ordinary_edge_primitives(positions, atom_types, cell, pbc, cutoff, radial_config, channels):
    """Build directed edges and complex ordinary-ACE primitives.

    Returns ``edge_index`` (``[2, edges]`` long), ``displacements``
    (``[edges, 3]``), and complex ``phi`` (``[edges, channels, max_width]``,
    zero-padded to the largest ``2*l+1`` among ``channels``).  Periodic
    images are enumerated as distinct edge occurrences.  ``positions`` must
    not be detached; the radial and angular primitives are ordinary torch
    tensor operations, so autograd differentiates through them directly.
    """

    positions = torch.as_tensor(positions)
    if positions.dtype != torch.float64:
        raise ValueError("ordinary_edge_primitives requires float64 positions.")
    atom_types = torch.as_tensor(atom_types, dtype=torch.long, device=positions.device)
    if int(atom_types.shape[0]) != int(positions.shape[0]):
        raise ValueError("atom_types must have one entry per atom.")
    channels = tuple(channels)
    if not channels:
        raise ValueError("channels must be nonempty.")
    cutoff = float(cutoff)
    lmbda = float(radial_config["lmbda"])
    cutoff_width = float(radial_config.get("cutoff_width", 0.0))
    radial_count = max(int(channel["radial_channel"]) for channel in channels) + 1
    species_order = _species_order_from_channels(channels)
    species_index = {name: index for index, name in enumerate(species_order)}
    max_width = max(2 * int(channel["l"]) + 1 for channel in channels)

    cell_t = None if cell is None else torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
    src, dst, disp, dist = directed_edges_all_images_bruteforce(positions, cutoff, cell=cell_t, pbc=pbc)
    edge_index = torch.stack((src, dst))

    if int(src.numel()) == 0:
        phi = positions.new_zeros((0, len(channels), max_width), dtype=torch.complex128)
        return edge_index, disp, phi

    radial_values, _ = _pace_cheb_exp_cos_table_with_derivative(
        dist,
        rc=cutoff,
        cutoff_width=cutoff_width,
        lmbda=lmbda,
        radial_count=radial_count,
    )
    dst_species = atom_types.index_select(0, dst)

    angular_cache = {}
    channel_tensors = []
    for channel in channels:
        angular_l = int(channel["l"])
        radial_n = int(channel["radial_channel"])
        neighbor_species = str(channel["neighbor_species"])
        if neighbor_species not in species_index:
            raise ValueError(
                f"Channel neighbor_species {neighbor_species!r} is absent from atom_types."
            )
        if angular_l not in angular_cache:
            angular_cache[angular_l] = _ANGULAR_BASIS.cartesian_values(angular_l, disp).transpose(0, 1)
        angular = angular_cache[angular_l]
        width = int(angular.shape[-1])
        radial_column = radial_values[:, radial_n].to(angular.dtype)
        value = radial_column.unsqueeze(-1) * angular
        species_mask = (dst_species == species_index[neighbor_species]).unsqueeze(-1)
        value = torch.where(species_mask, value, torch.zeros_like(value))
        if width < max_width:
            pad = value.new_zeros(tuple(value.shape[:-1]) + (max_width - width,))
            value = torch.cat((value, pad), dim=-1)
        channel_tensors.append(value)
    phi = torch.stack(channel_tensors, dim=1)
    return edge_index, disp, phi


def ordinary_edge_primitives_with_derivative(positions, atom_types, cell, pbc, cutoff, radial_config, channels):
    """Like :func:`ordinary_edge_primitives`, plus the analytic Cartesian
    derivative of ``phi`` with respect to each edge's OWN displacement
    (WP4e).

    Returns ``edge_index``, ``disp``, ``phi`` (as before) and ``dphi``
    (``[edges, channels, max_width, 3]`` complex, ``dphi[e,c,a,k] =
    d(phi[e,c,a]) / d(disp[e,k])``) -- purely LOCAL per edge (``phi[e]``
    depends only on ``disp[e,:]``, never on another edge's displacement),
    so this needs no autograd through the moment-reduction machinery at
    all: it is built directly from two already-analytic pieces already
    used elsewhere in this codebase --
    ``_pace_cheb_exp_cos_table_with_derivative``'s own radial derivative
    ``dR/dr`` (previously computed and discarded by
    :func:`ordinary_edge_primitives` itself, which only kept the value),
    chain-ruled through ``dr/d(disp) = disp/r``, and
    ``ComplexSphericalHarmonicsBasis.cartesian_derivative`` (the same
    ``real_spherical_harmonics_l_from_cartesian_with_derivatives`` path
    the brief points at for general ``l``; that function's own cost is
    ``O(2l+1)`` autograd calls per distinct ``l``, not one per edge or per
    feature, so this is still exact and only a small, BOUNDED amount of
    autograd -- never proportional to the number of edges, channels, or
    (critically) features). Combined via the ordinary product rule
    ``d(radial * angular)/d(disp) = (dR/d(disp)) * angular + R *
    (d(angular)/d(disp))``, both terms exact.
    """

    positions = torch.as_tensor(positions)
    if positions.dtype != torch.float64:
        raise ValueError("ordinary_edge_primitives_with_derivative requires float64 positions.")
    atom_types = torch.as_tensor(atom_types, dtype=torch.long, device=positions.device)
    if int(atom_types.shape[0]) != int(positions.shape[0]):
        raise ValueError("atom_types must have one entry per atom.")
    channels = tuple(channels)
    if not channels:
        raise ValueError("channels must be nonempty.")
    cutoff = float(cutoff)
    lmbda = float(radial_config["lmbda"])
    cutoff_width = float(radial_config.get("cutoff_width", 0.0))
    radial_count = max(int(channel["radial_channel"]) for channel in channels) + 1
    species_order = _species_order_from_channels(channels)
    species_index = {name: index for index, name in enumerate(species_order)}
    max_width = max(2 * int(channel["l"]) + 1 for channel in channels)

    cell_t = None if cell is None else torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
    src, dst, disp, dist = directed_edges_all_images_bruteforce(positions, cutoff, cell=cell_t, pbc=pbc)
    edge_index = torch.stack((src, dst))

    if int(src.numel()) == 0:
        phi = positions.new_zeros((0, len(channels), max_width), dtype=torch.complex128)
        dphi = positions.new_zeros((0, len(channels), max_width, 3), dtype=torch.complex128)
        return edge_index, disp, phi, dphi

    radial_values, radial_deriv = _pace_cheb_exp_cos_table_with_derivative(
        dist,
        rc=cutoff,
        cutoff_width=cutoff_width,
        lmbda=lmbda,
        radial_count=radial_count,
    )
    # d(dist)/d(disp) = disp / dist (dist > 0 always: directed_edges_*
    # never emits a zero-length self-edge within a positive cutoff).
    dist_safe = torch.clamp(dist, min=torch.finfo(dist.dtype).tiny)
    d_dist_d_disp = disp / dist_safe.unsqueeze(-1)
    dst_species = atom_types.index_select(0, dst)

    angular_cache = {}
    channel_tensors = []
    d_channel_tensors = []
    for channel in channels:
        angular_l = int(channel["l"])
        radial_n = int(channel["radial_channel"])
        neighbor_species = str(channel["neighbor_species"])
        if neighbor_species not in species_index:
            raise ValueError(
                f"Channel neighbor_species {neighbor_species!r} is absent from atom_types."
            )
        if angular_l not in angular_cache:
            angular_value, angular_deriv = _ANGULAR_BASIS.cartesian_derivative(angular_l, disp)
            angular_cache[angular_l] = (angular_value.transpose(0, 1), angular_deriv.transpose(0, 1))
        angular, d_angular = angular_cache[angular_l]
        width = int(angular.shape[-1])
        radial_column = radial_values[:, radial_n].to(angular.dtype)
        radial_deriv_column = radial_deriv[:, radial_n].to(angular.dtype)
        d_radial_column = radial_deriv_column.unsqueeze(-1) * d_dist_d_disp.to(angular.dtype)

        value = radial_column.unsqueeze(-1) * angular
        d_value = (
            d_radial_column.unsqueeze(1) * angular.unsqueeze(-1)
            + radial_column.unsqueeze(-1).unsqueeze(-1) * d_angular
        )

        species_mask = (dst_species == species_index[neighbor_species]).unsqueeze(-1)
        value = torch.where(species_mask, value, torch.zeros_like(value))
        d_value = torch.where(species_mask.unsqueeze(-1), d_value, torch.zeros_like(d_value))
        if width < max_width:
            pad = value.new_zeros(tuple(value.shape[:-1]) + (max_width - width,))
            value = torch.cat((value, pad), dim=-1)
            d_pad = d_value.new_zeros(tuple(d_value.shape[:-2]) + (max_width - width, 3))
            d_value = torch.cat((d_value, d_pad), dim=-2)
        channel_tensors.append(value)
        d_channel_tensors.append(d_value)
    phi = torch.stack(channel_tensors, dim=1)
    dphi = torch.stack(d_channel_tensors, dim=1)
    return edge_index, disp, phi, dphi


def _inverse_real_form_matrix(record):
    real_form_id = str(record["real_form_id"])
    cached = _INVERSE_REAL_FORM_CACHE.get(real_form_id)
    if cached is not None:
        return cached
    rows = record["real_to_complex_matrix"]
    width = len(rows)
    matrix = torch.zeros((width, width), dtype=torch.complex128)
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            matrix[row_index, col_index] = _binary_complex(value)
    inverse = torch.linalg.inv(matrix)
    residual = float((matrix @ inverse - torch.eye(width, dtype=torch.complex128)).abs().max())
    if residual > 1.0e-14:
        raise ValueError("real_to_complex_matrix inverse failed the identity check to 1e-14.")
    _INVERSE_REAL_FORM_CACHE[real_form_id] = (matrix, inverse)
    return matrix, inverse


def _complex_to_real_by_channel_list(phi, channel_real_form_ids, real_form_records):
    """Shared core of :func:`complex_to_artifact_real`, parameterized by an
    explicit per-channel-position ``real_form_id`` list and the records
    needed to resolve them, rather than one compiled artifact's own
    payload (``_artifact_channels(compiled)`` +
    ``compiled.payload["real_forms"]``/``["channel_real_form_ids"]``).

    WP4d: this is what lets a merged (multi-content) real moment program's
    evaluator do the same complex-to-real conversion at the ARM's global
    channel granularity, with no single compiled artifact to consult --
    ``channel_real_form_ids[position]`` is the ``real_form_id`` for
    ``phi``'s channel axis position ``position`` (i.e. already aligned,
    unlike ``complex_to_artifact_real`` which re-derives this alignment
    from one artifact's own dense ``channel_index`` ordering).
    """

    if int(phi.shape[-2]) != len(channel_real_form_ids):
        raise ValueError("phi channel axis does not match the channel_real_form_ids length.")
    max_width = int(phi.shape[-1])
    channel_tensors = []
    residuals = []
    for position, real_form_id in enumerate(channel_real_form_ids):
        record = real_form_records[str(real_form_id)]
        _, inverse = _inverse_real_form_matrix(record)
        width = int(inverse.shape[0])
        complex_slice = phi[..., position, :width].to(inverse.dtype)
        real_value = complex_slice @ inverse.T
        if real_value.numel():
            residuals.append(float(real_value.imag.detach().abs().max()))
        real_part = real_value.real
        if width < max_width:
            pad = real_part.new_zeros(tuple(real_part.shape[:-1]) + (max_width - width,))
            real_part = torch.cat((real_part, pad), dim=-1)
        channel_tensors.append(real_part)
    stacked = torch.stack(channel_tensors, dim=-2)
    if residuals:
        scale = max(1.0, float(phi.detach().abs().max()))
        if max(residuals) > 1.0e-12 * scale:
            raise ValueError("complex_to_artifact_real has a material imaginary residual.")
    return stacked


def complex_to_artifact_real(phi, compiled):
    """Convert complex Condon-Shortley primitives to the artifact real form.

    For each channel, ``real = phi_complex @ inv(real_to_complex_matrix)^T``
    with the inverse computed once (and identity-checked to 1e-14) in
    complex128.  The result must be real to 1e-12 relative to its own
    scale; the imaginary part is asserted small and discarded.
    """

    compiled = _load_compiled_artifact(compiled)
    channels = _artifact_channels(compiled)
    forms = {str(record["real_form_id"]): record for record in compiled.payload["real_forms"]}
    bindings = {
        int(record["channel_index"]): str(record["real_form_id"])
        for record in compiled.payload["channel_real_form_ids"]
    }
    channel_real_form_ids = [bindings[int(channel["channel_index"])] for channel in channels]
    return _complex_to_real_by_channel_list(phi, channel_real_form_ids, forms)


def resolve_tag_support_selection(compiled, role_bindings):
    """Return (selection, source) using the WP1 tag_support_report oracle."""

    report = tag_support_report(compiled, role_bindings)
    return tuple(int(value) for value in report["supported_indices"]), "wp1_tag_support_report"


class TaggedTupleEvaluator:
    """Pool a compiled Cauchy artifact over ordered distinct neighbor tuples.

    ``role_bindings`` is a tuple of pairs, one per artifact role index:
    ``("edge", h)`` binds role r to tuple slot h (0..k-1), ``("density",
    c)`` binds role r to the center's full density A_i.  Tuples are the
    batch dimension consumed by
    :meth:`ye3t_ace.lifted_cauchy_linear.LiftedCauchyTorchEvaluator.evaluate`.
    Formula (T1) in the handoff notes; see :class:`TaggedMomentEvaluator`
    for the equivalent formula (T2) exact moment reduction.
    """

    def __init__(self, compiled, role_bindings, budget=2_000_000, combination_matrix=None):
        self.compiled = _load_compiled_artifact(compiled)
        role_dimension = int(self.compiled.payload["role_dimension"])
        normalized = normalize_role_bindings(role_bindings, role_dimension)
        self.role_bindings = normalized["role_bindings"]
        self.tag_count = int(normalized["tag_count"])
        self.budget = int(budget)
        self._inner = LiftedCauchyTorchEvaluator(self.compiled)
        self.descriptor_selection = None
        self.combination_matrix = (
            None if combination_matrix is None else _real_float64_tensor(combination_matrix, "combination_matrix")
        )

    @property
    def descriptor_count(self):
        if self.combination_matrix is not None:
            return int(self.combination_matrix.shape[0])
        if self.descriptor_selection is None:
            return self._inner.descriptor_count
        return len(self.descriptor_selection)

    def descriptors(self, positions, atom_types, cell, pbc, primitives):
        del cell, pbc
        edge_index, _disp, phi_complex = primitives
        n_atoms = int(positions.shape[0])
        if int(atom_types.shape[0]) != n_atoms:
            raise ValueError("atom_types must have one entry per atom.")
        src = edge_index[0]
        n_channels = int(phi_complex.shape[1])
        max_width = int(phi_complex.shape[2])
        role_dimension = len(self.role_bindings)

        phi_real = complex_to_artifact_real(phi_complex, self.compiled)
        A_complex = phi_complex.new_zeros((n_atoms, n_channels, max_width))
        if int(src.numel()):
            A_complex.index_add_(0, src, phi_complex)
        A_real = complex_to_artifact_real(A_complex, self.compiled)

        k = self.tag_count
        src_list = src.detach().cpu().tolist()
        by_center = [[] for _ in range(n_atoms)]
        for row, center in enumerate(src_list):
            by_center[center].append(row)

        expected_total = sum(
            ordered_distinct_tuple_count(len(rows), k) for rows in by_center
        )
        if expected_total > self.budget:
            raise ValueError(
                f"Tagged tuple budget exceeded: {expected_total} tuples > budget {self.budget}."
            )

        columns = self.descriptor_count
        # Connects the (all-zero, by construction) result back into the
        # positions autograd graph: a plain new_zeros tensor here has no
        # graph history at all, so torch.autograd.grad(energy, positions)
        # would raise ("one of the differentiated tensors appears to not
        # have been used in the graph") on a structure where every center
        # has fewer neighbor occurrences than the tag count, even though
        # the correct force there is exactly zero. Adding a graph-connected
        # zero (positions.sum() * 0.0) makes autograd return an exact-zero
        # gradient instead of raising.
        zero_link = positions.sum() * 0.0
        if expected_total == 0:
            return positions.new_zeros((n_atoms, columns)) + zero_link

        tuple_centers = []
        tuple_edge_rows = []
        for center, rows in enumerate(by_center):
            for combo in itertools.permutations(rows, k):
                tuple_centers.append(center)
                tuple_edge_rows.append(combo)

        tuple_centers_t = torch.tensor(tuple_centers, dtype=torch.long)
        role_slices = []
        for kind, value in self.role_bindings:
            if kind == "edge":
                rows_for_role = torch.tensor(
                    [combo[value] for combo in tuple_edge_rows], dtype=torch.long
                )
                role_slices.append(phi_real.index_select(0, rows_for_role))
            else:
                role_slices.append(A_real.index_select(0, tuple_centers_t))
        density = torch.stack(role_slices, dim=2)
        if tuple(density.shape) != (len(tuple_centers), n_channels, role_dimension, max_width):
            raise RuntimeError("Assembled tuple density has an unexpected shape.")

        raw = self._inner.evaluate(density)
        if self.descriptor_selection is not None:
            selection_t = torch.tensor(self.descriptor_selection, dtype=torch.long)
            raw = raw.index_select(1, selection_t)

        out = raw.new_zeros((n_atoms, raw.shape[1]))
        out.index_add_(0, tuple_centers_t, raw)
        out = out + zero_link
        if self.combination_matrix is not None:
            if int(out.shape[1]) != int(self.combination_matrix.shape[1]):
                raise ValueError(
                    "combination_matrix column count must equal the "
                    "(selected) descriptor count."
                )
            out = out @ self.combination_matrix.T
        return out


def _set_partitions_of(elements):
    """Yield every set partition of ``elements`` as a tuple of block tuples.

    Own small implementation of the classical restricted-growth recursion
    (own write scope; WP1's equivalent private helper is not part of its
    public interface). ``elements=()`` yields exactly the empty partition
    ``()`` (zero blocks), matching the p=0 convention used below.
    """

    elements = tuple(elements)
    if not elements:
        yield ()
        return
    first, rest = elements[0], elements[1:]
    for partition in _set_partitions_of(rest):
        yield ((first,),) + partition
        for index in range(len(partition)):
            yield partition[:index] + ((first,) + partition[index],) + partition[index + 1 :]


def _partition_mobius_weight(partition):
    """Mobius function of the partition lattice: prod_S (-1)^(|S|-1)(|S|-1)!."""

    result = 1
    for block in partition:
        size = len(block)
        result *= (-1) ** (size - 1) * math.factorial(size - 1)
    return result


class _MomentProgram:
    """Compiled, interned formula-(T2) program for one (artifact, bindings, selection).

    Built once from ``compiled.payload["descriptors"]`` restricted to
    ``descriptor_selection`` (or all descriptors if ``None``).  Every
    canonical term is split into: density-role coordinates (interned as
    individual ``(channel, mi)`` keys), and edge-role coordinates grouped by
    tag h into ``F_h`` multisets.  Every set partition of the term's present
    tags ``P_t`` contributes one ``(weight, moment_index per block)`` row,
    where each block's moment key is the sorted combined multiset of its
    tags' ``F_h`` entries (interned globally, so identical moments across
    different terms/descriptors are computed once).
    """

    def __init__(self, compiled, role_bindings, descriptor_selection):
        role_dimension = int(compiled.payload["role_dimension"])
        normalized = normalize_role_bindings(role_bindings, role_dimension)
        self.role_bindings = normalized["role_bindings"]
        self.tag_count = int(normalized["tag_count"])
        role_tag_of = {role_index: h for h, role_index in enumerate(normalized["edge_roles"])}

        descriptors = compiled.payload["descriptors"]
        if descriptor_selection is None:
            output_position_of = {index: index for index in range(len(descriptors))}
            self.descriptor_count = len(descriptors)
        else:
            output_position_of = {
                int(original): local for local, original in enumerate(descriptor_selection)
            }
            self.descriptor_count = len(descriptor_selection)

        moment_index = {}
        self.moment_keys = []
        density_index = {}
        self.density_keys = []
        self.term_descriptor = []
        self.term_coefficient = []
        self.term_p = []
        self.term_density_factor_indices = []
        self.term_partition_entries = []

        for original_index, descriptor in enumerate(descriptors):
            if original_index not in output_position_of:
                continue
            output_position = output_position_of[original_index]
            for term in descriptor["canonical_terms"]:
                coefficient = _binary_complex(term["coefficient"])
                tag_coordinates = {}
                density_factor_indices = []
                for coordinate in term["coordinates"]:
                    channel, role, mi = (int(value) for value in coordinate)
                    if role in role_tag_of:
                        tag_coordinates.setdefault(role_tag_of[role], []).append((channel, mi))
                    else:
                        key = (channel, mi)
                        idx = density_index.get(key)
                        if idx is None:
                            idx = len(self.density_keys)
                            density_index[key] = idx
                            self.density_keys.append(key)
                        density_factor_indices.append(idx)
                tag_positions = tuple(sorted(tag_coordinates))
                p = len(tag_positions)
                partition_entries = []
                for partition in _set_partitions_of(tag_positions):
                    weight = _partition_mobius_weight(partition)
                    block_moment_indices = []
                    for block in partition:
                        combined = []
                        for h in block:
                            combined.extend(tag_coordinates[h])
                        key = tuple(sorted(combined))
                        idx = moment_index.get(key)
                        if idx is None:
                            idx = len(self.moment_keys)
                            moment_index[key] = idx
                            self.moment_keys.append(key)
                        block_moment_indices.append(idx)
                    partition_entries.append((int(weight), tuple(block_moment_indices)))
                self.term_descriptor.append(output_position)
                self.term_coefficient.append(coefficient)
                self.term_p.append(p)
                self.term_density_factor_indices.append(tuple(density_factor_indices))
                self.term_partition_entries.append(tuple(partition_entries))

        self.term_count = len(self.term_descriptor)
        self._build_evaluation_tensors()

    def _build_evaluation_tensors(self):
        by_degree = {}
        for idx, key in enumerate(self.moment_keys):
            by_degree.setdefault(len(key), []).append(idx)
        self._moment_buckets = {}
        for degree, indices in by_degree.items():
            if degree == 0:
                continue
            channels = torch.tensor(
                [[self.moment_keys[idx][j][0] for j in range(degree)] for idx in indices],
                dtype=torch.long,
            )
            mis = torch.tensor(
                [[self.moment_keys[idx][j][1] for j in range(degree)] for idx in indices],
                dtype=torch.long,
            )
            self._moment_buckets[degree] = (torch.tensor(indices, dtype=torch.long), channels, mis)

        if self.density_keys:
            self._density_channels = torch.tensor([key[0] for key in self.density_keys], dtype=torch.long)
            self._density_mis = torch.tensor([key[1] for key in self.density_keys], dtype=torch.long)
        else:
            self._density_channels = torch.zeros((0,), dtype=torch.long)
            self._density_mis = torch.zeros((0,), dtype=torch.long)

        self._term_descriptor_index = torch.tensor(self.term_descriptor, dtype=torch.long)
        self._term_coefficient_t = torch.tensor(self.term_coefficient, dtype=torch.complex128)

        density_buckets = {}
        for term_index, indices in enumerate(self.term_density_factor_indices):
            density_buckets.setdefault(len(indices), []).append((term_index, indices))
        self._density_buckets = {}
        for count, rows in density_buckets.items():
            term_idx = torch.tensor([row[0] for row in rows], dtype=torch.long)
            if count:
                factor_idx = torch.tensor([row[1] for row in rows], dtype=torch.long)
            else:
                factor_idx = torch.zeros((len(rows), 0), dtype=torch.long)
            self._density_buckets[count] = (term_idx, factor_idx)

        p_buckets = {}
        for term_index, p in enumerate(self.term_p):
            p_buckets.setdefault(p, []).append(term_index)
        self._p_buckets = {p: torch.tensor(rows, dtype=torch.long) for p, rows in p_buckets.items()}

        partition_buckets = {}
        for term_index, entries in enumerate(self.term_partition_entries):
            for weight, moment_indices in entries:
                b = len(moment_indices)
                partition_buckets.setdefault(b, []).append((term_index, weight, moment_indices))
        self._partition_buckets = {}
        for b, rows in partition_buckets.items():
            term_idx = torch.tensor([row[0] for row in rows], dtype=torch.long)
            weight = torch.tensor([row[1] for row in rows], dtype=torch.complex128)
            if b:
                moment_idx = torch.tensor([row[2] for row in rows], dtype=torch.long)
            else:
                moment_idx = torch.zeros((len(rows), 0), dtype=torch.long)
            self._partition_buckets[b] = (term_idx, weight, moment_idx)

    def summary(self):
        degrees = [len(key) for key in self.moment_keys]
        return {
            "term_count": int(self.term_count),
            "distinct_moment_count": len(self.moment_keys),
            "distinct_density_factor_count": len(self.density_keys),
            "max_moment_degree": max(degrees) if degrees else 0,
            "moment_degree_buckets": {
                int(d): int(v[0].numel()) for d, v in self._moment_buckets.items()
            },
            "density_factor_count_buckets": {
                int(c): int(v[0].numel()) for c, v in self._density_buckets.items()
            },
            "partition_block_count_buckets": {
                int(b): int(v[0].numel()) for b, v in self._partition_buckets.items()
            },
            "p_buckets": {int(p): int(v.numel()) for p, v in self._p_buckets.items()},
            "output_descriptor_count": int(self.descriptor_count),
        }

    def evaluate_complex(self, phi_complex, src, n_atoms):
        n_channels = int(phi_complex.shape[1])
        max_width = int(phi_complex.shape[2])
        edges = int(phi_complex.shape[0])

        A = phi_complex.new_zeros((n_atoms, n_channels, max_width))
        if edges:
            A.index_add_(0, src, phi_complex)
        z = torch.zeros(n_atoms, dtype=torch.float64)
        if edges:
            ones = torch.ones(edges, dtype=torch.float64)
            z.index_add_(0, src, ones)

        n_moments = len(self.moment_keys)
        M = phi_complex.new_zeros((n_atoms, n_moments))
        for degree, (global_idx, channels, mis) in self._moment_buckets.items():
            factor = phi_complex.new_ones((edges, channels.shape[0]))
            for j in range(degree):
                factor = factor * phi_complex[:, channels[:, j], mis[:, j]]
            pooled = phi_complex.new_zeros((n_atoms, channels.shape[0]))
            if edges:
                pooled.index_add_(0, src, factor)
            M.index_copy_(1, global_idx, pooled)

        if self.density_keys:
            A_gathered = A[:, self._density_channels, self._density_mis]
        else:
            A_gathered = A.new_zeros((n_atoms, 0))

        density_product = phi_complex.new_ones((n_atoms, self.term_count))
        for count, (term_idx, factor_idx) in self._density_buckets.items():
            if count == 0:
                continue
            block = phi_complex.new_ones((n_atoms, term_idx.shape[0]))
            for j in range(count):
                block = block * A_gathered[:, factor_idx[:, j]]
            density_product.index_copy_(1, term_idx, block)

        valid_mask = z >= float(self.tag_count)
        free_factor_real = torch.ones((n_atoms, self.term_count), dtype=torch.float64)
        for p, term_idx in self._p_buckets.items():
            count = self.tag_count - p
            top = z - float(p)
            value = torch.ones(n_atoms, dtype=torch.float64)
            for i in range(count):
                value = value * (top - i)
            value = torch.where(valid_mask, value, torch.zeros_like(value))
            free_factor_real.index_copy_(1, term_idx, value.unsqueeze(1).expand(n_atoms, term_idx.shape[0]))

        partition_sum = phi_complex.new_zeros((n_atoms, self.term_count))
        for b, (term_idx, weight, moment_idx) in self._partition_buckets.items():
            block_product = phi_complex.new_ones((n_atoms, term_idx.shape[0]))
            for j in range(b):
                block_product = block_product * M[:, moment_idx[:, j]]
            contribution = block_product * weight.unsqueeze(0)
            partition_sum.index_add_(1, term_idx, contribution)

        term_value = (
            self._term_coefficient_t.unsqueeze(0)
            * density_product
            * free_factor_real.to(phi_complex.dtype)
            * partition_sum
        )
        descriptors_complex = phi_complex.new_zeros((n_atoms, self.descriptor_count))
        descriptors_complex.index_add_(1, self._term_descriptor_index, term_value)
        return descriptors_complex

    def export_payload(self):
        return {
            "tag_count": int(self.tag_count),
            "output_descriptor_count": int(self.descriptor_count),
            "moment_keys": [[[int(c), int(m)] for c, m in key] for key in self.moment_keys],
            "density_keys": [[int(c), int(m)] for c, m in self.density_keys],
            "terms": [
                {
                    "descriptor_index": int(self.term_descriptor[i]),
                    "coefficient": [
                        float(self.term_coefficient[i].real),
                        float(self.term_coefficient[i].imag),
                    ],
                    "density_factor_indices": [int(v) for v in self.term_density_factor_indices[i]],
                    "p": int(self.term_p[i]),
                    "partition_entries": [
                        [int(weight), [int(v) for v in moment_indices]]
                        for weight, moment_indices in self.term_partition_entries[i]
                    ],
                }
                for i in range(self.term_count)
            ],
        }


class TaggedMomentEvaluator:
    """Exact set-partition moment reduction (formula T2) of the same pooling.

    Agrees with :class:`TaggedTupleEvaluator` for every descriptor and
    center (including unsupported descriptors, via the same
    per-center falling-factorial free count), without ever enumerating
    ordered tuples; see :func:`moment_equivalence_certificate`.  Operates
    directly on the complex Condon-Shortley primitives (no real-form
    round trip is needed here). ``combination_matrix`` (features x
    selected-descriptors, real or complex) is applied after selection and
    after the reality check, so a future pooled basis can be plugged in.
    """

    def __init__(self, compiled, role_bindings, descriptor_selection=None, combination_matrix=None):
        self.compiled = _load_compiled_artifact(compiled)
        self.descriptor_selection = (
            None if descriptor_selection is None else tuple(int(value) for value in descriptor_selection)
        )
        self.program = _MomentProgram(self.compiled, role_bindings, self.descriptor_selection)
        self.role_bindings = self.program.role_bindings
        self.tag_count = self.program.tag_count
        if combination_matrix is None:
            self.combination_matrix = None
        else:
            matrix = torch.as_tensor(combination_matrix)
            self.combination_matrix = matrix.to(
                dtype=torch.complex128 if torch.is_complex(matrix) else torch.float64
            )
            if int(self.combination_matrix.shape[1]) != self.program.descriptor_count:
                raise ValueError("combination_matrix column count must equal the selected descriptor count.")

    @property
    def descriptor_count(self):
        if self.combination_matrix is not None:
            return int(self.combination_matrix.shape[0])
        return self.program.descriptor_count

    def program_summary(self):
        return self.program.summary()

    def descriptors(self, positions, atom_types, cell, pbc, primitives):
        del cell, pbc
        edge_index, _disp, phi_complex = primitives
        n_atoms = int(positions.shape[0])
        if int(atom_types.shape[0]) != n_atoms:
            raise ValueError("atom_types must have one entry per atom.")
        src = edge_index[0]
        raw = self.program.evaluate_complex(phi_complex, src, n_atoms)
        if raw.numel():
            residual = float(raw.imag.detach().abs().max())
            scale = max(1.0, float(raw.detach().abs().max()))
            if residual > 1.0e-10 * scale:
                raise ValueError("TaggedMomentEvaluator produced a material imaginary residual.")
        # Same graph-connection guard as TaggedTupleEvaluator: when there are
        # no edges at all (an isolated structure), every quantity inside
        # evaluate_complex is built from new_zeros/new_ones calls with no
        # history, so the all-zero result would otherwise be disconnected
        # from positions and torch.autograd.grad would raise instead of
        # returning an exact-zero force.  Harmless (adds exactly 0.0) in the
        # normal case.
        real = raw.real + (positions.sum() * 0.0).to(raw.real.dtype)
        if self.combination_matrix is not None:
            real = real.to(self.combination_matrix.dtype) @ self.combination_matrix.T
        return real


def _exact_scalar_from_payload(terms):
    """Exact sympy value of one exact-payload scalar component (a list of
    ``{coefficient_numerator, coefficient_denominator, radicand_numerator,
    radicand_denominator}`` terms summing to ``sum (a/b)*sqrt(p/q)``; an
    empty list is exactly 0).  Matches the payload convention emitted by
    ``ye3t.couplings.lifted_cauchy_scalar._exact_scalar_payload`` (confirmed
    by direct inspection of a compiled artifact's JSON; reimplemented
    locally here rather than importing that private helper across the
    package boundary, matching this module's existing convention of
    reimplementing small helpers such as ``_binary_complex`` rather than
    reaching into ``ye3t``'s private internals).
    """

    total = _sympy.Integer(0)
    for term in terms:
        coefficient = _sympy.Rational(
            int(term["coefficient_numerator"]), int(term["coefficient_denominator"])
        )
        radicand = _sympy.Rational(
            int(term["radicand_numerator"]), int(term["radicand_denominator"])
        )
        total = total + coefficient * _sympy.sqrt(radicand)
    return total


def _exact_complex_pair_from_payload(payload):
    """Exact ``(real, imaginary)`` sympy pair for one ``{real, imag,
    binary64}`` exact-payload component."""

    return (
        _exact_scalar_from_payload(payload.get("real", [])),
        _exact_scalar_from_payload(payload.get("imag", [])),
    )


def _exact_complex_multiply(a, b):
    ar, ai = a
    br, bi = b
    return (ar * br - ai * bi, ar * bi + ai * br)


def _exact_complex_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _exact_complex_is_zero(pair):
    return pair[0] == 0 and pair[1] == 0


_EXACT_REAL_FORM_CACHE = {}


def _exact_real_form_rows(record):
    """Exact sparse rows of one artifact ``real_to_complex_matrix``.

    Returns a tuple indexed by complex component ``mi`` (the matrix row, per
    ``complex[mi] = sum_a matrix[mi, a] * real[a]``; this orientation is the
    one used by ``_inverse_real_form_matrix``/``complex_to_artifact_real``
    above), each entry a tuple of ``(real component index a, exact (re, im)
    pair)`` for the nonzero entries of that row.  ``_real_form_matrix`` in
    ``ye3t.couplings.lifted_cauchy_scalar`` gives every row at most two
    nonzero entries, so a degree-d product of components expands into at
    most ``2**d`` real monomials below, not ``(2l+1)**d``.  Cached by
    ``real_form_id`` (module-level; real forms are shared across many
    artifacts/channels, so this keeps the exact sympy parsing itself a
    one-time cost per distinct real form, independent of how many times
    ``real_moment_program`` is called).
    """

    real_form_id = str(record["real_form_id"])
    cached = _EXACT_REAL_FORM_CACHE.get(real_form_id)
    if cached is not None:
        return cached
    sparse_rows = []
    for row in record["real_to_complex_matrix"]:
        entries = []
        for column_index, payload in enumerate(row):
            pair = _exact_complex_pair_from_payload(payload)
            if not _exact_complex_is_zero(pair):
                entries.append((column_index, pair))
        sparse_rows.append(tuple(entries))
    result = tuple(sparse_rows)
    _EXACT_REAL_FORM_CACHE[real_form_id] = result
    return result


def _exact_channel_expansions(compiled):
    """Per artifact ``channel_index``: exact sparse real-form rows.

    ``expansions[channel][mi]`` is a tuple of ``(a, (re, im))`` giving the
    exact expansion of complex component ``mi`` of ``channel`` as a sum of
    real components ``a`` of the same channel.
    """

    forms = {str(record["real_form_id"]): record for record in compiled.payload["real_forms"]}
    bindings = {
        int(record["channel_index"]): str(record["real_form_id"])
        for record in compiled.payload["channel_real_form_ids"]
    }
    return {
        channel_index: _exact_real_form_rows(forms[real_form_id])
        for channel_index, real_form_id in bindings.items()
    }


def _expand_real_factor_list(factor_list, expansions):
    """Expand a product of complex ``(channel, mi)`` primitive components
    into real ``(channel, a)`` monomials.

    ``factor_list`` is a list (with multiplicity and order, as encountered)
    of complex ``(channel, mi)`` factors drawn from one canonical term
    (either its density coordinates, or one set-partition block's combined
    tag coordinates).  Substituting each factor's exact real expansion and
    distributing the product over the (at most two, per factor) choices
    gives, after merging options that collapse onto the same sorted
    ``(channel, a)`` multiset, a dict ``{sorted key: exact (re, im)
    coefficient}``.  An empty ``factor_list`` is the empty product and
    returns ``{(): (1, 0)}``.
    """

    options_per_factor = [
        [(channel, a, pair) for a, pair in expansions[channel][mi]] for channel, mi in factor_list
    ]
    one = (_sympy.Integer(1), _sympy.Integer(0))
    zero = (_sympy.Integer(0), _sympy.Integer(0))
    merged = {}
    for combo in itertools.product(*options_per_factor):
        key = tuple(sorted((channel, a) for channel, a, _pair in combo))
        coefficient = one
        for _channel, _a, pair in combo:
            coefficient = _exact_complex_multiply(coefficient, pair)
        merged[key] = _exact_complex_add(merged.get(key, zero), coefficient)
    return merged


def real_moment_program(compiled, role_bindings, selection, combination_matrix=None):
    """Exact real-arithmetic ("real tesseral") lowering of the moment program.

    WP3c: the native (ye3t-lammps) evaluator works in real arithmetic over
    real tesseral components, while :class:`_MomentProgram` (WP3b) is built
    over complex Condon-Shortley components with complex coefficients (see
    ``ye3t_ace.lifted_cauchy_io._physical_real_descriptor_rows`` for the
    established, binary64/tolerance-based reference pattern this mirrors in
    spirit but not in arithmetic: every step here is exact sympy rational
    and square-root-of-rational arithmetic, per the compiled artifact's
    exact ``{real, imag, binary64}`` payload convention).

    For every channel, the artifact stores ``real_to_complex_matrix`` U with
    ``complex[mi] = sum_a U[mi, a] * real[a]``.  Substituting this into a
    canonical term's coordinates: a density factor ``(channel, mi)`` becomes
    the real linear form ``sum_a U[mi, a] * A_real[channel][a]``; the
    combined tag-coordinate multiset of one set-partition block becomes the
    product of the corresponding real linear forms in the real edge
    components.  Expanding these products (:func:`_expand_real_factor_list`)
    and grouping the result by ``(feature index, real density monomial, p,
    real moment id tuple)`` (so that any imaginary parts from different
    contributions to the same real monomial cancel against each other
    exactly, not merely to within a tolerance) gives, for every group, an
    exact complex value whose imaginary part is asserted to be *exactly*
    zero (a symbolic sympy ``== 0`` decision after ``expand``/``simplify``,
    never a magnitude tolerance) before its real part is collapsed to
    binary64.  If any group's imaginary part is not exactly zero, this
    raises immediately with the offending group identified; it never drops
    or rounds a term.

    ``combination_matrix`` (features x selected-descriptors, real or
    numerically-real complex, following :class:`TaggedMomentEvaluator`'s own
    convention) is folded in as a final real linear combination over the
    per-original-descriptor exact real terms, strictly after the exact
    lowering and zero-imaginary-residual check above, so the emitted
    program's ``feature_index`` already addresses the pooled features
    directly (``combination_matrix=None`` leaves one feature per selected
    descriptor, i.e. the identity).

    Returns a dict with ``tag_count``, ``feature_count``,
    ``real_density_keys`` (interned individual ``(channel, a)`` factors),
    ``real_moment_keys`` (interned combined-per-block ``(channel, a)``
    tuples), ``terms`` (flat rows: ``feature_index``, binary64
    ``coefficient``, ``density_factor_indices``, ``p``, ``moment_indices``;
    unlike :class:`_MomentProgram`'s ``terms`` this is already flattened one
    row per elementary product -- the original per-descriptor canonical
    term, the chosen set partition, and the chosen real expansion option are
    no longer distinguishable, but summation is associative so this changes
    nothing observable), and ``lowering_certificate`` (term counts before
    lowering / after grouping / after folding in ``combination_matrix``, and
    the measured imaginary magnitude prior to the exactness proof).
    """

    compiled = _load_compiled_artifact(compiled)
    role_dimension = int(compiled.payload["role_dimension"])
    normalized = normalize_role_bindings(role_bindings, role_dimension)
    tag_count = int(normalized["tag_count"])
    role_tag_of = {role_index: h for h, role_index in enumerate(normalized["edge_roles"])}

    descriptors = compiled.payload["descriptors"]
    if selection is None:
        output_position_of = {index: index for index in range(len(descriptors))}
        n_selected = len(descriptors)
    else:
        selection = tuple(int(value) for value in selection)
        output_position_of = {int(original): local for local, original in enumerate(selection)}
        n_selected = len(selection)

    expansions = _exact_channel_expansions(compiled)

    real_density_index = {}
    real_density_keys = []
    real_moment_index = {}
    real_moment_keys = []
    grouped = {}
    zero_pair = (_sympy.Integer(0), _sympy.Integer(0))

    term_count_before = 0
    for original_index, descriptor in enumerate(descriptors):
        if original_index not in output_position_of:
            continue
        output_position = output_position_of[original_index]
        for term in descriptor["canonical_terms"]:
            term_count_before += 1
            coefficient = _exact_complex_pair_from_payload(term["coefficient"])
            tag_coordinates = {}
            density_factors = []
            for coordinate in term["coordinates"]:
                channel, role, mi = (int(value) for value in coordinate)
                if role in role_tag_of:
                    tag_coordinates.setdefault(role_tag_of[role], []).append((channel, mi))
                else:
                    density_factors.append((channel, mi))
            tag_positions = tuple(sorted(tag_coordinates))
            p = len(tag_positions)

            density_options = _expand_real_factor_list(density_factors, expansions)
            interned_density_options = []
            for key, coeff in density_options.items():
                idx_list = []
                for channel, a in key:
                    dkey = (channel, a)
                    idx = real_density_index.get(dkey)
                    if idx is None:
                        idx = len(real_density_keys)
                        real_density_index[dkey] = idx
                        real_density_keys.append(dkey)
                    idx_list.append(idx)
                interned_density_options.append((tuple(sorted(idx_list)), coeff))

            for partition in _set_partitions_of(tag_positions):
                weight = _partition_mobius_weight(partition)
                weight_pair = (_sympy.Integer(int(weight)), _sympy.Integer(0))
                block_option_lists = []
                for block in partition:
                    combined = []
                    for h in block:
                        combined.extend(tag_coordinates[h])
                    block_options = _expand_real_factor_list(combined, expansions)
                    interned_block_options = []
                    for key, coeff in block_options.items():
                        idx = real_moment_index.get(key)
                        if idx is None:
                            idx = len(real_moment_keys)
                            real_moment_index[key] = idx
                            real_moment_keys.append(key)
                        interned_block_options.append((idx, coeff))
                    block_option_lists.append(interned_block_options)

                for moment_combo in itertools.product(*block_option_lists):
                    moment_idx_tuple = tuple(sorted(idx for idx, _coeff in moment_combo))
                    moment_coeff = (_sympy.Integer(1), _sympy.Integer(0))
                    for _idx, coeff in moment_combo:
                        moment_coeff = _exact_complex_multiply(moment_coeff, coeff)
                    partial = _exact_complex_multiply(coefficient, weight_pair)
                    partial = _exact_complex_multiply(partial, moment_coeff)
                    for density_idx_tuple, density_coeff in interned_density_options:
                        final_coeff = _exact_complex_multiply(partial, density_coeff)
                        group_key = (output_position, density_idx_tuple, p, moment_idx_tuple)
                        grouped[group_key] = _exact_complex_add(
                            grouped.get(group_key, zero_pair), final_coeff
                        )

    max_imaginary_magnitude = 0.0
    rows = []
    for (descriptor_index, density_idx_tuple, p, moment_idx_tuple), (re_val, im_val) in grouped.items():
        im_expanded = _sympy.expand(im_val)
        max_imaginary_magnitude = max(max_imaginary_magnitude, abs(float(im_expanded.evalf(30))))
        is_zero = im_expanded == 0
        if not is_zero:
            im_simplified = _sympy.simplify(im_expanded)
            is_zero = im_simplified == 0
            if is_zero:
                im_expanded = im_simplified
        if not is_zero:
            raise ValueError(
                "real_moment_program: exact imaginary residual is not zero for "
                f"descriptor {descriptor_index}, density_factors {density_idx_tuple}, "
                f"p={p}, moment_blocks {moment_idx_tuple}: {im_expanded}"
            )
        re_expanded = _sympy.expand(re_val)
        rows.append(
            {
                "descriptor_index": int(descriptor_index),
                "coefficient": float(re_expanded.evalf()),
                "density_factor_indices": list(density_idx_tuple),
                "p": int(p),
                "moment_indices": list(moment_idx_tuple),
            }
        )

    term_count_after_lowering = len(rows)

    if combination_matrix is None:
        feature_count = n_selected
        terms = [
            {
                "feature_index": row["descriptor_index"],
                "coefficient": row["coefficient"],
                "density_factor_indices": row["density_factor_indices"],
                "p": row["p"],
                "moment_indices": row["moment_indices"],
            }
            for row in rows
        ]
    else:
        combination = (
            combination_matrix.detach().cpu().numpy()
            if isinstance(combination_matrix, torch.Tensor)
            else np.asarray(combination_matrix)
        )
        if np.iscomplexobj(combination):
            scale = max(1.0, float(np.max(np.abs(combination)))) if combination.size else 1.0
            residual = float(np.max(np.abs(combination.imag))) if combination.size else 0.0
            if residual > 1.0e-12 * scale:
                raise ValueError(
                    "real_moment_program: combination_matrix has a material imaginary part."
                )
            combination = combination.real
        combination = np.asarray(combination, dtype=np.float64)
        if combination.ndim != 2 or int(combination.shape[1]) != n_selected:
            raise ValueError(
                "combination_matrix column count must equal the selected descriptor count."
            )
        feature_count = int(combination.shape[0])
        terms = []
        for row in rows:
            weights = combination[:, row["descriptor_index"]]
            for feature_index in np.flatnonzero(weights).tolist():
                terms.append(
                    {
                        "feature_index": int(feature_index),
                        "coefficient": row["coefficient"] * float(weights[feature_index]),
                        "density_factor_indices": row["density_factor_indices"],
                        "p": row["p"],
                        "moment_indices": row["moment_indices"],
                    }
                )

    lowering_certificate = {
        "term_count_before_lowering": int(term_count_before),
        "grouped_term_count": len(grouped),
        "term_count_after_lowering": int(term_count_after_lowering),
        "term_count_after_combination": int(len(terms)),
        "max_imaginary_magnitude": float(max_imaginary_magnitude),
        "imaginary_residual_is_exactly_zero": True,
        "real_density_key_count": len(real_density_keys),
        "real_moment_key_count": len(real_moment_keys),
        "feature_count": int(feature_count),
        "tag_count": int(tag_count),
    }

    return {
        "tag_count": int(tag_count),
        "feature_count": int(feature_count),
        "real_density_keys": [[int(c), int(a)] for c, a in real_density_keys],
        "real_moment_keys": [[[int(c), int(a)] for c, a in key] for key in real_moment_keys],
        "terms": terms,
        "lowering_certificate": lowering_certificate,
    }


class RealMomentEvaluator:
    """Real-arithmetic ("real tesseral") exact moment reduction (WP3c).

    Native-deployment-ready counterpart to :class:`TaggedMomentEvaluator`:
    built from :func:`real_moment_program`'s exact, zero-imaginary-residual-
    verified lowering, so every gather index and coefficient used at
    evaluation time is already real.  Consumes the same ``primitives``
    convention (``edge_index, disp, phi_complex``) as the other evaluators
    for calling-convention consistency, converting to the artifact real form
    internally via :func:`complex_to_artifact_real` (the ordinary edge
    primitives themselves are still built as complex spherical harmonics
    upstream in :func:`ordinary_edge_primitives`; only the *evaluator*
    arithmetic from that point on is real).  Agrees with
    :class:`TaggedMomentEvaluator` to 1e-12 on descriptors and forces for
    k = 1, 2 (see ``tests/test_tagged_cauchy_linear.py``).

    WP4d: also accepts an already-built ``program`` directly (keyword-only),
    for a multi-content *merged* real moment program (see
    ``ye3t_ace.tagged_cauchy_fit.merge_real_moment_programs``) that has no
    single compiled artifact of its own. In that mode, pass ``tag_count``
    (shared across every content of one arm, since ``role_bindings`` --
    hence ``tag_count`` -- is an arm-level, not per-content, property) and
    ``channel_real_form_ids``/``real_form_records`` (the arm's global-
    channel-indexed real-form table, in place of ``compiled``) so
    :meth:`descriptors` can still do the complex-to-real conversion, via
    :func:`_complex_to_real_by_channel_list` instead of
    :func:`complex_to_artifact_real`. ``_build_evaluation_tensors``/
    :meth:`evaluate_real` (the bucketed kernel itself) are used completely
    unchanged either way -- they only ever read ``self.program`` plus
    already-real input tensors, never ``self.compiled``.
    """

    def __init__(
        self,
        compiled=None,
        role_bindings=None,
        descriptor_selection=None,
        combination_matrix=None,
        *,
        program=None,
        tag_count=None,
        channel_real_form_ids=None,
        real_form_records=None,
    ):
        if program is not None:
            self.compiled = None
            self.descriptor_selection = None
            self.program = program
            self.role_bindings = None
            self.tag_count = int(tag_count)
            self._channel_real_form_ids = list(channel_real_form_ids)
            self._real_form_records = dict(real_form_records)
            self._build_evaluation_tensors()
            return

        self.compiled = _load_compiled_artifact(compiled)
        self.descriptor_selection = (
            None if descriptor_selection is None else tuple(int(v) for v in descriptor_selection)
        )
        self._channel_real_form_ids = None
        self._real_form_records = None
        combination_for_program = None
        if combination_matrix is not None:
            matrix = torch.as_tensor(combination_matrix)
            combination_for_program = matrix.detach().cpu().numpy()
        self.program = real_moment_program(
            self.compiled, role_bindings, self.descriptor_selection, combination_for_program
        )
        role_dimension = int(self.compiled.payload["role_dimension"])
        normalized = normalize_role_bindings(role_bindings, role_dimension)
        self.role_bindings = normalized["role_bindings"]
        self.tag_count = int(normalized["tag_count"])
        self._build_evaluation_tensors()

    @property
    def descriptor_count(self):
        return int(self.program["feature_count"])

    def program_summary(self):
        return {
            "feature_count": int(self.program["feature_count"]),
            "tag_count": int(self.program["tag_count"]),
            "term_count": len(self.program["terms"]),
            "real_density_key_count": len(self.program["real_density_keys"]),
            "real_moment_key_count": len(self.program["real_moment_keys"]),
            "lowering_certificate": dict(self.program["lowering_certificate"]),
        }

    def _build_evaluation_tensors(self):
        real_moment_keys = self.program["real_moment_keys"]
        by_degree = {}
        for idx, key in enumerate(real_moment_keys):
            by_degree.setdefault(len(key), []).append(idx)
        self._moment_buckets = {}
        for degree, indices in by_degree.items():
            if degree == 0:
                continue
            channels = torch.tensor(
                [[real_moment_keys[idx][j][0] for j in range(degree)] for idx in indices],
                dtype=torch.long,
            )
            a_indices = torch.tensor(
                [[real_moment_keys[idx][j][1] for j in range(degree)] for idx in indices],
                dtype=torch.long,
            )
            self._moment_buckets[degree] = (torch.tensor(indices, dtype=torch.long), channels, a_indices)

        real_density_keys = self.program["real_density_keys"]
        if real_density_keys:
            self._density_channels = torch.tensor([key[0] for key in real_density_keys], dtype=torch.long)
            self._density_a = torch.tensor([key[1] for key in real_density_keys], dtype=torch.long)
        else:
            self._density_channels = torch.zeros((0,), dtype=torch.long)
            self._density_a = torch.zeros((0,), dtype=torch.long)

        terms = self.program["terms"]
        self.term_count = len(terms)
        if self.term_count:
            self._term_feature_index = torch.tensor([t["feature_index"] for t in terms], dtype=torch.long)
            self._term_coefficient = torch.tensor([t["coefficient"] for t in terms], dtype=torch.float64)
        else:
            self._term_feature_index = torch.zeros((0,), dtype=torch.long)
            self._term_coefficient = torch.zeros((0,), dtype=torch.float64)

        density_buckets = {}
        for term_index, term in enumerate(terms):
            density_buckets.setdefault(len(term["density_factor_indices"]), []).append(
                (term_index, term["density_factor_indices"])
            )
        self._density_buckets = {}
        for count, rows in density_buckets.items():
            term_idx = torch.tensor([row[0] for row in rows], dtype=torch.long)
            if count:
                factor_idx = torch.tensor([row[1] for row in rows], dtype=torch.long)
            else:
                factor_idx = torch.zeros((len(rows), 0), dtype=torch.long)
            self._density_buckets[count] = (term_idx, factor_idx)

        p_buckets = {}
        for term_index, term in enumerate(terms):
            p_buckets.setdefault(term["p"], []).append(term_index)
        self._p_buckets = {p: torch.tensor(rows, dtype=torch.long) for p, rows in p_buckets.items()}

        block_buckets = {}
        for term_index, term in enumerate(terms):
            block_buckets.setdefault(len(term["moment_indices"]), []).append(
                (term_index, term["moment_indices"])
            )
        self._partition_buckets = {}
        for b, rows in block_buckets.items():
            term_idx = torch.tensor([row[0] for row in rows], dtype=torch.long)
            if b:
                moment_idx = torch.tensor([row[1] for row in rows], dtype=torch.long)
            else:
                moment_idx = torch.zeros((len(rows), 0), dtype=torch.long)
            self._partition_buckets[b] = (term_idx, moment_idx)

    def evaluate_real(self, phi_real, A_real, src, n_atoms):
        edges = int(phi_real.shape[0])

        z = torch.zeros(n_atoms, dtype=torch.float64)
        if edges:
            ones = torch.ones(edges, dtype=torch.float64)
            z.index_add_(0, src, ones)

        n_moments = len(self.program["real_moment_keys"])
        M = phi_real.new_zeros((n_atoms, n_moments))
        for degree, (global_idx, channels, a_indices) in self._moment_buckets.items():
            factor = phi_real.new_ones((edges, channels.shape[0]))
            for j in range(degree):
                factor = factor * phi_real[:, channels[:, j], a_indices[:, j]]
            pooled = phi_real.new_zeros((n_atoms, channels.shape[0]))
            if edges:
                pooled.index_add_(0, src, factor)
            M.index_copy_(1, global_idx, pooled)

        if self._density_channels.numel():
            A_gathered = A_real[:, self._density_channels, self._density_a]
        else:
            A_gathered = A_real.new_zeros((n_atoms, 0))

        density_product = phi_real.new_ones((n_atoms, self.term_count))
        for count, (term_idx, factor_idx) in self._density_buckets.items():
            if count == 0:
                continue
            block = phi_real.new_ones((n_atoms, term_idx.shape[0]))
            for j in range(count):
                block = block * A_gathered[:, factor_idx[:, j]]
            density_product.index_copy_(1, term_idx, block)

        valid_mask = z >= float(self.tag_count)
        free_factor = torch.ones((n_atoms, self.term_count), dtype=torch.float64)
        for p, term_idx in self._p_buckets.items():
            count = self.tag_count - p
            top = z - float(p)
            value = torch.ones(n_atoms, dtype=torch.float64)
            for i in range(count):
                value = value * (top - i)
            value = torch.where(valid_mask, value, torch.zeros_like(value))
            free_factor.index_copy_(1, term_idx, value.unsqueeze(1).expand(n_atoms, term_idx.shape[0]))

        moment_product = phi_real.new_ones((n_atoms, self.term_count))
        for b, (term_idx, moment_idx) in self._partition_buckets.items():
            block = phi_real.new_ones((n_atoms, term_idx.shape[0]))
            for j in range(b):
                block = block * M[:, moment_idx[:, j]]
            moment_product.index_copy_(1, term_idx, block)

        term_value = self._term_coefficient.unsqueeze(0) * density_product * free_factor * moment_product
        out = phi_real.new_zeros((n_atoms, self.descriptor_count))
        if self.term_count:
            out.index_add_(1, self._term_feature_index, term_value)
        return out

    @staticmethod
    def _prefix_suffix_derivative(values, derivatives, count):
        """d(prod_{j=0}^{count-1} values[j]) via prefix/suffix products, no
        division (WP4e) -- ``values``/``derivatives`` are length-``count``
        lists of same-shape tensors (last shape a Cartesian axis on
        ``derivatives`` only); returns ``sum_j derivatives[j] * (prod of
        every OTHER values[j'])``. ``count == 0`` returns a zero-shaped
        broadcastable 0.0 (the empty product's derivative).
        """

        if count == 0:
            return 0.0
        prefix = [torch.ones_like(values[0])]
        for j in range(count):
            prefix.append(prefix[-1] * values[j])
        suffix = [torch.ones_like(values[0])]
        for j in reversed(range(count)):
            suffix.append(suffix[-1] * values[j])
        suffix.reverse()
        total = derivatives[0].new_zeros(derivatives[0].shape)
        for j in range(count):
            total = total + derivatives[j] * (prefix[j] * suffix[j + 1]).unsqueeze(-1)
        return total

    def evaluate_real_with_jacobian(
        self,
        phi_real,
        dphi_real,
        A_real,
        edge_index,
        n_atoms,
        term_chunk_size=None,
    ):
        """Explicit (autograd-free) per-edge feature-value derivative (WP4e).

        Same value as :meth:`evaluate_real` (returned identically, ``out``),
        plus ``dF_e`` (``[edges, feature_count, 3]``): the derivative of
        that edge's OWN center atom's per-term contribution with respect to
        the edge's displacement, ``d(term_value[src[e], term]) / d(disp[e,
        :])``, already summed into feature space via the same
        ``feature_index`` scatter :meth:`evaluate_real` uses for the value.
        Built by mirroring every bucketed product in :meth:`evaluate_real`
        with a PARALLEL product-rule derivative via prefix/suffix products
        (:meth:`_prefix_suffix_derivative`) -- never a division by a
        (possibly zero) factor value. ``free_factor`` (the falling-
        factorial edge-count combinatorial weight) is constant with respect
        to positions for a fixed edge topology (exactly like autograd: the
        edge-count ``z`` comes from ``index_add_`` of literal ``1.0``s, with
        no graph connection to positions either), so it contributes no
        derivative term, only as a multiplicative constant.

        ``term_chunk_size`` optionally bounds the number of lowered
        polynomial terms materialized at once. Terms are still traversed
        in their serialized order and scattered into the same output
        coordinates, so this changes only the evaluation schedule. The
        default keeps the historical one-chunk execution.

        Use :func:`position_jacobian_from_edge_derivative` to assemble
        ``dF_e`` into the same ``[feature_count, n_atoms, 3]`` position
        Jacobian :func:`~ye3t_ace.tagged_cauchy_fit._feature_sum_and_
        jacobian_chunked`'s autograd path returns.
        """

        src, dst = edge_index[0], edge_index[1]
        edges = int(phi_real.shape[0])

        z = torch.zeros(n_atoms, dtype=torch.float64)
        if edges:
            z.index_add_(0, src, torch.ones(edges, dtype=torch.float64))

        n_moments = len(self.program["real_moment_keys"])
        M = phi_real.new_zeros((n_atoms, n_moments))
        dM_e = phi_real.new_zeros((edges, n_moments, 3))
        for degree, (global_idx, channels, a_indices) in self._moment_buckets.items():
            vals = [phi_real[:, channels[:, j], a_indices[:, j]] for j in range(degree)]
            dvals = [dphi_real[:, channels[:, j], a_indices[:, j], :] for j in range(degree)]

            factor = phi_real.new_ones((edges, channels.shape[0]))
            for j in range(degree):
                factor = factor * vals[j]
            pooled = phi_real.new_zeros((n_atoms, channels.shape[0]))
            if edges:
                pooled.index_add_(0, src, factor)
            M.index_copy_(1, global_idx, pooled)

            dfactor = self._prefix_suffix_derivative(vals, dvals, degree)
            if not torch.is_tensor(dfactor):
                dfactor = dphi_real.new_zeros((edges, channels.shape[0], 3))
            dM_e.index_copy_(1, global_idx, dfactor)

        if self._density_channels.numel():
            A_gathered = A_real[:, self._density_channels, self._density_a]
            dA_gathered_e = dphi_real[:, self._density_channels, self._density_a, :]
        else:
            A_gathered = A_real.new_zeros((n_atoms, 0))
            dA_gathered_e = dphi_real.new_zeros((edges, 0, 3))
        A_gathered_e = A_gathered[src] if edges else A_gathered.new_zeros((0, A_gathered.shape[1]))
        M_e = M[src] if edges else M.new_zeros((0, M.shape[1]))

        valid_mask = z >= float(self.tag_count)
        out = phi_real.new_zeros((n_atoms, self.descriptor_count))
        dF_e = phi_real.new_zeros((edges, self.descriptor_count, 3))
        if not self.term_count:
            return out, dF_e

        if term_chunk_size is None:
            term_chunk_size = self.term_count
        elif not isinstance(term_chunk_size, int) or isinstance(term_chunk_size, bool):
            raise TypeError("term_chunk_size must be a positive integer or None.")
        if term_chunk_size <= 0:
            raise ValueError("term_chunk_size must be positive.")

        for start in range(0, self.term_count, term_chunk_size):
            stop = min(start + term_chunk_size, self.term_count)
            chunk_count = stop - start

            density_product = phi_real.new_ones((n_atoms, chunk_count))
            d_density_product_e = phi_real.new_zeros((edges, chunk_count, 3))
            for count, (term_idx, factor_idx) in self._density_buckets.items():
                if count == 0:
                    continue
                selected = (term_idx >= start) & (term_idx < stop)
                if not bool(torch.any(selected)):
                    continue
                local_idx = term_idx[selected] - start
                local_factor_idx = factor_idx[selected]
                block = phi_real.new_ones((n_atoms, local_idx.shape[0]))
                for j in range(count):
                    block = block * A_gathered[:, local_factor_idx[:, j]]
                density_product.index_copy_(1, local_idx, block)

                a_edge_vals = [
                    A_gathered_e[:, local_factor_idx[:, j]] for j in range(count)
                ]
                da_edge_vals = [
                    dA_gathered_e[:, local_factor_idx[:, j], :] for j in range(count)
                ]
                dblock = self._prefix_suffix_derivative(
                    a_edge_vals, da_edge_vals, count
                )
                d_density_product_e.index_copy_(1, local_idx, dblock)

            free_factor = torch.ones((n_atoms, chunk_count), dtype=torch.float64)
            for p, term_idx in self._p_buckets.items():
                selected = (term_idx >= start) & (term_idx < stop)
                if not bool(torch.any(selected)):
                    continue
                local_idx = term_idx[selected] - start
                count = self.tag_count - p
                top = z - float(p)
                value = torch.ones(n_atoms, dtype=torch.float64)
                for i in range(count):
                    value = value * (top - i)
                value = torch.where(valid_mask, value, torch.zeros_like(value))
                free_factor.index_copy_(
                    1, local_idx, value.unsqueeze(1).expand(n_atoms, local_idx.shape[0])
                )

            moment_product = phi_real.new_ones((n_atoms, chunk_count))
            d_moment_product_e = phi_real.new_zeros((edges, chunk_count, 3))
            for b, (term_idx, moment_idx) in self._partition_buckets.items():
                selected = (term_idx >= start) & (term_idx < stop)
                if not bool(torch.any(selected)):
                    continue
                local_idx = term_idx[selected] - start
                local_moment_idx = moment_idx[selected]
                block = phi_real.new_ones((n_atoms, local_idx.shape[0]))
                for j in range(b):
                    block = block * M[:, local_moment_idx[:, j]]
                moment_product.index_copy_(1, local_idx, block)

                if b == 0:
                    continue
                m_edge_vals = [
                    M_e[:, local_moment_idx[:, j]] for j in range(b)
                ]
                dm_edge_vals = [
                    dM_e[:, local_moment_idx[:, j], :] for j in range(b)
                ]
                dblock = self._prefix_suffix_derivative(
                    m_edge_vals, dm_edge_vals, b
                )
                d_moment_product_e.index_copy_(1, local_idx, dblock)

            coefficient = self._term_coefficient[start:stop].unsqueeze(0)
            term_value = (
                coefficient * density_product * free_factor * moment_product
            )
            feature_index = self._term_feature_index[start:stop]
            out.index_add_(1, feature_index, term_value)

            if edges:
                density_product_e = density_product[src]
                moment_product_e = moment_product[src]
                free_factor_e = free_factor[src]
                d_term_value_e = (
                    coefficient.unsqueeze(-1)
                    * free_factor_e.unsqueeze(-1)
                    * (
                        d_density_product_e * moment_product_e.unsqueeze(-1)
                        + density_product_e.unsqueeze(-1) * d_moment_product_e
                    )
                )
                dF_e.index_add_(1, feature_index, d_term_value_e)
        return out, dF_e

    def descriptors(self, positions, atom_types, cell, pbc, primitives):
        del cell, pbc
        edge_index, _disp, phi_complex = primitives
        n_atoms = int(positions.shape[0])
        if int(atom_types.shape[0]) != n_atoms:
            raise ValueError("atom_types must have one entry per atom.")
        src = edge_index[0]
        n_channels = int(phi_complex.shape[1])
        max_width = int(phi_complex.shape[2])

        if self._channel_real_form_ids is not None:
            def _to_real(value):
                return _complex_to_real_by_channel_list(
                    value, self._channel_real_form_ids, self._real_form_records
                )
        else:
            def _to_real(value):
                return complex_to_artifact_real(value, self.compiled)

        phi_real = _to_real(phi_complex)
        A_complex = phi_complex.new_zeros((n_atoms, n_channels, max_width))
        if int(src.numel()):
            A_complex.index_add_(0, src, phi_complex)
        A_real = _to_real(A_complex)

        out = self.evaluate_real(phi_real, A_real, src, n_atoms)
        # Same graph-connection guard as TaggedTupleEvaluator/TaggedMomentEvaluator
        # (see the comment there): keeps the all-zero, no-edges case connected to
        # the positions autograd graph so torch.autograd.grad returns an exact-zero
        # gradient instead of raising.
        zero_link = positions.sum() * 0.0
        return out + zero_link.to(out.dtype)

    def descriptors_and_jacobian(
        self, positions, atom_types, primitives, term_chunk_size=None
    ):
        """Explicit (autograd-free) descriptors AND their position Jacobian
        (WP4e).

        ``primitives`` is ``(edge_index, disp, phi_complex, dphi_complex)``
        from :func:`ordinary_edge_primitives_with_derivative` (or an
        arm-level equivalent already at the global channel granularity, the
        same convention :meth:`descriptors` uses for its own ``primitives``
        tuple). Returns ``(out, jacobian)``: ``out`` is identical to
        :meth:`descriptors`'s own return; ``jacobian`` is
        ``[feature_count, n_atoms, 3]``, the Jacobian of ``out.sum(dim=0)``
        with respect to ``positions`` -- the SAME quantity (and shape)
        :func:`~ye3t_ace.tagged_cauchy_fit._feature_sum_and_jacobian_chunked`'s
        autograd path returns, so the two are drop-in interchangeable at
        that call site (``jacobian_mode``).
        """

        edge_index, _disp, phi_complex, dphi_complex = primitives
        n_atoms = int(positions.shape[0])
        if int(atom_types.shape[0]) != n_atoms:
            raise ValueError("atom_types must have one entry per atom.")
        src = edge_index[0]
        n_channels = int(phi_complex.shape[1])
        max_width = int(phi_complex.shape[2])

        if self._channel_real_form_ids is not None:
            def _to_real(value):
                return _complex_to_real_by_channel_list(
                    value, self._channel_real_form_ids, self._real_form_records
                )
        else:
            def _to_real(value):
                return complex_to_artifact_real(value, self.compiled)

        def _to_real_with_cartesian_axis(value):
            # _complex_to_real_by_channel_list expects (..., channel, width);
            # move the trailing Cartesian axis out of the way (to the front)
            # first, then back (to the end) afterward.
            return _to_real(value.movedim(-1, 0)).movedim(0, -1)

        phi_real = _to_real(phi_complex)
        dphi_real = _to_real_with_cartesian_axis(dphi_complex)
        A_complex = phi_complex.new_zeros((n_atoms, n_channels, max_width))
        if int(src.numel()):
            A_complex.index_add_(0, src, phi_complex)
        A_real = _to_real(A_complex)

        out, dF_e = self.evaluate_real_with_jacobian(
            phi_real,
            dphi_real,
            A_real,
            edge_index,
            n_atoms,
            term_chunk_size=term_chunk_size,
        )
        jacobian = position_jacobian_from_edge_derivative(dF_e, edge_index, n_atoms, self.descriptor_count)
        return out, jacobian


def position_jacobian_from_edge_derivative(dF_e, edge_index, n_atoms, n_features):
    """Assemble ``[feature_count, n_atoms, 3]`` from per-edge feature
    derivatives ``dF_e`` (``[edges, feature_count, 3]``, the derivative of
    edge ``e``'s own center atom's contribution w.r.t. ``disp[e,:]``)
    (WP4e).

    ``disp[e,:] = positions[dst[e]] - positions[src[e]]``, so
    ``d(disp[e])/d(positions[dst[e]]) = +I`` and
    ``d(disp[e])/d(positions[src[e]]) = -I``; each edge's contribution is
    scattered to both its endpoints with the corresponding sign via
    ``index_add_``. Matches
    :func:`~ye3t_ace.tagged_cauchy_fit._feature_sum_and_jacobian_chunked`'s
    own ``jacobian`` shape/convention exactly (the Jacobian of the
    ATOM-SUMMED feature vector, not a per-atom tensor).
    """

    src, dst = edge_index[0], edge_index[1]
    jac = dF_e.new_zeros((n_features, n_atoms, 3))
    if int(src.numel()):
        dF_e_t = dF_e.transpose(0, 1)
        jac.index_add_(1, dst, dF_e_t)
        jac.index_add_(1, src, -dF_e_t)
    return jac


class TaggedCauchyModel:
    """Compiled artifact plus role bindings, selection, beta, and offsets.

    Builds both the tuple (T1) and moment (T2) evaluators, so either
    execution strategy is available for the same fitted beta/offsets. An
    optional ``combination_matrix`` (features x selected-descriptors, real;
    e.g. the exact pooled basis from ``ye3t.couplings.tagged_cauchy.
    pooled_tagged_basis``/``pooled_feature_matrix``) is applied identically
    by both evaluators, so ``beta`` and every execution strategy -- including
    the ``real_moment_program`` block :func:`export_tagged_model` embeds --
    always share exactly one feature space: whichever one ``beta`` was fit
    in. This is enforced below (both evaluators' ``descriptor_count`` must
    equal ``len(beta)``), not just documented.

    ``beta`` is either a plain 1-D array/tensor (legacy: one feature-space
    readout shared by every central species) or a mapping ``{species: 1-D
    array}`` (WP4c: ``E = sum_i [offset(s_i) + sum_a beta_{s_i,a} F_i,a]``,
    the energy coefficients depend on the CENTRAL species of each atom, not
    just its per-neighbor-species channels). A single-species mapping is
    mathematically identical to the legacy plain-array form (this is what
    "reduces to the single-species case" means for Ta). Every entry of a
    mapping must have the same length, equal to both evaluators'
    ``descriptor_count`` (checked below); the descriptors themselves
    (``TaggedTupleEvaluator``/``TaggedMomentEvaluator``/
    ``RealMomentEvaluator``) are unchanged and species-independent -- the
    per-species readout lives entirely in :func:`energy_and_forces` (a
    per-center species gather against ``self.beta``, not a mask baked into
    the descriptors).
    """

    def __init__(
        self,
        compiled,
        role_bindings,
        descriptor_selection,
        beta,
        offsets,
        radial_config,
        cutoff,
        species_order,
        combination_matrix=None,
        *,
        channels=None,
        real_moment_program=None,
        channel_real_form_ids=None,
        real_form_records=None,
    ):
        # WP4d: compiled=None selects "multi-content" (merged-arm) mode --
        # no single compiled artifact backs this model (it was assembled by
        # ye3t_ace.tagged_cauchy_fit.merge_real_moment_programs from several
        # contents' own artifacts), so there is no complex-arithmetic
        # TaggedTupleEvaluator/TaggedMomentEvaluator to build; the ONLY live
        # evaluator is a RealMomentEvaluator built directly from the given
        # (already merged, already in the space beta was fit in)
        # real_moment_program. Everything below that does not depend on
        # having one compiled artifact (offsets/radial_config/cutoff/
        # species_order/beta parsing) is shared verbatim between the two
        # modes; only channel/evaluator construction branches.
        self.multi_content = compiled is None
        if self.multi_content:
            if channels is None or real_moment_program is None or real_form_records is None:
                raise ValueError(
                    "TaggedCauchyModel(compiled=None) (multi-content mode) requires "
                    "channels, real_moment_program, and real_form_records."
                )
            if descriptor_selection is not None:
                raise ValueError("descriptor_selection must be None in multi-content mode.")
            if combination_matrix is not None:
                raise ValueError(
                    "combination_matrix must be None in multi-content mode -- any per-"
                    "content combination is already folded into real_moment_program by "
                    "merge_real_moment_programs."
                )
            self.compiled = None
            self.channels = tuple(dict(channel) for channel in channels)
            indices = tuple(int(channel["channel_index"]) for channel in self.channels)
            if indices != tuple(range(len(self.channels))):
                raise ValueError("Multi-content model channels require dense channel_index 0..N-1.")
        else:
            self.compiled = _load_compiled_artifact(compiled)
            self.channels = _artifact_channels(self.compiled)

        self.role_bindings = tuple((str(kind), int(value)) for kind, value in role_bindings)
        self.descriptor_selection = (
            None if descriptor_selection is None else tuple(int(value) for value in descriptor_selection)
        )
        self.offsets = {str(key): float(value) for key, value in dict(offsets).items()}
        self.radial_config = dict(radial_config)
        self.cutoff = float(cutoff)
        self.species_order = tuple(str(value) for value in species_order)
        self.species_index = {name: index for index, name in enumerate(self.species_order)}
        # atom_types built from self.species_index (here, and in energy_and_
        # forces for the per-species beta/offset gather) MUST use the same
        # numeric convention ordinary_edge_primitives uses internally for its
        # own neighbor-species masking (it recomputes species_index itself,
        # from _species_order_from_channels(channels) -- see its docstring/
        # body) -- otherwise a multi-species atom_types tensor would silently
        # mean two different things in the two places it is used. Every
        # existing caller (fit_tiny_tagged_model, fit_tagged_arm) already
        # derives species_order this exact way; enforced here explicitly so
        # a future caller cannot silently violate it (single-species models,
        # where the two orderings trivially coincide, are unaffected).
        expected_species_order = _species_order_from_channels(self.channels)
        if self.species_order != expected_species_order:
            raise ValueError(
                f"species_order {self.species_order} does not match the "
                f"channel-derived order {expected_species_order} (required: "
                "ordinary_edge_primitives derives its own neighbor-species "
                "index the same way, and atom_types must mean the same thing "
                "in both places)."
            )

        self.per_species_beta = isinstance(beta, dict)
        if self.per_species_beta:
            beta = {str(key): value for key, value in beta.items()}
            missing = [species for species in self.species_order if species not in beta]
            if missing:
                raise ValueError(f"beta mapping is missing species {missing}.")
            unknown = [species for species in beta if species not in self.species_index]
            if unknown:
                raise ValueError(f"beta mapping has entries for unknown species {unknown}.")
            per_species_tensors = {
                species: _real_float64_tensor(values, f"beta[{species}]").clone()
                for species, values in beta.items()
            }
            lengths = {species: int(tensor.shape[0]) for species, tensor in per_species_tensors.items()}
            if len(set(lengths.values())) > 1:
                raise ValueError(f"All species' beta vectors must have the same length; got {lengths}.")
            self.beta_by_species = per_species_tensors
            # [n_species, n_features], species_order-indexed: row s is the
            # central-species-s beta, gathered by atom_types in
            # energy_and_forces (self.beta[s] for a legacy plain array is
            # simply that array; both cases are handled uniformly there via
            # self.per_species_beta).
            self.beta = torch.stack([per_species_tensors[species] for species in self.species_order], dim=0)
            beta_width = int(self.beta.shape[1])
        else:
            self.beta_by_species = None
            self.beta = _real_float64_tensor(beta, "beta").clone()
            beta_width = int(self.beta.shape[0])

        if self.multi_content:
            self.combination_matrix = None
            self.evaluator = None
            self.moment_evaluator = None
            self.real_moment_program = dict(real_moment_program)
            self._channel_real_form_ids = list(channel_real_form_ids)
            self._real_form_records = dict(real_form_records)
            self.real_evaluator = RealMomentEvaluator(
                program=self.real_moment_program,
                tag_count=self.real_moment_program["tag_count"],
                channel_real_form_ids=self._channel_real_form_ids,
                real_form_records=self._real_form_records,
            )
            if beta_width != self.real_evaluator.descriptor_count:
                raise ValueError("beta length must equal the merged real program feature count.")
            return

        self.combination_matrix = (
            None if combination_matrix is None else _real_float64_tensor(combination_matrix, "combination_matrix")
        )
        self.evaluator = TaggedTupleEvaluator(
            self.compiled, self.role_bindings, combination_matrix=self.combination_matrix
        )
        self.evaluator.descriptor_selection = self.descriptor_selection
        self.moment_evaluator = TaggedMomentEvaluator(
            self.compiled, self.role_bindings, self.descriptor_selection, combination_matrix=self.combination_matrix
        )
        self.real_moment_program = None
        self._channel_real_form_ids = None
        self._real_form_records = None
        self.real_evaluator = None
        if beta_width != self.evaluator.descriptor_count:
            raise ValueError("beta length must equal the descriptor selection width.")
        if beta_width != self.moment_evaluator.descriptor_count:
            raise ValueError("beta length must equal the moment evaluator descriptor count.")

    def evaluator_for(self, execution_strategy):
        if execution_strategy == "streamed_reference":
            if self.evaluator is None:
                raise ValueError(
                    "streamed_reference is unavailable for a multi-content (merged-arm) "
                    "model; use 'real_moment_reduction'."
                )
            return self.evaluator
        if execution_strategy == "exact_moment_reduction":
            if self.moment_evaluator is None:
                raise ValueError(
                    "exact_moment_reduction is unavailable for a multi-content (merged-arm) "
                    "model; use 'real_moment_reduction'."
                )
            return self.moment_evaluator
        if execution_strategy == "real_moment_reduction":
            if self.real_evaluator is None:
                raise ValueError(
                    "real_moment_reduction is only available for a multi-content "
                    "(merged-arm) model (built with compiled=None)."
                )
            return self.real_evaluator
        raise ValueError(
            "execution_strategy must be 'streamed_reference', 'exact_moment_reduction', "
            "or 'real_moment_reduction'."
        )


def _energy_from_per_atom_features(model, per_atom, atom_types):
    """sum_i beta_{species(i)} . F_i -- handles both plain and per-species beta.

    ``per_atom`` is ``[n_atoms, n_features]`` (any evaluator's ``descriptors``
    output, including :class:`RealMomentEvaluator`'s); ``atom_types`` is
    ``[n_atoms]`` species indices into ``model.species_order``. With a plain
    (legacy) beta this is the ordinary shared-readout dot product; with a
    per-species beta, each atom's row is dotted against *its own* species'
    beta row (``model.beta[atom_types]``, a gather -- the "per-center species
    mask in the readout").
    """

    if model.per_species_beta:
        beta_matrix = model.beta.to(dtype=per_atom.dtype)
        beta_per_atom = beta_matrix.index_select(0, atom_types)
        return (per_atom * beta_per_atom).sum()
    beta = model.beta.to(dtype=per_atom.dtype)
    return (per_atom @ beta).sum()


def energy_and_forces(model, atoms, execution_strategy="streamed_reference"):
    """Return (energy, forces) for one ASE Atoms structure via float64 autograd."""

    evaluator = model.evaluator_for(execution_strategy)
    positions = torch.tensor(
        np.asarray(atoms.get_positions(), dtype=np.float64),
        dtype=torch.float64,
        requires_grad=True,
    )
    symbols = atoms.get_chemical_symbols()
    atom_types = torch.tensor(
        [model.species_index[symbol] for symbol in symbols], dtype=torch.long
    )
    periodic = bool(np.any(np.asarray(atoms.pbc, dtype=bool)))
    cell = torch.tensor(np.asarray(atoms.cell.array, dtype=np.float64), dtype=torch.float64) if periodic else None
    pbc = tuple(bool(value) for value in atoms.pbc) if periodic else None

    primitives = ordinary_edge_primitives(
        positions, atom_types, cell, pbc, model.cutoff, model.radial_config, model.channels
    )
    per_atom = evaluator.descriptors(positions, atom_types, cell, pbc, primitives)
    offset_values = torch.tensor(
        [model.offsets[symbol] for symbol in symbols], dtype=torch.float64
    )
    energy = _energy_from_per_atom_features(model, per_atom, atom_types) + offset_values.sum()
    (gradient,) = torch.autograd.grad(energy, positions, create_graph=False)
    forces = -gradient
    return energy.detach(), forces.detach()


def _frame_feature_rows(atoms, evaluator, compiled, cutoff, radial_config, species_index):
    """Per-structure energy/force feature rows for one fixed descriptor set.

    ``evaluator`` may be a :class:`TaggedTupleEvaluator` or a
    :class:`TaggedMomentEvaluator`; both expose the same
    ``descriptors(positions, atom_types, cell, pbc, primitives)`` contract.
    """

    channels = _artifact_channels(compiled)
    species_order = tuple(species_index)
    symbols = atoms.get_chemical_symbols()
    atom_types = torch.tensor([species_index[symbol] for symbol in symbols], dtype=torch.long)
    cell = torch.tensor(np.asarray(atoms.cell.array, dtype=np.float64), dtype=torch.float64)
    pbc = tuple(bool(value) for value in atoms.pbc)
    n_atoms = len(atoms)

    def feature_sum(pos):
        primitives = ordinary_edge_primitives(pos, atom_types, cell, pbc, cutoff, radial_config, channels)
        per_atom = evaluator.descriptors(pos, atom_types, cell, pbc, primitives)
        return per_atom.sum(dim=0)

    positions_value = torch.tensor(np.asarray(atoms.get_positions(), dtype=np.float64), dtype=torch.float64)
    with torch.no_grad():
        features = feature_sum(positions_value).numpy()

    positions_leaf = positions_value.clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(feature_sum, positions_leaf, create_graph=False)
    n_features = features.shape[0]
    force_feature_block = -jacobian.detach().numpy().reshape(n_features, n_atoms * 3).T

    species_counts = np.zeros(len(species_order), dtype=np.float64)
    for symbol in symbols:
        species_counts[species_index[symbol]] += 1.0

    energy_row = np.concatenate([features / n_atoms, species_counts / n_atoms])
    energy_target = float(atoms.get_potential_energy()) / n_atoms

    offset_force_block = np.zeros((n_atoms * 3, len(species_order)), dtype=np.float64)
    force_rows = np.concatenate([force_feature_block, offset_force_block], axis=1)
    force_targets = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)

    return energy_row, energy_target, force_rows, force_targets, n_atoms


def _rmse(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def fit_tiny_tagged_model(
    train_frames,
    val_frames,
    compiled,
    role_bindings,
    selection,
    ridge_alpha=1.0e-8,
    cutoff=TA_CUTOFF,
    radial_config=None,
    execution_strategy="streamed_reference",
):
    """Beta-only ridge fit on tiny energy/force rows; returns (model, metrics).

    Features are structure sums of the pooled tagged-Cauchy descriptors plus
    one per-species atom-count column for the fitted offset.  Force rows are
    the negative position-Jacobian of the same feature sums, computed with
    whichever ``execution_strategy`` evaluator is requested (the fitted
    ``beta``/``offsets`` apply identically to either, since
    :class:`TaggedCauchyModel` always builds both).  Feature columns are
    scaled by their train-set RMS before solving
    ``(X^T X + alpha_relative * diag(X^T X).mean() * I) beta = X^T y`` in
    float64, then unscaled back to raw units.  This is a development fit;
    it does not claim predictive quality.
    """

    radial_config = dict(TA_RADIAL_CONFIG) if radial_config is None else dict(radial_config)
    compiled = _load_compiled_artifact(compiled)
    channels = _artifact_channels(compiled)
    species_order = _species_order_from_channels(channels)
    species_index = {name: index for index, name in enumerate(species_order)}

    normalized_selection = None if selection is None else tuple(int(value) for value in selection)
    if execution_strategy == "streamed_reference":
        evaluator = TaggedTupleEvaluator(compiled, role_bindings)
        evaluator.descriptor_selection = normalized_selection
    elif execution_strategy == "exact_moment_reduction":
        evaluator = TaggedMomentEvaluator(compiled, role_bindings, normalized_selection)
    else:
        raise ValueError(
            "execution_strategy must be 'streamed_reference' or 'exact_moment_reduction'."
        )
    n_features = evaluator.descriptor_count

    def assemble(frames):
        energy_rows = []
        energy_targets = []
        force_rows = []
        force_targets = []
        for atoms in frames:
            e_row, e_target, f_rows, f_targets, _n_atoms = _frame_feature_rows(
                atoms, evaluator, compiled, cutoff, radial_config, species_index
            )
            energy_rows.append(e_row)
            energy_targets.append(e_target)
            force_rows.append(f_rows)
            force_targets.append(f_targets)
        X_energy = np.stack(energy_rows, axis=0)
        y_energy = np.asarray(energy_targets, dtype=np.float64)
        X_force = np.concatenate(force_rows, axis=0)
        y_force = np.concatenate(force_targets, axis=0)
        return X_energy, y_energy, X_force, y_force

    X_energy_train, y_energy_train, X_force_train, y_force_train = assemble(train_frames)
    X_energy_val, y_energy_val, X_force_val, y_force_val = assemble(val_frames)

    X_train = np.concatenate([X_energy_train, X_force_train], axis=0)
    y_train = np.concatenate([y_energy_train, y_force_train], axis=0)

    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    col_rms = torch.sqrt(torch.mean(X_train_t * X_train_t, dim=0))
    col_rms = torch.clamp(col_rms, min=1.0e-12)
    X_scaled = X_train_t / col_rms
    XtX = X_scaled.T @ X_scaled
    Xty = X_scaled.T @ y_train_t
    diag_scale = float(torch.diagonal(XtX).mean())
    alpha_absolute = float(ridge_alpha) * diag_scale
    normal_matrix = XtX + alpha_absolute * torch.eye(XtX.shape[0], dtype=torch.float64)
    beta_scaled = torch.linalg.solve(normal_matrix, Xty)
    beta_full = (beta_scaled / col_rms).numpy()

    beta = beta_full[:n_features]
    offsets = {species_order[i]: float(beta_full[n_features + i]) for i in range(len(species_order))}

    model = TaggedCauchyModel(
        compiled,
        role_bindings,
        normalized_selection,
        beta,
        offsets,
        radial_config,
        cutoff,
        species_order,
    )

    def predict(X):
        return X @ beta_full

    metrics = {
        "execution_strategy": execution_strategy,
        "train_energy_rmse_ev_per_atom": _rmse(predict(X_energy_train), y_energy_train),
        "val_energy_rmse_ev_per_atom": _rmse(predict(X_energy_val), y_energy_val),
        "train_force_rmse_ev_per_A": _rmse(predict(X_force_train), y_force_train),
        "val_force_rmse_ev_per_A": _rmse(predict(X_force_val), y_force_val),
        "train_structures": int(len(train_frames)),
        "val_structures": int(len(val_frames)),
        "train_energy_rows": int(X_energy_train.shape[0]),
        "train_force_rows": int(X_force_train.shape[0]),
        "ridge_alpha_relative": float(ridge_alpha),
        "ridge_alpha_absolute": alpha_absolute,
        "descriptor_count": int(n_features),
    }
    return model, metrics


def moment_equivalence_certificate(compiled, role_bindings, selection, seeds, ta_frame0_atoms=None):
    """Cross-check TaggedMomentEvaluator against TaggedTupleEvaluator.

    For each seed, builds a random non-periodic Ta cluster (5 to 8 atoms,
    cycling with the seed index), checks descriptor agreement (relative,
    over all descriptors including any unsupported ones) and, with a random
    beta, force agreement via :func:`energy_and_forces` under both
    ``execution_strategy`` values.  If ``ta_frame0_atoms`` is given (meant
    for the k=2, 57-descriptor artifact), also checks descriptor agreement
    there.  Raises if any tolerance is exceeded; otherwise returns the
    measured numbers (suitable for embedding in an export under
    ``execution_certificate``).
    """

    tuple_evaluator = TaggedTupleEvaluator(compiled, role_bindings)
    tuple_evaluator.descriptor_selection = None if selection is None else tuple(int(v) for v in selection)
    moment_evaluator = TaggedMomentEvaluator(compiled, role_bindings, selection)
    n_features = moment_evaluator.descriptor_count
    if tuple_evaluator.descriptor_count != n_features:
        raise ValueError("Tuple and moment evaluators disagree on descriptor_count.")

    descriptor_relative_tolerance = 1.0e-10
    force_absolute_tolerance = 1.0e-9
    descriptor_max_relative = 0.0
    force_max_absolute = 0.0
    cluster_reports = []
    channels = _artifact_channels(compiled)
    # The random test clusters below are single-species (one element
    # repeated for every atom); every existing caller's artifact is built
    # from a single-species channel registry, so the artifact's own first
    # (and only) referenced species is used here instead of a hardcoded
    # element -- this generalizes the oracle without changing its behavior
    # for any existing (single-species) caller.
    single_species = _species_order_from_channels(channels)[0]

    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        n_atoms = 5 + (index % 4)
        positions_np = rng.uniform(low=-2.1, high=2.1, size=(n_atoms, 3))
        atom_types = torch.zeros(n_atoms, dtype=torch.long)
        positions_value = torch.tensor(positions_np, dtype=torch.float64)
        primitives = ordinary_edge_primitives(
            positions_value, atom_types, None, None, TA_CUTOFF, TA_RADIAL_CONFIG, channels
        )
        tuple_descriptors = (
            tuple_evaluator.descriptors(positions_value, atom_types, None, None, primitives)
            .detach()
            .numpy()
        )
        moment_descriptors = (
            moment_evaluator.descriptors(positions_value, atom_types, None, None, primitives)
            .detach()
            .numpy()
        )
        scale = max(1.0, float(np.max(np.abs(tuple_descriptors))))
        descriptor_error = float(np.max(np.abs(tuple_descriptors - moment_descriptors))) / scale
        descriptor_max_relative = max(descriptor_max_relative, descriptor_error)

        beta = rng.normal(size=n_features)
        model = TaggedCauchyModel(
            compiled,
            role_bindings,
            tuple_evaluator.descriptor_selection,
            beta,
            {single_species: 0.0},
            TA_RADIAL_CONFIG,
            TA_CUTOFF,
            (single_species,),
        )
        atoms = _single_species_atoms(positions_np, species=single_species)
        _, forces_tuple = energy_and_forces(model, atoms, execution_strategy="streamed_reference")
        _, forces_moment = energy_and_forces(model, atoms, execution_strategy="exact_moment_reduction")
        force_error = float((forces_tuple - forces_moment).abs().max())
        force_max_absolute = max(force_max_absolute, force_error)

        cluster_reports.append(
            {
                "seed": int(seed),
                "n_atoms": int(n_atoms),
                "descriptor_relative_error": descriptor_error,
                "force_absolute_error": force_error,
            }
        )

    result = {
        "tag_count": tuple_evaluator.tag_count,
        "cluster_count": len(tuple(seeds)),
        "descriptor_max_relative_error": descriptor_max_relative,
        "force_max_absolute_error": force_max_absolute,
        "descriptor_relative_tolerance": descriptor_relative_tolerance,
        "force_absolute_tolerance": force_absolute_tolerance,
        "clusters": cluster_reports,
    }

    if ta_frame0_atoms is not None:
        species_order = _species_order_from_channels(channels)
        species_index = {name: i for i, name in enumerate(species_order)}
        positions_value = torch.tensor(
            np.asarray(ta_frame0_atoms.get_positions(), dtype=np.float64), dtype=torch.float64
        )
        atom_types = torch.tensor(
            [species_index[s] for s in ta_frame0_atoms.get_chemical_symbols()], dtype=torch.long
        )
        cell = torch.tensor(np.asarray(ta_frame0_atoms.cell.array, dtype=np.float64), dtype=torch.float64)
        pbc = tuple(bool(v) for v in ta_frame0_atoms.pbc)
        primitives = ordinary_edge_primitives(
            positions_value, atom_types, cell, pbc, TA_CUTOFF, TA_RADIAL_CONFIG, channels
        )
        tuple_d = (
            tuple_evaluator.descriptors(positions_value, atom_types, cell, pbc, primitives).detach().numpy()
        )
        moment_d = (
            moment_evaluator.descriptors(positions_value, atom_types, cell, pbc, primitives).detach().numpy()
        )
        scale = max(1.0, float(np.max(np.abs(tuple_d))))
        error = float(np.max(np.abs(tuple_d - moment_d))) / scale
        result["ta_frame0_descriptor_relative_error"] = error
        result["ta_frame0_natoms"] = int(len(ta_frame0_atoms))
        if error > descriptor_relative_tolerance:
            raise ValueError(f"Ta frame0 moment/tuple descriptor mismatch: {error}")

    if descriptor_max_relative > descriptor_relative_tolerance:
        raise ValueError(f"Moment/tuple descriptor mismatch: {descriptor_max_relative}")
    if force_max_absolute > force_absolute_tolerance:
        raise ValueError(f"Moment/tuple force mismatch: {force_max_absolute}")
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _canonical_json_sha256(body):
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def export_tagged_model(path, model, metrics=None, execution_certificate=None):
    """Write a versioned, hash-bound JSON export of a fitted tagged model.

    Schema v1 additionally embeds the moment-reduction program (moment
    keys, density keys, and the flat term table with exact [re, im]
    coefficients and integer partition weights) so a native runtime can
    consume the exact moment-reduction path directly, and an optional
    ``execution_certificate`` (see :func:`moment_equivalence_certificate`).

    Schema v2 (WP3c) additionally embeds ``real_moment_program`` (the exact
    real-arithmetic / "real tesseral" lowering from :func:`real_moment_program`,
    built from ``model.descriptor_selection`` and ``model.combination_matrix``
    -- the SAME feature space ``beta`` was fit in, always; there is no
    override), ``channel_real_forms`` (per channel: ``channel_index``, ``l``,
    ``radial_channel``, ``neighbor_species``, ``real_form_id``), and
    ``radial_definition`` (the PACE ChebExpCos parameters actually used and
    the explicit n=1,2 column mapping), so a native (real-arithmetic) loader
    needs nothing else -- including, when ``model.combination_matrix`` is
    set (see :class:`TaggedCauchyModel`), a pooled/combined feature space:
    the export also carries ``combination_matrix`` itself so
    :func:`load_tagged_model` can reconstruct the identical model. Neither
    ``moment_program`` nor ``real_moment_program`` is a live execution
    strategy of the Python :class:`TaggedCauchyModel` returned by
    :func:`load_tagged_model` (``available_execution_strategies`` is
    unchanged from v1); both blocks are for external/native consumption.

    Refuses to write (raises ``ValueError``) if ``real_moment_program``'s
    ``feature_count`` does not equal ``len(beta)`` -- in normal use this can
    only happen if ``model.beta`` was mutated after construction, since
    :class:`TaggedCauchyModel`'s constructor already enforces the same
    invariant for both live evaluators; this is a second, independent check
    at the export boundary itself, so a native consumer's beta and feature
    space can never silently disagree.

    Multi-content (WP4d, ``model.compiled is None`` -- see
    :class:`TaggedCauchyModel`'s own "multi-content" constructor mode, and
    :func:`~ye3t_ace.tagged_cauchy_fit.arm_lammps_model`, the ONLY intended
    way to build one): a top-level boolean ``multi_content`` (``true`` here,
    ``false`` for every ordinary single-artifact export) tells a native
    loader which of the two following representations to expect --
    ``compiled_artifact`` is ``null`` (there is no single compiled artifact
    behind a merged multi-content arm), and THREE fields replace what it
    would otherwise supply:

    - ``arm_channels``: the arm's global channel list (each entry:
      ``channel_index``, ``channel_id``, ``neighbor_species``,
      ``radial_channel``, ``l``, ``source_family_id`` -- the same shape
      ``channel_real_forms`` entries key off of via their own
      ``channel_index``, and dense/consecutive from 0 the same way a single
      artifact's own channel list always is), i.e. the multi-content
      stand-in for ``compiled_artifact["channels"]``.
    - ``real_form_records``: ``{real_form_id: record}`` for every distinct
      real form ``channel_real_forms`` references (each record the same
      exact ``real_to_complex_matrix``-bearing shape one compiled
      artifact's own ``payload["real_forms"]`` entries have) -- the
      multi-content stand-in for ``compiled_artifact["real_forms"]``, since
      there is no single artifact to read them from.
    - ``channel_real_forms``: unchanged in shape from the single-artifact
      case (still per global channel: ``channel_index``, ``l``,
      ``radial_channel``, ``neighbor_species``, ``real_form_id``) -- a
      native loader resolves each entry's ``real_form_id`` against
      ``real_form_records`` exactly as it would resolve a single artifact's
      own ``channel_real_form_ids``/``real_forms`` pair.

    ``real_moment_program`` itself is unchanged in shape either way (already
    arm-global-channel-indexed for a multi-content export, via
    :func:`~ye3t_ace.tagged_cauchy_fit.merge_real_moment_programs`) --
    together with the three fields above, it is everything
    :func:`~ye3t_ace.tagged_cauchy_linear._complex_to_real_by_channel_list`
    plus :class:`RealMomentEvaluator`'s bucketed kernel need, with no
    ``compiled_artifact`` at all. ``load_tagged_model`` (below) is the
    reference implementation of this contract -- treat it, not this
    docstring, as the authority on exact field names/shapes if the two ever
    disagree.
    """

    channels = model.channels
    if model.multi_content:
        channel_real_form_id_of = {i: str(fid) for i, fid in enumerate(model._channel_real_form_ids)}
    else:
        channel_real_form_id_of = {
            int(record["channel_index"]): str(record["real_form_id"])
            for record in model.compiled.payload["channel_real_form_ids"]
        }
    channel_real_forms = [
        {
            "channel_index": int(channel["channel_index"]),
            "l": int(channel["l"]),
            "radial_channel": int(channel["radial_channel"]),
            "neighbor_species": str(channel["neighbor_species"]),
            "real_form_id": channel_real_form_id_of[int(channel["channel_index"])],
        }
        for channel in channels
    ]
    radial_count = max(int(channel["radial_channel"]) for channel in channels) + 1
    radial_definition = {
        "kind": "pace_cheb_exp_cos",
        "parameters": {
            "rc": float(model.cutoff),
            "cutoff_width": float(model.radial_config.get("cutoff_width", 0.0)),
            "lmbda": float(model.radial_config["lmbda"]),
            "radial_count": int(radial_count),
        },
        "column_mapping": (
            "PACE ChebExpCos native radial labels are 1-based; evaluator output "
            "column j (0-based, j = 0 .. radial_count-1) is PACE n = j + 1. For "
            "the Ta slice (radial_count=2): column 0 = n=1, column 1 = n=2."
        ),
    }
    if model.multi_content:
        real_moment_program_payload = model.real_moment_program
    else:
        real_moment_program_payload = real_moment_program(
            model.compiled,
            model.role_bindings,
            model.descriptor_selection,
            combination_matrix=model.combination_matrix,
        )
    real_feature_count = int(real_moment_program_payload["feature_count"])
    if model.per_species_beta:
        bad = {
            species: int(tensor.shape[0])
            for species, tensor in model.beta_by_species.items()
            if int(tensor.shape[0]) != real_feature_count
        }
        if bad:
            raise ValueError(
                "export_tagged_model: refusing to write an inconsistent native "
                f"artifact -- real_moment_program feature_count ({real_feature_count}) "
                f"does not equal beta length for species {bad}. beta and the real "
                "program must share exactly one feature space."
            )
        beta_payload = {
            species: model.beta_by_species[species].detach().cpu().numpy().tolist()
            for species in model.species_order
        }
    else:
        beta_length = int(model.beta.shape[0])
        if real_feature_count != beta_length:
            raise ValueError(
                "export_tagged_model: refusing to write an inconsistent native "
                f"artifact -- real_moment_program feature_count ({real_feature_count}) "
                f"does not equal beta length ({beta_length}). beta and the real "
                "program must share exactly one feature space."
            )
        beta_payload = model.beta.detach().cpu().numpy().tolist()

    if isinstance(beta_payload, dict):
        portfolio_beta = beta_payload
    else:
        if len(model.species_order) != 1:
            raise ValueError(
                "export_tagged_model: a shared beta vector is only unambiguous "
                "for one central species."
            )
        portfolio_beta = {str(model.species_order[0]): beta_payload}
    tagged_execution_portfolio = compile_tagged_moment_execution_portfolio(
        real_moment_program_payload,
        portfolio_beta,
    )

    body = {
        "schema": TAGGED_CAUCHY_SLICE_SCHEMA,
        "multi_content": bool(model.multi_content),
        "compiled_artifact": None if model.multi_content else model.compiled.to_dict(),
        "arm_channels": [dict(c) for c in channels] if model.multi_content else None,
        "real_form_records": dict(model._real_form_records) if model.multi_content else None,
        "role_bindings": [[kind, value] for kind, value in model.role_bindings],
        "descriptor_selection": (
            None if model.descriptor_selection is None else list(model.descriptor_selection)
        ),
        "combination_matrix": (
            None
            if model.combination_matrix is None
            else model.combination_matrix.detach().cpu().numpy().tolist()
        ),
        "beta": beta_payload,
        "per_species_beta": bool(model.per_species_beta),
        "offsets": dict(model.offsets),
        "radial_config": dict(model.radial_config),
        "cutoff": float(model.cutoff),
        "species_order": list(model.species_order),
        "tag_count": (
            int(real_moment_program_payload["tag_count"])
            if model.multi_content
            else int(model.evaluator.tag_count)
        ),
        "context_policy": "inclusive",
        "pooling": "ordered_sum",
        "execution_strategy": "real_moment_reduction" if model.multi_content else "streamed_reference",
        "available_execution_strategies": (
            ["real_moment_reduction"]
            if model.multi_content
            else ["streamed_reference", "exact_moment_reduction"]
        ),
        "moment_program": (
            None if model.multi_content else model.moment_evaluator.program.export_payload()
        ),
        "real_moment_program": real_moment_program_payload,
        "tagged_execution_portfolio": tagged_execution_portfolio,
        "channel_real_forms": channel_real_forms,
        "radial_definition": radial_definition,
    }
    if metrics is not None:
        body["metrics"] = _jsonable(metrics)
    if execution_certificate is not None:
        body["execution_certificate"] = _jsonable(execution_certificate)
    self_hash = _canonical_json_sha256(body)
    payload = dict(body)
    payload["self_hash"] = self_hash
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
    return payload


def export_tagged_composite_model(
    path, ordinary_model_path, tagged_model_path, sector_inventory=None, metadata=None
):
    """
    Purpose:
        Bind one ordinary .yace component and one tagged-Cauchy component into
        a self-contained PairYE3T deployment manifest.
    Mathematical contract:
        The manifest changes no fitted coefficient or descriptor coordinate;
        its runtime value is exactly the additive sum of the two hash-bound
        component models.  ``sector_inventory`` is provenance only and does
        not affect evaluation.
    Inputs:
        Output path, colocated ordinary and tagged component paths, and
        optional JSON-compatible provenance records.
    Outputs:
        The canonical, self-hashed manifest mapping written to ``path``.
    Does not:
        Refit a model, enumerate symmetry labels, or make PairPACE a runtime
        dependency.
    """

    out_path = Path(path)
    ordinary_path = Path(ordinary_model_path)
    tagged_path = Path(tagged_model_path)
    out_parent = out_path.parent.resolve()
    for component_path, label in (
        (ordinary_path, "ordinary_model_path"),
        (tagged_path, "tagged_model_path"),
    ):
        if component_path.resolve().parent != out_parent:
            raise ValueError(
                f"{label} must be colocated with the composite manifest."
            )
        if not component_path.is_file():
            raise FileNotFoundError(component_path)

    with tagged_path.open(encoding="utf-8") as handle:
        tagged_payload = json.load(handle)
    tagged_body = dict(tagged_payload)
    tagged_self_hash = str(tagged_body.pop("self_hash", ""))
    if not tagged_self_hash or _canonical_json_sha256(tagged_body) != tagged_self_hash:
        raise ValueError("Tagged component self_hash mismatch.")
    if str(tagged_body.get("schema")) not in (
        TAGGED_CAUCHY_SLICE_SCHEMA_V1,
        TAGGED_CAUCHY_SLICE_SCHEMA_V2,
        "ye3t_tagged_cauchy_slice_v3",
    ):
        raise ValueError("Unsupported tagged component schema.")

    body = {
        "schema": TAGGED_CAUCHY_COMPOSITE_SCHEMA,
        "species_order": [str(value) for value in tagged_body["species_order"]],
        "ordinary_component": {
            "path": ordinary_path.name,
            "sha256": hashlib.sha256(ordinary_path.read_bytes()).hexdigest(),
        },
        "tagged_component": {
            "path": tagged_path.name,
            "sha256": hashlib.sha256(tagged_path.read_bytes()).hexdigest(),
        },
        "sector_inventory": _jsonable(sector_inventory or {}),
        "metadata": _jsonable(metadata or {}),
    }
    payload = dict(body)
    payload["self_hash"] = _canonical_json_sha256(body)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
    return payload


def load_tagged_model(path):
    """Load and hash-verify a JSON export produced by export_tagged_model.

    Rebuilds both evaluators (:class:`TaggedTupleEvaluator` and
    :class:`TaggedMomentEvaluator`) from the hash-verified compiled artifact,
    role_bindings, descriptor_selection, and (schema v2) combination_matrix;
    the exported ``moment_program`` and (schema v2) ``real_moment_program``/
    ``channel_real_forms``/``radial_definition`` blocks are for external/
    native consumption and are not re-parsed here (recompiling from the
    hash-verified artifact is deterministic and gives a byte-identical
    program). Accepts both schema v1 and v2 exports.

    Raises ``ValueError`` if a (schema v2) exported ``real_moment_program``'s
    ``feature_count`` does not equal ``len(beta)`` -- guards against a
    hand-edited or externally-produced file that would otherwise silently
    reconstruct a model whose beta and native feature space disagree (schema
    v1 files, and any file with no ``real_moment_program`` block, have
    nothing to check here and load as before).
    """

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload = dict(payload)
    expected = str(payload.pop("self_hash", ""))
    actual = _canonical_json_sha256(payload)
    if not expected or expected != actual:
        raise ValueError("Tagged-Cauchy model export self_hash mismatch.")
    if str(payload.get("schema")) not in (TAGGED_CAUCHY_SLICE_SCHEMA_V1, TAGGED_CAUCHY_SLICE_SCHEMA_V2):
        raise ValueError("Unsupported tagged-Cauchy model export schema.")
    role_bindings = tuple((str(kind), int(value)) for kind, value in payload["role_bindings"])
    beta_raw = payload["beta"]
    per_species_beta = bool(payload.get("per_species_beta", isinstance(beta_raw, dict)))
    if per_species_beta:
        beta = {species: torch.tensor(values, dtype=torch.float64) for species, values in beta_raw.items()}
        beta_lengths = {species: int(tensor.shape[0]) for species, tensor in beta.items()}
    else:
        beta = torch.tensor(beta_raw, dtype=torch.float64)
        beta_lengths = {None: int(beta.shape[0])}
    real_moment_program_payload = payload.get("real_moment_program")
    if real_moment_program_payload is not None:
        exported_feature_count = int(real_moment_program_payload["feature_count"])
        bad = {key: length for key, length in beta_lengths.items() if length != exported_feature_count}
        if bad:
            raise ValueError(
                "Tagged-Cauchy model export is inconsistent: real_moment_program "
                f"feature_count ({exported_feature_count}) does not equal beta "
                f"length(s) {bad}."
            )
    execution_portfolio = payload.get("tagged_execution_portfolio")
    if execution_portfolio is not None:
        if real_moment_program_payload is None:
            raise ValueError(
                "Tagged-Cauchy execution portfolio requires real_moment_program."
            )
        if isinstance(beta_raw, dict):
            portfolio_beta = beta_raw
        else:
            if len(payload["species_order"]) != 1:
                raise ValueError(
                    "Tagged-Cauchy shared beta vector is ambiguous for multiple species."
                )
            portfolio_beta = {str(payload["species_order"][0]): beta_raw}
        expected_portfolio = compile_tagged_moment_execution_portfolio(
            real_moment_program_payload,
            portfolio_beta,
        )
        if _jsonable(execution_portfolio) != _jsonable(expected_portfolio):
            raise ValueError(
                "Tagged-Cauchy execution portfolio differs from its exact "
                "program/readout compilation."
            )

    multi_content = bool(payload.get("multi_content", False)) or payload.get("compiled_artifact") is None
    if multi_content:
        # WP4d: a merged-arm export -- no single compiled_artifact; the
        # channel table, real-form records, and (already merged/pooled)
        # real_moment_program are carried directly in the file instead.
        arm_channels = payload.get("arm_channels")
        real_form_records = payload.get("real_form_records")
        if arm_channels is None or real_form_records is None or real_moment_program_payload is None:
            raise ValueError(
                "Multi-content tagged-Cauchy model export is missing arm_channels, "
                "real_form_records, or real_moment_program."
            )
        channel_real_form_ids = [None] * len(arm_channels)
        for record in payload["channel_real_forms"]:
            channel_real_form_ids[int(record["channel_index"])] = str(record["real_form_id"])
        if any(value is None for value in channel_real_form_ids):
            raise ValueError("channel_real_forms does not cover every arm_channels entry.")
        return TaggedCauchyModel(
            None,
            role_bindings,
            None,
            beta,
            payload["offsets"],
            payload["radial_config"],
            payload["cutoff"],
            payload["species_order"],
            combination_matrix=None,
            channels=arm_channels,
            real_moment_program=real_moment_program_payload,
            channel_real_form_ids=channel_real_form_ids,
            real_form_records=real_form_records,
        )

    compiled = CompiledLiftedCauchyScalar.from_dict(payload["compiled_artifact"])
    selection = payload.get("descriptor_selection")
    selection = None if selection is None else tuple(int(value) for value in selection)
    combination_matrix = payload.get("combination_matrix")
    model = TaggedCauchyModel(
        compiled,
        role_bindings,
        selection,
        beta,
        payload["offsets"],
        payload["radial_config"],
        payload["cutoff"],
        payload["species_order"],
        combination_matrix=combination_matrix,
    )
    return model
