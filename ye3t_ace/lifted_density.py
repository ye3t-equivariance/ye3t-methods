"""Filtered/lifted ``A_s`` sources and fixed-feature linear readouts.

This module implements the role-resolved ``A_s`` source. Its mathematical
object is an equivariant radial channel map, not an
independent scalar density for each role:

    A_{i,s,eta,l,m}
        = sum_{eta'} K^{(l)}_{i,s;eta,eta'} A_{i,eta',l,m},
    K_{i,s} = direct_sum_l (K^{(l)}_{i,s} tensor I_{V_l}).

Here ``eta`` collects non-angular chemical and radial labels, while ``s`` is a
retained role coordinate.  ``K^{(l)}_{i,s}`` acts only on non-angular channels,
so it commutes with the declared O(3) action.  The current radial-filter
implementation is the diagonal edge realization

    A_{i,s,c} = sum_j q_s(i, j) phi_c(i, j),

where ``c`` flattens ``(eta,l,m)``, ``phi_c`` is evaluated by the existing
``SiteBasisV2`` real-spherical edge-channel machinery, and ``q_s`` is the
pair-dependent matrix element of the implemented ``K`` map.  The default
``softmax_gaussian`` realization is smooth and forms a radial partition of
unity, so summing compatible role channels recovers the corresponding
unfiltered density channel.

Role-density source construction and analytic geometry derivatives are shared
by the retained fixed-feature linear readouts.

TODO: planned chemical/radial channel transforms ``T`` compose as
``K_bar^{(l)} = T^{(l)} K^{(l)}`` only on compiler-certified non-angular axes.
They must be tied over exchangeable role orbits and accumulate directly into
the packed carrier arena rather than materializing a complete uncompressed
``A_s`` bank.

Reference boundary: the A_s Specht slot projectors use finite-group
central-idempotent character projectors for symmetric groups.  The
Young-orthogonal/subduction coefficient convention used by the product-layer
runtime is documented in ``ye3t.representations.young_orthogonal`` with
references to de Mello Koch, Ives, and Stephanou, J. Phys. A 45, 135204
(2012), doi:10.1088/1751-8113/45/13/135204, and Chilla's subduction-graph
papers, arXiv:math-ph/0512011 and arXiv:math-ph/0606037.  The slot projector
decomposes the natural slot permutation representation; higher Specht sectors
require tensor/product powers of slot states.
"""

from ye3t_ace._record import recordclass
from collections.abc import Mapping
from dataclasses import field
from functools import lru_cache
from itertools import permutations, product
from math import comb, factorial, isfinite
from pathlib import Path
import time

import numpy as np
import torch

from ye3t.representations import (
    Partition,
    all_permutations,
    canonical_irrep_matrices_numeric,
    inverse_permutation,
    permutation_cycle_type,
    slot_orbits,
    standard_tableaux,
    symmetric_group_character,
)
from ye3t.couplings import (
    coupling_paths_for_l_tuple as _ye3t_coupling_paths_for_l_tuple,
    integer_partitions as _ye3t_integer_partitions,
    slot_specht_partitions as _ye3t_slot_specht_partitions,
    validate_slot_permutation_scope as _ye3t_validate_slot_permutation_scope,
)
from ye3t.runtime.symmetric_power import (
    allowed_symmetric_power_outputs,
    symmetric_power_output_multiplicity,
    symmetric_power_real_tesseral,
)
from ye3t.paired_cg import couple_packed_real_tesseral

try:
    from ye3t_ace.equivariant_calc.edge_geometry import (
        directed_edges_all_images_bruteforce as _directed_edges_all_images,
        directed_edges_bruteforce as _directed_edges,
        edge_displacements_from_indices,
        normalize_pbc as _normalize_pbc,
        unique_periodic_cutoff_margin as _unique_periodic_cutoff_margin,
        voigt_from_stress_tensor as _voigt_from_stress_tensor,
    )
    from ye3t_ace.equivariant_calc.labeling import SingleChannelLabel
    from ye3t_ace.equivariant_calc.neighbors import neighbor_data_from_ase_atoms
    from ye3t_ace.equivariant_calc.site_basis_v2 import (
        SiteBasisConfig,
        SiteBasisV2,
        _native_density_backend_label,
        _native_edge_outer_backend_label,
        _source_kernel_backend,
        site_real_block_to_ye3t_tesseral as _site_real_block_to_ye3t_tesseral,
    )
    from ye3t_ace.utils.element_defaults import ordered_pair_values
except Exception:  # pragma: no cover - local import fallback
    from .equivariant_calc.edge_geometry import (
        directed_edges_all_images_bruteforce as _directed_edges_all_images,
        directed_edges_bruteforce as _directed_edges,
        edge_displacements_from_indices,
        normalize_pbc as _normalize_pbc,
        unique_periodic_cutoff_margin as _unique_periodic_cutoff_margin,
        voigt_from_stress_tensor as _voigt_from_stress_tensor,
    )
    from .equivariant_calc.labeling import SingleChannelLabel
    from .equivariant_calc.neighbors import neighbor_data_from_ase_atoms
    from .equivariant_calc.site_basis_v2 import (
        SiteBasisConfig,
        SiteBasisV2,
        _native_density_backend_label,
        _native_edge_outer_backend_label,
        _source_kernel_backend,
        site_real_block_to_ye3t_tesseral as _site_real_block_to_ye3t_tesseral,
    )
    from .utils.element_defaults import ordered_pair_values

try:
    from ase.calculators.calculator import Calculator as _ASECalculatorBase
    from ase.calculators.calculator import PropertyNotImplementedError as _ASEPropertyNotImplementedError
    from ase.calculators.calculator import all_changes as _ASE_ALL_CHANGES

    _ASE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on optional ASE
    _ASECalculatorBase = object
    _ASEPropertyNotImplementedError = RuntimeError
    _ASE_ALL_CHANGES = ("positions", "numbers", "cell", "pbc")
    _ASE_IMPORT_ERROR = exc


BRANCH_A = "A"
BRANCH_LIFTED_DENSITY = "A_s"
ALL_LIFTED_DENSITY_BRANCHES = (BRANCH_A, BRANCH_LIFTED_DENSITY)
DEFAULT_A_S_DENSITY_NORMALIZATION = "soft_neighbor"


def _normalize_reference_energies(reference_energies=None):
    if reference_energies is None:
        return {}
    return {str(key): float(value) for key, value in dict(reference_energies).items()}


def _reference_energy_offset_from_atoms(atoms, reference_energies=None):
    refs = _normalize_reference_energies(reference_energies)
    if not refs:
        return 0.0
    return float(sum(refs[str(symbol)] for symbol in atoms.get_chemical_symbols()))


def _atom_type_indices_from_atoms(atoms, type_map):
    return np.asarray([type_map[symbol] for symbol in atoms.get_chemical_symbols()], dtype=int)


def _neighbor_cache_is_valid(cache, positions, cell, atom_types, cutoff, skin):
    if cache is None or float(skin) <= 0.0:
        return False
    if abs(float(cache.get("cutoff", -1.0)) - float(cutoff)) > 1.0e-12:
        return False
    if np.asarray(cache["positions"]).shape != np.asarray(positions).shape:
        return False
    if not np.array_equal(np.asarray(cache["atom_types"], int), np.asarray(atom_types, int)):
        return False
    if not np.allclose(np.asarray(cache["cell"], float), np.asarray(cell, float), rtol=1.0e-12, atol=1.0e-12):
        return False
    displacement = np.asarray(positions, float) - np.asarray(cache["positions"], float)
    if displacement.size == 0:
        return True
    max_displacement = float(np.linalg.norm(displacement, axis=1).max())
    return max_displacement <= 0.5 * float(skin)


def _normalize_lifted_branch_name(name):
    value = str(name).strip()
    aliases = {
        "a": BRANCH_A,
        "ace": BRANCH_A,
        "A": BRANCH_A,
        "as": BRANCH_LIFTED_DENSITY,
        "a_s": BRANCH_LIFTED_DENSITY,
        "A_s": BRANCH_LIFTED_DENSITY,
        "lifted_density": BRANCH_LIFTED_DENSITY,
        "filtered_density": BRANCH_LIFTED_DENSITY,
    }
    out = aliases.get(value, aliases.get(value.lower(), value))
    if out not in ALL_LIFTED_DENSITY_BRANCHES:
        raise ValueError(
            f"Unknown lifted-density branch {name!r}; expected a subset of "
            f"{ALL_LIFTED_DENSITY_BRANCHES!r}."
        )
    return out


def normalize_lifted_density_branches(branches):
    """Return a deterministic branch tuple preserving user order."""

    seen = []
    for branch in branches:
        normalized = _normalize_lifted_branch_name(branch)
        if normalized not in seen:
            seen.append(normalized)
    if not seen:
        raise ValueError("At least one branch must be selected.")
    return tuple(seen)


def normalize_A_s_density_normalization_name(mode):
    """Normalize descriptor-level ``A_s`` density-normalization names.

    The default follows the ordinary ``A`` site-basis convention: individual
    radial/angular factors are bounded by ``SiteBasisV2`` and the accumulated
    atom-centered density is divided by the same smooth cutoff-weighted
    neighbor count used by ``atomic_base_normalization='soft_neighbor'``.
    ``degree`` and ``sqrt_degree`` remain available for older diagnostics as
    preconditioning choices.
    """

    normalized = str(mode or DEFAULT_A_S_DENSITY_NORMALIZATION).strip().lower().replace("-", "_")
    aliases = {
        "default": DEFAULT_A_S_DENSITY_NORMALIZATION,
        "bounded": DEFAULT_A_S_DENSITY_NORMALIZATION,
        "ace": DEFAULT_A_S_DENSITY_NORMALIZATION,
        "soft": "soft_neighbor",
        "soft_neighbor": "soft_neighbor",
        "soft_neighbor_count": "soft_neighbor",
        "none": "none",
        "off": "none",
        "false": "none",
        "degree": "degree",
        "neighbor_count": "degree",
        "num_neighbors": "degree",
        "sqrt_degree": "sqrt_degree",
        "sqrt_neighbor_count": "sqrt_degree",
        "sqrt_num_neighbors": "sqrt_degree",
    }
    if normalized not in aliases:
        raise ValueError(
            "A_s density_normalization must be one of "
            "'soft_neighbor', 'degree', 'sqrt_degree', or 'none'."
        )
    return aliases[normalized]


def normalize_A_s_density(
    density,
    *,
    src=None,
    soft_count=None,
    mode=None,
    nugget=0.0,
):
    """Apply descriptor-level normalization to filtered ``A_s`` densities."""

    mode = normalize_A_s_density_normalization_name(mode)
    if mode == "none":
        return density
    if mode == "soft_neighbor":
        if soft_count is None:
            raise ValueError("soft_neighbor A_s density normalization requires soft_count.")
        raw_scale = soft_count.to(dtype=density.dtype, device=density.device)
        nugget = float(nugget)
        if not isfinite(nugget) or nugget < 0.0:
            raise ValueError(
                "soft_neighbor A_s density normalization nugget must be "
                "finite and nonnegative."
            )
        if nugget > 0.0:
            scale = raw_scale + raw_scale.new_tensor(nugget)
        else:
            active = raw_scale > torch.finfo(raw_scale.dtype).eps
            scale = torch.where(
                active,
                raw_scale,
                torch.ones_like(raw_scale),
            )
        if scale.ndim == 2:
            return density / scale.unsqueeze(-1)
        if scale.ndim != 1:
            raise ValueError("soft_neighbor A_s density normalization expects soft_count with shape [n_atoms] or [n_atoms, n_slots].")
    else:
        if src is None:
            raise ValueError(f"{mode} A_s density normalization requires source atom indices.")
        degree = density.new_zeros((int(density.shape[0]),))
        if src.numel():
            degree.index_add_(
                0,
                src.to(device=density.device, dtype=torch.long),
                torch.ones((int(src.numel()),), dtype=density.dtype, device=density.device),
            )
        scale = torch.clamp(degree, min=1.0)
        if mode == "sqrt_degree":
            scale = torch.sqrt(scale)
    return density / scale.reshape(-1, 1, 1)


DEFAULT_A_S_FEATURE_NORMALIZATION = "none"


def normalize_A_s_feature_normalization_name(mode):
    """Normalize deprecated emitted-feature normalization names.

    A_s normalization is implemented on the basis side through bounded
    SiteBasisV2 factors and descriptor-level filtered-density normalization.
    Nonlinear post-evaluation feature squashing is intentionally unsupported
    because it changes the linear model from a linear readout over the A_s
    basis.
    """

    normalized = str(mode or DEFAULT_A_S_FEATURE_NORMALIZATION).strip().lower().replace("-", "_")
    aliases = {
        "default": DEFAULT_A_S_FEATURE_NORMALIZATION,
        "none": "none",
        "off": "none",
        "false": "none",
    }
    if normalized not in aliases:
        raise ValueError(
            "A_s feature_normalization is deprecated; use density_normalization="
            "'soft_neighbor' plus bounded SiteBasisV2 factors instead."
        )
    return aliases[normalized]


def radial_filter_constant_reproduction(
    filter_values,
    *,
    target=1.0,
):
    """Least-squares diagnostic for reduction to an ordinary ACE density.

    ``filter_values`` has shape ``[n_samples, n_filters]``.  The returned
    coefficients minimize ``||Q alpha - target||_2`` on the sampled radial grid.
    If the residual is zero on the active interval, then a linear readout over
    filtered densities can exactly reproduce the corresponding ordinary
    unfiltered density channel on that grid.
    """

    q = torch.as_tensor(filter_values)
    if q.ndim != 2:
        raise ValueError("filter_values must have shape [n_samples, n_filters].")
    y = torch.full((int(q.shape[0]),), float(target), dtype=q.dtype, device=q.device)
    solution = torch.linalg.lstsq(q, y.unsqueeze(1)).solution.squeeze(1)
    residual = q @ solution - y
    return {
        "coefficients": solution,
        "max_abs_error": torch.max(torch.abs(residual)),
        "rms_error": torch.sqrt(torch.mean(residual.pow(2))),
    }


def slot_trivial_component(density):
    """Return the trivial slot component ``mu_{i,c}``.

    ``density`` has shape ``[n_atoms, n_slots, n_channels]``.  The result has
    shape ``[n_atoms, n_channels]`` and is the slot average.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    return density.mean(dim=1)


def slot_standard_residual(density):
    """Return the standard-representation residual ``eta_{i,s,c}``."""

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    return density - slot_trivial_component(density).unsqueeze(1)


def slot_standard_norm(density):
    """Return ``nu_{i,c} = sum_s eta_{i,s,c}^2``."""

    eta = slot_standard_residual(density)
    return eta.pow(2).sum(dim=1)


def slot_antisymmetric_squared_volume(density):
    """Return the squared exterior volume of slot vectors for each atom.

    This probes the fully antisymmetric/sign sector through the Gram
    determinant ``det(A_i A_i^T)`` of the slot-by-channel matrix.  It is a
    scalar invariant under slot relabeling because a permutation conjugates the
    Gram matrix.  It vanishes when the slot vectors are linearly dependent, so
    it is a sign-sector magnitude diagnostic rather than a full sign-carrier
    runtime.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    gram = torch.matmul(density, density.transpose(1, 2))
    return torch.linalg.det(gram)


def lifted_density_channel_blocks(channels):
    """Return complete real-spherical channel blocks grouped by ``(n,l,type)``.

    A block is emitted only when all ``m=-l,...,l`` components are present
    exactly once.  The returned dictionaries include ``indices`` in increasing
    ``m`` order.
    """

    by_key = {}
    for index, channel in enumerate(channels):
        key = (int(channel.n), int(channel.l), channel.neighbor_type)
        by_key.setdefault(key, {})[int(channel.m)] = int(index)
    blocks = []
    for (n, l, neighbor_type), m_to_index in sorted(by_key.items(), key=lambda item: (item[0][1], item[0][0], -1 if item[0][2] is None else item[0][2])):
        expected = tuple(range(-int(l), int(l) + 1))
        if tuple(sorted(m_to_index)) != expected:
            continue
        blocks.append(
            {
                "n": int(n),
                "l": int(l),
                "neighbor_type": neighbor_type,
                "m_values": expected,
                "indices": tuple(int(m_to_index[m]) for m in expected),
            }
        )
    return tuple(blocks)


def ye3_quadratic_invariants(density, channels):
    """Return blockwise trivial/standard Young-sector E3 scalar invariants.

    For each complete ``(n,l,type)`` real-spherical block, this returns
    ``||mu_l||^2`` and ``sum_s ||eta_{s,l}||^2``.  For ``l=0`` these are the
    squared scalar trivial and standard-slot-sector magnitudes; for ``l>0`` they
    are E3 scalar norms of covariant channel blocks.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    blocks = lifted_density_channel_blocks(channels)
    if not blocks:
        return density.new_zeros((int(density.shape[0]), 0)), blocks
    mu = slot_trivial_component(density)
    eta = slot_standard_residual(density)
    invariants = []
    for block in blocks:
        indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
        mu_block = mu.index_select(1, indices)
        eta_block = eta.index_select(2, indices)
        invariants.append(mu_block.pow(2).sum(dim=1))
        invariants.append(eta_block.pow(2).sum(dim=(1, 2)))
    return torch.stack(invariants, dim=1), blocks


def _normalize_ye3_slot_sectors(slot_sectors):
    if slot_sectors is None:
        return ("trivial", "standard")
    out = tuple(str(sector).strip().lower() for sector in slot_sectors)
    allowed = {"trivial", "standard"}
    if not out:
        raise ValueError("ye3_slot_sectors must contain at least one sector.")
    bad = tuple(sector for sector in out if sector not in allowed)
    if bad:
        raise ValueError(
            "ye3_slot_sectors currently supports only 'trivial' and 'standard' "
            f"for scalar ye3_power readouts; got {bad!r}."
        )
    return out


def _integer_partitions(n, max_part=None):
    return tuple(_ye3t_integer_partitions(n, max_part))


def _normalize_slot_specht_partitions(value, *, slot_count):
    return tuple(_ye3t_slot_specht_partitions(value, slot_count=slot_count))


@lru_cache(maxsize=None)
def _slot_tuple_basis(slot_count, power):
    return tuple(product(range(int(slot_count)), repeat=int(power)))


@lru_cache(maxsize=None)
def _slot_value_action_matrix(slot_count, power, perm):
    slot_count = int(slot_count)
    power = int(power)
    perm = tuple(int(value) for value in perm)
    if len(perm) != slot_count:
        raise ValueError("perm length must equal slot_count.")
    basis = _slot_tuple_basis(slot_count, power)
    index = {state: idx for idx, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.float64)
    for col, state in enumerate(basis):
        row_state = tuple(int(perm[int(value)]) for value in state)
        matrix[index[row_state], col] = 1.0
    return matrix


@lru_cache(maxsize=None)
def _slot_specht_central_projector_np(slot_count, power, partition):
    slot_count = int(slot_count)
    power = int(power)
    partition = tuple(int(part) for part in partition)
    if sum(partition) != slot_count:
        raise ValueError("Slot Specht partition size must equal slot_count.")
    if slot_count > 8:
        raise NotImplementedError("Slot Specht projectors currently use small symmetric groups up to S8.")
    basis = _slot_tuple_basis(slot_count, power)
    projector = np.zeros((len(basis), len(basis)), dtype=np.float64)
    prefactor = float(len(standard_tableaux(partition))) / float(factorial(slot_count))
    for perm in all_permutations(slot_count):
        char = symmetric_group_character(Partition(partition), permutation_cycle_type(perm))
        if int(char) == 0:
            continue
        projector += prefactor * float(int(char)) * _slot_value_action_matrix(slot_count, power, perm)
    rank = int(np.linalg.matrix_rank(projector, tol=1.0e-10))
    idempotent = bool(np.allclose(projector @ projector, projector, atol=1.0e-10, rtol=1.0e-10))
    return projector, rank, idempotent


def _slot_specht_central_projector_torch(slot_count, power, partition, *, dtype, device):
    projector, rank, idempotent = _slot_specht_central_projector_np(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
    )
    dense = torch.as_tensor(projector, dtype=dtype, device=device)
    return dense, int(rank), bool(idempotent)


_SLOT_SPECHT_PROJECTOR_TORCH_CACHE = {}
_SLOT_SPECHT_COMMUTANT_SYMMETRIC_TORCH_CACHE = {}
_SLOT_SPECHT_MATRIX_UNITS_TORCH_CACHE = {}
_SLOT_SPECHT_MULTIPLICITY_BASIS_TORCH_CACHE = {}


def _slot_specht_central_projector_torch_cached(slot_count, power, partition, *, dtype, device):
    """Return a cached dense torch central idempotent for the diagonal slot action.

    The exact SymPy construction is cached separately, but converting the same
    projector into a device tensor for every structure/sector is still
    avoidable overhead in descriptor-matrix builds.  This cache is deliberately
    limited to tensors with no gradient history; it does not change the
    mathematical object being applied.
    """

    device = torch.device(device)
    key = (
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
        str(dtype),
        str(device),
    )
    cached = _SLOT_SPECHT_PROJECTOR_TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    projector, rank, idempotent = _slot_specht_central_projector_torch(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
        dtype=dtype,
        device=device,
    )
    _SLOT_SPECHT_PROJECTOR_TORCH_CACHE[key] = (projector, int(rank), bool(idempotent))
    return _SLOT_SPECHT_PROJECTOR_TORCH_CACHE[key]


@lru_cache(maxsize=None)
def _slot_specht_matrix_units_np(slot_count, power, partition):
    """Return Young-orthogonal matrix units on the slot tuple-power module."""

    from ye3t.couplings import compile_A_s_slot_specht_matrix_units

    report = compile_A_s_slot_specht_matrix_units(
        slot_count=int(slot_count),
        power=int(power),
        partition=tuple(int(part) for part in partition),
        max_slot_count=6,
    )
    specht_dim = int(report.specht_dimension)
    tuple_dim = int(report.slot_count) ** int(report.power)
    units = np.zeros((specht_dim, specht_dim, tuple_dim, tuple_dim), dtype=np.float64)
    for record in report.records:
        units[int(record["row"]), int(record["col"])] = np.asarray(record["matrix"], dtype=np.float64)
    projector = np.asarray(report.projector, dtype=np.float64)
    idempotent = bool(report.validation_report.get("max_projector_idempotency_error", 1.0) <= 1.0e-10)
    return units, projector, int(report.rank), bool(idempotent)


def _slot_specht_matrix_units_torch_cached(slot_count, power, partition, *, dtype, device):
    device = torch.device(device)
    key = (
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
        str(dtype),
        str(device),
    )
    cached = _SLOT_SPECHT_MATRIX_UNITS_TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    units_np, projector_np, rank, idempotent = _slot_specht_matrix_units_np(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
    )
    units = torch.as_tensor(units_np, dtype=dtype, device=device)
    projector = torch.as_tensor(projector_np, dtype=dtype, device=device)
    _SLOT_SPECHT_MATRIX_UNITS_TORCH_CACHE[key] = (units, projector, int(rank), bool(idempotent))
    return _SLOT_SPECHT_MATRIX_UNITS_TORCH_CACHE[key]


@lru_cache(maxsize=None)
def _slot_specht_anchor_multiplicity_basis_np(slot_count, power, partition, anchor_index=0):
    slot_count = int(slot_count)
    power = int(power)
    partition = tuple(int(part) for part in partition)
    anchor_index = int(anchor_index)
    units_np, _projector_np, _rank, _idempotent = _slot_specht_matrix_units_np(
        slot_count,
        power,
        partition,
    )
    specht_dim = int(units_np.shape[0])
    if anchor_index < 0 or anchor_index >= specht_dim:
        raise ValueError(
            f"anchor_index={anchor_index} is outside the Specht tableau range [0, {specht_dim})."
        )
    anchor_projector = np.asarray(units_np[anchor_index, anchor_index], dtype=np.float64)
    if anchor_projector.size == 0:
        return np.zeros((int(slot_count) ** int(power), 0), dtype=np.float64)
    u, singular_values, _vh = np.linalg.svd(anchor_projector, full_matrices=False)
    keep = singular_values > 1.0e-10
    if not np.any(keep):
        return np.zeros((anchor_projector.shape[0], 0), dtype=np.float64)
    return np.asarray(u[:, keep], dtype=np.float64)


def _slot_specht_anchor_multiplicity_basis_torch_cached(
    slot_count,
    power,
    partition,
    *,
    anchor_index=0,
    dtype,
    device,
):
    device = torch.device(device)
    key = (
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
        int(anchor_index),
        str(dtype),
        str(device),
    )
    cached = _SLOT_SPECHT_MULTIPLICITY_BASIS_TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    basis_np = _slot_specht_anchor_multiplicity_basis_np(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
        int(anchor_index),
    )
    if int(basis_np.shape[1]) == 0:
        basis = torch.zeros(
            (int(slot_count) ** int(power), 0),
            dtype=dtype,
            device=device,
        )
    else:
        basis = torch.as_tensor(basis_np, dtype=dtype, device=device)
    _SLOT_SPECHT_MULTIPLICITY_BASIS_TORCH_CACHE[key] = basis
    return basis


def slot_specht_matrix_unit_carriers(
    carrier,
    *,
    slot_count,
    power,
    partition,
):
    """Project tuple-power slot carriers with explicit Specht matrix units.

    ``carrier`` must have shape ``[batch, slot_count**power]``.  The returned
    tensor has shape ``[batch, d_lambda, d_lambda, slot_count**power]`` and
    exposes the matrix-unit-resolved images ``c E_ab^T``.  Diagonal slices
    ``a=a`` are multiplicity-space copies; summing those diagonal projectors
    recovers the central projector image.
    """

    values = torch.as_tensor(carrier)
    if values.ndim != 2:
        raise ValueError("slot_specht_matrix_unit_carriers expects shape [batch, slot_count**power].")
    expected_dim = int(slot_count) ** int(power)
    if int(values.shape[1]) != expected_dim:
        raise ValueError(
            f"Carrier dimension must be slot_count**power={expected_dim}; got {int(values.shape[1])}."
        )
    units, _projector, _rank, _idempotent = _slot_specht_matrix_units_torch_cached(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
        dtype=values.dtype,
        device=values.device,
    )
    return torch.einsum("nd,aced->nace", values, units)


def _slot_permutations_for_matrix_unit_covariance(slot_count, permutations_to_test=None):
    slot_count = int(slot_count)
    if permutations_to_test is not None:
        out = []
        for perm in permutations_to_test:
            normalized = tuple(int(value) for value in perm)
            if sorted(normalized) != list(range(slot_count)):
                raise ValueError("Each slot permutation must be a permutation of range(slot_count).")
            if normalized not in out:
                out.append(normalized)
        return tuple(out)

    out = []
    for index in range(max(slot_count - 1, 0)):
        perm = list(range(slot_count))
        perm[index], perm[index + 1] = perm[index + 1], perm[index]
        out.append(tuple(perm))
    if slot_count > 2:
        cycle = tuple(list(range(1, slot_count)) + [0])
        if cycle not in out:
            out.append(cycle)
    return tuple(out)


def validate_slot_specht_matrix_unit_carrier_covariance(
    carrier,
    *,
    slot_count,
    power,
    partition,
    permutations_to_test=None,
    atol=1.0e-10,
    rtol=1.0e-10,
):
    """Validate the slot-permutation law for evaluated matrix-unit carriers.

    ``carrier`` may be either a projected tuple-power carrier with shape
    ``[batch, slot_count**power]`` or a matrix-unit carrier with shape
    ``[batch, d_lambda, d_lambda, slot_count**power]``.  The check uses the same
    row-vector slot action and Young-orthogonal irrep convention as
    :func:`slot_specht_matrix_unit_carriers`.
    """

    values = torch.as_tensor(carrier)
    expected_tuple_dim = int(slot_count) ** int(power)
    if values.ndim == 2:
        if int(values.shape[1]) != expected_tuple_dim:
            raise ValueError(
                "Projected carrier dimension must equal slot_count**power; "
                f"got {int(values.shape[1])} and expected {expected_tuple_dim}."
            )
        projected = values
        matrix_carrier = slot_specht_matrix_unit_carriers(
            projected,
            slot_count=int(slot_count),
            power=int(power),
            partition=tuple(int(part) for part in partition),
        )
        source_kind = "projected_slot_tuple_carrier"
    elif values.ndim == 4:
        if int(values.shape[-1]) != expected_tuple_dim:
            raise ValueError(
                "Matrix-unit carrier tuple axis must equal slot_count**power; "
                f"got {int(values.shape[-1])} and expected {expected_tuple_dim}."
            )
        if int(values.shape[1]) != int(values.shape[2]):
            raise ValueError("Matrix-unit carrier tableau row/column axes must be square.")
        matrix_carrier = values
        projected = matrix_carrier.diagonal(dim1=1, dim2=2).sum(dim=-1)
        source_kind = "matrix_unit_carrier_diagonal_trace"
    else:
        raise ValueError(
            "Matrix-unit covariance validation expects a projected carrier with shape "
            "[batch, carrier_dim] or a matrix-unit carrier with shape "
            "[batch, d_lambda, d_lambda, carrier_dim]."
        )

    partition = tuple(int(part) for part in partition)
    permutations = _slot_permutations_for_matrix_unit_covariance(
        int(slot_count),
        permutations_to_test=permutations_to_test,
    )
    irrep_matrices = canonical_irrep_matrices_numeric(partition)
    records = []
    max_error = 0.0
    max_threshold = 0.0
    for perm in permutations:
        action = torch.as_tensor(
            _slot_value_action_matrix(int(slot_count), int(power), perm),
            dtype=projected.dtype,
            device=projected.device,
        )
        rho = torch.as_tensor(irrep_matrices[perm], dtype=projected.dtype, device=projected.device)
        relabeled = slot_specht_matrix_unit_carriers(
            projected @ action,
            slot_count=int(slot_count),
            power=int(power),
            partition=partition,
        )
        predicted = torch.einsum("bc,nace->nabe", rho, matrix_carrier)
        error = float(torch.max(torch.abs(relabeled - predicted)).detach().cpu()) if relabeled.numel() else 0.0
        scale = float(torch.max(torch.abs(relabeled)).detach().cpu()) if relabeled.numel() else 0.0
        threshold = float(atol) + float(rtol) * max(1.0, scale)
        max_error = max(max_error, error)
        max_threshold = max(max_threshold, threshold)
        records.append(
            {
                "permutation": tuple(int(value) for value in perm),
                "inverse_permutation": inverse_permutation(perm),
                "max_covariance_error": error,
                "threshold": threshold,
                "passed": bool(error <= threshold),
            }
        )

    return {
        "passed": all(record["passed"] for record in records),
        "slot_group": f"S_{int(slot_count)}",
        "slot_count": int(slot_count),
        "power": int(power),
        "partition": partition,
        "carrier_shape": tuple(int(dim) for dim in values.shape),
        "source_kind": source_kind,
        "tested_permutations": tuple(record["permutation"] for record in records),
        "permutation_count": int(len(records)),
        "max_covariance_error": max_error,
        "max_threshold": max_threshold,
        "atol": float(atol),
        "rtol": float(rtol),
        "representation_law": "row_vector_slot_action_with_right_tableau_index_rho_mixing",
        "records": tuple(records),
    }


def validate_slot_specht_matrix_units(slot_count, power, partition):
    """Validate matrix-unit algebra for a small slot Specht carrier."""

    units, projector, rank, idempotent = _slot_specht_matrix_units_np(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
    )
    from ye3t.couplings import compile_A_s_slot_specht_matrix_units

    compiled_report = compile_A_s_slot_specht_matrix_units(
        slot_count=int(slot_count),
        power=int(power),
        partition=tuple(int(part) for part in partition),
        max_slot_count=6,
    )
    specht_dim = len(units)
    tuple_dim = int(slot_count) ** int(power)
    max_residual = 0.0
    for a in range(specht_dim):
        for b in range(specht_dim):
            for c in range(specht_dim):
                for d in range(specht_dim):
                    expected = units[a, d] if b == c else np.zeros((tuple_dim, tuple_dim), dtype=np.float64)
                    residual = units[a, b] @ units[c, d] - expected
                    if residual.size:
                        max_residual = max(max_residual, float(np.max(np.abs(residual))))
    diagonal_sum = np.zeros((tuple_dim, tuple_dim), dtype=np.float64)
    for a in range(specht_dim):
        diagonal_sum += units[a, a]
    central_matches = bool(np.allclose(diagonal_sum, projector, atol=1.0e-10, rtol=1.0e-10))
    diagonal_ranks = tuple(int(np.linalg.matrix_rank(units[a, a], tol=1.0e-10)) for a in range(specht_dim))
    expected_multiplicity = _slot_specht_permutation_module_multiplicity_exact(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
    )
    return {
        "slot_group": f"S_{int(slot_count)}",
        "power": int(power),
        "partition": tuple(int(part) for part in partition),
        "coefficient_source": "ye3t.couplings.compile_A_s_slot_specht_matrix_units",
        "compiled_matrix_unit_convention_hash": str(compiled_report.convention_hash),
        "compiled_matrix_unit_backend": str(compiled_report.backend),
        "compiled_matrix_unit_provenance": dict(compiled_report.provenance),
        "compiled_matrix_unit_validation": dict(compiled_report.validation_report),
        "specht_dimension": int(specht_dim),
        "isotypic_rank": int(rank),
        "central_projector_idempotent": bool(idempotent),
        "central_projector_matches_diagonal_sum": bool(central_matches),
        "matrix_unit_algebra_max_residual": float(max_residual),
        "matrix_unit_algebra_residual_zero": bool(max_residual <= 1.0e-10),
        "diagonal_ranks": diagonal_ranks,
        "expected_multiplicity": int(expected_multiplicity),
        "diagonal_ranks_match_multiplicity": all(int(value) == int(expected_multiplicity) for value in diagonal_ranks),
        "passed": bool(
            idempotent
            and central_matches
            and max_residual <= 1.0e-10
            and all(int(value) == int(expected_multiplicity) for value in diagonal_ranks)
        ),
    }


@lru_cache(maxsize=None)
def _slot_specht_tensor_product_multiplicity_exact(slot_count, left_partition, right_partition, target_partition):
    """Return exact multiplicity of ``target`` in ``left x right*`` for ``S_slot_count``.

    Symmetric-group irreducible characters are real-valued in the current
    backend, so the dual character equals the original character in this
    finite-group character inner product:

        <chi_left chi_right, chi_target> = |G|^{-1} sum_g chi_left(g) chi_right(g) chi_target(g).

    This plans multiplicities for future A_s matrix-unit couplings; it does
    not construct those runtime coupling matrices.
    """

    slot_count = int(slot_count)
    left_partition = tuple(int(part) for part in left_partition)
    right_partition = tuple(int(part) for part in right_partition)
    target_partition = tuple(int(part) for part in target_partition)
    for partition in (left_partition, right_partition, target_partition):
        if sum(partition) != slot_count or partition not in _integer_partitions(slot_count):
            raise ValueError(f"Invalid S_{slot_count} Specht partition {partition!r}.")
    if slot_count > 8:
        raise NotImplementedError("Exact slot Specht tensor-product multiplicities currently use S_n up to n=8.")
    total = 0
    left = Partition(left_partition)
    right = Partition(right_partition)
    target = Partition(target_partition)
    for perm in all_permutations(slot_count):
        cycle_type = permutation_cycle_type(perm)
        total += (
            int(symmetric_group_character(left, cycle_type))
            * int(symmetric_group_character(right, cycle_type))
            * int(symmetric_group_character(target, cycle_type))
        )
    group_order = factorial(slot_count)
    if int(total) % int(group_order) != 0:
        raise RuntimeError(
            "Internal symmetric-group tensor-product multiplicity was not integral; "
            f"got numerator={total!r}, denominator={group_order!r} for "
            f"{left_partition} x {right_partition} -> {target_partition}."
        )
    return int(total // group_order)


@lru_cache(maxsize=None)
def _slot_specht_permutation_module_multiplicity_exact(slot_count, power, partition):
    """Return multiplicity of ``S^partition`` in the slot tensor-power module.

    For the natural slot permutation module ``U`` of ``S_s``, the character of
    ``U^{tensor power}`` at a permutation is ``fixed_points(g) ** power``.  The
    multiplicity of ``S^partition`` is the exact character inner product with
    this permutation-module character.  This avoids materializing the
    ``s**power`` central projector when only an inventory/planning rank is
    needed.
    """

    slot_count = int(slot_count)
    power = int(power)
    partition = tuple(int(part) for part in partition)
    if power < 0:
        raise ValueError("power must be nonnegative.")
    if sum(partition) != slot_count or partition not in _integer_partitions(slot_count):
        raise ValueError(f"Invalid S_{slot_count} Specht partition {partition!r}.")
    if slot_count > 8:
        raise NotImplementedError("Exact slot Specht permutation-module multiplicities currently use S_n up to n=8.")
    total = 0
    specht = Partition(partition)
    for perm in all_permutations(slot_count):
        cycle_type = permutation_cycle_type(perm)
        fixed_points = sum(1 for index, image in enumerate(perm) if int(index) == int(image))
        total += int(symmetric_group_character(specht, cycle_type)) * int(fixed_points) ** int(power)
    group_order = factorial(slot_count)
    if int(total) % int(group_order) != 0:
        raise RuntimeError(
            "Internal slot Specht permutation-module multiplicity was not integral; "
            f"got numerator={total!r}, denominator={group_order!r} for "
            f"S_{slot_count}, power={power}, partition={partition}."
        )
    return int(total // group_order)


def _slot_specht_isotypic_report(slot_count, power, partition):
    slot_count = int(slot_count)
    power = int(power)
    partition = tuple(int(part) for part in partition)
    specht_dimension = len(standard_tableaux(partition))
    isotypic_multiplicity = _slot_specht_permutation_module_multiplicity_exact(
        slot_count,
        power,
        partition,
    )
    projector_rank = int(specht_dimension) * int(isotypic_multiplicity)
    return {
        "slot_group": f"S_{slot_count}",
        "power": int(power),
        "partition": partition,
        "specht_dimension": int(specht_dimension),
        "isotypic_multiplicity": int(isotypic_multiplicity),
        "projector_rank": int(projector_rank),
        "method": "exact_character_inner_product_with_slot_permutation_module",
    }


def slot_specht_tensor_product_multiplicities(
    slot_count,
    left_partition,
    right_partition,
    *,
    target_partitions=None,
):
    """Plan exact diagonal slot Specht tensor-product multiplicities.

    Returns records for finite-group multiplicities in
    ``S^left x (S^right)^* -> S^target`` under the diagonal ``S_slot_count``
    action. This helper reports multiplicities for a selected finite slot group.
    """

    slot_count = int(slot_count)
    left_partition = tuple(int(part) for part in left_partition)
    right_partition = tuple(int(part) for part in right_partition)
    if target_partitions is None:
        candidates = _integer_partitions(slot_count)
    else:
        candidates = tuple(tuple(int(part) for part in partition) for partition in target_partitions)
    records = []
    for target_partition in candidates:
        multiplicity = _slot_specht_tensor_product_multiplicity_exact(
            slot_count,
            left_partition,
            right_partition,
            tuple(target_partition),
        )
        if int(multiplicity) <= 0:
            continue
        records.append(
            {
                "slot_group": f"S_{slot_count}",
                "left_partition": left_partition,
                "right_dual_partition": right_partition,
                "target_partition": tuple(int(part) for part in target_partition),
                "multiplicity": int(multiplicity),
                "method": "exact_character_inner_product",
                "scope": "multiplicity_planning_not_runtime_matrix_units",
            }
        )
    return tuple(records)


def slot_specht_trivial_coupling_plan(
    slot_count,
    left_partition,
    right_partition=None,
    *,
    left_power,
    right_power=None,
):
    """Plan scalar trivial-target couplings for diagonal slot Specht carriers.

    This is an exact representation-planning helper for the slot group
    ``S_slot_count``.  It reports the trivial-target multiplicity in
    ``S^left x (S^right)^*`` and the multiplicity-space size of each central
    projector image inside the slot tensor powers.

    The currently implemented scalar runtime uses one summed carrier-dual
    pairing when the left and right carriers are the same projected tuple-power
    space.  A full multiplicity-resolved backend would expose the matrix units
    between the left/right isotypic multiplicity spaces instead of collapsing
    them to that one scalar.
    """

    slot_count = int(slot_count)
    left_partition = tuple(int(part) for part in left_partition)
    right_partition = left_partition if right_partition is None else tuple(int(part) for part in right_partition)
    left_power = int(left_power)
    right_power = left_power if right_power is None else int(right_power)
    left_report = _slot_specht_isotypic_report(slot_count, left_power, left_partition)
    right_report = _slot_specht_isotypic_report(slot_count, right_power, right_partition)
    trivial_multiplicity = _slot_specht_tensor_product_multiplicity_exact(
        slot_count,
        left_partition,
        right_partition,
        (slot_count,),
    )
    full_matrix_unit_count = (
        int(left_report["isotypic_multiplicity"])
        * int(right_report["isotypic_multiplicity"])
        * int(trivial_multiplicity)
    )
    symmetric_quadratic_count = (
        int(left_report["isotypic_multiplicity"]) * (int(left_report["isotypic_multiplicity"]) + 1) // 2
        if left_partition == right_partition and left_power == right_power and int(trivial_multiplicity) > 0
        else 0
    )
    restricted_runtime_supported = (
        left_partition == right_partition
        and left_power == right_power
        and int(trivial_multiplicity) > 0
        and int(left_report["projector_rank"]) == int(right_report["projector_rank"])
    )
    implemented_scalar_count = 1 if restricted_runtime_supported and full_matrix_unit_count > 0 else 0
    return {
        "slot_group": f"S_{slot_count}",
        "left_partition": left_partition,
        "right_dual_partition": right_partition,
        "target_partition": (slot_count,),
        "left_power": int(left_power),
        "right_power": int(right_power),
        "target_trivial_multiplicity": int(trivial_multiplicity),
        "left_isotypic": left_report,
        "right_isotypic": right_report,
        "full_trivial_matrix_unit_count": int(full_matrix_unit_count),
        "full_symmetric_quadratic_count": int(symmetric_quadratic_count),
        "implemented_runtime_scalar_count": int(implemented_scalar_count),
        "implemented_runtime": (
            "summed_identity_pairing_on_common_projected_tuple_carrier"
            if implemented_scalar_count
            else "not_implemented_for_this_left_right_pair"
        ),
        "scope": "trivial_target_plan_with_restricted_runtime_pairing",
        "requires_full_matrix_units_for_multiplicity_resolution": bool(
            int(full_matrix_unit_count) != int(implemented_scalar_count)
        ),
    }


@lru_cache(maxsize=None)
def _slot_specht_commutant_symmetric_basis_np(slot_count, power, partition):
    """Return a Frobenius-orthonormal symmetric commutant basis on ``P_lambda U``.

    The construction starts from orbital matrices of the diagonal slot action
    on ordered tuple pairs and projects them with the central idempotent
    ``P_lambda``.  The resulting span is the symmetric part of the commutant on
    the selected isotypic image, i.e. the scalar quadratic forms that satisfy
    ``q(c rho(g)) = q(c)`` for the row-vector slot action.  This is a runtime
    bridge toward multiplicity-resolved scalar couplings; it is still not an
    explicit Young-orthogonal matrix-unit basis.
    """

    slot_count = int(slot_count)
    power = int(power)
    partition = tuple(int(part) for part in partition)
    if slot_count > 5 or power > 5:
        raise NotImplementedError(
            "Dense slot-Specht commutant symmetric bases are currently enabled for slot_count<=5 and power<=5."
        )
    projector, projector_rank, idempotent = _slot_specht_central_projector_np(
        slot_count,
        power,
        partition,
    )
    if not bool(idempotent) or int(projector_rank) <= 0:
        return np.zeros((0, slot_count**power, slot_count**power), dtype=np.float64)
    tuples = list(product(range(slot_count), repeat=power))
    tuple_to_index = {tuple_value: index for index, tuple_value in enumerate(tuples)}
    tuple_count = len(tuples)
    visited = np.zeros((tuple_count, tuple_count), dtype=bool)
    vectors = []
    perms = tuple(all_permutations(slot_count))
    for left_index, left_tuple in enumerate(tuples):
        for right_index, right_tuple in enumerate(tuples):
            if visited[left_index, right_index]:
                continue
            matrix = np.zeros((tuple_count, tuple_count), dtype=np.float64)
            orbit = []
            for perm in perms:
                permuted_left = tuple(int(perm[value]) for value in left_tuple)
                permuted_right = tuple(int(perm[value]) for value in right_tuple)
                orbit.append((tuple_to_index[permuted_left], tuple_to_index[permuted_right]))
            for row_index, col_index in orbit:
                visited[row_index, col_index] = True
                matrix[row_index, col_index] = 1.0
            restricted = projector @ (0.5 * (matrix + matrix.T)) @ projector
            restricted = 0.5 * (restricted + restricted.T)
            norm = float(np.linalg.norm(restricted))
            if norm > 1.0e-12:
                vectors.append((restricted / norm).reshape(-1))
    if not vectors:
        return np.zeros((0, tuple_count, tuple_count), dtype=np.float64)
    stacked = np.stack(vectors, axis=0)
    _u, singular_values, vt = np.linalg.svd(stacked, full_matrices=False)
    rank = int(np.sum(singular_values > 1.0e-10 * max(float(singular_values[0]), 1.0)))
    basis = vt[:rank].reshape(rank, tuple_count, tuple_count)
    basis = 0.5 * (basis + np.swapaxes(basis, 1, 2))
    norms = np.linalg.norm(basis.reshape(rank, -1), axis=1)
    basis = basis / np.maximum(norms, 1.0e-30).reshape(-1, 1, 1)
    return basis.astype(np.float64, copy=False)


def _slot_specht_commutant_symmetric_basis_torch_cached(slot_count, power, partition, *, dtype, device):
    device = torch.device(device)
    key = (
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
        str(dtype),
        str(device),
    )
    cached = _SLOT_SPECHT_COMMUTANT_SYMMETRIC_TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    basis_np = _slot_specht_commutant_symmetric_basis_np(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
    )
    basis = torch.as_tensor(basis_np, dtype=dtype, device=device)
    _SLOT_SPECHT_COMMUTANT_SYMMETRIC_TORCH_CACHE[key] = basis
    return basis


def slot_specht_projected_commutant_quadratic_features(left_carrier, *, slot_count, power, partition):
    """Return invariant quadratic features from the symmetric commutant basis.

    ``left_carrier`` must already be a projected slot-Specht carrier with shape
    ``[n_atoms, slot_count**power]``.  The returned columns are
    ``c B_a c^T`` for a Frobenius-orthonormal basis ``B_a`` of symmetric
    invariant forms on the selected central-projector image.
    """

    carrier = torch.as_tensor(left_carrier)
    if carrier.ndim != 2:
        raise ValueError("Projected slot-Specht carriers must have shape [n_atoms, carrier_dim].")
    expected_dim = int(slot_count) ** int(power)
    if int(carrier.shape[1]) != expected_dim:
        raise ValueError(
            "Projected slot-Specht carrier dimension does not match slot_count**power; "
            f"got {int(carrier.shape[1])} and expected {expected_dim}."
        )
    basis = _slot_specht_commutant_symmetric_basis_torch_cached(
        int(slot_count),
        int(power),
        tuple(int(part) for part in partition),
        dtype=carrier.dtype,
        device=carrier.device,
    )
    if int(basis.shape[0]) == 0:
        return carrier.new_zeros((int(carrier.shape[0]), 0))
    return torch.einsum("bi,kij,bj->bk", carrier, basis, carrier)


def slot_specht_sector_feature_slices(sectors):
    """Return feature-column slices for slot-Specht sectors.

    ``ye3_slot_specht_power_invariants`` concatenates one block of
    columns per sector, with width equal to ``sector["multiplicity"]``.  This
    helper makes that column layout explicit for fitting/adjoint code.
    """

    slices = []
    start = 0
    for sector in sectors:
        width = int(sector.get("multiplicity", 1))
        if width < 0:
            raise ValueError("Slot-Specht sector multiplicity must be nonnegative.")
        stop = start + width
        slices.append(slice(start, stop))
        start = stop
    return tuple(slices)


def _left_associated_angular_paths_to_target(l_in, target_L_R):
    l_values = tuple(int(value) for value in l_in)
    target_L_R = int(target_L_R)
    if not l_values:
        return tuple()
    if len(l_values) == 1:
        return (tuple(),) if l_values[0] == target_L_R else tuple()
    if len(l_values) == 2:
        return (tuple(),) if abs(l_values[0] - l_values[1]) <= target_L_R <= l_values[0] + l_values[1] else tuple()
    paths = []

    def rec(index, current_L, prefix):
        ell = int(l_values[index])
        for next_L in range(abs(int(current_L) - ell), int(current_L) + ell + 1):
            if index == len(l_values) - 1:
                if int(next_L) == target_L_R:
                    paths.append(tuple(prefix))
            else:
                rec(index + 1, next_L, prefix + [next_L])

    rec(1, l_values[0], [])
    return tuple(paths)


def _normalize_rank_limit_mapping(value, *, name):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {int(key): int(item) for key, item in value.items()}
    values = tuple(int(item) for item in value)
    return {index + 1: item for index, item in enumerate(values)}


def _parse_ordered_pair_key(key, *, name):
    if isinstance(key, str):
        cleaned = key.strip().replace("->", ",").replace(":", ",").replace("-", ",")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"{name} pair key {key!r} must encode two integer type ids.")
        return (int(parts[0]), int(parts[1]))
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return (int(key[0]), int(key[1]))
    raise ValueError(f"{name} pair key {key!r} must be a length-2 pair or string such as '0,1'.")


def _normalize_pair_float_mapping(value, *, name):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            _parse_ordered_pair_key(key, name=name): float(item)
            for key, item in value.items()
        }
    out = {}
    for item in value:
        if isinstance(item, Mapping):
            if "pair" in item:
                pair = _parse_ordered_pair_key(item["pair"], name=name)
            else:
                pair = (int(item.get("center_type", item.get("center"))), int(item.get("neighbor_type", item.get("neighbor"))))
            if "value" in item:
                out[pair] = float(item["value"])
            elif name in item:
                out[pair] = float(item[name])
            else:
                raise ValueError(f"{name} pair record {item!r} must contain 'value' or {name!r}.")
        else:
            if len(item) != 3:
                raise ValueError(f"{name} sequence records must be (center_type, neighbor_type, value).")
            out[(int(item[0]), int(item[1]))] = float(item[2])
    return out


def _normalize_pair_filter_specs(value, *, default_num_filters, default_centers, default_width, default_kind):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        iterator = value.items()
    else:
        records = []
        for item in value:
            if not isinstance(item, Mapping):
                raise ValueError("pair_filter_specs sequence entries must be mappings.")
            if "pair" in item:
                pair = item["pair"]
            else:
                pair = (item.get("center_type", item.get("center")), item.get("neighbor_type", item.get("neighbor")))
            records.append((pair, item))
        iterator = records
    out = {}
    for key, item in iterator:
        pair = _parse_ordered_pair_key(key, name="pair_filter_specs")
        spec = dict(item)
        centers = tuple(float(x) for x in spec.get("filter_centers", spec.get("centers", ())))
        active_count = int(spec.get("num_filters", spec.get("active_filters", len(centers) if centers else default_num_filters)))
        if active_count < 1:
            raise ValueError("pair_filter_specs active filter count must be positive.")
        if active_count > int(default_num_filters):
            raise ValueError(
                "pair_filter_specs cannot request more active filters than the global num_filters; "
                f"got {active_count} > {int(default_num_filters)} for pair {pair}."
            )
        if not centers:
            if str(spec.get("filter_kind", default_kind)).strip().lower() in {"radial_gaussian", "softmax_gaussian", "cosine_shell"}:
                centers = (0.5,) if active_count == 1 else tuple(float(x) for x in np.linspace(0.2, 0.8, active_count))
            elif str(spec.get("filter_kind", default_kind)).strip().lower() == "bernstein":
                centers = (0.5,) if active_count == 1 else tuple(float(x) for x in np.linspace(0.0, 1.0, active_count))
            else:
                centers = tuple(float(x) for x in range(active_count))
        if len(centers) != active_count:
            raise ValueError(f"pair_filter_specs centers length must match active filter count for pair {pair}.")
        width = float(spec.get("filter_width", spec.get("width", default_width)))
        if width <= 0.0:
            raise ValueError("pair_filter_specs filter_width must be positive.")
        kind = str(spec.get("filter_kind", default_kind)).strip().lower()
        if kind not in {"radial_gaussian", "softmax_gaussian", "cosine_shell", "bernstein", "constant"}:
            raise ValueError(f"Unsupported pair_filter_specs filter_kind {kind!r}.")
        out[pair] = {
            "filter_kind": kind,
            "num_filters": int(active_count),
            "filter_centers": centers,
            "filter_width": width,
        }
    return out


def _rank_limit_allows_block(block, power, *, rank_nmax=None, rank_lmax=None, rank_lmin=None):
    rank = int(power)
    n_value = int(block["n"])
    l_value = int(block["l"])
    if rank_nmax and rank in rank_nmax and n_value > int(rank_nmax[rank]):
        return False
    if rank_lmax and rank in rank_lmax and l_value > int(rank_lmax[rank]):
        return False
    if rank_lmin and rank in rank_lmin and l_value < int(rank_lmin[rank]):
        return False
    return True


def ye3_power_sector_metadata(
    channels,
    *,
    max_power=4,
    slot_sectors=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
):
    """Return scalar Young-slot/E3 power sectors available for ``channels``.

    Each sector is built from a complete real-spherical ``(n,l,type)`` block
    and a symmetric-power scalar coupling ``Sym^p(V_l) -> V_0``.  Both the
    slot-trivial component and the slot-standard residual are included.
    """

    max_power = int(max_power)
    if max_power < 2:
        raise ValueError("ye3 power sectors require max_power >= 2.")
    slot_sectors = _normalize_ye3_slot_sectors(slot_sectors)
    rank_nmax = _normalize_rank_limit_mapping(rank_nmax, name="rank_nmax")
    rank_lmax = _normalize_rank_limit_mapping(rank_lmax, name="rank_lmax")
    rank_lmin = _normalize_rank_limit_mapping(rank_lmin, name="rank_lmin")
    sectors = []
    for block_index, block in enumerate(lifted_density_channel_blocks(channels)):
        l_value = int(block["l"])
        if bool(include_rank1) and l_value == 0 and _rank_limit_allows_block(
            block,
            1,
            rank_nmax=rank_nmax,
            rank_lmax=rank_lmax,
            rank_lmin=rank_lmin,
        ):
            sectors.append(
                {
                    "block_index": int(block_index),
                    "n": int(block["n"]),
                    "l": l_value,
                    "neighbor_type": block["neighbor_type"],
                    "power": 1,
                    "slot_sector": "trivial",
                    "multiplicity": 1,
                }
            )
        for power in range(2, max_power + 1):
            if not _rank_limit_allows_block(
                block,
                power,
                rank_nmax=rank_nmax,
                rank_lmax=rank_lmax,
                rank_lmin=rank_lmin,
            ):
                continue
            if 0 not in allowed_symmetric_power_outputs(power, l_value):
                continue
            multiplicity = symmetric_power_output_multiplicity(power, l_value, 0)
            if int(multiplicity) <= 0:
                continue
            for slot_sector in slot_sectors:
                sectors.append(
                    {
                        "block_index": int(block_index),
                        "n": int(block["n"]),
                        "l": l_value,
                        "neighbor_type": block["neighbor_type"],
                        "power": int(power),
                        "slot_sector": slot_sector,
                        "multiplicity": int(multiplicity),
                    }
                )
    return tuple(sectors)


def ye3_power_equivariant_sector_metadata(channels, *, max_power=4, target_L_R_values=None):
    """Return covariant Young-slot/E3 power sectors available for ``channels``.

    ``l_in`` is the angular momentum of one complete real-spherical channel
    block. ``target_L_R`` is the root/output angular momentum of the symmetric
    power contraction ``Sym^p(V_l_in) -> V_target_L_R``. This inventory is
    separate from scalar energy readouts: nonzero ``target_L_R`` sectors are
    equivariant feature blocks, not scalar site-energy contributions.
    """

    max_power = int(max_power)
    if max_power < 2:
        raise ValueError("ye3 power sectors require max_power >= 2.")
    requested = None if target_L_R_values is None else tuple(int(value) for value in target_L_R_values)
    if requested is not None and any(value < 0 for value in requested):
        raise ValueError("target_L_R_values must be nonnegative.")
    sectors = []
    for block_index, block in enumerate(lifted_density_channel_blocks(channels)):
        l_in = int(block["l"])
        for power in range(2, max_power + 1):
            allowed_outputs = tuple(int(value) for value in allowed_symmetric_power_outputs(power, l_in))
            outputs = allowed_outputs if requested is None else tuple(value for value in requested if value in allowed_outputs)
            for target_L_R in outputs:
                multiplicity = symmetric_power_output_multiplicity(power, l_in, target_L_R)
                if int(multiplicity) <= 0:
                    continue
                for slot_sector in ("trivial", "standard_summed"):
                    sectors.append(
                        {
                            "block_index": int(block_index),
                            "n": int(block["n"]),
                            "l_in": l_in,
                            "target_L_R": int(target_L_R),
                            "neighbor_type": block["neighbor_type"],
                            "power": int(power),
                            "slot_sector": slot_sector,
                            "multiplicity": int(multiplicity),
                        }
                    )
    return tuple(sectors)


def ye3_slot_specht_power_sector_metadata(
    channels,
    *,
    num_slots,
    max_power=4,
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
    slot_specht_coupling="projected_norm",
):
    """Return scalar A_s slot-Specht central-projector norm sectors.

    Each sector first forms an angular scalar from a rank-``power`` tensor
    product of one complete ``(n, l_in, type)`` channel block and then applies
    the exact central idempotent for a Specht partition of the diagonal slot
    group ``S_num_slots`` acting on the slot-value tensor-power basis.  The
    emitted scalar feature is ``||P_lambda T||^2``. This is a valid invariant
    using nontrivial slot character. It is a scalar central-projector norm,
    rather than a multiplicity-resolved Specht decomposition.
    """

    max_power = int(max_power)
    if max_power < 2:
        raise ValueError("ye3_slot_specht_power sectors require max_power >= 2.")
    num_slots = int(num_slots)
    partitions = _normalize_slot_specht_partitions(slot_specht_partitions, slot_count=num_slots)
    rank_nmax = _normalize_rank_limit_mapping(rank_nmax, name="rank_nmax")
    rank_lmax = _normalize_rank_limit_mapping(rank_lmax, name="rank_lmax")
    rank_lmin = _normalize_rank_limit_mapping(rank_lmin, name="rank_lmin")
    slot_specht_coupling = str(slot_specht_coupling).strip().lower()
    if slot_specht_coupling not in {"projected_norm", "commutant_symmetric"}:
        raise ValueError("slot_specht_coupling must be 'projected_norm' or 'commutant_symmetric'.")
    sectors = []
    for block_index, block in enumerate(lifted_density_channel_blocks(channels)):
        l_in = int(block["l"])
        powers = range(1, max_power + 1) if bool(include_rank1) else range(2, max_power + 1)
        for power in powers:
            if power == 1 and l_in != 0:
                continue
            if not _rank_limit_allows_block(
                block,
                power,
                rank_nmax=rank_nmax,
                rank_lmax=rank_lmax,
                rank_lmin=rank_lmin,
            ):
                continue
            angular_paths = _left_associated_angular_paths_to_target((l_in,) * power, 0)
            for angular_path in angular_paths:
                for partition in partitions:
                    trivial_plan = slot_specht_trivial_coupling_plan(
                        num_slots,
                        partition,
                        left_power=power,
                    )
                    projector_rank = int(trivial_plan["left_isotypic"]["projector_rank"])
                    if int(projector_rank) <= 0:
                        continue
                    if slot_specht_coupling == "commutant_symmetric":
                        feature_multiplicity = int(trivial_plan["full_symmetric_quadratic_count"])
                        runtime = "central_specht_projector_commutant_symmetric_quadratics"
                        coupling_scope = "multiplicity_resolved_symmetric_commutant_scalar_forms"
                    else:
                        feature_multiplicity = 1
                        runtime = "central_specht_projector_norm"
                        coupling_scope = trivial_plan["scope"]
                    if feature_multiplicity <= 0:
                        continue
                    sectors.append(
                        {
                            "block_index": int(block_index),
                            "n": int(block["n"]),
                            "l_in": l_in,
                            "target_L_R": 0,
                            "neighbor_type": block["neighbor_type"],
                            "power": int(power),
                            "slot_group": f"S_{num_slots}",
                            "slot_specht_partition": tuple(int(part) for part in partition),
                            "slot_projector_rank": int(projector_rank),
                            "slot_projector_idempotent": True,
                            "slot_specht_dimension": int(trivial_plan["left_isotypic"]["specht_dimension"]),
                            "slot_isotypic_multiplicity": int(
                                trivial_plan["left_isotypic"]["isotypic_multiplicity"]
                            ),
                            "angular_path": tuple(int(value) for value in angular_path),
                            "multiplicity": int(feature_multiplicity),
                            "full_trivial_matrix_unit_count": int(trivial_plan["full_trivial_matrix_unit_count"]),
                            "full_symmetric_quadratic_count": int(
                                trivial_plan["full_symmetric_quadratic_count"]
                            ),
                            "implemented_runtime_scalar_count": int(
                                feature_multiplicity
                            ),
                            "runtime": runtime,
                            "source_permutation_representation": "trivial"
                            if tuple(int(part) for part in partition) == (num_slots,)
                            else "nontrivial_specht",
                            "target_permutation_representation": "trivial",
                            "final_permutation_representation": "trivial",
                            "uses_nontrivial_source": tuple(int(part) for part in partition) != (num_slots,),
                            "trivial_target_coupling": "carrier_dual_projected_norm",
                            "slot_specht_coupling": slot_specht_coupling,
                            "coupling_scope": coupling_scope,
                            "requires_full_matrix_units_for_multiplicity_resolution": bool(
                                slot_specht_coupling != "commutant_symmetric"
                                and trivial_plan["requires_full_matrix_units_for_multiplicity_resolution"]
                            ),
                        }
                    )
    return tuple(sectors)


def ye3_rank2_orbit_equivariant_sector_metadata(channels, *, target_L_R_values=None, include_ordered_pairs=False):
    """Return rank-2 covariant sectors for arbitrary complete channel-block pairs.

    This inventory is the non-degenerate counterpart to the symmetric-power
    fast path.  Each sector couples two complete input angular blocks
    ``V_{l_in[0]} x V_{l_in[1]} -> V_{target_L_R}``.  The metadata records the
    orbit symmetry of the factor pair:

    - ``pair_orbits=(2,)`` for a repeated block pair;
    - ``pair_orbits=(1, 1)`` for a non-degenerate two-block pair.

    This is rank-2 only.  General N-rank orbit products still need a recursive
    product/path evaluator carrying explicit Young/permutation labels.
    """

    blocks = lifted_density_channel_blocks(channels)
    requested = None if target_L_R_values is None else tuple(int(value) for value in target_L_R_values)
    if requested is not None and any(value < 0 for value in requested):
        raise ValueError("target_L_R_values must be nonnegative.")
    sectors = []
    for left_index, left_block in enumerate(blocks):
        right_start = 0 if include_ordered_pairs else left_index
        for right_index in range(right_start, len(blocks)):
            right_block = blocks[int(right_index)]
            l_left = int(left_block["l"])
            l_right = int(right_block["l"])
            allowed = tuple(range(abs(l_left - l_right), l_left + l_right + 1))
            outputs = allowed if requested is None else tuple(value for value in requested if value in allowed)
            repeated = (
                int(left_block["n"]) == int(right_block["n"])
                and int(left_block["l"]) == int(right_block["l"])
                and left_block["neighbor_type"] == right_block["neighbor_type"]
            )
            for target_L_R in outputs:
                for slot_sector in ("trivial", "standard_summed"):
                    sectors.append(
                        {
                            "rank": 2,
                            "left_block_index": int(left_index),
                            "right_block_index": int(right_index),
                            "n_in": (int(left_block["n"]), int(right_block["n"])),
                            "l_in": (l_left, l_right),
                            "target_L_R": int(target_L_R),
                            "neighbor_types": (left_block["neighbor_type"], right_block["neighbor_type"]),
                            "pair_orbits": (2,) if repeated else (1, 1),
                            "orbit_symmetry": "repeated_pair" if repeated else "nondegenerate_pair",
                            "slot_sector": slot_sector,
                            "multiplicity": 1,
                        }
                    )
    return tuple(sectors)


def _partition_from_values(values):
    counts = {}
    for value in values:
        key = tuple(value) if isinstance(value, (tuple, list)) else value
        counts[key] = counts.get(key, 0) + 1
    return tuple(sorted((int(count) for count in counts.values()), reverse=True))


def _angular_resultants(left_L, right_L):
    return tuple(range(abs(int(left_L) - int(right_L)), int(left_L) + int(right_L) + 1))


def _coupling_paths_for_l_tuple(l_in, target_L_R, *, max_intermediate_L_R=None):
    """Adapter for :func:`ye3t.couplings.coupling_paths_for_l_tuple`."""

    return tuple(
        _ye3t_coupling_paths_for_l_tuple(
            l_in,
            target_L_R,
            max_intermediate_L=max_intermediate_L_R,
        )
    )


def _distinct_permutations(values):
    return tuple(sorted(set(permutations(tuple(values)))))


def ye3_orbit_product_equivariant_sector_metadata(
    channels,
    *,
    block_tuples,
    target_L_R_values,
    max_intermediate_L_R=None,
    symmetrize_factor_orbits=True,
):
    """Return general N-rank orbit-product sectors for complete channel blocks.

    This is the general product-path counterpart to the symmetric-power fast
    path.  It supports arbitrary repeated, partially repeated, and
    non-degenerate block tuples.  With ``symmetrize_factor_orbits=True`` it
    averages over distinct permutations of each block tuple, which realizes the
    factor-orbit trivial projection for that tuple.  Nontrivial Specht/Young
    projectors are not part of this helper yet.
    """

    blocks = lifted_density_channel_blocks(channels)
    normalized_tuples = tuple(tuple(int(index) for index in item) for item in block_tuples)
    targets = tuple(int(value) for value in target_L_R_values)
    if not normalized_tuples:
        raise ValueError("block_tuples must contain at least one block-index tuple.")
    if not targets:
        raise ValueError("target_L_R_values must contain at least one angular momentum.")
    sectors = []
    for tuple_index, block_tuple in enumerate(normalized_tuples):
        if not block_tuple:
            raise ValueError("Each block tuple must be nonempty.")
        if any(index < 0 or index >= len(blocks) for index in block_tuple):
            raise ValueError("block_tuples contain an index outside the complete channel-block inventory.")
        l_in = tuple(int(blocks[index]["l"]) for index in block_tuple)
        n_in = tuple(int(blocks[index]["n"]) for index in block_tuple)
        neighbor_types = tuple(blocks[index]["neighbor_type"] for index in block_tuple)
        pair_labels = tuple((n, l, neighbor_type) for n, l, neighbor_type in zip(n_in, l_in, neighbor_types))
        pair_orbits = _partition_from_values(pair_labels)
        factor_permutations = _distinct_permutations(block_tuple) if symmetrize_factor_orbits else (block_tuple,)
        for target_L_R in targets:
            for path_index, intermediate_L_R in enumerate(
                _coupling_paths_for_l_tuple(
                    l_in,
                    int(target_L_R),
                    max_intermediate_L_R=max_intermediate_L_R,
                )
            ):
                for slot_sector in ("trivial", "standard_summed"):
                    sectors.append(
                        {
                            "rank": int(len(block_tuple)),
                            "tuple_index": int(tuple_index),
                            "block_tuple": tuple(int(index) for index in block_tuple),
                            "n_in": n_in,
                            "l_in": l_in,
                            "target_L_R": int(target_L_R),
                            "intermediate_L_R": tuple(int(value) for value in intermediate_L_R),
                            "neighbor_types": neighbor_types,
                            "pair_orbits": pair_orbits,
                            "factor_permutation_count": int(len(factor_permutations)),
                            "factor_orbit_projection": "trivial_average" if symmetrize_factor_orbits else "ordered_path",
                            "slot_sector": slot_sector,
                            "multiplicity_path": int(path_index),
                        }
                    )
    return tuple(sectors)


@recordclass(('name', 'rank', 'slot_sector', 'angular_l', 'output_L', 'multiplicity', 'runtime_status', 'representation_level', 'channel_block', 'runtime_status_detail', 'notes'), frozen = True)
class ASDescriptorInventoryRecord:
    """Status record for an implemented or planned scalar ``A_s`` sector.

    The record is an inventory/planning object.  It does not claim a full
    Young/Specht decomposition unless ``representation_level`` says so.
    """
    channel_block = field(default_factory=dict)
    runtime_status_detail = ""
    notes = ()

    def as_dict(self):
        return {
            "name": self.name,
            "rank": int(self.rank),
            "slot_sector": self.slot_sector,
            "angular_l": self.angular_l,
            "output_L": self.output_L,
            "multiplicity": int(self.multiplicity),
            "runtime_status": self.runtime_status,
            "runtime_status_detail": self.runtime_status_detail,
            "representation_level": self.representation_level,
            "channel_block": dict(self.channel_block),
            "notes": list(self.notes),
        }


def _lifted_density_config_from_inventory_input(config=None, **kwargs):
    if isinstance(config, LiftedDensityConfig):
        if kwargs:
            payload = config.to_dict()
            payload.update(kwargs)
            return LiftedDensityConfig.from_dict(payload)
        return config
    payload = {} if config is None else dict(config)
    payload.update(kwargs)
    return LiftedDensityConfig.from_dict(payload)


def lifted_A_s_descriptor_inventory(config=None, **kwargs):
    """Return conservative scalar-sector inventory records for ``filtered_A_s``.

    This inventory describes currently implemented scalar readout sectors and
    records the boundary to the planned full representation-theoretic A_s
    descriptor backend.  It is not a full arbitrary Specht-sector descriptor
    materialization.
    """

    lifted = _lifted_density_config_from_inventory_input(config, **kwargs)
    mode = str(lifted.readout_mode)
    records = []
    representation_level = "permutation_module_scalar_readout"
    if mode in {"linear", "symmetric_linear"}:
        for index, channel in enumerate(lifted.channels):
            l_value = int(channel.l)
            status_detail = "implemented_scalar" if l_value == 0 else "not_rotational_scalar_without_L0_contraction"
            status = "legacy_scalar_readout" if l_value == 0 else "planned_not_public"
            notes = ()
            if l_value != 0:
                notes = ("linear channel readout is only a rotational scalar for l=0 channels",)
            records.append(
                ASDescriptorInventoryRecord(
                    name=f"{mode}_channel_{index}",
                    rank=1,
                    slot_sector="trivial" if mode == "symmetric_linear" else "slot_resolved",
                    angular_l=l_value,
                    output_L=0 if l_value == 0 else None,
                    multiplicity=1,
                    runtime_status=status,
                    runtime_status_detail=status_detail,
                    representation_level=representation_level,
                    channel_block=channel.to_dict(),
                    notes=notes,
                )
            )
    elif mode == "character_quadratic":
        for index, channel in enumerate(lifted.channels):
            l_value = int(channel.l)
            status_detail = "implemented_scalar" if l_value == 0 else "not_rotational_scalar_without_L0_contraction"
            status = "legacy_scalar_readout" if l_value == 0 else "planned_not_public"
            notes = ()
            if l_value != 0:
                notes = ("character_quadratic currently contracts slot sectors, not angular l>0 blocks",)
            records.append(
                ASDescriptorInventoryRecord(
                    name=f"character_mu_channel_{index}",
                    rank=1,
                    slot_sector="trivial",
                    angular_l=l_value,
                    output_L=0 if l_value == 0 else None,
                    multiplicity=1,
                    runtime_status=status,
                    runtime_status_detail=status_detail,
                    representation_level=representation_level,
                    channel_block=channel.to_dict(),
                    notes=notes,
                )
            )
            records.append(
                ASDescriptorInventoryRecord(
                    name=f"character_standard_norm_channel_{index}",
                    rank=2,
                    slot_sector="standard_norm",
                    angular_l=l_value,
                    output_L=0 if l_value == 0 else None,
                    multiplicity=1,
                    runtime_status=status,
                    runtime_status_detail=status_detail,
                    representation_level=representation_level,
                    channel_block=channel.to_dict(),
                    notes=notes,
                )
            )
    elif mode == "ye3_quadratic":
        for block_index, block in enumerate(lifted_density_channel_blocks(lifted.channels)):
            for slot_sector in ("trivial_norm", "standard_norm"):
                records.append(
                    ASDescriptorInventoryRecord(
                        name=f"ye3_quadratic_block_{block_index}_{slot_sector}",
                        rank=2,
                        slot_sector=slot_sector,
                        angular_l=int(block["l"]),
                        output_L=0,
                        multiplicity=1,
                        runtime_status="legacy_scalar_readout",
                        runtime_status_detail="implemented_scalar",
                        representation_level="permutation_module_scalar_readout_with_SO3_complete_m_block",
                        channel_block=dict(block),
                    )
                )
    elif mode == "ye3_power":
        for sector_index, sector in enumerate(
            ye3_power_sector_metadata(lifted.channels, max_power=lifted.ye3_max_power)
        ):
            records.append(
                ASDescriptorInventoryRecord(
                    name=f"ye3_power_sector_{sector_index}",
                    rank=int(sector["power"]),
                    slot_sector=str(sector["slot_sector"]),
                    angular_l=int(sector["l"]),
                    output_L=0,
                    multiplicity=int(sector["multiplicity"]),
                    runtime_status="legacy_scalar_readout",
                    runtime_status_detail="implemented_scalar",
                    representation_level="permutation_module_scalar_readout_with_symmetric_power_SO3_contraction",
                    channel_block=dict(sector),
                )
            )
    elif mode == "antisymmetric_quadratic":
        records.append(
            ASDescriptorInventoryRecord(
                name="antisymmetric_squared_wedge_volume",
                rank=2,
                slot_sector="antisymmetric_magnitude",
                angular_l=None,
                output_L=0,
                multiplicity=1,
                runtime_status="legacy_scalar_readout",
                runtime_status_detail="implemented_scalar_magnitude_not_sign_carrier",
                representation_level="permutation_module_scalar_readout",
                notes=("squared volume is an invariant magnitude, not a full antisymmetric sign-carrier runtime",),
            )
        )
    records.append(
        ASDescriptorInventoryRecord(
            name="full_A_s_young_specht_N_rank_runtime",
            rank=int(lifted.ye3_max_power),
            slot_sector="arbitrary_young_specht",
            angular_l=None,
            output_L=None,
            multiplicity=0,
            runtime_status="planned_not_public",
            runtime_status_detail="planned_not_implemented",
            representation_level="full_irrep_multiplicity_decomposition",
            notes=(
                "placeholder for general N-rank A_s descriptors with Young/Specht labels, angular labels, and multiplicities",
            ),
        )
    )
    return tuple(records)


def _young_subgroup_generators(slot_count, blocks=None):
    slot_count = int(slot_count)
    if slot_count <= 0:
        raise ValueError("slot_count must be positive.")
    identity = tuple(range(slot_count))
    if blocks is None:
        blocks = (tuple(range(slot_count)),)
    normalized_blocks = _normalize_young_subgroup_blocks(blocks, slot_count)
    if not normalized_blocks:
        raise ValueError("Young subgroup blocks must be nonempty.")
    if all(len(block) == 1 for block in normalized_blocks):
        return (identity,)
    generators = [identity]
    for block in normalized_blocks:
        for index in range(len(block) - 1):
            left = int(block[index])
            right = int(block[index + 1])
            perm = list(identity)
            perm[left], perm[right] = perm[right], perm[left]
            generators.append(tuple(perm))
    return tuple(generators)


def _normalize_young_subgroup_blocks(blocks, slot_count):
    slot_count = int(slot_count)
    if blocks is None:
        return ()
    normalized = tuple(tuple(int(slot) for slot in block) for block in blocks)
    if not normalized:
        return ()
    scope = _ye3t_validate_slot_permutation_scope(
        slot_count=slot_count,
        permuted_slot_count=slot_count,
        blocks=normalized,
    )
    return tuple(tuple(int(slot) for slot in block) for block in scope["blocks"])


def _integer_partitions(n, max_part=None):
    return tuple(_ye3t_integer_partitions(n, max_part))


def _young_subgroup_blocks_from_lifted_config(lifted):
    explicit = getattr(lifted, "young_subgroup_blocks", ())
    if explicit:
        return explicit
    if lifted.slot_group == "identity":
        return tuple((slot,) for slot in range(int(lifted.num_filters)))
    if lifted.slot_group in {"symmetric", "young_subgroup"}:
        return (tuple(range(int(lifted.num_filters))),)
    raise ValueError("slot_group must be 'symmetric', 'identity', or 'young_subgroup'.")


def build_A_s_young_subgroup_intertwiner_basis(config=None, *, blocks=None, **kwargs):
    """Build exact Young-subgroup intertwiners for filtered ``A_s`` slots.

    This returns the orbital basis for maps from slot densities to slot
    densities under a product of slot-symmetric groups.  It is an exact
    permutation-module Hom-space basis for this Young subgroup; it is not a
    full Young/Specht decomposition and it does not include SO(3)
    Clebsch--Gordan coupling.
    """

    lifted = _lifted_density_config_from_inventory_input(config, **kwargs)
    from ye3t.couplings import compile_A_s_young_subgroup_slot_intertwiners
    from ye3t_ace.permutation_orbitals import permutation_orbital_basis_from_compiled_report

    subgroup_blocks = _young_subgroup_blocks_from_lifted_config(lifted) if blocks is None else blocks
    report = compile_A_s_young_subgroup_slot_intertwiners(
        slot_count=int(lifted.num_filters),
        blocks=subgroup_blocks,
        normalization="frobenius",
    )
    return permutation_orbital_basis_from_compiled_report(report)


def build_A_s_slot_permutation_intertwiner_basis(config=None, **kwargs):
    return build_A_s_young_subgroup_intertwiner_basis(config, **kwargs)


def _permutation_matrix_from_tuple(perm, *, dtype=torch.float64, device=None):
    perm = tuple(int(value) for value in perm)
    matrix = torch.zeros((len(perm), len(perm)), dtype=dtype, device=device)
    for output_index, input_index in enumerate(perm):
        matrix[int(output_index), int(input_index)] = 1.0
    return matrix


def _embedded_block_permutation(slot_count, block, block_perm):
    perm = list(range(int(slot_count)))
    selected = [int(slot) for slot in block]
    for output_offset, input_offset in enumerate(block_perm):
        perm[selected[output_offset]] = selected[int(input_offset)]
    return tuple(perm)


def build_A_s_specht_slot_projectors(config=None, *, max_slot_count=8, dtype=torch.float64, device=None, **kwargs):
    """Build central Specht projectors for the A_s slot hidden-state axis.

    For ``slot_group='symmetric'`` this decomposes the natural slot
    permutation representation of ``S_s`` into its Specht isotypic components.
    For ``slot_group='young_subgroup'`` it returns the corresponding block-wise
    projectors for each Young-subgroup factor.  A single A_s slot vector only
    carries the natural permutation representation; higher Specht sectors
    require tensor/product powers of slot states and are handled by the n-ary
    Young-E3 product runtime.

    Projector validity and finite-group character provenance come from
    ``ye3t.couplings.compile_A_s_slot_specht_projectors``.  This function only
    converts the compiled report into Torch tensors for A_s materialization.
    """

    lifted = _lifted_density_config_from_inventory_input(config, **kwargs)
    slot_count = int(lifted.num_filters)
    from ye3t.couplings import compile_A_s_slot_specht_projectors

    blocks = _young_subgroup_blocks_from_lifted_config(lifted)
    report = compile_A_s_slot_specht_projectors(
        slot_count=slot_count,
        blocks=blocks,
        max_slot_count=int(max_slot_count),
    )
    report_dict = report.to_dict()
    records = []
    for record in report.records:
        projector = torch.tensor(record["projector"], dtype=dtype, device=device)
        records.append(
            {
                "block_index": int(record["block_index"]),
                "block_slots": tuple(int(slot) for slot in record["block_slots"]),
                "partition": tuple(int(part) for part in record["partition"]),
                "projector": projector,
                "rank": int(record["rank"]),
                "carrier_dim": int(record["carrier_dim"]),
                "scope": str(record["scope"]),
                "coordinate_dim": int(record["coordinate_dim"]),
                "coordinate_basis": tuple(tuple(float(value) for value in row) for row in record["coordinate_basis"]),
                "coordinate_orthonormality_error": float(record["coordinate_orthonormality_error"]),
                "projector_reconstruction_error": float(record["projector_reconstruction_error"]),
                "generator_actions": tuple(record["generator_actions"]),
                "max_generator_orthogonality_error": float(record["max_generator_orthogonality_error"]),
                "max_generator_equivariance_error": float(record["max_generator_equivariance_error"]),
                "compiled_projector_backend": str(report.backend),
                "compiled_projector_convention_hash": str(report.convention_hash),
                "compiled_projector_validation": dict(report.validation_report),
                "compiled_projector_provenance": dict(report.provenance),
                "compiled_projector_report": report_dict,
            }
        )
    return tuple(records)


class ASSpechtSlotProjector(torch.nn.Module):
    """Project A_s slot hidden states into Specht-isotypic slot blocks.

    Input and projected states have shape ``[n_atoms, n_slots, channels,
    2*L_R+1]``.  The projection acts only on the slot axis and therefore
    commutes with scalar radial edge kernels and SO(3) actions on the magnetic
    axis.  For full ``S_s`` natural slot states, only the trivial and standard
    sectors have nonzero rank.
    """

    def __init__(self, config=None, *, max_slot_count=8, dtype=torch.float64, device=None, **kwargs):
        super().__init__()
        lifted = _lifted_density_config_from_inventory_input(config, **kwargs)
        records = build_A_s_specht_slot_projectors(
            lifted,
            max_slot_count=max_slot_count,
            dtype=dtype,
            device=device,
        )
        if not records:
            raise ValueError("No nonzero A_s Specht slot projectors were constructed.")
        self.config = lifted
        self.compiled_projector_report = records[0].get("compiled_projector_report", None)
        self.records = tuple(
            {
                key: value
                for key, value in record.items()
                if key not in {"projector", "compiled_projector_report"}
            }
            for record in records
        )
        self.register_buffer("projectors", torch.stack([record["projector"] for record in records], dim=0))

    def forward(self, state):
        state = torch.as_tensor(state, dtype=self.projectors.dtype, device=self.projectors.device)
        if state.ndim != 4:
            raise ValueError("A_s Specht slot state must have shape [n_atoms, n_slots, channels, 2*L_R+1].")
        if int(state.shape[1]) != int(self.projectors.shape[-1]):
            raise ValueError("State slot dimension does not match the Specht projector slot count.")
        return {
            self._record_key(index): torch.einsum("oi,nicm->nocm", projector, state)
            for index, projector in enumerate(self.projectors)
        }

    def matrix_unit_norm_features(self, state):
        state = torch.as_tensor(state, dtype=self.projectors.dtype, device=self.projectors.device)
        if state.ndim != 4:
            raise ValueError("A_s Specht slot state must have shape [n_atoms, n_slots, channels, 2*L_R+1].")
        if int(state.shape[1]) != int(self.projectors.shape[-1]):
            raise ValueError("State slot dimension does not match the Specht projector slot count.")
        n_atoms = int(state.shape[0])
        channel_count = int(state.shape[2])
        magnetic_count = int(state.shape[3])
        feature_rows = []
        carriers = {}
        for index, record in enumerate(self.records):
            block_slots = tuple(int(slot) for slot in record["block_slots"])
            block_index = torch.tensor(block_slots, dtype=torch.long, device=state.device)
            block_state = state.index_select(1, block_index)
            flat = block_state.permute(0, 2, 3, 1).reshape(
                n_atoms * channel_count * magnetic_count,
                len(block_slots),
            )
            partition = tuple(int(part) for part in record["partition"])
            resolved = slot_specht_matrix_unit_carriers(
                flat,
                slot_count=len(block_slots),
                power=1,
                partition=partition,
            )
            specht_dim = int(resolved.shape[1])
            resolved = resolved.reshape(
                n_atoms,
                channel_count,
                magnetic_count,
                specht_dim,
                specht_dim,
                len(block_slots),
            ).permute(0, 3, 4, 5, 1, 2)
            carriers[self._record_key(index)] = resolved
            feature_rows.append((resolved ** 2).sum(dim=(1, 2, 3, 5)))
        return feature_rows, carriers

    def validate_matrix_units(self):
        records = []
        max_matrix_unit_algebra_residual = 0.0
        max_projector_difference = 0.0
        max_projector_idempotency_error = 0.0
        passed = True
        for record in self.records:
            block_slots = tuple(int(slot) for slot in record["block_slots"])
            partition = tuple(int(part) for part in record["partition"])
            report = validate_slot_specht_matrix_units(
                len(block_slots),
                1,
                partition,
            )
            validation = dict(report)
            compiled_validation = dict(report.get("compiled_matrix_unit_validation", {}))
            max_matrix_unit_algebra_residual = max(
                max_matrix_unit_algebra_residual,
                float(validation.get("matrix_unit_algebra_max_residual", 0.0)),
            )
            max_projector_difference = max(
                max_projector_difference,
                float(compiled_validation.get("max_projector_difference_from_diagonal_sum", 0.0)),
            )
            max_projector_idempotency_error = max(
                max_projector_idempotency_error,
                float(compiled_validation.get("max_projector_idempotency_error", 0.0)),
            )
            passed = bool(passed and validation.get("passed", False))
            records.append(
                {
                    "block_index": int(record["block_index"]),
                    "block_slots": block_slots,
                    "partition": partition,
                    "slot_count": int(len(block_slots)),
                    "power": 1,
                    "coefficient_source": "ye3t.couplings.compile_A_s_slot_specht_matrix_units",
                    "compiled_matrix_unit_report": report,
                    "validation_report": validation,
                }
            )
        return {
            "passed": bool(passed),
            "slot_count": int(self.projectors.shape[-1]),
            "sector_count": int(len(self.records)),
            "records": records,
            "coefficient_source": "ye3t.couplings.compile_A_s_slot_specht_matrix_units",
            "matrix_unit_resolved": True,
            "central_global_coupler_consumed": False,
            "runtime_scope": "natural_slot_matrix_unit_readout_not_global_YE3T_coupler",
            "max_matrix_unit_algebra_residual": float(max_matrix_unit_algebra_residual),
            "max_projector_difference_from_diagonal_sum": float(max_projector_difference),
            "max_projector_idempotency_error": float(max_projector_idempotency_error),
        }

    def _record_key(self, index):
        record = self.records[int(index)]
        return (
            int(record["block_index"]),
            tuple(int(part) for part in record["partition"]),
        )

    def validate_projectors(self, *, atol=1.0e-10, rtol=1.0e-10):
        projectors = self.projectors
        identity = torch.eye(int(projectors.shape[-1]), dtype=projectors.dtype, device=projectors.device)
        max_idempotency_error = 0.0
        max_orthogonality_error = 0.0
        max_completeness_error = 0.0
        for idx, projector in enumerate(projectors):
            error = torch.max(torch.abs(projector @ projector - projector))
            max_idempotency_error = max(max_idempotency_error, float(error.detach().cpu()))
            for jdx in range(idx + 1, int(projectors.shape[0])):
                left = self.records[idx]
                right = self.records[jdx]
                if int(left["block_index"]) != int(right["block_index"]):
                    continue
                error = torch.max(torch.abs(projector @ projectors[jdx]))
                max_orthogonality_error = max(max_orthogonality_error, float(error.detach().cpu()))
        for block_index in sorted({int(record["block_index"]) for record in self.records}):
            block_records = [
                (idx, record)
                for idx, record in enumerate(self.records)
                if int(record["block_index"]) == int(block_index)
            ]
            block_slots = tuple(int(slot) for slot in block_records[0][1]["block_slots"])
            block_identity = torch.zeros_like(identity)
            for slot in block_slots:
                block_identity[int(slot), int(slot)] = 1.0
            total = sum((projectors[idx] for idx, _record in block_records), torch.zeros_like(identity))
            error = torch.max(torch.abs(total - block_identity))
            max_completeness_error = max(max_completeness_error, float(error.detach().cpu()))
        passed = (
            max_idempotency_error <= float(atol) + float(rtol)
            and max_orthogonality_error <= float(atol) + float(rtol)
            and max_completeness_error <= float(atol) + float(rtol)
        )
        return {
            "passed": bool(passed),
            "slot_count": int(projectors.shape[-1]),
            "sector_count": int(projectors.shape[0]),
            "records": [dict(record) for record in self.records],
            "max_idempotency_error": max_idempotency_error,
            "max_orthogonality_error": max_orthogonality_error,
            "max_completeness_error": max_completeness_error,
            "scope": "natural A_s slot permutation representation",
            "compiled_projector_report": self.compiled_projector_report,
            "coefficient_source": "ye3t.couplings.compile_A_s_slot_specht_projectors",
            "coordinate_resolved": bool(
                self.compiled_projector_report.get("validation_report", {}).get("coordinate_resolved", False)
            )
            if self.compiled_projector_report is not None
            else False,
            "max_coordinate_orthonormality_error": float(
                self.compiled_projector_report.get("validation_report", {}).get(
                    "max_coordinate_orthonormality_error",
                    0.0,
                )
            )
            if self.compiled_projector_report is not None
            else 0.0,
            "max_projector_reconstruction_error": float(
                self.compiled_projector_report.get("validation_report", {}).get(
                    "max_projector_reconstruction_error",
                    0.0,
                )
            )
            if self.compiled_projector_report is not None
            else 0.0,
            "max_generator_orthogonality_error": float(
                self.compiled_projector_report.get("validation_report", {}).get(
                    "max_generator_orthogonality_error",
                    0.0,
                )
            )
            if self.compiled_projector_report is not None
            else 0.0,
            "max_generator_equivariance_error": float(
                self.compiled_projector_report.get("validation_report", {}).get(
                    "max_generator_equivariance_error",
                    0.0,
                )
            )
            if self.compiled_projector_report is not None
            else 0.0,
        }


def _flatten_scalar_power_output(value):
    if value.ndim == 1:
        return value.unsqueeze(1)
    if value.ndim == 2:
        return value
    raise ValueError("Scalar symmetric-power output must have shape [n_atoms] or [n_atoms, multiplicity].")


def _as_equivariant_power_output(value, atom_count, target_L_R):
    """Return symmetric-power output as ``[n_atoms, multiplicity, 2*L_R+1]``."""

    target_L_R = int(target_L_R)
    out_dim = 2 * target_L_R + 1
    if target_L_R == 0:
        flat = _flatten_scalar_power_output(value)
        return flat.reshape(int(atom_count), -1, 1)
    if value.ndim == 2:
        return value.reshape(int(atom_count), 1, out_dim)
    if value.ndim == 3:
        return value.reshape(int(atom_count), -1, out_dim)
    raise ValueError("Equivariant symmetric-power output must carry a final magnetic/component axis.")


class _LiftedDensityEdgeValuesWithAnalyticDx(torch.autograd.Function):
    """Return edge values while using SiteBasisV2 analytic ``dvalue/dx_ij``.

    The lifted A_s density path consumes per-edge site-basis values before
    aggregation over filtered slots.  ``SiteBasisV2.compute_channel_edges_with_dx``
    returns both values and analytic derivatives with respect to the Cartesian
    edge displacement.  For real-spherical ``l>0`` rows, relying on ordinary
    autograd through the value construction can give the wrong derivative for
    some real-tesseral components.  This wrapper makes the derivative convention
    explicit: if ``v_ea`` is the returned real edge value, backward applies

        dL/dx_e = sum_a (dL/dv_ea) (dv_ea/dx_e).

    The wrapper is intentionally local to lifted-density edge values; exact ACE
    descriptor force paths continue to use their existing analytic/streaming
    force backends.
    """

    @staticmethod
    def forward(ctx, disp, values, edge_dx):
        values = torch.as_tensor(values, dtype=disp.dtype, device=disp.device)
        edge_dx = torch.as_tensor(edge_dx, dtype=disp.dtype, device=disp.device)
        ctx.save_for_backward(edge_dx)
        return values.detach()

    @staticmethod
    def backward(ctx, grad_out):
        (edge_dx,) = ctx.saved_tensors
        grad_disp = None
        if ctx.needs_input_grad[0]:
            grad = grad_out.to(edge_dx.dtype)
            if grad.ndim == 2:
                grad_disp = torch.sum(grad.unsqueeze(-1) * edge_dx, dim=1)
            else:
                grad_disp = torch.sum(grad.unsqueeze(-1) * edge_dx.reshape((1,) * (grad.ndim - 2) + tuple(edge_dx.shape)), dim=-2)
        return grad_disp, None, None


def _edge_values_with_analytic_dx(disp, values, edge_dx):
    if not torch.is_grad_enabled() or not bool(getattr(disp, "requires_grad", False)):
        return values
    return _LiftedDensityEdgeValuesWithAnalyticDx.apply(disp, values, edge_dx)


def ye3_power_invariants(
    density,
    channels,
    *,
    max_power=4,
    optimization_policy="auto",
    slot_sectors=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
):
    """Return higher-order scalar Young-slot/E3 power invariants.

    For each complete ``(n,l,type)`` block and each allowed power ``p`` up to
    ``max_power``, this evaluates scalar real-tesseral couplings
    ``Sym^p(V_l) -> V_0``.  The trivial slot component is evaluated once per
    atom; the standard slot component is evaluated per slot and then summed
    over slots to obtain a slot-invariant scalar.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    blocks = lifted_density_channel_blocks(channels)
    sectors = ye3_power_sector_metadata(
        channels,
        max_power=max_power,
        slot_sectors=slot_sectors,
        include_rank1=include_rank1,
        rank_nmax=rank_nmax,
        rank_lmax=rank_lmax,
        rank_lmin=rank_lmin,
    )
    if not sectors:
        return density.new_zeros((int(density.shape[0]), 0)), blocks, sectors
    mu = slot_trivial_component(density)
    eta = slot_standard_residual(density)
    features = []
    for sector in sectors:
        block = blocks[int(sector["block_index"])]
        indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
        power = int(sector["power"])
        l_value = int(sector["l"])
        if power == 1:
            if sector["slot_sector"] != "trivial" or l_value != 0:
                raise ValueError("Rank-1 ye3_power scalar sectors must be slot-trivial l=0 sectors.")
            features.append(mu.index_select(1, indices))
        elif sector["slot_sector"] == "trivial":
            block_values = _site_real_block_to_ye3t_tesseral(mu.index_select(1, indices), l_value)
            value = symmetric_power_real_tesseral(
                block_values,
                power,
                l_value,
                0,
                optimization_policy=optimization_policy,
            )
            features.append(_flatten_scalar_power_output(value))
        else:
            eta_block = _site_real_block_to_ye3t_tesseral(eta.index_select(2, indices), l_value)
            flat_eta = eta_block.reshape(-1, 2 * l_value + 1)
            value = symmetric_power_real_tesseral(
                flat_eta,
                power,
                l_value,
                0,
                optimization_policy=optimization_policy,
            )
            value = _flatten_scalar_power_output(value).reshape(int(density.shape[0]), int(density.shape[1]), -1)
            features.append(value.sum(dim=1))
    return torch.cat(features, dim=1), blocks, sectors


def _slot_tuple_angular_scalar_values(block_values, *, l_in, power, angular_path, optimization_policy):
    """Evaluate one left-associated angular scalar for all slot tuples."""

    l_in = int(l_in)
    power = int(power)
    slot_count = int(block_values.shape[1])
    basis = _slot_tuple_basis(slot_count, power)
    if power == 1:
        if l_in != 0:
            raise ValueError("Rank-1 scalar slot-Specht sectors require l_in=0.")
        slot_indices = torch.tensor([state[0] for state in basis], dtype=torch.long, device=block_values.device)
        return block_values.index_select(1, slot_indices).squeeze(-1)
    pieces = []
    for factor_index in range(power):
        slot_indices = torch.tensor(
            [state[factor_index] for state in basis],
            dtype=torch.long,
            device=block_values.device,
        )
        pieces.append(block_values.index_select(1, slot_indices))
    current = pieces[0]
    current_L = l_in
    angular_path = tuple(int(value) for value in angular_path)
    for factor_index, piece in enumerate(pieces[1:], start=1):
        next_L = 0 if factor_index == power - 1 else int(angular_path[factor_index - 1])
        left_arg = current if int(current_L) != 0 or current.ndim == 3 else current.unsqueeze(-1)
        right_arg = piece if int(l_in) != 0 or piece.ndim == 3 else piece.unsqueeze(-1)
        current, _backend = couple_packed_real_tesseral(
            left_arg,
            right_arg,
            current_L,
            l_in,
            next_L,
            backend="pytorch",
        )
        current_L = next_L
    if current.ndim == 3:
        if int(current.shape[-1]) != 1:
            raise ValueError("Expected final scalar angular contraction with target_L_R=0.")
        current = current.squeeze(-1)
    return current


def _slot_tuple_angular_values(block_values, *, l_in, power, angular_path, target_L_R, optimization_policy):
    """Evaluate one left-associated angular path for all slot tuples.

    The returned tensor has shape ``[n_atoms, slot_count**power, 2*target_L_R+1]``.
    For scalar targets this keeps the final magnetic axis explicit as length 1.
    """

    del optimization_policy  # kept for interface parity with the scalar helper

    l_in = int(l_in)
    power = int(power)
    target_L_R = int(target_L_R)
    slot_count = int(block_values.shape[1])
    basis = _slot_tuple_basis(slot_count, power)
    pieces = []
    for factor_index in range(power):
        slot_indices = torch.tensor(
            [state[factor_index] for state in basis],
            dtype=torch.long,
            device=block_values.device,
        )
        gathered = block_values.index_select(1, slot_indices)
        pieces.append(_packed_block_for_cg(gathered, l_in))
    current, final_L = _couple_block_sequence(
        pieces,
        tuple(int(l_in) for _ in range(power)),
        tuple(int(value) for value in angular_path) + (int(target_L_R),),
        cg_backend="pytorch",
    )
    if int(final_L) != int(target_L_R):
        raise RuntimeError("Internal slot-tuple coupling path did not end at target_L_R.")
    return _as_rank2_equivariant_output(current, int(block_values.shape[0] * len(basis)), target_L_R).reshape(
        int(block_values.shape[0]),
        len(basis),
        2 * int(target_L_R) + 1,
    )


def ye3_slot_specht_power_projected_carriers(
    density,
    channels,
    *,
    max_power=4,
    optimization_policy="auto",
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
    slot_specht_coupling="projected_norm",
):
    """Return projected slot-Specht carrier components for scalar A_s powers.

    The slot group is the diagonal action of ``S_s`` on the slot labels inside
    tensor powers of the slot-resolved density.  This is different from
    inducing child Specht sectors from ``S_a x S_b`` into a larger ``S_N``.

    Each returned tensor is ``P_lambda T`` on the slot-tuple basis for one
    sector and has shape ``[n_atoms, num_slots**power]``.  The current scalar
    linear readout contracts each carrier with its dual via ``||P_lambda T||^2``.
    This exposes the representation-resolved intermediate used by that readout.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    slot_count = int(density.shape[1])
    blocks = lifted_density_channel_blocks(channels)
    sectors = ye3_slot_specht_power_sector_metadata(
        channels,
        num_slots=slot_count,
        max_power=max_power,
        slot_specht_partitions=slot_specht_partitions,
        include_rank1=include_rank1,
        rank_nmax=rank_nmax,
        rank_lmax=rank_lmax,
        rank_lmin=rank_lmin,
        slot_specht_coupling=slot_specht_coupling,
    )
    if not sectors:
        return tuple(), blocks, sectors
    grouped = {}
    for sector_index, sector in enumerate(sectors):
        key = (
            int(sector["block_index"]),
            int(sector["l_in"]),
            int(sector["power"]),
            tuple(int(value) for value in sector["angular_path"]),
        )
        grouped.setdefault(key, []).append((int(sector_index), sector))

    carriers_by_index = [None] * len(sectors)
    block_value_cache = {}
    for key, grouped_sectors in grouped.items():
        block_index, l_in, power, angular_path = key
        block = blocks[int(block_index)]
        block_cache_key = (int(block_index), int(l_in))
        block_values = block_value_cache.get(block_cache_key)
        if block_values is None:
            indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
            block_values = _site_real_block_to_ye3t_tesseral(density.index_select(2, indices), l_in)
            block_value_cache[block_cache_key] = block_values
        tuple_values = _slot_tuple_angular_scalar_values(
            block_values,
            l_in=l_in,
            power=power,
            angular_path=angular_path,
            optimization_policy=optimization_policy,
        )
        for sector_index, sector in grouped_sectors:
            projector, _rank, _idempotent = _slot_specht_central_projector_torch_cached(
                slot_count,
                int(sector["power"]),
                tuple(sector["slot_specht_partition"]),
                dtype=tuple_values.dtype,
                device=tuple_values.device,
            )
            projected = tuple_values @ projector.T
            carriers_by_index[int(sector_index)] = projected
    carriers = tuple(carrier for carrier in carriers_by_index if carrier is not None)
    if len(carriers) != len(sectors):
        raise RuntimeError("Internal slot-Specht sector grouping lost one or more descriptor sectors.")
    return carriers, blocks, sectors


def ye3_slot_specht_power_matrix_unit_carriers(
    density,
    channels,
    *,
    max_power=4,
    optimization_policy="auto",
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
    slot_specht_coupling="projected_norm",
):
    """Return matrix-unit-resolved A_s slot-Specht carrier tensors.

    Each carrier tensor has shape
    ``[n_atoms, d_lambda, d_lambda, num_slots**power]``.  The two
    ``d_lambda`` axes are Young-orthogonal matrix-unit row/column labels for
    the slot Specht sector.  Summing the diagonal slices recovers the central
    projector carrier returned by
    :func:`ye3_slot_specht_power_projected_carriers`.
    """

    projected_carriers, blocks, sectors = ye3_slot_specht_power_projected_carriers(
        density,
        channels,
        max_power=max_power,
        optimization_policy=optimization_policy,
        slot_specht_partitions=slot_specht_partitions,
        include_rank1=include_rank1,
        rank_nmax=rank_nmax,
        rank_lmax=rank_lmax,
        rank_lmin=rank_lmin,
        slot_specht_coupling=slot_specht_coupling,
    )
    density_t = torch.as_tensor(density)
    slot_count = int(density_t.shape[1])
    carriers = []
    metadata = []
    for sector_index, (carrier, sector) in enumerate(zip(projected_carriers, sectors)):
        partition = tuple(int(part) for part in sector["slot_specht_partition"])
        resolved = slot_specht_matrix_unit_carriers(
            carrier,
            slot_count=slot_count,
            power=int(sector["power"]),
            partition=partition,
        )
        carriers.append(resolved)
        metadata.append(
            {
                **dict(sector),
                "sector_index": int(sector_index),
                "matrix_unit_axes": ("tableau_row", "tableau_col"),
                "matrix_unit_count": int(resolved.shape[1]) * int(resolved.shape[2]),
                "carrier_axis": "slot_tuple_power_basis",
                "descriptor_axes": (
                    "atom",
                    "slot_specht_partition",
                    "tableau_row",
                    "tableau_col",
                    "slot_tuple_carrier",
                ),
                "runtime": "slot_specht_matrix_unit_carrier",
                "full_descriptor_contraction_status": "carrier_emitted_not_model_contracted",
            }
        )
    return tuple(carriers), blocks, tuple(metadata)


def ye3_slot_specht_power_equivariant_matrix_unit_carriers(
    density,
    channels,
    *,
    target_L_R_values,
    max_power=4,
    optimization_policy="auto",
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
):
    """Return exact slot-Specht matrix-unit carriers with an explicit ``M_R`` axis.

    Each returned carrier has shape
    ``[n_atoms, d_lambda, d_lambda, num_slots**power, 2*target_L_R+1]``.
    The slot projector is applied independently on each magnetic component.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    requested_targets = tuple(int(value) for value in target_L_R_values)
    if not requested_targets:
        raise ValueError("target_L_R_values must contain at least one nonnegative angular momentum.")
    if any(int(value) < 0 for value in requested_targets):
        raise ValueError("target_L_R_values must be nonnegative.")
    slot_count = int(density.shape[1])
    blocks = lifted_density_channel_blocks(channels)
    partitions = _normalize_slot_specht_partitions(slot_specht_partitions, slot_count=slot_count)
    rank_nmax = _normalize_rank_limit_mapping(rank_nmax, name="rank_nmax")
    rank_lmax = _normalize_rank_limit_mapping(rank_lmax, name="rank_lmax")
    rank_lmin = _normalize_rank_limit_mapping(rank_lmin, name="rank_lmin")
    max_power = int(max_power)
    if max_power < 1:
        raise ValueError("max_power must be positive.")
    sectors = []
    for block_index, block in enumerate(blocks):
        l_in = int(block["l"])
        powers = range(1, max_power + 1) if bool(include_rank1) else range(2, max_power + 1)
        for power in powers:
            if not _rank_limit_allows_block(
                block,
                power,
                rank_nmax=rank_nmax,
                rank_lmax=rank_lmax,
                rank_lmin=rank_lmin,
            ):
                continue
            for target_L_R in requested_targets:
                if power == 1 and int(l_in) != int(target_L_R):
                    continue
                angular_paths = _left_associated_angular_paths_to_target((l_in,) * power, target_L_R)
                for angular_path in angular_paths:
                    for partition in partitions:
                        specht_dimension = int(len(standard_tableaux(tuple(int(part) for part in partition))))
                        projector, projector_rank, _idempotent = _slot_specht_central_projector_torch_cached(
                            slot_count,
                            int(power),
                            tuple(int(part) for part in partition),
                            dtype=density.dtype,
                            device=density.device,
                        )
                        if int(projector_rank) <= 0:
                            continue
                        sectors.append(
                            {
                                "block_index": int(block_index),
                                "n": int(block["n"]),
                                "l_in": int(l_in),
                                "target_L_R": int(target_L_R),
                                "neighbor_type": block["neighbor_type"],
                                "power": int(power),
                                "slot_group": f"S_{slot_count}",
                                "slot_specht_partition": tuple(int(part) for part in partition),
                                "slot_projector_rank": int(projector_rank),
                                "slot_projector_idempotent": True,
                                "slot_specht_dimension": int(specht_dimension),
                                "slot_isotypic_multiplicity": int(projector_rank // max(1, specht_dimension)),
                                "angular_path": tuple(int(value) for value in angular_path),
                                "M_R_values": tuple(range(-int(target_L_R), int(target_L_R) + 1)),
                                "runtime": "slot_specht_matrix_unit_equivariant_carrier",
                                "full_descriptor_contraction_status": "carrier_emitted_not_model_contracted",
                            }
                        )
    if not sectors:
        return tuple(), blocks, tuple()

    grouped = {}
    for sector_index, sector in enumerate(sectors):
        key = (
            int(sector["block_index"]),
            int(sector["l_in"]),
            int(sector["power"]),
            int(sector["target_L_R"]),
            tuple(int(value) for value in sector["angular_path"]),
        )
        grouped.setdefault(key, []).append((int(sector_index), sector))

    carriers_by_index = [None] * len(sectors)
    block_value_cache = {}
    for key, grouped_sectors in grouped.items():
        block_index, l_in, power, target_L_R, angular_path = key
        block = blocks[int(block_index)]
        block_cache_key = (int(block_index), int(l_in))
        block_values = block_value_cache.get(block_cache_key)
        if block_values is None:
            indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
            block_values = _site_real_block_to_ye3t_tesseral(density.index_select(2, indices), l_in)
            block_value_cache[block_cache_key] = block_values
        tuple_values = _slot_tuple_angular_values(
            block_values,
            l_in=l_in,
            power=power,
            angular_path=angular_path,
            target_L_R=target_L_R,
            optimization_policy=optimization_policy,
        )
        tuple_values_batch = torch.movedim(tuple_values, -1, 1).reshape(
            int(tuple_values.shape[0]) * int(tuple_values.shape[-1]),
            int(tuple_values.shape[1]),
        )
        for sector_index, sector in grouped_sectors:
            partition = tuple(int(part) for part in sector["slot_specht_partition"])
            resolved = slot_specht_matrix_unit_carriers(
                tuple_values_batch,
                slot_count=slot_count,
                power=int(sector["power"]),
                partition=partition,
            )
            resolved = resolved.reshape(
                int(tuple_values.shape[0]),
                int(tuple_values.shape[-1]),
                int(resolved.shape[1]),
                int(resolved.shape[2]),
                int(tuple_values.shape[1]),
            )
            resolved = torch.movedim(resolved, 1, -1)
            carriers_by_index[int(sector_index)] = resolved
    carriers = []
    metadata = []
    for sector_index, (carrier, sector) in enumerate(zip(carriers_by_index, sectors)):
        if carrier is None:
            raise RuntimeError("Internal slot-Specht equivariant carrier grouping lost a descriptor sector.")
        carriers.append(carrier)
        metadata.append(
            {
                **dict(sector),
                "sector_index": int(sector_index),
                "matrix_unit_axes": ("tableau_row", "tableau_col"),
                "matrix_unit_count": int(carrier.shape[1]) * int(carrier.shape[2]),
                "carrier_axis": "slot_tuple_power_basis",
                "descriptor_axes": (
                    "atom",
                    "slot_specht_partition",
                    "tableau_row",
                    "tableau_col",
                    "slot_tuple_carrier",
                    "M_R",
                ),
            }
        )
    return tuple(carriers), blocks, tuple(metadata)


def slot_specht_projected_trivial_pairing(left_carrier, right_carrier):
    """Pair matching projected slot-Specht carriers to a trivial scalar.

    ``left_carrier`` and ``right_carrier`` are expected to already live in the
    same projected central-idempotent image, such as the tensors returned by
    :func:`ye3_slot_specht_power_projected_carriers` for the same sector.
    The returned scalar is the Euclidean carrier-dual pairing along the
    slot-tuple carrier axis.  This is the implemented restricted
    ``carrier x carrier* -> trivial`` runtime used by the current scalar linear
    A_s readout; it is not the full multiplicity-resolved
    ``lambda x mu* -> nu`` matrix-unit backend.
    """

    left = torch.as_tensor(left_carrier)
    right = torch.as_tensor(right_carrier, dtype=left.dtype, device=left.device)
    if left.shape != right.shape:
        raise ValueError(
            "Projected slot-Specht carriers must have matching shapes for trivial pairing; "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if left.ndim != 2:
        raise ValueError("Projected slot-Specht carriers must have shape [n_atoms, carrier_dim].")
    return (left * right).sum(dim=1, keepdim=True)


def ye3_slot_specht_power_invariants(
    density,
    channels,
    *,
    max_power=4,
    optimization_policy="auto",
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
    slot_specht_coupling="projected_norm",
):
    """Return scalar invariants from exact A_s slot-Specht central projectors.

    With ``slot_specht_coupling='projected_norm'`` the emitted scalar feature
    is the carrier-dual pairing ``<P_lambda T, P_lambda T>`` for each sector
    exposed by :func:`ye3_slot_specht_power_projected_carriers`.

    With ``slot_specht_coupling='commutant_symmetric'`` the emitted columns are
    all symmetric invariant quadratic forms from the finite slot-action
    commutant basis on each projected carrier image.
    """

    density = torch.as_tensor(density)
    slot_specht_coupling = str(slot_specht_coupling).strip().lower()
    if slot_specht_coupling not in {"projected_norm", "commutant_symmetric"}:
        raise ValueError("slot_specht_coupling must be 'projected_norm' or 'commutant_symmetric'.")
    carriers, blocks, sectors = ye3_slot_specht_power_projected_carriers(
        density,
        channels,
        max_power=max_power,
        optimization_policy=optimization_policy,
        slot_specht_partitions=slot_specht_partitions,
        include_rank1=include_rank1,
        rank_nmax=rank_nmax,
        rank_lmax=rank_lmax,
        rank_lmin=rank_lmin,
        slot_specht_coupling=slot_specht_coupling,
    )
    if not sectors:
        return density.new_zeros((int(density.shape[0]), 0)), blocks, sectors
    if slot_specht_coupling == "projected_norm":
        features = [slot_specht_projected_trivial_pairing(carrier, carrier) for carrier in carriers]
    else:
        features = [
            slot_specht_projected_commutant_quadratic_features(
                carrier,
                slot_count=int(density.shape[1]),
                power=int(sector["power"]),
                partition=tuple(int(part) for part in sector["slot_specht_partition"]),
            )
            for carrier, sector in zip(carriers, sectors)
        ]
    return torch.cat(features, dim=1), blocks, sectors


def ye3_slot_specht_power_l0_density_adjoint(
    density,
    channels,
    sector_indices,
    *,
    sectors=None,
    max_power=4,
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
):
    """Return exact density adjoints for l=0 slot-Specht scalar power sectors.

    For ``l_in = 0`` the angular scalar in each tuple slot is a product of
    scalar slot values.  The scalar feature is ``||P_lambda t||^2`` for the
    central idempotent ``P_lambda`` on the slot-tuple basis, so
    ``d/dt ||P t||^2 = 2 P t``.  This helper implements only that l=0 case and
    raises if a requested sector has nonzero ``l_in``.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    slot_count = int(density.shape[1])
    blocks = lifted_density_channel_blocks(channels)
    if sectors is None:
        sectors = ye3_slot_specht_power_sector_metadata(
            channels,
            num_slots=slot_count,
            max_power=max_power,
            slot_specht_partitions=slot_specht_partitions,
            include_rank1=include_rank1,
            rank_nmax=rank_nmax,
            rank_lmax=rank_lmax,
            rank_lmin=rank_lmin,
        )
    requested = tuple(int(index) for index in sector_indices)
    adjoints = density.new_zeros((len(requested),) + tuple(density.shape))
    for row, sector_index in enumerate(requested):
        sector = sectors[int(sector_index)]
        if int(sector.get("multiplicity", 1)) != 1 or sector.get("slot_specht_coupling", "projected_norm") != "projected_norm":
            raise ValueError("slot-Specht analytic density adjoints currently support only projected_norm one-feature sectors.")
        l_in = int(sector["l_in"])
        if l_in != 0:
            raise ValueError("ye3_slot_specht_power_l0_density_adjoint only supports l_in=0 sectors.")
        block = blocks[int(sector["block_index"])]
        block_indices = tuple(int(index) for index in block["indices"])
        if len(block_indices) != 1:
            raise ValueError("l_in=0 slot-Specht adjoint expects one scalar m=0 channel.")
        channel_index = block_indices[0]
        power = int(sector["power"])
        basis = _slot_tuple_basis(slot_count, power)
        values = density[:, :, channel_index]
        if power == 1:
            tuple_values = values
        else:
            factors = [
                values.index_select(
                    1,
                    torch.tensor([state[factor] for state in basis], dtype=torch.long, device=density.device),
                )
                for factor in range(power)
            ]
            tuple_values = factors[0]
            for factor_values in factors[1:]:
                tuple_values = tuple_values * factor_values
        projector, _rank, _idempotent = _slot_specht_central_projector_torch_cached(
            slot_count,
            power,
            tuple(sector["slot_specht_partition"]),
            dtype=tuple_values.dtype,
            device=tuple_values.device,
        )
        tuple_adjoint = 2.0 * (tuple_values @ projector)
        slot_adjoint = density.new_zeros((int(density.shape[0]), slot_count))
        if power == 1:
            slot_indices = torch.tensor([state[0] for state in basis], dtype=torch.long, device=density.device)
            slot_adjoint.index_add_(1, slot_indices, tuple_adjoint)
        else:
            for factor in range(power):
                slot_indices = torch.tensor(
                    [state[factor] for state in basis],
                    dtype=torch.long,
                    device=density.device,
                )
                partial = tuple_adjoint
                for other in range(power):
                    if other == factor:
                        continue
                    partial = partial * values.index_select(
                        1,
                        torch.tensor([state[other] for state in basis], dtype=torch.long, device=density.device),
                    )
                slot_adjoint.index_add_(1, slot_indices, partial)
        adjoints[row, :, :, channel_index] = slot_adjoint
    return adjoints


def ye3_slot_specht_power_rank2_density_adjoint(
    density,
    channels,
    sector_indices,
    *,
    sectors=None,
    max_power=4,
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
):
    """Return density adjoints for rank-2 slot-Specht scalar sectors.

    This handles sectors with ``power = 2`` and arbitrary ``l_in``.  The angular
    scalar is evaluated with the same real-tesseral paired CG helper used by the
    forward path, so the convention matches the public runtime.  Only the small
    tuple-block CG contraction is differentiated locally; the surrounding
    density/normalization reverse pass remains analytic.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    slot_count = int(density.shape[1])
    blocks = lifted_density_channel_blocks(channels)
    if sectors is None:
        sectors = ye3_slot_specht_power_sector_metadata(
            channels,
            num_slots=slot_count,
            max_power=max_power,
            slot_specht_partitions=slot_specht_partitions,
            include_rank1=include_rank1,
            rank_nmax=rank_nmax,
            rank_lmax=rank_lmax,
            rank_lmin=rank_lmin,
        )
    requested = tuple(int(index) for index in sector_indices)
    adjoints = density.new_zeros((len(requested),) + tuple(density.shape))
    for row, sector_index in enumerate(requested):
        sector = sectors[int(sector_index)]
        if int(sector.get("multiplicity", 1)) != 1 or sector.get("slot_specht_coupling", "projected_norm") != "projected_norm":
            raise ValueError("slot-Specht analytic density adjoints currently support only projected_norm one-feature sectors.")
        if int(sector["power"]) != 2:
            raise ValueError("ye3_slot_specht_power_rank2_density_adjoint only supports power=2 sectors.")
        l_in = int(sector["l_in"])
        block = blocks[int(sector["block_index"])]
        indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
        block_values = _site_real_block_to_ye3t_tesseral(density.index_select(2, indices), l_in)
        local_values = block_values.detach().clone().requires_grad_(True)
        basis = _slot_tuple_basis(slot_count, 2)
        left_slots = torch.tensor([state[0] for state in basis], dtype=torch.long, device=density.device)
        right_slots = torch.tensor([state[1] for state in basis], dtype=torch.long, device=density.device)
        left = local_values.index_select(1, left_slots)
        right = local_values.index_select(1, right_slots)
        left_arg = left if l_in != 0 else left.unsqueeze(-1)
        right_arg = right if l_in != 0 else right.unsqueeze(-1)
        tuple_values, _backend = couple_packed_real_tesseral(
            left_arg,
            right_arg,
            l_in,
            l_in,
            0,
            backend="pytorch",
        )
        if tuple_values.ndim == 3:
            tuple_values = tuple_values.squeeze(-1)
        projector, _rank, _idempotent = _slot_specht_central_projector_torch_cached(
            slot_count,
            2,
            tuple(sector["slot_specht_partition"]),
            dtype=tuple_values.dtype,
            device=tuple_values.device,
        )
        projected = tuple_values @ projector.T
        feature = projected.pow(2).sum()
        block_adjoint = torch.autograd.grad(feature, local_values, create_graph=False, retain_graph=False)[0]
        inverse_indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
        # ``_site_real_block_to_ye3t_tesseral`` flips m-order for l>0, so reverse
        # that flip before scattering back into the original channel order.
        if l_in != 0:
            block_adjoint = torch.flip(block_adjoint, dims=(-1,))
        adjoints[row].index_copy_(2, inverse_indices, block_adjoint)
    return adjoints


def ye3_slot_specht_power_local_density_adjoint(
    density,
    channels,
    sector_indices,
    *,
    sectors=None,
    max_power=4,
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
    optimization_policy="auto",
):
    """Return density adjoints for arbitrary scalar slot-Specht power sectors.

    This is the general local derivative path for the currently implemented
    scalar readout

        ``feature = || P_lambda T(A_s) ||^2``.

    It differentiates only the local slot-tuple/angular contraction for each
    requested sector and leaves the surrounding normalized filtered-density
    reverse pass to ``HybridACELiftedDensityEnergyModel.filtered_density_vjp``.
    This remains a scalar central-projector norm readout.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    slot_count = int(density.shape[1])
    blocks = lifted_density_channel_blocks(channels)
    if sectors is None:
        sectors = ye3_slot_specht_power_sector_metadata(
            channels,
            num_slots=slot_count,
            max_power=max_power,
            slot_specht_partitions=slot_specht_partitions,
            include_rank1=include_rank1,
            rank_nmax=rank_nmax,
            rank_lmax=rank_lmax,
            rank_lmin=rank_lmin,
        )
    requested = tuple(int(index) for index in sector_indices)
    adjoints = density.new_zeros((len(requested),) + tuple(density.shape))
    for row, sector_index in enumerate(requested):
        sector = sectors[int(sector_index)]
        if int(sector.get("multiplicity", 1)) != 1 or sector.get("slot_specht_coupling", "projected_norm") != "projected_norm":
            raise ValueError("slot-Specht analytic density adjoints currently support only projected_norm one-feature sectors.")
        if int(sector.get("target_L_R", 0)) != 0:
            raise ValueError("ye3_slot_specht_power_local_density_adjoint currently supports target_L_R=0.")
        l_in = int(sector["l_in"])
        power = int(sector["power"])
        block = blocks[int(sector["block_index"])]
        indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
        block_values = _site_real_block_to_ye3t_tesseral(density.index_select(2, indices), l_in)
        local_values = block_values.detach().clone().requires_grad_(True)
        tuple_values = _slot_tuple_angular_scalar_values(
            local_values,
            l_in=l_in,
            power=power,
            angular_path=tuple(int(value) for value in sector["angular_path"]),
            optimization_policy=optimization_policy,
        )
        projector, _rank, _idempotent = _slot_specht_central_projector_torch_cached(
            slot_count,
            power,
            tuple(sector["slot_specht_partition"]),
            dtype=tuple_values.dtype,
            device=tuple_values.device,
        )
        projected = tuple_values @ projector.T
        feature = projected.pow(2).sum()
        block_adjoint = torch.autograd.grad(
            feature,
            local_values,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
        if l_in != 0:
            block_adjoint = torch.flip(block_adjoint, dims=(-1,))
        adjoints[row].index_copy_(2, indices, block_adjoint)
    return adjoints


def ye3_slot_specht_power_commutant_density_adjoint(
    density,
    channels,
    feature_indices,
    *,
    sectors=None,
    max_power=4,
    slot_specht_partitions=None,
    include_rank1=False,
    rank_nmax=None,
    rank_lmax=None,
    rank_lmin=None,
    optimization_policy="auto",
):
    """Return density adjoints for commutant-symmetric slot-Specht features.

    ``feature_indices`` are column indices in the concatenated output of
    ``ye3_slot_specht_power_invariants(..., slot_specht_coupling=
    'commutant_symmetric')``.  The implementation groups requested columns by
    sector and differentiates the local tuple/angular contraction with batched
    autograd, then scatters the result back to the original ``A_s`` channel
    order.  The surrounding normalized density-to-position VJP remains handled
    by ``HybridACELiftedDensityEnergyModel.filtered_density_vjp``.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    slot_count = int(density.shape[1])
    blocks = lifted_density_channel_blocks(channels)
    if sectors is None:
        sectors = ye3_slot_specht_power_sector_metadata(
            channels,
            num_slots=slot_count,
            max_power=max_power,
            slot_specht_partitions=slot_specht_partitions,
            include_rank1=include_rank1,
            rank_nmax=rank_nmax,
            rank_lmax=rank_lmax,
            rank_lmin=rank_lmin,
            slot_specht_coupling="commutant_symmetric",
        )
    requested = tuple(int(index) for index in feature_indices)
    slices = slot_specht_sector_feature_slices(sectors)
    assignments = []
    for output_row, feature_index in enumerate(requested):
        for sector_index, sector_slice in enumerate(slices):
            if sector_slice.start <= feature_index < sector_slice.stop:
                assignments.append((output_row, sector_index, feature_index - sector_slice.start))
                break
        else:
            raise IndexError(f"Slot-Specht commutant feature index {feature_index} is out of range.")
    by_sector = {}
    for output_row, sector_index, local_feature_index in assignments:
        by_sector.setdefault(int(sector_index), []).append((int(output_row), int(local_feature_index)))

    adjoints = density.new_zeros((len(requested),) + tuple(density.shape))
    for sector_index, rows_and_features in by_sector.items():
        sector = sectors[int(sector_index)]
        if sector.get("slot_specht_coupling", "projected_norm") != "commutant_symmetric":
            raise ValueError("commutant density adjoint requires commutant_symmetric sector metadata.")
        if int(sector.get("target_L_R", 0)) != 0:
            raise ValueError("commutant density adjoint currently supports target_L_R=0.")
        l_in = int(sector["l_in"])
        power = int(sector["power"])
        block = blocks[int(sector["block_index"])]
        indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
        block_values = _site_real_block_to_ye3t_tesseral(density.index_select(2, indices), l_in)
        local_values = block_values.detach().clone().requires_grad_(True)
        tuple_values = _slot_tuple_angular_scalar_values(
            local_values,
            l_in=l_in,
            power=power,
            angular_path=tuple(int(value) for value in sector["angular_path"]),
            optimization_policy=optimization_policy,
        )
        projector, _rank, _idempotent = _slot_specht_central_projector_torch_cached(
            slot_count,
            power,
            tuple(sector["slot_specht_partition"]),
            dtype=tuple_values.dtype,
            device=tuple_values.device,
        )
        carrier = tuple_values @ projector.T
        basis = _slot_specht_commutant_symmetric_basis_torch_cached(
            slot_count,
            power,
            tuple(sector["slot_specht_partition"]),
            dtype=tuple_values.dtype,
            device=tuple_values.device,
        )
        local_indices = torch.tensor(
            [local_feature_index for _row, local_feature_index in rows_and_features],
            dtype=torch.long,
            device=density.device,
        )
        selected_basis = basis.index_select(0, local_indices)
        selected_features = torch.einsum("bi,kij,bj->k", carrier, selected_basis, carrier)
        eye = torch.eye(
            int(selected_features.numel()),
            dtype=selected_features.dtype,
            device=selected_features.device,
        )
        block_adjoint = torch.autograd.grad(
            selected_features,
            local_values,
            grad_outputs=eye,
            is_grads_batched=True,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
        if l_in != 0:
            block_adjoint = torch.flip(block_adjoint, dims=(-1,))
        for local_batch_row, (output_row, _local_feature_index) in enumerate(rows_and_features):
            adjoints[int(output_row)].index_copy_(2, indices, block_adjoint[int(local_batch_row)])
    return adjoints


def ye3_power_equivariants(
    density,
    channels,
    *,
    max_power=4,
    target_L_R_values=None,
    optimization_policy="auto",
):
    """Return Young-slot/E3 symmetric-power feature blocks grouped by ``target_L_R``.

    This evaluates equivariant feature blocks
    ``Sym^p(V_l_in) -> V_target_L_R`` from complete real-spherical channel
    blocks.  The returned dictionary maps each integer ``target_L_R`` to a
    tensor with shape ``[n_atoms, n_features_for_L_R, 2*target_L_R + 1]``.

    The slot-trivial component is evaluated once per atom.  The slot-standard
    residual is evaluated per slot and then summed over the slot axis, producing
    a slot-invariant but rotation-equivariant feature block.  These features are
    not scalar energy contributions unless ``target_L_R == 0``.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    atom_count = int(density.shape[0])
    blocks = lifted_density_channel_blocks(channels)
    sectors = ye3_power_equivariant_sector_metadata(
        channels,
        max_power=max_power,
        target_L_R_values=target_L_R_values,
    )
    if not sectors:
        return {}, blocks, sectors
    mu = slot_trivial_component(density)
    eta = slot_standard_residual(density)
    features_by_L_R = {}
    for sector in sectors:
        block = blocks[int(sector["block_index"])]
        indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
        power = int(sector["power"])
        l_in = int(sector["l_in"])
        target_L_R = int(sector["target_L_R"])
        if sector["slot_sector"] == "trivial":
            block_values = _site_real_block_to_ye3t_tesseral(mu.index_select(1, indices), l_in)
            value = symmetric_power_real_tesseral(
                block_values,
                power,
                l_in,
                target_L_R,
                optimization_policy=optimization_policy,
            )
            feature = _as_equivariant_power_output(value, atom_count, target_L_R)
        else:
            eta_block = _site_real_block_to_ye3t_tesseral(eta.index_select(2, indices), l_in)
            flat_eta = eta_block.reshape(-1, 2 * l_in + 1)
            value = symmetric_power_real_tesseral(
                flat_eta,
                power,
                l_in,
                target_L_R,
                optimization_policy=optimization_policy,
            )
            feature = _as_equivariant_power_output(
                value,
                atom_count * int(density.shape[1]),
                target_L_R,
            ).reshape(atom_count, int(density.shape[1]), -1, 2 * target_L_R + 1).sum(dim=1)
        features_by_L_R.setdefault(target_L_R, []).append(feature)
    return {
        target_L_R: torch.cat(parts, dim=1)
        for target_L_R, parts in sorted(features_by_L_R.items())
    }, blocks, sectors


def _packed_block_for_cg(value, l_value):
    l_value = int(l_value)
    if l_value == 0:
        return value.reshape(value.shape[0], -1)
    return value.reshape(value.shape[0], -1, 2 * l_value + 1)


def _as_rank2_equivariant_output(value, atom_count, target_L_R):
    target_L_R = int(target_L_R)
    out_dim = 2 * target_L_R + 1
    if target_L_R == 0:
        if value.ndim == 1:
            return value.reshape(int(atom_count), 1, 1)
        return value.reshape(int(atom_count), -1, 1)
    if value.ndim == 2:
        return value.reshape(int(atom_count), 1, out_dim)
    return value.reshape(int(atom_count), -1, out_dim)


def _couple_block_sequence(values, l_in, intermediate_L_R, *, cg_backend="pytorch"):
    l_in = tuple(int(value) for value in l_in)
    intermediate_L_R = tuple(int(value) for value in intermediate_L_R)
    if not values:
        raise ValueError("values must contain at least one block tensor.")
    current_L = int(l_in[0])
    current = _packed_block_for_cg(values[0], current_L)
    for step, next_value in enumerate(values[1:]):
        next_L = int(l_in[int(step) + 1])
        target_L = int(intermediate_L_R[int(step)])
        right = _packed_block_for_cg(next_value, next_L)
        if int(right.shape[1]) == 1 and int(current.shape[1]) != 1:
            expand_shape = (int(current.shape[0]), int(current.shape[1])) + tuple(right.shape[2:])
            right = right.expand(expand_shape)
        elif int(current.shape[1]) == 1 and int(right.shape[1]) != 1:
            expand_shape = (int(right.shape[0]), int(right.shape[1])) + tuple(current.shape[2:])
            current = current.expand(expand_shape)
        elif int(current.shape[1]) != int(right.shape[1]):
            raise ValueError("Packed CG factors must have matching channel axes after expansion.")
        if int(current_L) == 0 and int(target_L) == int(next_L):
            scalar = current.reshape(int(current.shape[0]), int(current.shape[1]), 1)
            covariant = right.reshape(int(right.shape[0]), int(right.shape[1]), 2 * int(next_L) + 1)
            current = scalar * covariant
        elif int(next_L) == 0 and int(target_L) == int(current_L):
            covariant = current.reshape(int(current.shape[0]), int(current.shape[1]), 2 * int(current_L) + 1)
            scalar = right.reshape(int(right.shape[0]), int(right.shape[1]), 1)
            current = covariant * scalar
        else:
            current, _backend = couple_packed_real_tesseral(
                current,
                right,
                current_L,
                next_L,
                target_L,
                backend=cg_backend,
            )
        current_L = target_L
    return current, current_L


def ye3_rank2_orbit_equivariants(
    density,
    channels,
    *,
    target_L_R_values=None,
    include_ordered_pairs=False,
    cg_backend="pytorch",
):
    """Return rank-2 orbit-product feature blocks grouped by ``target_L_R``.

    This evaluates arbitrary complete block pairs
    ``V_{l_in[0]} x V_{l_in[1]} -> V_{target_L_R}``, including non-degenerate
    pairs with ``pair_orbits=(1,1)``.  It is not restricted to fully degenerate
    repeated inputs.  The returned dictionary maps each ``target_L_R`` to a
    tensor with shape ``[n_atoms, n_features_for_L_R, 2*target_L_R + 1]``.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    atom_count = int(density.shape[0])
    slot_count = int(density.shape[1])
    blocks = lifted_density_channel_blocks(channels)
    sectors = ye3_rank2_orbit_equivariant_sector_metadata(
        channels,
        target_L_R_values=target_L_R_values,
        include_ordered_pairs=include_ordered_pairs,
    )
    if not sectors:
        return {}, blocks, sectors
    mu = slot_trivial_component(density)
    eta = slot_standard_residual(density)
    features_by_L_R = {}
    for sector in sectors:
        left_block = blocks[int(sector["left_block_index"])]
        right_block = blocks[int(sector["right_block_index"])]
        left_indices = torch.tensor(left_block["indices"], dtype=torch.long, device=density.device)
        right_indices = torch.tensor(right_block["indices"], dtype=torch.long, device=density.device)
        l_left, l_right = (int(value) for value in sector["l_in"])
        target_L_R = int(sector["target_L_R"])
        if sector["slot_sector"] == "trivial":
            left = _site_real_block_to_ye3t_tesseral(mu.index_select(1, left_indices), l_left)
            right = _site_real_block_to_ye3t_tesseral(mu.index_select(1, right_indices), l_right)
            coupled, _backend = couple_packed_real_tesseral(
                _packed_block_for_cg(left, l_left),
                _packed_block_for_cg(right, l_right),
                l_left,
                l_right,
                target_L_R,
                backend=cg_backend,
            )
            feature = _as_rank2_equivariant_output(coupled, atom_count, target_L_R)
        else:
            left = _site_real_block_to_ye3t_tesseral(eta.index_select(2, left_indices), l_left)
            right = _site_real_block_to_ye3t_tesseral(eta.index_select(2, right_indices), l_right)
            left_flat = left.reshape(atom_count * slot_count, -1)
            right_flat = right.reshape(atom_count * slot_count, -1)
            coupled, _backend = couple_packed_real_tesseral(
                _packed_block_for_cg(left_flat, l_left),
                _packed_block_for_cg(right_flat, l_right),
                l_left,
                l_right,
                target_L_R,
                backend=cg_backend,
            )
            feature = _as_rank2_equivariant_output(
                coupled,
                atom_count * slot_count,
                target_L_R,
            ).reshape(atom_count, slot_count, -1, 2 * target_L_R + 1).sum(dim=1)
        features_by_L_R.setdefault(target_L_R, []).append(feature)
    return {
        target_L_R: torch.cat(parts, dim=1)
        for target_L_R, parts in sorted(features_by_L_R.items())
    }, blocks, sectors


def ye3_orbit_product_equivariants(
    density,
    channels,
    *,
    block_tuples,
    target_L_R_values,
    max_intermediate_L_R=None,
    symmetrize_factor_orbits=True,
    cg_backend="pytorch",
):
    """Return general N-rank orbit-product feature blocks grouped by ``target_L_R``.

    The evaluator supports arbitrary complete channel-block tuples.  It uses
    left-associated packed Clebsch-Gordan paths and, by default, averages over
    distinct permutations of the factor tuple.  This implements the
    factor-orbit trivial projection for repeated/partially repeated/
    non-degenerate tuples.  It is not a full Specht-sector projector.
    """

    density = torch.as_tensor(density)
    if density.ndim != 3:
        raise ValueError("density must have shape [n_atoms, n_slots, n_channels].")
    atom_count = int(density.shape[0])
    slot_count = int(density.shape[1])
    blocks = lifted_density_channel_blocks(channels)
    sectors = ye3_orbit_product_equivariant_sector_metadata(
        channels,
        block_tuples=block_tuples,
        target_L_R_values=target_L_R_values,
        max_intermediate_L_R=max_intermediate_L_R,
        symmetrize_factor_orbits=symmetrize_factor_orbits,
    )
    if not sectors:
        return {}, blocks, sectors
    mu = slot_trivial_component(density)
    eta = slot_standard_residual(density)
    features_by_L_R = {}
    permutation_cache = {}
    for sector in sectors:
        block_tuple = tuple(int(index) for index in sector["block_tuple"])
        factor_permutations = permutation_cache.get(block_tuple)
        if factor_permutations is None:
            factor_permutations = (
                _distinct_permutations(block_tuple)
                if str(sector["factor_orbit_projection"]) == "trivial_average"
                else (block_tuple,)
            )
            permutation_cache[block_tuple] = factor_permutations
        target_L_R = int(sector["target_L_R"])
        pieces = []
        for permuted_tuple in factor_permutations:
            permuted_l_in = tuple(int(blocks[index]["l"]) for index in permuted_tuple)
            if sector["slot_sector"] == "trivial":
                values = []
                for block_index in permuted_tuple:
                    block = blocks[int(block_index)]
                    indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
                    values.append(
                        _site_real_block_to_ye3t_tesseral(
                            mu.index_select(1, indices).reshape(atom_count, 1, -1),
                            int(block["l"]),
                        )
                    )
                coupled, final_L = _couple_block_sequence(
                    values,
                    permuted_l_in,
                    sector["intermediate_L_R"],
                    cg_backend=cg_backend,
                )
                if int(final_L) != target_L_R:
                    raise RuntimeError("Internal coupling path did not end at target_L_R.")
                pieces.append(_as_rank2_equivariant_output(coupled, atom_count, target_L_R))
            else:
                values = []
                for block_index in permuted_tuple:
                    block = blocks[int(block_index)]
                    indices = torch.tensor(block["indices"], dtype=torch.long, device=density.device)
                    values.append(
                        _site_real_block_to_ye3t_tesseral(
                            eta.index_select(2, indices).reshape(atom_count * slot_count, 1, -1),
                            int(block["l"]),
                        )
                    )
                coupled, final_L = _couple_block_sequence(
                    values,
                    permuted_l_in,
                    sector["intermediate_L_R"],
                    cg_backend=cg_backend,
                )
                if int(final_L) != target_L_R:
                    raise RuntimeError("Internal coupling path did not end at target_L_R.")
                pieces.append(
                    _as_rank2_equivariant_output(
                        coupled,
                        atom_count * slot_count,
                        target_L_R,
                    ).reshape(atom_count, slot_count, -1, 2 * target_L_R + 1).sum(dim=1)
                )
        feature = sum(pieces) / float(len(pieces))
        features_by_L_R.setdefault(target_L_R, []).append(feature)
    return {
        target_L_R: torch.cat(parts, dim=1)
        for target_L_R, parts in sorted(features_by_L_R.items())
    }, blocks, sectors


@recordclass(('n', 'l', 'm', 'neighbor_type'), frozen = True)
class LiftedDensityChannel:
    """One real-spherical edge channel used in the filtered density."""

    n = 1
    l = 0
    m = 0
    neighbor_type = None

    def __post_init__(self):
        n = int(self.n)
        l = int(self.l)
        m = int(self.m)
        if n < 0:
            raise ValueError("LiftedDensityChannel.n must be nonnegative.")
        if l < 0:
            raise ValueError("LiftedDensityChannel.l must be nonnegative.")
        if abs(m) > l:
            raise ValueError("LiftedDensityChannel.m must satisfy -l <= m <= l.")
        neighbor_type = None if self.neighbor_type is None else int(self.neighbor_type)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "l", l)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "neighbor_type", neighbor_type)

    def to_dict(self):
        return {"n": self.n, "l": self.l, "m": self.m, "neighbor_type": self.neighbor_type}

    @classmethod
    def from_dict(cls, payload):
        return cls(
            n=int(payload.get("n", 1)),
            l=int(payload.get("l", 0)),
            m=int(payload.get("m", 0)),
            neighbor_type=payload.get("neighbor_type", None),
        )


@recordclass(('cutoff', 'channels', 'possible_types', 'filter_kind', 'num_filters', 'filter_centers', 'filter_width', 'radial_lambda', 'pair_cutoffs', 'pair_radial_lambdas', 'pair_filter_specs', 'pair_filter_execution_policy', 'readout_mode', 'slot_group', 'young_subgroup_blocks', 'density_normalization', 'density_normalization_nugget', 'feature_normalization', 'hidden_layers', 'ye3_max_power', 'ye3_optimization_policy', 'ye3_slot_sectors', 'ye3_slot_specht_partitions', 'ye3_slot_specht_coupling', 'ye3_include_rank1', 'ye3_rank_nmax', 'ye3_rank_lmax', 'ye3_rank_lmin', 'periodic_image_mode', 'enforce_unique_periodic_images', 'periodic_image_margin', 'source_backend', 'native_source_min_edges'), frozen = True)
class LiftedDensityConfig:
    """Configuration for the ``A_s`` filtered density branch."""

    cutoff = 4.0
    channels = (LiftedDensityChannel(),)
    possible_types = ()
    filter_kind = "softmax_gaussian"
    num_filters = 3
    filter_centers = ()
    filter_width = 0.25
    radial_lambda = 0.25
    pair_cutoffs = None
    pair_radial_lambdas = None
    pair_filter_specs = None
    pair_filter_execution_policy = "auto"
    readout_mode = "linear"
    slot_group = "symmetric"
    young_subgroup_blocks = ()
    density_normalization = DEFAULT_A_S_DENSITY_NORMALIZATION
    density_normalization_nugget = 0.0
    feature_normalization = DEFAULT_A_S_FEATURE_NORMALIZATION
    hidden_layers = 1
    ye3_max_power = 4
    ye3_optimization_policy = "auto"
    ye3_slot_sectors = ("trivial", "standard")
    ye3_slot_specht_partitions = None
    ye3_slot_specht_coupling = "projected_norm"
    ye3_include_rank1 = False
    ye3_rank_nmax = None
    ye3_rank_lmax = None
    ye3_rank_lmin = None
    periodic_image_mode = "unique"
    enforce_unique_periodic_images = True
    periodic_image_margin = 1.0e-8
    source_backend = "auto"
    native_source_min_edges = 0

    def __post_init__(self):
        channels = tuple(
            ch if isinstance(ch, LiftedDensityChannel) else LiftedDensityChannel.from_dict(ch)
            for ch in self.channels
        )
        if not channels:
            channels = (LiftedDensityChannel(),)
        possible_types = tuple(
            sorted({int(value) for value in self.possible_types})
        )
        if any(value < 0 for value in possible_types):
            raise ValueError(
                "LiftedDensityConfig.possible_types must be nonnegative."
            )
        if possible_types and any(
            channel.neighbor_type is not None
            and int(channel.neighbor_type) not in possible_types
            for channel in channels
        ):
            raise ValueError(
                "Every neighbor-specific lifted channel must belong to "
                "LiftedDensityConfig.possible_types."
            )
        filter_kind = str(self.filter_kind).strip().lower()
        allowed_filters = {"radial_gaussian", "softmax_gaussian", "cosine_shell", "bernstein", "constant"}
        if filter_kind not in allowed_filters:
            raise ValueError(
                "LiftedDensityConfig.filter_kind must be one of "
                "'radial_gaussian', 'softmax_gaussian', 'cosine_shell', 'bernstein', or 'constant'."
            )
        num_filters = int(self.num_filters)
        if num_filters < 1:
            raise ValueError("LiftedDensityConfig.num_filters must be positive.")
        centers = tuple(float(x) for x in self.filter_centers)
        if filter_kind in {"radial_gaussian", "softmax_gaussian", "cosine_shell"}:
            if not centers:
                if num_filters == 1:
                    centers = (0.5,)
                else:
                    centers = tuple(float(x) for x in np.linspace(0.2, 0.8, num_filters))
            num_filters = len(centers)
            if float(self.filter_width) <= 0.0:
                raise ValueError("LiftedDensityConfig.filter_width must be positive.")
        elif filter_kind == "bernstein":
            centers = (0.5,) if num_filters == 1 else tuple(float(x) for x in np.linspace(0.0, 1.0, num_filters))
        else:
            centers = tuple(float(x) for x in range(num_filters))
        radial_lambda = float(self.radial_lambda)
        if radial_lambda <= 0.0:
            raise ValueError("LiftedDensityConfig.radial_lambda must be positive.")
        pair_cutoffs = _normalize_pair_float_mapping(self.pair_cutoffs, name="pair_cutoffs")
        if any(float(value) <= 0.0 for value in pair_cutoffs.values()):
            raise ValueError("LiftedDensityConfig.pair_cutoffs values must be positive.")
        pair_radial_lambdas = _normalize_pair_float_mapping(self.pair_radial_lambdas, name="pair_radial_lambdas")
        if any(float(value) <= 0.0 for value in pair_radial_lambdas.values()):
            raise ValueError("LiftedDensityConfig.pair_radial_lambdas values must be positive.")
        pair_filter_specs = _normalize_pair_filter_specs(
            self.pair_filter_specs,
            default_num_filters=num_filters,
            default_centers=centers,
            default_width=float(self.filter_width),
            default_kind=filter_kind,
        )
        pair_filter_execution_policy = str(
            self.pair_filter_execution_policy
        ).strip().lower()
        if pair_filter_execution_policy == "off":
            pair_filter_execution_policy = "reference"
        if pair_filter_execution_policy == "force":
            pair_filter_execution_policy = "vectorized"
        if pair_filter_execution_policy not in {
            "auto",
            "reference",
            "masked",
            "vectorized",
        }:
            raise ValueError(
                "LiftedDensityConfig.pair_filter_execution_policy must be "
                "'auto', 'reference', 'masked', or 'vectorized'."
            )
        inhomogeneous_pair_slot_filters = any(
            int(spec["num_filters"]) != int(num_filters)
            for spec in pair_filter_specs.values()
        )
        readout_mode = str(self.readout_mode).strip().lower()
        if readout_mode not in {
            "linear",
            "symmetric_linear",
            "antisymmetric_quadratic",
            "character_quadratic",
            "ye3_quadratic",
            "ye3_power",
            "ye3_slot_specht_power",
        }:
            raise ValueError(
                "LiftedDensityConfig.readout_mode must be 'linear', 'symmetric_linear', "
                "'antisymmetric_quadratic', 'character_quadratic', "
                "'ye3_quadratic', 'ye3_power', or 'ye3_slot_specht_power'."
            )
        if (
            inhomogeneous_pair_slot_filters
            and readout_mode == "ye3_power"
            and _normalize_ye3_slot_sectors(self.ye3_slot_sectors) == ("trivial",)
        ):
            inhomogeneous_pair_slot_filters = False
        if inhomogeneous_pair_slot_filters and readout_mode in {
            "character_quadratic",
            "ye3_power",
            "ye3_slot_specht_power",
        }:
            raise ValueError(
                "Pair-dependent active slot counts create an inhomogeneous pair-slot carrier. "
                f"LiftedDensityConfig.readout_mode={readout_mode!r} requires a homogeneous slot carrier; "
                "use the same num_filters for every pair, restrict to a slot-trivial readout, or implement "
                "a direct-sum pair-slot representation backend."
            )
        nonangular_scalar_readouts = {
            "linear",
            "symmetric_linear",
            "antisymmetric_quadratic",
            "character_quadratic",
        }
        if readout_mode in nonangular_scalar_readouts and any(int(channel.l) != 0 for channel in channels):
            raise ValueError(
                f"LiftedDensityConfig.readout_mode={readout_mode!r} only supports l=0 channels "
                "in the scalar-energy runtime. Use 'ye3_quadratic' or 'ye3_power' "
                "for l>0 channels with explicit L_R=0 angular contractions."
            )
        ye3_max_power = int(self.ye3_max_power)
        if ye3_max_power < 2:
            raise ValueError("LiftedDensityConfig.ye3_max_power must be at least 2.")
        ye3_optimization_policy = str(self.ye3_optimization_policy).strip()
        if ye3_optimization_policy not in {"off", "auto", "aggressive", "product_evaluator"}:
            raise ValueError(
                "LiftedDensityConfig.ye3_optimization_policy must be one of "
                "'off', 'auto', 'aggressive', or 'product_evaluator'."
            )
        ye3_slot_sectors = _normalize_ye3_slot_sectors(self.ye3_slot_sectors)
        ye3_slot_specht_partitions = _normalize_slot_specht_partitions(
            self.ye3_slot_specht_partitions,
            slot_count=num_filters,
        )
        ye3_slot_specht_coupling = str(self.ye3_slot_specht_coupling).strip().lower()
        if ye3_slot_specht_coupling not in {"projected_norm", "commutant_symmetric"}:
            raise ValueError(
                "LiftedDensityConfig.ye3_slot_specht_coupling must be "
                "'projected_norm' or 'commutant_symmetric'."
            )
        ye3_rank_nmax = _normalize_rank_limit_mapping(self.ye3_rank_nmax, name="ye3_rank_nmax")
        ye3_rank_lmax = _normalize_rank_limit_mapping(self.ye3_rank_lmax, name="ye3_rank_lmax")
        ye3_rank_lmin = _normalize_rank_limit_mapping(self.ye3_rank_lmin, name="ye3_rank_lmin")
        slot_group = str(self.slot_group).strip().lower()
        if slot_group not in {"symmetric", "identity", "young_subgroup"}:
            raise ValueError("LiftedDensityConfig.slot_group must be 'symmetric', 'identity', or 'young_subgroup'.")
        young_subgroup_blocks = _normalize_young_subgroup_blocks(self.young_subgroup_blocks, num_filters)
        if slot_group == "young_subgroup" and not young_subgroup_blocks:
            raise ValueError("LiftedDensityConfig.slot_group='young_subgroup' requires young_subgroup_blocks.")
        density_normalization = normalize_A_s_density_normalization_name(self.density_normalization)
        density_normalization_nugget = float(
            self.density_normalization_nugget
        )
        if (
            not np.isfinite(density_normalization_nugget)
            or density_normalization_nugget < 0.0
        ):
            raise ValueError(
                "LiftedDensityConfig.density_normalization_nugget must be "
                "finite and nonnegative."
            )
        feature_normalization = normalize_A_s_feature_normalization_name(self.feature_normalization)
        periodic_image_mode = str(self.periodic_image_mode).strip().lower()
        if periodic_image_mode in {"non_periodic", "none", "off", "molecule"}:
            periodic_image_mode = "nonperiodic"
        if periodic_image_mode not in {"unique", "all_images", "nonperiodic"}:
            raise ValueError(
                "LiftedDensityConfig.periodic_image_mode must be 'unique', 'all_images', or 'nonperiodic'."
            )
        if float(self.periodic_image_margin) < 0.0:
            raise ValueError("LiftedDensityConfig.periodic_image_margin must be nonnegative.")
        source_backend = str(self.source_backend).strip().lower()
        if source_backend not in {
            "auto",
            "torch",
            "native",
            "native_cpu",
            "native_cuda",
            "deterministic",
        }:
            raise ValueError(
                "LiftedDensityConfig.source_backend must be 'auto', "
                "'torch', 'native', 'native_cpu', 'native_cuda', or "
                "'deterministic'."
            )
        native_source_min_edges = int(self.native_source_min_edges)
        if native_source_min_edges < 0:
            raise ValueError(
                "LiftedDensityConfig.native_source_min_edges must be "
                "nonnegative."
            )
        object.__setattr__(self, "cutoff", float(self.cutoff))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "possible_types", possible_types)
        object.__setattr__(self, "filter_kind", filter_kind)
        object.__setattr__(self, "num_filters", num_filters)
        object.__setattr__(self, "filter_centers", centers)
        object.__setattr__(self, "filter_width", float(self.filter_width))
        object.__setattr__(self, "radial_lambda", radial_lambda)
        object.__setattr__(self, "pair_cutoffs", pair_cutoffs)
        object.__setattr__(self, "pair_radial_lambdas", pair_radial_lambdas)
        object.__setattr__(self, "pair_filter_specs", pair_filter_specs)
        object.__setattr__(
            self,
            "pair_filter_execution_policy",
            pair_filter_execution_policy,
        )
        object.__setattr__(self, "readout_mode", readout_mode)
        object.__setattr__(self, "slot_group", slot_group)
        object.__setattr__(self, "young_subgroup_blocks", young_subgroup_blocks)
        object.__setattr__(self, "density_normalization", density_normalization)
        object.__setattr__(
            self,
            "density_normalization_nugget",
            density_normalization_nugget,
        )
        object.__setattr__(self, "feature_normalization", feature_normalization)
        object.__setattr__(self, "hidden_layers", int(self.hidden_layers))
        object.__setattr__(self, "ye3_max_power", ye3_max_power)
        object.__setattr__(self, "ye3_optimization_policy", ye3_optimization_policy)
        object.__setattr__(self, "ye3_slot_sectors", ye3_slot_sectors)
        object.__setattr__(self, "ye3_slot_specht_partitions", ye3_slot_specht_partitions)
        object.__setattr__(self, "ye3_slot_specht_coupling", ye3_slot_specht_coupling)
        object.__setattr__(self, "ye3_include_rank1", bool(self.ye3_include_rank1))
        object.__setattr__(self, "ye3_rank_nmax", ye3_rank_nmax)
        object.__setattr__(self, "ye3_rank_lmax", ye3_rank_lmax)
        object.__setattr__(self, "ye3_rank_lmin", ye3_rank_lmin)
        object.__setattr__(self, "periodic_image_mode", periodic_image_mode)
        object.__setattr__(self, "enforce_unique_periodic_images", bool(self.enforce_unique_periodic_images))
        object.__setattr__(self, "periodic_image_margin", float(self.periodic_image_margin))
        object.__setattr__(self, "source_backend", source_backend)
        object.__setattr__(
            self,
            "native_source_min_edges",
            native_source_min_edges,
        )

    @property
    def has_pair_cutoffs(self):
        return bool(self.pair_cutoffs)

    @property
    def has_pair_radial_lambdas(self):
        return bool(self.pair_radial_lambdas)

    @property
    def has_pair_filter_specs(self):
        return bool(self.pair_filter_specs)

    @property
    def has_inhomogeneous_pair_slot_filters(self):
        return any(int(spec["num_filters"]) != int(self.num_filters) for spec in self.pair_filter_specs.values())

    @property
    def max_edge_cutoff(self):
        if not self.pair_cutoffs:
            return float(self.cutoff)
        return max(float(self.cutoff), max(float(value) for value in self.pair_cutoffs.values()))

    def ordered_pair_cutoffs(self, possible_types):
        if not self.pair_cutoffs:
            return ordered_pair_values(float(self.cutoff), possible_types=possible_types, name="pair_cutoffs")
        return [
            float(self.pair_cutoffs.get((int(left), int(right)), float(self.cutoff)))
            for left in possible_types
            for right in possible_types
        ]

    def ordered_pair_radial_lambdas(self, possible_types):
        if not self.pair_radial_lambdas:
            return ordered_pair_values(float(self.radial_lambda), possible_types=possible_types, name="pair_radial_lambdas")
        return [
            float(self.pair_radial_lambdas.get((int(left), int(right)), float(self.radial_lambda)))
            for left in possible_types
            for right in possible_types
        ]

    def to_dict(self):
        return {
            "cutoff": self.cutoff,
            "channels": [ch.to_dict() for ch in self.channels],
            "possible_types": list(self.possible_types),
            "filter_kind": self.filter_kind,
            "num_filters": self.num_filters,
            "filter_centers": list(self.filter_centers),
            "filter_width": self.filter_width,
            "radial_lambda": self.radial_lambda,
            "pair_cutoffs": {f"{left},{right}": float(value) for (left, right), value in self.pair_cutoffs.items()},
            "pair_radial_lambdas": {
                f"{left},{right}": float(value)
                for (left, right), value in self.pair_radial_lambdas.items()
            },
            "pair_filter_specs": {
                f"{left},{right}": {
                    "filter_kind": spec["filter_kind"],
                    "num_filters": int(spec["num_filters"]),
                    "filter_centers": list(spec["filter_centers"]),
                    "filter_width": float(spec["filter_width"]),
                }
                for (left, right), spec in self.pair_filter_specs.items()
            },
            "pair_filter_execution_policy": (
                self.pair_filter_execution_policy
            ),
            "readout_mode": self.readout_mode,
            "slot_group": self.slot_group,
            "young_subgroup_blocks": [list(block) for block in self.young_subgroup_blocks],
            "density_normalization": self.density_normalization,
            "density_normalization_nugget": (
                self.density_normalization_nugget
            ),
            "feature_normalization": self.feature_normalization,
            "hidden_layers": self.hidden_layers,
            "ye3_max_power": self.ye3_max_power,
            "ye3_optimization_policy": self.ye3_optimization_policy,
            "ye3_slot_sectors": list(self.ye3_slot_sectors),
            "ye3_slot_specht_partitions": [list(partition) for partition in self.ye3_slot_specht_partitions],
            "ye3_slot_specht_coupling": self.ye3_slot_specht_coupling,
            "ye3_include_rank1": self.ye3_include_rank1,
            "ye3_rank_nmax": dict(self.ye3_rank_nmax),
            "ye3_rank_lmax": dict(self.ye3_rank_lmax),
            "ye3_rank_lmin": dict(self.ye3_rank_lmin),
            "periodic_image_mode": self.periodic_image_mode,
            "enforce_unique_periodic_images": self.enforce_unique_periodic_images,
            "periodic_image_margin": self.periodic_image_margin,
            "source_backend": self.source_backend,
            "native_source_min_edges": self.native_source_min_edges,
        }

    @classmethod
    def from_dict(cls, payload):
        def _channel_from_config(item):
            if isinstance(item, LiftedDensityChannel):
                return item
            return LiftedDensityChannel.from_dict(item)

        return cls(
            cutoff=float(payload.get("cutoff", 4.0)),
            channels=tuple(
                _channel_from_config(item)
                for item in payload.get("channels", ({"n": 1, "l": 0, "m": 0},))
            ),
            possible_types=tuple(
                int(value)
                for value in payload.get("possible_types", ())
            ),
            filter_kind=str(payload.get("filter_kind", "softmax_gaussian")),
            num_filters=int(payload.get("num_filters", 3)),
            filter_centers=tuple(float(x) for x in payload.get("filter_centers", ())),
            filter_width=float(payload.get("filter_width", 0.25)),
            radial_lambda=float(payload.get("radial_lambda", 0.25)),
            pair_cutoffs=payload.get("pair_cutoffs", payload.get("cutoff_by_pair", None)),
            pair_radial_lambdas=payload.get(
                "pair_radial_lambdas",
                payload.get("radial_lambda_by_pair", None),
            ),
            pair_filter_specs=payload.get("pair_filter_specs", payload.get("slot_filters_by_pair", None)),
            pair_filter_execution_policy=str(
                payload.get("pair_filter_execution_policy", "auto")
            ),
            readout_mode=str(payload.get("readout_mode", "linear")),
            slot_group=str(payload.get("slot_group", "symmetric")),
            young_subgroup_blocks=tuple(tuple(int(slot) for slot in block) for block in payload.get("young_subgroup_blocks", ())),
            density_normalization=str(
                payload.get(
                    "density_normalization",
                    payload.get("A_s_density_normalization", DEFAULT_A_S_DENSITY_NORMALIZATION),
                )
            ),
            density_normalization_nugget=float(
                payload.get("density_normalization_nugget", 0.0)
            ),
            feature_normalization=str(
                payload.get(
                    "feature_normalization",
                    payload.get("A_s_feature_normalization", DEFAULT_A_S_FEATURE_NORMALIZATION),
                )
            ),
            hidden_layers=int(payload.get("hidden_layers", 1)),
            ye3_max_power=int(payload.get("ye3_max_power", 4)),
            ye3_optimization_policy=str(payload.get("ye3_optimization_policy", "auto")),
            ye3_slot_sectors=tuple(payload.get("ye3_slot_sectors", ("trivial", "standard"))),
            ye3_slot_specht_partitions=payload.get("ye3_slot_specht_partitions", None),
            ye3_slot_specht_coupling=str(payload.get("ye3_slot_specht_coupling", "projected_norm")),
            ye3_include_rank1=bool(payload.get("ye3_include_rank1", False)),
            ye3_rank_nmax=payload.get("ye3_rank_nmax", None),
            ye3_rank_lmax=payload.get("ye3_rank_lmax", None),
            ye3_rank_lmin=payload.get("ye3_rank_lmin", None),
            periodic_image_mode=str(payload.get("periodic_image_mode", "unique")),
            enforce_unique_periodic_images=bool(payload.get("enforce_unique_periodic_images", True)),
            periodic_image_margin=float(payload.get("periodic_image_margin", 1.0e-8)),
            source_backend=str(payload.get("source_backend", "auto")),
            native_source_min_edges=int(
                payload.get("native_source_min_edges", 0)
            ),
        )


@recordclass(('branches', 'lifted_density', 'dtype'), frozen = True)
class HybridACELiftedDensityConfig:
    """Configuration for exact ``A`` plus filtered/lifted ``A_s`` branches."""

    branches = (BRANCH_LIFTED_DENSITY,)
    lifted_density = field(default_factory=LiftedDensityConfig)
    dtype = "float64"

    def __post_init__(self):
        object.__setattr__(self, "branches", normalize_lifted_density_branches(self.branches))
        lifted = (
            self.lifted_density
            if isinstance(self.lifted_density, LiftedDensityConfig)
            else LiftedDensityConfig.from_dict(self.lifted_density)
        )
        object.__setattr__(self, "lifted_density", lifted)
        dtype = str(self.dtype)
        if dtype not in {"float32", "float64"}:
            raise ValueError("HybridACELiftedDensityConfig.dtype must be 'float32' or 'float64'.")
        object.__setattr__(self, "dtype", dtype)

    @property
    def torch_dtype(self):
        return torch.float32 if self.dtype == "float32" else torch.float64

    def to_dict(self):
        return {
            "branches": list(self.branches),
            "lifted_density": self.lifted_density.to_dict(),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            branches=tuple(payload.get("branches", (BRANCH_LIFTED_DENSITY,))),
            lifted_density=LiftedDensityConfig.from_dict(payload.get("lifted_density", {})),
            dtype=str(payload.get("dtype", "float64")),
        )


class HybridACELiftedDensityEnergyModel(torch.nn.Module):
    """Conservative scalar model using role-resolved radial channel maps.

    The source obeys
    ``A_{i,s,eta,l,m} = sum_eta' K^{(l)}_{i,s;eta,eta'}
    A_{i,eta',l,m}``.  The current filter tensors implement a diagonal,
    pair-dependent realization of ``K`` while preserving the explicit role
    coordinate needed by nontrivial Young sectors.

    The direct tensor model intentionally does not implement the exact ``A``
    branch.  When ``A`` is requested, use ``HybridACELiftedDensityCalculator``
    with an exact linear ACE bundle.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, HybridACELiftedDensityConfig) else HybridACELiftedDensityConfig()
        lifted = self.config.lifted_density
        self.branches = self.config.branches
        self._site_basis_cache = {}
        self._direct_lifted_channel_schedule_cache = {}
        dtype = self.config.torch_dtype
        self.linear_weight = torch.nn.Parameter(torch.zeros(lifted.num_filters, len(lifted.channels), dtype=dtype))
        self.linear_bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.character_mu_weight = torch.nn.Parameter(torch.zeros(len(lifted.channels), dtype=dtype))
        self.character_nu_weight = torch.nn.Parameter(torch.zeros(len(lifted.channels), dtype=dtype))
        self.character_bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.antisymmetric_weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.antisymmetric_bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.ye3_blocks = lifted_density_channel_blocks(lifted.channels)
        self.ye3_weight = torch.nn.Parameter(torch.zeros(2 * len(self.ye3_blocks), dtype=dtype))
        self.ye3_bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.ye3_power_sectors = ye3_power_sector_metadata(
            lifted.channels,
            max_power=lifted.ye3_max_power,
            slot_sectors=lifted.ye3_slot_sectors,
            include_rank1=lifted.ye3_include_rank1,
            rank_nmax=lifted.ye3_rank_nmax,
            rank_lmax=lifted.ye3_rank_lmax,
            rank_lmin=lifted.ye3_rank_lmin,
        )
        power_feature_count = sum(int(sector["multiplicity"]) for sector in self.ye3_power_sectors)
        self.ye3_power_weight = torch.nn.Parameter(torch.zeros(power_feature_count, dtype=dtype))
        self.ye3_power_bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.ye3_slot_specht_power_sectors = ye3_slot_specht_power_sector_metadata(
            lifted.channels,
            num_slots=lifted.num_filters,
            max_power=lifted.ye3_max_power,
            slot_specht_partitions=lifted.ye3_slot_specht_partitions,
            include_rank1=lifted.ye3_include_rank1,
            rank_nmax=lifted.ye3_rank_nmax,
            rank_lmax=lifted.ye3_rank_lmax,
            rank_lmin=lifted.ye3_rank_lmin,
            slot_specht_coupling=lifted.ye3_slot_specht_coupling,
        )
        slot_specht_feature_count = sum(int(sector["multiplicity"]) for sector in self.ye3_slot_specht_power_sectors)
        self.ye3_slot_specht_power_weight = torch.nn.Parameter(
            torch.zeros(slot_specht_feature_count, dtype=dtype)
        )
        self.ye3_slot_specht_power_bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.channel_readout = torch.nn.Parameter(torch.zeros(len(lifted.channels), dtype=dtype))
        self.slot_equivariant_bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self._last_profile = {}
        self._last_role_density_backend = None
        self._last_role_density_operation = None
        self._last_lifted_site_basis_backend = None
        self._last_lifted_site_basis_operation = None
        self._last_lifted_shared_edge_geometry = False
        self._role_filter_centers_cache = {}
        self._pair_softmax_filter_table_cache = {}
        self._pair_filter_execution_policy = str(
            self.config.lifted_density.pair_filter_execution_policy
        )
        self._last_pair_filter_operation = None

    def _role_density_accumulate(self, edge_values, centers, atom_count):
        """Accumulate edge values while preserving an explicit role axis."""

        if edge_values.ndim not in {2, 3}:
            raise ValueError(
                "role density edge values must have shape [edges, width] "
                "or [edges, roles, channels]"
            )
        original_shape = tuple(edge_values.shape[1:])
        packed_width = 1
        for width in original_shape:
            packed_width *= int(width)
        packed = edge_values.reshape(
            int(edge_values.shape[0]),
            int(packed_width),
        )
        lifted = self.config.lifted_density
        requested = str(lifted.source_backend)
        runtime_backend = _source_kernel_backend(
            requested,
            packed,
            lifted.native_source_min_edges,
            cuda_operation="density_accumulate",
        )
        if runtime_backend != "reference":
            from ye3t.runtime import density_accumulate

            output = density_accumulate(
                packed,
                centers,
                int(atom_count),
                backend=runtime_backend,
            )
            native_label = _native_density_backend_label(
                runtime_backend,
                packed,
                packed.numel(),
            )
            self._last_role_density_backend = native_label or (
                "torch_cuda" if packed.is_cuda else "torch"
            )
        else:
            output = packed.new_zeros(
                (int(atom_count), int(packed.shape[1]))
            )
            output.index_add_(0, centers, packed)
            self._last_role_density_backend = (
                "torch_cuda" if packed.is_cuda else "torch"
            )
        self._last_role_density_operation = "density_accumulate"
        return output.reshape((int(atom_count),) + original_shape)

    def _role_density_outer_accumulate(
        self,
        filters,
        edge_values,
        soft_weights,
        centers,
        atom_count,
    ):
        """Apply the current diagonal ``K`` map and accumulate by center.

        For edge ``e=(i,j)``, ``filters[e,s]`` is the implemented matrix
        element of ``K^{(l)}_{i,s}`` for the compatible flattened channel, so
        the first output is
        ``A_raw[i,s,c] = sum_{e:src(e)=i} filters[e,s] edge_values[e,c]``.
        The second output applies the same map to cutoff weights and supplies
        the exact slot-resolved normalization denominator.
        """

        if filters.ndim != 2 or edge_values.ndim != 2:
            raise ValueError(
                "filters and edge_values must be two-dimensional"
            )
        if (
            int(filters.shape[0]) != int(edge_values.shape[0])
            or int(soft_weights.numel()) != int(filters.shape[0])
        ):
            raise ValueError(
                "lifted source factors must share one edge axis"
            )
        packed_right = torch.cat(
            (
                edge_values,
                soft_weights.reshape(-1, 1),
            ),
            dim=1,
        )
        lifted = self.config.lifted_density
        runtime_backend = _source_kernel_backend(
            lifted.source_backend,
            filters,
            lifted.native_source_min_edges,
            cuda_operation="edge_outer_accumulate",
        )
        from ye3t.runtime import edge_outer_accumulate

        packed = edge_outer_accumulate(
            filters,
            packed_right,
            centers,
            int(atom_count),
            backend=runtime_backend,
        )
        native_label = _native_edge_outer_backend_label(
            runtime_backend,
            filters,
            int(filters.numel() * packed_right.shape[1]),
        )
        self._last_role_density_backend = native_label or (
            "torch_cuda" if filters.is_cuda else "torch"
        )
        self._last_role_density_operation = "edge_outer_accumulate"
        return packed[..., :-1], packed[..., -1]

    def _softmax_gaussian_role_density_accumulate(
        self,
        dist,
        pair_cutoffs,
        edge_values,
        soft_weights,
        centers,
        atom_count,
    ):
        """Use the fused fixed-role source operation when its contract applies."""

        lifted = self.config.lifted_density
        if (
            str(lifted.filter_kind) != "softmax_gaussian"
            or bool(lifted.pair_filter_specs)
        ):
            return None
        runtime_backend = _source_kernel_backend(
            lifted.source_backend,
            dist,
            lifted.native_source_min_edges,
            cuda_operation="softmax_gaussian_role_density",
        )
        if runtime_backend == "reference":
            return None
        from ye3t.runtime import softmax_gaussian_role_density

        if runtime_backend != "native":
            from ye3t.runtime import native_execution_plan_capabilities

            capabilities = native_execution_plan_capabilities()
            operations = capabilities[
                (
                    "cuda_operations"
                    if dist.device.type == "cuda"
                    else "operations"
                )
            ]
            if "softmax_gaussian_role_density" not in operations:
                return None
        cache_key = (dist.device.type, dist.device.index, dist.dtype)
        filter_centers = self._role_filter_centers_cache.get(cache_key)
        if filter_centers is None:
            filter_centers = torch.tensor(
                lifted.filter_centers,
                dtype=dist.dtype,
                device=dist.device,
            )
            self._role_filter_centers_cache[cache_key] = filter_centers
        packed_right = torch.cat(
            (edge_values, soft_weights.reshape(-1, 1)),
            dim=1,
        )
        packed = softmax_gaussian_role_density(
            dist,
            pair_cutoffs,
            filter_centers,
            float(lifted.filter_width),
            packed_right,
            centers,
            int(atom_count),
            backend=runtime_backend,
        )
        if torch.compiler.is_compiling():
            return packed[..., :-1], packed[..., -1]
        self._last_role_density_backend = (
            "native_softmax_gaussian_role_density_cuda"
            if dist.device.type == "cuda"
            else "native_softmax_gaussian_role_density_cpu"
        )
        self._last_role_density_operation = (
            "softmax_gaussian_role_density"
        )
        return packed[..., :-1], packed[..., -1]

    def _scheduled_softmax_gaussian_role_density_accumulate(
        self,
        atom_types,
        src,
        dst,
        disp,
        dist,
        pair_cutoffs,
        atom_count,
    ):
        """Fuse scheduled direct channels with compatible role accumulation."""

        lifted = self.config.lifted_density
        if (
            str(lifted.filter_kind) != "softmax_gaussian"
            or bool(lifted.pair_filter_specs)
            or int(src.numel()) == 0
        ):
            return None
        runtime_backend = _source_kernel_backend(
            lifted.source_backend,
            dist,
            lifted.native_source_min_edges,
            cuda_operation=(
                "scheduled_softmax_gaussian_role_density"
            ),
        )
        if runtime_backend == "reference":
            return None
        direct = self._direct_lifted_site_basis_values_with_dx(
            atom_types,
            src,
            dst,
            disp,
            distance=dist,
            return_schedule_tables=True,
        )
        if direct is None:
            return None
        (
            radial_table,
            angular_table,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
            basis,
            bond_idx,
            distance,
            radial_backend,
            angular_backend,
            shared_edge_geometry,
        ) = direct
        soft_weights = basis._soft_neighbor_weights(
            bond_idx=bond_idx,
            r=distance,
        ).to(dtype=disp.dtype)
        cache_key = (dist.device.type, dist.device.index, dist.dtype)
        filter_centers = self._role_filter_centers_cache.get(cache_key)
        if filter_centers is None:
            filter_centers = torch.tensor(
                lifted.filter_centers,
                dtype=dist.dtype,
                device=dist.device,
            )
            self._role_filter_centers_cache[cache_key] = filter_centers
        from ye3t.runtime import (
            scheduled_softmax_gaussian_role_density,
        )

        packed = scheduled_softmax_gaussian_role_density(
            radial_table,
            angular_table,
            distance,
            pair_cutoffs,
            filter_centers,
            float(lifted.filter_width),
            soft_weights,
            edge_types,
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
            src,
            int(atom_count),
            backend=runtime_backend,
        )
        if torch.compiler.is_compiling():
            return packed[..., :-1], packed[..., -1]
        basis._last_radial_table_backend = (
            "native_cuda"
            if radial_backend != "reference"
            and disp.device.type == "cuda"
            else (
                "native"
                if radial_backend != "reference"
                else "reference"
            )
        )
        basis._last_angular_table_backend = (
            "native_cuda"
            if angular_backend != "reference"
            and disp.device.type == "cuda"
            else (
                "native"
                if angular_backend != "reference"
                else "reference"
            )
        )
        self._last_role_density_backend = (
            "native_scheduled_softmax_gaussian_role_density_cuda"
            if dist.device.type == "cuda"
            else "native_scheduled_softmax_gaussian_role_density_cpu"
        )
        self._last_role_density_operation = (
            "scheduled_softmax_gaussian_role_density"
        )
        self._last_lifted_site_basis_backend = (
            "direct_scheduled_native_cuda"
            if disp.device.type == "cuda"
            else "direct_scheduled_native_cpu"
        )
        self._last_lifted_site_basis_operation = (
            "scheduled_softmax_gaussian_role_density"
        )
        self._last_lifted_shared_edge_geometry = bool(
            shared_edge_geometry
        )
        return packed[..., :-1], packed[..., -1]

    def source_runtime_report(self):
        lifted = self.config.lifted_density
        return {
            "source_realization": "lifted_density_roles",
            "requested_backend": str(lifted.source_backend),
            "native_source_min_edges": int(
                lifted.native_source_min_edges
            ),
            "role_density_accumulation_backend": (
                self._last_role_density_backend
            ),
            "role_density_accumulation_operation": (
                self._last_role_density_operation
            ),
            "site_basis_channel_backend": (
                self._last_lifted_site_basis_backend
            ),
            "site_basis_channel_operation": (
                self._last_lifted_site_basis_operation
            ),
            "shared_edge_geometry": bool(
                self._last_lifted_shared_edge_geometry
            ),
            "pair_filter_operation": self._last_pair_filter_operation,
            "pair_filter_execution_policy": str(
                self._pair_filter_execution_policy
            ),
            "pair_filter_complete_type_coverage": bool(
                self._pair_filter_specs_cover_possible_types()
            ),
            "role_axis_retained": True,
            "role_count": int(lifted.num_filters),
            "materialization_owner": "ye3t-ace",
        }

    def _validate_periodic_cutoff_margin(self, cell, pbc):
        lifted = self.config.lifted_density
        if lifted.periodic_image_mode in {"all_images", "nonperiodic"} or cell is None or not bool(torch.any(pbc)):
            return float("inf")
        margin = _unique_periodic_cutoff_margin(cell.detach().cpu(), pbc.detach().cpu(), lifted.max_edge_cutoff)
        if bool(lifted.enforce_unique_periodic_images) and margin < float(lifted.periodic_image_margin):
            raise ValueError(
                "Unique-image lifted-density evaluation requires a unique minimum image for active distances. "
                f"Got half-shortest-lattice cutoff margin {margin:.6g}, below configured margin "
                f"{lifted.periodic_image_margin:.6g}. Use all_images mode, reduce cutoff, or enlarge the cell."
            )
        return margin

    def _site_basis_for_types(self, possible_types, device):
        lifted = self.config.lifted_density
        possible_types = tuple(sorted(int(x) for x in possible_types))
        dtype = self.config.torch_dtype
        rc_values = tuple(float(x) for x in lifted.ordered_pair_cutoffs(possible_types))
        lmbda_values = tuple(float(x) for x in lifted.ordered_pair_radial_lambdas(possible_types))
        key = (possible_types, str(dtype), str(device), rc_values, lmbda_values)
        cached = self._site_basis_cache.get(key)
        if cached is not None:
            return cached
        max_n = max((int(ch.n) for ch in lifted.channels), default=1)
        max_l = max((int(ch.l) for ch in lifted.channels), default=0)
        cfg = SiteBasisConfig(
            rc=list(rc_values),
            lmbda=list(lmbda_values),
            nradmax=max(1, max_n),
            lmax=max_l,
            possible_types=possible_types,
            charge_mode="none",
            atomic_base_normalization="none",
            factor_normalization="bounded",
            spherical_backend="real",
            dtype=dtype,
            complex_dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
        )
        basis = SiteBasisV2(cfg).to(device=device)
        self._site_basis_cache[key] = basis
        return basis

    def _possible_types_for_atom_types(self, atom_types):
        lifted = self.config.lifted_density
        if lifted.possible_types:
            return tuple(lifted.possible_types)
        type_values = set(
            int(value)
            for value in atom_types.detach().cpu().tolist()
        )
        for channel in lifted.channels:
            if channel.neighbor_type is not None:
                type_values.add(int(channel.neighbor_type))
        return tuple(sorted(type_values or {0}))

    def _direct_lifted_channel_schedule(self, basis, device, dtype):
        lifted = self.config.lifted_density
        key = (
            id(basis),
            str(device),
            str(dtype),
            tuple(
                (
                    int(channel.n),
                    int(channel.l),
                    int(channel.m),
                    None
                    if channel.neighbor_type is None
                    else int(channel.neighbor_type),
                )
                for channel in lifted.channels
            ),
        )
        cached = self._direct_lifted_channel_schedule_cache.get(key)
        if cached is not None:
            return cached
        radial_indices = torch.tensor(
            [int(channel.n) for channel in lifted.channels],
            dtype=torch.long,
            device=device,
        )
        angular_indices = torch.tensor(
            [
                int(channel.l) * int(channel.l)
                + int(channel.m)
                + int(channel.l)
                for channel in lifted.channels
            ],
            dtype=torch.long,
            device=device,
        )
        channel_types = torch.tensor(
            [
                -1
                if channel.neighbor_type is None
                else int(channel.neighbor_type)
                for channel in lifted.channels
            ],
            dtype=torch.long,
            device=device,
        )
        scales = []
        for channel in lifted.channels:
            radial_scale = basis._radial_factor_scale(
                int(channel.n),
                int(channel.l),
                device=device,
                dtype=dtype,
            )
            angular_scale = basis._spherical_factor_scale(
                int(channel.l),
                device=device,
                dtype=dtype,
            )
            scales.append(
                radial_scale.to(dtype) * angular_scale.to(dtype)
            )
        channel_scales = torch.stack(scales)
        cached = (
            radial_indices,
            angular_indices,
            channel_types,
            channel_scales,
        )
        self._direct_lifted_channel_schedule_cache[key] = cached
        return cached

    def _direct_lifted_site_basis_values_with_dx(
        self,
        atom_types,
        src,
        dst,
        disp,
        distance=None,
        return_schedule_tables=False,
    ):
        lifted = self.config.lifted_density
        shared_edge_geometry = distance is not None
        if (
            str(lifted.source_backend) == "torch"
            or not lifted.channels
            or int(src.numel()) == 0
        ):
            return None
        possible_types = self._possible_types_for_atom_types(
            atom_types
        )
        basis = self._site_basis_for_types(
            possible_types,
            disp.device,
        )
        radial_kind = str(basis.cfg.radial_basis).strip().lower()
        radial_kind = radial_kind.replace("_", "").replace("-", "")
        if (
            radial_kind != "chebexpcos"
            or str(basis.cfg.spherical_backend) != "real"
            or any(
                int(channel.m) < -int(channel.l)
                or int(channel.m) > int(channel.l)
                for channel in lifted.channels
            )
        ):
            return None

        atom_types = atom_types.to(
            device=disp.device,
            dtype=torch.long,
        )
        centers = src.to(device=disp.device, dtype=torch.long)
        neighbors = dst.to(device=disp.device, dtype=torch.long)
        bond_idx = basis._bond_index(
            atom_types[centers],
            atom_types[neighbors],
        )
        if distance is None:
            distance = torch.linalg.vector_norm(disp, dim=1)
        else:
            distance = distance.to(
                device=disp.device,
                dtype=disp.dtype,
            )
        safe_distance = torch.clamp(
            distance,
            min=torch.as_tensor(
                1.0e-12,
                dtype=disp.dtype,
                device=disp.device,
            ),
        )
        radial_directions = disp / safe_distance.unsqueeze(-1)
        maximum_n = max(
            int(channel.n) for channel in lifted.channels
        )
        maximum_l = max(
            int(channel.l) for channel in lifted.channels
        )

        from ye3t.runtime import (
            cheb_exp_cos_radial_table_with_derivative,
            scheduled_radial_angular_channels_with_derivative,
            spherical_harmonics_table_with_derivative,
        )

        radial_backend = _source_kernel_backend(
            lifted.source_backend,
            distance,
            lifted.native_source_min_edges,
            cuda_operation=(
                "cheb_exp_cos_radial_table_with_derivative"
            ),
        )
        cutoffs, lambdas = basis._prepare_cutoffs_on_device(
            disp.device
        )
        radial_table, radial_derivative_table = (
            cheb_exp_cos_radial_table_with_derivative(
                distance,
                cutoffs[bond_idx],
                lambdas[bond_idx],
                maximum_n,
                backend=radial_backend,
            )
        )
        angular_backend = _source_kernel_backend(
            lifted.source_backend,
            disp,
            lifted.native_source_min_edges,
            cuda_operation=(
                "spherical_harmonics_table_with_derivative"
            ),
        )
        angular_table, angular_derivative_table = (
            spherical_harmonics_table_with_derivative(
                disp,
                maximum_l,
                real_output=True,
                backend=angular_backend,
            )
        )

        (
            channel_radial_indices,
            channel_angular_indices,
            channel_types,
            channel_scales,
        ) = self._direct_lifted_channel_schedule(
            basis,
            disp.device,
            disp.dtype,
        )
        if return_schedule_tables:
            return (
                radial_table,
                angular_table,
                atom_types[neighbors],
                channel_radial_indices,
                channel_angular_indices,
                channel_types,
                channel_scales,
                basis,
                bond_idx,
                distance,
                radial_backend,
                angular_backend,
                shared_edge_geometry,
            )
        product_backend = _source_kernel_backend(
            lifted.source_backend,
            disp,
            lifted.native_source_min_edges,
            cuda_operation=(
                "scheduled_radial_angular_channels_with_derivative"
            ),
        )
        values, values_dx = (
            scheduled_radial_angular_channels_with_derivative(
                radial_table,
                radial_derivative_table,
                angular_table,
                angular_derivative_table,
                radial_directions,
                atom_types[neighbors],
                channel_radial_indices,
                channel_angular_indices,
                channel_types,
                channel_scales,
                backend=product_backend,
            )
        )
        if (
            radial_backend == "native"
            and angular_backend == "native"
            and product_backend == "native"
        ):
            radial_native = True
            angular_native = True
            product_native = True
        else:
            from ye3t.runtime import native_execution_plan_capabilities

            capabilities = native_execution_plan_capabilities()
            native_operations = capabilities[
                (
                    "cuda_operations"
                    if disp.device.type == "cuda"
                    else "operations"
                )
            ]
            radial_native = bool(
                radial_backend == "native"
                or (
                    radial_backend == "auto"
                    and capabilities[disp.device.type]
                    and (
                        "cheb_exp_cos_radial_table_with_derivative"
                        in native_operations
                    )
                )
            )
            angular_native = bool(
                angular_backend == "native"
                or (
                    angular_backend == "auto"
                    and capabilities[disp.device.type]
                    and (
                        "spherical_harmonics_table_with_derivative"
                        in native_operations
                    )
                )
            )
            product_native = bool(
                product_backend == "native"
                or (
                    product_backend == "auto"
                    and capabilities[disp.device.type]
                    and (
                        "scheduled_radial_angular_channels_with_derivative"
                        in native_operations
                    )
                )
            )
        if torch.compiler.is_compiling():
            return (
                values,
                values_dx,
                basis,
                bond_idx,
                distance,
                radial_directions,
            )
        basis._last_radial_table_backend = (
            "native_cuda"
            if radial_native
            and disp.device.type == "cuda"
            else (
                "native"
                if radial_native
                else "reference"
            )
        )
        basis._last_angular_table_backend = (
            "native_cuda"
            if angular_native
            and disp.device.type == "cuda"
            else (
                "native"
                if angular_native
                else "reference"
            )
        )
        basis._last_plain_product_backend = (
            "native_cuda"
            if product_native
            and disp.device.type == "cuda"
            else (
                "native_cpu"
                if product_native
                else "reference"
            )
        )
        self._last_lifted_site_basis_backend = (
            "direct_packed_native_cuda"
            if product_native
            and disp.device.type == "cuda"
            else (
                "direct_packed_native_cpu"
                if product_native
                else "direct_packed_reference"
            )
        )
        self._last_lifted_site_basis_operation = (
            "scheduled_radial_angular_channels_with_derivative"
        )
        self._last_lifted_shared_edge_geometry = bool(
            shared_edge_geometry
        )
        return (
            values,
            values_dx,
            basis,
            bond_idx,
            distance,
            radial_directions,
        )

    def _site_basis_edge_values_reference(self, atom_types, src, dst, disp, *, return_soft_weights=False, n_atoms=None):
        lifted = self.config.lifted_density
        values = disp.new_zeros((int(src.numel()), len(lifted.channels)))
        if src.numel() == 0 or not lifted.channels:
            if return_soft_weights:
                return values, disp.new_zeros((int(src.numel()),))
            return values
        possible_types = self._possible_types_for_atom_types(
            atom_types
        )
        basis = self._site_basis_for_types(possible_types, disp.device)
        site_channels = []
        reduced_channel_indices = []
        for channel_index, channel in enumerate(lifted.channels):
            neighbor_types = possible_types if channel.neighbor_type is None else (int(channel.neighbor_type),)
            for mu0 in possible_types:
                for mu in neighbor_types:
                    site_channels.append(
                        SingleChannelLabel(
                            mu0=int(mu0),
                            mu=int(mu),
                            kappa0=0,
                            kappa=0,
                            n=int(channel.n),
                            l=int(channel.l),
                            m=int(channel.m),
                        )
                    )
                    reduced_channel_indices.append(int(channel_index))
        edge_index = torch.stack((src, dst), dim=0)
        soft_weights = None
        if return_soft_weights:
            soft_weights = basis.compute_soft_neighbor_edge_weights(
                disp,
                edge_index,
                atom_types.to(device=disp.device, dtype=torch.long),
            ).to(dtype=disp.dtype)
        _labels, edge_values, _edge_dx = basis.compute_channel_edges_with_dx(
            disp,
            edge_index,
            atom_types,
            site_channels,
            real_output=True,
        )
        edge_values = edge_values.to(dtype=disp.dtype)
        edge_dx = _edge_dx.to(dtype=disp.dtype)
        reduced_channel_indices = torch.tensor(
            reduced_channel_indices,
            dtype=torch.long,
            device=disp.device,
        )
        values = edge_values.new_zeros(
            (int(src.numel()), len(lifted.channels))
        )
        values.index_add_(
            1,
            reduced_channel_indices,
            edge_values,
        )
        values_dx = edge_dx.new_zeros(
            (int(src.numel()), len(lifted.channels), 3)
        )
        values_dx.index_add_(
            1,
            reduced_channel_indices,
            edge_dx,
        )
        values = _edge_values_with_analytic_dx(disp, values, values_dx)
        if return_soft_weights:
            if soft_weights is None:
                soft_weights = disp.new_zeros((int(src.numel()),))
            return values, soft_weights
        return values

    def _site_basis_edge_values(
        self,
        atom_types,
        src,
        dst,
        disp,
        *,
        return_soft_weights=False,
        n_atoms=None,
        distance=None,
    ):
        direct = self._direct_lifted_site_basis_values_with_dx(
            atom_types,
            src,
            dst,
            disp,
            distance=distance,
        )
        if direct is None:
            self._last_lifted_site_basis_backend = (
                "expanded_site_basis_reference"
            )
            self._last_lifted_site_basis_operation = (
                "expanded_site_basis"
            )
            return self._site_basis_edge_values_reference(
                atom_types,
                src,
                dst,
                disp,
                return_soft_weights=return_soft_weights,
                n_atoms=n_atoms,
            )
        (
            values,
            values_dx,
            basis,
            bond_idx,
            distance,
            _radial_directions,
        ) = direct
        values = _edge_values_with_analytic_dx(
            disp,
            values,
            values_dx,
        )
        if not return_soft_weights:
            return values
        soft_weights = basis._soft_neighbor_weights(
            bond_idx=bond_idx,
            r=distance,
        ).to(dtype=disp.dtype)
        return values, soft_weights

    def _site_basis_edge_values_with_dx_reference(self, atom_types, src, dst, disp):
        """Return edge channel values, ``dv/dx_ij``, soft weights, and ``dw/dx_ij``.

        This exposes the compact edge derivative needed by the A_s density VJP.
        It follows the same channel construction as ``_site_basis_edge_values``.
        """

        lifted = self.config.lifted_density
        values = disp.new_zeros((int(src.numel()), len(lifted.channels)))
        values_dx = disp.new_zeros((int(src.numel()), len(lifted.channels), 3))
        soft_weights = disp.new_zeros((int(src.numel()),))
        soft_weights_dx = disp.new_zeros((int(src.numel()), 3))
        if src.numel() == 0 or not lifted.channels:
            return values, values_dx, soft_weights, soft_weights_dx
        possible_types = self._possible_types_for_atom_types(
            atom_types
        )
        basis = self._site_basis_for_types(possible_types, disp.device)
        site_channels = []
        column_slices = []
        for channel in lifted.channels:
            start = len(site_channels)
            neighbor_types = possible_types if channel.neighbor_type is None else (int(channel.neighbor_type),)
            for mu0 in possible_types:
                for mu in neighbor_types:
                    site_channels.append(
                        SingleChannelLabel(
                            mu0=int(mu0),
                            mu=int(mu),
                            kappa0=0,
                            kappa=0,
                            n=int(channel.n),
                            l=int(channel.l),
                            m=int(channel.m),
                        )
                    )
            column_slices.append(slice(start, len(site_channels)))
        edge_index = torch.stack((src, dst), dim=0)
        _labels, edge_values, edge_dx = basis.compute_channel_edges_with_dx(
            disp,
            edge_index,
            atom_types,
            site_channels,
        )
        edge_values = edge_values.real.to(dtype=disp.dtype)
        edge_dx = edge_dx.real.to(dtype=disp.dtype)
        value_parts = []
        dx_parts = []
        for column_slice in column_slices:
            value_parts.append(edge_values[:, column_slice].sum(dim=1))
            dx_parts.append(edge_dx[:, column_slice, :].sum(dim=1))
        values = torch.stack(value_parts, dim=1) if value_parts else values
        values_dx = torch.stack(dx_parts, dim=1) if dx_parts else values_dx
        centers = src.to(device=disp.device, dtype=torch.long)
        neighs = dst.to(device=disp.device, dtype=torch.long)
        bond_idx = basis._bond_index(atom_types[centers], atom_types[neighs])
        dist = torch.linalg.norm(disp, dim=1)
        safe_dist = torch.clamp(dist, min=torch.as_tensor(1.0e-12, dtype=disp.dtype, device=disp.device))
        dr_dx = disp / safe_dist.unsqueeze(-1)
        soft_weights, soft_weights_dx = basis._soft_neighbor_weights_with_dx(
            bond_idx=bond_idx,
            r=dist,
            dr_dx=dr_dx,
        )
        return values, values_dx, soft_weights.to(dtype=disp.dtype), soft_weights_dx.to(dtype=disp.dtype)

    def _site_basis_edge_values_with_dx(
        self,
        atom_types,
        src,
        dst,
        disp,
        distance=None,
    ):
        direct = self._direct_lifted_site_basis_values_with_dx(
            atom_types,
            src,
            dst,
            disp,
            distance=distance,
        )
        if direct is None:
            self._last_lifted_site_basis_backend = (
                "expanded_site_basis_reference"
            )
            self._last_lifted_site_basis_operation = (
                "expanded_site_basis"
            )
            return self._site_basis_edge_values_with_dx_reference(
                atom_types,
                src,
                dst,
                disp,
            )
        (
            values,
            values_dx,
            basis,
            bond_idx,
            distance,
            radial_directions,
        ) = direct
        soft_weights, soft_weights_dx = (
            basis._soft_neighbor_weights_with_dx(
                bond_idx=bond_idx,
                r=distance,
                dr_dx=radial_directions,
            )
        )
        return (
            values,
            values_dx,
            soft_weights.to(dtype=disp.dtype),
            soft_weights_dx.to(dtype=disp.dtype),
        )

    def _pair_value_tensor(self, atom_types, values, default_value, *, dtype, device):
        max_type = int(torch.max(atom_types).detach().cpu()) if int(atom_types.numel()) else 0
        for left, right in values.keys():
            max_type = max(max_type, int(left), int(right))
        table = torch.full((max_type + 1, max_type + 1), float(default_value), dtype=dtype, device=device)
        for (left, right), value in values.items():
            table[int(left), int(right)] = float(value)
        return table

    def _edge_pair_cutoffs(self, atom_types, src, dst, *, dtype, device):
        lifted = self.config.lifted_density
        if not lifted.pair_cutoffs:
            return torch.full((int(src.numel()),), float(lifted.cutoff), dtype=dtype, device=device)
        table = self._pair_value_tensor(
            atom_types,
            lifted.pair_cutoffs,
            float(lifted.cutoff),
            dtype=dtype,
            device=device,
        )
        return table[atom_types[src].to(device=device), atom_types[dst].to(device=device)].to(dtype=dtype)

    def _apply_pair_cutoff_mask(
        self,
        atom_types,
        src,
        dst,
        disp,
        dist,
        return_mask=False,
    ):
        lifted = self.config.lifted_density
        if not lifted.pair_cutoffs or int(src.numel()) == 0:
            pair_cutoffs = torch.full(
                (int(src.numel()),),
                float(lifted.cutoff),
                dtype=dist.dtype,
                device=dist.device,
            )
            if return_mask:
                return (
                    src,
                    dst,
                    disp,
                    dist,
                    pair_cutoffs,
                    torch.ones_like(src, dtype=torch.bool),
                )
            return src, dst, disp, dist, pair_cutoffs
        pair_cutoffs = self._edge_pair_cutoffs(atom_types, src, dst, dtype=dist.dtype, device=dist.device)
        mask = dist <= (pair_cutoffs + 1.0e-12)
        result = (
            src[mask],
            dst[mask],
            disp[mask],
            dist[mask],
            pair_cutoffs[mask],
        )
        if return_mask:
            return result + (mask,)
        return result

    def _edge_data(
        self,
        positions,
        atom_types,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
        edge_cell=None,
        include_scheduled_density=False,
    ):
        lifted = self.config.lifted_density
        atom_count = int(positions.shape[0])
        if atom_types is None:
            atom_types = torch.zeros(atom_count, dtype=torch.long, device=positions.device)
        else:
            atom_types = torch.as_tensor(atom_types, dtype=torch.long, device=positions.device)
        if edge_index is not None:
            edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=positions.device)
            if edge_index_t.ndim != 2 or tuple(edge_index_t.shape[:1]) != (2,):
                raise ValueError("edge_index must have shape [2, n_edges].")
            src = edge_index_t[0]
            dst = edge_index_t[1]
            disp = edge_displacements_from_indices(
                positions,
                edge_index_t,
                cell=cell,
                shifts=shifts,
                edge_cell=edge_cell,
            )
            dist = torch.linalg.norm(disp, dim=1)
        elif lifted.periodic_image_mode == "nonperiodic":
            src, dst, disp, dist = _directed_edges(positions, lifted.max_edge_cutoff, cell=None, pbc=None)
        elif lifted.periodic_image_mode == "all_images":
            src, dst, disp, dist = _directed_edges_all_images(positions, lifted.max_edge_cutoff, cell=cell, pbc=pbc)
        else:
            src, dst, disp, dist = _directed_edges(positions, lifted.max_edge_cutoff, cell=cell, pbc=pbc)
        src, dst, disp, dist, pair_cutoffs = self._apply_pair_cutoff_mask(
            atom_types,
            src,
            dst,
            disp,
            dist,
        )
        if include_scheduled_density:
            scheduled_density = (
                self._scheduled_softmax_gaussian_role_density_accumulate(
                    atom_types,
                    src,
                    dst,
                    disp,
                    dist,
                    pair_cutoffs,
                    atom_count,
                )
            )
            if scheduled_density is not None:
                return (
                    src,
                    dst,
                    disp,
                    dist,
                    None,
                    None,
                    pair_cutoffs,
                    scheduled_density,
                )
        values, soft_weights = self._site_basis_edge_values(
            atom_types,
            src,
            dst,
            disp,
            return_soft_weights=True,
            n_atoms=atom_count,
            distance=dist,
        )
        edge_data = (
            src,
            dst,
            disp,
            dist,
            values,
            soft_weights,
            pair_cutoffs,
        )
        if include_scheduled_density:
            return edge_data + (None,)
        return edge_data

    def _filter_values_for_spec(self, dist, *, cutoff, filter_kind, num_filters, centers, width):
        lifted = self.config.lifted_density
        if dist.numel() == 0:
            return dist.new_zeros((0, int(num_filters)))
        cutoff_t = torch.as_tensor(cutoff, dtype=dist.dtype, device=dist.device)
        if cutoff_t.ndim == 0:
            cutoff_t = cutoff_t.expand_as(dist)
        filter_kind = str(filter_kind)
        if filter_kind == "constant":
            return torch.ones((int(dist.numel()), int(num_filters)), dtype=dist.dtype, device=dist.device)
        if filter_kind == "bernstein":
            t = torch.clamp(dist / cutoff_t, min=0.0, max=1.0)
            degree = int(num_filters) - 1
            cols = []
            for power in range(int(num_filters)):
                coeff = float(comb(degree, power))
                cols.append(coeff * t.pow(power) * (1.0 - t).pow(degree - power))
            return torch.stack(cols, dim=1)
        centers = torch.tensor(centers, dtype=dist.dtype, device=dist.device)
        scaled = dist.unsqueeze(-1) / cutoff_t.unsqueeze(-1)
        width = torch.as_tensor(float(width), dtype=dist.dtype, device=dist.device)
        logits = -0.5 * (
            (scaled - centers.reshape(1, -1)) / width
        ).pow(2)
        if filter_kind == "softmax_gaussian":
            return torch.softmax(logits, dim=1)
        raw = torch.exp(logits)
        if filter_kind == "cosine_shell":
            x = torch.abs(scaled - centers.reshape(1, -1)) / width
            shell = 0.5 * (torch.cos(torch.pi * torch.clamp(x, max=1.0)) + 1.0)
            return torch.where(x <= 1.0, shell, torch.zeros_like(shell))
        return raw

    def _supports_vectorized_pair_softmax_filters(self):
        lifted = self.config.lifted_density
        policy = str(self._pair_filter_execution_policy)
        if policy not in {
            "auto",
            "reference",
            "vectorized",
            "masked",
        }:
            raise RuntimeError(
                "pair filter execution policy must be auto, reference, "
                "vectorized, or masked"
            )
        compatible = bool(
            lifted.pair_filter_specs
            and str(lifted.filter_kind) == "softmax_gaussian"
            and all(
                str(spec["filter_kind"]) == "softmax_gaussian"
                for spec in lifted.pair_filter_specs.values()
            )
        )
        if policy == "vectorized" and not compatible:
            raise RuntimeError(
                "forced pair softmax filter vectorization requires only "
                "softmax_gaussian global and pair filter specifications"
            )
        return bool(policy == "vectorized" and compatible)

    def _uses_masked_pair_filters(self):
        policy = str(self._pair_filter_execution_policy)
        return bool(
            self.config.lifted_density.pair_filter_specs
            and policy in {"auto", "masked"}
        )

    def _pair_filter_specs_cover_possible_types(self):
        lifted = self.config.lifted_density
        possible_types = tuple(
            int(value) for value in lifted.possible_types
        )
        if not possible_types:
            return False
        return all(
            (left, right) in lifted.pair_filter_specs
            for left in possible_types
            for right in possible_types
        )

    def _masked_pair_filter_values(
        self,
        dist,
        pair_cutoffs,
        src,
        dst,
        atom_types,
        *,
        with_derivative,
    ):
        lifted = self.config.lifted_density
        if self._pair_filter_specs_cover_possible_types():
            values = dist.new_zeros(
                (int(dist.numel()), int(lifted.num_filters))
            )
            if with_derivative:
                derivative = torch.zeros_like(values)
        else:
            default_cutoff = (
                float(lifted.cutoff)
                if pair_cutoffs is None
                else pair_cutoffs
            )
            if with_derivative:
                values, derivative = self._filter_values_with_dr_for_spec(
                    dist,
                    cutoff=default_cutoff,
                    filter_kind=lifted.filter_kind,
                    num_filters=lifted.num_filters,
                    centers=lifted.filter_centers,
                    width=lifted.filter_width,
                )
            else:
                values = self._filter_values_for_spec(
                    dist,
                    cutoff=default_cutoff,
                    filter_kind=lifted.filter_kind,
                    num_filters=lifted.num_filters,
                    centers=lifted.filter_centers,
                    width=lifted.filter_width,
                )
        source_types = atom_types.index_select(0, src).to(
            device=dist.device
        )
        target_types = atom_types.index_select(0, dst).to(
            device=dist.device
        )
        for pair, spec in lifted.pair_filter_specs.items():
            mask = (
                (source_types == int(pair[0]))
                & (target_types == int(pair[1]))
            ).unsqueeze(1)
            cutoff = (
                pair_cutoffs
                if pair_cutoffs is not None
                else float(
                    lifted.pair_cutoffs.get(pair, lifted.cutoff)
                )
            )
            active_width = int(spec["num_filters"])
            if active_width > int(values.shape[1]):
                raise ValueError(
                    "pair filter width exceeds the shared role-density width"
                )
            if with_derivative:
                block_values, block_derivative = (
                    self._filter_values_with_dr_for_spec(
                        dist,
                        cutoff=cutoff,
                        filter_kind=spec["filter_kind"],
                        num_filters=active_width,
                        centers=spec["filter_centers"],
                        width=spec["filter_width"],
                    )
                )
                block_derivative = torch.nn.functional.pad(
                    block_derivative,
                    (0, int(derivative.shape[1]) - active_width),
                )
                derivative = torch.where(
                    mask,
                    block_derivative,
                    derivative,
                )
            else:
                block_values = self._filter_values_for_spec(
                    dist,
                    cutoff=cutoff,
                    filter_kind=spec["filter_kind"],
                    num_filters=active_width,
                    centers=spec["filter_centers"],
                    width=spec["filter_width"],
                )
            block_values = torch.nn.functional.pad(
                block_values,
                (0, int(values.shape[1]) - active_width),
            )
            values = torch.where(mask, block_values, values)
        if not torch.compiler.is_compiling():
            self._last_pair_filter_operation = (
                "branchless_masked_pair_filters_with_derivative"
                if with_derivative
                else "branchless_masked_pair_filters"
            )
        if with_derivative:
            return values, derivative
        return values

    def _pair_softmax_filter_tables(self, *, dtype, device):
        lifted = self.config.lifted_density
        cache_key = (device.type, device.index, dtype)
        cached = self._pair_softmax_filter_table_cache.get(cache_key)
        if cached is not None:
            return cached
        maximum_type = max(
            tuple(int(value) for value in lifted.possible_types)
            + tuple(
                int(value)
                for pair in lifted.pair_filter_specs
                for value in pair
            )
        )
        type_count = int(maximum_type + 1)
        role_count = int(lifted.num_filters)
        row_count = int(type_count * type_count)
        default_centers = torch.tensor(
            lifted.filter_centers,
            dtype=dtype,
            device=device,
        )
        centers = default_centers.reshape(1, role_count).expand(
            row_count, role_count
        ).clone()
        widths = torch.full(
            (row_count, 1),
            float(lifted.filter_width),
            dtype=dtype,
            device=device,
        )
        active = torch.ones(
            (row_count, role_count),
            dtype=torch.bool,
            device=device,
        )
        cutoffs = torch.full(
            (row_count,),
            float(lifted.cutoff),
            dtype=dtype,
            device=device,
        )
        for (left, right), cutoff in lifted.pair_cutoffs.items():
            cutoffs[int(left) * type_count + int(right)] = float(cutoff)
        for (left, right), spec in lifted.pair_filter_specs.items():
            row = int(left) * type_count + int(right)
            current_count = int(spec["num_filters"])
            centers[row] = 0.0
            centers[row, :current_count] = torch.tensor(
                spec["filter_centers"],
                dtype=dtype,
                device=device,
            )
            widths[row] = float(spec["filter_width"])
            active[row] = False
            active[row, :current_count] = True
        cached = {
            "type_count": int(type_count),
            "centers": centers,
            "widths": widths,
            "active": active,
            "cutoffs": cutoffs,
        }
        self._pair_softmax_filter_table_cache[cache_key] = cached
        return cached

    def _vectorized_pair_softmax_filters(
        self,
        dist,
        pair_cutoffs,
        src,
        dst,
        atom_types,
        *,
        with_derivative,
    ):
        tables = self._pair_softmax_filter_tables(
            dtype=dist.dtype,
            device=dist.device,
        )
        atom_types = atom_types.to(device=dist.device)
        pair_rows = (
            atom_types.index_select(0, src) * int(tables["type_count"])
            + atom_types.index_select(0, dst)
        )
        centers = tables["centers"].index_select(0, pair_rows)
        widths = tables["widths"].index_select(0, pair_rows)
        active = tables["active"].index_select(0, pair_rows)
        if pair_cutoffs is None:
            cutoff = tables["cutoffs"].index_select(0, pair_rows)
        else:
            cutoff = pair_cutoffs.to(dtype=dist.dtype, device=dist.device)
        scaled = dist.unsqueeze(1) / cutoff.unsqueeze(1)
        delta = (scaled - centers) / widths
        logits = torch.where(
            active,
            -0.5 * delta.pow(2),
            torch.full_like(delta, -torch.inf),
        )
        values = torch.softmax(logits, dim=1)
        if not torch.compiler.is_compiling():
            self._last_pair_filter_operation = (
                "vectorized_pair_softmax_gaussian"
            )
        if not with_derivative:
            return values
        logarithmic_derivative = (
            -(scaled - centers)
            / (widths.pow(2) * cutoff.unsqueeze(1))
        )
        mean_derivative = (
            values * logarithmic_derivative
        ).sum(dim=1, keepdim=True)
        derivative = values * (
            logarithmic_derivative - mean_derivative
        )
        return values, derivative

    def _filter_values(self, dist, pair_cutoffs=None, src=None, dst=None, atom_types=None):
        lifted = self.config.lifted_density
        default_cutoff = float(lifted.cutoff) if pair_cutoffs is None else pair_cutoffs
        values = self._filter_values_for_spec(
            dist,
            cutoff=default_cutoff,
            filter_kind=lifted.filter_kind,
            num_filters=lifted.num_filters,
            centers=lifted.filter_centers,
            width=lifted.filter_width,
        )
        if not lifted.pair_filter_specs or dist.numel() == 0:
            return values
        if src is None or dst is None or atom_types is None:
            raise ValueError("Pair-dependent slot filters require src, dst, and atom_types.")
        if self._supports_vectorized_pair_softmax_filters():
            return self._vectorized_pair_softmax_filters(
                dist,
                pair_cutoffs,
                src,
                dst,
                atom_types,
                with_derivative=False,
            )
        if self._uses_masked_pair_filters():
            return self._masked_pair_filter_values(
                dist,
                pair_cutoffs,
                src,
                dst,
                atom_types,
                with_derivative=False,
            )
        if not torch.compiler.is_compiling():
            self._last_pair_filter_operation = (
                "pair_mask_reference"
            )
        # Pair replacement below is intentionally simple reference code.  Do
        # not modify the softmax output itself: its backward saves that tensor.
        values = values.clone()
        for pair, spec in lifted.pair_filter_specs.items():
            mask = (atom_types[src].to(device=dist.device) == int(pair[0])) & (atom_types[dst].to(device=dist.device) == int(pair[1]))
            if not bool(torch.any(mask)):
                continue
            cutoff = pair_cutoffs[mask] if pair_cutoffs is not None else float(lifted.pair_cutoffs.get(pair, lifted.cutoff))
            block = self._filter_values_for_spec(
                dist[mask],
                cutoff=cutoff,
                filter_kind=spec["filter_kind"],
                num_filters=spec["num_filters"],
                centers=spec["filter_centers"],
                width=spec["filter_width"],
            )
            values[mask] = 0.0
            values[mask, : int(spec["num_filters"])] = block
        return values

    def _filter_values_with_dr_for_spec(self, dist, *, cutoff, filter_kind, num_filters, centers, width):
        """Return slot filter values and derivatives with respect to distance."""

        if dist.numel() == 0:
            empty = dist.new_zeros((0, int(num_filters)))
            return empty, empty
        cutoff_t = torch.as_tensor(cutoff, dtype=dist.dtype, device=dist.device)
        if cutoff_t.ndim == 0:
            cutoff_t = cutoff_t.expand_as(dist)
        filter_kind = str(filter_kind)
        if filter_kind == "constant":
            values = torch.ones((int(dist.numel()), int(num_filters)), dtype=dist.dtype, device=dist.device)
            return values, torch.zeros_like(values)
        if filter_kind == "bernstein":
            t = torch.clamp(dist / cutoff_t, min=0.0, max=1.0)
            degree = int(num_filters) - 1
            cols = []
            dcols_dt = []
            for power in range(int(num_filters)):
                coeff = float(comb(degree, power))
                value = coeff * t.pow(power) * (1.0 - t).pow(degree - power)
                left = torch.zeros_like(t) if power == 0 else power * t.pow(power - 1) * (1.0 - t).pow(degree - power)
                right = (
                    torch.zeros_like(t)
                    if degree == power
                    else -(degree - power) * t.pow(power) * (1.0 - t).pow(degree - power - 1)
                )
                cols.append(value)
                dcols_dt.append(coeff * (left + right))
            values = torch.stack(cols, dim=1)
            deriv = torch.stack(dcols_dt, dim=1) / cutoff_t.unsqueeze(1)
            active = ((dist / cutoff_t) >= 0.0) & ((dist / cutoff_t) <= 1.0)
            return values, torch.where(active.unsqueeze(1), deriv, torch.zeros_like(deriv))
        centers = torch.tensor(centers, dtype=dist.dtype, device=dist.device)
        scaled = dist.unsqueeze(-1) / cutoff_t.unsqueeze(-1)
        width = torch.as_tensor(float(width), dtype=dist.dtype, device=dist.device)
        delta = (scaled - centers.reshape(1, -1)) / width
        logits = -0.5 * delta.pow(2)
        logarithmic_derivative = (
            -(scaled - centers.reshape(1, -1))
            / (width.pow(2) * cutoff_t.unsqueeze(-1))
        )
        if filter_kind == "softmax_gaussian":
            values = torch.softmax(logits, dim=1)
            mean_log_deriv = (
                values * logarithmic_derivative
            ).sum(dim=1, keepdim=True)
            deriv = values * (
                logarithmic_derivative - mean_log_deriv
            )
            return values, deriv
        raw = torch.exp(logits)
        draw = raw * logarithmic_derivative
        if filter_kind == "cosine_shell":
            x = torch.abs(scaled - centers.reshape(1, -1)) / width
            shell = 0.5 * (torch.cos(torch.pi * torch.clamp(x, max=1.0)) + 1.0)
            values = torch.where(x <= 1.0, shell, torch.zeros_like(shell))
            sign = torch.sign(scaled - centers.reshape(1, -1))
            deriv = -0.5 * torch.pi * torch.sin(torch.pi * x) * sign / (width * cutoff_t.unsqueeze(-1))
            return values, torch.where(x <= 1.0, deriv, torch.zeros_like(deriv))
        return raw, draw

    def _filter_values_with_dr(self, dist, pair_cutoffs=None, src=None, dst=None, atom_types=None):
        lifted = self.config.lifted_density
        default_cutoff = float(lifted.cutoff) if pair_cutoffs is None else pair_cutoffs
        values, deriv = self._filter_values_with_dr_for_spec(
            dist,
            cutoff=default_cutoff,
            filter_kind=lifted.filter_kind,
            num_filters=lifted.num_filters,
            centers=lifted.filter_centers,
            width=lifted.filter_width,
        )
        if not lifted.pair_filter_specs or dist.numel() == 0:
            return values, deriv
        if src is None or dst is None or atom_types is None:
            raise ValueError("Pair-dependent slot filters require src, dst, and atom_types.")
        if self._supports_vectorized_pair_softmax_filters():
            return self._vectorized_pair_softmax_filters(
                dist,
                pair_cutoffs,
                src,
                dst,
                atom_types,
                with_derivative=True,
            )
        if self._uses_masked_pair_filters():
            return self._masked_pair_filter_values(
                dist,
                pair_cutoffs,
                src,
                dst,
                atom_types,
                with_derivative=True,
            )
        if not torch.compiler.is_compiling():
            self._last_pair_filter_operation = (
                "pair_mask_reference_with_derivative"
            )
        for pair, spec in lifted.pair_filter_specs.items():
            mask = (atom_types[src].to(device=dist.device) == int(pair[0])) & (atom_types[dst].to(device=dist.device) == int(pair[1]))
            if not bool(torch.any(mask)):
                continue
            cutoff = pair_cutoffs[mask] if pair_cutoffs is not None else float(lifted.pair_cutoffs.get(pair, lifted.cutoff))
            block_values, block_deriv = self._filter_values_with_dr_for_spec(
                dist[mask],
                cutoff=cutoff,
                filter_kind=spec["filter_kind"],
                num_filters=spec["num_filters"],
                centers=spec["filter_centers"],
                width=spec["filter_width"],
            )
            active_width = int(spec["num_filters"])
            if active_width > int(values.shape[1]):
                raise ValueError(
                    "pair filter width exceeds the shared role-density width"
                )
            padded_values = torch.nn.functional.pad(
                block_values,
                (0, int(values.shape[1]) - active_width),
            )
            padded_deriv = torch.nn.functional.pad(
                block_deriv,
                (0, int(deriv.shape[1]) - active_width),
            )
            pair_indices = torch.nonzero(
                mask, as_tuple=False
            ).reshape(-1)
            values = values.index_copy(
                0, pair_indices, padded_values
            )
            deriv = deriv.index_copy(
                0, pair_indices, padded_deriv
            )
        return values, deriv

    def filtered_density_vjp(
        self,
        positions,
        density_adjoint,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
        edge_cell=None,
        return_geometry_derivatives=False,
        include_strain_derivative=True,
        normalization_density=None,
        normalization_density_adjoint=None,
        channel_transform=None,
        channel_transform_packed_maps=None,
        return_channel_map_adjoint=False,
    ):
        """Apply the reverse derivative of normalized ``A_s`` density to positions.

        ``density_adjoint`` is the adjoint of the normalized density returned by
        ``filtered_density``.  The returned tensor has the same shape as
        ``positions`` and is exact for the current fixed-edge, analytic
        site-basis/filter/normalization semantics.  It differentiates both the
        base density and the geometry-dependent ``K^{(l)}_{i,s}`` realization.
        """

        positions = torch.as_tensor(positions, dtype=self.config.torch_dtype)
        density_adjoint = torch.as_tensor(density_adjoint, dtype=positions.dtype, device=positions.device)
        squeeze_batch = False
        if density_adjoint.ndim == 3:
            density_adjoint = density_adjoint.unsqueeze(0)
            squeeze_batch = True
        if density_adjoint.ndim != 4:
            raise ValueError("density_adjoint must have shape [n_atoms, n_filters, n_channels] or [batch, n_atoms, n_filters, n_channels].")
        if bool(return_channel_map_adjoint):
            if channel_transform is None:
                raise ValueError(
                    "channel-map adjoints require a channel transform"
                )
            if normalization_density is None or normalization_density_adjoint is None:
                raise ValueError(
                    "channel-map adjoints require the transformed density "
                    "and its adjoint"
                )
        if (normalization_density is None) != (
            normalization_density_adjoint is None
        ):
            raise ValueError(
                "normalization density and adjoint must be supplied together"
            )
        normalization_density_t = None
        normalization_density_adjoint_t = None
        if normalization_density is not None:
            normalization_density_t = torch.as_tensor(
                normalization_density,
                dtype=positions.dtype,
                device=positions.device,
            )
            normalization_density_adjoint_t = torch.as_tensor(
                normalization_density_adjoint,
                dtype=positions.dtype,
                device=positions.device,
            )
            if normalization_density_t.ndim == 3:
                normalization_density_t = normalization_density_t.unsqueeze(0)
            if normalization_density_adjoint_t.ndim == 3:
                normalization_density_adjoint_t = (
                    normalization_density_adjoint_t.unsqueeze(0)
                )
            if (
                normalization_density_t.ndim != 4
                or normalization_density_adjoint_t.shape
                != normalization_density_t.shape
                or int(normalization_density_t.shape[0])
                != int(density_adjoint.shape[0])
            ):
                raise ValueError(
                    "normalization density and adjoint must match the "
                    "density-adjoint batch and role axes"
                )
        cell_t = None if cell is None else torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
        pbc_t = _normalize_pbc(pbc, positions.device)
        if edge_cell is None:
            self._validate_periodic_cutoff_margin(cell_t, pbc_t)
        lifted = self.config.lifted_density
        atom_count = int(positions.shape[0])
        if atom_types is None:
            atom_types = torch.zeros(atom_count, dtype=torch.long, device=positions.device)
        else:
            atom_types = torch.as_tensor(atom_types, dtype=torch.long, device=positions.device)
        if edge_index is not None:
            edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=positions.device)
            if edge_index_t.ndim != 2 or tuple(edge_index_t.shape[:1]) != (2,):
                raise ValueError("edge_index must have shape [2, n_edges].")
            src = edge_index_t[0]
            dst = edge_index_t[1]
            disp = edge_displacements_from_indices(
                positions,
                edge_index_t,
                cell=cell_t,
                shifts=shifts,
                edge_cell=edge_cell,
            )
            dist = torch.linalg.norm(disp, dim=1)
        elif lifted.periodic_image_mode == "nonperiodic":
            src, dst, disp, dist = _directed_edges(positions, lifted.max_edge_cutoff, cell=None, pbc=None)
        elif lifted.periodic_image_mode == "all_images":
            src, dst, disp, dist = _directed_edges_all_images(positions, lifted.max_edge_cutoff, cell=cell_t, pbc=pbc_t)
        else:
            src, dst, disp, dist = _directed_edges(positions, lifted.max_edge_cutoff, cell=cell_t, pbc=pbc_t)
        (
            src,
            dst,
            disp,
            dist,
            pair_cutoffs,
            active_edge_mask,
        ) = self._apply_pair_cutoff_mask(
            atom_types,
            src,
            dst,
            disp,
            dist,
            return_mask=True,
        )
        edge_values, edge_dx, soft_weights, soft_weights_dx = self._site_basis_edge_values_with_dx(
            atom_types,
            src,
            dst,
            disp,
            distance=dist,
        )
        filters, filters_dr = self._filter_values_with_dr(
            dist,
            pair_cutoffs=pair_cutoffs,
            src=src,
            dst=dst,
            atom_types=atom_types,
        )
        mode = normalize_A_s_density_normalization_name(lifted.density_normalization)
        raw_density = None
        slot_soft_count = None
        if mode == "soft_neighbor":
            if normalization_density_t is None:
                raw_density, slot_soft_count = (
                    self._role_density_outer_accumulate(
                        filters,
                        edge_values,
                        soft_weights,
                        src,
                        atom_count,
                    )
                )
            else:
                slot_soft_count = self._role_density_accumulate(
                    filters * soft_weights.reshape(-1, 1),
                    src,
                    atom_count,
                )
        if mode == "soft_neighbor":
            raw_scale = slot_soft_count
            nugget = float(lifted.density_normalization_nugget)
            if nugget > 0.0:
                scale = raw_scale + raw_scale.new_tensor(nugget)
                active = torch.ones_like(raw_scale, dtype=torch.bool)
            else:
                active = raw_scale > torch.finfo(raw_scale.dtype).eps
                scale = torch.where(
                    active,
                    raw_scale,
                    torch.ones_like(raw_scale),
                )
            numerator_adjoint = density_adjoint / scale.unsqueeze(0).unsqueeze(-1)
            if normalization_density_t is None:
                scale_adjoint = -(density_adjoint * raw_density.unsqueeze(0)).sum(dim=3) / scale.pow(2).unsqueeze(0)
            else:
                scale_adjoint = -(
                    normalization_density_adjoint_t
                    * normalization_density_t.conj()
                ).real.sum(dim=3) / scale.unsqueeze(0)
            scale_adjoint = torch.where(active.unsqueeze(0), scale_adjoint, torch.zeros_like(scale_adjoint))
        elif mode in {"none", "degree", "sqrt_degree"}:
            if mode == "none":
                scale = positions.new_ones((atom_count,))
            else:
                degree = positions.new_zeros((atom_count,))
                if src.numel():
                    degree.index_add_(0, src, torch.ones((int(src.numel()),), dtype=positions.dtype, device=positions.device))
                scale = torch.clamp(degree, min=1.0)
                if mode == "sqrt_degree":
                    scale = torch.sqrt(scale)
            numerator_adjoint = density_adjoint / scale.reshape(1, -1, 1, 1)
            scale_adjoint = None
        else:
            raise ValueError(f"Unsupported A_s density VJP normalization mode {mode!r}.")

        channel_map_adjoint = None
        if bool(return_channel_map_adjoint):
            if mode == "soft_neighbor":
                transformed_numerator_adjoint = (
                    normalization_density_adjoint_t
                    / scale.unsqueeze(0).unsqueeze(-1)
                )
            else:
                transformed_numerator_adjoint = (
                    normalization_density_adjoint_t
                    / scale.reshape(1, -1, 1, 1)
                )
            if int(transformed_numerator_adjoint.shape[0]) != 1:
                raise ValueError(
                    "streamed channel-map adjoints currently require one "
                    "autograd output adjoint"
                )
            channel_map_adjoint = channel_transform.edge_map_adjoint(
                edge_values,
                filters,
                src,
                transformed_numerator_adjoint[0],
                atom_types=atom_types,
                packed_maps=channel_transform_packed_maps,
            )

        batch_count = int(density_adjoint.shape[0])
        grad_pos = positions.new_zeros((batch_count,) + tuple(positions.shape))
        if not src.numel():
            position_gradient = grad_pos[0] if squeeze_batch else grad_pos
            if not return_geometry_derivatives and not bool(
                return_channel_map_adjoint
            ):
                return position_gradient
            strain_derivative = (
                positions.new_zeros((batch_count, 3, 3))
                if include_strain_derivative
                else None
            )
            return {
                "position_gradient": position_gradient,
                "strain_derivative": (
                    None
                    if strain_derivative is None
                    else (
                        strain_derivative[0]
                        if squeeze_batch
                        else strain_derivative
                    )
                ),
                "cell_gradient": None,
                "edge_cell_gradient": None,
                "channel_map_adjoint": channel_map_adjoint,
            }
        src_adj = numerator_adjoint.index_select(1, src)
        adj_values = torch.einsum("besc,es->bec", src_adj, filters)
        adj_filters = torch.einsum("besc,ec->bes", src_adj, edge_values)
        adj_soft = positions.new_zeros((batch_count, int(src.numel())))
        if scale_adjoint is not None:
            src_scale_adj = scale_adjoint.index_select(1, src)
            adj_filters = adj_filters + src_scale_adj * soft_weights.reshape(1, -1, 1)
            adj_soft = torch.einsum("bes,es->be", src_scale_adj, filters)
        safe_dist = torch.clamp(dist, min=torch.as_tensor(1.0e-12, dtype=positions.dtype, device=positions.device))
        dr_dx = disp / safe_dist.unsqueeze(-1)
        grad_disp = torch.einsum("bec,ecd->bed", adj_values, edge_dx)
        grad_disp = grad_disp + (adj_filters * filters_dr.unsqueeze(0)).sum(dim=2, keepdim=True) * dr_dx.unsqueeze(0)
        grad_disp = grad_disp + adj_soft.unsqueeze(2) * soft_weights_dx.unsqueeze(0)
        flat_grad_pos = grad_pos.reshape(batch_count * atom_count, 3)
        batch_offsets = torch.arange(batch_count, dtype=torch.long, device=positions.device).reshape(-1, 1) * atom_count
        flat_dst = (dst.reshape(1, -1) + batch_offsets).reshape(-1)
        flat_src = (src.reshape(1, -1) + batch_offsets).reshape(-1)
        flat_grad_disp = grad_disp.reshape(batch_count * int(src.numel()), 3)
        flat_grad_pos.index_add_(0, flat_dst, flat_grad_disp)
        flat_grad_pos.index_add_(0, flat_src, -flat_grad_disp)
        position_gradient = grad_pos[0] if squeeze_batch else grad_pos
        if not return_geometry_derivatives and not bool(
            return_channel_map_adjoint
        ):
            return position_gradient
        strain_derivative = (
            torch.einsum(
                "bea,ec->bac",
                grad_disp,
                disp,
            )
            if include_strain_derivative
            else None
        )
        cell_gradient = None
        edge_cell_gradient = None
        if shifts is not None:
            shifts_t = torch.as_tensor(
                shifts,
                dtype=positions.dtype,
                device=positions.device,
            )
            active_edge_indices = torch.nonzero(
                active_edge_mask,
                as_tuple=False,
            ).reshape(-1)
            active_shifts = shifts_t.index_select(
                0,
                active_edge_indices,
            )
            active_edge_cell_gradient = torch.einsum(
                "ea,bec->beac",
                active_shifts,
                grad_disp,
            )
            if edge_cell is None and cell is not None:
                cell_gradient = active_edge_cell_gradient.sum(dim=1)
            elif edge_cell is not None:
                edge_cell_gradient = positions.new_zeros(
                    (
                        batch_count,
                        int(active_edge_mask.numel()),
                        3,
                        3,
                    )
                ).index_copy(
                    1,
                    active_edge_indices,
                    active_edge_cell_gradient,
                )
        return {
            "position_gradient": position_gradient,
            "strain_derivative": (
                None
                if strain_derivative is None
                else (
                    strain_derivative[0]
                    if squeeze_batch
                    else strain_derivative
                )
            ),
            "cell_gradient": (
                None
                if cell_gradient is None
                else (
                    cell_gradient[0]
                    if squeeze_batch
                    else cell_gradient
                )
            ),
            "edge_cell_gradient": (
                None
                if edge_cell_gradient is None
                else (
                    edge_cell_gradient[0]
                    if squeeze_batch
                    else edge_cell_gradient
                )
            ),
            "channel_map_adjoint": channel_map_adjoint,
        }

    def filtered_density_analytic_autograd(
        self,
        positions,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
        edge_cell=None,
    ):
        """Return ``A_s`` through the compact analytic geometry VJP.

        The forward values are identical to :meth:`filtered_density`. The
        custom autograd boundary recomputes the exact analytic position and
        periodic-cell adjoints of the base channels and ``K`` map instead of
        retaining the complete source construction graph. Its backward remains
        differentiable, so force training and position HVPs retain the required
        double backward.
        """

        positions = torch.as_tensor(
            positions,
            dtype=self.config.torch_dtype,
        )
        atom_count = int(positions.shape[0])
        atom_types_t = (
            torch.zeros(
                atom_count,
                dtype=torch.long,
                device=positions.device,
            )
            if atom_types is None
            else torch.as_tensor(
                atom_types,
                dtype=torch.long,
                device=positions.device,
            )
        )
        empty_float = positions.new_empty((0,))
        empty_long = torch.empty(
            (0,), dtype=torch.long, device=positions.device
        )
        cell_t = (
            empty_float
            if cell is None
            else torch.as_tensor(
                cell,
                dtype=positions.dtype,
                device=positions.device,
            )
        )
        pbc_t = _normalize_pbc(pbc, positions.device)
        edge_index_t = (
            empty_long
            if edge_index is None
            else torch.as_tensor(
                edge_index,
                dtype=torch.long,
                device=positions.device,
            )
        )
        shifts_t = (
            empty_float
            if shifts is None
            else torch.as_tensor(
                shifts,
                dtype=positions.dtype,
                device=positions.device,
            )
        )
        edge_cell_t = (
            empty_float
            if edge_cell is None
            else torch.as_tensor(
                edge_cell,
                dtype=positions.dtype,
                device=positions.device,
            )
        )
        return _FilteredDensityAnalyticAutograd.apply(
            positions,
            atom_types_t,
            cell_t,
            pbc_t,
            edge_index_t,
            shifts_t,
            edge_cell_t,
            self,
        )

    def filtered_density_transformed_analytic_autograd(
        self,
        positions,
        atom_types,
        channel_transform,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
        edge_cell=None,
    ):
        """Return compressed ``A_s`` with an analytic derivative boundary.

        The protected lifted source and its separate nonangular transform are

        ``A[i,s,eta,l,m] = sum_eta' K[i,s,l][eta,eta'] A[i,eta',l,m]``

        and

        ``Abar[i,s,a,l,m] = sum_eta T[c_i,[s],l][a,eta] A[i,s,eta,l,m]``.

        Forward accumulates physical edge channels through ``K`` and the
        compiler-bound ``T`` directly into compressed role-density outputs. It
        stores neither a complete physical ``A_s`` bank nor a forward source
        autograd graph. Backward reconstructs only the physical analytic edge
        basis and geometry VJP needed for the request, streams the channel-map
        adjoint, and applies the exact position/map double-adjoint schedule. It
        does not replay the source forward graph or materialize full physical
        role densities.
        """

        positions = torch.as_tensor(
            positions,
            dtype=self.config.torch_dtype,
        )
        atom_count = int(positions.shape[0])
        atom_types_t = (
            torch.zeros(
                atom_count,
                dtype=torch.long,
                device=positions.device,
            )
            if atom_types is None
            else torch.as_tensor(
                atom_types,
                dtype=torch.long,
                device=positions.device,
            )
        )
        empty_float = positions.new_empty((0,))
        empty_long = torch.empty(
            (0,), dtype=torch.long, device=positions.device
        )
        cell_t = (
            empty_float
            if cell is None
            else torch.as_tensor(
                cell,
                dtype=positions.dtype,
                device=positions.device,
            )
        )
        pbc_t = _normalize_pbc(pbc, positions.device)
        edge_index_t = (
            empty_long
            if edge_index is None
            else torch.as_tensor(
                edge_index,
                dtype=torch.long,
                device=positions.device,
            )
        )
        shifts_t = (
            empty_float
            if shifts is None
            else torch.as_tensor(
                shifts,
                dtype=positions.dtype,
                device=positions.device,
            )
        )
        edge_cell_t = (
            empty_float
            if edge_cell is None
            else torch.as_tensor(
                edge_cell,
                dtype=positions.dtype,
                device=positions.device,
            )
        )
        return _FilteredDensityTransformedAnalyticAutograd.apply(
            positions,
            atom_types_t,
            cell_t,
            pbc_t,
            edge_index_t,
            shifts_t,
            edge_cell_t,
            channel_transform.packed_maps(),
            self,
            channel_transform,
        )

    def filtered_density(
        self,
        positions,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
        edge_cell=None,
        channel_transform=None,
        channel_transform_packed_maps=None,
    ):
        """Return the normalized role-resolved ``A_s`` source.

        Before normalization, the source satisfies
        ``A_{i,s,eta,l,m} = sum_eta' K^{(l)}_{i,s;eta,eta'}
        A_{i,eta',l,m}``.  The returned channel axis flattens ``(eta,l,m)`` and
        therefore has shape ``[n_atoms, n_filters, n_channels]``.  This method
        realizes ``K`` directly on edge channels; it does not first materialize
        an ordinary density bank.
        """

        if channel_transform is None and channel_transform_packed_maps is not None:
            raise ValueError(
                "packed channel maps require a channel transform"
            )
        positions = torch.as_tensor(positions, dtype=self.config.torch_dtype)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3).")
        cell_t = None if cell is None else torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
        pbc_t = _normalize_pbc(pbc, positions.device)
        if edge_cell is None:
            self._validate_periodic_cutoff_margin(cell_t, pbc_t)
        edge_data = self._edge_data(
            positions,
            atom_types,
            cell=cell_t,
            pbc=pbc_t,
            edge_index=edge_index,
            shifts=shifts,
            edge_cell=edge_cell,
            include_scheduled_density=channel_transform is None,
        )
        if channel_transform is None:
            (
                src,
                dst,
                _disp,
                dist,
                edge_values,
                soft_weights,
                pair_cutoffs,
                scheduled_density,
            ) = edge_data
        else:
            (
                src,
                dst,
                _disp,
                dist,
                edge_values,
                soft_weights,
                pair_cutoffs,
            ) = edge_data
            scheduled_density = None
        if not torch.compiler.is_compiling():
            self._last_density_edge_count = int(src.numel())
            self._last_density_edge_source = (
                "explicit_edge_index"
                if edge_index is not None
                else "bruteforce_tensor_fallback"
            )
        lifted = self.config.lifted_density
        fused = scheduled_density
        if channel_transform is not None:
            atom_types_t = (
                torch.zeros(
                    int(positions.shape[0]),
                    dtype=torch.long,
                    device=positions.device,
                )
                if atom_types is None
                else torch.as_tensor(
                    atom_types,
                    dtype=torch.long,
                    device=positions.device,
                )
            )
            edge_values = channel_transform.forward_edge_channels(
                edge_values,
                atom_types_t.index_select(0, src),
                packed_maps=channel_transform_packed_maps,
            )
        if fused is None:
            fused = self._softmax_gaussian_role_density_accumulate(
                dist,
                pair_cutoffs,
                edge_values,
                soft_weights,
                src,
                int(positions.shape[0]),
            )
        if fused is None:
            filters = self._filter_values(
                dist,
                pair_cutoffs=pair_cutoffs,
                src=src,
                dst=dst,
                atom_types=torch.zeros(
                    int(positions.shape[0]),
                    dtype=torch.long,
                    device=positions.device,
                )
                if atom_types is None
                else torch.as_tensor(
                    atom_types,
                    dtype=torch.long,
                    device=positions.device,
                ),
            )
            # Realize the diagonal K_{i,s}^{(l)} map on each compatible edge
            # channel and use the same map for its normalization measure.
            density, slot_soft_count = self._role_density_outer_accumulate(
                filters,
                edge_values,
                soft_weights,
                src,
                int(positions.shape[0]),
            )
        else:
            density, slot_soft_count = fused
        return normalize_A_s_density(
            density,
            src=src,
            soft_count=slot_soft_count,
            mode=lifted.density_normalization,
            nugget=lifted.density_normalization_nugget,
        )

    def character_quadratic_site_energy(self, density):
        """Return a scalar readout from trivial and standard slot sectors."""

        mu = slot_trivial_component(density)
        nu = slot_standard_norm(density)
        return (
            mu @ self.character_mu_weight.to(dtype=density.dtype)
            + nu @ self.character_nu_weight.to(dtype=density.dtype)
            + self.character_bias.to(dtype=density.dtype)[0]
        )

    def antisymmetric_quadratic_site_energy(self, density):
        """Return a scalar readout from the fully antisymmetric slot magnitude."""

        volume = slot_antisymmetric_squared_volume(density)
        return (
            volume * self.antisymmetric_weight.to(dtype=density.dtype)[0]
            + self.antisymmetric_bias.to(dtype=density.dtype)[0]
        )

    def ye3_quadratic_site_energy(self, density):
        """Return scalar readout from Young-slot/E3 block norm invariants."""

        invariants, _blocks = ye3_quadratic_invariants(density, self.config.lifted_density.channels)
        if invariants.shape[1] != int(self.ye3_weight.numel()):
            raise ValueError("ye3_quadratic requires complete channel blocks matching model initialization.")
        return invariants @ self.ye3_weight.to(dtype=density.dtype) + self.ye3_bias.to(dtype=density.dtype)[0]

    def ye3_power_site_energy(self, density):
        """Return scalar readout from higher-order Young-slot/E3 power sectors."""

        lifted = self.config.lifted_density
        invariants, _blocks, _sectors = ye3_power_invariants(
            density,
            lifted.channels,
            max_power=lifted.ye3_max_power,
            optimization_policy=lifted.ye3_optimization_policy,
            slot_sectors=lifted.ye3_slot_sectors,
            include_rank1=lifted.ye3_include_rank1,
            rank_nmax=lifted.ye3_rank_nmax,
            rank_lmax=lifted.ye3_rank_lmax,
            rank_lmin=lifted.ye3_rank_lmin,
        )
        if invariants.shape[1] != int(self.ye3_power_weight.numel()):
            raise ValueError("ye3_power requires sector metadata matching model initialization.")
        return (
            invariants @ self.ye3_power_weight.to(dtype=density.dtype)
            + self.ye3_power_bias.to(dtype=density.dtype)[0]
        )

    def ye3_slot_specht_power_site_energy(self, density):
        """Return scalar readout from A_s slot-Specht central-projector norms."""

        lifted = self.config.lifted_density
        invariants, _blocks, _sectors = ye3_slot_specht_power_invariants(
            density,
            lifted.channels,
            max_power=lifted.ye3_max_power,
            optimization_policy=lifted.ye3_optimization_policy,
            slot_specht_partitions=lifted.ye3_slot_specht_partitions,
            slot_specht_coupling=lifted.ye3_slot_specht_coupling,
            include_rank1=lifted.ye3_include_rank1,
            rank_nmax=lifted.ye3_rank_nmax,
            rank_lmax=lifted.ye3_rank_lmax,
            rank_lmin=lifted.ye3_rank_lmin,
        )
        if invariants.shape[1] != int(self.ye3_slot_specht_power_weight.numel()):
            raise ValueError("ye3_slot_specht_power requires sector metadata matching model initialization.")
        return (
            invariants @ self.ye3_slot_specht_power_weight.to(dtype=density.dtype)
            + self.ye3_slot_specht_power_bias.to(dtype=density.dtype)[0]
        )

    def ye3_slot_specht_power_matrix_unit_site_carriers(self, density):
        """Return matrix-unit resolved slot-Specht carriers from an A_s density.

        This is a carrier-evaluation hook for the current A_s slot-Specht path.
        """

        lifted = self.config.lifted_density
        return ye3_slot_specht_power_matrix_unit_carriers(
            density,
            lifted.channels,
            max_power=lifted.ye3_max_power,
            optimization_policy=lifted.ye3_optimization_policy,
            slot_specht_partitions=lifted.ye3_slot_specht_partitions,
            slot_specht_coupling=lifted.ye3_slot_specht_coupling,
            include_rank1=lifted.ye3_include_rank1,
            rank_nmax=lifted.ye3_rank_nmax,
            rank_lmax=lifted.ye3_rank_lmax,
            rank_lmin=lifted.ye3_rank_lmin,
        )

    def ye3_slot_specht_power_matrix_unit_carriers(
        self,
        positions,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
    ):
        """Evaluate A_s density and return matrix-unit slot-Specht carriers.

        The returned tuple is ``(carriers, blocks, sectors)``.  Each carrier has
        shape ``[n_atoms, d_lambda, d_lambda, num_slots**power]``.  The sector
        metadata labels this as a matrix-unit carrier runtime rather than a full
        descriptor/model contraction.
        """

        density = self.filtered_density(
            positions,
            atom_types,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            shifts=shifts,
        )
        return self.ye3_slot_specht_power_matrix_unit_site_carriers(density)

    def forward(
        self,
        positions,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        active_branches=None,
        return_site_energies=False,
        edge_index=None,
        shifts=None,
    ):
        positions = torch.as_tensor(positions, dtype=self.config.torch_dtype)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3).")
        branches = self.branches if active_branches is None else normalize_lifted_density_branches(active_branches)
        if BRANCH_A in branches:
            raise ValueError(
                "The A branch requires the existing exact LinearACEScalarCalculator path. "
                "Pass ace_bundle to HybridACELiftedDensityCalculator, or call the tensor model with "
                "active_branches excluding 'A'."
            )
        start = time.perf_counter()
        density = self.filtered_density(
            positions,
            atom_types,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            shifts=shifts,
        )
        density_seconds = time.perf_counter() - start
        site_energy = positions.new_zeros(int(positions.shape[0]))
        readout_start = time.perf_counter()
        if BRANCH_LIFTED_DENSITY in branches:
            lifted = self.config.lifted_density
            if lifted.readout_mode == "character_quadratic":
                site_energy = site_energy + self.character_quadratic_site_energy(density)
            elif lifted.readout_mode == "antisymmetric_quadratic":
                site_energy = site_energy + self.antisymmetric_quadratic_site_energy(density)
            elif lifted.readout_mode == "ye3_quadratic":
                site_energy = site_energy + self.ye3_quadratic_site_energy(density)
            elif lifted.readout_mode == "ye3_power":
                site_energy = site_energy + self.ye3_power_site_energy(density)
            elif lifted.readout_mode == "ye3_slot_specht_power":
                site_energy = site_energy + self.ye3_slot_specht_power_site_energy(density)
            elif lifted.readout_mode == "symmetric_linear":
                slot_sum = density.sum(dim=1)
                site_energy = site_energy + slot_sum @ self.channel_readout.to(dtype=density.dtype)
                site_energy = site_energy + self.slot_equivariant_bias.to(dtype=density.dtype)[0]
            else:
                site_energy = site_energy + (density * self.linear_weight.to(dtype=density.dtype)).sum(dim=(1, 2))
                site_energy = site_energy + self.linear_bias.to(dtype=density.dtype)[0]
        readout_seconds = time.perf_counter() - readout_start
        self._last_profile = {
            "branches": tuple(branches),
            "atom_count": int(positions.shape[0]),
            "filter_count": int(self.config.lifted_density.num_filters),
            "channel_count": int(len(self.config.lifted_density.channels)),
            "readout_mode": str(self.config.lifted_density.readout_mode),
            "slot_group": str(self.config.lifted_density.slot_group),
            "density_normalization": str(self.config.lifted_density.density_normalization),
            "feature_normalization": str(self.config.lifted_density.feature_normalization),
            "periodic_image_mode": str(self.config.lifted_density.periodic_image_mode),
            "edge_count": int(getattr(self, "_last_density_edge_count", -1)),
            "density_edge_source": str(getattr(self, "_last_density_edge_source", "unknown")),
            "lifted_density_seconds": float(density_seconds),
            "lifted_readout_seconds": float(readout_seconds),
        }
        return site_energy if return_site_energies else site_energy.sum()

    def energy_forces_cell_gradient(
        self,
        positions,
        atom_types=None,
        *,
        cell,
        pbc=None,
        active_branches=None,
        fixed_scaled_positions=True,
        edge_index=None,
        shifts=None,
    ):
        """Return energy, Cartesian forces, and ``dE/dcell`` for ``A_s`` branches."""

        cell0 = torch.as_tensor(cell, dtype=self.config.torch_dtype)
        if tuple(cell0.shape) != (3, 3):
            raise ValueError("cell must have shape (3, 3).")
        pos0 = torch.as_tensor(positions, dtype=self.config.torch_dtype, device=cell0.device)
        if pos0.ndim != 2 or pos0.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3).")
        branches = self.branches if active_branches is None else normalize_lifted_density_branches(active_branches)
        if BRANCH_A in branches:
            raise ValueError("Cell gradients for the A branch require HybridACELiftedDensityCalculator.")
        cell_req = cell0.detach().clone().requires_grad_(True)
        if fixed_scaled_positions:
            scaled = (pos0.detach() @ torch.linalg.inv(cell0.detach())).to(dtype=cell_req.dtype, device=cell_req.device)
            pos_req = scaled @ cell_req
        else:
            pos_req = pos0.detach().clone().requires_grad_(True)
        energy = self(
            pos_req,
            atom_types,
            cell=cell_req,
            pbc=pbc,
            active_branches=branches,
            edge_index=edge_index,
            shifts=shifts,
        )
        grad_pos, grad_cell = torch.autograd.grad(energy, (pos_req, cell_req), create_graph=False, retain_graph=False)
        return energy.detach(), (-grad_pos).detach(), grad_cell.detach()

    def profile_report(self):
        return dict(self._last_profile)

    def config_dict(self):
        return self.config.to_dict()


def _tensor_cache_token(value):
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return (tuple(int(x) for x in tensor.shape), str(tensor.dtype), tensor.numpy().tobytes())


class _FilteredDensityAnalyticAutograd(torch.autograd.Function):
    """Compact differentiable geometry boundary for role density."""

    @staticmethod
    def forward(
        ctx,
        positions,
        atom_types,
        cell,
        pbc,
        edge_index,
        shifts,
        edge_cell,
        model,
    ):
        ctx.model = model
        ctx.has_cell = bool(cell.numel())
        ctx.has_edge_index = bool(edge_index.numel())
        ctx.has_shifts = bool(shifts.numel())
        ctx.has_edge_cell = bool(edge_cell.numel())
        ctx.save_for_backward(
            positions,
            atom_types,
            cell,
            pbc,
            edge_index,
            shifts,
            edge_cell,
        )
        return model.filtered_density(
            positions,
            atom_types,
            cell=cell if ctx.has_cell else None,
            pbc=pbc,
            edge_index=edge_index if ctx.has_edge_index else None,
            shifts=shifts if ctx.has_shifts else None,
            edge_cell=edge_cell if ctx.has_edge_cell else None,
        )

    @staticmethod
    def backward(ctx, density_adjoint):
        (
            positions,
            atom_types,
            cell,
            pbc,
            edge_index,
            shifts,
            edge_cell,
        ) = ctx.saved_tensors
        needs_cell_gradient = bool(
            ctx.needs_input_grad[2] and ctx.has_cell
        )
        needs_edge_cell_gradient = bool(
            ctx.needs_input_grad[6] and ctx.has_edge_cell
        )
        needs_periodic_gradient = bool(
            needs_cell_gradient or needs_edge_cell_gradient
        )
        geometry = ctx.model.filtered_density_vjp(
            positions,
            density_adjoint,
            atom_types,
            cell=cell if ctx.has_cell else None,
            pbc=pbc,
            edge_index=edge_index if ctx.has_edge_index else None,
            shifts=shifts if ctx.has_shifts else None,
            edge_cell=edge_cell if ctx.has_edge_cell else None,
            return_geometry_derivatives=needs_periodic_gradient,
            include_strain_derivative=False,
        )
        if not needs_periodic_gradient:
            geometry = {
                "position_gradient": geometry,
                "cell_gradient": None,
                "edge_cell_gradient": None,
            }
        cell_gradient = None
        if needs_cell_gradient:
            cell_gradient = geometry["cell_gradient"]
            if cell_gradient is None:
                cell_gradient = torch.zeros_like(cell)
        edge_cell_gradient = None
        if needs_edge_cell_gradient:
            edge_cell_gradient = geometry["edge_cell_gradient"]
            if edge_cell_gradient is None:
                edge_cell_gradient = torch.zeros_like(edge_cell)
        return (
            geometry["position_gradient"],
            None,
            cell_gradient,
            None,
            None,
            None,
            edge_cell_gradient,
            None,
        )


class _FilteredDensityTransformedAnalyticAutograd(torch.autograd.Function):
    """Checkpointed derivative boundary for compressed role density."""

    @staticmethod
    def forward(
        ctx,
        positions,
        atom_types,
        cell,
        pbc,
        edge_index,
        shifts,
        edge_cell,
        packed_maps,
        model,
        channel_transform,
    ):
        ctx.model = model
        ctx.channel_transform = channel_transform
        ctx.has_cell = bool(cell.numel())
        ctx.has_edge_index = bool(edge_index.numel())
        ctx.has_shifts = bool(shifts.numel())
        ctx.has_edge_cell = bool(edge_cell.numel())
        density = model.filtered_density(
            positions,
            atom_types,
            cell=cell if ctx.has_cell else None,
            pbc=pbc,
            edge_index=edge_index if ctx.has_edge_index else None,
            shifts=shifts if ctx.has_shifts else None,
            edge_cell=edge_cell if ctx.has_edge_cell else None,
            channel_transform=channel_transform,
            channel_transform_packed_maps=packed_maps,
        )
        ctx.save_for_backward(
            positions,
            atom_types,
            cell,
            pbc,
            edge_index,
            shifts,
            edge_cell,
            packed_maps,
            density,
        )
        return density

    @staticmethod
    def backward(ctx, density_adjoint):
        (
            positions,
            atom_types,
            cell,
            pbc,
            edge_index,
            shifts,
            edge_cell,
            packed_maps,
            normalized_density,
        ) = ctx.saved_tensors
        needs_position_gradient = bool(ctx.needs_input_grad[0])
        needs_cell_gradient = bool(
            ctx.needs_input_grad[2] and ctx.has_cell
        )
        needs_edge_cell_gradient = bool(
            ctx.needs_input_grad[6] and ctx.has_edge_cell
        )
        needs_geometry_gradient = bool(
            needs_position_gradient
            or needs_cell_gradient
            or needs_edge_cell_gradient
        )
        position_gradient = None
        cell_gradient = None
        edge_cell_gradient = None
        packed_map_gradient = None
        needs_map_gradient = bool(ctx.needs_input_grad[7])
        if needs_geometry_gradient or needs_map_gradient:
            physical_adjoint = (
                ctx.channel_transform.physical_adjoint(
                    density_adjoint,
                    atom_types=atom_types,
                    packed_maps=packed_maps,
                )
            )
            geometry = ctx.model.filtered_density_vjp(
                positions,
                physical_adjoint,
                atom_types,
                cell=cell if ctx.has_cell else None,
                pbc=pbc,
                edge_index=edge_index if ctx.has_edge_index else None,
                shifts=shifts if ctx.has_shifts else None,
                edge_cell=edge_cell if ctx.has_edge_cell else None,
                return_geometry_derivatives=bool(
                    needs_cell_gradient or needs_edge_cell_gradient
                ),
                include_strain_derivative=False,
                normalization_density=normalized_density,
                normalization_density_adjoint=density_adjoint,
                channel_transform=ctx.channel_transform,
                channel_transform_packed_maps=packed_maps,
                return_channel_map_adjoint=needs_map_gradient,
            )
            if isinstance(geometry, dict):
                position_gradient = geometry["position_gradient"]
                cell_gradient = geometry["cell_gradient"]
                edge_cell_gradient = geometry["edge_cell_gradient"]
                packed_map_gradient = geometry["channel_map_adjoint"]
            else:
                position_gradient = geometry
        else:
            packed_map_gradient = None
        return (
            position_gradient,
            None,
            cell_gradient,
            None,
            None,
            None,
            edge_cell_gradient,
            packed_map_gradient,
            None,
            None,
        )


class RoleFilterSpec:
    """Derivative metadata for the role filters defining an ``A_s`` density."""

    def __init__(
        self,
        role_representation,
        filter_type,
        geometry_dependent,
        differentiable,
        derivative_backend,
        normalization_map,
        collapse_control,
    ):
        self.role_representation = str(role_representation)
        self.filter_type = str(filter_type)
        self.geometry_dependent = bool(geometry_dependent)
        self.differentiable = bool(differentiable)
        self.derivative_backend = str(derivative_backend)
        self.normalization_map = str(normalization_map)
        self.collapse_control = bool(collapse_control)

    @classmethod
    def from_lifted_config(cls, lifted):
        filter_kind = str(lifted.filter_kind)
        pair_filter_kinds = tuple(str(spec["filter_kind"]) for spec in lifted.pair_filter_specs.values())
        all_filter_kinds = (filter_kind,) + pair_filter_kinds
        geometry_dependent = any(kind != "constant" for kind in all_filter_kinds)
        differentiable_families = {"constant", "radial_gaussian", "softmax_gaussian", "cosine_shell", "bernstein"}
        differentiable = all(kind in differentiable_families for kind in all_filter_kinds)
        derivative_backend = "analytic" if geometry_dependent else "none"
        if not differentiable:
            derivative_backend = "unsupported"
        return cls(
            role_representation="natural_slot_permutation",
            filter_type=filter_kind,
            geometry_dependent=geometry_dependent,
            differentiable=differentiable,
            derivative_backend=derivative_backend,
            normalization_map=normalize_A_s_density_normalization_name(lifted.density_normalization),
            collapse_control=filter_kind == "constant" and not lifted.pair_filter_specs and int(lifted.num_filters) > 1,
        )

    def to_dict(self):
        return {
            "role_representation": self.role_representation,
            "filter_type": self.filter_type,
            "geometry_dependent": self.geometry_dependent,
            "differentiable": self.differentiable,
            "derivative_backend": self.derivative_backend,
            "normalization_map": self.normalization_map,
            "collapse_control": self.collapse_control,
        }


class RoleDensityCache:
    """Cached role-density values and compact derivative factors."""

    def __init__(
        self,
        model,
        positions,
        atom_types,
        cell,
        pbc,
        edge_index,
        shifts,
        edge_cell,
        src,
        dst,
        disp,
        dist,
        raw_density,
        normalized_density,
        slot_soft_count,
        filters,
        filter_distance_derivatives,
        edge_values,
        edge_value_derivatives,
        soft_weights,
        soft_weight_derivatives,
        role_filter_spec,
    ):
        self.model = model
        self.positions = positions
        self.atom_types = atom_types
        self.cell = cell
        self.pbc = pbc
        self.edge_index = edge_index
        self.shifts = shifts
        self.edge_cell = edge_cell
        self.src = src
        self.dst = dst
        self.disp = disp
        self.dist = dist
        self.raw_density = raw_density
        self.normalized_density = normalized_density
        self.slot_soft_count = slot_soft_count
        self.filters = filters
        self.filter_distance_derivatives = filter_distance_derivatives
        self.edge_values = edge_values
        self.edge_value_derivatives = edge_value_derivatives
        self.soft_weights = soft_weights
        self.soft_weight_derivatives = soft_weight_derivatives
        self.role_filter_spec = role_filter_spec

    @classmethod
    def from_model(
        cls,
        model,
        positions,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
        edge_cell=None,
    ):
        positions = torch.as_tensor(positions, dtype=model.config.torch_dtype)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3).")
        cell_t = None if cell is None else torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
        pbc_t = _normalize_pbc(pbc, positions.device)
        if edge_cell is None:
            model._validate_periodic_cutoff_margin(cell_t, pbc_t)
        lifted = model.config.lifted_density
        atom_count = int(positions.shape[0])
        if atom_types is None:
            atom_types_t = torch.zeros(atom_count, dtype=torch.long, device=positions.device)
        else:
            atom_types_t = torch.as_tensor(atom_types, dtype=torch.long, device=positions.device)
        if edge_index is not None:
            edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=positions.device)
            if edge_index_t.ndim != 2 or tuple(edge_index_t.shape[:1]) != (2,):
                raise ValueError("edge_index must have shape [2, n_edges].")
            src = edge_index_t[0]
            dst = edge_index_t[1]
            disp = edge_displacements_from_indices(
                positions,
                edge_index_t,
                cell=cell_t,
                shifts=shifts,
                edge_cell=edge_cell,
            )
            dist = torch.linalg.norm(disp, dim=1)
        elif lifted.periodic_image_mode == "nonperiodic":
            src, dst, disp, dist = _directed_edges(positions, lifted.max_edge_cutoff, cell=None, pbc=None)
        elif lifted.periodic_image_mode == "all_images":
            src, dst, disp, dist = _directed_edges_all_images(positions, lifted.max_edge_cutoff, cell=cell_t, pbc=pbc_t)
        else:
            src, dst, disp, dist = _directed_edges(positions, lifted.max_edge_cutoff, cell=cell_t, pbc=pbc_t)
        src, dst, disp, dist, pair_cutoffs = model._apply_pair_cutoff_mask(atom_types_t, src, dst, disp, dist)
        edge_values, edge_dx, soft_weights, soft_weights_dx = model._site_basis_edge_values_with_dx(
            atom_types_t,
            src,
            dst,
            disp,
            distance=dist,
        )
        filters, filters_dr = model._filter_values_with_dr(
            dist,
            pair_cutoffs=pair_cutoffs,
            src=src,
            dst=dst,
            atom_types=atom_types_t,
        )
        raw_density, slot_soft_count = (
            model._role_density_outer_accumulate(
                filters,
                edge_values,
                soft_weights,
                src,
                atom_count,
            )
        )
        normalized_density = normalize_A_s_density(
            raw_density,
            src=src,
            soft_count=slot_soft_count,
            mode=lifted.density_normalization,
            nugget=lifted.density_normalization_nugget,
        )
        return cls(
            model,
            positions,
            atom_types_t,
            cell_t,
            pbc_t,
            edge_index,
            shifts,
            edge_cell,
            src,
            dst,
            disp,
            dist,
            raw_density,
            normalized_density,
            slot_soft_count,
            filters,
            filters_dr,
            edge_values,
            edge_dx,
            soft_weights,
            soft_weights_dx,
            RoleFilterSpec.from_lifted_config(lifted),
        )

    def position_vjp(self, density_adjoint):
        return self.model.filtered_density_vjp(
            self.positions,
            density_adjoint,
            self.atom_types,
            cell=self.cell,
            pbc=self.pbc,
            edge_index=self.edge_index,
            shifts=self.shifts,
        )

    def report(self):
        return {
            "edge_count": int(self.src.numel()),
            "raw_density_shape": tuple(int(x) for x in self.raw_density.shape),
            "normalized_density_shape": tuple(int(x) for x in self.normalized_density.shape),
            "stores_raw_density": True,
            "stores_normalized_density": True,
            "stores_edge_value_derivatives": True,
            "stores_role_filter_distance_derivatives": True,
            "stores_soft_weight_derivatives": True,
            "role_filter_spec": self.role_filter_spec.to_dict(),
        }


class RoleResolvedDensity:
    """Materialize role-resolved ``A_s`` density with a YE3T role contract."""

    def __init__(self, config=None, model=None, enable_cache=True):
        from ye3t.role import RoleResolvedCarrierSpec

        self.model = model if model is not None else HybridACELiftedDensityEnergyModel(config)
        lifted = self.model.config.lifted_density
        self.carrier_spec = RoleResolvedCarrierSpec(
            role_count=int(lifted.num_filters),
            retain_role_coordinate=True,
            identical_role_filters=str(lifted.filter_kind) == "constant" and int(lifted.num_filters) > 1,
            carrier="A_s",
            materialization_owner="ye3t-ace",
        )
        self.enable_cache = bool(enable_cache)
        self._cache = {}

    def _cache_key(
        self,
        positions,
        atom_types,
        cell,
        pbc,
        edge_index,
        shifts,
        edge_cell,
    ):
        lifted = self.model.config.lifted_density
        return (
            _tensor_cache_token(positions),
            _tensor_cache_token(atom_types),
            _tensor_cache_token(cell),
            _tensor_cache_token(pbc),
            _tensor_cache_token(edge_index),
            _tensor_cache_token(shifts),
            _tensor_cache_token(edge_cell),
            str(self.model.config.torch_dtype),
            str(lifted.density_normalization),
            float(lifted.density_normalization_nugget),
            str(lifted.filter_kind),
            int(lifted.num_filters),
            str(lifted.source_backend),
            int(lifted.native_source_min_edges),
            tuple(channel.to_dict().items() for channel in lifted.channels),
        )

    def clear_cache(self):
        self._cache.clear()

    def materialize(
        self,
        positions,
        atom_types=None,
        *,
        cell=None,
        pbc=None,
        edge_index=None,
        shifts=None,
        edge_cell=None,
        cache_key=None,
        include_derivatives=False,
    ):
        key = cache_key
        if key is None and self.enable_cache:
            key = self._cache_key(
                positions,
                atom_types,
                cell,
                pbc,
                edge_index,
                shifts,
                edge_cell,
            )
        if not include_derivatives and self.enable_cache and key in self._cache:
            cached = self._cache[key]
            return {
                "density": cached["density"].clone(),
                "cache_report": {"enabled": True, "hit": True, "key_source": cached["key_source"]},
                "role_contract": dict(cached["role_contract"]),
                "role_filter_spec": dict(cached["role_filter_spec"]),
                "source_runtime": dict(cached["source_runtime"]),
            }
        density_cache = None
        if include_derivatives:
            density_cache = RoleDensityCache.from_model(
                self.model,
                positions,
                atom_types,
                cell=cell,
                pbc=pbc,
                edge_index=edge_index,
                shifts=shifts,
                edge_cell=edge_cell,
            )
            density = density_cache.normalized_density
        else:
            density = self.model.filtered_density(
                positions,
                atom_types,
                cell=cell,
                pbc=pbc,
                edge_index=edge_index,
                shifts=shifts,
                edge_cell=edge_cell,
            )
        role_size = int(self.carrier_spec.role_module.permuted_role_count)
        role_partitions = ((role_size - 1, 1),) if role_size >= 2 else ((role_size,),)
        role_contract = self.carrier_spec.carrier_policy_report(partitions=role_partitions)
        role_filter_spec = RoleFilterSpec.from_lifted_config(self.model.config.lifted_density).to_dict()
        result = {
            "density": density,
            "cache_report": {
                "enabled": bool(self.enable_cache),
                "hit": False,
                "key_source": "explicit" if cache_key is not None else "density_inputs",
            },
            "role_contract": role_contract,
            "role_filter_spec": role_filter_spec,
            "source_runtime": self.model.source_runtime_report(),
        }
        if density_cache is not None:
            result["density_cache"] = density_cache
            result["density_cache_report"] = density_cache.report()
        if self.enable_cache:
            self._cache[key] = {
                "density": density.detach().clone(),
                "key_source": result["cache_report"]["key_source"],
                "role_contract": role_contract,
                "role_filter_spec": role_filter_spec,
                "source_runtime": self.model.source_runtime_report(),
            }
        return result

    def validate_role_equivariance(self, density, permutation):
        density_t = torch.as_tensor(density, dtype=self.model.config.torch_dtype)
        permutation_t = torch.as_tensor(permutation, dtype=torch.long, device=density_t.device)
        if density_t.ndim != 3:
            raise ValueError("density must have shape [n_atoms, n_roles, n_channels].")
        if int(density_t.shape[1]) != int(permutation_t.numel()):
            raise ValueError("permutation length must match the role axis.")
        inverse = torch.empty_like(permutation_t)
        inverse[permutation_t] = torch.arange(int(permutation_t.numel()), device=density_t.device)
        expected = density_t.index_select(1, inverse).detach().cpu().numpy()
        observed = self.carrier_spec.role_module.apply_role_action(
            density_t.detach().cpu().numpy(),
            tuple(int(x) for x in permutation_t.detach().cpu().tolist()),
            role_axis=1,
        )
        max_abs_error = np.max(np.abs(observed - expected)) if observed.size else 0.0
        return {
            "passed": bool(float(max_abs_error) <= 1.0e-12),
            "max_abs_error": float(max_abs_error),
            "action_convention": self.carrier_spec.role_module.to_dict()["action_convention"],
        }


class HybridACELiftedDensityCalculator(_ASECalculatorBase):
    """ASE calculator for exact ``A`` plus filtered/lifted ``A_s`` models."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model,
        type_map=None,
        device=None,
        *,
        ace_bundle=None,
        ace_cutoff=None,
        ace_type_map=None,
        ace_calculator_kwargs=None,
        reference_energies=None,
        neighbor_backend="ase",
        neighbor_skin=0.0,
        **kwargs,
    ):
        if _ASE_IMPORT_ERROR is not None:
            raise ImportError("ASE is required for HybridACELiftedDensityCalculator.") from _ASE_IMPORT_ERROR
        super().__init__(**kwargs)
        self.model = model
        self.type_map = {} if type_map is None else {str(k): int(v) for k, v in dict(type_map).items()}
        self.reference_energies = _normalize_reference_energies(reference_energies)
        self.device = torch.device("cpu" if device is None else device)
        self.neighbor_backend = str(neighbor_backend)
        self.neighbor_skin = max(float(neighbor_skin), 0.0)
        self._neighbor_cache = None
        self.model.to(self.device)
        self._last_profile = {}
        self.ace_calculator = None
        if ace_bundle is not None:
            if ace_cutoff is None:
                raise ValueError("ace_cutoff is required when ace_bundle is supplied.")
            try:
                from ye3t_ace.ace.linear_ace import LinearACEScalarCalculator
            except Exception:  # pragma: no cover - local import fallback
                from .ace.linear_ace import LinearACEScalarCalculator
            ace_kwargs = {} if ace_calculator_kwargs is None else dict(ace_calculator_kwargs)
            self.ace_calculator = LinearACEScalarCalculator(
                ace_bundle,
                cutoff=float(ace_cutoff),
                type_map=dict(self.type_map if ace_type_map is None else ace_type_map),
                device=self.device,
                **ace_kwargs,
            )
        elif BRANCH_A in self.model.branches:
            raise ValueError(
                "HybridACELiftedDensityCalculator requires ace_bundle and ace_cutoff when the model includes "
                "the A branch."
            )

    def _atom_types(self, atoms, device):
        if self.type_map:
            return torch.tensor([self.type_map[symbol] for symbol in atoms.get_chemical_symbols()], dtype=torch.long, device=device)
        return torch.zeros(len(atoms), dtype=torch.long, device=device)

    def _neighbor_type_map(self, atoms):
        if self.type_map:
            return dict(self.type_map)
        return {str(symbol): 0 for symbol in atoms.get_chemical_symbols()}

    def _neighbor_data_from_atoms(self, atoms):
        positions = np.asarray(atoms.positions, float)
        cell = np.asarray(atoms.cell.array, float)
        type_map = self._neighbor_type_map(atoms)
        atom_types = _atom_type_indices_from_atoms(atoms, type_map)
        build_cutoff = float(self.model.config.lifted_density.max_edge_cutoff) + self.neighbor_skin
        if _neighbor_cache_is_valid(
            self._neighbor_cache,
            positions,
            cell,
            atom_types,
            build_cutoff,
            self.neighbor_skin,
        ):
            return self._neighbor_cache["neighbor_data"]
        nbr = neighbor_data_from_ase_atoms(
            atoms,
            build_cutoff,
            type_map,
            backend=self.neighbor_backend,
        )
        if self.neighbor_skin > 0.0:
            self._neighbor_cache = {
                "positions": positions.copy(),
                "cell": cell.copy(),
                "atom_types": atom_types.copy(),
                "cutoff": float(build_cutoff),
                "neighbor_data": nbr,
            }
        return nbr

    def _geometry_from_atoms(self, atoms, *, requires_grad):
        dtype = self.model.config.torch_dtype
        pos = torch.as_tensor(np.asarray(atoms.positions, float), dtype=dtype, device=self.device)
        if requires_grad:
            pos = pos.detach().clone().requires_grad_(True)
        nbr = self._neighbor_data_from_atoms(atoms)
        cell = torch.as_tensor(np.asarray(atoms.cell.array, float), dtype=dtype, device=self.device)
        edge_index = torch.as_tensor(nbr.edge_index, dtype=torch.long, device=self.device)
        shifts = torch.as_tensor(np.asarray(nbr.shifts, float), dtype=dtype, device=self.device)
        atom_types = torch.as_tensor(nbr.atom_types, dtype=torch.long, device=self.device)
        cutoff = float(self.model.config.lifted_density.max_edge_cutoff)
        if self.neighbor_skin > 0.0 and edge_index.numel() > 0:
            x_ij = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
            mask = torch.linalg.norm(x_ij, dim=1) <= (cutoff + 1.0e-12)
            edge_index = edge_index[:, mask]
            shifts = shifts[mask]
        return pos, cell, edge_index, shifts, atom_types

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        if system_changes is None:
            system_changes = _ASE_ALL_CHANGES
        _ASECalculatorBase.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("HybridACELiftedDensityCalculator requires atoms.")
        start = time.perf_counter()
        energy_offset = 0.0
        force_offset = None
        stress_offset = None
        active_branches = self.model.branches
        a_branch_source = "not_requested"
        stress_requested = "stress" in tuple(properties)
        if self.ace_calculator is not None and BRANCH_A in active_branches:
            ace_properties = ("energy", "forces", "stress") if stress_requested else ("energy", "forces")
            self.ace_calculator.calculate(atoms, properties=ace_properties, system_changes=system_changes)
            energy_offset = float(self.ace_calculator.results["energy"])
            force_offset = np.asarray(self.ace_calculator.results["forces"], dtype=float)
            if stress_requested:
                stress_offset = np.asarray(self.ace_calculator.results["stress"], dtype=float)
            active_branches = tuple(branch for branch in active_branches if branch != BRANCH_A)
            a_branch_source = "linear_ace_exact_basis"
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if active_branches and stress_requested:
            pos, cell, edge_index, shifts, atom_types = self._geometry_from_atoms(atoms, requires_grad=False)
            energy_t, forces_t, cell_grad_t = self.model.energy_forces_cell_gradient(
                pos,
                atom_types,
                cell=cell,
                pbc=pbc,
                active_branches=active_branches,
                fixed_scaled_positions=True,
                edge_index=edge_index,
                shifts=shifts,
            )
            forces = forces_t.detach().cpu().numpy()
            lifted_energy = float(energy_t.detach().cpu())
            cell_grad = cell_grad_t.detach().cpu().numpy()
            cell_np = np.asarray(atoms.cell.array, float)
            volume = float(atoms.get_volume())
            dE_dstrain = cell_np.T @ cell_grad
            stress_tensor = 0.5 * (dE_dstrain + dE_dstrain.T) / volume
            self.results["stress"] = _voigt_from_stress_tensor(stress_tensor)
        elif stress_requested:
            self.results["stress"] = np.zeros(6, dtype=float)
        elif active_branches:
            pos, cell, edge_index, shifts, atom_types = self._geometry_from_atoms(atoms, requires_grad=True)
            energy = self.model(
                pos,
                atom_types,
                cell=cell,
                pbc=pbc,
                active_branches=active_branches,
                edge_index=edge_index,
                shifts=shifts,
            )
            grad = torch.autograd.grad(energy, pos, create_graph=False, retain_graph=False)[0]
            forces = -grad.detach().cpu().numpy()
            lifted_energy = float(energy.detach().cpu())
        else:
            forces = np.zeros((len(atoms), 3), dtype=float)
            lifted_energy = 0.0
        if force_offset is not None:
            forces = forces + force_offset
        residual_energy = float(energy_offset + lifted_energy)
        reference_offset = _reference_energy_offset_from_atoms(atoms, self.reference_energies)
        total_energy = residual_energy + reference_offset
        self.results["energy"] = float(total_energy)
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = forces
        if stress_requested and stress_offset is not None:
            self.results["stress"] = np.asarray(self.results["stress"], dtype=float) + stress_offset
        self._last_profile = dict(self.model.profile_report())
        self._last_profile["residual_energy_eV"] = float(residual_energy)
        self._last_profile["reference_energy_offset_eV"] = float(reference_offset)
        self._last_profile["total_energy_eV"] = float(total_energy)
        self._last_profile["reference_energy_count"] = int(len(self.reference_energies))
        self._last_profile["ase_calculator_step_seconds"] = float(time.perf_counter() - start)
        self._last_profile["a_branch_source"] = str(a_branch_source)
        self._last_profile["neighbor_backend"] = str(self.neighbor_backend)
        self._last_profile["neighbor_skin"] = float(self.neighbor_skin)
        self._last_profile["edge_source"] = "ase_or_matscipy_neighbor_data" if active_branches else "not_requested"
        if self.ace_calculator is not None:
            self._last_profile["linear_ace_backend_report"] = dict(self.ace_calculator.backend_report())

    def profile_report(self):
        return dict(self._last_profile)

    def energy_forces_cell_gradient(self, atoms=None, *, fixed_scaled_positions=True):
        energy, forces, cell_grad = self.energy_forces_cell_gradient_torch(
            atoms,
            fixed_scaled_positions=fixed_scaled_positions,
        )
        return (
            float(energy.detach().cpu()) + _reference_energy_offset_from_atoms(atoms, self.reference_energies),
            forces.detach().cpu().numpy(),
            cell_grad.detach().cpu().numpy(),
        )

    def energy_forces_cell_gradient_torch(self, atoms=None, *, fixed_scaled_positions=True):
        """Evaluate hybrid energy, forces, and cell gradient as torch tensors."""

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("HybridACELiftedDensityCalculator requires atoms.")
        active_branches = self.model.branches
        dtype = self.model.config.torch_dtype
        energy_total = torch.zeros((), dtype=dtype, device=self.device)
        forces_total = torch.zeros((len(atoms), 3), dtype=dtype, device=self.device)
        cell_grad_total = torch.zeros((3, 3), dtype=dtype, device=self.device)
        if BRANCH_A in active_branches:
            if self.ace_calculator is None:
                raise ValueError("A-branch cell gradients require an exact ACE calculator.")
            ace_energy, ace_forces, ace_cell_grad, _site_energy = self.ace_calculator.energy_forces_cell_gradient(
                atoms,
                fixed_scaled_positions=fixed_scaled_positions,
            )
            energy_total = energy_total + ace_energy.detach().to(device=self.device, dtype=dtype)
            forces_total = forces_total + ace_forces.detach().to(device=self.device, dtype=dtype)
            cell_grad_total = cell_grad_total + ace_cell_grad.detach().to(device=self.device, dtype=dtype)
            active_branches = tuple(branch for branch in active_branches if branch != BRANCH_A)
        if active_branches:
            positions, cell, edge_index, shifts, atom_types = self._geometry_from_atoms(atoms, requires_grad=False)
            pbc = np.asarray(atoms.pbc, dtype=bool)
            energy, forces, cell_grad = self.model.energy_forces_cell_gradient(
                positions,
                atom_types,
                cell=cell,
                pbc=pbc,
                active_branches=active_branches,
                fixed_scaled_positions=fixed_scaled_positions,
                edge_index=edge_index,
                shifts=shifts,
            )
            energy_total = energy_total + energy.detach().to(device=self.device, dtype=dtype)
            forces_total = forces_total + forces.detach().to(device=self.device, dtype=dtype)
            cell_grad_total = cell_grad_total + cell_grad.detach().to(device=self.device, dtype=dtype)
        offset = _reference_energy_offset_from_atoms(atoms, self.reference_energies)
        if offset:
            energy_total = energy_total + torch.as_tensor(float(offset), dtype=dtype, device=self.device)
        return energy_total, forces_total, cell_grad_total


def save_hybrid_ace_lifted_density_ase_bundle(
    path,
    model,
    type_map=None,
    *,
    ace_bundle=None,
    ace_cutoff=None,
    ace_type_map=None,
    reference_energies=None,
):
    """Save a lifted-density model/config bundle restorable as an ASE calculator."""

    if BRANCH_A in model.branches and ace_bundle is None:
        raise ValueError("A-branch lifted-density ASE bundles require ace_bundle metadata for restoration.")
    if ace_bundle is not None and ace_cutoff is None:
        raise ValueError("ace_cutoff is required when ace_bundle is supplied.")
    payload = {
        "format": "hybrid_ace_lifted_density_ase_bundle",
        "version": 1,
        "config": model.config_dict(),
        "state_dict": model.state_dict(),
        "type_map": {} if type_map is None else {str(k): int(v) for k, v in dict(type_map).items()},
        "ace_bundle": ace_bundle,
        "ace_cutoff": None if ace_cutoff is None else float(ace_cutoff),
        "ace_type_map": None if ace_type_map is None else {str(k): int(v) for k, v in dict(ace_type_map).items()},
        "reference_energies": _normalize_reference_energies(reference_energies),
    }
    torch.save(payload, Path(path))
    return Path(path)


def load_hybrid_ace_lifted_density_ase_bundle(path, *, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format") != "hybrid_ace_lifted_density_ase_bundle":
        raise ValueError("Unsupported hybrid ACE/lifted-density bundle format.")
    model = HybridACELiftedDensityEnergyModel(HybridACELiftedDensityConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model._hybrid_ace_lifted_density_bundle_metadata = {
        "ace_bundle": payload.get("ace_bundle", None),
        "ace_cutoff": payload.get("ace_cutoff", None),
        "ace_type_map": payload.get("ace_type_map", None),
        "reference_energies": _normalize_reference_energies(payload.get("reference_energies", {})),
    }
    return model, dict(payload.get("type_map", {}))


def load_hybrid_ace_lifted_density_calculator(path, **kwargs):
    model, type_map = load_hybrid_ace_lifted_density_ase_bundle(
        path,
        map_location=kwargs.pop("map_location", "cpu"),
    )
    metadata = dict(getattr(model, "_hybrid_ace_lifted_density_bundle_metadata", {}))
    if metadata.get("ace_bundle", None) is not None:
        kwargs.setdefault("ace_bundle", metadata["ace_bundle"])
        kwargs.setdefault("ace_cutoff", metadata["ace_cutoff"])
        kwargs.setdefault("ace_type_map", metadata.get("ace_type_map", None))
    kwargs.setdefault("reference_energies", metadata.get("reference_energies", {}))
    return HybridACELiftedDensityCalculator(model, type_map=type_map, **kwargs)


__all__ = [
    "ALL_LIFTED_DENSITY_BRANCHES",
    "BRANCH_LIFTED_DENSITY",
    "ASDescriptorInventoryRecord",
    "ASSpechtSlotProjector",
    "HybridACELiftedDensityCalculator",
    "HybridACELiftedDensityConfig",
    "HybridACELiftedDensityEnergyModel",
    "LiftedDensityChannel",
    "LiftedDensityConfig",
    "RoleDensityCache",
    "RoleFilterSpec",
    "RoleResolvedDensity",
    "build_A_s_specht_slot_projectors",
    "build_A_s_slot_permutation_intertwiner_basis",
    "build_A_s_young_subgroup_intertwiner_basis",
    "load_hybrid_ace_lifted_density_ase_bundle",
    "load_hybrid_ace_lifted_density_calculator",
    "lifted_density_channel_blocks",
    "lifted_A_s_descriptor_inventory",
    "normalize_lifted_density_branches",
    "radial_filter_constant_reproduction",
    "save_hybrid_ace_lifted_density_ase_bundle",
    "slot_standard_norm",
    "slot_standard_residual",
    "slot_antisymmetric_squared_volume",
    "slot_specht_tensor_product_multiplicities",
    "slot_specht_matrix_unit_carriers",
    "slot_specht_projected_commutant_quadratic_features",
    "slot_specht_sector_feature_slices",
    "slot_specht_projected_trivial_pairing",
    "slot_specht_trivial_coupling_plan",
    "slot_trivial_component",
    "validate_slot_specht_matrix_unit_carrier_covariance",
    "validate_slot_specht_matrix_units",
    "ye3_power_equivariants",
    "ye3_power_equivariant_sector_metadata",
    "ye3_power_invariants",
    "ye3_power_sector_metadata",
    "ye3_slot_specht_power_invariants",
    "ye3_slot_specht_power_commutant_density_adjoint",
    "ye3_slot_specht_power_equivariant_matrix_unit_carriers",
    "ye3_slot_specht_power_matrix_unit_carriers",
    "ye3_slot_specht_power_projected_carriers",
    "ye3_slot_specht_power_sector_metadata",
    "ye3_quadratic_invariants",
    "ye3_orbit_product_equivariants",
    "ye3_orbit_product_equivariant_sector_metadata",
    "ye3_rank2_orbit_equivariants",
    "ye3_rank2_orbit_equivariant_sector_metadata",
]
