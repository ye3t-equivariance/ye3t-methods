"""Shared edge-geometry helpers for ACE, A_s, and motif prototypes.

These helpers intentionally do not define descriptor mathematics.  They only
construct directed edge displacements, periodic shifts, and ASE stress Voigt
layout data used by several runtimes.  The brute-force edge builders preserve
the legacy Phi/A_s behavior and are kept as a validation fallback; ASE/matscipy
neighbor-list construction should be preferred for calculator and high-throughput
paths.
"""

from itertools import product

import numpy as np
import torch

try:  # Keep ASE optional for non-calculator imports.
    from ase.stress import full_3x3_to_voigt_6_stress as _ase_full_3x3_to_voigt_6_stress
except ImportError:  # pragma: no cover - optional dependency
    _ase_full_3x3_to_voigt_6_stress = None


def normalize_pbc(pbc, device):
    """Return a length-3 torch bool PBC tensor."""

    if pbc is None:
        return torch.zeros(3, dtype=torch.bool, device=device)
    if isinstance(pbc, bool):
        return torch.full((3,), bool(pbc), dtype=torch.bool, device=device)
    values = torch.as_tensor(pbc, dtype=torch.bool, device=device)
    if values.numel() == 1:
        return torch.full((3,), bool(values.item()), dtype=torch.bool, device=device)
    if tuple(values.shape) != (3,):
        raise ValueError("pbc must be a bool or a length-3 boolean sequence.")
    return values


def minimum_image_displacement(displacement, cell=None, pbc=None):
    """Apply the minimum-image convention to Cartesian displacements."""

    if cell is None:
        return displacement
    cell_t = torch.as_tensor(cell, dtype=displacement.dtype, device=displacement.device)
    pbc_t = normalize_pbc(pbc, displacement.device)
    if not bool(torch.any(pbc_t)):
        return displacement
    if tuple(cell_t.shape) != (3, 3):
        raise ValueError("cell must have shape (3, 3) when periodic geometry is requested.")
    inv_cell = torch.linalg.inv(cell_t)
    frac = displacement @ inv_cell
    shift = torch.zeros_like(frac)
    shift[..., pbc_t] = -torch.round(frac[..., pbc_t])
    return displacement + shift @ cell_t


def shortest_periodic_lattice_vector_norm(cell, pbc):
    """Return the shortest periodic lattice-vector norm among nearby images."""

    cell_t = torch.as_tensor(cell)
    pbc_t = normalize_pbc(pbc, cell_t.device)
    if not bool(torch.any(pbc_t)):
        return float("inf")
    axes = [idx for idx, flag in enumerate(pbc_t.detach().cpu().tolist()) if bool(flag)]
    shortest = None
    for coeffs in product((-2, -1, 0, 1, 2), repeat=len(axes)):
        if not any(int(c) != 0 for c in coeffs):
            continue
        shift = torch.zeros(3, dtype=cell_t.dtype, device=cell_t.device)
        for axis, coeff in zip(axes, coeffs, strict=True):
            shift[int(axis)] = float(coeff)
        norm = torch.linalg.norm(shift @ cell_t)
        if shortest is None or bool(norm < shortest):
            shortest = norm
    if shortest is None:
        return float("inf")
    return float(shortest.detach().cpu())


def unique_periodic_cutoff_margin(cell, pbc, cutoff):
    """Margin before a cutoff violates the unique-image half-cell condition."""

    shortest = shortest_periodic_lattice_vector_norm(cell, pbc)
    if not np.isfinite(shortest):
        return float("inf")
    return 0.5 * float(shortest) - float(cutoff)


def directed_edges_bruteforce(positions, cutoff, cell=None, pbc=None):
    """Directed edges from an O(N^2) minimum-image scan.

    This is a clear reference/fallback path.  Calculator and matrix-building
    paths should prefer ASE/matscipy neighbor lists for larger systems.
    """

    atom_count = int(positions.shape[0])
    if atom_count <= 1:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, positions.new_zeros((0, 3)), positions.new_zeros((0,))
    src = []
    dst = []
    for i in range(atom_count):
        for j in range(atom_count):
            if i != j:
                src.append(i)
                dst.append(j)
    src_t = torch.tensor(src, dtype=torch.long, device=positions.device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=positions.device)
    disp = positions.index_select(0, dst_t) - positions.index_select(0, src_t)
    disp = minimum_image_displacement(disp, cell=cell, pbc=pbc)
    dist = torch.linalg.norm(disp, dim=1)
    mask = dist <= float(cutoff)
    return src_t[mask], dst_t[mask], disp[mask], dist[mask]


def periodic_shift_tuples(cell, pbc, cutoff):
    pbc_t = normalize_pbc(pbc, torch.device("cpu"))
    if not bool(torch.any(pbc_t)):
        return ((0.0, 0.0, 0.0),)
    shortest = shortest_periodic_lattice_vector_norm(torch.as_tensor(cell, dtype=torch.float64), pbc_t)
    if not np.isfinite(shortest) or shortest <= 0.0:
        raise ValueError("Cannot enumerate periodic images for a singular or invalid periodic cell.")
    radius = max(1, int(np.ceil(float(cutoff) / float(shortest))) + 1)
    axes = []
    for flag in pbc_t.detach().cpu().tolist():
        axes.append(range(-radius, radius + 1) if bool(flag) else (0,))
    return tuple(tuple(float(x) for x in shift) for shift in product(*axes))


def directed_edges_all_images_bruteforce(positions, cutoff, cell=None, pbc=None):
    """Directed edges from a brute-force periodic image scan."""

    if cell is None or not bool(torch.any(normalize_pbc(pbc, positions.device))):
        return directed_edges_bruteforce(positions, cutoff, cell=cell, pbc=pbc)
    cell_t = torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
    shifts = periodic_shift_tuples(cell_t.detach().cpu(), pbc, cutoff)
    src = []
    dst = []
    shift_rows = []
    atom_count = int(positions.shape[0])
    for i in range(atom_count):
        for j in range(atom_count):
            for shift in shifts:
                if i == j and not any(float(value) != 0.0 for value in shift):
                    continue
                src.append(i)
                dst.append(j)
                shift_rows.append(shift)
    if not src:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, positions.new_zeros((0, 3)), positions.new_zeros((0,))
    src_t = torch.tensor(src, dtype=torch.long, device=positions.device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=positions.device)
    shifts_t = torch.tensor(shift_rows, dtype=positions.dtype, device=positions.device)
    disp = positions.index_select(0, dst_t) - positions.index_select(0, src_t) + shifts_t @ cell_t
    dist = torch.linalg.norm(disp, dim=1)
    mask = dist <= float(cutoff)
    return src_t[mask], dst_t[mask], disp[mask], dist[mask]


def edge_displacements_from_indices(
    positions,
    edge_index,
    *,
    cell=None,
    shifts=None,
    edge_cell=None,
):
    """Return ``r_j - r_i + shifts @ cell`` for explicit directed edges."""

    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=positions.device)
    if edge_index_t.ndim != 2 or tuple(edge_index_t.shape[:1]) != (2,):
        raise ValueError("edge_index must have shape [2, n_edges].")
    if edge_index_t.shape[1] == 0:
        return positions.new_zeros((0, 3))
    src = edge_index_t[0]
    dst = edge_index_t[1]
    disp = positions.index_select(0, dst) - positions.index_select(0, src)
    if shifts is not None:
        if cell is None and edge_cell is None:
            raise ValueError(
                "cell or edge_cell is required when explicit edge shifts "
                "are supplied."
            )
        shifts_t = torch.as_tensor(shifts, dtype=positions.dtype, device=positions.device)
        if shifts_t.ndim != 2 or shifts_t.shape[1] != 3:
            raise ValueError("shifts must have shape [n_edges, 3].")
        if shifts_t.shape[0] != edge_index_t.shape[1]:
            raise ValueError("shifts and edge_index must contain the same number of edges.")
        if edge_cell is not None:
            edge_cell_t = torch.as_tensor(
                edge_cell,
                dtype=positions.dtype,
                device=positions.device,
            )
            if edge_cell_t.shape != (int(edge_index_t.shape[1]), 3, 3):
                raise ValueError(
                    "edge_cell must have shape [n_edges, 3, 3]."
                )
            disp = disp + torch.einsum(
                "ei,eij->ej",
                shifts_t,
                edge_cell_t,
            )
        else:
            cell_t = torch.as_tensor(
                cell,
                dtype=positions.dtype,
                device=positions.device,
            )
            if cell_t.shape != (3, 3):
                raise ValueError(
                    "cell must have shape [3, 3] for explicit edges; use "
                    "edge_cell for batched periodic graphs."
                )
            disp = disp + shifts_t @ cell_t
    return disp


def voigt_from_stress_tensor(stress_tensor):
    """Return ASE Voigt stress order ``xx, yy, zz, yz, xz, xy``."""

    if _ase_full_3x3_to_voigt_6_stress is not None:
        return _ase_full_3x3_to_voigt_6_stress(stress_tensor)
    stress_tensor = np.asarray(stress_tensor, dtype=float)
    return np.asarray(
        [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            0.5 * (stress_tensor[1, 2] + stress_tensor[2, 1]),
            0.5 * (stress_tensor[0, 2] + stress_tensor[2, 0]),
            0.5 * (stress_tensor[0, 1] + stress_tensor[1, 0]),
        ],
        dtype=float,
    )
