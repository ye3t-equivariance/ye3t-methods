
import numpy as np
import os
import torch
from ye3t_ace._record import recordclass


@recordclass(('positions', 'cell', 'atom_types', 'edge_index', 'x_ij', 'shifts'))
class NeighborData:
    """Neighbor-list data used by the refactored evaluator.

    shifts stores the periodic-image offsets such that
        x_ij = r_j - r_i + shifts @ cell
    for fixed cell geometry.
    """
    shifts = None


def minimum_image_displacements(positions, cell):
    inv_cell = np.linalg.inv(cell)
    n = positions.shape[0]
    i_idx = []
    j_idx = []
    disp = []
    shifts = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = positions[j] - positions[i]
            frac0 = d @ inv_cell
            shift = -np.round(frac0)
            frac = frac0 + shift
            dmin = frac @ cell
            i_idx.append(i)
            j_idx.append(j)
            disp.append(dmin)
            shifts.append(shift)
    return np.asarray(i_idx, dtype=int), np.asarray(j_idx, dtype=int), np.asarray(disp, dtype=float), np.asarray(shifts, dtype=float)


def brute_force_neighbor_data(
    positions,
    cell,
    atom_types,
    cutoff,
):
    i_idx, j_idx, disp, shifts = minimum_image_displacements(np.asarray(positions, float), np.asarray(cell, float))
    mask = np.linalg.norm(disp, axis=1) <= float(cutoff)
    edge_index = np.vstack([i_idx[mask], j_idx[mask]])
    x_ij = disp[mask]
    return NeighborData(
        positions=np.asarray(positions, float),
        cell=np.asarray(cell, float),
        atom_types=np.asarray(atom_types, int),
        edge_index=edge_index,
        x_ij=x_ij,
        shifts=shifts[mask],
    )


def _resolve_neighbor_backend(backend):
    requested = os.environ.get("YE3T_ACE_NEIGHBOR_BACKEND", backend)
    if requested is None:
        requested = "auto"
    requested = str(requested).strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "ase_neighborlist": "ase",
        "ase.neighborlist": "ase",
        "matscipy.neighbours": "matscipy",
        "matscipy.neighbors": "matscipy",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "ase", "matscipy"}:
        raise ValueError("neighbor backend must be 'auto', 'ase', or 'matscipy'.")
    return requested


def _neighbor_list_from_backend(atoms, cutoff, backend):
    requested = _resolve_neighbor_backend(backend)
    if requested in {"auto", "matscipy"}:
        try:
            from matscipy.neighbours import neighbour_list
            return neighbour_list("ijS", atoms=atoms, cutoff=float(cutoff))
        except ImportError:
            if requested == "matscipy":
                raise ImportError("matscipy is required for neighbor backend 'matscipy'.")
        except Exception:
            if requested == "matscipy":
                raise
    try:
        from ase.neighborlist import neighbor_list
    except ImportError as exc:
        raise ImportError("ASE is required for neighbor_data_from_ase_atoms") from exc
    return neighbor_list("ijS", atoms, float(cutoff))


def neighbor_data_from_ase_atoms(atoms, cutoff, type_map, backend = "ase"):
    i, j, S = _neighbor_list_from_backend(atoms, cutoff, backend)
    cell = np.asarray(atoms.cell.array, float)
    pos = np.asarray(atoms.positions, float)
    shifts = np.asarray(S, float)
    x_ij = (pos[j] - pos[i]) + shifts @ cell
    atom_types = np.asarray([type_map[s] for s in atoms.get_chemical_symbols()], dtype=int)
    return NeighborData(
        positions=pos,
        cell=cell,
        atom_types=atom_types,
        edge_index=np.vstack([i, j]),
        x_ij=x_ij,
        shifts=shifts,
    )


def rotate_positions(positions, R):
    return np.asarray(positions) @ np.asarray(R).T


def random_rotation_matrix(seed = 0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R
