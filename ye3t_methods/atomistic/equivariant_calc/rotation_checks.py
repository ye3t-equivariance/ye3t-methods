
"""Utilities for practical rotation-invariance/equivariance checks.

These helpers support two descriptor sources:
1. ``DescriptorGenerationSettings`` for enumerating a pool of compact labels and
   expanding them into practical descriptor specifications.
2. An explicit list of compact labels or raw label tuples
   ``[(n_tuple, l_tuple, internal_Ls), ...]``.

The main workflow is:
- generate a random point cloud that spans all octants,
- build a central-atom neighborhood with edge vectors ``r_ji``,
- evaluate one full ``M_R`` multiplet per descriptor,
- rotate the neighbor vectors by a matrix ``R``,
- compare the rotated outputs against the numerical Wigner-D action.
"""

import numpy as np
import torch

from .ace_eval_v2 import ACECovariantEvaluator, GeneralizedCouplingLibrary, compact_label_to_channels_with_mus_ks
from .descriptor_sets import DescriptorCollection, DescriptorGenerationSettings, compile_descriptor_artifacts
from .labeling import CompactLabel, DescriptorSpec, normalize_compact_label
from .neighbors import random_rotation_matrix
from .site_basis_v2 import SiteBasisConfig
from .trc_sph_harm import spherical_harmonics_l


def check_all_octants(points):
    """Return ``True`` if a point cloud spans all 8 octants in 3D."""
    if points.shape[0] < 8:
        return False
    signs = np.sign(points)
    unique_octants = set(map(tuple, signs))
    return len(unique_octants) == 8


def wigner_D_numeric(L_R, R, n_samples = 25):
    """Numerically estimate the complex Wigner-D matrix for rotation ``R``.

    Parameters
    ----------
    L_R
        Resultant angular momentum of the multiplet.
    R
        3x3 proper rotation matrix.
    n_samples
        Number of random directions used to fit the action of ``R`` on the
        spherical-harmonic basis of degree ``L_R``.
    """
    rng = np.random.default_rng(123 + L_R)
    pts = []
    while len(pts) < max(n_samples, 2 * L_R + 3):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pts.append(v)
    pts = np.asarray(pts)
    pts_rot = pts @ R.T

    def ang(v):
        theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
        phi = np.arctan2(v[:, 1], v[:, 0])
        return theta, phi

    th, ph = ang(pts)
    thr, phr = ang(pts_rot)
    Y = spherical_harmonics_l(L_R, torch.tensor(th, dtype=torch.float64), torch.tensor(ph, dtype=torch.float64)).cpu().numpy()
    Yrot = spherical_harmonics_l(L_R, torch.tensor(thr, dtype=torch.float64), torch.tensor(phr, dtype=torch.float64)).cpu().numpy()
    return Yrot @ np.linalg.pinv(Y)


def _default_settings_from_labels(
    labels,
    basis_type = "no_charge",
    elems = ("X",),
):
    """Build a minimal ``DescriptorGenerationSettings`` object from explicit labels.

    This is useful when the user wants to test rotation properties of an explicit
    compact-label list without going through the rank/nmax/lmax generator.
    """
    if not labels:
        raise ValueError("Need at least one label to build default settings")
    L_R = labels[0].L_R
    for lab in labels:
        if lab.L_R != L_R:
            raise ValueError("All labels supplied to the rotation test must have the same L_R")
    ranks = sorted(set(lab.rank for lab in labels))
    nmax = []
    lmax = []
    lmin = []
    k_max = []
    for rank in ranks:
        rank_labels = [lab for lab in labels if lab.rank == rank]
        nmax.append(max(max(lab.n_tuple) for lab in rank_labels))
        lmax.append(max(max(lab.l_tuple) for lab in rank_labels))
        lmin.append(min(min(lab.l_tuple) for lab in rank_labels))
        k_max.append(0)
    M_R_values = tuple(range(-L_R, L_R + 1))
    return DescriptorGenerationSettings(
        ranks=ranks,
        basis_type=basis_type,
        elems=tuple(elems),
        nmax=nmax,
        lmax=lmax,
        lmin=lmin,
        L_R=L_R,
        M_R_values=M_R_values,
        k_o_max=0,
        k_max=k_max,
        max_labels_per_rank=None,
        tree_type=labels[0].tree_type,
    )


def _ensure_full_multiplet(settings):
    """Require a full multiplet of ``M_R`` values whenever ``L_R > 0``."""
    expected = tuple(range(-settings.L_R, settings.L_R + 1))
    if settings.M_R_values is None:
        return
    actual = tuple(settings.M_R_values)
    if settings.L_R > 0 and actual != expected:
        raise ValueError(
            f"To test equivariance for L_R={settings.L_R}, M_R_values must be {expected}; got {actual}."
        )
    if settings.L_R == 0 and actual != (0,):
        raise ValueError(f"For L_R=0, M_R_values must be (0,); got {actual}.")


def make_single_center_neighborhood(points):
    """Build edge data for one central atom with many neighbors.

    Parameters
    ----------
    points
        Array of shape ``[n_neighbors, 3]`` containing neighbor vectors ``r_ji``
        from the central site to the neighbor sites.

    Returns
    -------
    x_ij, edge_index, atom_types
        Torch tensors suitable for ``ACECovariantEvaluator``. The central atom is
        atom 0 and all neighbors are atoms 1..N.
    """
    x_ij = torch.tensor(points, dtype=torch.float64)
    edge_index = torch.tensor([[0] * len(points), list(range(1, len(points) + 1))], dtype=torch.long)
    atom_types = torch.zeros(len(points) + 1, dtype=torch.long)
    return x_ij, edge_index, atom_types


def build_rotation_test_descriptor_collection(
    *,
    settings = None,
    raw_labels = None,
    center_mu_values = None,
    restrict_neighbor_mu = None,
    max_variants_per_label = 1,
):
    """Construct descriptor specs for rotation testing.

    Exactly one of ``settings`` or ``raw_labels`` must be provided.

    Returns
    -------
    settings
        Normalized descriptor-generation settings used for the test.
    compact_labels
        Compact labels included in the descriptor set.
    collection
        ``DescriptorCollection`` containing ``DescriptorSpec`` blocks for each ``M_R``.
    library
        Generalized coupling library backing the descriptor specifications.
    """
    if (settings is None) == (raw_labels is None):
        raise ValueError("Provide exactly one of settings or raw_labels")

    if raw_labels is not None:
        compact_labels = tuple(normalize_compact_label(lab) for lab in raw_labels)
        settings = _default_settings_from_labels(compact_labels)
    else:
        compact_labels = None

    _ensure_full_multiplet(settings)
    compact_labels, library, collection = compile_descriptor_artifacts(
        settings=settings,
        compact_labels=compact_labels,
        center_mu_values=center_mu_values,
        restrict_neighbor_mu=restrict_neighbor_mu,
        max_variants_per_label=max_variants_per_label,
    )
    return settings, list(compact_labels), collection, library


def build_default_rotation_site_basis_config(
    settings,
    *,
    cutoff = 5.0,
    radial_decay = 1.0,
    q_min = None,
    q_max = None,
):
    """Build a simple ``SiteBasisConfig`` suitable for rotation tests.

    The config uses one radial cutoff / decay value for each ordered bond type.
    Charge normalization defaults to ``[-1, 1]`` per element when
    ``basis_type='charge'`` and no explicit bounds are supplied.
    """
    n_types = len(settings.elems)
    n_bonds = n_types * n_types
    if settings.basis_type in {'charge', 'magnetic'}:
        if q_min is None:
            q_min = tuple(-1.0 for _ in range(n_types))
        if q_max is None:
            q_max = tuple(1.0 for _ in range(n_types))
        charge_mode = 'scalar'
        kmax = max(settings.k_max)
    else:
        charge_mode = 'none'
        kmax = 0
    return SiteBasisConfig(
        rc=[cutoff] * n_bonds,
        lmbda=[radial_decay] * n_bonds,
        nradmax=max(settings.nmax),
        lmax=max(settings.lmax),
        kmax=kmax,
        possible_types=tuple(range(n_types)),
        charge_mode=charge_mode,
        q_min=q_min,
        q_max=q_max,
    )


def evaluate_descriptor_multiplets(
    *,
    points,
    settings = None,
    raw_labels = None,
    charges = None,
    site_basis_config = None,
    center_mu_values = None,
    restrict_neighbor_mu = None,
    max_variants_per_label = 1,
):
    """Evaluate one full ``M_R`` multiplet per descriptor on a random neighborhood.

    Parameters
    ----------
    points
        Neighbor vectors ``r_ji`` of shape ``[n_neighbors, 3]``.
    settings / raw_labels
        Either a descriptor generator settings object or an explicit compact-label
        list. Exactly one must be provided.
    charges
        Optional per-atom charges of shape ``[n_neighbors + 1]``. Needed only when
        ``basis_type='charge'``.
    site_basis_config
        Optional explicit basis config. If ``None``, a simple default config is
        constructed from ``settings``.

    Returns
    -------
    settings, compact_labels, M_values, values
        ``values`` has shape ``[n_atoms, n_descriptors, n_M]`` with ``n_atoms``
        equal to ``1 + n_neighbors``. In the usual single-center test, atom 0 is
        the central site and carries the nonzero descriptor values.
    """
    settings, compact_labels, collection, _ = build_rotation_test_descriptor_collection(
        settings=settings,
        raw_labels=raw_labels,
        center_mu_values=center_mu_values,
        restrict_neighbor_mu=restrict_neighbor_mu,
        max_variants_per_label=max_variants_per_label,
    )
    if site_basis_config is None:
        site_basis_config = build_default_rotation_site_basis_config(settings)
    evaluator = ACECovariantEvaluator(site_basis_config)
    x_ij, edge_index, atom_types = make_single_center_neighborhood(points)

    if settings.basis_type == 'charge':
        if charges is None:
            charges = np.zeros(atom_types.shape[0], dtype=float)
        charge_tensor = torch.tensor(charges, dtype=torch.float64)
    else:
        charge_tensor = None

    blocks = []
    for M in settings.M_R_values:
        specs = collection.specs_by_M[int(M)]
        block = evaluator(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=specs,
            charges=charge_tensor,
            real_if_scalar=False,
        )
        blocks.append(block)
    values = torch.stack(blocks, dim=-1).detach().cpu().numpy()
    return settings, compact_labels, tuple(settings.M_R_values), values


def check_descriptor_rotation_equivariance(
    *,
    points,
    rotation,
    settings = None,
    raw_labels = None,
    charges = None,
    site_basis_config = None,
    center_mu_values = None,
    restrict_neighbor_mu = None,
    max_variants_per_label = 1,
):
    """Compare descriptor multiplets before/after rotating the neighbor vectors.

    Returns a dictionary with the descriptor values before rotation, after
    rotation, and the maximum equivariance error under the numerical Wigner-D
    action of ``rotation``.
    """
    points_rot = points @ rotation.T
    settings, labels, M_values, y = evaluate_descriptor_multiplets(
        points=points,
        settings=settings,
        raw_labels=raw_labels,
        charges=charges,
        site_basis_config=site_basis_config,
        center_mu_values=center_mu_values,
        restrict_neighbor_mu=restrict_neighbor_mu,
        max_variants_per_label=max_variants_per_label,
    )
    second_settings = settings if raw_labels is None else None
    second_labels = raw_labels if raw_labels is not None else None
    _, _, _, y_rot = evaluate_descriptor_multiplets(
        points=points_rot,
        settings=second_settings,
        raw_labels=second_labels,
        charges=charges,
        site_basis_config=site_basis_config,
        center_mu_values=center_mu_values,
        restrict_neighbor_mu=restrict_neighbor_mu,
        max_variants_per_label=max_variants_per_label,
    )

    D = wigner_D_numeric(settings.L_R, rotation)
    y0 = y[0]        # central atom
    y0_rot = y_rot[0]
    y0_pred = np.einsum('ab,db->da', D, y0)
    errors = np.max(np.abs(y0_rot - y0_pred), axis=-1)
    return {
        'settings': settings,
        'labels': labels,
        'M_values': M_values,
        'values': y,
        'values_rot': y_rot,
        'values_pred': y0_pred,
        'per_descriptor_errors': errors,
        'max_error': float(np.max(errors)) if len(errors) else 0.0,
    }


__all__ = [
    'check_all_octants',
    'wigner_D_numeric',
    'build_rotation_test_descriptor_collection',
    'build_default_rotation_site_basis_config',
    'evaluate_descriptor_multiplets',
    'check_descriptor_rotation_equivariance',
    'make_single_center_neighborhood',
    'random_rotation_matrix',
]
