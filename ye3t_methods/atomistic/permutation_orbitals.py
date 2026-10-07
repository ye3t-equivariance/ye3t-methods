"""Orbital bases for finite permutation-module equivariant maps.

This module implements the orbit-indicator construction for maps
``F^Omega_in -> F^Omega_out`` under an arbitrary finite permutation action.
The source idea is the elementary orbital-basis fact that equivariant matrix
entries are constant on diagonal group orbits in ``Omega_out x Omega_in``.
See the workspace orbital-basis implementation plan for the derivation and
normalization discussion.  The code here constructs permutation-module
intertwiners; it does not decompose arbitrary finite-group representations into
irreducibles.

TODO: add a higher-level representation-theoretic decomposition backend above
this orbital basis.  The orbital basis is useful for equivariant maps between
permutation modules, but full irrep/multiplicity-resolved finite-group
decomposition requires additional representation-specific machinery.
"""

from ye3t_methods.atomistic._record import recordclass
from collections.abc import Mapping, Sequence
from math import sqrt

import torch


def _apply_action(action, item):
    if isinstance(action, Mapping):
        return action[item]
    if callable(action):
        return action(item)
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
        return action[int(item)]
    raise TypeError("group actions must be mappings, callables, or indexable sequences.")


@recordclass(('beta', 'support', 'size'), frozen = True)
class PermutationOrbital:
    """One diagonal-orbit basis element for a permutation-module Hom space."""

    def matrix(self, omega_out, omega_in, *, normalization="frobenius", dtype=None, device=None):
        row_index = {item: idx for idx, item in enumerate(omega_out)}
        col_index = {item: idx for idx, item in enumerate(omega_in)}
        mat = torch.zeros((len(omega_out), len(omega_in)), dtype=dtype or torch.float64, device=device)
        if normalization == "raw":
            scale = 1.0
        elif normalization == "frobenius":
            scale = 1.0 / sqrt(float(self.size))
        else:
            raise ValueError("normalization must be 'raw' or 'frobenius'.")
        for row, col in self.support:
            mat[row_index[row], col_index[col]] = scale
        return mat


@recordclass(('omega_out', 'omega_in', 'group_actions', 'orbitals', 'normalization', 'compiled_report'), frozen = True)
class PermutationOrbitalBasis:
    """Orbital basis for equivariant maps between two permutation modules."""
    normalization = "frobenius"
    compiled_report = None

    def matrices(self, *, dtype=None, device=None):
        return tuple(
            orbital.matrix(
                self.omega_out,
                self.omega_in,
                normalization=self.normalization,
                dtype=dtype,
                device=device,
            )
            for orbital in self.orbitals
        )

    def validate_equivariance(self, *, dtype=None, atol=1.0e-12):
        matrices = self.matrices(dtype=dtype or torch.float64)
        row_index = {item: idx for idx, item in enumerate(self.omega_out)}
        col_index = {item: idx for idx, item in enumerate(self.omega_in)}
        max_residual = 0.0
        for action in self.group_actions:
            out_perm = torch.tensor([row_index[_apply_action(action, item)] for item in self.omega_out], dtype=torch.long)
            in_perm = torch.tensor([col_index[_apply_action(action, item)] for item in self.omega_in], dtype=torch.long)
            for mat in matrices:
                moved = mat.index_select(0, out_perm).index_select(1, in_perm)
                residual = float(torch.max(torch.abs(mat - moved)).item()) if mat.numel() else 0.0
                max_residual = max(max_residual, residual)
        return {
            "passed": bool(max_residual <= float(atol)),
            "orbital_count": int(len(self.orbitals)),
            "max_residual": float(max_residual),
            "normalization": self.normalization,
            "scope": "permutation-module orbital basis equivariance check",
        }


def compute_permutation_orbital_basis(omega_out, omega_in, group_actions, *, normalization="frobenius"):
    """Construct diagonal-orbit basis matrices for arbitrary finite permutation actions."""

    omega_out = tuple(omega_out)
    omega_in = tuple(omega_in)
    group_actions = tuple(group_actions)
    if not omega_out or not omega_in:
        raise ValueError("omega_out and omega_in must be nonempty.")
    if not group_actions:
        raise ValueError("group_actions must contain at least the identity action.")
    if normalization not in {"raw", "frobenius"}:
        raise ValueError("normalization must be 'raw' or 'frobenius'.")

    unseen = {(row, col) for row in omega_out for col in omega_in}
    orbitals = []
    while unseen:
        seed = min(unseen, key=lambda pair: (str(pair[0]), str(pair[1])))
        orbit = set()
        frontier = [seed]
        while frontier:
            pair = frontier.pop()
            if pair in orbit:
                continue
            orbit.add(pair)
            row, col = pair
            for action in group_actions:
                moved = (_apply_action(action, row), _apply_action(action, col))
                if moved not in orbit:
                    frontier.append(moved)
        unseen.difference_update(orbit)
        support = tuple(sorted(orbit, key=lambda pair: (str(pair[0]), str(pair[1]))))
        orbitals.append(PermutationOrbital(beta=len(orbitals), support=support, size=len(support)))
    return PermutationOrbitalBasis(
        omega_out=omega_out,
        omega_in=omega_in,
        group_actions=group_actions,
        orbitals=tuple(orbitals),
        normalization=normalization,
        compiled_report=None,
    )


def permutation_orbital_basis_from_compiled_report(report):
    """Materialize a Torch-facing orbital basis from a ``ye3t.couplings`` report."""

    payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    slot_count = int(payload["slot_count"])
    validation = dict(payload.get("validation_report", {}))
    actions = tuple(
        tuple(int(value) for value in tuple(generator["slot_permutation"]))
        for generator in tuple(validation.get("slot_generators", ()))
    )
    orbitals = []
    for record in tuple(payload["records"]):
        support = tuple(tuple(int(value) for value in tuple(pair)) for pair in tuple(record["support"]))
        orbitals.append(
            PermutationOrbital(
                beta=int(record["beta"]),
                support=support,
                size=int(record["size"]),
            )
        )
    return PermutationOrbitalBasis(
        omega_out=tuple(range(slot_count)),
        omega_in=tuple(range(slot_count)),
        group_actions=actions,
        orbitals=tuple(orbitals),
        normalization=str(payload["normalization"]),
        compiled_report=payload,
    )


__all__ = [
    "PermutationOrbital",
    "PermutationOrbitalBasis",
    "compute_permutation_orbital_basis",
    "permutation_orbital_basis_from_compiled_report",
]
