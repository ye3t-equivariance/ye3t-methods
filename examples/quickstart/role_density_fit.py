"""Fit a small role-density scalar model to manufactured energy/force labels."""

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ye3t_ace import YE3TDescriptors, YE3TModel, YE3TRepresentation
from ye3t_ace.lifted_density import HybridACELiftedDensityCalculator, LiftedDensityChannel


descriptor = YE3TDescriptors.ye3t_basis({
    "elements": ["Ta"],
    "type_map": {"Ta": 0},
    "representation": YE3TRepresentation.filtered_A_s(
        representation_subselection="equivariant", slot_sectors=("trivial",),
    ),
    "lifted_density": {
        "cutoff": 4.0,
        "channels": (LiftedDensityChannel(n=1, l=0, m=0, neighbor_type=0),),
        "filter_kind": "softmax_gaussian",
        "num_filters": 3,
        "filter_width": 0.5,
        "radial_lambda": 0.3,
        "readout_mode": "symmetric_linear",
        "density_normalization": "none",
    },
})
oracle = YE3TModel.lifted_density(descriptor)
with torch.no_grad():
    oracle.channel_readout.fill_(0.7)
    oracle.slot_equivariant_bias.fill_(-0.2)

structures = []
for distance in (1.1, 1.3, 1.5, 1.7):
    atoms = Atoms(
        "Ta3", positions=[[0, 0, 0], [distance, 0.2, 0], [0.2, 1.2, 0.3]],
        cell=[8, 8, 8], pbc=False,
    )
    atoms.calc = HybridACELiftedDensityCalculator(oracle, type_map={"Ta": 0})
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    structures.append(atoms)

model = YE3TModel.linear(
    descriptor,
    {"ridge_alpha": 1e-12, "energy_weight": 1.0, "force_weight": 1.0},
    structures=structures,
)
energy_errors = []
force_errors = []
for atoms in structures:
    candidate = atoms.copy()
    candidate.calc = HybridACELiftedDensityCalculator(model, type_map={"Ta": 0})
    energy_errors.append(abs(candidate.get_potential_energy() - atoms.get_potential_energy()))
    force_errors.append(np.max(np.abs(candidate.get_forces() - atoms.get_forces())))
print("fit_backend", model._ye3t_linear_fit_metadata["backend"])
print("max_energy_error_eV", max(energy_errors))
print("max_force_error_eV_per_A", max(force_errors))
