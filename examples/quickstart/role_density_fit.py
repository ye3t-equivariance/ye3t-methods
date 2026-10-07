"""Fit a lifted-density scalar readout to ASE energies and forces.

Lifted density mainly changes the radial basis; it has limited formal
expressivity in nontrivial permutation sectors.
"""

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ye3t_methods.atomistic import YE3TDescriptors, YE3TModel, YE3TRepresentation
from ye3t_methods.atomistic.lifted_density import HybridACELiftedDensityCalculator, LiftedDensityChannel


config = {
    "metadata": {
        "name": "role_density_fit", "formula": "Ta3",
        "neighbor_distances_A": (1.1, 1.3, 1.5, 1.7),
        "cell_A": [8, 8, 8], "pbc": False,
    },
    "basis": {
        "elements": ["Ta"], "type_map": {"Ta": 0}, "cutoff": 4.0,
        "channels": ({"n": 1, "l": 0, "m": 0, "neighbor_type": 0},),
        "filter_kind": "softmax_gaussian", "num_filters": 3,
        "filter_width": 0.5, "radial_lambda": 0.3,
        "readout_mode": "symmetric_linear", "density_normalization": "none",
    },
    "representation": {"representation_subselection": "equivariant",
                       "slot_sectors": ("trivial",)},
    "runtime": {"backend": "pytorch"},
    "model": {"ridge_alpha": 1e-12, "energy_weight": 1.0,
              "force_weight": 1.0, "oracle_channel_weight": 0.7,
              "oracle_slot_bias": -0.2},
    "targets": {"energy": "energy", "forces": "forces"},
    "validation": {"report_max_errors": True},
}
if config["runtime"]["backend"] != "pytorch":
    raise ValueError("Lifted-density ASE evaluation currently uses PyTorch.")

descriptor = YE3TDescriptors.ye3t_basis({
    "elements": config["basis"]["elements"],
    "type_map": config["basis"]["type_map"],
    "representation": YE3TRepresentation.filtered_A_s(**config["representation"]),
    "lifted_density": {
        **{key: value for key, value in config["basis"].items()
           if key not in {"elements", "type_map", "channels"}},
        "channels": tuple(LiftedDensityChannel(**item)
                          for item in config["basis"]["channels"]),
    },
})
oracle = YE3TModel.lifted_density(descriptor)
with torch.no_grad():
    oracle.channel_readout.fill_(config["model"]["oracle_channel_weight"])
    oracle.slot_equivariant_bias.fill_(config["model"]["oracle_slot_bias"])

structures = []
for distance in config["metadata"]["neighbor_distances_A"]:
    atoms = Atoms(
        config["metadata"]["formula"],
        positions=[[0, 0, 0], [distance, 0.2, 0], [0.2, 1.2, 0.3]],
        cell=config["metadata"]["cell_A"], pbc=config["metadata"]["pbc"],
    )
    atoms.calc = HybridACELiftedDensityCalculator(
        oracle, type_map=config["basis"]["type_map"]
    )
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    structures.append(atoms)

model = YE3TModel.linear(
    descriptor,
    {key: config["model"][key]
     for key in ("ridge_alpha", "energy_weight", "force_weight")},
    structures=structures,
)
energy_errors = []
force_errors = []
for atoms in structures:
    candidate = atoms.copy()
    candidate.calc = HybridACELiftedDensityCalculator(
        model, type_map=config["basis"]["type_map"]
    )
    energy_errors.append(abs(candidate.get_potential_energy() - atoms.get_potential_energy()))
    force_errors.append(np.max(np.abs(candidate.get_forces() - atoms.get_forces())))
print("fit_backend", model._ye3t_linear_fit_metadata["backend"])
print("max_energy_error_eV", max(energy_errors))
print("max_force_error_eV_per_A", max(force_errors))
