"""Protect the retained descriptor-first linear A_s fit after extraction."""

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ye3t_ace import YE3TDescriptors, YE3TModel, YE3TRepresentation
from ye3t_ace.lifted_density import HybridACELiftedDensityCalculator, LiftedDensityChannel


def test_descriptor_first_linear_a_s_fit_recovers_manufactured_predictions():
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
    fitted = YE3TModel.linear(
        descriptor, {"ridge_alpha": 1e-12, "energy_weight": 1.0,
                     "force_weight": 1.0}, structures=structures,
    )
    assert fitted._ye3t_linear_fit_metadata["backend"] == "A_s_streaming_normal_equations_linear_fit"
    for atoms in structures:
        candidate = atoms.copy()
        candidate.calc = HybridACELiftedDensityCalculator(fitted, type_map={"Ta": 0})
        np.testing.assert_allclose(candidate.get_potential_energy(), atoms.get_potential_energy(), atol=1e-8)
        np.testing.assert_allclose(candidate.get_forces(), atoms.get_forces(), atol=1e-8)
