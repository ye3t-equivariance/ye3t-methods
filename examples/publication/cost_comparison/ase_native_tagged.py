"""Evaluate the promoted Li tagged-plus-ACE model with its ZBL overlay."""

import os
from pathlib import Path

from ase.build import bulk
from ase.calculators.mixing import SumCalculator

from ye3t_methods.atomistic.tagged_cauchy_image import YE3TTaggedCauchyCalculator
from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator


example_root = Path(__file__).resolve().parent
config = {
    "metadata": {
        "name": "paper_li_tagged_native_ase",
        "status": "stable",
    },
    "basis": {
        "species": ["Li"],
        "lattice": "bcc",
        "lattice_constant_A": 3.427,
        "repeats": (2, 2, 2),
        "first_atom_displacement_A": (0.08, -0.05, 0.04),
    },
    "representation": {
        "carrier": "tagged_physical_image_plus_ACE_density",
        "target": {"permutation": "globally_trivial", "L": 0},
        "coupling": {"source": "ye3t.couplings"},
    },
    "runtime": {
        "ase_backend": "native_cpu",
        "native_library": os.environ.get("YE3T_TAGGED_C_API_LIBRARY"),
        "execution_policy": "direct",
    },
    "model": {
        "type": "linear",
        "artifact": example_root / "lammps" / "Li" / "models"
        / "ye3t_tagged_127" / "model.ye3t.json",
        "manifest": example_root / "lammps" / "Li" / "models"
        / "ye3t_tagged_127" / "model_manifest.json",
    },
    "targets": {"properties": ["energy", "forces", "stress"]},
    "validation": {
        "lammps_step_zero_energy_eV": -30.374446585553429,
        "energy_tolerance_eV": 1e-8,
    },
}

if config["runtime"]["ase_backend"] != "native_cpu":
    raise ValueError("The composite paper artifact currently has only a native_cpu ASE loader.")
atoms = bulk(
    config["basis"]["species"][0],
    config["basis"]["lattice"],
    a=config["basis"]["lattice_constant_A"],
    cubic=True,
).repeat(config["basis"]["repeats"])
atoms.positions[0] += config["basis"]["first_atom_displacement_A"]
linear = YE3TTaggedCauchyCalculator.from_artifact(
    config["model"]["artifact"],
    native_library=config["runtime"]["native_library"],
    execution_policy=config["runtime"]["execution_policy"],
)
zbl = YE3TZBLCalculator.from_model_manifest(config["model"]["manifest"])
atoms.calc = SumCalculator([linear, zbl])
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
print("total_energy_eV", energy)
print("linear_residual_eV", linear.results["energy"])
print("zbl_reference_eV", zbl.results["energy"])
print("maximum_force_eV_per_A", abs(forces).max())
print("stress_eV_per_A3", stress)
print("native_policy", linear.native_runtime.selected_policy)
print("neighbor_backend", linear.native_runtime.last_neighbor_backend)
expected = config["validation"]["lammps_step_zero_energy_eV"]
if expected is not None:
    assert abs(energy - expected) < config["validation"]["energy_tolerance_eV"]
