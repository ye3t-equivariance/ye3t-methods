"""Evaluate the promoted Ni tagged model on its LAMMPS reference cell."""

from pathlib import Path

from ase.build import bulk
from ase.calculators.mixing import SumCalculator

from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
from ye3t_methods.atomistic.tagged_cauchy_image import YE3TTaggedCauchyCalculator


paper = Path(__file__).resolve().parent / "lammps" / "Ni"
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "paper_ni_native_reference",
        "status": "publication_reference",
        "system": {"element": "Ni", "lattice": "fcc", "lattice_parameter_A": 3.508,
                   "repeat": (2, 2, 2),
                   "first_atom_displacement_A": (0.08, -0.05, 0.04)},
    },
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "native_cpu", "native_library": None,
                "execution_policy": "direct"},
    "model": {
        "artifact": paper / "models" / "ye3t_tagged_127" / "model.ye3t.json",
        "manifest": paper / "models" / "ye3t_tagged_127" / "model_manifest.json",
    },
    "targets": {"properties": ("energy", "forces", "stress")},
    "validation": {
        "lammps_step_zero_energy_eV": -184.89616285503024,
        "energy_tolerance_eV": 1e-8,
    },
}

system = config["metadata"]["system"]
atoms = bulk(system["element"], system["lattice"],
             a=system["lattice_parameter_A"], cubic=True).repeat(system["repeat"])
atoms.positions[0] += system["first_atom_displacement_A"]
linear = YE3TTaggedCauchyCalculator.from_artifact(
    config["model"]["artifact"],
    native_library=config["runtime"]["native_library"],
    execution_policy=config["runtime"]["execution_policy"],
)
atoms.calc = SumCalculator([
    linear, YE3TZBLCalculator.from_model_manifest(config["model"]["manifest"]),
])
energy = atoms.get_potential_energy()
print("energy_eV", energy)
print("maximum_force_eV_per_A", abs(atoms.get_forces()).max())
print("stress_eV_per_A3", atoms.get_stress())
assert abs(energy - config["validation"]["lammps_step_zero_energy_eV"]) < (
    config["validation"]["energy_tolerance_eV"]
)
