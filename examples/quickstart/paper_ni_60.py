"""Use the paper's 60-feature density and tagged Ni potentials in ASE.

Requires the native library from the default installation and the publication
models supplied with the example archive. Both calculators include their fitted
ZBL reference. This evaluates saved fits; refit_paper_ni.py shows the full fit.
"""
from pathlib import Path

from ase.build import bulk
from ase.calculators.mixing import SumCalculator

from ye3t_methods import LinearModel
from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
from ye3t_methods.atomistic.yace_native import YE3TYACENativeCalculator


paper = Path(__file__).resolve().parents[1] / "publication" / "cost_comparison"
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "paper_ni_60", "status": "stable",
        "system": {"element": "Ni", "lattice": "fcc", "lattice_parameter_A": 3.508,
                   "repeat": (2, 2, 2), "first_atom_displacement_A": (0.08, -0.05, 0.04)},
    },
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "native_cpu", "neighbors": "ase", "native_library": None,
                "device": "cpu", "dtype": "float64"},
    "model": {
        "feature_count": 60,
        "density_artifact": paper / "lammps/Ni/models/ace_60/potential.yace",
        "density_manifest": paper / "lammps/Ni/models/ace_60/model_manifest.json",
        "tagged_artifact": paper / "portable_models/Ni_ye3t_tagged_60.ye3t",
        "tagged_native_artifact": paper / "lammps/Ni/models/ye3t_tagged_60/model.ye3t.json",
    },
    "targets": {"properties": ("energy", "forces", "stress")},
    "validation": {"density_energy_eV": -184.97651021876612,
                   "tagged_energy_eV": -184.89170470661099, "energy_tolerance_eV": 1e-8},
}

if config["runtime"]["evaluator"] != "native_cpu":
    raise ValueError("The saved YACE density potential uses native_cpu.")
system = config["metadata"]["system"]
atoms = bulk(system["element"], system["lattice"],
             a=system["lattice_parameter_A"], cubic=True).repeat(system["repeat"])
atoms.positions[0] += system["first_atom_displacement_A"]

# The density fit is stored as YACE, with its reference in the manifest.
density = YE3TYACENativeCalculator.from_artifact(
    config["model"]["density_artifact"], native_library=config["runtime"]["native_library"],
    neighbors=config["runtime"]["neighbors"])
reference = YE3TZBLCalculator.from_model_manifest(config["model"]["density_manifest"])
atoms.calc = SumCalculator([density, reference])
density_energy = atoms.get_potential_energy()
density_forces = atoms.get_forces()
print("density energy / eV", density_energy)
print("density maximum force / eV per Angstrom", abs(density_forces).max())

# The portable tagged fit contains its basis, coefficients and reference.
tagged = LinearModel.read(config["model"]["tagged_artifact"])
basis = tagged.basis
descriptors = basis.create(atoms)
print("tagged descriptor shape", descriptors.shape)  # (32, 60), saved column order
# Its companion native export evaluates the same fit with C++.
tagged_native = LinearModel.read(config["model"]["tagged_native_artifact"])
atoms.calc = tagged_native.ase_calculator(
    evaluator=config["runtime"]["evaluator"], neighbors=config["runtime"]["neighbors"],
    native_library=config["runtime"]["native_library"])
tagged_energy = atoms.get_potential_energy()
tagged_forces = atoms.get_forces()
print("tagged energy / eV", tagged_energy)
print("tagged maximum force / eV per Angstrom", abs(tagged_forces).max())

# Reuse the basis for another ASE object, including a different atom count.
larger = atoms.repeat((2, 1, 1))
larger_descriptors = basis.create(larger)
print("larger tagged descriptor shape", larger_descriptors.shape)  # (64, 60)
