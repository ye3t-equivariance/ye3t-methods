"""Run 100 near-equilibrium ASE NVE steps with the saved Ni-127 model.

This is a short calculator/MD example, not a qualification of the potential at
compressed cells. Change the visible crystal, temperature, timestep, or model
path for another run. The CSV contains all 101 energy and temperature samples.
"""

import csv
from pathlib import Path

import numpy as np
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from ye3t_methods import LinearModel


paper = Path(__file__).resolve().parents[1] / "publication" / "cost_comparison"
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "paper_ni_nve", "status": "stable",
        "system": {
            "element": "Ni", "crystal": "fcc", "lattice_parameter_A": 3.508,
            "repeat": (2, 2, 2),
            "first_atom_displacement_A": (0.08, -0.05, 0.04),
        },
        "output_csv": "../ye3t-workflows/quickstart_linear/ni_paper_nve_100.csv",
    },
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "torch", "neighbors": "ase", "device": "cpu"},
    "model": {
        "artifact": paper / "portable_models" / "Ni_ye3t_tagged_127.ye3t",
        "feature_count": 127,
    },
    "targets": {"properties": ("energy", "forces")},
    "validation": {
        "steps": 100, "timestep_fs": 1.0,
        "temperature_K": 300.0, "velocity_seed": 731,
    },
}

system = config["metadata"]["system"]
atoms = bulk(
    system["element"], system["crystal"],
    a=system["lattice_parameter_A"], cubic=True,
).repeat(system["repeat"])
atoms.positions[0] += system["first_atom_displacement_A"]
model = LinearModel.read(config["model"]["artifact"])
assert len(model.labels) == config["model"]["feature_count"]
atoms.calc = model.ase_calculator(
    evaluator=config["runtime"]["evaluator"],
    neighbors=config["runtime"]["neighbors"],
)
MaxwellBoltzmannDistribution(
    atoms, temperature_K=config["validation"]["temperature_K"],
    rng=np.random.default_rng(config["validation"]["velocity_seed"]),
)
Stationary(atoms)
dynamic = VelocityVerlet(
    atoms, timestep=config["validation"]["timestep_fs"] * units.fs,
)

output = Path(config["metadata"]["output_csv"])
output.parent.mkdir(parents=True, exist_ok=True)
energies = []
with output.open("w", newline="", encoding="utf-8") as stream:
    rows = csv.writer(stream)
    rows.writerow(("step", "potential_eV", "kinetic_eV", "total_eV", "temperature_K"))
    for step in range(config["validation"]["steps"] + 1):
        if step:
            dynamic.run(1)
        potential = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        total = potential + kinetic
        if not np.isfinite(total):
            raise RuntimeError(f"Nonfinite energy at step {step}")
        energies.append(total)
        rows.writerow((step, potential, kinetic, total, atoms.get_temperature()))

print("atoms", len(atoms))
print("steps", config["validation"]["steps"])
print("initial_total_eV", energies[0])
print("final_total_eV", energies[-1])
print("maximum_energy_drift_eV_per_atom",
      max(abs(value - energies[0]) for value in energies) / len(atoms))
print("samples_csv", output)
