"""Run 100 ASE NVE steps with the fitted 60-feature density Ni model.

Run paper_ni_60_density_01_prep_train.py first. Use evaluator="torch" for
the analytic Python route or "native_cpu" for the installed C++ evaluator.
"""
import os
from pathlib import Path

import numpy as np
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ye3t_methods import LinearModel

workflow = Path(os.environ.get("YE3T_WORKFLOW_ROOT", "~/ye3t-workflows")).expanduser()
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "paper_ni_60_density_md", "status": "stable",
        "system": {"element": "Ni", "crystal": "fcc", "lattice_parameter_A": 3.508,
                   "repeat": (2, 2, 2)},
        "output": str(workflow / "MLIP/ni60_density/nve"),
    },
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "native_cpu", "neighbors": "auto", "device": "cpu"},
    "model": {"artifact": str(workflow / "MLIP/ni60_density/model.ye3t")},
    "targets": {"properties": ["energy", "forces"]},
    "validation": {"steps": 100, "timestep_fs": 1.0, "temperature_K": 300.0, "seed": 731},
}

system = config["metadata"]["system"]
atoms = bulk(system["element"], system["crystal"], a=system["lattice_parameter_A"],
             cubic=True).repeat(system["repeat"])
model = LinearModel.read(config["model"]["artifact"])
atoms.calc = model.ase_calculator(evaluator=config["runtime"]["evaluator"],
                                  neighbors=config["runtime"]["neighbors"])
MaxwellBoltzmannDistribution(atoms, temperature_K=config["validation"]["temperature_K"],
                             rng=np.random.default_rng(config["validation"]["seed"]))
Stationary(atoms)
output = Path(config["metadata"]["output"])
output.parent.mkdir(parents=True, exist_ok=True)
initial = atoms.get_total_energy()
dynamics = VelocityVerlet(atoms, timestep=config["validation"]["timestep_fs"] * units.fs,
                         trajectory=str(output.with_suffix(".traj")),
                         logfile=str(output.with_suffix(".log")))
dynamics.run(config["validation"]["steps"])
print("steps", dynamics.nsteps)
print("final temperature K", atoms.get_temperature())
print("total energy change eV/atom", (atoms.get_total_energy() - initial) / len(atoms))
print("trajectory", output.with_suffix(".traj"))
