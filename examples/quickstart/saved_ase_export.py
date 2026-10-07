"""Read a saved tagged model, evaluate it in ASE, and export to LAMMPS."""

from pathlib import Path

import numpy as np
from ase.io import read
from ye3t_methods import LinearModel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {"schema": "ye3t_config_v1", "name": "saved_ase_export",
                 "status": "stable"},
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "reference", "neighbors": "auto"},
    "model": {"path": fixtures / "ta3_demo.ye3t.json",
              "lammps_output": output_root / "ta3_deploy.ye3t.json"},
    "targets": {"energy": True, "forces": True},
    "validation": {"structure": fixtures / "ta3_structure.extxyz",
                   "checks": ["finite_energy", "finite_forces"]},
}

model = LinearModel.read(config["model"]["path"])
atoms = read(config["validation"]["structure"])
# Change evaluator to "native_cpu" after installing the optional native ABI.
atoms.calc = model.ase_calculator(
    evaluator=config["runtime"]["evaluator"],
    neighbors=config["runtime"]["neighbors"],
)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
assert np.isfinite(energy) and np.isfinite(forces).all()
print("energy_eV", energy)
print("forces_eV_per_A", forces)
output = config["model"]["lammps_output"]
output.parent.mkdir(parents=True, exist_ok=True)
print("lammps_model", model.export_lammps(output))
