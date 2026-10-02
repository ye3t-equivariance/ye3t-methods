"""Load a saved tagged model, evaluate ASE atoms, and export for LAMMPS."""

from pathlib import Path

from ase.io import read
from ye3t_methods import LinearModel


fixtures = Path(__file__).with_name("fixtures")
model = LinearModel.read(fixtures / "ta3_demo.ye3t.json")
atoms = read(fixtures / "ta3_structure.extxyz")
atoms.calc = model.ase_calculator(backend="reference")
print("energy_eV", atoms.get_potential_energy())
print("forces_eV_per_A", atoms.get_forces())
output = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear" / "ta3_deploy.ye3t.json"
output.parent.mkdir(parents=True, exist_ok=True)
print("lammps_model", model.export_lammps(output))
