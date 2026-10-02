"""Evaluate a saved tagged model in ASE and export it for LAMMPS."""

from pathlib import Path

from ase.io import read
from ye3t_methods import LinearModel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {"name": "saved_ase_export"},
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"ase_backend": "native_cpu", "native_library": None,
                "execution_policy": "direct"},
    "model": {"path": fixtures / "ta3_demo.ye3t.json",
              "lammps_output": output_root / "ta3_deploy.ye3t.json"},
    "targets": {"energy": True, "forces": True},
    "validation": {"structure": fixtures / "ta3_structure.extxyz"},
}

model = LinearModel.read(config["model"]["path"])
atoms = read(config["validation"]["structure"])
# ASE backends: "native_cpu", "native_polynomial", or "reference".
atoms.calc = model.ase_calculator(
    backend=config["runtime"]["ase_backend"],
    native_library=config["runtime"]["native_library"],
    execution_policy=config["runtime"]["execution_policy"],
)
print("energy_eV", atoms.get_potential_energy())
print("forces_eV_per_A", atoms.get_forces())
output = config["model"]["lammps_output"]
output.parent.mkdir(parents=True, exist_ok=True)
print("lammps_model", model.export_lammps(output))
