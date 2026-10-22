"""Export the model fitted by tagged_fit.py and write a LAMMPS AUTO input.

Run tagged_fit.py first, or edit model.path to an existing tagged linear model.
The exported JSON contains the compiler execution plans and AUTO metadata.
Run `lmp -in in.ye3t` from the output directory with ML-YE3T enabled.
"""
import os
from pathlib import Path
from ase.io import read, write
from ye3t_methods import LinearModel

fixtures = Path(__file__).with_name("fixtures")
workflow = Path(os.environ.get("YE3T_WORKFLOW_ROOT", "~/ye3t-workflows")).expanduser()
config = {
    "metadata": {"schema": "ye3t_config_v1", "name": "saved_tagged_export", "status": "stable",
                 "structure": str(fixtures / "ni_validation.extxyz"),
                 "output_directory": str(workflow / 'quickstart_linear/ni_lammps')},
    "basis": {"from_saved_model": True},
    "representation": {"from_saved_model": True},
    "runtime": {"evaluator": "native_cpu", "neighbors": "ase", "dtype": "float64", "device": "cpu"},
    "model": {"path": str(workflow / 'quickstart_linear/ni_tagged.ye3t.json')},
    "targets": {"properties": ["energy", "forces", "stress"]},
    "validation": {},
}

model_path = Path(config["model"]["path"])
if not model_path.is_file():
    raise FileNotFoundError(
        f"Model {model_path} is missing. Run tagged_fit.py first with the same "
        "YE3T_WORKFLOW_ROOT, or set config['model']['path'] to a saved tagged model."
    )
model = LinearModel.read(model_path)
atoms = read(config["metadata"]["structure"])
atoms.calc = model.ase_calculator(evaluator=config["runtime"]["evaluator"],
                                 neighbors=config["runtime"]["neighbors"])
print("energy / eV", atoms.get_potential_energy())
print("largest force component / eV per Angstrom", abs(atoms.get_forces()).max())

output = Path(config["metadata"]["output_directory"])
output.mkdir(parents=True, exist_ok=True)
artifact = model.export_lammps(output / "model.ye3t.json")
write(output / "atoms.data", atoms, format="lammps-data", atom_style="atomic",
      masses=True, specorder=list(model.basis.elements))
(output / "in.ye3t").write_text(
    "units metal\natom_style atomic\nboundary p p p\nnewton on\nread_data atoms.data\n"
    "neighbor 0.3 bin\nneigh_modify every 1 delay 0 check yes\n"
    "pair_style ye3t model_family tagged_cauchy block_policy auto\n"
    f"pair_coeff * * {artifact.name} {' '.join(model.basis.elements)}\n"
    "thermo_style custom step atoms pe\nthermo_modify norm no\nrun 0\n",
    encoding="utf-8",
)
print("LAMMPS model", artifact)
print("LAMMPS input", output / "in.ye3t")
