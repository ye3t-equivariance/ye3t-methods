"""Fit a scalar density model to ASE structures with energies and forces."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {
        "name": "density_fit", "structures": fixtures / "cu2_training.extxyz",
    },
    "basis": {
        "elements": ["Cu"], "source": "density", "cutoff": 3.5,
        "max_rank": 4, "nmax": (2, 2, 2, 2), "lmax": (1, 1, 1, 1),
        "radial_decay": 0.25,
    },
    "representation": {"parent_young": "trivial", "parent_L": 0},
    "runtime": {
        "basis_backend": "pytorch", "ase_backend": "pytorch",
        "force_method": "autograd",
        "output_path": output_root / "cu_density.pt",
    },
    "model": {"type": "linear", "regularization": 1e-8,
              "energy_weight": 1.0, "force_weight": 1.0},
    "targets": {"energy": "energy", "forces": "forces"},
    "validation": {"evaluate_structure": fixtures / "cu2_structure.extxyz"},
}
# Density products have a trivial global Young sector and this API builds L=0 scalars.
if config["representation"] != {"parent_young": "trivial", "parent_L": 0}:
    raise ValueError("This density example supports only the trivial Young sector and L=0.")

structures = read(config["metadata"]["structures"], index=":")
basis = Basis(**config["basis"], backend=config["runtime"]["basis_backend"])
model = LinearModel(basis).fit(
    structures,
    regularization=config["model"]["regularization"],
    energy_weight=config["model"]["energy_weight"],
    force_weight=config["model"]["force_weight"],
    energy_key=config["targets"]["energy"],
    force_key=config["targets"]["forces"],
)
config["runtime"]["output_path"].parent.mkdir(parents=True, exist_ok=True)
artifact = model.write(config["runtime"]["output_path"])
restored = LinearModel.read(artifact)
# "native_cpu" requires a radial basis that can be exported to YACE.
atoms = read(config["validation"]["evaluate_structure"])
calculator_options = {}
if config["runtime"]["ase_backend"] == "pytorch":
    calculator_options["force_method"] = config["runtime"]["force_method"]
atoms.calc = restored.ase_calculator(backend=config["runtime"]["ase_backend"],
                                     **calculator_options)
print(basis)
print("training_structures", len(structures))
print(model.describe(0))
print("saved_model", artifact)
print("energy_eV", atoms.get_potential_energy())
print("forces_eV_per_A", atoms.get_forces())
