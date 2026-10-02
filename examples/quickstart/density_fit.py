"""Fit a manufactured Cu2 density model; energies are eV and forces eV/A."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {"name": "density_fit_quickstart", "status": "stable"},
    "basis": {"elements": ["Cu"], "source": "density", "cutoff": 3.5,
              "max_rank": 1, "nmax": 1, "lmax": 0},
    "representation": {"carrier": "ACE_density", "target": {"permutation": "trivial", "L": 0},
                       "coupling": {"source": "ye3t.couplings"}},
    "runtime": {"basis_backend": "pytorch", "ase_backend": "pytorch",
                "force_method": "analytic_factorized",
                "output_path": output_root / "cu2_fitted.pt"},
    "model": {"type": "linear", "regularization": 1e-12},
    "targets": {"energy": "energy", "forces": "forces"},
    "validation": {"fixture": "manufactured Cu2", "checks": ["energy", "forces"]},
}
structures = read(fixtures / "cu2_training.extxyz", index=":")
basis = Basis(**config["basis"], backend=config["runtime"]["basis_backend"])
model = LinearModel(basis).fit(structures, regularization=config["model"]["regularization"])
config["runtime"]["output_path"].parent.mkdir(parents=True, exist_ok=True)
artifact = model.write(config["runtime"]["output_path"])
restored = LinearModel.read(artifact)
atoms = structures[0].copy()
atoms.calc = restored.ase_calculator(
    backend=config["runtime"]["ase_backend"],
    force_method=config["runtime"]["force_method"],
)
print(basis)
print(model.describe(0))
print("energy_eV", atoms.get_potential_energy())
print("forces_eV_per_A", atoms.get_forces())
