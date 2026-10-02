"""Fit optional LASSO and ARD readouts to a manufactured Cu2 fixture."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {"name": "sklearn_linear_fit_quickstart", "status": "stable"},
    "basis": {"elements": ["Cu"], "source": "density", "cutoff": 3.5,
              "max_rank": 1, "nmax": 1, "lmax": 0},
    "representation": {"carrier": "ACE_density", "target": {"permutation": "trivial", "L": 0},
                       "coupling": {"source": "ye3t.couplings"}},
    "runtime": {"basis_backend": "pytorch", "ase_backend": "pytorch",
                "force_method": "analytic_factorized", "output_root": output_root},
    "model": {"type": "linear", "lasso_params": {"alpha": 1e-9},
              "ard_params": {}},
    "targets": {"energy": "energy", "forces": "forces"},
    "validation": {"fixture": "manufactured Cu2",
                   "checks": ["ASE energy", "ARD atomic readout uncertainty"]},
}

structures = read(fixtures / "cu2_training.extxyz", index=":")
basis = Basis(**config["basis"], backend=config["runtime"]["basis_backend"])
config["runtime"]["output_root"].mkdir(parents=True, exist_ok=True)

lasso = LinearModel(basis).fit(
    structures, fit_method="lasso", sklearn_params=config["model"]["lasso_params"],
)
lasso_path = lasso.write(config["runtime"]["output_root"] / "cu2_lasso.pt")
ard = LinearModel(basis).fit(
    structures, fit_method="ardregression", sklearn_params=config["model"]["ard_params"],
)
ard_path = ard.write(config["runtime"]["output_root"] / "cu2_ard.pt")

restored = LinearModel.read(ard_path)
atoms = structures[0].copy()
atoms.calc = restored.ase_calculator(
    backend=config["runtime"]["ase_backend"],
    force_method=config["runtime"]["force_method"],
)
uncertainty = restored.predict_uncertainty(atoms)
print("lasso_artifact", lasso_path)
print("ard_artifact", ard_path)
print("energy_eV", atoms.get_potential_energy())
print("atomic_energy_std_eV", uncertainty["atomic_energy_std_eV"])
print("total_energy_std_eV", uncertainty["total_energy_std_eV"])
