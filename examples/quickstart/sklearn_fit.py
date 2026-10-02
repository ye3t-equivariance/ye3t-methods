"""Fit optional scikit-learn LASSO and ARD readouts to ASE structures."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {"name": "sklearn_fit", "structures": fixtures / "cu2_training.extxyz"},
    "basis": {"elements": ["Cu"], "source": "density", "cutoff": 3.5,
              "max_rank": 4, "nmax": (2, 2, 2, 2), "lmax": (1, 1, 1, 1)},
    "representation": {"parent_young": "trivial", "parent_L": 0},
    "runtime": {"basis_backend": "pytorch", "ase_backend": "pytorch",
                "force_method": "autograd", "output_dir": output_root},
    "model": {"type": "linear", "lasso_params": {"alpha": 1e-9},
              "ard_params": {}, "fit_methods": ("lasso", "ardregression")},
    "targets": {"energy": "energy", "forces": "forces"},
    "validation": {"structure": fixtures / "cu2_structure.extxyz"},
}

structures = read(config["metadata"]["structures"], index=":")
basis = Basis(**config["basis"], backend=config["runtime"]["basis_backend"])
config["runtime"]["output_dir"].mkdir(parents=True, exist_ok=True)

lasso = LinearModel(basis).fit(
    structures, fit_method=config["model"]["fit_methods"][0],
    sklearn_params=config["model"]["lasso_params"],
    energy_key=config["targets"]["energy"], force_key=config["targets"]["forces"],
)
lasso_path = lasso.write(config["runtime"]["output_dir"] / "cu2_lasso.pt")
ard = LinearModel(basis).fit(
    structures, fit_method=config["model"]["fit_methods"][1],
    sklearn_params=config["model"]["ard_params"],
    energy_key=config["targets"]["energy"], force_key=config["targets"]["forces"],
)
ard_path = ard.write(config["runtime"]["output_dir"] / "cu2_ard.pt")

restored = LinearModel.read(ard_path)
atoms = read(config["validation"]["structure"])
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
