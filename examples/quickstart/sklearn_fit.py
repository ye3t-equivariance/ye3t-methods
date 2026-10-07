"""Fit configured LASSO and ARD density models, then inspect uncertainty."""

from pathlib import Path

from ase.io import read
from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
output_root = Path(__file__).resolve().parents[2].parent / "ye3t-workflows" / "quickstart_linear"
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "cu_sklearn", "status": "stable",
        "training_structures": str(fixtures / "cu2_training.extxyz"),
        "evaluation_structure": str(fixtures / "cu2_structure.extxyz"),
        "output_path": str(output_root / "cu_sklearn"),
    },
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
            "l_max_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "single_factors": {
            "species": ["Cu"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 3.5,
                       "cutoff_width_A": 0.01, "lambda": 0.79},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "density"},
        "catalogue": {
            "ranks": [1, 2, 3, 4],
            "nmax_per_rank": {1: 2, 2: 2, 3: 2, 4: 2},
            "lmax_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
            "source_block_partitions_by_rank": {
                1: [[1]], 2: [[2]], 3: [[3]], 4: [[4]],
            },
        },
    },
    "runtime": {"evaluator": "torch", "neighbors": "ase",
                "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"},
    "model": {
        "kind": "linear",
        "fit": {"solver": "lasso", "alpha": 1e-9,
                "weights": {"energy": 1.0, "forces": 1.0}},
        "reference_energy": {"per_species_E0_eV": {"Cu": 0.0}, "fit_E0": True},
    },
    "targets": {"energy": "energy", "forces": "forces", "stress": None},
    "validation": {"checks": ["round_trip"]},
}

representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation,
                          runtime=config["runtime"])
structures = read(config["metadata"]["training_structures"], index=":")
output_dir = Path(config["metadata"]["output_path"])
output_dir.mkdir(parents=True, exist_ok=True)

lasso = LinearModel(basis).fit(structures, config=config)
lasso_path = lasso.write(output_dir / "cu2_lasso.pt")
config["model"]["fit"] = {"solver": "ard",
                          "weights": {"energy": 1.0, "forces": 1.0}}
ard = LinearModel(basis).fit(structures, config=config)
ard_path = ard.write(output_dir / "cu2_ard.pt")

restored = LinearModel.read(ard_path)
atoms = read(config["metadata"]["evaluation_structure"])
atoms.calc = restored.ase_calculator(evaluator=config["runtime"]["evaluator"],
                                      neighbors=config["runtime"]["neighbors"])
uncertainty = restored.predict_uncertainty(atoms)
print(representation)
print("features", len(basis.labels))
print("lasso_artifact", lasso_path)
print("ard_artifact", ard_path)
print("energy_eV", atoms.get_potential_energy())
print("atomic_energy_std_eV", uncertainty["atomic_energy_std_eV"])
print("total_energy_std_eV", uncertainty["total_energy_std_eV"])
