"""Inspect a fitted density basis, descriptor column, and coefficient."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
config = {
    "metadata": {"name": "inspect_features",
                 "training_structures": fixtures / "cu2_training.extxyz"},
    "basis": {"elements": ["Cu"], "source": "density", "cutoff": 3.5,
              "max_rank": 1, "nmax": 1, "lmax": 0},
    "representation": {"parent_young": "trivial", "parent_L": 0},
    "runtime": {"basis_backend": "pytorch"},
    "model": {"regularization": 1e-12},
    "targets": {"feature_index": 0, "energy": "energy", "forces": "forces"},
    "validation": {"structure": fixtures / "cu2_structure.extxyz"},
}
atoms = read(config["validation"]["structure"])
basis = Basis(**config["basis"], backend=config["runtime"]["basis_backend"])
structures = read(config["metadata"]["training_structures"], index=":")
model = LinearModel(basis).fit(
    structures, regularization=config["model"]["regularization"],
    energy_key=config["targets"]["energy"], force_key=config["targets"]["forces"],
)
index = config["targets"]["feature_index"]
print(basis)
print("descriptor_shape", basis.create(atoms).shape)
print(basis.labels[index])
print(basis.describe(index))
print(basis.describe(index, format="latex"))
print(model)
print(model.describe(index))
