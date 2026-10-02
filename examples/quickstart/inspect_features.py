"""Inspect source basis, actual descriptor column, and saved coefficient."""

from pathlib import Path

from ase.io import read
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
atoms = read(fixtures / "cu2_structure.extxyz")
basis = Basis(elements=["Cu"], cutoff=3.5, max_rank=1, nmax=1, lmax=0)
model = LinearModel.read(fixtures / "cu2_demo.pt")
print(basis)
print("descriptor_shape", basis.create(atoms).shape)
print(basis.labels[0])
print(basis.describe(0))
print(basis.describe(0, format="latex"))
print(model)
print(model.describe(0))
