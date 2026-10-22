"""Regenerate the analytic Cu vector-property fixture used by the ASE example."""
from pathlib import Path
import hashlib
import json
import numpy as np
from ase.build import bulk
from ase.io import write
from ase.neighborlist import neighbor_list

output = Path(__file__).resolve().parents[1]/"examples/quickstart/fixtures"
rng = np.random.default_rng(7)
structures = []
for _ in range(5):
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2,2,2))
    atoms.positions += rng.normal(scale=.04, size=atoms.positions.shape)
    centers, displacements = neighbor_list("iD", atoms, 3.5)
    vectors = np.zeros((len(atoms),3))
    np.add.at(vectors, centers, np.exp(-np.linalg.norm(displacements, axis=1))[:,None]*displacements)
    atoms.new_array("site_vectors", vectors)
    structures.append(atoms)
write(output/"cu_vectors_training.extxyz", structures[:4])
write(output/"cu_vectors_validation.extxyz", structures[4:])
provenance = {"source": "analytic vector sum over neighbors within 3.5 Angstrom",
    "formula": "v_i = sum_j exp(-r_ij / Angstrom) * r_ij", "seed": 7,
    "generator": "tools/generate_vector_example_data.py",
    "files": {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
              for name in ("cu_vectors_training.extxyz", "cu_vectors_validation.extxyz")}}
(output/"cu_vectors.json").write_text(json.dumps(provenance, indent=2)+"\n")
