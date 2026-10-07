"""Fit an ASE site-vector model and prepare the same model for LAMMPS.

The fixture labels below are an analytic, rotation-covariant demonstration,
not measured dipoles or a qualified material property. Run the written
``in.property`` from its output directory with an ML-YE3T-enabled LAMMPS.
"""

from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write
from ase.neighborlist import neighbor_list
from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel


config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "cu_site_vector", "status": "stable",
        "system": {
            "element": "Cu", "lattice": "fcc", "lattice_parameter_A": 3.615,
            "repeat": [2, 2, 2], "training_frames": 4,
            "displacement_A": 0.04, "vector_decay_per_A": 1.0,
            "seed": 7,
        },
        "output_path": "../ye3t-workflows/quickstart_linear/cu_site_vector/model.ye3t.json",
    },
    "representation": {
        "group": "O3", "ranks": [1, 2, 3, 4],
        "parent": {"young_lambda": "(N)", "L": 1, "parity": "odd"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 2, 2: 2, 3: 2, 4: 2},
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
                1: [[1]], 2: [[1, 1]], 3: [[3]], 4: [[3, 1]],
            },
        },
    },
    "runtime": {"evaluator": "torch", "neighbors": "ase",
                "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"},
    "model": {"kind": "linear", "output": {"scope": "per_atom"},
              "fit": {"solver": "ridge", "alpha": 1e-8}},
    "targets": {"per_atom": {"key": "site_vectors", "input": "cartesian",
                              "units": "Angstrom"}},
    "validation": {"checks": ["round_trip"]},
}

system = config["metadata"]["system"]
if config["basis"]["single_factors"]["species"] != [system["element"]]:
    raise ValueError("The ASE system and basis species must agree.")
rng = np.random.default_rng(system["seed"])
structures = []
for _frame in range(system["training_frames"]):
    atoms = bulk(system["element"], system["lattice"],
                 a=system["lattice_parameter_A"], cubic=True)
    atoms = atoms.repeat(tuple(system["repeat"]))
    atoms.positions += rng.normal(scale=system["displacement_A"],
                                  size=atoms.positions.shape)
    structures.append(atoms)
cutoff = config["basis"]["single_factors"]["radial"]["cutoff_A"]
for atoms in structures:
    centers, _neighbors, displacements = neighbor_list("ijD", atoms, cutoff)
    vectors = np.zeros((len(atoms), 3))
    np.add.at(vectors, centers,
              np.exp(-system["vector_decay_per_A"]
                     * np.linalg.norm(displacements, axis=1))[:, None]
              * displacements)
    atoms.new_array("site_vectors", vectors)

representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
model = LinearModel(basis).fit(structures, config=config)
model_path = Path(config["metadata"]["output_path"])
model_path.parent.mkdir(parents=True, exist_ok=True)
model.write(model_path)
restored = LinearModel.read(model_path)

atoms = bulk(system["element"], system["lattice"],
             a=system["lattice_parameter_A"], cubic=True)
atoms = atoms.repeat(tuple(system["repeat"]))
atoms.positions += rng.normal(scale=system["displacement_A"],
                              size=atoms.positions.shape)
prediction = restored.predict(atoms)["mean_real_tesseral"]
np.savetxt(model_path.parent / "reference_real_tesseral.txt", prediction)
write(model_path.parent / "atoms.data", atoms, format="lammps-data",
      atom_style="atomic", masses=True)
(model_path.parent / "in.property").write_text(
    "units metal\n"
    "atom_style atomic\n"
    "boundary p p p\n"
    "read_data atoms.data\n"
    f"pair_style zero {cutoff:.16g}\n"
    "pair_coeff * *\n"
    f"compute property all ye3t/property/atom model.ye3t.json {system['element']}\n"
    "dump values all custom 1 property.dump id type c_property[1] c_property[2] c_property[3]\n"
    "dump_modify values sort id format float %.16g\n"
    "run 0\n",
    encoding="utf-8",
)
print("features", len(basis.labels))
print("saved_model", model_path)
print("prediction_shape", prediction.shape)
print("lammps_input", model_path.parent / "in.property")
