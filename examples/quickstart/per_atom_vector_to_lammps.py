"""Fit an ASE site-vector model and prepare the same model for LAMMPS.

The fixture labels below are an analytic, rotation-covariant demonstration,
not measured dipoles or a qualified material property. Run the written
``in.property`` from its output directory with an ML-YE3T-enabled LAMMPS.
"""

import os
from pathlib import Path

import numpy as np
from ase.io import read, write
from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
workflow = Path(os.environ.get("YE3T_WORKFLOW_ROOT", "~/ye3t-workflows")).expanduser()
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "cu_site_vector", "status": "stable",
        "training_structures": str(fixtures / "cu_vectors_training.extxyz"),
        "evaluation_structure": str(fixtures / "cu_vectors_validation.extxyz"),
        "output_path": str(workflow / 'quickstart_linear/cu_site_vector/model.ye3t.json'),
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

structures = read(config["metadata"]["training_structures"], index=":")

representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
model = LinearModel(basis).fit(structures, config=config)
model_path = Path(config["metadata"]["output_path"])
model_path.parent.mkdir(parents=True, exist_ok=True)
model.write(model_path)
restored = LinearModel.read(model_path)

atoms = read(config["metadata"]["evaluation_structure"])
cutoff = config["basis"]["single_factors"]["radial"]["cutoff_A"]

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
    f"compute property all ye3t/property/atom {model_path.name} {' '.join(basis.elements)}\n"
    "dump values all custom 1 property.dump id type c_property[1] c_property[2] c_property[3]\n"
    "dump_modify values sort id format float %.16g\n"
    "run 0\n",
    encoding="utf-8",
)
print("features", len(basis.labels))
print("saved_model", model_path)
print("prediction_shape", prediction.shape)
print("lammps_input", model_path.parent / "in.property")
