"""Fit rank-four tagged scalar descriptors with Young and angular intermediates.

The bundled Ta labels are deterministic interface fixtures, not reference data
for a physical Ta potential. Replace the input file for a scientific fit.
"""

import json
from pathlib import Path

import numpy as np
from ase.io import read, write
from ye3t import YE3TRepresentation, couplings
from ye3t_methods import Basis, LinearModel


fixtures = Path(__file__).with_name("fixtures")
config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "ta_tagged", "status": "stable",
        "training_structures": str(fixtures / "ta3_training.extxyz"),
        "evaluation_structure": str(fixtures / "ta3_structure.extxyz"),
        "output_path": "../ye3t-workflows/quickstart_linear/ta_tagged_configured.ye3t.json",
    },
    "representation": {
        "group": "O3", "ranks": [4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {4: 2}, "l_max_per_rank": {4: 1},
        },
        "intermediates": {
            "young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"},
        },
    },
    "basis": {
        "single_factors": {
            "species": ["Ta"],
            "radial": {"family": "shifted_jacobi", "cutoff_A": 4.8},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {4: [2]}},
        "catalogue": {
            "ranks": [4], "nmax_per_rank": {4: 2}, "lmax_per_rank": {4: 1},
            "source_block_partitions_by_rank": {4: [[2, 2]]},
            "angular_patterns_by_rank": {4: [[1, 1, 1, 1]]},
        },
    },
    "runtime": {"evaluator": "native_cpu", "neighbors": "ase",
                "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"},
    "model": {
        "kind": "linear",
        "fit": {"solver": "ridge", "alpha": 1e-8,
                "weights": {"energy": 1.0, "forces": 1.0, "stress": 0.1}},
        "reference_energy": {"per_species_E0_eV": {"Ta": 0.0}, "fit_E0": False},
    },
    "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
    "validation": {"checks": ["round_trip"]},
}

structures = read(config["metadata"]["training_structures"], index=":")
representation = YE3TRepresentation.from_config(config["representation"])
preview = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
request = preview.cauchy_compiler_request()
compiled = couplings.compile(couplings.plan(couplings.count(request)))
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
    compiled_cauchy_artifact=compiled,
)
atoms = read(config["metadata"]["evaluation_structure"])
descriptors = basis.create(atoms)
rotated = atoms.copy()
rotated.rotate(37.0, "z", center="COP")
rotated_descriptors = basis.create(rotated)
np.testing.assert_allclose(rotated_descriptors, descriptors,
                           rtol=1e-9, atol=1e-9)
order = [2, 0, 1]
reordered_descriptors = basis.create(atoms[order])
np.testing.assert_allclose(reordered_descriptors, descriptors[order],
                           rtol=1e-9, atol=1e-9)
joint = next(
    (label, raw["label"])
    for label in basis.labels
    for raw in label.as_dict()["compiler_raw_opportunities"]
    if (tuple(tuple(partition) for partition in raw["label"]["block_kappas"])
        == ((1, 1), (1, 1))
        and tuple(raw["label"]["block_Lambdas"]) == (1, 1))
)
model = LinearModel(basis).fit(structures, config=config)
output_path = Path(config["metadata"]["output_path"])
output_path.parent.mkdir(parents=True, exist_ok=True)
artifact = model.write(output_path)
restored = LinearModel.read(artifact)
lammps_model = restored.export_lammps(
    output_path.with_name("ta_tagged_deploy.ye3t.json"))
payload = json.loads(lammps_model.read_text(encoding="utf-8"))
assert "tagged_execution_portfolio" in payload and "self_hash" in payload
data_path = output_path.with_name("ta3.data")
periodic = atoms.copy()
periodic.pbc = True
periodic.wrap()
write(data_path, periodic, format="lammps-data", atom_style="atomic")
input_path = output_path.with_name("in.ta_tagged_auto")
input_path.write_text(
    "units metal\natom_style atomic\nboundary p p p\nnewton on\n"
    f"read_data {data_path.name}\nmass 1 {atoms.get_masses()[0]:.8f}\n"
    "neighbor 0.3 bin\nneigh_modify every 1 delay 0 check yes\n"
    "pair_style ye3t model_family tagged_cauchy block_policy auto\n"
    f"pair_coeff * * {lammps_model.name} "
    f"{config['basis']['single_factors']['species'][0]}\n"
    "thermo_style custom step atoms pe\nthermo_modify norm no\n"
    "run 0\nprint \"energy_eV $(pe:%.16g)\"\n",
    encoding="utf-8",
)
atoms.calc = restored.ase_calculator(
    evaluator=config["runtime"]["evaluator"],
    neighbors=config["runtime"]["neighbors"],
)

print(representation)
print("features", len(basis.labels))
print("core_cauchy_artifact_hash", compiled.self_hash)
print("core_cauchy_request_hash", request["request_hash"])
print("descriptor_rows", descriptors.shape)
print("rotation_max_abs_error", np.max(abs(rotated_descriptors - descriptors)))
print("atom_relabeling_max_abs_error",
      np.max(abs(reordered_descriptors - descriptors[order])))
print("training_structures", len(structures))
print("joint_coordinate", joint[0].feature_index)
print("joint_block_kappas", joint[1]["block_kappas"])
print("joint_block_Lambdas", joint[1]["block_Lambdas"])
print("parent_L", joint[0].as_dict()["L"])
print("roundtrip_features", len(restored.labels))
print("saved_model", artifact)
print("lammps_model", lammps_model)
print("lammps_schema", payload["schema"])
print("lammps_input", input_path)
print("lammps_data", data_path)
print("energy_eV", atoms.get_potential_energy())
print("forces_eV_per_A", atoms.get_forces())
print("stress_eV_per_A3", atoms.get_stress())
