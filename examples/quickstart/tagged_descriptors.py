"""Evaluate scalar tagged YE3T ASE rows with Young and angular intermediates."""

from time import perf_counter

import numpy as np
from ase.build import bulk

from ye3t import YE3TRepresentation
from ye3t_methods import Basis


config = {
    "metadata": {
        "schema": "ye3t_config_v1", "name": "ni_tagged_descriptors",
        "status": "stable",
        "system": {"element": "Ni", "crystal": "fcc", "lattice_parameter_A": 3.52},
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
            "species": ["Ni"],
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
    "runtime": {
        "evaluator": "reference", "neighbors": "auto",
        "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu",
    },
    "model": {"kind": "descriptor_only"},
    "targets": {"output": "per_atom_descriptor_rows"},
    "validation": {
        "displacement_A": 0.05,
        "expected_tag_count": 2,
        "expected_source_block_young": ((1, 1), (1, 1)),
        "expected_block_Lambdas": (1, 1),
    },
}

# The scalar parent is fixed by this tagged source. The source-block Young
# partition below inspects compiler provenance; tag counts select channels.
atoms = bulk(
    config["metadata"]["system"]["element"],
    config["metadata"]["system"]["crystal"],
    a=config["metadata"]["system"]["lattice_parameter_A"],
    cubic=True,
)
displaced = atoms.copy()
displaced.positions[0, 2] += config["validation"]["displacement_A"]
structures = [atoms, displaced]

build_start = perf_counter()
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"],
)
build_seconds = perf_counter() - build_start
eval_start = perf_counter()
rows = [basis.create(structure) for structure in structures]
values = np.concatenate(rows, axis=0)
eval_seconds = perf_counter() - eval_start
row_slices = [slice(0, len(rows[0])), slice(len(rows[0]), len(values))]

opportunities = [
    raw
    for label in basis.labels
    for raw in label.as_dict()["compiler_raw_opportunities"]
]
expected = config["validation"]
witness_labels = [
    label for label in basis.labels
    if any(
        raw["tag_count"] == expected["expected_tag_count"]
        and tuple(tuple(partition) for partition in raw["label"]["block_kappas"])
        == expected["expected_source_block_young"]
        and tuple(raw["label"]["block_Lambdas"])
        == expected["expected_block_Lambdas"]
        for raw in label.as_dict()["compiler_raw_opportunities"]
    )
]
assert witness_labels, "Requested source-block Young type was absent from the compiled basis."
assert np.isfinite(values).all()
assert not np.allclose(rows[0], rows[1]), "The displaced Ni cell did not change its descriptors."

axis = np.asarray((1.0, -0.5, 2.0))
axis /= np.linalg.norm(axis)
cross = np.asarray(((0.0, -axis[2], axis[1]),
                    (axis[2], 0.0, -axis[0]),
                    (-axis[1], axis[0], 0.0)))
angle = 0.47
rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
rotated = displaced.copy()
rotated.positions = displaced.positions @ rotation.T
rotated.set_cell(np.asarray(displaced.cell) @ rotation.T, scale_atoms=False)
rotated_rows = basis.create(rotated)
np.testing.assert_allclose(rotated_rows, rows[1], rtol=1e-9, atol=1e-8)
order = [2, 0, 3, 1]
reordered_rows = basis.create(displaced[order])
np.testing.assert_allclose(reordered_rows, rows[1][order], rtol=1e-9, atol=1e-8)

print("representation:", representation)
print("basis:", basis)
print("descriptor columns:", len(basis.labels))
print("descriptor matrix shape:", values.shape)
print("row slices:", row_slices)
print("global parent Young, L:", representation.parent_partitions[0][1],
      representation.L)
print("tag count and source-block Young:", expected["expected_tag_count"],
      expected["expected_source_block_young"])
print("compiler raw opportunities:", len(opportunities))
print("matching descriptor:", witness_labels[0])
print("block Lambdas:", expected["expected_block_Lambdas"])
print("physical radial family:",
      basis.resolved["components"][0]["single_factors"]["radial"]["family"])
print("requested evaluator:", basis.resolved["runtime"]["evaluator"])
print("rotation max absolute error:", float(np.max(np.abs(rotated_rows - rows[1]))))
print("atom relabel max absolute error:",
      float(np.max(np.abs(reordered_rows - rows[1][order]))))
print(f"basis resolution seconds: {build_seconds:.6f}")
print(f"first materialization and evaluation seconds: {eval_seconds:.6f}")
