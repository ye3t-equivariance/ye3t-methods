"""Evaluate tagged YE3T descriptors for two small Ni structures."""

from time import perf_counter

import numpy as np
from ase.build import bulk

from ye3t_methods import Basis


config = {
    "metadata": {
        "name": "ni_tagged_descriptors",
        "system": {"element": "Ni", "crystal": "fcc", "lattice_parameter_A": 3.52},
    },
    "basis": {
        "elements": ["Ni"],
        "source": "tagged_cauchy_image",
        "cutoff": 4.8,
        "pair_cutoffs_A": {"Ni-Ni": 4.8},
        "rank": 4,
        "tag_counts": (0, 2),
        "nmax_per_rank": {4: 1},
        "lmax_per_rank": {4: 1},
        "source_block_partitions_by_rank": {4: ((4,),)},
        "angular_patterns_by_rank": {4: ((1, 1, 1, 1),)},
        "angular_basis_backend": "exact_weight_space_v1",
    },
    "representation": {
        "global_parent_young": (4,),
        "global_parent_L": 0,
        "global_parent_parity": "even",
    },
    "runtime": {"compiled_cache_dir": None},
    "model": {"purpose": "descriptors_only"},
    "targets": {"output": "per_atom_descriptor_rows"},
    "validation": {
        "displacement_A": 0.05,
        "expected_tag_count": 2,
        "expected_source_block_young": (2, 2),
    },
}

# The scalar parent is fixed by this tagged source. The source-block Young
# partition below inspects compiler provenance; tag_counts selects channels.
atoms = bulk(
    config["metadata"]["system"]["element"],
    config["metadata"]["system"]["crystal"],
    a=config["metadata"]["system"]["lattice_parameter_A"],
    cubic=True,
)
displaced = atoms.copy()
displaced.positions[0, 2] += config["validation"]["displacement_A"]
structures = [atoms, displaced]

assert config["representation"]["global_parent_young"] == (
    config["basis"]["rank"],
), "The tagged source currently fixes a symmetric global parent."
assert config["representation"]["global_parent_L"] == 0
assert config["representation"]["global_parent_parity"] == "even"

build_start = perf_counter()
basis = Basis(
    **config["basis"],
    compiled_cache_dir=config["runtime"]["compiled_cache_dir"],
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
        and expected["expected_source_block_young"] in tuple(
            tuple(partition) for partition in raw["label"]["block_kappas"]
        )
        for raw in label.as_dict()["compiler_raw_opportunities"]
    )
]
assert witness_labels, "Requested source-block Young type was absent from the compiled basis."
assert np.isfinite(values).all()
assert not np.allclose(rows[0], rows[1]), "The displaced Ni cell did not change its descriptors."

print("basis:", basis)
print("descriptor columns:", len(basis.labels))
print("descriptor matrix shape:", values.shape)
print("row slices:", row_slices)
print("global parent Young, L:", config["representation"]["global_parent_young"],
      config["representation"]["global_parent_L"])
print("tag count and source-block Young:", expected["expected_tag_count"],
      expected["expected_source_block_young"])
print("compiler raw opportunities:", len(opportunities))
print("matching descriptor:", witness_labels[0])
print("polynomial evaluator backend:", basis.resolved["polynomial_backend"])
print("angular compiler backend:", basis.resolved["angular_basis_backend"])
print("compiled cache directory:", basis.resolved["compiled_cache_dir"])
print(f"descriptor build seconds: {build_seconds:.6f}")
print(f"descriptor evaluation seconds: {eval_seconds:.6f}")
