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
        "tensor_order": 4,
        "tag_counts": (0, 2),
        "radial_degrees": (0,),
        "angular_degree": 1,
    },
    "representation": {
        "global_parent_young": (4,),
        "global_parent_L": 0,
        "global_parent_parity": "even",
    },
    "runtime": {"basis_backend": "reference", "compiled_cache_dir": None},
    "model": {"purpose": "descriptors_only"},
    "targets": {"output": "per_atom_descriptor_rows"},
    "validation": {
        "displacement_A": 0.05,
        "expected_internal_tag_young": (1, 1),
        "expected_internal_role_young": (2, 1, 1),
    },
}

# The scalar parent is fixed by this tagged source. The two internal Young
# partitions below inspect compiler provenance; tag_counts selects the channels.
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
    config["basis"]["tensor_order"],
), "The tagged source currently fixes a symmetric global parent."
assert config["representation"]["global_parent_L"] == 0
assert config["representation"]["global_parent_parity"] == "even"

build_start = perf_counter()
basis = Basis(
    **config["basis"], backend=config["runtime"]["basis_backend"],
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
young = config["validation"]
witness_labels = [
    label for label in basis.labels
    if any(
        tuple(raw["tag_kappa"]) == young["expected_internal_tag_young"]
        and tuple(raw["role_kappa"]) == young["expected_internal_role_young"]
        for raw in label.as_dict()["compiler_raw_opportunities"]
    )
]
assert witness_labels, "Requested internal Young types were absent from the compiled basis."
assert np.isfinite(values).all()
assert not np.allclose(rows[0], rows[1]), "The displaced Ni cell did not change its descriptors."

print("basis:", basis)
print("descriptor columns:", len(basis.labels))
print("descriptor matrix shape:", values.shape)
print("row slices:", row_slices)
print("global parent Young, L:", config["representation"]["global_parent_young"],
      config["representation"]["global_parent_L"])
print("internal tag and role Young:", young["expected_internal_tag_young"],
      young["expected_internal_role_young"])
print("compiler raw opportunities:", len(opportunities))
print("matching descriptor:", witness_labels[0])
print("compiled cache directory:", basis.resolved["compiled_cache_dir"])
print(f"descriptor build seconds: {build_seconds:.6f}")
print(f"descriptor evaluation seconds: {eval_seconds:.6f}")
