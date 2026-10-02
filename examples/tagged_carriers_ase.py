"""Evaluate zero-, one-, and two-tag carriers on an ASE Ni crystal."""

from pathlib import Path

from ase.build import bulk

from ye3t_ace import YE3TDescriptors


config = {
    "metadata": {
        "name": "ni_tagged_carriers_ase", "status": "experimental",
        "config_schema": "ye3t_example_config_v1",
        "element": "Ni", "crystal": "fcc",
        "lattice_constant_A": 3.52, "repeat": (2, 2, 2),
    },
    "basis": {
        # The certified shifted-Jacobi radial source uses these cutoffs; it has no radial lambda.
        "type": "tagged_cauchy_carriers", "species": ["Ni"], "cutoff_A": 3.0,
        "pair_cutoffs_A": {"Ni-Ni": 3.0},
        "catalogue": {
            "ranks": (1, 2, 3, 4), "tag_counts": (0, 1, 2),
            "nmax_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
            "lmax_per_rank": {1: 1, 2: 1, 3: 1, 4: 1},
            "input_Lmax": 2,
            "max_source_blocks": 2, "max_features_per_rank": 16,
        },
    },
    "representation": {
        "mode": "tagged_cauchy_carriers", "sector_policy": "tagged_mixed",
    },
    "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                "compiled_cache_dir": Path.home() / ".cache" / "ye3t-methods" / "tagged-carriers"},
    "model": {},
    "targets": {},
    "validation": {"expected_tag_counts": (0, 1, 2),
                   "require_mixed_two_tag_sector": True},
}

atoms = bulk(
    config["metadata"]["element"], config["metadata"]["crystal"],
    a=config["metadata"]["lattice_constant_A"], cubic=True,
).repeat(config["metadata"]["repeat"])
descriptor = YE3TDescriptors.ye3t_basis(config)
features = descriptor.create(atoms)
carriers = features["carriers"]

assert set(config["validation"]["expected_tag_counts"]).issubset(carriers)
mixed_two_tag = sum(label["tag_character"] == -1 for label in carriers[2]["labels"])
if config["validation"]["require_mixed_two_tag_sector"]:
    assert mixed_two_tag > 0
print("atoms", len(atoms))
print("representation", descriptor.representation.basis_mode)
print("compiled_cache_dir", descriptor.metadata["tagged_cauchy_carriers_config"]["compiled_cache_dir"])
print("carrier_shapes", {tags: block["values"].shape for tags, block in carriers.items()})
print("two_tag_mixed_labels", mixed_two_tag)
print("angular_outputs", sorted({label["target_L"] for block in carriers.values()
                                 for label in block["labels"]}))
