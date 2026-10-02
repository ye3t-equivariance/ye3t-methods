import numpy as np
import pytest
from ase import Atoms

from ye3t_ace import YE3TDescriptors


def test_tagged_carrier_periodic_occurrences_and_cache(tmp_path):
    config = {
        "basis": {
            "type": "tagged_cauchy_carriers", "species": ["Ni"],
            "cutoff_A": 2.0,
            "catalogue": {
                "ranks": (1, 2), "tag_counts": (0, 1, 2),
                "nmax_per_rank": {1: 1, 2: 1},
                "lmax_per_rank": {1: 1, 2: 1}, "input_Lmax": 2,
                "max_source_blocks": 2, "max_features_per_rank": 16,
            },
        },
        "representation": {"mode": "tagged_cauchy_carriers",
                           "sector_policy": "tagged_mixed"},
        "runtime": {"backend": "reference", "device": "cpu", "dtype": "float64",
                    "compiled_cache_dir": tmp_path},
        "model": {},
    }
    atoms = Atoms("Ni", positions=[[0, 0, 0]], cell=[1.5, 8, 8],
                  pbc=[True, False, False])
    descriptor = YE3TDescriptors.ye3t_basis(config)
    first = descriptor.create(atoms)
    evaluator = descriptor.metadata["_tagged_carrier_evaluators"][("cpu", "float64")]
    second = descriptor.create(atoms)
    assert descriptor.metadata["_tagged_carrier_evaluators"][("cpu", "float64")] is evaluator
    np.testing.assert_array_equal(first["carriers"][2]["values"],
                                  second["carriers"][2]["values"])
    assert {tuple(shift) for shift in first["shifts"]} == {(-1, 0, 0), (1, 0, 0)}
    assert first["carriers"][2]["tag_edges"].shape == (2, 2)
    assert np.all(first["carriers"][2]["tag_edges"][:, 0]
                  != first["carriers"][2]["tag_edges"][:, 1])
    assert any(label["tag_character"] == -1
               for label in first["carriers"][2]["labels"])
    two_tag = first["carriers"][2]
    assert {tuple(row) for row in two_tag["tag_edges"]} == {(0, 1), (1, 0)}
    for label in two_tag["labels"]:
        start, end = label["component_slice"]
        np.testing.assert_allclose(
            two_tag["values"][0, start:end],
            label["tag_character"] * two_tag["values"][1, start:end],
            rtol=0, atol=1e-12,
        )
    rotated = atoms.copy()
    rotated.rotate(90, "z", center=(0, 0, 0), rotate_cell=True)
    rotated_carriers = descriptor.create(rotated)["carriers"]
    for tag_count, block in first["carriers"].items():
        transformed = rotated_carriers[tag_count]
        for label, transformed_label in zip(block["labels"], transformed["labels"], strict=True):
            assert label == transformed_label
            start, end = label["component_slice"]
            np.testing.assert_allclose(
                np.sort(np.linalg.norm(block["values"][:, start:end], axis=1)),
                np.sort(np.linalg.norm(transformed["values"][:, start:end], axis=1)),
                rtol=1e-10, atol=1e-12,
            )
    cache_files = {path.name: path.stat().st_mtime_ns for path in tmp_path.glob("*.json")}
    assert cache_files
    YE3TDescriptors.ye3t_basis(config)
    assert cache_files == {
        path.name: path.stat().st_mtime_ns for path in tmp_path.glob("*.json")
    }
    bad_config = {**config, "basis": {**config["basis"], "radial_lambda": 0.3}}
    with pytest.raises(ValueError, match="Unsupported tagged carrier basis settings"):
        YE3TDescriptors.ye3t_basis(bad_config)
