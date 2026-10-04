import pytest
import numpy as np
from ase import Atoms

from ye3t.couplings.tagged_cauchy_general import _content_records
from ye3t_methods import Basis


def test_tagged_catalogue_rank_caps_and_cached_compiler(tmp_path):
    settings = {
        "elements": ["Ni"], "source": "tagged_cauchy_image", "cutoff": 4.8,
        "rank": 2, "tag_counts": (0,),
        "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: ((2,),)},
        "pair_cutoffs_A": {"Ni-Ni": 4.8},
        "compiled_cache_dir": tmp_path,
    }
    first = Basis(**settings)
    request = first._descriptor.metadata["tagged_cauchy_image_compiled"].plan.report.request
    assert request["catalogue"]["nmax_per_rank"] == {"2": 1}
    assert request["catalogue"]["lmax_per_rank"] == {"2": 1}
    assert request["catalogue"]["angular_basis_backend"] == "exact_weight_space_v1"
    assert "angular_patterns_by_rank" not in request["catalogue"]
    channels = [channel for record in _content_records(request)
                for channel in record["channels"]]
    assert {channel["radial_channel"] for channel in channels} == {0}
    assert {channel["l"] for channel in channels} == {0, 1}
    assert first.backend == "auto"
    assert first.resolved["angular_basis_backend"] == "exact_weight_space_v1"
    assert first.resolved["pair_cutoffs_A"] == {"Ni-Ni": 4.8}
    assert first.labels and str(first.labels[0])
    cache_files = {path.name: path.stat().st_mtime_ns for path in tmp_path.glob("*.json")}
    assert cache_files
    second = Basis(**settings)
    assert second._descriptor.metadata["tagged_cauchy_image_compiled"].self_hash == (
        first._descriptor.metadata["tagged_cauchy_image_compiled"].self_hash
    )
    assert cache_files == {path.name: path.stat().st_mtime_ns
                           for path in tmp_path.glob("*.json")}
    legacy = Basis(**{**settings, "angular_basis_backend": "legacy_exact",
                      "compiled_cache_dir": tmp_path / "legacy"})
    assert legacy.resolved["angular_basis_backend"] == "legacy_exact"


def test_tagged_catalogue_rejects_inconsistent_rank_caps(tmp_path):
    settings = {
        "elements": ["Ni"], "source": "tagged_cauchy_image", "cutoff": 4.8,
        "rank": 4, "tag_counts": (0, 2),
        "nmax_per_rank": {4: 1}, "lmax_per_rank": {4: 1},
        "source_block_partitions_by_rank": {4: ((4,),)},
        "compiled_cache_dir": tmp_path,
    }
    with pytest.raises(ValueError, match="nmax_per_rank must contain exactly"):
        Basis(**{**settings, "nmax_per_rank": {3: 1}})
    with pytest.raises(ValueError, match="nmax_per_rank must contain exactly"):
        Basis(**{**settings, "nmax_per_rank": {4: 1, "4": 2}})
    with pytest.raises(ValueError, match="nonnegative"):
        Basis(**{**settings, "lmax_per_rank": {4: -1}})
    with pytest.raises(ValueError, match="rank entries"):
        Basis(**{**settings, "angular_patterns_by_rank": {4: ((1, 1),)}})
    with pytest.raises(ValueError, match="without tensor_order"):
        Basis(**{**settings, "tensor_order": 4})
    with pytest.raises(ValueError, match="angular_basis_backend must"):
        Basis(**{**settings, "angular_basis_backend": "numeric_cached"})


@pytest.mark.parametrize("partition,pattern", (
    ((2, 1), (1, 1, 2)),
    ((3,), (2, 2, 2)),
))
def test_rank_three_lmax_two_scalar_basis_is_real(tmp_path, partition, pattern):
    basis = Basis(
        elements=["Ni"], source="tagged_cauchy_image", cutoff=4.8,
        rank=3, tag_counts=(0, 2),
        nmax_per_rank={3: 1}, lmax_per_rank={3: 2},
        source_block_partitions_by_rank={3: (partition,)},
        angular_patterns_by_rank={3: (pattern,)},
        backend="reference", compiled_cache_dir=tmp_path,
    )
    atoms = Atoms("Ni4", positions=((0, 0, 0), (1.5, 0.2, 0.4),
                                  (-0.3, 1.6, 0.7), (0.8, -0.4, 1.8)))
    values = np.asarray(basis.create(atoms))
    assert len(basis.labels) > 0
    assert values.shape == (len(atoms), len(basis.labels))
    assert np.isrealobj(values)
    assert np.isfinite(values).all()
    compiled = basis._descriptor.metadata["tagged_cauchy_image_compiled"]
    assert compiled.payload["real_schedule_core"]["certificate"]["exact_realification"] is True
    assert all(artifact["payload"]["physical_scalar_reality_report"]["exactly_real"]
               for artifact in compiled.payload["parent_artifacts"].values())
