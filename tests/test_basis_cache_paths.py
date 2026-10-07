from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from ye3t_methods import Basis


def _build_density_cache_in_process(cache_dir, output, max_rank=2):
    basis = Basis(
        elements=["Cu"], source="density", cutoff=3.5,
        max_rank=max_rank, nmax=1, lmax=0,
        descriptor_cache_dir=cache_dir,
    )
    Path(output).write_text(
        json.dumps({"labels": [str(label) for label in basis.labels]}),
        encoding="utf-8",
    )


def test_density_and_tagged_basis_reuse_persistent_compiler_cache(tmp_path):
    density_dir = tmp_path / "ordinary_density"
    density_settings = {
        "elements": ["Cu"], "source": "density", "cutoff": 3.5,
        "max_rank": 2, "nmax": 1, "lmax": 0,
        "descriptor_cache_dir": density_dir,
    }
    density = Basis(**density_settings)
    assert density.resolved["descriptor_cache_dir"] == str(density_dir)
    density_files = {path.relative_to(density_dir): path.stat().st_mtime_ns
                     for path in density_dir.glob("artifacts/ace_descriptor_*/*.json")}
    assert density_files
    assert not list(density_dir.rglob("*.pkl"))
    assert len(Basis(**density_settings).labels) == len(density.labels)
    assert density_files == {path.relative_to(density_dir): path.stat().st_mtime_ns
                             for path in density_dir.glob("artifacts/ace_descriptor_*/*.json")}

    tagged_dir = tmp_path / "tagged_cauchy_image"
    tagged_settings = {
        "elements": ["Ta"], "source": "tagged_cauchy_image", "cutoff": 4.8,
        "tensor_order": 4, "tag_counts": (0, 2), "radial_degrees": (0,),
        "angular_degree": 1, "backend": "reference",
        "compiled_cache_dir": tagged_dir,
    }
    tagged = Basis(**tagged_settings)
    assert tagged.resolved["compiled_cache_dir"] == str(tagged_dir)
    tagged_files = {path.relative_to(tagged_dir): path.stat().st_mtime_ns
                    for path in tagged_dir.rglob("*.json")}
    assert tagged_files
    assert len(Basis(**tagged_settings).labels) == len(tagged.labels)
    assert tagged_files == {path.relative_to(tagged_dir): path.stat().st_mtime_ns
                            for path in tagged_dir.rglob("*.json")}


def test_safe_descriptor_cache_replays_labels_library_and_specs_without_compiler(
    tmp_path, monkeypatch,
):
    from ye3t_methods.atomistic.cache import DescriptorBuildCache
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1, 2], "basis_type": "no_charge", "elems": ["Cu"],
        "nmax": [1, 1], "lmax": [0, 0], "lmin": [0, 0],
        "L_R": 0, "M_R_values": [0],
    })
    cold = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    assert len(list(tmp_path.glob("artifacts/ace_descriptor_*/*.json"))) == 2
    assert not list(tmp_path.rglob("*.pkl"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Warm descriptor cache recompiled coupling data.")

    monkeypatch.setattr(descriptor_sets, "count_couplings", forbidden)
    monkeypatch.setattr(descriptor_sets, "_compiled_scalar_coordinate_library", forbidden)
    warm = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    assert warm[0] == cold[0]
    assert warm[1].data == cold[1].data
    assert warm[1].metadata == cold[1].metadata
    assert warm[2] == cold[2]

    bounded = DescriptorBuildCache(
        cache_dir=tmp_path,
        max_compact_label_bytes=1,
        max_artifact_bytes=1,
    )
    bounded_replay = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=bounded,
    )
    assert bounded_replay[2] == cold[2]
    stats = bounded.get_stats()
    assert stats["compact_label_entries"] == 0
    assert stats["artifact_entries"] == 0
    assert stats["compact_label_bytes"] == 0
    assert stats["artifact_bytes"] == 0
    assert stats["compact_label_disk_hits"] >= 1
    assert stats["artifact_disk_hits"] >= 1


def test_descriptor_cache_byte_budget_counts_the_key():
    from ye3t_methods.atomistic.cache import DescriptorBuildCache, DescriptorEnumerationCacheKey

    key = DescriptorEnumerationCacheKey(
        settings=("large-key-" * 1000,),
        basis_mode=None,
        exact_primitive_timeout_seconds=None,
    )
    cache = DescriptorBuildCache(max_compact_label_bytes=2048)
    cache.put_compact_labels(key, ())
    assert cache.get_stats()["compact_label_entries"] == 0
    assert cache.get_stats()["compact_label_bytes"] == 0


def test_safe_descriptor_cache_corruption_rebuilds_or_fails_read_only(
    tmp_path, monkeypatch,
):
    from ye3t.cache.artifacts import ArtifactCacheValidationError
    from ye3t_methods.atomistic.cache import DescriptorBuildCache
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1], "basis_type": "no_charge", "elems": ["Cu"],
        "nmax": [1], "lmax": [0], "lmin": [0],
        "L_R": 0, "M_R_values": [0],
    })
    cold = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    artifact = next(tmp_path.glob("artifacts/ace_descriptor_artifacts/*.json"))
    artifact.write_bytes(b'{"partial":')
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    with pytest.raises(ArtifactCacheValidationError):
        descriptor_sets.compile_descriptor_artifacts(
            settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
        )
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    rebuilt = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    assert rebuilt[0] == cold[0]
    assert rebuilt[1].data == cold[1].data
    assert rebuilt[2] == cold[2]
    assert len(list(artifact.parent.glob(artifact.name + ".invalid.*"))) == 1
    monkeypatch.setenv("YE3T_CACHE_MODE", "refresh")
    refreshed = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    assert refreshed[2] == cold[2]
    assert artifact.is_file()


def test_descriptor_cache_numeric_sidecar_is_bound_and_repaired(tmp_path, monkeypatch):
    from ye3t.cache.artifacts import ArtifactCacheValidationError
    from ye3t_methods.atomistic.cache import DescriptorBuildCache
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1, 2], "basis_type": "no_charge", "elems": ["Cu"],
        "nmax": [1, 1], "lmax": [1, 1], "lmin": [0, 0],
        "L_R": 0, "M_R_values": [0],
    })
    cold = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    artifact = next(tmp_path.glob("artifacts/ace_descriptor_artifacts/*.json"))
    payload = json.loads(artifact.read_text(encoding="utf-8"))["payload"]
    inventory = payload["numeric_arrays"]
    assert inventory["inventory"]
    for rank_map in payload["library"]["data"].values():
        for label_map in rank_map.values():
            for coordinate in label_map.values():
                assert set(coordinate["ms_combs"]) == {"__ye3t_npz__"}
                assert "__ye3t_npz__" in coordinate["coeffs"]
    sidecar = tmp_path / "arrays" / (inventory["sha256"] + ".npz")
    assert sidecar.stat().st_size == inventory["bytes"]
    original = sidecar.read_bytes()
    sidecar.write_bytes(b"partial numeric payload")

    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    with pytest.raises(ArtifactCacheValidationError, match="numeric payload"):
        descriptor_sets.compile_descriptor_artifacts(
            settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
        )
    monkeypatch.setenv("YE3T_CACHE_MODE", "auto")
    rebuilt = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    assert rebuilt[0] == cold[0]
    assert rebuilt[1].data == cold[1].data
    assert rebuilt[2] == cold[2]
    assert sidecar.read_bytes() == original
    monkeypatch.setenv("YE3T_CACHE_MODE", "refresh")
    refreshed = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    assert refreshed[1].data == cold[1].data
    assert len(list((tmp_path / "arrays").glob("*.npz"))) == 1


def test_descriptor_cache_losing_writer_uses_published_coefficients(tmp_path):
    from ye3t_methods.atomistic.cache import DescriptorBuildCache
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets
    from ye3t_methods.atomistic.equivariant_calc.ace_eval_v2 import GeneralizedCouplingLibrary

    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1], "basis_type": "no_charge", "elems": ["Cu"],
        "nmax": [1], "lmax": [0], "lmin": [0],
        "L_R": 0, "M_R_values": [0],
    })
    cache = DescriptorBuildCache(cache_dir=tmp_path)
    published = descriptor_sets.compile_descriptor_artifacts(settings, descriptor_cache=cache)
    key = next(iter(cache._artifacts))
    data = deepcopy(published[1].data)
    first = next(iter(next(iter(data.values())).values()))
    coordinate = next(iter(first.values()))
    coordinate["coeffs"][0] = -coordinate["coeffs"][0]
    local_library = GeneralizedCouplingLibrary(
        data, L_R=published[1].L_R, metadata=published[1].metadata)
    local_specs = descriptor_sets.build_descriptor_specs_from_settings(
        published[0], settings, local_library)
    assert local_library.data != published[1].data
    winner = cache.put_artifacts(key, (published[0], local_library, local_specs))
    assert winner[1].data == published[1].data
    assert cache.get_artifacts(key)[1].data == published[1].data
    label_key = next(iter(cache._compact_labels))
    assert cache.put_compact_labels(label_key, ()) == published[0]
    assert cache.get_compact_labels(label_key) == published[0]


def test_descriptor_cache_manifest_separates_legacy_orphans_and_detects_staleness(
    tmp_path,
):
    from ye3t.cache.artifacts import ArtifactCacheValidationError
    from ye3t_methods.atomistic.cache import (
        DescriptorBuildCache,
        load_descriptor_build_cache_manifest,
        write_descriptor_build_cache_manifest,
    )
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1], "basis_type": "no_charge", "elems": ["Cu"],
        "nmax": [1], "lmax": [0], "lmin": [0],
        "L_R": 0, "M_R_values": [0],
    })
    descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path))
    artifact_dir = tmp_path / "artifacts" / "ace_descriptor_artifacts"
    (artifact_dir / "legacy.json").write_text(
        json.dumps({"artifact_schema": "descriptor_build_cache_v2"}), encoding="utf-8")
    (artifact_dir / "damaged.json").write_text("{", encoding="utf-8")
    orphan = b"orphan sidecar after interrupted cache write"
    orphan_path = tmp_path / "arrays" / (hashlib.sha256(orphan).hexdigest() + ".npz")
    orphan_path.write_bytes(orphan)
    written = write_descriptor_build_cache_manifest(tmp_path)
    assert written["entry_count"] == 2
    assert {item["reason"] for item in written["ignored_entries"] if
            item["reason"] == "legacy_schema"} == {"legacy_schema"}
    assert len(written["ignored_entries"]) == 2
    assert [item["cache_file"] for item in written["orphan_numeric_payloads"]] == [
        orphan_path.relative_to(tmp_path).as_posix()]
    saved = load_descriptor_build_cache_manifest(tmp_path)
    assert saved["entry_count"] == 2
    manifest_path = tmp_path / "descriptor_build_cache_manifest.json"
    modified = json.loads(manifest_path.read_text(encoding="utf-8"))
    modified["entry_count"] += 1
    manifest_path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ArtifactCacheValidationError, match="manifest"):
        load_descriptor_build_cache_manifest(tmp_path)
    write_descriptor_build_cache_manifest(tmp_path)
    (artifact_dir / "new_partial.json").write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactCacheValidationError, match="live verified inventory"):
        load_descriptor_build_cache_manifest(tmp_path)


def test_descriptor_cache_manifest_recognizes_uninstalled_source_tree(
    tmp_path, monkeypatch,
):
    import importlib.metadata

    from ye3t_methods.atomistic.cache import DescriptorBuildCache, load_descriptor_build_cache_manifest
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    def unavailable(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", unavailable)
    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1], "basis_type": "no_charge", "elems": ["Cu"],
        "nmax": [1], "lmax": [0], "lmin": [0],
        "L_R": 0, "M_R_values": [0],
    })
    descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path))
    manifest = load_descriptor_build_cache_manifest(tmp_path)
    assert manifest["package_versions"] == {"ye3t": "source-tree",
                                            "ye3t-methods": "source"}
    assert manifest["entry_count"] == 2
    assert manifest["numeric_payload_count"] == 1
    assert not manifest["ignored_entries"]


def test_refresh_recompiles_despite_warm_descriptor_memory(tmp_path, monkeypatch):
    from ye3t_methods.atomistic.cache import DescriptorBuildCache
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1], "basis_type": "no_charge", "elems": ["Cu"],
        "nmax": [1], "lmax": [0], "lmin": [0],
        "L_R": 0, "M_R_values": [0],
    })
    cache = DescriptorBuildCache(cache_dir=tmp_path)
    descriptor_sets.compile_descriptor_artifacts(settings, descriptor_cache=cache)
    assert cache.get_stats()["artifact_entries"] == 1
    monkeypatch.setenv("YE3T_CACHE_MODE", "refresh")

    def forbidden(*args, **kwargs):
        raise AssertionError("Refresh bypassed the warm descriptor cache.")

    monkeypatch.setattr(descriptor_sets, "_compiled_scalar_coordinate_library", forbidden)
    with pytest.raises(AssertionError, match="Refresh bypassed"):
        descriptor_sets.compile_descriptor_artifacts(settings, descriptor_cache=cache)


def test_safe_descriptor_cache_binds_physical_chemistry_and_manifest(tmp_path):
    from ye3t_methods.atomistic.cache import DescriptorBuildCache, load_descriptor_build_cache_manifest
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1, 2], "basis_type": "no_charge", "elems": ["Cu", "Ni"],
        "nmax": [2, 2], "lmax": [0, 0], "lmin": [0, 0],
        "L_R": 0, "M_R_values": [0],
    })
    channels = ((1, 0, 1), (2, 1, 1))
    request = {
        "physical_content_channels": channels,
        "center_mu_values": (0, 1),
        "max_variants_per_label": None,
    }
    cold = descriptor_sets.compile_descriptor_artifacts(
        settings,
        descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
        **request,
    )
    warm = descriptor_sets.compile_descriptor_artifacts(
        settings,
        descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
        **request,
    )
    assert warm[0] == cold[0]
    assert warm[1].data == cold[1].data
    assert warm[2] == cold[2]
    reordered = descriptor_sets.compile_descriptor_artifacts(
        settings,
        descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
        physical_content_channels=tuple(reversed(channels)),
        center_mu_values=(0, 1),
        max_variants_per_label=None,
    )
    assert reordered[2] == cold[2]
    assert len(list(tmp_path.glob("artifacts/ace_descriptor_artifacts/*.json"))) == 1
    swapped = descriptor_sets.compile_descriptor_artifacts(
        settings,
        descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
        physical_content_channels=((1, 1, 1), (2, 0, 1)),
        center_mu_values=(0, 1),
        max_variants_per_label=None,
    )
    assert swapped[0] == cold[0]
    assert swapped[2] != cold[2]
    manifest = load_descriptor_build_cache_manifest(tmp_path)
    assert manifest["cache_schema"] == "descriptor_build_cache_v3"
    assert manifest["entry_count"] == 3
    assert manifest["families"]["compact_labels"]["entry_count"] == 1
    assert manifest["families"]["artifacts"]["entry_count"] == 2
    assert manifest["numeric_payload_count"] >= 1
    assert manifest["numeric_payload_bytes"] > 0
    assert all(Path(item["cache_file"]).stem == item["sha256"]
               for item in manifest["numeric_payloads"])
    assert not manifest["ignored_entries"]
    assert not manifest["orphan_numeric_payloads"]


def test_safe_descriptor_cache_two_process_first_use(tmp_path):
    from ye3t_methods.atomistic.cache import load_descriptor_build_cache_manifest

    cache_dir = tmp_path / "cache"
    outputs = (tmp_path / "one.json", tmp_path / "two.json")
    context = multiprocessing.get_context("spawn")
    processes = tuple(
        context.Process(
            target=_build_density_cache_in_process,
            args=(str(cache_dir), str(output)),
        )
        for output in outputs
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    records = tuple(json.loads(path.read_text(encoding="utf-8")) for path in outputs)
    assert records[0] == records[1]
    assert len(list(cache_dir.glob("artifacts/ace_descriptor_*/*.json"))) == 2
    assert len(list((cache_dir / "arrays").glob("*.npz"))) == 1
    assert load_descriptor_build_cache_manifest(cache_dir)["entry_count"] == 2


def test_legacy_pickle_cache_is_ignored_without_loading(tmp_path):
    from ye3t_methods.atomistic.cache import load_descriptor_build_cache_manifest

    cache_dir = tmp_path / "cache"
    legacy = cache_dir / "descriptor_build_cache" / "compact_labels" / "old.pkl"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"not a pickle")
    basis = Basis(
        elements=["Cu"], source="density", cutoff=3.5,
        max_rank=1, nmax=1, lmax=0,
        descriptor_cache_dir=cache_dir,
    )
    assert basis.labels
    assert legacy.read_bytes() == b"not a pickle"
    assert load_descriptor_build_cache_manifest(cache_dir)["entry_count"] == 2


def test_manifest_tracks_concurrent_distinct_descriptor_requests(tmp_path):
    from ye3t_methods.atomistic.cache import load_descriptor_build_cache_manifest

    cache_dir = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    processes = tuple(
        context.Process(
            target=_build_density_cache_in_process,
            args=(str(cache_dir), str(tmp_path / f"rank{rank}.json"), rank),
        )
        for rank in (1, 2)
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert len(list(cache_dir.glob("artifacts/ace_descriptor_*/*.json"))) == 4
    manifest = load_descriptor_build_cache_manifest(cache_dir)
    assert manifest["entry_count"] == 4
    assert manifest["families"]["compact_labels"]["entry_count"] == 2
    assert manifest["families"]["artifacts"]["entry_count"] == 2


@pytest.mark.parametrize("target_L,M_values", [(0, [0]), (1, [-1, 0, 1])])
def test_safe_descriptor_cache_nontrivial_angular_replay(
    tmp_path, target_L, M_values,
):
    from ye3t_methods.atomistic.cache import DescriptorBuildCache
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    settings = descriptor_sets.DescriptorGenerationSettings.from_dict({
        "ranks": [1, 2], "basis_type": "no_charge", "elems": ["Cu"],
        "nmax": [1, 1], "lmax": [1, 1], "lmin": [0, 0],
        "L_R": target_L, "M_R_values": M_values,
    })
    cold = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    assert any(any(l > 0 for l in label.l_tuple) for label in cold[0])
    assert set(cold[2].specs_by_M) == set(M_values)
    assert all(cold[2].specs_by_M[M] for M in M_values)
    warm = descriptor_sets.compile_descriptor_artifacts(
        settings, descriptor_cache=DescriptorBuildCache(cache_dir=tmp_path),
    )
    assert warm[0] == cold[0]
    assert warm[1].data == cold[1].data
    assert warm[1].metadata == cold[1].metadata
    assert warm[2] == cold[2]
