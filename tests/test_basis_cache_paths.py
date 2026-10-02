from ye3t_methods import Basis


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
                     for path in density_dir.rglob("*.pkl")}
    assert density_files
    assert len(Basis(**density_settings).labels) == len(density.labels)
    assert density_files == {path.relative_to(density_dir): path.stat().st_mtime_ns
                             for path in density_dir.rglob("*.pkl")}

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
