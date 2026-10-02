def test_stable_public_import_is_available():
    from pathlib import Path

    import ye3t_ace
    import ye3t_methods

    assert ye3t_methods.YE3TDescriptors is not None
    assert Path(ye3t_ace.__file__).resolve().parents[1] == Path(ye3t_methods.__file__).resolve().parents[1]
    package = Path(ye3t_ace.__file__).parent
    assert not (package / "ml").exists()
    assert not (package / "nn").exists()
    assert not (package / "phi_graph.py").exists()
