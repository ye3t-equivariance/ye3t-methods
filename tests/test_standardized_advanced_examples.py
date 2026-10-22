"""Public supplied-factor and selected-cluster examples preserve their full output."""
from pathlib import Path
import runpy
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from ye3t.core.rotation import wigner_D_numeric
from ye3t.core.tesseral import real_tesseral_to_complex_multiplet
from ye3t import YE3TRepresentation
from ye3t_methods import Basis

EXAMPLES = Path(__file__).resolve().parents[1]/"examples"


def test_supplied_factors_use_complete_compiled_cauchy_coordinates():
    example = runpy.run_path(str(EXAMPLES/"quickstart/coupled_factors.py"))
    basis, factors, features = example["basis"], example["factors"], example["features"]
    report = basis.catalogue.counts()
    assert features.shape == (1, report["multiplet_count"], report["tableau_count"], 3)
    assert report["multiplet_count"] > 1
    rotation = Rotation.from_rotvec([.3,-.2,.4]).as_matrix()
    matrix = torch.as_tensor(wigner_D_numeric(1, rotation), dtype=factors.dtype)
    actual = basis.create_factors(factors @ matrix.T)
    torch.testing.assert_close(actual, features @ matrix.T, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(basis.create_factors(-factors), -features, rtol=1e-10, atol=1e-10)


def test_selected_phi_cluster_rotates_all_tableau_components():
    example = runpy.run_path(str(EXAMPLES/"advanced/explicit_phi_cluster.py"))
    basis, atoms, cfg = example["basis"], example["atoms"], example["config"]
    system = cfg["metadata"]["system"]
    expected = example["cluster"]
    assert expected.shape == (1,14,5)
    assert np.linalg.norm(expected) > 1e-12
    rotation = Rotation.from_rotvec([.3,-.2,.4]).as_matrix()
    moved = atoms.copy()
    moved.positions = atoms.positions @ rotation.T
    moved.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
    actual = basis.create_cluster(moved, system["center"], system["occurrences"])
    complex_actual = real_tesseral_to_complex_multiplet(torch.as_tensor(actual), 2).numpy()
    complex_expected = real_tesseral_to_complex_multiplet(torch.as_tensor(expected), 2).numpy()
    np.testing.assert_allclose(complex_actual, complex_expected @ wigner_D_numeric(2, rotation).T,
                               rtol=1e-9, atol=1e-9)
    # Both public spellings resolve to the same construction and saved identity.
    from copy import deepcopy
    legacy = deepcopy(cfg["basis"])
    legacy["tensor_product"]["motif"]["ordered_slots"] = legacy["tensor_product"]["motif"].pop("ordered_factors")
    old = Basis.from_config(legacy, representation=YE3TRepresentation.from_config(cfg["representation"]),
                           runtime=cfg["runtime"])
    assert old.resolution.sha256 == basis.resolution.sha256
    legacy["tensor_product"]["motif"]["ordered_factors"] = True
    with pytest.raises(ValueError, match="Specify ordered_factors once"):
        Basis.from_config(legacy, representation=YE3TRepresentation.from_config(cfg["representation"]),
                          runtime=cfg["runtime"])
