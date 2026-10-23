"""Real paper coordinates, independent derivative checks, and shared row reuse."""
from pathlib import Path

import numpy as np
import pytest
import torch

from ye3t_methods import LinearModel
from ye3t_methods.atomistic.equivariant_calc.edge_geometry import (
    directed_edges_all_images, directed_edges_all_images_bruteforce,
)
from ye3t_methods.portable_refit import _cached_design_row


ARCHIVE = (Path(__file__).resolve().parents[1] / "examples/publication/"
           "cost_comparison/portable_models/Ni_ye3t_tagged_60.ye3t")


@pytest.fixture(scope="module")
def paper_record():
    return LinearModel.read(ARCHIVE)._fitted


@pytest.mark.parametrize("pbc", [(True, True, True), (True, False, True), (False, False, False)])
def test_neighbor_images_and_position_cell_derivatives(pbc):
    positions = torch.tensor([[0.1, 0.2, 0.3], [6.8, -2.1, 1.2]],
                             dtype=torch.float64, requires_grad=True)
    cell = torch.tensor([[2.4, 0.3, 0.1], [0.2, 2.8, 0.0], [0.5, 0.2, 3.1]],
                        dtype=torch.float64, requires_grad=True)
    fast = directed_edges_all_images(positions, 3.5, cell, pbc)
    reference = directed_edges_all_images_bruteforce(positions, 3.5, cell, pbc)
    for actual, expected in zip(fast, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
    # An edge-dependent scalar probes each image's displacement and cell dependency.
    for result in (fast, reference):
        if result[2].numel():
            gradient = torch.autograd.grad((result[2] ** 2).sum(), (positions, cell),
                                           allow_unused=True, retain_graph=True)
            if result is fast:
                expected_gradient = gradient
            else:
                for actual, expected in zip(gradient, expected_gradient):
                    torch.testing.assert_close(actual, expected)


def test_neighbor_inclusive_cutoff_and_empty_geometry():
    for positions in (torch.empty((0, 3), dtype=torch.float64),
                      torch.tensor([[0., 0., 0.], [2., 0., 0.]], dtype=torch.float64)):
        actual = directed_edges_all_images(positions, 2.0)
        expected = directed_edges_all_images_bruteforce(positions, 2.0)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("zero", [False, True])
def test_center_derivatives_match_edge_reference_and_finite_difference(paper_record, zero):
    evaluator = paper_record["tagged_model"].real_evaluator
    channels = paper_record["tagged_model"].channels
    width = max(2 * int(channel["l"]) + 1 for channel in channels)
    generator = torch.Generator().manual_seed(20261022)
    phi = torch.randn((5, len(channels), width), generator=generator, dtype=torch.float64) * 0.1
    if zero:
        phi[:, ::2] = 0
    dphi = torch.randn((*phi.shape, 3), generator=generator, dtype=torch.float64) * 0.1
    edge_index = torch.tensor([[0, 0, 0, 1, 1], [1, 2, 3, 0, 2]])
    def pooled(value):
        return value.new_zeros((4, *value.shape[1:])).index_add_(0, edge_index[0], value)
    actual = evaluator.evaluate_real_with_jacobian(phi, dphi, pooled(phi), edge_index, 4, 2)
    expected = evaluator._evaluate_real_with_jacobian_reference(
        phi, dphi, pooled(phi), edge_index, 4, 113)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)
    step = 1e-5
    delta = torch.zeros_like(phi)
    delta[0] = dphi[0, ..., 1]
    plus = evaluator.evaluate_real(phi + step * delta, pooled(phi + step * delta), edge_index[0], 4)
    minus = evaluator.evaluate_real(phi - step * delta, pooled(phi - step * delta), edge_index[0], 4)
    torch.testing.assert_close(actual[1][0, :, 1], (plus[0] - minus[0]) / (2 * step),
                               rtol=2e-7, atol=1e-9)
    assert torch.count_nonzero(actual[0][2:]) == 0


def test_shared_rows_exclude_targets_and_invalidate_geometry_and_selection(paper_record, tmp_path, monkeypatch):
    from ase import Atoms
    import ye3t_methods.portable_refit as refit
    monkeypatch.setenv("YE3T_ACE_CACHE_DIR", str(tmp_path))
    atoms = Atoms("Ni3", positions=[[0, 0, 0], [2.4, 0.1, 0], [0.2, 2.5, 0.3]])
    row, hit = _cached_design_row(paper_record, atoms, b"paper-selected-columns", {})
    assert not hit
    def unexpected(*args, **kwargs):
        raise AssertionError("A warm row must not invoke descriptor evaluation")
    monkeypatch.setattr(refit, "portable_feature_design_row", unexpected)
    atoms.info["energy"] = -4.2
    atoms.arrays["forces"] = np.ones((3, 3))
    warm, hit = _cached_design_row(paper_record, atoms, b"paper-selected-columns", {})
    assert hit
    for key in row:
        np.testing.assert_array_equal(warm[key], row[key])
    for change in ("positions", "cell", "pbc", "selection"):
        changed = atoms.copy()
        key = b"paper-selected-columns"
        if change == "positions":
            changed.positions[0, 0] += 0.1
        elif change == "cell":
            changed.cell = np.eye(3) * 10
        elif change == "pbc":
            changed.pbc = True
        else:
            key = b"other-selected-columns"
        with pytest.raises(FileNotFoundError, match="read-only"):
            _cached_design_row(paper_record, changed, key, {"mode": "read_only"})
    data = next(tmp_path.rglob("forces.npy"))
    content = np.load(data).copy()
    content[0, 0] += 1
    np.save(data, content)
    with pytest.raises(RuntimeError, match="hash"):
        _cached_design_row(paper_record, atoms, b"paper-selected-columns", {})
