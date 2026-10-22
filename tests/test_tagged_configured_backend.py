"""Configured tagged rows execute the requested production polynomial backend."""
import copy

import numpy as np
import pytest
import torch
from ase import Atoms

from ye3t import YE3TRepresentation
from ye3t_methods import Basis


@pytest.mark.parametrize("ranks", [(2,), (1, 2)])
@pytest.mark.parametrize("backend", ["auto", "native_cpu"])
def test_configured_native_tagged_rows_and_derivatives(ranks, backend, monkeypatch, tmp_path):
    from ye3t.runtime.execution_plan import native_execution_plan_capabilities
    if not native_execution_plan_capabilities()["cpu"]:
        pytest.skip("requires native ye3t polynomial extension")
    if backend == "native_cpu":
        import os
        from ye3t_methods.atomistic.tagged_cauchy_native import _bundled_native_library_path
        if not (os.environ.get("YE3T_TAGGED_C_API_LIBRARY") or _bundled_native_library_path()):
            pytest.skip("requires optional native tagged C ABI library")
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    rep = YE3TRepresentation.from_config({
        "group": "O3", "ranks": list(ranks),
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {rank: 2 for rank in ranks},
            "l_max_per_rank": {rank: 1 for rank in ranks}},
        "intermediates": {"young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"}},
    })
    config = {
        "single_factors": {"species": ["Ni"],
            "radial": {"family": "shifted_jacobi", "cutoff_A": 4.8},
            "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {rank: [rank] for rank in ranks}},
        "catalogue": {"ranks": list(ranks), "nmax_per_rank": {rank: 2 for rank in ranks},
            "lmax_per_rank": {rank: 1 for rank in ranks},
            "source_block_partitions_by_rank": {rank: [[rank]] for rank in ranks}},
    }
    runtime = {"evaluator": backend, "neighbors": "auto", "cache": {"mode": "auto"},
               "dtype": "float64", "device": "cpu"}
    native = Basis.from_config(config, representation=rep, runtime=runtime)
    reference = Basis.from_config(copy.deepcopy(config), representation=rep,
                                 runtime=dict(runtime, evaluator="reference"))
    atoms = Atoms("Ni4", positions=[[0,0,0], [1.7,.2,.3], [.4,1.5,.1], [-.2,.3,1.6]],
                  cell=[8,8,8], pbc=True)
    np.testing.assert_allclose(native.create(atoms), reference.create(atoms), rtol=1e-11, atol=1e-11)
    assert native.backend == ("native_cpu" if backend == "native_cpu" else "native")
    assert reference.backend == "reference"
    assert [label.as_dict() for label in native.labels] == [label.as_dict() for label in reference.labels]
    actual = native._descriptor.metadata["tagged_cauchy_image_evaluator"]
    expected = reference._descriptor.metadata["tagged_cauchy_image_evaluator"]
    positions = torch.tensor(atoms.positions, dtype=torch.float64)
    types = torch.zeros(len(atoms), dtype=torch.long)
    actual_result = actual.materialize(positions, types, cell=atoms.cell.array, pbc=atoms.pbc)
    expected_result = expected.materialize(positions, types, cell=atoms.cell.array, pbc=atoms.pbc)
    for left, right in zip(actual_result, expected_result, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)
    for evaluator, complete in ((actual, actual_result), (expected, expected_result)):
        values_only = evaluator.materialize(
            positions, types, cell=atoms.cell.array, pbc=atoms.pbc, derivatives=False)
        assert values_only[3] is None
        torch.testing.assert_close(values_only[2], complete[2], rtol=0, atol=0)

    original = actual.evaluate_edge_list
    calls = []
    def check_values_only(*args, **kwargs):
        calls.append(kwargs.get("derivatives"))
        return original(*args, **kwargs)
    monkeypatch.setattr(actual, "evaluate_edge_list", check_values_only)
    shifted = atoms.copy()
    shifted.positions[0, 0] += 0.01
    before = native.create(atoms)
    after = native.create(shifted)
    assert calls == ([] if backend == "native_cpu" else [False, False])
    assert not np.array_equal(before, after)
    values, jacobian = native.create(atoms, derivatives=True)
    np.testing.assert_allclose(values, before, rtol=1e-10, atol=1e-10)
    assert jacobian.shape == (len(atoms), len(native.labels), len(atoms), 3)
    summed_values, summed = native.create(atoms, derivatives="summed")
    np.testing.assert_allclose(summed_values, values)
    np.testing.assert_allclose(summed, jacobian.sum(axis=0), atol=1e-10)
    np.testing.assert_allclose(native.create(atoms, derivatives="full")[1], jacobian)
    np.testing.assert_allclose(jacobian.sum(axis=2), 0, atol=1e-10)
    np.testing.assert_allclose(jacobian.sum(axis=0),
        actual.features_and_position_jacobian(positions, types, atoms.cell.array, atoms.pbc)[1],
        rtol=1e-10, atol=1e-10)
    step = 1e-5
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[1, 2] += step
    minus.positions[1, 2] -= step
    finite_difference = (native.create(plus) - native.create(minus)) / (2 * step)
    np.testing.assert_allclose(jacobian[:, :, 1, 2], finite_difference, rtol=2e-6, atol=2e-7)
    batch = native.create_many([atoms, shifted], derivatives=True)
    np.testing.assert_allclose(batch[0][1], jacobian)
    if backend == "native_cpu":
        from concurrent.futures import ThreadPoolExecutor
        from ye3t_methods.atomistic.tagged_cauchy_image import TaggedCauchyImageLinearModel
        from ye3t_methods.atomistic.tagged_cauchy_native import _TaggedCauchyNativeRuntime
        model = TaggedCauchyImageLinearModel(actual,
            {"Ni": np.ones(actual.feature_count)}, {"Ni": 0.0})
        first = _TaggedCauchyNativeRuntime(model)
        expected_energy = first.evaluate_atoms(atoms)[0]
        np.testing.assert_allclose(first.evaluate_atoms(atoms, features_only=True),
            first.evaluate_atoms(atoms, return_features=True)[-1], rtol=0, atol=0)
        second = _TaggedCauchyNativeRuntime(model)
        assert second.preparation_cache_status == "hit"
        assert second.resident_cache_hit and first.handle == second.handle
        first.close()
        assert second.evaluate_atoms(atoms)[0] == expected_energy
        with ThreadPoolExecutor(2) as pool:
            energies = list(pool.map(lambda _: second.evaluate_atoms(atoms)[0], range(8)))
        assert energies == [expected_energy] * 8
        model.beta_by_species["Ni"][0] += 1.0
        changed = _TaggedCauchyNativeRuntime(model)
        assert changed.handle != second.handle
        assert changed.evaluate_atoms(atoms)[0] != expected_energy
        second.close()
        with pytest.raises(RuntimeError, match="closed"):
            second.evaluate_atoms(atoms)
        changed.close()
        for mode, status in (("off", "off"), ("refresh", "miss")):
            actual.native_cache_mode = mode
            uncached = _TaggedCauchyNativeRuntime(model)
            another = _TaggedCauchyNativeRuntime(model)
            assert not uncached.resident_cache_hit and not another.resident_cache_hit
            assert uncached.handle != another.handle
            assert uncached.preparation_cache_status == status
            assert another.evaluate_atoms(atoms)[0] == uncached.evaluate_atoms(atoms)[0]
            uncached.close()
            another.close()
