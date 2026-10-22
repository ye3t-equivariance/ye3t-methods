"""Execute public descriptor scripts with their complete scientific settings."""
import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]/"examples"/"quickstart"


@pytest.fixture(scope="module", params=["ase_descriptors.py", "tagged_descriptors.py"])
def scalar_example(request):
    if request.param == "tagged_descriptors.py":
        import os
        from ye3t_methods.atomistic.tagged_cauchy_native import _bundled_native_library_path
        if not (os.environ.get("YE3T_TAGGED_C_API_LIBRARY") or _bundled_native_library_path()):
            pytest.skip("requires optional native tagged C ABI library")
        from ye3t.runtime.execution_plan import native_execution_plan_capabilities
        if not native_execution_plan_capabilities()["cpu"]:
            pytest.skip("requires native ye3t polynomial extension")
    return runpy.run_path(str(ROOT/request.param))


def test_scalar_examples_cover_ranks_and_physical_intermediates(scalar_example):
    basis = scalar_example["basis"]
    config = scalar_example["config"]
    atoms = scalar_example["atoms"].copy()
    atoms.positions[0] += [.07, -.04, .06]
    rows = basis.create(atoms)
    labels = [label.as_dict() for label in basis.labels]
    assert rows.shape == (len(atoms), len(labels))
    assert len(labels) >= 50
    assert {label["N"] for label in labels} == set(range(1, 9))
    for rank in range(1, 9):
        indices = [index for index, label in enumerate(labels) if label["N"] == rank]
        assert np.linalg.norm(rows[:, indices]) > 1e-12
    assert config["representation"]["parent"] == {"young_lambda": "(N)", "L": 0, "parity": "even"}
    if config["basis"]["tensor_product"]["kind"] == "tagged":
        assert basis.backend == "native_cpu"
        assert any(tuple(tuple(kappa) for kappa in raw["label"]["block_kappas"]) == ((1,1),(1,1))
                   and tuple(raw["label"]["block_Lambdas"]) == (1,1)
                   for label in labels for raw in label["compiler_raw_opportunities"])


def test_scalar_example_rotation_inversion_and_atom_permutation(scalar_example):
    basis = scalar_example["basis"]
    atoms = scalar_example["atoms"].copy()
    atoms.positions[0] += [.07, -.04, .06]
    expected = basis.create(atoms)
    rotation = Rotation.from_rotvec([.23,-.31,.17]).as_matrix()
    for matrix in (rotation, -np.eye(3)):
        transformed = atoms.copy()
        transformed.positions = atoms.positions @ matrix.T
        transformed.set_cell(atoms.cell.array @ matrix.T, scale_atoms=False)
        np.testing.assert_allclose(basis.create(transformed), expected, rtol=1e-9, atol=1e-8)
    order = np.array([2,0,3,1])
    np.testing.assert_allclose(basis.create(atoms[order]), expected[order], rtol=1e-9, atol=1e-8)


def test_octupole_example_complete_components_and_covariance():
    import torch
    from ye3t.core.rotation import wigner_D_numeric
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    example = runpy.run_path(str(ROOT/"ase_octupole_descriptors.py"))
    basis, atoms, rows = example["basis"], example["atoms"], example["descriptors"]
    assert rows.shape == (len(atoms), len(basis.labels), 7)
    assert len(basis.labels) > 10
    assert {label.as_dict()["N"] for label in basis.labels} == {1,2,3,4}
    rotation = Rotation.from_rotvec([.23,-.31,.17]).as_matrix()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
    # Use the same tested real-tesseral convention as the public descriptors.
    expected = real_tesseral_to_complex_multiplet(torch.as_tensor(rows), 3).numpy()
    actual = real_tesseral_to_complex_multiplet(torch.as_tensor(basis.create(rotated)), 3).numpy()
    np.testing.assert_allclose(actual, expected @ wigner_D_numeric(3, rotation).T, rtol=1e-9, atol=1e-9)
    inverted = atoms.copy()
    inverted.positions *= -1
    inverted.set_cell(-atoms.cell.array, scale_atoms=False)
    np.testing.assert_allclose(basis.create(inverted), -rows, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('legacy_missing_policy', [False, True])
@pytest.mark.parametrize('backend', ['torch', 'native_cpu'])
def test_density_model_evaluates_the_exact_public_basis(legacy_missing_policy, backend, tmp_path):
    from ye3t_methods import LinearModel
    from ye3t_methods.atomistic.ace.linear_ace import LinearACEScalarModelBundle
    from ye3t_methods.linear import _bind_density_public_labels
    if backend == 'native_cpu':
        import os
        from ye3t_methods.atomistic.tagged_cauchy_native import _bundled_native_library_path
        if not (os.environ.get('YE3T_TAGGED_C_API_LIBRARY') or _bundled_native_library_path()):
            pytest.skip('requires optional native YACE library')
    example = runpy.run_path(str(ROOT/'ase_descriptors.py'))
    basis, atoms = example['basis'], example['atoms'].copy()
    atoms.positions += np.random.default_rng(28).normal(scale=.12, size=atoms.positions.shape)
    rows = basis.create(atoms)
    # Balance contributions so large rank-eight radial values cannot conceal
    # a wrong angular coordinate at a smaller rank.
    weights = np.linspace(.1, .3, len(basis.labels)) / np.maximum(np.max(abs(rows),axis=0),1e-4)
    descriptor = basis._descriptor
    model = LinearModel(basis, reference_energies={'Ni': -.3})
    model._fitted = LinearACEScalarModelBundle(
        settings=descriptor.settings, site_basis_config=descriptor.site_basis_config,
        descriptor_specs=descriptor.descriptor_specs, weight=weights, bias=0.,
        basis_mode=None, fit_method='test', fit_metadata={})
    _bind_density_public_labels(basis, model._fitted)
    if legacy_missing_policy:
        model._fitted.fit_metadata.pop('factorized_descriptor_runtime_policy')
    expected = rows.sum(axis=0) @ weights - .3*len(atoms)
    restored = LinearModel.read(model.write(tmp_path/'density.pt'))
    for fitted in (model, restored):
        atoms.calc = fitted.ase_calculator(evaluator='torch',neighbors='ase')
        np.testing.assert_allclose(atoms.get_potential_energy(), expected, atol=1e-10,rtol=1e-11)
        with pytest.raises(ValueError, match='require their compiler tables'):
            fitted.ase_calculator(evaluator='torch',factorized_descriptor_runtime_policy='auto')
    force = atoms.get_forces()[0,1]
    step = 1e-5
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[0,1] += step
    minus.positions[0,1] -= step
    finite = -(basis.create(plus).sum(axis=0)-basis.create(minus).sum(axis=0))@weights/(2*step)
    np.testing.assert_allclose(force, finite, atol=2e-5,rtol=2e-5)

    if backend == 'native_cpu':
        expected_force = atoms.get_forces()
        expected_stress = atoms.get_stress()
        atoms.calc = restored.ase_calculator(evaluator='native_cpu',neighbors='ase')
        np.testing.assert_allclose(atoms.get_potential_energy(),expected,atol=1e-9,rtol=1e-10)
        np.testing.assert_allclose(atoms.get_forces(),expected_force,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(atoms.get_stress(),expected_stress,atol=1e-9,rtol=1e-8)
