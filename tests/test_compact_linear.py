import numpy as np
import pytest

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ye3t_methods import Basis, LinearModel
from ye3t_methods.atomistic.ace.linear_ace import LinearACEScalarCalculator, LinearACEScalarModelBundle


def _density_basis():
    return Basis(elements=["Cu"], source="density", cutoff=3.5,
                 max_rank=1, nmax=1, lmax=0)


def _manufactured_training(basis):
    descriptor = basis._descriptor
    oracle = LinearACEScalarModelBundle(
        settings=descriptor.settings,
        site_basis_config=descriptor.site_basis_config,
        descriptor_specs=descriptor.descriptor_specs,
        weight=np.array([0.7]),
        bias=-0.2,
        basis_mode=None,
        fit_method="manufactured",
    )
    structures = []
    for distance in (1.4, 1.6, 1.8, 2.0):
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [distance, 0.1, 0]],
                      cell=[8, 8, 8], pbc=False)
        atoms.calc = LinearACEScalarCalculator(oracle, basis.cutoff, {"Cu": 0})
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        structures.append(atoms)
    return structures


def test_density_fit_save_load_ase_and_label_identity(tmp_path, monkeypatch):
    basis = _density_basis()
    assert len(basis.labels) == basis.create(Atoms("Cu2", positions=[[0, 0, 0], [1.6, 0, 0]])).shape[1]
    assert len(basis.labels) == 1
    assert basis.labels[0].as_dict()["feature_index"] == 0
    assert basis.labels[0].as_dict()["L"] == 0
    assert basis.describe(0, format="latex").startswith("B_{0}")

    import ye3t.couplings

    def forbidden(*args, **kwargs):
        raise AssertionError("Display must not compile couplings")

    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", forbidden)
        assert "features=1" in str(basis)
        assert "B N=1" in basis.describe(0)

    training = _manufactured_training(basis)
    model = LinearModel(basis).fit(training, regularization=1e-12)
    np.testing.assert_allclose(model._fitted.weight, [0.7], atol=1e-8)
    np.testing.assert_allclose(model._fitted.bias, -0.2, atol=1e-8)
    assert "fitted" in str(model)
    assert "coefficient:" in model.describe(0)
    for atoms in training:
        inference = atoms.copy()
        inference.calc = model.ase_calculator()
        np.testing.assert_allclose(inference.get_potential_energy(), atoms.get_potential_energy(), atol=1e-8)
        np.testing.assert_allclose(inference.get_forces(), atoms.get_forces(), atol=1e-8)

    path = model.write(tmp_path / "cu_model")
    assert path.suffix == ".pt"
    moved = tmp_path / "relocated" / path.name
    moved.parent.mkdir()
    path.rename(moved)
    loaded = LinearModel.read(moved)
    assert loaded.labels[0].identity == model.labels[0].identity
    assert loaded.describe(0, format="latex") == model.describe(0, format="latex")
    inference = training[0].copy()
    inference.calc = loaded.ase_calculator()
    np.testing.assert_allclose(inference.get_forces(), training[0].get_forces(), atol=1e-8)


def test_density_sklearn_lasso_and_ard_uncertainty_survive_reload(tmp_path):
    pytest.importorskip("sklearn")
    basis = _density_basis()
    training = _manufactured_training(basis)
    lasso = LinearModel(basis).fit(
        training, fit_method="lasso", sklearn_params={"alpha": 1e-9},
    )
    assert lasso._fitted.fit_method == "lasso"
    assert 0.69 < float(lasso._fitted.weight[0]) < 0.7  # L1 shrinkage.
    with pytest.raises(ValueError, match="ARDRegression"):
        lasso.predict_uncertainty(training[0])

    ard = LinearModel(basis).fit(training, fit_method="ardregression")
    query = training[0]
    result = ard.predict_uncertainty(query)
    assert result["atomic_energy_std_eV"].shape == (len(query),)
    assert np.isfinite(result["atomic_energy_std_eV"]).all()
    assert result["total_energy_std_eV"] >= 0.0
    posterior = ard._fitted.fit_metadata["predictive_uncertainty"]
    assert posterior["variance_formula"] == "x_active @ sigma @ x_active.T (epistemic readout only)"
    design = np.column_stack((basis.create(query), np.ones(len(query))))
    active = posterior["active_column_indices"]
    covariance = np.asarray(posterior["coefficient_covariance_active"])
    selected = design[:, active]
    expected = np.sqrt(np.einsum("if,fg,ig->i", selected, covariance, selected))
    np.testing.assert_allclose(result["atomic_energy_std_eV"], expected, rtol=1e-11)
    expected_total = np.sqrt(selected.sum(axis=0) @ covariance @ selected.sum(axis=0))
    np.testing.assert_allclose(result["total_energy_std_eV"], expected_total, rtol=1e-11)
    loaded = LinearModel.read(ard.write(tmp_path / "density_ard"))
    np.testing.assert_allclose(
        loaded.predict_uncertainty(query)["atomic_energy_std_eV"], expected, rtol=1e-11,
    )

    pruned = LinearModel(basis).fit(
        training, fit_method="ardregression",
        sklearn_params={"threshold_lambda": 0.0},
    )
    assert pruned._fitted.fit_metadata["predictive_uncertainty"]["active_column_indices"] == []
    pruned_loaded = LinearModel.read(pruned.write(tmp_path / "density_ard_pruned"))
    pruned_result = pruned_loaded.predict_uncertainty(query)
    np.testing.assert_array_equal(pruned_result["atomic_energy_std_eV"], np.zeros(len(query)))
    assert pruned_result["total_energy_std_eV"] == 0.0


def test_fit_requires_precomputed_labels():
    from ase.calculators.calculator import Calculator

    class Expensive(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, *args, **kwargs):
            raise AssertionError("An attached calculator must not be run by fit")

    basis = _density_basis()
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [1.6, 0, 0]])
    atoms.calc = Expensive()
    with pytest.raises(ValueError, match="precomputed"):
        LinearModel(basis).fit([atoms])


def test_density_labels_follow_one_hot_columns_and_force_derivatives():
    basis = Basis(elements=["Cu"], cutoff=3.5, max_rank=1, nmax=2, lmax=0)
    assert len(basis.labels) == 2
    assert tuple(label.as_dict()["radial_indices"] for label in basis.labels) == ((1,), (2,))
    atoms = Atoms("Cu3", positions=[[0, 0, 0], [1.4, 0.1, 0], [0.2, 1.1, 0.3]])
    features = basis.create(atoms)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    order = [2, 0, 1]
    for column in range(2):
        oracle = LinearACEScalarModelBundle(
            settings=basis._descriptor.settings,
            site_basis_config=basis._descriptor.site_basis_config,
            descriptor_specs=basis._descriptor.descriptor_specs,
            weight=np.eye(2)[column], bias=0.0, basis_mode=None,
            fit_method="one_hot_validation",
        )
        reference = atoms.copy()
        reference.calc = LinearACEScalarCalculator(oracle, basis.cutoff, {"Cu": 0})
        np.testing.assert_allclose(reference.get_potential_energy(), features[:, column].sum(), atol=1e-11)
        forces = reference.get_forces()
        displaced = atoms.copy()
        displaced.positions[1, 0] += 1e-6
        displaced.calc = LinearACEScalarCalculator(oracle, basis.cutoff, {"Cu": 0})
        plus = displaced.get_potential_energy()
        displaced.positions[1, 0] -= 2e-6
        minus = displaced.get_potential_energy()
        np.testing.assert_allclose(forces[1, 0], -(plus - minus) / 2e-6, atol=2e-6)
        shifted = atoms.copy()
        shifted.positions += [2.0, -1.0, 0.5]
        shifted.calc = LinearACEScalarCalculator(oracle, basis.cutoff, {"Cu": 0})
        np.testing.assert_allclose(shifted.get_potential_energy(), reference.get_potential_energy(), atol=1e-11)
        permuted = atoms[order]
        permuted.calc = LinearACEScalarCalculator(oracle, basis.cutoff, {"Cu": 0})
        np.testing.assert_allclose(permuted.get_forces(), forces[order], atol=1e-10)
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.T
        rotated.calc = LinearACEScalarCalculator(oracle, basis.cutoff, {"Cu": 0})
        np.testing.assert_allclose(rotated.get_forces(), forces @ rotation.T, atol=1e-10)
