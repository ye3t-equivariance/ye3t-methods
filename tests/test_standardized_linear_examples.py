"""Run the public fits without reducing their radial, angular or rank settings."""
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]/"examples"/"quickstart"


@pytest.mark.parametrize("name", ["density_fit.py", "tagged_fit.py", "combined_density_tagged_fit.py",
                                  "shell_density_fit.py", "sklearn_fit.py",
                                  pytest.param("../long/tagged_fit_large.py", marks=pytest.mark.long_example)])
def test_public_fit_save_reload_and_position_derivative(name, tmp_path, monkeypatch):
    if name == "sklearn_fit.py":
        pytest.importorskip("sklearn", reason="requires optional scikit-learn fitting extra")
    if name in {"tagged_fit.py", "combined_density_tagged_fit.py", "shell_density_fit.py", "../long/tagged_fit_large.py"}:
        from ye3t.runtime.execution_plan import native_execution_plan_capabilities
        if not native_execution_plan_capabilities()["cpu"]:
            pytest.skip("requires native ye3t polynomial extension")
    monkeypatch.chdir(tmp_path)
    if name == "../long/tagged_fit_large.py":
        from ye3t_methods import LinearModel
        original_calculator = LinearModel.ase_calculator
        def preserve_before_validation(model, *args, **kwargs):
            retained = tmp_path/'fit_before_validation.ye3t.json'
            if not retained.exists():
                model.write(retained)
            return original_calculator(model, *args, **kwargs)
        monkeypatch.setattr(LinearModel, 'ase_calculator', preserve_before_validation)
    result = runpy.run_path(str(ROOT/name))
    basis, atoms = result["basis"], result["atoms"]
    assert len(basis.labels) > 15
    if name == "../long/tagged_fit_large.py":
        assert len(basis.labels) == 691
    assert len(result["structures"]) == 12
    assert np.isfinite(atoms.get_potential_energy())
    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3)
    assert np.isfinite(forces).all()
    if name == "sklearn_fit.py":
        reference = result["structures"][0].get_potential_energy()
        for fitted in (result["lasso"], result["ard"], result["restored"]):
            probe = result["structures"][0].copy()
            probe.calc = fitted.ase_calculator(evaluator="torch", neighbors="ase")
            assert abs(probe.get_potential_energy()-reference) / len(probe) < .1
            assert np.linalg.norm(probe.get_forces()) > 1e-3
        uncertainty = result["uncertainty"]
        assert np.isfinite(uncertainty["total_energy_std_eV"])
        assert uncertainty["total_energy_std_eV"] > 0
    original = atoms.positions.copy()
    step = 2e-5
    atoms.positions[0, 1] += step
    plus = atoms.get_potential_energy()
    atoms.positions[0, 1] -= 2*step
    minus = atoms.get_potential_energy()
    atoms.positions = original
    np.testing.assert_allclose(-(plus-minus)/(2*step), forces[0, 1], rtol=2e-4, atol=2e-5)
    if name != "sklearn_fit.py":
        atoms.calc = result["model"].ase_calculator(
            evaluator=result["config"]["runtime"]["evaluator"], neighbors="ase")
        np.testing.assert_allclose(atoms.get_forces(), forces, rtol=1e-10, atol=1e-10)
        assert Path(result["artifact"]).is_file()


@pytest.mark.parametrize("name", ["chemical_encoding.py", "chemical_channel_selection.py"])
def test_public_chemical_example_symmetry_and_selection(name):
    result = runpy.run_path(str(ROOT/name))
    basis, atoms, expected = result["basis"], result["atoms"], result["descriptors"]
    assert {row.as_dict()["N"] for row in basis.labels} == {1,2,3,4}
    assert expected.shape[1] > 15
    rotated = atoms.copy()
    rotated.rotate(37, "z", rotate_cell=True)
    np.testing.assert_allclose(basis.create(rotated), expected, rtol=1e-9, atol=1e-9)
    order = [2,0,3,1]
    np.testing.assert_allclose(basis.create(atoms[order]), expected[order], rtol=1e-9, atol=1e-9)
    if name == "chemical_channel_selection.py":
        moved = atoms.copy()
        moved.positions[[1,3]] += [.2,-.1,.3]
        np.testing.assert_allclose(basis.create(moved)[[0,2]], expected[[0,2]], rtol=1e-9, atol=1e-9)
