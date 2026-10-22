"""Execute both paper workflows with fewer frames and the complete 60 columns."""
import os
from pathlib import Path
import runpy

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from ye3t_methods import LinearModel
from ye3t_methods.studies import ni_linear


@pytest.mark.parametrize("family", ["density", "tagged"])
def test_paper_ni60_fit_export_reload(family, monkeypatch, tmp_path):
    monkeypatch.setenv("YE3T_WORKFLOW_ROOT", str(tmp_path))
    original_loader = ni_linear.load_refit_structures

    def short_frames(config):
        training, validation = original_loader(config)
        groups = sorted({frame.info["config_type"] for frame in training})
        selected = [min((frame for frame in training if frame.info["config_type"] == group),
                        key=len) for group in groups]
        config["validation"]["expected_train_count"] = len(selected)
        return selected, sorted(validation, key=len)[:2]

    monkeypatch.setattr(ni_linear, "load_refit_structures", short_frames)
    path = Path(__file__).resolve().parents[1] / "examples/publication/cost_comparison"
    result = runpy.run_path(str(path / f"paper_ni_60_{family}_01_prep_train.py"))
    fitted = result["restored"]
    assert len(fitted.labels) == 60
    assert {label.as_dict()["N"] for label in fitted.labels} == {1, 2, 3, 4, 6, 8}
    assert result["gradient"].shape[0] == 60
    assert result["exported"]["model"].is_file()
    reloaded = LinearModel.read(result["saved"])
    assert [label.identity for label in reloaded.labels] == [label.identity for label in fitted.labels]
    if family == "density":
        assert result["exported"]["manifest"].is_file()
        assert not reloaded.basis._portable["tagged_indices"]
    from ye3t_methods.atomistic.tagged_cauchy_native import _bundled_native_library_path
    if not (os.environ.get("YE3T_TAGGED_C_API_LIBRARY") or _bundled_native_library_path()):
        pytest.skip("fit/export passed; native comparison requires optional C ABI library")
    native = reloaded.ase_calculator(evaluator="native_cpu")
    reference = reloaded.ase_calculator(evaluator="torch")
    periodic = bulk("Ni", "fcc", a=3.52, cubic=True)
    periodic.positions[0] += [0.05, -0.03, 0.02]
    contact = Atoms("Ni2", positions=[[0, 0, 0], [1.4, 0, 0]])
    for geometry in (periodic, contact):
        geometry.calc = reference
        energy, forces = geometry.get_potential_energy(), geometry.get_forces()
        geometry.calc = native
        np.testing.assert_allclose(geometry.get_potential_energy(), energy, rtol=1e-9, atol=2e-7)
        np.testing.assert_allclose(geometry.get_forces(), forces, rtol=1e-9, atol=2e-7)
