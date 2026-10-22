"""Saved Ni-60 example reproduces independent LAMMPS energies and ASE forces."""
import os
from pathlib import Path
import runpy

import numpy as np
import pytest


def test_paper_ni_60_saved_models():
    from ye3t_methods.atomistic.tagged_cauchy_native import _bundled_native_library_path
    if not (os.environ.get("YE3T_TAGGED_C_API_LIBRARY") or _bundled_native_library_path()):
        pytest.skip("requires optional native YACE/tagged C ABI library")
    path = Path(__file__).resolve().parents[1] / "examples/quickstart/paper_ni_60.py"
    result = runpy.run_path(str(path))
    config = result["config"]
    assert result["descriptors"].shape == (32, 60)
    assert result["larger_descriptors"].shape == (64, 60)
    for source in ("density", "tagged"):
        assert result[source+"_energy"] == pytest.approx(
            config["validation"][source+"_energy_eV"], abs=config["validation"]["energy_tolerance_eV"])
    atoms = result["atoms"].copy()
    atoms.calc = result["tagged"].ase_calculator(evaluator="torch", neighbors="ase")
    assert atoms.get_potential_energy() == pytest.approx(result["tagged_energy"], abs=1e-8)
    np.testing.assert_allclose(atoms.get_forces(), result["tagged_forces"], rtol=2e-8, atol=2e-8)
    for calculator in (result["atoms"].calc, result["density"]):
        # This geometry is outside the reference core. The density check tests
        # the residual's own energy/force agreement, independently of that fact.
        probe = atoms.copy()
        probe.calc = calculator
        expected = probe.get_forces()[0, 0]
        step = 1e-5
        probe.positions[0, 0] += step
        plus = probe.get_potential_energy()
        probe.positions[0, 0] -= 2 * step
        minus = probe.get_potential_energy()
        assert -(plus-minus)/(2*step) == pytest.approx(expected, rel=2e-5, abs=2e-5)
