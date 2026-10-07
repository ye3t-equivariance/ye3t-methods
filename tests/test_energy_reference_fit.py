import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms

from ye3t_methods.atomistic.energy_references import fit_element_reference_energies


def _frame(symbols, energy):
    atoms = Atoms(symbols)
    atoms.info["energy"] = float(energy)
    return atoms


def test_element_reference_fit_recovers_exact_composition_energies():
    expected = {"H": -2.0, "O": -5.0, "K": -11.0}
    frames = [
        _frame("H2O", 2 * expected["H"] + expected["O"]),
        _frame("KOH", expected["K"] + expected["O"] + expected["H"]),
        _frame("K2O", 2 * expected["K"] + expected["O"]),
        _frame("H2", 2 * expected["H"]),
    ]

    report = fit_element_reference_energies(
        frames,
        elements=("H", "O", "K"),
        weighting="per_atom",
    )

    assert report["design_rank"] == 3
    assert report["train_residual_rmse_eV_per_atom"] < 1.0e-12
    for symbol, value in expected.items():
        assert np.isclose(report["reference_energies"][symbol], value)


def test_element_reference_fit_reports_rank_deficiency_and_initial_corrections():
    frames = [
        _frame("H2O", -9.0),
        _frame("H4O2", -18.0),
    ]
    report = fit_element_reference_energies(
        frames,
        elements=("H", "O"),
        initial_reference_energies={"H": -1.0, "O": -4.0},
        ridge=1.0e-8,
    )

    assert report["design_rank"] == 1
    assert np.isinf(report["condition_number"])
    assert set(report["corrections"]) == {"H", "O"}


def test_element_reference_fit_rejects_missing_species():
    with pytest.raises(ValueError, match="omits observed species"):
        fit_element_reference_energies(
            [_frame("OH", -7.0)],
            elements=("H",),
        )
