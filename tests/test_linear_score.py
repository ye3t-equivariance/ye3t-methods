"""Physical-unit RMSE uses structure and component denominators explicitly."""
from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from ye3t_methods import LinearModel

FIXTURES = Path(__file__).resolve().parents[1]/"examples/quickstart/fixtures"


def labeled_frames():
    model = LinearModel.read(FIXTURES/"ta3_demo.ye3t.json")
    atoms = read(FIXTURES/"ta3_structure.extxyz")
    frames = []
    for repeat, energy_error, force_error in ((1,2,1), (2,4,3)):
        frame = atoms.repeat((repeat,1,1))
        frame.calc = model.ase_calculator(evaluator="reference")
        energy, forces, stress = frame.get_potential_energy(), frame.get_forces(), frame.get_stress()
        frame.calc = SinglePointCalculator(frame, energy=energy+len(frame)*energy_error,
            forces=forces+force_error, stress=stress+2)
        frames.append(frame)
    return model, frames


def test_score_weights_components_and_preserves_references():
    model, frames = labeled_frames()
    original = [frame.calc for frame in frames]
    result = model.score(frames, evaluator="reference", config={"targets": {"stress": "stress"}})
    assert result["structure_count"] == 2
    np.testing.assert_allclose(result["energy_rmse_eV_per_atom"], np.sqrt(10), atol=1e-12)
    np.testing.assert_allclose(result["forces_rmse_eV_per_A"], np.sqrt(19/3), atol=1e-12)
    np.testing.assert_allclose(result["stress_rmse_eV_per_A3"], 2, atol=1e-12)
    assert result["forces_component_count"] == 27
    assert result["stress_component_count"] == 12
    assert all(frame.calc is calc for frame, calc in zip(frames, original))


def test_score_custom_keys_and_energy_only():
    model, frames = labeled_frames()
    for frame in frames:
        frame.info["reference_energy"] = frame.get_potential_energy()
    result = model.score(frames, config={"targets": {"energy": "reference_energy", "forces": None},
                                        "runtime": {"evaluator": "reference"}})
    np.testing.assert_allclose(result["energy_rmse_eV_per_atom"], np.sqrt(10), atol=1e-12)
    assert "forces_rmse_eV_per_A" not in result
    with pytest.raises(ValueError, match="at least one structure"):
        model.score([], evaluator="reference")
    with pytest.raises(ValueError, match="at least one labeled target"):
        model.score(frames, config={"targets": {"energy": None, "forces": None}})
    frames[0].info["reference_energy"] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite energy"):
        model.score(frames, evaluator="reference", config={"targets": {"energy": "reference_energy"}})
