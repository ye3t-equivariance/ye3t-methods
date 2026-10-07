"""Selected Ni refit, saved-coordinate identity, and evaluator safety."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ye3t_methods import LinearModel
from ye3t_methods.portable_archive import portable_feature_design_row


ARCHIVE = (Path(__file__).resolve().parents[1] / "examples" / "publication" /
           "cost_comparison" / "portable_models" / "Ni_ye3t_tagged_60.ye3t")


def _config():
    return {
        "metadata": {"schema": "ye3t_config_v1", "name": "selected_ni_refit_test"},
        "basis": {"from_saved_model": True},
        "representation": {"from_saved_model": True},
        "runtime": {"evaluator": "torch", "neighbors": "ase", "device": "cpu"},
        "model": {
            "kind": "linear", "source_archive_sha256": hashlib.sha256(
                ARCHIVE.read_bytes()).hexdigest(),
            "fit": {
                "solver": "paper_scaled_ridge", "alpha": 1e-8,
                "weights": {"energy": 1.0, "forces": 1.0},
                "group_weights": {"AIMD-NVT": 1.0},
                "tagged_penalty": 1.0,
            },
        },
        "targets": {"energy": "energy", "forces": "forces",
                    "group": "config_type", "stress": None},
        "validation": {"checks": ["force_fd", "round_trip"]},
    }


def _atoms():
    return Atoms("Ni3", positions=((0.0, 0.0, 0.0), (2.4, 0.0, 0.0),
                                   (0.4, 2.3, 0.0)), cell=(10, 10, 10), pbc=False)


def test_selected_refit_matches_real_design_and_validates_saved_artifact(tmp_path):
    original = LinearModel.read(ARCHIVE)
    atoms = _atoms()
    row = portable_feature_design_row(original._fitted, atoms)
    np.testing.assert_allclose(row["site_features"], original.basis.create(atoms),
                               rtol=0, atol=2e-12)
    coefficients = np.concatenate((original._fitted["weights"]["ordinary"],
                                   original._fitted["weights"]["tagged_selected"]))
    evaluated = atoms.copy()
    evaluated.calc = original.ase_calculator(evaluator="torch", neighbors="ase")
    from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
    zbl = atoms.copy()
    zbl.calc = YE3TZBLCalculator.from_model_manifest({
        "reference_potential": original._fitted["sources"]["reference_potential"]})
    np.testing.assert_allclose(
        row["energy"] @ coefficients + len(atoms) *
        original._fitted["weights"]["per_species_E0_Ni"][0],
        evaluated.get_potential_energy() - zbl.get_potential_energy(),
        rtol=0, atol=2e-8)
    np.testing.assert_allclose(row["forces"] @ coefficients,
                               (evaluated.get_forces() - zbl.get_forces()).reshape(-1),
                               rtol=0, atol=2e-8)

    labeled = atoms.copy()
    labeled.info["config_type"] = "AIMD-NVT"
    labeled.calc = SinglePointCalculator(
        labeled, energy=evaluated.get_potential_energy() + 0.01,
        forces=evaluated.get_forces().copy())
    fitted = LinearModel.read(ARCHIVE).fit([labeled], config=_config())
    metadata = fitted._fitted["refit"]["fit_metadata"]
    assert set(metadata["configured_validation"]["results"]) == {"force_fd", "round_trip"}
    assert metadata["training_score"]["energy_rmse_eV_per_atom"] < 0.01
    assert len(metadata["design_rows_sha256"]) == 64
    artifact = fitted.write(tmp_path / "refit.ye3t")
    restored = LinearModel.read(artifact)
    assert restored.basis._resolved["native_plan_status"] == "unavailable_for_refit"
    assert [label.identity for label in restored.labels] == [
        label.identity for label in original.labels]
    selected = restored._fitted["tagged_indices"]
    full = restored._fitted["weights"]["tagged_beta_69"]
    np.testing.assert_allclose(full[list(selected)],
                               restored._fitted["weights"]["tagged_selected"])
    assert np.count_nonzero(np.delete(full, selected)) == 0
    replay = atoms.copy()
    replay.calc = restored.ase_calculator(evaluator="torch")
    assert np.isfinite(replay.get_potential_energy())
    assert np.isfinite(replay.get_forces()).all()
    with pytest.raises(ValueError, match="no native AUTO plan"):
        restored.ase_calculator(evaluator="auto")
    with pytest.raises(ValueError):
        restored.export_lammps(tmp_path / "invalid")

    with zipfile.ZipFile(artifact) as source:
        members = {name: source.read(name) for name in source.namelist()}
    fit = json.loads(members["fit.json"])
    fit["ordinary"][0] += 1.0
    members["fit.json"] = json.dumps(fit).encode("utf-8")
    tampered = tmp_path / "tampered.ye3t"
    with zipfile.ZipFile(tampered, "w") as output:
        for name, body in members.items():
            output.writestr(name, body)
    with pytest.raises(ValueError, match="hash mismatch"):
        LinearModel.read(tampered)


def test_refit_config_rejects_changed_saved_basis():
    model = LinearModel.read(ARCHIVE)
    config = _config()
    config["model"]["source_archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source archive hash"):
        model.fit([_atoms()], config=config)
    config = _config()
    config["model"]["feature_count"] = 127
    with pytest.raises(ValueError, match="feature count"):
        model.fit([_atoms()], config=config)
    config = _config()
    config["validation"]["expected_train_count"] = 263
    with pytest.raises(ValueError, match="training count"):
        model.fit([_atoms()], config=config)


def test_selected_design_requires_full_rank_periodic_cell():
    model = LinearModel.read(ARCHIVE)
    atoms = _atoms()
    atoms.cell = ((10, 0, 0), (0, 10, 0), (0, 0, 0))
    atoms.pbc = (True, True, False)
    with pytest.raises(ValueError, match="full-rank cell"):
        portable_feature_design_row(model._fitted, atoms)


@pytest.mark.parametrize("count", (60, 127, 149))
def test_bundled_selected_ni_tiers_have_frozen_column_counts(count):
    path = ARCHIVE.with_name(f"Ni_ye3t_tagged_{count}.ye3t")
    model = LinearModel.read(path)
    assert len(model.labels) == count
    assert [label.feature_index for label in model.labels] == list(range(count))
