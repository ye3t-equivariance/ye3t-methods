"""Public portable Ni scalar archive reader and independent native replay."""

import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from ye3t_methods import LinearModel


WORKSPACE = Path(__file__).resolve().parents[2]
ARCHIVE = (WORKSPACE / "ye3t-workflows" / "MLIP" /
           "standardization_p2_20261005" /
           "Ni_ye3t_tagged_127_portable_candidate.ye3t")
PROMOTED = (WORKSPACE / "ye3t-ace" / "examples" / "publication" /
            "cost_comparison" / "Ni" / "models" / "ye3t_tagged_127" /
            "model.ye3t.json")


def _candidate():
    if not ARCHIVE.is_file():
        pytest.skip("requires the local Ni portable study archive")
    return ARCHIVE


def _candidate_for(name):
    path = ARCHIVE.with_name("Ni_" + name + "_portable_candidate.ye3t")
    if not path.is_file():
        pytest.skip("requires local Ni portable study archive: " + name)
    return path


def _atoms(case):
    if case == "contact":
        return Atoms("Ni2", positions=((0, 0, 0), (1.4, 0, 0)),
                     cell=(10, 10, 10), pbc=True)
    atoms = bulk("Ni", "fcc", a=3.508, cubic=True).repeat((2, 2, 1))
    atoms.positions[0] += (0.08, -0.05, 0.04)
    return atoms


def _repack(path, changed):
    with zipfile.ZipFile(_candidate()) as source:
        members = {name: source.read(name) for name in source.namelist()}
    members.update(changed)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_portable_reader_exposes_ordered_rows_and_roundtrips_bytes(tmp_path, monkeypatch):
    import ye3t.couplings
    import ye3t_methods.atomistic.tagged_cauchy_fit
    from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator

    path = _candidate()
    with zipfile.ZipFile(path) as archive:
        assert not any(name.startswith("compat/") for name in archive.namelist())
    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs:
                      pytest.fail("portable read compiled new coupling coefficients"))
        patch.setattr(ye3t_methods.atomistic.tagged_cauchy_fit, "compile_coupling",
                      lambda *args, **kwargs:
                      pytest.fail("portable read compiled a tagged coupler"))
        model = LinearModel.read(path)
    assert model.basis.source == "portable_linear"
    assert len(model.labels) == 127
    assert [item.feature_index for item in model.labels] == list(range(127))
    assert all("N=" in str(item) for item in model.labels)
    assert model.labels[0].as_dict()["branch"] == "ordinary"
    assert model.labels[-1].as_dict()["branch"] == "tagged"
    atoms = _atoms("16")
    rows = model.basis.create(atoms)
    assert rows.shape == (len(atoms), 127)
    assert rows.dtype == np.float64
    assert np.isfinite(rows).all()
    weights = model._fitted["weights"]
    coefficient = np.concatenate((weights["ordinary"], weights["tagged_selected"]))
    zbl_atoms = atoms.copy()
    zbl_atoms.calc = YE3TZBLCalculator.from_model_manifest(
        {"reference_potential": model._fitted["sources"]["reference_potential"]})
    expected = (float(rows.sum(axis=0) @ coefficient) +
                float(weights["per_species_E0_Ni"][0]) * len(atoms) +
                zbl_atoms.get_potential_energy())
    evaluated = atoms.copy()
    evaluated.calc = model.ase_calculator(evaluator="torch", neighbors="ase")
    assert evaluated.get_potential_energy() == pytest.approx(expected, abs=2e-9)
    assert np.isfinite(evaluated.get_forces()).all()
    assert np.isfinite(evaluated.get_stress()).all()
    saved = model.write(tmp_path / "copy")
    assert saved.read_bytes() == path.read_bytes()
    assert len(LinearModel.read(saved).labels) == 127
    with pytest.raises(ValueError, match="Torch evaluator"):
        model.ase_calculator(evaluator="native_cpu")
    with pytest.raises(ValueError, match="ARD posterior"):
        model.predict_uncertainty(atoms)


def test_portable_repeated_geometry_evaluation_does_not_compile(monkeypatch):
    import ye3t.couplings
    import ye3t_methods.atomistic.tagged_cauchy_fit

    model = LinearModel.read(_candidate())
    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs:
                      pytest.fail("repeated portable evaluation compiled a coupler"))
        patch.setattr(ye3t_methods.atomistic.tagged_cauchy_fit, "compile_coupling",
                      lambda *args, **kwargs:
                      pytest.fail("repeated portable evaluation compiled a tagged coupler"))
        calculator = model.ase_calculator(evaluator="torch", neighbors="ase")
        atoms = _atoms("16")
        atoms.calc = calculator
        for offset in (0.0, 0.025, -0.015):
            atoms.positions[0, 0] += offset
            assert np.isfinite(atoms.get_potential_energy())
            assert np.isfinite(atoms.get_forces()).all()
            assert np.isfinite(atoms.get_stress()).all()


def test_portable_replay_without_global_cache_invalidates_ase_state(tmp_path, monkeypatch):
    import ye3t.couplings
    import ye3t_methods.atomistic.tagged_cauchy_fit

    cache_dir = tmp_path / "absent_global_cache"
    monkeypatch.setenv("YE3T_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs:
                      pytest.fail("portable replay compiled a core coupler"))
        patch.setattr(ye3t_methods.atomistic.tagged_cauchy_fit, "compile_coupling",
                      lambda *args, **kwargs:
                      pytest.fail("portable replay compiled a tagged coupler"))
        model = LinearModel.read(_candidate())
        atoms = _atoms("16")
        atoms.calc = model.ase_calculator(evaluator="torch", neighbors="ase")

        def check_against_fresh_calculator():
            expected = atoms.copy()
            expected.calc = model.ase_calculator(evaluator="torch", neighbors="ase")
            assert atoms.get_potential_energy() == pytest.approx(
                expected.get_potential_energy(), abs=2e-9)
            np.testing.assert_allclose(atoms.get_forces(), expected.get_forces(),
                                       rtol=0, atol=2e-8)
            np.testing.assert_allclose(atoms.get_stress(), expected.get_stress(),
                                       rtol=0, atol=2e-8)

        check_against_fresh_calculator()
        atoms.positions[0] += (0.06, -0.03, 0.02)
        check_against_fresh_calculator()
        cell = np.asarray(atoms.cell).copy()
        cell[0, 1] += 0.12
        atoms.set_cell(cell, scale_atoms=True)
        check_against_fresh_calculator()
        atoms.pbc = (True, True, False)
        check_against_fresh_calculator()
        atoms.numbers[0] = 29
        with pytest.raises(KeyError, match="Cu"):
            atoms.get_potential_energy()
        atoms.numbers[0] = 28
        check_against_fresh_calculator()
    assert not cache_dir.exists()


def test_portable_scalar_rotation_inversion_and_atom_relabeling():
    model = LinearModel.read(_candidate())
    atoms = _atoms("16")
    rows = model.basis.create(atoms)
    atoms.calc = model.ase_calculator(evaluator="torch", neighbors="ase")
    energy = atoms.get_potential_energy()
    force = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    axis = np.asarray((1.0, 2.0, -0.5))
    axis /= np.linalg.norm(axis)
    cross = np.asarray(((0.0, -axis[2], axis[1]),
                        (axis[2], 0.0, -axis[0]),
                        (-axis[1], axis[0], 0.0)))
    angle = 0.37
    rotation = (np.eye(3) + np.sin(angle) * cross +
                (1 - np.cos(angle)) * cross @ cross)
    for transform in (rotation, -np.eye(3)):
        moved = atoms.copy()
        moved.positions = moved.positions @ transform.T
        moved.set_cell(np.asarray(atoms.cell) @ transform.T, scale_atoms=False)
        moved.calc = model.ase_calculator(evaluator="torch", neighbors="ase")
        np.testing.assert_allclose(model.basis.create(moved), rows,
                                   rtol=1e-12, atol=5e-9)
        assert moved.get_potential_energy() == pytest.approx(energy, abs=5e-9)
        np.testing.assert_allclose(moved.get_forces(), force @ transform.T,
                                   rtol=0, atol=5e-8)
        np.testing.assert_allclose(moved.get_stress(voigt=False),
                                   transform @ stress @ transform.T,
                                   rtol=0, atol=5e-8)
    order = np.arange(len(atoms))[::-1]
    relabeled = atoms[order]
    relabeled.calc = model.ase_calculator(evaluator="torch", neighbors="ase")
    np.testing.assert_allclose(model.basis.create(relabeled), rows[order],
                               rtol=1e-12, atol=5e-9)
    assert relabeled.get_potential_energy() == pytest.approx(energy, abs=5e-9)
    np.testing.assert_allclose(relabeled.get_forces(), force[order], rtol=0,
                               atol=5e-8)
    np.testing.assert_allclose(relabeled.get_stress(voigt=False), stress,
                               rtol=0, atol=5e-8)


@pytest.mark.parametrize("case", ("16", "contact"))
def test_portable_total_force_and_six_stress_components_match_energy_slope(case):
    model = LinearModel.read(_candidate())
    calculator = model.ase_calculator(evaluator="torch", neighbors="ase")
    atoms = _atoms(case)
    atoms.calc = calculator
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    position_step = 1e-5
    force_coordinates = ((0, 0), (0, 1), (1, 2)) if case == "16" else (
        (0, 0), (1, 0), (1, 2))
    for atom_index, axis in force_coordinates:
        energies = []
        for sign in (-1, 1):
            moved = atoms.copy()
            moved.positions[atom_index, axis] += sign * position_step
            moved.calc = calculator
            energies.append(moved.get_potential_energy())
        difference = -(energies[1] - energies[0]) / (2 * position_step)
        assert difference == pytest.approx(forces[atom_index, axis], abs=2e-5)
    cell0 = np.asarray(atoms.cell.array)
    volume = float(atoms.get_volume())
    strain_step = 1e-5
    voigt = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    for index, (left, right) in enumerate(voigt):
        energies = []
        for sign in (-1, 1):
            strain = np.eye(3)
            increment = sign * strain_step if left == right else sign * strain_step / 2
            strain[left, right] += increment
            if left != right:
                strain[right, left] += increment
            moved = atoms.copy()
            moved.set_cell(cell0 @ strain.T, scale_atoms=True)
            moved.calc = calculator
            energies.append(moved.get_potential_energy())
        difference = (energies[1] - energies[0]) / (2 * strain_step * volume)
        assert difference == pytest.approx(stress[index], abs=2e-5)


@pytest.mark.parametrize("case", ("16", "contact"))
def test_portable_torch_matches_pinned_native_energy_force_stress(case):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not Path(library).is_file():
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    portable = LinearModel.read(_candidate())
    direct = LinearModel.read(PROMOTED)
    atoms = _atoms(case)
    replay = atoms.copy()
    reference = atoms.copy()
    replay.calc = portable.ase_calculator(evaluator="torch", neighbors="ase")
    reference.calc = direct.ase_calculator(
        evaluator="native_cpu", neighbors="ase", native_library=library)
    assert replay.get_potential_energy() == pytest.approx(
        reference.get_potential_energy(), abs=2e-8)
    np.testing.assert_allclose(replay.get_forces(), reference.get_forces(),
                               rtol=0, atol=2e-8)
    np.testing.assert_allclose(replay.get_stress(), reference.get_stress(),
                               rtol=0, atol=2e-8)


@pytest.mark.parametrize("name,width", (
    ("ye3t_tagged_60", 60), ("ye3t_tagged_127", 127),
    ("ye3t_tagged_149", 149), ("ye3t_augmented_196", 196),
))
def test_all_portable_ni_widths_match_pinned_native(name, width):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not Path(library).is_file():
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    portable = LinearModel.read(_candidate_for(name))
    assert len(portable.labels) == width
    direct = LinearModel.read(PROMOTED.parent.parent / name / "model.ye3t.json")
    atoms = _atoms("16")
    assert portable.basis.create(atoms).shape == (len(atoms), width)
    replay = atoms.copy()
    reference = atoms.copy()
    replay.calc = portable.ase_calculator(evaluator="torch", neighbors="ase")
    reference.calc = direct.ase_calculator(
        evaluator="native_cpu", neighbors="ase", native_library=library)
    assert replay.get_potential_energy() == pytest.approx(
        reference.get_potential_energy(), abs=2e-8)
    np.testing.assert_allclose(replay.get_forces(), reference.get_forces(),
                               rtol=0, atol=2e-8)
    np.testing.assert_allclose(replay.get_stress(), reference.get_stress(),
                               rtol=0, atol=2e-8)


def test_portable_reader_rejects_corrupt_and_rehashed_wrong_label(tmp_path):
    path = _candidate()
    with zipfile.ZipFile(path) as archive:
        member = bytearray(archive.read("weights.npz"))
    member[-1] ^= 1
    bad = _repack(tmp_path / "bad.ye3t", {"weights.npz": bytes(member)})
    with pytest.raises(ValueError, match="trusted SHA-256 mismatch"):
        LinearModel.read(bad)
    with zipfile.ZipFile(path) as archive:
        labels = json.loads(archive.read("labels.json"))
        manifest = json.loads(archive.read("manifest.json"))
    labels["ordered"][0]["feature_id"] = "wrong feature"
    payload = json.dumps(labels, sort_keys=True).encode()
    manifest["members"]["labels.json"] = {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    repinned = _repack(tmp_path / "repinned.ye3t", {
        "labels.json": payload,
        "manifest.json": json.dumps(manifest, sort_keys=True).encode(),
    })
    with pytest.raises(ValueError, match="trusted SHA-256 mismatch"):
        LinearModel.read(repinned)


def test_portable_reader_rejects_rehashed_wrong_exact_image(tmp_path):
    with zipfile.ZipFile(_candidate()) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        image_name = next(name for name in archive.namelist()
                          if name.startswith("source/tagged_images/"))
        image = json.loads(archive.read(image_name))
    row = image["image_from_raw"][0]
    changed = next(item for item in row if item["binary64"] != [0.0, 0.0])
    changed["binary64"][0] += 0.125
    image["record_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in image.items() if key != "record_hash"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = json.dumps(image, sort_keys=True).encode()
    manifest["members"][image_name] = {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    bad = _repack(tmp_path / "wrong_image.ye3t", {
        image_name: payload,
        "manifest.json": json.dumps(manifest, sort_keys=True).encode(),
    })
    with pytest.raises(ValueError, match="trusted SHA-256 mismatch"):
        LinearModel.read(bad)
