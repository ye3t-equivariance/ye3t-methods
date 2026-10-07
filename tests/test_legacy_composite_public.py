"""Public reader regression for the exact promoted legacy Ni composite."""

import json
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.mixing import SumCalculator

from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
from ye3t_methods.atomistic.tagged_cauchy_image import YE3TTaggedCauchyCalculator
from ye3t_methods import LinearModel


FOLDER = (Path(__file__).resolve().parents[1] / "examples" / "publication"
          / "cost_comparison" / "lammps" / "Ni" / "models" / "ye3t_tagged_127")


def test_legacy_composite_reader_verifies_and_reports_missing_labels():
    model = LinearModel.read(FOLDER / "model.ye3t.json")
    assert model.basis.source == "legacy_composite"
    assert model.basis.resolved["feature_count"] == 127
    assert model.basis.elements == ("Ni",)
    assert "features=127" in str(model)
    with pytest.raises(RuntimeError, match="no serialized compiler label order"):
        _ = model.labels
    with pytest.raises(ValueError, match="only a validated native_cpu"):
        model.ase_calculator(evaluator="torch")
    with pytest.raises(ValueError, match="cannot be rewritten"):
        model.write(FOLDER / "unused.ye3t")


@pytest.mark.parametrize("tamper", ("component", "manifest"))
def test_legacy_composite_reader_rejects_tampering(tmp_path, tamper):
    for name in ("model.ye3t.json", "ordinary_backbone.yace",
                 "tagged_correction.ye3t.json", "model_manifest.json"):
        shutil.copyfile(FOLDER / name, tmp_path / name)
    if tamper == "component":
        path = tmp_path / "ordinary_backbone.yace"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        pattern = "SHA-256 mismatch"
    else:
        path = tmp_path / "model_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["model"]["feature_count"] = 126
        path.write_text(json.dumps(manifest), encoding="utf-8")
        pattern = "feature counts differ"
    with pytest.raises(ValueError, match=pattern):
        LinearModel.read(tmp_path / "model.ye3t.json")


def test_legacy_composite_reader_rejects_oversized_manifest_declaration(tmp_path):
    for name in ("model.ye3t.json", "ordinary_backbone.yace",
                 "tagged_correction.ye3t.json", "model_manifest.json"):
        shutil.copyfile(FOLDER / name, tmp_path / name)
    path = tmp_path / "model_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifacts"][1]["bytes"] = 64 * 1024 * 1024 + 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="64 MiB limit"):
        LinearModel.read(tmp_path / "model.ye3t.json")


@pytest.mark.parametrize("alter", (None, "portfolio_hash", "unknown_artifact"))
def test_legacy_composite_reader_verifies_optional_portfolio_sidecar(tmp_path, alter):
    for name in ("model.ye3t.json", "ordinary_backbone.yace",
                 "tagged_correction.ye3t.json", "model_manifest.json"):
        shutil.copyfile(FOLDER / name, tmp_path / name)
    composite = json.loads((tmp_path / "model.ye3t.json").read_text())
    tagged = json.loads((tmp_path / "tagged_correction.ye3t.json").read_text())
    upgrade = {
        "schema": "ye3t_tagged_portfolio_upgrade_v1",
        "composite_self_hash": composite["self_hash"],
        "tagged_self_hash": tagged["self_hash"],
        "portfolio_hash": tagged["tagged_execution_portfolio"]["portfolio_hash"],
    }
    if alter == "portfolio_hash":
        upgrade["portfolio_hash"] = "0" * 64
    name = "unknown.json" if alter == "unknown_artifact" else "portfolio_upgrade.json"
    payload = (json.dumps(upgrade, sort_keys=True) + "\n").encode()
    (tmp_path / name).write_bytes(payload)
    manifest_path = tmp_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].append({
        "path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    })
    manifest_path.write_text(json.dumps(manifest))
    if alter == "portfolio_hash":
        with pytest.raises(ValueError, match="portfolio sidecar differs"):
            LinearModel.read(tmp_path / "model.ye3t.json")
    elif alter == "unknown_artifact":
        with pytest.raises(ValueError, match="components do not match"):
            LinearModel.read(tmp_path / "model.ye3t.json")
    else:
        assert LinearModel.read(tmp_path / "model.ye3t.json").basis.resolved["feature_count"] == 127


def test_legacy_composite_reader_rejects_repinned_tagged_self_hash_mismatch(tmp_path):
    for name in ("model.ye3t.json", "ordinary_backbone.yace",
                 "tagged_correction.ye3t.json", "model_manifest.json"):
        shutil.copyfile(FOLDER / name, tmp_path / name)
    tagged_path = tmp_path / "tagged_correction.ye3t.json"
    tagged = json.loads(tagged_path.read_text())
    tagged["beta"][0] += 1.0
    tagged_path.write_text(json.dumps(tagged))
    tagged_sha = hashlib.sha256(tagged_path.read_bytes()).hexdigest()
    composite_path = tmp_path / "model.ye3t.json"
    composite = json.loads(composite_path.read_text())
    composite["tagged_component"]["sha256"] = tagged_sha
    body = {key: value for key, value in composite.items() if key != "self_hash"}
    composite["self_hash"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    composite_path.write_text(json.dumps(composite))
    manifest_path = tmp_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["artifacts"]:
        path = tmp_path / item["path"]
        item["bytes"] = path.stat().st_size
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tagged component self hash mismatch"):
        LinearModel.read(composite_path)


def test_legacy_composite_reader_recompiles_promoted_portfolio_readout(tmp_path):
    for name in ("model.ye3t.json", "ordinary_backbone.yace",
                 "tagged_correction.ye3t.json", "model_manifest.json"):
        shutil.copyfile(FOLDER / name, tmp_path / name)
    tagged_path = tmp_path / "tagged_correction.ye3t.json"
    tagged = json.loads(tagged_path.read_text())
    tagged["beta"][0] += 1.0
    tagged_body = {key: value for key, value in tagged.items() if key != "self_hash"}
    tagged["self_hash"] = hashlib.sha256(json.dumps(
        tagged_body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    tagged_path.write_text(json.dumps(tagged))
    composite_path = tmp_path / "model.ye3t.json"
    composite = json.loads(composite_path.read_text())
    composite["tagged_component"]["sha256"] = hashlib.sha256(tagged_path.read_bytes()).hexdigest()
    composite_body = {key: value for key, value in composite.items() if key != "self_hash"}
    composite["self_hash"] = hashlib.sha256(json.dumps(
        composite_body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()
    composite_path.write_text(json.dumps(composite))
    upgrade_path = tmp_path / "portfolio_upgrade.json"
    upgrade_path.write_text(json.dumps({
        "schema": "ye3t_tagged_portfolio_upgrade_v1",
        "composite_self_hash": composite["self_hash"],
        "tagged_self_hash": tagged["self_hash"],
        "portfolio_hash": tagged["tagged_execution_portfolio"]["portfolio_hash"],
    }))
    manifest_path = tmp_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].append({"path": "portfolio_upgrade.json"})
    for item in manifest["artifacts"]:
        path = tmp_path / item["path"]
        item["bytes"] = path.stat().st_size
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="compiler-owned program/readout plan"):
        LinearModel.read(composite_path)


def test_legacy_composite_loaded_bytes_survive_source_replacement(tmp_path):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    names = ("model.ye3t.json", "ordinary_backbone.yace",
             "tagged_correction.ye3t.json", "model_manifest.json")
    for name in names:
        shutil.copyfile(FOLDER / name, tmp_path / name)
    model = LinearModel.read(tmp_path / "model.ye3t.json")
    for name in names:
        (tmp_path / name).write_text("replaced after read", encoding="utf-8")
    atoms = Atoms("Ni2", positions=((0, 0, 0), (1.4, 0, 0)),
                  cell=(10, 10, 10), pbc=True)
    atoms.calc = model.ase_calculator(
        evaluator="native_cpu", neighbors="ase", native_library=library,
    )
    assert atoms.get_potential_energy() == pytest.approx(8.56503716763535, abs=1e-8)
    assert atoms.calc.mixer.calcs[1].results["energy"] == pytest.approx(
        6.29428825069838, abs=1e-10,
    )
    assert np.isfinite(atoms.get_forces()).all()
    assert np.isfinite(atoms.get_stress()).all()


@pytest.mark.parametrize("case,lammps_energy_eV", (
    ("16", -92.4218555466858),
    ("128", -739.740818686619),
    ("contact", 8.56503716763535),
))
def test_legacy_composite_public_native_matches_direct_and_zbl(case, lammps_energy_eV):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    if case == "contact":
        atoms = Atoms("Ni2", positions=((0, 0, 0), (1.4, 0, 0)),
                      cell=(10, 10, 10), pbc=True)
    else:
        repeat = (2, 2, 1) if case == "16" else (4, 4, 2)
        atoms = bulk("Ni", "fcc", a=3.508, cubic=True).repeat(repeat)
        atoms.positions[0] += (0.08, -0.05, 0.04)

    direct = atoms.copy()
    direct.calc = SumCalculator((
        YE3TTaggedCauchyCalculator.from_artifact(
            FOLDER / "model.ye3t.json", native_library=library, neighbors="ase"),
        YE3TZBLCalculator.from_model_manifest(FOLDER / "model_manifest.json"),
    ))
    public = atoms.copy()
    model = LinearModel.read(FOLDER / "model.ye3t.json")
    public.calc = model.ase_calculator(
        evaluator="native_cpu", neighbors="ase", native_library=library,
    )
    assert len(public.calc.mixer.calcs) == 2
    expected_energy = direct.get_potential_energy()
    assert expected_energy == pytest.approx(lammps_energy_eV, abs=1e-8)
    assert public.get_potential_energy() == pytest.approx(expected_energy, abs=1e-8)
    np.testing.assert_allclose(public.get_forces(), direct.get_forces(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(public.get_stress(), direct.get_stress(), rtol=0, atol=1e-8)
    if case == "contact":
        assert public.calc.mixer.calcs[1].results["energy"] == pytest.approx(
            6.29428825069838, abs=1e-10,
        )
    if case == "16":
        residual = public.calc.mixer.calcs[0]
        rebuilds = residual.native_runtime.topology_rebuilds
        public.positions[0, 0] += 0.002
        direct.positions[0, 0] += 0.002
        np.testing.assert_allclose(public.get_forces(), direct.get_forces(), rtol=0, atol=1e-8)
        assert residual.native_runtime.topology_rebuilds == rebuilds


@pytest.mark.parametrize("case", ("16", "contact"))
def test_legacy_composite_full_energy_force_and_stress_finite_differences(case):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    if case == "contact":
        atoms = Atoms("Ni2", positions=((0, 0, 0), (1.4, 0, 0)),
                      cell=(10, 10, 10), pbc=True)
    else:
        atoms = bulk("Ni", "fcc", a=3.508, cubic=True).repeat((2, 2, 1))
        atoms.positions[0] += (0.08, -0.05, 0.04)
    model = LinearModel.read(FOLDER / "model.ye3t.json")
    calculator = model.ase_calculator(
        evaluator="native_cpu", neighbors="ase", native_library=library,
    )
    atoms.calc = calculator
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    position_step = 1e-5
    for atom_index, axis in ((0, 0), (0, 1), (1, 2)):
        energies = []
        for direction in (-1, 1):
            shifted = atoms.copy()
            shifted.positions[atom_index, axis] += direction * position_step
            shifted.calc = calculator
            energies.append(shifted.get_potential_energy())
        derivative = -(energies[1] - energies[0]) / (2 * position_step)
        np.testing.assert_allclose(forces[atom_index, axis], derivative, rtol=0, atol=2e-5)
    if case == "contact":
        assert calculator.mixer.calcs[1].results["energy"] > 0
    strain_step = 1e-5
    voigt_axes = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    for index, (first, second) in enumerate(voigt_axes):
        energies = []
        for direction in (-1, 1):
            strain = np.zeros((3, 3))
            strain[first, second] = direction * strain_step
            if first != second:
                strain[second, first] = direction * strain_step
            deformed = atoms.copy()
            deformed.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + strain), scale_atoms=True)
            deformed.calc = calculator
            energies.append(deformed.get_potential_energy())
        multiplicity = 2 if first != second else 1
        derivative = (energies[1] - energies[0]) / (
            2 * strain_step * atoms.get_volume() * multiplicity)
        np.testing.assert_allclose(stress[index], derivative, rtol=0, atol=2e-5)
