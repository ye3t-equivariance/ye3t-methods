"""Bounded single-file compatibility reader for the promoted Ni composite."""

import hashlib
import json
import os
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest
from ase import Atoms

from ye3t_methods import LinearModel


FOLDER = (Path(__file__).resolve().parents[1] / "examples" / "publication"
          / "cost_comparison" / "lammps" / "Ni" / "models" / "ye3t_tagged_127")


def _parts():
    names = ("model.ye3t.json", "ordinary_backbone.yace",
             "tagged_correction.ye3t.json")
    parts = {"compat/" + name: (FOLDER / name).read_bytes() for name in names}
    composite = json.loads(parts["compat/model.ye3t.json"])
    tagged = json.loads(parts["compat/tagged_correction.ye3t.json"])
    upgrade = {
        "schema": "ye3t_tagged_portfolio_upgrade_v1",
        "composite_self_hash": composite["self_hash"],
        "tagged_self_hash": tagged["self_hash"],
        "portfolio_hash": tagged["tagged_execution_portfolio"]["portfolio_hash"],
    }
    parts["compat/portfolio_upgrade.json"] = (json.dumps(upgrade, sort_keys=True) + "\n").encode()
    legacy_manifest = json.loads((FOLDER / "model_manifest.json").read_bytes())
    legacy_manifest["artifacts"].append({
        "path": "portfolio_upgrade.json",
        "bytes": len(parts["compat/portfolio_upgrade.json"]),
        "sha256": hashlib.sha256(parts["compat/portfolio_upgrade.json"]).hexdigest(),
    })
    parts["compat/model_manifest.json"] = (json.dumps(legacy_manifest, sort_keys=True) + "\n").encode()
    manifest = {
        "schema": "ye3t_legacy_compat_archive_v1",
        "maturity": "compatibility_only", "model": "Ni/ye3t_tagged_127",
        "species_order": ["Ni"], "output_type": "energy_forces_stress",
        "members": {name: {"bytes": len(payload),
                           "sha256": hashlib.sha256(payload).hexdigest()}
                    for name, payload in parts.items()},
    }
    return parts, manifest


def _write_archive(path, parts, manifest, extra=(), manifest_bytes=None):
    if manifest_bytes is None:
        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in (("manifest.json", manifest_bytes), *parts.items(), *extra):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload)
    return path


def test_single_file_compat_reader_pins_bytes_and_preserves_native_zbl(tmp_path):
    parts, manifest = _parts()
    path = _write_archive(tmp_path / "Ni_127_compat.ye3t", parts, manifest)
    assert tuple(tmp_path.iterdir()) == (path,)
    model = LinearModel.read(path)
    assert model.basis.source == "legacy_composite"
    assert model.basis.resolved["feature_count"] == 127
    path.unlink()
    assert tuple(tmp_path.iterdir()) == ()
    with pytest.raises(RuntimeError, match="no serialized compiler label order"):
        _ = model.labels
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library for ASE replay")
    atoms = Atoms("Ni2", positions=((0, 0, 0), (1.4, 0, 0)),
                  cell=(10, 10, 10), pbc=True)
    atoms.calc = model.ase_calculator(evaluator="native_cpu", neighbors="ase",
                                      native_library=library)
    assert atoms.get_potential_energy() == pytest.approx(8.56503716763535, abs=1e-8)
    assert atoms.calc.mixer.calcs[1].results["energy"] == pytest.approx(
        6.29428825069838, abs=1e-10)
    assert np.isfinite(atoms.get_forces()).all()
    assert np.isfinite(atoms.get_stress()).all()
    direct = LinearModel.read(FOLDER / "model.ye3t.json")
    shifted = Atoms("Ni2", positions=((0, 0, 0), (1.35, 0.22, 0.17)),
                    cell=(10, 10, 10), pbc=True)
    reference = shifted.copy()
    shifted.calc = model.ase_calculator(evaluator="native_cpu", neighbors="ase",
                                        native_library=library)
    reference.calc = direct.ase_calculator(evaluator="native_cpu", neighbors="ase",
                                           native_library=library)
    assert shifted.get_potential_energy() == pytest.approx(
        reference.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(shifted.get_forces(), reference.get_forces(),
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(shifted.get_stress(), reference.get_stress(),
                               rtol=0, atol=1e-10)
    assert np.max(np.abs(shifted.get_forces())) > 1e-5
    assert np.max(np.abs(shifted.get_stress()[3:])) > 1e-5


def test_compat_archive_rejects_corrupt_and_outer_repinned_component(tmp_path):
    parts, manifest = _parts()
    changed = bytearray(parts["compat/ordinary_backbone.yace"])
    changed[0] ^= 1
    parts["compat/ordinary_backbone.yace"] = bytes(changed)
    path = _write_archive(tmp_path / "corrupt.ye3t", parts, manifest)
    with pytest.raises(ValueError, match="member SHA-256 mismatch"):
        LinearModel.read(path)
    manifest["members"]["compat/ordinary_backbone.yace"]["sha256"] = hashlib.sha256(
        parts["compat/ordinary_backbone.yace"]).hexdigest()
    path = _write_archive(tmp_path / "repinned.ye3t", parts, manifest)
    with pytest.raises(ValueError, match="Composite artifact SHA-256 mismatch"):
        LinearModel.read(path)


def test_compat_archive_rejects_duplicate_and_unsafe_member_inventory(tmp_path):
    parts, manifest = _parts()
    duplicate_parts = dict(parts)
    duplicate_parts.pop("compat/portfolio_upgrade.json")
    with pytest.warns(UserWarning, match="Duplicate name"):
        path = _write_archive(tmp_path / "duplicate.ye3t", duplicate_parts, manifest,
                              extra=(("compat/model.ye3t.json", parts["compat/model.ye3t.json"]),))
    with pytest.raises(ValueError, match="member inventory"):
        LinearModel.read(path)
    path = _write_archive(tmp_path / "too_many.ye3t", parts, manifest,
                          extra=(("extra.json", b"{}"),))
    with pytest.raises(ValueError, match="central directory limit"):
        LinearModel.read(path)
    parts.pop("compat/portfolio_upgrade.json")
    path = _write_archive(tmp_path / "unexpected.ye3t", parts, manifest,
                          extra=(("../outside.json", b"{}"),))
    with pytest.raises(ValueError, match="member inventory"):
        LinearModel.read(path)


def test_compat_archive_rejects_unsupported_schema_duplicate_json_and_zip_bomb(tmp_path):
    parts, manifest = _parts()
    manifest["schema"] = "ye3t_linear_archive_prototype_v0"
    path = _write_archive(tmp_path / "prototype.ye3t", parts, manifest)
    with pytest.raises(ValueError, match="unsupported manifest schema"):
        LinearModel.read(path)
    manifest["schema"] = "ye3t_legacy_compat_archive_v1"
    manifest_bytes = json.dumps(manifest, sort_keys=True).replace(
        '"schema":', '"schema":"duplicate", "schema":', 1).encode()
    path = _write_archive(tmp_path / "duplicate_json.ye3t", parts, manifest,
                          manifest_bytes=manifest_bytes)
    with pytest.raises(ValueError, match="duplicate key"):
        LinearModel.read(path)
    parts["compat/ordinary_backbone.yace"] = b"0" * (2 * 1024 * 1024)
    path = _write_archive(tmp_path / "compressed_bomb.ye3t", parts, manifest)
    with pytest.raises(ValueError, match="unsafe or oversized member"):
        LinearModel.read(path)


def test_compat_archive_rejects_wrong_identity_and_malformed_manifest(tmp_path):
    parts, manifest = _parts()
    manifest["model"] = "Ni/ye3t_tagged_149"
    path = _write_archive(tmp_path / "wrong_identity.ye3t", parts, manifest)
    with pytest.raises(ValueError, match="model identity differs"):
        LinearModel.read(path)
    path = _write_archive(tmp_path / "malformed.ye3t", parts, manifest,
                          manifest_bytes=b"[]")
    with pytest.raises(ValueError, match="unsupported manifest schema"):
        LinearModel.read(path)
    manifest["model"] = "Ni/ye3t_tagged_127"
    manifest["species_order"] = {"Ni": 1}
    path = _write_archive(tmp_path / "wrong_species_type.ye3t", parts, manifest)
    with pytest.raises(ValueError, match="species differ"):
        LinearModel.read(path)
    manifest["species_order"] = ["Ni"]
    manifest_bytes = json.dumps(manifest, sort_keys=True).replace(
        '"model":', '"unused_number":1e400, "model":', 1).encode()
    path = _write_archive(tmp_path / "nonfinite_number.ye3t", parts, manifest,
                          manifest_bytes=manifest_bytes)
    with pytest.raises(ValueError, match="nonfinite number"):
        LinearModel.read(path)


def test_compat_archive_checks_embedded_json_before_staging(tmp_path):
    parts, manifest = _parts()
    name = "compat/tagged_correction.ye3t.json"
    parts[name] = b'{"a":1,"a":2}'
    manifest["members"][name] = {"bytes": len(parts[name]),
                                 "sha256": hashlib.sha256(parts[name]).hexdigest()}
    path = _write_archive(tmp_path / "embedded_duplicate.ye3t", parts, manifest)
    with pytest.raises(ValueError, match="duplicate key"):
        LinearModel.read(path)
    parts[name] = '{"a":1}'.encode("utf-16")
    manifest["members"][name] = {"bytes": len(parts[name]),
                                 "sha256": hashlib.sha256(parts[name]).hexdigest()}
    path = _write_archive(tmp_path / "embedded_utf16.ye3t", parts, manifest)
    with pytest.raises(ValueError, match="requires UTF-8"):
        LinearModel.read(path)


def test_compat_archive_rejects_zip64_directory_override(tmp_path):
    parts, manifest = _parts()
    path = _write_archive(tmp_path / "zip64_override.ye3t", parts, manifest)
    payload = path.read_bytes()
    end = payload.rfind(b"PK\x05\x06")
    _, _, _, _, _, central_size, central_offset, _ = struct.unpack_from(
        "<4sHHHHIIH", payload, end)
    zip64_end = struct.pack("<4sQHHIIQQQQ", b"PK\x06\x06", 44, 45, 45,
                            0, 0, 6, 6, central_size, central_offset)
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, end, 1)
    legacy_end = bytearray(payload[end:])
    struct.pack_into("<I", legacy_end, 12, central_size + len(zip64_end) + len(locator))
    path.write_bytes(payload[:end] + zip64_end + locator + legacy_end)
    with pytest.raises(ValueError, match="does not accept ZIP64"):
        LinearModel.read(path)
