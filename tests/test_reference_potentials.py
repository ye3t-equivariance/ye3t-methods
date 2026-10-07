import os
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from ye3t_methods.atomistic.reference_potentials import (
    YE3TZBLCalculator,
    evaluate_lammps_zbl_reference,
    lammps_zbl_reference_config,
)


def test_lammps_zbl_reference_config_is_canonical_and_multispecies():
    config = lammps_zbl_reference_config(
        {
            "pair_style": "zbl",
            "inner_cutoff_A": 4.0,
            "outer_cutoff_A": 4.8,
            "atomic_numbers": {"O": 8, "Ta": 73},
        },
        ("O", "Ta"),
    )
    assert config["standalone_pair_coeff"] == [
        "1 1 8 8",
        "1 2 8 73",
        "2 2 73 73",
    ]
    assert config["pair_coeff"] == [
        "1 1 zbl 8 8",
        "1 2 zbl 8 73",
        "2 2 zbl 73 73",
    ]
    assert len(config["semantic_sha256"]) == 64
    relocated = lammps_zbl_reference_config(
        {
            "executable": "/another/machine/bin/lmp",
            "pair_style": "zbl",
            "inner_cutoff_A": 4.0,
            "outer_cutoff_A": 4.8,
            "atomic_numbers": {"O": 8, "Ta": 73},
        },
        ("O", "Ta"),
    )
    assert relocated["semantic_sha256"] == config["semantic_sha256"]
    assert relocated["executable"] != config["executable"]
    assert json.loads(json.dumps(config))["semantic_sha256"] == config[
        "semantic_sha256"
    ]


def test_lammps_zbl_reference_energy_force_and_hash_gate():
    executable = os.environ.get("YE3T_TEST_LMP")
    if not executable or not Path(executable).is_file():
        pytest.skip("requires YE3T_TEST_LMP pointing to a LAMMPS executable")
    atoms = Atoms(
        "Ta2",
        positions=((2.0, 2.0, 2.0), (3.5, 2.0, 2.0)),
        cell=(20.0, 20.0, 20.0),
        pbc=False,
    )
    config = lammps_zbl_reference_config(
        {
            "executable": executable,
            "pair_style": "zbl",
            "inner_cutoff_A": 4.0,
            "outer_cutoff_A": 4.8,
            "atomic_numbers": {"Ta": 73},
        },
        ("Ta",),
    )
    result = evaluate_lammps_zbl_reference(
        (atoms,), json.loads(json.dumps(config))
    )
    assert result["reference_energies"].shape == (1,)
    assert result["reference_energies"][0] > 0.0
    force = result["reference_forces"][0]
    assert force.shape == (2, 3)
    assert force[0, 0] < 0.0
    assert np.allclose(force[0], -force[1], rtol=1.0e-12, atol=1.0e-12)
    tampered = dict(config)
    tampered["outer_cutoff_A"] = 4.9
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        evaluate_lammps_zbl_reference((atoms,), tampered)


def test_ase_zbl_from_paper_manifest_matches_lammps_close_contacts():
    executable = os.environ.get("YE3T_TEST_LMP")
    if not executable or not Path(executable).is_file():
        pytest.skip("requires YE3T_TEST_LMP pointing to a LAMMPS executable")
    manifest = (
        Path(__file__).resolve().parents[1] / "examples" / "publication"
        / "cost_comparison" / "lammps" / "Li" / "models"
        / "ye3t_tagged_127" / "model_manifest.json"
    )
    reference = json.loads(manifest.read_text(encoding="utf-8"))["reference_potential"]
    reference["executable"] = executable
    structures = [
        Atoms("Li2", positions=((2, 2, 2), (2 + distance, 2.04, 1.98)),
              cell=(12, 12, 12), pbc=False)
        for distance in (1.4, 1.8, 2.05, 2.2)
    ]
    lammps = evaluate_lammps_zbl_reference(structures, reference)
    for atoms, expected_energy, expected_force in zip(
        structures, lammps["reference_energies"], lammps["reference_forces"], strict=True
    ):
        atoms.calc = YE3TZBLCalculator.from_model_manifest(manifest)
        np.testing.assert_allclose(atoms.get_potential_energy(), expected_energy, atol=2e-14)
        np.testing.assert_allclose(atoms.get_forces(), expected_force, atol=2e-14)
        np.testing.assert_allclose(sum(atoms.get_potential_energies()), expected_energy, atol=2e-14)


def test_ase_zbl_stress_is_energy_strain_derivative():
    manifest = (
        Path(__file__).resolve().parents[1] / "examples" / "publication"
        / "cost_comparison" / "lammps" / "Li" / "models"
        / "ye3t_tagged_127" / "model_manifest.json"
    )
    atoms = Atoms("Li2", positions=((2, 2, 2), (3.5, 2.1, 2)),
                  cell=(12, 12, 12), pbc=True)
    atoms.calc = YE3TZBLCalculator.from_model_manifest(manifest)
    stress = atoms.get_stress()[0]
    energies = []
    for sign in (1, -1):
        strained = atoms.copy()
        cell = atoms.cell.array.copy()
        cell[0] *= 1 + sign * 1e-6
        strained.set_cell(cell, scale_atoms=True)
        strained.calc = YE3TZBLCalculator.from_model_manifest(manifest)
        energies.append(strained.get_potential_energy())
    np.testing.assert_allclose(stress, (energies[0] - energies[1]) / (2e-6 * atoms.get_volume()), atol=2e-10)
