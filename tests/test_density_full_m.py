"""Ordinary-density full-M compiler and geometry regressions."""

import hashlib
import json
import struct
import numpy as np
import pytest
import torch
from ase import Atoms

from ye3t import YE3TRepresentation
from ye3t.core.rotation import wigner_D_numeric
from ye3t.core.tesseral import (complex_multiplet_to_real_tesseral,
                               real_tesseral_to_complex_multiplet)
from ye3t_methods import Basis, LinearModel


def _density_basis(rank, L, nmax=1, lmax=None, partition=None,
                   species=("Ni",), chemical=None, parity=None):
    parity = parity or ("odd" if L % 2 else "even")
    lmax = max(1, L) if lmax is None else lmax
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [rank],
        "parent": {"young_lambda": "(N)", "L": L, "parity": parity},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {rank: nmax}, "l_max_per_rank": {rank: lmax}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    basis = {
        "single_factors": {
            "species": list(species),
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 1.4,
                       "cutoff_width_A": .01, "lambda": .79},
            "chemical": chemical or {"kind": "explicit"}},
        "tensor_product": {"kind": "density"},
        "catalogue": {"ranks": [rank], "nmax_per_rank": {rank: nmax},
                      "lmax_per_rank": {rank: lmax},
                      "source_block_partitions_by_rank": {
                          rank: [partition if partition is not None else [rank]]}},
    }
    runtime = {"evaluator": "torch", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    return Basis.from_config(basis, representation=representation, runtime=runtime)


@pytest.mark.parametrize("rank,L", [(1, 1), (2, 2), (1, 3)])
def test_density_full_m_rotation_inversion_and_permutation(rank, L):
    basis = _density_basis(rank, L)
    atoms = Atoms("Ni3", positions=[[0, 0, 0], [.8, .2, .1], [.1, .9, .3]])
    values = basis.create(atoms)
    assert values.shape == (3, len(basis.labels), 2 * L + 1)
    assert len(basis.labels) == basis.catalogue.counts()["exact_total_per_center"]
    assert all(label.as_dict()["M_values"] == tuple(range(-L, L + 1))
               for label in basis.labels)
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    turned = atoms.copy()
    turned.positions = atoms.positions @ rotation.T
    turned_values = basis.create(turned)
    original_complex = real_tesseral_to_complex_multiplet(
        torch.as_tensor(values), L).numpy()
    turned_complex = real_tesseral_to_complex_multiplet(
        torch.as_tensor(turned_values), L).numpy()
    np.testing.assert_allclose(turned_complex,
                               original_complex @ wigner_D_numeric(L, rotation).T,
                               atol=2e-10, rtol=2e-10)
    inverted = atoms.copy()
    inverted.positions = -atoms.positions
    np.testing.assert_allclose(basis.create(inverted), (-1) ** L * values,
                               atol=2e-10, rtol=2e-10)
    order = [2, 0, 1]
    np.testing.assert_allclose(basis.create(atoms[order]), values[order],
                               atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("rank,L", [(1, 1), (2, 2), (1, 3)])
def test_density_full_m_periodic_crossing_uses_cartesian_image(rank, L):
    basis = _density_basis(rank, L)
    wrapped = Atoms("Ni2", positions=[[.1, .1, .1], [3.9, .1, .1]],
                    cell=[4, 4, 4], pbc=True)
    direct = Atoms("Ni2", positions=[[.1, .1, .1], [-.1, .1, .1]])
    np.testing.assert_allclose(basis.create(wrapped), basis.create(direct),
                               atol=2e-10, rtol=2e-10)


def test_density_full_m_declared_parity_can_differ_from_L_parity(tmp_path):
    basis = _density_basis(2, 1, nmax=2, lmax=1, partition=[1, 1],
                           parity="even")
    atoms = Atoms("Ni3", positions=[[0, 0, 0], [.8, .2, .1], [.1, .9, .3]])
    values = basis.create(atoms)
    assert values.shape == (3, len(basis.labels), 3)
    assert len(basis.labels) == basis.catalogue.counts()["exact_total_per_center"]
    assert np.linalg.norm(values) > 1e-10
    inverted = atoms.copy()
    inverted.positions = -atoms.positions
    np.testing.assert_allclose(basis.create(inverted), values,
                               atol=2e-10, rtol=2e-10)
    angle = .41
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    turned = atoms.copy()
    turned.positions = atoms.positions @ rotation.T
    original_complex = real_tesseral_to_complex_multiplet(
        torch.as_tensor(values), 1).numpy()
    turned_complex = real_tesseral_to_complex_multiplet(
        torch.as_tensor(basis.create(turned)), 1).numpy()
    np.testing.assert_allclose(turned_complex,
                               original_complex @ wigner_D_numeric(1, rotation).T,
                               atol=2e-10, rtol=2e-10)
    atoms.new_array("target", values[:, 0, :])
    source = basis._construction
    config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "even_vector",
                     "status": "experimental"},
        "representation": source["representation"],
        "basis": source["basis"], "runtime": source["runtime"],
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "target", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit([atoms], config=config)
    saved = model.write(tmp_path / "even_vector.ye3t.json")
    payload = json.loads(saved.read_text())
    assert payload["validation"]["declared_parity"] == "compiler_selected"
    restored = LinearModel.read(saved)
    np.testing.assert_allclose(restored.predict(atoms)["mean_real_tesseral"],
                               model.predict(atoms)["mean_real_tesseral"],
                               atol=2e-10, rtol=2e-10)


def test_density_l3_rank_one_magnetic_phase_matches_scipy_harmonics():
    special = pytest.importorskip("scipy.special")
    basis = _density_basis(1, 3, nmax=1, lmax=3)
    displacement = np.array([.71, .24, .39])
    atoms = Atoms("Ni2", positions=[[0, 0, 0], displacement])
    values = basis.create(atoms)
    assert values.shape == (2, 1, 7)
    complex_values = real_tesseral_to_complex_multiplet(
        torch.as_tensor(values[0, 0]), 3).numpy()
    radius = np.linalg.norm(displacement)
    theta = np.arccos(displacement[2] / radius)
    phi = np.mod(np.arctan2(displacement[1], displacement[0]), 2 * np.pi)
    harmonic = np.array([special.sph_harm_y(3, m, theta, phi)
                         for m in range(-3, 4)])
    scale = np.vdot(harmonic, complex_values) / np.vdot(harmonic, harmonic)
    assert abs(scale.imag) < 1e-11 and abs(scale.real) > 1e-10
    np.testing.assert_allclose(complex_values, scale.real * harmonic,
                               atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("rank,L", [(1, 1), (2, 2), (1, 3)])
def test_density_full_m_shared_coefficient_fit(rank, L, tmp_path, monkeypatch):
    basis = _density_basis(rank, L)
    frames = []
    beta = np.linspace(.2, .7, len(basis.labels))
    for shift in (.0, .08, -.06):
        atoms = Atoms("Ni3", positions=[[0, 0, 0], [.8 + shift, .2, .1],
                                        [.1, .9 - shift, .3]])
        atoms.new_array("target", np.einsum("nfm,f->nm", basis.create(atoms), beta))
        frames.append(atoms)
    construction = basis._construction
    fit_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "density_full_m_fit",
                     "status": "experimental"},
        "representation": construction["representation"],
        "basis": construction["basis"], "runtime": construction["runtime"],
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "target", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit(frames, config=fit_config)
    saved = model.write(tmp_path / "density.ye3t.json")
    payload = json.loads(saved.read_text())
    plan = payload["native_property_plan"]
    assert payload["schema"] == "ye3t_methods_density_full_m_per_atom_v2"
    assert plan["compiler_hash"] == payload["compiler_hash"]
    assert plan["coefficient_sha256"] == payload["compiled_density"]["coefficient_sha256"]
    assert plan["selected_coordinate_ids"] == payload["selected_coordinate_ids"]
    assert plan["readout"]["coefficients"] == payload["fit"]["beta"]
    assert len(plan["target"]["real_to_complex_matrix"]) == 2 * L + 1
    monkeypatch.setattr("ye3t.couplings.compile", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("reader recompiled density")))
    restored = LinearModel.read(saved)
    legacy = json.loads(saved.read_text())
    legacy["schema"] = "ye3t_methods_density_full_m_per_atom_v1"
    legacy.pop("native_property_plan")
    legacy["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in legacy.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    legacy_path = tmp_path / "density_v1.ye3t.json"
    legacy_path.write_text(json.dumps(legacy))
    legacy_model = LinearModel.read(legacy_path)
    np.testing.assert_allclose(legacy_model.predict(frames[0])["mean_real_tesseral"],
                               restored.predict(frames[0])["mean_real_tesseral"],
                               atol=1e-12, rtol=1e-12)
    changed = json.loads(saved.read_text())
    changed_plan = changed["native_property_plan"]
    changed_plan["source"]["pair_cutoffs_A"]["Ni-Ni"] *= .9
    changed_plan["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in changed_plan.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    changed["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in changed.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    changed_path = tmp_path / "density_native_plan_changed.ye3t.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="native property plan"):
        LinearModel.read(changed_path)
    assert model._fitted["beta"].shape == (len(basis.labels),)
    assert model._fitted["fit_metadata"]["n_cols"] == len(basis.labels)
    for atoms in frames:
        np.testing.assert_allclose(model.predict(atoms)["mean_real_tesseral"],
                                   atoms.arrays["target"], atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(restored.predict(atoms)["mean_real_tesseral"],
                                   atoms.arrays["target"], atol=2e-10, rtol=2e-10)
    calc_atoms = frames[0].copy()
    calc_atoms.calc = model.ase_calculator()
    np.testing.assert_allclose(calc_atoms.calc.get_property(
        "per_atom_real_tesseral_mean", calc_atoms), frames[0].arrays["target"],
        atol=2e-10, rtol=2e-10)


def test_density_full_m_rank_three_repeated_content_multipath():
    basis = _density_basis(3, 2, nmax=2, lmax=2, partition=[2, 1])
    atoms = Atoms("Ni4", positions=[[0, 0, 0], [.6, .2, .1],
                                    [.1, .7, .2], [.2, .1, .8]])
    rows = basis.create(atoms)
    assert rows.shape == (4, len(basis.labels), 5)
    assert len(basis.labels) == basis.catalogue.counts()["exact_total_per_center"]
    angle = .27
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    turned = atoms.copy()
    turned.positions = atoms.positions @ rotation.T
    original = real_tesseral_to_complex_multiplet(torch.as_tensor(rows), 2).numpy()
    transformed = real_tesseral_to_complex_multiplet(
        torch.as_tensor(basis.create(turned)), 2).numpy()
    np.testing.assert_allclose(transformed, original @ wigner_D_numeric(2, rotation).T,
                               atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("rank,partition", [(3, [3]), (4, [3, 1])])
def test_density_full_m_collapsed_blocks_rotate_invert_and_relabel(rank, partition):
    basis = _density_basis(rank, 1, nmax=2, lmax=1, partition=partition)
    atoms = Atoms("Ni4", positions=[[0, 0, 0], [.6, .2, .1],
                                    [.1, .7, .2], [.2, .1, .8]])
    values = basis.create(atoms)
    assert values.shape == (4, len(basis.labels), 3)
    assert len(basis.labels) == basis.catalogue.counts()["exact_total_per_center"]
    assert np.linalg.norm(values) > 1e-10
    angle = .39
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    turned = atoms.copy()
    turned.positions = atoms.positions @ rotation.T
    original = real_tesseral_to_complex_multiplet(torch.as_tensor(values), 1).numpy()
    transformed = real_tesseral_to_complex_multiplet(
        torch.as_tensor(basis.create(turned)), 1).numpy()
    np.testing.assert_allclose(transformed, original @ wigner_D_numeric(1, rotation).T,
                               atol=2e-10, rtol=2e-10)
    inverted = atoms.copy()
    inverted.positions = -atoms.positions
    np.testing.assert_allclose(basis.create(inverted), -values,
                               atol=2e-10, rtol=2e-10)
    order = [2, 0, 3, 1]
    np.testing.assert_allclose(basis.create(atoms[order]), values[order],
                               atol=2e-10, rtol=2e-10)


def test_density_l3_collapsed_symmetric_block_rotates_and_inverts():
    basis = _density_basis(3, 3, nmax=1, lmax=1, partition=[3])
    atoms = Atoms("Ni4", positions=[[0, 0, 0], [.6, .2, .1],
                                    [.1, .7, .2], [.2, .1, .8]])
    values = basis.create(atoms)
    assert values.shape == (4, 1, 7)
    angle = .31
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    turned = atoms.copy()
    turned.positions = atoms.positions @ rotation.T
    original = real_tesseral_to_complex_multiplet(torch.as_tensor(values), 3).numpy()
    transformed = real_tesseral_to_complex_multiplet(
        torch.as_tensor(basis.create(turned)), 3).numpy()
    np.testing.assert_allclose(transformed, original @ wigner_D_numeric(3, rotation).T,
                               atol=2e-10, rtol=2e-10)
    inverted = atoms.copy()
    inverted.positions = -atoms.positions
    np.testing.assert_allclose(basis.create(inverted), -values,
                               atol=2e-10, rtol=2e-10)


def test_density_full_m_multispecies_columns_are_not_duplicated(tmp_path):
    basis = _density_basis(1, 1, species=("Ni", "Cu"))
    frames = []
    beta = np.array([.2, -.3, .5, .7])
    for shift in (.0, .07, -.06):
        atoms = Atoms("NiCuNi", positions=[[0, 0, 0],
                                           [.8 + shift, .2, .1],
                                           [.1, .9 - shift, .3]])
        atoms.new_array("target", np.einsum("nfm,f->nm", basis.create(atoms), beta))
        frames.append(atoms)
    assert len(basis.labels) == 4
    construction = basis._construction
    fit_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "density_two_species_full_m",
                     "status": "experimental"},
        "representation": construction["representation"],
        "basis": construction["basis"], "runtime": construction["runtime"],
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "target", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit(frames, config=fit_config)
    assert model._fitted["beta"].shape == (4,)
    assert model._fitted["fit_metadata"]["n_cols"] == 4
    restored = LinearModel.read(model.write(tmp_path / "two_species.ye3t.json"))
    for atoms in frames:
        np.testing.assert_allclose(restored.predict(atoms)["mean_real_tesseral"],
                                   atoms.arrays["target"], atol=2e-10, rtol=2e-10)


def test_covariant_density_tagged_combine_fails_during_resolution():
    density = _density_basis(1, 1)
    source = density._construction
    config = {
        "single_factors": source["basis"]["single_factors"],
        "components": {
            "ordinary": {"tensor_product": source["basis"]["tensor_product"],
                         "catalogue": source["basis"]["catalogue"]},
            "tagged": {
                "single_factors": {
                    "species": ["Ni"],
                    "radial": {"family": "shifted_jacobi", "cutoff_A": 1.4},
                    "chemical": {"kind": "explicit"}},
                "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {1: [0, 1]}},
                "catalogue": {"ranks": [1], "nmax_per_rank": {1: 2},
                              "lmax_per_rank": {1: 1},
                              "source_block_partitions_by_rank": {1: [[1]]}},
            },
        },
    }
    runtime = {"evaluator": "auto", "neighbors": "auto",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(config, representation=YE3TRepresentation.from_config(
        source["representation"]), runtime=runtime)
    assert all(item["basis_create_available"] for item in
               basis.resolution.capability_report["component_runtime"].values())
    assert not basis.resolution.capability_report["basis_create_available"]
    with pytest.raises(RuntimeError, match="supported"):
        basis.create(Atoms("Ni2", positions=[[0, 0, 0], [.5, .2, .1]]))


@pytest.mark.parametrize("L", [1, 3])
def test_density_full_m_ard_covariance_reloads_in_compiler_columns(L, tmp_path):
    pytest.importorskip("sklearn")
    basis = _density_basis(1, L, lmax=L, species=("Ni", "Cu"))
    frames = []
    truth = np.array([.2, -.3, .5, .7])
    for shift in (.0, .07, -.06, .12, -.11):
        atoms = Atoms("NiCuNi", positions=[[0, 0, 0],
                                           [.8 + shift, .2, .1],
                                           [.1, .9 - shift, .3]])
        atoms.new_array("target", np.einsum("nfm,f->nm", basis.create(atoms), truth))
        frames.append(atoms)
    construction = basis._construction
    fit_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "density_full_m_ard",
                     "status": "experimental"},
        "representation": construction["representation"],
        "basis": construction["basis"], "runtime": construction["runtime"],
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ard", "solver_options": {"max_iter": 200}}},
        "targets": {"per_atom": {"key": "target", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit(frames, config=fit_config)
    restored = LinearModel.read(model.write(tmp_path / "ard.ye3t.json"))
    posterior = model._fitted["fit_metadata"]["predictive_uncertainty"]
    active = np.asarray(posterior["active_column_indices"], dtype=int)
    sigma = np.asarray(posterior["coefficient_covariance_active"], dtype=float)
    expected = np.einsum("nam,ab,nbk->nmk", basis.create(frames[0])[:, active, :],
                         sigma, basis.create(frames[0])[:, active, :])
    predicted = restored.predict(frames[0], uncertainty=True)
    np.testing.assert_allclose(predicted["covariance_real_tesseral"], expected,
                               atol=2e-10, rtol=2e-10)
    probe = frames[0].copy()
    probe.calc = restored.ase_calculator()
    np.testing.assert_allclose(probe.calc.get_property(
        "per_atom_real_tesseral_covariance", probe), expected,
        atol=2e-10, rtol=2e-10)
    if L == 3:
        assert np.linalg.eigvalsh(expected).min() > -2e-10
        angle = .29
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        rotated = frames[0].copy()
        rotated.positions = frames[0].positions @ rotation.T
        complex_axes = real_tesseral_to_complex_multiplet(
            torch.eye(2 * L + 1, dtype=torch.float64), L).numpy()
        real_action = complex_multiplet_to_real_tesseral(
            torch.as_tensor(complex_axes @ wigner_D_numeric(L, rotation).T),
            L, range(-L, L + 1)).numpy()
        turned_covariance = restored.predict(rotated, uncertainty=True)[
            "covariance_real_tesseral"]
        expected_rotated = np.einsum("ai,nab,bj->nij", real_action,
                                     expected, real_action)
        np.testing.assert_allclose(turned_covariance, expected_rotated,
                                   atol=2e-9, rtol=2e-9)


def test_density_full_m_archive_rejects_changed_coefficients_and_axes(tmp_path):
    basis = _density_basis(1, 1)
    atoms = Atoms("Ni2", positions=[[0, 0, 0], [.8, .2, .1]])
    atoms.new_array("target", basis.create(atoms)[:, 0, :])
    source = basis._construction
    request = {
        "metadata": {"schema": "ye3t_config_v1", "name": "density_archive_tamper",
                     "status": "experimental"},
        "representation": source["representation"], "basis": source["basis"],
        "runtime": source["runtime"],
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "target", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": []},
    }
    saved = LinearModel(basis).fit([atoms], config=request).write(
        tmp_path / "valid.ye3t.json")
    payload = json.loads(saved.read_text())
    payload["compiled_density"]["specs_by_M"][0][0]["coeffs"][0][0] += .1
    corrupted = tmp_path / "changed.ye3t.json"
    corrupted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="self-hash"):
        LinearModel.read(corrupted)
    compiled = payload["compiled_density"]
    payload["compiler_hash"] = hashlib.sha256(json.dumps(
        compiled, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    payload["fit"]["fit_metadata"]["coordinate_penalty_metric"][
        "physical_image_plan_hash"] = payload["compiler_hash"]
    payload["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in payload.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    corrupted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="coefficient bytes"):
        LinearModel.read(corrupted)
    payload = json.loads(saved.read_text())
    for block in (0, 2):
        payload["compiled_density"]["specs_by_M"][block][0]["coeffs"][0][0] *= 1.25
    compiled = payload["compiled_density"]
    digest = hashlib.sha256()
    for rows in compiled["specs_by_M"]:
        digest.update(struct.pack("<q", len(rows)))
        for row in rows:
            values = np.asarray(row["coeffs"], dtype=np.float64)
            digest.update(struct.pack("<q", len(values)))
            digest.update(np.asarray(values[:, 0], dtype="<f8").tobytes())
            digest.update(np.asarray(values[:, 1], dtype="<f8").tobytes())
    compiled["coefficient_sha256"] = digest.hexdigest()
    payload["compiler_hash"] = hashlib.sha256(json.dumps(
        compiled, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    payload["fit"]["fit_metadata"]["coordinate_penalty_metric"][
        "physical_image_plan_hash"] = payload["compiler_hash"]
    payload["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in payload.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    corrupted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="rotation covariance"):
        LinearModel.read(corrupted)
    payload = json.loads(saved.read_text())
    payload["selected_coordinate_ids"] = ["wrong"]
    payload["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in payload.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    corrupted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ordered public coordinates"):
        LinearModel.read(corrupted)
    corrupted.write_bytes(saved.read_bytes().replace(
        b'"schema":', b'"schema":"duplicate","schema":', 1))
    with pytest.raises(ValueError, match="Duplicate JSON"):
        LinearModel.read(corrupted)
    for field in ("kappa0", "kappa", "m", "l_aux", "m_aux", "eta", "n"):
        payload = json.loads(saved.read_text())
        channel = payload["compiled_density"]["specs_by_M"][0][0]["channels"][0]
        channel[field] = 0 if field == "n" else 1
        payload["compiler_hash"] = hashlib.sha256(json.dumps(
            payload["compiled_density"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        payload["self_hash"] = hashlib.sha256(json.dumps(
            {key: value for key, value in payload.items() if key != "self_hash"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest()
        corrupted.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="source"):
            LinearModel.read(corrupted)


def test_density_full_m_fixed_embedding_round_trip(tmp_path):
    basis = _density_basis(
        1, 1, species=("Ni", "Cu", "Al"),
        chemical={"kind": "fixed_embedding",
                  "species_order": ["Ni", "Cu", "Al"],
                  "matrix": [[1.0, 0.0], [0.0, 1.0], [.5, -.25]]})
    atoms = Atoms("NiCuAl", positions=[[0, 0, 0], [.8, .2, .1],
                                       [.1, .9, .3]])
    values = basis.create(atoms)
    assert values.shape == (3, len(basis.labels), 3)
    assert len(basis.labels) == 6
    atoms.new_array("target", np.einsum(
        "nfm,f->nm", values, np.linspace(.1, .6, len(basis.labels))))
    construction = basis._construction
    fit_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "density_embedded_full_m",
                     "status": "experimental"},
        "representation": construction["representation"],
        "basis": construction["basis"], "runtime": construction["runtime"],
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "target", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit([atoms], config=fit_config)
    restored = LinearModel.read(model.write(tmp_path / "embedded.ye3t.json"))
    np.testing.assert_allclose(restored.predict(atoms)["mean_real_tesseral"],
                               atoms.arrays["target"], atol=2e-10, rtol=2e-10)
