import pytest
import json
import os
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel


def _config():
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [1, 2],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 1},
            "l_max_per_rank": {1: 0, 2: 0},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    basis = {
        "single_factors": {
            "species": ["Ni", "Cu", "Al"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 4.5,
                       "cutoff_width_A": 0.01, "lambda": 0.79},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "density"},
        "catalogue": {
            "ranks": [1, 2],
            "nmax_per_rank": {1: 4, 2: 4},
            "lmax_per_rank": {1: 0, 2: 0},
            "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]]},
            "selection": {"repeated_content_min": {1: 1, 2: 2}},
        },
    }
    runtime = {"evaluator": "native_cpu", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    return representation, basis, runtime


def test_configured_fit_workflow_paths_do_not_change_model_identity():
    from ye3t_methods.config import resolve_linear_fit_config

    representation, basis_config, runtime = _config()
    basis = Basis.from_config(basis_config, representation=representation, runtime=runtime)
    config = {
        "metadata": {
            "schema": "ye3t_config_v1", "name": "path_metadata", "status": "stable",
            "training_structures": "training.extxyz",
            "evaluation_structure": "probe.extxyz",
            "output_path": "first.pt",
        },
        "representation": representation.to_dict(), "basis": basis_config,
        "runtime": runtime,
        "model": {"kind": "linear",
                  "fit": {"solver": "ridge", "alpha": 1e-8,
                          "weights": {"energy": 1.0, "forces": 0.0}},
                  "reference_energy": {
                      "per_species_E0_eV": {"Ni": 0.0, "Cu": 0.0, "Al": 0.0},
                      "fit_E0": False,
                  }},
        "targets": {"energy": "energy", "forces": None, "stress": None},
        "validation": {"checks": []},
    }
    first = resolve_linear_fit_config(config, basis)
    config["metadata"]["training_structures"] = "different.extxyz"
    config["metadata"]["output_path"] = "second.pt"
    second = resolve_linear_fit_config(config, basis)
    assert first["resolved_fit_config_sha256"] == second["resolved_fit_config_sha256"]
    config["metadata"]["system"] = {"element": "Ni", "repeat": [2, 2, 2]}
    third = resolve_linear_fit_config(config, basis)
    assert first["resolved_fit_config_sha256"] == third["resolved_fit_config_sha256"]
    config["metadata"]["system"] = "Ni cell"
    with pytest.raises(ValueError, match="metadata.system"):
        resolve_linear_fit_config(config, basis)
    del config["metadata"]["system"]
    config["metadata"]["output_path"] = ""
    with pytest.raises(ValueError, match="metadata.output_path"):
        resolve_linear_fit_config(config, basis)


def test_configured_rank_four_tagged_angular_selection_matches_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [4],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {4: 1}, "l_max_per_rank": {4: 1}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    config = {
        "single_factors": {
            "species": ["Ta"],
            "radial": {"family": "shifted_jacobi", "cutoff_A": 4.8},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {4: [0, 2]}},
        "catalogue": {
            "ranks": [4], "nmax_per_rank": {4: 1}, "lmax_per_rank": {4: 1},
            "source_block_partitions_by_rank": {4: [[4]]},
            "angular_patterns_by_rank": {4: [[1, 1, 1, 1]]},
        },
    }
    runtime = {"evaluator": "reference", "neighbors": "auto",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    configured = Basis.from_config(config, representation=representation,
                                   runtime=runtime)
    unrestricted_config = {**config, "catalogue": {
        key: value for key, value in config["catalogue"].items()
        if key != "angular_patterns_by_rank"}}
    unrestricted = Basis.from_config(unrestricted_config,
                                     representation=representation, runtime=runtime)
    assert configured.resolution.sha256 != unrestricted.resolution.sha256
    assert configured.catalogue.counts()["by_component"]["main"][
        "raw_opportunity_count"] == 22
    assert unrestricted.catalogue.counts()["by_component"]["main"][
        "raw_opportunity_count"] == 38
    legacy = Basis(
        elements=["Ta"], source="tagged_cauchy_image", cutoff=4.8,
        rank=4, tag_counts=(0, 2), nmax_per_rank={4: 1},
        lmax_per_rank={4: 1},
        source_block_partitions_by_rank={4: ((4,),)},
        angular_patterns_by_rank={4: ((1, 1, 1, 1),)},
    )
    atoms = Atoms("Ta4", positions=((0, 0, 0), (1.5, 0.2, 0.1),
                                    (-0.4, 1.6, 0.3), (0.5, -0.3, 1.7)))
    assert len(configured.labels) == len(legacy.labels) == 7
    assert [label.identity for label in configured.labels] == [
        label.identity for label in legacy.labels]
    rows = configured.create(atoms)
    np.testing.assert_allclose(rows, legacy.create(atoms), rtol=0, atol=1e-12)
    rotated = atoms.copy()
    rotated.rotate(37, (1, 2, 3), center=(0, 0, 0))
    np.testing.assert_allclose(configured.create(rotated), rows, rtol=0, atol=1e-9)
    inverted = atoms.copy()
    inverted.positions *= -1
    np.testing.assert_allclose(configured.create(inverted), rows, rtol=0, atol=1e-10)
    reordered = atoms[[2, 0, 3, 1]]
    np.testing.assert_allclose(configured.create(reordered), rows[[2, 0, 3, 1]],
                               rtol=0, atol=1e-10)
    invalid = {**config, "catalogue": {**config["catalogue"],
               "angular_patterns_by_rank": {4: [[1, 1, 1]]}}}
    with pytest.raises(ValueError, match="Angular patterns"):
        Basis.from_config(invalid, representation=representation, runtime=runtime)


def test_configured_multi_rank_tagged_angular_selection_materializes(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    representation, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["single_factors"]["radial"] = {
        "family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {
        "kind": "tagged", "tag_counts_per_rank": {1: [0], 2: [0, 1]}}
    config["catalogue"] = {
        "ranks": [1, 2], "nmax_per_rank": {1: 1, 2: 1},
        "lmax_per_rank": {1: 0, 2: 1},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2]]},
        "angular_patterns_by_rank": {1: [[0]], 2: [[1, 1]]},
    }
    runtime.update(evaluator="reference", neighbors="auto",
                   cache={"mode": "auto"})
    basis = Basis.from_config(config, representation=representation,
                              runtime=runtime)
    assert basis.catalogue.counts()["by_component"]["main"][
        "raw_opportunity_count"] > 0
    atoms = Atoms("Ni4", positions=((0, 0, 0), (1.5, 0.2, 0.1),
                                    (-0.4, 1.6, 0.3), (0.5, -0.3, 1.7)))
    rows = basis.create(atoms)
    assert rows.shape == (4, len(basis.labels))
    assert {label.as_dict()["N"] for label in basis.labels} == {1, 2}
    rotated = atoms.copy()
    rotated.rotate(29, (2, 1, 3), center=(0, 0, 0))
    np.testing.assert_allclose(basis.create(rotated), rows, rtol=0, atol=1e-9)


def test_public_resolution_expands_physical_eta_and_previews_exact_density_count(monkeypatch):
    import ye3t.couplings

    rep, config, runtime = _config()
    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs: pytest.fail("compiled during preview"))
        basis = Basis.from_config(config, representation=rep, runtime=runtime)
        summary = basis.catalogue.repeated_content_summary()
        counts = basis.catalogue.counts()
    resolved = basis.resolution.to_dict()
    component = resolved["components"][0]
    assert len(component["physical_eta_by_center"]["Ni"]) == 12
    assert sum(len(rows) for rows in component["physical_eta_by_center"].values()) == 36
    assert {tuple(row["chemical"].values()) for row in component["physical_eta_by_center"]["Ni"]}
    assert component["physical_eta_by_center"]["Ni"][0]["native_pace_n"] == 1
    assert component["physical_eta_by_center"]["Ni"][3]["native_pace_n"] == 2
    assert component["physical_eta_by_center"]["Ni"][9]["native_pace_n"] == 4
    assert resolved["representation"]["uncoupled_factor_inputs"]["eta_count_per_rank"] == {"1": 12, "2": 12}
    assert len(basis.resolution.warnings) == 3
    assert "physically trivial block sectors" in basis.resolution.warnings[-1]
    assert summary["by_component"]["main"] == [
        {"rank": 1, "partition": [1], "candidate_fixed_contents_before_parity": 12},
        {"rank": 2, "partition": [2], "candidate_fixed_contents_before_parity": 12},
    ]
    assert counts["by_component"]["main"]["by_rank_per_center"] == {1: 12, 2: 12}
    assert counts["exact_total_per_center"] == 24
    assert counts["coefficient_materialization_performed"] is False
    assert basis.resolution.capability_report["basis_create_available"] is False
    assert basis.resolution.sha256 in str(basis.resolution)
    with pytest.raises(RuntimeError, match="P2 evaluator integration"):
        basis.create(None)
    assert Basis.from_config(config, representation=rep, runtime=runtime).resolution.sha256 == basis.resolution.sha256
    restored_rep = YE3TRepresentation.from_config(json.loads(json.dumps(rep.to_dict())))
    restored_basis = Basis.from_config(json.loads(json.dumps(config)), representation=restored_rep,
                                       runtime=json.loads(json.dumps(runtime)))
    assert restored_basis.resolution.sha256 == basis.resolution.sha256


def test_configured_ordinary_pace_rows_match_independent_product_derivative():
    from ye3t_methods.atomistic.equivariant_calc import ACECovariantEvaluator, neighbor_data_from_ase_atoms
    from ye3t_methods.atomistic.equivariant_calc.gradients import (
        descriptor_sum_position_jacobian_analytic_product,
    )

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["catalogue"]["nmax_per_rank"] = {1: 1, 2: 1}
    config["catalogue"]["lmax_per_rank"] = {1: 0, 2: 1}
    config["catalogue"]["selection"]["repeated_content_min"] = {1: 1, 2: 1}
    runtime["evaluator"] = "torch"
    runtime["cache"]["mode"] = "off"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"] is True
    assert basis.resolution.capability_report["selected_evaluator"] == "torch"
    expected = basis.catalogue.counts()["exact_total_per_center"]
    atoms = Atoms("Ni3", positions=((0.0, 0.0, 0.0), (1.7, 0.2, 0.4),
                                    (-0.3, 1.8, 0.6)))
    rows = basis.create(atoms)
    assert isinstance(rows, np.ndarray) and rows.dtype == np.dtype("float64")
    assert rows.shape == (len(atoms), expected) and np.isfinite(rows).all()
    assert np.linalg.norm(rows) > 0
    assert all(value == 0 for label in basis.labels
               for value in label.as_dict()["radial_indices"])
    assert all(channel["eta"]["native_pace_n"] == 1 for label in basis.labels
               for channel in label.as_dict()["one_factor_channels"])
    reordered = atoms[[2, 0, 1]]
    np.testing.assert_allclose(basis.create(reordered), rows[[2, 0, 1]], rtol=0, atol=1e-10)
    rotated = atoms.copy()
    rotated.rotate(37.0, "x", center=(0, 0, 0))
    np.testing.assert_allclose(basis.create(rotated), rows, rtol=0, atol=1e-9)

    periodic = atoms.copy()
    periodic.set_cell((5.0, 5.0, 5.0))
    periodic.pbc = True
    assert np.any(neighbor_list("S", periodic, 4.5) != 0)
    periodic_rows = basis.create(periodic)
    assert np.linalg.norm(periodic_rows - rows) > 1e-6
    descriptor = basis._descriptor
    neighbor = neighbor_data_from_ase_atoms(periodic, descriptor.cutoff,
                                            descriptor.type_map)
    evaluator = ACECovariantEvaluator(
        descriptor.site_basis_config, backend="pytorch", strict_backend=True,
        validate_backend=True, factorized_descriptor_runtime_policy="disable")
    with torch.no_grad():
        site, jacobian = descriptor_sum_position_jacobian_analytic_product(
            evaluator, torch.as_tensor(periodic.positions, dtype=torch.float64),
            torch.as_tensor(periodic.cell.array, dtype=torch.float64),
            torch.as_tensor(neighbor.edge_index, dtype=torch.long),
            torch.as_tensor(neighbor.atom_types, dtype=torch.long),
            descriptor.descriptor_specs,
            shifts=torch.as_tensor(neighbor.shifts, dtype=torch.float64),
            real_if_scalar=True)
    np.testing.assert_allclose(periodic_rows, np.asarray(site), rtol=0, atol=1e-10)
    step = 1e-5
    displaced = []
    for direction in (-1, 1):
        trial = periodic.copy()
        trial.positions[1, 0] += direction * step
        displaced.append(basis.create(trial).sum(axis=0))
    finite_difference = (displaced[1] - displaced[0]) / (2 * step)
    np.testing.assert_allclose(finite_difference, np.asarray(jacobian)[:, 3],
                               rtol=0, atol=2e-5)

    runtime["evaluator"] = "native_cpu"
    native = Basis.from_config(config, representation=rep, runtime=runtime)
    assert native.resolution.capability_report["basis_create_available"] is True
    assert native.resolution.capability_report["selected_evaluator"] == "native_cpu"
    np.testing.assert_allclose(native.create(atoms), rows, rtol=0, atol=1e-12)
    assert [label.identity for label in native.labels] == [label.identity for label in basis.labels]
    native_auto_neighbors = Basis.from_config(
        config, representation=rep, runtime=dict(runtime, neighbors="auto"))
    assert native_auto_neighbors.resolution.capability_report["selected_neighbors"] == "auto"
    runtime["evaluator"] = "reference"
    reference = Basis.from_config(config, representation=rep, runtime=runtime)
    assert reference.resolution.capability_report["basis_create_available"] is False
    with pytest.raises(RuntimeError, match="P2 evaluator integration"):
        reference.create(atoms)


def test_configured_ordinary_pace_labels_survive_model_round_trip(tmp_path):
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 2}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["selected_evaluator"] == "torch"
    training = []
    for distance in (1.2, 1.5, 1.8, 2.1):
        atoms = Atoms("Ni2", positions=((0, 0, 0), (distance, 0.1, 0.2)))
        columns = basis.create(atoms).sum(axis=0)
        atoms.info["energy"] = float(0.17 * len(atoms) +
                                      columns @ np.array([0.6, -0.2]))
        training.append(atoms)
    assert {value for label in basis.labels for value in
            label.as_dict()["radial_indices"]} == {0, 1}
    assert {channel["eta"]["native_pace_n"] for label in basis.labels
            for channel in label.as_dict()["one_factor_channels"]} == {1, 2}
    model = LinearModel(basis).fit(training, regularization=1e-10,
                                   force_weight=0.0)
    full_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "ordinary_ni", "status": "stable"},
        "representation": rep.to_dict(), "basis": config, "runtime": runtime,
        "model": {"kind": "linear",
                  "fit": {"solver": "ridge", "alpha": 1e-10,
                          "weights": {"energy": 1.0, "forces": 0.0}},
                  "reference_energy": {"per_species_E0_eV": {"Ni": 0.0},
                                       "fit_E0": True}},
        "targets": {"energy": "energy", "forces": None, "stress": None},
        "validation": {"checks": ["force_fd", "round_trip"]},
    }
    configured_model = LinearModel(basis).fit(training, config=full_config)
    np.testing.assert_allclose(configured_model._fitted.weight, model._fitted.weight,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(configured_model.reference_energies["Ni"],
                               model._fitted.bias, rtol=0, atol=1e-10)
    assert configured_model._fitted.bias == 0.0
    assert configured_model._fitted.fit_metadata["configured_validation"]["results"][
        "round_trip"]["passed"]
    configured_path = configured_model.write(tmp_path / "configured_ordinary.pt")
    configured_loaded = LinearModel.read(configured_path)
    assert configured_loaded._fitted.fit_metadata["resolved_fit_config"] == (
        configured_model._fitted.fit_metadata["resolved_fit_config"])
    with pytest.raises(ValueError, match="Density ASE evaluator"):
        model.ase_calculator(evaluator="reference")
    with pytest.raises(ValueError, match="evaluator must"):
        model.ase_calculator(evaluator="bogus")
    ase_probe = training[1].copy()
    ase_probe.calc = model.ase_calculator(evaluator="torch", neighbors="ase")
    assert np.isfinite(ase_probe.get_potential_energy())
    auto_probe = training[1].copy()
    auto_probe.calc = model.ase_calculator(evaluator="auto", neighbors="ase")
    np.testing.assert_allclose(auto_probe.get_potential_energy(),
                               ase_probe.get_potential_energy(), rtol=0, atol=1e-12)
    saved = model.write(tmp_path / "configured_pace.pt")
    loaded = LinearModel.read(saved)
    assert [label.as_dict() for label in loaded.labels] == [
        label.as_dict() for label in model.labels]
    loaded_auto = training[1].copy()
    loaded_auto.calc = loaded.ase_calculator(evaluator="auto", neighbors="ase")
    np.testing.assert_allclose(loaded_auto.get_potential_energy(),
                               ase_probe.get_potential_energy(), rtol=0, atol=1e-12)
    probe = training[1].copy()
    probe.calc = model.ase_calculator(evaluator="torch")
    replay = training[1].copy()
    replay.calc = loaded.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(replay.get_potential_energy(),
                               probe.get_potential_energy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(replay.get_forces(), probe.get_forces(),
                               rtol=0, atol=1e-12)
    altered = torch.load(saved, weights_only=False)
    altered["bundle"].fit_metadata[
        "ye3t_methods_public_label_convention"
    ]["ordered_public_labels_sha256"] = "0" * 64
    altered_path = tmp_path / "altered_pace.pt"
    torch.save(altered, altered_path)
    with pytest.raises(ValueError, match="public labels differ"):
        LinearModel.read(altered_path)
    legacy = torch.load(saved, weights_only=False)
    del legacy["bundle"].fit_metadata["ye3t_methods_public_label_convention"]
    legacy_path = tmp_path / "legacy_pace.pt"
    torch.save(legacy, legacy_path)
    old_labels = LinearModel.read(legacy_path).labels
    assert {value for label in old_labels for value in
            label.as_dict()["radial_indices"]} == {1, 2}


def test_configured_ordinary_pace_torch_native_energy_force_stress():
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 2}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
    }
    runtime["evaluator"] = "native_cpu"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["selected_evaluator"] == "native_cpu"
    training = []
    for distance in (1.3, 1.6, 1.9, 2.2):
        atoms = Atoms("Ni2", positions=((0, 0, 0), (distance, 0.2, 0.1)))
        columns = basis.create(atoms).sum(axis=0)
        atoms.info["energy"] = float(0.14 * len(atoms) +
                                      columns @ np.array([0.55, -0.18]))
        training.append(atoms)
    model = LinearModel(basis).fit(training, force_weight=0.0)
    for index, (cell, pbc) in enumerate((
            ((8.0, 8.0, 8.0), (True, True, True)),
            (((3.5, 0.1, 0.0), (0.8, 4.0, 0.0), (0.2, 0.3, 9.0)),
             (True, True, False)))):
        atoms = Atoms("Ni4", positions=((0, 0, 0), (1.5, 0.2, 0.1),
                                        (-0.4, 1.6, 0.3), (0.5, -0.3, 1.7)),
                      cell=cell, pbc=pbc)
        reference = atoms.copy()
        reference.calc = model.ase_calculator(backend="torch" if index == 0 else None,
                                               evaluator="torch" if index else None,
                                               neighbors="ase")
        native = atoms.copy()
        native.calc = model.ase_calculator(evaluator="auto" if index == 0 else "native_cpu",
                                           native_library=library)
        try:
            assert native.calc.native_runtime.neighbors == "ase"
            np.testing.assert_allclose(native.get_potential_energy(),
                                       reference.get_potential_energy(), rtol=0,
                                       atol=1e-9)
            np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                                       rtol=0, atol=1e-8)
            np.testing.assert_allclose(native.get_stress(), reference.get_stress(),
                                       rtol=0, atol=1e-8)
        finally:
            native.calc.close()
    overridden = model.ase_calculator(evaluator="auto", neighbors="auto",
                                       native_library=library)
    try:
        assert overridden.native_runtime.neighbors == "auto"
    finally:
        overridden.close()


def test_configured_ordinary_native_matscipy_policy():
    pytest.importorskip("matscipy", reason="requires optional matscipy neighbor backend")
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
    }
    runtime["neighbors"] = "matscipy"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"] is True
    assert basis.resolution.capability_report["selected_evaluator"] == "native_cpu"
    assert basis.resolution.capability_report["selected_neighbors"] == "matscipy"
    training = []
    for distance in (1.4, 1.8):
        atoms = Atoms("Ni2", positions=((0, 0, 0), (distance, 0.1, 0.2)))
        atoms.info["energy"] = float(0.2 * len(atoms) +
                                      0.3 * basis.create(atoms).sum())
        training.append(atoms)
    model = LinearModel(basis).fit(training, force_weight=0.0)
    probe = training[0].copy()
    probe.calc = model.ase_calculator(evaluator="torch")
    assert np.isfinite(probe.get_potential_energy())
    with pytest.raises(ValueError, match="Torch density ASE supports"):
        model.ase_calculator(evaluator="torch", neighbors="matscipy")


def test_configured_density_stress_fit_matches_six_periodic_strain_differences(tmp_path):
    from ye3t_methods.atomistic.ace.yace import read_yace
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    atoms = Atoms("Ni2", positions=((0.3, 0.4, 0.2), (1.65, 1.25, 0.8)),
                  cell=((5.0, 0.0, 0.0), (0.5, 5.2, 0.0), (0.2, 0.4, 5.1)),
                  pbc=True)
    coefficient = 0.41
    bias = -0.17

    def energy(structure):
        return coefficient * float(basis.create(structure).sum()) + bias * len(structure)

    voigt_axes = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    step = 1.0e-5
    target_stress = []
    for left, right in voigt_axes:
        values = []
        for sign in (-1, 1):
            strain = np.zeros((3, 3))
            strain[left, right] += sign * step / (2 if left != right else 1)
            if left != right:
                strain[right, left] += sign * step / 2
            transform = np.eye(3) + strain
            deformed = atoms.copy()
            deformed.set_positions(atoms.positions @ transform.T)
            deformed.set_cell(atoms.cell.array @ transform.T)
            values.append(energy(deformed))
        target_stress.append((values[1] - values[0]) / (2 * step * atoms.get_volume()))
    target_stress = np.asarray(target_stress)
    assert np.linalg.norm(target_stress) > 1e-6
    assert np.linalg.norm(target_stress[3:]) > 1e-8
    atoms.info["energy"] = energy(atoms)
    atoms.info["stress"] = target_stress
    target_forces = np.zeros((len(atoms), 3))
    for atom_index in range(len(atoms)):
        for axis in range(3):
            values = []
            for sign in (-1, 1):
                displaced = atoms.copy()
                displaced.positions[atom_index, axis] += sign * step
                values.append(energy(displaced))
            target_forces[atom_index, axis] = -(values[1] - values[0]) / (2 * step)
    atoms.arrays["forces"] = target_forces
    fitted = LinearModel(basis, reference_energies={"Ni": 0.0}).fit(
        [atoms], force_weight=1.0, stress_weight=1.0, regularization=1e-12)
    assert fitted._fitted.fit_metadata["normal_equations"]["n_rows"] == 13
    np.testing.assert_allclose(fitted._fitted.weight, [coefficient], atol=1e-5)
    probe = atoms.copy()
    probe.calc = fitted.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(probe.get_stress(), target_stress, rtol=0, atol=1e-7)
    np.testing.assert_allclose(probe.get_forces(), target_forces, rtol=0, atol=1e-7)
    np.testing.assert_allclose(probe.get_potential_energy(), atoms.info["energy"], atol=1e-7)
    legacy_fixed = LinearModel(basis, reference_energies={"Ni": bias}).fit(
        [atoms], force_weight=1.0, stress_weight=1.0, regularization=1e-12)
    legacy_fixed._fitted.fit_metadata.pop("reference_energy_targets")
    assert "reference_energy_targets" not in legacy_fixed._fitted.fit_metadata
    legacy_yace = legacy_fixed.export_lammps(tmp_path / "legacy_fixed_E0.yace")
    legacy_export = read_yace(legacy_yace, compatibility="lammps_pace_linear_v1")
    np.testing.assert_allclose(legacy_export["E0"],
                               [legacy_fixed._fitted.bias + bias], rtol=0, atol=1e-12)

    for solver in ("lasso", "ard"):
        full_config = {
            "metadata": {"schema": "ye3t_config_v1", "name": "density_stress"},
            "representation": rep.to_dict(), "basis": config, "runtime": runtime,
            "model": {"kind": "linear",
                      "fit": {"solver": solver,
                              **({"alpha": 1e-10} if solver == "lasso" else {}),
                              **({"solver_options": {"tol": 1e-12, "max_iter": 100000}}
                                 if solver == "lasso" else {}),
                              "weights": {"energy": 1.0, "forces": 1.0, "stress": 1.0}},
                      "reference_energy": {"per_species_E0_eV": {"Ni": 0.0},
                                           "fit_E0": True}},
            "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
            "validation": {"checks": ["round_trip"]},
        }
        configured = LinearModel(basis).fit([atoms], config=full_config)
        assert configured._fitted.fit_metadata["n_rows"] == 13
        assert configured._fitted.fit_metadata["configured_validation"]["results"][
            "round_trip"]["passed"]
        probe.calc = configured.ase_calculator(evaluator="torch")
        np.testing.assert_allclose(probe.get_stress(), target_stress, rtol=0, atol=2e-5)
        np.testing.assert_allclose(probe.get_forces(), target_forces, rtol=0, atol=2e-5)
        np.testing.assert_allclose(probe.get_potential_energy(), atoms.info["energy"], atol=2e-5)
        yace = configured.export_lammps(tmp_path / ("density_stress_" + solver + ".yace"))
        exported = read_yace(yace, compatibility="lammps_pace_linear_v1")
        np.testing.assert_allclose(exported["E0"],
                                   [configured.reference_energies["Ni"]], rtol=0, atol=1e-12)
        if solver == "lasso":
            assert configured._fitted.fit_metadata["e0_correction_prior"]["baseline_dependent"]
            library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
            if library and os.path.isfile(library):
                native = atoms.copy()
                native.calc = configured.ase_calculator(
                    evaluator="native_cpu", native_library=library)
                try:
                    np.testing.assert_allclose(native.get_potential_energy(),
                                               probe.get_potential_energy(), rtol=0, atol=1e-8)
                    np.testing.assert_allclose(native.get_forces(),
                                               probe.get_forces(), rtol=0, atol=1e-7)
                finally:
                    native.calc.close()
        saved = configured.write(tmp_path / ("density_stress_" + solver + ".pt"))
        restored = LinearModel.read(saved)
        if solver == "lasso":
            from ye3t_methods.atomistic.ace.linear_ace import (
                load_linear_ace_ase_bundle, save_linear_ace_ase_bundle,
            )
            original_e0 = configured.reference_energies["Ni"]
            configured.reference_energies["Ni"] = original_e0 + 0.1
            with pytest.raises(ValueError, match="reference energies disagree"):
                configured.export_lammps(tmp_path / "mismatched_E0.yace")
            with pytest.raises(ValueError, match="reference energies disagree"):
                configured.write(tmp_path / "mismatched_E0.pt")
            configured.reference_energies["Ni"] = original_e0
            bundle, cutoff, type_map, references = load_linear_ace_ase_bundle(saved)
            references["Ni"] += 0.1
            tampered = tmp_path / "tampered_E0.pt"
            save_linear_ace_ase_bundle(bundle, tampered, cutoff=cutoff,
                                       type_map=type_map, reference_energies=references)
            with pytest.raises(ValueError, match="reference energies disagree"):
                LinearModel.read(tampered)
            bundle, cutoff, type_map, references = load_linear_ace_ase_bundle(saved)
            bundle.fit_metadata["per_species_E0_eV"]["Ni"] += 0.1
            tampered_final = tmp_path / "tampered_final_E0.pt"
            save_linear_ace_ase_bundle(bundle, tampered_final, cutoff=cutoff,
                                       type_map=type_map, reference_energies=references)
            with pytest.raises(ValueError, match="fitted E0 metadata disagrees"):
                LinearModel.read(tampered_final)
        replay = atoms.copy()
        replay.calc = restored.ase_calculator(evaluator="torch")
        np.testing.assert_allclose(replay.get_stress(), probe.get_stress(), rtol=0, atol=1e-10)
        if solver == "ard":
            uncertainty = restored.predict_uncertainty(atoms)
            assert np.isfinite(uncertainty["atomic_energy_std_eV"]).all()
            assert np.isfinite(uncertainty["total_energy_std_eV"])
    artifact = fitted.write(tmp_path / "density_stress.pt")
    replay = LinearModel.read(artifact)
    probe.calc = replay.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(probe.get_stress(), target_stress, rtol=0, atol=1e-7)
    np.testing.assert_allclose(probe.get_forces(), target_forces, rtol=0, atol=1e-7)
    np.testing.assert_allclose(probe.get_potential_energy(), atoms.info["energy"], atol=1e-7)


def test_configured_density_fits_independent_species_offsets_and_preserves_fixed_map(tmp_path):
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert len(basis.labels) == 4
    coefficients = np.asarray([0.24, 0.315, 0.315, 0.39])
    true_offsets = {"Ni": -0.12, "Cu": 0.27}
    frames = []
    for symbols in ("Ni2", "Cu2", "NiCu", "CuNi"):
        for distance in (1.4, 1.9, 2.3):
            atoms = Atoms(symbols, positions=((0, 0, 0), (distance, 0.2, 0.3)))
            atoms.info["energy"] = (float(basis.create(atoms).sum(axis=0) @ coefficients)
                                    + sum(true_offsets[name]
                                          for name in atoms.get_chemical_symbols()))
            frames.append(atoms)
    full_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "species_E0"},
        "representation": rep.to_dict(), "basis": config, "runtime": runtime,
        "model": {"kind": "linear",
                  "fit": {"solver": "ridge", "alpha": 1e-12,
                          "weights": {"energy": 1.0, "forces": 0.0}},
                  "reference_energy": {"per_species_E0_eV": {"Ni": 0.03, "Cu": -0.04},
                                       "fit_E0": True}},
        "targets": {"energy": "energy", "forces": None, "stress": None},
        "validation": {"checks": ["round_trip"]},
    }
    fitted = LinearModel(basis).fit(frames, config=full_config)
    # In a two-atom mixed pair the two directed rank-one terms share one energy column.
    np.testing.assert_allclose(fitted._fitted.weight[[0, 3]], coefficients[[0, 3]], atol=1e-8)
    np.testing.assert_allclose(fitted._fitted.weight[1:3].sum(),
                               coefficients[1:3].sum(), atol=1e-8)
    np.testing.assert_allclose([fitted.reference_energies[name] for name in basis.elements],
                               [true_offsets[name] for name in basis.elements], atol=1e-8)
    assert fitted._fitted.bias == 0.0
    saved = fitted.write(tmp_path / "species_E0.pt")
    loaded = LinearModel.read(saved)
    for atoms in frames[::3]:
        probe = atoms.copy()
        probe.calc = loaded.ase_calculator(evaluator="torch")
        np.testing.assert_allclose(probe.get_potential_energy(), atoms.info["energy"], atol=1e-8)
    full_config["model"]["reference_energy"] = {
        "per_species_E0_eV": true_offsets, "fit_E0": False}
    fixed = LinearModel(basis).fit(frames, config=full_config)
    np.testing.assert_allclose(fixed._fitted.weight[[0, 3]], coefficients[[0, 3]], atol=1e-8)
    np.testing.assert_allclose(fixed._fitted.weight[1:3].sum(),
                               coefficients[1:3].sum(), atol=1e-8)
    assert fixed.reference_energies == true_offsets
    assert fixed._fitted.bias == 0.0
    ard_config = json.loads(json.dumps(full_config))
    ard_config["model"]["fit"] = {
        "solver": "ard", "solver_options": {"threshold_lambda": 1e12},
        "weights": {"energy": 1.0, "forces": 0.0},
    }
    ard_config["model"]["reference_energy"] = {
        "per_species_E0_eV": {"Ni": 0.03, "Cu": -0.04}, "fit_E0": True,
    }
    ard = LinearModel(basis).fit(frames, config=ard_config)
    assert ard._fitted.fit_metadata["e0_correction_prior"]["origin_eV"] == (
        {"Ni": 0.03, "Cu": -0.04})
    selected = frames[6]  # mixed NiCu frame exercises both offset columns
    symbols = np.asarray(selected.get_chemical_symbols())
    design = np.column_stack((basis.create(selected), *(
        (symbols == name).astype(float) for name in basis.elements)))
    posterior = ard._fitted.fit_metadata["predictive_uncertainty"]
    active = np.asarray(posterior["active_column_indices"], dtype=int)
    assert set(range(len(basis.labels), design.shape[1])).issubset(set(active))
    covariance = np.asarray(posterior["coefficient_covariance_active"])
    chosen = design[:, active]
    expected_atomic = np.sqrt(np.einsum("if,fg,ig->i", chosen, covariance, chosen))
    total_row = chosen.sum(axis=0)
    expected_total = np.sqrt(total_row @ covariance @ total_row)
    for checked in (ard, LinearModel.read(ard.write(tmp_path / "species_ard.pt"))):
        predicted = checked.predict_uncertainty(selected)
        np.testing.assert_allclose(predicted["atomic_energy_std_eV"],
                                   expected_atomic, rtol=0, atol=1e-12)
        np.testing.assert_allclose(predicted["total_energy_std_eV"],
                                   expected_total, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="full column rank"):
        LinearModel(basis).fit([atoms for atoms in frames if set(
            atoms.get_chemical_symbols()) == {"Ni"}], config={
                **full_config,
                "model": {**full_config["model"],
                          "reference_energy": {"per_species_E0_eV": true_offsets,
                                               "fit_E0": True}},
            })
def test_density_angular_stress_rows_match_each_feature_strain_difference():
    from ye3t_methods.atomistic.ace.linear_ace import build_linear_ace_normal_equations
    from ye3t_methods.atomistic.equivariant_calc import ACECovariantEvaluator, neighbor_data_from_ase_atoms
    from ye3t_methods.atomistic.equivariant_calc.cy_factor_product import CYFactorProductEvaluator

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["catalogue"]["nmax_per_rank"] = {1: 1, 2: 1}
    config["catalogue"]["lmax_per_rank"] = {1: 0, 2: 1}
    config["catalogue"]["selection"]["repeated_content_min"] = {1: 1, 2: 1}
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    atoms = Atoms("Ni3", positions=((0.1, 0.2, 0.4), (1.4, 0.7, 0.9),
                                    (0.6, 1.9, 1.2)),
                  cell=((5.1, 0.0, 0.0), (0.6, 5.2, 0.0), (0.2, 0.3, 5.3)),
                  pbc=True)
    expected_rows = basis.create(atoms)
    descriptor = basis._descriptor
    neighbor = neighbor_data_from_ase_atoms(atoms, descriptor.cutoff, descriptor.type_map)
    evaluator = ACECovariantEvaluator(
        descriptor.site_basis_config, backend="pytorch", strict_backend=True,
        validate_backend=True, factorized_descriptor_runtime_policy="disable")
    result = CYFactorProductEvaluator(evaluator).evaluate(
        positions=torch.as_tensor(atoms.positions, dtype=torch.float64),
        cell=torch.as_tensor(atoms.cell.array, dtype=torch.float64),
        edge_index=torch.as_tensor(neighbor.edge_index, dtype=torch.long),
        atom_types=torch.as_tensor(neighbor.atom_types, dtype=torch.long),
        descriptors=descriptor.descriptor_specs,
        shifts=torch.as_tensor(neighbor.shifts, dtype=torch.float64),
        materialize_force_jacobian=False, materialize_charge_jacobian=False,
        materialize_stress_jacobian=True)
    assert result.report["backend"] != "cy_factor_product_explicit_product_rule"
    np.testing.assert_allclose(result.descriptor_values.detach().numpy(),
                               expected_rows, rtol=0, atol=1e-10)
    strain = result.stress_jacobian.detach().numpy()
    symmetric = 0.5 * (strain + np.swapaxes(strain, 1, 2)) / atoms.get_volume()
    assert np.max(np.abs(strain - np.swapaxes(strain, 1, 2))) < 1e-9
    voigt_axes = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    assert any(label.as_dict()["input_angular_momenta"] == (1, 1)
               for label in basis.labels)
    for left, right in voigt_axes:
        values = []
        for sign in (-1, 1):
            perturbation = np.zeros((3, 3))
            perturbation[left, right] += sign * 5e-6 / (2 if left != right else 1)
            if left != right:
                perturbation[right, left] += sign * 5e-6 / 2
            transform = np.eye(3) + perturbation
            deformed = atoms.copy()
            deformed.set_positions(atoms.positions @ transform.T)
            deformed.set_cell(atoms.cell.array @ transform.T)
            values.append(basis.create(deformed).sum(axis=0))
        finite_difference = (values[1] - values[0]) / (1e-5 * atoms.get_volume())
        np.testing.assert_allclose(symmetric[:, left, right], finite_difference,
                                   rtol=0, atol=2e-8)
    coefficients = np.linspace(0.2, 0.4, expected_rows.shape[1])
    atoms.info["energy"] = float(expected_rows.sum(axis=0) @ coefficients - 0.13 * len(atoms))
    atoms.info["stress"] = np.asarray([symmetric[:, left, right] @ coefficients
                                         for left, right in voigt_axes])
    fitted = LinearModel(basis, reference_energies={"Ni": 0.0}).fit(
        [atoms], force_weight=0.0, stress_weight=1.0, regularization=1e-12)
    probe = atoms.copy()
    probe.calc = fitted.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(probe.get_stress(), atoms.info["stress"],
                               rtol=0, atol=1e-7)
    axis = np.asarray((1.0, 2.0, 3.0))
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(37.0)
    cross = np.asarray(((0.0, -axis[2], axis[1]),
                        (axis[2], 0.0, -axis[0]),
                        (-axis[1], axis[0], 0.0)))
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    rotated = atoms.copy()
    rotated.set_positions(atoms.positions @ rotation.T)
    rotated.set_cell(atoms.cell.array @ rotation.T)
    rotated.calc = fitted.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(rotated.get_potential_energy(), probe.get_potential_energy(),
                               rtol=0, atol=1e-9)
    sigma = probe.get_stress()
    tensor = np.asarray(((sigma[0], sigma[5], sigma[4]),
                         (sigma[5], sigma[1], sigma[3]),
                         (sigma[4], sigma[3], sigma[2])))
    rotated_sigma = rotated.get_stress()
    rotated_tensor = np.asarray(((rotated_sigma[0], rotated_sigma[5], rotated_sigma[4]),
                                 (rotated_sigma[5], rotated_sigma[1], rotated_sigma[3]),
                                 (rotated_sigma[4], rotated_sigma[3], rotated_sigma[2])))
    np.testing.assert_allclose(rotated_tensor, rotation @ tensor @ rotation.T,
                               rtol=0, atol=1e-8)

    rank_two = tuple(spec for spec in descriptor.descriptor_specs if spec.label.rank == 2)
    factorized_evaluator = ACECovariantEvaluator(
        descriptor.site_basis_config, backend="pytorch", strict_backend=True,
        validate_backend=True, factorized_descriptor_runtime_policy="require")
    compiled = factorized_evaluator._compile_descriptors(rank_two)
    assert compiled.factorized_plan is not None
    assert len(compiled.factorized_plan.active_descriptor_indices) == len(rank_two)
    options = {
        "positions": torch.as_tensor(atoms.positions, dtype=torch.float64),
        "cell": torch.as_tensor(atoms.cell.array, dtype=torch.float64),
        "edge_index": torch.as_tensor(neighbor.edge_index, dtype=torch.long),
        "atom_types": torch.as_tensor(neighbor.atom_types, dtype=torch.long),
        "descriptors": rank_two,
        "shifts": torch.as_tensor(neighbor.shifts, dtype=torch.float64),
        "materialize_force_jacobian": True,
        "materialize_charge_jacobian": False,
        "materialize_stress_jacobian": True,
    }
    factorized_rows = CYFactorProductEvaluator(factorized_evaluator).evaluate(**options)
    reference_rows = CYFactorProductEvaluator(evaluator).evaluate(**options)
    np.testing.assert_allclose(factorized_rows.descriptor_values.detach().numpy(),
                               reference_rows.descriptor_values.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(factorized_rows.force_jacobian.detach().numpy(),
                               reference_rows.force_jacobian.detach().numpy(), atol=1e-9)
    np.testing.assert_allclose(factorized_rows.stress_jacobian.detach().numpy(),
                               reference_rows.stress_jacobian.detach().numpy(), atol=1e-9)
    rank_two_indices = [index for index, spec in enumerate(descriptor.descriptor_specs)
                        if spec.label.rank == 2]
    displaced_values = []
    for sign in (-1, 1):
        displaced = atoms.copy()
        displaced.positions[1, 0] += sign * 5e-6
        displaced_values.append(basis.create(displaced).sum(axis=0)[rank_two_indices])
    np.testing.assert_allclose(factorized_rows.force_jacobian[:, 3].detach().numpy(),
                               (displaced_values[1] - displaced_values[0]) / 1e-5,
                               rtol=0, atol=2e-8)
    for mode in ("adjoint", "analytic_vjp", "factorized_analytic", "cyprime"):
        normal = build_linear_ace_normal_equations(
            [atoms], descriptors=descriptor.descriptor_specs,
            site_basis_config=descriptor.site_basis_config,
            cutoff=descriptor.cutoff, type_map=descriptor.type_map,
            force_weight=0.0, stress_weight=1.0,
            force_jacobian_mode=mode, return_metadata=True)
        assert normal["n_rows"] == 7


def test_configured_ordinary_pace_reference_inventory_and_parity():
    from ye3t.core.basis.validation import iter_canonical_leaf_labelings
    from ye3t.couplings import count, normalize_compact_label
    from ye3t_methods.atomistic.ace.catalogue_selection import resolve_ordinary_scalar_catalogues

    rep = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {3: 3}, "l_max_per_rank": {3: 1},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    _unused, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["catalogue"] = {
        "ranks": [3], "nmax_per_rank": {3: 3}, "lmax_per_rank": {3: 1},
        "source_block_partitions_by_rank": {3: [[1, 1, 1]]},
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    expected = basis.catalogue.counts()["exact_total_per_center"]
    request = {
        "tensor_orders": [3], "nmax_by_tensor_order": {3: 3},
        "lmax_by_tensor_order": {3: 1},
        "channel_multiplicity_partitions_by_order": {3: [[1, 1, 1]]},
        "target_descriptor_counts": [1000],
    }
    all_source, _ = resolve_ordinary_scalar_catalogues(request)
    assert any(sum(row["compact_label"]["l_tuple"]) % 2 for row in all_source["rows"])
    with pytest.raises(ValueError, match="target_parity"):
        resolve_ordinary_scalar_catalogues(request, target_parity="odd")
    assert basis.resolution.capability_report["basis_create_available"] is True
    atoms = Atoms("Ni4", positions=((0, 0, 0), (1.2, 0, 0),
                                     (0, 1.3, 0), (0, 0, 1.4)))
    rows = basis.create(atoms)
    assert rows.shape == (len(atoms), expected) and np.isfinite(rows).all()
    assert np.linalg.norm(rows) > 0
    assert all(sum(label.as_dict()["input_angular_momenta"]) % 2 == 0
               for label in basis.labels)
    reference_labels = []
    for content, angular in iter_canonical_leaf_labelings(
            3, range(1, 4), range(2), multiplicity_partitions=((1, 1, 1),)):
        if sum(angular) % 2:
            continue
        report = count({
            "content": content,
            "target_rotation": {"L_R": 0, "parity": "even", "group": "O3"},
            "target_permutation": "young:3", "carrier": "ACE_density",
            "metadata": {"input_Ls": angular},
        }, input_Ls=angular)
        reference_labels.extend(normalize_compact_label(value).to_dict()
                                for value in report.labels_for_target(0))
    encoded = lambda labels: sorted(json.dumps(label, sort_keys=True)
                                    for label in labels)
    assert encoded(reference_labels) == encoded(
        [spec.label.to_dict() for spec in basis._descriptor.descriptor_specs])
    rotated = atoms.copy()
    rotated.rotate(41.0, "y", center=(0, 0, 0))
    np.testing.assert_allclose(basis.create(rotated), rows, rtol=0, atol=1e-9)
    inverted = atoms.copy()
    inverted.positions *= -1
    np.testing.assert_allclose(basis.create(inverted), rows, rtol=0, atol=1e-9)
    reordered = atoms[[2, 0, 3, 1]]
    np.testing.assert_allclose(basis.create(reordered), rows[[2, 0, 3, 1]],
                               rtol=0, atol=1e-9)


def test_fixed_embedding_channels_and_capacity_are_physical():
    rep, config, runtime = _config()
    config["single_factors"]["chemical"] = {
        "kind": "fixed_embedding", "species_order": ["Ni", "Cu", "Al"],
        "matrix": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
    }
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    component = basis.resolution.to_dict()["components"][0]
    assert len(component["physical_eta_by_center"]["Ni"]) == 8
    assert basis.resolution.to_dict()["representation"]["uncoupled_factor_inputs"]["eta_count_per_rank"] == {"1": 8, "2": 8}
    assert basis.catalogue.counts()["exact_total_per_center"] == 16


def test_rank_one_fixed_embedding_materializes_and_replays_physical_rows(tmp_path):
    from ye3t.core.labels import SingleChannelLabel
    from ye3t_methods.atomistic.ace.linear_ace import (
        load_linear_ace_ase_bundle, save_linear_ace_ase_bundle,
    )
    from ye3t_methods.atomistic.equivariant_calc import SiteBasisV2, neighbor_data_from_ase_atoms
    from ye3t_methods.atomistic.equivariant_calc.site_basis_serialization import (
        deserialize_site_basis_config, serialize_site_basis_config,
    )

    rep, config, runtime = _config()
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
    }
    matrix = ((1.0, 0.0), (0.0, 1.0), (0.5, -0.25))
    config["single_factors"]["chemical"] = {
        "kind": "fixed_embedding", "species_order": ["Ni", "Cu", "Al"],
        "matrix": [list(row) for row in matrix],
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"]
    assert basis.resolution.capability_report["selected_evaluator"] == "torch"
    atoms = Atoms("NiCuAl", positions=((0.1, 0.2, 0.3), (1.5, 0.3, 0.1),
                                       (0.4, 1.8, 0.7)),
                  cell=(7.0, 7.1, 7.2), pbc=True)
    rows = basis.create(atoms)
    assert rows.shape == (3, 6) and np.linalg.norm(rows) > 1e-8
    assert all("chemical_column" in channel["eta"] and
               "neighbor_species" not in channel["eta"]
               for label in basis.labels
               for channel in label.as_dict()["one_factor_channels"])
    descriptor = basis._descriptor
    neighbor = neighbor_data_from_ase_atoms(atoms, descriptor.cutoff,
                                            descriptor.type_map)
    site_payload = serialize_site_basis_config(descriptor.site_basis_config)
    site_payload.pop("chemical_embedding")
    site_payload["chemical_basis"] = "delta"
    delta = SiteBasisV2(deserialize_site_basis_config(site_payload))
    channels = tuple(SingleChannelLabel(mu0=center, mu=neighbor_type,
                                        kappa0=0, kappa=0, n=1, l=0, m=0)
                     for center in range(3) for neighbor_type in range(3))
    displacements = (atoms.positions[neighbor.edge_index[1]] + neighbor.shifts -
                     atoms.positions[neighbor.edge_index[0]])
    _, delta_A = delta.compute_site_basis(
        torch.as_tensor(displacements, dtype=torch.float64),
        torch.as_tensor(neighbor.edge_index, dtype=torch.long),
        torch.as_tensor(neighbor.atom_types, dtype=torch.long), channels)
    projected = torch.einsum(
        "acn,nk->ack", delta_A.reshape(3, 3, 3),
        torch.tensor(matrix, dtype=torch.complex128)).detach().numpy().real
    projected_rows = np.stack([
        projected[:, basis.elements.index(label.as_dict()["one_factor_channels"][0]["eta"]["central_species"]),
                  label.as_dict()["one_factor_channels"][0]["eta"]["chemical_column"]]
        for label in basis.labels], axis=1)
    np.testing.assert_allclose(rows, projected_rows,
                               rtol=0, atol=1e-10)
    explicit_config = json.loads(json.dumps(config))
    explicit_config["single_factors"]["chemical"] = {"kind": "explicit"}
    explicit = Basis.from_config(explicit_config, representation=rep, runtime=runtime)
    explicit_rows = explicit.create(atoms)
    assert explicit_rows.shape == (3, 9)
    explicit_columns = {
        (label.as_dict()["one_factor_channels"][0]["eta"]["central_species"],
         label.as_dict()["one_factor_channels"][0]["eta"]["neighbor_species"]): index
        for index, label in enumerate(explicit.labels)}
    explicit_projected = np.stack([
        sum(matrix[neighbor_index][label.as_dict()["one_factor_channels"][0]["eta"]["chemical_column"]]
            * explicit_rows[:, explicit_columns[(
                label.as_dict()["one_factor_channels"][0]["eta"]["central_species"], neighbor)]]
            for neighbor_index, neighbor in enumerate(basis.elements))
        for label in basis.labels], axis=1)
    np.testing.assert_allclose(
        rows, explicit_projected,
        rtol=0, atol=1e-10)
    one_hot_config = json.loads(json.dumps(explicit_config))
    one_hot_config["single_factors"]["chemical"] = {"kind": "one_hot"}
    one_hot = Basis.from_config(one_hot_config, representation=rep, runtime=runtime)
    np.testing.assert_allclose(one_hot.create(atoms), explicit_rows, rtol=0, atol=1e-12)
    changed_config = json.loads(json.dumps(config))
    changed_config["single_factors"]["chemical"]["matrix"][2][1] = -0.2
    changed = Basis.from_config(changed_config, representation=rep, runtime=runtime)
    assert changed.resolution.sha256 != basis.resolution.sha256
    assert np.linalg.norm(changed.create(atoms) - rows) > 1e-5
    reordered = atoms[[2, 0, 1]]
    np.testing.assert_allclose(basis.create(reordered), rows[[2, 0, 1]],
                               rtol=0, atol=1e-9)
    training = []
    for shift in (0.0, 0.1, -0.1, 0.2):
        frame = atoms.copy()
        frame.positions[1, 0] += shift
        frame.info["energy"] = float(
            basis.create(frame).sum(axis=0) @ np.linspace(0.2, 0.7, 6) - 0.3)
        training.append(frame)
    model = LinearModel(basis).fit(training, force_weight=0.0)
    explicit_training = []
    for frame in training:
        copied = frame.copy()
        copied.info["energy"] = float(
            explicit.create(copied).sum(axis=0) @ np.linspace(0.1, 0.9, 9))
        explicit_training.append(copied)
    explicit_model = LinearModel(explicit).fit(explicit_training, force_weight=0.0)
    explicit_artifact = explicit_model.write(tmp_path / "explicit_multi.pt")
    explicit_loaded = LinearModel.read(explicit_artifact)
    explicit_probe = atoms.copy()
    explicit_probe.calc = explicit_model.ase_calculator(evaluator="torch")
    explicit_replay = atoms.copy()
    explicit_replay.calc = explicit_loaded.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(explicit_replay.get_potential_energy(),
                               explicit_probe.get_potential_energy(), atol=1e-10)
    with pytest.raises(ValueError, match="no validated native_cpu evaluator"):
        explicit_loaded.ase_calculator(evaluator="native_cpu")
    with pytest.raises(ValueError, match="no validated LAMMPS/YACE export"):
        explicit_model.export_lammps(tmp_path / "explicit_multi.yace")
    artifact = model.write(tmp_path / "embedded.pt")
    loaded = LinearModel.read(artifact)
    fitted, cutoff, type_map, references = load_linear_ace_ase_bundle(artifact)
    fitted.site_basis_config.chemical_embedding = (
        (1.0, 0.0), (0.0, 1.0), (0.5, -0.2))
    altered_artifact = tmp_path / "altered_embedded.pt"
    save_linear_ace_ase_bundle(fitted, altered_artifact, cutoff=cutoff,
                               type_map=type_map, reference_energies=references)
    with pytest.raises(ValueError, match="physical source differs"):
        LinearModel.read(altered_artifact)
    radial_changed, cutoff, type_map, references = load_linear_ace_ase_bundle(artifact)
    radial_changed.site_basis_config.rc = [
        float(value) + 0.05 for value in radial_changed.site_basis_config.rc
    ]
    altered_radial_artifact = tmp_path / "altered_radial.pt"
    save_linear_ace_ase_bundle(radial_changed, altered_radial_artifact,
                               cutoff=cutoff, type_map=type_map,
                               reference_energies=references)
    with pytest.raises(ValueError, match="physical source differs"):
        LinearModel.read(altered_radial_artifact)
    unbound, cutoff, type_map, references = load_linear_ace_ase_bundle(artifact)
    unbound.fit_metadata.pop("ye3t_methods_public_label_convention")
    unbound_artifact = tmp_path / "unbound_embedded.pt"
    save_linear_ace_ase_bundle(unbound, unbound_artifact, cutoff=cutoff,
                               type_map=type_map, reference_energies=references)
    with pytest.raises(ValueError, match="requires a v3 public source record"):
        LinearModel.read(unbound_artifact)
    assert [label.as_dict() for label in loaded.labels] == [
        label.as_dict() for label in model.labels]
    original = atoms.copy()
    original.calc = model.ase_calculator(evaluator="torch")
    restored = atoms.copy()
    restored.calc = loaded.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(restored.get_potential_energy(),
                               original.get_potential_energy(), atol=1e-10)
    np.testing.assert_allclose(restored.get_forces(), original.get_forces(), atol=1e-9)
    np.testing.assert_allclose(restored.get_stress(), original.get_stress(), atol=1e-9)
    displaced_energies = []
    for sign in (-1, 1):
        displaced = atoms.copy()
        displaced.positions[1, 0] += sign * 1e-5
        displaced.calc = model.ase_calculator(evaluator="torch")
        displaced_energies.append(displaced.get_potential_energy())
    np.testing.assert_allclose(original.get_forces()[1, 0],
                               -(displaced_energies[1] - displaced_energies[0]) / 2e-5,
                               rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="no validated native_cpu evaluator"):
        model.ase_calculator(evaluator="native_cpu")


def test_rank_two_physical_eta_partitions_bind_compiler_before_products():
    from ye3t_methods.atomistic.equivariant_calc import SiteBasisV2, neighbor_data_from_ase_atoms

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["catalogue"] = {
        "ranks": [2], "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 0},
        "source_block_partitions_by_rank": {2: [[2]]},
    }
    runtime["evaluator"] = "torch"
    atoms = Atoms("NiCuNi", positions=((0.0, 0.0, 0.0), (1.4, 0.2, 0.3),
                                       (-0.2, 1.6, 0.5)), cell=(7.0, 7.0, 7.0), pbc=True)
    repeated = Basis.from_config(config, representation=rep, runtime=runtime)
    assert repeated.resolution.capability_report["basis_create_available"]
    assert repeated.catalogue.counts()["exact_total_per_center"] == 2
    repeated_rows = repeated.create(atoms)
    assert repeated_rows.shape == (3, 4)
    repeated_species = {
        tuple(channel["eta"]["neighbor_species"] for channel in label.as_dict()["one_factor_channels"])
        for label in repeated.labels
    }
    assert repeated_species == {("Ni", "Ni"), ("Cu", "Cu")}
    assert all(label.as_dict()["compiler_content_ids"] in {(1, 1), (2, 2)}
               for label in repeated.labels)
    descriptor = repeated._descriptor
    neighbor = neighbor_data_from_ase_atoms(atoms, descriptor.cutoff, descriptor.type_map)
    displacements = (atoms.positions[neighbor.edge_index[1]] + neighbor.shifts -
                     atoms.positions[neighbor.edge_index[0]])
    specs = descriptor.descriptor_specs
    unique_channels = tuple(dict.fromkeys(channel for spec in specs for channel in spec.channels))
    _, single_factors = SiteBasisV2(descriptor.site_basis_config).compute_site_basis(
        torch.as_tensor(displacements, dtype=torch.float64),
        torch.as_tensor(neighbor.edge_index, dtype=torch.long),
        torch.as_tensor(neighbor.atom_types, dtype=torch.long), unique_channels)
    by_channel = {channel: single_factors[:, index] for index, channel in enumerate(unique_channels)}
    direct = np.stack([
        (sum(complex(coeff) * by_channel[spec.channels[0]] *
             by_channel[spec.channels[1]] for coeff in spec.coeffs))
        .detach().numpy().real
        for spec in specs], axis=1)
    np.testing.assert_allclose(repeated_rows, direct, rtol=0, atol=1e-10)
    mixed_config = json.loads(json.dumps(config))
    mixed_config["catalogue"]["source_block_partitions_by_rank"] = {2: [[1, 1]]}
    mixed = Basis.from_config(mixed_config, representation=rep, runtime=runtime)
    assert mixed.catalogue.counts()["exact_total_per_center"] == 1
    mixed_rows = mixed.create(atoms)
    assert mixed_rows.shape == (3, 2)
    assert {tuple(channel["eta"]["neighbor_species"]
                  for channel in label.as_dict()["one_factor_channels"])
            for label in mixed.labels} == {("Ni", "Cu")}
    rotated = atoms.copy()
    rotated.rotate(37.0, "x", center=(0, 0, 0), rotate_cell=True)
    np.testing.assert_allclose(mixed.create(rotated), mixed_rows, rtol=0, atol=1e-9)
    reordered = atoms[[2, 0, 1]]
    np.testing.assert_allclose(mixed.create(reordered), mixed_rows[[2, 0, 1]],
                               rtol=0, atol=1e-9)


def test_multirank_physical_eta_prefix_identity_fit_and_replay(tmp_path):
    from ye3t_methods.atomistic.ace.linear_ace import (
        load_linear_ace_ase_bundle, save_linear_ace_ase_bundle,
    )

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["catalogue"] = {
        "ranks": [1, 2], "nmax_per_rank": {1: 1, 2: 2},
        "lmax_per_rank": {1: 0, 2: 0},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]]},
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    resolved = basis.resolution.to_dict()["components"][0]
    assert resolved["active_compiler_content_ids_by_rank"]["1"] == [1, 2]
    assert resolved["active_compiler_content_ids_by_rank"]["2"] == [1, 2, 3, 4]
    counts = basis.catalogue.counts()["by_component"]["main"]["by_rank_per_center"]
    assert counts == {1: 2, 2: 10}
    atoms = Atoms("NiCuNi", positions=((0.1, 0.2, 0.3), (1.4, 0.4, 0.2),
                                       (0.3, 1.7, 0.5)),
                  cell=((7.0, 0.1, 0.0), (0.0, 7.1, 0.2), (0.1, 0.0, 7.2)), pbc=True)
    rows = basis.create(atoms)
    assert rows.shape == (3, 24)
    from ye3t_methods.atomistic.ace.linear_ace import linear_ace_fit_preflight
    bindings = tuple((row["compiler_content_id"],
                      row["chemical"]["chemical_index"], row["native_pace_n"])
                     for row in resolved["physical_eta_by_center"]["Ni"])
    preflight = linear_ace_fit_preflight(
        basis._descriptor.compact_labels, settings=basis._descriptor.settings,
        physical_content_channels=bindings)
    assert preflight["logical_fit_feature_count_total"] == len(basis._descriptor.descriptor_specs)
    assert preflight["logical_fit_feature_count_by_rank"] == {"1": 4, "2": 20}
    assert preflight["emitted_yace_function_count"] is None
    rank_one_ids = {label.as_dict()["compiler_content_ids"]
                    for label in basis.labels if label.as_dict()["N"] == 1}
    assert rank_one_ids == {(1,), (2,)}
    assert all(tuple(channel["eta"]["radial_index"] for channel in
                     label.as_dict()["one_factor_channels"]) ==
               label.as_dict()["radial_indices"] for label in basis.labels)
    rank_one_config = json.loads(json.dumps(config))
    rank_one_config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
    }
    rank_one = Basis.from_config(rank_one_config, representation=rep, runtime=runtime)
    rank_one_rows = rank_one.create(atoms)
    multirank_columns = {label.identity: index for index, label in enumerate(basis.labels)
                         if label.as_dict()["N"] == 1}
    assert {label.identity for label in rank_one.labels} == set(multirank_columns)
    np.testing.assert_allclose(rank_one_rows,
                               rows[:, [multirank_columns[label.identity]
                                        for label in rank_one.labels]],
                               rtol=0, atol=1e-12)
    one_hot_config = json.loads(json.dumps(config))
    one_hot_config["single_factors"]["chemical"] = {"kind": "one_hot"}
    one_hot = Basis.from_config(one_hot_config, representation=rep, runtime=runtime)
    np.testing.assert_allclose(one_hot.create(atoms), rows, rtol=0, atol=1e-12)
    training = []
    for shift in (0.0, 0.12, -0.07, 0.19):
        frame = atoms.copy()
        frame.positions[1, 0] += shift
        frame.info["energy"] = float(basis.create(frame).sum(axis=0) @
                                     np.linspace(0.03, 0.27, rows.shape[1]))
        training.append(frame)
    model = LinearModel(basis).fit(training, force_weight=0.0, regularization=1e-8)
    assert model._fitted.fit_metadata["ye3t_methods_public_label_convention"]["schema"] == (
        "ye3t_methods_density_pace_labels_v3")
    artifact = model.write(tmp_path / "multirank.pt")
    loaded = LinearModel.read(artifact)
    assert loaded.basis.resolved["nmax"] == (1, 2)
    assert loaded.basis.resolved["compiler_content_capacity_nmax"] == (2, 4)
    assert [label.as_dict() for label in loaded.labels] == [
        label.as_dict() for label in model.labels]
    original = atoms.copy()
    original.calc = model.ase_calculator(evaluator="torch")
    restored = atoms.copy()
    restored.calc = loaded.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(restored.get_potential_energy(),
                               original.get_potential_energy(), atol=1e-9)
    np.testing.assert_allclose(restored.get_forces(), original.get_forces(), atol=1e-8)
    np.testing.assert_allclose(restored.get_stress(), original.get_stress(), atol=1e-8)
    step = 1e-5
    displaced_energies = []
    for direction in (-1, 1):
        displaced = atoms.copy()
        displaced.positions[1, 0] += direction * step
        displaced.calc = model.ase_calculator(evaluator="torch")
        displaced_energies.append(displaced.get_potential_energy())
    np.testing.assert_allclose(original.get_forces()[1, 0],
                               -(displaced_energies[1] - displaced_energies[0]) / (2 * step),
                               rtol=0, atol=5e-6)
    for index, (first, second) in enumerate(((0, 0), (1, 1), (2, 2),
                                              (1, 2), (0, 2), (0, 1))):
        strained_energies = []
        for direction in (-1, 1):
            strain = np.zeros((3, 3))
            strain[first, second] = direction * step
            if first != second:
                strain[second, first] = direction * step
            deformed = atoms.copy()
            deformed.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + strain),
                              scale_atoms=True)
            deformed.calc = model.ase_calculator(evaluator="torch")
            strained_energies.append(deformed.get_potential_energy())
        multiplier = 2 if first != second else 1
        derivative = (strained_energies[1] - strained_energies[0]) / (
            2 * step * atoms.get_volume() * multiplier)
        np.testing.assert_allclose(original.get_stress()[index], derivative,
                                   rtol=0, atol=5e-6)
    with pytest.raises(ValueError, match="no validated native_cpu evaluator"):
        loaded.ase_calculator(evaluator="native_cpu")
    with pytest.raises(ValueError, match="no validated LAMMPS/YACE export"):
        loaded.export_lammps(tmp_path / "multirank.yace")
    with pytest.raises(ValueError, match="no validated physical eta binding lowering"):
        loaded._fitted.export_lammps(tmp_path / "raw_multirank.yace",
                                     elements=loaded.basis.elements)
    changed, cutoff, type_map, references = load_linear_ace_ase_bundle(artifact)
    changed.fit_metadata["ye3t_methods_public_label_convention"][
        "physical_eta_binding_sha256"] = "0" * 64
    altered = tmp_path / "altered_multirank.pt"
    save_linear_ace_ase_bundle(changed, altered, cutoff=cutoff,
                               type_map=type_map, reference_energies=references)
    with pytest.raises(ValueError, match="physical eta binding differs"):
        LinearModel.read(altered)
    changed, cutoff, type_map, references = load_linear_ace_ase_bundle(artifact)
    changed.fit_metadata["ye3t_methods_public_label_convention"][
        "physical_radial_nmax_per_rank"]["1"] = 2
    altered_caps = tmp_path / "altered_caps.pt"
    save_linear_ace_ase_bundle(changed, altered_caps, cutoff=cutoff,
                               type_map=type_map, reference_energies=references)
    with pytest.raises(ValueError, match="physical eta binding differs"):
        LinearModel.read(altered_caps)


def test_rank_three_repeated_physical_eta_keeps_compiler_partition():
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {3: 2}, "l_max_per_rank": {3: 0},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    _, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["catalogue"] = {
        "ranks": [3], "nmax_per_rank": {3: 1}, "lmax_per_rank": {3: 0},
        "source_block_partitions_by_rank": {3: [[2, 1]]},
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=representation, runtime=runtime)
    assert basis.catalogue.counts()["exact_total_per_center"] == 2
    atoms = Atoms("NiCuNi", positions=((0, 0, 0), (1.5, 0.1, 0.2),
                                       (0.2, 1.6, 0.3)))
    rows = basis.create(atoms)
    assert rows.shape == (3, 4)
    assert {label.as_dict()["compiler_content_ids"] for label in basis.labels} == {
        (1, 1, 2), (1, 2, 2)}
    for label in basis.labels:
        fields = label.as_dict()
        channels = fields["one_factor_channels"]
        assert sorted((fields["compiler_content_ids"].count(content_id)
                       for content_id in set(fields["compiler_content_ids"])), reverse=True) == [2, 1]
        assert len({(channel["eta"]["neighbor_species"],
                     channel["eta"]["radial_index"], channel["l"])
                    for channel in channels}) == 2
    reordered = atoms[[2, 0, 1]]
    np.testing.assert_allclose(basis.create(reordered), rows[[2, 0, 1]],
                               rtol=0, atol=1e-9)


def test_ordinary_catalogue_selects_only_explicit_noncontiguous_content_ids():
    from ye3t_methods.atomistic.ace.catalogue_selection import resolve_ordinary_scalar_catalogues

    request = {
        "tensor_orders": (1,),
        "nmax_by_tensor_order": {1: 3},
        "content_ids_by_tensor_order": {1: [1, 3]},
        "lmax_by_tensor_order": {1: 0},
        "channel_multiplicity_partitions_by_order": {1: [[1]]},
        "target_descriptor_counts": [2],
    }
    source, manifest = resolve_ordinary_scalar_catalogues(
        request, catalogue_id="explicit_eta_gap", target_parity="even")
    assert [row["compact_label"]["n_tuple"] for row in source["rows"]] == [
        [1], [3]]
    assert manifest["preflight"]["candidate_fixed_contents"] == 2


def test_rank_two_fixed_embedding_is_projection_of_delta_single_factors():
    from ye3t.core.labels import SingleChannelLabel
    from ye3t_methods.atomistic.equivariant_calc import SiteBasisV2, neighbor_data_from_ase_atoms
    from ye3t_methods.atomistic.equivariant_calc.site_basis_serialization import (
        deserialize_site_basis_config, serialize_site_basis_config,
    )

    rep, config, runtime = _config()
    config["catalogue"] = {
        "ranks": [2], "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 0},
        "source_block_partitions_by_rank": {2: [[2], [1, 1]]},
    }
    matrix = np.asarray(((1.0, 0.1), (0.3, 1.0), (0.5, -0.2)))
    config["single_factors"]["chemical"] = {
        "kind": "fixed_embedding", "species_order": ["Ni", "Cu", "Al"],
        "matrix": matrix.tolist(),
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    atoms = Atoms("NiCuAl", positions=((0.0, 0.0, 0.0), (1.4, 0.1, 0.2),
                                       (0.3, 1.7, 0.4)))
    rows = basis.create(atoms)
    assert rows.shape == (3, 9)
    descriptor = basis._descriptor
    neighbor = neighbor_data_from_ase_atoms(atoms, descriptor.cutoff, descriptor.type_map)
    displacements = (atoms.positions[neighbor.edge_index[1]] + neighbor.shifts -
                     atoms.positions[neighbor.edge_index[0]])
    site_payload = serialize_site_basis_config(descriptor.site_basis_config)
    site_payload.pop("chemical_embedding")
    site_payload["chemical_basis"] = "delta"
    delta = SiteBasisV2(deserialize_site_basis_config(site_payload))
    channels = tuple(SingleChannelLabel(mu0=center, mu=neighbor_type,
                                        kappa0=0, kappa=0, n=1, l=0, m=0)
                     for center in range(3) for neighbor_type in range(3))
    _, delta_A = delta.compute_site_basis(
        torch.as_tensor(displacements, dtype=torch.float64),
        torch.as_tensor(neighbor.edge_index, dtype=torch.long),
        torch.as_tensor(neighbor.atom_types, dtype=torch.long), channels)
    projected = torch.einsum("acn,nk->ack", delta_A.reshape(3, 3, 3),
                             torch.as_tensor(matrix, dtype=torch.complex128))
    direct = []
    for spec in descriptor.descriptor_specs:
        center = int(spec.channels[0].mu0)
        first, second = (projected[:, center, int(channel.mu)]
                         for channel in spec.channels)
        direct.append((sum(complex(coeff) * first * second
                           for coeff in spec.coeffs)).detach().numpy().real)
    np.testing.assert_allclose(rows, np.stack(direct, axis=1), rtol=0, atol=1e-10)
    assert all("chemical_column" in channel["eta"] and
               "neighbor_species" not in channel["eta"]
               for label in basis.labels
               for channel in label.as_dict()["one_factor_channels"])


def test_rank_two_mixed_angular_physical_eta_rotation_and_inversion():
    from ye3t_methods.atomistic.equivariant_calc import ACECovariantEvaluator

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["catalogue"] = {
        "ranks": [2], "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: [[2], [1, 1]]},
    }
    runtime["evaluator"] = "torch"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    expected = basis.catalogue.counts()["exact_total_per_center"]
    atoms = Atoms("NiCuNi", positions=((0.1, 0.2, 0.3), (1.3, 0.4, 0.1),
                                       (0.2, 1.7, 0.6)))
    rows = basis.create(atoms)
    assert rows.shape == (3, 2 * expected)
    strict_evaluator = ACECovariantEvaluator(
        basis._descriptor.site_basis_config,
        factorized_descriptor_runtime_policy="require")
    with pytest.raises(RuntimeError, match="does not support bound physical eta"):
        strict_evaluator._compile_descriptors(basis._descriptor.descriptor_specs)
    assert {tuple(spec.label.l_tuple) for spec in basis._descriptor.descriptor_specs} == {
        (0, 0), (1, 1)}
    rotated = atoms.copy()
    rotated.rotate(29.0, "y", center=(0, 0, 0))
    np.testing.assert_allclose(basis.create(rotated), rows, rtol=0, atol=1e-9)
    inverted = atoms.copy()
    inverted.positions *= -1
    np.testing.assert_allclose(basis.create(inverted), rows, rtol=0, atol=1e-9)
    reordered = atoms[[2, 0, 1]]
    np.testing.assert_allclose(basis.create(reordered), rows[[2, 0, 1]],
                               rtol=0, atol=1e-9)


def test_named_components_keep_separate_radial_caps_and_expand_angular_hint():
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    ordinary = {
        "tensor_product": {"kind": "density"},
        "catalogue": {
            "ranks": [1], "nmax_per_rank": {1: 2}, "lmax_per_rank": {1: 2},
            "source_block_partitions_by_rank": {1: [[1]]},
        },
    }
    tagged = {
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {2: [0, 2]}},
        "catalogue": {
            "ranks": [2], "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 0},
            "source_block_partitions_by_rank": {2: [[2]]},
        },
    }
    config.pop("tensor_product")
    config.pop("catalogue")
    config["components"] = {"ordinary": ordinary, "tagged": tagged}
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    resolved = basis.resolution.to_dict()
    assert [row["name"] for row in resolved["components"]] == ["ordinary", "tagged"]
    assert len(resolved["components"][0]["physical_eta_by_center"]["Ni"]) == 2
    assert len(resolved["components"][1]["physical_eta_by_center"]["Ni"]) == 1
    assert resolved["representation"]["uncoupled_factor_inputs"]["l_max_per_rank"] == {"1": 2, "2": 0}
    assert basis.catalogue.counts()["by_component"]["ordinary"]["by_rank_per_center"] == {1: 2}
    assert basis.catalogue.counts()["by_component"]["tagged"]["status"] == "unsupported_tagged_source_family"


def test_fixed_embedding_rejects_dependent_columns():
    rep, config, runtime = _config()
    config["single_factors"]["chemical"] = {
        "kind": "fixed_embedding", "species_order": ["Ni", "Cu", "Al"],
        "matrix": [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
    }
    with pytest.raises(ValueError, match="independent column rank"):
        Basis.from_config(config, representation=rep, runtime=runtime)


def test_one_hot_keeps_chemical_choice_in_resolution_identity():
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    explicit = Basis.from_config(config, representation=rep, runtime=runtime)
    config["single_factors"]["chemical"] = {"kind": "one_hot"}
    one_hot = Basis.from_config(config, representation=rep, runtime=runtime)
    assert explicit.catalogue.counts()["exact_total_per_center"] == one_hot.catalogue.counts()["exact_total_per_center"]
    assert explicit.resolution.sha256 != one_hot.resolution.sha256
    assert explicit.resolution.capability_report["basis_create_available"]
    assert one_hot.resolution.capability_report["basis_create_available"]
    atoms = Atoms("Ni2", positions=((0, 0, 0), (1.7, 0.1, 0.2)))
    np.testing.assert_allclose(one_hot.create(atoms), explicit.create(atoms),
                               rtol=0, atol=1e-12)


def test_density_rejects_nontrivial_parent_and_block_sectors():
    rep, config, runtime = _config()
    rep_config = rep.to_dict()
    rep_config["parent"]["young_lambda"] = {1: [1], 2: [1, 1]}
    with pytest.raises(ValueError, match="nontrivial global Young parent"):
        Basis.from_config(config, representation=YE3TRepresentation.from_config(rep_config), runtime=runtime)
    rep_config = rep.to_dict()
    rep_config["intermediates"]["young_kappa"] = {
        "policy": "explicit", "by_block_size": {2: [[1, 1]]}
    }
    with pytest.raises(ValueError, match="nontrivial block Young"):
        Basis.from_config(config, representation=YE3TRepresentation.from_config(rep_config), runtime=runtime)
    rep_config = rep.to_dict()
    rep_config["intermediates"]["young_kappa"] = "symmetric_only"
    basis = Basis.from_config(config, representation=YE3TRepresentation.from_config(rep_config), runtime=runtime)
    assert basis.catalogue.counts()["exact_total_per_center"] == 24


def test_tagged_rejects_nontrivial_global_parent():
    rep, config, runtime = _config()
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {1: [0], 2: [0, 2]}}
    rep_config = rep.to_dict()
    rep_config["parent"]["young_lambda"] = {1: [1], 2: [1, 1]}
    with pytest.raises(ValueError, match="Pooled tagged density"):
        Basis.from_config(config, representation=YE3TRepresentation.from_config(rep_config), runtime=runtime)


def test_zero_symmetry_compatible_rank_errors_on_count():
    rep, config, runtime = _config()
    rep_config = rep.to_dict()
    rep_config["parent"]["parity"] = "odd"
    basis = Basis.from_config(config, representation=YE3TRepresentation.from_config(rep_config), runtime=runtime)
    with pytest.raises(ValueError, match="no symmetry-compatible compiler labels"):
        basis.catalogue.counts()


def test_public_config_rejects_bad_source_backend_and_legacy_selector_collision():
    from ye3t_methods import YE3TRepresentation as LegacySelector

    rep, config, runtime = _config()
    config["single_factors"]["radial"]["lambda"] = -1
    with pytest.raises(ValueError, match="radial parameters"):
        Basis.from_config(config, representation=rep, runtime=runtime)
    rep, config, runtime = _config()
    runtime["device"] = "cuda:0"
    with pytest.raises(ValueError, match="native_cpu evaluator"):
        Basis.from_config(config, representation=rep, runtime=runtime)
    with pytest.raises(ValueError, match="belongs to ye3t.YE3TRepresentation"):
        LegacySelector.from_config(rep.to_dict())
    with pytest.raises(TypeError, match="must be ye3t.YE3TRepresentation"):
        Basis.from_config(config, representation=LegacySelector.ace(), runtime=_config()[2])


def test_tagged_preview_marks_physical_image_count_pending():
    rep, config, runtime = _config()
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {1: [0], 2: [0, 2]}}
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    report = basis.catalogue.counts()
    assert report["by_component"]["main"]["status"] == "unsupported_tagged_source_family"
    assert report["exact_total_per_center"] is None
    assert basis.resolution.capability_report["component_count_status"]["main"] == "unsupported_tagged_source_family"


def test_shifted_jacobi_tagged_preview_reports_raw_upper_bound_only(monkeypatch):
    import ye3t.couplings

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {1: [0]}}
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
        "selection": {"repeated_content_min": 1},
    }
    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs: pytest.fail("compiled during preview"))
        basis = Basis.from_config(config, representation=rep, runtime=runtime)
        result = basis.catalogue.counts()["by_component"]["main"]
    assert result["status"] == "raw_upper_bound_deferred_to_compile"
    assert result["raw_opportunity_count"] == 1
    assert result["physical_image_upper_bound"] == 1
    assert result["exact_image_count"] is None


def test_single_species_tagged_config_materializes_compiler_image(monkeypatch, tmp_path):
    import ye3t.couplings

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {2: [0, 2]}}
    config["catalogue"] = {
        "ranks": [2], "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: [[2]]},
    }
    runtime = {"evaluator": "reference", "neighbors": "auto",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs: pytest.fail("compiled during preview"))
        basis = Basis.from_config(config, representation=rep, runtime=runtime)
        preview = basis.catalogue.counts()["by_component"]["main"]
    assert basis.resolution.capability_report["basis_create_available"] is True
    assert preview["exact_image_count"] is None
    atoms = Atoms("Ni4", positions=((0, 0, 0), (1.5, 0.2, 0.4),
                                   (-0.3, 1.6, 0.7), (0.8, -0.4, 1.8)))
    rows = np.asarray(basis.create(atoms))
    assert basis.source == "tagged_cauchy_image"
    assert basis.resolution.capability_report["selected_evaluator"] == "reference"
    assert basis.resolution.capability_report["selected_neighbors"] == (
        "directed_edges_all_images_bruteforce")
    assert rows.shape == (len(atoms), len(basis.labels))
    assert np.isrealobj(rows) and np.isfinite(rows).all()
    direct = Basis(
        elements=["Ni"], source="tagged_cauchy_image", cutoff=4.5,
        rank=2, tag_counts=(0, 2), nmax_per_rank={2: 1}, lmax_per_rank={2: 1},
        source_block_partitions_by_rank={2: ((2,),)},
        max_records_per_rank=sum(item["candidate_fixed_contents_before_parity"]
                                 for item in basis.catalogue.repeated_content_summary()["by_component"]["main"]),
        max_features_per_rank=preview["raw_opportunity_count"],
        backend="reference", compiler_validation="full",
    )
    np.testing.assert_allclose(rows, direct.create(atoms), rtol=0, atol=1e-12)
    assert [label.identity for label in basis.labels] == [label.identity for label in direct.labels]
    assert basis._descriptor.metadata["tagged_cauchy_image_compiled"].self_hash == (
        direct._descriptor.metadata["tagged_cauchy_image_compiled"].self_hash)
    rotated = atoms.copy()
    rotated.rotate(57.0, "z", center=(0, 0, 0))
    np.testing.assert_allclose(basis.create(rotated), rows, rtol=0, atol=1e-10)
    batch = basis.create_many((atoms, rotated))
    assert len(batch) == 2
    np.testing.assert_allclose(batch[0], rows, rtol=0, atol=1e-12)
    np.testing.assert_allclose(batch[1], rows, rtol=0, atol=1e-10)
    training = atoms.copy()
    training.info["energy"] = -1.23
    training.arrays["forces"] = np.zeros((len(training), 3))
    training2 = atoms.copy()
    training2.positions[1, 0] += 0.18
    training2.info["energy"] = -0.72
    training2.arrays["forces"] = np.zeros((len(training2), 3))
    model = LinearModel(basis).fit([training, training2], force_weight=0.0)
    assert np.linalg.norm(model._fitted.beta_by_species["Ni"].detach().numpy()) > 1e-6
    evaluated = atoms.copy()
    evaluated.calc = model.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(evaluated.get_potential_energy(), -1.23, rtol=0, atol=1e-6)
    assert np.isfinite(evaluated.get_forces()).all()
    artifact = model.write(tmp_path / "configured.ye3t.json")
    restored = LinearModel.read(artifact)
    replay = atoms.copy()
    replay.calc = restored.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(replay.get_potential_energy(), evaluated.get_potential_energy(),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(replay.get_forces(), evaluated.get_forces(),
                               rtol=0, atol=1e-12)
    periodic = atoms.copy()
    periodic.set_cell((8.0, 8.0, 8.0))
    periodic.pbc = True
    periodic.calc = model.ase_calculator(evaluator="torch")
    forces = periodic.get_forces()
    stress = periodic.get_stress()
    assert all(abs(forces[atom_index, axis]) > 1e-6
               for atom_index, axis in ((0, 0), (1, 2)))
    assert np.all(np.abs(stress) > 1e-8)
    step = 1e-5
    for atom_index, axis in ((0, 0), (1, 2)):
        energies = []
        for direction in (-1, 1):
            shifted = periodic.copy()
            shifted.positions[atom_index, axis] += direction * step
            shifted.calc = model.ase_calculator(evaluator="torch")
            energies.append(shifted.get_potential_energy())
        np.testing.assert_allclose(forces[atom_index, axis],
                                   -(energies[1] - energies[0]) / (2 * step),
                                   rtol=0, atol=2e-5)
    voigt_axes = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    for index, (first, second) in enumerate(voigt_axes):
        energies = []
        for direction in (-1, 1):
            strain = np.zeros((3, 3))
            strain[first, second] = direction * step
            if first != second:
                strain[second, first] = direction * step
            deformed = periodic.copy()
            deformed.set_cell(np.asarray(periodic.cell) @ (np.eye(3) + strain),
                              scale_atoms=True)
            deformed.calc = model.ase_calculator(evaluator="torch")
            energies.append(deformed.get_potential_energy())
        multiplicity = 2 if first != second else 1
        derivative = (energies[1] - energies[0]) / (
            2 * step * periodic.get_volume() * multiplicity)
        np.testing.assert_allclose(stress[index], derivative, rtol=0, atol=2e-5)

    periodic_images = atoms.copy()
    periodic_images.set_cell((5.0, 5.0, 5.0))
    periodic_images.pbc = True
    assert np.any(neighbor_list("S", periodic_images, 4.5) != 0)
    periodic_images.calc = model.ase_calculator(evaluator="torch")
    image_force = periodic_images.get_forces()[0, 0]
    image_stress = periodic_images.get_stress()[0]
    assert abs(image_force) > 1e-6 and abs(image_stress) > 1e-8
    displaced_energies = []
    strained_energies = []
    for direction in (-1, 1):
        displaced = periodic_images.copy()
        displaced.positions[0, 0] += direction * step
        displaced.calc = model.ase_calculator(evaluator="torch")
        displaced_energies.append(displaced.get_potential_energy())
        strain = np.eye(3)
        strain[0, 0] += direction * step
        strained = periodic_images.copy()
        strained.set_cell(np.asarray(periodic_images.cell) @ strain, scale_atoms=True)
        strained.calc = model.ase_calculator(evaluator="torch")
        strained_energies.append(strained.get_potential_energy())
    np.testing.assert_allclose(
        image_force, -(displaced_energies[1] - displaced_energies[0]) / (2 * step),
        rtol=0, atol=2e-5,
    )
    np.testing.assert_allclose(
        image_stress,
        (strained_energies[1] - strained_energies[0]) /
        (2 * step * periodic_images.get_volume()),
        rtol=0, atol=2e-5,
    )


def test_multi_rank_tagged_config_preserves_rank_rows_and_native_derivatives(monkeypatch, tmp_path):
    from ye3t_methods.atomistic.tagged_cauchy_image import TaggedCauchyImageLinearModel
    from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {1: [0], 2: [0, 1]}}
    config["catalogue"] = {
        "ranks": [1, 2], "nmax_per_rank": {1: 1, 2: 1},
        "lmax_per_rank": {1: 0, 2: 1},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2]]},
    }
    runtime.update(evaluator="native_cpu", neighbors="ase", cache={"mode": "auto"})
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"]
    preview = basis.catalogue.counts()["by_component"]["main"]
    assert set(preview["raw_opportunities_by_rank"]) == {1, 2}
    atoms = Atoms("Ni4", positions=((0, 0, 0), (1.5, 0.2, 0.1),
                                    (-0.4, 1.6, 0.3), (0.5, -0.3, 1.7)),
                  cell=((7.0, 0.0, 0.0), (0.6, 7.5, 0.0), (0.2, 0.3, 8.0)),
                  pbc=True)
    rows = basis.create(atoms)
    rank_columns = {rank: [index for index, label in enumerate(basis.labels)
                           if label.as_dict()["N"] == rank] for rank in (1, 2)}
    assert all(rank_columns.values())
    assert np.linalg.norm(rows[:, rank_columns[1]]) > 1e-6
    assert np.linalg.norm(rows[:, rank_columns[2]]) > 1e-6
    compiled = basis._descriptor.metadata["tagged_cauchy_image_compiled"]
    components = compiled.plan.report.resource_report["components"]
    angular_columns = [index for index, record in enumerate(
        compiled.payload["image_coordinate_provenance"])
        if tuple(components[record["component_index"]]["angular_pattern"]) == (1, 1)]
    assert angular_columns and np.linalg.norm(rows[:, angular_columns]) > 1e-6
    summary = basis.catalogue.repeated_content_summary()["by_component"]["main"]
    for rank in (1, 2):
        direct = Basis(
            elements=["Ni"], source="tagged_cauchy_image", cutoff=4.5,
            rank=rank, tag_counts=config["tensor_product"]["tag_counts_per_rank"][rank],
            nmax_per_rank={rank: 1}, lmax_per_rank={rank: 0 if rank == 1 else 1},
            source_block_partitions_by_rank={rank: config["catalogue"][
                "source_block_partitions_by_rank"][rank]},
            max_records_per_rank=sum(item["candidate_fixed_contents_before_parity"]
                                     for item in summary if item["rank"] == rank),
            max_features_per_rank=preview["raw_opportunities_by_rank"][rank],
            backend="reference", compiler_validation="full",
        )
        np.testing.assert_allclose(rows[:, rank_columns[rank]], direct.create(atoms),
                                   rtol=0, atol=1e-11)
    rotated = atoms.copy()
    rotated.rotate(37.0, (1, 2, 3), rotate_cell=True, center=(0, 0, 0))
    np.testing.assert_allclose(basis.create(rotated), rows, rtol=0, atol=1e-9)
    inverted = atoms.copy()
    inverted.positions *= -1
    inverted.set_cell(-np.asarray(atoms.cell), scale_atoms=False)
    np.testing.assert_allclose(basis.create(inverted), rows, rtol=0, atol=1e-10)
    reordered = atoms[[2, 0, 3, 1]]
    np.testing.assert_allclose(basis.create(reordered), rows[[2, 0, 3, 1]],
                               rtol=0, atol=1e-10)

    evaluator = basis._descriptor.metadata["tagged_cauchy_image_evaluator"]
    beta = np.asarray([0.31 + 0.17 * index for index in range(len(basis.labels))])
    model = LinearModel(basis)
    model._fitted = TaggedCauchyImageLinearModel(evaluator, {"Ni": beta}, {"Ni": 0.13})
    reference = atoms.copy()
    reference.calc = model.ase_calculator(evaluator="torch")
    native = atoms.copy()
    native.calc = model.ase_calculator(evaluator="native_cpu", native_library=library)
    try:
        np.testing.assert_allclose(native.get_potential_energy(),
                                   reference.get_potential_energy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(native.get_stress(), reference.get_stress(),
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(native.calc.native_runtime.evaluate_atoms(
            atoms, return_features=True)[4], rows, rtol=0, atol=1e-8)
        energy = native.get_potential_energy()
        forces = native.get_forces()
        stress = native.get_stress()
        rotation = np.linalg.solve(np.asarray(atoms.cell), np.asarray(rotated.cell))
        rotated.calc = native.calc
        np.testing.assert_allclose(rotated.get_potential_energy(), energy,
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(rotated.get_forces(), forces @ rotation,
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(rotated.get_stress(), full_3x3_to_voigt_6_stress(
            rotation.T @ voigt_6_to_full_3x3_stress(stress) @ rotation),
            rtol=0, atol=1e-8)
        np.testing.assert_allclose(native.calc.native_runtime.evaluate_atoms(
            rotated, return_features=True)[4], rows, rtol=0, atol=1e-8)
        inverted.calc = native.calc
        np.testing.assert_allclose(inverted.get_potential_energy(), energy,
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(inverted.get_forces(), -forces,
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(inverted.get_stress(), stress,
                                   rtol=0, atol=1e-8)
        reordered.calc = native.calc
        np.testing.assert_allclose(reordered.get_potential_energy(), energy,
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(reordered.get_forces(), forces[[2, 0, 3, 1]],
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(reordered.get_stress(), stress,
                                   rtol=0, atol=1e-8)
        force = native.get_forces()[0, 0]
        assert abs(force) > 1e-5
        step = 1e-5
        energies = []
        for direction in (-1, 1):
            moved = atoms.copy()
            moved.positions[0, 0] += direction * step
            moved.calc = native.calc
            energies.append(moved.get_potential_energy())
        np.testing.assert_allclose(force, -(energies[1] - energies[0]) / (2 * step),
                                   rtol=0, atol=2e-5)
        stress = native.get_stress()
        for index, (first, second) in enumerate(((0, 0), (1, 1), (2, 2),
                                                  (1, 2), (0, 2), (0, 1))):
            energies = []
            for direction in (-1, 1):
                strain = np.zeros((3, 3))
                strain[first, second] = direction * step
                if first != second:
                    strain[second, first] = direction * step
                deformed = atoms.copy()
                deformed.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + strain),
                                  scale_atoms=True)
                deformed.calc = native.calc
                energies.append(deformed.get_potential_energy())
            multiplicity = 2 if first != second else 1
            np.testing.assert_allclose(stress[index],
                (energies[1] - energies[0]) / (2 * step * atoms.get_volume() * multiplicity),
                rtol=0, atol=2e-5)
    finally:
        native.calc.native_runtime.close()

    artifact = model.write(tmp_path / "multi_rank_tagged.ye3t.json")
    loaded = LinearModel.read(artifact)
    assert [label.as_dict() for label in loaded.labels] == [
        label.as_dict() for label in basis.labels]
    assert loaded.basis.resolved["ranks"] == (1, 2)
    assert "ranks=(1, 2)" in str(loaded.basis)
    replay = atoms.copy()
    replay.calc = loaded.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(replay.get_potential_energy(),
                               reference.get_potential_energy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(replay.get_forces(), reference.get_forces(),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(replay.get_stress(), reference.get_stress(),
                               rtol=0, atol=1e-12)

    replay_native = atoms.copy()
    replay_native.calc = loaded.ase_calculator(
        evaluator="native_cpu", native_library=library)
    try:
        np.testing.assert_allclose(replay_native.get_potential_energy(),
                                   reference.get_potential_energy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(replay_native.get_forces(), reference.get_forces(),
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(replay_native.get_stress(), reference.get_stress(),
                                   rtol=0, atol=1e-8)
    finally:
        replay_native.calc.native_runtime.close()

    training = []
    for displacement in (0.0, 0.13, -0.19):
        sample = atoms.copy()
        sample.positions[1, 0] += displacement
        sample.calc = model.ase_calculator(evaluator="torch")
        energy = sample.get_potential_energy()
        forces = sample.get_forces()
        sample.calc = None
        sample.info["energy"] = energy
        sample.arrays["forces"] = forces
        training.append(sample)
    fitted = LinearModel(basis).fit(training, regularization=1e-12)
    for sample in training:
        checked = sample.copy()
        checked.calc = fitted.ase_calculator(evaluator="torch")
        np.testing.assert_allclose(checked.get_potential_energy(),
                                   sample.info["energy"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(checked.get_forces(), sample.arrays["forces"],
                                   rtol=0, atol=1e-6)


def test_two_species_multi_rank_tagged_native_and_saved_model(monkeypatch, tmp_path):
    from ye3t_methods.atomistic.tagged_cauchy_image import TaggedCauchyImageLinearModel

    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {1: [0], 2: [0, 1]}}
    config["catalogue"] = {
        "ranks": [1, 2], "nmax_per_rank": {1: 1, 2: 1},
        "lmax_per_rank": {1: 0, 2: 0},
        "source_block_partitions_by_rank": {1: [[1]], 2: [[2]]},
    }
    runtime.update(evaluator="native_cpu", neighbors="ase", cache={"mode": "auto"})
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.to_dict()["species"] == ["Cu", "Ni"]
    atoms = Atoms("NiCuNiCu", positions=((0, 0, 0), (1.5, 0.2, 0.1),
                                        (-0.4, 1.6, 0.3), (0.5, -0.3, 1.7)),
                  cell=(8.0, 8.0, 8.0), pbc=True)
    rows = basis.create(atoms)
    assert {label.as_dict()["N"] for label in basis.labels} == {1, 2}
    assert np.linalg.norm(rows) > 1e-6
    evaluator = basis._descriptor.metadata["tagged_cauchy_image_evaluator"]
    width = len(basis.labels)
    beta = {"Cu": np.linspace(-0.35, -0.12, width),
            "Ni": np.linspace(0.24, 0.51, width)}
    model = LinearModel(basis)
    model._fitted = TaggedCauchyImageLinearModel(
        evaluator, beta, {"Cu": -0.07, "Ni": 0.13})
    reference = atoms.copy()
    reference.calc = model.ase_calculator(evaluator="torch")
    native = atoms.copy()
    native.calc = model.ase_calculator(evaluator="native_cpu", native_library=library)
    try:
        np.testing.assert_allclose(native.calc.native_runtime.evaluate_atoms(
            atoms, return_features=True)[4], rows, rtol=0, atol=1e-8)
        np.testing.assert_allclose(native.get_potential_energy(),
                                   reference.get_potential_energy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(native.get_stress(), reference.get_stress(),
                                   rtol=0, atol=1e-8)
    finally:
        native.calc.native_runtime.close()
    artifact = model.write(tmp_path / "two_species_multi_rank.ye3t.json")
    loaded = LinearModel.read(artifact)
    assert loaded.basis.resolved["ranks"] == (1, 2)
    assert [label.as_dict() for label in loaded.labels] == [
        label.as_dict() for label in basis.labels]
    replay = atoms.copy()
    replay.calc = loaded.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(replay.get_potential_energy(),
                               reference.get_potential_energy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(replay.get_forces(), reference.get_forces(),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(replay.get_stress(), reference.get_stress(),
                               rtol=0, atol=1e-12)


def test_rank_three_tagged_config_matches_direct_rows_and_inversion(monkeypatch, tmp_path):
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {3: 1}, "l_max_per_rank": {3: 1},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    config = {
        "single_factors": {
            "species": ["Ni"],
            "radial": {"family": "shifted_jacobi", "cutoff_A": 4.5},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {3: [0, 2]}},
        "catalogue": {
            "ranks": [3], "nmax_per_rank": {3: 1}, "lmax_per_rank": {3: 1},
            "source_block_partitions_by_rank": {3: [[2, 1]]},
        },
    }
    runtime = {"evaluator": "torch", "neighbors": "auto",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    basis = Basis.from_config(config, representation=representation, runtime=runtime)
    preview = basis.catalogue.counts()["by_component"]["main"]
    atoms = Atoms("Ni4", positions=((0, 0, 0), (1.5, 0.2, 0.4),
                                   (-0.3, 1.6, 0.7), (0.8, -0.4, 1.8)))
    rows = np.asarray(basis.create(atoms))
    direct = Basis(
        elements=["Ni"], source="tagged_cauchy_image", cutoff=4.5,
        rank=3, tag_counts=(0, 2), nmax_per_rank={3: 1}, lmax_per_rank={3: 1},
        source_block_partitions_by_rank={3: ((2, 1),)},
        max_records_per_rank=sum(item["candidate_fixed_contents_before_parity"]
                                 for item in basis.catalogue.repeated_content_summary()["by_component"]["main"]),
        max_features_per_rank=preview["raw_opportunity_count"],
        backend="reference", compiler_validation="full",
    )
    assert rows.shape == (len(atoms), len(basis.labels))
    assert np.isrealobj(rows) and np.isfinite(rows).all()
    np.testing.assert_allclose(rows, direct.create(atoms), rtol=0, atol=1e-12)
    assert [label.identity for label in basis.labels] == [label.identity for label in direct.labels]
    rotated = atoms.copy()
    rotated.rotate(57.0, (1, 2, 3), center=(0, 0, 0))
    np.testing.assert_allclose(basis.create(rotated), rows, rtol=0, atol=1e-10)
    inverted = atoms.copy()
    inverted.positions *= -1
    np.testing.assert_allclose(basis.create(inverted), rows, rtol=0, atol=1e-10)


@pytest.mark.parametrize("rank,target_L,parity", [
    (1, 1, "odd"), (2, 2, "even"), (1, 3, "odd")])
def test_configured_tagged_full_m_basis_uses_exact_physical_image(
        rank, target_L, parity, monkeypatch, tmp_path):
    from ye3t.core.rotation import wigner_D_numeric
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    rep = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [rank],
        "parent": {"young_lambda": "(N)", "L": target_L, "parity": parity},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {rank: 2},
            "l_max_per_rank": {rank: max(1, target_L)}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    config = {
        "single_factors": {"species": ["Ni"],
                           "radial": {"family": "shifted_jacobi", "cutoff_A": 2.4},
                           "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {
            rank: [0, 1] if rank == 1 else [0, 1, 2]}},
        "catalogue": {"ranks": [rank], "nmax_per_rank": {rank: 2},
                      "lmax_per_rank": {rank: max(1, target_L)},
                      "source_block_partitions_by_rank": {rank: [[rank]]}},
    }
    runtime = {"evaluator": "reference", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"]
    assert basis.resolution.capability_report["selected_evaluator"] == "reference"
    preview = basis.catalogue.counts()["by_component"]["main"]
    assert preview["raw_opportunity_count"] > 0
    assert preview["exact_image_count"] is None
    atoms = Atoms("Ni4", positions=[
        [0, 0, 0], [1.0, .2, .1], [.1, 1.2, .2], [.2, .1, 1.3],
    ], cell=[8, 8, 8], pbc=False)
    rows = basis.create(atoms)
    assert rows.shape == (len(atoms), len(basis.labels), 2 * target_L + 1)
    with monkeypatch.context() as warm_patch:
        warm_patch.setattr("ye3t.couplings.tagged_cauchy_carrier_physical_image_plan",
                           lambda *args, **kwargs: (_ for _ in ()).throw(
                               AssertionError("warm full-M evaluation recomputed the exact image")))
        assert basis.create_many([atoms, atoms])[0].shape == rows.shape
    assert len(basis.labels) > 0 and np.linalg.norm(rows) > 1e-10
    assert all(label.as_dict()["L"] == target_L for label in basis.labels)
    assert all(label.as_dict()["target_parity"] == (1 if parity == "even" else -1)
               for label in basis.labels)
    assert basis.resolved["physical_image_plan_hash"]
    image = basis._descriptor.create(atoms, descriptor_evaluation="physical_image")
    for index, (tag_count, span) in enumerate(basis._tagged_carrier_selection):
        np.testing.assert_allclose(rows[:, index],
                                   image["physical_image"][tag_count]["values"][:, slice(*span)],
                                   rtol=0, atol=1e-12)
    angle = .33
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    turned = atoms.copy()
    turned.positions = atoms.positions @ rotation.T
    inverted = atoms.copy()
    inverted.positions = -atoms.positions
    ordered = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(basis.create(inverted),
                               (1 if parity == "even" else -1) * rows,
                               rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(basis.create(atoms[ordered]), rows[ordered],
                               rtol=1e-10, atol=1e-11)
    complex_rows = real_tesseral_to_complex_multiplet(torch.as_tensor(rows), target_L).numpy()
    complex_turned = real_tesseral_to_complex_multiplet(
        torch.as_tensor(basis.create(turned)), target_L).numpy()
    np.testing.assert_allclose(complex_turned,
                               complex_rows @ wigner_D_numeric(target_L, rotation).T,
                               rtol=1e-8, atol=1e-10)
    if target_L == 3:
        atoms.new_array("octupole", np.einsum(
            "nfm,f->nm", rows, np.linspace(.2, .7, len(basis.labels))))
        fit_config = {
            "metadata": {"schema": "ye3t_config_v1", "name": "tagged_l3",
                         "status": "experimental"},
            "representation": rep.to_dict(), "basis": config, "runtime": runtime,
            "model": {"kind": "linear", "output": {"scope": "per_atom"},
                      "fit": {"solver": "ridge", "alpha": 0.0}},
            "targets": {"per_atom": {"key": "octupole",
                                      "input": "real_tesseral", "units": "arbitrary"}},
            "validation": {"checks": ["round_trip"]},
        }
        model = LinearModel(basis).fit([atoms], config=fit_config)
        restored = LinearModel.read(model.write(tmp_path / "tagged_l3.ye3t.json"))
        np.testing.assert_allclose(restored.predict(atoms)["mean_real_tesseral"],
                                   model.predict(atoms)["mean_real_tesseral"],
                                   atol=2e-10, rtol=2e-10)
    if rank == 2:
        from ye3t_methods.linear import _tagged_full_m_property_plan

        convention = {"group": "O3", "basis": "real_tesseral_tensor_components",
                      "axis_order": "cos_L_to_cos_1_zero_sin_1_to_sin_L",
                      "M_values": list(range(-target_L, target_L + 1)),
                      "L": target_L, "parity": parity}
        fitted = {"beta": np.linspace(.2, .7, len(basis.labels)),
                  "fit_metadata": {"n_cols": len(basis.labels)}}
        native = _tagged_full_m_property_plan(basis, fitted, convention)
        assert native["selected_coordinate_ids"] == [label.identity
                                                     for label in basis.labels]
        assert all(schedule["support_tag_count"] == 1
                   for schedule in native["schedules"] if schedule["tag_count"] == 2)
        periodic = Atoms("Ni2", positions=[[0, 0, 0], [1.0, .2, .1]],
                         cell=[2.1, 6, 6], pbc=[True, False, False])
        reference = basis.create(periodic)
        descriptor = basis._descriptor
        old_schedules = descriptor.metadata["_tagged_carrier_physical_image_schedules"]
        old_evaluators = descriptor.metadata.pop("_tagged_carrier_evaluators", None)
        try:
            descriptor.metadata["_tagged_carrier_physical_image_schedules"] = tuple(
                native["schedules"])
            lowered = descriptor.create(periodic, descriptor_evaluation="physical_image")
        finally:
            descriptor.metadata["_tagged_carrier_physical_image_schedules"] = old_schedules
            descriptor.metadata.pop("_tagged_carrier_evaluators", None)
            if old_evaluators is not None:
                descriptor.metadata["_tagged_carrier_evaluators"] = old_evaluators
        for feature, record in enumerate(native["feature_slices"]):
            block = lowered["physical_image"][record["tag_count"]]["values"]
            np.testing.assert_allclose(
                block[:, slice(*record["component_slice"])], reference[:, feature],
                atol=2e-10, rtol=2e-10)
        atoms.new_array("quadrupole", np.einsum(
            "nfm,f->nm", rows, fitted["beta"]))
        fit_config = {
            "metadata": {"schema": "ye3t_config_v1", "name": "tagged_l2_native_plan",
                         "status": "experimental"},
            "representation": rep.to_dict(), "basis": config, "runtime": runtime,
            "model": {"kind": "linear", "output": {"scope": "per_atom"},
                      "fit": {"solver": "ridge", "alpha": 1e-12}},
            "targets": {"per_atom": {"key": "quadrupole",
                                      "input": "real_tesseral", "units": "arbitrary"}},
            "validation": {"checks": []},
        }
        model = LinearModel(basis).fit([atoms], config=fit_config)
        saved = model.write(tmp_path / "tagged_l2.ye3t.json")
        restored = LinearModel.read(saved)
        np.testing.assert_allclose(restored.predict(atoms)["mean_real_tesseral"],
                                   model.predict(atoms)["mean_real_tesseral"],
                                   atol=2e-10, rtol=2e-10)
        np.testing.assert_array_equal(json.loads(saved.read_text())[
            "native_property_plan"]["readout"]["coefficients_by_species"],
            [model._fitted["beta"]])
    bad_runtime = {**runtime, "evaluator": "native_cpu"}
    unavailable = Basis.from_config(config, representation=rep, runtime=bad_runtime)
    assert not unavailable.resolution.capability_report["basis_create_available"]
    with pytest.raises(RuntimeError, match="reference CPU"):
        unavailable.create(atoms)


def test_configured_tagged_full_m_fit_shares_coefficients_and_ard_covariance(
        tmp_path, monkeypatch):
    import hashlib
    from ye3t_methods.tesseral_targets import real_tesseral_to_cartesian
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [1],
        "parent": {"young_lambda": "(N)", "L": 1, "parity": "odd"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {"eta_count_per_rank": {1: 2},
                                    "l_max_per_rank": {1: 1}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    basis_config = {
        "single_factors": {"species": ["Ni"],
                           "radial": {"family": "shifted_jacobi", "cutoff_A": 2.4},
                           "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {1: [0, 1]}},
        "catalogue": {"ranks": [1], "nmax_per_rank": {1: 2},
                      "lmax_per_rank": {1: 1},
                      "source_block_partitions_by_rank": {1: [[1]]}},
    }
    runtime = {"evaluator": "reference", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(basis_config, representation=representation, runtime=runtime)
    frames = []
    for displacement in (0.0, .12, -.09, .21):
        atoms = Atoms("Ni4", positions=[
            [0, 0, 0], [1.0 + displacement, .2, .1],
            [.1, 1.2 - displacement, .2], [.2, .1, 1.3 + displacement],
        ], cell=[8, 8, 8], pbc=False)
        frames.append(atoms)
    rows = [basis.create(atoms) for atoms in frames]
    truth = np.linspace(.4, 1.2, len(basis.labels))
    for atoms, features in zip(frames, rows, strict=True):
        atoms.new_array("dipoles", np.einsum("nfm,f->nm", features, truth))
    fit_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "full_m_test",
                     "status": "experimental"},
        "representation": representation.to_dict(), "basis": basis_config,
        "runtime": runtime,
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "dipoles", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit(frames, config=fit_config)
    assert model._fitted["fit_metadata"]["configured_validation"]["results"][
        "round_trip"]["passed"]
    saved = model.write(tmp_path / "full_m.ye3t.json")
    monkeypatch.setattr("ye3t.couplings.compile", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("reader recompiled carrier")))
    restored = LinearModel.read(saved)
    np.testing.assert_allclose(restored.predict(frames[0])["mean_real_tesseral"],
                               model.predict(frames[0])["mean_real_tesseral"],
                               rtol=0, atol=1e-12)
    assert restored.basis.resolved["compiler_hash"] == basis.resolved["compiler_hash"]
    assert restored.basis.resolved["physical_image_plan_hash"] == (
        basis.resolved["physical_image_plan_hash"])
    assert [label.identity for label in restored.labels] == [
        label.identity for label in model.labels]
    calc_atoms = frames[0].copy()
    calc_atoms.calc = restored.ase_calculator()
    np.testing.assert_allclose(calc_atoms.calc.get_property(
        "per_atom_real_tesseral_mean", calc_atoms), frames[0].arrays["dipoles"],
                               rtol=1e-9, atol=1e-9)
    document = json.loads(saved.read_text())
    assert document["schema"] == "ye3t_methods_tagged_full_m_per_atom_v2"
    native = document["native_property_plan"]
    assert native["selected_coordinate_ids"] == document["selected_coordinate_ids"]
    assert native["readout"]["coefficients_by_species"] == [
        model._fitted["beta"].tolist()]
    changed = json.loads(saved.read_text())
    changed["native_property_plan"]["readout"]["coefficients_by_species"][0][0] += .25
    native_body = {key: value for key, value in changed["native_property_plan"].items()
                   if key != "self_hash"}
    changed["native_property_plan"]["self_hash"] = hashlib.sha256(json.dumps(
        native_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    changed["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in changed.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    corrupted = tmp_path / "corrupted.ye3t.json"
    corrupted.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="native property plan"):
        LinearModel.read(corrupted)
    legacy = json.loads(saved.read_text())
    legacy["schema"] = "ye3t_methods_tagged_full_m_per_atom_v1"
    legacy.pop("native_property_plan")
    legacy["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in legacy.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    old_path = tmp_path / "old_full_m.ye3t.json"
    old_path.write_text(json.dumps(legacy))
    np.testing.assert_allclose(LinearModel.read(old_path).predict(frames[0])[
        "mean_real_tesseral"], model.predict(frames[0])["mean_real_tesseral"],
        atol=1e-12, rtol=0)
    changed = dict(document)
    changed["selected_coordinate_ids"] = list(reversed(document["selected_coordinate_ids"]))
    corrupted.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="self-hash"):
        LinearModel.read(corrupted)
    changed["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in changed.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    corrupted.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="selected coordinates"):
        LinearModel.read(corrupted)
    changed = json.loads(saved.read_text())
    changed["fit"]["fit_metadata"]["coordinate_penalty_metric"]["diagonal"][0] = 2.0
    changed["self_hash"] = hashlib.sha256(json.dumps(
        {key: value for key, value in changed.items() if key != "self_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    corrupted.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fitted columns"):
        LinearModel.read(corrupted)
    for atoms in frames:
        atoms.new_array("vectors", real_tesseral_to_cartesian(
            atoms.arrays["dipoles"], 1, "odd"))
    cartesian_config = json.loads(json.dumps(fit_config))
    cartesian_config["targets"]["per_atom"] = {
        "key": "vectors", "input": "cartesian", "units": "arbitrary"}
    cartesian = LinearModel(basis).fit(frames, config=cartesian_config)
    np.testing.assert_allclose(cartesian._fitted["beta"], truth, rtol=1e-9, atol=1e-9)
    assert cartesian._fitted["fit_metadata"]["cartesian_tesseral_convention_sha256"]
    np.testing.assert_allclose(model._fitted["beta"], truth, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(model.predict(frames[0])["mean_real_tesseral"],
                               frames[0].arrays["dipoles"], rtol=1e-9, atol=1e-9)
    metric = model._fitted["fit_metadata"]["coordinate_penalty_metric"]
    assert metric["columns"] == [
        {"central_species": "Ni", "coordinate_id": label.identity}
        for label in basis.labels]
    assert metric["physical_image_plan_hash"] == basis.resolved["physical_image_plan_hash"]
    assert metric["diagonal"] == [1.0] * len(basis.labels)
    with pytest.raises(ValueError, match="ARD posterior"):
        model.predict(frames[0], uncertainty=True)
    fit_config["model"]["fit"] = {"solver": "lasso", "alpha": 1e-8}
    sparse = LinearModel(basis).fit(frames, config=fit_config)
    assert np.isfinite(sparse.predict(frames[0])["mean_real_tesseral"]).all()
    fit_config["model"]["fit"] = {"solver": "ard", "solver_options": {"max_iter": 50}}
    bayes = LinearModel(basis).fit(frames, config=fit_config)
    prediction = bayes.predict(frames[0], uncertainty=True)
    covariance = prediction["covariance_real_tesseral"]
    assert covariance.shape == (len(frames[0]), 3, 3)
    assert np.isfinite(covariance).all()
    posterior = bayes._fitted["fit_metadata"]["predictive_uncertainty"]
    active = posterior["active_column_indices"]
    sigma = np.asarray(posterior["coefficient_covariance_active"])
    for atom in range(len(frames[0])):
        design = rows[0][atom, active, :].T
        np.testing.assert_allclose(covariance[atom], design @ sigma @ design.T,
                                   rtol=1e-10, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(covariance[atom])) >= -1e-10
    angle = .47
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    axes = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    tesseral_rotation = axes @ rotation @ axes.T
    rotated = frames[0].copy()
    rotated.positions = frames[0].positions @ rotation.T
    rotated_prediction = bayes.predict(rotated, uncertainty=True)
    np.testing.assert_allclose(rotated_prediction["mean_real_tesseral"],
                               prediction["mean_real_tesseral"] @ tesseral_rotation.T,
                               rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(rotated_prediction["covariance_real_tesseral"],
                               tesseral_rotation @ covariance @ tesseral_rotation.T,
                               rtol=1e-9, atol=1e-10)
    inverted = frames[0].copy()
    inverted.positions *= -1
    inversion_prediction = bayes.predict(inverted, uncertainty=True)
    np.testing.assert_allclose(inversion_prediction["mean_real_tesseral"],
                               -prediction["mean_real_tesseral"],
                               rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(inversion_prediction["covariance_real_tesseral"],
                               covariance, rtol=1e-9, atol=1e-10)
    order = [2, 0, 3, 1]
    reordered = bayes.predict(frames[0][order], uncertainty=True)
    np.testing.assert_allclose(reordered["mean_real_tesseral"],
                               prediction["mean_real_tesseral"][order],
                               rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(reordered["covariance_real_tesseral"],
                               covariance[order], rtol=1e-9, atol=1e-10)
    loaded_bayes = LinearModel.read(bayes.write(tmp_path / "ard.ye3t.json"))
    np.testing.assert_allclose(loaded_bayes.predict(rotated, uncertainty=True)[
        "covariance_real_tesseral"], rotated_prediction["covariance_real_tesseral"],
        rtol=0, atol=1e-12)


def test_configured_rank2_full_m_fit_cartesian_stf_and_rotation(tmp_path):
    from ye3t.core.rotation import wigner_D_numeric
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet
    from ye3t_methods.tesseral_targets import real_tesseral_to_cartesian

    rank = 2
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [rank],
        "parent": {"young_lambda": "(N)", "L": 2, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {"eta_count_per_rank": {rank: 2},
                                    "l_max_per_rank": {rank: 1}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    basis_config = {
        "single_factors": {"species": ["Ni"],
                           "radial": {"family": "shifted_jacobi", "cutoff_A": 2.4},
                           "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {
            rank: [0, 1, 2]}},
        "catalogue": {"ranks": [rank], "nmax_per_rank": {rank: 2},
                      "lmax_per_rank": {rank: 1},
                      "source_block_partitions_by_rank": {rank: [[rank]]}},
    }
    runtime = {"evaluator": "reference", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(basis_config, representation=representation, runtime=runtime)
    frames = []
    for step in (0.0, .1, -.14):
        atoms = Atoms("Ni4", positions=[[0, 0, 0], [1 + step, .2, .1],
                                       [.1, 1.2 - step, .2], [.2, .1, 1.3 + step]],
                      cell=[8, 8, 8], pbc=False)
        frames.append(atoms)
    basis.create(frames[0])
    coefficients = np.linspace(.3, .8, len(basis.labels))
    assert len(coefficients) > 0
    for atoms in frames:
        target = np.einsum("nfm,f->nm", basis.create(atoms), coefficients)
        atoms.new_array("quadrupoles", real_tesseral_to_cartesian(target, 2, "even"))
    fit_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "full_m_stf",
                     "status": "experimental"},
        "representation": representation.to_dict(), "basis": basis_config,
        "runtime": runtime,
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "quadrupoles", "input": "cartesian",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit(frames, config=fit_config)
    for atoms in frames:
        np.testing.assert_allclose(real_tesseral_to_cartesian(
            model.predict(atoms)["mean_real_tesseral"], 2, "even"),
            atoms.arrays["quadrupoles"], rtol=1e-9, atol=1e-10)
    rotation = np.array([[np.cos(.37), -np.sin(.37), 0],
                         [np.sin(.37), np.cos(.37), 0], [0, 0, 1]]) @ np.array(
        [[np.cos(.23), 0, np.sin(.23)], [0, 1, 0],
         [-np.sin(.23), 0, np.cos(.23)]])
    rotated = frames[0].copy()
    rotated.positions = frames[0].positions @ rotation.T
    original = model.predict(frames[0])["mean_real_tesseral"]
    turned = model.predict(rotated)["mean_real_tesseral"]
    np.testing.assert_allclose(
        real_tesseral_to_complex_multiplet(torch.as_tensor(turned), 2).numpy(),
        real_tesseral_to_complex_multiplet(torch.as_tensor(original), 2).numpy()
        @ wigner_D_numeric(2, rotation).T, rtol=1e-9, atol=1e-10)
    saved = model.write(tmp_path / "rank2.ye3t.json")
    restored = LinearModel.read(saved)
    np.testing.assert_allclose(restored.predict(rotated)["mean_real_tesseral"],
                               turned, rtol=0, atol=1e-12)


def test_configured_full_m_two_central_species_keep_separate_columns(tmp_path):
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [1],
        "parent": {"young_lambda": "(N)", "L": 1, "parity": "odd"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {"eta_count_per_rank": {1: 2},
                                    "l_max_per_rank": {1: 1}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    basis_config = {
        "single_factors": {"species": ["Ni", "Cu"],
                           "radial": {"family": "shifted_jacobi", "cutoff_A": 2.4},
                           "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {1: [0, 1]}},
        "catalogue": {"ranks": [1], "nmax_per_rank": {1: 1},
                      "lmax_per_rank": {1: 1},
                      "source_block_partitions_by_rank": {1: [[1]]}},
    }
    runtime = {"evaluator": "reference", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(basis_config, representation=representation, runtime=runtime)
    frames = []
    for shift in (0, .1, -.13, .22):
        atoms = Atoms("NiCuNiCu", positions=[[0, 0, 0], [1 + shift, .1, .2],
                                            [.1, 1.1 - shift, .3], [.3, .2, 1.2]],
                      cell=[8, 8, 8], pbc=False)
        frames.append(atoms)
    basis.create(frames[0])
    width = len(basis.labels)
    assert width > 0
    beta = np.concatenate((np.full(width, .4), np.full(width, 1.3)))
    for atoms in frames:
        rows = basis.create(atoms)
        species = np.array(atoms.get_chemical_symbols())
        target = np.zeros((len(atoms), 3))
        target[species == "Ni"] = np.einsum("nfm,f->nm", rows[species == "Ni"], beta[:width])
        target[species == "Cu"] = np.einsum("nfm,f->nm", rows[species == "Cu"], beta[width:])
        atoms.new_array("vectors", target)
    fit_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "two_species_full_m",
                     "status": "experimental"},
        "representation": representation.to_dict(), "basis": basis_config,
        "runtime": runtime,
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "vectors", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit(frames, config=fit_config)
    for atoms in frames:
        np.testing.assert_allclose(model.predict(atoms)["mean_real_tesseral"],
                                   atoms.arrays["vectors"], rtol=1e-9, atol=1e-10)
    assert model._fitted["fit_metadata"]["coordinate_penalty_metric"]["columns"] == [
        {"central_species": name, "coordinate_id": label.identity}
        for name in basis.elements for label in basis.labels]
    saved = model.write(tmp_path / "two_species.ye3t.json")
    loaded = LinearModel.read(saved)
    np.testing.assert_allclose(loaded.predict(frames[0])["mean_real_tesseral"],
                               frames[0].arrays["vectors"], rtol=1e-9, atol=1e-10)


def test_configured_full_m_multirank_uses_rank_specific_tags_and_joint_image(tmp_path):
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [1, 2],
        "parent": {"young_lambda": "(N)", "L": 1, "parity": "odd"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {"eta_count_per_rank": {1: 2, 2: 2},
                                    "l_max_per_rank": {1: 1, 2: 1}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    basis_config = {
        "single_factors": {"species": ["Ni"],
                           "radial": {"family": "shifted_jacobi", "cutoff_A": 2.4},
                           "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {
            1: [0, 1], 2: [0, 2]}},
        "catalogue": {"ranks": [1, 2], "nmax_per_rank": {1: 2, 2: 2},
                      "lmax_per_rank": {1: 1, 2: 1},
                      "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]]}},
    }
    runtime = {"evaluator": "reference", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(basis_config, representation=representation, runtime=runtime)
    preview = basis.catalogue.counts()["by_component"]["main"]
    assert preview["exact_image_count"] is None
    assert all(preview["raw_opportunities_by_rank"][rank] > 0 for rank in (1, 2))
    atoms = Atoms("Ni4", positions=[[0, 0, 0], [1, .2, .1],
                                   [.1, 1.2, .2], [.2, .1, 1.3]],
                  cell=[8, 8, 8], pbc=False)
    rows = basis.create(atoms)
    assert rows.shape == (4, len(basis.labels), 3)
    assert {label.as_dict()["N"] for label in basis.labels} == {1, 2}
    compiled = basis._descriptor.metadata["tagged_cauchy_carriers_compiled"]
    assert compiled["request"]["catalogue"]["tag_counts_by_rank"] == {
        "1": (0, 1), "2": (0, 2)}
    assert basis._resolved["physical_image_plan_hash"]
    image = basis._descriptor.create(atoms, descriptor_evaluation="physical_image")
    for index, (tag_count, span) in enumerate(basis._tagged_carrier_selection):
        np.testing.assert_allclose(rows[:, index], image["physical_image"][
            tag_count]["values"][:, slice(*span)], rtol=0, atol=1e-12)
    rotated = atoms.copy()
    rotated.rotate(38, (1, 2, 3), center=(0, 0, 0))
    inverted = atoms.copy()
    inverted.positions *= -1
    np.testing.assert_allclose(basis.create(inverted), -rows, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(basis.create(atoms[[2, 0, 3, 1]]),
                               rows[[2, 0, 3, 1]], rtol=1e-9, atol=1e-10)
    truth = np.linspace(.2, .7, len(basis.labels))
    frames = [atoms, rotated]
    for frame in frames:
        frame.new_array("vector", np.einsum("nfm,f->nm", basis.create(frame), truth))
    fit_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "multirank_full_m",
                     "status": "experimental"},
        "representation": representation.to_dict(), "basis": basis_config,
        "runtime": runtime,
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "vector", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit(frames, config=fit_config)
    np.testing.assert_allclose(model.predict(rotated)["mean_real_tesseral"],
                               rotated.arrays["vector"], rtol=1e-9, atol=1e-10)
    loaded = LinearModel.read(model.write(tmp_path / "multirank.ye3t.json"))
    np.testing.assert_allclose(loaded.predict(rotated)["mean_real_tesseral"],
                               rotated.arrays["vector"], rtol=1e-9, atol=1e-10)


def test_full_m_tag_count_above_two_fails_before_compilation(monkeypatch):
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(N)", "L": 1, "parity": "odd"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {"eta_count_per_rank": {3: 1},
                                    "l_max_per_rank": {3: 1}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    basis_config = {
        "single_factors": {"species": ["Ni"],
                           "radial": {"family": "shifted_jacobi", "cutoff_A": 2.4},
                           "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {3: [3]}},
        "catalogue": {"ranks": [3], "nmax_per_rank": {3: 1},
                      "lmax_per_rank": {3: 1},
                      "source_block_partitions_by_rank": {3: [[3]]}},
    }
    runtime = {"evaluator": "reference", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(basis_config, representation=representation, runtime=runtime)
    assert not basis.resolution.capability_report["basis_create_available"]
    monkeypatch.setattr("ye3t.couplings.compile", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("capability failure compiled")))
    with pytest.raises(RuntimeError, match="0/1/2 tags"):
        basis.create(Atoms("Ni2", positions=[[0, 0, 0], [1, 0, 0]]))


def test_rank_three_configured_tagged_native_symmetry_and_derivatives(monkeypatch, tmp_path):
    from ase.stress import voigt_6_to_full_3x3_stress

    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {3: 1}, "l_max_per_rank": {3: 1},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    config = {
        "single_factors": {
            "species": ["Ni"],
            "radial": {"family": "shifted_jacobi", "cutoff_A": 4.5},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {3: [0, 2]}},
        "catalogue": {
            "ranks": [3], "nmax_per_rank": {3: 1}, "lmax_per_rank": {3: 1},
            "source_block_partitions_by_rank": {3: [[2, 1]]},
        },
    }
    runtime = {"evaluator": "native_cpu", "neighbors": "ase",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    basis = Basis.from_config(config, representation=representation, runtime=runtime)
    atoms = Atoms("Ni4", positions=((0, 0, 0), (1.5, 0.2, 0.4),
                                     (-0.3, 1.6, 0.7), (0.8, -0.4, 1.8)),
                  cell=(8.0, 8.0, 8.0), pbc=True)
    assert basis.create(atoms).shape == (len(atoms), len(basis.labels))
    training = atoms.copy()
    training.info["energy"] = -1.23
    training.arrays["forces"] = np.zeros((len(training), 3))
    shifted = atoms.copy()
    shifted.positions[1, 0] += 0.18
    shifted.info["energy"] = -0.72
    shifted.arrays["forces"] = np.zeros((len(shifted), 3))
    model = LinearModel(basis).fit([training, shifted], force_weight=0.0)
    assert np.linalg.norm(model._fitted.beta_by_species["Ni"].detach().numpy()) > 1e-6

    reference = atoms.copy()
    reference.calc = model.ase_calculator(evaluator="torch")
    native_calc = model.ase_calculator(evaluator="auto", native_library=library)
    native = atoms.copy()
    native.calc = native_calc
    try:
        energy = native.get_potential_energy()
        forces = native.get_forces()
        stress = native.get_stress()
        np.testing.assert_allclose(energy, reference.get_potential_energy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(forces, reference.get_forces(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(stress, reference.get_stress(), rtol=0, atol=1e-8)
        assert abs(forces[1, 0]) > 1e-4
        assert np.min(np.abs(stress)) > 1e-5

        axis = np.array((1.0, 2.0, 3.0))
        axis /= np.linalg.norm(axis)
        cross = np.array(((0.0, -axis[2], axis[1]),
                          (axis[2], 0.0, -axis[0]),
                          (-axis[1], axis[0], 0.0)))
        angle = np.deg2rad(37.0)
        rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
        rotated = atoms.copy()
        rotated.positions[:] = atoms.positions @ rotation.T
        rotated.set_cell(np.asarray(atoms.cell) @ rotation.T, scale_atoms=False)
        rotated.calc = native_calc
        np.testing.assert_allclose(rotated.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(rotated.get_forces(), forces @ rotation.T, rtol=0, atol=1e-8)
        np.testing.assert_allclose(voigt_6_to_full_3x3_stress(rotated.get_stress()),
                                   rotation @ voigt_6_to_full_3x3_stress(stress) @ rotation.T,
                                   rtol=0, atol=1e-8)
        inverted = atoms.copy()
        inverted.positions *= -1
        inverted.calc = native_calc
        np.testing.assert_allclose(inverted.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(inverted.get_forces(), -forces, rtol=0, atol=1e-8)
        np.testing.assert_allclose(inverted.get_stress(), stress, rtol=0, atol=1e-8)
        order = [2, 0, 3, 1]
        reordered = atoms[order]
        reordered.calc = native_calc
        np.testing.assert_allclose(reordered.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(reordered.get_forces(), forces[order], rtol=0, atol=1e-8)
        np.testing.assert_allclose(reordered.get_stress(), stress, rtol=0, atol=1e-8)

        step = 1e-5
        displaced_energies = []
        for direction in (-1, 1):
            displaced = atoms.copy()
            displaced.positions[1, 0] += direction * step
            displaced.calc = native_calc
            displaced_energies.append(displaced.get_potential_energy())
        np.testing.assert_allclose(forces[1, 0],
                                   -(displaced_energies[1] - displaced_energies[0]) / (2 * step),
                                   rtol=0, atol=5e-6)
        for index, (first, second) in enumerate(((0, 0), (1, 1), (2, 2),
                                                  (1, 2), (0, 2), (0, 1))):
            strained_energies = []
            for direction in (-1, 1):
                strain = np.zeros((3, 3))
                strain[first, second] = direction * step
                if first != second:
                    strain[second, first] = direction * step
                deformed = atoms.copy()
                deformed.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + strain),
                                  scale_atoms=True)
                deformed.calc = native_calc
                strained_energies.append(deformed.get_potential_energy())
            multiplier = 2 if first != second else 1
            derivative = (strained_energies[1] - strained_energies[0]) / (
                2 * step * atoms.get_volume() * multiplier)
            np.testing.assert_allclose(stress[index], derivative, rtol=0, atol=5e-6)
    finally:
        native_calc.native_runtime.close()


def test_two_species_tagged_eta_order_matches_compiler_channels():
    from ye3t.couplings import count, tagged_cauchy_image_request

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {2: [0]}}
    config["catalogue"] = {
        "ranks": [2], "nmax_per_rank": {2: 2}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: [[2]]},
    }
    runtime["evaluator"] = "torch"
    runtime["neighbors"] = "auto"
    runtime["cache"]["mode"] = "auto"
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    resolved = basis.resolution.to_dict()
    assert resolved["species"] == ["Cu", "Ni"]
    assert basis.elements == ("Cu", "Ni")
    assert "tagged species order normalized to compiler lexical order" in basis.resolution.warnings
    eta = resolved["components"][0]["physical_eta_by_center"]["Cu"]
    assert [(row["chemical"]["neighbor_species"], row["radial_index"])
            for row in eta] == [("Cu", 0), ("Ni", 0), ("Cu", 1), ("Ni", 1)]
    assert [row["compiler_content_id"] for row in eta] == [1, 2, 3, 4]
    assert [row["chemical"]["chemical_index"] for row in eta] == [0, 1, 0, 1]
    assert resolved["components"][0]["active_compiler_content_ids_by_rank"]["2"] == [1, 2, 3, 4]
    report = count(tagged_cauchy_image_request(species=["Ni", "Cu"], catalogue={
        "nmax_per_rank": {2: 2}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: [[2]]},
        "tag_counts_by_rank": {2: [0]},
        "max_records_per_rank": 8, "max_features_per_rank": 8,
    }))
    actual = {
        (channel["neighbor_species"], channel["radial_channel"], channel["l"])
        for row in report.labels for channel in row["label"]["block_complete_channel_keys"]
    }
    expected = {
        (row["chemical"]["neighbor_species"], row["radial_index"], angular_l)
        for row in eta for angular_l in (0, 1)
    }
    assert actual == expected
    assert report.resource_report["coefficient_materialization_performed"] is False


def test_tagged_config_rejects_zero_complete_channel_content_before_compile():
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {3: 2}, "l_max_per_rank": {3: 0},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    config = {
        "single_factors": {
            "species": ["Ni", "Cu"],
            "radial": {"family": "shifted_jacobi", "cutoff_A": 4.5},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {3: [0]}},
        "catalogue": {
            "ranks": [3], "nmax_per_rank": {3: 1}, "lmax_per_rank": {3: 0},
            "source_block_partitions_by_rank": {3: [[1, 1, 1]]},
        },
    }
    runtime = {"evaluator": "torch", "neighbors": "auto",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    with pytest.raises(ValueError, match="no complete-channel source records"):
        Basis.from_config(config, representation=representation, runtime=runtime)


def test_two_species_configured_tagged_native_matches_reference(monkeypatch, tmp_path):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["single_factors"]["chemical"] = {"kind": "explicit"}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {2: [0, 1, 2]}}
    config["catalogue"] = {
        "ranks": [2], "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 0},
        "source_block_partitions_by_rank": {2: [[1, 1]]},
    }
    runtime["evaluator"] = "native_cpu"
    runtime["neighbors"] = "ase"
    runtime["cache"]["mode"] = "auto"
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"] is True
    assert basis.resolution.capability_report["selected_evaluator"] == "native_cpu"
    assert basis.elements == ("Cu", "Ni")
    atoms = Atoms("NiCuNiCu", positions=((0.0, 0.0, 0.0), (1.3, 0.2, 0.4),
                                          (-0.4, 1.4, 0.6), (0.7, -0.5, 1.7)),
                  cell=(8.0, 8.0, 8.0), pbc=True)
    rows = basis.create(atoms)
    assert basis.elements == ("Cu", "Ni")
    assert rows.shape == (len(atoms), len(basis.labels))
    assert np.isfinite(rows).all() and np.linalg.norm(rows) > 1e-8
    reordered = atoms[[2, 0, 3, 1]]
    np.testing.assert_allclose(basis.create(reordered), rows[[2, 0, 3, 1]], rtol=0, atol=1e-10)
    changed_chemistry = atoms.copy()
    changed_chemistry.symbols[1] = "Ni"
    assert np.linalg.norm(basis.create(changed_chemistry) - rows) > 1e-8

    one_hot = json.loads(json.dumps(config))
    one_hot["single_factors"]["chemical"] = {"kind": "one_hot"}
    one_hot_basis = Basis.from_config(one_hot, representation=rep, runtime=runtime)
    np.testing.assert_allclose(one_hot_basis.create(atoms), rows, rtol=0, atol=1e-12)
    assert [label.identity for label in one_hot_basis.labels] == [
        label.identity for label in basis.labels]

    training = []
    for index, shift in enumerate((0.0, 0.11, -0.09, 0.23)):
        frame = atoms.copy()
        frame.positions[1, 0] += shift
        frame.info["energy"] = -1.3 + 0.17 * index + 0.04 * shift
        frame.arrays["forces"] = np.zeros((len(frame), 3))
        training.append(frame)
    model = LinearModel(basis).fit(training, force_weight=0.0)
    assert all(np.linalg.norm(coefficient.detach().numpy()) > 1e-8
               for coefficient in model._fitted.beta_by_species.values())
    reference = atoms.copy()
    reference.calc = model.ase_calculator(evaluator="torch")
    native_calc = model.ase_calculator(evaluator="auto", native_library=library)
    native = atoms.copy()
    native.calc = native_calc
    try:
        assert np.linalg.norm(reference.get_forces()) > 1e-8
        assert np.linalg.norm(reference.get_stress()) > 1e-8
        energy = native.get_potential_energy()
        forces = native.get_forces()
        stress = native.get_stress()
        np.testing.assert_allclose(energy, reference.get_potential_energy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(forces, reference.get_forces(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(stress, reference.get_stress(), rtol=0, atol=1e-8)
        np.testing.assert_allclose(native_calc.native_runtime.evaluate_atoms(
            atoms, return_features=True)[4], rows, rtol=0, atol=1e-8)
        assert native_calc.native_runtime.neighbors == "ase"
        reordered.calc = native_calc
        np.testing.assert_allclose(reordered.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(reordered.get_forces(), forces[[2, 0, 3, 1]],
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(reordered.get_stress(), stress, rtol=0, atol=1e-8)

        step = 1e-5
        displaced_energies = []
        for direction in (-1, 1):
            displaced = atoms.copy()
            displaced.positions[1, 0] += direction * step
            displaced.calc = native_calc
            displaced_energies.append(displaced.get_potential_energy())
        np.testing.assert_allclose(forces[1, 0],
                                   -(displaced_energies[1] - displaced_energies[0]) / (2 * step),
                                   rtol=0, atol=5e-6)
        for index, (first, second) in enumerate(((0, 0), (1, 1), (2, 2),
                                                  (1, 2), (0, 2), (0, 1))):
            strained_energies = []
            for direction in (-1, 1):
                strain = np.zeros((3, 3))
                strain[first, second] = direction * step
                if first != second:
                    strain[second, first] = direction * step
                deformed = atoms.copy()
                deformed.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + strain),
                                  scale_atoms=True)
                deformed.calc = native_calc
                strained_energies.append(deformed.get_potential_energy())
            multiplier = 2 if first != second else 1
            derivative = (strained_energies[1] - strained_energies[0]) / (
                2 * step * atoms.get_volume() * multiplier)
            np.testing.assert_allclose(stress[index], derivative, rtol=0, atol=5e-6)
    finally:
        native_calc.native_runtime.close()


def test_two_species_tagged_angular_radial_native_matches_reference(monkeypatch, tmp_path):
    from ase.stress import voigt_6_to_full_3x3_stress
    from ye3t_methods.atomistic.tagged_cauchy_image import TaggedCauchyImageLinearModel

    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni", "Cu"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {2: [0]}}
    config["catalogue"] = {
        "ranks": [2], "nmax_per_rank": {2: 2}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: [[2]]},
    }
    runtime["evaluator"] = "native_cpu"
    runtime["neighbors"] = "ase"
    runtime["cache"]["mode"] = "auto"
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    positions = ((0.0, 0.0, 0.0), (1.3, 0.2, 0.4),
                 (-0.4, 1.4, 0.6), (0.7, -0.5, 1.7))
    mixed = Atoms("NiCuNiCu", positions=positions, cell=(8.0, 8.0, 8.0), pbc=True)
    mixed_rows = basis.create(mixed)
    coordinates = [label.as_dict()["compiler_coordinate_provenance"] for label in basis.labels]
    angular_indices = [index for index, row in enumerate(coordinates)
                       if any(channel["l"] == 1 for channel in row["label"]["block_complete_channel_keys"])]
    radial_indices = [index for index, row in enumerate(coordinates)
                      if any(channel["radial_channel"] == 1
                             for channel in row["label"]["block_complete_channel_keys"])]
    assert angular_indices and radial_indices
    assert np.max(np.abs(mixed_rows[:, angular_indices])) > 1e-8
    assert np.max(np.abs(mixed_rows[:, radial_indices])) > 1e-8
    evaluator = basis._descriptor.metadata["tagged_cauchy_image_evaluator"]
    width = evaluator.feature_count
    model = TaggedCauchyImageLinearModel(
        evaluator,
        {"Cu": np.linspace(-0.4, -0.2, width),
         "Ni": np.linspace(0.3, 0.5, width)},
        {"Cu": 0.35, "Ni": -0.2})
    native_calc = model.ase_calculator(
        backend="native_cpu", native_library=library, neighbors="ase")
    try:
        results = {}
        for symbols in ("Ni4", "Cu4", "NiCuNiCu"):
            atoms = Atoms(symbols, positions=positions, cell=(8.0, 8.0, 8.0), pbc=True)
            reference = atoms.copy()
            reference.calc = model.ase_calculator(backend="reference")
            native = atoms.copy()
            native.calc = native_calc
            energy, forces, stress = (native.get_potential_energy(), native.get_forces(),
                                      native.get_stress())
            np.testing.assert_allclose(energy, reference.get_potential_energy(),
                                       rtol=0, atol=1e-9)
            np.testing.assert_allclose(forces, reference.get_forces(), rtol=0, atol=1e-8)
            np.testing.assert_allclose(stress, reference.get_stress(), rtol=0, atol=1e-8)
            np.testing.assert_allclose(native_calc.native_runtime.evaluate_atoms(
                atoms, return_features=True)[4], basis.create(atoms), rtol=0, atol=1e-8)
            results[symbols] = (energy, forces, stress)
        assert abs(results["Ni4"][0] - results["Cu4"][0]) > 1e-4

        axis = np.array((1.0, 2.0, 3.0))
        axis /= np.linalg.norm(axis)
        cross = np.array(((0.0, -axis[2], axis[1]),
                          (axis[2], 0.0, -axis[0]),
                          (-axis[1], axis[0], 0.0)))
        angle = np.deg2rad(37.0)
        rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
        rotated = mixed.copy()
        rotated.positions[:] = mixed.positions @ rotation.T
        rotated.set_cell(np.asarray(mixed.cell) @ rotation.T, scale_atoms=False)
        rotated.calc = native_calc
        energy, forces, stress = results["NiCuNiCu"]
        np.testing.assert_allclose(basis.create(rotated), mixed_rows, rtol=0, atol=1e-9)
        np.testing.assert_allclose(native_calc.native_runtime.evaluate_atoms(
            rotated, return_features=True)[4], mixed_rows, rtol=0, atol=1e-8)
        np.testing.assert_allclose(rotated.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(rotated.get_forces(), forces @ rotation.T, rtol=0, atol=1e-8)
        np.testing.assert_allclose(voigt_6_to_full_3x3_stress(rotated.get_stress()),
                                   rotation @ voigt_6_to_full_3x3_stress(stress) @ rotation.T,
                                   rtol=0, atol=1e-8)
        inverted = mixed.copy()
        inverted.positions *= -1
        inverted.calc = native_calc
        np.testing.assert_allclose(basis.create(inverted), mixed_rows, rtol=0, atol=1e-9)
        np.testing.assert_allclose(inverted.get_potential_energy(), energy, rtol=0, atol=1e-9)
        np.testing.assert_allclose(inverted.get_forces(), -forces, rtol=0, atol=1e-8)
        np.testing.assert_allclose(inverted.get_stress(), stress, rtol=0, atol=1e-8)

        step = 1e-5
        atom_index, axis_index = np.unravel_index(np.argmax(np.abs(forces)), forces.shape)
        assert abs(forces[atom_index, axis_index]) > 1e-5
        displaced_energies = []
        for direction in (-1, 1):
            displaced = mixed.copy()
            displaced.positions[atom_index, axis_index] += direction * step
            displaced.calc = native_calc
            displaced_energies.append(displaced.get_potential_energy())
        np.testing.assert_allclose(
            forces[atom_index, axis_index],
            -(displaced_energies[1] - displaced_energies[0]) / (2 * step),
            rtol=0, atol=5e-6)
        stress_index = 3 + int(np.argmax(np.abs(stress[3:])))
        assert abs(stress[stress_index]) > 1e-7
        first, second = ((1, 2), (0, 2), (0, 1))[stress_index - 3]
        strained_energies = []
        for direction in (-1, 1):
            strain = np.zeros((3, 3))
            strain[first, second] = direction * step
            strain[second, first] = direction * step
            deformed = mixed.copy()
            deformed.set_cell(np.asarray(mixed.cell) @ (np.eye(3) + strain),
                              scale_atoms=True)
            deformed.calc = native_calc
            strained_energies.append(deformed.get_potential_energy())
        np.testing.assert_allclose(
            stress[stress_index],
            (strained_energies[1] - strained_energies[0]) /
            (4 * step * mixed.get_volume()),
            rtol=0, atol=5e-6)
    finally:
        native_calc.native_runtime.close()


def test_configured_tagged_native_auto_matches_reference(monkeypatch, tmp_path):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {2: [0, 2]}}
    config["catalogue"] = {
        "ranks": [2], "nmax_per_rank": {2: 1}, "lmax_per_rank": {2: 1},
        "source_block_partitions_by_rank": {2: [[2]]},
    }
    runtime["evaluator"] = "native_cpu"
    runtime["neighbors"] = "ase"
    runtime["cache"]["mode"] = "auto"
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"] is True
    assert basis.resolution.capability_report["selected_evaluator"] == "native_cpu"
    assert basis.resolution.capability_report["selected_neighbors"] == "ase"
    atoms = Atoms("Ni4", positions=((0, 0, 0), (1.5, 0.2, 0.4),
                                     (-0.3, 1.6, 0.7), (0.8, -0.4, 1.8)),
                  cell=(8.0, 8.0, 8.0), pbc=True)
    rows = basis.create(atoms)
    assert rows.shape == (len(atoms), len(basis.labels))
    reference_runtime = dict(runtime, evaluator="reference", neighbors="auto")
    reference_basis = Basis.from_config(config, representation=rep,
                                        runtime=reference_runtime)
    np.testing.assert_allclose(rows, reference_basis.create(atoms), rtol=0, atol=1e-12)
    training = atoms.copy()
    training.info["energy"] = -1.23
    training.arrays["forces"] = np.zeros((len(training), 3))
    shifted = atoms.copy()
    shifted.positions[1, 0] += 0.18
    shifted.info["energy"] = -0.72
    shifted.arrays["forces"] = np.zeros((len(shifted), 3))
    model = LinearModel(basis).fit([training, shifted], force_weight=0.0)
    expected = atoms.copy()
    expected.calc = model.ase_calculator(backend="torch")
    actual = atoms.copy()
    actual.calc = model.ase_calculator(evaluator="auto", native_library=library)
    try:
        assert actual.calc.native_runtime.neighbors == "ase"
        assert actual.calc.native_runtime.last_neighbor_backend is None
        np.testing.assert_allclose(actual.get_potential_energy(),
                                   expected.get_potential_energy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(actual.get_forces(), expected.get_forces(),
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(actual.get_stress(), expected.get_stress(),
                                   rtol=0, atol=1e-8)
        assert actual.calc.native_runtime.last_neighbor_backend == "ase_neighbor_list"
    finally:
        actual.calc.native_runtime.close()


@pytest.mark.parametrize("runtime_change", (
    {"device": "cuda:0"},
    {"neighbors": "ase"},
    {"cache": {"mode": "off"}},
))
def test_tagged_config_explicit_unintegrated_runtime_fails_closed(runtime_change):
    rep, config, _ = _config()
    config["single_factors"]["species"] = ["Ni"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {1: [0]}}
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
    }
    runtime = {"evaluator": "reference", "neighbors": "auto",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    runtime.update(runtime_change)
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"] is False
    with pytest.raises(RuntimeError, match="P2 evaluator integration"):
        basis.create(Atoms("Ni", positions=((0, 0, 0),)))


def test_tagged_raw_preview_is_not_physical_image_dimension(monkeypatch, tmp_path):
    from ye3t.couplings import compile, plan, tagged_cauchy_image_request

    rep, config, runtime = _config()
    config["single_factors"]["species"] = ["H"]
    config["single_factors"]["radial"] = {"family": "shifted_jacobi", "cutoff_A": 4.5}
    config["tensor_product"] = {"kind": "tagged", "tag_counts_per_rank": {1: [0, 1]}}
    config["catalogue"] = {
        "ranks": [1], "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
        "selection": {"repeated_content_min": 1},
    }
    runtime = {"evaluator": "reference", "neighbors": "auto",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(config, representation=rep, runtime=runtime)
    preview = basis.catalogue.counts()["by_component"]["main"]
    request = tagged_cauchy_image_request(species=["H"], catalogue={
        "nmax_per_rank": {1: 1}, "lmax_per_rank": {1: 0},
        "source_block_partitions_by_rank": {1: [[1]]},
        "tag_counts_by_rank": {1: [0, 1]},
        "max_records_per_rank": 1, "max_features_per_rank": 3,
    })
    compiled = compile(plan(request))
    assert preview["raw_opportunity_count"] == 3
    assert preview["physical_image_upper_bound"] == 3
    assert preview["exact_image_count"] is None
    assert len(compiled.payload["image_rows"]) == 1
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    rows = basis.create(Atoms("H2", positions=((0, 0, 0), (1.2, 0, 0))))
    assert rows.shape == (2, 1)
    assert len(basis.labels) == 1


def test_basis_rejects_ambiguous_normalized_rank_keys():
    rep, config, runtime = _config()
    config["catalogue"]["nmax_per_rank"] = {1: 4, "1": 3, 2: 4}
    with pytest.raises(ValueError, match="duplicate normalized key 1"):
        Basis.from_config(config, representation=rep, runtime=runtime)
    rep, config, runtime = _config()
    config["single_factors"]["species"] = "Ni"
    with pytest.raises(TypeError, match="species must be a sequence"):
        Basis.from_config(config, representation=rep, runtime=runtime)
