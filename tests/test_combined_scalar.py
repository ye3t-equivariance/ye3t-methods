import hashlib
import importlib.util
import json
import numpy as np
import os
import pytest
import subprocess
import sys
import zipfile
from copy import deepcopy
from ase import Atoms

from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel
from ye3t_methods.linear import _combined_scalar_geometry_row


def test_combined_density_tagged_fit_matches_independent_derivatives(tmp_path):
    ordinary = Basis(elements=["Ni"], source="density", cutoff=3.4,
                     max_rank=1, nmax=1, lmax=0,
                     descriptor_cache_dir=tmp_path / "density")
    tagged = Basis(
        elements=["Ni"], source="tagged_cauchy_image", cutoff=3.2,
        rank=2, tag_counts=(0, 2), nmax_per_rank={2: 1},
        lmax_per_rank={2: 1}, source_block_partitions_by_rank={2: ((2,),)},
        max_records_per_rank=6, max_features_per_rank=8,
        backend="reference", compiled_cache_dir=tmp_path / "tagged",
        compiler_validation="full",
    )
    basis = Basis.combine({"ordinary": ordinary, "tagged": tagged})
    with pytest.raises(ValueError, match="density component"):
        Basis.combine({"first": tagged, "second": tagged})
    assert basis.cutoff == 3.4
    assert [row["cutoff_A"] for row in basis.resolved["components"]] == [3.4, 3.2]
    assert [row.as_dict()["component"] for row in basis.labels] == (
        ["ordinary"] * len(ordinary.labels) + ["tagged"] * len(tagged.labels))
    assert [row.feature_index for row in basis.labels] == list(range(len(basis.labels)))
    geometry = Atoms("Ni3", positions=((0.6, 0.4, 0.3), (1.8, 0.8, 0.5),
                                       (0.9, 2.0, 0.8)), cell=(6.2, 6.3, 6.4), pbc=True)
    np.testing.assert_allclose(basis.create(geometry),
                               np.column_stack((ordinary.create(geometry),
                                                tagged.create(geometry))), atol=1e-12)
    compiled = tagged._descriptor.metadata["tagged_cauchy_image_compiled"]
    angular_components = compiled.plan.report.resource_report["components"]
    angular_columns = [index for index, record in enumerate(
        compiled.payload["image_coordinate_provenance"])
        if tuple(angular_components[record["component_index"]]["angular_pattern"]) == (1, 1)]
    assert angular_columns
    assert np.linalg.norm(tagged.create(geometry)[:, angular_columns]) > 1e-8
    reordered = geometry[[2, 0, 1]]
    np.testing.assert_allclose(basis.create(reordered),
                               basis.create(geometry)[[2, 0, 1]], atol=2e-10)
    angle = np.deg2rad(31.0)
    rotation = np.array(((np.cos(angle), -np.sin(angle), 0.0),
                         (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0)))
    rotated = geometry.copy()
    rotated.positions[:] = geometry.positions @ rotation.T
    rotated.set_cell(geometry.cell.array @ rotation.T, scale_atoms=False)
    np.testing.assert_allclose(basis.create(rotated), basis.create(geometry),
                               atol=2e-9)
    coefficient = np.linspace(0.13, 0.28, len(basis.labels))
    offset = -0.19

    def energy(atoms):
        return float(basis.create(atoms).sum(axis=0) @ coefficient + len(atoms) * offset)

    step = 2e-5
    training = []
    for distance in (0.0, 0.12, -0.08):
        atoms = geometry.copy()
        atoms.positions[1, 0] += distance
        atoms.info["energy"] = energy(atoms)
        forces = np.empty((len(atoms), 3))
        for atom in range(len(atoms)):
            for axis in range(3):
                minus, plus = atoms.copy(), atoms.copy()
                minus.positions[atom, axis] -= step
                plus.positions[atom, axis] += step
                forces[atom, axis] = -(energy(plus) - energy(minus)) / (2 * step)
        atoms.arrays["forces"] = forces
        stress = np.empty(6)
        for index, (a, b) in enumerate(((0, 0), (1, 1), (2, 2),
                                        (1, 2), (0, 2), (0, 1))):
            values = []
            for sign in (-1, 1):
                strain = np.eye(3)
                strain[a, b] += sign * step
                if a != b:
                    strain[b, a] += sign * step
                trial = atoms.copy()
                trial.positions[:] = atoms.positions @ strain
                trial.set_cell(atoms.cell.array @ strain, scale_atoms=False)
                values.append(energy(trial))
            stress[index] = ((values[1] - values[0]) / (2 * step) /
                             atoms.get_volume() / (2 if a != b else 1))
        atoms.info["stress"] = stress
        training.append(atoms)

    model = LinearModel(basis, reference_energies={"Ni": -0.05}).fit(
        training, regularization=1e-12, stress_weight=1.0, fit_E0=True)
    invalid_cell = training[0].copy()
    invalid_cell.set_cell(np.zeros((3, 3)), scale_atoms=False)
    invalid_cell.pbc = False
    with pytest.raises(ValueError, match="positive full-rank cell"):
        LinearModel(basis).fit([invalid_cell], stress_weight=1.0, fit_E0=True)
    assert set(model._fitted["offsets_eV"]) == {"Ni"}
    for atoms in training:
        predicted = atoms.copy()
        predicted.calc = model.ase_calculator(evaluator="torch")
        np.testing.assert_allclose(predicted.get_potential_energy(),
                                   atoms.info["energy"], rtol=0, atol=3e-5)
        np.testing.assert_allclose(predicted.get_forces(), atoms.arrays["forces"],
                                   rtol=0, atol=3e-5)
        np.testing.assert_allclose(predicted.get_stress(), atoms.info["stress"],
                                   rtol=0, atol=3e-5)
    reordered.calc = model.ase_calculator(evaluator="torch")
    geometry.calc = model.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(reordered.get_potential_energy(),
                               geometry.get_potential_energy(), atol=2e-9)
    np.testing.assert_allclose(reordered.get_forces(),
                               geometry.get_forces()[[2, 0, 1]], atol=2e-8)
    np.testing.assert_allclose(reordered.get_stress(), geometry.get_stress(),
                               atol=2e-8)
    rotated.calc = model.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(rotated.get_potential_energy(),
                               geometry.get_potential_energy(), atol=2e-9)
    np.testing.assert_allclose(rotated.get_forces(),
                               geometry.get_forces() @ rotation.T, atol=2e-8)
    def tensor(voigt):
        return np.array(((voigt[0], voigt[5], voigt[4]),
                         (voigt[5], voigt[1], voigt[3]),
                         (voigt[4], voigt[3], voigt[2])))

    np.testing.assert_allclose(tensor(rotated.get_stress()),
                               rotation @ tensor(geometry.get_stress()) @ rotation.T,
                               atol=2e-8)
    inverted = geometry.copy()
    inverted.positions[:] = -geometry.positions
    inverted.calc = model.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(inverted.get_potential_energy(),
                               geometry.get_potential_energy(), atol=2e-9)
    np.testing.assert_allclose(inverted.get_forces(), -geometry.get_forces(),
                               atol=2e-8)
    np.testing.assert_allclose(inverted.get_stress(), geometry.get_stress(),
                               atol=2e-8)


def test_combine_rejects_different_species_and_missing_tagged(tmp_path):
    nickel = Basis(elements=["Ni"], source="density", cutoff=3.4,
                   max_rank=1, nmax=1, lmax=0,
                   descriptor_cache_dir=tmp_path / "ni")
    copper = Basis(elements=["Cu"], source="density", cutoff=3.4,
                   max_rank=1, nmax=1, lmax=0,
                   descriptor_cache_dir=tmp_path / "cu")
    with pytest.raises(ValueError, match="same ordered species"):
        Basis.combine({"nickel": nickel, "copper": copper})
    with pytest.raises(ValueError, match="tagged component"):
        Basis.combine({"first": nickel, "second": nickel})


def test_loaded_multispecies_combined_density_rejects_unvalidated_native(tmp_path):
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [1, 2],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 2, 2: 2},
            "l_max_per_rank": {1: 0, 2: 0},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    config = {
        "single_factors": {
            "species": ["Ni", "Cu"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 3.5,
                       "cutoff_width_A": 0.01, "lambda": 0.79},
            "chemical": {"kind": "explicit"},
        },
        "components": {
            "ordinary": {
                "tensor_product": {"kind": "density"},
                "catalogue": {"ranks": [1, 2], "nmax_per_rank": {1: 1, 2: 1},
                              "lmax_per_rank": {1: 0, 2: 0},
                              "source_block_partitions_by_rank": {
                                  1: [[1]], 2: [[2], [1, 1]]}},
            },
            "tagged": {
                "single_factors": {
                    "species": ["Ni", "Cu"],
                    "radial": {"family": "shifted_jacobi", "cutoff_A": 3.1},
                    "chemical": {"kind": "explicit"},
                },
                "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {2: [0, 2]}},
                "catalogue": {"ranks": [2], "nmax_per_rank": {2: 1},
                              "lmax_per_rank": {2: 0},
                              "source_block_partitions_by_rank": {2: [[2]]}},
            },
        },
    }
    runtime = {"evaluator": "auto", "neighbors": "auto",
               "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(config, representation=representation, runtime=runtime)
    assert basis.resolution.capability_report["basis_create_available"]
    frames = []
    for formula, shift in (("Ni2Cu", 0.0), ("NiCu2", 0.1),
                           ("Ni2Cu", -0.1), ("NiCu2", 0.2)):
        atoms = Atoms(formula, positions=((0.0, 0.0, 0.0),
                                          (1.4 + shift, 0.2, 0.1),
                                          (0.2, 1.6, 0.3)),
                      cell=(7.0, 7.0, 7.0), pbc=True)
        atoms.info["energy"] = float(basis.create(atoms).sum(axis=0) @
                                     np.linspace(0.02, 0.12, len(basis.labels)))
        frames.append(atoms)
    model = LinearModel(basis).fit(frames, force_weight=0.0, fit_E0=True)
    artifact = model.write(tmp_path / "two_species.ye3t")
    loaded = LinearModel.read(artifact)
    assert any(str(spec.key).endswith("|physical_eta_bound")
               for spec in loaded._fitted["components"]["ordinary"].descriptor_specs)
    assert any(spec.label.rank == 2
               for spec in loaded._fitted["components"]["ordinary"].descriptor_specs)
    original = frames[0].copy()
    original.calc = model.ase_calculator(evaluator="torch")
    restored = frames[0].copy()
    restored.calc = loaded.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(restored.get_potential_energy(),
                               original.get_potential_energy(), atol=1e-9)
    np.testing.assert_allclose(restored.get_forces(), original.get_forces(), atol=1e-8)
    np.testing.assert_allclose(restored.get_stress(), original.get_stress(), atol=1e-8)
    with pytest.raises(ValueError, match="no validated native_cpu evaluator"):
        loaded.ase_calculator(evaluator="native_cpu", neighbors="ase")


def test_combine_independently_configured_pace_and_jacobi(tmp_path, monkeypatch):
    monkeypatch.setenv("YE3T_CACHE_DIR", str(tmp_path))
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [1, 2],
        "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {1: 1, 2: 1},
            "l_max_per_rank": {1: 0, 2: 1},
        },
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    ordinary_config = {
        "single_factors": {
            "species": ["Ni"],
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 3.5,
                       "cutoff_width_A": 0.01, "lambda": 0.79},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "density"},
        "catalogue": {"ranks": [1], "nmax_per_rank": {1: 1},
                      "lmax_per_rank": {1: 0},
                      "source_block_partitions_by_rank": {1: [[1]]}},
    }
    tagged_config = {
        "single_factors": {
            "species": ["Ni"],
            "radial": {"family": "shifted_jacobi", "cutoff_A": 3.1},
            "chemical": {"kind": "explicit"},
        },
        "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {2: [0, 2]}},
        "catalogue": {"ranks": [2], "nmax_per_rank": {2: 1},
                      "lmax_per_rank": {2: 1},
                      "source_block_partitions_by_rank": {2: [[2]]}},
    }
    ordinary = Basis.from_config(
        ordinary_config, representation=representation,
        runtime={"evaluator": "torch", "neighbors": "ase",
                 "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"})
    tagged = Basis.from_config(
        tagged_config, representation=representation,
        runtime={"evaluator": "reference", "neighbors": "auto",
                 "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"})
    assert ordinary.resolution.capability_report["basis_create_available"]
    assert tagged.resolution.capability_report["basis_create_available"]
    basis = Basis.combine({"ordinary": ordinary, "tagged": tagged})
    assert [item["radial_source"]["family"] for item in basis.resolved["components"]] == [
        "pace_chebexp_cos", "shifted_jacobi"]
    atoms = Atoms("Ni3", positions=((0.0, 0.0, 0.0), (1.5, 0.2, 0.4),
                                    (-0.3, 1.7, 0.6)))
    np.testing.assert_allclose(basis.create(atoms),
                               np.column_stack((ordinary.create(atoms),
                                                tagged.create(atoms))), atol=1e-12)
    application = ordinary._descriptor.ace_descriptor.ordinary_scalar_catalogue
    assert application["schema"] == "ye3t_ordinary_scalar_catalogue_v2"
    import ye3t.couplings
    from ye3t_methods.atomistic.ace import lammps_export

    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs:
                      pytest.fail("core coupling compiled during saved-program replay"))
        patch.setattr(lammps_export, "compile_ordinary_scalar_catalogue",
                      lambda *args, **kwargs:
                      pytest.fail("ordinary catalogue compiled during saved-program replay"))
        replay = Basis.from_config(
            ordinary_config, representation=representation,
            runtime={"evaluator": "torch", "neighbors": "ase",
                     "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"})
        replay._materialize_configured(ordinary_catalogue=application)
        np.testing.assert_allclose(replay.create(atoms), ordinary.create(atoms), atol=1e-12)
    training = []
    for shift in (0.0, 0.12, -0.08, 0.22):
        frame = atoms.copy()
        frame.positions[1, 0] += shift
        frame.set_cell((6.4, 6.5, 6.6))
        frame.pbc = True
        frame.info["energy"] = float(basis.create(frame).sum(axis=0) @
                                      np.linspace(0.1, 0.2, len(basis.labels)) + 0.07 * len(frame))
        training.append(frame)
    named_config = {
        "single_factors": ordinary_config["single_factors"],
        "components": {
            "ordinary": {"tensor_product": ordinary_config["tensor_product"],
                         "catalogue": ordinary_config["catalogue"]},
            "tagged": {"single_factors": tagged_config["single_factors"],
                       "tensor_product": tagged_config["tensor_product"],
                       "catalogue": tagged_config["catalogue"]},
        },
    }
    named_runtime = {"evaluator": "auto", "neighbors": "auto",
                     "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"}
    named = Basis.from_config(named_config, representation=representation,
                              runtime=named_runtime)
    capability = named.resolution.capability_report
    assert capability["basis_create_available"]
    assert capability["selected_evaluator"] == "component_local"
    assert capability["selected_neighbors"] == "component_local"
    assert capability["component_runtime"]["ordinary"]["selected_evaluator"] == "torch"
    assert capability["component_runtime"]["tagged"]["selected_evaluator"] == "reference"
    assert not named.catalogue.counts()["coefficient_materialization_performed"]
    assert named.cutoff == 3.5
    np.testing.assert_allclose(named.create(training[0]), basis.create(training[0]), atol=1e-12)
    assert named.source == "combined_scalar"
    assert named.resolved["construction_resolution_sha256"] == named.resolution.sha256
    for entry in named.resolved["components"]:
        assert len(entry["resolution_sha256"]) == 64
        assert entry["single_factors"]["chemical"]["kind"] == "explicit"
    same_runtime = Basis.combine({
        "ordinary": Basis.from_config(ordinary_config, representation=representation,
                                      runtime=named_runtime),
        "tagged": Basis.from_config(tagged_config, representation=representation,
                                    runtime=named_runtime),
    })
    assert named.resolved["components"] == same_runtime.resolved["components"]
    named_fit = LinearModel(named).fit(training, force_weight=0.0, fit_E0=True)
    full_config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "joint_ni", "status": "stable"},
        "representation": representation.to_dict(),
        "basis": named_config,
        "runtime": named_runtime,
        "model": {"kind": "linear", "fit": {"solver": "ridge", "alpha": 1e-8,
                                               "weights": {"energy": 1.0, "forces": 0.0}},
                  "reference_energy": {"per_species_E0_eV": {"Ni": 0.0},
                                       "fit_E0": True}},
        "targets": {"energy": "energy", "forces": None, "stress": None},
        "validation": {"checks": ["force_fd", "round_trip"]},
    }
    configured_fit = LinearModel(named).fit(training, config=full_config)
    validation = configured_fit._fitted["fit_metadata"]["configured_validation"]
    assert validation["results"]["force_fd"]["absolute_error"] < 1e-4
    assert validation["results"]["round_trip"]["passed"]
    fit_record = configured_fit._fitted["fit_metadata"]["resolved_fit_config"]
    assert fit_record["basis_resolution_sha256"] == named.resolution.sha256
    assert fit_record["training_provenance_status"] == (
        "recorded_not_derivable_from_deployed_coefficients")
    assert fit_record["targets"] == {"energy": "energy", "forces": None,
                                     "stress": None}
    configured_artifact = configured_fit.write(tmp_path / "configured_fit.ye3t")
    configured_loaded = LinearModel.read(configured_artifact)
    assert configured_loaded._fitted["fit_metadata"]["resolved_fit_config"] == fit_record
    optional_config = deepcopy(full_config)
    optional_config["runtime"] = {**named_runtime, "evaluator": "native_cpu",
                                  "neighbors": "matscipy"}
    optional_config["validation"] = {"checks": []}
    find_spec = importlib.util.find_spec
    with monkeypatch.context() as available:
        available.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs:
                          object() if name == "matscipy" else
                          find_spec(name, *args, **kwargs))
        optional_basis = Basis.from_config(
            named_config, representation=representation,
            runtime=optional_config["runtime"])
        optional_fit = LinearModel(optional_basis).fit(training, config=optional_config)
        optional_artifact = optional_fit.write(tmp_path / "optional_neighbors.ye3t")
    with monkeypatch.context() as patch:
        patch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs:
                      None if name == "matscipy" else find_spec(name, *args, **kwargs))
        with pytest.raises(ImportError, match="optional matscipy"):
            Basis.from_config(named_config, representation=representation,
                              runtime=optional_config["runtime"])
        optional_loaded = LinearModel.read(optional_artifact)
        optional_probe = training[0].copy()
        optional_probe.calc = optional_loaded.ase_calculator(
            evaluator="native_cpu", neighbors="ase")
        original_probe = training[0].copy()
        original_probe.calc = optional_fit.ase_calculator(
            evaluator="native_cpu", neighbors="ase")
        np.testing.assert_allclose(optional_probe.get_potential_energy(),
                                   original_probe.get_potential_energy(), atol=1e-9)
        np.testing.assert_allclose(optional_probe.get_forces(),
                                   original_probe.get_forces(), atol=1e-8)
        np.testing.assert_allclose(optional_probe.get_stress(),
                                   original_probe.get_stress(), atol=1e-8)
        with monkeypatch.context() as missing_package:
            missing_package.setitem(sys.modules, "matscipy", None)
            missing_package.setitem(sys.modules, "matscipy.neighbours", None)
            with pytest.raises(ImportError, match="optional matscipy"):
                optional_loaded.ase_calculator(
                    evaluator="native_cpu", neighbors="matscipy")
    shifted_reference = deepcopy(full_config)
    shifted_reference["model"]["reference_energy"]["per_species_E0_eV"] = {"Ni": 0.2}
    shifted_fit = LinearModel(named).fit(training, config=shifted_reference)
    shifted_probe = training[1].copy()
    shifted_probe.calc = shifted_fit.ase_calculator(evaluator="torch")
    baseline_probe = training[1].copy()
    baseline_probe.calc = configured_fit.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(shifted_probe.get_potential_energy(),
                               baseline_probe.get_potential_energy(), atol=1e-9)
    np.testing.assert_allclose(shifted_probe.get_forces(),
                               baseline_probe.get_forces(), atol=1e-5)
    np.testing.assert_allclose(shifted_probe.get_stress(),
                               baseline_probe.get_stress(), atol=1e-6)
    assert shifted_fit._fitted["components"]["tagged"].offsets == (
        shifted_fit._fitted["offsets_eV"])
    prior_fit = named_fit._fitted
    with monkeypatch.context() as patch:
        def fail_validation(*args, **kwargs):
            raise RuntimeError("forced validation failure")
        patch.setattr(named_fit, "ase_calculator", fail_validation)
        with pytest.raises(RuntimeError, match="forced validation failure"):
            named_fit.fit(training, config=full_config)
    assert named_fit._fitted is prior_fit
    with zipfile.ZipFile(configured_artifact) as source:
        configured_members = {name: source.read(name) for name in source.namelist()}
    for name in named._components:
        if named._components[name].source == "density":
            np.testing.assert_allclose(configured_fit._fitted["components"][name].weight,
                                       named_fit._fitted["components"][name].weight, atol=1e-10)
    invalid_full = deepcopy(full_config)
    invalid_full["basis"]["components"]["tagged"]["single_factors"]["radial"]["cutoff_A"] = 2.9
    with pytest.raises(ValueError, match="differs from the constructed Basis"):
        LinearModel(named).fit(training, config=invalid_full)
    with pytest.raises(ValueError, match="config or direct fit arguments"):
        LinearModel(named).fit(training, config=full_config, force_weight=0.0)
    with pytest.raises(ValueError, match="config or direct fit arguments"):
        LinearModel(named).fit(training, config=full_config, force_weight=1.0)
    named_artifact = named_fit.write(tmp_path / "named_combined.ye3t")
    named_loaded = LinearModel.read(named_artifact)
    np.testing.assert_allclose(named_loaded.basis.create(training[0]),
                               named.create(training[0]), atol=1e-12)
    invalid_named = deepcopy(named_config)
    invalid_named["components"]["tagged"]["single_factors"]["radial"]["family"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported radial family"):
        Basis.from_config(invalid_named, representation=representation,
                          runtime=named_runtime)
    invalid_species = deepcopy(named_config)
    invalid_species["components"]["tagged"]["single_factors"]["species"] = ["Cu"]
    with pytest.raises(ValueError, match="Component-local species"):
        Basis.from_config(invalid_species, representation=representation,
                          runtime=named_runtime)
    model = LinearModel(basis).fit(training, force_weight=0.0, fit_E0=True)
    named_probe = training[1].copy()
    named_probe.calc = named_fit.ase_calculator(evaluator="torch")
    direct_probe = training[1].copy()
    direct_probe.calc = model.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(named_probe.get_potential_energy(),
                               direct_probe.get_potential_energy(), atol=1e-10)
    np.testing.assert_allclose(named_probe.get_forces(),
                               direct_probe.get_forces(), atol=1e-9)
    np.testing.assert_allclose(named_probe.get_stress(),
                               direct_probe.get_stress(), atol=1e-9)
    stress_training = []
    for frame in training:
        source = frame.copy()
        source.calc = model.ase_calculator(evaluator="torch")
        target = frame.copy()
        target.info["energy"] = float(source.get_potential_energy())
        target.arrays["forces"] = source.get_forces()
        target.info["stress"] = source.get_stress()
        stress_training.append(target)
    stress_config = deepcopy(full_config)
    stress_config["model"]["fit"]["alpha"] = 1e-12
    stress_config["model"]["fit"]["weights"] = {
        "energy": 1.0, "forces": 1.0, "stress": 0.1}
    stress_config["targets"] = {"energy": "energy", "forces": "forces",
                                "stress": "stress"}
    stress_fit = LinearModel(named).fit(stress_training, config=stress_config)
    stress_probe = training[1].copy()
    stress_probe.calc = stress_fit.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(stress_probe.get_potential_energy(),
                               direct_probe.get_potential_energy(), atol=1e-6)
    np.testing.assert_allclose(stress_probe.get_forces(),
                               direct_probe.get_forces(), atol=1e-5)
    np.testing.assert_allclose(stress_probe.get_stress(),
                               direct_probe.get_stress(), atol=1e-6)
    ard_config = deepcopy(full_config)
    ard_config["model"]["fit"] = {"solver": "ard", "solver_options": {},
                                  "weights": {"energy": 1.0, "forces": 0.0}}
    ard_config["validation"] = {"checks": []}
    ard_from_config = LinearModel(named).fit(training, config=ard_config)
    assert ard_from_config._fitted["fit_metadata"]["fit_method"] == "ardregression"
    assert np.isfinite(ard_from_config.predict_uncertainty(training[0])[
        "total_energy_std_eV"])
    invalid_options = deepcopy(ard_config)
    invalid_options["model"]["fit"]["solver_options"] = {"fit_intercept": True}
    with pytest.raises(ValueError, match="cannot override fit_intercept"):
        LinearModel(named).fit(training, config=invalid_options)
    invalid_options["model"]["fit"]["solver_options"] = {"bad": object()}
    with pytest.raises(ValueError, match="portable finite JSON"):
        LinearModel(named).fit(training, config=invalid_options)
    lasso_config = deepcopy(full_config)
    lasso_config["model"]["fit"] = {"solver": "lasso", "alpha": 1e-6,
                                    "weights": {"energy": 1.0, "forces": 0.0}}
    lasso_config["validation"] = {"checks": []}
    lasso_from_config = LinearModel(named).fit(training, config=lasso_config)
    assert lasso_from_config._fitted["fit_metadata"]["fit_method"] == "lasso"
    artifact = model.write(tmp_path / "combined.ye3t")
    assert artifact.suffix == ".ye3t"
    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs:
                      pytest.fail("core coupling compiled while loading combined artifact"))
        patch.setattr(lammps_export, "compile_ordinary_scalar_catalogue",
                      lambda *args, **kwargs:
                      pytest.fail("ordinary compiler ran while loading combined artifact"))
        loaded = LinearModel.read(artifact)
    assert [label.as_dict() for label in loaded.labels] == [
        label.as_dict() for label in model.labels]
    copied = loaded.write(tmp_path / "new_directory" / "copied.ye3t")
    assert copied.read_bytes() == artifact.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(ye3t.couplings, "compile", lambda *args, **kwargs:
                      pytest.fail("core coupling compiled during calculator replay"))
        patch.setattr(lammps_export, "compile_ordinary_scalar_catalogue",
                      lambda *args, **kwargs:
                      pytest.fail("ordinary compiler ran during calculator replay"))
        for frame in training:
            np.testing.assert_allclose(loaded.basis.create(frame), basis.create(frame), atol=1e-12)
            original = frame.copy()
            original.calc = model.ase_calculator(evaluator="torch")
            restored = frame.copy()
            restored.calc = loaded.ase_calculator(evaluator="torch")
            np.testing.assert_allclose(restored.get_potential_energy(),
                                       original.get_potential_energy(), atol=1e-10)
            np.testing.assert_allclose(restored.get_forces(), original.get_forces(), atol=1e-9)
            np.testing.assert_allclose(restored.get_stress(), original.get_stress(), atol=1e-9)
        native = training[0].copy()
        native.calc = loaded.ase_calculator(evaluator="native_cpu", neighbors="ase")
        reference = training[0].copy()
        reference.calc = loaded.ase_calculator(evaluator="torch")
        np.testing.assert_allclose(native.get_potential_energy(),
                                   reference.get_potential_energy(), atol=1e-8)
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(), atol=1e-7)
        np.testing.assert_allclose(native.get_stress(), reference.get_stress(), atol=1e-8)
        native.set_cell((7.1, 6.5, 6.6), scale_atoms=False)
        reference.set_cell((7.1, 6.5, 6.6), scale_atoms=False)
        native.pbc = (True, False, True)
        reference.pbc = (True, False, True)
        np.testing.assert_allclose(native.get_potential_energy(),
                                   reference.get_potential_energy(), atol=1e-8)
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(), atol=1e-7)
        np.testing.assert_allclose(native.get_stress(), reference.get_stress(), atol=1e-8)
        native.positions[:] = training[-1].positions
        reference.positions[:] = training[-1].positions
        np.testing.assert_allclose(native.get_potential_energy(),
                                   reference.get_potential_energy(), atol=1e-8)
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(), atol=1e-7)
        np.testing.assert_allclose(native.get_stress(), reference.get_stress(), atol=1e-8)
    fresh_cache = tmp_path / "unavailable_global_cache"
    code = "\n".join((
        "import os, sys, numpy as np",
        "os.environ['YE3T_CACHE_DIR'] = sys.argv[2]",
        "import ye3t.couplings",
        "from ye3t_methods.atomistic.ace import lammps_export",
        "def fail(*args, **kwargs): raise AssertionError('compiler ran after restart')",
        "ye3t.couplings.compile = fail",
        "lammps_export.compile_ordinary_scalar_catalogue = fail",
        "from ye3t_methods import LinearModel",
        "from ase import Atoms",
        "model = LinearModel.read(sys.argv[1])",
        "atoms = Atoms('Ni3', positions=((0, 0, 0), (1.5, .2, .4), (-.3, 1.7, .6)), cell=(6.4, 6.5, 6.6), pbc=True)",
        "reference = atoms.copy(); reference.calc = model.ase_calculator(evaluator='torch')",
        "native = atoms.copy(); native.calc = model.ase_calculator(evaluator='native_cpu', neighbors='ase')",
        "np.testing.assert_allclose(native.get_potential_energy(), reference.get_potential_energy(), atol=1e-8)",
        "np.testing.assert_allclose(native.get_forces(), reference.get_forces(), atol=1e-7)",
        "np.testing.assert_allclose(native.get_stress(), reference.get_stress(), atol=1e-8)",
        "native.positions[1, 0] += .1; reference.positions[1, 0] += .1",
        "np.testing.assert_allclose(native.get_potential_energy(), reference.get_potential_energy(), atol=1e-8)",
    ))
    result = subprocess.run([sys.executable, "-c", code, str(artifact), str(fresh_cache)],
                            capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    assert not fresh_cache.exists()
    with zipfile.ZipFile(artifact) as source:
        members = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(members["manifest.json"])
    assert manifest["implementation_versions"]["combined_archive"] == 1
    assert len(manifest["convention_sha256"]) == 64
    assert manifest["provenance"]["compiler_api"] == "ye3t.couplings"
    assert manifest["arrays"]["weights.npz"]["per_species_E0_eV"] == {
        "shape": [1], "dtype": "float64"}

    def rehashed_copy(name, originals, member, data):
        updated = dict(originals)
        updated[member] = data
        saved_manifest = json.loads(updated["manifest.json"])
        saved_manifest["members"][member] = {
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        updated["manifest.json"] = json.dumps(saved_manifest).encode()
        target = tmp_path / name
        with zipfile.ZipFile(target, "w") as output:
            for entry, payload in updated.items():
                output.writestr(entry, payload)
        return target

    configured_metadata = json.loads(configured_members["fit.json"])
    configured_metadata["resolved_fit_config"]["targets"]["energy"] = "other_energy"
    wrong_fit_record = rehashed_copy(
        "wrong_fit_record.ye3t", configured_members, "fit.json",
        json.dumps(configured_metadata).encode())
    with pytest.raises(ValueError, match="configured fit identity changed"):
        LinearModel.read(wrong_fit_record)
    configured_metadata = json.loads(configured_members["fit.json"])
    configured_metadata["resolved_fit_config"]["model"]["fit"]["solver"] = "lasso"
    configured_metadata["resolved_fit_config_sha256"] = hashlib.sha256(json.dumps(
        configured_metadata["resolved_fit_config"], sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    wrong_solver = rehashed_copy(
        "wrong_fit_solver.ye3t", configured_members, "fit.json",
        json.dumps(configured_metadata).encode())
    with pytest.raises(ValueError, match="configured fit semantics differ"):
        LinearModel.read(wrong_solver)

    fit = json.loads(members["fit.json"])
    fit["design_columns"][0]["label_identity"] = "wrong compiler column"
    changed = rehashed_copy("wrong_columns.ye3t", members, "fit.json",
                          json.dumps(fit).encode())
    with pytest.raises(ValueError, match="labels or final offsets changed"):
        LinearModel.read(changed)
    changed = rehashed_copy(
        "nonfinite_json.ye3t", members, "fit.json",
        members["fit.json"].replace(b'"n_cols":', b'"overflow":1e999,"n_cols":'))
    with pytest.raises(ValueError, match="nonfinite number"):
        LinearModel.read(changed)
    altered_manifest = json.loads(members["manifest.json"])
    altered_manifest["arrays"]["weights.npz"]["per_species_E0_eV"]["shape"] = [2]
    altered = dict(members)
    altered["manifest.json"] = json.dumps(altered_manifest).encode()
    wrong_shape = tmp_path / "wrong_inventory.ye3t"
    with zipfile.ZipFile(wrong_shape, "w") as output:
        for entry, payload in altered.items():
            output.writestr(entry, payload)
    with pytest.raises(ValueError, match="array dimensions or dtypes"):
        LinearModel.read(wrong_shape)
    altered_manifest = json.loads(members["manifest.json"])
    altered_manifest["components"][0] = ["invalid", "component"]
    altered = dict(members)
    altered["manifest.json"] = json.dumps(altered_manifest).encode()
    wrong_component = tmp_path / "wrong_component.ye3t"
    with zipfile.ZipFile(wrong_component, "w") as output:
        for entry, payload in altered.items():
            output.writestr(entry, payload)
    with pytest.raises(ValueError, match="component or species manifest"):
        LinearModel.read(wrong_component)
    members["weights.npz"] += b"tampered"
    corrupt = tmp_path / "corrupt.ye3t"
    with zipfile.ZipFile(corrupt, "w") as output:
        for name, data in members.items():
            output.writestr(name, data)
    with pytest.raises(ValueError, match="member hash mismatch"):
        LinearModel.read(corrupt)
    ard = LinearModel(basis).fit(training, fit_method="ard", force_weight=0.0,
                                 fit_E0=True)
    ard_path = ard.write(tmp_path / "combined_ard.ye3t")
    ard_loaded = LinearModel.read(ard_path)
    with zipfile.ZipFile(ard_path) as source:
        ard_members = {name: source.read(name) for name in source.namelist()}
    ard_fit = json.loads(ard_members["fit.json"])
    assert ard_fit["predictive_uncertainty"]["active_column_indices"]
    ard_fit["predictive_uncertainty"]["active_column_indices"][0] = 0.5
    fractional = rehashed_copy("fractional_active.ye3t", ard_members, "fit.json",
                               json.dumps(ard_fit).encode())
    with pytest.raises(ValueError, match="active columns are invalid"):
        LinearModel.read(fractional)
    original_uncertainty = ard.predict_uncertainty(training[0])
    saved_uncertainty = ard_loaded.predict_uncertainty(training[0])
    np.testing.assert_allclose(saved_uncertainty["atomic_energy_std_eV"],
                               original_uncertainty["atomic_energy_std_eV"], atol=1e-12)
    assert saved_uncertainty["total_energy_std_eV"] == pytest.approx(
        original_uncertainty["total_energy_std_eV"], abs=1e-12)


def test_combined_species_offsets_are_applied_once(tmp_path):
    species = ["Cu", "Ni"]
    ordinary = Basis(elements=species, source="density", cutoff=3.0,
                     max_rank=1, nmax=1, lmax=0,
                     descriptor_cache_dir=tmp_path / "density")
    tagged = Basis(
        elements=species, source="tagged_cauchy_image", cutoff=2.8,
        rank=2, tag_counts=(0, 2), nmax_per_rank={2: 1},
        lmax_per_rank={2: 0}, source_block_partitions_by_rank={2: ((2,),)},
        max_records_per_rank=1, max_features_per_rank=4,
        backend="reference", compiled_cache_dir=tmp_path / "tagged",
        compiler_validation="full",
    )
    basis = Basis.combine({"ordinary": ordinary, "tagged": tagged})
    training = []
    for symbol, energy in (("Cu", -0.3), ("Ni", 0.2), ("CuNi", -0.1)):
        atoms = Atoms(symbol, positions=[(5.0 * index, 0, 0)
                                         for index in range(len(Atoms(symbol)))],
                      cell=(16, 16, 16), pbc=False)
        atoms.info["energy"] = energy
        training.append(atoms)
    model = LinearModel(basis, reference_energies={"Cu": -0.05, "Ni": 0.05}).fit(
        training, force_weight=0.0, regularization=1e-8, fit_E0=True)
    collinear = []
    for spacing in (4.0, 4.5):
        atoms = Atoms("CuNi", positions=((0, 0, 0), (spacing, 0, 0)),
                      cell=(16, 16, 16), pbc=False)
        atoms.info["energy"] = -0.1
        collinear.append(atoms)
    with pytest.raises(ValueError, match="full column rank in training compositions"):
        LinearModel(basis).fit(collinear, force_weight=0.0, fit_E0=True)
    assert model._fitted["offsets_eV"] == pytest.approx({"Cu": -0.3, "Ni": 0.2},
                                                        abs=1e-8)
    for atoms in training:
        trial = atoms.copy()
        trial.calc = model.ase_calculator(evaluator="torch")
        assert trial.get_potential_energy() == pytest.approx(atoms.info["energy"], abs=1e-8)
    assert len(model._fitted["fit_metadata"]["design_columns"]) == (
        len(ordinary.labels) + len(species) * len(tagged.labels) + len(species))
    fixed = LinearModel(basis, reference_energies={"Cu": -0.3, "Ni": 0.2}).fit(
        training, force_weight=0.0, regularization=1e-8, fit_E0=False)
    assert fixed._fitted["offsets_eV"] == {"Cu": -0.3, "Ni": 0.2}
    for atoms in training:
        trial = atoms.copy()
        trial.calc = fixed.ase_calculator(evaluator="torch")
        assert trial.get_potential_energy() == pytest.approx(atoms.info["energy"], abs=1e-8)

    interacting = Atoms("CuNi2", positions=((0.3, 0.5, 0.4), (1.7, 0.7, 0.6),
                                           (0.6, 2.0, 0.9)), cell=(6.4, 6.5, 6.6), pbc=True)
    ordinary_beta = np.full(len(ordinary.labels), 0.17)
    cu_beta = np.linspace(0.11, 0.21, len(tagged.labels))
    ni_beta = np.linspace(-0.19, -0.09, len(tagged.labels))
    coefficient = np.concatenate((ordinary_beta, cu_beta, ni_beta,
                                  np.array((-0.3, 0.2))))

    def energy(atoms):
        ordinary_rows = ordinary.create(atoms)
        tagged_rows = tagged.create(atoms)
        symbols = np.asarray(atoms.get_chemical_symbols())
        return float(ordinary_rows.sum(axis=0) @ ordinary_beta +
                     tagged_rows[symbols == "Cu"].sum(axis=0) @ cu_beta +
                     tagged_rows[symbols == "Ni"].sum(axis=0) @ ni_beta +
                     np.count_nonzero(symbols == "Cu") * -0.3 +
                     np.count_nonzero(symbols == "Ni") * 0.2)

    energy_row, force_rows, stress_rows = _combined_scalar_geometry_row(
        basis, interacting, True, True)
    assert energy_row @ coefficient == pytest.approx(energy(interacting), abs=1e-10)
    reordered = interacting[[2, 0, 1]]
    reordered_energy, reordered_force, reordered_stress = _combined_scalar_geometry_row(
        basis, reordered, True, True)
    assert reordered_energy @ coefficient == pytest.approx(energy(interacting), abs=1e-10)
    np.testing.assert_allclose(reordered_force @ coefficient,
                               (force_rows @ coefficient).reshape(-1, 3)[[2, 0, 1]].reshape(-1),
                               atol=1e-9)
    np.testing.assert_allclose(reordered_stress @ coefficient,
                               stress_rows @ coefficient, atol=1e-9)
    step = 2e-5
    for atom in range(len(interacting)):
        for axis in range(3):
            minus, plus = interacting.copy(), interacting.copy()
            minus.positions[atom, axis] -= step
            plus.positions[atom, axis] += step
            independent = -(energy(plus) - energy(minus)) / (2 * step)
            assert force_rows[3 * atom + axis] @ coefficient == pytest.approx(
                independent, abs=3e-5)
    for index, (a, b) in enumerate(((0, 0), (1, 1), (2, 2),
                                    (1, 2), (0, 2), (0, 1))):
        values = []
        for sign in (-1, 1):
            deformation = np.eye(3)
            deformation[a, b] += sign * step
            if a != b:
                deformation[b, a] += sign * step
            trial = interacting.copy()
            trial.positions[:] = interacting.positions @ deformation
            trial.set_cell(interacting.cell.array @ deformation, scale_atoms=False)
            values.append(energy(trial))
        independent = ((values[1] - values[0]) / (2 * step) /
                       interacting.get_volume() / (2 if a != b else 1))
        assert stress_rows[index] @ coefficient == pytest.approx(independent, abs=3e-5)


def test_combined_ard_uncertainty_uses_component_site_columns(tmp_path):
    pytest.importorskip("sklearn")
    ordinary = Basis(elements=["Ni"], source="density", cutoff=3.3,
                     max_rank=1, nmax=1, lmax=0,
                     descriptor_cache_dir=tmp_path / "density")
    tagged = Basis(
        elements=["Ni"], source="tagged_cauchy_image", cutoff=3.0,
        rank=2, tag_counts=(0, 2), nmax_per_rank={2: 1},
        lmax_per_rank={2: 0}, source_block_partitions_by_rank={2: ((2,),)},
        max_records_per_rank=1, max_features_per_rank=4,
        backend="reference", compiled_cache_dir=tmp_path / "tagged",
        compiler_validation="full",
    )
    basis = Basis.combine({"ordinary": ordinary, "tagged": tagged})
    beta = np.linspace(0.07, 0.17, len(basis.labels))
    training = []
    for distance in (1.2, 1.4, 1.6, 1.8, 2.0, 2.2):
        atoms = Atoms("Ni3", positions=((0, 0, 0), (distance, 0.2, 0.1),
                                        (0.1, 1.7, 0.3)))
        atoms.info["energy"] = float(basis.create(atoms).sum(axis=0) @ beta + 0.3)
        training.append(atoms)
    for fit_offsets, reference in ((False, {"Ni": 0.1}), (True, {})):
        model = LinearModel(basis, reference_energies=reference).fit(
            training, fit_method="ard", force_weight=0.0, fit_E0=fit_offsets)
        report = model.predict_uncertainty(training[0])
        assert report["atomic_energy_std_eV"].shape == (3,)
        assert np.isfinite(report["atomic_energy_std_eV"]).all()
        assert np.isfinite(report["total_energy_std_eV"])
        metadata = model._fitted["fit_metadata"]
        posterior = metadata["predictive_uncertainty"]
        sites = basis.create(training[0])
        design = np.column_stack((sites, np.ones(3))) if fit_offsets else sites
        assert design.shape[1] == len(metadata["design_columns"])
        active = posterior["active_column_indices"]
        covariance = np.asarray(posterior["coefficient_covariance_active"])
        direct = np.einsum("if,fg,ig->i", design[:, active], covariance,
                           design[:, active])
        np.testing.assert_allclose(report["atomic_energy_std_eV"] ** 2,
                                   np.maximum(direct, 0.0), atol=1e-12)
