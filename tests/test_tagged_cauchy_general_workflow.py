"""General chemical image, explicit derivatives and fixed-reference fitting."""

import itertools
import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ye3t_methods.atomistic.ace.descriptors import YE3TDescriptors, YE3TModel
from ye3t_methods.atomistic.representations import YE3TRepresentation
from ye3t_methods.atomistic.reference_potentials import (
    evaluate_zbl_reference, evaluate_lammps_zbl_reference, lammps_zbl_reference_config,
)
from ye3t_methods.atomistic.tagged_cauchy_image import (
    TaggedCauchyImageEvaluator, TaggedCauchyImageLinearModel,
    export_tagged_cauchy_image_model, load_tagged_cauchy_image_model,
)
from ye3t_methods.atomistic.tagged_cauchy_image_fit import (
    tagged_cauchy_image_geometry_row, score_tagged_cauchy_image_model,
)


@pytest.fixture(scope="module")
def general_descriptor(tmp_path_factory):
    representation = YE3TRepresentation.tagged_cauchy_image()
    return YE3TDescriptors.ye3t_basis({
        "representation": representation, "elements": ["H", "O"], "backend": "auto",
        "tagged_cauchy_image": {
            "cutoff_A": 3.8,
            "compiled_cache_dir": tmp_path_factory.mktemp("compiled_tagged_image"),
            "pair_cutoffs_A": {"H-H": 2.7, "H-O": 3.2, "O-H": 3.2, "O-O": 3.8},
            "catalogue": {
                "nmax_per_rank": {1: 2, 2: 1}, "lmax_per_rank": {1: 0, 2: 1},
                "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]]},
                "angular_patterns_by_rank": {2: [[1, 1]]},
                "tag_counts_by_rank": {1: [0], 2: [0, 1, 2]},
                "max_records_per_rank": 8, "max_features_per_rank": {1: 4, 2: 4},
            },
        },
    })


def test_general_compiler_cache_round_trip(general_descriptor):
    payload = general_descriptor.metadata["tagged_cauchy_image_config"]
    cached = YE3TDescriptors.ye3t_basis({
        "representation": YE3TRepresentation.tagged_cauchy_image(),
        "elements": ["H", "O"], "backend": "auto", "tagged_cauchy_image": payload})
    assert cached.metadata["tagged_cauchy_image_compiled"].self_hash == general_descriptor.metadata[
        "tagged_cauchy_image_compiled"].self_hash
    assert len(list(payload["compiled_cache_dir"].glob("*.json"))) == 1
    assert not list(payload["compiled_cache_dir"].glob("*.tmp"))


def test_homogeneous_space_config_and_linear_fit(general_descriptor):
    tagged = general_descriptor.metadata["tagged_cauchy_image_config"]
    cfg_ye3t = {
        "metadata": {"status": "experimental"},
        "basis": {"type": "tagged_cauchy_image", "species": ["H", "O"], **tagged},
        "representation": {"mode": "tagged_cauchy_image", "carrier": "A_s",
                           "target": {"permutation": "trivial", "L": 0}},
        "runtime": {"backend": "auto"},
        "model": {"type": "linear", "ridge_alpha": 1e-8,
                  "energy_weight": 1.0, "force_weight": 1.0},
        "targets": {"energy_key": "energy", "force_key": "forces"},
        "validation": {"checks": []},
    }
    ye3t_space = YE3TDescriptors.ye3t_basis(cfg_ye3t)
    assert ye3t_space.metadata["tagged_cauchy_image_compiled"].self_hash == (
        general_descriptor.metadata["tagged_cauchy_image_compiled"].self_hash)
    evaluator = ye3t_space.metadata["tagged_cauchy_image_evaluator"]
    oracle = TaggedCauchyImageLinearModel(
        evaluator, {name: np.linspace(0.001, 0.002, evaluator.feature_count)
                    for name in evaluator.species_order}, {"H": 0.1, "O": -0.2})
    structures = []
    for scale in (0.9, 1.0, 1.1):
        atoms = Atoms("HOH", positions=scale*np.array(((0, 0, 0), (0.9, 0.1, 0),
                                                      (0.3, 1.0, 0.2))))
        energy, forces, _, _ = oracle.energy_forces_virial(atoms.positions, [0, 1, 0])
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=forces.numpy())
        structures.append(atoms)
    assert ye3t_space.create(structures[0]).shape == (3, evaluator.feature_count)
    matrix = ye3t_space.training_matrix(structures, properties=("energy", "forces"))
    assert matrix["matrix"].shape == (30, 2*evaluator.feature_count + 2)
    fitted = YE3TModel.linear(ye3t_space, cfg_ye3t, structures=structures)
    assert np.isfinite(float(fitted.energy_forces_virial(
        structures[0].positions, [0, 1, 0])[0]))


def test_certificate_policy_propagates_through_public_workflow(general_descriptor, tmp_path, monkeypatch):
    import ye3t.couplings.tagged_cauchy_general as general

    def no_replay(*args, **kwargs):
        raise AssertionError("unexpected representation reconstruction")

    monkeypatch.setattr(general, "_general_count", no_replay)
    payload = {**general_descriptor.metadata["tagged_cauchy_image_config"],
               "compiler_validation": "certificate"}
    cached = YE3TDescriptors.ye3t_basis({
        "representation": YE3TRepresentation.tagged_cauchy_image(),
        "elements": ["H", "O"], "backend": "auto", "tagged_cauchy_image": payload})
    evaluator = cached.metadata["tagged_cauchy_image_evaluator"]
    model = TaggedCauchyImageLinearModel(evaluator,
        {s: np.arange(evaluator.feature_count)*0.001 for s in evaluator.species_order},
        {s: 0.0 for s in evaluator.species_order})
    export_tagged_cauchy_image_model(tmp_path/"model.json", model)
    loaded = load_tagged_cauchy_image_model(tmp_path/"model.json", compiler_validation="certificate")
    assert loaded.evaluator.execution_report["compiler_validation"] == "certificate"
    positions = np.array([[0, 0, 0], [0.8, 0.3, 0.4]])
    for actual, expected in zip(loaded.energy_forces_virial(positions, [0, 1]),
                                model.energy_forces_virial(positions, [0, 1])):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_general_native_force_rows_and_invariance(general_descriptor, tmp_path):
    descriptor = general_descriptor
    evaluator = descriptor.metadata["tagged_cauchy_image_evaluator"]
    reference = TaggedCauchyImageEvaluator(evaluator.compiled, evaluator.source_plan,
                                          evaluator.program, backend="reference")
    beta = {species: np.linspace(0.001, 0.003, evaluator.feature_count)
            for species in evaluator.species_order}
    model = TaggedCauchyImageLinearModel(evaluator, beta, {"H": 0.0, "O": 0.0})
    oracle = TaggedCauchyImageLinearModel(reference, beta, model.offsets)
    atoms = Atoms("HOH", positions=[[0.1, 0.2, 0.3], [1.0, -0.2, 0.4], [0.3, 1.1, -0.4]])
    types = [evaluator.type_map[s] for s in atoms.get_chemical_symbols()]
    result = model.energy_forces_virial(atoms.positions, types)
    expected = oracle.energy_forces_virial(atoms.positions, types)
    for actual, value in zip(result, expected):
        np.testing.assert_allclose(actual, value, rtol=2e-12, atol=2e-12)
    step = 2e-6
    for atom, axis in itertools.product(range(3), range(3)):
        plus, minus = atoms.positions.copy(), atoms.positions.copy()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        derivative = (model.energy_forces_virial(plus, types)[0]
                      - model.energy_forces_virial(minus, types)[0])/(2*step)
        assert float(result[1][atom, axis]) == pytest.approx(-float(derivative), abs=1e-8)
    rotation = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
    for transform in (rotation, -rotation):
        moved = model.energy_forces_virial(atoms.positions @ transform.T + 2.3, types)
        np.testing.assert_allclose(moved[0], result[0], atol=2e-12)
        np.testing.assert_allclose(moved[1], result[1] @ transform.T, atol=2e-11)
    permutation = [2, 0, 1]
    moved = model.energy_forces_virial(atoms.positions[permutation], np.array(types)[permutation])
    np.testing.assert_allclose(moved[0], result[0], atol=2e-12)
    np.testing.assert_allclose(moved[1], result[1][permutation], atol=2e-11)
    row = tagged_cauchy_image_geometry_row(descriptor, atoms, tmp_path)
    coefficients = np.concatenate(list(beta.values()))
    np.testing.assert_allclose(row["force_design"] @ coefficients, result[1].reshape(-1), atol=2e-12)
    assert tagged_cauchy_image_geometry_row(descriptor, atoms, tmp_path)["cache_hit"]


def test_general_v4_native_ase_auto_matches_reference(general_descriptor):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    evaluator = general_descriptor.metadata["tagged_cauchy_image_evaluator"]
    model = TaggedCauchyImageLinearModel(
        evaluator,
        {name: np.linspace(0.001, 0.003, evaluator.feature_count)
         for name in evaluator.species_order},
        {"H": -0.1, "O": 0.2},
    )
    atoms = Atoms("HOH", positions=((0.1, 0.2, 0.3), (1.0, -0.2, 0.4),
                                    (0.3, 1.1, -0.4)), cell=(8.0, 8.0, 8.0), pbc=True)
    reference = atoms.copy()
    reference.calc = model.ase_calculator(backend="reference")
    native = atoms.copy()
    native.calc = model.ase_calculator(
        backend="native_cpu", native_library=library, execution_policy="auto")
    assert native.calc.native_runtime.selected_policy in {
        "compiled_direct", "generic_dag", "symmetric_power", "block",
    }
    np.testing.assert_allclose(native.get_potential_energy(), reference.get_potential_energy(),
                               rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(native.get_forces(), reference.get_forces(),
                               rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(native.get_stress(), reference.get_stress(),
                               rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(
        general_descriptor.create(atoms, backend="native_cpu", native_library=library,
                                  execution_policy="auto"),
        general_descriptor.create(atoms), rtol=2e-8, atol=2e-8,
    )


def test_fixed_references_fit_and_reload(general_descriptor, tmp_path):
    evaluator = general_descriptor.metadata["tagged_cauchy_image_evaluator"]
    oracle = TaggedCauchyImageLinearModel(evaluator,
        {s: np.linspace(0.001, 0.002, evaluator.feature_count) for s in evaluator.species_order},
        {"H": 0.0, "O": 0.0})
    refs = {"H": -13.0, "O": -2000.0}
    zbl = {"engine": "numpy", "inner_cutoff_A": 0.7, "outer_cutoff_A": 1.8,
           "atomic_numbers": {"H": 1, "O": 8}}
    oracle.reference_terms = {"atomic_energies": refs, "zbl": zbl}
    structures = []
    for scale in np.linspace(0.8, 1.4, 8):
        atoms = Atoms("HOH", positions=scale*np.array([[0, 0, 0], [0.9, 0.1, 0], [0.3, 1.0, 0.2]]))
        energy, forces, _, _ = oracle.energy_forces_virial(atoms.positions, [0, 1, 0])
        atoms.calc = SinglePointCalculator(atoms, energy=float(energy), forces=forces.numpy())
        structures.append(atoms)
    baseline = evaluate_zbl_reference(structures, zbl)
    fitted = YE3TModel.linear(general_descriptor, structures=structures, model_config={
        "fit_intercept": False, "restore_references": True, "reference_energies": refs,
        "reference_potential_metadata": baseline["metadata"],
        "target_energies": np.array([a.get_potential_energy() for a in structures])-baseline["reference_energies"],
        "target_forces": [a.get_forces()-f for a, f in zip(structures, baseline["reference_forces"])],
        "ridge_alpha": 1e-12, "geometry_cache_dir": tmp_path/"rows",
    })
    assert fitted.offsets == {"H": 0.0, "O": 0.0}
    assert fitted.fit_metadata["coefficient_count"] == 2*evaluator.feature_count
    direct = score_tagged_cauchy_image_model(fitted, structures)
    cached = score_tagged_cauchy_image_model(fitted, structures, descriptor=general_descriptor,
                                            geometry_cache_dir=tmp_path/"rows")
    assert direct["force_rmse_eV_per_A"] < 1e-6
    np.testing.assert_allclose(list(direct.values()), list(cached.values()), atol=1e-11)
    artifact = export_tagged_cauchy_image_model(tmp_path/"model.json", fitted)
    assert artifact["schema"] == "ye3t_tagged_cauchy_slice_v4"
    restored = load_tagged_cauchy_image_model(tmp_path/"model.json")
    for actual, expected in zip(restored.energy_forces_virial(structures[0].positions, [0, 1, 0]),
                                fitted.energy_forces_virial(structures[0].positions, [0, 1, 0])):
        np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_general_periodic_reference_and_strain_derivatives(general_descriptor):
    evaluator = general_descriptor.metadata["tagged_cauchy_image_evaluator"]
    reference = TaggedCauchyImageEvaluator(evaluator.compiled, evaluator.source_plan,
                                          evaluator.program, backend="reference")
    beta = {species: np.linspace(0.001, 0.003, evaluator.feature_count)
            for species in evaluator.species_order}
    models = [TaggedCauchyImageLinearModel(runtime, beta, {"H": 0.0, "O": 0.0})
              for runtime in (evaluator, reference)]
    for model in models:
        model.reference_terms = {"atomic_energies": {"H": -13.0, "O": -2000.0},
            "zbl": {"atomic_numbers": {"H": 1, "O": 8},
                    "inner_cutoff_A": 0.7, "outer_cutoff_A": 1.8}}
    positions = np.array([[0.1, 0.2, 0.3], [4.2, 0.4, 0.7], [0.6, 1.2, 0.4]])
    cell = np.array([[5.0, 0.0, 0.0], [0.3, 5.3, 0.0], [0.2, -0.1, 5.1]])
    types = [evaluator.type_map[s] for s in ("H", "O", "H")]
    model = models[0]
    actual = model.energy_forces_virial(positions, types, cell=cell, pbc=True)
    expected = models[1].energy_forces_virial(positions, types, cell=cell, pbc=True)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-12)
    shifted = positions.copy()
    shifted[1] -= cell[0]
    for left, right in zip(model.energy_forces_virial(shifted, types, cell=cell, pbc=True), actual):
        np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-11)
    step = 2e-6
    # Runtime order xx,yy,zz,xy,xz,yz; future SNAP adapter must reorder it.
    for component, (row, column) in enumerate(((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1))):
        strain = np.zeros((3, 3))
        strain[row, column] = step
        plus, minus = np.eye(3)+strain, np.eye(3)-strain
        energies = [model.energy_forces_virial(positions @ transform.T, types,
            cell=cell @ transform.T, pbc=True)[0] for transform in (plus, minus)]
        numeric = -float(energies[0]-energies[1])/(2*step)
        assert float(actual[2][component]) == pytest.approx(numeric, rel=1e-7, abs=2e-7)


def test_portable_zbl_force_and_cutoff():
    config = {"atomic_numbers": {"K": 19, "O": 8}, "inner_cutoff_A": 0.8, "outer_cutoff_A": 2.0}
    for distance in (0.4, 0.8, 1.2, 1.99, 2.0, 2.1):
        atoms = Atoms("KO", positions=[[0, 0, 0], [distance, 0, 0]])
        result = evaluate_zbl_reference([atoms], config)
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[1, 0] += 1e-6
        minus.positions[1, 0] -= 1e-6
        numeric = -(evaluate_zbl_reference([plus], config)["reference_energies"][0]
                    -evaluate_zbl_reference([minus], config)["reference_energies"][0])/(2e-6)
        np.testing.assert_allclose(result["reference_forces"][0][1, 0], numeric, rtol=3e-8, atol=1e-6)
        if distance >= 2.0:
            assert result["reference_energies"][0] == 0.0


def test_portable_zbl_matches_lammps():
    executable = os.environ.get("YE3T_TEST_LAMMPS_EXECUTABLE")
    if not executable:
        pytest.skip("requires optional LAMMPS executable via YE3T_TEST_LAMMPS_EXECUTABLE")
    config = {"atomic_numbers": {"H": 1, "K": 19, "O": 8, "S": 16},
              "inner_cutoff_A": 0.8, "outer_cutoff_A": 2.0, "executable": executable}
    structures = []
    for left, right in itertools.combinations_with_replacement(config["atomic_numbers"], 2):
        for distance in (0.5, 1.1, 1.9, 2.1):
            atoms = Atoms([left, right], positions=[[0, 0, 0], [distance, 0, 0]])
            atoms.center(vacuum=5.0)
            structures.append(atoms)
    expected = evaluate_lammps_zbl_reference(structures, lammps_zbl_reference_config(config, sorted(config["atomic_numbers"])))
    actual = evaluate_zbl_reference(structures, config)
    np.testing.assert_allclose(actual["reference_energies"], expected["reference_energies"], rtol=2e-12, atol=2e-10)
    np.testing.assert_allclose(actual["reference_forces"], expected["reference_forces"], rtol=2e-12, atol=2e-10)


def test_pair_specific_zbl_switches_and_validation():
    config = {"atomic_numbers": {"H": 1, "O": 8}, "pair_cutoffs_A": {
        "H-H": [0.31, 0.465], "H-O": [0.485, 0.7275], "O-O": [0.66, 0.99]}}
    for pair, (inner, outer) in config["pair_cutoffs_A"].items():
        for radius in (inner*0.8, inner, (inner+outer)/2, outer, outer+0.1):
            atoms = Atoms(pair.split("-"), positions=[[0, 0, 0], [radius, 0, 0]])
            actual = evaluate_zbl_reference([atoms], config)
            expected = evaluate_zbl_reference([atoms], {"atomic_numbers": config["atomic_numbers"],
                "inner_cutoff_A": inner, "outer_cutoff_A": outer})
            np.testing.assert_allclose(actual["reference_energies"], expected["reference_energies"], atol=1e-12)
            np.testing.assert_allclose(actual["reference_forces"], expected["reference_forces"], atol=1e-12)
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[1, 0] += 1e-6
            minus.positions[1, 0] -= 1e-6
            numerical = -(evaluate_zbl_reference([plus], config)["reference_energies"][0]
                          -evaluate_zbl_reference([minus], config)["reference_energies"][0])/(2e-6)
            np.testing.assert_allclose(actual["reference_forces"][0][1, 0], numerical, rtol=1e-8, atol=1e-6)
    with pytest.raises(ValueError, match="every unordered"):
        evaluate_zbl_reference([atoms], {**config, "pair_cutoffs_A": {"H-H": [0.3, 0.5]}})
    with pytest.raises(ValueError, match="symmetric"):
        evaluate_zbl_reference([atoms], {**config, "pair_cutoffs_A": {**config["pair_cutoffs_A"], "O-H": [0.5, 0.9]}})


def test_pair_specific_zbl_matches_lammps():
    executable = os.environ.get("YE3T_TEST_LAMMPS_EXECUTABLE")
    if not executable:
        pytest.skip("requires optional LAMMPS executable via YE3T_TEST_LAMMPS_EXECUTABLE")
    config = {"atomic_numbers": {"H": 1, "O": 8, "K": 19, "S": 16},
              "executable": executable, "pair_cutoffs_A": {}}
    radii = {"H": 0.31, "O": 0.66, "K": 2.03, "S": 1.05}
    structures = []
    for left, right in itertools.combinations_with_replacement(radii, 2):
        total = radii[left]+radii[right]
        config["pair_cutoffs_A"][left+"-"+right] = [0.5*total, 0.75*total]
        for fraction in (0.4, 0.5, 0.65, 0.75, 0.8):
            atoms = Atoms([left, right], positions=[[0, 0, 0], [fraction*total, 0, 0]])
            atoms.center(vacuum=5.0)
            structures.append(atoms)
    expected = evaluate_lammps_zbl_reference(structures, lammps_zbl_reference_config(config, sorted(radii)))
    actual = evaluate_zbl_reference(structures, config)
    np.testing.assert_allclose(actual["reference_energies"], expected["reference_energies"], rtol=2e-12, atol=2e-10)
    np.testing.assert_allclose(actual["reference_forces"], expected["reference_forces"], rtol=2e-12, atol=2e-10)


def test_general_native_deployment_with_bound_pair_references(general_descriptor, tmp_path):
    executable = os.environ.get("YE3T_TEST_NATIVE_TAGGED")
    if not executable:
        pytest.skip("requires optional native tagged CPU fixture executable via YE3T_TEST_NATIVE_TAGGED")
    evaluator = general_descriptor.metadata["tagged_cauchy_image_evaluator"]
    beta = {s: torch.linspace(0.001, 0.003, evaluator.feature_count, dtype=torch.float64)
            for s in evaluator.species_order}
    model = TaggedCauchyImageLinearModel(evaluator, beta, {s: 0.0 for s in evaluator.species_order})
    zbl = {"atomic_numbers": {"H": 1, "O": 8}, "pair_cutoffs_A": {
        "H-H": [0.31, 0.465], "H-O": [0.485, 0.7275], "O-O": [0.66, 0.99]}}
    refs = {"H": -13.0, "O": -2000.0}
    model.reference_terms = {"atomic_energies": refs, "zbl": zbl}
    artifact = export_tagged_cauchy_image_model(tmp_path/"model.json", model)
    fixtures = []
    for center in evaluator.species_order:
        neighbors = ["H", "O", "H", "O"]
        displacement = torch.tensor([[0.4, 0.03, 0.02], [-0.55, 0.07, 0.1],
                                     [2.8, 0, 0], [0.1, 3.3, 0]], dtype=torch.float64)
        types = [evaluator.type_map[s] for s in [center]+neighbors]
        edges = torch.tensor([[0]*4, [1, 2, 3, 4]])
        features, derivatives = evaluator.evaluate_edge_list(edges, displacement, types[1:], 5, atom_types=types)
        energy = refs[center]+float(features[0] @ beta[center])
        gradient = torch.einsum("f,efd->ed", beta[center], derivatives).numpy()
        for edge, neighbor in enumerate(neighbors):
            pair = Atoms([center, neighbor], positions=np.vstack((np.zeros(3), displacement[edge].numpy())))
            reference = evaluate_zbl_reference([pair], zbl)
            energy += 0.5*reference["reference_energies"][0]
            gradient[edge] -= 0.5*reference["reference_forces"][0][1]
        fixture = {"model_path": "model.json", "model_self_hash": artifact["self_hash"],
            "central_species": center, "energy_eV": energy,
            "edges": [{"neighbor_species": species, "displacement_A": row}
                      for species, row in zip(neighbors, displacement.tolist())],
            "edge_gradients_dE_dd_eV_per_A": gradient.tolist()}
        path = tmp_path/(center+".fixture.json")
        path.write_text(json.dumps(fixture))
        fixtures.append(str(path))
    result = subprocess.run([executable, *fixtures], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    # Recompute transport hashes: these must fail semantic validation, not just
    # detection of stale outer JSON checksums.
    from ye3t_methods.atomistic.tagged_cauchy_image import _binding, _deployment_identity, _payload_hash
    for fault, message in (("zbl_hash", "semantic hash"),
                           ("zbl_switch", "unsupported ZBL convention"),
                           ("missing_atomic_reference", "expected every species"),
                           ("unused_compiler_evidence", "compiler artifact self-hash mismatch")):
        corrupted = copy.deepcopy(artifact)
        readout = corrupted["readout_binding"]["payload"]
        if fault == "zbl_hash":
            readout["reference_terms"]["zbl"]["semantic_sha256"] = "0"*64
        elif fault == "zbl_switch":
            readout["reference_terms"]["zbl"]["switch"] = "unrecognized"
        elif fault == "missing_atomic_reference":
            del readout["reference_terms"]["atomic_energies"]["H"]
        else:
            # The native loader need not create a DOM for offline proof data,
            # but its checksum must still bind every original proof byte.
            corrupted["compiler_artifact"]["payload"]["certificate"]["independence_argument"] = "corrupt"
        corrupted["readout_binding"] = _binding(readout)
        corrupted["deployment_identity_hash"] = _payload_hash(_deployment_identity(corrupted))
        corrupted.pop("self_hash")
        corrupted["self_hash"] = _payload_hash(corrupted)
        (tmp_path/"model.json").write_text(json.dumps(corrupted, sort_keys=True))
        python_message = {"zbl_hash": "semantic hash", "zbl_switch": "Unsupported portable ZBL",
                          "missing_atomic_reference": "atomic references",
                          "unused_compiler_evidence": "artifact hash mismatch"}[fault]
        for policy in ("full", "certificate"):
            with pytest.raises(ValueError, match=python_message):
                load_tagged_cauchy_image_model(tmp_path/"model.json", compiler_validation=policy)
        invalid_fixture = json.loads(Path(fixtures[0]).read_text())
        invalid_fixture["model_self_hash"] = corrupted["self_hash"]
        Path(fixtures[0]).write_text(json.dumps(invalid_fixture))
        rejected = subprocess.run([executable, fixtures[0]], capture_output=True, text=True, timeout=120)
        assert rejected.returncode != 0
        assert message in rejected.stdout+rejected.stderr
