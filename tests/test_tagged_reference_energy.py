"""Configured fixed/fitted atomic energies use the same scalar tagged design."""
import copy
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pytest
from ase import Atoms

from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel
from ye3t_methods.atomistic.tagged_cauchy_image_fit import tagged_cauchy_image_geometry_row


@pytest.fixture(scope="module")
def tagged_reference_problem():
    config = {
        "metadata": {"schema": "ye3t_config_v1", "name": "reference_energy_test", "status": "stable"},
        "representation": {
            "group": "O3", "ranks": [1, 2],
            "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
            "factorization": "cauchy", "subspace": "full",
            "uncoupled_factor_inputs": {"eta_count_per_rank": {1: 2, 2: 2},
                                        "l_max_per_rank": {1: 0, 2: 1}},
            "intermediates": {"young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"}}},
        "basis": {
            "single_factors": {"species": ["Ni"],
                "radial": {"family": "shifted_jacobi", "cutoff_A": 4.8},
                "chemical": {"kind": "explicit"}},
            "tensor_product": {"kind": "tagged", "tag_counts_per_rank": {1: [1], 2: [2]}},
            "catalogue": {"ranks": [1, 2], "nmax_per_rank": {1: 2, 2: 2},
                "lmax_per_rank": {1: 0, 2: 1},
                "source_block_partitions_by_rank": {1: [[1]], 2: [[2], [1, 1]]}}},
        "runtime": {"evaluator": "reference", "neighbors": "auto",
                    "cache": {"mode": "auto"}, "dtype": "float64", "device": "cpu"},
        "model": {"kind": "linear", "fit": {"solver": "ridge", "alpha": 1e-12,
                  "weights": {"energy": 1., "forces": 1., "stress": 1.}},
                  "reference_energy": {"per_species_E0_eV": {"Ni": -.3}, "fit_E0": False}},
        "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
        "validation": {"checks": []}}
    rep = YE3TRepresentation.from_config(config["representation"])
    basis = Basis.from_config(config["basis"], representation=rep, runtime=config["runtime"])
    basis.create(Atoms("Ni2", positions=[[0,0,0], [2,.2,.1]], cell=[8,8,8], pbc=True))
    beta = np.linspace(-.04, .06, len(basis.labels))
    rng = np.random.default_rng(18)
    frames = []
    for index in range(10):
        atoms = Atoms("Ni4", positions=np.asarray([[0,0,0], [1.8,.2,.1], [.2,2,.3], [.1,.3,2.1]])
                      + rng.normal(0,.15,(4,3)), cell=[6+index*.1]*3, pbc=True)
        row = tagged_cauchy_image_geometry_row(basis._descriptor, atoms)
        atoms.info["energy"] = float(row["feature_sums"].reshape(-1) @ beta - .3*len(atoms))
        atoms.arrays["forces"] = (row["force_design"] @ beta).reshape(-1,3)
        atoms.info["stress"] = row["stress_design"] @ beta
        frames.append(atoms)
    return config, basis, frames


@pytest.mark.parametrize("solver", ["ridge", "ols"])
@pytest.mark.parametrize("fit_E0", [False, True])
def test_fixed_and_fitted_tagged_reference_energy(tagged_reference_problem, solver, fit_E0, tmp_path):
    if solver == "ols":
        pytest.importorskip("sklearn", reason="requires optional scikit-learn fitting extra")
    config, basis, frames = tagged_reference_problem
    config = copy.deepcopy(config)
    config["model"]["reference_energy"] = {"per_species_E0_eV": {"Ni": -.1 if fit_E0 else -.3},
                                             "fit_E0": fit_E0}
    if solver == "ridge":
        model = LinearModel(basis).fit(frames, config=config)
    else:
        model = LinearModel(basis, reference_energies=config["model"]["reference_energy"]["per_species_E0_eV"]).fit(
            frames, fit_method="ols", fit_E0=fit_E0, stress_weight=1.)
    assert model._fitted.fit_metadata["fit_intercept"] is fit_E0
    np.testing.assert_allclose(model._fitted.offsets["Ni"], -.2 if fit_E0 else 0., atol=1e-7)
    restored = LinearModel.read(model.write(tmp_path/"model.ye3t.json"))
    for fitted in (model, restored):
        for atoms in frames[::3]:
            probe = atoms.copy()
            probe.calc = fitted.ase_calculator(evaluator="torch")
            np.testing.assert_allclose(probe.get_potential_energy(), atoms.info["energy"], atol=2e-7)
            np.testing.assert_allclose(probe.get_forces(), atoms.arrays["forces"], atol=2e-7)
            np.testing.assert_allclose(probe.get_stress(), atoms.info["stress"], atol=2e-8)


def test_tagged_fitted_offsets_need_energy_targets(tagged_reference_problem):
    config, basis, frames = tagged_reference_problem
    config = copy.deepcopy(config)
    config["model"]["fit"]["weights"]["energy"] = 0.
    config["model"]["reference_energy"]["fit_E0"] = True
    with pytest.raises(ValueError, match="positive energy_weight"):
        LinearModel(basis).fit(frames, config=config)


@pytest.mark.parametrize("fit_E0", [False, True])
@pytest.mark.parametrize("scaling", ["none", "rms"])
def test_configured_tagged_ard_reference_columns_and_uncertainty(tagged_reference_problem, fit_E0, scaling, tmp_path):
    pytest.importorskip("sklearn", reason="requires optional scikit-learn fitting extra")
    config, basis, frames = tagged_reference_problem
    config = copy.deepcopy(config)
    config["model"]["fit"] = {"solver": "ard", "weights": {"energy": 1., "forces": 1., "stress": 1.}}
    config["model"]["fit"]["solver_options"] = {"column_scaling": scaling}
    config["model"]["reference_energy"]["fit_E0"] = fit_E0
    model = LinearModel(basis).fit(frames, config=config)
    assert model._fitted.fit_metadata["fit_intercept"] is fit_E0
    if not fit_E0:
        assert model._fitted.offsets == {"Ni": 0.}
    expected = model.predict_uncertainty(frames[0])
    assert np.isfinite(expected["atomic_energy_std_eV"]).all()
    restored = LinearModel.read(model.write(tmp_path/"ard.ye3t.json"))
    actual = restored.predict_uncertainty(frames[0])
    np.testing.assert_allclose(actual["atomic_energy_std_eV"], expected["atomic_energy_std_eV"], atol=1e-12)
    np.testing.assert_allclose(actual["total_energy_std_eV"], expected["total_energy_std_eV"], atol=1e-12)


def test_saved_export_example_preserves_model_and_auto_metadata(tagged_reference_problem):
    config, basis, frames = tagged_reference_problem
    model = LinearModel(basis).fit(frames, config=config)
    output = Path(os.environ["YE3T_WORKFLOW_ROOT"])/"quickstart_linear"
    output.mkdir(parents=True)
    model.write(output/"ni_tagged.ye3t.json")
    path = Path(__file__).resolve().parents[1]/"examples/quickstart/saved_ase_export.py"
    result = runpy.run_path(str(path))
    artifact = Path(result["artifact"])
    payload = json.loads(artifact.read_text())
    assert payload["compiler_artifact_hash"]
    assert payload["source_binding"]["hash"]
    assert payload["schedule_binding"]["hash"]
    assert payload["tagged_execution_portfolio"]["certificate"]["passed"]
    assert (artifact.parent/"atoms.data").is_file()
    assert "block_policy auto" in (artifact.parent/"in.ye3t").read_text()
    restored = LinearModel.read(artifact)
    probe = result["atoms"].copy()
    probe.calc = model.ase_calculator(evaluator="torch")
    expected = (probe.get_potential_energy(), probe.get_forces(), probe.get_stress())
    for policy in ("direct", "auto"):
        probe.calc = restored.ase_calculator(evaluator="native_cpu", neighbors="ase", execution_policy=policy)
        actual = (probe.get_potential_energy(), probe.get_forces(), probe.get_stress())
        for left, right in zip(actual, expected):
            np.testing.assert_allclose(left, right, atol=2e-9, rtol=1e-10)


def test_json_reader_accepts_exact_bound_and_rejects_one_extra_byte(tagged_reference_problem, tmp_path, monkeypatch):
    from ye3t_methods import linear
    config, basis, frames = tagged_reference_problem
    model = LinearModel(basis).fit(frames, config=config)
    path = model.write(tmp_path/'bounded.ye3t.json')
    payload = path.read_bytes()
    assert linear._JSON_MODEL_MAX_BYTES == 512*1024*1024
    monkeypatch.setattr(linear, '_JSON_MODEL_MAX_BYTES', len(payload))
    restored = LinearModel.read(path)
    assert [label.identity for label in restored.labels] == [label.identity for label in model.labels]
    # Trailing whitespace leaves valid JSON and model hashes intact; only the
    # bounded reader must reject this extra byte before parsing the model.
    path.write_bytes(payload+b' ')
    with pytest.raises(ValueError, match='reader limit'):
        LinearModel.read(path)


@pytest.mark.parametrize('facade', ['public', 'source', 'parsed'])
def test_tagged_load_parses_once_and_validates_compiler_once(tagged_reference_problem, tmp_path, monkeypatch, facade):
    from ye3t_methods.atomistic import tagged_cauchy_image as image
    config, basis, frames = tagged_reference_problem
    model = LinearModel(basis).fit(frames, config=config)
    path = model.write(tmp_path/'load_once.ye3t.json')
    payload = json.loads(path.read_text())
    original_payload = copy.deepcopy(payload)
    original_validate = image.CompiledTaggedCauchyImage.from_dict
    original_open = Path.open
    validations, reads = [], []
    def validate(cls, value, compiler_validation='full'):
        validations.append(compiler_validation)
        return original_validate(value, compiler_validation=compiler_validation)
    def open_file(self, *args, **kwargs):
        if self == path:
            reads.append(self)
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(image.CompiledTaggedCauchyImage, 'from_dict', classmethod(validate))
    monkeypatch.setattr(Path, 'open', open_file)
    if facade == 'public':
        restored = LinearModel.read(path)._fitted
    elif facade == 'source':
        restored = image.load_tagged_cauchy_image_model(path)
    else:
        restored = image._load_tagged_cauchy_image_payload(payload)
    assert validations == ['full']
    assert len(reads) == (0 if facade == 'parsed' else 1)
    assert payload == original_payload
    assert restored.evaluator.compiled.self_hash == model._fitted.evaluator.compiled.self_hash
    np.testing.assert_array_equal(restored.beta_by_species['Ni'], model._fitted.beta_by_species['Ni'])


def test_configured_tagged_fit_reuses_rows_with_new_targets(tagged_reference_problem, tmp_path, monkeypatch):
    config, basis, frames = tagged_reference_problem
    monkeypatch.setenv('YE3T_ACE_CACHE_DIR', str(tmp_path/'cache'))
    initial = LinearModel(basis).fit(frames, config=config)
    assert initial._fitted.fit_metadata['geometry_cache_hits'] == 0
    changed = [frame.copy() for frame in frames]
    for frame in changed:
        frame.info['energy'] += .2
    evaluator = basis._descriptor.metadata['tagged_cauchy_image_evaluator']
    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError('A target change must reuse the same target-free rows.')
    with monkeypatch.context() as patch:
        patch.setattr(evaluator, 'evaluate_edge_list', unexpected_evaluation)
        repeated = LinearModel(basis).fit(changed, config=config)
    assert repeated._fitted.fit_metadata['geometry_cache_hits'] == len(frames)
    assert not np.array_equal(initial._fitted.beta_by_species['Ni'], repeated._fitted.beta_by_species['Ni'])
    uncached_config = copy.deepcopy(config)
    uncached_config['runtime']['cache']['mode'] = 'off'
    monkeypatch.setenv('YE3T_ACE_CACHE_DIR', str(tmp_path/'off_cache'))
    rep = YE3TRepresentation.from_config(uncached_config['representation'])
    uncached_basis = Basis.from_config(uncached_config['basis'], representation=rep,
                                      runtime=uncached_config['runtime'])
    uncached = LinearModel(uncached_basis).fit(changed, config=uncached_config)
    assert not (tmp_path/'off_cache').exists()
    assert [label.identity for label in uncached_basis.labels] == [label.identity for label in basis.labels]
    np.testing.assert_allclose(uncached._fitted.beta_by_species['Ni'],
                               repeated._fitted.beta_by_species['Ni'], atol=1e-12, rtol=1e-12)


def test_tagged_sklearn_reuses_configured_geometry_rows(tagged_reference_problem, tmp_path, monkeypatch):
    pytest.importorskip('sklearn', reason='requires optional scikit-learn fitting extra')
    config, basis, frames = tagged_reference_problem
    monkeypatch.setenv('YE3T_ACE_CACHE_DIR', str(tmp_path/'cache'))
    LinearModel(basis).fit(frames, config=config)
    config = copy.deepcopy(config)
    config['model']['fit']['solver'] = 'ard'
    config['model']['fit'].pop('alpha')
    expected = LinearModel(basis).fit(frames, config=config)
    evaluator = basis._descriptor.metadata['tagged_cauchy_image_evaluator']
    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError('Changing solver must reuse the target-free geometry rows.')
    monkeypatch.setattr(evaluator, 'evaluate_edge_list', unexpected_evaluation)
    actual = LinearModel(basis).fit(frames, config=config)
    np.testing.assert_array_equal(actual._fitted.beta_by_species['Ni'],
                                  expected._fitted.beta_by_species['Ni'])


def test_tagged_row_cache_policies_and_corruption(tagged_reference_problem, tmp_path):
    _, basis, frames = tagged_reference_problem
    descriptor, atoms = basis._descriptor, frames[0]
    directory = tmp_path/'rows'
    with pytest.raises(FileNotFoundError, match='read-only'):
        tagged_cauchy_image_geometry_row(descriptor, atoms, directory, 'read_only')
    assert not directory.exists()
    fresh = tagged_cauchy_image_geometry_row(descriptor, atoms, directory)
    cached = tagged_cauchy_image_geometry_row(descriptor, atoms, directory, 'read_only')
    assert cached['cache_hit'] and not fresh['cache_hit']
    path = directory/(fresh['identity']['row_hash']+'.npz')
    with np.load(path, allow_pickle=False) as archive:
        contents = {key:archive[key] for key in archive.files}
    contents['force_design'][0,0] += 1
    np.savez_compressed(path, **contents)
    bad_bytes = path.read_bytes()
    with pytest.raises(ValueError, match='cache content'):
        tagged_cauchy_image_geometry_row(descriptor, atoms, directory)
    uncached = tagged_cauchy_image_geometry_row(descriptor, atoms, directory, 'off')
    assert path.read_bytes() == bad_bytes
    repaired = tagged_cauchy_image_geometry_row(descriptor, atoms, directory, 'refresh')
    assert not repaired['cache_hit']
    for name in ('feature_sums','feature_square_sums','force_design','stress_design','species_counts'):
        np.testing.assert_array_equal(fresh[name], uncached[name])
        np.testing.assert_array_equal(fresh[name], repaired[name])
    moved = atoms.copy()
    moved.positions[0,0] += .03
    with pytest.raises(FileNotFoundError, match='read-only'):
        tagged_cauchy_image_geometry_row(descriptor, moved, directory, 'read_only')
    new = tagged_cauchy_image_geometry_row(descriptor, moved, directory)
    assert new['identity']['row_hash'] != fresh['identity']['row_hash']
