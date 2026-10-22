"""Execute saved-model, vector, MD and full-basis refit examples."""
import ast
import csv
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'examples'


@pytest.mark.parametrize('relative', [
    'publication/cost_comparison/ase_native_density.py',
    'publication/cost_comparison/ase_native_tagged.py',
    'publication/cost_comparison/ase_native_ni.py',
    'quickstart/paper_ni_portable_ase.py',
])
def test_saved_example_matches_reference_energy(relative, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runpy.run_path(str(ROOT / relative))
    atoms = result['atoms']
    validation = result['config']['validation']
    expected = validation.get('reference_energy_eV', validation.get('lammps_step_zero_energy_eV'))
    np.testing.assert_allclose(atoms.get_potential_energy(), expected,
                               atol=validation['energy_tolerance_eV'], rtol=0)
    assert np.isfinite(atoms.get_forces()).all()
    assert np.isfinite(atoms.get_stress()).all()


def test_paper_nve_100_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runpy.run_path(str(ROOT / 'quickstart/paper_ni_nve.py'))
    with Path(result['output']).open() as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == 101
    assert int(samples[-1]['step']) == 100
    energies = np.asarray(result['energies'])
    assert np.isfinite(energies).all()
    # Near-equilibrium 1 fs NVE must conserve energy across the complete example.
    assert np.max(np.abs(energies - energies[0])) / len(result['atoms']) < 1e-3


def test_vector_fit_rotation_reload_and_lammps_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = ROOT / 'quickstart/per_atom_vector_to_lammps.py'
    tree = ast.parse(path.read_text())
    end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'config' for t in node.targets))
    result = {'__file__': str(path)}
    exec(compile(ast.Module(body=tree.body[:end+1], type_ignores=[]), str(path), 'exec'), result)
    # Exercise the editable filename as well as the complete scientific config.
    result['config']['metadata']['output_path'] = str(tmp_path/'copper_vectors.ye3t.json')
    exec(compile(ast.Module(body=tree.body[end+1:], type_ignores=[]), str(path), 'exec'), result)
    model, atoms = result['restored'], result['atoms']
    expected = model.predict(atoms)['mean_real_tesseral']
    assert expected.shape == (32, 3)
    np.testing.assert_allclose(result['model'].predict(atoms)['mean_real_tesseral'], expected,
                               rtol=1e-12, atol=1e-12)
    rotated = atoms.copy()
    rotated.rotate(90, 'z', rotate_cell=True)
    # The L=1 real-tesseral component order is y,z,x.
    cartesian = expected[:, [2, 0, 1]]
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    actual = model.predict(rotated)['mean_real_tesseral'][:, [2, 0, 1]]
    np.testing.assert_allclose(actual, cartesian @ rotation.T, rtol=1e-9, atol=1e-9)
    directory = Path(result['model_path']).parent
    assert (directory / 'atoms.data').is_file()
    assert 'ye3t/property/atom' in (directory / 'in.property').read_text()
    assert 'ye3t/property/atom copper_vectors.ye3t.json Cu' in (directory / 'in.property').read_text()


def test_saved_export_missing_model_explains_prerequisite(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_WORKFLOW_ROOT', str(tmp_path))
    with pytest.raises(FileNotFoundError, match='Run tagged_fit.py first'):
        runpy.run_path(str(ROOT/'quickstart/saved_ase_export.py'))


def test_selected_127_refit_preserves_full_example_basis(tmp_path):
    from ye3t_methods import LinearModel
    from ye3t_methods.studies.ni_linear import load_refit_structures

    path = ROOT / 'publication/cost_comparison/refit_paper_ni.py'
    tree = ast.parse(path.read_text())
    # Execute the visible imports and config, then shorten only the number of frames.
    end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'config' for t in node.targets))
    scope = {'__file__': str(path)}
    exec(compile(ast.Module(body=tree.body[:end+1], type_ignores=[]), str(path), 'exec'), scope)
    config = scope['config']
    training, validation = load_refit_structures(config)
    training = [min((atoms for atoms in training if atoms.info['config_type'] == group), key=len)
                for group in config['model']['fit']['group_weights']]
    validation = sorted(validation, key=len)[:1]
    config['validation']['expected_train_count'] = len(training)
    model = LinearModel.read(config['model']['source_archive'])
    original_labels = [label.identity for label in model.labels]
    assert len(original_labels) == 127
    model.fit(training, config=config)
    restored = LinearModel.read(model.write(tmp_path / 'refit.ye3t'))
    assert [label.identity for label in restored.labels] == original_labels
    score = restored.score(validation, config=config)
    assert score['structure_count'] == 1
    assert np.isfinite(score['forces_rmse_eV_per_A'])
    assert np.isfinite(score['energy_rmse_eV_per_atom'])
    before = validation[0].copy()
    before.calc = model.ase_calculator(evaluator='torch')
    after = validation[0].copy()
    after.calc = restored.ase_calculator(evaluator='torch')
    np.testing.assert_allclose(before.get_forces(), after.get_forces(), rtol=1e-11, atol=1e-11)
