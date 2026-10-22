"""Fit LASSO and ARD models on the same 100-feature Ni density basis.

Requires the scikit-learn fitting extra. ARD also provides predictive uncertainty.
"""
import os
from pathlib import Path
from ase.io import read
from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel

fixtures = Path(__file__).with_name("fixtures")
workflow = Path(os.environ.get("YE3T_WORKFLOW_ROOT", "~/ye3t-workflows")).expanduser()
config = {
    'metadata': {
        'schema': 'ye3t_config_v1',
        'name': 'ni_sklearn_fit',
        'status': 'stable',
        'training_structures': str(fixtures / 'ni_training.extxyz'),
        'evaluation_structure': str(fixtures / 'ni_validation.extxyz'),
        'output_path': str(workflow / 'quickstart_linear/ni_sklearn'),
    },
    'representation': {
        'group': 'O3',
        'ranks': [1, 2, 3, 4, 5, 6, 7, 8],
        'parent': {'young_lambda': '(N)', 'L': 0, 'parity': 'even'},
        'factorization': 'cauchy',
        'subspace': 'full',
        'uncoupled_factor_inputs': {
            'eta_count_per_rank': {1: 4, 2: 4, 3: 3, 4: 3, 5: 2, 6: 2, 7: 2, 8: 2},
            'l_max_per_rank': {1: 0, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
        },
        'intermediates': {'young_kappa': 'all_valid', 'block_rotation': {'policy': 'all_valid'}},
    },
    'basis': {
        'single_factors': {
            'species': ['Ni'],
            'radial': {
                'family': 'pace_chebexp_cos',
                'cutoff_A': 4.638049165633364,
                'cutoff_width_A': 0.01,
                'lambda': 0.7928781217554153,
            },
            'chemical': {'kind': 'explicit'},
        },
        'tensor_product': {'kind': 'density'},
        'catalogue': {
            'ranks': [1, 2, 3, 4, 5, 6, 7, 8],
            'nmax_per_rank': {1: 4, 2: 4, 3: 3, 4: 3, 5: 2, 6: 2, 7: 2, 8: 2},
            'lmax_per_rank': {1: 0, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
            'source_block_partitions_by_rank': {
                1: [[1]],
                2: [[2], [1, 1]],
                3: [[3], [2, 1]],
                4: [[4]],
                5: [[5]],
                6: [[6]],
                7: [[7]],
                8: [[8]],
            },
        },
    },
    'runtime': {
        'evaluator': 'torch',
        'neighbors': 'ase',
        'cache': {'mode': 'auto'},
        'dtype': 'float64',
        'device': 'cpu',
    },
    'model': {
        'kind': 'linear',
        'fit': {'solver': 'lasso', 'alpha': 1e-05,
                'solver_options': {'column_scaling': 'rms'},
                'weights': {'energy': 1.0, 'forces': 1.0}},
        'reference_energy': {'per_species_E0_eV': {'Ni': 0.0}, 'fit_E0': True},
    },
    'targets': {'energy': 'energy', 'forces': 'forces', 'stress': None},
    'validation': {'checks': ['round_trip']},
}

structures = read(config["metadata"]["training_structures"], index=":")
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation, runtime=config["runtime"])
output = Path(config["metadata"]["output_path"])
output.mkdir(parents=True, exist_ok=True)

lasso = LinearModel(basis).fit(structures, config=config)
lasso_path = lasso.write(output / "ni_lasso.pt")
config["model"]["fit"] = {"solver": "ard", "solver_options": {"column_scaling": "rms"},
                         "weights": {"energy": 1.0, "forces": 1.0}}
ard = LinearModel(basis).fit(structures, config=config)
ard_path = ard.write(output / "ni_ard.pt")

restored = LinearModel.read(ard_path)
atoms = read(config["metadata"]["evaluation_structure"])
atoms.calc = restored.ase_calculator(evaluator=config["runtime"]["evaluator"],
                                     neighbors=config["runtime"]["neighbors"])
uncertainty = restored.predict_uncertainty(atoms)
print(basis)
print("LASSO model", lasso_path)
print("ARD model", ard_path)
print("energy / eV", atoms.get_potential_energy())
print("atomic energy standard deviation range / eV",
      uncertainty["atomic_energy_std_eV"].min(), uncertainty["atomic_energy_std_eV"].max())
print("total energy standard deviation / eV", uncertainty["total_energy_std_eV"])
