"""Inspect a compiler-selected density feature and its mathematical label.
"""
from pathlib import Path
from ase.io import read
from ye3t import YE3TRepresentation
from ye3t_methods import Basis

fixtures = Path(__file__).with_name("fixtures")
config = {
    'metadata': {
        'schema': 'ye3t_config_v1',
        'name': 'inspect_features',
        'status': 'stable',
        'evaluation_structure': str(fixtures / 'ni_validation.extxyz'),
        'feature_index': 40,
        'preview_atoms': 5,
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
    'model': {},
    'targets': {},
    'validation': {},
}

representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation, runtime=config["runtime"])
atoms = read(config["metadata"]["evaluation_structure"])
descriptors = basis.create(atoms)
feature = config["metadata"]["feature_index"]
print(basis)
print("descriptor shape", descriptors.shape)
print(basis.labels[feature])
print(basis.describe(feature))
print(basis.describe(feature, format="latex"))
print("selected descriptor preview", descriptors[:config["metadata"]["preview_atoms"], feature])
