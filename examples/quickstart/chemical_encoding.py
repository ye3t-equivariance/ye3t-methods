"""Evaluate scalar ASE descriptors with a fixed two-channel species embedding.

Edit the species list, embedding matrix, radial source, and catalogue below.
Changing ``chemical.kind`` to ``explicit`` restores one-hot neighbor channels."""
from ase import Atoms
from ye3t import YE3TRepresentation
from ye3t_methods import Basis

config = {
    'metadata': {
        'schema': 'ye3t_config_v1',
        'name': 'chemical_encoding',
        'status': 'stable',
        'system': {
            'symbols': 'NiCuAlNi',
            'positions_A': [[0, 0, 0], [0, 1.8, 1.8], [1.8, 0, 1.8], [1.8, 1.8, 0]],
            'cell_A': [3.6, 3.6, 3.6],
            'pbc': True,
        },
    },
    'representation': {
        'group': 'O3',
        'ranks': [1, 2, 3, 4],
        'parent': {'young_lambda': '(N)', 'L': 0, 'parity': 'even'},
        'factorization': 'cauchy',
        'subspace': 'full',
        'uncoupled_factor_inputs': {'eta_count_per_rank': {1: 3, 2: 2, 3: 2, 4: 2}, 'l_max_per_rank': {1: 0, 2: 2, 3: 1, 4: 1}},
        'intermediates': {'young_kappa': 'all_valid', 'block_rotation': {'policy': 'all_valid'}},
    },
    'basis': {
        'single_factors': {
            'species': ['Ni', 'Cu', 'Al'],
            'radial': {'family': 'pace_chebexp_cos', 'cutoff_A': 4.5, 'cutoff_width_A': 0.01, 'lambda': 0.79},
            'chemical': {
                'kind': 'fixed_embedding',
                'species_order': ['Ni', 'Cu', 'Al'],
                'matrix': [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]],
            },
        },
        'tensor_product': {'kind': 'density'},
        'catalogue': {
            'ranks': [1, 2, 3, 4],
            'nmax_per_rank': {1: 3, 2: 2, 3: 2, 4: 2},
            'lmax_per_rank': {1: 0, 2: 2, 3: 1, 4: 1},
            'source_block_partitions_by_rank': {1: [[1]], 2: [[2], [1, 1]], 3: [[3], [2, 1]], 4: [[4]]},
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

system = config["metadata"]["system"]
atoms = Atoms(system["symbols"], positions=system["positions_A"],
              cell=system["cell_A"], pbc=system["pbc"])
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation, runtime=config["runtime"])
descriptors = basis.create(atoms)
print("chemical encoding", config["basis"]["single_factors"]["chemical"])
print("descriptor shape", descriptors.shape)
print("first feature", basis.labels[0])
