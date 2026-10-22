"""Scalar tagged YE3T descriptors from ASE, ranks one to eight.

Edit radial counts, angular patterns and complete-channel partitions to change
the input basis. The scalar output includes nontrivial intermediate Young and
angular types. Install the native ye3t backend.
"""
from ase.build import bulk
from ye3t import YE3TRepresentation
from ye3t_methods import Basis


config = {
    'metadata': {
        'schema': 'ye3t_config_v1',
        'name': 'ni_tagged_descriptors',
        'status': 'stable',
        'system': {'element': 'Ni', 'crystal': 'fcc', 'lattice_parameter_A': 3.52},
    },
    'representation': {
        'group': 'O3',
        'ranks': [1, 2, 3, 4, 5, 6, 7, 8],
        'parent': {'young_lambda': '(N)', 'L': 0, 'parity': 'even'},
        'factorization': 'cauchy',
        'subspace': 'full',
        'uncoupled_factor_inputs': {
            'eta_count_per_rank': {1: 4, 2: 3, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
            'l_max_per_rank': {1: 0, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
        },
        'intermediates': {'young_kappa': 'all_valid', 'block_rotation': {'policy': 'all_valid'}},
    },
    'basis': {
        'single_factors': {
            'species': ['Ni'],
            'radial': {'family': 'shifted_jacobi', 'cutoff_A': 4.8},
            'chemical': {'kind': 'explicit'},
        },
        'tensor_product': {
            'kind': 'tagged',
            'tag_counts_per_rank': {1: [1], 2: [2], 3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2]},
        },
        'catalogue': {
            'ranks': [1, 2, 3, 4, 5, 6, 7, 8],
            'nmax_per_rank': {1: 4, 2: 3, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
            'lmax_per_rank': {1: 0, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1},
            'source_block_partitions_by_rank': {
                1: [[1]],
                2: [[2], [1, 1]],
                3: [[3], [2, 1]],
                4: [[4], [2, 2]],
                5: [[3, 2]],
                6: [[4, 2]],
                7: [[5, 2]],
                8: [[6, 2]],
            },
            'angular_patterns_by_rank': {
                1: [[0]],
                2: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
                3: [[0, 0, 0], [0, 1, 1], [0, 2, 2]],
                4: [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 2, 2]],
                5: [[0, 0, 0, 1, 1]],
                6: [[0, 0, 0, 0, 1, 1]],
                7: [[0, 0, 0, 0, 0, 1, 1]],
                8: [[0, 0, 0, 0, 0, 0, 1, 1]],
            },
        },
    },
    'runtime': {
        'evaluator': 'native_cpu',
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
atoms = bulk(system["element"], system["crystal"], a=system["lattice_parameter_A"], cubic=True)
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation, runtime=config["runtime"])
descriptors = basis.create(atoms)

print(basis)
print("descriptor shape", descriptors.shape)
print("first feature", basis.labels[0])
print("last feature", basis.labels[-1])
print("descriptor backend", basis.backend)

# Reuse the same basis on another structure; its labels and compiled maps stay fixed.
displaced = atoms.copy()
displaced.positions[0, 0] += 0.02
displaced_descriptors = basis.create(displaced)
print("changed structure shape", displaced_descriptors.shape)

# Optional analytic derivatives, off by default:
# values, dvalues_dpositions = basis.create(displaced, derivatives=True)
# dvalues_dpositions[center, feature, displaced_atom, xyz] is dB/dR at fixed cell.
# For a shared linear energy readout, sum over centers without the large center axis:
# values, summed_gradient = basis.create(displaced, derivatives="summed")
# force_design = -summed_gradient.reshape(len(basis.labels), -1).T
