"""Evaluate a selected explicit-Phi coupling on one eight-factor star.

Each occurrence gives an atom index and periodic-image shift. The output keeps
all 14 tableau and five magnetic components of the selected (4,4), L=2 copy.
This specialized construction evaluates the supplied cluster. General cluster
orbit sums and linear potential fitting require additional source support.
"""
from ase.build import bulk
from ye3t import YE3TRepresentation
from ye3t_methods import Basis

config = {'metadata': {'schema': 'ye3t_config_v1',
              'name': 'ni_phi_rank8',
              'status': 'experimental',
              'system': {'element': 'Ni',
                         'crystal': 'fcc',
                         'lattice_parameter_A': 3.508,
                         'center': 0,
                         'occurrences': [[1, [0, 0, 0]],
                                         [1, [0, -1, 0]],
                                         [1, [0, 0, -1]],
                                         [1, [0, -1, -1]],
                                         [2, [0, 0, 0]],
                                         [2, [-1, 0, 0]],
                                         [2, [0, 0, -1]],
                                         [2, [-1, 0, -1]]]}},
 'representation': {'group': 'O3',
                    'ranks': [8],
                    'parent': {'young_lambda': '(4,4)', 'L': 2, 'parity': 'even'},
                    'factorization': 'cauchy',
                    'subspace': 'full',
                    'uncoupled_factor_inputs': {'eta_count_per_rank': {8: 2},
                                                'l_max_per_rank': {8: 1}},
                    'intermediates': {'young_kappa': {'policy': 'explicit',
                                                      'by_block_size': {4: ['(4)']}},
                                      'block_rotation': {'policy': 'explicit',
                                                         'Lambda_values_by_block_size': {4: [0,
                                                                                             2,
                                                                                             4]}}}},
 'basis': {'single_factors': {'species': ['Ni'],
                              'radial': {'family': 'pace_chebexp_cos',
                                         'cutoff_A': 4.638049165633364,
                                         'cutoff_width_A': 0.01,
                                         'lambda': 0.7928781217554153},
                              'chemical': {'kind': 'explicit'}},
           'tensor_product': {'kind': 'explicit_phi',
                              'motif': {'kind': 'rooted_star',
                                        'ordered_factors': True,
                                        'leaf_edges': []}},
           'catalogue': {'ranks': [8],
                         'nmax_per_rank': {8: 2},
                         'lmax_per_rank': {8: 1},
                         'source_block_partitions_by_rank': {8: [[4, 4]]},
                         'fixed_content': [{'factor': {'species': 'Ni',
                                                       'radial_index': 0,
                                                       'l': 1},
                                            'copies': 4},
                                           {'factor': {'species': 'Ni',
                                                       'radial_index': 1,
                                                       'l': 1},
                                            'copies': 4}],
                         'selection': {'coupling_paths': [{'young_kappa': ['(4)', '(4)'],
                                                           'Lambda': [0, 2]}]}}},
 'runtime': {'evaluator': 'torch',
             'neighbors': 'ase',
             'cache': {'mode': 'off'},
             'dtype': 'float64',
             'device': 'cpu'},
 'model': {'kind': 'descriptor_only'},
 'targets': {},
 'validation': {}}

system = config["metadata"]["system"]
atoms = bulk(system["element"], system["crystal"], a=system["lattice_parameter_A"], cubic=True)
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation, runtime=config["runtime"])
cluster = basis.create_cluster(atoms, system["center"], system["occurrences"])
print("cluster feature shape (a, tableau, M)", cluster.shape)
print("selected coupling", basis.labels[0])
