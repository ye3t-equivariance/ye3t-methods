"""Fit invariant shell-density features with joint rotation/permutation intermediates.

Three smooth radial windows define the shell factors. The compiler retains their
role action while constructing scalar energy features. This ASE workflow uses
the small labeled Ni subset; change the input paths to fit a complete dataset.
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
        'name': 'ni_shell_density_fit',
        'status': 'experimental',
        'training_structures': str(fixtures / 'ni_training.extxyz'),
        'evaluation_structure': str(fixtures / 'ni_validation.extxyz'),
        'output_path': str(workflow / 'quickstart_linear/ni_shell.ye3t.json'),
    },
    'representation': {
        'group': 'O3',
        'ranks': [1, 2, 3, 4],
        'parent': {'young_lambda': '(N)', 'L': 0, 'parity': 'even'},
        'factorization': 'cauchy',
        'subspace': 'full',
        'uncoupled_factor_inputs': {'eta_count_per_rank': {1: 2, 2: 2, 3: 1, 4: 1}, 'l_max_per_rank': {1: 1, 2: 2, 3: 1, 4: 1}},
        'intermediates': {'young_kappa': 'all_valid', 'block_rotation': {'policy': 'all_valid'}},
    },
    'basis': {
        'single_factors': {
            'species': ['Ni'],
            'chemical': {'kind': 'explicit'},
            'radial': {
                'family': 'shell_filtered_chebyshev_c2_v1',
                'cutoff_A': 4.6,
                'shells': {'kind': 'softmax_gaussian', 'centers': [0.4, 0.65, 0.85], 'width': 0.15},
                'orthogonalization': {'scope': 'joint_source', 'factorization': 'cholesky'},
            },
        },
        'tensor_product': {'kind': 'shell_density'},
        'catalogue': {
            'ranks': [1, 2, 3, 4],
            'nmax_per_rank': {1: 2, 2: 2, 3: 1, 4: 1},
            'lmax_per_rank': {1: 1, 2: 2, 3: 1, 4: 1},
            'max_source_blocks': 2,
        },
    },
    'runtime': {'device': 'cpu', 'dtype': 'float64', 'evaluator': 'auto', 'neighbors': 'ase'},
    'model': {
        'kind': 'linear',
        'fit': {'solver': 'ridge', 'alpha': 1e-08, 'weights': {'energy': 0.1, 'forces': 1.0}},
        'reference_energy': {'per_species_E0_eV': {'Ni': 0.0}, 'fit_E0': True},
    },
    'targets': {'energy': 'energy', 'forces': 'forces', 'stress': None},
    'validation': {'checks': ['force_fd', 'round_trip']},
}

structures = read(config["metadata"]["training_structures"], index=":")
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation, runtime=config["runtime"])
model = LinearModel(basis).fit(structures, config=config)

output = Path(config["metadata"]["output_path"])
output.parent.mkdir(parents=True, exist_ok=True)
artifact = model.write(output)
restored = LinearModel.read(artifact)
atoms = read(config["metadata"]["evaluation_structure"])
reference_energy = atoms.get_potential_energy()
atoms.calc = restored.ase_calculator(evaluator=config["runtime"]["evaluator"],
                                     neighbors=config["runtime"]["neighbors"])

print(basis)
print("training structures", len(structures))
print("saved model", artifact)
print("validation reference energy / eV", reference_energy)
print("validation predicted energy / eV", atoms.get_potential_energy())
print("force shape", atoms.get_forces().shape)
