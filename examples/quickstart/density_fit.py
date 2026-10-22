"""Fit a scalar density Ni potential with ranks one through eight.

The colocated labeled structures are a recorded subset of the paper dataset.
Replace the two input paths for another fit; the complete paper refitting
workflow is in publication/cost_comparison/refit_paper_ni.py. Outputs go under
~/ye3t-workflows or YE3T_WORKFLOW_ROOT. The density evaluator uses PyTorch.
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
        'name': 'ni_density_fit',
        'status': 'stable',
        'training_structures': str(fixtures / 'ni_training.extxyz'),
        'evaluation_structure': str(fixtures / 'ni_validation.extxyz'),
        'output_path': str(workflow / 'quickstart_linear/ni_density.pt'),
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
        'fit': {'solver': 'ridge', 'alpha': 1e-08, 'weights': {'energy': 1.0, 'forces': 1.0}},
        'reference_energy': {'per_species_E0_eV': {'Ni': 0.0}, 'fit_E0': True},
    },
    'targets': {'energy': 'energy', 'forces': 'forces', 'stress': None},
    'validation': {'checks': ['round_trip']},
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
