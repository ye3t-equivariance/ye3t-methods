"""Current embedded catalogues reload with complete provenance checks."""
from copy import deepcopy
import hashlib
import json

import pytest
from ye3t_methods.atomistic import YE3TDescriptors


def _hash(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def test_current_embedded_catalogue_reloads_and_rejects_tampering():
    config = {
        'basis': {'type': 'tagged_cauchy_carriers', 'species': ['Ni'], 'cutoff_A': 2.4,
                  'catalogue': {'ranks': [1], 'tag_counts': [0, 1],
                                'nmax_per_rank': {1: 1}, 'lmax_per_rank': {1: 1},
                                'source_block_partitions_by_rank': {1: [[1]]}, 'input_Lmax': 1}},
        'representation': {'mode': 'tagged_cauchy_carriers', 'sector_policy': 'tagged_mixed'},
        'runtime': {'backend': 'reference', 'compiled_cache_dir': None}, 'model': {},
    }
    descriptor = YE3TDescriptors.ye3t_basis(config)
    compiled = descriptor.metadata['tagged_cauchy_carriers_compiled']
    assert compiled['schema'] == 'ye3t_tagged_cauchy_carrier_catalogue_v2'
    reloaded = YE3TDescriptors.ye3t_basis(config, _compiled_tagged_carriers=compiled)
    assert reloaded.metadata['tagged_cauchy_carriers_compiled']['self_hash'] == compiled['self_hash']
    for mutation in ('hash', 'schema', 'source'):
        changed = deepcopy(compiled)
        if mutation == 'hash':
            changed['self_hash'] = '0' * 64
        elif mutation == 'schema':
            changed['schema'] = 'unsupported_catalogue'
            changed['self_hash'] = _hash({k:v for k,v in changed.items() if k != 'self_hash'})
        else:
            changed['sources'][0]['self_hash'] = '0' * 64
            changed['self_hash'] = _hash({k:v for k,v in changed.items() if k != 'self_hash'})
        with pytest.raises(ValueError):
            YE3TDescriptors.ye3t_basis(config, _compiled_tagged_carriers=changed)
