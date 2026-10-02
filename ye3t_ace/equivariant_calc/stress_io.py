
"""I/O helpers for per-particle tesseral stress examples."""

import re
from pathlib import Path

import numpy as np
from ye3t_ace._record import recordclass


@recordclass(('symbols', 'positions', 'cell', 'pbc', 'arrays'), frozen = True)
class StressExtXYZData:
    pass


def read_simple_extxyz(path):
    """Read the simple extended-XYZ file written by the tesseral-stress converter.

    The parser is intentionally lightweight and only handles the subset needed for
    the uploaded ``tess_ats.xyz`` example: ``species``, ``pos``, ``norm_sig``,
    ``T_00``, and ``S_2M``.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    header = lines[1]
    lat_match = re.search(r'Lattice="([^"]+)"', header)
    lattice_vals = np.fromstring(lat_match.group(1), sep=' ')
    cell = lattice_vals.reshape(3, 3)
    prop_match = re.search(r'Properties=([^\s]+)', header)
    prop_tokens = prop_match.group(1).split(':')
    props = []
    i = 0
    while i < len(prop_tokens):
        name = prop_tokens[i]; kind = prop_tokens[i+1]; width = int(prop_tokens[i+2]); props.append((name, kind, width)); i += 3
    symbols = []; prop_arrays = {name: [] for name, _, _ in props if name != 'species'}; positions = []
    for line in lines[2:2+n_atoms]:
        toks = line.split(); cursor = 0
        for name, _, width in props:
            vals = toks[cursor:cursor+width]; cursor += width
            if name == 'species':
                symbols.append(vals[0])
            elif name == 'pos':
                positions.append([float(x) for x in vals])
            else:
                prop_arrays[name].append([float(x) for x in vals])
    arrays = {name: np.asarray(vals, dtype=float) for name, vals in prop_arrays.items()}
    return StressExtXYZData(symbols=symbols, positions=np.asarray(positions, float), cell=np.asarray(cell, float), pbc=np.array([True, True, True], bool), arrays=arrays)
