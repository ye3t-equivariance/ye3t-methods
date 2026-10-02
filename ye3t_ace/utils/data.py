"""Data helpers for importable YE3T-ACE workflows."""

from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PACKAGE_ROOT / "examples"
EXAMPLE_DATA_DIR = EXAMPLES_DIR / "data"
EXAMPLE_GENERATED_DIR = EXAMPLES_DIR / "generated"
POST_RELEASE_DIR = PACKAGE_ROOT / "post_release_examples"
LONG_EXAMPLES_DIR = EXAMPLES_DIR / "long_examples"


def example_data_path(name):
    """Return a bundled public-example data path."""
    return EXAMPLE_DATA_DIR / name


def example_generated_path(*parts):
    """Return a public-example generated-output path."""
    return EXAMPLE_GENERATED_DIR.joinpath(*parts)


def load_latte_snapshot(path=None, elems=("H", "O")):
    """Return tensor inputs for the bundled H/O descriptor snapshot."""
    import json

    raw = json.loads(Path(path or example_data_path("latte_md_0.json")).read_text(encoding="utf-8"))["Dataset"]["Data"][0]
    type_map = {symbol: index for index, symbol in enumerate(tuple(elems))}
    return (
        np.asarray(raw["Positions"], dtype=float),
        np.asarray(raw["Lattice"], dtype=float),
        np.asarray(raw["Charges"], dtype=float),
        np.asarray([type_map[symbol] for symbol in raw["AtomTypes"]], dtype=int),
        list(elems),
    )


def load_latte_structure(path=None):
    """Return positions, cell, charges, and atom types for the bundled snapshot."""
    positions, cell, charges, atom_types, _elems = load_latte_snapshot(path=path)
    return positions, cell, charges, atom_types


def all_octant_points(seed=7):
    """Return deterministic all-octant point-cloud coordinates."""
    base = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=float,
    )
    scale = np.random.default_rng(int(seed)).uniform(0.8, 1.2, size=base.shape)
    return base * scale


def descriptor_rotation_probe_points(seed=7):
    """Return non-axis-aligned all-octant points for rotation diagnostics."""
    base = np.asarray(
        [
            [0.5, 0.3, 0.2],
            [-0.4, 0.2, 0.3],
            [0.3, -0.5, 0.4],
            [0.2, 0.4, -0.6],
            [-0.5, -0.4, 0.2],
            [-0.2, 0.5, -0.3],
            [0.4, -0.2, -0.5],
            [-0.3, -0.2, -0.4],
        ],
        dtype=float,
    )
    scale = np.random.default_rng(int(seed)).uniform(0.8, 1.2, size=base.shape)
    points = base * scale
    signs = {tuple(np.sign(row).astype(int)) for row in points}
    if len(signs) != 8:
        raise RuntimeError("rotation probe points must span all octants")
    return points
