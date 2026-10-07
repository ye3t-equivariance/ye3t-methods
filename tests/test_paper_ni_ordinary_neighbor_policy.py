"""Native YACE neighbor selection on the promoted Ni ordinary backbone."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from ye3t_methods.atomistic.yace_native import YE3TYACENativeCalculator, _YACENativeRuntime


MODEL = (Path(__file__).resolve().parents[1] / "examples" / "publication"
         / "cost_comparison" / "lammps" / "Ni" / "models"
         / "ye3t_tagged_127" / "ordinary_backbone.yace")


@pytest.mark.parametrize("repeat", ((2, 2, 1), (4, 4, 2)))
@pytest.mark.parametrize("neighbors,selected", (
    ("ase", "ase_neighbor_list"),
    ("matscipy", "matscipy_neighbor_list"),
))
def test_ordinary_ni_native_neighbor_policy_energy_force_stress_and_warm(
    repeat, neighbors, selected,
):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    if neighbors == "matscipy":
        pytest.importorskip("matscipy", reason="requires optional matscipy neighbor dependency")
    atoms = bulk("Ni", "fcc", a=3.508, cubic=True).repeat(repeat)
    atoms.positions[0] += (0.08, -0.05, 0.04)
    baseline = atoms.copy()
    baseline.calc = YE3TYACENativeCalculator.from_artifact(
        MODEL, native_library=library,
    )
    explicit = atoms.copy()
    explicit.calc = YE3TYACENativeCalculator.from_artifact(
        MODEL, native_library=library, neighbors=neighbors,
    )
    np.testing.assert_allclose(explicit.get_potential_energy(),
                               baseline.get_potential_energy(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(explicit.get_forces(), baseline.get_forces(),
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(explicit.get_stress(), baseline.get_stress(),
                               rtol=0, atol=1e-8)
    runtime = explicit.calc.native_runtime
    assert runtime.neighbors == neighbors
    assert runtime.last_neighbor_backend == selected
    rebuilds = runtime.topology_rebuilds
    explicit.positions[0, 0] += 0.002
    baseline.positions[0, 0] += 0.002
    np.testing.assert_allclose(explicit.get_forces(), baseline.get_forces(),
                               rtol=0, atol=1e-8)
    assert runtime.topology_rebuilds == rebuilds


def test_ordinary_native_neighbor_policy_rejects_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "matscipy.neighbours", None)
    with pytest.raises(ImportError, match="neighbors='matscipy'"):
        _YACENativeRuntime(MODEL, neighbors="matscipy")
    with pytest.raises(ValueError, match="neighbors must be auto, ase, or matscipy"):
        _YACENativeRuntime(MODEL, neighbors="unknown")
