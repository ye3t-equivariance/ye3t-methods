"""Exact promoted Ni composite regression for explicit ASE neighbor policies."""

import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.mixing import SumCalculator

from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
from ye3t_methods.atomistic.tagged_cauchy_image import YE3TTaggedCauchyCalculator


@pytest.mark.parametrize("size,repeat,energy_eV", (
    (16, (2, 2, 1), -92.42185554668458),
    (128, (4, 4, 2), -739.7408186866148),
))
@pytest.mark.parametrize("neighbors,selected", (
    ("ase", "ase_neighbor_list"),
    ("matscipy", "matscipy_neighbor_list"),
))
def test_paper_ni_composite_neighbor_policy_preserves_energy_force_stress(
    size, repeat, energy_eV, neighbors, selected,
):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library:
        pytest.skip("requires optional ye3t-lammps tagged C ABI library")
    if neighbors == "matscipy":
        pytest.importorskip("matscipy", reason="requires optional matscipy neighbor dependency")
    folder = (Path(__file__).resolve().parents[1] / "examples" / "publication"
              / "cost_comparison" / "lammps" / "Ni" / "models" / "ye3t_tagged_127")
    atoms = bulk("Ni", "fcc", a=3.508, cubic=True).repeat(repeat)
    assert len(atoms) == size
    atoms.positions[0] += (0.08, -0.05, 0.04)

    baseline = atoms.copy()
    baseline.calc = SumCalculator([
        YE3TTaggedCauchyCalculator.from_artifact(
            folder / "model.ye3t.json", native_library=library,
        ),
        YE3TZBLCalculator.from_model_manifest(folder / "model_manifest.json"),
    ])
    explicit = atoms.copy()
    linear = YE3TTaggedCauchyCalculator.from_artifact(
        folder / "model.ye3t.json", native_library=library, neighbors=neighbors,
    )
    explicit.calc = SumCalculator([
        linear, YE3TZBLCalculator.from_model_manifest(folder / "model_manifest.json"),
    ])

    assert baseline.get_potential_energy() == pytest.approx(energy_eV, abs=1e-8)
    np.testing.assert_allclose(explicit.get_potential_energy(),
                               baseline.get_potential_energy(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(explicit.get_forces(), baseline.get_forces(),
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(explicit.get_stress(), baseline.get_stress(),
                               rtol=0, atol=1e-8)
    runtime = linear.native_runtime
    assert runtime.neighbors == neighbors
    assert runtime.last_neighbor_backend == selected
    rebuilds = runtime.topology_rebuilds
    explicit.positions[0, 0] += 0.002
    baseline.positions[0, 0] += 0.002
    np.testing.assert_allclose(explicit.get_forces(), baseline.get_forces(),
                               rtol=0, atol=1e-8)
    assert runtime.topology_rebuilds == rebuilds
