"""Regression for the stable compiled ordinary-density ASE/native adapter."""

import numpy as np
import pytest
import torch
from ase import Atoms

from ye3t.couplings import (
    compile as compile_coupler,
    execution_plan_from_compiled_coupler,
    plan as coupling_plan,
)
from ye3t.runtime import native_execution_plan_capabilities
from ye3t_ace.execution_plan import (
    YE3TCompiledArtifactCalculator,
    YE3TCompiledModelArtifact,
    YE3TCompiledSourceEvaluator,
)
from ye3t_ace.equivariant_calc.site_basis_v2 import SiteBasisConfig, SiteBasisV2


def _artifact():
    compiled = compile_coupler(
        coupling_plan(content=(1, 2, 3), input_Ls=(1, 2, 1), target_L=0),
        subduction_materialization_backend="exact",
    )
    plan = execution_plan_from_compiled_coupler(compiled)
    width = YE3TCompiledSourceEvaluator(plan, backend="reference").output_dimension
    return YE3TCompiledModelArtifact(
        execution_plan=plan,
        radial_angular_metadata={
            "radial_basis": "ChebExpCos",
            "cutoff": 3.5,
            "angular_basis": "complex_condon_shortley",
            "source_slots": [
                {"content": c, "mu0": 0, "mu": 0, "kappa0": 0,
                 "kappa": 0, "n": c, "l": l}
                for c, l in ((1, 1), (2, 2), (3, 1))
            ],
        },
        species_mapping={"H": 0},
        normalization={"atomic_base": "none", "descriptor": "compiler_plan"},
        readout={"weights": torch.linspace(0.2, 0.8, width).tolist(), "bias": 0.15},
        model_metadata={"model_family": "linear_YE3T"},
        certificate={"passed": True, "scope": "stable linear adapter regression"},
    )


def _basis(source_backend):
    return SiteBasisV2(SiteBasisConfig(
        rc=[3.5], lmbda=[0.35], nradmax=4, lmax=2,
        possible_types=(0,), charge_mode="none",
        atomic_base_normalization="none", factor_normalization="none",
        spherical_backend="complex", source_backend=source_backend,
        native_source_min_edges=0, dtype=torch.float64,
        complex_dtype=torch.complex128,
    ))


def test_compiled_artifact_ase_native_cpu_matches_reference_after_relocation(tmp_path):
    artifact = _artifact()
    written = artifact.save(tmp_path / "before" / "artifact.ye3t.json")
    moved = tmp_path / "after" / written.name
    moved.parent.mkdir()
    written.rename(moved)
    loaded = YE3TCompiledModelArtifact.load(moved)
    native = loaded.ase_calculator(_basis("native_cpu"), backend="native", device="cpu")
    reference = loaded.ase_calculator(_basis("torch"), backend="reference", device="cpu")
    assert isinstance(native, YE3TCompiledArtifactCalculator)
    atoms = Atoms(
        "H3", positions=[[0.2, 0.3, 0.4], [1.1, 0.5, 0.7], [0.6, 1.4, 0.9]],
        cell=[8.0, 8.0, 8.0], pbc=True,
    )
    actual, expected = atoms.copy(), atoms.copy()
    actual.calc, expected.calc = native, reference
    np.testing.assert_allclose(actual.get_potential_energy(), expected.get_potential_energy(), atol=2e-10)
    np.testing.assert_allclose(actual.get_forces(), expected.get_forces(), atol=2e-9)
    np.testing.assert_allclose(actual.get_stress(), expected.get_stress(), atol=2e-9)
    report = native.model.runtime_report()
    assert report["source_runtime"]["density_accumulation_backend"] == "native_cpu"
    assert report["readout_fused_into_native_kernel"]
    original = actual.get_forces().copy()
    actual.positions[0, 0] += 0.02
    expected.positions[0, 0] += 0.02
    np.testing.assert_allclose(actual.get_forces(), expected.get_forces(), atol=2e-9)
    assert not np.allclose(original, actual.get_forces())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not native_execution_plan_capabilities()["cuda"],
    reason="requires the compiled YE3T native CUDA extension and CUDA hardware",
)
def test_compiled_artifact_native_cuda_matches_reference():
    artifact = _artifact()
    native = artifact.atomistic_linear_model(_basis("native_cuda"), backend="native", device="cuda")
    reference = artifact.atomistic_linear_model(_basis("torch"), backend="reference", device="cpu")
    edges = torch.tensor(
        [[0.31, 0.47, 0.83], [-0.42, 0.58, 0.27], [0.73, -0.24, 0.51],
         [-0.67, -0.19, 0.37], [0.28, 0.63, -0.44], [-0.35, 0.22, -0.91]],
        dtype=torch.float64,
    )
    index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    types = torch.zeros(3, dtype=torch.long)
    actual = native.energy_forces_virial(edges.cuda(), index.cuda(), types.cuda())
    expected = reference.energy_forces_virial(edges, index, types)
    for key in ("energy", "forces", "strain_derivative", "virial"):
        torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=2e-9, atol=2e-10)
    report = native.runtime_report()
    assert report["source_runtime"]["density_accumulation_backend"] == "native_cuda"
