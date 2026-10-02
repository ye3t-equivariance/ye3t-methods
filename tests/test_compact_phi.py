"""Explicit-cluster barPhi remains a fixed-feature linear model."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ye3t_methods import Basis, LinearModel
from ye3t_ace import YE3TModel
from ye3t_ace.cluster_phi import (
    HybridACEPhiCalculator, MotifTemplate, PhiMotifSpec, PhiSlotChannel,
)


def test_bar_phi_fit_save_load_labels_forces_and_shear_stress(tmp_path):
    channel = PhiSlotChannel(n=1, l=0, m=0, neighbor_type=0)
    motifs = (
        PhiMotifSpec(MotifTemplate("pair", 2, ()), (channel, channel)),
        PhiMotifSpec(MotifTemplate("star3", 3, ((0, 1), (0, 2))), (channel,) * 3),
    )
    basis = Basis(
        elements=["H"], source="bar_phi", cutoff=3.0,
        channels=(channel,), motif_specs=motifs, edge_basis_backend="simple",
    )
    assert len(basis.labels) == 2
    assert all(label.as_dict()["compiler_coupling_plan"] for label in basis.labels)
    assert basis.labels[0].as_dict()["N"] == 2
    assert "\\overline{\\Phi}" in basis.describe(0, format="latex")

    oracle = YE3TModel.phi(basis._descriptor, {"branches": ("bar_phi",)})
    with torch.no_grad():
        oracle.bar_phi_weight[:] = torch.tensor([0.7, -0.2], dtype=torch.float64)
        oracle.bar_phi_bias[:] = torch.tensor([0.1], dtype=torch.float64)
    structures = []
    for distance in (0.8, 1.0, 1.2, 1.4):
        atoms = Atoms(
            "H3", positions=[[0, 0, 0], [distance, 0.1, 0], [0.2, 1.1, 0.3]],
            cell=[8, 8, 8], pbc=True,
        )
        atoms.calc = HybridACEPhiCalculator(oracle, type_map={"H": 0})
        energy, forces, stress = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        structures.append(atoms)

    model = LinearModel(basis).fit(structures, regularization=1e-12, stress_weight=0.1)
    for atoms in structures:
        candidate = atoms.copy()
        candidate.calc = model.ase_calculator()
        np.testing.assert_allclose(candidate.get_potential_energy(), atoms.get_potential_energy(), atol=1e-8)
        np.testing.assert_allclose(candidate.get_forces(), atoms.get_forces(), atol=1e-8)
        np.testing.assert_allclose(candidate.get_stress(), atoms.get_stress(), atol=1e-8)

    path = model.write(tmp_path / "explicit_phi")
    assert path.name.endswith(".phi.pt")
    loaded = LinearModel.read(path)
    assert [label.as_dict() for label in loaded.labels] == [label.as_dict() for label in basis.labels]
    assert "coefficient:" in loaded.describe(0)
    check = structures[1].copy()
    check.calc = loaded.ase_calculator()
    np.testing.assert_allclose(check.get_forces(), structures[1].get_forces(), atol=1e-8)

    sites = basis.create(structures[1])
    assert sites.shape == (3, 2)
    one_hot = YE3TModel.phi(basis._descriptor, {"branches": ("bar_phi",)})
    with torch.no_grad():
        one_hot.bar_phi_weight[:] = torch.tensor([1.0, 0.0], dtype=torch.float64)
    check.calc = HybridACEPhiCalculator(one_hot, type_map={"H": 0})
    np.testing.assert_allclose(check.get_potential_energy(), sites[:, 0].sum(), atol=1e-12)

    periodic = structures[1].copy()
    periodic.calc = loaded.ase_calculator()
    stress = periodic.get_stress()
    volume = periodic.get_volume()
    shear = np.zeros((3, 3))
    shear[0, 1] = shear[1, 0] = 0.5
    energies = []
    for delta in (1e-6, -1e-6):
        strained = periodic.copy()
        deformation = np.eye(3) + delta * shear
        strained.set_cell(periodic.cell.array @ deformation.T, scale_atoms=False)
        strained.positions = periodic.positions @ deformation.T
        strained.calc = loaded.ase_calculator()
        energies.append(strained.get_potential_energy())
    np.testing.assert_allclose(stress[5], (energies[0] - energies[1]) / (2e-6 * volume), atol=2e-6)

    pytest.importorskip("sklearn")
    sparse = LinearModel(basis).fit(
        structures, fit_method="lasso", sklearn_params={"alpha": 1e-9},
        stress_weight=0.1,
    )
    sparse_atoms = structures[0].copy()
    sparse_atoms.calc = sparse.ase_calculator()
    assert np.isfinite(sparse_atoms.get_potential_energy())
    ard = LinearModel(basis).fit(
        structures, fit_method="ardregression", stress_weight=0.1,
    )
    uncertainty = ard.predict_uncertainty(structures[0])
    assert uncertainty["atomic_energy_std_eV"].shape == (3,)
    assert np.isfinite(uncertainty["atomic_energy_std_eV"]).all()
    assert uncertainty["total_energy_std_eV"] >= 0.0
    reloaded = LinearModel.read(ard.write(tmp_path / "phi_ard"))
    np.testing.assert_allclose(
        reloaded.predict_uncertainty(structures[0])["atomic_energy_std_eV"],
        uncertainty["atomic_energy_std_eV"], rtol=1e-11,
    )
    ridgecv = LinearModel(basis).fit(
        structures, fit_method="ridgecv",
        sklearn_params={"alphas": np.logspace(-8, 0, 4)}, stress_weight=0.1,
    )
    ridgecv_loaded = LinearModel.read(ridgecv.write(tmp_path / "phi_ridgecv"))
    assert ridgecv_loaded._fitted.fit_metadata["sklearn_params"]["alphas"] == [
        float(value) for value in np.logspace(-8, 0, 4)
    ]


def test_bar_phi_all_images_wrap_reindex_and_force_derivative():
    channel = PhiSlotChannel(n=1, l=0, m=0, neighbor_type=0)
    pair = PhiMotifSpec(MotifTemplate("pair", 2, ()), (channel, channel))
    basis = Basis(
        elements=["H"], source="bar_phi", cutoff=5.0,
        channels=(channel,), motif_specs=(pair,), edge_basis_backend="simple",
        periodic_image_mode="all_images",
    )
    oracle = YE3TModel.phi(basis._descriptor, {"branches": ("bar_phi",)})
    with torch.no_grad():
        oracle.bar_phi_weight[:] = torch.tensor([1.0], dtype=torch.float64)
        oracle.bar_phi_bias.zero_()

    atoms = Atoms(
        "H3", positions=[[0.2, 0.2, 0.2], [7.2, 0.2, 0.2], [0.2, 1.5, 0.2]],
        cell=[8.0, 8.0, 8.0], pbc=True,
    )

    def evaluate(current):
        current.calc = HybridACEPhiCalculator(oracle, type_map={"H": 0})
        return current.get_potential_energy(), current.get_forces()

    energy, forces = evaluate(atoms)
    np.testing.assert_allclose(
        energy, basis.create(atoms).sum().item(), atol=1e-11,
    )
    wrapped = atoms.copy()
    wrapped.positions[1] += atoms.cell.array[0]
    wrapped_energy, wrapped_forces = evaluate(wrapped)
    np.testing.assert_allclose(wrapped_energy, energy, atol=1e-10)
    np.testing.assert_allclose(wrapped_forces, forces, atol=1e-10)

    order = [2, 0, 1]
    permuted_energy, permuted_forces = evaluate(atoms[order])
    np.testing.assert_allclose(permuted_energy, energy, atol=1e-10)
    np.testing.assert_allclose(permuted_forces, forces[order], atol=1e-10)

    step = 1e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[1, 0] += step
    minus.positions[1, 0] -= step
    plus_energy, _ = evaluate(plus)
    minus_energy, _ = evaluate(minus)
    np.testing.assert_allclose(
        forces[1, 0], -(plus_energy - minus_energy) / (2 * step), atol=2e-6,
    )
