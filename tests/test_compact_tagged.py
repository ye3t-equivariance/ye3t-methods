import numpy as np
import pytest
import torch

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ye3t_methods import Basis, LinearModel
from ye3t_ace.tagged_cauchy_image import TaggedCauchyImageLinearModel


def test_tagged_fit_save_load_ase_and_compiler_labels(tmp_path):
    basis = Basis(
        elements=["Ta"], source="tagged_cauchy_image", cutoff=4.8,
        tensor_order=4, tag_counts=(0, 2), radial_degrees=(0,),
        angular_degree=1, backend="reference",
    )
    assert len(basis.labels) > 0
    assert len(basis.labels) == len(basis._descriptor.feature_keys)
    assert basis._descriptor.metadata["tagged_cauchy_image_preflight"].exact_image_dimension == len(basis.labels)
    catalogue = basis._descriptor.metadata["tagged_cauchy_image_compiled"].payload
    nontrivial = {
        record["raw_opportunity_id"] for record in catalogue["raw_coordinate_labels"]
        if record["tag_count"] == 2 and tuple(record["tag_kappa"]) == (1, 1)
        and tuple(record["role_kappa"]) == (2, 1, 1)
    }
    assert nontrivial
    assert any(
        any(item["raw_opportunity_id"] in nontrivial for item in label.as_dict()["compiler_coordinate_provenance"]["contributors"])
        for label in basis.labels
    )
    nontrivial_index = next(
        label.feature_index for label in basis.labels
        if any(item["raw_opportunity_id"] in nontrivial
               for item in label.as_dict()["compiler_coordinate_provenance"]["contributors"])
    )
    identity_atoms = Atoms(
        "Ta3", positions=[[0, 0, 0], [1.2, 0.1, 0], [-0.2, 1.3, 0.3]],
        cell=[12, 12, 12], pbc=False,
    )
    public_columns = basis.create(identity_atoms)
    for index in sorted({0, nontrivial_index, len(basis.labels) - 1}):
        one_hot = torch.zeros(len(basis.labels), dtype=torch.float64)
        one_hot[index] = 1.0
        selected = TaggedCauchyImageLinearModel(
            basis._descriptor.metadata["tagged_cauchy_image_evaluator"],
            {"Ta": one_hot}, {"Ta": 0.0},
        )
        energy, _forces, _virial, atomic = selected.energy_forces_virial(
            identity_atoms.positions, [0] * len(identity_atoms),
            cell=identity_atoms.cell.array, pbc=identity_atoms.pbc,
        )
        np.testing.assert_allclose(atomic, public_columns[:, index], atol=1e-12)
        np.testing.assert_allclose(energy, public_columns[:, index].sum(), atol=1e-12)

    beta = torch.linspace(0.13, 0.13 * len(basis.labels), len(basis.labels), dtype=torch.float64)
    oracle = TaggedCauchyImageLinearModel(
        basis._descriptor.metadata["tagged_cauchy_image_evaluator"],
        {"Ta": beta}, {"Ta": -0.16},
    )
    structures = []
    positions = (
        [[0, 0, 0], [1.2, 0.1, 0], [-0.2, 1.3, 0.3]],
        [[0, 0, 0], [1.4, -0.2, 0.1], [0.3, 1.0, 0.7]],
        [[0, 0, 0], [1.0, 0.4, -0.2], [-0.5, 1.1, 0.8]],
        [[0, 0, 0], [1.5, 0.2, 0.3], [-0.4, 0.9, 1.0]],
    )
    for coordinates in positions:
        atoms = Atoms("Ta3", positions=coordinates, cell=[12, 12, 12], pbc=False)
        atoms.calc = oracle.ase_calculator()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        structures.append(atoms)

    model = LinearModel(basis).fit(structures, regularization=1e-12, stress_weight=0.1)
    for atoms in structures:
        check = atoms.copy()
        check.calc = model.ase_calculator()
        np.testing.assert_allclose(check.get_potential_energy(), atoms.get_potential_energy(), atol=1e-8)
        np.testing.assert_allclose(check.get_forces(), atoms.get_forces(), atol=1e-8)
        np.testing.assert_allclose(check.get_stress(), atoms.get_stress(), atol=1e-8)

    path = model.write(tmp_path / "ta_tagged")
    assert path.name.endswith(".ye3t.json")
    loaded = LinearModel.read(path)
    assert len(loaded.labels) == len(model.labels)
    assert loaded.describe(0, format="latex") == model.describe(0, format="latex")
    check = structures[0].copy()
    check.calc = loaded.ase_calculator()
    np.testing.assert_allclose(check.get_forces(), structures[0].get_forces(), atol=1e-8)
    deployed = loaded.export_lammps(tmp_path / "deployed.ye3t.json")
    assert deployed.is_file()
    assert deployed.name == "deployed.ye3t.json"
    deployed_model = LinearModel.read(deployed)
    check.calc = deployed_model.ase_calculator()
    np.testing.assert_allclose(check.get_forces(), structures[0].get_forces(), atol=1e-8)

    periodic = structures[0].copy()
    periodic.pbc = True
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
    np.testing.assert_allclose(stress[5], (energies[0] - energies[1]) / (2e-6 * volume), atol=1e-7)

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
    reloaded = LinearModel.read(ard.write(tmp_path / "tagged_ard"))
    np.testing.assert_allclose(
        reloaded.predict_uncertainty(structures[0])["atomic_energy_std_eV"],
        uncertainty["atomic_energy_std_eV"], rtol=1e-11,
    )
    candidate = structures[0].copy()
    candidate.calc = reloaded.ase_calculator()
    assert np.isfinite(candidate.get_potential_energy())
