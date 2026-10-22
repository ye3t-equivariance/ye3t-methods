"""The canonical shell image through public scalar fitting and saved ASE models."""

from copy import deepcopy
import json
import hashlib

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ye3t import YE3TRepresentation, couplings
from ye3t_methods import Basis, LinearModel
from ye3t_methods.shell_linear import shell_geometry


def config(ranks=(1, 2, 3, 4), shells=3):
    return {
        "metadata": {"schema": "ye3t_config_v1", "name": "shell_linear_test"},
        "representation": {"group": "O3", "ranks": list(ranks), "factorization": "cauchy", "subspace": "full",
            "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
            "uncoupled_factor_inputs": {"eta_count_per_rank": {n: 1 for n in ranks}, "l_max_per_rank": {n: 1 for n in ranks}},
            "intermediates": {"young_kappa": "all_valid", "block_rotation": {"policy": "all_valid"}}},
        "basis": {"single_factors": {"species": ["Ni"], "chemical": {"kind": "explicit"},
            "radial": {"family": "shell_filtered_chebyshev_c2_v1", "cutoff_A": 3.6,
                "shells": {"kind": "softmax_gaussian", "centers": [0.3, 0.55, 0.8] if shells == 3 else [0.4, 0.8], "width": 0.2},
                "orthogonalization": {"scope": "joint_source", "factorization": "cholesky"}}},
            "tensor_product": {"kind": "shell_density"},
            "catalogue": {"ranks": list(ranks), "nmax_per_rank": {n: 1 for n in ranks},
                "lmax_per_rank": {n: 1 for n in ranks}, "max_source_blocks": 2}},
        "runtime": {"device": "cpu", "dtype": "float64", "evaluator": "auto", "neighbors": "ase"},
        "model": {"kind": "linear", "fit": {"solver": "ridge", "alpha": 1e-12,
            "weights": {"energy": 0.1, "forces": 1.0, "stress": 1.0}},
            "reference_energy": {"per_species_E0_eV": {"Ni": 0.0}, "fit_E0": True}},
        "targets": {"energy": "energy", "forces": "forces", "stress": "stress"},
        "validation": {"checks": ["force_fd", "round_trip"]}}


def basis_from(cfg):
    return Basis.from_config(cfg["basis"], representation=YE3TRepresentation.from_config(cfg["representation"]), runtime=cfg["runtime"])


def atoms_fixture():
    return Atoms("Ni4", positions=[[.2,.3,.4], [1.6,.5,.2], [.4,1.9,.7], [.6,.4,2.2]], cell=[4.1,4.3,4.5], pbc=True)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_planning_is_coefficient_free_and_retains_paired_types(monkeypatch):
    cfg = config()
    with monkeypatch.context() as patch:
        patch.setattr(couplings, "compile", lambda *a, **k: pytest.fail("count compiled coefficients"))
        basis = basis_from(cfg)
        count = basis.catalogue.counts()
    assert count["physical_image_rank"] == count["multiplet_count"] > 4
    values = basis.create(atoms_fixture())
    assert values.shape == (4, count["multiplet_count"])
    assert any(any(tuple(k) != (sum(k),) for k in row.as_dict()["compiler_label"]["block_kappas"]) for row in basis.labels)


def test_shell_linear_fit_reload_forces_stress_and_ase(tmp_path):
    cfg = config()
    cfg["runtime"]["compiled_cache_dir"] = str(tmp_path / "cache")
    basis = basis_from(cfg)
    rng = np.random.default_rng(17)
    beta = rng.normal(size=len(basis.labels)) * 0.03
    frames = []
    for index in range(12):
        atoms = atoms_fixture()
        atoms.positions += rng.normal(scale=.1, size=(4,3))
        atoms.set_cell(atoms.cell * (0.95 + .01*index), scale_atoms=True)
        rows = shell_geometry(basis, atoms, True, True)
        atoms.calc = SinglePointCalculator(atoms, energy=float(rows["energy"]@beta)-1.3*len(atoms),
            forces=(rows["forces"]@beta).reshape(-1,3), stress=rows["stress"]@beta)
        frames.append(atoms)
    model = LinearModel(basis).fit(frames, config=cfg)
    check = frames[-1].copy()
    check.calc = model.ase_calculator()
    np.testing.assert_allclose(check.get_forces(), frames[-1].get_forces(), atol=2e-7)
    np.testing.assert_allclose(check.get_stress(), frames[-1].get_stress(), atol=2e-8)
    np.testing.assert_allclose(check.get_potential_energy(), frames[-1].get_potential_energy(), atol=2e-7)
    path = model.write(tmp_path / "model.ye3t.json")
    assert str(tmp_path) not in json.loads(path.read_text())["construction"]["runtime"].values()
    moved = tmp_path / "relocated" / path.name
    moved.parent.mkdir()
    path.rename(moved)
    restored = LinearModel.read(moved)
    copy = check.copy()
    copy.calc = restored.ase_calculator()
    np.testing.assert_allclose(copy.get_forces(), check.get_forces(), atol=1e-12)
    np.testing.assert_allclose(copy.get_stress(), check.get_stress(), atol=1e-12)
    from ase.md.verlet import VelocityVerlet
    from ase import units
    VelocityVerlet(copy, timestep=.1*units.fs).run(3)
    assert np.isfinite(copy.get_potential_energy())
    changed = json.loads(moved.read_text())
    changed["beta"][0] += 1
    moved.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash"):
        LinearModel.read(moved)
    changed = json.loads(moved.read_text())
    changed["compiled_catalogue"]["self_hash"] = "0" * 64
    body = {k: v for k, v in changed.items() if k != "self_hash"}
    changed["self_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    moved.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="canonical construction"):
        LinearModel.read(moved)


def test_shell_linear_rows_symmetry_and_finite_differences(monkeypatch):
    basis = basis_from(config())
    atoms = atoms_fixture()
    reference = shell_geometry(basis, atoms, True, True)
    angle = .41
    rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    moved = atoms.copy()
    moved.positions = atoms.positions @ rotation.T
    moved.set_cell(atoms.cell.array @ rotation.T)
    np.testing.assert_allclose(basis.create(moved), reference["values"], rtol=3e-10, atol=1e-12)
    moved.positions *= -1
    moved.set_cell(-moved.cell.array)
    np.testing.assert_allclose(basis.create(moved), reference["values"], rtol=3e-10, atol=1e-12)
    order = [2,0,3,1]
    np.testing.assert_allclose(basis.create(atoms[order]), reference["values"][order], atol=1e-12)
    h = 2e-5
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[0,1] += h
    minus.positions[0,1] -= h
    fd = -(shell_geometry(basis, plus)["energy"] - shell_geometry(basis, minus)["energy"]) / (2*h)
    np.testing.assert_allclose(reference["forces"][1], fd, rtol=2e-6, atol=2e-8)
    for axis in range(6):
        strain = np.zeros((3,3))
        i,j = ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))[axis]
        strain[i,j] = 1 if i==j else .5
        strain[j,i] = strain[i,j]
        plus, minus = atoms.copy(), atoms.copy()
        plus.set_cell(atoms.cell.array @ (np.eye(3)+h*strain), scale_atoms=True)
        minus.set_cell(atoms.cell.array @ (np.eye(3)-h*strain), scale_atoms=True)
        fd = (shell_geometry(basis, plus)["energy"]-shell_geometry(basis, minus)["energy"])/(2*h*atoms.get_volume())
        np.testing.assert_allclose(reference["stress"][axis], fd, rtol=3e-6, atol=2e-8)
    source = basis._descriptor
    original = source.geometry_values
    def permuted(*args):
        edge, density = original(*args)
        offsets, columns = 0, {}
        for channel in source.channels:
            columns[channel["radial_channel"], channel["l"]] = list(range(offsets, offsets+2*channel["l"]+1))
            offsets += 2*channel["l"]+1
        indices = []
        for channel in source.channels:
            indices.extend(columns[(channel["radial_channel"]+1)%3, channel["l"]])
        return edge[:,indices], density[:,indices]
    monkeypatch.setattr(source, "geometry_values", permuted)
    actual = shell_geometry(basis, atoms, True, True)
    for key in reference:
        np.testing.assert_allclose(actual[key], reference[key], rtol=3e-10, atol=1e-12)


def test_invalid_shell_energy_sectors_fail_before_compile():
    cfg = config()
    cfg["representation"]["source_sectors"] = {"young_lambda":"all", "L":[0], "parity":"even"}
    with pytest.raises(ValueError, match="shell-symmetric"):
        basis_from(cfg)


def test_shell_catalogue_cannot_exceed_representation_inputs():
    cfg = config()
    cfg["basis"]["catalogue"]["lmax_per_rank"][2] = 2
    with pytest.raises(ValueError, match="input capacity"):
        basis_from(cfg)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_shell_rows_match_tensor_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires optional CUDA hardware")
    cfg = config()
    reference_cfg = deepcopy(cfg)
    reference_cfg["runtime"]["evaluator"] = "torch"
    reference = shell_geometry(basis_from(reference_cfg), atoms_fixture(), True, True)
    cfg["runtime"]["device"] = device
    basis = basis_from(cfg)
    actual = shell_geometry(basis, atoms_fixture(), True, True)
    for key in reference:
        np.testing.assert_allclose(actual[key], reference[key], rtol=3e-10, atol=1e-12)
    assert basis._descriptor.programs["0"].last_execution == "native_joint_polynomial"


def test_cuda_saved_shell_model_reloads_on_cpu(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires optional CUDA hardware for the saved-device fixture")
    cfg = config()
    cfg["runtime"]["device"] = "cuda"
    model = LinearModel(basis_from(cfg), reference_energies={"Ni":-1.})
    model._fitted = {"beta":np.linspace(-.01,.01,len(model.labels)), "fit_metadata":{"target":"manufactured"}}
    path = model.write(tmp_path/'cuda.ye3t.json')
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("reload required CUDA initialization"))
        restored = LinearModel.read(path)
        cpu = atoms_fixture()
        cpu.calc = restored.ase_calculator()
        force = cpu.get_forces()
    assert restored.basis.resolution.to_dict()["runtime"]["device"] == "cpu"
    gpu = atoms_fixture()
    gpu.calc = restored.ase_calculator(device="cuda")
    np.testing.assert_allclose(gpu.get_forces(), force, rtol=3e-10, atol=1e-12)
    reference = atoms_fixture()
    reference.calc = restored.ase_calculator(evaluator="torch")
    np.testing.assert_allclose(reference.get_forces(), force, rtol=3e-10, atol=1e-12)


def test_species_offsets_and_absent_neighbors():
    cfg = config(ranks=(1, 2), shells=2)
    cfg["basis"]["single_factors"]["species"] = ["Ni", "Cu"]
    basis = basis_from(cfg)
    atoms = Atoms("NiCu", positions=[[0,0,0], [8,0,0]], cell=[20,20,20])
    rows = shell_geometry(basis, atoms, True, True)
    for value in rows.values():
        np.testing.assert_array_equal(value, np.zeros_like(value))
    atoms.positions[1,0] = 2.1
    rows = shell_geometry(basis, atoms, True, True)
    assert np.linalg.norm(rows["forces"]) > 0
    assert len(rows["energy"]) == 2*len(basis.labels)
    np.testing.assert_allclose(rows["forces"].reshape(2,3,-1).sum(0), 0, atol=1e-14)
