from itertools import product

import numpy as np
import pytest
import torch
from ase import Atoms

from ye3t.core.basis import ExactACELabeler
from ye3t_methods.atomistic import YE3TDescriptors
from ye3t_methods.atomistic.ace.linear_ace import LinearACEScalarCalculator, LinearACEScalarModelBundle
from ye3t_methods.atomistic._record import record_replace
from ye3t_methods.atomistic.tagged_cauchy_image import TaggedCauchyImageLinearModel
from ye3t_methods import Basis


def test_default_density_rank_three_lmax_two_labels_are_real_scalars():
    basis = Basis(
        elements=["Ni"], source="density", cutoff=3.5,
        max_rank=3, nmax=1, lmax=2,
    )
    labels = basis._descriptor.compact_labels
    assert any(label.rank == 3 and label.l_tuple == (2, 2, 2) for label in labels)
    assert all(sum(label.l_tuple) % 2 == 0 for label in labels)
    atoms = Atoms(
        "Ni4",
        positions=((0.0, 0.0, 0.0), (1.4, 0.2, 0.1),
                   (-0.3, 1.2, 0.4), (0.5, -0.2, 1.5)),
    )
    values = basis.create(atoms)
    assert values.dtype == np.dtype("float64")
    assert np.isfinite(values).all()


@pytest.mark.parametrize(
    ("l_tuple", "parity_filter"),
    (
        ((1, 1, 1), "none"),
        ((1, 1, 2), "natural"),
        ((1, 2, 2), "none"),
        ((2, 2, 2), "natural"),
    ),
)
@pytest.mark.parametrize("spherical_backend", ("real", "complex"))
def test_rank_three_scalar_direct_factorized_and_force_paths_agree(
    monkeypatch, l_tuple, parity_filter, spherical_backend,
):
    labeler = ExactACELabeler((1, 2, 3), l_tuple, strict_target_validation=False)
    labels = labeler.compact_labels_for_target(0)
    assert len(labels) == 1
    config = {
        "elements": ["Ni"],
        "type_map": {"Ni": 0},
        "cutoff": 3.5,
        "ranks": [3],
        "nmax": [3],
        "lmax": [2],
        "lmin": [0],
        "L_R": 0,
        "M_R_values": [0],
        "basis_type": "no_charge",
        "k_o_max": 0,
        "k_max": [0],
        "manual_labels": labels,
        "parity_filter": parity_filter,
        "max_variants_per_label": 1,
        "site_basis": {"mode": "explicit", "rc": [3.5], "lmbda": [0.25]},
        "backend": "pytorch",
    }
    descriptor = YE3TDescriptors.ace(config)
    if spherical_backend == "complex":
        config["site_basis_config"] = record_replace(
            descriptor.site_basis_config, spherical_backend="complex",
        )
        descriptor = YE3TDescriptors.ace(config)
    assert descriptor.site_basis_config.spherical_backend == spherical_backend
    atoms = Atoms(
        "Ni4",
        positions=((0.0, 0.0, 0.0), (1.4, 0.2, 0.1),
                   (-0.3, 1.2, 0.4), (0.5, -0.2, 1.5)),
        cell=(8.0, 8.0, 8.0),
        pbc=False,
    )
    bundle = LinearACEScalarModelBundle(
        settings=descriptor.settings,
        site_basis_config=descriptor.site_basis_config,
        descriptor_specs=descriptor.descriptor_specs,
        weight=np.ones(len(descriptor.descriptor_specs)),
        bias=0.0,
        basis_mode=None,
        fit_method="scalar_reality_regression",
    )
    results = {}
    for policy in ("off", "require"):
        monkeypatch.setenv("YE3T_ACE_FACTORIZED_DESCRIPTOR_RUNTIME", policy)
        features = descriptor.create(atoms)
        report = descriptor.calculator.evaluator.backend_report()
        assert features.dtype == torch.float64
        assert torch.isfinite(features).all()
        if policy == "require":
            assert report["factorized_descriptor_count"] == len(descriptor.descriptor_specs)
        method_results = {}
        for force_method, real_reverse in (
            ("autograd", "auto"),
            ("analytic_factorized", "force"),
            ("analytic_factorized", "off"),
        ):
            monkeypatch.setenv("YE3T_ACE_REAL_FACTORIZED_REVERSE", real_reverse)
            evaluated = atoms.copy()
            evaluated.calc = LinearACEScalarCalculator(
                bundle, 3.5, {"Ni": 0}, force_method=force_method,
            )
            method_results[(force_method, real_reverse)] = (
                evaluated.get_potential_energy(), evaluated.get_forces(),
            )
        results[policy] = (features.detach().cpu().numpy(), method_results)
    np.testing.assert_allclose(results["off"][0], results["require"][0], atol=1e-10, rtol=1e-10)
    reference = results["off"][1][("autograd", "auto")]
    for methods in (results["off"][1], results["require"][1]):
        for direct, factorized in zip(reference, methods[("analytic_factorized", "force")]):
            np.testing.assert_allclose(direct, factorized, atol=1e-9, rtol=1e-9)
        for direct, factorized in zip(reference, methods[("analytic_factorized", "off")]):
            np.testing.assert_allclose(direct, factorized, atol=1e-9, rtol=1e-9)
    for direct, factorized in zip(reference, results["require"][1][("autograd", "auto")]):
        np.testing.assert_allclose(direct, factorized, atol=1e-10, rtol=1e-10)


def test_rank_three_odd_scalar_real_and_complex_sources_agree(monkeypatch):
    labels = ExactACELabeler(
        (1, 2, 3), (1, 1, 1), strict_target_validation=False,
    ).compact_labels_for_target(0)
    config = {
        "elements": ["Ni"], "type_map": {"Ni": 0}, "cutoff": 3.5,
        "ranks": [3], "nmax": [3], "lmax": [1], "lmin": [0],
        "L_R": 0, "M_R_values": [0], "basis_type": "no_charge",
        "k_o_max": 0, "k_max": [0], "manual_labels": labels,
        "parity_filter": "none", "max_variants_per_label": 1,
        "site_basis": {"mode": "explicit", "rc": [3.5], "lmbda": [0.25]},
        "backend": "pytorch",
    }
    real_descriptor = YE3TDescriptors.ace(config)
    config["site_basis_config"] = record_replace(
        real_descriptor.site_basis_config, spherical_backend="complex",
    )
    complex_descriptor = YE3TDescriptors.ace(config)
    atoms = Atoms(
        "Ni4",
        positions=((0.0, 0.0, 0.0), (1.4, 0.2, 0.1),
                   (-0.3, 1.2, 0.4), (0.5, -0.2, 1.5)),
        cell=(8.0, 8.0, 8.0),
    )
    monkeypatch.setenv("YE3T_ACE_FACTORIZED_DESCRIPTOR_RUNTIME", "require")
    results = []
    for descriptor in (real_descriptor, complex_descriptor):
        features = descriptor.create(atoms).detach().cpu().numpy()
        bundle = LinearACEScalarModelBundle(
            settings=descriptor.settings,
            site_basis_config=descriptor.site_basis_config,
            descriptor_specs=descriptor.descriptor_specs,
            weight=np.ones(len(descriptor.descriptor_specs)),
            bias=0.0, basis_mode=None, fit_method="scalar_reality_regression",
        )
        evaluated = atoms.copy()
        evaluated.calc = LinearACEScalarCalculator(
            bundle, 3.5, {"Ni": 0}, force_method="analytic_factorized",
        )
        results.append((features, evaluated.get_potential_energy(), evaluated.get_forces()))
    for actual, expected in zip(results[0], results[1]):
        np.testing.assert_allclose(actual, expected, atol=1.0e-10, rtol=1.0e-10)


def test_rank_three_lmax_two_compiler_label_sweep_values_forces_and_parity(monkeypatch):
    from ye3t.couplings import count

    labels = tuple(
        label
        for input_Ls in product(range(3), repeat=3)
        for label in count(
            content=(1, 2, 3), input_Ls=input_Ls, target_L=0,
            target_permutation="trivial", carrier="ACE_density",
        ).labels_for_target(0)
    )
    assert len(labels) == 15
    assert len(set(labels)) == 15
    config = {
        "elements": ["Ni"], "type_map": {"Ni": 0}, "cutoff": 3.5,
        "ranks": [3], "nmax": [3], "lmax": [2], "lmin": [0],
        "L_R": 0, "M_R_values": [0], "basis_type": "no_charge",
        "k_o_max": 0, "k_max": [0], "manual_labels": labels,
        "parity_filter": "none", "max_variants_per_label": 1,
        "site_basis": {"mode": "explicit", "rc": [3.5], "lmbda": [0.25]},
        "backend": "pytorch",
    }
    real_descriptor = YE3TDescriptors.ace(config)
    complex_descriptor = YE3TDescriptors.ace({
        **config,
        "site_basis_config": record_replace(
            real_descriptor.site_basis_config, spherical_backend="complex",
        ),
    })
    descriptors = {"real": real_descriptor, "complex": complex_descriptor}
    atoms = Atoms(
        "Ni5",
        positions=((0.0, 0.0, 0.0), (1.4, 0.2, 0.1),
                   (-0.3, 1.2, 0.4), (0.5, -0.2, 1.5),
                   (-0.7, -0.9, -0.6)),
        cell=(8.0, 8.0, 8.0), pbc=False,
    )
    axis = np.array((1.0, 2.0, 3.0)) / np.sqrt(14.0)
    skew = np.array(((0.0, -axis[2], axis[1]),
                     (axis[2], 0.0, -axis[0]),
                     (-axis[1], axis[0], 0.0)))
    angle = 0.37
    rotation = (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * skew
    )
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    inverted = atoms.copy()
    inverted.positions = -atoms.positions
    geometries = {"original": atoms, "rotated": rotated, "inverted": inverted}
    values = {}
    for source, descriptor in descriptors.items():
        assert len(descriptor.descriptor_specs) == len(labels)
        for policy in ("off", "require"):
            monkeypatch.setenv("YE3T_ACE_FACTORIZED_DESCRIPTOR_RUNTIME", policy)
            for geometry, structure in geometries.items():
                result = descriptor.create(structure)
                assert result.dtype == torch.float64
                values[(source, policy, geometry)] = result.detach().cpu().numpy()
                assert np.isfinite(values[(source, policy, geometry)]).all()
            if policy == "require":
                assert descriptor.calculator.evaluator.backend_report()["factorized_descriptor_count"] == len(labels)
    parity = np.array([(-1) ** sum(label.l_tuple) for label in labels])
    for source in descriptors:
        for geometry in geometries:
            np.testing.assert_allclose(
                values[(source, "require", geometry)],
                values[(source, "off", geometry)], atol=1e-10, rtol=1e-10,
            )
        base = values[(source, "off", "original")]
        assert np.min(np.max(np.abs(base), axis=0)) > 1e-8
        np.testing.assert_allclose(values[(source, "off", "rotated")], base, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(values[(source, "off", "inverted")], base * parity, atol=1e-10, rtol=1e-10)
    for policy in ("off", "require"):
        for geometry in geometries:
            np.testing.assert_allclose(
                values[("real", policy, geometry)], values[("complex", policy, geometry)],
                atol=1e-10, rtol=1e-10,
            )

    for label_index in range(len(labels)):
        weight = np.zeros(len(labels))
        weight[label_index] = 1.0
        evaluated = {}
        for source, descriptor in descriptors.items():
            bundle = LinearACEScalarModelBundle(
                settings=descriptor.settings,
                site_basis_config=descriptor.site_basis_config,
                descriptor_specs=descriptor.descriptor_specs,
                weight=weight,
                bias=0.0,
                basis_mode=None,
                fit_method="rank_three_label_sweep",
            )
            for policy, force_method in (("off", "autograd"), ("require", "analytic_factorized")):
                monkeypatch.setenv("YE3T_ACE_FACTORIZED_DESCRIPTOR_RUNTIME", policy)
                structure = atoms.copy()
                structure.calc = LinearACEScalarCalculator(
                    bundle, 3.5, {"Ni": 0}, force_method=force_method,
                )
                evaluated[(source, policy)] = (structure.get_potential_energy(), structure.get_forces())
        reference = evaluated[("real", "off")]
        for result in evaluated.values():
            np.testing.assert_allclose(result[0], reference[0], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(result[1], reference[1], atol=1e-9, rtol=1e-9)


def test_tagged_readout_rejects_imaginary_coefficients():
    class Evaluator:
        species_order = ("Ni",)
        feature_count = 2

    with pytest.raises(ValueError, match="beta.*real-valued"):
        TaggedCauchyImageLinearModel(
            Evaluator(), {"Ni": torch.tensor([1.0 + 0.2j, 2.0])}, {"Ni": 0.0},
        )
