"""Strict PACE export and the ordinary native ASE calculator."""

import os
import numpy as np
import pytest
import torch
import yaml
from ase import Atoms

from ye3t.core.basis import ExactACELabeler
from ye3t_ace import YE3TDescriptors
from ye3t_ace.ace.linear_ace import LinearACEScalarCalculator, LinearACEScalarModelBundle
from ye3t_ace.ace.yace import read_yace
from ye3t_ace.couplings.exact_catalog import ExactCouplingCatalog
from ye3t_ace.equivariant_calc.descriptor_sets import DescriptorGenerationSettings
from ye3t_ace.equivariant_calc.site_basis_v2 import SiteBasisConfig
from ye3t_ace.yace_native import YE3TYACENativeCalculator


@pytest.fixture
def strict_scalar_bundle():
    settings = DescriptorGenerationSettings(
        ranks=[1], basis_type="no_charge", elems=["Si"], nmax=[1],
        lmax=[0], lmin=[0], L_R=0, M_R_values=[0],
        max_labels_per_rank=None, tree_type="balanced",
    )
    catalog = ExactCouplingCatalog.from_settings(
        settings, center_mu_values=[0], restrict_neighbor_mu=[0],
        max_variants_per_label=None, basis_mode="exact",
    )
    descriptors = tuple(catalog.descriptor_collection.specs_by_M[0])
    assert descriptors
    bundle = LinearACEScalarModelBundle(
        settings=settings,
        site_basis_config=SiteBasisConfig(
            rc=[5.0], lmbda=[0.5723], nradmax=1, lmax=0,
            possible_types=(0,), radial_basis="PACE_ChebExpCos",
            pace_cutoff_width=[0.8], pace_spline_spacing=[0.1],
            spherical_backend="complex", spherical_normalization="pace_y00_one",
            source_backend="torch", atomic_base_normalization="none",
            factor_normalization="none", dtype=torch.float64,
            complex_dtype=torch.complex128,
        ),
        descriptor_specs=descriptors,
        weight=np.asarray([0.5] * len(descriptors)),
        bias=0.125, basis_mode="exact", fit_method="ridge_normal_equations",
        fit_metadata={
            "reference_energy_targets": {
                "enabled": True, "reference_energies": {"Si": -3.0},
            },
        },
    )
    return bundle


def test_strict_yace_export_preserves_coefficients_and_reference(tmp_path, strict_scalar_bundle):
    bundle = strict_scalar_bundle
    descriptors = bundle.descriptor_specs
    path = bundle.export_lammps(tmp_path / "linear_si.yace", elements=["Si"], format="yace")
    payload = read_yace(path, compatibility="lammps_pace_linear_v1")
    root = yaml.compose(path.read_text(encoding="utf-8"))
    assert {key.value for key, _ in root.value} == {
        "elements", "E0", "embeddings", "bonds", "functions", "deltaSplineBins",
    }
    assert payload["elements"] == ["Si"]
    assert payload["E0"] == pytest.approx([-2.875])
    assert len(payload["functions"][0]) == len(descriptors)
    for function, descriptor in zip(payload["functions"][0], descriptors):
        assert function.ctildes == pytest.approx(
            tuple(float((0.5 * complex(value)).real) for value in descriptor.coeffs)
        )


def test_strict_yace_native_ase_matches_python_and_finite_difference(tmp_path, strict_scalar_bundle):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    path = strict_scalar_bundle.export_lammps(
        tmp_path / "linear_si.yace", elements=["Si"], format="yace",
    )
    atoms = Atoms(
        "Si4", positions=((0.0, 0.0, 0.0), (1.9, 0.2, 0.1),
                          (-0.4, 2.1, 0.3), (0.5, -0.3, 2.2)),
        cell=(10.0, 10.0, 10.0), pbc=False,
    )
    reference = atoms.copy()
    reference.calc = LinearACEScalarCalculator(
        strict_scalar_bundle, cutoff=5.0, type_map={"Si": 0},
        force_method="autograd", reference_energies={"Si": -3.0},
    )
    native = atoms.copy()
    native.calc = YE3TYACENativeCalculator.from_artifact(path, native_library=library)
    try:
        np.testing.assert_allclose(native.get_potential_energy(), reference.get_potential_energy(), atol=1.0e-9)
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(), atol=1.0e-8)
        original = native.get_potential_energy()
        step = 1.0e-5
        native.positions[0, 0] += step
        plus = native.get_potential_energy()
        native.positions[0, 0] -= 2.0 * step
        minus = native.get_potential_energy()
        native.positions[0, 0] += step
        assert plus != pytest.approx(original, abs=1.0e-12)
        np.testing.assert_allclose(
            -(plus - minus) / (2.0 * step), native.get_forces()[0, 0], atol=2.0e-5,
        )
    finally:
        native.calc.close()


def _rank_three_bundle(l_tuple):
    labels = ExactACELabeler(
        (1, 2, 3), l_tuple, strict_target_validation=False,
    ).compact_labels_for_target(0)
    site_basis = SiteBasisConfig(
        rc=[5.0], lmbda=[0.5723], nradmax=3, lmax=max(l_tuple),
        possible_types=(0,), radial_basis="PACE_ChebExpCos",
        pace_cutoff_width=[0.8], pace_spline_spacing=[0.1],
        spherical_backend="complex", spherical_normalization="pace_y00_one",
        source_backend="torch", atomic_base_normalization="none",
        factor_normalization="none", dtype=torch.float64,
        complex_dtype=torch.complex128,
    )
    descriptor = YE3TDescriptors.ace({
        "elements": ["Si"], "type_map": {"Si": 0}, "cutoff": 5.0,
        "ranks": [3], "nmax": [3], "lmax": [max(l_tuple)], "lmin": [0],
        "L_R": 0, "M_R_values": [0], "basis_type": "no_charge",
        "k_o_max": 0, "k_max": [0], "manual_labels": labels,
        "parity_filter": "none", "max_variants_per_label": 1,
        "site_basis_config": site_basis, "backend": "pytorch",
    })
    return LinearACEScalarModelBundle(
        settings=descriptor.settings, site_basis_config=site_basis,
        descriptor_specs=descriptor.descriptor_specs,
        weight=np.ones(len(descriptor.descriptor_specs)), bias=0.0,
        basis_mode=None, fit_method="scalar_reality_regression",
    )


def test_strict_yace_rejects_odd_rank_three_scalar(tmp_path):
    bundle = _rank_three_bundle((1, 1, 1))
    with pytest.raises(ValueError, match="material imaginary C-tilde"):
        bundle.export_lammps(tmp_path / "odd_scalar.yace", elements=["Si"], format="yace")


@pytest.mark.parametrize("l_tuple", ((1, 1, 2), (2, 2, 2)))
def test_rank_three_lmax_two_native_ase_matches_python(tmp_path, l_tuple):
    library = os.environ.get("YE3T_TAGGED_C_API_LIBRARY")
    if not library or not os.path.isfile(library):
        pytest.skip("requires YE3T_TAGGED_C_API_LIBRARY pointing to the compiled native library")
    bundle = _rank_three_bundle(l_tuple)
    path = bundle.export_lammps(tmp_path / "rank_three.yace", elements=["Si"], format="yace")
    atoms = Atoms(
        "Si5", positions=((0.0, 0.0, 0.0), (1.9, 0.2, 0.1),
                          (-0.4, 2.1, 0.3), (0.5, -0.3, 2.2),
                          (-1.3, -1.1, -0.6)),
        cell=(12.0, 12.0, 12.0), pbc=False,
    )
    reference = atoms.copy()
    reference.calc = LinearACEScalarCalculator(
        bundle, cutoff=5.0, type_map={"Si": 0}, force_method="autograd",
    )
    native = atoms.copy()
    native.calc = YE3TYACENativeCalculator.from_artifact(path, native_library=library)
    try:
        assert abs(reference.get_potential_energy()) > 1.0e-10
        np.testing.assert_allclose(
            native.get_potential_energy(), reference.get_potential_energy(), atol=1.0e-8, rtol=1.0e-8,
        )
        np.testing.assert_allclose(native.get_forces(), reference.get_forces(), atol=1.0e-7, rtol=1.0e-7)
    finally:
        native.calc.close()
