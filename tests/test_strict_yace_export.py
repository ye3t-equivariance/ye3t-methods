"""A representable scalar linear model must survive strict PACE export."""

import numpy as np
import pytest
import torch
import yaml

from ye3t_ace.ace.linear_ace import LinearACEScalarModelBundle
from ye3t_ace.ace.yace import read_yace
from ye3t_ace.couplings.exact_catalog import ExactCouplingCatalog
from ye3t_ace.equivariant_calc.descriptor_sets import DescriptorGenerationSettings
from ye3t_ace.equivariant_calc.site_basis_v2 import SiteBasisConfig


def test_strict_yace_export_preserves_coefficients_and_reference(tmp_path):
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
