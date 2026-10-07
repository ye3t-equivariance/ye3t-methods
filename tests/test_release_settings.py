"""Serialized exact models must not silently accept active unknown options."""

import pytest

from ye3t_methods.atomistic.ace.descriptors import ACEDescriptor, YE3TDescriptors
from ye3t_methods.atomistic.equivariant_calc.descriptor_sets import DescriptorGenerationSettings
from ye3t_methods.atomistic.equivariant_calc.site_basis_serialization import (
    deserialize_site_basis_config,
    serialize_site_basis_config,
)
from ye3t_methods.atomistic.equivariant_calc.site_basis_v2 import SiteBasisConfig


def test_descriptor_settings_reject_active_unknown_option():
    settings = DescriptorGenerationSettings(
        ranks=(1,), basis_type="no_charge", elems=("H",), nmax=(1,),
        lmax=(0,), lmin=(0,), L_R=0,
    )
    payload = settings.as_dict()
    payload["unreviewed_projection"] = {"width": 2}
    with pytest.raises(ValueError, match="Unsupported descriptor settings"):
        DescriptorGenerationSettings.from_dict(payload)
    payload["unreviewed_projection"] = None
    assert DescriptorGenerationSettings.from_dict(payload) == settings


def test_site_basis_settings_reject_active_unknown_option():
    config = SiteBasisConfig(rc=(3.0,), lmbda=(0.3,), nradmax=1, lmax=0)
    payload = serialize_site_basis_config(config)
    payload["unreviewed_projection"] = {"width": 2}
    with pytest.raises(ValueError, match="Unsupported site-basis settings"):
        deserialize_site_basis_config(payload)
    payload["unreviewed_projection"] = None
    restored = deserialize_site_basis_config(payload)
    assert restored.rc == config.rc
    assert restored.nradmax == config.nradmax


def test_public_descriptor_inputs_reject_unsupported_tensor_options():
    with pytest.raises(ValueError, match="Unsupported descriptor settings"):
        ACEDescriptor.from_config({"tensor_projection": {"width": 2}})
    with pytest.raises(ValueError, match="Unsupported descriptor settings"):
        YE3TDescriptors.ye3t_basis({"tensor_projection": {"width": 2}})
