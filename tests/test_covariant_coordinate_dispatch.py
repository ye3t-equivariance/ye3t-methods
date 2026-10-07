"""Ordinary ACE full-M dispatch binds the native coordinate contract."""

import pytest

from ye3t.core.labels import normalize_compact_label
from ye3t_methods.atomistic.cache import descriptor_artifact_cache_key
from ye3t_methods.atomistic.equivariant_calc.descriptor_sets import (
    DescriptorGenerationSettings,
    _compiled_scalar_coordinate_library,
    _covariant_coordinate_compiler_request,
    compile_descriptor_artifacts,
)


def _settings(M_values):
    return DescriptorGenerationSettings(
        ranks=(3,), basis_type="no_charge", elems=("Ni",),
        nmax=(1,), lmax=(1,), lmin=(0,),
        L_R=1, M_R_values=M_values, parity_filter="none",
    )


def _label():
    return normalize_compact_label({
        "n_tuple": [1, 1, 1], "l_tuple": [1, 1, 1],
        "internal_Ls": [1], "L_R": 1, "tree_type": "balanced",
        "basis_key": ["sym", 1, 0],
    })


def test_covariant_coordinate_request_and_cache_bind_native_contract():
    settings = _settings((-1, 0, 1))
    label = _label()
    request = _covariant_coordinate_compiler_request(None)
    assert request["options"]["coordinate_contract"] == "native_compiled"
    library = _compiled_scalar_coordinate_library((label,), settings, None)
    metadata = library.metadata["scalar_coordinate_compiler"]
    assert metadata["schema"] == "ye3t_ace_coordinate_library_v1"
    assert metadata["options"]["coordinate_contract"] == "native_compiled"
    assert all(len(library.data[M][3][label.full_key()]["ms_combs"]) == 2
               for M in (-1, 0, 1))
    key = descriptor_artifact_cache_key(
        settings, (label,), basis_mode=None,
        exact_primitive_timeout_seconds=None, center_mu_values=None,
        restrict_neighbor_mu=None, max_variants_per_label=None,
        scalar_coordinate_compiler=request,
    )
    assert key.scalar_coordinate_compiler.startswith("ace_coordinate_compiler_v1:")
    assert "native_compiled" in key.scalar_coordinate_compiler


def test_covariant_coordinate_rejects_partial_axis_and_ignored_options():
    with pytest.raises(ValueError, match="complete magnetic axis"):
        compile_descriptor_artifacts(
            _settings((0,)), compact_labels=(_label(),),
            use_descriptor_cache=False,
        )
    for options in ({"membership_mode": "exact"},
                    {"coordinate_contract": "pace_compatible_exact"},
                    {"constructor_backend": "cpp"}):
        with pytest.raises(ValueError, match="Covariant ACE"):
            _compiled_scalar_coordinate_library(
                (_label(),), _settings((-1, 0, 1)),
                {"mode": "missing_only", "options": options},
            )
