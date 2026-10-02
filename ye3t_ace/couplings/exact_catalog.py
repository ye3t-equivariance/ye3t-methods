
from collections.abc import Mapping

import torch
from torch import nn

from ye3t_ace.cache import DescriptorBuildCache
from ye3t_ace.couplings.generalized import generate_library_for_labels
from ye3t_ace.equivariant_calc import (
    ACECovariantEvaluator,
    DescriptorCollection,
    DescriptorGenerationSettings,
    GeneralizedCouplingLibrary,
    SiteBasisConfig,
    build_descriptor_specs_from_settings,
    compile_descriptor_artifacts,
    complex_multiplet_to_real_tesseral,
    edge_vectors_from_positions,
)
from ye3t_ace._record import recordclass


YE3T_ACE_RUNTIME_CACHE_KEY = "_ye3t_ace_runtime_cache"


def _truncate_compact_labels_by_internal_lmax(compact_labels, internal_lmax_trunc):
    if internal_lmax_trunc is None:
        return list(compact_labels)
    out = []
    for label in compact_labels:
        internal = getattr(label, "internal_Ls", None)
        if internal is None:
            try:
                internal = tuple(label[2])
            except Exception:
                internal = tuple()
        if all(int(L) <= int(internal_lmax_trunc) for L in internal):
            out.append(label)
    return out


@recordclass(
    (
        'settings',
        'compact_labels',
        'library',
        'descriptor_collection',
    ),
    frozen = True,
)
class ExactCouplingCatalog:
    """Compiled exact basis for one output irrep ``L_R``."""

    @property
    def M_values(self):
        return tuple(self.settings.M_R_values)

    @property
    def num_labels(self):
        if not self.M_values:
            return 0
        return len(self.descriptor_collection.specs_by_M[self.M_values[0]])

    @property
    def alpha_LR(self):
        """Multiplicity-space dimension ``alpha_{L_R}``."""
        return self.num_labels

    @classmethod
    def from_settings(
        cls,
        settings,
        *,
        compact_labels = None,
        center_mu_values = None,
        restrict_neighbor_mu = None,
        max_variants_per_label = 1,
        internal_lmax_trunc = None,
        basis_mode = None,
        exact_primitive_timeout_seconds = None,
        descriptor_cache = None,
        use_descriptor_cache = True,
    ):
        """Compile the exact basis, optionally with a primitive reduction."""
        compact_labels, library, collection = compile_descriptor_artifacts(
            settings,
            compact_labels=compact_labels,
            basis_mode=basis_mode,
            exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
            center_mu_values=center_mu_values,
            restrict_neighbor_mu=restrict_neighbor_mu,
            max_variants_per_label=max_variants_per_label,
            descriptor_cache=descriptor_cache,
            use_descriptor_cache=use_descriptor_cache,
        )
        compact_labels = _truncate_compact_labels_by_internal_lmax(compact_labels, internal_lmax_trunc)
        if len(compact_labels) != len(collection.compact_labels):
            lib_data = generate_library_for_labels(compact_labels, M_R_values=settings.M_R_values)
            library = GeneralizedCouplingLibrary(lib_data, L_R=settings.L_R)
            collection = build_descriptor_specs_from_settings(
                compact_labels=compact_labels,
                settings=settings,
                library=library,
                center_mu_values=center_mu_values,
                restrict_neighbor_mu=restrict_neighbor_mu,
                max_variants_per_label=max_variants_per_label,
            )
        return cls(
            settings=settings,
            compact_labels=compact_labels,
            library=library,
            descriptor_collection=collection,
        )


class YE3TGraphDataAdapter:
    def __init__(self, data):
        self.data = data

    def _get(self, name, default = None):
        if isinstance(self.data, Mapping):
            return self.data.get(name, default)
        return getattr(self.data, name, default)

    @property
    def positions(self):
        x = self._get("positions", self._get("pos"))
        if x is None:
            raise KeyError("Expected `positions`/`pos` on the data object.")
        return x

    @property
    def edge_index(self):
        x = self._get("edge_index")
        if x is None:
            raise KeyError("Expected `edge_index` on the data object.")
        return x

    @property
    def shifts(self):
        return self._get("shifts", self._get("edge_cell_shift", None))

    @property
    def atom_types(self):
        atom_types = self._get("atom_types", None)
        if atom_types is not None:
            return atom_types
        node_attrs = self._get("node_attrs", None)
        if node_attrs is not None:
            if node_attrs.ndim == 1:
                return node_attrs.to(dtype=torch.long)
            return torch.argmax(node_attrs, dim=-1).to(dtype=torch.long)
        raise KeyError("Could not infer atom types.")

    @property
    def charges(self):
        return self._get("charges", None)


def get_ye3t_ace_runtime_cache(data):
    if not isinstance(data, Mapping):
        return None
    cache = data.get(YE3T_ACE_RUNTIME_CACHE_KEY)
    return cache if isinstance(cache, dict) else None


def get_or_compute_edge_vectors(data, *, dtype):
    adapter = YE3TGraphDataAdapter(data)
    runtime_cache = get_ye3t_ace_runtime_cache(data)
    geometry_inputs = None
    if runtime_cache is not None:
        geometry_inputs = runtime_cache.setdefault("geometry_inputs", {})
        cached = geometry_inputs.get(("edge_vectors", str(dtype)))
        if cached is not None:
            return cached
    if isinstance(data, Mapping) and data.get("edge_vectors", None) is not None:
        x_ij = data["edge_vectors"].to(dtype=dtype)
        if geometry_inputs is not None:
            geometry_inputs[("edge_vectors", str(dtype))] = x_ij
        return x_ij
    positions = adapter.positions.to(dtype=dtype)
    edge_index = adapter.edge_index
    shifts = None if adapter.shifts is None else adapter.shifts.to(dtype=dtype)
    cell = data.get("edge_cell", None) if isinstance(data, Mapping) else getattr(data, "edge_cell", None)
    if cell is None:
        cell = data.get("cell", None) if isinstance(data, Mapping) else getattr(data, "cell", None)
    if cell is None:
        cell = torch.eye(3, dtype=dtype, device=positions.device)
    else:
        cell = cell.to(dtype=dtype, device=positions.device)
    x_ij = edge_vectors_from_positions(
        positions,
        cell,
        edge_index,
        shifts=shifts,
    )
    if geometry_inputs is not None:
        geometry_inputs[("edge_vectors", str(dtype))] = x_ij
    return x_ij


