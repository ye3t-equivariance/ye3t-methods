"""One strict, coefficient-free resolver for public basis construction requests."""

from collections import namedtuple
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral
import numpy as np

from ye3t import YE3TRepresentation
from ye3t.core.basis.validation import (
    count_canonical_leaf_labelings,
    iter_canonical_leaf_labelings,
)


class BasisResolution(namedtuple("BasisResolutionFields", "payload_json sha256")):
    """Purpose: Hold a frozen normalized basis request and capability report.

    Mathematical contract: The hash binds physical source records and the
    rank-resolved representation before descriptor compilation.
    Inputs: Canonical JSON and its SHA-256 identity from the resolver.
    Outputs: Detached decoded views, warnings, and capability status.
    Does not: Materialize coefficients during resolution.
    """

    def to_dict(self):
        return json.loads(self.payload_json)

    @property
    def warnings(self):
        return tuple(self.to_dict()["warnings"])

    @property
    def capability_report(self):
        return self.to_dict()["capability_report"]

    def __repr__(self):
        payload = self.to_dict()
        names = tuple(row["name"] for row in payload["components"])
        return (f"BasisResolution(sha256={self.sha256}, components={names}, "
                f"warnings={len(payload['warnings'])}, "
                f"create_available={payload['capability_report']['basis_create_available']})")

    __str__ = __repr__


def _keys(value, allowed, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{name} has unsupported fields: {sorted(unknown)!r}.")


def _integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise TypeError(f"{name} must be an integer.")


def _integer_key_map(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    result = {}
    for raw_key, item in value.items():
        key = _integer(raw_key, f"{name} key")
        if key in result:
            raise ValueError(f"{name} contains duplicate normalized key {key}.")
        result[key] = item
    return result


def _rank_map(value, ranks, name, minimum):
    result = {rank: _integer(number, name) for rank, number in _integer_key_map(value, name).items()}
    if set(result) != set(ranks):
        raise ValueError(f"{name} must cover exactly ranks {tuple(ranks)!r}.")
    if any(number < minimum for number in result.values()):
        raise ValueError(f"{name} entries must be at least {minimum}.")
    return result


def _radial_config(raw):
    _keys(raw, {"family", "cutoff_A", "cutoff_width_A", "lambda"}, "single_factors.radial")
    family = raw.get("family")
    if family == "pace_chebexp_cos":
        if set(raw) != {"family", "cutoff_A", "cutoff_width_A", "lambda"}:
            raise ValueError("pace_chebexp_cos requires cutoff_A, cutoff_width_A, and lambda.")
        values = {key: float(raw[key]) for key in ("cutoff_A", "cutoff_width_A", "lambda")}
        if not all(math.isfinite(value) and value > 0 for value in values.values()):
            raise ValueError("PACE radial parameters must be finite and positive.")
        if values["cutoff_width_A"] >= values["cutoff_A"]:
            raise ValueError("cutoff_width_A must be smaller than cutoff_A.")
        return {"family": family, **values, "units": "Angstrom", "public_radial_index_origin": 0,
                "native_pace_n_offset": 1}
    if family == "shifted_jacobi":
        if set(raw) != {"family", "cutoff_A"}:
            raise ValueError("shifted_jacobi accepts cutoff_A and no PACE lambda or cutoff width.")
        cutoff = float(raw["cutoff_A"])
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("shifted_jacobi cutoff_A must be finite and positive.")
        from ye3t.couplings.orthogonal_shifted_jacobi import (
            ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
        )
        return {"family": family, "cutoff_A": cutoff, "units": "Angstrom",
                "public_radial_index_origin": 0,
                "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY}
    raise ValueError(f"Unsupported radial family {family!r}.")


def _chemical_config(raw, species):
    _keys(raw, {"kind", "species_order", "matrix"}, "single_factors.chemical")
    kind = raw.get("kind", "explicit")
    if kind in {"explicit", "one_hot"}:
        if set(raw) - {"kind"}:
            raise ValueError(f"{kind} chemistry does not accept an embedding matrix.")
        channels = tuple({"kind": kind, "neighbor_species": name, "chemical_index": index}
                         for index, name in enumerate(species))
        return {"kind": kind}, channels
    if kind != "fixed_embedding":
        raise ValueError("chemical.kind must be explicit, one_hot, or fixed_embedding.")
    if set(raw) != {"kind", "species_order", "matrix"}:
        raise ValueError("fixed_embedding requires species_order and matrix.")
    if not isinstance(raw["species_order"], (tuple, list)):
        raise TypeError("fixed_embedding.species_order must be a sequence of species names.")
    order = tuple(str(name) for name in raw["species_order"])
    if order != species:
        raise ValueError("fixed_embedding.species_order must match single_factors.species exactly.")
    matrix = tuple(tuple(float(value) for value in row) for row in raw["matrix"])
    if len(matrix) != len(species) or not matrix or not matrix[0]:
        raise ValueError("fixed_embedding.matrix must have one nonempty row per species.")
    width = len(matrix[0])
    if any(len(row) != width or any(not math.isfinite(value) for value in row) for row in matrix):
        raise ValueError("fixed_embedding.matrix must be finite and rectangular.")
    if any(all(row[column] == 0 for row in matrix) for column in range(width)):
        raise ValueError("fixed_embedding.matrix cannot contain an inactive zero column.")
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    rank_tolerance = max(len(species), width) * np.finfo(float).eps * float(singular_values[0])
    if width > len(species) or int(np.count_nonzero(singular_values > rank_tolerance)) != width:
        raise ValueError("fixed_embedding.matrix must have full independent column rank.")
    channels = tuple({"kind": kind, "embedding_column": column, "chemical_index": column}
                     for column in range(width))
    return {
        "kind": kind, "species_order": list(order), "matrix": [list(row) for row in matrix],
        "verified_column_rank": width, "rank_tolerance": rank_tolerance,
    }, channels


def _component(name, raw, representation, chemical_channels, species, radial):
    _keys(raw, {"tensor_product", "catalogue", "single_factors"},
          f"basis.components.{name}")
    product = raw["tensor_product"]
    _keys(product, {"kind", "tag_counts_per_rank"}, f"{name}.tensor_product")
    kind = product.get("kind")
    if kind not in {"density", "tagged"}:
        raise ValueError(f"Unsupported tensor_product.kind {kind!r} in P1 config resolver.")
    catalogue = raw["catalogue"]
    _keys(catalogue, {"ranks", "nmax_per_rank", "lmax_per_rank",
                      "source_block_partitions_by_rank", "angular_patterns_by_rank",
                      "selection"}, f"{name}.catalogue")
    if not isinstance(catalogue["ranks"], (tuple, list)):
        raise TypeError(f"{name}.catalogue.ranks must be a sequence.")
    ranks = tuple(_integer(rank, f"{name} rank") for rank in catalogue["ranks"])
    if not ranks or tuple(sorted(set(ranks))) != ranks or not set(ranks).issubset(representation.ranks):
        raise ValueError(f"{name}.catalogue.ranks must be ascending, unique, and selected by representation.")
    nmax = _rank_map(catalogue["nmax_per_rank"], ranks, f"{name}.nmax_per_rank", 1)
    lmax = _rank_map(catalogue["lmax_per_rank"], ranks, f"{name}.lmax_per_rank", 0)
    raw_partitions = _integer_key_map(catalogue["source_block_partitions_by_rank"],
                                     f"{name}.source_block_partitions_by_rank")
    if set(raw_partitions) != set(ranks):
        raise ValueError(f"{name}.source_block_partitions_by_rank must cover its ranks.")
    partitions = {}
    for rank in ranks:
        values = raw_partitions[rank]
        rows = tuple(tuple(_integer(size, "source block size") for size in row) for row in values)
        if not rows or len(set(rows)) != len(rows) or any(
            not row or min(row) < 1 or sum(row) != rank or tuple(sorted(row, reverse=True)) != row
            for row in rows
        ):
            raise ValueError(f"{name} source block partitions at rank {rank} must be unique valid partitions.")
        partitions[rank] = rows
    selection = catalogue.get("selection", {})
    _keys(selection, {"repeated_content_min"}, f"{name}.selection")
    repeated = selection.get("repeated_content_min", 1)
    if isinstance(repeated, Mapping):
        repeated = _rank_map(repeated, ranks, f"{name}.repeated_content_min", 1)
    else:
        repeated = {rank: _integer(repeated, f"{name}.repeated_content_min") for rank in ranks}
    if any(repeated[rank] > rank for rank in ranks):
        raise ValueError("repeated_content_min cannot exceed its rank.")
    partitions = {rank: tuple(row for row in partitions[rank] if row[0] >= repeated[rank]) for rank in ranks}
    if any(not partitions[rank] for rank in ranks):
        raise ValueError("repeated_content_min removed every requested source partition.")

    angular_patterns = None
    if "angular_patterns_by_rank" in catalogue:
        if kind != "tagged" or representation.L != 0:
            raise ValueError("Explicit angular patterns currently require a scalar tagged basis.")
        raw_patterns = _integer_key_map(catalogue["angular_patterns_by_rank"],
                                        f"{name}.angular_patterns_by_rank")
        if set(raw_patterns) != set(ranks):
            raise ValueError("angular_patterns_by_rank must cover each selected rank.")
        angular_patterns = {}
        for rank in ranks:
            values = raw_patterns[rank]
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError("angular_patterns_by_rank values must be nonempty sequences.")
            patterns = tuple(tuple(_integer(value, "angular degree") for value in pattern)
                             for pattern in values)
            if (len(set(patterns)) != len(patterns) or
                    any(len(pattern) != rank or min(pattern) < 0 or
                        max(pattern) > lmax[rank] for pattern in patterns)):
                raise ValueError("Angular patterns must be unique rank-length tuples in 0..lmax.")
            angular_patterns[rank] = [list(pattern) for pattern in patterns]

    if kind == "density":
        if "tag_counts_per_rank" in product:
            raise ValueError("density cannot request tag_counts_per_rank.")
        if any(representation.parent_partition(rank) != (rank,) for rank in ranks):
            raise ValueError("Commutative density cannot realize a nontrivial global Young parent.")
        if representation.young_kappa[0] == "explicit" and any(
            part != (size,) for size, parts in representation.young_kappa[1] for part in parts
        ):
            raise ValueError("Commutative density cannot realize explicit nontrivial block Young sectors.")
        tags = {}
    else:
        if any(representation.parent_partition(rank) != (rank,) for rank in ranks):
            raise ValueError("Pooled tagged density cannot realize a nontrivial global Young parent.")
        raw_tags = _integer_key_map(product.get("tag_counts_per_rank"),
                                    f"{name}.tag_counts_per_rank")
        if set(raw_tags) != set(ranks):
            raise TypeError("tag_counts_per_rank must be a rank map of lists.")
        tags = {}
        for rank in ranks:
            if not isinstance(raw_tags[rank], (tuple, list)):
                raise TypeError("tag_counts_per_rank values must be sequences.")
            values = tuple(_integer(value, "tag count") for value in raw_tags[rank])
            if not values or len(set(values)) != len(values) or min(values) < 0 or max(values) > rank:
                raise ValueError("tag_counts_per_rank must contain unique counts in 0..rank.")
            tags[rank] = list(values)

    maximum_radial = max(nmax.values())
    if kind == "tagged" and all("neighbor_species" in channel for channel in chemical_channels):
        ordered_channels = tuple(
            {**channel, "chemical_index": index}
            for index, channel in enumerate(sorted(
                chemical_channels, key=lambda channel: channel["neighbor_species"]))
        )
        eta_coordinates = tuple(
            (channel, radial_index)
            for radial_index in range(maximum_radial)
            for channel in ordered_channels
        )
    else:
        eta_coordinates = tuple(
            (channel, radial_index)
            for radial_index in range(maximum_radial)
            for channel in chemical_channels
        )
    eta_by_center = {}
    for center in species:
        eta_by_center[center] = [
            {"eta_index": index, "compiler_content_id": index + 1,
             "central_species": center,
             "chemical": channel, "radial_index": radial_index,
             **({"native_pace_n": radial_index + 1} if radial["family"] == "pace_chebexp_cos" else {})}
            for index, (channel, radial_index) in enumerate(eta_coordinates)
        ]
    active_content_ids = {
        rank: [index + 1 for index, (_channel, radial_index)
               in enumerate(eta_coordinates) if radial_index < nmax[rank]]
        for rank in ranks
    }
    if kind == "tagged":
        for rank in ranks:
            candidates = sum(count_canonical_leaf_labelings(
                rank, active_content_ids[rank], range(lmax[rank] + 1),
                multiplicity_partitions=(partition,))
                for partition in partitions[rank])
            if candidates == 0:
                raise ValueError(
                    f"Tagged rank {rank} has no complete-channel source records "
                    "for its selected partitions and physical channels.")
    return {
        "name": str(name), "kind": kind, "ranks": list(ranks),
        "nmax_per_rank": nmax, "lmax_per_rank": lmax,
        "source_block_partitions_by_rank": {rank: [list(row) for row in partitions[rank]] for rank in ranks},
        "repeated_content_min": repeated,
        "tag_counts_per_rank": tags,
        **({"angular_patterns_by_rank": angular_patterns}
           if angular_patterns is not None else {}),
        "physical_eta_by_center": eta_by_center,
        "active_compiler_content_ids_by_rank": active_content_ids,
        "chemical_channel_count": len(chemical_channels),
        "radial_family": radial["family"],
    }


def resolve_basis_config(config, representation, runtime, *,
                         check_optional_dependencies=True):
    """Purpose: Resolve one public basis request before materialization.

    Mathematical contract: Physical eta identities include chemistry and
    radial channels; Young and rotation restrictions stay compiler-owned.
    Inputs: Basis section, core YE3TRepresentation, and runtime section.
    Outputs: Frozen BasisResolution with source and capability records.
    Does not: Create descriptor labels, coupling coefficients, or evaluators.
    """
    if not isinstance(representation, YE3TRepresentation):
        raise TypeError("representation must be ye3t.YE3TRepresentation.from_config(...).")
    if (isinstance(config, Mapping)
            and isinstance(config.get("tensor_product"), Mapping)
            and config["tensor_product"].get("kind") == "explicit_phi"):
        return _resolve_ordered_phi_star_config(config, representation, runtime)
    _keys(config, {"single_factors", "components", "tensor_product", "catalogue"}, "basis")
    factors = config["single_factors"]
    _keys(factors, {"species", "radial", "chemical"}, "basis.single_factors")
    if not isinstance(factors["species"], (tuple, list)):
        raise TypeError("single_factors.species must be a sequence of species names.")
    species = tuple(str(name) for name in factors["species"])
    if not species or any(not name for name in species) or len(set(species)) != len(species):
        raise ValueError("single_factors.species must be a nonempty unique ordered list.")
    radial = _radial_config(factors["radial"])
    has_components = "components" in config
    if has_components == ("tensor_product" in config or "catalogue" in config):
        raise ValueError("basis requires components or sibling tensor_product and catalogue.")
    if has_components:
        raw_components = config["components"]
        if not isinstance(raw_components, Mapping) or not raw_components:
            raise ValueError("basis.components must be a nonempty mapping.")
        if any(not isinstance(name, str) or not name for name in raw_components):
            raise ValueError("basis.components names must be nonempty strings.")
    else:
        if "tensor_product" not in config or "catalogue" not in config:
            raise ValueError("Single-component basis needs tensor_product and catalogue.")
        raw_components = {"main": {"tensor_product": config["tensor_product"],
                                   "catalogue": config["catalogue"]}}
    requested_species = species
    requested_chemistry = factors.get("chemical", {"kind": "explicit"})
    has_tagged = any(isinstance(row, Mapping) and
                     isinstance(row.get("tensor_product"), Mapping) and
                     row["tensor_product"].get("kind") == "tagged"
                     for row in raw_components.values())
    if (isinstance(requested_chemistry, Mapping)
            and has_tagged
            and requested_chemistry.get("kind", "explicit") in {"explicit", "one_hot"}):
        species = tuple(sorted(species))
    chemical, _channels = _chemical_config(requested_chemistry, species)
    components = []
    for name, raw in raw_components.items():
        local = raw.get("single_factors", factors)
        _keys(local, {"species", "radial", "chemical"},
              f"basis.components.{name}.single_factors")
        if tuple(str(value) for value in local["species"]) != requested_species:
            raise ValueError("Component-local species must match the ordered basis species.")
        local_radial = _radial_config(local["radial"])
        local_chemical, local_channels = _chemical_config(
            local.get("chemical", {"kind": "explicit"}), species)
        component = _component(name, raw, representation, local_channels,
                               species, local_radial)
        component["single_factors"] = {
            "species": list(species), "radial": local_radial,
            "chemical": local_chemical}
        component["chemical_kind"] = local_chemical["kind"]
        components.append(component)

    _keys(runtime, {"evaluator", "neighbors", "cache", "dtype", "device"}, "runtime")
    evaluator = runtime.get("evaluator", "auto")
    neighbors = runtime.get("neighbors", "auto")
    device = runtime.get("device", "cpu")
    dtype = runtime.get("dtype", "float64")
    cache = runtime.get("cache", {"mode": "auto"})
    _keys(cache, {"mode"}, "runtime.cache")
    if evaluator not in {"auto", "native_cpu", "torch", "reference"}:
        raise ValueError("Unsupported runtime.evaluator.")
    if neighbors not in {"auto", "ase", "matscipy"}:
        raise ValueError("Unsupported runtime.neighbors.")
    if cache.get("mode", "auto") not in {"auto", "read_only", "refresh", "off"}:
        raise ValueError("Unsupported runtime.cache.mode.")
    if dtype != "float64":
        raise ValueError("Only float64 has a validated P1 config contract.")
    if device != "cpu" and not str(device).startswith("cuda"):
        raise ValueError("runtime.device must be cpu or a CUDA device.")
    if evaluator == "native_cpu" and device != "cpu":
        raise ValueError("native_cpu evaluator requires runtime.device='cpu'.")
    if neighbors == "matscipy" and check_optional_dependencies:
        import importlib.util
        if importlib.util.find_spec("matscipy") is None:
            raise ImportError("runtime.neighbors='matscipy' requires the optional matscipy dependency.")

    warnings = []
    if species != requested_species:
        warnings.append("tagged species order normalized to compiler lexical order")
    rep_config = representation.to_dict()
    capacity = rep_config["uncoupled_factor_inputs"]
    for rank in representation.ranks:
        required_eta = max((component["chemical_channel_count"] * component["nmax_per_rank"][rank]
                            for component in components if rank in component["ranks"]), default=0)
        required_l = max((component["lmax_per_rank"][rank]
                          for component in components if rank in component["ranks"]), default=0)
        hint_eta = capacity["eta_count_per_rank"][rank]
        hint_l = capacity["l_max_per_rank"][rank]
        if required_eta > hint_eta:
            warnings.append(f"rank {rank}: eta capacity expanded from {hint_eta} to {required_eta} physical channels per center")
            capacity["eta_count_per_rank"][rank] = required_eta
        if required_l > hint_l:
            warnings.append(f"rank {rank}: l_max capacity expanded from {hint_l} to {required_l}")
            capacity["l_max_per_rank"][rank] = required_l
    resolved_rep = YE3TRepresentation.from_config(rep_config)
    if resolved_rep.subspace != "full" or resolved_rep.block_rotation[0] != "all_valid":
        warnings.append("restricted subspace or block rotation has no P1 catalogue count route")
    if any(component["kind"] == "density" for component in components):
        warnings.append("commutative density resolves all_valid Young blocks to physically trivial block sectors")

    component_status = {}
    for component in components:
        if component["kind"] == "density":
            component_status[component["name"]] = (
                "exact_density_fixed_content" if resolved_rep.subspace == "full"
                and resolved_rep.young_kappa[0] in {"all_valid", "symmetric_only"}
                and resolved_rep.block_rotation[0] == "all_valid"
                else "restricted_compiler_plan_pending"
            )
        elif component["radial_family"] != "shifted_jacobi":
            component_status[component["name"]] = "unsupported_tagged_source_family"
        elif component["chemical_kind"] not in {"explicit", "one_hot"}:
            component_status[component["name"]] = "unsupported_tagged_chemistry"
        elif (resolved_rep.group != "O3"
              or (resolved_rep.L == 0 and resolved_rep.parity != "even")
              or resolved_rep.factorization != "cauchy" or resolved_rep.subspace != "full"
              or resolved_rep.young_kappa[0] != "all_valid"
              or resolved_rep.block_rotation[0] != "all_valid"):
            component_status[component["name"]] = "unsupported_tagged_symmetry_or_factorization"
        elif resolved_rep.L > 0:
            component_status[component["name"]] = "tagged_carrier_raw_opportunities_available"
        else:
            component_status[component["name"]] = "tagged_raw_upper_bound_available"
    tagged_scalar_available = (
        len(components) == 1 and
        components[0]["kind"] == "tagged" and
        component_status[components[0]["name"]] == "tagged_raw_upper_bound_available" and
        device == "cpu" and cache.get("mode", "auto") == "auto" and
        ((evaluator in {"auto", "torch", "reference"} and neighbors == "auto") or
         (evaluator == "native_cpu" and neighbors in {"auto", "ase", "matscipy"}))
    )
    tagged_multipole_available = (
        len(components) == 1 and components[0]["kind"] == "tagged" and
        all(max(tags) <= 2 for tags in
            components[0]["tag_counts_per_rank"].values()) and
        component_status[components[0]["name"]] == "tagged_carrier_raw_opportunities_available" and
        device == "cpu" and cache.get("mode", "auto") in {"auto", "off"} and
        evaluator in {"auto", "reference"} and neighbors in {"auto", "ase"}
    )
    ordinary_pace_base = (
        len(components) == 1 and
        components[0]["kind"] == "density" and
        components[0]["radial_family"] == "pace_chebexp_cos" and
        component_status[components[0]["name"]] == "exact_density_fixed_content" and
        resolved_rep.group == "O3" and resolved_rep.L == 0 and
        resolved_rep.parity == "even" and resolved_rep.factorization == "cauchy" and
        device == "cpu" and
        cache.get("mode", "auto") in {"auto", "off"}
    )
    ordinary_pace_available = ordinary_pace_base and (
        (len(species) == 1 and
         components[0]["chemical_kind"] in {"explicit", "one_hot"} and
         evaluator in {"auto", "torch", "native_cpu"} and
        (neighbors in {"auto", "ase"} or
         evaluator == "native_cpu" and neighbors == "matscipy")) or
        ((components[0]["chemical_kind"] == "fixed_embedding" or
          len(species) > 1 and components[0]["chemical_kind"] in {"explicit", "one_hot"}) and
         evaluator in {"auto", "torch"} and neighbors in {"auto", "ase"})
    )
    ordinary_covariant_available = (
        len(components) == 1 and components[0]["kind"] == "density" and
        components[0]["radial_family"] == "pace_chebexp_cos" and
        component_status[components[0]["name"]] == "exact_density_fixed_content" and
        resolved_rep.group == "O3" and
        resolved_rep.L > 0 and
        resolved_rep.factorization == "cauchy" and device == "cpu" and
        cache.get("mode", "auto") in {"auto", "off"} and
        evaluator in {"auto", "torch"} and neighbors in {"auto", "ase"} and
        components[0]["chemical_kind"] in {"explicit", "one_hot", "fixed_embedding"}
    )
    create_available = (tagged_scalar_available or tagged_multipole_available or
                        ordinary_pace_available or ordinary_covariant_available)
    component_runtime = {}
    if len(components) > 1:
        component_capabilities = []
        for name, raw in raw_components.items():
            local_config = {
                "single_factors": raw.get("single_factors", factors),
                "tensor_product": raw["tensor_product"],
                "catalogue": raw["catalogue"],
            }
            local_capability = resolve_basis_config(
                local_config, representation, runtime,
                check_optional_dependencies=check_optional_dependencies).capability_report
            component_capabilities.append(local_capability)
            component_runtime[name] = {
                "selected_evaluator": local_capability["selected_evaluator"],
                "selected_neighbors": local_capability["selected_neighbors"],
                "basis_create_available": local_capability["basis_create_available"],
            }
        create_available = (resolved_rep.L == 0 and resolved_rep.parity == "even" and
                            any(row["kind"] == "density" for row in components) and
                            any(row["kind"] == "tagged" for row in components) and
                            all(row["basis_create_available"] for row in component_capabilities))
    selected_evaluator = ("native_cpu" if evaluator == "native_cpu" and create_available else
                          "torch" if ordinary_pace_available or ordinary_covariant_available else
                          "reference" if tagged_multipole_available else
                          "reference" if tagged_scalar_available else
                          "component_local" if len(components) > 1 and create_available else None)
    selected_neighbors = (neighbors if ordinary_pace_available and evaluator == "native_cpu" else
                          "ase" if ordinary_pace_available or ordinary_covariant_available else
                          "ase" if tagged_multipole_available else
                          neighbors if tagged_scalar_available and evaluator == "native_cpu" else
                          "directed_edges_all_images_bruteforce"
                          if tagged_scalar_available else
                          "component_local" if len(components) > 1 and create_available else None)
    if len(components) == 1:
        component_runtime[components[0]["name"]] = {
            "selected_evaluator": selected_evaluator,
            "selected_neighbors": selected_neighbors,
            "basis_create_available": create_available,
        }
    capabilities = {
        "compiler_count_api": "ye3t.couplings.count",
        "coefficient_materialization_performed": False,
        "basis_create_available": create_available,
        "basis_create_reason": None if create_available else
            "Multi-component covariant combination is not supported by Basis.combine"
            if len(components) > 1 and resolved_rep.L > 0
            else
            "Tagged L>0 physical-image evaluation currently requires 0/1/2 tags, "
            "reference CPU, ASE neighbors, and cache auto/off" if any(
                row["kind"] == "tagged" for row in components) and resolved_rep.L > 0
            else "P2 evaluator integration is pending for this resolved config",
        "requested_evaluator": evaluator,
        "selected_evaluator": selected_evaluator,
        "requested_neighbors": neighbors,
        "selected_neighbors": selected_neighbors,
        "component_runtime": component_runtime,
        "component_count_status": component_status,
        "resolved_young_kappa_by_component": {
            component["name"]: (
                "symmetric_only_physical_density" if component["kind"] == "density"
                else resolved_rep.young_kappa[0]
            ) for component in components
        },
    }
    payload = {
        "schema": "ye3t_basis_resolution_v1",
        "species": list(species),
        "single_factors": {"species": list(species), "radial": radial, "chemical": chemical},
        "representation": resolved_rep.to_dict(),
        "components": components,
        "runtime": {"evaluator": evaluator, "neighbors": neighbors,
                    "cache": {"mode": cache.get("mode", "auto")}, "dtype": dtype, "device": device},
        "warnings": warnings,
        "capability_report": capabilities,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return BasisResolution(encoded, digest)


def _resolve_ordered_phi_star_config(config, representation, runtime):
    """Resolve the fixed first ordered Phi star against compiler-issued paths."""

    import torch
    from ye3t.couplings import plan
    from ye3t.core.tesseral import real_tesseral_to_complex_multiplet

    _keys(config, {"single_factors", "tensor_product", "catalogue"}, "basis")
    factors = config["single_factors"]
    _keys(factors, {"species", "radial", "chemical"}, "basis.single_factors")
    species = tuple(str(value) for value in factors["species"])
    if len(species) != 1 or not species[0]:
        raise ValueError("The first ordered Phi star requires one physical species.")
    radial = _radial_config(factors["radial"])
    if radial["family"] != "pace_chebexp_cos":
        raise ValueError("The first ordered Phi star requires the PACE ChebExpCos source.")
    chemical, _channels = _chemical_config(factors.get("chemical", {"kind": "explicit"}), species)
    if chemical["kind"] != "explicit":
        raise ValueError("The first ordered Phi star requires explicit one-species chemistry.")
    product = config["tensor_product"]
    _keys(product, {"kind", "motif"}, "basis.tensor_product")
    motif = product["motif"]
    _keys(motif, {"kind", "ordered_slots", "leaf_edges"}, "basis.tensor_product.motif")
    if (motif.get("kind") != "rooted_star" or motif.get("ordered_slots") is not True
            or motif.get("leaf_edges") != []):
        raise ValueError("The first ordered Phi motif must be an ordered rooted star without leaf edges.")
    if (representation.ranks != (8,) or representation.parent_partition(8) != (4, 4)
            or representation.L != 2 or representation.parity != "even"
            or representation.group != "O3" or representation.factorization != "cauchy"
            or representation.subspace != "full"):
        raise ValueError("The first ordered Phi star requires rank 8, Young (4,4), L=2 even O3 full Cauchy.")
    rep_payload = representation.to_dict()
    young = rep_payload["intermediates"]["young_kappa"]
    rotation = rep_payload["intermediates"]["block_rotation"]
    if (young != {"policy": "explicit", "by_block_size": {4: [[4]]}}
            or rotation.get("policy") != "explicit"
            or set(rotation.get("Lambda_values_by_block_size", {}).get(4, ())) != {0, 2, 4}):
        raise ValueError("The first ordered Phi star requires explicit kappa=(4) and block L=0,2,4.")
    catalogue = config["catalogue"]
    _keys(catalogue, {"ranks", "nmax_per_rank", "lmax_per_rank",
                      "source_block_partitions_by_rank", "fixed_content", "selection"},
          "basis.catalogue")
    if tuple(catalogue["ranks"]) != (8,):
        raise ValueError("The ordered Phi catalogue must select rank eight only.")
    if (_rank_map(catalogue["nmax_per_rank"], (8,), "nmax_per_rank", 1) != {8: 2}
            or _rank_map(catalogue["lmax_per_rank"], (8,), "lmax_per_rank", 0) != {8: 1}):
        raise ValueError("The first ordered Phi star requires two radial and l=1 source channels.")
    blocks = _integer_key_map(catalogue["source_block_partitions_by_rank"],
                              "source_block_partitions_by_rank")
    if blocks != {8: [[4, 4]]}:
        raise ValueError("The first ordered Phi star requires the [4,4] source blocks.")
    fixed = catalogue["fixed_content"]
    if not isinstance(fixed, (tuple, list)) or len(fixed) != 2:
        raise ValueError("The ordered Phi star requires exactly two fixed-content records.")
    for radial_index, record in enumerate(fixed):
        _keys(record, {"factor", "copies"}, "basis.catalogue.fixed_content")
        _keys(record["factor"], {"species", "radial_index", "l"},
              "basis.catalogue.fixed_content.factor")
        if (record["factor"] != {"species": species[0], "radial_index": radial_index, "l": 1}
                or _integer(record["copies"], "fixed-content copies") != 4):
            raise ValueError("Ordered Phi fixed content must have four n=0,l=1 then four n=1,l=1 factors.")
    selection = catalogue["selection"]
    _keys(selection, {"coupling_paths"}, "basis.catalogue.selection")
    paths = selection["coupling_paths"]
    if (not isinstance(paths, (tuple, list)) or len(paths) != 1
            or paths[0] != {"young_kappa": ["(4)", "(4)"], "Lambda": [0, 2]}):
        raise ValueError("The first ordered Phi star selects only kappa=(4),(4), block L=(0,2).")
    _keys(runtime, {"evaluator", "neighbors", "cache", "dtype", "device"}, "runtime")
    cache = runtime.get("cache", {"mode": "auto"})
    _keys(cache, {"mode"}, "runtime.cache")
    if (runtime.get("evaluator") != "torch" or runtime.get("dtype") != "float64"
            or runtime.get("device") != "cpu" or runtime.get("neighbors") not in {"ase", "auto"}
            or cache.get("mode", "auto") not in {"auto", "off"}):
        raise ValueError("The first ordered Phi star requires torch float64 CPU and auto/off cache.")
    request = {
        "content": [1, 1, 1, 1, 2, 2, 2, 2], "input_Ls": [1] * 8,
        "target_L": 2, "target_permutation": "young:4,4", "carrier": "Phi",
        "carrier_options": {
            "slot_count": 8, "permuted_slot_count": 8,
            "factor_action": "permute_explicit_phi_tensor_product_factors",
        },
    }
    compiler_plan = plan(**request)
    bindings = tuple(compiler_plan.validation_report["alpha_bindings"])
    selected = tuple(row for row in bindings
                     if row["block_partitions"] == ((4,), (4,))
                     and row["block_Ls"] == (0, 2))
    if (len(bindings) != 18 or len(selected) != 1
            or len(tuple(row for row in bindings
                         if row["block_partitions"] == ((4,), (4,)))) != 6):
        raise ArithmeticError("The first ordered Phi source disagrees with the compiler route count.")
    real_to_complex = real_tesseral_to_complex_multiplet(
        torch.eye(5, dtype=torch.float64), 2,
    ).numpy()
    output_convention = {
        "basis": "real_tesseral", "signed_M_order": list(range(-2, 3)),
        "conversion": "ye3t.core.tesseral.real_tesseral_to_complex_multiplet",
        "real_to_complex_sha256": hashlib.sha256(
            np.ascontiguousarray(real_to_complex, dtype="<c16").view("<f8").tobytes()
        ).hexdigest(),
    }
    component = {
        "name": "main", "kind": "explicit_phi", "ranks": [8],
        "nmax_per_rank": {8: 2}, "lmax_per_rank": {8: 1},
        "source_block_partitions_by_rank": {8: [[4, 4]]},
        "active_compiler_content_ids_by_rank": {8: [1, 2]},
        "single_factors": {"species": list(species), "radial": radial, "chemical": chemical},
        "compiler_request": request,
        "selected_typed_alpha": int(selected[0]["alpha_index"]),
        "full_multiplicity_count": len(bindings),
        "selected_block_path_count": 6,
        "compiler_plan_hash": compiler_plan.convention_hash,
        "output_convention": output_convention,
    }
    payload = {
        "schema": "ye3t_basis_resolution_v1", "species": list(species),
        "components": [component], "representation": rep_payload,
        "runtime": dict(runtime), "warnings": [],
        "capability_report": {
            "basis_create_available": True, "basis_create_reason": "ordered_cluster_only",
            "component_count_status": {"main": "exact_selected_phi_route"},
            "component_materialization_status": {"main": "experimental_ordered_phi_star"},
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return BasisResolution(encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest())


def resolve_linear_fit_config(config, basis, *, check_optional_dependencies=True):
    """Validate a seven-section linear fit against its constructed basis."""
    required = {"metadata", "representation", "basis", "runtime", "model",
                "targets", "validation"}
    _keys(config, required, "config")
    if set(config) != required:
        raise ValueError("Scalar fit config requires all seven standard sections.")
    _keys(config["metadata"], {"schema", "name", "status",
                               "training_structures", "evaluation_structure",
                               "output_path", "system"}, "metadata")
    if (config["metadata"].get("schema") != "ye3t_config_v1" or
            config["metadata"].get("status", "stable") not in {"stable", "experimental"}):
        raise ValueError("Scalar fit metadata must declare ye3t_config_v1 status.")
    for path_field in ("training_structures", "evaluation_structure", "output_path"):
        path = config["metadata"].get(path_field)
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ValueError("metadata." + path_field + " must be a nonempty path string.")
    if "system" in config["metadata"]:
        system = config["metadata"]["system"]
        if not isinstance(system, Mapping) or any(not isinstance(key, str) for key in system):
            raise ValueError("metadata.system must be a mapping of editable ASE system inputs.")
        try:
            json.dumps(dict(system), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata.system needs finite portable values.") from error
    if not hasattr(basis, "_resolution"):
        raise ValueError("Config-driven fitting requires a Basis.from_config basis.")
    representation = YE3TRepresentation.from_config(config["representation"])
    resolution = resolve_basis_config(
        config["basis"], representation, config["runtime"],
        check_optional_dependencies=check_optional_dependencies)
    if resolution.sha256 != basis._resolution.sha256:
        raise ValueError("Fit config basis, representation, or runtime differs from the constructed Basis.")
    model = config["model"]
    _keys(model, {"kind", "output", "fit", "reference_energy"}, "model")
    if model.get("kind") != "linear":
        raise ValueError("Configured scalar fitting requires model.kind='linear'.")
    output = model.get("output", {"scope": "total_energy"})
    _keys(output, {"scope"}, "model.output")
    if output.get("scope") == "per_atom":
        if basis.source == "configured":
            basis._materialize_configured()
        if (basis.source != "tagged_carriers" and not (
                basis.source == "density" and getattr(basis, "_density_full_m", False))
                or representation.L <= 0):
            raise ValueError("Per-atom full-M fitting requires a density or selected tagged L>0 Basis.")
        if "reference_energy" in model:
            raise ValueError("Per-atom covariant fits do not accept scalar reference energies.")
        fit = model.get("fit")
        _keys(fit, {"solver", "alpha", "solver_options"}, "model.fit")
        solver = fit.get("solver", "ridge")
        if solver not in {"ridge", "lasso", "ard"}:
            raise ValueError("Per-atom solver must be ridge, lasso, or ard.")
        options = fit.get("solver_options", {})
        if not isinstance(options, Mapping) or "fit_intercept" in options or any(
                not isinstance(key, str) for key in options):
            raise ValueError("Per-atom solver_options must be a mapping without fit_intercept.")
        try:
            portable_options = json.loads(json.dumps(dict(options), sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Per-atom solver_options must contain finite portable values.") from error
        if solver == "ridge" and portable_options:
            raise ValueError("Per-atom ridge does not accept solver_options.")
        if solver == "ard" and "alpha" in fit:
            raise ValueError("Per-atom ARD uses solver_options rather than alpha.")
        if solver == "lasso" and "alpha" not in fit:
            raise ValueError("Per-atom LASSO requires alpha.")
        alpha = float(fit.get("alpha", 1e-8))
        if not math.isfinite(alpha) or alpha < 0 or solver == "lasso" and alpha == 0:
            raise ValueError("Per-atom alpha must be finite and positive for LASSO, nonnegative otherwise.")
        targets = config["targets"]
        _keys(targets, {"per_atom", "energy", "forces", "stress"}, "targets")
        if any(targets.get(name) is not None for name in ("energy", "forces", "stress")):
            raise ValueError("Per-atom full-M fitting currently accepts only per_atom targets.")
        target = targets.get("per_atom")
        _keys(target, {"key", "input", "units"}, "targets.per_atom")
        if (not isinstance(target.get("key"), str) or not target["key"] or
                target.get("input") not in {"real_tesseral", "cartesian"} or
                not isinstance(target.get("units"), str) or not target["units"]):
            raise ValueError("Per-atom target needs key, real_tesseral/cartesian input, and units.")
        if target["input"] == "cartesian" and (
                representation.L, representation.parity) not in ((1, "odd"), (2, "even")):
            raise ValueError("Cartesian target input requires polar L=1 odd or STF L=2 even.")
        validation = config["validation"]
        _keys(validation, {"checks"}, "validation")
        checks = validation.get("checks", [])
        if (not isinstance(checks, (tuple, list)) or
                any(not isinstance(check, str) for check in checks) or
                len(set(checks)) != len(checks)
                or any(check not in {"round_trip"} for check in checks)):
            raise ValueError("Per-atom validation currently accepts round_trip only.")
        resolved = {"schema": "ye3t_configured_per_atom_fit_v1",
                    "construction_resolution_sha256": resolution.sha256,
                    "representation": representation.to_dict(),
                    "model": {"kind": "linear", "output": {"scope": "per_atom"},
                              "fit": {"solver": solver,
                                      **({"alpha": alpha} if solver != "ard" else {}),
                                      "solver_options": portable_options}},
                    "targets": {"per_atom": dict(target)},
                    "validation": {"checks": list(checks)}}
        digest = hashlib.sha256(json.dumps(
            resolved, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        return {"output_scope": "per_atom", "method": "ardregression" if solver == "ard" else solver,
                "regularization": alpha, "sklearn_params": (
                    {**portable_options, "alpha": alpha} if solver == "lasso" else portable_options),
                "target_key": target["key"], "target_input": target["input"],
                "target_units": target["units"], "validation_checks": tuple(checks),
                "resolved_fit_config": resolved, "resolved_fit_config_sha256": digest}
    if "fit" not in model or "reference_energy" not in model:
        raise ValueError("Configured scalar model needs fit and reference_energy sections.")
    if output.get("scope") != "total_energy":
        raise ValueError("Configured scalar fitting supports total_energy output only.")
    fit = model["fit"]
    _keys(fit, {"solver", "alpha", "solver_options", "weights"}, "model.fit")
    solver = fit.get("solver", "ridge")
    if solver not in {"ridge", "lasso", "ard"}:
        raise ValueError("Configured scalar solver must be ridge, lasso, or ard.")
    options = fit.get("solver_options", {})
    if not isinstance(options, Mapping):
        raise TypeError("model.fit.solver_options must be a mapping.")
    if "fit_intercept" in options or any(not isinstance(key, str) for key in options):
        raise ValueError("solver_options cannot override fit_intercept and needs string keys.")
    try:
        portable_options = json.loads(json.dumps(
            dict(options), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("solver_options must contain portable finite JSON values.") from error
    if solver == "ridge" and portable_options:
        raise ValueError("Ridge config does not accept solver_options.")
    if solver == "ard" and "alpha" in fit:
        raise ValueError("ARD config uses solver_options rather than alpha.")
    if solver == "lasso" and ("alpha" not in fit or "alpha" in portable_options):
        raise ValueError("LASSO config needs one alpha in model.fit.")
    alpha = float(fit.get("alpha", 1e-8))
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("model.fit.alpha must be finite and nonnegative.")
    if solver == "lasso" and alpha <= 0:
        raise ValueError("LASSO alpha must be positive.")
    reference = model["reference_energy"]
    _keys(reference, {"per_species_E0_eV", "fit_E0"}, "model.reference_energy")
    offsets = reference.get("per_species_E0_eV")
    if not isinstance(offsets, Mapping) or set(offsets) != set(basis.elements):
        raise ValueError("Config reference energy must cover every basis species exactly.")
    offsets = {name: float(offsets[name]) for name in basis.elements}
    if any(not math.isfinite(value) for value in offsets.values()):
        raise ValueError("Config reference energies must be finite.")
    fit_E0 = reference.get("fit_E0")
    if type(fit_E0) is not bool:
        raise TypeError("model.reference_energy.fit_E0 must be a boolean.")
    if fit_E0 and len(resolution.to_dict()["components"]) == 1 and (
            resolution.to_dict()["components"][0]["kind"] != "density"):
        raise ValueError("Fitted per-species E0 requires density or combined scalar sources.")
    targets = config["targets"]
    _keys(targets, {"energy", "forces", "stress", "per_atom"}, "targets")
    if "per_atom" in targets:
        raise ValueError("Per-atom configured targets require the later full-M route.")
    if not isinstance(targets.get("energy"), str) or not targets["energy"]:
        raise ValueError("Scalar fit targets.energy must name an energy field.")
    if any(targets.get(name) is not None and
           (not isinstance(targets[name], str) or not targets[name])
           for name in ("forces", "stress")):
        raise ValueError("Scalar force and stress targets must be names or None.")
    weights = fit.get("weights", {})
    _keys(weights, {"energy", "forces", "stress"}, "model.fit.weights")
    if targets.get("stress") is not None and "stress" not in weights:
        raise ValueError("A configured stress target needs an explicit stress weight.")
    row_weights = {"energy": float(weights.get("energy", 1.0)),
                   "forces": float(weights.get("forces", 1.0 if targets.get("forces") else 0.0)),
                   "stress": float(weights.get("stress", 0.0))}
    if (any(not math.isfinite(value) or value < 0 for value in row_weights.values()) or
            not any(row_weights.values()) or
            any(row_weights[name] and targets.get(name) is None
                for name in ("forces", "stress"))):
        raise ValueError("Configured row weights must be finite, nonnegative, and have targets.")
    validation = config["validation"]
    _keys(validation, {"checks"}, "validation")
    checks = validation.get("checks", [])
    if (not isinstance(checks, (tuple, list)) or
            any(not isinstance(name, str) or
                name not in {"force_fd", "round_trip"} for name in checks) or
            len(set(checks)) != len(checks)):
        raise ValueError("Configured scalar validation checks must be force_fd or round_trip.")
    resolved_fit = {
        "schema": "ye3t_configured_scalar_fit_v1",
        "training_provenance_status": "recorded_not_derivable_from_deployed_coefficients",
        "metadata": {"schema": "ye3t_config_v1",
                     "name": config["metadata"].get("name"),
                     "status": config["metadata"].get("status", "stable")},
        "construction": json.loads(json.dumps({
            "representation": representation.to_dict(),
            "basis": config["basis"], "runtime": config["runtime"]},
            sort_keys=True, allow_nan=False)),
        "basis_resolution": resolution.to_dict(),
        "basis_resolution_sha256": resolution.sha256,
        "model": {"kind": "linear", "output": {"scope": "total_energy"},
                  "fit": {"solver": solver,
                          **({"alpha": alpha} if solver != "ard" else {}),
                          "solver_options": portable_options,
                          "weights": row_weights},
                  "reference_energy": {"per_species_E0_eV": offsets,
                                       "fit_E0": fit_E0}},
        "targets": {"energy": targets["energy"],
                    "forces": targets.get("forces"),
                    "stress": targets.get("stress")},
        "validation": {"checks": list(checks)},
    }
    resolved_fit_hash = hashlib.sha256(json.dumps(
        resolved_fit, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    return {"method": {"ridge": "ridge", "lasso": "lasso",
                       "ard": "ardregression"}[solver],
            "regularization": alpha if solver == "ridge" else 1e-8,
            "sklearn_params": ({**portable_options, "alpha": alpha}
                               if solver == "lasso" else portable_options if solver == "ard" else None),
            "reference_energies": offsets, "fit_E0": fit_E0,
            "energy_weight": row_weights["energy"],
            "force_weight": row_weights["forces"],
            "stress_weight": row_weights["stress"],
            "energy_key": targets["energy"],
            "force_key": targets.get("forces") or "forces",
            "stress_key": targets.get("stress") or "stress",
            "validation_checks": tuple(checks),
            "resolved_fit_config": resolved_fit,
            "resolved_fit_config_sha256": resolved_fit_hash}


class CoupledFactorCataloguePreview:
    """Exact count preview for one externally supplied factor content."""

    def __init__(self, report):
        self._report = deepcopy(report)

    def counts(self, *, maximum_fixed_contents=4096):
        if int(maximum_fixed_contents) < 1:
            raise ValueError("maximum_fixed_contents must be positive.")
        return deepcopy(self._report)

    def repeated_content_summary(self):
        request = self._report["request"]
        return {"coefficient_materialization_performed": False,
                "content_source": "ye3t.couplings.count",
                "block_sizes": tuple(request["block_sizes"]),
                "channels": deepcopy(request["channels"])}


class CataloguePreview:
    """Purpose: Preview selected physical contents and compiler multiplicities.

    Mathematical contract: Ordinary density counts are exact for supported
    full-space policies; tagged raw counts are only physical-image bounds.
    Inputs: A frozen BasisResolution.
    Outputs: Structural repetition and count reports.
    Does not: Compile coefficients or certify an evaluator.
    """

    def __init__(self, resolution):
        self._resolution = resolution

    def repeated_content_summary(self):
        """Report candidate complete (eta,l) repetitions before parity filtering."""
        payload = self._resolution.to_dict()
        summary = {}
        for component in payload["components"]:
            if component["kind"] == "explicit_phi":
                summary[component["name"]] = [{
                    "rank": 8, "partition": [4, 4],
                    "candidate_fixed_contents_before_parity": 1,
                    "source": "one_user_selected_complete_content",
                }]
                continue
            rows = []
            for rank in component["ranks"]:
                lmax = component["lmax_per_rank"][str(rank)]
                for partition in component["source_block_partitions_by_rank"][str(rank)]:
                    candidates = count_canonical_leaf_labelings(
                        rank, component["active_compiler_content_ids_by_rank"][str(rank)], range(lmax + 1),
                        multiplicity_partitions=(partition,),
                    )
                    rows.append({"rank": rank, "partition": partition,
                                 "candidate_fixed_contents_before_parity": int(candidates)})
            summary[component["name"]] = rows
        return {"coefficient_materialization_performed": False,
                "content_source": "ye3t.core.basis.count_canonical_leaf_labelings",
                "repetition_definition": "maximum multiplicity of one complete (eta,l) factor",
                "parity_filter_applied": False,
                "component_capability_status": payload["capability_report"]["component_count_status"],
                "by_component": summary}

    def counts(self, *, maximum_fixed_contents=4096):
        """Query compiler counts while keeping tagged image bounds distinct."""
        from ye3t.couplings import count, tagged_cauchy_carriers_request, tagged_cauchy_image_request

        payload = self._resolution.to_dict()
        preflight = self.repeated_content_summary()
        selected = sum(row["candidate_fixed_contents_before_parity"] for rows in preflight["by_component"].values() for row in rows)
        if selected > int(maximum_fixed_contents):
            raise MemoryError(
                f"Catalogue has {selected} candidate fixed contents before parity filtering; "
                f"maximum_fixed_contents={maximum_fixed_contents}."
            )
        results = {}
        for component in payload["components"]:
            status = payload["capability_report"]["component_count_status"][component["name"]]
            if status == "exact_selected_phi_route":
                report = count(**component["compiler_request"])
                full = int(report.counts_by_target[2])
                if full != int(component["full_multiplicity_count"]):
                    raise ArithmeticError("Ordered Phi compiler count changed after resolution.")
                results[component["name"]] = {
                    "status": status, "full_sector_multiplicity": full,
                    "selected_block_path_count": int(component["selected_block_path_count"]),
                    "selected_path_count": 1,
                    "selected_full_alpha": int(component["selected_typed_alpha"]),
                    "provenance": "ye3t.couplings.count",
                }
                continue
            if status == "tagged_carrier_raw_opportunities_available":
                record_cap = max(sum(
                    row["candidate_fixed_contents_before_parity"]
                    for row in preflight["by_component"][component["name"]]
                    if row["rank"] == rank) for rank in component["ranks"])
                request = tagged_cauchy_carriers_request(catalogue={
                    "ranks": component["ranks"],
                    "nmax_per_rank": component["nmax_per_rank"],
                    "lmax_per_rank": component["lmax_per_rank"],
                    "source_block_partitions_by_rank": component["source_block_partitions_by_rank"],
                    **({"tag_counts": component["tag_counts_per_rank"][str(
                        component["ranks"][0])]} if len(component["ranks"]) == 1 else
                       {"tag_counts_by_rank": component["tag_counts_per_rank"]}),
                    "input_Lmax": payload["representation"]["parent"]["L"],
                    "max_records_per_rank": record_cap,
                }, species=payload["species"])
                report = count(request)
                target_L = payload["representation"]["parent"]["L"]
                parity = 1 if payload["representation"]["parent"]["parity"] == "even" else -1
                raw_by_rank = {rank: 0 for rank in component["ranks"]}
                for candidate in report["candidate_records"]:
                    fixed = count(candidate["request"])
                    raw_by_rank[candidate["rank"]] += sum(
                        label["target_L"] == target_L and label["target_parity"] == parity
                        for label in fixed["labels"])
                results[component["name"]] = {
                    "status": "raw_carrier_opportunities_deferred_to_exact_physical_image",
                    "candidate_records_by_rank": {
                        row["rank"]: row["selected_candidate_records"]
                        for row in report["rank_inventory"]},
                    "raw_opportunities_by_rank": raw_by_rank,
                    "raw_opportunity_count": sum(raw_by_rank.values()),
                    "physical_image_upper_bound": sum(raw_by_rank.values()),
                    "exact_image_count": None,
                    "provenance": "ye3t.couplings.count",
                }
                continue
            if status == "tagged_raw_upper_bound_available":
                component_catalogue = {
                    "nmax_per_rank": component["nmax_per_rank"],
                    "lmax_per_rank": component["lmax_per_rank"],
                    "source_block_partitions_by_rank": component["source_block_partitions_by_rank"],
                    "tag_counts_by_rank": component["tag_counts_per_rank"],
                    **({"angular_patterns_by_rank": component["angular_patterns_by_rank"]}
                       if "angular_patterns_by_rank" in component else {}),
                    "max_records_per_rank": {
                        rank: sum(row["candidate_fixed_contents_before_parity"]
                                  for row in preflight["by_component"][component["name"]]
                                  if row["rank"] == rank)
                        for rank in component["ranks"]
                    },
                }
                request = tagged_cauchy_image_request(
                    catalogue=component_catalogue, species=payload["species"]
                )
                report = count(request)
                raw_by_rank = {}
                for label in report.labels:
                    rank = int(label["tensor_order"])
                    raw_by_rank[rank] = raw_by_rank.get(rank, 0) + 1
                results[component["name"]] = {
                    "status": "raw_upper_bound_deferred_to_compile",
                    "raw_opportunities_by_rank": raw_by_rank,
                    "raw_opportunity_count": int(report.raw_label_count),
                    "physical_image_upper_bound": int(report.image_dimension_upper_bound),
                    "exact_image_count": None,
                    "provenance": report.provenance["api"],
                }
                continue
            if component["kind"] != "density" or payload["capability_report"]["component_count_status"][component["name"]] != "exact_density_fixed_content":
                results[component["name"]] = {"status": status,
                                               "count": None}
                continue
            per_rank = {}
            for rank in component["ranks"]:
                lmax = component["lmax_per_rank"][str(rank)]
                total = 0
                partitions = component["source_block_partitions_by_rank"][str(rank)]
                for content, angular in iter_canonical_leaf_labelings(
                    rank, component["active_compiler_content_ids_by_rank"][str(rank)], range(lmax + 1),
                    multiplicity_partitions=partitions,
                ):
                    if payload["representation"]["group"] == "O3":
                        requested = payload["representation"]["parent"]["parity"]
                        natural = "odd" if sum(angular) % 2 else "even"
                        if natural != requested:
                            continue
                    report = count({
                        "content": content,
                        "target_rotation": {
                            "L_R": payload["representation"]["parent"]["L"],
                            "parity": payload["representation"]["parent"]["parity"],
                            "group": payload["representation"]["group"],
                        },
                        "target_permutation": "young:" + str(rank),
                        "carrier": "ACE_density",
                        "metadata": {"input_Ls": angular},
                    }, input_Ls=angular)
                    total += int(report.counts_by_target[payload["representation"]["parent"]["L"]])
                if total == 0:
                    raise ValueError(f"{component['name']} rank {rank} has no symmetry-compatible compiler labels.")
                per_rank[rank] = total
            results[component["name"]] = {
                "status": "exact_fixed_content_compiler_count",
                "by_rank_per_center": per_rank,
                "per_center": sum(per_rank.values()),
                "all_centers": len(payload["species"]) * sum(per_rank.values()),
            }
        return {"coefficient_materialization_performed": False,
                "label_source": "ye3t.couplings.count",
                "by_component": results,
                "exact_total_per_center": sum(item["per_center"] for item in results.values())
                if all("per_center" in item for item in results.values()) else None}
