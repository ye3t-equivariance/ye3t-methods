
"""Read and write `.yace` YACE descriptor files.

The `functions:` section is organized by central type index `mu0`, with each
entry storing `(mus, ns, ls, ms_combs, ctildes)` for one descriptor function.
"""
import math
from pathlib import Path

import yaml
from ye3t_methods.atomistic._record import recordclass


_PACE_LINEAR_COMPATIBILITY = "lammps_pace_linear_v1"
_PACE_ROOT_FIELDS = {
    "elements",
    "E0",
    "deltaSplineBins",
    "embeddings",
    "bonds",
    "functions",
}
_PACE_EMBEDDING_FIELDS = {
    "ndensity",
    "FS_parameters",
    "npoti",
    "rho_core_cutoff",
    "drho_core_cutoff",
}
_PACE_BOND_FIELDS = {
    "nradmax",
    "lmax",
    "nradbasemax",
    "radbasename",
    "radparameters",
    "radcoefficients",
    "prehc",
    "lambdahc",
    "rcut",
    "dcut",
    "rcut_in",
    "dcut_in",
    "inner_cutoff_type",
}
_PACE_FUNCTION_FIELDS = {
    "mu0",
    "rank",
    "ndensity",
    "num_ms_combs",
    "mus",
    "ns",
    "ls",
    "ms_combs",
    "ctildes",
}


class _YACELoader(yaml.SafeLoader):
    """Safe YAML loader that accepts standard sequence-valued bond keys."""


def _hashable_yaml_key(value):
    if isinstance(value, list):
        return tuple(_hashable_yaml_key(item) for item in value)
    return value


def _construct_yace_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = _hashable_yaml_key(loader.construct_object(key_node, deep=True))
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_YACELoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_yace_mapping,
)


def _yace_error(path, message):
    raise ValueError(f"{path}: {message}")


def _require_mapping(value, path):
    if not isinstance(value, dict):
        _yace_error(path, "expected a mapping")
    return value


def _require_exact_fields(mapping, required, path):
    missing = sorted(required.difference(mapping))
    if missing:
        _yace_error(path, f"missing required field {missing[0]!r}")
    unknown = sorted(set(mapping).difference(required), key=repr)
    if unknown:
        _yace_error(path, f"unsupported field {unknown[0]!r}")


def _require_integer(value, path, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        _yace_error(path, "expected an integer")
    if minimum is not None and value < minimum:
        _yace_error(path, f"must be at least {minimum}")
    return value


def _require_finite(value, path):
    if isinstance(value, bool):
        _yace_error(path, "expected a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: expected a finite number") from exc
    if not math.isfinite(number):
        _yace_error(path, "expected a finite number")
    return number


def _require_finite_sequence(value, path, *, length=None):
    if not isinstance(value, list):
        _yace_error(path, "expected a sequence")
    if length is not None and len(value) != length:
        _yace_error(path, f"expected length {length}, got {len(value)}")
    return tuple(
        _require_finite(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


@recordclass(('mu0', 'rank', 'ndensity', 'num_ms_combs', 'mus', 'ns', 'ls', 'ms_combs', 'ctildes'), frozen = True)
class YACEFunction:

    def as_mapping(self):
        return {
            "mu0": int(self.mu0),
            "rank": int(self.rank),
            "ndensity": int(self.ndensity),
            "num_ms_combs": int(self.num_ms_combs),
            "mus": list(self.mus),
            "ns": list(self.ns),
            "ls": list(self.ls),
            "ms_combs": list(self.ms_combs),
            "ctildes": list(self.ctildes),
        }


YaceFunction = YACEFunction


def _normalize_functions(functions_by_mu0):
    normalized = {}
    for mu0, entries in functions_by_mu0.items():
        normalized[int(mu0)] = []
        for entry in entries:
            if isinstance(entry, YACEFunction):
                normalized[int(mu0)].append(entry)
                continue
            payload = dict(entry)
            rank = int(payload["rank"])
            if rank <= 0:
                raise ValueError(f"rank must be positive; got {rank}")
            ndensity = int(payload.get("ndensity", 1))
            if ndensity <= 0:
                raise ValueError(f"ndensity must be positive; got {ndensity}")
            ms_combs = tuple(int(v) for v in payload["ms_combs"])
            if len(ms_combs) % rank != 0:
                raise ValueError(f"ms_combs length must be divisible by rank={rank}; got {len(ms_combs)}")
            num_ms_combs = int(payload.get("num_ms_combs", len(ms_combs) // rank))
            if num_ms_combs < 0:
                raise ValueError(f"num_ms_combs must be nonnegative; got {num_ms_combs}")
            if len(ms_combs) != rank * num_ms_combs:
                raise ValueError(
                    "ms_combs length must equal rank*num_ms_combs; "
                    f"got {len(ms_combs)} and {rank}*{num_ms_combs}"
                )
            mus = tuple(int(v) for v in payload["mus"])
            ns = tuple(int(v) for v in payload["ns"])
            ls = tuple(int(v) for v in payload["ls"])
            for name, values in (("mus", mus), ("ns", ns), ("ls", ls)):
                if len(values) != rank:
                    raise ValueError(
                        f"{name} length must equal rank={rank}; got {len(values)}"
                    )
            ctildes = tuple(float(v) for v in payload["ctildes"])
            if len(ctildes) != ndensity * num_ms_combs:
                raise ValueError(
                    "ctildes length must equal ndensity*num_ms_combs; "
                    f"got {len(ctildes)} and {ndensity}*{num_ms_combs}"
                )
            normalized[int(mu0)].append(
                YACEFunction(
                    mu0=int(payload.get("mu0", mu0)),
                    rank=rank,
                    ndensity=ndensity,
                    num_ms_combs=num_ms_combs,
                    mus=mus,
                    ns=ns,
                    ls=ls,
                    ms_combs=ms_combs,
                    ctildes=ctildes,
                )
            )
    return normalized


def _validate_pace_linear_embedding(payload, species_index):
    path = f"embeddings.{species_index}"
    embedding = _require_mapping(payload, path)
    _require_exact_fields(embedding, _PACE_EMBEDDING_FIELDS, path)
    ndensity = _require_integer(
        embedding["ndensity"],
        f"{path}.ndensity",
        minimum=1,
    )
    if ndensity != 1:
        _yace_error(f"{path}.ndensity", "only ndensity=1 is supported")
    parameters = _require_finite_sequence(
        embedding["FS_parameters"],
        f"{path}.FS_parameters",
        length=2,
    )
    if parameters[1] != 1.0:
        _yace_error(
            f"{path}.FS_parameters[1]",
            "only the linear mexp=1 embedding is supported",
        )
    supported_embeddings = {
        "FinnisSinclair",
        "FinnisSinclairShiftedScaled",
    }
    if embedding["npoti"] not in supported_embeddings:
        _yace_error(
            f"{path}.npoti",
            "supported values are FinnisSinclair and "
            "FinnisSinclairShiftedScaled with mexp=1",
        )
    rho_cutoff = _require_finite(
        embedding["rho_core_cutoff"],
        f"{path}.rho_core_cutoff",
    )
    drho_cutoff = _require_finite(
        embedding["drho_core_cutoff"],
        f"{path}.drho_core_cutoff",
    )
    if rho_cutoff <= 0.0:
        _yace_error(f"{path}.rho_core_cutoff", "must be positive")
    if drho_cutoff < 0.0 or drho_cutoff >= rho_cutoff:
        _yace_error(
            f"{path}.drho_core_cutoff",
            "must satisfy 0 <= drho_core_cutoff < rho_core_cutoff",
        )


def _validate_pace_linear_bond(payload, central_species, neighbor_species):
    path = f"bonds.[{central_species},{neighbor_species}]"
    bond = _require_mapping(payload, path)
    _require_exact_fields(bond, _PACE_BOND_FIELDS, path)
    nradmax = _require_integer(bond["nradmax"], f"{path}.nradmax", minimum=1)
    lmax = _require_integer(bond["lmax"], f"{path}.lmax", minimum=0)
    nradbasemax = _require_integer(
        bond["nradbasemax"],
        f"{path}.nradbasemax",
        minimum=1,
    )
    if bond["radbasename"] != "ChebExpCos":
        _yace_error(f"{path}.radbasename", "only ChebExpCos is supported")
    parameters = _require_finite_sequence(
        bond["radparameters"],
        f"{path}.radparameters",
        length=1,
    )
    if parameters[0] <= 0.0:
        _yace_error(f"{path}.radparameters[0]", "must be positive")
    coefficients = bond["radcoefficients"]
    if not isinstance(coefficients, list) or len(coefficients) != nradmax:
        actual = len(coefficients) if isinstance(coefficients, list) else "non-sequence"
        _yace_error(
            f"{path}.radcoefficients",
            f"expected outer length {nradmax}, got {actual}",
        )
    for n, angular_rows in enumerate(coefficients):
        row_path = f"{path}.radcoefficients[{n}]"
        if not isinstance(angular_rows, list) or len(angular_rows) != lmax + 1:
            actual = len(angular_rows) if isinstance(angular_rows, list) else "non-sequence"
            _yace_error(row_path, f"expected length {lmax + 1}, got {actual}")
        for ell, radial_row in enumerate(angular_rows):
            _require_finite_sequence(
                radial_row,
                f"{row_path}[{ell}]",
                length=nradbasemax,
            )
    if _require_finite(bond["prehc"], f"{path}.prehc") != 0.0:
        _yace_error(f"{path}.prehc", "hard-core repulsion is not supported")
    _require_finite(bond["lambdahc"], f"{path}.lambdahc")
    rcut = _require_finite(bond["rcut"], f"{path}.rcut")
    dcut = _require_finite(bond["dcut"], f"{path}.dcut")
    if rcut <= 0.0:
        _yace_error(f"{path}.rcut", "must be positive")
    if dcut <= 0.0 or dcut >= rcut:
        _yace_error(f"{path}.dcut", "must satisfy 0 < dcut < rcut")
    if _require_finite(bond["rcut_in"], f"{path}.rcut_in") != 0.0:
        _yace_error(f"{path}.rcut_in", "active inner cutoffs are not supported")
    if _require_finite(bond["dcut_in"], f"{path}.dcut_in") != 0.0:
        _yace_error(f"{path}.dcut_in", "active inner cutoffs are not supported")
    if bond["inner_cutoff_type"] != "distance":
        _yace_error(
            f"{path}.inner_cutoff_type",
            "only an inactive distance inner cutoff is supported",
        )
    return nradmax, lmax, nradbasemax


def _validate_pace_linear_function(
    payload,
    index,
    central_species,
    element_count,
    bond_dimensions,
):
    path = f"functions.{central_species}[{index}]"
    function = _require_mapping(payload, path)
    _require_exact_fields(function, _PACE_FUNCTION_FIELDS, path)
    if (
        _require_integer(function["mu0"], f"{path}.mu0", minimum=0)
        != central_species
    ):
        _yace_error(
            f"{path}.mu0",
            f"must match central-species bucket {central_species}",
        )
    rank = _require_integer(function["rank"], f"{path}.rank", minimum=1)
    ndensity = _require_integer(
        function["ndensity"],
        f"{path}.ndensity",
        minimum=1,
    )
    if ndensity != 1:
        _yace_error(f"{path}.ndensity", "must match embedding ndensity=1")
    num_ms_combs = _require_integer(
        function["num_ms_combs"],
        f"{path}.num_ms_combs",
        minimum=1,
    )
    integer_fields = {}
    for name in ("mus", "ns", "ls", "ms_combs"):
        values = function[name]
        if not isinstance(values, list):
            _yace_error(f"{path}.{name}", "expected a sequence")
        integer_fields[name] = tuple(
            _require_integer(value, f"{path}.{name}[{position}]")
            for position, value in enumerate(values)
        )
    for name in ("mus", "ns", "ls"):
        if len(integer_fields[name]) != rank:
            _yace_error(
                f"{path}.{name}",
                f"expected length {rank}, got {len(integer_fields[name])}",
            )
    for slot, (neighbor_species, radial_index, angular_index) in enumerate(
        zip(
            integer_fields["mus"],
            integer_fields["ns"],
            integer_fields["ls"],
        )
    ):
        if neighbor_species < 0 or neighbor_species >= element_count:
            _yace_error(
                f"{path}.mus[{slot}]",
                f"species index must lie in [0, {element_count - 1}]",
            )
        nradmax, lmax, nradbasemax = bond_dimensions[
            (central_species, neighbor_species)
        ]
        maximum_n = nradbasemax if rank == 1 else nradmax
        if radial_index < 1 or radial_index > maximum_n:
            _yace_error(
                f"{path}.ns[{slot}]",
                "one-based radial index for bond "
                f"[{central_species},{neighbor_species}] must lie in "
                f"[1, {maximum_n}]",
            )
        if angular_index < 0 or angular_index > lmax:
            _yace_error(
                f"{path}.ls[{slot}]",
                "angular index for bond "
                f"[{central_species},{neighbor_species}] must lie in "
                f"[0, {lmax}]",
            )
    if rank == 1 and integer_fields["ls"] != (0,):
        _yace_error(f"{path}.ls", "rank-one functions require l=0")
    expected_ms = rank * num_ms_combs
    if len(integer_fields["ms_combs"]) != expected_ms:
        _yace_error(
            f"{path}.ms_combs",
            f"expected length {expected_ms}, got {len(integer_fields['ms_combs'])}",
        )
    for row in range(num_ms_combs):
        ms = integer_fields["ms_combs"][row * rank:(row + 1) * rank]
        if any(abs(m) > ell for m, ell in zip(ms, integer_fields["ls"])):
            _yace_error(f"{path}.ms_combs", f"row {row} has |m| > l")
        if sum(ms) != 0:
            _yace_error(f"{path}.ms_combs", f"row {row} does not couple to M=0")
    _require_finite_sequence(
        function["ctildes"],
        f"{path}.ctildes",
        length=ndensity * num_ms_combs,
    )


def _validate_lammps_pace_linear_v1(raw):
    document = _require_mapping(raw, "yace")
    _require_exact_fields(document, _PACE_ROOT_FIELDS, "yace")
    elements = document["elements"]
    if not isinstance(elements, list) or not elements:
        _yace_error("elements", "expected one or more element names")
    for species_index, element in enumerate(elements):
        if not isinstance(element, str) or not element:
            _yace_error(
                f"elements[{species_index}]",
                "expected a nonempty element name",
            )
    if len(set(elements)) != len(elements):
        _yace_error("elements", "element names must be unique")
    element_count = len(elements)
    species_indices = set(range(element_count))
    _require_finite_sequence(
        document["E0"],
        "E0",
        length=element_count,
    )
    if _require_finite(document["deltaSplineBins"], "deltaSplineBins") <= 0.0:
        _yace_error("deltaSplineBins", "must be positive")
    embeddings = _require_mapping(document["embeddings"], "embeddings")
    if set(embeddings) != species_indices or any(
        isinstance(key, bool) or not isinstance(key, int)
        for key in embeddings
    ):
        _yace_error(
            "embeddings",
            "expected exactly one key for each species index "
            f"{sorted(species_indices)}",
        )
    for species_index in range(element_count):
        _validate_pace_linear_embedding(
            embeddings[species_index],
            species_index,
        )
    bonds = _require_mapping(document["bonds"], "bonds")
    expected_bonds = {
        (central_species, neighbor_species)
        for central_species in range(element_count)
        for neighbor_species in range(element_count)
    }
    malformed_bond_key = any(
        not isinstance(key, tuple)
        or len(key) != 2
        or any(
            isinstance(species, bool) or not isinstance(species, int)
            for species in key
        )
        for key in bonds
    )
    if set(bonds) != expected_bonds or malformed_bond_key:
        _yace_error(
            "bonds",
            "expected one directed record for every ordered species pair "
            f"{sorted(expected_bonds)}",
        )
    bond_dimensions = {
        bond: _validate_pace_linear_bond(bonds[bond], *bond)
        for bond in sorted(expected_bonds)
    }
    functions = _require_mapping(document["functions"], "functions")
    if set(functions) != species_indices or any(
        isinstance(key, bool) or not isinstance(key, int)
        for key in functions
    ):
        _yace_error(
            "functions",
            "expected exactly one bucket for each central species index "
            f"{sorted(species_indices)}",
        )
    for central_species in range(element_count):
        species_functions = functions[central_species]
        if not isinstance(species_functions, list) or not species_functions:
            _yace_error(
                f"functions.{central_species}",
                "expected a nonempty function sequence",
            )
        for index, function in enumerate(species_functions):
            _validate_pace_linear_function(
                function,
                index,
                central_species,
                element_count,
                bond_dimensions,
            )


def write_yace(
    path,
    *,
    elements,
    functions_by_mu0,
    E0 = None,
    embeddings = None,
    bonds = None,
    delta_spline_bins = None,
    metadata = None,
    compatibility = None,
):
    """Write a `.yace` file with the expected YAML block structure.

    When ``compatibility`` is ``lammps_pace_linear_v1``, validate the complete
    in-memory document and its serialized YAML before replacing ``path``.  The
    check is initialization-only; it adds no work to descriptor evaluation.
    """
    output = Path(path)
    normalized = _normalize_functions(functions_by_mu0)
    document = {
        "elements": list(elements),
        "E0": list([0.0] * len(elements) if E0 is None else E0),
        "embeddings": dict({} if embeddings is None else embeddings),
        "bonds": dict({} if bonds is None else bonds),
        "functions": {
            int(mu0): [entry.as_mapping() for entry in entries]
            for mu0, entries in sorted(normalized.items())
        },
    }
    if delta_spline_bins is not None:
        document["deltaSplineBins"] = delta_spline_bins
    if metadata:
        extra = dict(metadata)
        duplicate = sorted(set(document).intersection(extra), key=repr)
        if duplicate:
            raise ValueError(
                "metadata must not replace YACE field " + repr(duplicate[0])
            )
        document.update(extra)
    if compatibility is not None:
        if compatibility != _PACE_LINEAR_COMPATIBILITY:
            raise ValueError(
                "compatibility: unsupported profile "
                f"{compatibility!r}; expected {_PACE_LINEAR_COMPATIBILITY!r}"
            )
        _validate_lammps_pace_linear_v1(document)
    serialized = yaml.safe_dump(
        document,
        sort_keys=False,
        default_flow_style=False,
    )
    if compatibility is not None:
        serialized_document = yaml.load(serialized, Loader=_YACELoader)
        _validate_lammps_pace_linear_v1(serialized_document)
    output.write_text(serialized, encoding="utf-8")
    return output


def read_yace(path, *, compatibility=None):
    """Read a `.yace` file and optionally validate a compatibility profile.

    Purpose:
        Load standard YACE data, including sequence-valued PACE bond keys.
    Mathematical contract:
        ``lammps_pace_linear_v1`` accepts the declared one-or-more-element,
        one-density-per-element, linear-embedding, no-core PACE subset and
        validates all explicit function dimensions before normalization.
    Inputs:
        A file path and an optional compatibility-profile name.
    Outputs:
        The parsed document with function records normalized to
        :class:`YACEFunction` objects.
    Does not:
        Infer coupling paths from C-tilde rows or add validation work to model
        evaluation; this profile is checked once while loading.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_YACELoader) or {}
    if compatibility is not None:
        if compatibility != _PACE_LINEAR_COMPATIBILITY:
            raise ValueError(
                "compatibility: unsupported profile "
                f"{compatibility!r}; expected {_PACE_LINEAR_COMPATIBILITY!r}"
            )
        _validate_lammps_pace_linear_v1(raw)
    if not isinstance(raw, dict):
        raise ValueError("yace: expected a mapping")
    functions_block = raw.get("functions", {})
    normalized = _normalize_functions({int(mu0): entries for mu0, entries in functions_block.items()})
    raw["functions"] = normalized
    return raw


def read_yace_functions(path, *, compatibility=None):
    """Convenience wrapper that returns only the normalized `functions:` block."""
    return read_yace(path, compatibility=compatibility)["functions"]
