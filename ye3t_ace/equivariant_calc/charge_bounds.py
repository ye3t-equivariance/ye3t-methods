
"""Helpers for choosing bounded scalar-charge ranges for charge ACE features."""

import numpy as np
from ye3t_ace._record import recordclass


COMMON_OXIDATION_STATE_RANGES = {
    "H": (-1.0, 1.0),
    "Li": (0.0, 1.0),
    "Na": (0.0, 1.0),
    "K": (0.0, 1.0),
    "Rb": (0.0, 1.0),
    "Cs": (0.0, 1.0),
    "Be": (0.0, 2.0),
    "Mg": (0.0, 2.0),
    "Ca": (0.0, 2.0),
    "Sr": (0.0, 2.0),
    "Ba": (0.0, 2.0),
    "B": (-3.0, 3.0),
    "C": (-4.0, 4.0),
    "N": (-3.0, 5.0),
    "O": (-2.0, 0.0),
    "F": (-1.0, 0.0),
    "Al": (0.0, 3.0),
    "Si": (-4.0, 4.0),
    "P": (-3.0, 5.0),
    "S": (-2.0, 6.0),
    "Cl": (-1.0, 0.0),
    "Ti": (0.0, 4.0),
    "V": (0.0, 5.0),
    "Cr": (0.0, 6.0),
    "Mn": (0.0, 7.0),
    "Fe": (0.0, 3.0),
    "Co": (0.0, 3.0),
    "Ni": (0.0, 3.0),
    "Cu": (0.0, 2.0),
    "Zn": (0.0, 2.0),
    "Br": (-1.0, 0.0),
    "I": (-1.0, 0.0),
    "Pt": (0.0, 6.0),
}


@recordclass(('elements', 'q_min', 'q_max', 'source_by_element'), frozen = True)
class ChargeBounds:

    def as_site_basis_kwargs(self):
        return {"q_min": self.q_min, "q_max": self.q_max}

    def as_dict(self):
        return {
            "elements": list(self.elements),
            "q_min": list(self.q_min),
            "q_max": list(self.q_max),
            "source_by_element": dict(self.source_by_element),
        }


def _ordered_elements(
    elements = None,
    *,
    type_map = None,
):
    if elements is not None:
        ordered = tuple(str(elem) for elem in elements)
    elif type_map is not None:
        ordered = tuple(str(elem) for elem, _ in sorted(type_map.items(), key=lambda item: int(item[1])))
    else:
        raise ValueError("Provide either elements or type_map.")
    if not ordered:
        raise ValueError("At least one element is required.")
    return ordered


def _pymatgen_oxidation_range(element):
    try:
        from pymatgen.core import Element  # type: ignore
    except ImportError:
        return None
    try:
        elem = Element(str(element))
        states = tuple(elem.common_oxidation_states) or tuple(elem.oxidation_states)
    except (AttributeError, TypeError, ValueError):
        return None
    if not states:
        return None
    return float(min(states)), float(max(states))


def _widen_if_needed(qmin, qmax, *, min_width):
    width = float(qmax) - float(qmin)
    if width >= float(min_width):
        return float(qmin), float(qmax)
    midpoint = 0.5 * (float(qmin) + float(qmax))
    half_width = 0.5 * float(min_width)
    return midpoint - half_width, midpoint + half_width


def oxidation_state_charge_bounds(
    elements = None,
    *,
    type_map = None,
    source = "auto",
    fallback = (-1.0, 1.0),
    padding = 0.0,
    min_width = 1.0e-8,
):
    """Return per-element charge bounds from oxidation-state ranges.

    ``source="auto"`` uses ``pymatgen`` if installed and otherwise falls back
    to the small curated table above. ASE does not currently provide oxidation
    state ranges in ``ase.data``.
    """

    ordered = _ordered_elements(elements, type_map=type_map)
    q_min = []
    q_max = []
    source_by_element = {}
    for elem in ordered:
        bounds = None
        source_name = ""
        if source in {"auto", "pymatgen"}:
            bounds = _pymatgen_oxidation_range(elem)
            source_name = "pymatgen_oxidation_states" if bounds is not None else source_name
        if bounds is None and source in {"auto", "curated"}:
            bounds = COMMON_OXIDATION_STATE_RANGES.get(elem)
            source_name = "curated_common_oxidation_states" if bounds is not None else source_name
        if bounds is None:
            bounds = fallback
            source_name = "fallback"
        lo, hi = _widen_if_needed(float(bounds[0]) - float(padding), float(bounds[1]) + float(padding), min_width=min_width)
        q_min.append(lo)
        q_max.append(hi)
        source_by_element[elem] = source_name
    return ChargeBounds(elements=ordered, q_min=tuple(q_min), q_max=tuple(q_max), source_by_element=source_by_element)


def charge_bounds_from_data(
    charges,
    *,
    elements = None,
    symbols = None,
    atom_types = None,
    type_map = None,
    quantile = 0.01,
    padding = 0.05,
    min_width = 0.25,
):
    """Estimate robust per-type charge bounds from observed atomic charges."""

    ordered = _ordered_elements(elements, type_map=type_map)
    charges_arr = np.asarray(charges, dtype=float).reshape(-1)
    if symbols is not None:
        if len(symbols) != charges_arr.shape[0]:
            raise ValueError("symbols and charges must have the same length.")
        labels = np.asarray([str(sym) for sym in symbols], dtype=object)
    elif atom_types is not None:
        atom_types_arr = np.asarray(atom_types, dtype=int).reshape(-1)
        if atom_types_arr.shape[0] != charges_arr.shape[0]:
            raise ValueError("atom_types and charges must have the same length.")
        inverse = (
            {int(idx): str(elem) for elem, idx in type_map.items()}
            if type_map is not None
            else {idx: elem for idx, elem in enumerate(ordered)}
        )
        labels = np.asarray([inverse[int(t)] for t in atom_types_arr], dtype=object)
    else:
        raise ValueError("Provide either symbols or atom_types with charges.")
    q = float(np.clip(quantile, 0.0, 0.5))

    q_min = []
    q_max = []
    source_by_element = {}
    for elem in ordered:
        values = charges_arr[labels == elem]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite charge values found for element {elem!r}.")
        lo = float(np.quantile(values, q)) - float(padding)
        hi = float(np.quantile(values, 1.0 - q)) + float(padding)
        lo, hi = _widen_if_needed(lo, hi, min_width=min_width)
        q_min.append(lo)
        q_max.append(hi)
        source_by_element[elem] = f"data_quantile_{q:g}_{1.0 - q:g}"
    return ChargeBounds(elements=ordered, q_min=tuple(q_min), q_max=tuple(q_max), source_by_element=source_by_element)


def resolve_charge_bounds(
    elements = None,
    *,
    charges = None,
    symbols = None,
    atom_types = None,
    type_map = None,
    strategy = "data_or_oxidation",
    min_samples_per_type = 4,
    data_quantile = 0.01,
    data_padding = 0.05,
    data_min_width = 0.25,
    oxidation_padding = 0.0,
    fallback = (-1.0, 1.0),
):
    """Resolve charge bounds for ``SiteBasisConfig``.

    Strategies:

    - ``"data"``: require observed charges.
    - ``"oxidation"``: use oxidation-state ranges only.
    - ``"data_or_oxidation"``: use data when each type has enough samples,
      otherwise use oxidation-state ranges.
    """

    ordered = _ordered_elements(elements, type_map=type_map)
    if strategy not in {"data", "oxidation", "data_or_oxidation"}:
        raise ValueError("strategy must be one of data, oxidation, data_or_oxidation")
    if strategy == "oxidation":
        return oxidation_state_charge_bounds(ordered, fallback=fallback, padding=oxidation_padding)
    if charges is None:
        if strategy == "data":
            raise ValueError("charges are required for strategy='data'.")
        return oxidation_state_charge_bounds(ordered, source="curated", fallback=fallback, padding=oxidation_padding)

    if strategy == "data":
        return charge_bounds_from_data(
            charges,
            elements=ordered,
            symbols=symbols,
            atom_types=atom_types,
            quantile=data_quantile,
            padding=data_padding,
            min_width=data_min_width,
        )

    try:
        if symbols is not None:
            counts = {elem: sum(str(sym) == elem for sym in symbols) for elem in ordered}
        elif atom_types is not None:
            atom_types_arr = np.asarray(atom_types, dtype=int)
            type_ids = (
                {str(elem): int(idx) for elem, idx in type_map.items()}
                if type_map is not None
                else {elem: idx for idx, elem in enumerate(ordered)}
            )
            counts = {elem: int(np.sum(atom_types_arr == type_ids[elem])) for elem in ordered}
        else:
            counts = {elem: 0 for elem in ordered}
        if all(counts[elem] >= int(min_samples_per_type) for elem in ordered):
            return charge_bounds_from_data(
                charges,
                elements=ordered,
                symbols=symbols,
                atom_types=atom_types,
                quantile=data_quantile,
                padding=data_padding,
                min_width=data_min_width,
            )
    except (IndexError, KeyError, TypeError, ValueError):
        return oxidation_state_charge_bounds(ordered, source="curated", fallback=fallback, padding=oxidation_padding)
    return oxidation_state_charge_bounds(ordered, source="curated", fallback=fallback, padding=oxidation_padding)
