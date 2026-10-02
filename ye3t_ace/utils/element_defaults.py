"""Element defaults for high-level ACE descriptor workflows."""

from collections.abc import Iterable, Mapping

import numpy as np

try:
    from ase.data import atomic_numbers, covalent_radii, vdw_radii
except Exception:  # pragma: no cover - optional dependency
    atomic_numbers = None  # type: ignore
    covalent_radii = None  # type: ignore
    vdw_radii = None  # type: ignore


def build_explicit_type_map(elements):
    """Return ``{symbol: type_id}`` in the supplied element order."""

    ordered = [str(elem) for elem in elements]
    if not ordered:
        raise ValueError("Need at least one element to build a type map.")
    if len(set(ordered)) != len(ordered):
        raise ValueError("Element list contains duplicates.")
    return {elem: idx for idx, elem in enumerate(ordered)}


def infer_elements_from_ase_atoms(atoms_or_structures):
    """Infer sorted chemical symbols from one ASE ``Atoms`` object or an iterable."""

    if hasattr(atoms_or_structures, "get_chemical_symbols"):
        symbols = atoms_or_structures.get_chemical_symbols()
    else:
        symbols = []
        for atoms in atoms_or_structures:
            if not hasattr(atoms, "get_chemical_symbols"):
                raise TypeError("Expected ASE Atoms objects when inferring elements.")
            symbols.extend(atoms.get_chemical_symbols())
    ordered = tuple(sorted({str(symbol) for symbol in symbols}))
    if not ordered:
        raise ValueError("Could not infer elements from empty structures.")
    return ordered


def type_ids_from_symbols(symbols, type_map):
    """Map chemical symbols to integer type ids."""

    mapping = {str(symbol): int(type_id) for symbol, type_id in dict(type_map).items()}
    return np.asarray([mapping[str(symbol)] for symbol in symbols], dtype=int)


def vdw_radius_for_element(element):
    """Return a usable van der Waals radius in Angstrom, with covalent fallback."""

    if atomic_numbers is None or vdw_radii is None:
        raise ImportError("ASE is required for van der Waals radius defaults.")
    z = int(atomic_numbers[str(element)])
    radius = float(vdw_radii[z])
    if not np.isfinite(radius) or radius <= 0.0:
        if covalent_radii is None:
            raise ValueError(f"No usable radius found for element {element!r}.")
        radius = float(covalent_radii[z])
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"No usable radius found for element {element!r}.")
    return radius


def vdw_scaled_bond_defaults(
    elements,
    *,
    type_map=None,
    cutoff_scale=1.0,
    lambda_scale=1.0,
    min_cutoff=1.5,
):
    """Return ordered-bond ``rc`` and ``lmbda`` defaults from VdW radii.

    For a same-species pair and ``cutoff_scale=1``, the cutoff is
    ``2 * r_vdw``. The returned ``lmbda`` values follow the existing
    ``SiteBasisConfig`` radial-decay convention: ``lambda_scale / (r_i + r_j)``.
    """

    ordered = tuple(str(elem) for elem in elements)
    resolved_type_map = build_explicit_type_map(ordered) if type_map is None else dict(type_map)
    radii_by_element = {elem: vdw_radius_for_element(elem) for elem in ordered}
    rc_by_pair = {}
    lmbda_by_pair = {}
    for center in ordered:
        for neighbor in ordered:
            pair = (int(resolved_type_map[center]), int(resolved_type_map[neighbor]))
            pair_radius = float(radii_by_element[center] + radii_by_element[neighbor])
            rc_by_pair[pair] = max(float(min_cutoff), float(cutoff_scale) * pair_radius)
            lmbda_by_pair[pair] = float(lambda_scale) / max(pair_radius, 1.0e-12)
    return rc_by_pair, lmbda_by_pair, radii_by_element


def ordered_pair_values(values, *, possible_types, name):
    """Return values in ``SiteBasisConfig`` ordered-bond convention."""

    pairs = [(int(left), int(right)) for left in possible_types for right in possible_types]
    if isinstance(values, Mapping):
        out = []
        for pair in pairs:
            if pair in values:
                out.append(float(values[pair]))
            elif tuple(pair) in values:
                out.append(float(values[tuple(pair)]))
            else:
                raise KeyError(f"Missing {name} value for ordered type pair {pair}.")
        return out
    if isinstance(values, (float, int)):
        return [float(values)] * len(pairs)
    if isinstance(values, Iterable):
        vals = [float(v) for v in values]
        if len(vals) == 1:
            return vals * len(pairs)
        if len(vals) == len(pairs):
            return vals
    raise ValueError(f"{name} must be a scalar, length-1 sequence, or length n_types**2 sequence.")


# Starting-point radii and metal/nonmetal branching for heuristic ACE cutoff
# initialization.  These values are used only as starting settings, not as
# fitted or validated physical cutoffs.
ACE_STARTING_IONIC_RADII = {
    "H": 0.25,
    "O": 0.60,
    "Cl": 1.00,
    "K": 2.20,
    "Ta": 1.45,
    "Pt": 1.35,
    "Au": 1.35,
    "Ag": 1.60,
    "Cu": 1.35,
    "Pd": 1.40,
}

ACE_STARTING_METALS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Cs",
    "Ba",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Fr",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Yb",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
}


def _ace_starting_ionic_radius(element):
    symbol = str(element)
    if symbol in ACE_STARTING_IONIC_RADII:
        return float(ACE_STARTING_IONIC_RADII[symbol])
    if atomic_numbers is None or covalent_radii is None:
        raise ImportError("ASE is required to fall back from ACE starting ionic radii.")
    z = int(atomic_numbers[symbol])
    radius = float(covalent_radii[z])
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"No ACE starting ionic-radius point is available for {symbol!r}; "
            "pass ionic_radii_override for this element."
        )
    return radius


def _ace_starting_default_rc_range(element_a, element_b, *, use_vdw=False, metal_max=True, ionic_radii_override=None):
    symbol_a = str(element_a)
    symbol_b = str(element_b)
    radii_override = {} if ionic_radii_override is None else {str(k): float(v) for k, v in dict(ionic_radii_override).items()}

    def ionic(symbol):
        return radii_override.get(symbol, _ace_starting_ionic_radius(symbol))

    if atomic_numbers is None or covalent_radii is None or vdw_radii is None:
        raise ImportError("ASE is required for ACE pair cutoff starting defaults.")
    ion_a = float(ionic(symbol_a))
    ion_b = float(ionic(symbol_b))
    z_a = int(atomic_numbers[symbol_a])
    z_b = int(atomic_numbers[symbol_b])
    vdw_a = float(vdw_radii[z_a])
    vdw_b = float(vdw_radii[z_b])
    if not np.isfinite(vdw_a) or vdw_a <= 0.0:
        vdw_a = 2.0 * float(covalent_radii[z_a])
    if not np.isfinite(vdw_b) or vdw_b <= 0.0:
        vdw_b = 2.0 * float(covalent_radii[z_b])

    minbond = ion_a + ion_b
    if metal_max:
        a_metal = symbol_a in ACE_STARTING_METALS
        b_metal = symbol_b in ACE_STARTING_METALS
        if not a_metal and not b_metal:
            maxbond = vdw_a + vdw_b
        elif a_metal and not b_metal:
            maxbond = ion_a + vdw_b
            minbond = 0.8 * (ion_a + ion_b)
        elif a_metal and b_metal:
            maxbond = ion_a + ion_b
            minbond = 0.8 * (ion_a + ion_b)
        else:
            maxbond = ion_a + ion_b
    else:
        maxbond = vdw_a + vdw_b
    returnmin = minbond
    returnmax = maxbond if bool(use_vdw) else 0.5 * (maxbond + minbond)
    return round(float(returnmin), 3), round(float(returnmax), 3)


def ace_pair_starting_defaults(
    elements,
    *,
    type_map=None,
    nshell=1.0,
    use_vdw=False,
    metal_max=True,
    inner_fraction=0.25,
    lambda_scale=0.05,
    ionic_radii_override=None,
):
    """Return ordered-pair ACE cutoff and radial starting points.

    For each ordered type pair it returns a pair cutoff, an inner cutoff, and
    a radial ``lambda`` starting value. These are modeling starting points only;
    they should be validated by cross validation, force/energy errors,
    coefficient magnitudes, and MD stability checks before publication use.
    """

    ordered = tuple(str(elem) for elem in elements)
    resolved_type_map = build_explicit_type_map(ordered) if type_map is None else dict(type_map)
    pair_cutoffs = {}
    pair_inner_cutoffs = {}
    pair_radial_lambdas = {}
    pair_ranges = {}
    for center in ordered:
        for neighbor in ordered:
            rc_min, rc_max = _ace_starting_default_rc_range(
                center,
                neighbor,
                use_vdw=use_vdw,
                metal_max=metal_max,
                ionic_radii_override=ionic_radii_override,
            )
            cutoff = float(nshell) * ((float(rc_max) + float(rc_min)) / 1.8)
            pair = (int(resolved_type_map[center]), int(resolved_type_map[neighbor]))
            pair_ranges[pair] = (float(rc_min), float(rc_max))
            pair_cutoffs[pair] = float(cutoff)
            pair_inner_cutoffs[pair] = float(inner_fraction) * float(rc_min)
            pair_radial_lambdas[pair] = float(lambda_scale) * float(cutoff)
    return {
        "pair_cutoffs": pair_cutoffs,
        "pair_inner_cutoffs": pair_inner_cutoffs,
        "pair_radial_lambdas": pair_radial_lambdas,
        "pair_ranges": pair_ranges,
        "source": "ACE pair starting heuristic",
    }
