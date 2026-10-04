
"""Descriptor-set helpers built on compact exact ACE labels."""
import warnings
from itertools import product

from ye3t_ace.cache import (
    DescriptorBuildCache,
    DescriptorEnumerationCacheKey,
    descriptor_artifact_cache_key,
    resolve_descriptor_build_cache,
    settings_cache_key,
)
from ye3t.couplings import (
    compile_scalar_ace_coordinate,
    count as count_couplings,
)
from ye3t_ace.ace_labeler.validation import iter_canonical_leaf_labelings, validate_tree_type
from ye3t_ace.couplings.generalized import generate_library_for_labels
from .ace_eval_v2 import GeneralizedCouplingLibrary, _scalar_real_phase
from .labeling import CompactLabel, DescriptorSpec, SingleChannelLabel, normalize_compact_label
from ye3t_ace._record import recordclass


@recordclass(('ranks', 'basis_type', 'elems', 'nmax', 'lmax', 'lmin', 'L_R', 'M_R_values', 'k_o_max', 'k_max', 'aux_lmax', 'max_labels_per_rank', 'tree_type', 'parity_filter'), frozen = True)
class DescriptorGenerationSettings:
    M_R_values = None
    k_o_max = 0
    k_max = None
    aux_lmax = 0
    max_labels_per_rank = None
    tree_type = "balanced"
    parity_filter = "natural"

    def __post_init__(self):
        if len(self.ranks) != len(self.nmax) or len(self.ranks) != len(self.lmax) or len(self.ranks) != len(self.lmin):
            raise ValueError("ranks, nmax, lmax, and lmin must have same length")
        if self.basis_type not in {"no_charge", "charge", "tensor", "magnetic"}:
            raise ValueError("basis_type must be 'no_charge', 'charge', 'tensor', or 'magnetic'")
        if self.k_max is None:
            object.__setattr__(self, 'k_max', tuple(0 for _ in self.ranks))
        elif len(self.k_max) != len(self.ranks):
            raise ValueError("k_max must have same length as ranks")
        if self.M_R_values is None:
            object.__setattr__(self, 'M_R_values', tuple(range(-self.L_R, self.L_R + 1)))
        else:
            M_R_values = tuple(int(value) for value in self.M_R_values)
            invalid_M_R = tuple(value for value in M_R_values if abs(value) > int(self.L_R))
            if invalid_M_R:
                raise ValueError("M_R_values must satisfy -L_R <= M_R <= L_R.")
            object.__setattr__(self, 'M_R_values', M_R_values)
        object.__setattr__(self, 'tree_type', validate_tree_type(self.tree_type))
        if self.parity_filter not in {"natural", "none"}:
            raise ValueError("parity_filter must be 'natural' or 'none'")
        if any(int(rank) < 1 for rank in self.ranks):
            raise ValueError(f"All ranks must be positive integers; got ranks={tuple(self.ranks)}")
        if int(self.L_R) < 0:
            raise ValueError(f"L_R must be non-negative; got {self.L_R}")
        for rank, lmax, lmin in zip(self.ranks, self.lmax, self.lmin):
            if int(lmin) < 0 or int(lmax) < int(lmin):
                raise ValueError(f"Require 0 <= lmin <= lmax for each rank; got lmin={self.lmin}, lmax={self.lmax}")
            if int(self.L_R) > int(rank) * int(lmax):
                raise ValueError(
                    f"L_R={self.L_R} exceeds the largest possible coupled angular momentum rank*lmax={int(rank) * int(lmax)}"
                )

    @property
    def mu_values(self):
        return tuple(range(len(self.elems)))

    def rank_index(self, rank):
        return list(self.ranks).index(rank)

    def as_dict(self):
        """Serialize descriptor-generation settings into JSON-compatible metadata."""
        return {
            "ranks": [int(v) for v in self.ranks],
            "basis_type": str(self.basis_type),
            "elems": [str(v) for v in self.elems],
            "nmax": [int(v) for v in self.nmax],
            "lmax": [int(v) for v in self.lmax],
            "lmin": [int(v) for v in self.lmin],
            "L_R": int(self.L_R),
            "M_R_values": None if self.M_R_values is None else [int(v) for v in self.M_R_values],
            "k_o_max": int(self.k_o_max),
            "k_max": None if self.k_max is None else [int(v) for v in self.k_max],
            "aux_lmax": int(self.aux_lmax),
            "max_labels_per_rank": None if self.max_labels_per_rank is None else int(self.max_labels_per_rank),
            "tree_type": str(self.tree_type),
            "parity_filter": str(self.parity_filter),
        }

    @classmethod
    def from_dict(cls, payload):
        """Restore descriptor-generation settings from JSON-compatible metadata."""
        if isinstance(payload, cls):
            return payload
        known = {
            "ranks", "basis_type", "elems", "nmax", "lmax", "lmin", "L_R",
            "M_R_values", "k_o_max", "k_max", "aux_lmax",
            "max_labels_per_rank", "tree_type", "parity_filter",
        }
        unsupported = sorted(str(key) for key, value in payload.items()
                             if key not in known and value is not None)
        if unsupported:
            raise ValueError("Unsupported descriptor settings: " + ", ".join(unsupported))
        return cls(
            ranks=tuple(int(v) for v in payload["ranks"]),
            basis_type=str(payload.get("basis_type", "no_charge")),
            elems=tuple(str(v) for v in payload.get("elems", ())),
            nmax=tuple(int(v) for v in payload["nmax"]),
            lmax=tuple(int(v) for v in payload["lmax"]),
            lmin=tuple(int(v) for v in payload.get("lmin", tuple(0 for _ in payload["ranks"]))),
            L_R=int(payload["L_R"]),
            M_R_values=None if payload.get("M_R_values") is None else tuple(int(v) for v in payload.get("M_R_values")),
            k_o_max=int(payload.get("k_o_max", 0)),
            k_max=None if payload.get("k_max") is None else tuple(int(v) for v in payload.get("k_max")),
            aux_lmax=int(payload.get("aux_lmax", 0)),
            max_labels_per_rank=payload.get("max_labels_per_rank"),
            tree_type=str(payload.get("tree_type", "balanced")),
            parity_filter=str(payload.get("parity_filter", "natural")),
        )


@recordclass(('mu0', 'mus', 'kappa0s', 'kappas', 'aux_lms'), frozen = True)
class ChannelVariant:
    aux_lms = None

    def suffix(self):
        mus = ",".join(map(str, self.mus))
        kappa0s = ",".join(map(str, self.kappa0s))
        kappas = ",".join(map(str, self.kappas))
        aux = ""
        if self.aux_lms is not None:
            aux = "|aux=" + ";".join(f"{l}:{m}" for l, m in self.aux_lms)
        return f"mu0={self.mu0}|mus={mus}|kappa0={kappa0s}|kappa={kappas}{aux}"


@recordclass(('compact_labels', 'specs_by_M'), frozen = True)
class DescriptorCollection:
    pass


def _balanced_variant_subset(variants, max_variants):
    """Return a deterministic spread across the sorted chemical/charge variants."""
    indices = _balanced_variant_indices(len(variants), max_variants)
    if len(indices) == len(variants):
        return variants
    return [variants[index] for index in indices]


def _balanced_variant_indices(variant_count, max_variants):
    variant_count = int(variant_count)
    limit = int(max_variants)
    if limit >= variant_count:
        return tuple(range(variant_count))
    if limit <= 0 or variant_count <= 0:
        return tuple()
    if limit == 1:
        return (0,)
    last = variant_count - 1
    indices = []
    seen = set()
    for idx in range(limit):
        pos = int(round(float(idx * last) / float(limit - 1)))
        if pos not in seen:
            indices.append(pos)
            seen.add(pos)
    candidate = 0
    while len(indices) < limit and candidate < variant_count:
        if candidate not in seen:
            indices.append(candidate)
            seen.add(candidate)
        candidate += 1
    return tuple(sorted(indices))


def count_channel_variants(
    label,
    settings,
    center_mu_values=None,
    restrict_neighbor_mu=None,
    max_variants_per_label=None,
):
    """Count descriptor channel variants without materializing them."""

    label = normalize_compact_label(label)
    rank = int(label.rank)
    center_values = tuple(
        settings.mu_values if center_mu_values is None else center_mu_values
    )
    neighbor_values = tuple(
        settings.mu_values
        if restrict_neighbor_mu is None
        else restrict_neighbor_mu
    )
    per_center = len(neighbor_values) ** rank
    if settings.basis_type in {"charge", "magnetic"}:
        center_charge_count = int(settings.k_o_max) + 1
        neighbor_charge_count = int(settings.k_max[settings.rank_index(rank)]) + 1
        per_center *= center_charge_count * neighbor_charge_count ** rank
    if settings.basis_type == "magnetic":
        auxiliary_state_count = (int(settings.aux_lmax) + 1) ** 2
        per_center *= auxiliary_state_count ** rank
    total = len(center_values) * per_center
    if max_variants_per_label is not None:
        total = min(total, max(0, int(max_variants_per_label)))
    return int(total)


def count_channel_variants_by_center(
    label,
    settings,
    center_mu_values=None,
    restrict_neighbor_mu=None,
    max_variants_per_label=None,
):
    """Count variants by central chemical channel without building them."""

    center_values = tuple(
        settings.mu_values if center_mu_values is None else center_mu_values
    )
    total = count_channel_variants(
        label,
        settings,
        center_mu_values=center_mu_values,
        restrict_neighbor_mu=restrict_neighbor_mu,
        max_variants_per_label=None,
    )
    if not center_values:
        return {}
    per_center = total // len(center_values)
    if max_variants_per_label is None or int(max_variants_per_label) >= total:
        return {int(value): int(per_center) for value in center_values}
    counts = {int(value): 0 for value in center_values}
    for index in _balanced_variant_indices(total, max_variants_per_label):
        center_index = min(index // per_center, len(center_values) - 1)
        counts[int(center_values[center_index])] += 1
    return counts


@recordclass(('values', 'M_values'), frozen = True)
class MultipletTensor:
    pass


def normalize_basis_mode(basis_mode, *, L_R = None):
    r"""Normalize the public graded-algebra basis-selection mode.

    Public modes
    ------------
    ``None`` or ``"exact"``
        Keep the original fixed-rank exact basis constructed directly from the
        tree-agnostic Schur--Weyl + Young-subgroup machinery.

    ``"primitive_invariant"``
        Quotient the scalar sector by the exact lower-generated invariant
        subspace.  This computes

        .. math::

            P_N^{(0)} = \mathcal A_N^{(0)} / \sum_{k=1}^{N-1}\mathcal A_k^{(0)}\,\mathcal A_{N-k}^{(0)}.

    ``"primitive_equivariant_module"``
        Quotient a target irrep sector by the exact invariant-times-equivariant
        generated subspace, i.e. view each equivariant sector as a graded module
        over the invariant ring.

    ``"primitive_full"``
        Quotient by the span of *all* lower-degree coupled products
        ``(N_1,L_1) \otimes (N_2,L_2) \to (N,L)``.

    Notes
    -----
    Primitive label validity is owned by ``ye3t.couplings``.  If the exact
    primitive quotient needs a conservative symbolic fallback, that fallback is
    reported by the ``MultiplicityReport`` validation metadata returned from
    ``ye3t.couplings.count``.
    """
    if basis_mode in {None, 'exact', 'full_exact_basis', 'original_exact'}:
        return None
    aliases = {
        'exact_invariant': 'primitive_invariant',
        'exact_equivariant_module': 'primitive_equivariant_module',
        'exact_full': 'primitive_full',
    }
    mode = aliases.get(basis_mode, basis_mode)
    valid = {'primitive_invariant', 'primitive_equivariant_module', 'primitive_full'}
    if mode not in valid:
        raise ValueError(f'Unsupported basis_mode={basis_mode!r}.')
    if mode == 'primitive_invariant' and L_R is not None and int(L_R) != 0:
        raise ValueError('primitive_invariant is only valid for scalar L_R=0 sectors.')
    return mode


def label_has_natural_parity(label, *, L_R):
    """Return whether a compact label has the natural SO(3) parity for ``L_R``."""

    return (sum(int(l) for l in label.l_tuple) - int(L_R)) % 2 == 0


def filter_compact_labels_for_settings(
    labels,
    settings,
):
    """Apply the descriptor settings' parity convention to compact labels."""

    if settings.parity_filter == "none":
        return list(labels)
    return [label for label in labels if label_has_natural_parity(label, L_R=settings.L_R)]


def select_compact_labels(
    settings,
    *,
    compact_labels = None,
    basis_mode = None,
    exact_primitive_timeout_seconds = None,
    descriptor_cache = None,
    use_descriptor_cache = True,
):
    """Select labels from settings or an explicit bank, including parity/reduction rules."""

    if compact_labels is None:
        return enumerate_compact_labels(
            settings,
            basis_mode=basis_mode,
            exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
            descriptor_cache=descriptor_cache,
            use_descriptor_cache=use_descriptor_cache,
        )
    selected = filter_compact_labels_for_settings(
        [normalize_compact_label(label) for label in compact_labels],
        settings,
    )
    selected_mode = normalize_basis_mode(basis_mode, L_R=settings.L_R)
    if selected_mode is None:
        return selected
    primitive_labels = set(
        enumerate_compact_labels(
            settings,
            basis_mode=selected_mode,
            exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
            descriptor_cache=descriptor_cache,
            use_descriptor_cache=use_descriptor_cache,
        )
    )
    return [label for label in selected if label in primitive_labels]


def enumerate_compact_labels(
    settings,
    primitive_mode = None,
    *,
    basis_mode = None,
    exact_primitive_timeout_seconds = None,
    descriptor_cache = None,
    use_descriptor_cache = True,
):
    r"""Enumerate compact labels, optionally filtering by an exact primitive mode."""
    selected_mode = normalize_basis_mode(basis_mode if basis_mode is not None else primitive_mode, L_R=settings.L_R)
    active_cache = resolve_descriptor_build_cache(descriptor_cache, use_cache=use_descriptor_cache)
    cache_key = DescriptorEnumerationCacheKey(
        settings=settings_cache_key(settings),
        basis_mode=None if selected_mode is None else str(selected_mode),
        exact_primitive_timeout_seconds=None if exact_primitive_timeout_seconds is None else float(exact_primitive_timeout_seconds),
    )
    cached = None if active_cache is None else active_cache.get_compact_labels(cache_key)
    if cached is not None:
        return list(cached)
    out = []
    seen = set()
    for idx, rank in enumerate(settings.ranks):
        n_values = range(1, settings.nmax[idx] + 1)
        l_values = range(settings.lmin[idx], settings.lmax[idx] + 1)
        per_rank = []
        rank_limit_reached = False
        for nin, lin in iter_canonical_leaf_labelings(rank, n_values, l_values):
            carrier_options = {}
            if selected_mode is not None:
                carrier_options = {
                    "basis_mode": selected_mode,
                    "primitive_fallback_policy": "symbolic",
                }
                if exact_primitive_timeout_seconds is not None:
                    carrier_options["exact_primitive_timeout_seconds"] = float(exact_primitive_timeout_seconds)
            report = count_couplings(
                content=tuple(nin),
                input_Ls=tuple(lin),
                target_L=int(settings.L_R),
                tree_schedule=settings.tree_type,
                carrier="ACE_density",
                carrier_options=carrier_options,
                target_permutation="trivial",
                validation_scope="counts",
                metadata={"consumer": "ye3t_ace.equivariant_calc.enumerate_compact_labels"},
            )
            primitive_validation = report.validation_report.get("primitive_validation")
            if primitive_validation is not None and primitive_validation.get("backend") == "symbolic_fallback":
                warnings.warn(
                    f'Exact primitive quotient did not complete in ye3t.couplings '
                    f'for nin={nin}, lin={lin}, L_R={settings.L_R}; using conservative fallback '
                    f'mode {primitive_validation.get("fallback_mode")!r}. '
                    f'Original exception: {primitive_validation.get("exact_error")}',
                    RuntimeWarning,
                )
            label_iter = [normalize_compact_label(x) for x in report.labels_for_target(settings.L_R)]
            for lab in label_iter:
                lab = normalize_compact_label(lab)
                if settings.parity_filter == "natural" and not label_has_natural_parity(lab, L_R=settings.L_R):
                    continue
                if lab not in seen:
                    seen.add(lab)
                    per_rank.append(lab)
                    if settings.max_labels_per_rank is not None and len(per_rank) >= settings.max_labels_per_rank:
                        rank_limit_reached = True
                        break
            if rank_limit_reached:
                break
        per_rank = sorted(
            per_rank,
            key=lambda x: (x.rank, x.n_tuple, x.l_tuple, x.internal_Ls, x.tree_type, x.basis_key),
        )
        out.extend(per_rank)
    if active_cache is not None:
        active_cache.put_compact_labels(cache_key, tuple(out))
    return out


def build_channel_variants(
    label,
    settings,
    center_mu_values = None,
    restrict_neighbor_mu = None,
    max_variants_per_label = None,
):
    rank = label.rank
    center_mu_values = tuple(settings.mu_values if center_mu_values is None else center_mu_values)
    neighbor_mu_values = tuple(settings.mu_values if restrict_neighbor_mu is None else restrict_neighbor_mu)
    kmax_rank = settings.k_max[settings.rank_index(rank)]
    variants = []
    for mu0 in center_mu_values:
        for mus in product(neighbor_mu_values, repeat=rank):
            if settings.basis_type == 'no_charge':
                variants.append(
                    ChannelVariant(
                        mu0=mu0,
                        mus=tuple(mus),
                        kappa0s=tuple(0 for _ in range(rank)),
                        kappas=tuple(0 for _ in range(rank)),
                    )
                )
            elif settings.basis_type == 'charge':
                for kappa0 in range(settings.k_o_max + 1):
                    for kappas in product(range(kmax_rank + 1), repeat=rank):
                        variants.append(
                            ChannelVariant(
                                mu0=mu0,
                                mus=tuple(mus),
                                kappa0s=tuple(kappa0 for _ in range(rank)),
                                kappas=tuple(kappas),
                            )
                        )
            elif settings.basis_type == 'magnetic':
                aux_states = [
                    (l_aux, m_aux)
                    for l_aux in range(settings.aux_lmax + 1)
                    for m_aux in range(-l_aux, l_aux + 1)
                ]
                for kappa0 in range(settings.k_o_max + 1):
                    for kappas in product(range(kmax_rank + 1), repeat=rank):
                        for aux_lms in product(aux_states, repeat=rank):
                            variants.append(
                                ChannelVariant(
                                    mu0=mu0,
                                    mus=tuple(mus),
                                    kappa0s=tuple(kappa0 for _ in range(rank)),
                                    kappas=tuple(kappas),
                                    aux_lms=tuple(aux_lms),
                                )
                            )
            else:  # tensor
                variants.append(
                    ChannelVariant(
                        mu0=mu0,
                        mus=tuple(mus),
                        kappa0s=tuple(0 for _ in range(rank)),
                        kappas=tuple(0 for _ in range(rank)),
                        aux_lms=tuple((0, 0) for _ in range(rank)),
                    )
                )
    variants = sorted(variants, key=lambda v: (v.mu0, v.mus, v.kappa0s, v.kappas, v.aux_lms or tuple()))
    if max_variants_per_label is not None:
        variants = _balanced_variant_subset(variants, max_variants_per_label)
    return variants


def _coerce_coeffs(coeffs_raw):
    coeffs = []
    for c in coeffs_raw:
        if isinstance(c, complex):
            coeffs.append(c)
        elif isinstance(c, (list, tuple)) and len(c) == 2:
            coeffs.append(complex(c[0], c[1]))
        else:
            coeffs.append(complex(c))
    return tuple(coeffs)


def _coerce_ms(ms_raw, rank):
    if not ms_raw:
        return tuple()
    if isinstance(ms_raw[0], (int, float)):
        if len(ms_raw) % rank != 0:
            raise ValueError(f"ms_combs length {len(ms_raw)} is not divisible by rank {rank}")
        return tuple(tuple(int(ms_raw[i * rank + j]) for j in range(rank)) for i in range(len(ms_raw) // rank))
    return tuple(tuple(int(x) for x in row) for row in ms_raw)


def build_descriptor_specs_from_settings(
    compact_labels,
    settings,
    library,
    center_mu_values = None,
    restrict_neighbor_mu = None,
    max_variants_per_label = None,
):
    labels = [normalize_compact_label(lab) for lab in compact_labels]
    specs_by_M = {int(M): [] for M in settings.M_R_values}
    for lab in labels:
        variants = build_channel_variants(lab, settings, center_mu_values, restrict_neighbor_mu, max_variants_per_label)
        rank = lab.rank
        for M_R in settings.M_R_values:
            rank_block = library.data[int(M_R)][rank]
            lookup_key = (
                lab.full_key()
                if lab.full_key() in rank_block
                else lab.angular_key()
            )
            payload = rank_block[lookup_key]
            ms_combinations = _coerce_ms(payload['ms_combs'], rank)
            phase = _scalar_real_phase(lab, L_R=settings.L_R, M_R=int(M_R))
            coeffs = tuple(phase * coeff for coeff in _coerce_coeffs(payload['coeffs']))
            for v_idx, variant in enumerate(variants):
                channels = []
                for leaf_idx, (n, l, mu, kappa0, kappa) in enumerate(zip(lab.n_tuple, lab.l_tuple, variant.mus, variant.kappa0s, variant.kappas)):
                    aux = None if variant.aux_lms is None else variant.aux_lms[leaf_idx]
                    channels.append(
                        SingleChannelLabel(
                            mu0=variant.mu0,
                            mu=mu,
                            kappa0=kappa0,
                            kappa=kappa,
                            n=n,
                            l=l,
                            m=0,
                            l_aux=None if aux is None else aux[0],
                            m_aux=None if aux is None else aux[1],
                        )
                    )
                specs_by_M[int(M_R)].append(
                    DescriptorSpec(
                        key=f"{lab.full_key()}|variant={v_idx}|{variant.suffix()}|M={M_R}",
                        label=lab,
                        channels=tuple(channels),
                        ms_combinations=ms_combinations,
                        coeffs=coeffs,
                        L_R=settings.L_R,
                        M_R=int(M_R),
                    )
                )
    return DescriptorCollection(compact_labels=tuple(labels), specs_by_M=specs_by_M)


def _scalar_coordinate_compiler_request(request):
    if request is None:
        return "missing_only", {
            "coordinate_contract": "pace_compatible_exact",
            "coefficient_materialization": "exact",
            "constructor_backend": "python",
        }, ()
    if not isinstance(request, dict):
        raise TypeError("scalar_coordinate_compiler must be a mapping.")
    unknown = set(request) - {"mode", "options", "coordinates"}
    if unknown:
        raise ValueError(
            "Unsupported scalar-coordinate compiler request keys: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    mode = str(request.get("mode", "all")).strip().lower()
    if mode not in {"all", "missing_only", "serialized"}:
        raise ValueError(
            "scalar-coordinate compiler mode must be all, missing_only, or serialized."
        )
    coordinates = tuple(request.get("coordinates", ()))
    if mode == "serialized":
        if request.get("options"):
            raise ValueError("serialized scalar coordinates do not accept global options.")
        if not coordinates or any(not isinstance(row, dict) for row in coordinates):
            raise ValueError("serialized scalar coordinates require ordered records.")
        return mode, {}, coordinates
    if coordinates:
        raise ValueError("coordinates are accepted only in serialized mode.")
    options = dict(request.get("options", {}))
    allowed_options = {
        "membership_mode",
        "coordinate_contract",
        "coefficient_materialization",
        "outer_coefficient_tolerance",
        "collection_tolerance",
        "maximum_unique_monomials",
        "maximum_term_contributions",
        "maximum_exact_symbolic_bytes",
        "maximum_coordinate_bytes",
        "constructor_backend",
    }
    unsupported = set(options) - allowed_options
    if unsupported:
        raise ValueError(
            "Unsupported scalar-coordinate compiler options: "
            + ", ".join(sorted(str(key) for key in unsupported))
        )
    options.setdefault("coordinate_contract", "pace_compatible_exact")
    options.setdefault("coefficient_materialization", "exact")
    options.setdefault("constructor_backend", "python")
    return mode, options, ()


def _compiled_scalar_coordinate_library(labels, settings, request):
    mode, options, serialized_coordinates = _scalar_coordinate_compiler_request(request)
    if mode == "serialized":
        if int(settings.L_R) != 0:
            raise ValueError("Serialized scalar coordinates require L_R=0.")
        if tuple(int(value) for value in settings.M_R_values) != (0,):
            raise ValueError("Serialized scalar coordinates require M_R_values=(0,).")
        if len(serialized_coordinates) != len(labels):
            raise ValueError("Serialized scalar-coordinate count does not match labels.")
        data = {0: {}}
        coordinate_records = []
        angular_alias_conflicts = []
        for label, binding in zip(labels, serialized_coordinates):
            coordinate = dict(binding.get("compiled_coordinate", {}))
            serialized_label = normalize_compact_label(coordinate.get("label", {}))
            if serialized_label != label:
                raise ValueError("Serialized scalar-coordinate order differs from labels.")
            magnetic_tuples = tuple(
                tuple(int(value) for value in row)
                for row in coordinate.get("magnetic_tuples", ())
            )
            coefficients = tuple(coordinate.get("coefficients", ()))
            if not magnetic_tuples or len(magnetic_tuples) != len(coefficients):
                raise ValueError("Serialized scalar coordinate has mismatched terms.")
            rank_block = data[0].setdefault(int(label.rank), {})
            angular_key = str(label.angular_key())
            full_key = str(label.full_key())
            coordinate_payload = {
                "rank": int(label.rank),
                "ms_combs": [list(row) for row in magnetic_tuples],
                "coeffs": [list(pair) for pair in coefficients],
            }
            existing_payload = rank_block.get(full_key)
            if existing_payload is not None and existing_payload != coordinate_payload:
                raise ValueError("Serialized scalar coordinates disagree for one label.")
            rank_block[full_key] = coordinate_payload
            existing_payload = rank_block.get(angular_key)
            if existing_payload is not None and existing_payload != coordinate_payload:
                angular_alias_conflicts.append(
                    {
                        "rank": int(label.rank),
                        "angular_key": angular_key,
                        "full_key": full_key,
                    }
                )
            rank_block.setdefault(angular_key, coordinate_payload)
            coordinate_records.append(
                {
                    "label": label.to_dict(),
                    "angular_key": angular_key,
                    "lookup_key": full_key,
                    "certificate": dict(coordinate.get("certificate", {})),
                    "compiler": dict(binding.get("compiler", {})),
                    "serialized_coordinate_sha256": str(
                        coordinate.get("payload_sha256", "")
                    ),
                }
            )
        return GeneralizedCouplingLibrary(
            data,
            L_R=settings.L_R,
            metadata={
                "scalar_coordinate_compiler": {
                    "schema": "ye3t_scalar_coordinate_library_v2",
                    "mode": mode,
                    "options": {},
                    "coordinates": tuple(coordinate_records),
                    "lookup_key_scope": "full_label_with_angular_alias",
                    "angular_alias_conflicts": tuple(angular_alias_conflicts),
                }
            },
        )

    if mode == "all":
        data = {int(M_R): {} for M_R in settings.M_R_values}
        for M_R in settings.M_R_values:
            for label in labels:
                rank_block = data[int(M_R)].setdefault(int(label.rank), {})
                rank_block.setdefault(
                    label.full_key(),
                    {"rank": int(label.rank), "ms_combs": [], "coeffs": []},
                )
    else:
        data = generate_library_for_labels(labels, M_R_values=settings.M_R_values)
    coordinate_records = []
    for M_R in settings.M_R_values:
        for label in labels:
            rank_block = data[int(M_R)][int(label.rank)]
            lookup_key = (
                label.full_key()
                if mode == "all"
                else label.angular_key()
            )
            payload = rank_block[lookup_key]
            rows = _coerce_ms(payload.get("ms_combs", ()), int(label.rank))
            coefficients = _coerce_coeffs(payload.get("coeffs", ()))
            if len(rows) != len(coefficients):
                raise ValueError(
                    "Generalized coupling payload has different m-row and "
                    f"coefficient counts for {label.full_key()!r}."
                )
            should_compile = mode == "all" or not rows
            if not should_compile:
                continue
            if int(settings.L_R) != 0 or int(M_R) != 0:
                raise ValueError(
                    "Scalar-coordinate compilation only supports L_R=0, M_R=0."
                )
            compilation = compile_scalar_ace_coordinate(label, **options)
            if compilation["label"] != label:
                raise RuntimeError("Scalar-coordinate compiler returned a different label.")
            certificate = dict(compilation["certificate"])
            if certificate.get("passed") is not True:
                raise RuntimeError("Scalar-coordinate compiler certificate did not pass.")
            table_rows, table_coefficients = compilation[
                "coefficient_table"
            ].component_terms(0)
            if len(table_rows) == 0 or len(table_rows) != len(table_coefficients):
                raise RuntimeError(
                    "Scalar-coordinate compiler returned empty or mismatched raw rows."
                )
            payload["ms_combs"] = table_rows.tolist()
            payload["coeffs"] = [
                [float(complex(value).real), float(complex(value).imag)]
                for value in table_coefficients.tolist()
            ]
            rank_block.setdefault(label.angular_key(), payload)
            coordinate_records.append(
                {
                    "label": label.to_dict(),
                    "angular_key": str(label.angular_key()),
                    "lookup_key": str(lookup_key),
                    "certificate": certificate,
                }
            )
    return GeneralizedCouplingLibrary(
        data,
        L_R=settings.L_R,
        metadata={
            "scalar_coordinate_compiler": {
                "schema": "ye3t_scalar_coordinate_library_v1",
                "mode": mode,
                "options": dict(options),
                "coordinates": tuple(coordinate_records),
                "lookup_key_scope": (
                    "full_label_with_angular_alias"
                    if mode == "all"
                    else "angular"
                ),
            }
        },
    )


def compile_descriptor_artifacts(
    settings,
    *,
    compact_labels = None,
    basis_mode = None,
    exact_primitive_timeout_seconds = None,
    center_mu_values = None,
    restrict_neighbor_mu = None,
    max_variants_per_label = 1,
    descriptor_cache = None,
    use_descriptor_cache = True,
    scalar_coordinate_compiler = None,
):
    """Compile labels, coupling payloads, and descriptor specs with reuse."""
    active_cache = resolve_descriptor_build_cache(descriptor_cache, use_cache=use_descriptor_cache)
    labels = (
        tuple(
            select_compact_labels(
                settings,
                compact_labels=compact_labels,
                basis_mode=basis_mode,
                exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
                descriptor_cache=active_cache,
                use_descriptor_cache=use_descriptor_cache,
            )
        )
    )
    key = descriptor_artifact_cache_key(
        settings,
        labels,
        basis_mode=basis_mode,
        exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
        center_mu_values=center_mu_values,
        restrict_neighbor_mu=restrict_neighbor_mu,
        max_variants_per_label=max_variants_per_label,
        scalar_coordinate_compiler=scalar_coordinate_compiler,
    )
    cached = None if active_cache is None else active_cache.get_artifacts(key)
    if cached is not None:
        return cached
    library = _compiled_scalar_coordinate_library(
        labels,
        settings,
        scalar_coordinate_compiler,
    )
    collection = build_descriptor_specs_from_settings(
        labels,
        settings,
        library,
        center_mu_values=center_mu_values,
        restrict_neighbor_mu=restrict_neighbor_mu,
        max_variants_per_label=max_variants_per_label,
    )
    value = (labels, library, collection)
    if active_cache is not None:
        active_cache.put_artifacts(key, value)
    return value
