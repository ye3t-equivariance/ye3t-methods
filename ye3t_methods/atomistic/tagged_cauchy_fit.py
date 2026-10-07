"""Streamed sufficient-statistics ridge fitter for tagged Cauchy models (WP4).

Mirrors the streaming pattern of the ordinary Ta linear workflow
(``ye3t_methods.atomistic.ace.linear_ace._build_structure_balanced_scaled_problem_streaming``
/ ``_solve_structure_balanced_scaled_ridge_from_normal``: two passes, a
symmetrized float64 Gram accumulated one structure at a time, an unpenalized
offset block, extreme-eigenvalue conditioning) but is written fresh here
because the feature source (:class:`ye3t_methods.atomistic.tagged_cauchy_linear.
TaggedMomentEvaluator`, composed over several per-content compiled
artifacts) and the offset-column contract (multiple unpenalized per-species
columns, or none in ``composition_fixed`` mode) are specific to tagged
Cauchy models.  No changes were made to ``tagged_cauchy_linear.py``; this
module only imports its public (and a few underscore-private, same-package)
symbols.

Model: ``E = sum_i sum_a beta_a F_i,a + sum_i offset_species(i)``, where
``F`` is ``combination_matrix @ (concatenated per-content descriptors)``
(identity by default).  Only ``beta`` and the fitted species offsets (if
any) are fitted; the compiled artifacts, radial tables, and
``combination_matrix`` are fixed buffers.
"""

import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import count as count_coupling
from ye3t.couplings import plan as plan_coupling
from ye3t.couplings.lifted_cauchy_scalar import CompiledLiftedCauchyScalar
from ye3t.couplings.tagged_cauchy import pooled_feature_matrix, pooled_tagged_basis, tag_support_report
from ye3t_methods.atomistic.lifted_cauchy_linear import _artifact_channels
from ye3t_methods.atomistic.tagged_cauchy_linear import (
    RealMomentEvaluator,
    TaggedCauchyModel,
    TaggedMomentEvaluator,
    _canonical_json_sha256,
    _complex_to_real_by_channel_list,
    _species_order_from_channels,
    export_tagged_model,
    ordinary_edge_primitives,
    ordinary_edge_primitives_with_derivative,
    position_jacobian_from_edge_derivative,
    real_moment_program,
)
from ye3t_methods.atomistic.utils.fit_weights import structure_fit_weights


def content_pooled_matrix(compiled, role_bindings):
    """The exact pooled-feature combination matrix for one content's artifact.

    ``pooled_tagged_basis`` drops unsupported descriptors internally (a
    descriptor's tag content must involve every edge role) and then finds
    the exact row-space basis surviving S_k tag-relabeling redundancy; at
    role_dimension 1 (no edge tags) every descriptor has a trivial
    stabilizer, so this is exactly the identity on the (fully supported)
    descriptor set. Returns (real_matrix [pooled x total_descriptors],
    supported_count, pooled_count).
    """

    total = len(compiled.payload["descriptors"])
    support = tag_support_report(compiled, role_bindings)
    supported_count = len(support["supported_indices"])
    basis = pooled_tagged_basis(compiled, role_bindings)
    matrix = pooled_feature_matrix(basis, total)
    if matrix.size:
        max_imag = float(np.abs(matrix.imag).max())
        scale = max(1.0, float(np.abs(matrix.real).max()))
        if max_imag > 1.0e-12 * scale:
            raise ValueError("pooled_feature_matrix has a material imaginary residual.")
    return matrix.real, supported_count, int(matrix.shape[0])


def _has_orthogonal_pooled_basis():
    """Report availability of a certified post-pooling image basis.

    The similarly named compiler helper orthogonalizes a diagnostic
    pre-moment frame and is deliberately not accepted here.
    """

    return False


def content_orthonormal_matrix(compiled, role_bindings):
    """Reject the retired pre-moment orthonormal-coordinate path."""

    raise NotImplementedError(
        "Certified post-pooling orthogonal image coordinates are not yet "
        "integrated into ye3t-ace. The old pre-moment metric is diagnostic "
        "only and cannot be used for fitting."
    )


def lower_orthonormal_beta_to_pooled(arm, beta_orth):
    """Lower a fitted orthonormal-coordinate beta back to pooled coordinates.

    ``theta_pooled = S^T theta_orth``, per content -- S is block-diagonal
    across contents by construction, so this decomposes into independent
    per-content slices; no full arm-level block-diagonal S is ever
    materialized. A content fit through an explicit
    ``combination_matrix_override`` (no orthonormal transform computed for
    it) passes its slice through unchanged. Only valid for an ``arm`` built
    with ``fit_coordinates="orthonormal_sector"``; the returned array is
    ready to export exactly like a plain pooled-coordinate beta (e.g. via
    :func:`export_tagged_fit_bundle`, or as
    :class:`ye3t_methods.atomistic.tagged_cauchy_linear.TaggedCauchyModel`'s ``beta`` for
    a single-content arm), with no further transform needed.

    Always operates on a RAW (``arm.raw_feature_count``-wide) vector, never
    on ``arm.feature_count`` directly -- for an arm with ``per_species_
    beta=False`` these are the same thing, so nothing changes there; for
    ``per_species_beta=True`` the caller must call this once per species,
    on that species' own ``raw_feature_count``-wide slice of the fitted
    (block-expanded) beta (see :func:`unblock_per_species_beta`, which
    produces exactly those slices), since the per-content orthonormal S
    only ever acted within one content's own raw feature columns -- it
    never saw the species-block expansion at all (that happens strictly
    after, uniformly, in ``descriptors()``; see :class:`TaggedArmEvaluator`'s
    docstring note on the two transforms' composition order).
    """

    if arm.fit_coordinates != "orthonormal_sector":
        raise ValueError(
            "lower_orthonormal_beta_to_pooled requires an arm built with "
            "fit_coordinates='orthonormal_sector'."
        )
    beta_orth = np.asarray(beta_orth, dtype=np.float64)
    if int(beta_orth.shape[0]) != arm.raw_feature_count:
        raise ValueError("beta_orth length must equal arm.raw_feature_count.")
    pieces = []
    for entry in arm.contents:
        start = entry["offset"]
        end = start + entry["descriptor_count"]
        slice_orth = beta_orth[start:end]
        S = entry.get("orthonormal_transform")
        pieces.append(slice_orth if S is None else S.T @ slice_orth)
    return np.concatenate(pieces) if pieces else np.zeros((0,), dtype=np.float64)


def _catalogue_cache_path(cache_dir, request_hash):
    return Path(cache_dir) / f"{request_hash}.json"


def _artifact_from_catalogue_content(content_record, cache_dir):
    """Recompile (or load-and-verify from ``cache_dir``) one catalogue
    content's artifact.

    Mirrors ``ye3t.couplings.tagged_catalogue``'s own on-disk builder-cache
    format (``<cache_dir>/<request_hash>.json``, a compiled artifact's
    ``to_dict()``) -- reimplemented locally rather than importing that
    module's private ``_load_cached_artifact``/``_store_cached_artifact``
    (this module's established convention: e.g. :func:`content_pooled_matrix`
    already reimplements its own imaginary-residual check rather than
    reaching into another package's private internals). Always verifies the
    result's ``self_hash`` against the catalogue's own recorded
    ``artifact_self_hash``, regardless of source (cache hit or fresh
    recompile through ``count``/``plan``/``compile`` with the catalogue's
    stored ``request_payload``), so a stale or mismatched cache entry can
    never silently produce the wrong artifact.
    """

    request_hash = content_record["request_hash"]
    expected_hash = content_record["artifact_self_hash"]
    compiled = None
    if cache_dir is not None:
        path = _catalogue_cache_path(cache_dir, request_hash)
        if path.exists():
            try:
                with path.open(encoding="utf-8") as handle:
                    body = json.load(handle)
                compiled = CompiledLiftedCauchyScalar.from_dict(body)
            except Exception:
                # Corrupted/truncated/schema-mismatched cache entry: fall
                # through to a fresh recompile (self_hash is still checked
                # below either way).
                compiled = None
    if compiled is None:
        request = content_record["request_payload"]
        compiled = compile_coupling(plan_coupling(count_coupling(request)))
    if str(compiled.self_hash) != str(expected_hash):
        raise ValueError(
            f"Catalogue content {content_record['content_index']}: recompiled/"
            f"cached artifact self_hash {compiled.self_hash!r} does not match "
            f"the catalogue's recorded artifact_self_hash {expected_hash!r}."
        )
    return compiled


def contents_from_catalogue(catalogue_record, cache_dir=None):
    """``TaggedArmEvaluator`` ``contents`` built from a ``tagged_catalogue`` record.

    Groups the catalogue's already-selected pooled ``features`` by their
    ``content_index``, recompiles (or loads-and-verifies from ``cache_dir``,
    default the catalogue's own recorded ``spec.cache_dir``) each such
    content's artifact via :func:`_artifact_from_catalogue_content`, and
    builds that content's exact combination matrix directly from the
    recorded ``combination`` entries (binary64 ``[descriptor_index, [re,
    im]]`` pairs; real part after checking the imaginary part is at most
    1e-12, the same convention as :func:`content_pooled_matrix`) --
    ``pooled_tagged_basis`` is never recomputed here, per the lead's
    instruction: the catalogue already carries the exact pooled features and
    their combinations. Contents with no selected features are skipped
    (nothing to fit against). Each returned content dict carries
    ``combination_matrix_override`` (so :class:`TaggedArmEvaluator` uses this
    matrix directly rather than recomputing one), ``supported_count``, and
    (for bookkeeping/reporting) ``catalogue_content_index`` and
    ``catalogue_strata`` (the distinct ``(N, kappa_class, lambda_class)``
    strata this content's selected features belong to).
    """

    if cache_dir is None:
        cache_dir = catalogue_record.get("spec", {}).get("cache_dir")
    cache_dir = None if cache_dir is None else Path(cache_dir)

    features_by_content = {}
    for feature in catalogue_record["features"]:
        features_by_content.setdefault(int(feature["content_index"]), []).append(feature)

    contents_by_index = {int(c["content_index"]): c for c in catalogue_record["contents"]}

    contents = []
    for content_index in sorted(features_by_content):
        content_features = sorted(features_by_content[content_index], key=lambda f: f["feature_index"])
        content_record = contents_by_index[content_index]
        compiled = _artifact_from_catalogue_content(content_record, cache_dir)
        total_descriptors = len(compiled.payload["descriptors"])
        matrix = np.zeros((len(content_features), total_descriptors), dtype=np.complex128)
        for row, feature in enumerate(content_features):
            for descriptor_index, coefficient in feature["combination"]:
                re, im = coefficient
                matrix[row, int(descriptor_index)] = complex(re, im)
        if matrix.size:
            max_imag = float(np.abs(matrix.imag).max())
            scale = max(1.0, float(np.abs(matrix.real).max()))
            if max_imag > 1.0e-12 * scale:
                raise ValueError(
                    f"Catalogue content {content_index}: combination matrix has "
                    "a material imaginary residual."
                )
        contents.append(
            {
                "content_id": f"catalogue_content_{content_index}",
                "compiled": compiled,
                "combination_matrix_override": matrix.real,
                "supported_count": int(content_record["supported_count"]),
                "catalogue_content_index": int(content_index),
                "catalogue_strata": sorted({tuple(f["stratum"]) for f in content_features}),
            }
        )
    return contents


def build_or_load_k0_catalogue(path, reference_spec, cache_dir=None):
    """Load a trivial-kappa ("K0" control) catalogue from ``path``, or build
    and save one there if absent.

    Built with the same builder (``ye3t.couplings.tagged_catalogue.
    build_tagged_catalogue``) and the same spec as ``reference_spec`` --
    typically another arm's own catalogue spec (same species/channels/
    rank_max/lambda_block_size_max/role_bindings/resource_limits, so "the
    same content scope at matched tag count") -- except ``kappa_policy`` is
    forced to ``"trivial"``. ``cache_dir`` overrides ``reference_spec``'s own
    ``cache_dir`` if given (else the reference's is reused, so a fresh K0
    build can still hit any already-compiled trivial-kappa contents from a
    prior run). Feature counts routinely differ from the reference arm's
    (matching by truncation is not performed here or anywhere in this
    module) -- report both counts, as they are.
    """

    from ye3t.couplings.tagged_catalogue import build_tagged_catalogue, load_catalogue, save_catalogue

    path = Path(path)
    if path.exists():
        return load_catalogue(str(path))
    spec = dict(reference_spec)
    spec["kappa_policy"] = "trivial"
    if cache_dir is not None:
        spec["cache_dir"] = str(cache_dir)
    record = build_tagged_catalogue(spec)
    save_catalogue(record, str(path))
    return record


TAGGED_FIT_BUNDLE_SCHEMA_V1 = "ye3t_tagged_cauchy_arm_fit_v1"
# v2 (WP4c) adds per_species_beta/central_species_order/fit_coordinates;
# load_tagged_fit_bundle accepts both.
TAGGED_FIT_BUNDLE_SCHEMA_V2 = "ye3t_tagged_cauchy_arm_fit_v2"
TAGGED_FIT_BUNDLE_SCHEMA = TAGGED_FIT_BUNDLE_SCHEMA_V2


# WP4d: one merged real moment program per arm, instead of evaluating each
# content through its own (complex-arithmetic) moment program every call.
#
# _arm_global_real_form_table asserts every content agrees on the real form
# of any global channel it touches (and, redundantly, that real_form_id is
# uniform within one angular momentum l across the whole arm -- it should
# always be, since real_form_id is a deterministic function of l alone, but
# a future channel-construction bug in either direction is asserted against
# explicitly rather than silently merging two different real bases).
#
# merge_real_moment_programs builds each content's own real_moment_program
# (in exactly the feature space matrix_selector picks -- by default the
# SAME (descriptor_selection, combination_matrix) pair that content's own
# TaggedMomentEvaluator already fits/evaluates through, so the merged
# program's features are bit-for-bit the arm's existing feature space),
# remaps its local channel indices to the arm's global ones, and interns
# the (now global-channel-indexed) density/moment keys into one shared pair
# of tables so identical keys arising in different contents collapse onto
# one row -- densities and moments are then computed once per center for
# the whole arm, not once per content. See TaggedArmEvaluator.descriptors
# for how the merged program is actually evaluated (one RealMomentEvaluator
# call over primitives already built at the arm's global channel
# granularity, replacing the per-content torch.cat loop).
MERGED_REAL_PROGRAM_SCHEMA_V1 = "ye3t_tagged_cauchy_merged_real_program_v1"


def _arm_global_real_form_table(contents, global_channels):
    """Global ``channel_index -> real_form_id`` table for one arm.

    Returns ``(channel_real_form_ids, real_form_records)``:
    ``channel_real_form_ids[global_channel_index]`` is that channel's
    ``real_form_id`` string (as consumed by
    ``ye3t_methods.atomistic.tagged_cauchy_linear._complex_to_real_by_channel_list``);
    ``real_form_records`` maps every such id to its exact record, taking the
    first content's copy and asserting byte-for-byte agreement
    (``real_to_complex_matrix``) against any later copies of the same id.

    A global channel not directly referenced by any content (possible if
    ``global_channels`` is a registry wider than this arm's own contents)
    is backfilled from another channel of the same ``l`` when one exists;
    it is an error only if no channel of that ``l`` is resolvable at all.
    """

    n_global = len(global_channels)
    channel_real_form_ids = [None] * n_global
    real_form_records = {}
    l_of_global = [int(channel["l"]) for channel in global_channels]

    for entry in contents:
        _fold_content_real_forms(entry["compiled"], entry["local_to_global"], channel_real_form_ids, real_form_records)

    return _finalize_real_form_table(channel_real_form_ids, real_form_records, l_of_global)


def _fold_content_real_forms(compiled, local_to_global, channel_real_form_ids, real_form_records):
    """Mutate ``channel_real_form_ids``/``real_form_records`` in place with
    one content's contribution (the per-content body of
    :func:`_arm_global_real_form_table`, factored out so the streamed
    builder -- :func:`build_streamed_merged_arm_from_catalogue` -- can fold
    one content at a time without holding every content's artifact at once).
    """

    local_channels = _artifact_channels(compiled)
    forms = {str(r["real_form_id"]): r for r in compiled.payload["real_forms"]}
    bindings = {
        int(r["channel_index"]): str(r["real_form_id"])
        for r in compiled.payload["channel_real_form_ids"]
    }
    for local_index, channel in enumerate(local_channels):
        global_index = int(local_to_global[local_index])
        real_form_id = bindings[int(channel["channel_index"])]
        existing = channel_real_form_ids[global_index]
        if existing is None:
            channel_real_form_ids[global_index] = real_form_id
        elif existing != real_form_id:
            raise ValueError(
                f"Inconsistent real_form_id for global channel {global_index}: "
                f"{existing!r} vs {real_form_id!r}."
            )
        record = forms[real_form_id]
        prior = real_form_records.get(real_form_id)
        if prior is None:
            real_form_records[real_form_id] = record
        elif prior["real_to_complex_matrix"] != record["real_to_complex_matrix"]:
            raise ValueError(
                f"real_form_id {real_form_id!r} has disagreeing "
                "real_to_complex_matrix payloads across contents."
            )


def _finalize_real_form_table(channel_real_form_ids, real_form_records, l_of_global):
    """Backfill any global channel no content directly referenced, from a
    same-``l`` sibling (the post-loop half of
    :func:`_arm_global_real_form_table`, factored out for reuse by the
    streamed builder); raises if none is resolvable.
    """

    by_l = {}
    for global_index, real_form_id in enumerate(channel_real_form_ids):
        if real_form_id is None:
            continue
        l = l_of_global[global_index]
        seen = by_l.setdefault(l, real_form_id)
        if seen != real_form_id:
            raise ValueError(
                f"Global channels of l={l} disagree on real_form_id "
                f"({seen!r} vs {real_form_id!r} at channel {global_index})."
            )

    for global_index, real_form_id in enumerate(channel_real_form_ids):
        if real_form_id is None:
            fallback = by_l.get(l_of_global[global_index])
            if fallback is None:
                raise ValueError(
                    f"Global channel {global_index} (l={l_of_global[global_index]}) is "
                    "not referenced by any content and no other channel of the same l "
                    "provides a real_form_id; cannot build the merged real-form table."
                )
            channel_real_form_ids[global_index] = fallback

    return channel_real_form_ids, real_form_records


def _default_matrix_selector(entry):
    return entry["evaluator"].descriptor_selection, entry["evaluator"].combination_matrix


def _content_export_matrix(entry):
    """The (descriptor_selection, combination_matrix) pair the EXPORTED,
    pooled-coordinate (never orthonormal) real program should use for one
    content.

    ``entry["pooled_combination_matrix"]`` when the content was built with
    its own separate pooled matrix (i.e. ``fit_coordinates=
    "orthonormal_sector"``, where ``entry["evaluator"].combination_matrix``
    is instead the orthonormal-transformed ``S @ pooled_matrix`` -- see
    ``content_orthonormal_matrix``), else exactly the content's own fitting
    matrix/selection unchanged (already pooled, an explicit catalogue
    override, or the plain unpooled selection -- in every one of those
    cases there is no separate pooled matrix to recover, because the
    content's fit space already IS its pooled/final space).
    """

    pooled = entry.get("pooled_combination_matrix")
    if pooled is not None:
        return None, pooled
    return _default_matrix_selector(entry)


def _fold_content_program(
    entry, role_bindings, matrix_selector,
    global_density_index, global_density_keys, global_moment_index, global_moment_keys,
    merged_terms, per_content_certificates,
):
    """Mutate the merge accumulators (``global_density_*``/``global_moment_*``/
    ``merged_terms``/``per_content_certificates``) with one content's own
    contribution: the per-content body of :func:`merge_real_moment_programs`,
    factored out so the streamed builder
    (:func:`build_streamed_merged_arm_from_catalogue`) can fold one content
    at a time, immediately after compiling it, without ever holding every
    content's artifact in a materialized ``contents`` list at once.

    ``entry`` needs only ``"compiled"``, ``"local_to_global"``,
    ``"offset"``, and ``"descriptor_count"`` (not the full
    :class:`TaggedArmEvaluator` content-entry shape). Returns
    ``(tag_count, feature_count)`` for that content's own program, for the
    caller's running ``tag_count`` agreement check and ``offset`` bookkeeping.
    """

    selection, combination_matrix = matrix_selector(entry)
    program = real_moment_program(
        entry["compiled"], role_bindings, selection, combination_matrix=combination_matrix
    )
    if int(program["feature_count"]) != int(entry["descriptor_count"]):
        raise ValueError(
            "_fold_content_program: per-content real program feature_count "
            f"({program['feature_count']}) does not match entry['descriptor_count'] "
            f"({entry['descriptor_count']})."
        )

    local_to_global = entry["local_to_global"]

    local_density_to_global = []
    for channel, a in program["real_density_keys"]:
        key = (int(local_to_global[channel]), int(a))
        idx = global_density_index.get(key)
        if idx is None:
            idx = len(global_density_keys)
            global_density_index[key] = idx
            global_density_keys.append(key)
        local_density_to_global.append(idx)

    local_moment_to_global = []
    for key in program["real_moment_keys"]:
        global_key = tuple(sorted((int(local_to_global[channel]), int(a)) for channel, a in key))
        idx = global_moment_index.get(global_key)
        if idx is None:
            idx = len(global_moment_keys)
            global_moment_index[global_key] = idx
            global_moment_keys.append(global_key)
        local_moment_to_global.append(idx)

    offset = int(entry["offset"])
    for term in program["terms"]:
        merged_terms.append(
            {
                "feature_index": int(term["feature_index"]) + offset,
                "coefficient": float(term["coefficient"]),
                "density_factor_indices": [
                    local_density_to_global[i] for i in term["density_factor_indices"]
                ],
                "p": int(term["p"]),
                "moment_indices": [local_moment_to_global[i] for i in term["moment_indices"]],
            }
        )
    per_content_certificates.append(
        {
            "content_id": entry.get("content_id"),
            "offset": offset,
            "lowering_certificate": dict(program["lowering_certificate"]),
        }
    )
    return int(program["tag_count"]), int(program["feature_count"])


def merge_real_moment_programs(contents, role_bindings, matrix_selector=None):
    """Build ONE merged real moment program for an arm from its contents.

    Each content's own program (:func:`ye3t_methods.atomistic.tagged_cauchy_linear.
    real_moment_program`, over that content's compiled artifact, using the
    ``(descriptor_selection, combination_matrix)`` pair ``matrix_selector``
    picks -- by default exactly what that content's own reference
    :class:`~ye3t_methods.atomistic.tagged_cauchy_linear.TaggedMomentEvaluator` already
    uses, so the merged program's features are bit-for-bit the same fit
    space as the arm's existing complex-arithmetic path) is remapped from
    that content's own LOCAL channel indices to the arm's GLOBAL channel
    indices (via ``entry["local_to_global"]``) and re-interned into one
    shared pair of global density/moment key tables, so identical
    ``(global_channel, a)`` keys arising in different contents collapse
    onto the same table row. Every content's ``terms`` are concatenated
    with ``feature_index`` shifted by that content's own ``entry["offset"]``
    (the same running offset ``TaggedArmEvaluator`` already uses to lay out
    ``torch.cat`` of the per-content complex path), so the merged program's
    feature axis is identical to an ``arm.raw_feature_count``-wide raw
    feature vector (species-block expansion for ``per_species_beta``, if
    any, happens strictly afterward and is untouched by this function).

    Returns a dict with the same keys as ``real_moment_program``'s own
    return value, plus ``per_content_certificates`` (that function's own
    ``lowering_certificate`` from each content, for audit).
    """

    if matrix_selector is None:
        matrix_selector = _default_matrix_selector

    global_density_index = {}
    global_density_keys = []
    global_moment_index = {}
    global_moment_keys = []
    merged_terms = []
    tag_count = None
    per_content_certificates = []
    total_feature_count = 0

    for entry in contents:
        content_tag_count, content_feature_count = _fold_content_program(
            entry, role_bindings, matrix_selector,
            global_density_index, global_density_keys, global_moment_index, global_moment_keys,
            merged_terms, per_content_certificates,
        )
        if tag_count is None:
            tag_count = content_tag_count
        elif tag_count != content_tag_count:
            raise ValueError(
                "merge_real_moment_programs: contents disagree on tag_count "
                f"({tag_count} vs {content_tag_count}) for the same arm-level "
                "role_bindings -- this should not be possible."
            )
        total_feature_count = max(total_feature_count, int(entry["offset"]) + content_feature_count)

    tag_count = 0 if tag_count is None else int(tag_count)
    return {
        "tag_count": tag_count,
        "feature_count": int(total_feature_count),
        "real_density_keys": [[int(c), int(a)] for c, a in global_density_keys],
        "real_moment_keys": [[[int(c), int(a)] for c, a in key] for key in global_moment_keys],
        "terms": merged_terms,
        "lowering_certificate": {
            "content_count": len(contents),
            "term_count_total": len(merged_terms),
            "real_density_key_count": len(global_density_keys),
            "real_moment_key_count": len(global_moment_keys),
            "feature_count": int(total_feature_count),
            "tag_count": tag_count,
            "imaginary_residual_is_exactly_zero": all(
                c["lowering_certificate"]["imaginary_residual_is_exactly_zero"]
                for c in per_content_certificates
            ),
        },
        "per_content_certificates": per_content_certificates,
    }


def merged_real_program_cache_path(catalogue_path, variant):
    catalogue_path = Path(catalogue_path)
    return catalogue_path.with_name(f"{catalogue_path.stem}_merged_real_program_{variant}.json")


def _load_or_build_merged_program(arm, *, variant, matrix_selector, catalogue_hash, cache_path):
    """Load a cached merged program for ``(catalogue_hash, variant)``, or
    build (and, when possible, cache) one.

    The cache is only trusted when the file's own recorded ``schema``,
    ``variant``, and ``catalogue_hash`` match exactly (a ``catalogue_hash``
    of ``None`` -- i.e. an arm not built from an on-disk catalogue -- never
    matches and never gets written, so ad hoc/test arms always rebuild), AND
    its recorded ``content_count``/``raw_feature_count`` match this arm's own
    -- belt-and-suspenders against a hash collision or a cache file reused
    across two genuinely different arms/catalogues by mistake; a mismatch
    here falls through to a fresh rebuild exactly like a missing file would.
    """

    cached = None
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                with cache_path.open(encoding="utf-8") as handle:
                    body = json.load(handle)
                if (
                    body.get("schema") == MERGED_REAL_PROGRAM_SCHEMA_V1
                    and body.get("variant") == variant
                    and catalogue_hash is not None
                    and body.get("catalogue_hash") == catalogue_hash
                    and int(body.get("content_count", -1)) == len(arm.contents)
                    and int(body.get("raw_feature_count", -1)) == int(arm.raw_feature_count)
                ):
                    cached = body
            except Exception:
                cached = None

    if cached is not None:
        return cached["program"], cached["channel_real_form_ids"], cached["real_form_records"]

    channel_real_form_ids, real_form_records = _arm_global_real_form_table(
        arm.contents, arm.global_channels
    )
    merged = merge_real_moment_programs(arm.contents, arm.role_bindings, matrix_selector=matrix_selector)
    program = {
        key: merged[key]
        for key in (
            "tag_count",
            "feature_count",
            "real_density_keys",
            "real_moment_keys",
            "terms",
            "lowering_certificate",
        )
    }
    if int(program["feature_count"]) != int(arm.raw_feature_count):
        raise ValueError(
            "_load_or_build_merged_program: merged program feature_count "
            f"({program['feature_count']}) does not match arm.raw_feature_count "
            f"({arm.raw_feature_count})."
        )

    _write_merged_program_cache(
        cache_path, variant, catalogue_hash, len(arm.contents), int(arm.raw_feature_count),
        program, channel_real_form_ids, real_form_records,
        per_content_certificates=merged.get("per_content_certificates"),
    )

    return program, channel_real_form_ids, real_form_records


def _write_merged_program_cache(
    cache_path, variant, catalogue_hash, content_count, raw_feature_count,
    program, channel_real_form_ids, real_form_records, per_content_certificates=None,
):
    """Write one merged-program cache file (the caching half of
    :func:`_load_or_build_merged_program`, factored out so the streamed
    builder can write both the fit-coordinate and pooled-export programs it
    builds in one pass, using the exact same on-disk schema/path convention
    (:func:`merged_real_program_cache_path`) so a later, separate-process
    call to :func:`build_merged_real_evaluator`/:func:`pooled_real_moment_
    program` hits the cache without recompiling anything. A no-op when
    ``cache_path`` or ``catalogue_hash`` is ``None`` (an arm not built from
    an on-disk catalogue never caches, exactly as before).
    """

    if cache_path is None or catalogue_hash is None:
        return
    cache_path = Path(cache_path)
    body = {
        "schema": MERGED_REAL_PROGRAM_SCHEMA_V1,
        "variant": variant,
        "catalogue_hash": catalogue_hash,
        "content_count": int(content_count),
        "raw_feature_count": int(raw_feature_count),
        "program": program,
        "channel_real_form_ids": channel_real_form_ids,
        "real_form_records": real_form_records,
        "per_content_certificates": per_content_certificates,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)


def build_merged_real_evaluator(arm, *, catalogue_hash=None, catalogue_path=None):
    """Build (or load from an on-disk cache) the merged real moment
    evaluator for one arm's FIT space (WP4d).

    Cache key is ``(catalogue_hash, variant)`` where ``variant`` is
    ``"orthonormal"`` when ``arm.fit_coordinates == "orthonormal_sector"``
    else ``"pooled"``; the file lives next to ``catalogue_path`` (see
    :func:`merged_real_program_cache_path`). Returns a
    :class:`~ye3t_methods.atomistic.tagged_cauchy_linear.RealMomentEvaluator` built via
    its keyword-only pre-built-``program`` constructor path.
    """

    variant = "orthonormal" if arm.fit_coordinates == "orthonormal_sector" else "pooled"
    cache_path = None if catalogue_path is None else merged_real_program_cache_path(catalogue_path, variant)
    program, channel_real_form_ids, real_form_records = _load_or_build_merged_program(
        arm, variant=variant, matrix_selector=None, catalogue_hash=catalogue_hash, cache_path=cache_path
    )
    return RealMomentEvaluator(
        program=program,
        tag_count=program["tag_count"],
        channel_real_form_ids=channel_real_form_ids,
        real_form_records=real_form_records,
    )


def pooled_real_moment_program(arm, *, catalogue_hash=None, catalogue_path=None):
    """The arm's merged real moment program in POOLED (never orthonormal)
    coordinates -- the space native/LAMMPS export uses, matching whatever
    beta :func:`lower_fitted_beta_to_pooled` produces.

    Identical to (and cached alongside) the arm's own FIT-space merged
    program when ``arm.fit_coordinates != "orthonormal_sector"`` (those two
    spaces coincide there); otherwise built with each content's own
    separately tracked ``pooled_combination_matrix`` in place of its
    orthonormal one (:func:`_content_export_matrix`), cached under its own
    ``"pooled_export"`` variant.

    Returns ``(program, channel_real_form_ids, real_form_records)``.
    """

    if arm.fit_coordinates == "orthonormal_sector":
        variant = "pooled_export"
        matrix_selector = _content_export_matrix
    else:
        variant = "pooled"
        matrix_selector = None
    cache_path = None if catalogue_path is None else merged_real_program_cache_path(catalogue_path, variant)
    return _load_or_build_merged_program(
        arm, variant=variant, matrix_selector=matrix_selector, catalogue_hash=catalogue_hash, cache_path=cache_path
    )


def build_streamed_merged_arm_from_catalogue(
    catalogue_record, *, cache_dir=None, fit_coordinates="pooled", per_species_beta=False, catalogue_path=None,
):
    """Bounded-memory construction of a catalogue-driven arm's merged real
    moment program(s), content by content (WPA streamed build).

    Unlike :func:`contents_from_catalogue` followed by
    ``TaggedArmEvaluator(..., use_merged_real_program=True)`` (which
    materializes every content's compiled artifact, in a Python list,
    before any of them are merged), this function compiles or loads ONE
    content's artifact at a time, immediately folds its contribution into
    the merged program(s) (:func:`_fold_content_program`) and the merged
    real-form table (:func:`_fold_content_real_forms`), then discards the
    artifact (``del compiled`` plus ``gc.collect()``) before moving to the
    next content -- so peak memory is bounded by one content's compiled
    artifact plus the merged-program accumulator, never by the sum over
    every content (the rank-4 k1 arm held 99 artifacts resident at once,
    peaking near 8.3 GB RSS, before this function existed).

    ``fit_coordinates="orthonormal_sector"`` is rejected. Older revisions
    accepted that spelling while applying no transform to catalogue-driven
    arms, so it falsely advertised orthonormal fitting. Until the exact
    post-pooling free-moment image is composed across all selected catalogue
    contents, the only production spelling is the honest raw spanning frame
    ``fit_coordinates="pooled"``. Only one merged program is built and cached next to
    ``catalogue_path`` via :func:`_write_merged_program_cache`, using the
    exact schema/path convention (:func:`merged_real_program_cache_path`)
    :func:`build_merged_real_evaluator`/:func:`pooled_real_moment_program`
    already read (both under their respective variant names, so either
    lookup hits the same cache entry), so a later, separate-process export
    step (``export_tagged_lammps_bundle.py``) hits the cache instead of
    recompiling anything.

    Returns a :class:`TaggedArmEvaluator` whose ``.contents`` entries carry
    no ``"compiled"``/``"evaluator"`` key -- only content summary fields
    (``content_id``, ``offset``, ``descriptor_count``, ``supported_count``,
    ``pooled_count``, ``total_descriptor_count``, ``catalogue_content_index``,
    ``catalogue_strata``, ``artifact_self_hash``) plus the small
    ``pooled_combination_matrix``/``orthonormal_transform`` numpy arrays
    (kept for provenance and so :func:`_arm_identity_hash`/
    :func:`_combination_hash` can still hash a real, arm-specific value).
    It is otherwise a normal, usable arm: ``.descriptors()``,
    ``.descriptors_and_jacobian()``, ``.feature_count``,
    ``.content_summary()`` all work exactly as for an eagerly-built arm,
    since ``descriptors()``/``descriptors_and_jacobian()`` only ever consult
    ``self._merged_real_evaluator`` when it is set (which it always is
    here) -- never ``entry["compiled"]``/``entry["evaluator"]``.

    ``export_tagged_fit_bundle`` records a ``catalogue_reference`` for each
    such content instead of an embedded compiled artifact (see its own
    docstring); ``fit_tagged_arm`` and every downstream fitting function are
    unaffected, since none of them read ``entry["compiled"]``/
    ``entry["evaluator"]`` either.
    """

    if fit_coordinates not in ("pooled", "orthonormal_sector"):
        raise ValueError("fit_coordinates must be 'pooled' or 'orthonormal_sector'.")
    if fit_coordinates == "orthonormal_sector":
        raise NotImplementedError(
            "Catalogue orthonormal coordinates require a certified exact "
            "post-pooling image transform. The historical mode was a no-op; "
            "use fit_coordinates='pooled' only for an explicitly raw frame."
        )
    role_bindings = tuple(tuple(binding) for binding in catalogue_record["spec"]["role_bindings"])
    tag_count = len(role_bindings) - 1
    # record["channels"] is already exactly the catalogue's own referenced
    # channel set, in dense channel_index order (see TaggedArmEvaluator.
    # __init__'s own docstring note on why the "restrict to referenced
    # channels" step is a no-op for a catalogue-built arm) -- used directly,
    # with no first pass over contents needed to compute it.
    global_channels = tuple(dict(c) for c in catalogue_record["channels"])
    global_index_of = {
        (str(c["neighbor_species"]), int(c["radial_channel"]), int(c["l"])): int(c["channel_index"])
        for c in global_channels
    }
    l_of_global = [int(c["l"]) for c in global_channels]
    if cache_dir is None:
        cache_dir = catalogue_record.get("spec", {}).get("cache_dir")
    cache_dir = None if cache_dir is None else Path(cache_dir)

    features_by_content = {}
    for feature in catalogue_record["features"]:
        features_by_content.setdefault(int(feature["content_index"]), []).append(feature)
    contents_by_index = {int(c["content_index"]): c for c in catalogue_record["contents"]}

    fit_real_form_ids = [None] * len(global_channels)
    fit_real_form_records = {}
    fit_density_index, fit_density_keys = {}, []
    fit_moment_index, fit_moment_keys = {}, []
    fit_terms = []
    fit_certificates = []

    content_summaries = []
    offset = 0
    catalogue_hash = catalogue_record.get("catalogue_hash")

    for content_index in sorted(features_by_content):
        content_features = sorted(features_by_content[content_index], key=lambda f: f["feature_index"])
        content_record = contents_by_index[content_index]
        compiled = _artifact_from_catalogue_content(content_record, cache_dir)
        try:
            total_descriptors = len(compiled.payload["descriptors"])
            pooled_from_catalogue = np.zeros((len(content_features), total_descriptors), dtype=np.complex128)
            for row, feature in enumerate(content_features):
                for descriptor_index, coefficient in feature["combination"]:
                    re, im = coefficient
                    pooled_from_catalogue[row, int(descriptor_index)] = complex(re, im)
            if pooled_from_catalogue.size:
                max_imag = float(np.abs(pooled_from_catalogue.imag).max())
                scale = max(1.0, float(np.abs(pooled_from_catalogue.real).max()))
                if max_imag > 1.0e-12 * scale:
                    raise ValueError(
                        f"Catalogue content {content_index}: combination matrix has "
                        "a material imaginary residual."
                    )
            pooled_from_catalogue = pooled_from_catalogue.real
            supported_count = int(content_record["supported_count"])

            artifact_channels = _artifact_channels(compiled)
            local_channels = tuple(
                dict(channel)
                for channel in content_record.get(
                    "channel_binding_override", artifact_channels
                )
            )
            if len(local_channels) != len(artifact_channels):
                raise ValueError(
                    f"Catalogue content {content_index}: channel binding override "
                    "must preserve the compiled channel count."
                )
            if any(
                int(bound["l"]) != int(artifact["l"])
                for bound, artifact in zip(
                    local_channels, artifact_channels, strict=True
                )
            ):
                raise ValueError(
                    f"Catalogue content {content_index}: channel binding override "
                    "must preserve every compiled angular momentum."
                )
            local_to_global = [
                global_index_of[(str(ch["neighbor_species"]), int(ch["radial_channel"]), int(ch["l"]))]
                for ch in local_channels
            ]

            # The catalogue's own recorded combination is always used as-is
            # for fitting, exactly like TaggedArmEvaluator.__init__'s
            # combination_matrix_override branch: no orthonormalization is
            # attempted here regardless of fit_coordinates (see this
            # function's own docstring for why that would NOT match the
            # eager path).
            fit_matrix = pooled_from_catalogue
            descriptor_count = int(fit_matrix.shape[0])
            entry_summary = {
                "content_id": f"catalogue_content_{content_index}",
                "offset": int(offset),
                "descriptor_count": descriptor_count,
                "supported_count": supported_count,
                "pooled_count": int(pooled_from_catalogue.shape[0]),
                "total_descriptor_count": int(total_descriptors),
                "compile_seconds": None,
                "catalogue_content_index": int(content_index),
                "catalogue_strata": sorted({tuple(f["stratum"]) for f in content_features}),
                "artifact_self_hash": str(compiled.self_hash),
                "pooled_combination_matrix": pooled_from_catalogue,
                "orthonormal_transform": None,
                "local_to_global": local_to_global,
            }
            content_summaries.append(entry_summary)

            _fold_content_real_forms(compiled, local_to_global, fit_real_form_ids, fit_real_form_records)
            fold_entry_fit = {
                "compiled": compiled, "local_to_global": local_to_global,
                "offset": offset, "descriptor_count": descriptor_count,
            }
            _fold_content_program(
                fold_entry_fit, role_bindings, lambda entry, m=fit_matrix: (None, m),
                fit_density_index, fit_density_keys, fit_moment_index, fit_moment_keys,
                fit_terms, fit_certificates,
            )
            offset += descriptor_count
        finally:
            del compiled
            gc.collect()

    raw_feature_count = int(offset)
    fit_real_form_ids, fit_real_form_records = _finalize_real_form_table(
        fit_real_form_ids, fit_real_form_records, l_of_global
    )
    fit_program = {
        "tag_count": tag_count,
        "feature_count": raw_feature_count,
        "real_density_keys": [[int(c), int(a)] for c, a in fit_density_keys],
        "real_moment_keys": [[[int(c), int(a)] for c, a in key] for key in fit_moment_keys],
        "terms": fit_terms,
        "lowering_certificate": {
            "content_count": len(content_summaries),
            "term_count_total": len(fit_terms),
            "real_density_key_count": len(fit_density_keys),
            "real_moment_key_count": len(fit_moment_keys),
            "feature_count": raw_feature_count,
            "tag_count": tag_count,
            "imaginary_residual_is_exactly_zero": True,
        },
    }
    # fit and pooled-export always coincide here (see the docstring): write
    # the ONE merged program under both variant names build_merged_real_
    # evaluator/pooled_real_moment_program look for, so either lookup is a
    # cache hit no matter which one a later process calls first.
    fit_variant = "orthonormal" if fit_coordinates == "orthonormal_sector" else "pooled"
    export_variant = "pooled_export" if fit_coordinates == "orthonormal_sector" else "pooled"
    for variant in {fit_variant, export_variant}:
        cache_path = None if catalogue_path is None else merged_real_program_cache_path(catalogue_path, variant)
        _write_merged_program_cache(
            cache_path, variant, catalogue_hash, len(content_summaries), raw_feature_count,
            fit_program, fit_real_form_ids, fit_real_form_records, per_content_certificates=fit_certificates,
        )

    central_species_order = _species_order_from_channels(global_channels) if per_species_beta else None
    arm = object.__new__(TaggedArmEvaluator)
    arm.role_bindings = role_bindings
    arm.global_channels = global_channels
    arm.use_pooled_basis = True
    arm.fit_coordinates = str(fit_coordinates)
    arm.per_species_beta = bool(per_species_beta)
    arm.central_species_order = central_species_order
    arm.contents = content_summaries
    arm.raw_feature_count = raw_feature_count
    arm.catalogue_hash = catalogue_hash
    arm.catalogue_path = None if catalogue_path is None else Path(catalogue_path)
    arm.use_merged_real_program = True
    arm._merged_real_evaluator = RealMomentEvaluator(
        program=fit_program, tag_count=tag_count,
        channel_real_form_ids=fit_real_form_ids, real_form_records=fit_real_form_records,
    )
    return arm


def global_channel_registry(radial_channel_count=2, angular_ls=(0, 1, 2), *, element):
    """A single-species channel space: n x l for one explicit ``element``.

    ``element`` has no default (WPA de-elementification): every caller must
    name the species explicitly instead of silently getting a fixed element.
    Kept for
    the single-species case (and for existing callers/tests that already
    pass ``element=`` explicitly); :func:`channel_registry` is the
    multi-species, per-tensor-order-union generalization config-driven
    workflows should use instead.
    """

    channels = []
    for n in range(int(radial_channel_count)):
        for l in angular_ls:
            channels.append(
                {
                    "channel_index": len(channels),
                    "channel_id": len(channels),
                    "neighbor_species": str(element),
                    "radial_channel": int(n),
                    "l": int(l),
                    "source_family_id": "primitive_polynomial_envelope_v1",
                }
            )
    return tuple(channels)


def channel_registry(
    species, radial_channel_count_by_order, angular_ls_by_order, source_family_id="primitive_polynomial_envelope_v1"
):
    """Multi-species channel registry, unioned over every requested tensor order.

    Replaces the fixed single-species Ta registry for config-driven
    workflows (WPA/T3): every element constant comes from the caller, not a
    library default. ``species`` is a sequence of symbols. ``radial_channel_
    count_by_order``/``angular_ls_by_order`` are ``{tensor_order: value}``
    maps (``basis.radial.n_max_per_tensor_order``/``basis.angular.
    leaf_lmax_per_tensor_order`` in ``ye3t_example_config_v1``, after
    ``int()``-keying); for one order, radial indices run ``0 ..
    radial_channel_count_by_order[order] - 1`` and angular momenta run
    ``0 .. angular_ls_by_order[order]`` inclusive (a single per-order l_max,
    matching the lifted example's own ``leaf_lmax_per_tensor_order``
    convention -- not a list of individual l values). Returns the UNION
    across every order of (species, radial_channel, l) triples, deduplicated,
    one entry per unique triple, ordered (species as given, then l, then
    radial_channel) -- the same convention ``ye3t.couplings.tagged_catalogue.
    _build_channel_list`` uses so a spec's own channel dict
    (:func:`channels_by_species_from_registry`) sorts identically.
    """

    species = tuple(str(value) for value in species)
    if not species:
        raise ValueError("channel_registry requires at least one species.")
    radial_by_order = {int(order): int(count) for order, count in dict(radial_channel_count_by_order).items()}
    l_by_order = {int(order): int(lmax) for order, lmax in dict(angular_ls_by_order).items()}
    if set(radial_by_order) != set(l_by_order):
        raise ValueError(
            "radial_channel_count_by_order and angular_ls_by_order must share the same tensor orders "
            f"({sorted(radial_by_order)!r} vs {sorted(l_by_order)!r})."
        )
    pairs = set()
    for order in radial_by_order:
        radial_count = radial_by_order[order]
        l_max = l_by_order[order]
        if radial_count <= 0:
            raise ValueError(f"radial_channel_count_by_order[{order}] must be positive.")
        if l_max < 0:
            raise ValueError(f"angular_ls_by_order[{order}] must be nonnegative.")
        for radial_index in range(radial_count):
            for l in range(l_max + 1):
                pairs.add((radial_index, l))
    channels = []
    for species_symbol in species:
        for radial_index, l in sorted(pairs, key=lambda pair: (pair[1], pair[0])):
            channels.append(
                {
                    "channel_index": len(channels),
                    "channel_id": len(channels),
                    "neighbor_species": species_symbol,
                    "radial_channel": int(radial_index),
                    "l": int(l),
                    "source_family_id": str(source_family_id),
                }
            )
    return tuple(channels)


def channels_by_species_from_registry(registry):
    """``{species: [(radial_channel, l), ...]}`` from a flat channel registry.

    The shape ``ye3t.couplings.tagged_catalogue``'s catalogue ``spec.
    channels`` field expects; the inverse reshape of :func:`channel_registry`
    into a per-species pair list (order within a species is whatever the
    registry already has -- ``_build_channel_list`` re-sorts it anyway).
    """

    by_species = {}
    for channel in registry:
        symbol = str(channel["neighbor_species"])
        by_species.setdefault(symbol, []).append((int(channel["radial_channel"]), int(channel["l"])))
    return by_species


def _content_channel_bindings(content):
    compiled_channels = _artifact_channels(content["compiled"])
    override = content.get("channel_binding_override")
    if override is None:
        return compiled_channels
    override = tuple(dict(channel) for channel in override)
    if len(override) != len(compiled_channels):
        raise ValueError(
            "channel_binding_override must contain one physical channel for "
            "every local compiler channel."
        )
    normalized = []
    keys = set()
    for local_index, (compiled, physical) in enumerate(
        zip(compiled_channels, override, strict=True)
    ):
        if int(physical["l"]) != int(compiled["l"]):
            raise ValueError(
                "channel_binding_override may change radial/species identity "
                "but must preserve each local channel's angular momentum."
            )
        channel = {
            **physical,
            "channel_index": int(local_index),
            "channel_id": int(local_index),
            "neighbor_species": str(physical["neighbor_species"]),
            "radial_channel": int(physical["radial_channel"]),
            "l": int(physical["l"]),
            "source_family_id": str(physical["source_family_id"]),
        }
        key = (
            channel["neighbor_species"],
            channel["radial_channel"],
            channel["l"],
            channel["source_family_id"],
        )
        if key in keys:
            raise ValueError("channel_binding_override contains duplicate channels.")
        keys.add(key)
        normalized.append(channel)
    return tuple(normalized)


class TaggedArmEvaluator:
    """One arm (fixed role_bindings) composed of several per-content artifacts.

    Each content keeps its own compiled artifact and a
    :class:`TaggedMomentEvaluator`; by default (``use_pooled_basis=True``)
    every content's descriptors are reduced through its own exact
    ``pooled_tagged_basis``/``pooled_feature_matrix`` (real part, after
    checking the imaginary part is at most 1e-12) - the exact pooled
    feature basis under S_k tag relabeling, which already drops unsupported
    descriptors internally and is the identity when there are no edge tags
    (role_dimension 1). Passing ``use_pooled_basis=False`` falls back to
    the plain supported-descriptor selection (no pooling) as a diagnostic
    alternative. Features are the concatenation across contents in the
    order given ("the descriptor offset table"). Edge primitives are
    computed once per structure over the shared global channel space and
    gathered (not recomputed) into each content's own channel order.
    """

    def __init__(
        self,
        role_bindings,
        contents,
        global_channels,
        use_pooled_basis=True,
        fit_coordinates="pooled",
        per_species_beta=False,
        use_merged_real_program=False,
        catalogue_hash=None,
        catalogue_path=None,
    ):
        self.role_bindings = tuple((str(k), int(v)) for k, v in role_bindings)
        contents = list(contents)
        full_global_channels = tuple(global_channels)
        # Bug fix (WP4f item 6): a manually-listed content scope (build_arm
        # in train_export.py, used by config_quick.json's demo catalogue)
        # need not touch every channel of the passed-in registry -- e.g. the
        # 4-content demo subset never references either l=2 channel.
        # _arm_global_real_form_table (called below via
        # build_merged_real_evaluator when use_merged_real_program=True)
        # requires every entry of arm.global_channels to be either directly
        # referenced by a content or backfillable from a same-l sibling that
        # is; a channel whose l is not referenced by ANY content has
        # neither, and previously raised unconditionally ("Global channel 2
        # (l=2) is not referenced by any content..."). Restrict
        # global_channels (and therefore local_to_global below, the
        # primitives ordinary_edge_primitives builds in descriptors()/
        # descriptors_and_jacobian(), the merged real-form table, and
        # arm_lammps_model's exported channel list -- arm_lammps_model
        # passes channels=list(arm.global_channels) directly, so this is
        # also "the export's channel list") to exactly the channels some
        # content references, renumbered to dense positions in the original
        # registry order. This is a no-op (identical channel set and order,
        # hence identical dense positions) whenever the passed-in registry
        # already equals the referenced set -- always true for a
        # catalogue-built arm (build_arm_from_catalogue /
        # build_k0_control_arm pass record["channels"], already exactly the
        # catalogue's own referenced channels) and for the full manual
        # content scope (generate_full_content_catalogue references every
        # channel by construction: l=2 is included at every block size, l=0
        # at every block size, l=1 at even block sizes) -- catalogue-mode
        # and full-scope behaviour are therefore unchanged; only the demo
        # scope's construction failure is fixed.
        content_channel_lists = [_content_channel_bindings(content) for content in contents]
        referenced_channel_keys = {
            (str(ch["neighbor_species"]), int(ch["radial_channel"]), int(ch["l"]))
            for channels in content_channel_lists
            for ch in channels
        }
        self.global_channels = tuple(
            {**dict(c), "channel_index": i, "channel_id": i}
            for i, c in enumerate(
                c
                for c in full_global_channels
                if (str(c["neighbor_species"]), int(c["radial_channel"]), int(c["l"]))
                in referenced_channel_keys
            )
        )
        self.use_pooled_basis = bool(use_pooled_basis)
        self.fit_coordinates = str(fit_coordinates)
        # WP4c: E = sum_i [offset(s_i) + sum_a beta_{s_i,a} F_i,a] -- the
        # energy coefficients depend on the CENTRAL species of each atom.
        # Implemented as a block-diagonal-by-central-species expansion of
        # the raw (species-independent) descriptors: descriptors() returns
        # [n_atoms, len(central_species_order) * raw_feature_count] with
        # each atom's raw_feature_count values placed in ITS OWN species'
        # column block (zero elsewhere), rather than changing any evaluator.
        # A single shared beta of that expanded width is then mathematically
        # identical to a per-species beta -- so every downstream streamed-
        # fitter function (structure_row's feature_sum/jacobian being the
        # expanded evaluator's sum/Jacobian already, compute_feature_scale,
        # accumulate_normal_equations, solve_ridge_grid, evaluate_rmse,
        # fit_tagged_arm) needs no changes at all; only feature_count and
        # descriptors() here are aware of this. Central species use the same
        # vocabulary/indexing as the (per-neighbor-species) global_channels'
        # own species order, since channels already carry neighbor_species
        # and atom_types is already indexed the same way by every caller
        # (structure_row, TaggedCauchyModel/energy_and_forces) -- see
        # unblock_per_species_beta for reshaping a fitted block-expanded
        # beta back into {species: array} for export.
        self.per_species_beta = bool(per_species_beta)
        self.central_species_order = (
            _species_order_from_channels(self.global_channels) if self.per_species_beta else None
        )
        if self.fit_coordinates not in ("pooled", "orthonormal_sector"):
            raise ValueError("fit_coordinates must be 'pooled' or 'orthonormal_sector'.")
        if self.fit_coordinates == "orthonormal_sector":
            raise NotImplementedError(
                "The former orthonormal_sector path used a pre-pooling metric. "
                "Use pooled raw coordinates until a certified exact post-pooling "
                "image transform is supplied."
            )
        if self.fit_coordinates == "orthonormal_sector" and not self.use_pooled_basis:
            raise ValueError("fit_coordinates='orthonormal_sector' requires use_pooled_basis=True.")
        global_index_of = {
            (str(c["neighbor_species"]), int(c["radial_channel"]), int(c["l"])): i
            for i, c in enumerate(self.global_channels)
        }
        self.contents = []
        offset = 0
        for content, content_channels in zip(contents, content_channel_lists):
            compiled = content["compiled"]
            pooled_matrix = None
            orthonormal_transform = None
            if self.use_pooled_basis:
                override = content.get("combination_matrix_override")
                if override is not None:
                    fit_matrix = np.asarray(override, dtype=np.float64)
                    supported_count = int(content.get("supported_count", fit_matrix.shape[0]))
                    # An explicit override (e.g. from contents_from_catalogue)
                    # is used as-is for fitting; it is not further composed
                    # with an orthonormal transform (there is no separate
                    # "pooled" matrix to recover it from here), so its slice
                    # of beta passes through lower_orthonormal_beta_to_pooled
                    # unchanged (see that function's docstring).
                elif self.fit_coordinates == "orthonormal_sector":
                    (
                        fit_matrix,
                        pooled_matrix,
                        orthonormal_transform,
                        supported_count,
                        _pooled_count,
                    ) = content_orthonormal_matrix(compiled, self.role_bindings)
                else:
                    pooled_matrix, supported_count, _pooled_count = content_pooled_matrix(
                        compiled, self.role_bindings
                    )
                    fit_matrix = pooled_matrix
                evaluator = TaggedMomentEvaluator(
                    compiled, self.role_bindings, None, combination_matrix=fit_matrix
                )
                pooled_count = evaluator.descriptor_count
            else:
                selection = content.get("supported_indices")
                evaluator = TaggedMomentEvaluator(compiled, self.role_bindings, selection)
                supported_count = evaluator.descriptor_count
                pooled_count = evaluator.descriptor_count
            local_to_global = torch.tensor(
                [
                    global_index_of[
                        (str(ch["neighbor_species"]), int(ch["radial_channel"]), int(ch["l"]))
                    ]
                    for ch in content_channels
                ],
                dtype=torch.long,
            )
            count = evaluator.descriptor_count
            entry = dict(content)
            entry["evaluator"] = evaluator
            entry["local_to_global"] = local_to_global
            entry["descriptor_count"] = int(count)
            entry["supported_count"] = int(supported_count)
            entry["pooled_count"] = int(pooled_count)
            entry["offset"] = int(offset)
            entry["pooled_combination_matrix"] = pooled_matrix
            entry["orthonormal_transform"] = orthonormal_transform
            self.contents.append(entry)
            offset += count
        self.raw_feature_count = int(offset)

        # WP4d: build ONE merged real moment program for the whole arm at
        # construction time (rather than evaluating each content through
        # its own per-content complex-arithmetic TaggedMomentEvaluator on
        # every descriptors() call) when requested. The per-content
        # evaluators above are still built either way -- they remain the
        # reference path (used when use_merged_real_program=False, and by
        # certificates/tests either way); only descriptors() below is
        # affected by which path actually runs.
        self.catalogue_hash = catalogue_hash
        self.catalogue_path = None if catalogue_path is None else Path(catalogue_path)
        self.use_merged_real_program = bool(use_merged_real_program)
        self._merged_real_evaluator = None
        if self.use_merged_real_program and self.contents:
            self._merged_real_evaluator = build_merged_real_evaluator(
                self, catalogue_hash=self.catalogue_hash, catalogue_path=self.catalogue_path
            )

    @property
    def feature_count(self):
        if self.per_species_beta:
            return self.raw_feature_count * len(self.central_species_order)
        return self.raw_feature_count

    def descriptors(self, positions, atom_types, cell, pbc, cutoff, radial_config):
        n_atoms = int(positions.shape[0])
        if not self.contents:
            return positions.new_zeros((n_atoms, self.feature_count)) + positions.sum() * 0.0
        edge_index, disp, phi_full = ordinary_edge_primitives(
            positions, atom_types, cell, pbc, cutoff, radial_config, self.global_channels
        )
        if self._merged_real_evaluator is not None:
            # One bucketed real-moment kernel pass over the WHOLE arm's
            # primitives (already built at the global channel granularity
            # above), replacing the per-content torch.cat loop below.
            raw = self._merged_real_evaluator.descriptors(
                positions, atom_types, cell, pbc, (edge_index, disp, phi_full)
            )
        else:
            pieces = []
            for entry in self.contents:
                phi_content = phi_full.index_select(1, entry["local_to_global"])
                primitives_content = (edge_index, disp, phi_content)
                pieces.append(
                    entry["evaluator"].descriptors(positions, atom_types, cell, pbc, primitives_content)
                )
            raw = torch.cat(pieces, dim=1)
        if not self.per_species_beta:
            return raw
        n_species = len(self.central_species_order)
        raw_width = int(raw.shape[1])
        expanded = raw.new_zeros((n_atoms, n_species * raw_width))
        column = atom_types.unsqueeze(1) * raw_width + torch.arange(raw_width, device=raw.device).unsqueeze(0)
        expanded.scatter_(1, column, raw)
        return expanded

    def descriptors_and_jacobian(
        self,
        positions,
        atom_types,
        cell,
        pbc,
        cutoff,
        radial_config,
        term_chunk_size=None,
    ):
        """Explicit (autograd-free) feature-sum and position Jacobian for
        the WHOLE arm (WP4e).

        Requires ``use_merged_real_program=True`` (the explicit Jacobian is
        built entirely from ``RealMomentEvaluator``'s bucketed program
        structure via :meth:`~ye3t_methods.atomistic.tagged_cauchy_linear.
        RealMomentEvaluator.evaluate_real_with_jacobian`, which only the
        merged evaluator exposes at the whole-arm level -- the per-content
        complex-arithmetic fallback path has no equivalent and is not
        supported here; use ``jacobian_mode="autograd"`` instead for an
        arm built with ``use_merged_real_program=False``). Returns
        ``(feature_sum, jacobian)`` with the SAME shapes
        :func:`_feature_sum_and_jacobian_autograd` returns: ``feature_sum``
        is ``descriptors(...).sum(dim=0)``, ``jacobian`` is
        ``[feature_count, n_atoms, 3]``.

        ``per_species_beta`` expansion happens at the PER-EDGE level for
        the Jacobian (mirroring ``descriptors()``'s own PER-ATOM scatter
        into ``[n_atoms, n_species * raw_feature_count]``): each edge's
        raw per-term derivative belongs in ITS OWN center atom's species
        block (``atom_types[src[e]]``, gathered per edge), exactly as each
        atom's raw feature row belongs in that atom's own species block in
        the value case -- the value case can sum over atoms AFTER
        expanding (order does not matter for a linear scatter-then-sum),
        but the Jacobian's own center-atom association is only available
        per EDGE (every edge has exactly one center), so the species
        expansion must happen before :func:`position_jacobian_from_edge_
        derivative`'s cross-edge accumulation, not after.
        """

        n_atoms = int(positions.shape[0])
        n_features = self.feature_count
        if self._merged_real_evaluator is None:
            raise ValueError(
                "descriptors_and_jacobian (jacobian_mode='explicit') requires an arm "
                "built with use_merged_real_program=True; use jacobian_mode='autograd' "
                "for a per-content arm."
            )
        if not self.contents:
            zero_link = positions.sum() * 0.0
            return (
                positions.new_zeros((n_features,)) + zero_link,
                positions.new_zeros((n_features, n_atoms, 3)),
            )
        edge_index, disp, phi_full, dphi_full = ordinary_edge_primitives_with_derivative(
            positions, atom_types, cell, pbc, cutoff, radial_config, self.global_channels
        )
        evaluator = self._merged_real_evaluator
        src = edge_index[0]
        n_channels = int(phi_full.shape[1])
        max_width = int(phi_full.shape[2])

        def _to_real(value):
            return _complex_to_real_by_channel_list(
                value, evaluator._channel_real_form_ids, evaluator._real_form_records
            )

        def _to_real_with_cartesian_axis(value):
            return _to_real(value.movedim(-1, 0)).movedim(0, -1)

        phi_real = _to_real(phi_full)
        dphi_real = _to_real_with_cartesian_axis(dphi_full)
        A_complex = phi_full.new_zeros((n_atoms, n_channels, max_width))
        if int(src.numel()):
            A_complex.index_add_(0, src, phi_full)
        A_real = _to_real(A_complex)

        raw, dF_e = evaluator.evaluate_real_with_jacobian(
            phi_real,
            dphi_real,
            A_real,
            edge_index,
            n_atoms,
            term_chunk_size=term_chunk_size,
        )

        if not self.per_species_beta:
            feature_sum = raw.sum(dim=0)
            jacobian = position_jacobian_from_edge_derivative(dF_e, edge_index, n_atoms, n_features)
            return feature_sum, jacobian

        n_species = len(self.central_species_order)
        raw_width = int(raw.shape[1])
        expanded = raw.new_zeros((n_atoms, n_species * raw_width))
        column = atom_types.unsqueeze(1) * raw_width + torch.arange(raw_width, device=raw.device).unsqueeze(0)
        expanded.scatter_(1, column, raw)
        feature_sum = expanded.sum(dim=0)

        edges = int(dF_e.shape[0])
        if edges:
            src_species = atom_types.index_select(0, src)
            edge_column = (
                src_species.unsqueeze(1) * raw_width + torch.arange(raw_width, device=raw.device).unsqueeze(0)
            )
            dF_e_expanded = dF_e.new_zeros((edges, n_species * raw_width, 3))
            dF_e_expanded.scatter_(1, edge_column.unsqueeze(-1).expand(-1, -1, 3), dF_e)
        else:
            dF_e_expanded = dF_e.new_zeros((0, n_species * raw_width, 3))
        jacobian = position_jacobian_from_edge_derivative(dF_e_expanded, edge_index, n_atoms, n_species * raw_width)
        return feature_sum, jacobian

    def content_summary(self):
        return tuple(
            {
                "content_id": entry.get("content_id"),
                "offset": entry["offset"],
                "descriptor_count": entry["descriptor_count"],
                "total_descriptor_count": (
                    entry["total_descriptor_count"]
                    if "total_descriptor_count" in entry
                    else len(entry["compiled"].payload["descriptors"])
                ),
                "supported_count": entry.get("supported_count"),
                "pooled_count": entry.get("pooled_count"),
                "compile_seconds": entry.get("compile_seconds"),
            }
            for entry in self.contents
        )


def unblock_per_species_beta(arm, beta_block):
    """Reshape a fitted block-expanded beta back into ``{species: array}``.

    ``beta_block`` (length ``arm.feature_count``, i.e.
    ``len(central_species_order) * raw_feature_count``) is exactly what
    :func:`fit_tagged_arm` returns for an ``arm`` built with
    ``per_species_beta=True`` (its ``beta_physical`` is generic over
    whatever feature space ``arm.descriptors`` produces -- see
    :class:`TaggedArmEvaluator`'s own docstring note). Species ``s``'s block
    is ``beta_block[s*raw : (s+1)*raw]``, per the same column convention
    ``descriptors()`` scatters into.
    """

    if not arm.per_species_beta:
        raise ValueError("unblock_per_species_beta requires an arm built with per_species_beta=True.")
    beta_block = np.asarray(beta_block, dtype=np.float64)
    raw = int(arm.raw_feature_count)
    expected = raw * len(arm.central_species_order)
    if int(beta_block.shape[0]) != expected:
        raise ValueError(f"beta_block length {beta_block.shape[0]} != feature_count ({expected}).")
    return {
        species: beta_block[index * raw : (index + 1) * raw].copy()
        for index, species in enumerate(arm.central_species_order)
    }


def lower_fitted_beta_to_pooled(arm, beta_fitted):
    """Lower a fitted beta (whatever ``arm.descriptors()`` produced it
    against) back to plain pooled coordinates, uniformly over all four
    ``(fit_coordinates, per_species_beta)`` combinations.

    ``fit_coordinates="pooled"``: no-op (already pooled), regardless of
    ``per_species_beta``. ``fit_coordinates="orthonormal_sector"`` with
    ``per_species_beta=False``: :func:`lower_orthonormal_beta_to_pooled`
    directly. ``per_species_beta=True`` (either ``fit_coordinates``):
    unblocks by species (:func:`unblock_per_species_beta`), lowers each
    species' own ``raw_feature_count``-wide slice independently when
    orthonormal (the per-content S only ever acted within one content's raw
    columns, before the species-block expansion -- see
    :func:`lower_orthonormal_beta_to_pooled`'s docstring), then
    re-concatenates in ``central_species_order`` -- exactly the column
    convention ``descriptors()`` itself uses, so the result is ready to
    export as a plain (pooled-coordinate, possibly per-species) beta with
    no further transform needed either way.
    """

    if arm.per_species_beta:
        per_species = unblock_per_species_beta(arm, beta_fitted)
        pieces = [
            (
                lower_orthonormal_beta_to_pooled(arm, per_species[species])
                if arm.fit_coordinates == "orthonormal_sector"
                else per_species[species]
            )
            for species in arm.central_species_order
        ]
        return np.concatenate(pieces) if pieces else np.zeros((0,), dtype=np.float64)
    if arm.fit_coordinates == "orthonormal_sector":
        return lower_orthonormal_beta_to_pooled(arm, beta_fitted)
    return np.asarray(beta_fitted, dtype=np.float64)


def arm_lammps_model(
    arm, beta_pooled, offsets, radial_config, cutoff, species_order, *, catalogue_hash=None, catalogue_path=None
):
    """A multi-content :class:`~ye3t_methods.atomistic.tagged_cauchy_linear.TaggedCauchyModel`
    ready for :func:`~ye3t_methods.atomistic.tagged_cauchy_linear.export_tagged_model` (WP4d).

    ``beta_pooled`` is the arm's fitted beta already lowered to pooled
    coordinates by :func:`lower_fitted_beta_to_pooled` (the exact array the
    caller already has right before calling :func:`export_tagged_fit_bundle`
    -- both exports share this one lowering step). The model's real moment
    program is :func:`pooled_real_moment_program` (the merged, POOLED-space
    -- never orthonormal -- program, cached next to ``catalogue_path`` when
    given), so the exported ``beta`` and native feature space always agree:
    exactly WP3e's invariant, extended to a merged multi-content arm.
    """

    if catalogue_hash is None:
        catalogue_hash = getattr(arm, "catalogue_hash", None)
    if catalogue_path is None:
        catalogue_path = getattr(arm, "catalogue_path", None)
    program, channel_real_form_ids, real_form_records = pooled_real_moment_program(
        arm, catalogue_hash=catalogue_hash, catalogue_path=catalogue_path
    )
    if arm.per_species_beta:
        by_species = unblock_per_species_beta(arm, beta_pooled)
        beta = {species: by_species[species].tolist() for species in arm.central_species_order}
    else:
        beta = np.asarray(beta_pooled, dtype=np.float64).tolist()
    return TaggedCauchyModel(
        None,
        arm.role_bindings,
        None,
        beta,
        offsets,
        radial_config,
        cutoff,
        species_order,
        channels=list(arm.global_channels),
        real_moment_program=program,
        channel_real_form_ids=channel_real_form_ids,
        real_form_records=real_form_records,
    )


def _feature_sum_and_jacobian_autograd(arm, positions, atom_types, cell, pbc, cutoff, radial_config, chunk_size=32):
    """Feature-sum vector and its position Jacobian via per-feature-chunk
    autograd (the pre-WP4e path; retained as the ``jacobian_mode="autograd"``
    oracle/fallback).

    Single forward pass (graph retained); the backward pass is split into
    ``chunk_size``-wide batched-grad calls (``is_grads_batched=True``, the
    same vectorized-VJP idiom already used by
    ``LiftedCauchyTorchEvaluator.vjp(method="autograd")``) so peak memory is
    bounded by one chunk of cotangents rather than the full feature count.
    """

    n_atoms = int(positions.shape[0])
    feature_sum = arm.descriptors(positions, atom_types, cell, pbc, cutoff, radial_config).sum(dim=0)
    n_features = int(feature_sum.shape[0])
    if n_features == 0:
        return feature_sum.detach(), positions.new_zeros((0, n_atoms, 3))
    chunk_size = max(1, int(chunk_size))
    identity = torch.eye(n_features, dtype=feature_sum.dtype)
    rows = []
    starts = list(range(0, n_features, chunk_size))
    for index, start in enumerate(starts):
        end = min(start + chunk_size, n_features)
        grad_outputs = identity[start:end]
        is_last = index == len(starts) - 1
        (chunk_jacobian,) = torch.autograd.grad(
            feature_sum,
            positions,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=not is_last,
        )
        rows.append(chunk_jacobian.detach())
    jacobian = torch.cat(rows, dim=0)
    return feature_sum.detach(), jacobian


def _feature_sum_and_jacobian_explicit(arm, positions, atom_types, cell, pbc, cutoff, radial_config):
    """Feature-sum vector and its position Jacobian via the explicit,
    autograd-free analytic path (WP4e default).

    Requires ``arm`` to have been built with ``use_merged_real_program=
    True`` (:meth:`TaggedArmEvaluator.descriptors_and_jacobian`'s own
    requirement -- the explicit Jacobian is built entirely from
    ``RealMomentEvaluator``'s bucketed program structure, which only the
    merged evaluator exposes at the whole-arm level). No chunking
    parameter: the explicit path's peak memory is already bounded by
    ``O(terms)`` per structure (the same bucketed working set
    :meth:`~ye3t_methods.atomistic.tagged_cauchy_linear.RealMomentEvaluator.evaluate_real`
    itself uses for the forward pass), not by the feature count, so there
    is no analogous chunk-size knob to tune.
    """

    with torch.no_grad():
        feature_sum, jacobian = arm.descriptors_and_jacobian(
            positions, atom_types, cell, pbc, cutoff, radial_config
        )
    return feature_sum.detach(), jacobian.detach()


def _feature_sum_and_jacobian_chunked(
    arm, positions, atom_types, cell, pbc, cutoff, radial_config, chunk_size=32, jacobian_mode="explicit",
):
    """Dispatch to the explicit (WP4e default) or autograd (oracle/fallback)
    feature-sum-and-Jacobian implementation.

    ``jacobian_mode="explicit"`` falls back to the autograd path when
    ``arm`` was not built with ``use_merged_real_program=True`` (the
    explicit path has no per-content, complex-arithmetic equivalent --
    see ``TaggedArmEvaluator.descriptors_and_jacobian``'s own docstring)
    rather than raising, so ``jacobian_mode``'s new default does not break
    any existing caller that builds an arm without requesting the merged
    program; ``arm.use_merged_real_program`` (already reported by the
    study runner) tells the full story of which path actually ran.
    """

    if jacobian_mode == "explicit":
        if arm._merged_real_evaluator is not None:
            return _feature_sum_and_jacobian_explicit(arm, positions, atom_types, cell, pbc, cutoff, radial_config)
        return _feature_sum_and_jacobian_autograd(
            arm, positions, atom_types, cell, pbc, cutoff, radial_config, chunk_size=chunk_size
        )
    if jacobian_mode == "autograd":
        return _feature_sum_and_jacobian_autograd(
            arm, positions, atom_types, cell, pbc, cutoff, radial_config, chunk_size=chunk_size
        )
    raise ValueError("jacobian_mode must be 'explicit' or 'autograd'.")


def _structure_sha256(atoms):
    payload = {
        "symbols": list(atoms.get_chemical_symbols()),
        "positions": np.asarray(atoms.get_positions(), dtype=np.float64).round(12).tolist(),
        "cell": np.asarray(atoms.cell.array, dtype=np.float64).round(12).tolist(),
        "pbc": [bool(v) for v in atoms.pbc],
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces(), dtype=np.float64).round(12).tolist(),
    }
    return _canonical_json_sha256(payload)


def _content_artifact_hash(entry):
    """One content's compiled-artifact identity for hashing, whether the
    entry holds the full artifact (eager build) or only its precomputed
    ``artifact_self_hash`` (streamed build; see
    :func:`build_streamed_merged_arm_from_catalogue`)."""

    if "artifact_self_hash" in entry:
        return str(entry["artifact_self_hash"])
    return str(entry["compiled"].self_hash)


def _arm_identity_hash(arm):
    body = {
        "role_bindings": [[k, v] for k, v in arm.role_bindings],
        "content_hashes": [_content_artifact_hash(entry) for entry in arm.contents],
        "content_selections": [
            None if entry.get("supported_indices") is None else list(entry.get("supported_indices"))
            for entry in arm.contents
        ],
    }
    return _canonical_json_sha256(body)


def _combination_hash(arm):
    """Combination is now per-content (baked into each content's own

    TaggedMomentEvaluator), deterministic from (compiled.self_hash,
    role_bindings, use_pooled_basis); hash each content's actual realized
    matrix bytes so a change in the pooled-basis algorithm still busts the
    cache even though the inputs are otherwise unchanged.
    """

    digest = hashlib.sha256()
    digest.update(b"pooled" if arm.use_pooled_basis else b"identity_supported")
    for entry in arm.contents:
        if "evaluator" in entry:
            matrix = entry["evaluator"].combination_matrix
            matrix = None if matrix is None else matrix.detach().cpu().numpy()
        else:
            # Streamed entry (no per-content evaluator object retained): the
            # fit-space matrix is the orthonormal transform composed with
            # the pooled matrix when orthonormalized, else the pooled matrix
            # itself -- exactly what TaggedArmEvaluator.__init__ would have
            # built as entry["evaluator"].combination_matrix.
            pooled = entry.get("pooled_combination_matrix")
            S = entry.get("orthonormal_transform")
            matrix = None if pooled is None else (pooled if S is None else S @ pooled)
        if matrix is not None:
            digest.update(np.ascontiguousarray(matrix).tobytes())
    return digest.hexdigest()


def _radial_config_hash(cutoff, radial_config):
    return _canonical_json_sha256({"cutoff": float(cutoff), "radial_config": dict(radial_config)})


def cache_key(arm_hash, combination_hash, radial_hash, structure_sha256):
    body = f"{arm_hash}:{combination_hash}:{radial_hash}:{structure_sha256}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _cache_path(cache_dir, key):
    return Path(cache_dir) / f"{key}.npz"


def structure_row(
    arm,
    atoms,
    *,
    cutoff,
    radial_config,
    chunk_size=32,
    cache_dir=None,
    arm_hash=None,
    combination_hash=None,
    radial_hash=None,
    jacobian_mode="explicit",
):
    """One structure's cached (feature_sum, jacobian, targets); recomputed on a cache miss.

    ``jacobian_mode`` (WP4e) is ``"explicit"`` (default: the autograd-free,
    analytic-derivative-based path, :func:`~ye3t_methods.atomistic.tagged_cauchy_linear.
    RealMomentEvaluator.evaluate_real_with_jacobian` via a merged real
    program -- requires ``arm`` built with ``use_merged_real_program=True``)
    or ``"autograd"`` (the original per-feature-chunk ``torch.autograd.grad``
    path, retained as the oracle/fallback). Included in the on-disk cache
    key so switching modes always recomputes rather than silently reusing a
    row computed the other way (both are certified to agree to 1e-10, but
    the cache key does not rely on that fact).
    """

    structure_hash = _structure_sha256(atoms)
    key = None
    if cache_dir is not None:
        arm_hash = _arm_identity_hash(arm) if arm_hash is None else arm_hash
        combination_hash = _combination_hash(arm) if combination_hash is None else combination_hash
        radial_hash = _radial_config_hash(cutoff, radial_config) if radial_hash is None else radial_hash
        key = cache_key(arm_hash, combination_hash, radial_hash, structure_hash)
        key = hashlib.sha256(f"{key}:{jacobian_mode}".encode("utf-8")).hexdigest()
        path = _cache_path(cache_dir, key)
        if path.exists():
            with np.load(path) as handle:
                return {
                    "feature_sum": handle["feature_sum"],
                    "jacobian": handle["jacobian"],
                    "energy_target": float(handle["energy_target"]),
                    "force_target": handle["force_target"],
                    "n_atoms": int(handle["n_atoms"]),
                    "symbols": list(handle["symbols"]),
                    "cache_hit": True,
                }
    symbols = list(atoms.get_chemical_symbols())
    n_atoms = len(atoms)
    positions = torch.tensor(np.asarray(atoms.get_positions(), dtype=np.float64), dtype=torch.float64)
    species_order = _species_order_from_channels(arm.global_channels)
    species_index = {name: i for i, name in enumerate(species_order)}
    atom_types = torch.tensor([species_index[s] for s in symbols], dtype=torch.long)
    periodic = bool(np.any(np.asarray(atoms.pbc, dtype=bool)))
    cell = torch.tensor(np.asarray(atoms.cell.array, dtype=np.float64), dtype=torch.float64) if periodic else None
    pbc = tuple(bool(v) for v in atoms.pbc) if periodic else None
    positions.requires_grad_(True)
    feature_sum, jacobian = _feature_sum_and_jacobian_chunked(
        arm, positions, atom_types, cell, pbc, cutoff, radial_config, chunk_size=chunk_size,
        jacobian_mode=jacobian_mode,
    )
    row = {
        "feature_sum": feature_sum.numpy(),
        "jacobian": jacobian.numpy(),
        "energy_target": float(atoms.get_potential_energy()),
        "force_target": np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1),
        "n_atoms": n_atoms,
        "symbols": symbols,
        "cache_hit": False,
    }
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _cache_path(cache_dir, key),
            feature_sum=row["feature_sum"],
            jacobian=row["jacobian"],
            energy_target=row["energy_target"],
            force_target=row["force_target"],
            n_atoms=row["n_atoms"],
            symbols=np.asarray(symbols),
        )
    return row


def _offset_columns_energy(symbols, species_order, n_atoms, mode):
    if mode == "composition_fixed":
        return np.zeros((0,), dtype=np.float64)
    counts = np.zeros(len(species_order), dtype=np.float64)
    for symbol in symbols:
        counts[species_order.index(symbol)] += 1.0
    return counts / float(n_atoms)


def compute_feature_scale(
    arm, frames, *, cutoff, radial_config, cache_dir=None, chunk_size=32,
    jacobian_mode="explicit",
):
    """Pass 1: unweighted per-column feature RMS from energy-row features only."""

    n_features = arm.feature_count
    accum = np.zeros(n_features, dtype=np.float64)
    for atoms in frames:
        row = structure_row(
            arm, atoms, cutoff=cutoff, radial_config=radial_config, cache_dir=cache_dir,
            chunk_size=chunk_size, jacobian_mode=jacobian_mode,
        )
        energy_row = row["feature_sum"] / float(row["n_atoms"])
        accum += energy_row * energy_row
    scale = np.sqrt(accum / max(1, len(frames)))
    return np.clip(scale, 1.0e-12, None)


def accumulate_normal_equations(
    arm,
    frames,
    *,
    feature_scale,
    offsets_mode,
    species_order,
    cutoff,
    radial_config,
    energy_weight=1.0,
    force_weight=1.0,
    structure_weights=None,
    cache_dir=None,
    fixed_offsets=None,
    chunk_size=32,
    jacobian_mode="explicit",
):
    """Pass 2: streamed, scaled, weighted float64 normal equations, one structure at a time.

    Column order: [feature columns (scaled by feature_scale)] + [offset
    columns (unscaled; empty in composition_fixed mode)]. In
    ``composition_fixed`` mode, ``fixed_offsets`` (species -> E0) is
    subtracted from each structure's energy target before it enters the
    accumulation, per the fitting contract ("offsets subtracted from energy
    targets; no column"). Returns XtX, Xty, yty, and row counts; never
    materializes a dense design matrix.
    """

    n_features = arm.feature_count
    n_offsets = 0 if offsets_mode == "composition_fixed" else len(species_order)
    width = n_features + n_offsets
    fixed_offsets = {} if fixed_offsets is None else fixed_offsets
    XtX = np.zeros((width, width), dtype=np.float64)
    Xty = np.zeros((width,), dtype=np.float64)
    yty = 0.0
    energy_rows = 0
    force_rows = 0
    if structure_weights is None:
        structure_weights = np.ones(len(frames), dtype=np.float64)
    for atoms, base_weight in zip(frames, structure_weights):
        row = structure_row(
            arm, atoms, cutoff=cutoff, radial_config=radial_config, cache_dir=cache_dir,
            chunk_size=chunk_size, jacobian_mode=jacobian_mode,
        )
        n_atoms = row["n_atoms"]
        symbols = row["symbols"]
        feature_energy = row["feature_sum"] / n_atoms / feature_scale
        offset_energy = _offset_columns_energy(symbols, species_order, n_atoms, offsets_mode)
        design_energy = np.concatenate([feature_energy, offset_energy])
        raw_energy = row["energy_target"]
        if offsets_mode == "composition_fixed":
            raw_energy -= sum(float(fixed_offsets.get(symbol, 0.0)) for symbol in symbols)
        energy_target = raw_energy / n_atoms
        w_energy = math.sqrt(max(float(base_weight), 0.0) * float(energy_weight))
        vector = w_energy * design_energy
        target = w_energy * energy_target
        XtX += np.outer(vector, vector)
        Xty += vector * target
        yty += target * target
        energy_rows += 1

        jac = row["jacobian"].reshape(n_features, 3 * n_atoms).T  # [3N, n_features]
        force_feature = -jac / feature_scale[None, :]
        offset_force = np.zeros((3 * n_atoms, n_offsets), dtype=np.float64)
        design_force = np.concatenate([force_feature, offset_force], axis=1)
        force_target = row["force_target"]
        w_force = math.sqrt(max(float(base_weight), 0.0) * float(force_weight))
        vectors = w_force * design_force
        targets = w_force * force_target
        XtX += vectors.T @ vectors
        Xty += vectors.T @ targets
        yty += float(np.dot(targets, targets))
        force_rows += vectors.shape[0]
    return {
        "XtX": XtX,
        "Xty": Xty,
        "yty": yty,
        "energy_rows": energy_rows,
        "force_rows": force_rows,
        "n_features": n_features,
        "n_offsets": n_offsets,
    }


def solve_ridge_grid(normal, alpha_grid, *, maximum_condition=1.0e24):
    """Solve every alpha from the same accumulated sufficient statistics.

    Offset columns are never penalized. Reports the extreme eigenvalues of
    the (unpenalized) scaled Gram as conditioning, matching the ordinary
    linear-ACE streamed-Gram convention. Fails closed (raises) on
    non-finite XtX/Xty or a non-finite solve.
    """

    XtX = np.asarray(normal["XtX"], dtype=np.float64)
    Xty = np.asarray(normal["Xty"], dtype=np.float64)
    n_features = int(normal["n_features"])
    n_offsets = int(normal["n_offsets"])
    width = n_features + n_offsets
    if not np.all(np.isfinite(XtX)) or not np.all(np.isfinite(Xty)):
        raise FloatingPointError("Streamed normal equations contain non-finite values.")
    symmetry_defect = float(np.max(np.abs(XtX - XtX.T)))
    gram = 0.5 * (XtX + XtX.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    conditioning = {
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "symmetry_defect": symmetry_defect,
    }
    penalty_mask = np.zeros(width, dtype=np.float64)
    penalty_mask[:n_features] = 1.0
    results = []
    for alpha in alpha_grid:
        alpha = float(alpha)
        system = gram + alpha * np.diag(penalty_mask)
        try:
            beta = np.linalg.solve(system, Xty)
        except np.linalg.LinAlgError as exc:
            results.append({"alpha": alpha, "ok": False, "error": str(exc)})
            continue
        if not np.all(np.isfinite(beta)):
            results.append({"alpha": alpha, "ok": False, "error": "non-finite solution"})
            continue
        system_eigs = np.linalg.eigvalsh(system)
        positive = system_eigs[system_eigs > 0.0]
        condition_number = (
            math.inf if positive.size != system_eigs.size else float(positive[-1] / positive[0])
        )
        results.append(
            {
                "alpha": alpha,
                "ok": condition_number <= maximum_condition,
                "beta_scaled": beta,
                "condition_number": condition_number,
            }
        )
    return results, conditioning


def _predict_energy_and_forces(row, beta_features, offsets, species_order, offsets_mode, fixed_offsets=None):
    n_atoms = row["n_atoms"]
    symbols = row["symbols"]
    feature_sum = row["feature_sum"]
    energy = float(np.dot(feature_sum, beta_features))
    if offsets_mode == "fitted_species_offsets":
        for symbol in symbols:
            energy += offsets[species_order.index(symbol)]
    else:
        fixed_offsets = {} if fixed_offsets is None else fixed_offsets
        for symbol in symbols:
            energy += float(fixed_offsets.get(symbol, 0.0))
    forces = -row["jacobian"].reshape(beta_features.shape[0], -1).T @ beta_features
    return energy, forces.reshape(n_atoms, 3)


def evaluate_rmse(
    arm,
    frames,
    beta_full,
    *,
    feature_scale,
    offsets_mode,
    species_order,
    cutoff,
    radial_config,
    cache_dir=None,
    fixed_offsets=None,
    chunk_size=32,
    jacobian_mode="explicit",
):
    """Streamed energy (eV/atom) and force (eV/A) RMSE for one beta vector."""

    n_features = arm.feature_count
    beta_features = beta_full[:n_features] / feature_scale
    offsets = beta_full[n_features:] if offsets_mode == "fitted_species_offsets" else None
    energy_sq = 0.0
    force_sq = 0.0
    force_count = 0
    for atoms in frames:
        row = structure_row(
            arm, atoms, cutoff=cutoff, radial_config=radial_config, cache_dir=cache_dir,
            chunk_size=chunk_size, jacobian_mode=jacobian_mode,
        )
        energy_pred, force_pred = _predict_energy_and_forces(
            row, beta_features, offsets, species_order, offsets_mode, fixed_offsets
        )
        energy_error = energy_pred / row["n_atoms"] - row["energy_target"] / row["n_atoms"]
        energy_sq += energy_error * energy_error
        force_error = force_pred.reshape(-1) - row["force_target"]
        force_sq += float(np.dot(force_error, force_error))
        force_count += force_error.size
    energy_rmse = math.sqrt(energy_sq / max(1, len(frames)))
    force_rmse = math.sqrt(force_sq / max(1, force_count))
    return energy_rmse, force_rmse


def select_alpha(train_metrics, val_metrics):
    """Pick alpha minimizing normalized energy MSE + normalized force MSE (train-set scales)."""

    train_energy_scale = max(1.0e-12, max(m["energy_rmse"] for m in train_metrics.values()))
    train_force_scale = max(1.0e-12, max(m["force_rmse"] for m in train_metrics.values()))
    best_alpha = None
    best_objective = math.inf
    for alpha, metrics in val_metrics.items():
        objective = (metrics["energy_rmse"] / train_energy_scale) ** 2 + (
            metrics["force_rmse"] / train_force_scale
        ) ** 2
        if objective < best_objective:
            best_objective = objective
            best_alpha = alpha
    return best_alpha, best_objective


def fit_tagged_arm(
    train_frames,
    val_frames,
    arm,
    *,
    offsets_mode="fitted_species_offsets",
    fixed_offsets=None,
    alpha_grid=(1.0e-6, 1.0e-4, 1.0e-2),
    cutoff,
    radial_config,
    energy_weight=1.0,
    force_weight=1.0,
    structure_weighting=None,
    cache_dir=None,
    chunk_size=32,
    maximum_condition=1.0e24,
    jacobian_mode="explicit",
):
    """Full two-pass streamed fit: scale, accumulate, ridge grid, select, evaluate.

    ``cutoff``/``radial_config`` have no default (WPA de-elementification):
    every caller passes the config-derived values explicitly. ``jacobian_
    mode`` (WP4e, default ``"explicit"``) and ``chunk_size`` are threaded
    through to every ``structure_row`` call below (``chunk_size`` is only
    consulted by the ``"autograd"`` path; the ``"explicit"`` path has no
    equivalent chunking knob -- see ``_feature_sum_and_jacobian_explicit``'s
    own docstring for why).
    """

    species_order = list(_species_order_from_channels(arm.global_channels))
    if offsets_mode not in ("fitted_species_offsets", "composition_fixed"):
        raise ValueError("offsets_mode must be 'fitted_species_offsets' or 'composition_fixed'.")

    t0 = time.time()
    train_frames = list(train_frames)
    val_frames = list(val_frames)
    fixed_offsets = {} if fixed_offsets is None else dict(fixed_offsets)

    weights = np.ones(len(train_frames), dtype=np.float64)
    if structure_weighting is not None:
        weights, _meta = structure_fit_weights(
            train_frames,
            structure_group_key=structure_weighting.get("group_key"),
            structure_group_weights=structure_weighting.get("group_weights"),
            structure_group_default_weight=structure_weighting.get("default_weight"),
            structure_group_normalize_mean=structure_weighting.get("normalize_mean", True),
        )

    feature_scale = compute_feature_scale(
        arm, train_frames, cutoff=cutoff, radial_config=radial_config, cache_dir=cache_dir,
        chunk_size=chunk_size, jacobian_mode=jacobian_mode,
    )
    scale_seconds = time.time() - t0

    t1 = time.time()
    normal = accumulate_normal_equations(
        arm,
        train_frames,
        feature_scale=feature_scale,
        offsets_mode=offsets_mode,
        species_order=species_order,
        cutoff=cutoff,
        radial_config=radial_config,
        energy_weight=energy_weight,
        force_weight=force_weight,
        structure_weights=weights,
        cache_dir=cache_dir,
        fixed_offsets=fixed_offsets,
        chunk_size=chunk_size,
        jacobian_mode=jacobian_mode,
    )
    accumulate_seconds = time.time() - t1

    t2 = time.time()
    solved, conditioning = solve_ridge_grid(normal, alpha_grid, maximum_condition=maximum_condition)
    solve_seconds = time.time() - t2

    t3 = time.time()
    train_metrics = {}
    val_metrics = {}
    for result in solved:
        if not result.get("ok"):
            continue
        alpha = result["alpha"]
        beta_full = result["beta_scaled"]
        train_e, train_f = evaluate_rmse(
            arm, train_frames, beta_full, feature_scale=feature_scale, offsets_mode=offsets_mode,
            species_order=species_order, cutoff=cutoff, radial_config=radial_config, cache_dir=cache_dir,
            fixed_offsets=fixed_offsets, chunk_size=chunk_size, jacobian_mode=jacobian_mode,
        )
        val_e, val_f = evaluate_rmse(
            arm, val_frames, beta_full, feature_scale=feature_scale, offsets_mode=offsets_mode,
            species_order=species_order, cutoff=cutoff, radial_config=radial_config, cache_dir=cache_dir,
            fixed_offsets=fixed_offsets, chunk_size=chunk_size, jacobian_mode=jacobian_mode,
        )
        train_metrics[alpha] = {"energy_rmse": train_e, "force_rmse": train_f}
        val_metrics[alpha] = {"energy_rmse": val_e, "force_rmse": val_f}
    evaluate_seconds = time.time() - t3

    selected_alpha, selected_objective = select_alpha(train_metrics, val_metrics)
    selected_result = next(r for r in solved if r["alpha"] == selected_alpha)
    beta_full = selected_result["beta_scaled"]
    n_features = arm.feature_count
    beta_physical = beta_full[:n_features] / feature_scale
    offsets_physical = (
        {species_order[i]: float(beta_full[n_features + i]) for i in range(len(species_order))}
        if offsets_mode == "fitted_species_offsets"
        else dict(fixed_offsets or {})
    )

    summary = {
        "offsets_mode": offsets_mode,
        "species_order": species_order,
        "feature_count": n_features,
        "train_structures": len(train_frames),
        "val_structures": len(val_frames),
        "train_energy_rows": normal["energy_rows"],
        "train_force_rows": normal["force_rows"],
        "conditioning": conditioning,
        "alpha_grid": [float(a) for a in alpha_grid],
        "train_metrics_by_alpha": {str(k): v for k, v in train_metrics.items()},
        "val_metrics_by_alpha": {str(k): v for k, v in val_metrics.items()},
        "selected_alpha": selected_alpha,
        "selected_objective": selected_objective,
        "timings_seconds": {
            "scale_pass": scale_seconds,
            "accumulate_pass": accumulate_seconds,
            "ridge_solve": solve_seconds,
            "evaluate": evaluate_seconds,
        },
    }
    return beta_physical, offsets_physical, feature_scale, summary


def export_tagged_fit_bundle(
    path, arm, beta, offsets, offsets_mode, *, cutoff, radial_config, metrics=None, manual_catalogue=None
):
    """Write a hash-bound JSON bundle: per-content artifacts, offsets, fit metadata.

    Same canonical-JSON sha256 self_hash convention as WP3b's ``export_
    tagged_model``, extended for a multi-artifact arm (WP3b's schema
    assumes exactly one compiled artifact; an arm concatenates several).

    Schema v2 (WP4c) additionally persists ``per_species_beta`` and (when
    true) ``central_species_order``, and ``fit_coordinates``, so
    :func:`load_tagged_fit_bundle` can reconstruct an ``arm`` whose
    ``feature_count`` actually matches the saved (possibly per-species-
    block-expanded) ``beta`` -- WP3e's same "beta and the feature space it
    was fit in must always agree" invariant, extended to the multi-content
    arm bundle. v1 bundles (neither field present) load exactly as before
    (``per_species_beta=False``, ``fit_coordinates="pooled"``).
    """

    catalogue_hash = getattr(arm, "catalogue_hash", None)
    catalogue_path = getattr(arm, "catalogue_path", None)

    def _content_body(entry):
        if "compiled" in entry:
            combination_matrix = (
                None
                if entry["evaluator"].combination_matrix is None
                else entry["evaluator"].combination_matrix.detach().cpu().numpy().tolist()
            )
            return {
                "content_id": entry.get("content_id"),
                "compiled_artifact": entry["compiled"].to_dict(),
                "channel_binding_override": entry.get("channel_binding_override"),
                "catalogue_reference": None,
                "descriptor_selection": (
                    None if entry.get("supported_indices") is None else list(entry.get("supported_indices"))
                ),
                "combination_matrix": combination_matrix,
                "offset": entry["offset"],
                "descriptor_count": entry["descriptor_count"],
                "supported_count": entry.get("supported_count"),
                "pooled_count": entry.get("pooled_count"),
            }
        # Streamed entry (build_streamed_merged_arm_from_catalogue): no full
        # compiled artifact was ever retained after its contribution was
        # merged. Record a catalogue reference (content index + the exact
        # hashes _artifact_from_catalogue_content verifies against) plus the
        # small pooled/orthonormal matrices instead of re-embedding a
        # multi-megabyte sympy artifact per content -- load_tagged_fit_
        # bundle rebuilds such a bundle by rerunning the streamed builder
        # against the SAME catalogue_path/catalogue_hash (recorded once at
        # the bundle's top level below), not by reloading this per-content
        # block directly.
        pooled = entry.get("pooled_combination_matrix")
        S = entry.get("orthonormal_transform")
        combination_matrix = None
        if pooled is not None:
            combination_matrix = (pooled if S is None else S @ pooled).tolist()
        return {
            "content_id": entry.get("content_id"),
            "compiled_artifact": None,
            "catalogue_reference": {
                "catalogue_content_index": entry.get("catalogue_content_index"),
                "artifact_self_hash": entry.get("artifact_self_hash"),
            },
            "descriptor_selection": None,
            "combination_matrix": combination_matrix,
            "offset": entry["offset"],
            "descriptor_count": entry["descriptor_count"],
            "supported_count": entry.get("supported_count"),
            "pooled_count": entry.get("pooled_count"),
        }

    body = {
        "schema": TAGGED_FIT_BUNDLE_SCHEMA,
        "role_bindings": [[k, v] for k, v in arm.role_bindings],
        "use_pooled_basis": bool(arm.use_pooled_basis),
        "fit_coordinates": str(arm.fit_coordinates),
        "per_species_beta": bool(arm.per_species_beta),
        "central_species_order": (
            list(arm.central_species_order) if arm.per_species_beta else None
        ),
        "catalogue_hash": catalogue_hash,
        "catalogue_path": None if catalogue_path is None else str(catalogue_path),
        "contents": [_content_body(entry) for entry in arm.contents],
        "global_channels": list(arm.global_channels),
        "beta": np.asarray(beta, dtype=np.float64).tolist(),
        "offsets": {str(k): float(v) for k, v in offsets.items()},
        "offsets_mode": str(offsets_mode),
        "cutoff": float(cutoff),
        "radial_config": dict(radial_config),
    }
    if metrics is not None:
        body["metrics"] = _jsonable(metrics)
    if manual_catalogue is not None:
        body["manual_catalogue"] = _jsonable(manual_catalogue)
    self_hash = _canonical_json_sha256(body)
    payload = dict(body)
    payload["self_hash"] = self_hash
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
    return payload


def load_tagged_fit_bundle(path, *, cache_dir=None):
    """Load and hash-verify a bundle written by export_tagged_fit_bundle; rebuilds the arm evaluator.

    A bundle whose contents are catalogue-referenced rather than embedded
    (:func:`export_tagged_fit_bundle`'s streamed-entry branch -- written for
    an arm built by :func:`build_streamed_merged_arm_from_catalogue`) is
    rebuilt by rerunning that SAME streamed builder against
    ``payload["catalogue_path"]`` (loaded and catalogue_hash-verified by
    :func:`~ye3t.couplings.tagged_catalogue.load_catalogue`, then checked
    against the bundle's own recorded ``catalogue_hash`` so a bundle can
    never silently be paired with the wrong catalogue file); pass
    ``cache_dir`` to reuse an on-disk compiled-artifact cache instead of
    recompiling every content from scratch. A bundle with at least one
    embedded content (the pre-WPA path, or a manual/non-catalogue arm)
    rebuilds exactly as before, content by content, via
    ``TaggedArmEvaluator``'s eager (list-based) constructor.
    """

    from ye3t.couplings import CompiledLiftedCauchyScalar
    from ye3t.couplings.tagged_catalogue import load_catalogue

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload = dict(payload)
    expected = str(payload.pop("self_hash", ""))
    actual = _canonical_json_sha256(payload)
    if not expected or expected != actual:
        raise ValueError("Tagged-Cauchy fit bundle self_hash mismatch.")
    if str(payload.get("schema")) not in (TAGGED_FIT_BUNDLE_SCHEMA_V1, TAGGED_FIT_BUNDLE_SCHEMA_V2):
        raise ValueError("Unsupported tagged-Cauchy fit bundle schema.")
    use_pooled_basis = bool(payload.get("use_pooled_basis", True))
    fit_coordinates = str(payload.get("fit_coordinates", "pooled"))
    per_species_beta = bool(payload.get("per_species_beta", False))
    content_records = payload["contents"]
    streamed = any(record.get("compiled_artifact") is None for record in content_records)

    if streamed:
        catalogue_path = payload.get("catalogue_path")
        if catalogue_path is None:
            raise ValueError(
                "Tagged-Cauchy fit bundle has catalogue-referenced (streamed) contents "
                "but no catalogue_path was recorded; cannot rebuild."
            )
        catalogue_record = load_catalogue(catalogue_path)
        if str(catalogue_record.get("catalogue_hash")) != str(payload.get("catalogue_hash")):
            raise ValueError(
                f"Bundle's recorded catalogue_hash ({payload.get('catalogue_hash')!r}) does not "
                f"match {catalogue_path!r}'s own catalogue_hash ({catalogue_record.get('catalogue_hash')!r})."
            )
        arm = build_streamed_merged_arm_from_catalogue(
            catalogue_record, cache_dir=cache_dir, fit_coordinates=fit_coordinates,
            per_species_beta=per_species_beta, catalogue_path=catalogue_path,
        )
    else:
        contents = []
        for record in content_records:
            compiled = CompiledLiftedCauchyScalar.from_dict(record["compiled_artifact"])
            selection = record.get("descriptor_selection")
            contents.append(
                {
                    "content_id": record.get("content_id"),
                    "compiled": compiled,
                    "channel_binding_override": record.get("channel_binding_override"),
                    "supported_indices": None if selection is None else tuple(int(v) for v in selection),
                    "combination_matrix_override": record.get("combination_matrix"),
                    "supported_count": record.get("supported_count"),
                }
            )
        global_channels = tuple(payload["global_channels"])
        role_bindings = tuple((str(k), int(v)) for k, v in payload["role_bindings"])
        arm = TaggedArmEvaluator(
            role_bindings,
            contents,
            global_channels,
            use_pooled_basis=use_pooled_basis,
            fit_coordinates=fit_coordinates,
            per_species_beta=per_species_beta,
        )
    saved_species_order = payload.get("central_species_order")
    if per_species_beta and saved_species_order is not None:
        saved_species_order = tuple(str(s) for s in saved_species_order)
        if saved_species_order != arm.central_species_order:
            raise ValueError(
                f"Bundle's saved central_species_order {saved_species_order} does not "
                f"match the reloaded arm's channel-derived order {arm.central_species_order}."
            )
    beta = np.asarray(payload["beta"], dtype=np.float64)
    if int(beta.shape[0]) != arm.feature_count:
        raise ValueError(
            f"Tagged-Cauchy fit bundle is inconsistent: beta length ({beta.shape[0]}) "
            f"does not equal the reconstructed arm's feature_count ({arm.feature_count})."
        )
    offsets = {str(k): float(v) for k, v in payload["offsets"].items()}
    cutoff = float(payload["cutoff"])
    radial_config = dict(payload["radial_config"])
    return arm, beta, offsets, str(payload["offsets_mode"]), payload.get("metrics"), cutoff, radial_config


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    return value


def preflight_report(arm_descriptions, *, train_structure_count=None, train_atom_count=None):
    """Print feature/row/byte estimates before any dataset access (runtime.execution_mode='preflight').

    ``arm_descriptions``: iterable of {"name", "feature_count", "content_count"}
    (computable from compiled artifacts alone). ``train_structure_count``/
    ``train_atom_count`` are optional pre-recorded aggregate counts (e.g.
    from a split manifest's own ``subset`` block), never read from the xyz
    dataset itself. A ``feature_count``/``content_count`` of ``None`` (WP4c:
    a catalogue-based arm whose catalogue file does not exist yet, so
    preflight has nothing to read and refuses to build/compile just to
    answer a preflight query) is reported as ``"unknown"`` rather than
    raising, and skips the byte/row estimates that need a real count.
    """

    lines = []
    for arm in arm_descriptions:
        raw_feature_count = arm["feature_count"]
        raw_content_count = arm["content_count"]
        feature_count = None if raw_feature_count is None else int(raw_feature_count)
        content_count = None if raw_content_count is None else int(raw_content_count)
        line = {
            "arm": arm["name"],
            "content_count": "unknown" if content_count is None else content_count,
            "feature_count": "unknown" if feature_count is None else feature_count,
        }
        if feature_count is not None:
            line["energy_row_bytes"] = 8 * (feature_count + 1)
            if train_atom_count is not None:
                line["train_force_rows_estimate"] = int(3 * train_atom_count)
                line["gram_bytes_estimate"] = 8 * (feature_count + 1) ** 2
        if train_structure_count is not None:
            line["train_energy_rows"] = int(train_structure_count)
        lines.append(line)
        print(json.dumps(line, sort_keys=True))
    return lines
