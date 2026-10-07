"""Preview and explicitly prewarm selected ordinary ACE descriptor artifacts."""

import argparse
from collections.abc import Mapping
import json
from math import prod
import os
from pathlib import Path

from ye3t.couplings import count
from ye3t.core.labels import normalize_compact_label
from ye3t_methods.atomistic.cache import DescriptorBuildCache
from ye3t_methods.atomistic.equivariant_calc.descriptor_sets import (
    DescriptorGenerationSettings,
    compile_descriptor_artifacts,
)


def prewarm_descriptor_catalogue(
    catalogue, *, cache_dir=None, max_labels=100, max_rank=4,
    max_magnetic_elements=100000, apply=False,
):
    """Validate a selected ACE catalogue; materialize only with ``apply=True``.

    ``magnetic_elements`` is a precompile size proxy, not a time estimate.
    The explicit label and rank limits also bound the count preview.
    """
    report_schemas = {
        "ye3t_methods_prewarm_v1": "ye3t_methods_prewarm_report_v1",
        "ye3t_ace_prewarm_v1": "ye3t_ace_prewarm_report_v1",
    }
    if not isinstance(catalogue, Mapping) or set(catalogue) != {
        "schema", "settings", "selected_labels",
    } or not isinstance(catalogue["schema"], str) or catalogue["schema"] not in report_schemas:
        raise ValueError("Expected a ye3t_methods_prewarm_v1 catalogue with settings and selected_labels.")
    limits = (max_labels, max_rank, max_magnetic_elements)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in limits):
        raise ValueError("Prewarm limits must be positive integers.")
    selected = catalogue["selected_labels"]
    if not isinstance(selected, list) or not selected:
        raise ValueError("selected_labels must be a nonempty list.")
    if len(selected) > max_labels:
        raise ValueError("Selected label count exceeds max_labels.")
    labels = tuple(normalize_compact_label(value) for value in selected)
    if len(set(labels)) != len(labels):
        raise ValueError("selected_labels contains duplicates.")
    if max(label.rank for label in labels) > max_rank:
        raise ValueError("Selected label rank exceeds max_rank.")
    magnetic_elements = sum(prod(2 * l + 1 for l in label.l_tuple)
                            for label in labels)
    if magnetic_elements > max_magnetic_elements:
        raise ValueError("Selected labels exceed max_magnetic_elements.")

    settings = DescriptorGenerationSettings.from_dict(catalogue["settings"])
    if settings.basis_type != "no_charge":
        raise ValueError("Prewarm currently supports ordinary no_charge ACE settings.")
    if settings.max_labels_per_rank is not None:
        raise ValueError("Prewarm requires settings.max_labels_per_rank=None.")
    for label in labels:
        if label.rank not in settings.ranks:
            raise ValueError("Selected label rank is outside settings.ranks.")
        index = settings.rank_index(label.rank)
        if (label.L_R != settings.L_R or label.tree_type != settings.tree_type
                or any(n < 1 or n > settings.nmax[index] for n in label.n_tuple)
                or any(l < settings.lmin[index] or l > settings.lmax[index]
                       for l in label.l_tuple)):
            raise ValueError("Selected label is outside descriptor settings.")
        if (settings.parity_filter == "natural"
                and (sum(label.l_tuple) - label.L_R) % 2):
            raise ValueError("Selected label violates the natural parity filter.")

    compiler_requests = {}
    for label in labels:
        key = (label.n_tuple, label.l_tuple, label.L_R, label.tree_type)
        if key not in compiler_requests:
            report = count(
                content=label.n_tuple, input_Ls=label.l_tuple,
                target_L=label.L_R, target_permutation="trivial",
                carrier="ACE_density", tree_schedule=label.tree_type,
                validation_scope="counts",
            )
            compiler_requests[key] = frozenset(
                normalize_compact_label(item)
                for item in report.labels_for_target(label.L_R)
            )
        if label not in compiler_requests[key]:
            raise ValueError("Selected label is not issued by ye3t.couplings.count.")

    result = {
        "schema": report_schemas[catalogue["schema"]],
        "selected_labels": len(labels),
        "compiler_count_requests": len(compiler_requests),
        "magnetic_input_elements_proxy": magnetic_elements,
        "max_labels": max_labels,
        "max_rank": max_rank,
        "max_magnetic_elements": max_magnetic_elements,
        "applied": False,
    }
    if not apply:
        return result
    if cache_dir is None:
        raise ValueError("Applying prewarm requires an explicit cache_dir.")
    if os.environ.get("YE3T_CACHE_MODE", "auto").strip().lower() not in {
        "auto", "refresh", "rebuild",
    }:
        raise ValueError("Applying prewarm requires a writable cache mode.")
    cache = DescriptorBuildCache(cache_dir=cache_dir)
    compiled_labels, _library, _collection = compile_descriptor_artifacts(
        settings, compact_labels=labels, descriptor_cache=cache,
        max_variants_per_label=1,
    )
    if tuple(compiled_labels) != labels:
        raise RuntimeError("Prewarm compiler changed the selected label order.")
    return {**result, "applied": True, "cache_dir": str(Path(cache_dir)),
            "cache_stats": cache.get_stats()}


def main():
    parser = argparse.ArgumentParser(prog="python -m ye3t_methods.cache")
    parser.add_argument("--catalogue", required=True, help="Selected catalogue JSON path")
    parser.add_argument("--cache-dir", help="Explicit destination; required with --apply")
    parser.add_argument("--max-labels", type=int, default=100)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--max-magnetic-elements", type=int, default=100000)
    parser.add_argument("--apply", action="store_true", help="Compile selected labels")
    args = parser.parse_args()
    try:
        path = Path(args.catalogue)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Catalogue JSON exceeds 1 MiB.")
        catalogue = json.loads(path.read_text(encoding="utf-8"))
        result = prewarm_descriptor_catalogue(
            catalogue, cache_dir=args.cache_dir, max_labels=args.max_labels,
            max_rank=args.max_rank,
            max_magnetic_elements=args.max_magnetic_elements, apply=args.apply,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
