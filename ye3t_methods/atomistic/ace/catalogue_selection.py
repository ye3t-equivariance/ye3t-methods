"""Generated, couplings-owned catalogues for linear scalar ACE comparisons."""

import hashlib
import json

from ye3t.core.basis.validation import (
    count_canonical_leaf_labelings,
    iter_canonical_leaf_labelings,
)
from ye3t.couplings import count as count_couplings
from ye3t.couplings import normalize_compact_label
from ye3t.couplings import symmetric_power_materialization_resource_report


def _stable_hash(value):
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _value_by_order(mapping, tensor_order):
    if tensor_order in mapping:
        return mapping[tensor_order]
    key = str(int(tensor_order))
    if key in mapping:
        return mapping[key]
    raise KeyError(f"Missing tensor-order setting for N={int(tensor_order)}.")


def _normalized_request(basis):
    basis = dict(basis)
    tensor_orders = tuple(int(value) for value in basis["tensor_orders"])
    if not tensor_orders or len(set(tensor_orders)) != len(tensor_orders):
        raise ValueError("tensor_orders must contain distinct positive integers.")
    if any(value <= 0 for value in tensor_orders):
        raise ValueError("tensor_orders must contain distinct positive integers.")
    nmax = {
        order: int(_value_by_order(basis["nmax_by_tensor_order"], order))
        for order in tensor_orders
    }
    explicit_content_ids = basis.get("content_ids_by_tensor_order")
    content_ids = {
        order: tuple(int(value) for value in (
            _value_by_order(explicit_content_ids, order)
            if explicit_content_ids is not None else range(1, nmax[order] + 1)))
        for order in tensor_orders
    }
    lmax = {
        order: int(_value_by_order(basis["lmax_by_tensor_order"], order))
        for order in tensor_orders
    }
    partitions = {
        order: tuple(
            tuple(int(value) for value in row)
            for row in _value_by_order(
                basis["channel_multiplicity_partitions_by_order"], order
            )
        )
        for order in tensor_orders
    }
    for order in tensor_orders:
        if nmax[order] <= 0 or lmax[order] < 0:
            raise ValueError("nmax must be positive and lmax must be non-negative.")
        if (not content_ids[order] or
                tuple(sorted(set(content_ids[order]))) != content_ids[order] or
                content_ids[order][0] < 1 or content_ids[order][-1] > nmax[order]):
            raise ValueError("Content IDs must be ascending, unique, and within nmax.")
        for partition in partitions[order]:
            if not partition or any(value <= 0 for value in partition):
                raise ValueError("Channel-multiplicity partitions must be positive.")
            if sum(partition) != order:
                raise ValueError(
                    f"Channel-multiplicity partition {partition} does not sum to N={order}."
                )
    targets = tuple(
        sorted({int(value) for value in basis["target_descriptor_counts"]})
    )
    if not targets or any(value <= 0 for value in targets):
        raise ValueError("target_descriptor_counts must contain positive integers.")
    compiler = dict(basis.get("compiler", {}))
    allowed_compiler = {
        "coefficient_materialization",
        "constructor_backend",
        "coordinate_contract",
        "maximum_exact_symbolic_bytes",
    }
    unknown_compiler = set(compiler) - allowed_compiler
    if unknown_compiler:
        raise ValueError(
            "Unsupported catalogue compiler fields: "
            + ", ".join(sorted(str(value) for value in unknown_compiler))
        )
    compiler.setdefault("coefficient_materialization", "exact")
    compiler.setdefault("constructor_backend", "python")
    compiler.setdefault("coordinate_contract", "pace_compatible_exact")
    compiler.setdefault("maximum_exact_symbolic_bytes", 256 * 1024 * 1024)
    compiler["maximum_exact_symbolic_bytes"] = int(
        compiler["maximum_exact_symbolic_bytes"]
    )
    if compiler["maximum_exact_symbolic_bytes"] <= 0:
        raise ValueError("maximum_exact_symbolic_bytes must be positive.")
    return {
        "tensor_orders": tensor_orders,
        "nmax": nmax,
        "content_ids": content_ids,
        "lmax": lmax,
        "partitions": partitions,
        "targets": targets,
        "compiler": compiler,
    }


def linear_catalogue_preflight(basis):
    """Count selected input contents without compiling coupling coefficients."""

    request = _normalized_request(basis)
    rows = []
    total = 0
    for tensor_order in request["tensor_orders"]:
        count = count_canonical_leaf_labelings(
            tensor_order,
            request["content_ids"][tensor_order],
            range(request["lmax"][tensor_order] + 1),
            multiplicity_partitions=request["partitions"][tensor_order],
        )
        total += int(count)
        rows.append(
            {
                "tensor_order_N": tensor_order,
                "nmax": request["nmax"][tensor_order],
                "lmax": request["lmax"][tensor_order],
                "channel_multiplicity_partitions": [
                    list(value) for value in request["partitions"][tensor_order]
                ],
                "candidate_fixed_contents": int(count),
            }
        )
    identity = {
        "tensor_orders": list(request["tensor_orders"]),
        "nmax_by_tensor_order": {
            str(key): value for key, value in request["nmax"].items()
        },
        "lmax_by_tensor_order": {
            str(key): value for key, value in request["lmax"].items()
        },
        "channel_multiplicity_partitions_by_order": {
            str(key): [list(value) for value in values]
            for key, values in request["partitions"].items()
        },
        "target_descriptor_counts": list(request["targets"]),
        "compiler": dict(request["compiler"]),
    }
    if basis.get("content_ids_by_tensor_order") is not None:
        identity["content_ids_by_tensor_order"] = {
            str(key): list(value) for key, value in request["content_ids"].items()
        }
    return {
        "schema": "ye3t_linear_catalogue_preflight_v1",
        "coefficient_compilation_performed": False,
        "label_source": "ye3t.couplings.count",
        "content_source": "ye3t.core.basis.iter_canonical_leaf_labelings",
        "request": identity,
        "request_sha256": _stable_hash(identity),
        "by_tensor_order": rows,
        "candidate_fixed_contents": total,
    }


def _content_partition(n_tuple, l_tuple):
    counts = {}
    for key in zip(n_tuple, l_tuple, strict=True):
        counts[key] = counts.get(key, 0) + 1
    return tuple(sorted(counts.values(), reverse=True))


def _content_resource_bytes(content):
    n_tuple, l_tuple = content
    counts = {}
    for key in zip(n_tuple, l_tuple, strict=True):
        counts[key] = counts.get(key, 0) + 1
    estimates = []
    for (_n, angular), power in counts.items():
        if power <= 1:
            continue
        report = symmetric_power_materialization_resource_report(
            power,
            angular,
            policy="exact",
            maximum_exact_symbolic_bytes=1 << 62,
        )
        estimates.append(int(report["estimated_exact_symbolic_workspace_bytes"]))
    return max(estimates, default=0)


def _content_sort_key(content, maximum_exact_symbolic_bytes):
    n_tuple, l_tuple = content
    partition = _content_partition(n_tuple, l_tuple)
    resource_bytes = _content_resource_bytes(content)
    return (
        int(resource_bytes > maximum_exact_symbolic_bytes),
        len(partition),
        resource_bytes,
        sum(int(value) for value in l_tuple),
        max((int(value) for value in l_tuple), default=0),
        sum(int(value) for value in n_tuple),
        tuple(int(value) for value in l_tuple),
        tuple(int(value) for value in n_tuple),
    )


def _candidate_contents(request, tensor_order):
    values = iter_canonical_leaf_labelings(
        tensor_order,
        request["content_ids"][tensor_order],
        range(request["lmax"][tensor_order] + 1),
        multiplicity_partitions=request["partitions"][tensor_order],
    )
    maximum = request["compiler"]["maximum_exact_symbolic_bytes"]
    return tuple(sorted(values, key=lambda value: _content_sort_key(value, maximum)))


def _single_tag_count(tagged):
    values = tagged.get("tag_counts_s", tagged.get("tag_count_s", (2,)))
    if isinstance(values, int):
        values = (values,)
    else:
        values = tuple(int(value) for value in values)
    if len(values) != 1 or values[0] < 0:
        raise ValueError(
            "This workflow builds one tag-count arm at a time; tag_counts_s "
            "must contain exactly one non-negative integer."
        )
    return values[0]


def _policy_by_order(tagged, field, tensor_order, allowed, default):
    mapping = tagged.get(f"{field}_by_tensor_order", {})
    value = _value_by_order(mapping, tensor_order) if mapping else tagged.get(field, default)
    value = str(value).strip().lower()
    if value not in allowed:
        raise ValueError(
            f"tagged.{field} for N={tensor_order} must be one of "
            f"{sorted(allowed)}; received {value!r}."
        )
    return value


def _content_limit(tagged, tensor_order):
    mapping = tagged.get("maximum_fixed_contents_by_tensor_order")
    if mapping is None:
        return None
    value = _value_by_order(mapping, tensor_order)
    if value is None or str(value).strip().lower() == "all":
        return None
    value = int(value)
    if value < 0:
        raise ValueError("maximum_fixed_contents_by_tensor_order must be non-negative.")
    return value


def _tagged_component(tensor_order, n_tuple, l_tuple, kappa_policy, lambda_policy):
    channels = {}
    for n_value, l_value in zip(n_tuple, l_tuple, strict=True):
        key = (int(n_value), int(l_value))
        channels[key] = channels.get(key, 0) + 1
    ordered = tuple(sorted(channels.items()))
    return {
        "tensor_order": int(tensor_order),
        "n": [int(value) for value in n_tuple],
        "l": [int(value) for value in l_tuple],
        "content_pattern": [int(count) for _key, count in ordered],
        "block_sizes": [int(count) for _key, count in ordered],
        "block_complete_channel_keys": [
            {"radial_channel": int(key[0]) - 1, "l": int(key[1])}
            for key, _count in ordered
        ],
        "kappa_policy": "all" if kappa_policy == "all_valid" else "trivial",
        "block_lambda_policy": "all" if lambda_policy == "all_valid" else "zero",
    }


def resolve_tagged_content_schedule(basis, catalogue_id="generated_tagged_cauchy"):
    """Generate fixed tagged contents from public rank/channel caps.

    This function chooses only fixed ``(n, l)`` contents and block sizes.  It
    never invents a coupling label: valid Young sectors, rotational
    intermediates, multiplicities, and coefficients are still obtained from
    ``ye3t.couplings.count/plan/compile`` by the catalogue materializer.
    """

    request = _normalized_request(basis)
    tagged = dict(basis.get("tagged", {}))
    mode = str(tagged.get("component_schedule_mode", "fixed_file")).strip().lower()
    if mode != "generated":
        raise ValueError(
            "resolve_tagged_content_schedule requires "
            "basis.tagged.component_schedule_mode='generated'."
        )
    tag_count = _single_tag_count(tagged)
    preflight = linear_catalogue_preflight(basis)
    preflight_by_order = {
        int(row["tensor_order_N"]): row for row in preflight["by_tensor_order"]
    }
    role_bindings = [
        *[["edge", index] for index in range(tag_count)],
        ["density", 0],
    ]
    components = []
    by_order = []
    for tensor_order in request["tensor_orders"]:
        limit = _content_limit(tagged, tensor_order)
        if tensor_order < tag_count or limit == 0:
            candidates = ()
        else:
            candidates = _candidate_contents(request, tensor_order)
            if limit is not None:
                candidates = candidates[:limit]
        kappa_policy = _policy_by_order(
            tagged,
            "intermediate_kappa_policy",
            tensor_order,
            {"all_valid", "trivial_only"},
            "all_valid",
        )
        lambda_policy = _policy_by_order(
            tagged,
            "intermediate_Lambda_policy",
            tensor_order,
            {"all_valid", "zero_only"},
            "all_valid",
        )
        start = len(components)
        for n_tuple, l_tuple in candidates:
            component = _tagged_component(
                tensor_order,
                n_tuple,
                l_tuple,
                kappa_policy,
                lambda_policy,
            )
            component["component_index"] = len(components) + 1
            components.append(component)
        by_order.append(
            {
                "tensor_order_N": int(tensor_order),
                "candidate_fixed_contents": int(
                    preflight_by_order[tensor_order]["candidate_fixed_contents"]
                ),
                "selected_fixed_contents": len(components) - start,
                "maximum_fixed_contents": limit,
                "intermediate_kappa_policy": kappa_policy,
                "intermediate_Lambda_policy": lambda_policy,
            }
        )
    if not components:
        raise RuntimeError("The generated tagged schedule contains no fixed contents.")
    identity = {
        "catalogue_id": str(catalogue_id),
        "ordinary_request_sha256": preflight["request_sha256"],
        "tag_count_s": int(tag_count),
        "role_bindings": role_bindings,
        "components": components,
    }
    schedule = {
        "schema": "ye3t_generated_tagged_content_schedule_v1",
        "selection_policy": (
            "rank-specific nmax/lmax and repeated-channel partitions; "
            "resource-aware repeated-content-first ordering"
        ),
        "label_source": "ye3t.couplings.count/plan/compile",
        "tag_count_s": int(tag_count),
        "role_bindings": role_bindings,
        "components": components,
    }
    manifest = {
        "schema": "ye3t_generated_tagged_content_manifest_v1",
        "catalogue_id": str(catalogue_id),
        "coefficient_compilation_performed": False,
        "tag_count_s": int(tag_count),
        "role_bindings": role_bindings,
        "by_tensor_order": by_order,
        "selected_fixed_contents": len(components),
        "schedule_sha256": _stable_hash(schedule),
        "identity_sha256": _stable_hash(identity),
    }
    return schedule, manifest


def _component(tensor_order, n_tuple, l_tuple):
    report = count_couplings(
        content=tuple(n_tuple),
        input_Ls=tuple(l_tuple),
        target_L=0,
        target_permutation="trivial",
        carrier="ACE_density",
        tree_schedule="balanced",
        validation_scope="counts",
        metadata={"consumer": "ye3t_methods.atomistic.ace.catalogue_selection"},
    )
    labels = tuple(
        sorted(
            (
                normalize_compact_label(value)
                for value in report.labels_for_target(0)
            ),
            key=lambda value: (
                value.rank,
                value.n_tuple,
                value.l_tuple,
                value.internal_Ls,
                value.tree_type,
                value.basis_key,
            ),
        )
    )
    return {
        "tensor_order_N": int(tensor_order),
        "n": [int(value) for value in n_tuple],
        "l": [int(value) for value in l_tuple],
        "channel_multiplicity_partition": list(
            _content_partition(n_tuple, l_tuple)
        ),
        "labels": labels,
        "descriptor_count": len(labels),
        "maximum_exact_symbolic_workspace_bytes": _content_resource_bytes(
            (n_tuple, l_tuple)
        ),
        "coupling_convention_hash": str(report.convention_hash),
    }


def _round_robin_components(request, maximum_features, target_parity=None):
    contents = {
        order: iter(
            content for content in _candidate_contents(request, order)
            if target_parity is None or
            ("odd" if sum(content[1]) % 2 else "even") == target_parity
        )
        for order in request["tensor_orders"]
    }
    active = list(request["tensor_orders"])
    components = []
    feature_count = 0
    while active and feature_count < maximum_features:
        next_active = []
        for tensor_order in active:
            try:
                n_tuple, l_tuple = next(contents[tensor_order])
            except StopIteration:
                continue
            next_active.append(tensor_order)
            component = _component(tensor_order, n_tuple, l_tuple)
            if component["descriptor_count"]:
                components.append(component)
                feature_count += component["descriptor_count"]
        active = next_active
    return components


def resolve_ordinary_scalar_catalogues(basis, catalogue_id="generated_linear_ace",
                                       target_parity=None):
    """Resolve nested ordinary scalar catalogues from user-level content caps.

    Labels and multiplicities come exclusively from ``ye3t.couplings.count``.
    No coupling coefficients are materialized by this function. The even
    parity source is compatible with the current ordinary scalar compiler.
    None preserves historical SO(3) selection, which can include odd rows
    outside that compiler's accepted scope.
    """

    if target_parity not in {None, "even"}:
        raise ValueError("target_parity must be None or 'even'.")
    request = _normalized_request(basis)
    maximum_target = max(request["targets"])
    components = _round_robin_components(request, maximum_target,
                                          target_parity=target_parity)
    rows = []
    prefix_stops = []
    for component_index, component in enumerate(components):
        for copy_index, label in enumerate(component["labels"]):
            identity = {
                "catalogue_id": str(catalogue_id),
                "tensor_order_N": component["tensor_order_N"],
                "n": component["n"],
                "l": component["l"],
                "copy_index": copy_index,
                "compact_label": label.to_dict(),
            }
            rows.append(
                {
                    "feature_id": (
                        f"ace_N{component['tensor_order_N']:02d}_"
                        f"{_stable_hash(identity)[:20]}"
                    ),
                    "compact_label": label.to_dict(),
                    "component_index": component_index,
                }
            )
        prefix_stops.append(len(rows))
    if not rows:
        raise RuntimeError("The requested catalogue contains no scalar descriptors.")
    profiles = {}
    resolution_rows = []
    for requested in request["targets"]:
        stop = min(prefix_stops, key=lambda value: (abs(value - requested), value))
        profile_id = f"{catalogue_id}_{requested}"
        profiles[profile_id] = {
            "application_schema": "ye3t_ordinary_scalar_catalogue_v2",
            "feature_ids": [row["feature_id"] for row in rows[:stop]],
            "requested_descriptor_count": requested,
            "resolved_descriptor_count": stop,
        }
        resolution_rows.append(
            {
                "profile_id": profile_id,
                "requested_descriptor_count": requested,
                "resolved_descriptor_count": stop,
                "exact_count_match": stop == requested,
            }
        )
    public_rows = [
        {
            "feature_id": row["feature_id"],
            "compact_label": row["compact_label"],
        }
        for row in rows
    ]
    source = {
        "schema": "ye3t_ordinary_scalar_catalogue_source_v1",
        "compiler": dict(request["compiler"]),
        "profiles": profiles,
        "rows": public_rows,
    }
    manifest = {
        "schema": "ye3t_generated_linear_catalogue_manifest_v1",
        "catalogue_id": str(catalogue_id),
        "preflight": linear_catalogue_preflight(basis),
        "selection_policy": (
            "round_robin_tensor_order_then_repeated_content_complexity; "
            "complete_fixed_content_multiplicities"
        ),
        "label_source": "ye3t.couplings.count",
        "coefficient_compilation_performed": False,
        "components": [
            {
                key: value
                for key, value in component.items()
                if key != "labels"
            }
            for component in components
        ],
        "profiles": resolution_rows,
        "source_sha256": _stable_hash(source),
    }
    if target_parity is not None:
        manifest["target_parity"] = target_parity
    return source, manifest


__all__ = [
    "linear_catalogue_preflight",
    "resolve_ordinary_scalar_catalogues",
    "resolve_tagged_content_schedule",
]
