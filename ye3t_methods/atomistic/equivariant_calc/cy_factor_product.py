"""Coupled Young/CG factor-product descriptor and force accumulators.

CYprime evaluates compiler-provided descriptor terms with cached atomic-base
values, analytic normalization derivatives, explicit product-rule
adjoints, and explicit force-row contractions.
"""
from ye3t_methods.atomistic._record import recordclass
import time

import torch

from ye3t.api import (
    symmetric_power_product_plan_batched_adjoint,
    symmetric_power_product_plan_contraction,
)
from ye3t.couplings import (
    blockwise_symmetric_power_product_plan,
    cy_factor_product_plan,
    symmetric_power_product_plan,
    ye3t_descriptor_adjoint_plan,
)

from .ace_eval_v2 import checked_real_scalar_projection
from .ace_symmetric_power import is_native_real_l1_even_scalar_symmetric_power
from .gradients import (
    _factorized_descriptor_plan_root_adjoint,
    edge_vectors_from_positions,
)
from .product_rule import ProductRuleResult, evaluate_explicit_product_rule


def _compiled_explicit_term_count(compiled):
    term_count = sum(int(rows.shape[0]) for rows in getattr(compiled, "channel_rows_cpu", ()))
    term_count += sum(int(rows.shape[0]) for rows in getattr(compiled, "grouped_channel_rows_cpu", ()))
    return int(term_count)


def _zero_product_rule_result(atomic_base, descriptor_count, *, dtype=None, channel_derivatives=True):
    dtype = atomic_base.dtype if dtype is None else dtype
    n_sites = int(atomic_base.shape[0])
    channel_count = int(atomic_base.shape[1])
    derivatives = None
    if channel_derivatives:
        derivatives = torch.zeros(
            (n_sites, int(descriptor_count), channel_count),
            dtype=dtype,
            device=atomic_base.device,
        )
    return ProductRuleResult(
        values=torch.zeros((n_sites, int(descriptor_count)), dtype=dtype, device=atomic_base.device),
        channel_derivatives=derivatives,
        report={
            "backend": "zero_product_rule_no_residual_terms",
            "descriptor_count": int(descriptor_count),
            "term_count": 0,
            "chunk_size": None,
            "chunk_count": 0,
            "channel_count": int(channel_count),
            "max_rank": 0,
            "value_dtype": str(dtype),
            "uses_recursive_products": False,
        },
    )


def _add_optional_tensor(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _descriptor_target_payload(descriptors, *, all_scalar):
    l_values = sorted({int(getattr(desc, "L_R", 0)) for desc in descriptors})
    m_values = sorted({int(getattr(desc, "M_R", 0)) for desc in descriptors})
    return {
        "permutation": "trivial",
        "rotation": {
            "L_R_values": tuple(l_values),
            "M_R_values": tuple(m_values),
            "all_scalar": bool(all_scalar),
        },
    }


def _complex_dtype_for(dtype):
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    return dtype


def _symmetric_coefficient_dtype(atomic_base, plan, *, imag_tol):
    if torch.is_complex(atomic_base):
        return atomic_base.dtype
    max_imag = 0.0
    for entry in plan.entries:
        for term in entry.component_terms:
            max_imag = max(max_imag, abs(complex(term["coefficient"]).imag))
    if max_imag <= float(imag_tol):
        return atomic_base.dtype
    return _complex_dtype_for(atomic_base.dtype)


def _compiled_without_accelerated_plans(compiled):
    return type(compiled)(
        channels=compiled.channels,
        channel_rows_cpu=compiled.channel_rows_cpu,
        coeffs_cpu=compiled.coeffs_cpu,
        grouped_descriptor_indices_cpu=compiled.grouped_descriptor_indices_cpu,
        grouped_channel_rows_cpu=compiled.grouped_channel_rows_cpu,
        grouped_coeffs_cpu=compiled.grouped_coeffs_cpu,
        all_scalar=compiled.all_scalar,
        factorized_plan=None,
        direct_symmetric_power_plan=None,
    )


def _symmetric_plan_entries_from_compiled(compiled):
    direct_plan = getattr(compiled, "direct_symmetric_power_plan", None)
    if direct_plan is None:
        return tuple()
    entries = []
    for entry in direct_plan.entries:
        spec = dict(entry.block_spec)
        entries.append(
            {
                "descriptor_index": int(entry.descriptor_index),
                "channel_indices": tuple(int(idx) for idx in entry.channel_indices),
                "power": int(spec["k_b"]),
                "input_L": int(spec["l"]),
                "output_L": int(spec["Lambda"]),
                "multiplicity_index": int(spec["multiplicity_index"]),
                "component_index": int(entry.component_index),
            }
        )
    return tuple(entries)


def _factorized_plan_active_indices(compiled):
    factorized = getattr(compiled, "factorized_plan", None)
    if factorized is None:
        return tuple()
    return tuple(int(idx) for idx in factorized.active_descriptor_indices)


def _factorized_plan_block_report(compiled):
    factorized = getattr(compiled, "factorized_plan", None)
    if factorized is None:
        return tuple()
    rows = []
    for group_index, group in enumerate(factorized.groups):
        descriptor_indices = tuple(int(idx) for idx in group.descriptor_indices)
        for descriptor_index, block_specs, block_indices in zip(
            descriptor_indices,
            group.block_specs,
            group.block_channel_indices,
        ):
            rank = int(sum(int(spec.get("k_b", 1)) for spec in block_specs))
            repeated_blocks = []
            largest = 0
            for spec, indices in zip(block_specs, block_indices):
                size = int(spec.get("k_b", 1))
                largest = max(largest, size)
                repeated_blocks.append(
                    {
                        "slot_group": f"S_{size}",
                        "multiplicity": int(size),
                        "n": int(spec.get("n", -1)),
                        "l": int(spec.get("l", spec.get("Lambda", -1))),
                        "Lambda": int(spec.get("Lambda", spec.get("l", 0))),
                        "kind": str(spec.get("kind", "unknown")),
                        "channel_indices": [int(idx) for idx in indices],
                    }
                )
            threshold = _symmetric_power_high_symmetry_threshold(rank)
            minimum_size = int(threshold["minimum_block_size"])
            rows.append(
                {
                    "descriptor_index": int(descriptor_index),
                    "rank": int(rank),
                    "stabilizer": " x ".join(block["slot_group"] for block in repeated_blocks),
                    "repeated_blocks": repeated_blocks,
                    "largest_repeated_block_size": int(largest),
                    "largest_repeated_block_fraction": 0.0 if rank == 0 else float(largest) / float(rank),
                    "component_term_count": int(group.schedule.term_count),
                    "lower_degree_monomial_count": 0,
                    "coefficient_entry_count": int(group.schedule.term_count),
                    "coefficient_backend": "ye3t_factorized_block_schedule",
                    "schedule_basis_count": int(group.schedule.basis_count),
                    "schedule_component_count": int(group.schedule.component_count),
                    "schedule_block_count": int(group.schedule.block_count),
                    "group_index": int(group_index),
                    "high_symmetry_requirement": dict(threshold),
                    "high_symmetry_requirement_satisfied": bool(minimum_size == 0 or largest >= minimum_size),
                }
            )
    return tuple(rows)


def _blockwise_plan_entries_from_compiled(compiled):
    factorized = getattr(compiled, "factorized_plan", None)
    if factorized is None:
        return tuple()
    entries = []
    for group in factorized.groups:
        for descriptor_index, block_specs, block_indices in zip(
            group.descriptor_indices,
            group.block_specs,
            group.block_channel_indices,
        ):
            blocks = []
            for spec, indices in zip(block_specs, block_indices):
                blocks.append(
                    {
                        "kind": str(spec.get("kind", "sym")),
                        "n": int(spec.get("n", -1)),
                        "l": int(spec.get("l", spec.get("Lambda", -1))),
                        "k_b": int(spec.get("k_b", 1)),
                        "Lambda": int(spec.get("Lambda", spec.get("l", 0))),
                        "multiplicity_index": int(spec.get("multiplicity_index", 0)),
                        "basis_key": tuple(spec.get("basis_key", ())),
                        "channel_indices": tuple(int(idx) for idx in indices),
                    }
                )
            rank = int(sum(int(block["k_b"]) for block in blocks))
            entries.append(
                {
                    "descriptor_index": int(descriptor_index),
                    "rank": int(rank),
                    "blocks": tuple(blocks),
                    "schedule_term_count": int(group.schedule.term_count),
                    "schedule_component_count": int(group.schedule.component_count),
                    "schedule_block_count": int(group.schedule.block_count),
                    "stabilizer": " x ".join(f"S_{int(block['k_b'])}" for block in blocks),
                }
            )
    return tuple(entries)


def _is_native_symmetric_entry(raw_entry):
    raw = dict(raw_entry)
    return is_native_real_l1_even_scalar_symmetric_power(
        int(raw["power"]),
        int(raw["input_L"]),
        int(raw["output_L"]),
        int(raw["multiplicity_index"]),
    )


def _split_symmetric_plan_entries(entries):
    native = []
    generic = []
    for entry in tuple(entries):
        if _is_native_symmetric_entry(entry):
            native.append(entry)
        else:
            generic.append(entry)
    return tuple(native), tuple(generic)


def _direct_symmetric_power_autograd_product(evaluator, atomic_base, compiled, *, real_if_scalar, imag_tol):
    direct_plan = getattr(compiled, "direct_symmetric_power_plan", None)
    descriptor_count = int(len(getattr(compiled, "channel_rows_cpu", ())))
    if direct_plan is None or int(direct_plan.active_descriptor_count) == 0:
        dtype = atomic_base.dtype
        return (
            torch.zeros((int(atomic_base.shape[0]), descriptor_count), dtype=dtype, device=atomic_base.device),
            torch.zeros(
                (int(atomic_base.shape[0]), descriptor_count, int(atomic_base.shape[1])),
                dtype=dtype,
                device=atomic_base.device,
            ),
            {
                "backend": "direct_symmetric_power_autograd_atomic_base",
                "active_descriptor_count": 0,
                "descriptor_count": int(descriptor_count),
                "channel_count": int(atomic_base.shape[1]),
                "uses_recursive_products": False,
            },
        )

    base = atomic_base.detach().clone().requires_grad_(True)
    values = evaluator._contract_direct_symmetric_power_descriptor_plan(
        direct_plan,
        base,
        real_if_scalar=real_if_scalar,
        imag_tol=imag_tol,
    )
    derivatives = torch.zeros(
        (int(base.shape[0]), int(values.shape[1]), int(base.shape[1])),
        dtype=base.dtype,
        device=base.device,
    )
    for descriptor_index in direct_plan.active_descriptor_indices:
        scalar = values[:, int(descriptor_index)].sum()
        grad = torch.autograd.grad(
            scalar,
            base,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        derivatives[:, int(descriptor_index), :] = grad
    return (
        values.detach(),
        derivatives.detach(),
        {
            "backend": "direct_symmetric_power_autograd_atomic_base",
            "active_descriptor_count": int(direct_plan.active_descriptor_count),
            "descriptor_count": int(values.shape[1]),
            "channel_count": int(base.shape[1]),
            "derivative_rule": "torch_autograd_of_compiled_direct_symmetric_power_forward",
            "coefficient_backend": "ace_complex_symmetric_power_entries_forward_autograd",
            "uses_recursive_products": False,
        },
    )


def _symmetric_power_high_symmetry_threshold(rank):
    rank = int(rank)
    if rank >= 16:
        return {"minimum_repeated_subgroup": "S_8 x S_8", "minimum_block_size": 8}
    if rank >= 8:
        return {"minimum_repeated_subgroup": "S_4 x S_4", "minimum_block_size": 4}
    if rank >= 6:
        return {"minimum_repeated_subgroup": "S_3 x S_3", "minimum_block_size": 3}
    return {"minimum_repeated_subgroup": "none", "minimum_block_size": 0}


def _direct_symmetric_power_block_report(compiled, compiler_plan):
    direct_plan = getattr(compiled, "direct_symmetric_power_plan", None)
    if direct_plan is None:
        return tuple()
    terms_by_descriptor = {}
    if compiler_plan is not None:
        for entry in compiler_plan.entries:
            descriptor_index = int(entry.descriptor_index)
            terms_by_descriptor[descriptor_index] = {
                "component_term_count": int(len(entry.component_terms)),
                "lower_degree_monomial_count": int(len(entry.lower_degree_exponents)),
                "coefficient_entry_count": int(entry.coefficient_entry_count),
                "coefficient_backend": str(entry.validation_report.get("coefficient_backend", "unknown")),
            }
    rows = []
    for entry in direct_plan.entries:
        spec = dict(entry.block_spec)
        rank = int(spec["k_b"])
        threshold = _symmetric_power_high_symmetry_threshold(rank)
        minimum_size = int(threshold["minimum_block_size"])
        term_payload = dict(terms_by_descriptor.get(int(entry.descriptor_index), {}))
        if not term_payload:
            term_payload = {
                "component_term_count": 0,
                "lower_degree_monomial_count": 0,
                "coefficient_entry_count": 0,
                "coefficient_backend": "compiled_direct_symmetric_power_forward_autograd",
            }
        rows.append(
            {
                "descriptor_index": int(entry.descriptor_index),
                "rank": int(rank),
                "stabilizer": f"S_{rank}",
                "largest_repeated_block_size": int(rank),
                "largest_repeated_block_fraction": 1.0,
                "repeated_blocks": [
                    {
                        "slot_group": f"S_{rank}",
                        "multiplicity": int(rank),
                        "n": int(spec["n"]),
                        "l": int(spec["l"]),
                    }
                ],
                "output_L": int(spec["Lambda"]),
                "multiplicity_index": int(spec["multiplicity_index"]),
                "component_index": int(entry.component_index),
                "channel_indices": [int(idx) for idx in entry.channel_indices],
                "high_symmetry_requirement": dict(threshold),
                "high_symmetry_requirement_satisfied": bool(minimum_size == 0 or rank >= 2 * minimum_size),
                **term_payload,
            }
        )
    return tuple(rows)


def _scheduler_policy_from_blocks(block_rows, *, has_factorized_blocks=False):
    if not block_rows:
        return {
            "name": "cyprime_fallback_no_direct_symmetric_power",
            "largest_repeated_block_fraction_cutoff": 0.5,
            "requires_compiler_owned_coefficients": True,
            "decision": "cyprime",
            "reason": "no direct repeated-block symmetric-power plan",
        }
    max_rank = max([int(row["rank"]) for row in block_rows] + [0])
    largest_fraction = max(
        [float(row.get("largest_repeated_block_fraction", 0.0)) for row in block_rows] + [0.0]
    )
    largest_block = max(
        [int(row.get("largest_repeated_block_size", 0)) for row in block_rows] + [0]
    )
    warning = ""
    recommended_scope = "all ranks"
    if max_rank >= 10 and largest_fraction < 0.5:
        warning = (
            "high-rank repeated-block descriptors with largest block below half the rank "
            "are not recommended for the symmetric-power fast path; use CYprime fallback, "
            "representative large-block sectors, or reduce the requested subgroup complexity"
        )
        recommended_scope = "rank >= 10: homogeneous or largest repeated block >= N/2"
    backends = {str(row.get("coefficient_backend", "")) for row in block_rows}
    native_backends = {"native_real_l1_even_scalar_norm_power"}
    blockwise_backends = {"ye3t_factorized_block_schedule"}
    if has_factorized_blocks and backends.issubset(blockwise_backends):
        min_rank = min([int(row["rank"]) for row in block_rows] + [0])
        if largest_fraction >= 0.5:
            return {
                "name": "prefer_blockwise_symmetric_power_for_large_mixed_repeated_blocks",
                "largest_repeated_block_fraction_cutoff": 0.5,
                "observed_largest_repeated_block_fraction": float(largest_fraction),
                "observed_largest_repeated_block_size": int(largest_block),
                "maximum_rank": int(max_rank),
                "minimum_rank": int(min_rank),
                "policy_warning": str(warning),
                "recommended_descriptor_scope": str(recommended_scope),
                "requires_compiler_owned_coefficients": True,
                "decision": "blockwise_symmetric_power",
                "reason": "compiler-owned factorized block schedule covers mixed repeated-block descriptors",
            }
        return {
            "name": "allow_blockwise_symmetric_power_probe_below_large_block_cutoff",
            "largest_repeated_block_fraction_cutoff": 0.5,
            "observed_largest_repeated_block_fraction": float(largest_fraction),
            "observed_largest_repeated_block_size": int(largest_block),
            "maximum_rank": int(max_rank),
            "minimum_rank": int(min_rank),
            "policy_warning": str(warning),
            "recommended_descriptor_scope": str(recommended_scope),
            "requires_compiler_owned_coefficients": True,
            "decision": "blockwise_symmetric_power_probe",
            "reason": (
                "compiler-owned factorized block schedule can run this explicit representative sector, "
                "but the repeated-block structure is below the conservative default large-block cutoff"
            ),
        }
    if not backends.issubset(native_backends):
        return {
            "name": "prefer_cyprime_until_generic_angular_symmetric_power_kernels_are_optimized",
            "largest_repeated_block_fraction_cutoff": 0.5,
            "observed_largest_repeated_block_fraction": float(largest_fraction),
            "observed_largest_repeated_block_size": int(largest_block),
            "maximum_rank": int(max_rank),
            "policy_warning": str(warning),
            "recommended_descriptor_scope": str(recommended_scope),
            "observed_coefficient_backends": sorted(backends),
            "requires_compiler_owned_coefficients": True,
            "decision": "cyprime_or_native_symmetric_power_only",
            "reason": "direct symmetric-power entries exist, but this angular/channel family does not yet have a proven optimized derivative kernel",
        }
    largest_fraction = 0.0
    min_rank = None
    for row in block_rows:
        rank = int(row["rank"])
        min_rank = rank if min_rank is None else min(int(min_rank), rank)
        for block in row.get("repeated_blocks", ()):
            fraction = float(block.get("multiplicity", 0)) / float(rank)
            largest_fraction = max(largest_fraction, fraction)
    if largest_fraction >= 0.5:
        return {
            "name": "prefer_native_symmetric_power_for_large_repeated_blocks",
            "largest_repeated_block_fraction_cutoff": 0.5,
            "observed_largest_repeated_block_fraction": float(largest_fraction),
            "observed_largest_repeated_block_size": int(largest_block),
            "maximum_rank": int(max_rank),
            "minimum_rank": int(min_rank),
            "policy_warning": str(warning),
            "recommended_descriptor_scope": str(recommended_scope),
            "requires_compiler_owned_coefficients": True,
            "decision": "symmetric_power",
            "reason": "largest repeated slot block is at least half the descriptor rank",
        }
    return {
        "name": "prefer_cyprime_for_small_repeated_blocks",
        "largest_repeated_block_fraction_cutoff": 0.5,
        "observed_largest_repeated_block_fraction": float(largest_fraction),
        "observed_largest_repeated_block_size": int(largest_block),
        "maximum_rank": int(max_rank),
        "minimum_rank": int(min_rank),
        "policy_warning": str(warning),
        "recommended_descriptor_scope": str(recommended_scope),
        "requires_compiler_owned_coefficients": True,
        "decision": "cyprime_or_generic_product_adjoint",
        "reason": "repeated slot blocks are below the conservative symmetric-power cutoff",
    }


def symmetric_power_schedule_decision(compiled, compiler_plan, site_basis_config=None, blockwise_compiler_plan=None):
    """Return reportable scheduler metadata for direct and blockwise symmetric-power blocks."""

    direct_rows = _direct_symmetric_power_block_report(compiled, compiler_plan)
    factorized_rows = _factorized_plan_block_report(compiled)
    block_rows = direct_rows + factorized_rows
    direct_plan = getattr(compiled, "direct_symmetric_power_plan", None)
    factorized_plan = getattr(compiled, "factorized_plan", None)
    if compiler_plan is not None and int(compiler_plan.active_descriptor_count) > 0:
        active_count = int(compiler_plan.active_descriptor_count)
        descriptor_count = int(compiler_plan.descriptor_count)
        channel_count = int(compiler_plan.channel_count)
        coefficient_source = str(compiler_plan.coefficient_source)
        label_source = str(compiler_plan.label_source)
        compiler_owner = str(compiler_plan.provenance.get("compiler_owner", "unknown"))
    elif blockwise_compiler_plan is not None and int(blockwise_compiler_plan.active_descriptor_count) > 0:
        direct_count = 0 if direct_plan is None else int(direct_plan.active_descriptor_count)
        active_count = int(direct_count + int(blockwise_compiler_plan.active_descriptor_count))
        descriptor_count = int(blockwise_compiler_plan.descriptor_count)
        channel_count = int(blockwise_compiler_plan.channel_count)
        coefficient_source = str(blockwise_compiler_plan.coefficient_source)
        label_source = str(blockwise_compiler_plan.label_source)
        compiler_owner = str(blockwise_compiler_plan.provenance.get("compiler_owner", "unknown"))
    else:
        direct_count = 0 if direct_plan is None else int(direct_plan.active_descriptor_count)
        factorized_count = 0 if factorized_plan is None else int(factorized_plan.active_descriptor_count)
        active_count = int(direct_count + factorized_count)
        descriptor_count = int(len(getattr(compiled, "channel_rows_cpu", ())))
        channel_count = int(len(getattr(compiled, "channels", ())))
        coefficient_source = (
            "ye3t compiled factorized block schedule"
            if factorized_count > 0 and direct_count == 0
            else "compiled direct symmetric-power forward"
        )
        label_source = "ye3t.couplings.count compact symmetric labels"
        compiler_owner = "ye3t"
    residual_count = int(descriptor_count - active_count)
    normalization = "unknown"
    if site_basis_config is not None:
        normalization = str(getattr(site_basis_config, "atomic_base_normalization", "unknown"))
    ranks = sorted({int(row["rank"]) for row in block_rows})
    high_symmetry_ok = all(bool(row["high_symmetry_requirement_satisfied"]) for row in block_rows)
    term_count = int(sum(int(row.get("component_term_count", 0)) for row in block_rows))
    lower_count = int(sum(int(row.get("lower_degree_monomial_count", 0)) for row in block_rows))
    explicit_terms = int(_compiled_explicit_term_count(compiled))
    factorized_terms = max(1, int(term_count))
    scheduler_policy = _scheduler_policy_from_blocks(
        block_rows,
        has_factorized_blocks=bool(factorized_rows),
    )
    selected = "symmetric_power" if active_count > 0 else "cyprime"
    if factorized_rows and direct_plan is None:
        selected = "blockwise_symmetric_power"
    if scheduler_policy["name"] == "prefer_cyprime_until_generic_angular_symmetric_power_kernels_are_optimized":
        selected = "direct_symmetric_power_autograd_diagnostic"
    if active_count > 0 and residual_count > 0:
        if selected == "direct_symmetric_power_autograd_diagnostic":
            selected = "direct_symmetric_power_autograd_diagnostic_with_cyprime_residual"
        else:
            selected = "symmetric_power_with_cyprime_residual"
    return {
        "selected_evaluator": selected,
        "eligible_evaluators": ["symmetric_power", "cyprime", "generic_product_adjoint"],
        "decision_reason": (
            "all active accelerated descriptors are direct repeated-block symmetric powers"
            if active_count > 0
            else "no direct repeated-block symmetric-power descriptors were compiled"
        ),
        "descriptor_count": int(descriptor_count),
        "active_symmetric_descriptor_count": int(active_count),
        "residual_explicit_descriptor_count": int(residual_count),
        "descriptor_ranks": [int(rank) for rank in ranks],
        "active_repeated_block_structure": [dict(row) for row in block_rows],
        "direct_symmetric_block_count": int(len(direct_rows)),
        "factorized_blockwise_descriptor_count": int(len(factorized_rows)),
        "scheduler_policy": dict(scheduler_policy),
        "normalization": str(normalization),
        "cost_model": {
            "component_term_count": int(term_count),
            "lower_degree_monomial_count": int(lower_count),
            "explicit_term_count": int(explicit_terms),
            "factorized_term_count": int(term_count),
            "explicit_to_factorized_term_ratio": float(explicit_terms) / float(factorized_terms),
            "cost_expression": "explicit_product_terms / factorized_component_terms",
            "decision_use": (
                "large ratios favor symmetric/blockwise plans when high-symmetry policy is satisfied; "
                "small-block high-rank sectors remain CYprime/generic-product candidates"
            ),
            "channel_count": int(channel_count),
        },
        "high_symmetry_policy": {
            "passed": bool(high_symmetry_ok),
            "rank6_minimum": "S_3 x S_3",
            "rank8_minimum": "S_4 x S_4",
            "rank16_minimum": "S_8 x S_8",
            "full_symmetric_groups_are_allowed": True,
        },
        "coefficient_source": str(coefficient_source),
        "label_source": str(label_source),
        "compiler_owner": str(compiler_owner),
    }


def _evaluate_factorized_blockwise_product(evaluator, atomic_base, compiled, *, chunk_size, real_if_scalar, imag_tol):
    factorized_plan = getattr(compiled, "factorized_plan", None)
    descriptor_count = int(len(getattr(compiled, "channel_rows_cpu", ())))
    n_sites = int(atomic_base.shape[0])
    channel_count = int(atomic_base.shape[1])
    if factorized_plan is None or int(factorized_plan.active_descriptor_count) == 0:
        return (
            torch.zeros((n_sites, descriptor_count), dtype=atomic_base.dtype, device=atomic_base.device),
            torch.zeros((n_sites, descriptor_count, channel_count), dtype=atomic_base.dtype, device=atomic_base.device),
            {
                "backend": "ye3t_factorized_block_schedule_root_adjoint",
                "active_descriptor_count": 0,
                "descriptor_count": int(descriptor_count),
                "channel_count": int(channel_count),
                "uses_recursive_products": False,
            },
        )

    active_indices = tuple(int(idx) for idx in factorized_plan.active_descriptor_indices)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = max(1, min(len(active_indices), 16))
    chunk_size = max(1, int(chunk_size))
    values = None
    derivatives = None
    chunks = 0
    start = 0
    while start < len(active_indices):
        selected = active_indices[start:start + chunk_size]
        output_adjoint = torch.zeros(
            (len(selected), n_sites, descriptor_count),
            dtype=atomic_base.dtype,
            device=atomic_base.device,
        )
        for local_index, descriptor_index in enumerate(selected):
            output_adjoint[int(local_index), :, int(descriptor_index)] = 1
        chunk_values, root = _factorized_descriptor_plan_root_adjoint(
            evaluator,
            factorized_plan,
            atomic_base,
            output_adjoint,
        )
        if values is None:
            values = chunk_values
        if derivatives is None:
            derivatives = torch.zeros(
                (n_sites, descriptor_count, channel_count),
                dtype=root.dtype,
                device=atomic_base.device,
            )
        descriptor_tensor = torch.tensor(selected, dtype=torch.long, device=atomic_base.device)
        derivatives.index_copy_(1, descriptor_tensor, root.permute(1, 0, 2).contiguous())
        chunks += 1
        start += chunk_size
    if values is None:
        values = torch.zeros((n_sites, descriptor_count), dtype=atomic_base.dtype, device=atomic_base.device)
    if derivatives is None:
        derivatives = torch.zeros(
            (n_sites, descriptor_count, channel_count),
            dtype=values.dtype,
            device=atomic_base.device,
        )
    if real_if_scalar and bool(factorized_plan.all_scalar):
        values = checked_real_scalar_projection(values, imag_tol=imag_tol, context="factorized blockwise symmetric-power CYPrime path")
        if torch.is_complex(derivatives):
            max_imag = torch.max(torch.abs(derivatives.imag)) if derivatives.numel() else torch.zeros((), dtype=derivatives.real.dtype, device=derivatives.device)
            if max_imag <= torch.as_tensor(float(imag_tol), dtype=derivatives.real.dtype, device=derivatives.device):
                derivatives = derivatives.real
    return (
        values.detach(),
        derivatives.detach(),
        {
            "backend": "ye3t_factorized_block_schedule_root_adjoint",
            "active_descriptor_count": int(factorized_plan.active_descriptor_count),
            "descriptor_count": int(descriptor_count),
            "channel_count": int(channel_count),
            "derivative_rule": "compiled_factorized_block_forward_reverse_product_rule",
            "coefficient_backend": "ye3t_factorized_block_schedule",
            "chunk_size": int(chunk_size),
            "chunk_count": int(chunks),
            "uses_recursive_products": False,
        },
    )


def _factorized_blockwise_root_chunk(evaluator, factorized_plan, atomic_base, descriptor_count, selected):
    n_sites = int(atomic_base.shape[0])
    output_adjoint = torch.zeros(
        (len(selected), n_sites, int(descriptor_count)),
        dtype=atomic_base.dtype,
        device=atomic_base.device,
    )
    for local_index, descriptor_index in enumerate(selected):
        output_adjoint[int(local_index), :, int(descriptor_index)] = 1
    return _factorized_descriptor_plan_root_adjoint(
        evaluator,
        factorized_plan,
        atomic_base,
        output_adjoint,
    )


def _evaluate_factorized_blockwise_values(evaluator, atomic_base, compiled, *, chunk_size, imag_tol):
    factorized_plan = getattr(compiled, "factorized_plan", None)
    descriptor_count = int(len(getattr(compiled, "channel_rows_cpu", ())))
    n_sites = int(atomic_base.shape[0])
    channel_count = int(atomic_base.shape[1])
    if factorized_plan is None or int(factorized_plan.active_descriptor_count) == 0:
        return (
            torch.zeros((n_sites, descriptor_count), dtype=atomic_base.dtype, device=atomic_base.device),
            {
                "backend": "ye3t_factorized_block_schedule_root_adjoint_streamed_values",
                "active_descriptor_count": 0,
                "descriptor_count": int(descriptor_count),
                "channel_count": int(channel_count),
                "uses_recursive_products": False,
            },
        )
    active_indices = tuple(int(idx) for idx in factorized_plan.active_descriptor_indices)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = max(1, min(len(active_indices), 16))
    selected = active_indices[:max(1, int(chunk_size))]
    values, _ = _factorized_blockwise_root_chunk(
        evaluator,
        factorized_plan,
        atomic_base,
        descriptor_count,
        selected,
    )
    return (
        checked_real_scalar_projection(
            values,
            imag_tol=imag_tol,
            context="factorized blockwise symmetric-power streamed values",
        )
        if bool(factorized_plan.all_scalar)
        else values,
        {
            "backend": "ye3t_factorized_block_schedule_root_adjoint_streamed_values",
            "active_descriptor_count": int(factorized_plan.active_descriptor_count),
            "descriptor_count": int(descriptor_count),
            "channel_count": int(channel_count),
            "derivative_rule": "compiled_factorized_block_forward_reverse_product_rule",
            "coefficient_backend": "ye3t_factorized_block_schedule",
            "uses_recursive_products": False,
        },
    )


def _factorized_blockwise_force_rows_streamed(evaluator, factorized_plan, atomic_base, vjp_record, descriptor_count, *, chunk_size):
    active_indices = tuple(int(idx) for idx in factorized_plan.active_descriptor_indices)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = max(1, min(len(active_indices), 16))
    chunk_size = max(1, int(chunk_size))
    rows = []
    chunks = 0
    start = 0
    while start < len(active_indices):
        selected = active_indices[start:start + chunk_size]
        _, root = _factorized_blockwise_root_chunk(
            evaluator,
            factorized_plan,
            atomic_base,
            descriptor_count,
            selected,
        )
        position_grad = evaluator.site_basis.position_vjp_from_record_batched(vjp_record, root)
        rows.append(position_grad.reshape(len(selected), -1))
        chunks += 1
        start += chunk_size
    if rows:
        return torch.cat(rows, dim=0), {"chunk_size": int(chunk_size), "chunk_count": int(chunks)}
    return (
        torch.zeros((0, int(vjp_record.final_atomic_base.shape[0]) * 3), dtype=atomic_base.dtype, device=atomic_base.device),
        {"chunk_size": int(chunk_size), "chunk_count": 0},
    )


def _evaluate_symmetric_power_product_plan(
    atomic_base,
    plan,
    *,
    chunk_size=None,
    imag_tol=1.0e-12,
):
    if atomic_base.ndim != 2:
        raise ValueError("atomic_base must have shape [n_sites, n_channels]")
    if int(atomic_base.shape[1]) != int(plan.channel_count):
        raise ValueError("symmetric-power plan channel_count does not match atomic_base")
    out_dtype = _symmetric_coefficient_dtype(atomic_base, plan, imag_tol=imag_tol)
    base = atomic_base.to(dtype=out_dtype)
    n_sites = int(base.shape[0])
    descriptor_count = int(plan.descriptor_count)
    channel_count = int(plan.channel_count)
    values = symmetric_power_product_plan_contraction(
        base,
        plan,
        backend="auto",
        imag_tol=imag_tol,
    )
    derivatives = torch.zeros(
        (n_sites, descriptor_count, channel_count),
        dtype=out_dtype,
        device=base.device,
    )
    active_indices = tuple(
        sorted(
            set(
                int(value)
                for value in plan.active_descriptor_indices
            )
        )
    )
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = max(1, min(len(active_indices), 16))
    chunk_size = max(1, int(chunk_size))
    chunks = 0
    for offset in range(0, len(active_indices), chunk_size):
        selected = active_indices[offset:offset + chunk_size]
        selected_tensor = torch.tensor(
            selected,
            dtype=torch.long,
            device=base.device,
        )
        seed_rows = torch.zeros(
            (len(selected), descriptor_count),
            dtype=out_dtype,
            device=base.device,
        )
        seed_rows.scatter_(
            1,
            selected_tensor.reshape(-1, 1),
            1,
        )
        output_adjoint = seed_rows.unsqueeze(1).expand(
            len(selected),
            n_sites,
            descriptor_count,
        )
        root = symmetric_power_product_plan_batched_adjoint(
            output_adjoint,
            base,
            plan,
            backend="auto",
            imag_tol=imag_tol,
        )
        if torch.is_complex(root):
            root = root.conj()
        derivatives.index_copy_(
            1,
            selected_tensor,
            root.permute(1, 0, 2),
        )
        chunks += 1
    term_count = sum(
        int(len(entry.component_terms))
        for entry in plan.entries
    )
    lower_degree_count = sum(
        int(len(entry.lower_degree_exponents))
        for entry in plan.entries
    )
    return values, derivatives, {
        "backend": "ye3t_grouped_symmetric_power_product_plan",
        "runtime_table_layout": (
            "measured_auto_grouped_or_shared_unique_monomial"
        ),
        "root_adjoint_backend": (
            "ye3t_shared_monomial_batched_adjoint"
        ),
        "batched_root_input_expansion": False,
        "descriptor_count": int(plan.descriptor_count),
        "active_descriptor_count": int(plan.active_descriptor_count),
        "channel_count": int(plan.channel_count),
        "term_count": int(term_count),
        "lower_degree_monomial_count": int(lower_degree_count),
        "adjoint_chunk_size": int(chunk_size),
        "adjoint_chunk_count": int(chunks),
        "uses_recursive_products": False,
    }


def _charge_jacobian_from_channel_derivatives(site_basis, vjp_record, channel_derivatives, *, chunk_size):
    descriptor_count = int(channel_derivatives.shape[1])
    if descriptor_count == 0:
        return torch.zeros(
            (0, int(vjp_record.n_atoms)),
            dtype=vjp_record.final_atomic_base.real.dtype,
            device=vjp_record.final_atomic_base.device,
        )
    rows = []
    for offset in range(0, descriptor_count, chunk_size):
        stop = min(descriptor_count, offset + chunk_size)
        for descriptor_index in range(offset, stop):
            rows.append(site_basis.charge_vjp_from_record(vjp_record, channel_derivatives[:, descriptor_index, :]))
    return torch.stack(rows, dim=0) if rows else torch.zeros(
        (0, int(vjp_record.n_atoms)),
        dtype=vjp_record.final_atomic_base.real.dtype,
        device=vjp_record.final_atomic_base.device,
    )


def _stress_jacobian_from_channel_derivatives(site_basis, vjp_record, channel_derivatives, *, chunk_size):
    descriptor_count = int(channel_derivatives.shape[1])
    if descriptor_count == 0:
        return torch.zeros(
            (0, 3, 3),
            dtype=vjp_record.final_atomic_base.real.dtype,
            device=vjp_record.final_atomic_base.device,
        )
    rows = []
    for offset in range(0, descriptor_count, chunk_size):
        stop = min(descriptor_count, offset + chunk_size)
        channel_adjoint = channel_derivatives[:, offset:stop, :].permute(1, 0, 2).contiguous()
        rows.append(site_basis.strain_vjp_from_record_batched(vjp_record, channel_adjoint))
    return torch.cat(rows, dim=0) if rows else torch.zeros(
        (0, 3, 3),
        dtype=vjp_record.final_atomic_base.real.dtype,
        device=vjp_record.final_atomic_base.device,
    )


@recordclass(('descriptor_values', 'channel_derivatives', 'force_jacobian', 'charge_jacobian', 'stress_jacobian', 'normalized_atomic_base', 'compiler_plan', 'descriptor_adjoint_plan', 'report'), frozen = True)
class CYFactorProductEvaluation:
    """Values, root-channel derivatives, compiler plan, force, charge, and strain rows."""


class CYFactorProductEvaluator:
    """Evaluate descriptors and force rows through ``dB/dA_norm`` factor products."""

    def __init__(self, evaluator, *, imag_tol=1.0e-10):
        self.evaluator = evaluator
        self.imag_tol = float(imag_tol)

    def evaluate(
        self,
        *,
        positions,
        cell,
        edge_index,
        atom_types,
        descriptors,
        shifts=None,
        charges=None,
        aux_tensor_basis=None,
        real_if_scalar=True,
        chunk_size=None,
        materialize_force_jacobian=True,
        materialize_charge_jacobian=True,
        materialize_stress_jacobian=False,
    ):
        profile = {}
        total_start = time.perf_counter()

        start = time.perf_counter()
        x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
        profile["edge_vectors_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        compiled = self.evaluator._compile_descriptors(descriptors)
        profile["compile_lookup_seconds"] = float(time.perf_counter() - start)
        if any(getattr(plan, "active_descriptor_indices", ()) for plan in (
                getattr(compiled, "factorized_plan", None),
                getattr(compiled, "direct_symmetric_power_plan", None)) if plan is not None):
            return self.evaluate_symmetric_power(
                positions=positions, cell=cell, edge_index=edge_index,
                atom_types=atom_types, descriptors=descriptors, shifts=shifts,
                charges=charges, aux_tensor_basis=aux_tensor_basis,
                real_if_scalar=real_if_scalar, chunk_size=chunk_size,
                materialize_force_jacobian=materialize_force_jacobian,
                materialize_charge_jacobian=materialize_charge_jacobian,
                materialize_stress_jacobian=materialize_stress_jacobian,
            )

        start = time.perf_counter()
        _, atomic_base, vjp_record = self.evaluator.site_basis.compute_channels_with_vjp_record(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        profile["atomic_base_seconds"] = float(time.perf_counter() - start)

        descriptor_count = int(len(descriptors))
        explicit_term_count = _compiled_explicit_term_count(compiled)
        compiler_plan = cy_factor_product_plan(
            descriptor_count=descriptor_count,
            channel_count=int(atomic_base.shape[1]),
            explicit_term_count=explicit_term_count,
            carrier="ACE_density",
            target=_descriptor_target_payload(descriptors, all_scalar=bool(compiled.all_scalar)),
            factor_basis="A_normalized",
            young_coupling={
                "sector": "globally_trivial",
                "carrier_restriction": "ordinary ACE density",
            },
            rotation_coupling={
                "backend": "compiled_CG_descriptor_rows",
                "all_scalar": bool(compiled.all_scalar),
            },
            coefficient_source="ye3t.couplings.compile compact descriptor payload",
            validation_report={
                "explicit_descriptor_terms_available": explicit_term_count > 0 or descriptor_count == 0,
                "compiled_descriptor_cache": "ACECovariantEvaluator._compile_descriptors",
            },
        )
        descriptor_adjoint_plan = ye3t_descriptor_adjoint_plan(compiler_plan)

        start = time.perf_counter()
        product = evaluate_explicit_product_rule(
            atomic_base,
            compiled,
            descriptor_count=descriptor_count,
            chunk_size=chunk_size,
            imag_tol=self.imag_tol,
        )
        profile["product_and_adjoint_seconds"] = float(time.perf_counter() - start)

        values = product.values
        if real_if_scalar and compiled.all_scalar:
            values = checked_real_scalar_projection(values, imag_tol=self.imag_tol, context="CY factor-product path")

        if chunk_size is None or int(chunk_size) <= 0:
            chunk_size = max(1, descriptor_count)
        chunk_size = max(1, int(chunk_size))

        jacobian_chunks = []
        force_jacobian = None
        if materialize_force_jacobian:
            start = time.perf_counter()
            for offset in range(0, descriptor_count, chunk_size):
                stop = min(descriptor_count, offset + chunk_size)
                channel_adjoint = product.channel_derivatives[:, offset:stop, :].permute(1, 0, 2).contiguous()
                position_grad = self.evaluator.site_basis.position_vjp_from_record_batched(vjp_record, channel_adjoint)
                jacobian_chunks.append(position_grad.reshape(stop - offset, -1))
            force_jacobian = (
                torch.cat(jacobian_chunks, dim=0)
                if jacobian_chunks
                else torch.zeros((0, int(positions.numel())), dtype=positions.dtype, device=positions.device)
            )
            profile["force_row_seconds"] = float(time.perf_counter() - start)
        else:
            profile["force_row_seconds"] = 0.0

        charge_jacobian = None
        if materialize_charge_jacobian:
            start = time.perf_counter()
            charge_jacobian = _charge_jacobian_from_channel_derivatives(
                self.evaluator.site_basis,
                vjp_record,
                product.channel_derivatives,
                chunk_size=chunk_size,
            )
            profile["charge_row_seconds"] = float(time.perf_counter() - start)
        else:
            profile["charge_row_seconds"] = 0.0

        stress_jacobian = None
        if materialize_stress_jacobian:
            start = time.perf_counter()
            stress_jacobian = _stress_jacobian_from_channel_derivatives(
                self.evaluator.site_basis,
                vjp_record,
                product.channel_derivatives,
                chunk_size=chunk_size,
            )
            profile["stress_row_seconds"] = float(time.perf_counter() - start)
        else:
            profile["stress_row_seconds"] = 0.0
        profile["total_seconds"] = float(time.perf_counter() - total_start)

        cache_report = self.evaluator.site_basis.last_atomic_base_cache_report()
        report = {
            "backend": "cy_factor_product_explicit_product_rule",
            "algorithmic_reference": "ACE C-tilde-style force-row contraction",
            "descriptor_count": int(descriptor_count),
            "channel_count": int(atomic_base.shape[1]),
            "explicit_term_count": int(explicit_term_count),
            "chunk_size": int(chunk_size),
            "chunk_count": int(len(jacobian_chunks)),
            "atomic_base_cache": dict(cache_report),
            "compiler_plan": compiler_plan.to_dict(),
            "descriptor_adjoint_plan": descriptor_adjoint_plan.to_dict(),
            "product_rule": dict(product.report),
            "timings": dict(profile),
            "uses_recursive_products": False,
            "materializes_force_jacobian": bool(materialize_force_jacobian),
            "materializes_charge_jacobian": bool(materialize_charge_jacobian),
            "materializes_stress_jacobian": bool(materialize_stress_jacobian),
            "stress_target": {
                "derivative_mode": "strain",
                "row_formula": "dphi_dr_alpha_times_r_beta",
                "volume_normalization": "not_applied",
                "ase_voigt_sign": "not_applied",
            },
        }
        return CYFactorProductEvaluation(
            descriptor_values=values,
            channel_derivatives=product.channel_derivatives,
            force_jacobian=force_jacobian,
            charge_jacobian=charge_jacobian,
            stress_jacobian=stress_jacobian,
            normalized_atomic_base=atomic_base,
            compiler_plan=compiler_plan,
            descriptor_adjoint_plan=descriptor_adjoint_plan,
            report=report,
        )

    def evaluate_symmetric_power(
        self,
        *,
        positions,
        cell,
        edge_index,
        atom_types,
        descriptors,
        shifts=None,
        charges=None,
        aux_tensor_basis=None,
        real_if_scalar=True,
        chunk_size=None,
        materialize_force_jacobian=True,
        materialize_charge_jacobian=True,
        materialize_stress_jacobian=False,
    ):
        profile = {}
        total_start = time.perf_counter()

        start = time.perf_counter()
        x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
        profile["edge_vectors_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        compiled = self.evaluator._compile_descriptors(descriptors)
        profile["compile_lookup_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        _, atomic_base, vjp_record = self.evaluator.site_basis.compute_channels_with_vjp_record(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        profile["atomic_base_seconds"] = float(time.perf_counter() - start)

        descriptor_count = int(len(descriptors))
        entries = _symmetric_plan_entries_from_compiled(compiled)
        factorized = getattr(compiled, "factorized_plan", None)
        factorized_active = set() if factorized is None else set(int(idx) for idx in factorized.active_descriptor_indices)
        direct_active = set(int(entry["descriptor_index"]) for entry in entries)
        if not entries and not factorized_active:
            raise ValueError("No direct or factorized symmetric-power descriptor entries were compiled for this descriptor batch.")
        active_accelerated = direct_active | factorized_active
        angular_convention = self.evaluator.site_basis.sph.convention_metadata()
        basis_convention = (
            "real_tesseral"
            if str(self.evaluator.site_basis.cfg.spherical_backend) == "real"
            else "complex_magnetic"
        )
        native_entries, generic_entries = _split_symmetric_plan_entries(entries)
        use_direct_autograd = bool(generic_entries)
        compiler_plan = None
        if native_entries and not use_direct_autograd:
            compiler_plan = symmetric_power_product_plan(
                native_entries,
                descriptor_count=descriptor_count,
                channel_count=int(atomic_base.shape[1]),
                carrier="ACE_density",
                target=_descriptor_target_payload(descriptors, all_scalar=bool(compiled.all_scalar)),
                factor_basis="A_normalized",
                normalization_convention=str(self.evaluator.site_basis.cfg.atomic_base_normalization),
                basis_convention=basis_convention,
                label_source="ye3t.couplings.count compact symmetric labels",
                validation_report={
                    "compiled_descriptor_cache": "ACECovariantEvaluator._compile_descriptors",
                    "residual_explicit_descriptor_count": int(descriptor_count - len(active_accelerated)),
                    "angular_convention_hash": str(angular_convention["convention_hash"]),
                    "generic_angular_direct_symmetric_entry_count": 0,
                },
                provenance={
                    "descriptor_entry_source": "ACECovariantEvaluator.direct_symmetric_power_plan",
                    "angular_convention": angular_convention,
                },
            )
        compiler_plan_report = (
            compiler_plan.to_dict()
            if compiler_plan is not None
            else {
                "entries": [],
                "descriptor_count": int(descriptor_count),
                "channel_count": int(atomic_base.shape[1]),
                "active_descriptor_indices": sorted(int(idx) for idx in active_accelerated),
                "carrier": "ACE_density",
                "target": _descriptor_target_payload(descriptors, all_scalar=bool(compiled.all_scalar)),
                "factor_basis": "A_normalized",
                "normalization_convention": str(self.evaluator.site_basis.cfg.atomic_base_normalization),
                "coefficient_source": (
                    "ye3t compiled factorized block schedule"
                    if factorized_active and not direct_active
                    else "compiled direct symmetric-power forward"
                ),
                "label_source": "ye3t.couplings.count compact symmetric labels",
                "backend": (
                    "factorized_blockwise_symmetric_power_forward_reverse"
                    if factorized_active and not direct_active
                    else "direct_symmetric_power_forward_autograd_adjoint"
                ),
                "convention_hash": str(angular_convention["convention_hash"]),
                "validation_report": {
                    "passed": True,
                    "scope": (
                        "factorized_blockwise_symmetric_power_derivative"
                        if factorized_active and not direct_active
                        else "generic_angular_direct_symmetric_power_derivative_fallback"
                    ),
                    "valid_labels_from": "ye3t.couplings.count compact symmetric labels",
                    "supports_descriptor_adjoint": True,
                    "native_entry_count": int(len(native_entries)),
                    "generic_angular_direct_symmetric_entry_count": int(len(generic_entries)),
                    "factorized_blockwise_descriptor_count": int(len(factorized_active)),
                    "derivative_rule": (
                        "compiled_factorized_block_forward_reverse_product_rule"
                        if factorized_active and not direct_active
                        else "torch_autograd_of_compiled_direct_symmetric_power_forward"
                    ),
                    "angular_convention_hash": str(angular_convention["convention_hash"]),
                },
                "provenance": {
                    "api": (
                        "ACECovariantEvaluator.factorized_plan"
                        if factorized_active and not direct_active
                        else "ACECovariantEvaluator.direct_symmetric_power_plan"
                    ),
                    "compiler_owner": "ye3t",
                    "descriptor_entry_source": (
                        "ACECovariantEvaluator.factorized_plan"
                        if factorized_active and not direct_active
                        else "ACECovariantEvaluator.direct_symmetric_power_plan"
                    ),
                    "angular_convention": angular_convention,
                },
            }
        )
        blockwise_compiler_plan = None
        if factorized_active:
            blockwise_compiler_plan = blockwise_symmetric_power_product_plan(
                _blockwise_plan_entries_from_compiled(compiled),
                descriptor_count=descriptor_count,
                channel_count=int(atomic_base.shape[1]),
                carrier="ACE_density",
                target=_descriptor_target_payload(descriptors, all_scalar=bool(compiled.all_scalar)),
                factor_basis="A_normalized",
                normalization_convention=str(self.evaluator.site_basis.cfg.atomic_base_normalization),
                label_source="ye3t.couplings.blockwise_symmetric_power_labels",
                validation_report={
                    "compiled_descriptor_cache": "ACECovariantEvaluator._compile_descriptors",
                    "residual_explicit_descriptor_count": int(descriptor_count - len(active_accelerated)),
                    "angular_convention_hash": str(angular_convention["convention_hash"]),
                },
                provenance={
                    "descriptor_entry_source": "ACECovariantEvaluator.factorized_plan",
                    "angular_convention": angular_convention,
                },
            )
        blockwise_compiler_plan_report = (
            None if blockwise_compiler_plan is None else blockwise_compiler_plan.to_dict()
        )
        descriptor_adjoint_plan = ye3t_descriptor_adjoint_plan(
            cy_factor_product_plan(
                descriptor_count=descriptor_count,
                channel_count=int(atomic_base.shape[1]),
                explicit_term_count=_compiled_explicit_term_count(compiled),
                carrier="ACE_density",
                target=_descriptor_target_payload(descriptors, all_scalar=bool(compiled.all_scalar)),
                factor_basis="A_normalized",
                backend="symmetric_power_cy_factor_product",
                rotation_coupling={
                    "angular_convention": angular_convention,
                    "cg_convention": "condon_shortley_orthonormal_spherical_harmonics",
                },
                validation_report={
                    "symmetric_power_plan": compiler_plan_report,
                    "blockwise_symmetric_power_plan": blockwise_compiler_plan_report,
                    "supports_descriptor_adjoint": True,
                },
                provenance={
                    "compiler_owner": "ye3t",
                    "symmetric_power_plan_api": "ye3t.couplings.symmetric_power_product_plan",
                    "blockwise_symmetric_power_plan_api": "ye3t.couplings.blockwise_symmetric_power_product_plan",
                },
            )
        )
        scheduler = symmetric_power_schedule_decision(
            compiled,
            compiler_plan,
            site_basis_config=self.evaluator.site_basis.cfg,
            blockwise_compiler_plan=blockwise_compiler_plan,
        )
        fully_factorized = bool(
            factorized is not None
            and int(len(factorized_active)) == int(descriptor_count)
            and not direct_active
        )
        stream_factorized_force_rows = bool(
            fully_factorized
            and materialize_force_jacobian
            and not materialize_charge_jacobian
            and not materialize_stress_jacobian
        )

        start = time.perf_counter()
        residual_product = (
            _zero_product_rule_result(
                atomic_base,
                descriptor_count,
                channel_derivatives=not stream_factorized_force_rows,
            )
            if int(len(active_accelerated)) == int(descriptor_count)
            else evaluate_explicit_product_rule(
                atomic_base,
                _compiled_without_accelerated_plans(compiled),
                descriptor_count=descriptor_count,
                chunk_size=chunk_size,
                imag_tol=self.imag_tol,
            )
        )
        if not entries:
            sym_values = torch.zeros(
                (int(atomic_base.shape[0]), descriptor_count),
                dtype=atomic_base.dtype,
                device=atomic_base.device,
            )
            sym_derivatives = None if stream_factorized_force_rows else torch.zeros(
                (int(atomic_base.shape[0]), descriptor_count, int(atomic_base.shape[1])),
                dtype=atomic_base.dtype,
                device=atomic_base.device,
            )
            sym_report = {
                "backend": "no_direct_symmetric_power_entries",
                "active_descriptor_count": 0,
                "descriptor_count": int(descriptor_count),
                "channel_count": int(atomic_base.shape[1]),
                "uses_recursive_products": False,
            }
        elif use_direct_autograd:
            sym_values, sym_derivatives, sym_report = _direct_symmetric_power_autograd_product(
                self.evaluator,
                atomic_base,
                compiled,
                real_if_scalar=real_if_scalar,
                imag_tol=self.imag_tol,
            )
        else:
            sym_values, sym_derivatives, sym_report = _evaluate_symmetric_power_product_plan(
                atomic_base,
                compiler_plan,
                chunk_size=chunk_size,
                imag_tol=self.imag_tol,
            )
        if stream_factorized_force_rows:
            factorized_values, factorized_report = _evaluate_factorized_blockwise_values(
                self.evaluator,
                atomic_base,
                compiled,
                chunk_size=chunk_size,
                imag_tol=self.imag_tol,
            )
            factorized_derivatives = None
        else:
            factorized_values, factorized_derivatives, factorized_report = _evaluate_factorized_blockwise_product(
                self.evaluator,
                atomic_base,
                compiled,
                chunk_size=chunk_size,
                real_if_scalar=False,
                imag_tol=self.imag_tol,
            )
        values = residual_product.values + sym_values + factorized_values
        channel_derivatives = _add_optional_tensor(
            _add_optional_tensor(residual_product.channel_derivatives, sym_derivatives),
            factorized_derivatives,
        )
        profile["symmetric_product_and_adjoint_seconds"] = float(time.perf_counter() - start)

        if real_if_scalar and compiled.all_scalar:
            values = checked_real_scalar_projection(values, imag_tol=self.imag_tol, context="symmetric-power CYPrime path")

        if chunk_size is None or int(chunk_size) <= 0:
            chunk_size = max(1, descriptor_count)
        chunk_size = max(1, int(chunk_size))

        jacobian_chunks = []
        force_jacobian = None
        if materialize_force_jacobian:
            start = time.perf_counter()
            if stream_factorized_force_rows:
                force_jacobian, stream_report = _factorized_blockwise_force_rows_streamed(
                    self.evaluator,
                    factorized,
                    atomic_base,
                    vjp_record,
                    descriptor_count,
                    chunk_size=chunk_size,
                )
                jacobian_chunks = [force_jacobian] if int(force_jacobian.shape[0]) else []
                factorized_report = {**dict(factorized_report), "force_row_streaming": dict(stream_report)}
            else:
                if channel_derivatives is None:
                    raise RuntimeError("channel derivatives are required for non-streamed force-row materialization.")
                for offset in range(0, descriptor_count, chunk_size):
                    stop = min(descriptor_count, offset + chunk_size)
                    channel_adjoint = channel_derivatives[:, offset:stop, :].permute(1, 0, 2).contiguous()
                    position_grad = self.evaluator.site_basis.position_vjp_from_record_batched(vjp_record, channel_adjoint)
                    jacobian_chunks.append(position_grad.reshape(stop - offset, -1))
                force_jacobian = (
                    torch.cat(jacobian_chunks, dim=0)
                    if jacobian_chunks
                    else torch.zeros((0, int(positions.numel())), dtype=positions.dtype, device=positions.device)
                )
            profile["force_row_seconds"] = float(time.perf_counter() - start)
        else:
            profile["force_row_seconds"] = 0.0

        charge_jacobian = None
        if materialize_charge_jacobian:
            if channel_derivatives is None:
                raise RuntimeError("channel derivatives are required for charge-row materialization.")
            start = time.perf_counter()
            charge_jacobian = _charge_jacobian_from_channel_derivatives(
                self.evaluator.site_basis,
                vjp_record,
                channel_derivatives,
                chunk_size=chunk_size,
            )
            profile["charge_row_seconds"] = float(time.perf_counter() - start)
        else:
            profile["charge_row_seconds"] = 0.0

        stress_jacobian = None
        if materialize_stress_jacobian:
            if channel_derivatives is None:
                raise RuntimeError("channel derivatives are required for stress-row materialization.")
            start = time.perf_counter()
            stress_jacobian = _stress_jacobian_from_channel_derivatives(
                self.evaluator.site_basis,
                vjp_record,
                channel_derivatives,
                chunk_size=chunk_size,
            )
            profile["stress_row_seconds"] = float(time.perf_counter() - start)
        else:
            profile["stress_row_seconds"] = 0.0
        profile["total_seconds"] = float(time.perf_counter() - total_start)

        cache_report = self.evaluator.site_basis.last_atomic_base_cache_report()
        report = {
            "backend": (
                "factorized_blockwise_symmetric_power_root_adjoint"
                if factorized_active and not direct_active
                else "symmetric_power_direct_forward_autograd_adjoint"
                if use_direct_autograd
                else "ye3t_grouped_symmetric_power_product_plan"
            ),
            "descriptor_count": int(descriptor_count),
            "channel_count": int(atomic_base.shape[1]),
            "active_symmetric_descriptor_indices": sorted(int(idx) for idx in active_accelerated),
            "active_direct_symmetric_descriptor_indices": sorted(int(idx) for idx in direct_active),
            "active_factorized_blockwise_descriptor_indices": sorted(int(idx) for idx in factorized_active),
            "residual_explicit_descriptor_count": int(descriptor_count - len(active_accelerated)),
            "chunk_size": int(chunk_size),
            "chunk_count": int(len(jacobian_chunks)),
            "atomic_base_cache": dict(cache_report),
            "compiler_plan": compiler_plan_report,
            "blockwise_compiler_plan": blockwise_compiler_plan_report,
            "descriptor_adjoint_plan": descriptor_adjoint_plan.to_dict(),
            "scheduler": dict(scheduler),
            "symmetric_product": dict(sym_report),
            "factorized_blockwise_product": dict(factorized_report),
            "residual_product_rule": dict(residual_product.report),
            "timings": dict(profile),
            "uses_recursive_products": False,
            "materializes_force_jacobian": bool(materialize_force_jacobian),
            "materializes_charge_jacobian": bool(materialize_charge_jacobian),
            "materializes_stress_jacobian": bool(materialize_stress_jacobian),
            "stress_target": {
                "derivative_mode": "strain",
                "row_formula": "dphi_dr_alpha_times_r_beta",
                "volume_normalization": "not_applied",
                "ase_voigt_sign": "not_applied",
            },
        }
        return CYFactorProductEvaluation(
            descriptor_values=values,
            channel_derivatives=channel_derivatives,
            force_jacobian=force_jacobian,
            charge_jacobian=charge_jacobian,
            stress_jacobian=stress_jacobian,
            normalized_atomic_base=atomic_base,
            compiler_plan=compiler_plan if compiler_plan is not None else compiler_plan_report,
            descriptor_adjoint_plan=descriptor_adjoint_plan,
            report=report,
        )

    def normal_equations(
        self,
        *,
        positions,
        cell,
        edge_index,
        atom_types,
        descriptors,
        shifts=None,
        charges=None,
        aux_tensor_basis=None,
        real_if_scalar=True,
        descriptor_chunk_size=None,
        coordinate_chunk_size=64,
        energy_ref=None,
        force_ref=None,
        energy_weight=1.0,
        force_weight=1.0,
        atom_count,
        force_atom_stride=None,
        include_bias_column=True,
        ridge_alpha=0.0,
        ridge_include_bias=False,
    ):
        profile = {}
        total_start = time.perf_counter()

        start = time.perf_counter()
        x_ij = edge_vectors_from_positions(positions, cell, edge_index, shifts=shifts)
        profile["edge_vectors_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        compiled = self.evaluator._compile_descriptors(descriptors)
        profile["compile_lookup_seconds"] = float(time.perf_counter() - start)

        start = time.perf_counter()
        _, atomic_base, vjp_record = self.evaluator.site_basis.compute_channels_with_vjp_record(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        profile["atomic_base_seconds"] = float(time.perf_counter() - start)

        descriptor_count = int(len(descriptors))
        explicit_term_count = _compiled_explicit_term_count(compiled)
        compiler_plan = cy_factor_product_plan(
            descriptor_count=descriptor_count,
            channel_count=int(atomic_base.shape[1]),
            explicit_term_count=explicit_term_count,
            carrier="ACE_density",
            target=_descriptor_target_payload(descriptors, all_scalar=bool(compiled.all_scalar)),
            factor_basis="A_normalized",
            young_coupling={
                "sector": "globally_trivial",
                "carrier_restriction": "ordinary ACE density",
            },
            rotation_coupling={
                "backend": "compiled_CG_descriptor_rows",
                "all_scalar": bool(compiled.all_scalar),
            },
            coefficient_source="ye3t.couplings.compile compact descriptor payload",
            validation_report={
                "explicit_descriptor_terms_available": explicit_term_count > 0 or descriptor_count == 0,
                "compiled_descriptor_cache": "ACECovariantEvaluator._compile_descriptors",
            },
        )
        descriptor_adjoint_plan = ye3t_descriptor_adjoint_plan(compiler_plan)

        product_chunk_size = descriptor_chunk_size
        start = time.perf_counter()
        product = evaluate_explicit_product_rule(
            atomic_base,
            compiled,
            descriptor_count=descriptor_count,
            chunk_size=product_chunk_size,
            imag_tol=self.imag_tol,
        )
        profile["product_and_adjoint_seconds"] = float(time.perf_counter() - start)

        values = product.values
        if real_if_scalar and compiled.all_scalar:
            values = checked_real_scalar_projection(values, imag_tol=self.imag_tol, context="CYPrime normal equations")

        start = time.perf_counter()
        update = NormalEquationAccumulator(
            include_bias_column=include_bias_column,
            descriptor_chunk_size=descriptor_chunk_size,
            coordinate_chunk_size=coordinate_chunk_size,
            ridge_alpha=ridge_alpha,
            ridge_include_bias=ridge_include_bias,
        ).accumulate(
            descriptor_values=values,
            channel_derivatives=product.channel_derivatives,
            site_basis=self.evaluator.site_basis,
            vjp_record=vjp_record,
            energy_ref=energy_ref,
            force_ref=force_ref,
            energy_weight=energy_weight,
            force_weight=force_weight,
            atom_count=atom_count,
            force_atom_stride=force_atom_stride,
        )
        profile["normal_equation_accumulation_seconds"] = float(time.perf_counter() - start)
        profile["total_seconds"] = float(time.perf_counter() - total_start)

        cache_report = self.evaluator.site_basis.last_atomic_base_cache_report()
        report = {
            **dict(update.report),
            "backend": "cyprime_streamed_normal_equations",
            "algorithmic_reference": "CY factor-product explicit force-row streaming",
            "descriptor_count": int(descriptor_count),
            "channel_count": int(atomic_base.shape[1]),
            "explicit_term_count": int(explicit_term_count),
            "atomic_base_cache": dict(cache_report),
            "compiler_plan": compiler_plan.to_dict(),
            "descriptor_adjoint_plan": descriptor_adjoint_plan.to_dict(),
            "product_rule": dict(product.report),
            "timings": dict(profile),
            "uses_recursive_products": False,
            "materializes_force_jacobian": False,
        }
        return NormalEquationUpdate(
            XtX=update.XtX,
            Xty=update.Xty,
            yty=update.yty,
            n_rows=update.n_rows,
            report=report,
        )


@recordclass(('XtX', 'Xty', 'yty', 'n_rows', 'report'), frozen = True)
class NormalEquationUpdate:
    pass


class ForceDesignAccumulator:
    """Accumulate linear ACE normal equations from energy and force rows."""

    def __init__(self, *, include_bias_column=True):
        self.include_bias_column = bool(include_bias_column)

    def accumulate(
        self,
        *,
        descriptor_values,
        force_jacobian,
        energy_ref=None,
        force_ref=None,
        energy_weight=1.0,
        force_weight=1.0,
        atom_count,
    ):
        total_desc = descriptor_values.sum(dim=0).to(dtype=torch.float64)
        n_feat = int(total_desc.numel())
        n_cols = n_feat + (1 if self.include_bias_column else 0)
        device = total_desc.device
        XtX = torch.zeros((n_cols, n_cols), dtype=torch.float64, device=device)
        Xty = torch.zeros((n_cols,), dtype=torch.float64, device=device)
        yty = torch.zeros((), dtype=torch.float64, device=device)
        n_rows = 0
        timing_start = time.perf_counter()

        sqrt_energy_weight = float(max(float(energy_weight), 0.0) ** 0.5)
        if energy_ref is not None and sqrt_energy_weight > 0.0:
            if self.include_bias_column:
                energy_row = torch.cat(
                    [
                        total_desc,
                        torch.as_tensor([float(atom_count)], dtype=torch.float64, device=device),
                    ],
                    dim=0,
                )
            else:
                energy_row = total_desc
            energy_row = energy_row * sqrt_energy_weight
            target = torch.as_tensor(float(energy_ref) * sqrt_energy_weight, dtype=torch.float64, device=device)
            XtX = XtX + torch.outer(energy_row, energy_row)
            Xty = Xty + energy_row * target
            yty = yty + target * target
            n_rows += 1

        sqrt_force_weight = float(max(float(force_weight), 0.0) ** 0.5)
        if force_ref is not None and sqrt_force_weight > 0.0:
            jac = force_jacobian.to(dtype=torch.float64)
            target = torch.as_tensor(force_ref, dtype=torch.float64, device=device).reshape(-1) * sqrt_force_weight
            weighted_jac = jac * sqrt_force_weight
            XtX[:n_feat, :n_feat] = XtX[:n_feat, :n_feat] + weighted_jac @ weighted_jac.transpose(0, 1)
            Xty[:n_feat] = Xty[:n_feat] - weighted_jac @ target
            yty = yty + torch.dot(target, target)
            n_rows += int(target.numel())

        report = {
            "backend": "force_design_accumulator",
            "include_bias_column": bool(self.include_bias_column),
            "feature_count": int(n_feat),
            "row_count": int(n_rows),
            "normal_update_seconds": float(time.perf_counter() - timing_start),
        }
        return NormalEquationUpdate(XtX=XtX, Xty=Xty, yty=yty, n_rows=int(n_rows), report=report)


class NormalEquationAccumulator:
    """Stream energy and force rows into normal equations without storing full J."""

    def __init__(
        self,
        *,
        include_bias_column=True,
        descriptor_chunk_size=None,
        coordinate_chunk_size=64,
        ridge_alpha=0.0,
        ridge_include_bias=False,
    ):
        self.include_bias_column = bool(include_bias_column)
        self.descriptor_chunk_size = None if descriptor_chunk_size is None else max(1, int(descriptor_chunk_size))
        self.coordinate_chunk_size = max(1, int(coordinate_chunk_size))
        self.ridge_alpha = float(ridge_alpha)
        self.ridge_include_bias = bool(ridge_include_bias)

    def accumulate(
        self,
        *,
        descriptor_values,
        channel_derivatives,
        site_basis,
        vjp_record,
        energy_ref=None,
        force_ref=None,
        energy_weight=1.0,
        force_weight=1.0,
        atom_count,
        force_atom_stride=None,
    ):
        total_desc = descriptor_values.sum(dim=0).to(dtype=torch.float64)
        n_feat = int(total_desc.numel())
        n_cols = n_feat + (1 if self.include_bias_column else 0)
        device = total_desc.device
        XtX = torch.zeros((n_cols, n_cols), dtype=torch.float64, device=device)
        Xty = torch.zeros((n_cols,), dtype=torch.float64, device=device)
        yty = torch.zeros((), dtype=torch.float64, device=device)
        n_rows = 0
        timing_start = time.perf_counter()

        sqrt_energy_weight = float(max(float(energy_weight), 0.0) ** 0.5)
        if energy_ref is not None and sqrt_energy_weight > 0.0:
            if self.include_bias_column:
                energy_row = torch.cat(
                    [
                        total_desc,
                        torch.as_tensor([float(atom_count)], dtype=torch.float64, device=device),
                    ],
                    dim=0,
                )
            else:
                energy_row = total_desc
            energy_row = energy_row * sqrt_energy_weight
            target = torch.as_tensor(float(energy_ref) * sqrt_energy_weight, dtype=torch.float64, device=device)
            XtX = XtX + torch.outer(energy_row, energy_row)
            Xty = Xty + energy_row * target
            yty = yty + target * target
            n_rows += 1

        sqrt_force_weight = float(max(float(force_weight), 0.0) ** 0.5)
        force_component_count = 0
        coordinate_chunk_count = 0
        descriptor_chunk_count = 0
        row_block_bytes = 0
        full_force_jacobian_bytes = 0
        if force_ref is not None and sqrt_force_weight > 0.0 and n_feat > 0:
            target_all = torch.as_tensor(force_ref, dtype=torch.float64, device=device).reshape(-1)
            if force_atom_stride is not None:
                stride = max(1, int(force_atom_stride))
                atom_indices = torch.arange(0, int(atom_count), stride, dtype=torch.long, device=device)
                component_indices = torch.cat([3 * atom_indices + axis for axis in range(3)]).sort().values
                target_all = target_all.index_select(0, component_indices)
            else:
                component_indices = torch.arange(int(target_all.numel()), dtype=torch.long, device=device)

            force_component_count = int(component_indices.numel())
            descriptor_chunk_size = self.descriptor_chunk_size or max(1, n_feat)
            full_force_jacobian_bytes = int(n_feat * int(atom_count) * 3 * torch.empty((), dtype=torch.float64).element_size())
            row_block_bytes = int(
                min(self.coordinate_chunk_size, max(force_component_count, 1))
                * n_cols
                * torch.empty((), dtype=torch.float64).element_size()
            )
            for coord_start in range(0, force_component_count, self.coordinate_chunk_size):
                coord_stop = min(force_component_count, coord_start + self.coordinate_chunk_size)
                coord_idx = component_indices[coord_start:coord_stop]
                row_block = torch.zeros((coord_stop - coord_start, n_cols), dtype=torch.float64, device=device)
                for desc_start in range(0, n_feat, descriptor_chunk_size):
                    desc_stop = min(n_feat, desc_start + descriptor_chunk_size)
                    channel_adjoint = (
                        channel_derivatives[:, desc_start:desc_stop, :]
                        .permute(1, 0, 2)
                        .contiguous()
                    )
                    position_grad = site_basis.position_vjp_from_record_batched(vjp_record, channel_adjoint)
                    jac_chunk = position_grad.reshape(desc_stop - desc_start, -1).to(dtype=torch.float64)
                    row_block[:, desc_start:desc_stop] = -jac_chunk.index_select(1, coord_idx).transpose(0, 1)
                    descriptor_chunk_count += 1
                row_block = row_block * sqrt_force_weight
                target = target_all[coord_start:coord_stop] * sqrt_force_weight
                XtX = XtX + row_block.transpose(0, 1) @ row_block
                Xty = Xty + row_block.transpose(0, 1) @ target
                yty = yty + torch.dot(target, target)
                n_rows += int(target.numel())
                coordinate_chunk_count += 1

        if self.ridge_alpha > 0.0:
            ridge_cols = n_cols if self.ridge_include_bias else n_feat
            if ridge_cols > 0:
                diag = torch.arange(ridge_cols, dtype=torch.long, device=device)
                XtX[diag, diag] = XtX[diag, diag] + float(self.ridge_alpha)

        report = {
            "backend": "normal_equation_accumulator_streamed_coordinate_blocks",
            "include_bias_column": bool(self.include_bias_column),
            "feature_count": int(n_feat),
            "row_count": int(n_rows),
            "force_component_count": int(force_component_count),
            "descriptor_chunk_size": None if self.descriptor_chunk_size is None else int(self.descriptor_chunk_size),
            "coordinate_chunk_size": int(self.coordinate_chunk_size),
            "descriptor_chunk_count": int(descriptor_chunk_count),
            "coordinate_chunk_count": int(coordinate_chunk_count),
            "deterministic_order": "coordinate_blocks_then_descriptor_chunks_in_index_order",
            "coordinate_incidence_source": "SiteBasisV2.position_vjp_from_record_batched aggregates all affected centers",
            "materialized_force_jacobian_bytes_estimate": int(full_force_jacobian_bytes),
            "stream_row_block_bytes_estimate": int(row_block_bytes),
            "memory_reduction_factor_estimate": (
                None if row_block_bytes <= 0 else float(full_force_jacobian_bytes) / float(row_block_bytes)
            ),
            "ridge_alpha": float(self.ridge_alpha),
            "ridge_include_bias": bool(self.ridge_include_bias),
            "normal_update_seconds": float(time.perf_counter() - timing_start),
        }
        return NormalEquationUpdate(XtX=XtX, Xty=Xty, yty=yty, n_rows=int(n_rows), report=report)


@recordclass(('site_energy', 'total_energy', 'forces', 'report'), frozen = True)
class ModelForceResult:
    pass


class ModelForceEvaluator:
    """Evaluate linear model forces without constructing force-design rows."""

    def evaluate(self, factor_product_evaluation, coefficients):
        values = factor_product_evaluation.descriptor_values
        coeff = torch.as_tensor(coefficients, dtype=values.dtype, device=values.device)
        if tuple(coeff.shape) != (int(values.shape[1]),):
            raise ValueError(f"coefficients must have shape {(int(values.shape[1]),)}; got {tuple(coeff.shape)}")
        site_energy = values @ coeff
        total_energy = site_energy.sum()
        forces = -(factor_product_evaluation.force_jacobian.transpose(0, 1) @ coeff).reshape(-1, 3)
        return ModelForceResult(
            site_energy=site_energy,
            total_energy=total_energy,
            forces=forces,
            report={
                "backend": "model_force_evaluator_from_cy_factor_product_rows",
                "descriptor_count": int(values.shape[1]),
                "atom_count": int(forces.shape[0]),
            },
        )
