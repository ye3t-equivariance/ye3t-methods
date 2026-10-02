
"""High-level descriptor and descriptor-gradient helpers for ye3t_ace."""

import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import field
from pathlib import Path

import numpy as np
import torch
from ye3t import (
    CompileBalancedTree,
    CompileGlobalYE3TCouplers,
    CompileGlobalYE3TCouplerFamily,
    CompileGlobalYE3TCouplersCached,
    YE3TCouplerCertificate,
    YE3TReadoutSpec,
    YE3TRotationTarget,
    YE3TSpec,
    evaluate_global_coupler_family_reference_torch,
    evaluate_global_coupler_reference_torch,
    plan_ye3t_backend,
)
from ye3t.couplings import blockwise_symmetric_power_labels

from ye3t_ace.cache import DescriptorBuildCache, default_linear_cache_directory
from ye3t_ace.equivariant_calc import (
    ACECovariantEvaluator,
    AtomicProductCollection,
    DescriptorGenerationSettings,
    GeneralizedCouplingLibrary,
    NeighborData,
    SiteBasisConfig,
    build_descriptor_specs_from_settings,
    compile_descriptor_artifacts,
    descriptor_gradients_wrt_positions,
    enumerate_compact_labels,
    format_lammps_compute_pace_like,
    label_has_natural_parity,
    magnetic_orientation_basis,
    neighbor_data_from_ase_atoms,
    resolve_charge_bounds,
    deserialize_site_basis_config,
    normalize_basis_mode,
)
from ye3t_ace.ace_labeler import ExactACELabeler
from ye3t_ace.equivariant_calc.gradients import LAMMPSPaceLikeOutput, edge_vectors_from_positions
from ye3t_ace.equivariant_calc.labeling import CompactLabel, DescriptorSpec, normalize_compact_label
from ye3t_ace._record import recordclass
from ye3t_ace.representations import YE3TRepresentation, normalize_a_s_subselection
from ye3t_ace.utils.element_defaults import (
    build_explicit_type_map,
    infer_elements_from_ase_atoms,
    ordered_pair_values,
    type_ids_from_symbols,
    vdw_scaled_bond_defaults,
)
from ye3t_ace.utils.fit_weights import structure_fit_weights


def _float_tensor(value, *, dtype, device = None):
    return torch.as_tensor(value, dtype=dtype, device=device)


def _long_tensor(value, *, device = None):
    return torch.as_tensor(value, dtype=torch.long, device=device)


@recordclass(('values', 'gradients'), frozen = True)
class DescriptorGradientResult:

    def as_lammps_compute_pace_like(self):
        return format_lammps_compute_pace_like(self.values, self.gradients)


def _resolve_descriptor_device(device):
    if device is None:
        return None
    text = str(device).strip().lower()
    if text in {"", "none"}:
        return None
    if text in {"auto", "cuda_if_available", "cuda-if-available"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Descriptor device 'cuda' was requested, but torch.cuda.is_available() is False.")
    return resolved


def _a_s_matrix_unit_contraction_plan(*, ye3t_axis=False):
    """Return the current A_s matrix-unit descriptor contraction boundary."""

    implemented_stages = [
        "geometry_to_A_s_slot_density",
        "slot_permutation_action",
        "slot_specht_matrix_unit_projection",
        "carrier_block_flattening",
    ]
    if ye3t_axis:
        implemented_stages.append("scalar_L_R_0_M_R_0_axis_packaging")
    missing_stages = [
        "global_content_block_map",
        "young_subduction_map",
        "young_induction_coset_shuffle_lift",
        "global_angular_CG_contraction",
        "normalization_from_global_coupler",
        "sparse_or_factorized_global_coefficient_table_contraction",
        "multiplicity_resolved_global_alpha_layout",
    ]
    if ye3t_axis:
        missing_stages.append("non_scalar_M_R_resolved_carrier_axes")
    return {
        "status": "carrier_axes_only_not_global_ye3t_contraction",
        "implemented_stages": tuple(implemented_stages),
        "missing_stages": tuple(missing_stages),
        "coefficient_table_source": "not_compiled_for_A_s_descriptor",
        "global_coupler_contracted": False,
        "descriptor_runtime_claim": "carrier_realization_under_validation",
        "supported_target_L_R": (0,) if ye3t_axis else tuple(),
        "unsupported_target_L_R_reason": (
            "A_s matrix-unit YE3T-axis packaging currently has only singleton scalar M_R=0 axes."
            if ye3t_axis
            else "This flattened carrier descriptor does not expose explicit Young-E3 L_R/M_R axes."
        ),
    }


def _a_s_role_coordinate_report(sector, carrier=None):
    """Report whether an A_s sector retains the role/slot coordinate."""

    sector = dict(sector)
    partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
    power = int(sector.get("power", 0))
    slot_count = int(sum(partition)) if partition else 0
    is_globally_symmetric = bool(partition and len(partition) == 1 and int(partition[0]) == int(slot_count))
    is_nontrivial_partition = bool(partition and not is_globally_symmetric)
    carrier_shape = tuple(int(dim) for dim in getattr(carrier, "shape", ())) if carrier is not None else tuple()
    retained_role_axis = bool(slot_count > 1 and power > 0)
    carrier_last_axis = int(carrier_shape[-1]) if carrier_shape else 0
    expected_slot_tuple_axis = int(slot_count) ** int(power) if retained_role_axis else 0
    role_axis_matches_carrier = bool(
        carrier_shape and retained_role_axis and carrier_last_axis == expected_slot_tuple_axis
    )
    return {
        "role_coordinate": "A_s_filter_slot_label",
        "role_coordinate_retained": bool(retained_role_axis),
        "role_group_action": "slot_label_permutation_on_A_s_role_axis",
        "physical_neighbor_invariance_source": "sum_over_neighbors_inside_each_role_field",
        "formal_slot_equivariance_source": "retained_role_coordinate_before_Young_projection",
        "slot_specht_partition": partition,
        "slot_group_size": int(slot_count),
        "power": int(power),
        "globally_symmetric_partition": bool(is_globally_symmetric),
        "nontrivial_partition": bool(is_nontrivial_partition),
        "collapse_condition": (
            "if all role filters are identical and the role coordinate is discarded before Young coupling, "
            "the role module becomes trivial and nontrivial Young sectors vanish"
        ),
        "identical_role_filters_declared": bool(sector.get("identical_role_filters_declared", False)),
        "role_coordinate_discarded_before_young_projection": bool(
            sector.get("role_coordinate_discarded_before_young_projection", False)
        ),
        "nontrivial_sector_would_vanish_if_roles_identical_and_discarded": bool(is_nontrivial_partition),
        "carrier_shape": carrier_shape,
        "expected_slot_tuple_axis": int(expected_slot_tuple_axis),
        "role_axis_matches_carrier": bool(role_axis_matches_carrier),
        "passed": bool(
            (not is_nontrivial_partition)
            or (retained_role_axis and (carrier is None or role_axis_matches_carrier))
        ),
    }


def _a_s_role_coordinate_summary(role_reports):
    reports = tuple(role_reports)
    return {
        "status": "role_coordinate_boundary_report",
        "sector_count": int(len(reports)),
        "nontrivial_sector_count": int(sum(bool(report.get("nontrivial_partition", False)) for report in reports)),
        "all_nontrivial_role_coordinates_retained": all(
            bool((not report.get("nontrivial_partition", False)) or report.get("role_coordinate_retained", False))
            for report in reports
        ),
        "all_role_axes_match_carrier_shape": all(
            bool((not report.get("nontrivial_partition", False)) or report.get("role_axis_matches_carrier", False))
            for report in reports
        ),
        "collapse_condition": (
            "if all role filters are identical and the role coordinate is discarded before Young coupling, "
            "the role module becomes trivial and nontrivial Young sectors vanish"
        ),
    }


def _a_s_matrix_unit_axis_scope(*, ye3t_axis=False):
    return {
        "carrier_axis_model": "A_s_slot_specht_matrix_unit_carrier",
        "slot_group_action": "diagonal_slot_permutation_on_slot_tuple_carrier",
        "tableau_axis_status": "explicit_matrix_unit_row_and_column_axes",
        "lambda_axis_status": (
            "slot_specht_partition_axis_exposed_not_global_S_N_lambda"
        ),
        "alpha_axis_status": (
            "slot_tuple_carrier_axis_exposed_not_global_alpha_label"
        ),
        "L_R_axis_status": (
            "explicit_scalar_L_R_0_axis" if ye3t_axis else "not_exposed_in_flattened_carrier_view"
        ),
        "M_R_axis_status": (
            "explicit_singleton_M_R_0_axis" if ye3t_axis else "not_exposed_in_flattened_carrier_view"
        ),
        "global_coupler_contracted": False,
        "global_label_status": "requires_global_coupler_induction_subduction_angular_contraction",
        "validation_hook": "YE3TDescriptorSet.validate_A_s_matrix_unit_carriers",
    }


def _a_s_matrix_unit_sector_axis_metadata(sector, carrier, *, ye3t_axis=False):
    partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
    power = int(sector.get("power", 0))
    shape = tuple(int(dim) for dim in carrier.shape)
    slot_count = int(sum(partition)) if partition else 0
    alpha_size = int(shape[-1]) if shape else 0
    metadata = {
        **_a_s_matrix_unit_axis_scope(ye3t_axis=ye3t_axis),
        "slot_specht_partition": partition,
        "power": int(power),
        "slot_count": int(slot_count),
        "tableau_shape": (int(shape[1]), int(shape[2])) if len(shape) >= 3 else tuple(),
        "alpha_size": int(alpha_size),
        "slot_tuple_carrier_dim": int(alpha_size),
        "expected_slot_tuple_carrier_dim": int(slot_count) ** int(power) if slot_count and power else 0,
        "slot_tuple_carrier_dim_matches": bool(
            bool(slot_count)
            and bool(power)
            and int(alpha_size) == int(slot_count) ** int(power)
        ),
        "source_sector_runtime": sector.get("runtime", "slot_specht_matrix_unit_carrier"),
    }
    return metadata


def _a_s_matrix_unit_validation_summary(sector_reports):
    reports = tuple(sector_reports)
    compatibility_reports = tuple(
        report.get("global_coupler_compatibility", {})
        for report in reports
    )
    role_reports = tuple(report.get("role_coordinate_report", {}) for report in reports)
    return {
        "validated_identities": (
            "matrix_unit_multiplication_law",
            "central_projector_equals_diagonal_matrix_unit_sum",
            "central_projector_idempotency",
            "diagonal_matrix_unit_ranks_match_isotypic_multiplicity",
            "evaluated_carrier_axes_match_sector_metadata",
            "evaluated_matrix_unit_carrier_slot_permutation_covariance",
            "role_coordinate_retained_for_nontrivial_A_s_Young_sectors",
        ),
        "global_validation_not_performed": (
            "global_content_block_map",
            "young_subduction_map",
            "young_induction_coset_shuffle_lift",
            "global_angular_CG_contraction",
            "global_alpha_multiplicity_resolution",
        ),
        "all_matrix_unit_algebras_passed": all(
            bool(report.get("matrix_unit_algebra", {}).get("passed", False))
            for report in reports
        ),
        "all_carrier_axes_match": all(
            bool(report.get("carrier_dim_matches", False))
            and bool(report.get("tableau_axes_match_specht_dimension", False))
            for report in reports
        ),
        "all_sector_axis_metadata_consistent": all(
            bool(report.get("axis_metadata", {}).get("slot_tuple_carrier_dim_matches", False))
            for report in reports
        ),
        "all_matrix_unit_covariance_checks_passed": all(
            bool(report.get("matrix_unit_covariance", {}).get("passed", False))
            for report in reports
        ),
        "role_coordinate_summary": _a_s_role_coordinate_summary(role_reports),
        "global_coupler_compatibility_summary": _a_s_matrix_unit_global_coupler_compatibility_summary(
            compatibility_reports
        ),
        "sector_count": int(len(reports)),
        "passed": all(bool(report.get("passed", False)) for report in reports),
    }


def _a_s_matrix_unit_global_coupler_compatibility_summary(compatibility_reports):
    compatibility_reports = tuple(compatibility_reports)
    slot_resolved_candidates = tuple(
        report.get("slot_resolved_product_slot_candidate", {})
        for report in compatibility_reports
        if report.get("slot_resolved_product_slot_candidate") is not None
    )
    repeated_candidates = tuple(
        report.get("repeated_channel_candidate", {})
        for report in compatibility_reports
        if report.get("repeated_channel_candidate") is not None
    )
    return {
        "status": "diagnostic_not_full_descriptor_evaluation",
        "sector_count": int(len(compatibility_reports)),
        "slot_resolved_candidate_count": int(len(slot_resolved_candidates)),
        "slot_resolved_candidate_certified_count": int(
            sum(bool(report.get("certificate_passed", False)) for report in slot_resolved_candidates)
        ),
        "repeated_channel_candidate_count": int(len(repeated_candidates)),
        "repeated_channel_candidate_certified_count": int(
            sum(bool(report.get("certificate_passed", False)) for report in repeated_candidates)
        ),
        "all_slot_resolved_candidates_certified": bool(
            slot_resolved_candidates
            and all(bool(report.get("certificate_passed", False)) for report in slot_resolved_candidates)
        ),
        "any_repeated_channel_candidate_zero_or_unavailable": bool(
            any(not bool(report.get("certificate_passed", False)) for report in repeated_candidates)
        ),
        "group_action_distinction": (
            "A_s matrix-unit carriers currently resolve filter/slot-label Specht actions; "
            "central global couplers resolve product/content-slot Young--E3 sectors."
        ),
        "full_descriptor_evaluation_status": (
            "global coupler candidates are compiled as compatibility metadata; "
            "the geometry carrier has not been evaluated through the global coefficient tables."
        ),
    }


def _a_s_matrix_unit_global_coupler_compatibility_report(sector, carrier):
    """Compile central-coupler candidates compatible with one A_s carrier sector.

    The current A_s matrix-unit runtime carries a Specht action of the
    filter/slot-label group on slot-tuple carriers.  The central coupler
    compiler acts on product/content slots.  This report records when the
    sector can be compared to a product-slot candidate and keeps the repeated
    channel candidate separate.
    """

    sector = dict(sector)
    partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
    power = int(sector.get("power", 0))
    slot_count = int(sum(partition)) if partition else 0
    target_L_R = int(sector.get("target_L_R", 0))
    l_in = sector.get("l_in", 0)
    if isinstance(l_in, (list, tuple)):
        input_Ls = tuple(int(value) for value in l_in)
    else:
        input_Ls = tuple(int(l_in) for _ in range(power))

    def _compile_candidate(name, content):
        if not partition or power <= 0:
            return {
                "candidate": name,
                "available": False,
                "reason": "missing slot_specht_partition or power metadata",
            }
        if len(input_Ls) != power:
            return {
                "candidate": name,
                "available": False,
                "reason": "angular input_Ls length does not match the A_s power",
                "input_Ls": input_Ls,
                "power": int(power),
            }
        spec = YE3TSpec(
            content=tuple(content),
            slot_roles=tuple(f"A_s_product_slot_{index}" for index in range(power)),
            target_permutation="young:" + ",".join(str(part) for part in partition),
            target_rotation=YE3TRotationTarget(L_R=target_L_R),
            carrier="A_s",
            task="descriptor_only",
            tree_schedule="balanced",
            coefficient_backend="global_coupler",
            fast_path_policy="disable",
            validation_scope="projectors",
            runtime_status="implemented_under_validation",
            metadata={
                "input_Ls": input_Ls,
                "source": "A_s_matrix_unit_global_coupler_compatibility_report",
                "source_slot_specht_partition": partition,
                "source_A_s_power": int(power),
            },
        )
        try:
            balanced_tree = CompileBalancedTree(spec, input_Ls=input_Ls)
            coupler = balanced_tree.coupler
        except Exception as exc:  # noqa: BLE001 - diagnostic report should preserve the compiler failure.
            return {
                "candidate": name,
                "available": False,
                "target_partition": partition,
                "content_rank": int(len(tuple(content))),
                "input_Ls": input_Ls,
                "target_L_R": int(target_L_R),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        table = coupler.sparse_coefficient_tables[0] if coupler.sparse_coefficient_tables else {}
        balanced_payload = balanced_tree.to_dict()
        balanced_certificate = balanced_payload["certificate"]
        local_image_maps = tuple(dict(record) for record in balanced_payload.get("local_repeated_content_image_maps", ()))
        nonroot_requirements = tuple(
            dict(record)
            for record in balanced_payload.get("balanced_tree_node_ledger", ())
            if (
                record.get("node_path") != "root"
                and bool(record.get("image_reduction_required", False))
                and record.get("local_image_map_status")
                != "materialized_exact_local_scalar_trivial_image_map"
            )
        )
        return {
            "candidate": name,
            "available": True,
            "target_partition": partition,
            "content_rank": int(len(tuple(content))),
            "input_Ls": input_Ls,
            "target_L_R": int(target_L_R),
            "certificate_passed": bool(coupler.certificate.passed),
            "certificate_runtime_status": coupler.certificate.runtime_status,
            "coefficient_hash": coupler.certificate.coefficient_hash,
            "coefficient_table_kind": table.get("kind"),
            "coefficient_table_shape": tuple(int(value) for value in table.get("shape", ())),
            "alpha_label_count": int(len(coupler.labels)),
            "block_maps": tuple(dict(block) for block in coupler.block_maps),
            "balanced_tree_certificate_passed": bool(balanced_certificate.get("passed", False)),
            "balanced_tree_node_ledger": tuple(
                dict(record) for record in balanced_payload.get("balanced_tree_node_ledger", ())
            ),
            "balanced_tree_node_count": int(len(balanced_payload.get("balanced_tree_node_ledger", ()))),
            "local_repeated_content_image_map_count": int(len(local_image_maps)),
            "local_repeated_content_image_maps": local_image_maps,
            "nonroot_image_map_requirement_count": int(len(nonroot_requirements)),
            "nonroot_image_map_requirements": nonroot_requirements,
            "global_coupler_applied_to_A_s_carrier": False,
        }

    slot_resolved_candidate = None
    if slot_count == power:
        slot_resolved_candidate = _compile_candidate(
            "slot_resolved_product_slots",
            tuple(f"A_s_slot_{index}" for index in range(power)),
        )

    repeated_channel_candidate = _compile_candidate(
        "repeated_channel_product_slots",
        tuple("A_s_repeated_channel" for _ in range(power)),
    )

    return {
        "status": "compatibility_metadata_not_descriptor_evaluation",
        "source_carrier_group_action": "filter_slot_label_permutation_on_slot_tuple_carrier",
        "central_coupler_group_action": "product_content_slot_permutation",
        "slot_specht_partition": partition,
        "slot_group_size": int(slot_count),
        "A_s_power": int(power),
        "slot_group_size_matches_product_rank": bool(slot_count == power),
        "carrier_shape": tuple(int(dim) for dim in carrier.shape),
        "target_L_R": int(target_L_R),
        "input_Ls": input_Ls,
        "slot_resolved_product_slot_candidate": slot_resolved_candidate,
        "repeated_channel_candidate": repeated_channel_candidate,
        "full_descriptor_evaluation_status": (
            "global coupler metadata compiled where meaningful; A_s geometry carriers are not "
            "yet evaluated through induction, subduction, angular, and normalization tables"
        ),
    }


def _compile_a_s_slot_resolved_global_coupler_for_sector(sector):
    sector = dict(sector)
    partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
    power = int(sector.get("power", 0))
    slot_count = int(sum(partition)) if partition else 0
    target_L_R = int(sector.get("target_L_R", 0))
    target_M_R_values = sector.get("target_M_R_values", sector.get("M_R_values", None))
    if int(target_L_R) == 0:
        target_M_R_values = (0,)
    l_in = sector.get("l_in", 0)
    if isinstance(l_in, (list, tuple)):
        input_Ls = tuple(int(value) for value in l_in)
    else:
        input_Ls = tuple(int(l_in) for _ in range(power))
    if not partition or power <= 0:
        raise ValueError("A_s slot-resolved global coupler requires slot_specht_partition and positive power.")
    if slot_count != power:
        raise ValueError(
            "A_s slot-resolved global coupler requires slot_group_size == power; "
            f"got slot_group_size={slot_count}, power={power}."
        )
    if len(input_Ls) != power:
        raise ValueError("A_s slot-resolved global coupler requires one angular input label per product slot.")
    spec = YE3TSpec(
        content=tuple(f"A_s_slot_{index}" for index in range(power)),
        slot_roles=tuple(f"A_s_product_slot_{index}" for index in range(power)),
        target_permutation="young:" + ",".join(str(part) for part in partition),
        target_rotation=YE3TRotationTarget(
            L_R=target_L_R,
            M_R=None if target_M_R_values is None else tuple(int(value) for value in target_M_R_values),
        ),
        carrier="A_s",
        task="descriptor_only",
        tree_schedule="balanced",
        coefficient_backend="global_coupler",
        fast_path_policy="disable",
        validation_scope="projectors",
        runtime_status="implemented_under_validation",
        metadata={
            "input_Ls": input_Ls,
            "source": "A_s_matrix_unit_slot_resolved_global_coupler_descriptor",
            "source_slot_specht_partition": partition,
            "source_A_s_power": int(power),
            "requested_target_M_R_values": (
                None
                if target_M_R_values is None
                else tuple(int(value) for value in target_M_R_values)
            ),
        },
    )
    return CompileGlobalYE3TCouplers(spec, input_Ls=input_Ls)


def _a_s_matrix_unit_runtime_scope_metadata(*, view):
    return {
        "runtime_scope": "A_s_slot_specht_matrix_unit_carrier_only",
        "global_ye3_descriptor_status": "planned_not_public",
        "global_ye3_descriptor_missing_stages": (
            "carrier_realization_to_global_block_map",
            "YoungSubductionMap_application",
            "YoungInductionCoupler_application",
            "AngularCGMap_application_for_L_R_greater_than_0",
            "normalization_and_sparse_factorized_table_application",
        ),
        "global_ye3_descriptor_runtime_claim": (
            f"{view} exposes A_s slot-Specht carrier data under validation; it is not "
            "a completed global YE3T descriptor evaluation."
        ),
    }


def _a_s_matrix_unit_runtime_capability_report(
    *,
    requested_target_L_R=0,
    requested_M_R_values=None,
):
    requested_target_L_R = int(requested_target_L_R)
    requested_M_R_values = (
        None
        if requested_M_R_values is None
        else tuple(int(value) for value in requested_M_R_values)
    )
    full_requested_M_R_values = (
        tuple(range(-int(requested_target_L_R), int(requested_target_L_R) + 1))
        if requested_target_L_R > 0
        else (0,)
    )
    requested_M_R_is_default = bool(
        requested_M_R_values is None or tuple(int(value) for value in requested_M_R_values) == full_requested_M_R_values
    )
    matrix_unit_route_supported = bool(
        requested_target_L_R == 0 or requested_M_R_is_default
    )
    return {
        "status": "A_s_matrix_unit_runtime_capability_report",
        "matrix_unit_carrier_status": "implemented_under_validation",
        "supported_runtime_routes": (
            "A_s_slot_specht_matrix_unit_carriers",
            "A_s_matrix_unit_slot_resolved_scalar_global_coupler_descriptor",
            "A_s_non_scalar_M_R_resolved_global_coupler_descriptor",
            "A_s_matrix_unit_global_coupler_energy_only_linear_model",
        ),
        "planned_runtime_routes": (
            "A_s_force_Jacobian_rows_for_matrix_unit_global_coupler_linear_model",
        ),
        "unsupported_runtime_routes": (),
        "supported_target_L_R": (0,),
        "supported_M_R_values": full_requested_M_R_values if matrix_unit_route_supported else (0,),
        "requested_target_L_R": int(requested_target_L_R),
        "requested_M_R_values": requested_M_R_values,
        "requested_target_supported_by_matrix_unit_runtime": matrix_unit_route_supported,
        "slot_resolved_global_coupler_supports_requested_target": matrix_unit_route_supported,
        "non_scalar_A_s_matrix_unit_status": (
            "not_requested"
            if requested_target_L_R == 0 and requested_M_R_values in {None, (0,)}
            else "implemented_under_validation" if matrix_unit_route_supported else "planned_not_public"
        ),
        "non_scalar_A_s_matrix_unit_missing_stage": (
            None
            if matrix_unit_route_supported
            else "M_R_resolved_A_s_matrix_unit_carrier_axes_and_angular_map_application"
        ),
        "high_degree_covariant_feature_hook": {
            "target_rotation_fields_preserved_in_spec": True,
            "stress_and_multipole_current_route": "ACE_covariant_L_R_greater_than_0_paths",
            "A_s_matrix_unit_stress_or_multipole_status": (
                "implemented_under_validation_via_slot_resolved_global_coupler_descriptor"
                if matrix_unit_route_supported and requested_target_L_R > 0
                else "planned_not_public"
            ),
        },
    }


def _descriptor_shared_spec_metadata(representation, cfg, *, runtime_status, carrier=None, task=None):
    source_spec_payload = None
    if isinstance(cfg.get("ye3t_spec"), Mapping):
        source_spec_payload = dict(cfg["ye3t_spec"])
    elif "ye3t_spec_file" in cfg:
        source_spec_payload = YE3TSpec.from_file(cfg["ye3t_spec_file"]).to_dict()
    elif isinstance(getattr(representation, "metadata", None), Mapping):
        metadata_spec = representation.metadata.get("ye3t_spec")
        if isinstance(metadata_spec, Mapping):
            source_spec_payload = dict(metadata_spec)
    source_spec = YE3TSpec.from_dict(source_spec_payload) if source_spec_payload is not None else None

    def _cfg_value(keys, default):
        for key in keys:
            if key in cfg:
                return cfg[key]
        return default

    default_rotation = source_spec.target_rotation.to_dict() if source_spec is not None else {}
    default_readout = source_spec.readout.to_dict() if source_spec is not None else {}
    content_default = source_spec.content if source_spec is not None else tuple()
    slot_roles_default = source_spec.slot_roles if source_spec is not None else tuple()
    block_permutation_default = source_spec.block_permutation if source_spec is not None else tuple()
    radial_filters_default = source_spec.radial_filters if source_spec is not None else {}
    validation_scope_default = source_spec.validation_scope if source_spec is not None else "counts"
    target_permutation_default = source_spec.target_permutation if source_spec is not None else None
    tree_schedule_default = source_spec.tree_schedule if source_spec is not None else getattr(representation, "coupling_tree", "balanced")
    coefficient_backend_default = (
        source_spec.coefficient_backend if source_spec is not None else getattr(representation, "coefficient_backend", "global_coupler")
    )
    fast_path_policy_default = source_spec.fast_path_policy if source_spec is not None else getattr(representation, "fast_path_policy", "auto")
    source_carrier = source_spec.carrier if source_spec is not None else None
    carrier_value = carrier if carrier is not None else cfg.get("carrier", source_carrier)
    source_task = source_spec.task if source_spec is not None else "descriptor_only"
    task_value = str(task if task is not None else cfg.get("task", source_task))
    metadata_default = dict(source_spec.metadata) if source_spec is not None else {}
    if isinstance(cfg.get("metadata"), Mapping):
        metadata_default.update(dict(cfg["metadata"]))
    if isinstance(cfg.get("radial_filters", None), Mapping):
        radial_filters = dict(cfg["radial_filters"])
    else:
        radial_filters = dict(radial_filters_default)
        for key in (
            "cutoff",
            "filter_kind",
            "num_filters",
            "filter_centers",
            "filter_width",
            "radial_lambda",
            "density_normalization",
            "density_normalization_nugget",
        ):
            if key in cfg:
                radial_filters[key] = cfg[key]
        if "site_basis" in cfg and isinstance(cfg["site_basis"], Mapping):
            radial_filters.setdefault("site_basis", dict(cfg["site_basis"]))
        if "lifted_density" in cfg and isinstance(cfg["lifted_density"], Mapping):
            lifted = dict(cfg["lifted_density"])
            for key in (
                "cutoff",
                "filter_kind",
                "num_filters",
                "filter_centers",
                "filter_width",
                "radial_lambda",
                "density_normalization",
                "density_normalization_nugget",
            ):
                if key in lifted:
                    radial_filters.setdefault(key, lifted[key])
    target_L_keys = ("target_L_R",) if source_spec is not None else ("L_R", "target_L_R")
    target_M_keys = ("target_M_R_values",) if source_spec is not None else ("M_R_values", "target_M_R_values")
    target_parity_keys = ("target_parity",) if source_spec is not None else ("parity", "target_parity")
    target_group_keys = (
        ("target_rotation_group",)
        if source_spec is not None
        else ("rotation_group", "group", "target_rotation_group")
    )
    target_L_value = _cfg_value(target_L_keys, default_rotation.get("L_R", 0))
    target_M_value = _cfg_value(target_M_keys, default_rotation.get("M_R", None))
    target_rotation = YE3TRotationTarget(
        L_R=int(target_L_value),
        M_R=(
            None
            if target_M_value is None
            else tuple(int(v) for v in target_M_value)
        ),
        parity=_cfg_value(target_parity_keys, default_rotation.get("parity", None)),
        group=str(_cfg_value(target_group_keys, default_rotation.get("group", "SO3"))),
    )
    readout = YE3TReadoutSpec(
        permutation=str(_cfg_value(("readout_permutation",), default_readout.get("permutation", "trivial"))),
        rotation=YE3TRotationTarget(
            L_R=int(_cfg_value(("readout_L_R",), default_readout.get("rotation", {}).get("L_R", 0))),
            parity=_cfg_value(
                ("readout_parity",),
                default_readout.get("rotation", {}).get(
                    "parity",
                    "even" if int(_cfg_value(("readout_L_R",), default_readout.get("rotation", {}).get("L_R", 0))) == 0 else None,
                ),
            ),
            group=str(
                _cfg_value(
                    ("readout_rotation_group",),
                    default_readout.get("rotation", {}).get("group", target_rotation.group),
                )
            ),
        ),
        aggregation=(
            "site_sum"
            if task_value == "atomic_scalar"
            and str(_cfg_value(("readout_aggregation",), default_readout.get("aggregation", "none"))) == "none"
            else str(_cfg_value(("readout_aggregation",), default_readout.get("aggregation", "none")))
        ),
    )
    spec_kwargs = {
        "content": tuple(_cfg_value(("content", "n_in", "nin"), content_default)),
        "slot_roles": tuple(_cfg_value(("slot_roles",), slot_roles_default)),
        "block_permutation": tuple(_cfg_value(("block_permutation",), block_permutation_default)),
        "target_rotation": target_rotation,
        "carrier": carrier_value,
        "task": task_value,
        "readout": readout,
        "radial_filters": radial_filters,
        "tree_schedule": str(_cfg_value(("tree_schedule", "coupling_tree"), tree_schedule_default)),
        "coefficient_backend": str(_cfg_value(("coefficient_backend",), coefficient_backend_default)),
        "fast_path_policy": str(_cfg_value(("fast_path_policy",), fast_path_policy_default)),
        "validation_scope": str(cfg.get("validation_scope", validation_scope_default)),
        "runtime_status": str(runtime_status),
        "metadata": metadata_default,
    }
    target_permutation_value = _cfg_value(("target_permutation", "permutation_sector"), target_permutation_default)
    if target_permutation_value is not None:
        spec_kwargs["target_permutation"] = str(target_permutation_value)
    spec = representation.to_spec(**spec_kwargs)
    plan = plan_ye3t_backend(spec)
    certificate = YE3TCouplerCertificate(
        validation_scope=spec.validation_scope,
        runtime_status=spec.runtime_status,
        passed=False,
        checks={},
        provenance={"source": "descriptor_metadata_not_coupler_validation"},
        limitations=("descriptor metadata only; no coupler certificate has been attached",),
    )
    return {
        "ye3t_spec": spec.to_dict(),
        "backend_plan": plan.to_dict(),
        "validation_certificate": certificate.to_dict(),
        "task_readout_selection_rule": spec.task_readout_selection_rule(),
        "runtime_status_strict": spec.runtime_status,
        "ye3t_spec_source": (
            "config_file"
            if "ye3t_spec_file" in cfg
            else "config_mapping"
            if isinstance(cfg.get("ye3t_spec"), Mapping)
            else "representation_metadata"
            if source_spec_payload is not None
            else "descriptor_defaults"
        ),
    }


def _ye3t_coefficient_descriptor_view_metadata(shared_spec_metadata):
    spec = YE3TSpec.from_dict(shared_spec_metadata["ye3t_spec"])
    has_content = bool(tuple(spec.content))
    supported_target = (
        spec.target_permutation in {"trivial", "antisymmetric", "full_irrep_decomposition"}
        or str(spec.target_permutation).startswith("young:")
    )
    supported_carrier = spec.carrier in {"external_tensor", "orbital"}
    available = bool(has_content and supported_target and supported_carrier)
    if not has_content:
        reason = "missing fixed content; coefficient tables cannot be dimensioned"
    elif not supported_target:
        reason = f"unsupported target_permutation={spec.target_permutation!r} for coefficient-table view"
    elif not supported_carrier:
        reason = f"carrier={spec.carrier!r} is not caller-supplied external/orbital data"
    else:
        reason = "caller-supplied carrier values can be evaluated through compiled coefficient tables"
    view_kind = (
        "direct_sum_global_coupler_family_caller_supplied_values"
        if spec.target_permutation == "full_irrep_decomposition"
        else "single_global_coupler_caller_supplied_values"
    )
    return {
        "coefficient_descriptor_view_status": (
            "implemented_under_validation" if available else "planned_not_public"
        ),
        "coefficient_descriptor_view_available": available,
        "coefficient_descriptor_view_kind": view_kind if available else "not_available",
        "coefficient_descriptor_view_reason": reason,
        "coefficient_descriptor_view_requires": (
            "caller_supplied_carrier_or_coefficient_basis_values",
            "explicit fixed content",
            "global coupler coefficient table",
        ),
        "coefficient_descriptor_geometry_realization": (
            "caller_supplied_values_only" if available else "not_available"
        ),
        "coefficient_descriptor_is_geometry_descriptor": False,
        "coefficient_descriptor_public_scope": (
            "applies compiled Young-E3 coefficient tables to caller-supplied values; "
            "does not construct site, slot, or orbital carriers from geometry"
        ),
        "coefficient_descriptor_view_supported_targets": (
            "trivial",
            "antisymmetric",
            "full_irrep_decomposition",
            "young:<partition>",
        ),
        "coefficient_descriptor_view_supported_carriers": (
            "external_tensor",
            "orbital",
        ),
        "full_geometry_descriptor_runtime_status": "planned_not_public",
        "full_geometry_descriptor_missing_stages": (
            "carrier_realization",
            "slot_or_site_input_layout",
            "block_map_application",
            "subduction_and_induction_maps",
            "angular_M_R_axis_realization_for_L_R_greater_than_0",
            "normalization_and_sparse_table_application_to_realized_geometry_carriers",
        ),
        "full_geometry_descriptor_runtime_claim": (
            "not implemented by YE3TDescriptors.ye3t; use coefficient descriptor views only with "
            "caller-supplied values for the supported targets above"
        ),
    }


def _phi_reachable_total_angular_momenta(l_values):
    values = tuple(int(value) for value in l_values)
    if not values:
        return (0,)
    reachable = {int(values[0])}
    for ell in values[1:]:
        next_reachable = set()
        for left in reachable:
            for target in range(abs(int(left) - int(ell)), int(left) + int(ell) + 1):
                next_reachable.add(int(target))
        reachable = next_reachable
    return tuple(sorted(reachable))


def _phi_orbit_content(orbit_partition):
    content = []
    for orbit_index, size in enumerate(tuple(int(value) for value in orbit_partition)):
        content.extend([int(orbit_index)] * int(size))
    return tuple(content)


def _phi_complete_inventory_from_phi_config(representation, phi_config, cfg):
    from types import SimpleNamespace
    from ye3t.representation_decomposition import build_young_subgroup_irrep_inventory

    subduction_cache_dir = cfg.get("subduction_cache_dir", cfg.get("coupler_cache_dir", None))
    bracketing = str(cfg.get("coupling_tree", "balanced"))
    task = str(cfg.get("task", "descriptor_only"))
    readout = cfg.get("readout", None)
    motif_records = []
    total_couplers = 0
    total_sectors = 0
    inventory_passed = True
    coupler_validation_passed = True

    for motif_spec in tuple(phi_config.motif_specs):
        orbit_partition = tuple(int(value) for value in motif_spec.slot_orbit_partition)
        if not orbit_partition:
            continue
        subgroup_partitions = tuple((int(size),) for size in orbit_partition)
        orbit_l_values = tuple(int(value) for value in motif_spec.orbit_l_values)
        slot_l_values = tuple(int(value) for value in motif_spec.slot_l_values)
        target_L_R_values = _phi_reachable_total_angular_momenta(orbit_l_values)
        if len(subgroup_partitions) == 1:
            rank = int(sum(orbit_partition))
            target_partition = tuple((rank,))
            target_L_R = int(target_L_R_values[0] if target_L_R_values else 0)
            synthetic_record = SimpleNamespace(
                rank=rank,
                subgroup_partitions=subgroup_partitions,
                target_partition=target_partition,
                l_in=orbit_l_values,
                L_R=target_L_R,
                multiplicity=1,
            )
            inventory = SimpleNamespace(
                records=(synthetic_record,),
                validation=SimpleNamespace(
                    passed=True,
                    record_count=1,
                    all_records_passed=True,
                    projectors_orthogonal_by_common_induced_space=True,
                    coherence_checked_count=0,
                    detail="one-factor identity sector",
                    as_dict=lambda: {
                        "record_count": 1,
                        "passed": True,
                        "all_records_passed": True,
                        "projectors_orthogonal_by_common_induced_space": True,
                        "coherence_checked_count": 0,
                        "detail": "one-factor identity sector",
                    },
                ),
                as_dict=lambda include_tensors=False: {
                    "subgroup_partitions": [list(partition) for partition in subgroup_partitions],
                    "l_in": [int(value) for value in orbit_l_values],
                    "target_L_R_values": [int(value) for value in target_L_R_values],
                    "records": [
                        {
                            "rank": rank,
                            "subgroup_partitions": [list(partition) for partition in subgroup_partitions],
                            "target_partition": [rank],
                            "l_in": [int(value) for value in orbit_l_values],
                            "L_R": int(target_L_R),
                            "multiplicity": 1,
                            "carrier_dim": 1,
                            "induced_dim": 1,
                            "bracketing": bracketing,
                            "coefficient_backend": "identity",
                            "product_runtime_status": "identity_sector",
                        }
                    ],
                    "validation": {
                        "record_count": 1,
                        "passed": True,
                        "all_records_passed": True,
                        "projectors_orthogonal_by_common_induced_space": True,
                        "coherence_checked_count": 0,
                        "detail": "one-factor identity sector",
                    },
                    "metadata": {
                        "scope": "synthetic_identity_sector",
                        "motif_name": motif_spec.name,
                    },
                },
            )
        else:
            inventory = build_young_subgroup_irrep_inventory(
                subgroup_partitions=subgroup_partitions,
                l_in=orbit_l_values,
                target_L_R_values=target_L_R_values,
                bracketing=bracketing,
                coefficient_backend="subduction_graph",
                validate_coherence=True,
            )
        coupler_records = []
        content = _phi_orbit_content(orbit_partition)
        for record in inventory.records:
            target_partition = tuple(int(part) for part in record.target_partition)
            target_permutation = "young:" + ",".join(str(part) for part in target_partition)
            target_rotation = YE3TRotationTarget(L_R=int(record.L_R))
            complete_spec = representation.to_spec(
                carrier="Phi",
                task=task,
                readout=readout,
                content=content,
                target_permutation=target_permutation,
                target_rotation=target_rotation,
                metadata={
                    "motif_family": "complete",
                    "motif_name": motif_spec.name,
                    "motif_template": motif_spec.template.to_dict(),
                    "motif_orbit_partition": list(orbit_partition),
                    "motif_orbit_l_values": list(orbit_l_values),
                    "motif_slot_l_values": list(slot_l_values),
                    "phi_inventory_scope": "young_subgroup_specht_exact_small_rank",
                },
            )
            coupler = CompileGlobalYE3TCouplersCached(
                complete_spec,
                input_Ls=slot_l_values,
                subduction_cache_dir=subduction_cache_dir,
                compare_exact_projector=bool(cfg.get("compare_exact_projector", False)),
            )
            coupler_records.append(
                {
                    "target_partition": [int(part) for part in target_partition],
                    "L_R": int(record.L_R),
                    "multiplicity": int(record.multiplicity),
                    "coupler_certificate": coupler.certificate.to_dict(),
                    "joint_cache_key": coupler.cache_key(),
                    "subduction_map": coupler.subduction_maps[0].to_dict(),
                    "angular_map": coupler.angular_maps[0].to_dict(),
                }
            )
            total_couplers += 1
        motif_inventory_passed = bool(inventory.validation.passed)
        motif_couplers_passed = all(bool(entry["coupler_certificate"]["passed"]) for entry in coupler_records)
        inventory_passed = bool(inventory_passed and motif_inventory_passed)
        coupler_validation_passed = bool(coupler_validation_passed and motif_couplers_passed)
        total_sectors += int(inventory.validation.record_count)
        motif_records.append(
            {
                "motif_name": motif_spec.name,
                "motif_template": motif_spec.template.to_dict(),
                "orbit_partition": list(orbit_partition),
                "orbit_l_values": list(orbit_l_values),
                "slot_l_values": list(slot_l_values),
                "subgroup_partitions": [list(partition) for partition in subgroup_partitions],
                "inventory": inventory.as_dict(),
                "validation": inventory.validation.as_dict(),
                "couplers": coupler_records,
                "coupler_count": len(coupler_records),
            }
        )
    return {
        "motif_count": len(motif_records),
        "sector_count": int(total_sectors),
        "coupler_count": int(total_couplers),
        "passed": bool(motif_records and inventory_passed),
        "coupler_validation_passed": bool(coupler_validation_passed),
        "motifs": motif_records,
        "scope": "exact small-rank Young-subgroup/Specht inventory with cached Young+CG couplers",
    }


@recordclass(('settings', 'site_basis_config', 'labels', 'descriptor_specs', 'coupling_library', 'evaluator', 'backend', 'strict_backend', 'validate_backend', 'M_R', 'device'))
class DescriptorCalculator:
    """Reusable exact-ACE descriptor runtime.

    Build the descriptor inventory once, then evaluate values and gradients on
    raw neighbor data or directly from ASE atoms.
    """
    backend = "pytorch"
    strict_backend = False
    validate_backend = True
    M_R = 0
    device = None

    @classmethod
    def from_settings(
        cls,
        settings,
        site_basis_config,
        *,
        coupling_library = None,
        basis_mode = None,
        exact_primitive_timeout_seconds = None,
        compact_labels = None,
        center_mu_values = None,
        restrict_neighbor_mu = None,
        max_variants_per_label = None,
        M_R = None,
        descriptor_cache = None,
        use_descriptor_cache = True,
        backend = "pytorch",
        strict_backend = False,
        validate_backend = True,
        device = None,
        factorized_descriptor_runtime_policy = None,
        scalar_coordinate_compiler = None,
        _suppress_legacy_warning = False,
    ):
        if not _suppress_legacy_warning:
            warnings.warn(
                "DescriptorCalculator.from_settings is a legacy descriptor-runtime entry point. "
                "Use YE3TDescriptors.ace(...) or YE3TDescriptors.ye3t(...) for new workflows.",
                FutureWarning,
                stacklevel=2,
            )
        chosen_basis_mode = normalize_basis_mode(
            basis_mode,
            L_R=settings.L_R,
        )
        library = coupling_library
        selected_M = int(0 if M_R is None and 0 in settings.M_R_values else (settings.M_R_values[0] if M_R is None else M_R))
        if library is None:
            labels, library, collection = compile_descriptor_artifacts(
                settings,
                compact_labels=compact_labels,
                basis_mode=chosen_basis_mode,
                exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
                center_mu_values=center_mu_values,
                restrict_neighbor_mu=restrict_neighbor_mu,
                max_variants_per_label=max_variants_per_label,
                descriptor_cache=descriptor_cache,
                use_descriptor_cache=use_descriptor_cache,
                scalar_coordinate_compiler=scalar_coordinate_compiler,
            )
        else:
            labels = tuple(
                compact_labels
                if compact_labels is not None
                else enumerate_compact_labels(
                    settings,
                    basis_mode=chosen_basis_mode,
                    exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
                    descriptor_cache=descriptor_cache,
                    use_descriptor_cache=use_descriptor_cache,
                )
            )
            collection = build_descriptor_specs_from_settings(
                labels,
                settings,
                library,
                center_mu_values=center_mu_values,
                restrict_neighbor_mu=restrict_neighbor_mu,
                max_variants_per_label=max_variants_per_label,
            )
        specs = tuple(collection.specs_by_M[selected_M])
        return cls(
            settings=settings,
            site_basis_config=site_basis_config,
            labels=labels,
            descriptor_specs=specs,
            coupling_library=library,
            evaluator=ACECovariantEvaluator(
                site_basis_config,
                backend=backend,
                strict_backend=strict_backend,
                validate_backend=validate_backend,
                factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
            ),
            backend=str(backend),
            strict_backend=bool(strict_backend),
            validate_backend=bool(validate_backend),
            M_R=selected_M,
            device=_resolve_descriptor_device(device),
        )

    def _runtime(self, device):
        return self.evaluator.to(device=device)

    def compute(
        self,
        positions,
        cell,
        edge_index,
        atom_types,
        *,
        shifts = None,
        charges = None,
        aux_tensor_basis = None,
        real_if_scalar = True,
    ):
        positions_t = _float_tensor(positions, dtype=self.site_basis_config.dtype, device=self.device)
        cell_t = _float_tensor(cell, dtype=self.site_basis_config.dtype, device=positions_t.device)
        edge_index_t = _long_tensor(edge_index, device=positions_t.device)
        atom_types_t = _long_tensor(atom_types, device=positions_t.device)
        shifts_t = None if shifts is None else _float_tensor(shifts, dtype=self.site_basis_config.dtype, device=positions_t.device)
        charges_t = None if charges is None else _float_tensor(charges, dtype=self.site_basis_config.dtype, device=positions_t.device)
        x_ij = edge_vectors_from_positions(positions_t, cell_t, edge_index_t, shifts=shifts_t)
        evaluator = self._runtime(positions_t.device)
        return evaluator(
            x_ij=x_ij,
            edge_index=edge_index_t,
            atom_types=atom_types_t,
            descriptors=self.descriptor_specs,
            charges=charges_t,
            aux_tensor_basis=aux_tensor_basis,
            real_if_scalar=real_if_scalar,
        )

    def compute_atomic_products(
        self,
        positions,
        cell,
        edge_index,
        atom_types,
        *,
        shifts = None,
        charges = None,
        aux_tensor_basis = None,
    ):
        positions_t = _float_tensor(positions, dtype=self.site_basis_config.dtype, device=self.device)
        cell_t = _float_tensor(cell, dtype=self.site_basis_config.dtype, device=positions_t.device)
        edge_index_t = _long_tensor(edge_index, device=positions_t.device)
        atom_types_t = _long_tensor(atom_types, device=positions_t.device)
        shifts_t = None if shifts is None else _float_tensor(shifts, dtype=self.site_basis_config.dtype, device=positions_t.device)
        charges_t = None if charges is None else _float_tensor(charges, dtype=self.site_basis_config.dtype, device=positions_t.device)
        x_ij = edge_vectors_from_positions(positions_t, cell_t, edge_index_t, shifts=shifts_t)
        evaluator = self._runtime(positions_t.device)
        return evaluator.atomic_products(
            x_ij=x_ij,
            edge_index=edge_index_t,
            atom_types=atom_types_t,
            descriptors=self.descriptor_specs,
            charges=charges_t,
            aux_tensor_basis=aux_tensor_basis,
        )

    def contract_atomic_products(
        self,
        products,
        *,
        real_if_scalar = True,
    ):
        return self.evaluator.contract_atomic_products(
            products,
            self.descriptor_specs,
            real_if_scalar=real_if_scalar,
        )

    def compute_with_gradients(
        self,
        positions,
        cell,
        edge_index,
        atom_types,
        *,
        shifts = None,
        charges = None,
        aux_tensor_basis = None,
        real_if_scalar = True,
    ):
        positions_t = _float_tensor(positions, dtype=self.site_basis_config.dtype, device=self.device)
        cell_t = _float_tensor(cell, dtype=self.site_basis_config.dtype, device=positions_t.device)
        edge_index_t = _long_tensor(edge_index, device=positions_t.device)
        atom_types_t = _long_tensor(atom_types, device=positions_t.device)
        shifts_t = None if shifts is None else _float_tensor(shifts, dtype=self.site_basis_config.dtype, device=positions_t.device)
        charges_t = None if charges is None else _float_tensor(charges, dtype=self.site_basis_config.dtype, device=positions_t.device)
        evaluator = self._runtime(positions_t.device)
        values, gradients = descriptor_gradients_wrt_positions(
            evaluator,
            positions_t,
            cell_t,
            edge_index_t,
            atom_types_t,
            self.descriptor_specs,
            shifts=shifts_t,
            charges=charges_t,
            aux_tensor_basis=aux_tensor_basis,
            real_if_scalar=real_if_scalar,
        )
        return DescriptorGradientResult(values=values, gradients=gradients)

    def compute_from_neighbor_data(
        self,
        neighbor_data,
        *,
        positions = None,
        charges = None,
        aux_tensor_basis = None,
        real_if_scalar = True,
    ):
        source_positions = neighbor_data.positions if positions is None else positions
        return self.compute(
            positions=source_positions,
            cell=neighbor_data.cell,
            edge_index=neighbor_data.edge_index,
            atom_types=neighbor_data.atom_types,
            shifts=neighbor_data.shifts,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            real_if_scalar=real_if_scalar,
        )

    def compute_atomic_products_from_neighbor_data(
        self,
        neighbor_data,
        *,
        positions = None,
        charges = None,
        aux_tensor_basis = None,
    ):
        source_positions = neighbor_data.positions if positions is None else positions
        return self.compute_atomic_products(
            positions=source_positions,
            cell=neighbor_data.cell,
            edge_index=neighbor_data.edge_index,
            atom_types=neighbor_data.atom_types,
            shifts=neighbor_data.shifts,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )

    def compute_gradients_from_neighbor_data(
        self,
        neighbor_data,
        *,
        positions = None,
        charges = None,
        aux_tensor_basis = None,
        real_if_scalar = True,
    ):
        source_positions = neighbor_data.positions if positions is None else positions
        return self.compute_with_gradients(
            positions=source_positions,
            cell=neighbor_data.cell,
            edge_index=neighbor_data.edge_index,
            atom_types=neighbor_data.atom_types,
            shifts=neighbor_data.shifts,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            real_if_scalar=real_if_scalar,
        )

    def compute_from_ase_atoms(
        self,
        atoms,
        *,
        cutoff,
        type_map,
        charges = None,
        aux_tensor_basis = None,
        real_if_scalar = True,
    ):
        neighbor_data = neighbor_data_from_ase_atoms(atoms, cutoff=float(cutoff), type_map=dict(type_map))
        return self.compute_from_neighbor_data(
            neighbor_data,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            real_if_scalar=real_if_scalar,
        )

    def compute_atomic_products_from_ase_atoms(
        self,
        atoms,
        *,
        cutoff,
        type_map,
        charges = None,
        aux_tensor_basis = None,
    ):
        neighbor_data = neighbor_data_from_ase_atoms(atoms, cutoff=float(cutoff), type_map=dict(type_map))
        return self.compute_atomic_products_from_neighbor_data(
            neighbor_data,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )

    def compute_gradients_from_ase_atoms(
        self,
        atoms,
        *,
        cutoff,
        type_map,
        charges = None,
        aux_tensor_basis = None,
        real_if_scalar = True,
    ):
        neighbor_data = neighbor_data_from_ase_atoms(atoms, cutoff=float(cutoff), type_map=dict(type_map))
        return self.compute_gradients_from_neighbor_data(
            neighbor_data,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            real_if_scalar=real_if_scalar,
        )


def _as_rank_tuple(values, ranks, *, name, dtype=int):
    if isinstance(values, (int, float)):
        return tuple(dtype(values) for _ in ranks)
    vals = tuple(dtype(v) for v in values)
    if len(vals) == 1 and len(tuple(ranks)) != 1:
        return tuple(vals[0] for _ in ranks)
    if len(vals) != len(tuple(ranks)):
        raise ValueError(f"{name} must be scalar, length 1, or match ranks length {len(tuple(ranks))}.")
    return vals


def _load_descriptor_config_file(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without optional yaml.
            raise ImportError("Reading YAML descriptor configs requires PyYAML.") from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("YE3T descriptor config files must contain a mapping.")
    return dict(payload)


def _normalize_config(config = None, **kwargs):
    if config is None:
        cfg = {}
    elif isinstance(config, Mapping):
        cfg = dict(config)
        config_file = cfg.pop("descriptor_config_file", cfg.pop("config_file", None))
        if config_file is not None:
            cfg = {**_load_descriptor_config_file(config_file), **cfg}
    elif isinstance(config, (str, Path)):
        cfg = _load_descriptor_config_file(config)
    else:
        raise TypeError("config must be a mapping, JSON/YAML path, or None.")
    if "descriptor" in cfg and isinstance(cfg["descriptor"], Mapping):
        nested = dict(cfg["descriptor"])
        nested.update({key: value for key, value in cfg.items() if key not in {"descriptor", "model"}})
        cfg = nested
    cfg.update(kwargs)
    if "representation" not in cfg:
        if "ye3t_spec_file" in cfg:
            cfg["representation"] = {"ye3t_spec_file": cfg["ye3t_spec_file"]}
        elif "ye3t_spec" in cfg:
            cfg["representation"] = {"ye3t_spec": cfg["ye3t_spec"]}
    return cfg


def _catalogue_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_SCALAR_COORDINATE_COMPILER_FIELDS = {
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


def _normalize_scalar_coordinate_compiler(payload):
    compiler = dict(payload or {})
    unknown = set(compiler) - _SCALAR_COORDINATE_COMPILER_FIELDS
    if unknown:
        raise ValueError(
            "ordinary_scalar_catalogue compiler has unsupported fields: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    compiler.setdefault("coordinate_contract", "pace_compatible_exact")
    compiler.setdefault("coefficient_materialization", "exact")
    compiler.setdefault("constructor_backend", "python")
    membership_mode = str(compiler.get("membership_mode", "auto")).strip().lower()
    coordinate_contract = str(compiler["coordinate_contract"]).strip().lower()
    materialization = str(compiler["coefficient_materialization"]).strip().lower()
    constructor_backend = str(compiler["constructor_backend"]).strip().lower()
    if membership_mode not in {"auto", "exact", "constructive_factorized"}:
        raise ValueError("ordinary_scalar_catalogue has an invalid membership_mode.")
    if coordinate_contract not in {
        "pace_compatible_exact",
        "pace_frozen_exact",
        "native_compiled",
    }:
        raise ValueError("ordinary_scalar_catalogue has an invalid coordinate_contract.")
    if materialization not in {"exact", "certified_numeric"}:
        raise ValueError(
            "ordinary_scalar_catalogue must explicitly select exact or "
            "certified_numeric coefficient materialization."
        )
    if constructor_backend not in {"python", "cpp", "cpp_low_memory"}:
        raise ValueError("ordinary_scalar_catalogue has an invalid constructor_backend.")
    if (
        coordinate_contract in {"pace_compatible_exact", "pace_frozen_exact"}
        and materialization != "exact"
    ):
        raise ValueError("PACE-compatible scalar coordinates must be exact.")
    if coordinate_contract == "pace_frozen_exact" and constructor_backend != "cpp":
        raise ValueError("pace_frozen_exact requires constructor_backend='cpp'.")
    canonical_strings = {
        "coordinate_contract": coordinate_contract,
        "coefficient_materialization": materialization,
        "constructor_backend": constructor_backend,
    }
    if "membership_mode" in compiler:
        canonical_strings["membership_mode"] = membership_mode
    for key, canonical in canonical_strings.items():
        if compiler[key] != canonical:
            raise ValueError(
                f"ordinary_scalar_catalogue compiler field {key!r} is not canonical."
            )
    for key in ("outer_coefficient_tolerance", "collection_tolerance"):
        if key in compiler:
            value = float(compiler[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "ordinary_scalar_catalogue compiler tolerances must be finite "
                    "and non-negative."
                )
    for key in (
        "maximum_unique_monomials",
        "maximum_term_contributions",
        "maximum_exact_symbolic_bytes",
        "maximum_coordinate_bytes",
    ):
        if key in compiler:
            value = compiler[key]
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(
                    "ordinary_scalar_catalogue compiler resource limits must be "
                    "positive integers."
                )
    return compiler


def _normalize_serialized_scalar_coordinate(payload, label, expected, compiler):
    if not isinstance(payload, Mapping):
        raise TypeError("compiled_coordinate must be a mapping.")
    allowed = {
        "schema",
        "label",
        "magnetic_tuples",
        "coefficients",
        "certificate",
        "payload_sha256",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            "compiled_coordinate has unsupported fields: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    if str(payload.get("schema", "")) != "ye3t_scalar_ace_coordinate_table_v1":
        raise ValueError("compiled_coordinate has an unsupported schema.")
    serialized_label = CompactLabel.from_dict(payload.get("label", {}))
    if serialized_label != label:
        raise ValueError("compiled_coordinate label differs from its catalogue row.")
    raw_magnetic_tuples = tuple(payload.get("magnetic_tuples", ()))
    if any(
        not isinstance(row, (tuple, list))
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in row
        )
        for row in raw_magnetic_tuples
    ):
        raise TypeError("compiled_coordinate magnetic tuples must contain integers.")
    magnetic_tuples = tuple(
        tuple(int(value) for value in row) for row in raw_magnetic_tuples
    )
    coefficients = tuple(
        tuple(float(value) for value in pair)
        for pair in payload.get("coefficients", ())
    )
    if not magnetic_tuples or len(magnetic_tuples) != len(coefficients):
        raise ValueError("compiled_coordinate requires matching nonempty terms.")
    if any(len(row) != int(label.rank) for row in magnetic_tuples):
        raise ValueError("compiled_coordinate magnetic tuple has the wrong rank.")
    if len(set(magnetic_tuples)) != len(magnetic_tuples):
        raise ValueError("compiled_coordinate magnetic tuples must be unique.")
    if any(
        any(abs(int(m_value)) > int(l_value) for m_value, l_value in zip(row, label.l_tuple))
        or sum(int(m_value) for m_value in row) != 0
        for row in magnetic_tuples
    ):
        raise ValueError(
            "compiled_coordinate magnetic tuples violate angular support or M_R=0."
        )
    if any(len(pair) != 2 for pair in coefficients) or not all(
        np.isfinite(value) for pair in coefficients for value in pair
    ):
        raise ValueError("compiled_coordinate coefficients must be finite complex pairs.")
    certificate = dict(payload.get("certificate", {}))
    if certificate.get("passed") is not True:
        raise ValueError("compiled_coordinate compiler certificate did not pass.")
    if CompactLabel.from_dict(certificate.get("label", {})) != label:
        raise ValueError("compiled_coordinate certificate label differs from its row.")
    if (
        int(label.rank) > 8
        and str(compiler.get("membership_mode", "auto"))
        != "constructive_factorized"
    ):
        raise ValueError(
            "serialized scalar coordinates above rank eight require explicit "
            "constructive_factorized membership."
        )
    certificate_payload = dict(certificate)
    certificate_sha256 = str(certificate_payload.pop("certificate_sha256", ""))
    if not certificate_sha256 or _catalogue_sha256(certificate_payload) != certificate_sha256:
        raise ValueError("compiled_coordinate certificate SHA-256 does not match.")
    for key, value in expected.items():
        if str(certificate.get(key, "")) != str(value):
            raise ValueError("compiled_coordinate certificate identity does not match.")
    if str(certificate.get("coordinate_contract", "")) != str(
        compiler["coordinate_contract"]
    ):
        raise ValueError("compiled_coordinate contract differs from compiler options.")
    expected_basis = (
        "certified_numeric_orthogonal_occupancy"
        if compiler["coefficient_materialization"] == "certified_numeric"
        else "exact_symbolic_independent_occupancy"
    )
    if str(certificate.get("basis_realization", "")) != expected_basis:
        raise ValueError("compiled_coordinate basis realization differs from its compiler options.")
    normalized = {
        "schema": "ye3t_scalar_ace_coordinate_table_v1",
        "label": label.to_dict(),
        "magnetic_tuples": [list(row) for row in magnetic_tuples],
        "coefficients": [list(pair) for pair in coefficients],
        "certificate": certificate,
    }
    payload_sha256 = _catalogue_sha256(normalized)
    if str(payload.get("payload_sha256", "")) != payload_sha256:
        raise ValueError("compiled_coordinate payload SHA-256 does not match.")
    return {**normalized, "payload_sha256": payload_sha256}


def _normalize_ordinary_scalar_catalogue(payload):
    if not isinstance(payload, Mapping):
        raise TypeError("ordinary_scalar_catalogue must be a mapping.")
    allowed = {
        "schema",
        "profile_id",
        "feature_ids",
        "membership_sha256",
        "rows",
        "compiler",
        "application_sha256",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            "ordinary_scalar_catalogue has unsupported fields: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    schema = str(payload.get("schema", ""))
    if schema not in {
        "ye3t_ordinary_scalar_catalogue_v1",
        "ye3t_ordinary_scalar_catalogue_v2",
    }:
        raise ValueError(
            "ordinary_scalar_catalogue.schema must be "
            "ye3t_ordinary_scalar_catalogue_v1 or v2."
        )
    profile_id = str(payload.get("profile_id", "")).strip()
    if not profile_id:
        raise ValueError("ordinary_scalar_catalogue.profile_id must be nonempty.")
    feature_ids = tuple(str(value) for value in payload.get("feature_ids", ()))
    rows = tuple(payload.get("rows", ()))
    if not feature_ids or len(feature_ids) != len(rows):
        raise ValueError(
            "ordinary_scalar_catalogue requires one ordered row per feature_id."
        )
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("ordinary_scalar_catalogue feature_ids must be unique.")

    compiler = _normalize_scalar_coordinate_compiler(payload.get("compiler", {}))

    expected_fields = {
        "certificate_sha256",
        "coordinate_identity_sha256",
        "collected_coefficient_sha256",
        "factorized_schedule_sha256",
        "blockwise_plan_convention_hash",
    }
    normalized_rows = []
    labels = []
    for feature_id, raw_row in zip(feature_ids, rows):
        if not isinstance(raw_row, Mapping):
            raise TypeError("ordinary_scalar_catalogue rows must be mappings.")
        row_allowed = {
            "feature_id",
            "compact_label",
            "expected_compiler",
        }
        if schema.endswith("_v2"):
            row_allowed.update({"compiler", "compiled_coordinate"})
        row_unknown = set(raw_row) - row_allowed
        if row_unknown:
            raise ValueError(
                "ordinary_scalar_catalogue row has unsupported fields: "
                + ", ".join(sorted(str(key) for key in row_unknown))
            )
        if str(raw_row.get("feature_id", "")) != feature_id:
            raise ValueError(
                "ordinary_scalar_catalogue row order must match feature_ids exactly."
            )
        label = CompactLabel.from_dict(raw_row.get("compact_label", {}))
        if int(label.L_R) != 0 or sum(int(value) for value in label.l_tuple) % 2:
            raise ValueError(
                "ordinary_scalar_catalogue accepts only even-parity scalar labels."
            )
        expected = raw_row.get("expected_compiler", {})
        if not isinstance(expected, Mapping) or set(expected) != expected_fields:
            raise ValueError(
                "ordinary_scalar_catalogue expected_compiler must contain exactly "
                + ", ".join(sorted(expected_fields))
            )
        normalized_expected = {
            key: str(expected[key]) for key in sorted(expected_fields)
        }
        normalized_row = {
            "feature_id": feature_id,
            "compact_label": label.to_dict(),
            "expected_compiler": normalized_expected,
        }
        if schema.endswith("_v2"):
            row_compiler = _normalize_scalar_coordinate_compiler(
                raw_row.get("compiler", compiler)
            )
            normalized_row["compiler"] = row_compiler
            normalized_row["compiled_coordinate"] = (
                _normalize_serialized_scalar_coordinate(
                    raw_row.get("compiled_coordinate", {}),
                    label,
                    normalized_expected,
                    row_compiler,
                )
            )
        normalized_rows.append(normalized_row)
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError("ordinary_scalar_catalogue compact labels must be unique.")

    membership_payload = {
        "profile_id": profile_id,
        "feature_ids": list(feature_ids),
    }
    membership_sha256 = _catalogue_sha256(membership_payload)
    if str(payload.get("membership_sha256", "")) != membership_sha256:
        raise ValueError("ordinary_scalar_catalogue membership_sha256 does not match.")
    application_payload = {
        "schema": schema,
        "profile_id": profile_id,
        "feature_ids": list(feature_ids),
        "rows": normalized_rows,
        "compiler": compiler,
    }
    application_sha256 = _catalogue_sha256(application_payload)
    if str(payload.get("application_sha256", "")) != application_sha256:
        raise ValueError("ordinary_scalar_catalogue application_sha256 does not match.")
    normalized = {
        **application_payload,
        "membership_sha256": membership_sha256,
        "application_sha256": application_sha256,
    }
    compiler_request = (
        {
            "mode": "serialized",
            "coordinates": [
                {
                    "compiler": dict(row["compiler"]),
                    "compiled_coordinate": dict(row["compiled_coordinate"]),
                }
                for row in normalized_rows
            ],
        }
        if schema.endswith("_v2")
        else {"mode": "all", "options": dict(compiler)}
    )
    return normalized, tuple(labels), compiler_request


def _validate_ordinary_scalar_catalogue_runtime(catalogue, ace_descriptor):
    labels = tuple(ace_descriptor.calculator.labels)
    expected_labels = tuple(
        CompactLabel.from_dict(row["compact_label"])
        for row in catalogue["rows"]
    )
    if labels != expected_labels:
        raise RuntimeError(
            "ordinary_scalar_catalogue labels were filtered, reordered, or replaced."
        )
    settings = ace_descriptor.settings
    for label in labels:
        if int(label.rank) not in tuple(int(value) for value in settings.ranks):
            raise ValueError(
                f"Catalogue rank {label.rank} is absent from descriptor settings."
            )
        rank_index = tuple(int(value) for value in settings.ranks).index(
            int(label.rank)
        )
        if max(int(value) for value in label.n_tuple) > int(
            settings.nmax[rank_index]
        ):
            raise ValueError("Catalogue radial index exceeds descriptor nmax.")
        if any(
            int(value) < int(settings.lmin[rank_index])
            or int(value) > int(settings.lmax[rank_index])
            for value in label.l_tuple
        ):
            raise ValueError("Catalogue angular index is outside descriptor l bounds.")

    library_metadata = dict(ace_descriptor.calculator.coupling_library.metadata)
    compiler_metadata = dict(
        library_metadata.get("scalar_coordinate_compiler", {})
    )
    compiled_records = tuple(compiler_metadata.get("coordinates", ()))
    if len(compiled_records) != len(catalogue["rows"]):
        raise RuntimeError(
            "ordinary_scalar_catalogue compiler record count does not match."
        )
    compact_records = []
    for row, record in zip(catalogue["rows"], compiled_records):
        if record.get("label") != row["compact_label"]:
            raise RuntimeError(
                "ordinary_scalar_catalogue compiler record order does not match."
            )
        certificate = dict(record.get("certificate", {}))
        expected = row["expected_compiler"]
        for key, expected_value in expected.items():
            if str(certificate.get(key, "")) != str(expected_value):
                raise RuntimeError(
                    "ordinary_scalar_catalogue compiler identity mismatch for "
                    f"{row['feature_id']!r}: {key}."
                )
        compact_records.append(
            {
                "feature_id": row["feature_id"],
                "certificate_sha256": certificate["certificate_sha256"],
                "coordinate_identity_sha256": certificate[
                    "coordinate_identity_sha256"
                ],
                "collected_coefficient_sha256": certificate[
                    "collected_coefficient_sha256"
                ],
                "factorized_schedule_sha256": certificate[
                    "factorized_schedule_sha256"
                ],
                "blockwise_plan_convention_hash": certificate[
                    "blockwise_plan_convention_hash"
                ],
            }
        )
    return tuple(compact_records)


def _normalize_a_s_subselection(value=None):
    return normalize_a_s_subselection(value)


def _a_s_readout_for_subselection(subselection, requested_readout=None):
    subselection = _normalize_a_s_subselection(subselection)
    if subselection is None:
        return requested_readout
    requested = None if requested_readout is None else str(requested_readout).strip().lower()
    defaults = {
        "fully_symmetric": "symmetric_linear",
        "fully_antisymmetric": "antisymmetric_quadratic",
        "equivariant": "slot_equivariant",
    }
    allowed = {
        "fully_symmetric": {"symmetric_linear"},
        "fully_antisymmetric": {"antisymmetric_quadratic"},
        "equivariant": {
            "symmetric_linear",
            "slot_equivariant",
            "character_quadratic",
            "ye3_quadratic",
            "ye3_power",
            "ye3_slot_specht_power",
        },
    }
    out = defaults[subselection] if requested is None else requested
    if out not in allowed[subselection]:
        raise ValueError(
            f"A_s representation_subselection={subselection!r} does not support readout_mode={out!r}; "
            f"allowed readout modes are {sorted(allowed[subselection])}."
        )
    return out


def _config_bool(value, *, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", ""}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}.")


def _a_s_slot_specht_partitions_from_payload(value):
    if value is None:
        return tuple()
    if isinstance(value, (str, bytes)):
        raise TypeError("A_s slot_specht_partitions must be a sequence of integer sequences, not a string.")
    return tuple(tuple(int(part) for part in partition) for partition in value)


def _a_s_role_coordinate_policy(cfg, representation):
    metadata = dict(getattr(representation, "metadata", {}) or {})
    role_payload = {}
    shared_spec_report = None
    shared_spec_payload = None
    if isinstance(cfg.get("ye3t_spec"), Mapping):
        shared_spec_payload = dict(cfg["ye3t_spec"])
    elif "ye3t_spec_file" in cfg:
        shared_spec_payload = YE3TSpec.from_file(cfg["ye3t_spec_file"]).to_dict()
    if shared_spec_payload is not None:
        shared_spec = YE3TSpec.from_dict(shared_spec_payload)
        role_payload.update(dict(shared_spec.carrier_options))
        shared_spec_report = shared_spec.carrier_policy_report()
    if isinstance(cfg.get("A_s_role_coordinate"), Mapping):
        role_payload.update(dict(cfg["A_s_role_coordinate"]))
    if isinstance(cfg.get("role_coordinate"), Mapping):
        role_payload.update(dict(cfg["role_coordinate"]))
    role_payload.update(
        {
            key: cfg[key]
            for key in (
                "role_coordinate_policy",
                "A_s_role_coordinate_policy",
                "identical_role_filters_declared",
                "role_filters_identical",
                "identical_role_filters",
                "role_coordinate_discarded_before_young_projection",
                "discard_role_coordinate",
            )
            if key in cfg
        }
    )
    role_payload.update(
        {
            key: metadata[key]
            for key in (
                "role_coordinate_policy",
                "A_s_role_coordinate_policy",
                "identical_role_filters_declared",
                "role_filters_identical",
                "identical_role_filters",
                "role_coordinate_discarded_before_young_projection",
                "discard_role_coordinate",
            )
            if key in metadata and key not in role_payload
        }
    )
    report_payload = role_payload.get("A_s_role_coordinate_policy", None)
    if isinstance(report_payload, Mapping):
        report_payload = dict(report_payload)
        for key in (
            "role_coordinate_policy",
            "identical_role_filters_declared",
            "role_coordinate_discarded_before_young_projection",
            "discard_role_coordinate",
        ):
            if key in report_payload:
                role_payload.setdefault(key, report_payload[key])
        role_payload.pop("A_s_role_coordinate_policy", None)
    policy = str(
        role_payload.get(
            "role_coordinate_policy",
            role_payload.get("A_s_role_coordinate_policy", "role_resolved"),
        )
    ).strip().lower()
    if policy in {"role_resolved", "retained", "retain", "slot_resolved", "explicit_role"}:
        identical = _config_bool(
            role_payload.get(
                "identical_role_filters_declared",
                role_payload.get("role_filters_identical", role_payload.get("identical_role_filters", False)),
            )
        )
        discarded = _config_bool(
            role_payload.get(
                "role_coordinate_discarded_before_young_projection",
                role_payload.get("discard_role_coordinate", False),
            )
        )
    elif policy in {"collapsed", "commutative_density", "identical_filters_discarded", "discarded"}:
        identical = True
        discarded = True
    else:
        raise ValueError(
            "A_s role_coordinate_policy must be role_resolved/retained or collapsed/commutative_density; "
            f"got {policy!r}."
        )
    return {
        "role_coordinate_policy": policy,
        "identical_role_filters_declared": bool(identical),
        "role_coordinate_discarded_before_young_projection": bool(discarded),
        "collapse_condition": (
            "if all role filters are identical and the role coordinate is discarded before Young coupling, "
            "the role module becomes trivial and nontrivial Young sectors vanish"
        ),
        "source": "descriptor_config_or_representation_metadata",
        "shared_spec_carrier_policy_report": shared_spec_report,
    }


def _validate_A_s_role_coordinate_policy_for_partitions(policy_report, partitions):
    partitions = _a_s_slot_specht_partitions_from_payload(partitions)
    nontrivial = tuple(partition for partition in partitions if len(partition) > 1)
    collapse_declared = bool(
        policy_report["identical_role_filters_declared"]
        and policy_report["role_coordinate_discarded_before_young_projection"]
    )
    if nontrivial and collapse_declared:
        raise ValueError(
            "Nontrivial A_s Young sectors require a retained role coordinate before Young coupling. "
            "The supplied config declares identical role filters and discards the role coordinate, "
            "which collapses the role module to the trivial representation and makes nontrivial "
            f"slot-Specht partitions vanish: {nontrivial!r}."
        )
    return {
        **dict(policy_report),
        "slot_specht_partitions": partitions,
        "nontrivial_slot_specht_partitions": nontrivial,
        "nontrivial_sector_requested": bool(nontrivial),
        "collapse_declared": bool(collapse_declared),
        "passed": bool((not nontrivial) or not collapse_declared),
    }


def _normalize_feature_filters(feature_filters):
    if feature_filters is None:
        return ()
    if isinstance(feature_filters, Mapping):
        return (dict(feature_filters),)
    return tuple(dict(item) for item in feature_filters)


def _value_tuple(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _label_matches_filter(label, spec):
    if "rank" in spec and int(label.rank) != int(spec["rank"]):
        return False
    if "ranks" in spec and int(label.rank) not in {int(v) for v in _value_tuple(spec["ranks"])}:
        return False
    if "n_tuple" in spec and tuple(label.n_tuple) != tuple(int(v) for v in spec["n_tuple"]):
        return False
    if "n_tuples" in spec and tuple(label.n_tuple) not in {tuple(int(x) for x in row) for row in spec["n_tuples"]}:
        return False
    if "l_tuple" in spec and tuple(label.l_tuple) != tuple(int(v) for v in spec["l_tuple"]):
        return False
    if "l_tuples" in spec and tuple(label.l_tuple) not in {tuple(int(x) for x in row) for row in spec["l_tuples"]}:
        return False
    if "l_values" in spec and not all(int(l) in {int(v) for v in _value_tuple(spec["l_values"])} for l in label.l_tuple):
        return False
    if "contains_l" in spec and not any(int(l) in {int(v) for v in _value_tuple(spec["contains_l"])} for l in label.l_tuple):
        return False
    if "internal_Ls" in spec and tuple(label.internal_Ls) != tuple(int(v) for v in spec["internal_Ls"]):
        return False
    if "tree_type" in spec and str(label.tree_type) != str(spec["tree_type"]):
        return False
    basis_key = tuple(label.basis_key)
    if "basis_key" in spec and basis_key != tuple(spec["basis_key"]):
        return False
    allowed_basis = spec.get("basis_keys", spec.get("pair_orbit_symmetries", None))
    if allowed_basis is not None:
        allowed = {tuple(v) if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) else (v,) for v in allowed_basis}
        if basis_key not in allowed and tuple(str(x) for x in basis_key) not in allowed:
            return False
    pair_orbit = spec.get("pair_orbit_symmetry", None)
    if pair_orbit is not None:
        candidate = tuple(pair_orbit) if isinstance(pair_orbit, Sequence) and not isinstance(pair_orbit, (str, bytes)) else (pair_orbit,)
        if basis_key != candidate and tuple(str(x) for x in basis_key) != candidate:
            return False
    return True


def filter_compact_labels_by_specs(labels, feature_filters=None):
    """Filter compact labels by rank, radial/angular tuples, and basis keys.

    ``feature_filters=None`` keeps the exhaustive label set selected by
    ``DescriptorGenerationSettings`` and ``basis_mode``.
    """

    filters = _normalize_feature_filters(feature_filters)
    labels = tuple(labels)
    if not filters:
        return labels
    out = []
    seen = set()
    for label in labels:
        if any(_label_matches_filter(label, spec) for spec in filters):
            if label not in seen:
                seen.add(label)
                out.append(label)
    return tuple(out)


def _descriptor_spec_matches_filter(descriptor_spec, spec):
    if not _label_matches_filter(descriptor_spec.label, spec):
        return False
    if "eta" in spec or "eta_values" in spec:
        allowed = {None if v is None else int(v) for v in _value_tuple(spec.get("eta", spec.get("eta_values")))}
        etas = {ch.eta for ch in descriptor_spec.channels}
        if not etas <= allowed:
            return False
    if "center_mu" in spec and not all(int(ch.mu0) == int(spec["center_mu"]) for ch in descriptor_spec.channels):
        return False
    if "center_mu_values" in spec:
        allowed = {int(v) for v in _value_tuple(spec["center_mu_values"])}
        if not all(int(ch.mu0) in allowed for ch in descriptor_spec.channels):
            return False
    if "neighbor_mu_values" in spec:
        allowed = {int(v) for v in _value_tuple(spec["neighbor_mu_values"])}
        if not all(int(ch.mu) in allowed for ch in descriptor_spec.channels):
            return False
    return True


def filter_descriptor_specs_by_specs(descriptor_specs, feature_filters=None):
    filters = _normalize_feature_filters(feature_filters)
    descriptor_specs = tuple(descriptor_specs)
    if not filters:
        return descriptor_specs
    return tuple(spec for spec in descriptor_specs if any(_descriptor_spec_matches_filter(spec, item) for item in filters))


def _explicit_tuple_options(spec, singular, plural):
    if singular in spec:
        return (tuple(int(v) for v in spec[singular]),)
    if plural in spec:
        return tuple(tuple(int(v) for v in row) for row in spec[plural])
    return None


def _tuple_fits_rank_settings(values, settings, rank):
    try:
        rank_index = settings.rank_index(int(rank))
    except ValueError:
        return False
    if len(values) != int(rank):
        return False
    return True


def _direct_compact_labels_from_feature_filters(settings, feature_filters):
    """Generate compact labels directly for explicit filtered sectors.

    This avoids enumerating the full ``nmax x lmax`` product space before
    filtering. It is intended for high-rank symmetric-power subselections where
    the requested ``n_tuple``/``l_tuple`` sectors are known up front.
    """

    filters = _normalize_feature_filters(feature_filters)
    if not filters:
        return None
    out = []
    seen = set()
    for spec in filters:
        n_options = _explicit_tuple_options(spec, "n_tuple", "n_tuples")
        l_options = _explicit_tuple_options(spec, "l_tuple", "l_tuples")
        if n_options is None or l_options is None:
            return None
        for n_tuple in n_options:
            for l_tuple in l_options:
                rank = int(spec.get("rank", len(n_tuple)))
                if not _tuple_fits_rank_settings(n_tuple, settings, rank):
                    return None
                if not _tuple_fits_rank_settings(l_tuple, settings, rank):
                    return None
                rank_index = settings.rank_index(rank)
                if any(int(n) < 1 or int(n) > int(settings.nmax[rank_index]) for n in n_tuple):
                    continue
                if any(
                    int(l) < int(settings.lmin[rank_index]) or int(l) > int(settings.lmax[rank_index])
                    for l in l_tuple
                ):
                    continue
                if "basis_key" in spec and "internal_Ls" in spec:
                    label = CompactLabel(
                        tuple(n_tuple),
                        tuple(l_tuple),
                        tuple(int(v) for v in spec["internal_Ls"]),
                        str(spec.get("tree_type", settings.tree_type)),
                        tuple(spec["basis_key"]),
                    )
                    label = normalize_compact_label(label)
                    if settings.parity_filter == "natural" and not label_has_natural_parity(label, L_R=settings.L_R):
                        continue
                    if _label_matches_filter(label, spec) and label not in seen:
                        seen.add(label)
                        out.append(label)
                    continue
                auto_label = _auto_symmetric_power_compact_label_from_filter(settings, spec, n_tuple, l_tuple)
                if auto_label is not None:
                    if auto_label not in seen:
                        seen.add(auto_label)
                        out.append(auto_label)
                    continue
                report = blockwise_symmetric_power_labels(
                    content=tuple(n_tuple),
                    input_Ls=tuple(l_tuple),
                    target_L=int(settings.L_R),
                    tree_schedule=str(spec.get("tree_type", settings.tree_type)),
                    label_strategy=str(
                        spec.get(
                            "blockwise_label_strategy",
                            "representative" if settings.max_labels_per_rank == 1 else "exhaustive",
                        )
                    ),
                    max_labels=spec.get("max_labels", settings.max_labels_per_rank),
                    validation_scope="counts",
                    metadata={"consumer": "ye3t_ace.ace.descriptors.direct_feature_filter_labels"},
                )
                for label in report["labels"]:
                    label = normalize_compact_label(label)
                    if settings.parity_filter == "natural" and not label_has_natural_parity(label, L_R=settings.L_R):
                        continue
                    if not _label_matches_filter(label, spec):
                        continue
                    if label not in seen:
                        seen.add(label)
                        out.append(label)
    return tuple(
        sorted(
            out,
            key=lambda x: (x.rank, x.n_tuple, x.l_tuple, x.internal_Ls, x.tree_type, x.basis_key),
        )
    )


def _run_lengths(values):
    runs = []
    start = 0
    values = tuple(values)
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[stop] == values[start]:
            stop += 1
        runs.append((start, stop, values[start]))
        start = stop
    return tuple(runs)


def _auto_symmetric_power_compact_label_from_filter(settings, spec, n_tuple, l_tuple):
    mode = str(spec.get("symmetric_power_label", spec.get("symmetric_power_mode", "auto"))).strip().lower()
    if mode in {"off", "false", "0", "none"}:
        return None
    rank = len(n_tuple)
    if int(settings.L_R) == 0:
        n_runs = _run_lengths(n_tuple)
        l_runs = _run_lengths(l_tuple)
        if len(n_runs) == 1 and len(l_runs) == 1:
            input_l = int(l_runs[0][2])
            if input_l == 1 and int(rank) >= 2 and int(rank) % 2 == 0:
                return CompactLabel(
                    tuple(n_tuple),
                    tuple(l_tuple),
                    (0,),
                    str(settings.tree_type),
                    ("sym", 0, 0),
                )
        if len(n_runs) == 2 and len(l_runs) == 1:
            left = int(n_runs[0][1] - n_runs[0][0])
            right = int(n_runs[1][1] - n_runs[1][0])
            input_l = int(l_runs[0][2])
            if left == right and input_l > 0:
                block_L = int(left * input_l)
                return CompactLabel(
                    tuple(n_tuple),
                    tuple(l_tuple),
                    (block_L, block_L, 0),
                    str(settings.tree_type),
                    ("node", ("sym", block_L, 0), ("sym", block_L, 0)),
                )
    if int(settings.L_R) == int(rank * int(l_tuple[0])) and len(set(zip(n_tuple, l_tuple))) == 1:
        output_L = int(settings.L_R)
        return CompactLabel(
            tuple(n_tuple),
            tuple(l_tuple),
            (output_L,),
            str(settings.tree_type),
            ("sym", output_L, 0),
        )
    return None


def _factorized_policy_from_representation(representation, cfg):
    explicit = cfg.get("factorized_descriptor_runtime_policy", cfg.get("factorized_runtime_policy", None))
    if explicit is not None:
        return explicit
    if representation.basis_mode not in {"symmetric_power_subselection", "symmetric_power"}:
        return None
    policy = str(representation.fast_path_policy).strip().lower()
    if policy in {"disable", "disabled", "off", "false", "0"}:
        return "disable"
    if policy in {"force", "require", "required", "strict"}:
        return "require" if policy in {"require", "required", "strict"} else "force"
    return "auto"


def _descriptor_basis_mode_from_representation(representation, cfg):
    if representation.basis_mode in {"symmetric_power_subselection", "symmetric_power"}:
        return cfg.get("descriptor_basis_mode", None)
    return representation.basis_mode


def _read_ase_path(path, index=":"):
    try:
        from ase.io import read
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("ASE is required to read descriptor input files.") from exc
    data = read(str(path), index=index)
    return list(data) if isinstance(data, list) else [data]


def read_ase_structures(paths, *, index=":"):
    """Read one or more structure files with ASE and return a flat list."""

    if isinstance(paths, (str, Path)):
        paths = [paths]
    structures = []
    for path in paths:
        structures.extend(_read_ase_path(path, index=index))
    return structures


def _atoms_symbols(atoms):
    return tuple(str(symbol) for symbol in atoms.get_chemical_symbols())


def _atoms_cell(atoms):
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    if cell.shape != (3, 3):
        return np.zeros((3, 3), dtype=float)
    return cell


def _extract_charges_from_atoms(atoms, source):
    if source is None:
        return None
    if source == "initial_charges":
        values = atoms.get_initial_charges()
    elif source == "charges":
        values = atoms.get_charges()
    elif isinstance(source, str) and source.startswith("array:"):
        values = atoms.arrays[source.split(":", 1)[1]]
    elif isinstance(source, str) and source in atoms.arrays:
        values = atoms.arrays[source]
    else:
        values = source(atoms) if callable(source) else source
    return np.asarray(values, dtype=float).reshape(-1)


def _extract_magnetic_vectors_from_atoms(atoms, source):
    if source is None:
        return None
    if source == "initial_magmoms":
        values = atoms.get_initial_magnetic_moments()
    elif isinstance(source, str) and source.startswith("array:"):
        values = atoms.arrays[source.split(":", 1)[1]]
    elif isinstance(source, str) and source in atoms.arrays:
        values = atoms.arrays[source]
    else:
        values = source(atoms) if callable(source) else source
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        raise ValueError("Non-collinear magnetic descriptors require per-atom vectors with shape [n_atoms, 3].")
    if arr.shape[1] != 3:
        raise ValueError("Magnetic vectors must have shape [n_atoms, 3].")
    return arr


def build_vdw_site_basis_config(
    elements,
    *,
    type_map=None,
    nradmax=8,
    lmax=2,
    kmax=0,
    charge_mode="none",
    charge_bounds=None,
    cutoff_scale=1.0,
    lambda_scale=1.0,
    min_cutoff=1.5,
    dtype=torch.float64,
):
    """Build a ``SiteBasisConfig`` from VdW-radii-derived ordered-pair defaults."""

    resolved_type_map = build_explicit_type_map(elements) if type_map is None else dict(type_map)
    possible_types = tuple(int(resolved_type_map[str(elem)]) for elem in elements)
    rc_by_pair, lmbda_by_pair, _ = vdw_scaled_bond_defaults(
        elements,
        type_map=resolved_type_map,
        cutoff_scale=cutoff_scale,
        lambda_scale=lambda_scale,
        min_cutoff=min_cutoff,
    )
    kwargs = {} if charge_bounds is None else charge_bounds.as_site_basis_kwargs()
    return SiteBasisConfig(
        rc=ordered_pair_values(rc_by_pair, possible_types=possible_types, name="rc"),
        lmbda=ordered_pair_values(lmbda_by_pair, possible_types=possible_types, name="lmbda"),
        nradmax=int(nradmax),
        lmax=int(lmax),
        kmax=int(kmax),
        possible_types=possible_types,
        charge_mode=str(charge_mode),
        dtype=dtype,
        complex_dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
        **kwargs,
    )


@recordclass(('values_by_structure', 'row_slices', 'elements', 'type_map', 'feature_keys'), frozen=True)
class ACEDescriptorBatch:
    """Container returned by ``ACEDescriptor.create_many`` when requested."""

    def as_concatenated(self):
        if not self.values_by_structure:
            return torch.empty((0, 0), dtype=torch.float64)
        return torch.cat(tuple(self.values_by_structure), dim=0)


class ACEDescriptor:
    """Hands-off ACE descriptor generator for ASE structures.

    The class is intentionally similar in spirit to descriptor front-ends such
    as ``dscribe.SOAP``: construct the descriptor inventory once, then call
    ``create`` for one structure or ``create_many`` for a sequence. It reuses
    the lower-level exact ACE descriptor calculator and cache machinery.
    """

    def __init__(
        self,
        settings,
        site_basis_config,
        *,
        elements,
        type_map=None,
        cutoff=None,
        basis_mode=None,
        exact_primitive_timeout_seconds=None,
        center_mu_values=None,
        restrict_neighbor_mu=None,
        max_variants_per_label=None,
        descriptor_cache=None,
        use_descriptor_cache=True,
        feature_filters=None,
        charge_source=None,
        magnetic_moment_source=None,
        backend="pytorch",
        strict_backend=False,
        validate_backend=True,
        device=None,
        factorized_descriptor_runtime_policy=None,
        scalar_coordinate_compiler=None,
        ordinary_scalar_catalogue=None,
        compact_labels=None,
        direct_feature_filter_labels=False,
        _suppress_legacy_warning=False,
    ):
        if not _suppress_legacy_warning:
            warnings.warn(
                "ACEDescriptor is a legacy descriptor front-end. "
                "Use YE3TDescriptors.ace(...) for new descriptor-first workflows.",
                FutureWarning,
                stacklevel=2,
            )
        self.elements = tuple(str(elem) for elem in elements)
        self.type_map = build_explicit_type_map(self.elements) if type_map is None else {str(k): int(v) for k, v in type_map.items()}
        self.cutoff = float(cutoff if cutoff is not None else max(site_basis_config.rc))
        self.feature_filters = feature_filters
        self.ordinary_scalar_catalogue = ordinary_scalar_catalogue
        self.charge_source = charge_source
        self.magnetic_moment_source = magnetic_moment_source
        self.device = _resolve_descriptor_device(device)
        active_cache = descriptor_cache
        labels = None if compact_labels is None else tuple(normalize_compact_label(label) for label in compact_labels)
        if feature_filters is not None:
            if labels is None and bool(direct_feature_filter_labels) and basis_mode in {None, "exact", "full_exact_basis", "original_exact"}:
                labels = _direct_compact_labels_from_feature_filters(settings, feature_filters)
            if labels is None:
                all_labels = enumerate_compact_labels(
                    settings,
                    basis_mode=basis_mode,
                    exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
                    descriptor_cache=active_cache,
                    use_descriptor_cache=use_descriptor_cache,
                )
                labels = filter_compact_labels_by_specs(all_labels, feature_filters)
            else:
                labels = filter_compact_labels_by_specs(labels, feature_filters)
        self.calculator = DescriptorCalculator.from_settings(
            settings,
            site_basis_config,
            basis_mode=basis_mode,
            exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
            compact_labels=labels,
            center_mu_values=center_mu_values,
            restrict_neighbor_mu=restrict_neighbor_mu,
            max_variants_per_label=max_variants_per_label,
            descriptor_cache=active_cache,
            use_descriptor_cache=use_descriptor_cache,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            device=self.device,
            factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
            scalar_coordinate_compiler=scalar_coordinate_compiler,
            _suppress_legacy_warning=True,
        )
        self.descriptor_specs = filter_descriptor_specs_by_specs(self.calculator.descriptor_specs, feature_filters)
        self.settings = settings
        self.site_basis_config = site_basis_config
        self.descriptor_cache = descriptor_cache

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        unsupported = sorted(str(key) for key, value in cfg.items()
                             if str(key).startswith("tensor_") and value is not None)
        if unsupported:
            raise ValueError("Unsupported descriptor settings: " + ", ".join(unsupported))
        if "elements" not in cfg and "elems" in cfg:
            cfg["elements"] = cfg["elems"]
        elements = tuple(str(elem) for elem in cfg["elements"])
        type_map = cfg.get("type_map", build_explicit_type_map(elements))
        settings_payload = cfg.get("settings", cfg.get("descriptor_settings", None))
        if settings_payload is not None:
            settings = DescriptorGenerationSettings.from_dict(settings_payload)
        else:
            ranks = tuple(int(v) for v in cfg.get("ranks", (1, 2)))
            properties = dict(cfg.get("properties", {}))
            basis_type = str(cfg.get("basis_type", "charge" if "charge" in properties else ("magnetic" if "magnetic_moments" in properties else "no_charge")))
            settings = DescriptorGenerationSettings(
                ranks=ranks,
                basis_type=basis_type,
                k_o_max=int(cfg.get("k_o_max", 0 if basis_type == "no_charge" else 1)),
                k_max=_as_rank_tuple(cfg.get("k_max", 0), ranks, name="k_max", dtype=int),
                elems=elements,
                nmax=_as_rank_tuple(cfg.get("nmax", 4), ranks, name="nmax", dtype=int),
                lmax=_as_rank_tuple(cfg.get("lmax", 2), ranks, name="lmax", dtype=int),
                lmin=_as_rank_tuple(cfg.get("lmin", 0), ranks, name="lmin", dtype=int),
                L_R=int(cfg.get("L_R", 0)),
                M_R_values=tuple(int(v) for v in cfg.get("M_R_values", (0,))),
                max_labels_per_rank=cfg.get("max_labels_per_rank", None),
                tree_type=str(cfg.get("tree_type", "balanced")),
                parity_filter=str(cfg.get("parity_filter", "natural")),
                aux_lmax=int(cfg.get("aux_lmax", 1)),
            )
        properties = dict(cfg.get("properties", {}))
        basis_type = str(settings.basis_type)
        charge_bounds = None
        if basis_type == "charge" or "charge" in properties:
            charge_cfg = dict(properties.get("charge", {}))
            charge_bounds = resolve_charge_bounds(
                elements,
                strategy=str(charge_cfg.get("bounds", charge_cfg.get("strategy", "oxidation"))),
                fallback=tuple(charge_cfg.get("fallback", (-1.0, 1.0))),
                oxidation_padding=float(charge_cfg.get("oxidation_padding", 0.0)),
            )
        site_basis_payload = cfg.get("site_basis_config", cfg.get("site_basis_config_payload", None))
        if site_basis_payload is not None:
            site_basis_config = (
                deserialize_site_basis_config(site_basis_payload)
                if isinstance(site_basis_payload, Mapping)
                else site_basis_payload
            )
        else:
            site_cfg = dict(cfg.get("site_basis", {}))
            if site_cfg.get("mode", "vdw_scaled") != "vdw_scaled":
                possible_types = tuple(int(type_map[str(elem)]) for elem in elements)
                site_basis_config = SiteBasisConfig(
                    rc=ordered_pair_values(site_cfg.get("rc", cfg.get("cutoff", 4.0)), possible_types=possible_types, name="rc"),
                    lmbda=ordered_pair_values(site_cfg.get("lmbda", 0.25), possible_types=possible_types, name="lmbda"),
                    nradmax=max(settings.nmax),
                    lmax=max(settings.lmax),
                    kmax=max(settings.k_max),
                    possible_types=possible_types,
                    charge_mode="scalar" if basis_type == "charge" else "none",
                    dtype=torch.float64,
                    complex_dtype=torch.complex128,
                    **({} if charge_bounds is None else charge_bounds.as_site_basis_kwargs()),
                )
            else:
                site_basis_config = build_vdw_site_basis_config(
                    elements,
                    type_map=type_map,
                    nradmax=max(settings.nmax),
                    lmax=max(settings.lmax),
                    kmax=max(settings.k_max),
                    charge_mode="scalar" if basis_type == "charge" else "none",
                    charge_bounds=charge_bounds,
                    cutoff_scale=float(site_cfg.get("cutoff_scale", cfg.get("vdw_cutoff_scale", 1.0))),
                    lambda_scale=float(site_cfg.get("lambda_scale", cfg.get("vdw_lambda_scale", 1.0))),
                    min_cutoff=float(site_cfg.get("min_cutoff", 1.5)),
                )
        cache_dir = cfg.get("descriptor_cache_dir", None)
        descriptor_cache = DescriptorBuildCache(cache_dir=cache_dir) if cache_dir is not None else cfg.get("descriptor_cache", None)
        return cls(
            settings,
            site_basis_config,
            elements=elements,
            type_map=type_map,
            cutoff=cfg.get("cutoff", max(site_basis_config.rc)),
            basis_mode=cfg.get("basis_mode", None),
            exact_primitive_timeout_seconds=cfg.get("exact_primitive_timeout_seconds", None),
            center_mu_values=cfg.get("center_mu_values", None),
            restrict_neighbor_mu=cfg.get("restrict_neighbor_mu", None),
            max_variants_per_label=cfg.get("max_variants_per_label", None),
            descriptor_cache=descriptor_cache,
            use_descriptor_cache=bool(cfg.get("use_descriptor_cache", True)),
            feature_filters=cfg.get("feature_filters", None),
            charge_source=properties.get("charge", {}).get("source", None) if "charge" in properties else cfg.get("charge_source", None),
            magnetic_moment_source=properties.get("magnetic_moments", {}).get("source", None) if "magnetic_moments" in properties else cfg.get("magnetic_moment_source", None),
            backend=cfg.get("backend", "pytorch"),
            strict_backend=bool(cfg.get("strict_backend", False)),
            validate_backend=bool(cfg.get("validate_backend", True)),
            device=cfg.get("device", None),
            factorized_descriptor_runtime_policy=cfg.get("factorized_descriptor_runtime_policy", None),
            scalar_coordinate_compiler=cfg.get("_scalar_coordinate_compiler", None),
            ordinary_scalar_catalogue=cfg.get("_ordinary_scalar_catalogue", None),
            compact_labels=cfg.get("compact_labels", None),
            direct_feature_filter_labels=str(cfg.get("feature_filter_label_strategy", "exhaustive")).strip().lower()
            in {"direct", "explicit", "from_filters"},
            _suppress_legacy_warning=bool(cfg.get("_suppress_legacy_warning", False)),
        )

    @classmethod
    def from_structures(cls, structures, **kwargs):
        atoms_list = list(structures) if not hasattr(structures, "get_chemical_symbols") else [structures]
        elements = kwargs.pop("elements", None)
        if elements is None:
            elements = infer_elements_from_ase_atoms(atoms_list)
        config = dict(kwargs)
        config["elements"] = list(elements)
        return cls.from_config(config)

    @property
    def feature_keys(self):
        return tuple(spec.key for spec in self.descriptor_specs)

    def radial_defaults(self):
        return {
            "elements": list(self.elements),
            "type_map": dict(self.type_map),
            "rc": list(self.site_basis_config.rc),
            "lmbda": list(self.site_basis_config.lmbda),
            "ordered_bond_types": [tuple(pair) for pair in self.site_basis_config.bond_types],
        }

    def print_defaults(self):
        defaults = self.radial_defaults()
        print("ACE descriptor elements:", defaults["elements"])
        print("ACE descriptor type_map:", defaults["type_map"])
        print("ACE descriptor ordered bond types:", defaults["ordered_bond_types"])
        print("ACE descriptor rc:", defaults["rc"])
        print("ACE descriptor lmbda:", defaults["lmbda"])

    def _neighbor_data(self, atoms):
        return neighbor_data_from_ase_atoms(atoms, cutoff=self.cutoff, type_map=self.type_map)

    def _auxiliary_inputs(self, atoms, neighbor_data):
        charges = _extract_charges_from_atoms(atoms, self.charge_source)
        aux_tensor_basis = None
        magnetic_vectors = _extract_magnetic_vectors_from_atoms(atoms, self.magnetic_moment_source)
        if magnetic_vectors is not None:
            edge_index = torch.as_tensor(neighbor_data.edge_index, dtype=torch.long, device=self.device)
            aux_tensor_basis = magnetic_orientation_basis(
                torch.as_tensor(magnetic_vectors, dtype=self.site_basis_config.dtype, device=self.device),
                edge_index,
                int(self.settings.aux_lmax),
                dtype=self.site_basis_config.dtype,
                complex_dtype=self.site_basis_config.complex_dtype,
            )
        return charges, aux_tensor_basis

    def create(self, atoms, *, real_if_scalar=True):
        neighbor_data = self._neighbor_data(atoms)
        charges, aux_tensor_basis = self._auxiliary_inputs(atoms, neighbor_data)
        original_specs = self.calculator.descriptor_specs
        if self.descriptor_specs is not original_specs:
            self.calculator.descriptor_specs = self.descriptor_specs
        try:
            values = self.calculator.compute_from_neighbor_data(
                neighbor_data,
                charges=charges,
                aux_tensor_basis=aux_tensor_basis,
                real_if_scalar=real_if_scalar,
            )
            return values
        finally:
            if self.descriptor_specs is not original_specs:
                self.calculator.descriptor_specs = original_specs

    def create_many(self, structures, *, concatenate=False, return_batch=False, real_if_scalar=True):
        values = []
        row_slices = []
        start = 0
        for atoms in structures:
            block = self.create(atoms, real_if_scalar=real_if_scalar)
            values.append(block)
            stop = start + int(block.shape[0])
            row_slices.append((start, stop))
            start = stop
        if return_batch:
            return ACEDescriptorBatch(
                values_by_structure=tuple(values),
                row_slices=tuple(row_slices),
                elements=self.elements,
                type_map=dict(self.type_map),
                feature_keys=self.feature_keys,
            )
        if concatenate:
            return torch.cat(tuple(values), dim=0) if values else torch.empty((0, len(self.feature_keys)), dtype=self.site_basis_config.dtype)
        return values

    def create_from_files(self, paths, *, index=":", concatenate=False, return_batch=False, real_if_scalar=True):
        return self.create_many(
            read_ase_structures(paths, index=index),
            concatenate=concatenate,
            return_batch=return_batch,
            real_if_scalar=real_if_scalar,
        )


@recordclass(('values', 'sector_slices', 'sectors', 'blocks', 'axes', 'unflattened_axes', 'metadata'))
class ASMatrixUnitDescriptorResult:
    """Descriptor-shaped view of evaluated A_s matrix-unit carrier blocks.

    The flattened ``values`` matrix is a runtime packaging of carrier tensors,
    not the completed global Young--E3 descriptor coefficient contraction.
    ``sector_slices`` records how to recover each unflattened carrier block.
    """
    axes = ("atom", "flattened_matrix_unit_carrier_feature")
    unflattened_axes = (
        "atom",
        "sector",
        "tableau_row",
        "tableau_col",
        "slot_tuple_carrier",
    )
    metadata = field(default_factory=dict)

    @property
    def num_atoms(self):
        return int(self.values.shape[0])

    @property
    def num_features(self):
        return int(self.values.shape[1])

    def sector_values(self, sector_index):
        """Return one sector block reshaped to its recorded carrier shape."""

        record = self.sector_slices[int(sector_index)]
        flat = self.values[:, int(record["start"]): int(record["stop"])]
        return flat.reshape(tuple(record["shape"]))


@recordclass(('values_by_structure', 'row_slices', 'axes', 'unflattened_axes', 'metadata'))
class ASMatrixUnitDescriptorBatch:
    """Batch container for A_s matrix-unit descriptor-shaped results."""
    axes = ("atom", "flattened_matrix_unit_carrier_feature")
    unflattened_axes = (
        "atom",
        "sector",
        "tableau_row",
        "tableau_col",
        "slot_tuple_carrier",
    )
    metadata = field(default_factory=dict)

    @property
    def num_structures(self):
        return int(len(self.values_by_structure))

    @property
    def total_rows(self):
        return int(sum(int(value.values.shape[0]) for value in self.values_by_structure))


class ASMatrixUnitGlobalCouplerLinearModel(torch.nn.Module):
    """Energy-only linear readout on evaluated A_s/global-coupler descriptors.

    The model consumes descriptor results produced by
    ``create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor``.  It is
    deliberately limited to scalar energy rows: force/Jacobian support requires
    differentiating the evaluated A_s matrix-unit/global-coupler descriptor
    path and is not implemented here.
    """

    def __init__(self, coefficients, *, include_bias_column, metadata):
        super().__init__()
        coeff = torch.as_tensor(coefficients)
        feature_count = int(coeff.numel()) - (1 if include_bias_column else 0)
        if feature_count < 0:
            raise ValueError("A_s matrix-unit/global-coupler linear coefficients have invalid length.")
        self.register_buffer("feature_coefficients", coeff[:feature_count].clone())
        bias = coeff[feature_count:].clone() if include_bias_column else coeff.new_zeros((1,))
        self.register_buffer("atom_count_bias_coefficient", bias.reshape(-1)[:1].clone())
        self.include_bias_column = bool(include_bias_column)
        self._ye3t_linear_fit_metadata = dict(metadata)
        self._ye3t_linear_fit_coefficients = coeff.detach().cpu()

    @property
    def num_features(self):
        return int(self.feature_coefficients.numel())

    def forward(self, structure_feature_sums, atom_counts=None):
        features = torch.as_tensor(
            structure_feature_sums,
            dtype=self.feature_coefficients.dtype,
            device=self.feature_coefficients.device,
        )
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if int(features.shape[-1]) != self.num_features:
            raise ValueError(
                "Expected structure-summed feature width "
                f"{self.num_features}, got {int(features.shape[-1])}."
            )
        energy = features @ self.feature_coefficients
        if self.include_bias_column:
            if atom_counts is None:
                counts = torch.ones(
                    (int(features.shape[0]),),
                    dtype=self.feature_coefficients.dtype,
                    device=self.feature_coefficients.device,
                )
            else:
                counts = torch.as_tensor(
                    atom_counts,
                    dtype=self.feature_coefficients.dtype,
                    device=self.feature_coefficients.device,
                ).reshape(-1)
            if int(counts.numel()) != int(features.shape[0]):
                raise ValueError("atom_counts length must match the number of structure feature rows.")
            energy = energy + counts * self.atom_count_bias_coefficient[0]
        return energy

    def predict_descriptor_result(self, result):
        if not isinstance(result, ASMatrixUnitDescriptorResult):
            raise TypeError("predict_descriptor_result expects an ASMatrixUnitDescriptorResult.")
        features = result.values.to(
            dtype=self.feature_coefficients.dtype,
            device=self.feature_coefficients.device,
        ).sum(dim=0)
        return self.forward(features, atom_counts=(int(result.num_atoms),)).reshape(())

    def predict_descriptor_batch(self, batch):
        if not isinstance(batch, ASMatrixUnitDescriptorBatch):
            raise TypeError("predict_descriptor_batch expects an ASMatrixUnitDescriptorBatch.")
        feature_sums = []
        atom_counts = []
        for result in batch.values_by_structure:
            feature_sums.append(
                result.values.to(
                    dtype=self.feature_coefficients.dtype,
                    device=self.feature_coefficients.device,
                ).sum(dim=0)
            )
            atom_counts.append(int(result.num_atoms))
        if not feature_sums:
            return self.feature_coefficients.new_empty((0,))
        return self.forward(torch.stack(tuple(feature_sums), dim=0), atom_counts=atom_counts)

    def predict_structures(self, descriptor, structures, *, require_all_sectors=False):
        """Evaluate the matching descriptor path and predict scalar energies."""

        if not isinstance(descriptor, YE3TDescriptorSet):
            raise TypeError("predict_structures expects a YE3TDescriptorSet descriptor.")
        batch = descriptor.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor_many(
            list(structures),
            return_batch=True,
            require_all_sectors=bool(require_all_sectors),
        )
        return self.predict_descriptor_batch(batch)


@recordclass(('values', 'sector_slices', 'sector_evaluations', 'axes', 'unflattened_axes', 'metadata'))
class YE3TSectorDescriptorResult:
    """Descriptor-shaped direct-sum package of concrete Young target sectors."""
    metadata = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(int(dim) for dim in self.values.shape)

    @property
    def target_partitions(self):
        return tuple(tuple(record["target_partition"]) for record in self.sector_slices)

    def sector_values(self, sector_index):
        record = self.sector_slices[int(sector_index)]
        axis = int(self.metadata.get("output_axis", -1))
        if axis < 0:
            axis += int(self.values.ndim)
        index = [slice(None)] * int(self.values.ndim)
        index[axis] = slice(int(record["start"]), int(record["stop"]))
        return self.values[tuple(index)]


@recordclass(('settings', 'site_basis_config', 'elements', 'type_map', 'cutoff', 'representation', 'compact_labels', 'descriptor_specs', 'backend', 'strict_backend', 'validate_backend', 'descriptor_cache', 'ace_descriptor', 'settings_by_L', 'metadata', 'compiled_basis'))
class YE3TDescriptorSet:
    """Canonical descriptor object consumed by descriptor-first model factories."""
    compact_labels = tuple()
    descriptor_specs = tuple()
    backend = "pytorch"
    strict_backend = False
    validate_backend = True
    descriptor_cache = None
    ace_descriptor = None
    settings_by_L = None
    metadata = field(default_factory=dict)
    compiled_basis = None

    @property
    def feature_keys(self):
        metadata_keys = self.metadata.get("feature_keys")
        if metadata_keys is not None:
            return tuple(metadata_keys)
        return tuple(getattr(spec, "key", None) for spec in self.descriptor_specs)

    @property
    def feature_labels(self):
        """Compiler-owned labels in the order returned by ``create``."""
        return tuple(self.metadata.get("feature_coordinate_provenance", self.feature_keys))

    @property
    def irrep_inventory(self):
        return self.metadata.get("ye3_irrep_inventory", None)

    @property
    def calculator(self):
        if self.ace_descriptor is None:
            return None
        return self.ace_descriptor.calculator

    @property
    def supports_runtime_evaluation(self):
        return (self.ace_descriptor is not None or
                self.metadata.get("descriptor_family") == "tagged_cauchy_carriers" or
                self.metadata.get("tagged_cauchy_image_evaluator") is not None)

    @property
    def binding_requirements(self):
        return tuple(self.metadata.get("required_binding_stages", ()))

    def _require_runtime(self):
        if self.ace_descriptor is None:
            if self.metadata.get("descriptor_family") == "fixed_content_basis_plan":
                raise NotImplementedError(
                    "This fixed-content basis is an exact representation plan, not an ASE feature "
                    "evaluator. Bind physical radial/source functions and a tensor-slot role or "
                    "placement action, then lower its Young/rotation coefficients to those "
                    "sources before create(atoms) can evaluate features. The requested backend "
                    f"{self.backend!r} has no fixed-content contraction kernel yet."
                )
            raise NotImplementedError(
                "Runtime evaluation for this YE3T descriptor representation is not implemented yet. "
                "The descriptor object records the requested representation selection, but only the "
                "trivial ACE fast path is currently wired to descriptor evaluation."
            )
        return self.ace_descriptor

    @staticmethod
    def _normalize_descriptor_evaluation_selector(descriptor_evaluation):
        if descriptor_evaluation is None:
            return None
        text = str(descriptor_evaluation).strip().lower().replace("-", "_")
        if not text:
            return None
        aliases = {
            "a_s_matrix_unit_descriptor": "matrix_unit_descriptor",
            "matrix_unit_descriptor": "matrix_unit_descriptor",
            "a_s_matrix_unit_carrier_descriptor": "matrix_unit_descriptor",
            "matrix_unit_carrier_descriptor": "matrix_unit_descriptor",
            "matrix_unit_carrier": "matrix_unit_descriptor",
            "a_s_matrix_unit_ye3t_axis": "matrix_unit_ye3t_axis",
            "a_s_matrix_unit_ye3t_axis_descriptor": "matrix_unit_ye3t_axis",
            "matrix_unit_ye3t_axis": "matrix_unit_ye3t_axis",
            "matrix_unit_ye3t_axis_descriptor": "matrix_unit_ye3t_axis",
            "a_s_matrix_unit_slot_resolved_global_coupler": "matrix_unit_slot_resolved_global_coupler",
            "a_s_matrix_unit_slot_resolved_global_coupler_descriptor": "matrix_unit_slot_resolved_global_coupler",
            "a_s_non_scalar_m_r_resolved_global_coupler": "matrix_unit_slot_resolved_global_coupler",
            "a_s_non_scalar_m_r_resolved_global_coupler_descriptor": "matrix_unit_slot_resolved_global_coupler",
            "matrix_unit_slot_resolved_global_coupler": "matrix_unit_slot_resolved_global_coupler",
            "matrix_unit_slot_resolved_global_coupler_descriptor": "matrix_unit_slot_resolved_global_coupler",
            "slot_resolved_global_coupler": "matrix_unit_slot_resolved_global_coupler",
            "a_s_matrix_unit_scalar_contraction": "matrix_unit_scalar_contraction",
            "a_s_matrix_unit_scalar_contraction_descriptor": "matrix_unit_scalar_contraction",
            "matrix_unit_scalar_contraction": "matrix_unit_scalar_contraction",
            "matrix_unit_scalar_contraction_descriptor": "matrix_unit_scalar_contraction",
            "a_s_matrix_unit_commutant_scalar": "matrix_unit_commutant_scalar",
            "a_s_matrix_unit_commutant_scalar_descriptor": "matrix_unit_commutant_scalar",
            "matrix_unit_commutant_scalar": "matrix_unit_commutant_scalar",
            "matrix_unit_commutant_scalar_descriptor": "matrix_unit_commutant_scalar",
        }
        if text not in aliases:
            raise ValueError(f"Unsupported descriptor_evaluation selector: {descriptor_evaluation!r}.")
        return aliases[text]

    def _A_s_matrix_unit_requested_target_L_R(self):
        spec_payload = self.metadata.get("ye3t_spec", None)
        if isinstance(spec_payload, Mapping):
            rotation = spec_payload.get("target_rotation", {})
            if isinstance(rotation, Mapping) and "L_R" in rotation:
                return int(rotation["L_R"])
        for source in (self.settings, self.metadata):
            if isinstance(source, Mapping):
                for key in ("target_L_R", "L_R"):
                    if key in source:
                        return int(source[key])
            elif source is not None:
                for key in ("target_L_R", "L_R"):
                    if hasattr(source, key):
                        return int(getattr(source, key))
        return 0

    def _A_s_matrix_unit_requested_M_R_values(self):
        spec_payload = self.metadata.get("ye3t_spec", None)
        if isinstance(spec_payload, Mapping):
            rotation = spec_payload.get("target_rotation", {})
            if isinstance(rotation, Mapping):
                for key in ("M_R", "M_R_values", "target_M_R_values"):
                    if key in rotation and rotation[key] is not None:
                        return tuple(int(value) for value in rotation[key])
        for source in (self.settings, self.metadata):
            if isinstance(source, Mapping):
                for key in ("target_M_R_values", "M_R_values", "M_R"):
                    if key in source and source[key] is not None:
                        value = source[key]
                        if isinstance(value, (list, tuple)):
                            return tuple(int(entry) for entry in value)
                        return (int(value),)
            elif source is not None:
                for key in ("target_M_R_values", "M_R_values", "M_R"):
                    if hasattr(source, key):
                        value = getattr(source, key)
                        if value is None:
                            continue
                        if isinstance(value, (list, tuple)):
                            return tuple(int(entry) for entry in value)
                        return (int(value),)
        return None

    def _require_A_s_matrix_unit_scalar_target(self, descriptor_evaluation):
        target_L_R = self._A_s_matrix_unit_requested_target_L_R()
        if target_L_R != 0:
            raise NotImplementedError(
                "A_s matrix-unit descriptor_evaluation selectors currently support only scalar "
                f"target_rotation L_R=0 and M_R=0 axes; requested L_R={target_L_R}. "
                "Use descriptor_evaluation='A_s_matrix_unit_slot_resolved_global_coupler' for the "
                "implemented first-pass non-scalar M_R-resolved A_s descriptor path."
            )
        return descriptor_evaluation

    def _require_A_s_matrix_unit_supported_target(self, descriptor_evaluation):
        target_L_R = self._A_s_matrix_unit_requested_target_L_R()
        key = self._normalize_descriptor_evaluation_selector(descriptor_evaluation)
        if key == "matrix_unit_slot_resolved_global_coupler":
            return descriptor_evaluation
        if target_L_R != 0:
            raise NotImplementedError(
                "A_s matrix-unit descriptor_evaluation selectors currently support only scalar "
                f"target_rotation L_R=0 and M_R=0 axes; requested L_R={target_L_R}. "
                "Use descriptor_evaluation='A_s_matrix_unit_slot_resolved_global_coupler' for the "
                "implemented first-pass non-scalar M_R-resolved A_s descriptor path."
            )
        return descriptor_evaluation

    def _create_selected_A_s_matrix_unit_descriptor(
        self,
        atoms,
        descriptor_evaluation,
        *,
        require_all_sectors=False,
    ):
        key = self._normalize_descriptor_evaluation_selector(descriptor_evaluation)
        if key is None:
            raise ValueError("descriptor_evaluation must name an A_s matrix-unit descriptor route.")
        if self.metadata.get("descriptor_family") not in {"ye3t_basis", "filtered_A_s"} and self.metadata.get(
            "legacy_descriptor_family"
        ) != "filtered_A_s":
            raise NotImplementedError(
                "A_s matrix-unit descriptor_evaluation selectors require a ye3t_basis / filtered_A_s descriptor."
            )
        self._require_A_s_matrix_unit_supported_target(descriptor_evaluation)
        if key == "matrix_unit_slot_resolved_global_coupler":
            return self.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(
                atoms,
                require_all_sectors=bool(require_all_sectors),
            )
        if bool(require_all_sectors):
            raise ValueError(
                "require_all_sectors applies only to descriptor_evaluation="
                "'A_s_matrix_unit_slot_resolved_global_coupler'."
            )
        if key == "matrix_unit_descriptor":
            return self.create_A_s_matrix_unit_descriptor(atoms)
        if key == "matrix_unit_ye3t_axis":
            return self.create_A_s_matrix_unit_ye3t_axis_descriptor(atoms)
        if key == "matrix_unit_scalar_contraction":
            return self.create_A_s_matrix_unit_scalar_contraction_descriptor(atoms)
        if key == "matrix_unit_commutant_scalar":
            return self.create_A_s_matrix_unit_commutant_scalar_descriptor(atoms)
        raise AssertionError(f"Unhandled descriptor_evaluation selector: {key!r}.")

    def create(self, atoms, *, real_if_scalar=True, descriptor_evaluation=None,
               require_all_sectors=False, backend=None, native_library=None,
               execution_policy="direct"):
        """Evaluate this space on ASE atoms.

        Scalar tagged physical-image descriptors retain compiler coordinate order.
        """
        if self.metadata.get("descriptor_family") == "fixed_content_basis_plan":
            self._require_runtime()
        if self.metadata.get("descriptor_family") == "tagged_cauchy_carriers":
            from ase.neighborlist import neighbor_list
            from ye3t_ace.tagged_cauchy_carriers import (
                _TaggedCauchyOccurrenceSource, tagged_support_chunks, tagged_support_layout,
            )

            if descriptor_evaluation is not None or require_all_sectors:
                raise ValueError("Tagged carriers do not use an A_s matrix-unit selector")
            if native_library is not None or execution_policy != "direct":
                raise ValueError("Tagged carriers use their compiled source evaluator")
            if backend not in (None, "reference"):
                raise ValueError("Standalone tagged carrier ASE backend is reference")
            source_config = self.metadata["tagged_cauchy_carriers_config"]
            device = str(source_config["device"])
            dtype = {"float32": torch.float32, "float64": torch.float64}.get(
                str(source_config["dtype"]))
            if dtype is None:
                raise ValueError("Tagged carrier dtype must be float32 or float64")
            cache = self.metadata.setdefault("_tagged_carrier_evaluators", {})
            cache_key = (device, str(source_config["dtype"]))
            if cache_key not in cache:
                cache[cache_key] = _TaggedCauchyOccurrenceSource(
                    self.metadata["tagged_cauchy_carrier_source_plan"]["schedules"],
                    self.elements, self.cutoff,
                    pair_cutoffs_A=source_config.get("pair_cutoffs_A"),
                    backend="reference", dtype=dtype,
                ).to(device)
            unknown = sorted(set(atoms.get_chemical_symbols()) - set(self.type_map))
            if unknown:
                raise ValueError(f"Tagged carriers contain unknown species: {unknown}")
            atom_types = torch.as_tensor(
                [self.type_map[name] for name in atoms.get_chemical_symbols()],
                dtype=torch.long, device=device,
            )
            center, neighbor, images = neighbor_list("ijS", atoms, self.cutoff)
            edges = torch.as_tensor(np.stack((center, neighbor)), dtype=torch.long,
                                    device=device)
            shifts = torch.as_tensor(images, dtype=torch.long, device=device)
            layout = tagged_support_layout(edges, len(atoms), shifts)
            positions = torch.as_tensor(np.asarray(atoms.positions), dtype=dtype,
                                        device=device)
            cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=dtype,
                                   device=device)
            displacements = (positions.index_select(0, edges[1])
                             - positions.index_select(0, edges[0])
                             + shifts.to(dtype) @ cell)
            if torch.any(torch.linalg.vector_norm(displacements, dim=1) <= 1e-12):
                raise ValueError("Tagged carrier neighbor occurrences must have nonzero distance")
            with torch.no_grad():
                edge_values, density_values = cache[cache_key].geometry_values(
                    displacements, atom_types, edges, len(atoms))
                carriers = {}
                for schedule in self.metadata["tagged_cauchy_carrier_source_plan"]["schedules"]:
                    tag_count = int(schedule["tag_count"])
                    values, centers, tag_edges = [], [], []
                    for support in tagged_support_chunks(
                            layout, int(schedule["support_tag_count"]),
                            int(source_config["support_chunk_size"])):
                        values.append(cache[cache_key](tag_count, edge_values, density_values,
                                                       support, normalized=False).cpu())
                        centers.append(support["centers"].cpu())
                        tag_edges.append(support["tag_edges"].cpu())
                    carriers[tag_count] = {
                        "values": (torch.cat(values, dim=0).numpy() if values else
                                   np.empty((0, int(schedule["output_dimension"])))),
                        "centers": (torch.cat(centers).numpy() if centers else
                                    np.empty(0, dtype=np.int64)),
                        "tag_edges": (torch.cat(tag_edges).numpy() if tag_edges else
                                      np.empty((0, int(schedule["support_tag_count"])),
                                               dtype=np.int64)),
                        "labels": tuple({**record["label"],
                                         "component_slice": tuple(record["component_slice"])}
                                        for record in schedule["inventory"]),
                        "support_tag_count": int(schedule["support_tag_count"]),
                    }
            return {
                "carriers": carriers, "edge_index": edges.cpu().numpy(),
                "shifts": shifts.cpu().numpy(), "normalization": "raw",
                "catalogue_hash": self.metadata["tagged_cauchy_carriers_compiled"]["self_hash"],
            }
        if self.metadata.get("descriptor_family") == "linear_tagged_cauchy_image":
            if descriptor_evaluation is not None or require_all_sectors:
                raise ValueError("Tagged scalar features do not use an A_s matrix-unit selector.")
            evaluator = self.metadata.get("tagged_cauchy_image_evaluator")
            if evaluator is None:
                raise RuntimeError("Compile the tagged descriptor before evaluating atoms.")
            if backend == "native_cpu":
                import os
                from ye3t_ace.tagged_cauchy_image import TaggedCauchyImageLinearModel
                from ye3t_ace.tagged_cauchy_native import _TaggedCauchyNativeRuntime

                key = (str(native_library or os.environ.get("YE3T_TAGGED_C_API_LIBRARY")),
                       str(execution_policy))
                cache = self.metadata.setdefault("_tagged_native_descriptor_runtimes", {})
                if key not in cache:
                    zeros = {name: np.zeros(evaluator.feature_count)
                             for name in evaluator.species_order}
                    model = TaggedCauchyImageLinearModel(
                        evaluator, zeros, {name: 0.0 for name in evaluator.species_order})
                    cache[key] = _TaggedCauchyNativeRuntime(
                        model, library_path=native_library,
                        execution_policy=execution_policy)
                return cache[key].evaluate_atoms(atoms, return_features=True)[-1]
            if backend is not None:
                raise ValueError("Tagged descriptor backend must be native_cpu or omitted to use the configured evaluator.")
            symbols = atoms.get_chemical_symbols()
            unknown = sorted(set(symbols) - set(evaluator.type_map))
            if unknown:
                raise ValueError(f"Tagged descriptor contains unknown species: {unknown}.")
            atom_types = [evaluator.type_map[symbol] for symbol in symbols]
            with torch.no_grad():
                _edges, _displacements, values, _derivatives = evaluator.materialize(
                    np.asarray(atoms.positions, dtype=np.float64), atom_types,
                    cell=np.asarray(atoms.cell.array, dtype=np.float64), pbc=atoms.pbc,
                )
            return values.detach().cpu().numpy()
        if descriptor_evaluation is not None:
            return self._create_selected_A_s_matrix_unit_descriptor(
                atoms,
                descriptor_evaluation,
                require_all_sectors=bool(require_all_sectors),
            )
        if bool(require_all_sectors):
            raise ValueError("require_all_sectors requires an explicit A_s matrix-unit descriptor_evaluation selector.")
        return self._require_runtime().create(atoms, real_if_scalar=real_if_scalar)

    def create_many(
        self,
        structures,
        *,
        concatenate=False,
        return_batch=False,
        real_if_scalar=True,
        descriptor_evaluation=None,
        require_all_sectors=False,
        backend=None,
        native_library=None,
        execution_policy="direct",
    ):
        if self.metadata.get("descriptor_family") == "fixed_content_basis_plan":
            self._require_runtime()
        if self.metadata.get("descriptor_family") == "tagged_cauchy_carriers":
            if descriptor_evaluation is not None or require_all_sectors:
                raise ValueError("Tagged carriers do not use an A_s matrix-unit selector")
            if concatenate or return_batch:
                raise ValueError("Tagged carrier supports differ by structure; use the returned tuple.")
            return tuple(self.create(atoms, backend=backend, native_library=native_library,
                                     execution_policy=execution_policy)
                         for atoms in structures)
        if self.metadata.get("descriptor_family") == "linear_tagged_cauchy_image":
            if descriptor_evaluation is not None or require_all_sectors:
                raise ValueError("Tagged scalar features do not use an A_s matrix-unit selector.")
            values = tuple(self.create(atoms, backend=backend, native_library=native_library,
                                       execution_policy=execution_policy) for atoms in structures)
            if return_batch:
                raise ValueError("Tagged create_many does not have an ASE batch carrier.")
            if concatenate:
                width = len(self.feature_keys)
                return np.concatenate(values, axis=0) if values else np.empty((0, width))
            return values
        key = self._normalize_descriptor_evaluation_selector(descriptor_evaluation)
        if key == "matrix_unit_slot_resolved_global_coupler":
            self._require_A_s_matrix_unit_supported_target(descriptor_evaluation)
            return self.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor_many(
                structures,
                concatenate=concatenate,
                return_batch=return_batch,
                require_all_sectors=bool(require_all_sectors),
            )
        if key is not None:
            self._require_A_s_matrix_unit_supported_target(descriptor_evaluation)
            if bool(require_all_sectors):
                raise ValueError(
                    "require_all_sectors applies only to descriptor_evaluation="
                    "'A_s_matrix_unit_slot_resolved_global_coupler'."
                )
            if key == "matrix_unit_descriptor":
                return self.create_A_s_matrix_unit_descriptor_many(
                    structures,
                    concatenate=concatenate,
                    return_batch=return_batch,
                )
            if key == "matrix_unit_ye3t_axis":
                return self.create_A_s_matrix_unit_ye3t_axis_descriptor_many(
                    structures,
                    concatenate=concatenate,
                    return_batch=return_batch,
                )
            if key == "matrix_unit_scalar_contraction":
                return self.create_A_s_matrix_unit_scalar_contraction_descriptor_many(
                    structures,
                    concatenate=concatenate,
                    return_batch=return_batch,
                )
            if key == "matrix_unit_commutant_scalar":
                return self.create_A_s_matrix_unit_commutant_scalar_descriptor_many(
                    structures,
                    concatenate=concatenate,
                    return_batch=return_batch,
                )
            raise AssertionError(f"Unhandled descriptor_evaluation selector: {key!r}.")
        if bool(require_all_sectors):
            raise ValueError("require_all_sectors requires an explicit A_s matrix-unit descriptor_evaluation selector.")
        return self._require_runtime().create_many(
            structures,
            concatenate=concatenate,
            return_batch=return_batch,
            real_if_scalar=real_if_scalar,
        )

    def create_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
        real_if_scalar=True,
        descriptor_evaluation=None,
        require_all_sectors=False,
        backend=None,
        native_library=None,
        execution_policy="direct",
    ):
        if self.metadata.get("descriptor_family") == "fixed_content_basis_plan":
            self._require_runtime()
        if self.metadata.get("descriptor_family") == "linear_tagged_cauchy_image":
            return self.create_many(
                read_ase_structures(paths, index=index), concatenate=concatenate,
                return_batch=return_batch, descriptor_evaluation=descriptor_evaluation,
                require_all_sectors=require_all_sectors,
                backend=backend, native_library=native_library,
                execution_policy=execution_policy,
            )
        key = self._normalize_descriptor_evaluation_selector(descriptor_evaluation)
        if key is not None:
            return self.create_many(
                read_ase_structures(paths, index=index),
                concatenate=concatenate,
                return_batch=return_batch,
                descriptor_evaluation=key,
                require_all_sectors=bool(require_all_sectors),
            )
        if bool(require_all_sectors):
            raise ValueError("require_all_sectors requires an explicit A_s matrix-unit descriptor_evaluation selector.")
        return self._require_runtime().create_from_files(
            paths,
            index=index,
            concatenate=concatenate,
            return_batch=return_batch,
            real_if_scalar=real_if_scalar,
        )

    def training_matrix(self, structures, *, properties=("energy", "forces", "stress"),
                        geometry_cache_dir=None):
        """Build linear energy, force, and stress rows from one tagged space.

        Purpose:
            Expose the same unweighted design rows used by linear fitting.
        Mathematical contract:
            Energy rows sum central-atom features; force rows are negative
            position derivatives; stress rows use ASE Voigt order and volume.
        Inputs:
            ASE structures and a subset of energy, forces, stress.
        Outputs:
            A matrix, compiler-owned column labels, and row descriptions.
        Does not:
            Read target values or perform regularized fitting.
        """

        if self.metadata.get("descriptor_family") != "linear_tagged_cauchy_image":
            raise NotImplementedError("training_matrix currently requires a tagged scalar descriptor space.")
        properties = tuple(properties)
        if not properties or len(set(properties)) != len(properties) or set(properties) - {"energy", "forces", "stress"}:
            raise ValueError("properties must be distinct energy, forces, or stress entries.")
        from ye3t_ace.tagged_cauchy_image_fit import tagged_cauchy_image_geometry_row

        species = tuple(self.metadata["tagged_cauchy_image_evaluator"].species_order)
        feature_count = len(self.feature_keys)
        beta_width = len(species) * feature_count
        labels = tuple((name, key) for name in species for key in self.feature_keys)
        labels += tuple((name, "atomic_offset") for name in species)
        blocks, row_labels = [], []
        for structure_index, atoms in enumerate(structures):
            row = tagged_cauchy_image_geometry_row(self, atoms, cache_dir=geometry_cache_dir)
            if "energy" in properties:
                blocks.append(np.concatenate((row["feature_sums"].reshape(-1), row["species_counts"]))[None, :])
                row_labels.append((structure_index, "energy", None))
            if "forces" in properties:
                blocks.append(np.pad(row["force_design"], ((0, 0), (0, len(species)))))
                row_labels.extend((structure_index, "forces", (atom, axis))
                                  for atom in range(len(atoms)) for axis in range(3))
            if "stress" in properties:
                if atoms.cell.rank != 3 or atoms.get_volume() <= 0.0:
                    raise ValueError("Stress rows require a positive-volume three-dimensional cell.")
                blocks.append(np.pad(row["stress_design"], ((0, 0), (0, len(species)))))
                row_labels.extend((structure_index, "stress", component) for component in
                                  ("xx", "yy", "zz", "yz", "xz", "xy"))
        matrix = np.concatenate(blocks, axis=0) if blocks else np.empty((0, beta_width + len(species)))
        return {"matrix": matrix, "columns": labels, "rows": tuple(row_labels),
                "convention": "total_energy_and_ASE_force_stress_rows_v1"}

    def to_spec(self):
        """Return the shared ``YE3TSpec`` recorded for this descriptor."""

        spec_payload = self.metadata.get("ye3t_spec", None)
        if spec_payload is not None:
            return YE3TSpec.from_dict(spec_payload)
        return self.representation.to_spec()

    def to_spec_file(self, path):
        """Write this descriptor's shared ``YE3TSpec`` view to JSON/YAML."""

        self.to_spec().to_file(path)
        return path

    def evaluate_global_coupler_reference(self, values, *, input_axis=-1, table_index=0, dtype=None, device=None):
        """Apply this descriptor's shared YE3T global coupler to supplied carrier values.

        This descriptor-first hook consumes the central `YE3TSpec` stored in
        metadata and applies one emitted coefficient table to a caller-supplied
        tensor.  It does not realize geometry carriers, perform the full
        induction/coset/angular descriptor contraction, or construct a model
        readout.
        """

        spec_payload = self.metadata.get("ye3t_spec", None)
        if spec_payload is None:
            raise NotImplementedError(
                "Descriptor-first global-coupler reference evaluation requires metadata['ye3t_spec']."
            )
        spec = YE3TSpec.from_dict(spec_payload)
        try:
            evaluation = evaluate_global_coupler_reference_torch(
                spec,
                values,
                table_index=table_index,
                input_axis=input_axis,
                dtype=dtype,
                device=device,
            )
        except ValueError as exc:
            raise NotImplementedError(
                "Descriptor-first global-coupler reference evaluation currently requires a concrete "
                "target_permutation ('trivial', 'antisymmetric', or 'young:<partition>'). "
                f"The descriptor spec has target_permutation={spec.target_permutation!r}."
            ) from exc
        evaluation.metadata.update(
            {
                "descriptor_first_flow": "YE3TRepresentation -> YE3TDescriptors / YE3TDescriptorSet -> evaluate_global_coupler_reference",
                "descriptor_family": self.metadata.get("descriptor_family"),
                "source_descriptor_runtime_status": self.metadata.get("runtime_status"),
                "geometry_carrier_realization": "caller_supplied_values",
            }
        )
        return evaluation

    def compile_global_coupler_family(self, *, input_Ls=None):
        """Compile concrete Young target-sector couplers from this descriptor's shared spec."""

        spec_payload = self.metadata.get("ye3t_spec", None)
        if spec_payload is None:
            raise NotImplementedError("Descriptor-first global coupler family compilation requires metadata['ye3t_spec'].")
        return CompileGlobalYE3TCouplerFamily(YE3TSpec.from_dict(spec_payload), input_Ls=input_Ls)

    def evaluate_global_coupler_family_reference(self, values, *, input_axis=-1, dtype=None, device=None):
        """Apply all concrete sector tables from this descriptor's global coupler family."""

        family = self.compile_global_coupler_family()
        evaluation = evaluate_global_coupler_family_reference_torch(
            family,
            values,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )
        evaluation.metadata.update(
            {
                "descriptor_first_flow": "YE3TRepresentation -> YE3TDescriptors / YE3TDescriptorSet -> evaluate_global_coupler_family_reference",
                "descriptor_family": self.metadata.get("descriptor_family"),
                "source_descriptor_runtime_status": self.metadata.get("runtime_status"),
                "geometry_carrier_realization": "caller_supplied_values",
            }
        )
        return evaluation

    def evaluate_global_coupler_family_descriptor_view(self, values, *, input_axis=-1, dtype=None, device=None):
        """Package a concrete-sector family reference evaluation as descriptor axes.

        This method concatenates already evaluated concrete-sector coefficient
        outputs along the coefficient/output axis while recording exact sector
        slices.  It keeps the direct-sum sector boundary in metadata and does
        not realize geometry carriers or perform full global descriptor
        contraction.
        """

        evaluation = self.evaluate_global_coupler_family_reference(
            values,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )
        if not evaluation.sector_evaluations:
            tensor = torch.as_tensor(values, dtype=dtype, device=device)
            axis = int(input_axis)
            if axis < 0:
                axis += int(tensor.ndim)
            empty_shape = list(tensor.shape)
            empty_shape[axis] = 0
            empty = torch.empty(tuple(empty_shape), dtype=tensor.dtype, device=tensor.device)
            return YE3TSectorDescriptorResult(
                values=empty,
                sector_slices=tuple(),
                sector_evaluations=tuple(),
                axes=("...", "direct_sum_young_sector_feature"),
                unflattened_axes=("...", "target_partition", "coefficient_axis"),
                metadata={
                    "runtime_status": "implemented_under_validation",
                    "descriptor_family": "global_coupler_family_descriptor_view",
                    "output_axis": int(axis),
                    "sector_count": 0,
                    "coefficient_axis_decomposition": (
                        "target_partition_lambda",
                        "L_R",
                        "alpha_label_metadata",
                        "flat_coefficient_axis",
                    ),
                    "coefficient_axes_status": (
                        "empty_direct_sum_target_partition_with_L_R_and_alpha_metadata_plus_flat_coefficient_axis"
                    ),
                    "available_explicit_axes": ("target_partition_lambda", "L_R", "alpha_label_metadata"),
                    "M_R_axis_status": "empty_direct_sum_no_sectors",
                    "missing_explicit_axes": ("tableau_or_matrix_unit", "M_R", "unflattened_alpha_coefficient_axis"),
                    "full_descriptor_contraction_status": "empty_direct_sum_sector_view_without_geometry_carriers",
                },
            )

        output_axis = int(input_axis)
        rank = int(evaluation.sector_evaluations[0].values.ndim)
        if output_axis < 0:
            output_axis += rank
        if output_axis < 0 or output_axis >= rank:
            raise ValueError(f"input_axis={input_axis!r} is outside evaluated tensor rank {rank}.")

        reference_shape = list(evaluation.sector_evaluations[0].values.shape)
        sector_slices = []
        start = 0
        tensors = []
        for sector_index, sector_eval in enumerate(evaluation.sector_evaluations):
            tensor = sector_eval.values
            if int(tensor.ndim) != rank:
                raise ValueError("All sector evaluations must have the same tensor rank for descriptor packaging.")
            shape = list(tensor.shape)
            comparable_shape = list(shape)
            comparable_shape[output_axis] = reference_shape[output_axis]
            if comparable_shape != reference_shape:
                raise ValueError(
                    "All sector evaluations must have matching non-coefficient axes for descriptor packaging."
                )
            width = int(shape[output_axis])
            stop = start + width
            target_partition = tuple(int(part) for part in evaluation.target_partitions[sector_index])
            global_labels = tuple(
                label.to_dict() if hasattr(label, "to_dict") else label
                for label in getattr(sector_eval.coupler, "labels", tuple())
            )
            sector_slices.append(
                {
                    "sector_index": int(sector_index),
                    "target_partition": target_partition,
                    "target_L_R": sector_eval.metadata.get("target_L_R"),
                    "M_R": (0,) if int(sector_eval.metadata.get("target_L_R", -1)) == 0 else None,
                    "M_R_axis_status": (
                        "scalar_M_R_0_implicit"
                        if int(sector_eval.metadata.get("target_L_R", -1)) == 0
                        else "not_resolved_in_coefficient_view"
                    ),
                    "global_labels": global_labels,
                    "alpha_label_count": int(len(global_labels)),
                    "start": int(start),
                    "stop": int(stop),
                    "width": int(width),
                    "coefficient_axes": tuple(sector_eval.coefficient_axes),
                    "coefficient_table_kind": sector_eval.metadata.get("coefficient_table_kind"),
                    "coefficient_table_hash": sector_eval.metadata.get("coefficient_table_hash"),
                    "full_descriptor_contraction_status": sector_eval.metadata.get(
                        "full_descriptor_contraction_status"
                    ),
                }
            )
            tensors.append(tensor)
            start = stop

        packaged = torch.cat(tuple(tensors), dim=output_axis)
        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "global_coupler_family_descriptor_view",
            "descriptor_first_flow": (
                "YE3TRepresentation -> YE3TDescriptors / YE3TDescriptorSet -> "
                "evaluate_global_coupler_family_descriptor_view"
            ),
            "source_descriptor_family": self.metadata.get("descriptor_family"),
            "source_runtime_status": self.metadata.get("runtime_status"),
            "geometry_carrier_realization": "caller_supplied_values",
            "output_axis": int(output_axis),
            "sector_count": int(len(sector_slices)),
            "target_partitions": tuple(record["target_partition"] for record in sector_slices),
            "coefficient_axis_decomposition": (
                "target_partition_lambda",
                "L_R",
                "alpha_label_metadata",
                "flat_coefficient_axis",
            ),
            "coefficient_axes_status": "direct_sum_target_partition_with_L_R_and_alpha_metadata_plus_flat_coefficient_axis",
            "available_explicit_axes": ("target_partition_lambda", "L_R", "alpha_label_metadata"),
            "M_R_axis_status": (
                "scalar_M_R_0_implicit"
                if all(record["M_R_axis_status"] == "scalar_M_R_0_implicit" for record in sector_slices)
                else "not_resolved_for_all_sectors"
            ),
            "missing_explicit_axes": ("tableau_or_matrix_unit", "M_R", "unflattened_alpha_coefficient_axis"),
            "full_descriptor_contraction_status": (
                "direct_sum_sector_tables_packaged_without_geometry_carriers_or_family_descriptor_contraction"
            ),
            "family_evaluation_metadata": dict(evaluation.metadata),
        }
        return YE3TSectorDescriptorResult(
            values=packaged,
            sector_slices=tuple(sector_slices),
            sector_evaluations=evaluation.sector_evaluations,
            axes=("...", "direct_sum_young_sector_feature"),
            unflattened_axes=("...", "target_partition", "coefficient_axis"),
            metadata=metadata,
        )

    def evaluate_ye3t_coefficient_descriptor_view(self, values, *, input_axis=-1, dtype=None, device=None):
        """Return a uniform descriptor view of supported YE3T coefficient tables.

        This dispatches from the shared ``YE3TSpec`` stored in descriptor
        metadata:

        - ``target_permutation="full_irrep_decomposition"`` uses the global
          coupler family descriptor view;
        - concrete Young/trivial targets use the single global-coupler table.

        The method still consumes caller-supplied carrier/coefficient-basis
        values.  It does not realize geometry carriers or complete the full
        induction/coset/angular descriptor contraction.
        """

        spec_payload = self.metadata.get("ye3t_spec", None)
        if spec_payload is None:
            raise NotImplementedError(
                "Descriptor-first YE3T coefficient descriptor view requires metadata['ye3t_spec']."
            )
        spec = YE3TSpec.from_dict(spec_payload)
        if spec.target_permutation == "full_irrep_decomposition":
            view = self.evaluate_global_coupler_family_descriptor_view(
                values,
                input_axis=input_axis,
                dtype=dtype,
                device=device,
            )
            view.metadata.update(
                {
                    "descriptor_family": "ye3t_coefficient_descriptor_view",
                    "dispatch_kind": "global_coupler_family",
                    "source_descriptor_view_family": "global_coupler_family_descriptor_view",
                }
            )
            return view

        evaluation = self.evaluate_global_coupler_reference(
            values,
            input_axis=input_axis,
            dtype=dtype,
            device=device,
        )
        target_partition = tuple(evaluation.metadata.get("target_partition") or ())
        dispatch_kind = "global_coupler"

        output_axis = int(input_axis)
        rank = int(evaluation.values.ndim)
        if output_axis < 0:
            output_axis += rank
        if output_axis < 0 or output_axis >= rank:
            raise ValueError(f"input_axis={input_axis!r} is outside evaluated tensor rank {rank}.")
        width = int(evaluation.values.shape[output_axis])
        global_labels = tuple(
            label.to_dict() if hasattr(label, "to_dict") else label
            for label in getattr(evaluation.coupler, "labels", tuple())
        )
        sector_slice = {
            "sector_index": 0,
            "target_partition": target_partition,
            "target_L_R": evaluation.metadata.get("target_L_R", int(spec.target_rotation.L_R)),
            "M_R": (0,) if int(evaluation.metadata.get("target_L_R", int(spec.target_rotation.L_R))) == 0 else None,
            "M_R_axis_status": (
                "scalar_M_R_0_implicit"
                if int(evaluation.metadata.get("target_L_R", int(spec.target_rotation.L_R))) == 0
                else "not_resolved_in_coefficient_view"
            ),
            "global_labels": global_labels,
            "alpha_label_count": int(len(global_labels)),
            "start": 0,
            "stop": int(width),
            "width": int(width),
            "coefficient_axes": tuple(evaluation.coefficient_axes),
            "coefficient_table_kind": evaluation.metadata.get("coefficient_table_kind"),
            "coefficient_table_hash": evaluation.metadata.get("coefficient_table_hash"),
            "full_descriptor_contraction_status": evaluation.metadata.get(
                "full_descriptor_contraction_status"
            ),
        }
        return YE3TSectorDescriptorResult(
            values=evaluation.values,
            sector_slices=(sector_slice,),
            sector_evaluations=(evaluation,),
            axes=("...", "ye3t_coefficient_feature"),
            unflattened_axes=("...", "target_partition", "coefficient_axis"),
            metadata={
                "runtime_status": "implemented_under_validation",
                "descriptor_family": "ye3t_coefficient_descriptor_view",
                "descriptor_first_flow": (
                    "YE3TRepresentation -> YE3TDescriptors / YE3TDescriptorSet -> "
                    "evaluate_ye3t_coefficient_descriptor_view"
                ),
                "source_descriptor_family": self.metadata.get("descriptor_family"),
                "source_runtime_status": self.metadata.get("runtime_status"),
                "geometry_carrier_realization": "caller_supplied_values",
                "dispatch_kind": dispatch_kind,
                "output_axis": int(output_axis),
                "sector_count": 1,
                "target_partitions": (target_partition,),
                "coefficient_axis_decomposition": (
                    "target_partition_lambda",
                    "L_R",
                    "alpha_label_metadata",
                    "flat_coefficient_axis",
                ),
                "coefficient_axes_status": "single_target_partition_with_L_R_and_alpha_metadata_plus_flat_coefficient_axis",
                "available_explicit_axes": ("target_partition_lambda", "L_R", "alpha_label_metadata"),
                "M_R_axis_status": sector_slice["M_R_axis_status"],
                "missing_explicit_axes": ("tableau_or_matrix_unit", "M_R", "unflattened_alpha_coefficient_axis"),
                "full_descriptor_contraction_status": (
                    "coefficient_descriptor_view_without_geometry_carriers_or_global_descriptor_contraction"
                ),
                "coefficient_evaluation_metadata": dict(evaluation.metadata),
            },
        )

    def create_A_s_matrix_unit_carriers(self, atoms):
        """Evaluate A_s slot-Specht matrix-unit carrier tensors for one structure.

        This method is intentionally separate from ``create`` because it returns
        carrier tensors plus sector metadata, not a flat descriptor matrix.  It
        is currently available only for ``ye3t_basis`` descriptors whose
        metadata reports ``matrix_unit_carrier_geometry_runtime=True``.
        """

        if not self.metadata.get("matrix_unit_carrier_geometry_runtime", False):
            raise NotImplementedError(
                "A_s matrix-unit carrier evaluation requires a ye3t_basis descriptor with "
                "readout_mode='ye3_slot_specht_power'. Full nontrivial descriptor "
                "coefficient contraction is still not implemented by this method."
            )
        if "lifted_density_config" not in self.metadata:
            raise ValueError("A_s matrix-unit carrier evaluation requires lifted_density_config metadata.")
        missing = [str(symbol) for symbol in atoms.get_chemical_symbols() if str(symbol) not in self.type_map]
        if missing:
            raise KeyError(f"Atoms contain elements missing from descriptor.type_map: {sorted(set(missing))}")
        from ye3t_ace.lifted_density import HybridACELiftedDensityConfig, HybridACELiftedDensityEnergyModel

        cfg = HybridACELiftedDensityConfig.from_dict(self.metadata["lifted_density_config"])
        model = HybridACELiftedDensityEnergyModel(cfg)
        positions = torch.as_tensor(atoms.get_positions(), dtype=model.config.torch_dtype)
        atom_types = torch.tensor(
            [int(self.type_map[str(symbol)]) for symbol in atoms.get_chemical_symbols()],
            dtype=torch.long,
            device=positions.device,
        )
        cell = None
        if hasattr(atoms, "cell") and atoms.cell is not None:
            cell = torch.as_tensor(atoms.cell.array, dtype=positions.dtype, device=positions.device)
        pbc = None
        if hasattr(atoms, "pbc"):
            pbc = torch.as_tensor(atoms.pbc, dtype=torch.bool, device=positions.device)
        return model.ye3_slot_specht_power_matrix_unit_carriers(
            positions,
            atom_types,
            cell=cell,
            pbc=pbc,
        )

    def create_matrix_unit_carriers(self, atoms):
        """Alias for the implemented A_s slot-Specht matrix-unit carrier hook."""

        return self.create_A_s_matrix_unit_carriers(atoms)

    def validate_A_s_matrix_unit_carriers(self, atoms):
        """Validate exact small-group matrix-unit metadata for A_s carriers.

        This descriptor-level report checks the finite slot-Specht matrix-unit
        algebra used by the current carrier runtime and verifies that evaluated
        carrier axes match the sector metadata.  It does not validate the full
        global Young--E3 induction/coset/angular descriptor contraction.
        """

        from ye3t_ace.lifted_density import (
            validate_slot_specht_matrix_unit_carrier_covariance,
            validate_slot_specht_matrix_units,
        )

        carriers, _blocks, sectors = self.create_A_s_matrix_unit_carriers(atoms)
        sector_reports = []
        for sector_index, carrier in enumerate(carriers):
            sector = dict(sectors[sector_index]) if sector_index < len(sectors) else {}
            partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
            if not partition:
                raise ValueError("A_s matrix-unit validation requires slot_specht_partition metadata.")
            power = int(sector.get("power"))
            slot_count = int(sum(partition))
            algebra_report = validate_slot_specht_matrix_units(
                slot_count,
                power,
                partition,
            )
            covariance_report = validate_slot_specht_matrix_unit_carrier_covariance(
                carrier,
                slot_count=slot_count,
                power=power,
                partition=partition,
            )
            expected_carrier_dim = int(slot_count) ** int(power)
            carrier_shape = tuple(int(dim) for dim in carrier.shape)
            carrier_dim_matches = bool(int(carrier.shape[-1]) == int(expected_carrier_dim))
            tableau_axes_match = bool(
                int(carrier.shape[1]) == int(algebra_report["specht_dimension"])
                and int(carrier.shape[2]) == int(algebra_report["specht_dimension"])
            )
            axis_metadata = _a_s_matrix_unit_sector_axis_metadata(
                sector,
                carrier,
                ye3t_axis=True,
            )
            role_coordinate_report = _a_s_role_coordinate_report(sector, carrier)
            global_coupler_compatibility = _a_s_matrix_unit_global_coupler_compatibility_report(
                sector,
                carrier,
            )
            matrix_unit_count = (
                int(algebra_report["specht_dimension"])
                * int(algebra_report["specht_dimension"])
            )
            rank_factorization_matches = bool(
                int(algebra_report["isotypic_rank"])
                == int(algebra_report["specht_dimension"])
                * int(algebra_report["expected_multiplicity"])
            )
            sector_reports.append(
                {
                    "sector_index": int(sector_index),
                    "slot_specht_partition": partition,
                    "slot_group": algebra_report["slot_group"],
                    "slot_count": int(slot_count),
                    "power": int(power),
                    "carrier_shape": carrier_shape,
                    "axis_metadata": axis_metadata,
                    "role_coordinate_report": role_coordinate_report,
                    "expected_carrier_dim": int(expected_carrier_dim),
                    "carrier_dim_matches": carrier_dim_matches,
                    "tableau_axes_match_specht_dimension": tableau_axes_match,
                    "global_coupler_compatibility": global_coupler_compatibility,
                    "specht_dimension": int(algebra_report["specht_dimension"]),
                    "isotypic_rank": int(algebra_report["isotypic_rank"]),
                    "slot_isotypic_multiplicity": int(algebra_report["expected_multiplicity"]),
                    "matrix_unit_count": int(matrix_unit_count),
                    "rank_factorization_matches_specht_dimension_times_multiplicity": (
                        rank_factorization_matches
                    ),
                    "matrix_unit_algebra": dict(algebra_report),
                    "matrix_unit_covariance": dict(covariance_report),
                    "passed": bool(
                        algebra_report["passed"]
                        and covariance_report["passed"]
                        and carrier_dim_matches
                        and tableau_axes_match
                        and axis_metadata["slot_tuple_carrier_dim_matches"]
                        and role_coordinate_report["passed"]
                        and rank_factorization_matches
                    ),
                }
            )
        validation_summary = _a_s_matrix_unit_validation_summary(sector_reports)
        return {
            "runtime_status": "implemented_under_validation",
            "validation_scope": "A_s_slot_specht_matrix_unit_carriers",
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=True),
            "validation_method": (
                "exact_small_slot_specht_matrix_unit_algebra_plus_runtime_axis_and_covariance_checks"
            ),
            "validation_summary": validation_summary,
            "descriptor_family": self.metadata.get("descriptor_family"),
            "source_descriptor_runtime_status": self.metadata.get("runtime_status"),
            "source_descriptor_runtime_status_detail": self.metadata.get("runtime_status_detail"),
            "matrix_unit_carrier_status": self.metadata.get("matrix_unit_carrier_status"),
            "full_descriptor_contraction_status": "carrier_validation_not_global_induction_contracted",
            "sector_count": int(len(sector_reports)),
            "sector_reports": tuple(sector_reports),
            "passed": bool(validation_summary["passed"]),
        }

    def validate_matrix_unit_carriers(self, atoms):
        """Alias for descriptor-level A_s matrix-unit carrier validation."""

        return self.validate_A_s_matrix_unit_carriers(atoms)

    def create_A_s_matrix_unit_descriptor(self, atoms):
        """Evaluate A_s matrix-unit carriers and expose a flat descriptor matrix.

        This is an intermediate descriptor-shaped runtime for the supported A_s
        carrier backend.  It preserves exact sector slices and explicitly marks
        that global induction/coset coefficient contraction is not complete.
        """

        carriers, blocks, sectors = self.create_A_s_matrix_unit_carriers(atoms)
        if carriers:
            n_atoms = int(carriers[0].shape[0])
            dtype = carriers[0].dtype
            device = carriers[0].device
        else:
            n_atoms = int(len(atoms))
            dtype = getattr(self.site_basis_config, "dtype", torch.float64)
            if not isinstance(dtype, torch.dtype):
                dtype = torch.float64
            device = None

        flat_blocks = []
        sector_slices = []
        compatibility_reports = []
        role_reports = []
        start = 0
        for sector_index, carrier in enumerate(carriers):
            if int(carrier.shape[0]) != n_atoms:
                raise ValueError("All A_s matrix-unit carrier blocks must have the same atom axis length.")
            flat = carrier.reshape(n_atoms, -1)
            stop = start + int(flat.shape[1])
            sector = dict(sectors[sector_index]) if sector_index < len(sectors) else {}
            axis_metadata = _a_s_matrix_unit_sector_axis_metadata(
                sector,
                carrier,
                ye3t_axis=False,
            )
            compatibility_report = _a_s_matrix_unit_global_coupler_compatibility_report(
                sector,
                carrier,
            )
            role_coordinate_report = _a_s_role_coordinate_report(sector, carrier)
            compatibility_reports.append(compatibility_report)
            role_reports.append(role_coordinate_report)
            sector_slices.append(
                {
                    "sector_index": int(sector_index),
                    "start": int(start),
                    "stop": int(stop),
                    "shape": tuple(int(dim) for dim in carrier.shape),
                    "axis_metadata": axis_metadata,
                    "slot_specht_partition": sector.get("slot_specht_partition"),
                    "power": sector.get("power"),
                    "role_coordinate_report": role_coordinate_report,
                    "global_coupler_compatibility": compatibility_report,
                    "runtime": sector.get("runtime", "slot_specht_matrix_unit_carrier"),
                }
            )
            flat_blocks.append(flat)
            start = stop

        if flat_blocks:
            values = torch.cat(tuple(flat_blocks), dim=1)
        else:
            values = torch.empty((n_atoms, 0), dtype=dtype, device=device)

        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_carrier_descriptor",
            "descriptor_first_flow": "YE3TRepresentation -> YE3TDescriptors.ye3t_basis -> YE3TDescriptorSet.create_A_s_matrix_unit_descriptor",
            "carrier_runtime": "slot_specht_matrix_unit_carrier",
            "full_descriptor_contraction_status": "carrier_flattened_not_global_induction_contracted",
            "contraction_plan": _a_s_matrix_unit_contraction_plan(ye3t_axis=False),
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=False),
            "global_coupler_compatibility_summary": _a_s_matrix_unit_global_coupler_compatibility_summary(
                compatibility_reports
            ),
            "role_coordinate_summary": _a_s_role_coordinate_summary(role_reports),
            "global_coupler_contracted": False,
            "coefficient_axes_status": "matrix_unit_carrier_axes_only",
            "coefficient_axis_decomposition": (
                "lambda_via_sector_slice",
                "tableau_row",
                "tableau_col",
                "slot_tuple_carrier",
            ),
            "lambda_axis_source": "sector_slices.slot_specht_partition",
            "multiplicity_axis_status": "slot_specht_sector_slices_only_not_global_multiplicity_resolved",
            "source_descriptor_runtime_status": self.metadata.get("runtime_status"),
            "source_descriptor_runtime_status_detail": self.metadata.get("runtime_status_detail"),
            "source_matrix_unit_carrier_status": self.metadata.get("matrix_unit_carrier_status"),
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            **_a_s_matrix_unit_runtime_scope_metadata(view="flat matrix-unit carrier descriptor"),
        }
        return ASMatrixUnitDescriptorResult(
            values=values,
            sector_slices=tuple(sector_slices),
            sectors=tuple(sectors),
            blocks=tuple(blocks),
            metadata=metadata,
        )

    def create_matrix_unit_descriptor(self, atoms):
        """Alias for the descriptor-shaped A_s matrix-unit carrier runtime."""

        return self.create_A_s_matrix_unit_descriptor(atoms)

    def create_A_s_matrix_unit_ye3t_axis_descriptor(self, atoms):
        """Expose supported A_s matrix-unit carriers with explicit YE3T axis metadata.

        This is a scalar-sector descriptor view for the current A_s
        slot-Specht matrix-unit carrier runtime.  It records lambda/tableau,
        L_R=0, M_R=0, and alpha/carrier-axis metadata, but it does not perform
        global induction/coset/angular coefficient contraction.
        """

        carriers, blocks, sectors = self.create_A_s_matrix_unit_carriers(atoms)
        if carriers:
            n_atoms = int(carriers[0].shape[0])
            dtype = carriers[0].dtype
            device = carriers[0].device
        else:
            n_atoms = int(len(atoms))
            dtype = torch.float64
            device = None

        flat_blocks = []
        sector_slices = []
        compatibility_reports = []
        role_reports = []
        start = 0
        for sector_index, carrier in enumerate(carriers):
            sector = dict(sectors[sector_index]) if sector_index < len(sectors) else {}
            target_L_R = int(sector.get("target_L_R", 0))
            if target_L_R != 0:
                raise NotImplementedError(
                    "A_s matrix-unit YE3T-axis descriptor currently supports only scalar target_L_R=0 "
                    "because the implemented carrier hook does not expose an explicit M_R axis."
                )
            if int(carrier.shape[0]) != n_atoms:
                raise ValueError("All A_s matrix-unit carrier blocks must have the same atom axis length.")
            shaped = carrier.reshape(
                n_atoms,
                int(carrier.shape[1]),
                int(carrier.shape[2]),
                1,
                1,
                int(carrier.shape[3]),
            )
            flat = shaped.reshape(n_atoms, -1)
            stop = start + int(flat.shape[1])
            partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
            axis_metadata = _a_s_matrix_unit_sector_axis_metadata(
                sector,
                carrier,
                ye3t_axis=True,
            )
            compatibility_report = _a_s_matrix_unit_global_coupler_compatibility_report(
                sector,
                carrier,
            )
            role_coordinate_report = _a_s_role_coordinate_report(sector, carrier)
            compatibility_reports.append(compatibility_report)
            role_reports.append(role_coordinate_report)
            sector_slices.append(
                {
                    "sector_index": int(sector_index),
                    "start": int(start),
                    "stop": int(stop),
                    "shape": tuple(int(dim) for dim in shaped.shape),
                    "axis_metadata": axis_metadata,
                    "lambda": partition,
                    "slot_specht_partition": partition,
                    "tableau_shape": (int(carrier.shape[1]), int(carrier.shape[2])),
                    "L_R": 0,
                    "M_R_values": (0,),
                    "alpha_axis": "slot_tuple_carrier",
                    "alpha_size": int(carrier.shape[3]),
                    "contraction_plan_status": "carrier_axes_only_not_global_ye3t_contraction",
                    "missing_global_contraction_stages": tuple(
                        _a_s_matrix_unit_contraction_plan(ye3t_axis=True)["missing_stages"]
                    ),
                    "role_coordinate_report": role_coordinate_report,
                    "global_coupler_compatibility": compatibility_report,
                    "global_coupler_contracted": False,
                    "runtime": sector.get("runtime", "slot_specht_matrix_unit_carrier"),
                }
            )
            flat_blocks.append(flat)
            start = stop

        if flat_blocks:
            values = torch.cat(tuple(flat_blocks), dim=1)
        else:
            values = torch.empty((n_atoms, 0), dtype=dtype, device=device)

        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_ye3t_axis_descriptor",
            "descriptor_first_flow": "YE3TRepresentation -> YE3TDescriptors.ye3t_basis -> YE3TDescriptorSet.create_A_s_matrix_unit_ye3t_axis_descriptor",
            "carrier_runtime": "slot_specht_matrix_unit_carrier",
            "full_descriptor_contraction_status": "A_s_matrix_unit_axes_exposed_not_global_induction_contracted",
            "contraction_plan": _a_s_matrix_unit_contraction_plan(ye3t_axis=True),
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=True),
            "global_coupler_compatibility_summary": _a_s_matrix_unit_global_coupler_compatibility_summary(
                compatibility_reports
            ),
            "role_coordinate_summary": _a_s_role_coordinate_summary(role_reports),
            "global_coupler_contracted": False,
            "coefficient_axes_status": "lambda_tableau_L0_M0_alpha_slot_tuple_carrier",
            "coefficient_axis_decomposition": (
                "lambda_via_sector_slice",
                "tableau_row",
                "tableau_col",
                "L_R",
                "M_R",
                "alpha_slot_tuple_carrier",
            ),
            "lambda_axis_source": "sector_slices.lambda",
            "multiplicity_axis_status": "slot_specht_sector_slices_only_not_global_multiplicity_resolved",
            "M_R_axis_status": "explicit_singleton_M_R_0_for_scalar_sectors",
            "source_descriptor_runtime_status": self.metadata.get("runtime_status"),
            "source_descriptor_runtime_status_detail": self.metadata.get("runtime_status_detail"),
            "source_matrix_unit_carrier_status": self.metadata.get("matrix_unit_carrier_status"),
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            **_a_s_matrix_unit_runtime_scope_metadata(view="YE3T-axis matrix-unit carrier descriptor"),
        }
        return ASMatrixUnitDescriptorResult(
            values=values,
            sector_slices=tuple(sector_slices),
            sectors=tuple(sectors),
            blocks=tuple(blocks),
            axes=("atom", "flattened_lambda_tableau_L_R_M_R_alpha_feature"),
            unflattened_axes=(
                "atom",
                "tableau_row",
                "tableau_col",
                "L_R",
                "M_R",
                "alpha_slot_tuple_carrier",
            ),
            metadata=metadata,
        )

    def create_matrix_unit_ye3t_axis_descriptor(self, atoms):
        """Alias for the scalar A_s matrix-unit YE3T-axis descriptor view."""

        return self.create_A_s_matrix_unit_ye3t_axis_descriptor(atoms)

    def _create_A_s_equivariant_matrix_unit_carriers(self, atoms, *, target_L_R_values):
        if "lifted_density_config" not in self.metadata:
            raise ValueError("A_s equivariant matrix-unit carrier evaluation requires lifted_density_config metadata.")
        missing = [str(symbol) for symbol in atoms.get_chemical_symbols() if str(symbol) not in self.type_map]
        if missing:
            raise KeyError(f"Atoms contain elements missing from descriptor.type_map: {sorted(set(missing))}")
        from ye3t_ace.lifted_density import (
            HybridACELiftedDensityConfig,
            HybridACELiftedDensityEnergyModel,
            ye3_slot_specht_power_equivariant_matrix_unit_carriers,
        )

        cfg = HybridACELiftedDensityConfig.from_dict(self.metadata["lifted_density_config"])
        model = HybridACELiftedDensityEnergyModel(cfg)
        positions = torch.as_tensor(atoms.get_positions(), dtype=model.config.torch_dtype)
        atom_types = torch.tensor(
            [int(self.type_map[str(symbol)]) for symbol in atoms.get_chemical_symbols()],
            dtype=torch.long,
            device=positions.device,
        )
        cell = None
        if hasattr(atoms, "cell") and atoms.cell is not None:
            cell = torch.as_tensor(atoms.cell.array, dtype=positions.dtype, device=positions.device)
        pbc = None
        if hasattr(atoms, "pbc"):
            pbc = torch.as_tensor(atoms.pbc, dtype=torch.bool, device=positions.device)
        density = model.filtered_density(
            positions,
            atom_types,
            cell=cell,
            pbc=pbc,
        )
        lifted = cfg.lifted_density
        return ye3_slot_specht_power_equivariant_matrix_unit_carriers(
            density,
            lifted.channels,
            target_L_R_values=tuple(int(value) for value in target_L_R_values),
            max_power=lifted.ye3_max_power,
            optimization_policy=lifted.ye3_optimization_policy,
            slot_specht_partitions=lifted.ye3_slot_specht_partitions,
            include_rank1=lifted.ye3_include_rank1,
            rank_nmax=lifted.ye3_rank_nmax,
            rank_lmax=lifted.ye3_rank_lmax,
            rank_lmin=lifted.ye3_rank_lmin,
        )

    def create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(
        self,
        atoms,
        *,
        require_all_sectors=False,
    ):
        """Assemble multiplicity-resolved A_s slot-Specht carriers with explicit ``M_R``.

        This route keeps the current public selector name for compatibility,
        but the implemented tensor assembly is the slot-Specht matrix-unit
        decomposition plus explicit angular ``M_R`` coupling.  It does not
        claim that the current A_s geometry carrier has been assembled through
        a full family-wide global Young-coupler backend.
        """

        requested_target_L_R = int(self._A_s_matrix_unit_requested_target_L_R())
        carriers, blocks, sectors = self._create_A_s_equivariant_matrix_unit_carriers(
            atoms,
            target_L_R_values=(requested_target_L_R,),
        )
        if carriers:
            n_atoms = int(carriers[0].shape[0])
            dtype = carriers[0].dtype
            device = carriers[0].device
        else:
            n_atoms = int(len(atoms))
            dtype = torch.float64
            device = None

        from ye3t_ace.lifted_density import _slot_specht_anchor_multiplicity_basis_torch_cached

        feature_blocks = []
        sector_slices = []
        evaluated_sector_reports = []
        role_reports = []
        start = 0
        requested_target_M_R_values = self._A_s_matrix_unit_requested_M_R_values()
        for sector_index, carrier in enumerate(carriers):
            sector = dict(sectors[sector_index]) if sector_index < len(sectors) else {}
            slot_tuple_view = carrier[..., 0] if int(carrier.shape[-1]) > 0 else carrier[..., :0]
            role_coordinate_report = _a_s_role_coordinate_report(sector, slot_tuple_view)
            role_reports.append(role_coordinate_report)
            if int(carrier.shape[0]) != n_atoms:
                raise ValueError("All A_s matrix-unit carrier blocks must have the same atom axis length.")
            if int(carrier.shape[1]) != int(carrier.shape[2]):
                raise ValueError("A_s multiplicity-resolved slot-Specht assembly expects square tableau axes.")
            partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
            power = int(sector.get("power", 0))
            slot_count = int(sum(partition)) if partition else 0
            target_L_R = int(sector.get("target_L_R", 0))
            target_partition = partition
            full_M_R_values = tuple(int(value) for value in sector.get("M_R_values", ()))
            if requested_target_M_R_values is None:
                selected_indices = tuple(range(len(full_M_R_values)))
                M_R_values = full_M_R_values
            else:
                selected_indices = tuple(
                    index for index, value in enumerate(full_M_R_values) if int(value) in requested_target_M_R_values
                )
                M_R_values = tuple(int(full_M_R_values[index]) for index in selected_indices)
            if not M_R_values:
                if require_all_sectors:
                    raise NotImplementedError(
                        "Requested target_M_R_values are incompatible with the emitted A_s slot-Specht sector."
                    )
                continue
            multiplicity_basis = _slot_specht_anchor_multiplicity_basis_torch_cached(
                slot_count,
                int(power),
                partition,
                dtype=carrier.dtype,
                device=carrier.device,
            )
            multiplicity_count = int(multiplicity_basis.shape[1])
            anchor_slice = carrier[:, 0, :, :, :]
            anchor_slice = anchor_slice.index_select(
                -1,
                torch.tensor(selected_indices, dtype=torch.long, device=anchor_slice.device),
            )
            evaluated_shaped = torch.einsum("nctm,tk->nkcm", anchor_slice, multiplicity_basis)
            flat = evaluated_shaped.reshape(n_atoms, -1)
            stop = start + int(flat.shape[1])
            compatibility_report = _a_s_matrix_unit_global_coupler_compatibility_report(sector, slot_tuple_view)
            slot_candidate = compatibility_report.get("slot_resolved_product_slot_candidate")
            compatibility_report = {
                **compatibility_report,
                "status": "slot_resolved_multiplicity_basis_applied_to_matrix_unit_axes",
                "slot_resolved_product_slot_candidate": {
                    **({} if slot_candidate is None else dict(slot_candidate)),
                    "global_coupler_applied_to_A_s_carrier": False,
                    "applied_axis": "anchor_tableau_row_then_exact_slot_multiplicity_basis",
                },
            }
            sector_slices.append(
                {
                    "sector_index": int(sector_index),
                    "start": int(start),
                    "stop": int(stop),
                    "shape": tuple(int(dim) for dim in evaluated_shaped.shape),
                    "source_matrix_unit_shape": tuple(int(dim) for dim in carrier.shape),
                    "slot_specht_partition": partition,
                    "target_partition": target_partition,
                    "power": int(power),
                    "slot_group_size": int(slot_count),
                    "L_R": int(target_L_R),
                    "M_R_values": M_R_values,
                    "slot_specht_dimension": int(carrier.shape[2]),
                    "multiplicity_count": int(multiplicity_count),
                    "multiplicity_basis_anchor_row": 0,
                    "coefficient_axes": (
                        "slot_specht_multiplicity_copy",
                        "target_specht_tableau_coordinate",
                        "M_R",
                    ),
                    "applied_axis": "slot_tuple_carrier",
                    "lambda_axis_source": "target_partition",
                    "multiplicity_axis_status": "explicit_slot_specht_isotypic_multiplicity_axis",
                    "tableau_coordinate_axis_status": "explicit_target_specht_tableau_coordinate_axis",
                    "M_R_axis_status": (
                        "explicit_singleton_M_R_0_axis"
                        if int(len(M_R_values)) == 1
                        else "explicit_M_R_resolved_output_axis"
                    ),
                    "role_coordinate_report": role_coordinate_report,
                    "global_coupler_compatibility": compatibility_report,
                    "global_coupler_contracted": False,
                    "runtime": "slot_resolved_multiplicity_resolved_A_s_matrix_unit_axes",
                }
            )
            evaluated_sector_reports.append(
                {
                    "sector_index": int(sector_index),
                    "slot_specht_partition": partition,
                    "target_partition": target_partition,
                    "L_R": int(target_L_R),
                    "power": int(power),
                    "slot_group_size": int(slot_count),
                    "output_width": int(flat.shape[1]),
                    "slot_tuple_carrier_dim": int(carrier.shape[3]),
                    "slot_specht_dimension": int(carrier.shape[2]),
                    "multiplicity_count": int(multiplicity_count),
                    "M_R_values": M_R_values,
                    "role_coordinate_report": role_coordinate_report,
                    "global_coupler_certificate_passed": bool(
                        compatibility_report.get("slot_resolved_product_slot_candidate", {}).get("certificate_passed", False)
                    ),
                    "status": "evaluated_under_validation",
                }
            )
            feature_blocks.append(flat)
            start = stop

        if feature_blocks:
            values = torch.cat(tuple(feature_blocks), dim=1)
        else:
            values = torch.empty((n_atoms, 0), dtype=dtype, device=device)
        total_sector_count = int(len(sectors))
        evaluated_sector_count = int(len(evaluated_sector_reports))
        skipped_sector_count = int(total_sector_count - evaluated_sector_count)
        all_sector_coverage_passed = bool(evaluated_sector_count == total_sector_count)
        sector_coverage_status = (
            "all_matrix_unit_sectors_evaluated_under_validation"
            if skipped_sector_count == 0
            else "partial_slot_resolved_sector_coverage_requested_M_R_subset_or_missing_sector"
        )
        evaluated_sector_signatures = tuple(
            {
                "sector_index": int(report["sector_index"]),
                "slot_specht_partition": tuple(int(part) for part in report["slot_specht_partition"]),
                "target_partition": tuple(int(part) for part in report["target_partition"]),
                "L_R": int(report.get("L_R", 0)),
                "M_R_values": tuple(int(value) for value in report.get("M_R_values", ())),
            }
            for report in evaluated_sector_reports
        )
        skipped_status_counts = {}
        skipped_reason_summary = tuple()
        supported_M_R_values = tuple(
            sorted(
                {
                    int(value)
                    for report in evaluated_sector_reports
                    for value in tuple(report.get("M_R_values", ()))
                }
            )
        )
        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_slot_resolved_global_coupler_descriptor",
            "descriptor_first_flow": (
                "YE3TRepresentation -> YE3TDescriptors.ye3t_basis -> "
                "YE3TDescriptorSet.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor"
            ),
            "runtime_scope": "A_s_matrix_unit_slot_resolved_multiplicity_resolved_angular_assembly",
            "carrier_runtime": "slot_specht_matrix_unit_equivariant_carrier",
            "global_coupler_contracted": False,
            "full_descriptor_contraction_status": (
                "slot_specht_multiplicity_resolved_irreducible_direct_sum_matrix_unit_assembly"
            ),
            "global_ye3_descriptor_status": (
                "implemented_under_validation_for_irreducible_direct_sum_slot_resolved_multiplicity_resolved_M_R_assembled_matrix_unit_sectors"
            ),
            "global_ye3_descriptor_missing_stages": (
                "repeated_channel_global_block_map_application",
                "factorized_large_runtime_kernel",
                "full_sector_catalog_completeness_validation",
            ),
            "implemented_stages": (
                "A_s_slot_density",
                "slot_permutation_action",
                "slot_specht_matrix_unit_projection",
                "slot_tuple_angular_coupling_to_target_L_R",
                "explicit_slot_specht_isotypic_multiplicity_basis",
                "explicit_target_specht_tableau_coordinate_axis",
                "explicit_M_R_resolved_output_axis_packaging",
            ),
            "missing_stages": (
                "repeated_channel_global_block_map_application",
                "factorized_large_runtime_kernel",
                "full_sector_catalog_completeness_validation",
            ),
            "coefficient_axis_decomposition": (
                "slot_specht_multiplicity_copy",
                "target_specht_tableau_coordinate",
                "M_R",
            ),
            "mathematical_feature_axes": (
                "atom",
                "lambda_via_sector_slice",
                "slot_specht_multiplicity_copy",
                "target_specht_tableau_coordinate",
                "L_R_via_sector_slice",
                "M_R_explicit_output_axis",
            ),
            "feature_axis_status": {
                "lambda": "sector_slices.target_partition",
                "tableau_or_matrix_unit": "anchor tableau row mapped to explicit tableau coordinate axis",
                "L_R": "sector_slices.L_R",
                "slot_specht_multiplicity_copy": "sector_slices.multiplicity_count",
                "target_specht_tableau_coordinate": "sector_slices.slot_specht_dimension",
                "M_R": "sector_slices.M_R_values",
            },
            "irreducible_direct_sum_family_assembly": True,
            "global_rectangular_alpha_backend_required": False,
            "family_assembly_layout": (
                "direct_sum_over_sector_slices_with_explicit_target_partition_L_R_multiplicity_tableau_and_M_R_axes"
            ),
            "supported_M_R_values": supported_M_R_values,
            "require_all_sectors": bool(require_all_sectors),
            "total_matrix_unit_sector_count": total_sector_count,
            "evaluated_sector_count": evaluated_sector_count,
            "skipped_sector_count": skipped_sector_count,
            "all_sector_coverage_passed": all_sector_coverage_passed,
            "partial_sector_coverage_allowed": bool(skipped_sector_count > 0 and not bool(require_all_sectors)),
            "sector_coverage_status": sector_coverage_status,
            "evaluated_sector_signatures": evaluated_sector_signatures,
            "skipped_status_counts": skipped_status_counts,
            "skipped_reason_summary": skipped_reason_summary,
            "evaluated_sector_reports": tuple(evaluated_sector_reports),
            "skipped_sector_reports": tuple(),
            "role_coordinate_summary": _a_s_role_coordinate_summary(role_reports),
            "source_descriptor_runtime_status": self.metadata.get("runtime_status"),
            "source_descriptor_runtime_status_detail": self.metadata.get("runtime_status_detail"),
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
        }
        return ASMatrixUnitDescriptorResult(
            values=values,
            sector_slices=tuple(sector_slices),
            sectors=tuple(sectors),
            blocks=tuple(blocks),
            axes=("atom", "flattened_slot_specht_multiplicity_tableau_M_R_feature"),
            unflattened_axes=(
                "atom",
                "slot_specht_multiplicity_copy",
                "target_specht_tableau_coordinate",
                "M_R",
            ),
            metadata=metadata,
        )

    def create_matrix_unit_slot_resolved_global_coupler_descriptor(self, atoms, *, require_all_sectors=False):
        """Alias for compatible A_s matrix-unit/global-coupler table application."""

        return self.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(
            atoms,
            require_all_sectors=require_all_sectors,
        )

    def validate_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(
        self,
        atoms,
        *,
        atol=1.0e-10,
        rtol=1.0e-10,
        require_all_sectors=False,
    ):
        """Validate multiplicity-resolved slot-Specht plus angular A_s assembly."""

        requested_target_L_R = int(self._A_s_matrix_unit_requested_target_L_R())
        carriers, _blocks, sectors = self._create_A_s_equivariant_matrix_unit_carriers(
            atoms,
            target_L_R_values=(requested_target_L_R,),
        )
        result = self.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(
            atoms,
            require_all_sectors=require_all_sectors,
        )
        from ye3t_ace.lifted_density import _slot_specht_anchor_multiplicity_basis_torch_cached

        sector_reports = []
        max_abs_error = 0.0
        irreducible_dimension_reports = []
        for result_sector_index, record in enumerate(result.sector_slices):
            source_sector_index = int(record["sector_index"])
            carrier = carriers[source_sector_index]
            sector = dict(sectors[source_sector_index])
            partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
            slot_count = int(sum(partition)) if partition else 0
            multiplicity_basis = _slot_specht_anchor_multiplicity_basis_torch_cached(
                slot_count,
                int(sector.get("power", 0)),
                partition,
                dtype=carrier.dtype,
                device=carrier.device,
            )
            full_M_R_values = tuple(int(value) for value in sector.get("M_R_values", ()))
            requested_M_R_values = self._A_s_matrix_unit_requested_M_R_values()
            if requested_M_R_values is None:
                selected_indices = tuple(range(len(full_M_R_values)))
            else:
                selected_indices = tuple(
                    index for index, value in enumerate(full_M_R_values) if int(value) in requested_M_R_values
                )
            anchor_slice = carrier[:, 0, :, :, :]
            anchor_slice = anchor_slice.index_select(
                -1,
                torch.tensor(selected_indices, dtype=torch.long, device=anchor_slice.device),
            )
            reference = torch.einsum("nctm,tk->nkcm", anchor_slice, multiplicity_basis)
            actual = result.sector_values(result_sector_index)
            close = bool(torch.allclose(actual, reference, atol=float(atol), rtol=float(rtol)))
            error = float(torch.max(torch.abs(actual - reference)).detach().cpu()) if actual.numel() else 0.0
            max_abs_error = max(max_abs_error, error)
            slot_specht_dimension = int(carrier.shape[2])
            multiplicity_count = int(multiplicity_basis.shape[1])
            expected_isotypic_dimension = int(sector.get("slot_projector_rank", slot_specht_dimension * multiplicity_count))
            observed_isotypic_dimension = int(slot_specht_dimension * multiplicity_count)
            irreducible_dimension_match = bool(observed_isotypic_dimension == expected_isotypic_dimension)
            selected_M_R_count = int(len(selected_indices))
            sector_reports.append(
                {
                    "result_sector_index": int(result_sector_index),
                    "source_sector_index": int(source_sector_index),
                    "slot_specht_partition": partition,
                    "target_partition": tuple(int(part) for part in record.get("target_partition", ())),
                    "multiplicity_count": multiplicity_count,
                    "slot_specht_dimension": slot_specht_dimension,
                    "slot_projector_rank": int(sector.get("slot_projector_rank", expected_isotypic_dimension)),
                    "expected_isotypic_dimension": expected_isotypic_dimension,
                    "observed_isotypic_dimension": observed_isotypic_dimension,
                    "selected_M_R_count": selected_M_R_count,
                    "observed_feature_width": int(actual.reshape(int(actual.shape[0]), -1).shape[1]),
                    "expected_feature_width": int(observed_isotypic_dimension * selected_M_R_count),
                    "irreducible_emitted_sector_complete": irreducible_dimension_match,
                    "matches_reference_application": close,
                    "max_abs_error": error,
                    "atol": float(atol),
                    "rtol": float(rtol),
                }
            )
            irreducible_dimension_reports.append(
                {
                    "result_sector_index": int(result_sector_index),
                    "source_sector_index": int(source_sector_index),
                    "slot_specht_partition": partition,
                    "expected_isotypic_dimension": expected_isotypic_dimension,
                    "observed_isotypic_dimension": observed_isotypic_dimension,
                    "slot_projector_rank": int(sector.get("slot_projector_rank", expected_isotypic_dimension)),
                    "selected_M_R_count": selected_M_R_count,
                    "observed_feature_width": int(actual.reshape(int(actual.shape[0]), -1).shape[1]),
                    "expected_feature_width": int(observed_isotypic_dimension * selected_M_R_count),
                    "passed": irreducible_dimension_match
                    and int(actual.reshape(int(actual.shape[0]), -1).shape[1])
                    == int(observed_isotypic_dimension * selected_M_R_count),
                }
            )
        emitted_sector_irreducible_completeness_passed = bool(
            irreducible_dimension_reports
            and all(bool(report["passed"]) for report in irreducible_dimension_reports)
        )
        passed = bool(
            result.metadata.get("runtime_status") == "implemented_under_validation"
            and result.metadata.get("all_sector_coverage_passed") is True
            and sector_reports
            and emitted_sector_irreducible_completeness_passed
            and all(report["matches_reference_application"] for report in sector_reports)
        )
        return {
            "runtime_status": "implemented_under_validation" if passed else "planned_not_public",
            "validation_scope": "A_s_matrix_unit_slot_resolved_global_coupler_descriptor",
            "passed": passed,
            "descriptor_family": result.metadata.get("descriptor_family"),
            "runtime_scope": result.metadata.get("runtime_scope"),
            "global_ye3_descriptor_status": result.metadata.get("global_ye3_descriptor_status"),
            "full_descriptor_contraction_status": result.metadata.get("full_descriptor_contraction_status"),
            "implemented_stages": tuple(result.metadata.get("implemented_stages", ())),
            "missing_stages": tuple(result.metadata.get("missing_stages", ())),
            "mathematical_feature_axes": tuple(result.metadata.get("mathematical_feature_axes", ())),
            "feature_axis_status": dict(result.metadata.get("feature_axis_status", {})),
            "supported_M_R_values": tuple(result.metadata.get("supported_M_R_values", ())),
            "global_coupler_contracted": bool(result.metadata.get("global_coupler_contracted", False)),
            "irreducible_direct_sum_family_assembly": bool(
                result.metadata.get("irreducible_direct_sum_family_assembly", False)
            ),
            "global_rectangular_alpha_backend_required": bool(
                result.metadata.get("global_rectangular_alpha_backend_required", True)
            ),
            "family_assembly_layout": result.metadata.get("family_assembly_layout"),
            "full_global_alpha_multiplicity_layout": False,
            "non_scalar_M_R_resolved_A_s_carrier_axes": any(
                len(tuple(report.get("M_R_values", ()))) > 1
                for report in tuple(result.metadata.get("evaluated_sector_reports", ()))
            ),
            "emitted_sector_catalog_scope": (
                "checks irreducible completeness for each emitted sector only; it is not a proof that all admissible sectors are emitted"
            ),
            "emitted_sector_irreducible_completeness_passed": emitted_sector_irreducible_completeness_passed,
            "emitted_sector_irreducible_dimension_reports": tuple(irreducible_dimension_reports),
            "require_all_sectors": bool(require_all_sectors),
            "total_matrix_unit_sector_count": int(result.metadata.get("total_matrix_unit_sector_count", 0)),
            "evaluated_sector_count": int(len(sector_reports)),
            "skipped_sector_count": int(result.metadata.get("skipped_sector_count", 0)),
            "all_sector_coverage_passed": bool(result.metadata.get("all_sector_coverage_passed", False)),
            "partial_sector_coverage_allowed": bool(result.metadata.get("partial_sector_coverage_allowed", False)),
            "sector_coverage_status": result.metadata.get("sector_coverage_status"),
            "evaluated_sector_signatures": tuple(result.metadata.get("evaluated_sector_signatures", ())),
            "skipped_status_counts": dict(result.metadata.get("skipped_status_counts", {})),
            "skipped_reason_summary": tuple(result.metadata.get("skipped_reason_summary", ())),
            "sector_reports": tuple(sector_reports),
            "skipped_sector_reports": tuple(result.metadata.get("skipped_sector_reports", ())),
            "max_abs_error": float(max_abs_error),
            "atol": float(atol),
            "rtol": float(rtol),
            "validation_method": "exact_slot_specht_anchor_multiplicity_basis_reapplication_to_equivariant_matrix_unit_carriers",
            "claim_scope": (
                "validates the implemented irreducible direct-sum slot-Specht multiplicity-resolved "
                "plus explicit M_R assembly for the current emitted A_s matrix-unit sector catalog; "
                "it does not by itself prove sector-catalog completeness beyond the emitted sectors"
            ),
        }

    def validate_matrix_unit_slot_resolved_global_coupler_descriptor(self, atoms, **kwargs):
        """Alias for slot-resolved A_s/global-coupler descriptor validation."""

        return self.validate_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(atoms, **kwargs)

    def create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor_many(
        self,
        structures,
        *,
        concatenate=False,
        return_batch=False,
        require_all_sectors=False,
    ):
        """Evaluate compatible slot-resolved global-coupler descriptors for structures."""

        results = []
        row_slices = []
        start = 0
        feature_count = None
        for atoms in structures:
            result = self.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(
                atoms,
                require_all_sectors=require_all_sectors,
            )
            if feature_count is None:
                feature_count = int(result.num_features)
            elif int(result.num_features) != feature_count:
                raise ValueError(
                    "All A_s slot-resolved global-coupler descriptor results must have the same feature count "
                    "for batching."
                )
            results.append(result)
            stop = start + int(result.num_atoms)
            row_slices.append((int(start), int(stop)))
            start = stop
        total_sector_count = sum(int(result.metadata.get("total_matrix_unit_sector_count", 0)) for result in results)
        evaluated_sector_count = sum(int(result.metadata.get("evaluated_sector_count", 0)) for result in results)
        skipped_sector_count = sum(int(result.metadata.get("skipped_sector_count", 0)) for result in results)
        all_sector_coverage_passed = bool(
            results and all(bool(result.metadata.get("all_sector_coverage_passed", False)) for result in results)
        )
        sector_coverage_statuses = tuple(
            str(result.metadata.get("sector_coverage_status", "unknown")) for result in results
        )
        evaluated_sector_signatures = tuple(
            signature
            for result in results
            for signature in tuple(result.metadata.get("evaluated_sector_signatures", ()))
        )
        skipped_status_counts = {}
        skipped_reason_counts = {}
        for result in results:
            for status, count in dict(result.metadata.get("skipped_status_counts", {})).items():
                skipped_status_counts[str(status)] = int(skipped_status_counts.get(str(status), 0)) + int(count)
            for report in tuple(result.metadata.get("skipped_reason_summary", ())):
                key = (str(report.get("status")), str(report.get("reason", "")))
                skipped_reason_counts[key] = int(skipped_reason_counts.get(key, 0)) + int(report.get("count", 0))
        skipped_reason_summary = tuple(
            {
                "status": status,
                "reason": reason,
                "count": int(count),
            }
            for (status, reason), count in sorted(skipped_reason_counts.items())
        )
        evaluated_component_map_families = tuple(
            families
            for result in results
            for families in tuple(result.metadata.get("evaluated_component_map_families", ()))
        )
        evaluated_component_map_kinds = tuple(
            kinds
            for result in results
            for kinds in tuple(result.metadata.get("evaluated_component_map_kinds", ()))
        )
        evaluated_component_map_roles = tuple(
            roles
            for result in results
            for roles in tuple(result.metadata.get("evaluated_component_map_roles", ()))
        )
        evaluated_component_map_sequence_all_validated = bool(
            results
            and evaluated_component_map_kinds
            and all(
                bool(result.metadata.get("evaluated_component_map_sequence_all_validated", False))
                for result in results
            )
        )
        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_slot_resolved_global_coupler_descriptor_batch",
            "runtime_scope": "A_s_matrix_unit_slot_resolved_M_R_resolved_global_coupler_application",
            "full_descriptor_contraction_status": (
                "batched_irreducible_direct_sum_slot_specht_multiplicity_resolved_matrix_unit_assembly"
            ),
            "global_ye3_descriptor_status": (
                "implemented_under_validation_for_batched_irreducible_direct_sum_slot_resolved_multiplicity_resolved_M_R_matrix_unit_sectors"
            ),
            "global_ye3_descriptor_missing_stages": (
                "repeated_channel_global_block_map_application",
                "factorized_large_runtime_kernel",
                "full_sector_catalog_completeness_validation",
            ),
            "coefficient_axis_decomposition": (
                "slot_specht_multiplicity_copy",
                "target_specht_tableau_coordinate",
                "M_R",
            ),
            "mathematical_feature_axes": (
                "atom",
                "lambda_via_each_result_sector_slice",
                "slot_specht_multiplicity_copy",
                "target_specht_tableau_coordinate",
                "L_R_via_each_result_sector_slice",
                "M_R_explicit_output_axis",
            ),
            "feature_axis_status": {
                "lambda": "each_result.sector_slices.target_partition",
                "tableau_or_matrix_unit": "each_result.sector_slices.target_specht_tableau_coordinate",
                "L_R": "each_result.sector_slices.L_R",
                "slot_specht_multiplicity_copy": "each_result.sector_slices.multiplicity_count",
                "target_specht_tableau_coordinate": "each_result.sector_slices.slot_specht_dimension",
                "M_R": "each_result.sector_slices.M_R_values",
            },
            "irreducible_direct_sum_family_assembly": bool(
                results
                and all(
                    bool(result.metadata.get("irreducible_direct_sum_family_assembly", False)) for result in results
                )
            ),
            "global_rectangular_alpha_backend_required": False,
            "family_assembly_layout": (
                "batched_direct_sum_over_structure_then_sector_slices_with_explicit_target_partition_L_R_multiplicity_tableau_and_M_R_axes"
            ),
            "supported_M_R_values": tuple(
                sorted(
                    {
                        int(value)
                        for result in results
                        for value in tuple(result.metadata.get("supported_M_R_values", ()))
                    }
                )
            ),
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            "structure_count": int(len(results)),
            "require_all_sectors": bool(require_all_sectors),
            "total_matrix_unit_sector_count": int(total_sector_count),
            "evaluated_sector_count": int(evaluated_sector_count),
            "skipped_sector_count": int(skipped_sector_count),
            "all_sector_coverage_passed": all_sector_coverage_passed,
            "partial_sector_coverage_allowed": bool(
                skipped_sector_count > 0 and not bool(require_all_sectors)
            ),
            "sector_coverage_statuses": sector_coverage_statuses,
            "evaluated_sector_signatures": evaluated_sector_signatures,
            "skipped_status_counts": skipped_status_counts,
            "skipped_reason_summary": skipped_reason_summary,
            "evaluated_component_map_families": evaluated_component_map_families,
            "evaluated_component_map_kinds": evaluated_component_map_kinds,
            "evaluated_component_map_roles": evaluated_component_map_roles,
            "evaluated_component_map_sequence_all_validated": evaluated_component_map_sequence_all_validated,
        }
        if return_batch:
            return ASMatrixUnitDescriptorBatch(
                values_by_structure=tuple(results),
                row_slices=tuple(row_slices),
                axes=("atom", "flattened_slot_specht_multiplicity_tableau_M_R_feature"),
                unflattened_axes=(
                    "atom",
                    "slot_specht_multiplicity_copy",
                    "target_specht_tableau_coordinate",
                    "M_R",
                ),
                metadata=metadata,
            )
        if concatenate:
            if results:
                return torch.cat(tuple(result.values for result in results), dim=0)
            return torch.empty((0, 0), dtype=torch.float64)
        return results

    def create_matrix_unit_slot_resolved_global_coupler_descriptor_many(
        self,
        structures,
        *,
        concatenate=False,
        return_batch=False,
        require_all_sectors=False,
    ):
        """Alias for batching compatible A_s matrix-unit/global-coupler descriptors."""

        return self.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor_many(
            structures,
            concatenate=concatenate,
            return_batch=return_batch,
            require_all_sectors=require_all_sectors,
        )

    def create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
        require_all_sectors=False,
    ):
        """Read ASE structures and evaluate compatible slot-resolved global-coupler descriptors."""

        return self.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor_many(
            read_ase_structures(paths, index=index),
            concatenate=concatenate,
            return_batch=return_batch,
            require_all_sectors=require_all_sectors,
        )

    def create_matrix_unit_slot_resolved_global_coupler_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
        require_all_sectors=False,
    ):
        """Alias for file-based compatible A_s matrix-unit/global-coupler descriptors."""

        return self.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor_from_files(
            paths,
            index=index,
            concatenate=concatenate,
            return_batch=return_batch,
            require_all_sectors=require_all_sectors,
        )

    def create_A_s_matrix_unit_scalar_contraction_descriptor(self, atoms):
        """Evaluate A_s matrix-unit carriers as scalar projected-norm features.

        This implements the currently validated carrier-level trace/pairing
        contraction for scalar A_s slot-Specht sectors.  For each matrix-unit
        carrier block with axes ``[atom, tableau_row, tableau_col,
        slot_tuple_carrier]``, the diagonal matrix-unit trace recovers the
        projected carrier and the Euclidean dual pairing of that carrier with
        itself gives the scalar feature used by the existing
        ``ye3_slot_specht_power`` projected-norm readout.

        This is not the full global Young--E3 induction/coset/angular
        coefficient contraction.
        """

        from ye3t_ace.lifted_density import slot_specht_projected_trivial_pairing

        carriers, blocks, sectors = self.create_A_s_matrix_unit_carriers(atoms)
        if carriers:
            n_atoms = int(carriers[0].shape[0])
            dtype = carriers[0].dtype
            device = carriers[0].device
        else:
            n_atoms = int(len(atoms))
            dtype = torch.float64
            device = None

        feature_blocks = []
        sector_slices = []
        start = 0
        for sector_index, carrier in enumerate(carriers):
            if carrier.ndim != 4:
                raise ValueError("A_s matrix-unit scalar contraction expects carrier blocks with four axes.")
            if int(carrier.shape[0]) != n_atoms:
                raise ValueError("All A_s matrix-unit carrier blocks must have the same atom axis length.")
            if int(carrier.shape[1]) != int(carrier.shape[2]):
                raise ValueError("Matrix-unit tableau row/column axes must be square for diagonal trace contraction.")
            sector = dict(sectors[sector_index]) if sector_index < len(sectors) else {}
            target_L_R = int(sector.get("target_L_R", 0))
            if target_L_R != 0:
                raise NotImplementedError(
                    "A_s matrix-unit scalar contraction currently supports only scalar target_L_R=0 "
                    "because the implemented carrier hook does not expose a non-scalar M_R axis."
                )
            projected = carrier.diagonal(dim1=1, dim2=2).sum(dim=-1)
            feature = slot_specht_projected_trivial_pairing(projected, projected)
            stop = start + int(feature.shape[1])
            axis_metadata = _a_s_matrix_unit_sector_axis_metadata(
                sector,
                carrier,
                ye3t_axis=True,
            )
            sector_slices.append(
                {
                    "sector_index": int(sector_index),
                    "start": int(start),
                    "stop": int(stop),
                    "shape": tuple(int(dim) for dim in feature.shape),
                    "source_axis_metadata": axis_metadata,
                    "source_matrix_unit_shape": tuple(int(dim) for dim in carrier.shape),
                    "slot_specht_partition": sector.get("slot_specht_partition"),
                    "power": sector.get("power"),
                    "target_L_R": int(target_L_R),
                    "M_R_values": (0,),
                    "contraction": "matrix_unit_diagonal_trace_then_projected_trivial_pairing",
                    "full_descriptor_contraction_status": (
                        "carrier_level_scalar_pairing_not_global_induction_contracted"
                    ),
                    "global_coupler_contracted": False,
                    "runtime": sector.get("runtime", "slot_specht_matrix_unit_carrier"),
                }
            )
            feature_blocks.append(feature)
            start = stop

        if feature_blocks:
            values = torch.cat(tuple(feature_blocks), dim=1)
        else:
            values = torch.empty((n_atoms, 0), dtype=dtype, device=device)

        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_scalar_contraction_descriptor",
            "descriptor_first_flow": "YE3TRepresentation -> YE3TDescriptors.ye3t_basis -> YE3TDescriptorSet.create_A_s_matrix_unit_scalar_contraction_descriptor",
            "carrier_runtime": "slot_specht_matrix_unit_carrier",
            "full_descriptor_contraction_status": "carrier_level_scalar_pairing_not_global_induction_contracted",
            "carrier_level_contraction_status": "implemented_under_validation",
            "contraction": "matrix_unit_diagonal_trace_then_projected_trivial_pairing",
            "global_coupler_contracted": False,
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=True),
            "coefficient_axes_status": "scalar_projected_norm_features_from_matrix_unit_trace",
            "source_descriptor_runtime_status": self.metadata.get("runtime_status"),
            "source_descriptor_runtime_status_detail": self.metadata.get("runtime_status_detail"),
            "source_matrix_unit_carrier_status": self.metadata.get("matrix_unit_carrier_status"),
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            **_a_s_matrix_unit_runtime_scope_metadata(view="scalar matrix-unit carrier pairing descriptor"),
            "contraction_plan": {
                **_a_s_matrix_unit_contraction_plan(ye3t_axis=True),
                "status": "carrier_level_scalar_pairing_not_global_ye3t_contraction",
                "implemented_stages": tuple(
                    list(_a_s_matrix_unit_contraction_plan(ye3t_axis=True)["implemented_stages"])
                    + [
                        "matrix_unit_diagonal_trace_to_projected_carrier",
                        "projected_carrier_dual_pairing_to_scalar",
                    ]
                ),
                "descriptor_runtime_claim": "scalar_carrier_pairing_under_validation",
            },
        }
        return ASMatrixUnitDescriptorResult(
            values=values,
            sector_slices=tuple(sector_slices),
            sectors=tuple(sectors),
            blocks=tuple(blocks),
            axes=("atom", "matrix_unit_trace_pairing_feature"),
            unflattened_axes=("atom", "sector_scalar_pairing_feature"),
            metadata=metadata,
        )

    def create_matrix_unit_scalar_contraction_descriptor(self, atoms):
        """Alias for scalar A_s matrix-unit carrier contraction features."""

        return self.create_A_s_matrix_unit_scalar_contraction_descriptor(atoms)

    def create_A_s_matrix_unit_commutant_scalar_descriptor(self, atoms):
        """Evaluate A_s matrix-unit carriers with symmetric commutant forms.

        This is a carrier-level scalar contraction for the implemented A_s
        slot-Specht matrix-unit runtime.  Each matrix-unit carrier is first
        diagonally traced back to its central-projector carrier, then evaluated
        against the Frobenius-orthonormal symmetric commutant basis on that
        projected image.

        This is not the full global Young--E3 induction/coset/angular
        coefficient contraction.
        """

        from ye3t_ace.lifted_density import slot_specht_projected_commutant_quadratic_features

        carriers, blocks, sectors = self.create_A_s_matrix_unit_carriers(atoms)
        if carriers:
            n_atoms = int(carriers[0].shape[0])
            dtype = carriers[0].dtype
            device = carriers[0].device
        else:
            n_atoms = int(len(atoms))
            dtype = torch.float64
            device = None

        feature_blocks = []
        sector_slices = []
        start = 0
        for sector_index, carrier in enumerate(carriers):
            if carrier.ndim != 4:
                raise ValueError("A_s matrix-unit commutant contraction expects carrier blocks with four axes.")
            if int(carrier.shape[0]) != n_atoms:
                raise ValueError("All A_s matrix-unit carrier blocks must have the same atom axis length.")
            if int(carrier.shape[1]) != int(carrier.shape[2]):
                raise ValueError("Matrix-unit tableau row/column axes must be square for diagonal trace contraction.")
            sector = dict(sectors[sector_index]) if sector_index < len(sectors) else {}
            partition = tuple(int(part) for part in sector.get("slot_specht_partition", ()))
            if not partition:
                raise ValueError("A_s matrix-unit commutant contraction requires slot_specht_partition metadata.")
            slot_count = int(sum(partition))
            power = int(sector.get("power"))
            target_L_R = int(sector.get("target_L_R", 0))
            if target_L_R != 0:
                raise NotImplementedError(
                    "A_s matrix-unit commutant scalar contraction currently supports only scalar target_L_R=0 "
                    "because the implemented carrier hook does not expose a non-scalar M_R axis."
                )
            projected = carrier.diagonal(dim1=1, dim2=2).sum(dim=-1)
            feature = slot_specht_projected_commutant_quadratic_features(
                projected,
                slot_count=slot_count,
                power=power,
                partition=partition,
            )
            stop = start + int(feature.shape[1])
            axis_metadata = _a_s_matrix_unit_sector_axis_metadata(
                sector,
                carrier,
                ye3t_axis=True,
            )
            sector_slices.append(
                {
                    "sector_index": int(sector_index),
                    "start": int(start),
                    "stop": int(stop),
                    "shape": tuple(int(dim) for dim in feature.shape),
                    "source_axis_metadata": axis_metadata,
                    "source_matrix_unit_shape": tuple(int(dim) for dim in carrier.shape),
                    "slot_specht_partition": partition,
                    "slot_count": int(slot_count),
                    "power": int(power),
                    "target_L_R": int(target_L_R),
                    "M_R_values": (0,),
                    "commutant_feature_count": int(feature.shape[1]),
                    "contraction": "matrix_unit_diagonal_trace_then_projected_commutant_symmetric_quadratics",
                    "full_descriptor_contraction_status": (
                        "carrier_level_commutant_scalar_forms_not_global_induction_contracted"
                    ),
                    "global_coupler_contracted": False,
                    "runtime": sector.get("runtime", "slot_specht_matrix_unit_carrier"),
                }
            )
            feature_blocks.append(feature)
            start = stop

        if feature_blocks:
            values = torch.cat(tuple(feature_blocks), dim=1)
        else:
            values = torch.empty((n_atoms, 0), dtype=dtype, device=device)

        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_commutant_scalar_descriptor",
            "descriptor_first_flow": "YE3TRepresentation -> YE3TDescriptors.ye3t_basis -> YE3TDescriptorSet.create_A_s_matrix_unit_commutant_scalar_descriptor",
            "carrier_runtime": "slot_specht_matrix_unit_carrier",
            "full_descriptor_contraction_status": "carrier_level_commutant_scalar_forms_not_global_induction_contracted",
            "carrier_level_contraction_status": "implemented_under_validation",
            "contraction": "matrix_unit_diagonal_trace_then_projected_commutant_symmetric_quadratics",
            "global_coupler_contracted": False,
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=True),
            "coefficient_axes_status": "scalar_commutant_features_from_matrix_unit_trace",
            "source_descriptor_runtime_status": self.metadata.get("runtime_status"),
            "source_descriptor_runtime_status_detail": self.metadata.get("runtime_status_detail"),
            "source_matrix_unit_carrier_status": self.metadata.get("matrix_unit_carrier_status"),
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            **_a_s_matrix_unit_runtime_scope_metadata(view="commutant scalar matrix-unit carrier descriptor"),
            "contraction_plan": {
                **_a_s_matrix_unit_contraction_plan(ye3t_axis=True),
                "status": "carrier_level_commutant_scalar_forms_not_global_ye3t_contraction",
                "implemented_stages": tuple(
                    list(_a_s_matrix_unit_contraction_plan(ye3t_axis=True)["implemented_stages"])
                    + [
                        "matrix_unit_diagonal_trace_to_projected_carrier",
                        "projected_carrier_commutant_symmetric_quadratic_forms_to_scalar",
                    ]
                ),
                "descriptor_runtime_claim": "scalar_commutant_carrier_pairing_under_validation",
            },
        }
        return ASMatrixUnitDescriptorResult(
            values=values,
            sector_slices=tuple(sector_slices),
            sectors=tuple(sectors),
            blocks=tuple(blocks),
            axes=("atom", "matrix_unit_commutant_scalar_feature"),
            unflattened_axes=("atom", "sector_commutant_scalar_feature"),
            metadata=metadata,
        )

    def create_matrix_unit_commutant_scalar_descriptor(self, atoms):
        """Alias for commutant-symmetric A_s matrix-unit scalar features."""

        return self.create_A_s_matrix_unit_commutant_scalar_descriptor(atoms)

    def create_A_s_matrix_unit_scalar_contraction_descriptor_many(
        self,
        structures,
        *,
        concatenate=False,
        return_batch=False,
    ):
        """Evaluate scalar A_s matrix-unit contraction descriptors for structures."""

        results = []
        row_slices = []
        start = 0
        feature_count = None
        for atoms in structures:
            result = self.create_A_s_matrix_unit_scalar_contraction_descriptor(atoms)
            if feature_count is None:
                feature_count = int(result.num_features)
            elif int(result.num_features) != feature_count:
                raise ValueError(
                    "All A_s matrix-unit scalar contraction descriptor results must have the same feature count for batching."
                )
            results.append(result)
            stop = start + int(result.num_atoms)
            row_slices.append((int(start), int(stop)))
            start = stop
        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_scalar_contraction_descriptor_batch",
            "full_descriptor_contraction_status": "carrier_level_scalar_pairing_not_global_induction_contracted",
            "carrier_level_contraction_status": "implemented_under_validation",
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=True),
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            "structure_count": int(len(results)),
            **_a_s_matrix_unit_runtime_scope_metadata(view="batched scalar matrix-unit carrier pairing descriptor"),
        }
        if return_batch:
            return ASMatrixUnitDescriptorBatch(
                values_by_structure=tuple(results),
                row_slices=tuple(row_slices),
                axes=("atom", "matrix_unit_trace_pairing_feature"),
                unflattened_axes=("atom", "sector_scalar_pairing_feature"),
                metadata=metadata,
            )
        if concatenate:
            if results:
                return torch.cat(tuple(result.values for result in results), dim=0)
            return torch.empty((0, 0), dtype=torch.float64)
        return results

    def create_matrix_unit_scalar_contraction_descriptor_many(
        self,
        structures,
        *,
        concatenate=False,
        return_batch=False,
    ):
        """Alias for batching scalar A_s matrix-unit contraction descriptors."""

        return self.create_A_s_matrix_unit_scalar_contraction_descriptor_many(
            structures,
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_A_s_matrix_unit_commutant_scalar_descriptor_many(
        self,
        structures,
        *,
        concatenate=False,
        return_batch=False,
    ):
        """Evaluate commutant-symmetric A_s matrix-unit scalar descriptors."""

        results = []
        row_slices = []
        start = 0
        feature_count = None
        for atoms in structures:
            result = self.create_A_s_matrix_unit_commutant_scalar_descriptor(atoms)
            if feature_count is None:
                feature_count = int(result.num_features)
            elif int(result.num_features) != feature_count:
                raise ValueError(
                    "All A_s matrix-unit commutant scalar descriptor results must have the same feature count for batching."
                )
            results.append(result)
            stop = start + int(result.num_atoms)
            row_slices.append((int(start), int(stop)))
            start = stop
        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_commutant_scalar_descriptor_batch",
            "full_descriptor_contraction_status": "carrier_level_commutant_scalar_forms_not_global_induction_contracted",
            "carrier_level_contraction_status": "implemented_under_validation",
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=True),
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            "structure_count": int(len(results)),
            **_a_s_matrix_unit_runtime_scope_metadata(view="batched commutant scalar matrix-unit carrier descriptor"),
        }
        if return_batch:
            return ASMatrixUnitDescriptorBatch(
                values_by_structure=tuple(results),
                row_slices=tuple(row_slices),
                axes=("atom", "matrix_unit_commutant_scalar_feature"),
                unflattened_axes=("atom", "sector_commutant_scalar_feature"),
                metadata=metadata,
            )
        if concatenate:
            if results:
                return torch.cat(tuple(result.values for result in results), dim=0)
            return torch.empty((0, 0), dtype=torch.float64)
        return results

    def create_matrix_unit_commutant_scalar_descriptor_many(
        self,
        structures,
        *,
        concatenate=False,
        return_batch=False,
    ):
        """Alias for batching commutant A_s matrix-unit scalar descriptors."""

        return self.create_A_s_matrix_unit_commutant_scalar_descriptor_many(
            structures,
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_A_s_matrix_unit_scalar_contraction_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
    ):
        """Read ASE structures and evaluate scalar A_s matrix-unit contractions."""

        return self.create_A_s_matrix_unit_scalar_contraction_descriptor_many(
            read_ase_structures(paths, index=index),
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_matrix_unit_scalar_contraction_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
    ):
        """Alias for file-backed scalar A_s matrix-unit contraction descriptors."""

        return self.create_A_s_matrix_unit_scalar_contraction_descriptor_from_files(
            paths,
            index=index,
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_A_s_matrix_unit_commutant_scalar_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
    ):
        """Read ASE structures and evaluate commutant matrix-unit scalars."""

        return self.create_A_s_matrix_unit_commutant_scalar_descriptor_many(
            read_ase_structures(paths, index=index),
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_matrix_unit_commutant_scalar_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
    ):
        """Alias for file-backed commutant matrix-unit scalar descriptors."""

        return self.create_A_s_matrix_unit_commutant_scalar_descriptor_from_files(
            paths,
            index=index,
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_A_s_matrix_unit_ye3t_axis_descriptor_many(self, structures, *, concatenate=False, return_batch=False):
        """Evaluate scalar A_s matrix-unit YE3T-axis descriptor views for structures."""

        results = []
        row_slices = []
        start = 0
        feature_count = None
        for atoms in structures:
            result = self.create_A_s_matrix_unit_ye3t_axis_descriptor(atoms)
            if feature_count is None:
                feature_count = int(result.num_features)
            elif int(result.num_features) != feature_count:
                raise ValueError(
                    "All A_s matrix-unit YE3T-axis descriptor results must have the same feature count for batching."
                )
            results.append(result)
            stop = start + int(result.num_atoms)
            row_slices.append((int(start), int(stop)))
            start = stop

        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_ye3t_axis_descriptor_batch",
            "full_descriptor_contraction_status": "A_s_matrix_unit_axes_exposed_not_global_induction_contracted",
            "contraction_plan": _a_s_matrix_unit_contraction_plan(ye3t_axis=True),
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=True),
            "coefficient_axis_decomposition": (
                "lambda_via_sector_slice",
                "tableau_row",
                "tableau_col",
                "L_R",
                "M_R",
                "alpha_slot_tuple_carrier",
            ),
            "lambda_axis_source": "sector_slices.lambda",
            "multiplicity_axis_status": "slot_specht_sector_slices_only_not_global_multiplicity_resolved",
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            "structure_count": int(len(results)),
            **_a_s_matrix_unit_runtime_scope_metadata(view="batched YE3T-axis matrix-unit carrier descriptor"),
        }
        if return_batch:
            return ASMatrixUnitDescriptorBatch(
                values_by_structure=tuple(results),
                row_slices=tuple(row_slices),
                axes=("atom", "flattened_lambda_tableau_L_R_M_R_alpha_feature"),
                unflattened_axes=(
                    "atom",
                    "tableau_row",
                    "tableau_col",
                    "L_R",
                    "M_R",
                    "alpha_slot_tuple_carrier",
                ),
                metadata=metadata,
            )
        if concatenate:
            if results:
                return torch.cat(tuple(result.values for result in results), dim=0)
            return torch.empty((0, 0), dtype=torch.float64)
        return results

    def create_matrix_unit_ye3t_axis_descriptor_many(self, structures, *, concatenate=False, return_batch=False):
        """Alias for batching scalar A_s matrix-unit YE3T-axis descriptor views."""

        return self.create_A_s_matrix_unit_ye3t_axis_descriptor_many(
            structures,
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_A_s_matrix_unit_ye3t_axis_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
    ):
        """Read ASE structures and evaluate scalar A_s matrix-unit YE3T-axis descriptors."""

        return self.create_A_s_matrix_unit_ye3t_axis_descriptor_many(
            read_ase_structures(paths, index=index),
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_matrix_unit_ye3t_axis_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
    ):
        """Alias for file-based scalar A_s matrix-unit YE3T-axis descriptors."""

        return self.create_A_s_matrix_unit_ye3t_axis_descriptor_from_files(
            paths,
            index=index,
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_A_s_matrix_unit_descriptor_many(self, structures, *, concatenate=False, return_batch=False):
        """Evaluate A_s matrix-unit descriptor-shaped results for structures."""

        results = []
        row_slices = []
        start = 0
        feature_count = None
        for atoms in structures:
            result = self.create_A_s_matrix_unit_descriptor(atoms)
            if feature_count is None:
                feature_count = int(result.num_features)
            elif int(result.num_features) != feature_count:
                raise ValueError(
                    "All A_s matrix-unit descriptor results must have the same feature count for batching."
                )
            results.append(result)
            stop = start + int(result.values.shape[0])
            row_slices.append((int(start), int(stop)))
            start = stop

        metadata = {
            "runtime_status": "implemented_under_validation",
            "descriptor_family": "A_s_matrix_unit_carrier_descriptor_batch",
            "full_descriptor_contraction_status": "carrier_flattened_not_global_induction_contracted",
            "carrier_axis_scope": _a_s_matrix_unit_axis_scope(ye3t_axis=False),
            "coefficient_axis_decomposition": (
                "lambda_via_sector_slice",
                "tableau_row",
                "tableau_col",
                "slot_tuple_carrier",
            ),
            "lambda_axis_source": "sector_slices.slot_specht_partition",
            "multiplicity_axis_status": "slot_specht_sector_slices_only_not_global_multiplicity_resolved",
            "validation_report_hook": "validate_A_s_matrix_unit_carriers",
            "structure_count": int(len(results)),
            **_a_s_matrix_unit_runtime_scope_metadata(view="batched flat matrix-unit carrier descriptor"),
        }
        if return_batch:
            return ASMatrixUnitDescriptorBatch(
                values_by_structure=tuple(results),
                row_slices=tuple(row_slices),
                metadata=metadata,
            )
        if concatenate:
            if results:
                return torch.cat(tuple(result.values for result in results), dim=0)
            dtype = getattr(self.site_basis_config, "dtype", torch.float64)
            if not isinstance(dtype, torch.dtype):
                dtype = torch.float64
            return torch.empty((0, 0), dtype=dtype)
        return results

    def create_matrix_unit_descriptor_many(self, structures, *, concatenate=False, return_batch=False):
        """Alias for batching descriptor-shaped A_s matrix-unit carrier results."""

        return self.create_A_s_matrix_unit_descriptor_many(
            structures,
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_A_s_matrix_unit_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
    ):
        """Read ASE structures from files and evaluate A_s matrix-unit descriptors."""

        return self.create_A_s_matrix_unit_descriptor_many(
            read_ase_structures(paths, index=index),
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def create_matrix_unit_descriptor_from_files(
        self,
        paths,
        *,
        index=":",
        concatenate=False,
        return_batch=False,
    ):
        """Alias for file-based A_s matrix-unit descriptor-shaped evaluation."""

        return self.create_A_s_matrix_unit_descriptor_from_files(
            paths,
            index=index,
            concatenate=concatenate,
            return_batch=return_batch,
        )

    def linear_fit_kwargs(self):
        if self.settings is None:
            raise ValueError("Linear ACE fitting requires descriptor settings.")
        if self.representation.basis_family != "ace" and not self.representation.uses_trivial_fast_path:
            raise ValueError("Linear ACE fitting currently requires ACE descriptors or the trivial YE3T fast path.")
        return {
            "settings": self.settings,
            "site_basis_config": self.site_basis_config,
            "type_map": dict(self.type_map),
            "cutoff": float(self.cutoff),
            "descriptor_specs": tuple(self.descriptor_specs),
            "compact_labels": tuple(self.compact_labels) if self.compact_labels else None,
            "coupling_library": self.calculator.coupling_library,
            "basis_mode": self.representation.basis_mode,
            "descriptor_cache": self.descriptor_cache,
            "backend": self.backend,
            "strict_backend": bool(self.strict_backend),
            "validate_backend": bool(self.validate_backend),
            "device": self.metadata.get("device", None),
            "factorized_descriptor_runtime_policy": self.metadata.get("factorized_descriptor_runtime_policy", None),
        }


    def as_dict(self):
        return {
            "elements": list(self.elements),
            "type_map": dict(self.type_map),
            "cutoff": None if self.cutoff is None else float(self.cutoff),
            "representation": self.representation.as_dict(),
            "descriptor_count": int(len(self.descriptor_specs)),
            "compact_label_count": int(len(self.compact_labels)),
            "metadata": dict(self.metadata),
        }


@recordclass(('representation', 'ranks', 'records_by_rank', 'symmetric_power_blocks', 'metadata'), frozen = True)
class YE3TSymmetrySetInventory:
    """Descriptor-side inventory for high-rank ACE/YE3T symmetry subselections."""
    metadata = field(default_factory=dict)

    def as_dict(self):
        return {
            "representation": self.representation.as_dict(),
            "ranks": [int(rank) for rank in self.ranks],
            "records_by_rank": {
                int(rank): [dict(record) for record in records]
                for rank, records in self.records_by_rank.items()
            },
            "symmetric_power_blocks": [dict(block) for block in self.symmetric_power_blocks],
            "metadata": dict(self.metadata),
        }


@recordclass(('model_family', 'supported', 'status', 'reasons', 'warnings', 'required_validation'), frozen = True)
class YE3TModelCompatibility:
    """Structured compatibility report for a descriptor/model pairing.

    The nontrivial-sector validation names used here are implementation
    obligations, not proof claims: intertwiners should satisfy residual tests,
    isotypic projectors should satisfy rank/idempotency checks, and evaluated
    maps should be tested for the intended rotation/permutation action before a
    runtime path is promoted from inventory-only or planned status.
    """
    reasons = ()
    warnings = ()
    required_validation = ()

    def require_supported(self):
        if not self.supported:
            detail = "; ".join(str(reason) for reason in self.reasons) or "descriptor/model combination is not supported"
            raise ValueError(f"YE3TModel.{self.model_family} cannot use the supplied descriptor: {detail}")
        return self

    def as_dict(self):
        return {
            "model_family": self.model_family,
            "supported": bool(self.supported),
            "status": self.status,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "required_validation": list(self.required_validation),
        }











class YE3TDescriptors:
    """Descriptor-first construction namespace."""

    @staticmethod
    def from_spec(spec, config=None, **kwargs):
        """Construct a descriptor from a shared ``YE3TSpec`` plus runtime config.

        The shared spec selects the mathematical representation request.  The
        optional config supplies descriptor-runtime data such as elements,
        radial/site-basis settings, lifted-density settings, and backend flags.
        """

        spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
        cfg = _normalize_config(config, **kwargs)
        cfg["ye3t_spec"] = spec.to_dict()
        cfg.setdefault("representation", {"ye3t_spec": spec.to_dict()})
        if spec.carrier == "ACE_density":
            return YE3TDescriptors.ace(cfg)
        if spec.carrier == "A_s":
            return YE3TDescriptors.ye3t_basis(cfg)
        if spec.carrier == "Phi":
            representation_cfg = cfg.get("representation", None)
            if representation_cfg is None or (
                isinstance(representation_cfg, Mapping)
                and set(representation_cfg.keys()) <= {"ye3t_spec"}
            ):
                phi_payload = cfg.get("phi", {})
                phi_payload = dict(phi_payload) if isinstance(phi_payload, Mapping) else {}
                motif_family = str(phi_payload.get("motif_family", "full")).strip().lower()
                if motif_family == "complete":
                    cfg["representation"] = YE3TRepresentation.phi_complete(
                        motif_group=str(cfg.get("motif_group", "decorated_motif_slots")),
                        coupling_tree=str(cfg.get("coupling_tree", "explicit")),
                    ).as_dict()
                if motif_family in {"star", "star_only", "star-graphs", "star_graphs"}:
                    cfg["representation"] = YE3TRepresentation.phi_star(
                        motif_group=str(cfg.get("motif_group", "decorated_motif_slots")),
                        coupling_tree=str(cfg.get("coupling_tree", "explicit")),
                    ).as_dict()
                else:
                    cfg["representation"] = YE3TRepresentation.phi(
                        motif_group=str(cfg.get("motif_group", "decorated_motif_slots")),
                        coupling_tree=str(cfg.get("coupling_tree", "explicit")),
                        motif_family=phi_payload.get("motif_family", "full"),
                    ).as_dict()
            return YE3TDescriptors.phi(cfg)
        return YE3TDescriptors.ye3t(cfg)

    @staticmethod
    def from_spec_file(path, config=None, **kwargs):
        """Construct a descriptor from a shared YE3TSpec config file."""

        cfg = _normalize_config(config, **kwargs)
        cfg.setdefault("ye3t_spec_file", path)
        return YE3TDescriptors.from_spec(YE3TSpec.from_file(path), cfg)

    @staticmethod
    def plan_fixed_content_basis(basis, *, backend="reference"):
        """Record the exact YE3T-to-ASE binding requirements for a saved basis.

        The returned descriptor set is plan-only. Its representation labels
        come from ``basis``; source functions, physical slot placement, and
        coefficient lowering remain explicit requirements before evaluation.
        ``backend`` reserves the reference or native CPU contraction route
        without silently falling back between them.
        """

        from ye3t.api import YE3TFixedContentBasis

        if not isinstance(basis, YE3TFixedContentBasis):
            raise TypeError("plan_fixed_content_basis requires a YE3TFixedContentBasis.")
        backend = str(backend).strip().lower()
        if backend not in {"reference", "native_cpu"}:
            raise ValueError("backend must be 'reference' or 'native_cpu'.")
        basis.calculate_pd()
        cases = tuple(case for case, _quotient in basis.basis.cases)
        channels = tuple(sorted({
            (int(n), int(l))
            for case in cases for n, l in zip(case["n"], case["l"], strict=True)
        }))
        role_ranks = tuple(sorted({
            int(case["rank"]) for case in cases
            if (case["scope"] == "global" and case["partition"] != (case["rank"],))
            or (case["scope"] == "local" and any(
                parts != (sum(parts),) for parts in case["partition"]
            ))
        }))
        representation = YE3TRepresentation.ye3t(
            permutation_sector="full_irrep_decomposition",
            construction_mode="experimental",
            basis_mode="fixed_content_primitive_quotient",
            coefficient_backend="exact_fixed_content_basis",
            validation_status="exact_representation_pending_physical_binding",
            metadata={"runtime_status": "binding_required"},
        )
        return YE3TDescriptorSet(
            settings=None,
            site_basis_config=None,
            elements=(),
            type_map={},
            cutoff=None,
            representation=representation,
            compact_labels=(),
            descriptor_specs=(),
            backend=backend,
            strict_backend=True,
            validate_backend=True,
            descriptor_cache=None,
            ace_descriptor=None,
            settings_by_L=None,
            compiled_basis=basis,
            metadata={
                "descriptor_family": "fixed_content_basis_plan",
                "runtime_status": "binding_required",
                "compiler_source": "ye3t.api.YE3TFixedContentBasis",
                "representations": tuple(dict(row) for row in basis.representations),
                "primitive_representations": tuple(dict(row) for row in basis.primitive.representations),
                "decomposable_representations": tuple(dict(row) for row in basis.decomposable.representations),
                "required_source_channels": channels,
                "required_slot_roles_by_rank": {rank: rank for rank in role_ranks},
                "requires_role_resolved_carrier": bool(role_ranks),
                "required_binding_stages": (
                    "physical_radial_source_map",
                    "tensor_slot_role_or_placement_action",
                    "exact_young_rotation_coefficient_lowering",
                    "validated_reference_feature_contraction",
                ) + (("native_cpu_feature_contraction_and_parity",)
                     if backend == "native_cpu" else ()),
                "requested_backend": backend,
                "backend_fallback_allowed": False,
                "generic_descriptor_create": False,
            },
        )

    @staticmethod
    def ace(config=None, **kwargs):
        cfg = _normalize_config(config, **kwargs)
        if "elements" not in cfg and "elems" in cfg:
            cfg["elements"] = cfg["elems"]
        manual_labels = cfg.pop("manual_labels", None)
        pace_compatible = cfg.pop("pace_compatible", None)
        if pace_compatible is not None and not isinstance(pace_compatible, bool):
            raise TypeError("pace_compatible must be boolean.")
        if manual_labels is not None:
            conflicts = tuple(
                key
                for key in ("compact_labels", "ordinary_scalar_catalogue")
                if cfg.get(key) is not None
            )
            if conflicts:
                raise ValueError(
                    "manual_labels cannot be combined with "
                    + ", ".join(conflicts)
                    + "."
                )
            selected = []
            for raw in manual_labels:
                payload = (
                    raw["compact_label"]
                    if isinstance(raw, Mapping) and "compact_label" in raw
                    else raw
                )
                selected.append(normalize_compact_label(payload))
            if not selected:
                raise ValueError("manual_labels must contain at least one label.")
            if len(set(selected)) != len(selected):
                raise ValueError("manual_labels must not contain duplicates.")
            cfg["compact_labels"] = tuple(selected)
            if pace_compatible and cfg.get("scalar_coordinate_compiler") is None:
                cfg["scalar_coordinate_compiler"] = {
                    "mode": "all",
                    "options": {
                        "coordinate_contract": "pace_compatible_exact",
                        "coefficient_materialization": "exact",
                        "constructor_backend": "python",
                    },
                }
        ordinary_scalar_catalogue = None
        if cfg.get("ordinary_scalar_catalogue") is not None:
            if cfg.get("scalar_coordinate_compiler") is not None:
                raise ValueError(
                    "ordinary_scalar_catalogue already freezes its scalar-coordinate "
                    "compiler request and cannot be combined with "
                    "scalar_coordinate_compiler."
                )
            conflicts = tuple(
                key
                for key in ("compact_labels", "feature_filters")
                if cfg.get(key) is not None
            )
            if conflicts:
                raise ValueError(
                    "ordinary_scalar_catalogue cannot be combined with "
                    + ", ".join(conflicts)
                    + "."
                )
            (
                ordinary_scalar_catalogue,
                catalogue_labels,
                compiler_request,
            ) = _normalize_ordinary_scalar_catalogue(
                cfg["ordinary_scalar_catalogue"]
            )
            cfg["compact_labels"] = catalogue_labels
            cfg["_ordinary_scalar_catalogue"] = ordinary_scalar_catalogue
            cfg["_scalar_coordinate_compiler"] = compiler_request
        elif cfg.get("scalar_coordinate_compiler") is not None:
            direct_compiler_request = cfg["scalar_coordinate_compiler"]
            if (
                isinstance(direct_compiler_request, Mapping)
                and str(direct_compiler_request.get("mode", "")).strip().lower()
                == "serialized"
            ):
                raise ValueError(
                    "Serialized scalar coordinates must be supplied through a "
                    "validated ordinary_scalar_catalogue."
                )
            cfg["_scalar_coordinate_compiler"] = direct_compiler_request
        representation = YE3TRepresentation.from_config(
            cfg.get("representation", None),
            default_family="ace",
            default_basis_mode=cfg.get("basis_mode", None),
        )
        descriptor_basis_mode = _descriptor_basis_mode_from_representation(representation, cfg)
        if pace_compatible:
            if (
                representation.basis_family != "ace"
                or representation.permutation_sector != "trivial"
                or any(
                    int(label.L_R) != 0
                    or sum(int(value) for value in label.l_tuple) % 2
                    for label in cfg.get("compact_labels", ())
                )
            ):
                raise ValueError(
                    "pace_compatible requires ordinary even-parity scalar ACE labels."
                )
        if (
            ordinary_scalar_catalogue is not None
            and normalize_basis_mode(descriptor_basis_mode, L_R=0) is not None
        ):
            raise ValueError(
                "ordinary_scalar_catalogue cannot be combined with a reducing "
                "descriptor basis_mode."
            )
        cfg["basis_mode"] = descriptor_basis_mode
        factorized_policy = _factorized_policy_from_representation(representation, cfg)
        if factorized_policy is not None:
            cfg["factorized_descriptor_runtime_policy"] = factorized_policy
        if (
            representation.basis_mode in {"symmetric_power_subselection", "symmetric_power"}
            and cfg.get("feature_filters") is not None
            and "compact_labels" not in cfg
        ):
            cfg.setdefault("feature_filter_label_strategy", "direct")
        cfg["_suppress_legacy_warning"] = True
        ace_descriptor = ACEDescriptor.from_config(cfg)
        catalogue_metadata = None
        if ordinary_scalar_catalogue is not None:
            compiled_coordinates = _validate_ordinary_scalar_catalogue_runtime(
                ordinary_scalar_catalogue,
                ace_descriptor,
            )
            feature_id_by_label = {
                CompactLabel.from_dict(row["compact_label"]): row["feature_id"]
                for row in ordinary_scalar_catalogue["rows"]
            }
            variant_index_by_label = {}
            descriptor_rows = []
            for descriptor_index, spec in enumerate(ace_descriptor.descriptor_specs):
                label = normalize_compact_label(spec.label)
                if label not in feature_id_by_label:
                    raise RuntimeError(
                        "Descriptor specification is not bound to the approved catalogue."
                    )
                variant_index = int(variant_index_by_label.get(label, 0))
                variant_index_by_label[label] = variant_index + 1
                descriptor_rows.append(
                    {
                        "descriptor_index": int(descriptor_index),
                        "feature_id": feature_id_by_label[label],
                        "descriptor_key": str(spec.key),
                        "variant_index": variant_index,
                    }
                )
            catalogue_metadata = {
                **ordinary_scalar_catalogue,
                "compiled_coordinates": list(compiled_coordinates),
                "descriptor_rows": descriptor_rows,
            }
        runtime_site_basis = ace_descriptor.calculator.site_basis_config
        return YE3TDescriptorSet(
            settings=ace_descriptor.settings,
            site_basis_config=runtime_site_basis,
            elements=tuple(ace_descriptor.elements),
            type_map=dict(ace_descriptor.type_map),
            cutoff=float(ace_descriptor.cutoff),
            representation=representation,
            compact_labels=tuple(ace_descriptor.calculator.labels),
            descriptor_specs=tuple(ace_descriptor.descriptor_specs),
            backend=ace_descriptor.calculator.backend,
            strict_backend=bool(ace_descriptor.calculator.strict_backend),
            validate_backend=bool(ace_descriptor.calculator.validate_backend),
            descriptor_cache=ace_descriptor.descriptor_cache,
            ace_descriptor=ace_descriptor,
            metadata={
                "descriptor_family": "ace",
                "runtime_status": "implemented_under_validation",
                "manual_labels_selected": manual_labels is not None,
                "pace_compatible_requested": bool(pace_compatible),
                "runtime_status_detail": "implemented_trivial_fast_path",
                "feature_filters": ace_descriptor.feature_filters,
                "factorized_descriptor_runtime_policy": factorized_policy,
                "descriptor_basis_mode": descriptor_basis_mode,
                "ordinary_scalar_catalogue": catalogue_metadata,
                "device": None if ace_descriptor.device is None else str(ace_descriptor.device),
                **_descriptor_shared_spec_metadata(
                    representation,
                    cfg,
                    runtime_status="implemented_under_validation",
                    carrier="ACE_density",
                ),
            },
        )

    @staticmethod
    def ace_from_structures(structures, config=None, **kwargs):
        """Build ACE descriptors for ASE structures, inferring elements if omitted.

        This is the descriptor-first replacement for legacy structure-driven
        descriptor setup. It returns a reusable ``YE3TDescriptorSet``; call
        ``create_many`` on the result for batched/high-throughput evaluation.
        """

        atoms_list = list(structures) if not hasattr(structures, "get_chemical_symbols") else [structures]
        cfg = _normalize_config(config, **kwargs)
        if "elements" not in cfg and "elems" in cfg:
            cfg["elements"] = cfg["elems"]
        if "elements" not in cfg:
            cfg["elements"] = list(infer_elements_from_ase_atoms(atoms_list))
        if "type_map" not in cfg:
            cfg["type_map"] = build_explicit_type_map(tuple(str(elem) for elem in cfg["elements"]))
        return YE3TDescriptors.ace(cfg)

    @staticmethod
    def ace_symmetry_set(config=None, **kwargs):
        """Build a high-rank ACE symmetry-subselection inventory.

        This is a descriptor-side inventory for repeated-block symmetry choices
        and symmetric-power kernel planning. It does not evaluate site
        descriptors on structures.
        """

        cfg = _normalize_config(config, **kwargs)
        representation = YE3TRepresentation.from_config(
            cfg.get("representation", None),
            default_family="ace",
            default_basis_mode="symmetric_power_subselection",
        )
        if representation.basis_family != "ace" or representation.basis_mode not in {
            "symmetric_power_subselection",
            "symmetric_power",
        }:
            raise ValueError(
                "YE3TDescriptors.ace_symmetry_set requires "
                "YE3TRepresentation.ace_symmetric_power(...)."
            )

        from collections import Counter
        from ye3t.api import (
            allowed_symmetric_power_outputs,
            enumerate_rank_labels,
            symmetric_power_output_multiplicity,
        )

        ranks = tuple(int(value) for value in cfg.get("ranks", ()))
        target_l_avs = tuple(float(value) for value in cfg.get("target_l_avs", (1.0,)))
        strict_by_rank = cfg.get("strict_lmax_per_rank", cfg.get("strict_max_li_per_rank", {}))
        spec_by_rank = cfg.get("orbit_spec_per_rank", {})
        max_labels = cfg.get("max_labels_per_rank", None)
        records_by_rank = {}
        symmetric_power_blocks = []

        for rank in ranks:
            strict_lmax = strict_by_rank.get(rank, strict_by_rank.get(str(rank), cfg.get("strict_lmax", None)))
            spec = spec_by_rank.get(rank, spec_by_rank.get(str(rank), None))
            if spec is None:
                raise ValueError(f"Missing orbit_spec_per_rank entry for rank {rank}.")
            labels = enumerate_rank_labels(
                rank,
                target_l_avs,
                strict_max_li=None if strict_lmax is None else int(strict_lmax),
                homogeneous_n=bool(cfg.get("homogeneous_n", False)),
                spec=spec,
                max_labels=max_labels,
            )
            rank_records = []
            for record_index, record in enumerate(labels):
                row = {
                    "rank": int(record["rank"]),
                    "n_in": tuple(int(value) for value in record["n_in"]),
                    "l_in": tuple(int(value) for value in record["l_in"]),
                    "n_orbits": tuple(int(value) for value in record["n_orbits"]),
                    "l_orbits": tuple(int(value) for value in record["l_orbits"]),
                    "pair_orbits": tuple(int(value) for value in record["pair_orbits"]),
                    "target_l_avs": tuple(float(value) for value in record["target_l_avs"]),
                    "realized_average_l": float(record["realized_average_l"]),
                }
                rank_records.append(row)
                repeated = Counter(zip(row["n_in"], row["l_in"]))
                for (eta, input_L), power in sorted(repeated.items()):
                    if int(power) < int(cfg.get("min_symmetric_power", 2)):
                        continue
                    outputs = tuple(int(value) for value in allowed_symmetric_power_outputs(power, input_L))
                    symmetric_power_blocks.append(
                        {
                            "rank": int(rank),
                            "record_index": int(record_index),
                            "eta": int(eta),
                            "input_L": int(input_L),
                            "power": int(power),
                            "outputs": outputs,
                            "multiplicities": tuple(
                                int(symmetric_power_output_multiplicity(power, input_L, output_L))
                                for output_L in outputs
                            ),
                        }
                    )
            records_by_rank[int(rank)] = tuple(rank_records)

        return YE3TSymmetrySetInventory(
            representation=representation,
            ranks=ranks,
            records_by_rank=records_by_rank,
            symmetric_power_blocks=tuple(symmetric_power_blocks),
            metadata={
                "descriptor_family": "ace",
                "runtime_status": "inventory_only",
                "inventory_kind": "ace_symmetric_power_subselection",
                "target_l_avs": target_l_avs,
                "max_labels_per_rank": max_labels,
                "runtime_limitation": "site descriptor evaluation is not wired for this symmetry-set inventory",
            },
        )


    @staticmethod
    def ye3t(config=None, **kwargs):
        cfg = _normalize_config(config, **kwargs)
        if "elements" not in cfg and "elems" in cfg:
            cfg["elements"] = cfg["elems"]
        representation = YE3TRepresentation.from_config(
            cfg.get("representation", None),
            default_family="ye3t",
            default_basis_mode=cfg.get("basis_mode", None),
        )
        if representation.construction_mode == "tagged_cauchy_exact":
            basis_cfg = dict(cfg)
            basis_cfg["representation"] = representation
            descriptor = YE3TDescriptors.ye3t_basis(basis_cfg)
            descriptor.metadata.update(
                {
                    "descriptor_family_requested": "ye3t",
                    "descriptor_family_routing": (
                        "YE3TDescriptors.ye3t tagged-Cauchy selector routed to "
                        "YE3TDescriptors.ye3t_basis"
                    ),
                    "descriptor_first_flow": (
                        "YE3TRepresentation.tagged_cauchy_image -> "
                        "YE3TDescriptors.ye3t -> YE3TDescriptors.ye3t_basis"
                    ),
                }
            )
            return descriptor
        if representation.basis_mode == "filtered_A_s" or representation.metadata.get("density") == "A_s":
            basis_cfg = dict(cfg)
            basis_cfg["representation"] = representation
            descriptor = YE3TDescriptors.ye3t_basis(basis_cfg)
            descriptor.metadata.update(
                {
                    "descriptor_family_requested": "ye3t",
                    "descriptor_family_routing": (
                        "YE3TDescriptors.ye3t filtered_A_s selector routed to YE3TDescriptors.ye3t_basis"
                    ),
                    "descriptor_first_flow": (
                        "YE3TRepresentation.filtered_A_s -> YE3TDescriptors.ye3t -> "
                        "YE3TDescriptors.ye3t_basis"
                    ),
                }
            )
            return descriptor
        if representation.uses_trivial_fast_path:
            ace_cfg = dict(cfg)
            ace_cfg["representation"] = {
                "basis_family": "ace",
                "basis_mode": representation.basis_mode,
                "fast_path_policy": representation.fast_path_policy,
            }
            descriptor = YE3TDescriptors.ace(ace_cfg)
            descriptor.representation = representation
            descriptor.metadata["descriptor_family"] = "ye3t"
            descriptor.metadata["runtime_status"] = "implemented_under_validation"
            descriptor.metadata["runtime_status_detail"] = "implemented_via_trivial_ace_fast_path"
            descriptor.metadata.update(
                _descriptor_shared_spec_metadata(
                    representation,
                    cfg,
                    runtime_status="implemented_under_validation",
                )
            )
            return descriptor
        if representation.construction_mode == "young_subgroup_exact":
            ace_cfg = dict(cfg)
            ace_cfg["representation"] = {
                "basis_family": "ace",
                "basis_mode": representation.basis_mode,
                "fast_path_policy": representation.fast_path_policy,
            }
            ace_cfg["_suppress_legacy_warning"] = True
            descriptor = YE3TDescriptors.ace(ace_cfg)
            descriptor.representation = representation
            descriptor.metadata.update(
                {
                    "descriptor_family": "ye3t",
                    "runtime_status": "implemented_under_validation",
                    "runtime_status_detail": "implemented_young_subgroup_exact",
                    "representation_level": "young_subgroup_invariant",
                    "basis_backend": "exact_block_first_young_subgroup",
                    "compact_labels_by_L": {int(descriptor.settings.L_R): tuple(descriptor.compact_labels)},
                    "maturity": (
                        "implemented and evaluated for Young-subgroup invariant ACE labels; "
                        "not a full nontrivial Specht-carrier runtime"
                    ),
                }
            )
            descriptor.metadata.update(
                _descriptor_shared_spec_metadata(
                    representation,
                    cfg,
                    runtime_status="implemented_under_validation",
                )
            )
            return descriptor

        raise NotImplementedError("This representation has no stable linear descriptor evaluator.")

    @staticmethod
    def ye3t_basis(config=None, **kwargs):
        cfg = _normalize_config(config, **kwargs)
        unsupported = sorted(str(key) for key, value in cfg.items()
                             if str(key).startswith("tensor_") and value is not None)
        if unsupported:
            raise ValueError("Unsupported descriptor settings: " + ", ".join(unsupported))
        basis_config = cfg.get("basis")
        if isinstance(basis_config, Mapping) and basis_config.get("type") == "tagged_cauchy_carriers":
            from ye3t.couplings import compile as compile_coupling
            from ye3t.couplings import tagged_cauchy_carriers_request, tagged_cauchy_carrier_schedule

            allowed_basis = {"type", "species", "cutoff_A", "pair_cutoffs_A",
                             "catalogue", "source_realization"}
            unknown_basis = set(basis_config) - allowed_basis
            if unknown_basis:
                raise ValueError("Unsupported tagged carrier basis settings: " +
                                 ", ".join(sorted(unknown_basis)))
            representation_config = cfg.get("representation", {})
            runtime_config = cfg.get("runtime", {})
            if not isinstance(representation_config, Mapping) or not isinstance(runtime_config, Mapping):
                raise TypeError("tagged carrier representation and runtime must be mappings")
            if representation_config.get("mode", "tagged_cauchy_carriers") != "tagged_cauchy_carriers":
                raise ValueError("tagged carrier representation mode must be tagged_cauchy_carriers")
            if set(representation_config) - {"mode", "sector_policy"}:
                raise ValueError("Unsupported tagged carrier representation settings")
            if basis_config.get("source_realization", "tagged_cauchy_occurrence") != "tagged_cauchy_occurrence":
                raise ValueError("tagged carriers require the tagged_cauchy_occurrence source")
            elements = tuple(str(value) for value in basis_config.get("species", ()))
            if not elements or len(set(elements)) != len(elements):
                raise ValueError("tagged carriers require unique species")
            cutoff = float(basis_config.get("cutoff_A", 0.0))
            if not np.isfinite(cutoff) or cutoff <= 0:
                raise ValueError("tagged carriers require positive cutoff_A")
            pair_cutoffs = basis_config.get("pair_cutoffs_A")
            if pair_cutoffs is not None:
                expected_pairs = {left + "-" + right for left in elements for right in elements}
                if set(pair_cutoffs) != expected_pairs:
                    raise ValueError("pair_cutoffs_A must cover every ordered species pair")
                if any(not np.isfinite(float(value)) or float(value) <= 0 or
                       float(value) > cutoff for value in pair_cutoffs.values()):
                    raise ValueError("pair_cutoffs_A values must be positive and <= cutoff_A")
            catalogue = basis_config.get("catalogue")
            if not isinstance(catalogue, Mapping):
                raise ValueError("tagged carriers require a basis.catalogue mapping")
            allowed_catalogue = {
                "ranks", "nmax_per_rank", "lmax_per_rank", "nmax", "lmax",
                "source_block_partitions_by_rank", "max_source_blocks", "tag_counts",
                "kappa_policy", "tag_sectors", "max_features_per_rank",
                "max_records_per_rank", "input_Lmax", "source_family_id",
            }
            unknown_catalogue = set(catalogue) - allowed_catalogue
            if unknown_catalogue:
                raise ValueError("Unsupported tagged carrier catalogue settings: " +
                                 ", ".join(sorted(unknown_catalogue)))
            if catalogue.get("source_family_id", "orthogonal_shifted_jacobi_origin_regular_v1") != (
                    "orthogonal_shifted_jacobi_origin_regular_v1"):
                raise ValueError("Standalone tagged carriers require the certified shifted-Jacobi source")
            allowed_runtime = {"backend", "device", "dtype", "support_chunk_size",
                               "compiled_cache_dir"}
            unknown_runtime = set(runtime_config) - allowed_runtime
            if unknown_runtime:
                raise ValueError("Unsupported tagged carrier runtime settings: " +
                                 ", ".join(sorted(unknown_runtime)))
            sector_policy = representation_config.get("sector_policy", "tagged_mixed")
            if sector_policy != "tagged_mixed":
                raise ValueError("Standalone tagged carriers currently require sector_policy='tagged_mixed'")
            if runtime_config.get("backend", "reference") != "reference":
                raise ValueError("standalone tagged carrier ASE evaluation uses backend='reference'")
            if "compiled_cache_dir" in runtime_config:
                cache_dir = runtime_config["compiled_cache_dir"]
            else:
                cache_dir = default_linear_cache_directory() / "compiler" / "tagged_cauchy_carriers"
            if cache_dir is not None:
                cache_dir = Path(cache_dir)
            request = tagged_cauchy_carriers_request(catalogue=catalogue, species=elements)
            compiled = compile_coupling(request, cache_dir=cache_dir)
            grouped_sources = {}
            for source in compiled["sources"]:
                tag_count = int(source["request"]["tag_count"])
                grouped_sources.setdefault(tag_count, []).append(source)
            schedules = tuple(
                tagged_cauchy_carrier_schedule(grouped_sources[tag_count])
                for tag_count in sorted(grouped_sources)
            )
            representation = YE3TRepresentation.tagged_cauchy_carriers()
            carrier_config = {
                "cutoff_A": cutoff,
                "pair_cutoffs_A": basis_config.get("pair_cutoffs_A"),
                "device": runtime_config.get("device", "cpu"),
                "dtype": runtime_config.get("dtype", "float64"),
                "support_chunk_size": runtime_config.get("support_chunk_size", 4096),
                "compiled_cache_dir": cache_dir,
            }
            return YE3TDescriptorSet(
                settings=None, site_basis_config=None, elements=elements,
                type_map={name: index for index, name in enumerate(elements)},
                cutoff=cutoff, representation=representation, compact_labels=(),
                descriptor_specs=(), backend="reference", strict_backend=True,
                validate_backend=True, descriptor_cache=None, ace_descriptor=None,
                settings_by_L=None, metadata={
                    "descriptor_family": "tagged_cauchy_carriers",
                    "runtime_status": "implemented_under_validation",
                    "materialization_status": "compiled",
                    "tagged_cauchy_carriers_config": carrier_config,
                    "tagged_cauchy_carriers_compiled": compiled,
                    "tagged_cauchy_carrier_source_plan": {"schedules": schedules},
                    "feature_count": sum(int(schedule["output_dimension"])
                                         for schedule in schedules),
                    "multiplet_count": sum(len(schedule["inventory"])
                                           for schedule in schedules),
                    "component_count": sum(int(schedule["output_dimension"])
                                           for schedule in schedules),
                    "runtime_capabilities": {"generic_descriptor_create": True},
                },
            )
        if isinstance(basis_config, Mapping) and basis_config.get("type") == "tagged_cauchy_image":
            representation_config = cfg.get("representation", {})
            if not isinstance(representation_config, Mapping) or (
                representation_config.get("carrier", "A_s") != "A_s" or
                representation_config.get("target", {"permutation": "trivial", "L": 0})
                    != {"permutation": "trivial", "L": 0} or
                representation_config.get("mode", "tagged_cauchy_image") != "tagged_cauchy_image"
            ):
                raise ValueError("Tagged physical config requires A_s, trivial permutation, L=0, and mode=tagged_cauchy_image.")
            runtime_config = cfg.get("runtime", {})
            if not isinstance(runtime_config, Mapping):
                raise TypeError("runtime must be a mapping.")
            tagged_config = {key: value for key, value in basis_config.items()
                             if key not in {"type", "species"}}
            if "tag_counts" in tagged_config:
                if ("selected_raw_tag_counts" in tagged_config and
                        tuple(tagged_config["tag_counts"]) != tuple(tagged_config["selected_raw_tag_counts"])):
                    raise ValueError("basis.tag_counts and selected_raw_tag_counts must agree")
                tagged_config["selected_raw_tag_counts"] = tagged_config.pop("tag_counts")
            if "catalogue" not in tagged_config and "tensor_order" not in tagged_config:
                raise ValueError("Tagged physical config requires basis.catalogue or basis.tensor_order.")
            if "compiled_cache_dir" in runtime_config:
                tagged_config["compiled_cache_dir"] = runtime_config["compiled_cache_dir"]
            else:
                tagged_config.setdefault("compiled_cache_dir",
                    default_linear_cache_directory() / "compiler" / "tagged_cauchy_image")
            cfg = {
                "elements": basis_config["species"],
                "representation": YE3TRepresentation.tagged_cauchy_image(),
                "tagged_cauchy_image": tagged_config,
                "backend": runtime_config.get("backend", "auto"),
            }
        tagged_cauchy_payload = cfg.get("tagged_cauchy_image", None)
        lifted_cauchy_payload = cfg.get("lifted_cauchy", None)
        if tagged_cauchy_payload is not None:
            if lifted_cauchy_payload is not None:
                raise ValueError(
                    "tagged_cauchy_image and lifted_cauchy are distinct descriptor families."
                )
            if not isinstance(tagged_cauchy_payload, Mapping):
                raise TypeError("tagged_cauchy_image must be a mapping.")
            tagged_cauchy_payload = dict(tagged_cauchy_payload)
            if "compiled_cache_dir" not in tagged_cauchy_payload:
                tagged_cauchy_payload["compiled_cache_dir"] = (
                    default_linear_cache_directory() / "compiler" / "tagged_cauchy_image"
                )
            elements = tuple(str(elem) for elem in cfg.get("elements", ()))
            if not elements:
                raise ValueError(
                    "tagged_cauchy_image requires a visible nonempty elements list."
                )
            if len(set(elements)) != len(elements):
                raise ValueError("tagged_cauchy_image elements must be unique.")
            canonical_elements = tuple(sorted(elements))
            expected_type_map = {
                element: index
                for index, element in enumerate(canonical_elements)
            }
            supplied_type_map = cfg.get("type_map", expected_type_map)
            supplied_type_map = {
                str(key): int(value)
                for key, value in dict(supplied_type_map).items()
            }
            if supplied_type_map != expected_type_map:
                raise ValueError(
                    "tagged_cauchy_image currently uses the compiler-canonical "
                    "alphabetical type_map "
                    f"{expected_type_map!r}."
                )
            cutoff = float(
                tagged_cauchy_payload.get(
                    "cutoff_A", tagged_cauchy_payload.get("cutoff", 0.0)
                )
            )
            if not np.isfinite(cutoff) or cutoff <= 0.0:
                raise ValueError(
                    "tagged_cauchy_image requires a positive cutoff_A."
                )
            if "catalogue" in tagged_cauchy_payload:
                from ye3t.couplings import compile as compile_coupling
                from ye3t.couplings import count as count_coupling
                from ye3t.couplings import plan as plan_coupling
                from ye3t.couplings import tagged_cauchy_image_request
                from ye3t_ace.tagged_cauchy_image import (
                    TaggedCauchyImageEvaluator, realify_tagged_cauchy_image,
                    tagged_cauchy_source_plan,
                )
                representation = cfg.get("representation")
                representation = (YE3TRepresentation.tagged_cauchy_image()
                    if representation is None else YE3TRepresentation.from_config(
                        representation, default_family="ye3t", default_basis_mode="tagged_cauchy_image"))
                if representation.construction_mode != "tagged_cauchy_exact":
                    raise ValueError("General tagged descriptors require the tagged-Cauchy representation.")
                request = tagged_cauchy_image_request(
                    catalogue=tagged_cauchy_payload["catalogue"], species=canonical_elements)
                compiler_validation = tagged_cauchy_payload.get("compiler_validation", "full")
                if compiler_validation not in {"full", "certificate"}:
                    raise ValueError("compiler_validation must be full or certificate.")
                cache_dir = tagged_cauchy_payload.get("compiled_cache_dir")
                cache_path = None if cache_dir is None else Path(cache_dir)/(
                    request["request_hash"]+".json")
                if cache_path is not None and cache_path.exists():
                    from ye3t.couplings import CompiledTaggedCauchyImage
                    compiled = CompiledTaggedCauchyImage.from_dict(json.loads(cache_path.read_text()),
                        compiler_validation=compiler_validation)
                    if compiled.plan.report.request["request_hash"] != request["request_hash"]:
                        raise ValueError("General tagged compiler cache request mismatch.")
                    preflight = compiled.plan.report
                else:
                    preflight = count_coupling(request)
                    compiled = compile_coupling(plan_coupling(preflight))
                    # A new cache entry always receives the full independent
                    # replay; certificate mode applies only to subsequent loads.
                    from ye3t.couplings import CompiledTaggedCauchyImage
                    compiled = CompiledTaggedCauchyImage.from_dict(compiled.to_dict())
                    if cache_path is not None:
                        import tempfile
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                dir=cache_path.parent, prefix=cache_path.name+".",
                                suffix=".tmp", delete=False) as handle:
                            temporary_path = Path(handle.name)
                            try:
                                json.dump(compiled.to_dict(), handle, sort_keys=True)
                            except BaseException:
                                handle.close()
                                temporary_path.unlink(missing_ok=True)
                                raise
                        temporary_path.replace(cache_path)
                program = realify_tagged_cauchy_image(compiled, compiler_validation=compiler_validation)
                source = tagged_cauchy_source_plan(compiled, cutoff, program,
                    pair_cutoffs=tagged_cauchy_payload.get("pair_cutoffs_A"),
                    compiler_validation=compiler_validation)
                evaluator = TaggedCauchyImageEvaluator(compiled, source, program,
                    backend=str(cfg.get("backend", "auto")), compiler_validation=compiler_validation)
                coordinates = tuple(compiled.payload["image_coordinate_provenance"])
                return YE3TDescriptorSet(
                    settings=None, site_basis_config=None, elements=canonical_elements,
                    type_map=expected_type_map, cutoff=cutoff, representation=representation,
                    compact_labels=(), descriptor_specs=(),
                    backend=str(cfg.get("backend", "pytorch")),
                    strict_backend=bool(cfg.get("strict_backend", False)),
                    validate_backend=bool(cfg.get("validate_backend", True)),
                    descriptor_cache=cfg.get("descriptor_cache"), ace_descriptor=None,
                    settings_by_L=None, metadata={
                        "descriptor_family": "linear_tagged_cauchy_image",
                        "implementation_branch": "exact_tagged_physical_pivots_v4",
                        "runtime_status": "implemented_under_validation",
                        "central_species_order": canonical_elements,
                        "tagged_cauchy_image_config": tagged_cauchy_payload,
                        "tagged_cauchy_image_request": request,
                        "tagged_cauchy_image_preflight": preflight,
                        "tagged_cauchy_image_compiled": compiled,
                        "tagged_cauchy_image_real_program": program,
                        "tagged_cauchy_image_source_plan": source,
                        "tagged_cauchy_image_evaluator": evaluator,
                        "feature_count": evaluator.feature_count,
                        "feature_keys": tuple(record["coordinate_id"] for record in coordinates),
                        "feature_coordinate_provenance": coordinates,
                        "raw_opportunity_labels": tuple(preflight.labels),
                        "materialization_status": "compiled",
                        "runtime_capabilities": {"fit_and_model_evaluation": True,
                            "generic_descriptor_create": True, "native_bundle_export": True},
                        **_descriptor_shared_spec_metadata(representation, cfg,
                            runtime_status="implemented_under_validation", carrier="A_s"),
                    })
            tensor_order = int(tagged_cauchy_payload.get("tensor_order", 4))
            angular_degree = int(
                tagged_cauchy_payload.get("angular_degree", 1)
            )
            radial_degrees = tuple(
                int(value)
                for value in tagged_cauchy_payload.get(
                    "radial_degrees", tagged_cauchy_payload.get("q_values", (0, 1))
                )
            )
            selected_raw_tag_counts = tuple(
                tagged_cauchy_payload.get("selected_raw_tag_counts", (0, 1, 2))
            )
            if tensor_order != 4 or angular_degree != 1:
                raise ValueError(
                    "The certified tagged_cauchy_image rung is bounded to "
                    "tensor_order=4 and angular_degree=1."
                )
            if not radial_degrees or any(value < 0 for value in radial_degrees):
                raise ValueError(
                    "tagged_cauchy_image radial_degrees must be nonempty and nonnegative."
                )
            if len(set(radial_degrees)) != len(radial_degrees):
                raise ValueError(
                    "tagged_cauchy_image radial_degrees must be unique."
                )
            source_family = str(
                tagged_cauchy_payload.get(
                    "source_family",
                    "orthogonal_shifted_jacobi_origin_regular_v1",
                )
            )
            if source_family != "orthogonal_shifted_jacobi_origin_regular_v1":
                raise ValueError(
                    "The certified tagged_cauchy_image rung requires the "
                    "orthogonal shifted-Jacobi source family."
                )
            representation_payload = cfg.get("representation", None)
            representation = (
                YE3TRepresentation.tagged_cauchy_image()
                if representation_payload is None
                else YE3TRepresentation.from_config(
                    representation_payload,
                    default_family="ye3t",
                    default_basis_mode="tagged_cauchy_image",
                )
            )
            if representation.construction_mode != "tagged_cauchy_exact":
                raise ValueError(
                    "tagged_cauchy_image descriptors require "
                    "YE3TRepresentation.tagged_cauchy_image()."
                )
            from ye3t.couplings import compile as compile_coupling
            from ye3t.couplings import count as count_coupling
            from ye3t.couplings import plan as plan_coupling
            from ye3t.couplings import racah_harmonic_product_plan
            from ye3t.couplings import tagged_cauchy_image_request
            from ye3t.couplings.orthogonal_shifted_jacobi import (
                build_radial_species_product_record,
            )

            angular_plan = racah_harmonic_product_plan(
                (angular_degree,), maximum_collision_arity=2
            )
            source_keys = tuple(
                {
                    "neighbor_species": element,
                    "q": int(radial_degree),
                    "l": angular_degree,
                    "source_family_id": source_family,
                }
                for element in canonical_elements
                for radial_degree in sorted(radial_degrees)
            )
            product_record = build_radial_species_product_record(
                source_keys,
                angular_plan,
                maximum_collision_arity=2,
                support_id=str(
                    tagged_cauchy_payload.get(
                        "support_id", "common_normalized_cutoff_support_v1"
                    )
                ),
            )
            request = tagged_cauchy_image_request(
                product_record,
                angular_plan,
                source_keys=source_keys,
                tensor_order=tensor_order,
                selected_raw_tag_counts=selected_raw_tag_counts,
            )
            preflight = count_coupling(request)
            materialization = str(
                tagged_cauchy_payload.get(
                    "coefficient_materialization", "compile"
                )
            ).strip().lower()
            if materialization not in {"compile", "defer"}:
                raise ValueError(
                    "coefficient_materialization must be 'compile' or 'defer'."
                )
            compiled = None
            real_program = None
            source_plan = None
            evaluator = None
            if materialization == "compile":
                from ye3t_ace.tagged_cauchy_image import (
                    TaggedCauchyImageEvaluator,
                    realify_tagged_cauchy_image,
                    tagged_cauchy_source_plan,
                )
                from ye3t.couplings import CompiledTaggedCauchyImage

                cache_dir = tagged_cauchy_payload.get("compiled_cache_dir")
                cache_path = (None if cache_dir is None else
                              Path(cache_dir) / (request["request_hash"] + ".json"))
                if cache_path is not None and cache_path.exists():
                    compiled = CompiledTaggedCauchyImage.from_dict(
                        json.loads(cache_path.read_text(encoding="utf-8")))
                    if compiled.plan.report.request["request_hash"] != request["request_hash"]:
                        raise ValueError("Tagged compiler cache request mismatch.")
                else:
                    compiled = compile_coupling(plan_coupling(preflight))
                    if cache_path is not None:
                        import tempfile
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                dir=cache_path.parent, prefix=cache_path.name + ".",
                                suffix=".tmp", delete=False) as handle:
                            temporary_path = Path(handle.name)
                            try:
                                json.dump(compiled.to_dict(), handle, sort_keys=True)
                            except BaseException:
                                handle.close()
                                temporary_path.unlink(missing_ok=True)
                                raise
                        temporary_path.replace(cache_path)
                real_program = realify_tagged_cauchy_image(compiled)
                source_plan = tagged_cauchy_source_plan(
                    compiled, cutoff, real_program
                )
                evaluator = TaggedCauchyImageEvaluator(
                    compiled, source_plan, real_program
                )
            planned_feature_slots = tuple(
                {
                    "feature_index": int(index),
                    "preflight_coordinate_id": (
                        "tagged_image_preflight_v1:"
                        f"{preflight.convention_hash}:{index}"
                    ),
                }
                for index in range(int(preflight.exact_image_dimension))
            )
            feature_keys = (
                tuple(
                    "tagged_image_v1:"
                    f"{compiled.payload['catalogue_hash']}:{index}"
                    for index in range(int(preflight.exact_image_dimension))
                )
                if compiled is not None
                else tuple()
            )
            normalized_config = {
                "tensor_order": tensor_order,
                "selected_raw_tag_counts": tuple(
                    sorted(int(value) for value in selected_raw_tag_counts)
                ),
                "source_family": source_family,
                "radial_degrees": tuple(sorted(radial_degrees)),
                "angular_degree": angular_degree,
                "cutoff_A": cutoff,
                "support_id": str(product_record["normalized_support"]["support_id"]),
                "coefficient_materialization": materialization,
                "compiled_cache_dir": (None if tagged_cauchy_payload.get("compiled_cache_dir") is None
                                       else str(tagged_cauchy_payload["compiled_cache_dir"])),
            }
            return YE3TDescriptorSet(
                settings=None,
                site_basis_config=None,
                elements=canonical_elements,
                type_map=expected_type_map,
                cutoff=cutoff,
                representation=representation,
                compact_labels=tuple(),
                descriptor_specs=tuple(),
                backend=str(cfg.get("backend", "pytorch")),
                strict_backend=bool(cfg.get("strict_backend", False)),
                validate_backend=bool(cfg.get("validate_backend", True)),
                descriptor_cache=cfg.get("descriptor_cache", None),
                ace_descriptor=None,
                settings_by_L=None,
                metadata={
                    "descriptor_family": "linear_tagged_cauchy_image",
                    "implementation_branch": "exact_tagged_physical_image_v3",
                    "runtime_status": (
                        "implemented_under_validation"
                        if evaluator is not None
                        else "preflight_only"
                    ),
                    "central_species_order": canonical_elements,
                    "source_family": source_family,
                    "tagged_cauchy_image_config": normalized_config,
                    "tagged_cauchy_image_request": request,
                    "tagged_cauchy_image_preflight": preflight,
                    "tagged_cauchy_image_compiled": compiled,
                    "tagged_cauchy_image_real_program": real_program,
                    "tagged_cauchy_image_source_plan": source_plan,
                    "tagged_cauchy_image_evaluator": evaluator,
                    "raw_opportunity_labels": tuple(preflight.labels),
                    "planned_feature_slots": planned_feature_slots,
                    "feature_keys": feature_keys,
                    "feature_coordinate_provenance": (
                        tuple(compiled.payload["image_coordinate_provenance"])
                        if compiled is not None
                        else tuple()
                    ),
                    "feature_count": int(preflight.exact_image_dimension),
                    "materialization_status": (
                        "compiled" if evaluator is not None else "preflight_only"
                    ),
                    "descriptor_first_flow": (
                        "YE3TRepresentation.tagged_cauchy_image -> "
                        "YE3TDescriptors.ye3t_basis -> YE3TModel.linear"
                    ),
                    "runtime_capabilities": {
                        "fit_and_model_evaluation": evaluator is not None,
                        "generic_descriptor_create": evaluator is not None,
                        "native_bundle_export": evaluator is not None,
                    },
                    **_descriptor_shared_spec_metadata(
                        representation,
                        cfg,
                        runtime_status=(
                            "implemented_under_validation"
                            if evaluator is not None
                            else "inventory_only"
                        ),
                        carrier="A_s",
                    ),
                },
            )
        if lifted_cauchy_payload is not None:
            if not isinstance(lifted_cauchy_payload, Mapping):
                raise TypeError("lifted_cauchy must be a mapping.")
            elements = tuple(str(elem) for elem in cfg.get("elements", ()))
            type_map = cfg.get(
                "type_map",
                build_explicit_type_map(elements) if elements else {},
            )
            type_map = {
                str(key): int(value) for key, value in dict(type_map).items()
            }
            if not elements:
                elements = tuple(
                    key for key, _value in sorted(
                        type_map.items(), key=lambda item: item[1]
                    )
                )
            if set(elements) != set(type_map):
                raise ValueError(
                    "lifted_cauchy elements and type_map species must agree exactly."
                )
            if len(set(type_map.values())) != len(type_map):
                raise ValueError("lifted_cauchy type_map values must be unique.")
            representation_payload = cfg.get("representation", None)
            representation = (
                YE3TRepresentation.lifted_cauchy_scalar()
                if representation_payload is None
                else YE3TRepresentation.from_config(
                    representation_payload,
                    default_family="ye3t",
                    default_basis_mode="lifted_cauchy_scalar",
                )
            )
            if representation.construction_mode != "lifted_cauchy_exact":
                raise ValueError(
                    "lifted_cauchy descriptors require "
                    "YE3TRepresentation.lifted_cauchy_scalar()."
                )
            from ye3t_ace.lifted_cauchy_linear import (
                prepare_lifted_cauchy_descriptor_payload,
            )

            prepared = prepare_lifted_cauchy_descriptor_payload(
                lifted_cauchy_payload,
                type_map,
            )
            source = dict(prepared["lifted_cauchy_source"])
            shared_cfg = dict(cfg)
            fixed_spec_values = {
                "target_L_R": 0,
                "target_parity": "even",
                "target_permutation": "trivial",
                "coefficient_backend": "linear_lifted_cauchy_scalar",
                "tree_schedule": "explicit",
            }
            for key, expected in fixed_spec_values.items():
                if key in shared_cfg and shared_cfg[key] != expected:
                    raise ValueError(
                        "lifted_cauchy descriptor semantics require "
                        f"{key}={expected!r}."
                    )
                shared_cfg[key] = expected
            return YE3TDescriptorSet(
                settings=None,
                site_basis_config=None,
                elements=elements,
                type_map=type_map,
                cutoff=float(source["cutoff_A"]),
                representation=representation,
                compact_labels=tuple(),
                descriptor_specs=tuple(),
                backend=str(cfg.get("backend", "pytorch")),
                strict_backend=bool(cfg.get("strict_backend", False)),
                validate_backend=bool(cfg.get("validate_backend", True)),
                descriptor_cache=None,
                ace_descriptor=None,
                settings_by_L=None,
                metadata={
                    "descriptor_family": "linear_lifted_cauchy_scalar",
                    "implementation_branch": "A_s_lifted_cauchy",
                    "runtime_status": "implemented_under_validation",
                    "runtime_status_detail": (
                        "compiler_owned_canonical_and_factored_torch_reference"
                    ),
                    "density_normalization": "none",
                    "role_axis_retained": True,
                    "central_species_order": elements,
                    "source_family": source["schema"],
                    "descriptor_first_flow": (
                        "YE3TRepresentation.lifted_cauchy_scalar -> "
                        "YE3TDescriptors.ye3t_basis -> YE3TModel.linear"
                    ),
                    **prepared,
                    **_descriptor_shared_spec_metadata(
                        representation,
                        shared_cfg,
                        runtime_status="implemented_under_validation",
                        carrier="A_s",
                    ),
                },
            )
        if "lifted_density" in cfg and isinstance(cfg["lifted_density"], Mapping):
            lifted_payload = dict(cfg["lifted_density"])
            unsupported = sorted(str(key) for key, value in lifted_payload.items()
                                 if str(key).startswith("tensor_") and value is not None)
            if unsupported:
                raise ValueError("Unsupported lifted-density settings: " + ", ".join(unsupported))
        else:
            lifted_payload = {
                key: cfg[key]
                for key in (
                    "cutoff",
                    "channels",
                    "filter_kind",
                    "num_filters",
                    "filter_centers",
                    "filter_width",
                    "radial_lambda",
                    "readout_mode",
                    "slot_group",
                    "young_subgroup_blocks",
                    "density_normalization",
                    "A_s_density_normalization",
                    "density_normalization_nugget",
                    "hidden_layers",
                    "ye3_max_power",
                    "ye3_optimization_policy",
                    "ye3_slot_sectors",
                    "ye3_slot_specht_partitions",
                    "ye3_include_rank1",
                    "ye3_rank_nmax",
                    "ye3_rank_lmax",
                    "ye3_rank_lmin",
                    "ye3_slot_specht_coupling",
                    "periodic_image_mode",
                    "enforce_unique_periodic_images",
                    "periodic_image_margin",
                )
                if key in cfg
            }
        representation_payload = cfg.get("representation", None)
        representation_subselection = cfg.get("representation_subselection", cfg.get("slot_subselection", None))
        if representation_subselection is None and isinstance(representation_payload, Mapping):
            representation_subselection = representation_payload.get(
                "representation_subselection",
                representation_payload.get("slot_subselection", None),
            )
        if representation_subselection is not None:
            lifted_payload["readout_mode"] = _a_s_readout_for_subselection(
                representation_subselection,
                lifted_payload.get("readout_mode", None),
            )
        if "ye3_slot_sectors" not in lifted_payload:
            if isinstance(representation_payload, YE3TRepresentation):
                sectors = representation_payload.metadata.get("slot_sectors", None)
                if sectors is not None:
                    lifted_payload["ye3_slot_sectors"] = tuple(str(sector) for sector in sectors)
            elif isinstance(representation_payload, Mapping):
                sectors = representation_payload.get("slot_sectors", None)
                if sectors is not None:
                    lifted_payload["ye3_slot_sectors"] = tuple(str(sector) for sector in sectors)
        if "ye3_slot_specht_partitions" not in lifted_payload:
            if isinstance(representation_payload, YE3TRepresentation):
                partitions = representation_payload.metadata.get("slot_specht_partitions", None)
                if partitions is not None:
                    lifted_payload["ye3_slot_specht_partitions"] = partitions
            elif isinstance(representation_payload, Mapping):
                partitions = representation_payload.get("slot_specht_partitions", None)
                if partitions is not None:
                    lifted_payload["ye3_slot_specht_partitions"] = partitions
        from ye3t_ace.lifted_density import (
            BRANCH_LIFTED_DENSITY,
            HybridACELiftedDensityConfig,
            LiftedDensityConfig,
            lifted_A_s_descriptor_inventory,
        )

        lifted_config = LiftedDensityConfig.from_dict(lifted_payload)
        hybrid_config = HybridACELiftedDensityConfig(
            branches=tuple(cfg.get("branches", (BRANCH_LIFTED_DENSITY,))),
            lifted_density=lifted_config,
            dtype=str(cfg.get("dtype", "float64")),
        )
        elements = tuple(str(elem) for elem in cfg.get("elements", ()))
        type_map = cfg.get("type_map", build_explicit_type_map(elements) if elements else {})
        representation = representation_payload
        if representation is None:
            representation = YE3TRepresentation.filtered_A_s(
                slot_group=lifted_config.slot_group,
                representation_subselection=representation_subselection,
                coupling_tree=str(cfg.get("coupling_tree", "balanced")),
            )
        elif isinstance(representation, Mapping) and (
            "representation_subselection" in representation or "slot_subselection" in representation
        ):
            representation = YE3TRepresentation.filtered_A_s(
                slot_group=lifted_config.slot_group,
                representation_subselection=representation.get(
                    "representation_subselection",
                    representation.get("slot_subselection", representation_subselection),
                ),
                coupling_tree=str(representation.get("coupling_tree", cfg.get("coupling_tree", "balanced"))),
                fast_path_policy=str(representation.get("fast_path_policy", "auto")),
                slot_specht_partitions=representation.get("slot_specht_partitions", None),
                metadata=dict(representation.get("metadata", {})),
            )
        else:
            representation = YE3TRepresentation.from_config(
                representation,
                default_family="ye3t",
                default_basis_mode="filtered_A_s",
            )
        basis_backend = representation.metadata.get("basis_backend", "orbital_basis")
        representation_level = representation.metadata.get(
            "representation_level",
            "permutation_module_equivariant_maps",
        )
        runtime_status_detail = "implemented_A_s_filtered_density"
        if lifted_config.readout_mode == "ye3_slot_specht_power":
            basis_backend = "slot_specht_central_projector_power"
            if lifted_config.ye3_slot_specht_coupling == "commutant_symmetric":
                representation_level = "slot_specht_commutant_symmetric_scalar_forms"
                runtime_status_detail = "implemented_A_s_slot_specht_commutant_symmetric_scalar_forms"
            else:
                representation_level = "slot_specht_central_projector_norms"
                runtime_status_detail = "implemented_A_s_slot_specht_scalar_projector_norms"
        role_coordinate_policy = _validate_A_s_role_coordinate_policy_for_partitions(
            _a_s_role_coordinate_policy(cfg, representation),
            getattr(lifted_config, "ye3_slot_specht_partitions", ()),
        )
        requested_target_L_R = int(cfg.get("target_L_R", cfg.get("L_R", 0)))
        requested_M_R_values = cfg.get("target_M_R_values", cfg.get("M_R_values", None))
        matrix_unit_capability = _a_s_matrix_unit_runtime_capability_report(
            requested_target_L_R=requested_target_L_R,
            requested_M_R_values=requested_M_R_values,
        )
        return YE3TDescriptorSet(
            settings=None,
            site_basis_config=None,
            elements=elements,
            type_map={str(k): int(v) for k, v in dict(type_map).items()},
            cutoff=float(lifted_config.cutoff),
            representation=representation,
            compact_labels=tuple(),
            descriptor_specs=tuple(),
            backend=str(cfg.get("backend", "pytorch")),
            strict_backend=bool(cfg.get("strict_backend", False)),
            validate_backend=bool(cfg.get("validate_backend", True)),
            descriptor_cache=None,
            ace_descriptor=None,
            settings_by_L=None,
            metadata={
                "descriptor_family": "ye3t_basis",
                "legacy_descriptor_family": "filtered_A_s",
                "implementation_branch": "A_s",
                "runtime_status": "legacy_scalar_readout",
                "runtime_status_detail": runtime_status_detail,
                "density_normalization": lifted_config.density_normalization,
                "density_normalization_nugget": (
                    lifted_config.density_normalization_nugget
                ),
                "representation_subselection": representation.metadata.get("representation_subselection", "equivariant"),
                "representation_level": representation_level,
                "basis_backend": basis_backend,
                "representation_hierarchy": list(
                    representation.metadata.get(
                        "representation_hierarchy",
                        (
                            "permutation_module_equivariant_maps",
                            "young_specht_subselection",
                            "trivial_or_antisymmetric_fast_path_when_applicable",
                        ),
                    )
                ),
                "implemented_slot_runtime": True,
                "matrix_unit_carrier_status": (
                    "implemented_under_validation"
                    if lifted_config.readout_mode == "ye3_slot_specht_power"
                    else "not_requested"
                ),
                "matrix_unit_carrier_backend": (
                    "slot_specht_matrix_units_small_exact"
                    if lifted_config.readout_mode == "ye3_slot_specht_power"
                    else None
                ),
                "matrix_unit_carrier_geometry_runtime": bool(
                    lifted_config.readout_mode == "ye3_slot_specht_power"
                ),
                "A_s_matrix_unit_runtime_capability": matrix_unit_capability,
                "A_s_supported_runtime_routes": matrix_unit_capability["supported_runtime_routes"],
                "A_s_planned_runtime_routes": matrix_unit_capability["planned_runtime_routes"],
                "A_s_unsupported_runtime_routes": matrix_unit_capability["unsupported_runtime_routes"],
                "A_s_role_coordinate_policy": role_coordinate_policy,
                "full_matrix_unit_descriptor_runtime": False,
                "A_s_descriptor_inventory": [
                    record.as_dict()
                    for record in lifted_A_s_descriptor_inventory(lifted_config)
                ],
                "lifted_density_config": hybrid_config.to_dict(),
                "ye3t_brief_alignment": {
                    "coupling_tree_recorded": True,
                    "coupling_path_definition": "hom_space_intertwiner_labels",
                    "implementation_schedule": str(representation.coupling_tree),
                    "full_hom_space_runtime": False,
                    "full_nontrivial_descriptor_status": "pending certified couplers and validation harness",
                },
                **_descriptor_shared_spec_metadata(
                    representation,
                    cfg,
                    runtime_status="legacy_scalar_readout",
                    carrier="A_s",
                ),
            },
        )

    @staticmethod
    def filtered_A_s(config=None, **kwargs):
        warnings.warn(
            "YE3TDescriptors.filtered_A_s(...) is a compatibility alias. "
            "Use YE3TDescriptors.ye3t_basis(...) for descriptor-first Young-E3 basis workflows; "
            "the current implementation uses the A_s filtered-density branch underneath.",
            FutureWarning,
            stacklevel=2,
        )
        return YE3TDescriptors.ye3t_basis(config, **kwargs)

    @staticmethod
    def phi(config=None, **kwargs):
        cfg = _normalize_config(config, **kwargs)
        if "phi" in cfg and isinstance(cfg["phi"], Mapping):
            phi_payload = dict(cfg["phi"])
        else:
            phi_payload = {
                key: cfg[key]
                for key in (
                    "cutoff",
                    "edge_cutoff",
                    "channels",
                    "motif_specs",
                    "motif_family",
                    "hidden_channels",
                    "hidden_layers",
                    "include_rank4",
                    "edge_basis_backend",
                    "periodic_image_mode",
                    "enforce_unique_periodic_images",
                    "periodic_image_margin",
                    "normalize_motif_features",
                )
                if key in cfg
            }
        from ye3t_ace.cluster_phi import HybridACEPhiConfig, PhiBranchConfig

        phi_config = PhiBranchConfig.from_dict(phi_payload)
        hybrid_config = HybridACEPhiConfig(
            branches=tuple(cfg.get("branches", ("bar_phi",))),
            phi=phi_config,
            dtype=str(cfg.get("dtype", "float64")),
        )
        elements = tuple(str(elem) for elem in cfg.get("elements", ()))
        type_map = cfg.get("type_map", build_explicit_type_map(elements) if elements else {})
        representation = cfg.get("representation", None)
        if representation is None:
            motif_family = str(phi_config.motif_family).strip().lower()
            if motif_family == "complete":
                representation = YE3TRepresentation.phi_complete(
                    motif_group=str(cfg.get("motif_group", "decorated_motif_slots")),
                    coupling_tree=str(cfg.get("coupling_tree", "explicit")),
                )
            elif motif_family in {"star", "star_only", "star-graphs", "star_graphs"}:
                representation = YE3TRepresentation.phi_star(
                    motif_group=str(cfg.get("motif_group", "decorated_motif_slots")),
                    coupling_tree=str(cfg.get("coupling_tree", "explicit")),
                )
            else:
                representation = YE3TRepresentation.phi(
                    motif_group=str(cfg.get("motif_group", "decorated_motif_slots")),
                    coupling_tree=str(cfg.get("coupling_tree", "explicit")),
                    motif_family=phi_config.motif_family,
                )
        else:
            representation = YE3TRepresentation.from_config(
                representation,
                default_family="ye3t",
                default_basis_mode="phi_motif",
            )
        complete_inventory = None
        if str(phi_config.motif_family).strip().lower() == "complete":
            complete_inventory = _phi_complete_inventory_from_phi_config(representation, phi_config, cfg)
        return YE3TDescriptorSet(
            settings=None,
            site_basis_config=None,
            elements=elements,
            type_map={str(k): int(v) for k, v in dict(type_map).items()},
            cutoff=float(phi_config.cutoff),
            representation=representation,
            compact_labels=tuple(),
            descriptor_specs=tuple(),
            backend=str(cfg.get("backend", "pytorch")),
            strict_backend=bool(cfg.get("strict_backend", False)),
            validate_backend=bool(cfg.get("validate_backend", True)),
            descriptor_cache=None,
            ace_descriptor=None,
            settings_by_L=None,
            metadata={
                "descriptor_family": "phi",
                "runtime_status": "implemented_under_validation",
                "runtime_status_detail": (
                    "implemented_phi_complete_reference_path"
                    if complete_inventory is not None
                    else "implemented_phi_reference_path"
                ),
                "maturity": (
                    "fixed_feature_explicit_motif_linear_reference_path"
                ),
                "phi_config": hybrid_config.to_dict(),
                "phi_complete_inventory": complete_inventory,
                "ye3t_brief_alignment": {
                    "motif_automorphism_group_recorded": True,
                    "full_hom_space_runtime": False,
                    "full_nontrivial_descriptor_status": "pending certified couplers and validation harness",
                    "complete_irrep_inventory_recorded": complete_inventory is not None,
                },
                **_descriptor_shared_spec_metadata(
                    representation,
                    cfg,
                    runtime_status="implemented_under_validation",
                    carrier="Phi",
                ),
            },
        )

    @staticmethod
    def phi_star(config=None, **kwargs):
        cfg = _normalize_config(config, **kwargs)
        phi_cfg = cfg.get("phi", {}) if isinstance(cfg.get("phi", {}), Mapping) else {}
        merged_phi = dict(phi_cfg)
        merged_phi.setdefault("motif_family", "star")
        cfg["phi"] = merged_phi
        return YE3TDescriptors.phi(cfg)

    @staticmethod
    def phi_complete(config=None, **kwargs):
        cfg = _normalize_config(config, **kwargs)
        phi_cfg = cfg.get("phi", {}) if isinstance(cfg.get("phi", {}), Mapping) else {}
        merged_phi = dict(phi_cfg)
        merged_phi.setdefault("motif_family", "complete")
        cfg["phi"] = merged_phi
        return YE3TDescriptors.phi(cfg)


def _is_ye3t_basis_descriptor(descriptor):
    if not isinstance(descriptor, YE3TDescriptorSet):
        return False
    family = descriptor.metadata.get("descriptor_family")
    legacy = descriptor.metadata.get("legacy_descriptor_family")
    return family in {
        "ye3t_basis",
        "filtered_A_s",
        "linear_lifted_cauchy_scalar",
        "linear_tagged_cauchy_image",
    } or legacy == "filtered_A_s"


def _normalize_linear_fit_reference_energies(reference_energies=None):
    if reference_energies is None:
        return {}
    normalized = {
        str(key): float(value) for key, value in dict(reference_energies).items()
    }
    if any(not np.isfinite(value) for value in normalized.values()):
        raise ValueError("Reference energies must be finite.")
    return normalized


def _reference_energy_offset_for_atoms(atoms, reference_energies):
    refs = _normalize_linear_fit_reference_energies(reference_energies)
    if not refs:
        return 0.0
    missing = sorted({str(symbol) for symbol in atoms.get_chemical_symbols()} - set(refs))
    if missing:
        raise KeyError(f"Missing reference energies for elements: {missing}")
    return float(sum(refs[str(symbol)] for symbol in atoms.get_chemical_symbols()))


def _linear_fit_energy_target_from_atoms(atoms, energy_key):
    key = str(energy_key)
    if key in getattr(atoms, "info", {}):
        return float(atoms.info[key])
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", {}) if calc is not None else {}
    if key in results:
        return float(results[key])
    if "energy" in results:
        return float(results["energy"])
    if calc is not None:
        return float(atoms.get_potential_energy())
    return None


def _linear_fit_force_target_from_atoms(atoms, force_key):
    if force_key is None:
        return None
    key = str(force_key)
    if key in getattr(atoms, "arrays", {}):
        return np.asarray(atoms.arrays[key], dtype=float)
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", {}) if calc is not None else {}
    if key in results:
        return np.asarray(results[key], dtype=float)
    if "forces" in results:
        return np.asarray(results["forces"], dtype=float)
    if calc is not None:
        try:
            return np.asarray(atoms.get_forces(), dtype=float)
        except Exception:
            return None
    return None


def _structures_with_reference_energy_targets(structures, *, energy_key, force_key=None, reference_energies):
    """Return copies whose scalar energy targets are residual energies.

    The force arrays are copied unchanged because isolated-atom reference
    energies are constants with respect to structure positions.
    """

    refs = _normalize_linear_fit_reference_energies(reference_energies)
    copied_structures = []
    offsets = []
    for index, atoms in enumerate(structures):
        raw_energy = _linear_fit_energy_target_from_atoms(atoms, energy_key)
        if raw_energy is None:
            copied_structures.append(atoms.copy())
            offsets.append(0.0)
            continue
        copied = atoms.copy()
        force_target = _linear_fit_force_target_from_atoms(atoms, force_key)
        if force_target is not None:
            copied.arrays[str(force_key)] = np.asarray(force_target, dtype=float)
        offset = _reference_energy_offset_for_atoms(copied, refs)
        copied.info[str(energy_key)] = float(raw_energy) - offset
        copied_structures.append(copied)
        offsets.append(offset)
    return copied_structures, {
        "enabled": bool(refs),
        "reference_energies": refs,
        "energy_key": str(energy_key),
        "target_convention": "E_target = E_raw - sum_type(n_type * E_ref[type])",
        "force_target_convention": "unchanged; isolated-atom offsets are position-independent",
        "structure_count": int(len(copied_structures)),
        "offset_min_eV": float(min(offsets)) if offsets else 0.0,
        "offset_max_eV": float(max(offsets)) if offsets else 0.0,
    }


def _fit_A_s_matrix_unit_global_coupler_energy_linear_model(
    descriptor,
    structures,
    *,
    energy_key="energy",
    energy_weight=1.0,
    ridge_alpha=0.0,
    include_bias_column=True,
    require_all_sectors=False,
    dtype=torch.float64,
    device=None,
    structure_weights=None,
    structure_weight_key=None,
    structure_group_key=None,
    structure_group_weights=None,
    structure_group_default_weight=None,
    structure_group_normalize_mean=True,
    boltzmann_temperature_K=None,
    boltzmann_energy_key=None,
    boltzmann_weight_nugget=0.0,
    boltzmann_weight_prefactor=1.0,
    boltzmann_normalize_mean=True,
    min_structure_weight=0.0,
):
    """Fit scalar energies from evaluated A_s matrix-unit/global-coupler rows."""

    structures = list(structures)
    if not structures:
        raise ValueError("Need at least one structure to fit an A_s matrix-unit/global-coupler linear model.")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    dtype = dtype if isinstance(dtype, torch.dtype) else torch.float64
    device = torch.device("cpu" if device is None else device)
    resolved_structure_weights, structure_weight_metadata = structure_fit_weights(
        structures,
        structure_weights=structure_weights,
        structure_weight_key=structure_weight_key,
        structure_group_key=structure_group_key,
        structure_group_weights=structure_group_weights,
        structure_group_default_weight=structure_group_default_weight,
        structure_group_normalize_mean=structure_group_normalize_mean,
        boltzmann_temperature_K=boltzmann_temperature_K,
        boltzmann_energy_key=boltzmann_energy_key,
        boltzmann_weight_nugget=boltzmann_weight_nugget,
        boltzmann_weight_prefactor=boltzmann_weight_prefactor,
        boltzmann_normalize_mean=boltzmann_normalize_mean,
        min_weight=min_structure_weight,
    )
    rows = []
    targets = []
    atom_counts = []
    descriptor_shapes = []
    result_metadata = None
    feature_count = None
    for structure_index, atoms in enumerate(structures):
        energy_target = _linear_fit_energy_target_from_atoms(atoms, energy_key)
        if energy_target is None:
            raise KeyError(f"Structure {structure_index} is missing energy target {str(energy_key)!r}.")
        result = descriptor.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(
            atoms,
            require_all_sectors=bool(require_all_sectors),
        )
        if feature_count is None:
            feature_count = int(result.num_features)
            result_metadata = dict(result.metadata)
        elif int(result.num_features) != int(feature_count):
            raise ValueError(
                "All evaluated A_s matrix-unit/global-coupler descriptors must have the same feature count."
            )
        row = result.values.to(dtype=dtype, device=device).sum(dim=0)
        descriptor_shapes.append(tuple(int(dim) for dim in result.values.shape))
        if include_bias_column:
            row = torch.cat(
                (
                    row,
                    torch.as_tensor((float(result.num_atoms),), dtype=dtype, device=device),
                ),
                dim=0,
            )
        rows.append(row)
        atom_counts.append(int(result.num_atoms))
        targets.append(float(energy_target))
    design = torch.stack(tuple(rows), dim=0)
    target = torch.as_tensor(targets, dtype=dtype, device=device)
    weight = float(energy_weight)
    if weight <= 0.0:
        raise ValueError("energy_weight must be positive for energy-only A_s matrix-unit/global-coupler fitting.")
    sqrt_rows = torch.as_tensor(
        np.sqrt(np.maximum(resolved_structure_weights, 0.0) * weight),
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)
    weighted_design = design * sqrt_rows
    weighted_target = target * sqrt_rows.reshape(-1)
    normal = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_target
    ridge = float(ridge_alpha)
    if ridge < 0.0:
        raise ValueError("ridge_alpha must be nonnegative.")
    if ridge > 0.0:
        normal = normal + ridge * torch.eye(int(normal.shape[0]), dtype=dtype, device=device)
    solve_method = "torch.linalg.solve"
    try:
        coefficients = torch.linalg.solve(normal, rhs)
    except RuntimeError:
        coefficients = torch.linalg.pinv(normal) @ rhs
        solve_method = "torch.linalg.pinv"
    predictions = design @ coefficients
    residual = predictions - target
    metadata = {
        "backend": "A_s_matrix_unit_slot_resolved_global_coupler_energy_linear_fit",
        "fit_method": "ridge_normal_equations",
        "ridge_alpha": ridge,
        "energy_weight": weight,
        "force_weight": 0.0,
        "energy_key": str(energy_key),
        "feature_count": int(feature_count or 0),
        "structure_count": int(len(structures)),
        "atom_counts": tuple(int(value) for value in atom_counts),
        "descriptor_shapes": tuple(descriptor_shapes),
        "include_bias_column": bool(include_bias_column),
        "intercept_policy": "atom_count_bias_column" if include_bias_column else "disabled_reference_energy_fit",
        "structure_weights": dict(structure_weight_metadata),
        "solve_method": solve_method,
        "normal_equation_shape": tuple(int(dim) for dim in normal.shape),
        "residual_l2": float(torch.linalg.vector_norm(residual).detach().cpu()),
        "descriptor_runtime_metadata": result_metadata,
        "descriptor_first_flow": (
            "YE3TRepresentation.filtered_A_s -> YE3TDescriptors.ye3t_basis -> "
            "YE3TDescriptorSet.create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor -> "
            "YE3TModel.linear"
        ),
        "runtime_status": "implemented_under_validation",
        "validation_scope": "energy_only_structure_summed_A_s_matrix_unit_global_coupler_rows",
        "force_status": "not_implemented_for_this_evaluated_descriptor_path",
        "known_limitations": (
            "Fits scalar energies only; force/Jacobian rows for this evaluated A_s descriptor are not implemented.",
            "The descriptor path remains slot-resolved scalar matrix-unit/global-coupler application, not the full A_s Young-E3 descriptor family.",
        ),
    }
    return ASMatrixUnitGlobalCouplerLinearModel(
        coefficients.detach(),
        include_bias_column=bool(include_bias_column),
        metadata=metadata,
    )


class YE3TModel:
    """Descriptor-consuming model factory namespace."""


    @staticmethod
    def validate(descriptor, model_family, model_config=None, **kwargs):
        if not isinstance(descriptor, YE3TDescriptorSet):
            family = str(model_family)
            if family == "linear":
                descriptor = YE3TDescriptors.ace(descriptor)
            elif family == "lifted_density":
                descriptor = YE3TDescriptors.ye3t_basis(descriptor)
            elif family == "phi":
                descriptor = YE3TDescriptors.phi(descriptor)
            else:
                descriptor = YE3TDescriptors.ye3t(descriptor)
        cfg = YE3TModel._model_config(model_config, **kwargs)
        family = str(model_family)
        reasons = []
        notes = []
        required = []
        representation = descriptor.representation
        descriptor_family = descriptor.metadata.get("descriptor_family")
        runtime_status = descriptor.metadata.get("runtime_status", "unknown")

        if family == "linear":
            if _is_ye3t_basis_descriptor(descriptor):
                if descriptor_family == "linear_tagged_cauchy_image":
                    if representation.construction_mode != "tagged_cauchy_exact":
                        reasons.append(
                            "tagged-Cauchy linear fitting requires the exact "
                            "tagged-Cauchy representation selector"
                        )
                    if descriptor.metadata.get("tagged_cauchy_image_compiled") is None:
                        reasons.append(
                            "tagged-Cauchy linear fitting requires coefficient materialization"
                        )
                    if descriptor.metadata.get("tagged_cauchy_image_evaluator") is None:
                        reasons.append(
                            "tagged-Cauchy linear fitting requires a bound source evaluator"
                        )
                    notes.append(
                        "linear fitting consumes the compiler-owned exact physical "
                        "image and its division-free analytic adjoint."
                    )
                    required.extend(
                        (
                            "energy and force finite-difference validation",
                            "neighbor permutation and rotation validation",
                            "native bundle and LAMMPS parity before deployment",
                        )
                    )
                    return YE3TModelCompatibility(
                        model_family=family,
                        supported=not reasons,
                        status=(
                            "implemented_under_validation"
                            if not reasons
                            else "unsupported"
                        ),
                        reasons=tuple(reasons),
                        warnings=tuple(notes),
                        required_validation=tuple(required),
                    )
                if descriptor_family == "linear_lifted_cauchy_scalar":
                    if representation.construction_mode != "lifted_cauchy_exact":
                        reasons.append(
                            "lifted-Cauchy linear fitting requires the exact "
                            "lifted-Cauchy representation selector"
                        )
                    if "lifted_cauchy_compiled" not in descriptor.metadata:
                        reasons.append(
                            "lifted-Cauchy linear fitting requires a compiled artifact"
                        )
                    if "lifted_cauchy_source" not in descriptor.metadata:
                        reasons.append(
                            "lifted-Cauchy linear fitting requires a bound source"
                        )
                    notes.append(
                        "linear fitting uses compiler-owned canonical or factored "
                        "symmetric-power block descriptors with exact source adjoints."
                    )
                    required.extend(
                        (
                            "energy and force finite-difference validation",
                            "proper/improper rotation and permutation validation",
                            "native bundle and LAMMPS parity before deployment",
                        )
                    )
                    return YE3TModelCompatibility(
                        model_family=family,
                        supported=not reasons,
                        status=(
                            "implemented_under_validation"
                            if not reasons
                            else "unsupported"
                        ),
                        reasons=tuple(reasons),
                        warnings=tuple(notes),
                        required_validation=tuple(required),
                    )
                descriptor_evaluation = str(
                    cfg.get(
                        "descriptor_evaluation",
                        cfg.get("A_s_descriptor_evaluation", ""),
                    )
                ).strip().lower()
                if descriptor_evaluation:
                    try:
                        if descriptor._normalize_descriptor_evaluation_selector(descriptor_evaluation) is not None:
                            descriptor._require_A_s_matrix_unit_scalar_target(descriptor_evaluation)
                    except (NotImplementedError, ValueError) as exc:
                        reasons.append(str(exc))
                if descriptor_evaluation in {
                    "a_s_matrix_unit_slot_resolved_global_coupler",
                    "matrix_unit_slot_resolved_global_coupler",
                    "slot_resolved_global_coupler",
                }:
                    if descriptor._A_s_matrix_unit_requested_target_L_R() != 0:
                        reasons.append(
                            "A_s matrix-unit/global-coupler linear fitting currently supports only scalar "
                            "structure-summed energy rows; non-scalar L_R>0 descriptor rows are implemented "
                            "for descriptor evaluation and balanced message-state use, not for linear energy fitting."
                        )
                    force_weight = float(cfg.get("force_weight", 0.0))
                    if force_weight != 0.0:
                        reasons.append(
                            "A_s matrix-unit/global-coupler linear fitting currently supports energy rows only; "
                            "set force_weight=0.0."
                        )
                    notes.append(
                        "linear fitting will evaluate "
                        "create_A_s_matrix_unit_slot_resolved_global_coupler_descriptor(...) and fit "
                        "structure-summed scalar energy rows."
                    )
                    required.extend(
                        (
                            "energy-only regression validation",
                            "slot/permutation validation inherited from the A_s matrix-unit descriptor checks",
                            "force/Jacobian implementation before force fitting claims",
                        )
                    )
                    status = "implemented_under_validation" if not reasons else "unsupported"
                    return YE3TModelCompatibility(
                        model_family=family,
                        supported=not reasons,
                        status=status,
                        reasons=tuple(reasons),
                        warnings=tuple(notes),
                        required_validation=tuple(required),
                    )
                if "lifted_density_config" not in descriptor.metadata:
                    reasons.append("A_s linear fitting requires lifted_density_config metadata")
                notes.append(
                    "ye3t_basis linear fitting uses the scalar A_s lifted-density readout backend; "
                    "ordinary ACE linear fitting still uses the ACE descriptor fast path."
                )
                required.extend(
                    (
                        "slot/permutation invariance validation for scalar A_s readout",
                        "force finite-difference or force-Jacobian validation when force rows are used",
                    )
                )
            else:
                if representation.basis_family != "ace" and not representation.uses_trivial_fast_path:
                    reasons.append("Linear ACE fitting currently requires ACE descriptors or the trivial YE3T fast path")
                if descriptor.settings is None or descriptor.site_basis_config is None:
                    reasons.append("linear fitting requires descriptor settings and a site-basis config")
                if not descriptor.supports_runtime_evaluation:
                    reasons.append("linear fitting requires an evaluated descriptor runtime")
            status = "implemented" if not reasons else "unsupported"
        elif family == "lifted_density":
            if not _is_ye3t_basis_descriptor(descriptor):
                reasons.append("lifted_density requires a ye3t_basis descriptor")
            if "lifted_density_config" not in descriptor.metadata:
                reasons.append("lifted_density requires lifted_density_config metadata")
            payload = descriptor.metadata.get("lifted_density_config", {})
            lifted_payload = payload.get("lifted_density", {}) if isinstance(payload, Mapping) else {}
            readout_mode = str(lifted_payload.get("readout_mode", "")).strip().lower()
            subselection = descriptor.metadata.get(
                "representation_subselection",
                representation.metadata.get("representation_subselection", None),
            )
            if subselection is not None:
                try:
                    _a_s_readout_for_subselection(subselection, readout_mode)
                except ValueError as exc:
                    reasons.append(str(exc))
            if subselection == "equivariant" and readout_mode == "symmetric_linear":
                notes.append(
                    "equivariant A_s descriptor uses readout_mode='symmetric_linear'; "
                    "this is supported as a trivial-projection baseline and does not consume "
                    "nontrivial permutation-character features."
                )
            if representation.validation_status not in {
                "implemented_filtered_density_not_full_ye3",
                "implemented_A_s_slot_subselection_not_full_ye3",
                "implemented_A_s_antisymmetric_magnitude_not_full_sign_carrier",
                "implemented_A_s_trivial_slot_fast_path",
                "implemented_A_s_orbital_slot_readout_not_irrep_resolved",
            }:
                notes.append("ye3t_basis currently uses the A_s filtered-density runtime, not a full arbitrary-sector YE3T descriptor runtime")
            if subselection == "fully_antisymmetric":
                notes.append("antisymmetric A_s uses a squared wedge-volume magnitude, not a full sign-carrier feature runtime")
            status = "implemented_filtered_density" if not reasons else "unsupported"
        elif family == "phi":
            if descriptor_family != "phi":
                reasons.append("phi requires a Phi descriptor from YE3TDescriptors.phi(...)")
            if "phi_config" not in descriptor.metadata:
                reasons.append("phi requires phi_config metadata")
            if descriptor.metadata.get("runtime_status_detail") == "implemented_phi_complete_reference_path":
                notes.append(
                    "explicit Phi complete-inventory path records exact small-rank Young-subgroup/Specht sectors "
                    "with cached couplers; benchmark before performance claims"
                )
                status = "implemented_phi_complete_reference_path" if not reasons else "unsupported"
            else:
                notes.append(
                    "explicit Phi/barPhi motifs are validation/ablation-scale paths; use benchmarks before performance claims"
                )
                status = "implemented_phi_reference_path" if not reasons else "unsupported"
        else:
            reasons.append(f"unknown model family {family!r}")
            status = "unknown_model_family"

        return YE3TModelCompatibility(
            model_family=family,
            supported=not reasons,
            status=status,
            reasons=tuple(reasons),
            warnings=tuple(notes),
            required_validation=tuple(required),
        )

    @staticmethod
    def validate_A_s_permutation_equivariance(descriptor, model_config=None, *, seed=0, atol=1.0e-10, rtol=1.0e-10):
        """Run a direct slot-permutation invariance check for implemented ``A_s`` readouts.

        This is an executable regression check on the selected readout, not a
        proof of the full Young-E3 construction.
        """

        if not isinstance(descriptor, YE3TDescriptorSet):
            descriptor = YE3TDescriptors.ye3t_basis(descriptor)
        YE3TModel.validate(descriptor, "lifted_density", model_config).require_supported()
        model = YE3TModel.lifted_density(descriptor, model_config)
        lifted = model.config.lifted_density
        slot_count = int(lifted.num_filters)
        channel_count = int(len(lifted.channels))
        generator = torch.Generator().manual_seed(int(seed))
        density = torch.randn(4, slot_count, channel_count, dtype=model.config.torch_dtype, generator=generator)
        if slot_count > 1:
            permutation = torch.arange(slot_count - 1, -1, -1, dtype=torch.long)
        else:
            permutation = torch.arange(slot_count, dtype=torch.long)

        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                parameter.fill_(0.02 * float(index + 1))
        if lifted.readout_mode == "slot_equivariant":
            left = model.slot_equivariant_site_energy(density)
            right = model.slot_equivariant_site_energy(density.index_select(1, permutation))
        elif lifted.readout_mode == "character_quadratic":
            left = model.character_quadratic_site_energy(density)
            right = model.character_quadratic_site_energy(density.index_select(1, permutation))
        elif lifted.readout_mode == "antisymmetric_quadratic":
            left = model.antisymmetric_quadratic_site_energy(density)
            right = model.antisymmetric_quadratic_site_energy(density.index_select(1, permutation))
        elif lifted.readout_mode == "ye3_quadratic":
            left = model.ye3_quadratic_site_energy(density)
            right = model.ye3_quadratic_site_energy(density.index_select(1, permutation))
        elif lifted.readout_mode == "ye3_power":
            left = model.ye3_power_site_energy(density)
            right = model.ye3_power_site_energy(density.index_select(1, permutation))
        elif lifted.readout_mode == "ye3_slot_specht_power":
            left = model.ye3_slot_specht_power_site_energy(density)
            right = model.ye3_slot_specht_power_site_energy(density.index_select(1, permutation))
        elif lifted.readout_mode == "symmetric_linear":
            slot_sum = density.sum(dim=1)
            relabeled_sum = density.index_select(1, permutation).sum(dim=1)
            left = slot_sum @ model.channel_readout.to(dtype=density.dtype)
            right = relabeled_sum @ model.channel_readout.to(dtype=density.dtype)
        else:
            raise ValueError(
                "Direct A_s permutation validation is only available for tied or sector-resolved readouts."
            )
        diff = torch.max(torch.abs(left - right)) if left.numel() else left.new_tensor(0.0)
        passed = bool(torch.allclose(left, right, atol=float(atol), rtol=float(rtol)))
        return {
            "passed": passed,
            "readout_mode": lifted.readout_mode,
            "representation_subselection": descriptor.metadata.get("representation_subselection"),
            "slot_count": slot_count,
            "channel_count": channel_count,
            "permutation": [int(v) for v in permutation.tolist()],
            "max_abs_error": float(diff.detach().cpu()),
            "atol": float(atol),
            "rtol": float(rtol),
            "scope": "slot-density readout permutation invariance check; not a proof of full Young-E3 runtime",
        }

    @staticmethod
    def _model_config(config=None, **kwargs):
        if config is None:
            cfg = {}
        elif isinstance(config, Mapping):
            cfg = dict(config.get("model", config))
            config_file = cfg.pop("model_config_file", cfg.pop("config_file", None))
            if config_file is not None:
                cfg = {**YE3TModel._load_model_config_file(config_file), **cfg}
        elif isinstance(config, (str, Path)):
            cfg = YE3TModel._load_model_config_file(config)
        else:
            raise TypeError("model_config must be a mapping, a JSON/YAML path, or None.")
        cfg.update(kwargs)
        return cfg

    @staticmethod
    def _load_model_config_file(path):
        path = Path(path)
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - exercised only without optional yaml.
                raise ImportError("Reading YAML model configs requires PyYAML.") from exc
            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("YE3T model config files must contain a mapping.")
        return dict(payload.get("model", payload))


























    @staticmethod
    def linear(descriptor, model_config=None, *, structures=None, **kwargs):
        if not isinstance(descriptor, YE3TDescriptorSet):
            descriptor = YE3TDescriptors.ace(descriptor)
        cfg = YE3TModel._model_config(model_config, **kwargs)
        if isinstance(model_config, Mapping) and "basis" in model_config and "model" in model_config:
            if cfg.pop("type", None) != "linear":
                raise ValueError("The homogeneous model.type must be linear for YE3TModel.linear.")
            targets_config = model_config.get("targets", {})
            for key in ("energy_key", "force_key", "stress_key"):
                if key in targets_config:
                    cfg.setdefault(key, targets_config[key])
        YE3TModel.validate(descriptor, "linear", cfg).require_supported()
        tagged_cauchy_fit = (
            descriptor.metadata.get("descriptor_family")
            == "linear_tagged_cauchy_image"
        )
        if structures is None:
            structures = cfg.pop("structures", None)
        has_cached_tagged_problem = (
            tagged_cauchy_fit and cfg.get("normal_equations") is not None
        )
        if structures is None and not has_cached_tagged_problem:
            raise KeyError("YE3TModel.linear requires structures or model_config['structures'].")
        structures = [] if structures is None else list(structures)
        energy_key = str(cfg.pop("energy_key", "energy"))
        force_key_config = cfg.pop("force_key", None)
        force_key = "forces" if force_key_config is None else str(force_key_config)
        reference_energies = _normalize_linear_fit_reference_energies(
            cfg.pop("reference_energies", cfg.pop("energy_reference_energies", None))
        )
        reference_target_metadata = {"enabled": False}
        if reference_energies:
            structures, reference_target_metadata = _structures_with_reference_energy_targets(
                structures,
                energy_key=energy_key,
                force_key=force_key,
                reference_energies=reference_energies,
            )
        structure_weight_options = {
            "structure_weights": cfg.pop("structure_weights", None),
            "structure_weight_key": cfg.pop("structure_weight_key", None),
            "structure_group_key": cfg.pop("structure_group_key", None),
            "structure_group_weights": cfg.pop("structure_group_weights", None),
            "structure_group_default_weight": cfg.pop(
                "structure_group_default_weight", None
            ),
            "structure_group_normalize_mean": bool(
                cfg.pop("structure_group_normalize_mean", True)
            ),
            "boltzmann_temperature_K": cfg.pop("boltzmann_temperature_K", cfg.pop("boltzmann_temperature", None)),
            "boltzmann_energy_key": cfg.pop("boltzmann_energy_key", None),
            "boltzmann_weight_nugget": float(cfg.pop("boltzmann_weight_nugget", 0.0)),
            "boltzmann_weight_prefactor": float(cfg.pop("boltzmann_weight_prefactor", 1.0)),
            "boltzmann_normalize_mean": bool(cfg.pop("boltzmann_normalize_mean", True)),
            "min_structure_weight": float(cfg.pop("min_structure_weight", 0.0)),
        }
        if tagged_cauchy_fit:
            from ye3t_ace.tagged_cauchy_image_fit import (
                fit_tagged_cauchy_image_linear_model,
                solve_tagged_cauchy_image_ridge,
                tagged_cauchy_reference_target_metadata,
            )

            reference_potential_metadata = cfg.pop(
                "reference_potential_metadata", None
            )
            restore_references = bool(cfg.pop("restore_references", False))
            if restore_references and reference_potential_metadata is not None and (
                reference_potential_metadata.get("schema") != "ye3t_portable_zbl_reference_v1"
            ):
                raise ValueError("Automatic reference restoration requires the portable ZBL reference.")
            target_energies = cfg.pop("target_energies", None)
            target_forces = cfg.pop("target_forces", None)
            target_stresses = cfg.pop("target_stresses", None)
            stress_key = str(cfg.pop("stress_key", "stress"))
            stress_weight = float(cfg.pop("stress_weight", 0.0))
            if reference_energies and target_energies is not None:
                offsets = np.asarray(
                    [
                        _reference_energy_offset_for_atoms(
                            atoms, reference_energies
                        )
                        for atoms in structures
                    ],
                    dtype=np.float64,
                )
                target_energies = np.asarray(
                    target_energies, dtype=np.float64
                ) - offsets
            target_transform_metadata = tagged_cauchy_reference_target_metadata(
                reference_energies=reference_energies,
                reference_potential_metadata=reference_potential_metadata,
            )
            legacy_l2 = cfg.pop("l2", None)
            ridge_alpha = cfg.pop("ridge_alpha", None)
            if legacy_l2 is not None and ridge_alpha is not None and not np.isclose(
                float(legacy_l2), float(ridge_alpha), rtol=0.0, atol=0.0
            ):
                raise ValueError("ridge_alpha and its legacy l2 alias disagree.")
            ridge_value = (
                float(ridge_alpha)
                if ridge_alpha is not None
                else float(legacy_l2)
                if legacy_l2 is not None
                else 0.0
            )
            fit_method = str(cfg.pop("fit_method", "ridge_streaming_gram"))
            if fit_method not in {"ridge", "ridge_streaming_gram"}:
                raise ValueError(
                    "Tagged-Cauchy V3 currently supports ridge_streaming_gram only."
                )
            device = str(cfg.pop("device", "cpu")).strip().lower()
            if device != "cpu":
                raise ValueError(
                    "Tagged-Cauchy V3 fitting currently requires device='cpu'."
                )
            normal_equations = cfg.pop("normal_equations", None)
            if normal_equations is None:
                if not structures:
                    raise ValueError(
                        "Tagged-Cauchy fitting requires structures when normal_equations "
                        "are not supplied."
                    )
                if reference_potential_metadata is not None and (
                    target_energies is None or target_forces is None
                ):
                    raise ValueError(
                        "An external reference potential requires explicit residual "
                        "target_energies and target_forces."
                    )
                if reference_potential_metadata is not None and stress_weight > 0.0 and target_stresses is None:
                    raise ValueError("An external reference potential requires explicit residual target_stresses.")
                fitted = fit_tagged_cauchy_image_linear_model(
                    descriptor,
                    structures,
                    energy_key=energy_key,
                    force_key=force_key,
                    target_energies=target_energies,
                    target_forces=target_forces,
                    target_stresses=target_stresses,
                    stress_key=stress_key,
                    energy_weight=float(cfg.pop("energy_weight", 1.0)),
                    force_weight=float(cfg.pop("force_weight", 1.0)),
                    stress_weight=stress_weight,
                    ridge_alpha=ridge_value,
                    geometry_cache_dir=cfg.pop("geometry_cache_dir", None),
                    reference_target_metadata=target_transform_metadata,
                    fit_intercept=cfg.pop("fit_intercept", True),
                    **structure_weight_options,
                )
            else:
                if target_energies is not None or target_forces is not None or target_stresses is not None:
                    raise ValueError(
                        "Explicit targets are already bound into normal_equations."
                    )
                if stress_weight != 0.0:
                    raise ValueError("Stress weighting is already bound into normal_equations.")
                nondefault_weight_options = {
                    key: value
                    for key, value in structure_weight_options.items()
                    if (
                        key
                        in {
                            "structure_weights",
                            "structure_weight_key",
                            "structure_group_key",
                            "structure_group_weights",
                            "structure_group_default_weight",
                            "boltzmann_temperature_K",
                            "boltzmann_energy_key",
                        }
                        and value is not None
                    )
                    or key == "boltzmann_weight_nugget"
                    and value != 0.0
                    or key == "boltzmann_weight_prefactor"
                    and value != 1.0
                    or key == "min_structure_weight"
                    and value != 0.0
                }
                if nondefault_weight_options:
                    raise ValueError(
                        "Structure-weight options are already bound into normal_equations."
                    )
                if (
                    (reference_energies or reference_potential_metadata is not None)
                    and normal_equations.get("reference_target_metadata")
                    != target_transform_metadata
                ):
                    raise ValueError(
                        "Reference-target metadata disagrees with normal_equations."
                    )
                fitted = solve_tagged_cauchy_image_ridge(
                    descriptor,
                    normal_equations,
                    ridge_alpha=ridge_value,
                )
            if cfg:
                raise ValueError(
                    "Unsupported YE3TModel.linear tagged-Cauchy config keys: "
                    f"{sorted(cfg)}"
                )
            bound_target_transform = fitted.fit_metadata[
                "reference_target_metadata"
            ]
            if bound_target_transform is None:
                bound_target_transform = tagged_cauchy_reference_target_metadata(
                    reference_energies={}, reference_potential_metadata=None
                )
            fitted.fit_metadata["reference_energy_targets"] = dict(
                bound_target_transform["elemental_energy_offsets"]
            )
            fitted.fit_metadata["reference_potential"] = dict(
                bound_target_transform["external_reference_potential"]
            )
            if restore_references:
                fitted.reference_terms = {
                    "atomic_energies": dict(bound_target_transform["elemental_energy_offsets"]["reference_energies"]),
                    "zbl": bound_target_transform["external_reference_potential"]["metadata"],
                }
            fitted.fit_metadata["restore_references"] = restore_references
            fitted._ye3t_linear_fit_metadata = dict(fitted.fit_metadata)
            return fitted
        if descriptor.metadata.get("descriptor_family") == "linear_lifted_cauchy_scalar":
            legacy_l2 = cfg.pop("l2", None)
            ridge_alpha = cfg.pop("ridge_alpha", None)
            if legacy_l2 is not None and ridge_alpha is not None and not np.isclose(
                float(legacy_l2), float(ridge_alpha), rtol=0.0, atol=0.0
            ):
                raise ValueError("ridge_alpha and its legacy l2 alias disagree.")
            ridge_value = (
                float(ridge_alpha)
                if ridge_alpha is not None
                else float(legacy_l2)
                if legacy_l2 is not None
                else 0.0
            )
            from ye3t_ace.lifted_cauchy_linear import (
                fit_lifted_cauchy_linear_model,
            )

            ordinary_descriptor = cfg.pop("ordinary_descriptor", None)
            ordinary_descriptor_matrix_cache = cfg.pop(
                "ordinary_descriptor_matrix_cache", None
            )
            fitted = fit_lifted_cauchy_linear_model(
                descriptor,
                structures,
                energy_key=energy_key,
                force_key=force_key,
                energy_weight=float(cfg.pop("energy_weight", 1.0)),
                force_weight=float(cfg.pop("force_weight", 1.0)),
                ridge_alpha=ridge_value,
                svd_rcond=float(cfg.pop("svd_rcond", 1.0e-12)),
                feature_chunk_size=cfg.pop("feature_chunk_size", "auto"),
                realization=str(cfg.pop("realization", "factored")),
                source_realization=str(cfg.pop("source_realization", "auto")),
                fit_coordinate_policy=str(
                    cfg.pop("fit_coordinate_policy", "orthogonal")
                ),
                fit_method=str(cfg.pop("fit_method", "ridge_streaming_gram")),
                normal_equation_maximum_condition=float(
                    cfg.pop("normal_equation_maximum_condition", 1.0e24)
                ),
                device=str(cfg.pop("device", "cpu")),
                evaluation_dtype=str(
                    cfg.pop("evaluation_dtype", cfg.pop("dtype", "float64"))
                ),
                accumulation_dtype=str(
                    cfg.pop("accumulation_dtype", "float64")
                ),
                progress=cfg.pop("progress", None),
                **structure_weight_options,
                ordinary_descriptor=ordinary_descriptor,
                ordinary_descriptor_matrix_cache=(
                    ordinary_descriptor_matrix_cache
                ),
            )
            if cfg:
                raise ValueError(
                    "Unsupported YE3TModel.linear lifted-Cauchy config keys: "
                    f"{sorted(cfg)}"
                )
            fitted.fit_metadata["reference_energy_targets"] = dict(
                reference_target_metadata
            )
            fitted._ye3t_linear_fit_metadata = dict(fitted.fit_metadata)
            return fitted
        if _is_ye3t_basis_descriptor(descriptor):
            descriptor_evaluation = str(
                cfg.pop(
                    "descriptor_evaluation",
                    cfg.pop("A_s_descriptor_evaluation", ""),
                )
            ).strip().lower()
            if descriptor_evaluation in {
                "a_s_matrix_unit_slot_resolved_global_coupler",
                "matrix_unit_slot_resolved_global_coupler",
                "slot_resolved_global_coupler",
            }:
                force_weight = float(cfg.pop("force_weight", 0.0))
                if force_weight != 0.0:
                    raise NotImplementedError(
                        "YE3TModel.linear with descriptor_evaluation="
                        "'A_s_matrix_unit_slot_resolved_global_coupler' supports energy rows only; "
                        "set force_weight=0.0 until descriptor force/Jacobian rows are implemented."
                    )
                if force_key_config is not None:
                    warnings.warn(
                        "force_key is ignored for energy-only A_s matrix-unit/global-coupler fitting.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                supported_keys = {
                    "energy_key",
                    "energy_weight",
                    "ridge_alpha",
                    "l2",
                    "include_bias_column",
                    "fit_intercept",
                    "require_all_sectors",
                    "dtype",
                    "device",
                }
                extras = sorted(set(cfg) - supported_keys)
                if extras:
                    raise ValueError(
                        "Unsupported YE3TModel.linear A_s matrix-unit/global-coupler config keys: "
                        f"{extras}"
                    )
                fitted = _fit_A_s_matrix_unit_global_coupler_energy_linear_model(
                    descriptor,
                    structures,
                    energy_key=energy_key,
                    energy_weight=float(cfg.pop("energy_weight", 1.0)),
                    ridge_alpha=float(cfg.pop("ridge_alpha", cfg.pop("l2", 0.0))),
                    include_bias_column=bool(cfg.pop("include_bias_column", cfg.pop("fit_intercept", True))),
                    require_all_sectors=bool(cfg.pop("require_all_sectors", False)),
                    dtype=cfg.pop("dtype", "float64"),
                    device=cfg.pop("device", None),
                    **structure_weight_options,
                )
                metadata = dict(getattr(fitted, "_ye3t_linear_fit_metadata", {}))
                metadata["reference_energy_targets"] = dict(reference_target_metadata)
                fitted._ye3t_linear_fit_metadata = metadata
                return fitted
            type_map = cfg.pop("type_map", descriptor.type_map)
            payload = dict(descriptor.metadata.get("lifted_density_config", {}))
            from ye3t_ace.lifted_density_fit import fit_lifted_density_linear_model

            model = YE3TModel.lifted_density(descriptor)
            fit_kwargs = {
                "type_map": type_map,
                "energy_key": energy_key,
                "force_key": force_key,
                "energy_weight": float(cfg.pop("energy_weight", 1.0)),
                "force_weight": float(cfg.pop("force_weight", 1.0)),
                "ridge_alpha": float(cfg.pop("ridge_alpha", cfg.pop("l2", 0.0))),
                "dtype": cfg.pop("dtype", payload.get("dtype", "float64")),
                "device": cfg.pop("device", None),
                "force_atom_stride": cfg.pop("force_atom_stride", cfg.pop("force_fit_atom_stride", None)),
                "force_jacobian_chunk_size": cfg.pop("force_jacobian_chunk_size", None),
                "force_jacobian_mode": str(cfg.pop("force_jacobian_mode", "batched_vjp")),
                "feature_budget": cfg.pop("feature_budget", cfg.pop("A_s_feature_budget", None)),
                "feature_selection_policy": cfg.pop(
                    "feature_selection_policy",
                    cfg.pop("A_s_feature_selection_policy", "normal_diagonal"),
                ),
                "selected_feature_indices": cfg.pop("selected_feature_indices", None),
                "include_bias_column": bool(cfg.pop("include_bias_column", cfg.pop("fit_intercept", True))),
                **structure_weight_options,
            }
            if cfg:
                raise ValueError(f"Unsupported YE3TModel.linear A_s config keys: {sorted(cfg)}")
            fitted = fit_lifted_density_linear_model(
                model,
                structures,
                **fit_kwargs,
            )
            metadata = dict(getattr(fitted, "_ye3t_linear_fit_metadata", {}))
            metadata["descriptor_first_flow"] = (
                "YE3TRepresentation.filtered_A_s -> "
                "YE3TDescriptors.ye3t_basis -> YE3TModel.linear"
            )
            metadata["reference_energy_targets"] = dict(reference_target_metadata)
            fitted._ye3t_linear_fit_metadata = metadata
            return fitted
        from ye3t_ace.ace.linear_ace import fit_linear_ace

        fit_kwargs = descriptor.linear_fit_kwargs()
        model_variant_cap = cfg.pop("max_variants_per_label", None)
        if model_variant_cap is not None:
            raise ValueError(
                "The descriptor catalogue and its chemical variants are already "
                "resolved. Set max_variants_per_label in YE3TDescriptors.ace(...), "
                "not YE3TModel.linear(...)."
            )
        legacy_l2 = cfg.pop("l2", None)
        ridge_alpha = cfg.pop("ridge_alpha", None)
        if (
            legacy_l2 is not None
            and ridge_alpha is not None
            and not np.isclose(
                float(legacy_l2),
                float(ridge_alpha),
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise ValueError("ridge_alpha and its legacy l2 alias disagree.")
        fit_kwargs.update(
            {
                "max_variants_per_label": None,
                "energy_key": energy_key,
                "force_key": force_key,
                "epochs": int(cfg.pop("epochs", 200)),
                "lr": float(cfg.pop("lr", 5.0e-2)),
                "energy_weight": float(cfg.pop("energy_weight", 1.0)),
                "force_weight": float(cfg.pop("force_weight", 1.0)),
                "l2": None if legacy_l2 is None else float(legacy_l2),
                "ridge_alpha": None
                if ridge_alpha is None
                else float(ridge_alpha),
                "fit_method": str(cfg.pop("fit_method", "ridge")),
                "fit_objective": cfg.pop("fit_objective", None),
                "svd_rcond": float(cfg.pop("svd_rcond", 1.0e-12)),
                "sklearn_params": cfg.pop("sklearn_params", None),
                "use_descriptor_cache": bool(cfg.pop("use_descriptor_cache", True)),
                "descriptor_matrix_cache": cfg.pop("descriptor_matrix_cache", None),
                "use_descriptor_matrix_cache": bool(cfg.pop("use_descriptor_matrix_cache", False)),
                "force_atom_stride": cfg.pop("force_atom_stride", cfg.pop("force_fit_atom_stride", None)),
                "force_jacobian_mode": str(cfg.pop("force_jacobian_mode", "product_adjoint")),
                "force_jacobian_chunk_size": cfg.pop("force_jacobian_chunk_size", None),
                "normal_equation_solver": str(cfg.pop("normal_equation_solver", "auto")),
                "normal_equation_solver_options": cfg.pop("normal_equation_solver_options", None),
                "progress": cfg.pop("progress", None),
                "feature_budget": cfg.pop("feature_budget", cfg.pop("ACE_feature_budget", None)),
                "feature_selection_policy": cfg.pop(
                    "feature_selection_policy",
                    cfg.pop("ACE_feature_selection_policy", "normal_diagonal"),
                ),
                "selected_feature_indices": cfg.pop("selected_feature_indices", None),
                "include_bias_column": bool(cfg.pop("include_bias_column", cfg.pop("fit_intercept", True))),
                "backend": str(cfg.pop("backend", fit_kwargs["backend"])),
                "strict_backend": bool(cfg.pop("strict_backend", fit_kwargs["strict_backend"])),
                "validate_backend": bool(cfg.pop("validate_backend", fit_kwargs["validate_backend"])),
                "device": cfg.pop("device", fit_kwargs.get("device", None)),
                "factorized_descriptor_runtime_policy": cfg.pop(
                    "factorized_descriptor_runtime_policy",
                    fit_kwargs.get("factorized_descriptor_runtime_policy", None),
                ),
                **structure_weight_options,
            }
        )
        if cfg:
            raise ValueError(f"Unsupported YE3TModel.linear config keys: {sorted(cfg)}")
        fitted = fit_linear_ace(structures, **fit_kwargs)
        if hasattr(fitted, "fit_metadata"):
            fitted.fit_metadata["reference_energy_targets"] = dict(reference_target_metadata)
            catalogue_metadata = descriptor.metadata.get(
                "ordinary_scalar_catalogue",
                None,
            )
            if catalogue_metadata is not None:
                fitted_keys = tuple(str(spec.key) for spec in fitted.descriptor_specs)
                descriptor_keys = tuple(
                    str(spec.key) for spec in descriptor.descriptor_specs
                )
                if fitted_keys != descriptor_keys:
                    raise RuntimeError(
                        "Fitted descriptor order does not match the approved "
                        "ordinary scalar catalogue."
                    )
                fitted.fit_metadata["ordinary_scalar_catalogue"] = json.loads(
                    json.dumps(catalogue_metadata)
                )
        return fitted



    @staticmethod
    def lifted_density(descriptor, model_config=None, **kwargs):
        if not isinstance(descriptor, YE3TDescriptorSet):
            descriptor = YE3TDescriptors.ye3t_basis(descriptor)
        cfg = YE3TModel._model_config(model_config, **kwargs)
        YE3TModel.validate(descriptor, "lifted_density", cfg).require_supported()
        payload = dict(descriptor.metadata.get("lifted_density_config", {}))
        if not payload:
            raise ValueError("YE3TModel.lifted_density requires a ye3t_basis descriptor.")
        if "lifted_density" in cfg:
            payload["lifted_density"] = dict(cfg.pop("lifted_density"))
        for key in ("branches", "dtype"):
            if key in cfg:
                payload[key] = cfg.pop(key)
        if cfg:
            raise ValueError(f"Unsupported YE3TModel.lifted_density config keys: {sorted(cfg)}")
        from ye3t_ace.lifted_density import HybridACELiftedDensityConfig, HybridACELiftedDensityEnergyModel

        return HybridACELiftedDensityEnergyModel(HybridACELiftedDensityConfig.from_dict(payload))

    @staticmethod
    def phi(descriptor, model_config=None, **kwargs):
        if not isinstance(descriptor, YE3TDescriptorSet):
            descriptor = YE3TDescriptors.phi(descriptor)
        cfg = YE3TModel._model_config(model_config, **kwargs)
        YE3TModel.validate(descriptor, "phi", cfg).require_supported()
        payload = dict(descriptor.metadata.get("phi_config", {}))
        if "phi" in cfg:
            payload["phi"] = dict(cfg.pop("phi"))
        for key in ("branches", "dtype"):
            if key in cfg:
                payload[key] = cfg.pop(key)
        if cfg:
            raise ValueError(f"Unsupported YE3TModel.phi config keys: {sorted(cfg)}")
        from ye3t_ace.cluster_phi import HybridACEPhiConfig, HybridACEPhiEnergyModel

        return HybridACEPhiEnergyModel(HybridACEPhiConfig.from_dict(payload))

    @staticmethod
    def phi_star(descriptor, model_config=None, **kwargs):
        if not isinstance(descriptor, YE3TDescriptorSet):
            descriptor = YE3TDescriptors.phi_star(descriptor)
        return YE3TModel.phi(descriptor, model_config, **kwargs)

    @staticmethod
    def phi_complete(descriptor, model_config=None, **kwargs):
        if not isinstance(descriptor, YE3TDescriptorSet):
            descriptor = YE3TDescriptors.phi_complete(descriptor)
        return YE3TModel.phi(descriptor, model_config, **kwargs)



def build_descriptor_calculator(
    settings,
    site_basis_config,
    **kwargs,
):
    warnings.warn(
        "build_descriptor_calculator is a legacy descriptor-runtime helper. "
        "Use YE3TDescriptors.ace(...) or YE3TDescriptors.ye3t(...) for new workflows.",
        FutureWarning,
        stacklevel=2,
    )
    return DescriptorCalculator.from_settings(settings, site_basis_config, _suppress_legacy_warning=True, **kwargs)


__all__ = [
    "ACEDescriptor",
    "ACEDescriptorBatch",
    "ASMatrixUnitDescriptorBatch",
    "ASMatrixUnitDescriptorResult",
    "ASMatrixUnitGlobalCouplerLinearModel",
    "DescriptorCalculator",
    "DescriptorGradientResult",
    "YE3TDescriptorSet",
    "YE3TDescriptors",
    "YE3TModel",
    "YE3TModelCompatibility",
    "YE3TRepresentation",
    "YE3TSectorDescriptorResult",
    "build_descriptor_calculator",
    "build_vdw_site_basis_config",
    "filter_compact_labels_by_specs",
    "filter_descriptor_specs_by_specs",
    "read_ase_structures",
]
