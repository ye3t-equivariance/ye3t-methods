
"""Friendly label-generation entry points for ye3t_ace."""

from ye3t.couplings import count as count_couplings

from ye3t_ace.ace_labeler import ExactACELabeler
from ye3t_ace.equivariant_calc.descriptor_sets import DescriptorGenerationSettings, enumerate_compact_labels, normalize_basis_mode
from ye3t_ace.equivariant_calc.labeling import CompactLabel, normalize_compact_label


def make_labeler(
    nin,
    lin,
    *,
    tree_type = "balanced",
    strict_target_validation = False,
):
    return ExactACELabeler(
        list(nin),
        list(lin),
        tree_type=tree_type,
        strict_target_validation=bool(strict_target_validation),
    )


def make_ye3t_ace_labels(
    nin,
    lin,
    target_l,
    *,
    tree_type = "balanced",
    strict_target_validation = False,
):
    report = count_couplings(
        content=tuple(nin),
        input_Ls=tuple(lin),
        target_L=int(target_l),
        tree_schedule=tree_type,
        carrier="ACE_density",
        target_permutation="trivial",
        validation_scope="counts",
        metadata={
            "consumer": "ye3t_ace.ace.labels.make_ye3t_ace_labels",
            "strict_target_validation": bool(strict_target_validation),
        },
    )
    return tuple(normalize_compact_label(label) for label in report.labels_for_target(int(target_l)))


def enumerate_descriptor_labels(
    settings,
    *,
    basis_mode = None,
    exact_primitive_timeout_seconds = None,
):
    chosen_basis_mode = normalize_basis_mode(
        basis_mode,
        L_R=settings.L_R,
    )
    return tuple(
        enumerate_compact_labels(
            settings,
            basis_mode=chosen_basis_mode,
            exact_primitive_timeout_seconds=exact_primitive_timeout_seconds,
        )
    )


__all__ = [
    "CompactLabel",
    "DescriptorGenerationSettings",
    "ExactACELabeler",
    "enumerate_descriptor_labels",
    "make_ye3t_ace_labels",
    "make_labeler",
    "normalize_basis_mode",
]
