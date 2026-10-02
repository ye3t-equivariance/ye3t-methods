"""Application-side adapters for shared YE3T target specs."""

from ye3t.targets import YE3TTargetSpec, target_spec, target_validation_report


def property_target_spec(name, representation=None, derivative_mode=None, convention=None, status=None, metadata=None):
    """Return a validated target spec for an atomistic/materialization workflow."""

    return target_spec(
        name=name,
        representation=representation,
        derivative_mode=derivative_mode,
        convention=convention,
        status=status,
        metadata=metadata,
    )


def stress_target_spec(convention=None, status="stable", metadata=None):
    """Target spec for homogeneous-strain stress rows."""

    convention_payload = {
        "derivative": "homogeneous_strain",
        "row_formula": "dphi_dr_alpha_times_r_beta",
        "volume_normalization": "caller_or_calculator_convention",
        "ase_voigt_sign": "documented_by_calculator_adapter",
    }
    if convention is not None:
        convention_payload.update(dict(convention))
    representation = {
        "kind": "symmetric_cartesian_rank2_or_tesseral_rank2",
        "rotation": {"L_R": (0, 2)},
    }
    return property_target_spec(
        "stress",
        representation=representation,
        derivative_mode="strain",
        convention=convention_payload,
        status=status,
        metadata=metadata,
    )


def charge_target_spec(mode="fixed_scalar_channel", status="experimental", metadata=None):
    """Target spec for scalar charge rows or descriptor charge VJPs."""

    representation = {
        "kind": "scalar_charge",
        "rotation": {"L_R": 0},
        "charge_mode": str(mode),
    }
    convention = {
        "fixed_charges": "scalar_channel_factor",
        "learned_or_equilibrated_charges": "requires_charge_gradient_terms",
    }
    return property_target_spec(
        "charge",
        representation=representation,
        derivative_mode="none",
        convention=convention,
        status=status,
        metadata=metadata,
    )


def target_provenance_report(spec, descriptor_plan=None, cache_report=None, derivative_chain=None, row_source=None):
    """Attach descriptor/cache provenance to a target spec."""

    if not isinstance(spec, YE3TTargetSpec):
        spec = YE3TTargetSpec.from_dict(spec)
    report = target_validation_report(
        spec,
        descriptor_plan=descriptor_plan,
        cache_report=cache_report,
        derivative_chain=derivative_chain,
    )
    report["row_source"] = row_source
    report["package_boundary"] = {
        "target_spec_owner": "ye3t",
        "materialization_owner": "ye3t-methods",
        "coupling_label_owner": "ye3t.couplings_or_descriptor_compiler",
    }
    return report


__all__ = [
    "YE3TTargetSpec",
    "charge_target_spec",
    "property_target_spec",
    "stress_target_spec",
    "target_provenance_report",
]
