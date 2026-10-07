"""Fixed training-data scales for complete YE3T source carriers."""

import math

import torch


def configure_source_output_scaling(
    module,
    inventory,
    feature_slices,
    normalization,
    context,
    schema,
):
    """Register one fixed scalar scale for each complete source carrier."""

    inventory = tuple(inventory)
    feature_slices = tuple(
        (int(start), int(stop)) for start, stop in tuple(feature_slices)
    )
    if not inventory or len(inventory) != len(feature_slices):
        raise ValueError(
            str(context)
            + " source-output scaling requires one slice per source carrier"
        )
    normalization = dict(normalization or {})
    module._source_output_scaling_inventory = inventory
    module._source_output_scaling_feature_slices = feature_slices
    module._source_output_scaling_context = str(context)
    module._source_output_scaling_schema = str(schema)
    module.source_output_normalization_kind = str(
        normalization.get("kind", "none")
    )
    if module.source_output_normalization_kind not in {
        "none",
        "training_rms",
    }:
        raise ValueError(
            str(context)
            + " source_output_normalization.kind must be none or training_rms"
        )
    module.source_output_catalogue_gain_policy = str(
        normalization.get("catalogue_gain_policy", "none")
    )
    if module.source_output_catalogue_gain_policy not in {
        "none",
        "inverse_sqrt_source_count",
    }:
        raise ValueError(
            str(context)
            + " source_output_normalization.catalogue_gain_policy must be "
            + "none or inverse_sqrt_source_count"
        )
    module.source_output_zero_image_policy = str(
        normalization.get("zero_image_policy", "error")
    )
    if module.source_output_zero_image_policy not in {"error", "identity"}:
        raise ValueError(
            str(context)
            + " source_output_normalization.zero_image_policy must be "
            + "error or identity"
        )
    module.source_output_target_total_power = float(
        normalization.get("target_total_source_power", 1.0)
    )
    if module.source_output_target_total_power <= 0.0:
        raise ValueError(
            str(context) + " target_total_source_power must be positive"
        )
    source_count = len(inventory)
    module.source_output_catalogue_gain = (
        math.sqrt(module.source_output_target_total_power / source_count)
        if module.source_output_catalogue_gain_policy
        == "inverse_sqrt_source_count"
        else 1.0
    )
    module.source_output_normalization_nugget = float(
        normalization.get("nugget", 1.0e-12)
    )
    module.source_output_normalization_minimum_scale = float(
        normalization.get("minimum_scale", 1.0e-6)
    )
    module.source_output_normalization_maximum_scale = float(
        normalization.get("maximum_scale", 1.0e12)
    )
    if module.source_output_normalization_nugget <= 0.0:
        raise ValueError(
            str(context) + " source-output normalization nugget must be positive"
        )
    if (
        module.source_output_normalization_minimum_scale <= 0.0
        or module.source_output_normalization_maximum_scale
        < module.source_output_normalization_minimum_scale
    ):
        raise ValueError(
            str(context) + " source-output normalization scales are inconsistent"
        )
    total_width = int(feature_slices[-1][1])
    module.register_buffer(
        "source_output_rms",
        torch.ones(source_count, dtype=torch.float64),
    )
    module.register_buffer(
        "source_output_scales",
        torch.ones(source_count, dtype=torch.float32),
    )
    module.register_buffer(
        "source_feature_scales",
        torch.ones(total_width, dtype=torch.float32),
    )
    module.register_buffer(
        "source_output_scaling_calibrated",
        torch.tensor(
            module.source_output_normalization_kind == "none",
            dtype=torch.bool,
        ),
    )


def reset_source_output_scaling(module):
    """Reset scales before measuring RMS values on the declared train split."""

    with torch.no_grad():
        module.source_output_rms.fill_(1.0)
        module.source_output_scales.fill_(1.0)
        module.source_feature_scales.fill_(1.0)
        module.source_output_scaling_calibrated.fill_(
            module.source_output_normalization_kind == "none"
        )


def set_source_output_rms(module, source_rms):
    """Install one scalar per carrier, shared over tableau and magnetic axes.

    For a complete carrier ``x[a,t,m]``, this applies
    ``x'[a,t,m] = s[a] x[a,t,m]``. Because ``s[a]`` is independent of
    tableau ``t`` and magnetic coordinate ``m``, the map commutes with the
    declared Young and O(3) actions. The scale is fixed from the training split
    and is neither a learned map nor a neighbor/role-dependent normalization.
    """

    source_rms = torch.as_tensor(
        source_rms,
        dtype=torch.float64,
        device=module.source_output_rms.device,
    ).reshape(-1)
    source_count = len(module._source_output_scaling_inventory)
    if int(source_rms.numel()) != source_count:
        raise ValueError(
            "source RMS count must match the "
            + module._source_output_scaling_context
            + " source inventory"
        )
    if not bool(torch.isfinite(source_rms).all()) or bool(
        (source_rms < 0.0).any()
    ):
        raise ValueError(
            module._source_output_scaling_context
            + " source RMS values must be finite and nonnegative"
        )
    scales = (
        module.source_output_catalogue_gain
        * torch.reciprocal(
            source_rms.clamp_min(module.source_output_normalization_nugget)
        )
    ).clamp(
        min=module.source_output_normalization_minimum_scale,
        max=module.source_output_normalization_maximum_scale,
    )
    zero_images = source_rms == 0.0
    if (
        module.source_output_zero_image_policy == "identity"
        and bool(zero_images.any())
    ):
        scales = torch.where(zero_images, torch.ones_like(scales), scales)
    feature_scales = torch.cat(
        tuple(
            scales[int(source_index)].expand(int(stop) - int(start))
            for source_index, (start, stop) in enumerate(
                module._source_output_scaling_feature_slices
            )
        ),
        dim=0,
    )
    with torch.no_grad():
        module.source_output_rms.copy_(source_rms)
        module.source_output_scales.copy_(
            scales.to(dtype=module.source_output_scales.dtype)
        )
        module.source_feature_scales.copy_(
            feature_scales.to(dtype=module.source_feature_scales.dtype)
        )
        module.source_output_scaling_calibrated.fill_(True)


def source_output_scaling_report(module):
    """Return a serializable audit of the fixed complete-carrier scales."""

    rows = []
    rms = module.source_output_rms.detach().cpu().tolist()
    scales = module.source_output_scales.detach().cpu().tolist()
    for inventory, value, scale in zip(
        module._source_output_scaling_inventory,
        rms,
        scales,
    ):
        layout = dict(inventory["carrier_layout"])
        identifier = inventory.get("support_graph_id")
        if identifier is None:
            identifier = inventory.get(
                "descriptor_id", inventory.get("path_id", "source")
            )
        row = {
            "source_index": int(inventory["source_index"]),
            "source_id": str(identifier),
            "rank": int(inventory["rank"]),
            "partition": tuple(layout["key"]["partition"]),
            "L": int(inventory["L_R"]),
            "rms": float(value),
            "scale": float(scale),
        }
        if inventory.get("support_graph_id") is not None:
            row["support_graph_id"] = str(inventory["support_graph_id"])
        if inventory.get("descriptor_id") is not None:
            row["descriptor_id"] = str(inventory["descriptor_id"])
        rows.append(row)
    return {
        "schema": module._source_output_scaling_schema,
        "kind": str(module.source_output_normalization_kind),
        "catalogue_gain_policy": str(
            module.source_output_catalogue_gain_policy
        ),
        "catalogue_gain": float(module.source_output_catalogue_gain),
        "zero_image_policy": str(module.source_output_zero_image_policy),
        "zero_image_source_indices": tuple(
            int(value)
            for value in torch.nonzero(
                module.source_output_rms == 0.0,
                as_tuple=False,
            ).reshape(-1).detach().cpu().tolist()
        ),
        "source_count": int(len(module._source_output_scaling_inventory)),
        "target_total_source_power": float(
            module.source_output_target_total_power
        ),
        "calibrated": bool(
            module.source_output_scaling_calibrated.detach().cpu()
        ),
        "nugget": float(module.source_output_normalization_nugget),
        "minimum_scale": float(
            module.source_output_normalization_minimum_scale
        ),
        "maximum_scale": float(
            module.source_output_normalization_maximum_scale
        ),
        "rows": tuple(rows),
    }
