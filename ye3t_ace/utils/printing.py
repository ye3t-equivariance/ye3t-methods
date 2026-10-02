"""Stable printing helpers for YE3T-ACE workflows."""

from pathlib import Path
import time

from ye3t_ace.ace.notation import format_ace_basis, format_ace_sector
from ye3t_ace.ace_labeler import ExactACELabeler


def print_config_summary(title, config, keys):
    """Print a compact configuration block with stable key ordering."""
    print(title)
    for key in keys:
        if key in config:
            print(f"  {key}: {config[key]}")


def resolve_output_path(output_dir, default_dir, name):
    """Resolve an output path, creating the selected directory."""
    directory = Path(default_dir if output_dir is None else output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def tensor_summary(values):
    """Return scale diagnostics for one descriptor matrix."""
    real_values = values.real if values.is_complex() else values
    return {
        "shape": tuple(int(v) for v in values.shape),
        "mean_abs": float(real_values.abs().mean().detach().cpu()),
        "max_abs": float(real_values.abs().max().detach().cpu()),
    }


def print_structured_ace_sector(case):
    """Print one ACE sector in a selected block-basis convention."""
    nin = list(case["nin"])
    lin = list(case["lin"])
    L_R = int(case["L_R"])
    block_basis_mode = str(case["block_basis_mode"])
    print_limit = int(case.get("print_limit", 4))
    labeler = ExactACELabeler(
        nin,
        lin,
        strict_target_validation=False,
        block_basis_mode=block_basis_mode,
    )
    sector = labeler.sector_data_for_target(L_R)
    entries = [] if sector is None else list(sector.entries)

    print(f"=== {case['name']} | block_basis_mode={block_basis_mode} ===")
    print(format_ace_sector(nin, lin, L_R, alpha=len(entries), convention=f"ace-compact-v1/{block_basis_mode}"))
    for index, entry in enumerate(entries[:print_limit]):
        print(" ", format_ace_basis(index, entry.compact_label))
    if len(entries) > print_limit:
        print(f"  ... and {len(entries) - print_limit} more")
    print()


def print_ace_covariant_counts(name, nin, lin, max_target_L):
    """Print exact ACE multiplicities without building compact labels."""
    labeler = ExactACELabeler(nin, lin, strict_target_validation=False)
    t0 = time.perf_counter()
    counts = {int(L): int(count) for L, count in labeler.counts_by_L().items() if int(count) > 0}
    if max_target_L is not None:
        counts = {int(L): int(count) for L, count in counts.items() if int(L) <= int(max_target_L)}
    elapsed_s = time.perf_counter() - t0

    print(name)
    print(f"mode=counts_by_L elapsed_s={elapsed_s:.6f}")
    for L_R, count in counts.items():
        print(format_ace_sector(nin, lin, L_R, alpha=count))
    print(f"total={sum(counts.values())}")
    print()


def print_ace_covariant_labels(name, nin, lin, max_target_L, print_limit):
    """Materialize compact ACE path labels grouped by target ``L_R``."""
    labeler = ExactACELabeler(nin, lin, strict_target_validation=False)
    t0 = time.perf_counter()
    labels_by_L = labeler.compact_labels_for_targets("all", max_target_L=max_target_L)
    elapsed_s = time.perf_counter() - t0

    print(name)
    print(f"mode=labels_by_L elapsed_s={elapsed_s:.6f}")
    for L_R, labels in labels_by_L.items():
        print(format_ace_sector(nin, lin, L_R, alpha=len(labels)))
        for index, label in enumerate(labels[:print_limit]):
            print(" ", format_ace_basis(index, label))
        if len(labels) > print_limit:
            print(f"  ... and {len(labels) - print_limit} more")
    print(f"total={sum(len(labels) for labels in labels_by_L.values())}")
    print()


def print_rotation_check_summary(title, result):
    """Print descriptor rotation-check size and maximum error."""
    print(title)
    settings = result.get("settings")
    if settings is not None:
        print("  ranks:", tuple(settings.ranks))
        print("  nmax:", tuple(settings.nmax))
        print("  lmax:", tuple(settings.lmax))
        print("  L_R:", int(settings.L_R))
    print("  labels:", len(result["labels"]))
    print("  max rotation error:", f"{float(result['max_error']):.3e}")


def print_bond_summary(type_map, rc_by_bond, radial_decay):
    """Print ordered pair cutoffs and radial decay scales."""
    inverse = {index: symbol for symbol, index in type_map.items()}
    for left, right in sorted(rc_by_bond):
        pair = f"{inverse[left]}->{inverse[right]}"
        print(f"  {pair}: rc={rc_by_bond[(left, right)]:.3f} decay={radial_decay[(left, right)]:.3f}")


def print_ye3t_mp_path_inventory(name, model, reconstruct_products):
    """Print the route inventory and parameter proxy for one MP model."""
    print(name)
    print(f"  reduced basis mode: {model.reduced_basis_mode}")
    print(f"  reconstruct products: {bool(reconstruct_products)}")
    print(f"  total parameters: {model.parameter_count()}")
    print(f"  message parameters: {model.message_parameter_count()}")
    print(f"  naive message proxy: {model.naive_message_parameter_count()}")
    for row in model.path_inventory():
        route = f"{row['L_in']} x {row['L_geom']} -> {row['L_out']}"
        print(
            "  "
            f"layer {row['layer_index']} | {route:<10} | "
            f"exact dim {row['geometry_exact_dim']} | "
            f"naive dim {row['geometry_naive_dim']} | "
            f"path params {row['path_parameter_count']}"
        )


def print_linear_ace_schedule_summary(schedule):
    """Print full and primitive linear-ACE descriptor group counts."""
    print("linear ACE schedule")
    print("  total labels across modes:", schedule.total_labels)
    for group in schedule.groups[:6]:
        print(
            "  "
            f"mode={group.basis_mode} rank={group.rank} L_R={group.L_R} "
            f"labels={group.label_count}"
        )


def print_ye3t_mp_schedule_summary(full, primitive):
    """Print full and primitive-reconstructed YE3T-equivariant MP route groups."""
    print("YE3T-equivariant MP schedule")
    print("  full exact routes:", full.total_routes)
    print("  primitive reconstructed routes:", primitive.total_routes)
    print("  first primitive group:", primitive.groups[0].as_dict())


def print_product_expansion_reverse_summary(summary):
    """Print a product-expansion reverse-pass comparison."""
    print("product expansion reverse")
    print("  descriptor shape:", summary["descriptor_shape"])
    print("  max adjoint difference:", summary["max_adjoint_difference"])


def print_symmetric_power_summary(title, summary):
    """Print candidate symmetric-power blocks."""
    print(title)
    print("  label source:", summary.metadata["label_source"])
    for group in summary.groups[:3]:
        row = group.as_dict()
        print(
            "  "
            f"rank={row['rank']} block=eta{row['eta']}:L{row['input_L']}^"
            f"{row['power']} outputs={row['outputs']} terms={row['term_counts']}"
        )


def print_symmetric_power_probe_summary(summary):
    """Print one symmetric-power kernel comparison."""
    print("  kernel probe output shape:", summary["output_shape"])
    print("  kernel probe max difference:", summary["max_difference"])


def print_projector_benchmark_summary(title, summary):
    """Print forward/backward projector comparison metrics."""
    print(title)
    print("  feature shape:", summary["feature_shape"])
    print("  max feature difference:", summary["max_feature_difference"])
    print("  gradient norm:", summary["gradient_norm"])
    print("  auto seconds:", f"{summary['auto_seconds']:.6f}")
    print("  reference seconds:", f"{summary['reference_seconds']:.6f}")
    print("  speed ratio:", f"{summary['speed_ratio']:.3f}")


def print_shapes(title, output):
    """Print output tensor shapes by angular momentum channel."""
    print(title)
    for L, tensor in sorted(output.items()):
        print(f"  L={L}: {tuple(tensor.shape)}")


def backend_report(model):
    """Return product backend counts when the model exposes them."""
    if hasattr(model, "product_backend_report"):
        return model.product_backend_report()
    return {"counts": {}}
