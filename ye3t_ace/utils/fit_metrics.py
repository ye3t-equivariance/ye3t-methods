"""Shared fit-metric table helpers for example workflows."""

import csv
import math
from pathlib import Path


FIT_METRIC_COLUMNS = (
    "model",
    "split",
    "target",
    "L_R",
    "Lambda",
    "metric",
    "value",
    "unit",
    "n_samples",
    "rotation_equivariance_error",
    "permutation_equivariance_error",
    "label",
    "notes",
)


def _optional_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return value


def fit_metric_row(
    *,
    model,
    split,
    target,
    L_R,
    Lambda="sym",
    metric,
    value,
    unit,
    n_samples=None,
    rotation_equivariance_error=None,
    permutation_equivariance_error=None,
    notes="",
):
    """Return one normalized metric row for example fit reports."""
    metric_name = str(metric).upper()
    label = f"{metric_name} {target} ({unit}) L_R={int(L_R)}, Lambda={Lambda}"
    return {
        "model": str(model),
        "split": str(split),
        "target": str(target),
        "L_R": int(L_R),
        "Lambda": str(Lambda),
        "metric": metric_name,
        "value": _optional_float(value),
        "unit": str(unit),
        "n_samples": "" if n_samples is None else int(n_samples),
        "rotation_equivariance_error": _optional_float(rotation_equivariance_error),
        "permutation_equivariance_error": _optional_float(permutation_equivariance_error),
        "label": label,
        "notes": str(notes),
    }


def potential_metric_rows(metrics, *, model, split="eval", Lambda="sym"):
    """Convert scalar-potential metric dictionaries to normalized rows."""
    rows = []
    target_specs = (
        ("E", 0, "energy_mae", "energy_rmse", "eV"),
        ("E", 0, "energy_mae_per_atom", "energy_rmse_per_atom", "eV/atom"),
        ("F", 1, "forces_mae", "forces_rmse", "eV/Angstrom"),
    )
    for target, L_R, mae_key, rmse_key, unit in target_specs:
        for metric_name, key in (("mae", mae_key), ("rmse", rmse_key)):
            if key not in metrics:
                continue
            rows.append(
                fit_metric_row(
                    model=model,
                    split=split,
                    target=target,
                    L_R=L_R,
                    Lambda=Lambda,
                    metric=metric_name,
                    value=metrics[key],
                    unit=unit,
                    n_samples=metrics.get("num_structures"),
                )
            )
    return rows


def regression_metric_rows(metrics_by_split, *, model, target, L_R, unit, Lambda="sym"):
    """Convert a nested split -> {mae, rmse, n_samples} mapping to rows."""
    rows = []
    for split, split_payload in metrics_by_split.items():
        for metric_name in ("mae", "rmse"):
            if metric_name not in split_payload:
                continue
            rows.append(
                fit_metric_row(
                    model=model,
                    split=split,
                    target=target,
                    L_R=L_R,
                    Lambda=Lambda,
                    metric=metric_name,
                    value=split_payload[metric_name],
                    unit=unit,
                    n_samples=split_payload.get("n_samples"),
                )
            )
    return rows


def write_fit_metric_table(rows, path):
    """Write normalized metric rows as CSV and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{column: row.get(column, "") for column in FIT_METRIC_COLUMNS} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIT_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized)
    return path


def print_fit_metric_table(title, rows):
    """Print a compact, consistent fit-metric table."""
    print(title)
    print("| split | label | value |")
    print("|---|---|---:|")
    for row in rows:
        value = row.get("value", "")
        if value == "":
            value_text = "nan"
        else:
            value_text = f"{float(value):.6e}"
        print(f"| {row.get('split', '')} | {row.get('label', '')} | {value_text} |")
