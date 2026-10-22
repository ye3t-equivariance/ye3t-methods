#!/usr/bin/env python3
"""Render source-backed publication figures for the frozen six-system study."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import optimize_cached as study


COLORS = {
    "ace": "#4c78a8",
    "ace_curated": "#1f4e79",
    "tagged_ye3t": "#e45756",
    "tagged_ye3t_augmented": "#7a5195",
    "lifted": "#59a14f",
}
LABELS = {
    "ace": "ordinary ACE/PACE",
    "ace_curated": "curated PACE anchor (61)",
    "tagged_ye3t": "tagged YE3T (matched)",
    "tagged_ye3t_augmented": "tagged YE3T (augmented)",
    "lifted": "lifted density",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path):
    resolved = Path(path).resolve()
    workspace = study.PUBLIC.parents[3].resolve()
    try:
        return str(resolved.relative_to(workspace)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def git_state(path):
    root = Path(path).resolve()
    revision = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"path": portable_path(root), "revision": revision, "dirty": dirty}


def optional_git_state(path):
    root = Path(path).resolve()
    if not (root / ".git").exists():
        return {"path": portable_path(root), "available": False}
    return {"available": True, **git_state(root)}


def save(figure, output, stem):
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    figure.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return (png, pdf)


def model_rows(system):
    root = study.system_root(system) / "finalist"
    tests = {row["model"]: row for row in read_csv(root / "test_metrics.csv")}
    models = study.read_json(root / "fits" / "catalogues.json")["models"]
    rows = []
    for model in models:
        fit = study.read_json(root / "fits" / f"{model['name']}.json")
        test = tests[model["name"]]
        rows.append(
            {
                "system": system,
                "model": model["name"],
                "family": model["family"],
                "descriptor_count": int(model["feature_count"]),
                "ordinary_descriptor_count": int(model["ordinary_feature_count"]),
                "tagged_descriptor_count": int(model["tagged_feature_count"]),
                "cv_energy_rmse_mean_eV_per_atom": float(
                    fit["selected"]["energy_rmse_mean_eV_per_atom"]
                ),
                "cv_energy_rmse_std_eV_per_atom": float(
                    fit["selected"]["energy_rmse_std_eV_per_atom"]
                ),
                "cv_force_rmse_mean_eV_per_A": float(
                    fit["selected"]["force_rmse_mean_eV_per_A"]
                ),
                "cv_force_rmse_std_eV_per_A": float(
                    fit["selected"]["force_rmse_std_eV_per_A"]
                ),
                "test_energy_rmse_eV_per_atom": float(
                    test["energy_rmse_eV_per_atom"]
                ),
                "test_force_rmse_eV_per_A": float(test["force_rmse_eV_per_A"]),
                "ridge_alpha": float(fit["selected"]["alpha"]),
                "energy_weight": float(fit["selected"]["energy_weight"]),
                "tagged_relative_penalty": float(
                    fit["selected"]["tagged_penalty"]
                ),
                "group_weights": json.dumps(
                    fit["selected"]["group_weights"], sort_keys=True
                ),
            }
        )
    lifted_path = root / "lifted_control" / "test_summary.json"
    if lifted_path.is_file():
        lifted = study.read_json(lifted_path)
        rows.append(
            {
                "system": system,
                "model": str(lifted["model"]),
                "family": "lifted",
                "descriptor_count": int(lifted["descriptor_count"]),
                "ordinary_descriptor_count": int(
                    lifted["ordinary_descriptor_count"]
                ),
                "tagged_descriptor_count": 0,
                "cv_energy_rmse_mean_eV_per_atom": float(
                    lifted["cv_energy_rmse_mean_eV_per_atom"]
                ),
                "cv_energy_rmse_std_eV_per_atom": float(
                    lifted["cv_energy_rmse_std_eV_per_atom"]
                ),
                "cv_force_rmse_mean_eV_per_A": float(
                    lifted["cv_force_rmse_mean_eV_per_A"]
                ),
                "cv_force_rmse_std_eV_per_A": float(
                    lifted["cv_force_rmse_std_eV_per_A"]
                ),
                "test_energy_rmse_eV_per_atom": float(
                    lifted["test_energy_rmse_eV_per_atom"]
                ),
                "test_force_rmse_eV_per_A": float(
                    lifted["test_force_rmse_eV_per_A"]
                ),
                "ridge_alpha": float(lifted["ridge_alpha"]),
                "energy_weight": float(lifted["energy_weight"]),
                "tagged_relative_penalty": 1.0,
                "group_weights": json.dumps(
                    lifted["group_weights"], sort_keys=True
                ),
            }
        )
    curated_path = study.system_root(system) / "screen" / "metrics.csv"
    if curated_path.is_file():
        curated = next(
            row
            for row in read_csv(curated_path)
            if row["arm"] == "pace_expanded_fixed_radial"
        )
        rows.append(
            {
                "system": system,
                "model": "pace_curated_61_historical_anchor",
                "family": "ace_curated",
                "descriptor_count": int(curated["feature_count"]),
                "ordinary_descriptor_count": int(curated["feature_count"]),
                "tagged_descriptor_count": 0,
                "cv_energy_rmse_mean_eV_per_atom": float(
                    curated["validation_energy_rmse_eV_per_atom"]
                ),
                "cv_energy_rmse_std_eV_per_atom": 0.0,
                "cv_force_rmse_mean_eV_per_A": float(
                    curated["validation_force_rmse_eV_per_A"]
                ),
                "cv_force_rmse_std_eV_per_A": 0.0,
                "test_energy_rmse_eV_per_atom": float(
                    curated["test_energy_rmse_eV_per_atom"]
                ),
                "test_force_rmse_eV_per_A": float(
                    curated["test_force_rmse_eV_per_A"]
                ),
                "ridge_alpha": float(curated["selected_alpha"]),
                "energy_weight": 1.0,
                "tagged_relative_penalty": 1.0,
                "group_weights": "{}",
                "comparison_note": (
                    "retained curated PACE anchor; fixed 5.0 A radial setting and "
                    "earlier fit protocol, shown to detect weak generated controls"
                ),
            }
        )
    return rows


def accuracy_figure(rows, systems, output, metric):
    if metric == "energy":
        cv_key = "cv_energy_rmse_mean_eV_per_atom"
        std_key = "cv_energy_rmse_std_eV_per_atom"
        test_key = "test_energy_rmse_eV_per_atom"
        ylabel = "energy RMSE (eV/atom)"
    else:
        cv_key = "cv_force_rmse_mean_eV_per_A"
        std_key = "cv_force_rmse_std_eV_per_A"
        test_key = "test_force_rmse_eV_per_A"
        ylabel = "force RMSE (eV/Å)"
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.5), squeeze=False)
    for axis, system in zip(axes.flat, systems, strict=True):
        subset = [row for row in rows if row["system"] == system]
        for family in ("ace", "tagged_ye3t"):
            curve = sorted(
                (row for row in subset if row["family"] == family),
                key=lambda row: row["descriptor_count"],
            )
            x = np.asarray([row["descriptor_count"] for row in curve])
            cv = np.asarray([row[cv_key] for row in curve])
            std = np.asarray([row[std_key] for row in curve])
            test = np.asarray([row[test_key] for row in curve])
            axis.plot(x, cv, color=COLORS[family], linestyle="--", linewidth=1.1)
            axis.fill_between(
                x,
                np.maximum(0.0, cv - std),
                cv + std,
                color=COLORS[family],
                alpha=0.13,
            )
            axis.plot(
                x,
                test,
                marker="o" if family == "ace" else "s",
                color=COLORS[family],
                linewidth=1.6,
                label=LABELS[family],
            )
        for family, marker in (
            ("ace_curated", "P"),
            ("tagged_ye3t_augmented", "*"),
            ("lifted", "D"),
        ):
            for row in subset:
                if row["family"] == family:
                    axis.scatter(
                        row["descriptor_count"],
                        row[test_key],
                        marker=marker,
                        s=90 if marker == "*" else 42,
                        color=COLORS[family],
                        edgecolor="black",
                        linewidth=0.45,
                        label=LABELS[family],
                        zorder=4,
                    )
        axis.set_title(system)
        axis.set_xlabel("descriptor count")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles = {}
    for axis in axes.flat:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels, strict=True):
            handles.setdefault(label, handle)
    figure.legend(
        handles.values(), handles.keys(), loc="lower center", ncol=4, frameon=False
    )
    figure.suptitle(
        "Five-fold training uncertainty (bands) and fixed published test errors (symbols)"
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.96))
    return save(figure, output, f"accuracy_vs_descriptor_count_{metric}")


def paired_benefit(rows, systems, output):
    records = []
    for system in systems:
        subset = [row for row in rows if row["system"] == system]
        ace = next(
            row
            for row in subset
            if row["family"] == "ace" and row["descriptor_count"] == 127
        )
        tagged = next(
            row
            for row in subset
            if row["family"] == "tagged_ye3t"
            and row["descriptor_count"] == 127
        )
        records.append(
            {
                "system": system,
                "descriptor_count": 127,
                "energy_improvement_percent": 100.0
                * (
                    1.0
                    - tagged["test_energy_rmse_eV_per_atom"]
                    / ace["test_energy_rmse_eV_per_atom"]
                ),
                "force_improvement_percent": 100.0
                * (
                    1.0
                    - tagged["test_force_rmse_eV_per_A"]
                    / ace["test_force_rmse_eV_per_A"]
                ),
                "ace_energy_rmse_eV_per_atom": ace[
                    "test_energy_rmse_eV_per_atom"
                ],
                "tagged_energy_rmse_eV_per_atom": tagged[
                    "test_energy_rmse_eV_per_atom"
                ],
                "ace_force_rmse_eV_per_A": ace["test_force_rmse_eV_per_A"],
                "tagged_force_rmse_eV_per_A": tagged[
                    "test_force_rmse_eV_per_A"
                ],
            }
        )
    for metric in ("energy", "force"):
        values = np.asarray(
            [row[f"{metric}_improvement_percent"] for row in records]
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        for row in records:
            row[f"aggregate_{metric}_mean_improvement_percent"] = mean
            row[f"aggregate_{metric}_95pct_t_half_width"] = 2.570582 * sem
    write_csv(output / "paired_ye3t_benefit.csv", records)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    x = np.arange(len(systems))
    for axis, metric in zip(axes, ("energy", "force"), strict=True):
        values = [row[f"{metric}_improvement_percent"] for row in records]
        axis.bar(
            x,
            values,
            color=["#59a14f" if value >= 0.0 else "#e15759" for value in values],
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, systems)
        axis.set_ylabel(f"{metric} RMSE improvement (%)")
        axis.set_title(f"Matched 127 descriptors: {metric}")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    artifacts = save(figure, output, "paired_ye3t_benefit")
    return records, artifacts


def timing_rows(systems):
    rows = []
    for system in systems:
        path = (
            study.system_root(system)
            / "finalist"
            / "validation"
            / "timing"
            / "summary.csv"
        )
        if not path.is_file():
            continue
        for row in read_csv(path):
            rows.append(
                {
                    "system": system,
                    "model": row["model"],
                    "family": row["family"],
                    "descriptor_count": int(row["descriptor_count"]),
                    "mpi_ranks": int(row["mpi_ranks"]),
                    "median_microseconds_per_atom_step": float(
                        row["median_microseconds_per_atom_step"]
                    ),
                    "minimum_microseconds_per_atom_step": float(
                        row["minimum_microseconds_per_atom_step"]
                    ),
                    "maximum_microseconds_per_atom_step": float(
                        row["maximum_microseconds_per_atom_step"]
                    ),
                }
            )
    return rows


def pareto_figure(accuracy, timings, systems, output, metric):
    if not timings:
        return ()
    if metric == "energy":
        error_key = "test_energy_rmse_eV_per_atom"
        ylabel = "test energy RMSE (eV/atom)"
    else:
        error_key = "test_force_rmse_eV_per_A"
        ylabel = "test force RMSE (eV/Å)"
    metric_lookup = {
        (row["system"], row["model"]): row for row in accuracy
    }
    source = []
    for timing in timings:
        accuracy_row = metric_lookup[(timing["system"], timing["model"])]
        source.append(
            {
                **timing,
                "test_energy_rmse_eV_per_atom": accuracy_row[
                    "test_energy_rmse_eV_per_atom"
                ],
                "test_force_rmse_eV_per_A": accuracy_row[
                    "test_force_rmse_eV_per_A"
                ],
            }
        )
    write_csv(output / "accuracy_runtime_pareto.csv", source)
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.5), squeeze=False)
    for axis, system in zip(axes.flat, systems, strict=True):
        subset = [
            row
            for row in source
            if row["system"] == system and row["mpi_ranks"] == 1
        ]
        for family in ("ace", "tagged_ye3t", "tagged_ye3t_augmented", "lifted"):
            family_rows = [row for row in subset if row["family"] == family]
            if not family_rows:
                continue
            axis.scatter(
                [row["median_microseconds_per_atom_step"] for row in family_rows],
                [row[error_key] for row in family_rows],
                color=COLORS[family],
                marker="o" if family == "ace" else "s",
                label=LABELS[family],
                alpha=0.9,
            )
        axis.set_title(system)
        axis.set_xlabel("CPU time (µs/atom-step, 1 MPI rank)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    return save(figure, output, f"accuracy_runtime_pareto_{metric}")


def property_figure(systems, output):
    rows = []
    for system in systems:
        path = (
            study.system_root(system)
            / "finalist"
            / "validation"
            / "properties"
            / "properties.csv"
        )
        if not path.is_file():
            continue
        for row in read_csv(path):
            rows.append(
                {
                    "system": system,
                    "model": row["model"],
                    "descriptor_count": int(row["descriptor_count"]),
                    "born_stable": str(row["born_stable"]).lower() == "true",
                    "lattice_constant_abs_percent_error": abs(
                        float(row["equilibrium_lattice_constant_A_percent_error"])
                    ),
                    "bulk_modulus_abs_percent_error": abs(
                        float(row["bulk_modulus_GPa_from_elastic_percent_error"])
                    ),
                    "C11_abs_percent_error": abs(float(row["C11_GPa_percent_error"])),
                    "C12_abs_percent_error": abs(float(row["C12_GPa_percent_error"])),
                    "C44_abs_percent_error": abs(float(row["C44_GPa_percent_error"])),
                }
            )
    if not rows:
        return ()
    write_csv(output / "applied_property_errors.csv", rows)
    model_order = (
        "ace_127",
        "ye3t_tagged_127",
        "ye3t_augmented_196",
        "lifted_125",
    )
    metric_order = (
        "lattice_constant_abs_percent_error",
        "bulk_modulus_abs_percent_error",
        "C11_abs_percent_error",
        "C12_abs_percent_error",
        "C44_abs_percent_error",
    )
    matrix = []
    labels = []
    lookup = {(row["system"], row["model"]): row for row in rows}
    for system in systems:
        for model in model_order:
            row = lookup.get((system, model))
            if row is not None:
                matrix.append([row[key] for key in metric_order])
                labels.append(f"{system} {model}")
    matrix = np.asarray(matrix, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(8.3, max(5.0, 0.32 * len(labels))))
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0)
    axis.set_xticks(
        np.arange(len(metric_order)), ("$a_0$", "$B$", "$C_{11}$", "$C_{12}$", "$C_{44}$")
    )
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axis.set_title("Absolute error to stated DFT/paper reference (%)")
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    figure.tight_layout()
    return save(figure, output, "applied_property_transfer")


def stability_figure(systems, output):
    summaries = []
    traces = []
    for system in systems:
        root = study.system_root(system) / "finalist" / "validation" / "stability"
        summary_path = root / "summary.json"
        if not summary_path.is_file():
            continue
        summary = study.read_json(summary_path)
        for row in summary["nve"]:
            summaries.append(
                {
                    "system": system,
                    "model": row["model"],
                    "nve_passed": bool(row["passed"]),
                    "close_range_passed": bool(summary["close_range_passed"]),
                    "overall_passed": bool(
                        row["passed"] and summary["close_range_passed"]
                    ),
                    "energy_drift_eV_per_atom_per_ps": float(
                        row["energy_drift_eV_per_atom_per_ps"]
                    ),
                    "maximum_energy_excursion_eV_per_atom": float(
                        row["maximum_energy_excursion_eV_per_atom"]
                    ),
                    "minimum_distance_A": float(row["minimum_distance_A"]),
                }
            )
            for trace in read_csv(root / f"nve_trace.{row['model']}.csv"):
                traces.append(
                    {
                        "system": system,
                        "model": row["model"],
                        "time_ps": float(trace["time_ps"]),
                        "total_energy_eV": float(trace["total_energy_eV"]),
                        "minimum_distance_A": float(trace["minimum_distance_A"]),
                    }
                )
    if not summaries:
        return ()
    write_csv(output / "stability_summary.csv", summaries)
    write_csv(output / "stability_traces.csv", traces)
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.5), squeeze=False)
    for axis, system in zip(axes.flat, systems, strict=True):
        subset = [row for row in traces if row["system"] == system]
        for model in sorted({row["model"] for row in subset}):
            curve = [row for row in subset if row["model"] == model]
            energy = np.asarray([row["total_energy_eV"] for row in curve])
            axis.plot(
                [row["time_ps"] for row in curve],
                energy - energy[0],
                linewidth=1.0,
                label=model,
            )
        axis.set_title(system)
        axis.set_xlabel("NVE time (ps)")
        axis.set_ylabel("total-energy change (eV)")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    return save(figure, output, "nve_energy_drift")


def walk_values(value, key, output):
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                output.append(child)
            walk_values(child, key, output)
    elif isinstance(value, list):
        for child in value:
            walk_values(child, key, output)


def sector_inventory(systems, output):
    rows = []
    for system in systems:
        root = study.system_root(system)
        catalogue = study.read_json(root / "finalist" / "fits" / "catalogues.json")
        model = next(row for row in catalogue["models"] if row["name"] == "ye3t_tagged_127")
        components = {row["component_index"]: row for row in catalogue["components"]}
        inactive = {int(index) for index in model.get("inactive_tagged_indices", ())}
        catalogue_root = root / "tagged_catalogue"
        if not catalogue_root.is_dir():
            catalogue_root = root / "catalogue"
        manifest = study.read_json(catalogue_root / "catalogue_manifest.json")
        certified = {row["component_index"]: row for row in manifest["components"]}
        for component_index in model["tagged_components"]:
            component = components[component_index]
            record = certified[component_index]
            artifact = study.read_json(
                catalogue_root / "artifacts" / f"{record['request_hash']}.json"
            )
            kappas = []
            lambdas = []
            walk_values(artifact, "block_kappas", kappas)
            walk_values(artifact, "block_Lambdas", lambdas)
            normalized_kappas = sorted(
                {
                    json.dumps(value, separators=(",", ":"))
                    for value in kappas
                    if isinstance(value, list)
                }
            )
            normalized_lambdas = sorted(
                {
                    json.dumps(value, separators=(",", ":"))
                    for value in lambdas
                    if isinstance(value, list)
                }
            )
            rows.append(
                {
                    "system": system,
                    "model": "ye3t_tagged_127",
                    "tensor_order_N": int(component["tensor_order_N"]),
                    "tag_count_s": int(manifest["tag_count_s"]),
                    "content_pattern": json.dumps(component["content_pattern"]),
                    "selected_feature_count": sum(
                        index not in inactive
                        for index in range(
                            int(component["start"]), int(component["stop"])
                        )
                    ),
                    "unique_block_kappas": ";".join(normalized_kappas),
                    "unique_intermediate_Lambdas": ";".join(normalized_lambdas),
                    "certificate_passed": bool(record["certificates"]["passed"]),
                }
            )
    write_csv(output / "permutation_sector_inventory.csv", rows)
    aggregate = {}
    for row in rows:
        key = (row["system"], row["tensor_order_N"])
        aggregate[key] = aggregate.get(key, 0) + row["selected_feature_count"]
    orders = sorted({key[1] for key in aggregate})
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    width = 0.12
    x = np.arange(len(orders), dtype=np.float64)
    for index, system in enumerate(systems):
        values = [aggregate.get((system, order), 0) for order in orders]
        axis.bar(x + (index - 2.5) * width, values, width=width, label=system)
    axis.set_xticks(x, orders)
    axis.set_xlabel("tensor order $N$")
    axis.set_ylabel("selected tagged coordinates")
    tag_counts = sorted({int(row["tag_count_s"]) for row in rows})
    tag_text = ",".join(str(value) for value in tag_counts)
    axis.set_title(
        f"Matched 127-descriptor tagged-sector inventory ($s={tag_text}$)"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=6, frameon=False)
    figure.tight_layout()
    return save(figure, output, "permutation_sector_inventory")


def artifact_manifest(output, sources, artifacts, systems):
    unique_sources = sorted({str(Path(path).resolve()) for path in sources})
    unique_artifacts = sorted({str(Path(path).resolve()) for path in artifacts})
    record = {
        "schema": "ye3t_mlearn_cost_comparison_publication_evidence_v2",
        "systems": list(systems),
        "outer_split": "published mlearn fixed train/test membership",
        "selection": "five group-stratified folds inside published training membership",
        "sources": [
            {"path": portable_path(path), "sha256": sha256(path)}
            for path in unique_sources
            if Path(path).is_file()
        ],
        "artifacts": [
            {
                "path": str(Path(path).relative_to(output)),
                "sha256": sha256(path),
            }
            for path in unique_artifacts
            if Path(path).is_file()
        ],
    }
    study.write_json(output / "results_manifest.json", record)


def write_publication_metadata(output, systems, benefits, visual_validation):
    force_values = [row["force_improvement_percent"] for row in benefits]
    timing = read_csv(output / "accuracy_runtime_pareto.csv")
    timing_lookup = {
        (row["system"], row["model"], int(row["mpi_ranks"])): float(
            row["median_microseconds_per_atom_step"]
        )
        for row in timing
    }
    runtime_ratios = []
    for system in systems:
        ace = timing_lookup[(system, "ace_127", 1)]
        tagged = timing_lookup[(system, "ye3t_tagged_127", 1)]
        runtime_ratios.append(tagged / ace)
    readme = f"""# Six-system linear ACE / tagged YE3T result

This directory is generated by `render_final_study.py` from frozen, hashed
workflow evidence for Li, Mo, Cu, Ni, Si, and Ge. Every PNG/PDF has a CSV
source table, and `checksums.sha256` covers the complete compact evidence set.

The matched 127-descriptor tagged model lowers held-out force RMSE on all six
systems. Improvements range from {min(force_values):.1f}% to
{max(force_values):.1f}% (mean {np.mean(force_values):.1f}%). This is an
accuracy result, not a runtime-speedup result: the current CPU tagged evaluator
is {min(runtime_ratios):.1f}–{max(runtime_ratios):.1f} times slower than the
matched ordinary ACE evaluator at one MPI rank.

All 18 promoted ACE127, tagged127, and augmented196 models pass LAMMPS
energy/force/virial parity, one-versus-four-rank MPI parity, finite differences,
Born stability, and 10,000-step NVE. The deliberately severe 0.75-lattice
compressed-cell aggregate passes for Ge only; Li, Mo, Cu, Ni, and Si retain a
recorded negative close-range qualification. Broader ZBL reference arms harmed
inner-fold accuracy and were not selected.

The ordinary controls are generated matched catalogues, not the best curated
PACE models in the cost-comparison paper. The retained curated 61-descriptor
PACE anchors expose that distinction. The lifted125 control uses a shifted-
Jacobi source, whereas the matched ordinary and tagged arms share their
ordinary backbone and cutoff. No higher-N adaptive branch was triggered because
the predeclared matched-count validation criterion was met.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    captions = """# Figure captions

1. **Accuracy versus descriptor count.** Five-fold training uncertainty is
   shown by bands; symbols are the fixed published held-out split. Curated PACE
   is a historical anchor and is not a count-matched generated catalogue.
2. **Accuracy–runtime Pareto.** Held-out RMSE versus isolated one-rank LAMMPS
   CPU time. This figure shows an accuracy gain but a current runtime deficit
   for tagged YE3T.
3. **Paired YE3T benefit.** Percentage RMSE change for tagged127 relative to
   ACE127 on the identical outer split; positive values favor tagged YE3T.
4. **Applied-property transfer.** Absolute error against the stated DFT/paper
   references for equilibrium lattice constant, bulk modulus, and cubic elastic
   constants. No virial targets were fitted.
5. **Permutation-sector inventory.** Active tagged coordinates only, grouped
   by tensor order. Training-inactive compiled coordinates are excluded.
6. **NVE stability.** Total-energy change during the promoted 10,000-step NVE
   trajectories after 2,000-step NVT equilibration at 0.8 of the reference
   melting temperature.
"""
    (output / "CAPTIONS.md").write_text(captions, encoding="utf-8")
    provenance = {
        "schema": "ye3t_cost_comparison_provenance_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "producer": portable_path(study.CODE_ROOT / "render_final_study.py"),
        "systems": list(systems),
        "repositories": {
            "ye3t": optional_git_state(study.PUBLIC.parents[3] / "ye3t"),
            "ye3t_ace": optional_git_state(study.PUBLIC.parents[2]),
            "ye3t_lammps": optional_git_state(
                study.PUBLIC.parents[3] / "ye3t-lammps"
            ),
        },
        "selection_policy": (
            "five group-stratified inner folds; fixed published outer test split"
        ),
        "adaptive_branch": "not_triggered",
    }
    study.write_json(output / "provenance.json", provenance)
    validation = {
        "schema": "ye3t_cost_comparison_visual_validation_v1",
        "visual_inspection_recorded": bool(visual_validation),
        "checks": {
            "labels_legible": bool(visual_validation),
            "legends_nonoverlapping": bool(visual_validation),
            "axes_and_units_present": bool(visual_validation),
            "negative_results_visible": bool(visual_validation),
            "source_data_present": True,
        },
        "note": (
            "Manual inspection of all six PNG figures at rendered resolution."
            if visual_validation
            else "Run with --record-visual-validation only after manual inspection."
        ),
    }
    study.write_json(output / "validation.json", validation)
    claims = {
        "schema": "ye3t_cost_comparison_claims_v1",
        "supported": [
            "tagged127 lowers held-out force RMSE relative to generated matched ACE127 on all six systems",
            "all promoted core models pass parity, MPI, finite-difference, Born, and NVE gates",
        ],
        "not_supported": [
            "tagged YE3T CPU runtime speedup over PACE",
            "universal extreme-compression stability",
            "superiority to every curated ACE/PACE catalogue",
            "multi-GPU performance portability",
        ],
    }
    study.write_json(output / "claims.json", claims)


def write_checksums(output):
    checksum_path = output / "checksums.sha256"
    paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS
    )
    parser.add_argument(
        "--output", type=Path, default=study.HERE / "publication_final"
    )
    parser.add_argument(
        "--record-visual-validation",
        action="store_true",
        help="Record that a human inspected every regenerated figure.",
    )
    args = parser.parse_args()
    systems = tuple(args.systems)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = []
    for system in systems:
        rows.extend(model_rows(system))
        root = study.system_root(system) / "finalist"
        catalogue_root = study.system_root(system) / "tagged_catalogue"
        if not catalogue_root.is_dir():
            catalogue_root = study.system_root(system) / "catalogue"
        sources.extend(
            (
                root / "test_metrics.csv",
                root / "fits" / "catalogues.json",
                root / "selection_frozen.json",
                root / "validation" / "timing" / "timings.csv",
                root / "validation" / "timing" / "summary.csv",
                root / "validation" / "properties" / "properties.csv",
                root / "validation" / "properties" / "eos.csv",
                root / "validation" / "properties" / "elastic_strains.csv",
                root / "validation" / "stability" / "summary.json",
                root / "validation" / "stability" / "dimer_scan.csv",
                root / "validation" / "stability" / "compressed_cells.csv",
                catalogue_root / "catalogue_manifest.json",
            )
        )
    write_csv(output / "accuracy_source_data.csv", rows)
    artifacts = [output / "accuracy_source_data.csv"]
    artifacts.extend(accuracy_figure(rows, systems, output, "energy"))
    artifacts.extend(accuracy_figure(rows, systems, output, "force"))
    benefits, benefit_artifacts = paired_benefit(rows, systems, output)
    artifacts.extend((output / "paired_ye3t_benefit.csv", *benefit_artifacts))
    timings = timing_rows(systems)
    artifacts.extend(pareto_figure(rows, timings, systems, output, "energy"))
    artifacts.extend(pareto_figure(rows, timings, systems, output, "force"))
    artifacts.extend(property_figure(systems, output))
    artifacts.extend(stability_figure(systems, output))
    artifacts.extend(sector_inventory(systems, output))
    for name in (
        "accuracy_runtime_pareto.csv",
        "applied_property_errors.csv",
        "stability_summary.csv",
        "stability_traces.csv",
        "permutation_sector_inventory.csv",
    ):
        path = output / name
        if path.is_file():
            artifacts.append(path)
    artifact_manifest(output, sources, artifacts, systems)
    write_publication_metadata(
        output, systems, benefits, args.record_visual_validation
    )
    write_checksums(output)
    print(
        json.dumps(
            {
                "systems": systems,
                "output": str(output),
                "artifacts": [str(path) for path in artifacts if Path(path).is_file()],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
