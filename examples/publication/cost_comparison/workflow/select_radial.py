#!/usr/bin/env python3
"""Freeze one paired radial realization per system from training-only folds."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import optimize_cached as study


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_value(row, model, field):
    value = float(row["metrics"][model][field])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"Invalid {model} {field} for candidate {row['candidate']}.")
    return value


def select_system(system, tolerance):
    root = study.system_root(system) / "radial_screen"
    energy_path = root / "energy_summary.json"
    force_path = root / "force_summary.json"
    force = study.read_json(force_path)
    rows = sorted(force["results"], key=lambda row: int(row["candidate"]))
    candidate_count = int(
        study.read_json(study.CONFIG_PATH)["basis"]["radial"]
        ["optimization"]["candidate_count"]
    )
    if [int(row["candidate"]) for row in rows] != list(range(candidate_count)):
        raise RuntimeError(
            f"{system} does not have all {candidate_count} force-screen candidates."
        )
    fold_identities = {
        json.dumps(row["fold_identity"], sort_keys=True) for row in rows
    }
    if len(fold_identities) != 1:
        raise RuntimeError(f"{system} radial candidates do not use identical folds.")
    models = ("ace_127", "ye3t_tagged_127")
    fields = (
        "energy_rmse_mean_eV_per_atom",
        "force_rmse_mean_eV_per_A",
    )
    minima = {
        (model, field): min(metric_value(row, model, field) for row in rows)
        for model in models
        for field in fields
    }
    scored = []
    for row in rows:
        ratios = {
            f"{model}.{field}": metric_value(row, model, field)
            / max(minima[(model, field)], np.finfo(np.float64).tiny)
            for model in models
            for field in fields
        }
        scored.append(
            {
                "candidate": int(row["candidate"]),
                "cutoff_A": float(row["cutoff_A"]),
                "radial_lambda": float(row["radial_lambda"]),
                "cutoff_width_A": float(row["cutoff_width_A"]),
                "paired_normalized_score": float(np.mean(tuple(ratios.values()))),
                "normalized_metric_ratios": ratios,
                "metrics": row["metrics"],
            }
        )
    best_score = min(row["paired_normalized_score"] for row in scored)
    near_best = [
        row
        for row in scored
        if row["paired_normalized_score"] <= best_score * (1.0 + tolerance)
    ]
    selected = min(
        near_best,
        key=lambda row: (
            row["cutoff_A"],
            row["paired_normalized_score"],
            row["candidate"],
        ),
    )
    per_model = {}
    for model in models:
        per_model[model] = min(
            scored,
            key=lambda row: (
                metric_value(row, model, "force_rmse_mean_eV_per_A")
                / max(minima[(model, "force_rmse_mean_eV_per_A")], np.finfo(np.float64).tiny)
                + metric_value(row, model, "energy_rmse_mean_eV_per_atom")
                / max(minima[(model, "energy_rmse_mean_eV_per_atom")], np.finfo(np.float64).tiny),
                row["candidate"],
            ),
        )["candidate"]
    record = {
        "schema": "ye3t_mlearn_paired_radial_selection_v1",
        "system": system,
        "selection_data": "published_training_split_only",
        "test_split_opened": False,
        "candidate_count": len(scored),
        "shared_radial_for_ace_and_tagged": True,
        "objective": "mean of ACE/tagged energy/force RMSE ratios to per-metric minima",
        "near_best_relative_tolerance": float(tolerance),
        "tie_break": "smallest_cutoff_then_score_then_candidate_id",
        "fold_identity": rows[0]["fold_identity"],
        "selected": selected,
        "per_model_diagnostic_winner": per_model,
        "candidates": scored,
        "source_files": {
            "force_summary": str(force_path),
            "force_summary_sha256": file_sha256(force_path),
        },
    }
    if energy_path.is_file():
        record["source_files"]["energy_diagnostic_summary"] = str(energy_path)
        record["source_files"]["energy_diagnostic_summary_sha256"] = file_sha256(
            energy_path
        )
    output = root / "selection_frozen.json"
    study.write_json(output, record)
    study.append_progress(
        "radial_selection_frozen",
        system=system,
        selected_candidate=selected["candidate"],
        cutoff_A=selected["cutoff_A"],
        radial_lambda=selected["radial_lambda"],
        selection_path=str(output),
    )
    print(json.dumps(record["selected"], sort_keys=True), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS)
    configured = study.read_json(study.CONFIG_PATH)["basis"]["radial"][
        "optimization"
    ]
    parser.add_argument(
        "--near-best-relative-tolerance",
        type=float,
        default=float(configured["near_best_relative_tolerance"]),
    )
    args = parser.parse_args()
    if args.near_best_relative_tolerance < 0.0:
        raise ValueError("The near-best relative tolerance must be nonnegative.")
    for system in args.systems:
        select_system(system, args.near_best_relative_tolerance)


if __name__ == "__main__":
    main()
