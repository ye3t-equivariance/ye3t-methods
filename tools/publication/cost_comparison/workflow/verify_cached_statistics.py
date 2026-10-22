#!/usr/bin/env python3
"""Verify cached-statistics training metrics against direct cached-row sums."""

import argparse
import hashlib
import json

import numpy as np

import optimize_cached as study
from ye3t_methods.atomistic.ace.linear_ace import load_xyz_structures


def direct_score(system, model_name):
    root = study.system_root(system)
    result = study.read_json(
        root / "optimized_fixed_catalogues" / (model_name + ".json")
    )
    with np.load(
        root / "optimized_fixed_catalogues" / (model_name + ".npz"),
        allow_pickle=False,
    ) as data:
        coefficients = np.asarray(data["runtime_coefficients"], dtype=np.float64)
        indices = np.asarray(data["feature_indices"], dtype=np.int64)
    prefix = system.lower()
    structures = load_xyz_structures(
        study.data_root(system) / f"{prefix}_training.xyz"
    )
    energy_error = []
    force_error = []
    row_hasher = hashlib.sha256()
    for index, atoms in enumerate(structures):
        row = study.combined_row(system, index, row_hasher)
        energy, forces = study.target(atoms)
        feature_sum = row["feature_sums"][indices]
        force_design = row["force_design"][:, indices]
        prediction = coefficients[0] + feature_sum @ coefficients[1:] / len(atoms)
        energy_error.append(prediction - energy / len(atoms))
        force_error.append(
            force_design @ coefficients[1:] - np.asarray(forces).reshape(-1)
        )
    energy_error = np.asarray(energy_error)
    force_error = np.concatenate(force_error)
    direct = {
        "energy_sse": float(energy_error @ energy_error),
        "force_sse": float(force_error @ force_error),
        "energy_rmse_eV_per_atom": float(
            np.sqrt(np.mean(energy_error * energy_error))
        ),
        "force_rmse_eV_per_A": float(
            np.sqrt(np.mean(force_error * force_error))
        ),
    }
    _index, _cache, parent_shards = study.load_statistics(system)
    selected_shards = study.model_shards(parent_shards, indices)
    cached = study.score_linear_statistics(
        study.sum_linear_statistics(selected_shards.values()), coefficients
    )
    residual = {
        name: abs(float(direct[name]) - float(cached[name]))
        for name in ("energy_rmse_eV_per_atom", "force_rmse_eV_per_A")
    }
    sse_residual = {
        "energy": abs(direct["energy_sse"] - cached["energy_sse"]),
        "force": abs(direct["force_sse"] - cached["force_sse"]),
    }
    roundoff_bound = {
        "energy": cached["energy_sse_roundoff_bound"],
        "force": cached["force_sse_roundoff_bound"],
    }
    report = {
        "schema": "ye3t_cached_statistics_direct_audit_v1",
        "system": system,
        "model": model_name,
        "direct": direct,
        "cached": cached,
        "absolute_residual": residual,
        "sse_absolute_residual": sse_residual,
        "sse_roundoff_bound": roundoff_bound,
        "passed": all(
            sse_residual[name] <= roundoff_bound[name]
            for name in sse_residual
        ),
    }
    path = root / "optimized_fixed_catalogues" / (model_name + "_direct_audit.json")
    study.write_json(path, report)
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS)
    parser.add_argument("--model", default="ye3t_tagged_127")
    args = parser.parse_args()
    for system in args.systems:
        direct_score(system, args.model)


if __name__ == "__main__":
    main()
