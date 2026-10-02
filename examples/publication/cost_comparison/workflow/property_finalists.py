#!/usr/bin/env python3
"""Run the frozen finalist EOS and elastic-property workflow through LAMMPS."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optimize_cached as study


DEFAULT_MODELS = ("ace_127", "ye3t_tagged_127", "ye3t_augmented_196")
PROPERTY_SCRIPT = Path(__file__).resolve().parent / "lammps_properties.py"


def run_system(system, lammps, model_ids, timeout):
    root = study.system_root(system) / "finalist"
    deploy = root / "deploy"
    matrix = deploy / "lammps_models.json"
    if not matrix.is_file():
        raise FileNotFoundError(matrix)
    available = {
        row["id"] for row in study.read_json(matrix)["models"]
    }
    unknown = set(model_ids) - available
    if unknown:
        raise ValueError(f"Unknown property model IDs for {system}: {sorted(unknown)}")
    output = root / "validation" / "properties"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROPERTY_SCRIPT),
        "--lammps",
        str(lammps),
        "--system",
        str(study.system_config_path(system)),
        "--models",
        str(matrix),
        "--model-directory",
        str(deploy),
        "--output",
        str(output),
    ]
    for model_id in model_ids:
        command.extend(("--model-id", model_id))
    screen = output / "screen.properties.log"
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    started = time.perf_counter()
    with screen.open("w", encoding="utf-8") as handle:
        handle.write("command: " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=deploy,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"{system} property workflow failed with code {completed.returncode}; "
            f"see {screen}."
        )
    summary = study.read_json(output / "summary.json")
    record = {
        "system": system,
        "model_ids": list(model_ids),
        "elapsed_seconds": elapsed,
        "born_stable": {
            row["model"]: bool(row["born_stable"])
            for row in summary["properties"]
        },
        "summary": str(output / "summary.json"),
    }
    study.append_progress("finalist_properties_complete", **record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS
    )
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    lammps = args.lammps.resolve()
    if not lammps.is_file():
        raise FileNotFoundError(lammps)
    model_ids = tuple(args.model) if args.model else DEFAULT_MODELS
    for system in args.systems:
        run_system(system, lammps, model_ids, args.timeout_seconds)


if __name__ == "__main__":
    main()
