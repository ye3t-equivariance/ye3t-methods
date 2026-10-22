#!/usr/bin/env python3
"""Run isolated one- and four-rank CPU timings for finalized models."""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np

import optimize_cached as study


LOOP = re.compile(
    r"Loop time of ([0-9.eE+-]+) on ([0-9]+) procs for ([0-9]+) steps with ([0-9]+) atoms"
)
AUTO_CONFIGURATION = re.compile(
    r"YE3T configuration:.*block_policy auto, plan ([^,\s]+)"
)
AUTO_SELECTION = re.compile(
    r": ([0-9]+) selected block routes and ([0-9]+) selected coupled-product "
    r"DAGs from ([0-9]+) compiled candidate routes"
)
AUTO_SCALAR = re.compile(
    r"([0-9]+) scalar-power routes, ([0-9]+) symmetric-power blocks"
)
TAGGED_AUTO = re.compile(
    r"YE3T tagged AUTO calibration: centers ([0-9]+), neighbors/center "
    r"([0-9]+), repeats ([0-9]+), direct ([0-9.eE+-]+) s, generic_dag "
    r"([0-9.eE+-]+) s, symmetric_power ([0-9.eE+-]+) s, block "
    r"([0-9.eE+-]+) s; selected ([a-z_]+); calibration "
    r"([0-9.eE+-]+) s"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auto_resolution(stdout, required, require_non_direct=False):
    result = {
        "auto_plan_required": bool(required),
        "auto_plan_path": "",
        "auto_compiled_candidate_routes": 0,
        "auto_selected_block_routes": 0,
        "auto_selected_coupled_product_dags": 0,
        "auto_selected_scalar_power_routes": 0,
        "auto_selected_non_direct_routes": 0,
        "auto_resolution_status": "not_applicable",
    }
    if not required:
        return result
    configuration = AUTO_CONFIGURATION.findall(stdout)
    if len(configuration) != 1 or configuration[0] == "none":
        raise RuntimeError("YE3T AUTO did not load exactly one compiled plan.")
    forbidden = (
        "budget_direct_fallback",
        "conservative_direct_fallback",
        "selected_direct_no_authorized_profile",
    )
    if any(value in stdout for value in forbidden):
        raise RuntimeError("YE3T AUTO used a direct fallback.")
    selections = AUTO_SELECTION.findall(stdout)
    scalars = AUTO_SCALAR.findall(stdout)
    if not selections or len(selections) != len(scalars):
        raise RuntimeError("YE3T AUTO did not report a complete resolved portfolio.")
    selected_block = sum(int(row[0]) for row in selections)
    selected_coupled = sum(int(row[1]) for row in selections)
    compiled = sum(int(row[2]) for row in selections)
    selected_scalar = sum(int(row[0]) for row in scalars)
    selected_non_direct = selected_block + selected_coupled + selected_scalar
    if compiled <= 0:
        raise RuntimeError("YE3T AUTO plan has no compiled optimized candidates.")
    if selected_non_direct <= 0 and require_non_direct:
        raise RuntimeError("YE3T AUTO resolved to an all-direct schedule.")
    result.update(
        {
            "auto_plan_path": configuration[0],
            "auto_compiled_candidate_routes": compiled,
            "auto_selected_block_routes": selected_block,
            "auto_selected_coupled_product_dags": selected_coupled,
            "auto_selected_scalar_power_routes": selected_scalar,
            "auto_selected_non_direct_routes": selected_non_direct,
            "auto_resolution_status": (
                "resolved_non_direct"
                if selected_non_direct > 0
                else "resolved_all_direct"
            ),
        }
    )
    return result


def summarized_auto_resolution(rows, required):
    fields = (
        "auto_plan_path",
        "auto_compiled_candidate_routes",
        "auto_selected_block_routes",
        "auto_selected_coupled_product_dags",
        "auto_selected_scalar_power_routes",
        "auto_selected_non_direct_routes",
        "auto_resolution_status",
    )
    if not required:
        return {
            "auto_plan_path": "",
            "auto_compiled_candidate_routes": 0,
            "auto_selected_block_routes": 0,
            "auto_selected_coupled_product_dags": 0,
            "auto_selected_scalar_power_routes": 0,
            "auto_selected_non_direct_routes": 0,
            "auto_resolution_status": "not_applicable",
        }
    summary = {}
    for field in fields:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise RuntimeError(
                f"YE3T AUTO resolution changed across repetitions for {field}: "
                f"{sorted(values, key=str)}"
            )
        summary[field] = values.pop()
    return summary


def tagged_auto_resolution(stdout, required):
    result = {
        "tagged_auto_required": bool(required),
        "tagged_auto_selected_evaluator": "",
        "tagged_auto_centers": 0,
        "tagged_auto_neighbors_per_center": 0,
        "tagged_auto_repeats": 0,
        "tagged_auto_direct_seconds": 0.0,
        "tagged_auto_generic_dag_seconds": 0.0,
        "tagged_auto_symmetric_power_seconds": 0.0,
        "tagged_auto_block_seconds": 0.0,
        "tagged_auto_calibration_seconds": 0.0,
    }
    if not required:
        return result
    matches = TAGGED_AUTO.findall(stdout)
    if len(matches) != 1:
        raise RuntimeError(
            "Tagged YE3T AUTO did not report exactly one complete calibration."
        )
    row = matches[0]
    selected = row[7]
    if selected not in {
        "compiled_direct",
        "generic_dag",
        "symmetric_power",
        "block",
    }:
        raise RuntimeError(f"Tagged YE3T AUTO selected invalid evaluator {selected}.")
    result.update(
        {
            "tagged_auto_selected_evaluator": selected,
            "tagged_auto_centers": int(row[0]),
            "tagged_auto_neighbors_per_center": int(row[1]),
            "tagged_auto_repeats": int(row[2]),
            "tagged_auto_direct_seconds": float(row[3]),
            "tagged_auto_generic_dag_seconds": float(row[4]),
            "tagged_auto_symmetric_power_seconds": float(row[5]),
            "tagged_auto_block_seconds": float(row[6]),
            "tagged_auto_calibration_seconds": float(row[8]),
        }
    )
    return result


def summarized_tagged_auto_resolution(rows, required):
    if not required:
        return tagged_auto_resolution("", False)
    identity_fields = (
        "tagged_auto_selected_evaluator",
        "tagged_auto_centers",
        "tagged_auto_neighbors_per_center",
        "tagged_auto_repeats",
    )
    result = {"tagged_auto_required": True}
    for field in identity_fields:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise RuntimeError(
                f"Tagged YE3T AUTO changed across repetitions for {field}: "
                f"{sorted(values, key=str)}"
            )
        result[field] = values.pop()
    timing_fields = (
        "tagged_auto_direct_seconds",
        "tagged_auto_generic_dag_seconds",
        "tagged_auto_symmetric_power_seconds",
        "tagged_auto_block_seconds",
        "tagged_auto_calibration_seconds",
    )
    for field in timing_fields:
        result[field] = float(np.median([row[field] for row in rows]))
    return result


def benchmark_deck(system_config, record, cells, warmup, steps):
    pair_coefficients = "\n".join(record["pair_coefficients"])
    return f"""# Fixed-position cost benchmark for {record['id']}.
units metal
atom_style atomic
boundary p p p
atom_modify map yes sort 0 0.0
newton on
lattice {system_config['crystal']} {float(system_config['initial_lattice_constant_A']):.17g}
region cell block 0 {cells} 0 {cells} 0 {cells} units lattice
create_box 1 cell
create_atoms 1 box
mass 1 {float(system_config['atomic_masses'][0]):.17g}
reset_atoms id sort yes
group displaced id 1
displace_atoms displaced move 0.08 -0.05 0.04 units box
neighbor 0.3 bin
neigh_modify every 1 delay 0 check no once yes
{record['pair_style']}
{pair_coefficients}
thermo {max(warmup, steps) + 1}
run {warmup} post no
timer full sync
run {steps} pre no post no
"""


def run_once(
    command,
    cwd,
    log,
    timeout,
    auto_required=False,
    auto_non_direct_required=False,
    tagged_auto_required=False,
):
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    Path(log).write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Timing command failed; see {log}.")
    matches = LOOP.findall(completed.stdout)
    if not matches:
        raise RuntimeError(f"No LAMMPS loop record in {log}.")
    seconds, ranks, steps, atoms = matches[-1]
    return {
        "loop_seconds": float(seconds),
        "mpi_ranks": int(ranks),
        "steps": int(steps),
        "atoms": int(atoms),
        "wall_seconds": time.perf_counter() - started,
        **auto_resolution(
            completed.stdout,
            auto_required,
            require_non_direct=auto_non_direct_required,
        ),
        **tagged_auto_resolution(completed.stdout, tagged_auto_required),
    }


def interleaved_records(records, repetition):
    records = tuple(records)
    if not records:
        return ()
    offset = int(repetition) % len(records)
    return records[offset:] + records[:offset]


def benchmark_system(
    system,
    lammps,
    mpiexec,
    ranks,
    cells,
    warmup,
    steps,
    repetitions,
    selected,
    model_matrix,
    output_label,
    deploy_override,
):
    root = (
        deploy_override.parent
        if deploy_override is not None
        else study.system_root(system) / "finalist"
    )
    deploy = deploy_override if deploy_override is not None else root / "deploy"
    matrix = study.read_json(deploy / model_matrix)
    records = [row for row in matrix["models"] if not selected or row["id"] in selected]
    unknown = selected - {row["id"] for row in records}
    if unknown:
        raise ValueError(f"Unknown benchmark model IDs for {system}: {sorted(unknown)}")
    system_config = study.read_system_config(system)
    output = root / "validation" / output_label
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    load_before = tuple(float(value) for value in os.getloadavg())
    input_paths = {}
    for record in records:
        input_path = output / f"in.{record['id']}.timing"
        input_path.write_text(
            benchmark_deck(system_config, record, cells, warmup, steps),
            encoding="utf-8",
        )
        input_paths[record["id"]] = input_path
    for rank_count in ranks:
        for repetition in range(repetitions):
            for order_index, record in enumerate(
                interleaved_records(records, repetition)
            ):
                input_path = input_paths[record["id"]]
                if rank_count == 1:
                    command = [lammps, "-log", "none", "-in", input_path]
                else:
                    command = [
                        mpiexec,
                        "-n",
                        str(rank_count),
                        lammps,
                        "-log",
                        "none",
                        "-in",
                        input_path,
                    ]
                result = run_once(
                    command,
                    deploy,
                    output / f"screen.{record['id']}.r{rank_count}.rep{repetition}.log",
                    1800,
                    auto_required=bool(record.get("auto_plan_required", False)),
                    auto_non_direct_required=bool(
                        record.get("auto_non_direct_required", False)
                    ),
                    tagged_auto_required=bool(
                        record.get("tagged_auto_required", False)
                    ),
                )
                row = {
                    "system": system,
                    "model": record["id"],
                    "family": record["family"],
                    "accuracy_model": record.get("accuracy_model", record["id"]),
                    "descriptor_count": int(record["descriptor_count"]),
                    "ordinary_descriptor_count": int(record["ordinary_descriptor_count"]),
                    "tagged_descriptor_count": int(record["tagged_descriptor_count"]),
                    "repetition": repetition,
                    "execution_order_index": order_index,
                    **result,
                }
                row["microseconds_per_atom_step"] = (
                    row["loop_seconds"] * 1.0e6 / (row["atoms"] * row["steps"])
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for record in records:
        for rank_count in ranks:
            model_rows = [
                row
                for row in rows
                if row["model"] == record["id"]
                and row["mpi_ranks"] == rank_count
            ]
            values = np.asarray(
                [row["microseconds_per_atom_step"] for row in model_rows],
                dtype=np.float64,
            )
            auto_summary = summarized_auto_resolution(
                model_rows,
                bool(record.get("auto_plan_required", False)),
            )
            tagged_auto_summary = summarized_tagged_auto_resolution(
                model_rows,
                bool(record.get("tagged_auto_required", False)),
            )
            summary_rows.append(
                {
                    "system": system,
                    "model": record["id"],
                    "family": record["family"],
                    "descriptor_count": int(record["descriptor_count"]),
                    "mpi_ranks": rank_count,
                    "median_microseconds_per_atom_step": float(np.median(values)),
                    "minimum_microseconds_per_atom_step": float(np.min(values)),
                    "maximum_microseconds_per_atom_step": float(np.max(values)),
                    "repetitions": len(values),
                    "accuracy_model": record.get("accuracy_model", record["id"]),
                    "auto_plan_required": bool(
                        record.get("auto_plan_required", False)
                    ),
                    **auto_summary,
                    **tagged_auto_summary,
                }
            )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    report = {
        "schema": "ye3t_mlearn_cpu_timing_v1",
        "system": system,
        "lammps_executable": str(Path(lammps).resolve()),
        "lammps_executable_sha256": file_sha256(lammps),
        "mpiexec_executable": str(Path(mpiexec).resolve()),
        "mpiexec_executable_sha256": file_sha256(mpiexec),
        "model_matrix": str((deploy / model_matrix).resolve()),
        "model_matrix_sha256": file_sha256(deploy / model_matrix),
        "host": platform.node(),
        "processor": platform.processor(),
        "python_platform": platform.platform(),
        "omp_threads": 1,
        "mpi_ranks": list(ranks),
        "cells_per_axis": cells,
        "warmup_steps": warmup,
        "timed_steps": steps,
        "repetitions": repetitions,
        "fixed_positions": True,
        "neighbor_list_once": True,
        "cache_state": "model_loaded_and_warmup_complete_before_timed_loop",
        "interleaving_policy": "cyclic_model_rotation_within_rank_and_repetition",
        "load_average_before": load_before,
        "load_average_after": tuple(float(value) for value in os.getloadavg()),
        "summary": summary_rows,
    }
    study.write_json(output / "summary.json", report)
    study.append_progress(
        "finalist_cpu_timing_complete",
        system=system,
        model_count=len(records),
        summary=str(output / "summary.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=(1, 4))
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--model-matrix", default="lammps_models.json")
    parser.add_argument("--output-label", default="timing")
    parser.add_argument("--deploy-root", type=Path)
    args = parser.parse_args()
    if any(value <= 0 for value in args.ranks):
        raise ValueError("MPI rank counts must be positive.")
    if args.repetitions < 3:
        raise ValueError("Publication timings require at least three repetitions.")
    lammps = args.lammps.resolve()
    mpiexec = args.mpiexec.resolve()
    if not lammps.is_file() or not mpiexec.is_file():
        raise FileNotFoundError("LAMMPS and mpiexec must both exist.")
    deploy_override = None
    if args.deploy_root is not None:
        if len(args.systems) != 1:
            raise ValueError("--deploy-root requires exactly one selected system.")
        deploy_override = args.deploy_root.resolve()
        if not deploy_override.is_dir():
            raise FileNotFoundError(f"Deploy root does not exist: {deploy_override}")
    selected = set(args.model)
    for system in args.systems:
        benchmark_system(
            system,
            lammps,
            mpiexec,
            tuple(args.ranks),
            args.cells,
            args.warmup,
            args.steps,
            args.repetitions,
            selected,
            args.model_matrix,
            args.output_label,
            deploy_override,
        )


if __name__ == "__main__":
    main()
