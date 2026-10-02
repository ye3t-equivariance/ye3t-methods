#!/usr/bin/env python3
"""Validate PACE-product and YE3T-symmetric AUTO on identical YACE models."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import benchmark_finalists as benchmark
import finalize_models as final
import optimize_cached as study
import validate_finalists as validation


SYSTEMS = ("Li", "Mo", "Cu", "Ni", "Si", "Ge")
ENERGY_TOLERANCE_EV = 1.0e-8
ATOM_ENERGY_TOLERANCE_EV = 1.0e-10
FORCE_TOLERANCE_EV_PER_A = 1.0e-8
ATOM_VIRIAL_TOLERANCE_EV = 1.0e-10


def compare_observables(system, descriptor_count, pace, symmetric, auto):
    pace_energy = np.asarray(pace["c_atom_energy"], dtype=np.float64)
    symmetric_energy = np.asarray(symmetric["c_atom_energy"], dtype=np.float64)
    pace_force = np.column_stack((pace["fx"], pace["fy"], pace["fz"]))
    symmetric_force = np.column_stack(
        (symmetric["fx"], symmetric["fy"], symmetric["fz"])
    )
    pace_virial = np.column_stack(
        tuple(pace[f"c_atom_stress[{axis}]"] for axis in range(1, 7))
    )
    symmetric_virial = np.column_stack(
        tuple(symmetric[f"c_atom_stress[{axis}]"] for axis in range(1, 7))
    )
    atom_virial_tolerance = (
        ATOM_VIRIAL_TOLERANCE_EV / validation.BAR_A3_TO_EV
    )
    row = {
        "system": system,
        "descriptor_count": int(descriptor_count),
        "total_energy_abs_error_eV": float(
            abs(np.sum(pace_energy) - np.sum(symmetric_energy))
        ),
        "per_atom_energy_max_abs_error_eV": float(
            np.max(np.abs(pace_energy - symmetric_energy))
        ),
        "force_max_abs_error_eV_per_A": float(
            np.max(np.abs(pace_force - symmetric_force))
        ),
        "per_atom_virial_max_abs_error_bar_A3": float(
            np.max(np.abs(pace_virial - symmetric_virial))
        ),
        "global_virial_max_abs_error_bar_A3": float(
            np.max(np.abs(np.sum(pace_virial, axis=0) - np.sum(symmetric_virial, axis=0)))
        ),
        "auto_plan_path": auto["auto_plan_path"],
        "auto_compiled_candidate_routes": auto["auto_compiled_candidate_routes"],
        "auto_selected_non_direct_routes": auto["auto_selected_non_direct_routes"],
        "auto_resolution_status": auto["auto_resolution_status"],
        "energy_tolerance_eV": ENERGY_TOLERANCE_EV,
        "per_atom_energy_tolerance_eV": ATOM_ENERGY_TOLERANCE_EV,
        "force_tolerance_eV_per_A": FORCE_TOLERANCE_EV_PER_A,
        "per_atom_virial_tolerance_bar_A3": atom_virial_tolerance,
    }
    row["passed"] = bool(
        row["total_energy_abs_error_eV"] <= ENERGY_TOLERANCE_EV
        and row["per_atom_energy_max_abs_error_eV"]
        <= ATOM_ENERGY_TOLERANCE_EV
        and row["force_max_abs_error_eV_per_A"] <= FORCE_TOLERANCE_EV_PER_A
        and row["per_atom_virial_max_abs_error_bar_A3"]
        <= atom_virial_tolerance
        and row["global_virial_max_abs_error_bar_A3"]
        <= atom_virial_tolerance * len(pace_energy)
    )
    return row


def validate_system(system, lammps, output_label, counts):
    root = study.system_root(system) / "finalist"
    deploy = root / "deploy"
    matrix_path = deploy / "lammps_models.three_way_auto.json"
    matrix = study.read_json(matrix_path)
    records = {row["id"]: row for row in matrix["models"]}
    output = root / "validation" / output_label
    output.mkdir(parents=True, exist_ok=True)
    system_config = study.read_system_config(system)
    rows = []
    for count in counts:
        pace_id = f"ace_{count}"
        symmetric_id = f"ye3t_symmetric_{count}"
        pace = records[pace_id]
        symmetric = records[symmetric_id]
        if "pace product" not in pace["pair_style"]:
            raise RuntimeError(f"{system}/{pace_id} does not request product.")
        if "block_policy auto" not in symmetric["pair_style"]:
            raise RuntimeError(f"{system}/{symmetric_id} does not request AUTO.")
        if pace["pair_coefficients"][0].replace(" pace ", " ye3t ") != symmetric[
            "pair_coefficients"
        ][0]:
            raise RuntimeError(
                f"{system}/{count} controls do not reference the same YACE path."
            )

        observed = {}
        screens = {}
        for record in (pace, symmetric):
            name = record["id"]
            input_path = output / f"in.{system.lower()}_{name}.parity"
            input_path.write_text(
                final.input_deck(system_config, record), encoding="utf-8"
            )
            dump_path = output / f"dump.{name}"
            screen = output / f"screen.{name}.log"
            validation.run(
                [
                    lammps,
                    "-var",
                    "dump_path",
                    dump_path,
                    "-log",
                    "none",
                    "-in",
                    input_path,
                ],
                deploy,
                screen,
                300,
            )
            _atoms, observed[name] = validation.read_dump(dump_path, system)
            screens[name] = screen
        auto = benchmark.auto_resolution(
            screens[symmetric_id].read_text(encoding="utf-8"),
            True,
        )
        row = compare_observables(
            system,
            count,
            observed[pace_id],
            observed[symmetric_id],
            auto,
        )
        yace_path = deploy / pace["model_directory"] / "potential.yace"
        row["source_yace_sha256"] = validation.sha256(yace_path)
        row["model_matrix_sha256"] = validation.sha256(matrix_path)
        study.write_json(output / f"summary.{count}.json", row)
        if count == 127:
            study.write_json(output / "summary.json", row)
        if not row["passed"]:
            raise RuntimeError(
                f"{system}/{count} PACE/YE3T symmetric parity failed."
            )
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
    study.write_json(
        output / "curve_summary.json",
        {
            "schema": "ye3t_three_way_symmetric_curve_parity_v1",
            "system": system,
            "rows": rows,
        },
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument("--counts", nargs="+", type=int, default=(127,))
    parser.add_argument(
        "--output-label",
        default="three_way_ordinary_parity_20260921",
    )
    args = parser.parse_args()
    lammps = args.lammps.resolve()
    if not lammps.is_file():
        raise FileNotFoundError(lammps)
    counts = tuple(dict.fromkeys(int(value) for value in args.counts))
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("Descriptor counts must be positive.")
    rows = []
    for system in args.systems:
        rows.extend(validate_system(system, lammps, args.output_label, counts))
    destination = study.HERE / "results" / "three_way_pace_ye3t_20260921"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "symmetric_parity_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
