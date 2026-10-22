#!/usr/bin/env python3
"""Run close-range and elevated-temperature stability gates in LAMMPS."""

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

import optimize_cached as study


DIMER_RESULT = re.compile(
    r"YE3T_DIMER energy_eV=([0-9.eE+-]+) force1x_eV_per_A=([0-9.eE+-]+)"
)
COMPRESSED_RESULT = re.compile(
    r"YE3T_COMPRESSED energy_eV_per_atom=([0-9.eE+-]+) pressure_bar=([0-9.eE+-]+)"
)
DEFAULT_MODELS = ("ace_127", "ye3t_tagged_127", "ye3t_augmented_196")


def run(command, cwd, screen, timeout):
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    with Path(screen).open("w", encoding="utf-8") as handle:
        handle.write("command: " + " ".join(str(value) for value in command) + "\n")
        handle.flush()
        completed = subprocess.run(
            [str(value) for value in command],
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed with code {completed.returncode}; see {screen}.")


def selected_records(system, model_names):
    deploy = study.system_root(system) / "finalist" / "deploy"
    matrix = study.read_json(deploy / "lammps_models.json")
    records = [row for row in matrix["models"] if row["id"] in model_names]
    unknown = set(model_names) - {row["id"] for row in records}
    if unknown:
        raise ValueError(f"Unknown stability model IDs for {system}: {sorted(unknown)}")
    return deploy, records


def dimer_deck(system, mass, record, distance):
    pair_coefficients = "\n".join(record["pair_coefficients"])
    left = 5.0 - 0.5 * distance
    right = 5.0 + 0.5 * distance
    return f"""units metal
atom_style atomic
boundary f f f
atom_modify map yes sort 0 0.0
newton on
region cell block 0 10 0 10 0 10 units box
create_box 1 cell
create_atoms 1 single {left:.17g} 5 5 units box
create_atoms 1 single {right:.17g} 5 5 units box
mass 1 {mass:.17g}
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
{record['pair_style']}
{pair_coefficients}
variable f1 equal fx[1]
run 0 post no
print "YE3T_DIMER energy_eV=$(pe:%.17g) force1x_eV_per_A=$(v_f1:%.17g)"
"""


def compressed_deck(system_config, record, scale):
    pair_coefficients = "\n".join(record["pair_coefficients"])
    lattice = float(system_config["initial_lattice_constant_A"]) * scale
    return f"""units metal
atom_style atomic
boundary p p p
atom_modify map yes sort 0 0.0
newton on
lattice {system_config['crystal']} {lattice:.17g}
region cell block 0 2 0 2 0 2 units lattice
create_box 1 cell
create_atoms 1 box
mass 1 {float(system_config['atomic_masses'][0]):.17g}
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
{record['pair_style']}
{pair_coefficients}
compute pair_pressure all pressure NULL pair
variable n equal count(all)
variable e equal pe/v_n
run 0 post no
print "YE3T_COMPRESSED energy_eV_per_atom=$(v_e:%.17g) pressure_bar=$(c_pair_pressure:%.17g)"
"""


def nve_deck(system_config, record, trajectory, temperature, seed):
    pair_coefficients = "\n".join(record["pair_coefficients"])
    return f"""# 2000-step NVT preparation plus 10000-step NVE stability gate.
units metal
atom_style atomic
boundary p p p
atom_modify map yes sort 0 0.0
newton on
lattice {system_config['crystal']} {float(system_config['initial_lattice_constant_A']):.17g}
region cell block 0 2 0 2 0 2 units lattice
create_box 1 cell
create_atoms 1 box
mass 1 {float(system_config['atomic_masses'][0]):.17g}
reset_atoms id sort yes
displace_atoms all random 0.01 0.01 0.01 {seed} units box
velocity all create {temperature:.17g} {seed + 7919} mom yes rot no dist gaussian
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
{record['pair_style']}
{pair_coefficients}
timestep 0.0005
thermo 100
thermo_style custom step temp pe ke etotal press
thermo_modify format float %.17g
fix thermostat all nvt temp {temperature:.17g} {temperature:.17g} 0.05
run 2000
unfix thermostat
reset_timestep 0
fix integrate all nve
dump trajectory all custom 100 {trajectory} id type x y z vx vy vz fx fy fz
dump_modify trajectory sort id
print "YE3T_NVE_BEGIN"
run 10000
print "YE3T_NVE_END"
"""


def parse_last_thermo_block(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if line.split() == ["Step", "Temp", "PotEng", "KinEng", "TotEng", "Press"]
    ]
    if not starts:
        raise ValueError(f"No NVE thermo block found in {path}.")
    rows = []
    for line in lines[starts[-1] + 1 :]:
        values = line.split()
        if len(values) != 6:
            if rows:
                break
            continue
        try:
            rows.append([float(value) for value in values])
        except ValueError:
            if rows:
                break
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 101:
        raise ValueError(f"Expected 101 NVE thermo samples, found {array.shape}.")
    return array


def minimum_distances(trajectory):
    frames = read(trajectory, index=":", format="lammps-dump-text")
    values = []
    for atoms in frames:
        cutoff = 0.5 * float(np.min(atoms.cell.lengths()))
        distances = neighbor_list("d", atoms, cutoff)
        values.append(float(np.min(distances)) if len(distances) else np.inf)
    return np.asarray(values, dtype=np.float64)


def nve_metrics(log, trajectory, atoms, timestep_ps, target_temperature):
    thermo = parse_last_thermo_block(log)
    distances = minimum_distances(trajectory)
    step = thermo[:, 0]
    total_per_atom = thermo[:, 4] / atoms
    time_ps = step * timestep_ps
    slope = float(np.polyfit(time_ps, total_per_atom, 1)[0])
    excursion = float(np.max(np.abs(total_per_atom - total_per_atom[0])))
    rows = [
        {
            "step": int(values[0]),
            "time_ps": float(values[0] * timestep_ps),
            "temperature_K": float(values[1]),
            "potential_energy_eV": float(values[2]),
            "kinetic_energy_eV": float(values[3]),
            "total_energy_eV": float(values[4]),
            "pressure_bar": float(values[5]),
            "minimum_distance_A": float(distances[index]),
        }
        for index, values in enumerate(thermo)
    ]
    summary = {
        "samples": len(rows),
        "target_temperature_K": target_temperature,
        "mean_temperature_K": float(np.mean(thermo[:, 1])),
        "maximum_temperature_K": float(np.max(thermo[:, 1])),
        "energy_drift_eV_per_atom_per_ps": slope,
        "maximum_energy_excursion_eV_per_atom": excursion,
        "minimum_distance_A": float(np.min(distances)),
        "finite": bool(np.all(np.isfinite(thermo)) and np.all(np.isfinite(distances))),
        "tolerances": {
            "absolute_energy_drift_eV_per_atom_per_ps": 1.0e-3,
            "maximum_energy_excursion_eV_per_atom": 5.0e-2,
            "minimum_distance_A": 0.6,
        },
    }
    summary["passed"] = bool(
        summary["finite"]
        and abs(slope)
        <= summary["tolerances"]["absolute_energy_drift_eV_per_atom_per_ps"]
        and excursion
        <= summary["tolerances"]["maximum_energy_excursion_eV_per_atom"]
        and summary["minimum_distance_A"]
        >= summary["tolerances"]["minimum_distance_A"]
    )
    return summary, rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def retained_close_range_status(output, model_names):
    dimer_path = Path(output) / "dimer_scan.csv"
    compressed_path = Path(output) / "compressed_cells.csv"
    if not dimer_path.is_file() or not compressed_path.is_file():
        raise FileNotFoundError(
            "NVE qualification requires retained dimer_scan.csv and "
            "compressed_cells.csv from the close-range stage."
        )
    with dimer_path.open(encoding="utf-8") as handle:
        dimer_rows = list(csv.DictReader(handle))
    with compressed_path.open(encoding="utf-8") as handle:
        compressed_rows = list(csv.DictReader(handle))
    requested = set(model_names)
    if requested - {row["model"] for row in dimer_rows}:
        raise ValueError("Retained dimer evidence does not cover every NVE model.")
    if requested - {row["model"] for row in compressed_rows}:
        raise ValueError("Retained compressed-cell evidence does not cover every NVE model.")
    passed = True
    for name in requested:
        rows = [row for row in dimer_rows if row["model"] == name]
        closest = min(rows, key=lambda row: float(row["distance_A"]))
        passed = passed and all(row["finite"] == "True" for row in rows)
        passed = passed and float(closest["atom_1_force_x_eV_per_A"]) < 0.0
        rows = [row for row in compressed_rows if row["model"] == name]
        passed = passed and all(
            row["finite"] == "True" and row["compression_positive"] == "True"
            for row in rows
        )
    return bool(passed)


def qualify_system(system, lammps, model_names, stage):
    deploy, records = selected_records(system, model_names)
    system_config = study.read_system_config(system)
    output = study.system_root(system) / "finalist" / "validation" / "stability"
    output.mkdir(parents=True, exist_ok=True)
    dimer_rows = []
    compressed_rows = []
    nve_summaries = []
    for model_index, record in enumerate(records):
        name = record["id"]
        if stage in {"close-range", "all"}:
            for distance in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0):
                input_path = output / f"in.{name}.dimer.{distance:.2f}"
                input_path.write_text(
                    dimer_deck(
                        system,
                        float(system_config["atomic_masses"][0]),
                        record,
                        distance,
                    ),
                    encoding="utf-8",
                )
                screen = output / f"screen.{name}.dimer.{distance:.2f}.log"
                run(
                    [lammps, "-log", "none", "-in", input_path],
                    deploy,
                    screen,
                    300,
                )
                matches = DIMER_RESULT.findall(screen.read_text(encoding="utf-8"))
                if not matches:
                    raise RuntimeError(f"No dimer result for {system} {name} {distance}.")
                energy, force = map(float, matches[-1])
                dimer_rows.append(
                    {
                        "system": system,
                        "model": name,
                        "distance_A": distance,
                        "energy_eV": energy,
                        "atom_1_force_x_eV_per_A": force,
                        "finite": bool(np.isfinite(energy) and np.isfinite(force)),
                    }
                )
            input_path = output / f"in.{name}.compressed"
            input_path.write_text(
                compressed_deck(system_config, record, 0.75), encoding="utf-8"
            )
            screen = output / f"screen.{name}.compressed.log"
            run([lammps, "-log", "none", "-in", input_path], deploy, screen, 300)
            matches = COMPRESSED_RESULT.findall(screen.read_text(encoding="utf-8"))
            if not matches:
                raise RuntimeError(f"No compressed-cell result for {system} {name}.")
            energy, pressure = map(float, matches[-1])
            compressed_rows.append(
                {
                    "system": system,
                    "model": name,
                    "lattice_scale": 0.75,
                    "energy_eV_per_atom": energy,
                    "pressure_bar": pressure,
                    "finite": bool(np.isfinite(energy) and np.isfinite(pressure)),
                    "compression_positive": bool(pressure > 0.0),
                }
            )
        if stage in {"nve", "all"}:
            reference = system_config["stability_temperature_reference"]
            temperature = float(reference["melting_temperature_K"]) * float(
                reference["nve_fraction_of_melting"]
            )
            trajectory = output / f"trajectory.{name}.lammpstrj"
            input_path = output / f"in.{name}.nve"
            input_path.write_text(
                nve_deck(
                    system_config,
                    record,
                    trajectory,
                    temperature,
                    77123 + 1009 * model_index + sum(map(ord, system)),
                ),
                encoding="utf-8",
            )
            log = output / f"log.{name}.nve"
            run(
                [lammps, "-log", log, "-in", input_path],
                deploy,
                output / f"screen.{name}.nve.log",
                3600,
            )
            atoms_per_cell = {"bcc": 2, "fcc": 4, "diamond": 8}[
                system_config["crystal"]
            ]
            summary, rows = nve_metrics(
                log,
                trajectory,
                8 * atoms_per_cell,
                0.0005,
                temperature,
            )
            summary.update(
                {
                    "system": system,
                    "model": name,
                    "temperature_reference": reference,
                    "nvt_steps": 2000,
                    "nve_steps": 10000,
                    "trajectory_stride": 100,
                    "trajectory": str(trajectory),
                }
            )
            write_csv(output / f"nve_trace.{name}.csv", rows)
            study.write_json(output / f"nve_summary.{name}.json", summary)
            nve_summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
    if dimer_rows:
        write_csv(output / "dimer_scan.csv", dimer_rows)
    if compressed_rows:
        write_csv(output / "compressed_cells.csv", compressed_rows)
    close_range_source = "current_stage"
    close_range_passed = True
    if dimer_rows:
        for name in {row["model"] for row in dimer_rows}:
            rows = [row for row in dimer_rows if row["model"] == name]
            closest = min(rows, key=lambda row: row["distance_A"])
            close_range_passed = close_range_passed and all(row["finite"] for row in rows)
            close_range_passed = close_range_passed and closest[
                "atom_1_force_x_eV_per_A"
            ] < 0.0
        close_range_passed = close_range_passed and all(
            row["finite"] and row["compression_positive"] for row in compressed_rows
        )
    elif stage == "nve":
        close_range_source = "retained_csv"
        close_range_passed = retained_close_range_status(output, model_names)
    report = {
        "schema": "ye3t_mlearn_stability_qualification_v1",
        "system": system,
        "models": [row["id"] for row in records],
        "close_range_passed": close_range_passed,
        "close_range_source": close_range_source,
        "nve": nve_summaries,
        "passed": bool(
            close_range_passed and all(row["passed"] for row in nve_summaries)
        ),
    }
    study.write_json(output / "summary.json", report)
    study.append_progress(
        "finalist_stability_qualification",
        system=system,
        stage=stage,
        passed=report["passed"],
        summary=str(output / "summary.json"),
    )
    if not report["passed"]:
        raise SystemExit(f"{system} stability qualification failed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--stage", choices=("close-range", "nve", "all"), default="all")
    args = parser.parse_args()
    lammps = args.lammps.resolve()
    if not lammps.is_file():
        raise FileNotFoundError(lammps)
    models = tuple(args.model) if args.model else DEFAULT_MODELS
    for system in args.systems:
        qualify_system(system, lammps, models, args.stage)


if __name__ == "__main__":
    main()
