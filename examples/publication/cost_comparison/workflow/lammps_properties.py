#!/usr/bin/env python3
"""Calculate cubic EOS and relaxed-ion elastic response through LAMMPS."""

import argparse
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EV_PER_A3_TO_GPA = 160.21766208
RESULT = re.compile(
    r"YE3T_PROPERTY energy_eV_per_atom=([0-9.eE+-]+) "
    r"volume_A3=([0-9.eE+-]+) atoms=([0-9]+)"
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def crystal_atoms_per_conventional_cell(crystal):
    values = {"bcc": 2, "fcc": 4, "diamond": 8}
    if crystal not in values:
        raise ValueError(f"Unsupported cubic crystal: {crystal}")
    return values[crystal]


def common_deck(system, model, lattice_constant, strain_mode, delta):
    crystal = str(system["crystal"])
    species = str(system["species"][0])
    mass = float(system["atomic_masses"][0])
    pair_coefficients = "\n".join(model["pair_coefficients"])
    strain = []
    if strain_mode == "hydrostatic":
        scale = 1.0 + delta
        strain.append(
            f"change_box all x scale {scale:.17g} y scale {scale:.17g} "
            f"z scale {scale:.17g} remap"
        )
    elif strain_mode == "tetragonal":
        sx = 1.0 + delta
        sy = 1.0 - delta
        sz = 1.0 / (1.0 - delta * delta)
        strain.append(
            f"change_box all x scale {sx:.17g} y scale {sy:.17g} "
            f"z scale {sz:.17g} remap"
        )
    elif strain_mode == "shear":
        box_length = 2.0 * lattice_constant
        strain.extend(
            (
                "change_box all triclinic",
                f"change_box all xy delta {delta * box_length:.17g} remap units box",
            )
        )
    elif strain_mode not in {None, "none"}:
        raise ValueError(f"Unknown strain mode: {strain_mode}")
    strain_text = "\n".join(strain)
    return f"""units metal
atom_style atomic
boundary p p p
atom_modify map yes sort 0 0.0
newton on
lattice {crystal} {lattice_constant:.17g}
region cell block 0 2 0 2 0 2 units lattice
create_box 1 cell
create_atoms 1 box
mass 1 {mass:.17g}
reset_atoms id sort yes
{strain_text}
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
{model['pair_style']}
{pair_coefficients}
min_style cg
minimize 1.0e-12 1.0e-12 1000 10000
variable n equal count(all)
variable e equal pe/v_n
variable v equal vol
print "YE3T_PROPERTY energy_eV_per_atom=$(v_e:%.17g) volume_A3=$(v_v:%.17g) atoms=$(v_n:%.0f)"
"""


def run_point(lammps, model_directory, run_directory, system, model, lattice, mode, delta):
    tag = f"{model['id']}.{mode or 'eos'}.{delta:+.6f}".replace("+", "p").replace("-", "m")
    input_path = run_directory / f"in.{tag}"
    log_path = run_directory / f"log.{tag}"
    input_path.write_text(
        common_deck(system, model, lattice, mode, delta), encoding="utf-8"
    )
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [str(lammps), "-in", str(input_path), "-log", str(log_path)],
        cwd=model_directory,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"LAMMPS failed for {tag} with code {result.returncode}:\n{result.stdout[-4000:]}"
        )
    matches = RESULT.findall(result.stdout)
    if not matches:
        raise RuntimeError(f"LAMMPS output for {tag} has no YE3T_PROPERTY record.")
    energy, volume, atoms = matches[-1]
    return {
        "model": model["id"],
        "strain_mode": mode or "eos",
        "delta": float(delta),
        "lattice_constant_A": float(lattice),
        "energy_eV_per_atom": float(energy),
        "volume_A3": float(volume),
        "atoms": int(atoms),
        "input": input_path.name,
        "log": log_path.name,
    }


def fit_eos(rows, crystal):
    volumes = np.asarray(
        [row["volume_A3"] / row["atoms"] for row in rows], dtype=np.float64
    )
    energies = np.asarray([row["energy_eV_per_atom"] for row in rows])
    coefficients = np.polyfit(volumes, energies, 4)
    polynomial = np.poly1d(coefficients)
    derivative = np.polyder(polynomial)
    curvature = np.polyder(derivative)
    candidates = [
        float(root.real)
        for root in np.roots(derivative)
        if abs(float(root.imag)) < 1.0e-10
        and min(volumes) <= float(root.real) <= max(volumes)
        and float(curvature(float(root.real))) > 0.0
    ]
    if not candidates:
        raise RuntimeError("Quartic EOS fit has no physical minimum in the scan range.")
    equilibrium_volume = min(candidates, key=lambda value: float(polynomial(value)))
    atoms_per_cell = crystal_atoms_per_conventional_cell(crystal)
    lattice = (atoms_per_cell * equilibrium_volume) ** (1.0 / 3.0)
    bulk = equilibrium_volume * float(curvature(equilibrium_volume)) * EV_PER_A3_TO_GPA
    residual = energies - polynomial(volumes)
    return {
        "equilibrium_lattice_constant_A": float(lattice),
        "equilibrium_volume_A3_per_atom": float(equilibrium_volume),
        "cohesive_energy_eV_per_atom": float(polynomial(equilibrium_volume)),
        "bulk_modulus_GPa_from_eos": float(bulk),
        "eos_fit_rmse_eV_per_atom": float(np.sqrt(np.mean(residual * residual))),
        "quartic_coefficients": [float(value) for value in coefficients],
    }


def fit_even_energy(rows, equilibrium_energy, equilibrium_volume):
    delta = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    density = np.asarray(
        [
            (row["energy_eV_per_atom"] - equilibrium_energy)
            / equilibrium_volume
            for row in rows
        ],
        dtype=np.float64,
    )
    design = np.column_stack((np.ones(delta.size), delta * delta, delta ** 4))
    beta = np.linalg.lstsq(design, density, rcond=None)[0]
    residual = density - design @ beta
    return float(beta[1]), float(np.sqrt(np.mean(residual * residual)))


def elastic_constants(rows, eos):
    by_mode = {
        mode: [row for row in rows if row["strain_mode"] == mode]
        for mode in ("hydrostatic", "tetragonal", "shear")
    }
    coefficients = {}
    residuals = {}
    for mode, values in by_mode.items():
        coefficients[mode], residuals[mode] = fit_even_energy(
            values,
            eos["cohesive_energy_eV_per_atom"],
            eos["equilibrium_volume_A3_per_atom"],
        )
    scalar_sum = (2.0 / 3.0) * coefficients["hydrostatic"]
    difference = coefficients["tetragonal"]
    c11 = (scalar_sum + 2.0 * difference) / 3.0
    c12 = (scalar_sum - difference) / 3.0
    c44 = 2.0 * coefficients["shear"]
    values = {
        "C11_GPa": c11 * EV_PER_A3_TO_GPA,
        "C12_GPa": c12 * EV_PER_A3_TO_GPA,
        "C44_GPa": c44 * EV_PER_A3_TO_GPA,
        "bulk_modulus_GPa_from_elastic": (c11 + 2.0 * c12) * EV_PER_A3_TO_GPA / 3.0,
        "elastic_fit_rmse_eV_per_A3": residuals,
    }
    values["born_stable"] = bool(
        values["C11_GPa"] - values["C12_GPa"] > 0.0
        and values["C11_GPa"] + 2.0 * values["C12_GPa"] > 0.0
        and values["C44_GPa"] > 0.0
    )
    return values


def add_reference_errors(row, reference):
    comparisons = {
        "equilibrium_lattice_constant_A": "lattice_constant_A",
        "C11_GPa": "C11_GPa",
        "C12_GPa": "C12_GPa",
        "C44_GPa": "C44_GPa",
        "bulk_modulus_GPa_from_eos": "bulk_modulus_GPa",
        "bulk_modulus_GPa_from_elastic": "bulk_modulus_GPa",
    }
    errors = {}
    for prediction, target in comparisons.items():
        value = float(row[prediction])
        expected = float(reference[target])
        errors[prediction + "_error"] = value - expected
        errors[prediction + "_percent_error"] = 100.0 * (value - expected) / expected
    elastic_keys = ("C11_GPa", "C12_GPa", "C44_GPa")
    errors["elastic_constants_mean_absolute_percent_error"] = float(
        np.mean([abs(errors[key + "_percent_error"]) for key in elastic_keys])
    )
    return errors


def render(eos_rows, properties, output, system):
    labels = {row["model"]: row["label"] for row in properties}
    reference = system.get("property_reference")
    figure, axes = plt.subplots(1, 3, figsize=(14.1, 4.3))
    for model in labels:
        rows = [row for row in eos_rows if row["model"] == model]
        volume = np.asarray([row["volume_A3"] / row["atoms"] for row in rows])
        energy = np.asarray([row["energy_eV_per_atom"] for row in rows])
        axes[0].plot(volume, energy - min(energy), marker="o", ms=3, label=labels[model])
    if reference:
        atoms_per_cell = crystal_atoms_per_conventional_cell(system["crystal"])
        reference_volume = float(reference["lattice_constant_A"]) ** 3 / atoms_per_cell
        axes[0].axvline(
            reference_volume,
            color="black",
            linestyle=":",
            linewidth=1.1,
            label="DFT equilibrium volume",
        )
    axes[0].set_xlabel("Volume (Å³/atom)")
    axes[0].set_ylabel("Energy relative to sampled minimum (eV/atom)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5)
    x = np.arange(len(properties))
    width = 0.25
    for offset, key, label in (
        (-width, "C11_GPa", "$C_{11}$"),
        (0.0, "C12_GPa", "$C_{12}$"),
        (width, "C44_GPa", "$C_{44}$"),
    ):
        axes[1].bar(x + offset, [row[key] for row in properties], width, label=label)
    axes[1].set_xticks(x, [row["label"] for row in properties], rotation=25, ha="right")
    axes[1].set_ylabel("Elastic constant (GPa)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    error_keys = (
        "equilibrium_lattice_constant_A_percent_error",
        "C11_GPa_percent_error",
        "C12_GPa_percent_error",
        "C44_GPa_percent_error",
        "bulk_modulus_GPa_from_elastic_percent_error",
    )
    error_labels = ("$a_0$", "$C_{11}$", "$C_{12}$", "$C_{44}$", "$B$")
    if reference:
        for key, color, label in (
            ("C11_GPa", "#4c78a8", "DFT $C_{11}$"),
            ("C12_GPa", "#f58518", "DFT $C_{12}$"),
            ("C44_GPa", "#54a24b", "DFT $C_{44}$"),
        ):
            axes[1].axhline(
                float(reference[key]),
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
                label=label,
            )
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
        error_matrix = np.asarray(
            [[abs(float(row[key])) for key in error_keys] for row in properties]
        )
        image = axes[2].imshow(error_matrix, aspect="auto", cmap="YlOrRd", vmin=0.0)
        for row_index in range(error_matrix.shape[0]):
            for column_index in range(error_matrix.shape[1]):
                axes[2].text(
                    column_index,
                    row_index,
                    f"{error_matrix[row_index, column_index]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        axes[2].set_xticks(np.arange(len(error_labels)), error_labels)
        axes[2].set_yticks(
            np.arange(len(properties)),
            [row["label"] for row in properties],
            fontsize=7,
        )
        axes[2].set_title("Absolute error to DFT (%)")
        figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].axis("off")
    figure.tight_layout()
    stem = str(system["system"]).lower() + "_eos_elastic"
    figure.savefig(output / f"{stem}.png", dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--system", type=Path, default=HERE / "systems" / "Si.json")
    parser.add_argument("--models", type=Path, default=HERE / "lammps_models" / "Si.json")
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", action="append", default=[])
    args = parser.parse_args()
    system = read_json(args.system)
    matrix = read_json(args.models)
    selected = set(args.model_id)
    models = [
        model for model in matrix["models"]
        if not selected or model["id"] in selected
    ]
    if selected - {model["id"] for model in models}:
        raise ValueError(f"Unknown model IDs: {sorted(selected - {model['id'] for model in models})}")
    output = args.output.resolve()
    run_directory = output / "lammps_runs"
    output.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    initial = float(system["initial_lattice_constant_A"])
    # The publication protocol fixes a 21-point lattice scan over +/-10%.
    # This also exposes unphysical compressed-state behavior that a narrow
    # local elastic scan can miss.
    eos_scales = np.linspace(0.90, 1.10, 21)
    strain_values = (-0.01, -0.0075, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.0075, 0.01)
    eos_rows = []
    strain_rows = []
    properties = []
    for model in models:
        model_eos = [
            run_point(
                args.lammps.resolve(), args.model_directory.resolve(), run_directory,
                system, model, initial * scale, None, float(scale - 1.0),
            )
            for scale in eos_scales
        ]
        eos_rows.extend(model_eos)
        eos = fit_eos(model_eos, system["crystal"])
        model_strains = []
        for mode in ("hydrostatic", "tetragonal", "shear"):
            for delta in strain_values:
                model_strains.append(
                    run_point(
                        args.lammps.resolve(), args.model_directory.resolve(), run_directory,
                        system, model, eos["equilibrium_lattice_constant_A"], mode, delta,
                    )
                )
        strain_rows.extend(model_strains)
        elastic = elastic_constants(model_strains, eos)
        property_row = {
            "model": model["id"],
            "label": model["label"],
            "descriptor_count": model["descriptor_count"],
            **eos,
            **elastic,
        }
        if system.get("property_reference"):
            property_row.update(
                add_reference_errors(property_row, system["property_reference"])
            )
        properties.append(property_row)
        print(json.dumps(properties[-1], sort_keys=True), flush=True)
    write_csv(output / "eos.csv", eos_rows)
    write_csv(output / "elastic_strains.csv", strain_rows)
    flat = []
    for row in properties:
        item = dict(row)
        item["elastic_fit_rmse_eV_per_A3"] = json.dumps(
            item["elastic_fit_rmse_eV_per_A3"], sort_keys=True
        )
        item["quartic_coefficients"] = json.dumps(item["quartic_coefficients"])
        flat.append(item)
    write_csv(output / "properties.csv", flat)
    write_json(
        output / "summary.json",
        {
            "schema": "ye3t_lammps_cubic_property_comparison_v1",
            "system": system,
            "model_matrix": matrix,
            "eos_scale_range": [float(eos_scales[0]), float(eos_scales[-1])],
            "elastic_strains": list(strain_values),
            "energy_and_force_engine": "LAMMPS",
            "property_reference": system.get("property_reference"),
            "properties": properties,
        },
    )
    render(eos_rows, properties, output, system)


if __name__ == "__main__":
    main()
