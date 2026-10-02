#!/usr/bin/env python3
"""Compile nested ACE catalogues and materialize reusable training rows."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ye3t.couplings import normalize_compact_label
from ye3t_ace import (
    YE3TDescriptors,
    YE3TRepresentation,
    save_linear_ace_ase_bundle,
)
from ye3t_ace.ace.lammps_export import compile_ordinary_scalar_catalogue
from ye3t_ace.ace.linear_ace import LinearACEScalarModelBundle, export_scalar_bundle_to_yace
from ye3t_ace.equivariant_calc import ACECovariantEvaluator

import build_rows as run_screen


HERE = Path(__file__).resolve().parent
PUBLIC = Path(os.environ.get("YE3T_COST_PUBLIC_ROOT", str(HERE.parent))).resolve()
CONFIG_PATH = Path(
    os.environ.get("YE3T_COST_CONFIG", str(PUBLIC / "config.json"))
).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
WORKFLOW_ROOT = Path(
    os.environ.get(
        "YE3T_COST_WORKFLOW_ROOT",
        str(CONFIG["runtime"]["workflow_root"]),
    )
)
if not WORKFLOW_ROOT.is_absolute():
    WORKFLOW_ROOT = (CONFIG_PATH.parent / WORKFLOW_ROOT).resolve()
SOURCE = WORKFLOW_ROOT / "Si" / "catalogues" / "ordinary_catalogue_source.json"
DEFAULT_OUTPUT = WORKFLOW_ROOT / "Si" / "generated_ordinary_controls"
TARGETS = tuple(int(value) for value in CONFIG["basis"]["target_descriptor_counts"])
ALPHAS = (1.0e-6, 1.0e-4, 1.0e-2)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_maximum_catalogue(source, output, system, targets):
    profile_id = f"{system.lower()}_generated_linear_ace_{max(targets)}"
    source = json.loads(json.dumps(source))
    source_path = output / "catalogue_source.json"
    write_json(source_path, source)
    path = output / "catalogue_application.json"
    if path.is_file():
        application = read_json(path)
    else:
        application = compile_ordinary_scalar_catalogue(
            source,
            profile_id,
            progress=lambda index, count, feature_id: print(
                f"catalogue compilation {index}/{count}: {feature_id}",
                flush=True,
            )
            if index == 1 or index == count or index % max(1, count // 10) == 0
            else None,
        )
        write_json(path, application)
    expected = len(source["profiles"][profile_id]["feature_ids"])
    if len(application["feature_ids"]) != expected:
        raise RuntimeError("The maximum generated ordinary catalogue size changed.")
    return source, source_path, application, path


def build_descriptor(application, system):
    labels = tuple(
        normalize_compact_label(row["compact_label"])
        for row in application["rows"]
    )
    ranks = tuple(sorted({int(label.rank) for label in labels}))
    nmax = tuple(
        max(
            max(int(value) for value in label.n_tuple)
            for label in labels
            if int(label.rank) == rank
        )
        for rank in ranks
    )
    lmax = tuple(
        max(
            max(int(value) for value in label.l_tuple)
            for label in labels
            if int(label.rank) == rank
        )
        for rank in ranks
    )
    representation = YE3TRepresentation.ace(
        basis_mode=None,
        fast_path_policy="disable",
        metadata={
            "global_young_sector": "(N)",
            "basis_convention": "pace_complex_magnetic_y00_1",
        },
    )
    descriptor = YE3TDescriptors.ace(
        {
            "elements": [system],
            "type_map": {system: 0},
            "cutoff": run_screen.CUTOFF,
            "ranks": ranks,
            "basis_type": "no_charge",
            "k_o_max": 0,
            "k_max": [0] * len(ranks),
            "nmax": nmax,
            "lmax": lmax,
            "lmin": [0] * len(ranks),
            "L_R": 0,
            "M_R_values": [0],
            "ordinary_scalar_catalogue": application,
            "factorized_descriptor_runtime_policy": "disable",
            "site_basis_config": {
                "rc": [run_screen.CUTOFF],
                "lmbda": [run_screen.RADIAL_CONFIG["lmbda"]],
                "nradmax": max(nmax),
                "lmax": max(lmax),
                "kmax": 0,
                "possible_types": [0],
                "radial_basis": "PACE_ChebExpCos",
                "chemical_basis": "delta",
                "charge_mode": "none",
                "atomic_base_normalization": "none",
                "factor_normalization": "none",
                "spherical_backend": "complex",
                "spherical_normalization": "pace_y00_one",
                "source_backend": "torch",
                "dtype": "float64",
                "pace_cutoff_width": [run_screen.RADIAL_CONFIG["cutoff_width"]],
                "pace_spline_spacing": [0.001],
                "pace_inner_cutoff": [0.0],
                "pace_inner_cutoff_width": [0.0],
                "pace_crad_policy": "identity",
            },
            "representation": representation,
            "backend": "pytorch",
            "strict_backend": True,
            "validate_backend": True,
            "device": "cpu",
        }
    )
    return descriptor


def bundle(descriptor, width, weight, bias, fit_path, alpha, system):
    return LinearACEScalarModelBundle(
        settings=descriptor.settings,
        site_basis_config=descriptor.site_basis_config,
        descriptor_specs=tuple(descriptor.descriptor_specs[:width]),
        weight=np.asarray(weight, dtype=np.float64),
        bias=float(bias),
        basis_mode=None,
        fit_method="structure_balanced_ridge",
        fit_metadata={
            "schema": "ye3t_mlearn_element_generated_ordinary_control_v1",
            "system": system,
            "feature_count": int(width),
            "ridge_alpha": float(alpha),
            "source_fit": fit_path.name,
            "source_fit_sha256": sha256(fit_path),
            "reference_potential": None,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--data", type=Path, default=run_screen.DATA)
    parser.add_argument("--system", default="Si")
    parser.add_argument("--cutoff", type=float, default=run_screen.CUTOFF)
    parser.add_argument("--seed", type=int, default=run_screen.SEED)
    parser.add_argument("--targets", type=int, nargs="+", default=TARGETS)
    parser.add_argument(
        "--stage", choices=("catalogue", "rows", "fit"), default="rows"
    )
    parser.add_argument(
        "--include-test-rows",
        action="store_true",
        help="Materialize target-free outer-test geometry rows as well as training rows.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    system = str(args.system)
    prefix = system.lower()
    targets = tuple(int(value) for value in args.targets)
    source_parent = args.source.resolve()
    run_screen.SYSTEM = system
    run_screen.CUTOFF = float(args.cutoff)
    run_screen.SEED = int(args.seed)
    run_screen.DATA = args.data.resolve()
    source = read_json(source_parent)
    source, source_path, application, application_path = (
        materialize_maximum_catalogue(source, output, system, targets)
    )
    descriptor = build_descriptor(application, system)
    evaluator = ACECovariantEvaluator(
        descriptor.site_basis_config,
        backend="pytorch",
        strict_backend=True,
        validate_backend=True,
        factorized_descriptor_runtime_policy="disable",
    )
    evaluator.precompile_descriptors(descriptor.descriptor_specs)

    if args.stage == "catalogue":
        payload = {
            "schema": "ye3t_mlearn_ordinary_catalogue_build_v1",
            "system": system,
            "requested_feature_counts": list(targets),
            "resolved_maximum_feature_count": len(application["feature_ids"]),
            "catalogue_source": str(source_path),
            "catalogue_source_sha256": sha256(source_path),
            "catalogue_application": str(application_path),
            "catalogue_application_sha256": sha256(application_path),
        }
        write_json(output / "catalogue_build.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return

    training_structures = run_screen.load_xyz_structures(
        run_screen.DATA / f"{prefix}_training.xyz"
    )
    if args.stage == "fit" or args.include_test_rows:
        test_structures = run_screen.load_xyz_structures(
            run_screen.DATA / f"{prefix}_test.xyz"
        )
    else:
        test_structures = []
    structures = list(training_structures) + list(test_structures)
    rows = run_screen.cached_rows(
        "ordinary",
        (descriptor, evaluator),
        structures,
        output / "row_cache_ordinary",
    )
    if args.stage == "rows":
        payload = {
            "schema": "ye3t_mlearn_target_free_ordinary_row_cache_v1",
            "system": system,
            "training_structures": len(training_structures),
            "test_geometry_structures": len(test_structures),
            "target_values_used": False,
            "feature_count": len(application["feature_ids"]),
            "catalogue_application": str(application_path),
            "catalogue_application_sha256": sha256(application_path),
            "row_cache": str(output / "row_cache_ordinary"),
        }
        write_json(output / "row_cache_manifest.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return

    energies = {}
    forces = {}
    for index, atoms in enumerate(structures):
        energies[index], forces[index] = run_screen.target(atoms)
    train_indices, validation_indices = run_screen.stratified_validation(
        training_structures
    )
    final_fit_indices = tuple(range(len(training_structures)))
    test_indices = tuple(range(len(training_structures), len(structures)))
    metrics = []
    trials = []
    models = output / "models"
    models.mkdir(exist_ok=True)
    source_ids = tuple(application["feature_ids"])
    for requested_count in targets:
        profile_id = f"{prefix}_generated_linear_ace_{requested_count}"
        requested_ids = tuple(source["profiles"][profile_id]["feature_ids"])
        width = len(requested_ids)
        if requested_ids != source_ids[:width]:
            raise RuntimeError(f"{profile_id} is not a nested prefix of the 150 profile.")
        selected_rows = {
            index: run_screen.select(row, tuple(range(width)))
            for index, row in rows.items()
        }
        blocks = ((f"pace_generated_{width}", selected_rows),)
        selected, _bias, _weights, arm_trials = run_screen.fit_arm(
            profile_id,
            blocks,
            train_indices,
            validation_indices,
            structures,
            energies,
            forces,
            ALPHAS,
            (1.0,),
        )
        trials.extend(arm_trials)
        _selected, bias, weights, _trials = run_screen.fit_arm(
            profile_id,
            blocks,
            final_fit_indices,
            validation_indices,
            structures,
            energies,
            forces,
            (selected["alpha"],),
            (1.0,),
        )
        test = run_screen.score(
            bias,
            weights,
            blocks,
            test_indices,
            structures,
            energies,
            forces,
        )
        model_path = models / f"{prefix}_pace_generated_{width}.npz"
        np.savez_compressed(
            model_path,
            bias=np.asarray(bias),
            weight=np.asarray(weights[0]),
            feature_count=np.asarray(width),
            selected_alpha=np.asarray(selected["alpha"]),
        )
        fitted = bundle(
            descriptor,
            width,
            weights[0],
            bias,
            model_path,
            selected["alpha"],
            system,
        )
        export_scalar_bundle_to_yace(
            fitted,
            models / f"{prefix}_pace_generated_{width}.yace",
            elements=(system,),
            compatibility="lammps_pace_linear_v1",
        )
        save_linear_ace_ase_bundle(
            fitted,
            models / f"{prefix}_pace_generated_{width}.pt",
            cutoff=run_screen.CUTOFF,
            type_map={system: 0},
        )
        row = {
            "arm": f"pace_generated_{width}",
            "requested_feature_count": requested_count,
            "feature_count": width,
            "selected_alpha": selected["alpha"],
            "validation_energy_rmse_eV_per_atom": selected[
                "validation_energy_rmse_eV_per_atom"
            ],
            "validation_force_rmse_eV_per_A": selected[
                "validation_force_rmse_eV_per_A"
            ],
            "test_energy_rmse_eV_per_atom": test["energy_rmse_eV_per_atom"],
            "test_force_rmse_eV_per_A": test["force_rmse_eV_per_A"],
            "yace_sha256": sha256(
                models / f"{prefix}_pace_generated_{width}.yace"
            ),
        }
        metrics.append(row)
        run_screen.write_json(output / f"test_groups_{width}.json", test["groups"])
        print(json.dumps(row, sort_keys=True), flush=True)

    run_screen.write_csv(output / "metrics.csv", metrics)
    run_screen.write_csv(output / "ridge_trials.csv", trials)
    write_json(
        output / "summary.json",
        {
            "schema": "ye3t_mlearn_element_generated_ordinary_controls_v1",
            "system": system,
            "passed": True,
            "dataset": f"mlearn {system} published fixed split",
            "catalogue_source": str(source_path),
            "catalogue_source_sha256": sha256(source_path),
            "catalogue_source_parent": str(source_parent),
            "catalogue_source_parent_sha256": sha256(source_parent),
            "catalogue_application": str(application_path),
            "catalogue_application_sha256": sha256(application_path),
            "radial_basis": "PACE_ChebExpCos",
            "cutoff_A": run_screen.CUTOFF,
            "requested_feature_counts": list(targets),
            "resolved_feature_counts": [row["feature_count"] for row in metrics],
            "ridge_alphas": list(ALPHAS),
            "selection_split": {
                "train_structures": len(train_indices),
                "validation_structures": len(validation_indices),
                "seed": run_screen.SEED,
            },
            "final_fit_structures": len(final_fit_indices),
            "test_structures": len(test_indices),
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    main()
