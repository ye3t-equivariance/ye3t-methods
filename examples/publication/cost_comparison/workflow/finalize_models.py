#!/usr/bin/env python3
"""Fit, export, and test frozen radial/ZBL cost-comparison finalists."""

import argparse
import csv
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

import optimize_cached as study
import radial_screen as radial
from ye3t_methods.atomistic.ace.lammps_export import export_scalar_bundle_to_lammps
from ye3t_methods.atomistic.ace.linear_ace import (
    LinearACEScalarModelBundle,
    export_scalar_bundle_to_yace,
    load_xyz_structures,
)
from ye3t_methods.atomistic.linear_statistics import (
    score_linear_statistics,
    select_linear_statistics,
    structure_linear_statistics,
    sum_linear_statistics,
)
from ye3t_methods.atomistic.tagged_cauchy_fit import arm_lammps_model
from ye3t_methods.atomistic.tagged_cauchy_linear import (
    export_tagged_composite_model,
    export_tagged_model,
    load_tagged_model,
)


RUNTIME = {}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_inputs(system):
    root = study.system_root(system)
    radial_path = root / "radial_screen" / "selection_frozen.json"
    zbl_path = root / "zbl_screen" / "selection_frozen.json"
    radial_selection = study.read_json(radial_path)
    zbl_selection = study.read_json(zbl_path)
    if radial_selection["test_split_opened"] or zbl_selection["test_split_opened"]:
        raise RuntimeError("A training-only selection record claims that test data was opened.")
    return {
        "radial": radial_selection["selected"],
        "radial_path": radial_path,
        "radial_sha256": file_sha256(radial_path),
        "zbl": zbl_selection["selected_config"],
        "zbl_path": zbl_path,
        "zbl_sha256": file_sha256(zbl_path),
    }


def structures(system, split):
    path = study.data_root(system) / f"{system.lower()}_{split}.xyz"
    return load_xyz_structures(path)


def runtime(system, selected):
    key = (
        system,
        int(selected["radial"]["candidate"]),
        float(selected["radial"]["cutoff_A"]),
        float(selected["radial"]["radial_lambda"]),
    )
    cached = RUNTIME.get(key)
    if cached is not None:
        return cached
    descriptor, evaluator = radial.build_ordinary_descriptor(
        system,
        float(selected["radial"]["cutoff_A"]),
        float(selected["radial"]["radial_lambda"]),
        float(selected["radial"]["cutoff_width_A"]),
    )
    arm = radial.build_tagged_arm(system)
    cached = (descriptor, evaluator, arm)
    RUNTIME[key] = cached
    return cached


def finalist_row_roots(system):
    root = study.system_root(system) / "finalist" / "rows"
    return root / "ordinary", root / "tagged"


def row_paths(system, global_index, selected):
    if int(selected["radial"]["candidate"]) == 0:
        return (
            study.ordinary_row_root(system) / f"frame_{global_index:04d}.npz",
            study.tagged_row_root(system) / f"frame_{global_index:04d}.npz",
        )
    ordinary_root, tagged_root = finalist_row_roots(system)
    return (
        ordinary_root / f"frame_{global_index:04d}.npz",
        tagged_root / f"frame_{global_index:04d}.npz",
    )


def materialize_split(system, split, selected):
    frames = structures(system, split)
    training_count = len(structures(system, "training"))
    offset = 0 if split == "training" else training_count
    missing = []
    for local_index in range(len(frames)):
        paths = row_paths(system, offset + local_index, selected)
        if not all(path.is_file() for path in paths):
            missing.append(local_index)
    if not missing:
        return {"split": split, "structure_count": len(frames), "computed": 0}
    if int(selected["radial"]["candidate"]) == 0:
        raise FileNotFoundError("The validated baseline row cache is incomplete.")
    descriptor, evaluator, arm = runtime(system, selected)
    ordinary_root, tagged_root = finalist_row_roots(system)
    ordinary_root.mkdir(parents=True, exist_ok=True)
    tagged_root.mkdir(parents=True, exist_ok=True)
    radial_config = {
        "lmbda": float(selected["radial"]["radial_lambda"]),
        "cutoff_width": float(selected["radial"]["cutoff_width_A"]),
    }
    cutoff = float(selected["radial"]["cutoff_A"])
    for position, local_index in enumerate(missing):
        global_index = offset + local_index
        ordinary_path, tagged_path = row_paths(system, global_index, selected)
        atoms = frames[local_index]
        ordinary = None
        tagged = None
        if not ordinary_path.is_file():
            ordinary = radial.ordinary_row(descriptor, evaluator, atoms)
            np.savez_compressed(ordinary_path, **ordinary)
        if not tagged_path.is_file():
            tagged = radial.tagged_row(
                arm,
                atoms,
                cutoff,
                radial_config,
                tagged_root / "structure_rows",
            )
            np.savez_compressed(tagged_path, **tagged)
        if position == 0 or (position + 1) % 20 == 0 or position + 1 == len(missing):
            print(
                f"{system} finalist {split} rows {position + 1}/{len(missing)}",
                flush=True,
            )
        del ordinary, tagged
        radial.release_row_memory()
    return {
        "split": split,
        "structure_count": len(frames),
        "computed": len(missing),
    }


def combined_finalist_row(system, global_index, selected, row_hasher):
    ordinary_path, tagged_path = row_paths(system, global_index, selected)
    ordinary = study.load_row(ordinary_path)
    tagged = study.load_row(tagged_path)
    if ordinary["atom_count"] != tagged["atom_count"]:
        raise RuntimeError("Ordinary and tagged finalist row atom counts disagree.")
    row = {
        "feature_sums": np.concatenate(
            (ordinary["feature_sums"], tagged["feature_sums"])
        ),
        "feature_square_sums": np.concatenate(
            (ordinary["feature_square_sums"], tagged["feature_square_sums"])
        ),
        "force_design": np.concatenate(
            (ordinary["force_design"], tagged["force_design"]), axis=1
        ),
        "atom_count": int(ordinary["atom_count"]),
        "ordinary_feature_count": int(ordinary["feature_sums"].size),
        "tagged_feature_count": int(tagged["feature_sums"].size),
    }
    row_hasher.update(np.asarray([global_index, row["atom_count"]], dtype="<i8").tobytes())
    for name in ("feature_sums", "feature_square_sums", "force_design"):
        value = np.ascontiguousarray(row[name], dtype="<f8")
        row_hasher.update(np.asarray(value.shape, dtype="<i8").tobytes())
        row_hasher.update(value.tobytes())
    return row


def residual_statistics(system, split, selected, fold_count, seed, output):
    frames = structures(system, split)
    training_count = len(structures(system, "training"))
    offset = 0 if split == "training" else training_count
    reference_path = output / "zbl_reference.json"
    study.write_json(reference_path, selected["zbl"])
    reference_output = output / f"reference_{split}"
    reference_output.mkdir(parents=True, exist_ok=True)
    energies, forces, reference = study.reference_targets(
        frames, system, reference_path, reference_output
    )
    assignments = (
        study.fold_assignments(frames, fold_count, seed)
        if split == "training"
        else {index: 0 for index in range(len(frames))}
    )
    records = {}
    row_hasher = hashlib.sha256()
    ordinary_count = None
    tagged_count = None
    for local_index, atoms in enumerate(frames):
        row = combined_finalist_row(
            system, offset + local_index, selected, row_hasher
        )
        if ordinary_count is None:
            ordinary_count = row["ordinary_feature_count"]
            tagged_count = row["tagged_feature_count"]
        if (
            row["ordinary_feature_count"] != ordinary_count
            or row["tagged_feature_count"] != tagged_count
        ):
            raise RuntimeError("Finalist row feature counts change between structures.")
        record = structure_linear_statistics(
            row["feature_sums"],
            row["feature_square_sums"],
            row["force_design"],
            len(atoms),
            float(energies[local_index]) / len(atoms),
            np.asarray(forces[local_index], dtype=np.float64),
        )
        key = (str(atoms.info["config_type"]), int(assignments[local_index]))
        if key in records:
            records[key] = sum_linear_statistics((records[key], record))
        else:
            records[key] = record
        del row, record
    radial.release_row_memory()
    shards = records
    arrays, shard_records = study.flatten_shards(shards)
    stats_path = output / f"{split}_statistics.npz"
    np.savez_compressed(stats_path, **arrays)
    metadata = {
        "schema": "ye3t_mlearn_finalist_statistics_v1",
        "system": system,
        "split": split,
        "row_content_sha256": row_hasher.hexdigest(),
        "reference": reference,
        "fold_count": fold_count if split == "training" else 1,
        "fold_seed": seed if split == "training" else None,
        "ordinary_feature_count": ordinary_count,
        "tagged_feature_count": tagged_count,
        "shards": shard_records,
        "statistics_npz": stats_path.name,
        "statistics_npz_sha256": file_sha256(stats_path),
    }
    study.write_json(output / f"{split}_statistics.json", metadata)
    return metadata, shards


def fit_system(system, fold_count, seed):
    started = time.perf_counter()
    selected = selected_inputs(system)
    root = study.system_root(system) / "finalist"
    fits = root / "fits"
    fits.mkdir(parents=True, exist_ok=True)
    materialization = materialize_split(system, "training", selected)
    statistics, shards = residual_statistics(
        system, "training", selected, fold_count, seed, root
    )
    index = {
        "ordinary_feature_count": statistics["ordinary_feature_count"],
        "tagged_feature_count": statistics["tagged_feature_count"],
    }
    components = study.component_inventory(system)
    activity = study.tagged_feature_activity(
        sum_linear_statistics(shards.values()),
        index["ordinary_feature_count"],
        index["tagged_feature_count"],
    )
    models = study.catalogue_models(
        index, components, tagged_activity=activity
    )
    study.write_json(
        fits / "catalogues.json",
        {"models": models, "components": components, "tagged_activity": activity},
    )
    results = []
    for position, model in enumerate(models):
        results.append(
            study.optimize_model(
                system,
                model,
                shards,
                fold_count,
                seed + 1009 * position + sum(map(ord, system)),
                fits,
            )
        )
    frozen = {
        "schema": "ye3t_mlearn_finalist_selection_v1",
        "system": system,
        "test_split_opened": False,
        "radial_selection": str(selected["radial_path"]),
        "radial_selection_sha256": selected["radial_sha256"],
        "zbl_selection": str(selected["zbl_path"]),
        "zbl_selection_sha256": selected["zbl_sha256"],
        "radial": selected["radial"],
        "zbl": selected["zbl"],
        "training_statistics": statistics,
        "materialization": materialization,
        "tagged_activity": activity,
        "models": [
            {
                "name": result["name"],
                "feature_count": result["feature_count"],
                "family": result["family"],
                "fit_json": f"fits/{result['name']}.json",
                "fit_json_sha256": file_sha256(fits / f"{result['name']}.json"),
                "fit_npz": f"fits/{result['name']}.npz",
                "fit_npz_sha256": file_sha256(fits / f"{result['name']}.npz"),
            }
            for result in results
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    study.write_json(root / "selection_frozen.json", frozen)
    study.append_progress(
        "finalist_fit_frozen",
        system=system,
        radial_candidate=int(selected["radial"]["candidate"]),
        model_count=len(results),
        elapsed_seconds=frozen["elapsed_seconds"],
    )
    return frozen


def ordinary_bundle(system, descriptor, indices, weights, bias, model, selected):
    indices = tuple(int(value) for value in indices)
    catalogue = json.loads(
        json.dumps(descriptor.metadata["ordinary_scalar_catalogue"])
    )
    descriptor_rows = []
    variants = {}
    for descriptor_index, source_index in enumerate(indices):
        row = dict(catalogue["descriptor_rows"][source_index])
        feature_id = str(row["feature_id"])
        row["descriptor_index"] = descriptor_index
        row["variant_index"] = int(variants.get(feature_id, 0))
        variants[feature_id] = row["variant_index"] + 1
        descriptor_rows.append(row)
    catalogue["descriptor_rows"] = descriptor_rows
    return LinearACEScalarModelBundle(
        settings=descriptor.settings,
        site_basis_config=descriptor.site_basis_config,
        descriptor_specs=tuple(descriptor.descriptor_specs[index] for index in indices),
        weight=np.asarray(weights, dtype=np.float64),
        bias=float(bias),
        basis_mode=None,
        fit_method="structure_balanced_ridge_sufficient_statistics",
        fit_metadata={
            "schema": "ye3t_mlearn_cost_comparison_finalist_v1",
            "system": system,
            "model": model["name"],
            "radial": selected["radial"],
            "reference_potential": selected["zbl"],
            "selection_blind_to_test": True,
            "ordinary_scalar_catalogue": catalogue,
        },
    )


def export_ordinary_auto_plan(bundle, model_directory, yace_path, system, model):
    model_directory = Path(model_directory)
    yace_path = Path(yace_path)
    auto_directory = model_directory / "ye3t_auto"
    outputs = [
        auto_directory / name
        for name in (
            "execution_plan.json",
            "yace_function_map.json",
            "manifest.json",
            "export_report.json",
        )
    ]
    if all(path.is_file() for path in outputs):
        manifest = study.read_json(auto_directory / "manifest.json")
        report = study.read_json(auto_directory / "export_report.json")
        source_hash = file_sha256(yace_path)
        payloads = manifest.get("payloads", {})
        payload_hashes_match = all(
            payloads.get(key, {}).get("sha256")
            == file_sha256(auto_directory / payloads[key]["path"])
            for key in ("execution_plan", "yace_function_map")
            if key in payloads and "path" in payloads[key]
        ) and all(key in payloads for key in ("execution_plan", "yace_function_map"))
        if (
            manifest.get("source_yace", {}).get("sha256") == source_hash
            and report.get("source_yace_sha256") == source_hash
            and int(report.get("optimized_candidate_count", 0)) > 0
            and manifest.get("dispatch", {}).get("fallback_policy")
            == "forbid_unlisted"
            and payload_hashes_match
        ):
            return outputs
    with tempfile.TemporaryDirectory(
        prefix=".ye3t_auto_export_", dir=model_directory
    ) as temporary:
        generated = export_scalar_bundle_to_lammps(
            bundle,
            Path(temporary) / "bundle",
            elements=(system,),
            provenance={
                "schema": "ye3t_cost_comparison_symmetric_auto_plan_v1",
                "system": system,
                "model": model["name"],
                "source_yace_sha256": file_sha256(yace_path),
            },
        )
        if generated["model"].read_bytes() != yace_path.read_bytes():
            raise RuntimeError(
                f"PairPACE and PairYE3T model bytes differ for {model['name']}."
            )
        auto_directory.mkdir(parents=True, exist_ok=True)
        for name in ("execution_plan.json", "yace_function_map.json", "manifest.json", "export_report.json"):
            source = generated["manifest"].parent / name
            target = auto_directory / name
            target.write_bytes(source.read_bytes())
    return outputs


def tagged_sector_inventory(system):
    catalogue_root = study.system_root(system) / "tagged_catalogue"
    manifest_path = catalogue_root / "catalogue_manifest.json"
    manifest = study.read_json(manifest_path)
    components = []
    joint_component_indices = []
    for component in manifest["components"]:
        artifact_path = (
            catalogue_root
            / "artifacts"
            / f"{component['request_hash']}.json"
        )
        artifact = study.read_json(artifact_path)
        sector_counts = {}
        joint_label_count = 0
        for descriptor in artifact["payload"]["descriptors"]:
            label = descriptor["label"]
            block_sizes = [int(value) for value in label["block_sizes"]]
            block_kappas = [
                [int(value) for value in partition]
                for partition in label["block_kappas"]
            ]
            block_lambdas = [int(value) for value in label["block_Lambdas"]]
            sector = json.dumps(
                {
                    "block_kappas": block_kappas,
                    "block_Lambdas": block_lambdas,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            nontrivial_permutation = any(
                partition != [block_size]
                for block_size, partition in zip(block_sizes, block_kappas)
            )
            nontrivial_rotation = any(value != 0 for value in block_lambdas)
            if nontrivial_permutation and nontrivial_rotation:
                joint_label_count += 1
        if joint_label_count:
            joint_component_indices.append(int(component["component_index"]))
        components.append(
            {
                "component_index": int(component["component_index"]),
                "tensor_order_N": int(component["tensor_order_N"]),
                "n": [int(value) for value in component["n"]],
                "l": [int(value) for value in component["l"]],
                "block_sizes": [int(value) for value in component["block_sizes"]],
                "independent_physical_image_coordinate_count": int(
                    component["independent_feature_count"]
                ),
                "compiler_descriptor_sector_counts": [
                    {"sector": json.loads(sector), "count": int(count)}
                    for sector, count in sorted(sector_counts.items())
                ],
                "joint_nontrivial_permutation_rotation_label_count": int(
                    joint_label_count
                ),
                "compiler_artifact_sha256": file_sha256(artifact_path),
            }
        )
    return {
        "schema": "ye3t_tagged_sector_inventory_v1",
        "catalogue_manifest_sha256": file_sha256(manifest_path),
        "coordinate_policy": manifest["coordinate_policy"],
        "tag_count_s": int(manifest["tag_count_s"]),
        "total_independent_physical_image_coordinate_count": int(
            manifest["total_independent_feature_count"]
        ),
        "joint_nontrivial_permutation_rotation_component_indices": (
            joint_component_indices
        ),
        "components": components,
    }


def lammps_model_record(system, model, relative_directory, selected):
    prefix = system.lower()
    zbl = selected["zbl"]
    inner = float(zbl["inner_cutoff_A"])
    outer = float(zbl["outer_cutoff_A"])
    atomic_number = int(zbl["atomic_numbers"][system])
    if model["family"] == "ace":
        pair_style = f"pair_style hybrid/overlay pace product zbl {inner:.17g} {outer:.17g}"
        coefficients = [
            f"pair_coeff * * pace {relative_directory}/potential.yace {system}",
            f"pair_coeff * * zbl {atomic_number} {atomic_number}",
        ]
    else:
        pair_style = (
            "pair_style hybrid/overlay "
            "ye3t model_family tagged_cauchy block_policy direct "
            f"zbl {inner:.17g} {outer:.17g}"
        )
        coefficients = [
            f"pair_coeff * * ye3t {relative_directory}/model.ye3t.json {system}",
            f"pair_coeff * * zbl {atomic_number} {atomic_number}",
        ]
    return {
        "id": model["name"],
        "label": model["name"],
        "family": model["family"],
        "descriptor_count": int(model["feature_count"]),
        "ordinary_descriptor_count": int(model["ordinary_feature_count"]),
        "tagged_descriptor_count": int(model["tagged_feature_count"]),
        "pair_style": pair_style,
        "pair_coefficients": coefficients,
        "model_directory": relative_directory,
    }


def lammps_symmetric_record(system, model, relative_directory, selected, yace_path):
    zbl = selected["zbl"]
    inner = float(zbl["inner_cutoff_A"])
    outer = float(zbl["outer_cutoff_A"])
    atomic_number = int(zbl["atomic_numbers"][system])
    feature_count = int(model["feature_count"])
    return {
        "id": f"ye3t_symmetric_{feature_count}",
        "label": f"ye3t_symmetric_{feature_count}",
        "accuracy_model": model["name"],
        "family": "ye3t_symmetric",
        "descriptor_count": feature_count,
        "ordinary_descriptor_count": feature_count,
        "tagged_descriptor_count": 0,
        "pair_style": (
            "pair_style hybrid/overlay "
            f"ye3t plan {relative_directory}/ye3t_auto/manifest.json "
            f"block_policy auto zbl {inner:.17g} {outer:.17g}"
        ),
        "pair_coefficients": [
            f"pair_coeff * * ye3t {relative_directory}/potential.yace {system}",
            f"pair_coeff * * zbl {atomic_number} {atomic_number}",
        ],
        "model_directory": relative_directory,
        "source_model": model["name"],
        "source_yace_sha256": file_sha256(yace_path),
        "auto_plan_required": True,
        "auto_fallback_forbidden": True,
        "auto_non_direct_required": False,
    }


def input_deck(system_config, model):
    system = str(system_config["system"])
    crystal = str(system_config["crystal"])
    lattice = float(system_config["initial_lattice_constant_A"])
    mass = float(system_config["atomic_masses"][0])
    pair_coefficients = "\n".join(model["pair_coefficients"])
    return f"""# YE3T mlearn cost-comparison finalist: {model['id']}
variable dump_path index dump.{model['id']}
units metal
atom_style atomic
boundary p p p
atom_modify map yes sort 0 0.0
newton on
lattice {crystal} {lattice:.17g}
region cell block 0 2 0 2 0 2 units lattice
create_box 1 cell
create_atoms 1 box
mass 1 {mass:.17g}
reset_atoms id sort yes
group displaced id 1
displace_atoms displaced move 0.08 -0.05 0.04 units box
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
{model['pair_style']}
{pair_coefficients}
compute atom_energy all pe/atom
compute atom_stress all stress/atom NULL pair
compute pair_pressure all pressure NULL pair
thermo 1
thermo_style custom step atoms vol pe c_pair_pressure[1] c_pair_pressure[2] &
  c_pair_pressure[3] c_pair_pressure[4] c_pair_pressure[5] c_pair_pressure[6]
thermo_modify format float %.17g
run 0 post no
write_dump all custom ${{dump_path}} id type x y z c_atom_energy &
  c_atom_stress[1] c_atom_stress[2] c_atom_stress[3] &
  c_atom_stress[4] c_atom_stress[5] c_atom_stress[6] fx fy fz &
  modify sort id format float %.17g
"""


def export_system(system):
    root = study.system_root(system) / "finalist"
    frozen_path = root / "selection_frozen.json"
    frozen = study.read_json(frozen_path)
    if frozen["test_split_opened"]:
        raise RuntimeError("The frozen pre-test selection record was modified.")
    selected = selected_inputs(system)
    descriptor, _evaluator, arm = runtime(system, selected)
    index, _cache, _shards = study.load_statistics(system)
    parent_ordinary_count = int(index["ordinary_feature_count"])
    catalogues = study.read_json(root / "fits" / "catalogues.json")
    output = root / "deploy"
    output.mkdir(parents=True, exist_ok=True)
    model_records = []
    three_way_records = []
    artifact_records = []
    sector_inventory = tagged_sector_inventory(system)
    for model in catalogues["models"]:
        model_directory = output / "models" / model["name"]
        model_directory.mkdir(parents=True, exist_ok=True)
        fit_path = root / "fits" / f"{model['name']}.npz"
        with np.load(fit_path, allow_pickle=False) as data:
            coefficients = np.asarray(data["runtime_coefficients"], dtype=np.float64)
        if coefficients.shape != (int(model["feature_count"]) + 1,):
            raise RuntimeError(f"Unexpected coefficient shape for {model['name']}.")
        bias = float(coefficients[0])
        weights = coefficients[1:]
        ordinary_count = int(model["ordinary_feature_count"])
        ordinary_indices = tuple(
            int(value) for value in model["feature_indices"][:ordinary_count]
        )
        if model["family"] == "ace":
            yace_path = model_directory / "potential.yace"
            bundle = ordinary_bundle(
                system,
                descriptor,
                ordinary_indices,
                weights,
                bias,
                model,
                selected,
            )
            export_scalar_bundle_to_yace(
                bundle,
                yace_path,
                elements=(system,),
                compatibility="lammps_pace_linear_v1",
            )
            auto_paths = export_ordinary_auto_plan(
                bundle, model_directory, yace_path, system, model
            )
            artifacts = [yace_path, *auto_paths]
        else:
            ordinary_weights = weights[:ordinary_count]
            tagged_weights = weights[ordinary_count:]
            tagged_parent_indices = [
                int(value) - parent_ordinary_count
                for value in model["feature_indices"][ordinary_count:]
            ]
            full_tagged_weights = np.zeros(arm.feature_count, dtype=np.float64)
            full_tagged_weights[np.asarray(tagged_parent_indices, dtype=np.int64)] = tagged_weights
            backbone_path = model_directory / "ordinary_backbone.yace"
            tagged_path = model_directory / "tagged_correction.ye3t.json"
            composite_path = model_directory / "model.ye3t.json"
            backbone = ordinary_bundle(
                system,
                descriptor,
                ordinary_indices,
                ordinary_weights,
                bias,
                model,
                selected,
            )
            export_scalar_bundle_to_yace(
                backbone,
                backbone_path,
                elements=(system,),
                compatibility="lammps_pace_linear_v1",
            )
            tagged_model = arm_lammps_model(
                arm,
                full_tagged_weights,
                {system: 0.0},
                {
                    "lmbda": float(selected["radial"]["radial_lambda"]),
                    "cutoff_width": float(selected["radial"]["cutoff_width_A"]),
                },
                float(selected["radial"]["cutoff_A"]),
                (system,),
            )
            export_tagged_model(
                tagged_path,
                tagged_model,
                metrics={
                    "schema": "ye3t_mlearn_cost_comparison_finalist_v1",
                    "system": system,
                    "model": model["name"],
                    "active_tagged_features": int(model["tagged_feature_count"]),
                    "reference_potential": selected["zbl"],
                },
            )
            loaded = load_tagged_model(tagged_path)
            if not np.array_equal(
                loaded.beta.detach().cpu().numpy(), full_tagged_weights
            ):
                raise RuntimeError(f"Tagged export did not round-trip for {model['name']}.")
            export_tagged_composite_model(
                composite_path,
                backbone_path,
                tagged_path,
                sector_inventory=sector_inventory,
                metadata={
                    "schema": "ye3t_mlearn_cost_comparison_finalist_v1",
                    "system": system,
                    "model": model["name"],
                    "feature_count": int(model["feature_count"]),
                    "ordinary_feature_count": int(model["ordinary_feature_count"]),
                    "tagged_feature_count": int(model["tagged_feature_count"]),
                    "fit_is_joint": True,
                    "pair_style_dependency": "ye3t_only",
                },
            )
            artifacts = [backbone_path, tagged_path, composite_path]
        relative_directory = f"models/{model['name']}"
        record = lammps_model_record(system, model, relative_directory, selected)
        model_records.append(record)
        if model["family"] == "ace":
            pace_record = dict(record)
            pace_record["family"] = "pace_product"
            pace_record["accuracy_model"] = model["name"]
            pace_record["evaluator_request"] = "product"
            three_way_records.append(pace_record)
            three_way_records.append(
                lammps_symmetric_record(
                    system, model, relative_directory, selected, yace_path
                )
            )
        else:
            mixed_record = dict(record)
            mixed_record["accuracy_model"] = model["name"]
            mixed_record["family"] = (
                "ye3t_mixed_augmented"
                if model["family"] == "tagged_ye3t_augmented"
                else "ye3t_mixed"
            )
            mixed_record["evaluator_request"] = "direct_compiled_tagged"
            three_way_records.append(mixed_record)
        input_path = output / f"in.{system.lower()}_{model['name']}"
        input_path.write_text(
            input_deck(
                study.read_system_config(system),
                record,
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema": "ye3t_mlearn_lammps_model_manifest_v1",
            "system": system,
            "model": model,
            "source_fit": str(fit_path),
            "source_fit_sha256": file_sha256(fit_path),
            "radial": selected["radial"],
            "reference_potential": selected["zbl"],
            "artifacts": [
                {
                    "path": str(path.relative_to(model_directory)),
                    "sha256": file_sha256(path),
                }
                for path in artifacts
            ],
        }
        study.write_json(model_directory / "model_manifest.json", manifest)
        artifact_records.append(manifest)
    matrix = {
        "schema": "ye3t_lammps_linear_model_matrix_v2",
        "system": system,
        "models": model_records,
    }
    study.write_json(output / "lammps_models.json", matrix)
    three_way_matrix = {
        "schema": "ye3t_lammps_three_way_evaluator_matrix_v1",
        "system": system,
        "models": three_way_records,
        "qualification": {
            "pace_evaluator": "product",
            "same_yace_bytes_required": True,
            "ye3t_auto_plan_required": True,
            "ye3t_auto_fallback_forbidden": True,
            "ye3t_auto_non_direct_required": False,
        },
    }
    study.write_json(output / "lammps_models.three_way_auto.json", three_way_matrix)
    deployment = {
        "schema": "ye3t_mlearn_finalist_deployment_v1",
        "system": system,
        "selection_frozen_sha256": file_sha256(frozen_path),
        "radial": selected["radial"],
        "reference_potential": selected["zbl"],
        "models": artifact_records,
    }
    study.write_json(output / "deployment_manifest.json", deployment)
    study.append_progress(
        "finalist_export_complete",
        system=system,
        model_count=len(model_records),
        deployment_manifest=str(output / "deployment_manifest.json"),
    )
    return deployment


def test_system(system):
    root = study.system_root(system) / "finalist"
    frozen_path = root / "selection_frozen.json"
    frozen = study.read_json(frozen_path)
    selected = selected_inputs(system)
    materialization = materialize_split(system, "test", selected)
    statistics, shards = residual_statistics(system, "test", selected, 1, 0, root)
    total = sum_linear_statistics(shards.values())
    by_group = {}
    for (group, _fold), shard in shards.items():
        by_group[group] = shard
    catalogues = study.read_json(root / "fits" / "catalogues.json")
    rows = []
    group_rows = []
    for model in catalogues["models"]:
        with np.load(root / "fits" / f"{model['name']}.npz", allow_pickle=False) as data:
            coefficients = np.asarray(data["runtime_coefficients"], dtype=np.float64)
        selected_total = select_linear_statistics(total, model["feature_indices"])
        metric = score_linear_statistics(selected_total, coefficients)
        rows.append(
            {
                "model": model["name"],
                "family": model["family"],
                "feature_count": int(model["feature_count"]),
                "energy_rmse_eV_per_atom": metric["energy_rmse_eV_per_atom"],
                "force_rmse_eV_per_A": metric["force_rmse_eV_per_A"],
            }
        )
        for group, shard in sorted(by_group.items()):
            group_metric = score_linear_statistics(
                select_linear_statistics(shard, model["feature_indices"]),
                coefficients,
            )
            group_rows.append(
                {
                    "model": model["name"],
                    "group": group,
                    "energy_rmse_eV_per_atom": group_metric[
                        "energy_rmse_eV_per_atom"
                    ],
                    "force_rmse_eV_per_A": group_metric["force_rmse_eV_per_A"],
                }
            )
    with (root / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (root / "test_group_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(group_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(group_rows)
    summary = {
        "schema": "ye3t_mlearn_finalist_test_v1",
        "system": system,
        "selection_frozen_sha256": file_sha256(frozen_path),
        "test_opened_after_selection_frozen": True,
        "materialization": materialization,
        "statistics": statistics,
        "models": rows,
        "group_metrics": group_rows,
    }
    study.write_json(root / "test_summary.json", summary)
    study.append_progress(
        "finalist_test_complete",
        system=system,
        model_count=len(rows),
        test_summary=str(root / "test_summary.json"),
    )
    print(json.dumps({"system": system, "models": rows}, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=study.SYSTEMS, default=study.SYSTEMS)
    parser.add_argument("--stage", choices=("fit", "export", "test", "all"), default="all")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    for system in args.systems:
        if args.stage in {"fit", "all"}:
            fit_system(system, args.fold_count, args.seed)
        if args.stage in {"export", "all"}:
            export_system(system)
        if args.stage in {"test", "all"}:
            test_system(system)
        RUNTIME.clear()
        radial.release_row_memory()


if __name__ == "__main__":
    main()
