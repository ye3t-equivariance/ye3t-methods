#!/usr/bin/env python3
"""Run the configurable linear YE3T cost-comparison publication workflow."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

from ye3t_methods.atomistic.ace.catalogue_selection import (
    linear_catalogue_preflight,
    resolve_ordinary_scalar_catalogues,
    resolve_tagged_content_schedule,
)


HERE = Path(__file__).resolve().parent
WORKFLOW = HERE / "workflow"
SYSTEM_NAMES = ("Li", "Mo", "Cu", "Ni", "Si", "Ge")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_workflow_base(config_path, config):
    requested = Path(config["runtime"]["workflow_root"])
    if not requested.is_absolute():
        requested = (Path(config_path).resolve().parent / requested).resolve()
    return requested


def resolve_workflow_root(config_path, config, system):
    return resolve_workflow_base(config_path, config) / str(system["system"])


def resolve_system_root(config_path, config):
    requested = config["runtime"].get("system_config_root")
    if requested is None:
        return (Path(config_path).resolve().parent / "systems").resolve()
    requested = Path(requested)
    if not requested.is_absolute():
        requested = (Path(config_path).resolve().parent / requested).resolve()
    return requested


def selected_cutoff(config, system):
    value = system.get("selected_cutoff_A")
    if value is None:
        value = config["basis"]["radial"].get("matched_control_cutoff_A")
    if value is None:
        raise ValueError(
            f"{system['system']} has no selected cutoff and no matched-control cutoff."
        )
    return float(value)


def tagged_schedule(config_path, config):
    tagged = config["basis"]["tagged"]
    mode = str(tagged.get("component_schedule_mode", "fixed_file")).strip().lower()
    if mode == "generated":
        return resolve_tagged_content_schedule(config["basis"])
    if mode != "fixed_file":
        raise ValueError(
            "basis.tagged.component_schedule_mode must be 'generated' or "
            f"'fixed_file'; received {mode!r}."
        )
    relative = tagged.get("component_schedule")
    if not relative:
        raise ValueError(
            "fixed_file tagged catalogues require basis.tagged.component_schedule."
        )
    path = Path(relative)
    if not path.is_absolute():
        path = (Path(config_path).resolve().parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    schedule = read_json(path)
    manifest = {
        "schema": "ye3t_fixed_tagged_content_manifest_v1",
        "coefficient_compilation_performed": False,
        "source_path": str(path),
        "source_sha256": sha256(path),
        "tag_count_s": int(schedule["tag_count_s"]),
        "role_bindings": schedule["role_bindings"],
        "selected_fixed_contents": len(schedule["components"]),
    }
    return schedule, manifest


def append_progress(workflow_base, record):
    path = Path(workflow_base) / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def process_tree_rss(process):
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError(
            "The publication workflow RSS guard requires the `examples` extra "
            "(or `pip install psutil`)."
        ) from error
    try:
        root = psutil.Process(process.pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for child in processes:
        try:
            total += int(child.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return total


def terminate_process_tree(process):
    try:
        import psutil
    except ImportError:
        process.kill()
        return
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in reversed(children):
            child.terminate()
        root.terminate()
        _gone, alive = psutil.wait_procs([*children, root], timeout=5.0)
        for child in alive:
            child.kill()
    except psutil.Error:
        process.kill()


def run_driver(
    script,
    arguments,
    config_path,
    workflow_base,
    timeout_seconds,
    memory_limit_gib,
    dry_run=False,
):
    script = WORKFLOW / script
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [sys.executable, str(script), *[str(value) for value in arguments]]
    logs = Path(workflow_base) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"_{time.time_ns()}"
    log_path = logs / f"{stamp}_{script.stem}.log"
    record = {
        "event": "driver_start",
        "script": script.name,
        "command": command,
        "started_epoch": time.time(),
        "timeout_seconds": int(timeout_seconds),
        "memory_limit_gib": float(memory_limit_gib),
        "log": str(log_path),
    }
    append_progress(workflow_base, record)
    print("RUN", " ".join(command), flush=True)
    if dry_run:
        record.update(
            {
                "event": "driver_dry_run",
                "finished_epoch": time.time(),
                "returncode": 0,
            }
        )
        append_progress(workflow_base, record)
        return record
    environment = os.environ.copy()
    environment.update(
        {
            "YE3T_COST_CONFIG": str(Path(config_path).resolve()),
            "YE3T_COST_PUBLIC_ROOT": str(HERE),
            "YE3T_COST_WORKFLOW_ROOT": str(Path(workflow_base).resolve()),
            "YE3T_COST_SYSTEM_ROOT": str(
                resolve_system_root(config_path, read_json(config_path))
            ),
        }
    )
    started = time.perf_counter()
    memory_limit_bytes = int(float(memory_limit_gib) * 1024**3)
    peak_rss = 0
    stop_reason = None
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.Popen(
            command,
            cwd=WORKFLOW,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while completed.poll() is None:
            rss = process_tree_rss(completed)
            peak_rss = max(peak_rss, rss)
            elapsed = time.perf_counter() - started
            if rss > memory_limit_bytes:
                stop_reason = (
                    f"resident process-tree memory {rss} exceeded "
                    f"{memory_limit_bytes} bytes"
                )
                terminate_process_tree(completed)
                break
            if elapsed > float(timeout_seconds):
                stop_reason = f"elapsed time exceeded {timeout_seconds} seconds"
                terminate_process_tree(completed)
                break
            time.sleep(0.25)
        returncode = completed.wait()
    if stop_reason is not None:
        record.update(
            {
                "event": "driver_guard_stop",
                "finished_epoch": time.time(),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_bytes": peak_rss,
                "stop_reason": stop_reason,
                "returncode": int(returncode),
            }
        )
        append_progress(workflow_base, record)
        raise RuntimeError(f"{script.name} stopped by resource guard: {stop_reason}")
    record.update(
        {
            "event": "driver_finish",
            "finished_epoch": time.time(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss,
            "returncode": int(returncode),
        }
    )
    append_progress(workflow_base, record)
    if returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(
            f"{script.name} failed with code {returncode}; "
            f"log={log_path}\n{tail}"
        )
    return record


def convert_record(record, split, index):
    structure = record["structure"]
    sites = structure["sites"]
    atoms = Atoms(
        [str(site["species"][0]["element"]) for site in sites],
        scaled_positions=np.asarray([site["abc"] for site in sites], dtype=np.float64),
        cell=np.asarray(structure["lattice"]["matrix"], dtype=np.float64),
        pbc=True,
    )
    forces = np.asarray(record["outputs"]["forces"], dtype=np.float64)
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"{split}[{index}] force shape {forces.shape} is invalid.")
    atoms.info.update(
        {
            "energy": float(record["outputs"]["energy"]),
            "config_type": str(record["group"]),
            "mlearn_group": str(record["group"]),
            "mlearn_tag": str(record["tag"]),
            "source_split": split,
            "source_index": int(index),
        }
    )
    atoms.arrays["forces"] = forces
    return atoms


def cutoff_diagnostics(frames, candidates):
    rows = []
    for cutoff in candidates:
        coordination = []
        distances = []
        for atoms in frames:
            centers, separation = neighbor_list("id", atoms, float(cutoff))
            coordination.extend(
                np.bincount(centers, minlength=len(atoms)).astype(int).tolist()
            )
            distances.extend(np.asarray(separation, dtype=np.float64).tolist())
        coordination = np.asarray(coordination, dtype=np.float64)
        distances = np.asarray(distances, dtype=np.float64)
        rows.append(
            {
                "cutoff_A": float(cutoff),
                "training_atoms": int(coordination.size),
                "directed_neighbor_edges": int(distances.size),
                "coordination_mean": float(np.mean(coordination)),
                "coordination_median": float(np.median(coordination)),
                "coordination_p95": float(np.quantile(coordination, 0.95)),
                "coordination_max": int(np.max(coordination)),
                "minimum_distance_A": (
                    None if distances.size == 0 else float(np.min(distances))
                ),
                "distance_p95_A": (
                    None
                    if distances.size == 0
                    else float(np.quantile(distances, 0.95))
                ),
            }
        )
    return {
        "selection_data": "published_training_structures_only",
        "target_values_used": False,
        "purpose": "cost_and_coordination_diagnostic_not_test_tuned_selection",
        "candidates": rows,
    }


def run_prepare(dataset_root, config, system, output):
    dataset_root = Path(dataset_root).resolve()
    data_output = output / "data"
    data_output.mkdir(parents=True, exist_ok=True)
    name = str(system["system"])
    prefix = name.lower()
    upstream_sources = {
        split: dataset_root
        / "data"
        / system["dataset_subdirectory"]
        / f"{split}.json"
        for split in ("training", "test")
    }
    portable_manifest_path = dataset_root / "MANIFEST.json"
    portable_element_root = dataset_root / system["dataset_subdirectory"]
    portable_split_path = portable_element_root / "moment_star_split.json"
    portable_manifest = (
        read_json(portable_manifest_path)
        if portable_manifest_path.is_file()
        else None
    )
    portable_entry = None if portable_manifest is None else portable_manifest.get(name)
    portable_combined_path = (
        None
        if portable_entry is None
        else portable_element_root / str(portable_entry["path"])
    )
    frames_by_split = {}
    split_sources = {}
    supplied_inner_indices = None
    if all(path.is_file() for path in upstream_sources.values()):
        revision = subprocess.check_output(
            ["git", "-C", str(dataset_root), "rev-parse", "HEAD"], text=True
        ).strip()
        source_layout = "upstream_mlearn_json"
        for split, source_path in upstream_sources.items():
            records = read_json(source_path)
            frames_by_split[split] = [
                convert_record(record, split, index)
                for index, record in enumerate(records)
            ]
            split_sources[split] = {
                "source": str(source_path),
                "source_sha256": sha256(source_path),
            }
    elif (
        portable_entry is not None
        and portable_combined_path.is_file()
        and portable_split_path.is_file()
    ):
        source_layout = "tracked_portable_extxyz"
        actual_sha256 = sha256(portable_combined_path)
        if actual_sha256 != str(portable_entry["sha256"]):
            raise ValueError(
                f"Portable {name} dataset hash mismatch: expected "
                f"{portable_entry['sha256']}, found {actual_sha256}."
            )
        frames = list(read(portable_combined_path, index=":"))
        if len(frames) != int(portable_entry["frame_count"]):
            raise ValueError(
                f"Portable {name} frame count mismatch: expected "
                f"{portable_entry['frame_count']}, found {len(frames)}."
            )
        portable_split = read_json(portable_split_path)
        if str(portable_split.get("element")) != name:
            raise ValueError(
                f"Portable split {portable_split_path} is not for {name}."
            )
        split_dataset = portable_split.get("dataset", {})
        if split_dataset.get("sha256") != actual_sha256:
            raise ValueError(
                f"Portable split {portable_split_path} does not bind the "
                f"tracked {name} dataset hash."
            )
        original_indices = {
            key: [int(index) for index in portable_split["indices"][key]]
            for key in ("train", "validation", "test")
        }
        flattened = [
            index
            for key in ("train", "validation", "test")
            for index in original_indices[key]
        ]
        if sorted(flattened) != list(range(len(frames))):
            raise ValueError(
                f"Portable {name} split must contain every frame exactly once."
            )
        published_training = sorted(
            original_indices["train"] + original_indices["validation"]
        )
        published_test = sorted(original_indices["test"])
        frames_by_split = {
            "training": [frames[index] for index in published_training],
            "test": [frames[index] for index in published_test],
        }
        old_to_new = {
            old: new
            for new, old in enumerate(published_training + published_test)
        }
        source_seed = int(
            portable_split.get("inner_split", {}).get(
                "linear_split_seed", config["validation"]["inner_seed"]
            )
        )
        if source_seed == int(config["validation"]["inner_seed"]):
            supplied_inner_indices = {
                key: sorted(old_to_new[index] for index in original_indices[key])
                for key in ("train", "validation", "test")
            }
        revision = "portable-snapshot:" + sha256(portable_manifest_path)
        common_source = {
            "source": str(portable_combined_path),
            "source_sha256": actual_sha256,
            "split_manifest": str(portable_split_path),
            "split_manifest_sha256": sha256(portable_split_path),
        }
        split_sources = {
            "training": {**common_source, "source_indices": published_training},
            "test": {**common_source, "source_indices": published_test},
        }
    else:
        raise FileNotFoundError(
            "mlearn dataset not found. Expected either upstream JSON files at "
            f"{upstream_sources['training']} and {upstream_sources['test']}, "
            "or the tracked portable MANIFEST.json/extxyz/split layout under "
            f"{dataset_root}."
        )
    manifest = {
        "schema": "ye3t_mlearn_element_dataset_v1",
        "system": name,
        "source_repository": "https://github.com/materialyzeai/mlearn",
        "source_revision": revision,
        "source_layout": source_layout,
        "license": "BSD-3-Clause",
        "stress_policy": "source virial_stress retained in JSON but not fitted",
        "splits": {},
    }
    for split, frames in frames_by_split.items():
        destination = data_output / f"{prefix}_{split}.xyz"
        write(destination, frames, format="extxyz")
        manifest["splits"][split] = {
            **split_sources[split],
            "converted": destination.name,
            "converted_sha256": sha256(destination),
            "structures": len(frames),
            "atoms": int(sum(len(frame) for frame in frames)),
            "groups": dict(
                sorted(Counter(frame.info["config_type"] for frame in frames).items())
            ),
        }
    combined = frames_by_split["training"] + frames_by_split["test"]
    combined_path = data_output / f"{prefix}_all.xyz"
    write(combined_path, combined, format="extxyz")
    seed = int(config["validation"]["inner_seed"])
    if supplied_inner_indices is None:
        rng = np.random.default_rng(seed)
        groups = {}
        for index, atoms in enumerate(frames_by_split["training"]):
            groups.setdefault(atoms.info["config_type"], []).append(index)
        validation = []
        for indices in groups.values():
            shuffled = np.asarray(indices, dtype=np.int64)
            rng.shuffle(shuffled)
            validation.extend(
                shuffled[: max(1, int(round(0.2 * len(shuffled))))].tolist()
            )
        validation = sorted(validation)
        held_out = set(validation)
        training = [
            index
            for index in range(len(frames_by_split["training"]))
            if index not in held_out
        ]
        test = list(range(len(frames_by_split["training"]), len(combined)))
    else:
        training = supplied_inner_indices["train"]
        validation = supplied_inner_indices["validation"]
        test = supplied_inner_indices["test"]
    split = {
        "schema": "ye3t_mlearn_element_inner_split_v1",
        "system": name,
        "selection_blind_to_test_targets": True,
        "seed": seed,
        "policy": "group_stratified_80_20_inside_published_training_file",
        "dataset": {
            "path": str(combined_path),
            "frame_count": len(combined),
            "sha256": sha256(combined_path),
        },
        "indices": {"train": training, "validation": validation, "test": test},
    }
    split_path = data_output / "inner_split.json"
    write_json(split_path, split)
    manifest["combined"] = {
        "converted": combined_path.name,
        "converted_sha256": sha256(combined_path),
        "structures": len(combined),
        "atoms": int(sum(len(frame) for frame in combined)),
    }
    manifest["inner_split"] = {
        "manifest": split_path.name,
        "manifest_sha256": sha256(split_path),
        "train_structures": len(training),
        "validation_structures": len(validation),
        "test_structures": len(test),
    }
    manifest["cutoff_diagnostics"] = cutoff_diagnostics(
        frames_by_split["training"], system["cutoff_candidates_A"]
    )
    write_json(data_output / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def validate_system(system):
    required = {
        "schema",
        "system",
        "species",
        "atomic_numbers",
        "atomic_masses",
        "crystal",
        "cutoff_candidates_A",
        "selected_cutoff_A",
        "dataset_subdirectory",
    }
    extras = set(system) - required - {
        "initial_lattice_constant_A",
        "property_reference",
        "stability_temperature_reference",
    }
    missing = required - set(system)
    if missing or extras:
        raise ValueError(
            f"Invalid system config; missing={sorted(missing)}, extras={sorted(extras)}"
        )
    if len(system["species"]) != len(system["atomic_numbers"]):
        raise ValueError("species and atomic_numbers must have equal length.")
    if len(system["species"]) != len(system["atomic_masses"]):
        raise ValueError("species and atomic_masses must have equal length.")


def run_preflight(config_path, system_path, config, system, output):
    validate_system(system)
    report = linear_catalogue_preflight(config["basis"])
    _tagged_source, tagged_report = tagged_schedule(config_path, config)
    matched_cutoff = config["basis"]["radial"].get(
        "matched_control_cutoff_A"
    )
    payload = {
        "schema": "ye3t_mlearn_linear_cost_preflight_v1",
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": sha256(config_path),
        },
        "system_config": {
            "path": str(Path(system_path).resolve()),
            "sha256": sha256(system_path),
        },
        "system": system,
        "catalogue": report,
        "tagged_catalogue": tagged_report,
        "selected_cutoff_ready": system["selected_cutoff_A"] is not None,
        "matched_control_cutoff_A": matched_cutoff,
        "fit_ready": (
            system["selected_cutoff_A"] is not None or matched_cutoff is not None
        ),
    }
    write_json(output / "preflight" / "preflight.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def run_catalogue(config_path, config, system, output):
    system_name = str(system["system"]).lower()
    source, manifest = resolve_ordinary_scalar_catalogues(
        config["basis"],
        catalogue_id=f"{system_name}_generated_linear_ace",
    )
    catalogue_dir = output / "catalogues"
    source_path = catalogue_dir / "ordinary_catalogue_source.json"
    manifest_path = catalogue_dir / "ordinary_catalogue_manifest.json"
    write_json(source_path, source)
    manifest["source_path"] = str(source_path)
    write_json(manifest_path, manifest)
    tagged_source, tagged_manifest = tagged_schedule(config_path, config)
    tagged_source_path = catalogue_dir / "tagged_component_schedule.json"
    tagged_manifest_path = catalogue_dir / "tagged_component_manifest.json"
    write_json(tagged_source_path, tagged_source)
    tagged_manifest["source_path"] = str(tagged_source_path)
    write_json(tagged_manifest_path, tagged_manifest)
    manifest["tagged_component_schedule"] = str(tagged_source_path)
    manifest["tagged_component_schedule_sha256"] = sha256(tagged_source_path)
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def selected_systems(args, config):
    if args.system and args.systems:
        raise ValueError("Use either --system paths or --systems names, not both.")
    if args.system:
        paths = tuple(Path(value).resolve() for value in args.system)
    else:
        names = tuple(args.systems) if args.systems else tuple(
            config["runtime"].get("default_systems", ("Si",))
        )
        if not names or any(name not in SYSTEM_NAMES for name in names):
            raise ValueError(
                f"runtime.default_systems must select names from {SYSTEM_NAMES}."
            )
        paths = tuple((HERE / "systems" / f"{name}.json").resolve() for name in names)
    records = []
    seen = set()
    for path in paths:
        system = read_json(path)
        validate_system(system)
        name = str(system["system"])
        if name in seen:
            raise ValueError(f"System {name} was selected more than once.")
        seen.add(name)
        records.append((path, system))
    return tuple(records)


def require_executable(path, label):
    if path is None:
        raise ValueError(f"runtime.{label}_executable is required for this stage.")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def driver_options(args, config_path, workflow_base):
    return {
        "config_path": config_path,
        "workflow_base": workflow_base,
        "timeout_seconds": args.stage_timeout_seconds,
        "memory_limit_gib": args.memory_limit_gib,
        "dry_run": args.dry_run,
    }


def run_stage(stage, args, config_path, config, records, workflow_base):
    names = tuple(str(system["system"]) for _path, system in records)
    fold_count = int(config["validation"]["inner_fold_count"])
    seed = int(config["validation"]["inner_seed"])
    targets = [str(value) for value in config["basis"]["target_descriptor_counts"]]
    options = driver_options(args, config_path, workflow_base)

    if stage in {"preflight", "prepare", "catalogue"}:
        for system_path, system in records:
            output = resolve_workflow_root(config_path, config, system)
            run_preflight(config_path, system_path, config, system, output)
            if stage == "prepare":
                run_prepare(args.dataset_root, config, system, output)
            elif stage == "catalogue":
                run_catalogue(config_path, config, system, output)
                cutoff = selected_cutoff(config, system)
                run_driver(
                    "build_ordinary_rows.py",
                    [
                        "--stage",
                        "catalogue",
                        "--system",
                        system["system"],
                        "--cutoff",
                        cutoff,
                        "--seed",
                        seed,
                        "--targets",
                        *targets,
                        "--source",
                        output / "catalogues" / "ordinary_catalogue_source.json",
                        "--data",
                        output / "data",
                        "--output",
                        output / "generated_ordinary_controls",
                    ],
                    **options,
                )
                run_driver(
                    "build_tagged_catalogue.py",
                    [
                        "--system",
                        system["system"],
                        "--cutoff",
                        cutoff,
                        "--source",
                        output / "catalogues" / "tagged_component_schedule.json",
                        "--output",
                        output / "tagged_catalogue",
                    ],
                    **options,
                )
        return

    if stage == "build-cache":
        for _system_path, system in records:
            output = resolve_workflow_root(config_path, config, system)
            cutoff = selected_cutoff(config, system)
            run_driver(
                "build_ordinary_rows.py",
                [
                    "--stage",
                    "rows",
                    "--system",
                    system["system"],
                    "--cutoff",
                    cutoff,
                    "--seed",
                    seed,
                    "--targets",
                    *targets,
                    "--source",
                    output / "catalogues" / "ordinary_catalogue_source.json",
                    "--data",
                    output / "data",
                    "--output",
                    output / "generated_ordinary_controls",
                ],
                **options,
            )
            run_driver(
                "build_rows.py",
                [
                    "--stage",
                    "rows",
                    "--kind",
                    "tagged",
                    "--system",
                    system["system"],
                    "--cutoff",
                    cutoff,
                    "--seed",
                    seed,
                    "--data",
                    output / "data",
                    "--catalogue",
                    output / "tagged_catalogue",
                    "--ordinary-application",
                    output
                    / "generated_ordinary_controls"
                    / "catalogue_application.json",
                    "--output",
                    output / "screen",
                ],
                **options,
            )
        run_driver(
            "optimize_cached.py",
            [
                "--stage",
                "statistics",
                "--systems",
                *names,
                "--fold-count",
                fold_count,
                "--seed",
                seed,
            ],
            **options,
        )
        return

    if stage == "fit":
        run_driver(
            "optimize_cached.py",
            ["--stage", "optimize", "--systems", *names, "--seed", seed],
            **options,
        )
        run_driver(
            "verify_cached_statistics.py",
            ["--systems", *names],
            **options,
        )
        if args.fit_scope == "publication":
            run_driver(
                "radial_screen.py",
                [
                    "--systems",
                    *names,
                    "--fold-count",
                    fold_count,
                    "--mode",
                    "force",
                ],
                **options,
            )
            run_driver("select_radial.py", ["--systems", *names], **options)
            lammps = require_executable(args.lammps, "lammps")
            run_driver(
                "zbl_screen.py",
                [
                    "--systems",
                    *names,
                    "--lammps",
                    lammps,
                    "--fold-count",
                    fold_count,
                    "--seed",
                    seed,
                ],
                **options,
            )
        return

    if stage == "export":
        run_driver(
            "finalize_models.py",
            [
                "--systems",
                *names,
                "--stage",
                "all",
                "--fold-count",
                fold_count,
                "--seed",
                seed,
            ],
            **options,
        )
        if not args.skip_lifted:
            run_driver(
                "lifted_finalists.py",
                [
                    "--systems",
                    *names,
                    "--timeout-seconds",
                    args.stage_timeout_seconds,
                ],
                **options,
            )
        return

    if stage == "validate":
        lammps = require_executable(args.lammps, "lammps")
        mpiexec = require_executable(args.mpiexec, "mpiexec")
        run_driver(
            "validate_finalists.py",
            [
                "--systems",
                *names,
                "--lammps",
                lammps,
                "--mpiexec",
                mpiexec,
                "--mpi-ranks",
                args.mpi_ranks,
            ],
            **options,
        )
        return

    if stage == "properties":
        lammps = require_executable(args.lammps, "lammps")
        run_driver(
            "property_finalists.py",
            ["--systems", *names, "--lammps", lammps],
            **options,
        )
        return

    if stage == "nve":
        lammps = require_executable(args.lammps, "lammps")
        run_driver(
            "stability_finalists.py",
            ["--systems", *names, "--lammps", lammps, "--stage", "all"],
            **options,
        )
        return

    if stage == "benchmark":
        lammps = require_executable(args.lammps, "lammps")
        mpiexec = require_executable(args.mpiexec, "mpiexec")
        run_driver(
            "benchmark_finalists.py",
            [
                "--systems",
                *names,
                "--lammps",
                lammps,
                "--mpiexec",
                mpiexec,
                "--ranks",
                1,
                args.mpi_ranks,
                "--model-matrix",
                "lammps_models.three_way_auto.json",
                "--output-label",
                "timing_three_way_auto",
            ],
            **options,
        )
        return

    if stage == "render":
        output = (
            Path(args.results_output).resolve()
            if args.results_output is not None
            else Path(workflow_base) / "publication_final"
        )
        render_arguments = ["--systems", *names, "--output", output]
        if args.record_visual_validation:
            render_arguments.append("--record-visual-validation")
        run_driver("render_final_study.py", render_arguments, **options)
        return
    raise ValueError(f"Unsupported stage {stage!r}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "config.json",
        help="editable paper workflow config (default: adjacent config.json)",
    )
    parser.add_argument(
        "--system", type=Path, action="append",
        help="path to a custom systems/*.json record; may be repeated",
    )
    parser.add_argument(
        "--systems", nargs="+", choices=SYSTEM_NAMES,
        help="published elements to process (default: runtime.default_systems)",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "prepare",
            "catalogue",
            "build-cache",
            "fit",
            "export",
            "validate",
            "properties",
            "nve",
            "benchmark",
            "render",
            "all",
        ),
        default=None,
        help="workflow stage (default: runtime.default_stage in config.json)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=HERE.parents[1] / "data" / "mlearn",
        help="mlearn snapshot directory (default: bundled examples/data/mlearn)",
    )
    parser.add_argument("--lammps", type=Path)
    parser.add_argument("--mpiexec", type=Path)
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument(
        "--fit-scope", choices=("fixed", "publication"), default="publication"
    )
    parser.add_argument("--skip-lifted", action="store_true")
    parser.add_argument("--results-output", type=Path)
    parser.add_argument("--record-visual-validation", action="store_true")
    parser.add_argument("--stage-timeout-seconds", type=int, default=7200)
    parser.add_argument("--memory-limit-gib", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.mpi_ranks < 2:
        raise ValueError("--mpi-ranks must be at least two.")
    if args.stage_timeout_seconds <= 0 or args.memory_limit_gib <= 0.0:
        raise ValueError("Stage timeout and memory limit must be positive.")
    config_path = args.config.resolve()
    config = read_json(config_path)
    if args.stage is None:
        args.stage = str(config["runtime"].get("default_stage", "preflight"))
        if args.stage not in (
            "preflight", "prepare", "catalogue", "build-cache", "fit", "export",
            "validate", "properties", "nve", "benchmark", "render", "all",
        ):
            raise ValueError(f"Unsupported runtime.default_stage {args.stage!r}.")
    records = selected_systems(args, config)
    workflow_base = resolve_workflow_base(config_path, config)
    workflow_base.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        stages = (
            "prepare",
            "catalogue",
            "build-cache",
            "fit",
            "export",
            "validate",
            "properties",
            "nve",
            "benchmark",
            "render",
        )
    else:
        stages = (args.stage,)
    for stage in stages:
        run_stage(stage, args, config_path, config, records, workflow_base)


if __name__ == "__main__":
    main()
