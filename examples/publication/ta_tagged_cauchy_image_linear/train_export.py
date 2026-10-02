"""Run the bounded tagged-Cauchy Ta learning diagnostic and export V3 models."""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import queue
import subprocess
import time
from pathlib import Path

import numpy as np

from ye3t_ace import (
    YE3TDescriptors,
    YE3TModel,
    YE3TRepresentation,
    load_xyz_structures,
)
from ye3t_ace.reference_potentials import (
    evaluate_lammps_zbl_reference,
    lammps_zbl_reference_config,
)
from ye3t_ace.tagged_cauchy_image import (
    export_tagged_cauchy_image_model,
    load_tagged_cauchy_image_model,
)
from ye3t_ace.tagged_cauchy_image_fit import (
    build_tagged_cauchy_image_normal_equations,
    score_tagged_cauchy_image_model,
    tagged_cauchy_reference_target_metadata,
)


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXAMPLE_DIRECTORY / "config.json"


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(config_directory, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_directory / path).resolve()


def _automatic_workflow_root():
    configured = os.environ.get("YE3T_WORKFLOW_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "ye3t-workflows").resolve()


def _automatic_cache_root():
    configured = os.environ.get("YE3T_CACHE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "ye3t" / "ta_tagged_cauchy_image_linear").resolve()


def _repository_for_module(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_revision(path):
    if path is None:
        return {"commit": None, "dirty": None, "status": []}
    path = Path(path)
    if not path.is_dir():
        return {"commit": None, "dirty": None, "status": []}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None, "status": []}
    status_lines = status.stdout.splitlines() if status.returncode == 0 else []
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status_lines) if status.returncode == 0 else None,
        "status": status_lines,
    }


def _installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_config(config):
    required = (
        "metadata",
        "basis",
        "representation",
        "runtime",
        "model",
        "targets",
        "validation",
        "extensions",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config is missing required sections: {missing}.")
    if config["metadata"].get("config_schema") != "ye3t_example_config_v1":
        raise ValueError("metadata.config_schema must be ye3t_example_config_v1.")
    basis = config["basis"]
    if basis.get("type") != "tagged_cauchy_image":
        raise ValueError("basis.type must be tagged_cauchy_image.")
    species = tuple(str(value) for value in basis.get("species", ()))
    if not species or len(set(species)) != len(species):
        raise ValueError("basis.species must be nonempty and unique.")
    catalogue = basis["descriptor_catalogue"]
    if int(catalogue["tensor_order"]) != 4 or int(basis["angular"]["l"]) != 1:
        raise ValueError("This diagnostic is certified only for N=4 and homogeneous l=1.")
    arms = tuple(catalogue.get("arms", ()))
    names = [str(arm["name"]) for arm in arms]
    if not names or len(set(names)) != len(names):
        raise ValueError("Descriptor arm names must be nonempty and unique.")
    for arm in arms:
        name = str(arm["name"])
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Arm name {name!r} is not a safe artifact name.")
        selected = tuple(int(value) for value in arm["selected_raw_tag_counts"])
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= {0, 1, 2}:
            raise ValueError(f"Arm {arm['name']!r} has invalid raw tag counts.")
    alphas = tuple(float(value) for value in config["model"]["ridge_alphas"])
    if not alphas or any(not np.isfinite(value) or value < 0.0 for value in alphas):
        raise ValueError("model.ridge_alphas must be finite and nonnegative.")
    manifests = config["validation"]["split"].get("manifests_by_size", {})
    if not manifests:
        raise ValueError("validation.split.manifests_by_size cannot be empty.")
    supported = {
        "basis.density_normalization": basis.get("density_normalization") == "none",
        "basis.radial.measure": basis["radial"].get("measure") == "x^2 dx",
        "basis.angular.kind": basis["angular"].get("kind")
        == "real_tesseral_from_exact_racah_complex_form",
        "basis.angular.normalization": basis["angular"].get("normalization")
        == "compiler_owned",
        "representation.carrier": config["representation"].get("carrier")
        == "tagged_density",
        "runtime.backend": config["runtime"].get("backend") == "pytorch",
        "runtime.device": config["runtime"].get("device") == "cpu",
        "runtime.dtype": config["runtime"].get("dtype") == "float64",
        "runtime.execution_mode": config["runtime"].get("execution_mode")
        == "fit_export",
        "model.type": config["model"].get("type") == "linear",
        "model.fit_method": config["model"].get("fit_method")
        == "ridge_streaming_gram",
        "model.objective": config["model"].get("objective")
        == "structure_balanced_train_scaled_E1_F1",
        "model.ridge_selection_metric": config["model"].get(
            "ridge_selection_metric"
        )
        == "validation_normalized_E2_F2",
        "validation.split.mode": config["validation"]["split"].get("mode")
        == "group_adapted_manifest",
    }
    rejected = [name for name, accepted in supported.items() if not accepted]
    if rejected:
        raise ValueError(f"Unsupported bounded-workflow settings: {rejected}.")
    target_weights = (
        float(config["targets"]["energy_weight"]),
        float(config["targets"]["force_weight"]),
    )
    if any(
        not np.isfinite(value) or value < 0.0 for value in target_weights
    ) or not any(value > 0.0 for value in target_weights):
        raise ValueError("Energy/force weights must be finite, nonnegative, and nonzero.")
    selection_scales = config["validation"]["selection_scales"].values()
    if any(
        not np.isfinite(float(value)) or float(value) <= 0.0
        for value in selection_scales
    ):
        raise ValueError("Validation selection scales must be finite and positive.")
    compile_timeout = config["runtime"].get("component_timeout_seconds", "auto")
    if compile_timeout != "auto" and (
        not np.isfinite(float(compile_timeout)) or float(compile_timeout) <= 0.0
    ):
        raise ValueError(
            "runtime.component_timeout_seconds must be 'auto' or finite and positive."
        )
    return species, arms


def _load_split_manifests(config, config_directory, dataset_hash):
    records = {}
    previous_train = ()
    fixed_validation = None
    fixed_test = None
    previous_subset_hash = None
    split_config = config["validation"]["split"]
    validation_name = split_config.get("validation_partition", "validation")
    test_name = split_config.get("historical_test_partition", "test")
    for text_size, manifest_config in sorted(
        split_config["manifests_by_size"].items(), key=lambda item: int(item[0])
    ):
        size = int(text_size)
        if not isinstance(manifest_config, dict):
            raise ValueError(
                "Each validation.split.manifests_by_size entry must bind a path "
                "and its exact manifest/index/subset hashes."
            )
        required = ("path", "manifest_sha256", "index_sha256", "subset_sha256")
        missing = [key for key in required if key not in manifest_config]
        if missing:
            raise ValueError(f"Split entry {text_size!r} is missing {missing}.")
        path = _resolved(config_directory, manifest_config["path"])
        file_hash = _sha256(path)
        if file_hash != str(manifest_config["manifest_sha256"]):
            raise ValueError(f"Split manifest SHA-256 does not match for {path}.")
        payload = _read_json(path)
        if payload.get("schema") != "ye3t_tantalum_nested_scaling_split_v1":
            raise ValueError(f"Unsupported split schema in {path}.")
        if str(payload["dataset"]["sha256"]) != dataset_hash:
            raise ValueError(f"Split {path} belongs to another dataset.")
        subset = payload.get("subset", {})
        if str(subset.get("index_sha256")) != str(manifest_config["index_sha256"]):
            raise ValueError(f"Split index SHA-256 does not match for {path}.")
        if str(subset.get("sha256")) != str(manifest_config["subset_sha256"]):
            raise ValueError(f"Split subset SHA-256 does not match for {path}.")
        if int(subset.get("seed", -1)) != int(split_config["expected_seed"]):
            raise ValueError(f"Unexpected split seed in {path}.")
        if previous_subset_hash is not None and str(
            subset.get("parent_subset_sha256")
        ) != previous_subset_hash:
            raise ValueError(f"Split parent-subset chain is broken in {path}.")
        partitions = {
            name: tuple(int(value) for value in values)
            for name, values in payload["indices"].items()
        }
        for name, values in partitions.items():
            if len(values) != len(set(values)) or any(value < 0 for value in values):
                raise ValueError(f"Split partition {name!r} in {path} is invalid.")
        names = tuple(partitions)
        frame_count = int(payload["dataset"]["frame_count"])
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if set(partitions[left]).intersection(partitions[right]):
                    raise ValueError(f"Split partitions {left!r}/{right!r} overlap.")
        train = partitions[split_config.get("train_partition", "train")]
        if not train or not partitions[validation_name]:
            raise ValueError(
                f"Training and validation partitions must be nonempty in {path}."
            )
        if any(
            value >= frame_count
            for values in partitions.values()
            for value in values
        ):
            raise ValueError(f"Split index exceeds the dataset frame count in {path}.")
        if len(train) != size:
            raise ValueError(
                f"Split {path} declares {size} but contains {len(train)} train frames."
            )
        if previous_train and not set(previous_train).issubset(train):
            raise ValueError("Training ladder memberships are not nested.")
        if previous_train:
            previous_set = set(previous_train)
            retained_order = tuple(value for value in train if value in previous_set)
            if retained_order != previous_train:
                raise ValueError("Training ladder nesting does not preserve frame order.")
        previous_train = train
        previous_subset_hash = str(subset["sha256"])
        validation = partitions[validation_name]
        historical_test = partitions[test_name]
        if len(validation) != int(split_config["expected_validation_size"]):
            raise ValueError(f"Unexpected validation size in {path}.")
        if len(historical_test) != int(
            split_config["expected_historical_test_size"]
        ):
            raise ValueError(f"Unexpected historical test size in {path}.")
        if fixed_validation is None:
            fixed_validation = validation
            fixed_test = historical_test
        elif validation != fixed_validation or historical_test != fixed_test:
            raise ValueError(
                "Validation and historical-test frame ordering must be identical "
                "throughout the nested training ladder."
            )
        records[size] = {
            "path": path,
            "file_sha256": file_hash,
            "payload": payload,
            "partitions": partitions,
        }
    return records


def _descriptor_config(config, arm, materialization):
    basis = config["basis"]
    radial = basis["radial"]
    return {
        "elements": list(basis["species"]),
        "type_map": {
            element: index for index, element in enumerate(sorted(basis["species"]))
        },
        "representation": YE3TRepresentation.tagged_cauchy_image(),
        "backend": config["runtime"].get("backend", "pytorch"),
        "strict_backend": True,
        "validate_backend": True,
        "tagged_cauchy_image": {
            "tensor_order": int(basis["descriptor_catalogue"]["tensor_order"]),
            "selected_raw_tag_counts": list(arm["selected_raw_tag_counts"]),
            "source_family": str(radial["type"]),
            "radial_degrees": list(radial["degrees"]),
            "angular_degree": int(basis["angular"]["l"]),
            "cutoff_A": float(radial["cutoff_A"]),
            "support_id": str(radial["support_id"]),
            "coefficient_materialization": materialization,
        },
    }


def _run_preflight(config, arms):
    reports = {}
    for arm in arms:
        started = time.perf_counter()
        descriptor = YE3TDescriptors.ye3t(
            _descriptor_config(config, arm, "defer")
        )
        report = descriptor.metadata["tagged_cauchy_image_preflight"]
        reports[str(arm["name"])] = {
            "selected_raw_tag_counts": list(arm["selected_raw_tag_counts"]),
            "raw_opportunity_count": int(report.raw_label_count),
            "exact_feature_count": int(report.exact_image_dimension),
            "raw_opportunity_labels": report.labels,
            "planned_feature_slots": descriptor.metadata["planned_feature_slots"],
            "resource_report": report.resource_report,
            "validation_report": report.validation_report,
            "count_convention_hash": str(report.convention_hash),
            "request_hash": str(report.request["request_hash"]),
            "elapsed_seconds": time.perf_counter() - started,
            "image_descriptor_coefficients_materialized": False,
            "base_racah_and_source_product_tables_supplied": True,
        }
    return reports


def _compile_descriptor_worker(result_queue, config, arm):
    try:
        descriptor = YE3TDescriptors.ye3t(
            _descriptor_config(config, arm, "compile")
        )
        result_queue.put(
            {
                "ok": True,
                "feature_count": int(descriptor.metadata["feature_count"]),
            }
        )
    except BaseException as error:  # noqa: BLE001 - report child failure exactly
        result_queue.put(
            {"ok": False, "error": f"{type(error).__name__}: {error}"}
        )


def _remove_worker_cache_locks(cache_directory, worker_pid, started_wall_time):
    removed = []
    for path in Path(cache_directory).rglob("*.lock"):
        try:
            if path.stat().st_mtime < started_wall_time - 1.0:
                continue
            owner = path.read_text(encoding="ascii").strip()
            if owner != str(worker_pid):
                continue
            path.unlink()
            removed.append(str(path.relative_to(cache_directory)))
        except FileNotFoundError:
            continue
    return removed


def _warm_descriptor_cache_bounded(
    config, arm, timeout_seconds, cache_directory
):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_compile_descriptor_worker,
        args=(result_queue, config, arm),
    )
    started = time.perf_counter()
    started_wall_time = time.time()
    process.start()
    worker_pid = process.pid
    process.join(float(timeout_seconds))
    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join()
        removed_locks = _remove_worker_cache_locks(
            cache_directory, worker_pid, started_wall_time
        )
        result_queue.close()
        result_queue.join_thread()
        return {
            "status": "NOT_RUN",
            "reason": "timeout",
            "elapsed_seconds": elapsed,
            "timeout_seconds": float(timeout_seconds),
            "removed_worker_locks": removed_locks,
        }
    exit_code = process.exitcode
    try:
        result = result_queue.get(timeout=2.0)
    except queue.Empty:
        result = None
    result_queue.close()
    result_queue.join_thread()
    if result is None:
        removed_locks = _remove_worker_cache_locks(
            cache_directory, worker_pid, started_wall_time
        )
        return {
            "status": "NOT_RUN",
            "reason": f"compiler subprocess exited with code {exit_code}",
            "elapsed_seconds": elapsed,
            "timeout_seconds": float(timeout_seconds),
            "removed_worker_locks": removed_locks,
        }
    if not result["ok"]:
        removed_locks = _remove_worker_cache_locks(
            cache_directory, worker_pid, started_wall_time
        )
        return {
            "status": "NOT_RUN",
            "reason": result["error"],
            "elapsed_seconds": elapsed,
            "timeout_seconds": float(timeout_seconds),
            "removed_worker_locks": removed_locks,
        }
    return {
        "status": "passed",
        "feature_count": int(result["feature_count"]),
        "elapsed_seconds": elapsed,
        "timeout_seconds": float(timeout_seconds),
    }


def _target_energy_force(atoms, energy_key, force_key):
    energy_key = str(energy_key)
    force_key = str(force_key)
    if energy_key in atoms.info:
        energy = atoms.info[energy_key]
    elif atoms.calc is not None and energy_key in atoms.calc.results:
        energy = atoms.calc.results[energy_key]
    elif energy_key in {"energy", "E"}:
        energy = atoms.get_potential_energy()
    else:
        raise KeyError(f"Structure is missing energy key {energy_key!r}.")
    if force_key in atoms.arrays:
        forces = atoms.arrays[force_key]
    elif atoms.calc is not None and force_key in atoms.calc.results:
        forces = atoms.calc.results[force_key]
    elif force_key in {"forces", "force", "F"}:
        forces = atoms.get_forces()
    else:
        raise KeyError(f"Structure is missing force key {force_key!r}.")
    forces = np.asarray(forces, dtype=np.float64)
    if (
        not np.isfinite(float(energy))
        or forces.shape != (len(atoms), 3)
        or not np.all(np.isfinite(forces))
    ):
        raise ValueError("Energy/force targets must be finite and match the structure.")
    return float(energy), forces


def _reference_component_identity(metadata):
    batch_fields = {
        "structure_count",
        "evaluation_sha256",
        "minimum_energy_eV_per_atom",
        "maximum_energy_eV_per_atom",
        "maximum_absolute_force_eV_per_A",
        "executable",
        "executable_path",
        "timeout_seconds",
    }
    return {
        "schema": "ye3t_reference_component_identity_v1",
        "component": {
            key: value for key, value in metadata.items() if key not in batch_fields
        },
    }


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_plots(output_directory, metrics, ridge_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_directory = Path(output_directory) / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    arm_names = sorted({row["arm"] for row in metrics})
    colors = dict(zip(arm_names, plt.get_cmap("tab10").colors, strict=False))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for arm in arm_names:
        rows = sorted(
            (row for row in metrics if row["arm"] == arm),
            key=lambda row: row["train_structures"],
        )
        x = [row["train_structures"] for row in rows]
        axes[0].plot(
            x,
            [row["validation_energy_rmse_eV_per_atom"] for row in rows],
            marker="o",
            label=arm,
            color=colors[arm],
        )
        axes[1].plot(
            x,
            [row["validation_force_rmse_eV_per_A"] for row in rows],
            marker="o",
            label=arm,
            color=colors[arm],
        )
    for axis, ylabel in zip(
        axes,
        (
            "Energy RMSE (eV/atom; ZBL-residual = total)",
            "Force RMSE (eV/Angstrom; ZBL-residual = total)",
        ),
        strict=True,
    ):
        axis.set_xlabel("Training structures")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Bounded N=4, l=1 tagged-Cauchy Ta diagnostic")
    for suffix in ("png", "pdf"):
        figure.savefig(figure_directory / f"learning_curves.{suffix}", dpi=220)
    plt.close(figure)

    sizes = sorted({row["train_structures"] for row in ridge_rows})
    largest = sizes[-1]
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for arm in arm_names:
        rows = [
            row
            for row in ridge_rows
            if row["arm"] == arm and row["train_structures"] == largest
        ]
        rows.sort(key=lambda row: row["alpha_index"])
        axis.plot(
            range(len(rows)),
            [row["selection_metric"] for row in rows],
            marker="o",
            label=arm,
            color=colors[arm],
        )
    labels = [
        f"{row['ridge_alpha']:.0e}" if row["ridge_alpha"] else "0"
        for row in sorted(
            (
                row
                for row in ridge_rows
                if row["arm"] == arm_names[0]
                and row["train_structures"] == largest
            ),
            key=lambda row: row["alpha_index"],
        )
    ]
    axis.set_xticks(range(len(labels)), labels)
    axis.set_xlabel("Ridge alpha")
    axis.set_ylabel("Validation (E/0.01)^2 + (F/0.1)^2")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title(f"Ridge selection at {largest} training structures")
    for suffix in ("png", "pdf"):
        figure.savefig(figure_directory / f"ridge_selection.{suffix}", dpi=220)
    plt.close(figure)


def _artifact_manifest(output_directory):
    output_directory = Path(output_directory)
    records = []
    for path in sorted(output_directory.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            records.append(
                {
                    "path": str(path.relative_to(output_directory)),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    body = {
        "schema": "ye3t_tagged_cauchy_ta_artifact_manifest_v1",
        "files": records,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        help="new result directory; the default is a unique run under YE3T_WORKFLOW_ROOT",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="root for reusable coupling and target-free geometry caches",
    )
    parser.add_argument("--coefficient-cache", type=Path)
    parser.add_argument("--geometry-cache", type=Path)
    parser.add_argument(
        "--lammps",
        help="LAMMPS executable used to evaluate the configured reference potential",
    )
    parser.add_argument(
        "--ye3t-lammps-root",
        type=Path,
        help="optional ye3t-lammps checkout recorded in result provenance",
    )
    parser.add_argument("--component-timeout-seconds", type=float)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="count selected features and resources without compiling or loading data",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_directory = config_path.parent
    config = _read_json(config_path)
    species, arms = _validate_config(config)
    runtime = config["runtime"]
    configured_output = runtime.get("output_directory", "auto")
    if args.output is not None:
        output_directory = args.output.expanduser().resolve()
    elif configured_output == "auto":
        output_directory = (
            _automatic_workflow_root()
            / "MLIP"
            / str(config["metadata"]["name"])
            / (
                "run-"
                + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                + f"-{os.getpid()}"
            )
        )
    else:
        output_directory = _resolved(config_directory, configured_output)

    cache_config = runtime.get("cache", {})
    if cache_config and cache_config.get("verify", "hash") != "hash":
        raise ValueError("runtime.cache.verify must be hash.")
    configured_cache_root = runtime.get("cache_root", "auto")
    if cache_config:
        cache_mode = cache_config.get("mode", "auto")
        cache_directory = cache_config.get("directory")
        if cache_mode != "auto":
            raise ValueError("runtime.cache.mode must be auto for this workflow.")
        configured_cache_root = (
            "auto" if cache_directory in (None, "auto") else cache_directory
        )
    if args.cache_root is not None:
        cache_root = args.cache_root.expanduser().resolve()
    elif configured_cache_root == "auto":
        cache_root = _automatic_cache_root()
    else:
        cache_root = _resolved(config_directory, configured_cache_root)
    if args.coefficient_cache is not None:
        coefficient_cache = args.coefficient_cache.expanduser().resolve()
    elif "coefficient_cache_directory" in runtime:
        coefficient_cache = _resolved(
            config_directory, runtime["coefficient_cache_directory"]
        )
    else:
        coefficient_cache = cache_root / "couplings"
    if args.geometry_cache is not None:
        geometry_cache = args.geometry_cache.expanduser().resolve()
    elif "geometry_cache_directory" in runtime:
        geometry_cache = _resolved(
            config_directory, runtime["geometry_cache_directory"]
        )
    else:
        geometry_cache = cache_root / "geometry"
    os.environ["YE3T_CACHE_DIR"] = str(coefficient_cache)
    if output_directory.exists():
        raise FileExistsError(
            f"Output path already exists; choose a new --output: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    preflight = _run_preflight(config, arms)
    preflight_payload = {
        "schema": "ye3t_tagged_cauchy_ta_preflight_v1",
        "scope": "bounded_N4_s_le_2_homogeneous_l1_diagnostic",
        "config_sha256": _sha256(config_path),
        "arms": preflight,
        "elapsed_seconds": time.perf_counter() - started,
        "dataset_loaded": False,
        "image_descriptor_coefficients_materialized": False,
        "note": (
            "The count path defers tagged image/descriptor coefficients; the small "
            "exact Racah and source-product tables are supplied to the request."
        ),
    }
    _write_json(output_directory / "preflight.json", preflight_payload)
    print(json.dumps(_jsonable(preflight_payload), indent=2, sort_keys=True))
    if args.preflight_only:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401

    descriptors = {}
    compile_seconds = {}
    configured_timeout = runtime.get("component_timeout_seconds", "auto")
    compile_timeout = (
        float(args.component_timeout_seconds)
        if args.component_timeout_seconds is not None
        else 600.0
        if configured_timeout == "auto"
        else float(configured_timeout)
    )
    if not np.isfinite(compile_timeout) or compile_timeout <= 0.0:
        raise ValueError("The effective component timeout must be finite and positive.")
    compile_records = {}
    for arm in arms:
        arm_name = str(arm["name"])
        record = _warm_descriptor_cache_bounded(
            config, arm, compile_timeout, coefficient_cache
        )
        compile_records[arm_name] = record
        compile_certificate = {
            "schema": "ye3t_tagged_cauchy_ta_compile_certificate_v1",
            "config_sha256": _sha256(config_path),
            "component_timeout_seconds": compile_timeout,
            "dataset_loaded": False,
            "expected_arms": [str(value["name"]) for value in arms],
            "arms": compile_records,
            "passed": len(compile_records) == len(arms)
            and all(
                value["status"] == "passed"
                for value in compile_records.values()
            ),
        }
        _write_json(
            output_directory / "compile_certificate.json",
            compile_certificate,
        )
        if record["status"] != "passed":
            raise RuntimeError(
                f"Descriptor arm {arm_name!r} did not compile: "
                f"{record['reason']}."
            )
        arm_started = time.perf_counter()
        descriptors[arm_name] = YE3TDescriptors.ye3t(
            _descriptor_config(config, arm, "compile")
        )
        hydration_seconds = time.perf_counter() - arm_started
        record["cache_hydration_seconds"] = hydration_seconds
        record["total_compile_stage_seconds"] = (
            float(record["elapsed_seconds"]) + hydration_seconds
        )
        compile_seconds[arm_name] = record["total_compile_stage_seconds"]
    compile_certificate["dataset_loaded"] = False
    compile_certificate["passed"] = True
    _write_json(
        output_directory / "compile_certificate.json", compile_certificate
    )

    dataset_path = _resolved(config_directory, config["targets"]["dataset"]["path"])
    dataset_hash = _sha256(dataset_path)
    if dataset_hash != str(config["targets"]["dataset"]["sha256"]):
        raise ValueError("Ta dataset SHA-256 does not match the config.")
    split_records = _load_split_manifests(config, config_directory, dataset_hash)
    train_partition = config["validation"]["split"].get("train_partition", "train")
    validation_partition = config["validation"]["split"].get(
        "validation_partition", "validation"
    )
    used_indices = sorted(
        {
            index
            for record in split_records.values()
            for name in (train_partition, validation_partition)
            for index in record["partitions"][name]
        }
    )
    structures = load_xyz_structures(dataset_path, indices=used_indices)
    structures_by_index = dict(zip(used_indices, structures, strict=True))
    raw_targets = {
        index: _target_energy_force(
            structures_by_index[index],
            config["targets"]["energy"],
            config["targets"]["forces"],
        )
        for index in used_indices
    }

    reference_input = dict(config["targets"]["reference_potential"])
    reference_input["executable"] = (
        args.lammps if args.lammps is not None else runtime["lammps_executable"]
    )
    reference_config = lammps_zbl_reference_config(reference_input, species)
    reference = evaluate_lammps_zbl_reference(structures, reference_config)
    reference_energy_by_index = dict(
        zip(used_indices, reference["reference_energies"], strict=True)
    )
    reference_force_by_index = dict(
        zip(used_indices, reference["reference_forces"], strict=True)
    )
    residual_energy = {
        index: raw_targets[index][0] - reference_energy_by_index[index]
        for index in used_indices
    }
    residual_force = {
        index: raw_targets[index][1] - reference_force_by_index[index]
        for index in used_indices
    }
    target_transform = tagged_cauchy_reference_target_metadata(
        reference_potential_metadata=_reference_component_identity(
            reference["metadata"]
        )
    )
    _write_json(
        output_directory / "reference_evaluation.json",
        {
            "used_frame_indices": used_indices,
            "metadata": reference["metadata"],
            "target_transform": target_transform,
        },
    )

    scales = config["validation"]["selection_scales"]
    energy_scale = float(scales["energy_eV_per_atom"])
    force_scale = float(scales["force_eV_per_A"])
    weight_config = config["targets"].get("structure_weighting", {})
    metrics = []
    ridge_rows = []
    for size, split in split_records.items():
        train_indices = split["partitions"][train_partition]
        validation_indices = split["partitions"][validation_partition]
        train_structures = [structures_by_index[index] for index in train_indices]
        validation_structures = [
            structures_by_index[index] for index in validation_indices
        ]
        train_energies = np.asarray(
            [residual_energy[index] for index in train_indices], dtype=np.float64
        )
        train_forces = tuple(residual_force[index] for index in train_indices)
        validation_energies = np.asarray(
            [residual_energy[index] for index in validation_indices], dtype=np.float64
        )
        validation_forces = tuple(
            residual_force[index] for index in validation_indices
        )
        for arm in arms:
            arm_name = str(arm["name"])
            descriptor = descriptors[arm_name]
            normal_started = time.perf_counter()
            normal = build_tagged_cauchy_image_normal_equations(
                descriptor,
                train_structures,
                target_energies=train_energies,
                target_forces=train_forces,
                energy_weight=float(config["targets"]["energy_weight"]),
                force_weight=float(config["targets"]["force_weight"]),
                geometry_cache_dir=geometry_cache,
                structure_group_key=weight_config.get("group_key"),
                structure_group_weights=weight_config.get("group_weights"),
                structure_group_default_weight=weight_config.get("default_weight"),
                structure_group_normalize_mean=bool(
                    weight_config.get("normalize_mean", True)
                ),
                reference_target_metadata=target_transform,
            )
            normal_seconds = time.perf_counter() - normal_started
            candidates = []
            for alpha_index, alpha in enumerate(config["model"]["ridge_alphas"]):
                model = YE3TModel.linear(
                    descriptor,
                    normal_equations=normal,
                    ridge_alpha=float(alpha),
                )
                train_score = score_tagged_cauchy_image_model(
                    model,
                    train_structures,
                    target_energies=train_energies,
                    target_forces=train_forces,
                    geometry_cache_dir=geometry_cache,
                    descriptor=descriptor,
                )
                validation_score = score_tagged_cauchy_image_model(
                    model,
                    validation_structures,
                    target_energies=validation_energies,
                    target_forces=validation_forces,
                    geometry_cache_dir=geometry_cache,
                    descriptor=descriptor,
                )
                selection_metric = (
                    validation_score["energy_rmse_eV_per_atom"] / energy_scale
                ) ** 2 + (
                    validation_score["force_rmse_eV_per_A"] / force_scale
                ) ** 2
                row = {
                    "arm": arm_name,
                    "train_structures": int(size),
                    "alpha_index": int(alpha_index),
                    "ridge_alpha": float(alpha),
                    "feature_count": int(descriptor.metadata["feature_count"]),
                    "selection_metric": float(selection_metric),
                    "train_energy_rmse_eV_per_atom": train_score[
                        "energy_rmse_eV_per_atom"
                    ],
                    "train_force_rmse_eV_per_A": train_score[
                        "force_rmse_eV_per_A"
                    ],
                    "validation_energy_rmse_eV_per_atom": validation_score[
                        "energy_rmse_eV_per_atom"
                    ],
                    "validation_force_rmse_eV_per_A": validation_score[
                        "force_rmse_eV_per_A"
                    ],
                    "normal_hash": str(normal["normal_hash"]),
                    "condition_number": float(model.fit_metadata["condition_number"]),
                    "numerical_rank": int(model.fit_metadata["numerical_rank"]),
                    "geometry_cache_hits": int(normal["geometry_cache_hits"]),
                }
                ridge_rows.append(row)
                candidates.append((selection_metric, row, model))
            _metric, best_row, best_model = min(candidates, key=lambda value: value[0])
            model_path = (
                output_directory
                / "models"
                / f"train_{size:03d}"
                / f"{arm_name}.ye3t.json"
            )
            payload = export_tagged_cauchy_image_model(model_path, best_model)
            loaded = load_tagged_cauchy_image_model(model_path)
            if loaded.fit_metadata.get("normal_hash") != normal["normal_hash"]:
                raise AssertionError("Exported fit provenance did not round-trip.")
            for element in loaded.species_order:
                if not np.array_equal(
                    loaded.beta_by_species[element].detach().cpu().numpy(),
                    best_model.beta_by_species[element].detach().cpu().numpy(),
                ):
                    raise AssertionError("Exported linear coefficients changed.")
            metrics.append(
                {
                    "arm": arm_name,
                    "selected_raw_tag_counts": "+".join(
                        str(value) for value in arm["selected_raw_tag_counts"]
                    ),
                    "train_structures": int(size),
                    "validation_structures": len(validation_structures),
                    "feature_count": int(descriptor.metadata["feature_count"]),
                    "raw_opportunity_count": int(
                        descriptor.metadata[
                            "tagged_cauchy_image_preflight"
                        ].raw_label_count
                    ),
                    "ridge_alpha": best_row["ridge_alpha"],
                    "selection_metric": best_row["selection_metric"],
                    "train_energy_rmse_eV_per_atom": best_row[
                        "train_energy_rmse_eV_per_atom"
                    ],
                    "train_force_rmse_eV_per_A": best_row[
                        "train_force_rmse_eV_per_A"
                    ],
                    "validation_energy_rmse_eV_per_atom": best_row[
                        "validation_energy_rmse_eV_per_atom"
                    ],
                    "validation_force_rmse_eV_per_A": best_row[
                        "validation_force_rmse_eV_per_A"
                    ],
                    "compile_seconds": compile_seconds[arm_name],
                    "normal_equation_seconds": normal_seconds,
                    "geometry_cache_hits": int(normal["geometry_cache_hits"]),
                    "normal_hash": str(normal["normal_hash"]),
                    "model_path": str(model_path.relative_to(output_directory)),
                    "model_sha256": _sha256(model_path),
                    "model_self_hash": str(payload["self_hash"]),
                }
            )

    _write_csv(output_directory / "ridge_trials.csv", ridge_rows)
    _write_csv(output_directory / "learning_metrics.csv", metrics)
    _render_plots(output_directory, metrics, ridge_rows)
    ye3t_repository = _repository_for_module("ye3t")
    ye3t_ace_repository = _repository_for_module("ye3t_ace")
    ye3t_lammps_repository = (
        args.ye3t_lammps_root.expanduser().resolve()
        if args.ye3t_lammps_root is not None
        else None
    )
    summary = {
        "schema": "ye3t_tagged_cauchy_ta_diagnostic_v1",
        "status": "bounded_diagnostic_not_publication_promotion",
        "scope": "N=4, s in selected subsets of {0,1,2}, homogeneous l=1",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "split_manifests": {
            str(size): {
                "path": str(record["path"]),
                "sha256": record["file_sha256"],
            }
            for size, record in split_records.items()
        },
        "coefficient_cache_directory": str(coefficient_cache),
        "geometry_cache_directory": str(geometry_cache),
        "software_revisions": {
            "ye3t-ace": _git_revision(ye3t_ace_repository),
            "ye3t": _git_revision(ye3t_repository),
            "ye3t-lammps": _git_revision(ye3t_lammps_repository),
        },
        "installed_versions": {
            "ye3t": _installed_version("ye3t"),
            "ye3t-ace": _installed_version("ye3t-ace"),
        },
        "reference_evaluation": reference["metadata"],
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "The current exact image compiler is bounded to N=4, s<=2, and homogeneous l=1.",
            "The s=1 arm is an exact same-source duplicate control, not a general one-tag carrier.",
            "The study does not establish publication accuracy, stability, multi-element, or Kokkos claims.",
            "The validation partition is used for alpha selection; historical-test metadata is verified but its structures and targets are not loaded.",
        ],
    }
    _write_json(output_directory / "summary.json", summary)
    resolved_config = copy.deepcopy(config)
    resolved_config["runtime"]["effective_output_directory"] = str(
        output_directory
    )
    resolved_config["runtime"]["effective_coefficient_cache_directory"] = str(
        coefficient_cache
    )
    resolved_config["runtime"]["effective_geometry_cache_directory"] = str(
        geometry_cache
    )
    resolved_config["runtime"]["effective_component_timeout_seconds"] = (
        compile_timeout
    )
    _write_json(output_directory / "resolved_config.json", resolved_config)
    _write_json(
        output_directory / "artifact_manifest.json",
        _artifact_manifest(output_directory),
    )
    print(f"Results: {output_directory / 'summary.json'}")
    print(f"Learning curves: {output_directory / 'figures' / 'learning_curves.png'}")


if __name__ == "__main__":
    main()
