#!/usr/bin/env python3
"""Build one elemental two-tag catalogue from the approved content schedule."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


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
COUPLING_CACHE_ROOT = Path(
    os.environ.get(
        "YE3T_CACHE_DIR",
        str(CONFIG["runtime"].get("coupling_cache_root", WORKFLOW_ROOT / "cache" / "couplings")),
    )
)
if not COUPLING_CACHE_ROOT.is_absolute():
    COUPLING_CACHE_ROOT = (CONFIG_PATH.parent / COUPLING_CACHE_ROOT).resolve()
os.environ.setdefault("YE3T_CACHE_DIR", str(COUPLING_CACHE_ROOT))

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import count as count_coupling
from ye3t.couplings import lifted_cauchy_fixed_content_scalar_request
from ye3t.couplings import plan as plan_coupling
from ye3t.couplings.lifted_cauchy_scalar import CompiledLiftedCauchyScalar
from ye3t.couplings.tagged_cauchy import _stable_free_moment_basis


SOURCE = PUBLIC / CONFIG["basis"]["tagged"].get(
    "component_schedule", "tagged_components.json"
)
DEFAULT_OUTPUT = WORKFLOW_ROOT / "Si" / "tagged_catalogue"
RESOURCE_LIMITS = {
    "maximum_ordered_basis_states": 200000,
    "maximum_static_bytes": 536870912,
    "maximum_descriptor_count": 10000,
    "maximum_parent_shuffle_count": 200000,
    "maximum_channel_assignment_count": 10000,
    "maximum_loader_symbolic_cells": 20000000,
}


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def role_bindings_for_source(source):
    tag_count = int(source["tag_count_s"])
    bindings = tuple(
        tuple(value)
        for value in source.get(
            "role_bindings",
            (*(("edge", index) for index in range(tag_count)), ("density", 0)),
        )
    )
    expected = tuple(("edge", index) for index in range(tag_count)) + (
        ("density", 0),
    )
    if bindings != expected:
        raise ValueError(
            f"Role bindings {bindings!r} do not match tag_count_s={tag_count}; "
            f"expected {expected!r}."
        )
    return tag_count, bindings


def component_request(component, species, source_family_id, role_dimension):
    channels = tuple(
        {
            **dict(value),
            "neighbor_species": species,
            "source_family_id": source_family_id,
        }
        for value in component["block_complete_channel_keys"]
    )
    block_sizes = tuple(int(value) for value in component["block_sizes"])
    request = lifted_cauchy_fixed_content_scalar_request(
        channels,
        block_sizes,
        role_dimension=int(role_dimension),
        kappa_policy=str(component.get("kappa_policy", "all")),
        block_lambda_policy=str(component.get("block_lambda_policy", "all")),
        emit_factored=False,
        resource_limits=RESOURCE_LIMITS,
    )
    return request, channels, block_sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--system", default="Si")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    output = args.output.resolve()
    source_path = args.source.resolve()
    system = str(args.system)
    source_family_id = (
        f"pace_cheb_exp_cos_{system.lower()}_{args.cutoff:.3f}_v1"
    )
    artifacts = output / "artifacts"
    images = output / "free_moment_images"
    artifacts.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    tag_count, role_bindings = role_bindings_for_source(source)
    rows = []
    started = time.perf_counter()
    for component in source["components"]:
        component_index = int(component["component_index"])
        row_started = time.perf_counter()
        row = {
            "component_index": component_index,
            "tensor_order_N": int(component["tensor_order"]),
            "n": [int(value) for value in component["n"]],
            "l": [int(value) for value in component["l"]],
            "content_pattern": [int(value) for value in component["content_pattern"]],
        }
        try:
            request, channels, block_sizes = component_request(
                component, system, source_family_id, len(role_bindings)
            )
            report = count_coupling(request)
            plan = plan_coupling(report)
            request_hash = stable_hash(request)
            artifact_path = artifacts / f"{request_hash}.json"
            if artifact_path.is_file():
                artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                compiled = None
                artifact_self_hash = str(artifact_payload["self_hash"])
                artifact_hit = True
            else:
                compiled = compile_coupling(plan)
                write_json(artifact_path, compiled.to_dict())
                artifact_self_hash = compiled.self_hash
                artifact_hit = False
            image_path = images / f"{request_hash}.json"
            if image_path.is_file():
                image = json.loads(image_path.read_text(encoding="utf-8"))
                image_hit = True
            else:
                if compiled is None:
                    compiled = CompiledLiftedCauchyScalar.from_dict(artifact_payload)
                image = _stable_free_moment_basis(compiled, role_bindings)
                write_json(image_path, image)
                image_hit = False
            row.update(
                {
                    "status": "passed",
                    "request_hash": request_hash,
                    "channels": channels,
                    "block_sizes": block_sizes,
                    "kappa_policy": str(component.get("kappa_policy", "all")),
                    "block_lambda_policy": str(
                        component.get("block_lambda_policy", "all")
                    ),
                    "raw_descriptor_count": int(report.descriptor_count),
                    "independent_feature_count": int(image["independent_feature_count"]),
                    "artifact_self_hash": artifact_self_hash,
                    "artifact_cache_hit": artifact_hit,
                    "image_cache_hit": image_hit,
                    "resource_report": plan.resource_report,
                    "certificates": image["certificates"],
                }
            )
        except Exception as error:
            row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        row["elapsed_seconds"] = time.perf_counter() - row_started
        rows.append(row)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "component_index",
                        "tensor_order_N",
                        "status",
                        "raw_descriptor_count",
                        "independent_feature_count",
                        "artifact_cache_hit",
                        "image_cache_hit",
                        "elapsed_seconds",
                    )
                    if key in row
                }
            ),
            flush=True,
        )
    manifest = {
        "schema": "ye3t_mlearn_element_tagged_catalogue_v1",
        "status": "portable_publication_catalogue",
        "source_schedule": str(source_path),
        "source_schedule_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "species": [system],
        "radial_basis": "PACE_ChebExpCos",
        "cutoff_A": float(args.cutoff),
        "tag_count_s": int(tag_count),
        "role_bindings": [list(value) for value in role_bindings],
        "selection_unit": "complete_fixed_channel_content",
        "coordinate_policy": "exact_stable_free_moment_image",
        "components": rows,
        "total_independent_feature_count": sum(
            row.get("independent_feature_count", 0) for row in rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "catalogue_manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "catalogue_manifest.json"), "features": manifest["total_independent_feature_count"], "seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
