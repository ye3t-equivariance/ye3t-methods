"""Dictionary workflow configuration helpers for public examples and APIs."""

import json
from pathlib import Path


HOMOGENEOUS_CONFIG_SECTIONS = (
    "basis",
    "representation",
    "runtime",
    "model",
    "targets",
    "validation",
)

HOMOGENEOUS_CONFIG_STATUSES = (
    "stable",
    "experimental",
    "planned",
)

HOMOGENEOUS_CONFIG_SCHEMA = {
    "name": "ye3t_homogeneous_config",
    "version": 1,
    "sections": HOMOGENEOUS_CONFIG_SECTIONS,
    "statuses": HOMOGENEOUS_CONFIG_STATUSES,
}


def load_workflow_config(path):
    """Load a JSON workflow configuration dictionary."""
    if path is None:
        return {}
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Workflow config JSON must contain an object at the top level.")
    return payload


def merge_workflow_config(base, *updates):
    """Recursively merge dictionary-style workflow settings."""
    merged = dict(base or {})
    for update in updates:
        if update is None:
            continue
        for key, value in dict(update).items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = merge_workflow_config(merged[key], value)
            elif value is not None:
                merged[key] = value
    return merged


def _section_dict(config, section):
    value = config.get(section, {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("Homogeneous config section " + repr(section) + " must be a dictionary.")
    return dict(value)


def _metadata_status(config, metadata, status):
    if status is not None:
        return str(status)
    if "status" in metadata:
        return str(metadata["status"])
    if "status" in config:
        return str(config["status"])
    return "stable"


def _legacy_top_level_payload(config):
    reserved = set(HOMOGENEOUS_CONFIG_SECTIONS)
    reserved.update(("metadata", "schema", "status", "extensions", "experimental"))
    return {key: value for key, value in config.items() if key not in reserved}


def normalize_homogeneous_config(config, *, status=None):
    """Return a canonical config dictionary with homogeneous top-level sections."""
    if not isinstance(config, dict):
        raise ValueError("Homogeneous config must be a dictionary.")

    normalized = {}
    for section in HOMOGENEOUS_CONFIG_SECTIONS:
        normalized[section] = _section_dict(config, section)

    metadata = _section_dict(config, "metadata")
    metadata["schema"] = dict(HOMOGENEOUS_CONFIG_SCHEMA)
    metadata["status"] = _metadata_status(config, metadata, status)
    normalized["metadata"] = metadata

    extensions = _section_dict(config, "extensions")
    if "experimental" in config:
        experimental = config["experimental"]
        if not isinstance(experimental, dict):
            raise ValueError("Homogeneous config experimental section must be a dictionary.")
        extensions.setdefault("experimental", dict(experimental))
    legacy = _legacy_top_level_payload(config)
    if legacy:
        extensions.setdefault("legacy_top_level", legacy)
    normalized["extensions"] = extensions
    return normalized


def validate_homogeneous_config(config, *, require_complete=True, require_status=True):
    """Return a validation report for the homogeneous workflow config schema."""
    failures = []
    normalized = None
    try:
        normalized = normalize_homogeneous_config(config)
    except ValueError as exc:
        return {
            "passed": False,
            "schema": dict(HOMOGENEOUS_CONFIG_SCHEMA),
            "failures": (str(exc),),
            "warnings": (),
            "normalized": None,
        }

    original_keys = set(config)
    missing_sections = tuple(section for section in HOMOGENEOUS_CONFIG_SECTIONS if section not in original_keys)
    if require_complete and missing_sections:
        failures.append("missing homogeneous config sections: " + ", ".join(missing_sections))

    metadata_input = config.get("metadata", {})
    status_explicit = "status" in config or (
        isinstance(metadata_input, dict) and "status" in metadata_input
    )
    status = normalized["metadata"]["status"]
    if require_status and not status_explicit:
        failures.append("homogeneous config metadata.status is required.")
    if status not in HOMOGENEOUS_CONFIG_STATUSES:
        failures.append(
            "homogeneous config status must be one of "
            + ", ".join(HOMOGENEOUS_CONFIG_STATUSES)
            + "; got "
            + repr(status)
            + "."
        )

    representation = normalized["representation"]
    if "labels" in representation or "manual_labels" in representation:
        failures.append("representation labels must be declarative and sourced from ye3t.couplings.")
    coupling = representation.get("coupling", {})
    if coupling is None:
        coupling = {}
    if not isinstance(coupling, dict):
        failures.append("representation.coupling must be a dictionary.")
    else:
        source = coupling.get("source", "ye3t.couplings")
        if source != "ye3t.couplings":
            failures.append("representation.coupling.source must be ye3t.couplings.")

    runtime = normalized["runtime"]
    if "fast_path" in runtime and not isinstance(runtime["fast_path"], (str, list, tuple, dict)):
        failures.append("runtime.fast_path must be a string, list, tuple, or dictionary.")
    if "derivatives" in runtime and str(runtime["derivatives"]).strip().lower() == "ctilde_force":
        failures.append("runtime.derivatives must use cyprime naming, not ctilde_force.")

    warnings = []
    if normalized["extensions"].get("legacy_top_level"):
        warnings.append("legacy top-level keys were preserved under extensions.legacy_top_level.")
    if status in {"experimental", "planned"} and not normalized["extensions"]:
        warnings.append("experimental or planned configs should declare extension/provenance metadata when possible.")

    return {
        "passed": not failures,
        "schema": dict(HOMOGENEOUS_CONFIG_SCHEMA),
        "failures": tuple(failures),
        "warnings": tuple(warnings),
        "missing_sections": missing_sections,
        "status": status,
        "normalized": normalized,
    }


def require_homogeneous_config(config, *, require_complete=True, require_status=True):
    """Normalize a homogeneous workflow config or raise with validation failures."""
    report = validate_homogeneous_config(
        config,
        require_complete=bool(require_complete),
        require_status=bool(require_status),
    )
    if not report["passed"]:
        raise ValueError("; ".join(report["failures"]))
    return report["normalized"]


def select_workflow_frames(frames, *, frame_indices = None, max_frames = None, max_structures = None):
    """Select a deterministic frame subset from an ordered workflow sequence."""
    selected = list(frames)
    if frame_indices is not None:
        selected = [selected[int(index)] for index in frame_indices]
    limit = max_structures if max_structures is not None else max_frames
    if limit is not None:
        count = int(limit)
        if count < 1:
            raise ValueError("max_frames/max_structures must be positive.")
        selected = selected[:count]
    if not selected:
        raise ValueError("The requested workflow frame selection is empty.")
    return selected


def workflow_config_summary(config, keys = None):
    """Return a compact JSON-safe summary of selected workflow settings."""
    chosen = sorted(config) if keys is None else tuple(keys)
    summary = {}
    for key in chosen:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, Path):
            summary[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, (list, tuple)):
            summary[key] = list(value)
        elif isinstance(value, dict):
            summary[key] = {
                str(sub_key): sub_value
                for sub_key, sub_value in value.items()
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None
            }
        else:
            summary[key] = value.__class__.__name__
    return summary


def print_workflow_description(title, config, keys = None):
    """Print a compact resolved workflow-configuration block."""
    print(title)
    for key, value in workflow_config_summary(config, keys=keys).items():
        print(str(key) + ": " + str(value))


__all__ = [
    "HOMOGENEOUS_CONFIG_SCHEMA",
    "HOMOGENEOUS_CONFIG_SECTIONS",
    "HOMOGENEOUS_CONFIG_STATUSES",
    "load_workflow_config",
    "merge_workflow_config",
    "normalize_homogeneous_config",
    "print_workflow_description",
    "require_homogeneous_config",
    "select_workflow_frames",
    "validate_homogeneous_config",
    "workflow_config_summary",
]
