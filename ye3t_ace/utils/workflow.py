"""Shared dictionary helpers for importable example workflows."""

from ye3t_ace.workflow_config import merge_workflow_config, print_workflow_description


def workflow_defaults(output_dir_default=None, max_frames_default=None):
    """Return common workflow defaults for editable workflow dictionaries."""
    return {
        "output_dir": output_dir_default,
        "max_frames": max_frames_default,
        "seed": 7,
        "timeout_seconds": 60.0,
        "describe": False,
    }


def merge_config(base, *updates):
    """Recursively merge workflow dictionaries."""
    return merge_workflow_config(base, *updates)


class ConfigNamespace:
    """Attribute view over one resolved workflow dictionary."""

    def __init__(self, config):
        self.config = dict(config)
        for key, value in self.config.items():
            setattr(self, str(key), value)


def config_namespace(defaults, config=None):
    """Return an attribute namespace from defaults plus optional overrides."""
    return ConfigNamespace(merge_config(defaults, config))


def resolved_common_config(args):
    """Return common workflow settings from a config namespace."""
    config = {}
    for key, default in (
        ("output_dir", None),
        ("max_frames", None),
        ("seed", 7),
        ("timeout_seconds", 60.0),
    ):
        value = getattr(args, key, default)
        if value is not None:
            config[key] = value
    return config


def maybe_describe_and_exit(title, args, config):
    """Print common workflow settings when the config requests it."""
    if not getattr(args, "describe", False):
        return False
    print_workflow_description(
        title,
        config,
        keys=("output_dir", "max_frames", "seed", "timeout_seconds"),
    )
    return True

