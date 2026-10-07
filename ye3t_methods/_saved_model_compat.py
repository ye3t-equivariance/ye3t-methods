"""Select this distribution's historical module aliases for Torch readers."""

import importlib.util
from pathlib import Path
import sys


def ensure_saved_model_imports():
    """Load only the local ye3t_ace alias package before legacy unpickling."""

    shim = Path(__file__).resolve().parents[1] / "ye3t_ace" / "__init__.py"
    loaded = sys.modules.get("ye3t_ace")
    if loaded is not None:
        if Path(getattr(loaded, "__file__", "")).resolve() != shim:
            raise ImportError(
                "A different ye3t_ace package was already imported. Read this "
                "saved model in a fresh process using ye3t-methods."
            )
        return
    spec = importlib.util.spec_from_file_location(
        "ye3t_ace", shim, submodule_search_locations=[str(shim.parent)]
    )
    if spec is None or spec.loader is None:
        raise ImportError("The ye3t-methods saved-model import shim is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ye3t_ace"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules["ye3t_ace"]
        raise
