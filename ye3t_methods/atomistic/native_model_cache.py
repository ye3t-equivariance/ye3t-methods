"""Reuse exported tagged models and safely share resident native workspaces.

Prepared model text uses the common persistent YE3T artifact store. Native
pointers stay process-local. Each resident handle has a lock because its C++
evaluator owns mutable scratch arrays; callers keep separate neighbor lists.
"""

from collections import OrderedDict
import ctypes
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import weakref

from ye3t.cache import YE3TArtifactStore


_HANDLES = OrderedDict()
_LOCK = threading.RLock()
_MAX_HANDLES = 4


def prepared_tagged_model(model):
    """Return cached native model text, including the current readout."""
    from .tagged_cauchy_image import export_tagged_cauchy_image_model, _payload_hash

    evaluator = model.evaluator
    # Public compiler validation checks mutations while reusing completed checks.
    evaluator.compiled.validate(compiler_validation=evaluator.compiler_validation)
    request = {
        "compiler": evaluator.compiled.self_hash,
        "source": evaluator.source_plan,
        "program": evaluator.program,
        "species": list(model.species_order),
        "beta": {name: model.beta_by_species[name].detach().cpu().tolist()
                 for name in model.species_order},
        "offsets": dict(model.offsets),
        "references": model.reference_terms,
        "fit_metadata": getattr(model, "fit_metadata", {}),
    }

    def build():
        with tempfile.TemporaryDirectory(prefix="ye3t_native_export_") as directory:
            path = Path(directory) / "model.json"
            export_tagged_cauchy_image_model(path, model)
            return {"text": path.read_text(encoding="utf-8")}

    # TODO(validation-terminology): migrate the store's serialized certificate
    # field with its reader compatibility policy; this is a validation record.
    resolved = YE3TArtifactStore(mode=getattr(evaluator, "native_cache_mode", None)).resolve(
        "tagged_native_model", "ye3t_prepared_tagged_model_v1",
        {"model_content": _payload_hash(request)}, build,
        certificate={"passed": True, "checks": {"compiler_identity": True}},
        required_certificate_checks=("compiler_identity",))
    return resolved["payload"]["text"], resolved["status"]


class _ResidentHandle:
    def __init__(self, library, path, automatic):
        error = ctypes.create_string_buffer(2048)
        self.handle = library.ye3t_tagged_open(os.fsencode(path), automatic, error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        self.lock = threading.RLock()
        self.policy = library.ye3t_tagged_selected_policy(self.handle).decode("ascii")
        self._finalizer = weakref.finalize(self, library.ye3t_tagged_close, self.handle)


def resident_tagged_handle(library, path, automatic, *, reusable=True, cache_mode=None):
    """Share an exact file/library/policy match; live owners survive eviction.

    Composite artifacts have external dependencies and are excluded by callers.
    Including PID prevents using a parent's cached workspace after fork.
    """
    mode = cache_mode if cache_mode is not None else os.environ.get("YE3T_CACHE_MODE")
    if not reusable or mode in {"off", "rebuild", "refresh"}:
        return _ResidentHandle(library, path, automatic), False
    binary = Path(library._name).resolve()
    stat = binary.stat() if binary.is_file() else None
    binary_identity = ((str(binary), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                       if stat is not None else (library._name, library._handle))
    key = (os.getpid(), binary_identity,
           automatic, hashlib.sha256(Path(path).read_bytes()).digest())
    with _LOCK:
        resident = _HANDLES.pop(key, None)
        hit = resident is not None
        if resident is None:
            resident = _ResidentHandle(library, path, automatic)
        _HANDLES[key] = resident
        while len(_HANDLES) > _MAX_HANDLES:
            _HANDLES.popitem(last=False)
        return resident, hit
