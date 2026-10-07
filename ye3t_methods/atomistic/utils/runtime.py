import os
import shutil
import sys
import tempfile
from pathlib import Path
import uuid


class RuntimePaths:
    """Filesystem locations used by optional runtime caches."""

    def __init__(self, runtime_root, tmp_dir, triton_cache_dir):
        self.runtime_root = Path(runtime_root)
        self.tmp_dir = Path(tmp_dir)
        self.triton_cache_dir = Path(triton_cache_dir)


class _WritableTemporaryDirectory:
    """Portable ``TemporaryDirectory`` using inherited ACLs."""

    def __init__(
        self,
        suffix=None,
        prefix=None,
        dir=None,
        ignore_cleanup_errors=False,
        **_,
    ):
        self._ignore_cleanup_errors = bool(ignore_cleanup_errors)
        self.name = self._make_dir(suffix=suffix or "", prefix=prefix or "tmp", directory=dir)

    @staticmethod
    def _make_dir(suffix, prefix, directory):
        """Create a unique temporary directory under a writable parent."""
        base = Path(directory) if directory is not None else Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            candidate = base / f"{prefix}{uuid.uuid4().hex}{suffix}"
            try:
                candidate.mkdir()
                return str(candidate)
            except FileExistsError:
                continue
        raise FileExistsError(f"Could not create unique temporary directory under {base}")

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()

    def cleanup(self):
        shutil.rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)


def _install_tempfile_workaround():
    """Install a tempdir helper when standard ACL inheritance fails."""
    if (
        os.name == "nt"
        and os.environ.get("YE3T_DISABLE_TEMPFILE_WORKAROUND") != "1"
        and os.environ.get("GNE3_DISABLE_TEMPFILE_WORKAROUND") != "1"
    ):
        tempfile.TemporaryDirectory = _WritableTemporaryDirectory


def configure_runtime_environment(
    runtime_root=None,
    force_local_temp=False,
):
    """Configure runtime cache folders without writing into package source."""
    default_root = Path(runtime_root) if runtime_root is not None else Path(tempfile.gettempdir()) / "ye3t_ace_runtime"
    resolved_root = Path(
        os.environ.get("YE3T_ACE_RUNTIME_DIR")
        or os.environ.get("gne3_ace_RUNTIME_DIR")
        or str(default_root)
    )
    resolved_tmp = Path(
        os.environ.get("YE3T_ACE_TMPDIR")
        or os.environ.get("gne3_ace_TMPDIR")
        or str(resolved_root / "tmp")
    )
    resolved_triton_cache = Path(os.environ.get("TRITON_CACHE_DIR", str(resolved_root / "triton_cache")))

    resolved_root.mkdir(parents=True, exist_ok=True)
    resolved_tmp.mkdir(parents=True, exist_ok=True)
    resolved_triton_cache.mkdir(parents=True, exist_ok=True)

    os.environ["YE3T_ACE_RUNTIME_DIR"] = str(resolved_root)
    os.environ["YE3T_ACE_TMPDIR"] = str(resolved_tmp)
    os.environ["TRITON_CACHE_DIR"] = str(resolved_triton_cache)

    should_force_local_temp = (
        force_local_temp
        or os.environ.get("YE3T_ACE_FORCE_LOCAL_TMP") == "1"
        or os.environ.get("gne3_ace_FORCE_LOCAL_TMP") == "1"
    )
    if should_force_local_temp:
        for key in ("TMPDIR", "TMP", "TEMP"):
            os.environ[key] = str(resolved_tmp)
        tempfile.tempdir = str(resolved_tmp)
        _install_tempfile_workaround()

    return RuntimePaths(
        runtime_root=resolved_root,
        tmp_dir=resolved_tmp,
        triton_cache_dir=resolved_triton_cache,
    )


def _existing_path(value):
    if value is None:
        return None
    path = Path(str(value))
    if path.exists():
        return str(path)
    return None


def _cuda_home_candidate():
    existing = _existing_path(os.environ.get("CUDA_HOME"))
    if existing is not None:
        return existing
    existing = _existing_path(os.environ.get("CUDA_PATH"))
    if existing is not None:
        return existing

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        if _existing_path(prefix / "bin" / "nvcc") is not None:
            return str(prefix)
        if _existing_path(prefix / "targets" / "x86_64-linux" / "include" / "cuda.h") is not None:
            return str(prefix)

    env_prefix = Path(sys.executable).resolve().parent.parent
    if _existing_path(env_prefix / "bin" / "nvcc") is not None:
        return str(env_prefix)
    if _existing_path(env_prefix / "targets" / "x86_64-linux" / "include" / "cuda.h") is not None:
        return str(env_prefix)
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return str(Path(nvcc).resolve().parent.parent)
    return None


def configure_triton_cuda_home():
    """Set CUDA toolkit variables for Triton when a toolkit is visible."""
    cuda_home = _cuda_home_candidate()
    if cuda_home is None:
        return None
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("CUDA_PATH", str(cuda_home))
    return str(cuda_home)


def triton_c_compiler_candidate():
    """Return a C compiler suitable for Triton CUDA helper builds."""
    existing = _existing_path(os.environ.get("CC"))
    if existing is not None:
        return existing

    for name in ("cc", "gcc", "clang"):
        compiler = shutil.which(name)
        if compiler is not None:
            return str(compiler)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env_bin = Path(conda_prefix) / "bin"
        for name in ("x86_64-conda-linux-gnu-gcc", "gcc", "clang", "cc"):
            existing = _existing_path(env_bin / name)
            if existing is not None:
                return existing

    env_bin = Path(sys.executable).resolve().parent
    for name in ("x86_64-conda-linux-gnu-gcc", "gcc", "clang", "cc"):
        existing = _existing_path(env_bin / name)
        if existing is not None:
            return existing
    return None


def configure_triton_c_compiler(required=False):
    """Set CC/CXX for CUDA/Triton JIT builds when a compiler is available."""
    configure_triton_cuda_home()
    compiler = triton_c_compiler_candidate()
    if compiler is None:
        if bool(required):
            raise RuntimeError(
                "CUDA is available, but Triton CUDA helper compilation needs a "
                "Linux C compiler. Activate the pa-sw-e3 conda environment with "
                "the conda compiler package installed, or set CC explicitly."
            )
        return None

    os.environ.setdefault("CC", str(compiler))
    cxx = str(compiler)
    if cxx.endswith("-gcc"):
        cxx = cxx[:-4] + "-g++"
    elif cxx.endswith("gcc"):
        cxx = cxx[:-3] + "g++"
    if Path(cxx).exists():
        os.environ.setdefault("CXX", cxx)
    return str(compiler)
