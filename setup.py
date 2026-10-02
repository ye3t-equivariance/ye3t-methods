"""Build the bundled CPU evaluator into source-installed wheels."""

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        root = Path(__file__).resolve().parent
        source = os.environ.get("YE3T_RUNTIME_SOURCE")
        if source is None:
            spec = importlib.util.find_spec("ye3t")
            if spec is not None and spec.origin is not None:
                source = str(Path(spec.origin).resolve().parents[1])
            else:
                source = sysconfig.get_paths()["purelib"]
        source = Path(source).resolve()
        if not (source / "ye3t/runtime/csrc/ye3t_runtime_core.cpp").is_file():
            raise RuntimeError(
                "YE3T C++ runtime source was not found in the installed ye3t package. "
                "Install ye3t first or set YE3T_RUNTIME_SOURCE to its source root."
            )
        build_dir = Path(self.build_temp).resolve() / "ye3t_methods_native"
        output_dir = Path(self.build_lib).resolve() / "ye3t_ace"
        build_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        cmake = shutil.which("cmake")
        if cmake is None:
            local_cmake = Path(sys.prefix) / "bin" / "cmake"
            cmake = str(local_cmake) if local_cmake.is_file() else None
        if cmake is None:
            raise RuntimeError("CMake 3.20 or newer is required to build the native evaluator.")
        args = [
            cmake, "-S", str(root / "native"), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DYE3T_RUNTIME_SOURCE={source}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_PREFIX_PATH={sys.prefix}",
        ]
        for name in ("YE3T_NATIVE_CPU", "YE3T_ENABLE_IPO", "YE3T_USE_SYSTEM_YAML_CPP"):
            if name in os.environ:
                args.append(f"-D{name}={os.environ[name]}")
        args.append(f"-DYE3T_ASE_ONLY={'ON' if native_setting == 'ase' else 'OFF'}")
        subprocess.check_call(args)
        subprocess.check_call([
            cmake, "--build", str(build_dir), "--target", "ye3t_tagged_c_api",
            "--config", "Release", "--parallel", "2",
        ])
        if sys.platform.startswith("linux"):
            library = output_dir / "libye3t_tagged_c_api.so"
            patchelf = shutil.which("patchelf")
            if patchelf is None:
                local_patchelf = Path(sys.prefix) / "bin" / "patchelf"
                patchelf = str(local_patchelf) if local_patchelf.is_file() else None
            if patchelf is not None and library.is_file():
                subprocess.check_call([patchelf, "--remove-rpath", str(library)])


native_setting = os.environ.get("YE3T_METHODS_BUILD_NATIVE", "1").strip()
if native_setting not in {"0", "1", "ase"}:
    raise RuntimeError("YE3T_METHODS_BUILD_NATIVE must be 0, 1, or ase.")

setup(
    ext_modules=(
        [Extension("ye3t_ace.libye3t_tagged_c_api", sources=[])]
        if native_setting != "0" else []
    ),
    cmdclass={"build_ext": CMakeBuild},
)
