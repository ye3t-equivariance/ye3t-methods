# Native CPU evaluators

This directory contains the standalone YE3T CPU evaluator and C ABI used by
the ASE adapters. It supports ordinary YACE density models, tagged models, and
tagged models with an ordinary YACE backbone. The C++ files were copied from
`ye3t-lammps` on 2026-10-02, from working tree revision
`09fee77c5c21c9783d84f8355eaa515ce8ead12a` (including a local change to
`src/ye3t_tagged_c_api.cpp`). The original source headers are retained.

The native sources are covered by the GNU General Public License provided in
[LICENSE](LICENSE). The Python package's root BSD 3-Clause license does not
replace the license on these native files.

With `ye3t` already installed, CMake 3.20+, and a C++17 compiler,
`python -m pip install --no-build-isolation .` builds this library into the
installed package. The build reads runtime C++ source from installed `ye3t`;
set `YE3T_RUNTIME_SOURCE` to a checkout if needed. Set
`YE3T_METHODS_BUILD_NATIVE=0` during pip installation to skip the native build
and install the Python evaluators only. For a separate CMake build:

```sh
cmake -S native -B ../build-ye3t-methods-native \
  -DYE3T_RUNTIME_SOURCE=/path/to/ye3t
cmake --build ../build-ye3t-methods-native --parallel
```

The build includes a static yaml-cpp 0.8.0 library from `third_party/yaml-cpp`
(MIT license in its `LICENSE` file), so no system yaml-cpp installation or
download is needed. The tagged and lifted JSON model loaders use it as well
as the YACE loader. Set `YE3T_USE_SYSTEM_YAML_CPP=ON` to use an installed
yaml-cpp package instead. The output is
`../build-ye3t-methods-native/libye3t_tagged_c_api.so` on Linux. Set
`YE3T_NATIVE_CPU=ON` to use host CPU instructions, or `YE3T_ENABLE_IPO=ON` for
interprocedural optimization. The build requires C++17; it does
not require LAMMPS.
