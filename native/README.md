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

Build with CMake and a YE3T source checkout containing its C++ runtime core:

```sh
cmake -S native -B ../build-ye3t-methods-native \
  -DYE3T_RUNTIME_SOURCE=/path/to/ye3t \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build ../build-ye3t-methods-native --parallel
```

The output is `../build-ye3t-methods-native/libye3t_tagged_c_api.so` on Linux. Set
`YE3T_NATIVE_CPU=ON` to use host CPU instructions, or `YE3T_ENABLE_IPO=ON` for
interprocedural optimization. The build requires C++17 and `yaml-cpp`; it does
not require LAMMPS.
