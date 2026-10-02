# ye3t-methods

`ye3t-methods` provides fixed-feature linear atomistic models using the
separate `ye3t` representation compiler. Its import name is `ye3t_methods`.
Energies, forces, and ASE stress use eV, eV/Å, and eV/Å³. The package includes
the established `ye3t_ace` module path for compatible saved linear models;
new compact workflows start with `ye3t_methods.Basis` and `LinearModel`.
The tagged compiler's exact coefficient materialization uses SymPy, so SymPy
is a base dependency here even though `ye3t` offers it as a reference extra.

## Install and run

Install the separate `ye3t` compiler and then the local methods distribution.
The methods metadata declares `ye3t>=0.1.0` as a runtime dependency; it
does not vendor the compiler. With sibling source checkouts, install PyTorch,
setuptools, and wheel first, then install `ye3t` without build isolation so its
build can import the installed PyTorch:

```bash
python -m pip install torch 'setuptools>=77,<82' wheel
python -m pip install --no-build-isolation ../ye3t
python -m pip install --no-build-isolation '.[examples]'
```

Optional packages are selected with pip extras after installing the local
`ye3t` checkout:

| Extra | Install from the `ye3t-methods` source directory | Adds |
| --- | --- | --- |
| `fit` | `python -m pip install --no-build-isolation '.[fit]'` | scikit-learn for LASSO, ARDRegression, and other optional linear fitters |
| `neighbors` | `python -m pip install --no-build-isolation '.[neighbors]'` | matscipy for faster neighbor rebuilds on eligible general cells |
| `examples` | `python -m pip install --no-build-isolation '.[examples]'` | plotting and example utilities |

Install all three with `python -m pip install --no-build-isolation '.[examples,fit,neighbors]'`.
ASE remains available for neighbor construction without matscipy.

## C++ evaluators

The tagged and ordinary density ASE evaluators use the CPU C++ library in
`native/`. Installing from source with `--no-build-isolation` builds and
installs the library automatically when `ye3t`, CMake 3.20+, a C++17 compiler,
and `yaml-cpp` development files are already installed. The ASE adapters find
the packaged library automatically. For a separate CMake build, run:

```bash
cmake -S native -B ../build-ye3t-methods-native \
  -DCMAKE_BUILD_TYPE=Release -DYE3T_RUNTIME_SOURCE="$PWD/../ye3t"
cmake --build ../build-ye3t-methods-native --target ye3t_tagged_c_api --parallel
export YE3T_TAGGED_C_API_LIBRARY="$PWD/../build-ye3t-methods-native/libye3t_tagged_c_api.so"
```

The build needs a C++17 compiler, CMake 3.20 or newer, and `yaml-cpp`
development files. For host-specific optimization, configure with
`-DYE3T_NATIVE_CPU=ON`; `-DYE3T_ENABLE_IPO=ON` enables supported
interprocedural optimization.

```python
model = LinearModel.read("tagged.ye3t.json")
atoms.calc = model.ase_calculator(
    backend="native_cpu",
    execution_policy="direct",
)
```

The same library contains the ordinary YACE C++ evaluator. For a density
model whose radial specification passes strict YACE export:

```python
model = LinearModel.read("ordinary.pt")
model.export_lammps("ordinary.yace")  # verifies strict YACE compatibility
atoms.calc = model.ase_calculator(
    backend="native_cpu",
)
```

The [evaluator guide](docs/evaluators.rst) gives the environment variable
option and the distinct settings for descriptor construction and evaluation.
The density default radial basis may not pass strict YACE export; choose a
PACE-compatible basis when fitting for C++ evaluation. The native source is
included under `native/` with the GNU General Public License in that directory.
Source-built wheels include the compiled library for their build platform.

Alternatively, with a compatible `ye3t` distribution available to pip,
install the local wheel:

```bash
python -m pip install ye3t_methods-0.1.0-*.whl
```

The source archive contains these complete scripts and their deterministic
manufactured fixtures. From its extracted `ye3t_methods-0.1.0` directory,
run each script independently:

```bash
python examples/quickstart/density_fit.py
python examples/quickstart/tagged_fit.py
python examples/quickstart/saved_ase_export.py
python examples/quickstart/inspect_features.py
python examples/quickstart/parent_coupling.py
python examples/quickstart/parent_coefficient.py
python examples/quickstart/phi_fit.py
python examples/quickstart/phi_depth2.py
python examples/quickstart/role_density_fit.py
python examples/quickstart/chemical_encoding.py
python examples/quickstart/chemical_channel_selection.py
python examples/quickstart/sklearn_fit.py  # requires the fit extra
```

The fitting scripts use only `examples/quickstart/fixtures` or generated
manufactured labels, write fitted artifacts under a sibling
`ye3t-workflows/quickstart_linear` directory, and require no downloaded data. The chemical encoding script
generates its own two-species source probe. These are numerical workflow
fixtures, not physically validated interatomic potentials. See the
[fixture record](examples/quickstart/fixtures/README.md).

The source archive now also contains the [six-element paper linear
models](examples/publication/cost_comparison/README.md): editable fitting
settings, the fixed mlearn snapshot, exact promoted ACE/tagged YE3T model
files, and LAMMPS input decks for Li, Mo, Cu, Ni, Si, and Ge. Verify the
bundled bytes with
`python examples/publication/cost_comparison/verify_models.py`. The wheel
installs the Python APIs; the paper examples and data are source-archive
assets. Neither distribution has been uploaded to a package index as part
of this local release preparation.

Species are one-hot channels by default. The
[chemical encoding guide](docs/chemical_encoding.rst) and
`examples/quickstart/chemical_encoding.py` show the delta/one-hot source and
a fixed embedding kernel through the low-level provider hook. Custom chemical
providers currently have no saved-model or LAMMPS export contract.

The [linear methods guide](docs/index.rst) includes density, tagged physical
image, explicit Phi, role-density, labels, derivatives, saved formats, stress,
and YACE export. The tagged basis has a
[dedicated page](docs/tagged_basis.rst).
The [evaluator guide](docs/evaluators.rst) lists the supported ASE backend
names, native library setup, optional matscipy neighbor lists, and paper
artifact limits.
The [scikit-learn fit guide](docs/scikit_linear_fit.rst) shows LASSO and
ARDRegression for all three compact basis sources and the saved per-atom
readout uncertainty API.
The [basis input guide](docs/basis_inputs.rst) explains how compact requests
are validated by `ye3t.couplings` and how core notation represents resolved
labels.
The [parent-type guide](docs/parent_types.rst) shows a bounded, exact rank-eight
coupling plan and states the current geometry-materialization limit.
With Sphinx installed, build the guide using `python -m sphinx -b html docs
docs/_build/html` from an extracted source archive.

## Compact linear workflow

The [runnable density quickstart](examples/quickstart/density_fit.py) keeps the
basis and ASE evaluator selections in its visible `config`:

```python
basis = Basis(**config["basis"], backend=config["runtime"]["basis_backend"])
model = LinearModel(basis).fit(structures, regularization=config["model"]["regularization"])
artifact = model.write(config["runtime"]["output_path"])
restored = LinearModel.read(artifact)
atoms = structures[0].copy()
atoms.calc = restored.ase_calculator(backend="pytorch", force_method="analytic_factorized")
```

Training structures must already contain energy and force labels. `fit` reads
their stored labels and does not invoke attached calculators to obtain missing
data. Tagged models additionally require the physical tag counts, radial
degrees, tensor order, and angular degree shown in `tagged_fit.py`. Explicit
cluster `bar_phi` models require motif templates and channels as shown in
`phi_fit.py`.

`print(basis)`, `print(model)`, `basis.labels[j].as_dict()`,
`basis.describe(j)`, and `basis.describe(j, format="latex")` inspect actual
descriptor columns. `feature_index` is the zero-based public column ordinal;
it is not a multiplicity-copy index. The full compiler identity is retained in
the structured label. Text output is bounded and never used to reconstruct a
saved model.

## Supported paths and files

| Source | Compact fit, read, ASE | File | Deployment |
| --- | --- | --- | --- |
| Ordinary density `A`/coupled `B` | Energy and forces | `.pt` | Strict `.yace` only for representable PACE radial specifications; native compiled artifact path is available through the retained low-level API. |
| Tagged physical image | Energy, forces, stress; genuinely nontrivial tag/role sectors | `.ye3t.json` | Hash-bound native tagged export; CPU C ABI when built. |
| Explicit motif `bar_phi` | Energy, forces, stress | `.phi.pt` | ASE reference evaluator; no LAMMPS schema. |
| Filtered role density `A_s`, lifted Cauchy, saved fixed Young descriptor sets | Retained low-level linear APIs | Existing formats where supported | The verified slot-trivial `A_s` fit is primarily a radial-channel change or expansion; compare radial-matched ACE before claiming a distinct benefit. See the release validation record for qualified operations. |

The compact density fit does not accept stress rows. A density `.pt` or Phi
`.phi.pt` file is a Torch artifact; load it only from a trusted source. Tagged
JSON and lifted Cauchy JSON retain their existing versioned schema and hashes.
The linear Python package is independent of the staged research application
tree. `../ye3t-experimental` is preserved for later migration and is not an
install or test prerequisite.

The [release validation record](RELEASE_VALIDATION.md) states exactly which
families, backends, artifacts, and numerical checks were executed in this
checkout. No package index upload is part of this local release preparation.

## Citation

Please cite this software using [CITATION.cff](CITATION.cff) and the YE3T
preprint: James M. Goff and Aidan P. Thompson, *The Young-E(3) Tensor Product
Decomposition for Rotation and Permutation Equivariant Cluster Expansions*,
[arXiv:2609.31895](https://arxiv.org/abs/2609.31895) (2026).
