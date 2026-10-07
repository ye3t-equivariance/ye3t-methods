# ye3t-methods

`ye3t-methods` provides fixed-feature linear atomistic models using the
separate `ye3t` representation compiler. Its import name is `ye3t_methods`.
Energies, forces, and ASE stress use eV, eV/Å, and eV/Å³. The package includes
the established `ye3t_ace` module path for compatible saved linear models;
new compact workflows start with `ye3t_methods.Basis` and `LinearModel`.
For ordinary scalar ACE descriptors from ASE, the compact constructor is:

```python
from ase.build import bulk
from ye3t_methods import Basis

atoms = bulk("Ni", "fcc", a=3.52, cubic=True)
basis = Basis(elements=["Ni"], cutoff=5.0, max_rank=3,
              nmax=(2, 2, 1), lmax=(1, 1, 1))
descriptors = basis.create(atoms)  # (n_atoms, n_features) NumPy array
many = basis.create_many([atoms, atoms.copy()])  # tuple of per-structure arrays
```

The same call handles multiple elements. For example, ASE's H2O molecule
returns three rows and 24 columns with:

```python
from ase.build import molecule
from ye3t_methods import Basis

atoms = molecule("H2O")
basis = Basis(elements=["H", "O"], cutoff=4.0, max_rank=2,
              nmax=(2, 1), lmax=(1, 1))
water_descriptors = basis.create(atoms)  # (3, 24)
```

This compact constructor retains the original ChebExpCos source settings;
it does not recreate a PACE paper model's radial channels. Use the complete
configured route or read the saved model when exact physical source identity
matters.

The complete representation selector and seven-section config are shown in
[`ase_descriptors.py`](examples/quickstart/ase_descriptors.py). Use that route
to select a non-scalar O(3) parent, exact catalogue constraints, or a different
physical factor source. Both routes obtain valid labels and coefficients from
`ye3t.couplings`.
The full example constructs the
`ye3t.YE3TRepresentation` and basis from its visible config, then calls
`descriptors = basis.create(atoms)`. The returned real NumPy array has one row
per atom and one column per compiler-selected scalar descriptor.
For non-scalar properties, [`ase_octupole_descriptors.py`](examples/quickstart/ase_octupole_descriptors.py)
uses the same interface with an `L=3` parent and returns seven real-tesseral
components per feature.
For fitting, use the same core representation and `Basis.from_config`, then
fit `LinearModel(basis)`. The rank-four
[`density_fit.py`](examples/quickstart/density_fit.py) and
[`tagged_fit.py`](examples/quickstart/tagged_fit.py) quickstarts demonstrate
the configured representation → basis → model path. The tagged example
explicitly selects its previous seven-column angular pattern.
[`combined_density_tagged_fit.py`](examples/quickstart/combined_density_tagged_fit.py)
uses one named-component basis with separate radial sources, one fit, and one
saved model.
The `YE3TRepresentation` re-export from `ye3t_methods` is the older
descriptor-first selector; it is not interchangeable with the core class used
by `Basis.from_config`.
For the configured workflow, import `YE3TRepresentation` from `ye3t`.
Maintained model code imports from `ye3t_methods`; a small `ye3t_ace` import
shim remains only to read models saved under historical module names. Do not
install the historical `ye3t-ace` distribution alongside `ye3t-methods` in
one environment, because both supply that saved-model import path.
The tagged compiler's exact coefficient materialization uses SymPy, so SymPy
is a base dependency here even though `ye3t` offers it as a reference extra.

## Install and run

Install the separate `ye3t` compiler and then the local methods distribution.
The methods metadata declares `ye3t>=0.1.0` as a runtime dependency; it
does not vendor the compiler. With sibling source checkouts, install PyTorch,
setuptools, and wheel first, then install `ye3t` without build isolation so its
build can import the installed PyTorch:

```bash
python -m pip install torch 'setuptools>=77,<82' wheel cmake
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
installs the library automatically when `ye3t`, CMake 3.20+, and a C++17
compiler are installed. The build uses its bundled yaml-cpp 0.8.0 source;
the ASE adapters find the packaged library automatically. For a separate
CMake build, run:

```bash
cmake -S native -B ../build-ye3t-methods-native \
  -DCMAKE_BUILD_TYPE=Release -DYE3T_RUNTIME_SOURCE="$PWD/../ye3t"
cmake --build ../build-ye3t-methods-native --target ye3t_tagged_c_api --parallel
export YE3T_TAGGED_C_API_LIBRARY="$PWD/../build-ye3t-methods-native/libye3t_tagged_c_api.so"
```

The build needs a C++17 compiler and CMake 3.20 or newer. To use an installed
yaml-cpp instead, configure with `-DYE3T_USE_SYSTEM_YAML_CPP=ON` and set
`CMAKE_PREFIX_PATH` if needed. For host-specific optimization, configure with
`-DYE3T_NATIVE_CPU=ON`; `-DYE3T_ENABLE_IPO=ON` enables supported
interprocedural optimization.

For a Python-only installation without a C++ toolchain, use
`YE3T_METHODS_BUILD_NATIVE=0 python -m pip install --no-build-isolation .`.
The PyTorch evaluators remain available; `backend="native_cpu"` requires a
native build.

To build only the fast ordinary-density and tagged ASE evaluators, use
`YE3T_METHODS_BUILD_NATIVE=ase python -m pip install --no-build-isolation .`.
This skips the lifted native evaluator sources. It still includes statically
linked yaml-cpp for loading YACE and tagged model artifacts.

```python
model = LinearModel.read("tagged.ye3t.json")
atoms.calc = model.ase_calculator(
    evaluator="native_cpu",
    neighbors="ase",
    execution_policy="direct",
)
```

The same library contains the ordinary YACE C++ evaluator. For a density
model whose radial specification passes strict YACE export:

```python
model = LinearModel.read("ordinary.pt")
model.export_lammps("ordinary.yace")  # verifies strict YACE compatibility
atoms.calc = model.ase_calculator(
    evaluator="native_cpu",
    neighbors="ase",
)
```

The [evaluator guide](docs/evaluators.rst) gives the environment variable
option and the distinct settings for descriptor construction and evaluation.
Native scalar ASE calculators also accept `neighbors="auto"` and
`neighbors="matscipy"`; the latter requires the optional `neighbors`
extra. A compatible standalone tagged or density model with
`evaluator="auto"` selects native CPU when an explicit neighbor builder is
requested.
The density default radial basis may not pass strict YACE export; choose a
PACE-compatible basis when fitting for C++ evaluation. The native source is
included under `native/` with the GNU General Public License in that directory.
Native source-built wheels include the compiled library for their build
platform; the Python-only option produces a pure wheel.

Alternatively, with a compatible `ye3t` distribution available to pip,
install the local wheel:

```bash
python -m pip install ye3t_methods-0.1.0-*.whl
```

The source archive contains the complete scripts and their deterministic
fixtures. From its extracted `ye3t_methods-0.1.0` directory, start with one
workflow:

```bash
python examples/quickstart/ase_descriptors.py             # ASE Atoms to NumPy rows
python examples/quickstart/chemical_encoding.py            # multi-species scalar rows
python examples/quickstart/density_fit.py                 # fit a scalar model
python examples/quickstart/combined_density_tagged_fit.py # combine physical sources
python examples/quickstart/per_atom_vector_to_lammps.py   # fit and export a vector model
python examples/quickstart/paper_ni_portable_ase.py       # read the pinned Ni model
python examples/quickstart/paper_ni_nve.py                # run 100 ASE NVE steps
```

These run independently. The vector script writes LAMMPS input for the
separate native application; its fitted property labels are manufactured
workflow data. Further tagged, lifted, explicit-Φ, source, solver, and
inspection examples are listed in the [quickstart guide](docs/quickstart.rst).

The scalar Ni descriptor example builds an fcc ASE cell and returns 96
independent scalar columns from ranks through eight. The chemical example
uses a fixed two-channel embedding for three species. Both follow the
configured representation → basis → descriptor workflow.
The Ni ASE quickstart loads the promoted tagged paper model, adds its ZBL
reference, and checks the 32-atom energy against the retained LAMMPS result.
It requires the native C++ library installed by the default or ASE-only build.
The portable Ni quickstart reads the vetted 127-column `.ye3t` archive from
the source archive, exposes its ordered descriptor rows, and checks the same
energy with the CPU Torch evaluator and the archived ZBL specification. Its
model bytes have SHA-256
`a57647406108a71273794e7786252147c93830d8954d8c44d9d7e813cb6502a2`.

Most short fitting scripts use `examples/quickstart/fixtures` or generated
manufactured labels and write fitted artifacts under a sibling
`ye3t-workflows/quickstart_linear` directory. The Ni density fit example
uses the bundled labeled mlearn snapshot and published training split.
The chemical encoding script creates a three-species ASE descriptor probe.
The short numerical workflow fixtures are not physically validated
interatomic potentials. See the
[fixture record](examples/quickstart/fixtures/README.md).
The vector quickstart uses a displaced periodic 32-atom Cu crystal and saves
a per-atom `L=1` model plus a ready-to-run LAMMPS property input. It exercises
rank-three and rank-four repeated-content ACE coordinates. The analytic
site-vector labels demonstrate the workflow; they are not measured Cu labels.

The source archive now also contains the [six-element paper linear
models](examples/publication/cost_comparison/README.md): editable fitting
settings, the fixed mlearn snapshot, exact promoted ACE/tagged YE3T model
files, and LAMMPS input decks for Li, Mo, Cu, Ni, Si, and Ge. Verify the
bundled bytes with
`python examples/publication/cost_comparison/verify_models.py`. The wheel
installs the Python APIs; the paper examples and data are source-archive
assets.
To recompute the saved Ni 60/127/149-model held-out energy and force RMSE
from the bundled 31-frame split, run
`python examples/publication/cost_comparison/reproduce_ni_rmse.py` after a
native installation. It checks the dataset, split, model, and manifest hashes
and includes each model's ZBL overlay; the
[paper example guide](examples/publication/cost_comparison/README.md) gives
the six reference values and output options.

Species are one-hot channels by default. The
[chemical encoding guide](docs/chemical_encoding.rst) and
`examples/quickstart/chemical_encoding.py` show a configured fixed
embedding that reduces the physical chemical channel count. The fixed matrix
is part of the basis identity; strict PACE/YACE export still requires delta
channels. Arbitrary low-level chemical providers have no general saved-model
or LAMMPS export contract.

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
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(
    config["basis"], representation=representation, runtime=config["runtime"]
)
model = LinearModel(basis).fit(structures, config=config)
artifact = model.write(config["metadata"]["output_path"])
restored = LinearModel.read(artifact)
atoms = structures[0].copy()
atoms.calc = restored.ase_calculator(evaluator="torch", neighbors="ase")
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
| Ordinary density `A`/coupled `B` | Energy, forces, stress | `.pt` | Strict `.yace` only for representable PACE radial specifications; native compiled artifact path is available through the retained low-level API. |
| Tagged physical image | Energy, forces, stress; genuinely nontrivial tag/role sectors | `.ye3t.json` | Hash-bound native tagged export; CPU C ABI when built. |
| Density full-M `L>0` | Per-atom mean and optional ARD component covariance | `.ye3t.json` v2 | ASE CPU evaluator tested through `L=3`; LAMMPS CPU property compute qualified for natural-parity `L=1,2`. |
| Tagged full-M `L>0` | Per-atom mean and optional ARD component covariance | `.ye3t.json` v2 | ASE CPU evaluator, tested through `L=3`; separate LAMMPS CPU mean-property compute is qualified for `L=1,2`, with a bounded experimental Kokkos device path for tagged means. |
| Configured density plus tagged scalar | Energy, forces, stress | indexed `.ye3t` bundle | Torch and native CPU ASE with one species E0 map. |
| Legacy tagged-plus-ACE composite | Read and ASE energy, forces, stress; saved compiler labels unavailable | colocated `model.ye3t.json` files or trusted single-file `.ye3t` compatibility archive | Native CPU composite with one ZBL overlay; the `.ye3t` archive is a compatibility format, not portable coupling-array execution. |
| Vetted Ni portable scalar composite | Ordered ordinary/tagged rows; ASE energy, forces, stress | SHA-256-pinned single-file `.ye3t` | CPU Torch ordinary + tagged + saved ZBL; exact 60/127/149/196 candidates only. |
| Explicit motif `bar_phi` | Energy, forces, stress | `.phi.pt` | ASE reference evaluator; no LAMMPS schema. |
| Filtered role density `A_s`, lifted Cauchy, saved fixed Young descriptor sets | Retained low-level linear APIs | Existing formats where supported | The verified slot-trivial `A_s` fit is primarily a radial-channel change or expansion; compare radial-matched ACE before claiming a distinct benefit. |

The compact density fit accepts stress rows with a positive cell volume. A density `.pt` or Phi
`.phi.pt` file is a Torch artifact; load it only from a trusted source. Tagged
scalar and lifted Cauchy JSON retain their versioned schema and hashes.
The legacy `.ye3t` archive checks internal byte consistency; compare its digest
with an independently retained value when model origin matters.
The bounded Ni portable reader pins the full archive SHA-256 and saved
69-column tagged program. It exposes `basis.create(atoms)` and
`model.ase_calculator(evaluator="torch")`. It does not provide native or
LAMMPS export from portable members; the retained native composite remains
available for those uses.
The [evaluator guide](docs/evaluators.rst) lists supported ASE backends and
the [paper-model guide](docs/paper_models.rst) identifies the promoted
artifacts and their validation inputs.

## Citation

Please cite this software using [CITATION.cff](CITATION.cff) and the YE3T
preprint: James M. Goff and Aidan P. Thompson, *The Young-E(3) Tensor Product
Decomposition for Rotation and Permutation Equivariant Cluster Expansions*,
[arXiv:2609.31895](https://arxiv.org/abs/2609.31895) (2026).
