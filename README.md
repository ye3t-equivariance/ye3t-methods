# ye3t-methods

`ye3t-methods` provides fixed-feature linear atomistic models using the
separate `ye3t` representation compiler. Its import name is `ye3t_methods`.
Energies, forces, and ASE stress use eV, eV/Å, and eV/Å³. The package includes
the `ye3t_ace` import shim for compatible saved linear models. New workflows
use `ye3t_methods.Basis` and `LinearModel`.

## Installation

Clone `ye3t` and `ye3t-methods` into the same directory. From that directory,
install PyTorch and the build tools before the two packages:

```bash
git clone https://github.com/ye3t-equivariance/ye3t.git
git clone https://github.com/ye3t-equivariance/ye3t-methods.git
python -m pip install torch 'setuptools>=77,<82' wheel cmake
python -m pip install --no-build-isolation ./ye3t
python -m pip install --no-build-isolation './ye3t-methods[examples]'
cd ye3t-methods
```

Both package installs must use the same Python interpreter. Building `ye3t`
requires a C++20 compiler; the `ye3t-methods` native ASE evaluators require
C++17 and CMake 3.20 or newer.
The `ye3t` compiler is a runtime dependency and is installed first so its
Torch extension can build without isolation.

Optional extras are `fit` for scikit-learn fitters and `neighbors` for
matscipy neighbor lists. Install both from the `ye3t-methods` directory with
`python -m pip install --no-build-isolation '.[fit,neighbors]'`. ASE neighbor
construction works without matscipy. See [C++ evaluators](#c-evaluators) for
Python-only and native build options.

## First ASE descriptors

For ordinary scalar ACE descriptors from ASE, use the compact constructor:

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

This constructor uses ChebExpCos radial settings. To reproduce a PACE paper
model's radial channels, use its full config or read the saved model.

## Configured descriptors and models

The complete representation selector and seven-section config are shown in
[`ase_descriptors.py`](examples/quickstart/ase_descriptors.py). Use that route
to select a non-scalar O(3) parent, exact catalogue constraints, or a different
physical factor source. Both routes obtain valid labels and coefficients from
`ye3t.couplings`.
The example constructs `ye3t.YE3TRepresentation` and a basis from its config,
then calls
`descriptors = basis.create(atoms)`. The returned real NumPy array has one row
per atom and one column per compiler-selected scalar descriptor.

### Tagged and supplied factors

For tagged scalar ASE rows with nontrivial local Young and angular
intermediates, use
[`tagged_descriptors.py`](examples/quickstart/tagged_descriptors.py).
It uses the same public representation and basis objects and shows local
`(1,1)` Young and `L=1` angular intermediates.
For non-scalar properties, [`ase_octupole_descriptors.py`](examples/quickstart/ase_octupole_descriptors.py)
uses the same interface with an `L=3` parent and returns seven real-tesseral
components per feature.
For factors computed by another method, [`coupled_factors.py`](examples/quickstart/coupled_factors.py)
shows the same representation → basis flow with a Cauchy coupling plan from
`ye3t`. `Basis.create_factors` accepts complete ordered role-by-`m`
multiplets and returns all valid `(a,t,M)` coordinates for the selected parent
Young partition, angular momentum, and product O(3) parity. The core count
report separates independent paths, tableau coordinates, and magnetic
components. This example supplies factors directly; a physical source such as
lifted density, explicit Φ, or a message must construct those factors and any
derivatives before coupling. It does not fit a scalar interatomic potential.
An intrinsic `parity` may be set per supplied factor type; omitting it uses
the polar spherical-harmonic value `(-1)^l`. Existing atomistic channel fields
remain accepted for saved-model compatibility.

### Linear model examples

For fitting, use the same core representation and `Basis.from_config`, then
fit `LinearModel(basis)`. The rank-four
[`density_fit.py`](examples/quickstart/density_fit.py) and
[`tagged_fit.py`](examples/quickstart/tagged_fit.py) quickstarts demonstrate
the configured representation → basis → model path. The tagged example
uses a seven-column angular pattern.
[`combined_density_tagged_fit.py`](examples/quickstart/combined_density_tagged_fit.py)
uses one named-component basis with separate radial sources, one fit, and one
saved model.
The `YE3TRepresentation` re-export from `ye3t_methods` serves the
descriptor-first API and is not interchangeable with the core class used by
`Basis.from_config`.
For the configured workflow, import `YE3TRepresentation` from `ye3t`.
To refit the exact selected 127-column Ni paper basis and evaluate the
published held-out split, run the full
[`refit_paper_ni.py`](examples/publication/cost_comparison/refit_paper_ni.py)
example. Its config supplies the saved source, paper loss weights, training
data, and output paths. The 60- and 149-column models are in the same
directory. A refit writes a new Torch ASE archive; changed coefficients
require a new native plan for LAMMPS export.
Model code imports from `ye3t_methods`; the `ye3t_ace` import shim reads
models saved under the old module name. Do not install the separate
`ye3t-ace` distribution alongside `ye3t-methods` in
one environment, because both supply that saved-model import path.
The tagged compiler's exact coefficient materialization uses SymPy, so SymPy
is a base dependency here even though `ye3t` offers it as a reference extra.

## Fit a linear model

The [runnable density quickstart](examples/quickstart/density_fit.py) shows the
full config, labeled ASE structures, fit, saved model, and calculator. Its main
calls are:

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
their stored labels and does not ask attached calculators to generate missing
data. The [tagged fit](examples/quickstart/tagged_fit.py) also configures
physical tag counts, radial degrees, tensor order, and angular degree.
Experimental `bar_phi` fitting requires motif templates and channels; see
[`phi_fit.py`](examples/experimental/phi_fit.py).

`print(basis)`, `print(model)`, `basis.labels[j].as_dict()`, and
`basis.describe(j, format="latex")` inspect descriptor columns. `feature_index`
is the zero-based column ordinal; the structured label stores the compiler
identity. Saved models are read from artifacts, not reconstructed from text.

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
install a built wheel:

```bash
python -m pip install ye3t_methods-0.1.0-*.whl
```

## Runnable examples

From the `ye3t-methods` source directory, run one of these independent
workflows:

```bash
python examples/quickstart/ase_descriptors.py             # ASE Atoms to NumPy rows
python examples/quickstart/tagged_descriptors.py          # tagged Young/angular rows
python examples/quickstart/chemical_encoding.py            # multi-species scalar rows
python examples/quickstart/chemical_channel_selection.py   # select neighbor species
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
independent scalar columns from ranks through eight. The chemical examples
show a fixed two-channel mixture for three species and a one-channel source
that excludes Na neighbors while retaining Na centers. They follow the
configured representation → basis → descriptor workflow.
The Ni ASE quickstart loads the bundled tagged paper model, adds its ZBL
reference, and checks the 32-atom energy against the bundled LAMMPS result.
It requires the native C++ library installed by the default or ASE-only build.
The portable Ni quickstart reads the SHA-256-checked 127-column `.ye3t` archive,
exposes its ordered descriptor rows, and checks the same energy with the CPU
Torch evaluator and the saved ZBL specification. Its
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

The [six-element paper examples](examples/publication/cost_comparison/README.md)
include editable fitting settings, the fixed mlearn snapshot, bundled
ACE/tagged YE3T model files, and LAMMPS inputs for Li, Mo, Cu, Ni, Si, and Ge.
Verify the bundled bytes with
`python examples/publication/cost_comparison/verify_models.py`. The wheel
installs the Python APIs; the paper examples and data are in the source
distribution.
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
The [parent-type guide](docs/parent_types.rst) shows an exact rank-eight
coupling plan and its geometry-materialization limits.
With Sphinx installed, build the guide from the source directory:

```bash
python -m sphinx -b html docs docs/_build/html
```

## Supported paths and files

| Source | Compact fit, read, ASE | File | Deployment |
| --- | --- | --- | --- |
| Ordinary density `A`/coupled `B` | Energy, forces, stress | `.pt` | Strict `.yace` only for representable PACE radial specifications; a native compiled artifact path is available through the low-level API. |
| Tagged physical image | Energy, forces, stress; nontrivial tag/role sectors | `.ye3t.json` | Hash-bound native tagged export; CPU C ABI when built. |
| Density full-M `L>0` | Per-atom mean and optional ARD component covariance | `.ye3t.json` v2 | ASE CPU evaluator tested through `L=3`; LAMMPS CPU property compute tested for natural-parity `L=1,2`. |
| Tagged full-M `L>0` | Per-atom mean and optional ARD component covariance | `.ye3t.json` v2 | ASE CPU evaluator tested through `L=3`; LAMMPS CPU mean-property compute tested for `L=1,2`, with an experimental Kokkos device path for tagged means. |
| Configured density plus tagged scalar | Energy, forces, stress | indexed `.ye3t` bundle | Torch and native CPU ASE with one species E0 map. |
| Legacy tagged-plus-ACE composite | Read and ASE energy, forces, stress; saved compiler labels unavailable | colocated `model.ye3t.json` files or trusted single-file `.ye3t` compatibility archive | Native CPU composite with one ZBL overlay; the `.ye3t` archive is a compatibility format, not portable coupling-array execution. |
| SHA-256-pinned Ni portable scalar composite | Ordered ordinary/tagged rows; ASE energy, forces, stress | single-file `.ye3t` | CPU Torch ordinary + tagged + saved ZBL; portable 60/127/149 models. |
| Explicit motif `bar_phi` | Energy, forces, stress | `.phi.pt` | ASE reference evaluator; no LAMMPS schema. |
| Filtered role density `A_s`, lifted Cauchy, saved fixed Young descriptor sets | Low-level linear APIs | Source-specific formats | The tested trivial-Young `A_s` fit primarily changes or expands radial channels; compare radial-matched ACE before claiming a distinct benefit. |

The compact density fit accepts stress rows with a positive cell volume. A
density `.pt` or Phi `.phi.pt` file is a Torch artifact; load it only from a
trusted source. Tagged scalar and lifted Cauchy JSON include versioned schemas
and hashes. The compatibility `.ye3t` archive checks internal byte consistency;
compare its digest with an independently recorded value when model origin
matters. The Ni portable reader pins the full archive SHA-256 and saved
69-column tagged program. It exposes `basis.create(atoms)` and
`model.ase_calculator(evaluator="torch")`. It does not provide native or
LAMMPS export from portable members; use the native composite for those
routes.
The [evaluator guide](docs/evaluators.rst) lists supported ASE backends and
the [paper-model guide](docs/paper_models.rst) identifies the bundled
artifacts and their validation inputs.

## Tests

Install the `dev` extra to run the Python suite:

```bash
python -m pip install --no-build-isolation '.[dev]'
python -m pytest
```

This installation builds the native library. To run tests directly from a
checkout without installing the package, install pytest and run
`python setup.py build_ext --inplace` first.
Tests requiring optional fit or neighbor backends skip unless those extras
are installed.

## Citation

Please cite this software using [CITATION.cff](CITATION.cff) and the YE3T
preprint: James M. Goff and Aidan P. Thompson, *The Young-E(3) Tensor Product
Decomposition for Rotation and Permutation Equivariant Cluster Expansions*,
[arXiv:2609.31895](https://arxiv.org/abs/2609.31895) (2026).
