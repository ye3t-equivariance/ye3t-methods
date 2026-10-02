# Paper linear models: six-element ACE / tagged YE3T comparison

This source-archive example contains the linear MLIP inputs and promoted
deployment models for Li, Mo, Cu, Ni, Si, and Ge in
[Goff and Thompson, arXiv:2609.31895v1, section IV.3](https://arxiv.org/abs/2609.31895).
It is the paper's fixed-feature comparison, not the manufactured quickstart
fixture. The six matched 127-feature ACE and tagged YE3T models, their LAMMPS
input decks, and reference logs are under [lammps](lammps/README.md).
Ni also has the 60, 149, and augmented 196-feature tiers retained by the
deployment bundle. Model bytes and per-model manifests were copied unchanged
from the promoted `ye3t-lammps` bundle; the model manifests contain SHA-256
checksums and the fitted ZBL reference settings.

The editable [config](config.json), [tagged content schedule](tagged_components.json),
[element records](systems), [fitting workflow](run.py), and fixed
[mlearn dataset snapshot](../../data/mlearn/README.md) are included in the
source archive. The model-creation workflow was retained from the original
`ye3t-ace` study. It uses the installed `ye3t-methods` compatibility
modules, with `ye3t` as the separate representation compiler. The paper
workflow uses one-hot chemical channels (`chemical_basis="delta"`);
no learned chemical embedding was fitted for these published models.

## Install and inspect

From the extracted `ye3t-methods` source archive, install the separate
`ye3t` compiler from an available wheel, package index, or sibling source
checkout, then install the methods package with its optional workflow tools.
For sibling source checkouts:

```bash
python -m pip install ../ye3t
python -m pip install '.[examples]'
python examples/publication/cost_comparison/run.py --stage preflight --systems Li
python examples/publication/cost_comparison/run.py --stage prepare --systems Li
```

The no-argument driver reads `config.json` and preflights Si. To work on Ni,
run `python examples/publication/cost_comparison/run.py --systems Ni`; the
`--stage` flag selects one restartable part of the fitting workflow. The
configuration keeps the published basis and fit settings visible:

| Edit | Purpose |
| --- | --- |
| `runtime.default_systems`, `runtime.default_stage` | No-argument driver selection; `--systems` and `--stage` override these. |
| `basis.tensor_orders`, `nmax_by_tensor_order`, `lmax_by_tensor_order` | Rank, radial, and angular limits for the catalogue. |
| `basis.channel_multiplicity_partitions_by_order` | Fixed-content channel partitions; the compiler still determines valid coupling labels. |
| `basis.tagged.tag_counts_s`, `component_schedule` | Tagged source role count and the retained physical-content schedule. |
| `representation.target` | Scalar, even-parity, globally invariant paper target. |
| `runtime.workflow_root`, `runtime.coupling_cache_root` | Output and coupling cache locations. |
| `model.arms`, `model.fit_method`, `model.hyperparameter_search` | Matched fit arms and search settings. |
| `validation.inner_seed`, `validation.inner_fold_count` | Training-only model selection split. |

The `systems/*.json` files expose each element's species, crystal, lattice
constant, and cutoff candidates. `run.py --help` lists the LAMMPS, MPI,
resource, and stage overrides. Changing a published config produces a new
study; it does not change the bundled models or their reference values.

The `prepare` command reads the bundled mlearn snapshot and checks its hashes.
The default output directory is a sibling `ye3t-workflows/MLIP/mlearn_linear_cost_comparison`
directory beside the extracted source archive. Edit `runtime.workflow_root`
and `runtime.coupling_cache_root` in the visible config for another location.
The full catalogue, fit, validation, and LAMMPS workflow can take substantial
CPU time; see `run.py --help` for restartable stages and resource guards.
The fitted model files are already present, so running LAMMPS does not require
refitting.

## Choose an ASE evaluator

The fitted `ye3t_tagged_127` paper artifact is a tagged-plus-ACE composite.
The default local `pip install` builds the native C++ ASE library. With `ye3t`
installed, run both editable ASE examples from the source checkout:

```bash
python examples/publication/cost_comparison/ase_native_density.py
python examples/publication/cost_comparison/ase_native_tagged.py
```

Both scripts show a Li BCC cell, the promoted artifact path, the matching ZBL
manifest, the native evaluator settings, and the retained LAMMPS step-zero
energy. Set `runtime.native_library` to a custom compiled library path when
needed; `None` loads the library installed by the wheel. The tagged evaluator
accepts `runtime.execution_policy` as `direct` or `auto`. The density
evaluator exposes `runtime.neighbor_skin_A`. If you edit the cell or model,
set `validation.lammps_step_zero_energy_eV` to a matching reference or `None`.

To build only the fast ordinary density and tagged ASE paths during install,
use `YE3T_METHODS_BUILD_NATIVE=ase python -m pip install '.[examples]'`.
These native model loaders still include a static bundled yaml-cpp parser;
no separate yaml-cpp system install is needed.

The example uses `YE3TTaggedCauchyCalculator.from_artifact(...,
execution_policy="direct")` for the linear residual and
`YE3TZBLCalculator.from_model_manifest(...)` for the paper's ZBL overlay.
`SumCalculator` combines their energy, forces, and stress. The C++ adapter
loads the ordinary `.yace` backbone and tagged correction into one resident
model. The optional `matscipy` package speeds up neighbor rebuilds for cells
that cannot use the adapter's SciPy cKDTree path; ASE remains the fallback.
The example structure is the 16-atom Li LAMMPS validation cell.

The standalone C++ adapter evaluates the **linear residual**; the example
combines it with the required ZBL reference from the model manifest. ZBL
contributes zero on the example Li cell. The ACE control's promoted `.yace`
also loads directly in ASE through `YE3TYACENativeCalculator.from_artifact`.
The package's compact Torch ASE loader reads saved `.pt` bundles instead.
The paper composite also is not a compact
`LinearModel.read` artifact. See [evaluator choices](../../../docs/evaluators.rst)
for `backend="pytorch"`, `reference`, `native_polynomial`, and `native_cpu`
on newly fitted compact models.

## Run a promoted potential in LAMMPS

Build LAMMPS with the separate
[ye3t-lammps](https://github.com/ye3t-equivariance/ye3t-lammps) ML-YE3T
package. Use ML-PACE for the independent ACE control. From the extracted
source archive:

```bash
cd examples/publication/cost_comparison/lammps/Li
lmp -in in.li_pace_product_127
lmp -in in.li_ye3t_mixed_127
```

The input decks include the ZBL overlay required by the fitted residual
models. Do not evaluate the deployed model without that overlay. The
`ye3t_tagged_127` model is a hash-bound composite: ordinary backbone plus
tagged correction, loaded by `pair_style ye3t`. The independent ACE control
is a `.yace` file loaded by PACE. Other element directories have the same
layout.

Verify the copied model bytes with:

```bash
python examples/publication/cost_comparison/verify_models.py
```

The wheel installs the Python APIs; examples, datasets, and LAMMPS decks are
source-archive assets. `pip install` from the local wheel does not place these
examples in site-packages. This release preparation does not publish either
`ye3t` or `ye3t-methods` to PyPI.

## Basis provenance and scope

The paper config expresses rank and radial/angular caps compactly through
`tensor_orders`, `nmax_by_tensor_order`, and `lmax_by_tensor_order`.
`tagged_components.json` supplies only physical fixed-channel contents and
block sizes. The workflow calls `ye3t.couplings.count`, `plan`, and `compile`
for valid Young/rotation sectors, multiplicities, and coefficients. It does not
list those labels by hand. Core `ye3t.BasisLabel` compact notation describes
resolved output labels; it is not an unchecked input that can invent basis
coordinates. See the [basis input guide](../../../docs/basis_inputs.rst).

The retained original workflow contains more historical analysis and promotion
helpers than this public source-archive route. The present archive includes the
fitting stages and exact deployed models. Full six-element refitting and the
10,000-step LAMMPS replay have not been rerun as part of packaging this copy;
the bundled logs and manifests are the previously promoted evidence.
