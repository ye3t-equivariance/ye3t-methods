# Paper linear models: six-element ACE / tagged YE3T comparison

This example contains the linear MLIP inputs and saved deployment models for
Li, Mo, Cu, Ni, Si, and Ge in
[Goff and Thompson, arXiv:2609.31895v1, section IV.3](https://arxiv.org/abs/2609.31895).
It is the paper's fixed-feature comparison, not the manufactured quickstart
fixture. The six matched 127-feature ACE and tagged YE3T models, their LAMMPS
input decks, and reference logs are under [lammps](lammps/README.md).
Ni also has 60-, 149-, and augmented 196-feature models. Model bytes and
per-model manifests match the `ye3t-lammps` bundle; the manifests contain SHA-256
checksums and the fitted ZBL reference settings.

The editable [config](config.json), [tagged content schedule](tagged_components.json),
[element records](systems), [fitting workflow](run.py), and fixed
[mlearn dataset snapshot](../../data/mlearn/README.md) are included in the
source distribution. The fitting workflow uses `ye3t-methods` compatibility
modules, with `ye3t` as the separate representation compiler. The paper
workflow uses one-hot chemical channels (`chemical_basis="delta"`);
no learned chemical embedding was fitted for these published models.

## Install and inspect

From the `ye3t-methods` source directory with a sibling `ye3t` checkout, install
both packages and run the first preflight:

```bash
python -m pip install torch 'setuptools>=77,<82' wheel cmake
python -m pip install --no-build-isolation ../ye3t
python -m pip install --no-build-isolation '.[examples]'
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
| `basis.tagged.tag_counts_s`, `component_schedule` | Tagged source role count and physical-content schedule. |
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
directory beside the `ye3t-methods` source directory. Edit `runtime.workflow_root`
and `runtime.coupling_cache_root` in the visible config for another location.
The full catalogue, fit, validation, and LAMMPS workflow can take substantial
CPU time; see `run.py --help` for restartable stages and resource guards.
The fitted model files are already present, so running LAMMPS does not require
refitting.

To reproduce the held-out Ni energy and force RMSE for the saved 60-, 127-,
and 149-feature ordinary ACE and tagged YE3T models, run:

```bash
python examples/publication/cost_comparison/reproduce_ni_rmse.py
```

The script reads the bundled 31-frame mlearn Ni test split and the six saved
model manifests under `lammps/Ni`, checks their hashes, includes each model's
ZBL overlay, and compares the resulting errors with the saved accuracy
table. Its JSON report defaults to the sibling
`ye3t-workflows/MLIP/cost_comparison/ni_rmse_recomputed.json` directory. Use
`--dataset-root` for a matching separately downloaded mlearn snapshot,
`--models-root` for a matching deployment bundle, and `--output` to choose a
report path. `--max-frames` is diagnostic only and does not claim baseline
reproduction. The ordinary controls require the packaged native YACE reader;
the tagged models use `LinearModel.read` and the native composite evaluator.

The fixed held-out reference values are:

| Features | Model | Energy RMSE (eV/atom) | Force RMSE (eV/Å) |
| ---: | --- | ---: | ---: |
| 60 | ACE | 0.003267 | 0.153414 |
| 60 | tagged YE3T | 0.000963 | 0.058738 |
| 127 | ACE | 0.001526 | 0.089734 |
| 127 | tagged YE3T | 0.000677 | 0.039418 |
| 149 | ACE | 0.001550 | 0.089478 |
| 149 | tagged YE3T | 0.000643 | 0.036319 |

These are accuracy baselines for the saved linear models, not targets from
refitting a new model. The script records full precision and compares against
the hash-checked reference table.

## Refit the selected Ni model with ASE

[`refit_paper_ni.py`](refit_paper_ni.py) shows the full seven-section config for
refitting the exact saved 127-column basis. It reads all 263 structures in the
published Ni training/validation partition, subtracts the saved ZBL reference,
uses the final selected weights and penalty in
`finalist/fits/ye3t_tagged_127.json` (not the earlier radial-screen trials
also embedded in model manifests), and writes a
new self-contained `.ye3t` artifact plus a 31-structure held-out RMSE report:

```bash
python examples/publication/cost_comparison/refit_paper_ni.py
```

The full refit is CPU-intensive. The 60- and 149-column selected portable
archives are also in `portable_models`; change `source_archive`, its SHA-256,
`feature_count`, and the `fit` hyperparameters in the visible config to refit
either tier. Use that tier's `finalist/fits/ye3t_tagged_<count>.json` selected
record for its hyperparameters. The input model archive fixes radial
functions, descriptor labels, column order, and the ZBL
reference. `LinearModel.read` restores its compiled representation and `Basis`;
rebuilding a nearby catalogue would change the selected paper columns. The fit
changes every selected coefficient and the per-atom
intercept. It never edits the input model archive. The new artifact records hashes
of the training data, selected design rows, residual targets, and fitted normal
equations, plus training RMSE and configured finite-difference/save-load checks.

The refitted bundle is evaluated through
`LinearModel.read(...).ase_calculator(evaluator="torch")`. It has no native
plan for its new weights; LAMMPS AUTO
export requires a separately validated native plan. The script's held-out
Ni-127 run on the published split gave **0.000677664 eV/atom** energy RMSE and
**0.039417111 eV/Å** force RMSE. The saved model gives approximately
**0.0006774455 eV/atom** and **0.0394175951 eV/Å** on that split. Raw fitted
coefficients need not match: the selected ridge normal system has an estimated
condition number of **3.86 × 10¹⁴**, so floating-point accumulation order
changes coefficients along nearly redundant feature directions. Compare
predictions and RMSE when checking a refit.

These are results of a newly fitted model using the saved selected basis and
final fit settings; they do not rerun feature selection or hyperparameter search.

## Choose an ASE evaluator

The source distribution includes self-contained Ni 60-, 127-, and 149-column
portable models with SHA-256 checks. The Ni-127 model at
`portable_models/Ni_ye3t_tagged_127.ye3t` has SHA-256
`a57647406108a71273794e7786252147c93830d8954d8c44d9d7e813cb6502a2`.
With the methods wheel installed, run
`python examples/quickstart/paper_ni_portable_ase.py` from the `ye3t-methods`
source directory. It loads the model with `LinearModel.read`, displays the 127 ordered
descriptor columns through `Basis.create`, and checks complete ordinary,
tagged, and ZBL ASE energy against the bundled Ni LAMMPS step-zero result.
A portable augmented-196 conversion is not included in this distribution.

The fitted `ye3t_tagged_127` paper artifact is a tagged-plus-ACE composite.
The default local `pip install` builds the native C++ ASE library. With `ye3t`
installed, run the editable ASE reference examples from the source checkout:

```bash
python examples/publication/cost_comparison/ase_native_density.py
python examples/publication/cost_comparison/ase_native_tagged.py
python examples/publication/cost_comparison/ase_native_ni.py
```

The first two scripts show a Li BCC cell, the bundled model path, the matching ZBL
manifest, the native evaluator settings, and the reference LAMMPS step-zero
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
The Li example structure is the 16-atom LAMMPS validation cell. The Ni script
checks its 32-atom tagged model against the reference step-zero energy.

The standalone C++ adapter evaluates the **linear residual**; the example
combines it with the required ZBL reference from the model manifest. ZBL
contributes zero on the example Li cell. The ACE control's `.yace`
also loads directly in ASE through `YE3TYACENativeCalculator.from_artifact`.
The package's compact Torch ASE loader reads saved `.pt` bundles instead.
The paper composite also is not a compact
`LinearModel.read` artifact. See [evaluator choices](../../../docs/evaluators.rst)
for `backend="pytorch"`, `reference`, `native_polynomial`, and `native_cpu`
on newly fitted compact models.

## Run a bundled potential in LAMMPS

Build LAMMPS with the separate
[ye3t-lammps](https://github.com/ye3t-equivariance/ye3t-lammps) ML-YE3T
package. Use ML-PACE for the independent ACE control. From the `ye3t-methods`
source directory:

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
source distribution files. `pip install` from the local wheel does not place
these examples in site-packages.

## Basis provenance and scope

The paper config expresses rank and radial/angular caps compactly through
`tensor_orders`, `nmax_by_tensor_order`, and `lmax_by_tensor_order`.
`tagged_components.json` supplies only physical fixed-channel contents and
block sizes. The workflow calls `ye3t.couplings.count`, `plan`, and `compile`
for valid Young/rotation sectors, multiplicities, and coefficients. It does not
list those labels by hand. Core `ye3t.BasisLabel` compact notation describes
resolved output labels; it is not an unchecked input that can invent basis
coordinates. See the [basis input guide](../../../docs/basis_inputs.rst).

The source distribution includes the fitting stages and saved deployment
models. Full six-element refitting and the 10,000-step LAMMPS replay were not
rerun when this copy was packaged; the bundled logs and manifests record the
reference runs.
