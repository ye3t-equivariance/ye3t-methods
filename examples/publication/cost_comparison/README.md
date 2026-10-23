# Paper linear models: six-element ACE / tagged YE3T comparison

This example contains the linear MLIP inputs and saved deployment models for
Li, Mo, Cu, Ni, Si, and Ge in
[Goff and Thompson, arXiv:2609.31895v1, section IV.3](https://arxiv.org/abs/2609.31895).
It contains the paper's fixed-feature comparison. The six matched 127-feature ACE and tagged YE3T models, their LAMMPS
input decks, and reference logs are under [lammps](lammps/README.md).
Ni also has 60-, 149-, and augmented 196-feature models. Model bytes and
per-model manifests match the `ye3t-lammps` bundle; the manifests contain SHA-256
checksums and the fitted ZBL reference settings.

## Install and run

Install `ye3t` and `ye3t-methods[examples]` following the package README.
Use the source archive for the scripts, data and saved models in this directory.
The wheel provides the Python APIs and native library.

The public scripts expose the same seven config sections as the quickstarts:
`metadata`, `basis`, `representation`, `runtime`, `model`, `targets`, and
`validation`. Edit paths and scientific settings directly in each script.
Output paths default to `~/ye3t-workflows`; set `YE3T_WORKFLOW_ROOT` or edit
the config to choose another directory.

### Fit and run Ni-60

Use one of these independent script pairs:

```bash
python examples/publication/cost_comparison/paper_ni_60_density_01_prep_train.py
python examples/publication/cost_comparison/paper_ni_60_density_02_run.py

python examples/publication/cost_comparison/paper_ni_60_tagged_01_prep_train.py
python examples/publication/cost_comparison/paper_ni_60_tagged_02_run.py
```

The first script inspects the paper's selected compiler labels, evaluates
descriptors and summed position gradients, fits all 263 development structures,
and evaluates the separate 31 frames. It writes a portable `.ye3t` model and a
LAMMPS export directory including AUTO execution data and the ZBL overlay.
Choose a new export directory when repeating a fit. The second script loads
the fitted model and runs 100 ASE NVE steps using the configured evaluator.
The density model has 60 density features; the tagged model has 15 density and
45 tagged features. Both include ranks 1, 2, 3, 4, 6 and 8.

The original six-element feature selection, fitting and deployment study is
preserved under [tools/publication/cost_comparison](../../../tools/publication/README.md).
That research driver uses its historical study-specific config. Its physical
content schedules, element records and restartable stages remain available for
reproducing the original study. The examples here use the public model workflow.

To reproduce the held-out Ni energy and force RMSE for the saved 60-, 127-,
and 149-feature ordinary ACE and tagged YE3T models, run:

```bash
python examples/publication/cost_comparison/reproduce_ni_rmse.py
```

The script reads the bundled 31-frame mlearn Ni test split and the six saved
model manifests under `lammps/Ni`, checks their hashes, includes each model's
ZBL overlay, and compares the resulting errors with the saved accuracy
table. Its JSON report defaults to
`~/ye3t-workflows/MLIP/cost_comparison/ni_rmse_recomputed.json`. Edit `metadata.dataset_root`, `metadata.models_root` and
`metadata.output_summary` in the visible config to use another matching data/model bundle
or output location. `validation.max_frames` limits a diagnostic run; full baseline
reproduction requires all 31 structures. The ordinary controls use the packaged
native YACE reader; the tagged models use the native composite evaluator.

The fixed held-out reference values are:

| Features | Model | Energy RMSE (eV/atom) | Force RMSE (eV/Å) |
| ---: | --- | ---: | ---: |
| 60 | ACE | 0.003267 | 0.153414 |
| 60 | tagged YE3T | 0.000963 | 0.058738 |
| 127 | ACE | 0.001526 | 0.089734 |
| 127 | tagged YE3T | 0.000677 | 0.039418 |
| 149 | ACE | 0.001550 | 0.089478 |
| 149 | tagged YE3T | 0.000643 | 0.036319 |

These are accuracy baselines for the saved linear models. The script records full precision and compares against
the hash-checked reference table.

## Refit the selected Ni model with ASE

[`refit_paper_ni.py`](refit_paper_ni.py) shows the full seven-section config for
refitting the exact saved 127-column basis. It reads all 263 structures in the
published Ni training/validation partition, subtracts the saved ZBL reference,
uses the final selected weights and penalty in
`finalist/fits/ye3t_tagged_127.json`, and writes a
new self-contained `.ye3t` artifact plus a 31-structure held-out RMSE report:

```bash
python examples/publication/cost_comparison/refit_paper_ni.py
```

The refit caches geometry rows across processes in the shared application
cache, normally `~/.cache/ye3t/applications`. Set `runtime.cache` to
`{"mode": "auto", "directory": null}` in JSON (`None` in Python), or choose
`off` to measure fresh row computation and `read_only` to require cached rows.
Changes to fitting weights and targets reuse the geometry rows. The saved
model archive supplies the selected catalogue; general coupling compilation
also has a shared per-user cache under `~/.cache/ye3t`.

The 60- and 149-column selected portable
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
`LinearModel.read(...).ase_calculator(evaluator="torch")` or `native_cpu`.
`model.export_lammps(directory)` generates execution tables for the fitted
weights. The script's held-out
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
descriptor columns through `Basis.create`, and displays complete ordinary,
tagged, and ZBL ASE energy for comparison with the bundled Ni LAMMPS step-zero result.
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
update the optional reference energy in `validation`. The example regression tests check the bundled values.

To build only the fast ordinary density and tagged ASE paths during install,
use `YE3T_METHODS_BUILD_NATIVE=ase python -m pip install '.[examples]'`.
These native model loaders still include a static bundled yaml-cpp parser;
no separate yaml-cpp system install is needed.

The tagged examples load the saved model through `LinearModel.read` and
`model.ase_calculator(evaluator="native_cpu")`. This includes the ordinary
backbone, tagged contribution and ZBL reference. The Li density example
demonstrates direct `.yace` interoperability using `YE3TYACENativeCalculator`
with the manifest's ZBL reference through ASE's `SumCalculator`.
The Li examples use the 16-atom validation cell; the Ni example uses 32 atoms.
See [evaluator choices](../../../docs/evaluators.rst) for the runtime settings
available to newly fitted models and saved paper models.

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
python tools/publication/cost_comparison/verify_models.py
```

The wheel installs the Python APIs; examples, datasets, and LAMMPS decks are
source distribution files. `pip install` from the local wheel does not place
these examples in site-packages.

## Basis provenance and scope

The archived paper config expresses rank and radial/angular limits through
`tensor_orders`, `nmax_by_tensor_order`, and `lmax_by_tensor_order`.
Its `tagged_components.json` supplies physical fixed-channel contents and
block sizes. Both the archived study and the standardized public workflows
obtain Young/rotation sectors, multiplicities and coefficients from
`ye3t.couplings.count`, `plan`, and `compile`.
Core `ye3t.BasisLabel` compact notation describes
resolved output labels; it is not an unchecked input that can invent basis
coordinates. See the [basis input guide](../../../docs/basis_inputs.rst).

The source distribution includes the fitting stages and saved deployment
models. Full six-element refitting and the 10,000-step LAMMPS replay were not
rerun when this copy was packaged; the bundled logs and manifests record the
reference runs.
