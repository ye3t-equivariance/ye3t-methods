# ASE descriptors and linear models

Install `ye3t` and `ye3t-methods` in the same Python environment. These scripts
import the installed packages and contain their complete editable configs.
Run them by path from any working directory. Small input datasets are colocated
under `quickstart/fixtures`; paper examples also use the supplied publication
models and dataset snapshot. Outputs default to `~/ye3t-workflows`. Set `YE3T_WORKFLOW_ROOT` or edit the
visible output paths in `config["metadata"]` to choose another location.

The common workflow is:

```python
representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation,
                          runtime=config["runtime"])
descriptors = basis.create(atoms)
model = LinearModel(basis).fit(structures, config=config)
```

## Descriptors

| Example | Inputs and output |
| --- | --- |
| [Scalar density](quickstart/ase_descriptors.py) | Ni ASE structure; 100 scalar features across ranks 1–8. |
| [Tagged](quickstart/tagged_descriptors.py) | 284 independent features across ranks 1–8; mixed internal Young/angular couplings and a scalar parent. Uses native CPU evaluation. |
| [Octupole](quickstart/ase_octupole_descriptors.py) | Ranks 1–4; 51 features, each with all seven `L=3` components. |
| [Chemical embedding](quickstart/chemical_encoding.py) | Fixed two-channel embedding for Ni, Cu and Al; ranks 1–4. |
| [Neighbor species selection](quickstart/chemical_channel_selection.py) | Select Li neighbors through a fixed embedding; ranks 1–4. |
| [Feature inspection](quickstart/inspect_features.py) | Structured labels, readable formulas and descriptor values. |
| [Supplied factors](quickstart/coupled_factors.py) | Core Cauchy plan passed into `Basis`; nontrivial parent with complete tableau/magnetic components. |
| [Selected explicit-Phi cluster](advanced/explicit_phi_cluster.py) | One eight-factor cluster, one selected `(4,4), L=2` coupling and output shape `(1,14,5)`. |

Initial tagged compilation can take longer than descriptor evaluation. The
compiler cache is reused when the same source and representation are requested.
Scientific settings are visible; no example truncates the feature list to a
small arbitrary number.

Reuse a basis directly with `basis.create(other_atoms)` or
`basis.create_many([atoms, other_atoms])`. The tagged quickstart's explicit
`runtime.evaluator="native_cpu"` evaluates radial/angular sources and the compiled
features in C++, with the configured neighbor builder. Its first call prepares
the native evaluator; later calls reuse it. Scalar tagged image bases also
support `values, jacobian = basis.create(atoms, derivatives=True)`; the default
returns only values. Jacobian axes are `[center, feature, displaced_atom, xyz]`.
The optional Jacobian uses the analytic product-rule path with Torch sources
and native polynomial evaluation. It can be large for large atom counts.

For repeated tagged fits, `runtime.cache.mode="auto"` also reuses target-free
energy, force and stress rows. Changing fit weights, targets or the solver
does not require evaluating those rows again. Geometry, source and compiler
identities are checked before reuse. `cache.mode="off"` recomputes them.
The cache is under `~/.cache/ye3t/applications` by default; set
`YE3T_ACE_CACHE_DIR` to choose its location separately from model outputs.

The [large tagged fit](long/tagged_fit_large.py) uses 691 independent features,
including more radial channels at high ranks and the rank-four `(2,2,2,2)`
angular pattern. Initial compilation and complete validation took roughly
twenty minutes in the development run. Its full execution test is marked
`long_example`; use `pytest -m long_example` to run it explicitly, or
`pytest -m 'not long_example'` for the shorter examples and other regressions.
The complete fit/save/reload check exceeded 10 GiB of process memory in the
development run; its JSON model is approximately 256 MiB. Start with the
284-feature quickstarts when exploring the interface.

## Fitting and deployment

| Example | Workflow |
| --- | --- |
| [Density fit](quickstart/density_fit.py) | Fit the 100-feature density basis to real Ni energies and forces. |
| [Tagged fit](quickstart/tagged_fit.py) | Fit the full tagged example catalogue, save/reload and use an ASE calculator. |
| [Combined fit](quickstart/combined_density_tagged_fit.py) | One fit and archive for the density and tagged sources together. |
| [Shell-density fit](quickstart/shell_density_fit.py) | Shell-role construction with internal joint couplings and scalar energy output. |
| [LASSO and ARD](quickstart/sklearn_fit.py) | Alternative fitters and ARD uncertainty; requires the `fit` extra. |
| [Saved tagged export](quickstart/saved_ase_export.py) | Run after `tagged_fit.py`; writes a LAMMPS model, AUTO input and atomic data. |
| [Vector property](quickstart/per_atom_vector_to_lammps.py) | Fit analytic Cu vector labels and prepare `compute ye3t/property/atom`. |
| [Saved paper model](quickstart/paper_ni_portable_ase.py) | Evaluate the complete Ni-127 potential and its descriptors. |
| [Paper Ni-60 models](quickstart/paper_ni_60.py) | Both saved density/tagged fits in native ASE, plus the exact 60 tagged-model descriptor columns on two structure sizes. |
| [100-step ASE MD](quickstart/paper_ni_nve.py) | NVE with the saved Ni-127 model, deterministic velocities and CSV output. |

The short scalar fits use 12 labeled structures from the Ni fitting split and
one held-out structure for example prediction. See the [fixture provenance](quickstart/fixtures/README.md).
Use the full paper workflow to reproduce accuracy results. The vector fixture
has analytic labels and a reproducible generator; it demonstrates fitting a
covariant observable.

Run the saved-model export in this order, with the same output root:

```bash
python examples/quickstart/tagged_fit.py
python examples/quickstart/saved_ase_export.py
```

The exported `in.ye3t` uses the tagged AUTO evaluator and the accompanying
JSON model includes its execution plans. Running that input requires a LAMMPS
installation with ML-YE3T. The vector example prepares a per-atom property
compute; its vector predictions are an observable, so its LAMMPS input uses
`pair_style zero`.

## Published model results

- [Refit the selected Ni basis](publication/cost_comparison/refit_paper_ni.py):
  full 263-structure fitting partition and 31-structure evaluation partition.
- [Reproduce all six Ni RMSE rows](publication/cost_comparison/reproduce_ni_rmse.py):
  saved ordinary/tagged models with 60, 127 and 149 features and their ZBL references.
- Native ASE evaluation: [Ni tagged](publication/cost_comparison/ase_native_ni.py),
  [Li tagged](publication/cost_comparison/ase_native_tagged.py) and
  [Li YACE control](publication/cost_comparison/ase_native_density.py).

The YACE control demonstrates its public native calculator and ASE
`SumCalculator`; the tagged examples use `LinearModel.read`.
The [publication README](publication/cost_comparison/README.md) records model
provenance and the fixed accuracy values.

## Historical studies and source limitations

[Research tools](../tools/publication/README.md) preserve full searches,
screening, normal-equation preparation and deployment checks. Their study
configs remain available for reproducing the original experiments.
[Legacy demonstrations](../tools/legacy_examples/README.md) preserve older
motif averages, role-density fits and low-level carrier probes.

The explicit-Phi example supports a selected cluster descriptor. General
motif-orbit energy sums and their standardized potential fitting remain work.
Shell-density models currently save and reload for ASE; LAMMPS execution of
their shell-filtered radial source remains work.
