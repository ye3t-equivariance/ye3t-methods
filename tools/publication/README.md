# Publication research tools

These restartable CLI pipelines reproduce the historical catalogue searches,
fitting sweeps and deployment checks. They retain their study-specific config
formats and scientific settings. `migration.json` records their original paths
and file hashes. Saved paper model bytes remain in `examples/publication`.

For direct user workflows, use the visible seven-section configs in
`examples/quickstart` and `examples/publication/cost_comparison`:

- `tagged_fit.py`, `density_fit.py`, and `shell_density_fit.py` construct a
  representation, basis and linear model from labeled ASE structures.
- `refit_paper_ni.py` refits the fixed published Ni basis on the full split.
- `reproduce_ni_rmse.py` evaluates the saved paper models.

## Historical pipelines

```bash
python tools/publication/cost_comparison/run.py --help
python tools/publication/cost_comparison/run.py --stage preflight --systems Ni
python tools/publication/ta_tagged_cauchy_image_linear/train_export.py --help
```

The cost-comparison pipeline includes catalogue selection, radial screening,
normal-equation construction, model selection, performance and property checks.
Its `workflow/` modules are implementation tools. The Ta pipeline includes its
original zero/one/two-tag controls, residual ZBL targets and deployment checks.
Changing those scientific settings creates a separate study.

Raw outputs go to `ye3t-workflows`; the CLI flags and config files expose their
locations. Native deployment checks require a compatible LAMMPS executable.
These tools are included in source distributions for reproducibility.
