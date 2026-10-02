# Tagged Ta diagnostic

This directory preserves the configuration, fitting driver, and mathematical
record for an earlier bounded Ta study. Its original data and split manifests
are held in the research archive, so the driver is not a standalone
installed-package example.

From a `ye3t-methods` source checkout, the bounded preflight runs without the
archived Ta dataset:

```bash
python examples/publication/ta_tagged_cauchy_image_linear/train_export.py \
  --config examples/publication/ta_tagged_cauchy_image_linear/config_quick.json \
  --preflight-only
```

The result goes to a unique directory under `~/ye3t-workflows` by default.
`--output` and `--cache-root` select other locations. Use `config.json` for
the original eight-radial-degree study; `config_quick.json` uses two degrees
while retaining the same rank, angular sector, and tagged arm types.

The two configs expose the main choices directly:

| Edit | Purpose |
| --- | --- |
| `basis.species`, `basis.radial.cutoff_A`, `basis.radial.degrees`, `basis.angular.l` | Atomic species, cutoff, radial channels, and angular input. |
| `basis.descriptor_catalogue.tensor_order` | Tensor order (the retained study supports rank four). |
| `basis.descriptor_catalogue.arms[*].selected_raw_tag_counts` | Ordinary, one-tag control, nontrivial two-tag, and combined arms. |
| `representation.target` | Globally invariant scalar output; selected tagged source sectors can still be nontrivial. |
| `runtime.backend`, `runtime.device`, `runtime.cache`, `runtime.output_directory` | Fitting backend and output/cache locations. |
| `model.ridge_alphas`, `targets.energy_weight`, `targets.force_weight` | Linear fit and target weights. |
| `targets.dataset`, `validation.split.manifests_by_size` | Hash-bound archived Ta dataset and train/validation partitions. |

The driver supports a PyTorch CPU fit. Full fitting also evaluates the
configured LAMMPS ZBL reference and requires the archived dataset and split
manifests; `run_end_to_end.sh --help` lists the LAMMPS validation inputs.

For runnable linear workflows in this source archive, use:

- `../../quickstart/tagged_fit.py` for fitting a tagged model and evaluating
  it with ASE;
- `../cost_comparison/` for the six-element paper models, PACE and YE3T
  LAMMPS inputs, and exact Li native ASE example;
- `../../quickstart/parent_coupling.py` for a bounded rank-eight parent plan.

The public tagged-basis guide at `../../../docs/tagged_basis.rst` records the
exact N=4, `l=1`, selected-tag scope and the distinction between source
opportunities and accepted physical-image coordinates. The configs and scripts
remain as historical inputs for the archived Ta data;
they are not release qualification for a new Ta potential. The original
research archive retains the detailed commands and validation record.
