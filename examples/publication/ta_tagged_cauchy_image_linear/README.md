# Tagged Ta diagnostic

This directory contains the configuration, fitting driver, and mathematical
record for a rank-four Ta study. The full fit requires data and split manifests
from the research archive; the installed package alone does not include them.

From a `ye3t-methods` source checkout, the bounded preflight runs without the
archived Ta dataset:

```bash
python examples/publication/ta_tagged_cauchy_image_linear/train_export.py \
  --config examples/publication/ta_tagged_cauchy_image_linear/config_quick.json \
  --preflight-only
```

The result goes to a unique directory under `~/ye3t-workflows` by default.
`--output` and `--cache-root` select other locations. `config.json` uses eight
radial degrees; `config_quick.json` uses two with the same rank, angular sector,
and tagged arm types.

The two configs expose the main choices directly:

| Edit | Purpose |
| --- | --- |
| `basis.species`, `basis.radial.cutoff_A`, `basis.radial.degrees`, `basis.angular.l` | Atomic species, cutoff, radial channels, and angular input. |
| `basis.descriptor_catalogue.tensor_order` | Tensor order (this driver supports rank four). |
| `basis.descriptor_catalogue.arms[*].selected_raw_tag_counts` | Ordinary, one-tag control, nontrivial two-tag, and combined arms. |
| `representation.target` | Globally invariant scalar output; selected tagged source sectors can still be nontrivial. |
| `runtime.backend`, `runtime.device`, `runtime.cache`, `runtime.output_directory` | Fitting backend and output/cache locations. |
| `model.ridge_alphas`, `targets.energy_weight`, `targets.force_weight` | Linear fit and target weights. |
| `targets.dataset`, `validation.split.manifests_by_size` | Hash-bound archived Ta dataset and train/validation partitions. |

The driver supports a PyTorch CPU fit. Full fitting also evaluates the
configured LAMMPS ZBL reference and requires the archived dataset and split
manifests; `run_end_to_end.sh --help` lists the LAMMPS validation inputs.

For complete linear workflows, use:

- `../../quickstart/tagged_fit.py` for fitting a tagged model and evaluating
  it with ASE;
- `../cost_comparison/` for the six-element paper models, PACE and YE3T
  LAMMPS inputs, and exact Li native ASE example;
- `../../quickstart/parent_coupling.py` for a bounded rank-eight parent plan.

The public tagged-basis guide at `../../../docs/tagged_basis.rst` records the
exact N=4, `l=1`, selected-tag scope and the distinction between source
opportunities and accepted physical-image coordinates. These configs and
scripts reproduce the archived Ta setup; they do not qualify a new Ta
potential. The research archive contains the full dataset, commands, and
validation record.
