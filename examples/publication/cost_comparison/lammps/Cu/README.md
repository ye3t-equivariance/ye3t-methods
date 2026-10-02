# Cu linear ACE / tagged YE3T comparison

This directory contains the exact promoted model bytes for the fixed published
`mlearn` split. `ace_127` is the matched ordinary ACE control and
`ye3t_tagged_127` is the 59 ordinary + 68 tagged YE3T comparison. Both models
are residuals to the LAMMPS ZBL reference and must use the included
`hybrid/overlay` settings.

Run from this directory after installing `ye3t-lammps` into LAMMPS:

```bash
lmp -in in.cu_pace_product_127
lmp -in in.cu_ye3t_symmetric_127
lmp -in in.cu_ye3t_mixed_127
lmp -in in.cu_ye3t_tagged_127
mpiexec -n 4 lmp -in in.cu_ye3t_mixed_127
lmp -in in.cu_ye3t_tagged_127.numdiff
lmp -in in.cu_ye3t_tagged_127.nve
```

The PACE input explicitly requests the `product` evaluator. The symmetric
input loads the byte-identical `.yace` with its compiler plan and
`block_policy auto`; the mixed input calibrates compiled-direct, generic DAG,
symmetric-power, and block candidates before fixing one route; the tagged
input forces the direct route. Missing or semantically mismatched plans are
errors rather than silent direct fallbacks. The `.numdiff` and `.nve` inputs
also exist for the PACE control.

The ordinary model is a standard `.yace`. The tagged model is a composite of
`ordinary_backbone.yace` and `tagged_correction.ye3t.json`; the nontrivial
correction cannot be represented by ordinary PACE alone.
`model_manifest.json` records every promoted model hash, and each
`models/<model>/model_manifest.json` records the fit configuration, held-out
metrics, and ZBL reference. The compact six-element evidence is in
[canonical ye3t-lammps results](https://github.com/ye3t-equivariance/ye3t-lammps/tree/main/docs/results/cost_comparison_three_way_auto_v3_20260921).
