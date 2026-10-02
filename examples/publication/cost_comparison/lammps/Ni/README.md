# Ni linear ACE / tagged YE3T comparison

This directory contains the exact promoted model bytes for the fixed published
`mlearn` split. The complete three-way curve contains 60-, 127-, and
149-descriptor ordinary controls (`ace_60`, `ace_127`, `ace_149`) and tagged
YE3T models (`ye3t_tagged_60`, `ye3t_tagged_127`, `ye3t_tagged_149`) with
15 + 45, 59 + 68, and 81 + 68 ordinary + tagged descriptors respectively.
`ye3t_augmented_196` is the separately labelled 128 + 68 arm. Every model is
a residual to the LAMMPS ZBL reference and must use the included
`hybrid/overlay` settings.

Run from this directory after installing `ye3t-lammps` into LAMMPS:

```bash
lmp -in in.ni_pace_product_127
lmp -in in.ni_ye3t_symmetric_127
lmp -in in.ni_ye3t_mixed_127
lmp -in in.ni_ye3t_tagged_127
mpiexec -n 4 lmp -in in.ni_ye3t_mixed_127
lmp -in in.ni_ye3t_tagged_127.numdiff
lmp -in in.ni_ye3t_tagged_127.nve
```

The same PACE product, YE3T-symmetric, and YE3T-mixed inputs exist at 60 and
149 descriptors, and `in.ni_ye3t_augmented_196` with its `.numdiff`/`.nve`
variants, plus `in.ni_ye3t_mixed_augmented_196`, run the augmented model.
PACE inputs explicitly request the `product` evaluator.
The symmetric input loads its compiler plan with `block_policy auto`; the
mixed input calibrates compiled-direct, generic DAG, symmetric-power, and
block candidates before fixing one route; the tagged input forces the direct
route. Missing or semantically mismatched plans are errors rather than silent
direct fallbacks.

The ordinary model is a standard `.yace`. A tagged model is a composite of
`ordinary_backbone.yace` and `tagged_correction.ye3t.json`; the nontrivial
correction cannot be represented by ordinary PACE alone.
`model_manifest.json` records every promoted model hash. Compact held-out
accuracy, timing, equation-of-state, elastic, dimer, and stability evidence
for the Ni curve is under `evidence/`.
