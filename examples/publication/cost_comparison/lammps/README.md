# Linear ACE / tagged YE3T comparison models

This directory covers Li, Mo, Cu, Ni, Si, and Ge on the fixed published
`mlearn` train/test split. Every element directory contains the exact
deployable model files, runnable LAMMPS inputs, and reference logs for the
matched 127-descriptor comparison:

| model | file | pair style |
|---|---|---|
| `ace_127` | `models/ace_127/potential.yace` | ordinary linear ACE control (127 descriptors) |
| `ye3t_tagged_127` | `models/ye3t_tagged_127/model.ye3t.json` | 59 ordinary + 68 tagged YE3T descriptors |

Ni additionally keeps the 60- and 149-descriptor tiers of both families and
the separately labelled `ye3t_augmented_196` model (128 ordinary + 68 tagged
descriptors), so the complete accuracy/runtime curve of one system can be
reproduced. Every model is a residual to the LAMMPS ZBL reference recorded in
its `model_manifest.json` and must be run with the included
`hybrid/overlay ... zbl` settings; adding ZBL to a model fitted on total
energies would double count the repulsion.

Each tagged input loads a single `model.ye3t.json` composite through
`pair_style ye3t` plus the ZBL overlay. It does not call `pair_style pace` or
load a PACE coefficient file; PairPACE appears only in the explicit ACE/PACE
control inputs. The ordinary model is a standard `.yace`. A tagged model is a
composite of `ordinary_backbone.yace` and `tagged_correction.ye3t.json` bound
by SHA-256; the nontrivial correction cannot be represented by ordinary PACE
alone.

Run from the element directory so the relative `models/` paths resolve:

```bash
cd Li
lmp -in in.li_pace_product_127
lmp -in in.li_ye3t_symmetric_127
lmp -in in.li_ye3t_mixed_127
lmp -in in.li_ye3t_tagged_127
```

The first explicitly requests PACE `product`. The second loads the
byte-identical ordinary `.yace`, its compiler-produced plan, and
`block_policy auto`. The third loads the self-contained ordinary-plus-tagged
YE3T model with `block_policy auto`, which calibrates compiled-direct, generic
DAG, symmetric-power, and block candidates at `pair_coeff` time before fixing
one route; the fourth loads the same model with the forced `direct` route.
AUTO selection is catalogue- and hardware-specific: the fitted catalogues here
select compiled-direct on the reference CPU, while repetition-rich
high-order catalogues select symmetric-power or block routes.

The `.numdiff` inputs run LAMMPS `fix numdiff` and `fix numdiff/virial`
(package EXTRA-FIX) and print `YE3T_NUMDIFF_FORCE_MAX_ABS` and
`YE3T_NUMDIFF_VIRIAL_L2`. The `.nve` inputs run 2,000 NVT preparation steps
followed by 10,000 NVE steps and write a trajectory. Each input has a
`log.<date>.<name>.g++.1` and `.g++.4` reference log beside it.

`model_manifest.json` in each element directory lists every promoted model
file with its size and SHA-256, and each `models/<model>/model_manifest.json`
records the fit configuration, held-out metrics, and ZBL reference of that
model. From the extracted ye3t-methods source archive, verify every copied
model and data file with:

```bash
python examples/publication/cost_comparison/verify_models.py
```

The compact accuracy, timing, parity, and qualification evidence and figures
remain in the canonical
[ye3t-lammps result directory](https://github.com/ye3t-equivariance/ye3t-lammps/tree/main/docs/results/cost_comparison_three_way_auto_v3_20260921).
Ni also keeps compact per-model evidence tables under `Ni/evidence/`. The
training workflow, editable catalogue/radial/fit configuration, and the
`mlearn` split references are alongside this copy in the ye3t-methods source
archive. The model files themselves are byte-identical to the promoted
ye3t-lammps bundle.
