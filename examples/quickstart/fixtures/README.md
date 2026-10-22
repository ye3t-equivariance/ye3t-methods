# Quickstart data

## Ni reference structures

`ni_training.extxyz` contains 12 structures from the bundled mlearn Ni fitting
split; `ni_validation.extxyz` contains one structure from its held-out split.
These carry the original energy and force labels. `ni_subset.json` records
the exact indices, source dataset/split hashes and output hashes. The files
are colocated so a copied quickstart directory can run from another working
directory without a sibling repository. Use the full publication examples
for model-quality comparisons.

## Manufactured interface fixtures

The Cu2, Ta3, and H3 structures, target energies, forces, and Ta3/H3 stresses are
deterministic outputs of fixed existing YE3T linear evaluators. They verify API,
fit, persistence, and calculator behavior; they are not fitted to physical
reference calculations. The generating script is in
`ye3t-workflows/split_linear_release/generate_fixtures.py`.

Units follow ASE: Angstrom, eV, eV/Angstrom, and eV/Angstrom^3.

## Site-vector example

`cu_vectors_training.extxyz` contains four displaced 32-atom Cu structures;
`cu_vectors_validation.extxyz` contains one independent displacement.
The site target is the covariant vector
`v_i = sum_j exp(-r_ij / Angstrom) r_ij` over neighbors inside 3.5 Angstrom.
These analytic labels illustrate fitting a complete L=1 output.
`tools/generate_vector_example_data.py` generates the structures and labels;
`cu_vectors.json` records their hashes and provenance.
