# Manufactured quickstart data

The Cu2, Ta3, and H3 structures, target energies, forces, and Ta3/H3 stresses are
deterministic outputs of fixed existing YE3T linear evaluators. They verify API,
fit, persistence, and calculator behavior; they are not fitted to physical
reference calculations. The generating script is in
`ye3t-workflows/split_linear_release/generate_fixtures.py`.

Units follow ASE: Angstrom, eV, eV/Angstrom, and eV/Angstrom^3.
