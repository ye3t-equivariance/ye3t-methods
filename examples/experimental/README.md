# Experimental source examples

These scripts exercise source-specific APIs outside the seven-section
`YE3TRepresentation -> Basis -> LinearModel` workflow. For a scalar density
or tagged model, start with the configured scripts in `examples/quickstart`.

- `phi_fit.py` fits a scalar `bar_phi` pair/star motif to the bundled
  H3 labels and saves a Torch model.
- `phi_depth2.py` evaluates an explicit depth-two motif.
- `role_density_fit.py` fits a lifted-density model to manufactured
  Ta labels. Lifted density mainly changes the radial basis; this example is
  not evidence for a general nontrivial permutation-sector physical image.

Run each script from the source checkout after installing `ye3t-methods`.
These scripts use source-specific configs. Use the configured quickstarts for
new projects.
