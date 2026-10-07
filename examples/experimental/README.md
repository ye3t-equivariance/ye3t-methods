# Experimental source examples

These retained examples exercise source-specific methods that are not yet part
of the standard seven-section `YE3TRepresentation -> Basis -> LinearModel`
construction route. For a new scalar density or tagged model, start with the
configured scripts in `examples/quickstart`.

- `phi_fit.py` fits the existing scalar `bar_phi` pair/star motif to the bundled
  H3 labels and saves a Torch model.
- `phi_depth2.py` evaluates the existing explicit depth-two motif.
- `role_density_fit.py` fits the lifted-density reference route to manufactured
  Ta labels. Lifted density mainly changes the radial basis; this example is
  not evidence for a general nontrivial permutation-sector physical image.

Run each script from the source checkout after installing `ye3t-methods`.
Their existing source-specific configs and saved artifacts remain useful for
validation, but they are not the recommended interface for new projects.
