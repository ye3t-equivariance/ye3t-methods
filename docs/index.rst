YE3T linear methods
===================

``ye3t-methods`` materializes fixed-feature atomistic models from the separate
``ye3t`` representation compiler. It turns ASE ``Atoms`` into per-atom NumPy
descriptor rows, fits scalar energies and selected tensor-valued per-atom
properties, saves models, and exposes ASE calculators. Scalar fits provide
analytic position derivatives. The configured linear flow is
``ye3t.YE3TRepresentation`` →
``ye3t_methods.Basis`` → ``ye3t_methods.LinearModel``. The basis owns the
compiler-derived descriptor columns. The older descriptor-first
``ye3t_methods.YE3TRepresentation`` → ``YE3TDescriptors`` → ``YE3TModel``
flow remains available for specialized and compatible models; its
representation selector is a distinct class and cannot be passed to
``Basis.from_config``.

For the shortest descriptor call, see :doc:`quickstart`. The configured
Python/ASE property path is tested through ``L=3``; native LAMMPS property
capabilities are narrower and documented in :doc:`density` and
:doc:`tagged_basis`.

The quickstart examples use deterministic manufactured labels to check the
software path. They are not validated physical potentials. The paper-model
examples contain the promoted model artifacts and their validation inputs.

.. toctree::
   :maxdepth: 2

   quickstart
   evaluators
   scikit_linear_fit
   paper_models
   basis_inputs
   parent_types
   chemical_encoding
   density
   tagged_basis
   phi_basis
   role_density
   labels_and_runtime
   deployment
   api_reference
