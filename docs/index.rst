YE3T linear methods
===================

``ye3t-methods`` materializes fixed-feature atomistic models from the separate
``ye3t`` representation compiler. It fits scalar energies and their analytic
position derivatives, saves models, and exposes ASE calculators. The compact
entry points are ``Basis`` and ``LinearModel``; the retained descriptor-first
entry points are ``YE3TRepresentation``, ``YE3TDescriptors``, and ``YE3TModel``.

The examples in this guide use deterministic manufactured labels to check the
software path. They are not validated physical potentials. The precise tested
families, runtimes, and deployment limits are in the source archive's
``RELEASE_VALIDATION.md``.

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
   migration_inventory
