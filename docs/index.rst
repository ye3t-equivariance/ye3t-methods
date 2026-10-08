YE3T linear methods
===================

``ye3t-methods`` turns ASE ``Atoms`` objects into symmetry-adapted descriptor
rows and fits fixed-feature atomistic models. It supports scalar energies,
forces from analytic position derivatives, and selected tensor-valued per-atom
properties. The configured linear workflow is
``ye3t.YE3TRepresentation`` → ``ye3t_methods.Basis`` →
``ye3t_methods.LinearModel``. The ``Basis`` obtains its coupling labels and
coefficients from ``ye3t``. The descriptor-first
``ye3t_methods.YE3TRepresentation`` → ``YE3TDescriptors`` → ``YE3TModel``
workflow serves specialized and compatible models. Its representation selector
is a separate class and cannot be passed to ``Basis.from_config``.

For the shortest descriptor call, see :doc:`quickstart`. The configured
Python/ASE property path is tested through ``L=3``; native LAMMPS property
capabilities are narrower and documented in :doc:`density` and
:doc:`tagged_basis`.

The quickstarts use manufactured labels to check descriptor evaluation and
fitting. Fitted paper models and their validation inputs are in the
paper-model examples.

.. toctree::
   :maxdepth: 2

   quickstart
   ase_workflows
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
