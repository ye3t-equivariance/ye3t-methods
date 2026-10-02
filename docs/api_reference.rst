Public linear API
=================

The compact import is ``from ye3t_methods import Basis, LinearModel``.
``ye3t_methods.FeatureLabel`` represents an actual fitted descriptor column.
The preserved ``ye3t_ace`` module path keeps established low-level linear
objects and saved Torch/JSON schema identities.

.. list-table::
   :header-rows: 1

   * - Object or method
     - Purpose
   * - ``Basis(elements=..., source=..., cutoff=..., ...)``
     - Construct density, tagged physical-image, or explicit ``bar_phi`` columns.
   * - ``basis.labels`` / ``basis.describe(index, format="text")``
     - Inspect compiler-accepted columns; ``format="latex"`` is also available.
   * - ``basis.create(atoms)``
     - Evaluate standalone descriptors from a newly constructed basis.
   * - ``LinearModel(basis, reference_energies=None)``
     - Create the fixed-feature scalar readout; Phi rejects reference offsets.
   * - ``model.fit(structures, regularization=..., energy_weight=...,
       force_weight=..., stress_weight=...)``
     - Fit to precomputed ASE energy, force, and supported stress labels.
   * - ``model.write(path)`` / ``LinearModel.read(path)``
     - Save or restore the source-specific artifact.
   * - ``model.ase_calculator(backend=None)``
     - Build an ASE calculator for a fitted or loaded model.
   * - ``model.export_lammps(path)``
     - Export density or tagged formats under their strict contracts.
   * - ``label.as_dict()`` / ``label.latex()``
     - Inspect structured identity or a short mathematical display.

The low-level descriptor-first flow is
``YE3TRepresentation -> YE3TDescriptors -> YE3TModel``. Use it for retained
``A_s`` fitting, lifted Cauchy models, fixed Young descriptor sets, manual ACE
coordinate requests, and specialized runtime controls. It accepts compiler
plans and validated labels rather than locally invented coupling paths.

See :doc:`density`, :doc:`tagged_basis`, :doc:`phi_basis`, and
:doc:`role_density` for source-specific arguments and tested use.
:doc:`evaluators` gives the exact ASE backend choices and native-library setup.
:doc:`basis_inputs` describes compact compiler requests, and
:doc:`chemical_encoding` states which species encodings can be deployed.
