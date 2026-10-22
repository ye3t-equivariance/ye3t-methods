ASE quickstarts
===============

The examples use one workflow:
``YE3TRepresentation`` → ``Basis`` → ``LinearModel``.
Descriptor calculations stop at ``basis.create(atoms)``, which returns a NumPy
array. Every script contains its editable config with the same seven sections:
``metadata``, ``basis``, ``representation``, ``runtime``, ``model``, ``targets``
and ``validation``.

Install ``ye3t`` and ``ye3t-methods`` in the same environment. The source archive
contains the example scripts, small labeled datasets and paper model files.
From its extracted directory:

.. code-block:: console

   python -m pip install --no-build-isolation '.[examples,fit]'
   python examples/quickstart/ase_descriptors.py
   python examples/quickstart/density_fit.py

The ``fit`` extra supplies scikit-learn. Native tagged evaluation requires the
C++ extensions built by the normal package installation; see :doc:`evaluators`.
Input paths are visible in each config. Colocated fixtures work when the
quickstart folder is copied elsewhere. Saved-paper examples also need their
referenced ``publication/cost_comparison`` model directory. Outputs default to
``~/ye3t-workflows``. Set ``YE3T_WORKFLOW_ROOT`` or edit the config output paths
to choose another location.

The tagged export example reads the artifact produced by the tagged fit:

.. code-block:: console

   python examples/quickstart/tagged_fit.py
   python examples/quickstart/saved_ase_export.py

Use the same ``YE3T_WORKFLOW_ROOT`` for both commands. Running the generated
LAMMPS input requires ML-YE3T. Export itself and ASE evaluation use the
installed methods package. See :doc:`scikit_linear_fit` for reusable fitting
rows and the cache location.

Choose an example
-----------------

.. list-table:: Public workflows
   :header-rows: 1
   :widths: 35 65

   * - Script under ``examples/quickstart``
     - Calculation
   * - ``ase_descriptors.py``
     - 100 scalar density features, ranks 1–8, four-atom Ni cell.
   * - ``tagged_descriptors.py``
     - Tagged scalar features with mixed Young/angular intermediates, ranks 1–8.
   * - ``paper_ni_60.py``
     - Saved 60-feature density and tagged Ni fits with native ASE evaluation;
       exact tagged-model descriptor columns on two structure sizes.
   * - ``ase_octupole_descriptors.py``
     - 51 odd-parity ``L=3`` features, ranks 1–4, all seven magnetic components.
   * - ``chemical_encoding.py``
     - A fixed two-channel species embedding, ranks 1–4.
   * - ``chemical_channel_selection.py``
     - Select neighbor species through a fixed chemical embedding, ranks 1–4.
   * - ``density_fit.py``, ``tagged_fit.py``
     - Energy/force fits on 12 real Ni structures with the descriptor catalogues above.
   * - ``combined_density_tagged_fit.py``
     - One linear fit using both physical sources and one saved model.
   * - ``shell_density_fit.py``
     - A scalar shell-density fit with joint rotation/permutation intermediates.
   * - ``sklearn_fit.py``
     - LASSO and ARD fitting, including ARD prediction uncertainty.
   * - ``inspect_features.py``
     - Inspect a descriptor label and its mathematical description.
   * - ``saved_ase_export.py``
     - Export the tagged fit and write a LAMMPS AUTO input and atomic data.
   * - ``per_atom_vector_to_lammps.py``
     - Fit an analytic vector-property fixture and write a LAMMPS property input.
   * - ``paper_ni_portable_ase.py``
     - Evaluate the saved 127-feature paper model in ASE.
   * - ``paper_ni_nve.py``
     - Run that model for 100 NVE steps and save energy/temperature samples.
   * - ``coupled_factors.py``
     - Apply a core Cauchy coupling plan to supplied factors, retaining tableau and magnetic components.

The small Ni training subset carries the original mlearn energies and forces;
its indices and hashes are in ``fixtures/ni_subset.json``. For published accuracy
comparisons use :doc:`paper_models`, which evaluates the full saved split and
provides the exact selected-basis refit. The vector fixture is an analytic
covariant function of its Cu neighbor vectors, with a recorded generator.

Scalar descriptors
------------------

The parent ``(N), L=0, even`` specifies a scalar output. The input angular caps
allow nonzero one-factor angular degrees. The compiler determines the
independent features within the selected channel contents.

.. literalinclude:: ../examples/quickstart/ase_descriptors.py
   :language: python

Tagged descriptors
------------------

The tagged example exposes radial counts, angular patterns and tag counts for
every rank. Its scalar output includes nontrivial internal Young and angular
couplings. Initial coefficient compilation costs more than density evaluation;
subsequent runs reuse the compiler cache. The execution tests check that the
native evaluator is actually selected.

.. literalinclude:: ../examples/quickstart/tagged_descriptors.py
   :language: python

Linear fitting
--------------

Read labeled ASE structures, construct the representation and basis, fit the
model and attach its calculator to another ``Atoms`` object. All training data
paths and target keys are editable.

.. literalinclude:: ../examples/quickstart/density_fit.py
   :language: python

The tagged and combined fits follow the same calls. ``model.score(structures,
config=config)`` returns energy RMSE per atom and force RMSE over Cartesian
components; configured stress targets also produce stress RMSE. Reference
calculators and labels on the supplied structures are preserved.

.. literalinclude:: ../examples/quickstart/tagged_fit.py
   :language: python

Shell-density fitting
---------------------

The shell source retains its role action during coupling and selects invariant
scalar energy coordinates. See :doc:`shell_linear` for source normalization,
derivatives and saved-model details.

.. literalinclude:: ../examples/quickstart/shell_density_fit.py
   :language: python

ASE molecular dynamics
----------------------

.. literalinclude:: ../examples/quickstart/paper_ni_nve.py
   :language: python

Deployment
----------

Run ``tagged_fit.py`` followed by ``saved_ase_export.py`` from the same working
directory, or change the saved-model path in the export config. The export
contains the compiler plans required by LAMMPS AUTO. Run ``lmp -in in.ye3t``
from the generated directory with ML-YE3T installed.

.. literalinclude:: ../examples/quickstart/saved_ase_export.py
   :language: python

The vector-property example uses ``compute ye3t/property/atom``. Its saved
predictions and LAMMPS dump use real-tesseral component order.

.. literalinclude:: ../examples/quickstart/per_atom_vector_to_lammps.py
   :language: python

Advanced sources and historical tools
--------------------------------------

``examples/advanced/explicit_phi_cluster.py`` evaluates one selected rank-eight
cluster coupling. Its complete output has shape ``(1,14,5)``. See
:doc:`phi_basis` for its supported geometry and the remaining orbit-sum fitting
work. Historical motif-average, role-density and coupling demonstrations live
under ``tools/legacy_examples`` with migration records and regression pointers.
Full publication search pipelines live under ``tools/publication``. Their
original scientific settings remain available for reproducing those studies.
