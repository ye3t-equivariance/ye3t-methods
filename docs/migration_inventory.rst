Documentation migration record
==============================

These pages retain the linear material from the original ``ye3t-ace``
documentation while using imports and examples available in the stable source
archive. The original pages and research examples remain preserved outside
this package; their old commands are not stable-package instructions.

.. list-table::
   :header-rows: 1

   * - Original document
     - Release destination
   * - ``linear_ace.rst``, ``descriptor_sets.rst``
     - :doc:`density`, :doc:`quickstart`, :doc:`labels_and_runtime`
   * - ``a_s_basis.rst``, ``a_s_permutation_recoupling.rst``
     - :doc:`role_density` and the standalone role-density quickstart
   * - ``coupling_provenance.rst``, ``descriptor_runtime.rst``,
       ``cache_performance_guide.rst``
     - :doc:`labels_and_runtime`
   * - ``symmetric_power_scheduler.rst``, ``advanced_schedules.rst``
     - :doc:`labels_and_runtime` for the retained linear fast path
   * - ``stress_workflows.rst``, ``yace_io.rst``
     - :doc:`deployment`
   * - Original quickstart pages and ``api_reference.rst``
     - :doc:`quickstart`, :doc:`api_reference` with installed-package imports

The tagged basis now has its own :doc:`tagged_basis` page. It draws on the
retained tagged Ta example's ``THEORY.md`` and the actual compact tagged API;
the old documentation index did not contain an equivalent tagged-basis guide.
The six-element linear paper fitting and LAMMPS workflow is retained under
``examples/publication/cost_comparison`` and described in
:doc:`paper_models`. :doc:`basis_inputs` covers the compact compiler
requests, and :doc:`chemical_encoding` covers one-hot and fixed in-memory
species mixing.

The release validation record identifies which fixed-feature behavior was
exercised. Other research examples remain in the original archive.
