Compact basis inputs and compiler labels
========================================

The compact ``Basis`` constructor takes an ordered ``elements`` list,
physical cutoff, source family, and short radial/angular truncation schedule.
For ordinary density, ``max_rank``, ``nmax``, and ``lmax`` are the inputs;
``nmax`` and ``lmax`` may each be a scalar or a value for every rank.
These values determine candidate source channels. The retained
``YE3TRepresentation -> YE3TDescriptors -> YE3TModel`` route accepts
additional fixed catalogues and source policies.

The paper comparison's
``examples/publication/cost_comparison/config.json`` exposes a compact
schedule for ranks 1, 2, 3, 4, 6, and 8 via ``tensor_orders``,
``nmax_by_tensor_order``, and ``lmax_by_tensor_order``. The companion
``tagged_components.json`` specifies fixed physical channel contents and
block sizes. These are *requests*, not lists of accepted descriptor labels.
The workflow asks ``ye3t.couplings.count`` and ``plan`` which
Young/rotation multiplicity spaces actually exist, then ``compile``
materializes the chosen coefficients. This keeps mathematical provenance in
the separate ``ye3t`` dependency.

For a small direct compiler inspection:

.. code-block:: python

   from ye3t.couplings import count, plan

   report = count(content=(1, 1, 2), input_Ls=(0, 0, 0), target_L=0)
   labels = report.labels_for_target(0)
   if labels:
       report.require_label(labels[0], target_L=0)
       coupling_plan = plan(report)
       print(labels[0], coupling_plan)

``ye3t.BasisLabel``, ``to_compact``, ``to_json``, and ``to_latex``
format *resolved* coordinates and preserve known convention metadata. A
compact text label alone does not supply missing Young sectors, coupling
paths, or coefficients. To select ordinary ACE coordinates manually through
``YE3TDescriptors.ace(manual_labels=...)``, use complete labels accepted
by the compiler; do not form independent Cartesian products of radial,
angular, and Young fields.

The compact ``FeatureLabel`` in this package records one actual application
descriptor column and its source channel identities. Its
``feature_index`` is a column ordinal and may differ from a core
multiplicity-copy coordinate.
