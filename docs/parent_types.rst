Nontrivial parent types
=======================

The final parent type of a rank-``N`` descriptor is a Young partition of
``N`` tensor factors together with a rotation irrep ``L``. These are
different from the tag and role partitions of a tagged source and from the
filter-role group carried by a role density. A scalar energy feature has a
trivial final parent and ``L=0``; a nontrivial parent plan supplies covariant
coordinates for a separate observable or a later invariant coupling.

The installed-package example
``examples/quickstart/parent_coupling.py`` asks the core compiler for a
validated rank-eight parent ``lambda=(4,4)``, ``L=2`` coupling plan. It prints
only the plan's rank, parent type, LR multiplicity, and tableau count. Change
``parent_partition`` and ``target_L`` to request another type; the compiler
checks whether that type is allowed for this two-branch construction.

.. literalinclude:: ../examples/quickstart/parent_coupling.py
   :language: python

``examples/quickstart/parent_coefficient.py`` applies a compiled rank-three
``lambda=(2,1)``, ``L=1`` parent coupler to caller-supplied tensor values.
This is a coefficient descriptor view: the caller supplies its input carrier,
and the example does not compute that carrier from atom positions.

.. literalinclude:: ../examples/quickstart/parent_coefficient.py
   :language: python

The rank-eight example produces a coupling plan. Materializing an arbitrary
nontrivial parent type from atom positions requires a suitable physical factor
source. The role-density evaluator exposes some nontrivial *filter-role*
Specht sectors with angular components; those sectors differ from the global
rank-``N`` parent Young irrep. The source example marks the missing geometry
path as a TODO. Check a nontrivial parent request against a sector-specific
compiled and validated plan before evaluating it.
