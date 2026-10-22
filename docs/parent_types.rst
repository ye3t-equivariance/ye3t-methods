Nontrivial parent types
=======================

A rank-``N`` formal tensor-product parent is a Young partition of ``N`` factors,
an angular momentum ``L`` and a parity. A selected multiplicity coordinate
contains every tableau component and every magnetic component. The physical
source determines which factor or role action these coordinates describe.

``coupled_factors.py`` supplies factor multiplets, compiles their Cauchy
coupling plan in ``ye3t`` and passes that plan into ``Basis.from_config``.
Its nontrivial ``(2,1), L=1`` output has axes ``(batch, a, tableau, M)``.
Each ``a`` identifies an independent coupling in the selected source space.
Edit the visible factor types, parent and values together for another input.

.. literalinclude:: ../examples/quickstart/coupled_factors.py
   :language: python

The example demonstrates the core-to-methods compiled-plan interface. General
multiplicity counting, symmetric/antisymmetric factors and larger-rank coupling
examples are supplied by ``ye3t/examples``. For an atomistic selected cluster,
see :doc:`phi_basis`; for shell-role scalar features, see :doc:`shell_linear`.

Historical parent-coefficient and rank-eight plan probes are preserved under
``tools/legacy_examples``. Their low-level construction is useful for
regressions; the example above is the public configured workflow.
