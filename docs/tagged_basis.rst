Tagged physical-image basis
===========================

The tagged route retains explicitly selected neighbor roles before the
compiler forms globally invariant scalar coordinates. It is available through
``Basis(source="tagged_cauchy_image")`` and the retained
``YE3TRepresentation.tagged_cauchy_image()`` descriptor-first route. This is
the ``tag`` family in the linear release.

What the basis distinguishes
----------------------------

``N`` is tensor-product order and ``s`` is the number of ordered tags. A tag
permutation, a permutation of formal tensor positions, and a relabeling of
physical neighbor atoms are different operations. Intermediate tag and role
Young sectors may be nontrivial, while the final energy coordinate is scalar
and globally invariant. Coupling paths and coefficients come from ``ye3t``;
the application does not enumerate them independently.

For the certified homogeneous two-tag witness, ``N=4``, input ``l=1``, and
``s=2``, the compiler includes a nontrivial tag sector
``tag_kappa=(1,1)`` with a ``role_kappa=(2,1,1)`` companion. The compiler
maps selected raw opportunities into the exact physical image and removes
source identities before it exposes fitted columns. A public column can
therefore combine multiple raw opportunities. Its printed tag summary is not
a claim that it is one pure tag partition. The tested example shows how to
inspect the full contributing records.

For the same-source inclusive-density construction in the retained Ta
diagnostic, the selected one-tag row aliases the ordinary zero-tag row. This
is a bounded source identity, not a general rule for all one-tag sources.
Other requested orders, contents, angular channels, and tag counts remain
subject to compiler validation; the N=4 witness does not certify them.

Building, fitting, and inspecting
---------------------------------

The full runnable script is on :doc:`quickstart`. Its essential basis request
is:

.. code-block:: python

   from ye3t_methods import Basis

   basis = Basis(
       elements=["Ta"], source="tagged_cauchy_image", cutoff=4.8,
       tensor_order=4, tag_counts=(0, 2), radial_degrees=(0,),
       angular_degree=1, backend="reference",
   )
   print(basis.labels[0].as_dict())

``tag_counts`` selects raw tag-count opportunities before the exact image is
formed. ``radial_degrees`` selects source degrees, ``angular_degree`` the
one-neighbor angular degree, and ``tensor_order`` the fixed tensor order.
``max_rank`` belongs to ordinary density and is rejected here. Inspect the
actual column count and each label's ``compiler_coordinate_provenance`` and
``compiler_raw_opportunities`` rather than inferring columns from the request.
The ``tag_kappa`` and ``role_kappa`` fields identify the compiler carriers on
which those partitions act; they are not interchangeable with a parent
``lambda`` or with the public column ordinal.

The compact linear model fits precomputed energies and forces and can include
ASE Voigt stress rows with ``stress_weight``. The fixture supplies these labels.
``LinearModel.write`` produces a versioned, hash-bound ``.ye3t.json`` model;
``LinearModel.read`` restores it. An ASE calculator can evaluate energy,
forces, and stress. ``export_lammps`` writes the tagged deployment artifact;
the separate LAMMPS consumer must support its schema. The runnable saved-model
path is ``examples/quickstart/saved_ase_export.py``.
Choose ``reference``, ``native_polynomial``, or ``native_cpu`` with
``model.ase_calculator(backend=...)``; see :doc:`evaluators` for the C++ library
argument, optional neighbor-list speedup, and the paper composite loader.

The tagged source uses a cutoff Jacobi radial/angular realization. Its exact
source algebra and floating evaluator are recorded in compiler artifacts; a
compiler certificate does not establish accuracy for a physical material.
The source-domain and deployment limits, including overlap handling and native
CPU/Kokkos qualification, are recorded in ``RELEASE_VALIDATION.md`` and the
retained ``examples/publication/ta_tagged_cauchy_image_linear/THEORY.md``.
