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

Ordered tagged carriers
-----------------------

``examples/tagged_carriers_ase.py`` evaluates raw zero-, one-, and two-tag
carriers on a Ni ASE cell. Its config exposes species, cell size, cutoff,
rank and radial/angular caps, tag counts, sector policy, and evaluator backend.
The result has separate center, edge, and ordered edge-pair value blocks.
These carrier values are not fitted scalar energy coordinates.

``nmax_per_rank`` and ``lmax_per_rank`` are exact maps covering every requested
rank. ``cutoff_A`` bounds the neighbor list, while ``pair_cutoffs_A`` sets the
radial cutoff for each ordered species pair and cannot exceed ``cutoff_A``.
The certified source is the fixed origin-regular shifted-Jacobi family. It
has no ``radial_lambda`` or ordinary-density ``radial_decay`` parameter;
unsupported settings are rejected rather than ignored.

``tag_character=-1`` on a two-tag label denotes odd exchange of the two
ordered tags. ``target_L`` identifies that carrier's rotation irrep. Neither
is the global parent Young partition of an atomistic descriptor. The example
uses ``sector_policy="tagged_mixed"`` to retain both tag-exchange types;
``backend="reference"`` selects the carrier reference evaluator. Standalone
carriers currently accept only that evaluator. The scalar tagged ``Basis``
below uses an ``auto`` polynomial evaluator, and saved linear models can use
the native C++ ASE calculator; those backend options do not apply to the raw
carrier example.

Building, fitting, and inspecting
---------------------------------

The runnable ``examples/evaluate_ni_descriptors.py`` script shows a Ni fcc
cell, a displaced copy, one visible config, descriptor row slices, and a check
for a nontrivial source-block Young partition. Its essential basis request is:

.. code-block:: python

   from ye3t_methods import Basis

   basis = Basis(
       elements=["Ni"], source="tagged_cauchy_image", cutoff=4.8,
       pair_cutoffs_A={"Ni-Ni": 4.8},
       rank=4, tag_counts=(0, 2),
       nmax_per_rank={4: 1}, lmax_per_rank={4: 1},
       source_block_partitions_by_rank={4: ((4,),)},
       angular_patterns_by_rank={4: ((1, 1, 1, 1),)},
       angular_basis_backend="exact_weight_space_v1",
   )
   print(basis.labels[0].as_dict())
   print(basis.resolved["polynomial_backend"])

The descriptor-first interface accepts the same general catalogue:

.. code-block:: python

   from ye3t_ace import YE3TDescriptors

   config = {
       "metadata": {"name": "ni_tagged_catalogue"},
       "basis": {
           "type": "tagged_cauchy_image", "species": ["Ni"], "cutoff_A": 4.8,
           "pair_cutoffs_A": {"Ni-Ni": 4.8},
           "catalogue": {
               "nmax_per_rank": {4: 1}, "lmax_per_rank": {4: 1},
               "source_block_partitions_by_rank": {4: ((4,),)},
               "angular_patterns_by_rank": {4: ((1, 1, 1, 1),)},
               "angular_basis_backend": "exact_weight_space_v1",
               "tag_counts_by_rank": {4: (0, 2)},
           },
       },
       "representation": {
           "carrier": "A_s", "target": {"permutation": "trivial", "L": 0},
           "mode": "tagged_cauchy_image",
       },
       "runtime": {"backend": "auto"},
       "model": {}, "targets": {}, "validation": {},
   }
   descriptors = YE3TDescriptors.ye3t_basis(config)
   print(len(descriptors.feature_labels))

``tag_counts`` in ``Basis`` and ``tag_counts_by_rank`` in the catalogue select
raw tag-count opportunities before the exact image is formed. ``rank`` fixes
the tensor order in ``Basis``; the catalogue rank comes from its per-rank keys.
``nmax_per_rank`` and
``lmax_per_rank`` are explicit maps from rank to radial and angular caps.
``angular_patterns_by_rank`` restricts the rank-four source factors to four
``l=1`` channels; the angular cap alone would also permit ``l=0``.
``source_block_partitions_by_rank`` requests one four-factor source block.
This general catalogue request has a different column inventory from the older
bounded rank-four witness stored in paper artifacts. ``cutoff`` is the global
neighbor cutoff in angstroms. Optional ``pair_cutoffs_A`` sets radial cutoffs
for a complete ordered species-pair map; each value must be at most ``cutoff``.
The certified shifted-Jacobi radial source has no
adjustable radial lambda; requesting one would require a different compiler
source. The coefficient catalogue is compiled exactly and cached. General
tagged scalar catalogues use YE3T's exact requested-weight angular compiler.
Set ``angular_basis_backend="legacy_exact"`` in ``Basis`` or its catalogue
to use the full-sector exact oracle. Saved models retain their compiled
coefficient convention.
It checks that final real-tesseral scalar coefficients have no imaginary part.
Imaginary entries in the intermediate complex-to-real basis matrix are
expected and do not imply complex descriptor values. The default tagged
polynomial evaluator selects its native path when available;
``basis.resolved["polynomial_backend"]`` reports the selected evaluator.
Its backend setting is separate from coefficient compilation. The generic
``numeric_cached`` subduction and fast Clebsch--Gordan route is not yet wired
to this tagged-image request. Cold rank-four Ni compilation can take tens of
seconds; a repeated request reloads the persistent cache more quickly.
The example's ``expected_source_block_young`` checks the compiled general
catalogue; it is not the tag or role Young partition of the older bounded
construction and does not select a different global parent. The tagged
constructor currently fixes
that parent to the symmetric partition ``(N)`` with ``L=0`` and even parity.
An explicit ``angular_patterns_by_rank`` entry with odd total angular degree
is rejected for this scalar catalogue.
``max_rank`` belongs to ordinary density and is rejected here. Inspect the
actual column count and each label's ``compiler_coordinate_provenance`` and
``compiler_raw_opportunities`` rather than inferring columns from the request.
The ``tag_kappa`` and ``role_kappa`` fields identify the compiler carriers on
which those partitions act; they are not interchangeable with a parent
``lambda`` or with the public column ordinal.

The compact linear model fits precomputed energies and forces and can include
ASE Voigt stress rows with ``stress_weight``. The fixture supplies these labels.
``examples/quickstart/tagged_fit.py`` uses the general catalogue above and
reports a source-block Young partition; it does not assert the bounded paper
model's tag-sign and role Young witness.
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
Exact overlap is outside the certified source domain. The separate
``ye3t-lammps`` consumer defines its own CPU and Kokkos deployment scope.

Certified rank-four construction
--------------------------------

For fixed tensor order and source content, the repeated source blocks follow
the Cauchy decomposition

.. math::

   \operatorname{Sym}^{k}(W \otimes V_l)
   = \bigoplus_{\kappa \vdash k} S_\kappa(W) \otimes S_\kappa(V_l).

The certified two-tag placement carrier has ordered basis vectors
``e_(i,j)`` for ``i != j`` and dimension ``N(N-1)``. Formal tensor-position
permutations act on the left, while swapping the two tag labels acts on the
right. The sign tag type ``(1,1)`` survives final scalar coupling only with a
matching physical source carrier. In the retained witness, ``N=4``, ``s=2``,
``placement_parent_lambda=(3,1)``, ``role_kappa=(2,1,1)``, and
``angular_kappa=(2,2)``. Each input has ``l=1`` and the final ``L=0``.
Physical-neighbor relabeling is a separate action. These labels and their
intertwiner come from ``ye3t``; they are not assembled by the methods package.

The exact physical-image map reduces selected raw rows to one commutative
moment algebra. With ``M[alpha] = sum_j product_(q in alpha) phi_(j,q)``, its
two-tag distinct-neighbor term is

.. code-block:: text

   (M[g_1] M[g_2] - M[g_1 union g_2]) product_a M[q_a]

Residual density factors may still include tagged neighbors. The raw rows
are selected before forming their exact physical image; selecting columns
from the full image afterward would change the subspace. For this bounded
homogeneous ``N=4``, ``l=1`` construction, the compiler checks
``dim I_S = M [1(S intersects {0,1}) + 1(2 in S)]`` against the materialized
image. The exact coefficient metric gives a moment monomial of occupations
``a_g`` weight ``product_g a_g!``. Orthonormality is a statement about those
compiler coordinates, not a fitted-data Gram matrix or a configuration-space
``L2`` claim.

The certified one-neighbor source uses a cutoff Jacobi radial function with
factor ``x^l (1-x)^2 P_q^(4,2l+2)(2x-1)``, where ``x=r/r_c``. The current
numerical source plan evaluates the Jacobi polynomial and derivative through
a differentiated three-term recurrence. Exact expanded coefficients remain
in the compiler artifact for provenance. The cutoff value and first radial
derivative vanish, but every channel is not certified Cartesian ``C1`` at
exact overlap. Native source parity is qualified for ``r > 1e-12`` Å;
``r=0`` is rejected. The archived Ta fit is a residual to a ZBL reference,
which must be restored during deployment, as in the paper ASE and LAMMPS
examples.
