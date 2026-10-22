Tagged physical-image basis
===========================

The tagged route keeps explicitly selected neighbor roles through coupling
and constructs globally invariant scalar coordinates. For a configured linear
fit, use ``ye3t.YE3TRepresentation`` with ``Basis.from_config`` and
``LinearModel``. The direct ``Basis(source="tagged_cauchy_image")`` constructor
and ``YE3TRepresentation.tagged_cauchy_image()`` descriptor-first route support
specialized catalogue inspection and compatible models.

What the basis distinguishes
----------------------------

``N`` is tensor-product order and ``s`` is the number of ordered tags. A tag
permutation, a permutation of formal tensor positions, and a relabeling of
physical neighbor atoms are different operations. Intermediate tag and role
Young sectors may be nontrivial, while the final energy coordinate is scalar
and globally invariant. Coupling paths and coefficients come from ``ye3t``;
the application does not enumerate them independently.
The explicit tag positions use distinct ordered neighbor occurrences. The
remaining density factors are inclusive: they may use a tagged neighbor or
reuse a neighbor used by another density factor. The compiler's collision
reduction determines the multiplicity weights in this physical image.
Summing over physical tag tuples requires a globally trivial tag output; an
overall tag-odd coordinate vanishes. The compact fitted route here is scalar.
The exact selected full-``M`` physical-image basis and a per-atom linear fit
are available for the CPU reference route with ``L>0`` below. Symmetry tests
currently cover ``L=1,2,3``.

For a new scalar fit, construct ``ye3t.YE3TRepresentation`` and pass it to
``Basis.from_config`` before fitting ``LinearModel``. The seven-section
``examples/quickstart/tagged_fit.py`` demonstrates this path. Its
``basis.catalogue.angular_patterns_by_rank`` explicitly selects the
input angular patterns at ranks 1–8, including the rank-four ``(1,1,1,1)``
pattern and repeated source blocks. Its scalar features include columns
with nontrivial local Young partitions ``(1,1)`` and
nonzero block angular outputs ``(1,1)``, while the final energy has ``L=0``.
The field accepts a nonempty list
of unique rank-length patterns for every selected rank, with angular degrees
within that rank's ``lmax_per_rank``. It currently applies to scalar tagged
bases only. Omitting it asks the compiler for every permitted pattern up to
the angular cap and can change the fitted column set. The example's
``native_cpu`` evaluator requires the installed tagged C ABI library.

``examples/long/tagged_fit_large.py`` expands this catalogue to 691 independent
features with additional high-rank radial channels and a rank-four
``(2,2,2,2)`` angular pattern. It uses the same public workflow and includes a
complete fit/save/reload test. Allow substantial initial compilation time.
The full workflow exceeded 10 GiB of process memory in the development check;
the saved JSON model is approximately 256 MiB.

For the tested homogeneous two-tag case, ``N=4``, input ``l=1``, and
``s=2``, the compiler includes a nontrivial tag sector
``tag_kappa=(1,1)`` with a ``role_kappa=(2,1,1)`` companion. The compiler
maps selected raw opportunities into the exact physical image and removes
source identities before it exposes fitted columns. A public column can
therefore combine multiple raw opportunities. Its printed tag summary is not
a claim that it is one pure tag partition. The tested example shows how to
inspect the full contributing records.

For the same-source inclusive-density Ta case, the selected one-tag row
aliases the ordinary zero-tag row. Other source families require their own
identity check. The compiler validates requested orders, contents, angular
channels, and tag counts separately.

Ordered tagged carriers
-----------------------

The historical ``tools/legacy_examples/tagged_carriers_ase.py`` inspects raw zero-, one-, and two-tag
carriers on a Ni ASE cell. Its config exposes species, cell size, cutoff,
rank and radial/angular caps, tag counts, sector policy, and evaluator backend.
The result has separate center, edge, and ordered edge-pair value blocks.
These carrier values are not fitted scalar energy coordinates.

``descriptor.create(atoms, descriptor_evaluation="pooled_carriers")`` sums
each complete magnetic carrier over distinct ordered neighbor-image supports
at its center. It retains compiler labels and reports the schedule hashes and
pooling convention. The result is an unreduced per-center orbit sum that may
contain zero or linearly dependent coordinates. This mode provides values for
inspection. The physical-image route below reduces those coordinates and has
focused value and derivative checks.
The integrated physical checks currently cover zero, one, and two tags.

``descriptor.create(atoms, descriptor_evaluation="physical_image")`` first
uses exact compiler lowering of distinct-tag collisions and exact pivots
within each rotation and parity sector. It then evaluates only the selected
original coordinates. Its returned plan contains selected coordinate IDs and
exact reconstruction coefficients for every candidate; one selection is
checked across all magnetic components. The output coordinate ID list records
the returned schedule order, which may differ from the plan's global candidate
order. This mode currently supports tag
counts zero, one, and two. It is a complete-multiplet image evaluator. Its
value and derivative paths have focused checks; high-level force/stress
fitting is still in progress.

``Basis.from_config`` also exposes selected tagged multiplets for one or more ranks
when the representation parent requests ``L>0``. Use the same
``single_factors`` shifted-Jacobi source, tagged tensor product, and catalogue
fields as the scalar config, with the desired ``parent.L`` and parity. Its
``Basis.create(atoms)`` result has shape
``(n_atoms, n_selected_multiplets, 2*L+1)`` in real-tesseral signed-``M``
order. ``basis.labels`` holds the original compiler coordinate IDs and
physical-image plan hash. This route accepts per-rank zero/one/two tag
count sets, explicit or one-hot chemistry, CPU ``reference`` evaluation, ASE
neighbors, and cache mode ``auto`` or ``off``. It reports unsupported native,
CUDA, or tag counts above two before materialization. Scalar tagged
``Basis.create`` and saved paper-model evaluation return 2D rows.
When more than one rank is requested, the compiler combines rankwise
compiled source records into one exact pooled physical O(3) image. The saved
labels retain each rank and its selected original coordinate ID. Exact
reconstruction may mix ranks: the pooled image is not rank graded and has no
common ``S_N`` action. Each original source keeps its own formal parent.

Full-multiplet per-atom fitting
---------------------------------

Construct ``Basis.from_config`` with a tagged ``L>0`` parent,
then call ``LinearModel(basis).fit(structures,
config=config)`` with the complete seven-section standard config. Set
``model.kind="linear"``, ``model.output.scope="per_atom"``, and
``model.fit.solver`` to ``ridge``, ``lasso``, or ``ard``. Set
``targets.per_atom`` to a mapping with ``key``, ``input``, and ``units``;
``input`` accepts ``real_tesseral`` for any supported ``L``, or ``cartesian``
for the specified vector/tensor cases. Each training ASE Atoms
stores the target in ``atoms.arrays[key]``. The Cartesian input is a polar
vector for odd ``L=1`` or a symmetric traceless ``3x3`` tensor for even
``L=2``. Scalar reference energies and energy, force, or stress targets do
not belong to this fit. ``validation.checks=["round_trip"]`` checks an
embedded-compiler save and reload on the first frame.

The coefficients are shared across every magnetic component of a selected
multiplet and kept separate for each central species. The selected compiler
coordinates define the saved coefficient order. Ridge and LASSO use an
identity penalty in those coordinates; ARD uses the same fixed coordinate
gauge. Changing to another basis for the same physical image can change
these regularized coefficients. The model saves that metric, every coordinate
ID, the image-plan hash, the fitted coefficients, and the full compiled
catalogue in a ``.ye3t.json`` artifact. Reading that artifact checks the
hashes and source requests and does not recompile its coupling coefficients.
These embedded hashes check consistency; they do not authenticate who produced
the model.

``model.predict(atoms)`` returns ``mean_real_tesseral`` with shape
``(n_atoms, 2*L+1)``. For an ARD fit, ``predict(atoms,
uncertainty=True)`` also returns the per-atom component covariance
``covariance_real_tesseral``. This is conditional coefficient uncertainty;
it excludes target noise and model error. ``model.ase_calculator()`` exposes
``per_atom_real_tesseral_mean`` and, for ARD, the covariance as ASE
properties. The fitted model is currently a reference CPU per-atom property
model. Its v2 saved JSON is consumed by the separate
``ye3t-lammps`` CPU ``compute ye3t/property/atom`` for mean inference.
The model has no ``export_lammps`` method or force/stress readout for this
per-atom target.

The real-tesseral component axis follows the compiler's cosine, zero, sine
order: ``cos(L)..cos(1), 0, sin(1)..sin(L)``. For a Cartesian polar vector
``(x,y,z)``, the ``L=1`` components are ``(x,z,-y)``. For a symmetric
traceless quadrupole ``Q``, the ``L=2`` components are
``((Qxx-Qyy)/sqrt(2), sqrt(2)*Qxz, (2*Qzz-Qxx-Qyy)/sqrt(6),
-sqrt(2)*Qyz, -sqrt(2)*Qxy)``. The conversion is norm preserving and its
exact convention hash is retained when Cartesian labels are fitted.
The hash includes both the Cartesian map and the core real-to-complex
tesseral phase convention.

``nmax_per_rank`` and ``lmax_per_rank`` are exact maps covering every requested
rank. ``cutoff_A`` bounds the neighbor list, while ``pair_cutoffs_A`` sets the
radial cutoff for each ordered species pair and cannot exceed ``cutoff_A``.
The validated source is the fixed origin-regular shifted-Jacobi family. It
has no ``radial_lambda`` or ordinary-density ``radial_decay`` parameter;
unsupported settings raise an error.

``tag_character=-1`` on a two-tag label denotes odd exchange of the two
ordered tags. ``target_L`` identifies that carrier's rotation irrep. Neither
is the global parent Young partition of an atomistic descriptor. The example
uses ``sector_policy="tagged_mixed"`` to retain both tag-exchange types;
``backend="reference"`` selects the carrier reference evaluator. Standalone
carriers currently accept only that evaluator. The scalar tagged ``Basis``
below also requests the reference evaluator for a portable descriptor-only
example. The separate tagged fit example can use the native C++ ASE
calculator; its backend option does not apply to the raw carrier example.

Building, fitting, and inspecting
---------------------------------

The runnable ``examples/quickstart/tagged_descriptors.py`` script shows a Ni fcc
cell and a complete editable seven-section config. It uses ranks 1–8, radial
counts 4,3,2,2,2,2,2,2 and angular caps 0,4,2,2,1,1,1,1. The selected
angular patterns include scalar factors, vector pairs and rank-four quadrupole
products. Source partitions include repeated blocks with nontrivial internal
Young/angular couplings. The parent is symmetric, scalar and even.

.. code-block:: python

   representation = YE3TRepresentation.from_config(config["representation"])
   basis = Basis.from_config(config["basis"], representation=representation,
                             runtime=config["runtime"])
   rows = basis.create(atoms)
   print(rows.shape, basis.labels[0].as_dict())

``tag_counts_per_rank`` selects one tag at rank one and two tags at higher
ranks in this example. ``angular_patterns_by_rank`` and
``source_block_partitions_by_rank`` select physical input contents; the compiler
determines their independent scalar image and coupling coefficients.
The default native CPU evaluator uses the compiled polynomial plan.
Full mathematical validation is performed when loading the compiled source;
the persistent cache avoids recomputing its coefficients for identical requests.
Cold compilation cost depends on the requested contents and angular resolution.

``examples/quickstart/tagged_fit.py`` fits this same catalogue to labeled ASE
structures. ``saved_ase_export.py`` loads that fitted model and writes a
LAMMPS AUTO model and input. The complete saved paper catalogue can be inspected
using ``paper_ni_portable_ase.py``.

For existing descriptor-first applications, ``YE3TDescriptors.ye3t_basis``
accepts the lower-level catalogue format in :doc:`api_reference`.
The direct ``Basis`` constructor uses ``rank`` for a single tensor order and
``tag_counts`` for its tag counts. The configured form uses explicit maps
over ranks. Both resolve their labels through the core compiler.
The shifted-Jacobi source uses ``cutoff_A`` in the standardized config and
has no adjustable radial-lambda parameter. General scalar tagged compilation
uses the exact requested-weight angular compiler; ``legacy_exact`` selects
the full-sector oracle. Final scalar coefficients must be real in the
real-tesseral convention. Intermediate complex-to-real conversion matrices
can contain imaginary entries.

The compact linear model fits precomputed energies and forces and can include
ASE Voigt stress rows with ``stress_weight``. The fixture supplies these labels.
``examples/quickstart/tagged_fit.py`` uses a compact two-block catalogue and
reports both source-block Young partitions and angular intermediates. Its
bundled Ta targets are manufactured interface fixtures, so its fitted error
is not an accuracy result for a physical Ta potential.
``LinearModel.write`` produces a versioned, hash-bound ``.ye3t.json`` model;
``LinearModel.read`` restores it. An ASE calculator can evaluate energy,
forces, and stress. ``export_lammps`` writes the tagged deployment artifact;
the separate LAMMPS consumer must support its schema. The runnable saved-model
path is ``examples/quickstart/saved_ase_export.py``.
Choose ``reference``, ``native_polynomial``, or ``native_cpu`` with
``model.ase_calculator(backend=...)``; see :doc:`evaluators` for the C++ library
argument, optional neighbor-list speedup, and the paper composite loader.

The tagged source uses a cutoff Jacobi radial/angular realization. Its exact
source algebra and floating evaluator are recorded in compiler artifacts.
Physical accuracy depends on training and validation for the intended system.
Exact overlap is outside the tested source domain. The separate
``ye3t-lammps`` consumer defines its own CPU and Kokkos deployment scope.

Rank-four reference construction
--------------------------------

For fixed tensor order and source content, the repeated source blocks follow
the Cauchy decomposition

.. math::

   \operatorname{Sym}^{k}(W \otimes V_l)
   = \bigoplus_{\kappa \vdash k} S_\kappa(W) \otimes S_\kappa(V_l).

The two-tag placement carrier has ordered basis vectors
``e_(i,j)`` for ``i != j`` and dimension ``N(N-1)``. Formal tensor-position
permutations act on the left, while swapping the two tag labels acts on the
right. The sign tag type ``(1,1)`` survives final scalar coupling only with a
matching physical source carrier. In the tested case, ``N=4``, ``s=2``,
``placement_parent_lambda=(3,1)``, ``role_kappa=(2,1,1)``, and
``angular_kappa=(2,2)``. Each input has ``l=1`` and the final ``L=0``.
Physical-neighbor relabeling is a separate action. These labels and their
intertwiner come from ``ye3t``; they are not assembled by the methods package.

The exact physical-image map reduces selected raw rows to one commutative
moment algebra. With ``M[gamma] = sum_j product_(q in gamma) phi_(j,q)``, where
``gamma`` is a source-channel multiset, its
two-tag distinct-neighbor term is

.. code-block:: text

   (M[g_1] M[g_2] - M[g_1 union g_2]) product_h M[q_h]

Residual density factors may still include tagged neighbors. The raw rows
are selected before forming their exact physical image; selecting columns
from the full image afterward would change the subspace. For this bounded
homogeneous ``N=4``, ``l=1`` construction, the compiler checks
``dim I_S = M [1(S intersects {0,1}) + 1(2 in S)]`` against the materialized
image. The exact coefficient metric gives a moment monomial of occupations
``a_g`` weight ``product_g a_g!``. Orthonormality is a statement about those
compiler coordinates, not a fitted-data Gram matrix or a configuration-space
``L2`` claim.

The one-neighbor source uses a cutoff Jacobi radial function with
factor ``x^l (1-x)^2 P_q^(4,2l+2)(2x-1)``, where ``x=r/r_c``. The
numerical source plan evaluates the Jacobi polynomial and derivative through
a differentiated three-term recurrence. Exact expanded coefficients remain
in the compiler artifact for provenance. The cutoff value and first radial
derivative vanish. Cartesian ``C1`` behavior at exact overlap lies outside the
tested domain. Native source parity is qualified for ``r > 1e-12`` Å;
``r=0`` is rejected. The archived Ta fit is a residual to a ZBL reference,
which must be restored during deployment, as in the paper ASE and LAMMPS
examples.
