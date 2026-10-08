Ordinary density and linear ACE
===============================

For a scalar ACE model, the descriptor map evaluates per-center invariant
columns ``B`` and a fixed linear readout:

.. math::

   E = \sum_i \sum_\alpha B_{i\alpha} w_\alpha + b,
   \qquad
   F_{ia} = -\frac{\partial E}{\partial r_{ia}}.

The one-neighbor site basis is first summed into ordinary density ``A``.
Coupled products of ``A`` become scalar ``B`` columns. Because ordinary
commutative density has already discarded neighbor ordering, its global Young
sector is trivial. Rank ``N`` still describes the number of density factors;
it is not a physical neighbor count when the same neighbor contributes to
several factors.
Products of density sums allow the same neighbor occurrence to contribute to
more than one factor. They do not impose distinct-neighbor sampling or retain
an ordered neighbor role. A nontrivial global Young projector therefore cannot
be added after this sum to recover an ordered physical carrier.

Constructing the basis
----------------------

``Basis(source="density")`` accepts an ordered ``elements`` list, cutoff in
angstrom, ``max_rank``, radial cap ``nmax``, angular cap ``lmax``, radial decay,
and backend. ``nmax`` and ``lmax`` may each be one integer or one value per rank.
The compact path fixes the final angular target to ``L=M=0`` and does not impose
a small label cap. Larger ranks and angular limits can increase compilation
time and descriptor width substantially; inspect ``len(basis.labels)`` before
fitting a large dataset.

``Basis.from_config`` materializes multi-species PACE density at the selected
ranks with explicit, one-hot, or fixed-embedding chemistry on the CPU
Torch/ASE route. The compiler sees complete chemical and radial content
before coupling; public labels retain physical radial indices separately
from compiler content IDs.
A fixed matrix acts on neighbor species before density aggregation; each
independent matrix column is one physical chemical channel. The matrix and
species row order are saved with the source and change its resolution hash.
Physical-content binding uses scalar factors without charge channels. Native
CPU and YACE export of bound multi-species or fixed-embedding density are
unavailable; an explicit request raises an error.

For an O(3) parent with ``L>0`` and an explicitly selected parity,
``Basis.from_config`` also creates complete density multiplets on the CPU
Torch/ASE route. ``Basis.create(atoms)`` returns
``(n_atoms, n_multiplets, 2*L+1)`` real-tesseral rows. One feature label
identifies one compiler multiplicity coordinate and all of its magnetic
components. The exact count preview and the compiled feature axis must agree
at each rank. Repeated radial/angular content is compiled by composing
validated symmetric-block polynomials with the full magnetic angular
schedule, including every valid multiplicity copy. This route supports
explicit, one-hot, and fixed chemical embeddings; periodic image
displacements use the ASE neighbor builder's Cartesian vectors. Rotational
and inversion tests cover ranks
through ``L=3``, including an even-parity ``L=1`` pseudovector. Native and
CUDA property evaluation remain separate capabilities.

The seven-section standard fit config accepts ``model.output.scope="per_atom"``
for these multiplets, with ``targets.per_atom`` naming a stored ASE atom array.
The target input is ``real_tesseral`` for any supported ``L``. ``cartesian``
also accepts a polar vector for odd ``L=1`` or a symmetric traceless tensor
for even ``L=2``. Ridge, LASSO,
and ARD use one coefficient per compiler feature shared across every magnetic
component. Density feature columns already encode the central species, so
the fit does not add another species block. ARD can return a per-atom
component covariance through ``model.predict(atoms, uncertainty=True)`` and
the ASE property calculator.

``LinearModel.write`` saves the full-M density model as a hashed
``.ye3t.json`` artifact containing all compiled magnetic specifications,
site-basis settings, coordinate order, real-form convention, coefficient
metric, and fit data. The writer includes a v2 native property plan bound to
the saved compiler blocks and fitted readout. ``LinearModel.read`` validates
these records and replays the saved coefficient basis without running the
coupling compiler. The separate LAMMPS CPU
``compute ye3t/property/atom`` reads these v2 artifacts for the qualified
``L=1`` odd and ``L=2`` even cases with explicit delta chemistry and identity
PACE radial channels. The artifact is a per-atom property model; it has no
energy/force/stress readout. Use ``model.write`` for that compute; the scalar
``model.export_lammps`` method does not apply. Python/ASE already exercises
``L=3``; the density Kokkos property evaluator remains a separate unfinished
backend gate. A multi-component
density-plus-tagged covariant basis is rejected during config resolution
while ``Basis.combine`` supports scalar outputs only.
The configured scalar combination is demonstrated in
``examples/quickstart/combined_density_tagged_fit.py``.
For a complete ASE fit, saved model, and native per-atom property input, run
``examples/quickstart/per_atom_vector_to_lammps.py``; see :doc:`quickstart`.

The default catalogue keeps even ``sum(l)`` scalar labels, so its columns are
invariant under spatial inversion. The lower-level
``YE3TDescriptors.ace(config)`` accepts ``parity_filter="none"`` for valid
odd-parity SO(3) pseudoscalars. Those columns change sign under inversion and
should only enter a model when that behavior is intended. Both direct and
factorized evaluation return checked real scalar values for these labels.

The runnable :doc:`quickstart` is a deliberately small numerical fixture. A
transferable model should choose its rank and radial/angular schedule for the
actual dataset; for example, one may construct ranks through four with visible
per-rank caps:

.. code-block:: python

   from ye3t_methods import Basis

   basis = Basis(
       elements=["Si"], cutoff=5.0, max_rank=4,
       nmax=(6, 4, 3, 2), lmax=(2, 2, 2, 1),
   )
   print(len(basis.labels))

Training labels
---------------

``LinearModel(basis).fit(structures)`` takes ASE ``Atoms`` with precomputed
``energy`` in ``atoms.info`` and, when ``force_weight`` is nonzero,
``forces`` in ``atoms.arrays``. With nonzero ``stress_weight``, each structure
also supplies six ASE Voigt stress components in ``atoms.info["stress"]``.
It can also read these keys from attached
calculator ``results``; it does not call a calculator to synthesize missing
targets. The keys and energy/force/stress weights are explicit ``fit`` arguments.
Density stress fitting uses analytic homogeneous-strain derivatives. Ridge
uses normal equations; optional LASSO and ARD use the same stress rows through
the compact ``LinearModel.fit`` interface. A positive cell volume is required.

The compact fit solves ridge normal equations over the selected descriptor
columns. Repeated fits to an identical fixed dataset may use the
training-side design-matrix cache. That cache represents the selected
``X`` and targets ``y``; it does not cache site-basis values across MD steps.

The descriptor-first compatibility flow uses ``YE3TDescriptors.ace(config)`` and
``YE3TModel.linear(descriptor, fit_config, structures=structures)``. It exposes
settings such as explicit ``manual_labels``, charge channels, training matrix
caches, and specialized fast paths. A manual label is a full coordinate request
including its multiplicity key; it is validated by the compiler. It must not
be assembled by taking an unconstrained Cartesian product of radial, angular,
and Young choices.

The low-level fitting interface can also select a scikit-learn ridge
solve with ``fit_method="ridge"`` and ``sklearn_params={"alpha": ...}`` after
descriptor rows are built. Install the optional ``fit`` extra for that route.
The compact ``LinearModel.fit`` uses its stated normal-equation ridge path;
changing solvers does not change the descriptor columns.

See :doc:`labels_and_runtime` for label provenance and derivatives.
:doc:`deployment` covers saved ``.pt`` bundles and strict ``.yace`` export.
:doc:`evaluators` distinguishes the Torch ASE backend from low-level source
kernels and gives an explicit analytic-force example.

Lifted density (experimental)
-----------------------------

The experimental lifted-Cauchy scalar route constructs a source with an
explicit role coordinate and compiler-issued coupling data. Its tested joint
source has direct and factorized shifted-Jacobi radial realizations with source
derivatives. Much of the observed change from ordinary density can be a
radial-basis change. A role coordinate alone does not prove that a nontrivial
global Young sector survives physical neighbor summation; that claim needs a
validated role-sensitive image. The route is experimental and has no compact
``Basis.from_config`` fit adapter in this release.

The source neighbor builder excludes self edges. Density products can reuse a
neighbor in different factors, so the lifted route does not acquire a
distinct-neighbor rule merely from its source name. Its saved native scalar
bundle and derivatives have their own tested scope; they do not establish the
ordered-``Phi`` or full-multiplet tagged results. Those routes have separate
compiler, physical-carrier, and symmetry checks.
