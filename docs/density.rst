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

Constructing the basis
----------------------

``Basis(source="density")`` accepts an ordered ``elements`` list, cutoff in
angstrom, ``max_rank``, radial cap ``nmax``, angular cap ``lmax``, radial decay,
and backend. ``nmax`` and ``lmax`` may each be one integer or one value per rank.
The compact path fixes the final angular target to ``L=M=0`` and does not impose
a small label cap. Larger ranks and angular limits can increase compilation
time and descriptor width substantially; inspect ``len(basis.labels)`` before
fitting a large dataset.

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
``forces`` in ``atoms.arrays``. It can also read these keys from attached
calculator ``results``; it does not call a calculator to synthesize missing
targets. The keys and energy/force weights are explicit ``fit`` arguments.
The compact density fit accepts energy and force rows; it rejects nonzero
``stress_weight``. The retained low-level scalar ACE calculator can evaluate
homogeneous-strain stress through its own derivative route.

The compact fit solves ridge normal equations over the selected descriptor
columns. Repeated fits to an identical fixed dataset may use the retained
low-level training-side design-matrix cache. That cache represents the selected
``X`` and targets ``y``; it does not cache site-basis values across MD steps.

The retained descriptor-first flow uses ``YE3TDescriptors.ace(config)`` and
``YE3TModel.linear(descriptor, fit_config, structures=structures)``. It exposes
settings such as explicit ``manual_labels``, charge channels, training matrix
caches, and specialized fast paths. A manual label is a full coordinate request
including its multiplicity key; it is validated by the compiler. It must not
be assembled by taking an unconstrained Cartesian product of radial, angular,
and Young choices.

The retained low-level fitting surface can also select a scikit-learn ridge
solve with ``fit_method="ridge"`` and ``sklearn_params={"alpha": ...}`` after
descriptor rows are built. Install the optional ``fit`` extra for that route.
The compact ``LinearModel.fit`` uses its stated normal-equation ridge path;
changing solvers does not change the descriptor columns.

See :doc:`labels_and_runtime` for label provenance and derivatives.
:doc:`deployment` covers saved ``.pt`` bundles and strict ``.yace`` export.
:doc:`evaluators` distinguishes the Torch ASE backend from low-level source
kernels and gives an explicit analytic-force example.
