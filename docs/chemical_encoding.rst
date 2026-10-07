Chemical channels and embeddings
================================

The ordinary linear ACE source uses a chemical channel for each ordered
center/neighbor species pair. With the stable default
``SiteBasisConfig.chemical_basis="delta"``, the chemical factor is

.. math::

   C_{ij}^{\mu_0\mu}
   = \delta_{\mathrm{type}(i),\mu_0}
     \delta_{\mathrm{type}(j),\mu}.

This is one-hot species encoding. ``Basis(elements=[...])`` orders the
species and builds these channels through the existing descriptor compiler.
Its ``FeatureLabel.one_factor_channels`` records the actual center and
neighbor species for each column. The paper's six elemental models also use
``chemical_basis="delta"``; one species makes the chemical factor constant
on their accepted edges. The distinct linear benefit in the paper is from
the tagged physical image and exact coupling selection, not a learned
species embedding.

Fixed chemical embeddings
-------------------------

The configured ``Basis.from_config`` route accepts
``single_factors.chemical.kind="fixed_embedding"`` with an explicit
``species_order`` and full-column-rank species-by-channel matrix ``E``.
Each neighbor species contributes its corresponding row of ``E`` to the
physical one-factor density source. The compiler counts and labels the
resulting physical channels, including their higher-rank couplings.

For ``E=I``, this recovers the one-hot neighbor factor. A narrower
full-rank ``E`` reduces the physical chemical channel count before
coupling. Set the matrix before fitting and keep it fixed for every
evaluation of that fit. The source archive's
``examples/quickstart/chemical_encoding.py`` builds a three-species ASE
``Atoms`` object, evaluates rank-one and rank-two scalar descriptors
through the public representation and basis objects, and checks rotation
and atom-order invariance.

Selecting fewer exact channels
------------------------------

If the model deliberately excludes a subset of neighbor species, the
descriptor-first ``YE3TDescriptors.ace`` constructor accepts
``restrict_neighbor_mu``. The
``examples/quickstart/chemical_channel_selection.py`` script selects Li
neighbor channels from a Li/Na source and reduces the rank-one descriptor
count from four to two. It retains exact one-hot encoding for the selected
channels. Excluded neighbors contribute no selected channel, so this option
is appropriate only when that restriction is part of the model design. It is
not a low-rank replacement for all species interactions.

The older low-level ``SiteBasisV2.chemical_provider`` hook remains an
in-memory research extension. It is separate from the configured fixed
embedding above and has no general model serialization contract. A
configured fixed embedding can be fitted and saved by the ordinary density
linear route, but strict PACE/YACE export requires delta channels and
rejects it. Trainable embeddings would make source features depend on
parameters and require a separate fit and derivative contract.

Chemical channels also appear in ``Phi`` motifs and tagged physical
source schedules. Their slot/role semantics differ from ordinary density:
do not apply an ordinary-density channel mix to a tagged or role-resolved
carrier without preserving the declared group action. The core
``ye3t.couplings`` report determines valid coupling labels after the
source channels are specified.
