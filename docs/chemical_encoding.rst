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

The retained low-level ``SiteBasisV2`` accepts a ``chemical_provider``
callable. A fixed species embedding matrix ``E`` can define a chemical
kernel ``K = E E^T``, with source factor

.. math::

   C_{ij}^{ab} = K_{\mathrm{type}(i),a}
                  K_{\mathrm{type}(j),b}.

For ``E=I``, this recovers the one-hot factor exactly. A lower-dimensional
fixed ``E`` mixes species channels but does not automatically reduce the
compiled descriptor width. Set the embedding before fitting and keep it fixed
for every evaluation of that fit. The
source archive's ``examples/quickstart/chemical_encoding.py`` is an
executable two-species demonstration: it constructs a compact one-hot
``Basis``, evaluates low-level source channels with both providers, and
checks the predicted linear channel transform.

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

A fixed embedding that both mixes all species and reduces the compiled
chemical width still needs a saved source-label and deployment contract.
The example marks that work as a TODO.

The custom provider hook is an in-memory source evaluation path. Current
``LinearModel.write/read``, strict YACE export, tagged JSON export, and
the LAMMPS consumer do not serialize or deploy an arbitrary custom
chemical provider. They therefore must not be used to save or deploy a
model fitted with this example's custom kernel. Trainable embeddings would
make the source features parameter-dependent and require a separate fit,
serialization, derivative, and deployment contract; the current fixed-feature
linear API does not claim that route.

Chemical channels also appear in ``Phi`` motifs and tagged physical
source schedules. Their slot/role semantics differ from ordinary density:
do not apply an ordinary-density channel mix to a tagged or role-resolved
carrier without preserving the declared group action. The core
``ye3t.couplings`` report determines valid coupling labels after the
source channels are specified.
