Descriptor labels, derivatives, and runtime
===========================================

One descriptor column is one accepted, materialized coordinate. ``ye3t`` owns
the representation-theory counts, valid multiplicity labels, coupling plans,
coefficients, and convention reports. ``ye3t-methods`` owns atomic site-basis
values, derivatives, descriptor matrices, fitting, saved models, and ASE use.
The application must not create its own Young/coupling paths by combining
independent option lists.

``Basis.labels`` is ordered exactly as the fitted feature columns. A
``FeatureLabel`` contains a zero-based ``feature_index``, source, stable
identity, and structured details. ``feature_index`` is a public column ordinal,
not a multiplicity-copy number. Density details include rank ``N``, target
``L,M``, radial and angular inputs, intermediate angular labels, one-factor
channels, and the compiler basis key. Tagged details include the image
coordinate provenance and its contributing raw opportunities; Phi details
include motif and compiler-plan metadata. ``basis.describe(j)`` and
``model.describe(j)`` provide bounded text, while ``as_dict()`` returns the
structured record. Text truncation does not alter saved identity or fitted
column order. The runnable ``examples/quickstart/inspect_features.py``
demonstrates these views.

Derivatives
-----------

For a product term ``c`` times density factors, the analytic derivative is
the explicit product rule. A repeated component monomial obeys

.. math::

   \frac{\partial}{\partial A_q}
   \left(c_\alpha\prod_p A_p^{\alpha_p}\right)
   =c_\alpha\alpha_q\prod_p A_p^{\alpha_p-\delta_{pq}}.

The descriptor adjoint is then passed through the site-basis derivative and
normalization record to positions. The CYprime path uses explicit
forward/backward products. An eligible repeated-block globally trivial ACE
sector may use the compiler-owned symmetric-power plan; unsupported blocks
use the validated general product path. A fast path is an evaluation choice,
not a new label source.

Caches and reports
------------------

Compiler counts and plans can be inspected before coefficient materialization.
Compiled coupling artifacts and source geometry rows have separate cache
identities. A low-level descriptor-matrix cache stores a fixed training design
matrix and targets, not the evolving geometry of ASE dynamics. Changes in
selected structures, cutoff, type map, source, or derivative convention must
invalidate affected geometry or training rows. Device, dtype, backend, and
convention metadata belong in runtime reports. A descriptor validation report
records accepted labels and mathematical provenance; calculator ``results``
records physical energy/force/stress outputs. Neither a shape check nor a
finite training error alone establishes equivariance or physical accuracy.

The low-level linear families include a fitted lifted Cauchy
scalar model with direct/factorized source derivatives and a saved native
bundle, and an evaluator for fixed Young descriptor sets. The release record
states their actual tested operations. They do not have compact ``Basis`` fit
adapters in this release.
