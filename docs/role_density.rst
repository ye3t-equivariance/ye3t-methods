Role-resolved density ``A_s``
=============================

.. warning::

   In the currently verified slot-trivial linear fit, filtering ``A`` into
   ``A_s`` mainly changes or expands the effective radial channel basis. Do
   not interpret an accuracy gain over a smaller ordinary ACE radial basis as
   evidence for a new permutation mechanism. Compare against ordinary ACE
   with matched radial capacity. A genuinely nontrivial slot-sector readout
   requires its own carrier and validation; merely retaining a slot axis is
   insufficient.

The retained descriptor-first linear path represents a center ``i`` with a
role or slot coordinate ``s``. For chemical/radial channel ``eta`` and angular
component ``(l,m)``, its general channel map is

.. math::

   A_{i,s,\eta lm}
   = \sum_{\eta'} K^{(l)}_{i,s;\eta\eta'} A_{i,\eta'lm}.

The implemented diagonal radial-filter form evaluates

.. math::

   A_{i,s,c} = \sum_j q_s^{\mu_i,\mu_j}(r_{ij})
      \phi_c^{\mu_i,\mu_j}(r_{ij}, \hat r_{ij}).

``K`` acts on non-angular channels and the identity on each angular irrep.
The slot axis is retained until the selected scalar readout has used it. A
slot permutation acts as

.. math::

   (g\cdot A_i)_{s,c} = A_{i,g^{-1}s,c}.

This differs from ordinary commutative ``A``, which has already pooled away
the role coordinate. Nontrivial slot information may appear in intermediate
channels, but a scalar linear energy readout must couple or project it back to
the trivial slot and ``L=0`` target.

For forces, differentiating the role map gives

.. math::

   dA_s=(dK)A+K(dA),
   \qquad
   dA_{i,s,c}=\sum_j\bigl[q_s(i,j)d\phi_c(i,j)
      +dq_s(i,j)\phi_c(i,j)\bigr].

A geometry-dependent filter needs both terms. Normalized and raw density
values, along with their derivatives, are distinct runtime quantities.

Current scalar routes
---------------------

The retained linear implementation exposes slot-trivial, trivial-plus-standard,
and selected slot-Specht scalar readouts. The central-projector norm and
commutant quadratic-form variants are scalar readouts of projected carriers.
Nontrivial
slot-standard or slot-Specht readouts require a homogeneous active slot count
across ordered pair types. Variable active slots can be used for the restricted
slot-trivial scalar readout. Zero padding unlike carriers does not create one
shared nontrivial permutation representation.

The installed-package :doc:`quickstart` includes a complete manufactured
``A_s`` fit using ``YE3TRepresentation.filtered_A_s`` followed by
``YE3TDescriptors.ye3t_basis`` and ``YE3TModel.linear``. It checks energy and
force predictions through ``HybridACELiftedDensityCalculator``. This is the
release-verified descriptor-first path; the compact ``Basis`` class has no
``A_s`` adapter. The older aqueous KCl comparison and model-selection
documentation remains in the original application archive as historical
research material; its pilot metrics are not release qualification.
