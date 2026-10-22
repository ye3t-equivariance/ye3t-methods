Role-resolved density ``A_s``
=============================

The configured shell-density workflow is
``YE3TRepresentation`` → ``Basis`` → ``LinearModel`` with
``basis.tensor_product.kind="shell_density"``. It retains shell-role actions
while constructing invariant scalar energy features. Use
``examples/quickstart/shell_density_fit.py`` and :doc:`shell_linear` for the
complete fitting and ASE workflow.

The descriptor-first linear path represents a center ``i`` with a
role coordinate ``s``. For chemical/radial channel ``eta`` and angular
component ``(l,m)``, its general channel map is

.. math::

   A_{i,s,\eta lm}
   = \sum_{\eta'} K^{(l)}_{i,s;\eta\eta'} A_{i,\eta'lm}.

The implemented diagonal radial-filter form evaluates

.. math::

   A_{i,s,c} = \sum_j q_s^{\mu_i,\mu_j}(r_{ij})
      \phi_c^{\mu_i,\mu_j}(r_{ij}, \hat r_{ij}).

``K`` acts on non-angular channels and the identity on each angular irrep.
The role axis remains available through the scalar readout. A
role permutation acts as

.. math::

   (g\cdot A_i)_{s,c} = A_{i,g^{-1}s,c}.

Ordinary commutative ``A`` pools away the role coordinate. Nontrivial role
information may appear in intermediate channels. A scalar linear energy
readout couples or projects it to the trivial role and ``L=0`` target.

For forces, differentiating the role map gives

.. math::

   dA_s=(dK)A+K(dA),
   \qquad
   dA_{i,s,c}=\sum_j\bigl[q_s(i,j)d\phi_c(i,j)
      +dq_s(i,j)\phi_c(i,j)\bigr].

A geometry-dependent filter needs both terms. Normalized and raw density
values, along with their derivatives, are distinct runtime quantities.

Scalar routes
-------------

The linear implementation exposes role-trivial, trivial-plus-standard,
and selected role-Specht scalar readouts. The central-projector norm and
commutant quadratic-form variants are scalar readouts of projected carriers.
Nontrivial
role-standard or role-Specht readouts require a homogeneous active role count
across ordered pair types. Variable active roles can be used for the restricted
role-trivial scalar readout. Zero padding unlike carriers does not create one
shared nontrivial permutation representation.

The historical descriptor-first ``filtered_A_s`` fit is preserved in
``tools/legacy_examples/role_density_fit.py`` and checked by
``tests/test_lifted_density_linear_fit.py``. Its scalar readout choices describe
that compatibility implementation. New linear workflows use the canonical
shell source in :doc:`shell_linear`, with complete internal rotation/permutation
couplings, position derivatives, stress rows and saved ASE models.
