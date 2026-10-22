Explicit Phi clusters
=====================

``Basis.create_cluster`` evaluates the selected explicit-Phi coupling below on
an ASE structure. Inputs identify a center and eight distinct physical
occurrences, each given by an atom index and integer periodic-image shift.
The first four factors use radial index zero and the last four use radial
index one; every factor has angular degree one.

This specialized construction selects the parent ``(4,4), L=2, even`` and the
local path ``kappa=((4),(4)), Lambda=(0,2)``. Its result has shape ``(1,14,5)``:
one multiplicity coordinate, 14 tableau components and five real-tesseral
magnetic components. The compiler obtains this selected coordinate from the
18-dimensional fixed-content multiplicity space. The public example checks
geometry through the basis API; covariance and factor-action checks belong to
the regression suite.

.. literalinclude:: ../examples/advanced/explicit_phi_cluster.py
   :language: python

The input order binds each occurrence to its radial factor. Permutation tests
must transport the factor identities with the physical arguments. Rotation
acts on the complete magnetic multiplet at each tableau coordinate.

This API currently supports the selected rank-eight construction shown above.
General motif-orbit sums, arbitrary cluster catalogues and their linear
potential fitting remain implementation work. The example evaluates the
supplied occurrence directly and makes no orbit-sum energy claim.

Legacy motif averages
---------------------

Saved ``.phi.pt`` models use the earlier ``bar_phi`` motif-average evaluator.
It averages fixed factor channels over compatible assignments and supports
the existing scalar fit, force and stress routes. Its fixed magnetic channels
do not provide the complete coupled tensor constructed above. The original
pair/star fit and depth-two demonstration are preserved under
``tools/legacy_examples``; ``tests/test_compact_phi.py`` protects their saved
models and derivatives. This compatibility path has no LAMMPS exporter.
