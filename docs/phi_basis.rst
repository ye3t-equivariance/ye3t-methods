Explicit motif ``bar_phi`` basis
================================

Ordinary density sums individual neighbor contributions before forming
products. ``bar_phi`` instead selects an explicit cluster motif, assigns
channels to its slots, and evaluates the motif over the local geometry. Its
``N`` counts motif vertices; it must not automatically be read as the physical
body order of an ordinary ACE density product. The compiler supplies the
fixed motif coupling plan and validates the scalar target.

This existing ``bar_phi`` route is a weighted average over ordered motif
embeddings, with distinct **atom indices** in its slots. The evaluator weights
each embedding and divides its sum by the decorated motif automorphism count.
By default it also divides by the correspondingly weighted sum at each center.
It excludes self edges and excludes two images of the
same atom from one motif. ``periodic_image_mode="unique"`` uses one minimum
image per pair and checks a cutoff margin; ``"all_images"`` enumerates image
edges but still enforces distinct atom indices within a motif. It does not
accept an explicit ordered ``(atom_index, cell_shift)`` list.

Each ``PhiSlotChannel`` selects one fixed signed ``m``. The physical evaluator
multiplies those selected scalar edge values and does not apply the full Young
and angular coefficient map returned by ``phi_motif_coupling_report``. That
report establishes label and carrier provenance for the requested motif; it
does not certify that these averaged columns form a full rotation multiplet
or a nontrivial Young image. Use the route only within its tested scalar
scope. The ordered rank-eight ``Phi`` path below uses a separate compiled
physical contraction and returns all 14 tableau by five magnetic outputs.

Use ``Basis(source="bar_phi")`` with a cutoff, ``PhiSlotChannel`` objects, and
``PhiMotifSpec`` templates. The :doc:`quickstart` constructs a pair and a
three-vertex star on a manufactured H3 fixture. It fits energies, forces, and
ASE stress, writes a ``.phi.pt`` bundle, reloads it, and evaluates through an
ASE calculator. ``motif_family``, ``edge_cutoff``, ``periodic_image_mode``, and
``normalize_motif_features`` are explicit basis controls. The label dictionary
records the motif name, template, channel list, slot orbit, normalization, and
compiler plan.

The compact Phi evaluator uses PyTorch. It has no per-species reference-energy
offset or LAMMPS export schema. Phi stress rows require a positive cell volume.
Position forces and fixed-fractional-position cell gradients use Torch
autograd through the selected fixed-cell motif geometry. Neighbor and image
selection is discrete, so this derivative statement applies while that
selection stays fixed; it does not supply an analytic derivative of a changing
neighbor list or establish a general rank-eight ordered-cluster derivative.
Select it explicitly with ``model.ase_calculator(backend="pytorch")``; the
full backend matrix is in :doc:`evaluators`.
``LinearModel.read`` accepts the saved ``bar_phi`` branch and restores its
fixed-feature evaluator; load ``.phi.pt`` only from a trusted source because it
is a Torch artifact. The example exercises finite periodic geometry; there is
no native C++ or LAMMPS evaluator for this branch.

For a four-vertex depth-two tree, use
``examples/quickstart/phi_depth2.py``. Its edges ``(0,1)``, ``(1,2)``,
and ``(1,3)`` specify the motif directly; the script builds and evaluates
one descriptor without fitting a model. The motif's vertex count and
slot-channel pattern remain visible in the editable input.

.. literalinclude:: ../examples/quickstart/phi_depth2.py
   :language: python

Ordered rank-eight ``Phi`` star (experimental)
-----------------------------------------------

``Basis.from_config`` accepts the explicit ordered-star configuration shown
below. ``Basis.create_cluster`` takes one center and eight distinct
``(atom_index, integer_cell_shift)`` occurrences, with four occurrences using
the first radial channel and four using the second. It preserves image identity
and their order. The compiler counts the complete 18-dimensional multiplicity
space for this fixed content and parent ``(4,4), L=2``. This example selects
one valid coordinate: local Young ``((4),(4))`` with block angular path
``(0,2)``. Its output has shape ``(1, 14, 5)`` for selected path, parent Young
tableau, and magnetic component. It does not pool clusters or provide model
fitting, forces, stress, or a native evaluator.

The public call uses a fixed factor order: its first four records use radial
index zero and its last four use radial index one. Moving a neighbor across
that boundary changes its radial factor. The compiler's separate slot test
checks the formal permutation action when factor identities move with slots.

The rank-eight Young map is numeric. Its rank gap, generator residuals,
projector, and an independently constructed subgroup-invariant direction are
checked; an exact symbolic rank-eight projector comparison has not been run.

.. literalinclude:: ../examples/quickstart/ordered_phi_star.py
   :language: python
