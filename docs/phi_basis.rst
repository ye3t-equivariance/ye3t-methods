Explicit motif ``bar_phi`` basis
================================

Ordinary density sums individual neighbor contributions before forming
products. ``bar_phi`` instead selects an explicit cluster motif, assigns
channels to its slots, and evaluates the motif over the local geometry. Its
``N`` counts motif vertices; it must not automatically be read as the physical
body order of an ordinary ACE density product. The compiler supplies the
fixed motif coupling plan and validates the scalar target.

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
Select it explicitly with ``model.ase_calculator(backend="pytorch")``; the
full backend matrix is in :doc:`evaluators`.
``LinearModel.read`` accepts the saved ``bar_phi`` branch and restores its
fixed-feature evaluator; load ``.phi.pt`` only from a trusted source because it
is a Torch artifact. For the verified operations and periodic-image checks, see
``RELEASE_VALIDATION.md``.

For a four-vertex depth-two tree, use
``examples/quickstart/phi_depth2.py``. Its edges ``(0,1)``, ``(1,2)``,
and ``(1,3)`` specify the motif directly; the script builds and evaluates
one descriptor without fitting a model. The motif's vertex count and
slot-channel pattern remain visible in the editable input.

.. literalinclude:: ../examples/quickstart/phi_depth2.py
   :language: python
