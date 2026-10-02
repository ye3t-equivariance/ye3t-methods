Saved models, stress, and export
================================

Model persistence
-----------------

``LinearModel.write(path)`` chooses a suffix when the path has none:

.. list-table::
   :header-rows: 1

   * - Source
     - Saved form
     - Evaluation
   * - Ordinary density
     - ``.pt`` Torch bundle
     - ASE energy/forces; retained low-level native and strict PACE routes
   * - Tagged physical image
     - versioned, hash-bound ``.ye3t.json``
     - ASE energy/forces/stress; tagged native/LAMMPS consumer where supported
   * - Explicit ``bar_phi``
     - ``.phi.pt`` Torch bundle
     - ASE reference energy/forces/stress

``LinearModel.read(path)`` restores the corresponding fitted model and label
order. Give an explicit file path when multiple suffixes could match one stem.
The paper's tagged-plus-ACE composite uses the separate native
``YE3TTaggedCauchyCalculator.from_artifact`` loader; the paper ACE ``.yace``
control has no ``LinearModel.read`` route. See :doc:`evaluators` for the
supported ASE backends and examples.
Torch artifacts use Python deserialization and must come from a trusted source.
The tagged JSON loader checks its versioned schema and hashes. A loaded model
can evaluate and describe its saved features; construct a new ``Basis`` to
call standalone ``basis.create(atoms)`` on fresh structures.

ASE stress convention
---------------------

Homogeneous strain changes each relative displacement according to its cell
deformation. For an ordinary site-basis contribution,

.. math::

   \frac{\partial \phi_q(\mathbf r_{ij})}{\partial\epsilon_{ab}}
   = \frac{\partial\phi_q}{\partial r_{ij,a}}r_{ij,b}.

The stress adapter symmetrizes the energy strain derivative and divides by
the positive cell volume:

.. math::

   \sigma = \frac{1}{2V}\left(\frac{\partial E}{\partial\epsilon}
      +\frac{\partial E}{\partial\epsilon}^{T}\right).

ASE Voigt order is ``xx, yy, zz, yz, xz, xy``; units are eV/angstrom cubed.
The compact tagged and Phi fits can consume precomputed ASE Voigt stress with
nonzero ``stress_weight``. The compact density fit does not accept stress rows,
although the retained low-level scalar ACE calculator has a stress derivative
route. The :doc:`quickstart` Phi and saved tagged scripts evaluate ASE stress.

YACE and LAMMPS
---------------

``LinearModel.export_lammps(path)`` delegates to the source's supported export:
ordinary density uses strict scalar PACE ``.yace`` lowering; tagged image uses
its native JSON schema; explicit Phi has no LAMMPS schema. Strict YACE export
requires a representable PACE radial/angular source and scalar invariant
channels. The compact density constructor's default explicit radial source
does not satisfy that strict representation and export fails closed. The
retained low-level ACE API can build an appropriate PACE-compatible source.
Do not treat a readable ``.yace`` descriptor file as proof that a fitted model
matches a LAMMPS-PACE runtime convention.

The retained ``ye3t_ace.ace.yace`` helpers ``YACEFunction``, ``read_yace``,
``read_yace_functions``, and ``write_yace`` handle the normalized descriptor
file representation. A function records its central type ``mu0``, rank,
neighbor types ``mus``, radial indices ``ns``, angular indices ``ls``, magnetic
combinations ``ms_combs``, and coefficients ``ctildes``. The strict exporter
folds fitted linear weights into emitted C-tilde rows after validating the
source and descriptor sector. The retained strict YACE tests cover positive
and rejected cases.

For a tagged artifact, the separate ``ye3t-lammps`` consumer supplies the
``pair_style ye3t`` implementation. Its tested CPU and Kokkos scopes are
qualified in ``RELEASE_VALIDATION.md``; the quickstart's JSON export does not
install a LAMMPS pair style. Any additive reference potential in a scientific
workflow must be configured consistently in fitting and deployment.
