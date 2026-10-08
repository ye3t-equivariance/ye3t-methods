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
     - ASE energy/forces; low-level native and strict PACE routes
   * - Density or selected tagged ``L>0`` per-atom multiplets
     - versioned, hash-bound ``.ye3t.json`` with saved all-M coefficients
     - ASE real-tesseral mean and optional ARD component covariance
   * - Tagged physical image
     - versioned, hash-bound ``.ye3t.json``
     - ASE energy/forces/stress; tagged native/LAMMPS consumer where supported
   * - Explicit ``bar_phi``
     - ``.phi.pt`` Torch bundle
     - ASE reference energy/forces/stress
   * - Configured density plus tagged scalar
     - verified indexed ``.ye3t`` JSON/NPZ bundle
     - Torch and native CPU ASE energy/forces/stress, with one species E0 map
   * - Vetted Ni portable scalar composite
     - single-file ``.ye3t`` with ordered labels and saved coupling products
     - CPU Torch ASE energy/forces/stress with its embedded ZBL specification

``LinearModel.read(path)`` restores the corresponding fitted model and label
order for these saved forms. Give an explicit file path when multiple
suffixes could match one stem. It also reads a trusted single-file ``.ye3t``
legacy compatibility archive, or a colocated legacy composite
``model.ye3t.json`` with its manifest and components. Those legacy routes
retain the model for inference but do not expose compiler labels. The paper
ACE ``.yace`` control uses ``YE3TYACENativeCalculator.from_artifact``;
``LinearModel.read`` does not read a standalone ``.yace`` file. See
:doc:`evaluators` for the
supported ASE backends and examples.
Torch artifacts use Python deserialization and must come from a trusted source.
The tagged and density full-M JSON loaders check their versioned schemas,
coordinate conventions, and hashes. Density compiler materialization and
the density reader check coefficient polynomials against the rotation
generator and real-form relation; they reject coefficients that break these
symmetries without recompiling them on load.
Internal archive hashes detect inconsistent bytes but do not authenticate an
archive's origin. A consistently rewritten, symmetry-valid coefficient basis
can change predictions while passing these checks. Compare the artifact digest
with an independently retained value when exact compiler coordinates or origin
matter. A loaded nonlegacy model can evaluate and describe its saved
features. The indexed combined and bounded Ni portable bundles also retain
standalone ``basis.create(atoms)`` rows; for older single-component bundles,
construct a new ``Basis`` for that operation.

Tagged full-M writes use the v2 JSON schema. They include a separate,
hash-bound native property plan with compiler schedules and the fitted
species/feature coefficient order. For two tagged occurrences, the plan uses
the compiler's exact one-edge marginalization; writing fails if its retained
feature coordinates differ from the fitted ones. The Python reader still
accepts v1 tagged full-M files. The separate ``ye3t-lammps``
``compute ye3t/property/atom`` reads v2 tagged models on the CPU and returns
the fitted per-atom mean in the saved real-tesseral order. The compute does
not return forces, stress, or posterior covariance. The native consumer
currently accepts ``L=1`` odd-parity and ``L=2`` even-parity models; Python
can write tagged ``L=2`` odd-parity artifacts, but native execution of those
artifacts remains unvalidated and is rejected.
An experimental ``ye3t/property/atom/kk/device`` consumer for tagged v2
models has matched CPU output in single-rank and two-rank CUDA/Kokkos tests.
Its property kernels use device data; final per-atom values are copied to
host for LAMMPS consumers. Density v2 artifacts are rejected by that device
consumer.

Density full-M writes use a v2 JSON schema with a native property
plan. It binds the saved all-M coefficient blocks, PACE site source,
coordinate order, real-form transform, and fitted readout. The Python reader
still accepts v1 density full-M files. The separate ``ye3t-lammps`` CPU
``compute ye3t/property/atom`` reads v2 density models with explicit delta
chemistry and identity PACE ChebExpCos radial channels. Its focused L1/L2
single-rank and two-rank regression cases pass. Density device execution and
broader source conventions have not passed native validation.
The runnable :doc:`quickstart` vector example writes a density v2 model,
an ASE structure in LAMMPS data format, and ``in.property``. It exercises
``model.write`` followed by the CPU property compute on a periodic Cu cell.
The Python prediction file uses the saved real-tesseral component order.

The indexed combined writer requires configured PACE density and
shifted-Jacobi tagged scalar components, built separately with ``Basis.combine``
or through one named ``basis.components`` request with component-local sources.
It stores verified source records,
selected ordinary coupling arrays, a validated tagged image, ordered labels,
readout weights, and optional ARD posterior arrays. Its loader verifies all
member hashes and source/compiler bindings, and reconstructs the ordinary
runtime from saved coordinates without recompiling couplings. Explicit Torch
and native CPU ASE evaluators are supported; combined LAMMPS export has no
validated schema yet.

The bounded Ni portable reader accepts only four exact SHA-256-pinned archive
bytes for the 60, 127, 149, and augmented 196 models. It restores ordered
``basis.labels`` and ``basis.create(atoms)`` rows, and its Torch ASE calculator
adds the saved ordinary, tagged, and ZBL energies once. ``model.write`` copies
the verified archive bytes. An explicit native evaluator or LAMMPS export
request rejects because this archive route has no validated native lowering.
The bounded reader accepts only the pinned checkpoint identities listed above.
Other checkpoint formats need an explicit integrity and compatibility check.

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
The compact density ridge, tagged, and Phi fits can consume precomputed ASE
Voigt stress with nonzero ``stress_weight``. The combined scalar fit uses the
same component strain rows and requires a positive full-rank cell. Density
uses analytic descriptor strain rows and requires a positive cell volume. The :doc:`quickstart` Phi
and saved tagged scripts evaluate ASE stress.

YACE and LAMMPS
---------------

``LinearModel.export_lammps(path)`` delegates to the source's supported export:
ordinary density uses strict scalar PACE ``.yace`` lowering; tagged image uses
its native JSON schema; explicit Phi has no LAMMPS schema. YACE export
does not apply to full-M density property models; use ``model.write`` and
``compute ye3t/property/atom`` for the native L1/L2 property route.
Strict YACE export
requires a representable PACE radial/angular source and scalar invariant
channels. The compact density constructor's default explicit radial source
does not satisfy that strict representation and export fails closed. The
low-level ACE API can build an appropriate PACE-compatible source.
Do not treat a readable ``.yace`` descriptor file as proof that a fitted model
matches a LAMMPS-PACE runtime convention.

The ``ye3t_methods.atomistic.ace.yace`` helpers ``YACEFunction``, ``read_yace``,
``read_yace_functions``, and ``write_yace`` handle the normalized descriptor
file representation. A function records its central type ``mu0``, rank,
neighbor types ``mus``, radial indices ``ns``, angular indices ``ls``, magnetic
combinations ``ms_combs``, and coefficients ``ctildes``. The strict exporter
folds fitted linear weights into emitted C-tilde rows after validating the
source and descriptor sector. The strict YACE tests cover positive
and rejected cases.

For a tagged artifact, the separate ``ye3t-lammps`` consumer supplies the
``pair_style ye3t`` implementation. The quickstart's JSON export does not
install a LAMMPS pair style. The CPU examples and model inputs are in
:doc:`paper_models`; check the consumer's documentation for its supported GPU
scope. Any additive reference potential in a scientific workflow must be
configured consistently in fitting and deployment.
