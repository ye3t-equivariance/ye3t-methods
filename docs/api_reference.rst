Public linear API
=================

For a new configured linear fit, import ``YE3TRepresentation`` from ``ye3t``
and ``Basis, LinearModel`` from ``ye3t_methods``. Construct the representation,
then call ``Basis.from_config(..., representation=representation,
runtime=...)`` and ``LinearModel(basis).fit(..., config=...)``. The basis is
the descriptor object in this flow. The same seven-section config supplies
representation, basis, runtime, fit, targets, and validation settings.
Optional ``metadata.training_structures``, ``metadata.evaluation_structure``,
and ``metadata.output_path`` are visible workflow paths used by examples;
they are validated as strings but excluded from fitted-model identity.
For scalar tagged catalogues, optional
``basis.catalogue.angular_patterns_by_rank`` restricts the requested input
angular tuples before compiler counting and materialization. The ranks-one-through-eight
tagged quickstart uses it to select its intended input angular patterns.

``ye3t_methods.FeatureLabel`` represents an actual fitted descriptor column.
Maintained low-level modules live under ``ye3t_methods.atomistic``. The
``ye3t_ace`` module path is a read-only import shim for historical saved
Torch models; versioned artifact schema identifiers remain readable.

.. list-table::
   :header-rows: 1

   * - Object or method
     - Purpose
   * - ``Basis(elements=..., source=..., cutoff=..., ...)``
     - Construct density, tagged physical-image, or explicit ``bar_phi`` columns.
   * - ``Basis.from_config(basis, representation=rep, runtime=runtime)``
     - Resolve physical channels and preview compiler counts. Standalone
       shifted-Jacobi tagged scalar requests at one or more selected ranks
       with explicit or one-hot
       chemistry, including multiple species,
       and configured PACE ordinary scalar or full-multiplet requests with
       supported chemistry, can materialize descriptor rows on CPU. Scalar
       configurations also accept
       ``native_cpu`` for a fitted ASE model. Scalar tagged descriptor rows
       use the native polynomial evaluator for ``native_cpu`` and the available
       production evaluator for ``auto``. The tagged route accepts
       ``reference`` explicitly. The capability report states the selected
       model evaluator and neighbor policy; unsupported source and runtime
       combinations reject before compilation. Supported tagged configurations
       normalize explicit/one-hot species to compiler lexical order in the
       frozen resolution; their non-angular η records list radial degree before
       species, while angular degree remains a separate compiler coordinate.
       Shifted-Jacobi tagged and PACE ordinary parents with ``L>0`` use
       compiler-selected full-multiplet routes on the CPU reference evaluator.
   * - ``basis.labels`` / ``basis.describe(index, format="text")``
     - Inspect compiler-accepted columns; ``format="latex"`` is also available.
   * - ``basis.create(atoms)``
     - Return finite real ``float64`` NumPy rows in catalogue order. Scalar
       bases use ``(n_atoms, n_features)``; full-multiplet ``L>0``
       basis uses ``(n_atoms, n_multiplets, 2*L+1)``.
   * - ``basis.create_many(structures)``
     - Return one ordered descriptor array per structure.
   * - ``LinearModel(basis, reference_energies=None)``
     - Create the fixed-feature scalar readout; Phi rejects reference offsets.
   * - ``model.fit(structures, regularization=..., energy_weight=...,
       force_weight=..., stress_weight=...)``
     - Fit to precomputed ASE energy, force, and supported stress labels.
   * - ``model.fit(structures, config=seven_section_config)``
     - Fit a configured scalar model or a selected density or tagged
       ``L>0`` per-atom full-multiplet model. Full-multiplet fits accept
       real-tesseral targets; Cartesian vectors at ``L=1`` and traceless
       symmetric tensors at ``L=2`` are also supported.
   * - ``model.predict(atoms, uncertainty=False)``
     - Return per-atom full-multiplet real-tesseral means; an ARD fit also
       returns full component covariance when ``uncertainty=True``.
   * - ``model.write(path)`` / ``LinearModel.read(path)``
     - Save or restore the source-specific artifact. Full-multiplet density
       and tagged models use hash-bound ``.ye3t.json`` files with embedded
       compiled coupling coefficients and selected coordinate IDs.
   * - ``model.ase_calculator(evaluator=None, neighbors=None)``
     - Build an ASE calculator for a fitted or loaded model; ``backend`` is a
       compatibility alias for ``evaluator``. An omitted neighbor policy uses
       the in-memory config when available and otherwise defaults to ``auto``.
   * - ``model.export_lammps(path)``
     - Export supported density or tagged formats with their compiler metadata.
   * - ``label.as_dict()`` / ``label.latex()``
     - Inspect structured identity or a short mathematical display.

The configured PACE ordinary route materializes the complete compiler-counted
catalogue selected by that config. It does not infer the paper Ni-127 fit
selection; that order and its coefficients belong to the hashed model archive.
Configured PACE model bundles retain the zero-based public radial labels across
``LinearModel.write`` and ``LinearModel.read``. For an in-memory model from a
config selecting ``native_cpu``, ``auto`` uses the strict YACE native route
and inherits its configured neighbor policy. For a config selecting ``torch``,
or for a reloaded compact model, ``auto`` selects Torch; a reloaded model can
still request ``native_cpu`` explicitly. Older density bundles retain their
original label display convention and neighbor-selection behavior.

``LinearModel.read`` also accepts a single-file ``.ye3t`` archive with schema
``ye3t_legacy_compat_archive_v1``. This compatibility format embeds and checks
a legacy composite, ordinary YACE model, tagged correction, portfolio upgrade,
and model manifest for internal consistency. Supply a trusted archive or check
its SHA-256 against an independently stored value; embedded hashes check
internal consistency without authenticating the model's origin. The loaded
calculator keeps the necessary model data in memory. It retains the legacy
composite source and can evaluate energy, forces, and stress with the supported
native evaluator. It does not expose compiler labels or the coupling-array
execution interface of the separate bounded Ni archive reader.

A bounded ``ye3t_linear_portable_bundle_v1`` reader accepts the four vetted
Ni 60/127/149/196 scalar archives by exact archive SHA-256. It exposes all
ordered ordinary and tagged ``Basis.create`` rows and evaluates the embedded
ordinary, tagged, and ZBL terms with ``evaluator="torch"`` on CPU. Loading
and repeated ASE evaluation use saved coupling products without recompiling.
The reader rejects other model identities or changed archive bytes. This
bounded route also prepares the saved or refitted readout for
``evaluator="native_cpu"`` and exports a LAMMPS directory with
``model.export_lammps(path)``. See :doc:`paper_models` for the complete
fit, export and ASE-MD examples. The reader remains specific to the
supported Ni compiler archives.

The low-level descriptor-first flow is
``YE3TRepresentation -> YE3TDescriptors -> YE3TModel``. It supports historical
source presets, fixed Young descriptor sets, manual ACE coordinate requests,
and specialized runtime controls. Use the configured ``Basis`` workflow in
:doc:`shell_linear` for shell-density fitting. The low-level flow accepts compiler
plans and validated labels from the ``ye3t`` compiler.
Its ``YE3TRepresentation`` is the legacy selector also re-exported by
``ye3t_methods``; it is distinct from ``ye3t.YE3TRepresentation`` and is not
an input to ``Basis.from_config``.

See :doc:`density`, :doc:`tagged_basis`, :doc:`phi_basis`, and
:doc:`role_density` for source-specific arguments and tested use.
:doc:`evaluators` gives the exact ASE backend choices and native-library setup.
:doc:`basis_inputs` describes compact compiler requests, and
:doc:`chemical_encoding` states which species encodings can be deployed.

Model evaluation against labels
-------------------------------

``model.score(structures, config=config)`` compares predictions with ASE labels
using the configured ``targets``. Energy RMSE is measured in eV/atom, with
equal structure weight. Force RMSE averages all Cartesian atom components in
eV/Angstrom. Optional stress RMSE averages the six ASE Voigt components in
eV/Angstrom cubed. The method leaves the reference calculators and labels
attached to the supplied structures unchanged.

Atomic reference energies
-------------------------

``model.reference_energy.per_species_E0_eV`` provides fixed energies in eV for
every model species. With ``fit_E0=False``, these remain fixed and fitting has
no additional atomic-offset columns. With ``fit_E0=True``, fitting includes
per-species corrections to those energies. This requires positive energy
weight and independent composition columns in the fitting structures.
The choices apply to configured density, tagged, combined and shell-density
scalar fits. Saved models include the references needed by their ASE evaluator.

Saved-model sizes
-----------------

Tagged models use compact JSON and retain their compiler data and hashes. The
Python JSON model reader and native tagged reader accept files up to 512 MiB.
Combined ``.ye3t``
archives have a 64 MiB limit per member and a 256 MiB total limit. These are
reader limits on serialized files. Combining larger compiler payloads requires
further persistence work; the rank-one-through-eight combined example is
tested within these bounds.
