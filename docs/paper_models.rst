Linear models used in the paper
===============================

The source archive includes the six-element fixed-feature comparison of
ordinary ACE/PACE and tagged YE3T described in
`arXiv:2609.31895v1, section IV.3
<https://arxiv.org/abs/2609.31895>`_. Li, Mo, Cu, Ni, Si, and Ge each
have the matched 127-feature models. The archive also contains Ni's additional
model-size tiers. The model bytes and LAMMPS inputs come from
the separate ``ye3t-lammps`` deployment repository.

The archive layout is:

.. code-block:: text

   examples/publication/cost_comparison/
     reproduce_ni_rmse.py        pinned six-model Ni accuracy replay
     refit_paper_ni.py            standardized selected Ni-127 refit
     portable_models/            selected 60/127/149-column Ni bundles
     lammps/<element>/           model files, inputs, reference logs
   examples/data/mlearn/         fixed data snapshot and split records
   tools/publication/           historical search and screening pipelines

Install the ``ye3t-methods[examples]`` wheel and a compatible ``ye3t``
package, then run from the extracted source archive:

.. code-block:: bash

   python tools/publication/cost_comparison/verify_models.py
   python tools/publication/cost_comparison/run.py --stage preflight --systems Li
   python tools/publication/cost_comparison/run.py --stage prepare --systems Li

For the pinned Ni accuracy comparison, use the default native installation
and run the complete 31-frame saved-model replay:

.. code-block:: bash

   python examples/publication/cost_comparison/reproduce_ni_rmse.py

This command reports energy RMSE in eV/atom and force RMSE in eV/Å for the
60-, 127-, and 149-feature ordinary ACE and tagged YE3T models. It checks the
dataset, split, model, and manifest hashes, adds each saved ZBL overlay, and
writes full-precision JSON. The paper example README gives the six reference
values and editable dataset and output paths. It replays saved models;
it does not refit them or measure their LAMMPS speed.

For a new fit using the exact selected Ni-127 basis, run the long ASE workflow:

.. code-block:: bash

   python examples/publication/cost_comparison/refit_paper_ni.py

The script's visible seven-section config fixes the source bundle hash,
published 263-frame training partition, ZBL reference, and selected paper
hyperparameters. It saves a new Torch ASE artifact and evaluates all 31
held-out frames. Change the source archive, hash, feature count, and frozen
selected fit hyperparameters to use the saved 60- or 149-column basis.
The completed Ni-127 run gave 0.000677664 eV/atom energy RMSE and
0.039417111 eV/Å force RMSE, close to the saved model's 0.0006774455
and 0.0394175951. The selected basis and source are fixed; catalogue
selection and hyperparameter search are separate workflows. The refit archive
requires a validated native plan for its coefficients before LAMMPS AUTO
evaluation. The fit is ill-conditioned, so compare predictions and RMSE rather
than raw coefficient bytes; the publication example README records the
numerical details.

The prefit LAMMPS artifacts can be used without rerunning the fitting workflow.
After building LAMMPS with ML-YE3T and, for the independent control, ML-PACE:

.. code-block:: bash

   cd examples/publication/cost_comparison/lammps/Li
   lmp -in in.li_pace_product_127
   lmp -in in.li_ye3t_mixed_127

ASE evaluation of the paper models
----------------------------------

Start with ``examples/quickstart/paper_ni_60.py`` for both 60-feature Ni fits.
Its visible config selects the saved density YACE model, tagged portable basis
and companion native model, and the ASE geometry. It prints energies, forces
and tagged descriptor shapes ``(32, 60)`` and ``(64, 60)``. The saved density
YACE interface provides energy, forces and stress; this example uses the
portable tagged archive to obtain ordered descriptor columns. The portable
archive's descriptor evaluation uses Torch; the companion model uses C++ for
energy and forces. Both saved potentials include their fitted ZBL reference.

.. code-block:: bash

   python examples/quickstart/paper_ni_60.py

.. literalinclude:: ../examples/quickstart/paper_ni_60.py
   :language: python

Both the exact tagged-plus-ACE composite and the independent ACE control run
in ASE through the C++ library built during the default source installation.
With ``ye3t`` already installed, run the editable examples:

.. code-block:: bash

   python -m pip install --no-build-isolation .
   python examples/publication/cost_comparison/ase_native_density.py
   python examples/publication/cost_comparison/ase_native_tagged.py
   python examples/publication/cost_comparison/ase_native_ni.py
   python examples/quickstart/paper_ni_portable_ase.py

The two Li scripts have editable configurations and evaluate the 16-atom Li
reference cell. Tests compare its energy with the saved LAMMPS result. It reports forces, stress,
and the neighbor-list backend. The C++ adapter evaluates the fitted
**residual**. ``LinearModel.read`` restores the tagged composite and its ZBL
reference together. The native YACE example adds its reference explicitly through
ASE ``SumCalculator``.
The ZBL overlay is zero on this example geometry.
The Ni quickstart checks the 32-atom tagged model against its retained
LAMMPS step-zero energy. A separate CMake build is described in
:doc:`evaluators` when the pip-built library is not used.
The portable Ni quickstart uses the source archive's vetted Ni-127
``.ye3t`` bundle through ``LinearModel.read``. It exposes 127 ordered
``Basis.create`` rows and evaluates the complete ordinary, tagged, and ZBL
energy with the CPU Torch ASE calculator. The bundle SHA-256 is
``a57647406108a71273794e7786252147c93830d8954d8c44d9d7e813cb6502a2``;
the source-archive model is the same bytes as the bounded conversion study.
The installed wheel supplies the reader, while the source archive supplies
the reference model and example. The native composite evaluator supports its
corresponding artifacts.
The ordinary ACE control is supplied as ``.yace`` and loads through
``YE3TYACENativeCalculator.from_artifact``. ``LinearModel.read`` reads compact
Torch ``.pt`` bundles, not ``.yace`` files. See :doc:`evaluators` for the
compact Torch paths and exact artifact limitations.

The inputs supply the ZBL overlay used in training. Read the
``examples/publication/cost_comparison/README.md`` and the local LAMMPS
README before running other systems or numerical-difference/NVE checks.
The original full paper study and LAMMPS replay require a suitable LAMMPS build
and more time than the quickstart tests. The reference logs document recorded
runs. Reproduction requires executing the model and comparison inputs.

The published elemental models use one-hot chemical channels. See
:doc:`chemical_encoding` for the stable one-hot semantics and the
in-memory fixed-embedding demonstration. :doc:`basis_inputs` explains how
the compact paper schedules become compiler-validated basis coordinates.

Fit and run the 60-feature Ni models
------------------------------------

The source distribution includes two independent pairs of scripts in
``examples/publication/cost_comparison``:

* ``paper_ni_60_density_01_prep_train.py`` and
  ``paper_ni_60_density_02_run.py``;
* ``paper_ni_60_tagged_01_prep_train.py`` and
  ``paper_ni_60_tagged_02_run.py``.

Each training script shows its complete editable config, loads the paper's
compiler catalogue and selected coordinates, evaluates descriptor and gradient
rows, fits all 263 development structures, and scores the separate 31-frame
evaluation set. The density model has 60 density coordinates. The tagged
model combines 15 density coordinates with 45 tagged coordinates.
The fitted objective, radial functions and ZBL reference match the paper.

The public workflow is:

.. code-block:: python

   from ye3t_methods import LinearModel
   from ye3t_methods.studies.ni_linear import load_paper_basis, load_refit_structures

   training, validation = load_refit_structures(config)
   basis = load_paper_basis(config)
   features, gradient = basis.create(training[0], derivatives="summed")
   model = LinearModel(basis).fit(training, config=config)
   model.write("ni60_refit.ye3t")
   model.export_lammps("ni60_lammps")

``load_paper_basis`` restores the representation and coordinate choices from
the configured compiler archive. These examples reproduce a specific paper
selection. For a new catalogue, use ``YE3TRepresentation.from_config`` and
``Basis.from_config`` as shown in :doc:`quickstart`.

The export directory contains the potential, AUTO execution data, ZBL settings
and ``pair_coeff.in``. Choose a new export directory for each fit. Density
AUTO data uses a separate manifest and execution-plan files; tagged AUTO data
is embedded in its component model. Run LAMMPS from the export directory and
include ``pair_coeff.in`` after creating the atom types. LAMMPS requires the
YE3T pair style.

The second script loads the fitted ``.ye3t`` model as an ASE calculator and
runs 100 NVE steps, saving a trajectory and energy log. Its visible config
selects ``native_cpu``; ``torch`` is available for comparison. Both evaluators
include the fitted intercept and the ZBL reference. Set
``YE3T_WORKFLOW_ROOT`` to choose the output location, and edit the dataset and
model paths in the scripts when using an extracted example archive.

Shared caches and repeated fits
-------------------------------

The paper archive contains its selected catalogue and coupling coefficients.
Loading it restores those coordinates. General basis construction also uses
YE3T's shared per-user coefficient cache at ``~/.cache/ye3t``. Set
``YE3T_CACHE_DIR`` to choose another directory; ``XDG_CACHE_HOME`` is respected.

The paper refit stores target-free descriptor and force rows in the shared
application cache, normally ``~/.cache/ye3t/applications``. Separate Python
processes reuse these rows. Atomic numbers, positions, cell, periodic flags,
source archive and selected coordinates determine reuse. Changing target
energies, forces, or fitting weights reuses the rows and rebuilds the fit.

.. code-block:: python

   config["runtime"]["cache"] = {"mode": "auto", "directory": None}
   model = LinearModel(basis).fit(training, config=config)

``auto`` reads or populates the shared cache. ``off`` computes fresh rows
without persistent I/O; ``read_only`` requires existing rows. Set ``directory``
for a study-specific cache. ``YE3T_ACE_CACHE_DIR`` sets the application cache
root; otherwise ``YE3T_CACHE_DIR/applications`` is used when configured.
Cached arrays are checked for corruption on load. Cache identities remain
internal to normal use.

The Torch fitting route uses analytic radial/angular derivatives and the
installed YE3T C++ monomial kernel for tagged center derivatives when available.
The product derivatives are combined at each center before contraction with
neighbor-edge derivatives. This avoids expanding each polynomial term over
every edge. The ``native_cpu`` ASE calculator uses the complete C++ model
evaluator; fitting and calculator dispatch are separate settings.
