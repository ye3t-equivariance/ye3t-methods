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
     config.json                 editable paper fitting settings
     tagged_components.json      fixed-channel tagged requests
     systems/                    six editable element records
     run.py and workflow/        fitting and validation stages
     reproduce_ni_rmse.py        pinned six-model Ni accuracy replay
     refit_paper_ni.py            standardized selected Ni-127 refit
     portable_models/            selected 60/127/149-column Ni bundles
     lammps/<element>/           model files, inputs, reference logs
   examples/data/mlearn/         fixed data snapshot and split records

Install the ``ye3t-methods[examples]`` wheel and a compatible ``ye3t``
package, then run from the extracted source archive:

.. code-block:: bash

   python examples/publication/cost_comparison/verify_models.py
   python examples/publication/cost_comparison/run.py --stage preflight --systems Li
   python examples/publication/cost_comparison/run.py --stage prepare --systems Li

For the pinned Ni accuracy comparison, use the default native installation
and run the complete 31-frame saved-model replay:

.. code-block:: bash

   python examples/publication/cost_comparison/reproduce_ni_rmse.py

This command reports energy RMSE in eV/atom and force RMSE in eV/Å for the
60-, 127-, and 149-feature ordinary ACE and tagged YE3T models. It checks the
dataset, split, model, and manifest hashes, adds each saved ZBL overlay, and
writes full-precision JSON. The paper example README gives the six reference
values and ``--dataset-root``/``--output`` options. It replays saved models;
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

Both the exact tagged-plus-ACE composite and the independent ACE control run
in ASE through the C++ library built during the default source installation.
With ``ye3t`` already installed, run the editable examples:

.. code-block:: bash

   python -m pip install --no-build-isolation .
   python examples/publication/cost_comparison/ase_native_density.py
   python examples/publication/cost_comparison/ase_native_tagged.py
   python examples/publication/cost_comparison/ase_native_ni.py
   python examples/quickstart/paper_ni_portable_ase.py

The two Li scripts have editable configurations and validate the 16-atom Li
step-zero energy against the retained LAMMPS log. It reports forces, stress,
and the neighbor-list backend. The C++ adapter evaluates the fitted
**residual**; each script combines it with
``YE3TZBLCalculator.from_model_manifest`` through ASE ``SumCalculator``.
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
