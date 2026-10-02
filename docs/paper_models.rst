Linear models used in the paper
===============================

The source archive includes the six-element fixed-feature comparison of
ordinary ACE/PACE and tagged YE3T described in
`arXiv:2609.31895v1, section IV.3
<https://arxiv.org/abs/2609.31895>`_. Li, Mo, Cu, Ni, Si, and Ge each
have the matched 127-feature models. The archive also retains Ni's additional
model-size tiers. These are the promoted model bytes and LAMMPS inputs from
the separate ``ye3t-lammps`` deployment repository.

The archive layout is:

.. code-block:: text

   examples/publication/cost_comparison/
     config.json                 editable paper fitting settings
     tagged_components.json      fixed-channel tagged requests
     systems/                    six editable element records
     run.py and workflow/        fitting and validation stages
     lammps/<element>/           model files, inputs, reference logs
   examples/data/mlearn/         fixed data snapshot and split records

Install the ``ye3t-methods[examples]`` wheel and a compatible ``ye3t``
package, then run from the extracted source archive:

.. code-block:: bash

   python examples/publication/cost_comparison/verify_models.py
   python examples/publication/cost_comparison/run.py --stage preflight --systems Li
   python examples/publication/cost_comparison/run.py --stage prepare --systems Li

The prefit LAMMPS artifacts can be used without rerunning the fitting workflow.
After building LAMMPS with ML-YE3T and, for the independent control, ML-PACE:

.. code-block:: bash

   cd examples/publication/cost_comparison/lammps/Li
   lmp -in in.li_pace_product_127
   lmp -in in.li_ye3t_mixed_127

ASE evaluation of the paper models
----------------------------------

Both the exact tagged-plus-ACE composite and the independent ACE control run
in ASE through the C++ library bundled in this source archive. Build it with
a sibling ``ye3t`` source checkout, then run the two editable examples:

.. code-block:: bash

   cmake -S native -B ../build-ye3t-methods-native \
     -DYE3T_RUNTIME_SOURCE=../ye3t
   cmake --build ../build-ye3t-methods-native --target ye3t_tagged_c_api --parallel
   export YE3T_TAGGED_C_API_LIBRARY="$PWD/../build-ye3t-methods-native/libye3t_tagged_c_api.so"
   python examples/publication/cost_comparison/ase_native_density.py
   python examples/publication/cost_comparison/ase_native_tagged.py

Each script has an editable configuration and validates the 16-atom Li
step-zero energy against the retained LAMMPS log. It reports forces, stress,
and the neighbor-list backend. The C++ adapter evaluates the fitted
**residual**; each script combines it with
``YE3TZBLCalculator.from_model_manifest`` through ASE ``SumCalculator``.
The ZBL overlay is zero on this example geometry.
The ordinary ACE control is supplied as ``.yace`` and loads through
``YE3TYACENativeCalculator.from_artifact``. ``LinearModel.read`` reads compact
Torch ``.pt`` bundles, not ``.yace`` files. See :doc:`evaluators` for the
compact Torch paths and exact artifact limitations.

The inputs supply the ZBL overlay used in training. Read the
``examples/publication/cost_comparison/README.md`` and the local LAMMPS
README before running other systems or numerical-difference/NVE checks.
Full paper refitting and LAMMPS replay require a suitable LAMMPS build and
more time than the quickstart tests. The retained logs are previous promoted
evidence; copying them into the source archive is not a fresh validation.

The published elemental models use one-hot chemical channels. See
:doc:`chemical_encoding` for the stable one-hot semantics and the
in-memory fixed-embedding demonstration. :doc:`basis_inputs` explains how
the compact paper schedules become compiler-validated basis coordinates.
