Choosing a linear evaluator
===========================

``Basis(backend=...)`` controls descriptor construction and fitting.
``LinearModel.ase_calculator(backend=...)`` selects evaluation of a saved or
fitted model. These are separate choices. The ordinary ACE low-level
``source_backend`` setting selects a one-neighbor source kernel; it does not
turn the whole ordinary ASE calculator into the standalone C++ runtime.

.. list-table:: Supported compact ASE choices
   :header-rows: 1

   * - Basis source
     - ``Basis`` backend
     - ``model.ase_calculator`` backend
     - Notes
   * - ``density``
     - ``pytorch`` by default
     - ``pytorch`` by default
     - Torch ASE energy, forces, and stress. Choose ``force_method="analytic_factorized"``
       for the explicit analytic force path; the default is ``autograd``.
   * - ``tagged_cauchy_image``
     - ``auto`` by default; ``reference`` or ``native`` explicitly
     - ``reference`` by default; ``native_polynomial`` or ``native_cpu`` explicitly
     - ``reference`` uses the Torch reference evaluator. ``native_polynomial``
       accelerates only the polynomial contraction. ``native_cpu`` uses the
       separate ``ye3t-lammps`` C++ library for the model evaluation.
   * - ``bar_phi``
     - ``pytorch`` only
     - ``pytorch`` or its ``reference`` alias
     - Torch ASE energy, forces, and stress; no C++ or LAMMPS export.

The ordinary low-level contraction selector also accepts ``auto``, ``triton``,
and ``openequivariance``. Those names select internal contraction paths where
supported; this release qualifies the ordinary compact ASE path with
``pytorch``. Passing ``backend="native"`` to the ordinary calculator is an
alias for ``pytorch`` in the core contraction selector, not a request for a
standalone C++ calculator. ``backend="native_cpu"`` is the tagged ASE option,
not an ordinary density option. Strict requests for unsupported low-level
paths can fail rather than silently using Torch.

Ordinary and Phi examples
-------------------------

The runnable :doc:`quickstart` scripts keep the evaluator in their editable
``config["runtime"]`` blocks. For an ordinary saved ``.pt`` model:

.. code-block:: python

   from ye3t_methods import LinearModel

   model = LinearModel.read("ordinary.pt")
   atoms.calc = model.ase_calculator(
       backend="pytorch", force_method="analytic_factorized",
   )
   energy, forces = atoms.get_potential_energy(), atoms.get_forces()

``force_method="autograd"`` remains available. The analytic force choice
changes evaluation, not fitted coefficients or descriptor labels. An explicit
``bar_phi`` model uses ``model.ase_calculator(backend="pytorch")``.

Tagged reference and native CPU
-------------------------------

For a compact tagged ``.ye3t.json`` artifact, choose the evaluator when
constructing the ASE calculator:

.. code-block:: python

   from ye3t_methods import LinearModel

   model = LinearModel.read("tagged.ye3t.json")
   atoms.calc = model.ase_calculator(backend="reference")
   reference_forces = atoms.get_forces()

   atoms.calc = model.ase_calculator(
       backend="native_cpu", native_library="./libye3t_tagged_c_api.so",
       execution_policy="direct",
   )
   native_forces = atoms.get_forces()

Build the optional C ABI library from ``ye3t-lammps`` with
``ML_YE3T_BUILD_TAGGED_C_API=ON``. Pass its path with ``native_library`` or set
``YE3T_TAGGED_C_API_LIBRARY``. ``execution_policy`` is ``direct`` or ``auto``;
``auto`` calibrates schedules when the model is opened and takes longer to
initialize. The native model and its schedules stay resident in the calculator.
It reuses neighbor topology while atoms move within the 0.3 Å skin.

The native ASE adapter uses SciPy's cKDTree for eligible orthorhombic or
nonperiodic cells. For other cells it uses ``matscipy.neighbours`` when
installed and otherwise falls back to ASE's neighbor list. ``matscipy`` is an
optional speedup; it changes neighbor construction, not the model or force
formula. The calculator's ``native_runtime.last_neighbor_backend`` reports the
path actually used.
Install it with ``python -m pip install '.[neighbors]'`` from the methods
source checkout after installing the local ``ye3t`` dependency.

Paper deployment artifacts
--------------------------

The exact paper tagged model is a composite of an ordinary ``.yace`` backbone
and a tagged correction. Load it directly with the native ASE adapter:

.. code-block:: python

   from ye3t_ace.tagged_cauchy_image import YE3TTaggedCauchyCalculator

   atoms.calc = YE3TTaggedCauchyCalculator.from_artifact(
       "lammps/Li/models/ye3t_tagged_127/model.ye3t.json",
       native_library="./libye3t_tagged_c_api.so",
       execution_policy="direct",
   )

The standalone adapter evaluates the **linear residual model**. For the full
paper potential, read the ZBL settings from its model manifest and combine
the two ASE calculators:

.. code-block:: python

   from ase.calculators.mixing import SumCalculator
   from ye3t_ace.reference_potentials import YE3TZBLCalculator

   linear = YE3TTaggedCauchyCalculator.from_artifact(
       "lammps/Li/models/ye3t_tagged_127/model.ye3t.json",
       native_library="./libye3t_tagged_c_api.so",
   )
   zbl = YE3TZBLCalculator.from_model_manifest(
       "lammps/Li/models/ye3t_tagged_127/model_manifest.json"
   )
   atoms.calc = SumCalculator([linear, zbl])

Run ``examples/publication/cost_comparison/ase_native_tagged.py`` from the
source archive for the exact Li step-zero validation structure. The ZBL
reference contributes zero on that cell. On four close-contact Li pairs,
the ASE ZBL energy and force differences from LAMMPS were at most
``5.6e-17`` eV and ``8.9e-16`` eV/Å, respectively.

The paper's ACE control is shipped as ``.yace`` for LAMMPS/PACE.
``LinearModel.read`` accepts ordinary ``.pt`` bundles, not ``.yace``. It also
does not read the paper's composite tagged schema as a compact tagged model.
The archived paper files therefore do not currently provide a single PyTorch
ASE calculator for both exact published models. Newly fitted compact density,
tagged, and Phi models do retain the Torch-backed ASE routes above.
