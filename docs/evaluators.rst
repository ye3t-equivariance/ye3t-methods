Choosing a linear evaluator
===========================

``Basis(backend=...)`` controls descriptor construction and fitting.
``LinearModel.ase_calculator(backend=...)`` selects evaluation of a saved or
fitted model. These are separate choices. The ordinary ACE low-level
``source_backend`` setting selects a one-neighbor source kernel. Full C++
ordinary ASE evaluation uses ``backend="native_cpu"`` and requires the
model to pass strict YACE export.

.. list-table:: Supported compact ASE choices
   :header-rows: 1

   * - Basis source
     - ``Basis`` backend
     - ``model.ase_calculator`` backend
     - Notes
   * - ``density``
     - ``pytorch`` by default
     - ``pytorch`` by default; ``native_cpu`` for strict YACE-compatible models
     - Torch ASE energy, forces, and stress by default. Choose
       ``force_method="analytic_factorized"`` for the explicit analytic force
       path; the default is ``autograd``. ``native_cpu`` calls the separate C++
       YACE evaluator and rejects models that cannot be lowered to strict YACE.
   * - ``tagged_cauchy_image``
     - ``auto`` by default; ``reference`` or ``native`` explicitly
     - ``reference`` by default; ``native_polynomial`` or ``native_cpu`` explicitly
     - ``reference`` uses the Torch reference evaluator. ``native_polynomial``
       accelerates only the polynomial contraction. ``native_cpu`` uses the
       optional C++ library built from this package's ``native/`` source.
   * - ``bar_phi``
     - ``pytorch`` only
     - ``pytorch`` or its ``reference`` alias
     - Torch ASE energy, forces, and stress; no C++ or LAMMPS export.

The ordinary low-level contraction selector also accepts ``auto``, ``triton``,
and ``openequivariance``. Those names select internal contraction paths where
supported. Use ``native_cpu`` explicitly for the standalone C++ evaluator;
``backend="native"`` is only a core contraction alias and may still run through
Torch. Strict requests for unsupported low-level paths fail rather than
silently using Torch.

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

To use C++ for a compatible ordinary model, build the shared library below
and select ``native_cpu``:

.. code-block:: python

   model = LinearModel.read("ordinary.pt")
   model.export_lammps("ordinary.yace")  # checks strict YACE compatibility
   atoms.calc = model.ase_calculator(backend="native_cpu")
   energy, forces = atoms.get_potential_energy(), atoms.get_forces()

The compact density default radial basis may not pass strict YACE export.
Choose a PACE-compatible radial specification when fitting a model intended
for the C++ evaluator. Unsupported models raise an export error; they are not
silently evaluated by Torch. The ``native_cpu`` route does not apply to
``bar_phi``.

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
       backend="native_cpu", execution_policy="direct",
   )
   native_forces = atoms.get_forces()

Build the optional C ABI library used by both tagged and ordinary density
models from this package's ``native/`` source directory. Run these commands
from the ``ye3t-methods`` source root with a sibling ``ye3t`` source checkout,
a C++17 compiler, and CMake 3.20 or newer. yaml-cpp is bundled and built
statically; a system installation is not needed:

.. code-block:: bash

   cmake -S native -B ../build-ye3t-methods-native \
     -DCMAKE_BUILD_TYPE=Release \
     -DYE3T_RUNTIME_SOURCE="$PWD/../ye3t"
   cmake --build ../build-ye3t-methods-native --target ye3t_tagged_c_api --parallel
   export YE3T_TAGGED_C_API_LIBRARY="$PWD/../build-ye3t-methods-native/libye3t_tagged_c_api.so"

The examples above use ``YE3T_TAGGED_C_API_LIBRARY``. To override it for a
calculator, pass ``native_library="../build-ye3t-methods-native/libye3t_tagged_c_api.so"``
when running from this source root. A source install using
``python -m pip install --no-build-isolation .`` builds and packages the library
automatically when installed ``ye3t``, CMake 3.20+, and a C++17 compiler are
available. The adapters find that packaged library without an
environment variable. Pass ``-DYE3T_NATIVE_CPU=ON`` at CMake configuration to
optimize for the build host, and ``-DYE3T_ENABLE_IPO=ON`` when interprocedural
optimization is supported. The optional native source has its own GPL-2.0-or-later
license in ``native/LICENSE``.

For a Python-only install, set ``YE3T_METHODS_BUILD_NATIVE=0`` before running
``python -m pip install --no-build-isolation .``. This skips CMake and the
native library; the PyTorch evaluators remain available.

``execution_policy`` is ``direct`` or ``auto``;
``auto`` calibrates schedules when the model is opened and takes longer to
initialize. The native model and its schedules stay resident in the calculator.
It reuses neighbor topology while atoms move within the 0.3 Å skin.

The native ASE adapter uses SciPy's cKDTree for eligible orthorhombic or
nonperiodic cells. For other cells it uses ``matscipy.neighbours`` when
installed and otherwise falls back to ASE's neighbor list. ``matscipy`` is an
optional speedup; it changes neighbor construction, not the model or force
formula. The calculator's ``native_runtime.last_neighbor_backend`` reports the
path actually used.
Install it with ``python -m pip install --no-build-isolation '.[neighbors]'`` from the methods
source checkout after installing the local ``ye3t`` dependency.

A previous local matched 127-feature Ni composite (59 ACE and 68 tagged
features) run on 256 atoms and 1,000 NVE steps recorded a 10.051 ms/step ASE
median and 8.248 ms/step LAMMPS median (ASE
1.22 times the LAMMPS time). Its initial energy and maximum force differences
were 2.51e-12 eV and 3.16e-13 eV/Å. The provenance is in the separate
``ye3t-workflows/tagged_right_tag_speed/README.md`` study; this is prior
tagged-composite evidence, not a timing result for the new ordinary YACE ASE
adapter or a portable performance guarantee.

Paper deployment artifacts
--------------------------

The exact paper tagged model is a composite of an ordinary ``.yace`` backbone
and a tagged correction. Load it directly with the native ASE adapter:

.. code-block:: python

   from ye3t_ace.tagged_cauchy_image import YE3TTaggedCauchyCalculator

   atoms.calc = YE3TTaggedCauchyCalculator.from_artifact(
       "lammps/Li/models/ye3t_tagged_127/model.ye3t.json",
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

The paper's ACE control is shipped as ``.yace`` for LAMMPS/PACE. Evaluate that
same saved artifact directly in ASE with the C++ library:

.. code-block:: python

   from ye3t_ace.yace_native import YE3TYACENativeCalculator

   atoms.calc = YE3TYACENativeCalculator.from_artifact(
       "lammps/Li/models/ace_127/potential.yace",
   )

For the complete paper potential, add the ZBL overlay from the ACE model
manifest with ``SumCalculator`` as in the tagged example above. The runnable
``examples/publication/cost_comparison/ase_native_density.py`` shows the exact
Li model and its step-zero reference. ``LinearModel.read`` accepts ordinary
``.pt`` bundles, not ``.yace``; use ``YE3TYACENativeCalculator.from_artifact``
for a saved YACE file. The paper's composite tagged schema likewise uses its
dedicated native adapter. Newly fitted compact density, tagged, and Phi models
retain the Torch ASE routes above.
