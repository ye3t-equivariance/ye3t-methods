Choosing a linear evaluator
===========================

``Basis(backend=...)`` controls descriptor construction and fitting.
``LinearModel.ase_calculator(evaluator=..., neighbors=...)`` selects evaluation
of a saved or fitted model. ``backend=...`` remains an alias for the evaluator
selector; specify one of them. These are separate choices. The ordinary ACE low-level
``source_backend`` setting selects a one-neighbor source kernel. Full C++
ordinary ASE evaluation uses ``backend="native_cpu"`` and requires the
model to pass strict YACE export.

For a native CPU scalar ASE calculator, ``neighbors="auto"`` uses the
validated default neighbor route. Select ``neighbors="ase"`` or
``neighbors="matscipy"`` explicitly to use that builder; ``matscipy`` needs
the optional ``neighbors`` extra. With ``evaluator="auto"``, an explicit
neighbor choice selects ``native_cpu`` for a compatible density or standalone
tagged model. An unsupported native export or missing native library raises an
error. For a configured PACE density model, ``Basis.create`` uses ordered
Torch descriptor rows independently of ``runtime.evaluator``. A config that
selects ``native_cpu`` uses strict YACE export for its fitted model when
``evaluator="auto"`` is requested, while ``evaluator="torch"`` remains an
explicit comparison route. A config that selects ``torch`` keeps that choice
for ``auto``. A reloaded compact ``.pt`` model retains its labels but does not
retain the config's evaluator choice; pass ``evaluator="native_cpu"`` to select
the native path after loading. Unsupported evaluator names raise an error.
For a supported shifted-Jacobi tagged scalar config at one or more selected
ranks with explicit
or one-hot chemistry,
``runtime.evaluator="native_cpu"`` likewise keeps reference descriptor rows
and selects the existing tagged native ASE calculator for an in-memory fitted
model. Its explicit ``neighbors`` choice applies to ASE model evaluation;
standalone descriptor rows retain the validated reference geometry route.
After reading the saved tagged ``.ye3t.json`` model, select ``native_cpu``
explicitly again.
For an in-memory configured model, omitting ``neighbors`` in
``ase_calculator`` uses the config's requested neighbor policy. Passing
``neighbors="auto"`` explicitly overrides it. Loaded compact models default
to ``auto`` because they do not retain that runtime preference.
When a native config requests ``matscipy``, an explicit Torch comparison
uses the compatible ASE or reference neighbor route; explicitly asking a
Torch calculator for ``matscipy`` still raises an error.

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
   * - Configured density ``L>0`` per-atom multiplets
     - ``pytorch`` on CPU
     - ``pytorch`` on CPU
     - ASE real-tesseral per-atom mean and optional ARD covariance; no scalar
       energy readout. The separate LAMMPS CPU property compute accepts the
       qualified ``L=1`` odd and ``L=2`` even v2 density artifacts. The
       density Kokkos property compute remains unfinished.
   * - ``tagged_cauchy_image``
     - ``auto`` by default; ``reference`` or ``native`` explicitly
     - ``reference`` by default; ``native_polynomial`` or ``native_cpu`` explicitly
     - ``reference`` uses the Torch reference evaluator. ``native_polynomial``
       accelerates only the polynomial contraction. ``native_cpu`` uses the
       optional C++ library built from this package's ``native/`` source.
   * - Configured tagged ``L>0`` per-atom multiplets
     - ``reference`` on CPU
     - ``reference`` on CPU
     - ASE real-tesseral mean and optional ARD covariance are tested through
       ``L=3``. A separate LAMMPS CPU property compute accepts qualified
       ``L=1,2`` means. The tagged device compute is experimental and has
       passed bounded CPU/device parity tests; no posterior covariance is
       exported to LAMMPS.
   * - ``bar_phi``
     - ``pytorch`` only
     - ``pytorch`` or its ``reference`` alias
     - Torch ASE energy, forces, and stress; no C++ or LAMMPS export.

The ordinary low-level contraction selector also accepts ``auto``, ``triton``,
and ``openequivariance``. Those names select internal contraction paths where
supported. Use ``native_cpu`` explicitly for the standalone C++ evaluator;
``backend="native"`` is only a core contraction alias and may still run through
Torch. Strict requests for unsupported low-level paths raise an error.

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
   atoms.calc = model.ase_calculator(evaluator="native_cpu", neighbors="ase")
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
   atoms.calc = model.ase_calculator(evaluator="torch")
   reference_forces = atoms.get_forces()

   atoms.calc = model.ase_calculator(
       evaluator="native_cpu", neighbors="ase", execution_policy="direct",
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

Set ``YE3T_METHODS_BUILD_NATIVE=ase`` during pip installation to build only the
ordinary-density and tagged C++ ASE evaluators. This omits the lifted native
evaluator sources; model loading still uses bundled static yaml-cpp.

``execution_policy`` is ``direct`` or ``auto``;
``auto`` calibrates schedules when the model is opened and takes longer to
initialize. The native model and its schedules stay resident in the calculator.
It reuses neighbor topology while atoms move within the 0.3 Å skin.
The tagged Torch reference enumerates all periodic images for validation. It
raises on numerically ill-conditioned periodic cells or a scan exceeding one
million candidate shifts; the native ASE/matscipy routes build neighbor lists
without that brute-force scan.

``LinearModel.read`` also accepts the saved Ni composite
``model.ye3t.json`` when its verified v4 ``model_manifest.json`` and component
files are colocated. Its ASE calculator applies the recorded ZBL term. The
legacy composite has no serialized ordered compiler labels, so standalone
descriptor rows and portable rewrite are unavailable for that artifact.

The vetted single-file Ni portable ``.ye3t`` archives expose ordered
``model.basis.create(atoms)`` rows and use ``evaluator="torch"`` or ``auto``
on CPU for the complete ordinary-plus-tagged-plus-ZBL energy, forces, and
stress. ``neighbors="ase"`` is supported. Explicit ``native_cpu`` and
``matscipy`` requests reject on this bounded portable route. The saved
couplings are read once; geometry changes do not compile a new coupler.

Compilation caches
------------------

Building a basis can compile Clebsch--Gordan, Young subduction, and descriptor
tables. Compact density ``Basis`` writes reusable tables through
``descriptor_cache_dir``; compact tagged ``Basis`` uses
``compiled_cache_dir``. Both default to a persistent directory under the
user's YE3T application cache, so constructing the same basis in a later
process can load validated artifacts. Set either path explicitly when a
project needs its own cache:

.. code-block:: python

   from pathlib import Path
   from ye3t_methods import Basis

   basis = Basis(
       elements=["Ni"], source="tagged_cauchy_image", cutoff=4.8,
       pair_cutoffs_A={"Ni-Ni": 4.8},
       rank=4, tag_counts=(0, 2),
       nmax_per_rank={4: 1}, lmax_per_rank={4: 1},
       source_block_partitions_by_rank={4: ((4,),)},
       angular_patterns_by_rank={4: ((1, 1, 1, 1),)},
       compiled_cache_dir=Path.home() / ".cache" / "ye3t" / "tagged",
   )
   print(basis.resolved["compiled_cache_dir"])

Density descriptor-build entries use verified JSON for labels and metadata,
with hash-bound NPZ sidecars for magnetic-index and coefficient arrays. A
per-key process lock protects the JSON entry. Legacy ``.pkl`` and older
JSON cache entries are ignored and can be pruned after a
successful rebuild; they are never loaded by this path. Set
``YE3T_CACHE_MODE`` to ``auto`` (default), ``read_only``, ``refresh``, or
``off`` for the global disk store. The descriptor cache's in-process label
and artifact LRUs have separate configurable byte budgets.

To prewarm an ordinary ACE descriptor catalogue, save a JSON object with
``schema: "ye3t_methods_prewarm_v1"``, ``settings`` in the
``DescriptorGenerationSettings.as_dict()`` format, and ``selected_labels``
containing only compiler-issued ``CompactLabel.to_dict()`` records. Preview
the work before compiling:

.. code-block:: bash

   python -m ye3t_methods.cache --catalogue selected.json --cache-dir /path/to/cache
   python -m ye3t_methods.cache --catalogue selected.json --cache-dir /path/to/cache --apply

The preview validates selected labels through ``ye3t.couplings.count`` and
reports label count and a magnetic-input-size proxy without writing cache
entries or materializing coefficients. The default limits are 100 labels,
rank 4, and 100,000 magnetic input elements; override them with
``--max-labels``, ``--max-rank``, and ``--max-magnetic-elements``. These are
precompile size limits, not a wall-time or memory guarantee. ``--apply``
requires an explicit cache directory and a writable cache mode. This route
prewarms selected ordinary ``no_charge`` ACE descriptor artifacts; other
source programmes still compile through their normal validated paths.
Existing ``ye3t_ace_prewarm_v1`` catalogues remain readable and retain their
historical report schema; new catalogues should use the methods name above.

The ordered tagged-carrier example also exposes
``config["runtime"]["compiled_cache_dir"]``. The cache location can be
controlled globally with ``YE3T_ACE_CACHE_DIR`` or ``YE3T_CACHE_DIR``.
The compiler checks saved request identities and hashes before reuse.
Construct an ASE calculator once and keep it attached to the atoms: fitted
coefficients, compiled schedules, and native model handles remain resident
through repeated energy, force, and stress calls. Geometry and neighbor lists
still update when atoms move; coupling compilation does not run for each call.

The native ASE adapter uses SciPy's cKDTree for eligible orthorhombic or
nonperiodic cells. For other cells it uses ``matscipy.neighbours`` when
installed and otherwise falls back to ASE's neighbor list. ``matscipy`` is an
optional speedup; it changes neighbor construction, not the model or force
formula. The calculator's ``native_runtime.last_neighbor_backend`` reports the
path actually used.
Install it with ``python -m pip install --no-build-isolation '.[neighbors]'`` from the methods
source checkout after installing the local ``ye3t`` dependency.

Paper deployment artifacts
--------------------------

The exact paper tagged model is a composite of an ordinary ``.yace`` backbone
and a tagged correction. Load it directly with the native ASE adapter:

.. code-block:: python

   from ye3t_methods.atomistic.tagged_cauchy_image import YE3TTaggedCauchyCalculator

   atoms.calc = YE3TTaggedCauchyCalculator.from_artifact(
       "lammps/Li/models/ye3t_tagged_127/model.ye3t.json",
       execution_policy="direct",
   )

The standalone adapter evaluates the **linear residual model**. For the full
paper potential, read the ZBL settings from its model manifest and combine
the two ASE calculators:

.. code-block:: python

   from ase.calculators.mixing import SumCalculator
   from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator

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

   from ye3t_methods.atomistic.yace_native import YE3TYACENativeCalculator

   atoms.calc = YE3TYACENativeCalculator.from_artifact(
       "lammps/Li/models/ace_127/potential.yace",
   )

For the complete paper potential, add the ZBL overlay from the ACE model
manifest with ``SumCalculator`` as in the tagged example above. The runnable
``examples/publication/cost_comparison/ase_native_density.py`` shows the exact
Li model and its step-zero reference. ``LinearModel.read`` accepts ordinary
``.pt`` bundles, not ``.yace``; use ``YE3TYACENativeCalculator.from_artifact``
for a saved YACE file. The paper's composite tagged schema likewise uses its
dedicated native adapter. Compact density, tagged, and Phi models fitted through
``LinearModel`` use the Torch ASE routes above.
