Installed-package quickstarts
=============================

Install the ``ye3t-methods`` wheel in an environment with a compatible
``ye3t``. When ``ye3t`` has not been published to an index, install it from a
sibling source checkout with ``python -m pip install --no-build-isolation ../ye3t``. Extract the
methods source archive to obtain the scripts and their fixture files. From the
extracted directory, run:

.. code-block:: console

   python -m pip install --no-build-isolation '.[examples,fit,neighbors]'

The ``fit`` extra adds scikit-learn for LASSO and ARDRegression. The
``neighbors`` extra adds matscipy for optional native ASE neighbor-list
acceleration. Either extra can be installed separately.
Use ``ye3t.YE3TRepresentation`` for the configured basis examples. The
``ye3t_methods.YE3TRepresentation`` re-export serves older descriptor-first
code. The maintained package supplies its own ``ye3t_ace`` compatibility
modules for saved models; avoid installing the historical ``ye3t-ace``
distribution in the same environment because it owns the same module path.

Start with one of these complete workflows:

.. code-block:: console

   python examples/quickstart/ase_descriptors.py
   python examples/quickstart/tagged_descriptors.py
   python examples/quickstart/ase_octupole_descriptors.py
   python examples/quickstart/chemical_encoding.py
   python examples/quickstart/chemical_channel_selection.py
   python examples/quickstart/density_fit.py
   python examples/quickstart/combined_density_tagged_fit.py
   python examples/quickstart/per_atom_vector_to_lammps.py
   python examples/quickstart/paper_ni_portable_ase.py
   python examples/quickstart/paper_ni_nve.py

The scalar, tagged, octupole, and chemical examples return per-atom NumPy
descriptor rows without training. The next three fit and save models; the combined example selects a distinct
physical radial source for each component. The vector example also writes a
LAMMPS input.
The Ni saved-model example reads a pinned paper model and checks its reference
energy. The NVE example runs 100 ASE Velocity-Verlet steps near equilibrium
and writes every energy sample to CSV; this short run does not establish
stability of the potential at compressed geometries.
For an exact selected-basis Ni refit, use the long publication example:

.. code-block:: console

   python examples/publication/cost_comparison/refit_paper_ni.py

Its visible seven-section config uses the 127-column saved basis, the full
published training partition, and frozen paper fit weights. It saves a new
Torch ASE artifact and computes held-out energy/force RMSE. The saved
artifact cannot use LAMMPS AUTO until a native plan for its new weights is
validated; see the publication example README for the 60/149 archive and
selected hyperparameter edits.
For the same configured representation → basis → model path, use these
focused inspection, solver, tagged-source, and saved-model examples:

.. code-block:: console

   python examples/quickstart/tagged_fit.py
   python examples/quickstart/inspect_features.py
   python examples/quickstart/sklearn_fit.py
   python examples/quickstart/saved_ase_export.py

These retained experimental scripts use source-specific or older
descriptor-first interfaces. Run them for the stated method; use the configured
quickstarts above as templates for a new density or tagged model:

.. code-block:: console

   python examples/experimental/phi_fit.py
   python examples/experimental/phi_depth2.py
   python examples/experimental/role_density_fit.py

The following inspect core coupling mathematics on supplied tensor factors.
They do not take an ASE ``Atoms`` object or produce a fitted model:

.. code-block:: console

   python examples/quickstart/parent_coupling.py
   python examples/quickstart/parent_coefficient.py

``phi_fit.py`` and ``phi_depth2.py`` use the tested scalar ``bar_phi`` motif
route with fixed signed magnetic channels; see :doc:`phi_basis` for its
rotation and deployment limits. ``role_density_fit.py`` uses the retained
descriptor-first lifted-density path; see :doc:`role_density` for its
slot-sector limits. ``chemical_encoding.py`` uses the configured physical
fixed-embedding source with ASE and checks scalar symmetries, while
``chemical_channel_selection.py`` masks excluded neighbor species with a
fixed zero embedding row;
see :doc:`chemical_encoding`. The separate experimental
``ordered_phi_star.py`` evaluates one full-tableau ``L=2`` ordered cluster;
it does not fit a model (see :doc:`phi_basis`). Use the first group above
for the recommended public workflow.

``ase_descriptors.py`` is the short descriptor-only workflow: build an ASE
``Atoms`` object, construct ``ye3t.YE3TRepresentation`` and
``ye3t_methods.Basis`` from its visible config, then call
``descriptors = basis.create(atoms)``. The result is a real NumPy array with
one row per atom and compiler-selected scalar columns. The Ni radial source
and rank schedule extend through rank eight, with larger radial and angular
caps at ranks one through four. The selected source partitions give 96
independent columns on the four-atom Ni cell. The output is scalar
because ``representation.parent.L`` is zero and the parent Young sector is
``(N)``. A nonzero ``lmax_per_rank`` allows angular input factors; it does
not make the output vector-valued. Edit the species, cutoff, radial family,
rank schedule, and angular/radial caps for another system. The exact saved
paper Ni-127 descriptors are shown by ``paper_ni_portable_ase.py``. This
descriptor example does not fit or load a model.
``tagged_descriptors.py`` follows the same public object flow with a rank-four
tagged source. Its scalar parent stays globally symmetric, while two local
Young ``(1,1)`` blocks carry angular ``L=1`` intermediates that couple to
``L=0``. It returns fourteen ASE columns for four Ni atoms and checks a
non-axis rotation and atom relabeling. The first materialization compiles
the selected tagged image, so this advanced example takes longer than the
ordinary descriptor quickstart.
For default scalar ACE descriptors, the compact ``Basis`` constructor gives
the same ASE ``Atoms`` to NumPy usage without a full fit config:

.. code-block:: python

   from ase.build import bulk
   from ye3t_methods import Basis

   atoms = bulk("Ni", "fcc", a=3.52, cubic=True)
   basis = Basis(elements=["Ni"], cutoff=5.0, max_rank=3,
                 nmax=(2, 2, 1), lmax=(1, 1, 1))
   descriptors = basis.create(atoms)  # shape (4, 10)

The same call handles more than one species; list every element in the ASE
structure and choose the desired basis settings:

.. code-block:: python

   from ase.build import molecule
   from ye3t_methods import Basis

   atoms = molecule("H2O")
   basis = Basis(elements=["H", "O"], cutoff=4.0, max_rank=2,
                 nmax=(2, 1), lmax=(1, 1))
   descriptors = basis.create(atoms)  # shape (3, 24)

Use the full representation and basis config when the target symmetry,
physical factor source, or catalogue policy is part of the calculation.
``ase_octupole_descriptors.py`` uses the same public objects for an ``L=3``
odd-parity output. Its rows have seven real-tesseral components per feature.
The displaced Ni atom ensures a nonzero demonstration; this is a descriptor
example rather than a fitted material property.

``density_fit.py`` is the first configured linear fit example. It constructs
``ye3t.YE3TRepresentation``, passes it to ``Basis.from_config``, fits through
``LinearModel``, then reloads and evaluates the saved model with ASE. Its
editable seven-section config includes rank, source, target, backend, and
workflow paths. It draws twelve labeled Ni structures across the published
training split, fits a 19-column paper-informed ordinary basis, and evaluates
one held-out structure. The radial settings match the paper; the angular
caps and training count are smaller, so its output is an interface example,
not a reproduction of the selected paper models or their held-out RMSE.
Change the paths and training count under ``metadata`` for another training
set or output location; these paths do not change the fitted-model identity.
``combined_density_tagged_fit.py`` uses one configured basis with ordinary
PACE density and shifted-Jacobi tagged components on a 32-atom periodic Ni
cell, with ranks through four, one scalar fit, and one saved artifact. Its
energy labels are manufactured to exercise the workflow;
they are not a fitted physical potential. The example prints the two source
families, descriptor shape, and restored ASE prediction.
``per_atom_vector_to_lammps.py`` fits an odd-parity vector to a displaced
32-atom periodic Cu cell, saves and reloads the model, and writes a LAMMPS
``compute ye3t/property/atom`` input. Its analytic site-vector labels only
exercise the workflow; they are not measured material properties. Change
``metadata.system`` for a different ASE crystal setup. The model output is
real-tesseral order, so compare the LAMMPS dump with
``reference_real_tesseral.txt`` rather than Cartesian components. To run the
native half of the example, use an ML-YE3T-enabled LAMMPS executable from the
generated output directory:

.. code-block:: console

   cd ../ye3t-workflows/quickstart_linear/cu_site_vector
   lmp -in in.property

The installed Python packages alone run the fit and write ``in.property``;
LAMMPS is a separate installed consumer. The example uses repeated-content
rank-three and rank-four ACE coordinates, so it also checks the full magnetic
compiler route used by ordinary density multiplets.
``tagged_fit.py`` follows the same object sequence. Its rank-four tagged
source has two repeated blocks and a compiler-derived coordinate with local
Young partitions ``(1,1)`` and block angular outputs ``(1,1)``. It actually
fits energy, forces, and stress from bundled Ta interface fixtures, saves and
reloads the result, then exports the hash-bound LAMMPS model, atomic data,
and an ``in.ta_tagged_auto`` input using ``block_policy auto``. Run the input
from its output directory with a LAMMPS build supporting that model schema.
The fixture targets are
manufactured; the fit is not a paper accuracy reproduction. Its ``native_cpu``
run needs the tagged C ABI library installed or
``YE3T_TAGGED_C_API_LIBRARY`` set to that library.

The Ni script evaluates a four-atom fcc cell and its displaced copy. Its
``validation`` settings inspect internal tag and role Young types supplied by
the compiler; the tagged basis keeps a fixed symmetric scalar parent.
``examples/publication/cost_comparison/ase_native_ni.py`` instead loads the
promoted Ni tagged potential, adds its
ZBL overlay, and checks the energy of the 32-atom reference cell against the
retained LAMMPS result. It needs a native C++ installation.
``paper_ni_portable_ase.py`` reads the portable model through
``LinearModel.read`` and verifies the same step-zero energy through the
Python/ASE route. Its basis and representation are recovered from the saved
artifact; the editable crystal geometry lives under ``metadata.system``.
``paper_ni_nve.py`` uses the same saved model for 100 near-equilibrium ASE
NVE steps with deterministic initial velocities and a CSV energy log.

Each script imports installed packages. The Ni density fit uses the bundled
mlearn labels; the tagged fit uses deterministic interface fixtures; other
fitting scripts use fixtures or generate small labels. The chemical example
constructs its own two-species source inputs. The fit scripts write their resulting model
under a sibling ``ye3t-workflows/quickstart_linear`` directory. The fitting
scripts expose their ASE evaluator choices in ``config["runtime"]``; the
short descriptor scripts show their inputs directly.
Change ``elements``, ``cutoff``, basis settings, and input structures for a new
system; the fixture configurations are deliberately small for a quick
numerical check. See :doc:`evaluators` for all supported choices.
The manufactured Cu, Ta, and H scripts verify fitting interfaces; they are
not paper-result reproductions. The promoted Li, Mo, Cu, Ni, Si, and Ge model
artifacts and their inputs are described in :doc:`paper_models`.

Ordinary density fit
--------------------

.. literalinclude:: ../examples/quickstart/density_fit.py
   :language: python

Combined physical sources
-------------------------

.. literalinclude:: ../examples/quickstart/combined_density_tagged_fit.py
   :language: python

Per-atom vector fit and LAMMPS input
------------------------------------

.. literalinclude:: ../examples/quickstart/per_atom_vector_to_lammps.py
   :language: python

ASE descriptors without fitting
-------------------------------

.. literalinclude:: ../examples/quickstart/ase_descriptors.py
   :language: python

.. literalinclude:: ../examples/quickstart/tagged_descriptors.py
   :language: python

.. literalinclude:: ../examples/quickstart/ase_octupole_descriptors.py
   :language: python

Tagged physical-image fit
-------------------------

.. literalinclude:: ../examples/quickstart/tagged_fit.py
   :language: python

Near-equilibrium ASE molecular dynamics
---------------------------------------

.. literalinclude:: ../examples/quickstart/paper_ni_nve.py
   :language: python

Experimental explicit motif fit
-------------------------------

.. literalinclude:: ../examples/experimental/phi_fit.py
   :language: python

Experimental role-density fit
-----------------------------

This quick variant uses the descriptor-first API and an oracle model to make
small training labels. It verifies the fitted linear path; it does not assess a
physical Ta potential.

.. literalinclude:: ../examples/experimental/role_density_fit.py
   :language: python

``inspect_features.py`` fits through the same configured object workflow and
prints one compiler-selected label, its full description, and its fitted
coefficient. ``sklearn_fit.py`` changes only the configured solver between
LASSO and ARD, saves both models, and evaluates the restored ARD uncertainty;
it requires the ``fit`` extra. ``saved_ase_export.py`` loads a tagged artifact
for ASE evaluation and LAMMPS export. Its explicit ``reference`` evaluator
works with the Python-only installation; change it to ``native_cpu`` when the
native ABI is installed. The
``examples/quickstart/fixtures/README.md`` file identifies the fixture labels
and their intended use.

Chemical encoding source check
------------------------------

This example checks one-hot source channels and a fixed in-memory chemical
embedding kernel, including source position derivatives. Custom providers
have no saved-model or LAMMPS export contract; see :doc:`chemical_encoding`.

.. literalinclude:: ../examples/quickstart/chemical_encoding.py
   :language: python
