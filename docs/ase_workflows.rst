ASE descriptors and linear models
=================================

These examples use ``ye3t.YE3TRepresentation`` to select the output symmetry,
``ye3t_methods.Basis`` to evaluate descriptors on ASE ``Atoms``, and
``ye3t_methods.LinearModel`` to fit labeled structures. Each script shows its
complete editable seven-section config. Run the commands from an extracted
``ye3t-methods`` source archive with ``ye3t``, ``ye3t-methods``, and ASE
installed. See :doc:`quickstart` for installation and optional extras.

Descriptors from one ASE structure
----------------------------------

The first calculation resembles an ASE calculator or an icet feature call:
construct an ``Atoms`` object, construct a representation and basis, then
call ``basis.create(atoms)``. The result is a NumPy array with one row per
atom. The Ni example uses a scalar, fully symmetric parent and ranks through
eight. Edit the species, cutoff, radial family, ranks, and per-rank caps for
another material.

The central calls are:

.. code-block:: python

   representation = YE3TRepresentation.from_config(config["representation"])
   basis = Basis.from_config(
       config["basis"], representation=representation, runtime=config["runtime"],
   )
   descriptors = basis.create(atoms)

The full runnable script supplies the editable config and ASE ``atoms``.

.. literalinclude:: ../examples/quickstart/ase_descriptors.py
   :language: python
   :caption: examples/quickstart/ase_descriptors.py

For the four-atom fcc cell, this prints ``descriptor shape (4, 96)``,
``parent L 0``, and ``compiler count 96``. The input angular cap permits
nonzero angular factors; the selected output remains scalar.

Fit from labeled ASE structures
-------------------------------

The next script reads bundled Ni structures with energy and force labels,
selects training structures using the bundled split, fits a rank-through-four
linear model, saves it, and evaluates a held-out ASE structure. Its config
shows the radial source, basis caps, target keys, fit weights, and output path.
Replace the dataset and split paths with your labeled ASE data for another
fit. This short fit demonstrates the user workflow; the paper-model refit
uses the selected basis and full training partition described in
:doc:`paper_models`.

After building the basis, fitting and ASE use require only:

.. code-block:: python

   model = LinearModel(basis).fit(structures, config=config)
   artifact = model.write(output_path)
   restored = LinearModel.read(artifact)
   atoms.calc = restored.ase_calculator(evaluator="torch", neighbors="ase")

The full script below reads labeled structures, builds the basis, and sets
``output_path``.

.. literalinclude:: ../examples/quickstart/density_fit.py
   :language: python
   :caption: examples/quickstart/density_fit.py

The script prints the feature count, saved-model path, held-out reference and
predicted energies, and the largest predicted force component. The result is
a fitted example model, not a paper RMSE reproduction.

Run ASE molecular dynamics with a saved model
---------------------------------------------

The saved Ni-127 paper model can also act as an ASE calculator. This example
builds a periodic cell, attaches the loaded model, and runs 100 NVE
Velocity-Verlet steps. It writes every energy and temperature sample to CSV.
Edit the model path, crystal, temperature, timestep, and output path in the
config for another run.

.. literalinclude:: ../examples/quickstart/paper_ni_nve.py
   :language: python
   :caption: examples/quickstart/paper_ni_nve.py

For a tagged basis with nontrivial local Young and angular intermediates,
see :doc:`tagged_basis` and
``examples/quickstart/tagged_descriptors.py``. Both descriptor scripts use
the same ``YE3TRepresentation`` → ``Basis`` → ``create(atoms)`` workflow.
