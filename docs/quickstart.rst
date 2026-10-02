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

Then run:

.. code-block:: console

   python examples/evaluate_ni_descriptors.py
   python examples/quickstart/paper_ni_ase.py
   python examples/quickstart/density_fit.py
   python examples/quickstart/tagged_fit.py
   python examples/quickstart/phi_fit.py
   python examples/quickstart/phi_depth2.py
   python examples/quickstart/role_density_fit.py
   python examples/quickstart/inspect_features.py
   python examples/quickstart/saved_ase_export.py
   python examples/quickstart/chemical_encoding.py
   python examples/quickstart/chemical_channel_selection.py
   python examples/quickstart/parent_coupling.py
   python examples/quickstart/parent_coefficient.py

The Ni script evaluates a four-atom fcc cell and its displaced copy. Its
``validation`` settings inspect internal tag and role Young types supplied by
the compiler; the tagged basis keeps a fixed symmetric scalar parent.
``paper_ni_ase.py`` instead loads the promoted Ni tagged potential, adds its
ZBL overlay, and checks the energy of the 32-atom reference cell against the
retained LAMMPS result. It needs a native C++ installation.

Each script imports installed packages. The fitting scripts use the fixtures
next to them or generate small labels; the chemical example constructs its
own two-species source inputs. The fit scripts write their resulting model
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

Tagged physical-image fit
-------------------------

.. literalinclude:: ../examples/quickstart/tagged_fit.py
   :language: python

Explicit motif fit
------------------

.. literalinclude:: ../examples/quickstart/phi_fit.py
   :language: python

Retained role-density fit
-------------------------

This quick variant uses the descriptor-first API and an oracle model to make
small training labels. It verifies the fitted linear path; it does not assess a
physical Ta potential.

.. literalinclude:: ../examples/quickstart/role_density_fit.py
   :language: python

``inspect_features.py`` inspects actual descriptor columns, and
``saved_ase_export.py`` loads a tagged artifact for ASE evaluation and
LAMMPS export. The
``examples/quickstart/fixtures/README.md`` file identifies the fixture labels
and their intended use.

Chemical encoding source check
------------------------------

This example checks one-hot source channels and a fixed in-memory chemical
embedding kernel, including source position derivatives. Custom providers
have no saved-model or LAMMPS export contract; see :doc:`chemical_encoding`.

.. literalinclude:: ../examples/quickstart/chemical_encoding.py
   :language: python
