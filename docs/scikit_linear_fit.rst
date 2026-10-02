Optional scikit-learn fits and ARD uncertainty
===============================================

The compact ``LinearModel`` supports fixed-feature fits with ``lasso``,
``ardregression`` (alias ``ard``), ``linear_regression``, and ``ridgecv``
for density, tagged physical-image, and explicit ``bar_phi`` bases. The
default ``fit_method="ridge"`` keeps the existing ridge solver. The optional
scikit-learn dependency is used only when one of its methods is selected.

Install from sibling source checkouts when ``ye3t`` is not on a package index:

.. code-block:: bash

   python -m pip install ../ye3t
   python -m pip install '.[fit]'

``structures`` below are ASE atoms with stored energy and force labels;
``basis`` is a constructed ``Basis``. The same call works for all three
compact basis sources. Tagged and Phi fits can also use stored ASE stress
labels through ``stress_weight``. Density fitting does not accept stress rows.

.. code-block:: python

   from ye3t_methods import LinearModel

   sparse = LinearModel(basis).fit(
       structures,
       fit_method="lasso",
       sklearn_params={"alpha": 1e-6, "max_iter": 100000},
   )
   sparse.write("sparse_model")

   ard = LinearModel(basis).fit(
       structures,
       fit_method="ardregression",
       sklearn_params={"threshold_lambda": 10000.0},
   )
   ard.write("ard_model")

The ``regularization`` argument belongs to the default ridge solver. For a
scikit-learn method, set its penalty or prior through ``sklearn_params``.
Energy, force, and stress weights multiply their respective squared-residual
rows; the parameters and fit method are stored with the fitted model.

ARD posterior predictions
-------------------------

An ARD fit stores its coefficient covariance in the model artifact. Load the
model and request per-atom and total-energy standard deviations:

.. code-block:: python

   restored = LinearModel.read("ard_model.pt")  # or .ye3t.json / .phi.pt
   prediction = restored.predict_uncertainty(atoms)
   atomic_std_eV = prediction["atomic_energy_std_eV"]
   total_std_eV = prediction["total_energy_std_eV"]

These are conditional *linear readout* uncertainties for the fixed descriptor
basis, computed as ``sqrt(x Sigma x.T)`` from the active ARD coefficient
covariance. They exclude descriptor/model error and observation noise. The
total-energy standard deviation uses the sum of the per-atom design rows;
the per-atom standard deviations must not be summed independently because
the atoms share coefficient uncertainty. The current API does not report
force or stress uncertainty. The posterior is available through Python after
reload, not through the LAMMPS or native C++ evaluator.
