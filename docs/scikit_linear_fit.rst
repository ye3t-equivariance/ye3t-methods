Optional scikit-learn fits and ARD uncertainty
===============================================

The compact ``LinearModel`` supports fixed-feature fits with ``lasso``,
``ardregression`` (alias ``ard``), ``linear_regression``, and ``ridgecv``
for density, tagged physical-image, and explicit ``bar_phi`` bases. The
default ``fit_method="ridge"`` keeps the existing ridge solver. The optional
scikit-learn dependency is used only when one of its methods is selected.
An independently configured density plus tagged scalar basis can be combined
with ``Basis.combine`` and fitted with the same methods. The combined fit
retains each component's compiler label order, uses one per-species E0 map,
and accepts stored energy, force, and ASE Voigt stress targets.
For the recommended seven-section config, see
``examples/quickstart/combined_density_tagged_fit.py``: one named-component
``Basis.from_config`` request, one configured fit, and one saved model.

Install from sibling source checkouts when ``ye3t`` is not on a package index:

.. code-block:: bash

   python -m pip install ../ye3t
   python -m pip install '.[fit]'

``structures`` below are ASE atoms with stored energy and force labels;
``basis`` is a constructed ``Basis``. The same call works for all three
compact basis sources. Tagged and Phi scikit-learn fits can also use stored ASE
stress labels through ``stress_weight``. ``LinearModel.fit`` uses the checked
density energy, force, and stress rows for ridge and optional scikit-learn
methods. The lower-level ``fit_linear_ace`` density interface still accepts
stress rows only with its normal-equation ridge solver. The combined scalar
fit supports stress rows for ridge and the optional scikit-learn methods.

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
For a configured standalone density or combined scalar basis,
``fit_E0=False`` keeps the supplied ``reference_energies`` fixed.
``fit_E0=True`` fits one correction per species and saves the final numerical
map. A standalone density fit with fixed offsets does not add an implicit
atom-count bias. For the older direct combined fit, an omitted ``fit_E0``
defaults to fixed offsets if a reference map was supplied, or fitted offsets
otherwise.
Fitting offsets requires a positive energy weight and at least one training
atom of each specified species.
With density or combined-scalar LASSO or ARD, the supplied E0 map is the prior origin:
the solver penalizes descriptor coefficients and E0 corrections together.
Changing that supplied map can change the fitted result. The fitted map and
its prior origin are saved, and export checks the map against the model bundle.
ARD uncertainty on weighted energy, force, and stress rows describes the
weighted regression objective; interpret it as calibrated physical uncertainty
only when the row weights reflect the corresponding noise scales.

Configured PACE density and shifted-Jacobi tagged scalar components can be
written together to a verified ``.ye3t`` bundle. The bundle embeds the
selected ordinary coupling tables and tagged compiled image, so reading it
does not need the training repository or a coupling compilation. Both Torch
and native CPU ASE calculators accept the loaded bundle. Legacy ``Basis(...)``
component constructors can participate in a joint fit, but the safe combined
writer currently requires ``Basis.from_config`` components. A single named
``basis.components`` request can use the same route: give each component a
``tensor_product`` and ``catalogue``, and put a complete ``single_factors``
mapping on a component when its radial source differs from the top-level
default. The component species order must match the top-level order. The
resolver validates each source and retains a coefficient-free catalogue
preview until descriptors are first requested.

For a ``Basis.from_config`` basis, ``LinearModel(basis).fit(structures,
config=config)`` accepts the complete seven-section configuration. The fit
checks that its representation, basis, and runtime resolve to the same basis,
then uses ``model.fit`` for ridge/LASSO/ARD and row weights,
``model.reference_energy`` for the numerical species offsets, and ``targets``
for the stored ASE energy/force/stress keys. A stress target needs an explicit
stress weight. ``validation.checks`` can request ``force_fd`` and
``round_trip``; those checks execute on the first training structure and
record their results with the fitted model. Per-atom non-scalar targets use
the separate selected tagged-carrier route described below.
The saved fit metadata also retains the normalized fit request and its SHA-256
identity. Fitting multiple species offsets requires training compositions
whose species-count columns are independent; otherwise the fit rejects the
unresolved E0 split.
The archive checks the saved construction and solver record against its
component sources and readout. Target keys, row weights, and validation
requests are retained as training provenance; deployed coefficients alone
cannot independently establish which training rows produced them.

A separate selected tagged-carrier ``L>0`` configuration, including
per-rank tag-count sets, accepts
``model.output.scope="per_atom"`` and a ``targets.per_atom`` mapping with
``key``, ``input`` (``real_tesseral`` or ``cartesian``), and ``units``. Its
ridge, LASSO, or ARD readout shares each coefficient over all magnetic
components and separates central-species coefficient blocks. The coefficient
penalty and ARD prior use the saved selected compiler coordinate basis; the
artifact records that coordinate metric, not a basis-independent physical
norm. ``model.predict(atoms)`` returns complete real-tesseral means and
``model.predict(atoms, uncertainty=True)`` returns an ARD component covariance.
The ``.ye3t.json`` artifact embeds and checks the complete compiled catalogue
so reload needs no coupling compilation. See :doc:`tagged_basis` for the
Cartesian component order, supported runtime, and fit scope.

ARD posterior predictions
-------------------------

An ARD fit stores its coefficient covariance in the model artifact. Load the
model and request per-atom and total-energy standard deviations:

.. code-block:: python

   restored = LinearModel.read("ard_model.pt")  # or .ye3t / .ye3t.json / .phi.pt
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
