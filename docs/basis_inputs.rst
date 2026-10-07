Compact basis inputs and compiler labels
========================================

For new work, use ``ye3t.YE3TRepresentation.from_config`` followed by
``ye3t_methods.Basis.from_config``. With an ASE ``Atoms`` object, one call to
``basis.create(atoms)`` returns real NumPy descriptor rows. The complete
copy/paste example is ``examples/quickstart/ase_descriptors.py``; its
``model`` and ``targets`` config sections are empty because it only evaluates
descriptors. The compiler selects the labels, and ``basis.labels`` and
``basis.catalogue.counts()`` expose their identity and count.

The retained direct ``Basis`` constructor takes an ordered ``elements`` list,
physical cutoff, source family, and short radial/angular truncation schedule.
For ordinary density, ``max_rank``, ``nmax``, and ``lmax`` are the inputs;
``nmax`` and ``lmax`` may each be a scalar or a value for every rank.
These values determine candidate source channels. The compact default uses
the original ChebExpCos radial source; it does not
reconstruct the separately configured PACE spline source or a saved paper
model's channel definition. For exact replay, read the model artifact.
The retained
``YE3TRepresentation -> YE3TDescriptors -> YE3TModel`` route accepts
additional fixed catalogues and source policies.

The paper comparison's
``examples/publication/cost_comparison/config.json`` exposes a compact
schedule for ranks 1, 2, 3, 4, 6, and 8 via ``tensor_orders``,
``nmax_by_tensor_order``, and ``lmax_by_tensor_order``. The companion
``tagged_components.json`` specifies fixed physical channel contents and
block sizes. These are *requests*, not lists of accepted descriptor labels.
The workflow asks ``ye3t.couplings.count`` and ``plan`` which
Young/rotation multiplicity spaces actually exist, then ``compile``
materializes the chosen coefficients. This keeps mathematical provenance in
the separate ``ye3t`` dependency.

For a small direct compiler inspection:

.. code-block:: python

   from ye3t.couplings import count, plan

   report = count(content=(1, 1, 2), input_Ls=(0, 0, 0), target_L=0)
   labels = report.labels_for_target(0)
   if labels:
       report.require_label(labels[0], target_L=0)
       coupling_plan = plan(report)
       print(labels[0], coupling_plan)

``ye3t.BasisLabel``, ``to_compact``, ``to_json``, and ``to_latex``
format *resolved* coordinates and preserve known convention metadata. A
compact text label alone does not supply missing Young sectors, coupling
paths, or coefficients. To select ordinary ACE coordinates manually through
``YE3TDescriptors.ace(manual_labels=...)``, use complete labels accepted
by the compiler; do not form independent Cartesian products of radial,
angular, and Young fields.

The compact ``FeatureLabel`` in this package records one actual application
descriptor column and its source channel identities. Its
``feature_index`` is a column ordinal and may differ from a core
multiplicity-copy coordinate.

Physical factor sources and extension boundary
----------------------------------------------

For ordinary density, the physical input to a coupled product is a neighbor
sum of one-factor channels. Schematically, a channel has the form

.. math::

   u_{i\eta lm} = \sum_{j\in\mathcal N(i)}
      C_{\eta}(z_i,z_j) R_{\eta l}(r_{ij})Y_{lm}(\widehat{r}_{ij}).

The configured chemical and radial channels define the physical ``eta``
identity; ``l,m`` are the angular coordinates. The compiler receives those
channel identities and the requested output representation, then supplies
valid product labels and coupling coefficients. A new radial function with
the same channel counts can reuse that representation calculation, but its
values, derivatives, fitted weights, source hash, and saved artifact must be
revalidated. A new tag, role, orbital, or other factor source may also change
the carrier action or physical image; it must declare that action before
compiler count/plan/compile. Ordinary density summation alone has only the
globally trivial Young sector.

The stable configured materializers currently accept PACE ChebExpCos for
ordinary density and shifted Jacobi for tagged physical images, subject to
the capability report of the selected evaluator. The low-level
``SiteBasisV2`` has radial, chemical, charge, and spherical provider hooks
for in-memory research use. A custom radial provider must supply finite,
JSON-serializable convention metadata with a stable provider identity; its
values, derivatives, and normalization bound enter the same source contract.
Process-specific object IDs are rejected as cache identities. An arbitrary
provider is not a portable
``Basis.from_config`` source and cannot be saved or sent to LAMMPS through the
stable model format.
The focused ``test_radial_provider_extension.py`` check injects a radial
provider at this low-level boundary and verifies both edge values and
Cartesian derivatives, including an ``l=3`` finite difference, plus summed
site channels through ``l=3``. It uses
a scaled existing radial source, so it does not certify a new radial family.

To certify another physical source in this interface, its implementation
needs a strict config and source identity, value and analytic derivative
checks at and near the cutoff, a tested backend capability, and a model
writer/reader that reproduces the same channel order without recompilation.
Energy models also need force and stress checks; covariant properties need
rotation, inversion, and physical atom-relabeling checks. Native CPU or
Kokkos availability is declared separately from Python/ASE availability.
This boundary lets source families be added without copying representation
logic into ``ye3t-methods``.
