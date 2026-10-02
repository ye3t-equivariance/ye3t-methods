# Linear release validation and migration record

Status: **release candidate for public upload; index upload not performed**. This
record covers the fixed-feature application split only. The inherited
projection and polynomial approximation modules have been removed from the
release source. The pre-removal source is preserved with a SHA-256 inventory
under the sibling workflows archive.
The source repository has been published to GitHub. The author clarified that
“flag” in the release brief meant the tagged route listed below; it does not
name a separate linear family.

## Final cleanup gate

The release now contains exact linear density, tagged, Phi, lifted-density,
and lifted-Cauchy paths. Input validation rejects active unsupported options
in serialized descriptor and site-basis settings. The removed research source
was saved outside this package before editing.

Focused source regressions passed for compact density, tagged, Phi,
lifted-density fitting, lifted-Cauchy fitting, strict YACE export, exact
compiled artifacts, paper-script imports, tagged runtime, native source basis,
and input validation. The saved-model ASE and depth-two Phi quickstarts passed
both from the source checkout and from the extracted source archive against an
isolated wheel installation. The 16-page Sphinx build passed with `-n -W`.

The source archive and wheel were built from the same extracted source tree.
Their package files are byte-identical. The archive audit found no excluded
research topics, private paths, key markers, or agent instruction files. The
source archive retains the paper workflows and data; its verifier accepted all
84 promoted model artifacts and six mlearn data files. Archive hashes and the
audit report live in the sibling workflows directory, separate from the
distribution itself.

The rank-eight nontrivial parent example validates a compiler plan; it does
not evaluate that parent descriptor on atomic geometry. The missing geometry
binding is marked as a TODO in the example and explained in the docs. Full
six-element refitting, MPI/GPU replay, and long molecular dynamics reruns were
deferred; the promoted paper artifacts and prior logs remain the evidence for
those calculations.

## Final source pass

The short saved-model ASE script, exact one-hot channel-selection example,
four-vertex explicit motif example, rank-three parent coefficient view, and
rank-eight parent coupling plan were executed in `pa-sw-e3`. The parent
examples use validated core couplers; neither computes a nontrivial global
parent descriptor from atomistic geometry. The role-density slot sector is a
different group action and is not used as a substitute. The channel-selection
example removes excluded species channels; the fixed embedding example mixes
species but does not yet shrink compiled width.

Twelve focused source tests passed across compact density, tagged, Phi,
role-density fitting, strict YACE export, compiled artifact evaluation, and
paper-script imports. The six-element artifact verifier accepted all 84 model
files and six data files. Sphinx completed all 16 pages with `-n -W`. The
rebuilt archive inventory under `../ye3t-workflows/split_linear_release/`
records file counts, archive hashes, optional fit and neighbor metadata, and
the remaining source-scope gate. Expensive refitting and full native replay
were not repeated in this short pass.

## Paper example and chemical encoding follow-up

The source distribution now includes
`examples/publication/cost_comparison`: the six-element Li/Mo/Cu/Ni/Si/Ge
paper fitting configuration and workflow, the fixed mlearn snapshot, and the
byte-identical promoted LAMMPS model/input/log bundle from `ye3t-lammps`.
The wheel remains code-only; source examples are run after extracting the
source archive. `examples/data/mlearn/LICENSE.mlearn` retains the upstream
BSD-3-Clause dataset notice. The model files were not modified. The source
archive verifier checked all 84 promoted model artifacts against the
per-element SHA-256 manifests and all six mlearn data files against
`MANIFEST.json`.

The paper input config uses the compact rank/cap schedule and fixed-channel
tagged requests; `ye3t.couplings` supplies valid labels and coefficients.
The published elemental models use one-hot chemical channels
(`chemical_basis="delta"`). A new quickstart demonstrates the existing
low-level `chemical_provider` hook with a fixed embedding Gram kernel.
It checked that the identity embedding reproduces one-hot values and that a
nontrivial fixed kernel transforms both source values and source position
derivatives as expected. This provider remains in-memory only: the compact
fit/save/load path and LAMMPS export do not serialize arbitrary custom
chemical providers, and no trainable embedding method is claimed.

Executed on the new source copy in WSL `pa-sw-e3`: Li paper preflight and
dataset preparation; verification from the extracted sdist; the chemical
example against the installed wheel; Sphinx `-n -W` from that sdist; and
single-rank CPU LAMMPS step-zero runs of the copied Li `ace_127` and
`ye3t_tagged_127` inputs. Both LAMMPS runs reproduced the reference log's
potential energy and six pressure components exactly. The LAMMPS smoke files
and generated preparation data remain under
`../ye3t-workflows/split_linear_release` and
`../ye3t-workflows/MLIP/mlearn_linear_cost_comparison`. Full six-element
refitting, numerical-difference reruns, MPI/GPU replay, and 10,000-step NVE
replay were not repeated for this packaging follow-up; existing promoted
logs remain evidence of the prior study.

An installed-wheel audit initially found copied workflow imports of legacy
`ye3t_ace` top-level convenience names that are absent from the stable
compatibility surface. Those eight scripts now import the retained
`ye3t_ace.ace.linear_ace` and `ye3t_ace.ace.lammps_export` functions
directly. A focused regression checks the package imports declared by the
paper scripts, and all 17 workflow drivers completed their `--help` load
path against the installed wheel. The focused
`test_compact_linear.py` plus `test_paper_example_imports.py` run passed
(4 tests). An initial pytest invocation from the workspace parent failed
because that directory's sibling `ye3t/` folder shadowed the installed
`ye3t` import as a namespace package; running pytest from the
`ye3t-methods` package root used the intended installed core and passed.

The local wheel was installed successfully in a separate WSL virtual
environment with dependency resolution against installed `ye3t==0.1.0`.
An offline `pip --target` attempt with only the methods wheel failed, as
expected, because that wheelhouse did not contain `ye3t`; this confirms
the declared external dependency rather than bundling core code. Public
package-index availability was not established and no index upload was made.
The updated artifact hashes and source/wheel inventories are recorded
outside the self-referential archive in
`../ye3t-workflows/split_linear_release/paper_artifact_inventory.json`.

## ASE evaluator and native-neighbor follow-up

The compact ASE choices are now tabulated in `docs/evaluators.rst` and shown
in the editable quickstart configs. The paper's exact tagged-plus-ACE
composite has a standalone C++ ASE loader; the independent ACE control is a
`.yace` deployment file and has no direct compact `LinearModel.read` route.
The paper models are residuals to a ZBL overlay. The standalone C++ ASE loader
evaluates the residual; the paper ASE example combines it with
`YE3TZBLCalculator.from_model_manifest` for the full potential. The 16-atom Li validation cell lies
outside that cutoff and was compared directly to the retained LAMMPS step-zero
energy: absolute error `3.6e-13` eV with the source-side adapter.

The optional `matscipy` neighbor-list route was added for geometries where
the native ASE adapter cannot use SciPy's cKDTree. With `matscipy 1.2.0`
installed, the 16/128/432-atom Li neighbor lists had the same complete
`(center, neighbor, cell shift)` multiset as ASE. The 16-atom neighbor rebuild
median was `0.089` ms with matscipy and `2.893` ms with ASE; this is a
neighbor-construction comparison, not a full force-call speedup. SciPy's
cKDTree remains the selected path for the 128- and 432-atom orthorhombic
cells. If matscipy is absent, the adapter uses ASE as before. The
`ye3t-methods` wheel does not depend on matscipy.

For the exact Li 127-feature tagged composite on this WSL CPU environment,
the median of seven cached-topology ASE force evaluations was `0.973` ms
(16 atoms), `4.612` ms (128 atoms), and `14.100` ms (432 atoms). The
published single-rank LAMMPS Li tagged NVE log records `3.81069` seconds
for 10,000 steps with 16 atoms, or `0.381` ms per full MD step. Its retained
432-atom timing log records `2.84552` seconds for 150 steps, or `18.970` ms
per step. The ASE values are force calls with tiny position changes and cached
topology; the LAMMPS values are complete MD steps from different runs, so the
ratios are descriptive rather than controlled speedups. Model initialization
was about `0.8` seconds in the probe and is excluded from the warm timings.
The separate `pytorch` probe loaded an older 61-descriptor Li ACE `.pt`
bundle and measured `92` ms for 16 atoms and `163` ms for 128 atoms; it is
not the exact 127-feature published ACE control and must not be used as a
head-to-head model comparison. Raw timings and parity are under
`../ye3t-workflows/split_linear_release/ase_linear_probe_results.json` and
`neighbor_probe_results.json`.

The optional neighbor API follows the official `matscipy.neighbours`
documentation at <https://libatoms.github.io/matscipy/generated/matscipy.neighbours.html>.
Only the public API was called; no external implementation was copied.

## ASE ZBL and optional scikit-learn follow-up

``YE3TZBLCalculator.from_model_manifest`` verifies the promoted model's
LAMMPS ZBL record and evaluates atomic energies, energy, forces, and stress
through the existing portable analytic reference. The paper ASE example now
uses ASE ``SumCalculator`` to combine this overlay with the resident C++
linear residual. On four isolated Li pairs at 1.40, 1.80, 2.05, and 2.20 Å,
the maximum absolute ASE-versus-LAMMPS difference was ``5.6e-17`` eV in
energy and ``8.9e-16`` eV/Å in a force component. The 16-atom paper geometry
has zero ZBL energy; the full ASE potential differs from the retained LAMMPS
step-zero energy by ``3.6e-13`` eV. The parity source data are under
``../ye3t-workflows/split_linear_release/zbl_parity_probe_results.json``.

The optional ``fit`` extra installs scikit-learn. Compact density, tagged,
and Phi models now accept ``fit_method="lasso"`` or ``"ardregression"``
with ``sklearn_params``; the latter saves its coefficient posterior and
provides ``LinearModel.predict_uncertainty(atoms)`` after reload. The result
is per-atom and total-energy *linear-readout epistemic* standard deviation,
conditional on the fixed descriptor basis. It is not a calibrated physical
error estimate or a force/stress uncertainty. The optional ``neighbors``
extra installs matscipy for the general-cell native ASE neighbor path.

The focused density, tagged, Phi, ZBL, native ASE, and low-level ARD regression
group passed ``13`` tests. The group includes all-pruned ARD posterior reload
and NumPy-valued RidgeCV parameter reload. The source-side scikit quickstart and full paper ASE example
executed. Sphinx 9.0.4 built all 16 pages with ``-n -W`` and no warnings.

## Identity and ownership

| Item | Identity and disposition |
| --- | --- |
| Compiler/runtime | `ye3t` at `de35c81d794a9c73efec9c1724b3594b59ac54b8`; unchanged. All coupling labels, plans, and coefficients originate there. |
| Original application | `ye3t-ace` at `ab48bad179dafd3aeeb7b63d55f33364c5142107`, with pre-existing staged ERI evidence, modified Phi graph and moment-star files, and untracked notes. Left intact. |
| New stable application | `ye3t-methods` 0.1.0, imported as `ye3t_methods`; compatibility `ye3t_ace` modules keep established Torch/JSON class and schema identities. |
| LAMMPS consumer | `ye3t-lammps` at `09fee77c5c21c9783d84f8355eaa515ce8ead12a`; unchanged. |
| Deferred research | `../ye3t-experimental/preserved`; 1,978 copied files / 402,537,993 bytes verified against the source, plus 348 raw training files / 315,345,705 bytes retained at their original paths with hashes. See `preservation_manifest.json` and `checksums.sha256`. Preserved; outside this release; intentionally not imported, installed, built, or tested. |

The stable wheel/source archive must contain no `ye3t_experimental`,
`ye3t_ace.ml`, `ye3t_ace.nn`, fermion/electronic operator, or
multipole application source/assets. The stable tests run
with the original editable application finder removed and the experimental
tree absent from the import path. Package file names and SHA-256 hashes are in
`../ye3t-workflows/split_linear_release/final_20261002/artifact_inventory.json` in this
workspace; that external inventory avoids a self-referential archive hash.

## Current linear capability and verification

| Family and entry point | Implementation | Executed verification | Limit |
| --- | --- | --- | --- |
| Ordinary density `A` and coupled scalar `B`: `Basis(source="density")`, `LinearModel` | Compact fit, optional LASSO/ARD, saved ARD posterior, `.pt` save/load, ASE energy/forces, labels, strict representable `.yace` export | Manufactured coefficient recovery; one-hot two-column identity, force finite difference, translation/rotation/atom reindexing, relocated load; native compiled CPU/CUDA reference comparison and strict YACE tests; LASSO shrinkage and ARD covariance/roundtrip | Compact fit rejects stress rows. The default explicit radial source fails closed for strict PACE export. |
| Compiler tagged physical image: `Basis(source="tagged_cauchy_image")` | Compact fit, optional LASSO/ARD, saved ARD posterior, hash-bound `.ye3t.json`, ASE energy/forces/stress, native CPU C ABI and LAMMPS export | `tag_kappa=(1,1)` with `role_kappa=(2,1,1)` appears in actual contributing compiler records; one-hot public columns; fit/load; shear finite strain; general multispecies, cutoff and ZBL tests; LAMMPS CPU one/two-rank energy/forces/virial and Kokkos CUDA single-rank parity; LASSO and ARD save/reload | Kokkos consumer calls itself `experimental_reference_unqualified`; its binary lacks EXTRA-FIX numerical-difference commands. |
| Explicit cluster `bar_phi`: `Basis(source="bar_phi")` | Fixed motif coupling plans, compact fit, optional LASSO/ARD, saved ARD posterior, `.phi.pt`, ASE energy/forces/stress | Two motifs, one-hot column, fit/load, finite shear strain and force comparison; LASSO and ARD save/reload | PyTorch evaluator only; no LAMMPS schema or per-species reference offset in this model. |
| Filtered role density `A_s` | Retained descriptor-first `YE3TModel.linear` path | Manufactured slot-trivial A_s energy/force fit and reference evaluation | The verified filtered slot-trivial route is mainly a radial-channel change or expansion; compare radial-matched ordinary ACE before attributing gains to role symmetry. No compact `Basis` adapter or LAMMPS claim for this route. |
| Lifted Cauchy scalar linear model | Retained fitting, direct/factorized source, native bundle export/load | Full retained linear regression file including source VJP, geometry symmetry, fitting, tamper rejection; exported bundle consumed by LAMMPS CPU with energy/force/virial and finite-difference comparison | Compact `Basis` adapter not supplied. |
| Saved fixed Young descriptor sets | Retained fixed-feature evaluator | Nontrivial copy-aware scalar column is finite, nonzero and physical-atom-reindex invariant | Low-level evaluator only; no new fit/artifact wrapper. |

The current high-level API is `Basis -> LinearModel.fit -> write/read ->
ase_calculator`. The original `YE3TRepresentation -> YE3TDescriptors ->
YE3TModel` flow remains for retained low-level linear families. Established
serialized schema keys and public coordinate order remain unchanged.

## Paper notation and labels

Local manuscript identities were recorded from `main.pdf` (SHA-256
`d93fdb01ef319ae0204dd3c7572b36f6939e9d7709fd19b5247e0e8fc5fe5d7e`),
local `supplemental.pdf` (SHA-256
`00c58b0959a96dd31fd8bfbeb50f009ccd1c928472e1d027d2a1163a94916ccc`),
and the public 2026-09-25 arXiv v1 text
<https://arxiv.org/abs/2609.31895>. Notation was checked against the public
text and the brief's author decisions. The public version's title differs
from the local PDF metadata, so full local PDF alignment remains unverified.

| Meaning | Paper symbol | Stable field and display | Index scope / legacy caution |
| --- | --- | --- | --- |
| Single-interaction input channel | `\phi_{ij,\eta l m}`, `\nu=(\eta,l)` | `one_factor_channels[].eta`, `.l`; density label shows radial `n` and input `l` | `eta` retains species, radial, and charge components; radial `n` is not `kappa`. |
| Density and uncoupled product | `A`, `Z` | Existing site basis and compiler product metadata | `Z` is not a public fitted column by itself. |
| Coupled feature | `B_{i\alpha}` | `FeatureLabel.identity`, `compiler_basis_key` or compiler coordinate provenance; text/LaTeX use `B` | `feature_index` is zero-based public column, not paper multiplicity copy `a` or carrier component `t`. |
| Explicit motif | `\Phi`; normalized motif view `\overline{\Phi}` | `source="bar_phi"`, `motif_name`, motif template and plan | `N` counts motif slots, not automatically physical body order. |
| Tensor rank / parent angular component | `N`, `L`, `M` | `N`, `L`, `M` fields | Scalar release labels use `L=M=0` where evaluated. |
| Tag/role partitions | Paper block `\kappa_b`, parent `\lambda` when compiler map proves them | `tag_kappa`, `role_kappa` retained under `compiler_raw_opportunities` | The fields are compiler partitions, never silently renamed to parent `lambda`; a public image coordinate may combine several opportunities. |

Canonical `\nu^\circ`, block `b`/`k_b`, intermediate `\Lambda_b`, multiplicity
copy `a`, component `t`, and a full composite `\alpha` are not fabricated for
coordinates lacking those resolved metadata fields. Physical neighbor order
and tensor-factor permutations remain separate. Text truncation changes only
presentation, never identity or coefficient order.

## Test disposition and observations

Retained in stable tests: `test_linear_ace_scaled_ridge`,
`test_native_source_basis`, `test_tagged_cauchy_image_runtime`,
`test_tagged_cauchy_general_workflow`, `test_lifted_cauchy_linear`,
`test_linear_statistics`, `test_energy_reference_fit`,
`test_reference_potentials`, and `test_yace_semantics`, with their required
fixtures. New focused tests cover compact density/tagged/Phi, A_s fitting,
saved Young evaluation, strict YACE export, compiled native execution, and
stable import isolation. The original mixed `test_cluster_phi`,
`test_lifted_density`, and `test_linear_young_character` are preserved in the
original and experimental staging tree; stable linear behavior is represented
by focused replacement tests. Original research tests are preserved assets and
were intentionally not collected or run.

The lifted Cauchy origin-derivative test failed identically in the original
checkout: the source masked a supplied zero-length edge before its regular
solid harmonic derivative. The stable copy now retains that derivative while
the neighbor builder excludes self edges. The exact test and the full retained
linear file then passed. No test was weakened or skipped to hide the failure.

The stable suite is run in WSL `pa-sw-e3`, with PyTorch native CPU/CUDA
available and the conda compiler wrappers activated. The source-side suite,
native tests, and LAMMPS runs are recorded under
`../ye3t-workflows/split_linear_release`. The final installed artifact run and
exact commands/results are included in the external artifact inventory. The
quickstart Cu2/Ta3/H3 data are deterministic manufactured labels, not physical
potential validation. Experimental application training and test execution
were intentionally outside this release.

## Bundled yaml-cpp build

The native source archive now includes upstream yaml-cpp 0.8.0 under
`native/third_party/yaml-cpp` with its MIT license. The default build links it
statically. A CMake build with `CMAKE_PREFIX_PATH=/nonexistent` passed, and
both Li paper ASE examples retained their prior energies. A platform wheel
built from the source archive installed and ran both examples using its
packaged library, without an explicit library path. `readelf` found no
`libyaml-cpp` dependency or build-machine RPATH in that installed library.
The source archive and wheel both retain the MIT license notice.
With `YE3T_METHODS_BUILD_NATIVE=0`, pip also produced a pure Python wheel
without a shared library; that setting skips the C++ build.
