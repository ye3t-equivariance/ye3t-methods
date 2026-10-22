"""Scalar shell-role features shared with the canonical message-passing source.

The energy readout uses S_q-trivial, L=0, even multiplets. Internal Cauchy
Young types remain compiler-selected. No quadratic neural readout is inserted.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from ye3t import couplings, YE3TRepresentation
from .config import BasisResolution, CoupledFactorCataloguePreview, _keys, _rank_map
from .atomistic.tagged_cauchy_carriers import _TaggedCauchyOccurrenceSource


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def resolve_shell_basis(config, representation, runtime):
    """Resolve the scalar physical shell image before compiling coefficients."""
    if not isinstance(representation, YE3TRepresentation):
        raise TypeError("Shell basis needs ye3t.YE3TRepresentation.from_config(...).")
    _keys(config, {"single_factors", "tensor_product", "catalogue"}, "basis")
    _keys(config["tensor_product"], {"kind"}, "tensor_product")
    factors = config["single_factors"]
    _keys(factors, {"species", "radial", "chemical"}, "single_factors")
    species = tuple(factors["species"])
    if not species or any(not isinstance(x, str) or not x for x in species) or len(set(species)) != len(species):
        raise ValueError("Shell species must be a nonempty unique sequence.")
    if factors.get("chemical", {"kind": "explicit"}) != {"kind": "explicit"}:
        raise ValueError("Shell linear models currently require explicit chemistry.")
    radial = factors["radial"]
    _keys(radial, {"family", "cutoff_A", "shells", "orthogonalization"}, "radial")
    cutoff = float(radial["cutoff_A"])
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Shell cutoff_A must be finite and positive.")
    if (representation.group != "O3" or representation.subspace != "full"
            or representation.factorization != "cauchy" or representation.L != 0 or representation.parity != "even"):
        raise ValueError("Shell linear energy requires Cauchy factorization, L=0 and even parity.")
    if any(representation.parent_partition(n) != (n,) for n in representation.ranks):
        raise ValueError("Commuting shell products require the symmetric complete-factor parent.")
    if representation.hidden_sectors is not None:
        raise ValueError("A fixed linear basis has no hidden sectors.")
    catalogue = deepcopy(config["catalogue"])
    _keys(catalogue, {"ranks", "nmax_per_rank", "lmax_per_rank", "max_source_blocks",
        "source_block_partitions_by_rank", "shell_occupancies_by_rank"}, "catalogue")
    ranks = tuple(catalogue["ranks"])
    if ranks != tuple(representation.ranks):
        raise ValueError("Shell catalogue ranks must match representation.ranks.")
    nmax = _rank_map(catalogue["nmax_per_rank"], ranks, "nmax_per_rank", 1)
    lmax = _rank_map(catalogue["lmax_per_rank"], ranks, "lmax_per_rank", 0)
    capacity = representation.to_dict()["uncoupled_factor_inputs"]
    for rank in ranks:
        if (nmax[rank] > capacity["eta_count_per_rank"][rank]
                or lmax[rank] > capacity["l_max_per_rank"][rank]):
            raise ValueError("Shell catalogue exceeds representation input capacity; update uncoupled_factor_inputs to cover its radial and angular channels.")
    _keys(runtime, {"device", "dtype", "evaluator", "neighbors", "compiled_cache_dir"}, "runtime")
    if runtime.get("dtype", "float64") != "float64":
        raise ValueError("Shell linear fitting uses float64.")
    device = str(runtime.get("device", "cpu"))
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("Shell device must be cpu or cuda.")
    evaluator = runtime.get("evaluator", "auto")
    if evaluator not in {"auto", "native", "torch"} or runtime.get("neighbors", "ase") != "ase":
        raise ValueError("Shell runtime uses auto/native/torch evaluation and ASE neighbors.")
    source = couplings.shell_filtered_source_plan(radial["family"], max(nmax.values()),
        max(lmax.values()), radial["shells"], radial.get("orthogonalization"),
        evaluation_coordinates="physical_shells")
    selection = representation.to_dict().get("source_sectors")
    if selection is None:
        selection = {"young_lambda": "symmetric_only", "L": [0], "parity": "even"}
    from ye3t.couplings.sector_selection import normalize_sector_selection
    selection = normalize_sector_selection(selection)
    if (selection["young_lambda"] != "symmetric_only" or selection["parity"] != "even"
            or selection.get("L", [selection.get("L_max")]) != [0]):
        raise ValueError("Linear scalar energy needs shell-symmetric L=0 even source sectors; internal Young types are selected by intermediates.")
    catalogue.update(source_family_id=source["family_id"], source_family_plan=source,
        basis_coordinates="formal_cauchy", shell_role_action="permutation",
        tag_counts=[0], input_Lmax=0, source_sectors=selection,
        intermediate_selection=representation.to_dict()["intermediates"])
    request = couplings.tagged_cauchy_carriers_request(catalogue=catalogue, species=species)
    report = couplings.count(request)
    if not report["multiplet_count"] or report["physical_image_rank"] != report["multiplet_count"]:
        raise ValueError("Shell basis needs a nonempty independent physical scalar image.")
    payload = {"source": "shell_density", "species": species, "cutoff_A": cutoff,
        "representation": representation.to_dict(), "request": request,
        "runtime": {"device": device, "dtype": "float64", "evaluator": evaluator, "neighbors": "ase"},
        "components": [{"name": "shell_density", "kind": "shell_density"}],
        "warnings": [], "capability_report": {"basis_create_available": True,
            "requested_neighbors": "ase", "component_count_status": {"shell_density": "exact_compiler_count"}},
        "multiplet_count": report["multiplet_count"]}
    encoded = _bytes(payload)
    return BasisResolution(encoded.decode(), hashlib.sha256(encoded).hexdigest()), report


def initialize_shell_basis(basis, config, representation, runtime):
    resolution, report = resolve_shell_basis(config, representation, runtime)
    basis.source = "shell_density"
    basis.elements = tuple(resolution.to_dict()["species"])
    basis.cutoff = resolution.to_dict()["cutoff_A"]
    basis.backend = runtime.get("evaluator", "auto")
    basis._resolution = resolution
    basis._resolved = resolution.to_dict()
    basis._catalogue = CoupledFactorCataloguePreview(report)
    basis._construction = {"basis": deepcopy(config), "representation": representation.to_dict(),
                           "runtime": deepcopy(runtime)}
    basis._descriptor = None
    basis._labels = ()
    basis._shell_compiled = None
    return basis


def materialize_shell_basis(basis):
    if basis._descriptor is not None:
        return
    from .linear import FeatureLabel

    compiled = couplings.compile(basis._resolved["request"],
        cache_dir=basis._construction["runtime"].get("compiled_cache_dir"))
    plan = couplings.tagged_cauchy_carrier_model_plan(compiled, sector_policy="tagged_mixed")
    if len(plan["schedules"]) != 1:
        raise ValueError("Shell energy must have one zero-tag source schedule.")
    inventory = plan["schedules"][0]["inventory"]
    labels = []
    for row in inventory:
        label = row["label"]
        if (label["target_L"] != 0 or label["target_parity"] != 1
                or tuple(label["shell_partition"]) != (label["shell_role_degree"],)
                or row["component_slice"][1] - row["component_slice"][0] != 1):
            raise ValueError("Shell scalar coefficient cannot weight a non-invariant multiplet.")
        labels.append(FeatureLabel(len(labels), "shell_density", label["coordinate_id"],
            {"N": label["rank"], "L": 0, "parity": "even", "compiler_label": label}))
    if len(labels) != basis._resolved["multiplet_count"]:
        raise ArithmeticError("Shell coefficient count differs from the independent plan.")
    device = basis._resolved["runtime"]["device"]
    source = _TaggedCauchyOccurrenceSource(plan["schedules"], basis.elements, basis.cutoff,
        backend="reference" if basis.backend == "torch" else "native", dtype=torch.float64).to(device)
    basis._descriptor = source
    basis._labels = tuple(labels)
    basis._shell_compiled = compiled


def shell_geometry(basis, atoms, derivatives=False, stress=False, coefficients=None):
    """Evaluate canonical site features and their total-energy design rows."""
    materialize_shell_basis(basis)
    if len(atoms) == 0 or set(atoms.get_chemical_symbols()) - set(basis.elements):
        raise ValueError("Shell structures must be nonempty and use the configured species.")
    if stress and atoms.get_volume() <= 0:
        raise ValueError("Shell stress requires a positive cell volume.")
    device = basis._resolved["runtime"]["device"]
    positions = torch.tensor(atoms.positions, dtype=torch.float64, device=device, requires_grad=derivatives)
    types = torch.tensor([basis.elements.index(s) for s in atoms.get_chemical_symbols()], device=device)
    i, j, shifts = neighbor_list("ijS", atoms, basis.cutoff)
    edges = torch.tensor(np.array([i, j]), dtype=torch.long, device=device)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float64, device=device)
    displacement = positions[edges[1]] - positions[edges[0]] + torch.tensor(shifts, dtype=torch.float64, device=device) @ cell
    if len(displacement) and bool((torch.linalg.vector_norm(displacement, dim=-1) == 0).any()):
        raise ValueError("Coincident occurrences are invalid for shell angular factors.")
    strain = torch.zeros((3, 3), dtype=torch.float64, device=device, requires_grad=stress)
    displacement = displacement @ (torch.eye(3, dtype=torch.float64, device=device) + strain)
    source = basis._descriptor
    edge, density = source.geometry_values(displacement, types, edges, len(atoms))
    support = {"centers": torch.arange(len(atoms), device=device),
               "tag_edges": torch.empty((len(atoms), 0), dtype=torch.long, device=device)}
    values = source(0, edge, density, support, normalized=False)
    sums = torch.stack([values[types == s].sum(0) for s in range(len(basis.elements))]).reshape(-1)
    if coefficients is not None:
        # Inference contracts the linear readout before reverse differentiation.
        # Force fitting differentiates each column to construct its design rows.
        sums = (sums @ torch.as_tensor(coefficients, dtype=sums.dtype, device=sums.device)).reshape(1)
    result = {"values": values.detach().cpu().numpy(), "energy": sums.detach().cpu().numpy()}
    if derivatives or stress:
        variables = ([positions] if derivatives else []) + ([strain] if stress else [])
        force_rows, stress_rows = [], []
        for component in sums:
            gradients = torch.autograd.grad(component, variables, retain_graph=True)
            if derivatives:
                force_rows.append(-gradients[0].detach().cpu().numpy().reshape(-1))
            if stress:
                matrix = gradients[-1].detach().cpu().numpy() / atoms.get_volume()
                matrix = (matrix + matrix.T) / 2
                stress_rows.append(matrix[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]])
        if derivatives:
            result["forces"] = np.stack(force_rows, axis=1)
        if stress:
            result["stress"] = np.stack(stress_rows, axis=1)
    if any(not np.isfinite(row).all() for row in result.values()):
        raise FloatingPointError("Shell design rows are nonfinite.")
    return result


def fit_shell_model(model, frames, regularization, method, sklearn_params,
                    weights, keys, fit_E0):
    from .linear import _fit_sklearn_design

    basis = model.basis
    materialize_shell_basis(basis)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    if any(not np.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError("Fit weights must be finite and nonnegative with a positive target.")
    if fit_E0 is None:
        fit_E0 = not bool(model.reference_energies)
    if type(fit_E0) is not bool or (fit_E0 and not weights[0]):
        raise ValueError("fit_E0 must be boolean and requires energy targets.")
    species = basis.elements
    offsets = {s: float(model.reference_energies.get(s, 0)) for s in species}
    if set(model.reference_energies) - set(species) or not all(np.isfinite(list(offsets.values()))):
        raise ValueError("Invalid shell reference energies.")
    counts = np.array([[atoms.get_chemical_symbols().count(s) for s in species] for atoms in frames], dtype=float)
    if fit_E0 and np.linalg.matrix_rank(counts) != len(species):
        raise ValueError("Fitting species offsets requires independent training compositions.")
    nf = len(species) * len(basis.labels)
    width = nf + (len(species) if fit_E0 else 0)
    gram, rhs = np.zeros((width, width)), np.zeros(width)
    blocks, nrows = [], 0
    for atoms, composition in zip(frames, counts):
        rows = shell_geometry(basis, atoms, bool(weights[1]), bool(weights[2]))
        for name, weight, key in zip(("energy", "forces", "stress"), weights, keys):
            if not weight:
                continue
            design = rows[name].reshape(-1, nf)
            target = np.asarray(atoms.arrays[key] if name == "forces" else atoms.info[key]).reshape(-1)
            if name == "energy":
                target = target - composition @ np.array(list(offsets.values()))
            if fit_E0:
                extra = composition[None, :] if name == "energy" else np.zeros((len(target), len(species)))
                design = np.column_stack((design, extra))
            design, target = np.sqrt(weight) * design, np.sqrt(weight) * target
            gram += design.T @ design
            rhs += design.T @ target
            nrows += len(target)
            if method != "ridge":
                blocks.append((design, target))
    if method == "ridge":
        penalty = np.r_[np.full(nf, regularization), np.zeros(width-nf)]
        beta = np.linalg.lstsq(gram + np.diag(penalty), rhs, rcond=None)[0]
        metadata = {"fit_method": method, "alpha": regularization, "n_rows": nrows, "n_cols": width}
    else:
        beta, metadata = _fit_sklearn_design(blocks, method, sklearn_params, "species_then_compiler_coordinates_then_offsets")
    if beta.shape != (width,) or not np.isfinite(beta).all():
        raise FloatingPointError("Invalid shell linear coefficients.")
    model.reference_energies = {s: offsets[s] + (float(beta[nf+k]) if fit_E0 else 0.) for k, s in enumerate(species)}
    metadata.update(fit_E0=fit_E0, compiler_hash=basis._shell_compiled["self_hash"],
        design_column_order="species_then_compiler_coordinates", weights=dict(zip(("energy", "forces", "stress"), weights)),
        coordinate_penalty_metric="euclidean_in_saved_compiler_coordinates")
    model._fitted = {"beta": beta[:nf], "fit_metadata": metadata}
    return model


class _ShellLinearCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        beta = self.model._fitted["beta"]
        rows = shell_geometry(self.model.basis, atoms, True, bool(atoms.cell.rank == 3), coefficients=beta)
        energy = float(rows["energy"][0]) + sum(self.model.reference_energies.get(s, 0) for s in atoms.get_chemical_symbols())
        self.results = {"energy": energy, "free_energy": energy, "forces": rows["forces"].reshape(-1, 3)}
        if "stress" in rows:
            self.results["stress"] = rows["stress"][:, 0]


def shell_calculator(model, device=None, evaluator=None):
    """Choose inference hardware while preserving source and coefficient identity."""
    from .linear import Basis, LinearModel
    runtime = deepcopy(model.basis._construction["runtime"])
    if device is not None:
        runtime["device"] = device
    if evaluator is not None:
        runtime["evaluator"] = evaluator
    if runtime != model.basis._construction["runtime"]:
        construction = model.basis._construction
        basis = Basis.from_config(construction["basis"],
            representation=YE3TRepresentation.from_config(construction["representation"]), runtime=runtime)
        selected = LinearModel(basis, reference_energies=model.reference_energies)
        selected._fitted = model._fitted
        model = selected
    return _ShellLinearCalculator(model)


def write_shell_model(model, path):
    materialize_shell_basis(model.basis)
    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(".ye3t.json")
    if not target.name.endswith(".ye3t.json"):
        raise ValueError("Shell linear models use .ye3t.json files.")
    construction = deepcopy(model.basis._construction)
    construction["runtime"].pop("compiled_cache_dir", None)
    payload = {"schema": "ye3t_methods_shell_linear_v1", "construction": construction,
        "resolution_sha256": model.basis._resolution.sha256,
        "compiler_sha256": model.basis._shell_compiled["self_hash"],
        "compiled_catalogue": model.basis._shell_compiled,
        "coordinate_ids": [x.identity for x in model.labels],
        "beta": np.asarray(model._fitted["beta"]).tolist(),
        "reference_energies": model.reference_energies,
        "fit_metadata": deepcopy(model._fitted["fit_metadata"])}
    resolved_fit = payload["fit_metadata"].get("resolved_fit_config")
    if resolved_fit is not None:
        resolved_fit["construction"]["runtime"].pop("compiled_cache_dir", None)
        payload["fit_metadata"]["resolved_fit_config_sha256"] = hashlib.sha256(_bytes(resolved_fit)).hexdigest()
    payload["self_hash"] = hashlib.sha256(_bytes(payload)).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_bytes(payload) + b"\n")
    return target


def read_shell_model(payload):
    from .linear import Basis, LinearModel

    if set(payload) != {"schema", "construction", "resolution_sha256", "compiler_sha256", "compiled_catalogue",
                        "coordinate_ids", "beta", "reference_energies", "fit_metadata", "self_hash"}:
        raise ValueError("Shell saved model has unsupported or missing fields.")
    body = {k: v for k, v in payload.items() if k != "self_hash"}
    if hashlib.sha256(_bytes(body)).hexdigest() != payload.get("self_hash"):
        raise ValueError("Shell model hash mismatch.")
    construction = deepcopy(payload["construction"])
    basis = Basis.from_config(construction["basis"],
        representation=YE3TRepresentation.from_config(construction["representation"]), runtime=construction["runtime"])
    saved_resolution = basis.resolution.sha256
    # Verify the saved training/runtime identity before choosing portable CPU
    # inference. CUDA can be selected explicitly on the ASE calculator.
    if construction["runtime"].get("device", "cpu") != "cpu":
        construction["runtime"]["device"] = "cpu"
        basis = Basis.from_config(construction["basis"],
            representation=YE3TRepresentation.from_config(construction["representation"]), runtime=construction["runtime"])
    materialize_shell_basis(basis)
    if (saved_resolution != payload["resolution_sha256"]
            or basis._shell_compiled["self_hash"] != payload["compiler_sha256"]
            or _bytes(basis._shell_compiled) != _bytes(payload["compiled_catalogue"])
            or [x.identity for x in basis.labels] != payload["coordinate_ids"]):
        raise ValueError("Shell saved source, coordinates or compiler identity differs from canonical construction.")
    beta = np.asarray(payload["beta"], dtype=float)
    offsets = payload["reference_energies"]
    if (beta.shape != (len(basis.elements)*len(basis.labels),) or not np.isfinite(beta).all()
            or set(offsets) != set(basis.elements) or not np.isfinite(list(offsets.values())).all()):
        raise ValueError("Invalid saved shell weights or species offsets.")
    model = LinearModel(basis, reference_energies=offsets)
    model._fitted = {"beta": beta, "fit_metadata": deepcopy(payload["fit_metadata"])}
    return model
