"""Compact entry points for the existing fixed-descriptor linear models."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from ye3t_ace import YE3TDescriptors, YE3TModel, YE3TRepresentation
from ye3t_ace.ace.linear_ace import (
    LinearACEScalarCalculator,
    LinearACEScalarModelBundle,
    _make_sklearn_estimator,
    load_linear_ace_ase_bundle,
    save_linear_ace_ase_bundle,
)
from ye3t_ace.tagged_cauchy_image import (
    TaggedCauchyImageLinearModel,
    load_tagged_cauchy_image_model,
)
from ye3t_ace.cluster_phi import (
    HybridACEPhiCalculator,
    PhiBranchConfig,
    load_hybrid_ace_phi_ase_bundle,
    phi_motif_coupling_report,
    save_hybrid_ace_phi_ase_bundle,
)
from ye3t_ace.linear_statistics import solve_ridge_statistics
from ye3t_ace.tagged_cauchy_image_fit import tagged_cauchy_image_geometry_row


class FeatureLabel:
    """One public descriptor column; its index is not a multiplicity index."""

    def __init__(self, feature_index, source, identity, details):
        self.feature_index = int(feature_index)
        self.source = str(source)
        self.identity = str(identity)
        self._details = deepcopy(details)

    def as_dict(self):
        return {
            "feature_index": self.feature_index,
            "source": self.source,
            "identity": self.identity,
            **deepcopy(self._details),
        }

    def __str__(self):
        fields = self._details
        if self.source == "density":
            radial = tuple(fields["radial_indices"])
            angular = tuple(fields["input_angular_momenta"])
            return (
                f"[{self.feature_index}] B N={fields['N']} L={fields['L']} "
                f"n={radial} l={angular}"
            )
        if self.source == "bar_phi":
            return (
                f"[{self.feature_index}] barPhi N={fields['N']} L=0 "
                f"motif={fields['motif_name']}"
            )
        raw = fields["compiler_raw_opportunities"]
        tags = sorted({tuple(item["tag_kappa"]) for item in raw})
        roles = sorted({tuple(item["role_kappa"]) for item in raw})
        tag_view = f"{tags[:2]}" + (f" +{len(tags) - 2} more" if len(tags) > 2 else "")
        role_view = f"{roles[:2]}" + (f" +{len(roles) - 2} more" if len(roles) > 2 else "")
        return (
            f"[{self.feature_index}] B N={fields['N']} L={fields['L']} "
            f"tag_kappa={tag_view} role_kappa={role_view} "
            f"({len(raw)} compiler opportunities)"
        )

    def latex(self):
        fields = self._details
        head = f"B_{{{self.feature_index}}}^{{N={fields['N']},L={fields['L']}}}"
        if self.source == "bar_phi":
            return rf"\overline{{\Phi}}_{{{self.feature_index}}}^{{N={fields['N']}}}"
        if self.source != "density":
            return head
        radial = ",".join(str(value) for value in fields["radial_indices"])
        angular = ",".join(str(value) for value in fields["input_angular_momenta"])
        return head + rf"\left[\mathbf{{n}}=({radial}),\boldsymbol{{\ell}}=({angular})\right]"


def _density_labels(specs, elements):
    labels = []
    for index, spec in enumerate(specs):
        label = spec.label
        channels = []
        for channel in spec.channels:
            channels.append({
                "eta": {
                    "central_species": str(elements[int(channel.mu0)]),
                    "neighbor_species": str(elements[int(channel.mu)]),
                    "radial_index": int(channel.n),
                    "central_charge_index": int(channel.kappa0),
                    "neighbor_charge_index": int(channel.kappa),
                },
                "l": int(channel.l),
            })
        labels.append(FeatureLabel(index, "density", str(spec.key), {
            "N": int(label.rank),
            "L": int(spec.L_R),
            "M": int(spec.M_R),
            "radial_indices": tuple(int(value) for value in label.n_tuple),
            "input_angular_momenta": tuple(int(value) for value in label.l_tuple),
            "angular_intermediates": tuple(int(value) for value in label.internal_Ls),
            "one_factor_channels": tuple(channels),
            "compiler_basis_key": deepcopy(label.basis_key),
        }))
    return tuple(labels)


def _tagged_labels(records, tensor_order, raw_labels):
    by_id = {str(item["raw_opportunity_id"]): item for item in raw_labels}
    labels = []
    for index, record in enumerate(records):
        opportunities = tuple(
            deepcopy(by_id[str(item["raw_opportunity_id"])])
            for item in record["contributors"]
        )
        labels.append(FeatureLabel(index, "tagged_cauchy_image", f"tagged-image:{index}", {
            "N": int(tensor_order),
            "L": 0,
            "M": 0,
            "compiler_coordinate_provenance": deepcopy(record),
            "compiler_raw_opportunities": opportunities,
        }))
    return tuple(labels)


def _bar_phi_labels(config):
    labels = []
    for index, spec in enumerate(config.motif_specs):
        report = phi_motif_coupling_report(spec, target_L=0)
        if not report["validation_report"]["passed"]:
            raise ValueError(f"YE3T Phi coupling plan rejected motif {spec.name!r}.")
        labels.append(FeatureLabel(index, "bar_phi", spec.name, {
            "N": int(spec.template.vertex_count), "L": 0, "M": 0,
            "motif_name": spec.name,
            "motif_template": spec.template.to_dict(),
            "one_factor_channels": tuple(ch.to_dict() for ch in spec.channels),
            "slot_orbit_partition": report["slot_orbit_partition"],
            "compiler_coupling_plan": report["plan"],
            "normalization": (
                "weighted_embedding_average" if config.normalize_motif_features
                else "weighted_embedding_sum"
            ),
        }))
    return tuple(labels)


def _bar_phi_feature_rows(model, atoms, type_map, *, forces, stress):
    """Evaluate the existing explicit-motif feature map and its derivatives."""
    dtype = model.config.torch_dtype
    cell0 = torch.as_tensor(np.asarray(atoms.cell.array, float), dtype=dtype)
    if stress:
        if float(atoms.get_volume()) <= 0.0:
            raise ValueError("Phi stress rows require a positive cell volume.")
        cell = cell0.detach().clone().requires_grad_(True)
        scaled = torch.as_tensor(np.asarray(atoms.positions, float), dtype=dtype) @ torch.linalg.inv(cell0)
        positions = scaled @ cell
    else:
        cell = cell0
        positions = torch.tensor(np.asarray(atoms.positions, float), dtype=dtype, requires_grad=bool(forces))
    types = torch.as_tensor(
        [type_map[name] for name in atoms.get_chemical_symbols()], dtype=torch.long,
    )
    pbc = np.asarray(atoms.pbc, dtype=bool)
    model._validate_periodic_cutoff_margin(cell, pbc)
    src, dst, disp, _dist, values = model._edge_data(positions, types, cell=cell, pbc=pbc)
    sites, _payloads, _count, _weights = model._motif_values(positions, src, dst, disp, values, cell=cell, pbc=pbc)
    sums = sites.sum(dim=0)
    energy = np.concatenate(([float(len(atoms))], sums.detach().cpu().numpy()))
    force_rows = np.zeros((len(atoms) * 3, sums.numel() + 1), dtype=float)
    stress_rows = np.zeros((6, sums.numel() + 1), dtype=float)
    for index, feature in enumerate(sums):
        if not feature.requires_grad or not (forces or stress):
            continue
        inputs = (positions, cell) if stress else (positions,)
        gradients = torch.autograd.grad(feature, inputs, retain_graph=True, allow_unused=True)
        if forces and gradients[0] is not None:
            force_rows[:, index + 1] = -gradients[0].detach().cpu().numpy().reshape(-1)
        if stress and gradients[1] is not None:
            cell_gradient = gradients[1].detach().cpu().numpy()
            strain_gradient = np.asarray(atoms.cell.array, float).T @ cell_gradient
            tensor = 0.5 * (strain_gradient + strain_gradient.T) / float(atoms.get_volume())
            stress_rows[:, index + 1] = tensor[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    return energy, force_rows, stress_rows, sites.detach().cpu().numpy()


def _rank_values(value, ranks, name):
    if isinstance(value, int):
        return tuple(int(value) for _ in ranks)
    values = tuple(int(item) for item in value)
    if len(values) != len(ranks):
        raise ValueError(f"{name} must be an integer or have one value per rank.")
    return values


class Basis:
    """Resolved density, scalar tagged, or explicit motif basis."""

    def __init__(self, *, elements, source="density", cutoff, max_rank=None,
                 nmax=4, lmax=2, radial_decay=0.25, tag_counts=None,
                 radial_degrees=None, tensor_order=None, angular_degree=None,
                 backend=None, motif_family="full", motif_specs=None,
                 channels=None, edge_cutoff=None, edge_basis_backend="site_basis",
                 periodic_image_mode="unique", normalize_motif_features=True):
        self.elements = tuple(str(value) for value in elements)
        if not self.elements or len(set(self.elements)) != len(self.elements):
            raise ValueError("elements must be a nonempty unique sequence.")
        self.cutoff = float(cutoff)
        if not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("cutoff must be finite and positive in Angstrom.")
        self.source = str(source)
        if self.source == "density":
            if tag_counts is not None or radial_degrees is not None or tensor_order is not None:
                raise ValueError("Tagged source options require source='tagged_cauchy_image'.")
            rank_count = 3 if max_rank is None else int(max_rank)
            if rank_count < 1:
                raise ValueError("max_rank must be positive.")
            ranks = tuple(range(1, rank_count + 1))
            radial = _rank_values(nmax, ranks, "nmax")
            angular = _rank_values(lmax, ranks, "lmax")
            self.backend = "pytorch" if backend is None else str(backend)
            config = {
                "elements": self.elements,
                "type_map": {name: index for index, name in enumerate(self.elements)},
                "cutoff": self.cutoff,
                "ranks": ranks,
                "nmax": radial,
                "lmax": angular,
                "lmin": (0,) * rank_count,
                "L_R": 0,
                "M_R_values": (0,),
                "basis_type": "no_charge",
                "k_o_max": 0,
                "k_max": (0,) * rank_count,
                "max_labels_per_rank": None,
                "site_basis": {
                    "mode": "explicit",
                    "rc": self.cutoff,
                    "lmbda": float(radial_decay),
                },
                "backend": self.backend,
            }
            self._descriptor = YE3TDescriptors.ace(config)
            self._resolved = {
                "source": self.source, "elements": self.elements,
                "cutoff_A": self.cutoff, "ranks": ranks,
                "nmax": radial, "lmax": angular,
                "radial_decay": float(radial_decay), "backend": self.backend,
            }
            self._labels = _density_labels(self._descriptor.descriptor_specs, self.elements)
        elif self.source == "tagged_cauchy_image":
            if tag_counts is None or radial_degrees is None:
                raise ValueError("Tagged basis requires explicit tag_counts and radial_degrees.")
            if max_rank is not None:
                raise ValueError("Tagged tensor order is specified by tensor_order, not max_rank.")
            order = 4 if tensor_order is None else int(tensor_order)
            angular = 1 if angular_degree is None else int(angular_degree)
            self.backend = "auto" if backend is None else str(backend)
            config = {
                "elements": self.elements,
                "representation": YE3TRepresentation.tagged_cauchy_image(),
                "tagged_cauchy_image": {
                    "tensor_order": order,
                    "selected_raw_tag_counts": tuple(int(v) for v in tag_counts),
                    "radial_degrees": tuple(int(v) for v in radial_degrees),
                    "angular_degree": angular,
                    "cutoff_A": self.cutoff,
                    "coefficient_materialization": "compile",
                },
                "backend": self.backend,
            }
            self._descriptor = YE3TDescriptors.ye3t_basis(config)
            self._resolved = {
                "source": self.source, "elements": tuple(self._descriptor.elements),
                "cutoff_A": self.cutoff, "N": order,
                "tag_counts": tuple(int(v) for v in tag_counts),
                "radial_degrees": tuple(int(v) for v in radial_degrees),
                "angular_degree": angular, "backend": self.backend,
            }
            self.elements = tuple(self._descriptor.elements)
            compiled = self._descriptor.metadata["tagged_cauchy_image_compiled"]
            self._labels = _tagged_labels(
                self._descriptor.feature_labels, order,
                compiled.payload["raw_coordinate_labels"],
            )
        elif self.source == "bar_phi":
            if max_rank is not None or tag_counts is not None or radial_degrees is not None or tensor_order is not None:
                raise ValueError("bar_phi uses explicit motif slots; density and tagged truncations do not apply.")
            self.backend = "pytorch" if backend is None else str(backend)
            if self.backend != "pytorch":
                raise ValueError("bar_phi currently uses the PyTorch reference evaluator.")
            phi = PhiBranchConfig(
                cutoff=self.cutoff,
                edge_cutoff=edge_cutoff,
                channels=() if channels is None else tuple(channels),
                motif_specs=() if motif_specs is None else tuple(motif_specs),
                motif_family=motif_family,
                edge_basis_backend=edge_basis_backend,
                periodic_image_mode=periodic_image_mode,
                normalize_motif_features=normalize_motif_features,
            )
            config = {
                "elements": self.elements,
                "type_map": {name: index for index, name in enumerate(self.elements)},
                "branches": ("bar_phi",),
                "phi": phi.to_dict(),
                "backend": self.backend,
            }
            self._descriptor = YE3TDescriptors.phi(config)
            self._resolved = {
                "source": self.source, "elements": self.elements,
                "cutoff_A": self.cutoff, "backend": self.backend,
                "phi_config": self._descriptor.metadata["phi_config"],
            }
            self._labels = _bar_phi_labels(phi)
        else:
            raise ValueError("source must be 'density', 'tagged_cauchy_image', or 'bar_phi'.")

    @classmethod
    def _from_density_bundle(cls, bundle, cutoff, type_map):
        basis = object.__new__(cls)
        basis.source = "density"
        basis.elements = tuple(bundle.settings.elems)
        basis.cutoff = float(cutoff)
        basis.backend = "pytorch"
        basis._descriptor = None
        basis._resolved = {
            "source": "density", "elements": basis.elements,
            "cutoff_A": basis.cutoff,
            "ranks": tuple(bundle.settings.ranks),
            "nmax": tuple(bundle.settings.nmax),
            "lmax": tuple(bundle.settings.lmax),
            "type_map": dict(type_map),
            "site_basis_config": bundle.site_basis_config,
        }
        basis._labels = _density_labels(bundle.descriptor_specs, basis.elements)
        return basis

    @classmethod
    def _from_tagged_model(cls, model):
        basis = object.__new__(cls)
        basis.source = "tagged_cauchy_image"
        basis.elements = tuple(model.species_order)
        basis.cutoff = float(model.evaluator.cutoff)
        basis.backend = str(model.evaluator.backend)
        basis._descriptor = None
        request = model.evaluator.compiled.plan.report.request
        basis._resolved = {
            "source": basis.source, "elements": basis.elements,
            "cutoff_A": basis.cutoff, "compiler_request": request,
            "compiler_hash": model.evaluator.compiled.self_hash,
        }
        records = model.evaluator.compiled.payload["image_coordinate_provenance"]
        basis._labels = _tagged_labels(
            records, request["tensor_order"],
            model.evaluator.compiled.payload["raw_coordinate_labels"],
        )
        return basis

    @classmethod
    def _from_bar_phi_model(cls, model, type_map):
        basis = object.__new__(cls)
        basis.source = "bar_phi"
        basis.elements = tuple(type_map)
        basis.cutoff = float(model.config.phi.cutoff)
        basis.backend = "pytorch"
        basis._descriptor = None
        basis._resolved = {
            "source": basis.source, "elements": basis.elements,
            "cutoff_A": basis.cutoff, "backend": basis.backend,
            "type_map": dict(type_map), "phi_config": model.config.to_dict(),
        }
        basis._labels = _bar_phi_labels(model.config.phi)
        return basis

    @property
    def labels(self):
        return self._labels

    @property
    def resolved(self):
        return deepcopy(self._resolved)

    def create(self, atoms):
        if self._descriptor is None:
            raise RuntimeError("A loaded model retains label identity; construct a Basis to evaluate standalone descriptors.")
        if self.source == "bar_phi":
            model = YE3TModel.phi(self._descriptor, {"branches": ("bar_phi",)})
            type_map = {name: index for index, name in enumerate(self.elements)}
            return _bar_phi_feature_rows(model, atoms, type_map, forces=False, stress=False)[3]
        return self._descriptor.create(atoms)

    def describe(self, index, format="text"):
        label = self.labels[int(index)]
        if format == "latex":
            return label.latex()
        if format not in {"text", "ascii"}:
            raise ValueError("format must be 'text', 'ascii', or 'latex'.")
        fields = label.as_dict()
        if self.source == "bar_phi":
            plan = fields.pop("compiler_coupling_plan")
            fields["compiler_coupling_summary"] = {
                "api": plan["multiplicity_report"]["provenance"]["api"],
                "convention_hash": plan["convention_hash"],
                "counts_by_target": plan["multiplicity_report"]["counts_by_target"],
            }
        details = json.dumps(fields, sort_keys=True, ensure_ascii=True, default=str)
        if len(details) > 1200:
            details = details[:1200] + f"... ({len(details) - 1200} characters omitted)"
        return str(label) + "\n" + details

    def __str__(self):
        truncation = (
            f"ranks={self._resolved['ranks']}, nmax={self._resolved['nmax']}, "
            f"lmax={self._resolved['lmax']}"
            if self.source == "density" else
            f"motifs={len(self.labels)}, family={self._resolved['phi_config']['phi']['motif_family']}"
            if self.source == "bar_phi" else
            f"N={self._resolved.get('N', self._resolved.get('compiler_request', {}).get('tensor_order'))}, "
            f"tag_counts={self._resolved.get('tag_counts', 'saved')}"
        )
        head = (
            f"Basis(source={self.source}, elements={self.elements}, "
            f"cutoff_A={self.cutoff:g}, {truncation}, features={len(self.labels)})"
        )
        rows = [str(label) for label in self.labels[:5]]
        if len(self.labels) > 5:
            rows.append(f"... {len(self.labels) - 5} rows omitted")
        return "\n".join((head, *rows))

    __repr__ = __str__


def _prepared_training_structures(structures, energy_key, force_key, stress_key,
                                  force_weight, stress_weight):
    prepared = []
    for index, atoms in enumerate(structures):
        if not hasattr(atoms, "get_chemical_symbols"):
            raise TypeError("fit expects a sequence of ASE Atoms objects.")
        results = getattr(getattr(atoms, "calc", None), "results", {}) or {}
        energy = getattr(atoms, "info", {}).get(energy_key, results.get(energy_key))
        if energy is None:
            raise ValueError(f"Structure {index} lacks precomputed {energy_key!r} energy.")
        energy = float(energy)
        if not np.isfinite(energy):
            raise ValueError(f"Structure {index} has a nonfinite {energy_key!r} energy.")
        force = getattr(atoms, "arrays", {}).get(force_key, results.get(force_key))
        if force_weight and force is None:
            raise ValueError(f"Structure {index} lacks precomputed {force_key!r} forces.")
        stress = getattr(atoms, "info", {}).get(stress_key, results.get(stress_key))
        if stress_weight and stress is None:
            raise ValueError(f"Structure {index} lacks precomputed {stress_key!r} stress.")
        clean = atoms.copy()
        clean.calc = None
        clean.info[energy_key] = energy
        if force is not None:
            values = np.asarray(force, dtype=float)
            if values.shape != (len(clean), 3) or not np.isfinite(values).all():
                raise ValueError(f"Structure {index} has invalid force shape or values.")
            clean.arrays[force_key] = values.copy()
        if stress is not None:
            values = np.asarray(stress, dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"Structure {index} has invalid ASE Voigt stress.")
            clean.info[stress_key] = values.copy()
        prepared.append(clean)
    if not prepared:
        raise ValueError("fit requires at least one structure.")
    return prepared


def _fit_sklearn_design(blocks, method, params, column_order):
    """Fit fixed compiler columns and retain the optional ARD covariance."""
    X = np.concatenate([block[0] for block in blocks], axis=0)
    y = np.concatenate([block[1] for block in blocks], axis=0)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("The scikit-learn design contains nonfinite values.")
    estimator = _make_sklearn_estimator(method, sklearn_params=params)
    estimator.fit(X, y)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    if coefficients.shape != (X.shape[1],) or not np.isfinite(coefficients).all():
        raise FloatingPointError("The scikit-learn fit returned invalid coefficients.")
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError("sklearn_params contains a value that cannot be saved portably.")

    metadata = {"fit_method": method, "sklearn_params": plain(dict(params or {})),
                "design_column_order": column_order, "n_rows": int(X.shape[0]),
                "n_cols": int(X.shape[1])}
    if method == "ardregression":
        precision = np.asarray(estimator.lambda_, dtype=np.float64)
        threshold = float(estimator.threshold_lambda)
        active = np.flatnonzero(precision < threshold)
        covariance = np.asarray(estimator.sigma_, dtype=np.float64)
        if covariance.shape != (active.size, active.size):
            raise RuntimeError("ARD posterior covariance does not match active columns.")
        metadata["predictive_uncertainty"] = {
            "schema": "ye3t_linear_ard_posterior_v1",
            "status": "python_offline_only",
            "design_column_order": column_order,
            "active_column_indices": active.tolist(),
            "coefficient_precision": precision.tolist(),
            "coefficient_covariance_active": covariance.tolist(),
            "noise_precision": float(estimator.alpha_),
            "threshold_lambda": threshold,
            "variance_formula": "x_active @ sigma @ x_active.T (epistemic readout only)",
        }
    return coefficients, metadata


class LinearModel:
    """Fit, persist, inspect, and evaluate a fixed YE3T descriptor readout."""

    def __init__(self, basis, *, reference_energies=None):
        if not isinstance(basis, Basis):
            raise TypeError("LinearModel requires a Basis.")
        if basis.source == "bar_phi" and reference_energies:
            raise ValueError("The existing bar_phi model has no per-species reference-energy term.")
        self.basis = basis
        self.reference_energies = dict(reference_energies or {})
        self._fitted = None

    @property
    def labels(self):
        return self.basis.labels

    def fit(self, structures, *, regularization=1e-8, fit_method="ridge",
            sklearn_params=None, energy_weight=1.0,
            force_weight=1.0, stress_weight=0.0, energy_key="energy",
            force_key="forces", stress_key="stress"):
        if self.basis._descriptor is None:
            raise RuntimeError("Fit requires a constructed Basis, not one reloaded from an artifact.")
        if self.basis.source == "density" and stress_weight:
            raise ValueError("Density ACE fitting does not support stress rows in this interface.")
        prepared = _prepared_training_structures(
            structures, energy_key, force_key, stress_key, force_weight, stress_weight,
        )
        method = str(fit_method).strip().lower()
        if method == "ard":
            method = "ardregression"
        if method not in {"ridge", "linear_regression", "ols", "lasso", "ridgecv", "ardregression"}:
            raise ValueError("fit_method must be ridge, linear_regression, lasso, ridgecv, or ardregression.")
        if method == "ridge" and sklearn_params:
            raise ValueError("sklearn_params requires a scikit-learn fit_method.")
        if method != "ridge" and regularization != 1e-8:
            raise ValueError("For scikit-learn methods set regularization through sklearn_params.")
        weights = (float(energy_weight), float(force_weight), float(stress_weight))
        if any(not np.isfinite(value) or value < 0.0 for value in weights) or not any(weights):
            raise ValueError("Fit weights must be finite, nonnegative, and not all zero.")
        if self.basis.source == "bar_phi":
            if not np.isfinite(regularization) or regularization < 0.0:
                raise ValueError("regularization must be finite and nonnegative.")
            fitted = YE3TModel.phi(self.basis._descriptor, {"branches": ("bar_phi",)})
            width = len(self.labels) + 1
            gram = np.zeros((width, width), dtype=float)
            rhs = np.zeros(width, dtype=float)
            blocks = []
            type_map = {name: index for index, name in enumerate(self.basis.elements)}
            for atoms in prepared:
                energy_row, force_rows, stress_rows, _sites = _bar_phi_feature_rows(
                    fitted, atoms, type_map, forces=bool(force_weight), stress=bool(stress_weight),
                )
                if energy_weight:
                    gram += energy_weight * np.outer(energy_row, energy_row)
                    rhs += energy_weight * energy_row * float(atoms.info[energy_key])
                    if method != "ridge":
                        blocks.append((np.sqrt(energy_weight) * energy_row[None, :],
                                       np.sqrt(energy_weight) * np.asarray([atoms.info[energy_key]])))
                if force_weight:
                    gram += force_weight * (force_rows.T @ force_rows)
                    rhs += force_weight * (force_rows.T @ np.asarray(atoms.arrays[force_key]).reshape(-1))
                    if method != "ridge":
                        blocks.append((np.sqrt(force_weight) * force_rows,
                                       np.sqrt(force_weight) * np.asarray(atoms.arrays[force_key]).reshape(-1)))
                if stress_weight:
                    gram += stress_weight * (stress_rows.T @ stress_rows)
                    rhs += stress_weight * (stress_rows.T @ np.asarray(atoms.info[stress_key]))
                    if method != "ridge":
                        blocks.append((np.sqrt(stress_weight) * stress_rows,
                                       np.sqrt(stress_weight) * np.asarray(atoms.info[stress_key])))
            if method == "ridge":
                coefficients = solve_ridge_statistics(
                    {"gram": gram, "rhs": rhs, "runtime_from_fit_coordinates": np.eye(width)},
                    float(regularization),
                )["runtime_coefficients"]
            else:
                coefficients, fitted.fit_metadata = _fit_sklearn_design(
                    blocks, method, sklearn_params, "atom_count_bias_then_descriptor_features",
                )
            if not np.isfinite(coefficients).all():
                raise FloatingPointError("bar_phi fit returned nonfinite coefficients.")
            with torch.no_grad():
                fitted.bar_phi_bias.copy_(torch.as_tensor([coefficients[0]], dtype=fitted.config.torch_dtype))
                fitted.bar_phi_weight.copy_(torch.as_tensor(coefficients[1:], dtype=fitted.config.torch_dtype))
            self._fitted = fitted
            return self
        if self.basis.source == "tagged_cauchy_image" and method != "ridge":
            evaluator = self.basis._descriptor.metadata["tagged_cauchy_image_evaluator"]
            species = tuple(evaluator.species_order)
            if self.reference_energies and set(self.reference_energies) != set(species):
                raise ValueError("Tagged reference energies must cover every model species.")
            feature_count = int(evaluator.feature_count)
            if feature_count != len(self.labels):
                raise RuntimeError("Tagged evaluator width differs from the Basis labels.")
            beta_width = len(species) * feature_count
            blocks = []
            for atoms in prepared:
                row = tagged_cauchy_image_geometry_row(self.basis._descriptor, atoms)
                energy_row = np.concatenate((row["feature_sums"].reshape(-1), row["species_counts"]))
                zero_offsets = np.zeros((len(atoms) * 3, len(species)))
                if energy_weight:
                    reference = sum(self.reference_energies.get(name, 0.0)
                                    for name in atoms.get_chemical_symbols())
                    blocks.append((np.sqrt(energy_weight) * energy_row[None, :],
                                   np.asarray([np.sqrt(energy_weight) *
                                               (float(atoms.info[energy_key]) - reference)])))
                if force_weight:
                    blocks.append((np.sqrt(force_weight) * np.column_stack((row["force_design"], zero_offsets)),
                                   np.sqrt(force_weight) * np.asarray(atoms.arrays[force_key]).reshape(-1)))
                if stress_weight:
                    blocks.append((np.sqrt(stress_weight) * np.column_stack((
                        row["stress_design"], np.zeros((6, len(species))))),
                        np.sqrt(stress_weight) * np.asarray(atoms.info[stress_key])))
            coefficients, metadata = _fit_sklearn_design(
                blocks, method, sklearn_params, "species_major_descriptor_features_then_species_offsets",
            )
            beta = coefficients[:beta_width].reshape(len(species), feature_count)
            fitted = TaggedCauchyImageLinearModel(
                evaluator, {name: beta[index] for index, name in enumerate(species)},
                {name: coefficients[beta_width + index] for index, name in enumerate(species)},
            )
            if self.reference_energies:
                fitted.reference_terms = {"atomic_energies": dict(self.reference_energies)}
            fitted.fit_metadata = metadata
            fitted._ye3t_linear_fit_metadata = dict(metadata)
            self._fitted = fitted
            return self
        config = {
            "ridge_alpha": float(regularization),
            "fit_method": ("ridge_normal_equations" if method == "ridge" else method)
                          if self.basis.source == "density" else "ridge_streaming_gram",
            "energy_weight": float(energy_weight),
            "force_weight": float(force_weight),
            "energy_key": str(energy_key),
            "force_key": str(force_key),
            "reference_energies": dict(self.reference_energies),
        }
        if method != "ridge":
            config["sklearn_params"] = dict(sklearn_params or {})
        if self.basis.source == "tagged_cauchy_image":
            config["stress_weight"] = float(stress_weight)
            config["stress_key"] = str(stress_key)
            config["restore_references"] = bool(self.reference_energies)
        fitted = YE3TModel.linear(self.basis._descriptor, config, structures=prepared)
        if self.basis.source == "density":
            actual = tuple(str(spec.key) for spec in fitted.descriptor_specs)
            expected = tuple(label.identity for label in self.labels)
            if actual != expected:
                raise RuntimeError("Fitted descriptor columns differ from the Basis label order.")
        elif int(fitted.evaluator.feature_count) != len(self.labels):
            raise RuntimeError("Fitted tagged image width differs from the Basis labels.")
        self._fitted = fitted
        return self

    def ase_calculator(self, *, backend=None, **kwargs):
        if self._fitted is None:
            raise RuntimeError("Fit or read a model before constructing an ASE calculator.")
        if self.basis.source == "density":
            return LinearACEScalarCalculator(
                self._fitted, self.basis.cutoff,
                self.basis._resolved.get(
                    "type_map", {name: index for index, name in enumerate(self.basis.elements)}
                ),
                reference_energies=self.reference_energies,
                backend="pytorch" if backend is None else backend,
                **kwargs,
            )
        if self.basis.source == "bar_phi":
            if backend not in (None, "pytorch", "reference"):
                raise ValueError("bar_phi ASE currently uses the PyTorch reference evaluator.")
            return HybridACEPhiCalculator(
                self._fitted, type_map=self.basis._resolved.get(
                    "type_map", {name: index for index, name in enumerate(self.basis.elements)}
                ), **kwargs,
            )
        return self._fitted.ase_calculator(
            backend="reference" if backend is None else backend, **kwargs,
        )

    def predict_uncertainty(self, atoms):
        """Return ARD readout standard deviations in eV for each site and total E.

        These are posterior coefficient uncertainties conditional on the fixed
        descriptor map. They exclude model error and observation noise.
        """
        if self._fitted is None:
            raise RuntimeError("Fit or read a model before predicting uncertainty.")
        metadata = dict(getattr(self._fitted, "fit_metadata", {}) or {})
        posterior = metadata.get("predictive_uncertainty")
        if not isinstance(posterior, dict) or posterior.get("schema") != "ye3t_linear_ard_posterior_v1":
            raise ValueError("predict_uncertainty requires a saved ARDRegression posterior.")
        if self.basis.source == "density":
            calculator = self.ase_calculator(backend="pytorch")
            pos, cell, edge_index, shifts, atom_types = calculator._geometry_from_atoms(
                atoms, requires_grad=False,
            )
            displacement = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
            with torch.no_grad():
                features = calculator.evaluator(
                    x_ij=displacement, edge_index=edge_index, atom_types=atom_types,
                    descriptors=self._fitted.descriptor_specs, real_if_scalar=True,
                ).detach().cpu().numpy()
            design = np.column_stack((features, np.ones(len(atoms))))
            order = "descriptor_features_then_atom_count_bias"
        elif self.basis.source == "bar_phi":
            type_map = self.basis._resolved.get(
                "type_map", {name: index for index, name in enumerate(self.basis.elements)},
            )
            sites = _bar_phi_feature_rows(
                self._fitted, atoms, type_map, forces=False, stress=False,
            )[3]
            design = np.column_stack((np.ones(len(atoms)), sites))
            order = "atom_count_bias_then_descriptor_features"
        else:
            evaluator = self._fitted.evaluator
            symbols = atoms.get_chemical_symbols()
            atom_types = np.asarray([evaluator.type_map[name] for name in symbols], dtype=int)
            with torch.no_grad():
                features = evaluator.materialize(
                    torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64),
                    torch.as_tensor(atom_types, dtype=torch.long),
                    cell=np.asarray(atoms.cell.array), pbc=np.asarray(atoms.pbc),
                )[2].detach().cpu().numpy()
            width = len(evaluator.species_order) * (evaluator.feature_count + 1)
            design = np.zeros((len(atoms), width), dtype=np.float64)
            for index, species_index in enumerate(atom_types):
                start = species_index * evaluator.feature_count
                design[index, start:start + evaluator.feature_count] = features[index]
                design[index, len(evaluator.species_order) * evaluator.feature_count + species_index] = 1.0
            order = "species_major_descriptor_features_then_species_offsets"
        if posterior.get("design_column_order") != order:
            raise ValueError("ARD posterior column order does not match this model.")
        active = np.asarray(posterior["active_column_indices"], dtype=int)
        covariance = np.asarray(
            posterior["coefficient_covariance_active"], dtype=np.float64,
        ).reshape((len(active), len(active)))
        if (covariance.shape != (len(active), len(active)) or
                np.any(active < 0) or np.any(active >= design.shape[1]) or
                len(np.unique(active)) != len(active) or not np.isfinite(covariance).all()):
            raise ValueError("Saved ARD posterior has invalid column indices or covariance.")
        selected = design[:, active]
        atomic_variance = np.einsum("if,fg,ig->i", selected, covariance, selected)
        total_row = selected.sum(axis=0)
        total_variance = float(total_row @ covariance @ total_row)
        if min(float(np.min(atomic_variance)), total_variance) < -1e-10:
            raise ValueError("Saved ARD posterior has a negative predictive variance.")
        return {
            "atomic_energy_std_eV": np.sqrt(np.maximum(atomic_variance, 0.0)),
            "total_energy_std_eV": float(np.sqrt(max(total_variance, 0.0))),
            "kind": "conditional_linear_readout_epistemic",
        }

    def write(self, path):
        if self._fitted is None:
            raise RuntimeError("Fit or read a model before writing it.")
        target = Path(path)
        if not target.suffix:
            suffix = {"density": ".pt", "tagged_cauchy_image": ".ye3t.json", "bar_phi": ".phi.pt"}[self.basis.source]
            target = target.with_suffix(suffix)
        if self.basis.source == "density":
            if target.suffix != ".pt":
                raise ValueError("Density bundles use a .pt path.")
            save_linear_ace_ase_bundle(
                self._fitted, target, cutoff=self.basis.cutoff,
                type_map=self.basis._resolved.get(
                    "type_map", {name: index for index, name in enumerate(self.basis.elements)}
                ),
                reference_energies=self.reference_energies,
            )
        elif self.basis.source == "bar_phi":
            if not target.name.endswith(".phi.pt"):
                raise ValueError("Explicit Phi bundles use a .phi.pt path.")
            save_hybrid_ace_phi_ase_bundle(
                target, self._fitted,
                type_map=self.basis._resolved.get(
                    "type_map", {name: index for index, name in enumerate(self.basis.elements)}
                ),
            )
        else:
            if target.suffix != ".json":
                raise ValueError("Tagged bundles use a .ye3t.json path.")
            self._fitted.export_lammps(target)
        return target

    @classmethod
    def read(cls, path):
        target = Path(path)
        if not target.exists() and not target.suffix:
            candidates = (target.with_suffix(".pt"), target.with_suffix(".ye3t.json"), target.with_suffix(".phi.pt"))
            existing = [candidate for candidate in candidates if candidate.exists()]
            if len(existing) != 1:
                raise FileNotFoundError(f"Expected exactly one model artifact for {target!s}.")
            target = existing[0]
        if target.name.endswith(".phi.pt"):
            fitted, type_map = load_hybrid_ace_phi_ase_bundle(target)
            if tuple(fitted.branches) != ("bar_phi",):
                raise ValueError("The compact linear Phi reader accepts only the bar_phi branch.")
            model = cls(Basis._from_bar_phi_model(fitted, type_map))
        elif target.suffix == ".pt":
            fitted, cutoff, type_map, refs = load_linear_ace_ase_bundle(target)
            model = cls(Basis._from_density_bundle(fitted, cutoff, type_map), reference_energies=refs)
        elif target.suffix == ".json":
            fitted = load_tagged_cauchy_image_model(target)
            model = cls(Basis._from_tagged_model(fitted))
        else:
            raise ValueError("Expected a .pt density or .ye3t.json tagged model.")
        model._fitted = fitted
        return model

    def export_lammps(self, path):
        if self._fitted is None:
            raise RuntimeError("Fit or read a model before export.")
        if self.basis.source == "bar_phi":
            raise ValueError("The explicit Phi reference model has no LAMMPS export schema.")
        if self.basis.source == "density":
            return self._fitted.export_lammps(path, elements=self.basis.elements)
        target = Path(path)
        self._fitted.export_lammps(target)
        return target

    def describe(self, index, format="text"):
        label = self.labels[int(index)]
        if format == "latex":
            return label.latex()
        detail = self.basis.describe(index, format=format)
        if self._fitted is None:
            return detail + "\ncoefficient: unfitted"
        if self.basis.source == "density":
            coefficient = float(np.asarray(self._fitted.weight)[int(index)])
        elif self.basis.source == "bar_phi":
            coefficient = float(self._fitted.bar_phi_weight.detach().cpu()[int(index)])
        else:
            coefficient = {
                name: float(self._fitted.beta_by_species[name][int(index)])
                for name in self.basis.elements
            }
        return detail + f"\ncoefficient: {coefficient}"

    def __str__(self):
        state = "fitted" if self._fitted is not None else "unfitted"
        head = (
            f"LinearModel(source={self.basis.source}, state={state}, "
            f"features={len(self.labels)}, elements={self.basis.elements})"
        )
        rows = [str(label) for label in self.labels[:5]]
        if len(self.labels) > 5:
            rows.append(f"... {len(self.labels) - 5} rows omitted")
        return "\n".join((head, *rows))

    __repr__ = __str__
