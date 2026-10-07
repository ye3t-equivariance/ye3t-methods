
r"""Linear ACE fitting utilities with optional primitive-basis pruning."""

import hashlib
from dataclasses import field
import json
import math
import os
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from ye3t_methods.atomistic._record import recordclass
try:
    from sklearn.linear_model import ARDRegression, Lasso, LinearRegression, Ridge, RidgeCV
except Exception:  # pragma: no cover - optional dependency
    ARDRegression = None  # type: ignore
    Lasso = None  # type: ignore
    LinearRegression = None  # type: ignore
    Ridge = None  # type: ignore
    RidgeCV = None  # type: ignore
try:
    from ase import Atoms
    from ase.io import read
    from ase.calculators.calculator import Calculator, all_changes
    from ase.stress import full_3x3_to_voigt_6_stress
except Exception:  # pragma: no cover - optional dependency
    Atoms = object  # type: ignore
    read = None  # type: ignore
    full_3x3_to_voigt_6_stress = None  # type: ignore
    class Calculator:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("ASE is required for LinearACEScalarCalculator")
    all_changes = object()

from ye3t_methods.atomistic.cache import DescriptorBuildCache
from ye3t_methods.atomistic.equivariant_calc.ace_eval_v2 import ACECovariantEvaluator, GeneralizedCouplingLibrary
from ye3t_methods.atomistic.equivariant_calc.descriptor_sets import (
    DescriptorGenerationSettings,
    build_descriptor_specs_from_settings,
    compile_descriptor_artifacts,
    count_channel_variants,
    count_channel_variants_by_center,
    enumerate_compact_labels,
    normalize_basis_mode,
)
from ye3t_methods.atomistic.equivariant_calc.labeling import CompactLabel, DescriptorSpec, normalize_compact_label
from ye3t_methods.atomistic.equivariant_calc.models import LinearScalarACEModel
from ye3t_methods.atomistic.equivariant_calc.neighbors import neighbor_data_from_ase_atoms
from ye3t_methods.atomistic.equivariant_calc.gradients import (
    descriptor_linear_form_position_vjp,
    descriptor_position_vjp,
    descriptor_sum_position_jacobian_analytic_product,
    descriptor_weighted_sum_position_vjp,
)
from ye3t_methods.atomistic.equivariant_calc.cy_factor_product import CYFactorProductEvaluator
from ye3t_methods.atomistic.equivariant_calc.site_basis_v2 import SiteBasisConfig
from ye3t_methods.atomistic.equivariant_calc.site_basis_serialization import deserialize_site_basis_config, serialize_site_basis_config
from ye3t_methods.atomistic.ace.yace import YACEFunction, write_yace
from ye3t_methods.atomistic.utils.fit_weights import structure_fit_weights
from ye3t_methods.atomistic.workflow_config import select_workflow_frames


def _normalize_reference_energies(reference_energies=None):
    if reference_energies is None:
        return {}
    return {str(key): float(value) for key, value in dict(reference_energies).items()}


def _reference_energy_offset_from_atoms(atoms, reference_energies):
    refs = _normalize_reference_energies(reference_energies)
    if not refs:
        return 0.0
    return float(sum(refs[str(symbol)] for symbol in atoms.get_chemical_symbols()))


def _add_reference_energy_to_site_energies(atoms, site_energy, reference_energies):
    refs = _normalize_reference_energies(reference_energies)
    if not refs:
        return site_energy
    values = np.asarray(site_energy, dtype=float).copy()
    for index, symbol in enumerate(atoms.get_chemical_symbols()):
        values[index] += refs[str(symbol)]
    return values


def _linear_ace_profile_enabled():
    raw = os.environ.get("YE3T_ACE_PROFILE_LINEAR_ACE_MD", os.environ.get("YE3T_ACE_PROFILE_FULL_MODEL", "0"))
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _profile_start_tensor(tensor):
    if torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return time.perf_counter()


def _profile_stop_tensor(profile, key, start, tensor):
    if profile is None:
        return
    if torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    profile[key] = float(profile.get(key, 0.0)) + float(time.perf_counter() - start)


def _resolve_torch_device(device = None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return resolved


def _atom_type_indices_from_atoms(atoms, type_map):
    return np.asarray([type_map[s] for s in atoms.get_chemical_symbols()], dtype=int)


def _neighbor_cache_is_valid(cache, positions, cell, atom_types, cutoff, skin):
    if cache is None or float(skin) <= 0.0:
        return False
    if abs(float(cache.get("cutoff", -1.0)) - float(cutoff)) > 1.0e-12:
        return False
    if np.asarray(cache["positions"]).shape != np.asarray(positions).shape:
        return False
    if not np.array_equal(np.asarray(cache["atom_types"], int), np.asarray(atom_types, int)):
        return False
    if not np.allclose(np.asarray(cache["cell"], float), np.asarray(cell, float), rtol=1.0e-12, atol=1.0e-12):
        return False
    displacement = np.asarray(positions, float) - np.asarray(cache["positions"], float)
    if displacement.size == 0:
        return True
    max_displacement = float(np.linalg.norm(displacement, axis=1).max())
    return max_displacement <= 0.5 * float(skin)


@recordclass(('settings', 'site_basis_config', 'descriptor_specs', 'weight', 'bias', 'basis_mode', 'fit_method', 'fit_metadata'))
class LinearACEScalarModelBundle:
    basis_mode = None
    fit_method = None
    fit_metadata = field(default_factory=dict)

    @property
    def n_features(self):
        return int(len(self.descriptor_specs))

    def export_lammps(self, path, *, elements, format="yace"):
        """Write a stock PACE model when the scalar basis has an exact YACE lowering."""

        if format != "yace":
            raise ValueError("Ordinary scalar ACE export requires format='yace'.")
        return export_scalar_bundle_to_yace(
            self, path, elements=elements, compatibility="lammps_pace_linear_v1")

    def symmetric_power_kernel_options(
        self,
        *,
        rank = None,
        target_l_avs = (1,),
        strict_max_li = None,
        max_labels = 8,
        min_power = 2,
        max_power = 8,
        include_term_counts = False,
    ):
        """Return optional symmetric-power kernels for repeated linear-ACE blocks."""
        from ye3t_methods.atomistic.schedules import build_symmetric_power_kernel_summary_from_exhaustive_labels

        if rank is None:
            ranks = tuple(int(value) for value in self.settings.ranks)
            rank = max(ranks) if ranks else 2
        return build_symmetric_power_kernel_summary_from_exhaustive_labels(
            source="linear_ace_descriptor",
            rank=int(rank),
            target_l_avs=tuple(target_l_avs),
            strict_max_li=strict_max_li,
            max_labels=max_labels,
            min_power=int(min_power),
            max_power=int(max_power),
            include_term_counts=bool(include_term_counts),
        )

    def build_symmetric_power_feature_projector(
        self,
        *,
        rank = None,
        target_l_avs = (1,),
        strict_max_li = None,
        max_labels = 8,
        min_power = 2,
        max_power = 8,
        include_term_counts = False,
        output_selector = "max",
        max_groups = 1,
        optimization_policy = "auto",
    ):
        """Build an optional symmetric-power feature projector for linear ACE."""
        from ye3t_methods.atomistic.symmetric_power import build_symmetric_power_feature_projector

        summary = self.symmetric_power_kernel_options(
            rank=rank,
            target_l_avs=target_l_avs,
            strict_max_li=strict_max_li,
            max_labels=max_labels,
            min_power=min_power,
            max_power=max_power,
            include_term_counts=include_term_counts,
        )
        return build_symmetric_power_feature_projector(
            summary,
            output_selector=output_selector,
            max_groups=max_groups,
            optimization_policy=optimization_policy,
        )


@recordclass(('groups', 'type_map', 'fit_metadata'))
class LinearACEMultiCutoffModelBundle:
    fit_metadata = field(default_factory=dict)

    @property
    def n_features(self):
        return int(sum(len(group["bundle"].descriptor_specs) for group in self.groups))


def load_xyz_structures(xyz_path, indices=None):
    """Load all XYZ frames, or only the requested non-negative frame indices."""
    if read is None:
        raise ImportError("ASE is required for loading extxyz structures")
    if indices is None:
        structures = list(read(str(xyz_path), index=':'))
    else:
        requested = tuple(int(index) for index in indices)
        if not requested:
            raise ValueError("At least one XYZ frame index is required.")
        if any(index < 0 for index in requested):
            raise ValueError("XYZ frame indices must be non-negative.")
        if len(set(requested)) != len(requested):
            raise ValueError("XYZ frame indices must be unique.")
        ordered = sorted(requested)
        runs = []
        start = ordered[0]
        stop = start + 1
        for index in ordered[1:]:
            if index == stop:
                stop += 1
            else:
                runs.append((start, stop))
                start = index
                stop = index + 1
        runs.append((start, stop))
        selected = {}
        for start, stop in runs:
            batch = list(read(str(xyz_path), index=slice(start, stop)))
            if len(batch) != stop - start:
                raise IndexError(
                    f"XYZ frame interval [{start}, {stop}) is outside {xyz_path!s}."
                )
            selected.update(zip(range(start, stop), batch))
        structures = [selected[index] for index in requested]
    if not structures:
        raise ValueError(f"No structures were found in {xyz_path!s}.")
    return structures


class LinearACEScalarCalculator(Calculator):
    """ASE calculator backed by a fitted scalar exact-ACE linear model."""

    implemented_properties = ["energy", "free_energy", "forces", "energies", "stress"]

    def __init__(
        self,
        bundle,
        cutoff,
        type_map,
        *,
        force_method = "autograd",
        device = None,
        backend = "pytorch",
        strict_backend = False,
        validate_backend = True,
        factorized_descriptor_runtime_policy = None,
        neighbor_backend = "ase",
        neighbor_skin = 0.0,
        reference_energies = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = _resolve_torch_device(device)
        self.backend = str(backend)
        self.strict_backend = bool(strict_backend)
        self.validate_backend = bool(validate_backend)
        bundle_policy = dict(bundle.fit_metadata or {}).get(
            "factorized_descriptor_runtime_policy",
            None,
        )
        self.factorized_descriptor_runtime_policy = (
            bundle_policy
            if factorized_descriptor_runtime_policy is None
            else str(factorized_descriptor_runtime_policy)
        )
        self.neighbor_backend = str(neighbor_backend)
        self.neighbor_skin = max(float(neighbor_skin), 0.0)
        self._neighbor_cache = None
        self.bundle = bundle
        self.cutoff = float(cutoff)
        self.type_map = dict(type_map)
        self.reference_energies = _normalize_reference_energies(reference_energies)
        self.evaluator = ACECovariantEvaluator(
            bundle.site_basis_config,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            factorized_descriptor_runtime_policy=(
                self.factorized_descriptor_runtime_policy
            ),
        ).to(device=self.device)
        self.descriptor_compile_report = self.evaluator.precompile_descriptors(bundle.descriptor_specs)
        self.weight = torch.as_tensor(bundle.weight, dtype=torch.float64, device=self.device)
        self.bias = torch.as_tensor(bundle.bias, dtype=torch.float64, device=self.device)
        self._last_force_profile = {}
        self._last_stress_profile = {}
        method = str(force_method).strip().lower().replace("-", "_")
        aliases = {
            "analytical": "analytic_factorized",
            "analytic": "analytic_factorized",
            "analytic_direct": "analytic",
            "direct": "analytic",
            "analytical_dag": "analytic_dag",
            "streaming": "analytic_streaming",
            "analytical_streaming": "analytic_streaming",
            "analytical_dag_streaming": "analytic_dag_streaming",
            "fused": "analytic_factorized_fused",
            "fused_factorized": "analytic_factorized_fused",
            "fused_streaming": "analytic_factorized_fused_streaming",
            "fused_factorized_streaming": "analytic_factorized_fused_streaming",
            "pace": "analytic_factorized_fused_streaming",
            "pace_streaming": "analytic_factorized_fused_streaming",
            "product_fused": "analytic_factorized_fused",
            "product_fused_streaming": "analytic_factorized_fused_streaming",
            "factorized": "analytic_factorized",
            "sym_power": "analytic_factorized",
            "symmetric_power": "analytic_factorized",
        }
        method = aliases.get(method, method)
        if method not in {
            "autograd",
            "analytic",
            "analytic_dag",
            "analytic_streaming",
            "analytic_dag_streaming",
            "analytic_factorized",
            "analytic_factorized_streaming",
            "analytic_factorized_fused",
            "analytic_factorized_fused_streaming",
        }:
            raise ValueError(
                "force_method must be 'autograd', 'analytic', 'analytic_direct', 'analytic_dag', "
                "'analytic_streaming', 'analytic_dag_streaming', 'analytic_factorized', or "
                "'analytic_factorized_fused'"
            )
        self.force_method = method

    def _neighbor_data_from_atoms(self, atoms):
        positions = np.asarray(atoms.positions, float)
        cell = np.asarray(atoms.cell.array, float)
        atom_types = _atom_type_indices_from_atoms(atoms, self.type_map)
        build_cutoff = self.cutoff + self.neighbor_skin
        if _neighbor_cache_is_valid(
            self._neighbor_cache,
            positions,
            cell,
            atom_types,
            build_cutoff,
            self.neighbor_skin,
        ):
            return self._neighbor_cache["neighbor_data"]
        nbr = neighbor_data_from_ase_atoms(
            atoms,
            build_cutoff,
            self.type_map,
            backend=self.neighbor_backend,
        )
        if self.neighbor_skin > 0.0:
            self._neighbor_cache = {
                "positions": positions.copy(),
                "cell": cell.copy(),
                "atom_types": atom_types.copy(),
                "cutoff": float(build_cutoff),
                "neighbor_data": nbr,
            }
        return nbr

    def backend_report(self):
        """Return descriptor-cache and backend-path provenance for the linear ACE calculator."""
        report = dict(self.evaluator.backend_report())
        report["descriptor_compile_report"] = dict(self.descriptor_compile_report)
        report["coupling_source"] = str(self.bundle.fit_metadata.get("coupling_source", "unknown"))
        report["coupling_cache_enabled"] = bool(self.bundle.fit_metadata.get("coupling_cache_enabled", False))
        report["cg_products_constructed_in_forward"] = False
        report["runtime_path"] = "cached_coupling_payloads_and_precompiled_descriptor_tables"
        report["linear_form_profile"] = dict(getattr(self.evaluator, "_last_linear_form_profile", {}))
        report["force_profile"] = dict(self._last_force_profile)
        report["stress_profile"] = dict(self._last_stress_profile)
        return report

    def symmetric_power_kernel_options(self, **kwargs):
        """Return optional symmetric-power kernels for this fitted linear ACE model."""
        return self.bundle.symmetric_power_kernel_options(**kwargs)

    def build_symmetric_power_feature_projector(self, **kwargs):
        """Build an optional symmetric-power feature projector for this model."""
        return self.bundle.build_symmetric_power_feature_projector(**kwargs)

    def _geometry_from_atoms(self, atoms, *, requires_grad):
        pos = torch.as_tensor(np.asarray(atoms.positions, float), dtype=torch.float64, device=self.device)
        if requires_grad:
            pos = pos.detach().clone().requires_grad_(True)
        nbr = self._neighbor_data_from_atoms(atoms)
        shifts = torch.as_tensor(np.asarray(nbr.shifts, float), dtype=torch.float64, device=self.device)
        cell = torch.as_tensor(np.asarray(atoms.cell.array, float), dtype=torch.float64, device=self.device)
        edge_index = torch.as_tensor(nbr.edge_index, dtype=torch.long, device=self.device)
        atom_types = torch.as_tensor(nbr.atom_types, dtype=torch.long, device=self.device)
        if self.neighbor_skin > 0.0 and edge_index.numel() > 0:
            x_ij = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
            mask = torch.linalg.norm(x_ij, dim=1) <= (self.cutoff + 1.0e-12)
            edge_index = edge_index[:, mask]
            shifts = shifts[mask]
        return pos, cell, edge_index, shifts, atom_types

    def _energy_from_atoms(self, atoms):
        pos, cell, edge_index, shifts, atom_types = self._geometry_from_atoms(atoms, requires_grad=True)
        x_ij = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
        B = self.evaluator(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=self.bundle.descriptor_specs,
            real_if_scalar=True,
        )
        site_energy = B @ self.weight + self.bias
        return site_energy.sum(), pos, site_energy

    def _energy_from_geometry(self, pos, cell, edge_index, shifts, atom_types):
        x_ij = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
        B = self.evaluator(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=self.bundle.descriptor_specs,
            real_if_scalar=True,
        )
        site_energy = B @ self.weight + self.bias
        return site_energy.sum(), site_energy

    @staticmethod
    def _voigt_from_stress_tensor(stress_tensor):
        stress_tensor = np.asarray(stress_tensor, dtype=float)
        if full_3x3_to_voigt_6_stress is not None:
            return full_3x3_to_voigt_6_stress(stress_tensor)
        return np.asarray(
            [
                stress_tensor[0, 0],
                stress_tensor[1, 1],
                stress_tensor[2, 2],
                0.5 * (stress_tensor[1, 2] + stress_tensor[2, 1]),
                0.5 * (stress_tensor[0, 2] + stress_tensor[2, 0]),
                0.5 * (stress_tensor[0, 1] + stress_tensor[1, 0]),
            ],
            dtype=float,
        )

    def energy_forces_cell_gradient(self, atoms, *, fixed_scaled_positions=True):
        pos0, cell0, edge_index, shifts, atom_types = self._geometry_from_atoms(atoms, requires_grad=False)
        cell = cell0.detach().clone().requires_grad_(True)
        if fixed_scaled_positions:
            scaled = pos0.detach() @ torch.linalg.inv(cell0.detach())
            pos = scaled @ cell
        else:
            pos = pos0.detach().clone().requires_grad_(True)
        total_energy, site_energy = self._energy_from_geometry(pos, cell, edge_index, shifts, atom_types)
        grad_pos, grad_cell = torch.autograd.grad(total_energy, (pos, cell), create_graph=False, retain_graph=False)
        return total_energy.detach(), (-grad_pos).detach(), grad_cell.detach(), site_energy.detach()

    def _stress_from_cell_gradient(self, atoms, cell_grad):
        cell_np = np.asarray(atoms.cell.array, float)
        volume = float(atoms.get_volume())
        dE_dstrain = cell_np.T @ np.asarray(cell_grad, dtype=float)
        stress_tensor = 0.5 * (dE_dstrain + dE_dstrain.T) / volume
        return self._voigt_from_stress_tensor(stress_tensor)

    def _stress_from_strain_gradient(self, atoms, strain_grad):
        volume = float(atoms.get_volume())
        if volume <= 0.0:
            raise ValueError("ASE stress requires a positive simulation-cell volume.")
        dE_dstrain = np.asarray(strain_grad, dtype=float)
        stress_tensor = 0.5 * (dE_dstrain + dE_dstrain.T) / volume
        return self._voigt_from_stress_tensor(stress_tensor)

    def _energy_and_forces_analytic_from_atoms(self, atoms):
        profile = {} if _linear_ace_profile_enabled() else None
        total_start = _profile_start_tensor(self.weight) if profile is not None else None
        start = _profile_start_tensor(self.weight) if profile is not None else None
        pos, cell, edge_index, shifts, atom_types = self._geometry_from_atoms(atoms, requires_grad=False)
        _profile_stop_tensor(profile, "neighbor_geometry_seconds", start, pos)
        if self.force_method in {"analytic_factorized_fused", "analytic_factorized_fused_streaming"}:
            method = "analytic_factorized"
            if self.force_method.endswith("_streaming"):
                method = "analytic_factorized_streaming"
            start = _profile_start_tensor(pos) if profile is not None else None
            site_linear, grad = descriptor_linear_form_position_vjp(
                evaluator=self.evaluator,
                positions=pos,
                cell=cell,
                edge_index=edge_index,
                atom_types=atom_types,
                descriptors=self.bundle.descriptor_specs,
                descriptor_weight=self.weight,
                shifts=shifts,
                real_if_scalar=True,
                method=method,
            )
            _profile_stop_tensor(profile, "linear_form_vjp_seconds", start, grad)
            site_energy = site_linear + self.bias
            total_energy = site_energy.sum()
        else:
            start = _profile_start_tensor(pos) if profile is not None else None
            values, grad = descriptor_weighted_sum_position_vjp(
                evaluator=self.evaluator,
                positions=pos,
                cell=cell,
                edge_index=edge_index,
                atom_types=atom_types,
                descriptors=self.bundle.descriptor_specs,
                descriptor_weight=self.weight,
                shifts=shifts,
                real_if_scalar=True,
                method=self.force_method,
            )
            _profile_stop_tensor(profile, "weighted_sum_vjp_seconds", start, grad)
            site_energy = (values * self.weight.reshape(1, -1)).sum(dim=1) + self.bias
            total_energy = site_energy.sum()
        if profile is not None:
            _profile_stop_tensor(profile, "total_seconds", total_start, total_energy)
            profile["force_method"] = str(self.force_method)
            profile["neighbor_backend"] = str(self.neighbor_backend)
            profile["edge_count"] = int(edge_index.shape[1])
            profile["atom_count"] = int(pos.shape[0])
            profile["linear_form_profile"] = dict(getattr(self.evaluator, "_last_linear_form_profile", {}))
            self._last_force_profile = dict(profile)
        return total_energy, (-grad).detach().cpu().numpy(), site_energy.detach().cpu().numpy()

    def _energy_forces_stress_cyprime_from_atoms(self, atoms):
        pos, cell, edge_index, shifts, atom_types = self._geometry_from_atoms(atoms, requires_grad=False)
        result = CYFactorProductEvaluator(self.evaluator).evaluate(
            positions=pos,
            cell=cell,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=self.bundle.descriptor_specs,
            shifts=shifts,
            real_if_scalar=True,
            materialize_force_jacobian=True,
            materialize_charge_jacobian=False,
            materialize_stress_jacobian=True,
        )
        site_linear = result.descriptor_values @ self.weight
        site_energy = site_linear + self.bias
        total_energy = site_energy.sum()
        if result.force_jacobian is None:
            forces = torch.zeros_like(pos)
        else:
            grad_flat = self.weight.reshape(1, -1) @ result.force_jacobian
            forces = -grad_flat.reshape_as(pos)
        if result.stress_jacobian is None:
            strain_grad = torch.zeros((3, 3), dtype=pos.dtype, device=pos.device)
        else:
            strain_grad = (result.stress_jacobian * self.weight.reshape(-1, 1, 1)).sum(dim=0)
        self._last_stress_profile = {
            "stress_method": "cyprime",
            "force_method": str(self.force_method),
            "cyprime_report": dict(result.report),
            "ase_convention": {
                "volume_normalization": "applied_by_calculator",
                "voigt_order": "xx_yy_zz_yz_xz_xy",
                "stress_tensor": "sym_dE_dstrain_over_volume",
            },
        }
        return (
            total_energy.detach(),
            forces.detach().cpu().numpy(),
            self._stress_from_strain_gradient(atoms, strain_grad.detach().cpu().numpy()),
            site_energy.detach().cpu().numpy(),
        )

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if "stress" in properties:
            if self.force_method in {
                "analytic",
                "analytic_dag",
                "analytic_streaming",
                "analytic_dag_streaming",
                "analytic_factorized",
                "analytic_factorized_streaming",
                "analytic_factorized_fused",
                "analytic_factorized_fused_streaming",
            }:
                total_energy, forces, stress, site_energy = self._energy_forces_stress_cyprime_from_atoms(atoms)
                offset = _reference_energy_offset_from_atoms(atoms, self.reference_energies)
                self.results["energy"] = float(total_energy.detach().cpu().item()) + offset
                self.results["free_energy"] = self.results["energy"]
                self.results["forces"] = forces
                self.results["energies"] = _add_reference_energy_to_site_energies(
                    atoms,
                    site_energy,
                    self.reference_energies,
                )
                self.results["stress"] = stress
            else:
                total_energy, forces, cell_grad, site_energy = self.energy_forces_cell_gradient(
                    atoms,
                    fixed_scaled_positions=True,
                )
                offset = _reference_energy_offset_from_atoms(atoms, self.reference_energies)
                self.results["energy"] = float(total_energy.detach().cpu().item()) + offset
                self.results["free_energy"] = self.results["energy"]
                self.results["forces"] = forces.detach().cpu().numpy()
                self.results["energies"] = _add_reference_energy_to_site_energies(
                    atoms,
                    site_energy.detach().cpu().numpy(),
                    self.reference_energies,
                )
                self.results["stress"] = self._stress_from_cell_gradient(atoms, cell_grad.detach().cpu().numpy())
                self._last_stress_profile = {
                    "stress_method": "autograd_cell_gradient",
                    "force_method": str(self.force_method),
                    "ase_convention": {
                        "volume_normalization": "applied_by_calculator",
                        "voigt_order": "xx_yy_zz_yz_xz_xy",
                        "stress_tensor": "sym_dE_dstrain_over_volume",
                    },
                }
            return
        if self.force_method in {
            "analytic",
            "analytic_dag",
            "analytic_streaming",
            "analytic_dag_streaming",
            "analytic_factorized",
            "analytic_factorized_streaming",
            "analytic_factorized_fused",
            "analytic_factorized_fused_streaming",
        } and "forces" in properties:
            with torch.no_grad():
                total_energy, forces, site_energy = self._energy_and_forces_analytic_from_atoms(atoms)
            offset = _reference_energy_offset_from_atoms(atoms, self.reference_energies)
            self.results["energy"] = float(total_energy.detach().cpu().item()) + offset
            self.results["free_energy"] = self.results["energy"]
            self.results["forces"] = forces
            self.results["energies"] = _add_reference_energy_to_site_energies(atoms, site_energy, self.reference_energies)
            return
        total_energy, pos, site_energy = self._energy_from_atoms(atoms)
        offset = _reference_energy_offset_from_atoms(atoms, self.reference_energies)
        self.results["energy"] = float(total_energy.detach().cpu().item()) + offset
        self.results["free_energy"] = self.results["energy"]
        self.results["energies"] = _add_reference_energy_to_site_energies(
            atoms,
            site_energy.detach().cpu().numpy(),
            self.reference_energies,
        )
        if "forces" in properties:
            grad = torch.autograd.grad(total_energy, pos)[0]
            self.results["forces"] = (-grad).detach().cpu().numpy()


class LinearACEMultiCutoffCalculator(Calculator):
    """ASE calculator that sums linear ACE groups with different cutoffs."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        bundle,
        *,
        force_method = "analytic_factorized",
        device = None,
        backend = "pytorch",
        strict_backend = False,
        validate_backend = True,
        neighbor_backend = "ase",
        neighbor_skin = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(bundle, LinearACEMultiCutoffModelBundle):
            raise TypeError("Expected a LinearACEMultiCutoffModelBundle.")
        self.bundle = bundle
        self.device = _resolve_torch_device(device)
        self.backend = str(backend)
        self.strict_backend = bool(strict_backend)
        self.validate_backend = bool(validate_backend)
        self.neighbor_backend = str(neighbor_backend)
        self.neighbor_skin = max(float(neighbor_skin), 0.0)
        self._neighbor_cache = None
        self.type_map = dict(bundle.type_map)
        method = str(force_method).strip().lower().replace("-", "_")
        aliases = {
            "analytical": "analytic_factorized",
            "analytic": "analytic_factorized",
            "analytic_direct": "analytic",
            "direct": "analytic",
            "streaming": "analytic_streaming",
            "analytical_streaming": "analytic_streaming",
            "analytical_dag": "analytic_dag",
            "analytical_dag_streaming": "analytic_dag_streaming",
            "factorized": "analytic_factorized",
            "sym_power": "analytic_factorized",
            "symmetric_power": "analytic_factorized",
        }
        self.force_method = aliases.get(method, method)
        if self.force_method not in {
            "analytic",
            "analytic_dag",
            "analytic_streaming",
            "analytic_dag_streaming",
            "analytic_factorized",
            "analytic_factorized_streaming",
        }:
            raise ValueError("Multi-cutoff linear ACE MD currently supports analytic force methods.")
        self.groups = []
        for group in bundle.groups:
            group_bundle = group["bundle"]
            cutoff = float(group["cutoff"])
            evaluator = ACECovariantEvaluator(
                group_bundle.site_basis_config,
                backend=backend,
                strict_backend=strict_backend,
                validate_backend=validate_backend,
            ).to(device=self.device)
            evaluator.precompile_descriptors(group_bundle.descriptor_specs)
            self.groups.append(
                {
                    "cutoff": cutoff,
                    "bundle": group_bundle,
                    "evaluator": evaluator,
                    "weight": torch.as_tensor(group_bundle.weight, dtype=torch.float64, device=self.device),
                    "bias": torch.as_tensor(group_bundle.bias, dtype=torch.float64, device=self.device),
                }
            )
        if not self.groups:
            raise ValueError("A multi-cutoff linear ACE bundle must contain at least one group.")
        self.max_cutoff = float(max(group["cutoff"] for group in self.groups))

    def _neighbor_data_from_atoms(self, atoms):
        positions = np.asarray(atoms.positions, float)
        cell = np.asarray(atoms.cell.array, float)
        atom_types = _atom_type_indices_from_atoms(atoms, self.type_map)
        build_cutoff = self.max_cutoff + self.neighbor_skin
        if _neighbor_cache_is_valid(
            self._neighbor_cache,
            positions,
            cell,
            atom_types,
            build_cutoff,
            self.neighbor_skin,
        ):
            return self._neighbor_cache["neighbor_data"]
        nbr = neighbor_data_from_ase_atoms(
            atoms,
            build_cutoff,
            self.type_map,
            backend=self.neighbor_backend,
        )
        if self.neighbor_skin > 0.0:
            self._neighbor_cache = {
                "positions": positions.copy(),
                "cell": cell.copy(),
                "atom_types": atom_types.copy(),
                "cutoff": float(build_cutoff),
                "neighbor_data": nbr,
            }
        return nbr

    def _geometry_from_atoms(self, atoms):
        nbr = self._neighbor_data_from_atoms(atoms)
        pos = torch.as_tensor(np.asarray(atoms.positions, float), dtype=torch.float64, device=self.device)
        cell = torch.as_tensor(np.asarray(atoms.cell.array, float), dtype=torch.float64, device=self.device)
        shifts = torch.as_tensor(np.asarray(nbr.shifts, float), dtype=torch.float64, device=self.device)
        edge_index = torch.as_tensor(nbr.edge_index, dtype=torch.long, device=self.device)
        atom_types = torch.as_tensor(nbr.atom_types, dtype=torch.long, device=self.device)
        x_ij = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
        distances = torch.linalg.norm(x_ij, dim=1)
        return pos, cell, edge_index, shifts, atom_types, distances

    def _energy_from_geometry(self, pos, cell, edge_index, shifts, atom_types, distances):
        total_energy = torch.zeros((), dtype=torch.float64, device=self.device)
        for group in self.groups:
            group_bundle = group["bundle"]
            if len(group_bundle.descriptor_specs) == 0:
                continue
            mask = distances <= (float(group["cutoff"]) + 1.0e-12)
            group_edge_index = edge_index[:, mask]
            group_shifts = shifts[mask]
            x_ij = pos[group_edge_index[1]] - pos[group_edge_index[0]] + group_shifts @ cell
            values = group["evaluator"](
                x_ij=x_ij,
                edge_index=group_edge_index,
                atom_types=atom_types,
                descriptors=group_bundle.descriptor_specs,
                real_if_scalar=True,
            )
            total_energy = (
                total_energy
                + (values * group["weight"].reshape(1, -1)).sum()
                + group["bias"] * float(pos.shape[0])
            )
        return total_energy

    @staticmethod
    def _voigt_from_stress_tensor(stress_tensor):
        stress_tensor = np.asarray(stress_tensor, dtype=float)
        if full_3x3_to_voigt_6_stress is not None:
            return full_3x3_to_voigt_6_stress(stress_tensor)
        return np.asarray(
            [
                stress_tensor[0, 0],
                stress_tensor[1, 1],
                stress_tensor[2, 2],
                0.5 * (stress_tensor[1, 2] + stress_tensor[2, 1]),
                0.5 * (stress_tensor[0, 2] + stress_tensor[2, 0]),
                0.5 * (stress_tensor[0, 1] + stress_tensor[1, 0]),
            ],
            dtype=float,
        )

    def energy_forces_cell_gradient(self, atoms, *, fixed_scaled_positions=True):
        pos0, cell0, edge_index, shifts, atom_types, distances0 = self._geometry_from_atoms(atoms)
        cell = cell0.detach().clone().requires_grad_(True)
        if fixed_scaled_positions:
            scaled = pos0.detach() @ torch.linalg.inv(cell0.detach())
            pos = scaled @ cell
        else:
            pos = pos0.detach().clone().requires_grad_(True)
        x_ij_all = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
        distances = torch.linalg.norm(x_ij_all, dim=1)
        total_energy = self._energy_from_geometry(pos, cell, edge_index, shifts, atom_types, distances)
        grad_pos, grad_cell = torch.autograd.grad(total_energy, (pos, cell), create_graph=False, retain_graph=False)
        return total_energy.detach(), (-grad_pos).detach(), grad_cell.detach()

    def _stress_from_cell_gradient(self, atoms, cell_grad):
        cell_np = np.asarray(atoms.cell.array, float)
        volume = float(atoms.get_volume())
        dE_dstrain = cell_np.T @ np.asarray(cell_grad, dtype=float)
        stress_tensor = 0.5 * (dE_dstrain + dE_dstrain.T) / volume
        return self._voigt_from_stress_tensor(stress_tensor)

    def _energy_and_forces_analytic_from_atoms(self, atoms):
        pos, cell, edge_index, shifts, atom_types, distances = self._geometry_from_atoms(atoms)
        total_energy = torch.zeros((), dtype=torch.float64, device=self.device)
        total_grad = torch.zeros_like(pos)
        for group in self.groups:
            group_bundle = group["bundle"]
            if len(group_bundle.descriptor_specs) == 0:
                continue
            mask = distances <= (float(group["cutoff"]) + 1.0e-12)
            group_edge_index = edge_index[:, mask]
            group_shifts = shifts[mask]
            values, grad = descriptor_weighted_sum_position_vjp(
                evaluator=group["evaluator"],
                positions=pos,
                cell=cell,
                edge_index=group_edge_index,
                atom_types=atom_types,
                descriptors=group_bundle.descriptor_specs,
                descriptor_weight=group["weight"],
                shifts=group_shifts,
                real_if_scalar=True,
                method=self.force_method,
            )
            total_energy = total_energy + (values * group["weight"].reshape(1, -1)).sum() + group["bias"] * float(pos.shape[0])
            total_grad = total_grad + grad
        return total_energy, (-total_grad).detach().cpu().numpy()

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if "stress" in properties:
            total_energy, forces, cell_grad = self.energy_forces_cell_gradient(
                atoms,
                fixed_scaled_positions=True,
            )
            self.results["energy"] = float(total_energy.detach().cpu().item())
            self.results["free_energy"] = self.results["energy"]
            self.results["forces"] = forces.detach().cpu().numpy()
            self.results["stress"] = self._stress_from_cell_gradient(atoms, cell_grad.detach().cpu().numpy())
            return
        with torch.no_grad():
            total_energy, forces = self._energy_and_forces_analytic_from_atoms(atoms)
        self.results["energy"] = float(total_energy.detach().cpu().item())
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = forces


def _reference_energy(atoms, energy_key):
    if energy_key in getattr(atoms, "info", {}):
        return float(atoms.info[energy_key])
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            return None
    return None


def _reference_forces(atoms, force_key):
    if force_key in getattr(atoms, "arrays", {}):
        return np.asarray(atoms.arrays[force_key], float)
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        try:
            return np.asarray(atoms.get_forces(), float)
        except Exception:
            return None
    return None


def _reference_stress(atoms, stress_key):
    if stress_key in getattr(atoms, "info", {}):
        value = atoms.info[stress_key]
    else:
        calc = getattr(atoms, "calc", None)
        if calc is None:
            return None
        try:
            value = atoms.get_stress(voigt=True)
        except Exception:
            return None
    stress = np.asarray(value, dtype=float)
    if stress.shape != (6,) or not np.isfinite(stress).all():
        raise ValueError("Reference stress must be finite ASE Voigt data with six components.")
    return stress


def _descriptor_specs_from_settings(
    *,
    settings,
    site_basis_config,
    coupling_library = None,
    basis_mode,
    compact_labels = None,
    max_variants_per_label = None,
    descriptor_cache = None,
    use_descriptor_cache = True,
    return_metadata = False,
):
    if coupling_library is None:
        labels, library, collection = compile_descriptor_artifacts(
            settings,
            compact_labels=compact_labels,
            basis_mode=basis_mode,
            max_variants_per_label=max_variants_per_label,
            descriptor_cache=descriptor_cache,
            use_descriptor_cache=use_descriptor_cache,
        )
        metadata = {
            "coupling_source": "compile_descriptor_artifacts",
            "coupling_cache_enabled": bool(use_descriptor_cache),
            "coupling_library_supplied": False,
            "compact_label_count": int(len(labels)),
            "descriptor_cache_stats": None if descriptor_cache is None else descriptor_cache.get_stats(),
        }
    else:
        labels = (
            [normalize_compact_label(label) for label in compact_labels]
            if compact_labels is not None
            else enumerate_compact_labels(
                settings,
                basis_mode=basis_mode,
                descriptor_cache=descriptor_cache,
                use_descriptor_cache=use_descriptor_cache,
            )
        )
        collection = build_descriptor_specs_from_settings(
            labels,
            settings,
            coupling_library,
            max_variants_per_label=max_variants_per_label,
        )
        metadata = {
            "coupling_source": "supplied_generalized_coupling_library",
            "coupling_cache_enabled": bool(use_descriptor_cache),
            "coupling_library_supplied": True,
            "compact_label_count": int(len(labels)),
            "descriptor_cache_stats": None if descriptor_cache is None else descriptor_cache.get_stats(),
        }
    specs = list(collection.specs_by_M[0])
    if return_metadata:
        return specs, metadata
    return specs


def linear_ace_fit_preflight(
    compact_labels,
    *,
    settings,
    max_variants_per_label=None,
    include_bias_column=True,
    structure_atom_counts=None,
    structure_count=None,
    atom_count=None,
    include_force_rows=True,
    physical_content_channels=None,
):
    """Report an exact linear-feature and optional regression-shape preflight.

    Purpose:
        Expose the fit size before coefficient compilation or dataset loading.
    Mathematical contract:
        Counts the channel expansion used by descriptor materialization for
        the supplied compiler-owned labels and optional physical eta binding.
    Inputs:
        Compact labels, descriptor settings, the optional variant cap, and
        optional aggregate or per-structure atom counts.
    Outputs:
        Exact logical feature, parameter, tensor-rank, product-factor, and
        optional regression-storage counts.
    Does not:
        Compile coupling coefficients, evaluate descriptors, access a dataset,
        estimate wall time, or certify PACE lowering semantics.
    """

    labels = tuple(normalize_compact_label(label) for label in compact_labels)
    bound_ids = None
    if physical_content_channels is not None:
        records = tuple(tuple(int(value) for value in row)
                        for row in physical_content_channels)
        if (settings.basis_type != "no_charge" or max_variants_per_label is not None or
                not records or any(len(row) != 3 or row[0] < 1 or row[1] < 0 or row[2] < 1
                                   for row in records) or
                len({row[0] for row in records}) != len(records) or
                len({row[1:] for row in records}) != len(records)):
            raise ValueError("Physical eta preflight requires an injective no-charge source binding.")
        bound_ids = {row[0] for row in records}
    if not labels:
        raise ValueError("linear ACE preflight requires at least one manual label.")
    if len(set(labels)) != len(labels):
        raise ValueError("linear ACE preflight labels must be unique.")
    if int(settings.L_R) != 0 or 0 not in tuple(
        int(value) for value in settings.M_R_values
    ):
        raise ValueError("linear scalar ACE preflight requires an L=0, M=0 target.")
    configured_ranks = {int(value) for value in settings.ranks}
    missing_ranks = sorted(
        {int(label.rank) for label in labels} - configured_ranks
    )
    if missing_ranks:
        raise ValueError(
            "Manual labels use tensor ranks absent from descriptor settings: "
            + ", ".join(str(value) for value in missing_ranks)
        )
    for label in labels:
        rank = int(label.rank)
        rank_index = settings.rank_index(rank)
        if int(label.L_R) != int(settings.L_R):
            raise ValueError("A manual label target L differs from descriptor settings.")
        if any(
            int(value) < 1 or int(value) > int(settings.nmax[rank_index])
            for value in label.n_tuple
        ):
            raise ValueError("A manual label radial index is outside descriptor settings.")
        if bound_ids is not None and any(int(value) not in bound_ids
                                         for value in label.n_tuple):
            raise ValueError("A manual label has an unbound physical eta content ID.")
        if any(
            int(value) < int(settings.lmin[rank_index])
            or int(value) > int(settings.lmax[rank_index])
            for value in label.l_tuple
        ):
            raise ValueError("A manual label angular index is outside descriptor settings.")
        if (
            settings.parity_filter == "natural"
            and (sum(int(value) for value in label.l_tuple) - int(settings.L_R)) % 2
        ):
            raise ValueError("A manual label violates the descriptor parity filter.")

    features_by_rank = {}
    features_by_center = {
        str(element): 0 for element in tuple(settings.elems)
    }
    logical_feature_count = 0
    product_factor_count = 0
    for label in labels:
        count = (len(settings.mu_values) if bound_ids is not None else
                 count_channel_variants(
                     label, settings,
                     max_variants_per_label=max_variants_per_label))
        rank = int(label.rank)
        logical_feature_count += count
        product_factor_count += rank * count
        features_by_rank[rank] = int(features_by_rank.get(rank, 0)) + count
        center_counts = ({int(center): 1 for center in settings.mu_values}
                         if bound_ids is not None else
                         count_channel_variants_by_center(
                             label, settings,
                             max_variants_per_label=max_variants_per_label))
        for center, center_count in center_counts.items():
            key = (
                str(settings.elems[int(center)])
                if 0 <= int(center) < len(settings.elems)
                else str(center)
            )
            features_by_center[key] = int(
                features_by_center.get(key, 0)
            ) + int(center_count)

    bias_count = 1 if include_bias_column else 0
    column_count = logical_feature_count + bias_count
    fit_shape = None
    if structure_atom_counts is not None:
        if structure_count is not None or atom_count is not None:
            raise ValueError(
                "Use structure_atom_counts or aggregate structure_count/atom_count, not both."
            )
        atom_counts = tuple(int(value) for value in structure_atom_counts)
        if not atom_counts or any(value <= 0 for value in atom_counts):
            raise ValueError("structure_atom_counts must contain positive integers.")
        resolved_structure_count = len(atom_counts)
        resolved_atom_count = sum(atom_counts)
    elif structure_count is not None or atom_count is not None:
        if structure_count is None or atom_count is None:
            raise ValueError("structure_count and atom_count must be supplied together.")
        resolved_structure_count = int(structure_count)
        resolved_atom_count = int(atom_count)
        if resolved_structure_count <= 0 or resolved_atom_count <= 0:
            raise ValueError("structure_count and atom_count must be positive.")
        if resolved_atom_count < resolved_structure_count:
            raise ValueError("atom_count cannot be smaller than structure_count.")
    else:
        resolved_structure_count = None
        resolved_atom_count = None
    if resolved_structure_count is not None:
        energy_rows = resolved_structure_count
        force_rows = 3 * resolved_atom_count if include_force_rows else 0
        row_count = energy_rows + force_rows
        itemsize = np.dtype(np.float64).itemsize
        gram_bytes = column_count * column_count * itemsize
        sufficient_statistics_bytes = gram_bytes + column_count * itemsize + itemsize
        dense_design_and_target_bytes = row_count * (column_count + 1) * itemsize
        fit_shape = {
            "structure_count": int(resolved_structure_count),
            "atom_count": int(resolved_atom_count),
            "energy_row_count": int(energy_rows),
            "force_row_count": int(force_rows),
            "regression_row_count": int(row_count),
            "regression_column_count": int(column_count),
            "gram_matrix_bytes": int(gram_bytes),
            "sufficient_statistics_bytes": int(sufficient_statistics_bytes),
            "dense_fallback_design_and_target_bytes": int(
                dense_design_and_target_bytes
            ),
        }

    return {
        "schema": "ye3t_linear_ace_fit_preflight_v1",
        "manual_label_count": int(len(labels)),
        "logical_fit_feature_count_total": int(logical_feature_count),
        "logical_fit_feature_count_by_rank": {
            str(rank): int(features_by_rank[rank])
            for rank in sorted(features_by_rank)
        },
        "logical_fit_feature_count_by_central_element": features_by_center,
        "logical_fit_feature_count_by_central_element_status": "exact_current_materializer",
        "maximum_tensor_rank": int(max(label.rank for label in labels)),
        "product_factor_count": int(product_factor_count),
        "intercept_parameter_count": int(bias_count),
        "linear_parameter_count": int(column_count),
        "emitted_yace_function_count": (None if bound_ids is not None else
                                        int(logical_feature_count)),
        "emitted_yace_function_count_status": (
            "unsupported_physical_eta_binding" if bound_ids is not None else
            "exact_current_materializer"),
        "unique_polynomial_function_count": None,
        "unique_polynomial_function_count_status": (
            "not_certified_without_joint_independence_certificate"
        ),
        "fit_shape": fit_shape,
        "coefficient_compilation_performed": False,
        "descriptor_evaluation_performed": False,
        "dataset_access_performed": False,
    }


def _require_sklearn():
    if LinearRegression is None:
        raise ImportError(
            "scikit-learn is required for sklearn-based linear ACE fit methods. "
            "Install scikit-learn or use fit_method='adam'."
        )


def _make_sklearn_estimator(
    fit_method,
    *,
    sklearn_params = None,
):
    _require_sklearn()
    params = dict(sklearn_params or {})
    method = str(fit_method).lower()
    if method in {"linear", "linear_regression", "ols"}:
        return LinearRegression(fit_intercept=False, **params)
    if method == "ridge":
        params.setdefault("alpha", 1.0)
        return Ridge(fit_intercept=False, **params)
    if method == "lasso":
        params.setdefault("alpha", 1e-6)
        params.setdefault("max_iter", 100000)
        return Lasso(fit_intercept=False, **params)
    if method == "ridgecv":
        params.setdefault("alphas", np.logspace(-8, 3, 16))
        return RidgeCV(fit_intercept=False, **params)
    if method in {"ard", "ardregression"}:
        return ARDRegression(fit_intercept=False, **params)
    raise ValueError(
        "Unsupported fit_method={!r}. Choose from 'adam', 'linear_regression', "
        "'ridge', 'lasso', 'ridgecv', or 'ardregression'.".format(fit_method)
    )


def _jsonable_cache_payload(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable_cache_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_cache_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if hasattr(value, "as_dict"):
        return _jsonable_cache_payload(value.as_dict())
    return repr(value)


def _descriptor_cache_signature(descriptors):
    signature = []
    for desc in descriptors:
        signature.append({
            "rank": int(desc.rank),
            "M_R": int(desc.M_R),
            "channels": [
                {
                    "mu0": int(ch.mu0),
                    "mu": int(ch.mu),
                    "n": int(ch.n),
                    "l": int(ch.l),
                    "kappa0": int(ch.kappa0),
                    "kappa": int(ch.kappa),
                    "eta": None if ch.eta is None else int(ch.eta),
                    "l_aux": None if getattr(ch, "l_aux", None) is None else int(ch.l_aux),
                    "m_aux": None if getattr(ch, "m_aux", None) is None else int(ch.m_aux),
                }
                for ch in desc.channels
            ],
            "ms_combinations": [[int(value) for value in row] for row in desc.ms_combinations],
            "coeffs": [[float(complex(value).real), float(complex(value).imag)] for value in desc.coeffs],
        })
    return signature


def _update_array_hash(digest, value):
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())


def _training_structures_cache_digest(structures, energy_key, force_key):
    digest = hashlib.sha256()
    digest.update(str(energy_key).encode("utf-8"))
    digest.update(str(force_key).encode("utf-8"))
    digest.update(str(len(structures)).encode("utf-8"))
    for index, atoms in enumerate(structures):
        digest.update(str(index).encode("utf-8"))
        digest.update("".join(atoms.get_chemical_symbols()).encode("utf-8"))
        _update_array_hash(digest, atoms.get_atomic_numbers())
        _update_array_hash(digest, np.asarray(atoms.positions, float))
        _update_array_hash(digest, np.asarray(atoms.cell.array, float))
        _update_array_hash(digest, np.asarray(atoms.pbc, bool))
        energy_ref = _reference_energy(atoms, energy_key)
        digest.update(b"energy-none" if energy_ref is None else repr(float(energy_ref)).encode("utf-8"))
        force_ref = _reference_forces(atoms, force_key)
        if force_ref is None:
            digest.update(b"forces-none")
        else:
            _update_array_hash(digest, np.asarray(force_ref, float))
    return digest.hexdigest()


def _linear_problem_cache_key(
    structures,
    *,
    settings,
    descriptors,
    site_basis_config,
    cutoff,
    type_map,
    energy_key,
    force_key,
    energy_weight,
    force_weight,
    backend,
    strict_backend,
    validate_backend,
    device = None,
    factorized_descriptor_runtime_policy = None,
    force_atom_stride = None,
    force_jacobian_mode = None,
    force_jacobian_chunk_size = None,
    fit_objective = None,
):
    payload = {
        "version": 2,
        "settings": None if settings is None else settings.as_dict(),
        "site_basis_config": serialize_site_basis_config(site_basis_config),
        "descriptors": _descriptor_cache_signature(descriptors),
        "structures_digest": _training_structures_cache_digest(structures, energy_key, force_key),
        "cutoff": float(cutoff),
        "type_map": {str(key): int(value) for key, value in sorted(dict(type_map).items(), key=lambda item: str(item[0]))},
        "energy_key": str(energy_key),
        "force_key": str(force_key),
        "energy_weight": float(energy_weight),
        "force_weight": float(force_weight),
        "backend": str(backend),
        "strict_backend": bool(strict_backend),
        "validate_backend": bool(validate_backend),
        "device": str(device),
        "factorized_descriptor_runtime_policy": None
        if factorized_descriptor_runtime_policy is None
        else str(factorized_descriptor_runtime_policy),
        "force_atom_stride": None if force_atom_stride is None else int(force_atom_stride),
        "force_jacobian_mode": None
        if force_jacobian_mode is None
        else str(force_jacobian_mode),
        "force_jacobian_chunk_size": None
        if force_jacobian_chunk_size is None
        else int(force_jacobian_chunk_size),
        "fit_objective": _jsonable_cache_payload(fit_objective),
    }
    text = json.dumps(_jsonable_cache_payload(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _descriptor_matrix_cache_location(descriptor_matrix_cache, cache_key):
    if not descriptor_matrix_cache:
        return None
    if descriptor_matrix_cache is True:
        location = "linear_ace_descriptor_matrix_cache"
    else:
        location = descriptor_matrix_cache
    if isinstance(descriptor_matrix_cache, dict):
        location = (
            descriptor_matrix_cache.get("path", None)
            or descriptor_matrix_cache.get("file", None)
            or descriptor_matrix_cache.get("dir", None)
            or descriptor_matrix_cache.get("directory", None)
        )
    if location is None:
        return None
    path = Path(location)
    if path.suffix == ".npz":
        return path
    return path / ("linear_ace_descriptor_matrix_" + str(cache_key)[:16] + ".npz")


def _load_descriptor_matrix_cache(path, cache_key):
    if path is None or not path.is_file():
        return None
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("cache_key") != cache_key:
            return None
        return (
            np.asarray(data["X"], dtype=float),
            np.asarray(data["y"], dtype=float),
            metadata,
        )


def _write_descriptor_matrix_cache(path, cache_key, X, y, metadata):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.pop("path", None)
    payload["cache_key"] = str(cache_key)
    payload["n_rows"] = int(X.shape[0])
    payload["n_cols"] = int(X.shape[1])
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=np.asarray(X, dtype=float),
            y=np.asarray(y, dtype=float),
            metadata=np.asarray(json.dumps(_jsonable_cache_payload(payload), sort_keys=True)),
        )
    tmp_path.replace(path)
    return path


def _build_linear_problem_for_structure(
    atoms,
    *,
    evaluator,
    descriptors,
    cutoff,
    type_map,
    energy_key,
    force_key,
    backend = "pytorch",
    strict_backend = False,
    validate_backend = True,
    device = None,
    force_atom_stride = None,
):
    device = _resolve_torch_device(device)
    pos = torch.tensor(np.asarray(atoms.positions, float), dtype=torch.float64, device=device, requires_grad=True)
    nbr = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
    shifts = torch.tensor(np.asarray(nbr.shifts, float), dtype=torch.float64, device=device)
    cell = torch.tensor(np.asarray(atoms.cell.array, float), dtype=torch.float64, device=device)
    edge_index = torch.tensor(nbr.edge_index, dtype=torch.long, device=device)
    atom_types = torch.tensor(nbr.atom_types, dtype=torch.long, device=device)
    x_ij = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
    B = evaluator(x_ij=x_ij, edge_index=edge_index, atom_types=atom_types, descriptors=descriptors, real_if_scalar=True)
    total_desc = B.sum(dim=0)

    energy_ref = _reference_energy(atoms, energy_key)
    energy_row = None
    if energy_ref is not None:
        energy_row = np.concatenate([total_desc.detach().cpu().numpy(), np.asarray([float(len(atoms))], dtype=float)])

    force_ref = _reference_forces(atoms, force_key)
    force_matrix = None
    if force_ref is not None:
        n_feat = int(total_desc.numel())
        force_matrix_t = torch.empty((pos.numel(), n_feat + 1), dtype=torch.float64, device=device)
        for feat_idx in range(n_feat):
            grad_j = torch.autograd.grad(total_desc[feat_idx], pos, retain_graph=(feat_idx + 1 < n_feat))[0]
            force_matrix_t[:, feat_idx] = (-grad_j).reshape(-1)
        force_matrix_t[:, -1] = 0.0
        force_matrix = force_matrix_t.detach().cpu().numpy()
    force_target = None if force_ref is None else np.asarray(force_ref, float).reshape(-1)
    if force_matrix is not None and force_target is not None and force_atom_stride is not None:
        stride = max(1, int(force_atom_stride))
        atom_indices = np.arange(0, len(atoms), stride, dtype=int)
        component_indices = np.concatenate([3 * atom_indices + axis for axis in range(3)])
        component_indices = np.sort(component_indices)
        force_matrix = force_matrix[component_indices]
        force_target = force_target[component_indices]
    return energy_row, energy_ref, force_matrix, force_target


def _descriptor_sum_for_structure(
    atoms,
    *,
    evaluator,
    descriptors,
    cutoff,
    type_map,
    device,
):
    pos = torch.tensor(np.asarray(atoms.positions, float), dtype=torch.float64, device=device, requires_grad=True)
    nbr = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
    shifts = torch.tensor(np.asarray(nbr.shifts, float), dtype=torch.float64, device=device)
    cell = torch.tensor(np.asarray(atoms.cell.array, float), dtype=torch.float64, device=device)
    edge_index = torch.tensor(nbr.edge_index, dtype=torch.long, device=device)
    atom_types = torch.tensor(nbr.atom_types, dtype=torch.long, device=device)

    def evaluate_from_flat(flat_pos):
        reshaped = flat_pos.reshape_as(pos)
        x_ij = reshaped[edge_index[1]] - reshaped[edge_index[0]] + shifts @ cell
        values = evaluator(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=descriptors,
            real_if_scalar=True,
        )
        return values.sum(dim=0)

    return pos, evaluate_from_flat


def _descriptor_site_design_for_structure(
    atoms,
    *,
    evaluator,
    descriptors,
    cutoff,
    type_map,
    device,
):
    """Evaluate one structure's per-site design without building force rows."""
    pos = torch.tensor(
        np.asarray(atoms.positions, float),
        dtype=torch.float64,
        device=device,
    )
    nbr = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
    shifts = torch.tensor(
        np.asarray(nbr.shifts, float),
        dtype=torch.float64,
        device=device,
    )
    cell = torch.tensor(
        np.asarray(atoms.cell.array, float),
        dtype=torch.float64,
        device=device,
    )
    edge_index = torch.tensor(nbr.edge_index, dtype=torch.long, device=device)
    atom_types = torch.tensor(nbr.atom_types, dtype=torch.long, device=device)
    with torch.no_grad():
        x_ij = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
        values = evaluator(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=descriptors,
            real_if_scalar=True,
        )
    return values.detach().to(dtype=torch.float64)


def _descriptor_position_jacobian_from_product_adjoint(
    total_desc,
    flat_pos,
    *,
    chunk_size = None,
):
    n_feat = int(total_desc.numel())
    if n_feat == 0:
        return torch.zeros((0, int(flat_pos.numel())), dtype=flat_pos.dtype, device=flat_pos.device)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = n_feat
    chunk_size = max(1, int(chunk_size))
    rows = []
    for start in range(0, n_feat, chunk_size):
        stop = min(n_feat, start + chunk_size)
        grad_outputs = torch.zeros((stop - start, n_feat), dtype=total_desc.dtype, device=total_desc.device)
        grad_outputs[:, start:stop] = torch.eye(stop - start, dtype=total_desc.dtype, device=total_desc.device)
        grad = torch.autograd.grad(
            total_desc,
            flat_pos,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=stop < n_feat,
            create_graph=False,
            allow_unused=False,
        )[0]
        rows.append(grad.reshape(stop - start, -1))
    return torch.cat(rows, dim=0)


def _descriptor_position_jacobian(
    *,
    mode,
    total_desc,
    evaluate_from_flat,
    flat_pos,
    chunk_size = None,
):
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"vectorized", "grouped", "batched", "functional_jacobian"}:
        return torch.autograd.functional.jacobian(
            evaluate_from_flat,
            flat_pos,
            vectorize=True,
            create_graph=False,
            strict=False,
        )
    if normalized in {"product_adjoint", "adjoint", "vjp", "batched_vjp", "chunked_vjp", "descriptor_adjoint"}:
        return _descriptor_position_jacobian_from_product_adjoint(
            total_desc,
            flat_pos,
            chunk_size=chunk_size,
        )
    if normalized in {
        "analytic_product_adjoint",
        "analytic_adjoint",
        "analytic_vjp",
        "factorized_analytic",
        "cyprime",
        "cyprime_product",
        "cyprime_product_rule",
        "cyprime_force",
    }:
        raise RuntimeError("analytic product-adjoint mode must be handled before total_desc autograd construction.")
    if normalized in {"loop", "per_descriptor"}:
        jac_rows = []
        n_feat = int(total_desc.numel())
        for feat_idx in range(n_feat):
            grad_j = torch.autograd.grad(
                total_desc[feat_idx],
                flat_pos,
                retain_graph=(feat_idx + 1 < n_feat),
            )[0]
            jac_rows.append(grad_j.reshape(-1))
        return torch.stack(jac_rows, dim=0)
    raise ValueError("Unknown force_jacobian_mode={!r}".format(mode))


def _linear_ace_geometry_row(atoms, *, evaluator, descriptors, cutoff, type_map,
                             device, forces, stress, chunk_size):
    """Return ordered scalar energy, force, and ASE stress design rows."""
    position = torch.as_tensor(np.asarray(atoms.positions, float), dtype=torch.float64, device=device)
    cell = torch.as_tensor(np.asarray(atoms.cell.array, float), dtype=torch.float64, device=device)
    neighbor = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
    result = CYFactorProductEvaluator(evaluator).evaluate(
        positions=position,
        cell=cell,
        edge_index=torch.as_tensor(neighbor.edge_index, dtype=torch.long, device=device),
        atom_types=torch.as_tensor(neighbor.atom_types, dtype=torch.long, device=device),
        descriptors=descriptors,
        shifts=torch.as_tensor(np.asarray(neighbor.shifts, float), dtype=torch.float64, device=device),
        real_if_scalar=True,
        chunk_size=chunk_size,
        materialize_force_jacobian=bool(forces),
        materialize_charge_jacobian=False,
        materialize_stress_jacobian=bool(stress),
    )
    energy_row = result.descriptor_values.sum(dim=0).to(dtype=torch.float64)
    force_rows = None if not forces else -result.force_jacobian.T.to(dtype=torch.float64)
    stress_rows = None
    if stress:
        volume = float(atoms.get_volume())
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("Stress fitting requires a positive simulation-cell volume.")
        strain = result.stress_jacobian
        symmetric = 0.5 * (strain + strain.transpose(1, 2)) / volume
        stress_rows = torch.stack((symmetric[:, 0, 0], symmetric[:, 1, 1],
                                   symmetric[:, 2, 2], symmetric[:, 1, 2],
                                   symmetric[:, 0, 2], symmetric[:, 0, 1]), dim=0).to(dtype=torch.float64)
    return {"energy": energy_row, "forces": force_rows, "stress": stress_rows,
            "site_features": result.descriptor_values, "report": result.report}


def _build_linear_normal_equations_for_structure(
    atoms,
    *,
    evaluator,
    descriptors,
    cutoff,
    type_map,
    energy_key,
    force_key,
    stress_key,
    energy_weight,
    force_weight,
    stress_weight,
    device,
    force_jacobian_mode = "product_adjoint",
    force_jacobian_chunk_size = None,
    normal_equation_coordinate_chunk_size = 64,
    force_atom_stride = None,
    include_bias_column = True,
    ridge_alpha = 0.0,
    ridge_include_bias = False,
    structure_weights = None,
    structure_weight_key = None,
    boltzmann_temperature_K = None,
    boltzmann_energy_key = None,
    boltzmann_weight_nugget = 0.0,
    boltzmann_weight_prefactor = 1.0,
    boltzmann_normalize_mean = True,
    min_structure_weight = 0.0,
):
    device = _resolve_torch_device(device)
    normalized_mode = str(force_jacobian_mode).strip().lower().replace("-", "_")
    if stress_weight:
        if normalized_mode not in {
                "product_adjoint", "adjoint", "vjp", "batched_vjp", "chunked_vjp",
                "descriptor_adjoint", "analytic_product_adjoint", "analytic_adjoint",
                "analytic_vjp", "factorized_analytic", "cyprime", "cyprime_product",
                "cyprime_product_rule", "cyprime_force"}:
            raise ValueError("Stress fitting requires a product-adjoint force-row mode.")
        stress_ref = _reference_stress(atoms, stress_key)
        if stress_ref is None:
            raise ValueError(f"Structure lacks precomputed {stress_key!r} stress.")
        row = _linear_ace_geometry_row(
            atoms, evaluator=evaluator, descriptors=descriptors, cutoff=cutoff,
            type_map=type_map, device=device, forces=bool(force_weight), stress=True,
            chunk_size=force_jacobian_chunk_size)
        n_feat = int(row["energy"].numel())
        n_cols = n_feat + int(bool(include_bias_column))
        XtX = torch.zeros((n_cols, n_cols), dtype=torch.float64, device=device)
        Xty = torch.zeros((n_cols,), dtype=torch.float64, device=device)
        yty = torch.zeros((), dtype=torch.float64, device=device)
        n_rows = 0
        energy_ref = _reference_energy(atoms, energy_key)
        if energy_ref is not None and energy_weight > 0.0:
            energy_row = row["energy"]
            if include_bias_column:
                energy_row = torch.cat((energy_row, energy_row.new_tensor([len(atoms)])))
            energy_row = energy_row * float(np.sqrt(energy_weight))
            target = energy_row.new_tensor(float(energy_ref) * np.sqrt(energy_weight))
            XtX += torch.outer(energy_row, energy_row)
            Xty += energy_row * target
            yty += target * target
            n_rows += 1
        force_ref = _reference_forces(atoms, force_key)
        if force_weight > 0.0:
            if force_ref is None:
                raise ValueError(f"Structure lacks precomputed {force_key!r} forces.")
            force_rows = row["forces"]
            target = torch.as_tensor(np.asarray(force_ref, float).reshape(-1),
                                     dtype=torch.float64, device=device)
            if force_atom_stride is not None:
                stride = max(1, int(force_atom_stride))
                atom_indices = torch.arange(0, len(atoms), stride, dtype=torch.long, device=device)
                selected = torch.cat([3 * atom_indices + axis for axis in range(3)]).sort().values
                force_rows = force_rows.index_select(0, selected)
                target = target.index_select(0, selected)
            weighted = force_rows * float(np.sqrt(force_weight))
            target = target * float(np.sqrt(force_weight))
            XtX[:n_feat, :n_feat] += weighted.T @ weighted
            Xty[:n_feat] += weighted.T @ target
            yty += torch.dot(target, target)
            n_rows += int(target.numel())
        weighted = row["stress"] * float(np.sqrt(stress_weight))
        target = torch.as_tensor(stress_ref, dtype=torch.float64, device=device) * float(np.sqrt(stress_weight))
        XtX[:n_feat, :n_feat] += weighted.T @ weighted
        Xty[:n_feat] += weighted.T @ target
        yty += torch.dot(target, target)
        n_rows += 6
        if float(ridge_alpha) > 0.0:
            ridge_cols = n_cols if bool(ridge_include_bias) else n_feat
            diagonal = torch.arange(ridge_cols, dtype=torch.long, device=device)
            XtX[diagonal, diagonal] += float(ridge_alpha)
        return XtX, Xty, yty, n_rows, {
            "backend": row["report"]["backend"],
            "materializes_force_jacobian": bool(force_weight),
            "materializes_stress_jacobian": True,
            "ridge_alpha": float(ridge_alpha),
            "ridge_include_bias": bool(ridge_include_bias),
        }
    if normalized_mode in {"cyprime", "cyprime_product", "cyprime_product_rule", "cyprime_force"}:
        pos = torch.tensor(np.asarray(atoms.positions, float), dtype=torch.float64, device=device)
        nbr = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
        shifts = torch.tensor(np.asarray(nbr.shifts, float), dtype=torch.float64, device=device)
        cell = torch.tensor(np.asarray(atoms.cell.array, float), dtype=torch.float64, device=device)
        edge_index = torch.tensor(nbr.edge_index, dtype=torch.long, device=device)
        atom_types = torch.tensor(nbr.atom_types, dtype=torch.long, device=device)
        update = CYFactorProductEvaluator(evaluator).normal_equations(
            positions=pos,
            cell=cell,
            edge_index=edge_index,
            atom_types=atom_types,
            descriptors=descriptors,
            shifts=shifts,
            real_if_scalar=True,
            descriptor_chunk_size=force_jacobian_chunk_size,
            coordinate_chunk_size=normal_equation_coordinate_chunk_size,
            energy_ref=_reference_energy(atoms, energy_key),
            force_ref=_reference_forces(atoms, force_key),
            energy_weight=energy_weight,
            force_weight=force_weight,
            atom_count=len(atoms),
            force_atom_stride=force_atom_stride,
            include_bias_column=include_bias_column,
            ridge_alpha=ridge_alpha,
            ridge_include_bias=ridge_include_bias,
        )
        return update.XtX, update.Xty, update.yty, update.n_rows, update.report
    pos, evaluate_from_flat = _descriptor_sum_for_structure(
        atoms,
        evaluator=evaluator,
        descriptors=descriptors,
        cutoff=cutoff,
        type_map=type_map,
        device=device,
    )
    flat_pos = pos.reshape(-1)
    analytic_jac = None
    if normalized_mode in {
        "analytic_product_adjoint",
        "analytic_adjoint",
        "analytic_vjp",
        "factorized_analytic",
    }:
        nbr = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
        shifts = torch.tensor(np.asarray(nbr.shifts, float), dtype=torch.float64, device=device)
        cell = torch.tensor(np.asarray(atoms.cell.array, float), dtype=torch.float64, device=device)
        edge_index = torch.tensor(nbr.edge_index, dtype=torch.long, device=device)
        atom_types = torch.tensor(nbr.atom_types, dtype=torch.long, device=device)
        values, analytic_jac = descriptor_sum_position_jacobian_analytic_product(
            evaluator,
            pos,
            cell,
            edge_index,
            atom_types,
            descriptors,
            shifts=shifts,
            real_if_scalar=True,
            chunk_size=force_jacobian_chunk_size,
        )
        total_desc = values.sum(dim=0)
    else:
        total_desc = evaluate_from_flat(flat_pos)
    n_feat = int(total_desc.numel())
    include_bias_column = bool(include_bias_column)
    n_cols = n_feat + (1 if include_bias_column else 0)
    XtX = torch.zeros((n_cols, n_cols), dtype=torch.float64, device=device)
    Xty = torch.zeros((n_cols,), dtype=torch.float64, device=device)
    yty = torch.zeros((), dtype=torch.float64, device=device)
    n_rows = 0

    energy_ref = _reference_energy(atoms, energy_key)
    sqrt_energy_weight = float(np.sqrt(max(energy_weight, 0.0)))
    if energy_ref is not None and sqrt_energy_weight > 0.0:
        if include_bias_column:
            row = torch.cat(
                [
                    total_desc.detach().to(dtype=torch.float64),
                    torch.as_tensor([float(len(atoms))], dtype=torch.float64, device=device),
                ],
                dim=0,
            ) * sqrt_energy_weight
        else:
            row = total_desc.detach().to(dtype=torch.float64) * sqrt_energy_weight
        target = torch.as_tensor(float(energy_ref) * sqrt_energy_weight, dtype=torch.float64, device=device)
        XtX = XtX + torch.outer(row, row)
        Xty = Xty + row * target
        yty = yty + target * target
        n_rows += 1

    force_ref = _reference_forces(atoms, force_key)
    sqrt_force_weight = float(np.sqrt(max(force_weight, 0.0)))
    if force_ref is not None and sqrt_force_weight > 0.0:
        if analytic_jac is None:
            jac = _descriptor_position_jacobian(
                mode=force_jacobian_mode,
                total_desc=total_desc,
                evaluate_from_flat=evaluate_from_flat,
                flat_pos=flat_pos,
                chunk_size=force_jacobian_chunk_size,
            )
        else:
            jac = analytic_jac
        jac = jac.to(dtype=torch.float64)
        target = torch.as_tensor(
            np.asarray(force_ref, float).reshape(-1),
            dtype=torch.float64,
            device=device,
        ) * sqrt_force_weight
        weighted_jac = jac * sqrt_force_weight
        if force_atom_stride is not None:
            stride = max(1, int(force_atom_stride))
            atom_indices = torch.arange(0, len(atoms), stride, dtype=torch.long, device=device)
            component_indices = torch.cat([3 * atom_indices + axis for axis in range(3)]).sort().values
            weighted_jac = weighted_jac.index_select(1, component_indices)
            target = target.index_select(0, component_indices)
        XtX[:n_feat, :n_feat] = XtX[:n_feat, :n_feat] + weighted_jac @ weighted_jac.transpose(0, 1)
        Xty[:n_feat] = Xty[:n_feat] - weighted_jac @ target
        yty = yty + torch.dot(target, target)
        n_rows += int(target.numel())
    if float(ridge_alpha) > 0.0:
        ridge_cols = n_cols if bool(ridge_include_bias) else n_feat
        if ridge_cols > 0:
            diag = torch.arange(ridge_cols, dtype=torch.long, device=device)
            XtX[diag, diag] = XtX[diag, diag] + float(ridge_alpha)
    report = {
        "backend": "materialized_descriptor_position_jacobian_normal_equations",
        "materializes_force_jacobian": bool(force_ref is not None and sqrt_force_weight > 0.0),
        "ridge_alpha": float(ridge_alpha),
        "ridge_include_bias": bool(ridge_include_bias),
        "materializes_stress_jacobian": False,
    }
    return XtX, Xty, yty, n_rows, report


def build_linear_ace_normal_equations(
    structures,
    *,
    settings = None,
    descriptors,
    site_basis_config,
    cutoff,
    type_map,
    energy_key = "energy",
    force_key = "forces",
    stress_key = "stress",
    energy_weight = 1.0,
    force_weight = 1.0,
    stress_weight = 0.0,
    backend = "pytorch",
    strict_backend = False,
    validate_backend = True,
    return_metadata = False,
    device = None,
    factorized_descriptor_runtime_policy = None,
    force_jacobian_mode = "product_adjoint",
    force_jacobian_chunk_size = None,
    normal_equation_coordinate_chunk_size = 64,
    force_atom_stride = None,
    include_bias_column = True,
    ridge_alpha = 0.0,
    ridge_include_bias = False,
    structure_weights = None,
    structure_weight_key = None,
    structure_group_key = None,
    structure_group_weights = None,
    structure_group_default_weight = None,
    structure_group_normalize_mean = True,
    boltzmann_temperature_K = None,
    boltzmann_energy_key = None,
    boltzmann_weight_nugget = 0.0,
    boltzmann_weight_prefactor = 1.0,
    boltzmann_normalize_mean = True,
    min_structure_weight = 0.0,
):
    """Accumulate exact linear ACE normal equations for energy/force/stress fitting.

    The descriptor block is evaluated by the ACE product evaluator on the
    selected torch device. Force rows are contracted into `X^T X`, `X^T y`,
    and `y^T y` directly from the descriptor-position Jacobian, so the full
    force design matrix is not materialized.
    """
    structures = list(structures)
    if not np.isfinite(stress_weight) or stress_weight < 0.0:
        raise ValueError("stress_weight must be finite and nonnegative.")
    device = _resolve_torch_device(device)
    normalized_force_jacobian_mode = str(force_jacobian_mode).strip().lower().replace("-", "_")
    resolved_factorized_descriptor_runtime_policy = factorized_descriptor_runtime_policy
    if (
        normalized_force_jacobian_mode in {"cyprime", "cyprime_product", "cyprime_product_rule", "cyprime_force"}
        and resolved_factorized_descriptor_runtime_policy is None
    ):
        resolved_factorized_descriptor_runtime_policy = "off"
    resolved_structure_weights, structure_weight_metadata = structure_fit_weights(
        structures,
        structure_weights=structure_weights,
        structure_weight_key=structure_weight_key,
        structure_group_key=structure_group_key,
        structure_group_weights=structure_group_weights,
        structure_group_default_weight=structure_group_default_weight,
        structure_group_normalize_mean=structure_group_normalize_mean,
        boltzmann_temperature_K=boltzmann_temperature_K,
        boltzmann_energy_key=boltzmann_energy_key,
        boltzmann_weight_nugget=boltzmann_weight_nugget,
        boltzmann_weight_prefactor=boltzmann_weight_prefactor,
        boltzmann_normalize_mean=boltzmann_normalize_mean,
        min_weight=min_structure_weight,
    )
    cfg = site_basis_config
    evaluator = ACECovariantEvaluator(
        cfg,
        backend=backend,
        strict_backend=strict_backend,
        validate_backend=validate_backend,
        factorized_descriptor_runtime_policy=resolved_factorized_descriptor_runtime_policy,
    )
    metadata = {
        "device": str(device),
        "force_jacobian_mode": str(force_jacobian_mode),
        "force_jacobian_chunk_size": None if force_jacobian_chunk_size is None else int(force_jacobian_chunk_size),
        "normal_equation_coordinate_chunk_size": int(normal_equation_coordinate_chunk_size),
        "force_atom_stride": None if force_atom_stride is None else int(force_atom_stride),
        "descriptor_precompile": evaluator.precompile_descriptors(descriptors),
        "factorized_descriptor_runtime_policy": None
        if resolved_factorized_descriptor_runtime_policy is None
        else str(resolved_factorized_descriptor_runtime_policy),
    }
    include_bias_column = bool(include_bias_column)
    feature_count = int(len(tuple(descriptors)))
    n_cols = feature_count + (1 if include_bias_column else 0)
    XtX = torch.zeros((n_cols, n_cols), dtype=torch.float64, device=device)
    Xty = torch.zeros((n_cols,), dtype=torch.float64, device=device)
    yty = torch.zeros((), dtype=torch.float64, device=device)
    n_rows = 0
    structure_reports = []
    for structure_index, atoms in enumerate(structures):
        row_weight = float(resolved_structure_weights[int(structure_index)])
        row_XtX, row_Xty, row_yty, row_count, row_report = _build_linear_normal_equations_for_structure(
            atoms,
            evaluator=evaluator,
            descriptors=descriptors,
            cutoff=cutoff,
            type_map=type_map,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
            energy_weight=float(energy_weight) * row_weight,
            force_weight=float(force_weight) * row_weight,
            stress_weight=float(stress_weight) * row_weight,
            device=device,
            force_jacobian_mode=force_jacobian_mode,
            force_jacobian_chunk_size=force_jacobian_chunk_size,
            normal_equation_coordinate_chunk_size=normal_equation_coordinate_chunk_size,
            force_atom_stride=force_atom_stride,
            include_bias_column=include_bias_column,
            ridge_alpha=0.0,
            ridge_include_bias=False,
        )
        XtX = XtX + row_XtX
        Xty = Xty + row_Xty
        yty = yty + row_yty
        n_rows += int(row_count)
        structure_reports.append({"structure_index": int(structure_index), **dict(row_report)})
    if float(ridge_alpha) > 0.0:
        ridge_cols = n_cols if bool(ridge_include_bias) else feature_count
        if ridge_cols > 0:
            diag = torch.arange(ridge_cols, dtype=torch.long, device=device)
            XtX[diag, diag] = XtX[diag, diag] + float(ridge_alpha)
    metadata["n_rows"] = int(n_rows)
    metadata["n_cols"] = int(n_cols)
    metadata["feature_count"] = int(feature_count)
    metadata["include_bias_column"] = bool(include_bias_column)
    metadata["ridge_alpha"] = float(ridge_alpha)
    metadata["ridge_include_bias"] = bool(ridge_include_bias)
    metadata["stress_weight"] = float(stress_weight)
    metadata["structure_reports"] = tuple(structure_reports)
    metadata["structure_weights"] = dict(structure_weight_metadata)
    if return_metadata:
        return {
            "XtX": XtX,
            "Xty": Xty,
            "yty": yty,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "metadata": metadata,
        }
    return XtX, Xty, yty


def _select_linear_ace_feature_indices_from_normal(
    normal,
    *,
    max_features=None,
    policy="normal_diagonal",
):
    xtx = normal["XtX"] if isinstance(normal, dict) else normal[0]
    metadata = normal.get("metadata", {}) if isinstance(normal, dict) else {}
    feature_count = int(metadata.get("feature_count", int(xtx.shape[0]) - 1))
    if feature_count < 0:
        raise ValueError("Normal equations must include at least one feature column.")
    if max_features is None:
        return torch.arange(feature_count, dtype=torch.long, device=xtx.device)
    max_features = int(max_features)
    if max_features < 1:
        raise ValueError("max_features must be positive when supplied.")
    if max_features >= feature_count:
        return torch.arange(feature_count, dtype=torch.long, device=xtx.device)
    policy = str(policy).strip().lower()
    if policy not in {"normal_diagonal", "design_norm", "energy_normal_diagonal"}:
        raise ValueError(
            "feature selection policy must be 'normal_diagonal', 'design_norm', or 'energy_normal_diagonal'."
        )
    scores = torch.diagonal(xtx[:feature_count, :feature_count]).detach()
    order = torch.argsort(scores, descending=True)
    return torch.sort(order[:max_features]).values.to(dtype=torch.long, device=xtx.device)


def _normal_equation_reduced_system(
    normal,
    *,
    ridge_alpha=0.0,
    feature_indices=None,
):
    xtx = normal["XtX"] if isinstance(normal, dict) else normal[0]
    xty = normal["Xty"] if isinstance(normal, dict) else normal[1]
    metadata = normal.get("metadata", {}) if isinstance(normal, dict) else {}
    n_cols = int(xtx.shape[0])
    if int(xtx.shape[1]) != n_cols or int(xty.shape[0]) != n_cols:
        raise ValueError("Normal-equation shapes are inconsistent.")
    include_bias_column = bool(metadata.get("include_bias_column", True))
    feature_count = int(metadata.get("feature_count", n_cols - (1 if include_bias_column else 0)))
    if include_bias_column and n_cols != feature_count + 1:
        raise ValueError("Normal-equation metadata is inconsistent with include_bias_column=True.")
    if not include_bias_column and n_cols != feature_count:
        raise ValueError("Normal-equation metadata is inconsistent with include_bias_column=False.")
    if feature_indices is None:
        active = torch.arange(feature_count, dtype=torch.long, device=xtx.device)
    else:
        active = torch.as_tensor(feature_indices, dtype=torch.long, device=xtx.device)
        if active.ndim != 1:
            raise ValueError("feature_indices must be a one-dimensional sequence.")
        if active.numel() == 0:
            raise ValueError("At least one feature must be selected.")
        if torch.any(active < 0) or torch.any(active >= feature_count):
            raise ValueError("feature_indices contain an index outside the feature range.")
        active = torch.unique(active, sorted=True)
    if include_bias_column:
        reduced = torch.cat((active, torch.tensor([feature_count], dtype=torch.long, device=xtx.device)))
    else:
        reduced = active
    xtx_reduced = xtx.index_select(0, reduced).index_select(1, reduced)
    xty_reduced = xty.index_select(0, reduced)
    penalty = torch.eye(int(xtx_reduced.shape[0]), dtype=xtx.dtype, device=xtx.device) * float(ridge_alpha)
    if include_bias_column:
        penalty[-1, -1] = 0.0
    return {
        "system_matrix": xtx_reduced + penalty,
        "rhs": xty_reduced,
        "reduced_indices": reduced,
        "active_feature_indices": active,
        "feature_count": int(feature_count),
        "include_bias_column": bool(include_bias_column),
        "n_cols": int(n_cols),
        "dtype": xty.dtype,
        "device": xty.device,
        "metadata": dict(metadata),
    }


def _solve_normal_equation_system(system_matrix, rhs, *, solver="auto", solver_options=None):
    solver_name = str(solver).strip().lower()
    solver_options = dict(solver_options or {})
    if solver_name in {"auto", "torch", "torch_auto"}:
        try:
            return torch.linalg.solve(system_matrix, rhs), "torch.linalg.solve"
        except RuntimeError:
            return torch.linalg.lstsq(system_matrix, rhs.unsqueeze(1)).solution.squeeze(1), "torch.linalg.lstsq"
    if solver_name in {"torch_solve", "solve"}:
        return torch.linalg.solve(system_matrix, rhs), "torch.linalg.solve"
    if solver_name in {"torch_lstsq", "lstsq", "least_squares"}:
        return torch.linalg.lstsq(system_matrix, rhs.unsqueeze(1)).solution.squeeze(1), "torch.linalg.lstsq"
    if solver_name in {"torch_cholesky", "cholesky"}:
        jitter = float(solver_options.get("jitter", 0.0))
        matrix = system_matrix
        if jitter > 0.0:
            matrix = matrix + torch.eye(int(matrix.shape[0]), dtype=matrix.dtype, device=matrix.device) * jitter
        factor = torch.linalg.cholesky(matrix)
        return torch.cholesky_solve(rhs.unsqueeze(1), factor).squeeze(1), "torch.cholesky_solve"
    if solver_name in {"sklearn", "sklearn_ridge"}:
        if Ridge is None:
            raise ImportError("scikit-learn is required for normal_equation_solver='sklearn_ridge'.")
        matrix_np = system_matrix.detach().cpu().numpy()
        rhs_np = rhs.detach().cpu().numpy()
        eigvals, eigvecs = np.linalg.eigh(0.5 * (matrix_np + matrix_np.T))
        tol = float(solver_options.get("eigenvalue_tol", 1.0e-12))
        mask = eigvals > tol
        if not np.any(mask):
            raise np.linalg.LinAlgError("Normal-equation matrix has no positive eigenvalues for sklearn reconstruction.")
        design = np.diag(np.sqrt(eigvals[mask])) @ eigvecs[:, mask].T
        target = (eigvecs[:, mask].T @ rhs_np) / np.sqrt(eigvals[mask])
        model = Ridge(
            alpha=float(solver_options.get("alpha", 0.0)),
            fit_intercept=False,
            solver=str(solver_options.get("ridge_solver", "auto")),
        )
        model.fit(design, target)
        coeff = torch.as_tensor(model.coef_, dtype=rhs.dtype, device=rhs.device).reshape_as(rhs)
        return coeff, "sklearn.linear_model.Ridge"
    raise ValueError(
        "normal_equation_solver must be 'auto', 'torch_solve', 'torch_lstsq', "
        "'torch_cholesky', 'sklearn_ridge', or a custom solver callable."
    )


def solve_linear_ace_normal_equations(
    normal,
    *,
    ridge_alpha=0.0,
    feature_indices=None,
    solver="auto",
    solver_options=None,
    solver_callable=None,
):
    """Solve linear ACE normal equations with a built-in or caller-provided solver."""

    reduced_system = _normal_equation_reduced_system(
        normal,
        ridge_alpha=ridge_alpha,
        feature_indices=feature_indices,
    )
    system_matrix = reduced_system["system_matrix"]
    rhs = reduced_system["rhs"]
    if solver_callable is not None:
        result = solver_callable(
            system_matrix=system_matrix,
            rhs=rhs,
            metadata={
                **dict(reduced_system["metadata"]),
                "active_feature_indices": reduced_system["active_feature_indices"],
                "include_bias_column": bool(reduced_system["include_bias_column"]),
            },
            solver_options=dict(solver_options or {}),
        )
        if isinstance(result, dict):
            coeff_reduced = result.get("coeff", result.get("coefficients", None))
            if coeff_reduced is None:
                raise ValueError("custom normal-equation solver result must include 'coeff'.")
            solve_method = str(result.get("solve_method", result.get("solver", "custom_normal_equation_solver")))
        else:
            coeff_reduced = result
            solve_method = "custom_normal_equation_solver"
        coeff_reduced = torch.as_tensor(coeff_reduced, dtype=rhs.dtype, device=rhs.device).reshape_as(rhs)
        solver_name = "custom"
    else:
        coeff_reduced, solve_method = _solve_normal_equation_system(
            system_matrix,
            rhs,
            solver=solver,
            solver_options=solver_options,
        )
        solver_name = str(solver)

    coeff = torch.zeros((int(reduced_system["n_cols"]),), dtype=reduced_system["dtype"], device=reduced_system["device"])
    coeff.index_copy_(0, reduced_system["reduced_indices"], coeff_reduced)
    return {
        "coeff": coeff,
        "active_feature_indices": reduced_system["active_feature_indices"],
        "active_feature_count": int(reduced_system["active_feature_indices"].numel()),
        "emitted_feature_count": int(reduced_system["feature_count"]),
        "include_bias_column": bool(reduced_system["include_bias_column"]),
        "solve_method": solve_method,
        "normal_equation_solver": solver_name,
    }


def _solve_linear_ace_ridge_from_normal_equations(
    normal,
    *,
    ridge_alpha=0.0,
    feature_indices=None,
    solver="auto",
    solver_options=None,
    solver_callable=None,
):
    return solve_linear_ace_normal_equations(
        normal,
        ridge_alpha=ridge_alpha,
        feature_indices=feature_indices,
        solver=solver,
        solver_options=solver_options,
        solver_callable=solver_callable,
    )


_STRUCTURE_BALANCED_OBJECTIVE = "structure_balanced_train_scaled_E1_F1"


def _normalize_linear_fit_objective(fit_objective):
    if fit_objective is None:
        return None
    if isinstance(fit_objective, str):
        payload = {"kind": fit_objective}
    else:
        payload = dict(fit_objective)
    kind = str(payload.pop("kind", "")).strip()
    if kind != _STRUCTURE_BALANCED_OBJECTIVE:
        raise ValueError(
            "fit_objective.kind must be "
            f"{_STRUCTURE_BALANCED_OBJECTIVE!r}."
        )
    feature_minimum_scale = float(payload.pop("feature_minimum_scale", 1.0e-12))
    target_minimum_scale = float(payload.pop("target_minimum_scale", 1.0e-12))
    if payload:
        raise ValueError(
            "Unsupported fit_objective fields: " + ", ".join(sorted(payload))
        )
    if not np.isfinite(feature_minimum_scale) or feature_minimum_scale <= 0.0:
        raise ValueError("feature_minimum_scale must be finite and positive.")
    if not np.isfinite(target_minimum_scale) or target_minimum_scale <= 0.0:
        raise ValueError("target_minimum_scale must be finite and positive.")
    return {
        "kind": kind,
        "feature_minimum_scale": feature_minimum_scale,
        "target_minimum_scale": target_minimum_scale,
    }


def _structure_balanced_scaled_design(records, objective):
    """Build the frozen structure-balanced, train-scaled ACE design."""
    if not records:
        raise ValueError("The scaled ACE objective needs at least one structure.")
    feature_count = int(np.asarray(records[0]["site_design"]).shape[1])
    for record in records:
        site = np.asarray(record["site_design"], dtype=np.float64)
        force_design = np.asarray(record["force_design"], dtype=np.float64)
        force_target = np.asarray(record["force_target"], dtype=np.float64).reshape(-1)
        if site.ndim != 2 or site.shape[1] != feature_count or site.shape[0] < 1:
            raise ValueError("Every scaled-fit site design must be nonempty and rectangular.")
        if force_design.shape != (3 * site.shape[0], feature_count):
            raise ValueError("Scaled-fit force-design shape does not match atom count.")
        if force_target.shape != (3 * site.shape[0],):
            raise ValueError("Scaled-fit force target shape does not match atom count.")
        energy_target = float(record["energy_target"])
        if not (
            np.all(np.isfinite(site))
            and np.all(np.isfinite(force_design))
            and np.all(np.isfinite(force_target))
            and np.isfinite(energy_target)
        ):
            raise ValueError("Scaled-fit designs and targets must be finite.")
    all_site = np.concatenate(
        [np.asarray(record["site_design"], dtype=np.float64) for record in records],
        axis=0,
    )
    feature_mean = np.mean(all_site, axis=0)
    feature_scale = np.maximum(
        np.std(all_site, axis=0),
        float(objective["feature_minimum_scale"]),
    )
    energy_per_atom = np.asarray(
        [
            float(record["energy_target"])
            / int(np.asarray(record["site_design"]).shape[0])
            for record in records
        ],
        dtype=np.float64,
    )
    all_forces = np.concatenate(
        [np.asarray(record["force_target"], dtype=np.float64).reshape(-1) for record in records]
    )
    energy_scale = max(
        float(np.std(energy_per_atom)),
        float(objective["target_minimum_scale"]),
    )
    force_scale = max(
        float(np.sqrt(np.mean(all_forces ** 2))),
        float(objective["target_minimum_scale"]),
    )
    structure_count = len(records)
    rows = []
    targets = []
    for record in records:
        site = np.asarray(record["site_design"], dtype=np.float64)
        atom_count = int(site.shape[0])
        energy_factor = 1.0 / (np.sqrt(structure_count) * energy_scale)
        energy_row = np.concatenate(
            ([1.0], (np.mean(site, axis=0) - feature_mean) / feature_scale)
        )
        rows.append(energy_factor * energy_row[None, :])
        targets.append(
            np.asarray(
                [energy_factor * float(record["energy_target"]) / atom_count],
                dtype=np.float64,
            )
        )
        force_columns = (
            np.asarray(record["force_design"], dtype=np.float64) / feature_scale
        )
        force_rows = np.column_stack(
            (np.zeros(force_columns.shape[0], dtype=np.float64), force_columns)
        )
        force_factor = 1.0 / (
            np.sqrt(structure_count * force_columns.shape[0]) * force_scale
        )
        rows.append(force_factor * force_rows)
        targets.append(
            force_factor
            * np.asarray(record["force_target"], dtype=np.float64).reshape(-1)
        )
    X = np.concatenate(rows, axis=0)
    y = np.concatenate(targets, axis=0)
    return X, y, {
        "fit_objective": dict(objective),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "energy_scale": float(energy_scale),
        "force_scale": float(force_scale),
        "structure_count": int(structure_count),
        "atom_count": int(all_site.shape[0]),
        "feature_count": int(feature_count),
        "column_order": "bias_then_train_scaled_features",
    }


def _structure_balanced_scale_metadata(
    *,
    feature_sum,
    feature_square_sum,
    atom_count,
    energy_per_atom,
    force_square_sum,
    force_component_count,
    objective,
):
    """Resolve train-only scales from bounded sufficient statistics."""
    feature_sum = np.asarray(feature_sum, dtype=np.float64)
    feature_square_sum = np.asarray(feature_square_sum, dtype=np.float64)
    atom_count = int(atom_count)
    force_component_count = int(force_component_count)
    energy_per_atom = np.asarray(energy_per_atom, dtype=np.float64)
    if atom_count < 1 or force_component_count < 1 or energy_per_atom.size < 1:
        raise ValueError("The scaled objective requires nonempty energy and force targets.")
    if feature_sum.shape != feature_square_sum.shape or feature_sum.ndim != 1:
        raise ValueError("Feature sufficient statistics have inconsistent shapes.")
    feature_mean = feature_sum / atom_count
    feature_variance = feature_square_sum / atom_count - feature_mean * feature_mean
    roundoff_floor = 64.0 * np.finfo(np.float64).eps * np.maximum(
        feature_square_sum / atom_count,
        1.0,
    )
    if np.any(feature_variance < -roundoff_floor):
        raise FloatingPointError("Feature variance became materially negative.")
    feature_variance = np.maximum(feature_variance, 0.0)
    feature_scale = np.maximum(
        np.sqrt(feature_variance),
        float(objective["feature_minimum_scale"]),
    )
    energy_scale = max(
        float(np.std(energy_per_atom)),
        float(objective["target_minimum_scale"]),
    )
    force_scale = max(
        float(np.sqrt(float(force_square_sum) / force_component_count)),
        float(objective["target_minimum_scale"]),
    )
    return {
        "fit_objective": dict(objective),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "energy_scale": energy_scale,
        "force_scale": force_scale,
        "structure_count": int(energy_per_atom.size),
        "atom_count": atom_count,
        "feature_count": int(feature_sum.size),
        "force_component_count": force_component_count,
        "column_order": "bias_then_train_scaled_features",
        "statistics_passes": 2,
    }


def _accumulate_structure_balanced_scaled_normal(
    XtX,
    Xty,
    yty,
    *,
    site_design,
    energy_target,
    force_design,
    force_target,
    scale_metadata,
    structure_weight,
    energy_weight,
    force_weight,
):
    """Accumulate one complete structure after atom-force contributions are summed."""
    site = np.asarray(site_design, dtype=np.float64)
    force_columns = np.asarray(force_design, dtype=np.float64)
    force_values = np.asarray(force_target, dtype=np.float64).reshape(-1)
    feature_count = int(scale_metadata["feature_count"])
    if site.ndim != 2 or site.shape[0] < 1 or site.shape[1] != feature_count:
        raise ValueError("Streamed site design has an inconsistent shape.")
    if force_columns.shape != (3 * site.shape[0], feature_count):
        raise ValueError("Streamed force design must contain one final Cartesian row.")
    if force_values.shape != (force_columns.shape[0],):
        raise ValueError("Streamed force target has an inconsistent shape.")
    if not (
        np.all(np.isfinite(site))
        and np.all(np.isfinite(force_columns))
        and np.all(np.isfinite(force_values))
        and np.isfinite(float(energy_target))
    ):
        raise ValueError("Streamed scaled-fit rows and targets must be finite.")
    row_weight = float(structure_weight)
    energy_weight = float(energy_weight)
    force_weight = float(force_weight)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (row_weight, energy_weight, force_weight)
    ):
        raise ValueError("Fit weights must be finite and nonnegative.")
    feature_mean = np.asarray(scale_metadata["feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(scale_metadata["feature_scale"], dtype=np.float64)
    structure_count = int(scale_metadata["structure_count"])
    energy_scale = float(scale_metadata["energy_scale"])
    force_scale = float(scale_metadata["force_scale"])

    energy_factor_squared = (
        row_weight * energy_weight / (structure_count * energy_scale * energy_scale)
    )
    energy_row = np.concatenate(
        ([1.0], (np.mean(site, axis=0) - feature_mean) / feature_scale)
    )
    energy_value = float(energy_target) / int(site.shape[0])
    XtX += energy_factor_squared * np.outer(energy_row, energy_row)
    Xty += energy_factor_squared * energy_row * energy_value
    yty += energy_factor_squared * energy_value * energy_value

    force_factor_squared = (
        row_weight
        * force_weight
        / (structure_count * force_columns.shape[0] * force_scale * force_scale)
    )
    scaled_force = force_columns / feature_scale
    XtX[1:, 1:] += force_factor_squared * (scaled_force.T @ scaled_force)
    Xty[1:] += force_factor_squared * (scaled_force.T @ force_values)
    yty += force_factor_squared * float(np.dot(force_values, force_values))
    return yty


def _solve_structure_balanced_scaled_ridge_from_normal(
    normal,
    scale_metadata,
    ridge_alpha,
    maximum_condition=1.0e8,
):
    """Solve a well-conditioned FP64 streamed Gram system and fold coefficients back."""
    XtX = np.asarray(normal["XtX"], dtype=np.float64)
    Xty = np.asarray(normal["Xty"], dtype=np.float64).reshape(-1)
    alpha = float(ridge_alpha)
    maximum_condition = float(maximum_condition)
    feature_count = int(scale_metadata["feature_count"])
    if XtX.shape != (feature_count + 1, feature_count + 1) or Xty.shape != (
        feature_count + 1,
    ):
        raise ValueError("Streamed normal-equation shapes are inconsistent.")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("ridge_alpha must be finite and nonnegative.")
    if not math.isfinite(maximum_condition) or maximum_condition <= 1.0:
        raise ValueError("maximum_condition must be finite and greater than one.")
    symmetry_defect = float(np.max(np.abs(XtX - XtX.T)))
    symmetry_scale = max(float(np.max(np.abs(XtX))), 1.0)
    symmetry_tolerance = 256.0 * np.finfo(np.float64).eps * symmetry_scale
    if symmetry_defect > symmetry_tolerance:
        raise FloatingPointError("Streamed Gram matrix is not symmetric within FP64 roundoff.")
    design_gram = 0.5 * (XtX + XtX.T)
    eigenvalues = np.linalg.eigvalsh(design_gram)
    negative_tolerance = 1024.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(eigenvalues))),
        1.0,
    )
    if float(eigenvalues[0]) < -negative_tolerance:
        raise FloatingPointError("Streamed Gram matrix is materially indefinite.")
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]

    system = design_gram.copy()
    if alpha > 0.0:
        system[1:, 1:] += alpha * np.eye(feature_count, dtype=np.float64)
    system_eigenvalues = np.linalg.eigvalsh(system)
    positive = system_eigenvalues[system_eigenvalues > 0.0]
    condition_number = (
        math.inf
        if positive.size != system_eigenvalues.size
        else float(positive[-1] / positive[0])
    )
    if not math.isfinite(condition_number) or condition_number > maximum_condition:
        raise np.linalg.LinAlgError(
            "Streamed Gram system exceeds the configured FP64 condition limit."
        )
    beta = np.linalg.solve(system, Xty)
    denominator = (
        float(np.linalg.norm(system, ord=2)) * float(np.linalg.norm(beta))
        + float(np.linalg.norm(Xty))
    )
    backward_error = float(np.linalg.norm(system @ beta - Xty)) / max(
        denominator,
        np.finfo(np.float64).tiny,
    )
    backward_limit = 100.0 * (feature_count + 1) * np.finfo(np.float64).eps
    if backward_error > backward_limit:
        raise np.linalg.LinAlgError("Streamed Gram solve failed its FP64 backward-error gate.")
    feature_scale = np.asarray(scale_metadata["feature_scale"], dtype=np.float64)
    feature_mean = np.asarray(scale_metadata["feature_mean"], dtype=np.float64)
    weight = beta[1:] / feature_scale
    bias = float(beta[0] - np.dot(feature_mean, weight))
    positive_singular = singular_values[singular_values > 0.0]
    return weight, bias, {
        "alpha": alpha,
        "scaled_coefficients": beta.tolist(),
        "scaled_coefficient_l2": float(np.linalg.norm(beta[1:])),
        "physical_coefficient_l2": float(np.linalg.norm(weight)),
        "singular_values": singular_values.tolist(),
        "effective_rank": int(positive_singular.size),
        "condition_number": (
            None
            if positive_singular.size != singular_values.size
            else float(positive_singular[0] / positive_singular[-1])
        ),
        "normal_system_condition_number": condition_number,
        "normal_system_maximum_condition": maximum_condition,
        "normal_system_backward_error": backward_error,
        "normal_system_backward_error_limit": backward_limit,
        "gram_symmetry_defect": symmetry_defect,
        "solver": "numpy.linalg.solve_streamed_gram",
        "intercept_penalized": False,
    }


def _solve_structure_balanced_scaled_ridge(X, y, metadata, ridge_alpha, svd_rcond):
    """Solve the exact augmented FP64 SVD objective and fold back coefficients."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    alpha = float(ridge_alpha)
    rcond = float(svd_rcond)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("ridge_alpha must be finite and nonnegative.")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("svd_rcond must be finite and positive.")
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] < 2:
        raise ValueError("Scaled ridge design and target shapes are inconsistent.")
    feature_count = int(X.shape[1] - 1)
    singular_values = np.linalg.svd(X, compute_uv=False)
    threshold = rcond * float(singular_values[0])
    effective_rank = int(np.count_nonzero(singular_values > threshold))
    condition_number = (
        None
        if float(singular_values[-1]) == 0.0
        else float(singular_values[0] / singular_values[-1])
    )
    if alpha > 0.0:
        penalty = np.zeros((feature_count, feature_count + 1), dtype=np.float64)
        penalty[:, 1:] = np.eye(feature_count, dtype=np.float64)
        augmented_X = np.concatenate((X, np.sqrt(alpha) * penalty), axis=0)
        augmented_y = np.concatenate((y, np.zeros(feature_count, dtype=np.float64)))
    else:
        augmented_X = X
        augmented_y = y
    beta = np.linalg.lstsq(augmented_X, augmented_y, rcond=rcond)[0]
    feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(metadata["feature_scale"], dtype=np.float64)
    if feature_mean.shape != (feature_count,) or feature_scale.shape != (feature_count,):
        raise ValueError("Scaled ridge metadata does not match the design columns.")
    weight = beta[1:] / feature_scale
    bias = float(beta[0] - np.dot(feature_mean, weight))
    report = {
        "alpha": alpha,
        "svd_rcond": rcond,
        "scaled_coefficients": beta.tolist(),
        "scaled_coefficient_l2": float(np.linalg.norm(beta[1:])),
        "physical_coefficient_l2": float(np.linalg.norm(weight)),
        "singular_values": singular_values.tolist(),
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "unaugmented_shape": [int(value) for value in X.shape],
        "augmented_shape": [int(value) for value in augmented_X.shape],
        "solver": "numpy.linalg.lstsq_augmented_design",
        "intercept_penalized": False,
    }
    return weight, bias, report


def _build_structure_balanced_scaled_problem_streaming(
    structures,
    *,
    settings,
    descriptors,
    site_basis_config,
    cutoff,
    type_map,
    energy_key,
    force_key,
    energy_weight,
    force_weight,
    backend,
    strict_backend,
    validate_backend,
    device,
    factorized_descriptor_runtime_policy,
    force_jacobian_mode,
    force_jacobian_chunk_size,
    objective,
    structure_weights,
    materialize_design=False,
    scale_metadata=None,
    progress=None,
):
    """Build a two-pass scaled Gram system or bounded dense fallback design."""
    structures = list(structures)
    if not structures:
        raise ValueError("The streamed scaled objective needs at least one structure.")
    device = _resolve_torch_device(device)
    normalized_mode = str(force_jacobian_mode).strip().lower().replace("-", "_")
    if normalized_mode not in {
        "analytic_product_adjoint",
        "analytic_adjoint",
        "analytic_vjp",
        "factorized_analytic",
    }:
        raise ValueError(
            "The streamed structure-balanced objective requires "
            "force_jacobian_mode='analytic_product_adjoint'."
        )
    cfg = site_basis_config
    evaluator = ACECovariantEvaluator(
        cfg,
        backend=backend,
        strict_backend=strict_backend,
        validate_backend=validate_backend,
        factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
    )
    descriptor_precompile = evaluator.precompile_descriptors(descriptors)
    feature_count = int(len(tuple(descriptors)))
    weights = np.asarray(structure_weights, dtype=np.float64)
    if weights.shape != (len(structures),):
        raise ValueError("Streamed structure weights must match the fit structures.")

    computed_scale_pass = scale_metadata is None
    if computed_scale_pass:
        feature_sum = np.zeros((feature_count,), dtype=np.float64)
        feature_square_sum = np.zeros((feature_count,), dtype=np.float64)
        energy_per_atom = []
        force_square_sum = 0.0
        force_component_count = 0
        atom_count = 0
        for structure_index, atoms in enumerate(structures, start=1):
            if progress is not None:
                progress("scale", structure_index, len(structures))
            energy_ref = _reference_energy(atoms, energy_key)
            force_ref = _reference_forces(atoms, force_key)
            if energy_ref is None or force_ref is None:
                raise ValueError(
                    "Every structure needs both energy and force targets for the "
                    "streamed structure-balanced objective."
                )
            site = _descriptor_site_design_for_structure(
                atoms,
                evaluator=evaluator,
                descriptors=descriptors,
                cutoff=cutoff,
                type_map=type_map,
                device=device,
            )
            if tuple(site.shape) != (len(atoms), feature_count):
                raise ValueError("Per-site descriptor shape does not match the fit request.")
            feature_sum += site.sum(dim=0).cpu().numpy()
            feature_square_sum += (site * site).sum(dim=0).cpu().numpy()
            atom_count += int(len(atoms))
            energy_per_atom.append(float(energy_ref) / len(atoms))
            force_values = np.asarray(force_ref, dtype=np.float64).reshape(-1)
            if force_values.shape != (3 * len(atoms),):
                raise ValueError("Force target shape does not match the atom count.")
            if not np.all(np.isfinite(force_values)):
                raise ValueError("Force targets must be finite.")
            force_square_sum += float(np.dot(force_values, force_values))
            force_component_count += int(force_values.size)
        scale_metadata = _structure_balanced_scale_metadata(
            feature_sum=feature_sum,
            feature_square_sum=feature_square_sum,
            atom_count=atom_count,
            energy_per_atom=energy_per_atom,
            force_square_sum=force_square_sum,
            force_component_count=force_component_count,
            objective=objective,
        )
    else:
        scale_metadata = dict(scale_metadata)
        if int(scale_metadata["feature_count"]) != feature_count:
            raise ValueError("Reused scale metadata has the wrong feature count.")

    n_rows = int(scale_metadata["structure_count"]) + int(
        scale_metadata["force_component_count"]
    )
    if materialize_design:
        X = np.empty((n_rows, feature_count + 1), dtype=np.float64)
        y = np.empty((n_rows,), dtype=np.float64)
        cursor = 0
    else:
        XtX = np.zeros((feature_count + 1, feature_count + 1), dtype=np.float64)
        Xty = np.zeros((feature_count + 1,), dtype=np.float64)
        yty = 0.0

    feature_mean = np.asarray(scale_metadata["feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(scale_metadata["feature_scale"], dtype=np.float64)
    structure_count = int(scale_metadata["structure_count"])
    energy_scale = float(scale_metadata["energy_scale"])
    force_scale = float(scale_metadata["force_scale"])
    stage = "dense_fallback" if materialize_design else "normal_equations"
    for structure_index, atoms in enumerate(structures):
        if progress is not None:
            progress(stage, structure_index + 1, len(structures))
        energy_ref = _reference_energy(atoms, energy_key)
        force_ref = _reference_forces(atoms, force_key)
        pos = torch.tensor(
            np.asarray(atoms.positions, float),
            dtype=torch.float64,
            device=device,
        )
        nbr = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
        shifts = torch.tensor(
            np.asarray(nbr.shifts, float),
            dtype=torch.float64,
            device=device,
        )
        cell = torch.tensor(
            np.asarray(atoms.cell.array, float),
            dtype=torch.float64,
            device=device,
        )
        edge_index = torch.tensor(nbr.edge_index, dtype=torch.long, device=device)
        atom_types = torch.tensor(nbr.atom_types, dtype=torch.long, device=device)
        site, position_jacobian = descriptor_sum_position_jacobian_analytic_product(
            evaluator,
            pos,
            cell,
            edge_index,
            atom_types,
            descriptors,
            shifts=shifts,
            real_if_scalar=True,
            chunk_size=force_jacobian_chunk_size,
        )
        record = {
            "site_design": site.detach().cpu().numpy(),
            "energy_target": float(energy_ref),
            "force_design": (-position_jacobian.transpose(0, 1)).detach().cpu().numpy(),
            "force_target": np.asarray(force_ref, dtype=np.float64).reshape(-1),
        }
        if materialize_design:
            row_weight = float(weights[structure_index])
            energy_factor = math.sqrt(row_weight * float(energy_weight)) / (
                math.sqrt(structure_count) * energy_scale
            )
            site_array = np.asarray(record["site_design"], dtype=np.float64)
            X[cursor] = energy_factor * np.concatenate(
                ([1.0], (np.mean(site_array, axis=0) - feature_mean) / feature_scale)
            )
            y[cursor] = energy_factor * float(energy_ref) / len(atoms)
            cursor += 1
            force_array = np.asarray(record["force_design"], dtype=np.float64)
            force_factor = math.sqrt(row_weight * float(force_weight)) / (
                math.sqrt(structure_count * force_array.shape[0]) * force_scale
            )
            X[cursor : cursor + force_array.shape[0], 0] = 0.0
            X[cursor : cursor + force_array.shape[0], 1:] = (
                force_factor * force_array / feature_scale
            )
            y[cursor : cursor + force_array.shape[0]] = (
                force_factor * np.asarray(record["force_target"], dtype=np.float64)
            )
            cursor += force_array.shape[0]
        else:
            yty = _accumulate_structure_balanced_scaled_normal(
                XtX,
                Xty,
                yty,
                **record,
                scale_metadata=scale_metadata,
                structure_weight=float(weights[structure_index]),
                energy_weight=energy_weight,
                force_weight=force_weight,
            )

    metadata = {
        **dict(scale_metadata),
        "device": str(device),
        "descriptor_precompile": descriptor_precompile,
        "factorized_descriptor_runtime_policy": (
            None
            if factorized_descriptor_runtime_policy is None
            else str(factorized_descriptor_runtime_policy)
        ),
        "force_jacobian_mode": str(force_jacobian_mode),
        "force_jacobian_chunk_size": (
            None
            if force_jacobian_chunk_size is None
            else int(force_jacobian_chunk_size)
        ),
        "n_rows": n_rows,
        "n_cols": feature_count + 1,
        "retained_structure_count": 1,
        "retained_force_edge_contributions": False,
        "materialized_training_design": bool(materialize_design),
        "dataset_passes": 2 if computed_scale_pass else 1,
    }
    if materialize_design:
        if cursor != n_rows:
            raise RuntimeError("Streamed dense design row count changed between passes.")
        return {"X": X, "y": y, "metadata": metadata}
    return {"XtX": XtX, "Xty": Xty, "yty": yty, "metadata": metadata}


def build_linear_ace_regression_problem(
    structures,
    *,
    settings = None,
    descriptors,
    site_basis_config,
    cutoff,
    type_map,
    energy_key = "energy",
    force_key = "forces",
    energy_weight = 1.0,
    force_weight = 1.0,
    backend = "pytorch",
    strict_backend = False,
    validate_backend = True,
    descriptor_matrix_cache = None,
    use_descriptor_matrix_cache = False,
    return_cache_metadata = False,
    device = None,
    factorized_descriptor_runtime_policy = None,
    force_atom_stride = None,
    force_jacobian_mode = "product_adjoint",
    force_jacobian_chunk_size = None,
    fit_objective = None,
    structure_weights = None,
    structure_weight_key = None,
    structure_group_key = None,
    structure_group_weights = None,
    structure_group_default_weight = None,
    structure_group_normalize_mean = True,
    boltzmann_temperature_K = None,
    boltzmann_energy_key = None,
    boltzmann_weight_nugget = 0.0,
    boltzmann_weight_prefactor = 1.0,
    boltzmann_normalize_mean = True,
    min_structure_weight = 0.0,
    progress = None,
):
    structures = list(structures)
    device = _resolve_torch_device(device)
    normalized_objective = _normalize_linear_fit_objective(fit_objective)
    resolved_structure_weights, structure_weight_metadata = structure_fit_weights(
        structures,
        structure_weights=structure_weights,
        structure_weight_key=structure_weight_key,
        structure_group_key=structure_group_key,
        structure_group_weights=structure_group_weights,
        structure_group_default_weight=structure_group_default_weight,
        structure_group_normalize_mean=structure_group_normalize_mean,
        boltzmann_temperature_K=boltzmann_temperature_K,
        boltzmann_energy_key=boltzmann_energy_key,
        boltzmann_weight_nugget=boltzmann_weight_nugget,
        boltzmann_weight_prefactor=boltzmann_weight_prefactor,
        boltzmann_normalize_mean=boltzmann_normalize_mean,
        min_weight=min_structure_weight,
    )
    cache_metadata = {
        "enabled": bool(use_descriptor_matrix_cache and descriptor_matrix_cache),
        "hit": False,
        "path": None,
        "device": str(device),
        "factorized_descriptor_runtime_policy": None
        if factorized_descriptor_runtime_policy is None
        else str(factorized_descriptor_runtime_policy),
    }
    cache_key = None
    cache_path = None
    if use_descriptor_matrix_cache and descriptor_matrix_cache:
        cache_key = _linear_problem_cache_key(
            structures,
            settings=settings,
            descriptors=descriptors,
            site_basis_config=site_basis_config,
            cutoff=cutoff,
            type_map=type_map,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=energy_weight,
            force_weight=force_weight,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            device=str(device),
            factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
            force_atom_stride=force_atom_stride,
            force_jacobian_mode=force_jacobian_mode,
            force_jacobian_chunk_size=force_jacobian_chunk_size,
            fit_objective=normalized_objective,
        )
        cache_key = hashlib.sha256(
            (
                cache_key
                + json.dumps(
                    {
                        "structure_weights": [float(value) for value in resolved_structure_weights.tolist()],
                        "structure_weight_metadata": _jsonable_cache_payload(structure_weight_metadata),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = _descriptor_matrix_cache_location(descriptor_matrix_cache, cache_key)
        cache_metadata["path"] = None if cache_path is None else str(cache_path)
        cache_metadata["cache_key"] = cache_key
        cached = _load_descriptor_matrix_cache(cache_path, cache_key)
        if cached is not None:
            X, y, loaded_metadata = cached
            cache_metadata.update(loaded_metadata)
            cache_metadata["enabled"] = True
            cache_metadata["hit"] = True
            cache_metadata["path"] = None if cache_path is None else str(cache_path)
            if return_cache_metadata:
                return X, y, cache_metadata
            return X, y

    cfg = site_basis_config
    evaluator = ACECovariantEvaluator(
        cfg,
        backend=backend,
        strict_backend=strict_backend,
        validate_backend=validate_backend,
        factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
    )
    cache_metadata["descriptor_precompile"] = evaluator.precompile_descriptors(descriptors)
    if normalized_objective is not None:
        if force_atom_stride is not None:
            raise ValueError(
                "The structure-balanced scaled objective does not permit force subsampling."
            )
        if not np.isclose(float(energy_weight), 1.0) or not np.isclose(
            float(force_weight), 1.0
        ):
            raise ValueError(
                "The structure-balanced_train_scaled_E1_F1 objective requires "
                "energy_weight=force_weight=1."
            )
        if not np.allclose(resolved_structure_weights, 1.0, rtol=0.0, atol=0.0):
            raise ValueError(
                "The structure-balanced scaled objective already weights each structure "
                "equally and does not accept additional structure weights."
            )
        normalized_mode = str(force_jacobian_mode).strip().lower().replace("-", "_")
        if normalized_mode not in {
            "analytic_product_adjoint",
            "analytic_adjoint",
            "analytic_vjp",
            "factorized_analytic",
        }:
            raise ValueError(
                "The structure-balanced scaled objective requires "
                "force_jacobian_mode='analytic_product_adjoint'."
            )
        records = []
        for structure_index, atoms in enumerate(structures, start=1):
            if progress is not None:
                progress("dense_design", structure_index, len(structures))
            energy_ref = _reference_energy(atoms, energy_key)
            force_ref = _reference_forces(atoms, force_key)
            if energy_ref is None or force_ref is None:
                raise ValueError(
                    "Every structure needs both energy and force targets for the "
                    "structure-balanced_train_scaled_E1_F1 objective."
                )
            pos = torch.tensor(
                np.asarray(atoms.positions, float),
                dtype=torch.float64,
                device=device,
            )
            nbr = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
            shifts = torch.tensor(
                np.asarray(nbr.shifts, float),
                dtype=torch.float64,
                device=device,
            )
            cell = torch.tensor(
                np.asarray(atoms.cell.array, float),
                dtype=torch.float64,
                device=device,
            )
            edge_index = torch.tensor(nbr.edge_index, dtype=torch.long, device=device)
            atom_types = torch.tensor(nbr.atom_types, dtype=torch.long, device=device)
            site_design, position_jacobian = (
                descriptor_sum_position_jacobian_analytic_product(
                    evaluator,
                    pos,
                    cell,
                    edge_index,
                    atom_types,
                    descriptors,
                    shifts=shifts,
                    real_if_scalar=True,
                    chunk_size=force_jacobian_chunk_size,
                )
            )
            records.append(
                {
                    "site_design": site_design.detach().cpu().numpy(),
                    "energy_target": float(energy_ref),
                    "force_design": (
                        -position_jacobian.transpose(0, 1)
                    ).detach().cpu().numpy(),
                    "force_target": np.asarray(force_ref, dtype=np.float64).reshape(-1),
                }
            )
        X, y, objective_metadata = _structure_balanced_scaled_design(
            records,
            normalized_objective,
        )
        cache_metadata.update(objective_metadata)
        cache_metadata["n_rows"] = int(X.shape[0])
        cache_metadata["n_cols"] = int(X.shape[1])
        cache_metadata["force_atom_stride"] = None
        cache_metadata["force_jacobian_mode"] = str(force_jacobian_mode)
        cache_metadata["structure_weights"] = dict(structure_weight_metadata)
        if use_descriptor_matrix_cache and descriptor_matrix_cache and cache_path is not None:
            _write_descriptor_matrix_cache(cache_path, cache_key, X, y, cache_metadata)
            cache_metadata["path"] = str(cache_path)
            cache_metadata["written"] = True
        if return_cache_metadata:
            return X, y, cache_metadata
        return X, y
    X_rows = []
    y_rows = []
    sqrt_energy_weight = float(np.sqrt(max(energy_weight, 0.0)))
    sqrt_force_weight = float(np.sqrt(max(force_weight, 0.0)))
    for structure_index, atoms in enumerate(structures):
        sqrt_structure_weight = float(np.sqrt(max(float(resolved_structure_weights[int(structure_index)]), 0.0)))
        energy_row, energy_ref, force_matrix, force_ref = _build_linear_problem_for_structure(
            atoms,
            evaluator=evaluator,
            descriptors=descriptors,
            cutoff=cutoff,
            type_map=type_map,
            energy_key=energy_key,
            force_key=force_key,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            device=device,
            force_atom_stride=force_atom_stride,
        )
        if energy_row is not None and energy_ref is not None and sqrt_energy_weight > 0.0:
            X_rows.append(sqrt_energy_weight * sqrt_structure_weight * energy_row)
            y_rows.append(sqrt_energy_weight * sqrt_structure_weight * float(energy_ref))
        if force_matrix is not None and force_ref is not None and sqrt_force_weight > 0.0:
            X_rows.extend(list(sqrt_force_weight * sqrt_structure_weight * force_matrix))
            y_rows.extend(list(sqrt_force_weight * sqrt_structure_weight * force_ref))
    if not X_rows:
        raise ValueError("No energy or force targets were found for the requested linear ACE fit.")
    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    cache_metadata["n_rows"] = int(X.shape[0])
    cache_metadata["n_cols"] = int(X.shape[1])
    cache_metadata["force_atom_stride"] = None if force_atom_stride is None else int(force_atom_stride)
    cache_metadata["structure_weights"] = dict(structure_weight_metadata)
    if use_descriptor_matrix_cache and descriptor_matrix_cache and cache_path is not None:
        _write_descriptor_matrix_cache(cache_path, cache_key, X, y, cache_metadata)
        cache_metadata["path"] = str(cache_path)
        cache_metadata["written"] = True
    if return_cache_metadata:
        return X, y, cache_metadata
    return X, y


def fit_linear_ace_from_xyz(
    xyz_path,
    *,
    settings,
    site_basis_config,
    type_map,
    cutoff,
    coupling_library = None,
    compact_labels = None,
    max_variants_per_label = 1,
    basis_mode = None,
    energy_key = "energy",
    force_key = "forces",
    epochs = 200,
    lr = 5e-2,
    energy_weight = 1.0,
    force_weight = 1.0,
    l2 = None,
    ridge_alpha = None,
    fit_method = "adam",
    fit_objective = None,
    svd_rcond = 1.0e-12,
    sklearn_params = None,
    descriptor_cache = None,
    use_descriptor_cache = True,
    descriptor_matrix_cache = None,
    use_descriptor_matrix_cache = False,
    backend = "pytorch",
    strict_backend = False,
    validate_backend = True,
    device = None,
    factorized_descriptor_runtime_policy = None,
    force_jacobian_mode = "product_adjoint",
    force_jacobian_chunk_size = None,
    force_atom_stride = None,
    include_bias_column = True,
    structure_weights = None,
    structure_weight_key = None,
    structure_group_key = None,
    structure_group_weights = None,
    structure_group_default_weight = None,
    structure_group_normalize_mean = True,
    boltzmann_temperature_K = None,
    boltzmann_energy_key = None,
    boltzmann_weight_nugget = 0.0,
    boltzmann_weight_prefactor = 1.0,
    boltzmann_normalize_mean = True,
    min_structure_weight = 0.0,
):
    """Fit a scalar linear ACE model from extended XYZ structures."""
    structures = load_xyz_structures(xyz_path)
    return fit_linear_ace(
        structures,
        settings=settings,
        site_basis_config=site_basis_config,
        coupling_library=coupling_library,
        compact_labels=compact_labels,
        max_variants_per_label=max_variants_per_label,
        type_map=type_map,
        cutoff=cutoff,
        basis_mode=basis_mode,
        energy_key=energy_key,
        force_key=force_key,
        epochs=epochs,
        lr=lr,
        energy_weight=energy_weight,
        force_weight=force_weight,
        l2=l2,
        ridge_alpha=ridge_alpha,
        fit_method=fit_method,
        fit_objective=fit_objective,
        svd_rcond=svd_rcond,
        sklearn_params=sklearn_params,
        descriptor_cache=descriptor_cache,
        use_descriptor_cache=use_descriptor_cache,
        descriptor_matrix_cache=descriptor_matrix_cache,
        use_descriptor_matrix_cache=use_descriptor_matrix_cache,
        backend=backend,
        strict_backend=strict_backend,
        validate_backend=validate_backend,
        device=device,
        factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
        force_jacobian_mode=force_jacobian_mode,
        force_jacobian_chunk_size=force_jacobian_chunk_size,
        force_atom_stride=force_atom_stride,
        include_bias_column=include_bias_column,
        structure_weights=structure_weights,
        structure_weight_key=structure_weight_key,
        structure_group_key=structure_group_key,
        structure_group_weights=structure_group_weights,
        structure_group_default_weight=structure_group_default_weight,
        structure_group_normalize_mean=structure_group_normalize_mean,
        boltzmann_temperature_K=boltzmann_temperature_K,
        boltzmann_energy_key=boltzmann_energy_key,
        boltzmann_weight_nugget=boltzmann_weight_nugget,
        boltzmann_weight_prefactor=boltzmann_weight_prefactor,
        boltzmann_normalize_mean=boltzmann_normalize_mean,
        min_structure_weight=min_structure_weight,
    )


def fit_linear_ace_from_config(
    config,
    structures = None,
):
    """Fit a scalar linear ACE model from a dictionary-style workflow config.

    The config stores descriptor settings, site-basis settings, target keys,
    weights, backend options, and the scikit-learn fit method. Structures may
    be supplied directly or loaded from ``xyz_path``.
    """
    warnings.warn(
        "fit_linear_ace_from_config is a legacy config-driven helper. "
        "Use YE3TDescriptors.ace(...) plus YE3TModel.linear(...) for new workflows.",
        FutureWarning,
        stacklevel=2,
    )
    cfg = dict(config)
    settings_payload = cfg.get("settings", cfg.get("descriptor_settings", None))
    if settings_payload is None:
        raise KeyError("Linear ACE config requires a settings or descriptor_settings entry.")
    settings = DescriptorGenerationSettings.from_dict(settings_payload)

    site_basis_payload = cfg.get("site_basis_config", cfg.get("site_basis_config_payload", None))
    if site_basis_payload is None:
        raise KeyError("Linear ACE config requires a site_basis_config or site_basis_config_payload entry.")
    site_basis_config = (
        deserialize_site_basis_config(site_basis_payload)
        if isinstance(site_basis_payload, dict)
        else site_basis_payload
    )
    if structures is None:
        xyz_path = cfg.get("xyz_path", None)
        if xyz_path is None:
            raise KeyError("Pass structures or provide xyz_path in the linear ACE config.")
        structures = load_xyz_structures(xyz_path)
    structures = select_workflow_frames(
        structures,
        frame_indices=cfg.get("frame_indices", None),
        max_frames=cfg.get("max_frames", None),
        max_structures=cfg.get("max_structures", None),
    )
    return fit_linear_ace(
        structures,
        settings=settings,
        site_basis_config=site_basis_config,
        type_map=dict(cfg["type_map"]),
        cutoff=float(cfg["cutoff"]),
        coupling_library=cfg.get("coupling_library", None),
        compact_labels=cfg.get("compact_labels", None),
        max_variants_per_label=cfg.get("max_variants_per_label", 1),
        basis_mode=cfg.get("basis_mode", None),
        energy_key=str(cfg.get("energy_key", "energy")),
        force_key=str(cfg.get("force_key", "forces")),
        epochs=int(cfg.get("epochs", 200)),
        lr=float(cfg.get("lr", 5.0e-2)),
        energy_weight=float(cfg.get("energy_weight", 1.0)),
        force_weight=float(cfg.get("force_weight", 1.0)),
        l2=None if "l2" not in cfg else float(cfg["l2"]),
        ridge_alpha=None
        if cfg.get("ridge_alpha", None) is None
        else float(cfg["ridge_alpha"]),
        fit_method=str(cfg.get("fit_method", "ridge")),
        fit_objective=cfg.get("fit_objective", None),
        svd_rcond=float(cfg.get("svd_rcond", 1.0e-12)),
        sklearn_params=cfg.get("sklearn_params", None),
        descriptor_cache=cfg.get("descriptor_cache", None),
        use_descriptor_cache=bool(cfg.get("use_descriptor_cache", True)),
        descriptor_matrix_cache=cfg.get("descriptor_matrix_cache", None),
        use_descriptor_matrix_cache=bool(cfg.get("use_descriptor_matrix_cache", False)),
        backend=str(cfg.get("backend", "pytorch")),
        strict_backend=bool(cfg.get("strict_backend", False)),
        validate_backend=bool(cfg.get("validate_backend", True)),
        device=cfg.get("device", None),
        factorized_descriptor_runtime_policy=cfg.get("factorized_descriptor_runtime_policy", None),
        force_jacobian_chunk_size=cfg.get("force_jacobian_chunk_size", None),
        force_jacobian_mode=cfg.get("force_jacobian_mode", "product_adjoint"),
        normal_equation_solver=cfg.get("normal_equation_solver", "auto"),
        normal_equation_solver_options=cfg.get("normal_equation_solver_options", None),
        force_atom_stride=cfg.get("force_atom_stride", None),
        feature_budget=cfg.get("feature_budget", cfg.get("ACE_feature_budget", None)),
        feature_selection_policy=cfg.get("feature_selection_policy", cfg.get("ACE_feature_selection_policy", "normal_diagonal")),
        selected_feature_indices=cfg.get("selected_feature_indices", None),
        include_bias_column=bool(cfg.get("include_bias_column", cfg.get("fit_intercept", True))),
        structure_weights=cfg.get("structure_weights", None),
        structure_weight_key=cfg.get("structure_weight_key", None),
        structure_group_key=cfg.get("structure_group_key", None),
        structure_group_weights=cfg.get("structure_group_weights", None),
        structure_group_default_weight=cfg.get("structure_group_default_weight", None),
        structure_group_normalize_mean=bool(
            cfg.get("structure_group_normalize_mean", True)
        ),
        boltzmann_temperature_K=cfg.get("boltzmann_temperature_K", cfg.get("boltzmann_temperature", None)),
        boltzmann_energy_key=cfg.get("boltzmann_energy_key", None),
        boltzmann_weight_nugget=float(cfg.get("boltzmann_weight_nugget", 0.0)),
        boltzmann_weight_prefactor=float(cfg.get("boltzmann_weight_prefactor", 1.0)),
        boltzmann_normalize_mean=bool(cfg.get("boltzmann_normalize_mean", True)),
        min_structure_weight=float(cfg.get("min_structure_weight", 0.0)),
    )


def fit_linear_ace(
    structures,
    *,
    settings,
    site_basis_config,
    type_map,
    cutoff,
    descriptor_specs = None,
    coupling_library = None,
    compact_labels = None,
    max_variants_per_label = 1,
    basis_mode = None,
    energy_key = "energy",
    force_key = "forces",
    stress_key = "stress",
    epochs = 200,
    lr = 5e-2,
    energy_weight = 1.0,
    force_weight = 1.0,
    stress_weight = 0.0,
    l2 = None,
    ridge_alpha = None,
    fit_method = "adam",
    fit_objective = None,
    svd_rcond = 1.0e-12,
    sklearn_params = None,
    descriptor_cache = None,
    use_descriptor_cache = True,
    descriptor_matrix_cache = None,
    use_descriptor_matrix_cache = False,
    backend = "pytorch",
    strict_backend = False,
    validate_backend = True,
    device = None,
    factorized_descriptor_runtime_policy = None,
    force_jacobian_mode = "product_adjoint",
    force_jacobian_chunk_size = None,
    normal_equation_solver = "auto",
    normal_equation_solver_options = None,
    normal_equation_solver_callable = None,
    force_atom_stride = None,
    feature_budget = None,
    feature_selection_policy = "normal_diagonal",
    selected_feature_indices = None,
    include_bias_column = True,
    structure_weights = None,
    structure_weight_key = None,
    structure_group_key = None,
    structure_group_weights = None,
    structure_group_default_weight = None,
    structure_group_normalize_mean = True,
    boltzmann_temperature_K = None,
    boltzmann_energy_key = None,
    boltzmann_weight_nugget = 0.0,
    boltzmann_weight_prefactor = 1.0,
    boltzmann_normalize_mean = True,
    min_structure_weight = 0.0,
    progress = None,
):
    structures = list(structures)
    if not structures:
        raise ValueError("Need at least one structure to fit a linear ACE model.")
    method = str(fit_method).lower()
    l2_supplied = l2 is not None
    resolved_l2 = 1.0e-10 if l2 is None else float(l2)
    if not math.isfinite(resolved_l2) or resolved_l2 < 0.0:
        raise ValueError("l2 must be finite and non-negative.")
    if not math.isfinite(float(stress_weight)) or float(stress_weight) < 0.0:
        raise ValueError("stress_weight must be finite and non-negative.")
    if stress_weight and method not in {"ridge_normal_equations", "ridge_normal", "normal_equations"}:
        raise ValueError("Density stress rows currently require ridge_normal_equations.")
    if ridge_alpha is not None:
        ridge_alpha = float(ridge_alpha)
        if not math.isfinite(ridge_alpha) or ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be finite and non-negative.")
    if ridge_alpha is not None and l2_supplied and not np.isclose(
        ridge_alpha,
        resolved_l2,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("ridge_alpha and its legacy l2 alias disagree.")
    streaming_scaled_methods = {
        "ridge_streaming_gram",
        "streaming_ridge",
        "streamed_ridge",
    }
    if method in {
        "ridge_augmented_svd",
        "augmented_svd",
        "structure_balanced_ridge",
    } | streaming_scaled_methods:
        svd_rcond = float(svd_rcond)
        if not math.isfinite(svd_rcond) or svd_rcond <= 0.0:
            raise ValueError("svd_rcond must be finite and positive.")
    if descriptor_specs is None:
        descriptors, descriptor_metadata = _descriptor_specs_from_settings(
            settings=settings,
            site_basis_config=site_basis_config,
            coupling_library=coupling_library,
            basis_mode=normalize_basis_mode(basis_mode, L_R=settings.L_R),
            compact_labels=compact_labels,
            max_variants_per_label=max_variants_per_label,
            descriptor_cache=descriptor_cache,
            use_descriptor_cache=use_descriptor_cache,
            return_metadata=True,
        )
    else:
        if max_variants_per_label is not None:
            raise ValueError(
                "descriptor_specs already resolve chemical variants; apply "
                "max_variants_per_label while constructing the descriptor catalogue."
            )
        descriptors = list(descriptor_specs)
        if not descriptors:
            raise ValueError("descriptor_specs must contain at least one descriptor.")
        descriptor_metadata = {
            "coupling_source": "resolved_descriptor_specs",
            "coupling_cache_enabled": bool(use_descriptor_cache),
            "coupling_library_supplied": coupling_library is not None,
            "compact_label_count": int(
                len({normalize_compact_label(spec.label) for spec in descriptors})
            ),
            "descriptor_cache_stats": (
                None if descriptor_cache is None else descriptor_cache.get_stats()
            ),
        }
    if fit_objective is not None and method not in {
        "ridge_augmented_svd",
        "augmented_svd",
        "structure_balanced_ridge",
    } | streaming_scaled_methods:
        raise ValueError(
            "fit_objective is currently supported only by "
            "fit_method='ridge_augmented_svd' or 'ridge_streaming_gram'."
        )
    include_bias_column = bool(include_bias_column)
    if not include_bias_column and method not in {"ridge_normal_equations", "ridge_normal", "normal_equations"}:
        raise ValueError("include_bias_column=False is currently implemented for ridge_normal_equations fits.")
    fit_device = _resolve_torch_device(device)
    resolved_structure_weights, structure_weight_metadata = structure_fit_weights(
        structures,
        structure_weights=structure_weights,
        structure_weight_key=structure_weight_key,
        structure_group_key=structure_group_key,
        structure_group_weights=structure_group_weights,
        structure_group_default_weight=structure_group_default_weight,
        structure_group_normalize_mean=structure_group_normalize_mean,
        boltzmann_temperature_K=boltzmann_temperature_K,
        boltzmann_energy_key=boltzmann_energy_key,
        boltzmann_weight_nugget=boltzmann_weight_nugget,
        boltzmann_weight_prefactor=boltzmann_weight_prefactor,
        boltzmann_normalize_mean=boltzmann_normalize_mean,
        min_weight=min_structure_weight,
    )
    fit_metadata = {"fit_method": fit_method}
    fit_metadata["backend"] = str(backend)
    fit_metadata["device"] = str(fit_device)
    fit_metadata["factorized_descriptor_runtime_policy"] = (
        None
        if factorized_descriptor_runtime_policy is None
        else str(factorized_descriptor_runtime_policy)
    )
    fit_metadata.update(descriptor_metadata)
    if method == "adam":
        evaluator = ACECovariantEvaluator(
            site_basis_config,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
            force_atom_stride=force_atom_stride,
        )
        fit_metadata["descriptor_precompile"] = evaluator.precompile_descriptors(descriptors)
        model = LinearScalarACEModel(len(descriptors)).double().to(device=fit_device)
        if ridge_alpha is not None:
            raise ValueError("ridge_alpha is not an Adam optimizer option; use l2.")
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=resolved_l2)

        for _ in range(int(epochs)):
            opt.zero_grad()
            loss = torch.zeros((), dtype=torch.float64, device=fit_device)
            for structure_index, atoms in enumerate(structures):
                row_weight = torch.as_tensor(
                    float(resolved_structure_weights[int(structure_index)]),
                    dtype=torch.float64,
                    device=fit_device,
                )
                pos = torch.tensor(
                    np.asarray(atoms.positions, float),
                    dtype=torch.float64,
                    device=fit_device,
                    requires_grad=True,
                )
                nbr = neighbor_data_from_ase_atoms(atoms, cutoff, type_map)
                shifts = torch.tensor(np.asarray(nbr.shifts, float), dtype=torch.float64, device=fit_device)
                cell = torch.tensor(np.asarray(atoms.cell.array, float), dtype=torch.float64, device=fit_device)
                edge_index = torch.tensor(nbr.edge_index, dtype=torch.long, device=fit_device)
                atom_types = torch.tensor(nbr.atom_types, dtype=torch.long, device=fit_device)
                x_ij = pos[edge_index[1]] - pos[edge_index[0]] + shifts @ cell
                B = evaluator(x_ij=x_ij, edge_index=edge_index, atom_types=atom_types, descriptors=descriptors, real_if_scalar=True)
                pred_energy = model(B).sum()
                ref_energy = _reference_energy(atoms, energy_key)
                if ref_energy is not None:
                    ref_e = torch.tensor(float(ref_energy), dtype=torch.float64, device=fit_device)
                    loss = loss + row_weight * energy_weight * (pred_energy - ref_e) ** 2
                ref_forces = _reference_forces(atoms, force_key)
                if ref_forces is not None:
                    ref_f = torch.tensor(np.asarray(ref_forces, float), dtype=torch.float64, device=fit_device)
                    pred_f = -torch.autograd.grad(pred_energy, pos, create_graph=True)[0]
                    loss = loss + row_weight * force_weight * torch.mean((pred_f - ref_f) ** 2)
            loss.backward()
            opt.step()
        weight = model.weight.detach().cpu().numpy().copy()
        bias = float(model.bias.detach().cpu().item())
        fit_metadata["epochs"] = int(epochs)
        fit_metadata["lr"] = float(lr)
        fit_metadata["l2"] = float(resolved_l2)
    elif method in streaming_scaled_methods:
        if not include_bias_column:
            raise ValueError(
                "The structure-balanced scaled objective requires an unpenalized intercept."
            )
        if feature_budget is not None or selected_feature_indices is not None:
            raise ValueError(
                "Feature selection must be frozen before the structure-balanced scaled fit."
            )
        if sklearn_params:
            raise ValueError(
                "ridge_streaming_gram uses ridge_alpha and normal_equation_solver_options, "
                "not sklearn_params."
            )
        if normal_equation_solver_callable is not None:
            raise ValueError("ridge_streaming_gram does not accept a custom solver callable.")
        if use_descriptor_matrix_cache or descriptor_matrix_cache is not None:
            raise ValueError(
                "ridge_streaming_gram keeps sufficient statistics in memory and does not "
                "use a descriptor-matrix cache."
            )
        objective = _normalize_linear_fit_objective(
            fit_objective
            if fit_objective is not None
            else {"kind": _STRUCTURE_BALANCED_OBJECTIVE}
        )
        alpha = resolved_l2 if ridge_alpha is None else ridge_alpha
        options = dict(normal_equation_solver_options or {})
        maximum_condition = float(options.pop("maximum_condition", 1.0e8))
        ill_conditioned_policy = str(
            options.pop("ill_conditioned_policy", "streamed_dense_svd")
        ).strip().lower()
        maximum_dense_fallback_bytes = int(
            options.pop("maximum_dense_fallback_bytes", 512 * 1024 * 1024)
        )
        if options:
            raise ValueError(
                "Unsupported ridge_streaming_gram solver options: "
                + ", ".join(sorted(options))
            )
        if ill_conditioned_policy not in {"error", "streamed_dense_svd"}:
            raise ValueError(
                "ill_conditioned_policy must be 'error' or 'streamed_dense_svd'."
            )
        if maximum_dense_fallback_bytes < 1:
            raise ValueError("maximum_dense_fallback_bytes must be positive.")
        problem = _build_structure_balanced_scaled_problem_streaming(
            structures,
            settings=settings,
            descriptors=descriptors,
            site_basis_config=site_basis_config,
            cutoff=cutoff,
            type_map=type_map,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=energy_weight,
            force_weight=force_weight,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            device=device,
            factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
            force_jacobian_mode=force_jacobian_mode,
            force_jacobian_chunk_size=force_jacobian_chunk_size,
            objective=objective,
            structure_weights=resolved_structure_weights,
            progress=progress,
        )
        fallback_reason = None
        try:
            weight, bias, solve_report = _solve_structure_balanced_scaled_ridge_from_normal(
                problem,
                problem["metadata"],
                alpha,
                maximum_condition=maximum_condition,
            )
        except np.linalg.LinAlgError as exc:
            fallback_reason = str(exc)
            if ill_conditioned_policy == "error":
                raise
            n_rows = int(problem["metadata"]["n_rows"])
            n_cols = int(problem["metadata"]["n_cols"])
            required_bytes = 8 * (n_rows * n_cols + n_rows)
            if required_bytes > maximum_dense_fallback_bytes:
                raise MemoryError(
                    "The streamed Gram system needs an SVD fallback, but the bounded "
                    f"design would require {required_bytes} bytes, above the configured "
                    f"limit {maximum_dense_fallback_bytes}."
                ) from exc
            dense = _build_structure_balanced_scaled_problem_streaming(
                structures,
                settings=settings,
                descriptors=descriptors,
                site_basis_config=site_basis_config,
                cutoff=cutoff,
                type_map=type_map,
                energy_key=energy_key,
                force_key=force_key,
                energy_weight=energy_weight,
                force_weight=force_weight,
                backend=backend,
                strict_backend=strict_backend,
                validate_backend=validate_backend,
                device=device,
                factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
                force_jacobian_mode=force_jacobian_mode,
                force_jacobian_chunk_size=force_jacobian_chunk_size,
                objective=objective,
                structure_weights=resolved_structure_weights,
                materialize_design=True,
                scale_metadata=problem["metadata"],
                progress=progress,
            )
            weight, bias, solve_report = _solve_structure_balanced_scaled_ridge(
                dense["X"],
                dense["y"],
                dense["metadata"],
                alpha,
                svd_rcond,
            )
            solve_report["fallback_from"] = "streamed_gram"
            solve_report["fallback_reason"] = fallback_reason
            solve_report["fallback_design_bytes"] = required_bytes
            solve_report["fallback_design_byte_limit"] = maximum_dense_fallback_bytes
            problem["metadata"]["materialized_training_design"] = True
            problem["metadata"]["dataset_passes"] = 3
            problem["metadata"]["fallback_dataset_passes"] = 1
            problem["metadata"]["fallback_design_bytes"] = required_bytes
        fit_metadata["n_rows"] = int(problem["metadata"]["n_rows"])
        fit_metadata["n_cols"] = int(problem["metadata"]["n_cols"])
        fit_metadata["include_bias_column"] = True
        fit_metadata["intercept_policy"] = "unpenalized_per_atom_bias"
        fit_metadata["alpha"] = float(alpha)
        fit_metadata["fit_objective"] = dict(objective)
        fit_metadata["scaled_ridge"] = solve_report
        fit_metadata["streaming_regression"] = dict(problem["metadata"])
        fit_metadata["descriptor_matrix_cache"] = {
            "enabled": False,
            "hit": False,
            "reason": "streamed_sufficient_statistics",
        }
        fit_metadata["descriptor_precompile"] = problem["metadata"].get(
            "descriptor_precompile",
            {},
        )
        fit_metadata["force_atom_stride"] = None
        fit_metadata["normal_equation_solver"] = solve_report["solver"]
        fit_metadata["feature_selection"] = {
            "status": "not_requested",
            "active_feature_count": int(len(descriptors)),
            "emitted_feature_count": int(len(descriptors)),
        }
    elif method in {
        "ridge_augmented_svd",
        "augmented_svd",
        "structure_balanced_ridge",
    }:
        if not include_bias_column:
            raise ValueError(
                "The structure-balanced scaled objective requires an unpenalized intercept."
            )
        if feature_budget is not None or selected_feature_indices is not None:
            raise ValueError(
                "Feature selection must be frozen before the structure-balanced scaled fit."
            )
        if sklearn_params:
            raise ValueError(
                "ridge_augmented_svd uses ridge_alpha and svd_rcond, not sklearn_params."
            )
        if normal_equation_solver_callable is not None:
            raise ValueError(
                "ridge_augmented_svd does not accept a normal-equation solver callable."
            )
        objective = _normalize_linear_fit_objective(
            fit_objective
            if fit_objective is not None
            else {"kind": _STRUCTURE_BALANCED_OBJECTIVE}
        )
        alpha = resolved_l2 if ridge_alpha is None else ridge_alpha
        X, y, matrix_cache_metadata = build_linear_ace_regression_problem(
            structures,
            settings=settings,
            descriptors=descriptors,
            site_basis_config=site_basis_config,
            cutoff=cutoff,
            type_map=type_map,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=energy_weight,
            force_weight=force_weight,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            descriptor_matrix_cache=descriptor_matrix_cache,
            use_descriptor_matrix_cache=use_descriptor_matrix_cache,
            return_cache_metadata=True,
            device=device,
            factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
            force_atom_stride=force_atom_stride,
            force_jacobian_mode=force_jacobian_mode,
            force_jacobian_chunk_size=force_jacobian_chunk_size,
            fit_objective=objective,
            structure_weights=resolved_structure_weights,
            boltzmann_temperature_K=None,
            progress=progress,
        )
        weight, bias, solve_report = _solve_structure_balanced_scaled_ridge(
            X,
            y,
            matrix_cache_metadata,
            alpha,
            svd_rcond,
        )
        fit_metadata["n_rows"] = int(X.shape[0])
        fit_metadata["n_cols"] = int(X.shape[1])
        fit_metadata["include_bias_column"] = True
        fit_metadata["intercept_policy"] = "unpenalized_per_atom_bias"
        fit_metadata["alpha"] = float(alpha)
        fit_metadata["fit_objective"] = dict(objective)
        fit_metadata["scaled_ridge"] = solve_report
        fit_metadata["descriptor_matrix_cache"] = matrix_cache_metadata
        fit_metadata["descriptor_precompile"] = matrix_cache_metadata.get(
            "descriptor_precompile",
            {},
        )
        fit_metadata["force_atom_stride"] = None
        fit_metadata["normal_equation_solver"] = None
        fit_metadata["feature_selection"] = {
            "status": "not_requested",
            "active_feature_count": int(len(descriptors)),
            "emitted_feature_count": int(len(descriptors)),
        }
    elif method in {"ridge_normal_equations", "ridge_normal", "normal_equations"}:
        normal = build_linear_ace_normal_equations(
            structures,
            settings=settings,
            descriptors=descriptors,
            site_basis_config=site_basis_config,
            cutoff=cutoff,
            type_map=type_map,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            return_metadata=True,
            device=device,
            factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
            force_jacobian_mode=force_jacobian_mode,
            force_jacobian_chunk_size=force_jacobian_chunk_size,
            force_atom_stride=force_atom_stride,
            include_bias_column=include_bias_column,
            structure_weights=resolved_structure_weights,
            boltzmann_temperature_K=None,
        )
        XtX = normal["XtX"]
        Xty = normal["Xty"]
        configured_alpha = (sklearn_params or {}).get("alpha", resolved_l2)
        if ridge_alpha is not None:
            if "alpha" in (sklearn_params or {}) and not np.isclose(
                float(configured_alpha),
                float(ridge_alpha),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError("ridge_alpha and sklearn_params['alpha'] disagree.")
            configured_alpha = ridge_alpha
        alpha = float(configured_alpha)
        if feature_budget is not None or selected_feature_indices is not None:
            active_indices = (
                _select_linear_ace_feature_indices_from_normal(
                    normal,
                    max_features=feature_budget,
                    policy=feature_selection_policy,
                )
                if selected_feature_indices is None
                else selected_feature_indices
            )
            solve = _solve_linear_ace_ridge_from_normal_equations(
                normal,
                ridge_alpha=alpha,
                feature_indices=active_indices,
                solver=normal_equation_solver,
                solver_options=normal_equation_solver_options,
                solver_callable=normal_equation_solver_callable,
            )
            coef_t = solve["coeff"]
            solver = solve["solve_method"]
            fit_metadata["feature_selection"] = {
                "status": "active",
                "policy": str(feature_selection_policy),
                "feature_budget": None if feature_budget is None else int(feature_budget),
                "active_feature_count": int(solve["active_feature_count"]),
                "emitted_feature_count": int(solve["emitted_feature_count"]),
                "selected_feature_indices": [
                    int(index) for index in solve["active_feature_indices"].detach().cpu().tolist()
                ],
                "intercept_penalized": False,
                "include_bias_column": bool(include_bias_column),
                "normal_equation_solver": str(solve.get("normal_equation_solver", normal_equation_solver)),
            }
        else:
            solve = solve_linear_ace_normal_equations(
                normal,
                ridge_alpha=alpha,
                solver=normal_equation_solver,
                solver_options=normal_equation_solver_options,
                solver_callable=normal_equation_solver_callable,
            )
            coef_t = solve["coeff"]
            solver = solve["solve_method"]
            fit_metadata["feature_selection"] = {
                "status": "not_requested",
                "active_feature_count": int(normal["metadata"].get("feature_count", normal["n_cols"] - 1)),
                "emitted_feature_count": int(normal["metadata"].get("feature_count", normal["n_cols"] - 1)),
                "intercept_penalized": bool(alpha > 0.0 and not include_bias_column),
                "include_bias_column": bool(include_bias_column),
                "normal_equation_solver": str(solve.get("normal_equation_solver", normal_equation_solver)),
            }
        coef = coef_t.detach().cpu().numpy().reshape(-1)
        if include_bias_column:
            weight = coef[:-1].copy()
            bias = float(coef[-1])
        else:
            weight = coef.copy()
            bias = 0.0
        fit_metadata["n_rows"] = int(normal["n_rows"])
        fit_metadata["n_cols"] = int(normal["n_cols"])
        fit_metadata["normal_equations"] = dict(normal["metadata"])
        fit_metadata["include_bias_column"] = bool(include_bias_column)
        fit_metadata["intercept_policy"] = "atom_count_bias_column" if include_bias_column else "disabled_reference_energy_fit"
        fit_metadata["alpha"] = float(alpha)
        fit_metadata["normal_equation_solver"] = solver
        fit_metadata["normal_equation_solver_requested"] = (
            "custom" if normal_equation_solver_callable is not None else str(normal_equation_solver)
        )
        fit_metadata["descriptor_precompile"] = normal["metadata"].get("descriptor_precompile", {})
        fit_metadata["force_atom_stride"] = None if force_atom_stride is None else int(force_atom_stride)
    else:
        X, y, matrix_cache_metadata = build_linear_ace_regression_problem(
            structures,
            settings=settings,
            descriptors=descriptors,
            site_basis_config=site_basis_config,
            cutoff=cutoff,
            type_map=type_map,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=energy_weight,
            force_weight=force_weight,
            backend=backend,
            strict_backend=strict_backend,
            validate_backend=validate_backend,
            descriptor_matrix_cache=descriptor_matrix_cache,
            use_descriptor_matrix_cache=use_descriptor_matrix_cache,
            return_cache_metadata=True,
            device=device,
            factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
            force_atom_stride=force_atom_stride,
            force_jacobian_mode=force_jacobian_mode,
            force_jacobian_chunk_size=force_jacobian_chunk_size,
            structure_weights=resolved_structure_weights,
            boltzmann_temperature_K=None,
        )
        estimator = _make_sklearn_estimator(method, sklearn_params=sklearn_params)
        estimator.fit(X, y)
        coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        weight = coef[:-1].copy()
        bias = float(coef[-1])
        fit_metadata["n_rows"] = int(X.shape[0])
        fit_metadata["n_cols"] = int(X.shape[1])
        fit_metadata["descriptor_matrix_cache"] = matrix_cache_metadata
        if sklearn_params:
            fit_metadata["sklearn_params"] = dict(sklearn_params)
        if hasattr(estimator, "alpha_"):
            fit_metadata["alpha_"] = float(estimator.alpha_)
        elif hasattr(estimator, "alpha"):
            alpha_value = getattr(estimator, "alpha")
            if np.isscalar(alpha_value):
                fit_metadata["alpha"] = float(alpha_value)
        if method in {"ard", "ardregression"}:
            coefficient_precision = np.asarray(estimator.lambda_, dtype=np.float64)
            threshold = float(estimator.threshold_lambda)
            active = np.flatnonzero(coefficient_precision < threshold)
            covariance = np.asarray(estimator.sigma_, dtype=np.float64)
            if covariance.shape != (active.size, active.size):
                raise RuntimeError("ARD posterior covariance does not match active columns.")
            fit_metadata["predictive_uncertainty"] = {
                "schema": "ye3t_linear_ard_posterior_v1",
                "status": "python_offline_only",
                "design_column_order": "descriptor_features_then_atom_count_bias",
                "active_column_indices": active.tolist(),
                "coefficient_precision": coefficient_precision.tolist(),
                "coefficient_covariance_active": covariance.tolist(),
                "noise_precision": float(estimator.alpha_),
                "threshold_lambda": threshold,
                "variance_formula": "x_active @ sigma @ x_active.T (epistemic readout only)",
                "observation_variance_formula": "1/alpha + x_active @ sigma @ x_active.T",
            }
        fit_metadata["descriptor_precompile"] = matrix_cache_metadata.get(
            "descriptor_precompile",
            {
                "runtime_path": "precompiled_descriptor_contraction_tables",
                "descriptor_count": int(len(descriptors)),
            },
        )

    fit_metadata["structure_weights"] = dict(structure_weight_metadata)
    return LinearACEScalarModelBundle(
        settings=settings,
        site_basis_config=site_basis_config,
        descriptor_specs=descriptors,
        weight=weight,
        bias=bias,
        basis_mode=normalize_basis_mode(basis_mode, L_R=settings.L_R),
        fit_method=fit_method,
        fit_metadata=fit_metadata,
    )


def evaluate_linear_ace_bundle(
    bundle,
    structures,
    *,
    cutoff,
    type_map,
    energy_key = "energy",
    force_key = "forces",
    backend = "pytorch",
    strict_backend = False,
    validate_backend = True,
    device = None,
    factorized_descriptor_runtime_policy = None,
):
    structures = list(structures)
    if not structures:
        raise ValueError("Need at least one structure to evaluate a linear ACE model.")
    calc = LinearACEScalarCalculator(
        bundle=bundle,
        cutoff=cutoff,
        type_map=type_map,
        backend=backend,
        strict_backend=strict_backend,
        validate_backend=validate_backend,
        device=device,
        factorized_descriptor_runtime_policy=factorized_descriptor_runtime_policy,
    )
    energy_abs_errors = []
    energy_squared_errors = []
    energy_per_atom_abs_errors = []
    energy_per_atom_squared_errors = []
    force_abs_errors = []
    force_squared_errors = []
    balanced_objectives = []
    fit_objective = dict(bundle.fit_metadata.get("fit_objective", {}) or {})
    scaled_metadata = dict(
        bundle.fit_metadata.get("descriptor_matrix_cache", {}) or {}
    )
    balanced_kind = fit_objective.get("kind") == _STRUCTURE_BALANCED_OBJECTIVE
    energy_scale = float(scaled_metadata.get("energy_scale", 0.0))
    force_scale = float(scaled_metadata.get("force_scale", 0.0))
    for atoms in structures:
        energy_pred, pos, _ = calc._energy_from_atoms(atoms)
        ref_energy = _reference_energy(atoms, energy_key)
        energy_error_per_atom = None
        if ref_energy is not None:
            error = float(energy_pred.detach().cpu().item()) - float(ref_energy)
            energy_abs_errors.append(abs(error))
            energy_squared_errors.append(error ** 2)
            per_atom_error = error / max(len(atoms), 1)
            energy_error_per_atom = per_atom_error
            energy_per_atom_abs_errors.append(abs(per_atom_error))
            energy_per_atom_squared_errors.append(per_atom_error ** 2)
        ref_forces = _reference_forces(atoms, force_key)
        force_mean_squared_error = None
        if ref_forces is not None:
            grad = torch.autograd.grad(energy_pred, pos)[0]
            pred_forces = (-grad).detach().cpu().numpy()
            force_delta = pred_forces - np.asarray(ref_forces, float)
            force_mean_squared_error = float(np.mean(force_delta ** 2))
            force_abs_errors.append(np.abs(force_delta))
            force_squared_errors.append(force_delta.reshape(-1) ** 2)
        if (
            balanced_kind
            and energy_error_per_atom is not None
            and force_mean_squared_error is not None
            and energy_scale > 0.0
            and force_scale > 0.0
        ):
            balanced_objectives.append(
                (energy_error_per_atom / energy_scale) ** 2
                + force_mean_squared_error / (force_scale ** 2)
            )
    metrics = {"num_structures": float(len(structures))}
    if energy_abs_errors:
        metrics["energy_mae"] = float(np.mean(energy_abs_errors))
        metrics["energy_rmse"] = float(np.sqrt(np.mean(energy_squared_errors)))
        metrics["energy_mae_per_atom"] = float(np.mean(energy_per_atom_abs_errors))
        metrics["energy_rmse_per_atom"] = float(np.sqrt(np.mean(energy_per_atom_squared_errors)))
    if force_abs_errors:
        metrics["forces_mae"] = float(np.mean(np.concatenate([x.reshape(-1) for x in force_abs_errors])))
        metrics["forces_rmse"] = float(np.sqrt(np.mean(np.concatenate(force_squared_errors))))
    if balanced_objectives:
        metrics[_STRUCTURE_BALANCED_OBJECTIVE] = float(
            np.mean(balanced_objectives)
        )
    return metrics


def export_scalar_bundle_to_yace(
    bundle,
    path,
    *,
    elements,
    compatibility = None,
    imaginary_tolerance = 1.0e-12,
):
    """Export a scalar linear ACE model to a `.yace`-style YACE basis file.

    Each compiler-produced descriptor contributes one function and the fitted
    linear coefficient is folded into its coupling coefficients. The optional
    ``lammps_pace_linear_v1`` profile emits a complete standalone file accepted
    by stock ML-PACE and rejects conventions that cannot be represented
    exactly by that profile.
    """
    strict_pace = compatibility == "lammps_pace_linear_v1"
    if compatibility is not None and not strict_pace:
        raise ValueError(
            "compatibility: unsupported profile "
            f"{compatibility!r}; expected 'lammps_pace_linear_v1'"
        )
    _validate_bundle_ordinary_scalar_catalogue(bundle)
    by_mu0 = _bundle_to_yace_functions(
        bundle,
        strict_pace=strict_pace,
        imaginary_tolerance=imaginary_tolerance,
    )
    if strict_pace:
        blocks = _pace_linear_yace_blocks(bundle, elements, by_mu0)
        return write_yace(
            path,
            elements=elements,
            functions_by_mu0=by_mu0,
            E0=blocks["E0"],
            embeddings=blocks["embeddings"],
            bonds=blocks["bonds"],
            delta_spline_bins=blocks["delta_spline_bins"],
            compatibility=compatibility,
        )
    out_path = write_yace(
        path,
        elements=elements,
        functions_by_mu0=by_mu0,
        metadata={
            "fitted_bias": float(bundle.bias),
            "basis_mode": bundle.basis_mode,
        },
    )
    return out_path


def _validate_bundle_ordinary_scalar_catalogue(bundle):
    metadata = dict(getattr(bundle, "fit_metadata", {}) or {})
    catalogue = metadata.get("ordinary_scalar_catalogue", None)
    if catalogue is None:
        return
    from ye3t_methods.atomistic.ace.descriptors import _normalize_ordinary_scalar_catalogue

    catalogue = dict(catalogue)
    base_fields = {
        key: catalogue[key]
        for key in (
            "schema",
            "profile_id",
            "feature_ids",
            "membership_sha256",
            "rows",
            "compiler",
            "application_sha256",
        )
    }
    normalized, _, _ = _normalize_ordinary_scalar_catalogue(base_fields)
    if normalized["application_sha256"] != catalogue["application_sha256"]:
        raise ValueError(
            "Fitted ordinary scalar catalogue application identity does not match."
        )
    compiled_coordinates = tuple(catalogue.get("compiled_coordinates", ()))
    if len(compiled_coordinates) != len(normalized["rows"]):
        raise ValueError(
            "Fitted ordinary scalar catalogue compiler-record count does not match."
        )
    for expected_row, compiled in zip(normalized["rows"], compiled_coordinates):
        if str(compiled.get("feature_id", "")) != expected_row["feature_id"]:
            raise ValueError(
                "Fitted ordinary scalar catalogue compiler-record order does not match."
            )
        for name, expected_value in expected_row["expected_compiler"].items():
            if str(compiled.get(name, "")) != str(expected_value):
                raise ValueError(
                    "Fitted ordinary scalar catalogue compiler identity does not match."
                )
    descriptor_rows = tuple(catalogue.get("descriptor_rows", ()))
    if len(descriptor_rows) != len(bundle.descriptor_specs):
        raise ValueError(
            "Fitted ordinary scalar catalogue descriptor-row count does not match."
        )
    catalogue_row_by_id = {
        row["feature_id"]: row for row in normalized["rows"]
    }
    variant_index_by_feature = {}
    for descriptor_index, (row, spec) in enumerate(
        zip(descriptor_rows, bundle.descriptor_specs)
    ):
        feature_id = str(row.get("feature_id", ""))
        catalogue_row = catalogue_row_by_id.get(feature_id)
        expected_variant = int(variant_index_by_feature.get(feature_id, 0))
        if (
            int(row.get("descriptor_index", -1)) != descriptor_index
            or str(row.get("descriptor_key", "")) != str(spec.key)
            or catalogue_row is None
            or normalize_compact_label(spec.label).to_dict()
            != catalogue_row["compact_label"]
            or int(row.get("variant_index", -1)) != expected_variant
        ):
            raise ValueError(
                "Fitted ordinary scalar catalogue descriptor order does not match."
            )
        variant_index_by_feature[feature_id] = expected_variant + 1
        if not spec.ms_combinations or len(spec.ms_combinations) != len(spec.coeffs):
            raise ValueError(
                "Fitted ordinary scalar catalogue contains empty or mismatched "
                "coupling rows."
            )


def _pace_linear_yace_blocks(bundle, elements, functions_by_mu0):
    if not isinstance(bundle, LinearACEScalarModelBundle):
        raise TypeError("strict PACE export requires a LinearACEScalarModelBundle")
    elements = tuple(str(element) for element in elements)
    if not elements or any(not element for element in elements):
        raise ValueError("strict PACE export requires nonempty element names")
    if len(set(elements)) != len(elements):
        raise ValueError("strict PACE export requires unique element names")

    cfg = bundle.site_basis_config
    if any(str(spec.key).endswith("|physical_eta_bound")
           for spec in bundle.descriptor_specs):
        raise ValueError("strict PACE export has no validated physical eta binding lowering")
    if str(cfg.chemical_basis) != "delta" or getattr(cfg, "chemical_embedding", None) is not None:
        raise ValueError("strict PACE export requires delta chemical channels")
    expected_types = tuple(range(len(elements)))
    if tuple(int(value) for value in cfg.possible_types) != expected_types:
        raise ValueError(
            "strict PACE export requires possible_types ordered as contiguous "
            f"indices {expected_types}"
        )
    settings_elements = tuple(str(value) for value in bundle.settings.elems)
    if settings_elements != elements:
        raise ValueError(
            "strict PACE export element order must match the fitted descriptor "
            f"settings; got {elements} and {settings_elements}"
        )
    radial_kind = str(cfg.radial_basis).strip().lower()
    radial_kind = radial_kind.replace("_", "").replace("-", "")
    if radial_kind not in {
        "pacechebexpcos",
        "pacechebexpcosidentityspline",
    }:
        raise ValueError(
            "strict PACE export requires radial_basis='PACE_ChebExpCos'"
        )
    if str(cfg.pace_crad_policy) != "identity":
        raise ValueError("strict PACE export requires identity radial contractions")
    if str(cfg.spherical_backend) != "complex":
        raise ValueError("strict PACE export requires complex spherical harmonics")
    if str(cfg.spherical_normalization) != "pace_y00_one":
        raise ValueError("strict PACE export requires spherical_normalization='pace_y00_one'")
    if str(cfg.atomic_base_normalization) != "none":
        raise ValueError("strict PACE export requires atomic_base_normalization='none'")
    if str(cfg.factor_normalization) != "none":
        raise ValueError("strict PACE export requires factor_normalization='none'")
    if str(cfg.charge_mode) != "none" or int(cfg.kmax) != 0:
        raise ValueError("strict PACE export does not support charge channels")
    if int(bundle.settings.L_R) != 0 or tuple(
        int(value) for value in bundle.settings.M_R_values
    ) != (0,):
        raise ValueError("strict PACE export requires scalar L=0, M=0 descriptors")
    if str(bundle.settings.basis_type) != "no_charge":
        raise ValueError("strict PACE export requires the ordinary no-charge ACE basis")

    expected_buckets = set(expected_types)
    if set(functions_by_mu0) != expected_buckets or any(
        not functions_by_mu0[mu0] for mu0 in expected_types
    ):
        raise ValueError(
            "strict PACE export requires at least one fitted function for each "
            "central element"
        )
    spacing = tuple(float(value) for value in cfg.pace_spline_spacing)
    if not spacing or any(
        not np.isclose(value, spacing[0], rtol=0.0, atol=1.0e-15)
        for value in spacing[1:]
    ):
        raise ValueError(
            "strict PACE export requires one common pace_spline_spacing because "
            ".yace stores a single deltaSplineBins value"
        )

    reference_metadata = dict(
        bundle.fit_metadata.get("reference_energy_targets", {})
    )
    reference_energies = {}
    if bool(reference_metadata.get("enabled", False)):
        reference_energies = {
            str(key): float(value)
            for key, value in dict(
                reference_metadata.get("reference_energies", {})
            ).items()
        }
        missing = tuple(element for element in elements if element not in reference_energies)
        if missing:
            raise ValueError(
                "strict PACE export is missing fitted reference energies for "
                f"{missing}"
            )
    bias = float(bundle.bias)
    if not np.isfinite(bias):
        raise ValueError("strict PACE export requires a finite fitted bias")
    E0 = [bias + float(reference_energies.get(element, 0.0)) for element in elements]

    embeddings = {
        mu0: {
            "ndensity": 1,
            "FS_parameters": [1.0, 1.0],
            "npoti": "FinnisSinclair",
            "rho_core_cutoff": 100000.0,
            "drho_core_cutoff": 250.0,
        }
        for mu0 in expected_types
    }
    radial_count = int(cfg.nradmax)
    lmax = int(cfg.lmax)
    identity_coefficients = [
        [
            [
                1.0 if basis_index == radial_index else 0.0
                for basis_index in range(radial_count)
            ]
            for _ in range(lmax + 1)
        ]
        for radial_index in range(radial_count)
    ]
    bonds = {}
    for central in expected_types:
        for neighbor in expected_types:
            bond = central * len(elements) + neighbor
            rcut = float(cfg.rc[bond])
            dcut = float(cfg.pace_cutoff_width[bond])
            if not 0.0 < dcut < rcut:
                raise ValueError(
                    "strict PACE export requires 0 < pace_cutoff_width < rc for "
                    f"bond {(central, neighbor)}"
                )
            if (
                float(cfg.pace_inner_cutoff[bond]) != 0.0
                or float(cfg.pace_inner_cutoff_width[bond]) != 0.0
            ):
                raise ValueError("strict PACE export requires inactive inner cutoffs")
            lmbda = float(cfg.lmbda[bond])
            bonds[(central, neighbor)] = {
                "nradmax": radial_count,
                "lmax": lmax,
                "nradbasemax": radial_count,
                "radbasename": "ChebExpCos",
                "radparameters": [lmbda],
                "radcoefficients": [
                    [list(row) for row in angular_rows]
                    for angular_rows in identity_coefficients
                ],
                "prehc": 0.0,
                "lambdahc": lmbda,
                "rcut": rcut,
                "dcut": dcut,
                "rcut_in": 0.0,
                "dcut_in": 0.0,
                "inner_cutoff_type": "distance",
            }
    return {
        "E0": E0,
        "embeddings": embeddings,
        "bonds": bonds,
        "delta_spline_bins": spacing[0],
    }


def _bundle_to_yace_functions(
    bundle,
    *,
    strict_pace = False,
    imaginary_tolerance = 1.0e-12,
):
    if len(bundle.weight) != len(bundle.descriptor_specs):
        raise ValueError(
            "YACE export requires one fitted weight per descriptor specification"
        )
    tolerance = float(imaginary_tolerance)
    if strict_pace and (not np.isfinite(tolerance) or tolerance < 0.0):
        raise ValueError(
            "strict PACE export requires a finite nonnegative imaginary_tolerance"
        )
    by_mu0 = {}
    for coeff, spec in zip(bundle.weight, bundle.descriptor_specs):
        if not spec.channels:
            if strict_pace:
                raise ValueError(
                    "strict PACE export requires every descriptor to contain channels"
                )
            continue
        if strict_pace:
            if int(spec.L_R) != 0 or int(spec.M_R) != 0:
                raise ValueError(
                    "strict PACE export requires scalar L=0, M=0 descriptor rows"
                )
            central_types = {int(channel.mu0) for channel in spec.channels}
            if len(central_types) != 1:
                raise ValueError(
                    "strict PACE export requires one central type per descriptor"
                )
            for channel in spec.channels:
                if int(channel.kappa0) != 0 or int(channel.kappa) != 0:
                    raise ValueError(
                        "strict PACE export does not support charge channels"
                    )
                if channel.l_aux is not None or channel.m_aux is not None:
                    raise ValueError(
                        "strict PACE export does not support auxiliary angular channels"
                    )
            if len(spec.ms_combinations) != len(spec.coeffs):
                raise ValueError(
                    "strict PACE export requires one coupling coefficient per m row"
                )
            if not spec.ms_combinations:
                raise ValueError(
                    "strict PACE export requires nonempty coupling rows"
                )
            if any(len(row) != int(spec.rank) for row in spec.ms_combinations):
                raise ValueError(
                    "strict PACE export requires each m row to match descriptor rank"
                )
        learned_coefficient = complex(coeff)
        if not (
            np.isfinite(learned_coefficient.real)
            and np.isfinite(learned_coefficient.imag)
        ):
            raise ValueError("YACE export requires finite fitted coefficients")
        mu0 = int(spec.channels[0].mu0)
        by_mu0.setdefault(mu0, [])
        ms_flat = tuple(int(m) for row in spec.ms_combinations for m in row)
        yace_coefficients = []
        for coupling_coefficient in spec.coeffs:
            coupling_value = complex(coupling_coefficient)
            if not (
                np.isfinite(coupling_value.real)
                and np.isfinite(coupling_value.imag)
            ):
                raise ValueError(
                    "YACE export requires finite coupling coefficients"
                )
            value = learned_coefficient * coupling_value
            if not (np.isfinite(value.real) and np.isfinite(value.imag)):
                raise ValueError("YACE export requires finite C-tilde coefficients")
            if strict_pace and abs(value.imag) > tolerance:
                raise ValueError(
                    "strict PACE export encountered a material imaginary "
                    f"C-tilde coefficient ({value.imag})"
                )
            yace_coefficients.append(float(value.real))
        by_mu0[mu0].append(
            YACEFunction(
                mu0=mu0,
                rank=spec.rank,
                ndensity=1,
                num_ms_combs=len(spec.ms_combinations),
                mus=tuple(int(ch.mu) for ch in spec.channels),
                ns=tuple(int(ch.n) for ch in spec.channels),
                ls=tuple(int(ch.l) for ch in spec.channels),
                ms_combs=ms_flat,
                ctildes=tuple(yace_coefficients),
            )
        )
    return by_mu0

def save_linear_ace_ase_bundle(
    bundle,
    path,
    *,
    cutoff,
    type_map,
    reference_energies = None,
):
    output = Path(path)
    torch.save({
        "bundle": bundle,
        "cutoff": float(cutoff),
        "type_map": dict(type_map),
        "reference_energies": _normalize_reference_energies(reference_energies),
    }, output)
    return output


def save_linear_ace_multi_cutoff_ase_bundle(
    bundle,
    path,
):
    output = Path(path)
    torch.save({
        "bundle": bundle,
    }, output)
    return output


def load_linear_ace_ase_bundle(path):
    from ye3t_methods._saved_model_compat import ensure_saved_model_imports

    ensure_saved_model_imports()
    payload = torch.load(Path(path), weights_only=False)
    bundle = payload["bundle"]
    if not isinstance(bundle, LinearACEScalarModelBundle):
        raise TypeError(f"Expected LinearACEScalarModelBundle in {path!s}; got {type(bundle)!r}")
    return bundle, float(payload["cutoff"]), dict(payload["type_map"]), _normalize_reference_energies(payload.get("reference_energies", {}))


def load_linear_ace_multi_cutoff_ase_bundle(path):
    from ye3t_methods._saved_model_compat import ensure_saved_model_imports

    ensure_saved_model_imports()
    payload = torch.load(Path(path), weights_only=False)
    bundle = payload["bundle"]
    if not isinstance(bundle, LinearACEMultiCutoffModelBundle):
        raise TypeError(f"Expected LinearACEMultiCutoffModelBundle in {path!s}; got {type(bundle)!r}")
    return bundle


def load_linear_ace_calculator(path, **kwargs):
    bundle, cutoff, type_map, reference_energies = load_linear_ace_ase_bundle(path)
    kwargs.setdefault("reference_energies", reference_energies)
    return LinearACEScalarCalculator(bundle=bundle, cutoff=cutoff, type_map=type_map, **kwargs)


def load_linear_ace_multi_cutoff_calculator(path, **kwargs):
    bundle = load_linear_ace_multi_cutoff_ase_bundle(path)
    return LinearACEMultiCutoffCalculator(bundle=bundle, **kwargs)
