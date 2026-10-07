"""Linear A_s normal equations and readout fit, retained from the comparison harness."""

import numpy as np
import torch
from time import perf_counter

from ye3t_methods.atomistic.lifted_density import (
    slot_standard_norm,
    slot_trivial_component,
    ye3_power_invariants,
    ye3_slot_specht_power_invariants,
    ye3_slot_specht_power_commutant_density_adjoint,
    ye3_slot_specht_power_l0_density_adjoint,
    ye3_slot_specht_power_local_density_adjoint,
    ye3_slot_specht_power_rank2_density_adjoint,
)
from ye3t_methods.atomistic.utils.fit_weights import structure_fit_weights


DEFAULT_A_S_FORCE_JACOBIAN_CHUNK_SIZE = 16


def dtype_from_name(name):
    value = str(name).lower()
    if value in {"float32", "torch.float32"}:
        return torch.float32
    if value in {"float64", "torch.float64"}:
        return torch.float64
    raise ValueError("A_s linear fit dtype must be float32 or float64.")

def _reference_forces(atoms, force_key):
    if force_key in getattr(atoms, "arrays", {}):
        return torch.tensor(np.asarray(atoms.arrays[force_key], float), dtype=torch.float64)
    calc = getattr(atoms, "calc", None)
    if calc is not None and force_key in getattr(calc, "results", {}):
        return torch.tensor(np.asarray(calc.results[force_key], float), dtype=torch.float64)
    return torch.tensor(np.asarray(atoms.get_forces(), float), dtype=torch.float64)


def _reference_energy(atoms, energy_key):
    if energy_key in getattr(atoms, "info", {}):
        return float(atoms.info[energy_key])
    calc = getattr(atoms, "calc", None)
    if calc is not None and energy_key in getattr(calc, "results", {}):
        return float(calc.results[energy_key])
    if calc is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            return None
    return None


def _atom_types(atoms, type_map):
    return torch.tensor([int(type_map[str(symbol)]) for symbol in atoms.get_chemical_symbols()], dtype=torch.long)


def _structure_tensors(atoms, type_map, *, dtype, device, requires_grad=False):
    return {
        "positions": torch.tensor(
            np.asarray(atoms.positions, float),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        ),
        "atom_types": _atom_types(atoms, type_map).to(device=device),
        "cell": torch.tensor(np.asarray(atoms.cell.array, float), dtype=dtype, device=device),
        "pbc": torch.tensor(np.asarray(atoms.pbc, bool), dtype=torch.bool, device=device),
    }


def _features_from_density(model, density):
    mode = model.config.lifted_density.readout_mode
    if mode == "character_quadratic":
        return torch.cat((slot_trivial_component(density).sum(dim=0), slot_standard_norm(density).sum(dim=0)))
    if mode == "ye3_power":
        invariants, _blocks, _sectors = ye3_power_invariants(
            density,
            model.config.lifted_density.channels,
            max_power=model.config.lifted_density.ye3_max_power,
            optimization_policy=model.config.lifted_density.ye3_optimization_policy,
            slot_sectors=model.config.lifted_density.ye3_slot_sectors,
            include_rank1=model.config.lifted_density.ye3_include_rank1,
            rank_nmax=model.config.lifted_density.ye3_rank_nmax,
            rank_lmax=model.config.lifted_density.ye3_rank_lmax,
            rank_lmin=model.config.lifted_density.ye3_rank_lmin,
        )
        return invariants.sum(dim=0)
    if mode == "ye3_slot_specht_power":
        invariants, _blocks, _sectors = ye3_slot_specht_power_invariants(
            density,
            model.config.lifted_density.channels,
            max_power=model.config.lifted_density.ye3_max_power,
            optimization_policy=model.config.lifted_density.ye3_optimization_policy,
            slot_specht_partitions=model.config.lifted_density.ye3_slot_specht_partitions,
            slot_specht_coupling=model.config.lifted_density.ye3_slot_specht_coupling,
            include_rank1=model.config.lifted_density.ye3_include_rank1,
            rank_nmax=model.config.lifted_density.ye3_rank_nmax,
            rank_lmax=model.config.lifted_density.ye3_rank_lmax,
            rank_lmin=model.config.lifted_density.ye3_rank_lmin,
        )
        return invariants.sum(dim=0)
    if mode == "symmetric_linear":
        return density.sum(dim=(0, 1))
    raise ValueError(f"Unsupported linear lifted-density readout mode {mode!r}.")


def _linear_lifted_feature_count(model):
    mode = model.config.lifted_density.readout_mode
    if mode == "symmetric_linear":
        return int(model.channel_readout.numel())
    if mode == "character_quadratic":
        return int(model.character_mu_weight.numel() + model.character_nu_weight.numel())
    if mode == "ye3_power":
        return int(model.ye3_power_weight.numel())
    if mode == "ye3_slot_specht_power":
        return int(model.ye3_slot_specht_power_weight.numel())
    raise ValueError(f"Unsupported linear lifted-density readout mode {mode!r}.")


def _feature_row(model, atoms, type_map, *, dtype, device, requires_grad=False):
    tensors = _structure_tensors(atoms, type_map, dtype=dtype, device=device, requires_grad=requires_grad)
    density = model.filtered_density(
        tensors["positions"],
        tensors["atom_types"],
        cell=tensors["cell"],
        pbc=tensors["pbc"],
    )
    features = _features_from_density(model, density)
    atom_count = density.new_tensor([atoms.get_global_number_of_atoms()])
    return torch.cat((features, atom_count)), tensors


def _A_s_feature_sum_for_structure(model, atoms, type_map, *, dtype, device):
    tensors = _structure_tensors(atoms, type_map, dtype=dtype, device=device, requires_grad=False)
    atom_types = tensors["atom_types"]
    cell = tensors["cell"]
    pbc = tensors["pbc"]
    positions = tensors["positions"].detach().clone().requires_grad_(True)

    def evaluate_from_flat(flat_pos):
        reshaped = flat_pos.reshape_as(positions)
        density = model.filtered_density(
            reshaped,
            atom_types,
            cell=cell,
            pbc=pbc,
        )
        return _features_from_density(model, density)

    return positions, evaluate_from_flat


def _A_s_density_features_for_structure(model, atoms, type_map, *, dtype, device):
    tensors = _structure_tensors(atoms, type_map, dtype=dtype, device=device, requires_grad=False)
    atom_types = tensors["atom_types"]
    cell = tensors["cell"]
    pbc = tensors["pbc"]
    positions = tensors["positions"].detach().clone().requires_grad_(True)
    density = model.filtered_density(
        positions,
        atom_types,
        cell=cell,
        pbc=pbc,
    )
    features = _features_from_density(model, density)
    return positions, density, features


def _A_s_density_context_for_structure(model, atoms, type_map, *, dtype, device):
    tensors = _structure_tensors(atoms, type_map, dtype=dtype, device=device, requires_grad=False)
    positions = tensors["positions"].detach().clone().requires_grad_(False)
    density = model.filtered_density(
        positions,
        tensors["atom_types"],
        cell=tensors["cell"],
        pbc=tensors["pbc"],
    ).detach().requires_grad_(True)
    features = _features_from_density(model, density)
    return positions, tensors["atom_types"], tensors["cell"], tensors["pbc"], density, features


def _A_s_feature_position_jacobian_from_batched_vjp(total_features, flat_pos, *, chunk_size=None):
    """Return ``dF/dx`` using the same batched-VJP pattern as linear ACE.

    This mirrors the ordinary ACE normal-equation path in
    ``ye3t_methods.atomistic.ace.linear_ace``: seed multiple feature adjoints at once with
    ``is_grads_batched=True`` and stream chunks of the feature-position
    Jacobian.  It is still an autograd path for A_s; analytic edge/site-basis
    derivatives are intentionally left as the next speed tier.
    """

    n_feat = int(total_features.numel())
    if n_feat == 0:
        return torch.zeros((0, int(flat_pos.numel())), dtype=flat_pos.dtype, device=flat_pos.device)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = n_feat
    chunk_size = max(1, int(chunk_size))
    rows = []
    for start in range(0, n_feat, chunk_size):
        stop = min(n_feat, start + chunk_size)
        grad_outputs = torch.zeros((stop - start, n_feat), dtype=total_features.dtype, device=total_features.device)
        grad_outputs[:, start:stop] = torch.eye(stop - start, dtype=total_features.dtype, device=total_features.device)
        grad = torch.autograd.grad(
            total_features,
            flat_pos,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=stop < n_feat,
            create_graph=False,
            allow_unused=False,
        )[0]
        rows.append(grad.reshape(stop - start, -1))
    return torch.cat(rows, dim=0)


def _A_s_feature_position_jacobian_from_split_density_vjp(
    total_features,
    density,
    positions,
    *,
    chunk_size=None,
):
    """Return ``dF/dx`` by composing ``dF/dA_s`` with ``dA_s/dx``.

    This is an exact chain-rule factorization for the current autograd-defined
    A_s feature map.  It avoids differentiating the full feature graph all the
    way to positions in one operation; instead it first computes feature
    adjoints on the slot-resolved density and then applies the density-position
    VJP.  It is not an analytic density derivative backend.
    """

    n_feat = int(total_features.numel())
    if n_feat == 0:
        return torch.zeros((0, int(positions.numel())), dtype=positions.dtype, device=positions.device)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = n_feat
    chunk_size = max(1, int(chunk_size))
    rows = []
    for start in range(0, n_feat, chunk_size):
        stop = min(n_feat, start + chunk_size)
        grad_outputs = torch.zeros((stop - start, n_feat), dtype=total_features.dtype, device=total_features.device)
        grad_outputs[:, start:stop] = torch.eye(stop - start, dtype=total_features.dtype, device=total_features.device)
        density_adjoint = torch.autograd.grad(
            total_features,
            density,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        grad_pos = torch.autograd.grad(
            density,
            positions,
            grad_outputs=density_adjoint,
            is_grads_batched=True,
            retain_graph=stop < n_feat,
            create_graph=False,
            allow_unused=False,
        )[0]
        rows.append(grad_pos.reshape(stop - start, -1))
    return torch.cat(rows, dim=0)


def _A_s_feature_position_jacobian_from_analytic_density_vjp(
    model,
    total_features,
    density,
    positions,
    atom_types,
    cell,
    pbc,
    *,
    chunk_size=None,
    feature_indices=None,
):
    """Return ``dF/dx`` using autograd ``dF/dA_s`` and analytic ``dA_s/dx`` VJP."""

    full_feature_count = int(total_features.numel())
    if feature_indices is None:
        active_indices = torch.arange(full_feature_count, dtype=torch.long, device=total_features.device)
    else:
        active_indices = torch.as_tensor(feature_indices, dtype=torch.long, device=total_features.device)
        if active_indices.ndim != 1:
            raise ValueError("feature_indices must be one-dimensional.")
        if active_indices.numel() and (torch.any(active_indices < 0) or torch.any(active_indices >= full_feature_count)):
            raise ValueError("feature_indices contain an index outside the feature range.")
        active_indices = torch.unique(active_indices, sorted=True)
    n_feat = int(active_indices.numel())
    if n_feat == 0:
        return torch.zeros((0, int(positions.numel())), dtype=positions.dtype, device=positions.device)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = n_feat
    chunk_size = max(1, int(chunk_size))
    rows = []
    for start in range(0, n_feat, chunk_size):
        stop = min(n_feat, start + chunk_size)
        row_indices = tuple(int(index) for index in active_indices[start:stop].detach().cpu().tolist())
        density_adjoint = torch.zeros(
            (stop - start,) + tuple(density.shape),
            dtype=density.dtype,
            device=density.device,
        )
        analytic_rows = []
        analytic_sector_indices = []
        rank2_rows = []
        rank2_sector_indices = []
        local_rows = []
        local_sector_indices = []
        commutant_rows = []
        commutant_feature_indices = []
        sectors = getattr(model, "ye3_slot_specht_power_sectors", None)
        slot_specht_coupling = getattr(
            model.config.lifted_density,
            "ye3_slot_specht_coupling",
            "projected_norm",
        )
        use_slot_specht_sector_adjoint = (
            model.config.lifted_density.readout_mode == "ye3_slot_specht_power"
            and sectors is not None
            and slot_specht_coupling == "projected_norm"
        )
        use_slot_specht_commutant_adjoint = (
            model.config.lifted_density.readout_mode == "ye3_slot_specht_power"
            and sectors is not None
            and slot_specht_coupling == "commutant_symmetric"
        )
        if use_slot_specht_sector_adjoint:
            for local_row, sector_index in enumerate(row_indices):
                sector = sectors[int(sector_index)]
                if int(sector["l_in"]) == 0:
                    analytic_rows.append(int(local_row))
                    analytic_sector_indices.append(int(sector_index))
                elif int(sector["power"]) == 2:
                    rank2_rows.append(int(local_row))
                    rank2_sector_indices.append(int(sector_index))
                else:
                    local_rows.append(int(local_row))
                    local_sector_indices.append(int(sector_index))
        elif use_slot_specht_commutant_adjoint:
            for local_row, feature_index in enumerate(row_indices):
                commutant_rows.append(int(local_row))
                commutant_feature_indices.append(int(feature_index))
        if analytic_rows:
            analytic = ye3_slot_specht_power_l0_density_adjoint(
                density,
                model.config.lifted_density.channels,
                analytic_sector_indices,
                sectors=sectors,
            )
            density_adjoint.index_copy_(
                0,
                torch.tensor(analytic_rows, dtype=torch.long, device=density.device),
                analytic,
            )
        if rank2_rows:
            rank2 = ye3_slot_specht_power_rank2_density_adjoint(
                density,
                model.config.lifted_density.channels,
                rank2_sector_indices,
                sectors=sectors,
            )
            density_adjoint.index_copy_(
                0,
                torch.tensor(rank2_rows, dtype=torch.long, device=density.device),
                rank2,
            )
        if local_rows:
            local = ye3_slot_specht_power_local_density_adjoint(
                density,
                model.config.lifted_density.channels,
                local_sector_indices,
                sectors=sectors,
                optimization_policy=model.config.lifted_density.ye3_optimization_policy,
            )
            density_adjoint.index_copy_(
                0,
                torch.tensor(local_rows, dtype=torch.long, device=density.device),
                local,
            )
        if commutant_rows:
            commutant = ye3_slot_specht_power_commutant_density_adjoint(
                density,
                model.config.lifted_density.channels,
                commutant_feature_indices,
                sectors=sectors,
                optimization_policy=model.config.lifted_density.ye3_optimization_policy,
            )
            density_adjoint.index_copy_(
                0,
                torch.tensor(commutant_rows, dtype=torch.long, device=density.device),
                commutant,
            )
        handled_rows = set(analytic_rows) | set(rank2_rows) | set(local_rows) | set(commutant_rows)
        autograd_rows = [local for local in range(stop - start) if local not in handled_rows]
        if autograd_rows:
            grad_outputs = torch.zeros(
                (len(autograd_rows), full_feature_count),
                dtype=total_features.dtype,
                device=total_features.device,
            )
            for output_row, local_row in enumerate(autograd_rows):
                grad_outputs[output_row, row_indices[int(local_row)]] = 1.0
            auto_adjoint = torch.autograd.grad(
                total_features,
                density,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=stop < n_feat,
                create_graph=False,
                allow_unused=False,
            )[0]
            density_adjoint.index_copy_(
                0,
                torch.tensor(autograd_rows, dtype=torch.long, device=density.device),
                auto_adjoint,
            )
        grad_pos = model.filtered_density_vjp(
            positions,
            density_adjoint,
            atom_types,
            cell=cell,
            pbc=pbc,
        )
        rows.append(grad_pos.reshape(stop - start, -1))
    return torch.cat(rows, dim=0)


def _force_design_rows_slow_reference(model, atoms, type_map, atom_indices, *, dtype, device):
    """Reference A_s force-design builder using one reverse pass per feature.

    This is intentionally retained only as a correctness oracle for tests and
    debugging.  It has the wrong scaling for high-throughput fitting because it
    evaluates ``grad(feature_k, positions)`` for every scalar feature ``k``.
    """

    features, tensors = _feature_row(model, atoms, type_map, dtype=dtype, device=device, requires_grad=True)
    columns = []
    for feature_index in range(int(features.numel()) - 1):
        grad = torch.autograd.grad(
            features[feature_index],
            tensors["positions"],
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(tensors["positions"])
        columns.append((-grad.index_select(0, atom_indices)).reshape(-1))
    columns.append(features.new_zeros(int(atom_indices.numel()) * 3))
    return torch.stack(columns, dim=1)


def _force_design_rows_batched_jacobian(model, atoms, type_map, atom_indices, *, dtype, device, vectorize=True):
    """Return A_s force-design rows from a batched feature Jacobian.

    Let ``F(x)`` be the scalar feature vector before the atom-count/intercept
    column.  For a linear energy ``E_c(x) = c . F(x) + b N``, the force rows
    needed by ridge fitting are ``-dF_k/dx_{a,alpha}``.  This routine asks
    PyTorch for the Jacobian of the entire feature vector with respect to the
    structure positions in one call, rather than launching one backward pass per
    feature column.

    This is still an automatic-differentiation implementation.  It deliberately
    does not claim the older ACE analytic/streaming derivative path; that is a
    later A_s-specific kernel task.
    """

    tensors = _structure_tensors(atoms, type_map, dtype=dtype, device=device, requires_grad=False)
    atom_types = tensors["atom_types"]
    cell = tensors["cell"]
    pbc = tensors["pbc"]
    base_positions = tensors["positions"].detach().clone().requires_grad_(True)

    def _features_only(positions):
        density = model.filtered_density(
            positions,
            atom_types,
            cell=cell,
            pbc=pbc,
        )
        return _features_from_density(model, density)

    try:
        jacobian = torch.autograd.functional.jacobian(
            _features_only,
            base_positions,
            create_graph=False,
            strict=False,
            vectorize=bool(vectorize),
            strategy="reverse-mode",
        )
        method = "torch.autograd.functional.jacobian_vectorized" if vectorize else "torch.autograd.functional.jacobian"
    except Exception:
        if vectorize:
            jacobian = torch.autograd.functional.jacobian(
                _features_only,
                base_positions,
                create_graph=False,
                strict=False,
                vectorize=False,
                strategy="reverse-mode",
            )
            method = "torch.autograd.functional.jacobian_nonvectorized_fallback"
        else:
            raise
    if jacobian.ndim != 3:
        raise ValueError("A_s feature Jacobian must have shape [n_features, n_atoms, 3].")
    atom_indices = atom_indices.to(device=jacobian.device, dtype=torch.long)
    selected = jacobian.index_select(1, atom_indices)
    rows = -selected.permute(1, 2, 0).reshape(int(atom_indices.numel()) * 3, int(jacobian.shape[0]))
    bias = rows.new_zeros((int(rows.shape[0]), 1))
    out = torch.cat((rows, bias), dim=1)
    out._ye3t_force_design_method = method  # private diagnostic for benchmark metadata
    return out


def _force_design_rows_batched_vjp(model, atoms, type_map, atom_indices, *, dtype, device, chunk_size=None):
    positions, evaluate_from_flat = _A_s_feature_sum_for_structure(
        model,
        atoms,
        type_map,
        dtype=dtype,
        device=device,
    )
    flat_pos = positions.reshape(-1)
    total_features = evaluate_from_flat(flat_pos)
    jacobian = _A_s_feature_position_jacobian_from_batched_vjp(
        total_features,
        flat_pos,
        chunk_size=chunk_size,
    )
    atom_indices = atom_indices.to(device=jacobian.device, dtype=torch.long)
    component_indices = torch.cat([3 * atom_indices + axis for axis in range(3)]).sort().values
    selected = jacobian.index_select(1, component_indices)
    rows = -selected.transpose(0, 1)
    bias = rows.new_zeros((int(rows.shape[0]), 1))
    out = torch.cat((rows, bias), dim=1)
    out._ye3t_force_design_method = "batched_vjp"
    return out


def _force_design_rows(model, atoms, type_map, atom_indices, *, dtype, device):
    return _force_design_rows_batched_vjp(
        model,
        atoms,
        type_map,
        atom_indices,
        dtype=dtype,
        device=device,
        chunk_size=DEFAULT_A_S_FORCE_JACOBIAN_CHUNK_SIZE,
    )


def build_lifted_density_linear_normal_equations(
    model,
    structures,
    *,
    type_map,
    energy_key,
    force_key,
    energy_weight=1.0,
    force_weight=1.0,
    dtype=torch.float64,
    device=None,
    force_atom_stride=None,
    force_jacobian_chunk_size=None,
    force_jacobian_mode="batched_vjp",
    selected_feature_indices=None,
    include_bias_column=True,
    structure_weights=None,
    structure_weight_key=None,
    structure_group_key=None,
    structure_group_weights=None,
    structure_group_default_weight=None,
    structure_group_normalize_mean=True,
    boltzmann_temperature_K=None,
    boltzmann_energy_key=None,
    boltzmann_weight_nugget=0.0,
    boltzmann_weight_prefactor=1.0,
    boltzmann_normalize_mean=True,
    min_structure_weight=0.0,
    return_metadata=False,
):
    """Accumulate A_s linear normal equations without dataset-wide row storage.

    This is the A_s analogue of ``build_linear_ace_normal_equations``.  It
    evaluates the feature vector once per structure, uses a chunked batched-VJP
    Jacobian for force terms, and streams the normal-equation blocks directly.
    The implementation is exact for the current autograd-defined A_s feature
    map, but it is not yet the analytic/streaming site-basis derivative backend.
    """

    structures = list(structures)
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    if not structures:
        raise ValueError("Need at least one structure to build A_s normal equations.")
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
    force_jacobian_mode = str(force_jacobian_mode).strip().lower()
    if force_jacobian_mode not in {"batched_vjp", "split_density_vjp", "analytic_density_vjp"}:
        raise ValueError("force_jacobian_mode must be 'batched_vjp', 'split_density_vjp', or 'analytic_density_vjp'.")
    effective_force_jacobian_mode = force_jacobian_mode
    include_bias_column = bool(include_bias_column)
    feature_count = _linear_lifted_feature_count(model)
    if selected_feature_indices is None:
        active_feature_indices = torch.arange(feature_count, dtype=torch.long, device=device)
    else:
        active_feature_indices = torch.as_tensor(selected_feature_indices, dtype=torch.long, device=device)
        if active_feature_indices.ndim != 1:
            raise ValueError("selected_feature_indices must be one-dimensional.")
        if active_feature_indices.numel() == 0:
            raise ValueError("selected_feature_indices must contain at least one feature.")
        if torch.any(active_feature_indices < 0) or torch.any(active_feature_indices >= feature_count):
            raise ValueError("selected_feature_indices contain an index outside the feature range.")
        active_feature_indices = torch.unique(active_feature_indices, sorted=True)
    if force_jacobian_chunk_size is None:
        force_jacobian_chunk_size = min(DEFAULT_A_S_FORCE_JACOBIAN_CHUNK_SIZE, max(1, int(active_feature_indices.numel())))
    n_cols = feature_count + (1 if include_bias_column else 0)
    XtX = torch.zeros((n_cols, n_cols), dtype=dtype, device=device)
    Xty = torch.zeros((n_cols,), dtype=dtype, device=device)
    yty = torch.zeros((), dtype=dtype, device=device)
    n_rows = 0
    feature_eval_seconds = 0.0
    force_jacobian_seconds = 0.0
    normal_accumulation_seconds = 0.0
    structure_timings = []
    sqrt_energy_weight = float(np.sqrt(max(float(energy_weight), 0.0)))
    sqrt_force_weight = float(np.sqrt(max(float(force_weight), 0.0)))
    for structure_index, atoms in enumerate(structures):
        structure_weight = float(resolved_structure_weights[int(structure_index)])
        sqrt_energy_weight_i = float(np.sqrt(max(float(energy_weight) * structure_weight, 0.0)))
        sqrt_force_weight_i = float(np.sqrt(max(float(force_weight) * structure_weight, 0.0)))
        structure_started = perf_counter()
        feature_started = perf_counter()
        if effective_force_jacobian_mode == "analytic_density_vjp":
            positions, atom_types, cell, pbc, density, total_features = _A_s_density_context_for_structure(
                model,
                atoms,
                type_map,
                dtype=dtype,
                device=device,
            )
            flat_pos = positions.reshape(-1)
        elif effective_force_jacobian_mode == "split_density_vjp":
            positions, density, total_features = _A_s_density_features_for_structure(
                model,
                atoms,
                type_map,
                dtype=dtype,
                device=device,
            )
            flat_pos = positions.reshape(-1)
        else:
            positions, evaluate_from_flat = _A_s_feature_sum_for_structure(
                model,
                atoms,
                type_map,
                dtype=dtype,
                device=device,
            )
            flat_pos = positions.reshape(-1)
            density = None
            total_features = evaluate_from_flat(flat_pos)
        feature_seconds = perf_counter() - feature_started
        feature_eval_seconds += feature_seconds
        if int(total_features.numel()) != feature_count:
            raise ValueError(
                "A_s normal-equation feature count does not match model readout "
                f"({int(total_features.numel())} vs {feature_count})."
            )

        accumulated_rows = 0
        energy_ref = _reference_energy(atoms, energy_key)
        if energy_ref is not None and sqrt_energy_weight_i > 0.0:
            accum_started = perf_counter()
            row = torch.cat(
                (
                    total_features.detach().to(dtype=dtype),
                    torch.as_tensor([float(len(atoms))], dtype=dtype, device=device),
                )
                if include_bias_column
                else (total_features.detach().to(dtype=dtype),)
            ) * sqrt_energy_weight_i
            target = torch.as_tensor(energy_ref * sqrt_energy_weight_i, dtype=dtype, device=device)
            XtX = XtX + torch.outer(row, row)
            Xty = Xty + row * target
            yty = yty + target * target
            n_rows += 1
            accumulated_rows += 1
            normal_accumulation_seconds += perf_counter() - accum_started

        force_ref = _reference_forces(atoms, force_key) if sqrt_force_weight_i > 0.0 else None
        if force_ref is not None and sqrt_force_weight_i > 0.0:
            jac_started = perf_counter()
            if effective_force_jacobian_mode == "analytic_density_vjp":
                jac = _A_s_feature_position_jacobian_from_analytic_density_vjp(
                    model,
                    total_features,
                    density,
                    positions,
                    atom_types,
                    cell,
                    pbc,
                    chunk_size=force_jacobian_chunk_size,
                    feature_indices=active_feature_indices,
                ).to(dtype=dtype)
            elif effective_force_jacobian_mode == "split_density_vjp":
                jac = _A_s_feature_position_jacobian_from_split_density_vjp(
                    total_features.index_select(0, active_feature_indices),
                    density,
                    positions,
                    chunk_size=force_jacobian_chunk_size,
                ).to(dtype=dtype)
            else:
                jac = _A_s_feature_position_jacobian_from_batched_vjp(
                    total_features.index_select(0, active_feature_indices),
                    flat_pos,
                    chunk_size=force_jacobian_chunk_size,
                ).to(dtype=dtype)
            force_jacobian_seconds += perf_counter() - jac_started
            accum_started = perf_counter()
            target = torch.as_tensor(
                np.asarray(force_ref, float).reshape(-1),
                dtype=dtype,
                device=device,
            ) * sqrt_force_weight_i
            weighted_jac = jac * sqrt_force_weight_i
            if force_atom_stride is not None:
                stride = max(1, int(force_atom_stride))
                atom_indices = torch.arange(0, len(atoms), stride, dtype=torch.long, device=device)
                component_indices = torch.cat([3 * atom_indices + axis for axis in range(3)]).sort().values
                weighted_jac = weighted_jac.index_select(1, component_indices)
                target = target.index_select(0, component_indices)
            force_xtx = weighted_jac @ weighted_jac.transpose(0, 1)
            current = XtX.index_select(0, active_feature_indices).index_select(1, active_feature_indices)
            XtX[active_feature_indices[:, None], active_feature_indices[None, :]] = current + force_xtx
            Xty.index_add_(0, active_feature_indices, -(weighted_jac @ target))
            yty = yty + torch.dot(target, target)
            n_rows += int(target.numel())
            accumulated_rows += int(target.numel())
            normal_accumulation_seconds += perf_counter() - accum_started
        structure_timings.append(
            {
                "structure_index": int(structure_index),
                "atom_count": int(len(atoms)),
                "feature_eval_seconds": float(feature_seconds),
                "rows_added": int(accumulated_rows),
                "total_structure_seconds": float(perf_counter() - structure_started),
            }
        )

    metadata = {
        "backend": "A_s_batched_vjp_streaming_normal_equations",
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "feature_count": int(feature_count),
        "include_bias_column": bool(include_bias_column),
        "active_force_feature_count": int(active_feature_indices.numel()),
        "emitted_feature_count": int(feature_count),
        "selected_feature_indices": [int(index) for index in active_feature_indices.detach().cpu().tolist()],
        "force_jacobian_mode": force_jacobian_mode,
        "force_jacobian_effective_mode": effective_force_jacobian_mode,
        "force_jacobian_chunk_size": None if force_jacobian_chunk_size is None else int(force_jacobian_chunk_size),
        "force_atom_stride": None if force_atom_stride is None else int(force_atom_stride),
        "feature_eval_seconds": float(feature_eval_seconds),
        "force_jacobian_seconds": float(force_jacobian_seconds),
        "normal_accumulation_seconds": float(normal_accumulation_seconds),
        "structure_timings": structure_timings,
        "structure_weights": dict(structure_weight_metadata),
        "status": "autograd-defined A_s force normal equations; analytic site-basis derivative backend pending",
    }
    if return_metadata:
        return {"XtX": XtX, "Xty": Xty, "yty": yty, "n_rows": int(n_rows), "n_cols": int(n_cols), "metadata": metadata}
    return XtX, Xty, yty


def select_lifted_density_linear_feature_indices(
    normal,
    *,
    max_features=None,
    policy="normal_diagonal",
):
    """Select active linear feature columns for capacity-controlled comparisons.

    This is a benchmark/model-selection helper, not a representation-theoretic
    projection.  ``policy='normal_diagonal'`` ranks features by the diagonal of
    the weighted normal-equation matrix, i.e. by the training-design column
    norm induced by the same energy/force objective used for the fit.  It does
    not inspect target correlations or fitted coefficients.
    """

    xtx = normal["XtX"] if isinstance(normal, dict) else normal[0]
    metadata = normal.get("metadata", {}) if isinstance(normal, dict) else {}
    if "feature_count" in metadata:
        feature_count = int(metadata["feature_count"])
    else:
        feature_count = int(xtx.shape[0]) - 1
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
        raise ValueError("feature selection policy must be 'normal_diagonal', 'design_norm', or 'energy_normal_diagonal'.")
    scores = torch.diagonal(xtx[:feature_count, :feature_count]).detach()
    order = torch.argsort(scores, descending=True)
    selected = torch.sort(order[:max_features]).values
    return selected.to(dtype=torch.long, device=xtx.device)


def solve_lifted_density_ridge_from_normal_equations(
    normal,
    *,
    ridge_alpha=0.0,
    feature_indices=None,
):
    """Solve a ridge linear readout, optionally on a selected feature subset.

    The returned coefficient vector always has the full model width plus the
    intercept.  Unselected feature coefficients are exactly zero, so the result
    can be assigned to the original descriptor/model without changing the
    descriptor object or public feature inventory.
    """

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
    try:
        coeff_reduced = torch.linalg.solve(xtx_reduced + penalty, xty_reduced)
        solve_method = "torch.linalg.solve"
    except RuntimeError:
        coeff_reduced = torch.linalg.lstsq((xtx_reduced + penalty), xty_reduced.unsqueeze(1)).solution.squeeze(1)
        solve_method = "torch.linalg.lstsq"
    coeff = torch.zeros((n_cols,), dtype=xty.dtype, device=xty.device)
    coeff.index_copy_(0, reduced, coeff_reduced)
    return {
        "coeff": coeff,
        "active_feature_indices": active,
        "active_feature_count": int(active.numel()),
        "emitted_feature_count": int(feature_count),
        "include_bias_column": bool(include_bias_column),
        "solve_method": solve_method,
    }


def fit_lifted_density_linear_model(
    model,
    structures,
    *,
    type_map,
    energy_key="energy",
    force_key="forces",
    energy_weight=1.0,
    force_weight=1.0,
    ridge_alpha=0.0,
    dtype=None,
    device=None,
    force_atom_stride=None,
    force_jacobian_chunk_size=None,
    force_jacobian_mode="batched_vjp",
    feature_budget=None,
    feature_selection_policy="normal_diagonal",
    selected_feature_indices=None,
    include_bias_column=True,
    structure_weights=None,
    structure_weight_key=None,
    structure_group_key=None,
    structure_group_weights=None,
    structure_group_default_weight=None,
    structure_group_normalize_mean=True,
    boltzmann_temperature_K=None,
    boltzmann_energy_key=None,
    boltzmann_weight_nugget=0.0,
    boltzmann_weight_prefactor=1.0,
    boltzmann_normalize_mean=True,
    min_structure_weight=0.0,
    return_metadata=False,
):
    """Fit a scalar linear readout for an existing A_s lifted-density model.

    This is the descriptor-first A_s analogue of the ordinary ACE ridge-normal
    equation path.  It uses the same streaming normal-equation backend as the
    comparison harness: feature rows are never materialized for the whole
    dataset, and force rows are obtained through chunked VJP machinery or the
    available analytic density VJP.  The feature-budget selector is a
    capacity-control/model-selection heuristic, not a representation-theoretic
    projection.
    """

    structures = list(structures)
    if not structures:
        raise ValueError("Need at least one structure to fit a lifted-density linear model.")
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
    if dtype is None:
        dtype = model.config.torch_dtype
    elif isinstance(dtype, str):
        dtype = dtype_from_name(dtype)
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    include_bias_column = bool(include_bias_column)
    model = model.to(device)
    preselection_normal = None
    preselection_status = "not_requested"
    active_indices = None
    if selected_feature_indices is not None:
        active_indices = torch.as_tensor(selected_feature_indices, dtype=torch.long, device=device)
        preselection_status = "explicit_selected_feature_indices"
    elif feature_budget is not None and str(feature_selection_policy).strip().lower() == "energy_normal_diagonal":
        preselection_normal = build_lifted_density_linear_normal_equations(
            model,
            structures,
            type_map=type_map,
            energy_key=energy_key,
            force_key=force_key,
            energy_weight=energy_weight,
            force_weight=0.0,
            dtype=dtype,
            device=device,
            force_atom_stride=force_atom_stride,
            force_jacobian_chunk_size=force_jacobian_chunk_size,
            force_jacobian_mode=force_jacobian_mode,
            include_bias_column=include_bias_column,
            structure_weights=resolved_structure_weights,
            boltzmann_temperature_K=None,
            return_metadata=True,
        )
        active_indices = select_lifted_density_linear_feature_indices(
            preselection_normal,
            max_features=feature_budget,
            policy=feature_selection_policy,
        )
        preselection_status = "energy_only_preselection"
    normal = build_lifted_density_linear_normal_equations(
        model,
        structures,
        type_map=type_map,
        energy_key=energy_key,
        force_key=force_key,
        energy_weight=energy_weight,
        force_weight=force_weight,
        dtype=dtype,
        device=device,
        force_atom_stride=force_atom_stride,
        force_jacobian_chunk_size=force_jacobian_chunk_size,
        force_jacobian_mode=force_jacobian_mode,
        selected_feature_indices=active_indices,
        include_bias_column=include_bias_column,
        structure_weights=resolved_structure_weights,
        boltzmann_temperature_K=None,
        return_metadata=True,
    )
    if active_indices is None:
        active_indices = select_lifted_density_linear_feature_indices(
            normal,
            max_features=feature_budget,
            policy=feature_selection_policy,
        )
        preselection_status = "post_normal_selection" if feature_budget is not None else "all_features"
    solve = solve_lifted_density_ridge_from_normal_equations(
        normal,
        ridge_alpha=float(ridge_alpha),
        feature_indices=active_indices,
    )
    coeff = solve["coeff"]
    _assign_closed_form_readout(model, coeff, include_bias_column=include_bias_column)
    metadata = {
        "backend": "A_s_streaming_normal_equations_linear_fit",
        "fit_method": "ridge_normal_equations",
        "ridge_alpha": float(ridge_alpha),
        "energy_weight": float(energy_weight),
        "force_weight": float(force_weight),
        "energy_key": str(energy_key),
        "force_key": str(force_key),
        "feature_selection": {
            "policy": str(feature_selection_policy),
            "preselection_status": preselection_status,
            "active_feature_count": int(solve["active_feature_count"]),
            "emitted_feature_count": int(solve["emitted_feature_count"]),
            "active_feature_indices": [int(index) for index in solve["active_feature_indices"].detach().cpu().tolist()],
        },
        "include_bias_column": bool(include_bias_column),
        "intercept_policy": "atom_count_bias_column" if include_bias_column else "disabled_reference_energy_fit",
        "structure_weights": dict(structure_weight_metadata),
        "normal_equations": dict(normal["metadata"]),
        "preselection_normal_equations": None
        if preselection_normal is None
        else dict(preselection_normal["metadata"]),
        "solve_method": str(solve["solve_method"]),
        "status": (
            "fitted scalar A_s lifted-density linear readout; force rows use the configured "
            "VJP backend and are not the ordinary ACE analytic derivative backend"
        ),
    }
    model._ye3t_linear_fit_metadata = metadata
    model._ye3t_linear_fit_coefficients = coeff.detach().cpu()
    if return_metadata:
        return {"model": model, "coeff": coeff.detach(), "metadata": metadata}
    return model


def _assign_closed_form_readout(model, coeff, *, include_bias_column=True):
    mode = model.config.lifted_density.readout_mode
    channel_count = len(model.config.lifted_density.channels)
    bias = coeff[-1] if include_bias_column else coeff.new_zeros(())
    weights = coeff[:-1] if include_bias_column else coeff
    with torch.no_grad():
        if mode == "character_quadratic":
            model.character_mu_weight.copy_(weights[:channel_count])
            model.character_nu_weight.copy_(weights[channel_count : 2 * channel_count])
            model.character_bias[0] = bias
        elif mode == "ye3_power":
            model.ye3_power_weight.copy_(weights)
            model.ye3_power_bias[0] = bias
        elif mode == "ye3_slot_specht_power":
            model.ye3_slot_specht_power_weight.copy_(weights)
            model.ye3_slot_specht_power_bias[0] = bias
        elif mode == "symmetric_linear":
            model.channel_readout.copy_(weights[:channel_count])
            model.slot_equivariant_bias[0] = bias
        else:
            raise ValueError(f"Unsupported linear lifted-density readout mode {mode!r}.")


