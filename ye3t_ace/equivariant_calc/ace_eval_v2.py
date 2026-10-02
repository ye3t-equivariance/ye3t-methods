
from dataclasses import field
from functools import lru_cache
import itertools
from pathlib import Path
import json
import math
import os

import torch

from ye3t.couplings import symmetric_power_product_plan
from ye3t.paired_cg import normalize_packed_cg_backend
from ye3t.runtime.symmetric_power import symmetric_power_product_plan_contraction
from ye3t.core.couplings import (
    evaluate_factorized_schedule_torch,
    generate_factorized_coefficient_schedule_for_labels,
    generate_real_factorized_coefficient_schedule_for_labels,
)
from ye3t.core.tesseral import real_tesseral_to_complex_multiplet
from .ace_symmetric_power import (
    ace_complex_symmetric_power_entries,
    ace_real_symmetric_power_metadata_for_block_specs,
    is_native_real_l1_even_scalar_symmetric_power,
    native_real_l1_even_scalar_symmetric_power,
)
from .labeling import CompactLabel, DescriptorSpec, SingleChannelLabel, normalize_compact_label
from .site_basis_v2 import SiteBasisConfig, SiteBasisV2
from ye3t_ace._record import recordclass


def scalar_real_projection_tolerance(values, imag_tol):
    """Return a device-aware imaginary tolerance for scalar real projections."""

    base = torch.as_tensor(float(imag_tol), dtype=values.real.dtype, device=values.device)
    scale = torch.max(torch.abs(values.real)) if values.numel() else base
    scale = torch.maximum(scale, torch.ones((), dtype=values.real.dtype, device=values.device))
    roundoff = 64.0 * torch.finfo(values.real.dtype).eps * scale
    base = torch.maximum(base, roundoff)
    if values.is_cuda:
        floor = 1.0e-4 if values.real.dtype == torch.float32 else 1.0e-5
        rel = 1.0e-4 if values.real.dtype == torch.float32 else 1.0e-5
        cuda_tol = torch.maximum(
            torch.as_tensor(floor, dtype=values.real.dtype, device=values.device),
            torch.as_tensor(rel, dtype=values.real.dtype, device=values.device) * scale,
        )
        base = torch.maximum(base, cuda_tol)
    return base


def checked_real_scalar_projection(values, *, imag_tol, context):
    """Project scalar invariants to real values after a device-aware sanity check."""

    if not torch.is_complex(values):
        return values
    max_imag = (
        torch.max(torch.abs(values.imag))
        if values.numel()
        else torch.zeros((), dtype=values.real.dtype, device=values.device)
    )
    effective_tol = scalar_real_projection_tolerance(values, imag_tol)
    if max_imag > effective_tol:
        max_real = (
            torch.max(torch.abs(values.real))
            if values.numel()
            else torch.zeros((), dtype=values.real.dtype, device=values.device)
        )
        relative_imag = max_imag / torch.maximum(
            max_real,
            torch.ones((), dtype=values.real.dtype, device=values.device),
        )
        raise RuntimeError(
            f"Expected real-valued scalar invariants for {context}, found max imaginary part {max_imag.item():.3e} "
            f"(tolerance {effective_tol.item():.3e}, max real magnitude {max_real.item():.3e}, "
            f"relative imaginary residual {relative_imag.item():.3e})"
        )
    return values.real


@recordclass(('label', 'rank', 'M_R', 'L_R', 'ms_combinations', 'coeffs'))
class CouplingPayload:
    pass


@recordclass(('channels', 'channel_rows_cpu', 'coeffs_cpu', 'grouped_descriptor_indices_cpu', 'grouped_channel_rows_cpu', 'grouped_coeffs_cpu', 'all_scalar', 'factorized_plan', 'direct_symmetric_power_plan', '_device_cache', '_grouped_device_cache'))
class CompiledDescriptorBatch:
    factorized_plan = None
    direct_symmetric_power_plan = None
    _device_cache = field(default_factory=dict, init=False, repr=False)
    _grouped_device_cache = field(default_factory=dict, init=False, repr=False)

    def materialize(self, *, device, dtype):
        key = (str(device), str(dtype))
        cached = self._device_cache.get(key)
        if cached is None:
            cached = (
                tuple(rows.to(device=device) for rows in self.channel_rows_cpu),
                tuple(coeffs.to(device=device, dtype=dtype) for coeffs in self.coeffs_cpu),
            )
            self._device_cache[key] = cached
        return cached

    def materialize_grouped(
        self,
        *,
        device,
        dtype,
    ):
        key = (str(device), str(dtype))
        cached = self._grouped_device_cache.get(key)
        if cached is None:
            cached = (
                tuple(indices.to(device=device) for indices in self.grouped_descriptor_indices_cpu),
                tuple(rows.to(device=device) for rows in self.grouped_channel_rows_cpu),
                tuple(coeffs.to(device=device, dtype=dtype) for coeffs in self.grouped_coeffs_cpu),
            )
            self._grouped_device_cache[key] = cached
        return cached


@recordclass(('descriptor_indices', 'channel_index_rows', 'coeffs', 'product_term_values'), frozen = True)
class AtomicProductGroup:

    @property
    def rank(self):
        return int(self.channel_index_rows.shape[-1])


@recordclass(('atomic_base', 'descriptor_keys', 'groups'), frozen = True)
class AtomicProductCollection:
    pass


@recordclass(('descriptor_indices', 'schedule', 'block_specs', 'block_channel_indices', 'real_schedule'), frozen = True)
class FactorizedDescriptorGroup:
    real_schedule = None
    pass


@recordclass(('groups', 'descriptor_count', 'all_scalar'), frozen = True)
class FactorizedDescriptorPlan:

    @property
    def group_count(self):
        return int(len(self.groups))

    @property
    def active_descriptor_indices(self):
        out = []
        for group in self.groups:
            out.extend(int(idx) for idx in group.descriptor_indices)
        return tuple(sorted(set(out)))

    @property
    def active_descriptor_count(self):
        return int(len(self.active_descriptor_indices))

    @property
    def residual_descriptor_count(self):
        return int(self.descriptor_count) - int(self.active_descriptor_count)


@recordclass(('descriptor_index', 'block_spec', 'channel_indices', 'component_index'), frozen = True)
class DirectSymmetricPowerDescriptorEntry:
    pass


@recordclass(('entries', 'descriptor_count', 'all_scalar'), frozen = True)
class DirectSymmetricPowerDescriptorPlan:

    @property
    def active_descriptor_indices(self):
        return tuple(sorted(int(entry.descriptor_index) for entry in self.entries))

    @property
    def active_descriptor_count(self):
        return int(len(self.entries))


class _ProjectedFactorizedDescriptorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, evaluator, plan, atomic_base, source_index, projection):
        ctx.evaluator = evaluator
        ctx.plan = plan
        ctx.save_for_backward(atomic_base, source_index, projection)
        with torch.no_grad():
            return evaluator._contract_factorized_descriptor_plan_projected_sources_impl(
                plan,
                atomic_base,
                source_index=source_index,
                projection=projection,
            )

    @staticmethod
    def backward(ctx, grad_output):
        atomic_base, source_index, projection = ctx.saved_tensors
        tensors = []
        atomic_req = atomic_base.detach().requires_grad_(ctx.needs_input_grad[2])
        projection_req = projection.detach().requires_grad_(ctx.needs_input_grad[4])
        if ctx.needs_input_grad[2]:
            tensors.append(atomic_req)
        if ctx.needs_input_grad[4]:
            tensors.append(projection_req)
        if not tensors:
            return None, None, None, None, None
        with torch.enable_grad():
            out = ctx.evaluator._contract_factorized_descriptor_plan_projected_sources_impl(
                ctx.plan,
                atomic_req,
                source_index=source_index,
                projection=projection_req,
            )
            grads = torch.autograd.grad(
                out,
                tuple(tensors),
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        grad_iter = iter(grads)
        grad_atomic = next(grad_iter) if ctx.needs_input_grad[2] else None
        grad_projection = next(grad_iter) if ctx.needs_input_grad[4] else None
        return None, None, grad_atomic, None, grad_projection


_CHANNEL_INDEX_TENSOR_CACHE = {}


def _ace_factorized_runtime_policy():
    return str(
        os.environ.get(
            "YE3T_ACE_FACTORIZED_DESCRIPTOR_RUNTIME",
            os.environ.get("gne3_ACE_FACTORIZED_DESCRIPTOR_RUNTIME", "auto"),
        )
    ).strip().lower()


def _ace_env_int(name, default):
    raw = os.environ.get("YE3T_ACE_" + str(name), os.environ.get("gne3_ACE_" + str(name), None))
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _ace_env_flag(name, default=False):
    raw = os.environ.get("YE3T_ACE_" + str(name), os.environ.get("gne3_ACE_" + str(name), None))
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "force"}


def _policy_forces_factorized(policy):
    return str(policy) in {"1", "true", "yes", "on", "force", "require", "required", "strict"}


def _policy_disables_factorized(policy):
    return str(policy) in {"0", "false", "no", "off", "disable", "disabled"}


def _direct_symmetric_power_disabled():
    return _ace_env_flag("DISABLE_DIRECT_SYMMETRIC_POWER", False)


def _generic_angular_direct_symmetric_power_enabled():
    return _ace_env_flag("ENABLE_GENERIC_ANGULAR_DIRECT_SYMMETRIC_POWER", False)


def _factorized_descriptor_disabled():
    return _ace_env_flag("DISABLE_FACTORIZED_DESCRIPTOR_RUNTIME", False)


def _channel_m_zero(channel):
    return SingleChannelLabel(
        mu0=channel.mu0,
        mu=channel.mu,
        kappa0=channel.kappa0,
        kappa=channel.kappa,
        n=channel.n,
        l=channel.l,
        m=0,
        l_aux=channel.l_aux,
        m_aux=channel.m_aux,
        eta=channel.eta,
    )


def _channel_with_m(channel, m):
    return SingleChannelLabel(
        mu0=channel.mu0,
        mu=channel.mu,
        kappa0=channel.kappa0,
        kappa=channel.kappa,
        n=channel.n,
        l=channel.l,
        m=int(m),
        l_aux=channel.l_aux,
        m_aux=channel.m_aux,
        eta=channel.eta,
    )


def _descriptor_channels_match_symmetric_blocks(desc, block_specs):
    pos = 0
    out = []
    for spec in block_specs:
        count = int(spec["k_b"])
        if pos + count > len(desc.channels):
            return None
        block_channels = tuple(desc.channels[pos:pos + count])
        pos += count
        if str(spec["kind"]) == "sym":
            proto = _channel_m_zero(block_channels[0])
            if any(_channel_m_zero(ch) != proto for ch in block_channels):
                return None
            if int(proto.n) != int(spec["n"]) or int(proto.l) != int(spec["l"]):
                return None
            out.append(proto)
        else:
            proto = _channel_m_zero(block_channels[0])
            if int(proto.n) != int(spec["n"]) or int(proto.l) != int(spec["l"]):
                return None
            out.append(proto)
    if pos != len(desc.channels):
        return None
    return tuple(out)


def _ensure_multiplet_channels(proto, index_of, channels):
    indices = []
    for m in range(-int(proto.l), int(proto.l) + 1):
        channel = _channel_with_m(proto, m)
        idx = index_of.get(channel)
        if idx is None:
            idx = len(channels)
            channels.append(channel)
            index_of[channel] = idx
        indices.append(int(idx))
    return tuple(indices)


def _label_orbit_pattern(label):
    classes = {}
    next_class = 0
    pattern = []
    for n_value, l_value in zip(label.n_tuple, label.l_tuple):
        key = (int(n_value), int(l_value))
        class_index = classes.get(key)
        if class_index is None:
            class_index = next_class
            classes[key] = class_index
            next_class += 1
        pattern.append((int(class_index), int(l_value)))
    return tuple(pattern)


@lru_cache(maxsize=None)
def _complex_symmetric_power_entries(power, input_L, output_L, multiplicity_index):
    return ace_complex_symmetric_power_entries(
        int(power),
        int(input_L),
        int(output_L),
        int(multiplicity_index),
    )


@lru_cache(maxsize=None)
def _complex_symmetric_power_product_plan(
    power,
    input_L,
    output_L,
    multiplicity_index,
):
    """Compile one repeated complex multiplet without ordered slot expansion."""

    power = int(power)
    input_L = int(input_L)
    output_L = int(output_L)
    multiplicity_index = int(multiplicity_index)
    input_dimension = 2 * input_L + 1
    output_dimension = 2 * output_L + 1
    entries = tuple(
        {
            "descriptor_index": component_index,
            "power": power,
            "input_L": input_L,
            "output_L": output_L,
            "multiplicity_index": multiplicity_index,
            "component_index": component_index,
            "channel_indices": tuple(range(input_dimension)),
        }
        for component_index in range(output_dimension)
    )
    return symmetric_power_product_plan(
        entries,
        descriptor_count=output_dimension,
        channel_count=input_dimension,
        carrier="ACE_density",
        target={
            "permutation": "trivial",
            "rotation": {
                "L_R": output_L,
                "M_R_values": tuple(range(-output_L, output_L + 1)),
            },
        },
        factor_basis="A",
        normalization_convention="ace_independent_homogeneous",
        basis_convention="complex_magnetic",
        coefficient_materialization="exact",
        maximum_exact_symbolic_bytes=128 * 1024 * 1024,
        coefficient_source="ye3t.couplings symmetric-power occupancy compiler",
        label_source="ye3t.couplings.count",
        validation_report={
            "scope": "ACE repeated complex multiplet occupancy contraction",
            "ordered_slot_expansion": False,
        },
        provenance={
            "consumer": "ye3t_ace.equivariant_calc.ACECovariantEvaluator",
        },
    )


def _evaluate_complex_symmetric_power_block(x, *, power, input_L, output_L, multiplicity_index):
    plan = _complex_symmetric_power_product_plan(
        int(power),
        int(input_L),
        int(output_L),
        int(multiplicity_index),
    )
    return symmetric_power_product_plan_contraction(x, plan, backend="auto")


def _complex_m_to_signed_real_basis(m):
    """Expansion coefficients for ``Y_l^m`` in the signed real tesseral basis."""

    m = int(m)
    if m == 0:
        return ((0, 1.0 + 0.0j),)
    rt2_inv = 1.0 / math.sqrt(2.0)
    if m > 0:
        sign = -1.0 if (m % 2) else 1.0
        return (
            (m, sign * rt2_inv),
            (-m, -1j * sign * rt2_inv),
        )
    p = -m
    return (
        (p, rt2_inv),
        (-p, 1j * rt2_inv),
    )


def _descriptor_terms_for_spherical_backend(
    desc,
    *,
    spherical_backend,
    coefficient_tol = 1.0e-14,
):
    if spherical_backend == "complex":
        return desc.ms_combinations, desc.coeffs
    if spherical_backend != "real":
        raise ValueError("spherical_backend must be one of complex or real")

    accum = {}
    for ms_row, coeff in zip(desc.ms_combinations, desc.coeffs):
        expansions = [_complex_m_to_signed_real_basis(int(m)) for m in ms_row]
        for expanded in itertools.product(*expansions):
            alpha_row = tuple(int(alpha) for alpha, _ in expanded)
            basis_coeff = 1.0 + 0.0j
            for _, transform_coeff in expanded:
                basis_coeff *= transform_coeff
            accum[alpha_row] = accum.get(alpha_row, 0.0 + 0.0j) + complex(coeff) * basis_coeff

    rows_and_coeffs = sorted(
        ((row, coeff) for row, coeff in accum.items() if abs(coeff) > coefficient_tol),
        key=lambda item: item[0],
    )
    if not rows_and_coeffs:
        return tuple(), tuple()
    rows, coeffs = zip(*rows_and_coeffs)
    return tuple(rows), tuple(coeffs)


class GeneralizedCouplingLibrary:
    """
    Flexible loader for the generalized coupling pickles/JSONs.

    Expected minimal per-label payload:
      {
        "rank": int,
        "ms_combs": flat or nested list,
        "coeffs": list,
      }

    The outer structure may be keyed as library[M_R][rank][label_key] = payload.
    """

    def __init__(self, data, L_R, metadata=None):
        self.data = data
        self.L_R = L_R
        self.metadata = dict(metadata or {})

    @classmethod
    def from_json(cls, path, L_R):
        with open(path, "r") as fh:
            raw = json.load(fh)
        data = {int(M): {int(r): v for r, v in rank_map.items()} for M, rank_map in raw.items()}
        return cls(data, L_R=L_R)

    @classmethod
    def from_pickle(cls, path, L_R):
        import pickle
        with open(path, "rb") as fh:
            raw = pickle.load(fh)
        data = {int(M): {int(r): v for r, v in rank_map.items()} for M, rank_map in raw.items()}
        return cls(data, L_R=L_R)

    def descriptor_specs(
        self,
        compact_labels,
        channels_by_descriptor,
        M_R,
    ):
        out = []
        mr_block = self.data[int(M_R)]
        for label in compact_labels:
            lab = normalize_compact_label(label)
            rank_block = mr_block[lab.rank]
            key = (
                lab.full_key()
                if lab.full_key() in rank_block
                else lab.angular_key()
            )
            if key not in rank_block:
                raise KeyError(f"Could not find coupling payload for key={key}, rank={lab.rank}, M_R={M_R}")
            payload = rank_block[key]
            ms_raw = payload["ms_combs"]
            coeffs_raw = payload["coeffs"]
            rank = int(payload.get("rank", lab.rank))
            if isinstance(ms_raw[0], (int, float)):
                if len(ms_raw) % rank != 0:
                    raise ValueError(f"ms_combs length {len(ms_raw)} is not divisible by rank {rank}")
                ms_combinations = tuple(
                    tuple(int(ms_raw[i * rank + j]) for j in range(rank))
                    for i in range(len(ms_raw) // rank)
                )
            else:
                ms_combinations = tuple(tuple(int(x) for x in row) for row in ms_raw)
            def _to_complex(x):
                if isinstance(x, complex):
                    return x
                if isinstance(x, (list, tuple)) and len(x) == 2:
                    return complex(x[0], x[1])
                return complex(x)
            phase = -1j if int(self.L_R) == 0 and int(M_R) == 0 and (sum(int(l) for l in lab.l_tuple) % 2) else 1.0 + 0.0j
            coeffs = tuple(phase * _to_complex(c) for c in coeffs_raw)
            out.append(
                DescriptorSpec(
                    key=lab.full_key(),
                    label=lab,
                    channels=tuple(channels_by_descriptor[lab.full_key()]),
                    ms_combinations=ms_combinations,
                    coeffs=coeffs,
                    L_R=self.L_R,
                    M_R=M_R,
                )
            )
        return out


class ACECovariantEvaluator(torch.nn.Module):
    """
    Evaluates site covariants from a set of compiled descriptor specs.

    Usage pattern:
      1. build/obtain CompactLabel objects from the new label generator
      2. map each label to its single-channel leaf specification(s)
      3. load generalized coupling payloads for the requested M_R values
      4. call forward(...) to get per-atom covariants
    """

    def __init__(
        self,
        basis_config,
        *,
        backend = "pytorch",
        strict_backend = False,
        validate_backend = True,
        factorized_descriptor_runtime_policy = None,
    ):
        super().__init__()
        self.site_basis = SiteBasisV2(basis_config)
        self.backend = normalize_packed_cg_backend(backend)
        self.strict_backend = bool(strict_backend)
        self.validate_backend = bool(validate_backend)
        self.factorized_descriptor_runtime_policy = (
            None
            if factorized_descriptor_runtime_policy is None
            else str(factorized_descriptor_runtime_policy).strip().lower()
        )
        self._compiled_descriptor_cache = {}
        self._last_backend_counts = {}
        self._last_compile_cache_status = None

    def backend_report(self):
        factorized_groups = 0
        factorized_descriptors = 0
        direct_sym_power_descriptors = 0
        for compiled in self._compiled_descriptor_cache.values():
            plan = getattr(compiled, "factorized_plan", None)
            if plan is not None:
                factorized_groups += int(plan.group_count)
                factorized_descriptors += int(plan.active_descriptor_count)
            direct_plan = getattr(compiled, "direct_symmetric_power_plan", None)
            if direct_plan is not None:
                direct_sym_power_descriptors += int(direct_plan.active_descriptor_count)
        fallback_descriptors = 0
        for compiled in self._compiled_descriptor_cache.values():
            plan = getattr(compiled, "factorized_plan", None)
            direct_plan = getattr(compiled, "direct_symmetric_power_plan", None)
            direct_count = 0 if direct_plan is None else int(direct_plan.active_descriptor_count)
            if plan is None:
                fallback_descriptors += max(0, int(len(compiled.channel_rows_cpu)) - direct_count)
            else:
                fallback_descriptors += max(0, int(plan.residual_descriptor_count) - direct_count)
        return {
            "backend": str(self.backend),
            "strict_backend": bool(self.strict_backend),
            "validate_backend": bool(self.validate_backend),
            "counts": dict(self._last_backend_counts),
            "backend_path": (
                "partial_factorized_block_schedule"
                if factorized_groups and fallback_descriptors
                else (
                    "factorized_block_schedule"
                    if factorized_groups
                    else (
                        "direct_symmetric_power_block"
                        if direct_sym_power_descriptors and not fallback_descriptors
                        else "atomic_product_grouped"
                    )
                )
            ),
            "factorized_descriptor_runtime": "enabled" if factorized_groups else "unavailable_or_not_compiled",
            "factorized_descriptor_group_count": int(factorized_groups),
            "factorized_descriptor_count": int(factorized_descriptors),
            "direct_symmetric_power_descriptor_count": int(direct_sym_power_descriptors),
            "fallback_descriptor_count": int(fallback_descriptors),
            "compiled_descriptor_cache_entries": int(len(self._compiled_descriptor_cache)),
            "last_compile_cache_status": self._last_compile_cache_status,
            "site_basis_profile": self.site_basis.last_profile(),
        }

    def _factorized_plan_failed(self, message):
        policy = self._factorized_runtime_policy()
        if policy in {"require", "required", "strict"}:
            raise RuntimeError(message)
        return None

    def _factorized_runtime_policy(self):
        if self.factorized_descriptor_runtime_policy is not None:
            return self.factorized_descriptor_runtime_policy
        return _ace_factorized_runtime_policy()

    def _auto_accepts_factorized_group(self, *, policy, schedule, group_descriptors, block_specs):
        if _policy_forces_factorized(policy):
            return True
        if str(policy) != "auto":
            return False
        if not any(str(spec["kind"]) == "sym" for specs in block_specs for spec in specs):
            return False
        group_size = len(group_descriptors)
        expanded_terms = sum(len(desc.coeffs) for desc in group_descriptors)
        min_group_size = _ace_env_int("FACTORIZED_DESCRIPTOR_MIN_GROUP_SIZE", 4)
        min_expanded_terms = _ace_env_int("FACTORIZED_DESCRIPTOR_MIN_EXPANDED_TERMS", 128)
        if group_size < min_group_size and expanded_terms < min_expanded_terms:
            return False
        return int(expanded_terms) > int(schedule.term_count)

    def _build_factorized_descriptor_plan(
        self,
        descriptors,
        *,
        index_of,
        channels,
    ):
        policy = self._factorized_runtime_policy()
        if _policy_disables_factorized(policy):
            return None
        if not descriptors:
            return None
        if self.site_basis.cfg.spherical_backend not in {"real", "complex"}:
            return self._factorized_plan_failed("Factorized ACE descriptor runtime requires real or complex spherical backend.")

        grouped = {}
        for descriptor_index, desc in enumerate(descriptors):
            label = normalize_compact_label(desc.label)
            if int(desc.L_R) != int(label.L_R):
                return self._factorized_plan_failed("Descriptor L_R does not match its compact label.")
            key = (
                int(desc.M_R),
                int(desc.L_R),
                int(label.rank),
                tuple(int(x) for x in label.l_tuple),
                _label_orbit_pattern(label),
                str(label.tree_type),
            )
            grouped.setdefault(key, []).append((int(descriptor_index), desc, label))

        groups = []
        for key, items in grouped.items():
            M_R = int(key[0])
            descriptor_indices = tuple(int(item[0]) for item in items)
            group_descriptors = tuple(item[1] for item in items)
            labels = tuple(item[2] for item in items)
            try:
                rich_schedule = generate_factorized_coefficient_schedule_for_labels(
                    labels,
                    M_R_values=(M_R,),
                    coeff_dtype=complex,
                )
                schedule = rich_schedule.to_torch(dtype=torch.complex128)
                real_rich_schedule = None
                if self.site_basis.cfg.spherical_backend == "real" and int(key[1]) == 0 and int(M_R) == 0:
                    real_rich_schedule = generate_real_factorized_coefficient_schedule_for_labels(
                        labels,
                        component_indices=(0,),
                    )
            except Exception as exc:
                failure = self._factorized_plan_failed(f"Could not build factorized descriptor schedule: {exc}")
                if failure is not None:
                    return failure
                continue
            if int(schedule.component_count) != len(group_descriptors):
                failure = self._factorized_plan_failed(
                    "Factorized descriptor schedule did not produce one component per descriptor."
                )
                if failure is not None:
                    return failure
                continue
            if int(schedule.basis_count) != len(group_descriptors):
                failure = self._factorized_plan_failed(
                    "Factorized descriptor schedule did not produce one basis row per descriptor."
                )
                if failure is not None:
                    return failure
                continue
            if len(rich_schedule.block_specs) != len(group_descriptors):
                failure = self._factorized_plan_failed("Factorized descriptor block metadata length mismatch.")
                if failure is not None:
                    return failure
                continue
            compatible_positions = []
            for position, (desc, block_specs) in enumerate(zip(group_descriptors, rich_schedule.block_specs)):
                block_protos = _descriptor_channels_match_symmetric_blocks(desc, block_specs)
                if block_protos is None:
                    failure = self._factorized_plan_failed(
                        "Descriptor channels are not compatible with the collapsed symmetric-block ACE runtime."
                    )
                    if failure is not None:
                        return failure
                    continue
                compatible_positions.append(int(position))
            if not compatible_positions:
                continue
            if len(compatible_positions) != len(group_descriptors):
                descriptor_indices = tuple(descriptor_indices[pos] for pos in compatible_positions)
                group_descriptors = tuple(group_descriptors[pos] for pos in compatible_positions)
                labels = tuple(labels[pos] for pos in compatible_positions)
                try:
                    rich_schedule = generate_factorized_coefficient_schedule_for_labels(
                        labels,
                        M_R_values=(M_R,),
                        coeff_dtype=complex,
                    )
                    schedule = rich_schedule.to_torch(dtype=torch.complex128)
                    real_rich_schedule = None
                    if self.site_basis.cfg.spherical_backend == "real" and int(key[1]) == 0 and int(M_R) == 0:
                        real_rich_schedule = generate_real_factorized_coefficient_schedule_for_labels(
                            labels,
                            component_indices=(0,),
                        )
                except Exception as exc:
                    failure = self._factorized_plan_failed(f"Could not build partial factorized descriptor schedule: {exc}")
                    if failure is not None:
                        return failure
                    continue
                if int(schedule.component_count) != len(group_descriptors) or int(schedule.basis_count) != len(group_descriptors):
                    failure = self._factorized_plan_failed(
                        "Partial factorized descriptor schedule did not produce one row per compatible descriptor."
                    )
                    if failure is not None:
                        return failure
                    continue
            block_channel_indices = []
            compatible = True
            for desc, block_specs in zip(group_descriptors, rich_schedule.block_specs):
                block_protos = _descriptor_channels_match_symmetric_blocks(desc, block_specs)
                if block_protos is None:
                    compatible = False
                    break
                block_channel_indices.append(
                    tuple(_ensure_multiplet_channels(proto, index_of, channels) for proto in block_protos)
                )
            if not compatible:
                continue
            if not self._auto_accepts_factorized_group(
                policy=policy,
                schedule=schedule,
                group_descriptors=group_descriptors,
                block_specs=rich_schedule.block_specs,
            ):
                continue
            groups.append(
                FactorizedDescriptorGroup(
                    descriptor_indices=descriptor_indices,
                    schedule=schedule,
                    block_specs=tuple(tuple(specs) for specs in rich_schedule.block_specs),
                    block_channel_indices=tuple(block_channel_indices),
                    real_schedule=real_rich_schedule,
                )
            )
        if not groups:
            return None
        return FactorizedDescriptorPlan(
            groups=tuple(groups),
            descriptor_count=len(descriptors),
            all_scalar=all((desc.L_R == 0 and desc.M_R == 0) for desc in descriptors),
        )

    def _build_direct_symmetric_power_descriptor_plan(
        self,
        descriptors,
        *,
        index_of,
        channels,
    ):
        entries = []
        for descriptor_index, desc in enumerate(descriptors):
            label = normalize_compact_label(desc.label)
            basis_key = tuple(label.basis_key)
            if len(basis_key) != 3 or str(basis_key[0]) != "sym":
                continue
            if int(label.rank) < 2:
                continue
            output_L = int(basis_key[1])
            multiplicity_index = int(basis_key[2])
            if int(desc.L_R) != output_L:
                continue
            if len(set((int(n), int(l)) for n, l in zip(label.n_tuple, label.l_tuple))) != 1:
                continue
            block_specs = (
                {
                    "kind": "sym",
                    "k_b": int(label.rank),
                    "n": int(label.n_tuple[0]),
                    "l": int(label.l_tuple[0]),
                    "Lambda": output_L,
                    "multiplicity_index": multiplicity_index,
                },
            )
            if not is_native_real_l1_even_scalar_symmetric_power(
                int(label.rank),
                int(label.l_tuple[0]),
                output_L,
                multiplicity_index,
            ) and not _generic_angular_direct_symmetric_power_enabled():
                continue
            block_protos = _descriptor_channels_match_symmetric_blocks(desc, block_specs)
            if block_protos is None:
                continue
            entries.append(
                DirectSymmetricPowerDescriptorEntry(
                    descriptor_index=int(descriptor_index),
                    block_spec=block_specs[0],
                    channel_indices=_ensure_multiplet_channels(block_protos[0], index_of, channels),
                    component_index=int(desc.M_R) + output_L,
                )
            )
        if not entries:
            return None
        return DirectSymmetricPowerDescriptorPlan(
            entries=tuple(entries),
            descriptor_count=len(descriptors),
            all_scalar=all((desc.L_R == 0 and desc.M_R == 0) for desc in descriptors),
        )

    def _compile_descriptors(
        self,
        descriptors,
    ):
        policy = self._factorized_runtime_policy()
        direct_disabled = _direct_symmetric_power_disabled()
        factorized_disabled = _factorized_descriptor_disabled()
        descriptor_identity = tuple(
            (
                str(desc.key),
                str(normalize_compact_label(desc.label).full_key()),
                tuple(
                    (
                        int(channel.mu0),
                        int(channel.mu),
                        int(channel.kappa0),
                        int(channel.kappa),
                        int(channel.n),
                        int(channel.l),
                        int(channel.m),
                        None if channel.l_aux is None else int(channel.l_aux),
                        None if channel.m_aux is None else int(channel.m_aux),
                        None if channel.eta is None else int(channel.eta),
                    )
                    for channel in desc.channels
                ),
                int(desc.L_R),
                int(desc.M_R),
                tuple(tuple(int(value) for value in row) for row in desc.ms_combinations),
                tuple(
                    (float(complex(value).real), float(complex(value).imag))
                    for value in desc.coeffs
                ),
            )
            for desc in descriptors
        )
        key = (
            f"spherical_backend={self.site_basis.cfg.spherical_backend}",
            f"factorized_policy={policy}",
            f"direct_symmetric_power_disabled={direct_disabled}",
            f"factorized_descriptor_disabled={factorized_disabled}",
        ) + descriptor_identity
        cached = self._compiled_descriptor_cache.get(key)
        if cached is not None:
            self._last_compile_cache_status = "hit"
            return cached

        expanded = []
        index_of = {}
        channel_rows_cpu = []
        coeffs_cpu = []
        spherical_backend = self.site_basis.cfg.spherical_backend
        direct_symmetric_power_plan = None
        if (
            not _policy_forces_factorized(policy)
            and not direct_disabled
        ):
            direct_symmetric_power_plan = self._build_direct_symmetric_power_descriptor_plan(
                tuple(descriptors),
                index_of=index_of,
                channels=expanded,
            )
        factorized_plan = None
        if (
            not _policy_disables_factorized(policy)
            and not factorized_disabled
            and (
                direct_symmetric_power_plan is None
                or int(direct_symmetric_power_plan.active_descriptor_count) != len(tuple(descriptors))
            )
        ):
            factorized_plan = self._build_factorized_descriptor_plan(
                tuple(descriptors),
                index_of=index_of,
                channels=expanded,
            )
        direct_indices = set() if direct_symmetric_power_plan is None else set(direct_symmetric_power_plan.active_descriptor_indices)
        factorized_indices = set() if factorized_plan is None else set(factorized_plan.active_descriptor_indices)
        accelerated_indices = direct_indices | factorized_indices
        for desc_index, desc in enumerate(descriptors):
            if int(desc_index) in accelerated_indices:
                channel_rows_cpu.append(torch.empty((0, desc.rank), dtype=torch.long))
                coeffs_cpu.append(torch.empty((0,), dtype=torch.complex128))
                continue
            term_rows = []
            ms_combinations, coeffs = _descriptor_terms_for_spherical_backend(
                desc,
                spherical_backend=spherical_backend,
            )
            if len(ms_combinations) != len(coeffs):
                raise RuntimeError(
                    "Descriptor coupling rows and coefficients have different lengths "
                    f"for {desc.key!r}."
                )
            if not ms_combinations:
                raise RuntimeError(
                    "Descriptor has neither an active compiler-generated route nor "
                    f"nonempty raw coupling rows: {desc.key!r}."
                )
            for ms_row in ms_combinations:
                row = []
                for proto, m in zip(desc.channels, ms_row):
                    channel = SingleChannelLabel(
                        mu0=proto.mu0,
                        mu=proto.mu,
                        kappa0=proto.kappa0,
                        kappa=proto.kappa,
                        n=proto.n,
                        l=proto.l,
                        m=int(m),
                        l_aux=proto.l_aux,
                        m_aux=proto.m_aux,
                        eta=proto.eta,
                    )
                    idx = index_of.get(channel)
                    if idx is None:
                        idx = len(expanded)
                        expanded.append(channel)
                        index_of[channel] = idx
                    row.append(int(idx))
                term_rows.append(row)
            if term_rows:
                channel_rows_cpu.append(torch.tensor(term_rows, dtype=torch.long))
            else:
                channel_rows_cpu.append(torch.empty((0, desc.rank), dtype=torch.long))
            coeffs_cpu.append(torch.tensor(coeffs, dtype=torch.complex128))

        grouped_descriptor_indices_cpu = []
        grouped_channel_rows_cpu = []
        grouped_coeffs_cpu = []
        grouped_indices = {}
        for d_idx, rows in enumerate(channel_rows_cpu):
            grouped_indices.setdefault((int(rows.shape[0]), int(rows.shape[1])), []).append(d_idx)
        for _, descriptor_indices in sorted(grouped_indices.items()):
            grouped_descriptor_indices_cpu.append(torch.tensor(descriptor_indices, dtype=torch.long))
            grouped_channel_rows_cpu.append(torch.stack([channel_rows_cpu[i] for i in descriptor_indices], dim=0))
            grouped_coeffs_cpu.append(torch.stack([coeffs_cpu[i] for i in descriptor_indices], dim=0))

        compiled = CompiledDescriptorBatch(
            channels=tuple(expanded),
            channel_rows_cpu=tuple(channel_rows_cpu),
            coeffs_cpu=tuple(coeffs_cpu),
            grouped_descriptor_indices_cpu=tuple(grouped_descriptor_indices_cpu),
            grouped_channel_rows_cpu=tuple(grouped_channel_rows_cpu),
            grouped_coeffs_cpu=tuple(grouped_coeffs_cpu),
            all_scalar=all((desc.L_R == 0 and desc.M_R == 0) for desc in descriptors),
            factorized_plan=factorized_plan,
            direct_symmetric_power_plan=direct_symmetric_power_plan,
        )
        self._compiled_descriptor_cache[key] = compiled
        self._last_compile_cache_status = "miss"
        return compiled

    def precompile_descriptors(self, descriptors):
        """Prepare descriptor contraction tables before model evaluation."""
        compiled = self._compile_descriptors(tuple(descriptors))
        real_schedules = [] if compiled.factorized_plan is None else [
            group.real_schedule for group in compiled.factorized_plan.groups if group.real_schedule is not None
        ]
        sym_power_metadata = []
        if compiled.direct_symmetric_power_plan is not None:
            sym_power_metadata.extend(
                ace_real_symmetric_power_metadata_for_block_specs(
                    tuple((entry.block_spec,) for entry in compiled.direct_symmetric_power_plan.entries)
                )
            )
        if compiled.factorized_plan is not None:
            for group in compiled.factorized_plan.groups:
                sym_power_metadata.extend(
                    ace_real_symmetric_power_metadata_for_block_specs(group.block_specs)
                )
        sym_power_backends = sorted(
            set(str(row.get("sym_power_backend", "none")) for row in sym_power_metadata)
        )
        sym_power_fallback_reasons = [
            str(row.get("sym_power_fallback_reason"))
            for row in sym_power_metadata
            if row.get("sym_power_fallback_reason") is not None
        ]
        return {
            "compiled_descriptor_cache_status": self._last_compile_cache_status,
            "compiled_descriptor_cache_entries": int(len(self._compiled_descriptor_cache)),
            "descriptor_count": int(len(tuple(descriptors))),
            "channel_count": int(len(compiled.channels)),
            "factorized_descriptor_group_count": 0 if compiled.factorized_plan is None else int(compiled.factorized_plan.group_count),
            "factorized_descriptor_count": 0 if compiled.factorized_plan is None else int(compiled.factorized_plan.active_descriptor_count),
            "direct_symmetric_power_descriptor_count": (
                0
                if compiled.direct_symmetric_power_plan is None
                else int(compiled.direct_symmetric_power_plan.active_descriptor_count)
            ),
            "fallback_descriptor_count": (
                max(
                    0,
                    (
                        int(len(tuple(descriptors)))
                        if compiled.factorized_plan is None
                        else int(compiled.factorized_plan.residual_descriptor_count)
                    )
                    - (
                        0
                        if compiled.direct_symmetric_power_plan is None
                        else int(compiled.direct_symmetric_power_plan.active_descriptor_count)
                    ),
                )
            ),
            "factorized_descriptor_indices": (
                []
                if compiled.factorized_plan is None
                else [int(idx) for idx in compiled.factorized_plan.active_descriptor_indices]
            ),
            "native_real_factorized_group_count": int(len(real_schedules)),
            "native_real_factorized_terms": [int(schedule.term_count) for schedule in real_schedules],
            "native_real_factorized_complex_terms": [int(schedule.complex_term_count) for schedule in real_schedules],
            "native_real_factorized_metadata": [schedule.metadata() for schedule in real_schedules],
            "native_real_sym_power_backend": sym_power_backends[0] if len(sym_power_backends) == 1 else tuple(sym_power_backends),
            "native_real_sym_power_metadata": tuple(sym_power_metadata),
            "native_real_sym_power_fallback_reasons": tuple(sym_power_fallback_reasons),
            "runtime_path": (
                "partial_factorized_block_schedule_and_precompiled_descriptor_tables"
                if compiled.factorized_plan is not None and int(compiled.factorized_plan.residual_descriptor_count) > 0
                else (
                    "factorized_block_schedule_and_precompiled_descriptor_tables"
                    if compiled.factorized_plan is not None
                    else (
                        "direct_symmetric_power_block_and_precompiled_descriptor_tables"
                        if compiled.direct_symmetric_power_plan is not None
                        else "precompiled_descriptor_contraction_tables"
                    )
                )
            ),
        }

    def forward(
        self,
        x_ij,
        edge_index,
        atom_types,
        descriptors,
        charges = None,
        aux_tensor_basis = None,
        real_if_scalar = True,
        imag_tol = 1e-12,
        runtime_cache = None,
    ):
        """
        Returns
        -------
        covariants
            Tensor of shape ``[n_atoms, n_descriptors]`` for a fixed ``M_R`` block.
            For general covariants the dtype is complex. If ``real_if_scalar`` is
            true and all requested descriptors satisfy ``L_R=M_R=0``, the return
            value is projected to a real tensor after checking that the imaginary
            part is numerically negligible.
        """
        if len(descriptors) == 0:
            return torch.zeros((atom_types.shape[0], 0), dtype=self.site_basis.cfg.complex_dtype, device=x_ij.device)

        self._last_backend_counts = {}
        compiled = self._compile_descriptors(descriptors)
        _, A = self._compute_atomic_base(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            compiled=compiled,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            runtime_cache=runtime_cache,
        )
        return self._contract_atomic_products(
            compiled,
            A,
            descriptor_count=len(descriptors),
            real_if_scalar=real_if_scalar,
            imag_tol=imag_tol,
        )

    def _compute_atomic_base(
        self,
        *,
        x_ij,
        edge_index,
        atom_types,
        compiled,
        charges = None,
        aux_tensor_basis = None,
        runtime_cache = None,
    ):
        channels, atomic_base = self.site_basis.compute_channels(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=compiled.channels,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            runtime_cache=runtime_cache,
        )
        return tuple(channels), atomic_base

    def atomic_products(
        self,
        x_ij,
        edge_index,
        atom_types,
        descriptors,
        charges = None,
        aux_tensor_basis = None,
    ):
        if len(descriptors) == 0:
            atomic_base = torch.zeros((atom_types.shape[0], 0), dtype=self.site_basis.cfg.complex_dtype, device=x_ij.device)
            return AtomicProductCollection(atomic_base=atomic_base, descriptor_keys=tuple(), groups=tuple())
        compiled = self._compile_descriptors(descriptors)
        _, atomic_base = self._compute_atomic_base(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            compiled=compiled,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
        )
        groups = self._collect_atomic_products(compiled, atomic_base)
        return AtomicProductCollection(
            atomic_base=atomic_base,
            descriptor_keys=tuple(str(desc.key) for desc in descriptors),
            groups=groups,
        )

    def contract_atomic_products(
        self,
        products,
        descriptors,
        *,
        real_if_scalar = True,
        imag_tol = 1e-12,
    ):
        compiled = self._compile_descriptors(descriptors)
        return self._contract_grouped_products(
            products.groups,
            descriptor_count=len(descriptors),
            all_scalar=compiled.all_scalar,
            dtype=products.atomic_base.dtype,
            device=products.atomic_base.device,
            real_if_scalar=real_if_scalar,
            imag_tol=imag_tol,
        )

    def _collect_atomic_products(
        self,
        compiled,
        atomic_base,
    ):
        descriptor_indices, grouped_channel_rows, grouped_coeff_rows = compiled.materialize_grouped(
            device=atomic_base.device,
            dtype=atomic_base.dtype,
        )
        n_atoms = atomic_base.shape[0]
        groups = []
        for out_indices, channel_index_rows, coeffs in zip(descriptor_indices, grouped_channel_rows, grouped_coeff_rows):
            gathered = atomic_base.index_select(1, channel_index_rows.reshape(-1))
            gathered = gathered.reshape(
                n_atoms,
                channel_index_rows.shape[0],
                channel_index_rows.shape[1],
                channel_index_rows.shape[2],
            )
            product_term_values = gathered.prod(dim=-1)
            groups.append(
                AtomicProductGroup(
                    descriptor_indices=out_indices,
                    channel_index_rows=channel_index_rows,
                    coeffs=coeffs,
                    product_term_values=product_term_values,
                )
            )
        return tuple(groups)

    def _contract_grouped_products(
        self,
        groups,
        *,
        descriptor_count,
        all_scalar,
        dtype,
        device,
        real_if_scalar,
        imag_tol,
    ):
        if descriptor_count == 0:
            return torch.zeros((0, 0), dtype=dtype, device=device)
        n_atoms = 0 if not groups else int(groups[0].product_term_values.shape[0])
        out = torch.zeros((n_atoms, descriptor_count), dtype=dtype, device=device)
        if self.backend in {"openequivariance", "triton"} and self.strict_backend:
            raise RuntimeError(
                f"{self.backend} was requested strictly, but ACECovariantEvaluator's grouped atomic-product "
                "contraction is currently evaluated by the PyTorch grouped fallback."
            )
        if self.backend == "openequivariance":
            backend_key = "atomic_product_backend:pytorch_after_openequivariance_fallback"
        elif self.backend == "triton":
            backend_key = "atomic_product_backend:pytorch_after_triton_fallback"
        else:
            backend_key = f"atomic_product_backend:{self.backend if self.backend != 'auto' else 'pytorch'}"
        self._last_backend_counts[backend_key] = self._last_backend_counts.get(backend_key, 0) + int(descriptor_count)
        for group in groups:
            coeffs = group.coeffs
            grouped_out = (group.product_term_values * coeffs.unsqueeze(0)).sum(dim=-1)
            out.index_copy_(1, group.descriptor_indices, grouped_out)
        if real_if_scalar and all_scalar:
            return checked_real_scalar_projection(out, imag_tol=imag_tol, context="L_R=0, M_R=0")
        return out

    def _contract_atomic_products(
        self,
        compiled,
        atomic_base,
        *,
        descriptor_count,
        real_if_scalar,
        imag_tol,
    ):
        n_atoms = atomic_base.shape[0]
        out = torch.zeros((n_atoms, descriptor_count), dtype=atomic_base.dtype, device=atomic_base.device)
        if compiled.factorized_plan is not None:
            out = out + self._contract_factorized_descriptor_plan(
                compiled.factorized_plan,
                atomic_base,
                real_if_scalar=False,
                imag_tol=imag_tol,
            )
        if compiled.direct_symmetric_power_plan is not None:
            out = out + self._contract_direct_symmetric_power_descriptor_plan(
                compiled.direct_symmetric_power_plan,
                atomic_base,
                real_if_scalar=False,
                imag_tol=imag_tol,
            )
        skip_indices = set()
        if compiled.factorized_plan is not None:
            skip_indices.update(compiled.factorized_plan.active_descriptor_indices)
        if compiled.direct_symmetric_power_plan is not None:
            skip_indices.update(compiled.direct_symmetric_power_plan.active_descriptor_indices)
        if compiled.factorized_plan is not None or compiled.direct_symmetric_power_plan is not None:
            self._add_grouped_atomic_product_residual(
                out,
                compiled,
                atomic_base,
                skip_descriptor_indices=skip_indices,
            )
            if real_if_scalar and compiled.all_scalar:
                return checked_real_scalar_projection(out, imag_tol=imag_tol, context="mixed factorized/fallback L_R=0, M_R=0")
            return out
        self._add_grouped_atomic_product_residual(
            out,
            compiled,
            atomic_base,
            skip_descriptor_indices=set(),
        )
        if real_if_scalar and compiled.all_scalar:
            return checked_real_scalar_projection(out, imag_tol=imag_tol, context="L_R=0, M_R=0")
        return out

    def _add_grouped_atomic_product_residual(
        self,
        out,
        compiled,
        atomic_base,
        *,
        skip_descriptor_indices,
    ):
        descriptor_indices, grouped_channel_rows, grouped_coeff_rows = compiled.materialize_grouped(
            device=atomic_base.device,
            dtype=atomic_base.dtype,
        )
        n_atoms = atomic_base.shape[0]
        if self.backend in {"openequivariance", "triton"} and self.strict_backend:
            raise RuntimeError(
                f"{self.backend} was requested strictly, but ACECovariantEvaluator's grouped atomic-product "
                "contraction is currently evaluated by the PyTorch grouped fallback."
            )
        if self.backend == "openequivariance":
            backend_key = "atomic_product_backend:pytorch_after_openequivariance_fallback"
        elif self.backend == "triton":
            backend_key = "atomic_product_backend:pytorch_after_triton_fallback"
        else:
            backend_key = f"atomic_product_backend:{self.backend if self.backend != 'auto' else 'pytorch'}"
        for out_indices, channel_index_rows, coeffs in zip(descriptor_indices, grouped_channel_rows, grouped_coeff_rows):
            if skip_descriptor_indices:
                keep_positions = [
                    int(pos)
                    for pos, descriptor_index in enumerate(out_indices.detach().cpu().tolist())
                    if int(descriptor_index) not in skip_descriptor_indices
                ]
                if not keep_positions:
                    continue
                keep_tensor = torch.tensor(keep_positions, dtype=torch.long, device=atomic_base.device)
                out_indices = out_indices.index_select(0, keep_tensor)
                channel_index_rows = channel_index_rows.index_select(0, keep_tensor)
                coeffs = coeffs.index_select(0, keep_tensor)
            self._last_backend_counts[backend_key] = self._last_backend_counts.get(backend_key, 0) + int(out_indices.numel())
            gathered = atomic_base.index_select(1, channel_index_rows.reshape(-1))
            gathered = gathered.reshape(
                n_atoms,
                channel_index_rows.shape[0],
                channel_index_rows.shape[1],
                channel_index_rows.shape[2],
            )
            product_term_values = gathered.prod(dim=-1)
            grouped_out = (product_term_values * coeffs.unsqueeze(0)).sum(dim=-1)
            out.index_copy_(1, out_indices, grouped_out)
        return out

    def _channel_multiplet_from_indices(
        self,
        atomic_base,
        indices,
        *,
        L,
    ):
        index_key = (tuple(int(idx) for idx in indices), str(atomic_base.device))
        index_tensor = _CHANNEL_INDEX_TENSOR_CACHE.get(index_key)
        if index_tensor is None:
            index_tensor = torch.tensor(index_key[0], dtype=torch.long, device=atomic_base.device)
            _CHANNEL_INDEX_TENSOR_CACHE[index_key] = index_tensor
        values = atomic_base.index_select(1, index_tensor)
        if self.site_basis.cfg.spherical_backend == "real":
            return real_tesseral_to_complex_multiplet(values.real, int(L))
        return values

    def _real_channel_values_from_indices(
        self,
        atomic_base,
        indices,
    ):
        index_key = (tuple(int(idx) for idx in indices), str(atomic_base.device), "real")
        index_tensor = _CHANNEL_INDEX_TENSOR_CACHE.get(index_key)
        if index_tensor is None:
            index_tensor = torch.tensor(index_key[0], dtype=torch.long, device=atomic_base.device)
            _CHANNEL_INDEX_TENSOR_CACHE[index_key] = index_tensor
        return atomic_base.index_select(1, index_tensor).real

    def _block_value_for_factorized_descriptor(
        self,
        atomic_base,
        *,
        spec,
        indices,
    ):
        input_L = int(spec["l"])
        output_L = int(spec["Lambda"])
        if str(spec["kind"]) == "leaf":
            x = self._channel_multiplet_from_indices(atomic_base, indices, L=input_L)
            return x
        if (
            self.site_basis.cfg.spherical_backend == "real"
            and is_native_real_l1_even_scalar_symmetric_power(
                int(spec["k_b"]),
                input_L,
                output_L,
                int(spec["multiplicity_index"]),
            )
        ):
            real_values = self._real_channel_values_from_indices(atomic_base, indices)
            return native_real_l1_even_scalar_symmetric_power(
                real_values,
                power=int(spec["k_b"]),
            )
        x = self._channel_multiplet_from_indices(atomic_base, indices, L=input_L)
        return _evaluate_complex_symmetric_power_block(
            x,
            power=int(spec["k_b"]),
            input_L=input_L,
            output_L=output_L,
            multiplicity_index=int(spec["multiplicity_index"]),
        )

    def _contract_direct_symmetric_power_descriptor_plan(
        self,
        plan,
        atomic_base,
        *,
        real_if_scalar,
        imag_tol,
    ):
        n_atoms = int(atomic_base.shape[0])
        out = torch.zeros(
            (n_atoms, int(plan.descriptor_count)),
            dtype=atomic_base.dtype,
            device=atomic_base.device,
        )
        backend_key = "atomic_product_backend:direct_symmetric_power_block"
        self._last_backend_counts[backend_key] = self._last_backend_counts.get(backend_key, 0) + int(plan.active_descriptor_count)
        for entry in plan.entries:
            value = self._block_value_for_factorized_descriptor(
                atomic_base,
                spec=entry.block_spec,
                indices=entry.channel_indices,
            )
            component = int(entry.component_index)
            if component < 0 or component >= int(value.shape[-1]):
                raise RuntimeError("Direct symmetric-power component index is outside the evaluated multiplet width.")
            out[:, int(entry.descriptor_index)] = value[:, component]
        if real_if_scalar and plan.all_scalar:
            return checked_real_scalar_projection(out, imag_tol=imag_tol, context="direct symmetric-power L_R=0, M_R=0")
        return out

    def _contract_factorized_descriptor_plan(
        self,
        plan,
        atomic_base,
        *,
        real_if_scalar,
        imag_tol,
    ):
        n_atoms = int(atomic_base.shape[0])
        out = torch.zeros(
            (n_atoms, int(plan.descriptor_count)),
            dtype=atomic_base.dtype,
            device=atomic_base.device,
        )
        runtime_backend = "torch" if atomic_base.requires_grad else "auto"
        backend_key = (
            "atomic_product_backend:factorized_block_schedule_autograd_torch"
            if runtime_backend == "torch"
            else "atomic_product_backend:factorized_block_schedule_auto"
        )
        self._last_backend_counts[backend_key] = self._last_backend_counts.get(backend_key, 0) + int(plan.active_descriptor_count)
        for group in plan.groups:
            schedule = group.schedule.to(device=atomic_base.device, dtype=atomic_base.dtype)
            block_values = torch.zeros(
                (
                    n_atoms,
                    int(schedule.basis_count),
                    int(schedule.block_count),
                    int(schedule.max_block_m_dim),
                ),
                dtype=atomic_base.dtype,
                device=atomic_base.device,
            )
            for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
                for block_index, (spec, indices) in enumerate(zip(block_specs, block_indices)):
                    value = self._block_value_for_factorized_descriptor(
                        atomic_base,
                        spec=spec,
                        indices=indices,
                    )
                    width = int(value.shape[-1])
                    block_values[:, int(label_index), int(block_index), :width] = value
            grouped_out = evaluate_factorized_schedule_torch(
                block_values,
                schedule,
                backend=runtime_backend,
            )
            descriptor_indices = torch.tensor(group.descriptor_indices, dtype=torch.long, device=atomic_base.device)
            out.index_copy_(1, descriptor_indices, grouped_out)
        if real_if_scalar and plan.all_scalar:
            return checked_real_scalar_projection(out, imag_tol=imag_tol, context="factorized L_R=0, M_R=0")
        return out

    def _contract_atomic_products_projected_sources(
        self,
        compiled,
        atomic_base,
        *,
        source_index,
        projection,
        descriptor_count,
    ):
        if descriptor_count == 0:
            return torch.zeros(
                (int(source_index.shape[0]), int(projection.shape[1])),
                dtype=atomic_base.dtype,
                device=atomic_base.device,
            )
        if compiled.factorized_plan is None:
            full = self._contract_atomic_products(
                compiled,
                atomic_base,
                descriptor_count=descriptor_count,
                real_if_scalar=False,
                imag_tol=0.0,
            )
            selected = full.index_select(0, source_index.to(device=full.device, dtype=torch.long))
            projection = projection.to(device=selected.device)
            value_dtype = torch.promote_types(selected.dtype, projection.dtype)
            self._last_backend_counts["atomic_product_backend:projected_materialized_fallback"] = (
                self._last_backend_counts.get("atomic_product_backend:projected_materialized_fallback", 0)
                + int(descriptor_count)
            )
            return torch.einsum(
                "ek,epk->ep",
                selected.to(dtype=value_dtype),
                projection.to(dtype=value_dtype),
            )
        return self._contract_factorized_descriptor_plan_projected_sources(
            compiled.factorized_plan,
            atomic_base,
            source_index=source_index,
            projection=projection,
        ) + self._contract_grouped_atomic_products_projected_residual(
            compiled,
            atomic_base,
            source_index=source_index,
            projection=projection,
            descriptor_count=descriptor_count,
            skip_descriptor_indices=set(compiled.factorized_plan.active_descriptor_indices),
        )

    def _contract_grouped_atomic_products_projected_residual(
        self,
        compiled,
        atomic_base,
        *,
        source_index,
        projection,
        descriptor_count,
        skip_descriptor_indices,
    ):
        if len(skip_descriptor_indices) >= int(descriptor_count):
            return torch.zeros(
                (int(source_index.shape[0]), int(projection.shape[1])),
                dtype=self.site_basis.cfg.complex_dtype,
                device=atomic_base.device,
            )
        residual = torch.zeros(
            (int(atomic_base.shape[0]), int(descriptor_count)),
            dtype=atomic_base.dtype,
            device=atomic_base.device,
        )
        self._add_grouped_atomic_product_residual(
            residual,
            compiled,
            atomic_base,
            skip_descriptor_indices=skip_descriptor_indices,
        )
        selected = residual.index_select(0, source_index.to(device=residual.device, dtype=torch.long))
        projection = projection.to(device=selected.device)
        value_dtype = torch.promote_types(selected.dtype, projection.dtype)
        self._last_backend_counts["atomic_product_backend:projected_partial_materialized_fallback"] = (
            self._last_backend_counts.get("atomic_product_backend:projected_partial_materialized_fallback", 0)
            + int(descriptor_count - len(skip_descriptor_indices))
        )
        return torch.einsum(
            "ek,epk->ep",
            selected.to(dtype=value_dtype),
            projection.to(dtype=value_dtype),
        ).to(dtype=self.site_basis.cfg.complex_dtype)

    def _contract_factorized_descriptor_plan_projected_sources(
        self,
        plan,
        atomic_base,
        *,
        source_index,
        projection,
    ):
        if (
            _ace_env_flag("PROJECTED_EXACT_CUSTOM_BACKWARD", False)
            and (atomic_base.requires_grad or projection.requires_grad)
        ):
            self._last_backend_counts["atomic_product_backend:projected_factorized_custom_backward"] = (
                self._last_backend_counts.get("atomic_product_backend:projected_factorized_custom_backward", 0)
                + int(plan.active_descriptor_count)
            )
            return _ProjectedFactorizedDescriptorFunction.apply(
                self,
                plan,
                atomic_base,
                source_index.to(device=atomic_base.device, dtype=torch.long),
                projection,
            )
        return self._contract_factorized_descriptor_plan_projected_sources_impl(
            plan,
            atomic_base,
            source_index=source_index,
            projection=projection,
        )

    def _contract_factorized_descriptor_plan_projected_sources_impl(
        self,
        plan,
        atomic_base,
        *,
        source_index,
        projection,
    ):
        edge_count = int(source_index.shape[0])
        projected_width = int(projection.shape[1])
        out_dtype = self.site_basis.cfg.complex_dtype
        out = torch.zeros(
            (edge_count, projected_width),
            dtype=out_dtype,
            device=atomic_base.device,
        )
        self._last_backend_counts["atomic_product_backend:projected_factorized_block_schedule_sources"] = (
            self._last_backend_counts.get("atomic_product_backend:projected_factorized_block_schedule_sources", 0)
            + int(plan.active_descriptor_count)
        )
        source_index = source_index.to(device=atomic_base.device, dtype=torch.long)
        batch_index = torch.arange(edge_count, device=atomic_base.device).unsqueeze(1)
        projection = projection.to(device=atomic_base.device, dtype=out_dtype)
        for group in plan.groups:
            schedule = group.schedule.to(device=atomic_base.device, dtype=atomic_base.dtype)
            coeffs = schedule.coeffs.to(device=atomic_base.device)
            if torch.is_complex(coeffs) and not torch.is_complex(atomic_base):
                if int(coeffs.numel()) and torch.max(torch.abs(coeffs.imag)).item() <= 1e-12:
                    coeffs = coeffs.real.to(dtype=atomic_base.dtype)
            value_dtype = torch.promote_types(atomic_base.dtype, coeffs.dtype)
            value_dtype = torch.promote_types(value_dtype, projection.dtype)
            coeffs = coeffs.to(dtype=value_dtype)
            term_values = coeffs.unsqueeze(0).expand(edge_count, -1)
            for block_index in range(int(schedule.block_count)):
                block_values = torch.zeros(
                    (
                        edge_count,
                        int(schedule.basis_count),
                        int(schedule.max_block_m_dim),
                    ),
                    dtype=value_dtype,
                    device=atomic_base.device,
                )
                for label_index, (block_specs, block_indices) in enumerate(zip(group.block_specs, group.block_channel_indices)):
                    spec = block_specs[int(block_index)]
                    indices = block_indices[int(block_index)]
                    value = self._block_value_for_factorized_descriptor(
                        atomic_base,
                        spec=spec,
                        indices=indices,
                    ).index_select(0, source_index)
                    width = int(value.shape[-1])
                    block_values[:, int(label_index), :width] = value.to(dtype=value_dtype)
                term_values = term_values * block_values[
                    batch_index,
                    schedule.term_label_index.unsqueeze(0),
                    schedule.block_m_indices[:, int(block_index)].unsqueeze(0),
                ]
            descriptor_indices = torch.tensor(group.descriptor_indices, dtype=torch.long, device=atomic_base.device)
            group_projection = projection.index_select(2, descriptor_indices).to(dtype=value_dtype)
            component_index = schedule.term_component_index.to(device=atomic_base.device, dtype=torch.long)
            gathered_projection = group_projection.index_select(2, component_index)
            out = out + torch.einsum("et,ept->ep", term_values, gathered_projection).to(dtype=out_dtype)
        return out

    def forward_projected_sources(
        self,
        x_ij,
        edge_index,
        atom_types,
        descriptors,
        *,
        source_index,
        projection,
        charges = None,
        aux_tensor_basis = None,
        runtime_cache = None,
    ):
        """Evaluate source-indexed linear combinations of descriptor rows.

        ``projection`` has shape ``[n_sources, projected_width, n_descriptors]``.
        When a factorized symmetric-block schedule is available this contracts
        directly into those projected rows, avoiding full descriptor-row
        materialization in the exact message path.
        """
        if len(descriptors) == 0:
            return torch.zeros(
                (int(source_index.shape[0]), int(projection.shape[1])),
                dtype=self.site_basis.cfg.complex_dtype,
                device=x_ij.device,
            )
        self._last_backend_counts = {}
        compiled = self._compile_descriptors(descriptors)
        _, atomic_base = self._compute_atomic_base(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            compiled=compiled,
            charges=charges,
            aux_tensor_basis=aux_tensor_basis,
            runtime_cache=runtime_cache,
        )
        return self._contract_atomic_products_projected_sources(
            compiled,
            atomic_base,
            source_index=source_index,
            projection=projection,
            descriptor_count=len(descriptors),
        )


def compile_ace_covariant_evaluator(
    evaluator,
    *,
    fullgraph = False,
    dynamic = False,
):
    """Best-effort ``torch.compile`` wrapper for exact ACE descriptor evaluation."""
    try:
        return torch.compile(evaluator, fullgraph=bool(fullgraph), dynamic=bool(dynamic))
    except Exception:
        return evaluator


def compact_label_to_channels_with_mus_kappas(
    label,
    mus,
    mu0 = 0,
    kappa0s = None,
    kappas = None,
):
    """
    Convenience helper for the common scalar/charge cases.

    Given a compact label and parallel per-leaf chemical/scalar indices, produce
    the ``SingleChannelLabel`` list used by the evaluator.
    """
    rank = label.rank
    if len(mus) != rank:
        raise ValueError("mus must have length equal to the descriptor rank")
    if kappa0s is None:
        kappa0s = [0] * rank
    if kappas is None:
        kappas = [0] * rank
    if len(kappa0s) != rank or len(kappas) != rank:
        raise ValueError("kappa0s and kappas must have length equal to the descriptor rank")

    channels = []
    for mu, kappa0, kappa, n, l in zip(mus, kappa0s, kappas, label.n_tuple, label.l_tuple):
        # ``m`` is filled later by the coupling payload.
        channels.append(SingleChannelLabel(mu0=mu0, mu=mu, kappa0=kappa0, kappa=kappa, n=n, l=l, m=0))
    return tuple(channels)


# Compatibility alias for k/kappa parameter spelling.
def compact_label_to_channels_with_mus_ks(
    label,
    mus,
    mu0 = 0,
    k0s = None,
    ks = None,
):
    return compact_label_to_channels_with_mus_kappas(
        label=label,
        mus=mus,
        mu0=mu0,
        kappa0s=k0s,
        kappas=ks,
    )
