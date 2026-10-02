"""Optional symmetric-power feature modules for ACE and YE3T-equivariant MP workflows."""

import torch

from ye3t.api import (
    allowed_symmetric_power_outputs,
    symmetric_power_monomial_table,
    symmetric_power_output_multiplicity,
    symmetric_power_outputs_real_tesseral,
    symmetric_power_real_tesseral,
)
from ye3t.runtime import compact_symmetric_pair_product
from ye3t_ace._record import recordclass


@recordclass(('source', 'input_L', 'power', 'output_L', 'name'), frozen = True)
class SymmetricPowerFeatureSpec:
    """One repeated-block symmetric-power feature projection."""

    def as_dict(self):
        return {
            "source": str(self.source),
            "input_L": int(self.input_L),
            "power": int(self.power),
            "output_L": int(self.output_L),
            "name": str(self.name),
        }


@recordclass(('input_L', 'power', 'output_Ls', 'multiplicities', 'term_counts', 'optimization_policy'), frozen = True)
class SymmetricPowerRuntimeSchedule:
    """Device/dtype agnostic runtime schedule for one collapsed repeated block."""

    def as_dict(self):
        return {
            "input_L": int(self.input_L),
            "power": int(self.power),
            "output_Ls": tuple(int(L) for L in self.output_Ls),
            "multiplicities": tuple(int(value) for value in self.multiplicities),
            "term_counts": tuple(int(value) for value in self.term_counts),
            "optimization_policy": str(self.optimization_policy),
            "schedule_kind": "symmetric_power_native_repeated_block",
        }


_RUNTIME_SCHEDULE_CACHE = {}


def build_symmetric_power_runtime_schedule(
    *,
    input_L,
    power,
    output_Ls = None,
    optimization_policy = "auto",
    include_term_counts = True,
):
    """Build a cached collapsed symmetric-power runtime schedule.

    The schedule describes ``Sym^power(V_input_L)`` outputs without expanding
    repeated leaves into generic heterogeneous pair products. It stores only
    static metadata; device-local monomial/count tables are still cached by
    ``ye3t.runtime.symmetric_power`` when the schedule is evaluated.
    """

    input_L = int(input_L)
    power = int(power)
    if output_Ls is None:
        outputs = tuple(int(L) for L in allowed_symmetric_power_outputs(power, input_L))
    else:
        allowed = set(int(L) for L in allowed_symmetric_power_outputs(power, input_L))
        outputs = tuple(int(L) for L in output_Ls)
        invalid = tuple(L for L in outputs if int(L) not in allowed and int(L) != power * input_L)
        if invalid:
            raise ValueError(f"Invalid Sym^{power}(V_{input_L}) output irreps: {invalid}.")
    key = (input_L, power, outputs, str(optimization_policy), bool(include_term_counts))
    cached = _RUNTIME_SCHEDULE_CACHE.get(key)
    if cached is not None:
        return cached
    multiplicities = tuple(
        int(symmetric_power_output_multiplicity(power, input_L, output_L))
        for output_L in outputs
    )
    if include_term_counts:
        term_counts = tuple(
            int(symmetric_power_monomial_table(power, input_L, output_L).term_count)
            for output_L in outputs
        )
    else:
        term_counts = tuple(0 for _ in outputs)
    schedule = SymmetricPowerRuntimeSchedule(
        input_L=input_L,
        power=power,
        output_Ls=outputs,
        multiplicities=multiplicities,
        term_counts=term_counts,
        optimization_policy=str(optimization_policy),
    )
    _RUNTIME_SCHEDULE_CACHE[key] = schedule
    return schedule


def evaluate_symmetric_power_runtime_schedule(x, schedule, *, optimization_policy = None):
    """Evaluate a collapsed repeated-block symmetric-power schedule."""

    policy = str(schedule.optimization_policy if optimization_policy is None else optimization_policy)
    return symmetric_power_outputs_real_tesseral(
        x,
        int(schedule.power),
        int(schedule.input_L),
        schedule.output_Ls,
        optimization_policy=policy,
    )


def compact_symmetric_square(x, backend="auto"):
    """Evaluate the normalized compact coordinate ``Sym^2`` building block."""

    return compact_symmetric_pair_product(x, x, backend=backend)


class SymmetricPowerFeatureProjector(torch.nn.Module):
    """Autograd-compatible repeated-block symmetric-power feature projector."""

    def __init__(self, specs, *, optimization_policy = "auto"):
        super().__init__()
        self.specs = tuple(specs)
        self.optimization_policy = str(optimization_policy)

    def forward(self, features_by_L, *, optimization_policy = None):
        policy = self.optimization_policy if optimization_policy is None else str(optimization_policy)
        outputs = []
        for spec in self.specs:
            x = features_by_L[int(spec.input_L)]
            value = symmetric_power_real_tesseral(
                x,
                int(spec.power),
                int(spec.input_L),
                int(spec.output_L),
                optimization_policy=policy,
            )
            outputs.append(value.reshape(value.shape[0], -1))
        if not outputs:
            first = next(iter(features_by_L.values()))
            return torch.zeros((first.shape[0], 0), dtype=first.dtype, device=first.device)
        return torch.cat(outputs, dim=-1)

    def feature_dim(self):
        dim = 0
        for spec in self.specs:
            dim += 2 * int(spec.output_L) + 1
        return int(dim)

    def as_dict(self):
        return {
            "optimization_policy": str(self.optimization_policy),
            "feature_dim": self.feature_dim(),
            "specs": [spec.as_dict() for spec in self.specs],
        }


def symmetric_power_specs_from_summary(summary, *, output_selector = "max", max_groups = 1):
    """Build feature specs from a repeated-block kernel summary."""
    specs = []
    for index, group in enumerate(summary.groups[: int(max_groups)]):
        outputs = tuple(int(value) for value in group.outputs)
        if not outputs:
            continue
        if output_selector == "all":
            chosen_outputs = outputs
        elif output_selector == "min":
            chosen_outputs = (min(outputs),)
        else:
            chosen_outputs = (max(outputs),)
        for output_L in chosen_outputs:
            specs.append(
                SymmetricPowerFeatureSpec(
                    source=str(group.source),
                    input_L=int(group.input_L),
                    power=int(group.power),
                    output_L=int(output_L),
                    name=f"{group.source}:sym{group.power}_L{group.input_L}_to_L{output_L}_{index}",
                )
            )
    return tuple(specs)


def build_symmetric_power_feature_projector(
    summary,
    *,
    output_selector = "max",
    max_groups = 1,
    optimization_policy = "auto",
):
    """Build an optional symmetric-power feature projector from a schedule summary."""
    return SymmetricPowerFeatureProjector(
        symmetric_power_specs_from_summary(
            summary,
            output_selector=output_selector,
            max_groups=max_groups,
        ),
        optimization_policy=optimization_policy,
    )


__all__ = [
    "SymmetricPowerFeatureSpec",
    "SymmetricPowerRuntimeSchedule",
    "SymmetricPowerFeatureProjector",
    "build_symmetric_power_runtime_schedule",
    "compact_symmetric_square",
    "evaluate_symmetric_power_runtime_schedule",
    "symmetric_power_specs_from_summary",
    "build_symmetric_power_feature_projector",
]
