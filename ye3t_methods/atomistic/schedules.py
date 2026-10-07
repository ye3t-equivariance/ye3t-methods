"""Targeted schedule summaries for ACE and YE3T-equivariant MP workflows."""

from collections import Counter
from collections import defaultdict

from ye3t_methods.atomistic._record import recordclass
from ye3t_methods.atomistic.equivariant_calc.descriptor_sets import enumerate_compact_labels, normalize_basis_mode


@recordclass(('basis_mode', 'rank', 'L_R', 'label_count', 'coefficient_layout'), frozen = True)
class LinearACEScheduleGroup:
    """Descriptor labels grouped by ACE basis mode, rank, and target irrep."""

    def as_dict(self):
        return {
            "basis_mode": str(self.basis_mode),
            "rank": int(self.rank),
            "L_R": int(self.L_R),
            "label_count": int(self.label_count),
            "coefficient_layout": str(self.coefficient_layout),
        }


@recordclass(('settings_by_L', 'basis_modes', 'groups', 'total_labels', 'metadata'), frozen = True)
class LinearACESchedule:
    """Static linear-ACE descriptor grouping summary."""

    def as_dict(self):
        return {
            "basis_modes": [str(mode) for mode in self.basis_modes],
            "groups": [group.as_dict() for group in self.groups],
            "total_labels": int(self.total_labels),
            "metadata": dict(self.metadata),
        }


@recordclass(('L_in', 'L_geom', 'L_out', 'reduced_basis_mode', 'reconstruct_products', 'route_count', 'exact_dim', 'naive_dim', 'path_parameter_count'), frozen = True)
class SymmetricPowerKernelGroup:
    """Repeated-block symmetric-power kernel option for descriptor/model routes."""

    def as_dict(self):
        return {
            "source": str(self.source),
            "rank": int(self.rank),
            "n_in": tuple(int(x) for x in self.n_in),
            "l_in": tuple(int(x) for x in self.l_in),
            "eta": int(self.eta),
            "input_L": int(self.input_L),
            "power": int(self.power),
            "outputs": tuple(int(x) for x in self.outputs),
            "multiplicities": tuple(int(x) for x in self.multiplicities),
            "term_counts": tuple(int(x) for x in self.term_counts),
        }


@recordclass(('source', 'rank', 'groups', 'metadata'), frozen = True)
class SymmetricPowerKernelSummary:
    """Summary of optional symmetric-power kernels for repeated descriptor blocks."""

    def as_dict(self):
        return {
            "source": str(self.source),
            "rank": int(self.rank),
            "groups": [group.as_dict() for group in self.groups],
            "metadata": dict(self.metadata),
        }


def _mode_name(mode):
    if mode in {None, "exact"}:
        return "exact"
    return str(mode)


def _settings_summary(settings):
    return {
        "ranks": tuple(int(x) for x in settings.ranks),
        "L_R": int(settings.L_R),
        "basis_type": str(settings.basis_type),
        "tree_type": str(settings.tree_type),
    }


def build_linear_ace_schedule(settings_by_L, *, basis_modes = ("exact", "primitive_invariant", "primitive_equivariant_module")):
    """Build a compact descriptor schedule summary from descriptor settings."""

    normalized_settings = {int(L): settings for L, settings in dict(settings_by_L).items()}
    groups = []
    total_labels = 0
    for L_R, settings in sorted(normalized_settings.items()):
        for mode in basis_modes:
            try:
                basis_mode = normalize_basis_mode(None if mode == "exact" else mode, L_R=int(settings.L_R))
            except ValueError:
                continue
            labels = enumerate_compact_labels(settings, basis_mode=basis_mode)
            counts_by_rank = defaultdict(int)
            for label in labels:
                counts_by_rank[int(label.rank)] += 1
            for rank, count in sorted(counts_by_rank.items()):
                groups.append(
                    LinearACEScheduleGroup(
                        basis_mode=_mode_name(basis_mode),
                        rank=int(rank),
                        L_R=int(L_R),
                        label_count=int(count),
                        coefficient_layout="compact_label_payloads",
                    )
                )
            total_labels += len(labels)
    return LinearACESchedule(
        settings_by_L={int(L): _settings_summary(settings) for L, settings in normalized_settings.items()},
        basis_modes=tuple(_mode_name(None if mode == "exact" else mode) for mode in basis_modes),
        groups=tuple(groups),
        total_labels=int(total_labels),
        metadata={
            "schedule_kind": "linear_ace",
            "group_key": "(basis_mode, rank, L_R)",
            "runtime_target": "static_descriptor_tables",
        },
    )


def _repeated_blocks_from_record(record, *, min_power, max_power):
    counter = Counter(zip(record["n_in"], record["l_in"]))
    blocks = []
    for (eta, input_L), power in sorted(counter.items()):
        if int(power) < int(min_power) or int(power) > int(max_power):
            continue
        blocks.append((int(eta), int(input_L), int(power)))
    return tuple(blocks)


def build_symmetric_power_kernel_summary_from_exhaustive_labels(
    *,
    source,
    rank,
    target_l_avs = (1,),
    strict_max_li = 2,
    max_labels = 8,
    min_power = 2,
    max_power = 8,
    include_term_counts = False,
):
    """Summarize optional kernels for repeated blocks found by exhaustive labels."""

    from ye3t.api import (
        allowed_symmetric_power_outputs,
        enumerate_rank_labels,
        symmetric_power_monomial_table,
        symmetric_power_output_multiplicity,
    )

    labels = enumerate_rank_labels(
        int(rank),
        tuple(target_l_avs),
        strict_max_li=None if strict_max_li is None else int(strict_max_li),
        max_labels=None if max_labels is None else int(max_labels),
    )
    candidates = []
    for record in labels:
        blocks = _repeated_blocks_from_record(
            record,
            min_power=int(min_power),
            max_power=int(max_power),
        )
        if blocks:
            max_block_power = max(block[2] for block in blocks)
            candidates.append((max_block_power, record, blocks))
    candidates.sort(key=lambda item: (-int(item[0]), tuple(item[1]["n_in"]), tuple(item[1]["l_in"])))

    groups = []
    for _, record, blocks in candidates:
        for eta, input_L, power in blocks:
            outputs = allowed_symmetric_power_outputs(power, input_L)
            multiplicities = tuple(
                symmetric_power_output_multiplicity(power, input_L, output_L)
                for output_L in outputs
            )
            if include_term_counts:
                term_counts = tuple(
                    symmetric_power_monomial_table(power, input_L, output_L).term_count
                    for output_L in outputs
                )
            else:
                term_counts = tuple(0 for _ in outputs)
            groups.append(
                SymmetricPowerKernelGroup(
                    source=str(source),
                    rank=int(record["rank"]),
                    n_in=tuple(int(x) for x in record["n_in"]),
                    l_in=tuple(int(x) for x in record["l_in"]),
                    eta=int(eta),
                    input_L=int(input_L),
                    power=int(power),
                    outputs=tuple(int(x) for x in outputs),
                    multiplicities=tuple(int(x) for x in multiplicities),
                    term_counts=tuple(int(x) for x in term_counts),
                )
            )
    return SymmetricPowerKernelSummary(
        source=str(source),
        rank=int(rank),
        groups=tuple(groups),
        metadata={
            "schedule_kind": "symmetric_power_kernel_options",
            "label_source": "exhaustive_enum",
            "target_l_avs": tuple(target_l_avs),
            "strict_max_li": strict_max_li,
            "max_labels": max_labels,
            "include_term_counts": bool(include_term_counts),
        },
    )



__all__ = [
    "LinearACEScheduleGroup",
    "LinearACESchedule",
    "SymmetricPowerKernelGroup",
    "SymmetricPowerKernelSummary",
    "build_linear_ace_schedule",
    "build_symmetric_power_kernel_summary_from_exhaustive_labels",
]
