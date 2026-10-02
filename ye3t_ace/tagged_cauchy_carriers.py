"""Occurrence supports for compiler-owned tagged-Cauchy carriers.

An occurrence is a directed edge including its integer periodic image. Tags
are ordered distinct occurrences, not distinct base atom IDs. The density
context remains inclusive. Pair indices are decoded from center CSR offsets
one chunk at a time; no quadratic pair catalogue is retained.
"""

import hashlib
import json
import math

import torch

from ye3t.couplings import (
    compile as compile_coupling,
    shifted_jacobi_ladder_with_derivative,
    shifted_jacobi_normalization_squared,
)
from ye3t.runtime.execution_plan import symmetric_power_shared_sparse_monomial_contraction
from ye3t.backends.triton_joint import SparseBilinearTable, sparse_bilinear_forward
from ye3t.backends._indexed_tagged_pair import indexed_tagged_pair
from ye3t.backends._deterministic_polynomial import polynomial_plan, deterministic_polynomial
from ye3t.backends._deterministic_bilinear import bilinear_plan, deterministic_bilinear

from .lifted_cauchy_linear import LiftedCauchyPolynomialSource


class _TaggedCauchyPolynomialSchedule(torch.nn.Module):
    """Device buffers and execution for one compiler-owned support schedule."""

    def __init__(self, schedule, *, backend="native", dtype=torch.float64, execution="joint_polynomial", accumulation_dtype="input", execution_plan=None):
        super().__init__()
        self.schedule = schedule
        self.backend = str(backend)
        if self.backend not in {"native", "reference"}:
            raise ValueError("tagged polynomial backend must explicitly be native or reference")
        self.tag_count = int(schedule["tag_count"])
        self.support_tag_count = int(schedule.get("support_tag_count", self.tag_count))
        if self.support_tag_count < 0 or self.support_tag_count > self.tag_count:
            raise ValueError("tagged source schedule has an invalid physical support tag count")
        self.inventory = tuple(schedule["inventory"])
        self._buffer_names = ("term_offsets", "term_components", "term_exponents", "output_offsets",
                              "coefficient_terms", "coefficient_outputs", "coefficient_values")
        for name in self._buffer_names:
            self.register_buffer(name, torch.tensor(schedule[name],
                dtype=dtype if name == "coefficient_values" else torch.long))
        coordinates = tuple(schedule["input_coordinates"])
        self.register_buffer("input_roles", torch.tensor([record[1] for record in coordinates], dtype=torch.long))
        offsets = [0]
        for channel in schedule["channels"]:
            offsets.append(offsets[-1] + 2 * int(channel["l"]) + 1)
        self.register_buffer("input_components", torch.tensor(
            [offsets[record[0]] + record[2] for record in coordinates], dtype=torch.long))
        self.channel_offsets = tuple(offsets)
        self.execution = str(execution)
        self.accumulation_dtype = str(accumulation_dtype)
        if self.accumulation_dtype not in {"input", "float64"}:
            raise ValueError("source accumulation dtype must be input or float64")
        self.last_execution = "unexecuted"
        if self.execution not in {"joint_polynomial", "role_factorized", "role_factorized_indexed"}:
            raise ValueError("unknown tagged source execution policy")
        if self.support_tag_count != self.tag_count and self.execution != "joint_polynomial":
            raise ValueError("edge-marginal sources require the compiled joint polynomial execution")
        self.factor_schedule = None
        if (self.tag_count == 2 and self.support_tag_count == 2
                and self.execution in {"role_factorized", "role_factorized_indexed"}):
            self.factor_schedule = execution_plan if execution_plan is not None else compile_coupling({
                "kind": "tagged_cauchy_carrier_execution", "execution": "role_factorized", "source_schedule": schedule})
            body = {key: value for key, value in self.factor_schedule.items() if key != "self_hash"}
            actual_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if (actual_hash != self.factor_schedule.get("self_hash")
                    or self.factor_schedule.get("schema") != "ye3t_tagged_role_factor_execution_v1"
                    or self.factor_schedule.get("source_schedule_hash") != schedule["self_hash"]
                    or not self.factor_schedule.get("certificate", {}).get("passed")):
                raise ValueError("tagged factor execution plan hash or source binding mismatch")
            for stage in ("density", "edge", "pair", "indexed_pair"):
                for name, values in self.factor_schedule[stage].items():
                    if isinstance(values, int) or (stage == "edge" and name == "coefficient"):
                        continue
                    self.register_buffer("factor_" + stage + "_" + name, torch.tensor(values,
                        dtype=dtype if name in {"coefficient", "coefficient_values"} else torch.long), persistent=False)
        self._deterministic_sizes = {}
        self._deterministic_table_counts = {}
        programs = {"joint": (schedule, len(coordinates), schedule["output_dimension"])}
        if self.factor_schedule is not None and self.backend == "native":
            factor = self.factor_schedule
            programs["density"] = (factor["density"], offsets[-1], len(factor["density"]["output_offsets"]) - 1)
            for stage, widths in (("edge", (offsets[-1], programs["density"][2])),
                                  ("pair", (factor["edge"]["output_width"], offsets[-1]))):
                table = factor[stage]
                self._register_deterministic(stage, bilinear_plan(table["left_index"], table["right_index"],
                    table["output_index"], *widths, table["output_width"]))
        if self.backend == "native":
            for stage, (program, input_width, output_width) in programs.items():
                self._register_deterministic(stage, polynomial_plan(*(program[name] for name in
                    ("term_offsets", "term_components", "term_exponents", "coefficient_terms", "coefficient_outputs")),
                    input_width, output_width))

    def _register_deterministic(self, stage, plan):
        tables, sizes = plan
        self._deterministic_sizes[stage] = sizes
        self._deterministic_table_counts[stage] = len(tables)
        for index, value in enumerate(tables):
            self.register_buffer("deterministic_" + stage + "_" + str(index), value, persistent=False)

    def _deterministic_tables(self, stage):
        return tuple(getattr(self, "deterministic_" + stage + "_" + str(index))
                     for index in range(self._deterministic_table_counts[stage]))

    def _polynomial(self, stage, values):
        prefix = "factor_density_" if stage == "density" else ""
        buffers = tuple(getattr(self, prefix + name) for name in self._buffer_names)
        if self.backend == "native" and torch.are_deterministic_algorithms_enabled():
            return deterministic_polynomial(values, buffers[-1].to(values.dtype),
                self._deterministic_tables(stage), self._deterministic_sizes[stage])
        return symmetric_power_shared_sparse_monomial_contraction(values, *buffers, backend=self.backend)

    def _bilinear(self, stage, left, right):
        table = SparseBilinearTable(**{name: getattr(self, "factor_" + stage + "_" + name)
            for name in ("left_index", "right_index", "output_index")},
            # Preserve the loaded artifact's actual coefficient precision,
            # including an f32 artifact subsequently evaluated in f64.
            coefficient=self.coefficient_values if stage == "edge" else self.factor_pair_coefficient,
            output_width=self.factor_schedule[stage]["output_width"])
        if self.backend == "native" and torch.are_deterministic_algorithms_enabled():
            return deterministic_bilinear(left, right, table.coefficient.to(left.dtype),
                self._deterministic_tables(stage), self._deterministic_sizes[stage])
        return sparse_bilinear_forward(left.contiguous(), right.contiguous(), table,
            prefer_triton=self.backend == "native", strict=self.backend == "native", indices_certified=True)

    def _role_factorized(self, edge_values, density_values, support):
        if "factor_maps" not in support:
            # Immutable periodic occurrence indices only; every density below
            # remains the full inclusive root density, never a chunk-local sum.
            edges, inverse = torch.unique(support["tag_edges"][:, 0], sorted=True, return_inverse=True)
            centers, center_inverse = torch.unique(support["centers"], sorted=True, return_inverse=True)
            edge_centers = torch.empty_like(edges)
            edge_centers.scatter_(0, inverse, center_inverse)
            support["factor_maps"] = (edges, inverse, centers, edge_centers)
        edges, inverse, centers, edge_centers = support["factor_maps"]
        density = self._polynomial("density", density_values.index_select(0, centers))
        edge = self._bilinear("edge", edge_values.index_select(0, edges), density.index_select(0, edge_centers))
        if self.execution == "role_factorized_indexed" and self.backend == "native":
            if "factor_left_incidence" not in support:
                counts = torch.bincount(inverse, minlength=len(edges))
                pointers = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
                support["factor_left_incidence"] = (pointers, torch.argsort(inverse, stable=True))
            return indexed_tagged_pair(edge, edge_values, (
                inverse, support["tag_edges"][:, 1].contiguous(), *support["factor_left_incidence"],
                *(getattr(self, "factor_indexed_pair_" + name) for name in (
                    "output_offsets", "output_left_indices", "right_offsets", "right_left_indices",
                    "left_outputs", "left_rights"))), self.factor_schedule["output_dimension"])
        return self._bilinear("pair", edge.index_select(0, inverse),
            edge_values.index_select(0, support["tag_edges"][:, 1]))

    def forward(self, edge_values, density_values, support):
        output_dtype = edge_values.dtype
        if self.accumulation_dtype == "float64":
            edge_values, density_values = edge_values.double(), density_values.double()
        if self.tag_count == 2 and self.execution in {"role_factorized", "role_factorized_indexed"}:
            result = self._role_factorized(edge_values, density_values, support)
            self.last_execution = ("native_density_and_triton_indexed_segmented_pair" if self.execution == "role_factorized_indexed"
                else "native_density_polynomial_and_triton_bilinear") if self.backend == "native" else "reference_role_factorized"
            if self.backend == "native" and torch.are_deterministic_algorithms_enabled():
                self.last_execution += "_deterministic_product_rule"
            return result.to(output_dtype)
        centers, tags = support["centers"], support["tag_edges"]
        roles = [edge_values.index_select(0, tags[:, role]) for role in range(self.support_tag_count)]
        roles.append(density_values.index_select(0, centers))
        values = torch.stack(roles, dim=1)[:, self.input_roles, self.input_components]
        result = self._polynomial("joint", values)
        self.last_execution = self.backend + "_joint_polynomial"
        if self.backend == "native" and torch.are_deterministic_algorithms_enabled():
            self.last_execution += "_deterministic_product_rule"
        return result.to(output_dtype)


class _TaggedCauchyOccurrenceSource(torch.nn.Module):
    """Analytic primitive values and unpooled complete tagged multiplets.

    Geometry is evaluated once per graph and reused by all support chunks.
    The source has no trainable weights and does not detach its geometry.
    """

    def __init__(self, schedules, species, cutoff, *, pair_cutoffs_A=None,
                 backend="native", dtype=torch.float64, source_output_normalization=None,
                 source_execution="joint_polynomial", source_accumulation_dtype="input", execution_plans=None):
        super().__init__()
        self.species = tuple(str(value) for value in species)
        self.cutoff = float(cutoff)
        if self.cutoff <= 0 or len(set(self.species)) != len(self.species):
            raise ValueError("tagged sources require a positive cutoff and unique species")
        table = [[self.cutoff if pair_cutoffs_A is None else pair_cutoffs_A[left + "-" + right]
                  for right in self.species] for left in self.species]
        if any(value <= 0 or value > self.cutoff for row in table for value in row):
            raise ValueError("every pair cutoff must be positive and bounded by the host cutoff")
        self.register_buffer("pair_cutoffs", torch.tensor(table, dtype=dtype))
        self.programs = torch.nn.ModuleDict({str(schedule["tag_count"]):
            _TaggedCauchyPolynomialSchedule(schedule, backend=backend, dtype=dtype, execution=source_execution,
                accumulation_dtype=source_accumulation_dtype,
                execution_plan=(execution_plans or {}).get(str(schedule["tag_count"]))) for schedule in schedules})
        if len(self.programs) != len(schedules):
            raise ValueError("source schedules must have unique support tag counts")
        self.backend = str(backend)
        self.channels = []
        by_key = {}
        for key, program in self.programs.items():
            selected = []
            for channel in program.schedule["channels"]:
                if channel["source_family_id"] != "orthogonal_shifted_jacobi_origin_regular_v1":
                    raise ValueError("tagged occurrence sources require the certified origin-regular shifted-Jacobi family")
                identity = (channel["neighbor_species"], int(channel["radial_channel"]),
                            int(channel["l"]), channel["source_family_id"])
                if identity not in by_key:
                    by_key[identity] = sum(2 * int(record["l"]) + 1 for record in self.channels)
                    self.channels.append(dict(channel))
                begin = by_key[identity]
                selected.extend(range(begin, begin + 2 * int(channel["l"]) + 1))
            self.register_buffer("primitive_indices_" + key, torch.tensor(selected, dtype=torch.long))
        self.channels = tuple(self.channels)
        self.radial_normalizations = tuple(math.sqrt(float(shifted_jacobi_normalization_squared(
            int(channel["radial_channel"]), int(channel["l"])))) for channel in self.channels)
        self.source_output_normalization_kind = "training_rms"
        normalization = dict(source_output_normalization or {})
        self.scale_policy = normalization.get("scale_policy", "unit_rms")
        self.reference_rms = float(normalization.get("reference_rms", 1.))
        if self.scale_policy not in {"unit_rms", "downscale_only"} or not math.isfinite(self.reference_rms) or self.reference_rms <= 0:
            raise ValueError("source scaling requires unit_rms/downscale_only and a positive finite reference_rms")
        self.source_output_zero_image_policy = "retain_with_unit_scale"
        coordinates, source_offsets = [], {}
        for key, program in self.programs.items():
            begin = len(coordinates)
            coordinates.extend(record["label"]["coordinate_id"] for record in program.inventory)
            source_offsets[key] = (begin, len(coordinates))
            component_multiplets = [index for index, record in enumerate(program.inventory)
                                   for _ in range(*record["component_slice"])]
            self.register_buffer("multiplet_indices_" + key, torch.tensor(component_multiplets, dtype=torch.long))
        self.source_coordinate_ids = tuple(coordinates)
        self.source_offsets = source_offsets
        self.register_buffer("source_output_rms", torch.ones(len(coordinates), dtype=dtype))
        self.register_buffer("source_output_scales", torch.ones(len(coordinates), dtype=dtype))
        self.register_buffer("source_output_scaling_calibrated", torch.tensor(False))

    def set_source_output_rms(self, rms):
        rms = torch.as_tensor(rms, dtype=self.source_output_rms.dtype, device=self.source_output_rms.device)
        if rms.shape != self.source_output_rms.shape or not torch.isfinite(rms).all() or (rms < 0).any():
            raise ValueError("source RMS must be finite, nonnegative and cover every complete multiplet")
        with torch.no_grad():
            self.source_output_rms.copy_(rms)
            scales = (self.reference_rms / rms.clamp_min(self.reference_rms) if self.scale_policy == "downscale_only"
                      else torch.where(rms > 0, rms.reciprocal(), torch.ones_like(rms)))
            if not torch.isfinite(scales).all():
                raise ValueError("source RMS rescaling overflowed; use downscale_only for tiny coordinates")
            self.source_output_scales.copy_(scales.clamp_min(torch.finfo(scales.dtype).tiny))
            self.source_output_scaling_calibrated.fill_(True)

    def reset_source_output_scaling(self):
        self.source_output_rms.fill_(1)
        self.source_output_scales.fill_(1)
        self.source_output_scaling_calibrated.fill_(False)

    def source_output_scaling_report(self):
        return {"kind": self.source_output_normalization_kind,
                "scale_policy": self.scale_policy, "reference_rms": self.reference_rms,
                "source_units": "dimensionless_polynomial_in_r_over_pair_cutoff",
                "positive_scale_floor": torch.finfo(self.source_output_scales.dtype).tiny,
                "coordinates_at_scale_floor": int((self.source_output_scales == torch.finfo(self.source_output_scales.dtype).tiny).sum()),
                "calibrated": bool(self.source_output_scaling_calibrated),
                "scope": "training_only_component_normalized_unpooled_support_RMS",
                "rows": tuple({"coordinate_id": key, "rms": rms, "scale": scale}
                              for key, rms, scale in zip(self.source_coordinate_ids,
                                  self.source_output_rms.detach().cpu().tolist(),
                                  self.source_output_scales.detach().cpu().tolist(), strict=True))}

    def primitive_values(self, displacements, central_types, neighbor_types):
        distance = torch.linalg.vector_norm(displacements, dim=-1)
        cutoff = self.pair_cutoffs[central_types, neighbor_types]
        # Graph builders reject coincident atoms. A finite inactive evaluation
        # point also prevents 0/0 from unused padded/outside-cutoff edges.
        active = (distance > 0) & (distance < cutoff)
        safe_distance = torch.where(distance > 0, distance, torch.ones_like(distance))
        unit = displacements / safe_distance[:, None]
        x = torch.where(active, distance / cutoff, torch.full_like(distance, 0.5))
        fallback = torch.zeros_like(unit)
        fallback[:, 0] = 1
        unit = torch.where(active[:, None], unit, fallback)
        geometry = (x[:, None] * unit, neighbor_types, x, unit, active, 1.0)
        angular, jacobi = {}, {}
        values = []
        for channel, normalization in zip(self.channels, self.radial_normalizations, strict=True):
            l, q = int(channel["l"]), int(channel["radial_channel"])
            if l not in angular:
                angular[l] = LiftedCauchyPolynomialSource._compiler_ordered_regular_solid(l, geometry)[0]
                maximum_q = max(int(record["radial_channel"]) for record in self.channels if int(record["l"]) == l)
                jacobi[l] = shifted_jacobi_ladder_with_derivative(maximum_q, 4, 2 * l + 2, x)[0]
            radial = normalization * (1 - x).square() * jacobi[l][q]
            species_index = self.species.index(channel["neighbor_species"])
            mask = active & (neighbor_types == species_index)
            values.append(angular[l] * (radial * mask)[:, None])
        return torch.cat(values, dim=1)

    def geometry_values(self, displacements, atom_types, edge_index, atom_count):
        # Accumulation precision covers the complete analytic source, including
        # primitive radial/angular evaluation and the inclusive density sum.
        # Casting only the final polynomial inputs cannot recover these bits.
        if any(program.accumulation_dtype == "float64" for program in self.programs.values()):
            displacements = displacements.double()
        edge_values = self.primitive_values(displacements, atom_types[edge_index[0]], atom_types[edge_index[1]])
        density = edge_values.new_zeros((int(atom_count), edge_values.shape[1])).index_add(0, edge_index[0], edge_values)
        return edge_values, density

    def forward(self, tag_count, edge_values, density_values, support, *, normalized=True):
        key = str(int(tag_count))
        indices = getattr(self, "primitive_indices_" + key)
        output = self.programs[key](edge_values.index_select(1, indices), density_values.index_select(1, indices), support)
        output = output.to(self.source_output_scales.dtype)
        if normalized:
            begin, end = self.source_offsets[key]
            scales = self.source_output_scales[begin:end].index_select(0, getattr(self, "multiplet_indices_" + key))
            output = output * scales
        return output


def tagged_support_layout(edge_index, atom_count, shifts=None):
    """Validate an immutable edge topology and build its linear-size CSR.

    Call outside the differentiable support rounds and reuse with the same
    neighbor graph. The returned layout stores O(atoms + edges) integers.
    It does not depend on positions, model weights, or target values.
    """

    edges = torch.as_tensor(edge_index, dtype=torch.long)
    atom_count = int(atom_count)
    if atom_count < 0 or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges] and atom_count >= 0")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= atom_count):
        raise ValueError("tagged edge endpoints lie outside atom_count")
    if shifts is None:
        images = torch.zeros((edges.shape[1], 3), dtype=torch.long, device=edges.device)
    else:
        supplied = torch.as_tensor(shifts, device=edges.device)
        if supplied.shape != (edges.shape[1], 3):
            raise ValueError("tagged periodic shifts must have shape [edges, 3]")
        images = supplied.to(torch.long)
        if not bool(torch.all(supplied == images)):
            raise ValueError("tagged occurrence identity requires integer periodic shifts")
    identity = torch.cat((edges.T, images), dim=1)
    unique_identity, identity_order = torch.unique(identity, dim=0, sorted=True, return_inverse=True)
    if unique_identity.shape[0] != edges.shape[1]:
        raise ValueError("duplicate directed periodic occurrences in tagged support graph")
    if bool(torch.any((edges[0] == edges[1]) & torch.all(images == 0, dim=1))):
        raise ValueError("the zero-image self edge is not a neighbor occurrence")
    edge_order = torch.argsort(edges[0], stable=True)
    counts = torch.bincount(edges[0], minlength=atom_count)
    zero = torch.zeros(1, dtype=torch.long, device=edges.device)
    offsets = torch.cat((zero, counts.cumsum(0)))
    pair_offsets = torch.cat((zero, (counts * (counts - 1)).cumsum(0)))
    # This host record is made only when the neighbor topology changes.
    identity_record = {
        "schema": "ye3t_tagged_occurrence_support_v1",
        "atom_count": atom_count,
        "edge_identity": identity.detach().cpu().tolist(),
        "ordered_tags": True,
        "distinctness": "directed_edge_including_periodic_image",
        "density_context": "inclusive",
    }
    encoded = json.dumps(identity_record, sort_keys=True, separators=(",", ":"))
    return {
        "schema": identity_record["schema"],
        "atom_count": atom_count,
        "edge_index": edges,
        "shifts": images,
        "edge_order": edge_order,
        "canonical_edge_order": torch.argsort(identity_order, stable=True),
        "neighbor_counts": counts,
        "edge_offsets": offsets,
        "pair_offsets": pair_offsets,
        "support_counts": (atom_count, int(edges.shape[1]), int(pair_offsets[-1])),
        "topology_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def tagged_support_count(layout, tag_count, *, unordered=False):
    """Count one support family, caching only center offsets for higher tags."""
    tag_count = int(tag_count)
    if tag_count < 0:
        raise ValueError("tag_count must be nonnegative")
    if tag_count <= 2:
        count = int(layout["support_counts"][tag_count])
        return count // 2 if unordered and tag_count == 2 else count
    cache = layout.setdefault("unordered_tag_offsets" if unordered else "higher_tag_offsets", {})
    if tag_count not in cache:
        degrees = layout["neighbor_counts"]
        bound = torch.iinfo(degrees.dtype).max
        counts = []
        total = 0
        for degree in degrees.detach().cpu().tolist():
            count = math.comb(degree, tag_count) if unordered else math.perm(degree, tag_count)
            total += count
            if total > bound:
                raise OverflowError("tag-support offsets exceed the integer index range")
            counts.append(total)
        cache[tag_count] = torch.tensor((0, *counts), dtype=torch.long, device=degrees.device)
    return int(cache[tag_count][-1])


def tagged_support_chunks(layout, tag_count, chunk_size, *, tag_swap_reduction="none"):
    """Yield bounded ordered-support index arrays for any feasible tag count.

    The two-tag decoder omits only an identical directed periodic occurrence.
    Two different periodic images of the same base atom remain distinct tags.
    Exact-character reduction enumerates one orientation with orbit weight two.
    No geometry is detached: this routine constructs indices only.
    """

    tag_count = int(tag_count)
    chunk_size = int(chunk_size)
    if tag_count < 0:
        raise ValueError("tag_count must be nonnegative")
    if chunk_size <= 0:
        raise ValueError("support chunk_size must be positive")
    if tag_swap_reduction not in {"none", "exact_tag_character", "exact_tag_orbit"}:
        raise ValueError("unknown tag-swap reduction policy")
    if tag_count > 2 and tag_swap_reduction == "exact_tag_character":
        raise ValueError("exact tag-character reduction is defined only for two tags")
    reduced = tag_count == 2 and tag_swap_reduction != "none"
    orbit_reduced = tag_count > 2 and tag_swap_reduction == "exact_tag_orbit"
    if tag_count <= 2:
        count = int(layout["support_counts"][tag_count])
        offsets = layout["pair_offsets"] if tag_count == 2 else None
    else:
        count = tagged_support_count(layout, tag_count, unordered=orbit_reduced)
        offsets = layout["unordered_tag_offsets" if orbit_reduced else "higher_tag_offsets"][tag_count]
    order = layout["canonical_edge_order"] if reduced or orbit_reduced else layout["edge_order"]
    pair_offsets = layout["pair_offsets"] // 2 if reduced else layout["pair_offsets"]
    if reduced:
        count //= 2
    for start in range(0, count, chunk_size):
        index = torch.arange(start, min(start + chunk_size, count), device=order.device)
        if tag_count == 0:
            centers = index
            tags = torch.empty((len(index), 0), dtype=torch.long, device=order.device)
        elif tag_count == 1:
            tags = order[index, None]
            centers = layout["edge_index"][0, tags[:, 0]]
        elif tag_count == 2:
            centers = torch.searchsorted(pair_offsets[1:], index, right=True)
            degree = layout["neighbor_counts"][centers]
            local = index - pair_offsets[centers]
            if reduced:
                second = torch.floor((torch.sqrt(1 + 8 * local.to(torch.float64)) + 1) / 2).to(torch.long)
                first = local - second * (second - 1) // 2
            else:
                first = torch.div(local, degree - 1, rounding_mode="floor")
                second = local.remainder(degree - 1)
                second = second + (second >= first).to(torch.long)
            origin = layout["edge_offsets"][centers]
            tags = torch.stack((order[origin + first], order[origin + second]), dim=1)
        else:
            centers = torch.searchsorted(offsets[1:], index, right=True)
            degree = layout["neighbor_counts"][centers]
            local = index - offsets[centers]
            positions = []
            if orbit_reduced:
                # Algorithmic reference: colex combination unranking,
                # Kruchinin et al. (2022), Algorithm 1. Independent implementation;
                # no source copied. Complete right-tag multiplets make scalar
                # pooling identical on the k! ordered representatives.
                maximum = int(layout["neighbor_counts"].max())
                tables = layout.setdefault("binomial_tables", {})
                bound = torch.iinfo(local.dtype).max
                for choose in range(tag_count, 0, -1):
                    if choose not in tables:
                        tables[choose] = torch.tensor(tuple(min(math.comb(value, choose), bound)
                            for value in range(maximum + 1)), dtype=torch.long, device=order.device)
                    position = torch.searchsorted(tables[choose], local, right=True) - 1
                    local = local - tables[choose][position]
                    positions.append(position)
                positions.reverse()
            else:
                # Algorithmic reference: Lehmer-code/factoradic unranking;
                # Tarau, arXiv:0808.0554. Independent partial-permutation
                # adaptation with falling-factorial place values.
                for slot in range(tag_count):
                    divisor = torch.ones_like(local)
                    for remaining in range(slot + 1, tag_count):
                        divisor = divisor * (degree - remaining)
                    digit = torch.div(local, divisor, rounding_mode="floor")
                    local = local.remainder(divisor)
                    position = digit
                    if positions:
                        for used in torch.sort(torch.stack(positions, dim=1), dim=1).values.unbind(1):
                            position = position + (position >= used).to(position.dtype)
                    positions.append(position)
            origin = layout["edge_offsets"][centers]
            tags = torch.stack([order[origin + position] for position in positions], dim=1)
        yield {"centers": centers, "tag_edges": tags, "support_indices": index,
               "orbit_weight": math.factorial(tag_count) if orbit_reduced else 2 if reduced else 1}
