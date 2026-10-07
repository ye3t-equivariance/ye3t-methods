"""Linear Young-character descriptor maps for scalar potential fits."""

import hashlib
import json
import math

import torch

from ye3t_methods.atomistic._record import recordclass
from ye3t_methods.atomistic.equivariant_calc.labeling import SingleChannelLabel
from ye3t_methods.atomistic.equivariant_calc.site_basis_v2 import (
    DEFAULT_ATOMIC_BASE_NORMALIZATION,
    SiteBasisConfig,
    SiteBasisV2,
    site_real_block_to_ye3t_tesseral,
)
from ye3t.representations.builder import GeneralizedExactSymbolicLabeler, _raw_tensor_basis_states
from ye3t.runtime.native import real_tesseral_to_complex_multiplet


@recordclass(("columns", "metadata"))
class YE3TLinearCharacterDesign:
    """Atom-summed scalar design matrix with descriptor provenance."""

    @property
    def descriptor_count(self):
        return int(self.columns.shape[1])


def _stable_hash(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in tuple(value)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sum_by_graph(values, graph_ids, num_graphs):
    if graph_ids is None:
        return values.sum(dim=0, keepdim=True)
    graph_ids = graph_ids.to(device=values.device, dtype=torch.long)
    out = torch.zeros((int(num_graphs), int(values.shape[1])), dtype=values.dtype, device=values.device)
    out.index_add_(0, graph_ids, values)
    return out


def _column_indices_from_budget(count, descriptor_budget):
    count = int(count)
    if descriptor_budget is None:
        return tuple(range(count))
    budget = int(descriptor_budget)
    if budget < 0:
        raise ValueError("descriptor_budget must be nonnegative.")
    if budget > count:
        raise ValueError("descriptor_budget exceeds the available descriptor count.")
    return tuple(range(budget))


def _descriptor_partitions(value):
    return tuple(tuple(int(part) for part in tuple(parts)) for parts in tuple(value or ()))


def _descriptor_source_from_feature(feature):
    input_n = tuple(int(x) for x in feature.get("n_in", ()))
    source = {
        "input_n": input_n,
        "density_n": tuple(int(x) for x in feature.get("density_n_in", input_n)),
        "input_l": tuple(int(x) for x in feature.get("l_in", ())),
        "L": int(feature.get("L_R", feature.get("L", 0))),
        "copy_index": int(feature.get("copy_index", 0)),
        "permutation_partitions": _descriptor_partitions(feature.get("permutation_partitions", ())),
        "permutation_representation": str(feature.get("permutation_sector", "trivial")),
        "permutation_irrep": str(feature.get("permutation_irrep", "")),
    }
    if "density_mu0_in" in feature:
        source["density_mu0"] = tuple(int(x) for x in feature.get("density_mu0_in", ()))
    if "density_mu_in" in feature:
        source["density_mu"] = tuple(int(x) for x in feature.get("density_mu_in", ()))
    return source


def _normal_optional_slot_tuple(source, key):
    values = source.get(key)
    if values is None:
        return None
    values = tuple(int(x) for x in tuple(values))
    if len(values) != len(tuple(source.get("input_l", ()))):
        raise ValueError(str(key) + " length must match source rank.")
    return values


def _descriptor_sources(descriptor):
    if "feature" in descriptor:
        return (_descriptor_source_from_feature(descriptor["feature"]),)
    label = descriptor.get("label", descriptor)
    out = []
    for side in ("left", "right"):
        source = dict(label[side])
        source["input_n"] = tuple(int(x) for x in source.get("input_n", ()))
        source["density_n"] = tuple(int(x) for x in source.get("density_n", source["input_n"]))
        source["input_l"] = tuple(int(x) for x in source.get("input_l", ()))
        source["L"] = int(source.get("L", 0))
        source["copy_index"] = int(source.get("copy_index", 0))
        source["permutation_partitions"] = _descriptor_partitions(source.get("permutation_partitions", ()))
        density_mu0 = _normal_optional_slot_tuple(source, "density_mu0")
        density_mu = _normal_optional_slot_tuple(source, "density_mu")
        if density_mu0 is not None or density_mu is not None:
            if density_mu0 is None or density_mu is None:
                raise ValueError("Saved descriptor chemical source labels require both density_mu0 and density_mu.")
            source["density_mu0"] = density_mu0
            source["density_mu"] = density_mu
        out.append(source)
    return tuple(out)


def _descriptor_max_n_l(descriptor_set):
    max_n = 0
    max_l = 0
    for descriptor in tuple(descriptor_set.get("descriptors", ())):
        for source in _descriptor_sources(descriptor):
            max_n = max((int(max_n),) + tuple(int(x) for x in source.get("density_n", source.get("input_n", ()))))
            max_l = max((int(max_l),) + tuple(int(x) for x in source.get("input_l", ())))
    return int(max_n), int(max_l)


def _saved_source_key(source):
    return (
        tuple(int(x) for x in source["input_n"]),
        tuple(int(x) for x in source["input_l"]),
        _descriptor_partitions(source["permutation_partitions"]),
        int(source["L"]),
    )


def _saved_density_key(source):
    key = (
        _saved_source_key(source),
        tuple(int(x) for x in source.get("density_n", source["input_n"])),
    )
    density_mu0 = source.get("density_mu0")
    density_mu = source.get("density_mu")
    if density_mu0 is not None or density_mu is not None:
        key = key + (
            tuple(int(x) for x in tuple(density_mu0)),
            tuple(int(x) for x in tuple(density_mu)),
        )
    return key


def _complex_vector_from_sympy(vector):
    return tuple(complex(value.evalf(30)) for value in tuple(vector))


def _normalize_complex_vector(values):
    norm_sq = sum((value.conjugate() * value).real for value in tuple(values))
    if norm_sq <= 0.0:
        raise ValueError("Exact lowered multiplet vector has zero norm.")
    norm = math.sqrt(float(norm_sq))
    return tuple(complex(value) / norm for value in tuple(values))


_SAVED_SECTOR_CONTRACTION_CACHE = {}


class _SavedExactSectorContractions:
    """Exact lowered-multiplet contraction vectors for one saved source sector."""

    def __init__(self, source, dtype):
        self.nin = tuple(int(x) for x in source["input_n"])
        self.lin = tuple(int(x) for x in source["input_l"])
        self.partitions = _descriptor_partitions(source["permutation_partitions"])
        self.L = int(source["L"])
        self.dtype = dtype
        labeler = GeneralizedExactSymbolicLabeler(self.nin, self.lin)
        self.sector = labeler.sector_for_partitions(self.partitions)
        self.raw_states = _raw_tensor_basis_states(self.lin)
        self.carrier_indices = tuple(
            tuple(int(x) for x in item)
            for item in self.sector.carrier_index_tuples_by_L.get(int(self.L), ())
        )
        lowered = self.sector.lowered_multiplets_by_L.get(int(self.L), {})
        if not lowered:
            raise ValueError("Saved descriptor source has no exact lowered multiplets for the requested L.")
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        rows = {}
        for key, multiplet in lowered.items():
            copy_index, carrier_index = key
            for magnetic, vector in multiplet.items():
                values = _normalize_complex_vector(_complex_vector_from_sympy(vector))
                rows[(int(copy_index), tuple(int(x) for x in carrier_index), int(magnetic))] = torch.tensor(
                    values,
                    dtype=complex_dtype,
                )
        self.rows = rows

    def to_device(self, device):
        for key, value in tuple(self.rows.items()):
            if value.device != device:
                self.rows[key] = value.to(device=device)
        return self

    def row(self, copy_index, carrier_index, magnetic):
        return self.rows[(int(copy_index), tuple(int(x) for x in carrier_index), int(magnetic))]

    def raw_monomials(self, density, density_n=None, density_mu0=None, density_mu=None):
        density_n = tuple(int(x) for x in (self.nin if density_n is None else density_n))
        if len(density_n) != len(self.lin):
            raise ValueError("density_n length must match the exact source rank.")
        if density_mu0 is not None or density_mu is not None:
            if density_mu0 is None or density_mu is None:
                raise ValueError("density_mu0 and density_mu must be supplied together.")
            density_mu0 = tuple(int(x) for x in tuple(density_mu0))
            density_mu = tuple(int(x) for x in tuple(density_mu))
            if len(density_mu0) != len(self.lin) or len(density_mu) != len(self.lin):
                raise ValueError("chemical density labels must match the exact source rank.")
        first = next(iter(density.values()))
        rows = []
        for state in self.raw_states:
            value = torch.ones(first.shape[0], dtype=first.dtype, device=first.device)
            for slot_index, magnetic in enumerate(tuple(state)):
                n_value = int(density_n[int(slot_index)])
                l_value = int(self.lin[int(slot_index)])
                if density_mu0 is None:
                    density_key = (n_value, l_value)
                else:
                    density_key = (
                        n_value,
                        l_value,
                        int(density_mu0[int(slot_index)]),
                        int(density_mu[int(slot_index)]),
                    )
                value = value * density[density_key][:, int(magnetic) + int(l_value)]
            rows.append(value)
        return torch.stack(tuple(rows), dim=1)

    def component(self, density, copy_index, carrier_index, magnetic, density_n=None, density_mu0=None, density_mu=None):
        raw = self.raw_monomials(density, density_n=density_n, density_mu0=density_mu0, density_mu=density_mu)
        row = self.row(copy_index, carrier_index, magnetic).to(device=raw.device)
        return raw @ row.conj()


class YE3TSavedDescriptorSetFeatureMap(torch.nn.Module):
    """Evaluate saved exact Young-character descriptor labels as scalars."""

    def __init__(
        self,
        descriptor_set,
        *,
        num_types,
        radial_count=None,
        cutoff=5.0,
        radial_lambda=0.25,
        lmax=None,
        dtype=torch.float64,
        atomic_base_normalization=DEFAULT_ATOMIC_BASE_NORMALIZATION,
        atomic_base_normalization_epsilon=0.0,
        factor_normalization="bounded",
        imaginary_tolerance=1.0e-8,
    ):
        super().__init__()
        self.descriptor_set = dict(descriptor_set)
        self.descriptors = tuple(dict(item) for item in tuple(self.descriptor_set.get("descriptors", ())))
        self.num_types = int(num_types)
        required_n, required_l = _descriptor_max_n_l(self.descriptor_set)
        self.required_radial_count = int(required_n)
        self.required_lmax = int(required_l)
        self.radial_count = int(required_n if radial_count is None else radial_count)
        self.lmax = int(required_l if lmax is None else lmax)
        if self.radial_count < int(required_n):
            raise ValueError("radial_count is below the largest fixed n index in the saved descriptor set.")
        if self.lmax < int(required_l):
            raise ValueError("lmax is below the largest fixed l index in the saved descriptor set.")
        self.cutoff = float(cutoff)
        self.radial_lambda = float(radial_lambda)
        self.dtype = dtype
        self.imaginary_tolerance = float(imaginary_tolerance)
        self.site_basis_config = SiteBasisConfig(
            rc=[float(cutoff)] * (self.num_types * self.num_types),
            lmbda=[float(radial_lambda)] * (self.num_types * self.num_types),
            nradmax=int(self.radial_count),
            lmax=int(self.lmax),
            kmax=0,
            possible_types=tuple(range(self.num_types)),
            charge_mode="none",
            atomic_base_normalization=str(atomic_base_normalization),
            atomic_base_normalization_epsilon=float(atomic_base_normalization_epsilon),
            factor_normalization=str(factor_normalization),
            spherical_backend="real",
            dtype=dtype,
            complex_dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
        )
        self.site_basis = SiteBasisV2(self.site_basis_config)
        self.channels, self.channel_groups = self._build_channels()
        self._sectors = {}
        self.records = self._compile_records()

    def _build_channels(self):
        required = {}
        for descriptor in self.descriptors:
            for source in _descriptor_sources(descriptor):
                density_n = tuple(int(x) for x in source.get("density_n", source["input_n"]))
                for n_value, l_value in zip(density_n, tuple(source["input_l"])):
                    required.setdefault(int(l_value), set()).add(int(n_value))
        channels = []
        groups = {}
        for l_value in sorted(required):
            start = len(channels)
            output_indices = []
            n_values = tuple(sorted(required[int(l_value)]))
            n_to_group = {int(n_value): int(index) for index, n_value in enumerate(n_values)}
            width = 2 * int(l_value) + 1
            for n_value in n_values:
                group_index = int(n_to_group[int(n_value)])
                for mu0 in range(self.num_types):
                    for mu in range(self.num_types):
                        for magnetic in range(-int(l_value), int(l_value) + 1):
                            channels.append(
                                SingleChannelLabel(
                                    mu0=int(mu0),
                                    mu=int(mu),
                                    kappa0=0,
                                    kappa=0,
                                    n=int(n_value),
                                    l=int(l_value),
                                    m=int(magnetic),
                                )
                            )
                            output_indices.append(int(group_index) * int(width) + int(magnetic) + int(l_value))
            groups[int(l_value)] = {
                "start": int(start),
                "stop": int(len(channels)),
                "n_values": n_values,
                "n_to_group": n_to_group,
                "output_indices": tuple(int(x) for x in output_indices),
            }
        return tuple(channels), groups

    def _sector(self, source):
        key = _saved_source_key(source)
        if key not in self._sectors:
            cache_key = (key, str(self.dtype))
            if cache_key not in _SAVED_SECTOR_CONTRACTION_CACHE:
                _SAVED_SECTOR_CONTRACTION_CACHE[cache_key] = _SavedExactSectorContractions(source, self.dtype)
            self._sectors[key] = _SAVED_SECTOR_CONTRACTION_CACHE[cache_key]
        return self._sectors[key]

    def _compile_records(self):
        records = []
        for index, descriptor in enumerate(self.descriptors):
            if "feature" in descriptor:
                source = _descriptor_source_from_feature(descriptor["feature"])
                sector = self._sector(source)
                carrier = sector.carrier_indices[0] if sector.carrier_indices else tuple()
                records.append(
                    {
                        "kind": "direct_scalar",
                        "descriptor_index": int(index),
                        "descriptor_id": str(descriptor.get("descriptor_id", index)),
                        "source": source,
                        "carrier": carrier,
                    }
                )
            else:
                left, right = _descriptor_sources(descriptor)
                if _saved_source_key(left) != _saved_source_key(right):
                    raise ValueError("Saved same-group descriptor must use one source sector.")
                self._sector(left)
                records.append(
                    {
                        "kind": "same_group_young_character_invariant",
                        "descriptor_index": int(index),
                        "descriptor_id": str(descriptor.get("descriptor_id", index)),
                        "left": left,
                        "right": right,
                    }
                )
        return tuple(records)

    def _edge_inputs(self, types, positions, graph_ids=None, pair_indices=None, pair_shifts=None, cell=None, edge_cell=None):
        n_atoms = int(types.shape[0])
        if pair_indices is None:
            indices = torch.arange(n_atoms, dtype=torch.long, device=positions.device)
            center = indices.repeat_interleave(n_atoms)
            neighbor = indices.repeat(n_atoms)
            keep = center != neighbor
            if graph_ids is not None:
                graph_ids = graph_ids.to(device=positions.device, dtype=torch.long)
                keep = keep & (graph_ids.index_select(0, center) == graph_ids.index_select(0, neighbor))
            center = center[keep]
            neighbor = neighbor[keep]
        else:
            pair_indices = pair_indices.to(device=positions.device, dtype=torch.long)
            center = pair_indices[0] if pair_indices.numel() else torch.zeros((0,), dtype=torch.long, device=positions.device)
            neighbor = pair_indices[1] if pair_indices.numel() else torch.zeros((0,), dtype=torch.long, device=positions.device)
        x_ij = positions.index_select(0, neighbor) - positions.index_select(0, center)
        if pair_shifts is not None:
            pair_shifts = pair_shifts.to(device=positions.device, dtype=positions.dtype)
            if edge_cell is not None:
                edge_cell = edge_cell.to(device=positions.device, dtype=positions.dtype)
                x_ij = x_ij + torch.einsum("ei,eij->ej", pair_shifts, edge_cell)
            else:
                if cell is None:
                    raise ValueError("pair_shifts require cell or edge_cell.")
                cell = cell.to(device=positions.device, dtype=positions.dtype)
                if cell.ndim == 2:
                    x_ij = x_ij + pair_shifts @ cell
                else:
                    if graph_ids is None:
                        raise ValueError("batched cell tensors require graph_ids.")
                    edge_graphs = graph_ids.to(device=positions.device, dtype=torch.long).index_select(0, center)
                    x_ij = x_ij + torch.einsum("ei,eij->ej", pair_shifts, cell.index_select(0, edge_graphs))
        keep = torch.linalg.norm(x_ij, dim=-1) < float(self.cutoff)
        return x_ij[keep], torch.stack((center[keep], neighbor[keep]), dim=0), types.to(device=positions.device, dtype=torch.long)

    def _density(self, types, positions, graph_ids=None, pair_indices=None, pair_shifts=None, cell=None, edge_cell=None):
        x_ij, edge_index, atom_types = self._edge_inputs(types, positions, graph_ids, pair_indices, pair_shifts, cell, edge_cell)
        _channels, atomic_base = self.site_basis.compute_atomic_base(
            x_ij=x_ij,
            edge_index=edge_index,
            atom_types=atom_types,
            channels=self.channels,
            real_output=True,
        )
        atomic_base = atomic_base.real if torch.is_complex(atomic_base) else atomic_base
        atomic_base = atomic_base.to(dtype=positions.dtype)
        density = {}
        for l_value, group in self.channel_groups.items():
            width = 2 * int(l_value) + 1
            n_values = tuple(group["n_values"])
            block = atomic_base[:, int(group["start"]) : int(group["stop"])]
            real_pair_values = block.reshape(
                positions.shape[0],
                len(n_values),
                self.num_types,
                self.num_types,
                width,
            )
            real_values = torch.sum(real_pair_values, dim=(2, 3))
            complex_values = real_tesseral_to_complex_multiplet(
                site_real_block_to_ye3t_tesseral(real_values.reshape(-1, width), int(l_value)),
                int(l_value),
            ).reshape(positions.shape[0], len(n_values), width)
            for n_value in n_values:
                density[(int(n_value), int(l_value))] = complex_values[:, int(group["n_to_group"][int(n_value)]), :]
            complex_pair_values = real_tesseral_to_complex_multiplet(
                site_real_block_to_ye3t_tesseral(real_pair_values.reshape(-1, width), int(l_value)),
                int(l_value),
            ).reshape(positions.shape[0], len(n_values), self.num_types, self.num_types, width)
            for n_value in n_values:
                n_index = int(group["n_to_group"][int(n_value)])
                for mu0 in range(self.num_types):
                    for mu in range(self.num_types):
                        density[(int(n_value), int(l_value), int(mu0), int(mu))] = complex_pair_values[
                            :,
                            n_index,
                            int(mu0),
                            int(mu),
                            :,
                        ]
        return density

    def _record_value(self, record, density, raw_cache):
        if record["kind"] == "direct_scalar":
            source = record["source"]
            sector = self._sector(source).to_device(next(iter(density.values())).device)
            key = _saved_density_key(source)
            if key not in raw_cache:
                raw_cache[key] = sector.raw_monomials(
                    density,
                    density_n=source.get("density_n", source["input_n"]),
                    density_mu0=source.get("density_mu0"),
                    density_mu=source.get("density_mu"),
                )
            row = sector.row(int(source["copy_index"]), tuple(record["carrier"]), 0).to(device=raw_cache[key].device)
            return raw_cache[key] @ row.conj()
        left = record["left"]
        right = record["right"]
        sector = self._sector(left).to_device(next(iter(density.values())).device)
        left_key = _saved_density_key(left)
        right_key = _saved_density_key(right)
        if left_key not in raw_cache:
            raw_cache[left_key] = sector.raw_monomials(
                density,
                density_n=left.get("density_n", left["input_n"]),
                density_mu0=left.get("density_mu0"),
                density_mu=left.get("density_mu"),
            )
        if right_key not in raw_cache:
            raw_cache[right_key] = sector.raw_monomials(
                density,
                density_n=right.get("density_n", right["input_n"]),
                density_mu0=right.get("density_mu0"),
                density_mu=right.get("density_mu"),
            )
        left_raw = raw_cache[left_key]
        right_raw = raw_cache[right_key]
        value = torch.zeros(left_raw.shape[0], dtype=left_raw.dtype, device=left_raw.device)
        L = int(left["L"])
        scale = 1.0 / math.sqrt(float(2 * L + 1))
        for carrier in sector.carrier_indices:
            for magnetic in range(-L, L + 1):
                phase = -1.0 if ((L - magnetic) % 2) else 1.0
                left_row = sector.row(int(left["copy_index"]), carrier, int(magnetic)).to(device=left_raw.device)
                right_row = sector.row(int(right["copy_index"]), carrier, -int(magnetic)).to(device=right_raw.device)
                left_component = left_raw @ left_row.conj()
                right_component = right_raw @ right_row.conj()
                value = value + float(phase * scale) * left_component * right_component
        return value

    def _atom_values(self, types, positions, graph_ids=None, pair_indices=None, pair_shifts=None, cell=None, edge_cell=None):
        density = self._density(types, positions, graph_ids, pair_indices, pair_shifts, cell, edge_cell)
        raw_cache = {}
        values = [self._record_value(record, density, raw_cache) for record in self.records]
        columns = torch.stack(tuple(values), dim=1)
        if torch.is_complex(columns):
            residual = torch.max(torch.abs(columns.imag)).detach().cpu().item() if columns.numel() else 0.0
            if residual > float(self.imaginary_tolerance):
                raise RuntimeError("Saved descriptor evaluation produced a non-negligible imaginary scalar residual.")
            columns = columns.real
        return columns.to(dtype=self.dtype)

    def forward(self, types, positions, graph_ids=None, num_graphs=None, pair_indices=None, pair_shifts=None, cell=None, edge_cell=None):
        atom_values = self._atom_values(types, positions, graph_ids, pair_indices, pair_shifts, cell, edge_cell)
        if graph_ids is None:
            graph_count = 1
        elif num_graphs is None:
            graph_count = int(torch.max(graph_ids).detach().cpu().item()) + 1 if graph_ids.numel() else 0
        else:
            graph_count = int(num_graphs)
        return _sum_by_graph(atom_values, graph_ids, graph_count)

    def design_from_samples(self, samples):
        rows = []
        for sample in tuple(samples):
            rows.append(
                self(
                    sample["types"],
                    sample["positions"],
                    sample.get("graph_ids"),
                    sample.get("num_graphs"),
                    sample.get("pair_indices"),
                    sample.get("pair_shifts"),
                    sample.get("cell"),
                    sample.get("edge_cell"),
                )
            )
        columns = torch.cat(tuple(rows), dim=0) if rows else torch.zeros((0, len(self.records)), dtype=self.dtype)
        return YE3TLinearCharacterDesign(columns=columns, metadata=self.report())

    def provenance(self):
        return tuple(
            {
                "descriptor_index": int(record["descriptor_index"]),
                "descriptor_id": str(record["descriptor_id"]),
                "source": str(record["kind"]),
                "uses_nontrivial_source": bool(record["kind"] != "direct_scalar"),
                "final_permutation_representation": "trivial",
            }
            for record in self.records
        )

    def report(self):
        nontrivial = sum(1 for record in self.records if record["kind"] != "direct_scalar")
        return {
            "feature_map": "YE3TSavedDescriptorSetFeatureMap",
            "descriptor_set_name": str(self.descriptor_set.get("name", "")),
            "descriptor_count": int(len(self.records)),
            "nontrivial_descriptor_count": int(nontrivial),
            "descriptor_readout_path": "saved exact Young-character labels evaluated as scalar design columns",
            "coordinate_readout_path": "normalized exact lowered-multiplet contraction vectors in complex magnetic coordinates",
            "density_angular_convention": "site_signed_m_reversed_to_ye3t_tesseral_v1",
            "uses_runtime_gram_matrix": False,
            "supports_chemical_pair_sources": True,
            "real_scalar_check": "imaginary residual is rejected above tolerance",
            "required_radial_count": int(self.required_radial_count),
            "required_lmax": int(self.required_lmax),
            "radial_count": int(self.radial_count),
            "lmax": int(self.lmax),
            "cutoff": float(self.cutoff),
            "radial_lambda": float(self.radial_lambda),
            "uses_scalar_proxy": False,
            "uses_norm_shortcut": False,
            "uses_approximate_intertwiner": False,
            "uses_dropped_sector_standin": False,
        }


