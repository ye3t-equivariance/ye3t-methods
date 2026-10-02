"""Fixed-feature explicit-cluster barPhi energies and ASE evaluation.

Motif products use the compiler's Phi coupling plans for label provenance.
The differentiable reference evaluator supports fixed-cell periodic motifs.
"""

from ye3t_ace._record import recordclass
from dataclasses import field
from itertools import permutations, product
from pathlib import Path
import time

import numpy as np
import torch

from ye3t.representations import (
    decorated_automorphism_group,
    graph_template_automorphisms,
    slot_orbits,
)

try:  # Reuse the existing tesseral helper rather than introducing another one.
    from ye3t_ace.equivariant_calc.labeling import SingleChannelLabel
    from ye3t_ace.equivariant_calc.site_basis_v2 import SiteBasisConfig, SiteBasisV2
    from ye3t_ace.equivariant_calc.site_basis_v2 import _real_spherical_harmonics_l_from_unit_cartesian
except Exception:  # pragma: no cover - local import fallback
    from .equivariant_calc.labeling import SingleChannelLabel
    from .equivariant_calc.site_basis_v2 import SiteBasisConfig, SiteBasisV2
    from .equivariant_calc.site_basis_v2 import _real_spherical_harmonics_l_from_unit_cartesian

try:  # Keep ASE optional for importing the feature/model helpers.
    from ase.calculators.calculator import Calculator as _ASECalculatorBase
    from ase.calculators.calculator import PropertyNotImplementedError as _ASEPropertyNotImplementedError
    from ase.calculators.calculator import all_changes as _ASE_ALL_CHANGES
    from ase.stress import full_3x3_to_voigt_6_stress as _ase_full_3x3_to_voigt_6_stress
    _ASE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on optional ASE
    _ASECalculatorBase = object
    _ASEPropertyNotImplementedError = RuntimeError
    _ASE_ALL_CHANGES = ("positions", "numbers", "cell", "pbc")
    _ase_full_3x3_to_voigt_6_stress = None
    _ASE_IMPORT_ERROR = exc


BRANCH_A = "A"
BRANCH_BAR_PHI = "bar_phi"
ALL_BRANCHES = (BRANCH_A, BRANCH_BAR_PHI)


def _normalize_branch_name(name):
    value = str(name).strip()
    aliases = {
        "a": BRANCH_A,
        "ace": BRANCH_A,
        "A": BRANCH_A,
        "barphi": BRANCH_BAR_PHI,
        "bar_phi": BRANCH_BAR_PHI,
        "bar-Phi": BRANCH_BAR_PHI,
        "barPhi": BRANCH_BAR_PHI,
    }
    out = aliases.get(value, aliases.get(value.lower(), value))
    if out not in ALL_BRANCHES:
        raise ValueError(f"Unknown branch {name!r}; expected a subset of {ALL_BRANCHES!r}.")
    return out


def normalize_branches(branches):
    """Return a deterministic branch tuple preserving user order."""

    seen = []
    for branch in branches:
        normalized = _normalize_branch_name(branch)
        if normalized not in seen:
            seen.append(normalized)
    if not seen:
        raise ValueError("At least one branch must be selected.")
    return tuple(seen)


@recordclass(('name', 'vertex_count', 'edges'), frozen = True)
class MotifTemplate:
    """A small undirected graph template on ordered motif slots."""
    edges = ()

    def __post_init__(self):
        vertex_count = int(self.vertex_count)
        if vertex_count < 1:
            raise ValueError("MotifTemplate.vertex_count must be positive.")
        edges = []
        for a, b in self.edges:
            a = int(a)
            b = int(b)
            if a == b:
                raise ValueError("MotifTemplate edges must not contain self loops.")
            if min(a, b) < 0 or max(a, b) >= vertex_count:
                raise ValueError("MotifTemplate edge references a slot outside the motif.")
            edges.append((min(a, b), max(a, b)))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "vertex_count", vertex_count)
        object.__setattr__(self, "edges", tuple(sorted(set(edges))))

    @property
    def automorphisms(self):
        return graph_template_automorphisms(self.vertex_count, self.edges)

    def to_dict(self):
        return {"name": self.name, "vertex_count": self.vertex_count, "edges": [list(edge) for edge in self.edges]}

    @classmethod
    def from_dict(cls, payload):
        return cls(str(payload["name"]), int(payload["vertex_count"]), tuple(tuple(edge) for edge in payload.get("edges", ())))


@recordclass(('n', 'l', 'm', 'neighbor_type'), frozen = True)
class PhiSlotChannel:
    """One scalar edge channel used in a motif slot.

    ``neighbor_type`` is an optional integer atom-type selector.  The angular
    channel uses the real tesseral convention already used by ``SiteBasisV2``.
    """

    n = 1
    l = 0
    m = 0
    neighbor_type = None

    def __post_init__(self):
        n = int(self.n)
        l = int(self.l)
        m = int(self.m)
        if n < 0:
            raise ValueError("PhiSlotChannel.n must be nonnegative.")
        if l < 0:
            raise ValueError("PhiSlotChannel.l must be nonnegative.")
        if abs(m) > l:
            raise ValueError("PhiSlotChannel.m must satisfy -l <= m <= l.")
        neighbor_type = None if self.neighbor_type is None else int(self.neighbor_type)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "l", l)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "neighbor_type", neighbor_type)

    @property
    def label(self):
        return (self.n, self.l, self.m, self.neighbor_type)

    @property
    def repeated_channel_label(self):
        return (self.n, self.l, self.neighbor_type)

    def to_dict(self):
        return {"n": self.n, "l": self.l, "m": self.m, "neighbor_type": self.neighbor_type}

    @classmethod
    def from_dict(cls, payload):
        return cls(
            n=int(payload.get("n", 1)),
            l=int(payload.get("l", 0)),
            m=int(payload.get("m", 0)),
            neighbor_type=payload.get("neighbor_type", None),
        )


@recordclass(('template', 'channels', 'name'), frozen = True)
class PhiMotifSpec:
    """One decorated motif and slot-channel pattern."""
    name = None

    def __post_init__(self):
        channels = tuple(ch if isinstance(ch, PhiSlotChannel) else PhiSlotChannel.from_dict(ch) for ch in self.channels)
        if len(channels) != self.template.vertex_count:
            raise ValueError("PhiMotifSpec channel count must match the motif vertex count.")
        name = self.name
        if name is None:
            name = self.template.name + ":" + ",".join(_channel_label_text(ch) for ch in channels)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "name", str(name))

    @property
    def decorated_automorphisms(self):
        return decorated_automorphism_group(
            self.template.vertex_count,
            self.template.edges,
            tuple(ch.repeated_channel_label for ch in self.channels),
        )

    @property
    def normalization(self):
        return max(1, len(self.decorated_automorphisms))

    @property
    def slot_orbit_partition(self):
        orbits = slot_orbits(self.template.vertex_count, self.decorated_automorphisms)
        return tuple(len(orbit) for orbit in orbits)

    @property
    def slot_l_values(self):
        return tuple(int(channel.l) for channel in self.channels)

    @property
    def slot_n_values(self):
        return tuple(int(channel.n) for channel in self.channels)

    @property
    def orbit_l_values(self):
        values = []
        for orbit in slot_orbits(self.template.vertex_count, self.decorated_automorphisms):
            first = int(orbit[0])
            values.append(int(self.channels[first].l))
        return tuple(values)

    def to_dict(self):
        return {
            "template": self.template.to_dict(),
            "channels": [ch.to_dict() for ch in self.channels],
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            template=MotifTemplate.from_dict(payload["template"]),
            channels=tuple(PhiSlotChannel.from_dict(item) for item in payload["channels"]),
            name=payload.get("name", None),
        )


@recordclass(('cutoff', 'edge_cutoff', 'channels', 'motif_specs', 'hidden_channels', 'hidden_layers', 'include_rank4', 'motif_family', 'edge_basis_backend', 'periodic_image_mode', 'enforce_unique_periodic_images', 'periodic_image_margin', 'normalize_motif_features'), frozen = True)
class PhiBranchConfig:
    """Configuration for the explicit motif branch."""

    cutoff = 4.0
    edge_cutoff = None
    channels = (PhiSlotChannel(),)
    motif_specs = ()
    hidden_channels = 8
    hidden_layers = 1
    include_rank4 = False
    motif_family = "full"
    edge_basis_backend = "site_basis"
    periodic_image_mode = "unique"
    enforce_unique_periodic_images = True
    periodic_image_margin = 1.0e-8
    normalize_motif_features = True

    def __post_init__(self):
        channels = tuple(ch if isinstance(ch, PhiSlotChannel) else PhiSlotChannel.from_dict(ch) for ch in self.channels)
        if not channels:
            channels = (PhiSlotChannel(),)
        motif_family = str(self.motif_family).strip().lower()
        motif_specs = tuple(
            spec if isinstance(spec, PhiMotifSpec) else PhiMotifSpec.from_dict(spec)
            for spec in self.motif_specs
        )
        if not motif_specs:
            if motif_family in {"star", "star_only", "star-graphs", "star_graphs"}:
                motif_specs = default_star_phi_motif_specs(
                    channels=(channels[0],),
                    include_rank4=bool(self.include_rank4),
                )
            else:
                motif_specs = default_phi_motif_specs(channels=(channels[0],), include_rank4=bool(self.include_rank4))
        edge_cutoff = self.cutoff if self.edge_cutoff is None else float(self.edge_cutoff)
        object.__setattr__(self, "cutoff", float(self.cutoff))
        object.__setattr__(self, "edge_cutoff", edge_cutoff)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "motif_specs", motif_specs)
        object.__setattr__(self, "hidden_channels", int(self.hidden_channels))
        object.__setattr__(self, "hidden_layers", int(self.hidden_layers))
        object.__setattr__(self, "include_rank4", bool(self.include_rank4))
        object.__setattr__(self, "motif_family", motif_family)
        edge_basis_backend = str(self.edge_basis_backend).strip().lower()
        if edge_basis_backend not in {"site_basis", "simple"}:
            raise ValueError("PhiBranchConfig.edge_basis_backend must be 'site_basis' or 'simple'.")
        periodic_image_mode = str(self.periodic_image_mode).strip().lower()
        if periodic_image_mode not in {"unique", "all_images"}:
            raise ValueError("PhiBranchConfig.periodic_image_mode must be 'unique' or 'all_images'.")
        if float(self.periodic_image_margin) < 0.0:
            raise ValueError("PhiBranchConfig.periodic_image_margin must be nonnegative.")
        object.__setattr__(self, "edge_basis_backend", edge_basis_backend)
        object.__setattr__(self, "periodic_image_mode", periodic_image_mode)
        object.__setattr__(self, "enforce_unique_periodic_images", bool(self.enforce_unique_periodic_images))
        object.__setattr__(self, "periodic_image_margin", float(self.periodic_image_margin))
        object.__setattr__(self, "normalize_motif_features", bool(self.normalize_motif_features))

    def to_dict(self):
        return {
            "cutoff": self.cutoff,
            "edge_cutoff": self.edge_cutoff,
            "channels": [ch.to_dict() for ch in self.channels],
            "motif_specs": [spec.to_dict() for spec in self.motif_specs],
            "hidden_channels": self.hidden_channels,
            "hidden_layers": self.hidden_layers,
            "include_rank4": self.include_rank4,
            "motif_family": self.motif_family,
            "edge_basis_backend": self.edge_basis_backend,
            "periodic_image_mode": self.periodic_image_mode,
            "enforce_unique_periodic_images": self.enforce_unique_periodic_images,
            "periodic_image_margin": self.periodic_image_margin,
            "normalize_motif_features": self.normalize_motif_features,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            cutoff=float(payload.get("cutoff", 4.0)),
            edge_cutoff=payload.get("edge_cutoff", None),
            channels=tuple(PhiSlotChannel.from_dict(item) for item in payload.get("channels", ({"n": 1, "l": 0, "m": 0},))),
            motif_specs=tuple(PhiMotifSpec.from_dict(item) for item in payload.get("motif_specs", ())),
            hidden_channels=int(payload.get("hidden_channels", 8)),
            hidden_layers=int(payload.get("hidden_layers", 1)),
            include_rank4=bool(payload.get("include_rank4", False)),
            motif_family=str(payload.get("motif_family", "full")),
            edge_basis_backend=str(payload.get("edge_basis_backend", "site_basis")),
            periodic_image_mode=str(payload.get("periodic_image_mode", "unique")),
            enforce_unique_periodic_images=bool(payload.get("enforce_unique_periodic_images", True)),
            periodic_image_margin=float(payload.get("periodic_image_margin", 1.0e-8)),
            normalize_motif_features=bool(payload.get("normalize_motif_features", True)),
        )


@recordclass(('branches', 'phi', 'dtype'), frozen = True)
class HybridACEPhiConfig:
    """Configuration for a branch-composable small energy model."""

    branches = (BRANCH_BAR_PHI,)
    phi = field(default_factory=PhiBranchConfig)
    dtype = "float64"

    def __post_init__(self):
        object.__setattr__(self, "branches", normalize_branches(self.branches))
        phi = self.phi if isinstance(self.phi, PhiBranchConfig) else PhiBranchConfig.from_dict(self.phi)
        object.__setattr__(self, "phi", phi)
        dtype = str(self.dtype)
        if dtype not in {"float32", "float64"}:
            raise ValueError("HybridACEPhiConfig.dtype must be 'float32' or 'float64'.")
        object.__setattr__(self, "dtype", dtype)

    @property
    def torch_dtype(self):
        return torch.float32 if self.dtype == "float32" else torch.float64

    def to_dict(self):
        return {"branches": list(self.branches), "phi": self.phi.to_dict(), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, payload):
        return cls(
            branches=tuple(payload.get("branches", (BRANCH_BAR_PHI,))),
            phi=PhiBranchConfig.from_dict(payload.get("phi", {})),
            dtype=str(payload.get("dtype", "float64")),
        )


def default_motif_templates(include_rank4=False):
    """Return the default low-order motif templates."""

    templates = [
        MotifTemplate("pair", 2, ()),
        MotifTemplate("path3", 3, ((0, 1), (1, 2))),
        MotifTemplate("triangle", 3, ((0, 1), (1, 2), (0, 2))),
        MotifTemplate("star3", 3, ((0, 1), (0, 2))),
    ]
    if include_rank4:
        templates.append(MotifTemplate("clique4", 4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))))
    return tuple(templates)


def default_star_motif_templates(include_rank4=False):
    """Return the star-only low-order motif templates."""

    templates = [
        MotifTemplate("pair", 2, ()),
        MotifTemplate("star3", 3, ((0, 1), (0, 2))),
    ]
    if include_rank4:
        templates.append(MotifTemplate("star4", 4, ((0, 1), (0, 2), (0, 3))))
    return tuple(templates)


def default_phi_motif_specs(*, channels=(PhiSlotChannel(),), include_rank4=False):
    """Return default motif specs using the first supplied channel in every slot."""

    channel = tuple(channels)[0]
    specs = []
    for template in default_motif_templates(include_rank4=include_rank4):
        specs.append(PhiMotifSpec(template=template, channels=tuple(channel for _ in range(template.vertex_count))))
    return tuple(specs)


def default_star_phi_motif_specs(*, channels=(PhiSlotChannel(),), include_rank4=False):
    """Return the star-only default motif specs using the first supplied channel in every slot."""

    channel = tuple(channels)[0]
    specs = []
    for template in default_star_motif_templates(include_rank4=include_rank4):
        specs.append(PhiMotifSpec(template=template, channels=tuple(channel for _ in range(template.vertex_count))))
    return tuple(specs)


def _channel_label_text(channel):
    text = f"n{channel.n}l{channel.l}m{channel.m}"
    if channel.neighbor_type is not None:
        text += f"t{channel.neighbor_type}"
    return text


def _phi_factor_content(channels, distinguish_slots):
    if bool(distinguish_slots):
        return tuple(range(1, len(tuple(channels)) + 1))
    label_to_id = {}
    content = []
    for channel in tuple(channels):
        label = tuple(channel.repeated_channel_label)
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id) + 1
        content.append(int(label_to_id[label]))
    return tuple(content)


def phi_motif_coupling_report(
    motif_spec,
    *,
    target_partition = None,
    target_L = 0,
    distinguish_slots = False,
):
    """Return YE3T coupling-plan metadata for one explicit Phi motif."""

    spec = motif_spec if isinstance(motif_spec, PhiMotifSpec) else PhiMotifSpec.from_dict(motif_spec)
    rank = int(spec.template.vertex_count)
    if target_partition is None:
        target_partition = (rank,)
    target_partition = tuple(int(part) for part in tuple(target_partition))
    content = _phi_factor_content(spec.channels, distinguish_slots)
    input_Ls = tuple(int(channel.l) for channel in spec.channels)
    target_permutation = "young:" + ",".join(str(int(part)) for part in target_partition)
    slot_orbit_partition = spec.slot_orbit_partition
    decorated_automorphism_group_size = int(len(spec.decorated_automorphisms))
    from ye3t import couplings

    plan = couplings.plan(
        content=content,
        input_Ls=input_Ls,
        target_L=int(target_L),
        target_permutation=target_permutation,
        carrier="Phi",
        carrier_options={
            "slot_count": rank,
            "permuted_slot_count": rank,
            "factor_action": "permute_explicit_phi_tensor_product_factors",
            "distinguish_slots": bool(distinguish_slots),
            "slot_orbit_partition": slot_orbit_partition,
            "decorated_automorphism_group_size": decorated_automorphism_group_size,
            "motif_template": spec.template.to_dict(),
        },
    )
    carrier_validation = dict(plan.validation_report.get("carrier_validation", {}))
    return {
        "carrier": "Phi",
        "motif_name": str(spec.name),
        "rank": int(rank),
        "content": tuple(int(value) for value in content),
        "input_Ls": input_Ls,
        "target_partition": target_partition,
        "target_L": int(target_L),
        "target_permutation": target_permutation,
        "distinguish_slots": bool(distinguish_slots),
        "slot_channels": tuple(channel.to_dict() for channel in spec.channels),
        "slot_l_values": spec.slot_l_values,
        "slot_n_values": spec.slot_n_values,
        "slot_orbit_partition": slot_orbit_partition,
        "decorated_automorphism_group_size": decorated_automorphism_group_size,
        "factor_action": "S_N permutes explicit single-phi tensor-product factors",
        "label_source": "ye3t.couplings.plan",
        "plan": plan.to_dict(),
        "validation_report": {
            "passed": bool(plan.validation_report.get("passed", False)),
            "plan_source": "ye3t.couplings.plan",
            "slot_action_not_lifted_density_specific": True,
            "carrier_validation": carrier_validation,
            "same_factor_permutation_contract_as_A_s_slots": bool(
                carrier_validation.get("same_factor_permutation_contract_as_role_resolved_A_s", False)
            ),
            "labels_owned_by_ye3t": True,
        },
    }


def smooth_cosine_cutoff(distance, cutoff):
    """C1 cosine cutoff with value and first derivative zero at ``cutoff``."""

    from ye3t_ace.equivariant_calc.radial_basis import (
        smooth_cosine_cutoff as shared_smooth_cosine_cutoff,
    )

    return shared_smooth_cosine_cutoff(distance, cutoff)


def _normalize_pbc(pbc, device):
    if pbc is None:
        return torch.zeros(3, dtype=torch.bool, device=device)
    if isinstance(pbc, bool):
        return torch.full((3,), bool(pbc), dtype=torch.bool, device=device)
    values = torch.as_tensor(pbc, dtype=torch.bool, device=device)
    if values.numel() == 1:
        return torch.full((3,), bool(values.item()), dtype=torch.bool, device=device)
    if tuple(values.shape) != (3,):
        raise ValueError("pbc must be a bool or a length-3 boolean sequence.")
    return values


def _minimum_image_displacement(displacement, cell=None, pbc=None):
    if cell is None:
        return displacement
    cell_t = torch.as_tensor(cell, dtype=displacement.dtype, device=displacement.device)
    pbc_t = _normalize_pbc(pbc, displacement.device)
    if not bool(torch.any(pbc_t)):
        return displacement
    if tuple(cell_t.shape) != (3, 3):
        raise ValueError("cell must have shape (3, 3) when periodic geometry is requested.")
    inv_cell = torch.linalg.inv(cell_t)
    frac = displacement @ inv_cell
    shift = torch.zeros_like(frac)
    shift[..., pbc_t] = -torch.round(frac[..., pbc_t])
    return displacement + shift @ cell_t


def _shortest_periodic_lattice_vector_norm(cell, pbc):
    cell_t = torch.as_tensor(cell)
    pbc_t = _normalize_pbc(pbc, cell_t.device)
    if not bool(torch.any(pbc_t)):
        return float("inf")
    axes = [idx for idx, flag in enumerate(pbc_t.detach().cpu().tolist()) if bool(flag)]
    shortest = None
    for coeffs in product((-2, -1, 0, 1, 2), repeat=len(axes)):
        if not any(int(c) != 0 for c in coeffs):
            continue
        shift = torch.zeros(3, dtype=cell_t.dtype, device=cell_t.device)
        for axis, coeff in zip(axes, coeffs, strict=True):
            shift[int(axis)] = float(coeff)
        norm = torch.linalg.norm(shift @ cell_t)
        if shortest is None or bool(norm < shortest):
            shortest = norm
    if shortest is None:
        return float("inf")
    return float(shortest.detach().cpu())


def _unique_periodic_cutoff_margin(cell, pbc, cutoff):
    shortest = _shortest_periodic_lattice_vector_norm(cell, pbc)
    if not np.isfinite(shortest):
        return float("inf")
    return 0.5 * float(shortest) - float(cutoff)


def _directed_edges(positions, cutoff, cell=None, pbc=None):
    atom_count = int(positions.shape[0])
    if atom_count <= 1:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, positions.new_zeros((0, 3)), positions.new_zeros((0,))
    src = []
    dst = []
    for i in range(atom_count):
        for j in range(atom_count):
            if i != j:
                src.append(i)
                dst.append(j)
    src_t = torch.tensor(src, dtype=torch.long, device=positions.device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=positions.device)
    disp = positions.index_select(0, dst_t) - positions.index_select(0, src_t)
    disp = _minimum_image_displacement(disp, cell=cell, pbc=pbc)
    dist = torch.linalg.norm(disp, dim=1)
    mask = dist <= float(cutoff)
    return src_t[mask], dst_t[mask], disp[mask], dist[mask]


def _periodic_shift_tuples(cell, pbc, cutoff):
    pbc_t = _normalize_pbc(pbc, torch.device("cpu"))
    if not bool(torch.any(pbc_t)):
        return ((0.0, 0.0, 0.0),)
    shortest = _shortest_periodic_lattice_vector_norm(torch.as_tensor(cell, dtype=torch.float64), pbc_t)
    if not np.isfinite(shortest) or shortest <= 0.0:
        raise ValueError("Cannot enumerate periodic images for a singular or invalid periodic cell.")
    radius = max(1, int(np.ceil(float(cutoff) / float(shortest))) + 1)
    axes = []
    for flag in pbc_t.detach().cpu().tolist():
        axes.append(range(-radius, radius + 1) if bool(flag) else (0,))
    return tuple(tuple(float(x) for x in shift) for shift in product(*axes))


def _directed_edges_all_images(positions, cutoff, cell=None, pbc=None):
    if cell is None or not bool(torch.any(_normalize_pbc(pbc, positions.device))):
        return _directed_edges(positions, cutoff, cell=cell, pbc=pbc)
    cell_t = torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
    shifts = _periodic_shift_tuples(cell_t.detach().cpu(), pbc, cutoff)
    src = []
    dst = []
    shift_rows = []
    atom_count = int(positions.shape[0])
    for i in range(atom_count):
        for j in range(atom_count):
            if i == j:
                continue
            for shift in shifts:
                src.append(i)
                dst.append(j)
                shift_rows.append(shift)
    if not src:
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return empty, empty, positions.new_zeros((0, 3)), positions.new_zeros((0,))
    src_t = torch.tensor(src, dtype=torch.long, device=positions.device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=positions.device)
    shifts_t = torch.tensor(shift_rows, dtype=positions.dtype, device=positions.device)
    disp = positions.index_select(0, dst_t) - positions.index_select(0, src_t) + shifts_t @ cell_t
    dist = torch.linalg.norm(disp, dim=1)
    mask = dist <= float(cutoff)
    return src_t[mask], dst_t[mask], disp[mask], dist[mask]


def _simple_edge_channel_values_from_edges(positions, atom_types, channels, cutoff, src, dst, disp, dist):
    values = positions.new_zeros((int(src.numel()), len(channels)))
    if src.numel() == 0 or not channels:
        return values
    safe_dist = torch.clamp(dist, min=torch.finfo(positions.dtype).eps)
    unit = disp / safe_dist.unsqueeze(-1)
    cutoff_values = smooth_cosine_cutoff(dist, cutoff)
    radial_base = torch.clamp(dist / float(cutoff), min=0.0)
    spherical_cache = {}
    for col, channel in enumerate(channels):
        radial = cutoff_values * radial_base.pow(int(channel.n))
        if channel.l == 0:
            angular = torch.ones_like(radial)
        else:
            if channel.l not in spherical_cache:
                spherical_cache[channel.l] = _real_spherical_harmonics_l_from_unit_cartesian(channel.l, unit)
            angular = spherical_cache[channel.l][channel.m + channel.l]
        typed = torch.ones_like(radial)
        if channel.neighbor_type is not None:
            typed = (atom_types.index_select(0, dst) == int(channel.neighbor_type)).to(dtype=positions.dtype)
        values[:, col] = radial * angular * typed
    return values


def edge_channel_values(positions, atom_types, channels, cutoff, *, cell=None, pbc=None):
    """Evaluate scalar one-neighbor edge channels before density scatter."""

    positions = torch.as_tensor(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3).")
    channels = tuple(channels)
    atom_count = int(positions.shape[0])
    if atom_types is None:
        atom_types = torch.zeros(atom_count, dtype=torch.long, device=positions.device)
    else:
        atom_types = torch.as_tensor(atom_types, dtype=torch.long, device=positions.device)
    src, dst, disp, dist = _directed_edges(positions, cutoff, cell=cell, pbc=pbc)
    values = _simple_edge_channel_values_from_edges(positions, atom_types, channels, cutoff, src, dst, disp, dist)
    return src, dst, disp, dist, values


def _voigt_from_stress_tensor(stress_tensor):
    if _ase_full_3x3_to_voigt_6_stress is not None:
        return _ase_full_3x3_to_voigt_6_stress(stress_tensor)
    stress_tensor = np.asarray(stress_tensor, dtype=float)
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


def _edge_lookup(atom_count, src, dst):
    lookup = {}
    for idx, (i, j) in enumerate(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist(), strict=True)):
        lookup[(int(i), int(j))] = int(idx)
    return lookup


def _embedding_rows_for_center(center, neighbors, order):
    if len(neighbors) < order:
        return ()
    return tuple(tuple(int(x) for x in row) for row in permutations(neighbors, int(order)))


def _embedding_edge_rows_for_center(edge_indices, edge_dst, order):
    if len(edge_indices) < order:
        return ()
    rows = []
    for row in permutations(edge_indices, int(order)):
        atoms = [int(edge_dst[int(edge_idx)]) for edge_idx in row]
        if len(set(atoms)) == int(order):
            rows.append(tuple(int(edge_idx) for edge_idx in row))
    return tuple(rows)


def _motif_weight(positions, template, center, row, cutoff, edge_cutoff, *, cell=None, pbc=None):
    weight = positions.new_ones(())
    for neighbor in row:
        disp = positions[int(neighbor)] - positions[int(center)]
        d = torch.linalg.norm(_minimum_image_displacement(disp, cell=cell, pbc=pbc))
        weight = weight * smooth_cosine_cutoff(d, cutoff)
    for a, b in template.edges:
        disp = positions[int(row[a])] - positions[int(row[b])]
        d = torch.linalg.norm(_minimum_image_displacement(disp, cell=cell, pbc=pbc))
        weight = weight * smooth_cosine_cutoff(d, edge_cutoff)
    return weight


def _motif_weight_from_edge_displacements(edge_disp, template, row, cutoff, edge_cutoff):
    weight = edge_disp.new_ones(())
    row = tuple(int(x) for x in row)
    for edge_idx in row:
        d = torch.linalg.norm(edge_disp[int(edge_idx)])
        weight = weight * smooth_cosine_cutoff(d, cutoff)
    for a, b in template.edges:
        d = torch.linalg.norm(edge_disp[row[int(b)]] - edge_disp[row[int(a)]])
        weight = weight * smooth_cosine_cutoff(d, edge_cutoff)
    return weight


def _motif_weights_from_edge_rows(edge_disp, template, row_edges, cutoff, edge_cutoff):
    if row_edges.numel() == 0:
        return edge_disp.new_zeros((0,))
    row_edges = row_edges.to(dtype=torch.long, device=edge_disp.device)
    slot_disp = edge_disp.index_select(0, row_edges.reshape(-1)).reshape(
        int(row_edges.shape[0]),
        int(row_edges.shape[1]),
        3,
    )
    slot_distance = torch.linalg.norm(slot_disp, dim=2)
    weight = smooth_cosine_cutoff(slot_distance, cutoff).prod(dim=1)
    for a, b in template.edges:
        pair_distance = torch.linalg.norm(slot_disp[:, int(b), :] - slot_disp[:, int(a), :], dim=1)
        weight = weight * smooth_cosine_cutoff(pair_distance, edge_cutoff)
    return weight


def _spec_slot_channel_indices(spec, channel_to_index):
    indices = []
    for channel in spec.channels:
        try:
            indices.append(int(channel_to_index[channel]))
        except KeyError as exc:
            raise ValueError(f"Motif spec {spec.name!r} uses a channel not present in PhiBranchConfig.channels.") from exc
    return tuple(indices)


class HybridACEPhiEnergyModel(torch.nn.Module):
    """Differentiable fixed-feature energy model with A and barPhi branches."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, HybridACEPhiConfig) else HybridACEPhiConfig.from_dict(config or {})
        dtype = self.config.torch_dtype
        phi = self.config.phi
        self.branches = self.config.branches
        self.channel_to_index = {channel: index for index, channel in enumerate(phi.channels)}
        self.register_parameter("a_weight", None)
        self.register_parameter("a_bias", None)
        if BRANCH_BAR_PHI in self.branches:
            self.bar_phi_weight = torch.nn.Parameter(torch.zeros(len(phi.motif_specs), dtype=dtype))
            self.bar_phi_bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        else:
            self.register_parameter("bar_phi_weight", None)
            self.register_parameter("bar_phi_bias", None)
        self._last_profile = {}
        self._site_basis_cache = {}

    def profile_report(self):
        return dict(self._last_profile)

    def _periodic_cutoff_margin(self, cell, pbc):
        if cell is None:
            return float("inf")
        if self.config.phi.periodic_image_mode == "all_images":
            return float("inf")
        pbc_t = _normalize_pbc(pbc, cell.device if torch.is_tensor(cell) else torch.device("cpu"))
        if not bool(torch.any(pbc_t)):
            return float("inf")
        phi = self.config.phi
        return _unique_periodic_cutoff_margin(cell, pbc_t, max(float(phi.cutoff), float(phi.edge_cutoff)))

    def _validate_periodic_cutoff_margin(self, cell, pbc):
        phi = self.config.phi
        margin = self._periodic_cutoff_margin(cell, pbc)
        if (
            bool(phi.enforce_unique_periodic_images)
            and np.isfinite(margin)
            and margin < float(phi.periodic_image_margin)
        ):
            raise ValueError(
                "Periodic Phi motif rastering requires a unique minimum image for all active distances. "
                f"Got half-shortest-lattice cutoff margin {margin:.6g}, below configured margin "
                f"{phi.periodic_image_margin:.6g}. Reduce cutoff/edge_cutoff or enlarge the periodic cell."
            )
        return margin

    def _site_basis_for_types(self, possible_types, device):
        possible_types = tuple(sorted(int(x) for x in possible_types))
        dtype = self.config.torch_dtype
        key = (possible_types, str(dtype), str(device), float(self.config.phi.cutoff))
        cached = self._site_basis_cache.get(key)
        if cached is not None:
            return cached
        max_n = max((int(ch.n) for ch in self.config.phi.channels), default=1)
        max_l = max((int(ch.l) for ch in self.config.phi.channels), default=0)
        cfg = SiteBasisConfig(
            rc=[float(self.config.phi.cutoff)],
            lmbda=[0.25],
            nradmax=max(1, max_n),
            lmax=max_l,
            possible_types=possible_types,
            charge_mode="none",
            atomic_base_normalization="none",
            factor_normalization="bounded",
            spherical_backend="real",
            dtype=dtype,
            complex_dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
        )
        basis = SiteBasisV2(cfg).to(device=device)
        self._site_basis_cache[key] = basis
        return basis

    def _site_basis_edge_values(self, atom_types, src, dst, disp):
        channels = self.config.phi.channels
        values = disp.new_zeros((int(src.numel()), len(channels)))
        if src.numel() == 0 or not channels:
            return values
        type_values = set(int(x) for x in atom_types.detach().cpu().tolist())
        for channel in channels:
            if channel.neighbor_type is not None:
                type_values.add(int(channel.neighbor_type))
        possible_types = tuple(sorted(type_values or {0}))
        basis = self._site_basis_for_types(possible_types, disp.device)
        site_channels = []
        column_slices = []
        for channel in channels:
            start = len(site_channels)
            neighbor_types = possible_types if channel.neighbor_type is None else (int(channel.neighbor_type),)
            for mu0 in possible_types:
                for mu in neighbor_types:
                    site_channels.append(
                        SingleChannelLabel(
                            mu0=int(mu0),
                            mu=int(mu),
                            kappa0=0,
                            kappa=0,
                            n=int(channel.n),
                            l=int(channel.l),
                            m=int(channel.m),
                        )
                    )
            column_slices.append(slice(start, len(site_channels)))
        edge_index = torch.stack((src, dst), dim=0)
        _labels, edge_values, _edge_dx = basis.compute_channel_edges_with_dx(
            disp,
            edge_index,
            atom_types,
            site_channels,
        )
        edge_values = edge_values.real.to(dtype=disp.dtype)
        for column, column_slice in enumerate(column_slices):
            values[:, column] = edge_values[:, column_slice].sum(dim=1)
        return values

    def _edge_data(self, positions, atom_types, *, cell=None, pbc=None):
        phi = self.config.phi
        atom_count = int(positions.shape[0])
        if atom_types is None:
            atom_types = torch.zeros(atom_count, dtype=torch.long, device=positions.device)
        else:
            atom_types = torch.as_tensor(atom_types, dtype=torch.long, device=positions.device)
        if phi.periodic_image_mode == "all_images":
            src, dst, disp, dist = _directed_edges_all_images(positions, phi.cutoff, cell=cell, pbc=pbc)
        else:
            src, dst, disp, dist = _directed_edges(positions, phi.cutoff, cell=cell, pbc=pbc)
        if phi.edge_basis_backend == "site_basis":
            values = self._site_basis_edge_values(atom_types, src, dst, disp)
        else:
            values = _simple_edge_channel_values_from_edges(positions, atom_types, phi.channels, phi.cutoff, src, dst, disp, dist)
        return src, dst, disp, dist, values

    def _motif_values(self, positions, src, dst, edge_disp, edge_values, *, cell=None, pbc=None):
        phi = self.config.phi
        atom_count = int(positions.shape[0])
        edge_indices_by_center = [[] for _ in range(atom_count)]
        for edge_idx, i in enumerate(src.detach().cpu().tolist()):
            edge_indices_by_center[int(i)].append(int(edge_idx))
        bar_features = positions.new_zeros((atom_count, len(phi.motif_specs)))
        bar_denominator = positions.new_zeros((atom_count, len(phi.motif_specs)))
        embedding_count = 0
        weight_sum = positions.new_zeros(())
        for spec_index, spec in enumerate(phi.motif_specs):
            channel_indices = _spec_slot_channel_indices(spec, self.channel_to_index)
            centers = []
            rows_for_spec = []
            for center in range(atom_count):
                rows = _embedding_edge_rows_for_center(edge_indices_by_center[center], dst, spec.template.vertex_count)
                for row in rows:
                    centers.append(center)
                    rows_for_spec.append(row)
            if rows_for_spec:
                centers_t = torch.tensor(centers, dtype=torch.long, device=positions.device)
                row_edges_t = torch.tensor(rows_for_spec, dtype=torch.long, device=positions.device)
                channel_indices_t = torch.tensor(channel_indices, dtype=torch.long, device=positions.device)
                gathered = edge_values.index_select(0, row_edges_t.reshape(-1)).reshape(
                    int(row_edges_t.shape[0]),
                    int(row_edges_t.shape[1]),
                    int(edge_values.shape[1]),
                )
                slot_values_t = gathered.gather(
                    2,
                    channel_indices_t.reshape(1, -1, 1).expand(int(row_edges_t.shape[0]), -1, 1),
                ).squeeze(-1)
                weights_t = _motif_weights_from_edge_rows(edge_disp, spec.template, row_edges_t, phi.cutoff, phi.edge_cutoff)
                embedding_count += int(row_edges_t.shape[0])
                weight_sum = weight_sum + weights_t.detach().sum()
                product_t = slot_values_t.prod(dim=1)
                bar_col = positions.new_zeros(atom_count)
                denom_col = positions.new_zeros(atom_count)
                bar_col.index_add_(0, centers_t, product_t * weights_t / float(spec.normalization))
                denom_col.index_add_(0, centers_t, weights_t / float(spec.normalization))
                bar_features[:, spec_index] = bar_col
                bar_denominator[:, spec_index] = denom_col
        if bool(phi.normalize_motif_features):
            bar_features = torch.where(
                bar_denominator > 0.0,
                bar_features / torch.clamp(bar_denominator, min=torch.finfo(bar_features.dtype).eps),
                torch.zeros_like(bar_features),
            )
        return bar_features, (), embedding_count, weight_sum

    def forward(self, positions, atom_types=None, *, cell=None, pbc=None, active_branches=None, return_site_energies=False):
        positions = torch.as_tensor(positions, dtype=self.config.torch_dtype)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3).")
        atom_count = int(positions.shape[0])
        branches = self.branches if active_branches is None else normalize_branches(active_branches)
        if BRANCH_A in branches:
            raise ValueError(
                "The A branch requires the existing exact LinearACEScalarCalculator path. "
                "Pass ace_bundle to HybridACEPhiCalculator, or call the tensor model with active_branches "
                "excluding 'A'."
            )
        start = time.perf_counter()
        cell_t = None if cell is None else torch.as_tensor(cell, dtype=positions.dtype, device=positions.device)
        pbc_t = _normalize_pbc(pbc, positions.device)
        periodic_cutoff_margin = self._validate_periodic_cutoff_margin(cell_t, pbc_t)
        src, dst, edge_disp, _dist, edge_values = self._edge_data(positions, atom_types, cell=cell_t, pbc=pbc_t)
        edge_seconds = time.perf_counter() - start
        site_energy = positions.new_zeros(atom_count)
        motif_start = time.perf_counter()
        needs_motifs = BRANCH_BAR_PHI in branches
        embedding_count = 0
        weight_sum = positions.new_zeros(())
        if needs_motifs:
            bar_features, _unused, embedding_count, weight_sum = self._motif_values(
                positions,
                src,
                dst,
                edge_disp,
                edge_values,
                cell=cell_t,
                pbc=pbc_t,
            )
        else:
            bar_features = positions.new_zeros((atom_count, len(self.config.phi.motif_specs)))
        motif_seconds = time.perf_counter() - motif_start
        if BRANCH_BAR_PHI in branches:
            site_energy = (
                site_energy
                + bar_features @ self.bar_phi_weight.to(dtype=positions.dtype)
                + self.bar_phi_bias.to(dtype=positions.dtype)[0]
            )
        self._last_profile = {
            "branch_count": int(len(branches)),
            "branches": tuple(branches),
            "periodic_axes": tuple(bool(x) for x in pbc_t.detach().cpu().tolist()),
            "periodic_unique_image_cutoff_margin": float(periodic_cutoff_margin),
            "edge_basis_backend": str(self.config.phi.edge_basis_backend),
            "atom_count": atom_count,
            "edge_count": int(src.numel()),
            "motif_embedding_count": int(embedding_count),
            "soft_motif_weight_sum": float(weight_sum.detach().cpu()) if weight_sum.numel() else 0.0,
            "edge_channel_basis_seconds": float(edge_seconds),
            "motif_enumeration_and_products_seconds": float(motif_seconds),
        }
        return site_energy if return_site_energies else site_energy.sum()

    def energy_forces_cell_gradient(
        self,
        positions,
        atom_types=None,
        *,
        cell,
        pbc=None,
        active_branches=None,
        fixed_scaled_positions=True,
    ):
        """Return energy, Cartesian forces, and ``dE/dcell`` for Phi branches.

        With ``fixed_scaled_positions=True`` the input Cartesian positions are
        converted to fractional coordinates using the supplied cell, then the
        energy is differentiated with respect to a new cell tensor while those
        fractional coordinates are held fixed.  The result is an explicit cell
        gradient; the ASE calculator maps this to Phi-only stress using ASE's
        numerical-stress convention.
        """

        cell0 = torch.as_tensor(cell, dtype=self.config.torch_dtype)
        if tuple(cell0.shape) != (3, 3):
            raise ValueError("cell must have shape (3, 3).")
        pos0 = torch.as_tensor(positions, dtype=self.config.torch_dtype, device=cell0.device)
        if pos0.ndim != 2 or pos0.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3).")
        branches = self.branches if active_branches is None else normalize_branches(active_branches)
        if BRANCH_A in branches:
            raise ValueError("Cell gradients for the A branch are not exposed through HybridACEPhiEnergyModel.")
        cell_req = cell0.detach().clone().requires_grad_(True)
        if fixed_scaled_positions:
            scaled = (pos0.detach() @ torch.linalg.inv(cell0.detach())).to(dtype=cell_req.dtype, device=cell_req.device)
            pos_req = scaled @ cell_req
        else:
            pos_req = pos0.detach().clone().requires_grad_(True)
        energy = self(
            pos_req,
            atom_types,
            cell=cell_req,
            pbc=pbc,
            active_branches=branches,
        )
        grad_pos, grad_cell = torch.autograd.grad(energy, (pos_req, cell_req), create_graph=False, retain_graph=False)
        return energy.detach(), (-grad_pos).detach(), grad_cell.detach()

    def config_dict(self):
        return self.config.to_dict()


class HybridACEPhiCalculator(_ASECalculatorBase):
    """ASE calculator for ``HybridACEPhiEnergyModel`` instances."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model,
        type_map=None,
        device=None,
        *,
        ace_bundle=None,
        ace_cutoff=None,
        ace_type_map=None,
        ace_calculator_kwargs=None,
        **kwargs,
    ):
        if _ASE_IMPORT_ERROR is not None:
            raise ImportError("ASE is required for HybridACEPhiCalculator.") from _ASE_IMPORT_ERROR
        super().__init__(**kwargs)
        self.model = model
        self.type_map = {} if type_map is None else {str(k): int(v) for k, v in dict(type_map).items()}
        self.device = torch.device("cpu" if device is None else device)
        self.model.to(self.device)
        self._last_profile = {}
        self.ace_calculator = None
        if ace_bundle is not None:
            if ace_cutoff is None:
                raise ValueError("ace_cutoff is required when ace_bundle is supplied.")
            try:
                from ye3t_ace.ace.linear_ace import LinearACEScalarCalculator
            except Exception:  # pragma: no cover - local import fallback
                from .ace.linear_ace import LinearACEScalarCalculator
            ace_kwargs = {} if ace_calculator_kwargs is None else dict(ace_calculator_kwargs)
            self.ace_calculator = LinearACEScalarCalculator(
                ace_bundle,
                cutoff=float(ace_cutoff),
                type_map=dict(self.type_map if ace_type_map is None else ace_type_map),
                device=self.device,
                **ace_kwargs,
            )
        elif BRANCH_A in self.model.branches:
            raise ValueError(
                "HybridACEPhiCalculator requires ace_bundle and ace_cutoff when the model includes the A branch. "
                "The lightweight A-like fallback has been removed."
            )

    def _atom_types(self, atoms, device):
        if self.type_map:
            return torch.tensor([self.type_map[symbol] for symbol in atoms.get_chemical_symbols()], dtype=torch.long, device=device)
        return torch.zeros(len(atoms), dtype=torch.long, device=device)

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        if system_changes is None:
            system_changes = _ASE_ALL_CHANGES
        _ASECalculatorBase.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("HybridACEPhiCalculator requires atoms.")
        start = time.perf_counter()
        energy_offset = 0.0
        force_offset = None
        stress_offset = None
        active_branches = self.model.branches
        a_branch_source = "not_requested"
        stress_requested = "stress" in tuple(properties)
        if self.ace_calculator is not None and BRANCH_A in active_branches:
            ace_properties = ("energy", "forces", "stress") if stress_requested else ("energy", "forces")
            self.ace_calculator.calculate(atoms, properties=ace_properties, system_changes=system_changes)
            energy_offset = float(self.ace_calculator.results["energy"])
            force_offset = np.asarray(self.ace_calculator.results["forces"], dtype=float)
            if stress_requested:
                stress_offset = np.asarray(self.ace_calculator.results["stress"], dtype=float)
            active_branches = tuple(branch for branch in active_branches if branch != BRANCH_A)
            a_branch_source = "linear_ace_exact_basis"
        pos = torch.tensor(atoms.positions, dtype=self.model.config.torch_dtype, device=self.device, requires_grad=True)
        atom_types = self._atom_types(atoms, self.device)
        cell = torch.tensor(np.asarray(atoms.cell.array, float), dtype=self.model.config.torch_dtype, device=self.device)
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if active_branches and stress_requested:
            energy_t, forces_t, cell_grad_t = self.model.energy_forces_cell_gradient(
                pos,
                atom_types,
                cell=cell,
                pbc=pbc,
                active_branches=active_branches,
                fixed_scaled_positions=True,
            )
            forces = forces_t.detach().cpu().numpy()
            phi_energy = float(energy_t.detach().cpu())
            cell_grad = cell_grad_t.detach().cpu().numpy()
            cell_np = np.asarray(atoms.cell.array, float)
            volume = float(atoms.get_volume())
            dE_dstrain = cell_np.T @ cell_grad
            stress_tensor = 0.5 * (dE_dstrain + dE_dstrain.T) / volume
            self.results["stress"] = _voigt_from_stress_tensor(stress_tensor)
        elif stress_requested:
            self.results["stress"] = np.zeros(6, dtype=float)
        elif active_branches:
            energy = self.model(pos, atom_types, cell=cell, pbc=pbc, active_branches=active_branches)
            grad = torch.autograd.grad(energy, pos, create_graph=False, retain_graph=False)[0]
            forces = -grad.detach().cpu().numpy()
            phi_energy = float(energy.detach().cpu())
        else:
            forces = np.zeros((len(atoms), 3), dtype=float)
            phi_energy = 0.0
        if force_offset is not None:
            forces = forces + force_offset
        self.results["energy"] = float(energy_offset + phi_energy)
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = forces
        if stress_requested and stress_offset is not None:
            self.results["stress"] = np.asarray(self.results["stress"], dtype=float) + stress_offset
        self._last_profile = dict(self.model.profile_report())
        self._last_profile["ase_calculator_step_seconds"] = float(time.perf_counter() - start)
        self._last_profile["a_branch_source"] = str(a_branch_source)
        if self.ace_calculator is not None:
            self._last_profile["linear_ace_backend_report"] = dict(self.ace_calculator.backend_report())

    def profile_report(self):
        return dict(self._last_profile)

    def energy_forces_cell_gradient(self, atoms=None, *, fixed_scaled_positions=True):
        """Evaluate hybrid energy, forces, and explicit ``dE/dcell`` for ASE atoms."""

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("HybridACEPhiCalculator requires atoms.")
        active_branches = self.model.branches
        energy_total = 0.0
        forces_total = np.zeros((len(atoms), 3), dtype=float)
        cell_grad_total = np.zeros((3, 3), dtype=float)
        if BRANCH_A in active_branches:
            if self.ace_calculator is None:
                raise ValueError("A-branch cell gradients require an exact ACE calculator.")
            ace_energy, ace_forces, ace_cell_grad, _site_energy = self.ace_calculator.energy_forces_cell_gradient(
                atoms,
                fixed_scaled_positions=fixed_scaled_positions,
            )
            energy_total += float(ace_energy.detach().cpu())
            forces_total += ace_forces.detach().cpu().numpy()
            cell_grad_total += ace_cell_grad.detach().cpu().numpy()
            active_branches = tuple(branch for branch in active_branches if branch != BRANCH_A)
        if active_branches:
            positions = torch.tensor(np.asarray(atoms.positions, float), dtype=self.model.config.torch_dtype, device=self.device)
            atom_types = self._atom_types(atoms, self.device)
            cell = torch.tensor(np.asarray(atoms.cell.array, float), dtype=self.model.config.torch_dtype, device=self.device)
            pbc = np.asarray(atoms.pbc, dtype=bool)
            energy, forces, cell_grad = self.model.energy_forces_cell_gradient(
                positions,
                atom_types,
                cell=cell,
                pbc=pbc,
                active_branches=active_branches,
                fixed_scaled_positions=fixed_scaled_positions,
            )
            energy_total += float(energy.detach().cpu())
            forces_total += forces.detach().cpu().numpy()
            cell_grad_total += cell_grad.detach().cpu().numpy()
        return float(energy_total), forces_total, cell_grad_total

def save_hybrid_ace_phi_ase_bundle(path, model, type_map=None, *, ace_bundle=None, ace_cutoff=None, ace_type_map=None):
    """Save a model/config bundle that can be restored as an ASE calculator."""

    if BRANCH_A in model.branches and ace_bundle is None:
        raise ValueError("A-branch hybrid ASE bundles require ace_bundle metadata for restoration.")
    if ace_bundle is not None and ace_cutoff is None:
        raise ValueError("ace_cutoff is required when ace_bundle is supplied.")
    payload = {
        "format": "hybrid_ace_phi_ase_bundle",
        "version": 1,
        "config": model.config_dict(),
        "state_dict": model.state_dict(),
        "type_map": {} if type_map is None else {str(k): int(v) for k, v in dict(type_map).items()},
        "ace_bundle": ace_bundle,
        "ace_cutoff": None if ace_cutoff is None else float(ace_cutoff),
        "ace_type_map": None if ace_type_map is None else {str(k): int(v) for k, v in dict(ace_type_map).items()},
        "fit_metadata": dict(getattr(model, "fit_metadata", {})),
    }
    torch.save(payload, Path(path))
    return Path(path)


def load_hybrid_ace_phi_ase_bundle(path, *, map_location="cpu"):
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format") != "hybrid_ace_phi_ase_bundle":
        raise ValueError("Unsupported hybrid ACE/Phi bundle format.")
    model = HybridACEPhiEnergyModel(HybridACEPhiConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.fit_metadata = dict(payload.get("fit_metadata", {}))
    model._hybrid_ace_phi_bundle_metadata = {
        "ace_bundle": payload.get("ace_bundle", None),
        "ace_cutoff": payload.get("ace_cutoff", None),
        "ace_type_map": payload.get("ace_type_map", None),
    }
    return model, dict(payload.get("type_map", {}))


def load_hybrid_ace_phi_calculator(path, **kwargs):
    model, type_map = load_hybrid_ace_phi_ase_bundle(path, map_location=kwargs.pop("map_location", "cpu"))
    metadata = dict(getattr(model, "_hybrid_ace_phi_bundle_metadata", {}))
    if metadata.get("ace_bundle", None) is not None:
        kwargs.setdefault("ace_bundle", metadata["ace_bundle"])
        kwargs.setdefault("ace_cutoff", metadata["ace_cutoff"])
        kwargs.setdefault("ace_type_map", metadata.get("ace_type_map", None))
    return HybridACEPhiCalculator(model, type_map=type_map, **kwargs)


__all__ = [
    "ALL_BRANCHES",
    "BRANCH_A",
    "BRANCH_BAR_PHI",
    "HybridACEPhiCalculator",
    "HybridACEPhiConfig",
    "HybridACEPhiEnergyModel",
    "MotifTemplate",
    "PhiBranchConfig",
    "PhiMotifSpec",
    "PhiSlotChannel",
    "default_motif_templates",
    "default_phi_motif_specs",
    "default_star_motif_templates",
    "default_star_phi_motif_specs",
    "edge_channel_values",
    "load_hybrid_ace_phi_ase_bundle",
    "load_hybrid_ace_phi_calculator",
    "normalize_branches",
    "phi_motif_coupling_report",
    "save_hybrid_ace_phi_ase_bundle",
    "smooth_cosine_cutoff",
]
