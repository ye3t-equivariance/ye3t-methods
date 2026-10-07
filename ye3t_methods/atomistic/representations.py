"""Representation selectors for YE3T descriptor and model construction.

This module owns the public representation-selection vocabulary. Descriptor
factories consume these selections; they should not define the mathematical
meaning of construction modes such as ``young_subgroup_exact`` or
``young_specht_exact``.
"""

from ye3t_methods.atomistic._record import recordclass
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import field
from pathlib import Path

from ye3t import YE3TReadoutSpec, YE3TRotationTarget, YE3TSpec, validate_fast_path_policy


def normalize_a_s_subselection(value=None):
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "symmetric": "fully_symmetric",
        "fully_symmetric": "fully_symmetric",
        "trivial": "fully_symmetric",
        "trivial_symmetric": "fully_symmetric",
        "antisymmetric": "fully_antisymmetric",
        "fully_antisymmetric": "fully_antisymmetric",
        "sign": "fully_antisymmetric",
        "sign_sector": "fully_antisymmetric",
        "equivariant": "equivariant",
        "standard": "equivariant",
        "standard_equivariant": "equivariant",
        "trivial_standard": "equivariant",
        "mixed": "equivariant",
    }
    if normalized not in aliases:
        raise ValueError(
            "A_s representation_subselection must be 'fully_symmetric', "
            "'fully_antisymmetric', or 'equivariant'."
        )
    return aliases[normalized]


def _default_block_mu_labels_from_content(content):
    counts = Counter(tuple(content))
    seen = set()
    out = []
    for label in tuple(content):
        if label in seen:
            continue
        seen.add(label)
        out.append((int(counts[label]),))
    return tuple(out)


def _slot_specht_partitions_tuple(value):
    if value is None:
        return tuple()
    if isinstance(value, (str, bytes)):
        return tuple()
    return tuple(tuple(int(part) for part in partition) for partition in value)


def _A_s_role_policy_report_from_metadata(slot_specht_partitions, metadata):
    metadata = {} if metadata is None else dict(metadata)
    carrier_options = {
        key: metadata[key]
        for key in (
            "role_coordinate_policy",
            "identical_role_filters_declared",
            "role_filters_identical",
            "identical_role_filters",
            "role_coordinate_discarded_before_young_projection",
            "discard_role_coordinate",
        )
        if key in metadata
    }
    partitions = _slot_specht_partitions_tuple(slot_specht_partitions)
    if partitions:
        carrier_options["slot_specht_partitions"] = partitions
    target_permutation = "trivial"
    nontrivial = tuple(partition for partition in partitions if len(partition) > 1)
    if nontrivial:
        target_permutation = "young:" + ",".join(str(part) for part in nontrivial[0])
    elif isinstance(slot_specht_partitions, (str, bytes)):
        selector = str(slot_specht_partitions).strip().lower()
        if selector in {"nontrivial", "non_trivial", "standard", "mixed", "all_nontrivial"}:
            target_permutation = "young:nontrivial"
    return YE3TSpec(
        carrier="A_s",
        target_permutation=target_permutation,
        carrier_options=carrier_options,
        runtime_status="planned_not_public",
    ).carrier_policy_report()


def _load_representation_config_file(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without optional yaml.
            raise ImportError("Reading YAML representation configs requires PyYAML.") from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("YE3T representation config files must contain a mapping.")
    return dict(payload)


@recordclass(('basis_family', 'permutation_sector', 'construction_mode', 'basis_mode', 'fast_path_policy', 'permutation_group', 'coupling_tree', 'coefficient_backend', 'validation_status', 'metadata'), frozen = True)
class YE3TRepresentation:
    """Representation selector for descriptor and model construction.

    Reference boundary:
    - ACE trivial-sector descriptor terminology follows Drautz, Phys. Rev. B
      99, 014104 (2019), doi:10.1103/PhysRevB.99.014104, and Phys. Rev. B
      102, 024104 (2020), doi:10.1103/PhysRevB.102.024104.
    - Nontrivial Young/Specht-sector selectors record the representation data
      needed for later certified-intertwiner and projector validation. The
      validation targets follow finite-group and Schur-Weyl representation
      theory references tracked in docs and references_to_add.md.
    """

    basis_family = "ace"
    permutation_sector = "trivial"
    construction_mode = "trivial_image_fast_path"
    basis_mode = None
    fast_path_policy = "auto"
    permutation_group = None
    coupling_tree = "balanced"
    coefficient_backend = "certified_intertwiner"
    validation_status = "not_validated"
    metadata = field(default_factory=dict)

    def __post_init__(self):
        family = str(self.basis_family)
        if family not in {"ace", "ye3t"}:
            raise ValueError("basis_family must be 'ace' or 'ye3t'.")
        mode = str(self.construction_mode)
        if mode not in {
            "trivial_image_fast_path",
            "orbital_basis",
            "schur_weyl_exact",
            "young_subgroup_exact",
            "young_specht_exact",
            "lifted_cauchy_exact",
            "tagged_cauchy_exact",
            "antisymmetric_fast_path",
            "experimental",
        }:
            raise ValueError(
                "construction_mode must be 'trivial_image_fast_path', 'orbital_basis', "
                "'schur_weyl_exact', 'young_subgroup_exact', 'young_specht_exact', "
                "'lifted_cauchy_exact', 'tagged_cauchy_exact', "
                "'antisymmetric_fast_path', or 'experimental'."
            )
        validate_fast_path_policy(str(self.fast_path_policy))
        tree = str(self.coupling_tree)
        if tree not in {"balanced", "left", "right", "explicit"}:
            raise ValueError("coupling_tree must be 'balanced', 'left', 'right', or 'explicit'.")
        if isinstance(self.permutation_sector, str):
            allowed = {
                "trivial",
                "antisymmetric",
                "orbital",
                "young_subgroup_trivial",
                "young_specht",
                "full_irrep_decomposition",
            }
            if self.permutation_sector not in allowed and not self.permutation_sector.startswith("young:"):
                raise ValueError(f"Unsupported permutation_sector={self.permutation_sector!r}.")
        if mode == "trivial_image_fast_path" and self.permutation_sector != "trivial":
            raise ValueError("construction_mode='trivial_image_fast_path' requires permutation_sector='trivial'.")
        if mode == "antisymmetric_fast_path" and self.permutation_sector != "antisymmetric":
            raise ValueError("construction_mode='antisymmetric_fast_path' requires permutation_sector='antisymmetric'.")
        if mode == "young_subgroup_exact" and self.permutation_sector not in {"trivial", "young_subgroup_trivial"}:
            raise ValueError(
                "construction_mode='young_subgroup_exact' currently supports only the "
                "Young-subgroup trivial/invariant sector."
            )
        if mode == "young_specht_exact" and self.permutation_sector not in {"young_specht", "full_irrep_decomposition"}:
            raise ValueError(
                "construction_mode='young_specht_exact' requires permutation_sector='young_specht' "
                "or 'full_irrep_decomposition'."
            )
        if mode == "lifted_cauchy_exact" and self.permutation_sector != "trivial":
            raise ValueError(
                "lifted-Cauchy internal sectors must retain a globally trivial output."
            )
        if mode == "lifted_cauchy_exact":
            if self.basis_mode != "lifted_cauchy_scalar":
                raise ValueError(
                    "construction_mode='lifted_cauchy_exact' requires "
                    "basis_mode='lifted_cauchy_scalar'."
                )
            if self.coupling_tree != "explicit":
                raise ValueError(
                    "construction_mode='lifted_cauchy_exact' requires the "
                    "compiler-owned explicit coupling tree."
                )
            if self.coefficient_backend != "linear_lifted_cauchy_scalar":
                raise ValueError(
                    "construction_mode='lifted_cauchy_exact' requires the "
                    "linear_lifted_cauchy_scalar coefficient backend."
                )
            reserved = {
                "density": "A_s",
                "model_family": "linear_lifted_cauchy_scalar",
                "coupling_family": "linear_lifted_cauchy_scalar",
                "role_coordinate_policy": "role_resolved",
                "global_target_partition": "(N)",
                "target_L_R": 0,
                "o3_parity": 1,
            }
            for key, expected in reserved.items():
                if key not in self.metadata or self.metadata[key] != expected:
                    raise ValueError(
                        "lifted-Cauchy representation metadata requires "
                        f"{key}={expected!r}."
                    )
        if mode == "tagged_cauchy_exact":
            if self.permutation_sector != "trivial":
                raise ValueError(
                    "tagged-Cauchy internal sectors require a globally trivial output."
                )
            if self.basis_mode != "tagged_cauchy_image":
                raise ValueError(
                    "construction_mode='tagged_cauchy_exact' requires "
                    "basis_mode='tagged_cauchy_image'."
                )
            if self.coupling_tree != "explicit":
                raise ValueError(
                    "construction_mode='tagged_cauchy_exact' requires the "
                    "compiler-owned explicit coupling tree."
                )
            if self.coefficient_backend != "linear_tagged_cauchy_image":
                raise ValueError(
                    "construction_mode='tagged_cauchy_exact' requires the "
                    "linear_tagged_cauchy_image coefficient backend."
                )
            reserved = {
                "density": "tagged_density",
                "model_family": "linear_tagged_cauchy_image",
                "coupling_family": "linear_tagged_cauchy_image",
                "global_target_partition": "(N)",
                "target_L_R": 0,
                "o3_parity": 1,
            }
            for key, expected in reserved.items():
                if key not in self.metadata or self.metadata[key] != expected:
                    raise ValueError(
                        "tagged-Cauchy representation metadata requires "
                        f"{key}={expected!r}."
                    )
        if family == "ace" and self.permutation_sector != "trivial":
            raise ValueError("YE3TDescriptors.ace supports only the trivial permutation sector.")
        if family == "ace" and mode != "trivial_image_fast_path":
            raise ValueError("YE3TDescriptors.ace uses construction_mode='trivial_image_fast_path'.")

    @classmethod
    def ace(cls, *, basis_mode=None, fast_path_policy="auto", metadata=None):
        payload = {} if metadata is None else dict(metadata)
        payload.setdefault("runtime_status", "implemented_under_validation")
        return cls(
            basis_family="ace",
            permutation_sector="trivial",
            construction_mode="trivial_image_fast_path",
            basis_mode=basis_mode,
            fast_path_policy=fast_path_policy,
            permutation_group="slot_density_sum",
            coefficient_backend="ace_trivial_fast_path",
            validation_status="implemented_for_current_tests",
            metadata=payload,
        )

    @classmethod
    def ace_symmetric_power(
        cls,
        *,
        basis_mode="symmetric_power_subselection",
        fast_path_policy="auto",
        metadata=None,
    ):
        """Select the ACE trivial sector with symmetric-power subselections.

        This selector is for high-rank repeated-block inventories and evaluated
        descriptor subselections that can be represented by collapsed
        symmetric-power blocks. Single-block non-stretched scalar sectors can
        still be slower than stretched/pair-block sectors until a dedicated
        recurrence is wired for those outputs.
        """

        payload = {} if metadata is None else dict(metadata)
        payload.setdefault("representation_level", "ace_trivial_symmetric_power_subselection")
        payload.setdefault("basis_backend", "symmetric_power_kernel_inventory")
        payload.setdefault(
            "status",
            "ACE trivial-sector repeated-block subselection for symmetric-power kernel planning "
            "and descriptor evaluation when explicit compatible sectors are selected",
        )
        return cls(
            basis_family="ace",
            permutation_sector="trivial",
            construction_mode="trivial_image_fast_path",
            basis_mode=basis_mode,
            fast_path_policy=fast_path_policy,
            permutation_group="slot_density_sum",
            coefficient_backend="ace_symmetric_power_kernel_inventory",
            validation_status="implemented_for_explicit_symmetric_power_subselections",
            metadata=payload,
        )

    @classmethod
    def symmetric_power(cls, **kwargs):
        """Alias for the explicit ACE global ``lambda=(N)`` fast path."""

        metadata = dict(kwargs.pop("metadata", {}) or {})
        metadata.setdefault("global_young_label", "lambda=(N)")
        metadata.setdefault("block_young_label_policy", "mu_b=(k_b) for repeated ACE blocks")
        metadata.setdefault("runtime_status", "implemented_under_validation")
        return cls.ace_symmetric_power(metadata=metadata, **kwargs)



    @classmethod
    def ye3t(
        cls,
        *,
        permutation_sector="full_irrep_decomposition",
        construction_mode=None,
        basis_mode=None,
        fast_path_policy="auto",
        permutation_group=None,
        coupling_tree="balanced",
        coefficient_backend="certified_intertwiner",
        validation_status="not_validated",
        metadata=None,
    ):
        if construction_mode is None:
            construction_mode = "trivial_image_fast_path" if permutation_sector == "trivial" else "schur_weyl_exact"
        return cls(
            basis_family="ye3t",
            permutation_sector=permutation_sector,
            construction_mode=construction_mode,
            basis_mode=basis_mode,
            fast_path_policy=fast_path_policy,
            permutation_group=permutation_group,
            coupling_tree=coupling_tree,
            coefficient_backend=coefficient_backend,
            validation_status=validation_status,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def ye3(cls, **kwargs):
        """Select an arbitrary joint Young--E3 sector request."""

        metadata = dict(kwargs.pop("metadata", {}) or {})
        metadata.setdefault("representation_level", "global_ye3_sector")
        metadata.setdefault("basis_backend", "global_coupler")
        metadata.setdefault("runtime_status", "planned_not_public")
        kwargs.setdefault("coefficient_backend", "global_coupler")
        kwargs.setdefault("validation_status", "planned_not_public")
        return cls.ye3t(metadata=metadata, **kwargs)



    @classmethod
    def young_subgroup(
        cls,
        *,
        basis_mode=None,
        fast_path_policy="auto",
        permutation_group="repeated_channel_young_subgroup",
        coupling_tree="balanced",
        metadata=None,
    ):
        """Select the exact Young-subgroup invariant descriptor runtime.

        This is the block-first exact ACE path for invariants under Young
        subgroups that permute repeated channel/angular slots. It is not the
        full Specht-carrier runtime.
        """

        payload = {} if metadata is None else dict(metadata)
        payload.setdefault("representation_level", "young_subgroup_invariant")
        payload.setdefault("basis_backend", "exact_block_first_young_subgroup")
        payload.setdefault(
            "status",
            "evaluated Young-subgroup invariant runtime via exact block-first ACE labels; not full Specht-carrier materialization",
        )
        return cls(
            basis_family="ye3t",
            permutation_sector="young_subgroup_trivial",
            construction_mode="young_subgroup_exact",
            basis_mode=basis_mode,
            fast_path_policy=fast_path_policy,
            permutation_group=permutation_group,
            coupling_tree=coupling_tree,
            coefficient_backend="exact_block_first_young_subgroup",
            validation_status="implemented_young_subgroup_invariant_runtime",
            metadata=payload,
        )



    @classmethod
    def filtered_A_s(
        cls,
        *,
        slot_group="symmetric",
        slot_sectors=None,
        slot_specht_partitions=None,
        representation_subselection=None,
        coupling_tree="balanced",
        fast_path_policy="auto",
        metadata=None,
    ):
        incoming_metadata = {} if metadata is None else dict(metadata)
        subselection = normalize_a_s_subselection(representation_subselection)
        if slot_sectors is None:
            if subselection == "fully_symmetric":
                slot_sectors = ("trivial",)
            elif subselection == "fully_antisymmetric":
                slot_sectors = ("antisymmetric",)
            else:
                slot_sectors = ("trivial", "standard")
        if subselection is None:
            normalized_sectors = tuple(str(sector) for sector in slot_sectors)
            if normalized_sectors == ("trivial",):
                subselection = "fully_symmetric"
            elif normalized_sectors == ("antisymmetric",):
                subselection = "fully_antisymmetric"
            else:
                subselection = "equivariant"
        normalized_sectors = tuple(str(sector) for sector in slot_sectors)
        has_slot_specht = slot_specht_partitions is not None
        role_policy_report = _A_s_role_policy_report_from_metadata(
            slot_specht_partitions,
            incoming_metadata,
        )
        role_policy_report = {
            **dict(role_policy_report),
            "role_coordinate_policy": incoming_metadata.get(
                "role_coordinate_policy",
                role_policy_report.get("role_coordinate_policy", "role_resolved"),
            ),
        }
        if str(role_policy_report["role_coordinate_policy"]).strip().lower() in {
            "collapsed",
            "commutative_density",
            "identical_filters_discarded",
            "discarded",
        } and has_slot_specht:
            raise ValueError(
                "Nontrivial A_s Young sectors require a retained role coordinate before Young coupling."
            )
        if not bool(role_policy_report.get("passed", False)):
            detail = "; ".join(str(reason) for reason in role_policy_report.get("reasons", ()))
            raise ValueError(
                "Nontrivial A_s Young sectors require a retained role coordinate before Young coupling. "
                + detail
            )
        if has_slot_specht:
            permutation_sector = "young_specht"
            construction_mode = "young_specht_exact"
            coefficient_backend = "A_s_slot_specht_central_projector_norms"
            validation_status = "implemented_A_s_slot_specht_scalar_projector_norms_not_full_matrix_units"
        elif subselection == "fully_antisymmetric":
            permutation_sector = "antisymmetric"
            construction_mode = "antisymmetric_fast_path"
            coefficient_backend = "filtered_density_antisymmetric_wedge_norm"
            validation_status = "implemented_A_s_antisymmetric_magnitude_not_full_sign_carrier"
        elif subselection == "fully_symmetric":
            permutation_sector = "trivial"
            construction_mode = "trivial_image_fast_path"
            coefficient_backend = "filtered_density_trivial_slot_fast_path"
            validation_status = "implemented_A_s_trivial_slot_fast_path"
        else:
            permutation_sector = "orbital"
            construction_mode = "orbital_basis"
            coefficient_backend = "finite_permutation_orbitals"
            validation_status = "implemented_A_s_orbital_slot_readout_not_irrep_resolved"
        base_metadata = {
            "density": "A_s",
            "representation_level": (
                "slot_specht_central_projector_norms"
                if has_slot_specht
                else "permutation_module_equivariant_maps"
            ),
            "basis_backend": "slot_specht_central_projector_power" if has_slot_specht else "orbital_basis",
            "representation_subselection": subselection,
            "slot_sectors": list(normalized_sectors),
            "slot_specht_partitions": slot_specht_partitions,
            "A_s_role_coordinate_policy": role_policy_report,
            "maturity": (
                "implemented_and_tested_for_scalar_A_s_slot_specht_central_projector_norms; "
                "not a full matrix-unit/multiplicity-resolved Specht tensor-product runtime"
                if has_slot_specht
                else "implemented_and_tested_for_selected_A_s_readouts; "
                "not a full representation-theoretic decomposition"
            ),
            "representation_hierarchy": [
                "permutation_module_equivariant_maps",
                "young_specht_subselection",
                "trivial_or_antisymmetric_fast_path_when_applicable",
            ],
            "status": "filtered/lifted density path; not a full arbitrary-sector Young-E3 descriptor runtime",
        }
        base_metadata.update(incoming_metadata)
        return cls(
            basis_family="ye3t",
            permutation_sector=permutation_sector,
            construction_mode=construction_mode,
            basis_mode="filtered_A_s",
            fast_path_policy=fast_path_policy,
            permutation_group=f"filter_slots:{slot_group}",
            coupling_tree=coupling_tree,
            coefficient_backend=coefficient_backend,
            validation_status=validation_status,
            metadata=base_metadata,
        )

    @classmethod
    def lifted_cauchy_scalar(
        cls,
        *,
        fast_path_policy="auto",
        metadata=None,
    ):
        """Select global scalars with compiler-owned internal Cauchy sectors."""

        payload = {} if metadata is None else dict(metadata)
        reserved = {
            "density": "A_s",
            "model_family": "linear_lifted_cauchy_scalar",
            "coupling_family": "linear_lifted_cauchy_scalar",
            "role_coordinate_policy": "role_resolved",
            "global_target_partition": "(N)",
            "target_L_R": 0,
            "o3_parity": 1,
        }
        for key, expected in reserved.items():
            if key in payload and payload[key] != expected:
                raise ValueError(
                    "lifted-Cauchy representation metadata cannot override "
                    f"{key}={expected!r}."
                )
            payload[key] = expected
        payload.setdefault("runtime_status", "implemented_under_validation")
        payload.setdefault(
            "status",
            "global scalar with compiler-owned matched internal role/angular Cauchy sectors",
        )
        return cls(
            basis_family="ye3t",
            permutation_sector="trivial",
            construction_mode="lifted_cauchy_exact",
            basis_mode="lifted_cauchy_scalar",
            fast_path_policy=fast_path_policy,
            permutation_group="fixed_content_global_symmetric_parent",
            coupling_tree="explicit",
            coefficient_backend="linear_lifted_cauchy_scalar",
            validation_status="exact_compiler_gate_passed_runtime_under_validation",
            metadata=payload,
        )


    @classmethod
    def tagged_cauchy_carriers(cls, *, fast_path_policy="auto", metadata=None):
        """Select compiler-owned ordered-occurrence carrier descriptors."""
        payload = dict(metadata or {})
        reserved = {
            "density": "tagged_occurrence",
            "source_realization": "tagged_cauchy_occurrence",
            "coupling_family": "tagged_cauchy_carriers",
            "source_formal_parent": "(N)",
            "tag_identity": "distinct_ordered_periodic_occurrences",
            "right_tag_action": "S_k_on_ordered_occurrences",
            "source_compiler": "ye3t.couplings.tagged_cauchy_carrier_schedule",
        }
        for key, expected in reserved.items():
            if key in payload and payload[key] != expected:
                raise ValueError("tagged carrier representation cannot override " + key)
            payload[key] = expected
        payload.setdefault("runtime_status", "implemented_under_validation")
        payload.setdefault("task", "descriptor_only")
        return cls(
            basis_family="ye3t", permutation_sector="trivial",
            construction_mode="experimental", basis_mode="tagged_cauchy_carriers",
            fast_path_policy=fast_path_policy,
            permutation_group="formal_factor_S_N",
            coupling_tree="explicit", coefficient_backend="certified_intertwiner",
            validation_status="implemented_under_validation", metadata=payload,
        )

    @classmethod
    def tagged_cauchy_image(
        cls,
        *,
        fast_path_policy="auto",
        metadata=None,
    ):
        """Select a compiler-owned globally scalar tagged physical image."""

        payload = {} if metadata is None else dict(metadata)
        reserved = {
            "density": "tagged_density",
            "model_family": "linear_tagged_cauchy_image",
            "coupling_family": "linear_tagged_cauchy_image",
            "global_target_partition": "(N)",
            "target_L_R": 0,
            "o3_parity": 1,
        }
        for key, expected in reserved.items():
            if key in payload and payload[key] != expected:
                raise ValueError(
                    "tagged-Cauchy representation metadata cannot override "
                    f"{key}={expected!r}."
                )
            payload[key] = expected
        payload.setdefault("runtime_status", "implemented_under_validation")
        payload.setdefault(
            "status",
            "globally scalar exact physical image of selected raw tag opportunities",
        )
        return cls(
            basis_family="ye3t",
            permutation_sector="trivial",
            construction_mode="tagged_cauchy_exact",
            basis_mode="tagged_cauchy_image",
            fast_path_policy=fast_path_policy,
            permutation_group="fixed_tensor_order_tag_placement_parent",
            coupling_tree="explicit",
            coefficient_backend="linear_tagged_cauchy_image",
            validation_status="exact_compiler_gate_passed_runtime_under_validation",
            metadata=payload,
        )

    @classmethod
    def phi(
        cls,
        *,
        motif_group="decorated_motif_slots",
        coupling_tree="explicit",
        fast_path_policy="auto",
        motif_family="full",
        metadata=None,
    ):
        payload = {} if metadata is None else dict(metadata)
        payload.setdefault("density", "Phi")
        payload.setdefault("motif_family", str(motif_family))
        payload.setdefault(
            "status",
            "explicit Phi/barPhi motif path; validation/ablation scale, not a full arbitrary-sector Young-E3 descriptor runtime",
        )
        payload.setdefault(
            "maturity",
            "implemented reference/ablation path with cluster-Phi tests; not validated as a scalable or arbitrary-sector Young-E3 construction",
        )
        return cls(
            basis_family="ye3t",
            permutation_sector="full_irrep_decomposition",
            construction_mode="experimental",
            basis_mode="phi_motif",
            fast_path_policy=fast_path_policy,
            permutation_group=str(motif_group),
            coupling_tree=coupling_tree,
            coefficient_backend="explicit_phi_motif_reference",
            validation_status="implemented_phi_reference_not_full_ye3",
            metadata=payload,
        )

    @classmethod
    def phi_star(
        cls,
        *,
        motif_group="decorated_motif_slots",
        coupling_tree="explicit",
        fast_path_policy="auto",
        metadata=None,
    ):
        payload = {} if metadata is None else dict(metadata)
        payload.setdefault("motif_family", "star")
        payload.setdefault("density", "Phi")
        return cls.phi(
            motif_group=motif_group,
            coupling_tree=coupling_tree,
            fast_path_policy=fast_path_policy,
            motif_family="star",
            metadata=payload,
        )

    @classmethod
    def phi_complete(
        cls,
        *,
        motif_group="decorated_motif_slots",
        coupling_tree="explicit",
        fast_path_policy="auto",
        metadata=None,
    ):
        payload = {} if metadata is None else dict(metadata)
        payload.setdefault("motif_family", "complete")
        payload.setdefault("density", "Phi")
        payload.setdefault(
            "status",
            "explicit Phi complete inventory path; validation required for publication claims",
        )
        payload.setdefault(
            "maturity",
            "implemented exact inventory path with cached Young-subgroup and angular couplers; not yet a generalized runtime guarantee",
        )
        return cls.phi(
            motif_group=motif_group,
            coupling_tree=coupling_tree,
            fast_path_policy=fast_path_policy,
            motif_family="complete",
            metadata=payload,
        )

    @classmethod
    def from_config(cls, config=None, *, default_family="ace", default_basis_mode=None):
        if config is None:
            return cls.ace(basis_mode=default_basis_mode) if default_family == "ace" else cls.ye3t(basis_mode=default_basis_mode)
        if isinstance(config, cls):
            return config
        if isinstance(config, (str, Path)):
            payload = _load_representation_config_file(config)
        else:
            if not isinstance(config, Mapping):
                raise TypeError("representation must be a YE3TRepresentation, mapping, JSON/YAML path, or None.")
            payload = dict(config)
        if "representation" in payload and isinstance(payload["representation"], Mapping):
            nested = dict(payload["representation"])
            nested.update({key: value for key, value in payload.items() if key not in {"representation", "descriptor", "model"}})
            payload = nested
        if set(payload) & {"group", "ranks", "parent", "factorization", "uncoupled_factor_inputs", "intermediates"}:
            raise ValueError(
                "The rank-resolved representation schema belongs to "
                "ye3t.YE3TRepresentation.from_config; the ye3t_methods selector accepts legacy configs only."
            )
        if "ye3t_spec_file" in payload:
            return cls.from_spec_file(payload["ye3t_spec_file"])
        if "ye3t_spec" in payload:
            return cls.from_spec(YE3TSpec.from_dict(payload["ye3t_spec"]))
        if "spec" in payload and isinstance(payload["spec"], dict):
            return cls.from_spec(YE3TSpec.from_dict(payload["spec"]))
        family = str(payload.get("basis_family", default_family))
        basis_mode = payload.get("basis_mode", default_basis_mode)
        if family == "ace":
            metadata = dict(payload.get("metadata", {}))
            if (
                basis_mode in {"symmetric_power_subselection", "symmetric_power"}
                or metadata.get("basis_backend") == "symmetric_power_kernel_inventory"
            ):
                return cls.ace_symmetric_power(
                    basis_mode=basis_mode or "symmetric_power_subselection",
                    fast_path_policy=str(payload.get("fast_path_policy", "auto")),
                    metadata=metadata,
                )
            return cls.ace(
                basis_mode=basis_mode,
                fast_path_policy=str(payload.get("fast_path_policy", "auto")),
            )
        mode = payload.get("construction_mode", None)
        sector = payload.get("permutation_sector", "full_irrep_decomposition")
        metadata = dict(payload.get("metadata", {}))
        if (
            basis_mode == "lifted_cauchy_scalar"
            or mode == "lifted_cauchy_exact"
            or metadata.get("model_family") == "linear_lifted_cauchy_scalar"
            or metadata.get("coupling_family") == "linear_lifted_cauchy_scalar"
        ):
            return cls.lifted_cauchy_scalar(
                fast_path_policy=str(payload.get("fast_path_policy", "auto")),
                metadata=metadata,
            )
        if basis_mode == "tagged_cauchy_carriers" or metadata.get("coupling_family") == "tagged_cauchy_carriers":
            return cls.tagged_cauchy_carriers(
                fast_path_policy=str(payload.get("fast_path_policy", "auto")),
                metadata=metadata,
            )
        if (
            basis_mode == "tagged_cauchy_image"
            or mode == "tagged_cauchy_exact"
            or metadata.get("model_family") == "linear_tagged_cauchy_image"
            or metadata.get("coupling_family") == "linear_tagged_cauchy_image"
        ):
            return cls.tagged_cauchy_image(
                fast_path_policy=str(payload.get("fast_path_policy", "auto")),
                metadata=metadata,
            )
        if (
            basis_mode == "filtered_A_s"
            or metadata.get("density") == "A_s"
            or "representation_subselection" in payload
            or "slot_subselection" in payload
            or "slot_sectors" in payload
            or "slot_specht_partitions" in payload
        ):
            return cls.filtered_A_s(
                slot_group=payload.get("slot_group", payload.get("permutation_group", "symmetric")),
                slot_sectors=payload.get("slot_sectors", metadata.get("slot_sectors", None)),
                slot_specht_partitions=payload.get(
                    "slot_specht_partitions",
                    metadata.get("slot_specht_partitions", None),
                ),
                representation_subselection=payload.get(
                    "representation_subselection",
                    payload.get("slot_subselection", metadata.get("representation_subselection", None)),
                ),
                coupling_tree=str(payload.get("coupling_tree", "balanced")),
                fast_path_policy=str(payload.get("fast_path_policy", "auto")),
                metadata=metadata,
            )
        if basis_mode in {"phi_motif", "phi_star_motif"} or metadata.get("density") == "Phi" or "motif_family" in metadata:
            motif_family = metadata.get("motif_family", "full")
            if str(motif_family).strip().lower() in {"star", "star_only", "star-graphs", "star_graphs"}:
                return cls.phi_star(
                    motif_group=payload.get("motif_group", payload.get("permutation_group", "decorated_motif_slots")),
                    coupling_tree=str(payload.get("coupling_tree", "explicit")),
                    fast_path_policy=str(payload.get("fast_path_policy", "auto")),
                    metadata=metadata,
                )
            return cls.phi(
                motif_group=payload.get("motif_group", payload.get("permutation_group", "decorated_motif_slots")),
                coupling_tree=str(payload.get("coupling_tree", "explicit")),
                fast_path_policy=str(payload.get("fast_path_policy", "auto")),
                motif_family=motif_family,
                metadata=metadata,
            )
        if mode == "young_subgroup_exact" or sector == "young_subgroup_trivial":
            return cls.young_subgroup(
                basis_mode=basis_mode,
                fast_path_policy=str(payload.get("fast_path_policy", "auto")),
                permutation_group=payload.get("permutation_group", "repeated_channel_young_subgroup"),
                coupling_tree=str(payload.get("coupling_tree", "balanced")),
                metadata=dict(payload.get("metadata", {})),
            )
        return cls.ye3t(
            permutation_sector=sector,
            construction_mode=mode,
            basis_mode=basis_mode,
            fast_path_policy=str(payload.get("fast_path_policy", "auto")),
            permutation_group=payload.get("permutation_group", None),
            coupling_tree=str(payload.get("coupling_tree", "balanced")),
            coefficient_backend=str(payload.get("coefficient_backend", "certified_intertwiner")),
            validation_status=str(payload.get("validation_status", "not_validated")),
            metadata=dict(payload.get("metadata", {})),
        )

    def as_dict(self):
        return {
            "basis_family": str(self.basis_family),
            "permutation_sector": self.permutation_sector,
            "construction_mode": str(self.construction_mode),
            "basis_mode": self.basis_mode,
            "fast_path_policy": str(self.fast_path_policy),
            "permutation_group": self.permutation_group,
            "coupling_tree": str(self.coupling_tree),
            "coefficient_backend": str(self.coefficient_backend),
            "validation_status": str(self.validation_status),
            "metadata": dict(self.metadata),
        }

    def to_spec(self, **overrides):
        """Return a shared ``YE3TSpec`` view of this representation selector."""

        metadata = dict(self.metadata)
        target_rotation = overrides.pop("target_rotation", None)
        if target_rotation is None:
            L_R = metadata.get("L_R", metadata.get("target_L_R", 0))
            target_rotation = YE3TRotationTarget(L_R=int(L_R))
        elif not isinstance(target_rotation, YE3TRotationTarget):
            target_rotation = YE3TRotationTarget.from_dict(target_rotation)
        carrier = overrides.pop("carrier", None)
        if carrier is None:
            if self.basis_family == "ace":
                carrier = "ACE_density"
            elif (
                self.basis_mode in {"filtered_A_s", "tagged_cauchy_image"}
                or metadata.get("density") in {"A_s", "tagged_density"}
            ):
                carrier = "A_s"
            elif self.basis_mode == "phi_motif" or metadata.get("density") == "Phi":
                carrier = "Phi"
            else:
                carrier = "external_tensor"
        task = overrides.pop("task", metadata.get("task", "descriptor_only"))
        readout_payload = overrides.pop("readout", metadata.get("readout", None))
        readout = (
            readout_payload
            if isinstance(readout_payload, YE3TReadoutSpec)
            else YE3TReadoutSpec.from_dict(readout_payload)
        )
        runtime_status = overrides.pop("runtime_status", metadata.get("runtime_status", "planned_not_public"))
        target_permutation = overrides.pop("target_permutation", None)
        if target_permutation is None:
            if self.permutation_sector == "trivial":
                target_permutation = "trivial"
            elif self.permutation_sector == "antisymmetric":
                target_permutation = "antisymmetric"
            else:
                target_permutation = str(self.permutation_sector)
        backend = overrides.pop("coefficient_backend", str(self.coefficient_backend))
        if backend == "certified_intertwiner":
            backend = "global_coupler"
        elif backend == "ace_symmetric_power_kernel_inventory":
            backend = "symmetric_power_fast_path"
        content = tuple(overrides.pop("content", metadata.get("content", ())))
        block_permutation = tuple(overrides.pop("block_permutation", metadata.get("block_permutation", ())))
        spec_metadata = {**metadata, **dict(overrides.pop("metadata", {}))}
        if backend == "symmetric_power_fast_path" and str(target_permutation) == "trivial" and str(carrier) == "ACE_density":
            spec_metadata.setdefault("global_young_label", "lambda=(N)")
            spec_metadata.setdefault("global_target_partition", (int(len(content)),))
            spec_metadata.setdefault("block_young_label_policy", "mu_b=(k_b) for repeated ACE blocks")
            spec_metadata.setdefault("block_mu_labels", _default_block_mu_labels_from_content(content))
            spec_metadata.setdefault("label_scope", "global_target_partition_is_not_a_block_mu_label")
        return YE3TSpec(
            content=content,
            slot_roles=tuple(overrides.pop("slot_roles", metadata.get("slot_roles", ()))),
            target_permutation=str(target_permutation),
            block_permutation=block_permutation,
            target_rotation=target_rotation,
            carrier=str(carrier),
            task=str(task),
            readout=readout,
            radial_filters=dict(overrides.pop("radial_filters", metadata.get("radial_filters", {}))),
            tree_schedule=str(overrides.pop("tree_schedule", self.coupling_tree)),
            coefficient_backend=str(backend),
            fast_path_policy=str(overrides.pop("fast_path_policy", self.fast_path_policy)),
            validation_scope=str(overrides.pop("validation_scope", metadata.get("validation_scope", "counts"))),
            runtime_status=str(runtime_status),
            metadata=spec_metadata,
        )

    @classmethod
    def from_spec(cls, spec):
        """Build a representation selector from a shared ``YE3TSpec``."""

        spec = spec if isinstance(spec, YE3TSpec) else YE3TSpec.from_dict(spec)
        metadata = dict(spec.metadata)
        metadata["ye3t_spec"] = spec.to_dict()
        metadata.setdefault("content", tuple(spec.content))
        metadata.setdefault("slot_roles", tuple(spec.slot_roles))
        metadata.setdefault("block_permutation", tuple(spec.block_permutation))
        metadata.setdefault("target_rotation", spec.target_rotation.to_dict())
        metadata.setdefault("target_permutation", spec.target_permutation)
        metadata.setdefault("task", spec.task)
        metadata.setdefault("readout", spec.readout.to_dict())
        metadata.setdefault("radial_filters", dict(spec.radial_filters))
        metadata.setdefault("validation_scope", spec.validation_scope)
        metadata.setdefault("runtime_status", spec.runtime_status)
        if spec.carrier == "ACE_density" and spec.target_permutation == "trivial":
            if spec.coefficient_backend == "symmetric_power_fast_path":
                return cls.symmetric_power(
                    fast_path_policy=spec.fast_path_policy,
                    metadata=metadata,
                )
            return cls.ace(
                basis_mode=metadata.get("basis_mode", None),
                fast_path_policy=spec.fast_path_policy,
                metadata=metadata,
            )
        if spec.carrier == "A_s":
            if (
                spec.coefficient_backend == "linear_tagged_cauchy_image"
                or metadata.get("model_family")
                == "linear_tagged_cauchy_image"
            ):
                return cls.tagged_cauchy_image(
                    fast_path_policy=spec.fast_path_policy,
                    metadata=metadata,
                )
            if (
                spec.coefficient_backend == "linear_lifted_cauchy_scalar"
                or metadata.get("model_family") == "linear_lifted_cauchy_scalar"
            ):
                return cls.lifted_cauchy_scalar(
                    fast_path_policy=spec.fast_path_policy,
                    metadata=metadata,
                )
            representation_subselection = metadata.get("representation_subselection", None)
            if representation_subselection is None:
                if spec.target_permutation == "trivial":
                    representation_subselection = "fully_symmetric"
                elif spec.target_permutation == "antisymmetric":
                    representation_subselection = "fully_antisymmetric"
                else:
                    representation_subselection = "equivariant"
            return cls.filtered_A_s(
                slot_group=metadata.get("slot_group", "symmetric"),
                slot_sectors=metadata.get("slot_sectors", None),
                slot_specht_partitions=metadata.get("slot_specht_partitions", None),
                representation_subselection=representation_subselection,
                coupling_tree=spec.tree_schedule,
                fast_path_policy=spec.fast_path_policy,
                metadata=metadata,
            )
        if spec.carrier == "Phi":
            motif_family = metadata.get("motif_family", "full")
            if str(motif_family).strip().lower() == "complete":
                return cls.phi_complete(
                    motif_group=metadata.get("motif_group", "decorated_motif_slots"),
                    coupling_tree=spec.tree_schedule,
                    fast_path_policy=spec.fast_path_policy,
                    metadata=metadata,
                )
            if str(motif_family).strip().lower() in {"star", "star_only", "star-graphs", "star_graphs"}:
                return cls.phi_star(
                    motif_group=metadata.get("motif_group", "decorated_motif_slots"),
                    coupling_tree=spec.tree_schedule,
                    fast_path_policy=spec.fast_path_policy,
                    metadata=metadata,
                )
            return cls.phi(
                motif_group=metadata.get("motif_group", "decorated_motif_slots"),
                coupling_tree=spec.tree_schedule,
                fast_path_policy=spec.fast_path_policy,
                motif_family=motif_family,
                metadata=metadata,
            )
        return cls.ye3(
            permutation_sector=spec.target_permutation,
            construction_mode="schur_weyl_exact",
            basis_mode=metadata.get("basis_mode", None),
            fast_path_policy=spec.fast_path_policy,
            coupling_tree=spec.tree_schedule,
            coefficient_backend=spec.coefficient_backend,
            validation_status=spec.runtime_status,
            metadata=metadata,
        )

    @classmethod
    def from_spec_file(cls, path):
        """Build a representation selector from a shared YE3TSpec config file."""

        return cls.from_spec(YE3TSpec.from_file(path))

    def to_spec_file(self, path, **overrides):
        """Write this selector's shared YE3TSpec view to a config file."""

        self.to_spec(**overrides).to_file(path)

    @property
    def uses_trivial_fast_path(self):
        return self.permutation_sector == "trivial" and self.construction_mode == "trivial_image_fast_path"


__all__ = [
    "YE3TRepresentation",
    "normalize_a_s_subselection",
]
