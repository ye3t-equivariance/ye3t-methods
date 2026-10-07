from .labeling import CompactLabel, SingleChannelLabel, DescriptorSpec
from .site_basis_v2 import (
    DEFAULT_ATOMIC_BASE_NORMALIZATION,
    SiteBasisConfig,
    SiteBasisV2,
    SiteBasisVJPRecord,
)
from .atomic_base_cache import AtomicBaseCache, DescriptorRuntimeCache, EvaluationContext, NormalizationMap
from .angular_basis import (
    AngularBasis,
    AngularConvention,
    ComplexSphericalHarmonicsBasis,
    RealSphericalHarmonicsBasis,
    angular_basis_for_backend,
    complex_to_real_tesseral_matrix,
    complex_to_real_tesseral_metadata,
)
from .radial_basis import (
    BesselRadialBasis,
    ChebExpCosRadialBasis,
    GaussianRadialBasis,
    RadialBasis,
    RadialConvention,
    TabulatedRadialBasis,
    radial_basis_for_kind,
)
from .site_basis_serialization import (
    deserialize_site_basis_config,
    serialize_site_basis_config,
    site_basis_dtype_from_name,
    site_basis_dtype_to_name,
)
from .ace_eval_v2 import (
    ACECovariantEvaluator,
    AtomicProductCollection,
    AtomicProductGroup,
    GeneralizedCouplingLibrary,
    compact_label_to_channels_with_mus_ks,
)
from .product_dag import (
    ProductDAG,
    ProductExpansionNode,
    ProductExpansionOutputTerm,
    ProductExpansionSchedule,
    ProductDAGValues,
    evaluate_product_dag,
    product_expansion_schedule_from_compiled_cached,
    product_dag_coefficients_are_real,
    reverse_product_dag_adjoint,
    reverse_product_dag_adjoint_chunked_triton,
    reverse_product_dag_adjoint_depth_triton,
    reverse_product_dag_adjoint_reference,
    reverse_product_dag_adjoint_triton,
)
from .product_rule import (
    ExplicitProductRuleEvaluator,
    ProductRuleResult,
    evaluate_explicit_product_rule,
    explicit_product_rule_linear_adjoint,
)
from .cy_factor_product import (
    CYFactorProductEvaluation,
    CYFactorProductEvaluator,
    ForceDesignAccumulator,
    ModelForceEvaluator,
    ModelForceResult,
    NormalEquationAccumulator,
    NormalEquationUpdate,
)
from .neighbors import NeighborData, brute_force_neighbor_data, neighbor_data_from_ase_atoms, random_rotation_matrix, rotate_positions
from .edge_geometry import (
    directed_edges_all_images_bruteforce,
    directed_edges_bruteforce,
    edge_displacements_from_indices,
    minimum_image_displacement,
    normalize_pbc,
    unique_periodic_cutoff_margin,
    voigt_from_stress_tensor,
)
from .descriptor_sets import (
    DescriptorGenerationSettings,
    DescriptorCollection,
    MultipletTensor,
    enumerate_compact_labels,
    filter_compact_labels_for_settings,
    label_has_natural_parity,
    select_compact_labels,
    compile_descriptor_artifacts,
    build_descriptor_specs_from_settings,
    normalize_basis_mode,
)
from ye3t_methods.atomistic.cache import DescriptorArtifactCacheKey, DescriptorBuildCache, DescriptorEnumerationCacheKey, get_shared_descriptor_build_cache
from .gradients import (
    EdgeGeometry,
    LAMMPSPaceLikeOutput,
    descriptor_charge_vjp,
    descriptor_gradients_wrt_positions,
    descriptor_position_vjp,
    edge_vectors_from_positions,
    format_lammps_compute_pace_like,
)
from .real_tesseral import complex_multiplet_to_real_tesseral
from .stress_io import StressExtXYZData, read_simple_extxyz
from .models import LinearScalarACEModel, StressAwareScalarACEModel
from .aux_fields import magnetic_orientation_basis
from .charge_bounds import (
    COMMON_OXIDATION_STATE_RANGES,
    ChargeBounds,
    charge_bounds_from_data,
    oxidation_state_charge_bounds,
    resolve_charge_bounds,
)

from .property_targets import (
    YE3TTargetSpec,
    charge_target_spec,
    property_target_spec,
    stress_target_spec,
    target_provenance_report,
)
from .stress_basis import StressBasisSettings, frobenius_norm_from_tesseral, build_stress_basis
from .rotation_checks import (
    check_all_octants,
    wigner_D_numeric,
    build_rotation_test_descriptor_collection,
    build_default_rotation_site_basis_config,
    evaluate_descriptor_multiplets,
    check_descriptor_rotation_equivariance,
)
