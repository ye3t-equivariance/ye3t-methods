
import torch

from .site_basis_v2 import DEFAULT_ATOMIC_BASE_NORMALIZATION, SiteBasisConfig


def site_basis_dtype_to_name(dtype):
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise ValueError(f"Unsupported dtype {dtype!r}")


def site_basis_dtype_from_name(name):
    if str(name) == "float32":
        return torch.float32
    if str(name) == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype name {name!r}")


def serialize_site_basis_config(site_basis_config):
    """Serialize all stable public ``SiteBasisConfig`` normalization knobs."""

    payload = {
        "rc": [float(v) for v in site_basis_config.rc],
        "lmbda": [float(v) for v in site_basis_config.lmbda],
        "nradmax": int(site_basis_config.nradmax),
        "lmax": int(site_basis_config.lmax),
        "kmax": int(site_basis_config.kmax),
        "possible_types": [int(v) for v in site_basis_config.possible_types],
        "radial_basis": str(site_basis_config.radial_basis),
        "chemical_basis": str(site_basis_config.chemical_basis),
        "charge_mode": str(site_basis_config.charge_mode),
        "charge_normalization_mode": str(site_basis_config.charge_normalization_mode),
        "charge_squash_scale": float(site_basis_config.charge_squash_scale),
        "q_min": [float(v) for v in site_basis_config.q_min],
        "q_max": [float(v) for v in site_basis_config.q_max],
        "atomic_base_normalization": str(site_basis_config.atomic_base_normalization),
        "atomic_base_normalization_epsilon": float(site_basis_config.atomic_base_normalization_epsilon),
        "factor_normalization": str(getattr(site_basis_config, "factor_normalization", "none")),
        "spherical_backend": str(site_basis_config.spherical_backend),
        "source_backend": str(site_basis_config.source_backend),
        "native_source_min_edges": int(site_basis_config.native_source_min_edges),
        "dtype": site_basis_dtype_to_name(site_basis_config.dtype),
    }
    if str(site_basis_config.spherical_normalization) != "orthonormal":
        payload["spherical_normalization"] = str(site_basis_config.spherical_normalization)
    if site_basis_config.pace_cutoff_width is not None:
        payload.update(
            {
                "pace_cutoff_width": [float(v) for v in site_basis_config.pace_cutoff_width],
                "pace_spline_spacing": [float(v) for v in site_basis_config.pace_spline_spacing],
                "pace_inner_cutoff": [float(v) for v in site_basis_config.pace_inner_cutoff],
                "pace_inner_cutoff_width": [float(v) for v in site_basis_config.pace_inner_cutoff_width],
                "pace_crad_policy": str(site_basis_config.pace_crad_policy),
            }
        )
    return payload


def deserialize_site_basis_config(
    payload,
    *,
    default_rc = (5.0,),
    default_lmbda = (0.25,),
    default_possible_types = (0,),
    default_lmax = 1,
    default_nradmax = 8,
):
    """Deserialize ``SiteBasisConfig`` payloads with backward-compatible defaults."""

    known = {
        "rc", "lmbda", "nradmax", "lmax", "kmax", "possible_types",
        "radial_basis", "chemical_basis", "charge_mode",
        "charge_normalization_mode", "charge_squash_scale", "q_min", "q_max",
        "atomic_base_normalization", "atomic_base_normalization_epsilon",
        "factor_normalization", "spherical_backend", "source_backend",
        "native_source_min_edges", "dtype", "pace_cutoff_width",
        "pace_spline_spacing", "pace_inner_cutoff", "pace_inner_cutoff_width",
        "pace_crad_policy", "spherical_normalization",
    }
    unsupported = sorted(str(key) for key, value in payload.items()
                         if key not in known and value is not None)
    if unsupported:
        raise ValueError("Unsupported site-basis settings: " + ", ".join(unsupported))
    dtype = site_basis_dtype_from_name(str(payload.get("dtype", "float64")))
    return SiteBasisConfig(
        rc=payload.get("rc", default_rc),  # type: ignore[arg-type]
        lmbda=payload.get("lmbda", default_lmbda),  # type: ignore[arg-type]
        nradmax=int(payload.get("nradmax", default_nradmax)),
        lmax=int(payload.get("lmax", default_lmax)),
        kmax=int(payload.get("kmax", 0)),
        possible_types=tuple(int(v) for v in payload.get("possible_types", default_possible_types)),  # type: ignore[arg-type]
        radial_basis=str(payload.get("radial_basis", "ChebExpCos")),
        chemical_basis=str(payload.get("chemical_basis", "delta")),
        charge_mode=str(payload.get("charge_mode", "none")),
        charge_normalization_mode=str(payload.get("charge_normalization_mode", "linear_clip")),
        charge_squash_scale=float(payload.get("charge_squash_scale", 1.0)),
        q_min=payload.get("q_min", [-1.0]),  # type: ignore[arg-type]
        q_max=payload.get("q_max", [1.0]),  # type: ignore[arg-type]
        atomic_base_normalization=str(payload.get("atomic_base_normalization", DEFAULT_ATOMIC_BASE_NORMALIZATION)),
        atomic_base_normalization_epsilon=float(payload.get("atomic_base_normalization_epsilon", 0.0)),
        factor_normalization=str(payload.get("factor_normalization", "bounded")),
        spherical_backend=str(payload.get("spherical_backend", "real")),
        source_backend=str(payload.get("source_backend", "auto")),
        native_source_min_edges=int(payload.get("native_source_min_edges", 0)),
        dtype=dtype,
        complex_dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
        pace_cutoff_width=payload.get("pace_cutoff_width", None),
        pace_spline_spacing=payload.get("pace_spline_spacing", None),
        pace_inner_cutoff=payload.get("pace_inner_cutoff", 0.0),
        pace_inner_cutoff_width=payload.get("pace_inner_cutoff_width", 0.0),
        pace_crad_policy=str(payload.get("pace_crad_policy", "identity")),
        spherical_normalization=str(payload.get("spherical_normalization", "orthonormal")),
    )
