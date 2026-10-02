"""Apply a rank-three global parent coupler to supplied tensor values."""

import torch

from ye3t import CompileYE3TCouplers, YE3TSpec
from ye3t_ace import YE3TDescriptors


descriptor = YE3TDescriptors.ye3t({
    "elements": ["Si"], "type_map": {"Si": 0}, "cutoff": 4.0,
    "ranks": [1], "basis_type": "no_charge", "k_o_max": 0, "k_max": [0],
    "nmax": [1], "lmax": [1], "lmin": [0], "L_R": 1, "M_R_values": [0],
    "max_labels_per_rank": 1, "max_variants_per_label": 1,
    "site_basis": {"mode": "explicit", "rc": [4.0], "lmbda": [0.25]},
    "backend": "pytorch", "content": (1, 1, 2),
    "metadata": {"input_Ls": (0, 0, 1)},
    "representation": {
        "permutation_sector": "young:(2,1)",
        "construction_mode": "schur_weyl_exact",
        "coupling_tree": "balanced",
    },
})
spec = YE3TSpec.from_dict(descriptor.metadata["ye3t_spec"])
coupler = CompileYE3TCouplers(spec)
width = coupler.sparse_coefficient_tables[0]["shape"][1]
values = torch.arange(width, dtype=torch.float64).reshape(1, width)
result = descriptor.evaluate_global_coupler_reference(values)
print("parent_partition, L", spec.target_permutation, spec.target_rotation.L_R)
print("coefficient_view_shape", result.values.shape)
print("input_source", result.metadata["geometry_carrier_realization"])
