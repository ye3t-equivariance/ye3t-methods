"""Apply an exact nontrivial Young/rotation parent coupler to tensor values.

The values below are supplied tensor slots, not atomistic descriptors.
"""

import torch

from ye3t import CompileYE3TCouplers, YE3TSpec


config = {
    "metadata": {"name": "parent_coefficient"},
    "basis": {"content": (1, 1, 2), "input_Ls": (0, 0, 1)},
    "representation": {
        "carrier": "Phi", "parent_young": "young:2,1", "parent_L": 1,
        "tree_schedule": "balanced",
    },
    "runtime": {"dtype": torch.complex128},
    "model": None,
    "targets": {"parent_coefficient": True},
    "validation": {"require_certificate": True},
}
spec = YE3TSpec.from_dict({
    "content": config["basis"]["content"],
    "slot_roles": tuple(f"slot_{index}" for index in range(len(config["basis"]["content"]))),
    "target_permutation": config["representation"]["parent_young"],
    "target_rotation": {"L_R": config["representation"]["parent_L"]},
    "carrier": config["representation"]["carrier"],
    "tree_schedule": config["representation"]["tree_schedule"],
    "validation_scope": "projectors",
    "runtime_status": "implemented_under_validation",
    "metadata": {"input_Ls": config["basis"]["input_Ls"]},
})
coupler = CompileYE3TCouplers(spec)
if config["validation"]["require_certificate"]:
    assert coupler.certificate.passed
width = coupler.sparse_coefficient_tables[0]["shape"][1]
values = torch.arange(width, dtype=torch.float64).to(config["runtime"]["dtype"])
result = coupler.apply_sparse_coefficient_table_torch(values)
print("parent_partition, L", spec.target_permutation, spec.target_rotation.L_R)
print("coefficient_view_shape", tuple(result.shape))
print("coefficient_values", result.tolist())
