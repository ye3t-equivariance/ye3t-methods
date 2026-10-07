"""Use a compiled YE3T Cauchy basis with factors supplied by another method.

Each factor has one role axis and a complete m=-l,...,l multiplet. Its values
could be produced from lifted density, an explicit Phi motif, or a message.
This example supplies the multiplets directly so the coupling contract is
visible. An atomistic source must also define factor ordering and derivatives.
"""

import torch

from ye3t import YE3TRepresentation, couplings
from ye3t_methods import Basis


config = {
    "metadata": {"schema": "ye3t_config_v1", "name": "coupled_factors",
                 "status": "experimental"},
    "basis": {
        "tensor_product": {"kind": "ordered_role_cauchy",
                           "role_kappa_policy": "all_valid"},
        "channels": [
            {"factor_type": f"u_{index}", "l": 1,
             "source_family_id": "supplied_multiplets"}
            for index in range(3)
        ],
        "block_sizes": [1, 1, 1], "role_dimension": 1,
    },
    "representation": {
        "group": "O3", "ranks": [3],
        "parent": {"young_lambda": "(2,1)", "L": 1, "parity": "odd"},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {3: 3}, "l_max_per_rank": {3: 1}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    },
    "runtime": {"device": "cpu", "dtype": "complex128",
                "magnetic_basis": "complex_condon_shortley"},
    "model": {}, "targets": {},
    "validation": {"checks": ["exact_count", "artifact_hash",
                              "factor_permutation", "rotation", "parity"]},
}


representation = YE3TRepresentation.from_config(config["representation"])
basis = Basis.from_config(config["basis"], representation=representation,
                          runtime=config["runtime"])
report = couplings.count(basis.coupling_request())
compiled = couplings.compile(couplings.plan(report))
execution_plan = couplings.lower_ordered_role_cauchy_execution_plan(compiled)
basis = Basis.from_config(config["basis"], representation=representation,
                          runtime=config["runtime"], compiled_plan=execution_plan)

factors = torch.tensor(
    [[[[0.2, 0.3, -0.5]], [[0.7, -0.1, 0.4]], [[0.1, 0.9, -0.2]]]],
    dtype=getattr(torch, config["runtime"]["dtype"]),
    device=config["runtime"]["device"],
)
features = basis.create_factors(factors)
assert features.shape == (1, report["multiplet_count"],
                          report["tableau_count"], 3)

print("independent coupling paths", len(basis.labels))
print("catalogue paths", basis.catalogue.counts()["multiplet_count"])
print("coupled (batch,a,t,M) shape", tuple(features.shape))
print("factor evaluation available", basis.resolution.capability_report["factor_evaluation_available"])
print("first label", basis.labels[0])
print("compiled hash", compiled["self_hash"])
print("execution plan hash", execution_plan.plan_hash)
