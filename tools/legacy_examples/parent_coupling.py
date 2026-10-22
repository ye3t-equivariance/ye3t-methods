"""Inspect an exact rank-eight coupling plan with a nontrivial parent type."""

from ye3t.couplings import typed_repeated_subtree_product_plan


config = {
    "metadata": {"name": "rank_eight_parent_coupling"},
    "basis": {
        "branch_input_Ls": (1, 1), "branch_output_L": 1,
        "repeat_count": 2, "block_output_L": 2,
    },
    "representation": {"parent_young": (4, 4), "parent_L": 2},
    "runtime": {"compile_coefficients": False},
    "model": None,
    "targets": {"coupling_plan": True},
    "validation": {"require_pass": True},
}
# Two repeated branches each contain four tensor positions.
plan = typed_repeated_subtree_product_plan(
    branch_classes=(config["basis"], config["basis"]),
    parent_partition=config["representation"]["parent_young"],
    target_L=config["representation"]["parent_L"],
)
if config["validation"]["require_pass"]:
    assert plan.validation_report["passed"]
print("rank, parent Young partition, parent L", plan.rank, plan.parent_partition, plan.target_L)
print("LR multiplicity, parent tableaux", plan.lr_multiplicity, plan.parent_tableau_count)

# TODO: connect a validated nontrivial formal parent plan to atomistic
# geometry inputs before presenting this as an evaluated site descriptor.
