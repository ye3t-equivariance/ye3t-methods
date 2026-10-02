"""Inspect an exact rank-eight coupling plan with a nontrivial parent type."""

from ye3t.couplings import typed_repeated_subtree_product_plan


# Two repeated branches each contain four tensor positions.
branch = {
    "branch_input_Ls": (1, 1),
    "branch_output_L": 1,
    "repeat_count": 2,
    "block_output_L": 2,
}
plan = typed_repeated_subtree_product_plan(
    branch_classes=(branch, branch), parent_partition=(4, 4), target_L=2,
)
assert plan.validation_report["passed"]
print("rank, parent Young partition, parent L", plan.rank, plan.parent_partition, plan.target_L)
print("LR multiplicity, parent tableaux", plan.lr_multiplicity, plan.parent_tableau_count)

# TODO: connect a validated nontrivial formal parent plan to atomistic
# geometry inputs before presenting this as an evaluated site descriptor.
