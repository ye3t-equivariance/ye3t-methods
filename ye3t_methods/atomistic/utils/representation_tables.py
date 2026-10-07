"""Representation-table helpers for descriptor inventory examples."""

from ye3t import format_block_trivial_lambda, format_nu, format_young_subgroup
from ye3t.core.basis.theory import ace_invariant_subspace_decomposition


def _format_block_decomposition(decomposition):
    parts = []
    for block_decomposition in decomposition.symmetric_power_decompositions:
        block = block_decomposition.block
        outputs = ", ".join(
            f"V_{int(Lambda)}^{int(multiplicity)}"
            for Lambda, multiplicity in sorted(block_decomposition.d_by_Lambda.items())
        )
        parts.append(
            f"eta={int(block.eta)}, l={int(block.l)}: "
            f"Sym^{int(block.multiplicity)}(V_{int(block.l)}) -> {outputs}"
        )
    return "; ".join(parts)


def symmetry_inventory_decomposition_rows(inventory, *, max_records_per_rank=None, max_output_L=None):
    """Return rows describing Young-subgroup and SO3 blocks in an inventory.

    The rows describe the block-first ACE trivial-sector decomposition used by
    the current symmetric-power inventory. They are not a full nontrivial
    Specht-sector decomposition.
    """

    rows = []
    for rank in tuple(int(value) for value in inventory.ranks):
        records = tuple(inventory.records_by_rank[int(rank)])
        if max_records_per_rank is not None:
            records = records[: int(max_records_per_rank)]
        for record_index, record in enumerate(records):
            n_in = tuple(int(value) for value in record["n_in"])
            l_in = tuple(int(value) for value in record["l_in"])
            decomposition = ace_invariant_subspace_decomposition(n_in, l_in)
            output_counts = {
                int(L_R): int(count)
                for L_R, count in sorted(decomposition.alpha_by_L_R.items())
                if max_output_L is None or int(L_R) <= int(max_output_L)
            }
            rows.append(
                {
                    "rank": int(rank),
                    "record_index": int(record_index),
                    "nu": format_nu(n_in, l_in),
                    "young_subgroup": format_young_subgroup(n_in, l_in),
                    "lambda": format_block_trivial_lambda(n_in, l_in),
                    "n_in": n_in,
                    "l_in": l_in,
                    "pair_orbits": tuple(int(value) for value in record["pair_orbits"]),
                    "block_decomposition": _format_block_decomposition(decomposition),
                    "output_irreps": tuple((int(L_R), int(count)) for L_R, count in output_counts.items()),
                    "status": "Young-subgroup invariant ACE trivial sector; not a full Specht-sector decomposition",
                }
            )
    return tuple(rows)


def print_symmetry_inventory_decomposition(title, rows):
    """Print a compact decomposition table for descriptor inventory rows."""

    print(title)
    for row in rows:
        outputs = ", ".join(f"L_R={L_R}: alpha={alpha}" for L_R, alpha in row["output_irreps"])
        print(f"rank {row['rank']} record {row['record_index']}")
        print(f"  nu: {row['nu']}")
        print(f"  G_nu: {row['young_subgroup']}")
        print(f"  lambda: {row['lambda']}")
        print(f"  pair_orbits: {row['pair_orbits']}")
        print(f"  blocks: {row['block_decomposition']}")
        print(f"  SO3 outputs: {outputs}")
        print(f"  status: {row['status']}")


__all__ = [
    "print_symmetry_inventory_decomposition",
    "symmetry_inventory_decomposition_rows",
]
