"""Compact ACE labels tied to the public sector notation."""

from ye3t import format_block_trivial_lambda, format_nu, format_ye3t_basis


def format_ace_sector(nin, lin, L_R, *, quotient = "full", alpha = None, convention = "ace-compact-v1"):
    """Return a compact ACE sector header with enough context for labels."""
    parts = [
        f"nu={format_nu(nin, lin)}",
        format_block_trivial_lambda(nin, lin),
        f"L_R={int(L_R)}",
        f"q={quotient}",
    ]
    if alpha is not None:
        parts.append(f"alpha={int(alpha)}")
    parts.append(f"convention={convention}")
    return "ACE[" + "; ".join(parts) + "]"


def format_ace_basis(index, compact_label = None, *, quotient = "full"):
    """Return one compact ACE basis coordinate inside a sector."""
    prefix = f"#{int(index)}"
    if compact_label is None:
        return prefix
    detail = format_ye3t_basis(index, compact_label, quotient=quotient)
    return prefix + " " + detail.removeprefix("Basis: ")
