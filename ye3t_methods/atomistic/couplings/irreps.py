
"""Compact rotation-irrep parsing used by exact interaction blocks."""

import re
from ye3t_methods.atomistic._record import recordclass

@recordclass(('mul', 'l', 'parity'), frozen = True)
class IrrepTerm:
    @property
    def dim(self):
        return self.mul * (2 * self.l + 1)
    def to_string(self):
        return f"{self.mul}x{self.l}{self.parity}"

_IRREP_RE = re.compile(r"\s*(?:(\d+)x)?(\d+)([eo])\s*")

def parse_irreps(irreps):
    if isinstance(irreps, str):
        terms=[]
        for part in irreps.split('+'):
            part=part.strip()
            if not part: continue
            m=_IRREP_RE.fullmatch(part)
            if m is None: raise ValueError(f"Could not parse irreps term: {part!r}")
            terms.append(IrrepTerm(int(m.group(1) or 1), int(m.group(2)), m.group(3)))
        return terms
    return list(irreps)

def format_irreps(terms):
    return ' + '.join(term.to_string() for term in terms)

def total_dim(terms):
    return sum(t.dim for t in parse_irreps(terms))

def default_parity_for_L(L):
    return 'e' if (L % 2 == 0) else 'o'

def make_output_irreps_string(L, multiplicity, parity = None):
    p = default_parity_for_L(L) if parity is None else parity
    return format_irreps([IrrepTerm(multiplicity, L, p)])
