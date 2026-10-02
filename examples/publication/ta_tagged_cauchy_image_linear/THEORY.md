# Theory contract for the tagged-Cauchy physical-image diagnostic

## Scope and notation

This example certifies one bounded part of a more general tagged YE3T basis.
`N` is tensor-product order and `s` is the number of explicit ordered tags.
Young partitions are always qualified by the carrier on which they act:

- `kappa_tag` acts on tag labels;
- `placement_parent_lambda` acts on tensor-factor positions;
- `kappa_b` acts inside repeated source/content block `b`;
- `Lambda_b` is the corresponding intermediate angular momentum.

The output is globally permutation trivial, even parity, and total `L=0`.
Nontrivial labels occur only at intermediate stages. Different `N` spaces are
not mixed in this diagnostic.

Four distinct permutation actions must not be conflated: relabeling physical
neighbors, permuting explicit tag labels, permuting formal tensor-factor
positions, and permuting equal factors inside a repeated content block. A
valid descriptor retains the relevant carrier until YE3T supplies the matching
intertwiner and final trivialization.

## Fixed-order complete-basis statement

For fixed `N`, fixed source content, total `L` and parity, the repeated blocks
have the Cauchy decomposition

```text
Sym^k(W tensor V_l)
  = direct_sum_{kappa partition k}
      S_kappa(W) tensor S_kappa(V_l).
```

If every valid `kappa`, block angular momentum, outer coupling, and true
multiplicity copy is retained, the resulting coordinates form a complete
orthogonal (and, after normalization, orthonormal) basis of that finite
globally symmetric parent sector. Selecting some coordinates from that basis
does not make them dependent. This is the fixed-`N`, fixed-content analogue of
the ordinary YE3T basis statement; no inter-`N` orthogonality is claimed or
needed.

That abstract statement is separate from the physical source map. If chosen
radial/role sources satisfy exact functional identities, an independent
abstract basis can become dependent after evaluation. The compiler therefore
forms the exact image after physical source substitution. This is not an
empirical design-matrix rank test.

## Bounded `s=2` tag-placement carrier

For two ordered unit tags, the placement carrier has basis `e_(ij)`, `i != j`,
and dimension `N(N-1)`. It carries commuting actions

```text
L(sigma) e_(ij) = e_(sigma(i),sigma(j)),
R(t) e_(ij) = e_(ji).
```

The right `S_2` sign sector `kappa_tag=(1,1)` cannot survive global pooling by
itself. It must be paired before pooling with a physical source carrier having
the matching typed left and right actions. In the present exact witness,

```text
N = 4, s = 2, tag_slot_multiplicities = (1,1)
kappa_tag = (1,1)
placement_parent_lambda = (3,1)
role_kappa = (2,1,1), role_content = (1,1,2)
angular_kappa = (2,2), input l = (1,1,1,1), total L = 0.
```

The relevant same-rank Kronecker multiplicity is one. The compiler constructs
the placement carrier, role/angular companions, intertwiner, and final scalar;
the application package does not hand-enumerate these couplings.

For this same-source inclusive-density construction, the selected `s=1` row is
exactly the ordinary `s=0` row. It is retained only as the explicitly named
`s1_duplicate_control`. It is not evidence for a general one-tag carrier.

## Exact physical image

Let a generalized one-neighbor moment be

```text
M[alpha] = sum_j product_{q in alpha} phi_{j,q}.
```

The bounded raw rows lower into one commutative moment algebra. For example,

```text
s=0: product_a M[q_a]
s=2: (M[g_1] M[g_2] - M[g_1 union g_2]) product_a M[q_a].
```

The second expression is the exact ordered-distinct-tag reduction. Residual
density factors remain self interacting and may include tagged neighbors. The
study deliberately does not compare purification or non-self-interacting
bases; that is deferred to Task `56T-SI`.

Raw opportunities are selected first, then their exact physical image is
formed. This ordering is essential: slicing coordinates from the full image
would generally define a different subspace. For `M` homogeneous source keys,
the present bounded construction has

```text
dim I_S = M [ 1(S intersects {0,1}) + 1(2 in S) ].
```

This formula is certified only for the exact `N=4`, homogeneous-`l=1`,
same-source construction and its serialized source-product conditions. The
compiler verifies it against every materialized image.

Exact Gram--Schmidt uses the declared symmetric-Fock coefficient metric, in
which a moment monomial with occupations `a_g` has weight `product_g a_g!`.
It emits exact raw-to-image and image-to-raw maps, an identity image Gram, a
sparse forward schedule, and a division-free transpose adjoint. This is
orthogonality in compiler coordinates, not an empirical configuration-space
`L2` statement. No Gram construction or solve occurs during fitting or a
LAMMPS timestep.

Image-coordinate provenance distinguishes the selected raw rows used to
construct a coordinate from all raw tag-count opportunities represented by
that coordinate. This prevents the dropped `s=1` alias from being mistaken for
a uniquely pure `s=0` sector.

## Radial and angular source

The certified one-neighbor source is

```text
phi_(zeta,q,l,m)(r) = 1_zeta N_(q,l) x^l (1-x)^2
                      P_q^(4,2l+2)(2x-1) C_(l,m)(r_hat),
x = r/r_c, 0 < r < r_c.
```

`P_q^(alpha,beta)` is a Jacobi polynomial and `C_(l,m)` is a
Racah-normalized Condon--Shortley spherical harmonic. Jacobi was chosen here
because the declared radial inner product and product linearization are exact.
This is not a claim that Jacobi evaluation is always faster than Chebyshev.
A Chebyshev realization of the same finite source space can be reorthogonalized
offline and used with its forward transform and transpose adjoint; that remains
an explicit performance task.

Same-neighbor angular products reuse YE3T's generalized Clebsch--Gordan
machinery,

```text
C_(l1,m1) C_(l2,m2)
  = sum_(L,M) <l1 0,l2 0|L 0><l1 m1,l2 m2|L M>C_(L,M),
```

so there is no separate Gaunt-coefficient convention or compiler. Products of
Clebsch--Gordan coefficients are referred to here as generalized
Clebsch--Gordan coefficients, following the project convention.

Exact overlap is outside the current source domain. The cutoff value and first
radial derivative vanish, but Cartesian `C1` regularity at `r=0` is not claimed
for every channel. Runtime overlap rejection and the separately fitted ZBL
reference are therefore explicit validation requirements.
The native CPU and Kokkos spherical-harmonic realization uses
`epsilon = 1e-12 Angstrom`. Exact Python/native source-and-derivative parity is
therefore certified only for `r > 1e-12 Angstrom`; `r=0` is rejected and
`0 < r <= 1e-12 Angstrom` remains unqualified.

### Exact source versus numerical realization

The exact Jacobi source definition, its product linearization, and the
physical-image coupling certificate do not prescribe an unstable floating-
point polynomial expansion. Source-plan V1 evaluated the exact expanded power
coefficients with Horner's rule. That remains readable for byte-compatible
legacy replay, but products of requested degrees `q=0,...,7` close through
`q=18`; the resulting power coefficients and cancellation make V1 unsuitable
for the fitted high-degree gate.

Source-plan V2 is the default numerical realization. It evaluates
`P_q^(4,2l+2)(2x-1)` and `d/dx` with the differentiated three-term Jacobi
recurrence, once per distinct edge and `l` ladder on CPU. The expanded exact
coefficients remain in the compiler artifact for provenance and independent
checking, but are not used by the V2 timestep evaluator. Python, native CPU,
and the experimental Kokkos correctness path consume the same recurrence and
source normalization. Because numerical realization is part of the
source-plan hash, adopting V2 invalidates only floating geometry rows; the
exact coupling and physical-image caches remain reusable.

## Fitting and deployment contract

The fit uses DFT energies and forces minus an executable-backed LAMMPS ZBL
reference. The exact reference metadata and residual target hashes are bound
into the normal-equation identity and exported fit provenance. The V3 model
contains only the learned residual, so physical deployment must use the same
ZBL component through `pair_style hybrid/overlay`.

Target-free descriptor/force rows are keyed by exact geometry, compiler
artifact, source plan, type map, and force convention. Ridge regularization is
identity `L2` in the compiler-orthogonal coordinates. Alpha selection uses the
declared validation scales; no virial/stress target is fitted. The test
partition is not accessed by this diagnostic.

## References

- R. Drautz, *Atomic cluster expansion for accurate and transferable
  interatomic potentials*, Phys. Rev. B 99, 014104 (2019).
- G. Dusson et al., *Atomic cluster expansion: completeness, efficiency and
  stability*, J. Comput. Phys. 454, 110946 (2022).
- A. P. Yutsis, I. B. Levinson, V. V. Vanagas, *Mathematical Apparatus of the
  Theory of Angular Momentum* (1962), for generalized angular-momentum
  coupling and graphical calculus, not the radial basis.
- NIST DLMF Sections 18.1 and 18.3 for Jacobi polynomials and orthogonality,
  and Section 34.3(vii) for spherical-harmonic products.
- C. H. Ho, T. S. Gutleb, C. Ortner, *Atomic cluster expansion without
  self-interaction*, J. Comput. Phys. 515, 113271 (2024). Purification is cited
  for deferred context and is not implemented or claimed in this study.
