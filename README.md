# WaveToySecondOrder.jl

Solve the scalar wave equation using a PDE formulation that is
second order in space.

## Numerical approach

### SAT boundary treatment

The element-local operators ($G$, $H$, $D$, $L = D \cdot G$) satisfy the
discrete summation-by-parts (SBP) identity

$$
H \, D + (H \, G)^{\top} \;=\; B,
$$

the discrete analogue of integration by parts
$\int v \, u' = [v \, u] - \int v' \, u$.
The boundary operator $B = \mathrm{diag}(-1, 0, \ldots, 0, +1)$ carries the
outward normal, so $B \, u$ evaluates $n \cdot u$ at the element edges only.

Boundary conditions and inter-element coupling are imposed *weakly* via
simultaneous-approximation-term (SAT) penalties, originally introduced by
Carpenter, Gottlieb & Abarbanel (1994) for first-order systems. The penalty
modifies the right-hand side rather than the operator's structure, so the
SBP property — and the discrete energy estimate it implies — is preserved.

#### The complication at second order

For first-order hyperbolic systems the SBP-SAT theory is mature: one
characteristic per incoming direction, one scalar penalty per face.
Second-order operators behave differently. From
$H \cdot L = B \cdot G - G^{\top} \cdot H \cdot G$,
the discrete energy is

$$
u^{\top} \, H \, L \, u
\;=\;
u_R \, (G u)_R \;-\; u_L \, (G u)_L \;-\; \lVert G u \rVert_H^{\,2}.
$$

The boundary term mixes the field value $u_b$ with its normal derivative
$(G u)_b$, so a SAT that exchanges only $u$ (Dirichlet-style) underconstrains
the interface: continuous-but-kinked piecewise-linear functions satisfy
$L \cdot u = 0$ and have $[u] = 0$, so they sit in the kernel of the global
operator. With $M$ elements this is an $(M-1)$-dimensional null space of
"kink modes" that gives $u(t) \sim a + b \, t$ growth when excited by the
wave equation $u_{tt} = L \, u$. Symmetrically, a Neumann-style coupling
that exchanges only $G u$ admits an $M$-dimensional step-function kernel.

Mattsson (2003) showed that energy-stable SAT for second-derivative SBP
operators requires *both* the value jump and the gradient trace at each
face. The clean symmetric choice is the SIPG (symmetric interior penalty
Galerkin) form (Arnold, Brezzi, Cockburn & Marini, 2002), discretized in
SBP language:

$$
L_{\mathrm{SAT}} \, u
\;=\;
L \, u
\;+\;
H^{-1} \, \Bigl[\;
  \alpha \, G^{\top} B \, \Delta u
  \;-\; \tfrac{1}{2}\, B \, \Delta G u
  \;-\; \tau \, |B| \, \Delta u
\;\Bigr],
$$

with $\Delta u = u - u_{\mathrm{neighbour}}$,
$\Delta G u = G u - G u_{\mathrm{neighbour}}$, and per-face weight $\alpha$.
At outer Dirichlet faces only one element is present, so $\alpha = 1$
(full Nitsche-Dirichlet); at interior interfaces $\alpha = \tfrac{1}{2}$
because the SIPG average $\{G u\}$ is shared between the two adjacent
elements. The mirror convention $g_{Gu} = G u_{\mathrm{local}}$ is used at
outer faces, making $\Delta G u = 0$ there.

For $\tau$ above a stability threshold ($\tau \gtrsim 17$ in unit-element
coordinates here), $H \cdot L_{\mathrm{SAT}}$ is symmetric and
negative-definite — the kink null space is gone, eigenvalues are real and
$\le 0$, and $u_{tt} = L_{\mathrm{SAT}} \, u$ is genuinely strictly stable
in the discrete $H$-norm.

#### References

- Carpenter, Gottlieb & Abarbanel (1994), *Time-Stable Boundary Conditions
  for Finite-Difference Schemes Solving Hyperbolic Systems: Methodology and
  Application to High-Order Compact Schemes*, J. Comput. Phys. **111**,
  220–236.
  doi: [10.1006/jcph.1994.1057](https://doi.org/10.1006/jcph.1994.1057).
- Mattsson (2003), *Boundary Procedures for Summation-by-Parts Operators*,
  J. Sci. Comput. **18**, 133–153.
  doi: [10.1023/A:1020342429644](https://doi.org/10.1023/A:1020342429644).
- Mattsson & Nordström (2004), *Summation by Parts Operators for Finite
  Difference Approximations of Second Derivatives*, J. Comput. Phys. **199**,
  503–540.
  doi: [10.1016/j.jcp.2004.03.001](https://doi.org/10.1016/j.jcp.2004.03.001).
- Arnold, Brezzi, Cockburn & Marini (2002), *Unified Analysis of
  Discontinuous Galerkin Methods for Elliptic Problems*, SIAM J. Numer.
  Anal. **39**, 1749–1779.
  doi: [10.1137/S0036142901384162](https://doi.org/10.1137/S0036142901384162).
- Lindblom, Scheel, Kidder, Owen & Rinne (2006), *A New Generalized Harmonic
  Evolution System*, Class. Quantum Grav. **23**, S447–S462.
  doi: [10.1088/0264-9381/23/16/S09](https://doi.org/10.1088/0264-9381/23/16/S09),
  arXiv: [gr-qc/0512093](https://arxiv.org/abs/gr-qc/0512093).
