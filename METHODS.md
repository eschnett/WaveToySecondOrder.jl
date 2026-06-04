# Numerical methods

This file records the numerical choices made in WaveToySecondOrder.
It documents the 1D implementation in full; the 2D implementation
mirrors it on axis-aligned affine meshes (`make_uniform_quad`) — see
the "2D" note at the end. 3D is not yet migrated; the old
second-order-in-time 3D path remains.

## 2D (summary)

The 2D scalar wave on a 2+1 ADM background reuses the entire 1D design
on the per-axis first-derivative operator `HexSBPSAT.apply_D!(·, d)`
(reference SBP-G along reference axis `d` + centred-flux SAT; on
axis-aligned affine meshes `H·D` is exactly skew, `D_1d ⊗ I`).

* Conservative system: `Π = (√γ/α)(∂_tΦ − βⁱ∂_iΦ)`,
  `∂_tΦ = βⁱ∂_iΦ + (α/√γ)Π`, `∂_tΠ = ∂_i(βⁱΠ + α√γ γ^{ij}∂_jΦ)`. The
  gradient `∂_jΦ` and the flux divergence `∂_iFⁱ` both go through
  `apply_D!(·, d)`; coefficient fields per node are `α, √γ, βⁱ,
  γ^{ij}` (`src/wave2d_curved.jl`, `wave2d_curved_rhs!`). KO `ε·μ⁻⁵·D⁶`
  applied per axis.
* Backgrounds: `Background2D` / `AnalyticBackground2D` /
  `MetricBackground2D` (SpacetimeMetrics `adm_decompose`, (x,y)
  sub-block of the 3-metric); `sample_background2d!` is one KA kernel.
* Boundaries (`src/boundaries2d.jl`): each axis-aligned face is
  classified from its *normal* speeds `c± = −β^n ± a_n`
  (`β^n = n̂·β^{dₙ}`, `a_n = α√γ^{dₙdₙ}`) — the 1D analysis in the
  normal direction. Same admissibility rules and the same
  characteristic-free field-radiation SAT (residual
  `r = Π + ((β^n+a_n)/a)·n̂·∂_nΦ`, penalty on Π̇, σ=1); excision and
  full-state Dirichlet as in 1D. Confirmed energy-stable by the
  dense-operator spectrum tests (flat / small shift / anisotropic γ).
  This axis-aligned rectangular boundary pass (`_apply_bc2d!`), like the
  curvilinear one below, is a single KernelAbstractions kernel run on
  both CPU and GPU (parallelised over output nodes, race-free at
  corners).
* Driver `evolve2d` and app `bin/wave2d.jl` mirror the 1D versions
  (first-order ODEProblem, `pick_integrator_first_order`, energy/L²
  monitoring, `bc ∈ {:periodic, :auto, 4-tuple}`). Backgrounds:
  `:minkowski`, `:constant_shift`, `:gaugewave`.
* Type/backend matrix as in 1D: Float64/Float32 CPU+CUDA, Float32
  Metal, Float64x2 CPU; the kernel and `apply_D!` have GPU paths
  (verified CPU↔Metal Float32).
* **Curvilinear (cubed-square)**: free-stream-preserving conservative
  first derivative (`HexSBPSAT.make_metric_terms2d` +
  `apply_gradient2d!`/`apply_divergence2d!`). Metric terms are computed
  *discretely* from the nodal coordinates (`make_geometry`'s analytic
  Jacobians fail the discrete metric identities); in 2D the identities
  `Σ_α D̂_α(aₐ^α)=0` then hold automatically (tensor-product SBP ops
  commute), giving free-stream by construction (∇const, ∇·const ≈
  1e-14). The operators use the **split (skew-symmetric) form**
  ½(conservative + advective) — the pure conservative form's adjoint is
  the advective form, so only the split form gives a skew-adjoint
  gradient/divergence pair (verified: interior `H_d·D` skew to ~1e-15;
  the consistent discrete mass `H_d = H_ref·detJ` is the energy norm).
  The KO term stays per-axis. The closed curved domain is unstable
  (uncancelled boundary term, max Re λ ≈ +0.87); the **physical-normal
  Sommerfeld outer BC** (`_apply_bc2d_curv!`, normal from the analytic
  Jacobian columns × handedness — the handedness factor is essential
  where detJ<0, as on half the wedges) restores stability (spectrum
  max Re λ ≤ round-off at M=2,4). Outer BCs: Sommerfeld (absorbing,
  default) and **Dirichlet** — the curved field-radiation Dirichlet
  forms its target from exact-solution boundary data (Π and ∇Φ) using
  the pass's own physical normal; at β=0 it is exact. Verified by a
  **convergence test against an analytic plane wave** on the
  cubed-square (L2 error vs exact converges at rate ~2, capped by the
  one-sided boundary). `evolve2d(mesh_kind=:cubed_square|:inflated_square, bc=
  :sommerfeld|:dirichlet)`, `bin/wave2d.jl --mesh …`. The
  **inflated-square** mesh (inner square + inflation + shell patches)
  carries non-zero connectivity orientation {0,1} between patches, so
  it exercises the SAT's `_neigh_p` orientation transform — free-stream,
  interior skew-adjointness, Sommerfeld-spectrum stability, and
  analytic plane-wave convergence all hold there with no operator
  change. **GPU / kernel structure**: every SBP-SAT operator
  (`apply_gradient2d!`/`apply_divergence2d!`, the affine `apply_D!`, and
  the 2D/3D Laplacian) is a *single* set of KernelAbstractions kernels
  run on both CPU and GPU — there is no separate hand-written CPU loop;
  the CPU backend exists to test the GPU path. The gradient/divergence
  use the **two-pass** structure (shared with the Laplacian): a gather
  kernel writes each element's face-node values into the mesh-workspace
  face trace, then a workgroup-per-element kernel stages the field into
  shared memory, does the split-form volume reduction, and applies the
  centred-flux SAT by reading the *neighbour's* gathered trace
  (orientation via `_neigh_p`). One write per output node (gather, not
  scatter) — no races; the two launches give the global barrier. The
  scalar/vector face gathers (`_gather_face2d_1ch!`/`_2ch!`) are shared
  with `apply_D!`. The physical-normal boundary pass
  (`_apply_bc2d_curv!`) is one node-parallel kernel (corner nodes
  race-free). A full cubed-square evolution (gradient/divergence, flux,
  per-axis KO, Sommerfeld/Dirichlet BC) runs entirely on-device.
  Verified CPU↔Metal Float32 (operator agreement on both curvilinear
  meshes; short curved-Sommerfeld evolution to 1e-3). The 3D
  curvilinear case (which needs the harder conservative-curl metric
  form) remains out of scope. **Curvilinear test parity** with the
  affine path: robust stability under √eps noise (Sommerfeld + KO),
  energy non-increasing under the absorbing boundary, and a variable
  background (gaugewave, varying lapse) with curved Dirichlet — all on
  cubed-square.
* **2D annulus + inner excision (BH-excision model).** `mesh_kind =
  :annulus` (`HexMeshes.make_annulus_mesh`) is a pure 4-patch shell ring
  `R1 ≤ |x| ≤ R2` — the 2D analog of the 3D `make_radial_shell_mesh` —
  with distinct inner/outer boundary tags (inner `:excision`→8, outer
  `:sommerfeld`→7). The curvilinear BC pass gives **no SAT** (pure
  outflow) to any face whose mesh tag is the excision tag (`make_bc2d`
  `excision_tag`); excision is declared by the mesh, not auto-classified
  by speed. The `:radial_shift` background (flat α, γ; outward radial
  shift ramping linearly from `V > 1` at `R1` to `< 0.1` at `R2`) makes
  the inner circle superluminal-outflow and the outer subluminal. The
  radial characteristic speeds are `dr/dt = −(β_r ± a)` (this solver
  advects with `+βⁱ∂_iΦ`, `a = α√γ^rr`); at `R1` both are `< 0` for
  `β_r > 1` (both characteristics fall into the hole) ⇒ the inner circle
  is **superluminal outflow**, correctly handled by **excision** (no
  SAT); the outer circle is subluminal ⇒ Sommerfeld. (The opposite sign,
  `β_r < −1`, would be superluminal *inflow* / full-Dirichlet — out of
  scope.) The shift is a **linear** radial ramp so it is well resolved on
  the grid; a steep `1/r²` profile would be under-resolved at these
  resolutions and is not an appropriate test. The RHS spectrum is stable
  (max Re λ ≤ round-off at `V = 1.2`) and a noisy annulus evolution stays
  bounded with non-increasing, decaying energy (`evolve2d`,
  `bin/wave2d.jl --mesh annulus`).
* **Deferred** (same as 1D's open items, plus): subluminal Dirichlet
  data-injection in the 2D driver (Sommerfeld is the radiative
  default); per-face characteristic-speed auto-classification on curved
  faces (excision is declared by the mesh tag instead).

## Scope (1D)

Scalar wave equation on a prescribed 1+1 ADM background with
space- and time-varying lapse α(t,x), shift βˣ(t,x), and spatial
metric γ_xx(t,x). Superluminal shift (|β| > α/√γ) is supported.
Boundaries: periodic, plus radiative Sommerfeld / Dirichlet (a
characteristic-free field-radiation SAT, for small shift), excision,
and full-state Dirichlet, validated against each face's characteristic
class (see Boundary conditions below).

## Continuous formulation

ADM line element: `ds² = −α² dt² + γ_xx (dx + βˣ dt)²`.

State: `(Φ, Π)` with the densitised momentum

    Π := (√γ_xx / α) · (∂_t Φ − βˣ ∂_x Φ)  =  √γ · n^μ ∂_μ Φ,

`n` the unit normal of the foliation. The covariant wave equation
`∂_μ(√|g| g^{μν} ∂_ν Φ) = 0` (with `√|g| = α √γ_xx`) becomes the
flux-conservative first-order (in time) system

    ∂_t Φ = βˣ ∂_x Φ + a · Π
    ∂_t Π = ∂_x ( a · ∂_x Φ + βˣ · Π ),       a := α / √γ_xx.

The single `∂_x` of the combined flux supplies both the `β ∂_x Π`
advection and the `(∂_x β) Π` source of a non-conservative form. Only
the two coefficient fields `a` and `β` enter the evolution; α and √γ
are needed individually only for initial data and diagnostics. For
α = γ = 1, β = const this is the textbook constant-shift wave
equation; the flat wave equation is the special case α = γ = 1, β = 0
(there is no separate flat-wave code path).

Energy monitor:

    E = ∫ ½ [ (Π/√γ)² + (∂_x Φ)²/γ ] · √γ dx,

discretised with the physical mass weights `Hphys`. E is exactly
conserved by the continuum system only for static backgrounds; for
time-periodic backgrounds (gauge wave, sine shift) it returns to its
initial value after each period, which the tests assert.

## Discretization

* Mesh: `HexMeshes.Mesh{1}` from `make_uniform_line(T, M, x0, x1;
  periodic = true)` — M affine line elements, ring connectivity.
* Reference element: N Gauss–Lobatto–Legendre nodes
  (`HexSBPSAT.make_element`), SBP operators from `make_operators`
  (`SBPOps`: G, H, …). N = 4 and N = 8 are the GPU-targeted sizes.
* Geometry: `MeshGeometry{1}` (`make_geometry`) — per-node coords,
  Jacobian (= element width), `Hphys[i,e] = H_ref[i]·|J|`.
* First derivative: `HexSBPSAT.apply_D!` — ONE consistent operator
  used everywhere: reference SBP-G plus a centred-flux SAT at every
  interior face with coefficient `1/(2·Hphys_face)`, neighbour
  relation read from `MeshConnectivity` (not hardwired `m ± 1`).
  With this SAT the assembled `H·D` is **exactly skew**
  (`H·D + (H·D)ᵀ = 0`, asserted in HexSBPSAT's tests), so the
  semidiscrete wave operator has a purely imaginary spectrum up to
  background variation. Faces with `bdry ≠ 0` currently get no SAT
  (one-sided derivative) — the hook for Dirichlet/Sommerfeld SATs.
* RHS (`wave1d_curved_rhs!` in `src/wave1d.jl`):
  `DΦ = D Φ; F = a·DΦ + β·Π; Π̇ = D F; Φ̇ = β·DΦ + a·Π`. All scratch
  lives in a preallocated `Wave1DWorkspace`; the kernel is
  allocation-free per call.
* Kreiss–Oliger dissipation: `u̇ += ε_KO · μ⁻⁵ · D⁶ u` (p = 2)
  applied to both Φ and Π, implemented as six `apply_D!` passes; μ is
  the spectral radius of the assembled D, computed once per
  `Wave1DWorkspace` by power iteration on D² (D is skew-like with
  eigenvalue pairs ±iμₖ, so D² has real spectrum −μₖ² and plain power
  iteration converges; deterministic alternating-sign start vector,
  ~30 iterations, accurate to a few %). Since D is skew, D⁶ is
  negative semidefinite (dissipative). On smooth data the term is
  `ε·μ⁻⁵·k⁶ = O(h^{2p−1})·k⁶` (μ ~ 1/h) → does not degrade the
  formal order.

  **Normalisation.** The μ⁻⁵ scaling pins the highest-mode damping
  rate to `λ_KO = ε·μ` — the same magnitude as the wave operator — so
  ε is the standard dimensionless NR coefficient and the KO term does
  not tighten the CFL limit for ε ≤ 1 (`evolve1d` still checks
  `dt ≤ 1.4/(ε·μ)` exactly). This mirrors the `1/2^{2p+2}` factor in
  the classic finite-difference KO operator, which serves exactly
  this purpose on uniform grids. The naive scaling `ε·h⁵·D⁶` with the
  *element* width h is over-strong by `(h·μ)⁵` — measured
  `λ_KO/(ε·λ_wave) ≈ 6.2·10⁴` at N = 4 and `7.1·10⁷` at N = 8
  (growing like ~N¹⁰), which made the nominal ε = 0.1 require a dt
  thousands to millions of times below the wave CFL.

  **When is KO needed?** Only at sonic horizons. Spectrum evidence
  (N = 4, M = 8): constant β and smooth subluminal variable β give
  max Re(λ) at round-off (H·D skew); variable β crossing |β| = α/√γ
  gives max Re(λ) = +0.42 without dissipation and ≤ 0 with ε = 0.1
  (ε = 0.05 is marginal: +0.002). Defaults: `ε_KO = 0` for
  convergence/energy tests, `ε_KO = 0.1` for the sonic/superluminal
  noise stress tests; the spectrum test asserts both directions.

  **Alternatives considered (deferred).** (a) Upwind/characteristic
  interface SAT instead of centred flux + KO: dissipation acts only
  on inter-element jumps (O(h^N) for resolved solutions), no CFL
  penalty, natural at sonic points since the splitting follows the
  characteristic speeds −β ± α/√γ; trades exact H·D skewness for
  provable energy decay. The leading candidate for the 2D/3D
  rebuild. (b) Per-element modal exponential filter applied
  post-step: zero CFL cost and very GPU-friendly, but lives outside
  the RHS, so RHS-spectrum analysis no longer captures the
  stabilised scheme. Both deferred to keep the exactly-skew operator
  and the spectrum-based test methodology.

## Background sampling

`Background1D` (in `src/wave1d.jl`) supplies (α, βˣ, γ_xx) on the
collocation points at every integrator stage time via
`sample_background!`, a KernelAbstractions kernel (CPU and GPU, one
code path, allocation-free):

* `MetricBackground1D(m)` — m an `SpacetimeMetrics.AbstractMetric`;
  ADM variables via `SpacetimeMetrics.adm_decompose(m, (t,x,0,0))`.
  Built-in metric backgrounds: `Minkowski()`, `GaugeWave(A, d)`
  (AwA gauge wave: α = √H, β = 0, γ = H — note a = α/√γ = 1, so in
  1+1 it does not exercise the variable-coefficient paths), and
  `SineShift(A, d)` (α = 1, β = −Ac/(1+Ac), γ = (1+Ac)²,
  c = cos(2π(x−t)/d) — genuinely varying coefficients, flat
  curvature, exact solution known).
* `AnalyticBackground1D(α_fn, β_fn, γ_fn)` — closures `(t, x) → value`;
  used for superluminal-shift stress tests (a coordinate effect no
  Lorentz boost can produce) and for number types without reliable
  transcendental functions (MultiFloats). Closures must not capture
  `Type` objects if the background is to be passed into GPU kernels.

## Time integration

First-order `ODEProblem` on `ArrayPartition(Φ, Π)`
(RecursiveArrayTools), explicit RK from OrdinaryDiffEq subpackages —
the variable-β system is not Hamiltonian, so symplectic integrators
are not used in 1D:

| element N | integrator | order |
|-----------|-----------|-------|
| ≤ 4       | `RK4()`   | 4     |
| 5–6       | `Tsit5()` | 5     |
| ≥ 7       | `Vern7()` | 7     |

(`pick_integrator_first_order` in `src/evolve.jl`.) Fixed step
`dt = cfl·dx_min/max_speed` with `max_speed = max(|β| + α/√γ)`
(coordinate characteristic speeds are `−β ± α/√γ`), tightened by the
KO limit when `ε_KO ≠ 0`; `adaptive = false` for reproducible
convergence measurements. The 2D/3D drivers still use the older
`SecondOrderODEProblem` + symplectic path (`pick_integrator`) pending
their rebuild.

## GPU / precision

Kernels are type- and backend-agnostic via KernelAbstractions:

* Float64, Float32 on CPU and CUDA; **Float32 only on Metal**.
* `Float64x2`/`Float32x2` (MultiFloats): CPU only, with
  `AnalyticBackground1D` (MultiFloat trig is unreliable; build trig
  ICs in Float64 and convert).
* `apply_D!` dispatches on `get_backend`: SVector per-element loop on
  CPU, workgroup-per-element `@kernel` with `@localmem` staging on
  GPU. `SBPOps` (SMatrix/SDiagonal fields, N ≤ 8) is isbits and passes
  through Adapt unchanged. Device migration: `to_device(mesh|geom,
  backend)`.
* Expect CPU↔GPU differences at the level of Float32 round-off; the
  KO D⁶ chain amplifies this strongly on rough data (different
  summation order), so GPU comparisons use smooth data.

## Boundary conditions

Implemented: periodic (ring connectivity in `Mesh{1}`) plus four
outer-boundary conditions on non-periodic meshes (`src/boundaries1d.jl`;
`make_uniform_line(...; periodic = false)` tags the −x/+x faces 1/2).

**Design intent — no eigenvector projection at radiative faces.** The
package is a testbed for the Einstein equations, where the
characteristic *eigenvectors* are metric/gauge-dependent and
expensive. The radiative BCs here deliberately avoid eigenvector
projection: they use only the characteristic *speeds* (eigenvalues),
which classify faces and set penalty magnitudes and are unavoidable
for any open boundary. The propagation speeds are `s_R = a − β`,
`s_L = −a − β` (`a = α/√γ`); a mode is outgoing at a face with
outward normal n̂ iff `s·n̂ > 0`. Face classes (`classify_face1d`):
**subluminal** (`|β| < a`, one in / one out), **superluminal outflow**
(both out), **superluminal inflow** (both in), **sonic** (`|β| ≈ a`
within `eps^(1/4)·a` — always an error: a vanishing speed leaves a
mode undetermined).

**Admissible conditions** (validated at setup *and* re-checked at
every stage time — time-dependent backgrounds may not change a face's
class mid-run; `validate_bc1d` throws otherwise):

* Subluminal → `:sommerfeld` (radiative/absorbing) or `:dirichlet`
  (data injection). Both are intended for **small shift** (`|β| ≲
  0.1`); see the field-radiation SAT below.
* Superluminal outflow → `:excision`: no boundary term at all — the
  one-sided `apply_D!` rows (no SAT at `bdry ≠ 0` faces) are already
  the correct outflow treatment.
* Superluminal inflow → `:full_dirichlet`: both modes enter, so the
  full state (Φ, Π) is pinned to data.

**Field-radiation SAT** (subluminal faces; a 2-node post-pass in
`wave1d_curved_rhs!` after the bulk + KO passes; HexSBPSAT stays
equation-agnostic). Instead of projecting onto the eigenvector
`∂_xΦ ∓ Π`, impose the scalar radiation condition on the *field*,
`∂_tΦ + a·n̂·∂_xΦ = (data rate)`, rewritten via `∂_tΦ = β∂_xΦ + aΠ`
and divided by `a` into the normalised residual

    r := Π + (n̂ + β/a)·∂_xΦ.

At β = 0, `r` is exactly the incoming characteristic, so the penalty
coincides with the textbook Sommerfeld SAT; for β ≠ 0 it differs by
`O(β)·∂_xΦ`. The penalty is `Π̇ += −σ·|s_in|/Hf·(r − g)` with
`Hf = Hphys[face]`, `|s_in| = a + n̂·β`, **σ = 1** (σ = 1/2 is
marginally unstable — the one-sided bulk operator leaves the full
boundary flux for the penalty to cancel). `:sommerfeld` uses `g = 0`
(absorbing); `:dirichlet` uses `g = r` evaluated on the boundary data
(incoming wave injected, outgoing wave free). Energy:
`dE/dt = −¼Σ(s·n̂)u²` per mode — outgoing modes drain, the penalty
controls the ingoing injection.

This is the NR-standard "apply `∂_t f + ∂_r f = 0` to each evolved
field" outer condition: it ports field-by-field to Einstein with no
eigendecomposition. **Properties / limits** (all confirmed by the
spectrum and convergence tests): stable for `|β| ≲ 0.1` (a
perturbation of the proven β = 0 operator; out of policy it is mildly
unstable, e.g. max Re(λ) ≈ +0.02 at β = 0.5); **exact** (spectrally
convergent) only at β = 0 — for β ≠ 0 there is an `O(β)` spurious
reflection floor; not constraint-preserving. Perfectly-absorbing or
constraint-preserving boundaries would require the eigenvector
projection and are out of scope for the testbed.

* Full-state Dirichlet (superluminal inflow): `Φ̇ += −τ/Hf(Φ−g_Φ)`,
  `Π̇ += −τ/Hf(Π−g_Π)` with `τ = σ(|s_R|+|s_L|)`; no characteristics
  (the whole state is pinned). Observed ≈ 2nd-order accurate at the
  boundary (vs. spectral interior) — acceptable for an inflow pin.

**Known limitation** (genuine physics, not a SAT defect): strongly
space-varying *superluminal* β on an open domain produces operator
eigenvalues with `0 < Re(λ) ≤ max|∂_xβ|` (compression amplification;
present already with pure excision and no penalties, absent on
periodic meshes where modes recirculate through the exactly-skew
operator; KO does not help — the growing mode is smooth). The tests
assert the continuum bound.

`evolve1d(; bc = ...)` accepts `:periodic`, `:auto` (classify per
face and pick the natural admissible condition; subluminal gets
Dirichlet-with-exact-data for `ic = :exact`, Sommerfeld for
`ic = :noise`), or `(left = :sym, right = :sym)`. The app exposes
`--bc`, `--bc-left`, `--bc-right`.

## Tests

`test/test_wave1d.jl` (kernel) and `test/test_evolve1d.jl` (driver);
operator-level identities live in `HexSBPSAT/test/test_apply_D1d.jl`,
`adm_decompose`/`SineShift` in SpacetimeMetrics' tests. Coverage:

1. **Spectrum**: column-probed RHS operator; `max Re(λ)` ≤ eigensolver
   round-off (`1e-5·|λ|_max`) for all stable configurations
   (constant β ∈ {0, 0.5, 2}, variable subluminal; sonic/superluminal
   variable β with ε_KO = 0.1), and a control asserting the sonic
   case **is** unstable with ε_KO = 0.
2. **Noise robustness**: √eps noise, 50 light-crossings, six shift
   configurations, boundedness asserted.
3. **Convergence**: plane wave (β = 0.5, rate ≳ 2.5×/doubling at
   N = 4), gauge wave (varying lapse), sine shift (variable β, γ;
   rate > 2, geometric mean > 4×).
4. **Energy**: drift < 1e-3 after one background period (sine shift,
   finest M); also reported by `evolve1d` and plotted by the app.
5. **Types/backends**: Float64x2 trajectory agrees with Float64 to
   < 1e-12; Metal Float32 run agrees with CPU Float32 to 1e-3
   (auto-skipped without hardware).
6. **Boundary conditions**: face classification + admissibility
   (`@test_throws` for every inappropriate combination); spectra of
   all admissible configs ≤ round-off within the `|β| ≤ 0.1` radiative
   policy (and the strongly-varying superluminal control within its
   continuum bound); convergence at β = 0 for travelling-wave
   Dirichlet→Sommerfeld and standing-wave radiation-data Dirichlet,
   plus superluminal advection excision/full-Dirichlet; the small-shift
   `O(β)` reflection-floor test (β ∈ {0.05, 0.1}); Sommerfeld
   pulse-exit energy absorption (E_final/E_0 < 1e-4, monotone decay);
   noise stability per BC regime; driver-level `bc` kwarg tests incl.
   `:auto`.

Each testset runs in seconds (full 1D set ≈ 15 s; Metal adds ≈ 30 s).

## Apps

`bin/wave1d.jl` — CLI app (flags for N, M, background, IC, ε_KO, FP
type, backend, output path) producing a ≤ 800 px four-panel figure:
grid structure, initial vs final Φ, total energy vs t, L² error vs t;
PNG via CairoMakie plus Sixel terminal display.
