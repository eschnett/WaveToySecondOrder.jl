# Numerical methods

This file records the numerical choices made in WaveToySecondOrder.
It currently covers the 1D implementation; 2D/3D will be migrated to
the same scheme and documented here as they are rebuilt.

## Scope (1D)

Scalar wave equation on a prescribed 1+1 ADM background with
space- and time-varying lapse α(t,x), shift βˣ(t,x), and spatial
metric γ_xx(t,x). Superluminal shift (|β| > α/√γ) is supported;
periodic boundaries only (see Boundary conditions below).

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
* Kreiss–Oliger dissipation: `u̇ += ε_KO · h⁵ · D⁶ u` (p = 2) applied
  to both Φ and Π, implemented as six `apply_D!` passes. Since D is
  skew, D⁶ is negative semidefinite (dissipative). On smooth data the
  term is `O(h^{2p−1}) = O(h³)` → does not degrade the formal order.
  Defaults: `ε_KO = 0` for convergence/energy tests; `ε_KO = 1e-4`
  for the noise stress tests with variable superluminal/sonic β
  (which are genuinely unstable without it — the spectrum test
  asserts both directions). NOTE: the KO term has its own, much
  tighter CFL limit, ≈ `2.8/(ε_KO·h⁵·μ⁶)` with `μ ≈ 2.5/dx_min`;
  `evolve1d` enforces it automatically.

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

* **Implemented: periodic** (ring connectivity in `Mesh{1}`).
* Planned (in order): Dirichlet SAT, then excision boundaries, then
  Sommerfeld SAT. Dirichlet and Sommerfeld are only meaningful for
  subluminal shift at the boundary; for superluminal shift all
  characteristics leave the domain on one side — use excision/outflow
  there (no SAT — exactly what `apply_D!` already does at `bdry ≠ 0`
  faces). Entry points: `make_uniform_line(...; periodic = false)`
  boundary tags 1/2 and the `bdry`-gated SAT in `apply_D!`.

## Tests

`test/test_wave1d.jl` (kernel) and `test/test_evolve1d.jl` (driver);
operator-level identities live in `HexSBPSAT/test/test_apply_D1d.jl`,
`adm_decompose`/`SineShift` in SpacetimeMetrics' tests. Coverage:

1. **Spectrum**: column-probed RHS operator; `max Re(λ)` ≤ eigensolver
   round-off (`1e-5·|λ|_max`) for all stable configurations
   (constant β ∈ {0, 0.5, 2}, variable subluminal; sonic/superluminal
   variable β with ε_KO = 1e-4), and a control asserting the sonic
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

Each testset runs in seconds (full 1D set ≈ 15 s; Metal adds ≈ 30 s).

## Apps

`bin/wave1d.jl` — CLI app (flags for N, M, background, IC, ε_KO, FP
type, backend, output path) producing a ≤ 800 px four-panel figure:
grid structure, initial vs final Φ, total energy vs t, L² error vs t;
PNG via CairoMakie plus Sixel terminal display.
