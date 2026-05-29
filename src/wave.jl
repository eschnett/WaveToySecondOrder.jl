# Wave-equation-specific layer on top of HexSBPSAT's generic 3D Laplacian
# `apply_laplacian3d!`. What lives here:
#
#   * `Params3d{T}` — the wave-equation parameter bundle (`A, k, ω, τ,
#     bdry_values`) used as the `p` argument of an
#     `OrdinaryDiffEq.SecondOrderODEProblem`.
#   * `initialize!` — 1D sin·cos IC used by the 1D test driver.
#   * `initialize3d!` — sin·sin·sin·cos Cartesian IC.
#   * `eigenmode_cartesian!`, `eigenmode_radial!`, `eigenmode_quadrupole!`
#     — three families of exact eigenmodes of `ü = ∇² u` (Dirichlet cube,
#     radial Bessel ball, quadrupole ball). Used both as IC and as the
#     analytic reference at each diagnostic sample.
#   * `outgoing_pulse!` — outgoing spherical Gaussian pulse, the canonical
#     IC for testing Sommerfeld outer BCs.
#   * `_SPHERICAL_BESSEL_J2_ZEROS`, `_j2_over_x2`, `_sinhc` — helpers
#     backing the quadrupole eigenmode and the outgoing pulse.
#   * `rhs_wave3d!` — the wave-equation RHS. Calls the equation-agnostic
#     `apply_laplacian3d!` for the bulk spatial operator, then adds a
#     dissipative SAT pass over outer faces tagged `7` to enforce
#     Sommerfeld radiative BCs.
#   * `recommended_dt` — Störmer–Verlet timestep limit
#     `cfl · 2/√|λ_max|`, wave-equation-specific.

const SOMMERFELD_BDRY_TAG = Int8(7)

################################################################################
# 1D wave-equation IC (used by the 1D test driver in `test/test_kernels1d.jl`).

function initialize!(u::AbstractVector, u̇::AbstractVector, x::AbstractVector, t;
                     A, k, ω)
    u .=  A   * sin.(k*x) * cos(ω*t)
    u̇ .= -A*ω * sin.(k*x) * sin(ω*t)
    return u, u̇
end

function initialize!(u::AbstractMatrix, u̇::AbstractMatrix, x::AbstractMatrix, t;
                     A, k, ω)
    M = size(u, 2)
    @assert size(u̇, 2) == size(x, 2) == M
    for m in 1:M
        initialize!(view(u, :, m), view(u̇, :, m), view(x, :, m), t; A, k, ω)
    end
    return u, u̇
end

################################################################################
# System parameters

"""
    Params3d{T}

Bundle of system-level scalar parameters for the 3D wave equation:

* `A` — IC amplitude.
* `k :: NTuple{3, T}` — IC wavenumber vector `(kx, ky, kz)`.
* `ω` — IC angular frequency.
* `τ` — SIPG penalty constant for `apply_laplacian3d!`.
* `bdry_values :: NTuple{6, T}` — per-face Dirichlet values, indexed
  by the boundary-condition tag stored on each outer face
  (`mesh.conn.bdry[f, e]`).
* `sommerfeld_R :: T` — radius of curvature of the Sommerfeld absorbing
  surface. Selects the Bayliss–Turkel order:

  | `sommerfeld_R` | BC enforced                       | exact for      |
  |----------------|-----------------------------------|----------------|
  | `Inf` (default)| `u̇ + ∂_n u = 0`                  | plane waves    |
  | finite `R > 0` | `u̇ + ∂_n u + u/R = 0`            | sphere of radius `R`, source at sphere centre |

  Use `Inf` (or omit) on planar / cube-face boundaries; use `R = R₂` on
  the inflated-cube outer sphere. The plane-wave version is fine when
  `R` is large compared to the dominant wavelength, but leaves an
  `O(1/R)` reflection coefficient otherwise — the spherical BGT-1
  correction `+ u/R` is exact for the canonical outgoing radial wave
  `u = f(r − t)/r`.

These are exactly the mesh-independent scalars that the driver assembles
once and feeds to the IC setup and the RHS. Bundling them lets us pass
the whole bundle as the `p` argument of a `SecondOrderODEProblem` and
dispatch on `f!(ü, u̇, u, p::Params3d, t)` instead of capturing globals.
"""
struct Params3d{T}
    A            :: T
    k            :: NTuple{3, T}
    ω            :: T
    τ            :: T
    bdry_values  :: NTuple{6, T}
    sommerfeld_R :: T
end

"""
    Params3d(; A, k, ω, τ, bdry_values, sommerfeld_R = Inf) → Params3d{T}

Keyword constructor. The element type `T` is taken by promoting all
inputs to a common floating-point type.
"""
function Params3d(; A, k, ω, τ, bdry_values, sommerfeld_R = Inf)
    # `sommerfeld_R` is deliberately *not* in the promotion set — the
    # default `Inf` (Float64) shouldn't force `T = Float64` and silently
    # upcast a Float32 caller. `T(sommerfeld_R)` below carries `Inf`
    # cleanly into both Float32 and Float64.
    T = promote_type(typeof(A), eltype(k), typeof(ω), typeof(τ),
                     eltype(bdry_values))
    return Params3d{T}(T(A),
                       NTuple{3, T}(k),
                       T(ω),
                       T(τ),
                       NTuple{6, T}(bdry_values),
                       T(sommerfeld_R))
end

################################################################################
# Initialisation

function initialize3d!(u::AbstractArray{T,3}, u̇::AbstractArray{T,3},
                       x::AbstractArray{T,3}, y::AbstractArray{T,3}, z::AbstractArray{T,3},
                       t; A, kx, ky, kz, ω) where {T}
    @. u  =  A   * sin(kx*x) * sin(ky*y) * sin(kz*z) * cos(ω*t)
    @. u̇ = -A*ω * sin(kx*x) * sin(ky*y) * sin(kz*z) * sin(ω*t)
    return u, u̇
end

# `Params3d`-bundled variant.
initialize3d!(u::AbstractArray{T,4}, u̇::AbstractArray{T,4},
              coords::AbstractArray{T,5}, t, params::Params3d{T}) where {T} =
    initialize3d!(u, u̇, coords, t;
                  A = params.A,
                  kx = params.k[1], ky = params.k[2], kz = params.k[3],
                  ω = params.ω)

function initialize3d!(u::AbstractArray{T,4}, u̇::AbstractArray{T,4},
                       coords::AbstractArray{T,5}, t;
                       A, kx, ky, kz, ω) where {T}
    @assert size(u̇) == size(u)
    Ne = size(u, 4)
    @assert size(coords, 5) == Ne
    for e in 1:Ne
        initialize3d!(view(u,  :, :, :, e),
                      view(u̇, :, :, :, e),
                      view(coords, 1, :, :, :, e),
                      view(coords, 2, :, :, :, e),
                      view(coords, 3, :, :, :, e), t;
                      A, kx, ky, kz, ω)
    end
    return u, u̇
end

################################################################################
# Analytic standing-wave eigenmodes
#
# Each of the functions below fills `u` and `u̇ = ∂_t u` at time `t` for
# one family of exact solutions to `ü = ∇² u`. They are pure broadcasts
# over `coords` (the 5-D `(3, N, N, N, Ne)` array), so they work
# unchanged on host arrays and on GPU device arrays. Used both to seed
# the initial condition and to evaluate the analytic reference at each
# diagnostic time sample.
#
# The `ic_kind` argument in `bin/waveplot3d.jl` selects between these.

"""
    eigenmode_cartesian!(u, u̇, coords, t; A, kx, ky, kz, ω, x0, x1)

Standing-wave eigenmode of `ü = ∇² u` on the cube `[x0, x1]³` with
homogeneous Dirichlet BC on the six cube faces:

```
u(x, y, z, t) = A · sin(kx · X) · sin(ky · Y) · sin(kz · Z) · cos(ω t)
```

with normalised coordinates `X = (x - x0)/(x1 - x0)`, similarly for
`Y, Z`. The exact-eigenmode dispersion relation is
`ω = √(kx² + ky² + kz²) / (x1 - x0)`. Choosing `kx, ky, kz` as integer
multiples of `π` makes the IC vanish on the six cube faces, so the
homogeneous Dirichlet BC is satisfied exactly.

Fills both `u` and `u̇` in-place at time `t`.
"""
function eigenmode_cartesian!(u::AbstractArray{T},
                                u̇::AbstractArray{T},
                                coords::AbstractArray{T},
                                t::Real;
                                A, kx, ky, kz, ω, x0, x1) where {T}
    Xv = @view coords[1, :, :, :, :]
    Yv = @view coords[2, :, :, :, :]
    Zv = @view coords[3, :, :, :, :]
    A_  = T(A)
    kx_ = T(kx);  ky_ = T(ky);  kz_ = T(kz)
    ω_  = T(ω)
    x0_ = T(x0)
    L_  = T(x1 - x0)
    ct  = cos(ω_ * T(t))
    st  = sin(ω_ * T(t))
    @. u  =  A_      * sin(kx_ * (Xv - x0_) / L_) *
                       sin(ky_ * (Yv - x0_) / L_) *
                       sin(kz_ * (Zv - x0_) / L_) * ct
    @. u̇ = -A_ * ω_ * sin(kx_ * (Xv - x0_) / L_) *
                       sin(ky_ * (Yv - x0_) / L_) *
                       sin(kz_ * (Zv - x0_) / L_) * st
    return u, u̇
end

"""
    eigenmode_radial!(u, u̇, coords, t; A, R, n = 1, center = (0, 0, 0))

Spherically-symmetric (`l = 0`, `n`-th radial) eigenmode of
`ü = ∇² u` on the ball `|x − center| ≤ R` with homogeneous Dirichlet
BC on `|x − center| = R`:

```
u(r, t) = A · sinc(n · r / R) · cos(ω t),    r = |x − center|
```

where `sinc(y) = sin(πy)/(πy)` (Julia's `Base.sinc`), and the
eigenfrequency is `ω = nπ / R`. The IC vanishes at `r = R` (since
`sin(nπ) = 0`) so the Dirichlet BC is satisfied exactly.

`n ≥ 1` selects the radial node count; `n = 1` is the fundamental.
`sinc` handles the removable singularity at `r = 0` (value `A · cos(ω t)`).

Fills both `u` and `u̇` in-place at time `t`.
"""
function eigenmode_radial!(u::AbstractArray{T},
                            u̇::AbstractArray{T},
                            coords::AbstractArray{T},
                            t::Real;
                            A, R, n::Integer = 1,
                            center = (zero(T), zero(T), zero(T))) where {T}
    Xv = @view coords[1, :, :, :, :]
    Yv = @view coords[2, :, :, :, :]
    Zv = @view coords[3, :, :, :, :]
    A_ = T(A)
    R_ = T(R)
    n_ = T(n)
    ω  = n_ * T(π) / R_
    ct = cos(ω * T(t))
    st = sin(ω * T(t))
    cx = T(center[1]);  cy = T(center[2]);  cz = T(center[3])
    @. u  =  A_     * sinc(n_ * sqrt((Xv - cx)^2 + (Yv - cy)^2 + (Zv - cz)^2) / R_) * ct
    @. u̇ = -A_ * ω * sinc(n_ * sqrt((Xv - cx)^2 + (Yv - cy)^2 + (Zv - cz)^2) / R_) * st
    return u, u̇
end

# Smooth evaluation of `sinh(x)/x`. Regular at x = 0 (value 1); for
# small `|x|` use a Taylor expansion to keep the broadcast finite at
# the r = 0 GLL node of the inflated-cube inner cube. Threshold chosen
# so the Taylor truncation error is below ~eps(Float32) for x² < 1e-3.
@inline function _sinhc(x::T) where {T<:AbstractFloat}
    x² = x * x
    if x² < T(1e-3)
        # sinh(x)/x = 1 + x²/6 + x⁴/120 + x⁶/5040 + …
        return one(T) + x² * (T(1/6) + x² * (T(1/120) + x² * T(1/5040)))
    else
        return sinh(x) / x
    end
end

"""
    outgoing_pulse!(u, u̇, coords, t;
                     A, s0, σ, center = (0, 0, 0))

Spherically-symmetric outgoing Gaussian pulse of the 3D wave equation
`ü = ∇² u`. Built from the antisymmetric Cauchy data
`u(r, t) = [F(t − r) − F(t + r)] / r` with
`F(s) = A · exp(−(s + s₀)² / (2σ²))`, which:

* is regular at `r = 0` (the F-difference vanishes linearly there);
* concentrates initial energy at `r = s₀` and propagates outward at
  unit speed — the pulse peak sits at `r = s₀ + t`;
* has an exponentially small ingoing image inside the domain
  (suppressed by `exp(−s₀² / (2σ²))`) when `σ ≪ s₀`, so the wave
  looks purely outgoing for all practical purposes.

Ideal IC for testing Sommerfeld outer BCs: with Dirichlet outer faces
the pulse reflects at `r = R`; with Sommerfeld it bleeds out cleanly.

# Arguments

* `A` — amplitude of the F-Gaussian (overall scale of `u`).
* `s0` — radial offset of the pulse peak at `t = 0`. Choose so the
  pulse fits inside the domain: `s0 + 4σ ≲ R_outer`.
* `σ`  — Gaussian width. Choose `σ ≲ s0/4` to keep the ingoing image
  exponentially small.
* `center` — sphere centre in physical coordinates (default origin).

# Closed form

Using `β ≡ t + s₀`, the antisymmetric F-difference factors via `sinh`:

```
u(r, t)  =  2A·(β/σ²) · sinhc(β r/σ²) · exp(−(β² + r²)/(2σ²))
u̇(r, t) = −(2A/σ²) · exp(−(β² + r²)/(2σ²)) ·
            [(β²/σ²) · sinhc(β r/σ²) − cosh(β r/σ²)]
```

where `sinhc(x) ≡ sinh(x)/x` (regularised at 0 by [`_sinhc`](@ref)).
"""
function outgoing_pulse!(u::AbstractArray{T},
                          u̇::AbstractArray{T},
                          coords::AbstractArray{T},
                          t::Real;
                          A, s0, σ,
                          center = (zero(T), zero(T), zero(T))) where {T}
    Xv = @view coords[1, :, :, :, :]
    Yv = @view coords[2, :, :, :, :]
    Zv = @view coords[3, :, :, :, :]
    A_ = T(A)
    s0_ = T(s0)
    σ_  = T(σ)
    σ²  = σ_ * σ_
    iσ² = one(T) / σ²
    β   = T(t) + s0_       # F-pulse centre at time t (peak of |u| is at r = β)
    β²  = β * β
    cx = T(center[1]);  cy = T(center[2]);  cz = T(center[3])

    # Use the inlined `sqrt((Xv-cx)^2 + …)` form (per the `eigenmode_*`
    # style elsewhere in this file) so the broadcast fuses with no
    # auxiliary buffers and lowers cleanly onto GPU backends.
    @. u  = T(2) * A_ * β * iσ² *
            _sinhc(β * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2) * iσ²) *
            exp(-T(0.5) * (β² + (Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2) * iσ²)

    @. u̇ = -T(2) * A_ * iσ² *
            exp(-T(0.5) * (β² + (Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2) * iσ²) *
            (β² * iσ² *
              _sinhc(β * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2) * iσ²) -
              cosh(β * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2) * iσ²))
    return u, u̇
end

# First ten positive zeros of `j_2(x)` (the spherical Bessel function of
# the first kind, order 2). Used to set the wavenumber `k = α_{2,n}/R`
# so that the IC vanishes on `r = R`. Computed once at literature
# precision; quoted here to Float64 accuracy.
const _SPHERICAL_BESSEL_J2_ZEROS = (
    5.763459196844682, 9.095011330476355, 12.322940970566583,
    15.514603010235257, 18.689036355362822, 21.853874222709703,
    25.012803202289611, 28.167829707106325, 31.320141707447065,
    34.470488875689556,
)

# Smooth evaluation of `g(x) = j_2(x) / x²`. For `x² < 0.01` use the
# Taylor expansion at the origin (`g(0) = 1/15`); otherwise call
# `SpecialFunctions.sphericalbesselj` and divide. The branch lets the
# broadcast skip the spurious `0/0` at `x = 0` and keeps SpecialFunctions
# off the device side of any broadcast — for the host-side IC build that
# matters not at all, but it keeps the eventual GPU `eigenmode_quadrupole!`
# call path easy to lower (no broadcast to a function the GPU backend
# can't compile).
@inline function _j2_over_x2(x::T) where {T<:AbstractFloat}
    x² = x * x
    if x² < T(0.01)
        # g(x) = 1/15 - x²/210 + x⁴/7560 - x⁶/498960 + …
        return T(1/15) - x² * (T(1/210) - x² * (T(1/7560) - x² * T(1/498960)))
    else
        return T(sphericalbesselj(2, x)) / x²
    end
end

"""
    eigenmode_quadrupole!(u, u̇, coords, t;
                           A, R, n = 1, m = 0, center = (0, 0, 0))

Quadrupole (l = 2) eigenmode of `ü = ∇² u` on the ball
`|x − center| ≤ R` with homogeneous Dirichlet BC at `r = R`:

```
u(x, t) = A · k² · g(k r) · 𝒬_m(x − center) · cos(ω t),
   g(s) = j_2(s) / s²,   k = α_{2,n} / R,   ω = k
```

where `α_{2,n}` is the `n`-th positive zero of `j_2`
(`α_{2,1} ≈ 5.7635`, see `_SPHERICAL_BESSEL_J2_ZEROS`) and `𝒬_m` is one
of the five real-valued homogeneous-quadratic Cartesian spherical
harmonics:

| `m`   | `𝒬_m(x, y, z)`              |
|-------|-----------------------------|
| `0`   | `2 z² − x² − y²`  (axial)   |
| `+1`  | `2 x z`                     |
| `−1`  | `2 y z`                     |
| `+2`  | `x² − y²`                   |
| `−2`  | `2 x y`                     |

The factor `j_2(k r) / r² = k² · g(k r)` is regularised near `r = 0`
by a Taylor series in `_j2_over_x2`, so the broadcast is smooth at the
origin. The full IC vanishes exactly at `r = R` (by choice of `k`),
satisfying homogeneous Dirichlet on the sphere.

Fills `u` and `u̇` in-place at time `t`. Five separate broadcasts
(one per `m`) keep each kernel type-stable.
"""
function eigenmode_quadrupole!(u::AbstractArray{T},
                                 u̇::AbstractArray{T},
                                 coords::AbstractArray{T},
                                 t::Real;
                                 A, R, n::Integer = 1, m::Integer = 0,
                                 center = (zero(T), zero(T), zero(T))) where {T}
    -2 ≤ m ≤ 2 || error("eigenmode_quadrupole!: m must be in -2..2, got $m")
    1 ≤ n ≤ length(_SPHERICAL_BESSEL_J2_ZEROS) ||
        error("eigenmode_quadrupole!: n must be in 1..$(length(_SPHERICAL_BESSEL_J2_ZEROS)), got $n")

    α  = T(_SPHERICAL_BESSEL_J2_ZEROS[n])
    R_ = T(R)
    k  = α / R_
    k² = k * k
    ω  = k
    A_ = T(A)
    cx = T(center[1]);  cy = T(center[2]);  cz = T(center[3])
    ct = cos(ω * T(t))
    st = sin(ω * T(t))

    Xv = @view coords[1, :, :, :, :]
    Yv = @view coords[2, :, :, :, :]
    Zv = @view coords[3, :, :, :, :]

    # Dispatch on `m` outside the broadcast so each branch fires a
    # single type-stable broadcast — important for both CPU performance
    # and clean lowering to GPU backends.
    if m == 0
        @. u  =  A_     * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 (2*(Zv-cz)^2 - (Xv-cx)^2 - (Yv-cy)^2) * ct
        @. u̇ = -A_ * ω * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 (2*(Zv-cz)^2 - (Xv-cx)^2 - (Yv-cy)^2) * st
    elseif m == 1
        @. u  =  A_     * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 (2*(Xv-cx)*(Zv-cz)) * ct
        @. u̇ = -A_ * ω * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 (2*(Xv-cx)*(Zv-cz)) * st
    elseif m == -1
        @. u  =  A_     * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 (2*(Yv-cy)*(Zv-cz)) * ct
        @. u̇ = -A_ * ω * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 (2*(Yv-cy)*(Zv-cz)) * st
    elseif m == 2
        @. u  =  A_     * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 ((Xv-cx)^2 - (Yv-cy)^2) * ct
        @. u̇ = -A_ * ω * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 ((Xv-cx)^2 - (Yv-cy)^2) * st
    else  # m == -2
        @. u  =  A_     * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 (2*(Xv-cx)*(Yv-cy)) * ct
        @. u̇ = -A_ * ω * k² * _j2_over_x2(k * sqrt((Xv-cx)^2 + (Yv-cy)^2 + (Zv-cz)^2)) *
                 (2*(Xv-cx)*(Yv-cy)) * st
    end
    return u, u̇
end

################################################################################
# Wave-equation RHS: apply_laplacian3d! + Sommerfeld dissipative pass
#
# `apply_laplacian3d!` treats outer faces with `bdry == SOMMERFELD_BDRY_TAG`
# as "free" (zero SAT contribution), so we can layer a dedicated Sommerfeld
# face SAT on top without double-counting. The Sommerfeld term drives the
# outgoing characteristic `w_out = u̇ + ∂_n u` to zero by adding the
# dissipative face flux `−wF · w_out / H_phys` to `ü` at every Sommerfeld
# face quadrature node.

@inline function _sommerfeld_face!(::Val{f}, i, j, k, e,
                                    ü::AbstractArray{T, 4},
                                    u̇::AbstractArray{T, 4},
                                    geom::MeshGeometry{T, N},
                                    H_1d::SVector{N, T},
                                    sommerfeld_tag::Int8,
                                    inv_R::T,
                                    ::Val{N}) where {f, T, N}
    a_idx          = _face_axis_idx(Val(f))
    face_r         = _face_row(Val(f), Val(N))
    sgn_f          = _face_sign(Val(f), T)
    sgn_c          = _cross_sign(Val(a_idx), T)
    axis_p, axis_q = _tangent_axes(Val(a_idx))

    ia = a_idx == 1 ? i : a_idx == 2 ? j : k
    ia == face_r || return nothing

    @inbounds nbr = geom.conn.neighbour[f, e]
    nbr == 0 || return nothing
    @inbounds tag = geom.conn.bdry[f, e]
    tag == sommerfeld_tag || return nothing

    p_local, q_local = a_idx == 1 ? (j, k) :
                       a_idx == 2 ? (i, k) :
                                    (i, j)

    @inbounds J11 = geom.jac[1,1,i,j,k,e]; @inbounds J12 = geom.jac[1,2,i,j,k,e]; @inbounds J13 = geom.jac[1,3,i,j,k,e]
    @inbounds J21 = geom.jac[2,1,i,j,k,e]; @inbounds J22 = geom.jac[2,2,i,j,k,e]; @inbounds J23 = geom.jac[2,3,i,j,k,e]
    @inbounds J31 = geom.jac[3,1,i,j,k,e]; @inbounds J32 = geom.jac[3,2,i,j,k,e]; @inbounds J33 = geom.jac[3,3,i,j,k,e]

    tp_x = axis_p == 1 ? J11 : axis_p == 2 ? J12 : J13
    tp_y = axis_p == 1 ? J21 : axis_p == 2 ? J22 : J23
    tp_z = axis_p == 1 ? J31 : axis_p == 2 ? J32 : J33
    tq_x = axis_q == 1 ? J11 : axis_q == 2 ? J12 : J13
    tq_y = axis_q == 1 ? J21 : axis_q == 2 ? J22 : J23
    tq_z = axis_q == 1 ? J31 : axis_q == 2 ? J32 : J33

    @inbounds sgn_out = sgn_f * sgn_c * T(geom.handedness[e])
    nx_u = sgn_out * (tp_y * tq_z - tp_z * tq_y)
    ny_u = sgn_out * (tp_z * tq_x - tp_x * tq_z)
    nz_u = sgn_out * (tp_x * tq_y - tp_y * tq_x)
    JF = sqrt(nx_u*nx_u + ny_u*ny_u + nz_u*nz_u)
    nx = nx_u / JF; ny = ny_u / JF; nz = nz_u / JF

    @inbounds wF = JF * H_1d[p_local] * H_1d[q_local]

    # Physical normal gradient at this face node — pre-computed by
    # apply_laplacian3d!'s pass-1 face-trace gather. Slot 1 already holds
    # `u` at the same face node, so the BGT-1 spherical correction
    # `+ u/R` reads from `face_trace[1, …]` rather than `u[…]` for
    # memory coalescence.
    @inbounds u_self = geom.face_trace[1, p_local, q_local, f, e]
    @inbounds gx = geom.face_trace[2, p_local, q_local, f, e]
    @inbounds gy = geom.face_trace[3, p_local, q_local, f, e]
    @inbounds gz = geom.face_trace[4, p_local, q_local, f, e]
    Gn = nx * gx + ny * gy + nz * gz

    @inbounds u̇_self = u̇[i, j, k, e]
    @inbounds Hp = geom.Hphys[i, j, k, e]

    # BGT-1 spherical Sommerfeld: drive `u̇ + ∂_n u + u/R → 0`. With
    # `inv_R = 1/R = 0` (R = Inf) this collapses to the plane-wave
    # BGT-0 BC. The `+ u/R` term is the exact correction for an
    # outgoing radial pulse `u = f(r-t)/r` on a sphere of radius R
    # centred at the source.
    @inbounds ü[i, j, k, e] -= wF * (u̇_self + Gn + inv_R * u_self) / Hp
    return nothing
end

@kernel function _sommerfeld_pass_kernel!(ü::AbstractArray{T},
                                          @Const(u̇::AbstractArray{T}),
                                          geom, H_1d,
                                          sommerfeld_tag::Int8,
                                          inv_R::T,
                                          ::Val{N}) where {T, N}
    e        = @index(Group, Linear)
    li       = @index(Local, Linear)
    i, j, k  = _ijk_from_li(li, Val(N))

    # Each face contributes to its own outgoing-flux drag. Workitems on
    # multiple faces (edges/corners) accumulate one drag per face they
    # lie on — matching the continuum surface integral.
    _sommerfeld_face!(Val(1), i, j, k, e, ü, u̇, geom, H_1d, sommerfeld_tag, inv_R, Val(N))
    _sommerfeld_face!(Val(2), i, j, k, e, ü, u̇, geom, H_1d, sommerfeld_tag, inv_R, Val(N))
    _sommerfeld_face!(Val(3), i, j, k, e, ü, u̇, geom, H_1d, sommerfeld_tag, inv_R, Val(N))
    _sommerfeld_face!(Val(4), i, j, k, e, ü, u̇, geom, H_1d, sommerfeld_tag, inv_R, Val(N))
    _sommerfeld_face!(Val(5), i, j, k, e, ü, u̇, geom, H_1d, sommerfeld_tag, inv_R, Val(N))
    _sommerfeld_face!(Val(6), i, j, k, e, ü, u̇, geom, H_1d, sommerfeld_tag, inv_R, Val(N))
end

"""
    rhs_wave3d!(ü, u, u̇, params; geom, ops)
    rhs_wave3d!(ü, u, u̇, bdry_values; geom, ops, τ, sommerfeld_R = T(Inf))

3D wave-equation RHS: `ü = L_h u` plus a Sommerfeld dissipative SAT at
outer faces tagged `7`. Internally calls
`HexSBPSAT.apply_laplacian3d!(ü, u, bdry_values; geom, ops, τ)` (which
sees Sommerfeld faces as "free" and contributes nothing on them), then a
second kernel iterates over outer faces tagged `7` and adds the
Bayliss–Turkel dissipative drag

```
Δü = −wF · (u̇ + ∂_n u + u/R) / H_phys
```

with `R = sommerfeld_R`. `R = Inf` (default) gives the plane-wave
Sommerfeld BC `u̇ + ∂_n u = 0` (BGT-0); a finite positive `R` gives the
spherical BGT-1 BC, exact for `u = f(r-t)/r` on a sphere of radius `R`
centred at the source. Use `R = R₂` on the inflated cube's outer
sphere to suppress the `O(1/R)` reflection coefficient that BGT-0
leaves behind.

# Boundary tag convention

* `0` — interior or unset.
* `1..6` — Dirichlet against `bdry_values[tag]`.
* `7` — Sommerfeld radiative.
"""
function rhs_wave3d!(ü::AbstractArray{T,4}, u::AbstractArray{T,4}, u̇::AbstractArray{T,4},
                     params::Params3d{T};
                     geom::MeshGeometry{T, N}, ops::SBPOps{N, T}) where {N, T}
    return rhs_wave3d!(ü, u, u̇, params.bdry_values;
                       geom, ops, τ = params.τ,
                       sommerfeld_R = params.sommerfeld_R)
end

function rhs_wave3d!(ü::AbstractArray{T,4}, u::AbstractArray{T,4}, u̇::AbstractArray{T,4},
                     bdry_values::NTuple{6, T};
                     geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ,
                     sommerfeld_R = T(Inf)) where {N, T}
    @assert size(ü) == size(u̇) == size(u)
    @assert size(u, 1) == size(u, 2) == size(u, 3) == N
    @assert size(u, 4) == geom.Ne

    # Pass 1: spatial Laplacian + interior SIPG + outer Dirichlet SAT.
    # Sommerfeld-tagged outer faces are skipped by `apply_laplacian3d!`.
    # As a side effect, `geom.face_trace` is populated with `(u, ∇u)` at
    # every face quadrature node, which the Sommerfeld pass below reads.
    apply_laplacian3d!(ü, u, bdry_values; geom, ops, τ)

    # Pass 2: Sommerfeld dissipative drag. Only fires on outer faces
    # tagged `SOMMERFELD_BDRY_TAG`; on all other faces each workitem's
    # `_sommerfeld_face!` returns immediately. `1/Inf = 0` cleanly turns
    # off the BGT-1 `+u/R` correction without a runtime branch.
    H_1d = SVector{N, T}(ntuple(i -> ops.H[i, i], Val(N)))
    inv_R = one(T) / T(sommerfeld_R)
    backend = get_backend(u)
    _sommerfeld_pass_kernel!(backend, N^3)(
        ü, u̇, geom, H_1d, SOMMERFELD_BDRY_TAG, inv_R, Val(N);
        ndrange = N^3 * geom.Ne)

    return ü
end

################################################################################
# Wave-equation timestep limit

"""
    recommended_dt(geom, ops, τ; cfl_safety = 0.9) → T

Suggest a stable timestep for an explicit symplectic integrator
(`KahanLi8` and friends) on the wave equation `ü = L u`. Returns

    cfl_safety · 2 / sqrt(|λ_max|)

where `|λ_max|` is estimated by
[`HexSBPSAT.spectral_radius_estimate`](@ref). The bare condition
`dt · ω_max < 2` is the Störmer–Verlet stability limit;
`cfl_safety = 0.9` keeps a margin away from the edge.

Typical usage at the top of a driver:

```julia
geom = make_geometry(mesh, elem)
ops  = make_operators(elem)
τ    = 100.0
dt   = recommended_dt(geom, ops, τ)
```
"""
function recommended_dt(geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ;
                         cfl_safety = T(0.9)) where {N, T}
    λ = spectral_radius_estimate(geom, ops, τ)
    λ == 0 && return T(Inf)
    return cfl_safety * T(2) / sqrt(λ)
end
