# 2D wave-equation layer. Mirror of the 3D content in `wave.jl` adapted
# to two spatial dimensions:
#
#   * `Params2d{T}` — system-level scalars (amplitude, wavenumber,
#     frequency, SIPG penalty, four Dirichlet face values).
#   * `initialize2d!` — sin·sin·cos IC on a square.
#   * `eigenmode_cartesian_2d!` — Cartesian Dirichlet eigenmode on
#     `[x0, x1]²`.
#   * `eigenmode_radial_2d!` — Bessel J₀ eigenmode on a disk of radius
#     `R` centred at `center`. Uses the tabulated first few zeros of
#     `J_0` below.
#   * `rhs_wave2d!` — 2D wave RHS: `ü = L_h u`. No Sommerfeld pass yet;
#     outer-face tags ≥ 5 are "free" (natural boundary lift only).
#   * `recommended_dt(geom::MeshGeometry{2, T, N}, …)` — Störmer–Verlet
#     timestep limit.

# First few positive zeros of `J_0(x)`. Hardcoded because
# `SpecialFunctions` provides `besselj` but no zero-finder; these are
# deterministic constants known to 16+ digits.
const _J0_ZEROS = (
    2.4048255576957727,
    5.520078110286311,
    8.653727912911012,
    11.791534439014281,
    14.930917708487785,
)

################################################################################
# System parameters

"""
    Params2d{T}

2D wave-equation parameter bundle:

* `A` — IC amplitude.
* `k :: NTuple{2, T}` — IC wavenumber vector `(kx, ky)`.
* `ω` — IC angular frequency.
* `τ` — SIPG penalty constant for `apply_laplacian!`.
* `bdry_values :: NTuple{4, T}` — per-face Dirichlet values
  (`mesh.conn.bdry[f, e] ∈ 1..4` → `bdry_values[bdry[f, e]]`).
* `sommerfeld_R :: T` — radius of curvature of the Sommerfeld
  absorbing surface. `Inf` (default) → plane-wave BGT-0
  `u̇ + ∂_n u = 0`. Finite values are accepted but ignored — 2D
  doesn't have a clean BGT-1 spherical correction (the 2D wave
  equation isn't Huygens, so outgoing radial waves decay as `1/√r`,
  not `1/r`). See the docstring on `rhs_wave2d!` for the full
  caveat.
"""
struct Params2d{T}
    A            :: T
    k            :: NTuple{2, T}
    ω            :: T
    τ            :: T
    bdry_values  :: NTuple{4, T}
    sommerfeld_R :: T
end

"""
    Params2d(; A, k, ω, τ, bdry_values, sommerfeld_R = Inf) → Params2d{T}

Keyword constructor that promotes all inputs to a common floating-
point type.
"""
function Params2d(; A, k, ω, τ, bdry_values, sommerfeld_R = Inf)
    # `sommerfeld_R` is deliberately excluded from the promotion set so
    # the default `Inf` (Float64) doesn't force `T = Float64` on a
    # Float32 caller (matches `Params3d`).
    T = promote_type(typeof(A), eltype(k), typeof(ω), typeof(τ),
                     eltype(bdry_values))
    return Params2d{T}(T(A),
                       NTuple{2, T}(k),
                       T(ω),
                       T(τ),
                       NTuple{4, T}(bdry_values),
                       T(sommerfeld_R))
end

################################################################################
# Initialisation

function initialize2d!(u::AbstractArray{T,2}, u̇::AbstractArray{T,2},
                       x::AbstractArray{T,2}, y::AbstractArray{T,2},
                       t; A, kx, ky, ω) where {T}
    @. u  =  A   * sin(kx*x) * sin(ky*y) * cos(ω*t)
    @. u̇ = -A*ω * sin(kx*x) * sin(ky*y) * sin(ω*t)
    return u, u̇
end

# `Params2d`-bundled variant — convenience for callers that build their
# coordinates already in element-shape.
initialize2d!(u::AbstractArray{T,3}, u̇::AbstractArray{T,3},
              coords::AbstractArray{T,4}, t, params::Params2d{T}) where {T} =
    initialize2d!(u, u̇, coords, t;
                  A = params.A,
                  kx = params.k[1], ky = params.k[2],
                  ω = params.ω)

function initialize2d!(u::AbstractArray{T,3}, u̇::AbstractArray{T,3},
                       coords::AbstractArray{T,4}, t;
                       A, kx, ky, ω) where {T}
    @assert size(u̇) == size(u)
    Ne = size(u, 3)
    @assert size(coords, 4) == Ne
    for e in 1:Ne
        initialize2d!(view(u,  :, :, e),
                      view(u̇, :, :, e),
                      view(coords, 1, :, :, e),
                      view(coords, 2, :, :, e), t;
                      A, kx, ky, ω)
    end
    return u, u̇
end

################################################################################
# Analytic standing-wave eigenmodes (2D)

"""
    eigenmode_cartesian_2d!(u, u̇, coords, t; A, kx, ky, ω, x0, x1)

Standing-wave eigenmode of `ü = ∇² u` on the square `[x0, x1]²` with
homogeneous Dirichlet BC on all four edges:

```
u(x, y, t) = A · sin(kx · X) · sin(ky · Y) · cos(ω t)
```

with normalised coordinates `X = (x - x0)/(x1 - x0)`, similarly `Y`.
The dispersion relation is `ω = √(kx² + ky²) / (x1 - x0)`. Choosing
`kx, ky` as integer multiples of `π` makes the IC vanish on the four
edges, so the homogeneous Dirichlet BC is satisfied exactly.

Fills both `u` and `u̇` in-place at time `t`.
"""
function eigenmode_cartesian_2d!(u::AbstractArray{T},
                                  u̇::AbstractArray{T},
                                  coords::AbstractArray{T},
                                  t::Real;
                                  A, kx, ky, ω, x0, x1) where {T}
    Xv = @view coords[1, :, :, :]
    Yv = @view coords[2, :, :, :]
    L_ = x1 - x0
    @. u  =  A   * sin(kx * (Xv - x0) / L_) * sin(ky * (Yv - x0) / L_) * cos(ω * t)
    @. u̇ = -A*ω * sin(kx * (Xv - x0) / L_) * sin(ky * (Yv - x0) / L_) * sin(ω * t)
    return u, u̇
end

"""
    eigenmode_radial_2d!(u, u̇, coords, t; A, R, n, center = (0, 0))

Spherically-symmetric Bessel eigenmode of `ü = ∇² u` on a disk of
radius `R` centred at `center`:

```
u(r, t) = A · J₀(j_{0,n} · r / R) · cos(ω t),  ω = j_{0,n} / R
```

where `j_{0,n}` is the n-th positive zero of `J₀`. Vanishes at
`r = R`, so this matches a Dirichlet-bounded disk exactly. Only
`n ∈ 1..5` is supported (tabulated zeros).
"""
function eigenmode_radial_2d!(u::AbstractArray{T},
                               u̇::AbstractArray{T},
                               coords::AbstractArray{T},
                               t::Real;
                               A, R, n::Integer,
                               center = (zero(T), zero(T))) where {T}
    1 ≤ n ≤ length(_J0_ZEROS) ||
        error("eigenmode_radial_2d!: radial mode n=$n out of tabulated range " *
              "(1..$(length(_J0_ZEROS)))")
    j0n = T(_J0_ZEROS[n])
    ω   = j0n / T(R)
    cx, cy = center
    Xv = @view coords[1, :, :, :]
    Yv = @view coords[2, :, :, :]
    rs = @. sqrt((Xv - cx)^2 + (Yv - cy)^2)
    @. u  =  A   * besselj(0, j0n * rs / R) * cos(ω * t)
    @. u̇ = -A*ω * besselj(0, j0n * rs / R) * sin(ω * t)
    return u, u̇
end

################################################################################
# Smooth Gaussian-pulse outgoing solution (Hankel transform of a
# Gaussian initial bump).
#
# The 2D scalar wave equation isn't Huygens, so there's no clean
# closed-form `f(r − t) / r` analog of a 3D outgoing pulse. The
# closest analytic solution that is (a) smooth, (b) localized near
# the origin at `t = 0`, and (c) propagates outward is the
# Hankel-transform solution to the Cauchy problem with Gaussian
# initial data:
#
#   u(r, 0) = A · exp(-r² / (2σ²)),    u_t(r, 0) = 0
#
# Then for `t ≥ 0`:
#
#   u(r, t)  =   A σ² ∫_0^∞ k · exp(-k²σ²/2) · J_0(kr) · cos(kt) dk
#   u̇(r, t) = − A σ² ∫_0^∞ k² · exp(-k²σ²/2) · J_0(kr) · sin(kt) dk
#
# The integral is evaluated by Gauss–Legendre quadrature on
# `[0, 8/σ]` (8σ above the Gaussian's effective support — the
# envelope is below ~e^{-32} = 1.3e-14 there).
#
# Precision: `exp`, `cos`, and `besselj` aren't defined for
# `MultiFloats`, so the integrand is evaluated internally at
# `Float64` and the result is converted to the array's element
# type `T`. For `T ∈ {Float32, Float64}` this matches or exceeds
# the natural precision; for `T ∈ {Float32x2, Float64x2}` the
# analytic reference is limited to ~Float64 accuracy (~1.5e-15),
# still far below the spatial discretization error at typical
# resolutions.
#
# Performance: the per-grid-point Bessel evaluations are the
# expensive part and are time-independent — `GaussianPulse2dCache`
# caches them so each new-time evaluation becomes a fast linear
# combination over the quadrature nodes.

# Convert any `Real` to Float64. Direct `Float64(x)` doesn't have a
# method for `MultiFloat{Float32, N}` (the conversion goes through the
# inner Float32 component only), so we route via `big` → `BigFloat`,
# which all `Real` subtypes implement. For Float32 / Float64 this
# costs one BigFloat allocation (~200 ns); we only call it at cache-
# build time, where it's dominated by the per-grid-point Bessel
# evaluation anyway.
@inline _to_f64(x::Real) = Float64(big(x))

"""
    GaussianPulse2dCache

Pre-computed quadrature data + per-grid-point Bessel-function table
backing [`outgoing_pulse_2d!`](@ref). Build once at evolve-loop setup
via [`outgoing_pulse_2d_cache`](@ref) and pass to the cache-based
`outgoing_pulse_2d!(u, u̇, cache, t; A)` at every sample time.

The cache pins to the specific `coords`, Gaussian width `σ`, and
`center` it was built from; reusing it with different ones gives
wrong results.
"""
struct GaussianPulse2dCache
    rgrid :: Array{Float64, 3}     # (N, N, Ne) — radial distance per grid point
    bess  :: Array{Float64, 4}     # (n_quad, N, N, Ne) — J_0(ks[q] · rgrid[i,j,e])
    ks    :: Vector{Float64}       # (n_quad,) — quadrature nodes on [0, 8/σ]
    ws    :: Vector{Float64}       # (n_quad,) — quadrature weights
    env   :: Vector{Float64}       # (n_quad,) — exp(-(k σ)²/2)
    σ     :: Float64
end

"""
    outgoing_pulse_2d_cache(coords::AbstractArray{<:Real, 4};
                            σ, center = (0, 0),
                            n_quad = 128) → GaussianPulse2dCache

Build the Bessel-function cache for the analytic 2D Gaussian-pulse
solution. `coords` is the `(2, N, N, Ne)` array from
`make_geometry(::Mesh{2})`.

`n_quad`-point Gauss–Legendre on `[0, 8/σ]` is accurate to ~14
digits for `t ≲ 10 σ`. Bump `n_quad` for longer integration windows
or sharper Gaussians.
"""
function outgoing_pulse_2d_cache(coords::AbstractArray{<:Real, 4};
                                  σ::Real,
                                  center = (zero(σ), zero(σ)),
                                  n_quad::Int = 128)
    σ64 = _to_f64(σ)
    cx  = _to_f64(center[1])
    cy  = _to_f64(center[2])
    σ64 > 0          || error("outgoing_pulse_2d_cache: σ must be positive (got $σ)")
    n_quad > 0       || error("outgoing_pulse_2d_cache: n_quad must be positive (got $n_quad)")
    size(coords, 1) == 2 ||
        error("outgoing_pulse_2d_cache: coords must be (2, N, N, Ne); got $(size(coords))")

    nodes_unit, weights_unit = gausslegendre(n_quad)
    kmax = 8.0 / σ64
    half = kmax / 2
    ks   = half .* (nodes_unit .+ 1.0)
    ws   = weights_unit .* half
    env  = exp.(-(ks .* σ64) .^ 2 ./ 2)

    _, N, _, Ne = size(coords)
    rgrid = Array{Float64, 3}(undef, N, N, Ne)
    bess  = Array{Float64, 4}(undef, n_quad, N, N, Ne)
    @inbounds for e in 1:Ne, j in 1:N, i in 1:N
        x = _to_f64(coords[1, i, j, e])
        y = _to_f64(coords[2, i, j, e])
        r = sqrt((x - cx)^2 + (y - cy)^2)
        rgrid[i, j, e] = r
        for q in 1:n_quad
            bess[q, i, j, e] = besselj(0, ks[q] * r)
        end
    end
    return GaussianPulse2dCache(rgrid, bess, ks, ws, env, σ64)
end

"""
    outgoing_pulse_2d!(u, u̇, cache::GaussianPulse2dCache, t; A)
    outgoing_pulse_2d!(u, u̇, coords, t;
                       A, σ, center = (0, 0), n_quad = 128)

Analytic 2D wave-equation solution at time `t` with initial data
`u(r, 0) = A · exp(-r² / (2σ²))` and `u_t(r, 0) = 0`. Fills `u`,
`u̇` in-place. See the block comment at the top of this section for
the integral form and the precision / performance caveats.

The first method (cache-based) is what you want inside an evolve
loop — it avoids re-computing the per-grid-point Bessel-function
table at every call. The second method is a convenience wrapper
that builds and discards the cache; use it for one-shot evaluation.

# Physics caveat

The 2D wave equation isn't Huygens. This solution starts as a
localized Gaussian bump and spreads outward, but doesn't entirely
clear the centre as `t` grows — energy lingers in a long wake.
That's a feature of the 2D equation, not a numerical artefact.
"""
function outgoing_pulse_2d!(u::AbstractArray{T, 3}, u̇::AbstractArray{T, 3},
                             cache::GaussianPulse2dCache, t::Real;
                             A) where {T}
    size(u) == size(u̇) == size(cache.rgrid) ||
        error("outgoing_pulse_2d!: u / u̇ / cache shape mismatch — got $(size(u)), $(size(u̇)), $(size(cache.rgrid))")
    backend = get_backend(u)
    if backend isa CPU
        _outgoing_pulse_2d_eval!(u, u̇, cache, t, A)
    else
        # The cache holds host-resident `Float64` arrays (the integrand
        # uses `besselj` / `cos` / `exp` which aren't defined on GPUs),
        # so the per-grid-point inner loop has to run on the CPU. For
        # GPU-resident `u`, `u̇` we evaluate into host scratch then
        # `copyto!` the result back to the device. Cost: one extra
        # host→device copy per call, dominated by the integral
        # evaluation itself.
        u_host  = Array{T, 3}(undef, size(u))
        u̇_host  = Array{T, 3}(undef, size(u̇))
        _outgoing_pulse_2d_eval!(u_host, u̇_host, cache, t, A)
        copyto!(u,  u_host)
        copyto!(u̇, u̇_host)
    end
    return u, u̇
end

# Inner loop — operates on plain `Array` only (the cache fields are
# `Array{Float64}` and the writes here are scalar). Called by the
# public method either directly (CPU) or via host scratch (GPU).
@inline function _outgoing_pulse_2d_eval!(u::AbstractArray{T, 3},
                                           u̇::AbstractArray{T, 3},
                                           cache::GaussianPulse2dCache,
                                           t::Real, A) where {T}
    σ64 = cache.σ
    A64 = _to_f64(A)
    t64 = _to_f64(t)
    n_quad = length(cache.ks)
    ct = cos.(cache.ks .* t64)
    st = sin.(cache.ks .* t64)
    pref = A64 * σ64^2

    N  = size(u, 1)
    Ne = size(u, 3)
    @inbounds for e in 1:Ne, j in 1:N, i in 1:N
        u_sum  = 0.0
        ud_sum = 0.0
        for q in 1:n_quad
            base = cache.ws[q] * cache.ks[q] * cache.env[q] * cache.bess[q, i, j, e]
            u_sum  += base * ct[q]
            ud_sum -= base * cache.ks[q] * st[q]
        end
        u[i, j, e]  = T(pref * u_sum)
        u̇[i, j, e]  = T(pref * ud_sum)
    end
    return nothing
end

function outgoing_pulse_2d!(u::AbstractArray{T, 3}, u̇::AbstractArray{T, 3},
                             coords::AbstractArray{<:Real, 4}, t::Real;
                             A, σ, center = (zero(T), zero(T)),
                             n_quad::Int = 128) where {T}
    cache = outgoing_pulse_2d_cache(coords; σ, center, n_quad)
    return outgoing_pulse_2d!(u, u̇, cache, t; A)
end

################################################################################
# Sommerfeld absorbing BC (2D)
#
# Mirror of the 3D Sommerfeld pass in `wave.jl`, adapted to 4 face
# axes and the 2D 90°-rotation outward normal:
#
#   n_u = sgn_out · (t_y, −t_x),  sgn_out = sgn_f · sgn_c · handedness[e]
#
# where `t = J[:, axis_p]` is the in-face tangent column of the
# Jacobian and `axis_p = 3 − a_idx` is the tangent reference axis for
# face axis `a_idx`. See `_face_sat_compute_2d!` in
# `HexSBPSAT/src/kernels2d.jl` for the same normal derivation.
#
# At every Sommerfeld face quadrature node the kernel adds the
# dissipative drag
#
#   ü += −wF · (u̇ + ∂_n u + u / R) / H_phys
#
# where `R = sommerfeld_R`. With `R = Inf` (default) this collapses to
# the plane-wave BGT-0 BC `u̇ + ∂_n u = 0`, which leaves an `O(1/R)`
# reflection coefficient on a curved outer boundary in 2D. The 3D-style
# BGT-1 `+ u/R` spherical correction is **not** exact in 2D (the 2D
# wave equation isn't Huygens; outgoing radial waves decay as `1/√r`,
# not `1/r`), so passing a finite `sommerfeld_R` here is a best-effort
# approximation only — useful for absorbing a wide-aperture pulse on a
# circular boundary, but not a substitute for the proper 2D Bayliss–
# Turkel correction.

@inline function _sommerfeld_face_2d!(::Val{f}, i, j, e,
                                       ü::AbstractArray{T, 3},
                                       u̇::AbstractArray{T, 3},
                                       geom::MeshGeometry{2, T, N},
                                       work::MeshWorkspace{2, T, N},
                                       H_1d::SVector{N, T},
                                       sommerfeld_tag::Int8,
                                       inv_R::T,
                                       ::Val{N}) where {f, T, N}
    a_idx  = _face_axis_idx_2d(Val(f))
    face_r = _face_row_2d(Val(f), Val(N))
    sgn_f  = _face_sign_2d(Val(f), T)
    sgn_c  = _cross_sign_2d(Val(a_idx), T)
    axis_p = _tangent_axis_2d(Val(a_idx))

    ia = a_idx == 1 ? i : j
    ia == face_r || return nothing

    @inbounds nbr = geom.conn.neighbour[f, e]
    nbr == 0 || return nothing
    @inbounds tag = geom.conn.bdry[f, e]
    tag == sommerfeld_tag || return nothing

    p_local = a_idx == 1 ? j : i

    @inbounds J11 = geom.jac[1,1,i,j,e]; @inbounds J12 = geom.jac[1,2,i,j,e]
    @inbounds J21 = geom.jac[2,1,i,j,e]; @inbounds J22 = geom.jac[2,2,i,j,e]

    tp_x = axis_p == 1 ? J11 : J12
    tp_y = axis_p == 1 ? J21 : J22

    @inbounds sgn_out = sgn_f * sgn_c * T(geom.handedness[e])

    # 90° rotation of the tangent → unnormalised outward physical normal.
    nx_u =  sgn_out * tp_y
    ny_u = -sgn_out * tp_x
    JF   = sqrt(nx_u * nx_u + ny_u * ny_u)
    nx   = nx_u / JF
    ny   = ny_u / JF

    @inbounds wF = JF * H_1d[p_local]

    # Trace populated by `apply_laplacian!`'s pass 1 — `(u, ∂xu, ∂yu)`
    # at every face quadrature node.
    @inbounds u_self = work.face_trace[1, p_local, f, e]
    @inbounds gx     = work.face_trace[2, p_local, f, e]
    @inbounds gy     = work.face_trace[3, p_local, f, e]
    Gn = nx * gx + ny * gy

    @inbounds u̇_self = u̇[i, j, e]
    @inbounds Hp     = geom.Hphys[i, j, e]

    # BGT-0 (`inv_R = 0`) or best-effort BGT-1-shaped (`inv_R = 1/R`)
    # dissipative drag. See the block comment above for the caveat.
    @inbounds ü[i, j, e] -= wF * (u̇_self + Gn + inv_R * u_self) / Hp
    return nothing
end

@kernel function _sommerfeld_pass_kernel_2d!(ü::AbstractArray{T},
                                              @Const(u̇::AbstractArray{T}),
                                              geom, work, H_1d,
                                              sommerfeld_tag::Int8,
                                              inv_R::T,
                                              ::Val{N}) where {T, N}
    e    = @index(Group, Linear)
    li   = @index(Local, Linear)
    i, j = _ij_from_li(li, Val(N))

    # Each face contributes to its own outgoing-flux drag. A workitem
    # on a corner (both `i` and `j` extremal) accumulates one drag per
    # face it lies on — matching the continuum surface integral.
    _sommerfeld_face_2d!(Val(1), i, j, e, ü, u̇, geom, work, H_1d, sommerfeld_tag, inv_R, Val(N))
    _sommerfeld_face_2d!(Val(2), i, j, e, ü, u̇, geom, work, H_1d, sommerfeld_tag, inv_R, Val(N))
    _sommerfeld_face_2d!(Val(3), i, j, e, ü, u̇, geom, work, H_1d, sommerfeld_tag, inv_R, Val(N))
    _sommerfeld_face_2d!(Val(4), i, j, e, ü, u̇, geom, work, H_1d, sommerfeld_tag, inv_R, Val(N))
end

################################################################################
# Wave-equation RHS (2D)

"""
    rhs_wave2d!(ü, u, u̇, params; geom, ops, work)
    rhs_wave2d!(ü, u, u̇, bdry_values; geom, ops, work, τ, sommerfeld_R = T(Inf))

2D wave-equation RHS: `ü = L_h u` plus an optional Sommerfeld
dissipative SAT at outer faces tagged `7`. Internally calls
`HexSBPSAT.apply_laplacian!(ü, u, bdry_values; geom, ops, work, τ)`
(which sees Sommerfeld faces as "free" and contributes nothing on
them), then — for finite `sommerfeld_R` *or* if the mesh has any
tag-7 outer face — a second kernel adds the BGT-0 (plane-wave)
dissipative drag

```
Δü = −wF · (u̇ + ∂_n u + u/R) / H_phys
```

at every Sommerfeld face quadrature node. State arrays `ü`, `u`, `u̇`
are 3-D of shape `(N, N, Ne)`.

# Caveat — physics

`R = Inf` (default) gives the plane-wave BGT-0 BC `u̇ + ∂_n u = 0`.
Finite `R` adds the `+ u/R` term, which is the 3D Bayliss–Turkel
spherical correction; it is **not** exact in 2D (the 2D wave
equation isn't Huygens, so outgoing radial waves decay as `1/√r`).
For demos on a circular boundary, BGT-0 is the safe choice.

# Boundary tag convention

* `0` — interior or unset.
* `1..4` — Dirichlet against `bdry_values[tag]` (1=−x, 2=+x, 3=−y, 4=+y).
* `7` — Sommerfeld radiative (this method's drag).
"""
function rhs_wave2d!(ü::AbstractArray{T,3}, u::AbstractArray{T,3}, u̇::AbstractArray{T,3},
                     params::Params2d{T};
                     geom::MeshGeometry{2, T, N},
                     ops::SBPOps{N, T},
                     work::MeshWorkspace{2, T, N}) where {N, T}
    return rhs_wave2d!(ü, u, u̇, params.bdry_values;
                       geom, ops, work, τ = params.τ,
                       sommerfeld_R = params.sommerfeld_R)
end

function rhs_wave2d!(ü::AbstractArray{T,3}, u::AbstractArray{T,3}, u̇::AbstractArray{T,3},
                     bdry_values::NTuple{4, T};
                     geom::MeshGeometry{2, T, N},
                     ops::SBPOps{N, T},
                     work::MeshWorkspace{2, T, N}, τ,
                     sommerfeld_R = T(Inf)) where {N, T}
    @assert size(ü) == size(u̇) == size(u)
    @assert size(u, 1) == size(u, 2) == N
    @assert size(u, 3) == geom.Ne

    # Pass 1: spatial Laplacian + interior SIPG + outer Dirichlet SAT.
    # As a side effect, `work.face_trace` is populated with
    # `(u, ∂xu, ∂yu)` at every face quadrature node, which the
    # Sommerfeld pass reads.
    apply_laplacian!(ü, u, bdry_values; geom, ops, work, τ)

    # Pass 2: dissipative drag on outer faces tagged `7`. On all other
    # faces each workitem's `_sommerfeld_face_2d!` returns immediately.
    # `1/Inf = 0` cleanly turns off the `+u/R` term without a runtime
    # branch.
    H_1d  = SVector{N, T}(ntuple(i -> ops.H[i, i], Val(N)))
    inv_R = one(T) / T(sommerfeld_R)
    backend = get_backend(u)
    _sommerfeld_pass_kernel_2d!(backend, N^2)(
        ü, u̇, geom, work, H_1d, SOMMERFELD_BDRY_TAG, inv_R, Val(N);
        ndrange = N^2 * geom.Ne)

    return ü
end

################################################################################
# Timestep limit (2D)

"""
    recommended_dt(geom::MeshGeometry{2, T, N}, ops, τ; cfl_safety = 0.9) → T

2D analog of the 3D [`recommended_dt`](@ref) — Störmer–Verlet stability
limit `cfl_safety · 2 / sqrt(|λ_max|)`, with `|λ_max|` estimated by
`HexSBPSAT.spectral_radius_estimate` on the 2D discrete Laplacian.
"""
function recommended_dt(geom::MeshGeometry{2, T, N}, ops::SBPOps{N, T}, τ;
                         cfl_safety = T(0.9)) where {N, T}
    λ = spectral_radius_estimate(geom, ops, τ)
    λ == 0 && return T(Inf)
    return cfl_safety * T(2) / sqrt(λ)
end
