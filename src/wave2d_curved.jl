# 2D scalar wave on a 2+1 ADM background with space- and time-varying
# lapse α(t,x,y), shift βⁱ(t,x,y), and spatial metric γ_ij(t,x,y).
# State (Φ, Π), Π := (√γ/α)(∂_tΦ − βⁱ∂_iΦ). The covariant wave
# equation ∂_μ(√|g| g^{μν} ∂_νΦ) = 0 (√|g| = α√γ) gives the
# flux-conservative first-order system
#
#     ∂_t Φ = βⁱ ∂_i Φ + (α/√γ) Π
#     ∂_t Π = ∂_i ( βⁱ Π + α√γ γ^{ij} ∂_j Φ ),       i, j ∈ {x, y}
#
# (the 1D case is the i=j=x restriction with γ^{xx}=1/γ_xx). The flux
# Fⁱ = βⁱΠ + α√γ γ^{ij}∂_jΦ is differentiated with the per-axis
# `HexSBPSAT.apply_D!(·, d)` (reference SBP-G + centred-flux SAT), the
# same exactly-skew operator used for the gradient ∂_jΦ — so on
# axis-aligned affine meshes (uniform_quad) the assembled RHS is skew
# up to background variation. Kreiss-Oliger dissipation ε·μ⁻⁵·D⁶ is
# applied per axis (see wave1d.jl for the μ⁻⁵ normalisation).
#
# CAPITAL Greek Φ, Π (lowercase π is the constant).

using HexSBPSAT: MeshGeometry, SBPOps, MeshWorkspace, make_workspace, apply_D!,
                 make_metric_terms2d, apply_gradient2d!, apply_divergence2d!
using KernelAbstractions: @kernel, @index, @Const, get_backend, allocate
using SpacetimeMetrics: adm_decompose
using StaticArrays: SVector, SMatrix

################################################################################
# ADM backgrounds (2D)

"""
    Background2D

Abstract 2+1 ADM background (α, βⁱ, γ_ij) for the 2D scalar wave.
Concrete subtypes implement
`_bg_adm2(bg, t, x, y) → (α, β1, β2, γ11, γ12, γ22)` (covariant
spatial metric); sample onto the grid with [`sample_background2d!`].
"""
abstract type Background2D end

"""
    AnalyticBackground2D(α_fn, β_fn, γ_fn) <: Background2D

Closures `α_fn(t,x,y)→α`, `β_fn(t,x,y)→(β1,β2)`,
`γ_fn(t,x,y)→(γ11,γ12,γ22)`. For backgrounds not expressible as a
4-metric (superluminal shift) or number types without reliable
transcendental functions. Closures must not capture `Type` objects if
the background is to be passed into GPU kernels.
"""
struct AnalyticBackground2D{Fα, Fβ, Fγ} <: Background2D
    α_fn :: Fα
    β_fn :: Fβ
    γ_fn :: Fγ
end

@inline function _bg_adm2(bg::AnalyticBackground2D, t, x, y)
    γ11, γ12, γ22 = bg.γ_fn(t, x, y)
    β1, β2 = bg.β_fn(t, x, y)
    return bg.α_fn(t, x, y), β1, β2, γ11, γ12, γ22
end

"""
    MetricBackground2D(m) <: Background2D

Background from a `SpacetimeMetrics.AbstractMetric`; ADM variables via
`adm_decompose(m, (t,x,y,0))`, using the (x,y) sub-block of the
spatial 3-metric and the in-plane shift components.
"""
struct MetricBackground2D{M} <: Background2D
    metric :: M
end

@inline function _bg_adm2(bg::MetricBackground2D, t, x, y)
    α, β, γ = adm_decompose(bg.metric, SVector(t, x, y, zero(x)))
    return α, β[1], β[2], γ[1, 1], γ[1, 2], γ[2, 2]
end

# Coefficient fields the RHS consumes, per node: lapse, √γ, shift
# components, and the contravariant spatial metric γ^{ij}.
@kernel function _sample_bg2d_kernel!(alpha, sqrtγ, b1, b2,
                                      gu11, gu12, gu22, bg, t::T,
                                      @Const(xg), @Const(yg)) where {T}
    idx = @index(Global, Linear)
    @inbounds begin
        αv, β1, β2, γ11, γ12, γ22 = _bg_adm2(bg, t, xg[idx], yg[idx])
        det = T(γ11) * T(γ22) - T(γ12)^2
        alpha[idx] = T(αv)
        sqrtγ[idx] = sqrt(det)
        b1[idx]    = T(β1)
        b2[idx]    = T(β2)
        gu11[idx]  =  T(γ22) / det
        gu12[idx]  = -T(γ12) / det
        gu22[idx]  =  T(γ11) / det
    end
end

"""
    sample_background2d!(coef, bg::Background2D, t, xg, yg)

Fill the coefficient NamedTuple `coef = (; alpha, sqrtγ, b1, b2,
gu11, gu12, gu22)` (each `(N,N,Ne)`) from `bg` at time `t` on the
collocation grids `xg, yg = geom.coords[1,…], geom.coords[2,…]`. A
KernelAbstractions kernel — allocation-free, GPU-compatible.
"""
function sample_background2d!(coef, bg::Background2D, t, xg, yg)
    backend = get_backend(coef.alpha)
    T = eltype(coef.alpha)
    _sample_bg2d_kernel!(backend)(coef.alpha, coef.sqrtγ, coef.b1, coef.b2,
                                  coef.gu11, coef.gu12, coef.gu22,
                                  bg, T(t), xg, yg; ndrange = length(coef.alpha))
    return nothing
end

# Allocate the coefficient bundle on the same backend as `geom`.
function make_coef2d(geom::MeshGeometry{2, T, N}) where {T, N}
    backend = get_backend(geom.coords)
    mk() = allocate(backend, T, N, N, geom.Ne)
    return (; alpha = mk(), sqrtγ = mk(), b1 = mk(), b2 = mk(),
            gu11 = mk(), gu12 = mk(), gu22 = mk())
end

################################################################################
# RHS workspace + kernel

"""
    Wave2DWorkspace{T}

Scratch for [`wave2d_curved_rhs!`](@ref): six `(N,N,Ne)` buffers
(`DΦ1, DΦ2, F1, F2, s1, s2`) plus the first-derivative spectral radius
`μ` and the Kreiss-Oliger scale `μ⁻⁵`.
"""
struct Wave2DWorkspace{T, AT <: AbstractArray{T,3}, MW}
    DΦ1 :: AT
    DΦ2 :: AT
    F1  :: AT
    F2  :: AT
    s1  :: AT
    s2  :: AT
    mw  :: MW          # HexSBPSAT.MeshWorkspace{2,T,N}: operator face trace
    μ      :: T
    inv_μ5 :: T
end

"""
    make_wave2d_workspace(geom::MeshGeometry{2, T, N}, ops) → Wave2DWorkspace

Allocate scratch and compute the spectral radius `μ` of the per-axis
`apply_D!` by power iteration on `D²` along axis 1 (same magnitude on
both axes for axis-aligned affine meshes).
"""
function make_wave2d_workspace(geom::MeshGeometry{2, T, N}, ops) where {T, N}
    backend = get_backend(geom.coords)
    Ne = geom.Ne
    bufs = ntuple(_ -> allocate(backend, T, N, N, Ne), 6)
    mw = make_workspace(geom)        # operator face-trace buffer
    v, w = bufs[5], bufs[6]
    v_host = T[isodd(i + j + m) ? one(T) : -one(T) for i in 1:N, j in 1:N, m in 1:Ne]
    copyto!(v, v_host)
    μ² = zero(T)
    for _ in 1:30
        apply_D!(w, v, 1; geom, ops, work = mw)
        apply_D!(v, w, 1; geom, ops, work = mw)
        μ² = sqrt(sum(abs2, v))
        μ² > 0 || break
        v ./= μ²
    end
    μ = sqrt(μ²)
    inv_μ5 = μ > 0 ? inv(μ)^5 : zero(T)
    return Wave2DWorkspace{T, typeof(v), typeof(mw)}(bufs..., mw, μ, inv_μ5)
end

# Six-fold per-axis KO application: out += koT · D_d⁶ field, ping-pong
# through the two scratch buffers. `field` is not modified.
@inline function _ko_axis!(dst, field, d, koT, ws, geom, ops)
    s1, s2 = ws.s1, ws.s2; mw = ws.mw
    apply_D!(s1, field, d; geom, ops, work = mw)
    apply_D!(s2, s1, d; geom, ops, work = mw)
    apply_D!(s1, s2, d; geom, ops, work = mw)
    apply_D!(s2, s1, d; geom, ops, work = mw)
    apply_D!(s1, s2, d; geom, ops, work = mw)
    apply_D!(s2, s1, d; geom, ops, work = mw)   # D⁶ field in s2
    @. dst += koT * s2
    return nothing
end

"""
    wave2d_curved_rhs!(Φ̇, Π̇, Φ, Π, coef; geom, ops, ws, ε_KO = 0.1,
                       bc2d = nothing) → (Φ̇, Π̇)

2D scalar-wave RHS on a 2+1 ADM background. `coef` is the coefficient
bundle from [`sample_background2d!`] at the current stage time. Flux
`Fⁱ = βⁱΠ + α√γ γ^{ij}∂_jΦ`, evolved as
`∂_tΦ = βⁱ∂_iΦ + (α/√γ)Π`, `∂_tΠ = ∂_iFⁱ`. Allocation-free.

`metric === nothing` (default) selects the axis-aligned affine path
(per-axis `apply_D!`, for `make_uniform_quad`). Passing the discrete
metric terms from `make_metric_terms2d` selects the curvilinear path
(free-stream-preserving `apply_gradient2d!`/`apply_divergence2d!`, for
cubed-square etc.). KO dissipation stays per-axis in both cases.
"""
function wave2d_curved_rhs!(Φ̇::AbstractArray{T,3}, Π̇::AbstractArray{T,3},
                            Φ::AbstractArray{T,3}, Π::AbstractArray{T,3},
                            coef;
                            geom::MeshGeometry{2, T, N},
                            ops::SBPOps{N, T},
                            ws::Wave2DWorkspace{T},
                            ε_KO::Real = 0.1,
                            bc2d = nothing,
                            metric = nothing) where {N, T}
    (; DΦ1, DΦ2, F1, F2, s1, s2) = ws
    (; alpha, sqrtγ, b1, b2, gu11, gu12, gu22) = coef
    εT = T(ε_KO); koT = εT * ws.inv_μ5

    mw = ws.mw
    if metric === nothing
        apply_D!(DΦ1, Φ, 1; geom, ops, work = mw)   # affine per-axis gradient
        apply_D!(DΦ2, Φ, 2; geom, ops, work = mw)
    else
        apply_gradient2d!(DΦ1, DΦ2, Φ; geom, ops, metric, work = mw)   # curvilinear
    end

    # Flux Fⁱ = βⁱΠ + α√γ γ^{ij}∂_jΦ.
    @. F1 = b1 * Π + alpha * sqrtγ * (gu11 * DΦ1 + gu12 * DΦ2)
    @. F2 = b2 * Π + alpha * sqrtγ * (gu12 * DΦ1 + gu22 * DΦ2)

    if metric === nothing
        apply_D!(s1, F1, 1; geom, ops, work = mw)
        apply_D!(s2, F2, 2; geom, ops, work = mw)
        @. Π̇ = s1 + s2
    else
        apply_divergence2d!(Π̇, F1, F2; geom, ops, metric, work = mw)
    end
    @. Φ̇ = b1 * DΦ1 + b2 * DΦ2 + (alpha / sqrtγ) * Π

    if εT != 0
        _ko_axis!(Φ̇, Φ, 1, koT, ws, geom, ops)
        _ko_axis!(Φ̇, Φ, 2, koT, ws, geom, ops)
        _ko_axis!(Π̇, Π, 1, koT, ws, geom, ops)
        _ko_axis!(Π̇, Π, 2, koT, ws, geom, ops)
    end

    if bc2d !== nothing
        if metric === nothing
            _apply_bc2d!(Φ̇, Π̇, Φ, Π, coef, ws; geom, ops, bc2d)
        else
            _apply_bc2d_curv!(Φ̇, Π̇, Φ, Π, coef, ws, metric; geom, ops, bc2d)
        end
    end
    return Φ̇, Π̇
end

"""
    wave2d_energy(Φ, Π, coef; geom, ops, ws, metric = nothing) → E

Discrete ADM energy `E = Σ ½[ Π²/√γ + √γ γ^{ij}∂_iΦ∂_jΦ ]·H`. On the
affine path the mass `H` is `geom.Hphys`; on the curvilinear path
(`metric` provided) it is the consistent discrete mass `metric.Hd` and
the gradient is the free-stream-preserving `apply_gradient2d!`.
This is the PHYSICAL ADM energy (gradient weight `1/√γ`); it is exactly
conserved only for static backgrounds with `α ≡ 1` (the skew operator's
own conserved norm weights the gradient by `∝ α/√γ`). Otherwise use it
as a drift/decay monitor. Overwrites `ws.DΦ1, ws.DΦ2`.
"""
function wave2d_energy(Φ::AbstractArray{T,3}, Π::AbstractArray{T,3}, coef;
                       geom::MeshGeometry{2, T, N}, ops::SBPOps{N, T},
                       ws::Wave2DWorkspace{T}, metric = nothing) where {N, T}
    (; DΦ1, DΦ2) = ws
    (; sqrtγ, gu11, gu12, gu22) = coef
    mw = ws.mw
    if metric === nothing
        apply_D!(DΦ1, Φ, 1; geom, ops, work = mw)
        apply_D!(DΦ2, Φ, 2; geom, ops, work = mw)
        H = geom.Hphys
    else
        apply_gradient2d!(DΦ1, DΦ2, Φ; geom, ops, metric, work = mw)
        H = metric.Hd
    end
    return sum(@. (Π^2 / sqrtγ +
                   sqrtγ * (gu11 * DΦ1^2 + 2 * gu12 * DΦ1 * DΦ2 +
                            gu22 * DΦ2^2)) * H / 2)
end
